//! Bounded chunks carry compact BF16 rank partials in token order.
use super::{
    V41BackboneRequest, V41Tp4Planes, EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16, V41_HIDDEN,
    V41_PARTIAL_ROW_BYTES,
};
use crate::{
    ExpertProtocolV2ResponseHeader, ExpertProtocolV2ResponseRef, ExpertProtocolV2ResponseView,
    ExpertProtocolV2Status, ExpertV2Dtype, EXPERT_PROTOCOL_V2_FLAG_DEBUG_CHECKSUM,
    EXPERT_PROTOCOL_V2_FLAG_RESPONSE_MORE_CHUNKS, EXPERT_PROTOCOL_V2_FLAG_RESPONSE_ROW_INDICES,
    EXPERT_PROTOCOL_V2_RESPONSE_DEBUG_HEADER_LEN, EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN,
};
use anyhow::{ensure, Context, Result};

impl V41BackboneRequest<'_> {
    fn response_header_bytes(&self) -> usize {
        if self.view.header.flags & EXPERT_PROTOCOL_V2_FLAG_DEBUG_CHECKSUM != 0 {
            EXPERT_PROTOCOL_V2_RESPONSE_DEBUG_HEADER_LEN
        } else {
            EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN
        }
    }
    /// Maximum token rows per indexed frame, including header and row-index overhead.
    pub fn response_chunk_rows(&self, max_frame_bytes: usize) -> Result<u32> {
        let payload = max_frame_bytes
            .checked_sub(self.response_header_bytes())
            .context("response frame cannot fit header")?;
        let rows = payload / (V41_PARTIAL_ROW_BYTES as usize + 4);
        ensure!(rows > 0, "response frame cannot fit one native token row");
        Ok(rows.min(self.rows() as usize) as u32)
    }
    /// Produce one contiguous chunk from a slice of the full route plane.
    /// `partials` contains only this chunk; caller reuses the supplied index scratch.
    pub fn response_chunk<'a>(
        &self,
        executor_id: u64,
        start_row: u32,
        partials: &'a [u8],
        row_indices: &'a mut [u32],
        max_frame_bytes: usize,
    ) -> Result<ExpertProtocolV2ResponseRef<'a>> {
        ensure!(
            executor_id != 0,
            "native response needs an executor identity"
        );
        ensure!(
            partials.len() % V41_PARTIAL_ROW_BYTES as usize == 0,
            "native response chunk has a partial token row"
        );
        let rows = u32::try_from(partials.len() / V41_PARTIAL_ROW_BYTES as usize)?;
        ensure!(
            rows > 0 && rows <= self.response_chunk_rows(max_frame_bytes)?,
            "native response chunk exceeds frame budget"
        );
        let end = start_row
            .checked_add(rows)
            .context("native response row overflow")?;
        ensure!(
            end <= self.rows(),
            "native response chunk exceeds request rows"
        );
        ensure!(
            row_indices.len() >= rows as usize,
            "response row-index scratch is too short"
        );
        for (index, row) in row_indices[..rows as usize].iter_mut().enumerate() {
            *row = start_row + index as u32;
        }
        let request = &self.view.header;
        let response = ExpertProtocolV2ResponseRef {
            header: ExpertProtocolV2ResponseHeader {
                request_id: request.request_id,
                placement_version: request.placement_version,
                layer_id: request.layer_id,
                row_count: rows,
                output_dim: V41_HIDDEN,
                output_dtype: ExpertV2Dtype::Bf16,
                output_row_stride_bytes: V41_PARTIAL_ROW_BYTES,
                output_payload_bytes: partials.len() as u64,
                status: ExpertProtocolV2Status::Ok,
                flags: (request.flags & !super::V41_EXL3_PAIRED_REQUEST_FLAG)
                    | EXPERT_PROTOCOL_V2_FLAG_RESPONSE_ROW_INDICES
                    | if end < self.rows() {
                        EXPERT_PROTOCOL_V2_FLAG_RESPONSE_MORE_CHUNKS
                    } else {
                        0
                    },
                executor_id,
            },
            row_indices: Some(&row_indices[..rows as usize]),
            partial_output_payload: partials,
        };
        response.validate()?;
        ensure!(
            response.wire_stats().wire_bytes <= max_frame_bytes,
            "native response frame budget mismatch"
        );
        Ok(response)
    }
}

/// Constant-size assembly state; a sink copies each validated chunk into its rank plane.
/// Frames from different ranks may interleave; each rank's rows must arrive in order.
/// A rank is complete only after every row and the matching final-chunk marker.
/// As with complete responses, request IDs must identify unique in-flight waves.
pub struct V41Tp4ChunkReceiver {
    identity: V41Tp4Planes<'static>,
    received: [u32; 4],
    finished: [bool; 4],
    max_frame_bytes: usize,
}
impl V41Tp4ChunkReceiver {
    pub fn new(
        request: &V41BackboneRequest<'_>,
        executors: [u64; 4],
        max_frame_bytes: usize,
    ) -> Result<Self> {
        request.response_chunk_rows(max_frame_bytes)?;
        Ok(Self {
            identity: V41Tp4Planes::new(request, executors)?,
            received: [0; 4],
            finished: [false; 4],
            max_frame_bytes,
        })
    }
    /// Collect exactly two complete rank planes while preserving TP4 APIs.
    pub fn new_tp2(
        request: &V41BackboneRequest<'_>,
        executors: [u64; 2],
        max_frame_bytes: usize,
    ) -> Result<Self> {
        request.response_chunk_rows(max_frame_bytes)?;
        ensure!(!request.is_paired(), "paired EXL3 requires four ranks");
        Ok(Self {
            identity: V41Tp4Planes::from_header_ranks(&request.view.header, &executors)?,
            received: [0; 4],
            finished: [false; 4],
            max_frame_bytes,
        })
    }
    /// Validate the owned request without serializing its activation payload.
    pub(crate) fn from_owned(
        request: &crate::ExpertProtocolV2Request,
        max_rows: u32,
        executors: [u64; 4],
        max_frame_bytes: usize,
    ) -> Result<Self> {
        Self::from_owned_ranks(request, max_rows, &executors, max_frame_bytes)
    }
    pub(crate) fn from_owned_ranks(
        request: &crate::ExpertProtocolV2Request,
        max_rows: u32,
        executors: &[u64],
        max_frame_bytes: usize,
    ) -> Result<Self> {
        if request.header.flags & super::V41_EXL3_PAIRED_REQUEST_FLAG != 0 {
            ensure!(executors.len() == 4, "paired EXL3 requires four ranks");
            V41BackboneRequest::validate_owned_paired(request, max_rows)?;
        } else {
            V41BackboneRequest::validate_owned(request, max_rows)?;
        }
        ensure!(request.wire_stats().wire_bytes <= max_frame_bytes,
            "native request exceeds RoCE frame budget");
        let header_bytes = if request.header.flags & EXPERT_PROTOCOL_V2_FLAG_DEBUG_CHECKSUM != 0 {
            EXPERT_PROTOCOL_V2_RESPONSE_DEBUG_HEADER_LEN
        } else {
            EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN
        };
        ensure!(max_frame_bytes >= header_bytes + V41_PARTIAL_ROW_BYTES as usize + 4,
            "response frame cannot fit one native token row");
        Ok(Self {
            identity: V41Tp4Planes::from_header_ranks(&request.header, executors)?,
            received: [0; 4],
            finished: [false; 4],
            max_frame_bytes,
        })
    }
    pub fn complete(&self) -> bool {
        self.finished[..self.identity.executors.len()].iter().all(|value| *value)
    }
    pub fn received_rows(&self) -> [u32; 4] {
        self.received
    }

    /// The verbs client has already checked framing and checksum before exposing
    /// these owned/recycled payload bytes. Keep their owner alive through the sink.
    pub(crate) fn push_rdma<'f, F>(
        &mut self,
        chunk: &'f crate::VerbsHostProtocolV2ResponseChunk,
        sink: F,
    ) -> Result<usize>
    where
        F: FnOnce(usize, u32, &'f [u8]) -> Result<()>,
    {
        ensure!(
            chunk.wire_bytes <= self.max_frame_bytes,
            "native response exceeds receive frame budget"
        );
        self.push_parts(
            &chunk.header,
            chunk.row_indices.as_deref(),
            chunk.partial_output_payload.as_ref(),
            sink,
        )
    }

    /// Validate before invoking the sink, then commit progress only if it succeeds.
    /// The sink receives (rank, first token row, contiguous BF16 rank-partial bytes).
    /// It must copy the bytes or keep their storage alive until asynchronous writes
    /// finish; `complete()` does not itself synchronize GPU copies.
    /// On sink error no progress is committed, but destination bytes may be partial.
    pub fn push<'f, F>(&mut self, frame: &'f [u8], sink: F) -> Result<usize>
    where
        F: FnOnce(usize, u32, &'f [u8]) -> Result<()>,
    {
        ensure!(
            frame.len() <= self.max_frame_bytes,
            "native response exceeds receive frame budget"
        );
        let response = ExpertProtocolV2ResponseView::parse(frame)?;
        let indices = if response.row_indexed() {
            Some(
                (0..response.header.row_count as usize)
                    .map(|i| response.request_row_index(i))
                    .collect::<Result<Vec<_>>>()?,
            )
        } else {
            None
        };
        self.push_parts(
            &response.header,
            indices.as_deref(),
            response.partial_output_payload(),
            sink,
        )
    }

    fn push_parts<'f, F>(
        &mut self,
        h: &ExpertProtocolV2ResponseHeader,
        indices: Option<&[u32]>,
        payload: &'f [u8],
        sink: F,
    ) -> Result<usize>
    where
        F: FnOnce(usize, u32, &'f [u8]) -> Result<()>,
    {
        let rank = self.identity.response_rank(h)?;
        ensure!(!self.finished[rank], "native TP rank already completed");
        let allowed = EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16
            | EXPERT_PROTOCOL_V2_FLAG_DEBUG_CHECKSUM
            | EXPERT_PROTOCOL_V2_FLAG_RESPONSE_ROW_INDICES
            | EXPERT_PROTOCOL_V2_FLAG_RESPONSE_MORE_CHUNKS;
        ensure!(
            h.flags & !allowed == 0,
            "unsupported native response chunk flags"
        );
        let start = self.received[rank];
        let end = start
            .checked_add(h.row_count)
            .context("native received row overflow")?;
        ensure!(
            h.row_count > 0 && end <= self.identity.rows,
            "native response chunk exceeds remaining rows"
        );
        ensure!(
            indices.is_some() == (h.flags & EXPERT_PROTOCOL_V2_FLAG_RESPONSE_ROW_INDICES != 0),
            "native row-index flag mismatch"
        );
        let more_chunks = h.flags & EXPERT_PROTOCOL_V2_FLAG_RESPONSE_MORE_CHUNKS != 0;
        ensure!(
            payload.len() == h.output_payload_bytes as usize,
            "native payload size mismatch"
        );
        if let Some(indices) = indices {
            ensure!(
                indices.len() == h.row_count as usize,
                "native row index count mismatch"
            );
            for i in 0..h.row_count {
                ensure!(
                    indices[i as usize] == start + i,
                    "native response rows overlap, skip or reorder"
                );
            }
        } else {
            ensure!(
                start == 0 && h.row_count == self.identity.rows && !more_chunks,
                "unindexed native response must contain the entire plane"
            );
        }
        ensure!(
            more_chunks == (end < self.identity.rows),
            "native response has an early or missing final marker"
        );
        sink(rank, start, payload)?;
        self.received[rank] = end;
        self.finished[rank] = end == self.identity.rows;
        Ok(rank)
    }
}
