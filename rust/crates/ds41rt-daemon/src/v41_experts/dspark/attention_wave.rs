//! One-stream draft attention with generation-checked committed cache reads.
use super::{DsparkAttentionOutput, DsparkProjection, DsparkWeights, ProjectionKind};
use crate::v41_dspark_cache::{DsparkWindow, WindowLease, WindowRead};
use crate::v41_memory::{DeviceAllocation, HostAllocation, LoadStream};
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary, V41AttentionOps, V41DsparkAttention};
use std::ffi::c_void;
pub(crate) struct DsparkAttentionWave<'weights, 'library> {
    stream: LoadStream<'library>,
    qa: DsparkProjection<'weights, 'library>,
    qb: DsparkProjection<'weights, 'library>,
    kv: DsparkProjection<'weights, 'library>,
    output: DsparkAttentionOutput<'weights, 'library>,
    ops: V41AttentionOps<'library>,
    attention: V41DsparkAttention<'library>,
    q_norm: Ds41rtDeviceBuffer,
    kv_norm: Ds41rtDeviceBuffer,
    sink: Ds41rtDeviceBuffer,
    draft: DeviceAllocation<'library>,
    positions: DeviceAllocation<'library>,
    descriptors: DeviceAllocation<'library>,
    staging: HostAllocation<'library>,
    requests: u32,
    width: usize,
    graph: Option<(*mut c_void, u32, u64)>,
    ready: Option<u32>,
}
impl<'library> DsparkWeights<'library> {
    pub fn attention_wave(
        &self,
        stage: usize,
        requests: u32,
        budget: usize,
    ) -> Result<DsparkAttentionWave<'_, 'library>> {
        ensure!(stage < 3, "invalid dSpark attention stage");
        let library = self.library;
        ensure!(
            DsparkAttentionWave::device_bytes_with_width(library, requests, self.draft_width)? <= budget,
            "dSpark attention wave exceeds budget"
        );
        let rows = requests * self.draft_width as u32;
        let capacity = DsparkAttentionWave::projection_capacity_with_width(requests, self.draft_width)?;
        let projection = |kind| {
            self.projection(
                kind,
                capacity,
                DsparkProjection::device_bytes(library, kind, capacity)?,
            )
        };
        Ok(DsparkAttentionWave {
            stream: LoadStream {
                library,
                raw: library.cuda_stream_create()?,
            },
            qa: projection(ProjectionKind::QueryA(stage))?,
            qb: projection(ProjectionKind::QueryB(stage))?,
            kv: projection(ProjectionKind::Kv(stage))?,
            output: self.attention_output(
                stage,
                capacity,
                DsparkAttentionOutput::device_bytes(library, capacity)?,
            )?,
            ops: library.v41_attention_ops_width(self.draft_width)?,
            attention: library.v41_dspark_attention_width(self.draft_width)?,
            q_norm: self.tensor(&format!("mtp.{stage}.attn.q_norm.weight"))?,
            kv_norm: self.tensor(&format!("mtp.{stage}.attn.kv_norm.weight"))?,
            sink: self.tensor(&format!("mtp.{stage}.attn.attn_sink"))?,
            draft: DeviceAllocation::new(library, rows as usize * 1024)?,
            positions: DeviceAllocation::new(library, rows as usize * 8)?,
            descriptors: DeviceAllocation::new(library, 128)?,
            staging: HostAllocation::new(library, 128 + rows as usize * 8)?,
            requests,
            width: self.draft_width,
            graph: None,
            ready: None,
        })
    }
}
impl DsparkAttentionWave<'_, '_> {
    pub(super) fn projection_capacity(requests: u32) -> Result<u32> {
        Self::projection_capacity_with_width(requests, 5)
    }
    pub(super) fn projection_capacity_with_width(requests: u32, width: usize) -> Result<u32> {
        ensure!(matches!(width, 5 | 7), "draft width must be five or seven");
        ensure!(
            (1..=16).contains(&requests),
            "invalid attention request capacity"
        );
        // Storage follows compiled buckets; launches use the actual live rows.
        let rows = requests as usize * width;
        Ok(if rows <= 16 { 16 } else if rows <= 40 { 40 } else if rows <= 80 { 80 } else { 256 })
    }
    pub fn additional_bytes(rows: u32) -> Result<usize> {
        ensure!((1..=4096).contains(&rows), "invalid attention wave rows");
        Ok(rows as usize * (1024 + 8) + 128)
    }
    pub fn device_bytes(library: &NativeLibrary, requests: u32) -> Result<usize> {
        Self::device_bytes_with_width(library, requests, 5)
    }
    pub fn device_bytes_with_width(library: &NativeLibrary, requests: u32, width: usize) -> Result<usize> {
        ensure!(
            (1..=16).contains(&requests),
            "invalid attention wave request capacity"
        );
        let rows = requests * width as u32;
        let capacity = Self::projection_capacity_with_width(requests, width)?;
        let mut bytes =
            Self::additional_bytes(rows)? + DsparkAttentionOutput::device_bytes(library, capacity)?;
        for kind in [
            ProjectionKind::QueryA(0),
            ProjectionKind::QueryB(0),
            ProjectionKind::Kv(0),
        ] {
            bytes = bytes
                .checked_add(DsparkProjection::device_bytes(library, kind, capacity)?)
                .context("attention wave storage overflow")?;
        }
        Ok(bytes)
    }
    pub fn input(&self) -> Ds41rtDeviceBuffer {
        self.qa.input()
    }
    fn synchronize(&self) -> Result<()> {
        unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) }
    }
    pub(super) fn prepare(
        &mut self,
        window: &DsparkWindow<'_>,
        requests: &[(WindowLease, u64)],
    ) -> Result<WindowRead> {
        self.ready = None;
        ensure!(
            !requests.is_empty() && requests.len() <= self.requests as usize,
            "attention wave exceeds request capacity"
        );
        window.attention_read_with_width(requests, self.width)
    }
    pub(super) fn upload(
        &mut self,
        read: &WindowRead,
        requests: &[(WindowLease, u64)],
    ) -> Result<()> {
        self.stage_upload(read, requests);
        let staging = self.staging.bytes_mut();
        self.stream.library.copy_h2d(self.descriptors.buffer, &staging[..128])?;
        self.stream.library.copy_h2d(self.positions.buffer, &staging[128..])
    }
    /// The containing chain retains staging and cache reads through completion.
    pub(super) unsafe fn upload_on(&mut self, read: &WindowRead,
        requests: &[(WindowLease, u64)], stream: *mut c_void) -> Result<()> {
        self.stage_upload(read, requests);
        let staging = self.staging.bytes_mut();
        unsafe {
            self.stream.library.copy_h2d_async(self.descriptors.buffer, &staging[..128], stream)?;
            self.stream.library.copy_h2d_async(self.positions.buffer, &staging[128..], stream)
        }
    }
    fn stage_upload(&mut self, read: &WindowRead, requests: &[(WindowLease, u64)]) {
        let bytes =
            unsafe { std::slice::from_raw_parts(read.descriptors.as_ptr().cast::<u8>(), 128) };
        let staging = self.staging.bytes_mut();
        staging[..128].copy_from_slice(bytes);
        staging[128..].fill(0);
        for (request, &(_, end)) in requests.iter().enumerate() {
            for draft in 0..self.width {
                let offset = 128 + (request * self.width + draft) * 8;
                // attention_read checked end+width before this upload.
                staging[offset..offset + 8].copy_from_slice(&(end + draft as u64).to_ne_bytes());
            }
        }
    }
    pub(super) fn output_storage(&self) -> Ds41rtDeviceBuffer {
        self.output.output_storage()
    }
    pub(super) unsafe fn enqueue_on(
        &mut self,
        read: &WindowRead,
        requests: u32,
        stream: *mut c_void,
    ) -> Result<()> {
        let rows = requests * self.width as u32;
        unsafe {
            self.ops.frequencies(
                self.positions.buffer,
                self.output.frequencies(),
                rows,
                stream,
            )?;
            self.qa
                .enqueue(self.qa.input(), self.qa.output_storage(), rows, stream)?;
            self.ops.norm(
                self.qa.output_storage(),
                self.q_norm,
                None,
                self.qb.input(),
                rows,
                1280,
                stream,
            )?;
            self.qb
                .enqueue(self.qb.input(), self.qb.output_storage(), rows, stream)?;
            self.ops.rope(
                self.qb.output_storage(),
                self.output.frequencies(),
                self.qb.output_storage(),
                rows,
                64,
                false,
                stream,
            )?;
            self.kv
                .enqueue(self.qa.input(), self.kv.output_storage(), rows, stream)?;
            self.ops.kv(
                self.kv.output_storage(),
                self.kv_norm,
                self.output.frequencies(),
                self.draft.buffer,
                rows,
                stream,
            )?;
            self.attention.launch(
                self.qb.output_storage(),
                read.ring,
                self.draft.buffer,
                self.sink,
                self.descriptors.buffer,
                self.output.input(),
                requests,
                read.slots,
                stream,
            )?;
            self.output.enqueue(rows, stream)
        }
    }
    /// # Safety
    /// Initialize finite normalized BF16 hidden rows [requests,K,5120];
    /// complete producer writes and serialize raw input view reuse. Frequencies
    /// are generated from the generation-checked committed ends on this stream.
    pub unsafe fn execute(
        &mut self,
        window: &DsparkWindow<'_>,
        requests: &[(WindowLease, u64)],
    ) -> Result<Ds41rtDeviceBuffer> {
        let read = self.prepare(window, requests)?;
        self.upload(&read, requests)?;
        let launched = unsafe { self.enqueue_on(&read, requests.len() as u32, self.stream.raw) };
        let drained = self.synchronize();
        launched.and(drained)?;
        self.ready = Some(requests.len() as u32);
        self.output()
    }
    /// # Safety
    /// Same input contract as execute. The graph may only replay against this
    /// window owner, revalidated on every call; destruction never reads its ring.
    pub unsafe fn capture(
        &mut self,
        window: &DsparkWindow<'_>,
        requests: &[(WindowLease, u64)],
    ) -> Result<()> {
        ensure!(self.graph.is_none(), "attention wave already captured");
        unsafe {
            self.execute(window, requests)?;
        }
        let read = self.prepare(window, requests)?;
        unsafe {
            self.stream
                .library
                .cuda_graph_begin_capture(self.stream.raw)?;
        }
        let launched = unsafe { self.enqueue_on(&read, requests.len() as u32, self.stream.raw) };
        let captured = unsafe { self.stream.library.cuda_graph_end_capture(self.stream.raw) };
        match (launched, captured) {
            (Ok(()), Ok(graph)) => {
                self.graph = Some((graph, requests.len() as u32, read.owner));
                Ok(())
            }
            (Err(error), Ok(graph)) => {
                if let Err(cleanup) = unsafe { self.stream.library.cuda_graph_exec_destroy(graph) }
                {
                    tracing::error!(%cleanup,"destroying failed attention wave capture");
                }
                Err(error)
            }
            (Err(error), Err(_)) | (Ok(()), Err(error)) => Err(error),
        }
    }
    /// # Safety
    /// Same input contract as execute; cache owner and captured request count must match.
    pub unsafe fn replay(
        &mut self,
        window: &DsparkWindow<'_>,
        requests: &[(WindowLease, u64)],
    ) -> Result<Ds41rtDeviceBuffer> {
        let read = self.prepare(window, requests)?;
        let (graph, count, owner) = self.graph.context("attention wave not captured")?;
        ensure!(
            count as usize == requests.len() && owner == read.owner,
            "attention wave capture binding differs"
        );
        self.upload(&read, requests)?;
        let launched = unsafe {
            self.stream
                .library
                .cuda_graph_launch(graph, self.stream.raw)
        };
        let drained = self.synchronize();
        launched.and(drained)?;
        self.ready = Some(count);
        self.output()
    }
    pub fn output(&self) -> Result<Ds41rtDeviceBuffer> {
        let requests = self.ready.context("attention wave output incomplete")?;
        let mut output = self.output.output_storage();
        output.bytes = requests as usize * self.width * 10240;
        Ok(output)
    }
}
impl Drop for DsparkAttentionWave<'_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self.synchronize() {
            tracing::error!(%error,"draining attention wave");
        }
        if let Some((graph, _, _)) = self.graph.take() {
            if let Err(error) = unsafe { self.stream.library.cuda_graph_exec_destroy(graph) } {
                tracing::error!(%error,"destroying attention wave graph");
            }
        }
    }
}
