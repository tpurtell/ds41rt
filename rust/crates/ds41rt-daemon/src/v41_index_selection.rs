//! Bounded, captured index selection and snapshot-checked candidate sharing.
use crate::v41_attention_binding::QueryBinding;
use crate::v41_compressor::{IndexBinding, IndexProposal};
use crate::v41_index_query::IndexQueryOutput;
use crate::v41_memory::{DeviceAllocation, HostAllocation, LoadStream};
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{
    Ds41rtDeviceBuffer, NativeLibrary, V41CandidateBlocks, V41IndexScores, V41IndexTopK,
};
use std::{collections::VecDeque, cell::Cell, ffi::c_void, marker::PhantomData, rc::Rc};
const WIDTH: usize = 16384;
const BLOCKS: usize = WIDTH / 8;
// Index fingerprints include request layouts; retain a bounded recent set.
// Per-layer projection graphs have a separate residency limit.
const MAX_RETAINED_DECODE_GRAPHS: usize = 512;

pub(crate) struct SelectionRequest<'a> {
    pub proposal: &'a IndexProposal<'a>,
    /// Increasing query token positions, in the query producer's row order.
    pub positions: &'a [u64],
}
pub(crate) struct IndexSelectionOutput<'a> {
    origin: Option<QueryBinding>,
    /// Logical compressed row IDs, sorted ascending, padded with -1. No window offset.
    pub selected: Ds41rtDeviceBuffer,
    pub rows: usize,
    layer: usize,
    blocks: Option<Ds41rtDeviceBuffer>,
    bindings: &'a [(IndexBinding, u64)],
    _sources: PhantomData<&'a ()>,
}
struct Ready {
    origin: Option<QueryBinding>,
    layer: usize,
    rows: usize,
    bindings: Vec<(IndexBinding, u64)>,
}
impl IndexSelectionOutput<'_> {
    /// Reuse logical row selections on a peer without changing their snapshot
    /// identity. Candidate blocks remain private to the original index producer.
    /// # Safety
    /// Selected contains an exact copy of this output on the consumer device.
    /// Order publication before consumption and retain its allocation until all
    /// consumers drain, including cold graph preparation and cancellation.
    pub unsafe fn peer_attention(&self, selected: Ds41rtDeviceBuffer) -> Result<IndexSelectionOutput<'_>> {
        ensure!(selected.device_id >= 0 && selected.device_id != self.selected.device_id
            && !selected.ptr.is_null() && selected.flags == 0
            && self.rows.checked_mul(2048) == Some(selected.bytes),
            "peer selection storage differs");
        Ok(IndexSelectionOutput { origin: self.origin, selected, rows: self.rows,
            layer: self.layer, blocks: None, bindings: self.bindings, _sources: PhantomData })
    }
    #[cfg(test)]
    pub(crate) fn candidate_blocks(&self) -> Option<Ds41rtDeviceBuffer> { self.blocks }
    pub fn validate_query(&self, query: QueryBinding) -> Result<()> {
        let origin = self.origin.context("selection has no query origin")?;
        ensure!(
            origin.layer() == self.layer && (query.layer() != self.layer || origin == query),
            "selection query snapshot differs"
        );
        Ok(())
    }

    /// Bind an attention row to the exact source execution and token used by
    /// selection. Intermediate layers reuse their nearest index producer.
    pub fn validate_attention(&self, layer: usize, bindings: &[(IndexBinding, u64)]) -> Result<()> {
        let producer = [2, 8, 14, 20, 24, 28, 32, 36]
            .into_iter()
            .rev()
            .find(|&n| n <= layer);
        ensure!(
            layer < 40
                && producer == Some(self.layer)
                && self.bindings == bindings
                && self.rows == bindings.len(),
            "attention selection layer, snapshot or row order differs"
        );
        Ok(())
    }
}
pub(crate) struct IndexSelectionWave<'a> {
    stream: LoadStream<'a>,
    buffers: Vec<Rc<DeviceAllocation<'a>>>,
    shared_scratch_busy: Option<Rc<Cell<bool>>>,
    staging: HostAllocation<'a>,
    capacity: usize,
    score: V41IndexScores<'a>,
    top: V41IndexTopK<'a>,
    candidates: V41CandidateBlocks<'a>,
    graph: Option<(*mut c_void, Vec<usize>)>,
    retained_graphs: VecDeque<(*mut c_void, Vec<usize>)>,
    retain_decode_graphs: bool,
    ready: Option<Ready>,
    pending: Option<Ready>,
    in_flight: bool,
}
fn slice(mut b: Ds41rtDeviceBuffer, offset: usize, bytes: usize) -> Ds41rtDeviceBuffer {
    debug_assert!(offset + bytes <= b.bytes);
    b.ptr = unsafe { b.ptr.cast::<u8>().add(offset).cast() };
    b.bytes = bytes;
    b
}
impl<'a> IndexSelectionWave<'a> {
    fn sizes(capacity: usize) -> Result<[usize; 13]> {
        ensure!(
            (1..=4096).contains(&capacity),
            "invalid index selection capacity"
        );
        Ok([
            48,
            8,
            8,
            WIDTH * 8,
            WIDTH * 4,
            BLOCKS * 4,
            BLOCKS * 8,
            4096,
            V41IndexTopK::scratch_bytes(1, WIDTH)?,
            2048,
            16384,
            V41IndexTopK::block_scratch_bytes(1, BLOCKS)?,
            8192,
        ]
        .map(|n| n * capacity))
    }
    pub fn device_bytes(capacity: usize) -> Result<usize> {
        Ok(Self::sizes(capacity)?.iter().sum())
    }
    pub fn new(library: &'a NativeLibrary, capacity: usize, budget: usize) -> Result<Self> {
        ensure!(
            Self::device_bytes(capacity)? <= budget,
            "index selection exceeds budget"
        );
        Ok(Self {
            stream: LoadStream {
                library,
                raw: library.cuda_stream_create()?,
            },
            buffers: Self::sizes(capacity)?
                .into_iter()
                .map(|n| DeviceAllocation::new(library, n).map(Rc::new))
                .collect::<Result<_>>()?,
            shared_scratch_busy: None,
            staging: HostAllocation::new(library, capacity * 56)?,
            capacity,
            score: library.v41_index_scores()?,
            top: library.v41_index_topk()?,
            candidates: library.v41_candidate_blocks()?,
            graph: None, retained_graphs: VecDeque::new(), retain_decode_graphs: false,
            ready: None,
            pending: None,
            in_flight: false,
        })
    }
    /// Only selected IDs and source candidate blocks survive another stage.
    pub fn shared_device_bytes(capacity: usize) -> Result<usize> {
        let sizes = Self::sizes(capacity)?;
        Ok(sizes[9] + sizes[12])
    }
    /// Create a follower in the same lane. Both owners retain independent
    /// outputs/graphs; their temporary scores and top-k workspace are shared.
    /// An outstanding queued selection excludes the other owner until polled
    /// complete or explicitly drained. No cross-lane storage is involved.
    pub fn sharing_scratch(source: &mut Self, budget: usize) -> Result<Self> {
        ensure!(!source.in_flight && source.shared_scratch_busy.is_none(), "index scratch already shared or pending");
        source.stream.require_complete()?;
        let capacity = source.capacity;
        ensure!(Self::shared_device_bytes(capacity)? <= budget, "shared index outputs exceed budget");
        let library = source.stream.library;
        ensure!(library.cuda_get_device()? == source.b(0).device_id, "shared index workspace device differs");
        let sizes = Self::sizes(capacity)?;
        let buffers = source.buffers.iter().enumerate().map(|(i, allocation)| {
            if [9, 12].contains(&i) { DeviceAllocation::new(library, sizes[i]).map(Rc::new) }
            else { Ok(allocation.clone()) }
        }).collect::<Result<Vec<_>>>()?;
        let busy = Rc::new(Cell::new(false));
        let value = Self {
            stream: LoadStream { library, raw: library.cuda_stream_create()? }, buffers,
            shared_scratch_busy: Some(busy.clone()), staging: HostAllocation::new(library, capacity * 56)?,
            capacity, score: library.v41_index_scores()?, top: library.v41_index_topk()?,
            candidates: library.v41_candidate_blocks()?, graph: None,
            retained_graphs: VecDeque::new(), retain_decode_graphs: false, ready: None, pending: None, in_flight: false,
        };
        source.shared_scratch_busy = Some(busy);
        Ok(value)
    }
    fn b(&self, i: usize) -> Ds41rtDeviceBuffer {
        self.buffers[i].buffer
    }
    fn synchronize(&self) -> Result<()> {
        unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) }
    }
    pub fn enable_small_graph_shapes(&mut self) {
        if !self.retain_decode_graphs {
            self.retained_graphs.reserve(MAX_RETAINED_DECODE_GRAPHS);
            self.retain_decode_graphs = true;
        }
    }
    /// Invalidate published selections without discarding immutable launch shapes.
    pub fn restart(&mut self) -> Result<()> {
        if !self.retain_decode_graphs { return self.clear_graph(); }
        ensure!(!self.in_flight, "index selection pending");
        self.stream.require_complete()?;
        self.ready = None;
        Ok(())
    }
    fn select_graph(&mut self, fingerprint: &[usize]) -> Result<()> {
        if self.graph.as_ref().is_some_and(|(_, key)| key == fingerprint) { return Ok(()); }
        if !self.retain_decode_graphs { return self.clear_graph(); }
        self.restart()?;
        let found = self.retained_graphs.iter().position(|(_, key)| key == fingerprint)
            .and_then(|index| self.retained_graphs.remove(index));
        if let Some(old) = self.graph.take() {
            // Fingerprint begins with layer and row count. All remaining fields
            // (including source pointers, widths and candidate mode) still match
            // exactly before replay. Large prefill graphs are never retained.
            if old.1[1] <= 8 * (ds41rt_core::MAX_DSPARK_PROPOSALS + 1) {
                if self.retained_graphs.len() == MAX_RETAINED_DECODE_GRAPHS {
                    let (graph, _) = self.retained_graphs.pop_front().unwrap();
                    unsafe { self.stream.library.cuda_graph_exec_destroy(graph)?; }
                }
                self.retained_graphs.push_back(old);
            } else {
                unsafe { self.stream.library.cuda_graph_exec_destroy(old.0)?; }
            }
        }
        self.graph = found;
        Ok(())
    }
    pub fn clear_graph(&mut self) -> Result<()> {
        ensure!(!self.in_flight, "index selection pending");
        self.ready = None;
        self.stream.require_complete()?;
        let mut failure = None;
        for (graph, _) in self.graph.take().into_iter().chain(self.retained_graphs.drain(..)) {
            if let Err(error) = unsafe { self.stream.library.cuda_graph_exec_destroy(graph) } {
                failure.get_or_insert(error);
            }
        }
        failure.map_or(Ok(()), Err)
    }
    unsafe fn enqueue(
        &self,
        query: &IndexQueryOutput<'_>,
        requests: &[SelectionRequest<'_>],
        shared: Option<&IndexSelectionOutput<'_>>,
        rows: usize,
        tiles: usize,
        width: usize,
        use_candidates: bool,
    ) -> Result<()> {
        let source = query.layer == 20;
        let follower = use_candidates;
        let stream = self.stream.raw;
        let passes = if follower { 1 } else { tiles };
        for tile in 0..passes {
            unsafe {
                if follower {
                    self.candidates.expand(
                        shared
                            .context("candidate source missing")?
                            .blocks
                            .context("candidate blocks missing")?,
                        self.b(1),
                        self.b(3),
                        rows,
                        stream,
                    )?;
                } else {
                    self.candidates.tile(
                        self.b(3),
                        self.b(2),
                        rows,
                        width,
                        (tile * width) as u64,
                        stream,
                    )?;
                }
                let mut offset = 0;
                for request in requests {
                    let n = request.positions.len();
                    let p = request.proposal;
                    self.score.execute_overlay(
                        slice(query.packed, offset * 2048, n * 2048),
                        slice(query.scales, offset * 128, n * 128),
                        slice(query.head_weights, offset * 64, n * 64),
                        p.cache.packed,
                        p.cache.scales,
                        p.cache.device_pages,
                        p.cache.device_rows,
                        slice(self.b(0), offset * 48, n * 48),
                        slice(self.b(3), offset * width * 8, n * width * 8),
                        slice(self.b(4), offset * width * 4, n * width * 4),
                        p.packed,
                        p.scales,
                        p.capacity,
                        n,
                        width,
                        1,
                        p.cache.device_pages.bytes / 4,
                        p.cache.packed.bytes / 64,
                        stream,
                    )?;
                    offset += n;
                }
                self.top.update(
                    self.b(4),
                    self.b(3),
                    self.b(7),
                    self.b(8),
                    self.b(9),
                    rows,
                    width,
                    tile == 0,
                    stream,
                )?;
                if source {
                    self.candidates.maxima(
                        self.b(4),
                        self.b(2),
                        self.b(1),
                        self.b(5),
                        self.b(6),
                        rows,
                        width,
                        stream,
                    )?;
                    self.top.update_blocks(
                        self.b(5),
                        self.b(6),
                        self.b(10),
                        self.b(11),
                        self.b(12),
                        rows,
                        width / 8,
                        tile == 0,
                        stream,
                    )?;
                }
            }
        }
        Ok(())
    }
    /// # Safety
    /// Query rows are finite and correspond exactly to the supplied request/token
    /// order; their producers have drained. Proposal borrows are from the matching
    /// source's current wave. No external writes race inputs, caches or outputs.
    /// Results borrow the proposal views until their consumers have drained.
    pub unsafe fn execute<'s>(
        &'s mut self,
        query: &IndexQueryOutput<'_>,
        requests: &[SelectionRequest<'_>],
        shared: Option<&IndexSelectionOutput<'_>>,
    ) -> Result<IndexSelectionOutput<'s>> {
        unsafe { self.execute_inner(query, requests, shared, false)?; }
        self.output()
    }
    /// # Safety
    /// Retain query, proposals and optional source candidates through polling or abort.
    pub unsafe fn enqueue_selection(&mut self, query: &IndexQueryOutput<'_>,
        requests: &[SelectionRequest<'_>], shared: Option<&IndexSelectionOutput<'_>>) -> Result<()> {
        let result = unsafe { self.execute_inner(query, requests, shared, true) };
        if result.is_err() { self.abort_pending()?; }
        result
    }
    pub fn poll_pending(&mut self) -> Result<bool> {
        ensure!(self.in_flight && self.pending.is_some(), "no pending index selection");
        let ready = unsafe { self.stream.library.cuda_stream_query(self.stream.raw) };
        match ready {
            Ok(false) => Ok(false),
            Ok(true) => { self.ready = self.pending.take(); self.in_flight = false;
                if let Some(busy) = &self.shared_scratch_busy { busy.set(false); }
                Ok(true) }
            Err(error) => { self.abort_pending()?; Err(error) }
        }
    }
    pub fn abort_pending(&mut self) -> Result<()> {
        if self.in_flight {
            self.synchronize()?;
            self.in_flight = false; self.pending = None; self.ready = None;
            if let Some(busy) = &self.shared_scratch_busy { busy.set(false); }
        }
        Ok(())
    }
    unsafe fn execute_inner(&mut self, query: &IndexQueryOutput<'_>,
        requests: &[SelectionRequest<'_>], shared: Option<&IndexSelectionOutput<'_>>,
        defer: bool) -> Result<()> {
        ensure!(!self.in_flight, "index selection pending");
        ensure!(!self.shared_scratch_busy.as_ref().is_some_and(|busy| busy.get()), "shared index scratch is pending");
        self.ready = None;
        ensure!(
            [2, 8, 14, 20, 24, 28, 32, 36].contains(&query.layer),
            "invalid selection producer layer"
        );
        ensure!(
            !requests.is_empty() && requests.len() <= 16,
            "invalid selection request count"
        );
        let rows: usize = requests.iter().map(|r| r.positions.len()).sum();
        ensure!(
            rows > 0 && rows <= self.capacity,
            "selection rows exceed capacity"
        );
        ensure!(
            query.packed.bytes == rows * 2048
                && query.scales.bytes == rows * 128
                && query.head_weights.bytes == rows * 64,
            "query output row count differs"
        );
        ensure!(
            query.packed.device_id == self.b(0).device_id,
            "selection query device differs"
        );
        let expected_source = if query.layer >= 20 { 20 } else { query.layer };
        let mut bindings = Vec::with_capacity(rows);
        let mut max_length = 0u64;
        let mut fingerprint = vec![
            query.layer,
            rows,
            query.packed.ptr as usize,
            query.scales.ptr as usize,
            query.head_weights.ptr as usize,
        ];
        let mut metadata = Vec::with_capacity(rows);
        for (i, r) in requests.iter().enumerate() {
            ensure!(
                !r.positions.is_empty() && r.positions.windows(2).all(|x| x[0] < x[1]),
                "empty or unordered selection request"
            );
            ensure!(
                r.proposal.source_layer == expected_source,
                "index key source layer differs"
            );
            ensure!(
                requests[..i]
                    .iter()
                    .all(|x| !x.proposal.binding().same_request(r.proposal.binding())),
                "duplicate selection request"
            );
            ensure!(
                requests[0]
                    .proposal
                    .binding()
                    .same_pool(r.proposal.binding()),
                "selection state owner differs"
            );
            let p = r.proposal;
            fingerprint.extend([
                r.positions.len(),
                p.cache.packed.ptr as usize,
                p.cache.scales.ptr as usize,
                p.cache.device_pages.ptr as usize,
                p.cache.device_rows.ptr as usize,
                p.cache.device_pages.bytes,
                p.cache.packed.bytes,
                p.packed.ptr as usize,
                p.scales.ptr as usize,
                p.capacity,
            ]);
            for &position in r.positions {
                let m = p.metadata(position)?;
                max_length = max_length.max(m[1]);
                metadata.push(m);
                bindings.push((p.binding(), position));
            }
        }
        if query.origin().is_some() {
            ensure!(
                bindings
                    .iter()
                    .map(|(_, p)| *p)
                    .eq(query.bound_tokens()?.iter().copied()),
                "selection token order differs from query producer"
            );
        }
        if query.layer > 20 {
            let s = shared.context("later index layer requires source candidates")?;
            ensure!(
                s.layer == 20 && s.rows == rows && s.bindings == bindings && s.blocks.is_some(),
                "candidate source snapshot or row order differs"
            );
            fingerprint.push(s.blocks.unwrap().ptr as usize);
        } else {
            ensure!(
                shared.is_none(),
                "candidate source is unexpected for this layer"
            );
        }
        let width = (max_length as usize).max(1).div_ceil(8).min(WIDTH / 8) * 8;
        let use_candidates = query.layer > 20 && max_length > WIDTH as u64;
        let tiles = (max_length as usize).div_ceil(width).max(1);
        fingerprint.extend([width, usize::from(use_candidates)]);
        fingerprint.push(if query.layer > 20 { 1 } else { tiles });
        let staging = self.staging.bytes_mut();
        for (i, m) in metadata.iter().enumerate() {
            for (j, value) in m.iter().enumerate() {
                staging[i * 48 + j * 8..i * 48 + j * 8 + 8].copy_from_slice(&value.to_ne_bytes());
            }
            staging[rows * 48 + i * 8..rows * 48 + i * 8 + 8].copy_from_slice(&m[1].to_ne_bytes());
        }
        self.select_graph(&fingerprint)?;
        if defer {
            self.in_flight = true;
            if let Some(busy) = &self.shared_scratch_busy { busy.set(true); }
            let host = self.staging.buffer;
            unsafe {
                self.stream.library.copy_host_buffer_h2d_async(self.b(0), host, rows * 48, self.stream.raw)?;
                let mut lengths = host;
                lengths.ptr = host.ptr.cast::<u8>().add(rows * 48).cast(); lengths.bytes = rows * 8;
                self.stream.library.copy_host_buffer_h2d_async(self.b(1), lengths, rows * 8, self.stream.raw)?;
            }
        } else {
            self.stream.library.copy_h2d(self.b(0), &self.staging.bytes_mut()[..rows * 48])?;
            self.stream.library.copy_h2d(self.b(1), &self.staging.bytes_mut()[rows * 48..rows * 56])?;
        }
        if self.graph.is_none() {
            unsafe {
                self.stream
                    .library
                    .cuda_graph_begin_capture(self.stream.raw)?;
            }
            let launched = unsafe {
                self.enqueue(query, requests, shared, rows, tiles, width, use_candidates)
            };
            let captured = unsafe { self.stream.library.cuda_graph_end_capture(self.stream.raw) };
            match (launched, captured) {
                (Ok(()), Ok(g)) => self.graph = Some((g, fingerprint)),
                (Err(e), Ok(g)) => {
                    unsafe {
                        self.stream.library.cuda_graph_exec_destroy(g)?;
                    }
                    return Err(e);
                }
                (Err(e), Err(_)) | (Ok(()), Err(e)) => return Err(e),
            }
        }
        let g = self.graph.as_ref().context("selection graph missing")?.0;
        let launched = unsafe { self.stream.library.cuda_graph_launch(g, self.stream.raw) };
        let ready = Ready { origin: query.origin(), layer: query.layer, rows, bindings };
        if defer {
            launched?;
            self.pending = Some(ready);
        } else {
            launched.and(self.synchronize())?;
            self.ready = Some(ready);
        }
        Ok(())
    }
    /// Borrow completed selection storage. Consumers validate its retained source
    /// bindings against live cache proposals before using these logical row IDs.
    pub fn output(&self) -> Result<IndexSelectionOutput<'_>> {
        let r = self.ready.as_ref().context("index selection output unpublished")?;
        let rows = r.rows;
        Ok(IndexSelectionOutput {
            origin: r.origin,
            selected: slice(self.b(9), 0, rows * 2048),
            rows: r.rows,
            layer: r.layer,
            blocks: (r.layer == 20).then(|| slice(self.b(12), 0, rows * 8192)),
            bindings: &r.bindings,
            _sources: PhantomData,
        })
    }
}
impl Drop for IndexSelectionWave<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.abort_pending() { tracing::error!(%error, "draining pending index selection"); }
        if let Err(error) = self.clear_graph() {
            tracing::error!(%error,"draining index selection graph");
        }
    }
}

#[cfg(test)]
mod graph_tests {
    use super::*;

    #[test]
    fn peer_selection_preserves_query_snapshot_validation() -> Result<()> {
        let origin=QueryBinding::new(2)?;
        // Only metadata is inspected; these pointers are never dereferenced.
        let buffer=Ds41rtDeviceBuffer { ptr: std::ptr::NonNull::<u8>::dangling().as_ptr().cast(),
            bytes: 2048, device_id: 0, flags: 0 };
        let original=IndexSelectionOutput { origin: Some(origin), selected: buffer,
            rows: 1, layer: 2, blocks: Some(buffer), bindings: &[], _sources: PhantomData };
        let mut destination=buffer; destination.device_id=1;
        let peer=unsafe { original.peer_attention(destination)? };
        peer.validate_query(origin)?;
        assert!(peer.validate_query(QueryBinding::new(2)?).is_err());
        // Intermediate layers may reuse this producer, as on the original GPU.
        peer.validate_query(QueryBinding::new(3)?)?;
        assert!(peer.blocks.is_none());
        assert!(unsafe { original.peer_attention(buffer) }.is_err());
        destination.bytes-=1;
        assert!(unsafe { original.peer_attention(destination) }.is_err());
        Ok(())
    }

    #[test]
    fn cuda_selection_shapes_replay_current_inputs_after_restart() -> Result<()> {
        let Some(path) = std::env::var_os("DS41RT_INDEX_GRAPH_LIBRARY") else {
            eprintln!("skip selection graph lifetime test: DS41RT_INDEX_GRAPH_LIBRARY unset");
            return Ok(());
        };
        let library = unsafe { NativeLibrary::load(path)? };
        library.cuda_set_device(0)?;
        let mut wave = IndexSelectionWave::new(&library, 80, IndexSelectionWave::device_bytes(80)?)?;
        wave.enable_small_graph_shapes();
        let input = slice(wave.b(0), 0, 320);
        let output = slice(wave.b(1), 0, 320);
        let mut handles = std::collections::BTreeMap::new();
        for (iteration, (layer, rows)) in [(2, 4), (8, 2), (2, 4), (8, 2), (2, 80), (2, 4), (8, 2)]
            .into_iter().enumerate() {
            wave.restart()?;
            let fingerprint = vec![layer, rows, input.ptr as usize, output.ptr as usize];
            wave.select_graph(&fingerprint)?;
            if wave.graph.is_none() {
                unsafe {
                    library.cuda_graph_begin_capture(wave.stream.raw)?;
                    library.copy_d2d_async(output, input, rows * 4, wave.stream.raw)?;
                    let graph = library.cuda_graph_end_capture(wave.stream.raw)?;
                    wave.graph = Some((graph, fingerprint));
                }
            }
            let graph = wave.graph.as_ref().unwrap().0;
            if rows <= 64 {
                assert_eq!(*handles.entry((layer, rows)).or_insert(graph), graph,
                    "restart or large-prefill transition discarded a decode graph");
            }
            let value = iteration as u8 + 1;
            library.copy_h2d(input, &vec![value; 320])?;
            library.copy_h2d(output, &[0; 320])?;
            unsafe { library.cuda_graph_launch(graph, wave.stream.raw)?; }
            wave.synchronize()?;
            let mut actual = vec![0; 320];
            library.copy_d2h(&mut actual, output)?;
            assert_eq!(&actual[..rows * 4], vec![value; rows * 4]);
            assert!(actual[rows * 4..].iter().all(|&x| x == 0));
        }
        wave.clear_graph()?;
        assert!(wave.graph.is_none() && wave.retained_graphs.is_empty());
        // Exercise actual CUDA handle destruction at the retention limit, then
        // replay current data through each newly captured graph. Distinct final
        // fingerprint fields stand in for changing request/source layouts.
        for key in 0..MAX_RETAINED_DECODE_GRAPHS + 3 {
            wave.restart()?;
            let fingerprint = vec![2, 4, input.ptr as usize, output.ptr as usize, key];
            wave.select_graph(&fingerprint)?;
            assert!(wave.graph.is_none());
            unsafe {
                library.cuda_graph_begin_capture(wave.stream.raw)?;
                library.copy_d2d_async(output, input, 16, wave.stream.raw)?;
                let graph = library.cuda_graph_end_capture(wave.stream.raw)?;
                wave.graph = Some((graph, fingerprint));
            }
            assert!(wave.retained_graphs.len() <= MAX_RETAINED_DECODE_GRAPHS);
            let value = (key % 251 + 1) as u8;
            library.copy_h2d(input, &vec![value; 320])?;
            library.copy_h2d(output, &[0; 320])?;
            unsafe { library.cuda_graph_launch(wave.graph.as_ref().unwrap().0, wave.stream.raw)?; }
            wave.synchronize()?;
            let mut actual = vec![0; 320];
            library.copy_d2h(&mut actual, output)?;
            assert_eq!(&actual[..16], &[value; 16]);
            assert!(actual[16..].iter().all(|&x| x == 0));
        }
        assert_eq!(wave.retained_graphs.len(), MAX_RETAINED_DECODE_GRAPHS);
        assert!(!wave.retained_graphs.iter().any(|(_, key)| key[4] == 0));
        wave.clear_graph()?;
        assert!(wave.graph.is_none() && wave.retained_graphs.is_empty());
        Ok(())
    }
}
