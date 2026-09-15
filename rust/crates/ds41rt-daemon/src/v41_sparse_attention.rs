//! Captured sparse attention bound to live window/source/selection proposals.
use crate::v41_attention_binding::QueryBinding;
use crate::v41_attention_query::AttentionQueryOutput;
use crate::v41_compressor::IndexProposal;
use crate::v41_index_selection::IndexSelectionOutput;
use crate::v41_memory::{DeviceAllocation, HostAllocation, LoadStream};
use crate::v41_window::WindowProposal;
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{
    Ds41rtDeviceBuffer, NativeLibrary, V41SparseAttention, V41SparseBatch, V41SparseSource, V41SparseWindow, V41Kv,
};
use std::{collections::VecDeque, ffi::c_void, marker::PhantomData};

pub(crate) struct AttentionRequest<'a> {
    pub window: &'a WindowProposal<'a>,
    pub source: Option<&'a IndexProposal<'a>>,
    pub positions: &'a [u64],
}
pub(crate) struct SparseAttentionOutput<'a> {
    query: Option<QueryBinding>,
    tokens: Option<&'a [u64]>,
    pub values: Ds41rtDeviceBuffer,
    pub layer: usize,
    pub rows: usize,
    _inputs: PhantomData<&'a ()>,
}
impl SparseAttentionOutput<'_> {
    pub fn binding(&self) -> Result<QueryBinding> {
        self.tokens()?;
        self.query.context("attention query missing")
    }

    pub fn tokens(&self) -> Result<&[u64]> {
        let query = self.query.context("attention has no query origin")?;
        ensure!(query.layer() == self.layer, "attention query layer differs");
        self.tokens.context("attention tokens missing")
    }
}
/// Internal view: producer is queued on the supplied stream, not host-ready.
pub(crate) struct QueuedSparseAttention {
    pub values: Ds41rtDeviceBuffer,
    pub layer: usize,
    pub rows: usize,
}

/// The containing lane retains captured consumer storage and immutable weights
/// until sparse graphs are destroyed. No callback may suspend during capture.
pub(crate) trait AttentionGraphTail {
    fn identity(&self) -> Vec<usize>;
    unsafe fn prepare(&mut self, stream: *mut c_void) -> Result<()>;
    unsafe fn enqueue(&mut self, attention: &QueuedSparseAttention, stream: *mut c_void) -> Result<()>;
    unsafe fn restore_warmup(&mut self) -> Result<()>;
    unsafe fn replay_state(&mut self) -> Result<()>;
}
pub(crate) struct SparseAttentionWave<'a> {
    stream: LoadStream<'a>,
    kernel: V41SparseAttention<'a>,
    query: DeviceAllocation<'a>,
    output: DeviceAllocation<'a>,
    split_scratch: DeviceAllocation<'a>,
    descriptors: DeviceAllocation<'a>,
    descriptor_staging: HostAllocation<'a>,
    replay_begins: DeviceAllocation<'a>,
    metadata: DeviceAllocation<'a>,
    staging: HostAllocation<'a>,
    replay_staging: HostAllocation<'a>,
    capacity: usize,
    batch_rows: usize,
    // Batched keys retain row count and stable wave/selection/sink storage;
    // current descriptors are validated and uploaded before EVERY replay.
    // For other paths, complete external-pointer and launch-geometry fingerprints are checked
    // against live proposals before every replay. Inactive graphs never launch.
    // Adaptive mode keeps at most batch_rows variants per layer, including prefill;
    // request layouts can have many more combinations than total row counts.
    graphs: [VecDeque<(*mut c_void, Vec<usize>)>; 40],
    graph_limit: usize,
    warmed_kernels: u16,
    cold: Option<ColdSparse>,
}
struct RequestLaunch {
    window: V41SparseWindow,
    source: Option<V41SparseSource>,
    rows: usize,
    width: usize,
}
// Owned launch description; the containing lane guard retains all referenced
// query/cache/selection allocations until warmup, replay and consumers finish.
struct ColdSparse {
    layer: usize,
    rows: usize,
    sink: Ds41rtDeviceBuffer,
    launches: Vec<RequestLaunch>,
    selected: Option<Ds41rtDeviceBuffer>,
    batch: Option<V41SparseBatch>,
    fingerprint: Vec<usize>,
    needed: u16,
    tail_identity: Option<Vec<usize>>,
}
fn slice(mut b: Ds41rtDeviceBuffer, offset: usize, bytes: usize) -> Ds41rtDeviceBuffer {
    debug_assert!(offset + bytes <= b.bytes);
    b.ptr = unsafe { b.ptr.cast::<u8>().add(offset).cast() };
    b.bytes = bytes;
    b
}
impl<'a> SparseAttentionWave<'a> {
    pub fn device_bytes(capacity: usize) -> Result<usize> {
        ensure!(
            (1..=4096).contains(&capacity),
            "invalid sparse attention capacity"
        );
        Ok(capacity * 131160 + V41SparseAttention::split_scratch_bytes(capacity.min(48), 10)?
            + V41SparseAttention::batch_descriptor_bytes(capacity.min(48))?)
    }
    pub fn new(library: &'a NativeLibrary, capacity: usize, budget: usize) -> Result<Self> {
        ensure!(
            Self::device_bytes(capacity)? <= budget,
            "sparse attention exceeds budget"
        );
        Ok(Self {
            stream: LoadStream {
                library,
                raw: library.cuda_stream_create()?,
            },
            kernel: library.v41_sparse_attention()?,
            query: DeviceAllocation::new(library, capacity * 65536)?,
            output: DeviceAllocation::new(library, capacity * 65536)?,
            split_scratch: DeviceAllocation::new(
                library,
                V41SparseAttention::split_scratch_bytes(capacity.min(48), 10)?,
            )?,
            descriptors: DeviceAllocation::new(library, V41SparseAttention::batch_descriptor_bytes(capacity.min(48))?)?,
            descriptor_staging: HostAllocation::new(library, V41SparseAttention::batch_descriptor_bytes(capacity.min(48))?)?,
            replay_begins: DeviceAllocation::new(library, capacity * 8)?,
            metadata: DeviceAllocation::new(library, capacity * 80)?,
            staging: HostAllocation::new(library, capacity * 80)?,
            replay_staging: HostAllocation::new(library, capacity * 8)?,
            capacity,
            batch_rows: capacity.min(48),
            graphs: std::array::from_fn(|_| VecDeque::new()),
            graph_limit: 1,
            warmed_kernels: 0,
            cold: None,
        })
    }
    /// Reserve wider decode batching before any execution and before KV sizing.
    /// K5 retains its original allocation; K7 pays only for the extra 16 rows.
    pub fn reserve_decode_rows(&mut self, rows: usize) -> Result<()> {
        ensure!(rows > 0 && rows <= self.capacity && rows <= 64,
            "invalid sparse decode reservation");
        ensure!(self.cold.is_none() && self.graphs.iter().all(VecDeque::is_empty),
            "sparse decode reservation requires an unused wave");
        if rows <= self.batch_rows { return Ok(()); }
        let library = self.stream.library;
        let scratch = DeviceAllocation::new(library, V41SparseAttention::split_scratch_bytes(rows, 10)?)?;
        let bytes = V41SparseAttention::batch_descriptor_bytes(rows)?;
        let descriptors = DeviceAllocation::new(library, bytes)?;
        let staging = HostAllocation::new(library, bytes)?;
        let additional_device_bytes = scratch.buffer.bytes + descriptors.buffer.bytes
            - self.split_scratch.buffer.bytes - self.descriptors.buffer.bytes;
        self.split_scratch = scratch;
        self.descriptors = descriptors;
        self.descriptor_staging = staging;
        self.batch_rows = rows;
        if self.graph_limit > 1 { self.graph_limit = rows; }
        tracing::info!(device=self.query.buffer.device_id, rows, additional_device_bytes,
            "sparse decode batch reservation before KV sizing");
        Ok(())
    }
    pub fn enable_small_graph_shapes(&mut self) { self.graph_limit = self.batch_rows; }
    pub fn input(&self) -> Ds41rtDeviceBuffer {
        self.query.buffer
    }
    fn synchronize(&self) -> Result<()> {
        unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) }
    }
    pub fn clear_graph(&mut self) -> Result<()> {
        self.drain_chain()?;
        // Attempt every destruction even if one CUDA call reports an error.
        let mut failure = None;
        for graph in &mut self.graphs {
            for (g, _) in graph.drain(..) {
                if let Err(error) = unsafe { self.stream.library.cuda_graph_exec_destroy(g) } {
                    failure.get_or_insert(error);
                }
            }
        }
        match failure {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
    unsafe fn enqueue(
        &self,
        sink: Ds41rtDeviceBuffer,
        launches: &[RequestLaunch],
        selected: Option<Ds41rtDeviceBuffer>,
        batch: Option<&V41SparseBatch>,
    ) -> Result<()> {
        if let Some(batch) = batch {
            return unsafe { self.kernel.launch_batch(batch, self.query.buffer, sink,
                self.metadata.buffer, selected, self.output.buffer, self.descriptors.buffer,
                self.replay_begins.buffer, self.split_scratch.buffer, self.stream.raw) };
        }
        let mut offset = 0;
        for l in launches {
            unsafe {
                self.kernel.launch(
                    slice(self.query.buffer, offset * 65536, l.rows * 65536),
                    sink,
                    slice(self.metadata.buffer, offset * 80, l.rows * 80),
                    selected.map(|b| slice(b, offset * 2048, l.rows * 2048)),
                    &l.window,
                    l.source.as_ref(),
                    slice(self.output.buffer, offset * 65536, l.rows * 65536),
                    l.rows,
                    l.width,
                    // All request launches share this arena on one stream;
                    // each merge finishes before the next partial write.
                    (l.rows <= 16).then_some((
                        self.split_scratch.buffer,
                        if l.source.is_some() { 10 } else { 2 },
                    )),
                    self.stream.raw,
                )?;
            }
            offset += l.rows;
        }
        Ok(())
    }
    /// # Safety
    /// No external writes race query, cache/selection views or this wave. Sink
    /// is the matching layer's finite checkpoint sink on this device.
    pub unsafe fn execute_query<'s>(
        &'s mut self,
        query: &'s AttentionQueryOutput<'s>,
        sink: Ds41rtDeviceBuffer,
        requests: &'s [AttentionRequest<'s>],
        selection: Option<&'s IndexSelectionOutput<'s>>,
    ) -> Result<SparseAttentionOutput<'s>> {
        let binding = query.binding()?;
        let tokens = query.tokens()?;
        ensure!(
            query.rows == tokens.len()
                && query.rows <= self.capacity
                && query.rotated.device_id == self.query.buffer.device_id
                && requests
                    .iter()
                    .flat_map(|r| r.positions.iter().copied())
                    .eq(tokens.iter().copied()),
            "attention query token order or device differs"
        );
        if let Some(s) = selection {
            s.validate_query(binding)?;
        }
        let copied = unsafe {
            self.stream.library.copy_d2d_async(
                self.query.buffer, query.rotated, query.rotated.bytes, self.stream.raw,
            )
        };
        if let Err(error) = copied {
            self.synchronize()?;
            return Err(error);
        }
        let mut out = unsafe { self.execute(query.layer, sink, requests, selection)? };
        out.query = Some(binding);
        out.tokens = Some(tokens);
        Ok(out)
    }
    /// # Safety
    /// Input contains finite, rotated BF16 attention queries in request/token
    /// order, with producer writes drained. Sink is this layer's finite FP32
    /// checkpoint tensor on the initialized device. No external writes race
    /// inputs or cache owners. Output retains proposal/selection borrows until
    /// its consumers finish, preventing accepted commits or producer reuse.
    pub unsafe fn execute<'s>(
        &'s mut self,
        layer: usize,
        sink: Ds41rtDeviceBuffer,
        requests: &'s [AttentionRequest<'s>],
        selection: Option<&'s IndexSelectionOutput<'s>>,
    ) -> Result<SparseAttentionOutput<'s>> {
        let library = self.stream.library;
        let stream = self.stream.raw;
        let result = unsafe { self.execute_staged(layer, sink, requests, selection) };
        let drained = unsafe { library.cuda_stream_synchronize(stream) };
        let queued = result.and_then(|v| drained.map(|()| v))?;
        Ok(SparseAttentionOutput { query: None, tokens: None, values: queued.values,
            layer: queued.layer, rows: queued.rows, _inputs: PhantomData })
    }
    /// # Safety
    /// Same completed query/cache inputs as execute_query. The continuation may
    /// enqueue consumers only on the supplied stream, retaining every allocation
    /// until this method returns. Returned values must not publish GPU results
    /// until the successful drain performed here. No asynchronous suspension.
    pub unsafe fn execute_query_then<T>(
        &mut self, query: &AttentionQueryOutput<'_>, sink: Ds41rtDeviceBuffer,
        requests: &[AttentionRequest<'_>], selection: Option<&IndexSelectionOutput<'_>>,
        consume: impl FnOnce(QueuedSparseAttention, *mut c_void) -> Result<T>,
    ) -> Result<T> {
        unsafe { self.query_then(query, sink, requests, selection, consume) }
    }
    pub async fn wait_chain(&self) -> Result<()> { self.stream.wait().await }
    pub fn drain_chain(&mut self) -> Result<()> {
        let drained = self.synchronize(); self.cold = None; drained
    }
    pub fn chain_stream(&self) -> *mut c_void { self.stream.raw }
    /// # Safety
    /// Retain query/cache/selection buffers until warmup and downstream consumers
    /// complete or drain. No new submission may reuse this owner while pending.
    pub unsafe fn enqueue_query_prepared(&mut self, query: &AttentionQueryOutput<'_>,
        sink: Ds41rtDeviceBuffer, requests: &[AttentionRequest<'_>],
        selection: Option<&IndexSelectionOutput<'_>>, defer_warmup: bool,
        tail: Option<&mut dyn AttentionGraphTail>) -> Result<Option<QueuedSparseAttention>> {
        ensure!(self.cold.is_none(), "cold attention preparation already pending");
        let binding = query.binding()?;
        let tokens = query.tokens()?;
        ensure!(query.rows == tokens.len() && query.rows <= self.capacity
            && query.rotated.device_id == self.query.buffer.device_id
            && requests.iter().flat_map(|r| r.positions.iter().copied()).eq(tokens.iter().copied()),
            "attention query token order or device differs");
        if let Some(s) = selection { s.validate_query(binding)?; }
        let result = (|| unsafe {
            self.stream.library.copy_d2d_async(self.query.buffer, query.rotated, query.rotated.bytes, self.stream.raw)?;
            self.execute_staged_inner(query.layer, sink, requests, selection, defer_warmup, tail)
        })();
        if result.is_err() { self.drain_chain()?; }
        result
    }
    /// # Safety
    /// The caller retains the external inputs represented by the owned cold plan.
    /// Returned attention is queued, ready for consumers on chain_stream().
    pub async unsafe fn finish_prepare(&mut self, mut tail: Option<&mut dyn AttentionGraphTail>) -> Result<QueuedSparseAttention> {
        let plan = self.cold.take().context("cold attention preparation absent")?;
        #[cfg(test)]
        eprintln!("queued sparse warmup layer={} rows={} batched={}", plan.layer, plan.rows, plan.batch.is_some());
        self.stream.wait().await?;
        ensure!(plan.tail_identity == tail.as_ref().map(|t| t.identity()), "cold attention tail identity changed");
        if let Some(tail) = tail.as_deref_mut() { unsafe { tail.restore_warmup()?; } }
        self.warmed_kernels |= plan.needed;
        let graph = unsafe { self.capture_plan(&plan, &mut tail)? };
        self.graphs[plan.layer].push_back((graph, plan.fingerprint));
        let launched = unsafe { self.stream.library.cuda_graph_launch(graph, self.stream.raw) };
        if let Err(error) = launched { self.synchronize()?; return Err(error); }
        Ok(QueuedSparseAttention { values: slice(self.output.buffer, 0, plan.rows * 65536),
            layer: plan.layer, rows: plan.rows })
    }
    unsafe fn capture_plan(&self, plan: &ColdSparse, tail: &mut Option<&mut dyn AttentionGraphTail>) -> Result<*mut c_void> {
        unsafe { self.stream.library.cuda_graph_begin_capture(self.stream.raw)?; }
        let launched = (|| unsafe {
            self.enqueue(plan.sink, &plan.launches, plan.selected, plan.batch.as_ref())?;
            if let Some(tail) = tail.as_deref_mut() { tail.enqueue(&QueuedSparseAttention {
                values: slice(self.output.buffer, 0, plan.rows * 65536), layer: plan.layer, rows: plan.rows }, self.stream.raw)?; }
            Ok(())
        })();
        let captured = unsafe { self.stream.library.cuda_graph_end_capture(self.stream.raw) };
        match (launched, captured) {
            (Ok(()), Ok(graph)) => Ok(graph),
            (Err(error), Ok(graph)) => { unsafe { self.stream.library.cuda_graph_exec_destroy(graph)?; } Err(error) }
            (Err(error), Err(_)) | (Ok(()), Err(error)) => Err(error),
        }
    }
    /// # Safety
    /// Retain all query/cache/selection and tail owners through this direct drain.
    pub unsafe fn execute_query_graph(&mut self, query: &AttentionQueryOutput<'_>,
        sink: Ds41rtDeviceBuffer, requests: &[AttentionRequest<'_>], selection: Option<&IndexSelectionOutput<'_>>,
        tail: &mut dyn AttentionGraphTail) -> Result<()> {
        struct Drain<'a>(&'a NativeLibrary, *mut c_void, bool);
        impl Drop for Drain<'_> {
            fn drop(&mut self) { if !self.2 {
                if let Err(error) = unsafe { self.0.cuda_stream_synchronize(self.1) } {
                    tracing::error!(%error, "draining direct attention graph on unwind");
                }
            } }
        }
        let mut drain = Drain(self.stream.library, self.stream.raw, false);
        let queued = unsafe { self.enqueue_query_prepared(query, sink, requests, selection, false, Some(tail)) };
        let drained = self.synchronize(); drain.2 = true;
        queued.and_then(|value| value.context("direct graph unexpectedly deferred")).and(drained)
    }
    unsafe fn query_then<T>(&mut self, query: &AttentionQueryOutput<'_>, sink: Ds41rtDeviceBuffer,
        requests: &[AttentionRequest<'_>], selection: Option<&IndexSelectionOutput<'_>>,
        consume: impl FnOnce(QueuedSparseAttention, *mut c_void) -> Result<T>) -> Result<T> {
        let binding = query.binding()?;
        let tokens = query.tokens()?;
        ensure!(query.rows == tokens.len() && query.rows <= self.capacity
            && query.rotated.device_id == self.query.buffer.device_id
            && requests.iter().flat_map(|r| r.positions.iter().copied()).eq(tokens.iter().copied()),
            "attention query token order or device differs");
        if let Some(s) = selection { s.validate_query(binding)?; }
        let library = self.stream.library;
        let stream = self.stream.raw;
        // Also drain on unwinding before the caller can release consumer owners.
        struct Drain<'a>(&'a NativeLibrary, *mut c_void, bool);
        impl Drop for Drain<'_> {
            fn drop(&mut self) { if !self.2 {
                if let Err(error) = unsafe { self.0.cuda_stream_synchronize(self.1) } {
                    tracing::error!(%error, "draining attention continuation on unwind");
                }
            }}
        }
        let mut drain = Drain(library, stream, false);
        let result = (|| unsafe {
            library.copy_d2d_async(self.query.buffer, query.rotated, query.rotated.bytes, stream)?;
            let queued = self.execute_staged(query.layer, sink, requests, selection)?;
            consume(queued, stream)
        })();
        let drained = unsafe { library.cuda_stream_synchronize(stream) };
        drain.2 = true;
        result.and_then(|v| drained.map(|()| v))
    }
    unsafe fn execute_staged<'s>(
        &'s mut self,
        layer: usize,
        sink: Ds41rtDeviceBuffer,
        requests: &'s [AttentionRequest<'s>],
        selection: Option<&'s IndexSelectionOutput<'s>>,
    ) -> Result<QueuedSparseAttention> {
        unsafe { self.execute_staged_inner(layer, sink, requests, selection, false, None)? }
            .context("direct attention unexpectedly deferred")
    }
    unsafe fn execute_staged_inner(&mut self, layer: usize, sink: Ds41rtDeviceBuffer,
        requests: &[AttentionRequest<'_>], selection: Option<&IndexSelectionOutput<'_>>,
        defer_warmup: bool, mut tail: Option<&mut dyn AttentionGraphTail>) -> Result<Option<QueuedSparseAttention>> {
        ensure!(self.cold.is_none(), "attention warmup is pending");
        ensure!(
            layer < 40 && !requests.is_empty() && requests.len() <= 16,
            "invalid attention layer or request count"
        );
        let rows: usize = requests.iter().map(|r| r.positions.len()).sum();
        ensure!(
            rows > 0 && rows <= self.capacity,
            "attention rows exceed capacity"
        );
        ensure!(
            sink.bytes >= 256
                && !sink.ptr.is_null()
                && sink.device_id == self.query.buffer.device_id,
            "attention sink differs"
        );
        ensure!(
            selection.is_some() == (layer >= 2),
            "attention selection presence differs"
        );
        let source_layer = if layer >= 20 {
            20
        } else if layer >= 14 {
            14
        } else if layer >= 8 {
            8
        } else {
            2
        };
        let mut bindings = Vec::with_capacity(rows);
        let mut launches = Vec::with_capacity(requests.len());
        let mut metadata = Vec::with_capacity(rows * 10);
        let mut fingerprint = vec![layer, rows, sink.ptr as usize];
        for (i, r) in requests.iter().enumerate() {
            let w = r.window;
            ensure!(
                w.layer == layer && w.binding.same_pool(requests[0].window.binding),
                "attention window layer or owner differs"
            );
            ensure!(
                !r.positions.is_empty() && r.positions.windows(2).all(|p| p[0] < p[1]),
                "empty or unordered attention request"
            );
            ensure!(
                requests[..i].iter().all(|p| p.window.request != w.request),
                "duplicate attention request"
            );
            ensure!(
                r.source.is_some() == (layer >= 2),
                "attention source presence differs"
            );
            // Device metadata supplies each query's causal width on replay,
            // independent of the selected draft suffix and its row count.
            let width = 0;
            for &position in r.positions {
                let wm = w.metadata(position)?;
                metadata.extend(wm);
                if let Some(s) = r.source {
                    ensure!(
                        s.request_id() == w.request
                            && s.source_layer == source_layer
                            && s.first_token() == wm[0],
                        "attention source request, layer or committed position differs"
                    );
                    ensure!(
                        s.binding()
                            .same_pool(requests[0].source.context("source absent")?.binding()),
                        "attention source owner differs"
                    );
                    metadata.extend(s.metadata(position)?);
                    bindings.push((s.binding(), position));
                } else {
                    metadata.extend([0; 6]);
                }
            }
            let window = V41SparseWindow {
                values: w.cache.values,
                scales: w.cache.scales,
                proposals: w.values,
                proposal_scales: w.scales,
                end: w.cache.device_end,
                proposal_capacity: w.capacity,
                replay_begins: (w.cache.begin != 0).then(|| slice(self.replay_begins.buffer,
                    (metadata.len() / 10 - r.positions.len()) * 8, r.positions.len() * 8)),
            };
            let source = r.source.map(|s| V41SparseSource {
                values: s.kv_cache.values,
                scales: s.kv_cache.scales,
                proposals: s.kv_values,
                proposal_scales: s.kv_scales,
                pages: s.kv_cache.device_pages,
                end: s.kv_cache.device_rows,
                capacity: s.kv_cache.values.bytes / V41Kv::COMPRESSED_VALUE_BYTES,
                proposal_capacity: s.capacity,
                page_stride: s.kv_cache.device_pages.bytes / 4,
            });
            fingerprint.extend([r.positions.len(), width, window.proposal_capacity]);
            let mut buffers = vec![
                window.values,
                window.scales,
                window.proposals,
                window.proposal_scales,
                window.end,
            ];
            fingerprint.push(usize::from(window.replay_begins.is_some()));
            if let Some(bounds) = window.replay_begins { buffers.push(bounds); }
            if let Some(s) = &source {
                fingerprint.extend([s.capacity, s.proposal_capacity, s.page_stride]);
                buffers.extend([
                    s.values,
                    s.scales,
                    s.proposals,
                    s.proposal_scales,
                    s.pages,
                    s.end,
                ]);
            }
            for b in buffers {
                ensure!(
                    b.device_id == self.query.buffer.device_id && !b.ptr.is_null(),
                    "attention proposal device differs"
                );
                fingerprint.extend([b.ptr as usize, b.bytes]);
            }
            launches.push(RequestLaunch {
                window,
                source,
                rows: r.positions.len(),
                width,
            });
        }
        let selected = if let Some(s) = selection {
            s.validate_attention(layer, &bindings)?;
            ensure!(
                s.selected.device_id == self.query.buffer.device_id
                    && s.selected.bytes == rows * 2048,
                "attention selection device or size differs"
            );
            fingerprint.push(s.selected.ptr as usize);
            Some(s.selected)
        } else {
            None
        };
        // Only small multi-request split launches use descriptor batching. C1
        // and large prefill retain their existing arithmetic and dispatch.
        let batch = if self.kernel.supports_batch() && launches.len() > 1 && rows <= self.batch_rows
            && launches.iter().all(|launch| launch.rows <= 16) {
            let bindings: Vec<_> = launches.iter()
                .map(|launch| (&launch.window, launch.source.as_ref(), launch.rows)).collect();
            let batch = self.kernel.prepare_batch(self.query.buffer, sink, self.metadata.buffer,
                selected, &bindings, self.output.buffer, self.descriptors.buffer,
                self.replay_begins.buffer, self.split_scratch.buffer)?;
            self.descriptor_staging.bytes_mut()[..batch.bytes().len()].copy_from_slice(batch.bytes());
            unsafe { self.stream.library.copy_host_buffer_h2d_async(self.descriptors.buffer,
                self.descriptor_staging.buffer, batch.bytes().len(), self.stream.raw)?; }
            // The kernel reads current descriptors rather than capturing external
            // cache pointers or request row counts. Selection is lane-owned and
            // stable; still include its address to guard any future owner change.
            fingerprint = vec![usize::MAX, layer, rows, sink.ptr as usize,
                selected.map_or(0, |buffer| buffer.ptr as usize), batch.backend_key()];
            Some(batch)
        } else { None };
        for (i, m) in metadata.into_iter().enumerate() {
            self.staging.bytes_mut()[i * 8..i * 8 + 8].copy_from_slice(&m.to_ne_bytes());
        }
        unsafe {
            self.stream.library.copy_host_buffer_h2d_async(
                self.metadata.buffer, self.staging.buffer, rows * 80, self.stream.raw,
            )?;
        }
        if batch.is_some() || requests.iter().any(|r| r.window.cache.begin != 0) {
            let mut row = 0;
            for request in requests {
                for _ in request.positions {
                    self.replay_staging.bytes_mut()[row * 8..row * 8 + 8]
                        .copy_from_slice(&request.window.cache.begin.to_ne_bytes());
                    row += 1;
                }
            }
            unsafe {
                self.stream.library.copy_host_buffer_h2d_async(
                    self.replay_begins.buffer, self.replay_staging.buffer, rows * 8, self.stream.raw,
                )?;
            }
        }
        let queued = QueuedSparseAttention { values: slice(self.output.buffer, 0, rows * 65536), layer, rows };
        let tail_identity = tail.as_ref().map(|t| t.identity());
        if let Some(identity) = &tail_identity {
            fingerprint.push(usize::MAX - 1); fingerprint.extend(identity);
            unsafe { tail.as_deref_mut().unwrap().prepare(self.stream.raw)?; }
        }
        let cached = self.graphs[layer].iter().position(|(_, f)| f == &fingerprint);
        let graph = if let Some(index) = cached {
            // Move a used binding to the newest end of the bounded LRU.
            let entry = self.graphs[layer].remove(index).unwrap();
            let graph = entry.0;
            self.graphs[layer].push_back(entry);
            if let Some(tail) = tail.as_deref_mut() { unsafe { tail.replay_state()?; } }
            graph
        } else {
            tracing::debug!(target: "ds41rt::timing", layer, rows, batched = batch.is_some(), "sparse graph capture");
            if !defer_warmup { self.synchronize()?; }
            let limit = if batch.is_some() { self.batch_rows } else { self.graph_limit };
            if self.graphs[layer].len() >= limit {
                let (old, _) = self.graphs[layer].pop_front().unwrap();
                unsafe { self.stream.library.cuda_graph_exec_destroy(old)?; }
            }
            // Native dispatch has four row recipes per cache format: split,
            // unsplit single-group, two-group and four-group. Changing pointers
            // or launch dimensions does not require executing a warmed recipe
            // again before capture. Keep the fixed-policy path unchanged.
            let needed = if batch.is_some() { 1u16 << (8 + usize::from(selected.is_some())) } else { launches.iter().fold(0u16, |mask, launch| {
                let recipe = if launch.rows <= 16 { 0 } else if launch.rows < 128 { 1 }
                    else if launch.rows < 256 { 2 } else { 3 };
                mask | (1 << (recipe + 4 * usize::from(launch.source.is_some())))
            }) };
            let needs_warmup = tail.is_some() || (batch.is_none() && self.graph_limit == 1) || self.warmed_kernels & needed != needed;
            let plan = ColdSparse { layer, rows, sink, launches, selected, batch, fingerprint, needed, tail_identity };
            if needs_warmup {
                let launched = (|| unsafe {
                    self.enqueue(plan.sink, &plan.launches, plan.selected, plan.batch.as_ref())?;
                    if let Some(tail) = tail.as_deref_mut() { tail.enqueue(&queued, self.stream.raw)?; }
                    Ok(())
                })();
                if defer_warmup {
                    launched?;
                    self.cold = Some(plan);
                    return Ok(None);
                }
                launched.and(self.synchronize())?;
                if let Some(tail) = tail.as_deref_mut() { unsafe { tail.restore_warmup()?; } }
                self.warmed_kernels |= needed;
            }
            // Metadata uploads remain ordered on this stream; capture itself
            // contains no await. Previous uses of an evicted graph are complete.
            let graph = unsafe { self.capture_plan(&plan, &mut tail)? };
            self.graphs[layer].push_back((graph, plan.fingerprint));
            graph
        };
        let launched = unsafe {
            self.stream
                .library
                .cuda_graph_launch(graph, self.stream.raw)
        };
        launched?;
        Ok(Some(QueuedSparseAttention {
            values: slice(self.output.buffer, 0, rows * 65536), layer, rows,
        }))
    }
}
impl Drop for SparseAttentionWave<'_> {
    fn drop(&mut self) {
        if let Err(e) = self.clear_graph() {
            tracing::error!(%e,"draining sparse attention graph");
        }
    }
}

#[cfg(test)]
#[path = "v41_sparse_attention/prefix_tests.rs"]
mod prefix_tests;
