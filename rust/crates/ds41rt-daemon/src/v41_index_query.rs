//! Learned index-query projections and fused rotary/FP4 preparation on the RTX.
use crate::v41_attention_binding::QueryBinding;
use crate::v41_attention_query::AttentionQueryOutput;
use crate::v41_layer_graphs::LayerGraphs;
use crate::v41_memory::{DeviceAllocation, LoadStream};
use crate::v41_tensors::NativeRtxTensors;
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary, V41AttentionOps, V41Compressor, V41Fp8Plan};
use ds41rt_loader::OfficialV41Catalog;
use std::marker::PhantomData;

pub(crate) struct IndexQueryWeights<'a> {
    library: &'a NativeLibrary,
    layer: usize,
    names: [String; 3],
    tensors: NativeRtxTensors<'a>,
    scales: DeviceAllocation<'a>,
}
impl<'a> IndexQueryWeights<'a> {
    fn names(layer: usize) -> Result<[String; 3]> {
        ensure!(
            [2, 8, 14, 20, 24, 28, 32, 36].contains(&layer),
            "layer has no learned index query producer"
        );
        Ok([
            format!("layers.{layer}.attn.indexer.wq_b.weight"),
            format!("layers.{layer}.attn.indexer.wq_b.scale"),
            format!("layers.{layer}.attn.indexer.weights_proj.weight"),
        ])
    }
    pub fn device_bytes(
        library: &NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: usize,
    ) -> Result<usize> {
        let resident = NativeRtxTensors::plan(catalog, &Self::names(layer)?)?;
        ensure!(
            resident == 5_575_680,
            "unexpected native index query tensor sizes"
        );
        Ok(resident
            + library
                .v41_fp8_matrix_info(1, 1280, 4096)?
                .packed_weight_scale_bytes as usize)
    }
    pub fn load(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: usize,
        budget: usize,
        staging_bytes: usize,
    ) -> Result<Self> {
        ensure!(
            Self::device_bytes(library, catalog, layer)? <= budget,
            "index query weights exceed budget"
        );
        let names = Self::names(layer)?;
        let tensors = NativeRtxTensors::load(library, catalog, &names, budget, staging_bytes)?;
        let kernel = library.v41_fp8_matrix_kernel(1, 1280, 4096)?;
        let scales =
            DeviceAllocation::new(library, kernel.info().packed_weight_scale_bytes as usize)?;
        let stream = LoadStream {
            library,
            raw: library.cuda_stream_create()?,
        };
        let launched =
            unsafe { kernel.pack_scales(tensors.get(&names[1])?, scales.buffer, stream.raw) };
        let drained = unsafe { library.cuda_stream_synchronize(stream.raw) };
        launched.and(drained)?;
        Ok(Self {
            library,
            layer,
            names,
            tensors,
            scales,
        })
    }
    pub fn wave(&self, capacity: u32, budget: usize) -> Result<IndexQueryWave<'_, 'a>> {
        ensure!(
            IndexQueryWave::device_bytes(self.library, capacity)? <= budget,
            "index query wave exceeds budget"
        );
        let fp8 = self.library.v41_fp8_matrix_plan(capacity, 1280, 4096)?;
        let scratch = DeviceAllocation::new(self.library, fp8.info().scratch_bytes as usize)?;
        let workspace = DeviceAllocation::new(self.library, V41Compressor::WORKSPACE_BYTES)?;
        ensure!(
            workspace.buffer.device_id == self.tensors.get(&self.names[0])?.device_id,
            "index query weight device differs"
        );
        let dense = unsafe { self.library.v41_compressor(workspace.buffer)? };
        let rows = capacity as usize;
        let value = IndexQueryWave {
            stream: LoadStream {
                library: self.library,
                raw: self.library.cuda_stream_create()?,
            },
            fp8,
            dense,
            _workspace: workspace,
            scratch,
            alpha: DeviceAllocation::new(self.library, 4)?,
            norm: self.library.v41_attention_ops()?,
            weights: self,
            qr: DeviceAllocation::new(self.library, rows * 2560)?,
            hidden: DeviceAllocation::new(self.library, rows * 10240)?,
            positions: DeviceAllocation::new(self.library, rows * 8)?,
            frequencies: DeviceAllocation::new(self.library, rows * 256)?,
            projected: DeviceAllocation::new(self.library, rows * 8192)?,
            head_projected: DeviceAllocation::new(self.library, rows * 64)?,
            packed: DeviceAllocation::new(self.library, rows * 2048)?,
            scales: DeviceAllocation::new(self.library, rows * 128)?,
            head_weights: DeviceAllocation::new(self.library, rows * 64)?,
            capacity,
            graphs: LayerGraphs::new(self.library),
            ready: None,
            pending: None,
            origin: None,
            tokens: Vec::new(),
        };
        let launched = unsafe {
            value
                .fp8
                .initialize_scratch(value.scratch.buffer, value.alpha.buffer, value.stream.raw)
        };
        launched.and(value.synchronize())?;
        Ok(value)
    }
}
pub(crate) struct IndexQueryOutput<'a> {
    origin: Option<QueryBinding>,
    tokens: &'a [u64],
    pub layer: usize,
    pub packed: Ds41rtDeviceBuffer,
    pub scales: Ds41rtDeviceBuffer,
    pub head_weights: Ds41rtDeviceBuffer,
    _owner: PhantomData<&'a ()>,
}
impl IndexQueryOutput<'_> {
    pub fn origin(&self) -> Option<QueryBinding> {
        self.origin
    }
    pub fn bound_tokens(&self) -> Result<&[u64]> {
        ensure!(self.origin.is_some(), "index query has no origin");
        Ok(self.tokens)
    }
}
pub(crate) struct IndexQueryWave<'w, 'a> {
    stream: LoadStream<'a>,
    fp8: V41Fp8Plan<'a>,
    dense: V41Compressor<'a>,
    _workspace: DeviceAllocation<'a>,
    scratch: DeviceAllocation<'a>,
    alpha: DeviceAllocation<'a>,
    norm: V41AttentionOps<'a>,
    weights: &'w IndexQueryWeights<'a>,
    qr: DeviceAllocation<'a>,
    hidden: DeviceAllocation<'a>,
    positions: DeviceAllocation<'a>,
    frequencies: DeviceAllocation<'a>,
    projected: DeviceAllocation<'a>,
    head_projected: DeviceAllocation<'a>,
    packed: DeviceAllocation<'a>,
    scales: DeviceAllocation<'a>,
    head_weights: DeviceAllocation<'a>,
    capacity: u32,
    graphs: LayerGraphs<'w, 'a, IndexQueryWeights<'a>>,
    ready: Option<u32>,
    pending: Option<(u32, QueryBinding, bool)>,
    origin: Option<QueryBinding>,
    tokens: Vec<u64>,
}
impl<'w, 'a> IndexQueryWave<'w, 'a> {
    /// Retain captured weight owners while reusing this lane's fixed buffers.
    pub fn rebind(&mut self, weights: &'w IndexQueryWeights<'a>) -> Result<()> {
        ensure!(self.pending.is_none(), "index query pending");
        self.ready = None;
        self.origin = None;
        self.tokens.clear();
        ensure!(
            std::ptr::eq(self.stream.library, weights.library)
                && weights.tensors.get(&weights.names[0])?.device_id
                    == self.hidden.buffer.device_id,
            "index query rebound library or device differs"
        );
        self.stream.require_complete()?;
        self.weights = weights;
        Ok(())
    }
}
impl IndexQueryWave<'_, '_> {
    pub fn device_bytes(library: &NativeLibrary, capacity: u32) -> Result<usize> {
        let info = library.v41_fp8_matrix_plan_info(capacity, 1280, 4096)?;
        Ok(V41Compressor::WORKSPACE_BYTES
            + info.scratch_bytes as usize
            + 4
            + capacity as usize * 23560)
    }
    /// Normalized BF16 query-rank input [capacity,1280], shared with target attention.
    pub fn query_rank(&self) -> Ds41rtDeviceBuffer {
        self.qr.buffer
    }
    /// BF16 attention-input hidden states [capacity,5120], in the same row order.
    pub fn hidden(&self) -> Ds41rtDeviceBuffer {
        self.hidden.buffer
    }
    /// U64 token positions below the official context limit, in the same row order.
    pub fn positions(&self) -> Ds41rtDeviceBuffer {
        self.positions.buffer
    }
    fn synchronize(&self) -> Result<()> {
        unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) }
    }
    fn validate(&mut self, rows: u32) -> Result<()> {
        ensure!(self.pending.is_none(), "index query pending");
        self.ready = None;
        self.origin = None;
        self.tokens.clear();
        ensure!(
            rows > 0 && rows <= self.capacity,
            "index query rows exceed capacity"
        );
        Ok(())
    }
    unsafe fn enqueue(&self, rows: u32) -> Result<()> {
        unsafe {
            self.norm.backbone_frequencies(
                self.positions.buffer,
                self.frequencies.buffer,
                rows,
                self.weights.layer as u32,
                self.stream.raw,
            )?;
            self.fp8.launch(
                self.qr.buffer,
                self.weights.tensors.get(&self.weights.names[0])?,
                self.weights.scales.buffer,
                self.scratch.buffer,
                self.alpha.buffer,
                self.projected.buffer,
                rows,
                self.stream.raw,
            )?;
            self.dense.weights_project(
                self.hidden.buffer,
                self.weights.tensors.get(&self.weights.names[2])?,
                self.head_projected.buffer,
                rows as usize,
                self.stream.raw,
            )?;
            self.dense.query_prepare(
                self.projected.buffer,
                self.frequencies.buffer,
                self.head_projected.buffer,
                self.packed.buffer,
                self.scales.buffer,
                self.head_weights.buffer,
                rows as usize,
                self.stream.raw,
            )?;
        }
        Ok(())
    }
    /// # Safety
    /// Inputs are finite, initialized in matching row order on this device, with
    /// producer writes complete. Positions are below 1048576. No writes may race
    /// this wave; callers bind outputs to the matching request/query snapshot.
    pub unsafe fn execute(&mut self, rows: u32) -> Result<IndexQueryOutput<'_>> {
        self.validate(rows)?;
        let launched = unsafe { self.enqueue(rows) };
        launched.and(self.synchronize())?;
        self.ready = Some(rows);
        self.output()
    }
    /// # Safety
    /// Same inputs as execute. Warmup is drained but not published after capture.
    pub unsafe fn capture(&mut self, rows: u32) -> Result<()> {
        self.ready = None;
        self.origin = None;
        self.tokens.clear();
        ensure!(
            self.graphs.get_shape(self.weights.layer, self.weights, rows).is_none(),
            "index query graph already captured"
        );
        unsafe {
            self.execute(rows)?;
        }
        self.ready = None;
        self.origin = None;
        self.tokens.clear();
        unsafe { self.capture_ready(rows) }
    }
    unsafe fn capture_ready(&mut self, rows: u32) -> Result<()> {
        unsafe {
            self.stream
                .library
                .cuda_graph_begin_capture(self.stream.raw)?;
        }
        let launched = unsafe { self.enqueue(rows) };
        let captured = unsafe { self.stream.library.cuda_graph_end_capture(self.stream.raw) };
        match (launched, captured) {
            (Ok(()), Ok(graph)) => {
                if let Err(error) = unsafe {
                    self.graphs
                        .insert(self.weights.layer, self.weights, rows, graph)
                } {
                    unsafe {
                        self.stream.library.cuda_graph_exec_destroy(graph)?;
                    }
                    return Err(error);
                }
                Ok(())
            }
            (Err(e), Ok(graph)) => {
                unsafe {
                    self.stream.library.cuda_graph_exec_destroy(graph)?;
                }
                Err(e)
            }
            (Err(e), Err(_)) | (Ok(()), Err(e)) => Err(e),
        }
    }
    /// # Safety
    /// Same matching initialized inputs as execute. Captured live row count is fixed.
    pub unsafe fn replay(&mut self, rows: u32) -> Result<IndexQueryOutput<'_>> {
        self.validate(rows)?;
        let (graph, count) = self
            .graphs
            .get_shape(self.weights.layer, self.weights, rows)
            .context("index query graph missing")?;
        ensure!(count == rows, "index query capture row count differs");
        let launched = unsafe {
            self.stream
                .library
                .cuda_graph_launch(graph, self.stream.raw)
        };
        launched.and(self.synchronize())?;
        self.ready = Some(rows);
        self.output()
    }
    /// # Safety
    /// No external writes race the completed query or this index wave.
    pub unsafe fn execute_attention(
        &mut self,
        query: &AttentionQueryOutput<'_>,
    ) -> Result<IndexQueryOutput<'_>> {
        self.ready = None;
        self.origin = None;
        self.tokens.clear();
        let binding = query.binding()?;
        let tokens = query.tokens()?;
        ensure!(
            binding.layer() == self.weights.layer
                && query.layer == self.weights.layer
                && query.rows == tokens.len()
                && query.rows <= self.capacity as usize
                && query.hidden.device_id == self.hidden.buffer.device_id,
            "index query origin differs"
        );
        for (dst, src) in [
            (self.qr.buffer, query.normalized_rank),
            (self.hidden.buffer, query.hidden),
            (self.positions.buffer, query.positions),
        ] {
            self.stream.library.copy_d2d(dst, src, src.bytes)?;
        }
        let rows = query.rows as u32;
        if self
            .graphs
            .get_shape(self.weights.layer, self.weights, rows)
            .is_none()
        {
            unsafe {
                self.capture(rows)?;
            }
        }
        unsafe {
            self.replay(rows)?;
        }
        self.origin = Some(binding);
        self.tokens.extend_from_slice(tokens);
        self.output()
    }
    /// # Safety
    /// Retain the main query and this wave until polling completes or abort drains.
    pub unsafe fn enqueue_attention(&mut self, query: &AttentionQueryOutput<'_>) -> Result<()> {
        self.validate(query.rows as u32)?;
        let binding = query.binding()?;
        let tokens = query.tokens()?;
        ensure!(query.rows <= self.capacity as usize && query.rows == tokens.len()
            && binding.layer() == self.weights.layer && query.layer == self.weights.layer
            && query.hidden.device_id == self.hidden.buffer.device_id, "queued index origin differs");
        self.tokens.extend_from_slice(tokens);
        let rows = query.rows as u32;
        let graph = self.graphs.get_shape(self.weights.layer, self.weights, rows);
        self.pending = Some((rows, binding, graph.is_none()));
        let result = (|| -> Result<()> { unsafe {
            crate::v41_memory::chain::join(self.stream.library, self.stream.raw)?;
            for (dst, src) in [(self.qr.buffer, query.normalized_rank),
                (self.hidden.buffer, query.hidden), (self.positions.buffer, query.positions)] {
                self.stream.library.copy_d2d_async(dst, src, src.bytes, self.stream.raw)?;
            }
            if let Some((graph, _)) = graph { self.stream.library.cuda_graph_launch(graph, self.stream.raw) }
            else { self.enqueue(rows) }
        }})();
        if result.is_err() { self.abort_pending()?; }
        result
    }
    pub fn poll_pending(&mut self) -> Result<bool> {
        let result = (|| -> Result<bool> {
            let (rows, binding, capture) = self.pending.context("no pending index query")?;
            let chained = crate::v41_memory::chain::active();
            if !chained && !unsafe { self.stream.library.cuda_stream_query(self.stream.raw)? } { return Ok(false); }
            if capture {
                unsafe { self.capture_ready(rows)?; }
                let graph = self.graphs.get_shape(self.weights.layer, self.weights, rows)
                    .context("queued index graph missing")?.0;
                self.pending.as_mut().unwrap().2 = false;
                unsafe { self.stream.library.cuda_graph_launch(graph, self.stream.raw)?; }
                if !chained { return Ok(false); }
            }
            if chained { unsafe { crate::v41_memory::chain::finish(self.stream.library, self.stream.raw)?; } }
            self.pending = None;
            self.ready = Some(rows);
            self.origin = Some(binding);
            Ok(true)
        })();
        if result.is_err() { self.abort_pending()?; }
        result
    }
    pub fn abort_pending(&mut self) -> Result<()> {
        if self.pending.is_some() { self.synchronize()?; }
        self.pending = None; self.ready = None; self.origin = None; self.tokens.clear();
        Ok(())
    }
    pub fn output(&self) -> Result<IndexQueryOutput<'_>> {
        let rows = self.ready.context("index query output unpublished")? as usize;
        let sized = |mut b: Ds41rtDeviceBuffer, n: usize| {
            b.bytes = rows * n;
            b
        };
        Ok(IndexQueryOutput {
            origin: self.origin,
            tokens: &self.tokens,
            layer: self.weights.layer,
            packed: sized(self.packed.buffer, 2048),
            scales: sized(self.scales.buffer, 128),
            head_weights: sized(self.head_weights.buffer, 64),
            _owner: PhantomData,
        })
    }
    pub fn enable_small_graph_shapes(&mut self) { self.graphs.enable_small_shapes(); }
    pub fn clear_graph(&mut self) -> Result<()> {
        self.ready = None;
        self.origin = None;
        self.tokens.clear();
        self.synchronize()?;
        unsafe { self.graphs.remove(self.weights.layer) }
    }
}
impl Drop for IndexQueryWave<'_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self
            .synchronize()
            .and_then(|_| unsafe { self.graphs.clear() })
        {
            tracing::error!(%error,"draining index query graph");
        }
    }
}

#[cfg(test)]
mod reuse_tests {
    use super::*;
    fn download(library: &NativeLibrary, buffer: Ds41rtDeviceBuffer) -> Result<Vec<u8>> {
        let mut bytes = vec![0; buffer.bytes];
        library.copy_d2h(&mut bytes, buffer)?;
        Ok(bytes)
    }
    #[test]
    fn real_index_queries_rebind_like_fresh_owners() -> Result<()> {
        let Some(path) = std::env::var_os("DS41RT_INDEX_REUSE_LIBRARY") else {
            eprintln!("skip index reuse GPU test: DS41RT_INDEX_REUSE_LIBRARY unset");
            return Ok(());
        };
        let model = std::env::var_os("DS41RT_INDEX_REUSE_MODEL")
            .context("DS41RT_INDEX_REUSE_MODEL required")?;
        let library = unsafe { NativeLibrary::load(path)? };
        let catalog = ds41rt_loader::read_official_v41_catalog(
            ds41rt_loader::OFFICIAL_V41_MODEL_ID,
            std::path::Path::new(&model),
        )?;
        let weights = [2, 8, 14, 20, 24, 28, 32, 36]
            .into_iter()
            .map(|layer| IndexQueryWeights::load(&library, &catalog, layer, 5_739_520, 1024 * 1024))
            .collect::<Result<Vec<_>>>()?;
        let budget = IndexQueryWave::device_bytes(&library, 4096)?;
        let mut reused = weights[0].wave(4096, budget)?;
        let pointers = [
            reused.query_rank().ptr,
            reused.hidden().ptr,
            reused.positions().ptr,
        ];
        let mut comparisons = 0;
        for rows in [1u32, 80, 4096] {
            for cycle in 0..2usize {
                for (index, weight) in weights.iter().enumerate() {
                    let mut fresh = weight.wave(4096, budget)?;
                    reused.rebind(weight)?;
                    assert!(reused.output().is_err());
                    assert_eq!(
                        pointers,
                        [
                            reused.query_rank().ptr,
                            reused.hidden().ptr,
                            reused.positions().ptr
                        ]
                    );
                    for (a, b, width) in [
                        (fresh.query_rank(), reused.query_rank(), 1280),
                        (fresh.hidden(), reused.hidden(), 5120),
                    ] {
                        let bytes = (0..rows as usize * width)
                            .flat_map(|i| {
                                let x = (((i * 17 + index * 31 + cycle * 13) % 257) as f32 - 128.0)
                                    / 256.0;
                                ((x.to_bits() >> 16) as u16).to_ne_bytes()
                            })
                            .collect::<Vec<_>>();
                        library.copy_h2d(a, &bytes)?;
                        library.copy_h2d(b, &bytes)?;
                    }
                    let positions = (0..rows as u64)
                        .flat_map(|i| (i + 8192 * cycle as u64).to_ne_bytes())
                        .collect::<Vec<_>>();
                    library.copy_h2d(fresh.positions(), &positions)?;
                    library.copy_h2d(reused.positions(), &positions)?;
                    let previous = reused.graphs.get(weight.layer, weight);
                    if previous.is_none_or(|(_, n)| n != rows) {
                        reused.clear_graph()?;
                        unsafe {
                            reused.capture(rows)?;
                        }
                    }
                    let handle = reused.graphs.get(weight.layer, weight).unwrap().0;
                    if cycle == 1 {
                        assert_eq!(Some((handle, rows)), previous);
                    }
                    let expected = unsafe { fresh.execute(rows)? };
                    let actual = unsafe { reused.replay(rows)? };
                    for (a, b) in [
                        (expected.packed, actual.packed),
                        (expected.scales, actual.scales),
                        (expected.head_weights, actual.head_weights),
                    ] {
                        assert_eq!(
                            download(&library, a)?,
                            download(&library, b)?,
                            "index layer={} rows={rows} cycle={cycle}",
                            weight.layer
                        );
                    }
                    let head = download(&library, actual.head_weights)?;
                    assert!(head
                        .chunks_exact(4)
                        .all(|b| f32::from_ne_bytes(b.try_into().unwrap()).is_finite()));
                    comparisons += 1;
                }
            }
        }
        assert!(unsafe { reused.replay(0) }.is_err());
        assert!(reused.output().is_err());
        reused.rebind(&weights[0])?;
        unsafe {
            reused.replay(4096)?;
        }
        assert_eq!(comparisons, 48);
        eprintln!("PASS 48 real index-query fresh/rebound comparisons, all eight layers, rows 1/80/4096, changed inputs, cached handles and invalid-row recovery");
        Ok(())
    }
}
