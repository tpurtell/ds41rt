//! Cache producers and the complete per-layer backbone execution handoff.
use crate::v41_backbone_cache::{BackboneCache, CacheBatch, CacheStage, CachePlacement};
use crate::v41_memory::device::{Device, DeviceOwner};
mod placement;
mod distributed;
pub(crate) use distributed::{DistributedExecution, PendingDistributedProduction};
pub(crate) use placement::PlacedProducerWaves;
use crate::v41_backbone_lane::{BackboneLane, LaneFfn, PendingLaneFfn};
use crate::v41_backbone_router::ExpertRow;
use crate::v41_compressor::{CompressorWave, CompressorWeights};
use crate::v41_experts::coordinator::{NativeTp4Wave, NativeFfnOutput};
use crate::v41_index_lane::IndexLane;
use crate::v41_tensors::NativeRtxTensors;
use crate::v41_window::{WindowWave, WindowWeights};
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::NativeLibrary;
use ds41rt_loader::OfficialV41Catalog;
const SOURCES: [usize; 4] = [2, 8, 14, 20];
const INDEX: [usize; 8] = [2, 8, 14, 20, 24, 28, 32, 36];

enum PreparedFfn<'l, 'w, 'a> {
    Ready(LaneFfn<'l, 'w, 'a>),
    Pending(PendingLaneFfn<'l, 'w, 'a>),
}
/// Owns a lane borrow, but no cache-bank or index borrow. The unsafe queued
/// preparation contract retains external producers through attention completion.
pub(crate) struct PreparedLayer<'l, 'w, 'a> {
    ffn: PreparedFfn<'l, 'w, 'a>,
    rows: Vec<ExpertRow>,
    batch: u64,
    layer: usize,
    started: std::time::Instant,
    produced_us: u64,
    indexed_us: u64,
    attended_us: u64,
}
pub(crate) struct CompletedLayer<'t> {
    result: NativeFfnOutput<'t>,
    batch: u64,
    layer: usize,
    rows: usize,
    started: std::time::Instant,
    produced_us: u64,
    indexed_us: u64,
    attended_us: u64,
    experts_us: u64,
    routed_backend: &'static str,
    shared_tp: usize,
}
impl CompletedLayer<'_> {
    /// Opt-in calibration uses routes already captured by the lane. These are
    /// elapsed stage times (including scheduling/transport), not GPU kernel times.
    fn log_cost(&self, lane: &BackboneLane<'_, '_>) {
        if !tracing::enabled!(target: "ds41rt::cost_model", tracing::Level::DEBUG) { return; }
        // Disabling capture retains allocated route buffers. Their old row
        // count can match a subsequent prefill; those are not its routes.
        if !lane.route_capture_enabled() { return; }
        let captured = lane.captured_routes();
        let Some(routes) = captured.get(self.layer).filter(|r| r.len() == self.rows) else { return; };
        let mut counts = [0usize; 384];
        for &expert in routes.iter().flatten() {
            let Some(count) = counts.get_mut(expert as usize) else { return; };
            *count += 1;
        }
        let total_us = self.started.elapsed().as_micros() as u64;
        tracing::debug!(target: "ds41rt::cost_model", batch=self.batch, layer=self.layer,
            rows=self.rows, routed_backend=self.routed_backend, shared_tp=self.shared_tp,
            distinct_experts=counts.iter().filter(|&&n| n != 0).count(),
            expert_groups_16=counts.iter().map(|&n| n.div_ceil(16)).sum::<usize>(),
            produced_us=self.produced_us, index_us=self.indexed_us-self.produced_us,
            attention_us=self.attended_us-self.indexed_us,
            experts_us=self.experts_us-self.attended_us,
            finish_us=total_us-self.experts_us, total_us, "verification layer cost");
    }
}
impl PreparedLayer<'_, '_, '_> {
    #[cfg(test)]
    pub async unsafe fn trace_ffn_input(mut self, library: &ds41rt_ffi::NativeLibrary,
        destination: ds41rt_ffi::Ds41rtDeviceBuffer, stream: *mut std::ffi::c_void)
        -> Result<(Self, (u64, usize, usize))> {
        let ffn = match self.ffn {
            PreparedFfn::Ready(ffn) => ffn,
            PreparedFfn::Pending(pending) => pending.complete().await?,
        };
        let position = ffn.input.tokens[0];
        let bytes = ffn.input.tokens.len() * 10240;
        ensure!(position as usize + ffn.input.tokens.len() <= 16, "FFN trace extent exceeded");
        let offset = (self.layer * 16 + position as usize) * 40976;
        let mut input = ffn.input.values;
        input.bytes = bytes;
        ensure!(offset + bytes <= destination.bytes, "FFN trace buffer exceeded");
        let destination = ds41rt_ffi::Ds41rtDeviceBuffer {
            ptr: unsafe { destination.ptr.cast::<u8>().add(offset).cast() }, bytes, ..destination
        };
        unsafe { library.copy_d2d_async(destination, input, bytes, stream)?; }
        self.ffn = PreparedFfn::Ready(ffn);
        let record = (position, self.layer, bytes);
        Ok((self, record))
    }
    #[cfg(test)]
    async unsafe fn check_queued_ffn(mut self, image_mask: &[u8],
        local: Option<&mut crate::v41_experts::local::LocalExpertWave<'_>>) -> Result<Self> {
        let mut ffn = match self.ffn {
            PreparedFfn::Ready(ffn) => ffn,
            PreparedFfn::Pending(pending) => pending.complete().await?,
        };
        unsafe { ffn.check_queued_components(image_mask, local).await?; }
        self.ffn = PreparedFfn::Ready(ffn);
        Ok(self)
    }
    /// # Safety
    /// The modality mask and placement describe this prepared batch. Poll on
    /// the CUDA owner; no external writes may race the borrowed lane.
    pub async unsafe fn execute<'t>(self, transport: &'t mut NativeTp4Wave<'_>,
        placement: u64, image_mask: &[u8]) -> Result<CompletedLayer<'t>> {
        let (mut ffn, attended_us) = match self.ffn {
            PreparedFfn::Ready(ffn) => (ffn, self.attended_us),
            PreparedFfn::Pending(pending) => {
                let ffn = pending.complete().await?;
                (ffn, self.started.elapsed().as_micros() as u64)
            }
        };
        let routed_backend = if transport.has_tp2_layer(self.layer) { "rtx_tp2" }
            else if transport.has_local_layer(self.layer) { "rtx_local" } else { "spark_tp4" };
        let shared_tp = if transport.has_tp2_shared_layer(self.layer) { 2 } else { 1 };
        let result = unsafe { ffn.execute_tp4(transport, placement, image_mask, &self.rows).await? };
        Ok(CompletedLayer { result, batch: self.batch, layer: self.layer, rows: self.rows.len(),
            routed_backend, shared_tp,
            started: self.started, produced_us: self.produced_us, indexed_us: self.indexed_us,
            attended_us, experts_us: self.started.elapsed().as_micros() as u64 })
    }
}

enum LayerCache<'b, 'a> {
    Ordinary(&'b BackboneCache<'a>),
    Encoder(&'b mut BackboneCache<'a>),
}
impl<'a> LayerCache<'_, 'a> {
    fn bank(&self) -> &BackboneCache<'a> {
        match self { Self::Ordinary(bank) => bank, Self::Encoder(bank) => bank }
    }
}

pub(crate) struct CacheProducerWeights<'a> {
    library: &'a NativeLibrary,
    windows: Vec<DeviceOwner<'a, WindowWeights<'a>>>,
    sources: Vec<DeviceOwner<'a, CompressorWeights<'a>>>,
    _sinks: Vec<DeviceOwner<'a, NativeRtxTensors<'a>>>,
    sink_views: [ds41rt_ffi::Ds41rtDeviceBuffer; 40],
    placement: Option<CachePlacement>,
}
impl<'a> CacheProducerWeights<'a> {
    fn sinks() -> Vec<String> {
        (0..40)
            .map(|layer| format!("layers.{layer}.attn.attn_sink"))
            .collect()
    }
    pub fn device_bytes(library: &NativeLibrary, catalog: &OfficialV41Catalog) -> Result<usize> {
        let mut total = NativeRtxTensors::plan(catalog, &Self::sinks())?;
        ensure!(total == 40 * 256, "unexpected backbone attention sink size");
        for layer in 0..40 {
            total = total
                .checked_add(WindowWeights::device_bytes(library, catalog, layer)?)
                .context("cache producer weights overflow")?;
        }
        for layer in SOURCES {
            total = total
                .checked_add(CompressorWeights::device_bytes(catalog, layer)?)
                .context("cache producer weights overflow")?;
        }
        Ok(total)
    }
    pub fn load(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        budget: usize,
        staging: usize,
    ) -> Result<Self> {
        ensure!(
            Self::device_bytes(library, catalog)? <= budget,
            "cache producer weights exceed budget"
        );
        Self::load_placed(library, catalog, staging, None)
    }

}

#[derive(Default)]
struct PassProgress {
    stage: CacheStage,
    source_ready: bool,
    batch: Option<u64>,
    next: usize,
    invalid: bool,
}
impl PassProgress {
    fn begin(&mut self, batch: u64, layer: usize) -> Result<()> {
        let valid = !std::mem::replace(&mut self.invalid, true);
        ensure!(
            valid && layer == self.next && self.stage.windows().contains(&layer) && self.batch.is_none_or(|id| id == batch),
            "backbone pass batch or layer differs; restart required"
        );
        self.batch = Some(batch);
        Ok(())
    }
    fn for_stage(stage: CacheStage) -> Self {
        Self { stage, next: stage.windows().start, ..Self::default() }
    }
    fn begin_decoder_source(&mut self, batch: u64) -> Result<()> {
        let valid = !std::mem::replace(&mut self.invalid, true);
        ensure!(valid && self.stage == CacheStage::Encoder && self.next == 20
            && self.batch == Some(batch) && !self.source_ready,
            "decoder source requires this batch's complete encoder pass");
        Ok(())
    }
    fn finish_decoder_source(&mut self) {
        self.source_ready = true;
        self.invalid = false;
    }
    fn finish(&mut self) {
        self.next += 1;
        self.invalid = false;
    }
    fn commit(&mut self, batch: u64) -> Result<()> {
        let valid = !std::mem::replace(&mut self.invalid, true);
        ensure!(
            valid && self.next == self.stage.windows().end && self.batch == Some(batch)
                && (self.stage != CacheStage::Encoder || self.source_ready),
            "cache commit requires this batch's complete execution phase"
        );
        Ok(())
    }
}

/// Every layer retains its private proposal until accepted-prefix commit. Two
/// alternating passes need independent producer storage and progress owners.
#[derive(Clone, Copy)]
struct Produced {
    batch: u64,
    layer: usize,
    started: std::time::Instant,
    producer_us: u64,
    indexed: bool,
}
pub(crate) struct BackboneExecution<'w, 'a> {
    weights: &'w CacheProducerWeights<'a>,
    windows: Vec<WindowWave<'w, 'a>>,
    sources: Vec<CompressorWave<'w, 'a>>,
    progress: PassProgress,
    queued_production: Option<Produced>,
    window_batch: Option<WindowBatch<'a>>,
}

/// All 40 window layers' accepted-row writes as one pinned upload and one
/// launch on a dedicated stream, replacing forty upload/launch/drain steps.
/// Each window records this stream as its pending commit's completion.
struct WindowBatch<'a> {
    library: &'a NativeLibrary,
    stream: crate::v41_memory::LoadStream<'a>,
    staging: crate::v41_memory::HostAllocation<'a>,
    table: crate::v41_memory::DeviceAllocation<'a>,
    kernel: ds41rt_ffi::V41KvStoreLayers<'a>,
    capacity: usize,
}
impl<'a> WindowBatch<'a> {
    const LAYERS: usize = 40;
    const CHUNKS: usize = 16;
    fn bytes(capacity: usize) -> usize {
        Self::LAYERS * (std::mem::size_of::<ds41rt_ffi::V41KvStoreLayer>() + capacity * 8 + Self::CHUNKS * 16)
    }
    fn new(library: &'a NativeLibrary, capacity: usize) -> Result<Self> {
        let bytes = Self::bytes(capacity);
        Ok(Self {
            library,
            stream: crate::v41_memory::LoadStream { library, raw: library.cuda_stream_create()? },
            staging: crate::v41_memory::HostAllocation::new(library, bytes)?,
            table: crate::v41_memory::DeviceAllocation::new(library, bytes)?,
            kernel: library.v41_kv_store_layers()?,
            capacity,
        })
    }
    /// # Safety
    /// Every staged window keeps its buffers and cache state alive until this
    /// stream drains; the previous batch on this owner has completed.
    unsafe fn enqueue(&mut self, layers: &[crate::v41_window::BatchedWindowCommit]) -> Result<()> {
        let rows = layers.first().context("empty window batch")?.destinations.len();
        let chunks = layers[0].ends.len();
        ensure!(layers.len() <= Self::LAYERS && rows <= self.capacity && (1..=Self::CHUNKS).contains(&chunks)
            && layers.iter().all(|l| l.destinations.len() == rows && l.ends.len() == chunks
                && l.device == self.table.buffer.device_id),
            "window batch shape differs");
        let entry = std::mem::size_of::<ds41rt_ffi::V41KvStoreLayer>();
        let base = self.table.buffer.ptr as u64;
        let mut offset = layers.len() * entry;
        let mut entries = Vec::with_capacity(layers.len());
        let staging = self.staging.bytes_mut();
        for layer in layers {
            let mut value = layer.layer;
            value.destinations = base + offset as u64;
            for (i, d) in layer.destinations.iter().enumerate() {
                staging[offset + i * 8..offset + i * 8 + 8].copy_from_slice(&d.to_ne_bytes());
            }
            offset += rows * 8;
            value.end_pairs = base + offset as u64;
            for (i, (slot, end)) in layer.ends.iter().enumerate() {
                staging[offset + i * 16..offset + i * 16 + 8].copy_from_slice(&slot.to_ne_bytes());
                staging[offset + i * 16 + 8..offset + i * 16 + 16].copy_from_slice(&end.to_ne_bytes());
            }
            offset += chunks * 16;
            entries.push(value);
        }
        for (i, value) in entries.iter().enumerate() {
            let bytes = unsafe { std::slice::from_raw_parts((value as *const ds41rt_ffi::V41KvStoreLayer).cast::<u8>(), entry) };
            staging[i * entry..(i + 1) * entry].copy_from_slice(bytes);
        }
        let mut host = self.staging.buffer; host.bytes = offset;
        unsafe {
            self.library.copy_host_buffer_h2d_async(self.table.buffer, host, offset, self.stream.raw)?;
            self.kernel.launch(self.table.buffer, layers.len(), rows, chunks, self.stream.raw)
        }
    }
    fn synchronize(&self) -> Result<()> {
        unsafe { self.library.cuda_stream_synchronize(self.stream.raw) }
    }
}
/// Scratch producers retain their wave owner; the caller retains the matching
/// query and admitted cache slots through completion or drained cancellation.
pub(crate) struct PendingProduction<'p, 'w, 'a> {
    execution: &'p mut BackboneExecution<'w, 'a>,
    batch: u64,
    layer: usize,
    source: Option<usize>,
    window_ready: bool,
    source_ready: bool,
    index: Option<&'p mut IndexLane<'w, 'a>>,
    projection_ready: bool,
    selection_started: bool,
    producer_us: Option<u64>,
    complete: bool,
    started: std::time::Instant,
}
impl PendingProduction<'_, '_, '_> {
    pub unsafe fn poll(&mut self, bank: &BackboneCache<'_>, batch: &CacheBatch) -> Result<bool> {
        ensure!(!self.complete && batch.identity() == self.batch, "queued production batch differs");
        bank.validate_batch(batch)?;
        if !self.window_ready {
            self.window_ready = unsafe { self.execution.windows[self.layer].poll_query(bank.window(batch, self.layer)?)? };
        }
        if !self.source_ready {
            let source = self.source.unwrap();
            self.source_ready = unsafe { self.execution.sources[source].poll_query(bank.source(batch, self.layer)?)? };
        }
        if let Some(index) = self.index.as_deref_mut() {
            if !self.projection_ready { self.projection_ready = index.poll_projection()?; }
        }
        if self.window_ready && self.source_ready {
            let producer_us = *self.producer_us.get_or_insert_with(|| self.started.elapsed().as_micros() as u64);
            if !self.projection_ready { return Ok(false); }
            if let Some(index) = self.index.as_deref_mut() {
                if !self.selection_started {
                    let source = SOURCES.iter().rposition(|&l| l <= self.layer)
                        .filter(|_| !batch.stage().reuses_sources()).map(|i| &self.execution.sources[i]);
                    let cache = bank.attention(batch, self.layer, &self.execution.windows[self.layer], source)?;
                    unsafe { index.enqueue_selection(&cache)?; }
                    self.selection_started = true;
                }
                if !index.poll_selection()? { return Ok(false); }
            }
            self.execution.queued_production = Some(Produced { batch: self.batch, layer: self.layer,
                started: self.started, producer_us, indexed: self.index.is_some() });
            self.complete = true;
        }
        Ok(self.complete)
    }
}
impl Drop for PendingProduction<'_, '_, '_> {
    fn drop(&mut self) {
        if !self.complete {
            if let Some(index) = self.index.as_deref_mut() {
                if let Err(error) = index.abort_pending() { tracing::error!(%error, "draining cancelled index work"); }
            }
            if let Err(error) = self.execution.windows[self.layer].abort_query() {
                tracing::error!(%error, "draining cancelled window production");
            }
            if let Some(source) = self.source {
                if let Err(error) = self.execution.sources[source].abort_query() {
                    tracing::error!(%error, "draining cancelled compressed production");
                }
            }
            self.execution.queued_production = None;
        }
    }
}
impl<'w, 'a> BackboneExecution<'w, 'a> {
    pub fn workspace_bytes(library: &NativeLibrary, capacity: u32) -> Result<usize> {
        let mut total = WindowWave::device_bytes(library, capacity)?
            .checked_mul(40)
            .context("window workspace budget overflow")?;
        for layer in SOURCES {
            total = total
                .checked_add(CompressorWave::device_bytes(layer, capacity as usize)?)
                .context("source workspace budget overflow")?;
        }
        Ok(total)
    }
    pub fn new(
        weights: &'w CacheProducerWeights<'a>,
        capacity: u32,
        budget: usize,
    ) -> Result<Self> {
        ensure!(weights.placement.is_none(), "distributed producer weights require placed execution workspaces");
        ensure!(
            Self::workspace_bytes(weights.library, capacity)? <= budget,
            "cache producer workspace exceeds budget"
        );
        ensure!(
            weights.windows.len() == 40 && weights.sources.len() == 4,
            "cache producer weight owners incomplete"
        );
        let windows = weights
            .windows
            .iter()
            .map(|w| {
                w.wave(
                    capacity,
                    WindowWave::device_bytes(weights.library, capacity)?,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let sources = weights
            .sources
            .iter()
            .zip(SOURCES)
            .map(|(w, layer)| {
                w.wave(
                    capacity as usize,
                    CompressorWave::device_bytes(layer, capacity as usize)?,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let window_batch = std::env::var("DS41RT_WINDOW_BATCH").map_or(true, |v| v != "0")
            .then(|| WindowBatch::new(weights.library, capacity as usize)).transpose()?;
        Ok(Self {
            weights,
            windows,
            sources,
            progress: PassProgress::default(),
            queued_production: None,
            window_batch,
        })
    }
    /// Stage every window layer's accepted rows into one batched store. Returns
    /// false (nothing staged) when the per-layer path must be used: encoder
    /// publication, replicated windows, or windows already staged.
    /// # Safety
    /// Same retention contract as enqueue_cache_commit.
    unsafe fn stage_window_batch(&mut self, bank: &BackboneCache<'_>, batch: &CacheBatch,
        accepted: &[u32]) -> Result<bool> {
        let Some(owner) = self.window_batch.as_mut() else { return Ok(false) };
        if batch.stage() != CacheStage::Full
            || self.windows.iter().any(|w| w.has_pending_commit() || w.has_replica()) {
            return Ok(false);
        }
        let (published, _) = bank.validate_commit(batch, &self.windows, &self.sources, accepted)?;
        if published != 0 { return Ok(false); }
        let stream = owner.stream.raw;
        let mut staged = Vec::with_capacity(40);
        let result = (|| -> Result<()> {
            for layer in batch.stage().windows() {
                staged.push(unsafe { self.windows[layer].stage_batched_commit(bank.window(batch, layer)?, accepted, stream)? });
            }
            unsafe { owner.enqueue(&staged) }
        })();
        if let Err(error) = result {
            // The caller's failure path aborts pending commits (drains both streams).
            let _ = owner.synchronize();
            return Err(error);
        }
        Ok(true)
    }
    /// Discard pass progress only after all consumers finish. The caller also
    /// restarts the backbone/index lanes and initializes the next query batch.
    pub fn restart(&mut self) {
        self.queued_production = None;
        self.progress = PassProgress::default();
    }

    pub fn restart_for(&mut self, stage: CacheStage) {
        self.queued_production = None;
        self.progress = PassProgress::for_stage(stage);
    }
    /// Publish source 20 after a complete reserved encoder chunk.
    /// # Safety
    /// The layer-20 query belongs to this chunk and all earlier readers drained.
    pub unsafe fn publish_encoder_boundary(&mut self, bank: &mut BackboneCache<'_>,
        batch: &CacheBatch, lane: &BackboneLane<'_, '_>) -> Result<()> {
        self.progress.begin_decoder_source(batch.identity())?;
        ensure!(batch.is_reserved() && batch.stage() == CacheStage::Encoder,
            "encoder boundary publication requires reserved prefill");
        bank.validate_batch(batch)?;
        let query = lane.query_output()?;
        ensure!(query.layer == 20, "encoder boundary projection layer differs");
        unsafe { bank.produce_source(batch, &query, &mut self.sources[3])?; }
        bank.publish_encoder_source(batch, 20, &mut self.sources[3])?;
        self.progress.finish_decoder_source();
        Ok(())
    }
    /// Produce global source 20 from the prepared encoder boundary, without
    /// running decoder attention, routing, shared FFN or remote experts.
    /// # Safety
    /// The completed layer-20 query preparation belongs to this encoder batch;
    /// its source input and all device owners remain exclusive through completion.
    pub unsafe fn produce_decoder_source(&mut self, bank: &BackboneCache<'_>,
        batch: &CacheBatch, lane: &BackboneLane<'_, '_>) -> Result<()> {
        self.progress.begin_decoder_source(batch.identity())?;
        ensure!(batch.stage() == CacheStage::Encoder, "decoder source batch phase differs");
        bank.validate_batch(batch)?;
        let query = lane.query_output()?;
        ensure!(query.layer == 20, "decoder source requires layer-20 projection input");
        unsafe { bank.produce_source(batch, &query, &mut self.sources[3])?; }
        self.progress.finish_decoder_source();
        Ok(())
    }

    /// # Safety
    /// Keep the lane query and this batch's admitted cache slots immutable and
    /// alive until the returned producer completes or drains on drop.
    pub unsafe fn enqueue_production(&mut self, bank: &BackboneCache<'_>, batch: &CacheBatch,
        lane: &BackboneLane<'_, '_>) -> Result<PendingProduction<'_, 'w, 'a>> {
        ensure!(self.queued_production.is_none(), "prior queued production not consumed");
        bank.validate_batch(batch)?;
        let query = lane.query_output()?;
        let layer = query.layer;
        ensure!(layer == self.progress.next && batch.stage() == self.progress.stage,
            "queued production layer or stage differs");
        let source = SOURCES.iter().position(|&l| l == layer).filter(|_| !batch.stage().reuses_sources());
        let pending = PendingProduction { execution: self, batch: batch.identity(), layer, source,
            window_ready: false, source_ready: source.is_none(), complete: false,
            index: None, projection_ready: true, selection_started: false, producer_us: None,
            started: std::time::Instant::now() };
        unsafe {
            pending.execution.windows[layer].enqueue_query(bank.window(batch, layer)?,
                &batch.window_chunks(layer)?, &query)?;
            if let Some(i) = source {
                pending.execution.sources[i].enqueue_query(bank.source(batch, layer)?,
                    &batch.source_chunks(layer)?, &query)?;
            }
        }
        Ok(pending)
    }
    /// # Safety
    /// Same retained query/cache contract as enqueue_production, with index
    /// storage retained by the returned owner until all producers/selection drain.
    pub unsafe fn enqueue_production_and_index<'p>(&'p mut self, bank: &BackboneCache<'_>,
        batch: &CacheBatch, lane: &BackboneLane<'_, '_>, index: &'p mut IndexLane<'w, 'a>)
        -> Result<PendingProduction<'p, 'w, 'a>> {
        let mut pending = unsafe { self.enqueue_production(bank, batch, lane)? };
        if INDEX.contains(&pending.layer) {
            pending.index = Some(index);
            pending.projection_ready = false;
            unsafe { pending.index.as_deref_mut().unwrap().enqueue_projection(&lane.query_output()?)?; }
        }
        Ok(pending)
    }
    /// Execute one already-prepared layer through completed FFN/mHC output.
    /// # Safety
    /// Lane query rows, modality mask and cache batch identify the same requests.
    /// All producers have completed; no external writes race any passed owner.
    /// The caller handles engram, decoder taps and next-layer preparation between
    /// calls. Cancellation after polling requires restarting these lane owners.
    pub async unsafe fn execute_layer(
        &mut self,
        bank: &BackboneCache<'_>,
        batch: &CacheBatch,
        lane: &mut BackboneLane<'_, '_>,
        index: &mut IndexLane<'_, '_>,
        transport: &mut NativeTp4Wave<'_>,
        placement: u64,
        image_mask: &[u8],
    ) -> Result<()> {
        let prepared = unsafe { self.prepare_layer(bank, batch, lane, index)? };
        let completed = unsafe { prepared.execute(transport, placement, image_mask).await? };
        unsafe { self.complete_layer(batch, lane, completed) }
    }
    /// Finish all cache production, indexing and attention without holding the
    /// bank across remote execution. Dropping prepared work leaves progress
    /// invalid until restart, just as cancelling ordinary layer execution does.
    /// # Safety
    /// Same completed-query, cache ownership and device contract as execute_layer.
    pub unsafe fn prepare_layer<'l, 'lw, 'la>(&mut self, bank: &BackboneCache<'_>,
        batch: &CacheBatch, lane: &'l mut BackboneLane<'lw, 'la>,
        index: &mut IndexLane<'_, '_>) -> Result<PreparedLayer<'l, 'lw, 'la>> {
        unsafe { self.prepare_layer_with_cache(LayerCache::Ordinary(bank), batch, lane, index, false) }
    }
    /// # Safety
    /// Same contract as prepare_layer, extended through returned work completion:
    /// preserve batch cache slots, this execution's producers and index storage.
    /// Cancellation must drop the returned work before releasing those owners.
    pub unsafe fn prepare_layer_cooperative<'l, 'lw, 'la>(&mut self, bank: &BackboneCache<'_>,
        batch: &CacheBatch, lane: &'l mut BackboneLane<'lw, 'la>,
        index: &mut IndexLane<'_, '_>) -> Result<PreparedLayer<'l, 'lw, 'la>> {
        unsafe { self.prepare_layer_with_cache(LayerCache::Ordinary(bank), batch, lane, index, true) }
    }
    /// Prepare a reserved encoder layer and publish source/window KV before
    /// returning its FFN owner. Published sources feed all later index consumers.
    /// # Safety
    /// Same query/owner contract as prepare_layer; earlier readers have drained.
    pub unsafe fn prepare_encoder_layer<'l, 'lw, 'la>(&mut self, bank: &mut BackboneCache<'_>,
        batch: &CacheBatch, lane: &'l mut BackboneLane<'lw, 'la>,
        index: &mut IndexLane<'_, '_>) -> Result<PreparedLayer<'l, 'lw, 'la>> {
        ensure!(batch.is_reserved() && batch.stage() == CacheStage::Encoder,
            "published execution requires a reserved encoder batch");
        unsafe { self.prepare_layer_with_cache(LayerCache::Encoder(bank), batch, lane, index, false) }
    }
    unsafe fn prepare_layer_with_cache<'l, 'lw, 'la>(&mut self, mut bank: LayerCache<'_, '_>,
        batch: &CacheBatch, lane: &'l mut BackboneLane<'lw, 'la>,
        index: &mut IndexLane<'_, '_>, cooperative: bool) -> Result<PreparedLayer<'l, 'lw, 'la>> {
        let publishing = matches!(&bank, LayerCache::Encoder(_));
        let queued_production = self.queued_production.take();
        let timing = queued_production.map_or_else(std::time::Instant::now, |p| p.started);
        // Invalidate even if obtaining the completed query or bank check fails.
        let layer = self.progress.next;
        self.progress.begin(batch.identity(), layer)?;
        ensure!(batch.stage() == self.progress.stage, "backbone execution/cache phase differs");
        bank.bank().validate_batch(batch)?;
        let query = lane.query_output()?;
        ensure!(
            query.layer == layer,
            "backbone query layer differs from pass"
        );
        if let Some(produced) = queued_production {
            ensure!(produced.batch == batch.identity() && produced.layer == layer && !publishing,
                "queued cache production identity differs");
        } else {
            unsafe { bank.bank().produce_window(batch, &query, &mut self.windows[layer])?; }
        }
        if let Some(i) = SOURCES
            .iter()
            .position(|&l| l == layer)
            .filter(|_| !batch.stage().reuses_sources())
        {
            if queued_production.is_none() {
                unsafe { bank.bank().produce_source(batch, &query, &mut self.sources[i])?; }
            }
            if let LayerCache::Encoder(bank) = &mut bank {
                bank.publish_encoder_source(batch, layer, &mut self.sources[i])?;
            }
        }
        let produced_us = queued_production.map_or_else(|| timing.elapsed().as_micros() as u64, |p| p.producer_us);
        let source = SOURCES
            .iter()
            .rposition(|&l| l <= layer)
            .filter(|_| !batch.stage().reuses_sources() && !publishing)
            .map(|i| &self.sources[i]);
        let cache = bank.bank().attention(batch, layer, &self.windows[layer], source)?;
        if INDEX.contains(&layer) && !queued_production.is_some_and(|p| p.indexed) {
            unsafe {
                lane.select_index(index, &cache)?;
            }
        }
        let indexed_us = timing.elapsed().as_micros() as u64;
        let sink = self.weights.sink_views[layer];
        let rows = batch.expert_rows();
        let ffn = if cooperative {
            PreparedFfn::Pending(unsafe { lane.enqueue_attention_indexed_ffn(sink, &cache, index)? })
        } else {
            PreparedFfn::Ready(unsafe { lane.attention_indexed_ffn(sink, &cache, index)? })
        };
        drop(cache);
        if let LayerCache::Encoder(bank) = &mut bank {
            bank.publish_encoder_window(batch, layer, &mut self.windows[layer])?;
        }
        Ok(PreparedLayer { ffn, rows, batch: batch.identity(), layer, started: timing,
            produced_us, indexed_us, attended_us: timing.elapsed().as_micros() as u64 })
    }
    /// # Safety
    /// The completed transport output and lane belong to this prepared layer;
    /// no external writes race the final mHC operation.
    pub unsafe fn complete_layer(&mut self, batch: &CacheBatch,
        lane: &mut BackboneLane<'_, '_>, completed: CompletedLayer<'_>) -> Result<()> {
        ensure!(self.progress.invalid && self.progress.batch == Some(completed.batch)
            && batch.identity() == completed.batch && self.progress.next == completed.layer
            && self.progress.stage == batch.stage(), "completed backbone layer identity differs");
        unsafe { lane.finish_ffn(completed.result.binding(), completed.result.values)?; }
        completed.log_cost(lane);
        tracing::debug!(target: "ds41rt::timing", layer=completed.layer, rows=completed.rows,
            produced_us=completed.produced_us, index_us=completed.indexed_us-completed.produced_us,
            attention_us=completed.attended_us-completed.indexed_us,
            experts_us=completed.experts_us-completed.attended_us,
            finish_us=completed.started.elapsed().as_micros() as u64-completed.experts_us, "target layer");
        self.progress.finish();
        Ok(())
    }
    /// # Safety
    /// Same exact batch/result contract as complete_layer; the future retains
    /// the transport output until mHC and queued next-input copies finish.
    pub async unsafe fn complete_layer_cooperative(&mut self, batch: &CacheBatch,
        lane: &mut BackboneLane<'_, '_>, completed: CompletedLayer<'_>) -> Result<()> {
        ensure!(self.progress.invalid && self.progress.batch == Some(completed.batch)
            && batch.identity() == completed.batch && self.progress.next == completed.layer
            && self.progress.stage == batch.stage(), "completed backbone layer identity differs");
        unsafe { lane.finish_ffn_cooperative(completed.result.binding(), completed.result.values).await?; }
        completed.log_cost(lane);
        tracing::debug!(target: "ds41rt::timing", layer=completed.layer, rows=completed.rows,
            produced_us=completed.produced_us, index_us=completed.indexed_us-completed.produced_us,
            attention_us=completed.attended_us-completed.indexed_us,
            experts_us=completed.experts_us-completed.attended_us,
            finish_us=completed.started.elapsed().as_micros() as u64-completed.experts_us, "target layer");
        self.progress.finish();
        Ok(())
    }
    /// Commit after the same batch completed its full or CED phase. The caller
    /// determines acceptance after target-head/sampling/verification and includes
    /// engram/dSpark history in the enclosing scheduler transaction.
    pub unsafe fn enqueue_cache_commit(&mut self, bank: &BackboneCache<'_>,
        batch: &CacheBatch, accepted: &[u32]) -> Result<()> {
        unsafe { self.stage_window_batch(bank, batch, accepted)?; }
        unsafe { bank.enqueue_cache_commit(batch, &mut self.windows, &mut self.sources, accepted) }
    }
    pub fn poll_cache_commit(&self) -> Result<bool> {
        for window in &self.windows { if !window.poll_commit()? { return Ok(false); } }
        for source in &self.sources { if !source.poll_commit()? { return Ok(false); } }
        Ok(true)
    }
    pub fn abort_cache_commit(&mut self, bank: &mut BackboneCache<'_>) -> Result<()> {
        bank.abort_cache_commit(&mut self.windows, &mut self.sources)
    }
    pub fn commit(
        &mut self,
        bank: &mut BackboneCache<'_>,
        batch: &CacheBatch,
        accepted: &[u32],
    ) -> Result<()> {
        self.progress.commit(batch.identity())?;
        ensure!(batch.stage() == self.progress.stage, "backbone commit/cache phase differs");
        // Direct commit: stage and drain the batched store, then publish every
        // window through its (now complete) pending commit.
        if unsafe { self.stage_window_batch(bank, batch, accepted)? } {
            self.window_batch.as_ref().unwrap().synchronize()?;
        }
        bank.commit(batch, &mut self.windows, &mut self.sources, accepted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn official_cache_producer_allocation_plan() -> Result<()> {
        let Some(path) = std::env::var_os("DS41RT_LANE_PLAN_LIBRARY") else {
            eprintln!("skip cache producer planning: DS41RT_LANE_PLAN_LIBRARY unset");
            return Ok(());
        };
        let model = std::env::var_os("DS41RT_LANE_PLAN_MODEL")
            .context("DS41RT_LANE_PLAN_MODEL required")?;
        let library = unsafe { NativeLibrary::load(path)? };
        let catalog = ds41rt_loader::read_official_v41_catalog(
            ds41rt_loader::OFFICIAL_V41_MODEL_ID,
            std::path::Path::new(&model),
        )?;
        let bytes = CacheProducerWeights::device_bytes(&library, &catalog)?;
        assert_eq!(bytes, 145_517_568);
        assert!(
            CacheProducerWeights::load(&library, &catalog, bytes - 1, 1024 * 1024)
                .err()
                .unwrap()
                .to_string()
                .contains("weights exceed budget")
        );
        for capacity in [1, 80, 4096] {
            let workspace = BackboneExecution::workspace_bytes(&library, capacity)?;
            eprintln!("cache producer capacity={capacity} workspace_bytes={workspace}");
        }
        for capacity in [0, 4097, u32::MAX] {
            assert!(BackboneExecution::workspace_bytes(&library, capacity).is_err());
        }
        Ok(())
    }
    #[test]
    fn ced_pass_requires_encoder_source_and_exact_decoder_range() -> Result<()> {
        let mut encoder = PassProgress::for_stage(CacheStage::Encoder);
        for layer in 0..20 { encoder.begin(7, layer)?; encoder.finish(); }
        let mut missing_source = PassProgress::for_stage(CacheStage::Encoder);
        for layer in 0..20 { missing_source.begin(7, layer)?; missing_source.finish(); }
        assert!(missing_source.commit(7).is_err());
        encoder.begin_decoder_source(7)?;
        assert!(encoder.invalid);
        encoder.finish_decoder_source();
        encoder.commit(7)?;
        assert!(encoder.commit(7).is_err());
        let mut interrupted = PassProgress::for_stage(CacheStage::Encoder);
        for layer in 0..20 { interrupted.begin(7, layer)?; interrupted.finish(); }
        interrupted.begin_decoder_source(7)?;
        assert!(interrupted.commit(7).is_err());
        let mut replay = PassProgress::for_stage(CacheStage::Replay);
        assert_eq!(replay.next, 20);
        for layer in 20..40 { replay.begin(8, layer)?; replay.finish(); }
        replay.commit(8)?;
        let mut wrong = PassProgress::for_stage(CacheStage::Replay);
        assert!(wrong.begin(8, 0).is_err());
        let mut wrong = PassProgress::for_stage(CacheStage::Encoder);
        assert!(wrong.begin_decoder_source(7).is_err());
        Ok(())
    }
    #[test]
    fn pass_commit_requires_all_layers_and_one_batch() -> Result<()> {
        let mut pass = PassProgress::default();
        assert!(pass.commit(1).is_err());
        assert!(pass.begin(1, 0).is_err());
        pass = PassProgress::default();
        pass.begin(1, 0)?;
        // An errored or cancelled operation never calls finish and cannot resume.
        assert!(pass.begin(1, 0).is_err());
        pass = PassProgress::default();
        for layer in 0..40 {
            pass.begin(1, layer)?;
            pass.finish();
        }
        assert!(pass.commit(2).is_err());
        assert!(pass.commit(1).is_err());
        pass = PassProgress::default();
        for layer in 0..40 {
            pass.begin(1, layer)?;
            pass.finish();
        }
        pass.commit(1)?;
        assert!(pass.commit(1).is_err());
        pass = PassProgress::default();
        pass.begin(1, 0)?;
        pass.finish();
        assert!(pass.begin(2, 1).is_err());
        pass = PassProgress::default();
        assert!(pass.begin(1, 1).is_err());
        Ok(())
    }
}

#[cfg(test)]
#[path = "v41_backbone_execution/distributed_tests.rs"]
mod distributed_tests;
