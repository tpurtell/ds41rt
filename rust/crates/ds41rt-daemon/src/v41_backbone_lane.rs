//! Reusable backbone execution storage; request caches and index selections live outside the lane.
use crate::v41_attention_binding::QueryBinding;
use crate::v41_attention_output::{AttentionOutputWave, AttentionOutputWeights};
use crate::v41_attention_query::{AttentionQueryOutput, AttentionQueryWave, AttentionQueryWeights};
use crate::v41_backbone_hc::BackboneHcWeights;
use crate::v41_backbone_router::{BackboneRouterWave, BackboneRouterWeights, ExpertRow};
use crate::v41_backbone_shared::{BackboneSharedWave, BackboneSharedWeights, SharedOutput};
use crate::v41_block::{BackboneBlockWave, BlockOutput, FfnInput, PreparedBlockInput};
use crate::v41_engram::{layer::EngramGate, EngramDeviceView};
use crate::v41_experts::coordinator::{NativeFfnOutput, NativeTp4Wave};
use crate::v41_index_selection::IndexSelectionOutput;
use crate::v41_sparse_attention::{AttentionRequest, SparseAttentionWave};
use crate::v41_target_embedding::TargetEmbedding;
use anyhow::{ensure, Context, Result};
use crate::v41_backbone_cache::CachePlacement;
use crate::v41_memory::device::{Device, DeviceOwner};
mod placement;
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary};
use ds41rt_loader::OfficialV41Catalog;

struct LayerWeights<'a> {
    hc: BackboneHcWeights<'a>,
    query: AttentionQueryWeights<'a>,
    projection: AttentionOutputWeights<'a>,
    shared: Option<BackboneSharedWeights<'a>>,
    router: BackboneRouterWeights<'a>,
}

/// All 40 layers' mHC, query, output, shared-expert and router weights. This
/// excludes window/compressor/index weights, engram, vision and dSpark.
pub(crate) struct BackboneLaneWeights<'a> {
    library: &'a NativeLibrary,
    layers: Vec<DeviceOwner<'a, LayerWeights<'a>>>,
    placement: Option<CachePlacement>,
    split_query_b: bool,
    split_output_b: bool,
}
impl<'a> BackboneLaneWeights<'a> {
    fn layer_bytes(
        library: &NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: usize,
    ) -> Result<[usize; 5]> {
        Self::layer_bytes_with_split(library,catalog,layer,false,false)
    }
    fn layer_bytes_with_split(library:&NativeLibrary,catalog:&OfficialV41Catalog,layer:usize,
        split_query_b:bool,split_output_b:bool)->Result<[usize;5]> {
        Ok([
            BackboneHcWeights::device_bytes(catalog, layer)?,
            AttentionQueryWeights::device_bytes_with_split(library, catalog, layer,split_query_b)?,
            AttentionOutputWeights::device_bytes_with_split(library, catalog, layer,split_output_b)?,
            BackboneSharedWeights::device_bytes(library, catalog, layer)?,
            BackboneRouterWeights::device_bytes(catalog, layer)?,
        ])
    }
    /// Conservative device budget including each loader's transient peak.
    pub fn device_bytes(library: &NativeLibrary, catalog: &OfficialV41Catalog) -> Result<usize> {
        (0..40).try_fold(0usize, |total, layer| {
            Self::layer_bytes(library, catalog, layer)?
                .into_iter()
                .try_fold(total, |n, b| {
                    n.checked_add(b)
                        .ok_or_else(|| anyhow::anyhow!("backbone weight budget overflow"))
                })
        })
    }
    pub fn load(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        budget: usize,
        staging: usize,
    ) -> Result<Self> {
        ensure!(
            Self::device_bytes(library, catalog)? <= budget,
            "backbone lane weights exceed budget"
        );
        Self::load_placed(library, catalog, staging, None)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Idle,
    Prepared,
    Query,
    Ffn,
    SharedReady,
    Complete,
    Invalid,
}

// The containing lane's pending owner drains all consumer work before drop.
// Keep this bank before the allocations referenced by its graphs.
struct Tp2FfnGraphs<'w,'a> {
    device: Device<'a>,
    bank: crate::v41_layer_graphs::LayerGraphs<'w,'a,BackboneHcWeights<'a>>,
}
impl<'w,'a> Tp2FfnGraphs<'w,'a> {
    fn new(device:Device<'a>)->Self {
        let mut bank=crate::v41_layer_graphs::LayerGraphs::new(device.library);
        bank.enable_small_shapes();Self {device,bank}
    }
    unsafe fn capture_ready(&mut self,block:&mut BackboneBlockWave<'w,'a>,
        weights:&'w BackboneHcWeights<'a>,binding:QueryBinding,rows:usize,
        projected:Ds41rtDeviceBuffer,stream:*mut std::ffi::c_void)->Result<()> {
        self.device.run(||unsafe {
            block.restore_ffn_graph_warmup(binding,rows)?;
            self.device.library.cuda_graph_begin_capture(stream)?;
            let queued=block.enqueue_ffn(binding,rows,projected,stream);
            let captured=self.device.library.cuda_graph_end_capture(stream);
            match (queued,captured) {
                (Ok(_),Ok(graph))=> {
                    if let Err(error)=self.bank.insert(binding.layer(),weights,rows as u32,graph) {
                        self.device.library.cuda_graph_exec_destroy(graph)?;return Err(error);
                    }
                    Ok(())
                },
                (Err(error),Ok(graph))=> {self.device.library.cuda_graph_exec_destroy(graph)?;Err(error)},
                (Err(error),Err(_)) | (Ok(_),Err(error))=>Err(error),
            }
        })
    }
}
impl Drop for Tp2FfnGraphs<'_,'_> {
    fn drop(&mut self) {
        if let Err(error)=self.device.run(||unsafe {self.bank.clear()}) {
            tracing::error!(%error,"destroying TP2 FFN continuation graphs");
        }
    }
}

struct AttentionTail<'s, 'w, 'a> {
    projection: &'s mut AttentionOutputWave<'w, 'a>,
    block: &'s mut BackboneBlockWave<'w, 'a>,
    binding: QueryBinding,
    tokens: &'s [u64],
    split_output: bool,
}
impl crate::v41_sparse_attention::AttentionGraphTail for AttentionTail<'_, '_, '_> {
    fn identity(&self) -> Vec<usize> {
        let mut identity = self.projection.chain_graph_identity().to_vec();
        identity.extend(self.block.chain_graph_identity()); identity.push(usize::from(self.split_output)); identity
    }
    unsafe fn prepare(&mut self, stream: *mut std::ffi::c_void) -> Result<()> {
        unsafe { self.projection.prepare_chain_graph(self.tokens, stream) }
    }
    unsafe fn enqueue(&mut self, attention: &crate::v41_sparse_attention::QueuedSparseAttention,
        stream: *mut std::ffi::c_void) -> Result<()> {
        if self.split_output {
            unsafe { self.projection.enqueue_grouped_chain_graph(attention,stream)?; }
            return Ok(());
        }
        let projected = unsafe { self.projection.enqueue_chain_graph(attention, stream)? };
        unsafe { self.block.enqueue_ffn(self.binding, self.tokens.len(), projected, stream)?; }
        Ok(())
    }
    unsafe fn restore_warmup(&mut self) -> Result<()> {
        if self.split_output { return Ok(()); }
        unsafe { self.block.restore_ffn_graph_warmup(self.binding, self.tokens.len()) }
    }
    unsafe fn replay_state(&mut self) -> Result<()> {
        if self.split_output { return Ok(()); }
        unsafe { self.block.prepare_ffn_graph_replay(self.binding, self.tokens.len()) }
    }
}

/// Retains all lane consumers until attention finishes. The unsafe constructor's
/// caller also retains the batch cache producers and index storage while pending.
pub(crate) struct PendingLaneFfn<'s, 'w, 'a> {
    lane: Option<&'s mut BackboneLane<'w, 'a>>,
    values: Option<Ds41rtDeviceBuffer>,
}
impl<'s, 'w, 'a> PendingLaneFfn<'s, 'w, 'a> {
    pub async fn complete(mut self) -> Result<LaneFfn<'s, 'w, 'a>> {
        if self.values.is_none() {
            let lane = self.lane.as_deref_mut().unwrap();
            let query = lane.query.output()?;
            let mut tail = AttentionTail { projection: &mut lane.projection, block: &mut lane.block,
                binding: query.binding()?, tokens: query.tokens()?, split_output:lane.tp2_output.is_some() };
            if let Some(dual)=lane.dual_sparse.as_mut() {
                unsafe { dual.complete_owned_tail(&mut tail).await?; }
            } else {
                unsafe { lane.sparse.as_mut().context("attention wave absent")?.finish_prepare(Some(&mut tail)).await?; }
            }
            self.values = Some(lane.block.graph_normalized_storage(query.rows));
        }
        let lane=self.lane.as_deref_mut().unwrap();
        if let Some(output)=&mut lane.tp2_output {
            let query=lane.query.output()?;
            let binding=query.binding()?;let rows=query.rows;
            // Full-head attention leaves its grouped prefix queued. Dual-head
            // completion already drained that prefix; neither case joins lanes.
            let producer=lane.sparse.as_ref().map(|s|s.chain_stream());
            let weights=&lane.weights.layers[lane.layer].hc;
            let graphs=lane.tp2_ffn_graphs.as_mut().context("TP2 FFN graph owner absent")?;
            let cached=graphs.bank.get_shape(lane.layer,weights,rows as u32);
            let library=lane.weights.library;
            let (values,stream,projected)=unsafe { lane.projection.finish_tp2_then(rows as u32,output,producer,
                |projected,stream| {
                    if let Some((graph,_))=cached {
                        lane.block.prepare_ffn_graph_replay(binding,rows)?;
                        library.cuda_graph_launch(graph,stream)?;
                    } else {lane.block.enqueue_ffn(binding,rows,projected,stream)?;}
                    Ok((lane.block.graph_normalized_storage(rows),stream,projected))
                }).await? };
            if cached.is_none() {
                // Warm output is complete. Recording the next invocation restores
                // host phase without executing the FFN prefix a second time.
                unsafe {graphs.capture_ready(&mut lane.block,weights,binding,rows,projected,stream)?;}
            }
            self.values=Some(values);
        } else if let Some(sparse)=&lane.sparse { sparse.wait_chain().await?; }

        let lane = self.lane.take().unwrap();
        let input = unsafe { lane.block.complete_queued_ffn(self.values.unwrap())? };
        lane.phase = Phase::Ffn;
        Ok(LaneFfn { input, cooperative: true, shared: &mut lane.shared, router: &mut lane.router,
            library: lane.weights.library, phase: &mut lane.phase,
            route_capture: if lane.capture_routes { Some(&mut lane.route_capture) } else { None },
            ffn_split: if lane.capture_routes { Some(&mut lane.ffn_split) } else { None },
 })
    }
}
impl Drop for PendingLaneFfn<'_, '_, '_> {
    fn drop(&mut self) {
        if let Some(lane) = self.lane.as_deref_mut() {
            let drained=if let Some(dual)=lane.dual_sparse.as_mut() { dual.drain() }
                else { lane.sparse.as_mut().map_or(Ok(()),|sparse|sparse.drain_chain()) };
            if let Err(error) = drained {
                tracing::error!(%error, "draining cancelled attention chain");
            }
            lane.block.reset();
            lane.phase = Phase::Invalid;
        }
    }
}

/// Dispatch routed work from input before executing the shared contribution on
/// RTX. Both consume the same preserved normalized rows and execution identity.
pub(crate) struct LaneFfn<'s, 'w, 'a> {
    pub input: FfnInput<'s>,
    cooperative: bool,
    shared: &'s mut Option<BackboneSharedWave<'w, 'a>>,
    router: &'s mut BackboneRouterWave<'w, 'a>,
    library: &'a NativeLibrary,
    phase: &'s mut Phase,
    route_capture: Option<&'s mut Vec<Vec<[u32; 6]>>>,
    ffn_split: Option<&'s mut FfnSplit>,
}

/// Host-clock split of a captured pass's FFN stages, summed over its layers.
///
/// Every `Instant` here is taken on every layer regardless; while routes are
/// captured the values are also summed per pass instead of only reaching the
/// `ds41rt::timing` trace. `routed` for a remote layer includes the router's
/// host wait for its route ids, so it tracks device progress; `collect` is the
/// wait for every Spark rank's reply plus the reduction.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct FfnSplit {
    pub local_layers: u32,
    pub local_routed_us: u64,
    pub local_total_us: u64,
    pub remote_layers: u32,
    pub remote_routed_us: u64,
    pub remote_dispatch_us: u64,
    pub remote_shared_us: u64,
    pub remote_collect_us: u64,
}
impl FfnSplit {
    pub fn add(&mut self, other: &Self) {
        self.local_layers += other.local_layers;
        self.local_routed_us += other.local_routed_us;
        self.local_total_us += other.local_total_us;
        self.remote_layers += other.remote_layers;
        self.remote_routed_us += other.remote_routed_us;
        self.remote_dispatch_us += other.remote_dispatch_us;
        self.remote_shared_us += other.remote_shared_us;
        self.remote_collect_us += other.remote_collect_us;
    }
}
impl LaneFfn<'_, '_, '_> {
    #[cfg(test)]
    pub async unsafe fn check_queued_components(&mut self, image_mask: &[u8],
        mut local: Option<&mut crate::v41_experts::local::LocalExpertWave<'_>>) -> Result<()> {
        use std::{future::Future, task::Poll};
        async fn cancel_once<F: Future>(future: F) -> bool {
            let mut future = std::pin::pin!(future);
            std::future::poll_fn(|cx| Poll::Ready(future.as_mut().poll(cx).is_pending())).await
        }
        let lib = self.library;
        let read = |buffers: &[Ds41rtDeviceBuffer]| -> Result<Vec<Vec<u8>>> {
            buffers.iter().map(|&buffer| { let mut out = vec![0; buffer.bytes];
                lib.copy_d2h(&mut out, buffer)?; Ok(out) }).collect()
        };
        let mut cancelled = 0;
        for local_mode in [false, true] {
            self.router.set_local_mode(local_mode)?;
            let routed = unsafe { self.router.execute_ffn(&self.input, image_mask)? };
            let reference = read(&[routed.ids, routed.routing, routed.expert_input])?;
            self.router.clear_graph()?;
            cancelled += cancel_once(unsafe { self.router.execute_ffn_cooperative(&self.input, image_mask) }).await as usize;
            for evict in [true, false, true, false] {
                if evict { self.router.clear_graph()?; }
                let routed = unsafe { self.router.execute_ffn_cooperative(&self.input, image_mask).await? };
                assert_eq!(read(&[routed.ids, routed.routing, routed.expert_input])?, reference);
                let mut ids = Vec::new();
                unsafe { routed.capture_route_ids(lib, &mut ids)?; }
                assert_eq!(ids.iter().flatten().flat_map(|v| v.to_ne_bytes()).collect::<Vec<_>>(), reference[0]);
            }
        }
        let shared_wave = self.shared.as_mut().context("local shared workspace absent")?;
        let shared = unsafe { shared_wave.execute_ffn(&self.input)? };
        let reference = read(&[shared.values])?;
        shared_wave.clear_graph()?;
        cancelled += cancel_once(unsafe { shared_wave.execute_ffn_cooperative(&self.input) }).await as usize;
        for evict in [true, false, true, false] {
            if evict { shared_wave.clear_graph()?; }
            let shared = unsafe { shared_wave.execute_ffn_cooperative(&self.input).await? };
            assert_eq!(read(&[shared.values])?, reference);
        }
        if let Some(local) = local.as_deref_mut() {
            let routed = self.router.output()?;
            let shared = shared_wave.output()?;
            let reference = read(&[unsafe { local.execute(&routed, &shared)? }])?;
            cancelled += cancel_once(unsafe { local.execute_cooperative(&routed, &shared) }).await as usize;
            for _ in 0..2 {
                assert_eq!(read(&[unsafe { local.execute_cooperative(&routed, &shared).await? }])?, reference);
            }
        }
        eprintln!("PASS queued FFN components layer {}: direct parity, cold/warm, {cancelled} pending cancellations/reuse", self.input.layer);
        Ok(())
    }
    /// # Safety
    /// Metadata and mask identify the actual requests/modality in input order.
    /// All external producers are complete and no writes race this lane.
    pub async unsafe fn execute_tp4<'t>(
        &mut self,
        transport: &'t mut NativeTp4Wave<'_>,
        placement: u64,
        image_mask: &[u8],
        rows: &[ExpertRow],
    ) -> Result<NativeFfnOutput<'t>> {
        let cooperative = self.cooperative;
        let input = &self.input;
        let router = &mut self.router;
        let shared = &mut self.shared;
        let library = self.library;
        let route_capture = &mut self.route_capture;
        let ffn_split = &mut self.ffn_split;
        let output = complete_ffn(self.phase, async {
            let timing = std::time::Instant::now();
            router.set_local_mode(transport.has_local_layer(input.layer))?;
            let routed = unsafe { if cooperative { router.execute_ffn_cooperative(input, image_mask).await? }
                else { router.execute_ffn(input, image_mask)? } };
            if transport.has_local_layer(input.layer) {
                routed.validate_request_rows(rows)?;
                let trace = tracing::enabled!(target: "ds41rt::route_policy", tracing::Level::DEBUG);
                let mut temporary = Vec::new();
                let mut captured = if let Some(capture) = route_capture.as_deref_mut() {
                    Some(&mut capture[input.layer])
                } else if trace { Some(&mut temporary) } else { None };
                if let Some(output) = captured.as_deref_mut() {
                    unsafe { routed.capture_route_ids(library, output)?; }
                }
                let routed_us = timing.elapsed().as_micros() as u64;
                let result = if transport.has_tp2_layer(input.layer) {
                    unsafe { transport.execute_tp2_ffn(input, &routed).await }
                } else {
                    let shared = shared.as_mut().context("local shared workspace absent; TP2 required")?;
                    let contribution = unsafe { if cooperative { shared.execute_ffn_cooperative(input).await? }
                        else { shared.execute_ffn(input)? } };
                    unsafe { if cooperative { transport.execute_local_ffn_cooperative(&routed, &contribution).await }
                        else { transport.execute_local_ffn(&routed, &contribution) } }
                };
                let ffn_us = timing.elapsed().as_micros() as u64;
                if let Some(split) = ffn_split.as_deref_mut() {
                    split.local_layers += 1;
                    split.local_routed_us += routed_us;
                    split.local_total_us += ffn_us;
                }
                tracing::debug!(target: "ds41rt::timing", layer=input.layer, rows=rows.len(), routed_us,
                    total_us=ffn_us, "target local experts");
                if trace {
                    let route_ids: Vec<_> = captured.as_ref().expect("trace capture exists").iter().flatten().copied().collect();
                    let owners: Vec<_> = rows.iter().map(|r| (r.request_id, r.position)).collect();
                    let unique_experts = route_ids.iter().collect::<std::collections::BTreeSet<_>>().len();
                    tracing::debug!(target: "ds41rt::route_policy", layer=input.layer, local=true,
                        rows=rows.len(), unique_experts, ffn_us, routed_us,
                        remote_and_shared_us=ffn_us-routed_us, owners=?owners, route_ids=?route_ids,
                        "native route policy observation");
                }
                return result;
            }
            let tp2_shared = transport.has_tp2_shared_layer(input.layer);
            ensure!(shared.is_some() || tp2_shared, "decoder shared TP2 execution required");
            let mut request = unsafe { routed.expert_request(library, placement, rows)? };
            if let Some(capture) = route_capture.as_deref_mut() {
                let output = &mut capture[input.layer];
                output.clear();
                output.extend(request.request().routes.chunks_exact(6)
                    .map(|routes| std::array::from_fn(|i| routes[i].expert_id)));
            }
            let routed_us = timing.elapsed().as_micros() as u64;
            transport.prepare_remote_request(&mut request)?;
            let pending = transport.dispatch_ffn(&request).await?;
            let dispatched_us = timing.elapsed().as_micros() as u64;
            if tp2_shared {
                let result = unsafe { pending.finish_tp2(input).await };
                if let Some(split) = ffn_split.as_deref_mut() {
                    split.remote_layers += 1;
                    split.remote_routed_us += routed_us;
                    split.remote_dispatch_us += dispatched_us - routed_us;
                    split.remote_collect_us += timing.elapsed().as_micros() as u64 - dispatched_us;
                }
                tracing::debug!(target: "ds41rt::timing", layer=input.layer, rows=rows.len(), routed_us,
                    dispatch_us=dispatched_us-routed_us, shared_and_collect_us=timing.elapsed().as_micros() as u64-dispatched_us,
                    "target experts with TP2 shared");
                return result;
            }
            let shared = shared.as_mut().context("decoder shared TP2 execution required")?;
            let contribution = unsafe { if cooperative { shared.execute_ffn_cooperative(input).await? }
                    else { shared.execute_ffn(input)? } };
            let shared_us = timing.elapsed().as_micros() as u64;
            let result = unsafe { if cooperative { pending.finish_cooperative(&contribution).await }
                else { pending.finish(&contribution).await } }?;
            if let Some(split) = ffn_split.as_deref_mut() {
                split.remote_layers += 1;
                split.remote_routed_us += routed_us;
                split.remote_dispatch_us += dispatched_us - routed_us;
                split.remote_shared_us += shared_us - dispatched_us;
                split.remote_collect_us += timing.elapsed().as_micros() as u64 - shared_us;
            }
            tracing::debug!(target: "ds41rt::timing", layer=input.layer, rows=rows.len(), routed_us, dispatch_us=dispatched_us-routed_us, shared_us=shared_us-dispatched_us, collect_us=timing.elapsed().as_micros() as u64-shared_us, "target experts");
            if tracing::enabled!(target: "ds41rt::route_policy", tracing::Level::DEBUG) {
                let ffn_us = timing.elapsed().as_micros() as u64;
                // The dispatch request already owns these CPU-side routes.
                // No extra device read or worker instrumentation is needed.
                let route_ids: Vec<_> = request.request().routes.iter().map(|r| r.expert_id & 511).collect();
                let owners: Vec<_> = rows.iter().map(|r| (r.request_id, r.position)).collect();
                let unique_experts = route_ids.iter().collect::<std::collections::BTreeSet<_>>().len();
                tracing::debug!(target: "ds41rt::route_policy", layer=input.layer,
                    rows=rows.len(), unique_experts, ffn_us, routed_us,
                    remote_and_shared_us=ffn_us-routed_us,
                    owners=?owners, route_ids=?route_ids,
                    "native route policy observation");
            }
            Ok(result)
        })
        .await;
        output
    }
    /// # Safety
    /// No external writes race the preserved FFN input or shared workspace.
    pub unsafe fn execute_shared(&mut self) -> Result<SharedOutput<'_>> {
        let prior = std::mem::replace(self.phase, Phase::Invalid);
        ensure!(prior == Phase::Ffn, "lane shared FFN is not pending");
        let output = unsafe { self.shared.as_mut().context("local shared workspace absent")?.execute_ffn(&self.input)? };
        *self.phase = Phase::SharedReady;
        Ok(output)
    }
}

/// Invalidate before the first operation is polled. Dropping a polled future
/// therefore requires lane restart; successful completion alone republishes it.
async fn complete_ffn<T>(
    phase: &mut Phase,
    work: impl std::future::Future<Output = Result<T>>,
) -> Result<T> {
    let prior = std::mem::replace(phase, Phase::Invalid);
    ensure!(prior == Phase::Ffn, "lane FFN work is not pending");
    let result = work.await?;
    *phase = Phase::SharedReady;
    Ok(result)
}

pub(crate) struct BackboneLane<'w, 'a> {
    tp2_ffn_graphs: Option<Tp2FfnGraphs<'w,'a>>,
    // Destroy containing graphs before their captured consumer allocations.
    sparse: Option<SparseAttentionWave<'a>>,
    dual_sparse: Option<crate::v41_sparse_attention::dual::DualAttentionWave<'a>>,
    capacity: usize,
    weights: &'w BackboneLaneWeights<'a>,
    block: BackboneBlockWave<'w, 'a>,
    query: AttentionQueryWave<'w, 'a>,
    tp2_query: Option<crate::v41_projection_tp2::Wave<'w,'a>>,
    tp2_output: Option<crate::v41_projection_tp2::Wave<'w,'a>>,
    projection: AttentionOutputWave<'w, 'a>,
    shared: Option<BackboneSharedWave<'w, 'a>>,
    router: BackboneRouterWave<'w, 'a>,
    layer: usize,
    phase: Phase,
    capture_routes: bool,
    route_capture: Vec<Vec<[u32; 6]>>,
    /// Host FFN stage times summed over the captured pass.
    ffn_split: FfnSplit,
    /// Timing events recorded on the FFN-finish stream while routes are
    /// captured; consecutive events give each layer's device time.
    layer_events: LayerEvents<'a>,
}

/// One timing event per layer, created on the lane's device.
struct LayerEvents<'a> {
    library: &'a NativeLibrary,
    events: Vec<*mut std::ffi::c_void>,
    recorded: Vec<bool>,
    /// Recorded when a layer's input arrives from the other GPU, so the first
    /// layer after a handoff has a same-GPU start.
    entry: *mut std::ffi::c_void,
    entry_layer: Option<usize>,
}
impl<'a> LayerEvents<'a> {
    fn new(library: &'a NativeLibrary) -> Result<Self> {
        let mut events = Vec::with_capacity(41);
        for _ in 0..41 {
            match library.cuda_event_create() {
                Ok(event) => events.push(event),
                Err(error) => {
                    for &event in &events { let _ = unsafe { library.cuda_event_destroy(event) }; }
                    return Err(error);
                }
            }
        }
        let entry = events.pop().expect("entry event");
        Ok(Self { library, events, recorded: vec![false; 40], entry, entry_layer: None })
    }
}
impl Drop for LayerEvents<'_> {
    fn drop(&mut self) {
        for &event in self.events.iter().chain([&self.entry]) {
            if let Err(error) = unsafe { self.library.cuda_event_destroy(event) } {
                tracing::error!(%error, "destroying layer timing event");
            }
        }
    }
}
impl<'w, 'a> BackboneLane<'w, 'a> {
    pub fn workspace_bytes(library: &NativeLibrary, capacity: u32) -> Result<[usize; 6]> {
        ensure!(
            (1..=4096).contains(&capacity),
            "invalid backbone lane capacity"
        );
        Ok([
            BackboneBlockWave::device_bytes(capacity as usize)?,
            AttentionQueryWave::device_bytes(library, capacity)?,
            AttentionOutputWave::device_bytes(library, capacity)?,
            BackboneSharedWave::device_bytes(library, capacity)?,
            SparseAttentionWave::device_bytes(capacity as usize)?,
            BackboneRouterWave::device_bytes(capacity)?,
        ])
    }
    /// One lane owns one allocation of each workspace group. Independent lanes
    /// borrow the same immutable weights and allocate distinct mutable storage.
    pub fn new(weights: &'w BackboneLaneWeights<'a>, capacity: u32, budget: usize) -> Result<Self> {
        ensure!(weights.placement.is_none(), "placed backbone weights require GPU-owned lanes");
        Self::new_inner(weights, capacity, budget, 0)
    }
    fn new_inner(weights: &'w BackboneLaneWeights<'a>, capacity: u32, budget: usize, first_layer: usize) -> Result<Self> {
        let mut sizes = Self::workspace_bytes(weights.library, capacity)?;
        if weights.placement.is_some() { sizes[3] = 0; }
        sizes[1]=AttentionQueryWave::device_bytes_with_split(weights.library,capacity,weights.split_query_b)?;
        sizes[2]=AttentionOutputWave::device_bytes_with_split(weights.library,capacity,weights.split_output_b)?;
        let total = sizes.into_iter().try_fold(0usize, |n, b| {
            n.checked_add(b)
                .ok_or_else(|| anyhow::anyhow!("backbone lane budget overflow"))
        })?;
        ensure!(total <= budget, "backbone lane exceeds budget");
        ensure!(
            weights.layers.len() == 40,
            "backbone lane requires all 40 layers"
        );
        let first = &weights.layers[first_layer];
        Ok(Self {
            tp2_ffn_graphs: weights.split_output_b.then(||Tp2FfnGraphs::new(first.device)),
            weights,
            block: first.hc.block(capacity as usize, sizes[0])?,
            query: first.query.wave(capacity, sizes[1])?,
            tp2_query: None,
            tp2_output: None,
            projection: first.projection.wave(capacity, sizes[2])?,
            shared: first.shared.as_ref().map(|w| w.wave(capacity,sizes[3])).transpose()?,
            sparse: Some(SparseAttentionWave::new(weights.library, capacity as usize, sizes[4])?),
            dual_sparse: None,
            capacity: capacity as usize,
            router: first.router.wave(capacity, sizes[5])?,
            layer: first_layer,
            phase: Phase::Idle,
            capture_routes: false,
            route_capture: Vec::new(),
            ffn_split: FfnSplit::default(),
            layer_events: LayerEvents::new(weights.library)?,
        })
    }
    fn enter(&mut self, expected: Phase) -> Result<()> {
        let prior = std::mem::replace(&mut self.phase, Phase::Invalid);
        ensure!(
            prior == expected,
            "backbone lane phase differs: expected {expected:?}, got {prior:?}"
        );
        Ok(())
    }
    /// Cancel or start another wave. External consumers must have finished;
    /// borrowed outputs prevent safe callers from restarting during consumption.
    pub fn invalidate(&mut self) {
        self.phase = Phase::Invalid;
        self.block.reset();
    }
    /// Start a new sequence on the GPU owning layer zero after consumers drain.
    pub fn restart(&mut self) -> Result<()> {
        self.phase = Phase::Invalid;
        self.block.reset();
        ensure!(self.weights.layers[0].device.id == self.block.inputs()[0].device_id, "layer zero belongs to another GPU");
        let first = &self.weights.layers[0];
        self.query.rebind(&first.query)?;
        self.projection.rebind(&first.projection)?;
        if let Some(shared) = &mut self.shared { shared.rebind(first.shared.as_ref().context("local shared weights absent")?)?; }
        self.router.rebind(&first.router)?;
        self.block.restart(&first.hc)?;
        self.layer = 0;
        self.phase = Phase::Idle;
        Ok(())
    }
    pub fn restart_decoder(&mut self, encoder: &BlockOutput<'_>) -> Result<()> {
        self.phase = Phase::Invalid;
        self.block.reset();
        ensure!(self.weights.layers[20].device.id == self.block.inputs()[0].device_id, "decoder belongs to another GPU");
        let decoder = &self.weights.layers[20];
        self.query.rebind(&decoder.query)?;
        self.projection.rebind(&decoder.projection)?;
        if let Some(shared) = &mut self.shared { shared.rebind(decoder.shared.as_ref().context("local shared weights absent")?)?; }
        self.router.rebind(&decoder.router)?;
        self.block.initialize_decoder(&decoder.hc, encoder)?;
        self.layer = 20;
        self.phase = Phase::Prepared;
        Ok(())
    }
    /// # Safety
    /// Same finite, completed embedding and exclusive-storage contract as the
    /// component owners. This text entry does not implement vision replacement.
    pub unsafe fn begin_embedded(
        &mut self,
        embedding: &TargetEmbedding<'_>,
    ) -> Result<AttentionQueryOutput<'_>> {
        self.enter(Phase::Idle)?;
        ensure!(self.layer == 0, "embedding requires layer zero");
        let result = unsafe {
            self.block
                .begin_embedded_attention(&mut self.query, embedding)?
        };
        self.phase = Phase::Query;
        Ok(result)
    }
    /// # Safety
    /// Own embedding, block and query storage through the drained producer chain.
    pub async unsafe fn begin_tokens_cooperative(&mut self,
        embedding: &mut crate::v41_target_embedding::TargetEmbeddingWave<'_, '_>,
        tokens: &[u32], positions: &[u64]) -> Result<AttentionQueryOutput<'_>> {
        self.enter(Phase::Idle)?;
        ensure!(self.layer == 0 && tokens.len() == positions.len(), "invalid token entry");
        let result = unsafe { self.block.begin_attention_with_projection_cooperative(&mut self.query, positions,self.tp2_query.as_mut(),
            |stream, destination| embedding.enqueue_into(tokens, stream, destination)).await? };
        self.phase = Phase::Query;
        Ok(result)
    }
    /// # Safety
    /// Same ownership contract as begin_prepared, retained across suspension.
    pub async unsafe fn begin_prepared_cooperative(&mut self) -> Result<AttentionQueryOutput<'_>> {
        self.enter(Phase::Prepared)?;
        let result = unsafe { self.block.begin_prepared_attention_with_projection_cooperative(&mut self.query,self.tp2_query.as_mut()).await? };
        self.phase = Phase::Query;
        Ok(result)
    }
    /// Test-only query parity. Caller reinitializes the block afterward because
    /// each execution intentionally creates a fresh query binding.
    #[cfg(test)]
    pub async unsafe fn check_queued_query(&mut self) -> Result<()> {
        let library = self.weights.library;
        let read = |output: &AttentionQueryOutput<'_>| -> Result<Vec<Vec<u8>>> {
            [output.hidden, output.raw_rank, output.normalized_rank,
                output.projected, output.rotated, output.positions, output.frequencies]
                .into_iter().map(|buffer| {
                    let mut bytes = vec![0; buffer.bytes];
                    library.copy_d2h(&mut bytes, buffer)?;
                    Ok(bytes)
                }).collect()
        };
        let output = self.query.output()?;
        let tokens = output.tokens()?.to_vec();
        let reference = read(&output)?;
        for evict in [true, false, true, false] {
            if evict { self.query.clear_graph()?; }
            let output = unsafe {
                self.query.execute_tokens_prepared_cooperative(&tokens, |_, _| Ok(())).await?
            };
            assert_eq!(read(&output)?, reference,
                "cooperative query differs after eviction={evict}");
        }
        eprintln!("PASS query layer {}: exact cold, warm and recapture output parity", self.layer);
        Ok(())
    }
    pub fn pending_engram(&self) -> Result<(usize, &[u64])> {
        ensure!(self.phase == Phase::Prepared, "backbone lane input not prepared");
        self.block.pending_engram()
    }
    pub fn prepared_input(&self) -> Result<PreparedBlockInput<'_>> {
        ensure!(
            self.phase == Phase::Prepared,
            "backbone lane input not prepared"
        );
        self.block.prepared_input()
    }
    /// # Safety
    /// Gathered rows/history are associated with this lane's exact request order,
    /// and all producers/consumers are complete as required by EngramGate.
    pub unsafe fn apply_engram(
        &mut self,
        gate: &mut EngramGate<'_, '_>,
        rows: &EngramDeviceView,
    ) -> Result<()> {
        self.enter(Phase::Prepared)?;
        unsafe {
            self.block.apply_engram(gate, rows)?;
        }
        self.phase = Phase::Prepared;
        Ok(())
    }
    #[cfg(test)]
    pub async unsafe fn check_queued_engram(&mut self, gate: &mut EngramGate<'_, '_>,
        rows: &EngramDeviceView) -> Result<()> {
        unsafe { gate.check_cooperative(self.block.inputs()[0], rows).await }
    }
    /// # Safety
    /// Retain the gathered upload while this lane's gate and residual copy finish.
    pub async unsafe fn apply_engram_cooperative(&mut self,
        gate: &mut crate::v41_engram::layer::EngramGate<'_, '_>,
        rows: &crate::v41_engram::EngramDeviceView) -> Result<()> {
        self.enter(Phase::Prepared)?;
        unsafe { self.block.apply_engram_cooperative(gate, rows).await?; }
        self.phase = Phase::Prepared;
        Ok(())
    }
    /// # Safety
    /// Required dSpark tap consumers must finish before this call. The block
    /// checks that engram has completed on layers 1 and 14.
    pub unsafe fn begin_prepared(&mut self) -> Result<AttentionQueryOutput<'_>> {
        self.enter(Phase::Prepared)?;
        let result = unsafe { self.block.begin_prepared_attention(&mut self.query)? };
        self.phase = Phase::Query;
        Ok(result)
    }
    pub fn query_output(&self) -> Result<AttentionQueryOutput<'_>> {
        ensure!(
            self.phase == Phase::Query,
            "backbone lane query unavailable"
        );
        self.query.output()
    }
    /// # Safety
    /// Request windows, sources, selected rows and sink obey the sparse attention
    /// owner's matching-layer and completed-producer contract. The returned FFN
    /// work keeps its input stable while routed dispatch and shared work proceed.
    pub unsafe fn attention_ffn(
        &mut self,
        sink: Ds41rtDeviceBuffer,
        requests: &[AttentionRequest<'_>],
        selection: Option<&IndexSelectionOutput<'_>>,
    ) -> Result<LaneFfn<'_, 'w, 'a>> {
        ensure!(self.tp2_output.is_none(),"TP2 output requires cooperative attention completion");
        self.enter(Phase::Query)?;
        let query = self.query.output()?;
        let timing = std::time::Instant::now();
        let binding = query.binding()?;
        let tokens = query.tokens()?;
        let mut tail = AttentionTail { projection: &mut self.projection, block: &mut self.block, binding, tokens, split_output:false };
        let result = unsafe { self.sparse.as_mut().context("direct attention requires full-head mode")?
            .execute_query_graph(&query, sink, requests, selection, &mut tail) };
        if let Err(error) = result { self.block.reset(); self.phase = Phase::Invalid; return Err(error); }
        let values = self.block.graph_normalized_storage(tokens.len());
        let input = unsafe { self.block.complete_queued_ffn(values)? };
        tracing::debug!(target: "ds41rt::timing", layer=self.layer, rows=query.rows,
            total_us=timing.elapsed().as_micros() as u64, "target attention chain");
        self.phase = Phase::Ffn;
        Ok(LaneFfn {
            input,
            cooperative: false,
            shared: &mut self.shared,
            router: &mut self.router,
            library: self.weights.library,
            phase: &mut self.phase,
            route_capture: if self.capture_routes { Some(&mut self.route_capture) } else { None },
            ffn_split: if self.capture_routes { Some(&mut self.ffn_split) } else { None },
        })
    }
    /// # Safety
    /// Keep this batch's window/source producers, admitted cache slots and index
    /// selection storage alive and immutable until the returned owner completes
    /// or drops. Peer lanes may mutate only disjoint request slots/page claims.
    pub unsafe fn enqueue_attention_indexed_ffn(&mut self, sink: Ds41rtDeviceBuffer,
        cache: &crate::v41_backbone_cache::CacheAttention<'_>,
        index: &crate::v41_index_lane::IndexLane<'_, '_>) -> Result<PendingLaneFfn<'_, 'w, 'a>> {
        self.enter(Phase::Query)?;
        let selection = if self.layer >= 2 { Some(index.output(self.layer, cache)?) } else { None };
        unsafe { self.enqueue_attention_with_selection(sink,cache,selection.as_ref()) }
    }
    /// # Safety
    /// Retain all cache/query/selection storage until the returned consumer completes or drains.
    pub unsafe fn enqueue_attention_cached_ffn(&mut self,sink:Ds41rtDeviceBuffer,
        cache:&crate::v41_backbone_cache::CacheAttention<'_>,selection:Option<&IndexSelectionOutput<'_>>)
        ->Result<PendingLaneFfn<'_,'w,'a>> {
        self.enter(Phase::Query)?;
        unsafe { self.enqueue_attention_with_selection(sink,cache,selection) }
    }
    /// # Safety
    /// Keep the bank, query, proposals and selection alive until completion or
    /// cancellation. All producers and committed replica publications are ready.
    pub unsafe fn enqueue_attention_replicated_ffn(&mut self,sink:Ds41rtDeviceBuffer,
        bank:&crate::v41_backbone_cache::BackboneCache<'_>,cache:&crate::v41_backbone_cache::CacheAttention<'_>,
        index:Option<&crate::v41_index_lane::IndexLane<'_, '_>>)->Result<PendingLaneFfn<'_,'w,'a>> {
        if self.dual_sparse.is_none() {
            return match index { Some(index)=>unsafe { self.enqueue_attention_indexed_ffn(sink,cache,index) },
                None=>unsafe { self.enqueue_attention_cached_ffn(sink,cache,None) } };
        }
        self.enter(Phase::Query)?;
        // The first two layers use only SWA and never produce an index selection.
        let selection=if self.layer>=2 {
            Some(index.context("attention index lane missing")?.output(self.layer,cache)?)
        } else { None };
        let mut pending=PendingLaneFfn { lane:Some(self),values:None };
        let lane=pending.lane.as_deref_mut().unwrap();
        let query=lane.query.output()?;
        unsafe { lane.dual_sparse.as_mut().unwrap().enqueue_cached_owned(&query,sink,bank,cache,selection.as_ref())?; }
        Ok(pending)
    }
    unsafe fn enqueue_attention_with_selection(&mut self,sink:Ds41rtDeviceBuffer,
        cache:&crate::v41_backbone_cache::CacheAttention<'_>,selection:Option<&IndexSelectionOutput<'_>>)
        ->Result<PendingLaneFfn<'_,'w,'a>> {
        ensure!(selection.is_some() == (self.layer >= 2), "attention index selection presence differs");
        let requests = cache.attention_requests();
        let query = self.query.output()?;
        let binding = query.binding()?;
        query.tokens()?;
        // Construct the guard before submitting so partial failures and unwinding
        // drain before any external query/cache/selection owner can be reused.
        let mut pending = PendingLaneFfn { lane: Some(self), values: None };
        let lane = pending.lane.as_deref_mut().unwrap();
        let query = lane.query.output()?;
        let mut tail = AttentionTail { projection: &mut lane.projection, block: &mut lane.block,
            binding, tokens: query.tokens()?, split_output:lane.tp2_output.is_some() };
        if unsafe { lane.sparse.as_mut().context("full-head attention wave absent")?
            .enqueue_query_prepared(&query, sink, &requests, selection, true, Some(&mut tail))? }.is_some() {
            pending.values = Some(lane.block.graph_normalized_storage(query.rows));
        }
        Ok(pending)
    }

    /// Produce learned index selections from this lane's completed query.
    /// # Safety
    /// Cache proposals correspond to the same admitted query batch, with all
    /// producers complete and no external writes racing these owners.
    pub unsafe fn select_index(
        &self,
        index: &mut crate::v41_index_lane::IndexLane<'_, '_>,
        cache: &crate::v41_backbone_cache::CacheAttention<'_>,
    ) -> Result<()> {
        let query = self.query_output()?;
        unsafe { index.select(&query, cache) }
    }
    /// Reuse the nearest learned selection, checking it against this cache batch.
    /// # Safety
    /// Same completed-producer and matching-sink contract as attention_cached_ffn.
    pub unsafe fn attention_indexed_ffn(
        &mut self,
        sink: Ds41rtDeviceBuffer,
        cache: &crate::v41_backbone_cache::CacheAttention<'_>,
        index: &crate::v41_index_lane::IndexLane<'_, '_>,
    ) -> Result<LaneFfn<'_, 'w, 'a>> {
        let selection = if self.layer >= 2 { Some(index.output(self.layer, cache)?) } else { None };
        unsafe { self.attention_cached_ffn(sink, cache, selection.as_ref()) }
    }
    /// Execute attention from one cache-bank batch. The bank view pins all
    /// request leases and proposal buffers through the completed attention call.
    /// # Safety
    /// Sink and selection match this layer; producers have drained and no
    /// external writes race the cache/query buffers (as for attention_ffn).
    pub unsafe fn attention_cached_ffn(
        &mut self,
        sink: Ds41rtDeviceBuffer,
        cache: &crate::v41_backbone_cache::CacheAttention<'_>,
        selection: Option<&IndexSelectionOutput<'_>>,
    ) -> Result<LaneFfn<'_, 'w, 'a>> {
        let requests = cache.attention_requests();
        unsafe { self.attention_ffn(sink, &requests, selection) }
    }
    /// # Safety
    /// Result is the completed shared plus routed reduction for the exact FFN
    /// binding and row order published by attention_ffn.
    pub unsafe fn finish_ffn(
        &mut self,
        binding: QueryBinding,
        result: Ds41rtDeviceBuffer,
    ) -> Result<BlockOutput<'_>> {
        self.enter(Phase::SharedReady)?;
        unsafe { self.block.finish_ffn(binding, result)?; }
        self.record_layer_finish();
        self.phase = Phase::Complete;
        self.block.output()
    }
    /// # Safety
    /// Same completed result and exclusive lane contract as finish_ffn, retained
    /// through cooperative completion or cancellation drain.
    pub async unsafe fn finish_ffn_cooperative(&mut self, binding: QueryBinding,
        result: Ds41rtDeviceBuffer) -> Result<BlockOutput<'_>> {
        self.enter(Phase::SharedReady)?;
        let handoff=self.weights.placement.is_some() && self.layer<39
            && self.weights.layers[self.layer+1].device.id != self.block.inputs()[0].device_id;
        unsafe { if handoff { self.block.finish_ffn_for_handoff_cooperative(binding,result).await?; }
            else { self.block.finish_ffn_cooperative(binding, result).await?; } }
        self.record_layer_finish();
        self.phase = Phase::Complete;
        self.block.output()
    }
    #[cfg(test)]
    pub async unsafe fn check_queued_finish(&mut self, binding: QueryBinding,
        result: Ds41rtDeviceBuffer) -> Result<()> {
        ensure!(self.phase == Phase::SharedReady, "test FFN phase differs");
        unsafe { self.block.check_queued_finish(binding, result).await }
    }
    pub fn output(&self) -> Result<BlockOutput<'_>> {
        ensure!(
            self.phase == Phase::Complete,
            "backbone lane output unavailable"
        );
        self.block.output()
    }
    #[cfg(test)]
    pub fn trace_stream(&self) -> *mut std::ffi::c_void { self.block.trace_stream() }
    pub fn set_route_capture(&mut self, enabled: bool) {
        self.capture_routes = enabled;
        if enabled {
            self.query.enable_small_graph_shapes();
            if let Some(sparse)=&mut self.sparse { sparse.enable_small_graph_shapes(); }
            if let Some(dual)=&mut self.dual_sparse { dual.enable_small_graph_shapes(); }
            self.projection.enable_small_graph_shapes();
            if let Some(shared) = &mut self.shared { shared.enable_small_graph_shapes(); }
            self.router.enable_small_graph_shapes();
            self.route_capture.resize_with(40, Vec::new);
            for rows in &mut self.route_capture { rows.clear(); }
            self.layer_events.recorded.fill(false);
            self.layer_events.entry_layer = None;
            self.ffn_split = FfnSplit::default();
        }
    }
    pub fn captured_routes(&self) -> &[Vec<[u32; 6]>] { &self.route_capture }
    /// Host FFN stage split of the last captured pass on this lane.
    pub fn captured_ffn_split(&self) -> FfnSplit { self.ffn_split }
    /// Opt-in TP2 attention or projection owners, whose peer transfers are
    /// host-ordered and therefore not stage-chained.
    pub fn uses_peer_projection(&self) -> bool {
        self.dual_sparse.is_some() || self.tp2_query.is_some() || self.tp2_output.is_some()
    }
    /// FFN completion instants of the captured pass, per layer.
    /// Device time from layer `from`'s FFN finish to layer `to`'s, for a
    /// captured pass whose work has completed.
    pub fn layer_elapsed_us(&self, from: usize, to: usize) -> Option<f64> {
        let events = &self.layer_events;
        if !(*events.recorded.get(from)? && *events.recorded.get(to)?) { return None; }
        unsafe { events.library.cuda_event_elapsed_ms(events.events[from], events.events[to]) }
            .ok().map(|ms| f64::from(ms) * 1e3)
    }
    /// Device time from the handoff arrival of `layer` to its FFN finish.
    pub fn entry_elapsed_us(&self, layer: usize) -> Option<f64> {
        let events = &self.layer_events;
        if events.entry_layer != Some(layer) || !*events.recorded.get(layer)? { return None; }
        unsafe { events.library.cuda_event_elapsed_ms(events.entry, events.events[layer]) }
            .ok().map(|ms| f64::from(ms) * 1e3)
    }
    /// Mark the arrival of the current layer's input from the other GPU.
    pub(crate) fn record_layer_entry(&mut self) {
        if !self.capture_routes { return; }
        let layer = self.layer;
        let event = self.layer_events.entry;
        if self.record_timing_event(event) { self.layer_events.entry_layer = Some(layer); }
    }
    fn record_timing_event(&self, event: *mut std::ffi::c_void) -> bool {
        let stream = self.block.finish_stream();
        let library = self.layer_events.library;
        if unsafe { library.cuda_event_record(event, stream) }.is_err() { return false; }
        // Unchained passes drained this stream and later rebinds require it
        // idle; the event completes at once, so wait for it here.
        if !crate::v41_memory::chain::active() {
            if let Err(error) = unsafe { library.cuda_event_synchronize(event) } {
                tracing::warn!(%error, "layer timing event wait failed");
            }
        }
        true
    }
    fn record_layer_finish(&mut self) {
        if !self.capture_routes { return; }
        let layer = self.layer;
        let event = self.layer_events.events[layer];
        if self.record_timing_event(event) {
            self.layer_events.recorded[layer] = true;
        }
    }
    pub fn route_capture_enabled(&self) -> bool { self.capture_routes }
    pub fn reserve_sparse_decode_rows(&mut self, rows: usize) -> Result<()> {
        if let Some(dual)=&mut self.dual_sparse { dual.reserve_decode_rows(rows) }
        else { self.sparse.as_mut().context("attention wave absent")?.reserve_decode_rows(rows) }
    }
    /// Replace full-head storage before execution and before final KV sizing.
    /// The explicit budget includes both local and peer workspaces.
    pub fn enable_dual_attention(&mut self,peer:Device<'a>,budgets:[usize;2])->Result<()> {
        ensure!(self.phase==Phase::Idle && self.dual_sparse.is_none(),"dual attention requires an unused lane");
        let source=Device { library:self.weights.library,id:self.query.input().device_id };
        let mut dual=crate::v41_sparse_attention::dual::DualAttentionWave::new([source,peer],self.capacity,budgets)?;
        if self.capture_routes { dual.enable_small_graph_shapes(); }
        source.run(|| { self.sparse=None; Ok(()) })?;
        self.dual_sparse=Some(dual);Ok(())
    }
    pub fn dual_attention_bytes(&self)->Result<[usize;2]> {
        crate::v41_sparse_attention::dual::DualAttentionWave::device_bytes(self.capacity)
    }
    /// Reserve adaptive route history during planning, before lane execution.
    pub fn reserve_route_capture(&mut self, layers: std::ops::Range<usize>, rows: usize) -> Result<()> {
        ensure!(layers.end <= 40 && rows <= 4096, "route history reservation exceeds model bounds");
        self.route_capture.resize_with(40, Vec::new);
        for layer in layers {
            self.route_capture[layer].reserve(rows);
        }
        Ok(())
    }
    /// Opt-in diagnostic at a completed layer boundary; never used by normal serving.
    pub fn trace_output(&self, directory: &std::path::Path) -> Result<()> {
        use std::io::Write;
        let output = self.output()?;
        for (name, buffer) in [("residual", output.residual), ("pre", output.pre)] {
            let path = directory.join(format!("layer{}-{name}.bin", output.layer));
            let mut file = std::fs::OpenOptions::new().write(true).create_new(true).open(path)?;
            let mut bytes = vec![0; buffer.bytes];
            self.weights.library.copy_d2h(&mut bytes, buffer)?;
            file.write_all(&bytes)?;
        }
        Ok(())
    }
    pub fn advance(&mut self) -> Result<()> {
        self.enter(Phase::Complete)?;
        ensure!(self.layer < 39, "backbone lane is at final layer");
        let next = &self.weights.layers[self.layer + 1];
        ensure!(next.device.id == self.block.inputs()[0].device_id, "next layer needs a peer handoff");
        self.query.rebind(&next.query)?;
        self.projection.rebind(&next.projection)?;
        if let Some(shared) = &mut self.shared { shared.rebind(next.shared.as_ref().context("local shared weights absent")?)?; }
        self.router.rebind(&next.router)?;
        self.block.advance(&next.hc)?;
        self.layer += 1;
        self.phase = Phase::Prepared;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ds41rt_loader::{read_official_v41_catalog, OFFICIAL_V41_MODEL_ID};

    #[test]
    fn lane_ffn_cancellation_and_failure_do_not_publish_completion() {
        use std::{
            future::Future,
            task::{Context, Poll, Waker},
        };
        let mut context = Context::from_waker(Waker::noop());
        let mut phase = Phase::Ffn;
        // An unpolled future has dispatched nothing and leaves work available.
        drop(complete_ffn(
            &mut phase,
            std::future::pending::<Result<()>>(),
        ));
        assert_eq!(phase, Phase::Ffn);
        let mut work = Box::pin(complete_ffn(
            &mut phase,
            std::future::pending::<Result<()>>(),
        ));
        assert!(work.as_mut().poll(&mut context).is_pending());
        drop(work);
        assert_eq!(phase, Phase::Invalid);
        let touched = std::cell::Cell::new(false);
        let mut work = Box::pin(complete_ffn(&mut phase, async {
            touched.set(true);
            Ok(())
        }));
        assert!(matches!(
            work.as_mut().poll(&mut context),
            Poll::Ready(Err(_))
        ));
        drop(work);
        assert!(!touched.get());
        phase = Phase::Ffn;
        let mut work = Box::pin(complete_ffn(&mut phase, async {
            anyhow::bail!("shared/reduction failure")
        }));
        let result: Poll<Result<()>> = work.as_mut().poll(&mut context);
        assert!(matches!(result, Poll::Ready(Err(_))));
        drop(work);
        assert_eq!(phase, Phase::Invalid);
        phase = Phase::Ffn;
        let mut work = Box::pin(complete_ffn(&mut phase, async { Ok(17) }));
        assert!(matches!(
            work.as_mut().poll(&mut context),
            Poll::Ready(Ok(17))
        ));
        drop(work);
        assert_eq!(phase, Phase::SharedReady);
    }

    #[test]
    fn official_lane_budget_rejects_before_device_allocation() -> Result<()> {
        let Some(path) = std::env::var_os("DS41RT_LANE_PLAN_LIBRARY") else {
            eprintln!("skip lane planning test: DS41RT_LANE_PLAN_LIBRARY unset");
            return Ok(());
        };
        let model = std::env::var_os("DS41RT_LANE_PLAN_MODEL")
            .ok_or_else(|| anyhow::anyhow!("DS41RT_LANE_PLAN_MODEL required"))?;
        let library = unsafe { NativeLibrary::load(path)? };
        let catalog =
            read_official_v41_catalog(OFFICIAL_V41_MODEL_ID, std::path::Path::new(&model))?;
        // Checkpoint payload extents plus the native AOT packed-scale metadata.
        let weight_bytes = BackboneLaneWeights::device_bytes(&library, &catalog)?;
        assert_eq!(weight_bytes, 6_896_423_360);
        let rejected = BackboneLaneWeights::load(&library, &catalog, weight_bytes - 1, 1024 * 1024);
        assert!(rejected
            .err()
            .unwrap()
            .to_string()
            .contains("weights exceed budget"));
        let empty = BackboneLaneWeights {
            library: &library,
            layers: vec![],
            placement: None,
            split_query_b: false,
            split_output_b: false,
        };
        // Native AOT variants supply scratch extents; validate budget boundaries
        // against this library instead of totals from an older export build.
        for capacity in [1, 80, 4096] {
            let groups = BackboneLane::workspace_bytes(&library, capacity)?;
            let total: usize = groups.iter().sum();
            // Both paths must reject without attempting a CUDA allocation. The
            // CPU-only qualification container exposes no GPU devices.
            let rejected = BackboneLane::new(&empty, capacity, total - 1);
            assert!(rejected
                .err()
                .unwrap()
                .to_string()
                .contains("lane exceeds budget"));
            let rejected = BackboneLane::new(&empty, capacity, total);
            assert!(rejected
                .err()
                .unwrap()
                .to_string()
                .contains("all 40 layers"));
            eprintln!(
                "PASS capacity={capacity} groups={groups:?} workspace_bytes={total} budget_guard"
            );
        }
        for capacity in [0, 4097, u32::MAX] {
            assert!(BackboneLane::workspace_bytes(&library, capacity).is_err());
        }
        eprintln!("PASS official forty-layer weight budget={weight_bytes}; CPU-only planning and allocation guards");
        Ok(())
    }
}
