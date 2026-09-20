//! One coordinator wave owns TP route planes through final native reduction.
use super::{DeviceAllocation, LoadStream};
use anyhow::{ensure, Context, Result};
use ds41rt_core::{
    replicated_expert_tie_seed_for, ReplicatedExpertCostModel, ReplicatedExpertScheduleConfig,
    ReplicatedExpertScheduler, ReplicatedExpertTieSeedMode, INACTIVE_REPLICATED_EXPERT_GROUP,
};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary, V41CompactReducer};
use ds41rt_transport::{
    v41_expert::{V41SparkTopology, V41Tp4RocePending, V41Tp4Roce, V41_PARTIAL_ROW_BYTES,
        V41_ROUTED_EXPERTS},
    ExpertProtocolV2Request, VerbsHostProtocolV2ResponsePayload,
};

/// One replicated lane's whole-expert ownership planner. Preallocated once per
/// independent lane; every field is lane-local, so two lanes can never perturb
/// each other's assignment. The histogram and encoded owners are reused across
/// layers and requests without allocating in the request path.
pub(crate) struct ReplicatedGroupPlanner {
    topology: V41SparkTopology,
    tie_seed_mode: ReplicatedExpertTieSeedMode,
    scheduler: ReplicatedExpertScheduler,
    histogram: [u32; V41_ROUTED_EXPERTS],
    owners: [u8; V41_ROUTED_EXPERTS],
}

/// Opt-in tie-seed control. Unset selects `dispatch`, which reproduces the
/// historical per-dispatch-counter seed byte for byte. `layer` is the narrow
/// diagnostic mode that pins the request component so repeats of the same work
/// at the same layer share ownership when their histogram is unchanged.
pub(crate) const REPLICATED_EXPERT_TIE_SEED_ENV: &str = "DS41RT_REPLICATED_EXPERT_TIE_SEED";

/// Whole-expert weight-only profile: one positive weight unit per active expert
/// and no per-row activation term, so ownership depends only on which experts a
/// batch routed to. `tile_rows = 16` keeps the model valid without claiming a
/// calibrated tile cost; supplying calibrated costs is an explicit opt-in
/// environment override, never an invented profile.
fn parse_replicated_cost(value: &str) -> Result<ReplicatedExpertCostModel> {
    let parts = value.split(',').map(str::trim).collect::<Vec<_>>();
    ensure!(
        parts.len() == 3,
        "DS41RT_REPLICATED_EXPERT_COST expects weight,tile_cost,tile_rows"
    );
    let weight = parts[0].parse::<u64>().context("replicated expert weight cost")?;
    let tile_cost = parts[1].parse::<u64>().context("replicated expert tile cost")?;
    let tile_rows = parts[2].parse::<u32>().context("replicated expert tile rows")?;
    let model = ReplicatedExpertCostModel::new(weight, tile_cost, tile_rows);
    model
        .validate()
        .map_err(|error| anyhow::anyhow!("invalid replicated expert cost model: {error}"))?;
    Ok(model)
}

fn replicated_cost_model() -> Result<ReplicatedExpertCostModel> {
    match std::env::var_os("DS41RT_REPLICATED_EXPERT_COST") {
        None => Ok(ReplicatedExpertCostModel::new(1, 0, 16)),
        Some(value) => parse_replicated_cost(
            value
                .to_str()
                .context("DS41RT_REPLICATED_EXPERT_COST is not valid UTF-8")?,
        ),
    }
}

/// Resolve the opt-in tie-seed mode once, at planner construction. Missing
/// environment selects the historical `dispatch` mode; every present value must
/// be exactly `dispatch` or `layer`, so a typo fails closed before any request.
fn replicated_tie_seed_mode() -> Result<ReplicatedExpertTieSeedMode> {
    let value = match std::env::var_os(REPLICATED_EXPERT_TIE_SEED_ENV) {
        None => None,
        Some(raw) => Some(
            raw.into_string()
                .map_err(|_| anyhow::anyhow!("{REPLICATED_EXPERT_TIE_SEED_ENV} is not valid UTF-8"))?,
        ),
    };
    ReplicatedExpertTieSeedMode::from_env_value(value.as_deref())
        .map_err(|error| anyhow::anyhow!("{error}"))
}

impl ReplicatedGroupPlanner {
    fn new(topology: V41SparkTopology) -> Result<Self> {
        Self::new_with_seed_mode(topology, replicated_tie_seed_mode()?)
    }

    /// Constructor with an explicit mode, used by focused tests so they never
    /// mutate process environment. Logs the resolved mode once per process.
    fn new_with_seed_mode(
        topology: V41SparkTopology,
        tie_seed_mode: ReplicatedExpertTieSeedMode,
    ) -> Result<Self> {
        static TIE_SEED_MODE_LOGGED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        TIE_SEED_MODE_LOGGED.get_or_init(|| {
            tracing::info!(
                env = REPLICATED_EXPERT_TIE_SEED_ENV,
                mode = tie_seed_mode.as_str(),
                "replicated expert tie-seed mode resolved"
            );
        });
        let config = ReplicatedExpertScheduleConfig::new(
            topology.group_count(),
            replicated_cost_model()?,
        );
        let scheduler = ReplicatedExpertScheduler::new(config, V41_ROUTED_EXPERTS)
            .map_err(|error| anyhow::anyhow!("replicated expert scheduler rejected: {error}"))?;
        Ok(Self {
            topology,
            tie_seed_mode,
            scheduler,
            histogram: [0; V41_ROUTED_EXPERTS],
            owners: [INACTIVE_REPLICATED_EXPERT_GROUP; V41_ROUTED_EXPERTS],
        })
    }

    #[cfg(test)]
    fn seed_mode(&self) -> ReplicatedExpertTieSeedMode {
        self.tie_seed_mode
    }

    pub(crate) fn topology(&self) -> V41SparkTopology {
        self.topology
    }

    /// Plan whole-expert ownership for one request from the router's host-side
    /// route IDs (no device read and no allocation), copy the encoded assignment
    /// into the request-owned route words, and prove the ownership contract
    /// before any bytes travel. Planning exactly once per request keeps the
    /// assignment reproducible from `(layer, request_id, histogram)`.
    pub(crate) fn encode(&mut self, request: &mut ExpertProtocolV2Request) -> Result<()> {
        ensure!(
            request.header.row_count > 0
                && request.header.row_count <= 4096
                && request.routes.len() == request.header.row_count as usize * 6,
            "replicated expert ownership needs one canonical bounded request"
        );
        self.histogram.fill(0);
        for route in &request.routes {
            let expert = route.expert_id as usize;
            ensure!(
                expert < V41_ROUTED_EXPERTS,
                "replicated expert route id out of range"
            );
            self.histogram[expert] += 1;
        }
        let seed = replicated_expert_tie_seed_for(
            self.tie_seed_mode,
            request.header.layer_id,
            request.header.request_id,
        );
        {
            let plan = self
                .scheduler
                .plan(&self.histogram, seed)
                .map_err(|error| anyhow::anyhow!("replicated expert schedule rejected: {error}"))?;
            ensure!(
                plan.group_count() == self.topology.group_count(),
                "replicated expert plan group count differs from the bound topology"
            );
            self.owners.fill(INACTIVE_REPLICATED_EXPERT_GROUP);
            for (expert, group) in plan.assignment().iter().enumerate() {
                // Every routed expert is active, so it carries an active owner;
                // the transport rejects inactive or out-of-range owners.
                self.owners[expert] = group.encoded();
            }
        }
        request.with_native_group_owners(&self.owners, self.topology)
    }
}

pub(crate) struct NativeTp4Wave<'a> {
    transport: V41Tp4Roce,
    // Drop drains the stream before fields release any GPU allocations.
    stream: LoadStream<'a>,
    planes: Vec<DeviceAllocation<'a>>,
    upload_frames: Vec<VerbsHostProtocolV2ResponsePayload>,
    shared: DeviceAllocation<'a>,
    output: DeviceAllocation<'a>,
    library: &'a NativeLibrary,
    reducer: V41CompactReducer<'a>,
    ready_rows: Option<u32>,
    local: Option<super::local::LocalExpertWave<'a>>,
    tp2: Option<Box<super::tp2_ffn::Wave<'a>>>,
    paired: Option<Box<(super::paired::PairedAssignment, std::rc::Rc<super::paired::PairedProfile>)>>,
    /// Present only for an explicit `TP×EP` topology; legacy transports stay on
    /// the canonical request contract.
    native: Option<ReplicatedGroupPlanner>,
}
impl<'a> NativeTp4Wave<'a> {
    pub(crate) fn spark_world(&self) -> usize { self.transport.world_size() }
    /// Explicit replicated topology of this lane, or `None` for the legacy path.
    pub(crate) fn native_topology(&self) -> Option<V41SparkTopology> {
        self.native.as_ref().map(ReplicatedGroupPlanner::topology)
    }
    pub(crate) fn install_paired(&mut self, profile: std::rc::Rc<super::paired::PairedProfile>) -> Result<()> {
        ensure!(self.paired.is_none(), "paired assignment already installed");
        ensure!(self.native.is_none(), "paired EXL3 and replicated groups are mutually exclusive");
        self.paired = Some(Box::new((super::paired::PairedAssignment::new(), profile)));
        Ok(())
    }
    /// Apply the lane's whole-expert ownership to one remote request. Legacy
    /// lanes are a no-op; replicated lanes re-encode the route words and set the
    /// native group flag, so every remote path that dispatches through this wave
    /// carries the same contract.
    pub(crate) fn prepare_remote_request(&mut self, request: &mut crate::v41_backbone_router::BoundExpertRequest) -> Result<()> {
        if let Some(paired) = &mut self.paired {
            ensure!(self.native.is_none(), "paired EXL3 and replicated groups are mutually exclusive");
            request.assign_paired(&mut paired.0, &paired.1)?;
        } else if let Some(native) = &mut self.native {
            request.assign_native(native)?;
        }
        Ok(())
    }

    pub fn install_local(&mut self, wave: super::local::LocalExpertWave<'a>) -> Result<()> {
        ensure!(self.local.is_none() && self.tp2.is_none(), "local expert lane already installed");
        self.local = Some(wave);
        Ok(())
    }
    pub fn has_local_layer(&self, layer: usize) -> bool {
        self.local.as_ref().is_some_and(|wave| wave.contains(layer)) || self.has_tp2_layer(layer)
    }
    pub fn install_tp2(&mut self, wave: super::tp2_ffn::Wave<'a>) -> Result<()> {
        ensure!(self.local.is_none() && self.tp2.is_none(), "local expert lane already installed");
        self.tp2 = Some(Box::new(wave));
        Ok(())
    }
    pub fn has_tp2_layer(&self, layer: usize) -> bool {
        self.tp2.as_ref().is_some_and(|wave| wave.contains(layer))
    }
    pub fn has_tp2_shared_layer(&self, layer: usize) -> bool {
        self.tp2.as_ref().is_some_and(|wave| wave.contains_shared(layer))
    }
    /// # Safety
    /// Input and router producers have completed. Both borrowed outputs remain
    /// immutable through this operation, including cancellation draining.
    pub async unsafe fn execute_tp2_ffn(&mut self, input: &crate::v41_block::FfnInput<'_>,
        routed: &crate::v41_backbone_router::RouterOutput<'_>) -> Result<NativeFfnOutput<'_>> {
        self.ready_rows = None;
        let binding = routed.binding()?;
        ensure!(binding == input.binding() && input.layer == routed.layer
            && input.tokens.len() == routed.rows as usize && input.tokens == routed.tokens,
            "TP2 FFN input/router identity mismatch");
        let values = unsafe { self.tp2.as_mut().context("TP2 expert lane missing")?
            .execute(input.layer, routed.rows, input.values, routed.expert_input, routed.ids, routed.routing).await? };
        self.ready_rows = Some(routed.rows);
        Ok(NativeFfnOutput { values, binding, _owner: std::marker::PhantomData })
    }
    /// # Safety
    /// Completed router/shared buffers stay live and unmodified through drain.
    pub unsafe fn execute_local_ffn(&mut self,
        routed: &crate::v41_backbone_router::RouterOutput<'_>,
        shared: &crate::v41_backbone_shared::SharedOutput<'_>) -> Result<NativeFfnOutput<'_>> {
        self.ready_rows = None;
        let values = unsafe { self.local.as_mut().context("local expert lane missing")?.execute(routed, shared)? };
        self.ready_rows = Some(routed.rows);
        Ok(NativeFfnOutput { values, binding: routed.binding()?, _owner: std::marker::PhantomData })
    }
    /// # Safety
    /// Completed router/shared inputs remain immutable until completion or drain.
    pub async unsafe fn execute_local_ffn_cooperative(&mut self,
        routed: &crate::v41_backbone_router::RouterOutput<'_>,
        shared: &crate::v41_backbone_shared::SharedOutput<'_>) -> Result<NativeFfnOutput<'_>> {
        self.ready_rows = None;
        let binding = routed.binding()?;
        let values = unsafe { self.local.as_mut().context("local expert lane missing")?
            .execute_cooperative(routed, shared).await? };
        self.ready_rows = Some(routed.rows);
        Ok(NativeFfnOutput { values, binding, _owner: std::marker::PhantomData })
    }
    /// Admission invalidates prior output; completed RoCE sessions remain reusable.
    pub fn begin_request(&mut self) {
        self.ready_rows = None;
    }
    pub fn reset_connections(&mut self) {
        self.ready_rows = None;
        self.transport.reset_connections();
    }
    /// Legacy TP4 upper-bound reservation (four rank planes), also sufficient
    /// for a legacy TP2 transport.
    pub fn device_bytes(capacity: u32) -> Result<usize> {
        Self::device_bytes_for(capacity, 4)
    }
    /// Reservation for exactly `ranks` physical rank planes plus the shared
    /// input and final output planes. Six-rank layouts need their own, larger
    /// reservation; reusing the four-rank reserve undercounts by two planes.
    pub fn device_bytes_for(capacity: u32, ranks: usize) -> Result<usize> {
        ensure!(
            capacity > 0 && capacity <= 4096,
            "invalid native TP wave capacity"
        );
        ensure!(
            matches!(ranks, 2 | 3 | 4 | 6),
            "native TP wave requires two, three, four or six ranks"
        );
        (capacity as usize)
            .checked_mul(ranks * V41_PARTIAL_ROW_BYTES as usize + 2 * 5120 * 2)
            .context("native TP wave budget overflow")
    }
    pub fn new(
        library: &'a NativeLibrary,
        transport: V41Tp4Roce,
        available_bytes: usize,
    ) -> Result<Self> {
        let capacity = transport.capacity();
        let world_size = transport.world_size();
        ensure!(
            matches!(world_size, 2 | 3 | 4 | 6),
            "native TP wave requires two, three, four or six ranks"
        );
        ensure!(
            Self::device_bytes_for(capacity, world_size)? <= available_bytes,
            "native TP wave exceeds device budget"
        );
        let reducer = library.v41_compact_reducer()?;
        // Prove the exact physical-rank reduction is available before any plane
        // or output allocation. Legacy two-rank transports also fail here rather
        // than at their first request.
        reducer.require_rank_count(world_size as u32)?;
        let plane_bytes = capacity as usize * V41_PARTIAL_ROW_BYTES as usize;
        let mut planes = Vec::with_capacity(world_size);
        for _ in 0..world_size {
            planes.push(DeviceAllocation::new(library, plane_bytes)?);
        }
        let native = transport
            .topology()
            .map(ReplicatedGroupPlanner::new)
            .transpose()?;
        let hidden_bytes = capacity as usize * 5120 * 2;
        Ok(Self {
            transport,
            stream: LoadStream {
                library,
                raw: library.cuda_stream_create()?,
            },
            planes,
            upload_frames: Vec::with_capacity(capacity as usize * world_size),
            shared: DeviceAllocation::new(library, hidden_bytes)?,
            output: DeviceAllocation::new(library, hidden_bytes)?,
            library,
            reducer,
            ready_rows: None,
            local: None,
            tp2: None,
            paired: None,
            native,
        })
    }
    /// RoCE execution with optional host BF16 shared-expert contribution.
    /// All GPU copies finish before frame storage can be reused, and reduction
    /// finishes before the borrowed output view is exposed. Cancellation leaves
    /// output unavailable until an entirely successful subsequent execution.
    pub async fn execute(
        &mut self,
        request: &ExpertProtocolV2Request,
        shared: Option<&[u8]>,
    ) -> Result<Ds41rtDeviceBuffer> {
        self.ready_rows = None;
        self.synchronize()?;
        let rows = request.header.row_count;
        ensure!(
            rows > 0 && rows <= self.transport.capacity(),
            "native wave exceeds capacity"
        );
        let hidden_bytes = rows as usize * 5120 * 2;
        if let Some(shared) = shared {
            ensure!(
                shared.len() == hidden_bytes,
                "native shared output has wrong BF16 geometry"
            );
            self.library.copy_h2d(self.shared.buffer, shared)?;
        }
        self.execute_prepared(request, shared.is_some()).await
    }
    /// # Safety
    /// Shared device values are complete and immutable through the copy. The
    /// request and shared result must derive from the same actual block input.
    pub async unsafe fn execute_ffn<'w>(
        &'w mut self,
        request: &crate::v41_backbone_router::BoundExpertRequest,
        shared: &crate::v41_backbone_shared::SharedOutput<'_>,
    ) -> Result<NativeFfnOutput<'w>> {
        self.ready_rows = None;
        validate_shared(
            request,
            shared,
            self.shared.buffer,
            self.transport.capacity(),
        )?;
        unsafe { self.dispatch_ffn(request).await?.finish(shared).await }
    }
    /// Enqueue all TP rank expert requests before returning. The caller can then run
    /// shared FFN work on RTX while the Spark workers execute the routed experts.
    pub async fn dispatch_ffn<'w, 'r>(
        &'w mut self,
        request: &'r crate::v41_backbone_router::BoundExpertRequest,
    ) -> Result<NativePendingFfn<'w, 'a, 'r>> {
        self.ready_rows = None;
        // Previous output or cancellation cleanup completed this wave.
        self.stream.require_complete()?;
        let header = &request.request().header;
        ensure!(
            header.layer_id as usize == request.binding().layer()
                && header.row_count > 0
                && header.row_count <= self.transport.capacity(),
            "native dispatched FFN identity or rows differ"
        );
        let capacity = self.transport.capacity();
        let pending = self.transport.dispatch(request.request()).await?;
        Ok(NativePendingFfn {
            pending,
            request,
            capacity,
            library: self.library,
            stream: &self.stream,
            planes: &self.planes,
            upload_frames: &mut self.upload_frames,
            shared: self.shared.buffer,
            output: self.output.buffer,
            reducer: &self.reducer,
            ready_rows: &mut self.ready_rows,
            tp2: self.tp2.as_deref_mut(),
        })
    }

    async fn execute_prepared(
        &mut self,
        request: &ExpertProtocolV2Request,
        has_shared: bool,
    ) -> Result<Ds41rtDeviceBuffer> {
        let rows = request.header.row_count;
        let library = self.library;
        let planes = &self.planes;
        self.transport
            .execute(request, |rank, first_row, bytes| {
                copy_chunk(library, planes, rank, first_row, bytes)
            })
            .await?;
        reduce_planes(
            self.library,
            &self.reducer,
            &self.stream,
            &self.planes,
            self.output.buffer,
            has_shared.then_some(self.shared.buffer),
            rows,
        )?;
        self.ready_rows = Some(rows);
        self.output()
    }
    /// Borrowed device view; never free it or retain it across wave reuse/drop.
    pub fn output(&self) -> Result<Ds41rtDeviceBuffer> {
        let rows = self
            .ready_rows
            .context("native TP wave output is not complete")?;
        let mut output = self.output.buffer;
        output.bytes = rows as usize * 5120 * 2;
        Ok(output)
    }
    pub fn synchronize(&self) -> Result<()> {
        unsafe { self.library.cuda_stream_synchronize(self.stream.raw) }
    }
}
impl Drop for NativeTp4Wave<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.synchronize() {
            tracing::error!(%error, "draining native coordinator TP wave");
        }
    }
}

/// Complete ordered TP reduction plus the shared expert, borrowed until consumed.
pub(crate) struct NativeFfnOutput<'a> {
    pub values: Ds41rtDeviceBuffer,
    binding: crate::v41_attention_binding::QueryBinding,
    _owner: std::marker::PhantomData<&'a ()>,
}
impl NativeFfnOutput<'_> {
    pub fn binding(&self) -> crate::v41_attention_binding::QueryBinding {
        self.binding
    }
}

fn validate_shared(
    request: &crate::v41_backbone_router::BoundExpertRequest,
    shared: &crate::v41_backbone_shared::SharedOutput<'_>,
    destination: Ds41rtDeviceBuffer,
    capacity: u32,
) -> Result<()> {
    let header = &request.request().header;
    ensure!(
        request.binding() == shared.binding()?
            && header.layer_id as usize == shared.layer
            && header.row_count == shared.rows
            && header.row_count > 0
            && header.row_count <= capacity
            && shared.values.bytes == header.row_count as usize * 10240
            && shared.values.device_id == destination.device_id,
        "native TP shared contribution differs from routed request"
    );
    Ok(())
}
fn copy_chunk(
    library: &NativeLibrary,
    planes: &[DeviceAllocation<'_>],
    rank: usize,
    first_row: u32,
    bytes: &[u8],
) -> Result<()> {
    library.copy_h2d(chunk_destination(planes, rank, first_row, bytes.len())?, bytes)
}
fn chunk_destination(
    planes: &[DeviceAllocation<'_>],
    rank: usize,
    first_row: u32,
    bytes: usize,
) -> Result<Ds41rtDeviceBuffer> {
    ensure!(rank < planes.len(), "native route rank exceeds TP plane count");
    let offset = (first_row as usize)
        .checked_mul(V41_PARTIAL_ROW_BYTES as usize)
        .context("native route chunk offset overflow")?;
    let end = offset
        .checked_add(bytes)
        .context("native route chunk extent overflow")?;
    ensure!(
        end <= planes[rank].buffer.bytes,
        "native route chunk exceeds destination"
    );
    let mut destination = planes[rank].buffer;
    destination.ptr = unsafe { destination.ptr.cast::<u8>().add(offset).cast() };
    destination.bytes = bytes;
    Ok(destination)
}

/// Retain received pinned frames until uploads finish. Drop also drains on
/// cancellation or errors, before recycling their host storage.
struct PlaneUploads<'s, 'a> {
    library: &'a NativeLibrary,
    stream: &'s LoadStream<'a>,
    planes: &'s [DeviceAllocation<'a>],
    frames: &'s mut Vec<VerbsHostProtocolV2ResponsePayload>,
    rows: u32,
    pending: bool,
}
impl PlaneUploads<'_, '_> {
    fn copy(&mut self, rank: usize, first_row: u32, payload: VerbsHostProtocolV2ResponsePayload) -> Result<()> {
        let bytes = payload.as_ref();
        let destination = chunk_destination(self.planes, rank, first_row, bytes.len())?;
        let offset = first_row as usize * V41_PARTIAL_ROW_BYTES as usize;
        ensure!(offset + bytes.len() <= self.rows as usize * 10240,
            "native upload exceeds live rows");
        if payload.pinned_host_buffer().is_none() {
            return self.library.copy_h2d(destination, bytes);
        }
        ensure!(self.frames.len() < self.frames.capacity(), "native retained frame capacity exhausted");
        // Transfer ownership before enqueue, including a possible partial enqueue error.
        self.frames.push(payload);
        self.pending = true;
        unsafe { self.library.copy_h2d_async(destination, self.frames.last().unwrap().as_ref(), self.stream.raw) }
    }
}
impl Drop for PlaneUploads<'_, '_> {
    fn drop(&mut self) {
        if self.pending {
            if let Err(error) = unsafe { self.library.cuda_stream_synchronize(self.stream.raw) } {
                tracing::error!(%error, "draining interrupted native rank uploads");
            }
        }
        self.frames.clear();
    }
}
unsafe fn enqueue_reduce_planes(reducer: &V41CompactReducer<'_>, stream: &LoadStream<'_>,
    planes: &[DeviceAllocation<'_>], output: Ds41rtDeviceBuffer,
    shared: Option<Ds41rtDeviceBuffer>, rows: u32) -> Result<()> {
    let shared = shared.map_or(std::ptr::null(), |b| b.ptr.cast());
    match planes.len() {
        // The 2- and 4-plane entry points are kept bit-identical for the legacy
        // paths; the generic N-plane entry point carries 3- and 6-rank
        // replicated layouts (and is ordered identically for 2/4).
        2 => unsafe { reducer.reduce_tp2(
            std::array::from_fn(|rank| planes[rank].buffer.ptr.cast::<u16>().cast_const()),
            shared, output.ptr.cast(), rows, stream.raw) },
        4 => unsafe { reducer.reduce(
            std::array::from_fn(|rank| planes[rank].buffer.ptr.cast::<u16>().cast_const()),
            shared, output.ptr.cast(), rows, stream.raw) },
        3 | 6 => unsafe { reducer.reduce_planes(
            std::array::from_fn(|rank| planes.get(rank)
                .map_or(std::ptr::null(), |plane| plane.buffer.ptr.cast::<u16>().cast_const())),
            planes.len() as u32, shared, output.ptr.cast(), rows, stream.raw) },
        _ => anyhow::bail!("native compact reduction requires two, three, four or six TP planes"),
    }
}
fn reduce_planes(library: &NativeLibrary, reducer: &V41CompactReducer<'_>,
    stream: &LoadStream<'_>, planes: &[DeviceAllocation<'_>], output: Ds41rtDeviceBuffer,
    shared: Option<Ds41rtDeviceBuffer>, rows: u32) -> Result<()> {
    let launched = unsafe { enqueue_reduce_planes(reducer, stream, planes, output, shared, rows) };
    launched.and(unsafe { library.cuda_stream_synchronize(stream.raw) })
}
/// Retain planes, upload frames, output and shared input through completion.
async unsafe fn reduce_planes_cooperative(reducer: &V41CompactReducer<'_>,
    stream: &LoadStream<'_>, planes: &[DeviceAllocation<'_>], output: Ds41rtDeviceBuffer,
    shared: Option<Ds41rtDeviceBuffer>, rows: u32) -> Result<()> {
    let launched = unsafe { enqueue_reduce_planes(reducer, stream, planes, output, shared, rows) };
    let drained = stream.wait().await;
    launched.and(drained)
}

/// Borrows every mutable reduction buffer and owns all unread response sockets.
/// Dropping before completion leaves the wave unpublished and closes the sockets.
pub(crate) struct NativePendingFfn<'w, 'a, 'r> {
    pending: V41Tp4RocePending<'w, 'r>,
    request: &'r crate::v41_backbone_router::BoundExpertRequest,
    capacity: u32,
    library: &'a NativeLibrary,
    stream: &'w LoadStream<'a>,
    planes: &'w [DeviceAllocation<'a>],
    upload_frames: &'w mut Vec<VerbsHostProtocolV2ResponsePayload>,
    shared: Ds41rtDeviceBuffer,
    output: Ds41rtDeviceBuffer,
    reducer: &'w V41CompactReducer<'a>,
    ready_rows: &'w mut Option<u32>,
    tp2: Option<&'w mut super::tp2_ffn::Wave<'a>>,
}
impl<'w> NativePendingFfn<'w, '_, '_> {
    /// # Safety
    /// Shared values hold the completed contribution for this exact request and
    /// remain immutable until the final reduction drains. Producers must be drained.
    pub async unsafe fn finish(
        self,
        shared: &crate::v41_backbone_shared::SharedOutput<'_>,
    ) -> Result<NativeFfnOutput<'w>> {
        unsafe { self.finish_inner(shared, false).await }
    }
    /// # Safety
    /// Same input retention as finish; cancellation drains before releasing frames.
    pub async unsafe fn finish_cooperative(self,
        shared: &crate::v41_backbone_shared::SharedOutput<'_>) -> Result<NativeFfnOutput<'w>> {
        unsafe { self.finish_inner(shared, true).await }
    }
    /// Run shared TP2 after Spark dispatch, reduce on the transport GPU, and
    /// return the result to the block's GPU. The pending owner retains every lane workspace.
    /// # Safety
    /// Input is the completed normalized FFN input for the dispatched request;
    /// its storage remains immutable through completion or cancellation drain.
    pub async unsafe fn finish_tp2(mut self, input: &crate::v41_block::FfnInput<'_>) -> Result<NativeFfnOutput<'w>> {
        let header = &self.request.request().header;
        ensure!(self.request.binding() == input.binding() && header.layer_id as usize == input.layer
            && header.row_count as usize == input.tokens.len()
            && matches!(input.values.device_id, 0 | 1),
            "TP2 shared input/request/device differs");
        let values = unsafe { self.tp2.as_mut().context("TP2 shared workspace missing")?
            .execute_shared_on(input.layer,header.row_count,input.values,self.output.device_id as usize).await? };
        let destination = input.values.device_id;
        if destination != self.output.device_id {
            let device = crate::v41_memory::device::Device { library: self.library, id: self.output.device_id };
            device.future(unsafe { self.finish_values(values,true,Some(destination)) }).await
        } else {
            unsafe { self.finish_values(values,true,Some(destination)).await }
        }
    }
    async unsafe fn finish_inner(self,
        shared: &crate::v41_backbone_shared::SharedOutput<'_>, cooperative: bool) -> Result<NativeFfnOutput<'w>> {
        validate_shared(self.request, shared, self.shared, self.capacity)?;
        unsafe { self.finish_values(shared.values,cooperative,None).await }
    }
    async unsafe fn finish_values(mut self, values: Ds41rtDeviceBuffer, cooperative: bool,
        destination: Option<i32>) -> Result<NativeFfnOutput<'w>> {
        ensure!(values.device_id == self.output.device_id
            && values.bytes == self.request.request().header.row_count as usize * 10240,
            "shared reduction device/extent differs");
        let timing = std::time::Instant::now();
        // The shared owner remains borrowed until reduction drains, so consume
        // its completed device output directly instead of copying it first.
        let shared_copy_us = 0u64;
        let mut uploads = PlaneUploads {
            library: self.library,
            stream: self.stream,
            planes: self.planes,
            frames: self.upload_frames,
            rows: self.request.request().header.row_count,
            pending: false,
        };
        let mut upload_us = 0u64;
        self.pending
            .receive_owned(|rank, first_row, payload| {
                let copy_start = std::time::Instant::now();
                let result = uploads.copy(rank, first_row, payload);
                upload_us += copy_start.elapsed().as_micros() as u64;
                result
            })
            .await?;
        let received_us = timing.elapsed().as_micros() as u64;
        let rows = self.request.request().header.row_count;
        if cooperative {
            unsafe { reduce_planes_cooperative(self.reducer, self.stream, self.planes,
                self.output, Some(values), rows).await?; }
        } else {
            reduce_planes(self.library, self.reducer, self.stream, self.planes,
                self.output, Some(values), rows)?;
        }
        uploads.pending = false; // uploads and reduction completed on the same stream.
        tracing::debug!(target: "ds41rt::timing", layer=self.request.request().header.layer_id, rows, shared_copy_us, upload_us, receive_us=received_us-shared_copy_us-upload_us, reduce_us=timing.elapsed().as_micros() as u64-received_us, "target collection");
        *self.ready_rows = Some(rows);
        let mut values = self.output;
        values.bytes = rows as usize * 10240;
        if let Some(device) = destination.filter(|&device| device != values.device_id) {
            values = unsafe { self.tp2.as_mut().context("TP2 return workspace missing")?
                .return_result(values, device as usize, rows).await? };
        }
        Ok(NativeFfnOutput {
            values,
            binding: self.request.binding(),
            _owner: std::marker::PhantomData,
        })
    }
}

#[cfg(test)]
mod replicated_tests {
    use super::*;
    use ds41rt_transport::{
        v41_expert::{V41BackboneRequest, V41NativeOwnerRouteWord, V41_NATIVE_GROUP_REQUEST_FLAG},
        ExpertProtocolV2RowDescriptor, ExpertProtocolV2RouteEntry, ExpertV2Dtype, ExpertV2SourceKind,
    };

    fn request(layer: u32, request_id: u64, experts: &[u32]) -> ExpertProtocolV2Request {
        let rows = experts.len() / 6;
        assert_eq!(rows * 6, experts.len());
        let mut request = ExpertProtocolV2Request::new(
            request_id,
            7,
            layer,
            5120,
            ExpertV2Dtype::Fp8E4m3Ue8m0K32,
            (0..rows).map(|r| ExpertProtocolV2RowDescriptor {
                row_id: r as u64,
                source_kind: ExpertV2SourceKind::Decode,
                source_request_id: request_id,
                token_position: r as u64,
                route_offset: r as u32 * 6,
                route_count: 6,
            }).collect(),
            experts.iter().enumerate().map(|(i, &expert_id)| ExpertProtocolV2RouteEntry {
                row_index: (i / 6) as u32,
                expert_id,
                gate_weight: 1.0 / 6.0,
            }).collect(),
            vec![0; rows * 5280],
        ).unwrap();
        request.header.flags |=
            ds41rt_transport::v41_expert::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
        request
    }

    fn owners(request: &ExpertProtocolV2Request, topology: V41SparkTopology) -> Vec<u8> {
        request.routes.iter().map(|route| {
            let word = V41NativeOwnerRouteWord::decode(route.expert_id, topology.group_count()).unwrap();
            assert!(word.expert_id < 384);
            word.owner
        }).collect()
    }

    #[test]
    fn six_route_batches_split_evenly_across_replicated_groups() {
        let unique = [0u32, 7, 19, 88, 200, 383];
        for (topology, expected) in [
            (V41SparkTopology::new(2, 2).unwrap(), 3usize),
            (V41SparkTopology::new(3, 2).unwrap(), 3),
            (V41SparkTopology::new(2, 3).unwrap(), 2),
        ] {
            let mut planner = ReplicatedGroupPlanner::new(topology).unwrap();
            let mut request = request(3, 11, &unique);
            planner.encode(&mut request).unwrap();
            assert_ne!(request.header.flags & V41_NATIVE_GROUP_REQUEST_FLAG, 0);
            V41BackboneRequest::validate_owned_native_group(&request, 1, topology).unwrap();
            let owner = owners(&request, topology);
            let mut per_group = [0usize; 3];
            for group in &owner {
                per_group[*group as usize] += 1;
            }
            assert!(
                per_group[..topology.group_count() as usize]
                    .iter()
                    .all(|&count| count == expected),
                "{:?}",
                &per_group[..topology.group_count() as usize]
            );
            // Every route keeps its true expert id and exact gate weight.
            for (route, &expert) in request.routes.iter().zip(&unique) {
                let word = V41NativeOwnerRouteWord::decode(route.expert_id, topology.group_count()).unwrap();
                assert_eq!(word.expert_id, expert);
                assert_eq!(route.gate_weight, 1.0 / 6.0);
            }
        }
    }

    #[test]
    fn seed_mode_is_wired_and_layer_is_stable_across_request_ids() {
        // Uniform-cost experts make every ordering decision a tie, so only the
        // seed can change the assignment.
        let topology = V41SparkTopology::new(2, 2).unwrap();
        let experts = [0u32, 7, 19, 88, 200, 383];

        let mut layer = ReplicatedGroupPlanner::new_with_seed_mode(
            topology,
            ReplicatedExpertTieSeedMode::Layer,
        )
        .unwrap();
        assert_eq!(layer.seed_mode(), ReplicatedExpertTieSeedMode::Layer);
        let mut base = request(3, 11, &experts);
        layer.encode(&mut base).unwrap();
        let base_owners = owners(&base, topology);
        for request_id in [12_u64, 99, u64::MAX] {
            let mut repeat = request(3, request_id, &experts);
            layer.encode(&mut repeat).unwrap();
            assert_eq!(
                owners(&repeat, topology),
                base_owners,
                "layer mode must ignore the dispatch request id"
            );
        }

        let mut dispatch = ReplicatedGroupPlanner::new_with_seed_mode(
            topology,
            ReplicatedExpertTieSeedMode::Dispatch,
        )
        .unwrap();
        assert_eq!(dispatch.seed_mode(), ReplicatedExpertTieSeedMode::Dispatch);
        let mut dispatch_base = request(3, 11, &experts);
        dispatch.encode(&mut dispatch_base).unwrap();
        let dispatch_owners = owners(&dispatch_base, topology);
        let mut varied = false;
        for request_id in 2_u64..512 {
            let mut repeat = request(3, request_id, &experts);
            dispatch.encode(&mut repeat).unwrap();
            if owners(&repeat, topology) != dispatch_owners {
                varied = true;
                break;
            }
        }
        assert!(
            varied,
            "dispatch mode must still depend on the request id on tie-heavy routes"
        );
    }

    #[test]
    fn single_group_seed_modes_have_no_effect() {
        // EP1 negative: group_count == 1 makes the seed irrelevant.
        let topology = V41SparkTopology::new(4, 1).unwrap();
        let experts = [0u32, 7, 19, 88, 200, 383];
        for mode in [
            ReplicatedExpertTieSeedMode::Dispatch,
            ReplicatedExpertTieSeedMode::Layer,
        ] {
            let mut planner = ReplicatedGroupPlanner::new_with_seed_mode(topology, mode).unwrap();
            for request_id in [1_u64, 42, u64::MAX] {
                let mut request = request(3, request_id, &experts);
                planner.encode(&mut request).unwrap();
                assert!(
                    owners(&request, topology).iter().all(|&group| group == 0),
                    "single-group ownership must always be group 0"
                );
            }
        }
    }

    #[test]
    fn duplicate_experts_in_one_row_are_rejected_by_the_protocol() {
        // A real router emits six distinct experts per token. This one-row
        // fixture repeats expert 0 six times: it is an adversarial protocol case,
        // and the canonical admission contract rejects it before ownership is
        // ever encoded. Duplicate expert ids *across* rows are legitimate and are
        // covered by the reuse fixture below.
        let topology = V41SparkTopology::new(2, 2).unwrap();
        let mut planner = ReplicatedGroupPlanner::new(topology).unwrap();
        let mut encoded = request(5, 100, &[0u32; 6]);
        let error = planner.encode(&mut encoded).unwrap_err().to_string();
        assert!(error.contains("duplicate native expert route"), "{error}");
    }

    #[test]
    fn real_row_reuse_keeps_one_group_per_expert_and_masks_weights_exactly() {
        // Production-shaped fixture: each row routes six distinct experts; rows
        // repeat a base set and frequently reuse one hot six-expert subset. Every
        // occurrence of an expert must carry the same owner, and only owned routes
        // may keep the exact FP32 gate weight; every other route is masked to the
        // 384 sentinel with weight zero.
        let topology = V41SparkTopology::new(2, 2).unwrap();
        let base = [0u32, 1, 2, 3, 4, 5];
        let hot = [200u32, 201, 202, 203, 204, 205];
        let mut routes = Vec::new();
        for row in 0..12usize {
            if row % 4 == 3 { routes.extend_from_slice(&hot); }
            else {
                let offset = row % 6;
                routes.extend((0..6).map(|slot| base[(slot + offset) % 6]));
            }
        }
        let rows = routes.len() / 6;
        let mut planner = ReplicatedGroupPlanner::new(topology).unwrap();
        let mut encoded = request(6, 777, &routes);
        planner.encode(&mut encoded).unwrap();
        V41BackboneRequest::validate_owned_native_group(&encoded, rows as u32, topology).unwrap();
        // Ownership is a pure function of the expert id, stable across rows.
        let mut expert_owner = std::collections::BTreeMap::new();
        for (route, &expert) in encoded.routes.iter().zip(&routes) {
            let word = V41NativeOwnerRouteWord::decode(route.expert_id, topology.group_count()).unwrap();
            assert_eq!(word.expert_id, expert);
            if let Some(previous) = expert_owner.insert(expert, word.owner) {
                assert_eq!(previous, word.owner, "expert {expert} changed group");
            }
        }
        // Duplicate reuse actually happened, and both hot/base experts are active.
        assert_eq!(routes.len(), 72);
        assert!(expert_owner.contains_key(&0) && expert_owner.contains_key(&200));
        // The hot six-expert subset appears in rows 3, 7 and 11.
        assert_eq!(routes.iter().filter(|&&expert| expert == 200).count(), 3);
        // The exact gate weights survive the ownership encode.
        for route in &encoded.routes {
            assert_eq!(route.gate_weight, 1.0 / 6.0);
        }
        // Unpacking each group preserves owned weights and masks the rest to the
        // kernel's unassigned sentinel with a zero weight, exactly.
        let frame = encoded.encode().unwrap();
        for group in 0..topology.group_count() {
            let mut ids = vec![0i32; encoded.routes.len()];
            let mut weights = vec![0f32; encoded.routes.len()];
            let view = V41BackboneRequest::parse_native_group(
                &frame, rows as u32, topology).unwrap();
            view.copy_native_group_routes_into(&mut ids, &mut weights, group).unwrap();
            for (index, (route, &expert)) in encoded.routes.iter().zip(&routes).enumerate() {
                let word = V41NativeOwnerRouteWord::decode(route.expert_id, topology.group_count()).unwrap();
                if word.owner == group {
                    assert_eq!(ids[index], expert as i32);
                    assert_eq!(weights[index], 1.0 / 6.0);
                } else {
                    assert_eq!(ids[index], ds41rt_transport::v41_expert::V41_NATIVE_UNASSIGNED_EXPERT_ID);
                    assert_eq!(weights[index], 0.0);
                }
            }
        }
    }

    #[test]
    fn an_empty_group_masks_every_route_and_returns_a_zero_plane() {
        // The baseline scheduler balances any batch with at least two active
        // experts, so an empty group is not reachable through whole-expert LPT on
        // a canonical histogram. It is still a real transport state (for example
        // a heavily skewed calibration), so this fixture drives the ownership
        // contract directly: group 0 owns every expert, group 1 owns none, and
        // group 1 must mask all six routes to the sentinel/zero plane.
        let topology = V41SparkTopology::new(2, 2).unwrap();
        let experts = [10u32, 11, 12, 13, 14, 15];
        let mut encoded = request(5, 101, &experts);
        let mut all_zero_owners = [0u8; 384];
        all_zero_owners[10] = 0; // explicit: group 0 owns the whole batch
        encoded.with_native_group_owners(&all_zero_owners, topology).unwrap();
        let frame = encoded.encode().unwrap();
        V41BackboneRequest::validate_owned_native_group(&encoded, 1, topology).unwrap();
        for group in 0..topology.group_count() {
            let mut ids = vec![0i32; 6];
            let mut weights = vec![0f32; 6];
            let view = V41BackboneRequest::parse_native_group(&frame, 1, topology).unwrap();
            view.copy_native_group_routes_into(&mut ids, &mut weights, group).unwrap();
            if group == 0 {
                assert_eq!(ids, experts.map(|expert| expert as i32));
            } else {
                assert_eq!(
                    ids,
                    vec![ds41rt_transport::v41_expert::V41_NATIVE_UNASSIGNED_EXPERT_ID; 6]
                );
                assert!(weights.iter().all(|&weight| weight == 0.0));
            }
        }
    }

    #[test]
    fn planning_is_reproducible_for_a_fixed_layer_and_request_id() {
        // Same (histogram, layer, request_id) must reproduce the same assignment
        // so a graph replay with unchanged input keeps identical ownership.
        let topology = V41SparkTopology::new(2, 3).unwrap();
        let experts = [3u32, 9, 12, 40, 77, 300];
        let mut replan = || {
            let mut planner = ReplicatedGroupPlanner::new(topology).unwrap();
            let mut request = request(9, 4242, &experts);
            planner.encode(&mut request).unwrap();
            owners(&request, topology)
        };
        let first = replan();
        let second = replan();
        assert_eq!(first, second);
        // A distinct request identity is allowed to break ties differently, but
        // must still be a valid exactly-once assignment.
        let mut planner = ReplicatedGroupPlanner::new(topology).unwrap();
        let mut other = request(9, 4243, &experts);
        planner.encode(&mut other).unwrap();
        let mut per_group = [0usize; 3];
        for group in owners(&other, topology) {
            per_group[group as usize] += 1;
        }
        assert!(per_group.iter().all(|&count| count == 2));
    }

    #[test]
    fn single_group_topology_owns_every_active_expert() {
        let topology = V41SparkTopology::new(2, 1).unwrap();
        let mut planner = ReplicatedGroupPlanner::new(topology).unwrap();
        let mut request = request(0, 1, &[1, 2, 3, 4, 5, 6]);
        planner.encode(&mut request).unwrap();
        assert!(owners(&request, topology).iter().all(|&group| group == 0));
    }

    #[test]
    fn wave_reservation_scales_with_the_physical_rank_count() {
        for capacity in [1u32, 16, 80, 4096] {
            let legacy = NativeTp4Wave::device_bytes(capacity).unwrap();
            assert_eq!(legacy, NativeTp4Wave::device_bytes_for(capacity, 4).unwrap());
            assert_eq!(legacy, capacity as usize * 6 * 10240);
            for ranks in [2usize, 3, 4, 6] {
                let expected = capacity as usize * (ranks * 10240 + 20480);
                assert_eq!(NativeTp4Wave::device_bytes_for(capacity, ranks).unwrap(), expected);
                assert!(expected >= capacity as usize * ranks * 10240);
            }
            // Six ranks reserve two more planes than the legacy TP4 forecast.
            assert!(
                NativeTp4Wave::device_bytes_for(capacity, 6).unwrap() > legacy
            );
        }
        for (capacity, ranks) in [(0u32, 6usize), (4097, 6), (80, 0), (80, 1), (80, 5), (80, 8)] {
            assert!(NativeTp4Wave::device_bytes_for(capacity, ranks).is_err(), "{capacity}/{ranks}");
        }
    }

    #[test]
    fn replicated_cost_override_is_validated_or_rejected() {
        assert_eq!(parse_replicated_cost("1,0,16").unwrap(), ReplicatedExpertCostModel::new(1, 0, 16));
        assert_eq!(parse_replicated_cost(" 4 , 2 , 32 ").unwrap(), ReplicatedExpertCostModel::new(4, 2, 32));
        for invalid in ["", "1,0", "1,0,16,2", "1,0,0", "0,0,16", "x,0,16", "1,2"] {
            assert!(parse_replicated_cost(invalid).is_err(), "{invalid:?}");
        }
        // The default profile is the documented whole-expert weight-only model.
        assert_eq!(ReplicatedExpertCostModel::new(1, 0, 16).expert_cost(6).unwrap(), 1);
        assert_eq!(ReplicatedExpertCostModel::new(1, 0, 16).expert_cost(0).unwrap(), 0);
    }
}

#[cfg(test)]
mod upload_tests {
    use super::*;

    #[test]
    fn tp_wave_budget_remains_tp4_upper_bound() -> Result<()> {
        assert!(NativeTp4Wave::device_bytes(0).is_err());
        assert!(NativeTp4Wave::device_bytes(4097).is_err());
        for capacity in [1, 16, 80, 4096] {
            let reserved = NativeTp4Wave::device_bytes(capacity)?;
            assert_eq!(reserved, capacity as usize * 6 * 10240);
            assert!(reserved >= capacity as usize * (2 * V41_PARTIAL_ROW_BYTES as usize + 2 * 10240));
        }
        Ok(())
    }

    #[test]
    fn missing_tp_planes_reject_copy_before_pointer_arithmetic() {
        for rank in [0, 1, 2, 4, usize::MAX] {
            assert!(chunk_destination(&[], rank, 0, 10240).is_err());
        }
    }

    #[test]
    fn pageable_rank_upload_fallback_matches_sync_and_checks_bounds() -> Result<()> {
        let Some(path) = std::env::var_os("DS41RT_PLANE_UPLOAD_LIBRARY") else {
            eprintln!("skip GPU rank upload test: DS41RT_PLANE_UPLOAD_LIBRARY unset");
            return Ok(());
        };
        let library = unsafe { NativeLibrary::load(path)? };
        for world_size in [2, 4] {
            check_pageable_uploads(&library, world_size)?;
        }
        Ok(())
    }

    fn check_pageable_uploads(library: &NativeLibrary, world_size: usize) -> Result<()> {
        let stream = LoadStream { library, raw: library.cuda_stream_create()? };
        let planes = (0..world_size)
            .map(|_| DeviceAllocation::new(&library, 4096 * 10240))
            .collect::<Result<Vec<_>>>()?;
        let mut frames = Vec::with_capacity(4096 * world_size);
        let shared = DeviceAllocation::new(&library, 4096 * 10240)?;
        let output = DeviceAllocation::new(&library, 4096 * 10240)?;
        let reducer = library.v41_compact_reducer()?;
        let frames_address = frames.as_ptr();
        for rows in [1u32, 6, 80, 81, 256, 1024, 4096, 6] {
            let bytes = rows as usize * 10240;
            let shared_bytes: Vec<u8> = (0..bytes / 2)
                .flat_map(|_| 0x3f00u16.to_ne_bytes()).collect();
            library.copy_h2d(shared.buffer, &shared_bytes)?;
            let payloads: Vec<Vec<u8>> = (0..world_size).map(|rank| {
                (0..bytes / 2).flat_map(|i| {
                    let value = rank as f32 + 1.0 + (i % 31) as f32 / 32.0;
                    ((value.to_bits() >> 16) as u16).to_ne_bytes()
                }).collect()
            }).collect();
            for (rank, payload) in payloads.iter().enumerate() {
                copy_chunk(&library, &planes, rank, 0, payload)?;
            }
            reduce_planes(&library, &reducer, &stream, &planes,
                output.buffer, Some(shared.buffer), rows)?;
            let mut expected = vec![0u8; bytes];
            library.copy_d2h(&mut expected, Ds41rtDeviceBuffer { bytes, ..output.buffer })?;
            for plane in &planes { library.copy_h2d(plane.buffer, &vec![0; bytes])?; }
            {
                let mut uploads = PlaneUploads {
                    library: &library, stream: &stream, planes: &planes,
                    frames: &mut frames, rows, pending: false,
                };
                for first in (0..rows).step_by(3) {
                    let end = (first + 3).min(rows);
                    for rank in [3, 1, 0, 2].into_iter().filter(|&rank| rank < world_size) {
                        uploads.copy(rank, first,
                            VerbsHostProtocolV2ResponsePayload::from_owned(payloads[rank][first as usize * 10240..end as usize * 10240].to_vec()))?;
                    }
                }
                let runtime = tokio::runtime::Builder::new_current_thread().build()?;
                runtime.block_on(async {
                    use std::{future::Future, task::Poll};
                    let cancelled = {
                        let mut work = std::pin::pin!(unsafe { reduce_planes_cooperative(&reducer, &stream,
                            &planes, output.buffer, Some(shared.buffer), rows) });
                        std::future::poll_fn(|cx| Poll::Ready(work.as_mut().poll(cx).is_pending())).await
                    };
                    stream.require_complete()?;
                    unsafe { reduce_planes_cooperative(&reducer, &stream, &planes,
                        output.buffer, Some(shared.buffer), rows).await?; }
                    eprintln!("PASS cooperative TP reduction rows={rows} pending_cancel={cancelled} reuse=true");
                    Ok::<_, anyhow::Error>(())
                })?;
                stream.require_complete()?;
                uploads.pending = false;
            }
            let mut actual = vec![0u8; bytes];
            library.copy_d2h(&mut actual, Ds41rtDeviceBuffer { bytes, ..output.buffer })?;
            assert_eq!(actual, expected, "rows={rows}");
            assert_eq!(frames.as_ptr(), frames_address);
            assert!(frames.is_empty());
            // Pageable fallback completes synchronously, including before a later bounds error.
            for error in [false, true] {
                library.copy_h2d(planes[0].buffer, &vec![0; 10240])?;
                {
                    let mut uploads = PlaneUploads {
                        library: &library, stream: &stream, planes: &planes,
                        frames: &mut frames, rows: 1, pending: false,
                    };
                    uploads.copy(0, 0, VerbsHostProtocolV2ResponsePayload::from_owned(payloads[0][..10240].to_vec()))?;
                    if error {
                        assert!(uploads.copy(world_size, 0, VerbsHostProtocolV2ResponsePayload::from_owned(payloads[0][..10240].to_vec())).is_err());
                        assert!(uploads.copy(0, 1, VerbsHostProtocolV2ResponsePayload::from_owned(payloads[0][..10240].to_vec())).is_err());
                    }
                }
                assert!(frames.is_empty());
                let mut actual = vec![0; 10240];
                library.copy_d2h(&mut actual, Ds41rtDeviceBuffer { bytes: 10240, ..planes[0].buffer })?;
                assert_eq!(actual, payloads[0][..10240]);
            }
            eprintln!("PASS TP{world_size} rows={rows}: pageable interleaved chunks/reduction exact, stable owner capacity, bounds errors");
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires four idle live Spark workers and CUDA native library"]
    fn retained_roce_uploads_drain_before_recycling_live() -> Result<()> {
        use ds41rt_transport::{ExpertProtocolV2RowDescriptor, ExpertProtocolV2RouteEntry,
            ExpertV2SourceKind, ExpertV2Dtype, TcpTransportConfig};
        let library = unsafe { NativeLibrary::load(std::env::var("DS41RT_PLANE_UPLOAD_LIBRARY")?)? };
        let stream = LoadStream { library: &library, raw: library.cuda_stream_create()? };
        let peers = std::env::var("DS41RT_LIVE_ROCE_PEERS")?.split(',')
            .map(str::parse).collect::<std::result::Result<Vec<std::net::SocketAddr>, _>>()?
            .try_into().map_err(|_| anyhow::anyhow!("four peers required"))?;
        let mut client = V41Tp4Roce::new(peers, [1,2,3,4], 4096, TcpTransportConfig { timing: false,
            timeout: std::time::Duration::from_secs(30), max_frame_bytes: 64 << 20,
        })?;
        let planes: [DeviceAllocation<'_>; 4] = (0..4).map(|_| DeviceAllocation::new(&library, 4096 * 10240))
            .collect::<Result<Vec<_>>>()?.try_into().ok().expect("four planes");
        let mut frames = Vec::with_capacity(4096 * 4);
        let frame_address = frames.as_ptr();
        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
        runtime.block_on(async {
            let mut request_id = 900_000;
            for rows in [1u32,6,80,81,256,1024,4096,6] {
                let mut hidden = Vec::with_capacity(rows as usize * 5280);
                for row in 0..rows {
                    hidden.extend((0..5120).map(|i| 0x30 + ((i+row)%8) as u8));
                    hidden.extend([120;160]);
                }
                let mut request = ExpertProtocolV2Request::new(request_id, 17, 39, 5120,
                    ExpertV2Dtype::Fp8E4m3Ue8m0K32,
                    (0..rows).map(|r| ExpertProtocolV2RowDescriptor {
                        row_id: r as u64, source_kind: ExpertV2SourceKind::Prefill,
                        source_request_id: request_id, token_position: r as u64,
                        route_offset: r*6, route_count: 6,
                    }).collect(),
                    (0..rows).flat_map(|r| (0..6).map(move |j| ExpertProtocolV2RouteEntry {
                        row_index: r, expert_id: (r*7+j)%384, gate_weight: 1.0/6.0,
                    })).collect(), hidden)?;
                request.header.flags = ds41rt_transport::v41_expert::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
                for fail in [false,true,false] {
                    request_id += 1;
                    request.header.request_id = request_id;
                    let mut expected: Vec<(usize,u32,Vec<u8>)> = Vec::new();
                    {
                        let mut uploads = PlaneUploads { library: &library, stream: &stream,
                            planes: &planes, frames: &mut frames, rows, pending: false };
                        let result = client.dispatch(&request).await?.receive_owned(|rank, first, payload| {
                            ensure!(payload.pinned_host_buffer().is_some(), "live payload is not pinned");
                            ensure!(payload.retains_receive_slot(), "final response is not a retained receive slot");
                            expected.push((rank, first, payload.as_ref().to_vec()));
                            uploads.copy(rank, first, payload)?;
                            ensure!(uploads.pending && !uploads.frames.is_empty(), "upload ownership missing");
                            if fail { anyhow::bail!("injected failure after pinned upload enqueue"); }
                            Ok(())
                        }).await;
                        assert_eq!(result.is_err(), fail);
                        // QP teardown must not free payloads owned by an active upload.
                        if fail { client.reset_connections(); }
                        // Drop simulates cancellation after enqueue and must synchronize
                        // before returning retained storage to the transport pool.
                    }
                    assert!(frames.is_empty());
                    assert_eq!(frames.as_ptr(), frame_address);
                    assert!(!expected.is_empty());
                    if !fail {
                        assert_eq!(expected.iter().map(|(_,_,b)| b.len()).sum::<usize>(), rows as usize * 4 * 10240);
                    }
                    for (rank, first, bytes) in expected {
                        let mut actual = vec![0;bytes.len()];
                        library.copy_d2h(&mut actual, chunk_destination(&planes, rank, first, bytes.len())?)?;
                        assert_eq!(actual, bytes, "rows={rows} fail={fail} rank={rank}");
                    }
                }
                eprintln!("PASS retained rows={rows}: real pinned frames, exact device copies, failure/reset/drop drain and recovery");
            }
            Ok(())
        })
    }

}
