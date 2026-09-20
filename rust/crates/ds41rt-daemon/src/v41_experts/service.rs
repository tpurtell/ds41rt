//! Native GPU resources and inference QPs share one owning thread.
mod local;
mod backend;

use super::{ExpertLayer, ExpertWeights, HostExpertExchange};
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::NativeLibrary;
use ds41rt_loader::{read_official_v41_catalog, OfficialV41Catalog, OFFICIAL_V41_MODEL_ID};
use ds41rt_transport::v41_expert::{V41BackboneRequest, V41SparkTopology};
use std::{path::PathBuf, sync::mpsc, thread};

pub(crate) async fn run(args: crate::cli::NativeExpertDaemonArgs) -> Result<()> {
    let topology = crate::v41_spark_topology::resolve(
        args.spark_tp,
        args.spark_ep,
        args.world as usize,
        "expertd-native",
    )?;
    if let Some(topology) = topology {
        ensure!(
            (args.rank as usize) < topology.world_size(),
            "expertd-native rank {} is outside the {}x{} topology",
            args.rank,
            topology.tp(),
            topology.ep()
        );
    }
    let config = NativeExpertServiceConfig {
        library: args.native_lib,
        exl3_aot_dir: args.exl3_aot_dir,
        snapshot: args.snapshot,
        rank: args.rank as usize,
        world: args.world as usize,
        first_layer: args.first_layer as usize,
        capacity: args.capacity,
        device_budget: args.device_budget_bytes,
        max_frame_bytes: args.max_frame_bytes,
        topology,
    };
    tokio::task::spawn_blocking(move || local::run(config, &args.listen))
        .await
        .context("native local RoCE owner failed")?
}

pub(crate) struct NativeExpertServiceConfig {
    pub library: PathBuf,
    pub exl3_aot_dir: Option<PathBuf>,
    pub snapshot: PathBuf,
    pub rank: usize,
    pub world: usize,
    pub first_layer: usize,
    pub capacity: u32,
    pub device_budget: usize,
    pub max_frame_bytes: usize,
    /// Explicit replicated `TP×EP` topology; `None` keeps the legacy world 2/4
    /// behavior (EXL3 compact may still select a TP2 RTX pair).
    pub topology: Option<V41SparkTopology>,
}

fn load_weights<'a>(
    library: &'a NativeLibrary,
    config: &NativeExpertServiceConfig,
) -> Result<(backend::Weights<'a>, usize)> {
    let catalog = read_official_v41_catalog(OFFICIAL_V41_MODEL_ID, &config.snapshot)?;
    ensure!(config.first_layer < 40, "native first layer must be 0..39");
    validate_topology(config, &catalog)?;
    if catalog.exl3().is_some() { return backend::load_exl3(library, &catalog, config); }
    log_spark_memory_if_enabled(library, config, "worker startup", None, None);
    // NVFP4 backbone experts load through the format-aware ExpertWeights path.
    let nvfp4 = catalog.nvfp4().is_some();
    let mut resident = 0usize;
    let mut staging = 0usize;
    let mut pinned_host = 0usize;
    let mut read_scratch = 0usize;
    for layer in config.first_layer..40 {
        let plan = ExpertWeights::plan(library, &catalog, config.selection(layer)?)?;
        resident = resident
            .checked_add(plan.resident_bytes)
            .context("resident budget overflow")?;
        staging = staging.max(plan.device_staging_bytes);
        pinned_host = pinned_host.max(plan.pinned_host_bytes);
        read_scratch = read_scratch.max(plan.read_scratch_bytes);
    }
    ensure!(
        resident
            .checked_add(staging)
            .context("loading budget overflow")?
            <= config.device_budget,
        "native TP weights and staging exceed device budget"
    );
    // The workspace is planned from the same layer selection as the load, so a
    // TP2/TP3 topology cannot be budgeted with TP4 scratch.
    let workspace = ExpertWeights::plan_execution(
        config.selection(0)?,
        library,
        config.capacity,
        nvfp4,
    )?
    .total()?;
    ensure!(
        resident
            .checked_add(workspace)
            .context("execution budget overflow")?
            <= config.device_budget,
        "native TP weights and execution workspace exceed device budget"
    );
    // Explicit replicated topologies additionally admit every known Spark-side
    // transient. On the unified GB10 pool the loader's pinned host staging, the
    // read scratch, the host exchange and the row-index scratch all compete with
    // device memory, so the weight-only and workspace-only checks are not a fit
    // proof. This does not change the legacy EXL3/TP4 admission above.
    if config.topology.is_some() {
        let budget = spark_admission_budget(config, resident, staging, pinned_host,
            read_scratch, workspace)?;
        ensure!(
            budget.load_peak <= config.device_budget
                && budget.serve_peak <= config.device_budget,
            "explicit {}x{} Spark admission exceeds the {}-byte device budget: \
             resident {} + load peak {} / serve peak {}",
            config.topology.map_or(0, |t| t.tp()),
            config.topology.map_or(0, |t| t.ep()),
            config.device_budget,
            resident,
            budget.load_peak,
            budget.serve_peak
        );
        // A configured budget above the real pool must not admit more than the
        // startup measurement can hold. The OS reserve stays outside this budget
        // by construction; page-cache bytes are reclaimable and are not added.
        // This is a safety gate, so a failed query fails closed instead of
        // silently skipping the real-availability check.
        let (free, _total) = library
            .cuda_memory_info()
            .context("query Spark memory for admission")?;
        ensure!(
            budget.load_peak <= free && budget.serve_peak <= free,
            "explicit replicated Spark admission needs {} bytes at load / {} at serve \
             but only {free} bytes are actually available",
            budget.load_peak,
            budget.serve_peak
        );
        tracing::info!(
            rank = config.rank,
            resident_bytes = resident,
            staging_bytes = staging,
            pinned_host_bytes = pinned_host,
            read_scratch_bytes = read_scratch,
            workspace_bytes = workspace,
            exchange_bytes = budget.exchange,
            row_indices_bytes = budget.row_indices,
            ring_bytes = budget.rings,
            runtime_headroom_bytes = budget.headroom,
            load_peak_bytes = budget.load_peak,
            serve_peak_bytes = budget.serve_peak,
            device_budget_bytes = config.device_budget,
            "explicit Spark replicas admission"
        );
    }
    let mut weights = Vec::with_capacity(40 - config.first_layer);
    let mut remaining = config.device_budget;
    for layer in config.first_layer..40 {
        let started = std::time::Instant::now();
        let weight = ExpertWeights::load(
            library,
            &catalog,
            config.selection(layer)?,
            remaining,
        )?;
        remaining = remaining
            .checked_sub(weight.budget().resident_bytes)
            .context("resident budget exhausted")?;
        tracing::info!(
            rank = config.rank,
            layer,
            resident_bytes = weight.budget().resident_bytes,
            elapsed_ms = started.elapsed().as_millis() as u64,
            "native expert layer loaded"
        );
        weights.push(weight);
    }
    log_spark_memory_if_enabled(library, config, "weights resident", None, None);
    Ok((backend::Weights::Full(weights), remaining))
}

/// Known explicit-topology peak bytes for the two live phases.
struct SparkAdmissionBudget {
    exchange: usize,
    row_indices: usize,
    rings: usize,
    headroom: usize,
    load_peak: usize,
    serve_peak: usize,
}

fn env_usize(name: &str, default: usize) -> Result<usize> {
    match std::env::var_os(name) {
        None => Ok(default),
        Some(value) => value
            .to_str()
            .with_context(|| format!("{name} is not valid UTF-8"))?
            .trim()
            .parse::<usize>()
            .with_context(|| format!("{name} must be an unsigned integer")),
    }
}

fn align_up(value: usize, alignment: usize) -> Result<usize> {
    ensure!(alignment > 0, "ring alignment must be non-zero");
    value
        .checked_add(alignment - 1)
        .map(|value| value / alignment * alignment)
        .context("ring slot alignment overflow")
}

/// Registered/mapped RDMA ring bytes the Spark worker pins for `endpoints`
/// persistent ProtocolV2 endpoints. Mirrors the transport's
/// `verbs_host_persistent_rings` rule exactly for the native compact-BF16
/// request shape: one request ring and one response ring per endpoint, each
/// `depth` slots of `align_up(capacity, verbs-host alignment)`, with
/// `capacity = min(max(slot, wire_bytes), max_frame_bytes)`.
///
/// A request whose wire size exceeds the frame budget would be rejected by the
/// transport at connect time, so it fails admission here instead.
fn spark_ring_bytes(
    capacity: u32,
    max_frame_bytes: usize,
    depth: usize,
    slot_bytes: usize,
    alignment: usize,
    endpoints: usize,
) -> Result<usize> {
    use ds41rt_transport::protocol_v2::{
        EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN, EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN,
        EXPERT_PROTOCOL_V2_ROUTE_ENTRY_LEN, EXPERT_PROTOCOL_V2_ROW_DESCRIPTOR_LEN,
    };
    ensure!(
        (1..=8).contains(&depth),
        "verbs-host ring depth must be in 1..=8"
    );
    ensure!(endpoints > 0, "at least one RDMA endpoint is required");
    ensure!(slot_bytes > 0, "RDMA ring slot bytes must be non-zero");
    ensure!(max_frame_bytes > 0, "native frame budget must be non-zero");
    let rows = capacity as usize;
    let request_per_row = EXPERT_PROTOCOL_V2_ROW_DESCRIPTOR_LEN
        .checked_add(6 * EXPERT_PROTOCOL_V2_ROUTE_ENTRY_LEN)
        .and_then(|per_row| per_row.checked_add(5280))
        .context("native request row wire size overflow")?;
    let request_wire = EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN
        .checked_add(
            rows.checked_mul(request_per_row)
                .context("native request wire size overflow")?,
        )
        .context("native request wire size overflow")?;
    // Ingress K32 rows are 5280 bytes; the compact-BF16 rank partial is
    // 5120 * 2 bytes and dominates the negotiated response row.
    let response_wire = EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN
        .checked_add(
            rows.checked_mul(4 + 5120 * 2)
                .context("native response wire size overflow")?,
        )
        .context("native response wire size overflow")?;
    ensure!(
        request_wire <= max_frame_bytes && response_wire <= max_frame_bytes,
        "native compact-BF16 wire frames ({request_wire} request / {response_wire} response) \
         exceed the {max_frame_bytes}-byte native frame budget"
    );
    let per_endpoint = align_up(request_wire.max(slot_bytes).min(max_frame_bytes), alignment)?
        .checked_add(align_up(
            response_wire.max(slot_bytes).min(max_frame_bytes),
            alignment,
        )?)
        .context("registered ring span overflow")?
        .checked_mul(depth)
        .context("registered ring span overflow")?;
    per_endpoint
        .checked_mul(endpoints)
        .context("registered ring span overflow")
}

/// The Spark worker always opens exactly two persistent endpoints (decode and
/// prefill); that is the count the admission model reserves. The obsolete
/// `DS41RT_SPARK_RDMA_ENDPOINTS` knob is rejected unless it names that value, so
/// a stale launch cannot silently under-reserve the mapped rings.
fn parse_rdma_endpoints(value: Option<&str>) -> Result<usize> {
    match value.map(str::trim) {
        None => Ok(2),
        Some("2") => {
            tracing::warn!(
                "DS41RT_SPARK_RDMA_ENDPOINTS is obsolete; the Spark worker always opens two persistent endpoints"
            );
            Ok(2)
        }
        Some(other) => anyhow::bail!(
            "DS41RT_SPARK_RDMA_ENDPOINTS={other} is obsolete and unsupported; the Spark worker \
             requires exactly 2 persistent endpoints (decode and prefill)"
        ),
    }
}

/// Ring bytes for this worker's configured topology and frame budget. The
/// decode and prefill lanes each open one persistent endpoint.
fn spark_transport_bytes(config: &NativeExpertServiceConfig) -> Result<usize> {
    let depth = env_usize("DS41RT_VERBS_HOST_RING_DEPTH", 8)?;
    let slot_bytes = env_usize("DS41RT_VERBS_HOST_RING_SLOT_BYTES", 8 << 20)?;
    let requested = std::env::var("DS41RT_SPARK_RDMA_ENDPOINTS").ok();
    let endpoints = parse_rdma_endpoints(requested.as_deref())?;
    let alignment = ds41rt_transport::verbs_host_capabilities().preferred_alignment;
    spark_ring_bytes(
        config.capacity,
        config.max_frame_bytes,
        depth,
        slot_bytes,
        alignment,
        endpoints,
    )
}

/// Optional explicit reserve for allocation granularity, the CUDA context and
/// library state that the source cannot see. It defaults to zero: no arbitrary
/// headroom is invented, and a real reserve is supplied from an actual
/// measurement. `DS41RT_SPARK_RUNTIME_HEADROOM_BYTES` sets it.
fn spark_runtime_headroom() -> Result<usize> {
    env_usize("DS41RT_SPARK_RUNTIME_HEADROOM_BYTES", 0)
}

fn spark_admission_budget(
    config: &NativeExpertServiceConfig,
    resident: usize,
    staging: usize,
    pinned_host: usize,
    read_scratch: usize,
    workspace: usize,
) -> Result<SparkAdmissionBudget> {
    let exchange = HostExpertExchange::bytes_for(config.capacity)?;
    let row_indices = config.capacity as usize * 4;
    let rings = spark_transport_bytes(config)?;
    let headroom = spark_runtime_headroom()?;
    // The loader's pinned buffers and read scratch live only for one layer load;
    // the exchange, row-index scratch and registered rings live while serving.
    let load_peak = resident
        .checked_add(staging)
        .and_then(|bytes| bytes.checked_add(pinned_host))
        .and_then(|bytes| bytes.checked_add(read_scratch))
        .and_then(|bytes| bytes.checked_add(headroom))
        .context("Spark load peak overflow")?;
    let serve_peak = resident
        .checked_add(workspace)
        .and_then(|bytes| bytes.checked_add(exchange))
        .and_then(|bytes| bytes.checked_add(row_indices))
        .and_then(|bytes| bytes.checked_add(rings))
        .and_then(|bytes| bytes.checked_add(headroom))
        .context("Spark serve peak overflow")?;
    Ok(SparkAdmissionBudget { exchange, row_indices, rings, headroom, load_peak, serve_peak })
}

/// Startup/load/serve-milestone memory diagnostics: the actual CUDA free/total
/// and one `/proc/meminfo` snapshot (`MemTotal`, `MemFree`, `MemAvailable`,
/// `Cached`, `Mlocked`, `Unevictable`, `SReclaimable`, all in KiB) on the
/// unified GB10 pool. Reclaimable page cache is reported, never added to the
/// permanent footprint; the extra fields separate host-available from UMA free.
///
/// `owned_connections` and `rings` are optional point-in-time observations; the
/// ring peak is an atomic high-water mark that concurrent admissions can raise,
/// so it is reported as a sampled value, not a reservation or a claim.
fn log_spark_memory(
    library: &NativeLibrary,
    config: &NativeExpertServiceConfig,
    stage: &str,
    owned_connections: Option<usize>,
    rings: Option<(usize, usize)>,
) {
    let cuda = library.cuda_memory_info().ok();
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok();
    let field = |name: &str| -> Option<u64> {
        meminfo.as_deref()?.lines().find_map(|line| {
            let (key, value) = line.split_once(':')?;
            (key.trim() == name).then(|| value.split_whitespace().next()?.parse::<u64>().ok())?
        })
    };
    tracing::info!(
        target: "ds41rt::spark_memory",
        rank = config.rank,
        stage,
        topology = ?config.topology.map(|topology| (topology.tp(), topology.ep())),
        owned_connections = ?owned_connections,
        cuda_free_bytes = cuda.map(|(free, _)| free),
        cuda_total_bytes = cuda.map(|(_, total)| total),
        host_mem_total_kib = field("MemTotal"),
        host_mem_free_kib = field("MemFree"),
        host_mem_available_kib = field("MemAvailable"),
        host_mem_cached_kib = field("Cached"),
        host_mem_mlocked_kib = field("Mlocked"),
        host_mem_unevictable_kib = field("Unevictable"),
        host_mem_reclaimable_kib = field("SReclaimable"),
        ring_budget_used_bytes_point_in_time = rings.map(|(used, _)| used),
        ring_budget_peak_bytes_point_in_time = rings.map(|(_, peak)| peak),
        "Spark unified memory state"
    );
}

/// Query and log only when INFO is enabled, so the new serve-time milestones do
/// not pay a `cudaMemGetInfo` plus `/proc/meminfo` read on every connection when
/// diagnostics are off. The guard uses the same explicit target as the emitted
/// event, so a per-target filter cannot let the guard fire while the event is
/// suppressed (or vice versa).
fn log_spark_memory_if_enabled(
    library: &NativeLibrary,
    config: &NativeExpertServiceConfig,
    stage: &str,
    owned_connections: Option<usize>,
    rings: Option<(usize, usize)>,
) {
    if tracing::enabled!(target: "ds41rt::spark_memory", tracing::Level::INFO) {
        log_spark_memory(library, config, stage, owned_connections, rings);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(world: usize, rank: usize, topology: Option<V41SparkTopology>) -> NativeExpertServiceConfig {
        NativeExpertServiceConfig {
            library: PathBuf::from("/native.so"),
            exl3_aot_dir: None,
            snapshot: PathBuf::from("/model"),
            rank,
            world,
            first_layer: 0,
            capacity: 16,
            device_budget: 1 << 40,
            max_frame_bytes: 64 << 20,
            topology,
        }
    }

    #[test]
    fn explicit_topology_maps_every_physical_rank_to_its_local_shard() {
        for (tp, ep) in [(2u8, 1u8), (3, 1), (4, 1), (2, 2), (3, 2), (2, 3)] {
            let topology = V41SparkTopology::new(tp, ep).unwrap();
            for rank in 0..topology.world_size() {
                let selection = config(topology.world_size(), rank, Some(topology))
                    .selection(7)
                    .unwrap();
                assert_eq!(selection.layer(), 7);
                if tp == 4 {
                    assert_eq!(selection, ExpertLayer::Backbone { layer: 7, rank: rank });
                } else {
                    assert_eq!(
                        selection,
                        ExpertLayer::BackboneReplicatedTp {
                            layer: 7,
                            rank: rank % tp as usize,
                            world: tp as usize,
                        }
                    );
                    assert_eq!(selection.role(), if tp == 2 {
                        crate::v41_spark_topology::SPARK_TP2_ROLE
                    } else {
                        crate::v41_spark_topology::SPARK_TP3_ROLE
                    });
                    assert_eq!(
                        selection.expert(3),
                        ds41rt_loader::V41ExpertSelection::BackboneTp {
                            layer: 7,
                            expert: 3,
                            rank: rank % tp as usize,
                            world: tp as usize,
                        }
                    );
                }
            }
        }
    }

    #[test]
    fn legacy_selection_is_unchanged_and_topology_must_agree() {
        assert_eq!(config(4, 3, None).selection(2).unwrap(), ExpertLayer::Backbone { layer: 2, rank: 3 });
        assert_eq!(config(2, 1, None).selection(2).unwrap(), ExpertLayer::BackboneTp2 { layer: 2, rank: 1 });
        // The explicit topology must match the launched rank count.
        let topology = V41SparkTopology::new(2, 2).unwrap();
        assert!(resolve_topology_mismatch(&config(6, 0, Some(topology))));
        // A rank outside the topology is rejected before selection is used.
        assert!(config(4, 4, Some(topology)).selection(0).is_err());
    }

    fn resolve_topology_mismatch(config: &NativeExpertServiceConfig) -> bool {
        crate::v41_spark_topology::resolve(
            Some(config.topology.unwrap().tp()),
            Some(config.topology.unwrap().ep()),
            config.world,
            "expertd-native",
        )
        .is_err()
    }

    /// The explicit-topology admission is exactly the source-known transient sum,
    /// with no invented headroom when the override is unset.
    #[test]
    fn explicit_admission_counts_every_known_transient_exactly() {
        let topology = V41SparkTopology::new(2, 2).unwrap();
        let mut config = config(4, 0, Some(topology));
        config.capacity = 4096;
        let headroom = spark_runtime_headroom().unwrap();
        let rings = spark_transport_bytes(&config).unwrap();
        let budget = spark_admission_budget(&config, 10, 1, 2, 3, 4).unwrap();
        let exchange = HostExpertExchange::bytes_for(4096).unwrap();
        assert_eq!(exchange, 4096 * (2 * 6 * 4 + 5120 * 2));
        assert_eq!(budget.exchange, exchange);
        assert_eq!(budget.row_indices, 4096 * 4);
        assert_eq!(budget.rings, rings);
        assert_eq!(budget.load_peak, 10 + 1 + 2 + 3 + headroom);
        assert_eq!(budget.serve_peak, 10 + 4 + exchange + 4096 * 4 + rings + headroom);
        // The registered rings and the 40 MiB host exchange dominate the serve
        // peak at full capacity.
        assert!(budget.serve_peak > budget.load_peak);
        assert!(rings > exchange);
        assert_eq!(HostExpertExchange::bytes_for(16).unwrap(), 16 * (48 + 10240));
    }

    /// The ring term mirrors the transport's own sizing rule: minimum 8 MiB per
    /// slot, `depth` slots per ring, one request and one response ring per
    /// persistent endpoint, rounded up to the verbs-host alignment.
    #[test]
    fn registered_ring_bytes_match_the_transport_sizing_rule() {
        let alignment = 4096;
        let depth = 8;
        let slot = 8 << 20;
        let max_frame = 64 << 20;
        // At capacity 4096 the wire frames exceed the 8 MiB minimum slot.
        let request_wire = 96 + 4096 * (40 + 6 * 12 + 5280);
        let response_wire = 96 + 4096 * (4 + 5120 * 2);
        let expected = 2 * depth * (align_up(request_wire, alignment).unwrap()
            + align_up(response_wire, alignment).unwrap());
        assert_eq!(
            spark_ring_bytes(4096, max_frame, depth, slot, alignment, 2).unwrap(),
            expected
        );
        assert!(expected > 900 << 20 && expected < 1100 << 20, "{expected}");
        // At small capacity the 8 MiB minimum slot dominates: 2 endpoints x
        // (64 MiB request + 64 MiB response) = 256 MiB.
        assert_eq!(
            spark_ring_bytes(1, max_frame, depth, slot, alignment, 2).unwrap(),
            256 << 20
        );
        assert_eq!(
            spark_ring_bytes(1, max_frame, depth, slot, alignment, 1).unwrap(),
            128 << 20
        );
        // Depth, endpoints, slot and frame budget are validated.
        assert!(spark_ring_bytes(1, max_frame, 0, slot, alignment, 2).is_err());
        assert!(spark_ring_bytes(1, max_frame, 9, slot, alignment, 2).is_err());
        assert!(spark_ring_bytes(1, max_frame, depth, 0, alignment, 2).is_err());
        assert!(spark_ring_bytes(1, max_frame, depth, slot, alignment, 0).is_err());
        assert!(spark_ring_bytes(1, max_frame, depth, slot, 0, 2).is_err());
        // A frame that cannot fit the configured budget fails before connect.
        assert!(spark_ring_bytes(4096, 1 << 20, depth, slot, alignment, 2).is_err());
    }

    #[test]
    fn explicit_admission_peak_overflow_is_rejected() {
        let topology = V41SparkTopology::new(3, 2).unwrap();
        let mut config = config(6, 0, Some(topology));
        config.capacity = 16;
        assert!(spark_admission_budget(&config, usize::MAX, 1, 0, 0, 0).is_err());
        assert!(spark_admission_budget(&config, 1, usize::MAX, 1, 1, 1).is_err());
        assert!(spark_admission_budget(&config, 1, 0, 0, 0, usize::MAX).is_err());
        assert!(HostExpertExchange::bytes_for(0).is_err());
        assert!(HostExpertExchange::bytes_for(4097).is_err());
    }

    #[test]
    fn host_exchange_allocation_matches_its_declared_extents() {
        for capacity in [1u32, 16, 80, 256, 1024, 4096] {
            let exchange = HostExpertExchange::new(capacity).unwrap();
            assert_eq!(exchange.ids.len(), capacity as usize * 6);
            assert_eq!(exchange.routing.len(), capacity as usize * 6);
            assert_eq!(exchange.partials.len(), capacity as usize * 5120 * 2);
            assert_eq!(
                HostExpertExchange::bytes_for(capacity).unwrap(),
                capacity as usize * 6 * 8 + capacity as usize * 10240
            );
        }
    }

    /// The obsolete endpoint override is gone: exactly two endpoints (decode and
    /// prefill) are always reserved, and a stale non-2 value fails closed.
    #[test]
    fn endpoint_count_is_fixed_at_two_and_stale_overrides_are_rejected() {
        assert_eq!(parse_rdma_endpoints(None).unwrap(), 2);
        assert_eq!(parse_rdma_endpoints(Some("2")).unwrap(), 2);
        assert_eq!(parse_rdma_endpoints(Some(" 2 ")).unwrap(), 2);
        for obsolete in ["1", "0", "3", "16", "x", ""] {
            assert!(
                parse_rdma_endpoints(Some(obsolete)).is_err(),
                "{obsolete:?} must be rejected"
            );
        }
    }

    /// The runtime ring budget equals the two-endpoint allowance that admission
    /// already charged, and a peer advertising larger negotiated slots (or a
    /// second oversized endpoint) is rejected before any allocation.
    #[test]
    fn runtime_ring_budget_bounds_aggregate_endpoint_advertisements() {
        use ds41rt_transport::RingBudget;
        let topology = V41SparkTopology::new(2, 2).unwrap();
        let mut config = config(4, 0, Some(topology));
        config.capacity = 4096;
        let limit = spark_transport_bytes(&config).unwrap();
        // One capacity-sized endpoint is half the allowance; two fit exactly.
        let planned_endpoint = spark_ring_bytes(
            config.capacity,
            config.max_frame_bytes,
            env_usize("DS41RT_VERBS_HOST_RING_DEPTH", 8).unwrap(),
            env_usize("DS41RT_VERBS_HOST_RING_SLOT_BYTES", 8 << 20).unwrap(),
            ds41rt_transport::verbs_host_capabilities().preferred_alignment,
            1,
        )
        .unwrap();
        assert_eq!(limit, 2 * planned_endpoint);
        let budget = RingBudget::new(limit);
        // A peer whose advertised rings are larger than the planned allowance
        // cannot even get one endpoint admitted.
        assert!(budget.reserve(limit + 1).is_err());
        assert_eq!(budget.used(), 0);
        // Two endpoints of a larger negotiated size are bounded in aggregate:
        // the first fits, the second is rejected, and dropping the first
        // releases the credit for a compliant peer.
        let oversize = planned_endpoint + (1 << 20);
        assert!(oversize > planned_endpoint && 2 * oversize > limit);
        let first = budget.reserve(oversize).unwrap();
        assert!(budget.reserve(oversize).is_err());
        assert_eq!(budget.used(), oversize);
        drop(first);
        assert_eq!(budget.used(), 0);
        assert!(budget.reserve(planned_endpoint).is_ok());
    }
}

/// Reject an explicit topology before any weight allocation: it is defined only
/// for the official native checkpoint, its rank count must match `--world`, and
/// every rank must be inside the topology. The legacy world 2/4 rules are kept
/// exactly (two ranks still require EXL3).
fn validate_topology(config: &NativeExpertServiceConfig, catalog: &OfficialV41Catalog) -> Result<()> {
    if let Some(topology) = config.topology {
        crate::v41_spark_topology::require_native(Some(topology), catalog)?;
        ensure!(
            config.world == topology.world_size() && config.rank < config.world,
            "explicit Spark topology {}x{} needs --world {} and --rank below it, got world {} rank {}",
            topology.tp(),
            topology.ep(),
            topology.world_size(),
            config.world,
            config.rank
        );
        return Ok(());
    }
    ensure!(
        matches!(config.world, 2 | 4) && config.rank < config.world,
        "Spark world must be 2 or 4 and rank must be below world"
    );
    ensure!(
        config.world == 4 || catalog.exl3().is_some(),
        "two Spark ranks require EXL3 experts"
    );
    Ok(())
}

impl NativeExpertServiceConfig {
    /// Resident layer selection for the running checkpoint format. Explicit
    /// replicated topologies always select the native generic shard; the legacy
    /// path keeps the fixed TP4/EXL3-TP2 behavior byte-for-byte.
    fn selection(&self, layer: usize) -> Result<ExpertLayer> {
        let Some(topology) = self.topology else {
            return Ok(if self.world == 2 {
                ExpertLayer::BackboneTp2 { layer, rank: self.rank }
            } else {
                ExpertLayer::Backbone { layer, rank: self.rank }
            });
        };
        let shard = topology.tp_rank(self.rank)? as usize;
        Ok(match topology.tp() {
            4 => ExpertLayer::Backbone { layer, rank: shard },
            world => ExpertLayer::BackboneReplicatedTp { layer, rank: shard, world: world as usize },
        })
    }
    /// Replicated group this worker unpacks routes for, or `None` for legacy.
    fn native_group(&self) -> Result<Option<u8>> {
        crate::v41_spark_topology::group_of(self.topology, self.rank)
    }
    /// Resolve this rank's EXL3 AOT package for the running checkpoint's
    /// decoder tiers (multi-family images) with the legacy single-family
    /// location as fallback. An explicit --exl3-aot-dir is used verbatim.
    fn exl3_directory_for(&self, tiers: &[usize]) -> PathBuf {
        self.exl3_aot_dir.clone().unwrap_or_else(|| {
            crate::v41_experts::exl3::aot_layout_directory(
                &self.library,
                tiers,
                &format!("tp{}-rank{}", self.world, self.rank),
            )
        })
    }
}
