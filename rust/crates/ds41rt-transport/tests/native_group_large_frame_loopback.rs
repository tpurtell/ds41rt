//! Direct large-frame transport proof: real RC ring growth + reconnect + budget
//! release + byte-exact planes, without the model or a GPU.
//!
//! Tests:
//!
//! 1. `large_frame_boundaries_and_wire_roundtrip` (CPU, always runs): checks the
//!    818/819 response and 1555/1556 request boundaries against the transport's
//!    own wire formula, and round-trips a **TP2EP2** native-group request at
//!    M=819 and M=1556 with a real per-expert owner map (ownership coverage).
//!
//! 2. `native_group_large_frame_wire_loopback_ep1` (`#[ignore]`): **EP1 wire
//!    transport** test only. N=4 `TP4EP1` (or N=2 `TP2EP1`) self-RC loopback on
//!    the coordinator's fabric address. Server: a dedicated acceptor thread
//!    (production-shaped) admits
//!    `LocalVerbsExpertConnection::accept_with_budget` and hands it to a poller,
//!    which echoes host memory (no expert kernel, no GPU compute) under a
//!    synthetic `RingBudget`; client: existing `V41Tp4Roce`. M=1 -> 819 -> 1556 forces the
//!    response ring and then the request ring to grow, dropping/reconnecting the
//!    session so the old `RingReservation` is released. It does **not** claim
//!    replicated-group ownership coverage (that is test 1 / the roundtrip test).
//!
//! Live run (only in an authorized window; needs a real RoCE HCA, the staged
//! native library, and a per-host device map for the coordinator address):
//!
//! ```bash
//! DS41RT_LOOPBACK_IP=10.55.0.22 \
//! DS41RT_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP=10.55.0.22=mlx5_0 \
//! DS41RT_NATIVE_LIB=<staged>/libds41rt_native.so \
//!   timeout 180 cargo test -p ds41rt-transport --release \
//!   --test native_group_large_frame_loopback -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Synthetic transport credit only; not an admission or E2E-reachability claim.

use std::net::{IpAddr, SocketAddr, TcpListener};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use anyhow::{bail, ensure, Context, Result};
use ds41rt_transport::v41_expert::{
    V41BackboneRequest, V41SparkTopology, V41Tp4ChunkReceiver, V41Tp4Roce,
    V41_NATIVE_GROUP_REQUEST_FLAG, V41_NATIVE_UNASSIGNED_EXPERT_ID, V41_PARTIAL_ROW_BYTES,
    V41_ROUTED_EXPERTS, EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16,
};
use ds41rt_transport::{
    ExpertProtocolV2Request, ExpertProtocolV2RequestView, ExpertProtocolV2RouteEntry,
    ExpertProtocolV2RowDescriptor, ExpertV2Dtype, ExpertV2SourceKind, LocalVerbsExpertConnection,
    ProtocolV2ExecutorResponseRef, RingBudget, TcpTransportConfig,
    EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN, EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN,
};

const HIDDEN: usize = 5120;
const K32_ROW_BYTES: usize = 5280;
const REQUEST_ROW_BYTES: usize = 40 + 6 * 12 + K32_ROW_BYTES;
const RESPONSE_ROW_BYTES: usize = 4 + V41_PARTIAL_ROW_BYTES as usize;
const SLOT: usize = 8 << 20;
const DEPTH: usize = 8;
/// One-direction floor span: 8 MiB slot x depth.
const FLOOR_DIRECTION_SPAN: usize = SLOT * DEPTH;
/// Whole floor endpoint: request + response spans, the value the worker charges.
const FLOOR_ENDPOINT_SPAN: usize = 2 * FLOOR_DIRECTION_SPAN;
const MAX_FRAME: usize = 64 << 20;
const CAPACITY: u32 = 4096;
/// Synthetic credit bound for the live test (<= 4 GiB per the review).
const RING_LIMIT: usize = 4 << 30;
const TIMEOUT: Duration = Duration::from_secs(120);

fn request_wire(rows: u32) -> usize {
    EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN + rows as usize * REQUEST_ROW_BYTES
}
fn response_wire(rows: u32) -> usize {
    EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN + rows as usize * RESPONSE_ROW_BYTES
}

/// Deterministic finite BF16 bit pattern for one output element.
fn echo_bits(row: usize, col: usize) -> u16 {
    (((row * 31 + col * 7 + 1) & 0x3ff) as u16) | 0x1000
}

fn build_native_request(rows: u32, request_id: u64) -> Result<ExpertProtocolV2Request> {
    ensure!(rows > 0 && rows <= CAPACITY, "rows out of range");
    let mut rng = 0x5eed_1234_9abc_def0u64;
    let mut next = || {
        rng = rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let descriptors: Vec<ExpertProtocolV2RowDescriptor> = (0..rows)
        .map(|row| ExpertProtocolV2RowDescriptor {
            row_id: row as u64,
            source_kind: ExpertV2SourceKind::Prefill,
            source_request_id: request_id,
            token_position: row as u64,
            route_offset: row * 6,
            route_count: 6,
        })
        .collect();
    let mut routes = Vec::with_capacity(rows as usize * 6);
    for row in 0..rows {
        let mut chosen = [0u32; 6];
        for slot in 0..6 {
            loop {
                let expert = (next() % V41_ROUTED_EXPERTS as u64) as u32;
                if !chosen[..slot].contains(&expert) {
                    chosen[slot] = expert;
                    break;
                }
            }
        }
        for &expert in &chosen {
            routes.push(ExpertProtocolV2RouteEntry {
                row_index: row,
                expert_id: expert,
                gate_weight: 1.0 + (row % 5) as f32,
            });
        }
    }
    let mut request = ExpertProtocolV2Request::new(
        request_id,
        7,
        20,
        HIDDEN as u32,
        ExpertV2Dtype::Fp8E4m3Ue8m0K32,
        descriptors,
        routes,
        vec![0u8; rows as usize * K32_ROW_BYTES],
    )?;
    request.header.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
    Ok(request)
}

/// CPU: boundaries plus a real TP2EP2 owner-map round trip at M=819/1556.
#[test]
fn large_frame_boundaries_and_wire_roundtrip() -> Result<()> {
    assert_eq!(request_wire(128), 96 + 128 * 5392);
    assert_eq!(response_wire(128), 96 + 128 * 4 + 128 * 10240);
    assert!(response_wire(818) <= SLOT, "818 rows must fit the 8 MiB slot");
    assert!(response_wire(819) > SLOT, "819 rows must force response growth");
    assert!(request_wire(1555) <= SLOT, "1555 rows must fit the 8 MiB slot");
    assert!(request_wire(1556) > SLOT, "1556 rows must force request growth");
    assert!(response_wire(128) < SLOT && request_wire(128) < SLOT);
    assert_eq!(FLOOR_ENDPOINT_SPAN, 134_217_728);

    // TP2EP2 with an explicit owner map: real replicated-group coverage.
    let topology = V41SparkTopology::NATIVE_TP2_EP2;
    let mut owners = vec![0u8; V41_ROUTED_EXPERTS];
    for (expert, owner) in owners.iter_mut().enumerate() {
        *owner = (expert % topology.group_count() as usize) as u8;
    }
    for rows in [819u32, 1556] {
        let request = build_native_request(rows, 100 + rows as u64)?;
        let canonical: Vec<(u32, u32)> = request
            .routes
            .iter()
            .map(|route| (route.expert_id, route.gate_weight.to_bits()))
            .collect();
        let mut flagged = request.clone();
        flagged.with_native_group_owners(&owners, topology)?;
        ensure!(flagged.header.flags & V41_NATIVE_GROUP_REQUEST_FLAG != 0);
        let frame = flagged.encode()?;
        ensure!(frame.len() == request_wire(rows));
        let native = V41BackboneRequest::parse_native_group(&frame, CAPACITY, topology)?;

        let mut owned_per_route = vec![0usize; canonical.len()];
        for rank in 0..topology.world_size() {
            let group = topology.group(rank)?;
            let mut ids = vec![-1i32; canonical.len()];
            let mut weights = vec![-1.0f32; canonical.len()];
            native.copy_native_group_routes_into(&mut ids, &mut weights, group)?;
            for (index, (expert, weight)) in canonical.iter().enumerate() {
                if owners[*expert as usize] == group {
                    ensure!(ids[index] == *expert as i32);
                    ensure!(weights[index].to_bits() == *weight);
                    owned_per_route[index] += 1;
                } else {
                    ensure!(ids[index] == V41_NATIVE_UNASSIGNED_EXPERT_ID);
                    ensure!(weights[index] == 0.0);
                }
            }
        }
        // Exactly-once per TP shard: TP ranks of the one owning group.
        for owned in &owned_per_route {
            ensure!(*owned == topology.tp() as usize);
        }
        // Chunked response coverage still holds at these sizes.
        let executors = topology.executor_ids();
        let chunk_cap = native.response_chunk_rows(MAX_FRAME)?;
        let mut receiver = V41Tp4ChunkReceiver::new_ranks(&native, &executors, MAX_FRAME)?;
        for rank in (0..executors.len()).rev() {
            let mut indices = vec![0u32; chunk_cap as usize];
            let mut row = 0u32;
            while row < rows {
                let chunk_rows = (rows - row).min(chunk_cap);
                let payload = vec![0u8; chunk_rows as usize * V41_PARTIAL_ROW_BYTES as usize];
                let chunk = native
                    .response_chunk(executors[rank], row, &payload, &mut indices, MAX_FRAME)?
                    .to_owned()?
                    .encode()?;
                receiver.push(&chunk, |actual, start, bytes| {
                    ensure!(actual == rank && start == row);
                    ensure!(bytes.len() == payload.len());
                    Ok(())
                })?;
                row += chunk_rows;
            }
        }
        ensure!(receiver.complete());
        ensure!(receiver.received_rows_slice() == &[rows; 6][..executors.len()]);
    }
    Ok(())
}

struct AcceptReport {
    rank: usize,
    accept_index: usize,
    request_capacity: usize,
    request_span: usize,
    response_capacity: usize,
    response_span: usize,
    depth: usize,
    peak: usize,
}

struct Servers {
    stop: Arc<AtomicBool>,
    handles: Vec<thread::JoinHandle<()>>,
}
impl Drop for Servers {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

/// Production-shaped worker: a dedicated acceptor thread admits connections
/// (like `v41-roce-bootstrap` in `service/local.rs`) and hands them to a poller
/// thread, so a resize reconnect is admitted while the previous connection is
/// still being polled and reaped. The earlier single-threaded per-rank server
/// could not accept a reconnect while parked on the old connection, which
/// stalled the client's ready read; that was a harness defect, not a transport
/// one. Assertions are unchanged.
fn spawn_echo_servers(
    ip: IpAddr,
    ranks: usize,
    topology: V41SparkTopology,
    budget: Arc<RingBudget>,
    stop: Arc<AtomicBool>,
) -> Result<(Vec<SocketAddr>, Receiver<AcceptReport>, Servers)> {
    let (tx, rx) = mpsc::channel();
    let mut addrs = Vec::with_capacity(ranks);
    let mut handles = Vec::with_capacity(ranks * 2);
    for rank in 0..ranks {
        let listener = TcpListener::bind(SocketAddr::new(ip, 0)).context("bind loopback expert")?;
        addrs.push(listener.local_addr()?);
        listener.set_nonblocking(true)?;
        let executor_id = topology.executor_id(rank)?;
        let (admit_tx, admit_rx) = mpsc::channel::<LocalVerbsExpertConnection>();

        let acceptor_budget = Arc::clone(&budget);
        let acceptor_stop = Arc::clone(&stop);
        let acceptor_tx = tx.clone();
        handles.push(thread::spawn(move || {
            let mut accept_index = 0usize;
            while !acceptor_stop.load(Ordering::Acquire) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        match LocalVerbsExpertConnection::accept_with_budget(
                            stream,
                            MAX_FRAME,
                            false,
                            Arc::clone(&acceptor_budget),
                        ) {
                            Ok(connection) => {
                                accept_index += 1;
                                let (
                                    request_capacity,
                                    request_span,
                                    response_capacity,
                                    response_span,
                                    depth,
                                ) = connection.negotiated_ring_geometry();
                                let _ = acceptor_tx.send(AcceptReport {
                                    rank,
                                    accept_index,
                                    request_capacity,
                                    request_span,
                                    response_capacity,
                                    response_span,
                                    depth,
                                    peak: acceptor_budget.peak(),
                                });
                                if admit_tx.send(connection).is_err() {
                                    break;
                                }
                            }
                            Err(error) => {
                                eprintln!("rank {rank} loopback admission failed: {error:#}");
                            }
                        }
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(1));
                    }
                    Err(error) => {
                        eprintln!("rank {rank} loopback accept error: {error}");
                        break;
                    }
                }
            }
        }));

        let poller_stop = Arc::clone(&stop);
        handles.push(thread::spawn(move || {
            let mut connections: Vec<LocalVerbsExpertConnection> = Vec::new();
            while !poller_stop.load(Ordering::Acquire) {
                while let Ok(connection) = admit_rx.try_recv() {
                    connections.push(connection);
                }
                let mut dead = Vec::new();
                let mut progressed = false;
                for (index, connection) in connections.iter_mut().enumerate() {
                    match connection.poll(None, |view, _mapped, emit| {
                        echo_request(view, topology, executor_id, emit)
                    }) {
                        Ok(did) => progressed |= did,
                        Err(_) => dead.push(index),
                    }
                }
                for index in dead.into_iter().rev() {
                    connections.swap_remove(index);
                }
                if !progressed {
                    thread::sleep(Duration::from_micros(200));
                }
            }
        }));
    }
    Ok((addrs, rx, Servers { stop, handles }))
}

fn echo_request(
    view: &ExpertProtocolV2RequestView<'_>,
    topology: V41SparkTopology,
    executor_id: u64,
    emit: &mut dyn FnMut(ProtocolV2ExecutorResponseRef<'_>) -> Result<()>,
) -> Result<()> {
    let native = V41BackboneRequest::parse_native_group(view.frame_bytes(), CAPACITY, topology)?;
    let rows = native.rows();
    let mut plane = vec![0u8; rows as usize * V41_PARTIAL_ROW_BYTES as usize];
    for row in 0..rows as usize {
        for col in 0..HIDDEN {
            let offset = (row * HIDDEN + col) * 2;
            plane[offset..offset + 2].copy_from_slice(&echo_bits(row, col).to_le_bytes());
        }
    }
    let chunk_rows = native.response_chunk_rows(MAX_FRAME)? as usize;
    let mut indices = vec![0u32; chunk_rows];
    for start in (0..rows as usize).step_by(chunk_rows) {
        let end = (start + chunk_rows).min(rows as usize);
        let bytes = &plane
            [start * V41_PARTIAL_ROW_BYTES as usize..end * V41_PARTIAL_ROW_BYTES as usize];
        let response =
            native.response_chunk(executor_id, start as u32, bytes, &mut indices, MAX_FRAME)?;
        emit(ProtocolV2ExecutorResponseRef::Host(response))?;
    }
    Ok(())
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "requires a live RoCE HCA on the coordinator fabric IP, DS41RT_NATIVE_LIB, and the per-host device map; host-memory echo, no GPU"]
async fn native_group_large_frame_wire_loopback_ep1() -> Result<()> {
    // 127.0.0.1 has no RoCE GID; bind the coordinator's real fabric address and
    // require an explicit device-map entry so native selects that HCA.
    let loopback_ip: IpAddr = std::env::var("DS41RT_LOOPBACK_IP")
        .unwrap_or_else(|_| "10.55.0.22".to_owned())
        .parse()
        .context("DS41RT_LOOPBACK_IP is not an IP address")?;
    let map = std::env::var("DS41RT_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP").context(
        "set DS41RT_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP=<loopback-ip>=<hca>, e.g. 10.55.0.22=mlx5_0",
    )?;
    let entry = format!("{loopback_ip}=");
    ensure!(
        map.split(',').any(|value| value.trim().starts_with(&entry)),
        "device map must contain {entry}<hca> for the loopback control IP"
    );

    let ranks: usize = std::env::var("DS41RT_LARGE_FRAME_RANKS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(4);
    let topology = match ranks {
        6 => V41SparkTopology::NATIVE_TP6_EP1,
        4 => V41SparkTopology::NATIVE_TP4_EP1,
        2 => V41SparkTopology::NATIVE_TP2_EP1,
        other => bail!("loopback ranks must be 6 (TP6EP1), 4 (TP4EP1) or 2 (TP2EP1), got {other}"),
    };
    ensure!(topology.world_size() == ranks);

    let budget = RingBudget::new(RING_LIMIT);
    let stop = Arc::new(AtomicBool::new(false));
    let (addrs, reports, _servers) =
        spawn_echo_servers(loopback_ip, ranks, topology, Arc::clone(&budget), Arc::clone(&stop))?;

    let config = TcpTransportConfig {
        timeout: Duration::from_secs(30),
        max_frame_bytes: MAX_FRAME,
        timing: true,
    };
    let mut transport = V41Tp4Roce::new_topology(topology, &addrs, CAPACITY, config)?;

    tokio::time::timeout(TIMEOUT, async {
        for &rows in &[1u32, 819, 1556] {
            let mut request = build_native_request(rows, 1000 + rows as u64)?;
            request.with_native_group_owners(&[0u8; V41_ROUTED_EXPERTS], topology)?;
            let mut seen = vec![vec![false; rows as usize]; ranks];
            transport
                .execute(&request, |rank, start, bytes| {
                    ensure!(rank < ranks);
                    let chunk_rows = bytes.len() / V41_PARTIAL_ROW_BYTES as usize;
                    for chunk_row in 0..chunk_rows {
                        let row = start as usize + chunk_row;
                        ensure!(row < rows as usize);
                        for col in 0..HIDDEN {
                            let offset = chunk_row * V41_PARTIAL_ROW_BYTES as usize + col * 2;
                            let bits = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
                            ensure!(
                                bits == echo_bits(row, col),
                                "rank {rank} row {row} col {col} differs"
                            );
                        }
                        seen[rank][row] = true;
                    }
                    Ok(())
                })
                .await?;
            for (rank, rows_seen) in seen.iter().enumerate() {
                ensure!(
                    rows_seen.iter().all(|value| *value),
                    "rank {rank} did not observe every row at M={rows}"
                );
            }
        }
        Ok::<(), anyhow::Error>(())
    })
    .await
    .context("loopback large-frame test exceeded the 120 s bound")??;

    // Wait for the server to reap stale endpoints (>= 1 s liveness) so only the
    // final grown endpoint per rank stays reserved.
    thread::sleep(Duration::from_millis(2500));

    let mut by_rank: Vec<Vec<AcceptReport>> = (0..ranks).map(|_| Vec::new()).collect();
    let mut peak = 0usize;
    while let Ok(report) = reports.try_recv() {
        peak = peak.max(report.peak);
        by_rank[report.rank].push(report);
    }
    for rank_reports in by_rank.iter_mut() {
        rank_reports.sort_by_key(|report| report.accept_index);
    }
    for rank in 0..ranks {
        let rank_reports = &by_rank[rank];
        ensure!(
            rank_reports.len() == 3,
            "rank {rank} expected 3 endpoints (floor + response grow + request grow), saw {}",
            rank_reports.len()
        );
        let floor = &rank_reports[0];
        ensure!(
            floor.request_capacity == SLOT
                && floor.response_capacity == SLOT
                && floor.request_span == FLOOR_DIRECTION_SPAN
                && floor.response_span == FLOOR_DIRECTION_SPAN
                && floor.depth == DEPTH,
            "rank {rank} first endpoint is not the 8 MiB x {DEPTH} floor"
        );
        let grown = &rank_reports[1];
        ensure!(
            grown.response_span > FLOOR_DIRECTION_SPAN,
            "rank {rank} response ring did not grow at M=819"
        );
        ensure!(
            grown.request_span == FLOOR_DIRECTION_SPAN,
            "rank {rank} request ring grew too early at M=819"
        );
        let largest = &rank_reports[2];
        ensure!(
            largest.request_span > FLOOR_DIRECTION_SPAN,
            "rank {rank} request ring did not grow at M=1556"
        );
        ensure!(
            largest.response_span >= grown.response_span,
            "rank {rank} response ring shrank at M=1556"
        );
        ensure!(largest.depth == DEPTH);
    }

    let expected_live: usize = by_rank
        .iter()
        .map(|rank_reports| {
            let last = rank_reports.last().expect("three reports");
            last.request_span + last.response_span
        })
        .sum();
    ensure!(
        budget.used() == expected_live,
        "stale reservation not released while live: used={} live={expected_live}",
        budget.used()
    );
    ensure!(
        expected_live <= RING_LIMIT && peak <= RING_LIMIT,
        "synthetic ring budget exceeded: live={expected_live} peak={peak} limit={RING_LIMIT}"
    );

    // Closing the client must drop every reservation.
    stop.store(true, Ordering::Release);
    drop(transport);
    for _ in 0..50 {
        if budget.used() == 0 {
            break;
        }
        thread::sleep(Duration::from_millis(100));
    }
    ensure!(
        budget.used() == 0,
        "all endpoints dropped but budget still holds {} bytes",
        budget.used()
    );
    println!("PASS ranks={ranks} peak={peak} live={expected_live}");
    Ok(())
}
