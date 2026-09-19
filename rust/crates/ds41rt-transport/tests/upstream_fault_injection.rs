//! Upstream fault-injection coverage ported to ds41rt-transport (component C9).
//!
//! Sources (invariant classes extracted, proxy architecture NOT ported):
//! - sglang sgl-router tests/proxy/timeout.rs
//!   -> deadline expiry: a stalled upstream must error at the configured
//!      client deadline, never wedge the caller.
//! - sglang sgl-router tests/proxy/failover.rs
//!   -> worker death mid-traffic: a dead/reset connection must surface a clean
//!      error and the client must retry/reconnect on a fresh connection.
//! - sglang sgl-router tests/proxy/graceful_shutdown.rs
//!   -> drain/cancellation: a partial (mid-stream) response must never be
//!      reported as complete; the truncated connection is discarded and the
//!      next request starts on a fresh connection (no replay of the
//!      partially delivered wave).
//! - vllm tests/v1/executor/test_multiproc_executor_timeout.py
//!   -> stale-deadline math: an absolute deadline re-based against the
//!      current monotonic clock must clamp to zero, never go negative, and
//!      `None` (no deadline) must propagate as `None`. ds41rt-transport
//!      exposes no absolute-deadline helper (its TCP client uses fresh
//!      per-phase `tokio::time::timeout` budgets), so the math is ported as
//!      contract-model tests the same way the upstream file models the
//!      contract with fakes.
//! - vllm tests/distributed/test_mq_connect_ip.py
//!   -> NOT PORTED: ds41rt-transport has no public address-resolution
//!      surface. TCP entry points take `SocketAddr` directly; rail-address
//!      resolution (`resolve_verbs_host_rail_addr`) is private and reachable
//!      only through the verbs-host (native RDMA) path, which cannot run in a
//!      hermetic test.
//!
//! Unmapped upstream invariants (no ds41rt-transport surface):
//! - circuit-breaker exclusion of failed workers from round-robin
//!   (failover.rs): ds41rt has no breaker/health registry; a failed host-batch
//!   target fails the whole dispatch until it is reachable again.
//! - HTTP-level graceful-drain guarantee (graceful_shutdown.rs):
//!   `serve_protocol_v2_tcp_listener_with_executor` runs an infinite accept
//!   loop with no shutdown handle, so "server stops accepting but drains
//!   in-flight streams" cannot be expressed here. Cancellation of the CLIENT
//!   side mid-stream IS covered below.
//! - connect-IP normalization (test_mq_connect_ip.py), see above.
//!
//! All servers are loopback `127.0.0.1:0` listeners; no native/ffi calls.

// COVERAGE CLASS: standalone reference oracle. These cases document upstream
// behavior with no ds41rt dependency; they cannot detect product regressions
// by themselves. They are the comparison references for the deferred GPU
// parity tests (docs/test-coverage/DEFERRED.md) and are counted separately
// from product regression coverage (review MAJOR 4/6, 2026-09-15).

use anyhow::{Context, Result};
use ds41rt_core::{
    DType, ExpertBatch, ExpertBatchRoute, ExpertBatchRow, ExpertHostBatchSet, GraphBucket, LayerId,
    ModelFacts, PlacementPolicy, PlacementVersion, PositionId, RequestId, RowSourceKind,
    DS4_FLASH_ROUTED_EXPERTS,
};
use ds41rt_transport::{
    ExpertProtocolV2Request, ExpertProtocolV2Response, ExpertProtocolV2RouteEntry,
    ExpertProtocolV2RowDescriptor, ExpertProtocolV2Status, ExpertV2Dtype, ExpertV2SourceKind,
    TcpProtocolV2HostBatchSetPersistentClient, TcpProtocolV2HostBatchTarget,
    TcpProtocolV2PersistentClient, TcpTransportConfig, EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN,
};
use std::collections::{HashSet, VecDeque};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinHandle;

const TEST_HIDDEN_DIM: u32 = 4;
const FAST_TIMEOUT: Duration = Duration::from_millis(200);

// ---------------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------------

fn test_request(request_id: u64) -> Result<ExpertProtocolV2Request> {
    let rows = vec![ExpertProtocolV2RowDescriptor {
        row_id: 0,
        source_kind: ExpertV2SourceKind::Decode,
        source_request_id: request_id,
        token_position: 0,
        route_offset: 0,
        route_count: 1,
    }];
    let routes = vec![ExpertProtocolV2RouteEntry {
        row_index: 0,
        expert_id: 3,
        gate_weight: 0.5,
    }];
    let mut hidden_payload = Vec::with_capacity(TEST_HIDDEN_DIM as usize * 2);
    for value in [0.25_f32, 0.5, 0.75, 1.0] {
        hidden_payload.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
    }
    ExpertProtocolV2Request::new(
        request_id,
        0x51CE,
        13,
        TEST_HIDDEN_DIM,
        ExpertV2Dtype::Bf16,
        rows,
        routes,
        hidden_payload,
    )
}

async fn read_request(stream: &mut TcpStream) -> Result<ExpertProtocolV2Request> {
    let mut frame = vec![0_u8; EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN];
    stream.read_exact(&mut frame).await?;
    let wire_bytes = ExpertProtocolV2Request::wire_bytes_from_header(&frame)?;
    frame.resize(wire_bytes, 0);
    stream
        .read_exact(&mut frame[EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN..])
        .await?;
    ExpertProtocolV2Request::decode(&frame)
}

fn echo_response(request: &ExpertProtocolV2Request) -> Result<ExpertProtocolV2Response> {
    ExpertProtocolV2Response::new(
        request.header.request_id,
        request.header.placement_version,
        request.header.layer_id,
        request.header.row_count,
        request.header.hidden_dim,
        request.header.hidden_dtype,
        ExpertProtocolV2Status::Ok,
        request.hidden_payload.to_vec(),
    )
}

/// One healthy worker connection: serves requests until the peer goes away.
async fn serve_echo_connection(mut stream: TcpStream) {
    while let Ok(request) = read_request(&mut stream).await {
        let Ok(frame) = echo_response(&request).and_then(|r| r.encode()) else {
            break;
        };
        if stream.write_all(&frame).await.is_err() || stream.flush().await.is_err() {
            break;
        }
    }
}

/// Accept loop that owns its connection tasks. Signalling `shutdown_rx`
/// (or dropping its sender) closes the listener AND every accepted
/// connection, modelling whole-worker death rather than a bare accept-loop
/// abort that would leave orphaned handlers serving existing sockets.
fn spawn_echo_accept_loop(
    listener: TcpListener,
    accepts: Arc<AtomicUsize>,
    mut shutdown_rx: tokio::sync::oneshot::Receiver<()>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut handlers = tokio::task::JoinSet::new();
        loop {
            tokio::select! {
                _ = &mut shutdown_rx => break,
                accepted = listener.accept() => {
                    let Ok((stream, _peer)) = accepted else {
                        break;
                    };
                    accepts.fetch_add(1, Ordering::SeqCst);
                    handlers.spawn(serve_echo_connection(stream));
                }
            }
        }
        handlers.abort_all();
    })
}

/// Healthy reference worker: serves every request on a persistent connection
/// until shut down via the returned sender.
async fn spawn_echo_server() -> Result<(
    SocketAddr,
    tokio::sync::oneshot::Sender<()>,
    Arc<AtomicUsize>,
)> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let accepts = Arc::new(AtomicUsize::new(0));
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    spawn_echo_accept_loop(listener, Arc::clone(&accepts), shutdown_rx);
    Ok((addr, shutdown_tx, accepts))
}

async fn wait_for_port_closed(addr: SocketAddr) {
    tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            if TcpStream::connect(addr).await.is_err() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("port should refuse connections after server shutdown");
}

async fn rebind_listener(addr: SocketAddr) -> Result<TcpListener> {
    let started = Instant::now();
    loop {
        match TcpListener::bind(addr).await {
            Ok(listener) => return Ok(listener),
            Err(err) => {
                if started.elapsed() > Duration::from_secs(3) {
                    return Err(err).with_context(|| format!("rebinding {addr}"));
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// vLLM test_multiproc_executor_timeout.py — stale-deadline / timeout math.
// Ported as contract models: ds41rt-transport has no absolute-deadline helper;
// its TCP client issues a fresh per-phase tokio timeout, so a negative or
// stale deadline can never be constructed. These tests pin the invariant the
// upstream suite pins: deadline - now, clamped at zero; None stays None.
// ---------------------------------------------------------------------------

struct FakeClock {
    now_secs: f64,
}

impl FakeClock {
    fn new(start: f64) -> Self {
        Self { now_secs: start }
    }
    fn monotonic(&self) -> f64 {
        self.now_secs
    }
    fn advance(&mut self, seconds: f64) {
        self.now_secs += seconds;
    }
}

/// Port of vLLM `_dequeue_timeout`: a deadline captured earlier in the call
/// must be re-based against the CURRENT monotonic time; once expired the
/// wait primitive receives 0.0, never a negative timeout.
fn dequeue_timeout(deadline: Option<f64>, now: f64) -> Option<f64> {
    deadline.map(|deadline| (deadline - now).max(0.0))
}

/// Port of vLLM `recv_timeout_ms`: milliseconds must clamp negatives to 0
/// (a stale deadline must not surface as a negative socket timeout).
fn recv_timeout_ms(timeout: Option<f64>) -> Option<i64> {
    timeout.map(|timeout| ((timeout * 1000.0) as i64).max(0))
}

#[test]
fn stale_deadline_clamps_to_zero_and_never_goes_negative() {
    let mut clock = FakeClock::new(100.0);
    let deadline = clock.monotonic() + 1.0;
    clock.advance(5.0);

    let observed: Vec<Option<f64>> = vec![dequeue_timeout(Some(deadline), clock.monotonic())];
    assert_eq!(observed, vec![Some(0.0)]);
}

#[test]
fn fresh_deadline_passes_bounded_positive_timeout() {
    let mut clock = FakeClock::new(100.0);
    let deadline = clock.monotonic() + 5.0;
    clock.advance(2.0);

    let timeout = dequeue_timeout(Some(deadline), clock.monotonic())
        .expect("a deadline was set, so the dequeue must be bounded");
    assert!(timeout > 0.0, "fresh deadline must remain positive: {timeout}");
    assert!(timeout <= 5.0, "timeout must never exceed the original budget: {timeout}");
}

#[test]
fn no_deadline_passes_none_through_unchanged() {
    assert_eq!(dequeue_timeout(None, 123.456), None);
}

#[test]
fn recv_timeout_ms_clamps_negative_and_truncates_positive() {
    assert_eq!(recv_timeout_ms(None), None);
    assert_eq!(recv_timeout_ms(Some(-1.0)), Some(0));
    assert_eq!(recv_timeout_ms(Some(-0.001)), Some(0));
    assert_eq!(recv_timeout_ms(Some(0.0)), Some(0));
    assert_eq!(recv_timeout_ms(Some(0.001)), Some(1));
    assert_eq!(recv_timeout_ms(Some(2.5)), Some(2500));
}

/// Port of vLLM `test_future_wrapper_drains_pending_before_own_get_response`:
/// resolving one in-flight RPC must first pump every earlier-submitted
/// pending RPC (FIFO), so a response can never overtake an older request.
/// ds41rt satisfies this structurally on one persistent connection
/// (exclusive in-flight ownership per `dispatch_chunks`), but the ordering
/// contract itself is pinned here.
#[test]
fn response_pump_drains_earlier_pending_futures_first() {
    struct PendingQueue {
        pending: VecDeque<String>,
        completed: HashSet<String>,
        call_order: Vec<String>,
    }
    impl PendingQueue {
        fn submit(&mut self, label: &str) {
            self.pending.push_back(label.to_owned());
        }
        /// Mirrors `FutureWrapper.result()`: loop pumping responses until the
        /// requested future completes; each pump round services the oldest
        /// pending future first.
        fn resolve(&mut self, label: &str) {
            while !self.completed.contains(label) {
                let next = self.pending.pop_front().expect("pending future exists");
                self.call_order.push(next.clone());
                self.completed.insert(next);
            }
        }
    }

    let mut queue = PendingQueue {
        pending: VecDeque::new(),
        completed: HashSet::new(),
        call_order: Vec::new(),
    };
    queue.submit("first");
    queue.submit("second");

    queue.resolve("second");

    assert!(queue.completed.contains("first"));
    assert!(queue.completed.contains("second"));
    assert_eq!(queue.call_order, vec!["first", "second"]);
}

// ---------------------------------------------------------------------------
// sglang timeout.rs — upstream accepts the connection but never responds.
// ---------------------------------------------------------------------------

// Paused Tokio time: the client deadline fires deterministically via
// advance(), real TCP I/O still completes, and the wall-clock asserts are
// replaced by a generous wedge-watchdog only (review 2026-09-15: elapsed<1s
// could flake on an oversubscribed CI worker).
#[tokio::test(start_paused = true)]
async fn hanging_upstream_times_out_within_configured_deadline() -> Result<()> {
    // Worker accepts, reads the request, then sleeps far past the deadline.
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let server = tokio::spawn(async move {
        loop {
            let Ok((mut stream, _peer)) = listener.accept().await else {
                break;
            };
            tokio::spawn(async move {
                let _ = read_request(&mut stream).await;
                tokio::time::sleep(Duration::from_secs(5)).await;
            });
        }
    });

    let request = test_request(6_000)?;
    let config = TcpTransportConfig { timing: false,
        timeout: FAST_TIMEOUT,
        ..TcpTransportConfig::default()
    };

    let roundtrip = tokio::spawn(async move {
        ds41rt_transport::tcp_protocol_v2_roundtrip(addr, &request, config).await
    });
    // Drive the client's internal deadline; the mock never responds, so only
    // timers can complete the roundtrip.
    tokio::time::advance(FAST_TIMEOUT + Duration::from_millis(50)).await;
    // Generous wall-clock wedge-watchdog only; the correctness signal is the
    // deadline-expiry error below, not the elapsed time.
    let result = tokio::time::timeout(Duration::from_secs(60), roundtrip)
        .await
        .expect("watchdog: roundtrip must finish")
        .expect("roundtrip task must not panic");

    let err = result.expect_err("hanging upstream must fail, not wedge or succeed");
    let message = format!("{err:#}");
    assert!(
        message.contains("timed out reading ProtocolV2 response header"),
        "expected a deadline-expiry error, got: {message}"
    );
    server.abort();
    Ok(())
}

#[tokio::test(start_paused = true)]
async fn response_header_then_stall_times_out_on_payload_read() -> Result<()> {
    // Partial message: valid header, then the payload never arrives. The
    // deadline must also cover the payload phase, not just the header.
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let server = tokio::spawn(async move {
        loop {
            let Ok((mut stream, _peer)) = listener.accept().await else {
                break;
            };
            tokio::spawn(async move {
                let Ok(request) = read_request(&mut stream).await else {
                    return;
                };
                let Ok(response) = echo_response(&request) else {
                    return;
                };
                let Ok(frame) = response.encode() else {
                    return;
                };
                // Header only; payload bytes withheld indefinitely.
                if stream.write_all(&frame[..response.header_len()]).await.is_err() {
                    return;
                }
                let _ = stream.flush().await;
                tokio::time::sleep(Duration::from_secs(5)).await;
            });
        }
    });

    let request = test_request(6_010)?;
    let config = TcpTransportConfig { timing: false,
        timeout: FAST_TIMEOUT,
        ..TcpTransportConfig::default()
    };
    let roundtrip = tokio::spawn(async move {
        ds41rt_transport::tcp_protocol_v2_roundtrip(addr, &request, config).await
    });
    tokio::time::advance(FAST_TIMEOUT + Duration::from_millis(50)).await;
    let err = tokio::time::timeout(Duration::from_secs(60), roundtrip)
        .await
        .expect("watchdog: roundtrip must finish")
        .expect("roundtrip task must not panic")
        .expect_err("stalled payload must fail at the deadline");
    let message = format!("{err:#}");
    assert!(
        message.contains("timed out reading ProtocolV2 response payload"),
        "expected payload-phase deadline error, got: {message}"
    );
    server.abort();
    Ok(())
}

#[tokio::test]
async fn truncated_response_payload_is_rejected_without_wedging() -> Result<()> {
    // Partial message: peer closes mid-payload. read_exact must surface the
    // truncation immediately (EOF), not spin or succeed.
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let server = tokio::spawn(async move {
        loop {
            let Ok((mut stream, _peer)) = listener.accept().await else {
                break;
            };
            tokio::spawn(async move {
                let Ok(request) = read_request(&mut stream).await else {
                    return;
                };
                let Ok(response) = echo_response(&request) else {
                    return;
                };
                let Ok(frame) = response.encode() else {
                    return;
                };
                // Write header + half the payload, then drop the connection.
                let partial = response.header_len() + (frame.len() - response.header_len()) / 2;
                let _ = stream.write_all(&frame[..partial]).await;
            });
        }
    });

    let request = test_request(6_020)?;
    let started = Instant::now();
    let err = ds41rt_transport::tcp_protocol_v2_roundtrip(
        addr,
        &request,
        TcpTransportConfig::default(),
    )
    .await
    .expect_err("truncated response must be rejected");
    let elapsed = started.elapsed();

    assert!(
        elapsed < Duration::from_secs(2),
        "truncation must surface promptly; elapsed {elapsed:?}"
    );
    let message = format!("{err:#}");
    assert!(
        message.contains("reading ProtocolV2 response payload"),
        "expected truncation on the payload read, got: {message}"
    );
    server.abort();
    Ok(())
}

// ---------------------------------------------------------------------------
// sglang failover.rs — worker death mid-traffic.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn killed_first_connection_is_retried_on_a_fresh_connection() -> Result<()> {
    // Worker dies mid-traffic: the first connection is dropped by the server
    // right after it consumes the request. The persistent client must surface
    // the disconnect, discard the dead stream, and transparently retry the
    // same request on a fresh connection.
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let accepts = Arc::new(AtomicUsize::new(0));
    let accepts_server = Arc::clone(&accepts);
    let server = tokio::spawn(async move {
        loop {
            let Ok((mut stream, _peer)) = listener.accept().await else {
                break;
            };
            let connection = accepts_server.fetch_add(1, Ordering::SeqCst);
            tokio::spawn(async move {
                if connection == 0 {
                    // Worker death: consume the request, then drop the socket.
                    // Whether the client observes EOF on read or a reset on
                    // write is racy; both are connection-closed errors.
                    let _ = read_request(&mut stream).await;
                    drop(stream);
                    return;
                }
                while let Ok(request) = read_request(&mut stream).await {
                    let Ok(frame) = echo_response(&request).and_then(|r| r.encode()) else {
                        break;
                    };
                    if stream.write_all(&frame).await.is_err() || stream.flush().await.is_err() {
                        break;
                    }
                }
            });
        }
    });

    let mut client = TcpProtocolV2PersistentClient::new(addr, TcpTransportConfig::default());
    let request = test_request(6_100)?;
    let response = client
        .roundtrip(&request)
        .await
        .expect("client must transparently retry the request killed mid-traffic");
    assert_eq!(response.header.request_id, request.header.request_id);
    assert_eq!(
        accepts.load(Ordering::SeqCst),
        2,
        "killed connection plus the retry connection"
    );
    server.abort();
    Ok(())
}

#[tokio::test]
async fn dead_server_fails_fast_and_recovers_after_listener_restart() -> Result<()> {
    // Full worker death: the listener goes away between requests. The next
    // call must fail fast with a connect error (not wedge), and once a worker
    // is listening again the same client must recover via reconnect.
    let (addr, shutdown, _accepts) = spawn_echo_server().await?;
    let mut client = TcpProtocolV2PersistentClient::new(addr, TcpTransportConfig::default());
    let request = test_request(6_110)?;
    client.roundtrip(&request).await?;

    // Whole-worker death: listener and every accepted connection go away.
    let _ = shutdown.send(());
    wait_for_port_closed(addr).await;

    let started = Instant::now();
    let err = client
        .roundtrip(&request)
        .await
        .expect_err("roundtrip against a dead server must fail");
    let elapsed = started.elapsed();
    assert!(
        elapsed < Duration::from_secs(2),
        "connection-refused must surface promptly; elapsed {elapsed:?}"
    );
    let message = format!("{err:#}");
    assert!(
        message.contains("connecting TCP ProtocolV2 transport"),
        "expected a connect-phase error against the dead server, got: {message}"
    );

    // Worker restart on the same address: the client reconnects transparently.
    let listener = rebind_listener(addr).await?;
    let (restart_shutdown, restart_rx) = tokio::sync::oneshot::channel::<()>();
    let restarted = spawn_echo_accept_loop(listener, Arc::new(AtomicUsize::new(0)), restart_rx);
    let recovered = client
        .roundtrip(&request)
        .await
        .expect("client must recover once the worker is reachable again");
    assert_eq!(recovered.header.request_id, request.header.request_id);
    let _ = restart_shutdown.send(());
    let _ = restarted.await;
    Ok(())
}

#[tokio::test]
async fn host_batch_dispatch_recovers_after_single_host_death() -> Result<()> {
    // Multi-worker fan-out (failover.rs shape): two workers serve one host
    // batch set each. One worker dies; the dispatch must fail cleanly; after
    // it is restarted the persistent fan-out client must recover (its
    // reset-on-error contract drops all dead connections).
    let (addr_a, shutdown_a, accepts_a) = spawn_echo_server().await?;
    let (addr_b, shutdown_b, accepts_b) = spawn_echo_server().await?;
    let targets = vec![
        TcpProtocolV2HostBatchTarget {
            host: "ostrich".to_owned(),
            addr: addr_a,
        },
        TcpProtocolV2HostBatchTarget {
            host: "dodo".to_owned(),
            addr: addr_b,
        },
    ];
    let (set, global_hidden) = host_batch_fixture()?;
    let mut client = TcpProtocolV2HostBatchSetPersistentClient::new(
        targets,
        TcpTransportConfig::default(),
    );

    let first = client
        .dispatch_bf16(&set, &global_hidden, 6_200)
        .await
        .expect("initial fan-out dispatch succeeds");
    assert_eq!(first.stats.hosts, 2);

    // Kill worker "dodo" mid-traffic.
    let _ = shutdown_b.send(());
    wait_for_port_closed(addr_b).await;

    let err = client
        .dispatch_bf16(&set, &global_hidden, 6_210)
        .await
        .expect_err("dispatch with one dead host must fail");
    let message = format!("{err:#}");
    assert!(
        message.contains("connecting TCP ProtocolV2 transport"),
        "expected connect-phase failure for the dead host, got: {message}"
    );

    // Restart the dead worker; the failed dispatch reset every pooled
    // connection, so this dispatch reconnects both hosts from scratch.
    let listener = rebind_listener(addr_b).await?;
    let (restart_b_shutdown, restart_b_rx) = tokio::sync::oneshot::channel::<()>();
    let restarted_b = spawn_echo_accept_loop(listener, Arc::clone(&accepts_b), restart_b_rx);
    let recovered = client
        .dispatch_bf16(&set, &global_hidden, 6_220)
        .await
        .expect("fan-out dispatch must recover after the dead host restarts");
    assert_eq!(recovered.stats.hosts, 2);
    assert_eq!(
        accepts_b.load(Ordering::SeqCst),
        2,
        "dead worker saw the first dispatch and the post-restart dispatch only"
    );

    let _ = restart_b_shutdown.send(());
    let _ = restarted_b.await;
    let _ = shutdown_a.send(());
    let _ = accepts_a;
    Ok(())
}

fn host_batch_fixture() -> Result<(ExpertHostBatchSet, Vec<u8>)> {
    let batch = ExpertBatch {
        layer_id: LayerId(3),
        placement_version: PlacementVersion("upstream-fault-injection".to_owned()),
        hidden_dim: TEST_HIDDEN_DIM as usize,
        hidden_bytes_per_row: TEST_HIDDEN_DIM as usize * 2,
        hidden_dtype: DType::Bf16,
        routed_experts: DS4_FLASH_ROUTED_EXPERTS,
        graph_bucket: GraphBucket::new(2),
        quantization_recipe: ModelFacts::default().quantization_recipe,
        rows: vec![
            ExpertBatchRow {
                row_id: 0,
                source_kind: RowSourceKind::DecodeStep,
                request_id: RequestId("fault-row-0".to_owned()),
                sequence_id: "seq-0".to_owned(),
                token_position: PositionId(0),
                route_offset: 0,
                route_count: 1,
            },
            ExpertBatchRow {
                row_id: 1,
                source_kind: RowSourceKind::PrefillChunk,
                request_id: RequestId("fault-row-1".to_owned()),
                sequence_id: "seq-1".to_owned(),
                token_position: PositionId(1),
                route_offset: 1,
                route_count: 1,
            },
        ],
    };
    let routes = vec![
        ExpertBatchRoute {
            row_index: 0,
            expert_id: 0,
            gate_weight: 1.0,
        },
        ExpertBatchRoute {
            row_index: 1,
            expert_id: 1,
            gate_weight: 1.0,
        },
    ];
    let hosts = vec!["ostrich".to_owned(), "dodo".to_owned()];
    let set = ExpertHostBatchSet::from_expert_batch(&batch, &routes, &hosts, PlacementPolicy::Modulo)?;
    let global_hidden = [[0.0_f32, 0.25, 0.5, 0.75], [1.0, 1.25, 1.5, 1.75]]
        .iter()
        .flat_map(|row| {
            row.iter()
                .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
        })
        .collect();
    Ok((set, global_hidden))
}

// ---------------------------------------------------------------------------
// sglang graceful_shutdown.rs — drain/cancellation on the ds41rt surface.
// ds41rt-transport has no server-side drain API; the mappable invariant is
// client-side: a partially delivered (mid-stream) response must never be
// reported as complete, the truncated connection is discarded, and the next
// request starts on a fresh connection without replaying the partial wave.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn mid_stream_stall_fails_and_never_replays_the_partial_wave() -> Result<()> {
    // Worker streams chunk 1 of 2, then stalls past the deadline. The chunked
    // receive must fail; the sink saw the partial chunk but the call is an
    // error; the next request runs on a fresh connection.
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let accepts = Arc::new(AtomicUsize::new(0));
    let accepts_server = Arc::clone(&accepts);
    let server = tokio::spawn(async move {
        loop {
            let Ok((mut stream, _peer)) = listener.accept().await else {
                break;
            };
            let connection = accepts_server.fetch_add(1, Ordering::SeqCst);
            tokio::spawn(async move {
                let Ok(request) = read_request(&mut stream).await else {
                    return;
                };
                if connection == 0 {
                    // Deliver a non-final chunk, then hang: the stream stalls
                    // mid-flight like a worker stopped mid-drain.
                    // (The more-chunks flag requires the row-index table;
                    // set it via the constructor rather than raw header bits.)
                    let chunk = match echo_response(&request).and_then(|r| r.with_row_indices(vec![0], true)) {
                        Ok(chunk) => chunk,
                        Err(_) => return,
                    };
                    let Ok(frame) = chunk.encode() else {
                        return;
                    };
                    if stream.write_all(&frame).await.is_err() {
                        return;
                    }
                    let _ = stream.flush().await;
                    tokio::time::sleep(Duration::from_secs(5)).await;
                    return;
                }
                let Ok(frame) = echo_response(&request).and_then(|r| r.encode()) else {
                    return;
                };
                let _ = stream.write_all(&frame).await;
                let _ = stream.flush().await;
            });
        }
    });

    let config = TcpTransportConfig { timing: false,
        timeout: FAST_TIMEOUT,
        ..TcpTransportConfig::default()
    };
    let mut client = TcpProtocolV2PersistentClient::new(addr, config);
    let request = test_request(6_300)?;

    let mut chunks_seen = 0_usize;
    let err = client
        .roundtrip_chunks(&request, 4, |_frame| {
            chunks_seen += 1;
            Ok(())
        })
        .await
        .expect_err("a stalled mid-stream response must fail, never complete");
    assert_eq!(chunks_seen, 1, "exactly the one delivered chunk reached the sink");
    let message = format!("{err:#}");
    assert!(
        message.contains("timed out reading ProtocolV2 response"),
        "expected a deadline error on the withheld chunk, got: {message}"
    );

    // Cancellation/no-replay: the next request must run on a NEW connection
    // and succeed in full.
    let follow_up = test_request(6_301)?;
    let response = client
        .roundtrip(&follow_up)
        .await
        .expect("follow-up request must succeed on a fresh connection");
    assert_eq!(response.header.request_id, follow_up.header.request_id);
    assert_eq!(
        accepts.load(Ordering::SeqCst),
        2,
        "stalled connection discarded; follow-up reconnected"
    );
    server.abort();
    Ok(())
}

#[tokio::test]
async fn close_before_final_chunk_errors_instead_of_completing() -> Result<()> {
    // Worker sends a non-final chunk and then dies (connection closed before
    // the final chunk). The receive must fail rather than report success on a
    // partially delivered wave.
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let accepts = Arc::new(AtomicUsize::new(0));
    let accepts_server = Arc::clone(&accepts);
    let server = tokio::spawn(async move {
        loop {
            let Ok((mut stream, _peer)) = listener.accept().await else {
                break;
            };
            let connection = accepts_server.fetch_add(1, Ordering::SeqCst);
            tokio::spawn(async move {
                let Ok(request) = read_request(&mut stream).await else {
                    return;
                };
                if connection == 0 {
                    let chunk = match echo_response(&request).and_then(|r| r.with_row_indices(vec![0], true)) {
                        Ok(chunk) => chunk,
                        Err(_) => return,
                    };
                    let Ok(frame) = chunk.encode() else {
                        return;
                    };
                    let _ = stream.write_all(&frame).await;
                    let _ = stream.flush().await;
                    drop(stream); // worker dies before the final chunk
                    return;
                }
                let Ok(frame) = echo_response(&request).and_then(|r| r.encode()) else {
                    return;
                };
                let _ = stream.write_all(&frame).await;
                let _ = stream.flush().await;
            });
        }
    });

    let mut client = TcpProtocolV2PersistentClient::new(addr, TcpTransportConfig::default());
    let request = test_request(6_310)?;
    let mut chunks_seen = 0_usize;
    let err = client
        .roundtrip_chunks(&request, 4, |_frame| {
            chunks_seen += 1;
            Ok(())
        })
        .await
        .expect_err("closing before the final chunk must fail the wave");
    assert_eq!(chunks_seen, 1, "only the non-final chunk was delivered");
    let message = format!("{err:#}");
    assert!(
        message.contains("ProtocolV2 response"),
        "expected a response-read failure, got: {message}"
    );

    let follow_up = test_request(6_311)?;
    let response = client
        .roundtrip(&follow_up)
        .await
        .expect("follow-up request must succeed on a reconnected socket");
    assert_eq!(response.header.request_id, follow_up.header.request_id);
    assert_eq!(accepts.load(Ordering::SeqCst), 2);
    server.abort();
    Ok(())
}

#[tokio::test]
async fn dropping_pending_dispatch_cancels_the_in_flight_request() -> Result<()> {
    // Cancellation propagation: `dispatch_chunks` hands out exclusive
    // ownership of the in-flight socket; dropping the guard cancels the
    // request without replay, and the next roundtrip reconnects cleanly.
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let accepts = Arc::new(AtomicUsize::new(0));
    let accepts_server = Arc::clone(&accepts);
    let observed_requests = Arc::new(AtomicUsize::new(0));
    let observed_requests_server = Arc::clone(&observed_requests);
    let server = tokio::spawn(async move {
        loop {
            let Ok((mut stream, _peer)) = listener.accept().await else {
                break;
            };
            accepts_server.fetch_add(1, Ordering::SeqCst);
            let observed_requests = Arc::clone(&observed_requests_server);
            tokio::spawn(async move {
                // Count requests; the first connection must observe exactly
                // one request before the client's cancel closes the socket.
                while let Ok(request) = read_request(&mut stream).await {
                    observed_requests.fetch_add(1, Ordering::SeqCst);
                    let Ok(frame) = echo_response(&request).and_then(|r| r.encode()) else {
                        break;
                    };
                    if stream.write_all(&frame).await.is_err() || stream.flush().await.is_err() {
                        break;
                    }
                }
            });
        }
    });

    let mut client = TcpProtocolV2PersistentClient::new(addr, TcpTransportConfig::default());
    let request = test_request(6_320)?;
    let pending = client
        .dispatch_chunks(&request, 2)
        .await
        .expect("request write succeeds");
    // The write completing does not guarantee the worker task has been
    // polled yet; wait until it has consumed the fully written request.
    tokio::time::timeout(Duration::from_secs(2), async {
        while observed_requests.load(Ordering::SeqCst) == 0 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("worker consumed the fully written request");
    drop(pending); // client cancels mid-flight, before any response chunk

    // Give the worker a beat to observe the socket close.
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(
        accepts.load(Ordering::SeqCst),
        1,
        "no replay on the cancelled connection"
    );

    let follow_up = test_request(6_321)?;
    let response = client
        .roundtrip(&follow_up)
        .await
        .expect("follow-up must run on a fresh connection after cancel");
    assert_eq!(response.header.request_id, follow_up.header.request_id);
    assert_eq!(
        accepts.load(Ordering::SeqCst),
        2,
        "cancelled connection discarded; follow-up reconnected"
    );
    server.abort();
    Ok(())
}
