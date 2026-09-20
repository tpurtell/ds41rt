//! TP2/TP3/TP4/TP6 dispatch through persistent RoCE QPs; TCP is used only for bootstrap.
#[cfg(test)]
use super::V41BackboneRequest;
use super::{V41SparkTopology, V41Tp4ChunkReceiver, V41_NATIVE_GROUP_REQUEST_FLAG};
use crate::verbs::LocalTp4Client;
use crate::{
    ExpertProtocolV2Request, ExpertProtocolV2RowDescriptor, ExpertV2SourceKind, TcpTransportConfig,
};
use anyhow::{ensure, Result};
use std::net::SocketAddr;

fn poll_quantum(rows: &[ExpertProtocolV2RowDescriptor]) -> std::time::Duration {
    let decode = !rows.is_empty()
        && rows.iter().all(|row| {
            matches!(
                row.source_kind,
                ExpertV2SourceKind::Decode | ExpertV2SourceKind::MtpVerify
            )
        });
    std::time::Duration::from_micros(if decode { 0 } else { 250 })
}

pub struct V41Tp4Roce {
    clients: LocalTp4Client,
    executors: Vec<u64>,
    capacity: u32,
    max_frame_bytes: usize,
    topology: Option<V41SparkTopology>,
}
impl V41Tp4Roce {
    pub fn new(
        peers: [SocketAddr; 4],
        executors: [u64; 4],
        capacity: u32,
        config: TcpTransportConfig,
    ) -> Result<Self> {
        Self::with_clients(&peers, &executors, capacity, &config,
            LocalTp4Client::new(peers, config.clone()), None)
    }
    /// Two actual Spark TP ranks; no placeholder peers or responses are used.
    pub fn new_tp2(
        peers: [SocketAddr; 2],
        executors: [u64; 2],
        capacity: u32,
        config: TcpTransportConfig,
    ) -> Result<Self> {
        Self::with_clients(&peers, &executors, capacity, &config,
            LocalTp4Client::new_tp2(peers, config.clone()), None)
    }
    /// Generic constructor for the validated physical rank counts 2, 3, 4 and 6
    /// under the legacy (non-ownership) frame contract.
    pub fn new_ranks(
        peers: &[SocketAddr],
        executors: &[u64],
        capacity: u32,
        config: TcpTransportConfig,
    ) -> Result<Self> {
        let clients = LocalTp4Client::new_ranks(peers.to_vec(), config.clone());
        Self::with_clients(peers, executors, capacity, &config, clients, None)
    }
    /// Topology-bound constructor: canonical executor identities are derived from
    /// `topology`, so a stale worker admitted for another `TP×EP` layout cannot
    /// satisfy this transport's response coverage. Requests must then carry the
    /// native group flag.
    pub fn new_topology(
        topology: V41SparkTopology,
        peers: &[SocketAddr],
        capacity: u32,
        config: TcpTransportConfig,
    ) -> Result<Self> {
        ensure!(
            peers.len() == topology.world_size(),
            "native group topology requires exactly its physical rank count"
        );
        let executors = topology.executor_ids();
        let clients = LocalTp4Client::new_ranks(peers.to_vec(), config.clone());
        Self::with_clients(peers, &executors, capacity, &config, clients, Some(topology))
    }
    fn with_clients(
        peers: &[SocketAddr],
        executors: &[u64],
        capacity: u32,
        config: &TcpTransportConfig,
        clients: LocalTp4Client,
        topology: Option<V41SparkTopology>,
    ) -> Result<Self> {
        ensure!(peers.len() == executors.len() && matches!(peers.len(), 2 | 3 | 4 | 6),
            "native TP/EP requires two, three, four or six matching peers and executors");
        if let Some(topology) = topology {
            ensure!(
                peers.len() == topology.world_size(),
                "native group topology requires exactly its physical rank count"
            );
        }
        ensure!(
            capacity > 0 && capacity <= 4096,
            "invalid native TP capacity"
        );
        ensure!(
            !config.timeout.is_zero(),
            "native RoCE timeout must be positive"
        );
        ensure!(
            config.max_frame_bytes >= 128 + 10240 + 40 + 6 * 12
                && config.max_frame_bytes <= 64 * 1024 * 1024,
            "invalid native RoCE frame budget"
        );
        for rank in 0..peers.len() {
            ensure!(
                executors[rank] != 0 && !executors[..rank].contains(&executors[rank]),
                "native TP executor identities must be distinct and nonzero"
            );
            ensure!(
                !peers[..rank].contains(&peers[rank]),
                "native TP endpoints must be distinct"
            );
        }
        Ok(Self {
            clients,
            executors: executors.to_vec(),
            capacity,
            max_frame_bytes: config.max_frame_bytes,
            topology,
        })
    }
    pub fn capacity(&self) -> u32 {
        self.capacity
    }
    pub fn world_size(&self) -> usize {
        self.executors.len()
    }
    /// Topology this transport was bound to, if any.
    pub fn topology(&self) -> Option<V41SparkTopology> {
        self.topology
    }
    /// Reset persistent QPs before a new admission. Pending dispatches borrow this
    /// owner exclusively, so an in-flight wave cannot be reset through this API.
    pub fn reset_connections(&mut self) {
        self.clients.reset();
    }

    /// Send the same canonical request to all ranks and accept every route row.
    /// The synchronous sink must consume/copy its frame slice before returning.
    /// It runs on the polling thread, allowing CUDA owners to stay thread-local.
    /// On error or cancellation the caller must discard partial destination state;
    /// the streaming QP path does not replay partially delivered requests.
    pub async fn execute<F>(&mut self, request: &ExpertProtocolV2Request, sink: F) -> Result<()>
    where
        F: FnMut(usize, u32, &[u8]) -> Result<()>,
    {
        self.dispatch(request).await?.receive(sink).await
    }

    /// Post every active rank's request directly from the inference owner. Remote work
    /// overlaps the shared FFN; dispatch completion is not send completion.
    ///
    /// Flag and topology must agree: a topology-bound transport accepts only
    /// native group requests, and a legacy transport rejects them. There is no
    /// silent canonical fallback for an EP1 topology-bound transport; legacy EP1
    /// uses [`Self::new`]/[`Self::new_tp2`] with canonical requests.
    pub async fn dispatch<'c, 'r>(
        &'c mut self,
        request: &'r ExpertProtocolV2Request,
    ) -> Result<V41Tp4RocePending<'c, 'r>> {
        let flagged = request.header.flags & V41_NATIVE_GROUP_REQUEST_FLAG != 0;
        let receiver = match self.topology {
            Some(topology) => {
                ensure!(
                    flagged,
                    "topology-bound native transport requires a native group request"
                );
                V41Tp4ChunkReceiver::from_owned_topology(
                    request,
                    self.capacity,
                    &self.executors,
                    topology,
                    self.max_frame_bytes,
                )?
            }
            None => {
                ensure!(
                    !flagged,
                    "native group request requires a topology-bound transport"
                );
                if self.executors.len() == 4 {
                    V41Tp4ChunkReceiver::from_owned(
                        request,
                        self.capacity,
                        self.executors.as_slice().try_into().expect("four executors"),
                        self.max_frame_bytes,
                    )?
                } else {
                    V41Tp4ChunkReceiver::from_owned_ranks(
                        request,
                        self.capacity,
                        &self.executors,
                        self.max_frame_bytes,
                    )?
                }
            }
        };
        self.clients.dispatch(request)?;
        Ok(V41Tp4RocePending {
            receiver,
            poll_quantum: poll_quantum(&request.rows),
            owner: self,
            _request: std::marker::PhantomData,
            complete: false,
        })
    }
}

/// Holds exclusive admission until every active rank plane has been consumed.
/// Cancellation resets the QPs before another wave can reuse them.
pub struct V41Tp4RocePending<'c, 'r> {
    receiver: V41Tp4ChunkReceiver,
    poll_quantum: std::time::Duration,
    owner: &'c mut V41Tp4Roce,
    _request: std::marker::PhantomData<&'r ExpertProtocolV2Request>,
    complete: bool,
}
impl V41Tp4RocePending<'_, '_> {
    pub async fn receive<F>(self, mut sink: F) -> Result<()>
    where
        F: FnMut(usize, u32, &[u8]) -> Result<()>,
    {
        self.receive_owned(|rank, start, payload| sink(rank, start, payload.as_ref())).await
    }
    /// Transfer validated payload ownership to the sink. Retain each payload
    /// until any asynchronous consumer completes, including on failure. A sink
    /// error abandons this whole wave and resets its QPs; it cannot be resumed.
    pub async fn receive_owned<F>(mut self, mut sink: F) -> Result<()>
    where
        F: FnMut(usize, u32, crate::VerbsHostProtocolV2ResponsePayload) -> Result<()>,
    {
        // Give the other execution lane its first opportunity as soon as this
        // wave must wait. A 250us initial spin can consume an entire small-row
        // FFN and serialize two otherwise independent decode stacks. Subsequent
        // decode polls yield after every unsuccessful poll so short GPU completions
        // on the peer lane are not delayed by host-side receive spinning.
        // Prefill/mixed waves retain 250us. Ready responses never yield.
        let mut quantum = std::time::Instant::now();
        let mut first_wait = true;
        loop {
            let receiver = &mut self.receiver;
            if self.owner.clients.poll(|chunk| {
                let mut location = None;
                receiver.push_rdma(&chunk, |rank, start, _bytes| {
                    ensure!(
                        rank == chunk.stream_id,
                        "native executor identity does not match its RoCE peer"
                    );
                    location = Some((rank, start));
                    Ok(())
                })?;
                let (rank, start) = location.expect("validated chunk has a location");
                sink(rank, start, chunk.partial_output_payload)
            })? {
                break;
            }
            if first_wait || quantum.elapsed() >= self.poll_quantum {
                first_wait = false;
                tokio::task::yield_now().await;
                quantum = std::time::Instant::now();
            } else {
                std::hint::spin_loop();
            }
        }
        ensure!(
            self.receiver.complete(),
            "native TP RoCE response coverage is incomplete"
        );
        self.complete = true;
        Ok(())
    }
}
impl Drop for V41Tp4RocePending<'_, '_> {
    fn drop(&mut self) {
        if !self.complete {
            self.owner.reset_connections();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{VerbsHostProtocolV2ResponseChunk, VerbsHostProtocolV2ResponsePayload};

    #[test]
    fn tp2_constructor_validates_real_peers_and_identities() -> Result<()> {
        let peers = ["127.0.0.1:19441".parse()?, "127.0.0.1:19442".parse()?];
        let config = TcpTransportConfig { timing: false,
            timeout: std::time::Duration::from_secs(1), max_frame_bytes: 200_000,
        };
        let mut client = V41Tp4Roce::new_tp2(peers, [11, 27], 2, config.clone())?;
        assert_eq!(client.world_size(), 2);
        client.reset_connections();
        assert_eq!(client.world_size(), 2);
        assert!(V41Tp4Roce::new_tp2([peers[0]; 2], [11, 27], 2, config.clone()).is_err());
        for ids in [[0, 27], [11, 11]] {
            assert!(V41Tp4Roce::new_tp2(peers, ids, 2, config.clone()).is_err());
        }
        Ok(())
    }

    #[test]
    fn tp2_chunks_require_both_exact_rank_planes() -> Result<()> {
        let owned = super::super::tests::request(2);
        let frame = owned.encode()?;
        let native = V41BackboneRequest::parse(&frame, 2)?;
        let mut receiver = V41Tp4ChunkReceiver::new_tp2(&native, [11, 27], 200_000)?;
        assert!(!receiver.complete());
        let payload = vec![0; super::super::V41_PARTIAL_ROW_BYTES as usize];
        let mut indices = [0];
        let mut make = |id, row| -> Result<Vec<u8>> {
            native.response_chunk(id, row, &payload, &mut indices, 200_000)?.to_owned()?.encode()
        };
        let wrong = make(3, 0)?;
        assert!(receiver.push(&wrong, |_, _, _| panic!("unknown executor reached sink")).is_err());
        for (id, rank) in [(27, 1), (11, 0)] {
            for row in 0..2 {
                let chunk = make(id, row)?;
                let mut stale = crate::ExpertProtocolV2Response::decode(&chunk)?;
                stale.header.request_id += 1;
                assert!(receiver.push(&stale.encode()?, |_, _, _| panic!("stale response reached sink")).is_err());
                // A failed sink must not commit row coverage.
                assert!(receiver.push(&chunk, |_, _, _| anyhow::bail!("injected sink failure")).is_err());
                receiver.push(&chunk, |actual, start, _| {
                    assert_eq!((actual, start), (rank, row)); Ok(())
                })?;
                assert!(receiver.push(&chunk, |_, _, _| panic!("duplicate reached sink")).is_err());
                assert_eq!(receiver.complete(), id == 11 && row == 1);
            }
        }
        assert_eq!(receiver.received_rows(), [2, 2, 0, 0]);
        let mut paired = owned;
        paired.header.flags |= super::super::V41_EXL3_PAIRED_REQUEST_FLAG;
        assert!(V41Tp4ChunkReceiver::from_owned_ranks(&paired, 2, &[11, 27], 200_000).is_err());
        Ok(())
    }

    #[test]
    fn prefill_or_benchmark_rows_keep_original_polling_in_mixed_waves() {
        let row = |kind| ExpertProtocolV2RowDescriptor {
            row_id: 0,
            source_kind: kind,
            source_request_id: 1,
            token_position: 0,
            route_offset: 0,
            route_count: 6,
        };
        use ExpertV2SourceKind::{Benchmark, Decode, MtpVerify, Prefill};
        assert_eq!(poll_quantum(&[row(Decode), row(MtpVerify)]).as_micros(), 0);
        for other in [Prefill, Benchmark] {
            assert_eq!(poll_quantum(&[row(other)]).as_micros(), 250);
            assert_eq!(
                poll_quantum(&[row(Decode), row(other), row(MtpVerify)]).as_micros(),
                250
            );
        }
        assert_eq!(poll_quantum(&[]).as_micros(), 250);
    }

    #[test]
    #[ignore = "requires four idle live V4.1 FP8 RoCE expert workers"]
    fn local_qps_replay_cancel_and_recover_live() -> Result<()> {
        replay_live(false)
    }

    #[test]
    #[ignore = "requires four idle ABI-3 V4.1 atomic-output RoCE expert workers"]
    fn local_qps_atomic_prefill_cancel_and_recover_live() -> Result<()> {
        replay_live(true)
    }

    // A separate numerical gate for atomic prefill outputs. Protocol identity,
    // row/rank coverage and small ordered outputs keep their exact checks.
    fn atomic_planes_close(previous: &[Vec<u8>], current: &[Vec<u8>]) -> Result<()> {
        ensure!(previous.len() == current.len(), "rank count changed");
        for (rank, (old, new)) in previous.iter().zip(current).enumerate() {
            ensure!(!old.is_empty() && old.len() == new.len() && old.len() % 2 == 0,
                "atomic plane extent changed");
            let (mut error, mut energy, mut changed) = (0.0_f64, 0.0_f64, 0_usize);
            for (a, b) in old.chunks_exact(2).zip(new.chunks_exact(2)) {
                let decode = |v: &[u8]| f32::from_bits((u16::from_le_bytes([v[0],v[1]]) as u32) << 16) as f64;
                let (a, b) = (decode(a), decode(b));
                ensure!(a.is_finite() && b.is_finite(), "nonfinite atomic replay");
                let delta = (a-b).abs();
                ensure!(delta <= 2e-5 + a.abs().max(b.abs())/128.0,
                    "atomic replay exceeds one BF16 rounding step plus absolute floor");
                error += delta*delta;
                energy += a*a;
                changed += usize::from(a != b);
            }
            ensure!(changed*1000 <= old.len()/2, "too many atomic replay values changed");
            ensure!(error <= energy.max(1e-30)*2.5e-7, "atomic replay relative L2 exceeds 5e-4");
            eprintln!("atomic rank={rank}: changed={changed}/{} rel_l2={}", old.len()/2,
                (error/energy.max(1e-30)).sqrt());
        }
        Ok(())
    }

    #[test]
    fn atomic_replay_gate_rejects_corruption_and_broad_changes() {
        let plane = |bits: u16| vec![bits.to_le_bytes();2000].concat();
        let baseline = vec![plane(0x3f80)];
        let mut rounded = baseline.clone();
        rounded[0][..2].copy_from_slice(&0x3f81_u16.to_le_bytes());
        assert!(atomic_planes_close(&baseline, &rounded).is_ok());
        rounded[0][..2].copy_from_slice(&0x4080_u16.to_le_bytes());
        assert!(atomic_planes_close(&baseline, &rounded).is_err());
        let mut broad = baseline.clone();
        broad[0][..6].copy_from_slice(&[0x81,0x3f,0x81,0x3f,0x81,0x3f]);
        assert!(atomic_planes_close(&baseline, &broad).is_err());
        assert!(atomic_planes_close(&baseline, &[plane(0x7fc0)]).is_err());
        assert!(atomic_planes_close(&baseline, &[vec![0;2]]).is_err());
    }

    fn replay_live(atomic: bool) -> Result<()> {
        let peers: [SocketAddr; 4] = std::env::var("DS41RT_LIVE_ROCE_PEERS")?
            .split(',')
            .map(str::parse)
            .collect::<std::result::Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| anyhow::anyhow!("four peers required"))?;
        let capacity: u32 = std::env::var("DS41RT_LIVE_ROCE_CAPACITY")
            .unwrap_or_else(|_| if atomic { "4096" } else { "80" }.into()).parse()?;
        ensure!([80, 256, 1024, 4096].contains(&capacity), "unsupported fixture capacity");
        ensure!(!atomic || capacity >= 256, "atomic fixture requires prefill capacity");
        let mut shapes = vec![1, 6, 16, 80];
        shapes.extend([256, 1024, 4096].into_iter().filter(|&rows| rows <= capacity));
        shapes.extend([6, 1]);
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        runtime.block_on(async {
            let mut client = V41Tp4Roce::new(
                peers,
                [1, 2, 3, 4],
                capacity,
                TcpTransportConfig { timing: false,
                    timeout: std::time::Duration::from_secs(10),
                    max_frame_bytes: 64 * 1024 * 1024,
                },
            )?;
            for rows in shapes {
                let base = super::super::tests::request(rows);
                let mut payload = Vec::new();
                for row in 0..rows {
                    payload.extend(
                        (0..5120).map(|i| {
                            0x30 + ((i + row) % 8) as u8 + if i % 2 == 0 { 0x80 } else { 0 }
                        }),
                    );
                    payload.extend([120; 160]);
                }
                let mut request = ExpertProtocolV2Request::new(
                    1000 + rows as u64,
                    base.header.placement_version,
                    39,
                    5120,
                    crate::ExpertV2Dtype::Fp8E4m3Ue8m0K32,
                    base.rows,
                    base.routes,
                    payload,
                )?;
                request.header.flags = base.header.flags;
                let mut expected: Option<Vec<Vec<u8>>> = None;
                for repetition in 0..3 {
                    request.header.request_id += 1;
                    let mut planes = vec![vec![0; rows as usize * 10240]; 4];
                    client
                        .execute(&request, |rank, start, bytes| {
                            let offset = start as usize * 10240;
                            planes[rank][offset..offset + bytes.len()].copy_from_slice(bytes);
                            Ok(())
                        })
                        .await?;
                    for plane in &planes {
                        ensure!(
                            plane
                                .chunks_exact(2)
                                .any(|b| u16::from_le_bytes([b[0], b[1]]) & 0x7fff != 0),
                            "zero expert plane"
                        );
                        ensure!(
                            plane
                                .chunks_exact(2)
                                .all(|b| u16::from_le_bytes([b[0], b[1]]) & 0x7f80 != 0x7f80),
                            "nonfinite expert plane"
                        );
                    }
                    if let Some(ref previous) = expected {
                        if atomic && rows > 80 {
                            atomic_planes_close(previous, &planes)?;
                        } else {
                            ensure!(previous == &planes, "replay changed rank planes");
                        }
                    }
                    expected = Some(planes);
                    if repetition == 0 {
                        request.header.request_id += 1;
                        // Abandon after posting all ranks, before receiving any.
                        drop(client.dispatch(&request).await?);
                    } else if repetition == 1 {
                        request.header.request_id += 1;
                        let failed = client
                            .execute(&request, |_, _, _| anyhow::bail!("injected sink failure"))
                            .await;
                        ensure!(failed.is_err(), "sink failure was swallowed");
                    }
                }
                eprintln!(
                    "local QPs rows={rows}: {} replay after abandoned dispatch and sink failure",
                    if atomic && rows > 80 { "bounded numerical" } else { "exact" }
                );
            }
            Ok(())
        })
    }

    #[test]
    fn rdma_chunks_preserve_rank_rows_and_reject_stale_or_reordered_data() -> Result<()> {
        let request = super::super::tests::request(2);
        let frame = request.encode()?;
        let native = V41BackboneRequest::parse(&frame, 2)?;
        let mut receiver = V41Tp4ChunkReceiver::new(&native, [1, 2, 3, 4], 200_000)?;
        for row in 0..2 {
            for rank in [3, 1, 0, 2] {
                let bytes = vec![(rank + row) as u8; super::super::V41_PARTIAL_ROW_BYTES as usize];
                let mut indices = [0];
                let response = native
                    .response_chunk(rank as u64 + 1, row, &bytes, &mut indices, 200_000)?
                    .to_owned()?;
                let wire_bytes = response.encode()?.len();
                let mut chunk = VerbsHostProtocolV2ResponseChunk {
                    stream_id: rank as usize,
                    header: response.header,
                    row_indices: Some(vec![row]),
                    partial_output_payload: VerbsHostProtocolV2ResponsePayload::from_owned(
                        bytes.clone(),
                    ),
                    wire_bytes,
                };
                chunk.header.request_id += 1;
                assert!(receiver
                    .push_rdma(&chunk, |_, _, _| panic!("stale data reached sink"))
                    .is_err());
                chunk.header.request_id -= 1;
                chunk.row_indices = Some(vec![1 - row]);
                assert!(receiver
                    .push_rdma(&chunk, |_, _, _| panic!("reordered data reached sink"))
                    .is_err());
                chunk.row_indices = Some(vec![row]);
                receiver.push_rdma(&chunk, |r, start, payload| {
                    assert_eq!((r, start), (rank as usize, row));
                    assert_eq!(payload, bytes);
                    Ok(())
                })?;
                assert!(receiver
                    .push_rdma(&chunk, |_, _, _| panic!("duplicate data reached sink"))
                    .is_err());
            }
        }
        assert!(receiver.complete());
        Ok(())
    }
}
