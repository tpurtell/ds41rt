//! Concurrent TP2/TP3/TP4/TP6 TCP dispatch with bounded per-rank response storage.
use super::{
    V41BackboneRequest, V41SparkTopology, V41Tp4ChunkReceiver, V41_NATIVE_GROUP_REQUEST_FLAG,
};
use crate::{
    ExpertProtocolV2Request, TcpProtocolV2PendingChunks, TcpProtocolV2PersistentClient,
    TcpTransportConfig,
};
use anyhow::{ensure, Result};
use std::{cell::RefCell, net::SocketAddr};

pub struct V41Tp4Tcp {
    clients: Vec<TcpProtocolV2PersistentClient>,
    executors: Vec<u64>,
    capacity: u32,
    max_frame_bytes: usize,
    topology: Option<V41SparkTopology>,
}
impl V41Tp4Tcp {
    pub fn new(
        peers: [SocketAddr; 4],
        executors: [u64; 4],
        capacity: u32,
        config: TcpTransportConfig,
    ) -> Result<Self> {
        Self::new_ranks(&peers, &executors, capacity, config)
    }
    /// Generic constructor for the validated physical rank counts 2, 3, 4 and 6
    /// under the legacy (non-ownership) frame contract.
    pub fn new_ranks(
        peers: &[SocketAddr],
        executors: &[u64],
        capacity: u32,
        config: TcpTransportConfig,
    ) -> Result<Self> {
        Self::with_clients(peers, executors, capacity, config, None)
    }
    /// Topology-bound constructor: canonical executor identities come from
    /// `topology`, and the transport then accepts native group requests.
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
        Self::with_clients(peers, &executors, capacity, config, Some(topology))
    }
    fn with_clients(
        peers: &[SocketAddr],
        executors: &[u64],
        capacity: u32,
        config: TcpTransportConfig,
        topology: Option<V41SparkTopology>,
    ) -> Result<Self> {
        ensure!(
            peers.len() == executors.len() && matches!(peers.len(), 2 | 3 | 4 | 6),
            "native TP/EP requires two, three, four or six matching peers and executors"
        );
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
            "native TCP timeout must be positive"
        );
        ensure!(
            config.max_frame_bytes >= 128 + 10240 + 40 + 6 * 12 && config.max_frame_bytes <= 64 * 1024 * 1024,
            "invalid native TCP frame budget"
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
            clients: peers
                .iter()
                .map(|peer| TcpProtocolV2PersistentClient::new(*peer, config.clone()))
                .collect(),
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
    /// Drop idle sockets before a new admission. Pending dispatches borrow this
    /// owner exclusively, so an in-flight wave cannot be reset through this API.
    pub fn reset_connections(&mut self) {
        for client in &mut self.clients {
            client.reset();
        }
    }

    /// Send the same canonical request to all ranks and accept every route row.
    /// The synchronous sink must consume/copy its frame slice before returning.
    /// It runs on the polling thread, allowing CUDA owners to stay thread-local.
    /// On error or cancellation the caller must discard partial destination state;
    /// no partially delivered request is automatically replayed.
    pub async fn execute<F>(&mut self, request: &ExpertProtocolV2Request, sink: F) -> Result<()>
    where
        F: FnMut(usize, u32, &[u8]) -> Result<()>,
    {
        self.dispatch(request).await?.receive(sink).await
    }

    /// Return after every complete request has been written. Expert workers can
    /// execute while the caller produces the coordinator's shared FFN result.
    ///
    /// Flag and topology must agree: a topology-bound transport accepts only
    /// native group requests, and a legacy transport rejects them. There is no
    /// silent canonical fallback for an EP1 topology-bound transport; legacy EP1
    /// uses [`Self::new`]/[`Self::new_ranks`] with canonical requests.
    pub async fn dispatch<'c, 'r>(
        &'c mut self,
        request: &'r ExpertProtocolV2Request,
    ) -> Result<V41Tp4Pending<'c, 'r>> {
        let frame = request.encode()?;
        ensure!(
            frame.len() <= self.max_frame_bytes,
            "native request exceeds TCP frame budget"
        );
        let flagged = request.header.flags & V41_NATIVE_GROUP_REQUEST_FLAG != 0;
        let native = match self.topology {
            Some(topology) => {
                ensure!(
                    flagged,
                    "topology-bound native transport requires a native group request"
                );
                V41BackboneRequest::parse_native_group(&frame, self.capacity, topology)?
            }
            None => {
                ensure!(
                    !flagged,
                    "native group request requires a topology-bound transport"
                );
                if request.header.flags & super::V41_EXL3_PAIRED_REQUEST_FLAG != 0 {
                    V41BackboneRequest::parse_paired(&frame, self.capacity)?
                } else {
                    V41BackboneRequest::parse(&frame, self.capacity)?
                }
            }
        };
        let rows = native.rows() as usize;
        let receiver = V41Tp4ChunkReceiver::new_ranks(&native, &self.executors, self.max_frame_bytes)?;
        let pending = futures::future::try_join_all(
            self.clients
                .iter_mut()
                .map(|client| client.dispatch_chunks(request, rows)),
        )
        .await?;
        Ok(V41Tp4Pending { pending, receiver })
    }
}

/// Owns the in-flight sockets and constant-size response coverage state.
/// Dropping it abandons unread responses and prevents their reuse by later waves.
pub struct V41Tp4Pending<'c, 'r> {
    pending: Vec<TcpProtocolV2PendingChunks<'c, 'r>>,
    receiver: V41Tp4ChunkReceiver,
}
impl V41Tp4Pending<'_, '_> {
    pub async fn receive<F>(mut self, mut sink: F) -> Result<()>
    where
        F: FnMut(usize, u32, &[u8]) -> Result<()>,
    {
        let state = RefCell::new((&mut self.receiver, &mut sink));
        let responses = self
            .pending
            .into_iter()
            .enumerate()
            .map(|(expected_rank, pending)| {
                let state = &state;
                pending.receive(move |frame| {
                    let mut state = state.borrow_mut();
                    let (receiver, sink) = &mut *state;
                    receiver.push(frame, |rank, start, bytes| {
                        ensure!(
                            rank == expected_rank,
                            "native executor identity does not match its peer"
                        );
                        sink(rank, start, bytes)
                    })?;
                    Ok(())
                })
            });
        futures::future::try_join_all(responses).await?;
        ensure!(
            self.receiver.complete(),
            "native TP/EP response coverage is incomplete"
        );
        Ok(())
    }
}
