//! Concurrent TP4 TCP dispatch with bounded per-rank response storage.
use super::{V41BackboneRequest, V41Tp4ChunkReceiver};
use crate::{
    ExpertProtocolV2Request, TcpProtocolV2PendingChunks, TcpProtocolV2PersistentClient,
    TcpTransportConfig,
};
use anyhow::{ensure, Result};
use std::{cell::RefCell, net::SocketAddr};

pub struct V41Tp4Tcp {
    clients: [TcpProtocolV2PersistentClient; 4],
    executors: [u64; 4],
    capacity: u32,
    max_frame_bytes: usize,
}
impl V41Tp4Tcp {
    pub fn new(
        peers: [SocketAddr; 4],
        executors: [u64; 4],
        capacity: u32,
        config: TcpTransportConfig,
    ) -> Result<Self> {
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
        for rank in 0..4 {
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
            clients: peers.map(|peer| TcpProtocolV2PersistentClient::new(peer, config.clone())),
            executors,
            capacity,
            max_frame_bytes: config.max_frame_bytes,
        })
    }
    pub fn capacity(&self) -> u32 {
        self.capacity
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

    /// Return after all four complete requests have been written. Expert workers
    /// can execute while the caller produces the coordinator's shared FFN result.
    pub async fn dispatch<'c, 'r>(
        &'c mut self,
        request: &'r ExpertProtocolV2Request,
    ) -> Result<V41Tp4Pending<'c, 'r>> {
        let frame = request.encode()?;
        ensure!(
            frame.len() <= self.max_frame_bytes,
            "native request exceeds TCP frame budget"
        );
        let native = if request.header.flags & super::V41_EXL3_PAIRED_REQUEST_FLAG != 0 {
            V41BackboneRequest::parse_paired(&frame, self.capacity)?
        } else {
            V41BackboneRequest::parse(&frame, self.capacity)?
        };
        let receiver = V41Tp4ChunkReceiver::new(&native, self.executors, self.max_frame_bytes)?;
        let pending = futures::future::try_join_all(
            self.clients
                .iter_mut()
                .map(|client| client.dispatch_chunks(request, native.rows() as usize)),
        )
        .await?;
        Ok(V41Tp4Pending { pending, receiver })
    }
}

/// Owns four in-flight sockets and constant-size response coverage state.
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
            "native TP4 response coverage is incomplete"
        );
        Ok(())
    }
}
