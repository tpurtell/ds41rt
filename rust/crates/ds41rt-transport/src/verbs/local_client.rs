//! Persistent coordinator QPs progressed by the inference owner. The existing
//! response assembler/recycling code runs locally; no QP worker is spawned.
use super::*;

pub(crate) struct LocalTp4Client {
    peers: Vec<SocketAddr>,
    config: TcpTransportConfig,
    sessions: Vec<Option<VerbsHostProtocolV2PersistentClientSession>>,
    pending: Vec<VecDeque<VerbsHostProtocolV2PendingChunkRoundtrip>>,
    chunks: Option<tokio::sync::mpsc::UnboundedReceiver<VerbsHostProtocolV2ResponseChunk>>,
    done: Vec<tokio::sync::oneshot::Receiver<Result<VerbsHostProtocolV2ResponseStreamStats>>>,
    deadline: Option<Instant>,
}
impl LocalTp4Client {
    pub(crate) fn new(peers: [SocketAddr; 4], config: TcpTransportConfig) -> Self {
        Self::with_peers(peers.to_vec(), config)
    }
    pub(crate) fn new_tp2(peers: [SocketAddr; 2], config: TcpTransportConfig) -> Self {
        Self::with_peers(peers.to_vec(), config)
    }
    fn with_peers(peers: Vec<SocketAddr>, config: TcpTransportConfig) -> Self {
        let world = peers.len();
        Self {
            peers,
            config,
            sessions: (0..world).map(|_| None).collect(),
            pending: (0..world).map(|_| VecDeque::with_capacity(1)).collect(),
            chunks: None,
            done: Vec::with_capacity(world),
            deadline: None,
        }
    }
    pub(crate) fn reset(&mut self) {
        // Drop queued payloads before sessions unregister their receive rings.
        self.chunks = None;
        self.done.clear();
        for pending in &mut self.pending {
            pending.clear();
        }
        for session in &mut self.sessions {
            *session = None;
        }
        self.deadline = None;
    }
    pub(crate) fn dispatch(&mut self, request: &ExpertProtocolV2Request) -> Result<()> {
        anyhow::ensure!(self.deadline.is_none(), "local TP4 request already pending");
        let result = self.post(request);
        if result.is_err() {
            self.reset();
        }
        result
    }
    fn post(&mut self, request: &ExpertProtocolV2Request) -> Result<()> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        self.chunks = Some(rx);
        for rank in 0..self.peers.len() {
            if self.sessions[rank]
                .as_ref()
                .map(|s| s.fits(request))
                .transpose()?
                == Some(false)
            {
                self.sessions[rank] = None;
            }
            if self.sessions[rank].is_none() {
                self.sessions[rank] = Some(VerbsHostProtocolV2PersistentClientSession::connect_local(
                    self.peers[rank],
                    &self.config,
                    request,
                )?);
            }
            let timing = self.sessions[rank]
                .as_mut()
                .unwrap()
                .post_chunk_request(request, &self.config)?;
            let (response_tx, response_rx) = tokio::sync::oneshot::channel();
            self.pending[rank].push_back(VerbsHostProtocolV2PendingChunkRoundtrip::new(
                VerbsHostProtocolV2QueuedChunkCommand {
                    request: request.clone(),
                    stream_id: rank,
                    chunk_tx: tx.clone(),
                    response_tx,
                },
                timing,
            ));
            self.done.push(response_rx);
        }
        // The same-thread channels adapt the existing assembler; they never
        // wake or hand off work to another thread. Each sink takes ownership of its chunk; retained pinned frames
        // return to the session pool only after the owner releases them.
        self.deadline = Some(
            Instant::now()
                .checked_add(self.config.timeout)
                .context("local TP4 deadline overflow")?,
        );
        Ok(())
    }
    pub(crate) fn poll<F>(&mut self, mut sink: F) -> Result<bool>
    where
        F: FnMut(VerbsHostProtocolV2ResponseChunk) -> Result<()>,
    {
        let deadline = self.deadline.context("local TP4 has no pending request")?;
        anyhow::ensure!(
            Instant::now() < deadline,
            "local TP4 response deadline expired"
        );
        for rank in 0..self.peers.len() {
            if !self.pending[rank].is_empty() {
                self.sessions[rank]
                    .as_mut()
                    .context("local TP4 session missing")?
                    .try_progress_chunk_requests(&mut self.pending[rank], &self.config)?;
            }
            let chunks = self
                .chunks
                .as_mut()
                .context("local TP4 response queue missing")?;
            while let Ok(chunk) = chunks.try_recv() {
                sink(chunk)?;
            }
        }
        if self.pending.iter().any(|p| !p.is_empty()) {
            return Ok(false);
        }
        // Final validation and completion publication happen synchronously in
        // accept_chunk_response_frame before it removes a pending request.
        for done in &mut self.done {
            done.try_recv().context("local TP4 completion missing")??;
        }
        self.done.clear();
        self.chunks = None;
        self.deadline = None;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tp2_reset_retains_only_two_actual_peer_slots() -> Result<()> {
        let peers = ["127.0.0.1:19441".parse()?, "127.0.0.1:19442".parse()?];
        let mut client = LocalTp4Client::new_tp2(peers, TcpTransportConfig { timing: false,
            timeout: std::time::Duration::from_secs(1), max_frame_bytes: 200_000,
        });
        let (_sender, receiver) = tokio::sync::mpsc::unbounded_channel();
        let (_completion, completed) = tokio::sync::oneshot::channel();
        client.chunks = Some(receiver);
        client.done.push(completed);
        client.deadline = Some(Instant::now());
        client.reset();
        assert_eq!(client.peers, peers);
        assert_eq!(client.sessions.len(), 2);
        assert_eq!(client.pending.len(), 2);
        assert!(client.sessions.iter().all(Option::is_none));
        assert!(client.pending.iter().all(VecDeque::is_empty));
        assert!(client.chunks.is_none() && client.done.is_empty() && client.deadline.is_none());
        Ok(())
    }
}
