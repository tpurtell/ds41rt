//! Two encoder lanes advance across chunk boundaries without a pair barrier.
use super::*;
use crate::v41_backbone_cache::CacheLease;
use crate::v41_requests::RequestTokens;
use ds41rt_transport::ExpertV2SourceKind;
use std::cell::RefCell;
use tokio::sync::Notify;

struct ChunkGuard<'r, 'q, 'w, 'a> {
    pass: &'r mut TargetPass<'w, 'a>,
    requests: &'r RefCell<&'q mut Requests<'a>>,
    batch: RequestBatch,
    complete: bool,
}
impl Drop for ChunkGuard<'_, '_, '_, '_> {
    fn drop(&mut self) {
        if !self.complete {
            self.requests.borrow_mut().revoke_batch(&mut self.batch);
            if let Err(error) = self.pass.discard(&mut self.batch) {
                tracing::error!(%error, "discarding cancelled encoder stream lane");
            }
        }
    }
}

impl<'w, 'a> TargetPass<'w, 'a> {
    /// At most two chunks own Engram staging and mutable execution storage.
    /// Each lane reserves its next chunk after committing its previous one.
    /// # Safety
    /// Both lanes and transport waves own independent storage on this CUDA device.
    pub async unsafe fn execute_encoder_stream(&mut self, other: &mut Self,
        requests: &mut Requests<'a>, lease: CacheLease, chunks: &[&[u32]],
        transports: [&mut NativeTp4Wave<'a>; 2], suffix: &mut EncoderSuffix<'a>,
        keep_running: &dyn Fn() -> bool,
    ) -> Result<()> {
        unsafe {
            self.execute_encoder_stream_held(other, requests, lease, chunks, transports, suffix,
                keep_running, &|| Ok(())).await
        }
    }

    /// `execute_encoder_stream` with a hook the engine runs at the top of every chunk
    /// iteration, after the `keep_running` check and before the chunk is reserved or any
    /// transport work begins. The host-cache prefill hold (packet HC-9) paces the stream
    /// against pending store copies here. Hooks must not error: the engine's hook logs and
    /// ignores its own errors, so in production the stream is never aborted by the hook
    /// (the `?` below is a defensive last resort for hooks that do return an error).
    /// # Safety
    /// Both lanes and transport waves own independent storage on this CUDA device.
    pub async unsafe fn execute_encoder_stream_held(&mut self, other: &mut Self,
        requests: &mut Requests<'a>, lease: CacheLease, chunks: &[&[u32]],
        transports: [&mut NativeTp4Wave<'a>; 2], suffix: &mut EncoderSuffix<'a>,
        keep_running: &dyn Fn() -> bool, before_chunk: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        ensure!(!chunks.is_empty(), "empty encoder stream");
        let requests = RefCell::new(requests);
        let suffix = RefCell::new(suffix);
        let published: Vec<[Notify; 20]> = (0..chunks.len())
            .map(|_| std::array::from_fn(|_| Notify::new())).collect();
        let reserved: Vec<Notify> = (0..chunks.len()).map(|_| Notify::new()).collect();
        let committed: Vec<Notify> = (0..chunks.len()).map(|_| Notify::new()).collect();
        let [first, second] = transports;
        tokio::try_join!(
            biased;
            unsafe { self.encoder_stream_lane(0, &requests, lease, chunks, first, &suffix,
                &published, &reserved, &committed, keep_running, before_chunk) },
            unsafe { other.encoder_stream_lane(1, &requests, lease, chunks, second, &suffix,
                &published, &reserved, &committed, keep_running, before_chunk) },
        )?;
        Ok(())
    }

    async unsafe fn encoder_stream_lane(&mut self, parity: usize,
        requests: &RefCell<&mut Requests<'a>>, lease: CacheLease, chunks: &[&[u32]],
        transport: &mut NativeTp4Wave<'a>, suffix: &RefCell<&mut EncoderSuffix<'a>>,
        published: &[[Notify; 20]], reserved: &[Notify], committed: &[Notify],
        keep_running: &dyn Fn() -> bool, before_chunk: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        for index in (parity..chunks.len()).step_by(2) {
            if index != 0 { reserved[index - 1].notified().await; }
            ensure!(keep_running(), "client disconnected");
            before_chunk()?;
            let chunk = chunks[index];
            let batch = requests.borrow_mut().reserve_encoder(&[RequestTokens {
                lease, tokens: chunk, image_mask: None, kind: ExpertV2SourceKind::Prefill,
            }])?;
            let mut guard = ChunkGuard { pass: self, requests, batch, complete: false };
            reserved[index].notify_one();
            let started = Instant::now();
            let pass = &mut *guard.pass;
            let batch = &mut guard.batch;
            pass.state.begin()?;
            pass.execution.restart_for(CacheStage::Encoder);
            pass.lane.restart()?; pass.index.restart()?; pass.taps.reset();
            unsafe { requests.borrow_mut().begin_input(batch, &mut pass.embedding, &mut pass.lane)?; }
            unsafe { pass.execute_encoder_chunk(requests, batch, transport,
                index.checked_sub(1).map(|i| &published[i]),
                (index + 1 < chunks.len()).then_some(&published[index])).await?; }
            // A following chunk may finish first. Retain its output until its
            // predecessor's suffix, source-20 boundary and histories are committed.
            if index != 0 { committed[index - 1].notified().await; }
            ensure!(keep_running(), "client disconnected");
            suffix.borrow_mut().capture(&pass.lane.output()?)?;
            pass.lane.advance()?;
            unsafe {
                pass.lane.begin_prepared()?;
                requests.borrow_mut().publish_encoder_boundary(batch, &mut pass.execution, &pass.lane)?;
            }
            pass.state = State::Encoded(batch.cache()?.identity());
            pass.commit(&mut requests.borrow_mut(), batch, &[chunk.len() as u32])?;
            guard.complete = true;
            committed[index].notify_one();
            crate::v41_native_serve::console::totals::prefill(chunk.len());
            crate::v41_native_serve::console::Prefill::done(crate::v41_native_serve::console::PrefillKind::Chunk,
                parity, index, chunks.len(), chunk.len(), started);
            tracing::debug!(target: "ds41rt::timing", index, rows=chunk.len(),
                total_us=started.elapsed().as_micros() as u64, "target encoder stream chunk");
        }
        Ok(())
    }
}
