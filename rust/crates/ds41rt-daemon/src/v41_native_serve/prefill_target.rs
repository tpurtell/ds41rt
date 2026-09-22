//! Static prefill operations for the shared admission/continuation workflow.
use super::*;
use crate::v41_backbone_cache::{CacheLease, CacheStage};
use crate::v41_block::EncoderSuffix;
use crate::v41_memory::device::DeviceOwner;
use crate::v41_requests::RequestBatch;
use crate::v41_target_pass::{DistributedTargetPass, VerificationTarget};
use speculative::DraftChain;
use std::cell::RefCell;

#[cfg(test)]
pub(crate) fn exercise_prefill<'a, P: PrefillTarget<'a>, C: DraftChain<'a>>(
    lib: &'a NativeLibrary, runtime: &tokio::runtime::Runtime, pass: &mut P, other: &mut P,
    requests: &mut Requests<'a>, transports: [&mut P::Transport; 2], lease: CacheLease,
    tokens: &[u32], chunk_rows: usize, draft: Option<&mut DraftRuntime<'_, 'a, C>>,
) -> Result<u32> {
    let (events, _receive) = mpsc::channel(4);
    let job = NativeRequest { prompt: String::new(), constraint: None, images: Vec::new(), max_tokens: 16, sampling: Default::default(), events };
    let [first_transport, second_transport] = transports;
    super::prefill(lib, runtime, pass, other, requests, first_transport, second_transport,
        lease, tokens, chunk_rows, &job, draft, &mut || Ok(()))?.select(None)
}

pub(crate) trait PrefillTarget<'a>: VerificationTarget<'a> + Sized {
    type Suffix;
    fn new_suffix(&self, lib: &'a NativeLibrary, end: u64) -> Result<Self::Suffix>;
    /// # Safety
    /// Batch and suffix belong to this request and layout; all owners remain
    /// live through completion/cancellation, with no conflicting access.
    async unsafe fn encoder_part(&mut self, requests: &mut Requests<'a>, batch: &mut RequestBatch,
        transport: &mut Self::Transport, suffix: &mut Self::Suffix) -> Result<()>;
    /// # Safety
    /// The passes have independent workspaces and the same layout/model. Input
    /// chunks and suffix follow the encoder stream's request/cache contract.
    async unsafe fn encoder_stream(&mut self, other: &mut Self, requests: &mut Requests<'a>, lease: CacheLease,
        chunks: &[&[u32]], transports: [&mut Self::Transport; 2], suffix: &mut Self::Suffix,
        keep_running: &dyn Fn() -> bool) -> Result<()>;
    /// # Safety
    /// Same contract as `encoder_stream`, plus: `before_chunk` runs synchronously
    /// before each dispatched chunk (host-cache store pacing, packet HC-9).
    async unsafe fn execute_encoder_stream_held(&mut self, other: &mut Self, requests: &mut Requests<'a>,
        lease: CacheLease, chunks: &[&[u32]], transports: [&mut Self::Transport; 2],
        suffix: &mut Self::Suffix, keep_running: &dyn Fn() -> bool,
        before_chunk: &dyn Fn() -> Result<()>) -> Result<()> {
        // Default for passes without a held dispatch (dual-RTX lane): the hold is
        // load-bearing on the single-GPU 5090 path only; log once so degradation
        // is observable rather than silent (rc2 lesson).
        static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            tracing::warn!("held encoder stream unavailable on this pass; HC-9 per-chunk hold inactive");
        }
        self.encoder_stream(other, requests, lease, chunks, transports, suffix, keep_running).await
    }
    /// # Safety
    /// Batch inputs and optional completed encoder suffix match this target.
    async unsafe fn prefill_logits(&mut self, lib: &'a NativeLibrary, requests: &mut Requests<'a>,
        batch: &mut RequestBatch, transport: &mut Self::Transport, selected: &[usize],
        suffix: Option<&Self::Suffix>) -> Result<Vec<u8>>;
    async fn commit_prefill<C: DraftChain<'a>>(&mut self, requests: &mut Requests<'a>, batch: &mut RequestBatch,
        draft: Option<&mut DraftRuntime<'_, 'a, C>>, accepted: u32) -> Result<()>;
}
impl<'a> PrefillTarget<'a> for TargetPass<'_, 'a> {
    type Suffix = EncoderSuffix<'a>;
    fn new_suffix(&self, lib: &'a NativeLibrary, end: u64) -> Result<Self::Suffix> {
        EncoderSuffix::new(lib, end, EncoderSuffix::device_bytes(end)?)
    }
    async unsafe fn encoder_part(&mut self, requests: &mut Requests<'a>, batch: &mut RequestBatch,
        transport: &mut Self::Transport, suffix: &mut Self::Suffix) -> Result<()> {
        if batch.cache()?.stage() == CacheStage::EncoderReplay {
            unsafe { self.execute_encoder_replay(requests, batch, transport, 0, suffix).await }
        } else { unsafe { self.execute_encoder(requests, batch, transport, 0, suffix).await } }
    }
    async unsafe fn encoder_stream(&mut self, other: &mut Self, requests: &mut Requests<'a>, lease: CacheLease,
        chunks: &[&[u32]], transports: [&mut Self::Transport; 2], suffix: &mut Self::Suffix,
        keep_running: &dyn Fn() -> bool) -> Result<()> {
        unsafe { self.execute_encoder_stream(other, requests, lease, chunks, transports, suffix, keep_running).await }
    }
    async unsafe fn execute_encoder_stream_held(&mut self, other: &mut Self, requests: &mut Requests<'a>,
        lease: CacheLease, chunks: &[&[u32]], transports: [&mut Self::Transport; 2],
        suffix: &mut Self::Suffix, keep_running: &dyn Fn() -> bool,
        before_chunk: &dyn Fn() -> Result<()>) -> Result<()> {
        unsafe { self.execute_encoder_stream_held(other, requests, lease, chunks, transports, suffix,
            keep_running, before_chunk).await }
    }
    async unsafe fn prefill_logits(&mut self, lib: &'a NativeLibrary, requests: &mut Requests<'a>,
        batch: &mut RequestBatch, transport: &mut Self::Transport, selected: &[usize],
        suffix: Option<&Self::Suffix>) -> Result<Vec<u8>> {
        let logits = if let Some(suffix) = suffix {
            let encoder = suffix.output()?;
            unsafe { self.execute_replay(requests, batch, transport, 0, selected, &encoder).await? }
        } else { unsafe { self.execute(requests, batch, transport, 0, selected).await? } };
        let mut bytes = vec![0; logits.logits.bytes];
        lib.copy_d2h(&mut bytes, logits.logits)?;
        Ok(bytes)
    }
    async fn commit_prefill<C: DraftChain<'a>>(&mut self, requests: &mut Requests<'a>, batch: &mut RequestBatch,
        draft: Option<&mut DraftRuntime<'_, 'a, C>>, accepted: u32) -> Result<()> {
        if let Some(draft) = draft { draft.commit(self, requests, batch, accepted) }
        else { self.commit(requests, batch, &[accepted]) }
    }
}
impl<'a> PrefillTarget<'a> for DistributedTargetPass<'_, 'a> {
    type Suffix = DeviceOwner<'a, EncoderSuffix<'a>>;
    fn new_suffix(&self, lib: &'a NativeLibrary, end: u64) -> Result<Self::Suffix> {
        self.encoder_device()?.own(|| EncoderSuffix::new(lib, end, EncoderSuffix::device_bytes(end)?))
    }
    async unsafe fn encoder_part(&mut self, requests: &mut Requests<'a>, batch: &mut RequestBatch,
        transport: &mut Self::Transport, suffix: &mut Self::Suffix) -> Result<()> {
        unsafe { self.execute(&RefCell::new(requests), batch, transport, 0, &[], Some(suffix), None, false).await }
    }
    async unsafe fn encoder_stream(&mut self, other: &mut Self, requests: &mut Requests<'a>, lease: CacheLease,
        chunks: &[&[u32]], transports: [&mut Self::Transport; 2], suffix: &mut Self::Suffix,
        keep_running: &dyn Fn() -> bool) -> Result<()> {
        unsafe { self.execute_encoder_stream(other, requests, lease, chunks, transports, suffix, keep_running).await }
    }
    async unsafe fn prefill_logits(&mut self, _lib: &'a NativeLibrary, requests: &mut Requests<'a>,
        batch: &mut RequestBatch, transport: &mut Self::Transport, selected: &[usize],
        suffix: Option<&Self::Suffix>) -> Result<Vec<u8>> {
        let encoder = suffix.map(|suffix| suffix.output()).transpose()?;
        unsafe { self.execute(&RefCell::new(requests), batch, transport, 0, selected, None, encoder.as_ref(), false).await?; }
        self.download_logits(batch, &(0..selected.len()).collect::<Vec<_>>()).await
    }
    async fn commit_prefill<C: DraftChain<'a>>(&mut self, requests: &mut Requests<'a>, batch: &mut RequestBatch,
        draft: Option<&mut DraftRuntime<'_, 'a, C>>, accepted: u32) -> Result<()> {
        struct Commit<'s, 'w, 'a, P: VerificationTarget<'a>, C: DraftChain<'a>> {
            pass: &'s mut P, requests: &'s mut Requests<'a>, batch: &'s mut RequestBatch,
            draft: Option<&'s mut DraftRuntime<'w, 'a, C>>, armed: bool,
        }
        impl<'a, P: VerificationTarget<'a>, C: DraftChain<'a>> Drop for Commit<'_, '_, 'a, P, C> {
            fn drop(&mut self) {
                if self.armed {
                    if let Err(error) = self.pass.abort_cache_commit(self.requests) { tracing::error!(%error, "aborting prefill target commit"); }
                    if let Some(draft) = &mut self.draft {
                        if let Err(error) = draft.abort_queued_commit(0, self.requests, self.batch) { tracing::error!(%error, "aborting prefill draft commit"); }
                    }
                }
            }
        }
        let mut commit = Commit { pass: self, requests, batch, draft, armed: true };
        if let Some(draft) = &mut commit.draft {
            draft.begin_queued_commit(0, commit.pass, commit.requests, commit.batch, &[accepted])?;
        }
        commit.pass.enqueue_cache_commit(commit.requests, commit.batch, &[accepted])?;
        loop {
            let target_ready = commit.pass.poll_cache_commit()?;
            let draft_ready = commit.draft.as_ref().map(|draft| draft.poll_queued_commit(0)).transpose()?.unwrap_or(true);
            if target_ready && draft_ready { break; }
            tokio::task::yield_now().await;
        }
        if let Some(draft) = &mut commit.draft {
            draft.finish_queued_commit(0, commit.pass, commit.requests, commit.batch, &[accepted])?;
        } else { commit.pass.commit(commit.requests, commit.batch, &[accepted])?; }
        commit.armed = false;
        Ok(())
    }
}

#[cfg(test)]
pub(crate) use super::scheduler::exercise_distributed_decode;
