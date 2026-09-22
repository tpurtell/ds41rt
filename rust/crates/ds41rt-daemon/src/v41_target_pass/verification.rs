//! Static dispatch for independent verification lanes in either GPU layout.
use super::{CacheStage, DistributedTargetPass, NativeTp4Wave, RequestBatch, Requests,
    Result, TargetCache, TargetPass};
use crate::v41_target_head::{SampledTargetRows, TargetSamplingRowRequest};
use crate::v41_memory::device::DeviceOwner;
use std::cell::RefCell;

/// Futures remain concrete: selecting a layout does not box lane work or add a
/// common stream. Each implementation retains its own device/transport owners.
pub(crate) trait VerificationTarget<'a>: TargetCache<'a> {
    type Transport;
    fn set_route_capture(&mut self, enabled: bool) -> Result<()>;
    fn captured_routes(&self) -> &[Vec<[u32; 6]>];
    async unsafe fn execute_shared(&mut self, requests: &RefCell<&mut Requests<'a>>,
        batch: &mut RequestBatch, transport: &mut Self::Transport, placement: u64,
        selected: &[usize]) -> Result<()>;
    async unsafe fn execute_shared_greedy(&mut self, requests: &RefCell<&mut Requests<'a>>,
        batch: &mut RequestBatch, transport: &mut Self::Transport, placement: u64,
        selected: &[usize]) -> Result<Vec<(u32, f32)>>;
    /// Whether this layout can select rows on the device at all.
    ///
    /// A layout without the sampled terminal must never be routed into it: the
    /// default `execute_shared_sampled` bails, and bailing would fail a whole
    /// lane round that the CPU path handles today. Callers gate on this and
    /// keep the CPU fallback. Chunk 1 implements the terminal only for
    /// [`TargetPass`]; `DistributedTargetPass` is a documented limitation to be
    /// implemented later.
    ///
    /// This is an associated const rather than a method so the two layouts'
    /// capability is a property of the type, which lets a test pin it without
    /// constructing a GPU-owning pass.
    const SUPPORTS_SAMPLED_TERMINAL: bool = false;
    /// Run one verification pass and select every row on the device with the
    /// v4.1 target-sampler (K1). Only used when every row of the round is
    /// greedy (including constrained greedy), which is what chunk 1 supports.
    async unsafe fn execute_shared_sampled(&mut self, requests: &RefCell<&mut Requests<'a>>,
        batch: &mut RequestBatch, transport: &mut Self::Transport, placement: u64,
        selected: &[usize], sampling: &[TargetSamplingRowRequest], masks: Option<&[u32]>,
        mask_words: usize, ordered_rows: bool) -> Result<()> {
        let _ = (requests, batch, transport, placement, selected, sampling, masks, mask_words,
            ordered_rows);
        anyhow::bail!("this target layout has no device-selected sampling terminal")
    }
    /// Take the device-selected rows published by
    /// [`Self::execute_shared_sampled`].
    fn sampled_rows(&mut self) -> Result<SampledTargetRows> {
        anyhow::bail!("this target layout publishes no sampled rows")
    }
    /// Download full logits for a subset of the sampled rows, in the same order.
    async fn download_sampled_rows(&mut self, rows: &SampledTargetRows,
        selection: &[usize]) -> Result<Vec<u8>> {
        let _ = (rows, selection);
        anyhow::bail!("this target layout cannot download sampled rows")
    }
    async fn download_logits(&mut self, batch: &RequestBatch, rows: &[usize]) -> Result<Vec<u8>>;
    fn enqueue_cache_commit(&mut self, requests: &Requests<'a>, batch: &RequestBatch,
        accepted: &[u32]) -> Result<()>;
    fn poll_cache_commit(&self) -> Result<bool>;
    fn abort_cache_commit(&mut self, requests: &mut Requests<'a>) -> Result<()>;
    fn discard(&mut self, batch: &mut RequestBatch) -> Result<()>;
}

impl<'a> VerificationTarget<'a> for TargetPass<'_, 'a> {
    type Transport = NativeTp4Wave<'a>;
    const SUPPORTS_SAMPLED_TERMINAL: bool = true;
    fn set_route_capture(&mut self, enabled: bool) -> Result<()> {
        TargetPass::set_route_capture(self, enabled); Ok(())
    }
    fn captured_routes(&self) -> &[Vec<[u32; 6]>] { TargetPass::captured_routes(self) }
    async unsafe fn execute_shared(&mut self, requests: &RefCell<&mut Requests<'a>>,
        batch: &mut RequestBatch, transport: &mut Self::Transport, placement: u64,
        selected: &[usize]) -> Result<()> {
        unsafe { TargetPass::execute_shared(self, requests, batch, transport, placement, selected).await?; }
        Ok(())
    }
    async unsafe fn execute_shared_greedy(&mut self, requests: &RefCell<&mut Requests<'a>>,
        batch: &mut RequestBatch, transport: &mut Self::Transport, placement: u64,
        selected: &[usize]) -> Result<Vec<(u32, f32)>> {
        unsafe { TargetPass::execute_shared_greedy(self, requests, batch, transport, placement, selected).await }
    }
    async unsafe fn execute_shared_sampled(&mut self, requests: &RefCell<&mut Requests<'a>>,
        batch: &mut RequestBatch, transport: &mut Self::Transport, placement: u64,
        selected: &[usize], sampling: &[TargetSamplingRowRequest], masks: Option<&[u32]>,
        mask_words: usize, ordered_rows: bool) -> Result<()> {
        unsafe {
            TargetPass::execute_shared_sampled(self, requests, batch, transport, placement,
                selected, sampling, masks, mask_words, ordered_rows).await
        }
    }
    fn sampled_rows(&mut self) -> Result<SampledTargetRows> { TargetPass::sampled_rows(self) }
    async fn download_sampled_rows(&mut self, rows: &SampledTargetRows,
        selection: &[usize]) -> Result<Vec<u8>> {
        TargetPass::download_sampled_rows(self, rows, selection).await
    }
    async fn download_logits(&mut self, batch: &RequestBatch, rows: &[usize]) -> Result<Vec<u8>> {
        TargetPass::download_logits(self, batch, rows).await
    }
    fn enqueue_cache_commit(&mut self, requests: &Requests<'a>, batch: &RequestBatch,
        accepted: &[u32]) -> Result<()> { TargetPass::enqueue_cache_commit(self, requests, batch, accepted) }
    fn poll_cache_commit(&self) -> Result<bool> { TargetPass::poll_cache_commit(self) }
    fn abort_cache_commit(&mut self, requests: &mut Requests<'a>) -> Result<()> {
        TargetPass::abort_cache_commit(self, requests)
    }
    fn discard(&mut self, batch: &mut RequestBatch) -> Result<()> { TargetPass::discard(self, batch) }
}

impl<'a> VerificationTarget<'a> for DistributedTargetPass<'_, 'a> {
    type Transport = DeviceOwner<'a, NativeTp4Wave<'a>>;
    fn set_route_capture(&mut self, enabled: bool) -> Result<()> {
        DistributedTargetPass::set_route_capture(self, enabled)
    }
    fn captured_routes(&self) -> &[Vec<[u32; 6]>] { DistributedTargetPass::captured_routes(self) }
    async unsafe fn execute_shared(&mut self, requests: &RefCell<&mut Requests<'a>>,
        batch: &mut RequestBatch, transport: &mut Self::Transport, placement: u64,
        selected: &[usize]) -> Result<()> {
        anyhow::ensure!(batch.cache()?.stage() == CacheStage::Full, "verification requires full phase");
        unsafe { self.execute(requests, batch, transport, placement, selected, None, None, false).await }
    }
    async unsafe fn execute_shared_greedy(&mut self, requests: &RefCell<&mut Requests<'a>>,
        batch: &mut RequestBatch, transport: &mut Self::Transport, placement: u64,
        selected: &[usize]) -> Result<Vec<(u32, f32)>> {
        anyhow::ensure!(batch.cache()?.stage() == CacheStage::Full, "verification requires full phase");
        unsafe { self.execute(requests, batch, transport, placement, selected, None, None, true).await?; }
        self.greedy_output(batch)
    }
    async fn download_logits(&mut self, batch: &RequestBatch, rows: &[usize]) -> Result<Vec<u8>> {
        DistributedTargetPass::download_logits(self, batch, rows).await
    }
    fn enqueue_cache_commit(&mut self, requests: &Requests<'a>, batch: &RequestBatch,
        accepted: &[u32]) -> Result<()> { DistributedTargetPass::enqueue_cache_commit(self, requests, batch, accepted) }
    fn poll_cache_commit(&self) -> Result<bool> { DistributedTargetPass::poll_cache_commit(self) }
    fn abort_cache_commit(&mut self, requests: &mut Requests<'a>) -> Result<()> {
        DistributedTargetPass::abort_cache_commit(self, requests)
    }
    fn discard(&mut self, batch: &mut RequestBatch) -> Result<()> { DistributedTargetPass::discard(self, batch) }
}
