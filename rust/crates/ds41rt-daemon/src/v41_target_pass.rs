//! One target pass from request-owned text/image rows through layer 39 and logits.
use crate::v41_backbone_execution::BackboneExecution;
use crate::v41_backbone_cache::CacheStage;
use crate::v41_block::{BlockOutput, EncoderSuffix};
use crate::v41_backbone_lane::BackboneLane;
use crate::v41_engram::{layer::EngramGate, EngramDeviceRows};
use crate::v41_experts::coordinator::NativeTp4Wave;
use crate::v41_index_lane::IndexLane;
use crate::v41_requests::{RequestBatch, Requests};
use crate::v41_target_embedding::TargetEmbeddingWave;
use crate::v41_target_head::{SampledTargetRows, TargetHeadWave, TargetLogits, TargetSamplingRowRequest};
use anyhow::{ensure, Context, Result};
use std::time::{Duration, Instant};
mod taps;
mod distributed;
pub(crate) use distributed::DistributedTargetPass;
mod verification;
pub(crate) use verification::VerificationTarget;
mod encoder_pair;
mod encoder_stream;
pub(crate) use taps::{TargetTapWave, TargetTaps};

/// Target cache publication used by the independent scheduler's queued dSpark
/// transaction. Static dispatch keeps ordinary serving on its existing path.
pub(crate) trait TargetCache<'a> {
    fn taps(&self, batch: &RequestBatch) -> Result<TargetTaps<'_>>;
    fn commit(&mut self, requests: &mut Requests<'a>, batch: &mut RequestBatch,
        accepted: &[u32]) -> Result<()>;
}
impl<'a> TargetCache<'a> for TargetPass<'_, 'a> {
    fn taps(&self, batch: &RequestBatch) -> Result<TargetTaps<'_>> { TargetPass::taps(self, batch) }
    fn commit(&mut self, requests: &mut Requests<'a>, batch: &mut RequestBatch,
        accepted: &[u32]) -> Result<()> { TargetPass::commit(self, requests, batch, accepted) }
}
impl<'a> TargetCache<'a> for DistributedTargetPass<'_, 'a> {
    fn taps(&self, batch: &RequestBatch) -> Result<TargetTaps<'_>> { DistributedTargetPass::taps(self, batch) }
    fn commit(&mut self, requests: &mut Requests<'a>, batch: &mut RequestBatch,
        accepted: &[u32]) -> Result<()> { DistributedTargetPass::commit(self, requests, batch, accepted) }
}

// A remote FFN wait owns its prepared lane state, never a request-bank borrow.
// Both ordinary serving and independent lane scheduling use the same execution
// body, so cache production and numerical operation order remain identical.
trait RequestAccess<'a> {
    fn cooperative_completion(&self) -> bool { false }
    fn with_requests<T>(&self, operation: impl FnOnce(&Requests<'a>) -> T) -> T;
}
impl<'a> RequestAccess<'a> for Requests<'a> {
    fn with_requests<T>(&self, operation: impl FnOnce(&Requests<'a>) -> T) -> T { operation(self) }
}
impl<'a> RequestAccess<'a> for std::cell::RefCell<&mut Requests<'a>> {
    fn cooperative_completion(&self) -> bool { true }
    fn with_requests<T>(&self, operation: impl FnOnce(&Requests<'a>) -> T) -> T {
        operation(&self.borrow())
    }
}

/// Which head terminal `execute_phase` should run for a decode/verification
/// pass. The compact greedy lane keeps its own terminal byte-for-byte; the
/// regular terminal returns the logits for the CPU path; the sampled terminal
/// runs the v4.1 GPU target-sampler after the head graph.
enum HeadTerminal<'m> {
    Regular,
    Greedy,
    Sampled { requests: &'m [TargetSamplingRowRequest], masks: Option<&'m [u32]>, mask_words: usize },
}

#[derive(Default, Debug, PartialEq, Eq)]
enum State {
    #[default]
    Idle,
    Running,
    Ready(u64),
    Encoded(u64),
}
impl State {
    fn begin(&mut self) -> Result<()> {
        ensure!(
            *self == Self::Idle,
            "target pass must be committed or discarded before reuse"
        );
        *self = Self::Running;
        Ok(())
    }
    fn ready(&self, batch: u64) -> Result<()> {
        ensure!(
            *self == Self::Ready(batch),
            "target pass has no completed logits for this batch"
        );
        Ok(())
    }
}
/// Cancelling the async future must cancel mapped I/O and invalidate the batch.
struct BatchGuard<'a> {
    batch: &'a mut RequestBatch,
    completed: bool,
}
impl Drop for BatchGuard<'_> {
    fn drop(&mut self) {
        if !self.completed {
            self.batch.cancel();
        }
    }
}

/// Each alternating wave owns a separate instance; immutable weights are shared.
/// The CUDA-owning executor must poll this owner on one thread. Transport remains
/// external so a scheduler can assign a separate connection set to each wave.
pub(crate) struct TargetPass<'w, 'a> {
    embedding: TargetEmbeddingWave<'w, 'a>,
    lane: BackboneLane<'w, 'a>,
    index: IndexLane<'w, 'a>,
    execution: BackboneExecution<'w, 'a>,
    upload: EngramDeviceRows<'a>,
    gates: [EngramGate<'w, 'a>; 2],
    head: TargetHeadWave<'w, 'a>,
    taps: TargetTapWave<'a>,
    engram_timeout: Duration,
    state: State,
    /// Per-row selections from the last [`HeadTerminal::Sampled`] pass. `None`
    /// for every other terminal.
    sampled: Option<SampledTargetRows>,
}
impl<'w, 'a> TargetPass<'w, 'a> {
    pub fn set_route_capture(&mut self, enabled: bool) {
        self.lane.set_route_capture(enabled);
        if enabled { self.index.enable_small_graph_shapes(); }
    }
    pub fn captured_routes(&self) -> &[Vec<[u32; 6]>] { self.lane.captured_routes() }
    pub fn reserve_sparse_decode_rows(&mut self, rows: usize) -> Result<()> {
        self.lane.reserve_sparse_decode_rows(rows)
    }
    pub fn new(
        embedding: TargetEmbeddingWave<'w, 'a>,
        lane: BackboneLane<'w, 'a>,
        index: IndexLane<'w, 'a>,
        execution: BackboneExecution<'w, 'a>,
        upload: EngramDeviceRows<'a>,
        gates: [EngramGate<'w, 'a>; 2],
        head: TargetHeadWave<'w, 'a>,
        taps: TargetTapWave<'a>,
        engram_timeout: Duration,
    ) -> Result<Self> {
        ensure!(
            gates[0].layer() == 1 && gates[1].layer() == 14,
            "target engram gates out of order"
        );
        ensure!(!engram_timeout.is_zero(), "engram timeout must be positive");
        Ok(Self {
            embedding,
            lane,
            index,
            execution,
            upload,
            gates,
            head,
            taps,
            engram_timeout,
            state: State::Idle,
            sampled: None,
        })
    }
    /// # Safety
    /// All components belong to the same device, capacity and official model.
    /// The caller exclusively owns CUDA buffers and polls on the owning thread.
    /// Selected rows are in this batch's flattened request order, at most 80.
    /// Produces private dSpark taps for every input row, including prepared image spans.
    pub async unsafe fn execute(
        &mut self,
        requests: &Requests<'a>,
        batch: &mut RequestBatch,
        transport: &mut NativeTp4Wave<'a>,
        placement: u64,
        selected: &[usize],
    ) -> Result<TargetLogits<'_>> {
        ensure!(batch.cache()?.stage() == CacheStage::Full, "ordinary target execute requires full phase");
        unsafe { self.execute_phase(requests, batch, transport, placement, selected, None, None, HeadTerminal::Regular).await?; }
        self.head.output()
    }
    /// Execute one full verification while allowing unrelated request commits
    /// during remote waits. The caller owns disjoint leases and drains queued
    /// lane work before releasing any request or device storage.
    pub async unsafe fn execute_shared(&mut self,
        requests: &std::cell::RefCell<&mut Requests<'a>>, batch: &mut RequestBatch,
        transport: &mut NativeTp4Wave<'a>, placement: u64, selected: &[usize],
    ) -> Result<TargetLogits<'_>> {
        ensure!(batch.cache()?.stage() == CacheStage::Full, "shared target execute requires full phase");
        unsafe { self.execute_phase(requests, batch, transport, placement, selected, None, None, HeadTerminal::Regular).await?; }
        self.head.output()
    }
    pub async unsafe fn execute_greedy(&mut self, requests: &Requests<'a>, batch: &mut RequestBatch,
        transport: &mut NativeTp4Wave<'a>, placement: u64, selected: &[usize]) -> Result<Vec<(u32, f32)>> {
        ensure!(batch.cache()?.stage() == CacheStage::Full, "greedy target execute requires full phase");
        unsafe { self.execute_phase(requests, batch, transport, placement, selected, None, None, HeadTerminal::Greedy).await?; }
        self.head.greedy_output()
    }
    pub async unsafe fn execute_shared_greedy(&mut self,
        requests: &std::cell::RefCell<&mut Requests<'a>>, batch: &mut RequestBatch,
        transport: &mut NativeTp4Wave<'a>, placement: u64, selected: &[usize]) -> Result<Vec<(u32, f32)>> {
        ensure!(batch.cache()?.stage() == CacheStage::Full, "shared greedy target execute requires full phase");
        unsafe { self.execute_phase(requests, batch, transport, placement, selected, None, None, HeadTerminal::Greedy).await?; }
        self.head.greedy_output()
    }
    /// Run one verification pass and select every row on the device with the
    /// v4.1 target-sampler. The caller names the per-row parameters and, for
    /// constrained rows, the packed mask arena; `sampled_rows` then exposes the
    /// device's choices.
    pub async unsafe fn execute_sampled(&mut self, requests: &Requests<'a>,
        batch: &mut RequestBatch, transport: &mut NativeTp4Wave<'a>, placement: u64,
        selected: &[usize], sampling: &[TargetSamplingRowRequest], masks: Option<&[u32]>,
        mask_words: usize) -> Result<()> {
        ensure!(batch.cache()?.stage() == CacheStage::Full, "sampled target execute requires full phase");
        let terminal = HeadTerminal::Sampled { requests: sampling, masks, mask_words };
        unsafe { self.execute_phase(requests, batch, transport, placement, selected, None, None, terminal).await?; }
        ensure!(self.sampled.is_some(), "sampled target pass published no rows");
        Ok(())
    }
    /// The cooperative (independent-lane) twin of [`Self::execute_sampled`].
    pub async unsafe fn execute_shared_sampled(&mut self,
        requests: &std::cell::RefCell<&mut Requests<'a>>, batch: &mut RequestBatch,
        transport: &mut NativeTp4Wave<'a>, placement: u64, selected: &[usize],
        sampling: &[TargetSamplingRowRequest], masks: Option<&[u32]>, mask_words: usize,
    ) -> Result<()> {
        ensure!(batch.cache()?.stage() == CacheStage::Full, "shared sampled target execute requires full phase");
        let terminal = HeadTerminal::Sampled { requests: sampling, masks, mask_words };
        unsafe { self.execute_phase(requests, batch, transport, placement, selected, None, None, terminal).await?; }
        ensure!(self.sampled.is_some(), "shared sampled target pass published no rows");
        Ok(())
    }
    /// Take the device-selected rows published by [`Self::execute_sampled`].
    pub fn sampled_rows(&mut self) -> Result<SampledTargetRows> {
        self.sampled.take().context("target pass has no sampled rows")
    }
    /// Download full logits for a subset of the sampled rows, in the same order.
    pub async fn download_sampled_rows(&mut self, rows: &SampledTargetRows,
        selection: &[usize]) -> Result<Vec<u8>> {
        self.head.download_sampled_rows(rows, selection).await
    }
    pub async unsafe fn execute_encoder(&mut self, requests: &Requests<'a>, batch: &mut RequestBatch,
        transport: &mut NativeTp4Wave<'a>, placement: u64, suffix: &mut EncoderSuffix<'a>) -> Result<()> {
        ensure!(batch.cache()?.stage() == CacheStage::Encoder, "encoder execute phase differs");
        unsafe { self.execute_phase(requests, batch, transport, placement, &[], Some(suffix), None, HeadTerminal::Regular).await }
    }
    pub async unsafe fn execute_encoder_replay(
        &mut self,
        requests: &Requests<'a>,
        batch: &mut RequestBatch,
        transport: &mut NativeTp4Wave<'a>,
        placement: u64,
        suffix: &mut EncoderSuffix<'a>,
    ) -> Result<()> {
        ensure!(
            batch.cache()?.stage() == CacheStage::EncoderReplay,
            "encoder replay phase differs"
        );
        unsafe {
            self.execute_phase(
                requests,
                batch,
                transport,
                placement,
                &[],
                Some(suffix),
                None,
                HeadTerminal::Regular,
            )
            .await
        }
    }
    pub async unsafe fn execute_replay(
        &mut self,
        requests: &Requests<'a>,
        batch: &mut RequestBatch,
        transport: &mut NativeTp4Wave<'a>,
        placement: u64,
        selected: &[usize],
        encoder: &BlockOutput<'_>,
    ) -> Result<TargetLogits<'_>> {
        ensure!(
            batch.cache()?.stage() == CacheStage::Replay,
            "decoder execute phase differs"
        );
        unsafe {
            self.execute_phase(
                requests,
                batch,
                transport,
                placement,
                selected,
                None,
                Some(encoder),
                HeadTerminal::Regular,
            )
            .await?;
        }
        self.head.output()
    }
    async unsafe fn execute_phase(&mut self, requests: &impl RequestAccess<'a>, batch: &mut RequestBatch,
        transport: &mut NativeTp4Wave<'a>, placement: u64, selected: &[usize],
        mut suffix: Option<&mut EncoderSuffix<'a>>, encoder: Option<&BlockOutput<'_>>,
        terminal: HeadTerminal<'_>) -> Result<()> {
        requests.with_requests(|requests| requests.validate(batch))?;
        let id = batch.cache()?.identity();
        let rows = batch.cache()?.positions().len();
        let stage = batch.cache()?.stage();
        let activation_trace = if tracing::enabled!(target: "ds41rt::activation_trace", tracing::Level::DEBUG) {
            let position: u64 = std::env::var("DS41RT_ACTIVATION_TRACE_POSITION")?.parse()?;
            if batch.cache()?.positions().first() == Some(&position) {
                let directory = std::path::PathBuf::from(std::env::var("DS41RT_ACTIVATION_TRACE_DIR")?)
                    .join(format!("batch{id}-{stage:?}-{rows}rows"));
                std::fs::create_dir_all(&directory)?;
                std::fs::write(directory.join("positions.json"), serde_json::to_vec(&batch.cache()?.positions())?)?;
                Some(directory)
            } else { None }
        } else { None };
        ensure!(
            (stage.is_encoder() || !selected.is_empty())
                && selected.len() <= 80
                && selected.iter().all(|&i| i < rows)
                && selected
                    .iter()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
                    == selected.len(),
            "invalid target head row selection"
        );
        self.state.begin()?;
        self.sampled = None;
        let mut guard = BatchGuard {
            batch,
            completed: false,
        };
        if !stage.is_encoder() {
            self.taps.begin(guard.batch.cache()?)?;
        }
        self.execution.restart_for(stage);
        if stage == CacheStage::Replay {
            let encoder = encoder.context("missing retained encoder suffix")?;
            ensure!(encoder.tokens == guard.batch.cache()?.positions(), "encoder suffix/replay row order differs");
            self.lane.restart_decoder(encoder)?;
            self.index.restart_decoder()?;
            unsafe { if requests.cooperative_completion() { self.lane.begin_prepared_cooperative().await?; }
                    else { self.lane.begin_prepared()?; } }
        } else {
            self.lane.restart()?;
            self.index.restart()?;
            let text = if requests.cooperative_completion() {
                requests.with_requests(|requests| requests.text_embedding_input(guard.batch))?
            } else { None };
            if let Some((tokens, positions)) = text {
                unsafe { self.lane.begin_tokens_cooperative(&mut self.embedding, tokens, &positions).await?; }
            } else {
                unsafe { requests.with_requests(|requests| requests.begin_input(guard.batch, &mut self.embedding, &mut self.lane))?; }
            }
        }
        for layer in stage.windows() {
            if layer != stage.windows().start {
                let prepare_timing = Instant::now();
                self.lane.advance()?;
                let advance_us = prepare_timing.elapsed().as_micros() as u64;
                if let Some(gate) = [1, 14].iter().position(|&l| l == layer) {
                    let start = Instant::now();
                    if requests.cooperative_completion() {
                        loop {
                            let gathered = requests.with_requests(|requests| requests.poll_engram_gather(guard.batch, &self.lane))?;
                            match gathered {
                                ds41rt_loader::EngramGatherPoll::Ready(lease) => {
                                    let rows = self.upload.upload_cooperative(&lease.view()?).await?;
                                    unsafe { self.lane.apply_engram_cooperative(&mut self.gates[gate], &rows).await?; }
                                    break;
                                }
                                ds41rt_loader::EngramGatherPoll::Cancelled => anyhow::bail!("engram gather cancelled"),
                                ds41rt_loader::EngramGatherPoll::Pending => {}
                            }
                            ensure!(start.elapsed() < self.engram_timeout, "target engram gather timed out at layer {layer}");
                            tokio::time::sleep(Duration::from_millis(1)).await;
                        }
                    } else { while !unsafe {
                        requests.with_requests(|requests| requests.poll_engram(
                            guard.batch,
                            &mut self.upload,
                            &mut self.gates[gate],
                            &mut self.lane,
                        ))?
                    } {
                        ensure!(
                            start.elapsed() < self.engram_timeout,
                            "target engram gather timed out at layer {layer}"
                        );
                        tokio::time::sleep(Duration::from_millis(1)).await;
                    } }
                }
                let engram_us = prepare_timing.elapsed().as_micros() as u64 - advance_us;
                if layer >= 37 {
                    unsafe {
                        if requests.cooperative_completion() {
                            self.taps.capture_cooperative(guard.batch.cache()?, &self.lane.prepared_input()?).await?;
                        } else { self.taps.capture(guard.batch.cache()?, &self.lane.prepared_input()?)?; }
                    }
                }
                let tapped_us = prepare_timing.elapsed().as_micros() as u64;
                unsafe {
                    if requests.cooperative_completion() { self.lane.begin_prepared_cooperative().await?; }
                    else { self.lane.begin_prepared()?; }
                }
                tracing::debug!(target: "ds41rt::timing", layer, rows, advance_us, engram_us, taps_us=tapped_us-advance_us-engram_us, begin_us=prepare_timing.elapsed().as_micros() as u64-tapped_us, "target layer preparation");
            }
            unsafe {
                let cooperative = requests.cooperative_completion();
                if cooperative {
                    let mut production = requests.with_requests(|requests| self.execution.enqueue_production_and_index(
                        requests.cache(), guard.batch.cache()?, &self.lane, &mut self.index))?;
                    while !requests.with_requests(|requests| production.poll(requests.cache(), guard.batch.cache()?))? {
                        tokio::task::yield_now().await;
                    }
                }
                let prepared = requests.with_requests(|requests| {
                    if cooperative {
                        self.execution.prepare_layer_cooperative(requests.cache(), guard.batch.cache()?,
                            &mut self.lane, &mut self.index)
                    } else {
                        self.execution.prepare_layer(requests.cache(), guard.batch.cache()?,
                            &mut self.lane, &mut self.index)
                    }
                })?;
                // No RefCell guard or bank reference survives into this await.
                let completed = prepared.execute(transport, placement, guard.batch.image_mask()).await?;
                if cooperative {
                    self.execution.complete_layer_cooperative(guard.batch.cache()?, &mut self.lane, completed).await?;
                } else { self.execution.complete_layer(guard.batch.cache()?, &mut self.lane, completed)?; }
            }
            if let Some(directory) = &activation_trace {
                self.lane.trace_output(directory)?;
            }
        }
        if stage.is_encoder() {
            suffix
                .as_mut()
                .context("missing encoder retention owner")?
                .capture(&self.lane.output()?)?;
            if stage == CacheStage::Encoder {
                self.lane.advance()?;
                unsafe {
                    if requests.cooperative_completion() { self.lane.begin_prepared_cooperative().await?; }
                    else { self.lane.begin_prepared()?; }
                    requests.with_requests(|requests| self.execution.produce_decoder_source(
                        requests.cache(),
                        guard.batch.cache()?,
                        &self.lane,
                    ))?;
                }
            }
            self.state = State::Encoded(id);
        } else {
            self.taps.output(guard.batch.cache()?)?;
            let output = self.lane.output()?;
            match terminal {
                HeadTerminal::Greedy => {
                    unsafe { self.head.execute_block_greedy(&output, selected, requests.cooperative_completion()).await?; }
                }
                HeadTerminal::Regular => {
                    if requests.cooperative_completion() {
                        unsafe { self.head.execute_block_cooperative(&output, selected).await?; }
                    } else {
                        unsafe { self.head.execute_block(&output, selected)?; }
                    }
                }
                HeadTerminal::Sampled { requests: rows, masks, mask_words } => {
                    let sampled = unsafe {
                        self.head
                            .execute_block_sampled(&output, selected, rows, masks, mask_words,
                                requests.cooperative_completion())
                            .await?
                    };
                    self.sampled = Some(sampled);
                }
            }
            self.state = State::Ready(id);
        }
        guard.completed = true;
        Ok(())
    }
    pub fn output(&self, batch: &RequestBatch) -> Result<TargetLogits<'_>> {
        self.state.ready(batch.cache()?.identity())?;
        self.head.output()
    }
    pub async fn download_logits(&mut self, batch: &RequestBatch, rows: &[usize]) -> Result<Vec<u8>> {
        self.state.ready(batch.cache()?.identity())?;
        self.head.download_rows(rows).await
    }
    pub fn taps(&self, batch: &RequestBatch) -> Result<TargetTaps<'_>> {
        self.state.ready(batch.cache()?.identity())?;
        self.taps.output(batch.cache()?)
    }
    /// The scheduler samples/validates logits before publishing accepted input
    /// prefixes. A failed commit consumes this pass; discard before reuse.
    pub fn enqueue_cache_commit(&mut self, requests: &Requests<'a>, batch: &RequestBatch,
        accepted: &[u32]) -> Result<()> {
        let id = batch.cache()?.identity();
        self.state.ready(id)?;
        requests.validate_acceptance(batch, accepted)?;
        unsafe { self.execution.enqueue_cache_commit(requests.cache(), batch.cache()?, accepted) }
    }
    pub fn poll_cache_commit(&self) -> Result<bool> { self.execution.poll_cache_commit() }
    pub fn abort_cache_commit(&mut self, requests: &mut Requests<'a>) -> Result<()> {
        requests.abort_cache_commit(&mut self.execution)
    }
    pub fn commit(
        &mut self,
        requests: &mut Requests<'a>,
        batch: &mut RequestBatch,
        accepted: &[u32],
    ) -> Result<()> {
        let id = batch.cache()?.identity();
        ensure!(self.state == State::Ready(id) || self.state == State::Encoded(id), "target commit phase incomplete");
        self.state = State::Running;
        self.taps.reset();
        requests.commit(batch, &mut self.execution, accepted)?;
        self.state = State::Idle;
        Ok(())
    }
    /// Publish the same accepted prefixes to dSpark, backbone and engram owners.
    /// Preflight errors retain the pass; failures after publication starts revoke
    /// every participating admission, including requests accepting zero rows.
    /// # Safety
    /// Proposal input belongs to this completed target batch. All windows and
    /// producer outputs are on this pass's CUDA device. Producer scratch is
    /// exclusive; reserved draft readers of disjoint window slots may remain active.
    pub unsafe fn commit_with_dspark(
        &mut self,
        requests: &mut Requests<'a>,
        batch: &mut RequestBatch,
        proposal: &mut crate::v41_experts::dspark::MainProposal<'_, '_, '_>,
        windows: &mut [&mut crate::v41_dspark_cache::DsparkWindow<'_>; 3],
        leases: [&[crate::v41_dspark_cache::WindowLease]; 3],
        accepted: &[u32],
    ) -> Result<()> {
        let id = batch.cache()?.identity();
        self.state.ready(id)?;
        requests.validate_acceptance(batch, accepted)?;
        proposal.validate_commit(id, windows, leases, accepted)?;
        self.state = State::Running;
        self.taps.reset();
        let result = (|| -> Result<()> {
            unsafe {
                proposal.commit(id, windows, leases, accepted)?;
            }
            requests.commit(batch, &mut self.execution, accepted)
        })();
        if let Err(error) = result {
            requests.revoke_batch(batch);
            for stage in 0..3 {
                for &lease in leases[stage] {
                    if windows[stage].request_id(lease).is_ok() {
                        if let Err(cleanup) = windows[stage].release(lease) {
                            tracing::error!(%cleanup, "releasing failed dSpark transaction");
                        }
                    }
                }
            }
            return Err(error);
        }
        self.state = State::Idle;
        Ok(())
    }
    /// Call only after execution future/borrowed outputs and transport consumers
    /// have been dropped. Request admission survives discarded private proposals.
    pub fn discard(&mut self, batch: &mut RequestBatch) -> Result<()> {
        self.sampled = None;
        batch.cancel();
        self.taps.reset();
        self.state = State::Running;
        self.lane.restart()?;
        self.index.restart()?;
        self.execution.restart();
        self.state = State::Idle;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::State;
    #[test]
    fn incomplete_foreign_and_consumed_passes_cannot_publish() {
        let mut state = State::default();
        assert!(state.ready(1).is_err());
        state.begin().unwrap();
        assert!(state.begin().is_err());
        assert!(state.ready(1).is_err());
        state = State::Ready(1);
        state.ready(1).unwrap();
        assert!(state.ready(2).is_err());
        assert!(state.begin().is_err());
        state = State::Idle;
        assert!(state.ready(1).is_err());
        state.begin().unwrap();
    }
}

#[cfg(test)]
mod distributed_tests;
