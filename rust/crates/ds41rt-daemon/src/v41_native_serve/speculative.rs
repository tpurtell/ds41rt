use super::*;
use crate::v41_dspark_cache::{DsparkWindow, WindowLease};
use crate::v41_experts::dspark::{DsparkChain, DsparkMainContext, DsparkWeights};
use crate::v41_requests::RequestBatch;
mod chain;
mod policy;
pub(crate) use policy::snapshot as policy_snapshot;
pub(crate) use chain::DraftChain;
mod distributed;

/// Execution workspaces share a request-indexed bank of persistent draft state.
pub(crate) struct DraftRuntime<'w, 'a, C = DsparkChain<'w, 'a>> {
    mains: Vec<DsparkMainContext<'w, 'a>>,
    pending_commit_ids: Vec<Vec<u64>>,
    pending_prefix_ids: [Option<u64>; 2],
    chains: Vec<C>,
    windows: [DsparkWindow<'a>; 3],
    requests: std::collections::BTreeMap<u64, DraftRequest>,
    pending: Vec<Option<(Vec<(u64, u32, u64, usize)>, Instant)>>,
    request_limit: usize,
    draft_limit: usize,
    draft_width: usize,
    // Downloaded whenever the length policy is bound, or for diagnostics.
    confidence_trace: std::collections::BTreeMap<u64, Vec<f32>>,
    /// Verify every available draft instead of selecting lengths.
    fixed: bool,
    /// Online bandwidth-balance length policy, bound to the installed placement.
    policy: Option<ds41rt_core::DsparkPolicy>,
    /// The policy's predicted round time per lane, awaiting its observation.
    predicted: [Option<f64>; 2],
    published: Option<Instant>,
}
struct DraftRequest {
    leases: [WindowLease; 3],
    rng: ds41rt_core::DsparkRng,
    slot: usize,
}
pub(crate) struct DraftPrefix<'a> {
    windows: crate::v41_memory::device::DeviceOwner<'a, Vec<crate::v41_dspark_cache::DsparkPrefix<'a>>>,
}
impl<'w, 'a> DraftRuntime<'w, 'a> {
    pub fn new(
        lib: &'a NativeLibrary,
        weights: &'w DsparkWeights<'a>,
        table: &'w NativeRtxTensors<'a>,
        head: &'w VocabularyHead<'a>,
        capacity: u32,
    ) -> Result<Self> {
        Self::with_requests(lib, weights, table, head, capacity, 16)
    }
    pub fn with_requests(
        lib: &'a NativeLibrary, weights: &'w DsparkWeights<'a>,
        table: &'w NativeRtxTensors<'a>, head: &'w VocabularyHead<'a>,
        capacity: u32, requests: u32,
    ) -> Result<Self> {
        ensure!((1..=16).contains(&requests), "invalid draft request limit");
        let window = || DsparkWindow::new(lib, requests as usize, capacity,
            DsparkWindow::device_bytes(requests as usize, capacity)?);
        let lane_count = if requests == 1 { 1 } else { 2 };
        let lane_requests = requests.div_ceil(lane_count as u32);
        let shared_bytes = weights.draft_bytes(requests)?;
        let lane_bytes = weights.draft_bytes(lane_requests)?;
        let mut mains = vec![weights.main_context(capacity, DsparkMainContext::device_bytes(lib, capacity)?)?];
        if lane_count > 1 { mains.push(weights.main_context(80, DsparkMainContext::device_bytes(lib, 80)?)?); }
        let chains = (0..lane_count).map(|_| weights.draft(table, head, lane_requests, lane_bytes))
            .collect::<Result<Vec<_>>>()?;
        tracing::info!(lanes=lane_count, lane_requests, draft_width=weights.draft_width(), shared_workspace_bytes=shared_bytes,
            lane_workspace_bytes=lane_bytes, total_workspace_bytes=lane_bytes*lane_count,
            additional_workspace_bytes=(lane_bytes*lane_count).saturating_sub(shared_bytes),
            "lane-local dSpark draft workspaces (weights shared)");
        Ok(Self {
            mains,
            pending_commit_ids: vec![Vec::new(); lane_count],
            pending_prefix_ids: [None, None],
            chains,
            windows: [window()?, window()?, window()?],
            requests: Default::default(),
            pending: vec![None; lane_count],
            request_limit: requests as usize,
            draft_limit: 5,
            draft_width: weights.draft_width(),
            confidence_trace: Default::default(),
            fixed: false,
            policy: None,
            predicted: [None; 2],
            published: None,
        })
    }
}
impl<'w, 'a, C: DraftChain<'a>> DraftRuntime<'w, 'a, C> {
    pub fn admit(&mut self, id: u64) -> Result<()> {
        let _device = self.chains[0].execution_device().map(|device| device.enter()).transpose()?;
        ensure!(!self.requests.contains_key(&id), "draft request already admitted");
        let slot = (0..self.request_limit).find(|slot| self.requests.values().all(|request| request.slot != *slot))
            .context("draft request capacity exhausted")?;
        let mut leases = Vec::new();
        for stage in 0..3 {
            match self.windows[stage].begin_request(slot, id) {
                Ok(lease) => leases.push(lease),
                Err(error) => {
                    for (stage, lease) in leases.into_iter().enumerate() {
                        let _ = self.windows[stage].release(lease);
                    }
                    return Err(error);
                }
            }
        }
        self.requests.insert(id, DraftRequest {
            leases: leases.try_into().ok().context("draft admission incomplete")?,
            rng: ds41rt_core::DsparkRng::new(id),
            slot,
        });
        Ok(())
    }
    pub fn max_verify_rows(&self) -> usize { self.draft_limit + 1 }
    pub fn set_draft_limit(&mut self, limit: u8) -> Result<()> {
        ensure!((1..=self.draft_width as u8).contains(&limit), "draft limit exceeds generated width");
        self.draft_limit = limit as usize;
        Ok(())
    }
    /// Verify every available draft; the policy still fits and reports.
    pub fn set_fixed(&mut self, fixed: bool) {
        self.fixed = fixed;
        if let Some(policy) = &mut self.policy {
            *policy = ds41rt_core::DsparkPolicy::new(policy.placement().clone(), fixed);
        }
    }
    /// Bind the length policy to the installed expert placement before serving.
    pub fn configure_policy(&mut self, transport: &NativeTp4Wave<'_>, nvfp4: bool) -> Result<()> {
        let placement = policy::placement(transport, nvfp4)?;
        let local = (0..ds41rt_core::DSPARK_LAYERS)
            .filter(|&l| placement.class(l) == ds41rt_core::DsparkLayerClass::Local).count();
        tracing::info!(fixed=self.fixed, draft_limit=self.draft_limit, local_layers=local,
            remote_expert_bytes=placement.expert_bytes(ds41rt_core::DSPARK_LAYERS - 1),
            "dSpark bandwidth-balance length policy bound");
        self.bind_policy(placement);
        Ok(())
    }
    pub(crate) fn bind_policy(&mut self, placement: ds41rt_core::DsparkPlacement) {
        let policy = ds41rt_core::DsparkPolicy::new(placement, self.fixed);
        policy::publish(&policy, self.draft_limit);
        self.policy = Some(policy);
    }
    /// Routes and layer timings feed the policy on every verification round.
    pub fn capture_routes(&self) -> bool { self.policy.is_some() }
    fn wants_confidence(&self) -> bool {
        self.policy.is_some() || tracing::enabled!(target: "ds41rt::draft_policy", tracing::Level::DEBUG)
    }
    /// Choose each request's draft length for one lane round. `requests` are
    /// (identity, available drafts) in lane order; `shared` means the other lane
    /// has active requests. `None` verifies every available draft (fixed mode
    /// or a fit still warming up).
    pub fn select_lengths(&mut self, lane: usize, requests: &[(u64, usize)], shared: bool)
        -> Result<Option<Vec<usize>>> {
        ensure!(lane < self.predicted.len(), "invalid draft policy lane");
        self.predicted[lane] = None;
        let Some(policy) = &mut self.policy else { return Ok(None) };
        let probabilities = requests.iter().map(|&(id, maximum)| {
            if maximum == 0 { return Ok(Vec::new()); }
            let logits = self.confidence_trace.get(&id).context("missing draft confidence")?;
            ensure!(maximum <= logits.len(), "draft confidence prefix exceeds output");
            Ok(logits[..maximum].iter().map(|&x| policy::sigmoid(x)).collect::<Vec<_>>())
        }).collect::<Result<Vec<_>>>()?;
        let candidates: Vec<_> = requests.iter().zip(&probabilities)
            .map(|(&(id, _), confidence)| ds41rt_core::DsparkCandidate { id, confidence }).collect();
        let started = Instant::now();
        let selection = policy.select(shared, &candidates).map_err(anyhow::Error::msg)?;
        Ok(match selection {
            Some(selection) => {
                tracing::debug!(target: "ds41rt::draft_policy", lane, shared,
                    available=?requests.iter().map(|r| r.1).collect::<Vec<_>>(), selected=?selection.lengths,
                    expected_tokens=selection.expected_tokens, predicted_us=selection.predicted_us,
                    evaluated=selection.evaluated, selection_us=started.elapsed().as_micros() as u64,
                    "native length selection");
                self.predicted[lane] = Some(selection.predicted_us);
                Some(selection.lengths)
            }
            None => {
                let ids: Vec<_> = requests.iter().map(|r| r.0).collect();
                let full: Vec<_> = requests.iter().map(|r| r.1).collect();
                self.predicted[lane] = policy.predict(shared, &ids, &full);
                None
            }
        })
    }
    /// Feed one completed lane round to the policy. `requests` are (identity,
    /// verifier rows, accepted inputs) in lane order; `total_us` spans draft
    /// start to commit. Observation problems are logged, never fatal.
    pub fn observe_round(&mut self, lane: usize, shared: bool, routes: &[Vec<[u32; 6]>],
        layer_done: &[Option<Instant>], requests: &[(u64, usize, u32)], total_us: u64) {
        let predicted = self.predicted.get_mut(lane).and_then(Option::take);
        let Some(policy) = &mut self.policy else { return };
        let confidence: Vec<Option<Vec<f64>>> = requests.iter().map(|&(id, rows, _)|
            self.confidence_trace.get(&id).map(|logits| logits[..rows.saturating_sub(1).min(logits.len())]
                .iter().map(|&x| policy::sigmoid(x)).collect())).collect();
        let observed: Vec<_> = requests.iter().zip(&confidence).map(|(&(id, rows, accepted), confidence)|
            ds41rt_core::DsparkObservedRequest { id, rows, accepted: accepted as usize,
                confidence: confidence.as_deref() }).collect();
        let layer_us = policy::layer_us(layer_done);
        if let Err(error) = policy.observe(ds41rt_core::DsparkRoundObservation { shared, requests: &observed,
            routes, layer_us: &layer_us, total_us: total_us as f64, predicted_us: predicted }) {
            tracing::warn!(error, lane, "dSpark policy round observation skipped");
        }
        tracing::debug!(target: "ds41rt::draft_policy", lane, shared, predicted_us=predicted, total_us,
            rows=requests.iter().map(|r| r.1).sum::<usize>(), "native length policy observation");
        if self.published.is_none_or(|at| at.elapsed() >= std::time::Duration::from_millis(250)) {
            policy::publish(policy, self.draft_limit);
            self.published = Some(Instant::now());
        }
    }
    pub fn release(&mut self, id: u64) -> Result<()> {
        let _device = self.chains[0].execution_device().map(|device| device.enter()).transpose()?;
        ensure!(!self.pending_prefix_ids.contains(&Some(id)), "cannot release a request with pending snapshot copies");
        ensure!(!self.pending.iter().flatten().any(|(seeds, _)| seeds.iter().any(|seed| seed.0 == id)),
            "cannot release a request with a pending draft");
        ensure!(!self.pending_commit_ids.iter().any(|ids| ids.contains(&id)),
            "cannot release a request with a pending commit");
        if let Some(policy) = &mut self.policy { policy.release(id); }
        self.confidence_trace.remove(&id);
        let mut failure = None;
        if let Some(request) = self.requests.remove(&id) {
            for (window, lease) in self.windows.iter_mut().zip(request.leases) {
                if window.request_id(lease).is_ok() {
                    if let Err(error) = window.release(lease) {
                        failure = Some(error);
                    }
                }
            }
        }
        failure.map_or(Ok(()), Err)
    }
    pub fn reserve_prefixes(&mut self, slots: usize) -> Result<usize> {
        let _device = self.chains[0].execution_device().map(|device| device.enter()).transpose()?;
        let mut bytes = 0;
        for window in &mut self.windows { bytes += window.reserve_prefixes(slots)?; }
        Ok(bytes)
    }
    pub fn retain_prefix(&mut self, id: u64, end: u64) -> Result<DraftPrefix<'a>> {
        let _device = self.chains[0].execution_device().map(|device| device.enter()).transpose()?;
        let request = self.requests.get(&id).context("draft request not admitted")?;
        for (window, lease) in self.windows.iter().zip(request.leases) {
            ensure!(window.committed_end(lease)? == Some(end), "draft and target prefix frontiers differ");
        }
        let device = self.windows[0].device();
        let windows = device.own(|| self.windows.iter_mut().zip(request.leases)
            .map(|(window, lease)| window.retain_prefix(lease)).collect::<Result<Vec<_>>>())?;
        Ok(DraftPrefix { windows })
    }
    pub fn queue_prefix(&mut self, lane: usize, id: u64, end: u64) -> Result<()> {
        let _device = self.chains[0].execution_device().map(|device| device.enter()).transpose()?;
        ensure!(self.pending_prefix_ids.get(lane).context("invalid draft snapshot lane")?.is_none(),
            "draft snapshot lane is occupied");
        ensure!(!self.pending_prefix_ids.contains(&Some(id)), "draft snapshot already pending");
        let request = self.requests.get(&id).context("draft request not admitted")?;
        for (window, lease) in self.windows.iter().zip(request.leases) {
            ensure!(window.committed_end(lease)? == Some(end), "draft and target prefix frontiers differ");
        }
        let leases = request.leases;
        self.pending_prefix_ids[lane] = Some(id);
        let queued = self.windows.iter_mut().zip(leases)
            .try_for_each(|(window, lease)| window.queue_prefix(lane, lease));
        if let Err(error) = queued {
            if let Err(cleanup) = self.abort_prefix(lane) {
                tracing::error!(%cleanup, "draining failed draft snapshot enqueue");
            }
            return Err(error);
        }
        Ok(())
    }
    pub fn prefix_ready(&self, lane: usize, id: u64) -> Result<bool> {
        let _device = self.chains[0].execution_device().map(|device| device.enter()).transpose()?;
        ensure!(self.pending_prefix_ids.get(lane) == Some(&Some(id)), "draft snapshot owner differs");
        let request = self.requests.get(&id).context("draft snapshot request missing")?;
        for (window, lease) in self.windows.iter().zip(request.leases) {
            if !window.prefix_ready(lane, lease)? { return Ok(false); }
        }
        Ok(true)
    }
    pub fn finish_prefix(&mut self, lane: usize, id: u64) -> Result<DraftPrefix<'a>> {
        let _device = self.chains[0].execution_device().map(|device| device.enter()).transpose()?;
        ensure!(self.prefix_ready(lane, id)?, "draft snapshot copies are incomplete");
        let leases = self.requests[&id].leases;
        let device = self.windows[0].device();
        let windows = device.own(|| self.windows.iter_mut().zip(leases)
            .map(|(window, lease)| window.finish_prefix(lane, lease)).collect::<Result<Vec<_>>>())?;
        self.pending_prefix_ids[lane] = None;
        Ok(DraftPrefix { windows })
    }
    pub fn abort_prefix(&mut self, lane: usize) -> Result<()> {
        let _device = self.chains[0].execution_device().map(|device| device.enter()).transpose()?;
        ensure!(lane < self.pending_prefix_ids.len(), "invalid draft snapshot lane");
        let mut failure = None;
        for window in &mut self.windows {
            if let Err(error) = window.abort_prefix(lane) { failure.get_or_insert(error); }
        }
        self.pending_prefix_ids[lane] = None;
        failure.map_or(Ok(()), Err)
    }
    pub fn restore_prefix(&mut self, id: u64, end: u64, prefix: &DraftPrefix<'a>) -> Result<()> {
        let _device = self.chains[0].execution_device().map(|device| device.enter()).transpose()?;
        let device = self.windows[0].device();
        ensure!(prefix.windows.device.id == device.id
            && std::ptr::eq(prefix.windows.device.library, device.library), "draft prefix device differs from runtime");
        let request = self.requests.get(&id).context("draft request not admitted")?;
        ensure!(prefix.windows.len() == 3 && prefix.windows.iter().all(|p| p.end() == end),
            "retained draft and target frontiers differ");
        let leases = request.leases;
        for (stage, saved) in prefix.windows.iter().enumerate() {
            if let Err(error) = self.windows[stage].restore_prefix(leases[stage], saved) {
                if let Err(cleanup) = self.release(id) {
                    tracing::error!(%cleanup, "releasing failed draft prefix restore");
                }
                return Err(error);
            }
        }
        Ok(())
    }
    #[cfg(test)]
    pub(crate) fn validate_position(&self, id: u64, end: u64) -> Result<()> {
        let request = self.requests.get(&id).context("draft request not admitted")?;
        for (window, lease) in self.windows.iter().zip(request.leases) {
            ensure!(window.request_id(lease)? == id, "draft request identity differs");
            ensure!(window.committed_end(lease)? == Some(end), "draft position differs");
        }
        Ok(())
    }
    pub fn commit(
        &mut self,
        pass: &mut TargetPass<'_, 'a>,
        requests: &mut Requests<'a>,
        batch: &mut RequestBatch,
        accepted: u32,
    ) -> Result<()> {
        let _device = self.chains[0].execution_device().map(|device| device.enter()).transpose()?;
        self.commit_batch(pass, requests, batch, &[accepted])
    }
    pub fn commit_batch(
        &mut self,
        pass: &mut TargetPass<'_, 'a>,
        requests: &mut Requests<'a>,
        batch: &mut RequestBatch,
        accepted: &[u32],
    ) -> Result<()> {
        let _device = self.chains[0].execution_device().map(|device| device.enter()).transpose()?;
        let ids = batch.cache()?.request_ids();
        ensure!(ids.len() == accepted.len(), "draft acceptance count differs");
        let leases = ids.iter().map(|id| self.requests.get(id)
            .map(|request| request.leases).context("draft request not admitted"))
            .collect::<Result<Vec<_>>>()?;
        let stage_leases: [Vec<WindowLease>; 3] = std::array::from_fn(|stage|
            leases.iter().map(|request| request[stage]).collect());
        let taps = pass.taps(batch)?;
        let mut proposal = unsafe {
            self.mains[0]
                .execute_rows(taps.values(), taps.batch_identity(), taps.rows())?
        };
        let [a, b, c] = &mut self.windows;
        unsafe {
            pass.commit_with_dspark(
                requests,
                batch,
                &mut proposal,
                &mut [a, b, c],
                [&stage_leases[0], &stage_leases[1], &stage_leases[2]],
                accepted,
            )
        }
    }
    /// Begin a completed decode batch's accepted-cache transaction on its own lane.
    pub fn begin_queued_commit(&mut self, lane: usize, pass: &impl crate::v41_target_pass::TargetCache<'a>,
        requests: &Requests<'a>, batch: &RequestBatch, accepted: &[u32]) -> Result<()> {
        let _device = self.chains[0].execution_device().map(|device| device.enter()).transpose()?;
        ensure!(lane < self.mains.len() && self.pending_commit_ids[lane].is_empty(), "commit lane busy or invalid");
        requests.validate_acceptance(batch, accepted)?;
        ensure!(accepted.iter().any(|&n| n > 0), "queued decode commit has no accepted rows");
        let ids = batch.cache()?.request_ids();
        let leases = ids.iter().map(|id| self.requests.get(id)
            .map(|request| request.leases).context("draft commit request not admitted"))
            .collect::<Result<Vec<_>>>()?;
        let stage_leases: [Vec<WindowLease>; 3] = std::array::from_fn(|stage|
            leases.iter().map(|request| request[stage]).collect());
        let taps = pass.taps(batch)?;
        let chunks = crate::v41_experts::dspark::prepare_commit_rows(taps.rows(), self.windows.each_ref(),
            stage_leases.each_ref().map(|leases| leases.as_slice()), accepted)?;
        let writes = [self.windows[0].prepare_async_write(&chunks[0], taps.rows().len() as u32)?,
            self.windows[1].prepare_async_write(&chunks[1], taps.rows().len() as u32)?,
            self.windows[2].prepare_async_write(&chunks[2], taps.rows().len() as u32)?];
        self.pending_commit_ids[lane] = ids.to_vec();
        unsafe { self.mains[lane].enqueue_commit(taps.values(), &batch.cache()?.positions(),
            self.windows.each_ref(), writes) }
    }
    pub fn poll_queued_commit(&self, lane: usize) -> Result<bool> {
        let _device = self.chains[0].execution_device().map(|device| device.enter()).transpose()?;
        self.mains[lane].poll_commit()
    }
    pub fn finish_queued_commit(&mut self, lane: usize, pass: &mut impl crate::v41_target_pass::TargetCache<'a>,
        requests: &mut Requests<'a>, batch: &mut RequestBatch, accepted: &[u32]) -> Result<()> {
        let _device = self.chains[0].execution_device().map(|device| device.enter()).transpose()?;
        self.mains[lane].publish_commit(&mut self.windows)?;
        pass.commit(requests, batch, accepted)?;
        self.pending_commit_ids[lane].clear();
        Ok(())
    }
    /// Drain before revoking any target/cache storage, including partial enqueue.
    pub fn abort_queued_commit(&mut self, lane: usize, requests: &mut Requests<'a>,
        batch: &mut RequestBatch) -> Result<()> {
        let _device = self.chains[0].execution_device().map(|device| device.enter()).transpose()?;
        // Target publication may have cancelled the batch before returning an
        // error. Keep cleanup independent of its now-unavailable cache view.
        let ids = if self.pending_commit_ids[lane].is_empty() {
            batch.cache().map(|cache| cache.request_ids().to_vec()).unwrap_or_default()
        } else { self.pending_commit_ids[lane].clone() };
        let mut result = self.mains[lane].abort_commit(&mut self.windows);
        self.pending_commit_ids[lane].clear();
        requests.revoke_batch(batch);
        for id in ids { if let Err(error) = self.release(id) { result = Err(error); } }
        result
    }

    /// Each lane owns its draft scratch and stream; both can replay concurrently.
    /// Short RefCell borrows never survive a scheduler yield.
    pub fn poll_propose(&mut self, lane: usize,
        inputs: &[(u64, u32, u64, usize)],
    ) -> Result<Option<(Vec<Vec<u32>>, u64)>> {
        let _device = self.chains[0].execution_device().map(|device| device.enter()).transpose()?;
        ensure!(lane < self.chains.len(), "invalid draft lane");
        if let Some((seeds, started)) = &self.pending[lane] {
            let started = *started;
            ensure!(seeds == inputs, "pending draft request inputs changed");
            let completed = match self.chains[lane].poll_replay() {
                Ok(None) => return Ok(None),
                Ok(Some(value)) => value,
                Err(error) => { self.pending[lane] = None; return Err(error); },
            };
            self.pending[lane] = None;
            let (packed, values) = completed;
            let active: Vec<_> = inputs.iter().enumerate()
                .filter(|(_, (_, _, end, remaining))| *end >= 2 && *remaining > 1).collect();
            let count = active.len();
            ensure!(packed.len() == (self.draft_width+1)*count && values.len() == self.draft_width*count, "draft output extent differs");
            let mut outputs: Vec<_> = inputs.iter().map(|(_, anchor, _, _)| vec![*anchor]).collect();
            for (row, &(output, &(id, anchor, _, remaining))) in active.iter().enumerate() {
                if self.wants_confidence() {
                    self.confidence_trace.insert(id, (0..self.draft_width).map(|step| values[step*count+row]).collect());
                }
                let tokens: Vec<_> = (0..=self.draft_width).map(|step| packed[step*count+row]).collect();
                ensure!(tokens[0] == anchor && tokens.iter().all(|&t| t < 129280), "invalid draft tokens");
                outputs[output] = tokens[..remaining.min(self.draft_limit+1)].to_vec();
            }
            return Ok(Some((outputs, started.elapsed().as_micros() as u64)));
        }
        let started = Instant::now();
        ensure!(self.pending[lane].is_none(), "shared draft proposal still pending");
        ensure!(!inputs.is_empty() && inputs.len() <= 16, "invalid draft batch size");
        let mut seen = std::collections::BTreeSet::new();
        for &(id, anchor, _, remaining) in inputs {
            ensure!(seen.insert(id) && self.requests.contains_key(&id), "invalid draft request identity");
            ensure!(anchor < 129280 && remaining > 0, "invalid draft request input");
        }
        for &(id, _, _, _) in inputs {
            self.confidence_trace.remove(&id);
        }
        let active: Vec<_> = inputs.iter().enumerate()
            .filter(|(_, (_, _, end, remaining))| *end >= 2 && *remaining > 1).collect();
        let outputs: Vec<_> = inputs.iter().map(|(_, anchor, _, _)| vec![*anchor]).collect();
        if active.is_empty() { return Ok(Some((outputs, started.elapsed().as_micros() as u64))); }
        let count = active.len();
        let tokens: Vec<_> = active.iter().map(|(_, (_, anchor, _, _))| *anchor as i32).collect();
        let bindings: [Vec<_>; 3] = std::array::from_fn(|stage| active.iter()
            .map(|(_, (id, _, end, _))| (self.requests[id].leases[stage], *end)).collect());
        self.chains[lane].stage_tokens(&tokens)?;
        let mut rngs: Vec<_> = self.requests.iter_mut().filter_map(|(id, request)|
            active.iter().position(|(_, (active_id, _, _, _))| active_id == id)
                .map(|index| (index, &mut request.rng))).collect();
        rngs.sort_by_key(|(index, _)| *index);
        self.chains[lane].stage_sampling(&mut rngs.into_iter().map(|(_, rng)| rng).collect::<Vec<_>>(),
            &vec![0.0; count])?;
        let windows = self.windows.each_ref();
        let bindings = bindings.each_ref().map(|rows| rows.as_slice());
        unsafe { self.chains[lane].begin_replay(windows, bindings)?; }
        self.pending[lane] = Some((inputs.to_vec(), started));
        Ok(None)
    }

    pub fn confidence_trace(&self, id: u64) -> Option<&[f32]> {
        self.confidence_trace.get(&id).map(Vec::as_slice)
    }
}

impl<'a> DraftPrefix<'a> {
    pub fn parts(&self) -> &[crate::v41_dspark_cache::DsparkPrefix<'a>] {
        self.windows.get().as_slice()
    }
    pub fn from_parts(library: &'a NativeLibrary, windows: Vec<crate::v41_dspark_cache::DsparkPrefix<'a>>) -> Result<Self> {
        ensure!(windows.len() == 3, "draft prefix requires three windows");
        let device = crate::v41_memory::device::Device {
            library, id: windows[0].parts().2.buffer.device_id,
        };
        ensure!(windows.iter().all(|window| window.parts().2.buffer.device_id == device.id),
            "draft prefix windows span devices");
        Ok(Self { windows: device.own(move || Ok(windows))? })
    }
}

#[cfg(test)]
mod restored_prefix_tests {
    use super::*;
    use crate::v41_dspark_cache::DsparkPrefix;
    use crate::v41_memory::{device::Device, SnapshotStorage};

    #[test]
    #[ignore = "requires DS41RT_NATIVE_LIB and two CUDA devices"]
    fn native_host_draft_prefix_keeps_gpu1_owner() -> Result<()> {
        let lib = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        lib.cuda_set_device(0)?;
        let device = Device { library: &lib, id: 1 };
        let mut window = device.own(|| DsparkWindow::new(&lib, 1, 1, usize::MAX))?;
        // Exercise both the allocation fallback and the preallocated serving arena.
        for pooled in [false, true] {
            if pooled { device.run(|| window.reserve_prefixes(3))?; }
            let rings = (0..3).map(|_| {
                window.device().run(|| SnapshotStorage::new(
                    window.library(), 64, window.prefix_pool()))
                    .map(|ring| DsparkPrefix::from_parts(window.owner(), 2, ring))
            }).collect::<Result<Vec<_>>>()?;
            assert_eq!(lib.cuda_get_device()?, 0);
            let prefix = DraftPrefix::from_parts(&lib, rings)?;
            assert_eq!(prefix.windows.device.id, 1);
            assert!(prefix.parts().iter().all(|p| p.parts().2.buffer.device_id == 1));
            drop(prefix);
            assert_eq!(lib.cuda_get_device()?, 0);
        }
        Ok(())
    }
}

impl<'w, 'a, C: DraftChain<'a>> DraftRuntime<'w, 'a, C> {
    pub fn windows(&self) -> &[DsparkWindow<'a>; 3] {
        &self.windows
    }
}
impl<'w, 'a> DraftRuntime<'w, 'a> {
    /// Inputs are (request identity, anchor, committed end, remaining output budget).
    /// Returned rows retain input order; short histories/budgets use the anchor only.
    pub fn propose(&mut self, lib: &'a NativeLibrary,
        inputs: &[(u64, u32, u64, usize)],
    ) -> Result<Vec<Vec<u32>>> {
        ensure!(self.pending.iter().all(Option::is_none), "shared draft proposal still pending");
        ensure!(!inputs.is_empty() && inputs.len() <= 16, "invalid draft batch size");
        let mut seen = std::collections::BTreeSet::new();
        for &(id, anchor, _, remaining) in inputs {
            ensure!(seen.insert(id) && self.requests.contains_key(&id), "invalid draft request identity");
            ensure!(anchor < 129280 && remaining > 0, "invalid draft request input");
        }
        for &(id, _, _, _) in inputs {
            self.confidence_trace.remove(&id);
        }
        let active: Vec<_> = inputs.iter().enumerate()
            .filter(|(_, (_, _, end, remaining))| *end >= 2 && *remaining > 1).collect();
        let mut outputs: Vec<_> = inputs.iter().map(|(_, anchor, _, _)| vec![*anchor]).collect();
        if active.is_empty() { return Ok(outputs); }
        let count = active.len();
        let tokens: Vec<_> = active.iter().map(|(_, (_, anchor, _, _))| *anchor as i32).collect();
        let bindings: [Vec<_>; 3] = std::array::from_fn(|stage| active.iter()
            .map(|(_, (id, _, end, _))| (self.requests[id].leases[stage], *end)).collect());
        self.chains[0].set_tokens(&tokens)?;
        let mut rngs: Vec<_> = self.requests.iter_mut().filter_map(|(id, request)|
            active.iter().position(|(_, (active_id, _, _, _))| active_id == id)
                .map(|index| (index, &mut request.rng))).collect();
        rngs.sort_by_key(|(index, _)| *index);
        self.chains[0].prepare_sampling(&mut rngs.into_iter().map(|(_, rng)| rng).collect::<Vec<_>>(),
            &vec![0.0; count])?;
        let windows = self.windows.each_ref();
        let bindings = bindings.each_ref().map(|rows| rows.as_slice());
        if !self.chains[0].has_graph(count) {
            unsafe { self.chains[0].capture(windows, bindings)?; }
        }
        unsafe { self.chains[0].replay(windows, bindings)?; }
        let buffer = self.chains[0].draft_output()?[0];
        let mut bytes = vec![0; buffer.bytes];
        lib.copy_d2h(&mut bytes, buffer)?;
        ensure!(bytes.len() == (self.draft_width + 1) * count * 4, "draft token extent differs");
        let packed: Vec<_> = bytes.chunks_exact(4)
            .map(|b| u32::from_ne_bytes(b.try_into().unwrap())).collect();
        if self.wants_confidence() {
            let confidence = self.chains[0].draft_output()?[2];
            ensure!(confidence.bytes == self.draft_width * count * 4, "draft confidence extent differs");
            let mut bytes = vec![0; confidence.bytes];
            lib.copy_d2h(&mut bytes, confidence)?;
            let values: Vec<_> = bytes.chunks_exact(4)
                .map(|b| f32::from_ne_bytes(b.try_into().unwrap())).collect();
            for (row, &(_, &(id, _, _, _))) in active.iter().enumerate() {
                self.confidence_trace.insert(id,
                    (0..self.draft_width).map(|step| values[step * count + row]).collect());
            }
        }
        for (row, &(output, &(_, anchor, _, remaining))) in active.iter().enumerate() {
            let tokens: Vec<_> = (0..=self.draft_width).map(|step| packed[step * count + row]).collect();
            ensure!(tokens[0] == anchor && tokens.iter().all(|&token| token < 129280), "invalid draft tokens");
            outputs[output] = tokens[..remaining.min(self.draft_limit + 1)].to_vec();
        }
        Ok(outputs)
    }

}
