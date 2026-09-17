//! One admission identity for persistent backbone caches and mapped engram history.
use crate::v41_backbone_cache::{BackboneCache, CacheBatch, CacheLease, CacheWork, CacheStage};
use crate::v41_backbone_execution::BackboneExecution;
use crate::v41_backbone_lane::BackboneLane;
use crate::v41_engram::layer::EngramGate;
use crate::v41_engram::{EngramDeviceRows, EngramUploadPoll};
use crate::v41_target_embedding::TargetEmbeddingWave;
use anyhow::{ensure, Context, Result};
use ds41rt_core::{EngramHistory, EngramPrefillCursor};
mod reservation;
mod images;
pub(crate) use images::RequestImages;
use ds41rt_ffi::NativeLibrary;
use ds41rt_loader::{EngramPipeline, EngramRequestTokens, EngramWave};
use ds41rt_transport::ExpertV2SourceKind;

struct Request {
    lease: CacheLease,
    history: EngramHistory,
    prefill: Option<EngramPrefillCursor>,
    images: RequestImages,
}
pub(crate) struct RequestPrefix<'a> {
    cache: crate::v41_backbone_cache::BackbonePrefix<'a>,
    history: EngramHistory,
}
impl RequestPrefix<'_> {
    pub fn end(&self) -> u64 { self.cache.end() }
}
pub(crate) struct RequestTokens<'a> {
    pub lease: CacheLease,
    pub tokens: &'a [u32],
    pub image_mask: Option<&'a [bool]>,
    pub kind: ExpertV2SourceKind,
}
pub(crate) struct RequestBatch {
    cache: CacheBatch,
    prepared: Vec<EngramPrefillCursor>,
    engram: Option<EngramWave>,
    leases: Vec<CacheLease>,
    tokens: Vec<u32>,
    image_mask: Vec<u8>,
    finished: bool,
}
impl RequestBatch {
    pub fn cache(&self) -> Result<&CacheBatch> {
        ensure!(!self.finished, "request batch finished");
        Ok(&self.cache)
    }
    pub fn image_mask(&self) -> &[u8] {
        &self.image_mask
    }
    pub fn cancel(&mut self) {
        self.finished = true;
        if let Some(engram) = &mut self.engram { engram.cancel(); }
    }
}
impl Drop for RequestBatch {
    fn drop(&mut self) {
        if let Some(engram) = &mut self.engram { engram.cancel(); }
    }
}
pub(crate) struct Requests<'a> {
    cache: BackboneCache<'a>,
    prefix_histories: [Option<(CacheLease, EngramHistory)>; 2],
    pipeline: EngramPipeline,
    slots: Vec<Option<Request>>,
    image_requests: usize,
}
impl<'a> Requests<'a> {
    pub fn new(
        library: &'a NativeLibrary,
        pipeline: EngramPipeline,
        slots: usize,
        pages: [usize; 4],
        cache_budget: usize,
    ) -> Result<Self> {
        Ok(Self {
            cache: BackboneCache::new(library, slots, pages, cache_budget)?,
            prefix_histories: [None, None],
            pipeline,
            slots: (0..slots).map(|_| None).collect(),
            image_requests: 0,
        })
    }
    pub fn new_distributed(
        library: &'a NativeLibrary,
        pipeline: EngramPipeline,
        slots: usize,
        pages: [usize; 4],
        map: crate::v41_backbone_cache::CachePlacement,
        budgets: [usize; 2],
    ) -> Result<Self> {
        Ok(Self {
            cache: BackboneCache::new_distributed(library, map, slots, pages, budgets)?,
            prefix_histories: [None, None],
            pipeline,
            slots: (0..slots).map(|_| None).collect(),
            image_requests: 0,
        })
    }
    pub fn cache(&self) -> &BackboneCache<'a> {
        &self.cache
    }
    pub fn new_replicated(library:&'a NativeLibrary,pipeline:EngramPipeline,slots:usize,
        pages:[usize;4],map:crate::v41_backbone_cache::CachePlacement,budgets:[usize;2])->Result<Self> {
        Ok(Self { cache:BackboneCache::new_replicated(library,map,slots,pages,budgets)?,
            prefix_histories:[None,None],pipeline,slots:(0..slots).map(|_|None).collect(),image_requests:0 })
    }
    fn request(&self, lease: CacheLease) -> Result<&Request> {
        self.cache.request_id(lease)?;
        self.slots
            .iter()
            .flatten()
            .find(|r| r.lease == lease)
            .context("request history missing")
    }
    pub fn admit(&mut self, slot: usize, id: u64) -> Result<CacheLease> {
        ensure!(
            slot < self.slots.len() && self.slots[slot].is_none(),
            "request slot occupied"
        );
        let history = self.pipeline.new_history()?;
        let lease = self.cache.begin_request(slot, id)?;
        self.slots[slot] = Some(Request { lease, history, prefill: None, images: RequestImages::default() });
        Ok(lease)
    }
    pub fn attach_images(&mut self, lease: CacheLease, images: RequestImages) -> Result<()> {
        let request = self.request(lease)?;
        ensure!(request.history.position() == 0 && self.cache.committed_end(lease)? == 0
            && request.prefill.is_none() && request.images.is_empty(),
            "request images must be attached before prefix restoration or prefill");
        self.image_requests += usize::from(!images.is_empty());
        self.slots.iter_mut().flatten().find(|r| r.lease == lease).unwrap().images = images;
        Ok(())
    }
    pub fn images(&self, lease: CacheLease) -> Result<&RequestImages> { Ok(&self.request(lease)?.images) }
    pub fn install_image_features(&mut self, lease: CacheLease, index: usize, features: Vec<u8>) -> Result<()> {
        ensure!(self.request(lease)?.prefill.is_none(), "image preparation cannot race reserved prefill");
        self.slots.iter_mut().flatten().find(|r| r.lease == lease).unwrap().images.install(index, features)
    }
    fn resolved_masks<'m>(&self, requests: &[RequestTokens<'m>], reserved: bool)
        -> Result<Option<Vec<Option<std::borrow::Cow<'m, [bool]>>>>> {
        if self.image_requests == 0 { return Ok(None); }
        requests.iter().map(|r| {
            let request = self.request(r.lease)?;
            let history = if reserved { request.prefill.as_ref().map_or(&request.history, |p| p.history()) }
                else { &request.history };
            request.images.resolve_mask(history.position(), r.tokens.len(), r.image_mask)
        }).collect::<Result<Vec<_>>>().map(Some)
    }
    pub fn release(&mut self, lease: CacheLease) -> Result<()> {
        if self.cache.request_id(lease).is_ok() { self.cache.ensure_releasable(lease)?; }
        let slot = self
            .slots
            .iter()
            .position(|r| r.as_ref().is_some_and(|r| r.lease == lease))
            .context("request already released")?;
        // Revoke the history owner even if device-cache cleanup fails.
        if !self.slots[slot].as_ref().unwrap().images.is_empty() { self.image_requests -= 1; }
        self.slots[slot] = None;
        // Cache failure invalidation may already have revoked its root lease.
        if self.cache.request_id(lease).is_ok() {
            self.cache.release(&[lease])?;
        }
        Ok(())
    }
    pub fn release_if_present(&mut self, lease: CacheLease) -> Result<()> {
        if self.slots.iter().flatten().any(|r| r.lease == lease) { self.release(lease)?; }
        Ok(())
    }
    pub fn install_prefix_pool(&mut self, pool: crate::v41_memory::SnapshotPool<'a>) -> Result<()> {
        self.cache.install_prefix_pool(pool)
    }
    pub fn retain_prefix(&mut self, lease: CacheLease, budget: usize) -> Result<RequestPrefix<'a>> {
        let request = self.request(lease)?;
        ensure!(request.prefill.is_none() && request.history.position() == self.cache.committed_end(lease)?,
            "request prefix has pending or inconsistent history");
        let history = request.history.fork()?;
        let cache = self.cache.retain_prefix(lease, budget)?;
        Ok(RequestPrefix { cache, history })
    }
    pub fn queue_prefix(&mut self, lane: usize, lease: CacheLease) -> Result<()> {
        ensure!(self.prefix_histories.get(lane).context("invalid snapshot lane")?.is_none(),
            "request snapshot lane is occupied");
        let request = self.request(lease)?;
        ensure!(request.prefill.is_none() && request.history.position() == self.cache.committed_end(lease)?,
            "request prefix has pending or inconsistent history");
        let history = request.history.fork()?;
        self.cache.queue_prefix(lane, lease, crate::v41_backbone_cache::BackbonePrefix::device_bytes())?;
        self.prefix_histories[lane] = Some((lease, history));
        Ok(())
    }
    pub fn prefix_ready(&self, lane: usize, lease: CacheLease) -> Result<bool> {
        ensure!(self.prefix_histories.get(lane).and_then(Option::as_ref).is_some_and(|(l, _)| *l == lease),
            "request snapshot owner differs");
        self.cache.prefix_ready(lane, lease)
    }
    pub fn finish_prefix(&mut self, lane: usize, lease: CacheLease) -> Result<RequestPrefix<'a>> {
        ensure!(self.prefix_ready(lane, lease)?, "request snapshot copies are incomplete");
        let cache = self.cache.finish_prefix(lane, lease)?;
        let history = self.prefix_histories[lane].take().unwrap().1;
        Ok(RequestPrefix { cache, history })
    }
    pub fn abort_prefix(&mut self, lane: usize) -> Result<()> {
        let drained = self.cache.abort_prefix(lane);
        if let Some(history) = self.prefix_histories.get_mut(lane) { *history = None; }
        drained
    }
    pub fn restore_prefix(&mut self, lease: CacheLease, prefix: &RequestPrefix<'a>) -> Result<()> {
        self.restore_retained(lease, prefix, None)
    }
    pub fn restore_encoder_continuation(&mut self, lease: CacheLease,
        prefix: &RequestPrefix<'a>, prompt_end: u64) -> Result<()> {
        self.restore_retained(lease, prefix, Some(prompt_end))
    }
    fn restore_retained(&mut self, lease: CacheLease, prefix: &RequestPrefix<'a>,
        prompt_end: Option<u64>) -> Result<()> {
        let request = self.request(lease)?;
        ensure!(request.history.position() == 0 && request.prefill.is_none()
            && prefix.history.position() == prefix.cache.end(), "invalid request prefix restore");
        let history = prefix.history.fork()?;
        let restored = match prompt_end {
            Some(end) => self.cache.restore_encoder_continuation(lease, &prefix.cache, end),
            None => self.cache.restore_prefix(lease, &prefix.cache),
        };
        if let Err(error) = restored {
            if let Err(cleanup) = self.release(lease) {
                tracing::error!(%cleanup, "releasing failed request prefix restore");
            }
            return Err(error);
        }
        self.slots.iter_mut().flatten().find(|r| r.lease == lease)
            .expect("validated restored request").history = history;
        Ok(())
    }
    fn encoder_history_at(&self, lease: CacheLease, start: usize, prompt: &[u32]) -> Result<EngramHistory> {
        ensure!(start <= prompt.len(), "encoder history position exceeds prompt");
        let recent_start = start.saturating_sub(3);
        let image_mask = self.request(lease)?.images.mask(recent_start as u64, start - recent_start)?
            .map(|m| m.into_iter().map(u8::from).collect::<Vec<_>>());
        self.pipeline.history_at(start as u64, &prompt[recent_start..start], image_mask.as_deref())
    }
    pub fn restore_encoder_prefix(
        &mut self,
        lease: CacheLease,
        prefix: &RequestPrefix<'a>,
        end: usize,
        prompt: &[u32],
    ) -> Result<usize> {
        let request = self.request(lease)?;
        ensure!(
            request.history.position() == 0 && request.prefill.is_none() && end <= prompt.len(),
            "invalid encoder history restore"
        );
        let start = end.saturating_sub(128);
        let history = self.encoder_history_at(lease, start, prompt)?;
        if let Err(error) =
            self.cache
                .restore_encoder_prefix(lease, &prefix.cache, end as u64, prompt.len() as u64)
        {
            if let Err(cleanup) = self.release_if_present(lease) {
                tracing::error!(%cleanup, "releasing failed encoder history restore");
            }
            return Err(error);
        }
        self.slots
            .iter_mut()
            .flatten()
            .find(|r| r.lease == lease)
            .expect("validated restored request")
            .history = history;
        Ok(start)
    }
    pub fn validate(&self, batch: &RequestBatch) -> Result<()> {
        ensure!(!batch.finished, "request batch finished");
        self.cache.validate_batch(&batch.cache)?;
        for &lease in &batch.leases {
            ensure!(
                self.request(lease)?.history.position() == self.cache.history_end(lease)?,
                "engram and backbone history differ"
            );
        }
        Ok(())
    }
    pub fn begin_encoder(&mut self, lease: CacheLease, prompt_end: u64) -> Result<()> {
        self.request(lease)?;
        self.cache.begin_encoder(lease, prompt_end)
    }
    pub fn begin_decoder_replay(&mut self, lease: CacheLease) -> Result<u64> {
        ensure!(self.request(lease)?.history.position() == self.cache.committed_end(lease)?,
            "encoder history differs before replay");
        let start = self.cache.begin_decoder_replay(lease)?;
        self.slots.iter_mut().flatten().find(|r| r.lease == lease)
            .expect("validated request").prefill = None;
        Ok(start)
    }
    pub fn prepare_replay(&self, work: &[CacheWork]) -> Result<RequestBatch> {
        let cache = self.cache.plan_replay(work)?;
        let rows = cache.positions().len();
        let mut image_mask = vec![0; rows];
        if self.image_requests != 0 {
            let mut offset = 0;
            for (w, chunk) in work.iter().zip(cache.window_chunks(20)?) {
                if let Some(mask) = self.request(w.lease)?.images.mask(chunk.position, chunk.tokens as usize)? {
                    for (dst, src) in image_mask[offset..offset + mask.len()].iter_mut().zip(mask) { *dst = u8::from(src); }
                }
                offset += chunk.tokens as usize;
            }
        }
        let batch = RequestBatch { cache, prepared: Vec::new(), engram: None, leases: work.iter().map(|w| w.lease).collect(),
            tokens: Vec::new(), image_mask, finished: false };
        self.validate(&batch)?;
        Ok(batch)
    }
    /// Start hashes, prefetch and bounded gather as soon as token IDs are known.
    pub fn prepare(&self, requests: &[RequestTokens<'_>]) -> Result<RequestBatch> {
        let work = requests
            .iter()
            .map(|r| {
                Ok(CacheWork {
                    lease: r.lease,
                    tokens: u32::try_from(r.tokens.len())?,
                    kind: r.kind,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let cache = self.cache.plan(&work)?;
        let masks = self.resolved_masks(requests, false)?;
        let mut inputs = Vec::with_capacity(requests.len());
        let mut tokens = Vec::new();
        let mut mask = Vec::new();
        for (i, r) in requests.iter().enumerate() {
            let image_mask = masks.as_ref().map_or(r.image_mask, |m| m[i].as_deref());
            let request = self.request(r.lease)?;
            ensure!(request.prefill.is_none(), "reserved encoder requires reservation preparation");
            let history = &request.history;
            ensure!(
                history.position() == self.cache.history_end(r.lease)?,
                "request histories differ"
            );
            ensure!(
                r.image_mask.is_none_or(|m| m.len() == r.tokens.len()),
                "image mask length differs"
            );
            inputs.push(EngramRequestTokens {
                history,
                token_ids: r.tokens,
                image_mask,
            });
            tokens.extend_from_slice(r.tokens);
            mask.extend((0..r.tokens.len()).map(|i| u8::from(image_mask.is_some_and(|m| m[i]))));
        }
        let engram = self.pipeline.prepare(&inputs)?;
        Ok(RequestBatch {
            cache,
            prepared: Vec::new(),
            engram: Some(engram),
            leases: requests.iter().map(|r| r.lease).collect(),
            tokens,
            image_mask: mask,
            finished: false,
        })
    }
    fn embedding_features(&self, batch: &RequestBatch) -> Result<Vec<(usize, &[u8])>> {
        self.validate(batch)?;
        let mut features = Vec::new();
        let mut offset = 0;
        for (&lease, chunk) in batch.leases.iter().zip(batch.cache.window_chunks(batch.cache.stage().windows().start)?) {
            let images = &self.request(lease)?.images;
            for i in 0..chunk.tokens as usize {
                if batch.image_mask[offset+i] != 0 {
                    features.push((offset+i, images.row(chunk.position+i as u64)
                        .context("request image embedding not prepared")?));
                }
            }
            offset += chunk.tokens as usize;
        }
        Ok(features)
    }
    /// The token borrow belongs to the owned batch, not the shared request bank.
    pub fn text_embedding_input<'b>(&self, batch: &'b RequestBatch) -> Result<Option<(&'b [u32], Vec<u64>)>> {
        self.validate(batch)?;
        Ok(batch.image_mask.iter().all(|&b| b == 0)
            .then(|| (batch.tokens.as_slice(), batch.cache.positions())))
    }
    /// # Safety
    /// No external writes race embedding or lane buffers. Image rows require
    /// completed features owned by the corresponding admitted request.
    pub unsafe fn begin_input(
        &self,
        batch: &RequestBatch,
        embedding: &mut TargetEmbeddingWave<'_, '_>,
        lane: &mut BackboneLane<'_, '_>,
    ) -> Result<()> {
        self.validate(batch)?;
        let positions = batch.cache.positions();
        let embedded = if batch.image_mask.iter().all(|&b| b == 0) {
            embedding.execute(&batch.tokens, &positions)?
        } else {
            let features = self.embedding_features(batch)?;
            embedding.execute_with_images(&batch.tokens, &positions, &features)?
        };
        unsafe {
            lane.begin_embedded(&embedded)?;
        }
        Ok(())
    }
    /// # Safety
    /// This lane was initialized/advanced for batch and all device producers have
    /// completed. Poll on the CUDA-owning thread. False means I/O is still pending.
    pub unsafe fn poll_engram(
        &self,
        batch: &mut RequestBatch,
        upload: &mut EngramDeviceRows<'_>,
        gate: &mut EngramGate<'_, '_>,
        lane: &mut BackboneLane<'_, '_>,
    ) -> Result<bool> {
        self.validate(batch)?;
        let (pending_layer, positions) = lane.pending_engram()?;
        let layer = ds41rt_core::ENGRAM_LAYERS
            .iter()
            .position(|&l| l as usize == pending_layer)
            .context("prepared layer has no engram")?;
        ensure!(
            positions == batch.cache.positions(),
            "engram lane positions differ"
        );
        let histories = if batch.prepared.is_empty() {
            batch.leases.iter().map(|&l| Ok(&self.request(l)?.history)).collect::<Result<Vec<_>>>()?
        } else {
            batch.prepared.iter().map(|cursor| cursor.history()).collect()
        };
        match upload.poll_wave(&self.pipeline, batch.engram.as_mut().context("decoder replay has no engram work")?, &histories, layer)? {
            EngramUploadPoll::Pending => Ok(false),
            EngramUploadPoll::Cancelled => anyhow::bail!("engram gather cancelled"),
            EngramUploadPoll::Ready(rows) => {
                unsafe {
                    lane.apply_engram(gate, &rows)?;
                }
                Ok(true)
            }
        }
    }
    /// Poll only the I/O gather under the shared request borrow. The returned
    /// lease owns its staging so GPU upload/gating can finish outside that borrow.
    pub fn poll_engram_gather(&self, batch: &mut RequestBatch,
        lane: &BackboneLane<'_, '_>) -> Result<ds41rt_loader::EngramGatherPoll> {
        self.validate(batch)?;
        let (pending_layer, positions) = lane.pending_engram()?;
        let layer = ds41rt_core::ENGRAM_LAYERS
            .iter()
            .position(|&l| l as usize == pending_layer)
            .context("prepared layer has no engram")?;
        ensure!(
            positions == batch.cache.positions(),
            "engram lane positions differ"
        );
        let histories = if batch.prepared.is_empty() {
            batch.leases.iter().map(|&l| Ok(&self.request(l)?.history)).collect::<Result<Vec<_>>>()?
        } else {
            batch.prepared.iter().map(|cursor| cursor.history()).collect()
        };
        self.pipeline.poll(batch.engram.as_mut().context("decoder replay has no engram work")?, &histories, layer)
    }
    pub fn validate_acceptance(&self, batch: &RequestBatch, accepted: &[u32]) -> Result<()> {
        self.validate(batch)?;
        let counts = accepted.iter().map(|&n| n as usize).collect::<Vec<_>>();
        let histories = batch
            .leases
            .iter()
            .map(|&l| Ok(&self.request(l)?.history))
            .collect::<Result<Vec<_>>>()?;
        if let Some(engram) = &batch.engram {
            engram.validate_commit(&histories, &counts)?;
        } else {
            ensure!(batch.cache.stage() == CacheStage::Replay, "missing engram outside decoder replay");
            let chunks = batch.cache.window_chunks(20)?;
            ensure!(accepted.len() == chunks.len() && chunks.iter().zip(accepted).all(|(c, &n)| n <= c.tokens),
                "decoder acceptance exceeds proposal");
        }
        Ok(())
    }
    pub fn abort_cache_commit(&mut self, execution: &mut BackboneExecution<'_, '_>) -> Result<()> {
        execution.abort_cache_commit(&mut self.cache)
    }
    /// Invalidate every participant after a partially applied combined commit.
    pub fn revoke_batch(&mut self, batch: &mut RequestBatch) {
        batch.cancel();
        for &lease in &batch.leases {
            // An inner commit may have already released this participant.
            if self.slots.iter().flatten().any(|r| r.lease == lease) {
                if let Err(cleanup) = self.release(lease) {
                    tracing::error!(%cleanup, "releasing failed combined transaction");
                }
            }
        }
    }
    /// Preflight engram histories before any device commit. Publication failure
    /// revokes participating requests; the enclosing scheduler still owns dSpark.
    pub fn commit(
        &mut self,
        batch: &mut RequestBatch,
        execution: &mut BackboneExecution<'_, '_>,
        accepted: &[u32],
    ) -> Result<()> {
        self.commit_with(batch, accepted, |cache, batch, accepted| {
            execution.commit(cache, batch, accepted)
        })
    }
    pub fn commit_distributed(
        &mut self,
        batch: &mut RequestBatch,
        execution: &mut crate::v41_backbone_execution::DistributedExecution<'_, '_>,
        accepted: &[u32],
    ) -> Result<()> {
        self.commit_with(batch, accepted, |cache, batch, accepted| {
            execution.finish_cache_commit(cache, batch, accepted)
        })
    }
    pub fn abort_distributed_cache_commit(
        &mut self,
        execution: &mut crate::v41_backbone_execution::DistributedExecution<'_, '_>,
    ) -> Result<()> {
        execution.abort_cache_commit(&mut self.cache)
    }
    fn commit_with(
        &mut self,
        batch: &mut RequestBatch,
        accepted: &[u32],
        publish: impl FnOnce(
            &mut BackboneCache<'a>,
            &crate::v41_backbone_cache::CacheBatch,
            &[u32],
        ) -> Result<()>,
    ) -> Result<()> {
        self.validate_acceptance(batch, accepted)?;
        let counts = accepted.iter().map(|&n| n as usize).collect::<Vec<_>>();
        let result = (|| -> Result<()> {
            publish(&mut self.cache, &batch.cache, accepted)?;
            let mut histories = self
                .slots
                .iter_mut()
                .flatten()
                .filter_map(|r| {
                    batch
                        .leases
                        .iter()
                        .position(|&l| l == r.lease)
                        .map(|i| (i, &mut r.history))
                })
                .collect::<Vec<_>>();
            histories.sort_by_key(|(i, _)| *i);
            let mut histories = histories.into_iter().map(|(_, h)| h).collect::<Vec<_>>();
            if let Some(engram) = &mut batch.engram { engram.commit(&mut histories, &counts) } else { Ok(()) }
        })();
        batch.cancel();
        if let Err(error) = result {
            for &lease in &batch.leases {
                if let Err(cleanup) = self.release(lease) {
                    tracing::error!(%cleanup,"releasing failed request transaction");
                }
            }
            return Err(error);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v41_backbone_execution::CacheProducerWeights;
    #[test]
    fn real_request_prefetch_upload_cancel_and_failed_commit() -> Result<()> {
        let Some(path) = std::env::var_os("DS41RT_REQUESTS_LIBRARY") else {
            eprintln!("skip request integration GPU test: DS41RT_REQUESTS_LIBRARY unset");
            return Ok(());
        };
        let model = std::path::PathBuf::from(
            std::env::var_os("DS41RT_REQUESTS_MODEL").context("DS41RT_REQUESTS_MODEL required")?,
        );
        let lib = unsafe { NativeLibrary::load(path)? };
        let catalog =
            ds41rt_loader::read_official_v41_catalog(ds41rt_loader::OFFICIAL_V41_MODEL_ID, &model)?;
        let map = ds41rt_loader::EngramTokenMap::from_file(&model.join("tokenizer.json"))?;
        let pipeline = unsafe { EngramPipeline::new(&catalog, map, 80, 2, 8 * 1024 * 1024)? };
        let mut requests = Requests::new(
            &lib,
            pipeline,
            16,
            [16; 4],
            BackboneCache::device_bytes(16, [16; 4])?,
        )?;
        let leases = (0..16)
            .map(|slot| requests.admit(slot, 700 + slot as u64))
            .collect::<Result<Vec<_>>>()?;
        let tokens = [17, 29, 31, 47, 61];
        let mask = [false, false, true, false, false];
        let input = leases
            .iter()
            .map(|&lease| RequestTokens {
                lease,
                tokens: &tokens,
                image_mask: Some(&mask),
                kind: ExpertV2SourceKind::Prefill,
            })
            .collect::<Vec<_>>();
        let mut batch = requests.prepare(&input)?;
        requests.validate(&batch)?;
        assert_eq!(batch.tokens.len(), 80);
        assert_eq!(
            batch.cache.positions(),
            (0..16).flat_map(|_| 0..5).collect::<Vec<_>>()
        );
        let mut cancelled = requests.prepare(&input)?;
        cancelled.cancel();
        assert!(requests.validate(&cancelled).is_err());
        assert!(cancelled.cache().is_err());
        let mut upload = EngramDeviceRows::new(&lib, 80, EngramDeviceRows::device_bytes(80)?)?;
        for layer in 0..2 {
            let histories = batch
                .leases
                .iter()
                .map(|&l| Ok(&requests.request(l)?.history))
                .collect::<Result<Vec<_>>>()?;
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
            loop {
                match upload.poll_wave(&requests.pipeline, batch.engram.as_mut().context("decoder replay has no engram work")?, &histories, layer)? {
                    EngramUploadPoll::Ready(view) => {
                        assert_eq!(view.rows, 80);
                        assert_eq!(view.layer_index, layer);
                        let mut text_mask = vec![0; 80];
                        lib.copy_d2h(&mut text_mask, view.text_mask)?;
                        assert_eq!(
                            text_mask,
                            batch.image_mask.iter().map(|&m| 1 - m).collect::<Vec<_>>()
                        );
                        eprintln!("PASS request batch layer={layer} mapped prefetch/gather/upload and image barriers");
                        break;
                    }
                    EngramUploadPoll::Cancelled => {
                        anyhow::bail!("live batch unexpectedly cancelled")
                    }
                    EngramUploadPoll::Pending => {
                        ensure!(
                            std::time::Instant::now() < deadline,
                            "engram gather timed out"
                        );
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }
                }
            }
        }
        let weights = CacheProducerWeights::load(
            &lib,
            &catalog,
            CacheProducerWeights::device_bytes(&lib, &catalog)?,
            1024 * 1024,
        )?;
        let mut execution =
            BackboneExecution::new(&weights, 80, BackboneExecution::workspace_bytes(&lib, 80)?)?;
        // Acceptance preflight must preserve a usable batch on invalid counts.
        assert!(requests
            .commit(&mut batch, &mut execution, &[6; 16])
            .is_err());
        requests.validate(&batch)?;
        // An incomplete model pass must not publish engram-only acceptance.
        assert!(requests
            .commit(&mut batch, &mut execution, &[2; 16])
            .is_err());
        assert!(requests.validate(&batch).is_err());
        for &lease in &leases {
            assert!(requests.request(lease).is_err());
        }
        for slot in 0..16 {
            let lease = requests.admit(slot, 700 + slot as u64)?;
            assert_ne!(lease, leases[slot]);
            assert_eq!(requests.request(lease)?.history.position(), 0);
            assert_eq!(requests.cache.committed_end(lease)?, 0);
            requests.release(lease)?;
        }
        let queued = (0..16).map(|slot| requests.admit(slot, 1700 + slot as u64))
            .collect::<Result<Vec<_>>>()?;
        for &lease in &queued { requests.begin_encoder(lease, 10)?; }
        let input = queued.iter().map(|&lease| RequestTokens { lease, tokens: &tokens,
            image_mask: Some(&mask), kind: ExpertV2SourceKind::Prefill }).collect::<Vec<_>>();
        let mut first = requests.reserve_encoder(&input)?;
        let mut second = requests.reserve_encoder(&input)?;
        assert!(requests.reserve_encoder(&input).is_err());
        assert!(requests.prepare(&input).is_err());
        let map = ds41rt_loader::EngramTokenMap::from_file(&model.join("tokenizer.json"))?;
        let all_tokens = tokens.repeat(2);
        let all_mask = mask.repeat(2);
        for (i, &lease) in queued.iter().enumerate() {
            let live = requests.request(lease)?;
            assert_eq!(live.history.position(), 0);
            assert_eq!(live.prefill.as_ref().unwrap().history().position(), 10);
            assert_eq!(requests.cache.encoder_prepared_end(lease)?, 10);
            let full = map.prepare_batch(&live.history, 0, &all_tokens, Some(&all_mask), 10)?;
            assert_eq!(first.engram.as_ref().unwrap().batches()[i].hashes(), &full.hashes()[..5]);
            assert_eq!(second.engram.as_ref().unwrap().batches()[i].hashes(), &full.hashes()[5..]);
        }
        for batch in [&mut first, &mut second] {
            requests.validate(batch)?;
            for layer in 0..2 {
                let histories = batch.prepared.iter().map(|p| p.history()).collect::<Vec<_>>();
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
                loop {
                    match upload.poll_wave(&requests.pipeline, batch.engram.as_mut().unwrap(), &histories, layer)? {
                        EngramUploadPoll::Ready(view) => { assert_eq!(view.rows, 80); break; }
                        EngramUploadPoll::Cancelled => anyhow::bail!("reserved gather cancelled"),
                        EngramUploadPoll::Pending => {
                            ensure!(std::time::Instant::now() < deadline, "reserved gather timed out");
                            std::thread::sleep(std::time::Duration::from_millis(1));
                        }
                    }
                }
            }
        }
        assert!(requests.validate_acceptance(&second, &[5; 16]).is_err());
        for &lease in &queued { requests.release(lease)?; }
        assert!(requests.validate(&first).is_err());
        assert!(requests.validate(&second).is_err());
        first.cancel(); second.cancel();
        eprintln!("PASS reserved Engram requests: 16 admissions, two chunks, exact hashes/image barriers, mapped I/O, extent rejection and release");
        eprintln!("PASS 16 request owners: cancelled batch rejected; incomplete commit revoked cache/engram admission; fresh generations recovered");
        Ok(())
    }
}

#[cfg(test)]
mod image_tests;

impl<'a> RequestPrefix<'a> {
    pub fn parts(&self) -> (&crate::v41_backbone_cache::BackbonePrefix<'a>, &EngramHistory) {
        (&self.cache, &self.history)
    }
    pub fn from_parts(cache: crate::v41_backbone_cache::BackbonePrefix<'a>, history: EngramHistory) -> Self {
        Self { cache, history }
    }
}
