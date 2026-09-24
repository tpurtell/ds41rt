//! One request lease spans every backbone window and compressed source.
use crate::v41_backbone_router::ExpertRow;
use crate::v41_compressor::{
    CompressorChunk, CompressorLease, CompressorState, CompressorWave, IndexProposal,
};
use crate::v41_index_selection::SelectionRequest;
use crate::v41_sparse_attention::AttentionRequest;
use crate::v41_window::{WindowChunk, WindowLease, WindowProposal, WindowState, WindowWave};
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::NativeLibrary;
use ds41rt_transport::ExpertV2SourceKind;
use crate::v41_memory::device::{Device, DeviceOwner};
use std::sync::atomic::{AtomicU64, Ordering};

mod ced;
mod commit_access;
pub(crate) use commit_access::CacheWave;
mod publication;
mod prefix;
mod placement;
pub(crate) mod peer_inputs;
pub(crate) use placement::CachePlacement;
pub(crate) use prefix::BackbonePrefix;
use ced::CachePhase;
pub(crate) use ced::CacheStage;

const SOURCES: [usize; 4] = [2, 8, 14, 20];
static NEXT_BATCH: AtomicU64 = AtomicU64::new(1);
static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CacheLease {
    owner: u64,
    slot: usize,
    generation: u64,
}
struct Request {
    id: u64,
    version: u64,
    end: u64,
    phase: CachePhase,
    publication: std::collections::VecDeque<publication::EncoderPublication>,
    windows: [WindowLease; 40],
    sources: [CompressorLease; 4],
}
#[derive(Clone, Copy)]
pub(crate) struct CacheWork {
    pub lease: CacheLease,
    pub tokens: u32,
    pub kind: ExpertV2SourceKind,
}
struct BatchRequest {
    work: CacheWork,
    id: u64,
    version: u64,
    position: u64,
    windows: [WindowLease; 40],
    sources: [CompressorLease; 4],
}
/// Owned metadata can survive producer execution. Every cache access/commit
/// revalidates it against the originating bank and live request versions.
pub(crate) struct CacheBatch {
    stage: CacheStage,
    replay_snapshot: Option<u64>,
    reserved: bool,
    identity: u64,
    owner: u64,
    requests: Vec<BatchRequest>,
}
impl CacheBatch {
    pub fn stage(&self) -> CacheStage { self.stage }
    pub fn is_reserved(&self) -> bool { self.reserved }
    pub fn identity(&self) -> u64 {
        self.identity
    }
    pub fn request_ids(&self) -> Vec<u64> {
        self.requests.iter().map(|request| request.id).collect()
    }
    pub fn positions(&self) -> Vec<u64> {
        self.requests
            .iter()
            .flat_map(|r| r.position..r.position + u64::from(r.work.tokens))
            .collect()
    }
    pub fn expert_rows(&self) -> Vec<ExpertRow> {
        self.requests
            .iter()
            .flat_map(|r| {
                (0..r.work.tokens).map(move |i| ExpertRow {
                    request_id: r.id,
                    position: r.position + u64::from(i),
                    kind: r.work.kind,
                })
            })
            .collect()
    }
    pub fn window_chunks(&self, layer: usize) -> Result<Vec<WindowChunk>> {
        ensure!(self.stage.windows().contains(&layer), "invalid batch window layer for phase");
        Ok(self
            .requests
            .iter()
            .map(|r| WindowChunk {
                lease: r.windows[layer],
                position: r.position,
                tokens: r.work.tokens,
            })
            .collect())
    }
    pub fn source_chunks(&self, layer: usize) -> Result<Vec<CompressorChunk>> {
        ensure!(self.stage.source_count() != 0, "decoder replay cannot produce global sources");
        let source = SOURCES
            .iter()
            .position(|&l| l == layer)
            .context("invalid batch source layer")?;
        Ok(self
            .requests
            .iter()
            .map(|r| CompressorChunk {
                lease: r.sources[source],
                position: r.position,
                tokens: r.work.tokens,
            })
            .collect())
    }
}

/// Borrows the bank and completed producers for the whole attention use. The
/// request vectors are built from one validated batch, never independently zipped
/// scheduler arrays. Drop all views before advancing or releasing cache history.
pub(crate) struct CacheAttention<'a> {
    windows: Vec<WindowProposal<'a>>,
    sources: Vec<IndexProposal<'a>>,
    positions: Vec<Vec<u64>>,
}
impl CacheAttention<'_> {
    pub fn attention_requests_fixed(&self)->Result<([AttentionRequest<'_>;16],usize)> {
        let count=self.windows.len();
        ensure!((1..=16).contains(&count) && self.positions.len()==count,
            "invalid fixed attention request count");
        Ok((std::array::from_fn(|i| {
            let i=if i<count {i} else {0};
            AttentionRequest { window:&self.windows[i],source:self.sources.get(i),positions:&self.positions[i] }
        }),count))
    }
    pub fn attention_requests(&self) -> Vec<AttentionRequest<'_>> {
        self.windows
            .iter()
            .enumerate()
            .map(|(i, window)| AttentionRequest {
                window,
                source: self.sources.get(i),
                positions: &self.positions[i],
            })
            .collect()
    }
    pub fn selection_requests(&self) -> Result<Vec<SelectionRequest<'_>>> {
        ensure!(
            self.sources.len() == self.windows.len(),
            "window-only layers have no index selection"
        );
        Ok(self
            .sources
            .iter()
            .zip(&self.positions)
            .map(|(proposal, positions)| SelectionRequest {
                proposal,
                positions,
            })
            .collect())
    }
}

pub(crate) struct BackboneCache<'a> {
    prefix_copies: [crate::v41_memory::SnapshotCopies<'a, (CacheLease, BackbonePrefix<'a>)>; 2],
    prefix_pool: Option<crate::v41_memory::SnapshotPool<'a>>,
    prefix_stream: crate::v41_memory::LoadStream<'a>,
    windows: Vec<DeviceOwner<'a, WindowState<'a>>>,
    sources: Vec<DeviceOwner<'a, CompressorState<'a>>>,
    requests: Vec<Option<Request>>,
    generations: Vec<u64>,
    owner: u64,
    poisoned: bool,
}
impl<'a> BackboneCache<'a> {
    /// DCP1 storage: replicate FP8 windows and FP4 source payloads, retaining
    /// index keys and compressor carry on their original devices. Token capacity
    /// remains source_pages; replica bytes must not be counted as extra tokens.
    pub fn replicated_device_bytes(placement:CachePlacement,slots:usize,
        source_pages:[usize;4])->Result<[usize;2]> {
        let mut bytes=Self::distributed_device_bytes(placement,slots,source_pages)?;
        for layer in 0..40 {
            let peer=1-placement.attention(layer)?;
            bytes[peer]=bytes[peer].checked_add(WindowState::device_bytes(layer,slots)?)
                .context("replicated window budget overflow")?;
        }
        for (i,gpu) in placement.sources().into_iter().enumerate() {
            bytes[1-gpu]=bytes[1-gpu].checked_add(
                crate::v41_compressor::SourceReplica::device_bytes(source_pages[i],slots)?)
                .context("replicated source budget overflow")?;
        }
        Ok(bytes)
    }
    /// All replica storage is allocated before returning a usable bank. Failure
    /// drops the partially built bank; admission cannot observe partial enablement.
    pub fn new_replicated(library:&'a NativeLibrary,placement:CachePlacement,slots:usize,
        source_pages:[usize;4],budgets:[usize;2])->Result<Self> {
        let needed=Self::replicated_device_bytes(placement,slots,source_pages)?;
        ensure!(needed.into_iter().zip(budgets).all(|(n,b)|n<=b),"replicated cache exceeds a device budget");
        let mut bank=Self::new_distributed(library,placement,slots,source_pages,budgets)?;
        for state in &mut bank.windows {
            let device=state.device;
            device.run(||state.enable_replica(Device { library,id:1-device.id }).map(|_|()))?;
        }
        for state in &mut bank.sources {
            let device=state.device;
            device.run(||state.enable_replica(Device { library,id:1-device.id }).map(|_|()))?;
        }
        Ok(bank)
    }
    /// Bind fresh producer workspaces to this bank's optional replica owners.
    /// Called separately for each independent lane, before any production.
    pub fn configure_producer_replicas(&self,windows:&mut [DeviceOwner<'a,WindowWave<'_, 'a>>],
        sources:&mut [DeviceOwner<'a,CompressorWave<'_, 'a>>])->Result<()> {
        ensure!(windows.len()==self.windows.len() && sources.len()==self.sources.len(),
            "replicated producer layer count differs");
        for (wave,state) in windows.iter_mut().zip(&self.windows) {
            if let Some(replica)=state.replica() {
                let device=wave.device;
                device.run(||wave.enable_replica(state,replica))?;
            }
        }
        for (wave,state) in sources.iter_mut().zip(&self.sources) {
            if let Some(replica)=state.replica() {
                let device=wave.device;
                device.run(||wave.enable_replica(state,replica))?;
            }
        }
        Ok(())
    }
    /// Explicit cache storage only; snapshots and CUDA state are budgeted separately.
    pub fn distributed_device_bytes(placement: CachePlacement, slots: usize,
        source_pages: [usize; 4]) -> Result<[usize; 2]> {
        let mut bytes = [0usize;2];
        for layer in 0..40 {
            let gpu = placement.attention(layer)?;
            bytes[gpu] = bytes[gpu].checked_add(WindowState::device_bytes(layer,slots)?)
                .context("distributed window budget overflow")?;
        }
        for ((layer,pages),gpu) in SOURCES.into_iter().zip(source_pages).zip(placement.sources()) {
            bytes[gpu] = bytes[gpu].checked_add(CompressorState::device_bytes(layer,slots,pages)?)
                .context("distributed source budget overflow")?;
        }
        Ok(bytes)
    }
    pub fn new_distributed(library: &'a NativeLibrary, placement: CachePlacement,
        slots: usize, source_pages: [usize;4], budgets: [usize;2]) -> Result<Self> {
        let bytes = Self::distributed_device_bytes(placement,slots,source_pages)?;
        ensure!(bytes.into_iter().zip(budgets).all(|(need,budget)| need <= budget),
            "distributed backbone cache exceeds a device budget");
        for id in 0..2 {
            let device = Device { library, id };
            device.run(|| library.cuda_enable_peer(1-id))?;
        }
        // Prefix-copy coordination remains on GPU0; per-source retained pages
        // stay owned by their source GPU. Snapshot pool placement is separate.
        Device { library, id: 0 }.run(|| Self::new_inner(library,slots,source_pages,Some(placement)))
    }
    pub fn attention_device(&self, layer: usize) -> Result<Device<'a>> {
        self.windows.get(layer).map(|window| window.device).context("invalid cache attention layer")
    }
    pub fn pages_for_context(slots: usize, context: usize) -> Result<[usize; 4]> {
        let mut pages = [0; 4];
        for (i, layer) in SOURCES.into_iter().enumerate() {
            pages[i] = CompressorState::pages_for_context(layer, slots, context)?;
        }
        Ok(pages)
    }

    pub fn device_bytes(slots: usize, source_pages: [usize; 4]) -> Result<usize> {
        let mut total = 40 * WindowState::device_bytes(0, slots)?;
        for (layer, pages) in SOURCES.into_iter().zip(source_pages) {
            total = total
                .checked_add(CompressorState::device_bytes(layer, slots, pages)?)
                .context("backbone cache budget overflow")?;
        }
        Ok(total)
    }
    pub fn new(
        library: &'a NativeLibrary,
        slots: usize,
        source_pages: [usize; 4],
        budget: usize,
    ) -> Result<Self> {
        ensure!(
            Self::device_bytes(slots, source_pages)? <= budget,
            "backbone cache exceeds budget"
        );
        Self::new_inner(library, slots, source_pages, None)
    }
    fn new_inner(library: &'a NativeLibrary, slots: usize, source_pages: [usize;4],
        placement: Option<CachePlacement>) -> Result<Self> {
        let original_device = library.cuda_get_device()?;
        let windows = (0..40)
            .map(|layer| {
                let id = match placement { Some(p) => p.attention(layer)? as i32, None => original_device };
                Device { library, id }.own(|| WindowState::new(
                    library,
                    layer,
                    slots,
                    WindowState::device_bytes(layer, slots)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let sources = SOURCES
            .into_iter()
            .zip(source_pages)
            .map(|(layer, pages)| {
                let id = match placement { Some(p) => p.attention(layer)? as i32, None => original_device };
                Device { library, id }.own(|| CompressorState::new(
                    library,
                    layer,
                    slots,
                    pages,
                    CompressorState::device_bytes(layer, slots, pages)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let owner = NEXT_OWNER
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| anyhow::anyhow!("backbone cache IDs exhausted"))?;
        Ok(Self {
            prefix_copies: [crate::v41_memory::SnapshotCopies::new(library)?,
                crate::v41_memory::SnapshotCopies::new(library)?],
            prefix_pool: None,
            prefix_stream: crate::v41_memory::LoadStream { library, raw: library.cuda_stream_create()? },
            windows,
            sources,
            requests: (0..slots).map(|_| None).collect(),
            generations: vec![0; slots],
            owner,
            poisoned: false,
        })
    }
    fn healthy(&self) -> Result<()> {
        ensure!(
            !self.poisoned,
            "backbone cache must be recreated after initialization or cleanup failure"
        );
        Ok(())
    }
    fn request_identity(&self, lease: CacheLease) -> Result<&Request> {
        self.healthy()?;
        ensure!(
            lease.owner == self.owner
                && lease.slot < self.requests.len()
                && self.generations[lease.slot] == lease.generation,
            "foreign or stale backbone cache lease"
        );
        self.requests[lease.slot]
            .as_ref()
            .context("backbone cache request released")
    }
    fn request(&self, lease: CacheLease) -> Result<&Request> {
        let request = self.request_identity(lease)?;
        ensure!(!self.prefix_copies.iter().any(|p| p.pending.as_ref().is_some_and(|(l, _)| *l == lease)),
            "backbone request has pending snapshot copies");
        Ok(request)
    }
    pub fn begin_request(&mut self, slot: usize, id: u64) -> Result<CacheLease> {
        self.healthy()?;
        ensure!(
            slot < self.requests.len() && self.requests[slot].is_none(),
            "backbone cache slot unavailable"
        );
        ensure!(
            !self.requests.iter().flatten().any(|r| r.id == id),
            "duplicate backbone cache request"
        );
        let generation = self.generations[slot]
            .checked_add(1)
            .context("cache generation exhausted")?;
        let mut windows = Vec::with_capacity(40);
        let mut sources = Vec::with_capacity(4);
        let acquired = (|| -> Result<()> {
            for state in &mut self.windows {
                windows.push(state.begin_request(slot, id)?);
            }
            for state in &mut self.sources {
                sources.push(state.begin_request(slot, id)?);
            }
            Ok(())
        })();
        if let Err(error) = acquired {
            // A failed device initialization can leave unknown component state.
            // Revoke all acquired leases and require reconstruction of the bank.
            self.poisoned = true;
            for (state, lease) in self.windows.iter_mut().zip(windows) {
                let _ = state.release(lease);
            }
            for (state, lease) in self.sources.iter_mut().zip(sources) {
                let _ = state.release(lease);
            }
            return Err(error);
        }
        self.generations[slot] = generation;
        self.requests[slot] = Some(Request {
            id,
            version: 0,
            end: 0,
            phase: CachePhase::Full,
            publication: std::collections::VecDeque::new(),
            windows: windows.try_into().ok().expect("40 windows"),
            sources: sources.try_into().ok().expect("four sources"),
        });
        Ok(CacheLease {
            owner: self.owner,
            slot,
            generation,
        })
    }
    pub fn request_id(&self, lease: CacheLease) -> Result<u64> {
        Ok(self.request_identity(lease)?.id)
    }
    pub fn committed_end(&self, lease: CacheLease) -> Result<u64> {
        let r = self.request(lease)?;
        for (layer, (state, &l)) in self.windows.iter().zip(&r.windows).enumerate() {
            ensure!(
                state.request_id(l)? == r.id && state.end(l)? == r.publication.iter().rev().find(|p| p.windows & (1u64 << layer) != 0)
                    .map_or(r.phase.window_end(layer, r.end), |p| p.end),
                "window request history differs"
            );
        }
        for (i, (state, &l)) in self.sources.iter().zip(&r.sources).enumerate() {
            ensure!(
                state.request_id(l)? == r.id && state.committed_end(l)? == r.publication.iter().rev()
                    .find(|p| p.sources & (1 << i) != 0).map_or(r.end, |p| p.end),
                "source request history differs"
            );
        }
        Ok(r.end)
    }
    pub fn plan(&self, work: &[CacheWork]) -> Result<CacheBatch> {
        self.plan_stage(work, false, false)
    }
    pub fn stage(&self, lease: CacheLease) -> Result<CacheStage> {
        Ok(self.request(lease)?.phase.stage())
    }
    /// Engram advances through encoder replay, while global source ownership
    /// remains at the cached prefix end. Decoder replay has no Engram work.
    pub fn history_end(&self, lease: CacheLease) -> Result<u64> {
        let end = self.committed_end(lease)?;
        Ok(match self.request(lease)?.phase {
            CachePhase::EncoderReplay { end, .. } => end,
            _ => end,
        })
    }
    pub fn check_append_capacity(&self, work: &[(CacheLease, u32)]) -> Result<()> {
        for (i, source) in self.sources.iter().enumerate() {
            let appends = work.iter().map(|&(lease, tokens)|
                Ok((self.request(lease)?.sources[i], tokens))).collect::<Result<Vec<_>>>()?;
            source.check_append_capacity(&appends)?;
        }
        Ok(())
    }
    pub fn plan_replay(&self, work: &[CacheWork]) -> Result<CacheBatch> {
        self.plan_stage(work, true, false)
    }
    fn plan_stage(&self, work: &[CacheWork], replay: bool, reserve: bool) -> Result<CacheBatch> {
        self.healthy()?;
        ensure!(
            !work.is_empty() && work.len() <= 16,
            "invalid cache batch request count"
        );
        let mut stage = None;
        let mut rows = 0usize;
        let mut requests = Vec::with_capacity(work.len());
        for (i, &item) in work.iter().enumerate() {
            ensure!(
                item.tokens > 0 && !work[..i].iter().any(|w| w.lease == item.lease),
                "empty or duplicate cache batch request"
            );
            rows = rows
                .checked_add(item.tokens as usize)
                .context("cache batch rows overflow")?;
            ensure!(rows <= 4096, "cache batch exceeds lane capacity");
            self.committed_end(item.lease)?;
            let live = self.request(item.lease)?;
            let current = live.phase.stage();
            if reserve {
                ensure!(current == CacheStage::Encoder && live.publication.len() < 16
                    && live.publication.iter().all(|p| p.reserved),
                    "encoder reservation capacity or phase differs");
            } else {
                ensure!(live.publication.is_empty(), "encoder chunk publication is still pending");
            }
            ensure!(current == CacheStage::Full || item.kind == ExpertV2SourceKind::Prefill,
                "CED phase requires prefill work");
            ensure!((current == CacheStage::Replay) == replay, "wrong cache planning phase");
            ensure!(stage.is_none_or(|s| s == current), "mixed cache phases in batch");
            stage = Some(current);
            let position = if reserve { live.publication.back().map_or(live.end, |p| p.end) }
                else { live.phase.position(live.end) };
            live.phase.validate_tokens(position, item.tokens)?;
            ensure!(
                position
                    .checked_add(u64::from(item.tokens))
                    .is_some_and(|end| end <= 1048576),
                "cache batch exceeds model context"
            );
            let r = self.request(item.lease)?;
            requests.push(BatchRequest {
                work: item,
                id: r.id,
                version: r.version,
                position,
                windows: r.windows,
                sources: r.sources,
            });
        }
        let identity = NEXT_BATCH
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| anyhow::anyhow!("cache batch IDs exhausted"))?;
        Ok(CacheBatch {
            reserved: reserve,
            stage: stage.context("empty cache batch")?,
            replay_snapshot: if stage != Some(CacheStage::Full) {
                Some(crate::v41_compressor::reserve_source_snapshot()?)
            } else {
                None
            },
            identity,
            owner: self.owner,
            requests,
        })
    }
    pub fn validate_batch(&self, batch: &CacheBatch) -> Result<()> {
        self.healthy()?;
        ensure!(batch.owner == self.owner, "foreign backbone cache batch");
        for r in &batch.requests {
            self.committed_end(r.work.lease)?;
            let live = self.request(r.work.lease)?;
            let position_valid = if batch.reserved {
                live.publication.iter().any(|p| p.reserved && p.batch == batch.identity
                    && p.start == r.position && p.end == r.position + u64::from(r.work.tokens))
            } else {
                live.publication.front().is_none_or(|p| !p.reserved && p.batch == batch.identity)
                    && live.version == r.version && live.phase.position(live.end) == r.position
            };
            ensure!(live.id == r.id && live.phase.stage() == batch.stage && position_valid,
                "stale backbone cache batch");
        }
        Ok(())
    }
    pub fn window(&self, batch: &CacheBatch, layer: usize) -> Result<&WindowState<'a>> {
        self.validate_batch(batch)?;
        ensure!(batch.stage.windows().contains(&layer), "window outside cache phase");
        self.windows
            .get(layer)
            .map(DeviceOwner::get)
            .context("invalid backbone window layer")
    }
    pub fn source(&self, batch: &CacheBatch, layer: usize) -> Result<&CompressorState<'a>> {
        self.validate_batch(batch)?;
        let source = SOURCES
            .iter()
            .position(|&l| l == layer)
            .context("invalid backbone source layer")?;
        Ok(&self.sources[source])
    }
    /// # Safety
    /// Query rows correspond to this admitted batch and all query writes have
    /// completed. No external writes race query, producer or persistent cache.
    pub unsafe fn produce_window(
        &self,
        batch: &CacheBatch,
        query: &crate::v41_attention_query::AttentionQueryOutput<'_>,
        wave: &mut WindowWave<'_, '_>,
    ) -> Result<()> {
        let state = self.window(batch, query.layer)?;
        let chunks = batch.window_chunks(query.layer)?;
        unsafe {
            wave.execute_query(state, &chunks, query)?;
        }
        Ok(())
    }
    /// # Safety
    /// Same query/batch association and completed-producer contract as
    /// produce_window. Only the four source layers may produce compressed KV.
    pub unsafe fn produce_source(
        &self,
        batch: &CacheBatch,
        query: &crate::v41_attention_query::AttentionQueryOutput<'_>,
        wave: &mut CompressorWave<'_, '_>,
    ) -> Result<()> {
        let state = self.source(batch, query.layer)?;
        let chunks = batch.source_chunks(query.layer)?;
        unsafe {
            wave.execute_query(state, &chunks, query)?;
        }
        Ok(())
    }
    /// Bind the matching layer's completed window and nearest compressed source
    /// to this batch. Later reindex layers continue using source 20's proposal.
    pub fn attention<'s>(
        &'s self,
        batch: &CacheBatch,
        layer: usize,
        window: &'s WindowWave<'_, '_>,
        source: Option<&'s CompressorWave<'_, '_>>,
    ) -> Result<CacheAttention<'s>> {
        let state = self.window(batch, layer)?;
        let chunks = batch.window_chunks(layer)?;
        window.validate_batch(state, &chunks)?;
        let source_layer = SOURCES.iter().copied().rev().find(|&n| n <= layer);
        let source_index = source_layer.and_then(|l| SOURCES.iter().position(|&s| s == l));
        let published_sources = self.publication_masks(batch)?.1;
        let committed_source = source_index
            .is_some_and(|i| batch.stage.reuses_sources() || published_sources & (1 << i) != 0);
        ensure!(
            source.is_some() == (source_layer.is_some() && !committed_source),
            "attention batch source presence differs"
        );
        let mut sources = Vec::new();
        if committed_source {
            let i = source_index.context("committed attention source absent")?;
            let state = self.source(batch, SOURCES[i])?;
            let snapshot = batch.replay_snapshot.context("missing committed source snapshot")?;
            sources = batch.requests.iter().map(|r| state.committed_proposal(
                r.sources[i], r.position..r.position + u64::from(r.work.tokens), snapshot))
                .collect::<Result<Vec<_>>>()?;
        } else if let Some(source_layer) = source_layer {
            let wave = source.context("attention source absent")?;
            let state = self.source(batch, source_layer)?;
            let chunks = batch.source_chunks(source_layer)?;
            wave.validate_batch(state, &chunks)?;
            sources = chunks.iter().map(|c| wave.index_proposal(state, c.lease))
                .collect::<Result<Vec<_>>>()?;
        }
        let windows = chunks
            .iter()
            .map(|c| window.proposal(state, c.lease))
            .collect::<Result<Vec<_>>>()?;
        let positions = chunks
            .iter()
            .map(|c| (c.position..c.position + u64::from(c.tokens)).collect())
            .collect();
        Ok(CacheAttention {
            windows,
            sources,
            positions,
        })
    }
    /// Revoke host leases first, then release every associated device cache. All
    /// proposal/attention consumers must have drained before this mutable call.
    pub fn release(&mut self, leases: &[CacheLease]) -> Result<()> {
        for (i, &lease) in leases.iter().enumerate() {
            self.ensure_releasable(lease)?;
            ensure!(!leases[..i].contains(&lease), "duplicate cache release");
        }
        let removed = leases
            .iter()
            .map(|l| self.requests[l.slot].take().unwrap())
            .collect::<Vec<_>>();
        let mut first_error = None;
        for r in removed {
            for (state, lease) in self.windows.iter_mut().zip(r.windows) {
                // Component commit failure may already have revoked this lease.
                if state.request_id(lease).is_ok() {
                    if let Err(e) = state.release(lease) {
                        first_error.get_or_insert(e);
                    }
                }
            }
            for (state, lease) in self.sources.iter_mut().zip(r.sources) {
                if state.request_id(lease).is_ok() {
                    if let Err(e) = state.release(lease) {
                        first_error.get_or_insert(e);
                    }
                }
            }
        }
        if let Some(e) = first_error {
            self.poisoned = true;
            return Err(e);
        }
        Ok(())
    }
    pub fn ensure_releasable(&self, lease: CacheLease) -> Result<()> {
        let request = self.request(lease)?;
        for (state, &window) in self.windows.iter().zip(&request.windows) {
            if state.request_id(window).is_ok() { state.ensure_not_writing(window)?; }
        }
        for (state, &source) in self.sources.iter().zip(&request.sources) {
            if state.request_id(source).is_ok() { state.ensure_not_writing(source)?; }
        }
        Ok(())
    }
    /// Check every component before starting or publishing accepted cache writes.
    pub(crate) fn validate_commit<'w,'wa:'w,'s,'sa:'s,W: CacheWave<WindowWave<'w,'wa>>,C: CacheWave<CompressorWave<'s,'sa>>>
        (&self, batch: &CacheBatch, windows: &[W], sources: &[C], accepted: &[u32]) -> Result<(u64, u8)> {
        self.validate_batch(batch)?;
        ensure!(
            windows.len() == 40 && sources.len() == 4 && accepted.len() == batch.requests.len(),
            "backbone commit owner/count differs"
        );
        for (r, &n) in batch.requests.iter().zip(accepted) {
            ensure!(
                n <= r.work.tokens,
                "backbone accepted prefix exceeds proposal"
            );
            r.version
                .checked_add(1)
                .context("backbone cache version exhausted")?;
        }
        let (published_windows, published_sources) = self.publication_masks(batch)?;
        if batch.reserved {
            ensure!(published_windows == (1u64 << 20) - 1 && published_sources == 15,
                "reserved encoder completion requires all KV published");
            for r in &batch.requests {
                let live = self.request(r.work.lease)?;
                ensure!(live.publication.front().is_some_and(|p| p.batch == batch.identity)
                    && live.end == r.position, "encoder chunks must complete in order");
                live.version.checked_add(1).context("cache version exhausted")?;
            }
        }
        if published_windows != 0 || published_sources != 0 {
            ensure!(batch.requests.iter().zip(accepted).all(|(r, &n)| n == r.work.tokens),
                "published encoder chunk requires full acceptance");
        }
        for layer in batch.stage.windows() {
            if published_windows & (1u64 << layer) != 0 { continue; }
            let wave = windows[layer].wave_ref();
            ensure!(wave.input().device_id == self.windows[layer].device.id,
                "window producer and cache GPU differ at layer {layer}");
            wave.validate_batch(&self.windows[layer], &batch.window_chunks(layer)?)?;
        }
        for i in 0..batch.stage.source_count() {
            if published_sources & (1 << i) != 0 { continue; }
            let wave = sources[i].wave_ref();
            ensure!(wave.input().device_id == self.sources[i].device.id,
                "compressed producer and cache GPU differ at source {}", SOURCES[i]);
            wave.validate_batch(&self.sources[i], &batch.source_chunks(SOURCES[i])?)?;
        }
        Ok((published_windows, published_sources))
    }
    /// # Safety
    /// The enclosing target pass retains bank and producer ownership until all
    /// writes finish. Abort/drain before releasing any participating request.
    pub unsafe fn enqueue_cache_commit<'w,'wa:'w,'s,'sa:'s,W: CacheWave<WindowWave<'w,'wa>>,C: CacheWave<CompressorWave<'s,'sa>>>
        (&self, batch: &CacheBatch, windows: &mut [W], sources: &mut [C], accepted: &[u32]) -> Result<()> {
        let (published, published_sources) = self.validate_commit(batch, windows, sources, accepted)?;
        for layer in batch.stage.windows() {
            // A batched multi-layer store may already have staged this window.
            if published & (1u64 << layer) == 0 && !windows[layer].wave_ref().has_pending_commit() {
                windows[layer].on_device_mut(|wave| unsafe { wave.enqueue_commit(&self.windows[layer], accepted) })?;
            }
        }
        for i in 0..batch.stage.source_count() {
            if published_sources & (1 << i) == 0 {
                sources[i].on_device_mut(|wave| unsafe { wave.enqueue_commit(&self.sources[i], accepted) })?;
            }
        }
        Ok(())
    }
    pub fn abort_cache_commit<'w,'wa:'w,'s,'sa:'s,W: CacheWave<WindowWave<'w,'wa>>,C: CacheWave<CompressorWave<'s,'sa>>>
        (&mut self, windows: &mut [W], sources: &mut [C]) -> Result<()> {
        let mut result = Ok(());
        for (state, wave) in self.windows.iter_mut().zip(windows) {
            if let Err(error) = wave.on_device_mut(|wave| wave.abort_commit(state)) { result = Err(error); }
        }
        for (state, wave) in self.sources.iter_mut().zip(sources) {
            if let Err(error) = wave.on_device_mut(|wave| wave.abort_commit(state)) { result = Err(error); }
        }
        result
    }
    /// Publish this phase's window/source owners. Queued windows must have
    /// completed; direct calls retain their synchronous path.
    /// Execution failure revokes all participants after draining queued writes.
    pub fn commit<'w,'wa:'w,'s,'sa:'s,W: CacheWave<WindowWave<'w,'wa>>,C: CacheWave<CompressorWave<'s,'sa>>>(
        &mut self,
        batch: &CacheBatch,
        windows: &mut [W],
        sources: &mut [C],
        accepted: &[u32],
    ) -> Result<()> {
        let (published_windows, published_sources) = self.validate_commit(batch, windows, sources, accepted)?;
        let committed = (|| -> Result<()> {
            for layer in batch.stage.windows() {
                if published_windows & (1u64 << layer) != 0 { continue; }
                windows[layer].on_device_mut(|wave| wave.commit(&mut self.windows[layer], accepted))?;
            }
            for i in 0..batch.stage.source_count() {
                if published_sources & (1 << i) != 0 { continue; }
                sources[i].on_device_mut(|wave| wave.commit(&mut self.sources[i], accepted))?;
            }
            Ok(())
        })();
        if let Err(error) = committed {
            if let Err(cleanup) = self.abort_cache_commit(windows, sources) {
                tracing::error!(%cleanup, "draining failed window writes");
            }
            let leases = batch
                .requests
                .iter()
                .map(|r| r.work.lease)
                .collect::<Vec<_>>();
            if let Err(cleanup) = self.release(&leases) {
                tracing::error!(%cleanup,"invalidating failed backbone cache transaction");
            }
            return Err(error);
        }
        for (r, &n) in batch.requests.iter().zip(accepted) {
            let live = self.requests[r.work.lease.slot]
                .as_mut()
                .expect("validated live request");
            live.phase.advance(&mut live.end, r.position + u64::from(n));
            live.version += 1;
            if !live.publication.is_empty() { live.publication.pop_front(); }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_bank_lifecycle_and_batch_isolation() -> Result<()> {
        let Some(path) = std::env::var_os("DS41RT_BACKBONE_CACHE_LIBRARY") else {
            eprintln!("skip GPU cache bank test: DS41RT_BACKBONE_CACHE_LIBRARY unset");
            return Ok(());
        };
        let library = unsafe { NativeLibrary::load(path)? };
        let pages = [1, 2, 3, 4];
        let budget = BackboneCache::device_bytes(16, pages)?;
        assert!(BackboneCache::device_bytes(0, pages).is_err());
        assert!(BackboneCache::device_bytes(17, pages).is_err());
        assert!(BackboneCache::device_bytes(16, [0, 1, 1, 1]).is_err());
        assert!(BackboneCache::device_bytes(16, [1, 1, 1, 65537]).is_ok());
        assert!(BackboneCache::device_bytes(16, [1, 1, 1, 262145]).is_err());
        assert!(BackboneCache::new(&library, 16, pages, budget - 1).is_err());
        let mut bank = BackboneCache::new(&library, 16, pages, budget)?;
        let mut other = BackboneCache::new(&library, 16, pages, budget)?;
        let foreign = other.begin_request(0, 100)?;
        for cycle in 0..2 {
            let leases = (0..16)
                .map(|slot| bank.begin_request(slot, 100 + slot as u64))
                .collect::<Result<Vec<_>>>()?;
            assert!(bank.begin_request(16, 999).is_err());
            assert!(bank.begin_request(0, 999).is_err());
            let work = leases
                .iter()
                .enumerate()
                .map(|(i, &lease)| CacheWork {
                    lease,
                    tokens: (i + 1) as u32,
                    kind: ExpertV2SourceKind::Prefill,
                })
                .collect::<Vec<_>>();
            let batch = bank.plan(&work)?;
            assert_ne!(batch.identity(), bank.plan(&work)?.identity());
            bank.validate_batch(&batch)?;
            assert!(other.validate_batch(&batch).is_err());
            assert!(bank.request_id(foreign).is_err());
            assert!(bank.plan(&[]).is_err());
            assert!(bank.plan(&[work[0], work[0]]).is_err());
            assert!(bank
                .plan(&[CacheWork {
                    tokens: 0,
                    ..work[0]
                }])
                .is_err());
            assert!(bank
                .plan(&[CacheWork {
                    tokens: 4097,
                    ..work[0]
                }])
                .is_err());
            assert!(bank
                .plan(&[
                    CacheWork {
                        tokens: 4096,
                        ..work[0]
                    },
                    work[1]
                ])
                .is_err());
            assert_eq!(
                bank.plan(&[CacheWork {
                    tokens: 4096,
                    ..work[0]
                }])?
                .positions()
                .len(),
                4096
            );
            let positions = work
                .iter()
                .flat_map(|w| 0..u64::from(w.tokens))
                .collect::<Vec<_>>();
            assert_eq!(batch.positions(), positions);
            let rows = batch.expert_rows();
            assert_eq!(rows.len(), 136);
            let mut offset = 0;
            for (i, w) in work.iter().enumerate() {
                assert_eq!(bank.request_id(w.lease)?, 100 + i as u64);
                assert_eq!(bank.committed_end(w.lease)?, 0);
                for row in &rows[offset..offset + w.tokens as usize] {
                    assert_eq!(row.request_id, 100 + i as u64);
                    assert_eq!(row.kind, ExpertV2SourceKind::Prefill);
                }
                offset += w.tokens as usize;
            }
            for layer in 0..40 {
                let state = bank.window(&batch, layer)?;
                for (i, chunk) in batch.window_chunks(layer)?.iter().enumerate() {
                    assert_eq!(state.request_id(chunk.lease)?, 100 + i as u64);
                    assert_eq!(state.end(chunk.lease)?, chunk.position);
                    assert_eq!(chunk.tokens, (i + 1) as u32);
                }
            }
            for layer in SOURCES {
                let state = bank.source(&batch, layer)?;
                for (i, chunk) in batch.source_chunks(layer)?.iter().enumerate() {
                    assert_eq!(state.request_id(chunk.lease)?, 100 + i as u64);
                    assert_eq!(state.committed_end(chunk.lease)?, chunk.position);
                    assert_eq!(chunk.tokens, (i + 1) as u32);
                }
            }
            assert!(batch.window_chunks(40).is_err());
            assert!(batch.source_chunks(21).is_err());
            assert!(bank.window(&batch, 40).is_err());
            assert!(bank.source(&batch, 21).is_err());
            // All release arguments must validate before any live slot changes.
            assert!(bank.release(&[leases[0], foreign]).is_err());
            assert!(bank.release(&[leases[0], leases[0]]).is_err());
            bank.validate_batch(&batch)?;
            // Commit preflight rejects missing producers without consuming history.
            assert!(bank.commit::<WindowWave<'_, '_>, CompressorWave<'_, '_>>(&batch, &mut [], &mut [], &[0; 16]).is_err());
            bank.validate_batch(&batch)?;
            bank.release(&leases[..1])?;
            assert!(bank.validate_batch(&batch).is_err());
            assert!(bank.window(&batch, 0).is_err());
            assert!(bank.source(&batch, 2).is_err());
            assert!(bank.begin_request(0, 101).is_err());
            let replacement = bank.begin_request(0, 100)?;
            assert_ne!(replacement, leases[0]);
            assert!(bank.request_id(leases[0]).is_err());
            assert!(bank.plan(&work).is_err());
            assert_eq!(bank.committed_end(replacement)?, 0);
            let surviving = bank.plan(&work[1..])?;
            bank.validate_batch(&surviving)?;
            bank.release(&leases[1..])?;
            bank.release(&[replacement])?;
            assert!(bank.validate_batch(&surviving).is_err());
            eprintln!("cache bank cycle {cycle}: 16 requests, all 44 component leases, batch/release/reuse guards passed");
        }
        other.release(&[foreign])?;
        Ok(())
    }
}

#[cfg(test)]
#[path = "v41_backbone_cache/commit_tests.rs"]
mod commit_tests;

#[cfg(test)]
mod context_geometry_tests {
    use super::BackboneCache;
    #[test]
    fn context_pool_provisions_each_slot_and_compression_ratio() {
        assert_eq!(BackboneCache::pages_for_context(16, 32768).unwrap(), [1024, 1024, 1024, 2048]);
        assert_eq!(BackboneCache::pages_for_context(3, 513).unwrap(), [6, 6, 6, 9]);
        assert_eq!(BackboneCache::pages_for_context(16, 1048576).unwrap(), [32768, 32768, 32768, 65536]);
        assert!(BackboneCache::pages_for_context(0, 32768).is_err());
        assert!(BackboneCache::pages_for_context(17, 32768).is_err());
        assert!(BackboneCache::pages_for_context(1, 0).is_err());
        assert!(BackboneCache::pages_for_context(1, 1048577).is_err());
    }
}

impl<'a> BackboneCache<'a> {
    pub fn owner(&self) -> u64 {
        self.owner
    }
    pub fn prefix_pool(&self) -> Option<&crate::v41_memory::SnapshotPool<'a>> {
        self.prefix_pool.as_ref()
    }
    pub fn prefix_library(&self) -> &'a NativeLibrary {
        self.prefix_stream.library
    }
    pub fn sources(&self) -> &[DeviceOwner<'a, CompressorState<'a>>] {
        &self.sources
    }
}
