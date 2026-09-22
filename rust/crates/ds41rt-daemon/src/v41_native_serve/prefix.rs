use super::speculative::DraftChain;
use super::*;
use crate::v41_backbone_cache::{BackbonePrefix, CacheLease};
use crate::v41_requests::RequestPrefix;
use speculative::DraftPrefix;
mod images;
pub(super) use images::ImageKeys;
use images::ImageKeySpace;
pub(super) use ds41rt_core::prefix::{Retention, SnapshotKind};
mod host_cache;
pub(super) use host_cache::HostCacheBinding;
struct Saved<'a> {
    _images: ImageKeys,
    target: RequestPrefix<'a>,
    draft: Option<DraftPrefix<'a>>,
    next: TokenScores,
    /// The host cache's write-behind copy of this snapshot, if one was issued.
    ticket: Option<ds41rt_hostcache::cache::StoreTicket>,
}
pub(super) struct PrefixCache<'a> {
    retained: Retention<Saved<'a>>,
    images: ImageKeySpace,
    pending: [Option<PendingRetention>; 2],
    host: Option<HostCacheBinding<'a>>,
}
struct PendingRetention {
    kind: SnapshotKind,
    keys: Vec<u32>,
    images: ImageKeys,
    next: TokenScores,
    id: u64,
    lease: CacheLease,
    draft: bool,
}
impl<'a> PrefixCache<'a> {
    pub fn new(limit: usize) -> Self {
        Self {
            retained: Retention::new(limit),
            images: ImageKeySpace::default(),
            pending: [None, None],
            host: None,
        }
    }
    /// Attach the host snapshot cache (`None` keeps every path exactly as before).
    pub fn with_host_cache(mut self, host: Option<HostCacheBinding<'a>>) -> Self {
        self.host = host;
        self
    }
    /// Poll the host cache's copies; called once per scheduler step.
    pub fn tick(&mut self) {
        if let Some(host) = &mut self.host {
            host.tick();
        }
    }
    pub fn host_metrics(&self) -> Option<ds41rt_hostcache::metrics::Snapshot> {
        self.host.as_ref().map(HostCacheBinding::metrics)
    }
    /// Whether the completed-turn bank will actually store a frontier.
    ///
    /// This is the same `bank.limit()` guard `retain`/`queue_retain` apply
    /// (`prefix.rs:177`, `:206`): a zero limit disables retention entirely, so a
    /// finishing request must not pay the one-row frontier D2H (design §10.4).
    /// The scheduler consults this **before** scheduling a retention download, so
    /// a cache-disabled deployment downloads no frontier row at all.
    pub fn turn_bank_enabled(&self) -> bool {
        self.retained.bank(SnapshotKind::Turn).limit() > 0
    }
    /// The host cache's effective configuration, exported with the metrics.
    pub fn host_config(&self) -> Option<&ds41rt_hostcache::config::Config> {
        self.host.as_ref().map(HostCacheBinding::config)
    }
    /// Once per prefill chunk (packet HC-9): observe the store stream (`tick`) first — a
    /// completion is otherwise only seen at the scheduler loop's `tick`, which a synchronous
    /// prefill blocks for its whole duration — then the bounded pacing hold that limits how
    /// long the oldest pending store copy may stay outstanding. The observation runs on
    /// every prefill chunk regardless of `store_pace_ns`; the pacing hold itself is a no-op
    /// and moves no hold metric when `store_pace_ns` is 0 (store metrics may move, because
    /// the observation commits completions). No-op without a host cache; the full contract
    /// lives at `ds41rt_hostcache::HostCache::prefill_hold`.
    pub fn prefill_hold(&mut self) -> anyhow::Result<()> {
        match &mut self.host {
            Some(host) => {
                host.tick();
                host.prefill_hold()
            }
            None => Ok(()),
        }
    }
    /// Replace a same-key snapshot or evict the oldest when the bank is full, before another
    /// arena slot is taken; every dropped snapshot passes through the host cache first.
    fn make_bank_room(&mut self, kind: SnapshotKind, keys: &[u32]) {
        let bank = self.retained.bank_mut(kind);
        let dropped = match bank.remove_exact(keys) {
            Some(replaced) => Some(replaced),
            None if bank.entries() >= bank.limit() => bank.evict_oldest(),
            None => None,
        };
        if let Some(saved) = dropped {
            self.host_dropped(saved);
        }
    }
    /// Issue the write-behind copy, then insert; an insertion-time eviction also passes through
    /// the host cache.
    fn insert_saved(&mut self, kind: SnapshotKind, keys: &[u32], mut saved: Saved<'a>, requests: &Requests<'a>) {
        if let Some(host) = &mut self.host {
            saved.ticket = match host.store(kind, keys, &saved, requests) {
                Ok(ticket) => ticket,
                Err(error) => {
                    tracing::warn!(target: "ds41rt::host_cache", %error, "snapshot not stored in the host cache");
                    None
                }
            };
        }
        if let Some(evicted) = self.retained.bank_mut(kind).insert(keys, saved) {
            self.host_dropped(evicted);
        }
    }
    /// A snapshot leaves the device: let its host copy finish within budget, then drop it.
    fn host_dropped(&mut self, saved: Saved<'a>) {
        if let Some(host) = &mut self.host {
            host.before_evict(saved.ticket);
        }
        drop(saved);
    }
    /// On a device-bank miss, rebuild the best host snapshot into the bank so the engine's own
    /// restore finds it. The snapshot needs device pages exactly as a prefill of the same tokens
    /// would, so room is made the same way first: the oldest retained device snapshots are
    /// evicted (through the host cache) until the pool can take it.
    fn host_restore<C: speculative::DraftChain<'a>>(
        &mut self,
        keys: &[u32],
        lease: CacheLease,
        requests: &Requests<'a>,
        draft: Option<&DraftRuntime<'_, 'a, C>>,
    ) -> Result<()> {
        let Some(host) = &mut self.host else {
            return Ok(());
        };
        let Some(hit) = host.lookup(keys) else {
            return Ok(());
        };
        let tokens = host.snapshot_tokens(hit.key).context("host snapshot has no tokens")?;
        if let Err(error) = self.make_room(requests, &[(lease, tokens.len() as u32)]) {
            if error.downcast_ref::<crate::v41_compressor::SourcePoolExhausted>().is_none() {
                return Err(error);
            }
            // Nothing left to evict and the pool still cannot hold the snapshot: the prefill
            // that follows faces the same pool, so this is the engine's pressure path, not a
            // cache fault.
            tracing::warn!(target: "ds41rt::host_cache", %error, "host restore abandoned: no device room; prefilling");
            self.host.as_mut().unwrap().count_abandoned_restore();
            return Ok(());
        }
        let host = self.host.as_mut().unwrap();
        // A restore that still cannot complete (a missing part, a copy failure) is abandoned and
        // counted; the request then prefills exactly as it would with no cache. The cache must
        // never turn a cache miss into a request error.
        let saved = match host.restore(&hit, requests, draft) {
            Ok(Some(saved)) => saved,
            Ok(None) => return Ok(()),
            Err(error) => {
                tracing::warn!(target: "ds41rt::host_cache", %error, "host restore abandoned; prefilling");
                host.count_abandoned_restore();
                return Ok(());
            }
        };
        if let Some(evicted) = self.retained.bank_mut(hit.kind).insert(&tokens, saved) {
            self.host_dropped(evicted);
        }
        Ok(())
    }
    pub fn prepare_key(&mut self, tokens: &[u32], images: &[ds41rt_loader::V41ImageSpan]) -> Result<ImageKeys> {
        self.images.prepare(tokens, images)
    }
    pub fn retain<C: DraftChain<'a>>(
        &mut self,
        kind: SnapshotKind,
        tokens: &[u32],
        images: &ImageKeys,
        next: &TokenScores,
        id: u64,
        lease: CacheLease,
        requests: &mut Requests<'a>,
        draft: Option<&mut DraftRuntime<'_, 'a, C>>,
    ) -> Result<()> {
        let bank = self.retained.bank_mut(kind);
        if bank.limit() == 0 {
            return Ok(());
        }
        let end = requests.cache().committed_end(lease)?;
        ensure!(
            end > 0 && end as usize <= tokens.len(),
            "retained token frontier differs"
        );
        let keys = images.encode(&tokens[..end as usize])?;
        // Evict before allocating another tail, keeping peak retained residency
        // within the configured number of completed states.
        self.make_bank_room(kind, &keys);
        let target = requests.retain_prefix(lease, BackbonePrefix::device_bytes())?;
        let draft = draft.map(|d| d.retain_prefix(id, end)).transpose()?;
        let saved = Saved {
            _images: images.through(end as usize),
            target,
            draft,
            next: next.clone(),
            ticket: None,
        };
        self.insert_saved(kind, &keys, saved, requests);
        Ok(())
    }
    pub fn queue_retain<C: DraftChain<'a>>(&mut self, lane: usize, kind: SnapshotKind, tokens: &[u32],
        images: &ImageKeys, next: &TokenScores, id: u64, lease: CacheLease,
        requests: &mut Requests<'a>, mut draft: Option<&mut DraftRuntime<'_, 'a, C>>) -> Result<bool> {
        ensure!(self.pending.get(lane).context("invalid retention lane")?.is_none(), "retention lane occupied");
        let bank = self.retained.bank_mut(kind);
        if bank.limit() == 0 { return Ok(false); }
        let end = requests.cache().committed_end(lease)?;
        ensure!(end > 0 && end as usize <= tokens.len(), "retained token frontier differs");
        let keys = images.encode(&tokens[..end as usize])?;
        self.make_bank_room(kind, &keys);
        requests.queue_prefix(lane, lease)?;
        if let Some(draft) = draft.as_deref_mut() {
            if let Err(error) = draft.queue_prefix(lane, id, end) {
                if let Err(cleanup) = requests.abort_prefix(lane) {
                    tracing::error!(%cleanup, "draining target snapshot after draft failure");
                }
                return Err(error);
            }
        }
        self.pending[lane] = Some(PendingRetention { kind, keys: keys.into_owned(), images: images.through(end as usize),
            next: next.clone(), id, lease, draft: draft.is_some() });
        Ok(true)
    }
    pub fn poll_retain<C: DraftChain<'a>>(&mut self, lane: usize, requests: &mut Requests<'a>,
        mut draft: Option<&mut DraftRuntime<'_, 'a, C>>) -> Result<bool> {
        let pending = self.pending.get(lane).and_then(Option::as_ref).context("retention is not pending")?;
        ensure!(pending.draft == draft.is_some(), "pending retention execution mode differs");
        if !requests.prefix_ready(lane, pending.lease)? { return Ok(false); }
        if let Some(draft) = draft.as_deref() {
            if !draft.prefix_ready(lane, pending.id)? { return Ok(false); }
        }
        let target = requests.finish_prefix(lane, pending.lease)?;
        let saved_draft = draft.as_deref_mut().map(|d| d.finish_prefix(lane, pending.id)).transpose()?;
        let pending = self.pending[lane].take().unwrap();
        // Another lane may have inserted while these copies ran. Radix insertion
        // enforces the bank limit again; two extra arena slots cover both pending copies.
        let saved = Saved { _images: pending.images, target, draft: saved_draft, next: pending.next, ticket: None };
        self.insert_saved(pending.kind, &pending.keys, saved, requests);
        Ok(true)
    }
    pub fn abort_retain<C: DraftChain<'a>>(&mut self, lane: usize, requests: &mut Requests<'a>,
        draft: Option<&mut DraftRuntime<'_, 'a, C>>) -> Result<()> {
        let target = requests.abort_prefix(lane);
        let speculative = draft.map(|d| d.abort_prefix(lane)).transpose();
        if let Some(pending) = self.pending.get_mut(lane) { *pending = None; }
        target.and(speculative.map(|_| ()))
    }
    pub fn restore<C: DraftChain<'a>>(
        &mut self,
        tokens: &[u32],
        images: &ImageKeys,
        id: u64,
        lease: CacheLease,
        requests: &mut Requests<'a>,
        draft: Option<&mut DraftRuntime<'_, 'a, C>>,
    ) -> Result<Option<(usize, Option<TokenScores>)>> {
        let keys = images.encode(tokens)?;
        if self.retained.lookup_reusable(&keys).is_none() {
            self.host_restore(&keys, lease, requests, draft.as_deref())?;
        }
        let Some((end, frontier, saved)) = self.retained.lookup_reusable(&keys) else {
            return Ok(None);
        };
        ensure!(
            saved.target.end() == frontier as u64 && saved.draft.is_some() == draft.is_some(),
            "retained execution mode or token frontier differs"
        );
        if end != frontier {
            let start =
                requests.restore_encoder_prefix(lease, &saved.target, end / 2 * 2, tokens)?;
            // Draft rings stay fresh until decoder replay seeds the final window.
            // The saved next token belongs to a different frontier and is unused.
            return Ok(Some((start, None)));
        }
        if tokens.len() - end >= 128 {
            requests.restore_encoder_continuation(lease, &saved.target, tokens.len() as u64)?;
            // Every final decoder/draft row comes from the new encoder suffix.
            return Ok(Some((end, None)));
        }
        requests.restore_prefix(lease, &saved.target)?;
        if let (Some(draft), Some(saved)) = (draft, saved.draft.as_ref()) {
            draft.restore_prefix(id, end as u64, saved)?;
        }
        Ok(Some((end, Some(saved.next.clone()))))
    }
    pub fn make_room(&mut self, requests: &Requests<'a>, work: &[(CacheLease, u32)]) -> Result<()> {
        loop {
            match requests.cache().check_append_capacity(work) {
                Ok(()) => return Ok(()),
                Err(error) => {
                    if error.downcast_ref::<crate::v41_compressor::SourcePoolExhausted>().is_none() {
                        return Err(error);
                    }
                    match self.retained.evict_oldest() {
                        Some((_, saved)) => self.host_dropped(saved),
                        None => return Err(error),
                    }
                }
            }
        }
    }
}
