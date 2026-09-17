//! The simulator (packet HC-4): a model of the engine around the cache, so the suites can drive
//! a cache exactly as the daemon will, on a virtual clock, deterministically.
//!
//! - **Device model**: a page pool per compressor with generations (a freed index becomes a new
//!   identity), copy-on-write prefix sharing between requests, retention banks using
//!   `ds41rt_core::prefix::Retention` with the engine's bank sizes, and the `make_room`
//!   eviction the engine performs under exhaustion. Pool exhaustion raises the modelled
//!   `SourcePoolExhausted`: the run aborts with an invariant failure, because a well-formed
//!   workload must fit its live lanes on the device. The map is dense — pool pages striped
//!   per compressor, then a tails arena (one slot per retained snapshot and lane), then draft
//!   rings — so `EngineModel::device_bytes` sizes the stub's flat fake device exactly and
//!   every address the simulator hands out is in bounds.
//! - **Requests**: each advances through admission → device-bank `lookup_reusable` → on miss
//!   [`CacheOps::lookup`] → [`CacheOps::restore`] into freshly reserved pages (or prefill at
//!   the modelled rate on miss/timeout) → decode → retire ([`CacheOps::store`]). Up to
//!   `EngineModel::lanes` requests run concurrently in lane slots; a workload is a FIFO queue
//!   of requests the slots pull.
//! - **Interleaver**: a seeded PRNG picks the next runnable action — one occupied lane's
//!   request step, or advancing the copy engine ([`CacheOps::tick`]) so asynchronous store
//!   completions interleave with lane steps; when no request is runnable the clock advances to
//!   the next admission. The schedule log holds one line per interleaver choice, plus one line
//!   per deterministic bookkeeping event (admissions, eviction consultations, clock advances,
//!   invariant violations), so a failure reproduces from its seed.
//! - **Workloads**: `AgentLoop` (sessions × turns with context growth and think time),
//!   `Burst` (many cold prompts), `Churn` (many sessions revisited round-robin, more live
//!   conversations than the device banks hold — the pressure case).
//!
//! The simulator asserts the invariants after every step: device pool accounting exact, pages
//! never freed while referenced, restore targets reproducing exactly what was stored (content
//! fidelity), and every store ticket settled exactly once — by `tick` or on the evict path.
//!
//! Two caches can be driven: the real [`HostCache`] over [`StubCopyEngine`] (the HC-6 suites
//! drive it through [`clocked::ClockedCache`], which advances the stub's clock one poll quantum
//! per `tick` so issued copies complete between polls) and [`testing::RecordingCache`], a
//! trivial cache for the HC-4 suites that records every call and never waits.
//!
//! Modelling notes: sessions own disjoint token streams (every token is hashed with the session
//! id), so a reusable hit — device or host — is always an exact ancestor of the request, and a
//! restored snapshot is re-inserted under the request's own prefix, as the engine re-inserts
//! the rebuilt `Saved`. Copy latency on the recording cache is counted in scheduler ticks, not
//! nanoseconds; restores and evict waits report zero nanoseconds, which is what "never waits"
//! means. `ds41rt_core::prefix::Retention` takes one limit for both banks (the engine's single
//! `--prefix-cache-entries` flag), so the model carries a single `retain` limit.
use crate::cache::{
    DevicePage, DeviceSnapshot, EvictDecision, HostCache, RestoreOutcome, RestoreTarget,
    StoreOutcome, StoreTicket, TickReport,
};
use crate::copy::{CopyEngine, DeviceRange, StubCopyEngine};
use crate::snapshot::{DevicePageId, Hit, Key, SnapshotMeta};
use crate::{
    SnapshotKind, COMPRESSORS, DRAFT_BYTES, KV_BYTES_PER_TOKEN, PAGE_BYTES, REPLAY_WINDOW_TOKENS,
    TAIL_BYTES,
};
use ds41rt_core::prefix::Retention;
use serde::Serialize;
use std::collections::{HashSet, VecDeque};

/// How many tokens one request prefills per scheduler step: the interleaving quantum.
pub const PREFILL_QUANTUM_TOKENS: usize = 2048;

/// A runaway schedule aborts here; well-formed workloads finish in far fewer steps.
const MAX_SCHEDULER_STEPS: u64 = 10_000_000;

/// Drain budget: ticks issued after the last request retires to settle store tickets.
const MAX_DRAIN_TICKS: u64 = 10_000;

/// Context growth per `Churn` visit, in tokens.
const CHURN_TURN_GROWTH_TOKENS: usize = 16;

/// A page's rows as the engine addresses them: packed index, index scales, KV values, KV scales
/// (64 B + 4 B + 288 B per row over `PAGE_ROWS` rows; the four lengths sum to [`PAGE_BYTES`]).
const PAGE_SEGMENTS: [usize; 4] = [64 * 256, 4 * 256, 144 * 256, 144 * 256];

/// The splitmix64 increment: the state advance between draws.
const SPLITMIX_INCREMENT: u64 = 0x9E37_79B9_7F4A_7C15;

/// The splitmix64 finalizer: a bijective, well-mixed 64-bit hash. The token stream, the
/// interleaver's PRNG and the tail-address derivation all share this one mixer, so a schedule
/// reproduces from its seed across every use.
fn splitmix64(z: u64) -> u64 {
    let z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The dense fake-device map: pool pages striped per compressor, then a tails arena (one slot
/// per retained snapshot and per lane — the most tails alive at once), then draft rings (the
/// model never addresses them, but the layout reserves their space so `EngineModel::device_bytes`
/// sizes the whole map). Every address the simulator hands out lies below
/// [`EngineModel::device_bytes`], so a suite can give the stub's flat fake device exactly that
/// many bytes.
struct Addressing {
    /// Pages per compressor pool; `Device::new` derives the same number from the model.
    pool_pages: usize,
    /// First byte of the tails arena, directly after the pages.
    tails_base: u64,
    /// Tail arena slots: `2 * retain + lanes`, every simultaneously live tail fits.
    tail_slots: u64,
}

impl Addressing {
    fn new(model: &EngineModel) -> Self {
        let pool_pages = model.pool_pages();
        let tails_base = (pool_pages * COMPRESSORS * PAGE_BYTES) as u64;
        Self {
            pool_pages,
            tails_base,
            tail_slots: (2 * model.retain + model.lanes).max(1) as u64,
        }
    }

    /// A deterministic device address for a pool slot and segment offset: the compressor bank at
    /// its true page stride, every bank `pool_pages` pages wide. `PAGE_BYTES` is not a power of
    /// two, so a stride-based layout (not bit packing) is what keeps every page's range disjoint
    /// from every other page's.
    fn page_addr(&self, id: DevicePageId, offset: usize) -> u64 {
        (id.compressor as u64 * self.pool_pages as u64 + id.page as u64) * PAGE_BYTES as u64
            + offset as u64
    }

    /// The modelled device address of a snapshot's backbone tail arena slot. The slot is
    /// token-hash derived: it lies in the tails arena, and the token fold keeps a restore target
    /// built for another snapshot's tokens from reproducing this one by construction. The arena
    /// holds one slot per simultaneously live tail; two live snapshots can share a slot, which
    /// is sound here because the suites check shapes and accounting, never the arena's bytes
    /// (the fake device carries zeros and the cache keys every part host-side).
    fn tail_addr(&self, tokens: &[u32]) -> u64 {
        let mut hash = tokens.len() as u64;
        for &token in tokens {
            hash = splitmix64(hash ^ u64::from(token));
        }
        self.tails_base + (hash % self.tail_slots) * TAIL_BYTES as u64
    }

    /// The modelled tail arena slot of the snapshot covering `tokens`: one range.
    fn tail_range(&self, tokens: &[u32]) -> Vec<DeviceRange> {
        vec![DeviceRange {
            addr: self.tail_addr(tokens),
            bytes: TAIL_BYTES,
        }]
    }

    /// The page's rows as the engine addresses them: packed index, index scales, KV values, KV
    /// scales (64 B + 4 B + 288 B per row over `PAGE_ROWS` rows; the four lengths sum to
    /// [`PAGE_BYTES`]).
    fn page_segments(&self, id: DevicePageId) -> Vec<DeviceRange> {
        let mut offset = 0;
        PAGE_SEGMENTS
            .iter()
            .map(|&bytes| {
                let range = DeviceRange {
                    addr: self.page_addr(id, offset),
                    bytes,
                };
                offset += bytes;
                range
            })
            .collect()
    }

    /// Group logically-ordered pages per compressor, preserving logical order within each bank.
    fn group_pages(&self, pages: &[DevicePageId]) -> [Vec<DevicePage>; COMPRESSORS] {
        let mut by_compressor: [Vec<DevicePage>; COMPRESSORS] = std::array::from_fn(|_| Vec::new());
        for &id in pages {
            by_compressor[id.compressor as usize].push(DevicePage {
                id,
                segments: self.page_segments(id),
            });
        }
        by_compressor
    }
}

/// Tokens of context occupy this many whole pages, rounded up.
fn pages_for(tokens: usize) -> usize {
    tokens
        .saturating_mul(KV_BYTES_PER_TOKEN)
        .div_ceil(PAGE_BYTES)
}

/// The engine's reuse rule mirrored from `ds41rt_core::prefix::Reusable::skipped`: an exact
/// match reuses everything; a partial match resumes at the even-aligned position one replay
/// window before the matched prefix.
fn reusable_tokens(common: usize, frontier: usize) -> usize {
    if common == frontier {
        common
    } else {
        (common / 2 * 2).saturating_sub(REPLAY_WINDOW_TOKENS)
    }
}

/// Key-space token `i` of session `s`: splitmix64 over the pair, so every session owns a
/// disjoint token stream and prefixes are stable across visits.
fn token_at(session: usize, i: usize) -> u32 {
    splitmix64(((session as u64) << 32 | i as u64).wrapping_add(SPLITMIX_INCREMENT)) as u32
}

fn token_prefix(session: usize, len: usize) -> Vec<u32> {
    (0..len).map(|i| token_at(session, i)).collect()
}

/// splitmix64: a tiny deterministic PRNG so a failing schedule reproduces from its seed.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(SPLITMIX_INCREMENT);
        splitmix64(self.0)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

/// The engine configuration the simulator models: the knobs a deployment sets.
/// [`EngineModel::default`] is the capacity-1024 configuration of record.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EngineModel {
    /// Device KV pool capacity in tokens, shared by in-flight requests and the retention
    /// banks; exhausting it is the modelled `SourcePoolExhausted`.
    pub device_pool_tokens: u64,
    /// Retention limit of each device bank, in entries. `ds41rt_core::prefix::Retention`
    /// takes one limit for both banks (the engine's single `--prefix-cache-entries` flag), so
    /// the model carries a single value.
    pub retain: usize,
    /// Modelled prefill throughput in tokens per second; prefill advances the virtual clock.
    pub prefill_tokens_per_s: f64,
    /// Modelled decode throughput in tokens per second; decode advances the virtual clock.
    pub decode_tokens_per_s: f64,
    /// Concurrent request lane slots; the workload's FIFO queue pulls through these.
    pub lanes: usize,
}

impl Default for EngineModel {
    /// The capacity-1024 configuration of record.
    fn default() -> Self {
        Self {
            device_pool_tokens: 2_500_000,
            retain: 24,
            prefill_tokens_per_s: 5_377.0,
            decode_tokens_per_s: 74.1,
            lanes: 8,
        }
    }
}

impl EngineModel {
    /// Pages in each compressor pool: the device model spreads the pool capacity over the
    /// four compressors at [`KV_BYTES_PER_TOKEN`] per token.
    fn pool_pages(&self) -> usize {
        (self.device_pool_tokens as usize).saturating_mul(KV_BYTES_PER_TOKEN)
            / PAGE_BYTES
            / COMPRESSORS
    }

    /// Bytes of the dense fake device the simulator addresses: the pool pages, then one tail
    /// arena slot per retained snapshot and lane (the most tails alive at once), then the same
    /// number of draft-ring slots. A suite sizes `StubCopyEngine::new(model, device_bytes,
    /// host_bytes)` with exactly this so every address the simulator hands out is in bounds.
    pub fn device_bytes(&self) -> usize {
        let pages = self.pool_pages() * COMPRESSORS * PAGE_BYTES;
        let slots = 2 * self.retain + self.lanes;
        pages + slots * (TAIL_BYTES + DRAFT_BYTES)
    }
}

/// A workload the simulator runs to completion: a FIFO queue of requests the lane slots pull.
/// Sessions own disjoint token streams (see the module doc), so every reusable match — device
/// or host — is an exact ancestor of the request.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub enum Workload {
    /// Sessions run their turns back to back, the context growing by `new_tokens_per_turn` and
    /// the session pausing `think_ns` between turns: the standing-conversation case.
    AgentLoop {
        /// Distinct conversations in the workload.
        sessions: usize,
        /// Turns each session runs.
        turns: usize,
        /// Context length at the first turn, in tokens.
        context_tokens: u32,
        /// Tokens each turn adds to the context.
        new_tokens_per_turn: u32,
        /// Pause after each turn but the last, in virtual nanoseconds.
        think_ns: u64,
    },
    /// Independent cold prompts: the all-miss case.
    Burst {
        /// Number of prompts.
        prompts: usize,
        /// Tokens per prompt.
        tokens: u32,
    },
    /// Sessions revisited round-robin (visit-major order), the context growing a little each
    /// visit: more live conversations than the device banks hold, so returning visits miss the
    /// device and are served by the host cache — the pressure case.
    Churn {
        /// Distinct conversations in the workload.
        sessions: usize,
        /// Visits each session receives.
        turns: usize,
        /// Context length at the first visit, in tokens.
        context_tokens: u32,
        /// Fraction of the lane slots the run uses, within `[0, 1]`.
        live_ratio: f64,
    },
}

/// What a run measured: every counter the engine would report, plus the seed and schedule log
/// that reproduce the run exactly.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct RunReport {
    /// The seed the interleaver ran from: rebuilding the simulator with it reproduces this
    /// report and the schedule log byte for byte.
    pub seed: u64,
    /// Interleaver choices taken: lane steps plus copy-engine advances.
    pub steps: u64,
    /// Tokens prefilled from cold across the run.
    pub prefilled_tokens: u64,
    /// Frontier tokens replayed from restored host snapshots across the run.
    pub restored_tokens: u64,
    /// Requests answered by a device-bank hit.
    pub device_hits: u64,
    /// Requests answered by a host-cache hit (a restore).
    pub host_hits: u64,
    /// Requests that found no reusable snapshot anywhere and prefilled from cold.
    pub misses: u64,
    /// One time-to-first-token sample per request, in virtual nanoseconds.
    pub ttft_ns: Vec<u64>,
    /// Invariant violations the run aborted on; empty exactly when the run was clean.
    pub invariant_failures: Vec<String>,
    /// One line per interleaver choice, plus deterministic bookkeeping lines; see the module
    /// doc for the format.
    pub schedule_log: Vec<String>,
}

/// The engine-facing surface the simulator drives: the five calls the engine's scheduler makes
/// ([`store`](CacheOps::store), [`tick`](CacheOps::tick),
/// [`before_device_evict`](CacheOps::before_device_evict), [`lookup`](CacheOps::lookup),
/// [`restore`](CacheOps::restore)) plus the page-free notification the engine raises on the
/// eviction path. Implementations must keep the `HostCache` invariants the simulator asserts: a
/// hit is a `Retention` hit, a restore target reproduces exactly what was stored (content
/// fidelity), and every store ticket is settled exactly once — by `tick` reporting it among
/// the completed or failed, or by `before_device_evict` resolving it on the evict path: a
/// [`EvictDecision::WaitedClean`] or [`EvictDecision::DroppedUncached`] decision settles the
/// ticket there, and a settled ticket must never be reported by a later `tick`.
pub trait CacheOps {
    /// The engine's host-side descriptor; handed back on a hit and dropped on eviction.
    type Payload;
    /// Plan and issue the store; the returned ticket, if any, is reported exactly once.
    fn store(&mut self, snapshot: &DeviceSnapshot, payload: Self::Payload) -> StoreOutcome;
    /// Poll the store stream; completed stores become lookup-visible.
    fn tick(&mut self) -> TickReport;
    /// The engine is about to drop a device snapshot; resolves its ticket, waiting within the
    /// copy budget (zero on caches that never wait).
    fn before_device_evict(&mut self, ticket: Option<StoreTicket>) -> EvictDecision;
    /// The engine's reuse rule over host snapshots; `None` falls through to prefill.
    fn lookup(&mut self, tokens: &[u32]) -> Option<Hit>;
    /// Copy every part into `target`, wait within the restore budget, unpin.
    fn restore(&mut self, key: Key, target: &RestoreTarget) -> RestoreOutcome;
    /// A device page was freed: its identity can no longer be shared from the device.
    fn device_page_freed(&mut self, id: DevicePageId);
}

impl<E: CopyEngine, P> CacheOps for HostCache<E, P> {
    type Payload = P;
    fn store(&mut self, snapshot: &DeviceSnapshot, payload: P) -> StoreOutcome {
        HostCache::store(self, snapshot, payload)
    }
    fn tick(&mut self) -> TickReport {
        HostCache::tick(self)
    }
    fn before_device_evict(&mut self, ticket: Option<StoreTicket>) -> EvictDecision {
        HostCache::before_device_evict(self, ticket)
    }
    fn lookup(&mut self, tokens: &[u32]) -> Option<Hit> {
        HostCache::lookup(self, tokens)
    }
    fn restore(&mut self, key: Key, target: &RestoreTarget) -> RestoreOutcome {
        HostCache::restore(self, key, target)
    }
    fn device_page_freed(&mut self, id: DevicePageId) {
        HostCache::device_page_freed(self, id)
    }
}

/// A snapshot as retained in a device bank: its pages in logical order and its store ticket
/// while the host copy is in flight.
#[derive(Clone)]
struct Retained {
    pages: Vec<DevicePageId>,
    ticket: Option<StoreTicket>,
}

struct CompressorPool {
    /// Generation per slot; increments on every free, so a reused index is a new identity.
    generation: Vec<u32>,
    refcount: Vec<u32>,
    /// Sum of `refcount`, maintained at every mutation; the per-step check verifies this
    /// against the device total, and `Device::audit` recomputes it from the array.
    slot_refs: usize,
    /// Indices with `refcount == 0`, ready for allocation.
    free: Vec<u32>,
    allocated: usize,
}

impl CompressorPool {
    fn new(pages: usize) -> Self {
        Self {
            generation: vec![0; pages],
            refcount: vec![0; pages],
            slot_refs: 0,
            free: (0..pages as u32).rev().collect(),
            allocated: 0,
        }
    }
    fn pages(&self) -> usize {
        self.generation.len()
    }
}

/// The device around the cache: per-compressor page pools with copy-on-write sharing and the
/// two retention banks. Invariants: a slot is free exactly while its refcount is zero; the sum
/// of slot refcounts equals the references banks and requests hold; `make_room` frees pages —
/// through [`CacheOps::device_page_freed`] — only as refcounts hit zero.
struct Device {
    pools: Vec<CompressorPool>,
    banks: Retention<Retained>,
    /// The dense fake-device map this device addresses pages and tails through.
    addressing: Addressing,
    /// Page references held by bank entries.
    bank_refs: usize,
    /// Page references held by in-flight requests.
    request_refs: usize,
    /// Sum of all slot refcounts; equals `bank_refs + request_refs`.
    refs_sum: usize,
}

impl Device {
    fn new(model: &EngineModel) -> Self {
        assert!(model.lanes >= 1, "EngineModel.lanes must be at least 1");
        let pages = model.pool_pages();
        assert!(pages >= 1, "device_pool_tokens yields an empty page pool");
        Self {
            pools: (0..COMPRESSORS)
                .map(|_| CompressorPool::new(pages))
                .collect(),
            banks: Retention::new(model.retain),
            addressing: Addressing::new(model),
            bank_refs: 0,
            request_refs: 0,
            refs_sum: 0,
        }
    }

    fn add_ref(&mut self, id: DevicePageId) -> Result<(), String> {
        let pool = &mut self.pools[id.compressor as usize];
        let slot = pool
            .refcount
            .get_mut(id.page as usize)
            .ok_or_else(|| format!("page reference out of range: {id:?}"))?;
        *slot = slot
            .checked_add(1)
            .ok_or_else(|| format!("refcount overflow on {id:?}"))?;
        pool.slot_refs += 1;
        self.refs_sum += 1;
        Ok(())
    }

    fn unref(&mut self, id: DevicePageId, cache: &mut dyn CacheOpsHandle) -> Result<(), String> {
        let pool = &mut self.pools[id.compressor as usize];
        let slot = pool
            .refcount
            .get_mut(id.page as usize)
            .ok_or_else(|| format!("free of unknown page: {id:?}"))?;
        *slot = slot
            .checked_sub(1)
            .ok_or_else(|| format!("refcount underflow freeing {id:?}"))?;
        pool.slot_refs -= 1;
        self.refs_sum -= 1;
        if *slot == 0 {
            if pool.generation[id.page as usize] != id.generation {
                return Err(format!("freed page generation drifted: {id:?}"));
            }
            pool.generation[id.page as usize] += 1;
            pool.free.push(id.page);
            pool.allocated -= 1;
            cache.device_page_freed(id);
        }
        Ok(())
    }

    /// Allocate `count` pages for logical indices `from..from + count` (striped across the
    /// compressors), evicting bank snapshots for room first.
    fn allocate(
        &mut self,
        from: usize,
        count: usize,
        cache: &mut dyn CacheOpsHandle,
        log: &mut dyn FnMut(String),
    ) -> Result<Vec<DevicePageId>, String> {
        self.make_room(from, count, cache, log)?;
        let mut out = Vec::with_capacity(count);
        for j in from..from + count {
            let c = j % COMPRESSORS;
            let pool = &mut self.pools[c];
            let Some(&idx) = pool.free.last() else {
                return Err(EXHAUSTED.to_string());
            };
            if pool.refcount[idx as usize] != 0 {
                return Err(format!(
                    "allocated page {idx} on compressor {c} was not free"
                ));
            }
            pool.free.pop();
            pool.refcount[idx as usize] = 1;
            pool.slot_refs += 1;
            pool.allocated += 1;
            self.refs_sum += 1;
            out.push(DevicePageId {
                compressor: c as u8,
                page: idx,
                generation: pool.generation[idx as usize],
            });
        }
        self.request_refs += count;
        Ok(out)
    }

    /// Free bank snapshots (prompts first, oldest access first) until `count` pages starting at
    /// logical index `from` can be striped across the pools.
    fn make_room(
        &mut self,
        from: usize,
        count: usize,
        cache: &mut dyn CacheOpsHandle,
        log: &mut dyn FnMut(String),
    ) -> Result<(), String> {
        let mut need = [0usize; COMPRESSORS];
        for j in from..from + count {
            need[j % COMPRESSORS] += 1;
        }
        while (0..COMPRESSORS).any(|c| self.pools[c].free.len() < need[c]) {
            match self.banks.evict_oldest() {
                Some((_kind, retained)) => self.release_entry(&retained, cache, log),
                None => return Err(EXHAUSTED.to_string()),
            }?;
        }
        Ok(())
    }

    /// The engine's eviction of a retained snapshot: consult the cache before dropping it, then
    /// free its pages, reporting each page whose count hits zero.
    fn release_entry(
        &mut self,
        retained: &Retained,
        cache: &mut dyn CacheOpsHandle,
        log: &mut dyn FnMut(String),
    ) -> Result<(), String> {
        let decision = cache.before_device_evict(retained.ticket);
        log(format!(
            "evict bank snapshot pages={} decision={decision:?}",
            retained.pages.len()
        ));
        self.bank_refs -= retained.pages.len();
        for &id in &retained.pages {
            self.unref(id, cache)?;
        }
        Ok(())
    }

    /// Insert a snapshot into a bank, taking a fresh page reference per page (the snapshot
    /// shares pages that stay alive elsewhere), and release whatever the bank overflows.
    fn insert_entry(
        &mut self,
        kind: SnapshotKind,
        tokens: &[u32],
        pages: Vec<DevicePageId>,
        ticket: Option<StoreTicket>,
        cache: &mut dyn CacheOpsHandle,
        log: &mut dyn FnMut(String),
    ) -> Result<(), String> {
        for &id in &pages {
            self.add_ref(id)?;
        }
        self.adopt_entry(kind, tokens, pages, ticket, cache, log)
    }

    /// Insert a snapshot into a bank that takes over references the request already holds (the
    /// retire path); no new references are taken.
    fn adopt_entry(
        &mut self,
        kind: SnapshotKind,
        tokens: &[u32],
        pages: Vec<DevicePageId>,
        ticket: Option<StoreTicket>,
        cache: &mut dyn CacheOpsHandle,
        log: &mut dyn FnMut(String),
    ) -> Result<(), String> {
        self.bank_refs += pages.len();
        let evicted = self
            .banks
            .bank_mut(kind)
            .insert(tokens, Retained { pages, ticket });
        if let Some(retained) = evicted {
            self.release_entry(&retained, cache, log)?;
        }
        Ok(())
    }

    /// Invariant check after every scheduler step, cheap enough to run per step: the per-pool
    /// reference sums (maintained at every mutation) must equal the device's running total,
    /// which must equal the references the banks and the requests declare. Together with
    /// [`Device::audit`]'s from-scratch recomputation these imply every page a bank entry or a
    /// request holds is live: a dead identity — freed, or re-generated under a stale one —
    /// would leave the slot sums, the free lists or the generation discipline inconsistent.
    fn check(&self) -> Result<(), String> {
        let mut slot_refs = 0usize;
        for (c, pool) in self.pools.iter().enumerate() {
            if pool.allocated + pool.free.len() != pool.pages() {
                return Err(format!(
                    "compressor {c} page accounting broken: {} allocated + {} free != {}",
                    pool.allocated,
                    pool.free.len(),
                    pool.pages()
                ));
            }
            slot_refs += pool.slot_refs;
        }
        if slot_refs != self.refs_sum {
            return Err(format!(
                "refcount drift: slots hold {slot_refs} refs, the device counted {}",
                self.refs_sum
            ));
        }
        if self.refs_sum != self.bank_refs + self.request_refs {
            return Err(format!(
                "refcount drift: slots hold {} refs, banks hold {} and requests {}",
                self.refs_sum, self.bank_refs, self.request_refs
            ));
        }
        Ok(())
    }

    /// The from-scratch version of [`Device::check`], run once when a simulation ends: sums
    /// the per-slot refcounts straight from the arrays and scans the free lists, proving the
    /// incremental counters the per-step check relies on still mirror the arrays. This is
    /// `O(pool pages)`; per scheduler step it would dominate the suites' debug runtime.
    fn audit(&self) -> Result<(), String> {
        let mut slot_refs = 0usize;
        for (c, pool) in self.pools.iter().enumerate() {
            slot_refs += pool.refcount.iter().map(|&rc| rc as usize).sum::<usize>();
            for &idx in &pool.free {
                let rc = pool.refcount[idx as usize];
                if rc != 0 {
                    return Err(format!(
                        "free slot {idx} on compressor {c} holds refcount {rc}"
                    ));
                }
            }
        }
        if slot_refs != self.refs_sum {
            return Err(format!(
                "refcount drift: slots hold {slot_refs} refs, the device counted {}",
                self.refs_sum
            ));
        }
        self.check()
    }
}

/// The modelled error of a device pool that cannot satisfy an allocation even after evicting
/// every bank snapshot.
const EXHAUSTED: &str =
    "device pool exhausted (SourcePoolExhausted): workload exceeds device capacity";

/// The object-safe view of `CacheOps` the device model needs; blanket-implemented so any cache
/// the simulator drives answers `make_room`.
trait CacheOpsHandle {
    fn device_page_freed(&mut self, id: DevicePageId);
    fn before_device_evict(&mut self, ticket: Option<StoreTicket>) -> EvictDecision;
}

impl<T: CacheOps> CacheOpsHandle for T {
    fn device_page_freed(&mut self, id: DevicePageId) {
        CacheOps::device_page_freed(self, id)
    }
    fn before_device_evict(&mut self, ticket: Option<StoreTicket>) -> EvictDecision {
        CacheOps::before_device_evict(self, ticket)
    }
}

/// The handle the simulator hands the device model: forwards the two eviction-path calls and
/// folds `before_device_evict` into the simulator's ticket ledger. `WaitedClean` and
/// `DroppedUncached` both settle the ticket on the evict path — the snapshot leaves the device
/// and the ticket must never come through `tick` — so the ledger drops it on either decision.
struct EvictHandle<'a, C: CacheOps> {
    cache: &'a mut C,
    outstanding: &'a mut HashSet<StoreTicket>,
}

impl<C: CacheOps> CacheOpsHandle for EvictHandle<'_, C> {
    fn device_page_freed(&mut self, id: DevicePageId) {
        self.cache.device_page_freed(id);
    }
    fn before_device_evict(&mut self, ticket: Option<StoreTicket>) -> EvictDecision {
        let decision = self.cache.before_device_evict(ticket);
        if matches!(
            decision,
            EvictDecision::WaitedClean { .. } | EvictDecision::DroppedUncached
        ) {
            if let Some(ticket) = ticket {
                self.outstanding.remove(&ticket);
            }
        }
        decision
    }
}

/// The eviction handle for one device-model call: the cache under test plus the run's
/// outstanding-ticket ledger.
fn evict_handle<'a, C: CacheOps>(
    cache: &'a mut C,
    outstanding: &'a mut HashSet<StoreTicket>,
) -> EvictHandle<'a, C> {
    EvictHandle { cache, outstanding }
}

/// A schedule-log sink for one device-model call.
fn log_sink(log: &mut Vec<String>) -> impl FnMut(String) + '_ {
    move |line| log.push(line)
}

/// One request of a workload, as queued for the lane slots.
#[derive(Clone)]
struct RequestPlan {
    session: usize,
    len: usize,
    decode: usize,
    /// Think time the session pauses after its previous request before this one admits.
    think_ns: u64,
    retire_kind: SnapshotKind,
}

impl RequestPlan {
    /// The workload as a FIFO request queue.
    ///
    /// `AgentLoop` orders session-major (each session's turns run back to back) and paces
    /// turns with `think_ns`; `Burst` is independent cold prompts; `Churn` orders visit-major
    /// — every live session's `v`-th visit before any `v+1` — so a session's consecutive
    /// visits are separated by every other live session's snapshots and the device banks (24 +
    /// 24 entries) cannot hold them all: returning visits miss the device and hit the host
    /// cache. `live_ratio` scales the lane count for the run.
    fn queue(workload: &Workload, lanes: usize) -> (Vec<Self>, usize) {
        match *workload {
            Workload::AgentLoop {
                sessions,
                turns,
                context_tokens,
                new_tokens_per_turn,
                think_ns,
            } => {
                let mut queue = Vec::new();
                for session in 0..sessions {
                    for turn in 0..turns.max(1) {
                        queue.push(RequestPlan {
                            session,
                            len: context_tokens as usize + turn * new_tokens_per_turn as usize,
                            decode: new_tokens_per_turn as usize,
                            // The session thinks after every turn but the last.
                            think_ns: if turn + 1 == turns.max(1) {
                                0
                            } else {
                                think_ns
                            },
                            retire_kind: SnapshotKind::Turn,
                        });
                    }
                }
                (queue, lanes)
            }
            Workload::Burst { prompts, tokens } => (
                (0..prompts)
                    .map(|session| RequestPlan {
                        session,
                        len: tokens as usize,
                        decode: 0,
                        think_ns: 0,
                        retire_kind: SnapshotKind::Prompt,
                    })
                    .collect(),
                lanes,
            ),
            Workload::Churn {
                sessions,
                turns,
                context_tokens,
                live_ratio,
            } => {
                assert!(
                    (0.0..=1.0).contains(&live_ratio),
                    "Churn.live_ratio must be within [0, 1]"
                );
                let mut queue = Vec::new();
                for visit in 0..turns.max(1) {
                    for session in 0..sessions {
                        queue.push(RequestPlan {
                            session,
                            len: context_tokens as usize + visit * CHURN_TURN_GROWTH_TOKENS,
                            decode: CHURN_TURN_GROWTH_TOKENS,
                            think_ns: 0,
                            retire_kind: SnapshotKind::Turn,
                        });
                    }
                }
                (
                    queue,
                    ((lanes as f64 * live_ratio).ceil() as usize).clamp(1, lanes),
                )
            }
        }
    }
}

/// A request's stage within its lane slot.
enum Stage {
    /// Device-bank `lookup_reusable`.
    Lookup,
    /// The device missed; consult the host cache.
    HostLookup,
    /// A host hit: reserve pages and restore.
    Restore {
        hit: Hit,
    },
    Prefill {
        remaining: usize,
    },
    Decode {
        remaining: usize,
    },
    /// Retain the completed turn and store it to the cache.
    Retire,
    /// The request retired; the scheduler frees the slot and paces the session.
    Finished,
}

/// One request in flight: its plan, its tokens, and the pages it holds (request-owned
/// references).
struct Request {
    plan: RequestPlan,
    tokens: Vec<u32>,
    pages: Vec<DevicePageId>,
    admit_ns: u64,
    ttft_recorded: bool,
    stage: Stage,
}

/// The simulator: an engine model around a cache, driven by a seeded interleaver on a virtual
/// clock. Generic over the cache through [`CacheOps`]; the default type argument is the real
/// facade over the stub copy engine, the HC-4 suites drive [`testing::RecordingCache`] and the
/// HC-6 suites drive the real facade through [`clocked::ClockedCache`].
pub struct Simulator<C: CacheOps = HostCache<StubCopyEngine, ()>> {
    model: EngineModel,
    device: Device,
    cache: C,
    seed: u64,
    rng: Rng,
    now: u64,
    steps: u64,
    schedule_log: Vec<String>,
}

impl<C> Simulator<C>
where
    C: CacheOps,
    C::Payload: Default,
{
    /// Build a simulator: the device model from `model`, the cache to drive, and the seed the
    /// interleaver and the schedule log reproduce from.
    pub fn new(model: EngineModel, cache: C, seed: u64) -> Self {
        Self {
            device: Device::new(&model),
            model,
            cache,
            seed,
            rng: Rng::new(seed),
            now: 0,
            steps: 0,
            schedule_log: Vec::new(),
        }
    }

    /// The cache under test; its recorded calls are the suite's observability.
    pub fn cache(&self) -> &C {
        &self.cache
    }

    /// The virtual clock in nanoseconds.
    pub fn now_ns(&self) -> u64 {
        self.now
    }

    /// Run a workload to completion, asserting the invariants after every step. A violation
    /// aborts the run and lands in `RunReport::invariant_failures`; with a well-formed workload
    /// that list is empty.
    ///
    /// The run must be monotonic in the key space: a request's tokens are never shorter than a
    /// snapshot retained by a previous run on this simulator, because a host hit is restored
    /// exactly and the retained snapshot must stay an ancestor of the request. Fresh
    /// simulators always qualify; so do phased workloads whose contexts only grow (the HC-6
    /// soak's phases). Re-running a workload whose contexts restart at a shorter prefix panics
    /// on the restore slice instead of returning a misleading report.
    pub fn run(&mut self, workload: &Workload) -> RunReport {
        let (queue, lanes) = RequestPlan::queue(workload, self.model.lanes);
        let sessions = queue
            .iter()
            .map(|plan| plan.session)
            .max()
            .map_or(0, |m| m + 1);
        let mut queue: VecDeque<RequestPlan> = queue.into();
        let mut session_ready = vec![0u64; sessions];
        let mut prompt_retained = vec![false; sessions];
        let mut slots: Vec<Option<Request>> = (0..lanes).map(|_| None).collect();
        let mut report = RunReport {
            seed: self.seed,
            ..RunReport::default()
        };
        let mut outstanding: HashSet<StoreTicket> = HashSet::new();
        self.steps = 0;

        while !(queue.is_empty() && slots.iter().all(Option::is_none)) {
            if self.steps >= MAX_SCHEDULER_STEPS {
                report
                    .invariant_failures
                    .push(format!("scheduler step cap {MAX_SCHEDULER_STEPS} reached"));
                break;
            }
            self.admit(&mut queue, &mut session_ready, &mut slots);
            let line = match occupied_slots(&slots) {
                occupied if occupied.is_empty() => {
                    // No request can run: wait for the front session's next admission. (A queued
                    // session is never blocked by itself: its in-flight predecessor occupies a
                    // slot, so `ready` here is always finite.) Forced, not an interleaver
                    // choice: no PRNG draw.
                    let next = queue
                        .front()
                        .map(|plan| session_ready[plan.session])
                        .filter(|&ready| ready > self.now && ready != u64::MAX)
                        .unwrap_or(self.now);
                    self.now = next;
                    format!("step={} advance now={next}", self.steps)
                }
                occupied => {
                    // The interleaver picks the next runnable action: one occupied lane's
                    // request step, or advancing the copy engine so asynchronous store
                    // completions interleave with the lanes.
                    let pick = self.rng.below(occupied.len() + 1);
                    if pick == occupied.len() {
                        self.tick(&mut report, &mut outstanding)
                    } else {
                        let idx = occupied[pick];
                        let result = match slots[idx].as_mut() {
                            Some(request) => self.step_request(
                                request,
                                &mut prompt_retained,
                                &mut report,
                                &mut outstanding,
                            ),
                            None => Err("interleaver picked an empty slot".to_string()),
                        };
                        let line = match result {
                            Ok(line) => line,
                            Err(failure) => {
                                report.invariant_failures.push(failure.clone());
                                self.log(format!(
                                    "step={} INVARIANT VIOLATION: {failure}",
                                    self.steps
                                ));
                                break;
                            }
                        };
                        // A finished request frees its slot; the session thinks before its
                        // next admission.
                        if let Some(request) =
                            slots[idx].take_if(|r| matches!(r.stage, Stage::Finished))
                        {
                            session_ready[request.plan.session] = self.now + request.plan.think_ns;
                        }
                        line
                    }
                }
            };
            self.log(line);
            self.steps += 1;
            report.steps = self.steps;
            if let Err(failure) = self.check(&slots) {
                report.invariant_failures.push(failure.clone());
                self.log(format!(
                    "step={} INVARIANT VIOLATION: {failure}",
                    self.steps
                ));
                break;
            }
            if !report.invariant_failures.is_empty() {
                break;
            }
        }

        // Settle store tickets that outlived their requests. Drain ticks are not interleaver
        // choices and are not logged.
        let mut drain = 0;
        while !outstanding.is_empty() && drain < MAX_DRAIN_TICKS {
            drain += 1;
            self.tick(&mut report, &mut outstanding);
        }
        if !outstanding.is_empty() {
            report.invariant_failures.push(format!(
                "{} store tickets never reported after {MAX_DRAIN_TICKS} ticks",
                outstanding.len()
            ));
        }
        if let Err(failure) = self.device.audit() {
            report.invariant_failures.push(failure);
        }
        report.schedule_log = std::mem::take(&mut self.schedule_log);
        report
    }

    /// Admit queued requests into free lane slots, in FIFO order, while their session's think
    /// time has passed.
    fn admit(
        &mut self,
        queue: &mut VecDeque<RequestPlan>,
        session_ready: &mut [u64],
        slots: &mut [Option<Request>],
    ) {
        while let Some(plan) = queue.front() {
            if session_ready[plan.session] > self.now {
                break;
            }
            let Some(slot) = slots.iter_mut().find(|slot| slot.is_none()) else {
                break;
            };
            let Some(plan) = queue.pop_front() else {
                break;
            };
            // The session is busy until this request retires.
            session_ready[plan.session] = u64::MAX;
            self.log(format!(
                "session={} admit tokens={} now={}",
                plan.session, plan.len, self.now
            ));
            *slot = Some(Request {
                tokens: token_prefix(plan.session, plan.len),
                plan,
                pages: Vec::new(),
                admit_ns: self.now,
                ttft_recorded: false,
                stage: Stage::Lookup,
            });
        }
    }

    fn log(&mut self, line: String) {
        self.schedule_log.push(line);
    }

    /// Advance the copy engine: one `CacheOps::tick`, folding the report into the outstanding
    /// set and flagging any ticket that was not issued or was reported twice. Returns the
    /// schedule-log line for this interleaver choice.
    fn tick(&mut self, report: &mut RunReport, outstanding: &mut HashSet<StoreTicket>) -> String {
        let tick = self.cache.tick();
        for ticket in tick.completed.iter().chain(tick.failed.iter()) {
            if !outstanding.remove(ticket) {
                let failure = format!("ticket {ticket:?} reported that was not outstanding");
                self.log(format!(
                    "step={} INVARIANT VIOLATION: {failure}",
                    self.steps
                ));
                report.invariant_failures.push(failure);
            }
        }
        format!(
            "tick completed={:?} failed={:?}",
            tick.completed, tick.failed
        )
    }

    /// Advance one request one stage; returns the schedule-log line for the choice.
    fn step_request(
        &mut self,
        request: &mut Request,
        prompt_retained: &mut [bool],
        report: &mut RunReport,
        outstanding: &mut HashSet<StoreTicket>,
    ) -> Result<String, String> {
        match &mut request.stage {
            Stage::Lookup => self.device_lookup(request, report, outstanding),
            Stage::HostLookup => self.host_lookup(request, report, outstanding),
            Stage::Restore { hit } => {
                let hit = hit.clone();
                self.restore(request, &hit, report, outstanding)
            }
            Stage::Prefill { remaining } => {
                let quantum = (*remaining).min(PREFILL_QUANTUM_TOKENS);
                self.now += self.ns_for(quantum as f64, self.model.prefill_tokens_per_s);
                *remaining -= quantum;
                report.prefilled_tokens += quantum as u64;
                let rest = *remaining;
                let line = format!("prefill tokens={quantum} remaining={rest} now={}", self.now);
                if rest == 0 {
                    self.finish_prefill(request, prompt_retained, report, outstanding)?;
                }
                Ok(line)
            }
            Stage::Decode { remaining } => {
                self.now += self.ns_for(1.0, self.model.decode_tokens_per_s);
                *remaining -= 1;
                let rest = *remaining;
                let line = format!("decode tokens=1 remaining={rest} now={}", self.now);
                if rest == 0 {
                    request.stage = Stage::Retire;
                }
                Ok(line)
            }
            Stage::Retire => self.retire(request, outstanding),
            Stage::Finished => Err("finished request stepped".to_string()),
        }
    }

    fn ns_for(&self, tokens: f64, rate: f64) -> u64 {
        (tokens / rate * 1e9).ceil() as u64
    }

    /// Allocate the pages a request still needs beyond the ones it already holds.
    fn allocate_rest(
        &mut self,
        request: &mut Request,
        outstanding: &mut HashSet<StoreTicket>,
    ) -> Result<Vec<DevicePageId>, String> {
        let total = pages_for(request.tokens.len());
        let mut handle = evict_handle(&mut self.cache, outstanding);
        self.device.allocate(
            request.pages.len(),
            total - request.pages.len(),
            &mut handle,
            &mut log_sink(&mut self.schedule_log),
        )
    }

    fn record_ttft(&self, request: &mut Request, report: &mut RunReport) {
        if !request.ttft_recorded {
            request.ttft_recorded = true;
            report
                .ttft_ns
                .push(self.now.saturating_sub(request.admit_ns));
        }
    }

    fn device_lookup(
        &mut self,
        request: &mut Request,
        report: &mut RunReport,
        outstanding: &mut HashSet<StoreTicket>,
    ) -> Result<String, String> {
        let found = self
            .device
            .banks
            .lookup_reusable(&request.tokens)
            .map(|(common, frontier, retained)| (common, frontier, retained.pages.clone()));
        match found {
            Some((common, frontier, pages)) => {
                report.device_hits += 1;
                let usable = reusable_tokens(common, frontier).min(request.tokens.len());
                let shared = pages_for(usable).min(pages.len());
                for &id in &pages[..shared] {
                    self.device.add_ref(id)?;
                }
                self.device.request_refs += shared;
                request.pages.extend_from_slice(&pages[..shared]);
                let rest = request.tokens.len() - usable;
                if rest > 0 {
                    let fresh = self.allocate_rest(request, outstanding)?;
                    request.pages.extend(fresh);
                    request.stage = Stage::Prefill { remaining: rest };
                } else {
                    self.record_ttft(request, report);
                    self.enter_post_prefill(request)?;
                }
                Ok(format!(
                    "device hit common={common} frontier={frontier} usable={usable} rest={rest}"
                ))
            }
            None => {
                request.stage = Stage::HostLookup;
                Ok("device miss".to_string())
            }
        }
    }

    fn host_lookup(
        &mut self,
        request: &mut Request,
        report: &mut RunReport,
        outstanding: &mut HashSet<StoreTicket>,
    ) -> Result<String, String> {
        match self.cache.lookup(&request.tokens) {
            Some(hit) => {
                report.host_hits += 1;
                request.stage = Stage::Restore { hit: hit.clone() };
                Ok(format!(
                    "host hit key={} common={} frontier={}",
                    hit.key, hit.common, hit.frontier
                ))
            }
            None => {
                report.misses += 1;
                let fresh = self.allocate_rest(request, outstanding)?;
                request.pages.extend(fresh);
                request.stage = Stage::Prefill {
                    remaining: request.tokens.len(),
                };
                Ok(format!(
                    "cold miss, prefill tokens={}",
                    request.tokens.len()
                ))
            }
        }
    }

    fn restore(
        &mut self,
        request: &mut Request,
        hit: &Hit,
        report: &mut RunReport,
        outstanding: &mut HashSet<StoreTicket>,
    ) -> Result<String, String> {
        let usable = reusable_tokens(hit.common, hit.frontier).min(request.tokens.len());
        let fresh = {
            let mut handle = evict_handle(&mut self.cache, &mut *outstanding);
            self.device.allocate(
                0,
                pages_for(hit.frontier),
                &mut handle,
                &mut log_sink(&mut self.schedule_log),
            )?
        };
        request.pages.extend_from_slice(&fresh);
        // The restore covers the hit's frontier tokens; a host hit is an exact ancestor of the
        // request, so that slice is the stored snapshot's token sequence.
        let target = restore_target(
            &self.device.addressing,
            &request.tokens[..hit.frontier],
            &request.pages,
        );
        match self.cache.restore(hit.key, &target) {
            RestoreOutcome::Done { ns, bytes } => {
                self.now += ns;
                report.restored_tokens += hit.frontier as u64;
                // The engine re-inserts the rebuilt snapshot into the device bank.
                let entry = request.pages[..pages_for(usable)].to_vec();
                let mut handle = evict_handle(&mut self.cache, &mut *outstanding);
                self.device.insert_entry(
                    hit.kind,
                    &request.tokens[..usable],
                    entry,
                    None,
                    &mut handle,
                    &mut log_sink(&mut self.schedule_log),
                )?;
                let rest = request.tokens.len() - usable;
                if rest > 0 {
                    let extra = self.allocate_rest(request, outstanding)?;
                    request.pages.extend(extra);
                    request.stage = Stage::Prefill { remaining: rest };
                } else {
                    self.record_ttft(request, report);
                    self.enter_post_prefill(request)?;
                }
                Ok(format!(
                    "restore done key={} bytes={bytes} ns={ns} usable={usable} rest={rest}",
                    hit.key
                ))
            }
            RestoreOutcome::TimedOut => {
                let restored = fresh.len();
                for id in fresh {
                    self.device.unref(id, &mut self.cache)?;
                }
                self.device.request_refs -= restored;
                request.pages.clear();
                let fresh = self.allocate_rest(request, outstanding)?;
                request.pages.extend(fresh);
                request.stage = Stage::Prefill {
                    remaining: request.tokens.len(),
                };
                Ok(format!(
                    "restore timed out key={}, prefilling cold",
                    hit.key
                ))
            }
            RestoreOutcome::Failed => Err(format!("restore failed for key {}", hit.key)),
        }
    }

    fn finish_prefill(
        &mut self,
        request: &mut Request,
        prompt_retained: &mut [bool],
        report: &mut RunReport,
        outstanding: &mut HashSet<StoreTicket>,
    ) -> Result<(), String> {
        // Prefill retains a prompt snapshot once per session.
        if request.plan.retire_kind == SnapshotKind::Turn && !prompt_retained[request.plan.session]
        {
            prompt_retained[request.plan.session] = true;
            let snapshot = build_snapshot(
                &self.device.addressing,
                SnapshotKind::Prompt,
                &request.tokens,
                &request.pages,
            );
            let outcome = self.cache.store(&snapshot, C::Payload::default());
            let ticket = track_ticket(outcome, outstanding);
            let mut handle = evict_handle(&mut self.cache, outstanding);
            self.device.insert_entry(
                SnapshotKind::Prompt,
                &request.tokens,
                request.pages.clone(),
                ticket,
                &mut handle,
                &mut log_sink(&mut self.schedule_log),
            )?;
        }
        self.record_ttft(request, report);
        self.enter_post_prefill(request)
    }

    fn enter_post_prefill(&mut self, request: &mut Request) -> Result<(), String> {
        request.stage = if request.plan.decode > 0 {
            Stage::Decode {
                remaining: request.plan.decode,
            }
        } else {
            Stage::Retire
        };
        Ok(())
    }

    fn retire(
        &mut self,
        request: &mut Request,
        outstanding: &mut HashSet<StoreTicket>,
    ) -> Result<String, String> {
        let kind = request.plan.retire_kind;
        let snapshot = build_snapshot(
            &self.device.addressing,
            kind,
            &request.tokens,
            &request.pages,
        );
        let outcome = self.cache.store(&snapshot, C::Payload::default());
        let ticket = track_ticket(outcome, outstanding);
        // The retained snapshot takes over the request's page references.
        let pages = std::mem::take(&mut request.pages);
        self.device.request_refs -= pages.len();
        let mut handle = evict_handle(&mut self.cache, outstanding);
        self.device.adopt_entry(
            kind,
            &request.tokens,
            pages,
            ticket,
            &mut handle,
            &mut log_sink(&mut self.schedule_log),
        )?;
        request.stage = Stage::Finished;
        Ok(format!(
            "retire session={} kind={kind:?} tokens={} now={}",
            request.plan.session,
            request.tokens.len(),
            self.now
        ))
    }

    /// The invariants asserted after every scheduler step: device accounting exact and every
    /// request holds only live pages.
    fn check(&self, slots: &[Option<Request>]) -> Result<(), String> {
        self.device.check()?;
        for request in slots.iter().flatten() {
            for &id in &request.pages {
                let pool = &self.device.pools[id.compressor as usize];
                let live = pool
                    .refcount
                    .get(id.page as usize)
                    .is_some_and(|&rc| rc >= 1)
                    && pool.generation[id.page as usize] == id.generation;
                if !live {
                    return Err(format!("request holds invalid page {id:?}"));
                }
            }
        }
        Ok(())
    }
}

fn occupied_slots(slots: &[Option<Request>]) -> Vec<usize> {
    slots
        .iter()
        .enumerate()
        .filter_map(|(i, slot)| slot.is_some().then_some(i))
        .collect()
}

/// Track a store outcome: an issued or deferred ticket lands in the outstanding ledger and
/// must be settled exactly once — by `tick` reporting it, or by `before_device_evict`
/// resolving it (`WaitedClean`/`DroppedUncached`). A skipped store has no ticket.
fn track_ticket(
    outcome: StoreOutcome,
    outstanding: &mut HashSet<StoreTicket>,
) -> Option<StoreTicket> {
    match outcome {
        StoreOutcome::Issued(t) | StoreOutcome::Deferred(t) => {
            outstanding.insert(t);
            Some(t)
        }
        StoreOutcome::Skipped(_) => None,
    }
}

/// The parts a modelled snapshot never carries: no dSpark draft rings, host-side scores.
fn empty_parts() -> (Option<Vec<DeviceRange>>, Vec<DeviceRange>) {
    (None, Vec::new())
}

/// The store/restore shape of a snapshot covering `tokens` tokens: striped pages of four
/// segments each, one snapshot-derived tail range, no draft, no scores (the engine layout
/// keeps scores host-side).
fn build_snapshot(
    addressing: &Addressing,
    kind: SnapshotKind,
    tokens: &[u32],
    pages: &[DevicePageId],
) -> DeviceSnapshot {
    let (draft, scores) = empty_parts();
    DeviceSnapshot {
        meta: SnapshotMeta {
            kind,
            tokens: tokens.to_vec(),
            end: tokens.len() as u32,
            has_draft: false,
        },
        pages: addressing.group_pages(pages),
        tail: addressing.tail_range(tokens),
        draft,
        scores,
    }
}

/// The restore destination for the first `tokens` of a request, in the same shape as the
/// stored snapshot.
fn restore_target(
    addressing: &Addressing,
    tokens: &[u32],
    pages: &[DevicePageId],
) -> RestoreTarget {
    let (draft, scores) = empty_parts();
    RestoreTarget {
        pages: addressing.group_pages(pages),
        tail: addressing.tail_range(tokens),
        draft,
        scores,
    }
}

/// A trivial cache for the suites on this branch: records every call, never waits. Store
/// copies land `completion_ticks` scheduler ticks after issue (each `tick` is one tick);
/// restores and evict waits complete immediately; every ticket is settled exactly once — by
/// `tick` reporting it, or by `before_device_evict` resolving it (`WaitedClean` on a pending
/// copy, `Clean` when there is nothing to wait for).
#[doc(hidden)]
pub mod testing {
    use super::{CacheOps, Hit};
    use crate::cache::{
        DevicePage, EvictDecision, RestoreOutcome, RestoreTarget, StoreOutcome, StoreTicket,
    };
    use crate::copy::DeviceRange;
    use crate::snapshot::{DevicePageId, Key};
    use crate::SnapshotKind;
    use ds41rt_core::prefix::Retention;
    use std::collections::{HashMap, VecDeque};

    /// Bytes carried by an iterator of device ranges.
    fn part_bytes(parts: impl Iterator<Item = DeviceRange>) -> u64 {
        parts.map(|range| range.bytes as u64).sum()
    }

    /// Bytes a stored snapshot or a restore target carries across all its parts.
    fn snapshot_bytes(
        pages: &[Vec<DevicePage>],
        tail: &[DeviceRange],
        draft: &Option<Vec<DeviceRange>>,
        scores: &[DeviceRange],
    ) -> u64 {
        part_bytes(
            pages
                .iter()
                .flat_map(|pages| pages.iter())
                .flat_map(|page| page.segments.iter().copied())
                .chain(tail.iter().copied())
                .chain(draft.iter().flatten().copied())
                .chain(scores.iter().copied()),
        )
    }

    /// The structural fingerprint of a snapshot's parts: per page the segment lengths in
    /// order (flattened across the compressor banks), then the tail, draft and scores range
    /// lists verbatim. A restore target must reproduce the stored snapshot's fingerprint
    /// exactly; tail addresses are snapshot-derived, so a target built for the wrong snapshot
    /// cannot alias them.
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct PartShape {
        pages: Vec<Vec<usize>>,
        tail: Vec<DeviceRange>,
        draft: Option<Vec<DeviceRange>>,
        scores: Vec<DeviceRange>,
    }

    /// The fingerprint of the given parts, in a canonical order both sides build.
    fn part_shape(
        pages: &[Vec<DevicePage>],
        tail: &[DeviceRange],
        draft: &Option<Vec<DeviceRange>>,
        scores: &[DeviceRange],
    ) -> PartShape {
        PartShape {
            pages: pages
                .iter()
                .flat_map(|bank| bank.iter())
                .map(|page| page.segments.iter().map(|segment| segment.bytes).collect())
                .collect(),
            tail: tail.to_vec(),
            draft: draft.clone(),
            scores: scores.to_vec(),
        }
    }

    struct PendingStore {
        ticket: StoreTicket,
        key: Key,
        tokens: Vec<u32>,
        kind: SnapshotKind,
        bytes: u64,
        shape: PartShape,
        ticks_remaining: u32,
    }

    /// A record of one completed store.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct RecordedStore {
        /// The key the stored snapshot was assigned.
        pub key: Key,
        /// Bytes the snapshot carried across all its parts.
        pub bytes: u64,
    }

    /// A record of one completed restore.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct RecordedRestore {
        /// The key of the restored snapshot.
        pub key: Key,
        /// Bytes the restore target carried across all its parts.
        pub bytes: u64,
    }

    /// The recording cache. The public fields are the suite's window into what the simulator
    /// did between runs.
    pub struct RecordingCache {
        /// Completed stores, in completion order.
        pub stores: Vec<RecordedStore>,
        /// Completed restores, in issue order.
        pub restores: Vec<RecordedRestore>,
        /// `before_device_evict` calls, with the ticket it was asked to resolve.
        pub evictions: Vec<Option<StoreTicket>>,
        /// Pages the simulator reported freed, in free order.
        pub freed_pages: Vec<DevicePageId>,
        /// Host lookups issued.
        pub lookups: u64,
        /// Host lookup hits returned.
        pub hits: u64,
        /// The most stores concurrently in flight at any point.
        pub max_pending: usize,
        next_key: Key,
        next_ticket: u64,
        completion_ticks: u32,
        pending: VecDeque<PendingStore>,
        outstanding: std::collections::HashSet<StoreTicket>,
        resident: Retention<Key>,
        kinds: HashMap<Key, SnapshotKind>,
        stored_bytes: HashMap<Key, u64>,
        shapes: HashMap<Key, PartShape>,
        restored_bytes: u64,
    }

    impl Default for RecordingCache {
        fn default() -> Self {
            Self::new()
        }
    }

    impl RecordingCache {
        /// A cache whose store copies complete after two scheduler ticks.
        pub fn new() -> Self {
            Self::with_completion_ticks(2)
        }
        /// A cache whose store copies complete after `completion_ticks` scheduler ticks (at
        /// least one: zero would report every store a full tick before it was issued).
        pub fn with_completion_ticks(completion_ticks: u32) -> Self {
            assert!(
                completion_ticks >= 1,
                "completion_ticks must be at least one tick"
            );
            Self {
                stores: Vec::new(),
                restores: Vec::new(),
                evictions: Vec::new(),
                freed_pages: Vec::new(),
                lookups: 0,
                hits: 0,
                max_pending: 0,
                next_key: 0,
                next_ticket: 0,
                completion_ticks,
                pending: VecDeque::new(),
                outstanding: std::collections::HashSet::new(),
                resident: Retention::new(usize::MAX),
                kinds: HashMap::new(),
                stored_bytes: HashMap::new(),
                shapes: HashMap::new(),
                restored_bytes: 0,
            }
        }
        /// Bytes stored across every completed store.
        pub fn bytes_stored(&self) -> u64 {
            self.stored_bytes.values().sum()
        }
        /// Bytes restored across every completed restore.
        pub fn bytes_restored(&self) -> u64 {
            self.restored_bytes
        }
        /// Resolve a ticket exactly once: complete it and make the snapshot visible.
        fn resolve(&mut self, pending: PendingStore) {
            if self.outstanding.remove(&pending.ticket) {
                self.resident
                    .bank_mut(pending.kind)
                    .insert(&pending.tokens, pending.key);
                self.kinds.insert(pending.key, pending.kind);
                self.stored_bytes.insert(pending.key, pending.bytes);
                self.shapes.insert(pending.key, pending.shape);
                self.stores.push(RecordedStore {
                    key: pending.key,
                    bytes: pending.bytes,
                });
            }
        }
    }

    impl CacheOps for RecordingCache {
        type Payload = ();
        fn store(&mut self, snapshot: &crate::cache::DeviceSnapshot, (): ()) -> StoreOutcome {
            let shape = part_shape(
                &snapshot.pages,
                &snapshot.tail,
                &snapshot.draft,
                &snapshot.scores,
            );
            let bytes = snapshot_bytes(
                &snapshot.pages,
                &snapshot.tail,
                &snapshot.draft,
                &snapshot.scores,
            );
            let key = self.next_key;
            self.next_key += 1;
            let ticket = StoreTicket(self.next_ticket);
            self.next_ticket += 1;
            self.outstanding.insert(ticket);
            self.pending.push_back(PendingStore {
                ticket,
                key,
                tokens: snapshot.meta.tokens.clone(),
                kind: snapshot.meta.kind,
                bytes,
                shape,
                ticks_remaining: self.completion_ticks,
            });
            self.max_pending = self.max_pending.max(self.pending.len());
            StoreOutcome::Issued(ticket)
        }
        fn tick(&mut self) -> crate::cache::TickReport {
            let mut completed = Vec::new();
            for mut pending in std::mem::take(&mut self.pending) {
                pending.ticks_remaining -= 1;
                if pending.ticks_remaining == 0 {
                    let ticket = pending.ticket;
                    self.resolve(pending);
                    completed.push(ticket);
                } else {
                    self.pending.push_back(pending);
                }
            }
            crate::cache::TickReport {
                completed,
                failed: Vec::new(),
            }
        }
        fn before_device_evict(&mut self, ticket: Option<StoreTicket>) -> EvictDecision {
            self.evictions.push(ticket);
            let pos =
                ticket.and_then(|wanted| self.pending.iter().position(|p| p.ticket == wanted));
            match pos.and_then(|pos| self.pending.remove(pos)) {
                Some(pending) => {
                    self.resolve(pending);
                    EvictDecision::WaitedClean { ns: 0 }
                }
                None => EvictDecision::Clean,
            }
        }
        fn lookup(&mut self, tokens: &[u32]) -> Option<Hit> {
            self.lookups += 1;
            let hit = self
                .resident
                .lookup_reusable(tokens)
                .and_then(|(common, frontier, &key)| {
                    self.kinds.get(&key).map(|&kind| Hit {
                        key,
                        kind,
                        common,
                        frontier,
                    })
                });
            if hit.is_some() {
                self.hits += 1;
            }
            hit
        }
        fn restore(&mut self, key: Key, target: &RestoreTarget) -> RestoreOutcome {
            // Content fidelity: a restore must ask for exactly the parts that were stored —
            // per page the segment lengths in order, and the tail, draft and scores lists.
            if self.shapes.get(&key)
                == Some(&part_shape(
                    &target.pages,
                    &target.tail,
                    &target.draft,
                    &target.scores,
                ))
            {
                let bytes =
                    snapshot_bytes(&target.pages, &target.tail, &target.draft, &target.scores);
                self.restored_bytes += bytes;
                self.restores.push(RecordedRestore { key, bytes });
                RestoreOutcome::Done { ns: 0, bytes }
            } else {
                RestoreOutcome::Failed
            }
        }
        fn device_page_freed(&mut self, id: DevicePageId) {
            self.freed_pages.push(id);
        }
    }
}

/// The HC-6 bridge from the simulator to the real facade: a [`HostCache`] over the
/// [`StubCopyEngine`] whose `tick` first advances the stub's clock by one scheduler poll
/// quantum. The simulator's virtual clock advances on prefill and decode, but the stub's copies
/// complete only when its own clock passes their completion time, and the simulator never
/// touches the engine directly — without this, stores issued after the last restore would stay
/// pending forever and the run's ticket drain would abort. One quantum per tick models the
/// daemon's poll cadence; restores and eviction waits advance the clock themselves, inside the
/// facade's bounded waits.
#[doc(hidden)]
pub mod clocked {
    use super::{CacheOps, Hit};
    use crate::cache::{
        DeviceSnapshot, EvictDecision, HostCache, RestoreOutcome, RestoreTarget, StoreOutcome,
        StoreTicket, TickReport,
    };
    use crate::config::{Config, StoreMode};
    use crate::copy::{CopyModel, StubCopyEngine};
    use crate::pool::Layout;
    use crate::snapshot::{DevicePageId, Key};
    use crate::KV_BYTES_PER_TOKEN;

    /// The HC-6 chunk granularity: 8 MiB chunks carve two tail slabs (2.6 MiB each) or 91 page
    /// slabs apiece, with little waste either way.
    pub const CHUNK_BYTES: u64 = 8 << 20;

    /// The daemon's config at `quota_bytes` of pinned host memory: on-retain stores, the
    /// documented budgets, 512-token minimum, both kinds.
    pub fn config(quota_bytes: u64) -> Config {
        Config {
            bytes: quota_bytes,
            chunk_bytes: CHUNK_BYTES,
            store: StoreMode::OnRetain,
            ..Config::default()
        }
    }

    /// A stub over the dense fake device the model addresses (`EngineModel::device_bytes`,
    /// so every range the simulator hands out is in bounds) with `host_bytes` of fake pinned
    /// host memory, on the design's copy model.
    pub fn engine(model: &super::EngineModel, host_bytes: u64) -> StubCopyEngine {
        StubCopyEngine::new(
            CopyModel::default(),
            model.device_bytes(),
            host_bytes as usize,
        )
    }

    /// One third of the live sessions' KV bytes: the soak's quota policy — under the
    /// serialized `AgentLoop` the working set is the few concurrent sessions, so this holds
    /// them comfortably while forcing eviction of the accumulated store stream. Note this is
    /// far too small to serve visit-major `Churn` revisits (see the concurrency suite's quota
    /// policy); the two workloads size their quotas differently on purpose.
    pub fn third_of_live(sessions: usize, context_tokens: usize) -> u64 {
        sessions as u64 * context_tokens as u64 * KV_BYTES_PER_TOKEN as u64 / 3
    }

    /// The modelled scheduler poll quantum: one `tick` advances the stub clock this many
    /// nanoseconds, so copies issued this quantum complete by the next poll (a store's copies
    /// cost ~150 µs on the default copy model).
    pub const POLL_QUANTUM_NS: u64 = 1_000_000;

    /// The real facade over the stub copy engine, driven by the simulator through [`CacheOps`].
    pub struct ClockedCache {
        cache: HostCache<StubCopyEngine, ()>,
    }

    impl ClockedCache {
        /// Build the facade: `config` and `engine` exactly as the daemon would, over the engine
        /// layout the simulator's snapshots are shaped for (`Layout::engine(0)`).
        pub fn new(config: Config, engine: StubCopyEngine) -> anyhow::Result<Self> {
            Ok(Self {
                cache: HostCache::new(config, Layout::engine(0), engine)?,
            })
        }

        /// The facade under test; its `metrics` are the suite's observability.
        pub fn cache(&self) -> &HostCache<StubCopyEngine, ()> {
            &self.cache
        }
    }

    impl CacheOps for ClockedCache {
        type Payload = ();
        fn store(&mut self, snapshot: &DeviceSnapshot, (): ()) -> StoreOutcome {
            self.cache.store(snapshot, ())
        }
        fn tick(&mut self) -> TickReport {
            self.cache.engine_mut().advance(POLL_QUANTUM_NS);
            self.cache.tick()
        }
        fn before_device_evict(&mut self, ticket: Option<StoreTicket>) -> EvictDecision {
            self.cache.before_device_evict(ticket)
        }
        fn lookup(&mut self, tokens: &[u32]) -> Option<Hit> {
            self.cache.lookup(tokens)
        }
        fn restore(&mut self, key: Key, target: &RestoreTarget) -> RestoreOutcome {
            self.cache.restore(key, target)
        }
        fn device_page_freed(&mut self, id: DevicePageId) {
            self.cache.device_page_freed(id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::RecordingCache;
    use super::*;

    #[test]
    fn pages_for_matches_engine_arithmetic() {
        assert_eq!(pages_for(0), 0);
        assert_eq!(pages_for(1), 1);
        assert_eq!(pages_for(512), 5);
        assert_eq!(pages_for(512) * PAGE_BYTES, 5 * PAGE_BYTES);
        assert_eq!(pages_for(1024), 10);
    }

    #[test]
    fn page_segments_sum_to_page_bytes() {
        assert_eq!(PAGE_SEGMENTS.iter().sum::<usize>(), PAGE_BYTES);
        let addressing = Addressing::new(&EngineModel::default());
        let id = DevicePageId {
            compressor: 2,
            page: 7,
            generation: 1,
        };
        let segments = addressing.page_segments(id);
        assert_eq!(
            segments.iter().map(|segment| segment.bytes).sum::<usize>(),
            PAGE_BYTES
        );
        assert_eq!(segments[0].addr, addressing.page_addr(id, 0));
        assert!(segments
            .windows(2)
            .all(|w| w[0].addr + w[0].bytes as u64 <= w[1].addr));
    }

    #[test]
    fn reusable_tokens_mirrors_engine_rule() {
        assert_eq!(reusable_tokens(1000, 1000), 1000);
        assert_eq!(reusable_tokens(300, 1000), 172);
        assert_eq!(reusable_tokens(128, 1000), 0);
    }

    #[test]
    fn token_streams_are_session_disjoint_and_stable() {
        assert_eq!(token_at(3, 17), token_at(3, 17));
        assert_ne!(token_at(3, 17), token_at(4, 17));
        let short = token_prefix(5, 64);
        let long = token_prefix(5, 128);
        assert_eq!(short, long[..64]);
    }

    #[test]
    fn rng_is_deterministic() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..16 {
            assert_eq!(a.next(), b.next());
        }
        assert_ne!(Rng::new(1).next(), Rng::new(2).next());
    }

    #[test]
    fn device_pool_allocates_stripes_and_reuses_with_new_generation() {
        let model = EngineModel {
            device_pool_tokens: 200_000,
            ..EngineModel::default()
        };
        let mut device = Device::new(&model);
        let id = device
            .allocate(0, 8, &mut RecordingCache::new(), &mut |_| {})
            .expect("allocation must fit");
        assert_eq!(id[0].compressor, 0);
        assert_eq!(id[1].compressor, 1);
        assert_eq!(id[4].compressor, 0);
        let mut cache = RecordingCache::new();
        for page in &id {
            device.unref(*page, &mut cache).expect("release");
        }
        assert_eq!(cache.freed_pages.len(), 8);
        // The request released its references without retiring a snapshot.
        device.request_refs = 0;
        // Freed slots come back with a fresh generation (a freed index is a new identity).
        let again = device
            .allocate(0, 8, &mut cache, &mut |_| {})
            .expect("reallocation must fit");
        let mut before: Vec<_> = id.iter().map(|page| page.page).collect();
        let mut after: Vec<_> = again.iter().map(|page| page.page).collect();
        before.sort_unstable();
        after.sort_unstable();
        assert_eq!(before, after);
        assert!(again.iter().all(|page| page.generation == 1));
        for page in &again {
            device.unref(*page, &mut cache).expect("release");
        }
        device.request_refs = 0;
        device.check().expect("device invariants hold");
    }

    #[test]
    fn device_make_room_evicts_prompts_before_turns() {
        let model = EngineModel {
            device_pool_tokens: 60_000,
            ..EngineModel::default()
        };
        let mut device = Device::new(&model);
        let mut cache = RecordingCache::new();
        let pool_pages = device.pools[0].pages();
        let fill = pool_pages * 2; // two full stripes per compressor
        let pages = device
            .allocate(0, fill, &mut cache, &mut |_| {})
            .expect("allocation must fit");
        let (prompt_pages, turn_pages) = pages.split_at(fill / 2);
        // The requests hand their pages to the bank entries (the retire-path transfer).
        device.request_refs -= fill;
        device
            .adopt_entry(
                SnapshotKind::Prompt,
                &[1, 2, 3],
                prompt_pages.to_vec(),
                None,
                &mut cache,
                &mut |_| {},
            )
            .expect("prompt bank insert");
        device
            .adopt_entry(
                SnapshotKind::Turn,
                &[1, 2, 3, 4],
                turn_pages.to_vec(),
                None,
                &mut cache,
                &mut |_| {},
            )
            .expect("turn bank insert");
        // One page beyond the pool forces eviction of the prompt bank first.
        device
            .allocate(0, fill + 1, &mut cache, &mut |_| {})
            .expect("make room by evicting the prompt snapshot");
        assert_eq!(cache.evictions.len(), 1);
        assert_eq!(device.banks.bank(SnapshotKind::Prompt).entries(), 0);
        assert_eq!(device.banks.bank(SnapshotKind::Turn).entries(), 1);
        device.check().expect("device invariants hold");
    }

    #[test]
    fn device_exhaustion_reports_source_pool_exhausted() {
        let model = EngineModel {
            device_pool_tokens: 30_000,
            ..EngineModel::default()
        };
        let mut device = Device::new(&model);
        let too_many = device.pools[0].pages() * 4 + 1;
        let err = device
            .allocate(0, too_many, &mut RecordingCache::new(), &mut |_| {})
            .expect_err("empty banks cannot make room");
        assert!(err.contains("SourcePoolExhausted"));
    }

    #[test]
    fn recording_cache_reports_every_ticket_once() {
        let mut cache = RecordingCache::with_completion_ticks(3);
        let addressing = Addressing::new(&EngineModel::default());
        let tokens = token_prefix(0, 512);
        let pages = (0..5)
            .map(|i| DevicePageId {
                compressor: (i % 4) as u8,
                page: i,
                generation: 0,
            })
            .collect::<Vec<_>>();
        let snapshot = build_snapshot(&addressing, SnapshotKind::Prompt, &tokens, &pages);
        let StoreOutcome::Issued(ticket) = CacheOps::store(&mut cache, &snapshot, ()) else {
            panic!("store must issue a ticket");
        };
        assert!(CacheOps::tick(&mut cache).completed.is_empty());
        assert!(CacheOps::tick(&mut cache).completed.is_empty());
        assert_eq!(CacheOps::tick(&mut cache).completed, vec![ticket]);
        assert!(cache.lookup(&tokens).is_some());
        assert!(CacheOps::tick(&mut cache).completed.is_empty());
    }

    #[test]
    fn recording_cache_restore_checks_fidelity() {
        let mut cache = RecordingCache::new();
        let addressing = Addressing::new(&EngineModel::default());
        let tokens = token_prefix(0, 512);
        let pages = (0..5)
            .map(|i| DevicePageId {
                compressor: (i % 4) as u8,
                page: i,
                generation: 0,
            })
            .collect::<Vec<_>>();
        let snapshot = build_snapshot(&addressing, SnapshotKind::Prompt, &tokens, &pages);
        let StoreOutcome::Issued(_) = CacheOps::store(&mut cache, &snapshot, ()) else {
            panic!("store must issue a ticket");
        };
        CacheOps::tick(&mut cache);
        CacheOps::tick(&mut cache);
        let done = CacheOps::restore(&mut cache, 0, &restore_target(&addressing, &tokens, &pages));
        assert!(matches!(done, RestoreOutcome::Done { .. }));
        // An unknown key fails.
        let unknown = restore_target(&addressing, &tokens, &pages);
        assert!(matches!(
            CacheOps::restore(&mut cache, 99, &unknown),
            RestoreOutcome::Failed
        ));
        // A target for another snapshot's tokens fails: the tail address is snapshot-derived,
        // so the fingerprints cannot match.
        let other_tokens = token_prefix(1, 512);
        let wrong_tail = restore_target(&addressing, &other_tokens, &pages);
        assert!(matches!(
            CacheOps::restore(&mut cache, 0, &wrong_tail),
            RestoreOutcome::Failed
        ));
        // A target with a different page structure fails.
        let wrong_pages = restore_target(&addressing, &tokens, &pages[..4]);
        assert!(matches!(
            CacheOps::restore(&mut cache, 0, &wrong_pages),
            RestoreOutcome::Failed
        ));
    }

    #[test]
    fn page_addresses_are_disjoint_across_pages() {
        // Every page's segments must be disjoint from every other page's: addresses are the
        // true page stride apart, so no two ranges may overlap.
        let model = EngineModel::default();
        let addressing = Addressing::new(&model);
        let ids: Vec<DevicePageId> = (0..4)
            .flat_map(|compressor| {
                (0..8).map(move |page| DevicePageId {
                    compressor,
                    page,
                    generation: 0,
                })
            })
            .collect();
        let mut ranges: Vec<(u64, u64)> = ids
            .iter()
            .flat_map(|&id| addressing.page_segments(id))
            .map(|range| (range.addr, range.addr + range.bytes as u64))
            .collect();
        ranges.sort_unstable();
        assert!(
            ranges.windows(2).all(|w| w[0].1 <= w[1].0),
            "segment ranges must be pairwise disjoint: {ranges:?}"
        );
    }

    /// Every address the simulator can hand out — every pool page's segments, every tail slot —
    /// must lie inside `EngineModel::device_bytes`, the exact size a suite gives the stub's
    /// flat fake device.
    #[test]
    fn every_address_lies_below_device_bytes() {
        for model in [
            EngineModel::default(),
            EngineModel {
                device_pool_tokens: 65_536,
                ..EngineModel::default()
            },
            EngineModel {
                device_pool_tokens: 30_000,
                retain: 2,
                lanes: 3,
                ..EngineModel::default()
            },
        ] {
            let device_bytes = model.device_bytes() as u64;
            let addressing = Addressing::new(&model);
            let pages = model.pool_pages() as u32;
            for compressor in 0..COMPRESSORS as u8 {
                for page in 0..pages {
                    let id = DevicePageId {
                        compressor,
                        page,
                        generation: 0,
                    };
                    for range in addressing.page_segments(id) {
                        assert!(
                            range.addr + range.bytes as u64 <= device_bytes,
                            "page range {range:?} on {id:?} escapes {device_bytes}"
                        );
                    }
                }
            }
            // Tails: hash-derived slots across the arena, and many distinct snapshots to
            // sample the hash (the arena holds one slot per live tail, so slot reuse is
            // expected across a run; every slot must still be in bounds).
            for session in 0..64 {
                for len in [1usize, 512, 4096] {
                    let tokens = token_prefix(session, len);
                    for range in addressing.tail_range(&tokens) {
                        assert!(
                            range.addr + range.bytes as u64 <= device_bytes,
                            "tail range {range:?} escapes {device_bytes}"
                        );
                        assert!(
                            range.addr >= addressing.tails_base,
                            "tail range {range:?} is not in the tails arena"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn device_hit_shares_pages_without_fresh_allocation() {
        // Copy-on-write: a request that hits a bank entry takes a reference per shared page —
        // the page's refcount rises above one and no fresh allocation happens.
        let model = EngineModel {
            device_pool_tokens: 100_000,
            ..EngineModel::default()
        };
        let mut device = Device::new(&model);
        let mut cache = RecordingCache::new();
        let tokens = token_prefix(0, 512);
        let pages = device
            .allocate(0, 5, &mut cache, &mut |_| {})
            .expect("allocation must fit");
        // The request retires its turn; the bank takes over the references.
        device.request_refs -= pages.len();
        device
            .adopt_entry(
                SnapshotKind::Turn,
                &tokens,
                pages.clone(),
                None,
                &mut cache,
                &mut |_| {},
            )
            .expect("bank insert");
        for &id in &pages {
            assert_eq!(
                device.pools[id.compressor as usize].refcount[id.page as usize],
                1
            );
        }
        // A returning request hits the bank and shares every page (the device_lookup path).
        let shared = pages.len();
        for &id in &pages {
            device.add_ref(id).expect("share bank page");
        }
        device.request_refs += shared;
        for &id in &pages {
            assert_eq!(
                device.pools[id.compressor as usize].refcount[id.page as usize], 2,
                "page {id:?} is shared between the bank entry and the request"
            );
        }
        let free: usize = device.pools.iter().map(|pool| pool.free.len()).sum();
        assert_eq!(
            free,
            device.pools[0].pages() * COMPRESSORS - pages.len(),
            "a device hit allocates no fresh pages"
        );
        device.check().expect("device invariants hold");
    }

    #[test]
    fn device_check_sums_refcounts_and_flags_free_slots() {
        let model = EngineModel {
            device_pool_tokens: 100_000,
            ..EngineModel::default()
        };
        let mut device = Device::new(&model);
        let mut cache = RecordingCache::new();
        let pages = device
            .allocate(0, 4, &mut cache, &mut |_| {})
            .expect("allocation must fit");
        device.check().expect("fresh device is consistent");
        device.audit().expect("fresh device audits clean");

        // A reference the per-pool sums counted but the device total did not is drift the
        // per-step check catches.
        device.pools[0].slot_refs += 1;
        let err = device.check().expect_err("uncounted reference must trip");
        assert!(err.contains("slots hold"), "got {err}");
        device.pools[0].slot_refs -= 1;

        // A reference taken without going through the device is drift the from-scratch audit
        // catches straight from the refcount array.
        device.pools[0].refcount[pages[0].page as usize] += 1;
        let err = device.audit().expect_err("array corruption must trip");
        assert!(err.contains("slots hold"), "got {err}");
        device.pools[0].refcount[pages[0].page as usize] -= 1;

        // A live slot parked on the free list is caught by the audit's free-slot scan.
        device.pools[0].free.push(pages[0].page);
        let err = device
            .audit()
            .expect_err("live slot on the free list must trip");
        assert!(err.contains("free slot"), "got {err}");
        device.pools[0].free.pop();

        device.check().expect("restored device is consistent");
        device.audit().expect("restored device audits clean");
    }
}
