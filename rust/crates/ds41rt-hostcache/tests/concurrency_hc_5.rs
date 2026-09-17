//! Concurrency suite for packet HC-5. The crate is single-threaded by design, so "concurrency"
//! means interleaving: eight lanes of store / tick / before_device_evict / lookup / restore /
//! device-page-overwrite steps run in a seeded, logged order against the stub engine under a
//! small quota that forces host eviction. The HC-2 model is driven in lockstep, and the
//! invariants are checked after every step; a failure prints the schedule and reproduces from
//! its seed.
mod common;
mod common_hc_5;

use common::{reconcile, Model};
use common_hc_5::{
    default_cache, device_pages, restored_bytes, settle, snapshot, stored_bytes, target, tokens,
    write_snapshot, Device, Payload, DEVICE_BYTES,
};
use ds41rt_hostcache::cache::{
    DeviceSnapshot, EvictDecision, HostCache, RestoreOutcome, StoreOutcome, StoreTicket,
};
use ds41rt_hostcache::config::StoreMode;
use ds41rt_hostcache::copy::{CopyEngine, StubCopyEngine};
use ds41rt_hostcache::pool::testing::CHUNK;
use ds41rt_hostcache::snapshot::Key;
use ds41rt_hostcache::{SnapshotKind, COMPRESSORS};
use std::collections::{HashMap, HashSet};

/// A small deterministic generator so a schedule reproduces from its seed.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }
    fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }
}

#[derive(Clone, Copy, Debug)]
enum Step {
    Store,
    Tick,
    Evict,
    Lookup,
    Restore,
    Overwrite,
}

/// A store in flight: its ticket, the model key, the snapshot it copies and the bytes the copy
/// must land (updated when a device page is overwritten before the copy executes).
struct Pending {
    ticket: StoreTicket,
    key: Key,
    snapshot: DeviceSnapshot,
    expected: Vec<u8>,
    /// The virtual clock when the copies were issued; a later clock means they may have run.
    issued_at: u64,
}

/// The interleaver's whole state: the cache, the model, the fake device and the bookkeeping the
/// invariants need.
struct State {
    cache: HostCache<StubCopyEngine, Payload>,
    model: Model,
    device: Device,
    quota: u64,
    evict_quota: u64,
    pending: Vec<Pending>,
    stored: HashMap<Key, (DeviceSnapshot, Vec<u8>)>,
    issued: HashSet<StoreTicket>,
    reported: HashSet<StoreTicket>,
    generation: u32,
    /// Device-page overwrites that landed before their copy executed.
    overwrites: usize,
}

impl State {
    fn new(quota: u64) -> Self {
        Self {
            cache: default_cache(quota, StoreMode::OnRetain),
            model: Model::new(),
            device: Device::new(DEVICE_BYTES),
            quota,
            evict_quota: quota,
            pending: Vec::new(),
            stored: HashMap::new(),
            issued: HashSet::new(),
            reported: HashSet::new(),
            generation: 0,
            overwrites: 0,
        }
    }

    fn apply(&mut self, step: Step, rng: &mut Rng) {
        match step {
            Step::Store => self.store(),
            Step::Tick => self.tick(),
            Step::Evict => self.evict(),
            Step::Lookup => self.lookup(rng),
            Step::Restore => self.restore(rng),
            Step::Overwrite => self.overwrite(rng),
        }
        self.check();
    }

    fn store(&mut self) {
        self.generation += 1;
        let tokens = tokens(8 + self.generation as usize);
        let snapshot = snapshot(
            &mut self.device,
            SnapshotKind::Turn,
            &tokens,
            1,
            false,
            self.generation,
        );
        write_snapshot(self.cache.engine_mut(), &snapshot);
        let expected = stored_bytes(self.cache.engine_mut(), &snapshot);
        let issued_at = self.cache.engine_mut().now_ns();
        let outcome = self.cache.store(&snapshot, self.generation as u64);
        // The cache may evict unpinned snapshots while planning under class exhaustion.
        reconcile(&self.cache, &mut self.model);
        self.prune_stored();
        if let StoreOutcome::Issued(ticket) | StoreOutcome::Deferred(ticket) = outcome {
            let key = self.model.plan(&snapshot.meta, &device_pages(&snapshot));
            self.pending.push(Pending {
                ticket,
                key,
                snapshot,
                expected,
                issued_at,
            });
            self.issued.insert(ticket);
        }
    }

    fn tick(&mut self) {
        settle(&mut self.cache);
        let report = self.cache.tick();
        for ticket in report.completed {
            let pending = self.take_pending(ticket);
            self.model.commit(pending.key);
            self.model.evict_to(self.evict_quota);
            self.stored
                .insert(pending.key, (pending.snapshot, pending.expected));
            self.prune_stored();
            assert!(self.reported.insert(ticket), "ticket reported twice");
        }
        for ticket in report.failed {
            let pending = self.take_pending(ticket);
            self.model.abort(pending.key);
            assert!(self.reported.insert(ticket), "ticket reported twice");
        }
    }

    fn evict(&mut self) {
        let Some(index) = self.pending.len().checked_sub(1) else {
            return;
        };
        let pending = self.pending.remove(index);
        match self.cache.before_device_evict(Some(pending.ticket)) {
            EvictDecision::WaitedClean { .. } => {
                self.model.commit(pending.key);
                self.model.evict_to(self.evict_quota);
                self.stored
                    .insert(pending.key, (pending.snapshot, pending.expected));
                self.prune_stored();
            }
            EvictDecision::DroppedUncached => self.model.abort(pending.key),
            EvictDecision::Clean => panic!("a pending ticket was clean"),
        }
        assert!(
            self.reported.insert(pending.ticket),
            "ticket reported twice"
        );
    }

    fn lookup(&mut self, rng: &mut Rng) {
        let tokens = if self.stored.is_empty() {
            tokens(8)
        } else {
            let index = rng.below(self.stored.len() as u64) as usize;
            self.stored
                .values()
                .nth(index)
                .expect("stored snapshot")
                .0
                .meta
                .tokens
                .clone()
        };
        let hit = self.cache.lookup(&tokens);
        let expected = self.model.lookup(&tokens);
        match (hit, expected) {
            (Some(hit), Some((common, frontier, key))) => {
                assert_eq!(hit.key, key, "lookup key");
                assert_eq!(hit.common, common, "lookup common");
                assert_eq!(hit.frontier, frontier, "lookup frontier");
            }
            (None, None) => {}
            (hit, expected) => panic!("lookup mismatch: {hit:?} vs {expected:?}"),
        }
    }

    fn restore(&mut self, rng: &mut Rng) {
        if self.stored.is_empty() {
            return;
        }
        let index = rng.below(self.stored.len() as u64) as usize;
        let (&key, (snapshot, expected)) = self.stored.iter().nth(index).expect("stored snapshot");
        self.generation += 1;
        let target = target(&mut self.device, snapshot, self.generation);
        let outcome = self.cache.restore(key, &target);
        assert!(
            self.cache.snapshot_tokens(key).is_some(),
            "a restored snapshot was evicted"
        );
        if matches!(outcome, RestoreOutcome::Done { .. }) {
            assert_eq!(
                restored_bytes(self.cache.engine_mut(), &target),
                *expected,
                "restored bytes differ from the stored bytes"
            );
        }
    }

    fn overwrite(&mut self, rng: &mut Rng) {
        // Only a store issued since the last clock advance still has its copy in flight: any
        // advance (a tick, an evict wait or a restore wait) may already have run it. Overwrite
        // such a store's device bytes so the copy reads the overwritten ones, and record that
        // expectation.
        let now = self.cache.engine_mut().now_ns();
        let eligible: Vec<usize> = self
            .pending
            .iter()
            .enumerate()
            .filter(|(_, pending)| pending.issued_at == now)
            .map(|(index, _)| index)
            .collect();
        if eligible.is_empty() {
            return;
        }
        let index = eligible[rng.below(eligible.len() as u64) as usize];
        let pending = &mut self.pending[index];
        let compressor = rng.below(COMPRESSORS as u64) as usize;
        if pending.snapshot.pages[compressor].is_empty() {
            return;
        }
        let page = rng.below(pending.snapshot.pages[compressor].len() as u64) as usize;
        let segment =
            rng.below(pending.snapshot.pages[compressor][page].segments.len() as u64) as usize;
        let range = pending.snapshot.pages[compressor][page].segments[segment];
        let bytes: Vec<u8> = (0..range.bytes).map(|offset| 0x5A ^ offset as u8).collect();
        self.cache.engine_mut().write_device(range, &bytes);
        pending.expected = stored_bytes(self.cache.engine_mut(), &pending.snapshot);
        self.overwrites += 1;
    }

    fn take_pending(&mut self, ticket: StoreTicket) -> Pending {
        let index = self
            .pending
            .iter()
            .position(|pending| pending.ticket == ticket)
            .expect("reported ticket was pending");
        self.pending.remove(index)
    }

    fn prune_stored(&mut self) {
        self.stored
            .retain(|key, _| self.model.snapshots.contains_key(key));
    }

    fn check(&self) {
        let metrics = self.cache.metrics();
        assert!(metrics.bytes_used <= self.quota, "bytes over quota");
        assert_eq!(
            metrics.bytes_used, self.model.bytes,
            "pool bytes diverged from the model"
        );
        assert_eq!(
            metrics.resident_snapshots as usize,
            self.model.snapshots.len(),
            "resident count diverged from the model"
        );
    }

    /// Drain every in-flight store and assert every ticket was reported exactly once.
    fn drain(&mut self) {
        while let Some(pending) = self.pending.pop() {
            match self.cache.before_device_evict(Some(pending.ticket)) {
                EvictDecision::WaitedClean { .. } => {
                    self.model.commit(pending.key);
                    self.model.evict_to(self.evict_quota);
                }
                EvictDecision::DroppedUncached => self.model.abort(pending.key),
                EvictDecision::Clean => {}
            }
            assert!(
                self.reported.insert(pending.ticket),
                "ticket reported twice"
            );
        }
        assert_eq!(
            self.issued, self.reported,
            "tickets not reported exactly once"
        );
    }
}

/// A lane's script: a mix of every step, offset per lane so the lanes differ.
fn lane_script(lane: usize, rng: &mut Rng) -> Vec<Step> {
    (0..250)
        .map(|step| match (lane + step) % 6 {
            0 => Step::Store,
            1 => Step::Tick,
            2 => Step::Evict,
            3 => Step::Lookup,
            4 => Step::Restore,
            _ => Step::Overwrite,
        })
        .collect::<Vec<_>>()
        .into_iter()
        .map(|step| {
            // A little jitter so the lanes are not perfectly periodic.
            if rng.below(8) == 0 {
                Step::Tick
            } else {
                step
            }
        })
        .collect()
}

fn run_seed(seed: u64) -> Vec<String> {
    let mut rng = Rng::new(seed);
    let mut lanes: Vec<Vec<Step>> = (0..8).map(|lane| lane_script(lane, &mut rng)).collect();
    let mut state = State::new(6 * CHUNK as u64);
    let mut log = Vec::new();
    let mut step = 0;
    while lanes.iter().any(|lane| !lane.is_empty()) {
        let lane = rng.below(lanes.len() as u64) as usize;
        if lanes[lane].is_empty() {
            continue;
        }
        let op = lanes[lane].remove(0);
        log.push(format!("{step}: lane {lane} {op:?}"));
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            state.apply(op, &mut rng);
        }));
        if let Err(payload) = outcome {
            eprintln!("seed {seed}: invariant failed after step {step}");
            for line in &log {
                eprintln!("  {line}");
            }
            std::panic::resume_unwind(payload);
        }
        step += 1;
    }
    state.drain();
    assert!(
        state.cache.metrics().host_hits > 0,
        "the schedule never hit the host cache"
    );
    assert!(
        state.cache.metrics().host_evictions > 0,
        "the schedule never evicted from the host cache: {:?}",
        state.cache.metrics()
    );
    assert!(
        state.overwrites > 0,
        "the schedule never overwrote a pending store's device page"
    );
    log
}

#[test]
fn seeded_lanes_interleave_under_quota_pressure() {
    let log = run_seed(0x5EED_1234_ABCD_0005);
    assert_eq!(log.len(), 2000, "schedule length");
    assert!(log[0].starts_with("0: lane "));
}

#[test]
fn every_seed_reproduces_and_keeps_the_invariants() {
    for seed in 1..=4 {
        let first = run_seed(seed);
        let second = run_seed(seed);
        assert_eq!(first, second, "seed {seed} did not reproduce");
    }
}
