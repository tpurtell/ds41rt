//! Stability suite for packet HC-5: a wall-clock soak of the concurrency interleaver under quota
//! pressure. Between phases the pool must hold exactly the model's bytes (no leaked slabs) and
//! the counters must balance: every completed store is either resident, evicted or a replacement.
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
use ds41rt_hostcache::copy::StubCopyEngine;
use ds41rt_hostcache::pool::testing::CHUNK;
use ds41rt_hostcache::snapshot::Key;
use ds41rt_hostcache::SnapshotKind;
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

/// A small deterministic generator so a soak reproduces from its seed.
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

struct Pending {
    ticket: StoreTicket,
    key: Key,
    snapshot: DeviceSnapshot,
    expected: Vec<u8>,
}

/// The soak's state: the cache, the model, the fake device and the ticket bookkeeping.
struct Soak {
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
    rng: Rng,
}

impl Soak {
    fn new(quota: u64, seed: u64) -> Self {
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
            rng: Rng::new(seed),
        }
    }

    /// One interleaved step, chosen from the same mix the concurrency suite uses.
    fn step(&mut self) {
        match self.rng.below(6) {
            0 | 1 => self.store(),
            2 => self.tick(),
            3 => self.lookup(),
            4 => self.restore(),
            _ => self.evict(),
        }
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

    fn lookup(&mut self) {
        let tokens = if self.stored.is_empty() {
            tokens(8)
        } else {
            let index = self.rng.below(self.stored.len() as u64) as usize;
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

    fn restore(&mut self) {
        if self.stored.is_empty() {
            return;
        }
        let index = self.rng.below(self.stored.len() as u64) as usize;
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

    /// The between-phase invariants: the pool holds exactly the model's bytes (no leak) and the
    /// counters balance.
    fn check(&self) {
        let metrics = self.cache.metrics();
        assert!(metrics.bytes_used <= self.quota, "bytes over quota");
        assert_eq!(
            metrics.bytes_used, self.model.bytes,
            "pool occupancy leaked past the model"
        );
        assert_eq!(
            metrics.resident_snapshots as usize,
            self.model.snapshots.len(),
            "resident count diverged from the model"
        );
        assert_eq!(
            metrics.stores_completed,
            metrics.resident_snapshots + metrics.host_evictions + metrics.stores_replaced,
            "store counters do not balance"
        );
    }
}

/// The soak's wall-clock length; override with `HC5_SOAK_SECONDS` for a quick run.
fn soak_seconds() -> u64 {
    std::env::var("HC5_SOAK_SECONDS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(60)
}

#[test]
fn soak_has_no_leak_and_balances_the_counters() {
    let deadline = Instant::now() + Duration::from_secs(soak_seconds());
    let mut soak = Soak::new(8 * CHUNK as u64, 0x50AC_0005);
    let mut phases = 0;
    while Instant::now() < deadline {
        let phase_end = Instant::now() + Duration::from_secs(10);
        while Instant::now() < phase_end && Instant::now() < deadline {
            soak.step();
        }
        soak.drain();
        soak.check();
        phases += 1;
    }
    assert!(phases >= 1, "the soak ran no phase");
    assert!(
        soak.cache.metrics().host_evictions > 0,
        "the soak never evicted from the host cache"
    );
    assert!(
        soak.cache.metrics().host_hits > 0,
        "the soak never hit the host cache"
    );
}
