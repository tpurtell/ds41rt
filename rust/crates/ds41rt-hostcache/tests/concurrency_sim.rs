//! Concurrency (interleaving) suite for the simulator (HC-4). The crate is single-threaded by
//! design, so "concurrency" here means many requests in flight on eight lane slots, their steps
//! interleaved by the seeded scheduler while store copies complete asynchronously a few ticks
//! after issue — with the copy engine's `tick` itself an interleaver choice, so completions
//! land between arbitrary lane steps. Every failure carries its seed, and the schedule log
//! reproduces it exactly.
use ds41rt_hostcache::cache::{
    DeviceSnapshot, EvictDecision, RestoreOutcome, RestoreTarget, SkipReason, StoreOutcome,
    StoreTicket, TickReport,
};
use ds41rt_hostcache::sim::testing::RecordingCache;
use ds41rt_hostcache::sim::{CacheOps, EngineModel, Simulator, Workload};
use ds41rt_hostcache::snapshot::{DevicePageId, Hit, Key};

/// The churn workload both tests drive: eight lane slots, far more conversations than the
/// device banks hold, so every stage of the request life cycle is exercised concurrently.
const SEED: u64 = 0xC0FFEE;
fn churn() -> Workload {
    Workload::Churn {
        sessions: 48,
        turns: 6,
        context_tokens: 2048,
        live_ratio: 1.0,
    }
}

#[test]
fn eight_lanes_interleave_with_pending_copies() {
    let model = EngineModel::default();
    let workload = churn();
    let mut sim = Simulator::new(model, RecordingCache::with_completion_ticks(3), SEED);
    let report = sim.run(&workload);
    assert!(
        report.invariant_failures.is_empty(),
        "interleaved run must be clean, got {:?}; log tail: {:?}",
        report.invariant_failures,
        report
            .schedule_log
            .iter()
            .rev()
            .take(10)
            .collect::<Vec<_>>()
    );
    let cache = sim.cache();
    assert!(
        cache.max_pending >= 2,
        "store copies must overlap the lanes: max_pending={}",
        cache.max_pending
    );
    assert!(
        cache.stores.len() >= 8,
        "many snapshots are stored during the run: {}",
        cache.stores.len()
    );
    // Eight slots admitted in the first wave: the opening of the log names eight sessions.
    let admitted = report
        .schedule_log
        .iter()
        .filter(|line| line.contains("admit"))
        .take(8)
        .count();
    assert_eq!(admitted, 8, "the first wave fills all eight lane slots");
}

#[test]
fn seeded_schedule_reproduces_under_interleaving() {
    let workload = churn();
    let run = || {
        Simulator::new(
            EngineModel::default(),
            RecordingCache::with_completion_ticks(3),
            SEED,
        )
        .run(&workload)
    };
    let first = run();
    let second = run();
    assert_eq!(first, second, "the same seed must reproduce report and log");
}

/// A cache that plants a fault: on the `inject_at_tick`-th `tick` it reports a ticket that was
/// never issued. The simulator's "every ticket reported once" invariant must trip, and the seed
/// must reproduce the failure.
struct InjectedFaultCache {
    inner: RecordingCache,
    inject_at_tick: u64,
    ticks: u64,
}

impl CacheOps for InjectedFaultCache {
    type Payload = ();
    fn store(&mut self, snapshot: &DeviceSnapshot, (): ()) -> StoreOutcome {
        self.inner.store(snapshot, ())
    }
    fn tick(&mut self) -> TickReport {
        self.ticks += 1;
        let mut report = self.inner.tick();
        if self.ticks == self.inject_at_tick {
            report.completed.push(StoreTicket(999_999));
        }
        report
    }
    fn before_device_evict(&mut self, ticket: Option<StoreTicket>) -> EvictDecision {
        self.inner.before_device_evict(ticket)
    }
    fn lookup(&mut self, tokens: &[u32]) -> Option<Hit> {
        self.inner.lookup(tokens)
    }
    fn restore(&mut self, key: Key, target: &RestoreTarget) -> RestoreOutcome {
        self.inner.restore(key, target)
    }
    fn device_page_freed(&mut self, id: DevicePageId) {
        self.inner.device_page_freed(id);
    }
}

#[test]
fn planted_fault_reproduces_from_its_seed() {
    let workload = churn();
    let inject_at_tick = 5;
    let run = || {
        Simulator::new(
            EngineModel::default(),
            InjectedFaultCache {
                inner: RecordingCache::new(),
                inject_at_tick,
                ticks: 0,
            },
            SEED,
        )
        .run(&workload)
    };

    // Without the injection the workload is clean: the fault is the injection, not the model.
    let clean = Simulator::new(EngineModel::default(), RecordingCache::new(), SEED).run(&workload);
    assert!(
        clean.invariant_failures.is_empty(),
        "workload must be clean without the planted fault: {:?}",
        clean.invariant_failures
    );

    let first = run();
    assert_eq!(
        first.invariant_failures.len(),
        1,
        "the planted double-report must trip exactly one invariant, got {:?}",
        first.invariant_failures
    );
    assert!(
        first.invariant_failures[0].contains("not outstanding"),
        "the failure names the bogus ticket: {:?}",
        first.invariant_failures
    );

    let second = run();
    assert_eq!(first, second, "same seed, same planted failure, same log");
    // The log pinpoints where the schedule was when the fault landed.
    let violation_step = first
        .schedule_log
        .iter()
        .find(|line| line.contains("INVARIANT VIOLATION"))
        .expect("the log records the failing step");
    let second_violation_step = second
        .schedule_log
        .iter()
        .find(|line| line.contains("INVARIANT VIOLATION"))
        .expect("the log records the failing step");
    assert_eq!(violation_step, second_violation_step);
}

/// The faults a [`FaultInjectingCache`] plants, each chosen by an ordinal so a seed reproduces
/// exactly where it fired. Every fault records that it fired, so a test proves the path was
/// exercised rather than silently skipped.
#[derive(Default)]
struct Faults {
    /// On the `n`-th restore call, report `TimedOut` without delegating.
    timeout_restore: Option<u64>,
    /// On the `n`-th store call, answer `Skipped(TooSmall)` without recording anything.
    skip_store: Option<u64>,
    /// On the `n`-th store call, answer `Deferred` instead of `Issued`; the copy still runs.
    defer_store: Option<u64>,
    /// At the first tick on or after the `n`-th, re-label one completed ticket `failed`.
    fail_tick: Option<u64>,
    /// On the `n`-th `before_device_evict` call, answer `DroppedUncached`.
    drop_evict: Option<u64>,
}

/// A recording cache with one fault planted (the `InjectedFaultCache` pattern above, extended
/// along every outcome the engine contract defines). The wrapped cache does the real work; the
/// wrapper only bends one answer, exactly when its ordinal comes up.
struct FaultInjectingCache {
    inner: RecordingCache,
    faults: Faults,
    restores: u64,
    stores: u64,
    ticks: u64,
    evicts: u64,
    /// The faults that fired, in firing order.
    fired: Vec<&'static str>,
}

impl CacheOps for FaultInjectingCache {
    type Payload = ();
    fn store(&mut self, snapshot: &DeviceSnapshot, (): ()) -> StoreOutcome {
        self.stores += 1;
        if self.faults.skip_store == Some(self.stores) {
            self.fired.push("skip_store");
            return StoreOutcome::Skipped(SkipReason::TooSmall);
        }
        let outcome = self.inner.store(snapshot, ());
        if self.faults.defer_store == Some(self.stores) {
            self.fired.push("defer_store");
            if let StoreOutcome::Issued(ticket) = outcome {
                return StoreOutcome::Deferred(ticket);
            }
        }
        outcome
    }
    fn tick(&mut self) -> TickReport {
        self.ticks += 1;
        let mut report = self.inner.tick();
        if self.faults.fail_tick.is_some_and(|at| self.ticks >= at)
            && !report.completed.is_empty()
            && !self.fired.contains(&"fail_tick")
        {
            report.failed.push(report.completed.remove(0));
            self.fired.push("fail_tick");
        }
        report
    }
    fn before_device_evict(&mut self, ticket: Option<StoreTicket>) -> EvictDecision {
        self.evicts += 1;
        if self.faults.drop_evict == Some(self.evicts) {
            self.fired.push("drop_evict");
            // Settle the copy through the inner cache if it is still pending, then report the
            // drop: the ticket is settled on the evict path and must never reach `tick`.
            if let Some(ticket) = ticket {
                let _ = self.inner.before_device_evict(Some(ticket));
            }
            return EvictDecision::DroppedUncached;
        }
        self.inner.before_device_evict(ticket)
    }
    fn lookup(&mut self, tokens: &[u32]) -> Option<Hit> {
        self.inner.lookup(tokens)
    }
    fn restore(&mut self, key: Key, target: &RestoreTarget) -> RestoreOutcome {
        self.restores += 1;
        if self.faults.timeout_restore == Some(self.restores) {
            self.fired.push("timeout_restore");
            return RestoreOutcome::TimedOut;
        }
        self.inner.restore(key, target)
    }
    fn device_page_freed(&mut self, id: DevicePageId) {
        self.inner.device_page_freed(id);
    }
}

/// A churn workload with tiny device banks: returning visits miss the device and restore from
/// the host cache.
fn host_hit_churn() -> Workload {
    Workload::Churn {
        sessions: 8,
        turns: 3,
        context_tokens: 1024,
        live_ratio: 1.0,
    }
}

fn tiny_banks() -> EngineModel {
    EngineModel {
        retain: 2,
        ..EngineModel::default()
    }
}

#[test]
fn restore_timeout_unrefs_fresh_pages_and_prefills_cold() {
    let workload = host_hit_churn();
    let mut sim = Simulator::new(
        tiny_banks(),
        FaultInjectingCache {
            inner: RecordingCache::new(),
            faults: Faults {
                timeout_restore: Some(1),
                ..Faults::default()
            },
            restores: 0,
            stores: 0,
            ticks: 0,
            evicts: 0,
            fired: Vec::new(),
        },
        SEED,
    );
    let report = sim.run(&workload);
    assert!(
        report.invariant_failures.is_empty(),
        "the timeout path must settle cleanly: {:?}",
        report.invariant_failures
    );
    let cache = sim.cache();
    assert_eq!(cache.fired, ["timeout_restore"]);
    assert!(
        report
            .schedule_log
            .iter()
            .any(|line| line.contains("restore timed out")),
        "the timed-out request must log its cold prefill"
    );
    // The timed-out restore never completed; every other host hit restored through the inner
    // cache, so the accounting closes: completed restores plus the one timed out equal the
    // host hits the report counted.
    assert_eq!(
        cache.inner.restores.len() as u64 + 1,
        report.host_hits,
        "exactly the injected restore timed out"
    );
    // The fresh pages the restore reserved were unref'd through the cache, and the request
    // re-allocated and prefilled cold: the per-step device checks and the end-of-run audit
    // passing are the proof, and at least the eviction-path frees are visible.
    assert!(!cache.inner.freed_pages.is_empty());
    assert!(
        report.prefilled_tokens >= 1024,
        "the timed-out request prefilled its full context from cold"
    );
}

#[test]
fn skipped_store_is_not_stored_and_settles_without_a_ticket() {
    let workload = Workload::Burst {
        prompts: 2,
        tokens: 1024,
    };
    let mut sim = Simulator::new(
        EngineModel::default(),
        FaultInjectingCache {
            inner: RecordingCache::new(),
            faults: Faults {
                skip_store: Some(1),
                ..Faults::default()
            },
            restores: 0,
            stores: 0,
            ticks: 0,
            evicts: 0,
            fired: Vec::new(),
        },
        SEED,
    );
    let report = sim.run(&workload);
    assert!(
        report.invariant_failures.is_empty(),
        "a skipped store leaves no dangling ticket: {:?}",
        report.invariant_failures
    );
    let cache = sim.cache();
    assert_eq!(cache.fired, ["skip_store"]);
    assert_eq!(
        cache.inner.stores.len(),
        1,
        "only the second prompt's snapshot was stored"
    );
    assert_eq!(report.misses, 2, "both prompts prefilled from cold");
}

#[test]
fn deferred_store_completes_through_ticks() {
    let workload = Workload::Burst {
        prompts: 2,
        tokens: 1024,
    };
    let mut sim = Simulator::new(
        EngineModel::default(),
        FaultInjectingCache {
            inner: RecordingCache::new(),
            faults: Faults {
                defer_store: Some(1),
                ..Faults::default()
            },
            restores: 0,
            stores: 0,
            ticks: 0,
            evicts: 0,
            fired: Vec::new(),
        },
        SEED,
    );
    let report = sim.run(&workload);
    assert!(
        report.invariant_failures.is_empty(),
        "a deferred ticket settles like an issued one: {:?}",
        report.invariant_failures
    );
    let cache = sim.cache();
    assert_eq!(cache.fired, ["defer_store"]);
    assert_eq!(
        cache.inner.stores.len(),
        2,
        "the deferred copy completed through later ticks"
    );
}

#[test]
fn failed_ticket_is_settled_exactly_once() {
    let workload = Workload::Burst {
        prompts: 3,
        tokens: 1024,
    };
    let mut sim = Simulator::new(
        EngineModel::default(),
        FaultInjectingCache {
            inner: RecordingCache::new(),
            faults: Faults {
                fail_tick: Some(1),
                ..Faults::default()
            },
            restores: 0,
            stores: 0,
            ticks: 0,
            evicts: 0,
            fired: Vec::new(),
        },
        SEED,
    );
    let report = sim.run(&workload);
    assert!(
        report.invariant_failures.is_empty(),
        "a failed ticket is settled like a completed one: {:?}",
        report.invariant_failures
    );
    let cache = sim.cache();
    assert_eq!(cache.fired, ["fail_tick"]);
    assert_eq!(
        cache.inner.stores.len(),
        3,
        "every store resolved, one re-labelled failed"
    );
}

#[test]
fn dropped_uncached_settles_the_ticket_on_the_evict_path() {
    // A small pool and more prompts than the device banks hold: evictions consult the cache
    // while the injected store copies are still pending.
    let model = EngineModel {
        device_pool_tokens: 40_000,
        ..EngineModel::default()
    };
    let workload = Workload::Burst {
        prompts: 30,
        tokens: 512,
    };
    let mut sim = Simulator::new(
        model,
        FaultInjectingCache {
            inner: RecordingCache::with_completion_ticks(50),
            faults: Faults {
                drop_evict: Some(1),
                ..Faults::default()
            },
            restores: 0,
            stores: 0,
            ticks: 0,
            evicts: 0,
            fired: Vec::new(),
        },
        SEED,
    );
    let report = sim.run(&workload);
    assert!(
        report.invariant_failures.is_empty(),
        "a dropped ticket must never reach tick: {:?}",
        report.invariant_failures
    );
    let cache = sim.cache();
    assert_eq!(cache.fired, ["drop_evict"]);
    assert!(
        report
            .schedule_log
            .iter()
            .any(|line| line.contains("decision=DroppedUncached")),
        "the eviction consultation logged the drop"
    );
    assert_eq!(
        cache.inner.stores.len() as u64,
        report.misses,
        "every burst prompt was stored, one settling on the evict path"
    );
}
