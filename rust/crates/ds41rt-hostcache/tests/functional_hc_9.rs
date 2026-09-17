//! Functional suite for packet HC-9: the prefill pacing guard. `HostCache::prefill_hold` is
//! the scheduler's per-chunk boundary call: with `store_pace_ns` at zero (or the cache off,
//! or nothing in flight) the hold is a pure no-op — no clock read, no hold metric moves
//! (the daemon's wrapper observes the store stream with `tick` on every prefill chunk
//! regardless of the knob, so store metrics may still move at that level); when the oldest
//! in-flight store copy is older than the pace it waits on that store's event for at most the
//! pace again, counts the hold (and a timeout, when the budget ran out), and returns
//! regardless — never failing the request, never blocking longer than the pace, and never
//! double-counting the completing store's own latency. Proptest properties cover the named
//! invariants over random pending-store ages, paces and stream health.
mod common_hc_5;

use common_hc_5::{
    cache as build_cache, config as base_config, modelled_store_ns, snapshot, tokens,
    write_snapshot, Device, Payload, DEVICE_BYTES,
};
use ds41rt_hostcache::cache::{DeviceSnapshot, HostCache, StoreOutcome, StoreTicket};
use ds41rt_hostcache::config::{Config, StoreMode};
use ds41rt_hostcache::copy::{CopyEngine, CopyFault, CopyModel, Stream, StubCopyEngine};
use ds41rt_hostcache::metrics::Snapshot as Metrics;
use ds41rt_hostcache::pool::testing::CHUNK;
use ds41rt_hostcache::SnapshotKind;
use proptest::prelude::*;

/// The suite's pace: 100 ms, far above the default model's store duration and below the slow
/// model's in `completion_during_the_hold_returns_early_and_counts_once`.
const PACE_NS: u64 = 100_000_000;

type TestCache = HostCache<StubCopyEngine, Payload>;

/// The test config with the pacing knob set.
fn paced_config(bytes: u64, store: StoreMode, pace_ns: u64) -> Config {
    Config {
        store_pace_ns: pace_ns,
        ..base_config(bytes, store)
    }
}

/// A cache with the pacing knob set over the default copy model.
fn paced_cache(bytes: u64, store: StoreMode, pace_ns: u64) -> TestCache {
    build_cache(
        paced_config(bytes, store, pace_ns),
        CopyModel::default(),
        DEVICE_BYTES,
    )
}

/// The cache clock, for hold-duration arithmetic.
fn now(cache: &mut TestCache) -> u64 {
    cache.engine_mut().now_ns()
}

/// One page per compressor, no draft, over the pool suites' layout (see the HC-7 suite).
fn probe(device: &mut Device, generation: u32) -> DeviceSnapshot {
    snapshot(device, SnapshotKind::Turn, &tokens(8), 1, false, generation)
}

/// Write `snap`'s pattern into the engine and issue its store, arming a store-stream stall
/// first when `stalled` (the copies accepted after the stall and the event never complete).
fn issue(cache: &mut TestCache, snap: &DeviceSnapshot, stalled: bool) -> StoreTicket {
    write_snapshot(cache.engine_mut(), snap);
    if stalled {
        cache
            .engine_mut()
            .inject(CopyFault::StreamStalls(Stream::Store));
    }
    match cache.store(snap, 1) {
        StoreOutcome::Issued(ticket) => ticket,
        other => panic!("expected an issued store, got {other:?}"),
    }
}

fn hold_metrics(metrics: &Metrics) -> (u64, u64, u64) {
    (
        metrics.prefill_holds,
        metrics.prefill_hold_ns_sum,
        metrics.prefill_hold_timeouts,
    )
}

#[test]
fn nothing_pending_is_a_no_op() {
    let mut cache = paced_cache(4 * CHUNK as u64, StoreMode::OnRetain, PACE_NS);
    let before = now(&mut cache);
    cache.prefill_hold().expect("hold");
    assert_eq!(
        now(&mut cache),
        before,
        "the clock moved with nothing pending"
    );
    assert_eq!(hold_metrics(&cache.metrics()), (0, 0, 0));
}

#[test]
fn knob_zero_never_holds_even_with_an_overdue_store() {
    let mut cache = paced_cache(4 * CHUNK as u64, StoreMode::OnRetain, 0);
    let mut device = Device::new(DEVICE_BYTES);
    let snap = probe(&mut device, 1);
    issue(&mut cache, &snap, true);
    cache.engine_mut().advance(10 * PACE_NS);
    let before = now(&mut cache);
    cache.prefill_hold().expect("hold");
    assert_eq!(
        now(&mut cache),
        before,
        "the knob is zero: the hold must be a no-op"
    );
    assert_eq!(hold_metrics(&cache.metrics()), (0, 0, 0));
}

#[test]
fn disabled_cache_is_a_no_op() {
    let mut cache = paced_cache(0, StoreMode::OnRetain, PACE_NS);
    assert!(!cache.enabled());
    cache.prefill_hold().expect("hold");
    assert_eq!(hold_metrics(&cache.metrics()), (0, 0, 0));
}

#[test]
fn store_within_pace_is_not_held() {
    let mut cache = paced_cache(4 * CHUNK as u64, StoreMode::OnRetain, PACE_NS);
    let mut device = Device::new(DEVICE_BYTES);
    let snap = probe(&mut device, 1);
    issue(&mut cache, &snap, false);
    // The copy completes (age below the pace, still uncommitted in `pending`).
    cache.engine_mut().advance(PACE_NS / 2);
    let before = now(&mut cache);
    cache.prefill_hold().expect("hold");
    assert_eq!(now(&mut cache), before, "a store within its pace was held");
    assert_eq!(hold_metrics(&cache.metrics()), (0, 0, 0));
}

#[test]
fn overdue_stalled_store_holds_exactly_the_pace_and_times_out() {
    let mut cache = paced_cache(4 * CHUNK as u64, StoreMode::OnRetain, PACE_NS);
    let mut device = Device::new(DEVICE_BYTES);
    let snap = probe(&mut device, 1);
    issue(&mut cache, &snap, true);
    cache.engine_mut().advance(PACE_NS + 1);
    let before = now(&mut cache);
    cache.prefill_hold().expect("hold");
    assert_eq!(
        now(&mut cache).saturating_sub(before),
        PACE_NS,
        "the hold must wait at most one pace, then return"
    );
    let metrics = cache.metrics();
    assert_eq!(metrics.prefill_holds, 1);
    assert_eq!(metrics.prefill_hold_timeouts, 1);
    assert_eq!(metrics.prefill_hold_ns_sum, PACE_NS);
    assert!(
        metrics.prefill_hold_timeouts <= metrics.prefill_holds,
        "timeouts never exceed holds"
    );
    // The request is never failed; the wedged store stays pending (in-order polling never
    // reaches past it), reported neither completed nor failed.
    let report = cache.tick();
    assert!(report.completed.is_empty());
    assert!(report.failed.is_empty());
}

#[test]
fn completion_during_the_hold_returns_early_and_counts_once() {
    // A slow model: one probe store takes ~160 ms, comfortably between one and two paces.
    let model = CopyModel {
        d2h_bytes_per_ns: 0.00016,
        ..CopyModel::default()
    };
    let mut device = Device::new(DEVICE_BYTES);
    let snap = probe(&mut device, 1);
    let duration = modelled_store_ns(model, &snap);
    assert!(
        duration > PACE_NS && duration <= 2 * PACE_NS,
        "the modelled store ({duration} ns) must straddle the pace"
    );
    let mut cache = build_cache(
        paced_config(4 * CHUNK as u64, StoreMode::OnRetain, PACE_NS),
        model,
        DEVICE_BYTES,
    );
    issue(&mut cache, &snap, false);
    let start = now(&mut cache);
    // Age the store past the pace while its copy is still in flight.
    cache.engine_mut().advance(PACE_NS + 1);
    let hold_start = now(&mut cache);
    cache.prefill_hold().expect("hold");
    let hold_ns = now(&mut cache) - hold_start;
    assert_eq!(
        now(&mut cache),
        start + duration,
        "the hold ends at the copy's completion"
    );
    assert_eq!(
        hold_ns,
        duration - PACE_NS - 1,
        "early return at the completion"
    );
    let metrics = cache.metrics();
    assert_eq!(metrics.prefill_holds, 1);
    assert_eq!(metrics.prefill_hold_timeouts, 0);
    assert_eq!(metrics.prefill_hold_ns_sum, hold_ns);
    // The commit records the store's full latency exactly once: no double count.
    cache.tick();
    let metrics = cache.metrics();
    assert_eq!(metrics.stores_completed, 1);
    assert_eq!(metrics.store_latency_sum_ns, duration);
    assert_eq!(metrics.store_latency_buckets.iter().sum::<u64>(), 1);
    assert_eq!(metrics.prefill_holds, 1, "the commit counts no second hold");
}

#[test]
fn the_hold_waits_on_the_oldest_pending_store() {
    let mut cache = paced_cache(4 * CHUNK as u64, StoreMode::OnRetain, PACE_NS);
    let mut device = Device::new(DEVICE_BYTES);
    // First store completes (but stays uncommitted until `tick`); second is stalled.
    let first = probe(&mut device, 1);
    issue(&mut cache, &first, false);
    cache.engine_mut().advance(1_000_000);
    let second = probe(&mut device, 2);
    issue(&mut cache, &second, true);
    cache.engine_mut().advance(10 * PACE_NS);
    let before = now(&mut cache);
    cache.prefill_hold().expect("hold");
    assert_eq!(
        now(&mut cache),
        before,
        "the oldest store's copy already landed: the wait returns immediately"
    );
    let metrics = cache.metrics();
    assert_eq!(metrics.prefill_holds, 1);
    assert_eq!(metrics.prefill_hold_timeouts, 0);
    assert_eq!(metrics.prefill_hold_ns_sum, 0);
}

/// The daemon wrapper's tick-then-hold composition, pinned at crate level: a store whose
/// copy completed but is still uncommitted (the scheduler loop's `tick` cannot run while a
/// synchronous prefill owns the thread) is committed by the observation step; the hold that
/// follows then sees nothing overdue, returns immediately and counts nothing.
#[test]
fn tick_then_hold_commits_the_overdue_store_and_never_holds() {
    let mut cache = paced_cache(4 * CHUNK as u64, StoreMode::OnRetain, PACE_NS);
    let mut device = Device::new(DEVICE_BYTES);
    let snap = probe(&mut device, 1);
    issue(&mut cache, &snap, false);
    // The copy completes but nothing observes it: far past the pace, still uncommitted.
    cache.engine_mut().advance(10 * PACE_NS);
    let report = cache.tick();
    assert_eq!(
        report.completed.len(),
        1,
        "the observation commits the completion"
    );
    assert_eq!(cache.metrics().stores_completed, 1);
    // The hold the wrapper calls right after: nothing is pending anymore.
    let before = now(&mut cache);
    cache.prefill_hold().expect("hold");
    assert_eq!(
        now(&mut cache),
        before,
        "the hold after the tick never waits on a committed store"
    );
    assert_eq!(
        hold_metrics(&cache.metrics()),
        (0, 0, 0),
        "no hold is counted when the observation already committed the store"
    );
}

#[test]
fn deferred_on_evict_stores_are_not_pending_copies() {
    // `OnEvict` records without copying: nothing is in flight, so nothing can be held.
    let mut cache = paced_cache(4 * CHUNK as u64, StoreMode::OnEvict, PACE_NS);
    let mut device = Device::new(DEVICE_BYTES);
    let snap = probe(&mut device, 1);
    let StoreOutcome::Deferred(ticket) = cache.store(&snap, 1) else {
        panic!("expected a deferred store");
    };
    cache.engine_mut().advance(10 * PACE_NS);
    let before = now(&mut cache);
    cache.prefill_hold().expect("hold");
    assert_eq!(
        now(&mut cache),
        before,
        "a deferred store has no copy to wait on"
    );
    assert_eq!(hold_metrics(&cache.metrics()), (0, 0, 0));
    let _ = ticket;
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// With the knob at zero the hold never runs and no hold metric moves, whatever is
    /// pending (a `tick` may still commit completions and move store metrics).
    #[test]
    fn knob_off_freezes_everything(age in 0u64..10_000_000_000, stalled in prop::bool::ANY) {
        let mut cache = paced_cache(4 * CHUNK as u64, StoreMode::OnRetain, 0);
        let mut device = Device::new(DEVICE_BYTES);
        let snap = probe(&mut device, 1);
        issue(&mut cache, &snap, stalled);
        cache.engine_mut().advance(age);
        let before = now(&mut cache);
        cache.prefill_hold().expect("hold");
        prop_assert_eq!(now(&mut cache), before);
        prop_assert_eq!(hold_metrics(&cache.metrics()), (0, 0, 0));
    }

    /// Whatever the age, the pace and the stream's health, one hold never exceeds the pace,
    /// the counted sum equals the wall time waited, and timeouts never exceed holds.
    #[test]
    fn one_hold_is_always_bounded_by_the_pace(
        age in 0u64..10_000_000_000,
        pace in 1u64..1_000_000_000,
        stalled in prop::bool::ANY,
    ) {
        let mut cache = paced_cache(4 * CHUNK as u64, StoreMode::OnRetain, pace);
        let mut device = Device::new(DEVICE_BYTES);
        let snap = probe(&mut device, 1);
        issue(&mut cache, &snap, stalled);
        cache.engine_mut().advance(age);
        let before = now(&mut cache);
        cache.prefill_hold().expect("hold");
        let elapsed = now(&mut cache) - before;
        let metrics = cache.metrics();
        prop_assert!(elapsed <= pace, "held {elapsed} ns against a {pace} ns pace");
        prop_assert!(metrics.prefill_holds <= 1);
        prop_assert_eq!(metrics.prefill_hold_ns_sum, elapsed);
        prop_assert!(metrics.prefill_hold_ns_sum <= metrics.prefill_holds * pace);
        prop_assert!(metrics.prefill_hold_timeouts <= metrics.prefill_holds);
        if stalled && age > pace {
            prop_assert_eq!(elapsed, pace, "a stalled overdue store burns the full pace");
            prop_assert_eq!(metrics.prefill_hold_timeouts, 1);
        }
    }
}
