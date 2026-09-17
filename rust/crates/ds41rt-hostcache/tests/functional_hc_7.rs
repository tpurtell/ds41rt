//! Functional suite for packet HC-7: the two instrumentation additions on the cache facade —
//! `device_evictions` (one per `before_device_evict` consultation, before any budget wait, in
//! every outcome branch) and the store-copy latency histogram (recorded once per completed
//! store at its commit, wherever the completion is observed), plus the disabled no-ops and
//! proptest properties over the named invariants.
mod common_hc_5;

use common_hc_5::{
    cache as build_cache, config, default_cache, modelled_store_ns, settle, snapshot, tokens,
    write_snapshot, Device, FailAfter, DEVICE_BYTES,
};
use ds41rt_hostcache::cache::{DeviceSnapshot, EvictDecision, StoreOutcome};
use ds41rt_hostcache::config::StoreMode;
use ds41rt_hostcache::copy::{CopyFault, CopyModel, Stream};
use ds41rt_hostcache::metrics::RESTORE_BUCKETS_NS;
use ds41rt_hostcache::pool::testing::{layout, CHUNK};
use ds41rt_hostcache::SnapshotKind;
use proptest::prelude::*;

/// The histogram bucket a latency belongs to: the first bound it does not exceed, else overflow.
fn expected_bucket(latency_ns: u64) -> usize {
    RESTORE_BUCKETS_NS
        .iter()
        .position(|&bound| latency_ns <= bound)
        .unwrap_or(RESTORE_BUCKETS_NS.len())
}

fn buckets_sum(metrics: &ds41rt_hostcache::metrics::Snapshot) -> u64 {
    metrics.store_latency_buckets.iter().sum()
}

/// One page per compressor, no draft, over the pool suites' layout: 21 copies (16 page
/// segments, 4 tail segments, 1 scores), so the modelled duration is an exact number.
fn probe_snapshot(device: &mut Device, generation: u32) -> DeviceSnapshot {
    snapshot(device, SnapshotKind::Turn, &tokens(8), 1, false, generation)
}

#[test]
fn device_evictions_counts_every_consultation() {
    let mut cache = default_cache(4 * CHUNK as u64, StoreMode::OnRetain);
    // No ticket: the engine evicts a snapshot the cache never saw — still one consultation.
    assert_eq!(cache.before_device_evict(None), EvictDecision::Clean);
    assert_eq!(cache.metrics().device_evictions, 1);
    // An unknown ticket resolves clean and counts too.
    assert_eq!(
        cache.before_device_evict(Some(ds41rt_hostcache::cache::StoreTicket(99))),
        EvictDecision::Clean
    );
    assert_eq!(cache.metrics().device_evictions, 2);
    assert_eq!(cache.metrics().evict_waits, 0);
    assert_eq!(cache.metrics().evict_drops_uncached, 0);
}

#[test]
fn device_evictions_once_per_eviction_outcome() {
    // Waited clean: the pending copy completes within the budget.
    let mut cache = default_cache(4 * CHUNK as u64, StoreMode::OnRetain);
    let mut device = Device::new(DEVICE_BYTES);
    let snap = probe_snapshot(&mut device, 1);
    write_snapshot(cache.engine_mut(), &snap);
    let StoreOutcome::Issued(ticket) = cache.store(&snap, 1) else {
        panic!("expected an issued store");
    };
    assert!(matches!(
        cache.before_device_evict(Some(ticket)),
        EvictDecision::WaitedClean { .. }
    ));
    let metrics = cache.metrics();
    assert_eq!(metrics.device_evictions, 1);
    assert!(metrics.device_evictions >= metrics.evict_waits);
    assert!(metrics.device_evictions >= metrics.evict_drops_uncached);

    // Already complete: tick committed the store, so the evict path resolves clean.
    let mut cache = default_cache(4 * CHUNK as u64, StoreMode::OnRetain);
    let mut device = Device::new(DEVICE_BYTES);
    let snap = probe_snapshot(&mut device, 1);
    write_snapshot(cache.engine_mut(), &snap);
    let StoreOutcome::Issued(ticket) = cache.store(&snap, 1) else {
        panic!("expected an issued store");
    };
    settle(&mut cache);
    cache.tick();
    assert_eq!(
        cache.before_device_evict(Some(ticket)),
        EvictDecision::Clean
    );
    assert_eq!(cache.metrics().device_evictions, 1);

    // Dropped uncached: the store stream stalls, so the copy never lands.
    let mut cache = default_cache(4 * CHUNK as u64, StoreMode::OnRetain);
    let mut device = Device::new(DEVICE_BYTES);
    let stalled = probe_snapshot(&mut device, 1);
    write_snapshot(cache.engine_mut(), &stalled);
    cache
        .engine_mut()
        .inject(CopyFault::StreamStalls(Stream::Store));
    let StoreOutcome::Issued(ticket) = cache.store(&stalled, 1) else {
        panic!("expected an issued store");
    };
    assert_eq!(
        cache.before_device_evict(Some(ticket)),
        EvictDecision::DroppedUncached
    );
    let metrics = cache.metrics();
    assert_eq!(metrics.device_evictions, 1);
    assert_eq!(metrics.evict_drops_uncached, 1);
    assert!(metrics.device_evictions >= metrics.evict_drops_uncached);

    // Budget timeout: the copies are modelled far slower than the copy budget.
    let slow = CopyModel {
        d2h_bytes_per_ns: 0.001,
        ..CopyModel::default()
    };
    let mut config = config(4 * CHUNK as u64, StoreMode::OnRetain);
    config.copy_budget_ns = 1_000;
    let mut cache = build_cache(config, slow, DEVICE_BYTES);
    let mut device = Device::new(DEVICE_BYTES);
    let slow_snap = probe_snapshot(&mut device, 1);
    write_snapshot(cache.engine_mut(), &slow_snap);
    let StoreOutcome::Issued(ticket) = cache.store(&slow_snap, 1) else {
        panic!("expected an issued store");
    };
    assert_eq!(
        cache.before_device_evict(Some(ticket)),
        EvictDecision::DroppedUncached
    );
    let metrics = cache.metrics();
    assert_eq!(metrics.device_evictions, 1);
    assert_eq!(metrics.evict_waits, 1);
    assert_eq!(metrics.evict_drops_uncached, 1);
}

#[test]
fn store_latency_matches_the_modelled_copy_time() {
    let model = CopyModel::default();
    let mut cache = default_cache(4 * CHUNK as u64, StoreMode::OnRetain);
    let mut device = Device::new(DEVICE_BYTES);
    let snap = probe_snapshot(&mut device, 1);
    write_snapshot(cache.engine_mut(), &snap);
    let expected = modelled_store_ns(model, &snap);

    // Issue at t0 = 0: advance past the modelled duration, then tick.
    let StoreOutcome::Issued(_) = cache.store(&snap, 1) else {
        panic!("expected an issued store");
    };
    cache.engine_mut().advance(expected);
    let report = cache.tick();
    assert_eq!(report.completed.len(), 1);
    let metrics = cache.metrics();
    assert_eq!(metrics.store_latency_sum_ns, expected);
    assert_eq!(buckets_sum(&metrics), 1);
    assert_eq!(
        metrics.store_latency_buckets[expected_bucket(expected)],
        1,
        "latency {expected} ns belongs in bucket {}",
        expected_bucket(expected)
    );

    // Issue at t0 > 0: the latency is the completion clock minus the issue clock, so the
    // same modelled duration lands in the same bucket with the same sum delta.
    let t0 = 7_000_000u64;
    cache.engine_mut().advance(t0);
    let StoreOutcome::Issued(_) = cache.store(&snap, 1) else {
        panic!("expected an issued store");
    };
    cache.engine_mut().advance(expected);
    cache.tick();
    let metrics = cache.metrics();
    assert_eq!(metrics.stores_completed, 2);
    assert_eq!(metrics.store_latency_sum_ns, 2 * expected);
    assert_eq!(buckets_sum(&metrics), 2);
}

#[test]
fn store_latency_lands_past_the_first_bucket() {
    // A modelled latency in the 10–50 ms band must land in bucket 3, not bucket 0: the
    // placement is by value, not by "any copy fits the first bound". A store is one batch
    // since HC-8, so the band is set by the per-submission latency alone.
    let model = CopyModel {
        per_copy_latency_ns: 20_000_000,
        ..CopyModel::default()
    };
    let config = config(4 * CHUNK as u64, StoreMode::OnRetain);
    let engine =
        ds41rt_hostcache::copy::StubCopyEngine::new(model, DEVICE_BYTES, config.bytes as usize);
    let mut cache =
        ds41rt_hostcache::cache::HostCache::new(config, layout(), engine).expect("cache");
    let mut device = Device::new(DEVICE_BYTES);
    let snap = probe_snapshot(&mut device, 1);
    write_snapshot(cache.engine_mut(), &snap);
    let expected = modelled_store_ns(model, &snap);
    assert!(
        (10_000_000..50_000_000).contains(&expected),
        "probe latency {expected} ns must stay in the 10–50 ms band"
    );
    let StoreOutcome::Issued(_) = cache.store(&snap, 1) else {
        panic!("expected an issued store");
    };
    cache.engine_mut().advance(expected);
    cache.tick();
    let metrics = cache.metrics();
    assert_eq!(metrics.store_latency_sum_ns, expected);
    assert_eq!(buckets_sum(&metrics), 1);
    assert_eq!(metrics.store_latency_buckets[expected_bucket(expected)], 1);
}

#[test]
fn failed_store_and_drain_timeout_record_no_latency() {
    // An issue failure: tick reports the ticket failed; no copy ever completed.
    let mut cache = default_cache(4 * CHUNK as u64, StoreMode::OnRetain);
    let mut device = Device::new(DEVICE_BYTES);
    let snap = probe_snapshot(&mut device, 1);
    write_snapshot(cache.engine_mut(), &snap);
    cache
        .engine_mut()
        .inject(CopyFault::IssueFails(Stream::Store));
    let StoreOutcome::Issued(ticket) = cache.store(&snap, 1) else {
        panic!("expected an issued store");
    };
    let report = cache.tick();
    assert_eq!(report.failed, vec![ticket]);
    let metrics = cache.metrics();
    assert_eq!(metrics.stores_failed, 1);
    assert_eq!(metrics.stores_completed, 0);
    assert_eq!(metrics.store_latency_sum_ns, 0);
    assert_eq!(buckets_sum(&metrics), 0);

    // A drain timeout: the mid-plan failure's copies do not drain within the budget, the
    // slabs stay held, and the failure records no latency either.
    let slow = CopyModel {
        d2h_bytes_per_ns: 0.001,
        ..CopyModel::default()
    };
    let mut config = config(4 * CHUNK as u64, StoreMode::OnRetain);
    config.copy_budget_ns = 1_000;
    let engine = FailAfter::new(
        ds41rt_hostcache::copy::StubCopyEngine::new(slow, DEVICE_BYTES, config.bytes as usize),
        1,
    );
    let mut cache =
        ds41rt_hostcache::cache::HostCache::new(config, layout(), engine).expect("cache");
    let mut device = Device::new(DEVICE_BYTES);
    let snap = probe_snapshot(&mut device, 1);
    write_snapshot(cache.engine_mut().inner_mut(), &snap);
    let StoreOutcome::Issued(ticket) = cache.store(&snap, 1) else {
        panic!("expected an issued store");
    };
    let report = cache.tick();
    assert_eq!(report.failed, vec![ticket]);
    let metrics = cache.metrics();
    assert_eq!(metrics.stores_failed, 1);
    assert_eq!(metrics.store_drain_timeouts, 1);
    assert_eq!(metrics.stores_completed, 0);
    assert_eq!(metrics.store_latency_sum_ns, 0);
    assert_eq!(buckets_sum(&metrics), 0);
}

#[test]
fn a_replaced_store_records_latency_exactly_once() {
    for mode in [StoreMode::OnRetain, StoreMode::OnEvict] {
        let model = CopyModel::default();
        let mut cache = default_cache(4 * CHUNK as u64, mode);
        let mut device = Device::new(DEVICE_BYTES);
        let first = probe_snapshot(&mut device, 1);
        write_snapshot(cache.engine_mut(), &first);
        let expected = modelled_store_ns(model, &first);
        let (StoreOutcome::Issued(first_ticket) | StoreOutcome::Deferred(first_ticket)) =
            cache.store(&first, 1)
        else {
            panic!("expected a ticket");
        };
        // A same-kind, same-tokens store with fresh device identities replaces the first.
        // Tick between the stores so each completion is observed exactly at its modelled
        // duration (back-to-back stores would chain on the stream and the first completion
        // would be observed late).
        if mode == StoreMode::OnRetain {
            cache.engine_mut().advance(expected);
            let report = cache.tick();
            assert_eq!(report.completed, vec![first_ticket]);
        }
        let second = probe_snapshot(&mut device, 2);
        write_snapshot(cache.engine_mut(), &second);
        let (StoreOutcome::Issued(second_ticket) | StoreOutcome::Deferred(second_ticket)) =
            cache.store(&second, 1)
        else {
            panic!("expected a ticket");
        };
        if mode == StoreMode::OnRetain {
            cache.engine_mut().advance(expected);
            let report = cache.tick();
            assert_eq!(report.completed, vec![second_ticket]);
        } else {
            for ticket in [first_ticket, second_ticket] {
                assert!(matches!(
                    cache.before_device_evict(Some(ticket)),
                    EvictDecision::WaitedClean { .. }
                ));
            }
        }
        let metrics = cache.metrics();
        assert_eq!(metrics.stores_completed, 2);
        assert_eq!(metrics.stores_replaced, 1);
        assert_eq!(
            buckets_sum(&metrics),
            2,
            "each completion records latency exactly once"
        );
        assert_eq!(metrics.store_latency_sum_ns, 2 * expected);
    }
}

#[test]
fn on_evict_records_latency_at_the_evict_commit() {
    // Under OnEvict the copy is issued by `before_device_evict`, not by `store`; the latency
    // is still the modelled copy time, recorded once at the commit the evict path drives.
    let model = CopyModel::default();
    let mut cache = default_cache(4 * CHUNK as u64, StoreMode::OnEvict);
    let mut device = Device::new(DEVICE_BYTES);
    let snap = probe_snapshot(&mut device, 1);
    write_snapshot(cache.engine_mut(), &snap);
    let expected = modelled_store_ns(model, &snap);
    let StoreOutcome::Deferred(ticket) = cache.store(&snap, 1) else {
        panic!("expected a deferred store");
    };
    assert_eq!(buckets_sum(&cache.metrics()), 0);
    assert!(matches!(
        cache.before_device_evict(Some(ticket)),
        EvictDecision::WaitedClean { .. }
    ));
    let metrics = cache.metrics();
    assert_eq!(metrics.stores_completed, 1);
    assert_eq!(metrics.store_latency_sum_ns, expected);
    assert_eq!(buckets_sum(&metrics), 1);
    assert_eq!(metrics.device_evictions, 1);
}

#[test]
fn disabled_cache_records_nothing() {
    let mut cache = default_cache(0, StoreMode::OnRetain);
    assert!(!cache.enabled());
    let mut device = Device::new(DEVICE_BYTES);
    let snap = snapshot(&mut device, SnapshotKind::Turn, &tokens(8), 1, false, 1);
    write_snapshot(cache.engine_mut(), &snap);
    assert!(matches!(
        cache.store(&snap, 1),
        StoreOutcome::Skipped(ds41rt_hostcache::cache::SkipReason::KindOff)
    ));
    settle(&mut cache);
    assert_eq!(cache.tick(), ds41rt_hostcache::cache::TickReport::default());
    assert_eq!(cache.before_device_evict(None), EvictDecision::Clean);
    let metrics = cache.metrics();
    assert_eq!(metrics.device_evictions, 0);
    assert_eq!(metrics.evict_waits, 0);
    assert_eq!(metrics.evict_drops_uncached, 0);
    assert_eq!(metrics.store_latency_sum_ns, 0);
    assert_eq!(buckets_sum(&metrics), 0);
    assert_eq!(metrics.stores_completed, 0);
}

/// Order-of-magnitude floor: the instrumentation adds two u64 reads and one bucket index per
/// store; a broken fast path fails `cargo test`, not a human. Debug build, so the bound is
/// generous (the HC-5 floor test for the same shape allows five seconds).
#[test]
fn store_tick_instrumentation_stays_within_the_floor() {
    let mut cache = default_cache(64 * CHUNK as u64, StoreMode::OnRetain);
    let mut device = Device::new(DEVICE_BYTES);
    let start = std::time::Instant::now();
    for generation in 0..1000u32 {
        let snapshot = snapshot(
            &mut device,
            SnapshotKind::Turn,
            &tokens(4096),
            1,
            false,
            generation,
        );
        write_snapshot(cache.engine_mut(), &snapshot);
        let _ = cache.store(&snapshot, generation as u64);
        settle(&mut cache);
        cache.tick();
    }
    let elapsed = start.elapsed();
    assert!(
        elapsed < std::time::Duration::from_secs(5),
        "1000 instrumented stores+tick took {elapsed:?}"
    );
    let metrics = cache.metrics();
    assert_eq!(buckets_sum(&metrics), metrics.stores_completed);
    assert_eq!(metrics.stores_completed, 1000);
}

#[derive(Clone, Debug)]
enum Op {
    Store { tokens: Vec<u32>, pages: usize },
    Tick,
    Evict,
    Stall,
}

fn operation() -> impl Strategy<Value = Op> {
    prop_oneof![
        (prop::collection::vec(0u32..4, 1..8), 0usize..3,)
            .prop_map(|(tokens, pages)| Op::Store { tokens, pages }),
        Just(Op::Tick),
        Just(Op::Evict),
        Just(Op::Stall),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]
    #[test]
    fn random_sequences_keep_the_instrumentation_invariants(
        ops in prop::collection::vec(operation(), 1..40),
    ) {
        for mode in [StoreMode::OnRetain, StoreMode::OnEvict] {
            let mut cache = default_cache(8 * CHUNK as u64, mode);
            let mut device = Device::new(DEVICE_BYTES);
            let mut pending: Vec<ds41rt_hostcache::cache::StoreTicket> = Vec::new();
            let mut evict_calls = 0u64;
            let mut generation = 0u32;

            for op in ops.clone() {
                match op {
                    Op::Store { tokens, pages } => {
                        generation += 1;
                        let snap = snapshot(
                            &mut device,
                            SnapshotKind::Turn,
                            &tokens,
                            pages,
                            false,
                            generation,
                        );
                        write_snapshot(cache.engine_mut(), &snap);
                        if let StoreOutcome::Issued(ticket)
                        | StoreOutcome::Deferred(ticket) = cache.store(&snap, generation as u64)
                        {
                            pending.push(ticket);
                        }
                    }
                    Op::Tick => {
                        // A random advance so completions land at varying latencies.
                        cache.engine_mut().advance(100_000);
                        let report = cache.tick();
                        for ticket in report.completed.iter().chain(report.failed.iter()) {
                            let index = pending
                                .iter()
                                .position(|candidate| candidate == ticket)
                                .expect("reported ticket was pending");
                            pending.remove(index);
                        }
                    }
                    Op::Evict => {
                        evict_calls += 1;
                        let ticket = pending.pop();
                        let _ = cache.before_device_evict(ticket);
                        // A dropped uncached store is reported by no later tick; a clean or
                        // waited one committed already. Either way the ticket is settled.
                    }
                    Op::Stall => {
                        // Wedge the store stream so subsequent issues fail or never land.
                        cache.engine_mut().inject(CopyFault::StreamStalls(Stream::Store));
                    }
                }
                let metrics = cache.metrics();
                prop_assert!(
                    metrics.device_evictions >= metrics.evict_waits,
                    "device_evictions {} < evict_waits {}",
                    metrics.device_evictions,
                    metrics.evict_waits
                );
                prop_assert!(
                    metrics.device_evictions >= metrics.evict_drops_uncached,
                    "device_evictions {} < evict_drops_uncached {}",
                    metrics.device_evictions,
                    metrics.evict_drops_uncached
                );
                prop_assert_eq!(buckets_sum(&metrics), metrics.stores_completed,
                    "every completion records latency exactly once");
            }

            // Quiescence: settle everything, then the histogram accounts for every completed
            // store and the eviction counter for every consultation.
            settle(&mut cache);
            let report = cache.tick();
            for ticket in report.completed.iter().chain(report.failed.iter()) {
                let index = pending
                    .iter()
                    .position(|candidate| candidate == ticket)
                    .expect("reported ticket was pending");
                pending.remove(index);
            }
            while let Some(ticket) = pending.pop() {
                evict_calls += 1;
                let _ = cache.before_device_evict(Some(ticket));
            }
            let metrics = cache.metrics();
            prop_assert_eq!(metrics.device_evictions, evict_calls,
                "one device_eviction per before_device_evict call");
            prop_assert_eq!(buckets_sum(&metrics), metrics.stores_completed,
                "at quiescence the histogram holds exactly one entry per completed store");
            prop_assert!(
                metrics.store_latency_sum_ns > 0 || metrics.stores_completed == 0,
                "a completion records a positive latency"
            );
        }
    }
}
