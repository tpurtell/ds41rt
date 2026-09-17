//! Functional suite for packet HC-5: the cache facade's five calls on their happy paths, every
//! documented error path (skips, exhaustion, issue failure, budget timeouts), the disabled
//! no-ops, exact sharing end to end, content fidelity, the late-overwrite hazard, and proptest
//! properties over random store/tick/evict/lookup/restore sequences against the HC-2 model.
mod common;
mod common_hc_5;

use common::{reconcile, Model};
use common_hc_5::{
    cache, config, default_cache, device_pages, restored_bytes, settle, snapshot, store_resident,
    stored_bytes, target, tokens, write_snapshot, Device, FailAfter, Payload, DEVICE_BYTES,
};
use ds41rt_hostcache::cache::{
    DeviceSnapshot, EvictDecision, HostCache, RestoreOutcome, SkipReason, StoreOutcome,
    StoreTicket, TickReport,
};
use ds41rt_hostcache::config::StoreMode;
use ds41rt_hostcache::copy::{CopyFault, CopyModel, DeviceRange, Stream, StubCopyEngine};
use ds41rt_hostcache::pool::testing::{layout, CHUNK};
use ds41rt_hostcache::pool::Layout;
use ds41rt_hostcache::snapshot::{DevicePageId, Key};
use ds41rt_hostcache::SnapshotKind;
use proptest::prelude::*;
use std::collections::{HashMap, HashSet};

#[test]
fn store_tick_lookup_restore_happy_path() {
    for mode in [StoreMode::OnRetain, StoreMode::OnEvict] {
        happy_path(mode);
    }
}

fn happy_path(mode: StoreMode) {
    let mut cache = default_cache(4 * CHUNK as u64, mode);
    let mut device = Device::new(DEVICE_BYTES);
    let snapshot = snapshot(&mut device, SnapshotKind::Turn, &tokens(8), 2, true, 1);
    write_snapshot(cache.engine_mut(), &snapshot);
    let expected = stored_bytes(cache.engine_mut(), &snapshot);

    let _ticket = store_resident(&mut cache, &snapshot, 42);
    assert_eq!(cache.metrics().stores_issued, 1);
    assert_eq!(cache.metrics().stores_completed, 1);
    assert!(cache.metrics().bytes_used > 0);

    let hit = cache.lookup(&tokens(8)).expect("host hit");
    assert_eq!(hit.kind, SnapshotKind::Turn);
    assert_eq!(hit.common, 8);
    assert_eq!(cache.payload(hit.key), Some(&42));
    assert_eq!(cache.snapshot_tokens(hit.key), Some(tokens(8).as_slice()));

    let target = target(&mut device, &snapshot, 2);
    let RestoreOutcome::Done { bytes, .. } = cache.restore(hit.key, &target) else {
        panic!("expected a completed restore");
    };
    assert!(bytes > 0);
    assert_eq!(restored_bytes(cache.engine_mut(), &target), expected);
    assert_eq!(cache.metrics().restores, 1);
    assert_eq!(cache.metrics().restore_bytes, bytes);
    assert_eq!(
        cache.metrics().restore_latency_buckets.iter().sum::<u64>(),
        1
    );
}

#[test]
fn store_skips_by_size_kind_and_shape() {
    let mut config = config(4 * CHUNK as u64, StoreMode::OnRetain);
    config.min_tokens = 512;
    config.max_tokens = 1024;
    config.kinds.prompt = false;
    let mut cache = cache(config, CopyModel::default(), DEVICE_BYTES);
    let mut device = Device::new(DEVICE_BYTES);

    let small = snapshot(&mut device, SnapshotKind::Turn, &tokens(8), 1, false, 1);
    assert_eq!(
        cache.store(&small, 1),
        StoreOutcome::Skipped(SkipReason::TooSmall)
    );
    let large = snapshot(&mut device, SnapshotKind::Turn, &tokens(2048), 1, false, 2);
    assert_eq!(
        cache.store(&large, 2),
        StoreOutcome::Skipped(SkipReason::TooLarge)
    );
    let prompt = snapshot(&mut device, SnapshotKind::Prompt, &tokens(600), 1, false, 3);
    assert_eq!(
        cache.store(&prompt, 3),
        StoreOutcome::Skipped(SkipReason::KindOff)
    );
    let mut malformed = snapshot(&mut device, SnapshotKind::Turn, &tokens(600), 1, false, 4);
    malformed.pages[0][0].segments = vec![DeviceRange {
        addr: 0,
        bytes: 8192,
    }];
    assert_eq!(
        cache.store(&malformed, 4),
        StoreOutcome::Skipped(SkipReason::Malformed)
    );
    assert_eq!(cache.metrics().stores_skipped, 4);
    assert_eq!(cache.metrics().stores_issued, 0);
    assert_eq!(cache.metrics().bytes_used, 0);
}

#[test]
fn pool_exhaustion_skips_without_holding_anything() {
    // Two chunks: the tail and scores classes take one each, leaving none for a page.
    let mut cache = default_cache(2 * CHUNK as u64, StoreMode::OnRetain);
    let mut device = Device::new(DEVICE_BYTES);
    let snapshot = snapshot(&mut device, SnapshotKind::Turn, &tokens(8), 1, false, 1);
    assert_eq!(
        cache.store(&snapshot, 1),
        StoreOutcome::Skipped(SkipReason::Exhausted)
    );
    assert_eq!(cache.metrics().stores_skipped, 1);
    assert_eq!(cache.metrics().bytes_used, 0);
}

#[test]
fn disabled_cache_is_a_no_op() {
    let mut cache = default_cache(0, StoreMode::OnRetain);
    assert!(!cache.enabled());
    let mut device = Device::new(DEVICE_BYTES);
    let snapshot = snapshot(&mut device, SnapshotKind::Turn, &tokens(8), 1, false, 1);
    assert_eq!(
        cache.store(&snapshot, 1),
        StoreOutcome::Skipped(SkipReason::KindOff)
    );
    assert_eq!(cache.tick(), TickReport::default());
    assert_eq!(cache.before_device_evict(None), EvictDecision::Clean);
    assert_eq!(
        cache.before_device_evict(Some(StoreTicket(0))),
        EvictDecision::Clean
    );
    assert!(cache.lookup(&tokens(8)).is_none());
    assert!(cache.payload(0).is_none());
    assert!(cache.snapshot_tokens(0).is_none());
    assert_eq!(
        cache.restore(0, &target(&mut device, &snapshot, 2)),
        RestoreOutcome::Failed
    );
    let metrics = cache.metrics();
    assert_eq!(metrics.bytes_used, 0);
    assert_eq!(metrics.quota_bytes, 0);
    assert_eq!(metrics.stores_issued, 0);
    assert_eq!(metrics.lookups, 0);
}

#[test]
fn shared_device_pages_are_copied_once() {
    for mode in [StoreMode::OnRetain, StoreMode::OnEvict] {
        let mut cache = default_cache(8 * CHUNK as u64, mode);
        let mut device = Device::new(DEVICE_BYTES);
        let first = snapshot(&mut device, SnapshotKind::Turn, &tokens(8), 2, false, 1);
        write_snapshot(cache.engine_mut(), &first);
        store_resident(&mut cache, &first, 1);
        assert_eq!(cache.metrics().pages_copied, 8);

        // The second snapshot shares the first two pages of every compressor and adds one.
        let second = snapshot(&mut device, SnapshotKind::Turn, &tokens(9), 3, false, 1);
        write_snapshot(cache.engine_mut(), &second);
        store_resident(&mut cache, &second, 2);
        assert_eq!(cache.metrics().pages_copied, 12);
        assert_eq!(cache.metrics().pages_shared, 8);
    }
}

#[test]
fn content_fidelity_through_store_and_restore() {
    for mode in [StoreMode::OnRetain, StoreMode::OnEvict] {
        let mut cache = default_cache(4 * CHUNK as u64, mode);
        let mut device = Device::new(DEVICE_BYTES);
        let snapshot = snapshot(&mut device, SnapshotKind::Prompt, &tokens(16), 2, true, 1);
        write_snapshot(cache.engine_mut(), &snapshot);
        let expected = stored_bytes(cache.engine_mut(), &snapshot);
        store_resident(&mut cache, &snapshot, 7);
        let key = cache.lookup(&tokens(16)).expect("host hit").key;
        let target = target(&mut device, &snapshot, 2);
        assert!(matches!(
            cache.restore(key, &target),
            RestoreOutcome::Done { .. }
        ));
        assert_eq!(restored_bytes(cache.engine_mut(), &target), expected);
    }
}

#[test]
fn late_overwrite_before_tick_is_what_restores() {
    let mut cache = default_cache(4 * CHUNK as u64, StoreMode::OnRetain);
    let mut device = Device::new(DEVICE_BYTES);
    let snapshot = snapshot(&mut device, SnapshotKind::Turn, &tokens(8), 1, false, 1);
    write_snapshot(cache.engine_mut(), &snapshot);
    let StoreOutcome::Issued(_) = cache.store(&snapshot, 1) else {
        panic!("expected an issued store");
    };
    // The copy reads its source when it executes, so an overwrite before `tick` is what lands.
    let overwritten = snapshot.pages[0][0].segments[0];
    let new_bytes: Vec<u8> = (0..overwritten.bytes).map(|i| 0xAB ^ i as u8).collect();
    cache.engine_mut().write_device(overwritten, &new_bytes);
    settle(&mut cache);
    cache.tick();
    let key = cache.lookup(&tokens(8)).expect("host hit").key;
    let target = target(&mut device, &snapshot, 2);
    assert!(matches!(
        cache.restore(key, &target),
        RestoreOutcome::Done { .. }
    ));
    let restored = restored_bytes(cache.engine_mut(), &target);
    assert_eq!(&restored[..overwritten.bytes], new_bytes.as_slice());
}

#[test]
fn before_device_evict_waits_or_drops() {
    let mut cache = default_cache(4 * CHUNK as u64, StoreMode::OnRetain);
    assert_eq!(cache.before_device_evict(None), EvictDecision::Clean);

    let mut device = Device::new(DEVICE_BYTES);
    let snap = snapshot(&mut device, SnapshotKind::Turn, &tokens(8), 1, false, 1);
    write_snapshot(cache.engine_mut(), &snap);
    let StoreOutcome::Issued(ticket) = cache.store(&snap, 1) else {
        panic!("expected an issued store");
    };
    assert!(matches!(
        cache.before_device_evict(Some(ticket)),
        EvictDecision::WaitedClean { .. }
    ));
    assert_eq!(cache.metrics().stores_completed, 1);
    assert_eq!(cache.metrics().evict_waits, 1);
    assert!(cache.metrics().evict_wait_ns > 0);
    // A ticket that is no longer pending is clean.
    assert_eq!(
        cache.before_device_evict(Some(ticket)),
        EvictDecision::Clean
    );

    let mut cache = default_cache(4 * CHUNK as u64, StoreMode::OnRetain);
    let mut device = Device::new(DEVICE_BYTES);
    let stalled = snapshot(&mut device, SnapshotKind::Turn, &tokens(8), 1, false, 1);
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
    assert_eq!(cache.metrics().stores_failed, 1);
    assert_eq!(cache.metrics().evict_drops_uncached, 1);
    assert_eq!(cache.metrics().bytes_used, 0);
}

#[test]
fn an_issue_failure_is_reported_by_tick() {
    let mut cache = default_cache(4 * CHUNK as u64, StoreMode::OnRetain);
    let mut device = Device::new(DEVICE_BYTES);
    let snapshot = snapshot(&mut device, SnapshotKind::Turn, &tokens(8), 1, false, 1);
    write_snapshot(cache.engine_mut(), &snapshot);
    cache
        .engine_mut()
        .inject(CopyFault::IssueFails(Stream::Store));
    let StoreOutcome::Issued(ticket) = cache.store(&snapshot, 1) else {
        panic!("expected an issued store");
    };
    let report = cache.tick();
    assert_eq!(report.failed, vec![ticket]);
    assert!(report.completed.is_empty());
    assert_eq!(cache.metrics().stores_failed, 1);
    assert_eq!(cache.metrics().bytes_used, 0);
}

#[test]
fn restore_timeout_leaves_the_snapshot_resident() {
    for mode in [StoreMode::OnRetain, StoreMode::OnEvict] {
        let slow = CopyModel {
            d2h_bytes_per_ns: 25.0,
            h2d_bytes_per_ns: 0.001,
            per_copy_latency_ns: 10_000,
        };
        let mut config = config(4 * CHUNK as u64, mode);
        config.restore_budget_ns = 1_000;
        let mut cache = cache(config, slow, DEVICE_BYTES);
        let mut device = Device::new(DEVICE_BYTES);
        let snapshot = snapshot(&mut device, SnapshotKind::Turn, &tokens(8), 1, false, 1);
        write_snapshot(cache.engine_mut(), &snapshot);
        store_resident(&mut cache, &snapshot, 1);
        let key = cache.lookup(&tokens(8)).expect("host hit").key;
        let target = target(&mut device, &snapshot, 2);
        assert_eq!(cache.restore(key, &target), RestoreOutcome::TimedOut);
        assert!(cache.snapshot_tokens(key).is_some());
        assert_eq!(cache.metrics().restore_timeouts, 1);
        assert_eq!(cache.metrics().restores, 0);
    }
}

#[test]
fn a_mismatched_restore_target_fails() {
    for mode in [StoreMode::OnRetain, StoreMode::OnEvict] {
        let mut cache = default_cache(4 * CHUNK as u64, mode);
        let mut device = Device::new(DEVICE_BYTES);
        let snapshot = snapshot(&mut device, SnapshotKind::Turn, &tokens(8), 1, false, 1);
        write_snapshot(cache.engine_mut(), &snapshot);
        store_resident(&mut cache, &snapshot, 1);
        let key = cache.lookup(&tokens(8)).expect("host hit").key;
        let mut target = target(&mut device, &snapshot, 2);
        target.pages[0].pop();
        assert_eq!(cache.restore(key, &target), RestoreOutcome::Failed);
        assert_eq!(cache.metrics().restore_failures, 1);
        assert!(cache.snapshot_tokens(key).is_some());
    }
}

#[test]
fn on_evict_defers_until_the_device_evicts() {
    let mut cache = default_cache(4 * CHUNK as u64, StoreMode::OnEvict);
    let mut device = Device::new(DEVICE_BYTES);
    let snapshot = snapshot(&mut device, SnapshotKind::Turn, &tokens(8), 1, false, 1);
    write_snapshot(cache.engine_mut(), &snapshot);
    let StoreOutcome::Deferred(ticket) = cache.store(&snapshot, 1) else {
        panic!("expected a deferred store");
    };
    // Nothing is planned or copied until the engine evicts the snapshot.
    assert_eq!(cache.engine_mut().pending(Stream::Store), 0);
    assert_eq!(cache.metrics().bytes_used, 0);
    assert_eq!(cache.tick(), TickReport::default());
    assert!(cache.lookup(&tokens(8)).is_none());
    assert!(matches!(
        cache.before_device_evict(Some(ticket)),
        EvictDecision::WaitedClean { .. }
    ));
    assert!(cache.lookup(&tokens(8)).is_some());
    assert_eq!(cache.metrics().stores_completed, 1);
}

#[test]
fn payload_follows_the_snapshot() {
    for mode in [StoreMode::OnRetain, StoreMode::OnEvict] {
        let mut cache = default_cache(4 * CHUNK as u64, mode);
        let mut device = Device::new(DEVICE_BYTES);
        let first = snapshot(&mut device, SnapshotKind::Turn, &tokens(8), 1, false, 1);
        write_snapshot(cache.engine_mut(), &first);
        store_resident(&mut cache, &first, 10);
        let key = cache.lookup(&tokens(8)).expect("host hit").key;
        assert_eq!(cache.payload(key), Some(&10));

        // A same-tokens store replaces the snapshot and drops the old payload.
        let second = snapshot(&mut device, SnapshotKind::Turn, &tokens(8), 1, false, 2);
        write_snapshot(cache.engine_mut(), &second);
        store_resident(&mut cache, &second, 20);
        let new_key = cache.lookup(&tokens(8)).expect("host hit").key;
        assert_ne!(new_key, key);
        assert_eq!(cache.payload(new_key), Some(&20));
        assert_eq!(cache.payload(key), None);
        assert_eq!(cache.metrics().stores_replaced, 1);
    }
}

#[test]
fn host_eviction_keeps_bytes_within_quota() {
    for mode in [StoreMode::OnRetain, StoreMode::OnEvict] {
        let quota = 4 * CHUNK as u64;
        let mut cache = default_cache(quota, mode);
        let mut device = Device::new(DEVICE_BYTES);
        for generation in 0..40u32 {
            let snapshot = snapshot(
                &mut device,
                SnapshotKind::Turn,
                &tokens(8 + generation as usize),
                1,
                false,
                generation,
            );
            write_snapshot(cache.engine_mut(), &snapshot);
            store_resident(&mut cache, &snapshot, generation as u64);
            assert!(cache.metrics().bytes_used <= quota);
        }
        assert!(cache.metrics().host_evictions > 0);
        assert!(cache.metrics().resident_snapshots > 0);
        assert!(cache.metrics().host_evicted_bytes > 0);
    }
}

#[test]
fn a_restored_page_is_shared_by_a_later_store() {
    for mode in [StoreMode::OnRetain, StoreMode::OnEvict] {
        let mut cache = default_cache(8 * CHUNK as u64, mode);
        let mut device = Device::new(DEVICE_BYTES);
        let snapshot = snapshot(&mut device, SnapshotKind::Turn, &tokens(8), 1, false, 1);
        write_snapshot(cache.engine_mut(), &snapshot);
        store_resident(&mut cache, &snapshot, 1);
        let copied = cache.metrics().pages_copied;
        let key = cache.lookup(&tokens(8)).expect("host hit").key;
        let target = target(&mut device, &snapshot, 2);
        assert!(matches!(
            cache.restore(key, &target),
            RestoreOutcome::Done { .. }
        ));
        // A store of the restored identities shares every page: nothing is copied again.
        let restored = DeviceSnapshot {
            meta: snapshot.meta.clone(),
            pages: target.pages.clone(),
            tail: target.tail.clone(),
            draft: target.draft.clone(),
            scores: target.scores.clone(),
        };
        store_resident(&mut cache, &restored, 2);
        assert_eq!(cache.metrics().pages_copied, copied);
    }
}

#[test]
fn device_page_freed_forgets_the_share() {
    for mode in [StoreMode::OnRetain, StoreMode::OnEvict] {
        let mut cache = default_cache(8 * CHUNK as u64, mode);
        let mut device = Device::new(DEVICE_BYTES);
        let first = snapshot(&mut device, SnapshotKind::Turn, &tokens(8), 1, false, 1);
        write_snapshot(cache.engine_mut(), &first);
        store_resident(&mut cache, &first, 1);
        let copied = cache.metrics().pages_copied;
        cache.device_page_freed(first.pages[0][0].id);
        let second = snapshot(&mut device, SnapshotKind::Turn, &tokens(9), 1, false, 1);
        write_snapshot(cache.engine_mut(), &second);
        store_resident(&mut cache, &second, 2);
        // The freed identity is copied again; the other three pages stay shared.
        assert_eq!(cache.metrics().pages_copied, copied + 1);
    }
}

/// A class can be exhausted while `bytes_used` is under quota (chunks are carved per class).
/// Stores then evict the least recently used unpinned snapshot and retry, and only skip once
/// every resident snapshot is pinned.
#[test]
fn class_exhaustion_evicts_until_it_fits() {
    let quota = 3 * CHUNK as u64;
    let mut cache = default_cache(quota, StoreMode::OnRetain);
    let mut device = Device::new(DEVICE_BYTES);
    // The page class fills first: four one-page-per-compressor snapshots fill the page chunk.
    for generation in 0..4u32 {
        let snapshot = snapshot(
            &mut device,
            SnapshotKind::Turn,
            &tokens(8 + generation as usize),
            1,
            false,
            generation,
        );
        write_snapshot(cache.engine_mut(), &snapshot);
        let StoreOutcome::Issued(_) = cache.store(&snapshot, generation as u64) else {
            panic!("store {generation} skipped");
        };
        settle(&mut cache);
        cache.tick();
    }
    let evictions_before = cache.metrics().host_evictions;
    // Further stores keep succeeding by evicting, never skipping.
    for generation in 4..12u32 {
        let snapshot = snapshot(
            &mut device,
            SnapshotKind::Turn,
            &tokens(8 + generation as usize),
            1,
            false,
            generation,
        );
        write_snapshot(cache.engine_mut(), &snapshot);
        let StoreOutcome::Issued(_) = cache.store(&snapshot, generation as u64) else {
            panic!("store {generation} skipped");
        };
        settle(&mut cache);
        cache.tick();
        assert!(cache.metrics().bytes_used <= quota);
    }
    assert!(cache.metrics().host_evictions > evictions_before);

    // Pin every resident snapshot: a store now has nothing to evict and skips.
    for generation in 0..12u32 {
        if let Some(hit) = cache.lookup(&tokens(8 + generation as usize)) {
            cache.pin(hit.key);
        }
    }
    let snapshot = snapshot(&mut device, SnapshotKind::Turn, &tokens(64), 1, false, 99);
    write_snapshot(cache.engine_mut(), &snapshot);
    assert_eq!(
        cache.store(&snapshot, 99),
        StoreOutcome::Skipped(SkipReason::Exhausted)
    );
}

/// A stub that fails the `skip + 1`-th store-stream copy, with a configurable copy budget.
fn fail_after_cache(
    bytes: u64,
    model: CopyModel,
    skip: usize,
    copy_budget_ns: u64,
) -> HostCache<FailAfter, Payload> {
    let mut config = config(bytes, StoreMode::OnRetain);
    config.copy_budget_ns = copy_budget_ns;
    let engine = FailAfter::new(
        StubCopyEngine::new(model, DEVICE_BYTES, bytes as usize),
        skip,
    );
    HostCache::new(config, layout(), engine).expect("cache")
}

/// A mid-plan issue failure must drain the copies already issued before the plan's slabs are
/// released; when the drain times out the slabs stay held so no later store reuses them.
#[test]
fn a_mid_plan_issue_failure_drains_before_releasing_slabs() {
    // Fast copies: the already-issued copy drains within the budget, so the plan is released.
    let mut cache = fail_after_cache(4 * CHUNK as u64, CopyModel::default(), 1, 50_000_000);
    let mut device = Device::new(DEVICE_BYTES);
    let snap = snapshot(&mut device, SnapshotKind::Turn, &tokens(8), 2, false, 1);
    write_snapshot(cache.engine_mut().inner_mut(), &snap);
    let StoreOutcome::Issued(ticket) = cache.store(&snap, 1) else {
        panic!("expected an issued store");
    };
    let report = cache.tick();
    assert_eq!(report.failed, vec![ticket]);
    assert_eq!(cache.metrics().stores_failed, 1);
    assert_eq!(cache.metrics().store_drain_timeouts, 0);
    assert_eq!(cache.metrics().bytes_used, 0);

    // Slow copies and a tiny budget: the drain times out, so the plan's slabs stay held.
    let slow = CopyModel {
        d2h_bytes_per_ns: 0.001,
        h2d_bytes_per_ns: 25.0,
        per_copy_latency_ns: 10_000,
    };
    let mut cache = fail_after_cache(4 * CHUNK as u64, slow, 1, 1_000);
    let mut device = Device::new(DEVICE_BYTES);
    let snap = snapshot(&mut device, SnapshotKind::Turn, &tokens(8), 2, false, 1);
    write_snapshot(cache.engine_mut().inner_mut(), &snap);
    let StoreOutcome::Issued(ticket) = cache.store(&snap, 1) else {
        panic!("expected an issued store");
    };
    let held = cache.metrics().bytes_used;
    assert!(held > 0);
    let report = cache.tick();
    assert_eq!(report.failed, vec![ticket]);
    assert_eq!(cache.metrics().stores_failed, 1);
    assert_eq!(cache.metrics().store_drain_timeouts, 1);
    // The slabs stay held after the copy completes: no later store can reuse them.
    cache.engine_mut().inner_mut().advance(1_000_000_000);
    assert_eq!(cache.metrics().bytes_used, held);
}

/// Every shape inconsistency `parts_fit` documents is a `Malformed` skip.
#[test]
fn malformed_shapes_are_skipped() {
    let mut cache = default_cache(4 * CHUNK as u64, StoreMode::OnRetain);
    let mut device = Device::new(DEVICE_BYTES);

    // Draft present without `has_draft`.
    let mut snap = snapshot(&mut device, SnapshotKind::Turn, &tokens(8), 1, false, 1);
    snap.draft = Some(device.ranges(1, layout().draft));
    assert_eq!(
        cache.store(&snap, 1),
        StoreOutcome::Skipped(SkipReason::Malformed)
    );

    // Draft absent with `has_draft`.
    let mut snap = snapshot(&mut device, SnapshotKind::Turn, &tokens(8), 1, true, 2);
    snap.draft = None;
    assert_eq!(
        cache.store(&snap, 2),
        StoreOutcome::Skipped(SkipReason::Malformed)
    );

    // Empty tail.
    let mut snap = snapshot(&mut device, SnapshotKind::Turn, &tokens(8), 1, false, 3);
    snap.tail.clear();
    assert_eq!(
        cache.store(&snap, 3),
        StoreOutcome::Skipped(SkipReason::Malformed)
    );

    // Empty scores with the class enabled.
    let mut snap = snapshot(&mut device, SnapshotKind::Turn, &tokens(8), 1, false, 4);
    snap.scores.clear();
    assert_eq!(
        cache.store(&snap, 4),
        StoreOutcome::Skipped(SkipReason::Malformed)
    );
}

/// A part the layout disables must be absent, not merely empty.
#[test]
fn disabled_classes_reject_present_parts() {
    for layout in [
        Layout {
            scores: 0,
            ..layout()
        },
        Layout {
            draft: 0,
            ..layout()
        },
    ] {
        let config = config(4 * CHUNK as u64, StoreMode::OnRetain);
        let engine = StubCopyEngine::new(CopyModel::default(), DEVICE_BYTES, config.bytes as usize);
        let mut cache = HostCache::new(config, layout, engine).expect("cache");
        let mut device = Device::new(DEVICE_BYTES);
        let snapshot = snapshot(&mut device, SnapshotKind::Turn, &tokens(8), 1, true, 1);
        assert_eq!(
            cache.store(&snapshot, 1),
            StoreOutcome::Skipped(SkipReason::Malformed)
        );
    }
}

/// Order-of-magnitude floor: a broken fast path fails `cargo test`, not a human. Debug build,
/// so the bound is generous.
#[test]
fn store_tick_floor_for_4k_snapshots() {
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
        "1000 stores+tick took {elapsed:?}"
    );
}

#[derive(Clone, Debug)]
enum Op {
    Store {
        kind: SnapshotKind,
        tokens: Vec<u32>,
        pages: usize,
        has_draft: bool,
    },
    Tick,
    Evict,
    Lookup {
        tokens: Vec<u32>,
    },
    Restore,
    Free {
        id: DevicePageId,
    },
}

fn operation() -> impl Strategy<Value = Op> {
    prop_oneof![
        (
            any::<bool>(),
            prop::collection::vec(0u32..4, 1..8),
            0usize..3,
            any::<bool>(),
        )
            .prop_map(|(turn, tokens, pages, has_draft)| Op::Store {
                kind: if turn {
                    SnapshotKind::Turn
                } else {
                    SnapshotKind::Prompt
                },
                tokens,
                pages,
                has_draft,
            }),
        Just(Op::Tick),
        Just(Op::Evict),
        prop::collection::vec(0u32..4, 1..8).prop_map(|tokens| Op::Lookup { tokens }),
        Just(Op::Restore),
        (0u8..4, 0u32..8, 0u32..3).prop_map(|(compressor, page, generation)| Op::Free {
            id: DevicePageId {
                compressor,
                page,
                generation,
            },
        }),
    ]
}

/// Take the model key of a reported ticket out of the pending list.
fn take_pending(pending: &mut Vec<(StoreTicket, Key)>, ticket: StoreTicket) -> Key {
    let index = pending
        .iter()
        .position(|(candidate, _)| *candidate == ticket)
        .expect("reported ticket was pending");
    pending.remove(index).1
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]
    #[test]
    fn random_sequences_keep_the_invariants(ops in prop::collection::vec(operation(), 1..40)) {
        let quota = 8 * CHUNK as u64;
        // The facade evicts to the pool's quota exactly after every commit.
        let evict_quota = quota;
        let mut cache = default_cache(quota, StoreMode::OnRetain);
        let mut model = Model::new();
        let mut device = Device::new(DEVICE_BYTES);
        let mut stored: HashMap<Key, DeviceSnapshot> = HashMap::new();
        let mut planned: HashMap<Key, DeviceSnapshot> = HashMap::new();
        let mut pending: Vec<(StoreTicket, Key)> = Vec::new();
        let mut issued: HashSet<StoreTicket> = HashSet::new();
        let mut reported: HashSet<StoreTicket> = HashSet::new();
        let mut generation = 0u32;

        for op in ops {
            match op {
                Op::Store { kind, tokens, pages, has_draft } => {
                    generation += 1;
                    let snapshot = snapshot(&mut device, kind, &tokens, pages, has_draft, generation);
                    write_snapshot(cache.engine_mut(), &snapshot);
                    let outcome = cache.store(&snapshot, generation as u64);
                    // The cache may evict unpinned snapshots while planning under class
                    // exhaustion; mirror that before planning the model's copy.
                    reconcile(&cache, &mut model);
                    stored.retain(|key, _| model.snapshots.contains_key(key));
                    if let StoreOutcome::Issued(ticket) | StoreOutcome::Deferred(ticket) = outcome {
                        let key = model.plan(&snapshot.meta, &device_pages(&snapshot));
                        planned.insert(key, snapshot);
                        pending.push((ticket, key));
                        issued.insert(ticket);
                    }
                }
                Op::Tick => {
                    settle(&mut cache);
                    let report = cache.tick();
                    for ticket in report.completed {
                        let key = take_pending(&mut pending, ticket);
                        model.commit(key);
                        model.evict_to(evict_quota);
                        if let Some(snapshot) = planned.remove(&key) {
                            stored.insert(key, snapshot);
                        }
                        stored.retain(|key, _| model.snapshots.contains_key(key));
                        prop_assert!(reported.insert(ticket), "ticket reported twice");
                    }
                    for ticket in report.failed {
                        let key = take_pending(&mut pending, ticket);
                        model.abort(key);
                        planned.remove(&key);
                        prop_assert!(reported.insert(ticket), "ticket reported twice");
                    }
                }
                Op::Evict => {
                    if let Some(index) = pending.len().checked_sub(1) {
                        let (ticket, key) = pending.remove(index);
                        match cache.before_device_evict(Some(ticket)) {
                            EvictDecision::WaitedClean { .. } => {
                                model.commit(key);
                                model.evict_to(evict_quota);
                                if let Some(snapshot) = planned.remove(&key) {
                                    stored.insert(key, snapshot);
                                }
                                stored.retain(|key, _| model.snapshots.contains_key(key));
                            }
                            EvictDecision::DroppedUncached => {
                                model.abort(key);
                                planned.remove(&key);
                            }
                            EvictDecision::Clean => {
                                prop_assert!(false, "a pending ticket was clean");
                            }
                        }
                        prop_assert!(reported.insert(ticket), "ticket reported twice");
                    }
                }
                Op::Lookup { tokens } => {
                    let hit = cache.lookup(&tokens);
                    let expected = model.lookup(&tokens);
                    match (hit, expected) {
                        (Some(hit), Some((common, frontier, key))) => {
                            prop_assert_eq!(hit.key, key);
                            prop_assert_eq!(hit.common, common);
                            prop_assert_eq!(hit.frontier, frontier);
                        }
                        (None, None) => {}
                        (hit, expected) => {
                            prop_assert!(false, "lookup mismatch: {:?} vs {:?}", hit, expected);
                        }
                    }
                }
                Op::Restore => {
                    if let Some((&key, snapshot)) = stored.iter().next() {
                        generation += 1;
                        let target = target(&mut device, snapshot, generation);
                        let _ = cache.restore(key, &target);
                        prop_assert!(
                            cache.snapshot_tokens(key).is_some(),
                            "a restored snapshot was evicted"
                        );
                    }
                }
                Op::Free { id } => {
                    cache.device_page_freed(id);
                    model.device_page_freed(id);
                }
            }
            prop_assert!(cache.metrics().bytes_used <= quota);
        }

        // Drain whatever is still in flight; every issued ticket is reported exactly once.
        while let Some((ticket, key)) = pending.pop() {
            match cache.before_device_evict(Some(ticket)) {
                EvictDecision::WaitedClean { .. } => {
                    model.commit(key);
                    model.evict_to(evict_quota);
                }
                EvictDecision::DroppedUncached => model.abort(key),
                EvictDecision::Clean => {}
            }
            planned.remove(&key);
            prop_assert!(reported.insert(ticket), "ticket reported twice");
        }
        prop_assert_eq!(issued, reported);
    }
}
