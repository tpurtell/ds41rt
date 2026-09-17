//! Upstream-ported eviction / block-accounting invariants (component C5).
//!
//! Sources, stripped of upstream API specifics and restated over the ds41rt hostcache:
//! - vllm `tests/v1/core/test_prefix_caching.py`: BlockPool/BlockHashToBlockMap accounting
//!   (`test_maybe_evict_cached_block`: evicting one block must not drop a hash entry another
//!   block shares; `test_evict`: LRU free-queue order and touch-to-revive), prefix-cache hit
//!   counting, computed-block protection under pressure.
//! - sglang `test/registered/unit/mem_cache/test_rust_tree_core.py`: radix insert/match
//!   round-trip, split keeps both sides matchable, lock (pin) protects from eviction,
//!   namespaces isolate identical token sequences.
//! - vllm `tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py`: partial-hit
//!   boundaries — hash alignment, replay window, exact-ancestor vs longer-partial choice.
//!
//! Invariants NOT already covered by `functional_hc2.rs` / `functional_hc_5.rs`, targeted here:
//! eviction frees exactly the evicted share of shared pages (identity map survives while a
//! reference lives); LRU recency is refreshed by full AND partial hits; the partial-hit
//! replay-window floor (even-aligned `common - 128`); "most computation saved" tie-breaks;
//! per-commit byte accounting closure under sustained capacity pressure; hit counting vs
//! lookup outcomes; restore failure is a fall-through (the daemon's 5173e44 prefill path),
//! never a cache-state loss.
mod common_hc_5;

use common_hc_5::{
    cache, config, default_cache, store_resident, target, write_snapshot, Device, Payload,
    DEVICE_BYTES,
};
use ds41rt_hostcache::cache::{DevicePage, DeviceSnapshot, EvictDecision, HostCache, RestoreOutcome};
use ds41rt_hostcache::config::{Config, StoreMode};
use ds41rt_hostcache::copy::{CopyFault, CopyModel, Stream, StubCopyEngine};
use ds41rt_hostcache::pool::testing::{layout, CHUNK};
use ds41rt_hostcache::snapshot::{DevicePageId, SnapshotMeta};
use ds41rt_hostcache::{SnapshotKind, COMPRESSORS};

/// The chunk size the quota-accounting tests use: the smallest chunk that still carves every
/// class of the pool suites' layout (tail is the largest class at 8192).
const TEST_CHUNK: u64 = 8192;

/// Bytes one unshared one-page snapshot holds: page + tail + scores of the pool test layout.
fn snap_bytes() -> u64 {
    let layout = layout();
    (layout.page + layout.tail + layout.scores) as u64
}

/// A cache over a `chunks`-chunk pool: fine-grained enough to force eviction in a few stores.
fn quota_cache(chunks: u64, mode: StoreMode) -> HostCache<StubCopyEngine, Payload> {
    let config = Config {
        bytes: chunks * TEST_CHUNK,
        chunk_bytes: TEST_CHUNK,
        store: mode,
        min_tokens: 1,
        ..config(0, mode)
    };
    cache(config, CopyModel::default(), DEVICE_BYTES)
}

/// A snapshot over `tokens` holding exactly `page_ids.len()` pages, all in compressor 0, so a
/// later snapshot listing the same `DevicePageId` shares the page instead of copying it.
fn snap(
    device: &mut Device,
    kind: SnapshotKind,
    tokens: &[u32],
    page_ids: &[DevicePageId],
) -> DeviceSnapshot {
    let layout = layout();
    let mut lists: [Vec<DevicePage>; COMPRESSORS] = std::array::from_fn(|_| Vec::new());
    for &id in page_ids {
        lists[0].push(DevicePage {
            id,
            segments: vec![device.range(layout.page)],
        });
    }
    DeviceSnapshot {
        meta: SnapshotMeta {
            kind,
            tokens: tokens.to_vec(),
            end: tokens.len() as u32,
            has_draft: false,
        },
        pages: lists,
        tail: vec![device.range(layout.tail)],
        draft: None,
        scores: vec![device.range(layout.scores)],
    }
}

/// `count` tokens starting at `base`.
fn seq(base: u32, count: usize) -> Vec<u32> {
    (0..count as u32).map(|offset| base + offset).collect()
}

/// A device page identity in compressor 0.
fn id(page: u32, generation: u32) -> DevicePageId {
    DevicePageId {
        compressor: 0,
        page,
        generation,
    }
}

/// Port of vllm `test_maybe_evict_cached_block`'s shared-hash invariant: evicting one holder of
/// a shared page/identity must not drop the share while another snapshot references it, and the
/// bytes freed are exactly the evicted snapshot's unique share — not its full footprint.
#[test]
fn eviction_frees_only_the_evicted_share_of_a_shared_page() {
    for mode in [StoreMode::OnRetain, StoreMode::OnEvict] {
        // Four chunks hold the first two snapshots (page 7 shared, one page plus tail and
        // scores each = 17_408 bytes); a third store's tail class is then exhausted and the
        // cache evicts the least recently used snapshot before its plan fits.
        let mut cache = quota_cache(4, mode);
        let mut device = Device::new(DEVICE_BYTES);

        let first = snap(&mut device, SnapshotKind::Turn, &seq(1, 9), &[id(7, 1)]);
        write_snapshot(cache.engine_mut(), &first);
        store_resident(&mut cache, &first, 1);

        // Second snapshot extends the first's token prefix and shares its page 7. Both fit.
        let mut second_tokens = seq(1, 8);
        second_tokens.push(100);
        let second = snap(&mut device, SnapshotKind::Turn, &second_tokens, &[id(7, 1), id(8, 2)]);
        write_snapshot(cache.engine_mut(), &second);
        store_resident(&mut cache, &second, 2);
        assert_eq!(cache.metrics().resident_snapshots, 2);

        // Pressure: the third store's plan is exhausted (no tail chunk free) and the cache
        // evicts. The first eviction happens before the plan holds any page reference (the
        // tail class failed first), so the first snapshot's full footprint — page 7 included —
        // comes back and its device identity is unmapped: the page was genuinely dead. The
        // retry then fails on the page class (one chunk of two slabs holds pages 7 and 8); the
        // second eviction drops the second snapshot but NOT the shared page 7, which the
        // in-flight plan references: the second eviction frees exactly the unique share.
        let layout = layout();
        let third = snap(&mut device, SnapshotKind::Turn, &seq(500, 2), &[id(7, 1), id(9, 3)]);
        write_snapshot(cache.engine_mut(), &third);
        store_resident(&mut cache, &third, 3);

        assert_eq!(cache.metrics().host_evictions, 2);
        assert_eq!(
            cache.metrics().host_evicted_bytes,
            2 * snap_bytes(),
            "first evicted whole (no references yet); second freed its share minus the shared page"
        );
        // Resident: pages 7 (carried across two evictions) and 9, one tail, one scores.
        assert_eq!(
            cache.metrics().bytes_used,
            2 * layout.page as u64 + (layout.tail + layout.scores) as u64
        );
        assert_eq!(cache.metrics().resident_snapshots, 1);
        // Both evicted token sequences fall through to prefill now.
        assert!(cache.lookup(&seq(1, 9)).is_none());
        assert!(cache.lookup(&second_tokens).is_none());
        let hit = cache.lookup(&seq(500, 2)).expect("third resident");
        assert_eq!(cache.payload(hit.key), Some(&3));
        // Pages 7 (died with the first eviction, recopied) and 9: four copies in total.
        assert_eq!(cache.metrics().pages_copied, 4);

        // A fourth store of the SAME device identity as the third's first page: the identity
        // entry survived the second eviction (a reference was live), so the share is found and
        // nothing is copied; the store fits the slabs the second eviction freed, without
        // evicting anyone. This is vllm's `test_maybe_evict_cached_block` invariant: an
        // identity shared by a live holder is not dropped when another holder is evicted.
        let fourth = snap(&mut device, SnapshotKind::Turn, &seq(600, 2), &[id(7, 1)]);
        write_snapshot(cache.engine_mut(), &fourth);
        store_resident(&mut cache, &fourth, 4);
        assert_eq!(cache.metrics().pages_copied, 4, "page 7 shared, not recopied");
        assert_eq!(cache.metrics().host_evictions, 2, "the fourth store fits free slabs");
        assert_eq!(
            cache.metrics().bytes_used,
            2 * layout.page as u64 + 2 * (layout.tail + layout.scores) as u64
        );
        assert!(cache.lookup(&seq(500, 2)).is_some(), "third survives");
        let hit = cache.lookup(&seq(600, 2)).expect("fourth resident");
        assert_eq!(cache.payload(hit.key), Some(&4));
    }
}

/// Port of vllm `test_evict`'s touch-to-revive and `test_computed_blocks_not_evicted`: a lookup
/// hit — exact-prefix OR partial — refreshes eviction recency, so under capacity pressure the
/// recently-hit snapshot is evicted last.
#[test]
fn lookup_hits_refresh_lru_eviction_order() {
    // Four chunks hold two one-page snapshots (2 * 13_312 = 26_624 <= 32_768).
    let mut cache = quota_cache(4, StoreMode::OnRetain);
    let mut device = Device::new(DEVICE_BYTES);
    let a_tokens = seq(0, 200);
    let b_tokens = seq(1000, 200);
    let a = snap(&mut device, SnapshotKind::Turn, &a_tokens, &[id(10, 1)]);
    write_snapshot(cache.engine_mut(), &a);
    store_resident(&mut cache, &a, 1);
    let b = snap(&mut device, SnapshotKind::Turn, &b_tokens, &[id(11, 2)]);
    write_snapshot(cache.engine_mut(), &b);
    store_resident(&mut cache, &b, 2);
    assert_eq!(cache.metrics().resident_snapshots, 2);

    // A PARTIAL hit on A (divergence at 150, inside the replay window boundary) revives it.
    let mut partial = seq(0, 150);
    partial.push(777);
    let hit = cache.lookup(&partial).expect("partial hit revives A");
    assert_eq!(hit.common, 150);
    assert_eq!(hit.frontier, 200);

    // Pressure: C's commit evicts the least recently used, which is now B, not A.
    let c = snap(&mut device, SnapshotKind::Turn, &seq(2000, 200), &[id(12, 3)]);
    write_snapshot(cache.engine_mut(), &c);
    store_resident(&mut cache, &c, 3);
    assert!(cache.lookup(&b_tokens).is_none(), "untouched B evicted first");
    let hit = cache.lookup(&a_tokens).expect("revived A survives");
    assert_eq!(cache.payload(hit.key), Some(&1));

    // Again: another partial hit on A, then D evicts C instead.
    let mut partial = seq(0, 150);
    partial.push(888);
    assert!(cache.lookup(&partial).is_some());
    let d = snap(&mut device, SnapshotKind::Turn, &seq(3000, 200), &[id(13, 4)]);
    write_snapshot(cache.engine_mut(), &d);
    store_resident(&mut cache, &d, 4);
    assert!(cache.lookup(&seq(2000, 200)).is_none(), "C evicted before A");
    assert!(cache.lookup(&a_tokens).is_some(), "A survives both rounds");
    assert!(cache.lookup(&seq(3000, 200)).is_some());
}

/// Port of the vllm partial-prefix-hits replay-window invariants: a query that diverges from a
/// resident snapshot at `common` tokens is reusable exactly when the even-aligned replay
/// `(common / 2 * 2) - 128` saves computation — the boundary is divergence at 130, not 128.
#[test]
fn partial_hit_replay_window_boundary() {
    let mut cache = default_cache(4 * CHUNK as u64, StoreMode::OnRetain);
    let mut device = Device::new(DEVICE_BYTES);
    let stored = seq(0, 300);
    let snapshot = snap(&mut device, SnapshotKind::Turn, &stored, &[id(20, 1)]);
    write_snapshot(cache.engine_mut(), &snapshot);
    store_resident(&mut cache, &snapshot, 7);

    // Divergence below the floor: skipped() saturates to 0, the cache must fall through.
    for divergence in [127, 128, 129] {
        let mut query = seq(0, divergence);
        query.push(999);
        assert!(
            cache.lookup(&query).is_none(),
            "divergence at {divergence} is inside the replay window: miss"
        );
    }
    // Divergence at the floor: even-aligned common - 128 > 0, a hit with the exact geometry.
    for divergence in [130, 131] {
        let mut query = seq(0, divergence);
        query.push(999);
        let hit = cache
            .lookup(&query)
            .unwrap_or_else(|| panic!("divergence at {divergence} must hit"));
        assert_eq!(hit.common, divergence);
        assert_eq!(hit.frontier, 300);
        assert_eq!(hit.kind, SnapshotKind::Turn);
    }
    // Exact and extending queries match the whole snapshot.
    let hit = cache.lookup(&stored).expect("exact hit");
    assert_eq!((hit.common, hit.frontier), (300, 300));
    let mut extended = stored.clone();
    extended.extend([501, 502]);
    let hit = cache.lookup(&extended).expect("extending hit");
    assert_eq!((hit.common, hit.frontier), (300, 300));

    // Hit counting matches outcomes: 7 lookups, 4 hits (130, 131, exact, extending).
    let metrics = cache.metrics();
    assert_eq!(metrics.lookups, 7);
    assert_eq!(metrics.host_hits, 4);
}

/// Port of the vllm/sglang "most computation saved" reuse rule: a populated exact ancestor
/// beats a longer partial match whose even-aligned replay saves less, and loses once the
/// partial match's replay passes it.
#[test]
fn exact_ancestor_trades_off_against_a_longer_partial_match() {
    let mut cache = default_cache(4 * CHUNK as u64, StoreMode::OnRetain);
    let mut device = Device::new(DEVICE_BYTES);
    let ancestor = snap(&mut device, SnapshotKind::Turn, &seq(0, 250), &[id(30, 1)]);
    write_snapshot(cache.engine_mut(), &ancestor);
    store_resident(&mut cache, &ancestor, 1);
    // 500 tokens so the winning query below diverges INSIDE the snapshot
    // (review 2026-09-15: at 380 the query matched the whole frontier and the
    // winning-partial-match branch was never exercised).
    let longer = snap(&mut device, SnapshotKind::Turn, &seq(0, 500), &[id(31, 2)]);
    write_snapshot(cache.engine_mut(), &longer);
    store_resident(&mut cache, &longer, 2);

    let key_of_ancestor = cache.lookup(&seq(0, 250)).expect("ancestor").key;
    let key_of_longer = cache.lookup(&seq(0, 500)).expect("longer").key;

    // Divergence at 377: the partial candidate saves (377/2*2)-128 = 248 < 250: ancestor wins.
    let mut query = seq(0, 377);
    query.push(999);
    let hit = cache.lookup(&query).expect("hit");
    assert_eq!(hit.key, key_of_ancestor);
    assert_eq!((hit.common, hit.frontier), (250, 250));

    // Divergence at 379: the partial candidate saves exactly 250 — a tie keeps the ancestor.
    let mut query = seq(0, 379);
    query.push(999);
    let hit = cache.lookup(&query).expect("hit");
    assert_eq!(hit.key, key_of_ancestor, "tie prefers the exact ancestor");
    assert_eq!((hit.common, hit.frontier), (250, 250));

    // Divergence at 381, inside the 500-token snapshot: saves 380-128 = 252 > 250,
    // so the longer PARTIAL match wins over the exact ancestor.
    let mut query = seq(0, 381);
    query.push(999);
    let hit = cache.lookup(&query).expect("hit");
    assert_eq!(hit.key, key_of_longer, "winning partial match beats exact ancestor");
    assert_eq!(hit.common, 381);

    // Full-frontier query still resolves to the complete longer snapshot.
    let mut query = seq(0, 500);
    query.push(999);
    let hit = cache.lookup(&query).expect("hit");
    assert_eq!(hit.key, key_of_longer);
    assert_eq!((hit.common, hit.frontier), (500, 500));
}

/// Port of the sglang radix split invariants (`test_insert_then_match_round_trips` plus edge
/// splitting): inserting a shorter and then a longer key over the same prefix keeps every
/// inserted frontier matchable, and an interior query resolves to the deepest exact ancestor.
#[test]
fn radix_split_keeps_every_inserted_frontier_matchable() {
    let mut cache = default_cache(8 * CHUNK as u64, StoreMode::OnRetain);
    let mut device = Device::new(DEVICE_BYTES);

    let long = snap(&mut device, SnapshotKind::Turn, &seq(0, 300), &[id(40, 1)]);
    write_snapshot(cache.engine_mut(), &long);
    store_resident(&mut cache, &long, 1);
    // Inserting the shorter key splits the long key's edge.
    let short = snap(&mut device, SnapshotKind::Turn, &seq(0, 200), &[id(41, 2)]);
    write_snapshot(cache.engine_mut(), &short);
    store_resident(&mut cache, &short, 2);
    // Inserting a sibling of the short key splits below it.
    let mut sibling_tokens = seq(0, 200);
    sibling_tokens.push(700);
    let sibling = snap(&mut device, SnapshotKind::Turn, &sibling_tokens, &[id(42, 3)]);
    write_snapshot(cache.engine_mut(), &sibling);
    store_resident(&mut cache, &sibling, 3);

    let key_long = cache.lookup(&seq(0, 300)).expect("long");
    assert_eq!((key_long.common, key_long.frontier), (300, 300));
    let key_short = cache.lookup(&seq(0, 200)).expect("short");
    assert_eq!((key_short.common, key_short.frontier), (200, 200));
    let key_sibling = cache.lookup(&sibling_tokens).expect("sibling");
    assert_eq!((key_sibling.common, key_sibling.frontier), (201, 201));
    assert_eq!(cache.metrics().resident_snapshots, 3);

    // An interior query with no frontier of its own takes the deepest exact ancestor: the
    // short snapshot saves 200 tokens; the long partial beneath it would save only 250-128.
    let hit = cache.lookup(&seq(0, 250)).expect("interior hit");
    assert_eq!(hit.key, key_short.key);
    assert_eq!((hit.common, hit.frontier), (200, 200));
}

/// Port of the sglang namespace-isolation invariant (`test_namespaces_isolate_the_same_tokens`):
/// identical token sequences in different banks (here: Prompt vs Turn) are independent
/// identities; replacing one leaves the other, and a tie prefers the completed turn.
#[test]
fn same_tokens_in_different_banks_are_isolated() {
    let mut cache = default_cache(8 * CHUNK as u64, StoreMode::OnRetain);
    let mut device = Device::new(DEVICE_BYTES);
    let tokens = seq(0, 100);

    let prompt = snap(&mut device, SnapshotKind::Prompt, &tokens, &[id(50, 1)]);
    write_snapshot(cache.engine_mut(), &prompt);
    store_resident(&mut cache, &prompt, 100);
    let key_prompt = cache.lookup(&tokens).expect("prompt").key;
    assert_eq!(cache.payload(key_prompt), Some(&100));

    let turn = snap(&mut device, SnapshotKind::Turn, &tokens, &[id(51, 2)]);
    write_snapshot(cache.engine_mut(), &turn);
    store_resident(&mut cache, &turn, 200);
    assert_eq!(cache.metrics().resident_snapshots, 2, "banks isolate identity");
    // A tie prefers the completed turn (the engine's reuse rule).
    let hit = cache.lookup(&tokens).expect("hit");
    assert_eq!(hit.kind, SnapshotKind::Turn);
    assert_eq!(cache.payload(hit.key), Some(&200));
    let key_turn = hit.key;

    // Replacing the turn replaces only the turn.
    let turn2 = snap(&mut device, SnapshotKind::Turn, &tokens, &[id(52, 3)]);
    write_snapshot(cache.engine_mut(), &turn2);
    store_resident(&mut cache, &turn2, 300);
    assert_eq!(cache.metrics().stores_replaced, 1);
    assert_eq!(cache.metrics().resident_snapshots, 2);
    let hit = cache.lookup(&tokens).expect("hit");
    assert_ne!(hit.key, key_turn);
    assert_eq!(cache.payload(hit.key), Some(&300));
    assert_eq!(cache.payload(key_prompt), Some(&100), "prompt untouched");
    assert_eq!(cache.payload(key_turn), None, "old turn payload dropped");
}

/// Per-commit byte accounting closure under sustained capacity pressure (vllm BlockPool
/// free-accounting): every commit brings exactly one snapshot's bytes into residency and evicts
/// the difference, so increase + evicted == the snapshot's bytes, and the cumulative counters
/// close exactly at the end.
#[test]
fn byte_accounting_closes_under_sustained_eviction_pressure() {
    let quota_chunks = 4; // two snapshots resident at a time
    let mut cache = quota_cache(quota_chunks, StoreMode::OnRetain);
    let mut device = Device::new(DEVICE_BYTES);
    let snapshot_bytes = snap_bytes();
    let mut issued = 0u64;

    for generation in 0..10u32 {
        let snapshot =
            snap(
            &mut device,
            SnapshotKind::Turn,
            &seq(generation * 1000, 200),
            &[id(u32::from(generation), generation)],
        );
        write_snapshot(cache.engine_mut(), &snapshot);
        let before_bytes = cache.metrics().bytes_used;
        let before_evicted = cache.metrics().host_evicted_bytes;
        store_resident(&mut cache, &snapshot, u64::from(generation));
        issued += snapshot_bytes;
        let metrics = cache.metrics();
        let evicted = metrics.host_evicted_bytes - before_evicted;
        // What the commit took in plus what the commit evicted is exactly one snapshot.
        assert_eq!(
            metrics.bytes_used + evicted - before_bytes,
            snapshot_bytes,
            "commit {generation} accounting"
        );
        assert_eq!(evicted % snapshot_bytes, 0, "whole snapshots are evicted");
        assert!(metrics.bytes_used <= quota_chunks * TEST_CHUNK);
    }

    let metrics = cache.metrics();
    assert_eq!(metrics.stores_completed, 10);
    assert_eq!(metrics.host_evictions, 8);
    assert_eq!(metrics.bytes_used, 2 * snapshot_bytes);
    // The global closure: everything that entered residency is resident or accounted evicted.
    assert_eq!(
        metrics.host_evicted_bytes,
        issued - metrics.bytes_used,
        "evicted + resident == issued"
    );
    // Store-side accounting: every store copied one page plus its tail and scores.
    assert_eq!(metrics.pages_copied, 10);
    assert_eq!(metrics.pages_shared, 0);
    let layout = layout();
    assert_eq!(
        metrics.store_bytes,
        metrics.pages_copied * layout.page as u64
            + metrics.stores_completed * (layout.tail + layout.scores) as u64
    );
}

/// The 5173e44 fall-through: a restore that cannot complete (issue failure, wedged stream) is
/// a data outcome, never a cache-state loss — the snapshot stays resident and lookup-visible,
/// the failure is counted under restore_failures/restore_timeouts, and the engine proceeds to
/// prefill (the daemon) or retries (here) exactly as if the lookup had missed.
#[test]
fn failed_restore_falls_through_without_losing_the_snapshot() {
    // Issue failure on the restore stream: Failed, counted, state untouched, retry succeeds.
    let mut cache = default_cache(4 * CHUNK as u64, StoreMode::OnRetain);
    let mut device = Device::new(DEVICE_BYTES);
    let snapshot = snap(&mut device, SnapshotKind::Turn, &seq(0, 100), &[id(60, 1)]);
    write_snapshot(cache.engine_mut(), &snapshot);
    store_resident(&mut cache, &snapshot, 1);
    let key = cache.lookup(&seq(0, 100)).expect("hit").key;
    let bytes_before = cache.metrics().bytes_used;

    cache.engine_mut().inject(CopyFault::IssueFails(Stream::Restore));
    let restore_target = target(&mut device, &snapshot, 2);
    assert_eq!(
        cache.restore(key, &restore_target),
        RestoreOutcome::Failed,
        "the engine falls through to prefill on a failed restore"
    );
    assert_eq!(cache.metrics().restore_failures, 1);
    assert_eq!(cache.metrics().bytes_used, bytes_before, "no state lost");
    let hit = cache.lookup(&seq(0, 100)).expect("snapshot still resident");
    assert_eq!(hit.key, key);

    // The fault fired exactly once: a retry restores cleanly.
    assert!(matches!(
        cache.restore(key, &restore_target),
        RestoreOutcome::Done { .. }
    ));
    assert_eq!(cache.metrics().restore_failures, 1);
    assert_eq!(cache.metrics().restores, 1);

    // A wedged restore stream times out within budget and also leaves the snapshot resident.
    let mut cache = default_cache(4 * CHUNK as u64, StoreMode::OnRetain);
    let mut device = Device::new(DEVICE_BYTES);
    let snapshot = snap(&mut device, SnapshotKind::Turn, &seq(0, 100), &[id(61, 1)]);
    write_snapshot(cache.engine_mut(), &snapshot);
    store_resident(&mut cache, &snapshot, 1);
    let key = cache.lookup(&seq(0, 100)).expect("hit").key;
    cache.engine_mut().inject(CopyFault::StreamStalls(Stream::Restore));
    let restore_target = target(&mut device, &snapshot, 2);
    assert_eq!(
        cache.restore(key, &restore_target),
        RestoreOutcome::TimedOut,
        "budget expiry is a fall-through, not an error"
    );
    assert_eq!(cache.metrics().restore_timeouts, 1);
    assert!(cache.lookup(&seq(0, 100)).is_some(), "still resident");
    assert_eq!(cache.metrics().bytes_used, bytes_before);
}

/// Store-then-load round-trip identity at the crate level: a stored snapshot is found by its
/// own tokens, carries its payload and token sequence back identically, and a prefix long
/// enough to clear the replay floor hits the same snapshot as a partial match. A prefix below
/// the floor falls through even though the snapshot covers it — the same boundary the
/// partial-hit tests pin.
#[test]
fn store_then_lookup_roundtrip_identity() {
    for mode in [StoreMode::OnRetain, StoreMode::OnEvict] {
        let mut cache = default_cache(4 * CHUNK as u64, mode);
        let mut device = Device::new(DEVICE_BYTES);
        let tokens = seq(0, 200);
        let snapshot = snap(&mut device, SnapshotKind::Turn, &tokens, &[id(70, 1)]);
        write_snapshot(cache.engine_mut(), &snapshot);
        store_resident(&mut cache, &snapshot, 0xABCD);

        // Exact tokens: the same snapshot, its full identity attached.
        let hit = cache.lookup(&tokens).expect("exact hit");
        assert_eq!(hit.kind, SnapshotKind::Turn);
        assert_eq!((hit.common, hit.frontier), (200, 200));
        assert_eq!(cache.payload(hit.key), Some(&0xABCD));
        assert_eq!(cache.snapshot_tokens(hit.key), Some(tokens.as_slice()));

        // A prefix past the replay floor hits the same snapshot as a partial match.
        let hit = cache.lookup(&seq(0, 150)).expect("prefix hit");
        assert_eq!((hit.common, hit.frontier), (150, 200));
        assert_eq!(cache.snapshot_tokens(hit.key), Some(tokens.as_slice()));
        // Below the floor even a covering prefix falls through to prefill.
        assert!(cache.lookup(&seq(0, 48)).is_none());

        // And the hit refreshes it: it is still lookup-visible afterwards.
        let peer = snap(&mut device, SnapshotKind::Turn, &seq(1000, 200), &[id(71, 2)]);
        write_snapshot(cache.engine_mut(), &peer);
        store_resident(&mut cache, &peer, 2);
        assert_eq!(cache.lookup(&tokens).unwrap().key, hit.key);
    }
}

// OnEvict mode consults `before_device_evict` for the deferred copy; keep both modes in the
// sharing test honest about it having been exercised there.
#[test]
fn on_evict_deferred_store_commits_clean_through_the_evict_path() {
    let mut cache = quota_cache(3, StoreMode::OnEvict);
    let mut device = Device::new(DEVICE_BYTES);
    let snapshot = snap(&mut device, SnapshotKind::Turn, &seq(0, 100), &[id(80, 1)]);
    write_snapshot(cache.engine_mut(), &snapshot);
    let outcome = cache.store(&snapshot, 1);
    let ds41rt_hostcache::cache::StoreOutcome::Deferred(ticket) = outcome else {
        panic!("OnEvict defers the copy, got {outcome:?}");
    };
    // Nothing copied yet: not lookup-visible before the device eviction.
    assert!(cache.lookup(&seq(0, 100)).is_none());
    assert!(matches!(
        cache.before_device_evict(Some(ticket)),
        EvictDecision::WaitedClean { .. }
    ));
    let hit = cache.lookup(&seq(0, 100)).expect("committed by the evict wait");
    assert_eq!(cache.payload(hit.key), Some(&1));
    assert_eq!(cache.metrics().device_evictions, 1);
}
