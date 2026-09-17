//! Functional suite for packet HC-8: coalesced snapshot copies. The cache builds a snapshot's
//! whole copy list (store D2H, restore H2D), merges adjacent copies that are contiguous on
//! both sides, and issues one batch per snapshot through the `CopyEngine::d2h_many`/`h2d_many`
//! entry points. The suites check: the batch path moves exactly the same bytes to the same
//! places (content equality through the stub's memory model), the batch error paths reject a
//! bad extent before any submission exactly as the 1D path does, the `copy_submissions` gauge
//! tracks the engine's submission counter, and a snapshot allocated from a fresh pool costs at
//! least an order of magnitude fewer submissions than the 1D plan. Proptest properties over
//! random page sets and slab layouts assert the coalesced plan covers exactly the same
//! ordered `(src, dst, bytes)` mapping as the 1D plan with no overlaps, and that a random
//! snapshot round-trips byte-for-byte through the batch path.
mod common_hc_5;

use common_hc_5::{
    default_cache, restored_bytes, snapshot, stored_bytes, target, tokens, write_snapshot, Device,
};
use ds41rt_hostcache::cache::{
    DevicePage, DeviceSnapshot, EvictDecision, HostCache, RestoreOutcome, RestoreTarget,
    StoreOutcome,
};
use ds41rt_hostcache::config::{Config, StoreMode};
use ds41rt_hostcache::copy::{
    coalesce, CopyEngine, CopyFault, CopyModel, DeviceRange, Stream, StubCopyEngine,
};
use ds41rt_hostcache::pool::testing::{layout, CHUNK};
use ds41rt_hostcache::pool::{HostRange, Layout, PinnedMemory};
use ds41rt_hostcache::snapshot::{DevicePageId, SnapshotMeta};
use ds41rt_hostcache::{SnapshotKind, COMPRESSORS, KV_BYTES_PER_TOKEN, PAGE_BYTES, TAIL_BYTES};
use proptest::prelude::*;

/// xorshift64* for the deterministic generators (same algorithm as `common::Rng`, local so
/// this suite builds no HC-2 model).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Hands out non-overlapping device ranges over `capacity` bytes.
struct Ranges {
    addr: u64,
    capacity: u64,
}

impl Ranges {
    fn new(capacity: u64) -> Self {
        Self { addr: 0, capacity }
    }
    fn range(&mut self, bytes: usize) -> DeviceRange {
        if self.addr + bytes as u64 > self.capacity {
            panic!("ranges exceed the fake device");
        }
        let range = DeviceRange {
            addr: self.addr,
            bytes,
        };
        self.addr += bytes as u64;
        range
    }
}

/// A snapshot of `tokens` tokens at the engine's 890 bytes/token: `pages_per_compressor`
/// four-segment pages per compressor (segments address-contiguous, as the engine allocates
/// them), one tail. Non-overlapping consecutive ranges from `ranges`.
fn engine_snapshot(tokens: usize, ranges: &mut Ranges) -> DeviceSnapshot {
    let pages_per_compressor = tokens * KV_BYTES_PER_TOKEN / (PAGE_BYTES * COMPRESSORS);
    let mut lists: [Vec<DevicePage>; COMPRESSORS] = std::array::from_fn(|_| Vec::new());
    for (compressor, list) in lists.iter_mut().enumerate() {
        for index in 0..pages_per_compressor {
            list.push(DevicePage {
                id: DevicePageId {
                    compressor: compressor as u8,
                    page: index as u32,
                    generation: 1,
                },
                segments: vec![
                    ranges.range(PAGE_BYTES / 4),
                    ranges.range(PAGE_BYTES / 4),
                    ranges.range(PAGE_BYTES / 4),
                    ranges.range(PAGE_BYTES / 4),
                ],
            });
        }
    }
    DeviceSnapshot {
        meta: SnapshotMeta {
            kind: SnapshotKind::Turn,
            tokens: (0..tokens as u32).collect(),
            end: tokens as u32,
            has_draft: false,
        },
        pages: lists,
        tail: vec![ranges.range(TAIL_BYTES)],
        draft: None,
        scores: vec![],
    }
}

/// The 1D plan's submission count for a snapshot shaped like [`engine_snapshot`]: four copies
/// per page plus one tail copy.
fn one_d_submissions(snapshot: &DeviceSnapshot) -> u64 {
    let pages: u64 = snapshot.pages.iter().map(|list| list.len() as u64).sum();
    4 * pages + snapshot.tail.len() as u64
}

/// The ordered concatenation of the deterministic per-range pattern over a snapshot's parts.
fn expected_bytes(snapshot: &DeviceSnapshot) -> Vec<u8> {
    fn pattern(range: DeviceRange) -> Vec<u8> {
        (0..range.bytes)
            .map(|offset| (range.addr as usize + offset) as u8)
            .collect()
    }
    let mut bytes = Vec::new();
    for list in &snapshot.pages {
        for page in list {
            for segment in &page.segments {
                bytes.extend(pattern(*segment));
            }
        }
    }
    for segment in &snapshot.tail {
        bytes.extend(pattern(*segment));
    }
    if let Some(draft) = &snapshot.draft {
        for segment in draft {
            bytes.extend(pattern(*segment));
        }
    }
    for segment in &snapshot.scores {
        bytes.extend(pattern(*segment));
    }
    bytes
}

#[test]
fn store_and_restore_move_the_same_bytes_as_the_1d_path() {
    let mut cache = default_cache(8 * CHUNK as u64, StoreMode::OnRetain);
    let mut device = Device::new(common_hc_5::DEVICE_BYTES);
    let snap = snapshot(&mut device, SnapshotKind::Turn, &tokens(8), 2, true, 1);
    write_snapshot(cache.engine_mut(), &snap);
    let StoreOutcome::Issued(_) = cache.store(&snap, 1) else {
        panic!("expected an issued store");
    };
    // One batch, however many extents the coalesced plan has.
    let store_submissions = cache.engine_mut().submission_count();
    assert_eq!(cache.metrics().copy_submissions, store_submissions);
    assert_eq!(store_submissions, 1, "a store is one batch");
    common_hc_5::settle(&mut cache);
    let report = cache.tick();
    assert_eq!(report.completed.len(), 1);
    let hit = cache.lookup(&tokens(8)).expect("host hit");
    let restore_target = target(&mut device, &snap, 2);
    let RestoreOutcome::Done { .. } = cache.restore(hit.key, &restore_target) else {
        panic!("expected a completed restore");
    };
    assert_eq!(
        restored_bytes(cache.engine_mut(), &restore_target),
        expected_bytes(&snap)
    );
    assert_eq!(
        cache.engine_mut().submission_count(),
        2,
        "store and restore are one batch each"
    );
    assert_eq!(
        cache.metrics().copy_submissions,
        2,
        "the gauge tracks the engine counter"
    );
}

#[test]
fn failed_and_held_batches_report_the_right_submissions() {
    // An injected issue fault fails the whole store batch: nothing is submitted, the ticket
    // reports failed, and the next store succeeds as one batch.
    let mut cache = default_cache(8 * CHUNK as u64, StoreMode::OnRetain);
    let mut device = Device::new(common_hc_5::DEVICE_BYTES);
    let snap = snapshot(&mut device, SnapshotKind::Turn, &tokens(8), 1, false, 1);
    write_snapshot(cache.engine_mut(), &snap);
    cache
        .engine_mut()
        .inject(CopyFault::IssueFails(Stream::Store));
    let StoreOutcome::Issued(ticket) = cache.store(&snap, 1) else {
        panic!("expected an issued store");
    };
    let report = cache.tick();
    assert_eq!(report.failed, vec![ticket]);
    assert_eq!(cache.engine_mut().submission_count(), 0);
    assert_eq!(cache.metrics().copy_submissions, 0);

    let StoreOutcome::Issued(_) = cache.store(&snap, 1) else {
        panic!("expected the reissued store");
    };
    assert_eq!(cache.engine_mut().submission_count(), 1);

    // A wedged store stream holds the whole batch: every extent is pending and counted, none
    // lands, and the evict consultation drops the snapshot uncached.
    let mut cache = default_cache(8 * CHUNK as u64, StoreMode::OnRetain);
    let mut device = Device::new(common_hc_5::DEVICE_BYTES);
    let snap = snapshot(&mut device, SnapshotKind::Turn, &tokens(8), 1, false, 1);
    write_snapshot(cache.engine_mut(), &snap);
    cache
        .engine_mut()
        .inject(CopyFault::StreamStalls(Stream::Store));
    let StoreOutcome::Issued(ticket) = cache.store(&snap, 1) else {
        panic!("expected an issued store");
    };
    // The wedged stream holds the whole batch: all 21 raw copies coalesce to 3 extents on the
    // fresh pool (one for the four consecutive page slabs, one tail, one scores), every held
    // extent is counted as pending, and the batch is one submission.
    assert_eq!(cache.engine_mut().pending(Stream::Store), 3);
    assert_eq!(cache.engine_mut().submission_count(), 1);
    assert_eq!(
        cache.before_device_evict(Some(ticket)),
        EvictDecision::DroppedUncached
    );
}

#[test]
fn disabled_cache_counts_no_submissions() {
    let mut cache = default_cache(0, StoreMode::OnRetain);
    assert!(!cache.enabled());
    let mut device = Device::new(common_hc_5::DEVICE_BYTES);
    let snap = snapshot(&mut device, SnapshotKind::Turn, &tokens(8), 1, false, 1);
    write_snapshot(cache.engine_mut(), &snap);
    assert!(matches!(cache.store(&snap, 1), StoreOutcome::Skipped(_)));
    assert_eq!(cache.metrics().copy_submissions, 0);
}

/// Direct engine check that a batch with an invalid extent is rejected before any submission,
/// exactly like a 1D issue, and that the armed fault survives the rejected batch.
#[test]
fn invalid_extent_in_a_batch_is_rejected_before_submission() {
    let engine = &mut StubCopyEngine::new(CopyModel::default(), 1024, 1024);
    let chunk = PinnedMemory::allocate_chunk(engine, 1024).expect("chunk");
    let good = (
        DeviceRange {
            addr: 0,
            bytes: 100,
        },
        HostRange {
            chunk: chunk.id,
            offset: 0,
            bytes: 100,
        },
    );
    let out_of_bounds = (
        DeviceRange {
            addr: 0,
            bytes: 2000,
        },
        HostRange {
            chunk: chunk.id,
            offset: 0,
            bytes: 2000,
        },
    );
    engine.inject(CopyFault::IssueFails(Stream::Store));
    for batch in [vec![good, out_of_bounds], vec![out_of_bounds, good]] {
        assert!(engine.d2h_many(Stream::Store, &batch).is_err());
        assert_eq!(engine.submission_count(), 0, "invalid batch was submitted");
        assert_eq!(
            engine.pending(Stream::Store),
            0,
            "invalid batch left copies"
        );
    }
    // The armed fault is intact: a valid batch now fails on the fault, not on validation,
    // and still submits nothing.
    assert!(engine.d2h_many(Stream::Store, &[good]).is_err());
    assert_eq!(engine.submission_count(), 0);
    assert_eq!(engine.pending(Stream::Store), 0);
}

/// Order-of-magnitude floor: a 320k-token snapshot (4 segments per page, the engine's shape)
/// allocated from a fresh pool must cost at least ten times fewer submissions than the 1D
/// plan — the fleet evidence is thousands of submissions per snapshot dominated by per-
/// submission cost. Debug build, so the bound is generous (the measured ratio is ~300x).
#[test]
fn coalesced_store_is_an_order_of_magnitude_cheaper_than_1d() {
    const TOKENS: usize = 320_000;
    let kv_bytes = TOKENS * KV_BYTES_PER_TOKEN / PAGE_BYTES * PAGE_BYTES + TAIL_BYTES;
    let config = Config {
        bytes: kv_bytes as u64 + 2 * (8 << 20),
        chunk_bytes: 8 << 20,
        store: StoreMode::OnRetain,
        min_tokens: 1,
        ..Config::default()
    };
    // The device holds the source ranges and, above them, the restore targets.
    let engine = StubCopyEngine::new(CopyModel::default(), 2 * kv_bytes, config.bytes as usize);
    let mut cache = HostCache::new(config, Layout::engine(0), engine).expect("cache");
    let mut ranges = Ranges::new(kv_bytes as u64);
    let snap = engine_snapshot(TOKENS, &mut ranges);
    write_snapshot(cache.engine_mut(), &snap);
    let StoreOutcome::Issued(_) = cache.store(&snap, 1) else {
        panic!("expected an issued store");
    };
    let coalesced = cache.engine_mut().submission_count();
    let one_d = one_d_submissions(&snap);
    assert!(
        one_d >= 10 * coalesced,
        "1D plan {one_d} submissions vs coalesced {coalesced}"
    );
    // The store actually lands: settle, tick, restore into fresh destinations, byte equality.
    cache.engine_mut().advance(1_000_000_000);
    let report = cache.tick();
    assert_eq!(report.completed.len(), 1);
    let hit = cache.lookup(&snap.meta.tokens).expect("host hit");
    let mut target_ranges = Ranges::new(2 * kv_bytes as u64);
    target_ranges.addr = ranges.addr;
    let target_pages = std::array::from_fn(|compressor| {
        snap.pages[compressor]
            .iter()
            .map(|page| DevicePage {
                id: DevicePageId {
                    compressor: compressor as u8,
                    page: page.id.page,
                    generation: 2,
                },
                segments: page
                    .segments
                    .iter()
                    .map(|segment| target_ranges.range(segment.bytes))
                    .collect(),
            })
            .collect()
    });
    let restore_target = RestoreTarget {
        pages: target_pages,
        tail: vec![target_ranges.range(TAIL_BYTES)],
        draft: None,
        scores: vec![],
    };
    let outcome = cache.restore(hit.key, &restore_target);
    let RestoreOutcome::Done { .. } = outcome else {
        panic!("expected a completed restore, got {outcome:?}");
    };
    assert_eq!(
        restored_bytes(cache.engine_mut(), &restore_target),
        expected_bytes(&snap)
    );
    assert_eq!(
        cache.engine_mut().submission_count(),
        2 * coalesced,
        "store and restore are one batch each"
    );
}

/// Expand a plan into its ordered unit byte mappings; two plans with equal expansions move
/// exactly the same bytes to the same places in the same order, with no overlaps and no gaps.
fn expand(plan: &[(DeviceRange, HostRange)]) -> Vec<(u64, u32, usize)> {
    plan.iter()
        .flat_map(|&(device, host)| {
            (0..device.bytes as u64)
                .map(move |i| (device.addr + i, host.chunk, host.offset + i as usize))
        })
        .collect()
}

prop_compose! {
    /// A random store-orientation plan: segment-sized copies over a few chunks, with device
    /// and host cursors that drift in and out of contiguity (gap 0 merges, gaps do not) and an
    /// occasional chunk switch on the host side.
    fn random_plan()(seed in any::<u64>(), count in 0usize..24)
        -> Vec<(DeviceRange, HostRange)>
    {
        let mut rng = Rng::new(seed);
        let mut plan = Vec::with_capacity(count);
        let mut addr = 0u64;
        let mut chunk = 0u32;
        let mut offset = 0usize;
        for _ in 0..count {
            // Drift the device and host cursors independently in and out of adjacency.
            addr += rng.below(3) * rng.below(128);
            if rng.below(6) == 0 {
                chunk += 1;
                offset = 0;
            } else {
                offset += (rng.below(3) * rng.below(64)) as usize;
            }
            let bytes = 1 + rng.below(64) as usize;
            plan.push((
                DeviceRange { addr, bytes },
                HostRange { chunk, offset, bytes },
            ));
            addr += bytes as u64;
            offset += bytes;
        }
        plan
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// The coalesced plan covers exactly the same ordered byte mapping as the 1D plan: no
    /// overlaps inside either plan, no gaps, no reordering, and no two merged copies touch on
    /// both sides (otherwise they would have merged).
    #[test]
    fn coalesced_plan_covers_exactly_the_1d_mapping(plan in random_plan()) {
        let merged = coalesce(&plan);
        prop_assert!(merged.len() <= plan.len());
        prop_assert_eq!(expand(&plan), expand(&merged), "byte mapping changed");
        let expansion = expand(&merged);
        let mut sorted = expansion.clone();
        sorted.sort_unstable();
        sorted.dedup();
        prop_assert_eq!(sorted.len(), expansion.len(), "merged plan overlaps itself");
        // No two merged copies are mergeable again: the plan is fully coalesced.
        for pair in merged.windows(2) {
            prop_assert!(
                !(pair[0].0.addr + pair[0].0.bytes as u64 == pair[1].0.addr
                    && pair[0].1.chunk == pair[1].1.chunk
                    && pair[0].1.offset + pair[0].1.bytes == pair[1].1.offset),
                "merged plan is not fully coalesced"
            );
        }
    }

    /// A random snapshot — random page sets per compressor, random per-page segment counts
    /// and sizes with random device gaps, optional draft — round-trips byte-for-byte through
    /// the batch path, one batch per store and per restore.
    #[test]
    fn random_snapshots_round_trip_through_one_batch(
        seed in any::<u64>(),
        pages in 0usize..4,
        has_draft in any::<bool>(),
    ) {
        let mut rng = Rng::new(seed);
        let mut cache = default_cache(8 * CHUNK as u64, StoreMode::OnRetain);
        let layout = layout();
        let mut device = Device::new(common_hc_5::DEVICE_BYTES);
        let mut lists: [Vec<DevicePage>; COMPRESSORS] = std::array::from_fn(|_| Vec::new());
        for (compressor, list) in lists.iter_mut().enumerate() {
            for index in 0..pages {
                let segments = 1 + rng.below(4) as usize;
                let mut ranges = Vec::with_capacity(segments);
                let mut used = 0usize;
                for _ in 0..segments {
                    // Random gaps in and out of contiguity between a page's segments.
                    if !ranges.is_empty() {
                        device.range((rng.below(65) * 16) as usize);
                    }
                    let remaining = layout.page - used;
                    let bytes = (256 + rng.below(769) as usize).min(remaining);
                    ranges.push(device.range(bytes));
                    used += bytes;
                }
                list.push(DevicePage {
                    id: DevicePageId {
                        compressor: compressor as u8,
                        page: index as u32,
                        generation: 1,
                    },
                    segments: ranges,
                });
            }
        }
        let snap = DeviceSnapshot {
            meta: SnapshotMeta {
                kind: SnapshotKind::Turn,
                tokens: tokens(8),
                end: 8,
                has_draft,
            },
            pages: lists,
            tail: device.ranges(1, layout.tail),
            draft: has_draft.then(|| device.ranges(2, layout.draft / 2)),
            scores: device.ranges(1, layout.scores),
        };
        write_snapshot(cache.engine_mut(), &snap);
        let StoreOutcome::Issued(_) = cache.store(&snap, 1) else {
            panic!("expected an issued store");
        };
        assert_eq!(cache.engine_mut().submission_count(), 1, "one store, one batch");
        common_hc_5::settle(&mut cache);
        let report = cache.tick();
        assert_eq!(report.completed.len(), 1);
        let hit = cache.lookup(&tokens(8)).expect("host hit");
        let restore_target = target(&mut device, &snap, 2);
        let RestoreOutcome::Done { .. } = cache.restore(hit.key, &restore_target) else {
            panic!("expected a completed restore");
        };
        assert_eq!(cache.engine_mut().submission_count(), 2);
        prop_assert_eq!(
            restored_bytes(cache.engine_mut(), &restore_target),
            stored_bytes(cache.engine_mut(), &snap),
            "restored bytes differ from the stored bytes"
        );
    }
}
