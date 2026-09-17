//! Concurrency (interleaving) suite for packet HC-8: several lanes store and restore through
//! the real `HostCache<StubCopyEngine, ()>` in a seeded order, with the virtual clock advanced
//! between operations so batch copies land asynchronously between polls. The crate is
//! single-threaded by design: "concurrency" is the interleaving, and every invariant is
//! checked after every step with the schedule logged, so a failure reproduces from its seed.
//!
//! The suite's specific instrument is `copy_submissions`: every store and every restore is
//! exactly one batch, so the gauge must equal the number of store+restore issues, and the
//! merged plan must stay an order of magnitude below the raw 1D plan across the run.
mod common;

use common::Rng;
use ds41rt_hostcache::cache::{
    DevicePage, DeviceSnapshot, RestoreOutcome, RestoreTarget, StoreOutcome,
};
use ds41rt_hostcache::config::StoreMode;
use ds41rt_hostcache::copy::{CopyEngine, CopyModel, DeviceRange, StubCopyEngine};
use ds41rt_hostcache::pool::testing::{layout, CHUNK};
use ds41rt_hostcache::snapshot::{DevicePageId, SnapshotMeta};
use ds41rt_hostcache::{SnapshotKind, COMPRESSORS};

/// Lanes (conversations) the schedule interleaves.
const LANES: usize = 4;
/// Pages per compressor per snapshot: enough that the merged plan is an order of magnitude
/// below the raw plan (raw = 4 segments/page + tail 1 + scores 1).
const PAGES: usize = 8;
/// Steps per seeded schedule.
const STEPS: usize = 160;

/// Device and pool bytes one snapshot occupies: whole page slabs plus the tail and scores
/// slabs (no draft).
fn snapshot_bytes() -> u64 {
    let layout = layout();
    (PAGES * COMPRESSORS) as u64 * layout.page as u64 + layout.tail as u64 + layout.scores as u64
}

/// The most stores a schedule can issue (2 of 5 ops are stores) plus one in flight per lane:
/// sized so the pool never evicts and no store is ever skipped, keeping every lane's
/// generations resident.
fn quota() -> u64 {
    (2 * STEPS / 5 + LANES) as u64 * snapshot_bytes() + 4 * CHUNK as u64
}

/// The fake device must hold every store's source ranges and every restore's target ranges.
fn device_bytes() -> usize {
    (3 * STEPS / 5) * snapshot_bytes() as usize + (1 << 20)
}

/// A deterministic snapshot for `lane` and `generation`: unique tokens, unique device
/// identities, ranges handed out sequentially from `next_addr`.
fn lane_snapshot(lane: usize, generation: u32, next_addr: &mut u64) -> DeviceSnapshot {
    let layout = layout();
    let mut lists: [Vec<DevicePage>; COMPRESSORS] = std::array::from_fn(|_| Vec::new());
    let mut range = |bytes: usize| {
        let start = *next_addr;
        *next_addr += bytes as u64;
        DeviceRange { addr: start, bytes }
    };
    for (compressor, list) in lists.iter_mut().enumerate() {
        for index in 0..PAGES {
            // Four contiguous segments per page, as the engine allocates them.
            list.push(DevicePage {
                id: DevicePageId {
                    compressor: compressor as u8,
                    page: (lane * 10_000 + index) as u32,
                    generation,
                },
                segments: vec![
                    range(layout.page / 4),
                    range(layout.page / 4),
                    range(layout.page / 4),
                    range(layout.page / 4),
                ],
            });
        }
    }
    DeviceSnapshot {
        meta: SnapshotMeta {
            kind: SnapshotKind::Turn,
            tokens: (0..8u32).map(|t| t ^ (lane as u32 + 1)).collect(),
            end: 8,
            has_draft: false,
        },
        pages: lists,
        tail: vec![range(layout.tail)],
        draft: None,
        scores: vec![range(layout.scores)],
    }
}

/// The expected content of a snapshot: the per-range pattern concatenated in store order.
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
    for segment in &snapshot.scores {
        bytes.extend(pattern(*segment));
    }
    bytes
}

/// Fresh restore destinations mirroring `snapshot`'s shape above `base`.
fn restore_target(snapshot: &DeviceSnapshot, base: u64) -> RestoreTarget {
    let mut next = base;
    let mut range = |bytes: usize| {
        let start = next;
        next += bytes as u64;
        DeviceRange { addr: start, bytes }
    };
    let pages = std::array::from_fn(|compressor| {
        snapshot.pages[compressor]
            .iter()
            .map(|page| DevicePage {
                id: DevicePageId {
                    compressor: compressor as u8,
                    page: page.id.page,
                    generation: page.id.generation + 1,
                },
                segments: page.segments.iter().map(|s| range(s.bytes)).collect(),
            })
            .collect()
    });
    RestoreTarget {
        pages,
        tail: snapshot.tail.iter().map(|s| range(s.bytes)).collect(),
        draft: None,
        scores: snapshot.scores.iter().map(|s| range(s.bytes)).collect(),
    }
}

/// The bytes a restore wrote, in store order.
fn restored_bytes(engine: &StubCopyEngine, target: &RestoreTarget) -> Vec<u8> {
    let mut bytes = Vec::new();
    for list in &target.pages {
        for page in list {
            for segment in &page.segments {
                bytes.extend(engine.read_device(*segment));
            }
        }
    }
    for segment in &target.tail {
        bytes.extend(engine.read_device(*segment));
    }
    for segment in &target.scores {
        bytes.extend(engine.read_device(*segment));
    }
    bytes
}

/// Per-lane state the schedule mirrors.
#[derive(Default)]
struct Lane {
    /// Snapshots retained by generation, with the bytes each must restore to.
    snapshots: Vec<(DeviceSnapshot, Vec<u8>)>,
    /// Store tickets still pending, in issue order per lane.
    pending_ticket: Option<u64>,
    /// Resident snapshots already restored (each restore re-checks content equality).
    restores: usize,
}

/// Run one seeded schedule; returns the schedule log. Every invariant is checked after every
/// step; a panic carries the seed and the log.
fn run_schedule(seed: u64) -> Vec<String> {
    let config = ds41rt_hostcache::config::Config {
        bytes: quota(),
        chunk_bytes: CHUNK as u64,
        store: StoreMode::OnRetain,
        min_tokens: 1,
        ..ds41rt_hostcache::config::Config::default()
    };
    let engine = StubCopyEngine::new(CopyModel::default(), device_bytes(), quota() as usize);
    let mut cache =
        ds41rt_hostcache::cache::HostCache::new(config, layout(), engine).expect("cache");
    let mut lanes: Vec<Lane> = (0..LANES).map(|_| Lane::default()).collect();
    let mut rng = Rng::new(seed);
    let mut log = Vec::new();
    let mut next_addr = 0u64;
    let mut raw_plan_copies = 0u64;
    let mut batches = 0u64;

    for step in 0..STEPS {
        let lane = rng.below(LANES as u64) as usize;
        match rng.below(5) {
            // Store a fresh generation of the lane's snapshot.
            0 | 1 => {
                let generation = lanes[lane].snapshots.len() as u32 + 1;
                let snap = lane_snapshot(lane, generation, &mut next_addr);
                let expected = expected_bytes(&snap);
                // Raw 1D copies this store would cost without coalescing: 4 segments per
                // page plus one tail and one scores copy (see lane_snapshot's shape).
                raw_plan_copies += (4 * PAGES * COMPRESSORS + 2) as u64;
                for list in &snap.pages {
                    for page in list {
                        for segment in &page.segments {
                            cache.engine_mut().write_device(
                                *segment,
                                &(0..segment.bytes)
                                    .map(|i| (segment.addr as usize + i) as u8)
                                    .collect::<Vec<_>>(),
                            );
                        }
                    }
                }
                for segment in snap.tail.iter().chain(snap.scores.iter()) {
                    cache.engine_mut().write_device(
                        *segment,
                        &(0..segment.bytes)
                            .map(|i| (segment.addr as usize + i) as u8)
                            .collect::<Vec<_>>(),
                    );
                }
                let msg = format!("seed {seed} step {step}: store lane {lane} gen {generation}");
                let StoreOutcome::Issued(ticket) = cache.store(&snap, ()) else {
                    panic!("{msg}: store not issued");
                };
                lanes[lane].pending_ticket = Some(ticket.0);
                batches += 1;
                // One batch per store, and the gauge tracks the engine counter.
                let submissions = cache.engine_mut().submission_count();
                assert_eq!(
                    submissions, batches,
                    "seed {seed} step {step}: {submissions} submissions for {batches} batches"
                );
                assert_eq!(
                    cache.metrics().copy_submissions,
                    submissions,
                    "seed {seed} step {step}: gauge diverged from the engine counter\n{log:?}"
                );
                lanes[lane].snapshots.push((snap, expected));
                log.push(format!("{step}: store lane {lane} gen {generation}"));
            }
            // Advance the clock and poll: batches issued earlier land here.
            2 | 3 => {
                let nanos = rng.below(200_000);
                cache.engine_mut().advance(nanos);
                let report = cache.tick();
                for &ticket in &report.completed {
                    for (l, lane_state) in lanes.iter_mut().enumerate() {
                        if lane_state.pending_ticket == Some(ticket.0) {
                            lane_state.pending_ticket = None;
                            log.push(format!("{step}: tick completed lane {l} ticket {ticket:?}"));
                        }
                    }
                }
                assert!(
                    report.failed.is_empty(),
                    "seed {seed} step {step}: stores failed\n{log:?}"
                );
                log.push(format!("{step}: advance {nanos}"));
            }
            // Restore the lane's latest resident generation into fresh destinations.
            _ => {
                let Some((snap, expected)) = lanes[lane].snapshots.last().cloned() else {
                    log.push(format!("{step}: restore skipped (lane {lane} empty)"));
                    continue;
                };
                if lanes[lane].pending_ticket.is_some() {
                    log.push(format!(
                        "{step}: restore skipped (lane {lane} store pending)"
                    ));
                    continue;
                }
                let key = {
                    let hit = cache.lookup(&snap.meta.tokens);
                    match hit {
                        Some(hit) => hit.key,
                        None => {
                            log.push(format!(
                                "{step}: restore skipped (lane {lane} not resident)"
                            ));
                            continue;
                        }
                    }
                };
                let base = next_addr;
                next_addr += snapshot_bytes();
                let target = restore_target(&snap, base);
                let msg = format!("seed {seed} step {step}: restore lane {lane}");
                let RestoreOutcome::Done { .. } = cache.restore(key, &target) else {
                    panic!("{msg}: restore did not complete\n{log:?}");
                };
                batches += 1;
                let submissions = cache.engine_mut().submission_count();
                assert_eq!(
                    submissions, batches,
                    "seed {seed} step {step}: {submissions} submissions for {batches} batches"
                );
                assert_eq!(
                    cache.metrics().copy_submissions,
                    submissions,
                    "seed {seed} step {step}: gauge diverged\n{log:?}"
                );
                assert_eq!(
                    restored_bytes(cache.engine_mut(), &target),
                    expected,
                    "seed {seed} step {step}: restored bytes differ\n{log:?}"
                );
                lanes[lane].restores += 1;
                log.push(format!("{step}: restore lane {lane}"));
            }
        }
    }

    // Quiesce: every pending store completes, and the run's submission totals tell the
    // coalescing story — the merged batches must be an order of magnitude below the raw plan.
    cache.engine_mut().advance(1_000_000_000);
    let report = cache.tick();
    assert!(
        report.failed.is_empty(),
        "seed {seed}: stores failed at quiesce\n{log:?}"
    );
    let submissions = cache.engine_mut().submission_count();
    assert_eq!(
        submissions, batches,
        "seed {seed}: quiesce committed more stores than batches\n{log:?}"
    );
    assert!(
        raw_plan_copies >= 10 * submissions,
        "seed {seed}: raw plan {raw_plan_copies} copies vs {submissions} merged batches"
    );
    log
}

#[test]
fn interleaved_lanes_keep_the_batch_invariants() {
    for seed in 1..=32 {
        run_schedule(seed);
    }
}

#[test]
fn schedules_reproduce_from_their_seed() {
    assert_eq!(run_schedule(7), run_schedule(7));
}
