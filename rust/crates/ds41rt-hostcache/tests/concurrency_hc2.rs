//! Concurrency suite for packet HC-2. The crate is single-threaded by design, so "concurrency"
//! means interleaving: several lanes of plan/commit/abort/lookup/pin/evict/free steps are run in
//! a seeded, logged order with shared device pages, and the model is checked after every step.
//! A failure prints the schedule log and reproduces from its seed.
mod common;

use common::{apply, Model, Op};
use ds41rt_hostcache::snapshot::testing::snapshots;
use ds41rt_hostcache::snapshot::{DevicePageId, StorePlan};
use ds41rt_hostcache::SnapshotKind;

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

fn device_ids() -> Vec<DevicePageId> {
    (0..6)
        .map(|page| DevicePageId {
            compressor: 0,
            page,
            generation: 0,
        })
        .collect()
}

fn token_sets() -> Vec<Vec<u32>> {
    vec![
        vec![1, 2, 3],
        vec![1, 2, 4],
        vec![1, 2, 3, 5],
        vec![7, 8],
        vec![7, 8, 9],
    ]
}

fn page_list(rng: &mut Rng, ids: &[DevicePageId]) -> Vec<DevicePageId> {
    let count = rng.below(4) as usize;
    (0..count)
        .map(|_| ids[rng.below(ids.len() as u64) as usize])
        .collect()
}

/// A lane's script: plans interleaved with commits and aborts, plus lookups, evictions, frees
/// and pins. A commit or abort with no pending plan is a no-op.
fn lane_script(lane: usize, rng: &mut Rng, ids: &[DevicePageId], tokens: &[Vec<u32>]) -> Vec<Op> {
    (0..24)
        .map(|step| match (lane + step) % 6 {
            0 | 2 => Op::Plan {
                kind: if (lane + step).is_multiple_of(4) {
                    SnapshotKind::Prompt
                } else {
                    SnapshotKind::Turn
                },
                tokens: tokens[rng.below(tokens.len() as u64) as usize].clone(),
                has_draft: rng.below(2) == 0,
                pages: page_list(rng, ids),
            },
            1 => Op::Commit,
            3 => Op::Abort,
            _ => match rng.below(5) {
                0 => Op::Lookup {
                    tokens: tokens[rng.below(tokens.len() as u64) as usize].clone(),
                },
                1 => Op::Evict {
                    quota: rng.below(3) * 1_000_000,
                },
                2 => Op::Free {
                    id: ids[rng.below(ids.len() as u64) as usize],
                },
                3 => Op::Pin { key: rng.below(20) },
                _ => Op::Unpin { key: rng.below(20) },
            },
        })
        .collect()
}

fn run_seed(seed: u64) -> Vec<String> {
    let mut rng = Rng::new(seed);
    let ids = device_ids();
    let tokens = token_sets();
    let mut lanes: Vec<Vec<Op>> = (0..4)
        .map(|lane| lane_script(lane, &mut rng, &ids, &tokens))
        .collect();
    let mut store = snapshots(1 << 24);
    let mut model = Model::new();
    let mut pending: Vec<Option<StorePlan>> = (0..lanes.len()).map(|_| None).collect();
    let mut log = Vec::new();
    let mut now = 0u64;
    while lanes.iter().any(|lane| !lane.is_empty()) {
        let lane = rng.below(lanes.len() as u64) as usize;
        if lanes[lane].is_empty() {
            continue;
        }
        let op = lanes[lane].remove(0);
        log.push(format!("{now}: lane {lane} {op:?}"));
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            apply(&mut store, &mut model, &op, now, &mut pending[lane]);
        }));
        if let Err(payload) = outcome {
            eprintln!("seed {seed}: model mismatch after step {now}");
            for line in &log {
                eprintln!("  {line}");
            }
            std::panic::resume_unwind(payload);
        }
        now += 1;
    }
    assert_eq!(store.len(), model.snapshots.len());
    log
}

#[test]
fn seeded_lanes_interleave_with_shared_pages() {
    let log = run_seed(0x5EED_1234_ABCD_0001);
    assert!(log.len() > 50, "schedule too short: {}", log.len());
    assert!(log[0].starts_with("0: lane "));
}

#[test]
fn every_seed_reproduces_and_keeps_the_invariants() {
    for seed in 1..=8 {
        let first = run_seed(seed);
        let second = run_seed(seed);
        assert_eq!(first, second, "seed {seed} did not reproduce");
    }
}
