//! Concurrency (interleaving) suite for the pinned slab pool (HC-1). The crate is
//! single-threaded by design, so "concurrency" here means eight lanes whose take/give_back
//! operations are interleaved by a seeded schedule, with releases deferred to model asynchronous
//! copy completions. Every step re-checks the pool's invariants; a failure prints its seed and
//! the tail of the schedule that produced it.
use ds41rt_hostcache::pool::testing::{layout, FakePinned, CHUNK};
use ds41rt_hostcache::pool::{Class, Slab, SlabPool};
use std::collections::HashSet;

const QUOTA: u64 = 4 * CHUNK as u64;
const LANES: usize = 8;
const STEPS: usize = 4_000;
const LOG_TAIL: usize = 200;

/// splitmix64: a tiny deterministic PRNG so a failing schedule reproduces from its seed.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

#[test]
fn eight_lanes_interleaved_near_quota() {
    for seed in 0..8 {
        run(seed);
    }
}

fn run(seed: u64) {
    let mut mem = FakePinned::new(usize::MAX);
    let mut pool = SlabPool::new(QUOTA, CHUNK, layout(), &mut mem).expect("pool");
    let mut rng = Rng::new(seed);
    let mut lanes: Vec<Vec<Slab>> = vec![Vec::new(); LANES];
    // (due step, lane, slab): a release whose asynchronous copy has not completed yet.
    let mut pending: Vec<(usize, usize, Slab)> = Vec::new();
    let mut log: Vec<String> = Vec::new();
    let mut used = 0u64;

    for step in 0..STEPS {
        // Complete every release whose copy has finished.
        let mut index = 0;
        while index < pending.len() {
            if pending[index].0 <= step {
                let (_, lane, slab) = pending.swap_remove(index);
                pool.give_back(slab);
                used -= layout().size(slab.class) as u64;
                log.push(format!("{step}: lane {lane} release {slab:?}"));
            } else {
                index += 1;
            }
        }
        // One lane acts: take a slab, or schedule a release of one it holds.
        let lane = rng.below(LANES);
        let class = Class::ALL[rng.below(Class::ALL.len())];
        let take = rng.below(4) != 0 || lanes[lane].is_empty();
        if take {
            match pool.take(class) {
                Ok(slab) => {
                    lanes[lane].push(slab);
                    used += layout().size(class) as u64;
                    log.push(format!("{step}: lane {lane} take {slab:?}"));
                }
                Err(err) => log.push(format!("{step}: lane {lane} take {class:?} -> {err}")),
            }
        } else {
            let pick = rng.below(lanes[lane].len());
            let slab = lanes[lane].swap_remove(pick);
            let due = step + 1 + rng.below(8);
            pending.push((due, lane, slab));
            log.push(format!(
                "{step}: lane {lane} issue release {slab:?} due {due}"
            ));
        }
        check(&pool, &lanes, &pending, used, seed, step, &log);
    }

    // Drain: complete every outstanding release.
    for (_, lane, slab) in pending.drain(..) {
        pool.give_back(slab);
        used -= layout().size(slab.class) as u64;
        log.push(format!("drain: lane {lane} release {slab:?}"));
    }
    for lane in &mut lanes {
        for slab in lane.drain(..) {
            pool.give_back(slab);
            used -= layout().size(slab.class) as u64;
        }
    }
    assert_eq!(used, 0, "seed {seed}: bytes_used after drain");
    assert_eq!(
        pool.bytes_used(),
        0,
        "seed {seed}: pool bytes_used after drain"
    );
    for (_, occupancy) in pool.occupancy() {
        assert_eq!(occupancy.slabs_in_use, 0, "seed {seed}: slabs still in use");
    }
}

fn check(
    pool: &SlabPool,
    lanes: &[Vec<Slab>],
    pending: &[(usize, usize, Slab)],
    used: u64,
    seed: u64,
    step: usize,
    log: &[String],
) {
    let context = || {
        let tail = &log[log.len().saturating_sub(LOG_TAIL)..];
        format!("seed {seed} step {step}\n{}", tail.join("\n"))
    };
    assert_eq!(pool.bytes_used(), used, "{}", context());
    assert!(used <= QUOTA, "{}", context());
    let mut seen: HashSet<Slab> = HashSet::new();
    let mut in_use = 0u64;
    for slab in lanes
        .iter()
        .flatten()
        .chain(pending.iter().map(|(_, _, slab)| slab))
    {
        assert!(
            seen.insert(*slab),
            "slab handed out twice: {slab:?}\n{}",
            context()
        );
        let range = pool.location(*slab);
        assert!(
            range.offset + range.bytes <= CHUNK,
            "range outside chunk\n{}",
            context()
        );
        in_use += 1;
    }
    let occupied: u64 = pool
        .occupancy()
        .iter()
        .map(|(_, occ)| occ.slabs_in_use)
        .sum();
    assert_eq!(occupied, in_use, "{}", context());
}
