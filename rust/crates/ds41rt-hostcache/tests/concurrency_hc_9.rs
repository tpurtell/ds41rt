//! Concurrency (interleaving) suite for packet HC-9: the prefill pacing hold under a seeded
//! interleaving of stores, device evictions, holds from both scheduler lanes, poll ticks and
//! clock advances that land copies between polls. The crate is single-threaded by design, so
//! "concurrency" is the interleave: a small xorshift interleaver inside this file drives the
//! real `HostCache<StubCopyEngine, Payload>` and, after every single step, re-checks the
//! packet's invariants: one hold never waits longer than `store_pace_ns`, the counted sum
//! never exceeds `holds * pace`, timeouts never exceed holds, and with the knob at zero no
//! hold metric ever moves. Every run is a pure function of its seed, so a failure reproduces
//! from the seed alone; the run's log is kept for humans. Also carries the performance
//! floor test (debug build) so a broken fast path fails `cargo test`, not a human.
mod common_hc_5;

use common_hc_5::{cache as build_cache, config as base_config, snapshot, tokens, write_snapshot};
use common_hc_5::{Device, Payload, DEVICE_BYTES};
use ds41rt_hostcache::cache::{HostCache, StoreOutcome, StoreTicket};
use ds41rt_hostcache::config::{Config, StoreMode};
use ds41rt_hostcache::copy::{CopyEngine, CopyModel, StubCopyEngine};
use ds41rt_hostcache::pool::testing::CHUNK;
use ds41rt_hostcache::SnapshotKind;
use proptest::prelude::*;

/// The suite's pace: 100 µs, *below* the modelled ~211 µs store duration, so holds on
/// overdue stores exercise both outcomes — the copy landing mid-hold and the timeout.
const PACE_NS: u64 = 100_000;

/// One scheduler poll quantum (HC-4's model): a tick advances the clock this far, so a store
/// issued this quantum completes by the next poll unless a hold already waited it clean.
const POLL_QUANTUM_NS: u64 = 1_000_000;

type TestCache = HostCache<StubCopyEngine, Payload>;

/// xorshift64*: the seeded interleaver's only randomness. One seed fixes a whole run.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// A value below `n` (`n > 0`).
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// The interleaved operations; `lane` is the scheduler lane an op belongs to (both lanes
/// hold, exactly as the daemon's two-lane prefill does).
#[derive(Clone, Copy, Debug)]
enum Op {
    Store { lane: u64 },
    Evict { lane: u64 },
    Hold { lane: u64 },
    Tick { lane: u64 },
    Advance { lane: u64 },
}

/// Pick the next op: stores and holds dominate, evictions and ticks keep the pool and the
/// pending list turning, advances land copies between polls.
fn op(rng: &mut Rng) -> Op {
    let lane = rng.below(2);
    match rng.below(10) {
        0..=2 => Op::Store { lane },
        3..=4 => Op::Evict { lane },
        5..=7 => Op::Hold { lane },
        8 => Op::Tick { lane },
        _ => Op::Advance { lane },
    }
}

/// The invariants checked after every step. `pace == 0` is the knob-off rule.
fn check_invariants(cache: &TestCache, pace: u64, log: &[String]) -> Result<(), String> {
    let metrics = cache.metrics();
    let fail = |what: &str| {
        let tail = log
            .iter()
            .rev()
            .take(8)
            .rev()
            .cloned()
            .collect::<Vec<_>>()
            .join("\n");
        Err(format!("{what}\n  log tail:\n{tail}"))
    };
    if metrics.prefill_hold_timeouts > metrics.prefill_holds {
        return fail("prefill_hold_timeouts exceeded prefill_holds");
    }
    if pace == 0 {
        if metrics.prefill_holds != 0 || metrics.prefill_hold_ns_sum != 0 {
            return fail("the knob is off but hold metrics moved");
        }
    } else if metrics.prefill_hold_ns_sum > metrics.prefill_holds * pace {
        return fail("prefill_hold_ns_sum exceeded holds * pace");
    }
    Ok(())
}

/// Run one seeded interleaving of `steps` operations. Deterministic in `seed`; every
/// invariant violation names the step, the lane and the recent log.
fn run(seed: u64, steps: usize, pace: u64) -> Result<(), String> {
    let config = Config {
        store_pace_ns: pace,
        ..base_config(4 * CHUNK as u64, StoreMode::OnRetain)
    };
    let mut cache = build_cache(config, CopyModel::default(), DEVICE_BYTES);
    let mut rng = Rng(seed);
    let mut device = Device::new(DEVICE_BYTES);
    let mut generation = 0u32;
    let mut tickets: Vec<StoreTicket> = Vec::new();
    let mut log = Vec::new();
    for step in 0..steps {
        let op = op(&mut rng);
        match op {
            Op::Store { lane } => {
                generation += 1;
                let snap = snapshot(
                    &mut device,
                    SnapshotKind::Turn,
                    &tokens(8),
                    1,
                    false,
                    generation,
                );
                write_snapshot(cache.engine_mut(), &snap);
                match cache.store(&snap, u64::from(generation)) {
                    StoreOutcome::Issued(ticket) => {
                        log.push(format!("step {step} lane {lane}: store issued {ticket:?}"));
                        tickets.push(ticket);
                    }
                    StoreOutcome::Skipped(reason) => {
                        log.push(format!("step {step} lane {lane}: store skipped {reason:?}"));
                    }
                    StoreOutcome::Deferred(_) => {
                        return Err(format!(
                            "step {step} lane {lane}: unexpected deferred store"
                        ));
                    }
                }
            }
            Op::Evict { lane } => {
                let ticket = if tickets.is_empty() {
                    None
                } else {
                    let index = rng.below(tickets.len() as u64) as usize;
                    Some(tickets.swap_remove(index))
                };
                let decision = cache.before_device_evict(ticket);
                log.push(format!(
                    "step {step} lane {lane}: evict {ticket:?} -> {decision:?}"
                ));
            }
            Op::Hold { lane } => {
                let before = cache.engine_mut().now_ns();
                let result = cache.prefill_hold();
                let elapsed = cache.engine_mut().now_ns() - before;
                if result.is_err() {
                    return Err(format!("step {step} lane {lane}: prefill_hold failed"));
                }
                if elapsed > pace {
                    return Err(format!(
                        "step {step} lane {lane}: hold waited {elapsed} ns against a {pace} ns pace"
                    ));
                }
                log.push(format!("step {step} lane {lane}: hold {elapsed} ns"));
            }
            Op::Tick { lane } => {
                cache.engine_mut().advance(POLL_QUANTUM_NS);
                let report = cache.tick();
                let done: Vec<StoreTicket> = report
                    .completed
                    .iter()
                    .chain(&report.failed)
                    .copied()
                    .collect();
                tickets.retain(|ticket| !done.contains(ticket));
                log.push(format!(
                    "step {step} lane {lane}: tick +{} -{}",
                    report.completed.len(),
                    report.failed.len()
                ));
            }
            Op::Advance { lane } => {
                let nanos = rng.below(10 * POLL_QUANTUM_NS);
                cache.engine_mut().advance(nanos);
                log.push(format!("step {step} lane {lane}: advance {nanos} ns"));
            }
        }
        check_invariants(&cache, pace, &log)
            .map_err(|error| format!("step {step} lane {}: {error}", op_lane(&op)))?;
    }
    Ok(())
}

fn op_lane(op: &Op) -> u64 {
    match *op {
        Op::Store { lane }
        | Op::Evict { lane }
        | Op::Hold { lane }
        | Op::Tick { lane }
        | Op::Advance { lane } => lane,
    }
}

/// Fixed seeds at the suite's pace: both lanes interleave holds with stores, evictions,
/// ticks and advances; the invariants hold on every step.
#[test]
fn interleaved_holds_never_exceed_the_pace() {
    for seed in [0x9E37_79B9_7F4A_7C15, 1, 0xDEAD_BEEF, 987_654_321] {
        run(seed, 600, PACE_NS).unwrap_or_else(|error| panic!("seed {seed:#x}: {error}"));
    }
}

/// With the knob at zero the interleaving runs but the guard never engages: no hold waits,
/// no hold metric ever moves.
#[test]
fn knob_off_interleaving_freezes_the_hold_metrics() {
    for seed in [0x9E37_79B9_7F4A_7C15, 7] {
        run(seed, 600, 0).unwrap_or_else(|error| panic!("seed {seed:#x}: {error}"));
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Random seeds, both knob settings: the invariants hold on every step of every run.
    #[test]
    fn arbitrary_seeds_hold_the_invariants(seed in any::<u64>(), pace in prop_oneof![Just(0u64), Just(PACE_NS)]) {
        run(seed, 150, pace).map_err(|error| TestCaseError::Fail(error.into()))?;
    }
}

/// Order-of-magnitude floor (debug build): the hold decision is O(pending) with no
/// allocation, so even the knob-on decision paths run far below a microsecond each; a
/// broken fast path (an allocation or an unbounded scan per call) blows past these.
#[test]
fn hold_decision_floor() {
    let decisions = 100_000;
    // Knob off: the pure no-op.
    let mut cache = build_cache(
        Config {
            store_pace_ns: 0,
            ..base_config(4 * CHUNK as u64, StoreMode::OnRetain)
        },
        CopyModel::default(),
        DEVICE_BYTES,
    );
    let started = std::time::Instant::now();
    for _ in 0..decisions {
        cache.prefill_hold().expect("hold");
    }
    assert!(
        started.elapsed() < std::time::Duration::from_millis(500),
        "knob-off hold decision too slow: {:?} for {decisions} calls",
        started.elapsed()
    );

    // Knob on, nothing pending: one scan of an empty pending list, one clock read.
    let mut cache = build_cache(
        Config {
            store_pace_ns: PACE_NS,
            ..base_config(4 * CHUNK as u64, StoreMode::OnRetain)
        },
        CopyModel::default(),
        DEVICE_BYTES,
    );
    let started = std::time::Instant::now();
    for _ in 0..decisions {
        cache.prefill_hold().expect("hold");
    }
    assert!(
        started.elapsed() < std::time::Duration::from_millis(500),
        "nothing-pending hold decision too slow: {:?} for {decisions} calls",
        started.elapsed()
    );

    // Knob on, a pending store within its pace: the scan sees one entry, no wait.
    let mut device = Device::new(DEVICE_BYTES);
    let snap = snapshot(&mut device, SnapshotKind::Turn, &tokens(8), 1, false, 1);
    write_snapshot(cache.engine_mut(), &snap);
    assert!(matches!(cache.store(&snap, 1), StoreOutcome::Issued(_)));
    let started = std::time::Instant::now();
    for _ in 0..decisions {
        cache.prefill_hold().expect("hold");
    }
    assert!(
        started.elapsed() < std::time::Duration::from_millis(500),
        "within-pace hold decision too slow: {:?} for {decisions} calls",
        started.elapsed()
    );

    // Knob on, an overdue stalled store: each call burns the pace on the virtual clock
    // (100 µs of model time per call), which must stay far below the wall-clock floor.
    // Commit the first store away first, so the hold waits on the stalled one.
    cache.engine_mut().advance(10 * PACE_NS);
    cache.tick();
    assert_eq!(cache.metrics().stores_completed, 1);
    cache
        .engine_mut()
        .inject(ds41rt_hostcache::copy::CopyFault::StreamStalls(
            ds41rt_hostcache::copy::Stream::Store,
        ));
    let mut device = Device::new(DEVICE_BYTES);
    let snap = snapshot(&mut device, SnapshotKind::Turn, &tokens(8), 1, false, 2);
    write_snapshot(cache.engine_mut(), &snap);
    assert!(matches!(cache.store(&snap, 2), StoreOutcome::Issued(_)));
    cache.engine_mut().advance(10 * PACE_NS);
    let started = std::time::Instant::now();
    for _ in 0..decisions {
        cache.prefill_hold().expect("hold");
    }
    assert!(
        started.elapsed() < std::time::Duration::from_millis(500),
        "overdue hold decision too slow: {:?} for {decisions} calls",
        started.elapsed()
    );
    let metrics = cache.metrics();
    assert_eq!(metrics.prefill_holds, decisions as u64);
    assert_eq!(metrics.prefill_hold_timeouts, decisions as u64);
}
