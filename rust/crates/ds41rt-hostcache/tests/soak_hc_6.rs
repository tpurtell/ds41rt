//! The simulator-driven soak (packet HC-6): `Workload::AgentLoop` over 64 sessions × 40 turns
//! (8k-token contexts growing 512 tokens per turn), run in phases of 10 turns on the real
//! `HostCache<StubCopyEngine, ()>`. The accumulated live set far exceeds the scaled device
//! model (`device_pool_tokens` 65,536), so the device makes room through the eviction path
//! all run long, the cache quota (a third of the live sessions' KV bytes) forces host
//! eviction, and the pool turns over continuously. After every phase — every store ticket
//! drained — the suite asserts the occupancy invariant (a growth-aware baseline: contexts
//! legitimately grow, a leak would not), the counter balance, the stub model's exact restore
//! latency bound, and the absence of invariant failures.
//!
//! Note: the session-major `AgentLoop` queue runs one session at a time (a finished session's
//! next turn sits at the FIFO head, so `admit` never skips it; see the packet summary's
//! interface-change proposal). The device therefore serves every revisit and the host cache
//! is exercised on the store/evict path; restores are covered by the HC-6 churn suite.
use ds41rt_hostcache::metrics::Snapshot as Metrics;
use ds41rt_hostcache::sim::clocked::{self, ClockedCache};
use ds41rt_hostcache::sim::{EngineModel, RunReport, Simulator, Workload};
use ds41rt_hostcache::{KV_BYTES_PER_TOKEN, PAGE_BYTES, TAIL_BYTES};
use std::time::{Duration, Instant};

const SESSIONS: usize = 64;
const TURNS: usize = 40;
const PHASE_TURNS: usize = 10;
const CONTEXT_TOKENS: usize = 8_192;
const NEW_TOKENS_PER_TURN: u32 = 512;
/// A millisecond between turns: models a session's think time without disturbing the
/// prefill-dominated turn duration.
const THINK_NS: u64 = 1_000_000;

/// The soak's device model: 640 pages per compressor; the live set (64 sessions × 8k+ tokens)
/// exceeds it by an order of magnitude.
fn model() -> EngineModel {
    EngineModel {
        device_pool_tokens: 65_536,
        ..EngineModel::default()
    }
}

/// The quota of record: a third of the live sessions' KV bytes (~155 MB), forcing host
/// eviction all run long.
fn quota() -> u64 {
    clocked::third_of_live(SESSIONS, CONTEXT_TOKENS)
}

/// The phase-`phase` workload: ten turns per session, the context continuing where the
/// previous phase left off so prefixes chain across phases.
fn phase_workload(phase: usize) -> Workload {
    Workload::AgentLoop {
        sessions: SESSIONS,
        turns: PHASE_TURNS,
        context_tokens: (CONTEXT_TOKENS + phase * PHASE_TURNS * NEW_TOKENS_PER_TURN as usize)
            as u32,
        new_tokens_per_turn: NEW_TOKENS_PER_TURN,
        think_ns: THINK_NS,
    }
}

/// Whole pages `tokens` tokens occupy.
fn pages_for(tokens: usize) -> u64 {
    (tokens.saturating_mul(KV_BYTES_PER_TOKEN) / PAGE_BYTES) as u64
}

/// What one soak measured; printed and returned for the JSON summary.
#[derive(Debug)]
struct SoakSummary {
    phases: usize,
    steps: u64,
    simulated_ns: u64,
    wall: Duration,
    metrics: Metrics,
}

/// Assert the between-phase invariants of one phase and print its line. `baseline` is the
/// pool occupancy after phase one; `phase` is zero-based.
fn check_phase(seed: u64, phase: usize, baseline: u64, report: &RunReport, metrics: &Metrics) {
    assert!(
        report.invariant_failures.is_empty(),
        "seed {seed:#x} phase {phase}: soak phase must be clean, got {:?}; log tail: {:?}",
        report.invariant_failures,
        report
            .schedule_log
            .iter()
            .rev()
            .take(10)
            .collect::<Vec<_>>()
    );
    assert!(
        metrics.bytes_used <= quota(),
        "seed {seed:#x} phase {phase}: bytes_used {} exceeded the quota {}",
        metrics.bytes_used,
        quota()
    );
    // No leak: occupancy returns to the phase-one baseline plus the pages the contexts
    // legitimately grew — every resident snapshot holds at most `growth` more pages than at
    // baseline, and the tail slab never changes.
    let context_end = CONTEXT_TOKENS + (phase + 1) * PHASE_TURNS * NEW_TOKENS_PER_TURN as usize;
    let context_base = CONTEXT_TOKENS + PHASE_TURNS * NEW_TOKENS_PER_TURN as usize;
    let growth = (pages_for(context_end) - pages_for(context_base)) * PAGE_BYTES as u64;
    let allowance = baseline + metrics.resident_snapshots * growth;
    assert!(
        metrics.bytes_used <= allowance,
        "seed {seed:#x} phase {phase}: bytes_used {} exceeded the leak bound {allowance} \
        (baseline {baseline} + {} residents x {growth} growth)",
        metrics.bytes_used,
        metrics.resident_snapshots
    );
    // Counters balance: every completed store is resident, evicted, or a replacement.
    assert_eq!(
        metrics.stores_completed,
        metrics.resident_snapshots + metrics.host_evictions + metrics.stores_replaced,
        "seed {seed:#x} phase {phase}: store counters do not balance"
    );
    // Leak and failure signals: no held slabs, no failed stores, no uncached drops.
    assert_eq!(metrics.stores_failed, 0, "seed {seed:#x} phase {phase}");
    assert_eq!(
        metrics.store_drain_timeouts, 0,
        "seed {seed:#x} phase {phase}: a held plan leaks slabs"
    );
    assert_eq!(
        metrics.evict_drops_uncached, 0,
        "seed {seed:#x} phase {phase}"
    );
    // Restore latency within the stub model's exact expectation (skipped when the phase
    // restored nothing; the churn suite covers restores exhaustively).
    if metrics.restores > 0 {
        let copy_model = ds41rt_hostcache::copy::CopyModel::default();
        let page_bytes = metrics.restore_bytes - metrics.restores * TAIL_BYTES as u64;
        let copies = 4 * page_bytes / PAGE_BYTES as u64 + metrics.restores;
        let latency_bound = copies * copy_model.per_copy_latency_ns
            + (metrics.restore_bytes as f64 / copy_model.h2d_bytes_per_ns) as u64
            + copies;
        assert!(
            metrics.restore_latency_sum_ns <= latency_bound,
            "seed {seed:#x} phase {phase}: restore latency {} ns exceeds the stub model bound {latency_bound} ns",
            metrics.restore_latency_sum_ns
        );
    }
    println!(
        "seed={seed:#x} phase={phase} steps={} device_hits={} host_hits={} misses={} \
        restored_tokens={} resident={} bytes_used={} host_evictions={} restores={}",
        report.steps,
        report.device_hits,
        report.host_hits,
        report.misses,
        report.restored_tokens,
        metrics.resident_snapshots,
        metrics.bytes_used,
        metrics.host_evictions,
        metrics.restores,
    );
}

/// Run the phased soak (40 turns in 4 phases of 10) on a fresh simulator at `seed`,
/// asserting the between-phase invariants after every phase.
fn run_phased_soak(seed: u64) -> SoakSummary {
    let model = model();
    let cache = ClockedCache::new(clocked::config(quota()), clocked::engine(&model, quota()))
        .expect("cache builds");
    let mut sim = Simulator::new(model, cache, seed);
    let start = Instant::now();
    let mut baseline = None;
    let mut steps = 0;
    for phase in 0..TURNS / PHASE_TURNS {
        let report = sim.run(&phase_workload(phase));
        steps += report.steps;
        let metrics = sim.cache().cache().metrics();
        let occ = metrics.bytes_used;
        if phase == 0 {
            baseline = Some(occ);
        }
        check_phase(seed, phase, baseline.unwrap_or(occ), &report, &metrics);
    }
    let metrics = sim.cache().cache().metrics();
    assert!(
        metrics.host_evictions > 0,
        "seed {seed:#x}: the quota never forced a host eviction"
    );
    assert!(
        metrics.stores_completed > 0,
        "seed {seed:#x}: the soak stored nothing"
    );
    let summary = SoakSummary {
        phases: TURNS / PHASE_TURNS,
        steps,
        simulated_ns: sim.now_ns(),
        wall: start.elapsed(),
        metrics,
    };
    println!(
        "soak summary: seed={seed:#x} phases={} steps={} simulated={:.1}s wall={:?} \
        stores_completed={} resident={} host_evictions={} bytes_used={} quota={} \
        restores={} restore_failures={}",
        summary.phases,
        summary.steps,
        summary.simulated_ns as f64 / 1e9,
        summary.wall,
        summary.metrics.stores_completed,
        summary.metrics.resident_snapshots,
        summary.metrics.host_evictions,
        summary.metrics.bytes_used,
        summary.metrics.quota_bytes,
        summary.metrics.restores,
        summary.metrics.restore_failures,
    );
    summary
}

#[test]
#[ignore = "soak: run explicitly with `cargo test -p ds41rt-hostcache -- --ignored soak`"]
fn soak_phases_have_no_leak_and_balance_the_counters() {
    let summary = run_phased_soak(0x50AC_0006);
    assert_eq!(summary.phases, TURNS / PHASE_TURNS);
}

/// The wall-clock variant: repeat the phase loop with fresh seeds until the 20-minute
/// deadline, asserting the same invariants after every phase of every iteration.
#[test]
#[ignore = "soak: run explicitly with `cargo test -p ds41rt-hostcache -- --ignored soak`"]
fn soak_wall_clock_twenty_minutes() {
    let seconds = std::env::var("HC6_SOAK_WALL_SECONDS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(20 * 60);
    let deadline = Instant::now() + Duration::from_secs(seconds);
    let mut iterations = 0;
    while Instant::now() < deadline {
        let seed = 0x50AC_0006u64
            .wrapping_add(iterations)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15);
        run_phased_soak(seed);
        iterations += 1;
    }
    assert!(
        iterations > 0,
        "the deadline passed before one soak iteration completed"
    );
    println!("wall-clock soak: {iterations} iterations in {seconds}s");
}
