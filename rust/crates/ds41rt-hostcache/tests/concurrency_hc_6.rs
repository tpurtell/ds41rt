//! Concurrency (interleaving) suite for packet HC-6: the seeded simulator drives the real
//! `HostCache<StubCopyEngine, ()>` through the dense fake device — eight lanes interleaved by
//! the scheduler while store copies land on the stub's clock between polls, device evictions
//! consult the cache on the evict path, and returning visits restore through the host. The
//! memory budget is scaled (`device_pool_tokens` 65,536) so the stub's real byte arrays stay
//! within the suite's envelope. Every failure carries its seed, and the schedule log
//! reproduces it exactly.
//!
//! Quota policy: with the visit-major churn order, a session's revisit comes exactly
//! `sessions` stores after its snapshot, so an LRU cache yields a host hit only if the quota
//! holds about one snapshot per live session (probed exhaustively during HC-6: halving the
//! quota zeroes the hit rate). The quota is therefore sized to the live sessions' snapshot
//! bytes; see the packet summary's interface-change proposal for the brief's
//! "a third of the live sessions" figure, which cannot produce a hit here.
use ds41rt_hostcache::metrics::Snapshot as Metrics;
use ds41rt_hostcache::sim::clocked::{self, ClockedCache};
use ds41rt_hostcache::sim::{EngineModel, RunReport, Simulator, Workload};
use ds41rt_hostcache::{KV_BYTES_PER_TOKEN, PAGE_BYTES, TAIL_BYTES};

/// The churn run both suites drive: more conversations than the device banks can serve under
/// page pressure, so returning visits miss the device and are served — or prefilled — through
/// the host cache.
const SESSIONS: usize = 48;
const CONTEXT_TOKENS: usize = 1024;
const TURNS: usize = 8;

/// The HC-6 device model of record: 640 pages per compressor (~233 MB of fake device with
/// the tails and drafts arena), well under the suite's 512 MB envelope.
fn model() -> EngineModel {
    EngineModel {
        device_pool_tokens: 65_536,
        ..EngineModel::default()
    }
}

fn workload() -> Workload {
    Workload::Churn {
        sessions: SESSIONS,
        turns: TURNS,
        context_tokens: CONTEXT_TOKENS as u32,
        live_ratio: 1.0,
    }
}

/// Bytes one retained snapshot of `context_tokens` tokens occupies in the pool: whole page
/// slabs plus one tail slab.
fn snapshot_bytes(context_tokens: usize) -> u64 {
    let pages = context_tokens
        .saturating_mul(KV_BYTES_PER_TOKEN)
        .div_ceil(PAGE_BYTES);
    pages as u64 * PAGE_BYTES as u64 + TAIL_BYTES as u64
}

/// The quota of record: the live sessions' snapshot bytes — the smallest quota at which the
/// LRU still serves revisits (see the module doc).
fn quota() -> u64 {
    SESSIONS as u64 * snapshot_bytes(CONTEXT_TOKENS)
}

/// The cache under test: the real facade over the stub, on the dense fake device.
fn sim(seed: u64) -> Simulator<ClockedCache> {
    let model = model();
    let cache = ClockedCache::new(clocked::config(quota()), clocked::engine(&model, quota()))
        .expect("cache builds");
    Simulator::new(model, cache, seed)
}

/// Assert one churn run is clean and the facade's counters agree with the report. Shared by
/// the seeded tests and the performance floor.
fn check_churn(seed: u64, report: &RunReport, metrics: &Metrics) {
    assert!(
        report.invariant_failures.is_empty(),
        "seed {seed:#x}: interleaved run against the real cache must be clean, got {:?}; log tail: {:?}",
        report.invariant_failures,
        report
            .schedule_log
            .iter()
            .rev()
            .take(10)
            .collect::<Vec<_>>()
    );
    assert!(
        report.steps >= 5_000,
        "seed {seed:#x}: churn must run at least 5,000 scheduler steps, got {}",
        report.steps
    );
    assert!(
        report.host_hits > 0,
        "seed {seed:#x}: more live sessions than the device banks hold must hit the host cache"
    );
    assert!(
        report.restored_tokens > 0,
        "seed {seed:#x}: host hits must restore tokens"
    );
    assert!(!report.schedule_log.is_empty(), "the schedule log is empty");

    // The facade's counters must tell the same story as the report.
    assert_eq!(
        metrics.restores, report.host_hits,
        "seed {seed:#x}: every host hit restored (restores == host_hits)"
    );
    assert_eq!(
        metrics.restore_timeouts, 0,
        "seed {seed:#x}: restore budgets never ran out"
    );
    assert_eq!(
        metrics.restore_failures, 0,
        "seed {seed:#x}: no restore failed"
    );
    assert_eq!(metrics.stores_failed, 0, "seed {seed:#x}: no store failed");
    assert_eq!(
        metrics.evict_drops_uncached, 0,
        "seed {seed:#x}: the copy budget never dropped a snapshot uncached"
    );
    assert!(
        metrics.bytes_used <= quota(),
        "seed {seed:#x}: bytes_used {} exceeded the quota {}",
        metrics.bytes_used,
        quota()
    );
    assert!(
        metrics.host_evictions > 0,
        "seed {seed:#x}: the quota must force host eviction"
    );
    assert_eq!(
        metrics.stores_completed,
        metrics.resident_snapshots + metrics.host_evictions + metrics.stores_replaced,
        "seed {seed:#x}: store counters do not balance"
    );
    // Restored bytes cover at least the restored tokens' KV bytes (pages round up, the tail
    // adds more), so the report's restored_tokens and the facade's byte accounting agree.
    assert!(
        metrics.restore_bytes >= report.restored_tokens * KV_BYTES_PER_TOKEN as u64,
        "seed {seed:#x}: restore_bytes {} shy of restored_tokens {}",
        metrics.restore_bytes,
        report.restored_tokens
    );
    // Restore latency: the stub model is exact — every restore is a serial chain on the
    // restore stream of four segment copies per copied page plus one tail copy, each costing
    // the per-copy latency plus its bytes at the modelled bandwidth. A wedged stream or a
    // copy issuing twice would blow past this bound.
    let copy_model = ds41rt_hostcache::copy::CopyModel::default();
    if metrics.restores > 0 {
        // Every restore moves whole slabs: pages of `PAGE_BYTES` plus one `TAIL_BYTES` tail.
        let page_bytes = metrics.restore_bytes - metrics.restores * TAIL_BYTES as u64;
        let copies = 4 * page_bytes / PAGE_BYTES as u64 + metrics.restores;
        let latency_bound = copies * copy_model.per_copy_latency_ns
            + (metrics.restore_bytes as f64 / copy_model.h2d_bytes_per_ns) as u64
            + copies;
        assert!(
            metrics.restore_latency_sum_ns <= latency_bound,
            "seed {seed:#x}: restore latency {} ns exceeds the stub model bound {latency_bound} ns",
            metrics.restore_latency_sum_ns
        );
    }
}

#[test]
fn churn_against_the_real_cache_is_clean() {
    for seed in [0xC0FFEE_u64, 0xD5_20_24_11_07, 42] {
        let mut sim = sim(seed);
        let report = sim.run(&workload());
        let metrics = sim.cache().cache().metrics();
        check_churn(seed, &report, &metrics);
    }
}

#[test]
fn seeded_schedule_reproduces_against_the_real_cache() {
    let run = || sim(0xC0FFEE).run(&workload());
    let first = run();
    let second = run();
    assert_eq!(first, second, "the same seed must reproduce report and log");
    // The log names the interleaving: lane steps, copy-engine ticks, evictions, restores.
    assert!(
        first
            .schedule_log
            .iter()
            .any(|line| line.contains("decision=")),
        "the schedule log records eviction consultations"
    );
    assert!(
        first
            .schedule_log
            .iter()
            .any(|line| line.contains("restore done")),
        "the schedule log records restores"
    );
}

#[test]
fn the_first_wave_fills_every_lane() {
    let report = sim(0xC0FFEE).run(&workload());
    let admitted = report
        .schedule_log
        .iter()
        .filter(|line| line.contains("admit"))
        .take(8)
        .count();
    assert_eq!(admitted, 8, "the opening of the log names eight sessions");
}

#[test]
fn step_rate_stays_above_the_floor() {
    // A broken fast path (or a debug build pathologically slower than expected) fails here,
    // in `cargo test`, not in a human's bench review. The floor is an order of magnitude
    // below the observed debug step rate (~28,000 steps/s on the reference machine).
    let start = std::time::Instant::now();
    let report = sim(0xC0FFEE).run(&workload());
    let elapsed = start.elapsed();
    let steps_per_s = report.steps as f64 / elapsed.as_secs_f64();
    assert!(
        steps_per_s >= 1_000.0,
        "churn ran at {steps_per_s:.0} steps/s ({} steps in {elapsed:?}); the floor is 1,000",
        report.steps
    );
}

#[cfg(test)]
mod properties {
    use ds41rt_hostcache::sim::clocked::{self, ClockedCache};
    use ds41rt_hostcache::sim::{EngineModel, Simulator, Workload};
    use ds41rt_hostcache::{KV_BYTES_PER_TOKEN, PAGE_BYTES};
    use proptest::prelude::*;

    /// Any well-formed small churn run against the real cache is clean: the invariants the
    /// simulator asserts after every step hold, the counters balance, and the pool never
    /// exceeds its quota. Sized small so 256 cases run in seconds.
    #[test]
    fn random_small_churn_runs_are_clean() {
        proptest!(|(seed: u64,
                    sessions in 4usize..16,
                    turns in 2usize..5,
                    context_tokens in 512usize..2_048,
                    device_pool_tokens in 20_000u64..300_000,
                    quota_chunks in 2u64..12,
        )| {
            let model = EngineModel {
                device_pool_tokens,
                ..EngineModel::default()
            };
            // Well-formedness: the live lanes must fit on the device, or the modelled
            // SourcePoolExhausted aborts the run by design.
            let max_len = context_tokens + turns * 16;
            let pages = max_len
                .saturating_mul(KV_BYTES_PER_TOKEN)
                .div_ceil(PAGE_BYTES);
            let pool_pages = device_pool_tokens as usize * KV_BYTES_PER_TOKEN / PAGE_BYTES;
            prop_assume!(pool_pages > 8 * pages);

            let quota = quota_chunks * clocked::CHUNK_BYTES;
            let workload = Workload::Churn {
                sessions,
                turns,
                context_tokens: context_tokens as u32,
                live_ratio: 1.0,
            };
            let cache = ClockedCache::new(clocked::config(quota), clocked::engine(&model, quota))
                .expect("cache builds");
            let mut sim = Simulator::new(model, cache, seed);
            let report = sim.run(&workload);
            prop_assert!(
                report.invariant_failures.is_empty(),
                "seed {seed:#x}: {:?}",
                report.invariant_failures
            );
            let metrics = sim.cache().cache().metrics();
            prop_assert!(metrics.bytes_used <= quota, "bytes_used exceeds quota");
            prop_assert_eq!(
                metrics.stores_completed,
                metrics.resident_snapshots + metrics.host_evictions + metrics.stores_replaced
            );
            prop_assert_eq!(metrics.restore_failures, 0);
            prop_assert_eq!(metrics.stores_failed, 0);
        });
    }
}
