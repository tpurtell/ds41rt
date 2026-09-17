//! Concurrency (interleaving) suite for packet HC-7: the seeded simulator drives the real
//! `HostCache<StubCopyEngine, ()>` through the dense fake device while store copies land on
//! the stub's clock between polls and device evictions consult the cache on the evict path —
//! and the two new instruments must tell the run's exact story: `device_evictions` counts the
//! ticket-ledger's eviction consultations exactly (HC-4's settle-exactly-once contract: every
//! `before_device_evict` call corresponds to one `release_entry` schedule-log line), and the
//! store-latency histogram holds exactly one entry per completed store at quiescence, whether
//! the completion was observed by `tick` or by an evict path's bounded wait. Every failure
//! carries its seed, and the schedule log reproduces it exactly.
//!
//! Quota policy: same as the HC-6 suite — sized to the live sessions' snapshot bytes, the
//! smallest quota at which the LRU still serves revisits under visit-major churn.
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

/// The modelled store copy's floor: one per-copy latency per store, since every store copies
/// at least its tail.
const PER_COPY_LATENCY_NS: u64 = 10_000;

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

/// The quota of record: the live sessions' snapshot bytes (see the module doc).
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

/// The eviction consultations the device model logged: one `release_entry` line per
/// `before_device_evict` call, covering both `make_room` evictions and bank overflows.
fn eviction_consultations(report: &RunReport) -> u64 {
    report
        .schedule_log
        .iter()
        .filter(|line| line.starts_with("evict bank snapshot"))
        .count() as u64
}

/// Assert one churn run is clean and the new instruments agree with the run's ledger. Shared
/// by the seeded tests, the floor test and the proptest.
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
        report.host_hits > 0,
        "seed {seed:#x}: more live sessions than the device banks hold must hit the host cache"
    );

    // One counted device eviction per eviction consultation: the engine evicted that many
    // snapshots from its banks (or via make_room) and consulted the cache on each, before
    // any budget wait and whatever the decision. The disabled-cache case never arises here.
    assert_eq!(
        metrics.device_evictions,
        eviction_consultations(report),
        "seed {seed:#x}: device_evictions must count every before_device_evict consultation"
    );
    assert!(
        metrics.device_evictions >= metrics.evict_waits,
        "seed {seed:#x}: device_evictions {} < evict_waits {}",
        metrics.device_evictions,
        metrics.evict_waits
    );
    assert!(
        metrics.device_evictions >= metrics.evict_drops_uncached,
        "seed {seed:#x}: device_evictions {} < evict_drops_uncached {}",
        metrics.device_evictions,
        metrics.evict_drops_uncached
    );
    // The run is quiescent (the simulator drained every ticket), so the histogram holds
    // exactly one entry per completed store — completions observed by `tick` and by evict
    // waits alike — and each entry is a real modelled copy time, not a zero.
    let bucket_entries: u64 = metrics.store_latency_buckets.iter().sum();
    assert_eq!(
        bucket_entries, metrics.stores_completed,
        "seed {seed:#x}: the store-latency histogram must hold exactly one entry per completed store"
    );
    assert!(
        metrics.store_latency_sum_ns >= metrics.stores_completed * PER_COPY_LATENCY_NS,
        "seed {seed:#x}: store_latency_sum_ns {} is shy of {} completions",
        metrics.store_latency_sum_ns,
        metrics.stores_completed
    );
}

#[test]
fn churn_instrumentation_matches_the_ticket_ledger() {
    for seed in [0xC0FFEE_u64, 0xD5_20_24_11_07, 42] {
        let mut sim = sim(seed);
        let report = sim.run(&workload());
        let metrics = sim.cache().cache().metrics();
        check_churn(seed, &report, &metrics);
        // The churn pressure case exercises both completion paths: most stores commit on a
        // tick, and an evict-path wait commits the rest. Both land in the histogram.
        assert!(
            metrics.stores_completed > 0,
            "seed {seed:#x}: churn must complete stores"
        );
        assert!(
            metrics.store_latency_sum_ns > 0,
            "seed {seed:#x}: completed stores must record latency"
        );
    }
}

#[test]
fn seeded_schedule_reproduces_the_instrumentation() {
    let run = || sim(0xC0FFEE).run(&workload());
    let first = run();
    let second = run();
    assert_eq!(first, second, "the same seed must reproduce report and log");
    // The log names the interleaving the counters must match: eviction consultations,
    // ticks, restores.
    assert!(
        eviction_consultations(&first) > 0,
        "churn must evict from the device banks"
    );
    assert!(
        first
            .schedule_log
            .iter()
            .any(|line| line.starts_with("tick")),
        "the schedule log records copy-engine ticks"
    );
}

#[test]
fn step_rate_stays_above_the_floor() {
    // A broken fast path (or a debug build pathologically slower than expected) fails here,
    // in `cargo test`, not in a human's bench review. The floor is an order of magnitude
    // below the observed debug step rate (~28,000 steps/s on the reference machine).
    let start = std::time::Instant::now();
    let mut sim = sim(0xC0FFEE);
    let report = sim.run(&workload());
    let elapsed = start.elapsed();
    let steps_per_s = report.steps as f64 / elapsed.as_secs_f64();
    assert!(
        steps_per_s >= 1_000.0,
        "churn ran at {steps_per_s:.0} steps/s ({} steps in {elapsed:?}); the floor is 1,000",
        report.steps
    );
    let metrics = sim.cache().cache().metrics();
    check_churn(0xC0FFEE, &report, &metrics);
}

#[cfg(test)]
mod properties {
    use ds41rt_hostcache::metrics::Snapshot as Metrics;
    use ds41rt_hostcache::sim::clocked::{self, ClockedCache};
    use ds41rt_hostcache::sim::{EngineModel, RunReport, Simulator, Workload};
    use ds41rt_hostcache::{KV_BYTES_PER_TOKEN, PAGE_BYTES};
    use proptest::prelude::*;

    /// One `release_entry` schedule-log line per `before_device_evict` call.
    fn eviction_consultations(report: &RunReport) -> u64 {
        report
            .schedule_log
            .iter()
            .filter(|line| line.starts_with("evict bank snapshot"))
            .count() as u64
    }

    /// Any well-formed small churn run keeps the HC-7 instruments exact. Sized small so 256
    /// cases run in seconds.
    #[test]
    fn random_small_churn_keeps_the_instruments_exact() {
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
            let metrics: Metrics = sim.cache().cache().metrics();
            prop_assert_eq!(
                metrics.device_evictions,
                eviction_consultations(&report),
                "seed {:#x}: one device_eviction per consultation",
                seed
            );
            prop_assert!(metrics.device_evictions >= metrics.evict_waits);
            prop_assert!(metrics.device_evictions >= metrics.evict_drops_uncached);
            prop_assert_eq!(
                metrics.store_latency_buckets.iter().sum::<u64>(),
                metrics.stores_completed,
                "seed {:#x}: one histogram entry per completed store at quiescence",
                seed
            );
        });
    }
}
