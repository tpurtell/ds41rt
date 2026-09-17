//! Functional suite for the simulator (HC-4): every workload on its happy path, the documented
//! device-pool exhaustion error path, determinism of the seeded interleaver, TTFT and virtual
//! clock accounting, content fidelity through the recording cache, the recording cache's
//! lookup counters, and a proptest that random workload parameters never violate the
//! invariants.
use ds41rt_hostcache::sim::testing::RecordingCache;
use ds41rt_hostcache::sim::{EngineModel, RunReport, Simulator, Workload, PREFILL_QUANTUM_TOKENS};
use proptest::strategy::Strategy as _;

/// A small-pool model for tests that want device pressure without a huge page pool.
fn small_pool_model(device_pool_tokens: u64) -> EngineModel {
    EngineModel {
        device_pool_tokens,
        ..EngineModel::default()
    }
}

/// The number of requests a workload queues.
fn total_requests(workload: &Workload) -> usize {
    match *workload {
        Workload::AgentLoop {
            sessions, turns, ..
        } => sessions * turns.max(1),
        Workload::Burst { prompts, .. } => prompts,
        Workload::Churn {
            sessions, turns, ..
        } => sessions * turns.max(1),
    }
}

/// The invariants every happy-path run must end with: no step failed, every request was
/// accounted as exactly one of device hit / host hit / cold miss, and every request produced
/// a TTFT sample.
fn assert_happy(report: &RunReport, workload: &Workload) {
    assert!(
        report.invariant_failures.is_empty(),
        "invariants must hold, got {:?}; log tail: {:?}",
        report.invariant_failures,
        report
            .schedule_log
            .iter()
            .rev()
            .take(10)
            .collect::<Vec<_>>()
    );
    let total = total_requests(workload) as u64;
    assert_eq!(
        report.device_hits + report.host_hits + report.misses,
        total,
        "every request is exactly one of device hit, host hit, or miss"
    );
    assert_eq!(
        report.ttft_ns.len() as u64,
        total,
        "every request accounts one TTFT sample"
    );
    assert!(report.steps > 0);
}

#[test]
fn burst_of_cold_prompts_prefills_everything() {
    let workload = Workload::Burst {
        prompts: 12,
        tokens: 4096,
    };
    let mut sim = Simulator::new(EngineModel::default(), RecordingCache::new(), 1);
    let report = sim.run(&workload);
    assert_happy(&report, &workload);
    assert_eq!(report.misses, 12, "cold prompts all miss");
    assert_eq!(report.device_hits, 0);
    assert_eq!(report.host_hits, 0);
    assert_eq!(report.prefilled_tokens, 12 * 4096);
    assert_eq!(report.restored_tokens, 0);
    let cache = sim.cache();
    assert_eq!(cache.stores.len(), 12, "every burst prompt is retained");
    assert!(cache.evictions.is_empty(), "nothing is under pressure");
}

#[test]
fn agent_loop_that_fits_the_device_hits_it_every_turn() {
    let workload = Workload::AgentLoop {
        sessions: 2,
        turns: 6,
        context_tokens: 4096,
        new_tokens_per_turn: 256,
        think_ns: 1_000_000,
    };
    let mut sim = Simulator::new(EngineModel::default(), RecordingCache::new(), 2);
    let report = sim.run(&workload);
    assert_happy(&report, &workload);
    assert_eq!(
        report.misses, 2,
        "only the first turn of each session is cold"
    );
    assert_eq!(
        report.device_hits, 10,
        "every later turn extends the previous one and hits the device bank"
    );
    assert_eq!(report.host_hits, 0);
    let cache = sim.cache();
    assert!(
        cache.evictions.is_empty(),
        "two sessions fit the 24+24 banks and the pool untouched"
    );
    assert_eq!(cache.stores.len(), 14, "2 prompts + 12 turns retained");
}

#[test]
fn churn_beyond_the_device_produces_host_hits() {
    let workload = Workload::Churn {
        sessions: 48,
        turns: 6,
        context_tokens: 2048,
        live_ratio: 1.0,
    };
    let mut sim = Simulator::new(EngineModel::default(), RecordingCache::new(), 3);
    let report = sim.run(&workload);
    assert_happy(&report, &workload);
    assert_eq!(
        report.misses, 48,
        "only the first visit of each session is cold"
    );
    assert!(
        report.device_hits > 0,
        "recently refreshed snapshots still serve some returning visits"
    );
    assert!(
        report.host_hits > 0,
        "evicted snapshots serve the rest through the host cache"
    );
    assert!(
        report.restored_tokens > 0,
        "host hits restore instead of recompute"
    );
    let cache = sim.cache();
    assert!(
        !cache.evictions.is_empty(),
        "the device evicted bank snapshots under churn"
    );
    assert_eq!(
        cache.restores.len(),
        report.host_hits as usize,
        "every host hit ends in a restore"
    );
    // Content fidelity: restores carry exactly the bytes that were stored for that key.
    assert!(cache.bytes_restored() > 0);
    assert!(cache.bytes_restored() <= cache.bytes_stored());
    assert!(
        report
            .schedule_log
            .iter()
            .all(|line| !line.contains("restore failed")),
        "no restore may fail"
    );
}

#[test]
fn same_seed_reproduces_the_report_and_log() {
    let workload = Workload::Churn {
        sessions: 24,
        turns: 4,
        context_tokens: 2048,
        live_ratio: 1.0,
    };
    let first = Simulator::new(EngineModel::default(), RecordingCache::new(), 42).run(&workload);
    let second = Simulator::new(EngineModel::default(), RecordingCache::new(), 42).run(&workload);
    assert_eq!(first, second);
    assert!(!first.schedule_log.is_empty());
}

#[test]
fn different_seeds_produce_different_schedules() {
    let workload = Workload::AgentLoop {
        sessions: 8,
        turns: 3,
        context_tokens: 4096,
        new_tokens_per_turn: 128,
        think_ns: 0,
    };
    let baseline = Simulator::new(EngineModel::default(), RecordingCache::new(), 1).run(&workload);
    let differs = (2..10).any(|seed| {
        Simulator::new(EngineModel::default(), RecordingCache::new(), seed).run(&workload)
            != baseline
    });
    assert!(differs, "at least one other seed must schedule differently");
}

#[test]
fn ttft_is_the_modelled_time_to_first_token() {
    // A lone burst prompt: TTFT is exactly the modelled prefill, quantum rounding included.
    let tokens = 4 * PREFILL_QUANTUM_TOKENS as u32;
    let workload = Workload::Burst { prompts: 1, tokens };
    let model = EngineModel::default();
    let report = Simulator::new(model.clone(), RecordingCache::new(), 7).run(&workload);
    assert_happy(&report, &workload);
    let pure_ns = (tokens as f64 / model.prefill_tokens_per_s * 1e9).ceil() as u64;
    let ttft = report.ttft_ns[0];
    assert!(
        (pure_ns..pure_ns + 1000).contains(&ttft),
        "ttft {ttft} must be the modelled prefill {pure_ns} plus quantum rounding"
    );
    // Device-hit turns answer in microseconds-scale leftover prefill: far below a cold TTFT.
    let agent = Workload::AgentLoop {
        sessions: 1,
        turns: 3,
        context_tokens: tokens,
        new_tokens_per_turn: 32,
        think_ns: 0,
    };
    let report = Simulator::new(model.clone(), RecordingCache::new(), 7).run(&agent);
    assert_happy(&report, &agent);
    assert!(
        report.ttft_ns[1] < ttft / 100,
        "a warm turn must answer far faster than a cold prefill: {:?}",
        report.ttft_ns
    );
}

#[test]
fn now_ns_is_the_sum_of_modelled_prefill_and_decode_time() {
    let model = EngineModel::default();
    let quantum_ns = |tokens: u32, rate: f64| (tokens as f64 / rate * 1e9).ceil() as u64;
    // A lone cold prompt: the clock is exactly its prefills, quantum rounding included.
    let tokens = 3 * PREFILL_QUANTUM_TOKENS as u32;
    let workload = Workload::Burst { prompts: 1, tokens };
    let mut sim = Simulator::new(model.clone(), RecordingCache::new(), 7);
    let report = sim.run(&workload);
    assert_happy(&report, &workload);
    assert_eq!(
        sim.now_ns(),
        3 * quantum_ns(PREFILL_QUANTUM_TOKENS as u32, model.prefill_tokens_per_s),
        "the clock advanced by exactly the modelled prefill time"
    );
    // One turn with decode: prefill of one quantum plus one modelled decode step per token.
    let new_tokens = 5;
    let agent = Workload::AgentLoop {
        sessions: 1,
        turns: 1,
        context_tokens: PREFILL_QUANTUM_TOKENS as u32,
        new_tokens_per_turn: new_tokens,
        think_ns: 0,
    };
    let mut sim = Simulator::new(model.clone(), RecordingCache::new(), 7);
    let report = sim.run(&agent);
    assert_happy(&report, &agent);
    assert_eq!(
        sim.now_ns(),
        quantum_ns(PREFILL_QUANTUM_TOKENS as u32, model.prefill_tokens_per_s)
            + u64::from(new_tokens) * quantum_ns(1, model.decode_tokens_per_s),
        "the clock advanced by exactly the modelled prefill and decode time"
    );
}

#[test]
fn recording_cache_lookup_and_hit_counters_match_the_report() {
    // Every request that misses the device consults the host cache exactly once, so the
    // recording cache's counters must mirror the report's miss and host-hit counts.
    let workload = Workload::Churn {
        sessions: 48,
        turns: 6,
        context_tokens: 2048,
        live_ratio: 1.0,
    };
    let mut sim = Simulator::new(EngineModel::default(), RecordingCache::new(), 3);
    let report = sim.run(&workload);
    assert_happy(&report, &workload);
    let cache = sim.cache();
    assert!(
        report.host_hits > 0 && report.misses > 0,
        "the workload must exercise both host hits and misses"
    );
    assert_eq!(
        cache.lookups,
        report.misses + report.host_hits,
        "one host lookup per request the device missed"
    );
    assert_eq!(
        cache.hits, report.host_hits,
        "one recorded hit per host hit the report counted"
    );
}

#[test]
fn a_simulator_serves_back_to_back_workloads() {
    // State carries over between runs like a live engine: the second burst repeats the first
    // burst's sessions and hits the snapshots the first run retained.
    let burst = || Workload::Burst {
        prompts: 2,
        tokens: 1024,
    };
    let mut sim = Simulator::new(EngineModel::default(), RecordingCache::new(), 3);
    let first = sim.run(&burst());
    let second = sim.run(&burst());
    assert_happy(&first, &burst());
    assert_happy(&second, &burst());
    assert!(
        second.steps < first.steps,
        "hits skip prefill: {} vs {}",
        second.steps,
        first.steps
    );
    assert_eq!(first.misses, 2);
    assert_eq!(
        second.device_hits, 2,
        "returning prompts hit the device bank"
    );
    assert_eq!(second.misses, 0);
}

#[test]
fn device_pool_exhaustion_is_the_documented_error_path() {
    // 8,000 tokens of pool is 78 pages, 19 per compressor: one 4096-token request (10 pages
    // per compressor) fits, two do not, and the banks are empty at admission — so the second
    // admission raises the modelled SourcePoolExhausted.
    let model = small_pool_model(8_000);
    let workload = Workload::Burst {
        prompts: 4,
        tokens: 4096,
    };
    let report = Simulator::new(model, RecordingCache::new(), 5).run(&workload);
    assert_eq!(report.invariant_failures.len(), 1);
    assert!(
        report.invariant_failures[0].contains("SourcePoolExhausted"),
        "got {:?}",
        report.invariant_failures
    );
}

#[test]
fn make_room_evicts_before_it_ever_exhausts() {
    // A pool that holds several requests plus a few bank snapshots: allocations succeed by
    // evicting bank entries, which consult the cache and free pages through it.
    let model = small_pool_model(120_000);
    let workload = Workload::Churn {
        sessions: 24,
        turns: 4,
        context_tokens: 4096,
        live_ratio: 1.0,
    };
    let mut sim = Simulator::new(model, RecordingCache::new(), 9);
    let report = sim.run(&workload);
    assert_happy(&report, &workload);
    let cache = sim.cache();
    assert!(
        !cache.evictions.is_empty(),
        "a tight pool must evict bank snapshots to make room"
    );
    assert!(
        !cache.freed_pages.is_empty(),
        "evicted pages are freed through the cache"
    );
}

// Pools are generated far above the lanes' worst concurrent footprint, so exhaustion — the
// documented error path — cannot trigger inside this property.
proptest::proptest! {
    #[test]
    fn invariants_hold(
        seed in proptest::prelude::any::<u64>(),
        pool_tokens in 400_000u64..5_000_000,
        retain in 2usize..=24,
        completion_ticks in 1u32..=4,
        workload in proptest::prelude::prop_oneof![
            (
                1usize..=8, 1usize..=6, 512u32..=4096, 0u32..=512, proptest::prelude::any::<u64>()
            ).prop_map(|(sessions, turns, context, new, think)| {
                Workload::AgentLoop {
                    sessions,
                    turns,
                    context_tokens: context,
                    new_tokens_per_turn: new,
                    think_ns: think % 10_000_000,
                }
            }),
            (1usize..=16, 512u32..=4096).prop_map(|(prompts, tokens)| {
                Workload::Burst { prompts, tokens }
            }),
            (
                2usize..=48, 1usize..=6, 512u32..=4096, 0.25f64..=1.0
            ).prop_map(|(sessions, turns, context, live_ratio)| {
                Workload::Churn { sessions, turns, context_tokens: context, live_ratio }
            }),
        ],
    ) {
        let model = EngineModel {
            device_pool_tokens: pool_tokens,
            retain,
            ..EngineModel::default()
        };
        let mut sim = Simulator::new(
            model,
            RecordingCache::with_completion_ticks(completion_ticks),
            seed,
        );
        let report = sim.run(&workload);
        assert_happy(&report, &workload);
    }
}

/// Performance floor: a 1,000-step run must finish in seconds even in a debug build, so a
/// broken fast path (e.g. an invariant check that became a scan) fails `cargo test`.
#[test]
fn thousand_step_run_floor() {
    let workload = Workload::AgentLoop {
        sessions: 8,
        turns: 6,
        context_tokens: 8192,
        new_tokens_per_turn: 192,
        think_ns: 0,
    };
    let start = std::time::Instant::now();
    let report = Simulator::new(EngineModel::default(), RecordingCache::new(), 11).run(&workload);
    let elapsed = start.elapsed();
    assert!(
        report.invariant_failures.is_empty(),
        "floor run must be clean: {:?}",
        report.invariant_failures
    );
    assert!(
        report.steps >= 1_000,
        "the floor workload must exceed 1,000 steps, got {}",
        report.steps
    );
    assert!(
        elapsed < std::time::Duration::from_secs(5),
        "{} steps took {elapsed:?} in debug",
        report.steps
    );
}
