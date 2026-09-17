//! Criterion benches for the simulator (HC-4 and HC-6): scheduler steps per second at eight
//! lanes, for the agent-loop and churn workloads. The HC-4 group drives the recording cache
//! under the capacity-1024 configuration of record; the HC-6 group drives the real
//! `HostCache<StubCopyEngine, ()>` over the dense fake device at the scaled memory budget.
//! The throughput unit is the scheduler step, the unit of work the interleaver spends.
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ds41rt_hostcache::sim::clocked::{self, ClockedCache};
use ds41rt_hostcache::sim::testing::RecordingCache;
use ds41rt_hostcache::sim::{EngineModel, Simulator, Workload};
use ds41rt_hostcache::{KV_BYTES_PER_TOKEN, PAGE_BYTES, TAIL_BYTES};
use std::hint::black_box;

const SEED: u64 = 0x5EED;

fn steps_of(workload: &Workload) -> u64 {
    Simulator::new(EngineModel::default(), RecordingCache::new(), SEED)
        .run(workload)
        .steps
}

/// Bytes one retained snapshot of `context_tokens` tokens occupies in the pool.
fn snapshot_bytes(context_tokens: usize) -> u64 {
    let pages = context_tokens
        .saturating_mul(KV_BYTES_PER_TOKEN)
        .div_ceil(PAGE_BYTES);
    pages as u64 * PAGE_BYTES as u64 + TAIL_BYTES as u64
}

fn bench_steps(c: &mut Criterion) {
    let model = EngineModel::default();
    let workloads = [
        (
            "agent_loop",
            Workload::AgentLoop {
                sessions: 8,
                turns: 12,
                context_tokens: 32_768,
                new_tokens_per_turn: 256,
                think_ns: 0,
            },
        ),
        (
            "churn",
            Workload::Churn {
                sessions: 48,
                turns: 6,
                context_tokens: 8_192,
                live_ratio: 1.0,
            },
        ),
    ];
    let mut group = c.benchmark_group("sim_steps_per_s");
    for (name, workload) in workloads {
        let steps = steps_of(&workload);
        group.throughput(Throughput::Elements(steps));
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &workload,
            |b, workload| {
                b.iter(|| {
                    let mut sim =
                        Simulator::new(model.clone(), RecordingCache::new(), black_box(SEED));
                    black_box(sim.run(black_box(workload)))
                });
            },
        );
    }
    group.finish();
}

/// The HC-6 group: the real facade over the stub on the dense fake device. The soak showed no
/// step-rate cliff across phases; this group benched the real-cache fast path so a regression
/// shows up here. A run needs a fresh simulator (retained snapshots must stay ancestors of
/// every request), so each measured iteration builds its own ~600 MB of fake memory outside
/// the timed section via `iter_custom`.
fn bench_real_cache_steps(c: &mut Criterion) {
    // Quotas per the HC-6 suites: the churn quota holds the live sessions' snapshots; the
    // agent-loop quota is a third of the live sessions' KV bytes.
    let workloads = [
        (
            "churn",
            Workload::Churn {
                sessions: 48,
                turns: 8,
                context_tokens: 1024,
                live_ratio: 1.0,
            },
            48 * snapshot_bytes(1024),
        ),
        (
            "agent_loop_phase",
            Workload::AgentLoop {
                sessions: 8,
                turns: 10,
                context_tokens: 8_192,
                new_tokens_per_turn: 512,
                think_ns: 1_000_000,
            },
            8 * snapshot_bytes(8_192),
        ),
    ];
    let mut group = c.benchmark_group("sim_hc6_real_cache_steps_per_s");
    for (name, workload, quota) in workloads {
        let model = EngineModel {
            device_pool_tokens: 65_536,
            ..EngineModel::default()
        };
        let cache = ClockedCache::new(clocked::config(quota), clocked::engine(&model, quota))
            .expect("cache builds");
        let steps = Simulator::new(model, cache, SEED).run(&workload).steps;
        group.throughput(Throughput::Elements(steps));
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &workload,
            |b, workload| {
                b.iter_custom(|iters| {
                    let mut total = std::time::Duration::ZERO;
                    for _ in 0..iters {
                        let model = EngineModel {
                            device_pool_tokens: 65_536,
                            ..EngineModel::default()
                        };
                        let cache = ClockedCache::new(
                            clocked::config(quota),
                            clocked::engine(&model, quota),
                        )
                        .expect("cache builds");
                        let mut sim = Simulator::new(model, cache, black_box(SEED));
                        let start = std::time::Instant::now();
                        black_box(sim.run(black_box(workload)));
                        total += start.elapsed();
                    }
                    total
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_steps, bench_real_cache_steps);
criterion_main!(benches);
