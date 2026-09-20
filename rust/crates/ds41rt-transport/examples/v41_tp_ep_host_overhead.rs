//! CPU-only host-overhead bench for the native replicated-group wire path.
//!
//! Measures only host CPU cost (no GPU, no network, no hosts): the real
//! `ReplicatedExpertScheduler` planning cost and greedy-vs-modulo imbalance,
//! the actual `with_native_group_owners` encode+validate cost, the coordinator
//! dispatch-side ownership validation, the worker `parse_native_group` cost, and
//! `ExpertProtocolV2Request` clone/encode cost as the transport reference.
//!
//! Every measured closure black-boxes both its inputs and its outputs (including
//! the scheduler assignment/group-load slices and the modulo owner/load buffers)
//! so the optimizer can neither constant-fold the work nor discard the stores.
//!
//! Run with:
//! `CARGO_TARGET_DIR=~/.cache/ds41rt/builds/tp-ep-host-overhead \
//!   cargo run --release --example v41_tp_ep_host_overhead -p ds41rt-transport`

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use anyhow::Result;
use ds41rt_core::{
    replicated_expert_tie_seed, ReplicatedExpertCostModel, ReplicatedExpertScheduleConfig,
    ReplicatedExpertScheduler, INACTIVE_REPLICATED_EXPERT_GROUP,
};
use ds41rt_transport::v41_expert::{
    V41BackboneRequest, V41SparkTopology, EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16,
};
use ds41rt_transport::{
    ExpertProtocolV2Request, ExpertProtocolV2RouteEntry, ExpertProtocolV2RowDescriptor,
    ExpertV2Dtype, ExpertV2SourceKind,
};

const WARMUP: usize = 200;
const REPS: usize = 2000;
const ROWS: [usize; 6] = [1, 2, 8, 16, 80, 256];
const EPS: [u8; 3] = [1, 2, 3];
const EXPERTS: usize = 384;
const HIDDEN_BYTES: usize = 5120 * 2;

/// Illustrative abstract cost model from the implementation plan
/// `(expert_weight_cost, tile_cost, tile_rows)`. Coefficients are model inputs.
const MODEL_ILLUSTRATIVE: (u64, u64, u32) = (2304, 1536, 64);
/// Runtime-default-shaped model: one unit of weight cost per active expert, no
/// tile term, tile height 16. Also an input, not a measured production
/// coefficient; the balance comparison is reported for both.
const MODEL_RUNTIME_DEFAULT: (u64, u64, u32) = (1, 0, 16);

// ---------------------------------------------------------------------------
// Counting allocator (single-threaded bench; counts every alloc/realloc).
// ---------------------------------------------------------------------------

struct Counting;

static ALLOCS: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn allocs_for<F: FnOnce()>(f: F) -> u64 {
    let before = ALLOCS.load(Ordering::Relaxed);
    f();
    ALLOCS.load(Ordering::Relaxed) - before
}

// ---------------------------------------------------------------------------
// Deterministic seeded inputs.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Seeded canonical top-6 rows: six distinct experts per token, skewed so hot
/// low-id experts repeat across tokens (realistic whole-expert reuse pressure).
fn build_request(rows: usize, seed: u64) -> Result<ExpertProtocolV2Request> {
    let mut rng = Rng(seed);
    let descriptors: Vec<ExpertProtocolV2RowDescriptor> = (0..rows)
        .map(|row| ExpertProtocolV2RowDescriptor {
            row_id: row as u64,
            source_kind: ExpertV2SourceKind::Decode,
            source_request_id: 1000 + row as u64,
            token_position: row as u64,
            route_offset: (row * 6) as u32,
            route_count: 6,
        })
        .collect();
    let mut routes = Vec::with_capacity(rows * 6);
    for row in 0..rows {
        let mut chosen = [0u32; 6];
        for slot in 0..6 {
            loop {
                let u = rng.unit();
                let expert = ((u * u * EXPERTS as f64) as u32).min(EXPERTS as u32 - 1);
                if !chosen[..slot].contains(&expert) {
                    chosen[slot] = expert;
                    break;
                }
            }
        }
        for &expert in &chosen {
            routes.push(ExpertProtocolV2RouteEntry {
                row_index: row as u32,
                expert_id: expert,
                gate_weight: (0.05 + 0.95 * rng.unit()) as f32,
            });
        }
    }
    let hidden = vec![0u8; rows * HIDDEN_BYTES];
    let mut request = ExpertProtocolV2Request::new(
        7,
        17,
        39,
        5120,
        ExpertV2Dtype::Bf16,
        descriptors,
        routes,
        hidden,
    )?;
    request.header.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
    Ok(request)
}

fn route_histogram(request: &ExpertProtocolV2Request) -> [u32; EXPERTS] {
    let mut counts = [0u32; EXPERTS];
    for route in &request.routes {
        counts[route.expert_id as usize] += 1;
    }
    counts
}

fn topology_for(ep: u8) -> V41SparkTopology {
    match ep {
        1 => V41SparkTopology::NATIVE_TP4_EP1,
        2 => V41SparkTopology::NATIVE_TP2_EP2,
        _ => V41SparkTopology::NATIVE_TP2_EP3,
    }
}

fn cost_model(spec: (u64, u64, u32)) -> ReplicatedExpertCostModel {
    ReplicatedExpertCostModel::new(spec.0, spec.1, spec.2)
}

// ---------------------------------------------------------------------------
// Timing helpers.
// ---------------------------------------------------------------------------

struct Stats {
    median_ns: u64,
    p90_ns: u64,
}

impl Stats {
    fn us(&self) -> f64 {
        self.median_ns as f64 / 1000.0
    }
    fn p90_us(&self) -> f64 {
        self.p90_ns as f64 / 1000.0
    }
}

fn median_p90(mut samples: Vec<u64>) -> Stats {
    samples.sort_unstable();
    Stats {
        median_ns: samples[samples.len() / 2],
        p90_ns: samples[(samples.len() * 9) / 10],
    }
}

/// Read-only operation: no state reset needed.
fn bench_readonly<O: FnMut() -> Result<()>>(mut op: O) -> Result<Stats> {
    for _ in 0..WARMUP {
        op()?;
    }
    let mut samples = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let start = Instant::now();
        op()?;
        samples.push(start.elapsed().as_nanos() as u64);
    }
    Ok(median_p90(samples))
}

/// Mutating operation: `prepare` restores preallocated canonical state outside
/// the timed window. The state is passed in once so the two closures cannot
/// create overlapping mutable borrows.
fn bench_mutating<T, P, O>(state: &mut T, mut prepare: P, mut op: O) -> Result<Stats>
where
    P: FnMut(&mut T),
    O: FnMut(&mut T) -> Result<()>,
{
    for _ in 0..WARMUP {
        prepare(state);
        op(state)?;
    }
    let mut samples = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        prepare(state);
        let start = Instant::now();
        op(state)?;
        samples.push(start.elapsed().as_nanos() as u64);
    }
    Ok(median_p90(samples))
}

fn reset_route_words(request: &mut ExpertProtocolV2Request, canonical_ids: &[u32]) {
    request.header.flags &= !ds41rt_transport::v41_expert::V41_NATIVE_GROUP_REQUEST_FLAG;
    for (route, expert) in request.routes.iter_mut().zip(canonical_ids) {
        route.expert_id = *expert;
    }
}

// ---------------------------------------------------------------------------
// Main.
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    println!("# CPU-only host-overhead: actual scheduler + native-group wire path");
    println!("# BENCHMARK COSTS, NOT PRODUCTION COSTS. Host CPU medians only; no GPU, wire or serving claim.");
    println!("# warm={WARMUP} reps={REPS} (medians; allocations per single call)");
    println!();

    println!("## Scheduler planning: preallocated greedy LPT vs expert-modulo baseline");
    println!("# cost model = (expert_weight_cost, tile_cost, tile_rows); coefficients are inputs, not calibration");
    println!("rows,ep,model,plan_us,plan_p90_us,plan_allocs,greedy_max,greedy_imbalance,modulo_us,modulo_p90_us,modulo_allocs,modulo_max,modulo_imbalance");
    for &rows in &ROWS {
        let request = build_request(rows, 0xA5A5_0000 + rows as u64)?;
        let counts = route_histogram(&request);
        for (model_name, model_spec) in [
            ("illustrative", MODEL_ILLUSTRATIVE),
            ("runtime_default", MODEL_RUNTIME_DEFAULT),
        ] {
            let model = cost_model(model_spec);
            for &ep in &EPS {
                let config = ReplicatedExpertScheduleConfig::new(ep, model);
                let mut scheduler = ReplicatedExpertScheduler::new(config, EXPERTS)?;
                let tie_seed = replicated_expert_tie_seed(39, 7);

                // Warm and verify allocation stability of the planning path.
                let plan = scheduler.plan(black_box(&counts), tie_seed)?;
                let greedy_max = plan.group_loads().iter().map(|load| load.cost).max().unwrap_or(0);
                let greedy_total: u64 = plan.group_loads().iter().map(|load| load.cost).sum();
                let plan_allocs = allocs_for(|| {
                    let plan = scheduler
                        .plan(black_box(&counts), tie_seed)
                        .expect("plan");
                    black_box(plan.assignment());
                    black_box(plan.group_loads());
                    black_box(plan.total_cost());
                });
                let plan_stats = bench_readonly(|| {
                    let plan = scheduler.plan(black_box(&counts), tie_seed)?;
                    black_box(plan.assignment());
                    black_box(plan.group_loads());
                    black_box(plan.total_cost());
                    Ok(())
                })?;

                // Preallocated expert-modulo baseline reuses both buffers.
                let mut modulo_loads = vec![0u64; ep as usize];
                let mut modulo_owners = vec![INACTIVE_REPLICATED_EXPERT_GROUP; EXPERTS];
                let compute_modulo = |loads: &mut [u64], owners: &mut [u8]| -> Result<u64> {
                    for load in loads.iter_mut() {
                        *load = 0;
                    }
                    for owner in owners.iter_mut() {
                        *owner = INACTIVE_REPLICATED_EXPERT_GROUP;
                    }
                    for (expert, &count) in counts.iter().enumerate() {
                        if count == 0 {
                            continue;
                        }
                        let group = (expert % ep as usize) as u8;
                        owners[expert] = group;
                        loads[group as usize] += model.expert_cost(count as u64)?;
                    }
                    Ok(loads.iter().copied().max().unwrap_or(0))
                };
                let modulo_max =
                    compute_modulo(black_box(&mut modulo_loads), black_box(&mut modulo_owners))?;
                let modulo_allocs = allocs_for(|| {
                    let max = compute_modulo(
                        black_box(&mut modulo_loads),
                        black_box(&mut modulo_owners),
                    )
                    .expect("modulo");
                    black_box(max);
                    black_box(&modulo_owners[..]);
                    black_box(&modulo_loads[..]);
                });
                let modulo_stats = bench_mutating(
                    &mut (),
                    |_| {},
                    |_| {
                        let max = compute_modulo(
                            black_box(&mut modulo_loads),
                            black_box(&mut modulo_owners),
                        )?;
                        black_box(max);
                        black_box(&modulo_owners[..]);
                        black_box(&modulo_loads[..]);
                        Ok(())
                    },
                )?;

                let greedy_imbalance =
                    greedy_max as f64 / (greedy_total as f64 / ep as f64).max(1.0);
                let modulo_total: u64 = modulo_loads.iter().sum();
                let modulo_imbalance =
                    modulo_max as f64 / (modulo_total as f64 / ep as f64).max(1.0);
                println!(
                    "{rows},{ep},{model_name},{:.3},{:.3},{plan_allocs},{greedy_max},{greedy_imbalance:.4},{:.3},{:.3},{modulo_allocs},{modulo_max},{modulo_imbalance:.4}",
                    plan_stats.us(),
                    plan_stats.p90_us(),
                    modulo_stats.us(),
                    modulo_stats.p90_us(),
                );
            }
        }
    }
    println!();

    println!("## Native-group host path (per request, median CPU), runtime-default cost model");
    println!("# BENCHMARK COSTS, NOT PRODUCTION COSTS. Ownership values do not change encode cost.");
    println!("rows,ep,builder_us,builder_p90_us,builder_allocs,mutate_only_us,mutate_allocs,validate_owned_us,validate_allocs,dispatch_validate_us,dispatch_allocs,worker_parse_us,parse_allocs,legacy_parse_us,clone_us,clone_allocs,encode_us,encode_allocs");
    let model = cost_model(MODEL_RUNTIME_DEFAULT);
    for &rows in &ROWS {
        let base = build_request(rows, 0xA5A5_0000 + rows as u64)?;
        let canonical_ids: Vec<u32> = base.routes.iter().map(|route| route.expert_id).collect();
        let counts = route_histogram(&base);
        for &ep in &EPS {
            let topology = topology_for(ep);
            let config = ReplicatedExpertScheduleConfig::new(ep, model);
            let mut scheduler = ReplicatedExpertScheduler::new(config, EXPERTS)?;
            let tie_seed = replicated_expert_tie_seed(39, 7);
            let owners: Vec<u8> = {
                let plan = scheduler.plan(black_box(&counts), tie_seed)?;
                black_box(plan.group_loads());
                plan.assignment().iter().map(|group| group.encoded()).collect()
            };

            // Mutable working request, reset before each timed builder call.
            let mut work = base.clone();
            let builder_allocs = {
                let mut request = base.clone();
                reset_route_words(&mut request, &canonical_ids);
                allocs_for(|| {
                    let result = request
                        .with_native_group_owners(black_box(&owners), topology);
                    black_box(&result);
                    result.expect("encode owners");
                })
            };
            let builder = bench_mutating(
                &mut work,
                |request| reset_route_words(request, &canonical_ids),
                |request| {
                    let result = request
                        .with_native_group_owners(black_box(&owners), topology);
                    black_box(&result);
                    result?;
                    Ok(())
                },
            )?;

            // Encode-only reference: the same one-hot rewrite without validation.
            let mut mutate_work = base.clone();
            let mutation = bench_mutating(
                &mut mutate_work,
                |request| reset_route_words(request, &canonical_ids),
                |request| {
                    for (route, &expert) in request.routes.iter_mut().zip(black_box(&canonical_ids)) {
                        let owner = black_box(&owners)[expert as usize];
                        route.expert_id = expert | (1u32 << (9 + u32::from(owner)));
                    }
                    black_box(&request.routes[..]);
                    Ok(())
                },
            )?;
            let mutation_allocs = {
                let mut request = base.clone();
                reset_route_words(&mut request, &canonical_ids);
                allocs_for(|| {
                    for (route, &expert) in request.routes.iter_mut().zip(black_box(&canonical_ids)) {
                        let owner = black_box(&owners)[expert as usize];
                        route.expert_id = expert | (1u32 << (9 + u32::from(owner)));
                    }
                    black_box(&request.routes[..]);
                })
            };

            // The three canonical validations across one request lifecycle.
            let canonical = base.clone();
            let validate_owned = bench_readonly(|| {
                let result = V41BackboneRequest::validate_owned(black_box(&canonical), 4096);
                black_box(&result);
                result?;
                Ok(())
            })?;
            let validate_allocs = allocs_for(|| {
                let result = V41BackboneRequest::validate_owned(black_box(&canonical), 4096);
                black_box(&result);
                result.expect("validate");
            });
            let flagged = {
                let mut request = base.clone();
                let result = request.with_native_group_owners(black_box(&owners), topology);
                black_box(&result);
                result?;
                request
            };
            let dispatch_validate = bench_readonly(|| {
                let result = V41BackboneRequest::validate_owned_native_group(
                    black_box(&flagged),
                    4096,
                    black_box(topology),
                );
                black_box(&result);
                result?;
                Ok(())
            })?;
            let dispatch_allocs = allocs_for(|| {
                let result = V41BackboneRequest::validate_owned_native_group(
                    black_box(&flagged),
                    4096,
                    black_box(topology),
                );
                black_box(&result);
                result.expect("dispatch validate");
            });
            let frame = flagged.encode()?;
            let worker_parse = bench_readonly(|| {
                let parsed = V41BackboneRequest::parse_native_group(
                    black_box(&frame),
                    4096,
                    black_box(topology),
                )?;
                black_box(&parsed);
                Ok(())
            })?;
            let parse_allocs = allocs_for(|| {
                let parsed = V41BackboneRequest::parse_native_group(
                    black_box(&frame),
                    4096,
                    black_box(topology),
                )
                .expect("parse");
                black_box(&parsed);
            });
            // Legacy TP4 worker admission for the same canonical bytes.
            let canonical_frame = canonical.encode()?;
            let legacy_parse = bench_readonly(|| {
                let parsed = V41BackboneRequest::parse(black_box(&canonical_frame), 4096)?;
                black_box(&parsed);
                Ok(())
            })?;

            // Transport reference costs.
            let clone_stats = bench_readonly(|| {
                let cloned = ExpertProtocolV2Request::clone(black_box(&canonical));
                black_box(cloned);
                Ok(())
            })?;
            let clone_allocs = allocs_for(|| {
                let cloned = ExpertProtocolV2Request::clone(black_box(&canonical));
                black_box(cloned);
            });
            let encode_stats = bench_readonly(|| {
                let encoded = black_box(&canonical).encode()?;
                black_box(encoded);
                Ok(())
            })?;
            let encode_allocs = allocs_for(|| {
                let encoded = black_box(&canonical).encode().expect("encode");
                black_box(encoded);
            });

            println!(
                "{rows},{ep},{:.3},{:.3},{builder_allocs},{:.3},{mutation_allocs},{:.3},{validate_allocs},{:.3},{dispatch_allocs},{:.3},{parse_allocs},{:.3},{:.3},{clone_allocs},{:.3},{encode_allocs}",
                builder.us(),
                builder.p90_us(),
                mutation.us(),
                validate_owned.us(),
                dispatch_validate.us(),
                worker_parse.us(),
                legacy_parse.us(),
                clone_stats.us(),
                encode_stats.us(),
            );
        }
    }
    Ok(())
}
