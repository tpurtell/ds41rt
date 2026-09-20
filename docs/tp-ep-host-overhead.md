# Host CPU overhead of the native replicated-group wire path

> **BENCHMARK COSTS, NOT PRODUCTION COSTS.** Everything below is host CPU
> microbenchmark data from one release binary on one workstation. It is not
> serving throughput, not wire/driver time, not GPU or kernel time, and not a
> total cost of the replicated-group feature. Coefficients used by the
> scheduler comparison are model *inputs*, not calibrated production numbers.

Status: bounded CPU-only audit + microbenchmark. No GPU, no network, no remote
hosts (`rhea`/`moa` not used), no `/mnt/scratch`, no service, no WIP build, no
commit, no optimization implementation. The frozen wire/API in
[tp-ep-transport.md](tp-ep-transport.md) was not changed.

Owner: transport agent. This task produced only this document and
`rust/crates/ds41rt-transport/examples/v41_tp_ep_host_overhead.rs`.

## 1. Question

For the frozen native replicated-group path: what is the host CPU cost of
request encoding, ownership validation and transport preparation versus legacy
TP4; do the three canonical validations per request dominate the `M=1` decode
case; and is any safe optimization indicated?

## 2. Environment, method and source identity

| Item | Value |
| --- | --- |
| Host | `raptor`, AMD Ryzen Threadripper 9970X 32-Cores, 64 logical CPUs, 1 socket, 1 NUMA node |
| Clock state | scaling governor `powersave`; ~3.38 GHz current, max 5487.6 MHz, "CPU scaling MHz 40%" during the run |
| Kernel | Linux 7.0.0-31-generic x86_64 |
| Toolchain | rustc 1.98.1 (48a229cea 2026-09-01), cargo 1.98.1 |
| Build | `--release`, `-j 2`, isolated target `~/.cache/ds41rt/builds/tp-ep-host-overhead` (root NVMe, not `/mnt/scratch`) |
| Run | warm 200, timed 2000, medians; single-threaded benchmark process; counting `#[global_allocator]` |
| Inputs | seeded SplitMix64 canonical top-6, six distinct experts/row, squared-uniform skew; `M ∈ {1,2,8,16,80,256}`, `EP ∈ {1,2,3}`; no debug checksum |

**Timing caveat:** the `powersave` governor held the measured cores near 3.38 GHz
of a 5.49 GHz maximum. Absolute microseconds are therefore a low-clock
measurement and can move with DVFS; use them for relative comparison, not as a
clock-normalized figure.

**black_box discipline:** every measured closure passes its inputs through
`black_box` and observes its outputs, including the scheduler
`assignment()` and `group_loads()` slices, `total_cost()`, the modulo owner/load
buffers, the validation `Result`s, the parsed request, the cloned request and the
encoded frame. This prevents the optimizer from constant-folding the work or
discarding the stores. The validation and parse numbers were essentially
unchanged after adding this discipline (clone/encode drifted within DVFS noise),
so the measured validation work was real and not eliminated.

### Source identity

Repository `HEAD = 92e29c7c7e5e867abe0730ee28c81e7193ca5fce` (branch `dev`) with
85 dirty files from several agents, so the commit alone does not identify the
measured sources. SHA-256 of the measured files at run time:

```text
1afaf55d310b88272f28739b3cec58f6b705bf13942b249a7b81fefee1fe5061  rust/crates/ds41rt-transport/examples/v41_tp_ep_host_overhead.rs
6bf27a07310f7ad10ddb7d9183c28cf6733108b81987940b59f9d6f5e3c8d122  rust/crates/ds41rt-transport/src/v41_expert/native_group.rs
698a53b2f1a4cdd72832a91de2cb6f274fde48ea0d4a666b4110a1ec0943de38  rust/crates/ds41rt-transport/src/v41_expert.rs
a2d7ca5d5140921a5654ea65aa1414e0beae0921faf66d8f526c814523bdc382  rust/crates/ds41rt-transport/src/v41_expert/chunks.rs
d5c00d3f85b79a6451c79635be36615f5b5535e8b66d518bba3d880e05ffad1d  rust/crates/ds41rt-transport/src/v41_expert/roce.rs
463652239cf35b97863b051846a5aae4f68b62bb1d3faf1e755b25928537ef35  rust/crates/ds41rt-transport/src/v41_expert/tcp.rs
f622aab5fbe64f7fcc800b3779db4ce67dc380186237f5673b8b12269248cdcb  rust/crates/ds41rt-transport/src/protocol_v2.rs
9d26b010d1f82ce5ed9e892912740294a1aadf6dc0a905fbb1870b2925e3133b  rust/crates/ds41rt-core/src/replicated_expert_schedule.rs
afe95f2561945721b4b7b5cbd62df4281911e61b8145d7ad2b0435069317890a  rust/Cargo.lock
```

Reproduce:

```bash
cd /home/tj/Developer/ds41rt
CARGO_TARGET_DIR="$HOME/.cache/ds41rt/builds/tp-ep-host-overhead" \
  cargo run --release --manifest-path rust/Cargo.toml \
  --example v41_tp_ep_host_overhead -p ds41rt-transport -j 2
```

## 3. Cost models (assumptions, not calibration)

The scheduler table is reported for two `ReplicatedExpertCostModel`
`(expert_weight_cost, tile_cost, tile_rows)` inputs:

- **`runtime_default` = (1, 0, 16)** — pure weight cost, one unit per active
  expert, no tile term. This is the runtime-default-shaped case for the balance
  comparison.
- **`illustrative` = (2304, 1536, 64)** — the heavier abstract model used in the
  implementation plan.

Neither is a measured production coefficient, and the grouped costs are model
units, not microseconds. Both models rank the same way because the per-expert
term dominates, which is why their imbalance ratios coincide; the absolute
`*_max` columns differ by construction.

## 4. Read-only source-cost audit

Three canonical-shape validations run per request, all over the same canonical
route table:

| # | Where | Call | Source |
| --- | --- | --- | --- |
| 1 | Coordinator, request builder | `validate_owned` → `request.validate()` + `validate_canonical` | `native_group.rs:315` |
| 2 | Coordinator, transport dispatch | `validate_owned_native_group` → `request.validate()` + `validate_native_group` | `chunks.rs:217` |
| 3 | Worker admission | `parse_native_group` → view `validate` + `validate_native_group` | `v41_expert.rs:198/204` |

Plus the builder's trailing generic `self.validate()` (`native_group.rs:338`),
the one-hot rewrite, the 384-byte ownership batch in #2/#3, and — on the
coordinator — per-rank `clone()` (RoCE) or full-frame `encode()` (TCP).

## 5. Scheduler planning: greedy LPT vs expert-modulo

```text
rows,ep,model,plan_us,plan_p90_us,plan_allocs,greedy_max,greedy_imbalance,modulo_us,modulo_p90_us,modulo_allocs,modulo_max,modulo_imbalance
1,1,illustrative,0.190,0.200,0,23040,1.0000,0.120,0.121,0,23040,1.0000
1,2,illustrative,0.220,0.221,0,11520,1.0000,0.120,0.130,0,15360,1.3333
1,3,illustrative,0.211,0.221,0,7680,1.0000,0.120,0.130,0,7680,1.0000
1,1,runtime_default,0.190,0.200,0,6,1.0000,0.120,0.121,0,6,1.0000
1,2,runtime_default,0.220,0.221,0,3,1.0000,0.120,0.130,0,4,1.3333
1,3,runtime_default,0.211,0.221,0,2,1.0000,0.120,0.141,0,2,1.0000
2,1,illustrative,0.350,0.541,0,46080,1.0000,0.130,0.131,0,46080,1.0000
2,2,illustrative,0.320,0.321,0,23040,1.0000,0.140,0.141,0,26880,1.1667
2,3,illustrative,0.301,0.311,0,15360,1.0000,0.131,0.140,0,15360,1.0000
2,1,runtime_default,0.271,0.280,0,12,1.0000,0.131,0.141,0,12,1.0000
2,2,runtime_default,0.320,0.321,0,6,1.0000,0.131,0.141,0,7,1.1667
2,3,runtime_default,0.301,0.311,0,4,1.0000,0.140,0.141,0,4,1.0000
8,1,illustrative,0.721,0.731,0,157440,1.0000,0.180,0.181,0,157440,1.0000
8,2,illustrative,0.891,0.901,0,80640,1.0244,0.180,0.181,0,96000,1.2195
8,3,illustrative,0.821,1.553,0,53760,1.0244,0.180,0.190,0,65280,1.2439
8,1,runtime_default,0.731,0.741,0,41,1.0000,0.180,0.181,0,41,1.0000
8,2,runtime_default,0.891,0.901,0,21,1.0244,0.180,0.190,0,25,1.2195
8,3,runtime_default,0.821,0.831,0,14,1.0244,0.180,0.190,0,17,1.2439
16,1,illustrative,1.412,1.432,0,314880,1.0000,0.271,0.281,0,314880,1.0000
16,2,illustrative,1.742,1.753,0,157440,1.0000,0.280,0.281,0,161280,1.0244
16,3,illustrative,1.593,2.934,0,107520,1.0244,0.271,0.281,0,115200,1.0976
16,1,runtime_default,1.412,1.422,0,82,1.0000,0.271,0.281,0,82,1.0000
16,2,runtime_default,1.733,1.743,0,41,1.0000,0.271,0.281,0,42,1.0244
16,3,runtime_default,1.583,2.895,0,28,1.0244,0.271,0.290,0,30,1.0976
80,1,illustrative,4.567,4.587,0,921600,1.0000,0.691,0.692,0,921600,1.0000
80,2,illustrative,5.549,5.569,0,460800,1.0000,0.691,0.711,0,499200,1.0833
80,3,illustrative,5.098,5.118,0,307200,1.0000,0.681,0.701,0,341760,1.1125
80,1,runtime_default,4.567,4.577,0,240,1.0000,0.671,0.681,0,240,1.0000
80,2,runtime_default,5.538,5.568,0,120,1.0000,0.681,0.711,0,130,1.0833
80,3,runtime_default,5.097,5.118,0,80,1.0000,0.691,0.701,0,89,1.1125
256,1,illustrative,7.021,7.041,0,1405440,1.0000,1.042,1.052,0,1405440,1.0000
256,2,illustrative,8.503,8.533,0,702720,1.0000,1.052,1.081,0,702720,1.0000
256,3,illustrative,7.822,7.841,0,468480,1.0000,1.041,1.042,0,476160,1.0164
256,1,runtime_default,7.021,7.041,0,366,1.0000,1.042,1.052,0,366,1.0000
256,2,runtime_default,8.503,8.533,0,183,1.0000,1.052,1.081,0,183,1.0000
256,3,runtime_default,7.842,7.862,0,122,1.0000,1.051,1.061,0,124,1.0164
```

- Both planners are allocation-free at 384 experts (`plan_allocs=0`,
  `modulo_allocs=0`).
- Greedy LPT costs 1.6-8x the modulo planner (0.19 vs 0.12 us at M1; ~8.5 vs
  ~1.05 us at M256) and never exceeds ~8.6 us.
- Greedy imbalance is 1.000-1.024 for both models; expert-modulo reaches 1.333
  (M1 EP2), 1.244 (M8 EP3), 1.113 (M80 EP3), and is balanced only where expert
  ids happen to distribute evenly.
- **The imbalance is predicted model load, not measured GPU time.** It is a
  property of the assignment plus the cost-model coefficients above.

## 6. Native-group host path (per request)

Runtime-default cost model; ownership values do not change encode cost.

```text
rows,ep,builder_us,builder_p90_us,builder_allocs,mutate_only_us,mutate_allocs,validate_owned_us,validate_allocs,dispatch_validate_us,dispatch_allocs,worker_parse_us,parse_allocs,legacy_parse_us,clone_us,clone_allocs,encode_us,encode_allocs
1,1,0.050,0.051,0,0.020,0,0.030,0,0.040,0,0.150,0,0.130,0.060,2,0.130,2
1,2,0.050,0.050,0,0.020,0,0.030,0,0.041,0,0.150,0,0.130,0.060,2,0.131,2
1,3,0.050,0.050,0,0.020,0,0.030,0,0.041,0,0.150,0,0.130,0.060,2,0.130,2
2,1,0.061,0.070,0,0.030,0,0.041,0,0.070,0,0.231,0,0.210,0.060,2,0.200,2
2,2,0.060,0.070,0,0.030,0,0.040,0,0.070,0,0.231,0,0.210,0.060,2,0.200,2
2,3,0.070,0.070,0,0.030,0,0.040,0,0.070,0,0.231,0,0.210,0.060,2,0.200,2
8,1,0.170,0.171,0,0.040,0,0.110,0,0.190,0,0.751,0,0.651,0.070,2,0.982,2
8,2,0.170,0.171,0,0.040,0,0.110,0,0.190,0,0.751,0,0.651,0.070,2,0.991,2
8,3,0.170,0.171,0,0.040,0,0.110,0,0.190,0,0.751,0,0.651,0.070,2,0.992,2
16,1,0.310,0.311,0,0.070,0,0.191,0,0.360,0,1.423,0,1.262,0.100,2,1.783,2
16,2,0.310,0.311,0,0.070,0,0.190,0,0.361,0,1.422,0,1.252,0.100,2,1.783,2
16,3,0.310,0.311,0,0.070,0,0.191,0,0.360,0,1.422,0,1.252,0.100,2,1.783,2
80,1,1.422,1.423,0,0.280,0,0.882,0,1.733,0,6.791,0,6.059,0.321,2,10.366,2
80,2,1.412,1.423,0,0.280,0,0.881,0,1.732,0,6.790,0,6.060,0.331,2,10.386,2
80,3,1.422,1.432,0,0.280,0,0.881,0,1.733,0,6.791,0,6.059,0.331,2,10.366,2
256,1,4.407,4.437,0,0.851,0,2.764,0,5.448,0,21.713,0,19.310,0.902,2,49.085,2
256,2,4.447,4.487,0,0.851,0,2.764,0,5.459,0,21.723,0,19.299,0.902,2,47.803,2
256,3,4.436,4.457,0,0.851,0,2.764,0,5.438,0,21.713,0,19.289,0.902,2,48.794,2
```

Column meanings: `builder_us` = full `with_native_group_owners`;
`mutate_only_us` = the one-hot rewrite with no validation; `validate_owned_us` =
validation #1; `dispatch_validate_us` = validation #2; `worker_parse_us` =
validation #3 including view parse; `legacy_parse_us` = legacy
`V41BackboneRequest::parse` on the same bytes; `clone_us`/`encode_us` = transport
reference operations.

- **Allocation-free:** builder, rewrite, all three validations, worker parse,
  greedy plan and modulo planner = 0 allocations. Only `clone()` (2) and TCP
  `encode()` (2) allocate.
- Validation #1: 0.030 us at M1, 0.110 at M8, 0.882 at M80, 2.764 at M256.
- Validation #2: 0.040 us at M1, 5.44 at M256 (ownership batch roughly doubles
  #1 at M256).
- Validation #3: 0.150 us at M1, 21.7 at M256; legacy parse is 19.3 at M256, so
  the ownership layer adds ~2.4 us and the rest is the common view+canonical
  two-pass.
- The pure rewrite is 0.020 us at M1 and 0.85 us at M256: encoding itself is
  cheap; validation dominates the builder at large M.

## 7. Replicated vs legacy: coordinator-validation delta only

| M | legacy coordinator (`validate_owned`) | replicated coordinator (builder + dispatch) | delta | replicated worker parse | legacy worker parse | worker delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.030 us | 0.090 us | **+0.060 us** | 0.150 us | 0.130 us | +0.020 us |
| 8 | 0.110 | 0.360 | +0.250 | 0.751 | 0.651 | +0.100 |
| 80 | 0.882 | 3.155 | +2.273 | 6.791 | 6.059 | +0.732 |
| 256 | 2.764 | 9.855 | +7.091 | 21.713 | 19.310 | +2.403 |

**Label correction:** the `+0.060 us` M1 figure is the **coordinator
canonical-validation delta only**. It deliberately **excludes** the scheduler
plan (0.19-0.22 us at M1, up to ~8.6 us at M256), histogram construction and
owner-assignment materialization, the per-rank `request.clone()` (0.060 us plus
2 allocations each) or TCP `encode()` (0.130 us plus 2 allocations), the
client-side session/channel/QP work, and all wire/driver time. It is **not** the
total cost of enabling replicated groups; it is one measured component of it.

## 8. Do the three canonical validations dominate M1?

**No.** At M1 the three canonical bodies cost about
`0.030 + 0.040 + ~0.040 ≈ 0.11 us`, against a scheduler plan of 0.19-0.22 us, a
worker parse of 0.150 us, a TCP frame `encode()` of 0.130 us (plus 2
allocations), and a per-rank `clone()` of 0.060 us (plus 2 allocations). The
largest single M1 host item is scheduler planning, then worker parse, then TCP
encode. The coordinator's own two validations total 0.070 us, below both the
encode and the plan. At M256 the coordinator validation total is 9.86 us against
a 21.7 us worker parse and a 43-49 us TCP encode, so validation is still not the
dominant host term.

## 9. Safe optimizations identified (none authorized, none implemented)

Recorded only. The parent has explicitly not authorized any optimization, and
none was implemented.

1. Process-local validated-admission token from `with_native_group_owners` to
   skip the duplicate dispatch validation — saves 0.040 us at M1 / 5.44 us at
   M256, but must be provenance-scoped, not a wire trust bit.
2. Reduce the builder's trailing `self.validate()` to a flags-only check —
   ~1.0 us at M256.
3. EP=1 fast path skipping the 384-byte ownership batch — small fixed fraction.
4. Fused worker view+canonical parse to remove one route-table pass — bounded by
   roughly half of `legacy_parse_us` (~10 us at M256).

Validation #1 and #3 must not be removed: #1 prevents a rejected assignment from
leaving a half-mutated request, and #3 is the worker's only ownership check.

## 10. Honesty notes

- Single-binary host CPU medians under a `powersave` governor at ~3.38 GHz;
  `p90` values are in the tables. No end-to-end, GPU, wire, driver or serving
  number is implied.
- Greedy-vs-modulo reports predicted model load, not measured group time.
- Six-rank layouts remain hardware-unqualified; only their CPU path was
  exercised.
- The example is the only code artifact of this task; the frozen transport
  source was not modified.
