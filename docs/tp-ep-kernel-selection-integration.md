# TP/EP kernel-selection integration readiness

CPU/source trace only. No build, no GPU, no remote, no benchmark edits; no winner
selected and no production/default change. Purpose: state whether the measured
`bench_tp_ep_kernel.py` widths are the exact production kernel or only a related
benchmark, and the smallest opt-in route if a measured width wins later.

Measured provenance: the recorded timing artifact
(`runs/tp-ep-kernel/e384-tp4-ep1.json` provenance) is a GB10 **SM121**, 48 SMs,
24 MiB L2 — the Spark AOT target arch, so there is **no RTX/arch uncertainty**.
Historical benchmark provenance: the saved baseline in
`runs/tp-ep-kernel/source/bench_tp_ep_kernel.py` is `c741d34a…`; the accepted older
96-arm run was associated with `452d6eb6…` in its execution ledger, but that exact
source copy was not retained and its arm JSONs lack an embedded harness hash.
Neither identifies the current run. The new width-192/M64 experiment uses a
separately frozen harness, recorded in its per-arm provenance and run manifest;
use those artifacts, not this historical source trace, for current identities.

## 1. Verdict: same kernel class, not yet an AOT-identical measurement

The benchmark compiles the **same b12x source class and schedule** as the
production slice export: `Case.__init__` builds
`V41SlicePipeline(capacity, width, experts=..., topk=..., intermediate=...)`
(`python/tools/benchmark_v41_ep_groups.py:600`), and the exporter builds the same
`V41SlicePipeline` (`python/tools/export_b12x_v41_slices_aot.py:149-151`). Both use
`V41FusedSliceKernel(width, …, intermediate)` plus `V41SliceReduce(width, …)`
(`third_party/sparkinfer/b12x/moe/_shared/kernels/w4a8_v41_slice.py:49-61`,
`v41_slice_pipeline.py:17-24`) and the same `kernel_intermediate = align_up(I,128)`
(benchmark `benchmark_v41_ep_groups.py:555`; exporter via b12x
`_dynamic_kernel_intermediate_size`).

It is **not yet an exact production-AOT measurement** for two reasons:

1. **Atomic path (capacity-dependent).** The benchmark case has no `atomic_tokens`
   and no compact specialization. Production Spark exports pass
   `--atomic-min-capacity 256`, so capacities ≥256 use direct token accumulation
   (`export_b12x_v41_experts_aot.py:258-261`, `v41_spark_tp_experts.cmake:63`).
   A capacity-80 measurement is the closest exact match; 256/1024/4096 are
   non-atomic in the benchmark.
2. **Dispatch leg.** The benchmark docstring states the native AOT leg is not
   implemented; it times the Python/JIT pipeline, not the compiled
   `v41_{role}_m{capacity}` object and its interface selected at runtime.

Shared property (not a fidelity difference between JIT and AOT):
`V41FusedSliceKernel` uses a single `width` for both FC1 (gate/up) and FC2 (down),
and `V41SliceReduce(width, …)` uses the same value, so a per-projection width is
not selectable with the current knob.

## 2. Explicit mapping (benchmark → exporter → symbol → runtime)

| Benchmark input | Production knob | Object / symbol | Runtime selection |
| --- | --- | --- | --- |
| `--topologies tp2/tp3/tp4` → per-rank `intermediate` 1152/768/576 (`bench_tp_ep_kernel.py:900`) | role `spark_tp2`/`spark_tp3`/`spark`; geometry in `export_b12x_v41_slices_aot.py:29-32` | `native/cmake/v41_spark_tp_experts.cmake` (TP2/TP3) and `v41_experts.cmake` (spark); wrapper `native/src/v41_spark_tp{2,3}_experts.cc` | FFI interfaces 8/9 (roles 5/6) and interface 0 (role 1) |
| `--widths 64/128/192` (`bench_tp_ep_kernel.py:805`) | `--width` (`export_b12x_v41_slices_aot.py:339-367`); CMake cache `DS41RT_V41_EXPERT_SLICE_WIDTH` (`v41_experts.cmake:14-26`); per-role Spark caches `DS41RT_V41_SPARK_TP2_SLICE_WIDTH` / `DS41RT_V41_SPARK_TP3_SLICE_WIDTH` (`v41_spark_tp_experts.cmake:41-44`) | `v41_slices_m{cap}_w{width}` experimental, `v41_{role}_m{cap}` with `--standard-names` | baked at build; **not** runtime-selectable |
| `--capacity` (default 80) | `--rows 1,16,80,256,1024,4096` | one object per capacity | worker `--capacity` (80/256/1024/4096); `execution_state` picks decode cap 1, small ≤80, else main (`rust/crates/ds41rt-daemon/src/v41_experts/execution.rs:333-351`) |
| `--topk 6`, 384 experts (fixed) | role geometry `(384, I, kernel_I, 6)` | — | unchanged |
| ownership mask via sentinel ids/zero weights (`bench_tp_ep_kernel.py:216-222`) | native route-word owner bits + sentinel 384 | — | unchanged by width |

Existing defaults: with `DS41RT_V41_EXPERT_SLICE_WIDTH` empty, the exporter uses
`{capacity: 64 if capacity == 1 else 192}` (`export_b12x_v41_experts_aot.py:258`);
the Spark TP2/TP3 CMake map is the same shape. **No 128 exists in production
today.** Spark TP2/TP3 AOT objects are built only when
`DS41RT_V41_SPARK_TP_ROLES` lists `tp2`/`tp3` (`native/CMakeLists.txt:803-804`,
`build.sh:76-91`), and the launcher refuses an image that does not advertise the
role (`run.sh:285-292`).

Padding: `spark` has `intermediate=576`, `kernel_intermediate=640`
(`ROLE_GEOMETRY`); TP2 1152 and TP3 768 need no padding. `slices = ceil(I/width)`
with `slices*width <= kernel_intermediate` enforced
(`w4a8_v41_slice.py:57-61`), so width 128 on TP4 gives 5 slices (5·128 = 640) and
is legal.

## 3. Smallest opt-in implementation route (if a measured winner differs)

Checked against the actual CMake variables, not assumed:

- `DS41RT_V41_EXPERT_SLICE_WIDTH` is a **cache STRING** accepting a scalar or a
  `capacity:width` map, and drives the one arch-selected `v41_experts` role
  (`v41_experts.cmake:14-26`; SM121 → `spark`, SM120 → `coordinator`). A TP4
  `spark` width change is therefore a `-D` build-flag change plus re-export of that
  role. A generic TP4 `--width` override switches to the slices exporter, whose
  `atomic_min_capacity` defaults to `None` (unlike the Spark TP2/TP3 command, which
  hardcodes 256), so a TP4 override must **also** pass
  `-DDS41RT_V41_EXPERT_ATOMIC_MIN_CAPACITY=256`; otherwise the unchanged large
  capacities silently switch accumulation path.
- `DS41RT_V41_SPARK_TP2_SLICE_WIDTH` and `DS41RT_V41_SPARK_TP3_SLICE_WIDTH` are
  **per-role `CACHE STRING` knobs** in `v41_spark_tp_experts.cmake:41-44`, each
  defaulting to the current full map
  `1:64,16:192,80:192,256:192,1024:192,4096:192`. The loop forwards the
  role-selected value to the existing exporter `--width`
  (`v41_spark_tp_experts.cmake:49,54,75`), so each role is independently
  overridable with `-D` and an unconfigured build exports the same objects as
  before. Each value is a scalar `64/128/192` or a **full** `capacity:width` map;
  the exporter validates coverage and domain
  (`export_b12x_v41_slices_aot.py:356-367`), so a partial or invalid map fails at
  export time instead of being silently accepted, and the domain parser is not
  duplicated in CMake. The generic TP4 knob `DS41RT_V41_EXPERT_SLICE_WIDTH` is
  untouched.
- Re-export invalidation: the exported stems stay `v41_{role}_m{capacity}`, so a
  content-stable per-role stamp (`v41_{role}_width.stamp`, generated from the
  role/width/atomic identity) is a dependency of the export command. It is
  rewritten only when that identity changes, so Make re-exports on a width override
  without churning AOT objects on an unchanged configure, while Ninja also sees the
  changed command line. `--atomic-min-capacity 256` stays in the same exporter
  command as `--width`.
- Width is baked into the per-capacity object; the runtime selector (`--capacity`,
  `execution_state`), the wire format and the ABI are unchanged, so no FFI or daemon
  change is needed.
- A per-projection (gate/up vs down) width, or two widths for the same
  (role, capacity) in one library, is **out of scope here**: the symbol family is per
  role/capacity, so it would need a new symbol/AOT-info field plus FFI/daemon
  selection. Do not put that on the critical path.

The critical path is therefore **re-export of the changed role/capacity plus its
oracle/comparator**, not a kernel or AOT rewrite.

## 4. Concrete missing coverage before safe adoption (scope = changed role/capacity)

Example: an opt-in **capacity-80** width change for one Spark role needs
capacity-80 evidence, plus smoke coverage that untouched capacities still load:

1. Re-export that role and check the width-baked identity: `v41_experts.json`
   variant `implementation`/`width` for the changed capacity, and interfaces 0/8/9
   report the expected `kernel_intermediate` (`640` spark, `1152` spark_tp2,
   `768` spark_tp3) and `input_dtype` 7.
2. Capacity-80 AOT only: production-mask oracle (empty group, single group, mixed
   groups), changed-input graph replay, finite/nonzero, no-replay-allocation, and
   `info.scratch_bytes` for capacity 80 (width changes `planes`).
3. Unchanged-capacity smoke only: capacity 1 (width 64 decode path) and the prefill
   capacity in use (4096 at prefill 2048) must still load and replay; no full
   width×capacity sweep.
4. **Default-192 comparator at M8/M16 before any gain claim** (the benchmark row
   list includes 8 and 16; `bench_tp_ep_kernel.py:806`), same lease/device/clocks.
   **Measured-workload mapping (accepted profile):** the segmented counting run
   observed C1 decode `rows=8` and C16 decode `rows=64`, consistent with the 2-RTX
   two-8-request lanes at draft width 7
   (`runs/tp-ep-preflight/g2-live/profile/PROFILE-INTERPRETATION.md:20,73-75`), and
   both are served by the **capacity-80** AOT variant because `execution_state`
   selects the small kernel for `rows <= 80`
   (`rust/crates/ds41rt-daemon/src/v41_experts/execution.rs:341-351`; only `rows==1`
   uses capacity 1). The current micro-sweep rows M1/M8/M16 cover the real C1 shape
   but **not** the observed C16 `M64`. Because the capacity-80 object has one baked
   width and cannot select by live M8 vs M64, a capacity-80 width winner must
   include **one focused M64 comparison against the default width 192** (same
   device/lease/clocks) plus the M64 oracle and changed-input replay; otherwise the
   recommendation must be explicitly limited to M8/M16 with the C16 risk stated.
   This is part of the changed-capacity gate, not a new matrix. The latest profile
   does not establish a specific optimization: a 192 comparator on the benchmark
   followed by AOT evidence is required, and no large padding benefit may be claimed
   from the micro-sweep alone.
5. If the changed capacity is ≥256, add atomic-path coverage
   (`--atomic-min-capacity 256`; the benchmark is non-atomic today).
6. No RTX/coordinator kernel work is needed unless the RTX role itself changes
   (none requested). Capacity 16 is compiled but is not selected by the standard
   launch path (prefill floor is 80), so it is excluded.
7. Keep the default `{1:64, else:192}` until gates 1–6 pass on the measured target.
