# TP×EP native support: Spark TP2/TP3 roles, packer and N-plane reducer

Status: **implementation landed; GPU/numerical correctness PENDING.** No native
build, export compile, GPU run or WIP/release action was performed. Every kernel
statement below is a source/contract statement; none is a measured result. This
document owns the native/FFI/export slice of the TP×EP work described in
[tp-ep-implementation-plan.md](tp-ep-implementation-plan.md); it does not
restate or override the read-only kernel audit
([tp-ep-kernel-audit.md](tp-ep-kernel-audit.md)).

## 1. Scope and unchanged default behavior

Delivered, behind an opt-in build switch that leaves every release default
**behaviorally unchanged** (no new expert family or kernel is compiled, linked
or bound unless the option is set). This is a behavior claim, not a binary
identity claim: `export_b12x_v41_slices_aot.py` always writes one additive
manifest field, `spark_tp_degree` (its historical value for every existing
role), so default-exported `v41_experts.json` bytes differ from the previous
revision even though no compiled object, symbol or default launch path changes:

- two new plan-time Spark expert roles, **Spark TP2** and **Spark TP3**, with
  distinct AOT symbol families, native role ids and FFI entry points;
- a packer accept-list extension for the TP3 local intermediate **768** with
  explicit 32-scale alignment;
- a generic **2/3/4/6-plane** coordinator-side BF16 partial reducer, with the
  historical 2- and 4-plane entry points preserved and test-equivalent;
- exporter role guards/geometry tables, manifest role metadata and CMake wiring
  behind a new opt-in option;
- CPU-only tests that check the role geometry, guards, symbol families and
  reducer ABI without a GPU, plus a GPU selftest that is skipped when absent.

Unchanged and explicitly preserved: Spark TP4 (`role 1`, logical 576 → kernel
640 storage padding, SM121), RTX TP2 (`role 3`, 1152, **SM120 guard
untouched**), full RTX backbone (`role 2`, 2304, SM120), coordinator dSpark
(`role 0`, 128×2304×top-3), dSpark TP2 (`role 4`, 128×1152×top-3). No default
build option, config or launcher changes, and the default (empty
`DS41RT_V41_SPARK_TP_ROLES`) compiles/links no new object or symbol. The only
default-path artifact delta is the additive `spark_tp_degree` manifest field
described above; binary identity of default-exported JSON is **not** claimed.

## 2. Roles and AOT symbol families

`ds41rt_v41_expert_info_t.role` gains two values in the native FP8 K32 family
(`native/include/ds41rt_v41_experts.h`):

| role | Family | experts | hidden | logical I | kernel I | top-k | input | SM |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 5 | Spark TP2 shard | 384 | 5120 | 1152 | 1152 | 6 | FP8 E4M3 + UE8M0 K32 (`input_dtype 7`) | 121 |
| 6 | Spark TP3 shard | 384 | 5120 | 768 | 768 | 6 | FP8 E4M3 + UE8M0 K32 (`input_dtype 7`) | 121 |

TP2/TP3 are 128-aligned and take **no** storage padding
(`kernel_intermediate == logical_intermediate`). Each degree has its own symbol
family, defined by a wrapper that renames the shared launch ABI and includes one
generated variant table:

- `native/src/v41_spark_tp2_experts.cc` → `ds41rt_v41_spark_tp2_expert_{info,initialize,launch,bind_scratch,initialize_scratch_async,output_kind}`
- `native/src/v41_spark_tp3_experts.cc` → `ds41rt_v41_spark_tp3_expert_{info,initialize,launch,bind_scratch,initialize_scratch_async,output_kind}`

All families coexist in one `libds41rt_native.so`. The new wrappers suppress the
canonical FP8 row quantizer (it stays defined exactly once by the primary native
expert translation unit), so no duplicate `ds41rt_v41_expert_input_quant_*`
symbols are introduced.

## 3. Packer extents

`ds41rt_v41_expert_packed_sizes` / `ds41rt_v41_pack_expert_async`
(`native/cuda/kernels/v41_expert_pack.cu`) now accept intermediate
**{576, 768, 1152, 2304}** and reject any value that is not a multiple of 32.
The packed byte model is unchanged and applied to 768 with
`padded = align_up(intermediate, 128) = 768`:

| Tensor | Bytes |
| --- | --- |
| W13 | `padded * hidden` |
| S13 | `padded * hidden / 16` |
| W2 | `hidden * padded / 2` |
| S2 | `hidden * padded / 32` |

The N256/K128 lane-major representation transform is unchanged; out-of-range
rows still write zeros. TP3's `w2` contraction axis is `768 = 6*128`, so its
packed source windows are byte-aligned. The FFI packer validates the accepted
set before calling native (`v41_pack_intermediate_supported`, re-exported from
`ds41rt-ffi`), and the historical TP4/TP2/full extents are untouched.

## 4. Reduction contract (all physical ranks → coordinator)

Chosen contract: **every physical rank returns one BF16 `[M,5120]` partial plane
directly to the coordinator; there is no group-local reduction.** The
coordinator sums the planes in rank order in FP32, adds the BF16 shared expert
exactly once, and rounds once to BF16. `ranks` is the number of contributing
physical ranks (`TP × EP`): 2, 3, 4 or 6.

New C ABI (`native/include/ds41rt_v41_experts.h`):

```c
int32_t ds41rt_v41_reduce_compact_bf16_planes_async(
    const uint16_t* const planes[6], const uint16_t* shared, uint16_t* output,
    uint32_t rows, uint32_t ranks, void* stream);
```

- `ranks ∈ {2,3,4,6}`; `planes[0..ranks)` must be non-null, `planes[ranks..6)`
  must be null.
- Ordered `__fadd_rn` accumulation (`reduce_compact<Ranks>`), one final
  `__float2bfloat16_rn`; shared is added after the rank sum and before the
  rounding.
- `1 <= rows <= 4096`; every extent is computed in `uint64_t` and every pointer
  upper bound is checked against `UINTPTR_MAX - bytes` (no size overflow).
- The host pointer array is consumed synchronously to fill **six by-value kernel
  pointer arguments**. No device pointer array, no per-call device allocation,
  and no dangling stack-array asynchronous use.
- Rejected prelaunch: bad `ranks`, `rows == 0`, `rows > 4096`, null active slot,
  non-null inactive slot, misaligned plane/output/shared, plane/output overlap,
  and partial shared/output overlap. Exact `output == shared` is permitted.

The historical `ds41rt_v41_reduce_tp2_compact_bf16_async` (2) and
`ds41rt_v41_reduce_compact_bf16_async` (4) remain, with their original
validation, and call the same ordered template; a GPU selftest asserts the
generic and fixed entry points are bit-identical for ranks 2 and 4.

FFI (`rust/crates/ds41rt-ffi/src/v41_experts.rs`):

- `NativeLibrary::v41_compact_reducer()` gains
  `reduce_planes(planes: [*const u16; 6], ranks: u32, shared, output, rows, stream) -> Result<()>`
  (loaded optionally so older libraries still expose the 2/4 paths and fail with
  a clear message on the new one).

## 5. FFI surface for the daemon agent

Interface selectors are extended (existing 0/2–7 unchanged):

| selector | `NativeLibrary` methods | native prefix | role |
| ---: | --- | --- | ---: |
| 8 | `v41_spark_tp2_expert_info` / `v41_spark_tp2_expert_kernel` | `ds41rt_v41_spark_tp2` | 5 |
| 9 | `v41_spark_tp3_expert_info` / `v41_spark_tp3_expert_kernel` | `ds41rt_v41_spark_tp3` | 6 |

`expert_info_for` validates the exact `(experts, logical_intermediate,
kernel_intermediate, topk)` tuple and requires `input_dtype == 7` for selectors
8/9, so a BF16-hidden artifact is rejected with a specific message rather than
silently used. A missing artifact fails with
`native library lacks expert interface N; enable its AOT build`. Selection is a
plan-time, static-topology decision (which role and which precompiled capacity
bucket); **live row counts are never part of a compile key.**

## 6. Admission bytes

Admission receives exact persistent and transient numbers from the exported
geometry via `V41ExpertInfo`:

- persistent per-expert weights: the four packed extents from
  `ds41rt_v41_expert_packed_sizes(logical_intermediate)` — for TP3,
  `768*5120`, `768*5120/16`, `5120*768/2`, `5120*768/32` bytes per expert;
- transient workspace: `info.scratch_bytes` (the exported per-capacity core
  scratch), plus the caller-owned input row extent
  `rows * info.input_row_bytes()` (5280 bytes/row for `input_dtype 7`);
- output plane extent: `rows * 5120 * 2` bytes per physical rank, summed by the
  reducer.

The older `reduce`/`reduce_tp2` methods are unchanged; the new `reduce_planes`
uses `6 * rows * 5120 * 2` as its worst-case plane bound.

## 7. Exporter and CMake wiring

`python/tools/export_b12x_v41_slices_aot.py` now owns explicit pure role tables
(`ROLE_GEOMETRY`, `ROLE_SM`, `ROLE_NATIVE_ID`, `SPARK_TP_DEGREE`):

| role | geometry (experts, logical I, kernel I, topk) | SM | native id |
| --- | --- | ---: | ---: |
| `spark_tp2` | (384, 1152, 1152, 6) | 121 | 5 |
| `spark_tp3` | (384, 768, 768, 6) | 121 | 6 |
| `spark` (unchanged) | (384, 576, 640, 6) | 121 | 1 |
| `rtx_tp2` (unchanged) | (384, 1152, 1152, 6) | 120 | 3 |
| `rtx_backbone` (unchanged) | (384, 2304, 2304, 6) | 120 | 2 |
| `dspark_tp2` (unchanged) | (128, 1152, 1152, 3) | 120 | 4 |
| `coordinator` (unchanged) | (128, 2304, 2304, 3) | 120 | 0 |

SM guards are now enforced from the table (Spark family requires SM121; RTX
family SM120), and the manifest records `spark_tp_degree` alongside `role` and
`geometry`. `python/tools/export_b12x_v41_experts_aot.py` accepts the new roles,
routes them only through the native FP8 K32 slice export, and rejects
`spark_tp2`/`spark_tp3` with a BF16 input format.

CMake (`native/cmake/v41_spark_tp_experts.cmake`, new):

```bash
# SM121 Spark expert build only; default is empty and builds only TP4.
cmake -S native -B <build> -DDS41RT_ENABLE_V41_EXPERT_AOT=ON \
  -DDS41RT_CUDA_ARCHITECTURES=121 \
  -DDS41RT_V41_SPARK_TP_ROLES="tp2;tp3" ...
```

The option pre-compiles capacities **1, 16, 80, 256, 1024, 4096** ahead of
runtime for each selected degree, with widths **64 (capacity 1) / 192 (all
others)**, `--atomic-min-capacity 256` (ABI 3 FP32 token accumulation for
256/1024/4096; ABI 2 FP32 route planes for 1/16/80) and `--standard-names`, so
the labels are `v41_spark_tp2_m{cap}` / `v41_spark_tp3_m{cap}`. No runtime
compilation path is added. The build option to report upstream is
`DS41RT_V41_SPARK_TP_ROLES`; leaving it unset is the release default.

## 8. Replication and invalid-route sentinel

Workers mask unassigned routes with id `384` and weight `0`; canonical requests
stay top-6. Findings from inspection (no third-party change made, none needed):

- **Grouped pipeline (recommended):** `V41RoutePlan.pack` never assigns an id
  `>= experts` (or `< 0`) to a group, the route gets `inverse = -1`,
  `V41SliceReduce` writes `0` for `inverse < 0`, and a group with
  `metadata[group,1] == 0` early-exits after two integer comparisons. This is a
  genuine skip of unowned work — the contract the EP design needs.
- **Compact/hybrid pipeline:** every direct expert index is guarded and
  out-of-range experts produce zero route output, so it is **correct**, but the
  route CTA still launches and writes zeros (phase2 zeroes rather than skipping).
  It is therefore *not* used for EP measured arms; EP dispatch must use the
  grouped pipeline. This matches the kernel audit, and is reported here rather
  than worked around, because the compact pipeline lives in `third_party/`.
- `topk` remains a compiled constant (6) in the AOT `info` struct and every
  route-shaped scratch buffer; a token may legitimately assign 0..6 routes to a
  group, so no per-group top-3 restriction is assumed.

## 9. Tests

CPU-only (run now, no GPU/build lease):

- `python/tests/test_v41_spark_tp_roles.py` — AST-checks the pure role tables
  (TP2 1152 / TP3 768 unpadded; TP4 640 and RTX TP2 SM120 unchanged; opt-in
  CMake wiring; header/reducer/packer ABI text).
- `python/tests/test_v41_spark_tp_expert_symbols.py` — compiles the real
  `v41_spark_tp2_experts.cc` / `v41_spark_tp3_experts.cc` against a host CUDA
  shim and mock variant tables; asserts exact role ids/geometry, FP8 input
  `input_dtype 7`, launch contract and distinct families.
- `cargo test -p ds41rt-ffi --lib` — `packed_intermediate_accept_list_covers_spark_tp_degrees`
  and `expected_geometry_covers_spark_tp2_tp3_without_touching_rtx_tp2`.

GPU (must run in the matching-architecture dev container, serialized):

- `native/tests/v41_route_reduce_planes_selftest.cc` (CTest target
  `ds41rt_v41_route_reduce_planes_selftest`, skipped without a CUDA device) —
  independent scalar reference for ranks 2/3/4/6 across rows 1/3/16/80/4096,
  shared/no-shared and exact shared/output alias, bit-equivalence of the generic
  and fixed 2/4 entry points, exact write extent (the capacity-sized destination
  is poisoned and every byte past `rows*5120` must stay poison), graph
  capture/replay at ranks 3 and 6 with changed plane and shared contents across
  two replays, single-zero and all-zero active planes with and without shared,
  and malformed cases where each rejection has exactly one violation in an
  otherwise-valid six-slot argument set (bad ranks/rows, null active slot,
  non-null inactive slot, plane/output aliasing, partial shared/output overlap,
  byte-misaligned output and byte-misaligned plane).
- The existing FFI `#[ignore]` GPU oracles remain the model for the real-weight
  ladder; the reducer and packer selftests below were executed on the L5 lease.

### 9.1 GPU qualification evidence (L5, DODO GB10 / SM121, ABI 2 and 3)

Executed on the DODO GB10 lease (`GPU-c08bcbd8-eb8e-1353-5e59-ede0a4dc974c`,
torch 2.12.0a0+5aff3928d8, CUDA 13.2) against the frozen SM121 DODO library
`libds41rt_native.so` sha256
`d453812c5c53e916fe8ea0b54d72a51c19b0df8e4db8ce9b7c1ea5b967b1f513`. Raw logs:
`runs/tp-ep-native/dodo-l5/`.

**Standalone reducer, ranks 2/3/4/6 (SM120 RTX1, L1 lease).** Exact host scalar
reference, rows 1/3/16/80/4096, shared/no-shared/exact-alias, generic-vs-fixed
bit-equivalence, poisoned destination write extent, graph capture/replay at
ranks 3 and 6 with changed planes/shared, and single-/all-zero planes: exit 0,
`v41 route reduce planes selftest: ok`, 2.37 s. Source sha256 route_reduce
`0c12ff6f...`, selftest `67b1e118...`.

**TP3 packer (intermediate 768), SM120 RTX1.** Native packer bytes vs an
independent host port of the N256/K128 transform, full byte comparison, plus
the accepted-extent list: exit 0, extents 3,932,160 / 245,760 / 1,966,080 /
122,880 bytes. Source sha256 pack `7468874b...`, selftest `03d81853...`.

**Synthetic replicated-group ABI qualification (roles 5/6, DODO).**
`python/tools/qualify_v41_replicated_native.py` reuses the frozen grouped-slice
fixture and its independent scalar oracle; 384-expert arenas with the fixture's
32 residents placed at expert offsets 352..383. 12/12 cases pass (TP2 and TP3,
capacities 1/16/80, ABI 2), max rel_l2 0.001679, min cosine 0.9999986, sentinel
(id 384, weight 0) rows exactly zero, graph replay allocates nothing.

**Native Spark TP3 launch-geometry identity.** Every exported `spark_tp3`
variant records its compiled `launch_geometry` in `v41_experts.json` (built by
`python/tools/v41_spark_tp3_launch_geometry.py`, schema version 2): the pinned
`m16n8k32` atom, the fixed 16-row route group, the per-CTA slice width (fc1
logical N per projection, fused `w13` N = 2x, fc2 K contribution), the K128 /
40-stage FC1 schedule, the `3 + 16` route-metadata columns, the compiled
`output_kind`, per-stage control authority (`controls.fc1/fc2/grid`; M and K are
pinned constants, only N is manifest-planned), and the pinned source revision.
Any other hidden extent, or a storage extent that is not the exact 128-element
roundup of the logical intermediate, fails closed. `qualify_v41_replicated_native.py`
resolves and verifies that identity per arm (`verify_manifest_geometry`),
cross-checks the loaded native info and the manifest scratch/capability identity,
and records -- with GPU identity, revision, named tolerances, geometric
route-weight mode (uniform is diagnostic only), input/route/wire hashes,
activation/routing/wire/operand immutability flags and pre/post `nvidia-smi` state
selected by device UUID against the exact `0x0` throttle mask (an unparseable or
`[N/A]` reason field fails closed with the device row) -- in every `TIMING`
record. Records are
buffered and printed only after the whole run completes, so a partial line can
never look final. `--aggregate` then requires one measurement context, at least
two widths backed by *distinct* libraries, three repeats per width with identical
per-width identity and one identical cross-width identity (seed, routing,
tolerances, revision, model geometry, device); it reports per-width medians only,
with no fastest width and no default promotion. CPU contract and pinned-source
tripwires live in `python/tests/test_v41_spark_tp3_launch_geometry.py`, gated by
`scripts/run-tp-ep-kernel-checks.sh test`.

**Real official checkpoint, all TP ranks, ABI 2 and 3 (DODO).** Official
revision `dba1be0a40aa45a94ad051997016db3960a90277`, layers **11/20/39**
(remote boundary candidates), **original checkpoint expert ids 0..5 and 383**
(7 residents) mapped into arena slots **377..383** of full 384-expert arenas
(`base = 384 - 7 = 377`; the routed ids are the arena ids, the oracle uses the
loaded-index ids). Weights built by the **native packer**
`ds41rt_v41_pack_expert_async`, compared per case against the independent
full-oracle logical slice. **135/135 cases pass** across **TP2 ranks 0/1 and
TP3 ranks 0/1/2**, capacities **1/80 (ABI 2) and 256/4096 (ABI 3 token
output)**, live rows **1/80/256**. Each case also runs changed-activation and
mask-restore graph replays and both runtime BF16 compactors:

| Metric | Value |
| --- | --- |
| cases | 135 (90 ABI 3, 45 ABI 2) |
| rank coverage | TP2 ranks 0/1; TP3 ranks 0/1/2 |
| max rel_l2 | 0.0017589 |
| min cosine | 0.99999845 |
| max BF16 compaction rel_l2 | 0.0027429 (min cosine 0.99999619) |
| max changed-activation rel_l2 | 0.0017137 (min cosine 0.99999851) |
| max mask-restored rel_l2 | 0.0017137 (min cosine 0.99999851) |
| native packer vs b12x repack differing bytes | **0** (all cases) |
| sentinel / all-inactive / changed-activation-masked rows | exact zero (all cases) |
| graph replay allocation delta | 0 (all cases) |

Changed-activation gate: the captured graph's `wire` buffer is rewritten with a
new activation, the canonical mask is restored, and the replay must match the
**new** oracle; then the rows are masked (exact zero) and the mask is restored,
and the replay must again match the new oracle.

BF16 compaction scope: ABI 3 exercises
`ds41rt_v41_compact_tokens_bf16_async` on the native FP32 token output; ABI 2
exercises `ds41rt_v41_compact_routes_bf16_async` on the native FP32 route
planes. Both are compared to the Python BF16 oracle with tolerance (not bit
identity). This covers the native compaction kernels inside the per-rank AOT
scope; it does not execute the Rust runtime's exact call sequence, nor the
cross-rank N-plane reducer here (that reducer is covered by the standalone
`v41_route_reduce_planes_selftest` and by the daemon's assembly).

These prove the emitted SM121 roles 5/6 AOT artifacts execute the official
checkpoint slice on every TP rank correctly through the production C ABI,
including the ABI 3 token-output capacities, sentinel masking, changed
activation across replay, mask restore, and native BF16 compaction, and that
the native packer is byte-identical to the Python repack on this evidence set.

Not covered by this evidence: cross-rank assembly / exactly-once (the daemon
assembles the physical-rank planes), the Rust runtime's exact call sequence, a
measured serving path, and any performance claim.

The earlier TP4 padding discrepancy is resolved and is **not** an open STOP:
the ceil-tiled repack pads 1152 -> 1280 (640 per half), the native packer and
b12x repack agree byte-for-byte, and fdcd's four-way TP4 byte assertions
(19,660,800 / 1,228,800 / 9,830,400 / 614,400 for 6 experts) plus the TP4
numerics match; TP4 is unchanged and remains out of scope for *native role
expansion* only.

### 9.2 M1 device timing crosscheck (DODO, scoped)

Same frozen library, layer 20, tp_rank 0, six resident experts (0..5) at arena
slots 378..383, masks of 6/3/2 active routes with weights `1/active`.

Method: one CUDA graph is captured with `with torch.cuda.graph(g): for _ in
range(200): native.run(1)` — direct native launches, a single graph launch per
timed sample. Device microseconds are the stream-event window divided by 200; a
separate 1-launch graph is timed identically. Host microseconds bracket only
`graph.replay()` (the enqueue, not the wait). Preflight poisons the output,
replays once and asserts the written region is finite; then 10 warmups and 30
warm samples (median). Cold is a 32 MiB device flush followed by one timed
single-launch replay, 5 samples (median).

| role | I / K | cap | act | warm device µs | warm-1 µs | cold µs |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 (TP4) | 576 / 640 | 1 | 6 | 156.4 | 159.7 | 226.0 |
| 1 | 576 / 640 | 1 | 3 | 61.6 | 64.8 | 169.7 |
| 1 | 576 / 640 | 1 | 2 | 61.4 | 64.4 | 143.1 |
| 1 | 576 / 640 | 80 | 6 | 180.7 | 183.8 | 246.5 |
| 1 | 576 / 640 | 80 | 3 | 98.9 | 102.1 | 202.5 |
| 1 | 576 / 640 | 80 | 2 | 98.7 | 102.0 | 176.1 |
| 5 (TP2) | 1152 / 1152 | 1 | 6 | 279.7 | 285.9 | 334.6 |
| 5 | 1152 / 1152 | 1 | 3 | 155.1 | 158.6 | 229.1 |
| 5 | 1152 / 1152 | 1 | 2 | 62.2 | 65.0 | 196.3 |
| 5 | 1152 / 1152 | 80 | 6 | 281.2 | 283.5 | 341.2 |
| 5 | 1152 / 1152 | 80 | 3 | 173.4 | 176.8 | 245.5 |
| 5 | 1152 / 1152 | 80 | 2 | 102.3 | 105.3 | 209.0 |
| 6 (TP3) | 768 / 768 | 1 | 6 | 202.5 | 205.7 | 258.6 |
| 6 | 768 / 768 | 1 | 3 | 62.3 | 66.4 | 192.5 |
| 6 | 768 / 768 | 1 | 2 | 61.9 | 65.3 | 159.5 |
| 6 | 768 / 768 | 80 | 6 | 215.3 | 218.7 | 279.3 |
| 6 | 768 / 768 | 80 | 3 | 99.1 | 102.3 | 206.5 |
| 6 | 768 / 768 | 80 | 2 | 98.6 | 102.2 | 196.3 |

Host launch latency: one graph replay enqueues in 2.26-2.41 µs; the 200-node
amortized graph enqueues in 3.07-3.17 µs total (**0.0155 µs/launch**). Warm
amortized and warm single agree within 1-3%, so the device numbers are not
host-gap artifacts. cap1 (width 64) is materially faster for 2/3 active routes
and equal at 6 active. **Cold is a 5-sample pilot**, not a statistical claim.

Native cross-arm parity within this pilot: TP4-EP1 active-6 warm 156.4 µs vs
TP2-EP2 active-3 155.1 µs (cold 226.0 vs 229.1); TP3-EP2 active-3 62.3 vs
TP2-EP3 active-2 62.2 (six-Spark projection only). On the same arm the Python
~92 µs cap-80 figure is near the native cap-80 TP3 active-3 value (99.1 µs),
not below it; earlier cross-ownership comparisons are withdrawn.

Launch geometry (read-only source evidence, `w4a8_v41_slice.py:79-83`):
`grid=(slices, groups, 1)`, `block=(128,1,1)`, no cluster and no cooperative
launch. `slices = ceil(intermediate/width)`: TP2 width 64 / I=1152 -> 18, TP3
-> 12, TP4 -> 9. At M1 the pipeline passes `groups = max(1, min(rows*topk,
experts + max(rows*topk-experts,0)//16)) = 6`, so the grid is always
`slices x 6` and inactive groups exit at `active = rows > 0` before any shared
memory is allocated. Active CTAs are therefore `slices x active_routes`.

**Two competing mechanisms; the warm-only data does not separate them.** Warm
device time is flat at ~62 µs while the active weight working set fits in L2 and
steps as it exceeds it. Per-expert native packed bytes: TP4 5.2224 MB (active-3
15.67, active-6 31.33), TP3 6.26688 MB (active-3 18.80, active-6 37.60), TP2
9.40032 MB (active-2 18.80, active-3 28.20, active-6 56.40). Every arm at or
below 18.8 MB is ~62 µs and every arm above ~28 MB steps (TP2 a3 155.1, TP4 a6
156.4, TP3 a6 202.5, TP2 a6 280); the cold pilot removes the plateau (all cold
143-341 µs). The threshold is consistent with a **24 MiB L2** and with an
occupancy/wave effect, and these are confounded in warm-only data. Both figures
are **theoretical placeholders until an actual device property is recorded**:
the 24 MiB L2 is not in the repo evidence (which records 128 MB only for the
SM120 RTX), and the 48 SMs come from the DODO AOT manifest `physical_sms`, not
from a live device-property record in this pass. The earlier claims that this is
"not linear weight bandwidth" and "one wave -> two" are **withdrawn**.

Resource census (CPU-only `cuobjdump --dump-resource-usage` on the embedded
cap1 cubins extracted from the frozen library; no GPU):

| kernel | registers | static SHARED | stack/local |
| --- | ---: | ---: | ---: |
| `V41FusedSliceKernel` (compute) | 122 | 1024 B | 0 / 0 |
| `V41SliceReduce` | 26 / 32 / 39 (TP4/TP3/TP2) | 1024 B | 0 / 0 |
| `V41RoutePlan.scatter` | 32 | 1024 B | 0 / 0 |
| `V41RoutePlan.prefix` | 27 | 1024 B | 0 / 0 |
| `V41RoutePlan.pack` | 18 | 1024 B | 0 / 0 |
| `publish` | 8 | 1024 B | 0 / 0 |

Block is 128 threads, no cluster. Dynamic shared memory is not in the ELF;
source-derived for width 64 it is ~14.5 KiB (`b` 8192 + `sf` 1024 + `mid` 4096 +
`qa` 1024 + `qs` 128). A register-derived upper bound of ~4 CTAs/SM (65536 / (128
regs x 128 threads)) leaves room for roughly 4 x SM concurrent CTAs, so the
census **does not support a simple one-wave/two-wave explanation**; it also does
not establish an SMEM bound, which needs the exact device property (an SMEM/SM
figure cannot be assumed). Runtime occupancy was not measured. Disambiguating
needs a future profile (L2 hit rate plus an interleaved multi-expert-set stream
that exceeds L2). The cross-layer stream's working set is large (~GB scale), so
a full-warm-reuse microbenchmark is not representative of that stream; repeated
expert ids may nevertheless leave part of L2 resident, so "always cold" is not
guaranteed. No implementation and no kernel change is proposed.

Reproduction source: `python/tools/bench_v41_native_ep.py`
(entry point, sha256 `06aa25de...`) importing
`python/tools/qualify_v41_replicated_native.py` (measurement logic, sha256
`0a956c8b...`) and `python/tools/_v41_expert_native.py` (ABI wrapper, sha256
`4bacd01b...`). L5/DODO lease released after this run.


## 10. Explicit PENDING flags (no claims)

| Item | Status |
| --- | --- |
| New kernels compile on SM121/SM120 | PASSED — SM120 L1 (reducer, packer) and SM121 L5 (roles 5/6 AOT) |
| TP2/TP3 numerics vs independent oracle | PASSED (synthetic 12/12; real checkpoint 108/108, layers 11/20/39) |
| Generic 2/3/4/6 reducer numerics vs scalar oracle | PASSED (RTX1 SM120 selftest) |
| Packer 768 byte round trip | PASSED (synthetic host oracle; native==b12x on real checkpoint) |
| ABI 3 token output at capacities 256/1024/4096 | PASSED (live rows 1/16/80/256, graph replay no alloc) |
| Sentinel id 384 / all-inactive exact zero | PASSED (synthetic and real-checkpoint cases) |
| Real weights for all TP ranks (roles 5/6) | PASSED — TP2 ranks 0/1; TP3 ranks 0/1/2, layers 11/20/39 |
| Changed-activation graph replay + mask restore | PASSED (135/135, max rel 0.0017137, min cos 0.99999851) |
| Native BF16 compaction (ABI 3 token, ABI 2 routes) | PASSED vs Python BF16 oracle (max rel 0.0027429); Rust runtime exact call sequence not exercised |
| Masked6 grouped execution cost/skip proof | PENDING (audit-level evidence only) |
| Cross-rank assembly, exactly-once | PENDING (daemon agent) |
| Build option `DS41RT_V41_SPARK_TP_ROLES="tp2;tp3"` produces artifacts | PASSED (frozen DODO library exposes both roles, all capacities) |
| End-to-end serving / performance | PENDING |

No performance result is claimed. The numbers above are correctness gates on a
single lease, not a serving measurement.


## 11. Files changed

- `native/include/ds41rt_v41_experts.h` — role doc, new symbol declarations, N-plane reducer ABI.
- `native/cuda/kernels/v41_route_reduce.cu` — ordered 2/3/4/6-plane reducer.
- `native/cuda/kernels/v41_expert_pack.cu` — 768 accept-list + 32 alignment.
- `native/src/v41_experts.cc` — include branches and quantizer suppression.
- `native/src/v41_spark_tp2_experts.cc`, `native/src/v41_spark_tp3_experts.cc` — new families.
- `native/cmake/v41_spark_tp_experts.cmake`, `native/CMakeLists.txt` — opt-in wiring, CUDA selftest targets.
- `native/tests/v41_route_reduce_planes_selftest.cc` — GPU scalar-reference reducer selftest.
- `native/tests/v41_expert_pack_tp3_selftest.cc` — GPU TP3 (768) packer byte-oracle selftest.
- `python/tools/export_b12x_v41_slices_aot.py`, `python/tools/export_b12x_v41_experts_aot.py` — role tables, guards, metadata.
- `python/tools/_v41_expert_native.py` — `spark_tp=None|2|3` family selection + native packer-size binding.
- `python/tools/qualify_v41_replicated_native.py` — synthetic and real-checkpoint native qualification (ABI 2 and 3), the `--timing` M1 harness, the verified per-variant `launch_geometry` identity record, and `--aggregate` repeated-width comparison.
- `python/tools/v41_spark_tp3_launch_geometry.py` — the native Spark TP3 launch-geometry identity contract shared by the exporter and the qualifier.
- `python/tools/bench_v41_native_ep.py` — tracked timing entry point (single reproducible command).
- `python/tests/test_v41_spark_tp_roles.py`, `python/tests/test_v41_spark_tp_expert_symbols.py`, `python/tests/test_v41_expert_native_roles.py` — CPU-only tests.
- `rust/crates/ds41rt-ffi/src/v41_experts.rs`, `rust/crates/ds41rt-ffi/src/lib.rs` — FFI methods, validation, reducer, rank-count capability, tests.
- `runs/tp-ep-native/dodo-l5/` — raw GPU logs and provenance (ignored, disposable).
- `docs/tp-ep-native.md` — this document.

No file outside the assigned ownership was modified. No commit or push was made.
