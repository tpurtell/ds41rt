# TP/EP expert-kernel evidence run record

Everything here is reproducible from the parent repository plus the pinned
SparkInfer tree. Raw logs in this directory are copied verbatim from the
ostrich run host.

## Provenance

| Field | Value |
| --- | --- |
| Host | `ostrich` (NVIDIA GB10, SM121, ARM64) |
| Container | `ds41rt-tpep-dev` (image `ds41rt-spark-expert-dev`) |
| Staged checkout | `/home/tj/ds41rt-tpep/source` → `/workspace/ds41rt` |
| Verified SparkInfer tree | `/home/tj/ds41rt-tpep/sparkinfer-verified` → `/scratch/sparkinfer-verified` |
| Output directory | `/home/tj/ds41rt-tpep/ep-kernel` → `/scratch/ep-kernel` |
| SparkInfer revision | `4b0954148523b5a2e93813f963d483ffd350b9c9` |
| `source_tree_sha256` | `eca542ddc8e9d2f047d31bde78505ae2ff28b8477a67053f72e9397631234a87` |
| Toolchain | Python 3.12, torch 2.12.0a0 (nv26.05), cutlass 4.6.2, CUDA 13.2 |

The verified tree was confirmed with
`scripts/verify-sparkinfer-source.py --source ... --lock third_party/sparkinfer.lock.json`,
plus an explicit post-import check that `b12x.moe._shared.kernels.v41_slice_pipeline`
and `tests.moe.test_v41_expert_numerics` resolve inside that tree.

## Exact command

```sh
docker exec \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/scratch/sparkinfer-verified:/workspace/ds41rt/python/reference:/workspace/ds41rt/python \
  ds41rt-tpep-dev bash -lc '
    cd /workspace/ds41rt &&
    python -m pytest python/tests/test_v41_sentinel_masking.py -s -q 2>&1 \
      | tee /scratch/ep-kernel/smoke-3topo.log'
```

The same run is available through the repository runner, which performs the
lock verification and the import-provenance assertion before importing
anything:

```sh
DS41RT_SPARKINFER_SOURCE_DIR=/scratch/sparkinfer-verified \
  scripts/run-tp-ep-kernel-checks.sh test -k tp2-i1152
```

## `smoke-3topo.log` — rank-local forward and masked arm, three topologies

One physical rank (rank 0 of each layout), 6 experts, distinct top-6, M1,
width 192. The oracle is built from **that rank's own sliced logical weights**
(`bench.rank_logical_slice`), because a single rank only ever produces its own
N/K shard; the full-width oracle applies only after every rank is summed.

| Topology | per-rank intermediate | kernel extent | slices/expert | forward rel-L2 | forward cosine | masked rel-L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TP2 | 1152 | 1152 | 6 | 0.00167027 | 0.99999857 | 0.00166728 |
| TP4 | 576 | 640 | 3 | 0.00164703 | 0.99999863 | 0.00163012 |
| TP3 | 768 | 768 | 4 | 0.00166284 | 0.99999857 | 0.00160901 |

Gates: `rel_l2 < 0.01`, `cosine > 0.9999`. Result line: `3 passed`.
Every masked arm also asserts that routes the rank does not own are an exact
zero, which is what makes a rank's plane the identity for the cross-rank sum.

## `allocation-tp4.log` — allocation contract, TP4 (576 padded to 640)

TP4 is the discriminating case: `intermediate = 576` differs from the prepared
extent `640`, so it is the only topology that can tell a padded stride from a
logical-width stride. All four slots agree byte-for-byte across four
independent computations.

| Slot | buffer elements | element bytes | buffer bytes | repack bytes | exporter bytes | native packer bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| w13 | 4915200 | 4 | 19660800 | 19660800 | 19660800 | 19660800 |
| s13 | 307200 | 4 | 1228800 | 1228800 | 1228800 | 1228800 |
| w2 | 2457600 | 4 | 9830400 | 9830400 | 9830400 | 9830400 |
| s2 | 153600 | 4 | 614400 | 614400 | 614400 | 614400 |

Sources for each column:

* *buffer* — `numel * element_size()` of the persistent device buffers the
  kernel strides over, printed from the live process.
* *repack* — the bytes one expert's slice actually occupies after
  `_logical_weight_to_w4a8_rp_inplace` / `_e8m0_scale_to_w4a8_sfb_inplace`.
* *exporter* — `experts * kernel_intermediate * n` in uint32 elements for
  `n in (1280, 80, 640, 40)`, from the `specs` list in
  `python/tools/export_b12x_v41_slices_aot.py`.
* *native packer* — `ds41rt_v41_expert_packed_sizes` in
  `native/cuda/kernels/v41_expert_pack.cu`:
  `[padded*hidden, padded*hidden/16, hidden*padded/2, hidden*padded/32]` with
  `padded = align_up(intermediate, 128)`, scaled to the full expert count.

The harness asserts buffer == native packer at construction and raises
otherwise, so an under-allocation fails loudly rather than passing silently.
A runtime pass alone would not have proven absence of an out-of-bounds read;
this assertion is the actual guard.

## Measured tensor memory

`torch.cuda.memory_allocated` / `memory_reserved` cover tensor allocations only,
NOT the whole process VRAM or the CUDA context. Do not read them as a process
footprint.

| Topology | baseline allocated | peak allocated | weight buffers |
| --- | ---: | ---: | ---: |
| TP2 i1152 | 227.9 MB | 332.3 MB | 56.4 MB |
| TP4 i576 | 176.9 MB | 245.9 MB | 31.3 MB |
| TP3 i768 | 223.7 MB | 271.0 MB | 37.6 MB |

## Correctness matrix — `correctness-matrix.log`

`8 passed, 1 skipped`. Logs: `runs/tp-ep-kernel/correctness-matrix.log` (passing
run) and `runs/tp-ep-kernel/correctness-matrix-pre-fix.log` (retained, shows the
two defects fixed below). No timing was collected.

### Physical rank assembly

One shared full-width (2304) operand set and one input; each physical rank of the
layout executes its own TP shard with the EP owner mask applied; per rank the
route sum is taken in FP32 with ONE BF16 rounding, then the rank planes are
summed in order in FP32 and rounded once to BF16. Compared against the shared
full-width oracle.

| Layout | Physical ranks | Assembled rel-L2 | Assembled cosine |
| --- | ---: | ---: | ---: |
| TP4E1 | 4 | 0.003185 | 0.99999493 |
| TP2E2 | 4 | 0.003207 | 0.99999487 |
| TP3E2 | 6 | 0.003157 | 0.99999505 |
| TP2E3 | 6 | 0.003178 | 0.99999499 |

Every individual rank also passed its own shard oracle (~0.00165 rel-L2,
0.9999986 cosine) and had exact zeros on all unowned routes. The EP comparison
is tolerance-gated and records `bit_exact: false`; a different per-rank
intermediate width changes the FP8 intermediate quantization granularity.

### Live row counts on one compiled callable

Rows are a launch scalar, so one compiled callable must serve several live
counts. `id(case.full)` is asserted unchanged at every M, and rows at or beyond
the live count are asserted to retain their prefill.

| M | rel-L2 | cosine |
| ---: | ---: | ---: |
| 1 | 0.0016359 | 0.99999869 |
| 2 | 0.0016420 | 0.99999863 |
| 4 | 0.0016348 | 0.99999869 |
| 8 | 0.0016541 | 0.99999863 |

### Two defects found and fixed in the graph gate

1. **`publish` overwrites the live scalar on every replay.** The pipeline writes
   the launch row count into `live[0]` at the start of each replay, so a host
   `live.fill_()` before replay is discarded. A test that claimed to change rows
   under a frozen callable was actually computing the capture-time row count.
   Frozen-callable coverage is now fixed at one M, and live-count reuse is a
   separate non-graph test.
2. **An empty graph capture looked like a pass.** The cute-compiled callable did
   not record into the torch CUDA graph (torch warned "The CUDA Graph is empty"),
   so replay was a no-op and the test read stale prefill. The failure output made
   this provable: the "result" was `74070 = 12345 * 6`, i.e. the poisoned buffer
   reduced over six routes rather than any kernel output.
   **Root cause:** `Case` cached `current_cuda_stream()` at construction, which is
   the default stream, and the compiled callable takes the stream as a *runtime*
   argument — so a launch inside `torch.cuda.graph` went to a stream the capture
   was not recording. **Fix:** resolve the stream inside the capture context (and
   at every direct launch). The empty-capture guard now FAILS with a message;
   `pytest.skip` remains only in `_require_blackwell` for absent hardware, so an
   empty capture can never be reported as green.

## Correctness matrix, complete

`17 passed` (exit 0). Logs: `runs/tp-ep-kernel/correctness-matrix.log`,
`m-width-sweep.log`, `real-checkpoint.log`, `graph-fix.log`.

### M and width sweep — 35 arms

M 1/2/8/16/80 at widths 64/128/192 (TP2), plus TP4 i576 and TP3 i768, plus the
reuse (all tokens to the same top-k) and skew (one expert dominant) route
distributions. Every arm asserted `id(case.full)` unchanged across all five M on
one compiled callable, and asserted that rows past the live extent were not
written. Representative arm, TP3 i768:

| M | owned routes | rel-L2 | cosine |
| ---: | ---: | ---: | ---: |
| 1 | 3 | 0.0016433 | 0.99999869 |
| 2 | 5 | 0.0016470 | 0.99999875 |
| 8 | 18 | 0.0016513 | 0.99999881 |
| 16 | 36 | 0.0016593 | 0.99999857 |
| 80 | 180 | 0.0016560 | 0.99999863 |

### Frozen-callable mask changes (now genuinely passing)

One capture at fixed M=8; six replays of one graph. `owns_all_again` is
byte-identical to `owns_all`; `rank_owns_none` is an exact zero.

### Real official checkpoint

Layer 0, 6 experts, TP2 i1152, from `model-00003-of-00048.safetensors`:

| Field | Value |
| --- | --- |
| snapshot revision | `dba1be0a40aa45a94ad051997016db3960a90277` |
| `index_sha256` | `74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8` |
| `operands_sha256` | `020d42f71933d001deec739e09a807742b97cf4c48c878ad9dbfed967ab17bd2` |
| rank rel-L2 / cosine | 0.0016411 / 0.99999869 |
| masked rel-L2 / cosine | 0.0016765 / 0.99999857 |

Per-expert shapes: w1/w3 `[6,2304,2560]` weight + `[6,2304,160]` scale, w2
`[6,5120,1152]` + `[6,5120,72]`. Two dtype facts matter: the checkpoint stores
the FP4 payload as **int8** and the E8M0 scales as **float8_e8m0fnu**, while the
repack helper requires **uint8**. The reader reinterprets bits (`view(uint8)`)
rather than resampling, and normalises before hashing because numpy cannot
serialise float8.


## Timing: capacity-1 production-shaped matrix (M1)

Host: GB10, SM clock 2405-2522 MHz observed, 47-51 C. Container
`ds41rt-tpep-dev`, verified tree `/scratch/sparkinfer-verified`. Harness
`python/tools/bench_tp_ep_kernel.py` (hash in
`runs/tp-ep-kernel/source/SHA256SUMS`). 30 intervals per condition; cold =
flush 256 MiB then exactly ONE timed replay. The harness reads the
device L2 size at runtime and refuses a flush that does not exceed it; no
particular L2 size is asserted here and the value was not verified under this
lease.

| Topology | width | slices/expert | warm | GPU-resident | cold | rel-L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TP4 i576 | 64 | 9 | **115.1** | 117.3 | **214.7** | 0.001617 |
| TP4 i576 | 128 | 5 | 158.1 | 158.0 | 237.2 | 0.001663 |
| TP4 i576 | 192 | 3 | 138.0 | 140.8 | 233.2 | 0.001693 |
| TP3 i768 | 64 | 12 | 195.9 | 198.0 | 245.6 | 0.001666 |
| TP3 i768 | 128 | 6 | 201.6 | 204.3 | 258.1 | 0.001617 |
| TP3 i768 | 192 | 4 | 212.4 | 213.9 | 268.0 | 0.001704 |
| TP2 i1152 | 64 | 18 | 268.7 | 270.6 | 316.1 | 0.001622 |
| TP2 i1152 | 128 | 9 | 273.9 | 273.7 | 327.4 | 0.001664 |
| TP2 i1152 | 192 | 6 | 272.7 | 272.0 | 328.4 | 0.001625 |

All values are medians in microseconds, rank-local critical path only.

### The floor is real GPU work, not host submission

| Arm | warm (per-interval) | GPU-resident (N direct launches) | delta |
| --- | ---: | ---: | ---: |
| TP4 i576 w192 | 155.3 | 152.8 | +1.6% |
| TP3 i768 w192 | 206.8 | 205.0 | +0.9% |
| TP2 i1152 w192 | 272.6 | 270.4 | +0.8% |

The GPU-resident figure captures N **direct compiled launches** as N nodes of one
outer graph and replays it once, so there is no host gap between the inner
launches. It agrees with the host-submitted per-interval figure to within 1.6%,
so the ~115 us floor is GPU-resident work, not submission overhead.

An earlier version of this comparison repeated host-submitted *graph replays* and
claimed to amortise the per-launch gap. That was wrong: `T = N*(gpu+gap)/N` is
constant, so it could not distinguish the two. It is retracted; the captured
direct-launch construction above replaces it.

### Capacity matters for the width choice

| Topology | capacity 1 warm | capacity 80 warm | delta |
| --- | ---: | ---: | ---: |
| TP4 | 138.0 | 164.9 | +19.5% |
| TP3 | 212.4 | 211.2 | -0.6% |
| TP2 | 272.7 | 274.7 | +0.7% |

At capacity 1 the best width is **64** for every topology (TP4 115.1 vs 138.0 at
w192), whereas the capacity-80 grid preferred w64 only at M1-M2 and w192 at
M8+. The capacity-80 M1 arm carries 480 route slots for one live row, which
inflates the route-plan and reduce portions and changes the width tradeoff, so
capacity-80 and capacity-1 M1 numbers are not interchangeable.

## Diagnostic LPT cost model

`cost(expert) = expert_weight_cost + ceil(rows / tile_rows) * tile_cost` for an
active expert, and exactly 0 for an inactive one. This matches
`ds41rt-core`'s `ReplicatedExpertCostModel`
(`rust/crates/ds41rt-core/src/replicated_expert_schedule.rs`). The weight term is
charged **once per active expert**, never per routed row.

It is a diagnostic projection, not the Rust scheduler: tie-breaking is a stable
descending cost then ascending expert id, the Rust implementation may break ties
differently, and the harness forms its histogram with one device-to-host
bincount that the real scheduler does not need. Seven CPU unit tests in
`python/tests/test_tp_ep_cost_model.py` pin the arithmetic, including the
inactive-expert-is-free and weight-charged-once cases.

Ownership is recomputed per live row count from that row count's own route
histogram. Deciding ownership once at capacity and slicing it would schedule a
histogram the arm never runs.


## E384 production-geometry M1 (INCOMPLETE — one of four deployments)

Capacity 1, M1, E384 arena, layer-0-shaped synthetic operands, 30 intervals,
cold = 256 MiB flush then one timed replay. Raw: `runs/tp-ep-kernel/e384-tp4-ep1.json`.

| Deployment | width | warm | GPU-resident | cold | per-group max (warm only) | rel-L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TP4EP1 (owns 6) | 64 | 112.4 | 110.2 | 223.7 | 112.3 | 0.001634 |
| TP4EP1 (owns 6) | 128 | 192.3 | 190.4 | 250.8 | 192.4 | 0.001644 |
| TP4EP1 (owns 6) | 192 | 178.5 | 177.2 | 244.5 | 178.8 | 0.001714 |

**TP2EP2, TP3EP2 and TP2EP3 at E384 are NOT measured.** They aborted on the
harness's own post-timing assertion (`post-timing drift 1.33-1.45`, i.e.
`tail_rel >= 0.01`) and the run was stopped for the E2E lease before diagnosis.
The per-group values in this table are warm-only: they were produced by a
harness revision that measured group maxima under the warm condition only, so
the "max" here is not comparable to the E8 capacity-1 table, whose maxima cover
warm, GPU-resident and cold. Do not compare the two columns.


## CPU-identified defect in the E384 run (not a GPU-validated fix)

The E384 arms for TP2EP2, TP3EP2 and TP2EP3 aborted on
`post-timing drift 1.33-1.45`. Cause, identified on CPU from the failure
signature and the source:

1. `--max-group` measures every EP group in turn and leaves `case.ids` /
   `case.routing` holding the **last** group's route table.
2. The post-timing check then replayed the arm's graph and compared the result
   against the **first** arm's oracle.

So the check compared group N's output with group 0's expectation. It only
passed for `ep_degree == 1`, where there is a single group and the two coincide
— which is exactly the observed pattern (TP4EP1 passed, every multi-group
deployment failed). This is a harness state bug, not a kernel defect.

Fix applied on CPU only: the post-timing check runs **before** the optional
mutating blocks, against the arm's own oracle, and the arm's live ids/routing are
restored afterwards. A restore snapshot is recorded in each record.

**This is not GPU-validated.** The E384 multi-group matrix has not been re-run
and must not be quoted until it is.

### Shadowed definitions (same class of bug)

The frozen E384-run source `runs/tp-ep-kernel/source/bench_tp_ep_kernel.E384-FAILED.py`
(sha256 `42307152eec2768a3eafd1307e9ed8f3312db2d529b13acf2c1b65958e0ecc4b`) defined
**seven** top-level names twice: `_padded`, `modulo_owner`, `expert_route_counts`,
`expert_cost`, `lpt_owner`, `_stage_breakdown`, `_per_group_latency`. The later
copies shadowed the earlier ones, so the corrected LPT cost model and the
corrected per-group measurement were **not** the code that ran. The duplicate
block is removed, and `test_no_duplicate_top_level_definitions` now fails on any
harness file that defines a top-level name twice, so this cannot recur silently.
Any LPT number produced before this fix is invalid and is not reported.

## Runtime dispatch buckets (authoritative)

`rust/crates/ds41rt-daemon/.../Execution::execution_state` selects capacity by
live rows: **rows 1 -> capacity 1; rows 2..80 -> capacity 80; otherwise the
worker maximum, 4096** (prefill 2048 uses the 4096 worker capacity). There is
**no capacity-16 runtime path**. Capacity 16 was exercised by a qualification
library but is not a current serving choice, so benchmark arms at capacity 16
would describe a potential new policy, not production. Only capacity 1 and
capacity 80 appear in this document, and they are never mixed.

## What is measured versus what is not

Measured, with correctness gates and raw samples retained:

* capacity-1 E8 M1 across TP4/TP3/TP2 and widths 64/128/192: warm, GPU-resident
  and cold, with per-group maxima;
* GPU-resident (captured direct-launch) versus per-interval agreement, within
  1.6%, which is what establishes that the floor is GPU work and not host
  submission;
* E384 TP4EP1 M1 warm/GPU-resident/cold at three widths;
* 17 correctness tests and 7 CPU cost-model tests.

Not measured, and therefore not claimed:

* any E384 value for TP2EP2, TP3EP2 or TP2EP3;
* the masked 3-active and 2-active arms at E384;
* any transport, inter-rank reduction or group-sum cost;
* any end-to-end serving effect, and any kernel selection.

Two cold conditions appear across the wider evidence set: 256 MiB here and 32
MiB in the native pilot. They are not merged or compared.

## Scope limits that must travel with these numbers

* Single physical rank only. No parallel TP rank sum, no inter-rank reducer, no
  group-sum cost and no cross-Spark transport latency is measured.
* Group maxima are measured **sequentially** and are projections of the critical
  group, not a concurrent distributed path.
* The E8 arena used here is smaller than the production 384-expert geometry, so
  these are diagnostic geometry numbers, not production-representative.
* No end-to-end serving claim and no production default was changed.

## Not yet run

* Phase-0 timing beyond M1 capacity-1 and the single-M1 amortised check.
* The native AOT leg: disabled and fail-closed. The unreachable body and the
  unused ctypes native layer are removed; `--phase native` fails at the parser.
