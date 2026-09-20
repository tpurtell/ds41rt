# Official V4.1 expert kernels: TP/EP replication audit

Read-only discovery for configurable fully replicated Spark expert groups
(existing TP4 vs TP2xEP2, TP3xEP2, TP2xEP3). Official
`deepseek-ai/DeepSeek-V4.1-Flash` format only; no NVFP4 and no EXL3 work is
used or recommended here. No default serving configuration or release is
proposed.

Scope of this document: kernel geometry and contracts, the numerical
reference and benchmark infrastructure that already exists, a microbenchmark
matrix, and the isolated build/run mechanism. Everything below is either
**measured/verified by reading source or records** or explicitly marked
**hypothesis** and **unknown**.

Parent design contract assumed throughout: canonical top-6 routes; a native
group owner carried in a new request flag/route word; `group = global_rank/TP`
and `local_tp_rank = global_rank % TP`; the worker builds sentinel `384` ids
and zero weights for routes it does not own while retaining valid ids/weights
exactly; every physical rank returns a BF16 plane; the coordinator performs one
ordered FP32 sum over N = 2/3/4/6 planes followed by one final BF16 round plus
optional shared contribution.

---

## 1. What actually executes an official routed expert

Two independent implementations exist. Only the first is in the shipped native
release.

### 1.1 Grouped slice pipeline (`slices`) — shipped for Spark backbone TP4

- `third_party/sparkinfer/b12x/moe/_shared/kernels/v41_slice_pipeline.py`
  - `V41SlicePipeline.__init__` (`:17`) defaults `experts=384, topk=6, intermediate=576`.
  - `__call__` (`:27`): `publish` live-row scalar → `V41RoutePlan` → optional
    output clear → fused compute → `V41SliceReduce`.
  - Compute launch (`:71-84`): `groups = max(1, min(rows*topk, experts + max(rows*topk - experts, 0)//16))`.
  - `V41DraftSlicePipeline` (`:100`) is the local dSpark equivalent:
    `experts=128, topk=3, intermediate=2304, DEFAULT_WIDTH=192`.
- `third_party/sparkinfer/b12x/moe/_shared/kernels/v41_route_plan.py`
  - `V41RoutePlan.__init__` (`:15`): `routes = capacity*topk`.
  - Three kernels: `pack` (`:47`), `prefix` (`:77`), `scatter` (`:90`).
  - `pack` predicates `ids[pair] == expert` for `expert` in `0..experts-1`
    (`:66-67`); it also writes `metadata[i,1] = 0` and `inverse[i] = -1`
    for every unclaimed slot (`:59-60`).
  - `scatter` writes per-group `metadata[group] = (expert, min(16, count-j), base)`
    and per-row `metadata[group, 3+local]`, with `row = -1` for padding rows
    (`:105-118`).
- `third_party/sparkinfer/b12x/moe/_shared/kernels/w4a8_v41_slice.py`
  - `V41FusedSliceKernel.__init__` (`:49`): `width in (64,128,192)`,
    `intermediate % 32 == 0`,
    `kernel_intermediate = align_up(intermediate, 128)`,
    `slices = ceil(intermediate/width)`.
  - `__call__` (`:63`): grid `(slices, groups, 1)`, block 128.
  - `kernel` (`:85`): reads `expert, rows, route_base` from `metadata[group, :]`;
    `active = Int32(rows > 0)` (`:122-124`) and the **entire FC1/activation/FC2
    body is inside `if active > 0:`** (`:125`). This is a genuine early exit:
    no weight staging, no MMA, no activation for that group.
  - FC1 loop `for kt in range(40)` (`:150`) — 40 K-tiles of 128 over
    hidden=5120. FC2 loop `for ot in range(40)` (`:290`) — 40 N-tiles.
  - Both loops are **single-buffered**: `cp_async4_shared_global` staging
    followed by `cp_async_commit_group()` / `cp_async_wait_group(0)` /
    `sync_threads()` per iteration (`:195-197`, `:313-315`).
  - Output masking: `if row < rows and start + col < self.intermediate` (`:266`).
- `third_party/sparkinfer/b12x/moe/_shared/kernels/w4a8_staging.py` — the
  N256/K128 lane-major staging primitives (weights, scales, K-slices). 64-bit
  offsets throughout.
- `third_party/sparkinfer/b12x/moe/_shared/kernels/v41_reduce.py` —
  `_reduce_tp4_kernel`, Triton, `TOPK` as `tl.constexpr`, ranks hardcoded to 4
  pointers, `row < M` masking, per-route BF16 rounding before FP32 accumulation.

### 1.2 Compact micro pipeline (`compact`) — experimental, not in the shipped Spark expert

- `third_party/sparkinfer/b12x/moe/_shared/kernels/v41_compact_pipeline.py`
  - `V41CompactPipeline.__init__` (`:12`): `_DirectW4A8CompactLaunch(num_topk=6,
    n=kernel_intermediate, experts=384, native_v41=True, wire_rows=True, unit_scales=True)`.
  - `V41HybridPipeline` (`:36`): chooses compact when `rows <= cutoff`, else the
    grouped pipeline. **Only instantiated when `--compact-max-capacity` is
    passed to the exporter.**
- `w4a8_compact_projection.py`, `w4a8_compact_activation.py`, `w4a8_phase2.py`
  implement FC1 / SwiGLU / FC2 with direct per-pair expert indexing.

**Measured fact:** `dist/spark-expert/V41_EXPERT_AOT.json` reports
`compact_max_capacity: null`, `compact_live_rows: null`, and all six variants
`implementation: "slices"` with `width` 64 at capacity 1 and 192 elsewhere.
So the shipped Spark expert always uses the grouped slice pipeline, including
the M1 decode path, and therefore already has genuine group-level early exit.

**Measured fact:** the `rtx_tp2` role's historical export
(`docs/sparkinfer-upstream-expert-compact-aot-20260916.json`) does the opposite:
`compact_max_capacity: 16`, with `v41_rtx_tp2_m1` and `m16` as `implementation:
"compact"` and `m80`+ as `"slices"`.

### 1.3 Sentinel behaviour, both paths (parent question)

Sentinel `384` (or any id `>= experts`, and any id `< 0`) is safe on **both**
paths, but the two differ in whether it saves compute:

| Path | Guard | Behaviour for sentinel routes |
| --- | --- | --- |
| Grouped (`V41RoutePlan.pack`) | `ids[pair] == expert` for `expert < experts` | **Never owned → not in any group.** Real early skip. `inverse = -1`. `V41SliceReduce` writes `0` for `inverse < 0`, and its launch is bounded by `min(live_rows*topk, routes)`. |
| Compact projection (`w4a8_compact_projection.py:470-471`) | `if expert >= 0 and expert < alpha.shape[0]` | Guarded, no OOB access, the pair's FC1 is skipped. |
| Compact activation (`w4a8_compact_activation.py:87-89`) | same guard, with the comment "invalid routes must not inspect their (intentionally uninitialized) FC1 boundary" | Guarded, skipped. |
| Compact phase2 (`w4a8_phase2.py:857-866`) | `expert_idx < 0` → `valid_rows = 0`; `expert_idx >= down_alpha.shape[0]` → `valid_rows = 0`; else-branch explicitly zeroes the route-output row | Correct output, but the route CTA still runs and writes zeros: **zero-out, not early skip.** |

There is already a test that exercises the all-invalid case through the public
binding: `third_party/sparkinfer/tests/moe/test_v41_expert_numerics.py:143-146`
sets `ids.fill_(-1)`, replays a captured graph, and asserts
`torch.count_nonzero(result) == 0`. That is a direct sentinel-zero precedent for
the grouped path, and `:139-142` asserts that on SM121 the M1/576/384 shape
selects `dynamic_route_mode == "direct"`.

### 1.4 Grid behaviour — precise statement (no over-claiming)

`groups` is a **runtime scalar argument** (`metadata` capacity is a separate
static argument), but its **value is computed by the compiled pipeline from
static geometry and the live row count**, not read from a device-side active
count:

```
groups = max(1, min(rows*topk, experts + max(rows*topk - experts, 0)//16))
```

Consequences that must be stated exactly:

1. With sentinel masking the launch grid does **not** shrink for `rows*topk <= experts`.
   At M1 top-6 it is 6 groups whether 3 or 6 of them are assigned.
2. What shrinks is **executed work**: an unassigned group has
   `metadata[group,1] == 0`, so `active == 0` and the kernel returns after two
   integer comparisons. The cost of a fully inactive group is the CTA launch,
   not the expert computation.
3. For `rows*topk > experts` the expression is a loose upper bound (it does not
   divide by 16 until the route count exceeds the expert count). Groups beyond
   `min(rows*topk, capacity*topk)` simply have `active == 0`.
4. A group is 16 rows of **one** expert, so a partially filled group already
   guards each row (`if g < rows`, `if g + 8 < rows`) and the MMA accumulates
   zeros for the padding rows. There is no per-route waste inside an active group.

Net: with the sentinel contract, at M1 each rank launches 6 group CTAs, executes
exactly the experts it owns, and the unowned groups cost CTA launch only. That
is the correctness-first minimal extension and needs **no third-party kernel
edit**.

### 1.5 Supported tiles and geometry

- `V41FusedSliceKernel` widths: `64`, `128`, `192` (`:50`).
- Logical intermediate widths in use: **576** (Spark backbone TP4),
  **1152** (TP2 / draft-TP2 / local), **2304** (full / coordinator dSpark).
- `assert intermediate % 32 == 0` and `slices * width <= kernel_intermediate`.
- Exported Spark widths by capacity (`dist/spark-expert/V41_EXPERT_AOT.json`):
  capacity 1 → width 64; 16/80/256/1024/4096 → width 192.
- Exported capacities: 1, 16, 80, 256, 1024, 4096. ABI 2 (FP32 route planes)
  for 1/16/80, ABI 3 (FP32 token accumulation) for 256/1024/4096.
- The dSpark local expert export at M1 uses `[9, 6, 5120]` FP32 route planes
  plus a `[6, 5120]` FP32 token output (slot 10 / slot 41) — the per-route
  `topk` axis is compiled into the scratch shape.

### 1.6 Weight representation, quantization contract, shard alignment

**Measured facts.**

- Activation contract: BF16 `[rows,5120]` is quantized to contiguous
  5280-byte rows — 5120 E4M3 payload bytes then 160 UE8M0 K/32 scale bytes,
  amax floor 1e-4 (`native/include/ds41rt_v41_experts.h:49-56`);
  `row_bytes: 5280` in both shipped manifests. Rows `1..4096`.
- Weight contract: official source tensors are `fp4_e8m0_k32`
  (`python/tools/export_b12x_v41_experts_aot.py:272`, `source_format=
  "fp4_e8m0_k32"`), packed at load time into the b12x N256/K128 lane-major
  W4A8-MX layout by `native/cuda/kernels/v41_expert_pack.cu`.
- Packed per-expert sizes (`v41_expert_pack.cu:52-62`), with
  `padded = align_up(intermediate, 128)`:
  `W13 = padded*hidden`, `S13 = padded*hidden/16`,
  `W2 = hidden*padded/2`, `S2 = hidden*padded/32`.
  `ds41rt_v41_expert_packed_sizes` **only accepts intermediate in {576, 1152,
  2304}**.
- `pack_expert_async` source order is `W1, W3, W2, S1, S3, S2`; destination
  halves are "first half up (W3), second half gate (W1)"
  (`v41_expert_pack.cu:85-86`); out-of-range rows write zero.
- Padding rule (`third_party/sparkinfer/b12x/moe/fused_moe/_impl.py:1522-1533`,
  `_dynamic_kernel_intermediate_size`): for `w4a8_mx` return
  `align_up(n, 128)`. Therefore:

| Config | Logical intermediate | Kernel/packed extent | Padding |
| --- | ---: | ---: | --- |
| TP4 | 576 | 640 | **storage** +11.1% on the N/K axis |
| TP2 | 1152 | 1152 | none |
| TP3 | 768 (=6x128) | 768 | none |
| Full | 2304 | 2304 | none |

**Correction on the padding claim.** The executed slices depend on the chosen
width, not on the padded extent. At width 192, TP4 uses `ceil(576/192) = 3`
slices = `3*192 = 576` columns of real compute; the padded 64 columns are
storage and are masked out at `w4a8_v41_slice.py:266`. The quantifiable padding
effect is therefore (a) **+11.1% packed weight and scale bytes read per expert**
and (b) a 640-wide prepared stride, **not** an 11.1% or 5.6% compute penalty.
No measured number for either effect exists yet; treat any figure as a
hypothesis to be produced by the matrix in section 4.

- Shard alignment (measured): official tensors are split along the
  **intermediate** axis of `w1/w3` and the **contraction** axis of `w2`.
  `rust/crates/ds41rt-loader/src/v41_catalog.rs:145-151` divides the byte length
  by 4; `rust/crates/ds41rt-loader/src/v41_expert_staging.rs:78-92` uses
  `Rank` in `0..4` with `config.moe_intermediate_size / 4`, and
  `BackboneTp2 { .. }` / `DsparkTp2 { .. }` use `/2` with `rank < 2`.
  `BackboneFull` uses the whole tensor.
- TP2/TP3-per-expert-EP therefore requires a **new** loader role that slices on
  the intermediate axis by 2 or 3 instead of 4, or replicates whole experts.
  The native packer additionally requires the loader-level
  `intermediate in {576,1152,2304}` accept-list to grow for TP3's 768.
- Global weight scales and activation scales are replicated per tensor
  (`V41TensorPlacement::BackboneExpertReplicated`, `v41_catalog.rs:29-34`).
- ABI surface: `ds41rt_v41_expert_launch_t` has 44 pointer slots + 7 scalars +
  stream + status (asserted 392 bytes, `native/src/v41_experts.cc:25-27`).
  `native/src/v41_experts.cc:204-209` requires
  `scatter_rows == num_tokens * info.topk` and
  `max_active_clusters <= 2*DS41RT_V41_SMS`, i.e. **`topk` is baked into the
  AOT `info` struct** (`export_b12x_v41_slices_aot.py:225-241`) and into every
  route-shaped scratch buffer, not read from a device scalar.
- **Unknown:** whether the TP2xEP2/TP3xEP2 `w2` slice boundary is
  byte-aligned for the packed N256/K128 layout at 1152 and 768; the packer
  computes destinations from `padded` and writes zeros out of range, but the
  loader must hand it correctly aligned source rows. Needs a check, not an
  assumption.

### 1.7 Current EP/ownership state versus the target contract

**Measured facts.**

- Every Spark rank currently receives the **same** full canonical top-6 route
  list: `rust/crates/ds41rt-transport/src/v41_expert.rs:119` ("the same request
  must reach every TP rank") and
  `rust/crates/ds41rt-daemon/src/v41_experts/execution.rs:642`
  (`request.copy_routes_into(&mut exchange.ids, &mut exchange.routing)`).
  Each rank computes all six routes over its TP slice.
- An ownership mechanism already exists but for a different axis and a
  different checkpoint format: `V41PairedRouteWord` packs `expert_id` in bits
  0..8 and a 2-bit `owners` field in bits 9..10
  (`rust/crates/ds41rt-transport/src/v41_expert/paired.rs:9-13`), gated by
  `V41_EXL3_PAIRED_REQUEST_FLAG = 1<<17`. Bits 11..31 are reserved.
  `V41PairedOwnershipBatch::write_local_ownership`
  (`paired.rs:76-89`) expands ownership into a **384-entry int32 device row**
  for kernels that take ownership as a separate tensor.
- `PairedProfile` / `PairedAssignment`
  (`rust/crates/ds41rt-daemon/src/v41_experts/paired.rs`) implement whole-expert
  ownership planning with a cost model, currently EXL3-only and driven by
  `catalog.exl3()`.
- Native world size is restricted to 2 or 4 in three places:
  `rust/crates/ds41rt-daemon/src/v41_experts/service.rs:133-136`
  (`validate_world`, plus "two Spark ranks require EXL3 experts"),
  `v41_expert.rs:29-33` (`v41_spark_executor_id`), and
  `coordinator.rs:120` (`ensure!(matches!(world_size, 2 | 4))`).
- Native host-side reduction has exactly two ABI paths, 2 and 4 planes:
  `rust/crates/ds41rt-ffi/src/v41_experts.rs:298-331` (`reduce_tp2` /
  `reduce`) into `ds41rt_v41_reduce_tp2_compact_bf16_async` /
  `ds41rt_v41_reduce_compact_bf16_async`
  (`native/cuda/kernels/v41_route_reduce.cu:161-200`), which are template
  specialisations `reduce_compact<2>` and `reduce_compact<4>` accumulating in
  FP32 with a single `__float2bfloat16_rn` at the end. There is no 3- or
  6-plane variant.
- `ReducePlan`-side size: `NativeTp4Wave::device_bytes(capacity)` reserves
  `capacity * (4*V41_PARTIAL_ROW_BYTES + 2*5120*2)` = `capacity*6*10240` bytes
  (`coordinator.rs:104-112`, asserted in `:517-526`) — a TP4 upper bound that
  is also sufficient for a TP2 transport.

**Implication.** Group count and route ownership already reduce executed work on
the grouped path; what is missing is (a) the coordinator/transport change that
assigns each expert to one group, (b) 3- and 6-world support, (c) a 3/6-plane
ordered FP32 reducer, and (d) the loader/packer role for TP2/TP3 intermediate
slicing. Items (a)-(c) are the parent's native plumbing agent's track; only (d)
partially touches third-party code, and only at export time.

---

## 2. Resource and readiness constraints

**Measured facts (read-only).**

- A full official native deployment is **live** and must not be disturbed:
  `ds41rt-coordinator` (v8 image, net=host, `--rtx-gpus 1`) on RTX GPU0, plus
  `ds41rt-spark-expert-{ostrich,dodo,emu,kiwi}-19441` (v8) as ranks 0-3.
  `127.0.0.1:8000` answers `/health` and `/v1/models` with HTTP 200; each Spark
  listens on `0.0.0.0:19441`.
- Also running: `ds41rt-coordinator-wip` and `ds41rt-coordinator-wip-dual`
  (dev images, 40 h / 31 h, net=host, no distinct listener) and
  `ds41rt-spark-expert-wip` on each Spark.
- **No `/wip/slots/` exists on any of the five hosts**, and
  `/home/tj/.cache/ds41rt-experiments` does not exist. The WIP slot mechanism
  documented in `AGENT_DEV_HINTS.md` is therefore not currently materialised.
- Disk: workspace filesystem is at **90% (377 GB free of 3.6 TB)**.
  `/mnt/scratch` has 2.3 TB free.
- `/home/tj/Developer/ds41rt/runs/tp-ep-preflight/` was being written during
  discovery (a Rust `cargo` debug build in `target/`) — treat as another
  agent's active work; do not clean it.
- Second rail mismatch (config vs actual): `ds41rt.config` declares
  `SPARK_*_LANE_B = 10.55.0.5-.8`, measured is `10.55.0.7-.10`
  (ostrich `.7` … kiwi `.10`). Recorded here because any topology change that
  exercises the secondary rail must resolve this first.
- Thermal history: `~/Developer/nvidiatempgraph/gpu_monitor.db` exists
  (5.3 MB, updated 2026-09-20 10:48) with schema
  `samples(ts, host, gpu, name, temp, util, mem_used, mem_total, power, mem_unified)`
  covering local + 4 Sparks for 24 h; the collector is live on port 13456.

**Measured fact that constrains the 6-rank topologies.** The expert service
spins one full CPU core per Spark **at idle** — a structural
`std::hint::spin_loop()` in the connection loop
(`rust/crates/ds41rt-daemon/src/v41_experts/service/local.rs`), with ~34 % of
the spin in a `getenv`/`strncmp` chain reached from
`protocol_v2_transport_timing_enabled` on every poll. This is recorded in
`runs/c1-cpu-profile/EXPERT-PROFILE.md` (in-flight work, not a doc). TP2xEP3
would add two more hosts each burning a core before any measurement starts.
Budget for it and do not attribute any timing anomaly to the topology before
that is understood. It is a host-CPU cost, not a GPU cost.

**Isolated build/run mechanism (verified from the repo, not yet exercised).**

- Disposable dev shell with the checkout mounted, one RTX GPU:
  `./scripts/ds41rt-dev.sh coordinator -- bash` (`AGENT_DEV_HINTS.md:19`).
- Persistent dev containers and kernel checks: `docker exec -it
  ds41rt-coordinator-wip bash`, `ssh -t ostrich docker exec -it
  ds41rt-spark-expert-wip bash`, `./scripts/run-on-hosts.sh
  ostrich,dodo,emu,kiwi 'nvidia-smi'` (`AGENT_DEV_HINTS.md:20-24`).
- WIP flow: `S=<slot>; ./wip.sh --slot "$S" --role both`, then
  `./scripts/run-wip.sh --wip-slot "$S" --dry-run` / `--restart`
  (`AGENT_DEV_HINTS.md:7-16`). **`--recreate` destroys shared containers and
  build caches — do not use it.** `scripts/run-wip.sh` checks for partial or
  config-mismatched active service state and refuses (`:512`).
- Kernel-only probes need no WIP slot at all: mount the frozen SparkInfer tree
  read-only into a dev container and run a Python probe, exactly as
  `docs/ds41-expert-tp-efficiency.json` did
  (`docker run --rm --gpus all -v /tmp/ds41-direct-native-source/third_party/sparkinfer:/b12x:ro ...`).
  That record also sets `B12X_DYNAMIC_W4A8_SHARE_INPUT=0` and
  `SPARKINFER_COMPILE_DISK_CACHE=0`. Note the exporter cache controls are
  `B12X_COMPILE_*`, not the obsolete `SPARKINFER_COMPILE_*`
  (`docs/ds41-expert-native-official.md:47-55`).
- Python wrapper: it is `scripts/run-with-python-env.sh`, **not**
  `python/tools/run-with-python-env.sh` (`AGENT_DEV_HINTS.md:27`,
  `DEVELOPER.md:45`). Do not cite the latter.
- Constraint from `AGENT_DEV_HINTS.md:5`: slots isolate **artifacts, not GPUs,
  ports, build staging or containers**; WIP builds and performance runs must be
  serialized. The live deployment occupies RTX GPU0 and all four Sparks'
  service ports, so a fair measurement needs either a coordinated window or
  GPU1-only / spark-only probes that never touch the service.

**Real-checkpoint fixtures are container-mount paths, not committed files.**
`qualify_v41_expert_slices.py --inputs DIR` expects `{0,1}-hidden.bin`
(80x5280 u8), `{0,1}-ids.bin` (80x6 i32) and `{0,1}-routing.bin` (80x6 f32).
Those are captured DS41RTE3 wire frames (e.g. `router/l{L}-m80-c{C}-input.bin`,
as used by `scripts/qualify-ds41-real-tp4.py`), and neither the fixtures nor
`qualify-ds41-real-tp4.py` are checked in. The route banks consumed by the
reduction-analysis tools are likewise runtime artifacts. The documented
invocation mounts them at `/inputs`, `/baseline`, `/hf` and `/native`:

```sh
python python/tools/qualify_v41_expert_slices.py \
  --snapshot /hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a... \
  --native-lib /native/libds41rt_native.so --candidate-dir /output \
  --inputs /inputs --baseline /baseline --rank 0 --layer 0 \
  --output /output/native-layer0.json
```

Consequence for planning: a new microbenchmark must either (a) reuse the
synthetic grouped-slices fixture, which is self-contained, or (b) budget time to
re-capture route banks. Do not assume real route fixtures are on disk.

---

## 3. Benchmark and numerical-reference infrastructure that already exists

| Artifact | What it gives |
| --- | --- |
| `third_party/sparkinfer/tests/moe/test_v41_expert_numerics.py` | **Primary oracle.** `reference()` (`:19-50`) is an FP32 PyTorch oracle for the exact BF16→FP8→BF16 sequence: LUT decode of FP4 codes × E8M0 scales, `qx @ w1.T` rounded to BF16 then clamped at 10, `silu(gate)*up*routing` rounded to BF16, quantized again, `@ w2.T` rounded to BF16. Also holds the all-invalid-ids zero test (`:143-149`). |
| `third_party/sparkinfer/tests/moe/test_v41_grouped_slices.py` | Grouped-path oracle and in-graph mutation harness. Tolerances `rel_l2 < 0.01` and `cosine > 0.9999` (`:174`); asserts unwritten output is untouched (`out[:, routes:] == 12345`, `:189`); rebuilds metadata and route weights between replays inside one captured graph (`:150-183`). Cases: `one`, `shared2`, `mixed6`, `shared16`, `shared80`, `return_one`. Parametrised over `(n,topk) ∈ {(576,6),(2304,3)}` and `width ∈ {64,128,192}` (`:44-46`). |
| `third_party/sparkinfer/benchmarks/benchmark_moe.py` | **The authoritative existing MoE harness and the preferred base for the matrix.** 4175 lines. Loads **real** V4.1 expert weights from `deepseek-ai/DeepSeek-V4.1-Flash` (`:1191-1290`, `fp4_e8m0_k32`, prefix `layers.{l}.ffn.experts`, `_slice_v41_tp_shard`); the V4.1 profile at `:467-478` is `tp_size=4, quant_mode=w4a8_mx, routing=model, validate=oracle`. CUDA graph on by default (`:3231`); L2 flush on by default at 2x L2 (`--flush-l2`, `--l2-flush-bytes`, `:3265-3276`); `--tp-size/--tp-rank/--tp-parallel` (`:3118-3120`); oracle validation with `ORACLE_TOLERANCES` (`:2505-2530`); `--graph-mode {single-op,multi-layer}`, `--activation`, `--swiglu-limit`, `--reference {flashinfer,flashinfer-mxfp8,none}`. Model path resolves `--model-path` > `B12X_MODEL_PATH` > HF cache. Use `moe_checkpoint_snapshot.py` for exact operand identity. |
| `third_party/sparkinfer/b12x/moe/_shared/kernels` unit tests (`tests/moe/test_v41_route_plan.py`, `test_v41_fused_slice.py`, `test_v41_token_accumulation.py`, `test_v41_tp4.py`) | Precedent for route-plan/reduce/token-accumulation contract tests. |
| `docs/measurements/mxfp4-adaptive-screen/{bench,make}-native-adaptive.py` | **Closest existing harness shape to what this goal needs.** Runs `V41FusedSliceKernel(width, grouped=True, intermediate=N)` over the grouped-slices fixture, compiles once, captures a graph, alternates arms, `assert_close(..., rtol=0, atol=0)` against the fixture output. Logs `{case, groups, exact, medians_us, samples}` over 9 samples x 50 replays. **Its `adaptive_sms` overlay is not a free knob and must be understood before reuse:** `make-native-adaptive.py` shards only the FC2 output-tile loop across grid-z (`for ot in range(output_shard*(40//shards), (output_shard+1)*(40//shards))`) and picks the shard count on-device from `active_groups[0]`. The FC1 loop is **not** sharded, so FC1 weight traffic is *duplicated* per shard. The measured split follows exactly: shared-expert cases improve (Spark one 124.35→111.03, shared16 219.48→201.17, shared8 178.58→165.02 µs) while distinct-route cases regress or stall (Spark mixed6 456.38→456.75, shared2 138.59→145.14; RTX one 79.91→82.85, shared8 80.35→84.97). Any EP arm that shards output must therefore be measured in the repeated-expert regime, not only distinct routes. |
| `python/tools/qualify_v41_expert_slices.py` | Real-checkpoint comparison of official TP4 slices vs a deployed native FP8 consumer; takes `--snapshot --native-lib --inputs --layer --rank`, includes GPU route planning and ordered slice reduction in candidate timing. |
| `python/tools/_v41_expert_native.py` | ctypes bindings for the native AOT entry points (`Native`, `library`, `Launch`, `Info`), incl. the 44-slot tensor table. The right harness base for a per-rank kernel measurement. |
| `python/tools/qualify_v41_tp2_numerics.py`, `qualify_v41_tp2_reduction.py`, `qualify_v41_expert_packing.py`, `qualify_v41_grouped_fp8.py`, `qualify_v41_token_accumulation.py` | Existing numerics qualifications for alternate widths/reductions. |
| `python/tools/collect_expert_route_bank.py`, `analyze_expert_reduction_coverage.py`, `analyze_expert_reduction_replay.py`, `plan_expert_reduction_replay.py` | Route-distribution capture and reduction coverage/replay analysis — reuse for the routed-distribution matrix. |
| `scripts/bench-ds41-release-decode.py`, `bench-ds41-concurrent-api.py`, `bench-ds41-release-prefill-matrix.py`, `scripts/api-smoke.sh`, `scripts/api-constrained-smoke.sh` | End-to-end serving measurement and smoke checks. |

**Baseline evidence (measured, for context only).**

- Official native end-to-end decode, published v7 images, official checkpoint
  `dba1be0a…`, 1x topology, 3 repeats at 400 W:
  **C1 code 134.38 tok/s**, weighted nine-category **93.57**,
  counting 1-200 **166.05** (`docs/release-v7-official-regression.md`).
  v6 recorded 130.41 / 92.00 / 161.58.
- The **currently running** v8 native deployment has **no published
  measurement**; v8 documentation only campaigned NVFP4.
- Per-Spark expert component timings, **native MXFP8, official geometry
  (intermediate 576, topk 6, experts 6, groups 6, exact oracle)**:
  M1 `146.03/130.95 µs` on GB10 at width 192
  (`docs/measurements/mxfp4-adaptive-screen/native-adaptive-spark-extra.log`).
  This is the single most relevant published number for the M1 decode path and
  it is a **kernel+planner+reduce** measurement, not a clean kernel-only number.
- **Strongest existing per-rank official-native expert timing**
  (`docs/ds41-expert-native-official.md:20-37`, `.json` retained): rank-0
  layers 0/1, all 24 permutations of four arms, ten graph replays per sample,
  after correctness, on one GB10. The launch sequence includes row-count
  publication, GPU route grouping, fused expert compute and ordered FP32 route
  reduction, and **excludes** input encoding, compact return, transport and
  coordinator execution. Median ranges in µs:

  | Rows | Deployed | w64 | w128 | w192 |
  | ---: | ---: | ---: | ---: | ---: |
  | 1 | 204-208 | **127-150** | 179-183 | 146-166 |
  | 2 | 449-451 | **252-292** | 276-316 | 259-296 |
  | 6 | 667-914 | 580-634 | 574-645 | **514-576** |
  | 16 | 1467-1748 | 1215-1370 | 1240-1492 | **1092-1262** |
  | 80 | 4388-4564 | 3792-4085 | 3743-4007 | **3198-3459** |

  Width ordering is non-monotonic (w64 wins at rows 1-2, w192 wins at rows
  6-80), one-row variance is large, and the record itself declines to call
  these release performance. This is the table the new matrix must reproduce
  and then extend along the EP axis; it is also the best available anchor for
  how much room a K-loop change might have.
- Historical synthetic TP1/2/4 scaling
  (`docs/ds41-expert-tp-efficiency.md/.json`, GB10, one GPU, b12x revision
  `f9ce62ca…`, 821.75 → 408.35 → 222.27 µs warm; 837.31 → 446.19 → 264.99 µs
  cache-pressure; TP4 kernel extent 640): **explicitly not serving evidence**,
  same random-weight distribution rather than shards of one tensor, and it
  varies **TP width with a fixed expert count**, so it does **not** measure
  expert replication. Use only as a machine-calibration reference.
- Existing NVFP4 numbers (`docs/release-v7-nvfp4-optimization.md:500-511`) are
  a **different kernel, format and precision** and cannot support or refute any
  official-native claim. Not usable as evidence for this goal.
- `docs/sparkinfer-upstream-integration-20260916.md`,
  `docs/ds41-expert-official-slice-comparison.md`,
  `docs/ds41-expert-native-official.json` and
  `docs/ds41-expert-native-slices.md` carry the export/link procedure and the
  retained hashes behind the table above — read them before re-exporting
  anything, so the new arms are built the same way.

**Baseline gap.** There is no current, isolated, per-rank official-native expert
kernel timing that (a) separates route planning, fused compute and reduction
within one process, or (b) varies the number of experts per rank at fixed
topology. The table above bounds the aggregate but not the split, and it
contains no EP arm. Filling that gap is the first thing the matrix must do; it
does not need new tooling beyond §5 Task 2.

---

## 4. Proposed microbenchmark matrix

### 4.1 Measurement phases

**Phase 0 — baseline profile (gates everything else).**
Establish where the M1 time actually goes before any optimization claim.
Decompose, on one GB10 at a time (Spark rank 0 of a non-serving, freshly
exported library):

1. `V41RoutePlan` alone (pack + prefix + scatter).
2. `V41FusedSliceKernel` alone, single launch, no graph.
3. The full `V41SlicePipeline` (plan + compute + reduce).
4. `V41SliceReduce` alone.
5. The native `ds41rt_v41_expert_launch` entry through `_v41_expert_native.Native`
   with real packed weights, including its internal planner.
6. Route-data upload (ids/routing H2D) and input-quantization cost.

Report per-arm: min / median / max over ≥ 9 intervals of ≥ 50 replays, plus
the set of raw samples. Only if phase 0 shows the compute is **not**
latency-dominated does a K-loop pipelining change become the leading candidate;
if it does show latency domination, pipelining is the leading candidate. Either
way, **do not start the pipelining work before phase 0 is reported.**

**Phase 1 — replication matrix.** The core question.

### 4.2 Matrix axes

| Axis | Values | Why |
| --- | --- | --- |
| Rows M (live tokens per rank) | 1, 2, 8, 16, 32, 80, 256 | 1-16 are decode/verify; 80+ is prefill-ish. M1 is the stated concern. |
| Config | **TP4** (intermediate 576/640, 4 ranks, each with 6 routes) vs **TP2xEP2** (1152, 2 groups x 2 TP ranks, each with **3 of 6 routes**) vs **static-3 diagnostic** (1152, single group, exactly 3 routes, no masking) vs **TP3xEP2** (768, 2 groups x 3 TP ranks, 3 routes) vs **TP2xEP3** (1152, 3 groups x 2 TP ranks, 2 routes) | Isolates replication from TP width. TP3's 768 is padding-free at both storage and compute. |
| Route distribution | (a) distinct experts; (b) **shared/repeated expert ids across M rows** (the amortization case); (c) zipf-skewed towards a few experts; (d) uniform random over 384 | Repeated ids decide whether grouped amortization survives replication. Distribution (b) is the one the parent asked about. |
| Masking | **masked6** (sentinel 384, canonical top-6 shape, `inverse=-1`) vs **static-3 diagnostic** | masked6 is the correctness-first contract. static-3 measures the cost of the 3 extra inactive group CTAs and is a diagnostic only, never a production shape. |
| Cache state | warm (L2-resident weights) vs cold (128 MiB device write before the timed replay) | Reload the exact method of `docs/ds41-expert-tp-efficiency.json` so numbers are comparable to the historical record. |
| Execution | eager launch vs captured CUDA graph replay | Graph replay is a serving requirement; the delta itself is a result. |
| Scope | full FFN (plan + compute + reduce) vs fused kernel only vs compute+reduce only | Separates kernel effect from fixed overhead. |
| Width | 64, 128, 192 (and 64 for M1) | Determines executed slice count and therefore the padding question. |

### 4.3 Arm definitions (must be explicit)

Every arm fixes: snapshot revision `dba1be0a…`, SparkInfer revision, kernel
width, intermediate, topk, mask policy, rows, distribution seed, and graph-vs-eager.
For the EP arms the route assignment is **expert → group by
`expert % EP`** so that each rank's route set is exactly the intended size, and
the masking is applied on the worker side per the parent contract
(`ids = 384`, `weights = 0` for unowned routes).

Both directions are required. Do not report a one-sided comparison: run
TP4→EP and EP→TP4 in the same process, reversing arm order every repetition, as
the adaptive screen harness does (`bench-native-adaptive.py:35-40`).

### 4.4 What is genuinely being compared (and what is not)

For M1 with 6 distinct experts. "Slices/expert" is `ceil(intermediate/width)`;
"active group CTAs" is `experts/rank x slices/expert`, counting only the groups
the mask leaves active. With masking a rank still launches its full group grid,
but the unowned groups return after `active = Int32(rows > 0)` (§1.4).

At width 192:

| Config | Ranks | Experts/rank | Routes/rank (masked6 / static-3) | Slices/expert | Active group CTAs | Intermediate extent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TP4 | 4 | 6 | 6 / n.a. | 3 | 18 | 576 (640 storage) |
| TP2xEP2 | 4 (2 groups x TP2) | 3 | 3 / 3 | **6** | 18 | 1152 |
| TP3xEP2 | 6 (2 groups x TP3) | 3 | 3 / 3 | **4** | 12 | 768 |
| TP2xEP3 | 6 (3 groups x TP2) | 2 | 2 / 2 | **6** | 12 | 1152 |

Slices/expert by width for the same geometries:

| Intermediate | w64 | w128 | w192 |
| ---: | ---: | ---: | ---: |
| 576 (TP4) | 9 | 5 | 3 |
| 768 (TP3) | 12 | 6 | 4 |
| 1152 (TP2) | 18 | 9 | 6 |

Two corrections to the original framing must be stated in any result:

1. **EP does not reduce active group CTAs at width 192 — it is not a work
   reduction.** TP4 and TP2xEP2 both execute 18 fused-expert CTAs per rank,
   because the intermediate grows exactly as fast as the expert count falls:
   `6 experts x 3 slices = 18` and `3 experts x 6 slices = 18`. TP3xEP2 and
   TP2xEP3 execute 12 (`3 x 4`, `2 x 6`). **Total useful arithmetic is
   invariant; only its partitioning across ranks changes.** What changes is
   per-rank CTA count at other widths, per-CTA weight bytes, the L2 working set,
   the padding tax, and the reduction plane width. At M1 the six experts of one
   token can all land in one group, so per-rank load is **not** guaranteed
   balanced — the parent's whole-expert scheduling concern is real. The matrix
   must include the balanced 3/3 and 2/2/2 masks, a reuse/imbalance mask, and
   the worst case (a token whose top-6 all map to one group).
2. **Per-expert K-loop length is identical (hidden/128 = 40 tiles) in every
   configuration**, because TP slices the intermediate axis, not hidden. So a
   per-CTA-latency-dominated kernel predicts near-equal M1 wall time across all
   arms. That is a **hypothesis**, and phase 0 is what tests it.

Also keep **launched CTAs** and **expensive math** separate when reporting. On
the grouped path an unowned route does no FC1, no activation and no FC2 — that
is the real saving. On the compact path an invalid route skips FC1 and
activation but its phase2 CTA still runs and writes zeros (§1.3); never describe
that as a skip and never add the two paths' counts together. The grouped path
remains the chosen production target.

### 4.5 Correctness gates (before any timing is interpreted)

1. `rel_l2 < 0.01` and `cosine > 0.9999` against
   `test_v41_expert_numerics.reference` — the existing gate, not a new one.
2. **Sentinel-zero**: every remapped route's output must be exactly zero.
   Assert `torch.count_nonzero(...) == 0` on a route whose id is 384 and whose
   weight is 0, with a poison value pre-written into its output row. This
   catches an uninitialized scratch alias.
3. **Bit-exact valid routes**: routes that are assigned must be bit-identical
   across a re-run with and without masking present elsewhere in the batch
   (`rtol=0, atol=0`), as `bench-native-adaptive.py:24` does.
4. **Changed masks in the same captured graph**: capture once, then replay with
   (a) rank 0 owning routes 0-2, (b) rank 0 owning routes 3-5, (c) rank 0
   owning all 6, (d) rank 0 owning none, then back to (a). Assert zero
   allocation across replays and correct output every time. This is the direct
   analogue of the existing in-graph metadata mutation at
   `test_v41_grouped_slices.py:150-190`.
5. **Entirely inactive group / rank**: all six routes sentinel for one rank;
   assert its BF16 plane is exactly zero, that the other ranks' planes are
   unaffected, and that the coordinator reduction with that zero plane is
   bit-equal to the no-plan variant. This is the case where a `counts`-based
   or `prefixes`-based implementation most easily breaks.
6. **Reduction order**: for N = 3 and 6 planes, assert the ordered FP32 sum
   with one final BF16 round matches a host-side FP32 reference exactly
   (`torch.equal` on the BF16 result), including with an optional shared
   contribution, and that `output == shared` aliasing still works.
7. **Multiple live counts under frozen resolution**: 1/2/8/16/32/80 through one
   prewarmed, graph-captured kernel, proving no new compile key was introduced
   by the masking change (`third_party/sparkinfer/AGENTS.md` requirement).
8. Quantization semantics unchanged: E4M3/UE8M0 K32 rows, amax floor 1e-4, and
   a tiny-activation case in the spirit of
   `test_v41_expert_numerics.py:156-192`.

### 4.6 Measurement hygiene

- Three repeats for any final claim; exploratory screens may be one repeat but
  must be labelled as such.
- Record: source/submodule revisions, geometry, mask policy, route seed and
  distribution, raw per-interval samples, medians, the power/clock sample
  window from `gpu_monitor.db`, graph-vs-eager, and cache state.
- Never compare across different widths or capacities without saying so.
- `docs/ds41-expert-tp-efficiency.md:13` already warns that a reverse-order,
  graph-replay, synthetic probe is not a DRAM measurement; keep that caveat on
  every new number.

### 4.7 Budget estimate

| Step | Effort | Notes |
| --- | --- | --- |
| Phase 0 decomposition on one GB10 | ~0.5 day of agent time, minutes of GPU | Reuses `_v41_expert_native.Native` and the adaptive-screen harness |
| Export two new geometry sets (TP2xEP2 1152, TP3xEP2 768) | ~0.5 day, needs one SM121 and one SM120 export | Export-time only; no serving |
| Masking + sentinel test suite in the third-party tree | ~1 day | Test-only; no kernel change expected |
| Phase 1 matrix on GB10 (5 configs x 7 M x 4 dists x 2 masks x 2 cache x 2 exec) | 1-2 days wall clock, serialized | Must not overlap the live service; prune to the rows that phase 0 identifies as decision-relevant |
| SM120 side (RTX expert layers / local groups) | ~0.5 day | Only if the GB10 result changes the config choice |
| End-to-end serving A/B for the winner | out of scope here; that is a configuration change | Requires a coordinated service window |

Disk: each exported variant set is small (the shipped Spark AOT tree is in the
tens of MB); the six-capacity export plus objects is comfortably under 1 GB.
The workspace is at 90 % full — write intermediates under `/mnt/scratch` or
`runs/<slot>/` and clean them.

---

## 5. Recommended implementation tasks and file ownership

Ordered, smallest-first. Tasks 1-3 need no GPU lease and no service downtime.

### Task 1 — Sentinel/masking contract tests (third-party, no kernel change)
Files:
- `third_party/sparkinfer/tests/moe/test_v41_grouped_slices.py` — extend with
  the masked6 cases and the entirely-inactive-group case.
- `third_party/sparkinfer/tests/moe/test_v41_route_plan.py` — assert
  `counts[e] == 0` for sentinel-only experts, `inverse == -1` on masked routes,
  and `metadata[group,1] == 0` for every group emitted from a masked route set.
- `third_party/sparkinfer/tests/moe/test_v41_token_accumulation.py` — the
  atomic-token path must also produce zero for masked routes (larger capacities).
Expected diff: tests only.

### Task 2 — Microbenchmark harness
Files:
- `third_party/sparkinfer/benchmarks/benchmark_v41_ep_groups.py` (new) —
  the matrix in section 4. Must print one JSON object per arm with the fields
  listed in 4.6.
- **Preferred base: `third_party/sparkinfer/benchmarks/benchmark_moe.py`.** It
  already loads real official V4.1 expert weights, exposes
  `--tp-size/--tp-rank/--tp-parallel`, has CUDA-graph and default L2-flush
  support, and has an `oracle` validation mode with declared tolerances. Drive
  the EP arms through it where the geometry allows rather than reimplementing
  weight loading and cache flushing.
- Fall back to the grouped-slices fixture plus the `_v41_expert_native.Native`
  ctypes binding for the arms `benchmark_moe.py` cannot express (per-rank
  sentinel masking, the entirely-inactive group, the 3/6-plane reduction).
- `docs/measurements/mxfp4-adaptive-screen/bench-native-adaptive.py` is the
  best template for the **arm-alternation and exactness-check shape**, not for
  its `adaptive_sms` overlay, which duplicates FC1 (§3).
- Use `benchmarks/moe_checkpoint_snapshot.py` to pin exact operand identity
  between arms.
Expected diff: one new benchmark file.

### Task 3 — Export geometry for the replication configs
Files:
- `python/tools/export_b12x_v41_slices_aot.py` — the role table at `:45-50`
  currently hardcodes `(384, 576, 640, 6)` for `spark`, `(384, 1152, 1152, 6)`
  for `rtx_tp2`, and `(384, 2304, 2304, 6)` for `rtx_backbone`. Add explicit
  `tp2ep2` / `tp3ep2` / `tp2ep3` roles or an `--intermediate` override; keep
  `topk=6`.
- `python/tools/export_b12x_v41_experts_aot.py:269` — the same `(384, 576, 6)`
  table.
Ownership note: the parent has assigned `python/tools/export_b12x_v41_*_aot.py`
to the native plumbing agent. Task 3 should be theirs; I list it only so the
geometry change is not duplicated.

### Task 4 — Only after phase 0: K-loop pipelining
File: `third_party/sparkinfer/b12x/moe/_shared/kernels/w4a8_v41_slice.py` —
overlap the next K-tile's `cp.async` group with the current tile's MMA in the
FC1 loop (`:150-196`) and the FC2 loop (`:290-315`). This is the only
third-party **kernel** change I would recommend, and only if phase 0 shows a
per-CTA stall cost. It is shared with every configuration, so it is worth doing
independently of the EP decision.

### Explicitly not recommended for this goal
- Any edit to `native/**`, `rust/**`, or the route-word/transport layer — that is
  the native plumbing agent's track and would collide.
- Any redesign that makes `topk` a live count or a per-rank variable shape.
  `topk` is compiled into the AOT `info` struct and every route-shaped scratch
  buffer; making it dynamic would create a new compile key, which
  `third_party/sparkinfer/AGENTS.md` forbids.
- Any use of the compact/hybrid pipeline for the measured arms: it zeroes rather
  than skips masked routes and its `rtx_tp2` historical record shows it is
  selected only for M <= 16 on the RTX side, which would confound the comparison.

---

## 6. Measured facts vs hypotheses vs unknowns

### Measured / verified by reading source or records
- Shipped Spark expert is the grouped slice pipeline at all six capacities,
  width 64 only at capacity 1; `compact_max_capacity` is null.
- `V41FusedSliceKernel` genuinely early-exits a group whose `metadata[group,1]`
  is 0 (`active > 0`), and `V41RoutePlan.pack` never assigns an id >= 384 to any
  expert, so sentinel masking produces no work for unowned routes and no OOB.
- The compact path also guards every direct expert index; phase2 explicitly
  zeroes the route output for an out-of-range expert instead of skipping it.
- Grid `y` is computed in the compiled pipeline as
  `max(1, min(rows*topk, experts + max(rows*topk-experts,0)//16))` — an upper
  bound with live-count awareness at the `min`, but not an active-group count.
- Padding: `align_up(n,128)`; TP4 576→640 (+11.1 % packed bytes and stride),
  TP2 1152 and TP3 768 unpadded. Executed compute columns depend on width.
- Quantization: E4M3 payload + UE8M0 K/32 scales, 5280-byte rows, amax floor
  1e-4; official weights are FP4 E8M0 K/32 repacked to N256/K128 lane-major.
- Loader splits on the intermediate axis by 4 (`/4`, rank<4) or by 2 for
  TP2 roles; whole-tensor for `BackboneFull`.
- Every Spark currently receives the identical full top-6 route list.
- A paired-ownership word/flag mechanism exists for EXL3 only, with 2 bits of
  owner at bits 9..10 and bits 11..31 reserved.
- Native world size is constrained to {2,4} in three places; host reduction has
  only 2- and 4-plane ABI variants.
- Existing tolerances: `rel_l2 < 0.01`, `cosine > 0.9999`; an all-`-1` ids
  replay already asserts an all-zero result on the grouped path.
- Baseline: official native C1 code decode 134.38 tok/s (v7, 1x, 3 repeats);
  no published measurement for the currently running v8 native deployment.
- Spark native MXFP8 M1 component timing through the full native pipeline
  (576/6/6 groups, exact oracle): 146.03/130.95 µs at width 192. For the
  compute+grouping+reduce-only anchor use the 127-150 µs w64 row-1 range from
  `docs/ds41-expert-native-official.md`; the two differ by exactly the pipeline
  scope.
- Live deployment occupies RTX GPU0 and all four Sparks; no WIP slots exist;
  workspace is 90 % full; the Spark expert burns a full core at idle.
- Config declares `LANE_B .5-.8`, actual is `.7-.10`.

### Hypotheses (must be tested, not asserted)
- **H1.** M1 grouped compute is dominated by per-CTA K-loop latency (80
  single-buffered stage/wait cycles at 40 K-tiles for FC1 and 40 for FC2),
  so per-rank CTA count is nearly irrelevant at M1 and EP replication alone
  will not speed up M1 decode.
- **H2.** Double-buffering/pipelining the two K-loops is the largest available
  M1 win, independent of configuration.
- **H3.** Replication wins at larger M and in the repeated-expert regime through
  a smaller per-rank L2 working set and the removal of the 640-vs-576 padding
  tax, not through reduced total arithmetic. Supporting evidence and a warning:
  the existing `adaptive_sms` split improves only cache-friendly shapes and
  regresses distinct-route shapes precisely because it duplicates FC1 traffic
  (see §3). Any replication scheme that re-reads weights across CTAs will show
  the same L2-dependent split, so measure the repeated-expert arm first.
- **H4.** Per-rank load imbalance from whole-expert scheduling is material at
  M1 and grows with EP degree (the all-six-to-one-group case).
- **H5.** The 640 storage padding contributes a low-single-digit-percent effect
  at M1, not the 8-21 % recorded for the historical synthetic TP4 probe.

### Bookkeeping discrepancy to resolve before sizing buffers

The two recorded Spark export manifests disagree on scratch bytes for every
capacity. `docs/ds41-expert-aot-qualification.json` records Spark m1
`736440` and m4096 `728644348`; the shipped `dist/spark-expert/V41_EXPERT_AOT.json`
records m1 `1246240` and m4096 `123706912`. Both agree on geometry
(`E384, hidden 5120, intermediate 576, kernel_intermediate 640, topk 6`) and
both differ from the coordinator's numbers. **Use the current `dist/` manifest
for any buffer sizing**, and treat the older JSON as a superseded snapshot; do
not mix them when computing EP-arm workspace. A 510 KB gap on the capacity-1
decode path is worth understanding before adding new per-rank scratch.

### Unknowns / cannot determine without work
- Whether the TP2xEP2 / TP3xEP2 `w2` intermediate-axis slice boundary is
  byte-aligned for the packed N256/K128 layout at 1152 and 768.
- The exact split of the native M1 number between route planning, fused
  compute, reduction and host upload.
- Which of the two recorded Spark scratch tables above is authoritative for the
  capacity-1 variant, and what produced the 510 KB delta.
- Whether the RoCE dispatch/collection path scales to 3 and 6 ranks within the
  existing lane design, and what the added per-layer latency is (transport work,
  not kernel work).
- The cost of one fully inactive group CTA at each width (expected tiny;
  unmeasured).
- Whether `V41SliceReduce` at `planes = ceil(1152/width)` for the EP arms
  (`6` at width 192, `18` at width 64) changes the M1 balance versus TP4's `3`
  / `9`, since the reduce reads all planes.
- Whether the second rail mismatch (`.5-.8` declared vs `.7-.10` actual)
  affects an EP topology that exercises lane B.
- Whether the rtx/local (SM120) expert path needs a parallel geometry for
  TP3xEP2's 768, given `_dynamic_kernel_intermediate_size` already returns 768
  unchanged and `export_b12x_v41_slices_aot.py` accepts `rtx_backbone` with
  2304 only.
