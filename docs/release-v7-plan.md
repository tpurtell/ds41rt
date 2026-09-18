# V7: NVFP4 W4A4 and EXL3 K2 compact checkpoints

V7 adds first-class serving for two published quantized checkpoints of
DeepSeek V4.1 Flash:

- `nvidia/DeepSeek-V4.1-Flash-NVFP4` (ModelOpt NVFP4 routed experts with
  W4A4 compute indication) in the current 1-RTX and 2-RTX + 4-Spark
  topologies.
- `diffbot/DeepSeek-V4.1-Flash-EXL3-2.0bpw-2x-RTX-PRO-6000` (uniform fixed
  K=2 EXL3 routed experts, source-precision MTP) in two compact profiles:
  2x RTX fully GPU-resident (no Sparks) and 1x RTX simulating a 32 GB RTX
  5090 budget with 2 Sparks.

Quality qualification is the tool-call eval only. Performance qualification
repeats the full v6-style measurement set per configuration. README gains
one headline table per checkpoint linking to full per-checkpoint reports.

## Checkpoint contracts (verified from safetensors headers)

### Loading findings (NVFP4)

- The engine ALREADY has a complete GLM-class ModelOpt NVFP4 path:
  `load_routed_quant_projection` (route.rs:18034-18098) reads U8 weight +
  F8E4M3-K16 weight_scale + F32 input_scale + F32 weight_scale_2 with TP4
  sharding; slab alphas = weight_scale_2 x 2^119 (route.rs:6490); W4A16
  e4m3_k16 AOT kernels qualified on BOTH GB10 and RTX PRO 6000 (but
  compiled for GLM geometry 6144/256/topk8).
- sparkinfer quant modes: nvfp4 (W4A4), w4a16, w4a8_mx, w4a8_nvfp4;
  source_format modelopt_nvfp4 accepted by nvfp4/w4a8_nvfp4.
- NVFP4 checkpoint MTP experts = standard native FP4 6-tensor layout
  (I8 + F8_E8M0) VERIFIED first-hand: dSpark needs no new work.
- NVFP4 config.json == official config.json except quantization_config
  (verified by diff).
- Gaps: OfficialV41Config::from_json model-id/config gate
  (v41_config.rs:17-34, deny_unknown_fields); expected_tensors contract
  (v41_catalog.rs:353-401 + count :465-470); recipe derivation
  (catalog.rs:327-335); TP4 K16 shard math (expert_format.rs:289-356);
  device pack for b12x swizzled sfb planes (v41_expert_pack.cu is
  E8M0-K32 byte-preserving; GLM cuda_b12x_w4a16_pack_* exists);
  V4.1-geometry e4m3_k16 AOT exports; W4A4 activation quant + ABI.
- Strategy: (1) loader recipe + W4A16 e4m3_k16 serving baseline (proven
  kernel family, exact weight reproduction); (2) W4A4 nvfp4 AOT exports +
  activation NVFP4 quant as the requested W4A4 optimization; A/B and
  report.

### NVFP4 loader contract (N1, committed c781297)

- `v41_nvfp4.rs` validates the full ModelOpt block (quant_method=fp8,
  quant_algo=MIXED_PRECISION, moe_quant_algo=NVFP4, expert_dtype=fp4,
  group_size=16, W4A4 config groups, the exact ignore list, a modelopt
  producer, per-layer `layers.N.ffn.experts` NVFP4 entries, unquantized KV),
  substitutes the canonical official FP8 block and still validates every
  other config field against the official snapshot.
- Catalog: backbone expert tensors become packed E2M1 (U8), F8_E4M3 per-16
  scale planes (TP4-sharded like native FP4) plus replicated FP32 global
  weight and activation scales; draft experts keep the native FP4 6-tensor
  layout. New `BackboneExpertReplicated` placement with whole-tensor reads
  and per-rank budget accounting.
- Verified against the local NVIDIA snapshot: 188,245 catalog tensors,
  92,160 TP4 payload/scale, 92,160 replicated scalars, 2,304 native draft.
- Next: N2 daemon staging/pack plus V4.1-geometry W4A16 e4m3_k16 AOT
  exports (baseline), then N3 W4A4 nvfp4 exports + activation quantization.

### nvidia/DeepSeek-V4.1-Flash-NVFP4 @ 3431dde3247c13b5957f682b1e3c6fcae2566079

- `quantization_config`: `quant_method=fp8`, `quant_algo=MIXED_PRECISION`,
  `moe_quant_algo=NVFP4`, `expert_dtype=fp4`, `group_size=16`, W4A4 group
  (`input_activations` and `weights` both 4-bit float group 16),
  `ignore=["*.attn.*","*.ffn.shared_experts.*","head","mtp.*"]`,
  producer modelopt v0.47.0rc0 (`dsv4-nvfp4-experts`).
- Routed experts (40 layers x 384): `w{1,2,3}.weight` U8 packed E2M1 pairs
  ([2304,2560] / [5120,1152] / [2304,2560]), `w*.weight_scale` F8_E4M3
  per-16-group ([2304,320] / [5120,144] / [2304,320]), `w*.weight_scale_2`
  F32 scalar global scale, `w*.input_scale` F32 scalar activation scale.
- MTP (dSpark) experts: native MXFP4 source format (I8 weight + F8_E8M0
  group-32 scale), identical to the official checkpoint.
- Attention, shared experts, embeddings, head, vision, Engram: identical
  FP8 E4M3 + E8M0 block-32 layout to the official checkpoint.
- Local size: 515 GB on every host (verified all 5).

### diffbot/DeepSeek-V4.1-Flash-EXL3-2.0bpw-2x-RTX-PRO-6000 @ 28b7ab71ba2eb15569b08b91a8ea07df8eda8a75

- `quantization_config`: `quant_method=exl3`, `bits=2`, `codebook=mcg`,
  `mtp_experts=source`, `mtp_experts_start_layer=40`,
  `weight_block_size=[32,32]`, `non_routed_quantization` = official FP8
  block (e4m3, dynamic, ue8m0, [32,32], expert_dtype fp4). No
  quantize_config.json, no tensor_storage, no ds41rt section.
- Routed experts (40 layers x 384): uniform fixed K=2.
  `w{1,3}.trellis` I16 [320,144,32], `w2.trellis` I16 [144,320,32],
  `suh` F16 [5120], `svh` F16 [2304], `mcg` I32 scalar 0xCBAC1FED (matches
  sparkinfer trellis_codebooks.py). Checkpoint-native naming, exactly the
  layout `v41_exl3.rs` expects.
- MTP (dSpark) experts: native MXFP4 source format (I8 + F8_E8M0),
  identical to the official checkpoint.
- Routed EXL3 payload ~3.16 GiB/layer (world 1), ~1.7 GiB/rank at TP2;
  ~126.5 GiB total routed + ~7.4 GiB non-routed: fits 2x 96 GB with KV.
- Local size: 334 GB on every host (verified all 5).

## Engine findings (subagent maps, file:line references in session)

### EXL3 path

- Production serve-native path parses recipe from checkpoint files:
  `v41_catalog.rs:427` -> `v41_exl3.rs:180-279` requires quantize_config.json
  with `meta.ds41rt` (schema `ds41rt.v41-routed-exl3.v1`), per-projection
  integer `bits_per_weight` K2..K5 from `tensor_storage`, and
  `native_quantization_config` swapped into config.json before strict
  validation. Uniform checkpoints get an empty adjacent tier appended
  (`v41_exl3.rs:150-164`): uniform K2 -> tiers [2,3].
- Kernels: sparkinfer has a homogeneous fixed-K fused kernel
  (`w4a16/kernel.py:9536`, K2..K6) and the mixed/variable-K projection
  kernel (`w4a16/mixed_trellis.py`). DS41RT's native path uses ONLY the
  mixed family (`export_b12x_v41_exl3_aot.py:118-161`), always packed
  routing; uniform runs as two-tier with an empty adjacent tier.
  **Historical proposal, not a measured verdict:** retain mixed tiers (2,3)
  pending an A/B. The claim that the empty tier is near-free was unverified.
  A homogeneous fixed-K2 export needs a separate bridge; see the homogeneous
  K2 A/B record below for the actual native qualification and serving results.
- **Two-family shipping (user requirement):** K3.25 (3;4) support must not
  break even though the staged model is no longer downloaded (no runtime
  verification possible - compile-time cleanliness only). v7 builds ship
  BOTH families as sibling package dirs `exl3-k23/` and `exl3-k34/`; a
  daemon resolver picks the family whose manifest bits equal the checkpoint
  decoder tiers, falling back to the legacy single `exl3/` dir (v5/v6
  images keep working). run.sh package identity covers both manifests.
- **Dev loop:** manual `serve-native`/`expertd-native` (RoCE) launches from
  WIP containers; the legacy pure-TCP real-full path is not used.
- AOT packaging: `package_v41_exl3_aot.py` + `native/cmake/v41_exl3.cmake`;
  capacities 1/16/80/256/1024/4096; spark tp4-rank0..3 (width 640/640/512/512,
  BF16), coordinator rtx-tp1 (2304, FP32), rtx-tp2 (1152, FP32), dspark
  (2304, 128 experts, top-k 3, BF16). Default `DS41RT_V41_EXL3_BITS="3;4"`;
  for diffbot must build `"2;3"`. Execution asserts meta.bits == resident
  tiers (`v41_experts/exl3/execution.rs:329`).
- Paired TP4 (EXL3_PAIRED_TP4=on) is a K3.25-oriented residency scheme:
  paired kernels require exactly two tiers and width 640; legal with [2,3]
  but tied to 4-Spark TP4; not used in the compact profiles.
- Gaps for the diffbot checkpoint:
  1. No quantize_config.json -> hard fail at `v41_exl3.rs:182`.
  2. `mtp_experts:"source"` unsupported: manifest reader requires MTP EXL3
     projections (`v41_exl3.rs:239-268`); catalog replaces ALL
     `.ffn.experts.` tensors with EXL3 projections
     (`v41_catalog.rs:448-464`); daemon dSpark stage selection is
     all-or-nothing on `catalog.exl3().is_some()` (`dspark.rs:172-183,303-311`).
  3. AOT packages must be rebuilt with bits "2;3" (SM120 coordinator +
     SM121 spark).

### Deployment / memory planning

- `MEMORY_RESERVATION=32GiB` is exactly the "simulate 5090 32 GB incl.
  headroom" knob (total-occupancy ceiling; KV + expert residency honor it).
  Zero code change for budget simulation.
- Zero-Spark (2x RTX fully resident): compute paths exist and are qualified
  (`LocalExpertWave` TP1 world=1 up to 40 layers; `tp2_ffn::Wave` TP2
  1..=40; layer-gated dispatch means all-local never calls Sparks; RoCE
  transport connects lazily). Blockers: daemon `peers.len()==4` ensures and
  [T;4] types, run.sh mandatory 4-host validation/launch + Spark-gated
  ready.json, 4-plane reduction ABI (unused when all-local).
  Interim: `--rtx-gpus 2 --rtx-expert-layers 40` works today with 4 idle
  Sparks (layer 39 kept as dead endpoint).
- 2-Spark (1x RTX 5090-sim): TP4 hardcoded end-to-end. Spark EXL3 worker
  requires world==4; AOT packages only tp4-width640/512; transport/reduction
  4-plane. EXL3 *byte planning* already supports world=2
  (`v41_exl3.rs:106`). Needs: SPARK_COUNT/world plumbing in launcher +
  daemon, `--world` flag for expertd-native, Exl3Worker world relaxation,
  new Spark AOT profile tp2-width1152 BF16, 2-plane reduction (reuse TP2
  reducer), qualification.
- Sparks shard intermediate WIDTH (all 384 experts per rank), not expert
  count. Native FP4 Spark format is TP4-only; the 1x compact profile must
  use EXL3 experts (which it is anyway).
- `select-release-gpus.py` DUAL fit constants are native-format measured;
  parameterize per checkpoint format (EXL3 K2 ~1.7 GiB/rank/layer TP2).

### v6 campaign machinery

- 5 repo bench scripts + per-release harness (recovered to
  `~/.cache/ds41rt-v7-package/` from the v6 evidence tarball) + 3 repo
  aggregation scripts (summarize/render/update, hardcoded v6 + official
  checkpoint; must fork for v7).
- Per configuration: 3 launches (dspark, retained_2k, target) with phases;
  ~50 min per config; ~2 h for two configs. Tool eval ~15 min (3 runs).
- 3 samples per cell, 400 W + stock memory hard-asserted, telemetry must
  survive each phase, corpus/context sha256 pinned.

## NVFP4 execution plan (from the subsystem maps)

Confirmed by read-only analysis of the export and daemon paths:

- **Packing needs no new CUDA.** `cuda_b12x_w4a16_pack_weight_async` /
  `pack_scale_async` (`native/cuda/kernels/b12x_direct.cu:292-357`, FFI
  `ds41rt-ffi/src/lib.rs:7081+`) are geometry-agnostic and byte-equivalent
  to sparkinfer's host `weight_layout="packed", scale_format="e4m3_k16"`
  preparation. V4.1 satisfies their divisibility rules. Alpha =
  `weight_scale_2 * 2^119` (matches `route.rs:6489-6491` and sparkinfer's
  bf16 global compensation).
- **W4A16 e4m3_k16 at V4.1 shapes is structurally valid**
  (`compile_w4a16_fused_moe(hidden=5120, intermediate in {576,1152,2304},
  experts=384, top_k=6)`), so the qualified W4A16 family is the baseline.
- **W4A4 hits an ABI gap**: `quant_mode="nvfp4"` selects
  `_DynamicMoELaunch` (37 pointers), not the engine's 44-slot
  `_DynamicMoEW4A8Launch`; it needs a new ABI variant plus real per-expert
  scale binding (slots 37/40 are constant 1.0 today, 38/39 alias 37).
- **Roles**: spark 384x576(topk6), rtx_backbone 384x2304, rtx_tp2 384x1152,
  dspark_tp2 128x1152; the dual zero-Spark critical path is rtx_tp2.
- Sliced plan: 0 staging (done) - 1 daemon load - 2 device pack (reuse GLM
  packers) - 3 AOT export + FFI/ABI - 4 three-state format branch - 5 W4A4.

### W4A4 AOT export (implemented)

`python/tools/export_b12x_v41_nvfp4_aot.py` exports the b12x dynamic
nvfp4 kernel per role/capacity and emits the engine's 44-slot bridge:

  slot 0..21 -> kernel 0..21; slot 22/24 -> FC1/FC2 payload; slot 23 ->
  FC1 and FC2 scale planes (the gate view aliases the fused plane because
  separate_w13_halves is false at every V4.1 n); slot 34..43 -> row counts,
  write rows, alpha vectors, token map/weights; engine scalars 44..51 ->
  the kernel's scheduling scalars and stream. The kernel's 46-entry
  parameter array matches exactly, so no ABI struct change or runtime
  wrapper is needed.

Small-M decode uses direct routing (validated for m<=4) and larger
capacities use grouped routing; `dynamic_tile_m=16` with explicit
`swiglu_limit=10` / `deterministic_output=True` avoids the CuTe optional
argument export bug. Kernel-owned scratch comes from the b12x core
workspace plan (24 tensors mapped to their slots); alpha (38/39) and the
per-expert scale vectors (37/40) stay weight-bound.

Validated on this SM120 host for rtx_tp2 (m1, m16), rtx_backbone (m1) and
dspark_tp2 (m1); the spark role needs the SM121 build host.

### W4A4 pivot (user direction)

W4A16 is NOT a target: the base DeepSeek checkpoint is already W4A8, so the
NVFP4 deliverable must serve 4-bit activations. The integration targets
`quant_mode="nvfp4", source_format="modelopt_nvfp4"` (the b12x W4A4 MoE
kernel: FP4 weights x FP4 activations with per-16 E4M3 scales and
per-expert FP32 runtime alphas). The 44-slot ABI gap remains: `nvfp4`
selects `_DynamicMoELaunch` (37 pointers), not the engine's 44-slot
`_DynamicMoEW4A8Launch`, so a new ABI variant is required.

Probe findings on this SM120 host (V4.1 geometry):

- `plan_b12x_fp4_moe_weights(quant_modes="nvfp4",
  source_format="modelopt_nvfp4", activation="silu", ...)` is accepted.
  `activation="silu_v41"` is MXFP4-only and must not be used.
- `moe._get_dynamic_kernel(...)` compiles V4.1 W4A4 and returns
  `(CompiledCuTeProgram, grid_x)`.
- `export_to_c` fails with an MLIR `custom op 'None'` error for default
  optional arguments. Passing explicit `swiglu_limit=10.0` and/or
  `deterministic_output=True` exports successfully; `direct_routing=True`
  also avoids it. `nvfp4_materialize_intermediate=True` is rejected for
  this specialization (needs the repacked W4A8 MX non-streaming path).
- Full role/row matrix validation is in flight.

### Slice 0 committed

`rust/crates/ds41rt-loader/src/v41_nvfp4_staging.rs`: twelve staged regions
(W1,W3,W2 packed weights and scales, then six replicated FP32 scalars) in
the device packer's order, with W1/W3 and their scale planes adjacent so
one pack call can consume each `[2*intermediate, ...]` matrix. TP4, TP2 and
full selections read through the existing catalog windows. Verified against
the local NVIDIA snapshot (packed payload non-zero, scalar scales finite
and non-zero, adjacency holds).

## Quick performance read (2x RTX EXL3 K2 zero-Spark, dSpark width 7)

Same bench scripts and controls as the v6 campaign; quick subset, not the
full release protocol. 400 W stock memory, 3 samples per cell, context
sha256 1881a1d1..., corpus sha256 1972e572....

**Prefill** (0K base, median effective tok/s after one warmup):

| Suffix | tok/s |
|---:|---:|
| 1K | 3,566 |
| 4K | 5,500 |
| 16K | 5,572 |
| 32K | 5,502 |

**Content-type decode** (median tok/s over three repeats; weighted nine
category median 150.64):

| Case | tok/s | Case | tok/s |
|---|---:|---|---:|
| Code | 216.9 | Natural JSON | 163.9 |
| Code + reasoning | 168.0 | Schema JSON | 150.6 |
| Math | 188.8 | Multilingual | 106.4 |
| Fable | 85.7 | Counting 1-200 | 337.4 |
| Hello | 127.6 | | |
| Topic | 98.5 | | |

Reference v6 dual native official checkpoint: weighted nine-category
dSpark 109.44, counting 221.64, prefill 0K row 4,725 (1K) .. 8,355 (16K).
The zero-Spark all-TP2 layout wins decode (no Spark round trip, 2-bit
weights) and trails on prefill (trellis decode compute). One
code-reasoning repeat produced an empty final response because all tokens
went to reasoning; the campaign should confirm the budget is sufficient.

### NVFP4 launch-contract diagnosis (September 19)

The first `cudaErrorInvalidValue` is a **host guard failure**, not a CUDA
kernel launch failure: the exporter confused the public b12x **policy**
sentinel `policy_max_active_clusters=-1` with the compiled entry's resolved
positive `max_active_clusters`. `_launch_dynamic_impl` passes `mac` returned
by `_get_dynamic_kernel` to the compiled callable (`_impl.py:12114`), and
`dynamic.py:2744` uses it directly as the cooperative grid dimension. The
shared positive-cap guard is correct and remains unchanged. The exporter
must record the returned positive count, not the policy sentinel.

The temporary launch diagnostic sat **after** that guard, explaining its
absence. A faithful probe of the pre-fix binary on SM120 proves the boundary:
NVFP4 m1 with all 44 slots non-null rejects -1/0; caps 1 and 188 reach the
compiled entry, return status 0, synchronize, and publish zero BF16 routes.
W4A8 m1 with corrected binding similarly succeeds for 1/188 and rejects -1/0;
its captured launch passes three graph replays.

The old W4A8 probe result was unrelated: it left slots 26–33 null, used BF16
instead of the FP8-K32 input wire, and did not allocate native packed planes.
The corrected probe follows the native binder aliases and advertised sizes;
it does not mutate workspace extent scalars or pass undersized allocations.

Additional integration defects found before the full rebuild:

- Deterministic NVFP4 slot 41 is BF16 `[rows,topk,5120]` **routes**, not token
  sums. The b12x public wrapper runs a separate top-k reduction after the
  exported dynamic kernel. The engine needs an explicit BF16-routes output
  kind and format-aware TP2 and Spark reduction, with full route-plane sizing.
- The former TP2 BF16 branch called a four-plane reducer with two null
  planes; that reducer correctly rejects null planes.
- Giving every missing scratch slot offset zero overwrote request pointers
  when binding a smaller Spark decode arena. Only unused W4A8 slots 26–33
  need placeholders; request and weight pointers must be preserved.
- Per-expert scale uploads reused one pinned source before asynchronous H2D
  copies completed. Each queued source must remain immutable until drained.
- Native weight sizing and Spark small/decode scratch sizing must query the
  selected family rather than an unconditional opposite-family interface.
- TP2 FFN was passing the native FP8 wire to NVFP4 kernels instead of the
  already-broadcast BF16 normalized rows. The remote Spark sender likewise
  needs BF16 bytes and a BF16 protocol tag, not its unconditional FP8 payload.

Initial qualification: `./wip.sh --slot v7q-a1 --role both` completed on
SM120/SM121; all six RTX variants now advertise positive clusters=188,
BF16 routes, and preserved external scratch slots. Host launch guards pass
for both family fixtures; Rust FFI metadata tests pass. The real CUDA BF16
route compaction/TP2 reduction test matches its exact nonzero oracle across
rows 1/16/80/3/1. RTX NVFP4 launch contracts pass rows 1/16/80 and W4A8
passes rows 1/16 (including poisoned outputs and three graph replays).
All 17 W4A8 generated artifact hashes and its manifest are unchanged.
Nonzero NVFP4 RTX/public-b12x parity at rows 1 and 16 is **bit-exact**:
initial absmax 2.671875, mutated input/routing absmax 5.46875, max error 0;
native BF16 reduction and three graph replays pass. The public m1 oracle
uses supported grouped deterministic routing while the exported native m1
variant uses direct routing. Spark NVFP4 launch contracts also pass rows
1/16/80; nonzero parity at rows 1/16 is bit-exact (absmax 1.3359375 before
mutation, 2.734375 after), including native reduction and three graph replays.
The numerical selftest therefore defaults to zero tolerance.

**End-to-end verified:** after the sender build, the exact 8-token smoke
returned a completion (reasoning exhausted that tiny budget); at 128 tokens
it returned `OK`/`finish_reason=stop` with a prefix-cache hit. Three concurrent
fresh requests returned `42`, `BLUE`, and `READY`, including 1,838-token
prefill. Full root cause, artifact identities, commands, exact API responses,
and qualification limits: [NVFP4 inference fix](release-v7-nvfp4-inference-fix.md).

### 5090 (and any same-capability GPU) coordinator support

Finding: the coordinator AOT artifacts are **not** SM-count specific. Asking
b12x to compile the same NVFP4 TP2 kernel with `max_active_clusters` 188, 170
or 128 on this 188-SM host returns the identical compile-time cluster count
(188) and an identical scratch plan (25 tensors), so the .o is the same
artifact. The cluster cap is a launch-time scalar, not a property of the
cubin, and the RTX 5090 and RTX PRO 6000 are both SM120.

What actually blocks a 5090 today is the engine's device pin:
`reject_expert_device` rejects a device whose compute capability *or SM count*
differs from the export host, and the recorded launch scalar is the export
host's count (188), which a 170-SM part cannot accommodate as a cooperative
grid.

So the change is runtime, not packaging:
1. Accept a device with the same compute capability but a different SM count.
2. Clamp the launch `max_active_clusters` to the current device's SM count
   (min(recorded, device SMs)) at initialize/launch.
3. One numeric test still owed: run the public b12x path with a reduced cap on
   this 188-SM card and confirm the output is unchanged, which proves the cap
   is a ceiling rather than a hard requirement.

This means the release image needs no second AOT set and no rebuild for a 5090
owner - which is the outcome the plan wanted. A per-SM-count export could not
even be generated correctly here without the hardware.

### 5090 support landed; the ceiling test needs the engine path

Implemented: `reject_expert_device` now requires only the same compute
capability (an SM-count difference logs a note instead of refusing to load),
and the launch clamps `max_active_clusters` to the live device's SM count.
Both are in `native/src/v41_experts.cc`. Verified regression-free: the dual
NVFP4 config still serves correctly after the change.

Why the ceiling test is not a config change: setting a positive
`max_active_clusters` through the public b12x config raises
"max_active_clusters requires the Triton route planner", and the internal
planner ignores the cap entirely - which is exactly why the runtime passes
-1 and why the engine supplies the value to the compiled entry itself. So
the proof that a reduced cap reproduces the same output has to go through the
engine's own launch, e.g. the Python bridge probe driving
`ds41rt_v41_nvfp4_tp2_expert_launch` with two different scalar values and
comparing slot 41. `native/tests/v41_nvfp4_cluster_cap_selftest.py` is the
scaffold for that and currently stops at the upstream planner restriction.

Also: the single-card NVFP4 prefill campaign was cut short by a service
restart during the run (my own regression check), not by a fault; it needs a
re-run for the 1x best-prefill cell, which stays a dash until then.

### GPU-time trade-off for the last prefill cell (decision point)

The full 30-cell single-card NVFP4 prefill matrix is running. It is a
multi-hour job, and the only reason it is running is that the report renderer
gates a *headline* prefill on a completed, passing campaign, so the base-0 row
alone (measured: best 4,131 tok/s at +32K) is disclosed in the matrix but not
promoted to the headline. That gate is the right default - it stops a partial
row being quoted as the headline figure - but it means one dash in the README
currently costs hours of exclusive GPU time in a configuration the release
owner has described as degenerate.

Decision taken: the full matrix was stopped and the GPU given to the
tool-call evaluation instead, because the release owner named the evaluation
as the only quality evidence they want and described the single-card
configuration as degenerate. The NVFP4 single-card evaluation is running now.

Consequence, stated so it is not mistaken for an oversight: the NVFP4 1x
best-prefill headline cell stays an em dash. The measured base-0 row (best
4,131 tok/s at +32K, three samples per cell) remains visible in the quant
report's prefill table with its partial-coverage disclosure, but it is not
promoted to a headline. If the matrix is later completed, the headline gate
opens by itself and no document needs editing.

### What the AArch64 expert image build needs

Ostrich currently has no release base image - `docker images` shows only the
local `ds41rt-spark-expert-dev` used by the WIP container, and nothing matching
`pytorch` or `26.05`. So building the expert image there requires the AArch64
`nvcr.io/nvidia/pytorch:26.05-py3` to be present or pulled first (it is
published multi-arch, so the pull should work where registry auth is
available), plus the build context.

The context is the repository root with the expert artifacts staged at
`.ds41rt-release-image/` (copy `.ds41rt-release-expert` there), because the
Dockerfile always copies from that path. The build command is the same as the
coordinator's with `DS41RT_ROLE=expert CUDA_ARCH=121`; the runtime digest check
then compares against the AArch64 `libcute_dsl_runtime.so`, which is the one
the expert packages recorded, so it should verify.

Do not cross-build this image on x86_64: its CuTe runtime cannot match
AArch64-built kernels and the packaging check will - correctly - reject it.

### Expert image needs the expert AOT built in the release runtime

The coordinator image builds and verifies (`ghcr.io/tpurtell/ds41rt-coordinator:v7`,
35.7 GB). The expert image fails its own packaging check:

    ValueError: EXL3 CuTe runtime mismatch

`python/tools/package_v41_exl3_aot.py` verify compares the digest of the
image's installed `libcute_dsl_runtime.so` against `manifest['runtime']`
recorded when the EXL3 package was exported.

The cause is **architecture, not version**: both containers carry
nvidia-cutlass-dsl 4.6.2, but the runtime library differs by target -

  release image (x86_64) 035a7e4cb4901cbf22aeba0af9682b23da4d30a46ba564f9affbb01a448d2b50
  ostrich (aarch64)      95e2fc4718588ef646f0fa3cb645ec9b42fdbff6d7c362ddce8c44e7ea465479  = the digest the expert packages recorded

I built the expert image on this x86_64 host from an x86_64 base, so its
runtime cannot match AArch64-built kernels. The expert image is for the
Sparks and must be **built on the AArch64 host** (ostrich) from an AArch64
base image, which is also what the release normally does. Every EXL3 family in
the cross-built set is rejected correctly: a kernel built against a different
target runtime is not the artifact the image claims to contain.

The fix is to build the expert role inside the release runtime rather than the
WIP container: run `scripts/build-release-artifacts.sh` on the SM121 host from
a container based on the release base image (the same
`nvcr.io/nvidia/pytorch:26.05-py3` and cutlass-dsl that `Dockerfile.release`
ships), with the source synced by the same guarded recipe. Do not work around
this by relaxing the check: the mismatch is the check doing its job - a CuTe
runtime difference can change generated code, so an artifact built against a
different one should not ship under this image's provenance.

### Publishing the images (next step, recipe derived from the Dockerfile)

`docker/Dockerfile.release` is role-parameterised (`ARG DS41RT_ROLE`,
`ARG CUDA_ARCH`) and always copies from the build context's
`.ds41rt-release-image/` directory, so each image is built from the artifact
set for its role. Both sets now exist:

  .ds41rt-release-image/    coordinator: ds41rt, libds41rt_native.so, exl3/
  .ds41rt-release-expert/   expert: ds41rt, libds41rt_native.so, exl3/

Build the coordinator image straight from the coordinator set, then stage the
expert set into the context directory for the expert image (keeping the
coordinator set aside first, since both use the same paths):

  docker build -f docker/Dockerfile.release \
    --build-arg DS41RT_ROLE=coordinator --build-arg CUDA_ARCH=120 \
    --build-arg DS41RT_ENGINE_COMMIT=$(git rev-parse HEAD) \
    --build-arg DS41RT_SPARKINFER_COMMIT=63e2140e4a32a977faa777c172b86679344fdc6a \
    --build-arg DS41RT_RELEASE_VERSION=v7 \
    -t ghcr.io/tpurtell/ds41rt-coordinator:v7 .

then swap in the expert set, build with DS41RT_ROLE=expert CUDA_ARCH=121 and
tag `ghcr.io/tpurtell/ds41rt-spark:v7`, and restore the coordinator set.

The image build itself re-verifies each `exl3-k*` family against the pinned
SparkInfer revision, so a staged family that does not match fails the build
rather than shipping.

### The release build needs the PyO3 ABI flag (or the container environment)

`scripts/build-release-artifacts.sh` run from the host shell fails in
`ds41rt-py` (or its dependents) with PyO3's forward-compatibility check:

  please check if an updated version of PyO3 is available
  set PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 to suppress this check

The host Python is 3.14, ahead of what pinned PyO3 0.22.6 accepts, and nothing
in the repository sets that variable - so the release build does not work
out of the box on a current host, only inside the WIP containers where the
interpreter matches. Prefix the build with
`PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1`, or run it in the container
environment that sets it implicitly.

Related, and the second time this release: stopping the serving process for a
build needs an explicit kill by PID inside the container. `pkill -f` from
outside silently left `ds41rt serve-native` running and holding 92 GB, and the
AOT export would then fail with an out-of-memory status while `nvidia-smi`
still looked busy rather than broken.

### Two operating rules learned the hard way this release

1. **Verify the expert layer boundary explicitly, and smoke before measuring.**
   The coordinator logs both the resident count and the remote count
   (`local RTX experts ready layers=N` and a remote-dispatch figure). Taking
   the wrong one starts the Sparks at the wrong first layer, which leaves a
   hole in the middle of the model: the service still reports ready and
   `nvidia-smi` still looks normal, but requests hang. A 40-minute campaign
   was launched against exactly that state. Always start the Sparks from the
   *resident* count and confirm a one-line request answers before starting any
   campaign; the launcher script now encodes that gate.

2. **A wedged CUDA context looks like an empty card.** Killing a serving
   process can leave its context behind: `nvidia-smi` reports a few MiB used
   and `--query-compute-apps` lists nothing, while every allocation inside the
   container fails with `cudaMalloc out of memory` (CUDA status 4). The
   coordinator dies at startup and the AOT exporters will fail the same way.
   Restart the coordinator container and confirm a fresh allocation succeeds
   before starting a build. This bit both a release-adjacent build and the
   prefill run during this cycle.

### Exact release build sequence (for the next round)

`scripts/build-release-artifacts.sh SOURCE_DIR ROLE CUDA_ARCH OUTPUT_DIR` where
ROLE is `coordinator` or `expert`, CUDA_ARCH is `120` for the RTX role and
`121` for the Spark role, and the source tree must be a clean checkout of the
release commit. Both roles must be built, and the Spark role builds on the
SM121 host (ostrich), not here.

Run the build inside the CUDA container, not on the host shell: the host has
no `nvcc` (CMake fails at `enable_language(CUDA)`) and its Python 3.14 trips
PyO3's forward-compatibility check. The container has nvcc and Python 3.12.

**Sync the source into the container first.** The container's `/wip/source`
lags the working tree between WIP builds, and it was missing the NVFP4
placement fix - building from it would have shipped a release without that
correction and nothing would have said so. The check that catches it is one
line: `grep -c 'expert_format=nvfp4' /wip/source/run.sh` must be 1 before the
build starts. Use the same rsync excludes as `wip.sh` (`-a --delete
--delete-excluded`, excluding `.git`, `rust/target/`, `native/build*/`, the
`.ds41rt-*` staging dirs), then `docker cp` into `/wip/source.next` and move it
into place.

Order that keeps the GPUs exclusive:
1. Let the single-card NVFP4 prefill finish (it is running now) and re-render
   the reports with `scripts/render-ds41-v7-quant-reports.py` and the headline
   with `scripts/render-ds41-v7-headline.py`.
2. `runs/v7q-a1/serve-diffbot.sh stop` and
   `COORD_CONTAINER=ds41rt-coordinator-wip-dual runs/v7q-a1/serve-diffbot.sh stop`,
   then confirm `nvidia-smi` shows both cards idle. The AOT exports OOM if any
   serving process still holds device memory.
3. Build the coordinator role for SM120; build the expert role for SM121 on
   ostrich. `docker/Dockerfile.release` already verifies the packaged EXL3
   families, including the new `exl3-k*` trees and the Spark TP2 shards.
4. Publish the images, cut `release/v7` from the release commit, and merge to
   `main`.

Reference for how the previous release did this: `docs/release-v6-plan.md` and
the v6 sections of `scripts/build-release-artifacts.sh`.

### Report integration discloses failures but not partials

The renderer's tool-eval section reads the harness aggregate, whose `failures`
list contains only `status == "fail"` entries. Partially credited scenarios
appear in the `statuses` counts and in the points arithmetic but are not named.
Run 1 currently has two of them (TC-66 and TC-58, one point each), so the
section as written would report "53 pass, 2 partial, 1 fail; 1xx/176 points"
and name only TC-43.

That is not wrong, but naming each partial is better: a scenario worth one of
two points is a real quality signal and a reader should not have to infer it
from a count. The per-run `tool-eval.json` under the run directory holds all
88 scenario results with points and status, so extending the reader to list
non-passing scenarios from that file (falling back to the aggregate) is the
next refinement. Left unwired deliberately: the aggregate path is written and
tested, and changing it without re-running the tests has already cost this
release one broken commit.

### First non-passing scenario in the NVFP4 evaluation

Run 1 of the single-card NVFP4 evaluation reached TC-43 "Omitted Required
Parameter" (category K, tool-call correctness) and scored **0 points**,
status fail, 47.9 s - the first non-passing scenario after a clean first
thirty-nine.

Two things follow. The harness discriminates rather than rubber-stamping, so
the decision to gate a published quality result on a completed, passing
campaign is doing real work. And this failure must be disclosed in the report
alongside the pass counts, which is exactly what the missing report
integration would currently drop - see the note above. Do not read a single
scenario failure as a quantization defect: it is one scenario, and the two-card
runs and the other configuration will show whether it is profile-specific,
model behaviour, or prompt sensitivity.

### The quant reports cannot yet publish eval results

`scripts/render-ds41-v7-quant-reports.py` has no code path that reads tool
evaluation output. Its quality section is fixed text, and the pending lists
name the evaluations, so a completed run would change nothing in either
report - the same integration gap as the prefill filename, found before it
could silently drop the evidence.

The harness writes `<output-dir>/summaries.json` (one entry per run, plus a
`tool-eval.json` per run directory) via `scripts/qualify-ds41-tool-eval.py`.
Wiring it needs the exact summary shape, which cannot be inspected until a run
completes, so it is deliberately left as a to-do rather than guessed at. When
the shape is known, the renderer should emit the points and pass counts for
each configuration that has a completed run, and must keep disclosing failed
samples explicitly instead of reporting only a pass rate.

### Reproducing the compact multi-turn issue

The harness shells out to an external `tool-eval-bench` CLI, which accepts a
scenario selector, so the failing case can be replayed on its own without a
full 88-scenario run:

  tool-eval-bench --model deepseek-ai/DeepSeek-V4.1-Flash --backend vllm \
    --base-url http://127.0.0.1:8000 --api-key local --format openai \
    --temperature 0 --hardmode --parallel 16 --timeout 900 --max-turns 12 \
    --reference-date 2026-09-19 --no-live --no-probe-engine \
    --scenarios TC-26 --label v7-exl3-compact-tc26 \
    --json-file /tmp/tc26.json --output-dir /tmp/tc26-report

Run it against the EXL3 compact profile to confirm the hang, then against the
2x zero-Spark profile, which has a far larger KV pool. If compact hangs and 2x
does not, the difference is the 2 GiB pool or the two-Spark TP2 path rather
than the harness; if both hang, it is the scenario plus this model. Either
result is actionable, and both are cheap compared with another full run.

### The TC-26 wedge is profile-specific, not a harness defect (correction)

Earlier in this cycle I concluded the tool-evaluation wedge on TC-26 was
"harness-side" because a three-message streamed request against the same
compact profile returned its chunks and a `[DONE]` terminator. That conclusion
was too broad and is now corrected.

The NVFP4 single-card evaluation ran **past** TC-26 and on to TC-27 without
intervention, on the same harness and the same protocol. So the wedge is
specific to the EXL3 compact profile (one card, 2 GiB KV pool, two TP2
Sparks) under that scenario, not something the harness does in general. The
ordinary-streaming test proved only that a *short* multi-turn request
terminates; it did not reproduce the scenario's shape.

TC-26 is "State Consistency (Multi-Turn)", the first multi-turn scenario in
the set, so the compact profile's cross-turn handling under a 2 GiB pool is
still the leading candidate and remains untested rather than exonerated. The
compact report's quality section should not claim the quality result is
merely pending-harness; it is pending an investigation into that profile.

### Tool-call evaluation notes (round 27)

The EXL3 compact qualification run stalled on scenario TC-26 ("State
Consistency (Multi-Turn)") and did not advance for thirteen minutes: only ten
`scenario_result` events are in the run log and its last write was at 03:55:32
while the process stayed alive and the endpoint kept answering `/v1/models`.
The harness's 900 s per-request timeout is what will close it, so the run will
eventually continue; it is not deadlocked at the process level.

Server side at that moment, from the coordinator's own startup and residency
line: `device_occupied_bytes=30,430,986,240` of `device_budget_bytes=34,359,738,368`
(28.3 of 32 GiB) with `runtime_headroom_bytes=2,147,483,648`, `rtx_layers=1`,
`remote_dispatch_layers=39`, `spark_world=2`. That is exactly the designed
compact budget, and no error or admission failure accompanies the stall, so
the profile was not out of memory or refusing work.

Resolution of the above: this is **not** the serving profile computing for a
long time. Sampling the card while the harness was waiting on TC-26 showed
`utilization.gpu = 0%` with the 31,264 MiB residency intact, and the second
card idle. The server was not generating, had not errored, and had not
refused work - it was simply idle while the client still waited.

So the stall is on the request/response or client side, not in the model or
the compact budget. That also retires the earlier 2 GiB-KV hypothesis for
this symptom: a truncated multi-turn chain would still show the GPU working.
The earlier "incomplete streaming response" seen on a prefill run was my own
service restart; this one is a stream that the client never saw finish while
the server stood idle.

Tested that hypothesis directly: a three-message multi-turn request with
`stream: true` against the same compact profile returned its chunks and then
`data: [DONE]` with curl exiting 0. Ordinary multi-turn streaming therefore
terminates correctly, so this is not a general streaming defect and the
harness wedge is not something a normal client would hit.

The qualification run was stopped after this (it sat on one scenario for over
fifteen minutes with the card idle and would not have advanced), and the
single-card NVFP4 prefill campaign was started in its place to close the last
performance cell.

Remaining sequence once the evaluation and the missing measurements are done,
in the order that keeps the GPUs free when they are needed:
1. NVFP4 1x best prefill re-run (needs the GPUs, about forty minutes).
2. Release docker images: `scripts/build-release-artifacts.sh` for both roles,
   then publish. This needs the GPUs exclusively for the AOT exports, so it
   must not overlap a serving campaign.
3. Cut `release/v7` and merge to `main`.

Also fixed this round: `python/tests/test_upstream_fp4_pack_math.py` had a
malformed module docstring that raised SyntaxError on import, hiding 51
reference tests from every suite run. Repaired; combined suites are now 648
passed, 3 skipped.

### The fixed-K2 kernel question, answered with evidence

The release owner asked whether a uniform K=2 checkpoint would be better served
by a fixed-size kernel than by the mixed projection kernel, and flagged the
premise as unverified. Checked:

* The premise holds. The checkpoint's `config.json` reports
  `quant_method: exl3, bits: 2, codebook: mcg` - a single scalar, so the
  decoder is uniform K=2.
* We nevertheless serve it through the **two-tier** family. `run.sh` derives
  the family tag from the bits value (`k${base}${base+1}`), which yields
  `k23`, so the deployed kernel is the mixed `[2,3]` projection kernel with the
  K3 tier unused for every weight in this model.
* A homogeneous alternative exists in SparkInfer:
  `b12x/moe/_shared/kernels/w4a16/mixed_trellis.py` documents "the single
  cooperative FC1/activation/FC2 grid used by homogeneous trellis", and
  `.../w4a16/kernel.py` refers to "the single-tier fused kernel" whose phase
  assembly is shared with the hybrid path.
* Our exporter cannot emit it today: `export_b12x_v41_exl3_aot.py` rejects
  fewer than two tiers and requires them distinct
  (`len(set(bits)) != len(bits)`), so a `[2]` family is a build-system change,
  not an export flag.

Historical conclusion: this was an untested and plausible optimisation; none
of the shipped numbers established the cost of the dead tier. The isolated
homogeneous K2 A/B below supersedes that open item. Release defaults and
published headline figures are not changed by the experiment.

### Homogeneous K2 A/B: measured, do not ship

The [full homogeneous K2 experiment record](release-v7-exl3-k2-homogeneous.md)
settles the open optimisation item for the tested dual-RTX zero-Spark profile.
A separate low-level `compile_w4a16_fused_moe` exporter and bridge can emit true
`bits:[2]` without touching k23/k34. Six TP2 capacity variants compiled;
57 actual-checkpoint checks were bitwise equal: 33 cross-native homogeneous/
mixed comparisons and 24 same-arm graph/eager comparisons, covering replay
and live tails. No vendor kernel change was needed. The generic
mixed exporter cannot express this merely by accepting one tier: homogeneous
has one weight tuple, different scratch aliases and two Int64 length arguments.

Three-sample homogeneous results (weighted/code/counting/best base0 prefill):
**149.95 / 204.75 / 336.73 / 5,336.26 tok/s**. Against the supplied shipped
145.10 / 222.06 / 337.35 / 5,702 figures, changes are **+3.34% / −7.80% /
−0.18% / −6.41%**. Historical weighted improvement is not kernel evidence:
the nonce seed and completion history differ.

A same-seed79001 mixed control, same binary/native library/topology, produced
identical outputs on all 30 requests and measured **155.00 / 206.80 / 323.65 /
5,489.99 tok/s**. Homogeneous therefore changed **−3.26% weighted, −1.00%
code, +4.04% counting, −2.80% best prefill**; every corresponding weighted
repeat and every prefill suffix lost. All decode completion checks and base0
prefill requests passed in both arms. This is sequential end-to-end evidence,
not proof that every homogeneous schedule is slower; counting alone improved.
It does not justify replacing the mixed family.

**Disposition: keep the release unchanged.** No image build/push/tag or
release launcher/build-default edit occurred. All 566 protected release-stage
files and 96 original WIP artifact files remained byte-identical. Experimental
code/patch, candidate slot, source/binary identities, native qualification,
telemetry and reproduction scripts are archived under
`~/.cache/ds41rt-v7-package/k2homog/`; raw requested files are
`performance/dual-exl3-k2homog.json` and
`performance/dual-exl3-k2homog-prefill-base0.json`, alongside the new
`dual-exl3-k23-control*.json` control. Only documentation is committed.
No compact profile was attempted. Reconsidering this would require owner
approval plus a new adapter/build/launcher integration and fresh evidence,
not silently selecting the archived candidate.

### v7 status at round 25

Landed and verified:
- Both new quants serve, measured, and documented: NVFP4 W4A4 (1x/2x) and EXL3
  2 bpw (2x zero-Spark and the 1x 32 GiB compact profile with two TP2 Sparks).
- The 32 GiB compact budget is real (peak 29,568 MiB, headroom-inclusive
  31,616 MiB, second card idle).
- The AOT family is SM-count agnostic; the engine accepts a same-capability
  part with fewer SMs and clamps the launch cluster cap. Astra checked all 18
  K2 variants at 188 vs 170 SM inventories.
- The Spark TP2 artifacts are packaged by the build, not just the WIP slot:
  `native/cmake/v41_exl3.cmake` appends `tp2-rank0`/`tp2-rank1` for the Spark
  role unless the paired-TP4 package is requested, and the release script
  ships both EXL3 bit families by default.
- Documentation: six-configuration headline in the README with the two
  per-quant reports linked beneath it, per-quant reports with SHA-256
  provenance, the configuration chart, and an Astra review that removed
  unsupported numbers and preserved the honest disclosures.

Still owed before the release can be called done:
1. Tool-call evaluation runs: the EXL3 compact profile is running now; the
   NVFP4 and EXL3 2x configurations still need theirs.
2. NVFP4 1x best prefill: campaign was interrupted by a service restart and
   has not been re-run, so that headline cell is still a dash.
3. Release docker images: only WIP artifacts have been built. The release
   build and publish has not been run.
4. `release/v7` branch cut and merged to `main`.
5. Pre-existing collection error in `python/tests/test_upstream_fp4_pack_math.py`
   (unrelated to v7, last touched in 6ab9475) - worth resolving before the
   release branch.

### v7 release deliverables (agreed scope)

Three performance table sets, all measured with the v7 protocol:
- Main README: the official (MXFP4) image, and only its headline table is
  re-rendered (the rest of that README's metrics stay as measured for v6
  because the official image is not being re-campaigned).
- `docs/release-v7-exl3-k2-performance.md`: EXL3 2 bpw, 1x (RTX 5090 profile,
  2 Sparks) and 2x (both RTX cards).
- `docs/release-v7-nvfp4-performance.md`: NVIDIA NVFP4 W4A4, 1x and 2x.

The headline table carries all six configurations: official 1x/2x, NVFP4
1x/2x, EXL3 1x/2x, with 1->2 change rates for the official and NVFP4 pairs
and a separate change row for EXL3 (its two configurations are not a 1x/2x
pair of the same topology). Titles shortened, C1 code included. The two
per-quant report links sit directly below the headline table.

Final state to deliver: a `release/v7` branch, everything merged to `main`,
and updated docker images. Before finishing, have Astra review the README and
the performance reports against the conventions of the previous releases.

### NVFP4 single-card placement: progress and the next blocker

The single-card W4A4 placement is wired: the full-width RTX backbone role is
exported (its own TU and symbol family, cmake packaging, FFI interface 7 with
role 2 geometry, `BackboneFull` selection), the single-RTX guard is gone, and
`LocalExpertWave` no longer hardcodes the W4A8 accessors - it followed the
resident weights' family for kernel selection and workspace sizing. That last
one was the real bug: the MXFP4 local kernel shares role 2 with the W4A4 one
and reports `Fp32Routes`, so W4A4 weights were being launched against the
wrong kernel and rejected by the family check. The failure message now names
the layer, both families and the kernel kind.

The 1x run now loads 4 full-width layers on the card (30.6 GB resident of
96 GiB), places layers 4-39 on the four Sparks, reaches API ready, and then a
request fails with:

    ExpertProtocolV2 route gate_weight must be finite

so the next step is the Spark dispatch path for the single-card placement:
the route weights reaching the transport are not finite, which points at the
routing buffer or the BF16 row path that Astra reworked for the TP2 case.

### NVFP4 W4A4 serving RESOLVED

Astra fixed and verified end-to-end inference; see
`docs/release-v7-nvfp4-inference-fix.md` for the full analysis and evidence.
Independently re-verified here: the dual-RTX + 4-Spark NVFP4 stack answers
`6*7` with `42`, and a Fibonacci request produced 803 characters of correct
reasoning plus working code, so the pipeline is numerically sound rather
than merely returning tokens. `cargo test -p ds41rt-loader` is 82 passed /
8 ignored and `scripts/tests` is 209 passed / 1 skipped.

Root cause and the correction to my earlier reading: `-1` is the *Python
policy sentinel* for `policy_max_active_clusters`, not the value the
compiled entry receives. The engine calls the compiled entry directly, so
it needs the resolved positive grid count (188 on RTX). My round-10 change
to record -1 was therefore wrong, and it masked the round-11 null-slot
fix. Both are now correct: every slot is bindable, and the grid scalar is
the positive count.

Beyond the launch, Astra fixed the deeper semantics I had flagged as open
but had not implemented:

* Deterministic NVFP4 output is **BF16 routes `[rows,6,5120]`**, not token
  sums. TP2 now retains all six routes and reduces them with dedicated
  FP32-accumulating BF16-route compaction and a two-rank reduction; the
  previous code also handed the existing four-plane reducer two null
  planes, which it correctly rejected.
* Input representation: TP2 supplies already-broadcast normalized BF16
  values instead of the native FP8 wire, and remote Sparks download BF16
  and use the BF16 protocol tag. Native and EXL3 keep their FP8 paths.
* Binding/lifetime: scratch binding preserves external request and weight
  slots while supplying placeholders only for the unused W4A8-only slots
  26..33; Spark small/decode arenas and native weight sizing query the
  selected family; each asynchronous scalar upload has its own immutable
  pinned source range until stream completion; cross-format rebinding is
  rejected.

New regression harnesses: `native/tests/v41_nvfp4_launch_selftest.py`,
`native/tests/v41_nvfp4_numerics_selftest.py`,
`python/tests/test_v41_expert_launch_contract.py`.

### NVFP4 launch probe (scaffold, not yet faithful)

`runs/v7q-a1/nvfp4_bridge_probe.py` drives an exported variant through the
engine's own 44-slot ABI from Python: it reads the C info, initializes a
handle, binds and initializes scratch, fills the slots the generated bridge
reads, and calls the family launch with the engine's scalars, then bisects
the scalars. Against the real library it reads the metadata correctly
(max_active_clusters now -1) and reproduces `status=1` for the W4A4 TP2
variant at every scalar setting.

**But the same probe also returns status 1 for the known-good W4A8 TP2
variant**, which the daemon launches successfully. So the probe is missing
something the daemon does and cannot yet be used to indict the W4A4 bridge.
Run it as:

  python3 nvfp4_bridge_probe.py <libds41rt_native.so> <variants.json> <rows> [family-prefix]

Next round should instrument the daemon's own launch instead of
reconstructing it: log the 53-slot array and the seven scalars for a
working W4A8 request, capture the same for W4A4, and diff the two. That is
a direct A/B on the real path and avoids guessing at the convention.

Also worth noting for that comparison: the b12x runtime writes the
deterministic kernel's output into `workspace.route_output` (not the
caller's scatter buffer), so the engine's slot 41 is the right target only
because the core workspace maps that tensor to the scatter slot - worth
re-confirming in the same diff.

### NVFP4 W4A4 serves end to end; request path fails at the expert launch

Status: the full serving stack now starts on the real NVFP4 checkpoint in the
current topology. The dual coordinator plans 20 TP2 layers on the RTX pair
(3.82 GB/rank/layer) and the four Sparks load layers 20-39 (~0.97 GB/rank/layer);
the API reaches ready and the placement handoff completes. Both sides load
W4A4 exactly: payload lands as [up(w3); gate(w1)] plus FC2 with one D2D copy
per expert, both scale planes are swizzled on device, and per-expert alphas
and activation scales ride in device vectors.

A chat request is admitted and reaches the expert launch, then fails with
`V4.1 expert launch failed with CUDA status 1` (cudaErrorInvalidValue) on the
RTX TP2 side (the Spark logs stay clean). Ruled out so far:

* `scatter_rows`: the runtime uses `routed_rows = num_tokens * num_topk` when
  `deterministic_output` is set (verified: rows=1 -> routed_rows=6,
  rows=16 -> routed_rows=96), which is what the engine already passes.
* slot coverage: every engine slot the generated bridge reads (0..25, 34..43)
  is populated on this path (scratch, weights, vectors).
* input format: the engine sizes the wire from `input_row_bytes()` = 10,240 B
  for BF16, and the Spark execution now selects the NVFP4 kernel and info
  (this was a real bug: ExpertExecution resolved the native accessors).

Remaining hypothesis for the next round: the launch scalars recorded at export
time. `max_active_clusters` in particular is a compile-time policy value (188)
in the exporter, while the b12x runtime derives the launch value from the
occupancy API per shape; a stale value reaching the cluster launch surfaces as
exactly cudaErrorInvalidValue. Next step is to record the runtime's own launch
policy for each capacity by planning an execution through the public API and
reading the encoded policy, then re-export and retry. If that is not it, the
next discriminator is a bridge-level GPU test that calls the exported variant
with the engine's exact 53-slot argument array and compares against the public
runtime's launch of the same shape, which isolates the scalar values from the
weights and workspace.

### NVFP4 topology correction (important)

W4A4 resident cost is ~3.82 GB per rank per TP2 layer (0.5 byte/weight plus
the E4M3 K16 scale planes, twice the MXFP4 scale bytes), so all 40 layers on
the RTX pair do NOT fit - unlike EXL3 K2 at 1.73 GB/rank/layer. NVFP4
therefore belongs in the **current 1+2 RTX + 4 Spark topologies** exactly as
the objective says, while EXL3 K2 Compact owns the all-resident profiles.
The dual planner still holds the layers that fit (roughly 17-19 of 40) and
the Sparks carry the remainder, so the **Spark NVFP4 path is required** for
any W4A4 serve.

Spark-path plan: `ExpertExecution` is currently typed to `ExpertWeights`
(`bind_layer(&ExpertWeights)`), so introduce a small `ExpertBinder` trait
(`bind` + `resident_bytes`) implemented by both `ExpertWeights` and
`Nvfp4Weights`, then reuse the existing execution/graph machinery with
`v41_nvfp4_expert_kernel` on the Spark role. The input wire becomes BF16
(10,240 B/row) instead of FP8-K32 (5,280 B/row) - the same dtype the
coordinator dspark role already ships - so the transport sizing follows
`input_row_bytes()` (already 10,240 for input_dtype 1) and needs no new
format, only the wider plane.

### NVFP4 build integration verified

The real SM120 WIP build now produces and links the complete W4A4 family:
six capacities (m1/m16/m80/m256/m1024/m4096) with their CuTe kernels plus
the engine symbols `ds41rt_v41_nvfp4_tp2_expert_{info,initialize,
output_kind,bind_scratch,initialize_scratch_async,launch}`. That validates
the exporter bridge, the cmake module, the two native TUs and the FFI
interface names end to end.

Also resolved: the kernel publishes a **token-major BF16 [rows,5120]**
result through `scatter_ptr` (engine slot 41), and TP rank partials sum
linearly, so the existing `ds41rt_v41_reduce_compact_bf16_async` reducer
covers multi-rank reductions - no new reducer is needed. And the
checkpoint's MTP drafts stay MXFP4, so dSpark reuses the existing path and
**no NVFP4 draft role is required** (only `rtx_tp2` for a zero-Spark
serve, plus `spark` for the 4-Spark topologies).

Remaining before a first W4A4 serve: the TP2 wave integration - a
`RankStorage::Nvfp4` / `RankBackend::Nvfp4` arm, a three-state routed
output layout (FP32 routes / FP32 tokens / BF16 tokens) in `enqueue`, a
BF16 output allocation, and the compact-BF16 reduction branch in
`PeerReduction`.

### NVFP4 open integration questions (next)

1. **Expert output semantics.** The nvfp4 core workspace exposes a bf16
   token-major `route_output` (with `scatter_ptr`), not the FP32 route
   planes the W4A8 slices emit. Before wiring the engine I must confirm how
   the dynamic kernel publishes results (scatter vs per-route partials under
   `deterministic_output=True`) and what `ds41rt_v41_expert_output_kind`
   should report per capacity, because the TP2 rank reduction and the
   dSpark/local reducers branch on it. Getting this wrong yields plausible
   but incorrect numbers, so it needs a bridge-level end-to-end test rather
   than inspection alone.
2. **Remaining wiring** once (1) is settled: cmake module + per-role native
   TUs with nvfp4 symbol names (`ds41rt_v41_nvfp4_*`), FFI resolution for
   those symbols (input_dtype=1 is already accepted), the Rust
   `Nvfp4Weights` plan/load/bind (payload H2D in `[up; gate]` order, the
   verified scale swizzle, alpha vectors computed host-side), and the
   three-site format branch.
3. **Wire decision for the 4-Spark topology**: BF16 routed rows (10,240
   B/row) versus today's FP8-K32 (5,280 B/row). The RTX-resident profile
   avoids the question entirely and should be the first served target.

## Compact TP2 implementation (current)

The new single-RTX compact path uses Spark TP2, not whole-layer Spark
partitioning. See [compact serving](release-v7-exl3-compact.md) for the design,
absolute 32 GiB ceiling, 2 GiB KV default, two-worker launch, residency reporting,
and the EXL3 same-SM120 reduced-grid compatibility work. This supersedes the
older "TP4-only" gap descriptions above; qualification status is recorded in
that document and the generated performance report.

## Implementation sequence

1. [ ] EXL3: raw-publication contract in the loader (synthesize manifest
   when quantize_config.json is absent and the config matches the diffbot
   contract; honor `mtp_experts:"source"`; keep native MTP tensors; dSpark
   native stages under EXL3 catalog).
2. [ ] EXL3: build "2;3" AOT packages (SM120 coordinator + SM121 spark),
   smoke diffbot serving in the current 4-Spark topology (1 and 2 RTX).
3. [ ] EXL3 compact profiles: zero-Spark 2x RTX (daemon peer relaxation +
   launcher SPARK_COUNT=0) and 1x RTX 32 GiB + 2 Sparks (TP2-Spark world
   plumbing + tp2-width1152 kernels + 2-plane reduction).
4. [ ] NVFP4: loader recipe for the modelopt NVFP4 contract (U8 E2M1 +
   E4M3 g16 scales + F32 global + F32 input scales); native MTP experts
   unchanged.
5. [ ] NVFP4: W4A4 execution - sparkinfer nvfp4 kernels
   (nvfp4_phase1/phase2, `modelopt_nvfp4` source format) exported for DS4
   Flash TP4 shapes (spark) and SM120 coordinator layouts; AOT packaging;
   runtime selection.
6. [ ] Qualification: fixed-K2-vs-mixed kernel A/B (informational),
   full v6-style performance campaigns per checkpoint/config, tool-call
   eval per checkpoint/config.
7. [ ] Reports: fork summarize/render/update for v7; per-checkpoint full
   mds; README headline tables + links.

## Evidence and release

- Work branch: `work/v7-nvfp4-exl3`. WIP slot: `v7q-a1`.
- Campaign workspace: `~/.cache/ds41rt-v7-package/` (harness adapted from
  v6). Disposable outputs under `runs/v7q-a1/`.

## Log

### September 18 (round 2 continued)

- **Raw EXL3 publication serving validated end-to-end** (WIP slot v7q-a1,
  1 RTX + 4 Sparks): all four ranks load 40 layers at ~925 MiB/rank/layer,
  coordinator places 12 RTX-resident K2 layers, API answers with coherent
  reasoning, prefix reuse hits, and strict JSON-schema smoke passes exact
  match. Fixed in the loop: DS41RT_NATIVE_LIB must be set on both roles for
  manual WIP launches (release images carry it as ENV); WIP binaries went
  stale because rsync/docker-cp mtimes confused cargo freshness - WIP
  builds now content-fingerprint the tree (build-wip-artifacts.sh).
- **Dual-RTX (2x RTX + 4 Sparks attached) also validated**: with EXL3 K2's
  ~1.73 GiB/rank TP2 layers, the dual memory planner places ALL 40 layers
  on the RTX pair by default (rtx_expert_layers=40, ~69 GiB peak per GPU,
  ~89 GiB occupied with the KV pool), leaving Sparks with zero dispatched
  layers. The model genuinely fits 2x 96 GB - the "EXL3 K2 Compact 2x fully
  RTX" profile is a planner reality already; the remaining profile work is
  letting the daemon/launcher run without the four Spark peers. Exact
  JSON-schema output verified on dual too (dual uses more reasoning tokens
  cold; budget accordingly).
- The legacy DS4_FLASH AOT kernels no longer compile against the pinned
  SparkInfer (mixed-kernel launcher ABI drift) - WIP flags now match
  release (DS4_FLASH_AOT=OFF); dev loop is serve-native only.
- verify-release-source-manifest.py rejected in-tree symlinks from the
  pinned transformers submodule (latent WIP break since Sep 14); now
  records resolvable in-tree file links by content.

### September 18 (rounds 1-2)

- Verified both checkpoints on all 5 hosts; mapped tensor contracts from
  safetensors headers (above).
- Completed four subsystem maps (loading, EXL3 kernels, deployment,
  campaign). Key outcomes: mixed kernel with empty tier is the designed K2
  uniform mode; `MEMORY_RESERVATION=32GiB` is the 5090 sim; zero-Spark needs
  daemon+launcher peer work; sparkinfer already ships NVFP4 W4A4 MoE
  kernels (nvfp4_phase1/phase2 + `modelopt_nvfp4` source format) and the
  blockscaled `mm_nvfp4` primitive.
- Found and cleared a zombie kernel pinning GPU 0 at 100% util (server
  reboot by user); both RTX cards idle clean at 0%/24 W afterward.
