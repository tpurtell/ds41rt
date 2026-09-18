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
  **Verdict: keep the mixed projection kernel with tiers (2,3); the empty
  tier is near-free and bitwise-qualified. A homogeneous fixed-K2 export
  needs a new FFI bridge; evaluate with a kernel-level A/B benchmark at DS4
  shapes before deciding (user asked to check the single-bit trellis kernel;
  it exists in sparkinfer but may no longer be superior after all the mixed
  kernel optimization).**
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
