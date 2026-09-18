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
  needs a new FFI bridge; treat as optional unmeasured follow-up.**
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
