# NVFP4 shared-input activation quantization: findings and plan

Status: investigation complete, change not yet landed. Branch `dev`.

## Objective

Eliminate redundant NVFP4 (W4A4) activation quantization on the Spark side. The
producer currently quantizes a token's activation row once per *route*; a token
routed to six experts pays six quantizations of an identical BF16 row.

## Finding 1: the redundancy is real and located

`third_party/sparkinfer/b12x/moe/_shared/kernels/dynamic.py`:

- Non-shared (today): the producer loop runs over `total_pairs = num_tokens *
  topk` and quantizes per route at `quantize_block_fp4{,_fast}` ~4274/4278,
  writing each route's own physical row.
- Shared (`share_input_across_experts`): `producer_limit = num_tokens`
  (`dynamic.py:3605-3610`); the token row is quantized once at ~3923/3972 and
  fanned out to every routed expert's `phys_row`.

The ctor comment (`dynamic.py:1139-1152`) states the contract: the front-end
"quantizes each token exactly ONCE with `input_global_scale[0]` ... and fans the
same quantized row out to every routed expert's physical row", and "ALL experts'
FC1 input global scales must be identical".

## Finding 2: the wiring already exists

- `nvfp4_share_input: bool = False` is a first-class decode control
  (`b12x/moe/fused_moe/_tuning.py:90`), exposed as tuning knob
  `Knob(name="nvfp4_share_input", ...)` at `_tuning.py:478`.
- Planner wiring: `_impl.py:13949-13957` sets `share_input_across_experts` for
  `quant_mode == "nvfp4" and decode_config.nvfp4_share_input`.
- Feasibility predicate `can_share_input()` at `_impl.py:583-606`;
  `shared_input_scales` is exactly `can_share_input(input_scales_static=True)`
  (`_preparation.py:161`).

So this is not a new kernel feature. It is an unset knob plus a precondition.

## Finding 3: the precondition fails on the stock checkpoint (decisive)

`can_share_input` requires `a1_gscale.numel() == 1` or `_uniform_a1_scale` —
every routed expert's FC1 input scale identical.

Measured on `nvidia/DeepSeek-V4.1-Flash-NVFP4`
(snapshot `3431dde3247c13b5957f682b1e3c6fcae2566079`), 384 experts/layer:

| layer | proj | distinct | min | max | spread |
|---|---|---:|---:|---:|---:|
| 0 | w1 | 70 | 2.630e-04 | 4.302e-04 | 1.64x |
| 0 | w2 | 243 | 5.195e-05 | 4.092e-03 | 78.8x |
| 20 | w1 | 63 | 5.609e-04 | 1.302e-03 | 2.32x |
| 20 | w2 | 234 | 1.315e-04 | 7.068e-03 | 53.7x |
| 39 | w1 | 157 | 1.767e-03 | 1.037e-02 | 5.87x |
| 39 | w2 | 254 | 1.286e-04 | 5.580e-02 | 434x |

The scales are not uniform, so share-input cannot be enabled against the
checkpoint as published. A calibration is mandatory, not optional.

## Finding 4: the calibration is sound

FC1 computes with `alpha[e] = weight_scale_2[e] * input_scale[e]`
(`rust/crates/ds41rt-daemon/src/v41_experts/nvfp4.rs:276-279`). Choosing one
common activation scale `S` per layer and setting

- `input_scales[e] = 1/S` for all `e`
- `alpha_values[e] = w1_weight_scale_2[e] * S`

keeps the arithmetic consistent, because `weight_scale_2` stays per-expert and
absorbs the delta. Only the activation quantization range is shared. Because the
block scale is per-16 and adaptive, `S` needs to sit in a representable band
rather than match each expert exactly; the spread above bounds the risk.

`nvfp4.rs:266-275` already asserts gate/up symmetry per expert
(`w1_input_scale == w3_input_scale`). The new requirement is uniformity *across*
experts, which is strictly stronger.

## Finding 5: the hard gate is weaker than the heuristic gate

Two different gates exist and they are easy to conflate:

- `validate_moe_decode_config` (`_tuning.py:243-246`) — the hard gate — requires
  only `quant_mode == "nvfp4" and backend == "dynamic" and
  query.shared_input_scales`.
- `_nvfp4_query` (`_tuning.py:185-196`) — the *auto-enable* heuristic — also
  requires `not query.deterministic_output` and `dynamic_tile_m == 128`.

Explicitly passing `nvfp4_share_input=True` is therefore not blocked by the
validator from keeping `deterministic_output=True` and the M16 tile that the
decode path uses. No ctor guard rejects NVFP4 share-input on tile size or
determinism; the only share-input ctor rejection is for `w6a8_mx`
(`dynamic.py:949`). This is the experiment worth running, and it preserves the
current schedule and determinism.

Beware `nvfp4_materialize_intermediate`: the split-materialized admission
(`dynamic.py:1162-1177`) *does* require `mma_tiler_mn == (128,128)` and
`not deterministic_output`. Keep materialization off.

## Finding 6: competing hypothesis for the regression

Quantization is unlikely to be the whole 28% weighted-decode gap. One row/top-6
is 6 quantized activation rows; the same row feeds FC1 GEMMs of
`5120 x 640 x 2` per route. The dominant cost is the GEMM and its schedule.

`docs/release-v7-nvfp4-optimization.md` finding (a) already records the likely
real cause: the deterministic fused NVFP4 task collapses intermediate slices and
"one valid row/top-6 thus exposes six useful compute items, NOT six physical
launch blocks", whereas the W4A8 slice pipeline yields 54 CTAs
(`ceil(576/64)=9` slices x 6 routes). Decode parallelism, not quantization, is
the leading explanation.

Share-input saves the redundant quantize plus 6x packed-A/SFA write traffic and
the per-route atomic row allocation. That is worth having, but it should be
measured against the parallelism hypothesis rather than assumed to be the fix.

## Plan

1. Rust: add a common per-layer FC1 activation scale `S` to the NVFP4 load path
   (`v41_experts/nvfp4.rs`) — `input_scales[e] = 1/S`,
   `alpha_values[e] = w1_weight_scale_2[e] * S` — with `S` recorded as
   calibration provenance.
2. Export: pass `nvfp4_share_input=True` explicitly in
   `python/tools/export_b12x_v41_nvfp4_aot.py`, keeping
   `nvfp4_materialize_intermediate=False`, `deterministic_output=True`, and the
   existing tile.
3. Validate: `can_share_input` must pass; compare logits against the current
   BF16-input path on real weights; confirm the fan-out writes each token's row
   once.
4. Decompose the regression with the parallelism hypothesis in parallel:
   instrument per-route quantize time and CTA count before/after.
5. C1 code+topic decode and a 32k prefill run, before vs after.
6. Only then pursue a coordinator-side NVFP4 encoder and request wire format,
   which additionally needs a new E2M1/E4M3-K16 encoder and a request-side
   dtype flag (code 4 is currently response-only), and which does not remove the
   per-expert scale problem — it inherits the same `S` calibration.

## Progress

### Share-input library builds and links (SM120)

`-DDS41RT_ENABLE_V41_NVFP4_AOT=ON -DDS41RT_V41_NVFP4_SHARE_INPUT=ON` in
`ds41rt-coordinator-dev` on the local RTX PRO 6000 (SM120):

- `v41_nvfp4_rtx_tp2` and `v41_nvfp4_rtx_backbone` manifests both report
  `share_input: True`, `tile_m: 16`, 6 variants each (capacities
  1/16/80/256/1024/4096).
- `libds41rt_native.so` links: 16,960,712 B, versus 14,855,152 B built without
  the NVFP4 AOT.
- The library exports 204 `v41_nvfp4*` symbols, including
  `ds41rt_v41_nvfp4_local_expert_launch` / `_bind_scratch` / `_info`.

Trap to remember: `cmake/v41_nvfp4_experts.cmake` is included only when
`DS41RT_ENABLE_V41_NVFP4_AOT` is set, and that flag is absent from the
documented `build-native-coordinator-test` recipe. A build without it succeeds
while exporting no NVFP4 kernels at all. Verify artifacts, not exit codes.

Command used (the `ds41rt-dev.sh` wrapper cannot resolve a GPU selector on this
host because host `nvidia-smi` is broken; drive Docker directly):

```
docker run --rm --gpus all -v $PWD:/workspace/ds41rt \
  -v $HOME/.cache/huggingface:$HOME/.cache/huggingface:ro \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface:ro \
  -e HF_HOME=$HOME/.cache/huggingface -e HF_HUB_OFFLINE=1 \
  -w /workspace/ds41rt ds41rt-coordinator-dev:latest bash -lc '
    cmake -S native -B native/build-nvfp4-share -G Ninja \
      -DDS41RT_ENABLE_CUDA=ON -DDS41RT_ENABLE_RDMA=ON \
      -DDS41RT_ENABLE_SPARKINFER_COORDINATOR_AOT=ON \
      -DDS41RT_ENABLE_V41_NVFP4_AOT=ON \
      -DDS41RT_V41_NVFP4_SHARE_INPUT=ON \
      -DDS41RT_SPARKINFER_SOURCE_DIR=/workspace/ds41rt/third_party/sparkinfer \
      -DDS41RT_SPARKINFER_LOCK_FILE=/workspace/ds41rt/third_party/sparkinfer.lock.json \
      -DPython3_EXECUTABLE=/usr/bin/python3 -DDS41RT_CUDA_ARCHITECTURES=120
    cmake --build native/build-nvfp4-share -j 16'
```

Use `/usr/bin/python3` (container Python 3.12 with the b12x deps); the repo
`.venv` is host-built and cannot be executed inside the container.

Tests run against the pinned SparkInfer tree, all passing:
`tests/moe/test_nvfp4_shared_input_scales.py` (15),
`tests/moe/test_nvfp4_split_backend.py` (6),
`python/tests/test_v41_nvfp4_tile_policy.py` (39).

### The Spark kernels cannot be built here

`ROLES["spark"]` requires device capability `(12, 1)` (SM121, GB10). The local
cards are SM120, and the exporter hard-fails on a mismatch, so the
wire-facing Spark NVFP4 kernels must be exported on a Spark host or in the
ARM Spark image. Everything verified locally so far is the RTX path.


### Step 2 done and verified (compile level)

`python/tools/export_b12x_v41_nvfp4_aot.py` now takes `--share-input`, which sets
`nvfp4_share_input` on the decode config and `share_input_across_experts` on the
kernel. `nvfp4_materialize_intermediate` stays `False`, `deterministic_output`
stays `True`, and the tile is unchanged.

Evidence, `rtx_tp2` capacity 16 (`--rows 16 --tile-m 16`), run in
`ds41rt-coordinator-wip` against the pinned SparkInfer revision:

| build | `v41_nvfp4_rtx_tp2_m16.o` | sha256 |
|---|---:|---|
| baseline | 157,728 B | `ebd6ae244a12c80cde173cd9e2fca71294eea760034559181f3be6675be4cdfd` |
| `--share-input` | 176,416 B | `826845abbc5e837763c95e52ecf46008d72edfb576a0f3db456d759088ca8c2a` |

Both export cleanly and the objects differ, so the flag is not silently ignored:
the NVFP4 share-input producer compiles at M16 with deterministic output. This
confirms Finding 5 — no ctor guard rejects the combination, and the
`deterministic_output`/M128 conditions live only in the auto-enable heuristic.

This is a compile-level result. It does not yet show that the kernel is
numerically correct, nor that the runtime accepts it: `can_share_input()` still
fails against the shipped per-expert scales, so step 1 must land before the
kernel can be bound.

### Step 1 done and verified (compile + contract level)

`v41_experts/nvfp4.rs` now captures the raw FC1 scalars per expert, chooses a
single `S = max(w1_input_scale)` after the final batch, and publishes
`input_scales[e] = 1/S` with `alpha_values[e] = w1_weight_scale_2[e] * S`.

Scale direction is settled from the kernel body
(`b12x/_lib/intrinsics.py:6125`):

```
scale_float = max_abs * global_scale_val / 6      # clamped to E4M3 max
```

with `global_scale_val = input_global_scale = 1/S`. A larger `S` therefore
lowers the block scale and buys headroom against the saturation clamp, so
`max` is the safe direction. Only FC1 is broadcast — each expert produces its
own FC2 intermediate — so `down_input_scales` and `down_alpha_values` keep
their calibrated per-expert values.

Wiring: `DS41RT_V41_NVFP4_SHARE_INPUT` (default OFF) in
`native/cmake/v41_nvfp4_experts.cmake` passes `--share-input`; the export
manifest records `share_input`.

Evidence:

- `cargo check -p ds41rt-daemon` (via `scripts/run-with-python-env.sh` with
  `DS41RT_PYTHON=python3.12`; system Python 3.14 exceeds PyO3 0.22's ceiling)
  finishes with warnings only.
- `python/tests/test_v41_nvfp4_tile_policy.py` — 39 passed.
- `third_party/sparkinfer/tests/moe/test_nvfp4_shared_input_scales.py` — 15
  passed.

### The runtime has no safety net for this invariant

`can_share_input()` is a Python **planning/preparation**-time guard. The AOT
export builds a *synthetic* weight plan, so it has no real per-expert scales to
check, and the engine does not go through `_impl.py`'s launch path at runtime —
the Rust daemon binds the 44-slot ABI and calls the exported kernel directly.

The kernel itself does not validate either: it silently reads
`input_global_scale[0]` for every expert (`dynamic.py:3571-3573`). So if the
host published non-uniform scales while running a shared-input kernel, every
expert except the first would be quantized with the wrong scale and `alpha`
would not compensate — a silent numerical error, not a failure.

Correctness therefore rests entirely on the host publishing uniform
`input_scales` with matching `alpha`. The Rust change guarantees that by
construction (`1.0 / S` for every expert is bit-identical), which matters
because the contract is exact-bit, not approximate:
`test_one_ulp_difference_does_not_share`.

This is why the numerical comparison against the per-route baseline is a
required gate, not a formality.

### Shared-scale precision cost measured: negligible

The dequant step is `e4m3(max_abs * gs / 6) / gs` with `gs = 1/S`, which is
`≈ max_abs / 6` — the global scale largely cancels, so `S` only affects how
finely the *block scale itself* is represented in E4M3. That is a relative
granularity of ~2^-4 regardless of exponent, so the activation error stays
dominated by the E2M1 mantissa.

Measured against real per-expert FC1 scales from the published checkpoint
(4096-row synthetic blocks, real E4M3 rounding, code clamped to the E2M1
range):

| layer | spread | step err at max S | step err at min S | relL2 per-expert | relL2 shared |
|---|---:|---:|---:|---:|---:|
| 0 | 1.64x | 0.9% | 1.0-29.3% | 6.87e-03 | 6.76e-03 |
| 20 | 2.32x | ~0% | 3.1% | 6.93e-03 | 6.94e-03 |
| 39 | 5.87x | 0.4% | 1.8% | 6.94e-03 | 6.84e-03 |

The shared (max) scale is consistently *closer* to the ideal step than the
minimum-calibrated scale, and the end-to-end activation quantization error is
within ~1-2% of the per-expert path. Sharing `S` is therefore safe in this
regime, which is the main numerical risk retired.

Caveat: this bounds the *activation* quantization only, on synthetic blocks
whose magnitude was swept over 0.25-1.0. It does not exercise the FC1 GEMM,
routing, or the reduction. The full-model logit comparison is still owed.




## Environment notes

- Host `nvidia-smi` fails ("couldn't communicate with the NVIDIA driver") but
  GPUs are reachable inside `ds41rt-coordinator-wip`; use the container.
- NVFP4 checkpoint is mounted in-container at
  `/root/.cache/huggingface/hub/models--nvidia--DeepSeek-V4.1-Flash-NVFP4`.
