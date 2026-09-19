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

### Step 1 open

`v41_experts/nvfp4.rs` fills `input_scales`, `alpha_values`, `down_input_scales`
and `down_alpha_values` inside a per-batch loop
(`for (offset, (plan, host)) in plans.iter().zip(&hosts).enumerate()`,
`expert = first + offset`, lines 183-281). Uniformity is required across every
expert the rank holds, which spans all batches, so this needs a two-pass
restructure: accumulate the four per-expert scalars into temporaries during the
loop, then after the final batch choose `S` and fill the vectors. The kernel
reads `input_global_scale[0]` for the shared row
(`dynamic.py:3571-3573`), so making every entry equal is both necessary and
sufficient.

`S` selection is unresolved. `max(w1_input_scale)` is the conservative choice
(most headroom against E4M3 block-scale overflow); the per-16 adaptive block
scale should absorb the spread, but this must be validated, not assumed.


## Environment notes

- Host `nvidia-smi` fails ("couldn't communicate with the NVIDIA driver") but
  GPUs are reachable inside `ds41rt-coordinator-wip`; use the container.
- NVFP4 checkpoint is mounted in-container at
  `/root/.cache/huggingface/hub/models--nvidia--DeepSeek-V4.1-Flash-NVFP4`.
