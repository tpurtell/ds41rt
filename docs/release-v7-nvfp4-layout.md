# DS41RT v7 NVFP4 W4A4 device layout (verified)

Provenance: reverse-engineered and empirically verified on the RTX PRO 6000
(SM120) host against the b12x `quant_mode="nvfp4"`,
`source_format="modelopt_nvfp4"` dynamic MoE kernel at DeepSeek V4.1
geometry. Evidence scripts live in `runs/v7q-a1/` (disposable):
`packer.py` (reference index math), `test_nvfp4_layout.py`,
`test_full_geometry.py`, `probe_grid_exact.py`, `diag_frontend_dump.py`,
`diag_micro_vs_dynamic.py`, `diag_precision.py`.

Independently confirmed here: gate (w1) and up (w3) share identical
`weight_scale_2` and `input_scale` across 4,608 sampled scalars
(3 layers x 384 experts x 2 fields, zero mismatches), so the fused-W13
single-alpha contract holds for this checkpoint.

# NVFP4 Dynamic MoE Kernel — Verified Device Buffer Layout (b12x, SM120)

Target kernel: the b12x dynamic fused-MoE kernel selected by
`quant_mode="nvfp4"`, `source_format="modelopt_nvfp4"`, `activation="silu"`
on 2× RTX PRO 6000 (SM120), at DeepSeek V4.1 geometry
(`K=5120`, `n∈{576, 1152, 2304}`, `E=384`, `topk=6`).
All statements below were verified empirically on the GPU (see "Evidence").

---

## 1. VERIFIED LAYOUT SPEC

Notation: `E` experts, `K` hidden (5120), `n` per-rank intermediate
(576 / 1152 / 2304). "Row" = output channel of the projection.

### 1.1 `b_w13` — fused FC1 payload (w1=gate, w3=up)

* Physical tensor: contiguous `uint8 [E, 2n, K/2]`, row-major.
* Row order within an expert plane: **rows `[0, n)` = up (w3), rows
  `[n, 2n)` = gate (w1)** ("kernel-native [up; gate]", i.e. b12x
  `w13_layout="w13"`; vLLM's `[gate; up]` is "w31" and is flipped in place
  by b12x, not by the packer).
* Byte `j` of row `r` packs FP4 elements `2j` (low nibble) and `2j+1`
  (high nibble), E2M1 with sign in bit 3 — exactly the checkpoint's
  `[rows, cols/2]` byte order; **the checkpoint payload bytes are consumed
  unchanged** (KEEP_SOURCE policy, no repack).
* Expert stride: `2n * K/2` bytes (no per-expert padding). Row stride
  `K/2` bytes. Base must be 16 B aligned (`assumed_align=16`; row stride
  `K/2` is a multiple of 16 when `K%32==0`).
* The kernel views this as a cute tensor `[w1_n=2n, K, E]` of
  `Float4E2M1FN` with stride order `(1,0,2)` — a zero-copy permute of the
  `[E, 2n, K/2]` storage.

### 1.2 `b_down` — FC2 (w2=down) payload

* Contiguous `uint8 [E, K, n/2]`, row-major. Row `r` = hidden output
  channel `r`; byte `j` packs intermediate elements `2j` (low) / `2j+1`
  (high).
* Expert stride `K * n/2` bytes, row stride `n/2` (multiple of 16 when
  `n%32==0`). 16 B base alignment.

### 1.3 `sfb_w13` / `sfb_down` — E4M3 block-scale planes ("F8_128x4" swizzle)

The kernel consumes the checkpoint's plain `[rows, cols/16]` E4M3 plane
**re-swizzled into 128×4 SF atoms** (the CUTLASS block-scaled SF layout;
this is exactly `b12x._lib.intrinsics.swizzle_block_scale`):

* Logical scale `(row r, k16-block j)` of expert `e` sits at physical byte
  offset from the plane base:
  ```
  off(e, r, j) = e * (Rp * Cp)
               + (r // 128) * (Cp * 128)     # 128-row atom
               + (j // 4)   * 512            # one 512-B chunk per 4 k16 blocks
               + (r % 32)   * 16
               + ((r // 32) % 4) * 4
               + (j % 4)                     # little-endian within the u32 word
  ```
  with `Rp = ceil128(rows)`, `Cp = ceil4(cols/16)`:
  - `sfb_w13`: rows = `2n`, cols/16 = `K/16` → `Rp=ceil128(2n)`, `Cp=ceil4(K/16)`.
    At V4.1 geometry `2n ∈ {1152, 2304, 4608}` (all `%128==0`) and
    `K/16 = 320` (`%4==0`), so the plane is exactly `2n × K/16` bytes with
    **no padding**; expert stride = `Rp*Cp` (= `2n*K/16`).
  - `sfb_down`: rows = `K` (5120, `%128==0`), cols/16 = `n/16 ∈ {36, 72, 144}`
    (`%4==0`); expert stride = `K * n/16`.
* The 128-row atoms span the **fused** `[up; gate]` row space — atoms can
  mix up/gate rows (e.g. n=576: atom 4 covers up rows 512–575 and gate rows
  576–639). Swizzle AFTER concatenating the halves, never per half.
* Padded rows/cols (when `rows%128!=0` or `(cols/16)%4!=0`) must be **zero**.
* Per 512-byte chunk: the u32 word at `(r%32)*16 + ((r//32)%4)*4` holds the
  four E4M3 scales of k16-blocks `4*(j//4)+0..3` of row `r`, byte `b` =
  block `j%4==b`.
* The kernel builds the SF TMA descriptor via
  `blockscaled_utils.tile_atom_to_shape_SF(b.shape, sf_vec_size=16)`; the
  front-end writes the activation SFA plane with the **same** atom formula
  (verified byte-exact, see Evidence 4).
* Base alignment 16 B; expert stride is a multiple of 512 B.

### 1.4 Per-expert f32 scale vectors (each `[E] float32`, contiguous, 16 B aligned)

Given the checkpoint's per-projection scalars
`weight_scale_2` (FP32, dequant multiplier: `W = E2M1 × E4M3_SF × weight_scale_2`)
and `input_scale` (FP32, **dequant** multiplier: `A = A_q × SF_A × input_scale`):

| launch vector         | value                                            |
|-----------------------|--------------------------------------------------|
| `input_global_scale`  | `a1_gscale = 1 / gate_proj.input_scale`          |
| `alpha`               | `weight_scale_2(gate) × input_scale(gate)`       |
| `down_alpha`          | `weight_scale_2(down) × input_scale(down)`       |
| `global_scale`        | `a2_gscale = 1 / down_proj.input_scale`          |

Derivation (what b12x does): the packer hands
`w1_global_scale = weight_scale_2(gate)`, `w2_global_scale = weight_scale_2(down)`,
`a1_gscale = 1/input_scale(gate)`, `a2_gscale = 1/input_scale(down)` to
`prepare_b12x_fp4_moe_weights`, which computes the runtime alphas
(`_prepare_modelopt_nvfp4_runtime_alphas`)
`alpha = w1_global_scale / a1_gscale = ws2 × input_scale` and
`down_alpha = w2_global_scale / a2_gscale` (torch.div, on device, f32).
`input_global_scale`/`global_scale` are the prepared `a1_gscale`/`a2_gscale`
tensors themselves (bound directly when contiguous f32 `[E]`).

There is **no 2^119 folding** anywhere in the nvfp4 dynamic path (that lift
exists only in the w4a16 kernel). The only reciprocal folding is
`a*_gscale = 1/input_scale` — the ModelOpt `input_scale` is the reciprocal
of the b12x `a*_gscale` quantization-scale convention.

Kernel math per route (token `x`, expert `e`, router weight `w`):
```
A_q,SF_a = nvfp4_quant(x, g=input_global_scale[e])   # SF=e4m3(amax·g/6), A_q≈x·g/SF
gate = alpha[e] · Σ_k (A_q·SF_a)(k)·(W1_q·SF_w1)(r_gate,k)
up   = alpha[e] · Σ_k (A_q·SF_a)(k)·(W3_q·SF_w3)(r_up,k)
interm = silu(gate)·up                    # staged through BF16
I_q,SF_i = nvfp4_quant(interm, g=global_scale[e])
out[x] += w · down_alpha[e] · Σ_k (I_q·SF_i)(k)·(W2_q·SF_w2)(r,k)   # BF16 scatter-add
```
The algebra `alpha × (A_q·SF_a) = (ws2·is) × (x/is) = ws2·x` makes the
quantization scales cancel exactly — verified empirically by perturbation
invariance (Evidence 6).

### 1.5 swap_ab and other kernel-mode effects on the layout

* `swap_ab = (quant_mode=="nvfp4" and gated and n%128!=0 and n%32==0)` →
  **True for n=576, False for n=1152/2304**. It only swaps the FC1 MMA's
  operand roles internally (weights ride the M role); **the published
  payload/scale bytes are identical in both regimes** — verified
  end-to-end at n=576 and n=1152/2304 (Evidence 1, 7).
* Gate-half byte offset inside `b_w13`: `n * K/2` from the expert base, in
  both regimes (up half at offset 0). Same for the scale plane: gate rows
  start at logical row `n` of the fused swizzled plane (byte `n/128` atoms
  in — atoms may mix halves, §1.3).
* `separate_w13_halves` (n%32==16) never triggers at V4.1 geometry; only
  then would up/gate get separate base pointers (gate base =
  `base + n*K/2`). Not applicable here.

---

## 2. Reference packer

`runs/v7q-a1/packer.py` — pure host torch/numpy index math (transcribable
to CUDA). API:

* `swizzle_sf_128x4(sf_u8[R, Cb]) -> [ceil128(R), ceil4(Cb)]` — the §1.3
  mapping, vectorized index arithmetic, zero padding. Inverse
  `unswizzle_sf_128x4` included for verification.
* `pack_w13(up_payload, up_sf, gate_payload, gate_sf)` →
  `b_w13 = cat([up; gate], 1)`, `sfb_w13 = swizzle(cat([up_sf; gate_sf], 1))`.
* `pack_down(down_payload, down_sf)` → `b_down`, `sfb_down`.
* `map_checkpoint_scales(ws2_gate, is_gate, ws2_down, is_down)` → the four
  launch vectors of §1.4 plus the `prepare_b12x_fp4_moe_weights` inputs
  (`w1_global_scale = ws2_gate` etc.).

Byte-exactness: `swizzle_sf_128x4` output equals
`b12x._lib.intrinsics.swizzle_block_scale` on every tested shape
(`R,Cb` ∈ {(1152,320), (2304,320), (4608,320), (5120,36), (5120,144),
(128,4), (100,5), (129,7), (48,3)} — exact shapes and padded shapes),
and round-trips through `unswizzle_sf_128x4` (Evidence 3).

---

## 3. GPU correctness test

`runs/v7q-a1/test_nvfp4_layout.py` — builds deterministic synthetic raw
ModelOpt tensors (random payload bytes, E4M3 scales in [1/8, 1], chosen
`weight_scale_2`/`input_scale`), packs them with `packer.py`, runs the
REAL kernel through the public b12x API (`plan_weights` → `prepare_weights`
→ `plan_execution(override=MoeDecodeConfig(backend="dynamic", ...))` →
`bind` → `run`), and compares against a dequantized torch reference
(`silu(gate)·up`, top-k routing, weighted sum) computed **from the raw
tensors only** (never touching the packer's swizzle).

Command:
```
./scripts/run-with-python-env.sh python runs/v7q-a1/test_nvfp4_layout.py
./scripts/run-with-python-env.sh python runs/v7q-a1/test_full_geometry.py
```

Observed errors (kernel vs raw-tensor dequant reference):

| case                                   | max_abs | rmse   | cos        | out/ref absmax   |
|----------------------------------------|---------|--------|------------|------------------|
| A K=5120 n=576 E=4 m=64 (swap_ab=True) | 0.0747  | 0.0114 | 0.99660522 | 0.7773 / 0.7812  |
| B K=5120 n=1152 E=4 m=64 (swap=False)  | 0.0908  | 0.0155 | 0.99690592 | 1.0938 / 1.1172  |
| C K=256 n=64 E=4 m=16 (minimal)        | 3.7e-4  | 3.3e-5 | 0.99981868 | 0.01526 / 0.01526|
| G E=384 n=576 topk=6 (spark)           | 0.0334  | 0.0061 | 0.99698102 | 0.3867 / 0.3828  |
| H E=384 n=1152 topk=6 (rtx_tp2)        | 0.0469  | 0.0087 | 0.99715489 | 0.7539 / 0.7617  |
| I E=384 n=2304 topk=6 (rtx_backbone)   | 0.0684  | 0.0126 | 0.99695414 | 0.8047 / 0.7852  |
| D ws2 ×4 / ÷4 perturbation             | 0.293   | 0.0507 | 0.99726045 | 3.9688 / 3.9531  |
| E input_scale ×¼ / ×4 perturbation     | 0.0747  | 0.0114 | 0.99660343 | 0.7812 / 0.7812  |
| F ws2 + input_scale perturbation       | 0.291   | 0.0507 | 0.99725991 | 3.9844 / 3.9531  |

The ~1% rmse residual is **intrinsic W4A4 noise, not layout error**:
* **Kernel-vs-kernel mutual diff** (`probe_mutual_and_sensitivity.py`):
  micro vs dynamic on identical packed buffers: **max_abs=0.0117,
  rmse=6.6e-4** at out_absmax 0.777 — the two independent kernel
  implementations agree ~17× tighter than either agrees with the exact-f32
  oracle (rmse 0.0114). The residual gap to the oracle is therefore the
  oracle's idealized arithmetic, not a layout/semantic error.
* Evidence 5 (grid-exact probe): with unit scales and FP4-grid data
  (every quantized operand bit-identical between kernel and host), the
  dynamic kernel matches the exact-staging oracle to **max 2 bf16 ulps**
  (max_abs 4096 at out_absmax 4.87e5, rmse 553 = 0.27 ulp, cos 0.99998).
  The kernel computes exactly the documented math.
* Error budget for the random-data oracle gap: (i) 0.16% of activation
  nibbles flip one FP4 step at exact dyadic threshold ties (synthetic
  bf16-data artifact; ~1e-5 frequency on non-dyadic data); (ii) the FC2
  input requant flips similarly on the dyadic-ish bf16 intermediates —
  each flip is a 12–25% step on one element and dominates the rmse
  (oracle self-perturbation of gate/up by ε gives rmse ≈ 0.8e-3·(ε/1e-4)
  … 4.0e-3·(ε/3e-3), same order); (iii) QMMA internal accumulation
  differences vs exact f32 (shared by both kernels, ~0.1%).
* Case E proves the scale algebra: changing `input_scale` (and thus
  `a_gscale` and `alpha` together) leaves outputs bit-near-identical
  (out_absmax 0.7812 vs case A ref 0.7812) — `alpha = ws2 × input_scale`
  cancels `a_gscale = 1/input_scale` exactly.

Minimal geometry: the dynamic kernel accepted `K=256, n=64, E=4, m=16,
topk=2` (case C, near-exact 3.7e-4). `k=5120`/`E=384` are NOT required;
the layout is geometry-independent given `K%32==0` (payload row 16 B) and
`n%32==0`. `K%128==0` for unpadded SF atoms; `2n%128==0` avoids mixed-atom
padding (both hold at V4.1 geometry). m≤8 routes to the micro backend
(same published weight contract — also verified, Evidence 2). Tests
forced `dynamic_tile_m=16` via `MoeDecodeConfig` override and
`fast_math=False` (for an exactly-modelable quantizer); the MMA tile only
re-tiles routed rows and changes no weight bytes. A `fast_math=True`
control run of case A lands in the same error envelope
(max_abs=0.0747, rmse=0.0115, cos=0.9966).

---

## 4. Evidence log (all runs on RTX PRO 6000 SM120, torch 2.13+cu130)

1. **End-to-end layout proof** — `test_nvfp4_layout.py` cases A/B/C and
   `test_full_geometry.py` cases G/H/I (table above). Any half-order,
   nibble-order, swizzle-offset, or expert-stride error would give cos≈0
   garbage; observed cos ≥ 0.9966 with matching output magnitudes at all
   three production geometries and both swap_ab regimes.
2. **Cross-kernel agreement** — `diag_micro_vs_dynamic.py` +
   `probe_mutual_and_sensitivity.py`: the independent micro kernel
   consuming the same packed buffers agrees with the dynamic kernel to
   **max_abs=0.0117, rmse=6.6e-4** (out_absmax 0.777) — shared published
   contract, no dynamic-specific layout.
3. **Swizzle byte-exactness** — `swizzle_sf_128x4` ≡
   `b12x._lib.intrinsics.swizzle_block_scale` on 9 shapes incl. padded;
   inverse round-trip exact.
4. **Front-end A/SFA dump** — `diag_frontend_dump.py`: after a real
   dynamic-kernel run (m=64, K=5120, n=576, fast_math=False compiled —
   confirmed via kernel cache key), the kernel's `packed_input` payload
   and `packed_input_scale` SFA plane were read back and compared to the
   host `quantize_block_fp4` model: **SF bytes 0/40960 mismatches**
   (SFA uses the same 128×4 atom swizzle as SFB — §1.3 formula),
   payload nibbles exact except 0.16% one-step flips at exact dyadic
   threshold ties (and value-neutral `-0.0` vs `+0.0` encodings).
5. **Grid-exact MMA probe** — `probe_grid_exact.py`: unit scales,
   FP4-grid inputs/weights ⇒ operands bit-identical; dynamic kernel ==
   oracle to 2 bf16 ulps (micro: 1 ulp).
6. **Scale perturbation** — cases D/E/F: outputs track
   `alpha = ws2×is`, `a_gscale = 1/is` across ×4 perturbations; case E is
   invariant to within the base-case residual, proving the reciprocal
   folding empirically.
7. **swap_ab** — n=576 (swap_ab True, verified in the compiled cache-key
   geometry rule) and n=1152/2304 (False) both pass with byte-identical
   packing; swap_ab changes no published bytes.

Exact commands (all from `/home/tj/Developer/ds41rt`):
```
./scripts/run-with-python-env.sh python runs/v7q-a1/test_nvfp4_layout.py
./scripts/run-with-python-env.sh python runs/v7q-a1/test_full_geometry.py
./scripts/run-with-python-env.sh python runs/v7q-a1/diag_precision.py
./scripts/run-with-python-env.sh python runs/v7q-a1/diag_micro_vs_dynamic.py
./scripts/run-with-python-env.sh python runs/v7q-a1/diag_frontend_dump.py
./scripts/run-with-python-env.sh python runs/v7q-a1/probe_grid_exact.py
./scripts/run-with-python-env.sh python runs/v7q-a1/probe_mutual_and_sensitivity.py
# swizzle unit checks: inline python (see section "host swizzle check" in session log)
```

---

## 5. Remaining unknowns / risks

1. **`up_proj.weight_scale_2` vs `gate_proj.weight_scale_2`**: the fused
   w13 has a SINGLE `alpha` applied to both gate and up rows. The b12x
   benchmark loader uses `gate`'s `weight_scale_2` for the whole fused
   tensor. If the real checkpoint's `up.weight_scale_2` differs from
   `gate`'s, the up projection carries a systematic
   `ws2_gate/ws2_up` error. Fix if needed: fold the ratio into the up
   block scales, `sfb_up' = e4m3(sfb_up × ws2_up/ws2_gate)` (lossy, ≤1 e4m3
   ulp) — not required if the checkpoint calibrates them equal (typical).
   Not verified against the real checkpoint (not available locally).
2. **FP4-tie rounding of the activation/intermediate quantizer**:
   0.16% of nibbles land on exact dyadic threshold ties and may flip one
   FP4 step vs the slow-path host model (kernel compiled
   `fast_math=False`; production default `fast_math=True` additionally
   uses `rcp.approx` + `cvt_e2m1x8` hardware rounding — same value domain,
   slightly different tie/ulp behavior). Value-bounded, affects only the
   last FP4 ulp of individual elements; irrelevant to layout correctness
   but means a bit-exact host model of the *activations* is not practical.
   The *weight* path is byte-exact (no runtime quantization of weights).
3. **Subnormal/zero E4M3 weight scales** are passed through as-is; the
   kernel's quantizer maps zero scale → zero payload for activations.
   NaN (0x7F) bytes are never produced by the packer and must not appear.
4. **Deterministic vs atomic output**: tests ran the default policy
   (atomic bf16 scatter or deterministic per-route reduce at the policy's
   choice). The weight layout is identical for both; output ordering
   differences are absorbed in the reported tolerances (run-to-run
   envelope ≪ the reported residuals). **AOT integration contract:** the
   NVFP4 exporter forces `deterministic_output=True`, so the compiled
   dynamic entry publishes BF16 **`[tokens, topk, hidden]` per-route rows**
   through slot 41, not token sums. The public b12x `run` wrapper performs
   a separate `_launch_dynamic_topk_sum` after that entry (see `_impl.py`,
   deterministic branch following `_launch_dynamic`); exporting only the
   dynamic kernel does not export that reduction. The engine must reduce
   those BF16 routes explicitly and must reserve all `tokens*topk` rows.
5. **`input_scales_static=True`** was used (the serving contract);
   mutable-scale refresh writes the same `a1/a2` values into scratch —
   no layout impact.
6. Only `silu` was exercised (production recipe; `silu_v41` is MXFP4-only
   and uses a different source format `fp4_e8m0_k32`).
