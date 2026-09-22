# GPU target-sampler design (implementation gate)

**Status:** design gate. No kernel has been written. Implementation follows this
document; an adversarial reviewer signs off *before* any kernel is written and
again per staged chunk (§14).

**Repository revision read:** `e1f5d495b5a82fb7ddad8514cad419b6ae62c0cc` (`e1f5d49`).

**Deliverable rule:** this task produces exactly this one file. No code, config,
cluster, hardware or git state was touched. Every claim about existing code
carries a `path:line` anchor; every number is labelled MEASURED (with its
artifact), COMPUTED, ARITHMETIC or UNKNOWN.

**Inputs read**

| Input | Actual path in this checkout | State |
| --- | --- | --- |
| Kernel/build inventory | `runs/sampling-gpu-recon-2026-09-22.md` (the task spelled it `runs/gpu-sampling-recon-2026-09-22.md`; that path does not exist) | being corrected; corrected facts in §2 override it |
| Serving-path trace | `runs/gpu-sampling-recon-paths-2026-09-22.md` | accurate as-is |
| Sampler contract | `docs/gpu-sampler-contract-v11.md` | being corrected; corrected facts in §2 override it |
| Phase-0 baseline | `runs/gpu-sampling-phase0/REPORT.md` + `runs/gpu-sampling-phase0/out/`, `logits/`, `harness/` | being corrected; corrected facts in §2 override it |
| Product code | `rust/crates/ds41rt-core/src/target_sampling.rs`, `rust/crates/ds41rt-daemon/src/v41_native_serve/{scheduler.rs,scheduler/independent.rs,scheduler/layout.rs,scores.rs,constraints.rs,prefix.rs,speculative.rs}`, `rust/crates/ds41rt-daemon/src/{v41_target_head.rs,v41_target_pass.rs,v41_memory.rs,v41_memory/download.rs}`, `native/cuda/kernels/sampling.cu`, `native/cuda/kernels/v41_dspark.cu`, `native/CMakeLists.txt`, `native/tests/cuda_selftest.cc` | — |
| Reference headers | `.venv/lib/python3.12/site-packages/flashinfer/data/include/flashinfer/{sampling.cuh,air_top_p.cuh,topk.cuh}` | found via `flashinfer.__file__`; not a repo dependency (§3) |

`runs/` is gitignored (`.gitignore:48`), so every artifact this design relies on
must be re-expressed as a committed document, a committed hash, or an in-repo
test before it can gate anything (§15, risk R10).

### 0.1 Anchor shorthand

Anchors after §0 use the basenames below; all resolve under
`/home/tj/Developer/ds41rt`:

| Shorthand | Full path |
| --- | --- |
| `target_sampling.rs` | `rust/crates/ds41rt-core/src/target_sampling.rs` |
| `dspark_verify.rs` | `rust/crates/ds41rt-core/src/dspark_verify.rs` |
| `dspark_rng.rs` | `rust/crates/ds41rt-core/src/dspark_rng.rs` |
| `scheduler.rs` | `rust/crates/ds41rt-daemon/src/v41_native_serve/scheduler.rs` |
| `independent.rs` | `rust/crates/ds41rt-daemon/src/v41_native_serve/scheduler/independent.rs` |
| `layout.rs` | `rust/crates/ds41rt-daemon/src/v41_native_serve/scheduler/layout.rs` |
| `scores.rs` | `rust/crates/ds41rt-daemon/src/v41_native_serve/scores.rs` |
| `constraints.rs` | `rust/crates/ds41rt-daemon/src/v41_native_serve/constraints.rs` |
| `prefix.rs` | `rust/crates/ds41rt-daemon/src/v41_native_serve/prefix.rs` |
| `speculative.rs` | `rust/crates/ds41rt-daemon/src/v41_native_serve/speculative.rs` |
| `v41_target_head.rs` | `rust/crates/ds41rt-daemon/src/v41_target_head.rs` |
| `v41_target_pass.rs` | `rust/crates/ds41rt-daemon/src/v41_target_pass.rs` |
| `v41_memory.rs` | `rust/crates/ds41rt-daemon/src/v41_memory.rs` |
| `download.rs` | `rust/crates/ds41rt-daemon/src/v41_memory/download.rs` |
| `v41_memory/download.rs` | `rust/crates/ds41rt-daemon/src/v41_memory/download.rs` |
| `v41_dspark.cu` | `native/cuda/kernels/v41_dspark.cu` |
| `sampling.cu` | `native/cuda/kernels/sampling.cu` |
| `common.h` | `native/cuda/kernels/common.h` |
| `native_v41.rs` | `rust/crates/ds41rt-api/src/native_v41.rs` |
| `upstream_native_v41.rs` | `rust/crates/ds41rt-api/src/tests/upstream_native_v41.rs` |
| `lib.rs` | `rust/crates/ds41rt-ffi/src/lib.rs` |
| `v41_native_serve.rs` | `rust/crates/ds41rt-daemon/src/v41_native_serve.rs` |
| `cli.rs` | `rust/crates/ds41rt-daemon/src/cli.rs` |
| `commands/real_full/constraint.rs` | `rust/crates/ds41rt-daemon/src/commands/real_full/constraint.rs` |
| `REPORT.md` | `runs/gpu-sampling-phase0/REPORT.md` |
| `harness/Cargo.toml` | `runs/gpu-sampling-phase0/harness/Cargo.toml` |
| `sampling.cuh`, `air_top_p.cuh`, `topk.cuh` | `.venv/lib/python3.12/site-packages/flashinfer/data/include/flashinfer/` (resolved via `flashinfer.__file__`) |
| `ds41rt_native.cc` | `native/src/ds41rt_native.cc` |
| `cuda_selftest.cc` | `native/tests/cuda_selftest.cc` |

---

## 1. Scope, goals, explicit non-goals

### 1.1 Goal

Move the production target-token selection for **normal and constrained
decoding** from the CPU host sampler onto the GPU, for the `serve-native` path
only (`rust/crates/ds41rt-daemon/src/cli.rs:26`), with:

1. **No greedy throughput regression.** The compact greedy lane keeps the exact
   device argmax path (`rust/crates/ds41rt-daemon/src/v41_target_head.rs:350-377`,
   `rust/crates/ds41rt-daemon/src/v41_native_serve/scheduler.rs:458-462`,
   `.../scheduler/independent.rs:131-135`).
2. **No correctness regression.** The device sampler implements the frozen filter
   chain of `rust/crates/ds41rt-core/src/target_sampling.rs:9-25` and the
   verifier contract of `docs/gpu-sampler-contract-v11.md` §3.
3. **No full-vocabulary host materialization on the hot path.** The
   `rows × 517,120 B` D2H (`scheduler.rs:401-403`,
   `independent.rs:143` → `v41_memory/download.rs:17-42`) and the per-row
   517 KB `Vec<f32>` (`.../scores.rs:11-17`, `:47`, `:119`) leave the hot path.
4. A release and README measurement update with an honest, falsifiable claim
   (§13, §14 chunk 7).

### 1.2 Explicit non-goals

- **Full-draft-distribution rejection correction is OUT of scope.** v11 ships
  exact speculative-sampling rejection correction with the draft as a point mass
  (`q = δ`), exact for any draft and temperature
  (`rust/crates/ds41rt-core/src/dspark_verify.rs:18-52`,
  `scheduler.rs:543-564`, `docs/release-v11-performance.md:228-249`). This design
  changes *how the target sample is produced*, not the verifier. The
  full-draft-distribution variant stays the recorded future item
  (`docs/release-v11-performance.md:220-230`).
- **No Python in the serving process.** The daemon loads C ABI symbols by name
  through `libloading` (`rust/crates/ds41rt-ffi/src/lib.rs:49`), and the native
  library is C++/CUDA. No new runtime dependency on Torch/Triton/cuTe-DSL.
- **The grammar matcher stays on the CPU.** `fill_bitmask` is invoked against a
  CPU `DLTensor` (`native/src/xgrammar_adapter.cc:349-352`); only mask
  *application* moves to the device.
- **No full-vocabulary sort anywhere**, on any path (§4).
- **Not a change to draft (dSpark) proposal sampling.** The draft is greedy in
  production (`speculative.rs:511-512`, `:613` passes `&vec![0.0; count]`).
- **No change to the CPU sampler's semantics or to the legal values of
  `TargetSamplingParams`** (`target_sampling.rs:108-134`).
- Not a new `real_full` path: the legacy GPU sampler
  (`native/cuda/kernels/sampling.cu:598-676`, `:1712-1751`) is not promoted
  (§3.4).

---

## 2. Corrected evidence summary and honest expected uplift

### 2.1 Corrections that override the artifacts

These are the facts this design builds on. Each names the artifact claim it
replaces.

| # | Corrected fact | Artifact claim it overrides | Consequence for the design |
| --- | --- | --- | --- |
| 1 | `air_top_p.cuh` does **not** use cooperative grid sync and **is graph-capturable**; its cost is multi-pass launches plus a workspace. Verified: `grep -n 'cooperative\|grid.sync\|this_grid' .venv/.../include/flashinfer/air_top_p.cuh` returns nothing; the header's only kernels are `AirTopPRenormRadixKernel` (`air_top_p.cuh:284`), `AirTopPRenormInitKernel` (`:407`), `AirTopPRenormApplyKernel` (`:434`) | `runs/sampling-gpu-recon-2026-09-22.md:110`, `:315`, `:448` say grid-cooperative / capture-legal-unknown | The radix-histogram mechanism is a legal fallback for §4's pass-budget problem; no cooperative-launch requirement anywhere in the sampler |
| 2 | There **is** in-kernel Philox in the draft path: `curand_init(..., &state)` at `native/cuda/kernels/v41_dspark.cu:249` feeding `curand(&state)` at `:263` (Gumbel-max at `:262-265`), unused at `T = 0` because the branch is guarded by `if (temperature != 0)` (`:246`, `:262`) and production passes `0.0` (`speculative.rs:511-512`) | `runs/sampling-gpu-recon-2026-09-22.md:81` ("seed + RNG absent from all GPU code"), `:113` | cuRAND/Philox is already a linked, graph-captured dependency; but only the **target** sampler lacks an on-device draw. §6 ports SplitMix64 rather than reusing the draft stream |
| 3 | FlashInfer `min_p`/top-k/top-p are **reference mechanisms, not semantic equivalents**: (a) min_p compares *probability* `p >= max_val * p` (`sampling.cuh:1137`) while ds41rt compares *logit* `scaled >= max_scaled + ln(min_p)` (`target_sampling.rs:449-457`); (b) top-k accepts when `aggregate_gt_pivot_0.count < k` (`sampling.cuh:945`), keeping every token equal to the k-th value (vLLM rule), while ds41rt keeps exactly k with the lowest id (`target_sampling.rs:30-33`, `:477-501`); (c) top-p accepts `aggregate_gt_pivot_0 < top_p` (`sampling.cuh:1072`), a strict tail-mass criterion **excluding** the boundary tie, while ds41rt is the inclusive descending prefix reaching `top_p` (`target_sampling.rs:544-554`) | `runs/sampling-gpu-recon-2026-09-22.md:106` calls min_p "exactly ds41rt's semantics"; `:294-297`, `:316-320` treat ties as a caveat only | Ports must implement **our** rules and be oracle-tested at boundaries (§4.4, §4.5, §12) |
| 4 | A logits row is **517,120 B ≈ 0.52 MB** (`129,280 × 4`); the "3.1 MB" figure was a 6-row batch | `docs/gpu-sampler-contract-v11.md:447` ("≈ 3.1 MB/row"; that file is being corrected concurrently — the claim is contract §4.4) | All bandwidth arithmetic in §4 is per *row*, not per *batch* |
| 5 | The recoverable amount is **0.55–1.79 ms/token** (host sampler + D2H/argmax/materialization) on a measured 10.5–12.3 ms/token, i.e. ceilings of **+5.1% to +16.9%** on the four stochastic profiles, greedy untouched. These ceilings are **OPTIMISTIC**: they assume a zero-cost sampler, part of them is derived from cross-host microbench numbers, and greedy's 10.517 ms/token is not a clean shared device floor because lower acceptance means more target-forward passes per emitted token (greedy accepted fraction 0.724 vs 0.326–0.548) | none; this is the correction of the phase-0 report's own framing | §2.3 states the ceilings with caveats; §13 discounts top_p0.95 hardest; §14 chunk 6 exists specifically to cut pass cost if the kernel is not near-zero-cost |
| 6 | Term (b) accept/verify is identifiable **only as a residual**; separating acceptance-driven device time needs per-round timing on the TIMED campaign | `runs/gpu-sampling-phase0/REPORT.md:244-255` already says this; the corrected fact forbids treating it as measured | §13 requires per-round timing in the timed campaign before any term-(b) claim |
| 7 | The weak batch-layout unit test does not pin batch independence; the real pin is `scheduler.rs:666-691` | `target_sampling.rs:998-1007` is self-referential (it re-derives the same hash twice) | §12 uses the scheduler test as the batch-independence pin and adds a GPU analogue |

### 2.2 MEASURED baseline in one place

From `runs/gpu-sampling-phase0/REPORT.md` §4.3 (`attempt-03/campaign-aggregate.json`)
and §5 (`out/cost-decomposition.json`):

| Profile | ms/token (MEASURED) | (a) pure sampler ms/token | (a′) D2H+argmax+materialize | host path a+a′ | ceiling tok/s | measured tok/s | uplift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| greedy | 10.517 | 0 | 0 | 0 | — | 95.08 | — |
| temp0.7 + top_k40 | 11.434 | 0.462 | 0.130 | 0.592 | 92.24 | 87.46 | +5.5% |
| temp0.7 + min_p0.05 | 11.250 | 0.381 | 0.164 | 0.545 | 93.42 | 88.89 | +5.1% |
| temp0.7 + top_p0.9 | 12.004 | 1.221 | 0.166 | 1.388 | 94.19 | 83.30 | +13.1% |
| temp0.2 + top_p0.95 | 12.296 | 1.464 | 0.322 | 1.787 | 95.08 | 81.33 | +16.9% |

Ceiling = `min(gap, a+a′)` with a *zero-cost* GPU sampler, unchanged output and
unchanged acceptance (REPORT §5). Per-row CPU cost is MEASURED at
77.7 µs greedy, 379.6 µs top_k40, 291.9 µs min_p0.05, 1062.7 µs top_p0.9,
1335.2 µs top_p0.95 on the `peaked` row (`out/latency.json`).

### 2.3 Honest reading of the uplift (must be quoted in the release)

- The four ceilings are **upper bounds**, not forecasts:
  (i) a real kernel pays launches, full-vocabulary reductions and the same
  per-step sync;
  (ii) term (b) is 0–35% of the gap and is untouchable by a faster sampler
  (`REPORT.md:244-255`);
  (iii) the top_p0.95 host-path model overshoots its own measured gap by
  ~0.5–2%, so that ceiling is the least trustworthy and is discounted hardest.
- Greedy's 10.517 ms/token is **not** a clean shared device floor: acceptance is
  0.724 for greedy versus 0.326–0.548 in the instrumented pass
  (`REPORT.md:253-255`), so stochastic profiles issue more target-forward passes
  per emitted token. Term (b) is a residual, not a measurement (corrected fact 6).
- The planning range is therefore **the measured host path 0.55–1.79 ms/token,
  discounted by the measured kernel cost and by whatever term (b) turns out to
  be**. A defensible public claim is a *floor*: "at least X% on profile P, with
  workload identity and ≥3 repeats", where X is set from the campaign, not from
  this table.
- Cross-host contamination: part of the per-row cost in the ceilings came from
  microbench numbers measured on a different host than the campaign
  (`REPORT.md:100-104`, §4.4), so the split between (a) and (a′) is a model.

---

## 3. Architecture and placement decision

### 3.1 Decision

Implement in this repository, in a **new file**
`native/cuda/kernels/v41_sampling_gpu.cu` (plus an optional
`native/cuda/kernels/v41_sampling_gpu.cuh` if shared device helpers are needed),
added to `DS41RT_NATIVE_SOURCES` next to `cuda/kernels/sampling.cu`
(`native/CMakeLists.txt:282`), exposing new `extern "C"` `_async` entry points in
the existing C ABI (`native/include/ds41rt_native.h`), with Rust wrappers and
validators in `rust/crates/ds41rt-ffi/src/lib.rs` following the
`validate_logits_sample_topk_topp_buffers` pattern (`:17859-17896`).

### 3.2 FlashInfer is a reference, never a dependency

The algorithms are ported from the installed headers
(`.venv/lib/python3.12/site-packages/flashinfer/data/include/flashinfer/sampling.cuh`,
`air_top_p.cuh`, `topk.cuh`). They are **not** a build dependency:

- FlashInfer is not a submodule (`.gitmodules` lists only `sparkinfer`,
  `xgrammar`, `gptqmodel`, `transformers`) and has no lock file
  (`third_party/` holds only `sparkinfer.lock.json` and `xgrammar.lock.json`).
- The tree is an artifact of `.venv`; only `scripts/kernel-cache-identity.py`
  hashes it (`:159-170`), and that hash is a cache identity, not a source pin.
- The existing consumer of the FlashInfer tree
  (`native/CMakeLists.txt:291-320`, `cuda/kernels/packed_fp8_mla_exact.cu`) uses
  `find_package(Python3)` + an import of `flashinfer` to locate headers, so it is
  gated behind `DS41RT_BUILD_FLASHINFER_PACKED_MLA`. We explicitly do **not**
  extend that pattern: the new file is self-contained CUDA C++.
- Provenance practice: the design's algorithm citations are the line anchors in
  this document. A header comment in the new `.cu` must name each ported
  mechanism and its FlashInfer anchor, matching the way `sampling.cu` cites
  TRT-LLM (`air_top_p.cuh:16-18`). If a future maintainer wants a pinned
  dependency, the correct move is a vendored `.cuh` plus a lock entry, not a
  `.venv` lookup.

### 3.3 Why not sparkinfer/b12x now

- **No samplers exist there.** The `b12x` op catalog
  (`third_party/sparkinfer/b12x/preparation/catalog.py:24-71`) has attention,
  GEMM, MoE, norm, quantization, sequence and comm ops; a grep for
  `sampl|top_p|top_k|min_p|philox` finds only unrelated calibration/argmax hits
  (`runs/sampling-gpu-recon-2026-09-22.md:149-154`). The closest op is
  `b12x/comm/pcie/pcie_vocab_argmax.py`.
- **Cost**: adding an op there is ~5–8 Python files plus a catalog entry, a
  tuning/qualification profile and a wheel/CI change
  (`runs/sampling-gpu-recon-2026-09-22.md:164-181`), and ds41rt consumes b12x
  only through AOT-exported `.o`/`.h` (`native/CMakeLists.txt:322-…`,
  `native/cuda/kernels/b12x_direct.cu`), so a new AOT exporter is also needed.
- **Constraint**: `serve-native` must not load Python, and b12x is
  Torch/CuTe-DSL; a sampler there would have to be AOT-packaged and re-pinned in
  `third_party/sparkinfer.lock.json` (revision + `source_tree_sha256`).
- **Precise evidence that would flip this**: (1) a decision that a non-ds41rt
  engine (the `b12x/integration/vllm` path) is a first-class consumer of the
  sampler; (2) a requirement that the sampler run inside a Spark-side (SM121)
  fused head, making it part of expert AOT packaging anyway; (3) a *measured*
  build/release finding that a new native kernel cannot be absorbed while a
  b12x wheel bump can — with build-time and release-process numbers, not
  assumed; (4) a *measured* speedup of a CuTe-DSL implementation over
  hand-written CUDA at `vocab = 129,280`, rows 1–80, that exceeds its packaging
  cost. None of these hold today.

### 3.4 Why not reuse the legacy GPU sampler

`native/cuda/kernels/sampling.cu` cannot express the served contract:
temperature/top-k(≤64)/top-p only, mandatory finite `top_k`
(`sampling.cu:598-606`, validator `lib.rs:17880-17883`), top-p computed *inside*
the top-k set (`sampling.cu:651-661`, `:74-95`), no `min_p` anywhere, no
`(seed, position)` draw (it takes `const float* random_uniforms`,
`sampling.cu:598`), raw-logit ranking instead of scale-first filtering
(`sampling.cu:623-641`), and `out_scores` divided by `nucleus_mass`
(`sampling.cu:675`) which is a different diagnostic scale. It is reachable only
from the legacy `real_full` command, not `serve-native`. **Do not promote it.**

### 3.5 Upstreaming later

A stabilized kernel may be contributed back as a plain `.cuh`/`.cu` once (a) the
oracle/boundary suite in §12 is green in this repo and (b) the chosen rules are
accepted as general (notably: our top-k tie rule is a deliberate deviation from
vLLM/FlashInfer, `target_sampling.rs:30-33`, so an upstream contribution must
expose it as an option). Upstreaming is a *follow-on*, not part of this plan.

---

## 4. Kernel decomposition

### 4.0 Shared invariants

- **One CTA per row**, 256 threads (`kBlock`, `native/cuda/kernels/common.h:16`);
  `gridDim.x = rows ≤ 80` (`v41_target_head.rs:124-129`,
  `v41_target_pass.rs:266-276`). All cross-thread communication is
  `__syncthreads()` + shared memory; **no grid-wide sync, no cooperative launch**
  (not needed — every stage is per-row independent).
- **Mask is applied first, as a predicate, without mutating `b(4)`.** No filter
  ever sees a masked-out token: `survivor(t) = allowed(t) && scaled(t) >=
  min_scaled` (`target_sampling.rs:456-458`). Masked logits are never read, so a
  masked NaN is legal (`target_sampling.rs:1009-1024`). The logits buffer stays
  byte-identical, which is required because the retained frontier is downloaded
  from it later (`scores.rs:80-90`). This is why the design does **not** copy
  `apply_token_bitmask_f32_candidate_kernel`'s scratch-copy approach
  (`sampling.cu:678-691`) for the hot path.
- **Temperature is a multiply by the reciprocal**: `inv = 1.0f / temperature`,
  `scaled = logit * inv` (`target_sampling.rs:427`, `:438`, `:457`, `:485`,
  `:507`). Never `logit / temperature`.
- **Reductions have a fixed, row-independent shape**: contiguous thread
  segments + fixed binary-tree combine. No atomics for floating-point. This is
  what makes claim (b) of §6 hold.
- **Device compilation must not enable `-use_fast_math` or FTZ.** The existing
  tree applies `-use_fast_math` to exactly one source
  (`native/CMakeLists.txt:308-310`, `packed_fp8_mla_exact.cu`); the new file must
  not join it, or subnormal scaled values (the adversarial
  `f32::from_bits(1)` row, `target_sampling.rs:1481`) and `expf`/`logf` results
  would differ from the host beyond the residual declared in §6.3. Set an
  explicit `COMPILE_OPTIONS "-O3"` (no fast-math, no `--use_fast_math`) and add a
  test that a subnormal-scaled row is handled bit-identically to the host.
- **Colour-coded provenance**: `[PORT]` = mechanism ported (anchor given),
  `[NEW]` = new code required to meet ds41rt's rules.

### 4.1 K1 — `v41_sample_prepare_kernel`: mask, finiteness, scaled max, survivor count, greedy

- **Inputs**: `logits` row `r` (`vocab` f32), `params[r]` (temperature, `top_k`,
  `min_p`, host-precomputed `ln_min_p`, mask row, flags), `mask_words`
  (nullable).
- **Outputs**: `scratch[r] = {max_scaled, inv_temperature, allowed_count,
  survivor_count, nonfinite_token, status, status_detail}`; for greedy rows also
  `out_indices[r]`, `out_scores[r]` (the raw max logit, so `from_greedy`'s
  finite-score check still holds, `scores.rs:72-77`).
- **Algorithm**
  1. **Resolve greedy FIRST**, before any reciprocal:
     `greedy = (temperature < GREEDY_TEMPERATURE_EPS) || (top_k == 1)`
     (`target_sampling.rs:172-174`), preferring the host `flags` bit but
     re-deriving it. A greedy row never computes `inv = 1.0f / temperature` and
     never evaluates `expf` or the draw. `temperature = 0.0` is the *normal*
     greedy case (`TargetSamplingParams::greedy`, `:96-104`), so computing `1/T`
     first would produce `inf`/`NaN` and cascade into `max_scaled`; the CPU
     short-circuits greedy at `:234-236` before any scaling, and the device must
     do the same.
  2. **Greedy branch**: masked argmax with strict `>` and lowest-id-wins
     (CPU `:259-262`; device form as `sampling.cu:568,583-587`). Its finiteness
     check is **strict over the whole row, independent of the mask**, because
     `scores.rs::argmax` runs `ensure!(value.is_finite())` *before* the mask test
     (`scores.rs:160-163`) — a greedy/constrained row must reject a non-finite
     logit even when that token is masked out. Record the lowest offending token
     id over all tokens.
  3. **Stochastic branch**: compute `inv = 1.0f / temperature`; per-thread scan
     over tokens with `allowed(t)` only; a masked non-finite value is **not**
     read and not an error, matching `target_sampling.rs:430-436` and the
     `non_finite_allowed_logits_are_rejected` / masked-NaN pair (`:1009-1024`).
     Record the lowest offending *allowed* id, `max_scaled` and `allowed_count`.
  4. Block reductions in fixed tree order for `(max_scaled, argmax score/id,
     allowed_count, nonfinite flag+min id)`.
  5. Thread 0 writes the status with **CPU error precedence**: non-finite →
     `NONFINITE_LOGIT`; else `allowed_count == 0` → `EMPTY_CANDIDATES`; else
     `!isfinite(max_scaled)` → `INVALID_TEMPERATURE` (`target_sampling.rs:434-448`).
     `EMPTY_CANDIDATES` must beat the `-inf` maximum error.
  6. `min_scaled = max_scaled + params[r].ln_min_p` when `min_p > 0`, else
     `-inf` (`:449-454`). **`ln_min_p` is host-precomputed per row** (see §5.1
     and §6.3a) so the device performs exactly one f32 add identical to the
     host; the device never calls `logf`.
  7. Second in-CTA pass (stochastic rows only) counts `survivor_count` and,
     when `top_k` is enabled, whether `top_k >= survivor_count` (the no-op case,
     `:477-501`).
- **Masked non-finite discipline is mode-specific and pinned** (review item 7):
  - greedy rows (including constrained greedy): strict over all tokens, exactly
    as `scores.rs::argmax` (`:160-163`);
  - stochastic rows: permissive on masked tokens, exactly as
    `target_sampling.rs:1009-1024`.
  - **Recorded behaviour delta:** today's daemon also rejects a masked
    non-finite value on a stochastic lane, because `BatchScores::new`
    pre-validates every materialized row with `argmax(row, None)`
    (`scores.rs:67-71`, `:160-163`) before `sample` ever runs. Adopting the
    permissive stochastic rule therefore *loosens* that one case relative to the
    daemon's incidental behaviour. It is a deliberate, documented change (release
    note + open decision D9), it never loosens the greedy/constrained path, and
    both behaviours are pinned by §12.6. If the reviewer prefers zero observable
    change, set the per-row strict bit for stochastic rows too; the flag exists
    precisely to make that a one-line host choice.
- **Shapes/limits**: `vocab ≥ 1`; `vocab` discovered from the buffer, never
  hard-coded (contract §1; `scores.rs:7` is the checkpoint constant, not the
  interface). `top_k ∈ {0=disabled, 1..=vocab, >vocab=no-op}`.
- **Workspace**: per row: 4 f32 + 6 u32 in `scratch`; 256×(4+4+4) B shared.
- **Launch**: `grid = rows`, `block = 256`, 2 row reads (greedy: 2 reads for the
  strict scan + argmax reduction, no second pass).
- **Provenance**: `[PORT]` of the argmax/scan structure of
  `sampling.cu:547-596` extended with scaling, min_p and the ds41rt status rules;
  the "scale-then-compare" order is `[NEW]` relative to FlashInfer (corrected
  fact 3a). The host-`ln` decision is `[NEW]`.
- **Greedy fast exit**: when greedy, K1 is the only kernel that runs for that
  row, and it is byte-equivalent to `argmax_allowed`
  (`target_sampling.rs:246-265`) *including* the stricter whole-row finiteness
  check that the daemon's `argmax` applies.

### 4.2 K2 — `v41_sample_categorical_kernel`: fast path (no top_k, `top_p ≥ 1`)

Selected exactly when `top_k.is_none() && top_p >= 1.0`
(`target_sampling.rs:463-466`). This is the one branch whose **draw order is
ascending token id** (`sample_categorical`, `:596-618`), unlike the ordered
path.

- **Inputs**: row logits, `max_scaled`, `min_scaled`, `inv`, `params[r].seed`,
  `params[r].position`, mask.
- **Algorithm**
  1. Two-pass within the CTA: (i) fixed-order sum of `w_t = expf(scaled_t -
     max_scaled)` over survivors → `total`; (ii) `total = fmaxf(total, 1e-20f)`,
     `target = clamp(u, 0, MAX_UNIFORM) * total`.
  2. **Ascending-id inclusive-prefix scan** `[NEW]`: partition `[0, vocab)` into
     256 contiguous segments; thread `t` sequentially prefix-sums its segment in
     ascending token order; block-exclusive-scan the segment totals (fixed tree);
     each thread walks its segment from its exclusive offset and reports the
     first token (if any) whose inclusive cumulative satisfies
     `target <= cumulative`; the answer is the **minimum token id** over
     reporting threads.
     Because weights are non-negative and the comparison is inclusive, **more
     than one segment can report** (every segment after the first crossing
     reports its first token). The minimum over reported ids is still exactly the
     first crossing token, because the earliest reporting segment contains it.
     Do not implement this as "exactly one segment crosses".
  3. **No-crossing fallback `[NEW]`**: if no thread reports, select the **last
     survivor in ascending token order** (the CPU's `last` fallback,
     `target_sampling.rs:606-618`). This is defensive rather than expected — the
     best survivor has `w = 1`, so `total >= 1` and
     `target <= 0.99999994 * total < total` — but the CPU carries the fallback
     and the device must mirror it so that any future change to the clamp or the
     floor cannot produce a silently different token.
  4. Write `out_indices[r]` and `out_total[r] = total`.
- **Why contiguous segments**: they preserve the CPU's ascending accumulation
  order *within* a segment; only the cross-segment combine uses a tree. Together
  with the `expf` difference (§6.3b) this is one of the two accumulation-order
  sources of the residual acknowledged in §6.3c; the *option* of a strictly
  sequential combine that would remove the K2 part is recorded in §6.5.3 and is
  evaluated in chunk 2.
- **Limits**: `top_p >= 1.0` and `top_k` disabled only; `MAX_UNIFORM` is
  `0x3F7FFFFF` written as a bit pattern, not a decimal literal.
- **Passes**: 2 reads.
- **Provenance**: `[PORT]` of `DeviceSamplingFromProb`'s "sum then inclusive
  scan then lowest crossing id" structure (`sampling.cuh:567-642`), with
  `[NEW]` contiguous-segment ordering to match ds41rt's token-order walk and
  avoid a `BlockAdjacentDifference` setup.

### 4.3 K3 — `v41_sample_topk_pivot_kernel`: general-k, exact-k, sorting-free

Runs only when `top_k ∈ 1..vocab` and `top_k < survivor_count` (otherwise
`:499-501` is a no-op).

- **Order primitive `[NEW]`**: `order_key(x)` = standard IEEE ordered u32
  (`bits ^ 0x80000000` for non-negative, `~bits` for negative) with `-0.0`
  canonicalized to `+0.0`. Larger = better. This is the bitwise complement of
  `target_sampling.rs:335-344`'s `descending_radix_key`; using the *ascending*
  form makes "accept `C_gt(v) < k`" read naturally. The full total order for the
  tie rule is the u64 key `(order_key(scaled) << 32) | ~id`, i.e. larger is
  better and ties go to the **lowest id**, matching `sampling.cu:28-33` and
  `target_sampling.rs:283-291`.
- **Algorithm** (`[PORT]` of FlashInfer's count-pivot acceptance,
  `sampling.cuh:837-965`, especially `:930-940`)
  1. Bisect on `order_key` over `[key(-inf), key(+inf)]`; each probe `v` is one
     pass computing `C_gt(v) = #{survivors : order_key(scaled) > v}`. Two probes
     `v0 < v1` per pass (the FlashInfer dual-pivot trick) form a **ternary**
     search: each pass keeps one of three sub-intervals, so the pass count is
     `~log_3` of the value range rather than `~log_2` — about 1.6× fewer passes
     than one-probe bisection. (This is *not* a halving per pass; only one probe
     per pass halves.)
     Accept `v` when `C_gt(v) < k`; the k-th value is the **smallest** accepted
     key, so a satisfying probe moves the upper bound down (Appendix A.1).
  2. Stop when the interval has ≤ 1 representable f32 value or after
     `MAX_PIVOT_STEPS = 32` passes (defensive cap; report `INTERNAL` if hit).
     Write `kth_value = ordered_bits_to_float(v)`, `above_count = C_gt(v)`.
  3. **ds41rt tie rule `[NEW]`** (this is the deliberate deviation documented at
     `target_sampling.rs:30-33`): membership is
     `{order_key > kth_key} ∪ {the lowest-id `k - above_count` tokens with
     `order_key == kth_key`}`. K4 resolves the tie prefix; K3 only finds the
     value and the strict count.
- **All-tied row**: the bisection converges immediately to the shared value with
  `above_count = 0`; the tie cut then admits ids `0..k`, which is exactly
  "lowest id wins" for an all-equal row.
- **Limits**: `k` up to `vocab`; `k ≥ survivor_count` never enters K3. No
  compile-time k cap (contrast `kMaxSampleTopK = 64`, `common.h:19`, and
  `kParallelSampleTopK = 8`, `sampling.cu:7`).
- **Workspace**: `kth_value` f32, `above_count` u32 in `scratch`.
- **Passes**: ≤ 20 with two probes per pass (ternary search over a 32-bit key).
- **No sort**: this replaces both the `real_full`-style
  `cub::DeviceSegmentedRadixSort` over `rows × vocab` (`sampling.cu:1739-1742`)
  and the single-thread serial top-k (`sampling.cu:598-676`).

### 4.4 K4 — `v41_sample_topk_membership_kernel`: exact tie prefix

- **Inputs**: `kth_key`, `above_count`, `target_tie = k - above_count`,
  survivors predicate.
- **Algorithm** `[NEW]`: single pass over `[0, vocab)` in 256 contiguous
  segments; each thread counts `order_key == kth_key` tokens in its segment;
  block-exclusive-scan gives the number of equals before each segment; each
  thread admits equals while the running count is `< target_tie`. Writes a
  per-row bitmap `topk_bits[vocab/32]` (16,160 B/row) or, when `target_tie`
  covers the whole tie group, nothing (membership is then just
  `order_key >= kth_key`, a predicate K5 can evaluate inline — the common case).
- **Why a bitmap**: K5 must evaluate membership repeatedly during bisection, and
  recomputing the tie prefix each pass would be O(vocab) per pass. The bitmap is
  16 KB/row versus 517 KB of logits, so re-reading it is cheap.
- **Limits/edge**: works for `vocab % 32 != 0` because the bitmap is only indexed
  by in-range token ids (see §5.3).
- **Passes**: 1 write pass (+ bitmap reads in K5).

### 4.5 K5 — `v41_sample_nucleus_kernel`: inclusive-prefix top-p + rank-order draw

- **Inputs**: survivors, top-k bitmap (if K4 ran), `total`, `max_scaled`,
  `top_p`, `uniform`, `top_k` state.
- **Set `S`**: survivors ∧ top-k membership. Weights
  `w_t = expf(scaled_t - max_scaled)`; `total = Σ w_t` (fixed order);
  `p_t = w_t / fmaxf(total, 1e-20f)` — per-token division, matching
  `target_sampling.rs:539-542`. **This is where the dominant residual enters**:
  CUDA `expf` is not bit-identical to Rust `f32::exp` (glibc), so every `w_t`
  may differ by 1–2 ulp (§6.3b). The reduction order over `p_t` differs as well.
  The per-token division and the `total` floor are the same operations.
- **Boundary primitive.** `M(K) = Σ_{t ∈ S, key(t) >= K} p_t` for the u64
  total-order key `(order_key(scaled) << 32) | ~id` (larger key = better rank).
  **`M` is non-increasing in `K`**: a larger key admits fewer tokens. Every
  boundary we need is therefore the **largest key `K` with
  `M(K) >= threshold`** — equivalently *the worst key that is still inside the
  prefix* (the boundary rank itself).
  - Direction is load-bearing and was wrong in the first revision: "the worst
    key satisfying `M(K) >= top_p`" walks `K` downward toward `key(-inf)`, where
    `M` is the full mass, and selects the whole vocabulary. The bisection
    invariant is `M(lo) >= threshold > M(hi)` with the answer in `[lo, hi)`,
    starting from `lo = key(-inf)` (all tokens) and `hi = key(+inf)` (no tokens),
    and it moves `lo` **up** whenever a probe satisfies the threshold.
- **Top-p nucleus**
  - CPU rule: smallest descending prefix whose mass is `>= top_p_clamped`, at
    least one token, `top_p = clamp(top_p, 1e-6, 1.0)` (`:544-554`); afterwards
    `nucleus_mass = nucleus_mass.max(1.0e-20)` (`:555`). The `1e-20` floor is
    unreachable in practice (the largest retained `p` is positive and the nucleus
    always includes the largest-`p` token) but it is part of the contract and must
    be applied.
  - Boundary: `K_p` = **largest key with `M(K_p) >= top_p`**; the nucleus is
    `{t ∈ S : key(t) >= K_p}`.
  - Search in two levels, because a 64-bit search would be too many passes:
    1. **Value level**: ternary search on `order_key` (two probes per pass,
       ~`log_3` passes) for the value `v` where the prefix mass crosses `top_p`.
       This mirrors FlashInfer's tail-mass probe (`sampling.cuh:1072`) but the
       acceptance is ds41rt's inclusive `>=`, not FlashInfer's strict `<`.
    2. **Tie level**: within `{scaled == v}`, ternary search on id (~`log_3` of
       the id range) for the last lowest-id equal the prefix includes.
  - **No `top_p >= 1.0` special case.** The first revision claimed the nucleus is
    then "all of `S`"; that is false. The CPU loop
    `for &weight in &weights { nucleus_mass += weight; if nucleus_mass >= top_p
    { break } }` (`:548-554`) stops at the **first** prefix whose f32 running
    sum reaches the target, so when the normalized prefix sum rounds to `>= 1.0`
    before the last survivor (entirely possible once `p_t = w_t / total` is
    rounded and accumulated in f32), the CPU nucleus is a **strict prefix even
    for `top_p = 1.0`**. This is directly reachable: the served
    `temperature 0.7 + top_k 40` profile sets `top_p = 1.0` with a finite
    `top_k`, so it takes the ordered path (`target_sampling.rs:463-466` fails on
    `top_k.is_some()`), and `top_p = 1.0` is clamped to `1.0` at `:545`.
    K5 therefore **always runs the boundary search** on the ordered path.
    If *no* key satisfies `M(K) >= top_p` (possible when the normalized total
    itself rounds below `top_p`), fall back to the **full retained set `S`** —
    the CPU's "consumed every weight without breaking" case. Zero-division and
    empty-nucleus cases remain impossible because `S` always contains the best
    survivor.
  - Tie inclusivity: an exact-boundary uniform belongs to the **earlier** rank on
    the CPU (`target <= cumulative`, `:559-565`); the largest-key formulation
    yields exactly that rank.
- **Draw**: `nucleus_mass = M(K_p)`, `target = clamp(u,0,MAX_UNIFORM) *
  nucleus_mass`, then `K*` = **largest key with `M(K*) >= target`**; the token at
  `K*` is `out_indices[r]`. Because `M(K_p) = nucleus_mass >= target` and `M` is
  non-increasing, `K* <= K_p`, so `K*` is inside the nucleus by construction.
  Reuse the value/tie intervals already narrowed by the top-p search; in the
  typical case `target` is uniform on `[0, nucleus_mass)` and only a few extra
  passes are needed, but the worst case (`target` just below `nucleus_mass`) is a
  full search of the nucleus interval.
- **No-crossing fallback `[NEW]`**: the CPU initializes
  `selected = nucleus_count - 1` (`:558`) and keeps it when no rank satisfies
  `target <= cumulative`. K5 must therefore select the **worst key of the
  nucleus, `K_p`** (its last rank) when no probe satisfies `M(K) >= target`.
- **Outputs**: `out_indices[r]`, `out_total[r] = total`,
  `out_nucleus_count[r] = #{nucleus}`.
- **Limits**: nucleus up to `vocab` (the phase-0 `near_uniform` top_p0.9 cell
  reports support 116,344 of 129,280 — `REPORT.md:120`); nothing is
  materialized, so nucleus width costs passes only, not memory.
- **Passes (ordered path, uncompressed)**: K1 2 + K3 ≤20 (ternary over a 32-bit
  key) + K4 1 + K5 (top-p ≤20 value + ≤11 tie + draw ≤20 worst / typically ≤4)
  ≈ **up to ~74 row reads** worst case, ~35 typical. ARITHMETIC at 517,120 B/row:
  up to ~38 MB/row, ~1.8 GB for 48 rows. **This is the central cost risk**
  (§13.1 gate, §14 chunk 6) and the reason the ceilings in §2 are optimistic.
  Note the `temp0.7 + top_k40` profile now also pays the top-p search because
  `top_p = 1.0` is not skipped, so its +5.5% ceiling is the one most exposed to a
  pass-count overrun.

### 4.6 Fused fast paths worth having

| Row class | Kernels | Row reads | Notes |
| --- | --- | --- | --- |
| greedy (`T < 1e-5` or `top_k == 1`), unconstrained | compact lane: existing `execute_block_greedy`, sampler never invoked | 0 extra | byte-identical greedy path (`v41_target_head.rs:350-377`) |
| greedy, constrained (mixed lane) | K1 only | 2 | mask-aware argmax; no CPU argmax, no full-row D2H |
| stochastic, `top_k none && top_p >= 1` | K1 + K2 | 4 | the min_p-only and temperature-only profiles |
| stochastic, `top_k` only (`top_p = 1.0`) | K1 + K3 + K4 + K5; the top-p search **still runs** (§4.5) | ~60 | the served `top_k40` profile; strict-prefix overshoot → boundary search, else full `S` |
| stochastic, `top_p < 1` only | K1 + K5 | ~45 | top_p0.9/0.95 profiles |
| stochastic, `min_p` only | K1 + K2 | 4 | min_p is folded into the survivor predicate, no extra pass |
| all-greedy lane | untouched compact path | 0 | §8.4 |

### 4.7 No full-vocabulary sort

No stage sorts the row or the nucleus. K3 finds a k-th value by ternary search;
K4 resolves only the k-th tie group by prefix count; K5 finds boundaries by
ternary search on mass. The largest ordered structure built anywhere is the
≤2048-entry radix histogram of chunk 6 (workspace, deterministic integer
counts), never a vocabulary-sized permutation.

### 4.8 Determinism and batch independence

- The draw is `splitmix64(seed_r, position_r)` computed per row from
  `params[r]` — there is **no batch-wide RNG stream and no mutable RNG state**
  on device (contrast the draft's per-request Philox subsequence reserve,
  `dspark_rng.rs:34-49`). A rejected draft row consumes nothing because only
  emitted tokens advance `request.generated` (`scheduler.rs:77-87`,
  `:594`, `:601`).
- All reductions are fixed-shape per row: a row's result cannot depend on the
  number of rows in the wave, their order, the lane, or peer activity. The
  host-side pin is `scheduler.rs:666-691` (`sample_target_rows_keys_draws_on_absolute_position`),
  which also asserts that row-keyed draws differ from emitted-index draws. The
  GPU test in §12 reproduces this at the device level.
- `MAX_UNIFORM` clamp (`target_sampling.rs:51-52`, `:459`) is applied on device
  as `fminf(fmaxf(u, 0.0f), __uint_as_float(0x3F7FFFFFu))`.

---

## 5. Device ABI

### 5.1 Per-row parameter block

One struct per row, 64 B, natural alignment, **all parameters in device
memory** (no scalars baked into kernel arguments):

```c
/* native/cuda/kernels/v41_sampling_gpu.h  (shared: included by the .cu and mirrored in the C ABI) */
typedef struct ds41rt_v41_sampler_row_s {
  uint64_t seed;        /* +0  served request seed, two's complement (target_sampling.rs:161-167) */
  uint64_t position;    /* +8  absolute emitted-token index (scheduler.rs:542, :508-518) */
  float    temperature; /* +16 validated range 0..=2 (target_sampling.rs:46,115) */
  float    top_p;       /* +20 validated (0,1]; >= 1.0 means disabled (target_sampling.rs:118,177-179) */
  float    min_p;       /* +24 validated [0,1]; 0.0 means disabled (target_sampling.rs:124,449-454) */
  uint32_t top_k;       /* +28 0 = Option::None (disabled); 1 = greedy; k > vocab = no-op (target_sampling.rs:121-123,499-501) */
  uint32_t mask_row;    /* +32 0xFFFFFFFF = unconstrained; else index into the mask arena (constraints.rs:44-46) */
  uint32_t flags;       /* +36 bit0 greedy; bit1 diagnose; bit2 mask remainder-masked; bit3 cpu-oracle cross-check */
  uint32_t output_row;  /* +40 row index into logits/out_* (equals the struct index in practice) */
  float    ln_min_p;    /* +44 HOST-precomputed ln(min_p); negative infinity when min_p == 0.
                              Hard requirement: the device never calls logf (see §6.3a). */
  uint32_t reserved0;   /* +48 must be 0 */
  uint32_t reserved1;   /* +52 must be 0 */
  uint64_t reserved2;   /* +56 must be 0 */
} ds41rt_v41_sampler_row_t; /* exactly 64 B */
```

`flags` bits: bit0 greedy; bit1 diagnose; bit2 mask remainder-masked; bit3
CPU-oracle cross-check; **bit4 strict whole-row finiteness** (set for greedy and
constrained rows; see §4.1 and D9).

Rationale:

- **Why 64 B and an array of rows**: one `copy_h2d_async` per step, one
  contiguous buffer, `rows ≤ 80` (`v41_target_head.rs:124-129`) so the upload is
  ≤ 5 KB. Access is fully coalesced across a warp (one row per 64 B).
- **Why no scalars in the launch**: `sampling.cu:1287-1307` patches captured
  kernel nodes with `cudaGraphKernelNodeSetParams` and finds nodes **by index**
  (`sampling.cu:1203,1268`), so inserting a node shifts later indices and every
  parameter change requires a graph update. Reading parameters from device
  memory means the kernel node is invariant across steps, which is what makes
  §11's capture/replay story trivial.
- **Why `mask_row` instead of a pointer**: the mask arena has a fixed base and a
  fixed `words_per_row`, so a row index is stable across steps and needs no
  pointer patching. `0xFFFFFFFF` is the sentinel.
- **Disabled encodings are resolved on the host** when the struct is filled, so
  the kernel branch is a plain integer/float test: greedy `= temperature < 1e-5
  || top_k == 1` (`target_sampling.rs:172-174`); `top_p >= 1.0` means "use the
  ordered path's boundary search, not a skip" (§4.5); `min_p == 0.0` means
  disabled, signalled by `ln_min_p = -inf`. The host fills `ln_min_p` with the
  **same Rust `f32::ln`** the CPU sampler evaluates (`target_sampling.rs:451`),
  so the device performs exactly one f32 add and the `min_p` threshold comparison
  is bit-identical to the host. The kernel still re-derives greedy from
  `temperature`/`top_k` when `bit0` is clear, so a host bug cannot silently turn
  a stochastic row greedy.
- **Why `ln_min_p` is on the wire, not computed on device**: CUDA `logf` is
  allowed up to ~2 ulp error and is not bit-identical to glibc-backed Rust
  `f32::ln`, so computing it on device would make the `min_p` threshold — and
  therefore membership at the threshold — differ from the CPU for reasons that
  have nothing to do with accumulation. Precomputing it removes that class of
  divergence entirely (review item 3a).

**ABI surface (decision D8).** Three files, consistent with the existing sampler
entries:

1. `native/cuda/kernels/v41_sampling_gpu.h` — the struct above plus the
   kernel-facing declarations; included by the `.cu`.
2. `native/include/ds41rt_native.h` — the two `extern "C"` declarations, placed
   next to the argmax/sampler declarations (`:1671-1680`).
3. `rust/crates/ds41rt-ffi/src/lib.rs` — the `NativeLibrary` wrappers plus
   `validate_v41_sampling_buffers`, mirroring
   `validate_logits_sample_topk_topp_buffers` (`:17859-17896`).

The `.cu` is added to `DS41RT_NATIVE_SOURCES` next to `sampling.cu`
(`native/CMakeLists.txt:282`).

### 5.2 Constraint-mask layout

- Packed `u32`, `words_per_row = ceil(vocab/32)` (`= 4040` for `vocab = 129,280`,
  `scores.rs:7`), row-major, bit `t % 32` of word `t / 32` = token `t`
  (`scores.rs:154-169`, `target_sampling.rs:231-233`,
  `constraints.rs:44-46`).
- A row with `needs_mask == false` from `fill_bitmask`
  (`constraints.rs:45`, `lib.rs:3358-3379`) gets `mask_row = 0xFFFFFFFF`; the
  kernel treats it as all-allowed. There is no all-ones fill in the native path
  (contrast the legacy `real_full` representation,
  `commands/real_full/constraint.rs:237-245`).
- Arena: `capacity × words_per_row × 4 B` = 80 × 16,160 B = **1,292,800 B**,
  allocated once (§11).
- Width validation mirrors `target_sampling.rs:222-230`: the FFI validator
  checks `mask_words == vocab.div_ceil(32)`, and the kernel re-checks
  `mask_words` against its derived value and reports `MASK_WIDTH`.

### 5.3 Unused high bits of the final mask word

For the official checkpoint `129,280 % 32 == 0`, so the last word is exactly
full. The interface must still be width-driven (`target_sampling.rs:27-28`), so
the rule is:

1. Every kernel loop is bounded by `vocab`, so no code path can ever emit a
   token id `>= vocab` regardless of mask bits. This is the primary guard.
2. Before upload, the host sets any bit `>= vocab` in the last word to 0:
   `last &= (vocab % 32 == 0) ? u32::MAX : (1u32 << (vocab % 32)) - 1`. This
   makes a whole-word reader safe even if a future kernel reads 32 tokens per
   mask word (`apply_token_bitmask_f32_candidate_kernel` reads whole words,
   `sampling.cu:687-690`).
3. The kernel's `allowed(t)` is `mask_word(t/32) >> (t%32) & 1` evaluated only
   for `t < vocab`.

Today's CPU path never masks those bits and is harmless only because `allowed`
is per real token id (contract §6.1). The GPU path must not rely on that.
Pinned by §12's `vocab = 33` / `last word = u32::MAX` test.

### 5.4 Outputs

```c
ds41rt_status_t ds41rt_cuda_v41_target_sample_async(
    const float* logits, size_t rows, size_t vocab, size_t logits_stride,
    const ds41rt_v41_sampler_row_t* params, const uint32_t* mask_words,
    size_t mask_words_per_row,
    uint32_t* out_indices,      /* rows u32, REQUIRED: the only production read */
    uint32_t* out_status,       /* rows u32, REQUIRED: hard-error channel */
    uint32_t* out_status_detail,/* rows u32, REQUIRED-cheap: offending token / actual width */
    float*    out_scores,       /* rows f32, REQUIRED for greedy rows (retained-frontier check, scores.rs:61-66) */
    float*    out_total,        /* rows f32, OPTIONAL (diagnose bit) */
    uint32_t* out_nucleus_count,/* rows u32, OPTIONAL (diagnose bit) */
    void*     workspace, size_t workspace_bytes, void* cuda_stream);
/* plus the synchronizing wrapper with the same signature minus cuda_stream,
   matching sampling.cu:1321-1341's convention. */
```

Status codes: `0 OK`, `1 EMPTY_CANDIDATES`, `2 NONFINITE_LOGIT`,
`3 INVALID_TEMPERATURE`, `4 MASK_WIDTH`, `5 INTERNAL`. `out_status_detail`
carries the token id for `2`, the actual word count for `4`, else 0.

- **Genuinely useful diagnostics**: `out_status`/`out_status_detail` (error
  semantics otherwise lost), `out_scores` for greedy rows (the existing
  GPU-vs-CPU retained-frontier cross-check, `scores.rs:61-66`), `out_total` and
  `out_nucleus_count` (they let the oracle compare internals exactly, which is
  how §12 achieves bit-level filter-chain verification; `RankedSample.total` /
  `nucleus_count` are otherwise `dead_code`, `target_sampling.rs:403-410`).
- **Dead code to avoid**: a per-row probability (`out_scores` in the legacy
  kernel divides by `nucleus_mass`, `sampling.cu:675`) — no production caller
  reads it (`scores.rs:48-51`, `:120-123`); a separate top-two buffer — the
  trace (`scheduler.rs:565-576`, `scores.rs:127-142`) stays on an explicit,
  trace-gated download path (§8.5).
- The FFI validator must reject `rows == 0`, `vocab == 0`,
  `vocab > u32::MAX`, `mask_words_per_row != vocab.div_ceil(32)` when masks are
  present, non-finite/out-of-range params, and buffer extents — mirroring
  `target_sampling.rs:108-134` and `lib.rs:17859-17896`.

### 5.5 Buffer ownership

| Buffer | Owner | Lifetime |
| --- | --- | --- |
| logits `b(4)` | `TargetHeadWave` (`v41_target_head.rs:57-60`) | unchanged; read-only to the sampler |
| `param_staging` (pinned) + `param_device` | new `TargetSamplingWave`, owned by `TargetHeadWave` | allocated once in `TargetHeadWeights::wave` (`:44-87`), freed with the wave |
| `mask_staging` (pinned) + `mask_device` | same | same |
| `scratch_device` (per-row reductions + top-k bitmap arena) | same | same |
| `ids_host`/`status_host` (pinned), `ids_device`/`status_device` | same | same |
| optional diag device buffers | same | allocated only when the diagnose flag is enabled |

Charged to `TargetHeadWave::device_bytes` (`:124-129`) so head budgeting stays
honest. **Zero per-call cudaMalloc**, matching the existing pattern
(`greedy_staging: HostAllocation::new(library, capacity * 8)`, `:83`;
`RowDownload::new`, `:85`).

---

## 6. RNG decision and its consequences

### 6.1 Decision: port SplitMix64 bit-identically on device

Do **not** use Philox for the target draw. Reproduce
`TargetSamplingParams::random_uniform` (`target_sampling.rs:187-199`) exactly:

```cuda
__device__ __forceinline__ float v41_target_uniform(uint64_t seed, uint64_t position) {
  const uint64_t kDomain = 0x7f4a7c159e3779b9ULL;
  const uint64_t kMul    = 0x9e3779b97f4a7c15ULL;
  uint64_t mixed = seed + kDomain + position * kMul + kMul; /* wrapping, u64 */
  mixed = (mixed ^ (mixed >> 30)) * 0xbf58476d1ce4e5b9ULL;
  mixed = (mixed ^ (mixed >> 27)) * 0x94d049bb133111ebULL;
  mixed ^= mixed >> 31;
  return (float)(uint32_t)(mixed >> 40) * (1.0f / 16777216.0f);
}
```

Properties that make this exactly testable:

- `mantissa = (uint32_t)(mixed >> 40) < 2^24` is exactly representable in f32,
  and `2^-24 = 1.0f / 16777216.0f` is exact, so the product is a single exact
  f32 operation. The device value is bit-identical to the host value for every
  `(seed, position)`.
- `u64` wraparound is defined; the host is Rust `wrapping_*`
  (`target_sampling.rs:189-196`).
- `-1` is a real seed via two's-complement (`seed_from_i64`, `:165-167`), and
  `seed = 0` is a real seed (`:727-735`).
- The clamp `MAX_UNIFORM = 0.999_999_94` is applied after the draw, in the CDF
  comparison (`:459`), as `__uint_as_float(0x3F7FFFFFu)`.

Consequence: the seeded-output mapping is **unchanged**. This is the strongest
possible position for release wording and it avoids the entire
"mapping changed, re-qualify every seeded stream" branch of contract §7.2/§7.3.

### 6.2 Three separate reproducibility claims

| Claim | Status | How it is tested |
| --- | --- | --- |
| **(a) Bit-identity vs the current CPU sampler** | **Asserted for the uniform stream and for every count-only / order-independent decision. NOT asserted for the token**: any comparison against an f32 accumulation may differ (§6.3), because `expf` is not bit-identical and the reduction order differs. The *rate* of token divergence is a published measurement, not a claim | device-vs-host `random_uniform` over seeds {0, 1, `-1`→`u64::MAX`, `i64::MIN`, 4242} × positions {0..64, `2^31`, `2^32-1`, `2^63`, `u64::MAX`} — exact f32 bits; plus the K1/K3/K4 count-only oracle equality on the grid of §12; plus the measured per-cell mismatch rate (§12.9, §13) |
| **(b) Deterministic replay within the GPU implementation** | **Asserted, unconditionally** | same row re-run in different batch sizes (1, 6, 8, 16, 48, 64), row orderings, lane assignments, and with rejected drafts interleaved: the emitted id must be identical |
| **(c) Speculative vs non-speculative bitwise parity** | **Explicitly NOT claimed** | stated in release docs, matching `docs/release-v11-notes.md:85-86` and `docs/release-v11-performance.md:249` |

### 6.3 The residual difference, quantified, and the experiment that pins it

**6.3a — `min_p` threshold: made exact by construction (hard requirement).**
The device never calls `logf`. The host precomputes
`ln_min_p = min_p.ln()` with the same Rust operation as
`target_sampling.rs:451` and ships it in the per-row block (§5.1); the device
performs exactly **one** f32 add, `min_scaled = max_scaled + ln_min_p`, and the
comparison `scaled >= min_scaled` is then bit-identical to the host's. Without
this, CUDA `logf` (documented up to ~2 ulp) could flip membership at the
threshold permanently and the `±ULP` sweep would be unsatisfiable. Pinned by
§12.5.

**6.3b — `expf`: not bit-identical, and the dominant residual.**
Every weight `w_t = expf(scaled_t - max_scaled)` is evaluated by CUDA `expf`,
which is not bit-identical to Rust `f32::exp` (glibc-backed). A 1–2 ulp
difference in `w_t` propagates into `total`, into the normalized
`p_t = w_t / total`, and therefore into every comparison made against an f32
accumulation. **Chunk 2 must run the experiment**: compare CUDA `expf` (and a
double-precision `exp`-then-round variant) against Rust `f32::exp` on all six
phase-0 rows over the full parameter grid, record the per-weight and per-token
differences, and either pin a bit-matching path or declare the measured residual
here and in the release notes. This decision is a chunk-2 acceptance item, not an
assumption.

**6.3c — it is not "ULP-boundary rare"; it is any accumulation comparison.**
The first revision understated this. The differences are:

- the **top-p nucleus membership** comparison `M(K) >= top_p`, which can move a
  token into or out of the nucleus and hence move the draw by a whole rank (and
  change `nucleus_count`);
- the **draw** comparison `M(K) >= target`, which can select an adjacent rank
  (or the `:558` last-rank fallback);
- the **K2 fast-path** crossing `target <= cumulative`, where the device
  accumulation order differs from the CPU's strict token order
  (`target_sampling.rs:597-618`).

What remains exact and bit-comparable: the mask predicate, `scaled = logit * inv`
(the same IEEE f32 multiply, with no fast-math/FTZ, §4.0), the finite checks, the
`min_p` threshold via 6.3a, and the entire K3/K4 top-k **count** search (counts
are order-independent). The tie-id prefix in K4 is also count-based and exact.

**Quantification.** The phase-0 `near_uniform` `top_p0.9` cell has nucleus
support 116,344 of 129,280 (`REPORT.md:120`), so typical rank gaps in `p` are
~`1/116,344 ≈ 8.6e-6` — orders of magnitude larger than f32 rounding near 1.0
(~6e-8) and larger than a 1–2 ulp `expf` difference. Where the target lands in a
shifted gap, the selected rank differs; in that wide-nucleus regime the
GPU-vs-CPU token mismatch rate can therefore be **O(0.1–1)**, not ULP-rare. The
residual is a measurable divergence, and the design accepts it explicitly.

**Precedent already in-tree.** The CPU itself uses two different accumulation
orders for the same draw: the fast path accumulates in **token order**
(`target_sampling.rs:597-618`) and the ordered path in **rank order**
(`:534-570`). Accumulation-order sensitivity of the token is therefore already an
accepted property of the released sampler; this design adds a third, fixed,
per-row order and does not introduce a new class of behaviour.

**Required measurement (not optional).** The GPU-vs-CPU token mismatch rate per
phase-0 cell must be measured with the planned 8,192 seeded draws per cell
(§12.9), reported in the chunk-7 campaign, and published in the release notes.
Bit-identity is not claimed for the token; the mismatch rate is.

**Documentation rules**

1. Release notes must state: the uniform stream is **unchanged and
   bit-identical**; the filter rules are unchanged; the device accumulation order
   is fixed and deterministic; the **measured per-cell GPU-vs-CPU mismatch rate**;
   and that after this work the **GPU sampler — not the CPU sampler — defines
   served stochastic tokens** when the GPU path is active.
2. The same release notes must say the CPU sampler remains the correctness
   *oracle*, is not the default serving path, and how the A/B override behaves
   (§16 D2).
3. `docs/release-v11-performance.md`'s existing guarantee wording — "same logits
   + same execution path within this build" (`:249`) — remains literally true and
   must be kept.
4. Greedy is unaffected: it consumes no draw and keeps the compact device path
   (`target_sampling.rs:172-174`, `:234-236`; `v41_target_head.rs:350-377`).
5. If, during §12, a token differs for a reason traceable to a *non-accumulation*
   cause — a mask bug, a `min_p` threshold bug, a tie-rule bug — that is a design
   failure, not a documentation item.

### 6.4 Exactly what the release documentation must state

1. The RNG algorithm, domain constant `0x7f4a_7c15_9e37_79b9`, 24-bit
   truncation, `2^-24` scaling, `MAX_UNIFORM` clamp, and `(seed, position)`
   keying — explicitly unchanged from v11 (`target_sampling.rs:181-199`).
2. That greedy consumes no draw and is **unaffected** (`target_sampling.rs:172-174`,
   `:234-236`; `v41_target_head.rs:350-377`).
3. Claim (a) as scoped in §6.3 — uniform stream bit-identical, count-only
   decisions bit-identical, **token not bit-identical** and the measured
   per-cell mismatch rate published; claim (b) unconditionally; claim (c) not
   claimed.
4. Plainly: **after this work the GPU sampler, not the CPU sampler, defines
   served stochastic tokens**; the CPU sampler remains the oracle and an
   env-gated A/B fallback and is never the default (§16 D2).
5. That the CPU sampler is retained as the oracle and how the diagnostic
   override behaves (D2), including that the fallback must **not** silently
   reintroduce the full-row download on the default path.
6. That `EmptyCandidates` keeps today's worker-error status in this project and
   is recorded as a separate follow-up (D1), rather than being remapped now.
7. Which path served the measurement (GPU sampler, `dspark` on/off, drafts
   greedy) and the exact image/source revisions, following the existing
   provenance style (`README.md:214-226`).
8. Workload identity for the new measurement: equal weighted token counts and
   equal per-case completion tokens, as v11 already requires
   (`README.md:250-270`, `docs/release-v11-notes.md:60-72`).

### 6.5 Chosen design and the documented alternatives

#### 6.5.1 Chosen (this release): keep the CPU mapping intact, accept a measured GPU divergence

The device draws the same SplitMix64 uniform and then resolves it against a fixed
per-row accumulation order. The CPU sampler's `(seed, position)` mapping is
untouched, the existing oracle tests and every previously published seeded stream
remain defined, and the only new quantity is the measured GPU-vs-CPU token
mismatch rate (§6.3c). This is what the project brief authorizes: a measured
divergence, not a mapping change.

#### 6.5.2 Documented alternative (NOT chosen, out of scope): order-independent Gumbel-max

Replace the categorical scan with a maximum over the retained set of
`scaled_t + Gumbel_t`, where each token's Gumbel is derived deterministically from
`(request seed, absolute emitted position, token id)`. Properties, stated
precisely:

- It makes CPU and GPU agree **exactly** regardless of accumulation order,
  because no cumulative sum decides the result; the comparison is a plain max
  over independent per-token values, and a per-token order-independent value is
  reproducible in any reduction tree.
- It removes the rank-order dependence entirely — a strictly stronger
  reproducibility property than the chosen scheme, and one that would make claim
  (a) hold for the *token* as well as the uniform.
- **But it changes the seeded mapping for the CPU sampler too.** The CPU
  reference (`target_sampling.rs`), the in-tree oracle
  (`reference_select`, `:1235-1399`) and every released seeded stream would have
  to be updated and re-validated, and the change would be a documented
  seeded-output mapping change under contract §7.2/§7.3.
- Cost: one deterministic hash per retained token per row (up to ~129k hashes/row)
  plus a max reduction, replacing the K5 rank scan; still no sort, but a new
  per-token RNG domain must be specified and tested.

**Disposition.** Out of scope for this release. Reconsider **only if** the
measured GPU-vs-CPU token mismatch rate from §12.9/§13 proves unacceptably large
(the chunk-7 gate decides). The design keeps the CPU mapping intact and accepts a
measured GPU divergence, which is the authorized scope.

#### 6.5.3 Documented option for the fast path: strict token-order accumulation

The K2 fast path (`top_k` disabled and `top_p >= 1`, `target_sampling.rs:463-466`)
is the one branch whose CPU draw order is **ascending token id**
(`sample_categorical`, `:597-618`). The chosen K2 already scans in token order
inside contiguous segments but combines segment totals with a fixed tree. A
strictly sequential combine — each segment's carry taken from the previous
segment's final cumulative, e.g. produced by a single-pass carry propagation or a
one-thread-per-row fallback for small `vocab` — would make K2's crossing
comparison bit-identical to the CPU's token-order accumulation, leaving `expf`
(§6.3b) as the only residual on that path.

**Disposition.** Evaluate in the chunk-2 experiment: measure the sequential
combine's cost against the tree combine at `rows` 1–64 and report which is used.
It is cheaper and lower-risk than the Gumbel alternative because it changes no
mapping at all; it is not adopted up front only to avoid a serialization
bottleneck in the wide-nucleus case.

---

## 7. Semantics preservation checklist

Every item of the contract's MUST-NOT-CHANGE list
(`docs/gpu-sampler-contract-v11.md` §7.1) mapped to a device rule and a pinning
test. "Anchor" is the CPU rule the device must reproduce.

| # | MUST NOT CHANGE | Device rule | Pinning test (§12) |
| --- | --- | --- | --- |
| 1 | Filter order mask → T → min_p → top_k → top_p → draw (`target_sampling.rs:9-25`) | K1 mask/max, survivor predicate folds min_p, K3/K4 top-k, K5 top-p, then draw. min_p is a predicate on `logit*inv`, evaluated before top-k consumes anything | `mask_is_applied_before_sampling_filters` analogue; ordered-stage oracle on the full grid |
| 2 | Greedy: `T < 1e-5` or `top_k == 1`; consumes no draw (`:172-174`, `:234-236`) | `flags` bit0 resolved from both; K1-only; no `random_uniform` call | device greedy id equals CPU argmax for `T ∈ {0,1e-6,1e-5-ulp}` and `top_k = 1` |
| 3 | Greedy tie-break first/lowest id; compact path stays device-only (`:259-262`, `scores.rs:44-46`, `v41_target_head.rs:349-386`) | K1 argmax uses strict `>` + lowest-id reduction; compact lane never calls the sampler | `test_cuda_logits_argmax_matches_ref` still green; mixed-lane greedy probe |
| 4 | Served defaults: no temperature → greedy; unset filters disabled (`native_v41.rs:145-159`) | No kernel change: host fills `flags`/`top_p = 1`/`min_p = 0`/`top_k = 0` exactly as `request_target_sampling` resolves (`native_v41.rs:149-188`) | existing API tests `upstream_native_v41.rs:138-191` unchanged |
| 5 | `top_k` spellings (`absent/null/0/-1` → None; `k >= vocab` no-op) (`native_v41.rs:162-176`, `target_sampling.rs:499-501`) | `top_k = 0` disabled; `k >= survivor_count` skips K3/K4 | grid includes `top_k ∈ {0,1,2,40,V-1,V,V+1}` |
| 6 | `min_p` in **logit** space, relative to scaled max, inclusive `>=`, `0` disabled (`:449-458`) | `min_scaled = max_scaled + ln_min_p` where `ln_min_p` is the host-computed Rust `f32::ln` (§5.1, §6.3a); predicate `scaled >= min_scaled`; `min_p == 0` → `-inf` | min_p near-threshold sweep at `max_scaled + ln_min_p ± 1 ULP`, both sides, must pass permanently |
| 7 | top_p inclusive `>=` prefix, ≥1 token, clamp `1e-6..=1.0`, `1.0` reaches the ordered path when `top_k` is set (`:544-554`) | K5 **largest key with `M(K) >= top_p`** (the last rank of the prefix); nucleus never empty because the best token always satisfies; no `top_p = 1.0` skip | boundary-uniform probes at reference `boundaries ± 1e-6` (`:1545-1562`) **plus** the `top_k = 40, top_p = 1.0` prefix-overshoot oracle of §12.3 |
| 8 | Chain never empties; all-masked is a hard error, never a fallback (`:23-25`, `:1026-1048`) | K1 `EMPTY_CANDIDATES` status; `out_indices` never written for that row | all-zero-mask device test in every mode (K1-only, fast, ordered) |
| 9 | Draw keyed on `(seed, absolute position)`; batch independent; rejected drafts consume nothing (`:35-38`, `:998-1007`, `:1100-1208`) | per-row `(seed, position)` in the param block; stateless kernels | device RNG bit-equality; `scheduler.rs:666-691` analogue under reordering |
| 10 | `seed_from_i64` two's complement; `seed = 0` real; unset seed generated (`:161-167`, `:727-735`, `native_v41.rs:129-137`) | host resolves exactly as today; kernel treats seed as opaque u64 | RNG grid includes `u64::MAX`, `0`, `1<<63` |
| 11 | Target/draft RNG domain separation (`:182-186`) | target only uses `(seed, position)` + domain constant; the draft's Philox stream (`v41_dspark.cu:245-266`, `dspark_rng.rs:34-49`) is untouched | draft RNG reservation tests unchanged; no target/draft cross-talk test |
| 12 | Verifier semantics (`dspark_verify.rs:18-52`, `scheduler.rs:539-604`, `constraints.rs:61-102`) | unchanged code; the device only replaces the per-row `selected[i]` producer | `speculative_sample_match_equals_sequential_sampling` + device analogue |
| 13 | Mask layout/word count/mask-first/hard error (`scores.rs:154-169`, `constraints.rs:44-46`) | §5.2/§5.3; K1 predicate first | vocab-not-divisible-by-32 tests; mask-width status test |
| 14 | NaN/Inf discipline: reject non-finite allowed logits; non-finite scaled max → invalid temperature; never NaN/panic (`:256-258`, `:434-448`) | K1 status precedence of §4.1; **mode-specific masked-non-finite rule**: greedy/constrained strict over the whole row (`scores.rs:160-163`), stochastic permissive on masked tokens (`target_sampling.rs:1009-1024`), per-row flag bit4 | `-f32::MAX` at `T=1e-5`; huge logit at tiny T; masked NaN per mode; strict-mode masked NaN must error |
| 15 | Greedy throughput not poisoned by a stochastic peer (`scheduler.rs:397-407`, `:458-462`) | per-row params + per-row greedy branch; no whole-batch full download | mixed-lane test asserting zero full-row D2H and greedy id equality |

---

## 8. Integration plan

### 8.1 Seams to change

| Seam | Today | After |
| --- | --- | --- |
| `execute_logits` (`scheduler.rs:389-407`) | non-compact: `pass.execute` → `vec![0; logits.logits.bytes]` → blocking `lib.copy_d2h` (`:400-403`); compact: `execute_greedy` (`:397-398`) | non-compact: new `pass.execute_sampled(...)` terminal that runs the head graph **and** the sampler on the same stream, then copies `rows×4` ids + `rows×4` status + `rows×4` scores; compact unchanged |
| `independent.rs:137-144` | compact: `execute_shared_greedy`; non-compact: `execute_shared` + `BatchScores::new(pass.download_logits(...).await?)` | compact unchanged; non-compact: `execute_shared_sampled(...)` + `BatchScores::from_sampled(ids, status, scores)` |
| `scores.rs` `BatchScores` | full `bytes: Vec<u8>` or compact `best: Vec<u32>`; `new` runs a CPU argmax per row (`:67-71`) | add `from_sampled`; keep `new` only for the CPU fallback/oracle; hot path never calls `new` |
| `scores.rs::row_logits` (`:11-17`) | called per sampled row (`:47`, `:119`) | removed from the hot path; retained for the fallback/oracle |
| `v41_target_head.rs` | `execute_block` (`:324-331`), `execute_block_cooperative` (`:334-348`), `execute_block_greedy` (`:350-377`) | add `execute_block_sampled` / `execute_block_sampled_cooperative`, modelled on `execute_block_greedy`: `copy_block` → graph ensure/launch → sampler launch → existing `wait()`/`synchronize()` → published outputs |
| `v41_target_pass.rs` | `execute_greedy`/`execute_shared_greedy` (`:177-189`) | add `execute_sampled`/`execute_shared_sampled` mirroring them, passing the prepared `TargetSamplingWave` |
| `constraints.rs` | `select_verification` (`:61-73`), `select_verification_sampled` (`:80-102`) select on the CPU from host rows | split into `prepare_verification_masks(input) -> MaskSet` (fork + `fill_bitmask` per row, CPU, unchanged semantics) and device selection; both greedy and stochastic constrained rows then use the device |
| `prepare_commit_lane` (`scheduler.rs:528-607`) | `finishing && next.has_full_logits()` retains host bytes (`:598-599`) | `finishing && bank_enabled` schedules a single-row device download (`:601` → `:638-640`) |

### 8.2 What replaces the full-row D2H

Per round, a non-compact lane downloads `rows × (4 + 4 + 4)` B (ids, status,
greedy scores) instead of `rows × 517,120` B. ARITHMETIC: 48 rows × 12 B = 576 B
versus 24.8 MB before. The retained frontier is the only remaining full-row
transfer and it is one row per *finishing* request (§10).

### 8.3 How per-row parameters and masks reach the device

1. `prepare_decode_lane` already flattens members in order
   (`scheduler.rs:409-416`) and `selected` is `0..positions().len()`
   (`scheduler.rs:396`, `independent.rs:129`), so the row order is known.
2. Build `Vec<ds41rt_v41_sampler_row_t>` in that order: for member `m` with
   `input.len()` rows, row `i` gets `seed = params.seed()`,
   `position = request.generated + i` (`scheduler.rs:542`, `:516`),
   `temperature/top_p/min_p/top_k` from `request.job.sampling` (`:541`),
   `flags.greedy = params.is_greedy()`, `mask_row` from the mask set (§9).
3. Constrained members' masks are prepared once per step (§9) into
   `mask_staging`.
4. `TargetSamplingWave::upload` fills the pinned staging and issues one
   `copy_h2d_async` for params and one for masks on the head stream, then the
   sampler launch, all before the existing drain. `require_complete()` guards
   overwriting staging on the next round (the pattern at `v41_memory.rs:75-78`).

### 8.4 Greedy compact path and mixed-lane poisoning

- The compact predicate is unchanged (`scheduler.rs:458-462`,
  `independent.rs:131-135`). An all-greedy, unconstrained lane never calls the
  sampler and keeps `execute_block_greedy` byte-for-byte
  (`v41_target_head.rs:350-377`).
- A lane with at least one stochastic/constrained member is non-compact **as
  today**, but now every row is selected on device: greedy rows via K1's greedy
  branch (mask-aware when constrained), stochastic rows via K1+K2/K3/K4/K5. No
  CPU argmax, no full-row download. Per-row parameters make the lane-wide
  heterogeneity safe by construction: a row's behaviour is a pure function of its
  own struct.
- The existing lane-level split is preserved: two requests in *different* lanes
  never contaminate each other (`independent.rs:52-54`,
  `layout.rs:33-36`).

### 8.5 `BatchScores::new`'s redundant CPU argmax and the 517 KB host Vec

- The CPU argmax in `BatchScores::new` (`scores.rs:69`) is deleted from the hot
  path. Greedy rows in a sampled lane read `out_indices`/`out_scores` from K1.
  That reproduces the checked-argmax contract (lowest id, non-finite handling)
  and keeps the retained-frontier cross-check meaningful (`scores.rs:61-66`).
- The per-row 517,120 B host `Vec` (`scores.rs:47`, `:119`) never exists.
- `ds41rt::logit_trace` (`scheduler.rs:458`, `:565-576`, `scores.rs:127-142`)
  needs full rows for `top_two`. Keep it working by having the trace path
  request an explicit, trace-gated per-row download (only the rows it logs) —
  it already disables compact (`scheduler.rs:458`) and is documented as an
  untimed diagnostic (`runs/gpu-sampling-phase0/README.md`). Do **not** add a
  per-row top-two reduction to the production kernel just for the trace.
- The single-request path stops using blocking pageable `cudaMemcpy`
  (`ds41rt_native.cc:1625-1647`) because `single_lane_round` calls the same
  `execute_sampled` terminal as the lane path (`scheduler.rs:471`,
  `layout.rs:33-43`). This also removes the ambiguity recorded at
  `runs/gpu-sampling-recon-paths-2026-09-22.md:175` about which download path a
  1-request benchmark used.

### 8.6 The CPU sampler as retained reference/fallback

- Retained as (i) the test oracle, via `reference_select`
  (`target_sampling.rs:1235-1399`) and `compare_to_reference` (`:1401-1450`);
  (ii) a **build- and env-gated diagnostic** fallback.
- Recommendation (D2): default is the GPU path; `DS41RT_TARGET_SAMPLER=cpu`
  selects the CPU path for diagnosis only, and it is documented as changing the
  in-build token stream at cumulative-boundary cases. It is *not* an automatic
  fallback: if the native symbol is missing, fail at startup, consistent with the
  existing missing-symbol diagnosis (`DEVELOPER.md:67-68`). A silent automatic
  fallback would make the reproducibility claims unverifiable.

---

## 9. Constrained decoding on GPU

### 9.1 Matcher stays on the CPU

`fill_bitmask` targets a CPU `DLTensor` (`native/src/xgrammar_adapter.cc:349-352`),
and the FFI wrapper is CPU-side (`lib.rs:3358-3379`). Nothing in this design
moves xgrammar. Only mask storage and application move.

### 9.2 Mask rows for the speculative prefix, uploaded once per step

Refactor `State::select_verification_sampled` (`constraints.rs:80-102`) into:

- `prepare_verification_masks(&self, input) -> Result<MaskSet>` — fork the
  matcher once (`self.matcher.fork()`, `:88`), then for each row `index`:
  `if index > 0 { branch.accept_token(input[index])? }` (`:92`, error
  `"illegal verification draft token"`), `branch.fill_bitmask(&mut mask)` (`:93`),
  record `needs_mask` and copy the mask row into a flat
  `rows × words_per_row` buffer. Row 0's mask is the pre-draft state, exactly as
  today (`:66-69`).
- The flat buffer is uploaded once per step (§8.3). `needs_mask == false` rows
  get `mask_row = 0xFFFFFFFF`.

Preserved exactly: row 0 uses the mask before any draft accept; later rows use
the mask after accepting the preceding draft; `input[0]` is the committed anchor
already accepted by the authoritative matcher; the authoritative matcher still
advances only on emitted tokens.

### 9.3 Mask applied before filtering

K1's masked max/count scan is the first thing that touches the row, and every
later predicate re-applies `allowed(t)`. No filter sees a masked token
(§4.0). `min_p`, top-k and top-p all operate on the masked, scaled values.

### 9.4 fork / commit / rollback and the authoritative advance

Unchanged in code:

- fork = `matcher.fork()` (`lib.rs:3344-3356`);
- commit = `State::accept` → `matcher.accept_token` on emitted tokens only
  (`constraints.rs:47-50`, called from `scheduler.rs:79`);
- rollback = dropping the fork; the authoritative state is never mutated by
  verification (`constraints.rs:62-72`, `:88-101`);
- proposals are trimmed first by `truncate_proposal` (`constraints.rs:51-60`,
  called `scheduler.rs:436`, `independent.rs:91`).

The device sampler only consumes the prepared masks and returns ids; it cannot
observe or mutate grammar state.

### 9.5 Empty-mask hard error and the HTTP-status decision

- An all-zero mask yields `EMPTY_CANDIDATES` in every mode, never a silent
  unmasked fallback (`target_sampling.rs:1026-1048`). The device reports it per
  row; the host maps it to the same errors as today: greedy
  `select_verification` currently produces `"grammar allows no target token"`
  (`scores.rs:168`) and stochastic produces `EmptyCandidates`
  (`target_sampling.rs:58`, `:440-444`).
- Today `is_bad_request` (`target_sampling.rs:70-76`) has **no consumer** in
  `rust/crates` (grep returns only the definition), so a grammar admitting
  nothing surfaces as a worker error (HTTP 500), not 400. The contract flags
  this as an open ambiguity (§6.4).
- **Decision (D1): preserve today's behaviour in this project.** A grammar that
  admits no token keeps surfacing as a worker error (HTTP 500), not 400. The
  device reports the per-row status; the host maps it through the **same error
  strings as today** — `"grammar allows no target token"` for the greedy twin
  (`scores.rs:168`) and `EmptyCandidates` for the stochastic twin
  (`target_sampling.rs:58`, `:440-444`). Rationale: this project must not change
  an observable HTTP status as a side effect of moving the sampler; `is_bad_request`
  has no consumer today (`target_sampling.rs:70-76`), so wiring it is a separate
  API-behaviour change with its own review. The 400 mapping is recorded as a
  follow-up item, not implemented here. The device status codes are unchanged in
  either case, so the follow-up is a one-line host mapping.

### 9.6 No per-position logits download or extra synchronization

Masks are prepared CPU-side (as today) and uploaded asynchronously before the
launch. There is no per-position logits download anywhere in the constrained
path, and the per-step sync count is unchanged (§11). The mask upload rides the
existing head stream.

---

## 10. Retention and the frontier

### 10.1 Genuine need: one raw row per retained snapshot frontier

A retained snapshot stores `TokenScores` (`prefix.rs:12-21`, `:191-197`) so a
later, different grammar can re-select the *first* token without replaying
prefill: `TokenScores::select(Some(mask))` runs a CPU masked argmax over the
stored row (`scores.rs:30-35`), pinned by
`new_constraint_reselects_an_exact_cached_frontier` (`scores.rs:195-206`). The
row cannot be reconstructed from an id, so one raw row per retained frontier is
genuinely required. Restore additionally requires it for an exact-frontier hit
(`scheduler.rs:312`, `prefix.rs:248-…`).

### 10.2 Decision: selective download, not device-side retention

Choose the **selective single-row download** (the existing
`TargetHeadWave::download_rows`, `v41_target_head.rs:405-409` →
`v41_memory/download.rs:17-42`; and `BatchScores::retain_from_device`,
`scores.rs:80-90`). Justification:

- The row is needed **once per finishing request**, not per step, and only when
  the request is cacheable. Frequency is already low.
- `PrefixCache::Saved.next` is a host `TokenScores` shared by clone
  (`prefix.rs:195`, `:237`); device-side retention would force the snapshot bank
  to own a device logits row per retained turn and would require the
  grammar-re-selection to run on device, adding a new sync at restore. Device
  residency (20 retained turns × 517 KB ≈ 10 MB) is affordable but buys nothing:
  the restore path already exists and is exact.
- Keeping the host row preserves the exact-frontier behaviour byte-for-byte and
  keeps the existing GPU-vs-CPU argmax cross-check
  (`retain_downloaded`, `scores.rs:61-66`) as a live invariant.
- This is the only full-row D2H left on the hot path, and §13 counts it
  explicitly.

### 10.3 Preserving exact-frontier behavior

`publish_commit_lane` stores `next_after_commit` (`scheduler.rs:625-627`);
retirement calls `PrefixCache::retain`/`queue_retain` (`scheduler.rs:121-124`,
`independent.rs:242-243`) which clones the host `TokenScores` into `Saved.next`
(`prefix.rs:195`, `:237`). None of that changes. Only the *source* of the row
changes: from "slice of the already-downloaded batch" (`scores.rs:144-151`) to
"one-row device download at commitment" (`scores.rs:80-90`).

### 10.4 The incidental unconditional frontier download

`scheduler.rs:597-603` schedules `frontier_downloads` whenever a request is
`finishing && !next.has_full_logits()`, and `commit_lane` performs it at
`:638-639`; `independent.rs:151-158` does the same. That happens **regardless of
whether prefix caching is enabled**, and only later does
`prefix.rs:177`/`:206` early-return when `bank.limit() == 0`. So a greedy
finishing request in a cache-disabled deployment still pays a 517,120 B D2H.

Replacement: query the bank state before scheduling (add
`PrefixCache::turn_bank_enabled(lane)` reading the same `bank.limit()` used at
`prefix.rs:177`), and skip the frontier download entirely when caching is
disabled or the request is not cacheable (`cacheable` is already tracked,
`scheduler.rs:325`, set in `emit_one` `:106`). When caching is enabled, keep the
single-row download.

---

## 11. Workspace, streams, and graph compatibility

### 11.1 Workspace, sized for maximum rows × vocab

Allocated once per `TargetHeadWave`:

| Region | Bytes | Formula |
| --- | ---: | --- |
| sampler params (device) | 5,120 | `capacity × 64` |
| sampler params (pinned staging) | 5,120 | `capacity × 64` |
| mask arena (device + pinned) | 2 × 1,292,800 | `capacity × ceil(vocab/32) × 4` at capacity 80 |
| scratch (per-row reductions) | `capacity × 64` | 4 f32 + 6 u32 padded |
| top-k bitmap arena | `capacity × ceil(vocab/32) × 4` | 1,292,800 at capacity 80 |
| ids/status/detail/scores | `capacity × 4 × 4` | device + pinned |
| optional diag (`total`, `nucleus_count`) | `capacity × 8` | only when diagnose |
| chunk-6 radix histogram (optional) | `capacity × 2048 × 4` | 655,360 at capacity 80 |

Total ≈ 4.2 MB at capacity 80, well within the existing head budget
(`TargetHeadWave::device_bytes`, `v41_target_head.rs:124-129`). **Zero per-call
allocations**: no `cudaMalloc`, no `Vec` growth, no `to_vec()` on the hot path
(contrast `download.rs:41`).

### 11.2 Streams and synchronization

- Everything is enqueued on the head stream (`self.stream.raw`,
  `v41_target_head.rs:67-70`), after `cuda_graph_launch` and before the existing
  drain — exactly where the greedy argmax kernel and its two small D2H run today
  (`:363-372`).
- **No new host synchronizations.** The single per-step sync is the existing
  one: `stream.wait()` for the cooperative lane path (`:344`,
  `v41_memory.rs:82-99`) or `stream.synchronize()` for the single-lane path
  (`:327`, `:232`). Because the sampler and its small D2H are enqueued before it,
  the same drain returns the ids. The blocking pageable `cudaMemcpy`
  (`scheduler.rs:402`) disappears.
- **No grid-wide sync, no cooperative launch.** Every kernel is per-row
  independent with intra-CTA `__syncthreads()` only. This is what makes capture
  legality trivial; note that even the FlashInfer radix fallback has no grid sync
  (corrected fact 1), so it stays available.

### 11.3 Capture/replay legality and the row-count key

- The head graph captures only `copy_block` + collapse/norm/projection
  (`v41_target_head.rs:198-220`); the sampler is launched **outside** the
  capture, like the greedy argmax kernel at `:366`. Therefore:
  - the captured-row-count key `ensure!(rows == count, "target head captured rows differ")`
    (`:226`) and `capture_block_head` (`:301-307`) are unaffected;
  - no kernel-node index patching is involved, so the fragile
    `find_kernel_node_by_index` pattern (`sampling.cu:1203,1268`) is irrelevant;
  - parameters come from device memory, so a captured sampler (if we later add
    one) would not need `cudaGraphKernelNodeSetParams`.
- The sampler's grid depends on `rows`, which is already validated against
  capacity (`:147-154`). The design deliberately **does not capture the sampler**
  in the first version; if capture is wanted later, rows becomes part of the key
  and must be bucketed.
- Ordering hazard: the sampler reads `b(4)` and the next round overwrites it via
  `copy_block`. Same-stream ordering covers this. Filling the param/mask staging
  must not race a prior in-flight launch; the single staging buffer is safe
  because the previous round's drain completes before the next upload, and
  `require_complete()` is asserted as at `v41_memory.rs:75-78`. If a future
  pipeline overlaps rounds, double-buffer params and masks.

---

## 12. Correctness validation plan

### 12.1 Device tests in the `cuda_selftest.cc` style

Add to `native/tests/cuda_selftest.cc` (registered next to line 2869-2872):

- A CPU oracle `cpu_v41_target_sample(...)` in the style of
  `cpu_logits_sample_topk_topp` (`:545-624`) that is a **faithful port of the
  production `reference_select`** (`target_sampling.rs:1235-1399`) — mask,
  temperature, min_p, top_k exact-k-lowest-id, inclusive top-p, SplitMix64 draw,
  status codes and `total`/`nucleus_count`. The reviewer must diff this port
  against `reference_select` line by line; it is the only oracle that can
  validate the new rules.
- `test_cuda_v41_sampling_greedy_matches_ref`, `..._fast_path_matches_ref`,
  `..._ordered_matches_ref`, `..._masked_matches_ref`,
  `..._status_precedence`, `..._vocab_remainder`, `..._rng_bit_equality`.
- Follow the existing conventions: `device_buffer`, `copy_h2d`, `copy_d2h`,
  `assert_close`, `require_status` (`:2280-2300`, `:2158-2224`).

### 12.2 Rust oracle tests against a port of `reference_select`

- Reuse `reference_select`/`compare_to_reference` (`target_sampling.rs:1235-1450`)
  as the oracle. Add a `#[cfg(test)]` module that drives the **device** entry
  point through the FFI when `DS41RT_NATIVE_LIB` is set, following the
  ignored-test pattern at `v41_memory/download.rs:45-70`, so
  `cargo test` stays green without a GPU.
- The device test compares token id, status, `total` bits and `nucleus_count`
  where the reduction is order-exact (count-only and `min_p`-threshold decisions),
  and compares token id + distribution elsewhere. It also **records** the token
  mismatch count against the CPU oracle per cell rather than asserting zero
  (§6.3c).

### 12.3 Full parameter grid × masks × boundary uniforms

Mirror `param_grid` (`target_sampling.rs:1496-1524`):

- temperatures `{1e-5, 1e-4, 0.2, 0.7, 2.0}`;
- top_k `{1, 2, 40, V-1, V, V+1, None}`;
- top_p `{1e-6, 0.5, 0.9, 0.95, 1-1e-7, 1.0}`;
- min_p `{0, 1e-6, 0.05, 0.5, 1.0}`;
- masks `{None, token<5, token%3 != 0}`;
- uniforms `{0.0, 0.25, 0.5, 0.75, MAX_UNIFORM, 1.0}`;
- plus reference `boundaries ± 1e-6` (`:1545-1562`) and the adversarial rows
  `all_equal`, `two_value`, `geometric`, `huge_gap`, `single_dominant`,
  `ties_at_max`, `negpos_zero_subnormal`, `negative_only`, `flat_negative`
  (`:1452-1494`).
- **`top_k` + `top_p = 1.0` prefix-overshoot oracle (required by the review).**
  For `top_k ∈ {2, 40, V-1}` with `top_p = 1.0`, on rows where the f32 prefix sum
  of normalized weights reaches `1.0` *before* the last retained survivor
  (construct rows with a dominant group followed by a long tail of tiny weights,
  and rows whose normalized weights sum to just above `1.0`), assert the device
  nucleus is the **same strict prefix** as the CPU (`:548-554`) and that
  `nucleus_count` matches. This is the regression pin for the removed
  "nucleus = all of `S`" shortcut; if that shortcut is ever reintroduced, this
  case fails.

### 12.4 Tie cases

- **Exact-k lowest-id**: rows where tokens `j..j+k` share the k-th value; expect
  ids `j..j+k` and never a higher id, and expect exactly `k` survivors.
- **top-p boundary ties**: a row where the prefix boundary falls inside an
  equal-value group; expect the lowest-id members up to the boundary rank and
  nothing beyond.
- **All-tied rows**: `all_equal` / `ties_at_max` with every `top_k` and
  `top_p`; expect deterministic lowest-id behaviour and never an empty nucleus.
- **Strict prefix at `top_p = 1.0`**: the `:545-554` break can fire before the
  last retained survivor, so a `top_k`-truncated row at `top_p = 1.0` must be
  allowed to produce a strict prefix; assert the same prefix as the oracle and
  that `out_nucleus_count` is not simply `survivor_count`.

### 12.5 min_p near-threshold sweeps

For each adversarial row and each `min_p ∈ {1e-6, 0.05, 0.5, 1.0}`, construct a
token with `scaled == max_scaled + ln(min_p)` exactly, and one ULP below; assert
inclusive retention and exclusive rejection. Because the threshold is computed by
Rust `f32::ln` on the host and added on device (6.3a), this sweep must pass
**permanently**; a device-side `logf` implementation would fail it intermittently
and is forbidden. Also assert `min_p = 0` keeps `-inf`-scaled survivors
(`negative_infinity_survivor_ordered_paths_match_reference`, `:1748-1764`) and
`min_p = 1` keeps every tied maximum (`:737-753`).

### 12.6 `-inf` / NaN / finite discipline (mode-specific)

- **Masked NaN is mode-specific and both modes are pinned** (review item 7):
  - stochastic rows: legal and unread (`target_sampling.rs:1009-1024`);
  - greedy and constrained rows: an **error**, matching `scores.rs::argmax`'s
    `ensure!(value.is_finite())` that runs *before* the mask test
    (`scores.rs:160-163`). The strict-mode test must use a masked non-finite token
    and assert `NONFINITE_LOGIT`, so the constrained path is never silently
    loosened.
- Allowed NaN/Inf → status `NONFINITE_LOGIT` with the lowest offending id
  (lowest *allowed* id in stochastic mode, lowest id over the whole row in strict
  mode).
- `logit = -f32::MAX` at `T = 1e-5` scales to `-inf` and remains a legal
  survivor when `min_p = 0` (`:1744-1763`).
- `logit = f32::MAX` at tiny `T` → `INVALID_TEMPERATURE`, never NaN, never a
  panic (`:755-776`).
- Error precedence: non-finite allowed logit beats `EmptyCandidates` beats
  invalid temperature, per `:434-448`.
- **Subnormal discipline**: the `negpos_zero_subnormal` adversarial row
  (`f32::from_bits(1)`, `:1481`) must produce bit-identical scaled values and
  weights to the host; this is the test that catches an accidental FTZ/fast-math
  build (§4.0).

### 12.7 Vocabulary not divisible by 32

Vocabuli `{1, 32, 33, 100, 127, 129_280, 129_281}` as in
`large_vocabulary_paths_match_reference_at_boundaries` (`:1573-1603`), with the
last mask word set to `u32::MAX`. Assert no out-of-vocab id is ever returned and
that `MASK_WIDTH` fires when the provided word count differs. The official
checkpoint `129,280 % 32 == 0` means this is a purely synthetic-width guarantee;
say so.

### 12.8 Seeded replay across batch sizes, lanes, orderings, rejected drafts

- **Bit equality of the RNG**: device vs host over the seed/position grid of
  §6.2, including positions `2^63` and `u64::MAX`.
- **Batch independence**: run the same logical row in waves of 1, 6, 8, 16, 48,
  64, in forward/reverse/shuffled order, in lane 0 and lane 1, with unrelated
  peers, and with interleaved mismatching drafts; assert the id is identical.
  This is the device analogue of `scheduler.rs:666-691`.
- **Speculative equivalence**: the device analogue of
  `speculative_sample_match_equals_sequential_sampling`
  (`target_sampling.rs:1100-1208`) — over ≥24 positions, constrained and
  unconstrained, with deliberate mismatches on odd rounds, assert the speculative
  sequence equals sequential device sampling at the same absolute positions, and
  that rejected rows consume no draw.
- **Bonus token** (adopting contract §8.4's required check): on an all-match
  round the verifier emits `input.len()` tokens, the last of which was never
  itself an input row (`dspark_verify.rs:6-8`, `:41-44`). Add a target-sampling
  test asserting that the bonus token equals a *sequential* device draw at that
  same absolute emitted position, closing the ambiguity noted at contract §8.4
  item 4.

### 12.9 Distributional comparison on the fixed-logit harness inputs

Replay the phase-0 harness rows (`runs/gpu-sampling-phase0/logits/*.f32`,
manifest `out/logits-manifest.json`, six rows: `peaked`, `moderate`,
`near_uniform`, `tied`, `long_tail`, `trace_anchored`) through the device entry
point at 8,192 seeded draws per cell, and compare the device histogram to the
CPU-sampler histogram and to the independent f64 analytic reference already
built in the phase-0 harness (`REPORT.md:106-124`,
`out/histogram-oracle-validation.json`). Total-variation must be within the
Monte-Carlo noise reference for every cell. Because `runs/` is gitignored, land
the harness rows in the repo as part of chunk 7 or assert their sha256
(§16 D4).

The same 8,192-draw sweep must also record the **GPU-vs-CPU token mismatch count
and rate per cell** (§6.3c). That number is a required chunk-7 deliverable and
must appear in the release notes; the distribution can match within noise while
the per-draw token differs, and both facts must be reported.

### 12.9a `expf` parity experiment (chunk 2 gate, review item 3b)

On the six phase-0 rows over the full parameter grid, measure:

- `max |w_cuda - w_rust|` in ulp for every survivor, and the count of survivors
  whose weight differs;
- the number of tokens whose selected id differs, comparing CUDA `expf` against
  Rust `f32::exp`, and against a double-precision `exp`-then-round variant;
- the end-to-end effect on `total`, on the modelled `p_t`, and on the
  top-p membership.

The outcome is one of: (i) pin the variant that minimises or eliminates weight
differences and record it in §6.3b; or (ii) keep CUDA `expf` and record the
measured residual. Either way the result is written into this document before
chunk 3a/3b start, and no bit-identity claim is made on the strength of an
unmeasured assumption.

### 12.10 Constrained speculation tests

- Device mask application equals CPU `select_verification_sampled`
  (`constraints.rs:80-102`) on the same logits for the full grid, including
  `needs_mask == false` rows.
- All-zero mask → `EMPTY_CANDIDATES` on device; today's worker-error status is
  preserved (D1), not remapped.
- A constrained **greedy** row with a masked non-finite logit must error
  (strict mode, §4.1), matching `scores.rs::argmax`.
- `truncate_proposal` (`:51-60`) results are unchanged (still CPU).
- A constrained greedy row produces the same masked argmax as
  `select_verification` (`:61-73`).
- Rollback: a round that emits fewer tokens than it verified leaves the
  authoritative matcher exactly where the emitted prefix puts it
  (`scheduler.rs:77-87`).

### 12.11 Independent review requirement

- **Gate A (before any kernel):** this document, reviewed adversarially.
- **Gate B (per chunk):** §14 lists a review requirement per chunk.
- Additionally: the `reference_select` port in `cuda_selftest.cc` must be
  reviewed as a separate artifact from the kernel, by a reviewer who does not
  read the kernel first — otherwise a shared misreading of the contract passes
  both. This is the same reason the phase-0 oracle was built independently of the
  sampler (`REPORT.md:106-111`).

---

## 13. Measurement plan

Protocol discipline is inherited from the phase-0 harness and the v11 campaign:
explicit warmups, batched means plus per-call distribution, fixed seeds and
positions, and provenance capture (`REPORT.md:44-48`,
`README.md:214-232`).

**What the measurements are for, stated plainly.** After this work the **GPU
sampler — not the CPU sampler — defines served stochastic tokens**. Greedy is
unaffected (it consumes no draw and keeps the compact device path). The
GPU-vs-CPU token mismatch rate is a **published measurement**, not a claim of
bit-identity: it must be reported per phase-0 cell (§12.9) and in the release
notes, alongside the uniform-stream bit-identity result. A headline speedup may
never be published without that mismatch rate beside it.

### 13.1 Fixed-logit kernel latency vs CPU

- Extend `runs/gpu-sampling-phase0/harness` (path-dependency on the production
  `ds41rt-core`, `harness/Cargo.toml:9-11`) with a CUDA branch that uploads the
  six rows and drives the new entry point; or add an in-repo equivalent (§16 D4).
- Measure µs/row for every profile at row counts `{1, 6, 8, 16, 48, 64}`, and
  the full-round latency including the ids D2H and the existing sync.
- Report the pass-count effect directly: the ordered path's cost is expected to
  scale with nucleus width (`near_uniform`/`tied` push top_p/min_p to the full
  vocabulary, `REPORT.md:100-104`). Note that after the §4.5 correction the
  `top_k40` profile also pays the top-p search (`top_p = 1.0`), so its ~60-read
  budget must be measured, not assumed.
- Also measure the §6.3b `expf` variants and the §6.5.3 sequential-combine option
  here, so the chunk-2 decision is made on data.
- **Proposed gate (not measured yet):** the sampler+D2H must add ≤ 0.25 ms to a
  48-row ordered round and ≤ 0.10 ms to a 4-row round. If it does not, chunk 6
  (radix histogram) is mandatory before any E2E claim.

### 13.2 D2H bytes and synchronizations before/after

Count from code plus a device-side byte counter test:

| Item | Before | After |
| --- | --- | --- |
| non-compact round D2H | `rows × 517,120 B` (`scheduler.rs:401-403`, `independent.rs:143`) | `rows × 12 B` (ids, status, scores) |
| per-row host materialization | `rows × 517,120 B` host Vec (`scores.rs:47`, `:119`) | 0 |
| finishing frontier | `1 × 517,120 B` (`scores.rs:88`, `independent.rs:153`) | unchanged, now gated on caching being enabled (§10.4) |
| per-step syncs | head `wait()`/`synchronize()` + one pinned async D2H; single-lane adds a blocking pageable `cudaMemcpy` | head `wait()`/`synchronize()` + one pinned async D2H; nothing added |
| greedy compact round | `2 × rows × 4 B` (`v41_target_head.rs:367-369`) | unchanged |

### 13.3 End-to-end per profile on 1× RTX + 4× Spark with dSpark

- Use the existing campaign: `scripts/bench-ds41-release-decode.py`, validated
  by `scripts/validate-sampling-campaign.py`, aggregated by
  `scripts/aggregate-sampling-decode.py` — the same tools that produced
  `README.md:214-260`.
- Five profiles with all four filters stated explicitly (greedy control plus the
  four stochastic profiles), dSpark on, one deployment on the real topology.
- Matched workloads: the nonce-per-(profile, repeat, case) discipline and the
  explicit statement of that caveat (`README.md:280-288`); ≥5 discarded warmups;
  ≥3 repeats; report the median of per-repeat weighted ratios, not a single run.
- **Workload identity** is mandatory: equal weighted token totals and equal
  per-case median completion tokens across the compared datasets
  (`README.md:250-270`), so any gain is elapsed time, not output drift.

### 13.4 Single-request and batched verification shapes

- Concurrency 1 (exercises `single_lane_round`, `layout.rs:39-43`) and
  concurrency 16 (exercises `independent::run`, `layout.rs:33-36`).
- Draft limit 5 (head capacity 48) and 7 (capacity 64)
  (`cli.rs:705-706`, `v41_native_serve.rs:384`, `:415`).
- dSpark on and `--no-dspark` control, because the no-dSpark shape is exactly
  one row and one emitted token per step and isolates the sampler
  (`REPORT.md:216-221`).

### 13.5 Mixed and constrained regression checks

- A greedy-only lane next to a stochastic lane: assert the greedy weighted rate
  is unchanged within the noise floor.
- A mixed lane: assert the greedy rows' ids are exactly the CPU argmax over the
  same logits (using the retained-frontier cross-check).
- Constrained (JSON schema) runs at both concurrencies: assert no error-status
  change other than the documented `EmptyCandidates` decision.

### 13.6 Per-round timing on the TIMED campaign

Term (b) is a residual (§2.3, corrected fact 6). To identify it, run the
`ds41rt::timing` `"native scheduler round"` line (`scheduler.rs:487-492`)
during the timed campaign (not only the untimed debug pass) and split the round
into `target_forward_us`, `sampler_us`, `ids_download_us`, plus the existing
`draft_us`, `prepare_us`, `verify_us`. With the sampler on device, `sampler_us`
is a positive measurement rather than a residual, so accept/verify device time
becomes separable for the first time.

### 13.7 Explicit falsification criteria

The design has failed, or the uplift is not real, if any of these holds:

1. **Greedy regression:** weighted greedy tok/s drops more than the measured
   repeat noise floor (v11 greedy spread 4.34 tok/s on 95.46,
   `README.md:236-241`) — i.e. any drop beyond ~2%.
2. **No real stochastic uplift:** a profile's weighted tok/s does not improve by
   at least the *lower* bound of the measured host path
   (`0.545 ms/token` → ~+4.7% on the slowest stochastic profile). Failing to
   clear the min_p ceiling is a fail for min_p specifically.
3. **Sampler-cost blow-up:** measured sampler + D2H exceeds 0.4 ms on a 48-row
   ordered round, which would consume most of the top_k/min_p ceiling and make
   the top_p claim unreachable.
4. **Nucleus-width-dependent uplift inversion:** top_p0.95 improves by < 5%
   while min_p improves by ≥ 5% — evidence that the cost tracks nucleus width and
   the "zero-cost sampler" assumption is the dominant error.
5. **Determinism failure:** any row's id changes across batch size, ordering,
   lane, or peer activity; or a seeded replay differs from the documented RNG.
6. **Semantics failure:** a token differs from the CPU sampler for a reason
   traceable to a mask, `min_p`-threshold, tie-rule or ordering *bug* rather than
   the declared accumulation residual of §6.3c; or any tie/boundary case diverges;
   or `EmptyCandidates` falls back to an unmasked distribution.
7. **Unmeasured divergence:** the GPU-vs-CPU token mismatch rate per phase-0 cell
   is not measured and reported (§12.9) — publishing a benchmark without it is an
   attribution failure, because the design claims a *measured* divergence, and
   "not measured" cannot be distinguished from a bug.
8. **Workload drift:** weighted token totals or per-case median completion
   tokens differ between the compared datasets, so the timing comparison is not
   workload-identical.
9. **Attribution failure:** term (b) cannot be measured on the timed campaign,
   in which case no per-profile claim may attribute the gain to the sampler.
10. **Divergence escalation:** the measured per-cell mismatch rate is high enough
    to break a user-visible guarantee the release intends to keep (for example an
    A/B replay check or a documented steady-seed comparison). That does not fail
    the design by itself — measured divergence is authorized — but it forces the
    §6.5.2 order-independent (Gumbel-max) alternative back onto the table before
    release.

If (4) holds, the release must state the top_p uplift honestly as
"not yet demonstrated" rather than publishing +16.9% or any part of it. If (7)
holds, the release must not publish any seeded-reproducibility statement for the
GPU path at all.

---

## 14. Staged implementation plan

Each chunk is sized to be delegated whole, has its own acceptance gate, and
requires its own review before merge. Chunk 0 is this document.

| # | Chunk | Contents | Acceptance gate | Review |
| --- | --- | --- | --- | --- |
| 0 | **Design gate** | this document | adversarial review passes; corrected facts §2 accepted | required |
| 1 | **ABI + plumbing + GPU prepare/greedy** | `ds41rt_v41_sampler_row_t` in `v41_sampling_gpu.h`, mask arena, `TargetSamplingWave`, C-ABI registration, FFI wrappers + `validate_v41_sampling_buffers`, `execute_block_sampled` terminal, scheduler wiring, K1 with the greedy/constrained branch. **Stochastic rows temporarily keep the CPU path** and therefore still download full rows for those rows only | what chunk 1 can actually achieve: ABI validated end to end; greedy and constrained-greedy rows device-selected and exactly equal to the CPU masked argmax (including strict whole-row finiteness); compact greedy path byte-identical; **no full-row D2H for greedy rows**; no regression to compact greedy throughput; `cargo test` + `ds41rt_cuda_selftest` green. Removing the full-row D2H for stochastic rows is explicitly chunk 2/3a/3b work | required |
| 2 | **Fast path + device RNG + `expf` experiment** | K1 stochastic reductions with host `ln_min_p`, K2 contiguous-prefix scan with the no-crossing fallback, SplitMix64 device port, `total` diagnostic, the §12.9a `expf` experiment and the §6.5.3 sequential-combine evaluation | device RNG bit-equal on the **extended** seed/position grid (`2^63`, `u64::MAX`); `expf` outcome written into §6.3b (pinned variant or declared residual); fast-path token mismatch **rate recorded** per cell (not asserted zero); phase-0 `moderate`/`long_tail` histogram within TV noise | required |
| 3a | **Ordered path — top-k** | K3 pivot top-k, K4 exact tie prefix; stochastic `top_k` rows are completed by a **temporary CPU draw over the selected ≤k-token set** (≤ k×8 B download, exact because the set is tiny), so the top_k40 profile leaves the full-row path early | device top-k membership set equals the oracle's `ranked[..k]` for the full grid, all tie rows, `k ∈ {2,40,V-1,V,V+1}`, `top_p = 1.0` included; end-to-end `top_k40` token equality up to the declared accumulation residual; pass budget for K3+K4 measured against §13.1 | required |
| 3b | **Ordered path — top-p + draw** | K5 inclusive-prefix top-p (largest-key boundary, no `top_p = 1.0` shortcut, full-`S` fallback) and the rank-order draw with the `nucleus_count - 1` fallback; removes the temporary CPU draw | full parameter grid × masks × boundaries vs the ported `reference_select`; the §12.3 `top_k + top_p = 1.0` prefix-overshoot oracle; all tie cases; `-inf`/NaN/min_p sweeps; vocab-not-divisible-by-32; total pass budget measured | required |
| 4 | **Constrained path** | `prepare_verification_masks`, per-prefix mask upload, mask-first application, `EMPTY_CANDIDATES`/`MASK_WIDTH` status with **today's status preserved** (D1) | constrained device selection equals `select_verification_sampled`; all-zero mask hard-errors with the unchanged worker error; strict-mode masked non-finite errors; fork/commit/rollback unchanged; no per-position download | required |
| 5 | **Hot-path cleanup** | delete `BatchScores::new` CPU argmax from the hot path; delete the single-lane blocking `copy_d2h`; gate the frontier download on caching being enabled; trace-gated download for `logit_trace` | D2H byte accounting per §13.2; sync count unchanged; no `Vec` of 517 KB per row; retention tests green | required |
| 6 | **Pass-budget optimization (conditional)** | deterministic integer radix histogram (≤2048 buckets, fixed-order combine, u64 fixed-point mass) to cut K3/K5 passes; only if chunk 3b's measurement misses the §13.1 gate | measured pass budget ≤ the gate; determinism tests still exact; no new nondeterminism | required |
| 7 | **Measurement campaign + release/README** | fixed-logit kernel bench, timed E2E campaign with per-round timing, **the per-cell GPU-vs-CPU token mismatch rate**, README `:214-260` update, `docs/release-v11-performance.md` update, release-note wording per §6.4 | §13.7 falsification criteria, including criterion 7 (unmeasured divergence); workload identity; provenance/identity files; validated campaign | required |
| 8 | **Upstream/placement decision memo** | whether to vendor the kernel for upstream, expose the top-k tie rule as an option | written memo with the flip evidence of §3.3 | required |

Do not start chunk 3a before chunk 2's RNG **and** `expf` gates pass exactly: the
RNG equality test is the only cheap way to separate a draw-stream bug from a
filter bug, and the `expf` experiment is what tells the implementer whether the
remaining token differences are the expected residual or a real defect.

---

## 15. Risk register

| ID | Risk | Why it bites | Mitigation / evidence |
| --- | --- | --- | --- |
| R1 | **Provenance / pinning of reference algorithms** | FlashInfer is an unpinned `.venv` tree (no submodule, no lock; only a cache-identity hash at `scripts/kernel-cache-identity.py:159-170`). Copying code without recording provenance creates an unverifiable derivation | Port the *mechanism* as new code, keep a `.cuda` header comment naming each FlashInfer anchor, and record the anchors here. If a header must be vendored later, add a lock entry |
| R2 | **Tie-rule drift** | FlashInfer/vLLM keep all k-th ties (`sampling.cuh:945`); ds41rt keeps exactly k lowest-id (`target_sampling.rs:30-33`). A naive port silently changes every tied row | K3/K4 implement the total-order rule explicitly; §12.4 pins it; the reviewer diffs the `reference_select` port |
| R3 | **top-p tail-mass vs inclusive prefix, and boundary direction** | FlashInfer accepts the strict tail-mass criterion (`sampling.cuh:1072`), excluding the boundary tie; ds41rt is inclusive (`target_sampling.rs:544-554`). A monotonicity slip (`M` decreases as the key gets *better*) would select the whole vocabulary | K5 uses the **largest key with `M >= top_p`** = the last rank of the prefix; §4.5 states the direction and the bisection invariant; no `top_p = 1.0` shortcut; §12.3 boundary-uniform probes and the `top_k + top_p = 1.0` overshoot oracle |
| R4 | **Mask-width bug / out-of-vocab id** | The native path never masks the last word's high bits; a whole-word GPU reader could select `id >= vocab` on a non-divisible vocabulary | Bounded loops + host remainder-mask + kernel re-check (§5.3); vocab-33/127/129281 tests |
| R5 | **FP non-bit-identity** | `expf` is not bit-identical (1–2 ulp per weight) and the reduction order differs, so any comparison against an f32 accumulation can move the token — including the top-p membership, which shifts the draw by a rank; in the wide-nucleus case the mismatch rate can be O(0.1–1), not ULP-rare | `min_p` made exact by host `ln` (§6.3a); `expf` experiment and pin/declare in chunk 2 (§6.3b, §12.9a); measured per-cell mismatch rate required before any claim (§6.3c, §13.7 criterion 7); Gumbel-max alternative recorded and out of scope (§6.5.2) |
| R6 | **Graph-capture issues** | Kernel-node index patching is fragile (`sampling.cu:1203,1268`); captured row count is a key (`v41_target_head.rs:226`) | Sampler launched outside the graph; parameters from device memory; no cooperative launch; §11.3 |
| R7 | **Greedy regression** | Mixed-lane heterogeneity, an extra kernel launch, or per-row parameters could slow greed; forcing a lane-wide full download would be catastrophic | Compact path untouched; per-row params; §13.7 criterion 1; mixed-lane test |
| R8 | **Arch 120/121 build and AOT packaging** | Coordinator builds `CUDA_ARCH=120` (rewritten to `120f` only for the FlashInfer packed-MLA source, `native/CMakeLists.txt:317-319`) and Spark `121` (`build.sh:427`, `:558`); only one native arch per image | New plain `.cu` in `DS41RT_NATIVE_SOURCES` needs no AOT exporter; build and selftest on both images; verify the `120f` rewrite is not accidentally triggered |
| R9 | **Draft RNG co-existence** | `v41_dspark.cu:245-266` already has a Philox stream; mixing the target into it would break domain separation (`target_sampling.rs:182-186`) | **Verified**: the draft Gumbel branch is temperature-gated (`if (temperature != 0)` at `v41_dspark.cu:246`, `:262`) and production passes `0.0` (`speculative.rs:511-512`, `:613`), so no draft draw is consumed today. Target uses only SplitMix64 on `(seed, position)`; draft reservation semantics (`dspark_rng.rs:34-49`) untouched |
| R10 | **Evidence hygiene: `runs/` is gitignored** (`.gitignore:48`) | The phase-0 report, harness rows and campaign artifacts are not in the release tree, so a design or release claim can rest on an artifact a reviewer cannot fetch | Land the fixed-logit rows and the harness (or their sha256 manifest) in the repo before gating on them (§16 D4); keep release-page hashes as the fallback |
| R11 | **Pass-count blow-up** | The ordered path's ternary searches can reach ~74 row reads worst case (and `top_k40` now also pays the top-p search), which could cost more than the 0.55–1.79 ms it removes | §13.1 gate; chunk 6 histogram; §13.7 criteria 3 and 4 |
| R20 | **Masked-non-finite mode drift** | The strict (`scores.rs:160-163`) and permissive (`target_sampling.rs:1009-1024`) rules differ, and the daemon today is incidentally strict for stochastic rows too because `BatchScores::new` pre-validates; a careless unification silently changes one of them | Per-row `STRICT_FINITE` flag (bit4) set per call site; §4.1 records the delta; §12.6 pins both; D9 records the choice. The greedy/constrained path is never loosened |
| R12 | **Reduction nondeterminism** | f32 atomics or batch-dependent combine orders would break claim (b) | No f32 atomics; fixed contiguous segments and fixed tree combines; chunk 6's histogram uses integer/fixed-point counts |
| R13 | **`EmptyCandidates` status change** | `is_bad_request` has no consumer today, so mapping to 400 is a behavior change | Open decision D1; release note either way |
| R14 | **Mask upload pressure** | `rows × 16,160 B` per step (1.3 MB at 48 rows) on the head stream could add latency | One async H2D per step, not per row; measure in §13.1; masks only exist for constrained rows |
| R15 | **Loss of logit-trace observability** | Removing the full-row download would silently break `ds41rt::logit_trace` (`scheduler.rs:565-576`) | Explicit trace-gated per-row download (§8.5); the trace already forces non-compact (`:458`) and is documented untimed |
| R16 | **Frontier retention regression** | Device-side retention or a missing row would break exact-frontier restore (`scores.rs:195-206`, `prefix.rs:248-…`) | Keep the host `TokenScores` and the one-row download (§10); retention tests stay green |
| R17 | **CPU fallback divergence** | If the CPU path can serve the same seeded request, two token streams exist in one build | D2 recommendation: oracle + explicit diagnostic override only; fail closed when the symbol is missing |
| R18 | **Single-row (`rows = 1`) shapes** | The dSpark-off shape and the first-token path (`scheduler.rs:324`) are common and must not pay a launch-heavy kernel path | Fast path is 4 row reads; measured explicitly (§13.4) |
| R19 | **`logits_stride` assumptions** | The head buffer is contiguous today (`STRIDES[4] = 517120`, `v41_target_head.rs:12`, `output()` fixes bytes at `:387-404`), but a future layout could be padded | ABI takes `logits_stride` and validates it |

---

## 16. Open decisions with recommendations

**D1 — `EmptyCandidates` / `MASK_WIDTH` HTTP status.**
*Decision (planner): preserve today's behaviour in this project — worker error
(HTTP 500), no remap.* The device still reports the status; the host keeps the
current error strings. Mapping `is_bad_request` to `BadRequest`/400 is recorded
as a **separate follow-up item** with its own review and release note, because it
changes an observable HTTP surface and is unrelated to whether sampling runs on
the device. `MASK_WIDTH` is a caller fault too but is likewise left as-is.

**D2 — CPU-sampler fallback retention.**
*Decision (planner): retain as the reference oracle **and** as an env-gated A/B
fallback (`DS41RT_TARGET_SAMPLER=cpu|gpu`), never the default serving path.*
Default `gpu`; a missing native symbol fails at startup (`DEVELOPER.md:67-68`),
and there is no automatic fallback. State explicitly that the fallback must
**not** silently reintroduce the full-row download on the default path: `cpu`
selects the whole CPU route (including its downloads) only when an operator asks
for it, and the default path never probes the CPU sampler on a miss. The A/B
fallback is documented as possibly changing accumulation-residual tokens within
the same build.

**D3 — Diagnostics surface.**
*Recommendation: `out_indices`, `out_status`, `out_status_detail`, and (for
greedy rows) `out_scores` are required; `out_total` and `out_nucleus_count` are
optional behind the diagnose flag and used by the oracle; no per-row
probability; no device top-two — the trace keeps its own gated download.* This
matches what production actually reads (`scores.rs:48-51`, `:120-123`) and keeps
`RankedSample` internals test-only.

**D4 — Add a fixed-logit GPU harness to the repo.**
*Recommendation: yes.* Add the six phase-0 rows (or a generated equivalent with
committed sha256) plus a device-driving test under the repo, because `runs/` is
gitignored and a measurement gate that depends on an unfetchable artifact is not
a gate. Reuse the phase-0 generation rules (documented in
`runs/gpu-sampling-phase0/harness/src/main.rs` and the manifest) so the CPU
histogram reference is reproducible.

**D5 — Capture the sampler into the target graph.**
*Recommendation: no in the first version.* Launch it after `cuda_graph_launch`
like the greedy argmax kernel (`v41_target_head.rs:363-369`); revisit only if
launch overhead is measured to matter, in which case bucket by row count.

**D6 — Fixed-point deterministic mass histogram (chunk 6).**
*Recommendation: defer to chunk 6 and gate on measurement.* If the ordered-path
pass budget misses §13.1, use a ≤2048-bucket histogram with deterministic
integer counts and u64 fixed-point mass (scaled by 2^40, saturating), combined
per CTA in a fixed order. Do not use f32 atomics under any circumstances (R12).

**D7 — `top_k == 1` routing.**
*Recommendation: keep it greedy and route it to K1/compact.*
`is_greedy()` includes `top_k == Some(1)` (`target_sampling.rs:172-174`), so it
must never enter K3/K4/K5. The API already maps `top_k = 1` to a greedy request.

**D8 — ABI placement.**
*Decision (planner): shared header under `native/cuda/kernels/` plus C-ABI
registration and a validator.* Concretely: `native/cuda/kernels/v41_sampling_gpu.h`
holds the `ds41rt_v41_sampler_row_t` struct and the kernel-facing declarations;
`native/include/ds41rt_native.h` gets the two `extern "C"` entry-point
declarations next to the existing argmax/sampler declarations (`:1671-1680`); and
`rust/crates/ds41rt-ffi/src/lib.rs` gets the wrappers plus
`validate_v41_sampling_buffers`, consistent with
`validate_logits_sample_topk_topp_buffers` (`:17859-17896`).

**D9 — Masked-non-finite default for stochastic rows.**
*Decision (planner): mode-specific — greedy and constrained rows keep the strict
whole-row check, stochastic rows keep the permissive masked rule; pin both with
tests.* Adopted in §4.1 with the per-row `STRICT_FINITE` flag (bit4) and pinned
in §12.6. Recorded caveat: today's daemon is *incidentally* strict for stochastic
rows too, because `BatchScores::new` pre-validates each materialized row with
`argmax(row, None)` (`scores.rs:67-71`, `:160-163`). Adopting the permissive rule
is therefore a deliberate, narrow loosening for masked-non-finite stochastic
rows; it must be in the release notes. The flag exists so that a reviewer who
prefers zero observable change can set strict for stochastic rows with a one-line
host change, and so that the constrained/greedy path can never be loosened by
accident.

---

## Appendix A — two primitives, precisely

**A.1 The probe primitive.** One pass over the row, with the mask and survivor
predicates applied inline, returns two probes' aggregates. Both `C_gt` and `M`
are **non-increasing in `K`**: a larger key admits fewer tokens.

```
probe(k0, k1):                       # k0 < k1 in key order (larger key = better)
  C0 = 0; C1 = 0; M0 = 0; M1 = 0
  for t in thread's tokens:          # strided; vocab-bounded
    if not survivor(t): continue
    key = order_key(scaled(t))
    p   = expf(scaled(t) - max_scaled) / total   # normalized, as CPU :540-542
    if key > k0: C0 += 1; M0 += p
    if key > k1: C1 += 1; M1 += p
  reduce (C0, C1, M0, M1) in a fixed tree order
  return (C0, M0, C1, M1)
```

The un-normalized form (`M += expf(...)` with `target = clamp(u) * Σ_nucleus w`)
is algebraically equivalent but is deliberately **not** used, because the CPU
divides each weight by `total` before summing (`target_sampling.rs:540-542`) and
the design matches that operation order (§4.5, §6.3c). `total` is available from
K5's preceding sum pass; K3 uses the count outputs only.

**The two searches run in opposite directions, and that is the trap.**

- **top-k** — predicate `C_gt(K) < k` is satisfied for large `K` (few tokens
  above), so the satisfying set is *upward-closed* and the boundary is its
  **smallest** member: the k-th value `v_k` = the smallest key with
  `C_gt(K) < k`. Invariant: `rejected < v_k <= accepted`; a satisfying probe
  moves `hi` **down**; answer `hi`.
- **top-p and draw** — predicate `M(K) >= threshold` is satisfied for small `K`
  (the prefix contains the best tokens), so the satisfying set is
  *downward-closed* and the boundary is its **largest** member:
  `K_b = max{K : M(K) >= threshold}`, the boundary (worst) rank of the prefix.
  Invariant `M(lo) >= threshold > M(hi)` with `lo = key(-inf)`,
  `hi = key(+inf)`, answer in `[lo, hi)`; a satisfying probe moves `lo` **up**.
  This applies to `threshold = top_p` and to `threshold = target`, and the
  tie-level id search uses the same direction (the u64 key packs `~id` so larger
  key = lower id).
- **Fallbacks**: if a top-k search terminates without an accepted key (only
  possible at a degenerate `k` cap), report `INTERNAL`; if the top-p search finds
  no satisfying key (the normalized total rounds below `top_p`), use the full
  retained set `S` (§4.5); if the **draw** search finds no satisfying key, use
  the nucleus's last rank `K_p` — the CPU's `selected = nucleus_count - 1`
  initialization (`target_sampling.rs:558`).

All three share one kernel body parameterised by (count-only | mass), the search
direction and the acceptance predicate. Two probes per pass form a **ternary**
search — one pass keeps one of three sub-intervals, so the pass count is `~log_3`
rather than `~log_2`; it is not a halving per pass. This is the FlashInfer
dual-pivot pattern (`sampling.cuh:930-945`, `:1050-1072`), re-aimed at ds41rt's
rules.

**A.2 The ascending inclusive scan (K2 and the K4 tie prefix).** With
`T = 256` and `L = ceil(vocab / T)`:

```
seg_t   = [t*L, min((t+1)*L, vocab))
local_t = sequential prefix sums of w over seg_t in ascending token order
total_t = local_t[L-1]                       # 0 if empty
excl_t  = fixed-tree exclusive scan of total_t
# find the minimum token id whose inclusive cumulative reaches the target
for each t: walk seg_t accumulating from excl_t; report the first token crossing
answer  = min over reporting t
```

K2 sums raw `w = expf(scaled - max_scaled)` because the CPU fast path does **not**
normalize (`target_sampling.rs:597-618`); in K4 the scanned quantity is the
indicator "token equals the k-th value", and the answer is the admit boundary for
the lowest-id tie prefix.

Deterministic for any grid shape; order-exact within a segment; the cross-segment
tree is one of the two accumulation-order differences from the CPU (the other is
`expf`) — §6.3c.
