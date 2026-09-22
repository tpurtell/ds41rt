# GPU target-sampler design (implementation gate)

**Status:** design gate. **Chunks 1–5 are delivered and reviewed.** Chunk 1 is
`9dffad0` (§17); chunk 2 `f0e7902` (§18); chunk 3a `90f745c` (§19); chunk 3b
`ad9ea72` (§21); chunk 4 is split — **4a committed as `3572101`, 4b committed as
`b38e910`**, both recorded in **§22** (which also answers §20's hand-off
list); **chunk 5 / 5b is delivered as `714fc1d`** with its validation record in
**§22.7** and its harness-power record in **§12.14**. The normative sections below
have been reconciled with all six (notably §4.1, §4.2–§4.5, §5.4–§5.5, §6.3,
§8.2–§8.4, §9.2, §10.4, §12.4, §12.6, §12.12–§12.14, §13.1, §13.2, §16, §20,
§22.7). **Chunks 6/7 (pass-budget optimisation and the published measurement) and
the phase-3–4 campaign are next**; they remain design-only and still require the
adversarial review of §14.

**Repository revision read:** `e1f5d495b5a82fb7ddad8514cad419b6ae62c0cc` (`e1f5d49`).
Chunk 1 reads `9dffad0`, chunk 2 `f0e7902`, chunk 3a `90f745c`, chunk 3b
`ad9ea72`, chunk 4a `3572101`, chunk 4b `b38e910`, and chunk 5/5b `714fc1d`.

**Deliverable rule:** the original design task produced exactly this one file;
the as-built updates are design text only. Every claim about existing code
carries a `path:line` anchor or a commit anchor; every number is labelled
MEASURED (with its artifact), COMPUTED, ARITHMETIC or UNKNOWN.

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
   (§13, §14 Phase 3–4).

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
| 5 | The recoverable amount is **0.55–1.79 ms/token** (host sampler + D2H/argmax/materialization) on a measured 10.5–12.3 ms/token, i.e. ceilings of **+5.1% to +16.9%** on the four stochastic profiles, greedy untouched. These ceilings are **OPTIMISTIC**: they assume a zero-cost sampler, part of them is derived from cross-host microbench numbers, and greedy's 10.517 ms/token is not a clean shared device floor because lower acceptance means more target-forward passes per emitted token (greedy accepted fraction 0.724 vs 0.326–0.548) | none; this is the correction of the phase-0 report's own framing | §2.3 states the ceilings with caveats; §13 discounts top_p0.95 hardest; §14 chunk 6/7 exists specifically to cut pass cost if the kernel is not near-zero-cost |
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
     (`scores.rs:407-420`, the `ensure!` at `:415`) — a greedy/constrained row
     must reject a non-finite
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
    as `scores.rs::argmax` (`scores.rs:407-420`, the `ensure!` at `:415`);
  - stochastic rows: permissive on masked tokens, exactly as
    `target_sampling.rs:430-436` (the CPU skips masked tokens *before* the
    finiteness check), pinned by `target_sampling.rs:1009-1024`.
  - **The shipped host now builds exactly that (chunk 5b).**
    `build_target_sampling_plan` sets `STRICT_FINITE` **only for greedy rows**
    (`if greedy`, `scheduler.rs:725-735`); the earlier `needs_mask || greedy`
    condition was the design/wiring divergence chunk 5b found and fixed. The
    daemon's recording double mirrors the same rule (`v41_target_head.rs`,
    `nonfinite_where_allowed`; test
    `the_recording_double_is_strict_only_where_the_row_can_act`).
  - **As built, `STRICT_FINITE` is functional and additive.** A row that carries
    `flags.bit4` keeps the permissive pass's results and additionally runs a
    whole-row finiteness pre-scan; it is not a replacement branch. The host sets
    the bit only for greedy rows (`v41_sampling_gpu.h`; validator
    `lib.rs:18252-18260`), so a strict stochastic row is possible and would
    report `NONFINITE_LOGIT` while still computing the same `max_scaled` /
    survivor state. The device selftest pins both modes and the additive
    behaviour (`native/tests/v41_sampling_selftest.cu`, "masked non-finite is
    mode-specific and STRICT_FINITE is functional").
  - **Recorded behaviour delta — REALIZED in the shipped plan (chunk 5b).** The
    daemon rejects a masked non-finite value only on the **whole-round CPU
    path**: `execute_logits` → `BatchScores::new` (`scheduler.rs:1119`) and the
    independent lane's equivalent branch (`independent.rs:161`), whose
    `argmax` applies `ensure!(value.is_finite())` *before* the mask test
    (`scores.rs:407-420`). Those paths are reached only for a `compact`
    all-greedy round, a round with **no** device-servable row, or a layout
    without the sampled terminal. On the **device** path the finiteness decision
    was the host `needs_mask || greedy` flag; chunk 5b aligned that flag to the
    CPU arbiter (`target_sampling.rs:430-436`), so a masked-out non-finite value
    on a stochastic device-selected row is now **accepted and returns the token
    the CPU sampler returns**. That is a deliberate, documented,
    consumer-visible loosening (release note + D9); it never loosens the
    greedy/constrained path, and both behaviours are pinned by §12.6. The revert
    is a one-line host change (restore `needs_mask || greedy`).
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
- **Algorithm — chunk-3b rewrite (replaces the chunk-2 tree combine; the
  Hillis-Steele f32 inclusive scan is no longer called by K2)**
  1. Two passes within the CTA: (i) per-segment sequential local sums
     `local(i)` over `w_t = expf(scaled_t - max_scaled)` for survivors; (ii) ONE
     consistent sequential segment prefix built by **thread 0**,
     `C(0) = 0`, `C(i+1) = fl(C(i) + local(i))` — at most `kBlock = 256`
     shared-memory adds, with barriers before and after — and
     `total = C(kBlock)`, `total = fmaxf(total, 1e-20f)`,
     `target = clamp(u, 0, MAX_UNIFORM) * total`.
  2. Pass 2 may report from a segment only when its start is strictly below the
     target (`may_report`: `C(i) < target`), and inside it a token is a crossing
     only when the cumulative **strictly before** the token is `< target`
     (`cumulative_before < target`). Because `fl(before + w) >= target > before`
     forces `w > 0` in f32, **a zero-weight token is structurally unreachable for
     `target > 0`**. `target == 0` still selects the first survivor, matching the
     CPU. The retained per-segment sums, `tree_max_u32(lasts)` and
     `tree_min_u32(hits)` remain.
  3. **Saturation fallback:** if the bracketing segment
     (`C(i) < target <= C(i+1)`) cannot reach `target` because its own in-segment
     walk association rounds the tail away, it reports its **first positive-weight
     survivor**; `tree_min_u32` keeps the earliest report. That segment always
     contains a positive weight, so the fallback is positive. The old
     owner-segment re-walk (which derived `total` from the tree prefix) is gone.
  4. Write `out_indices[r]` and `out_total[r] = total` (diagnostic).
- **Invariant established by the rewrite (MEASURED).** No zero-weight selection
  for `u > 0`. Witness (chunk-3b `k2witness`): vocab 2048, `T = 1`, `top_p = 1`,
  `min_p = 0`, `logits[0] = 0`, `logits[8..15] = -17`, the rest `-200`, seed
  `0x9216a62488dbaa7b`, position 0 (`u = MAX_UNIFORM`): the pre-fix tree build
  selected token **16** with weight `expf(-200) = 0`; the fixed build selects
  token **8** with weight `4.13994e-08` (positive), while the CPU sequential
  oracle selects token **0** — a documented residual, not an equality. Pinned by
  `test_k2_zero_weight_invariant`; mutant **M6** (pre-fix walk) fails it
  (`runs/chunk3b-scratch/REPORT.md` §2).
- **CORRECTION — the chunk-2 wording "one consistent `W`, and the minimum over
  reports is exactly the first crossing token" is no longer literally true.** The
  in-segment walk re-adds weights from `C(i)` with a *different* f32 association
  than the fold that produced `C(i+1)`, so in the saturation case the reported
  token's own cumulative can be **below** `target` and the draw is **non-monotone
  in `u`**. This is device-confirmed (chunk-3b review, `/tmp/advrev/k2mono.cu`; row
  `vocab = 1024`, `token0 δ=0`, `token4 δ=ln(0.4·2^-23)`, `token5 δ=ln(1.4·2^-23)`,
  `tokens6,7 δ=ln(0.4·2^-23)`):
  `u = 0.9999997615814209 → token 5`, and the **larger**
  `u = 0.99999988079071045 → token 4` (backward), both with status OK. The
  affected range is the last few ulps of the uniform (≈`1e-7` of the mass), i.e.
  rows whose tail weights are below half an ulp of the running sum. It is a
  **measured, bounded, declared residual**: every selection is still a valid
  survivor with strictly positive f32 weight for `u > 0` — an independent
  983,040-draw scan found **0/983,040** zero-weight or non-survivor selections,
  and in every cell `TV(device, oracle) <= TV(cpu, oracle) + Monte-Carlo noise`.
  A cheap correct fix, if one exists, is a **chunk-6 candidate**; the design does
  not claim monotonicity in `u` for K2.
- **Why contiguous segments**: they preserve the CPU's ascending accumulation
  order *within* a segment, and the chunk-3b fold makes the *cross-segment*
  combine a single sequential chain as well; the remaining difference from the
  CPU is the in-segment walk association in the saturation case above plus the
  `expf` difference (§6.3b), both acknowledged in §6.3c. The strictly sequential
  token-order combine was implemented and measured in chunk 2 (§6.5.3) and is not
  used.
- **Limits**: `top_p >= 1.0` and `top_k` disabled only; `MAX_UNIFORM` is
  `0x3F7FFFFF` written as a bit pattern, not a decimal literal;
  `DS41RT_V41_K2_NO_WALK_TOTAL` is now vestigial (kept so old measurement scripts
  compile); `DS41RT_V41_K2_SEQUENTIAL_COMBINE` remains a compile-time diagnostic
  switch, never an ABI field.
- **Measured cost and divergence after the rewrite (chunk 3b, MEASURED).** The
  rewrite is **faster**: 392.6 → **305.0 µs** per call at 4 rows and 404.9 →
  **314.8 µs** at 48 rows (−22.3%), with the K2-non-applicable baseline unchanged
  (176.9 → 176.3 µs at 4 rows, 183.1 → 182.6 µs at 48 rows), because the ≤256-step
  serial fold plus the retained reductions are cheaper than the removed owner walk
  + tree scan. Full table in §13.1. The fast-path GPU-vs-CPU divergence on the
  **same 983,040-draw harness** moved 119,138/983,040 = **12.1193% →**
  121,047/983,040 = **12.3135%** (per-row: `near_uniform` 57.7502 → 59.1241%,
  `long_tail` 12.3169 → 12.0947%, `peaked` 2.0184 → 2.0294%, `moderate`
  0.6293 → 0.6317%, `tied` 0 → 0%, `trace_anchored` unchanged; 21/120 cells
  changed). **The two figures are not apples-to-apples**: the earlier one included
  the invalid zero-weight selections this rewrite removes, and the independent
  validity scan confirmed `TV(device, oracle) <= TV(cpu, oracle)` + noise in all
  120 cells — so the increase is a legitimate mapping difference, not residual
  invalidity. **The chunk-6 number to publish is 12.3135%.**
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
  2. Terminate when the interval collapses to one key. **Correction (chunk 3a):
     the true worst case is 21 passes, not the first revision's "≤ 20"**, and the
     termination condition is the load-bearing part. The loop probes while
     `hi - lo >= 2` and, when `hi - lo == 2`, takes the single final probe
     `m0 = lo + 1` (`m1 = hi`, already known accepted), stopping at `hi - lo == 1`.
     The 32-pass cap remains as a defensive bound, reporting `INTERNAL` if hit.
     Write `kth_value = ordered_bits_to_float(v)`, `above_count = C_gt(v)`.
     **Termination bug found during implementation and fixed:** the first version
     stopped at `hi - lo < 3`, but the middle branch of a length-4 interval
     produces a length-2 interval (`lo' = lo+1`, `hi' = lo+3`), and when a survivor
     key sits at `lo+2` the true k-th key is `lo+2` while the loop returns
     `hi = lo+3`. `C_gt` stays correct but no survivor carries `kth+1`, so the tie
     cut admits nothing and the retained set comes up short. A host simulation
     over random key sets failed **261/20,000** cases with the `>= 3` termination
     and **0/20,000** with the fix (the reviewer's independent generator measured
     131/20,000 for the same mechanism). The `hi - lo == 2` final probe is
     **load-bearing** — removing it reintroduces the bug — and seven concrete keys
     are pinned as a regression by `test_k3_k4_length_two_interval`.
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
- **Passes**: **21** worst case with two probes per pass (ternary search over a
  32-bit key) against the 32-step defensive cap; **corrected in chunk 3a from the
  first revision's "≤ 20"**, confirmed three independent ways — a host interval
  recurrence, a randomized host simulation, and device measurement of 21
  (`runs/chunk3a-scratch/REPORT.md` §2, §5). The ternary middle branch is
  `L − 2·floor(L/3)`, which is why the bound is not 20.
- **No sort**: this replaces both the `real_full`-style
  `cub::DeviceSegmentedRadixSort` over `rows × vocab` (`sampling.cu:1739-1742`)
  and the single-thread serial top-k (`sampling.cu:598-676`).

### 4.4 K4 — `v41_sample_topk_membership_kernel`: exact tie prefix, rank-ordered output

- **Inputs**: `kth_key`, `above_count`, `target_tie = k - above_count`,
  survivors predicate.
- **Algorithm** `[NEW]` (as built in chunk 3a): single pass over `[0, vocab)` in
  256 contiguous segments; each thread counts `above` (`order_key > kth_key`) and
  `equal` (`== kth_key`) survivors; a fixed-tree exclusive scan of the equals
  gives `eq_before`; each thread admits equal-key survivors while its running
  index is `< target_tie`. Correct for any multiplicity (thousands tied, whole
  row tied) and for `vocab % 32 != 0`, with no value-carrying sentinel — every
  decision is an integer count or a scan, so the chunk-2 `tree_max_u32` sentinel
  class cannot recur.
- **DEVIATION, recorded as implemented: K4 emits a rank-ordered id list, not a
  bitmap.** The retained set is written to `rank_order_ids` in the CPU's **exact
  rank order** — descending `order_key(scaled)`, ties ascending token id, with
  `-0.0` canonicalized to `+0.0` exactly as `descending_radix_key` does
  (`target_sampling.rs:283-291`, `:323-344`; the same permutation the CPU's
  comparison sort and stable LSD radix sort both produce). This replaces the
  first revision's per-row `topk_bits[vocab/32]` bitmap. The list subsumes the
  bitmap for membership (a scan or a direct rank lookup) and gives K5 the CPU's
  inclusive-prefix nucleus and rank-order draw directly, in **O(k) rather than
  O(vocab)** per-row memory; the production region is the §11.1 top-k arena.
- **Rank assignment is O(k²/256) counting, not a sort.** Entries are compacted in
  token order into a u64 staging arena as `(order_key << 32) | id`; each entry's
  rank is the number of staging entries strictly better under the u64 total
  order. Nothing is sorted. Scaling (COMPUTED, tests in
  `runs/chunk3a-scratch/REPORT.md` §5): a few hundred comparisons at the served
  `k = 40`; fine to `k ≈ 1000` (the tests materialize `k = 1000` at vocab
  129,280 — ≈1M comparisons, sub-millisecond); sluggish by `k ≈ 4093`; unusable
  near `k ≈ vocab`. **Wide-k coverage is therefore selection-only, and the
  chunk-6 histogram owns the large-k case.**
- **Arena sizing and the k-beyond-capacity fallback are chunk-4 decisions.**
  `rank_order_capacity == 0` selects a selection-only call (null arenas,
  `kth_value_bits`/`above_count` still published); a non-zero capacity must be
  `>= max_r params[r].top_k`, enforced by the FFI validator and re-checked
  in-kernel as `capacity >= total_retained`.
- **Passes**: 1 membership pass over the vocabulary + 1 rank-placement pass over
  the k-sized staging set (no vocabulary-sized second pass).

### 4.5 K5 — `v41_sample_nucleus_kernel`: inclusive-prefix top-p + rank-order draw

**Entry point and interface (fixed by chunk 3a; K5 lands in chunk 3b on top of
it).** The K3/K4 entry point is exactly:

```c
ds41rt_status_t ds41rt_cuda_v41_topk_select_async(
    const float* logits, size_t rows, size_t vocab, size_t logits_stride,
    const ds41rt_v41_sampler_row_t* params, const uint32_t* mask_words,
    size_t mask_words_per_row, uint32_t* rank_order_ids,
    uint64_t* rank_order_scratch, size_t rank_order_capacity,
    uint32_t* out_retained_count, uint32_t* out_pivot_passes,
    ds41rt_v41_sampler_scratch_t* scratch, void* cuda_stream);
/* plus the blocking ds41rt_cuda_v41_topk_select(...) without cuda_stream */
```

- `scratch` must already hold a completed K1 pass (the chunk-2 convention), so K5
  is a third kernel on the same stream after K1 and K3/K4.
- K3/K4 write **block row `b`**'s retained ids at
  `rank_order_ids + b * rank_order_capacity`, rank 0 = best, in exact CPU rank
  order; `rank_order_scratch` is the matching u64 staging (`rows × capacity`),
  scratch only, never observable. `out_retained_count` / `out_pivot_passes` are
  indexed by `output_row` like the other ABI outputs (the daemon makes the two
  equal).
- **O(k) memory**: `4 B` per id + `8 B` staging, no vocabulary-sized structure.
- `rank_order_capacity == 0` means **selection-only** with null arenas; a
  non-zero capacity must be `>= max_r params[r].top_k` (FFI validator enforces,
  kernel re-checks `capacity >= total_retained`).
- `scratch[r].kth_value_bits` / `above_count` are published.
- **Eligibility matches the CPU exactly** (`target_sampling.rs:477-501`): K1
  `status == OK`, not greedy (`temperature < 1e-5 || top_k == 1`), `top_k != 0`,
  `top_k < survivor_count`. `top_k >= survivor_count` and `k > vocab` are
  **no-ops** (K5 must then treat the retained set as *all* survivors);
  `k == 1` is greedy and never enters K3/K4; `top_k == 0` with `top_p >= 1` is
  the disjoint K2 fast path.
- K5 consumes the list as a plain rank-ordered id array: the inclusive-prefix
  nucleus and the rank-order draw iterate ranks `0..k-1`, recomputing
  `scaled = logits[id] * inv` and `w = expf(scaled − max_scaled)` per rank
  exactly as the CPU's `sample_from_ranked` does.

**K5 entry point (as built and adversarially verified in chunk 3b):**

```c
ds41rt_status_t ds41rt_cuda_v41_nucleus_async(
    const float* logits, size_t rows, size_t vocab, size_t logits_stride,
    const ds41rt_v41_sampler_row_t* params, const uint32_t* mask_words,
    size_t mask_words_per_row, const uint32_t* rank_order_ids,
    size_t rank_order_capacity, const uint32_t* rank_retained_count,
    uint32_t* out_indices, uint32_t* out_status, float* out_total,
    uint32_t* out_nucleus_count, ds41rt_v41_sampler_scratch_t* scratch,
    void* cuda_stream);
/* plus the blocking ds41rt_cuda_v41_nucleus(...) without cuda_stream */
```

- **`output_row == block_row` is REQUIRED for every K5-class row**, checked
  **unconditionally and before `rank_retained_count` is read**; a violation writes
  `INTERNAL` to both `out_status` and `scratch.status` and writes no token. (The
  earlier *conditional* guard was bypassable: a raw-C permutation witness made
  block 0 read K4's count for the other row, silently fall into the survivor
  domain and write a wrong token with status OK — `ADVERSARIAL-REVIEW-rev3` §4.)
  K1–K4 still permit scatter; only K5's block-row consumption needs identity, and
  the FFI validator rejects a non-identity batch outright.
- **`out_status` is a caller-visible channel** (pass K1's buffer). K5's loud
  per-row statuses reach it, not just scratch: non-finite `top_p`, zero-survivor
  non-greedy rows, `top_k ∈ [257, survivor_count)` (unsupported, loud INTERNAL),
  non-identity/output-row violations, and capacity-0/wide shapes. The entry point
  still returns `OK` for a per-row failure; the caller must read `out_status`.
- The K3/K4/K5 **FFI wrappers take a validated `params_device` buffer and launch
  its pointer**, exactly as K1 does (§5.5, §11.1); passing a host
  `params.as_ptr()` as a device pointer is a protocol violation. This closes the
  chunk-1 defect class that K3/K4 predated and K5 had reintroduced
  (`ADVERSARIAL-REVIEW-rev3`, mandatory item).
- **Verified K5 fixes from the three chunk-3b review rounds:**
  (a) the **mass-probe shared-memory race** (the reduction returned `phase[0]` to
  all threads with no barrier before the next probe overwrote it) is fixed by a
  snapshot + trailing barrier mirroring K3, pinned by a 32-launch bit-identical
  determinism test (`test_k5_repeated_run_determinism`); no other shared-array
  re-read hazard exists in K3/K4/K5.
  (b) the **probe-dependent prefix association** (bit-level `0x3f800001` vs
  `0x3f800000` for weights `[1, 2^-24, 2^-24]`) is fixed by one fixed sequential
  prefix per row read by every search. The prefix is bounded by
  `retained <= kBlock = 256` (**not** ~129k), so the production-shape cost is
  **+1.6 µs at `top_k = 40` and +4.5 µs at `top_k = 256`** (0.07%/0.19%) — no
  chunk-6 impact. A fixed-but-parallel association would be probe-invariant but
  **not** CPU-bit-identical, so the tid0 sequential walk is load-bearing for the
  retained-domain exactness in §6.3c.
  (c) the **zero-uniform non-member** bug (`u = 0` is reachable, e.g. seed
  310147/position 0) is fixed by requiring `mass > 0`; `u = 0` now selects rank 0,
  the CPU's choice.

- **Inputs**: survivors, the rank-ordered retained id list (if K3/K4 ran),
  `total`, `max_scaled`, `top_p`, `uniform`, `top_k` state.
- **Set `S`**: survivors ∧ the K4 rank-ordered retained membership (all survivors
  when K3/K4 were a no-op). Weights
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
- **Passes (ordered path, uncompressed)**: K1 2 + K3 ≤21 (ternary over a 32-bit
  key, §4.3) + K4 1 + rank placement O(k²/256) over the k-sized set + K5
  (top-p ≤20 value + ≤11 tie + draw ≤20 worst / typically ≤4)
  ≈ **up to ~74 row reads** worst case, ~35 typical. ARITHMETIC at 517,120 B/row:
  up to ~38 MB/row, ~1.8 GB for 48 rows. **This is the central cost risk**
  (§13.1 gate, §14 chunk 6/7) and the reason the ceilings in §2 are optimistic.
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
  uint32_t flags;       /* +36 bit0 GREEDY; bit1 DIAGNOSE; bit2 NO_MASK; bit3 ORACLE_CROSSCHECK; bit4 STRICT_FINITE */
  uint32_t output_row;  /* +40 row index into logits/out_* (equals the struct index in practice) */
  float    ln_min_p;    /* +44 HOST-precomputed ln(min_p); negative infinity when min_p == 0.
                              Hard requirement: the device never calls logf (see §6.3a). */
  uint32_t reserved0;   /* +48 must be 0 */
  uint32_t reserved1;   /* +52 must be 0 */
  uint64_t reserved2;   /* +56 must be 0 */
} ds41rt_v41_sampler_row_t; /* exactly 64 B */
```

`flags` bits (as built in `v41_sampling_gpu.h`; constants
`DS41RT_V41_SAMPLER_FLAG_*`):

| bit | name | meaning |
| --- | --- | --- |
| 0 | `GREEDY` | host resolved `temperature < 1e-5 \|\| top_k == 1`; the kernel re-derives it and ORs, so a host bug cannot turn a stochastic row greedy |
| 1 | `DIAGNOSE` | optional `out_total`/`out_nucleus_count` are written |
| 2 | `NO_MASK` | **the row has no meaningful mask bits**; the kernel treats every `t < vocab` as allowed and never reads the arena. Requires `mask_row == 0xFFFFFFFF` |
| 3 | `ORACLE_CROSSCHECK` | diagnostic only; unused in production |
| 4 | `STRICT_FINITE` | whole-row finiteness for a row that would otherwise take the permissive branch; the host sets it for greedy and constrained rows only (§4.1) |

**`bit2` is `NO_MASK`, not "mask remainder-masked"** (the first revision's
wording). The remainder rule is not signalled by a flag at all; it is enforced
unconditionally host-side immediately before upload by
`ds41rt_v41_sampler_clear_remainder` (§5.3). `NO_MASK` exists because
`fill_bitmask`'s `needs_mask == false` is the production signal for an
unconstrained row (`constraints.rs:44-46`) and nothing else told the kernel to
skip the arena. The kernel additionally treats `mask_row == 0xFFFFFFFF` as
unconstrained even without the flag, so a raw-C caller that follows only §5.2
cannot index the arena out of bounds; the **host validator is stricter and
requires the flag and the sentinel to agree** (§5.4).

`top_k` is **not range-validated yet** (only `top_k == 1` participates in the
greedy derivation). That is harmless for chunk 1, which never selects with
`k`; a full `top_k ∈ {0} ∪ 1..=vocab` validation is a chunk-2+ item and is
recorded in §17.

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
  pointer patching. `0xFFFFFFFF` is the sentinel (`DS41RT_V41_SAMPLER_NO_MASK_ROW`).
- **Launch-shape rule (found in chunk-1 review)**: `mask_words_per_row` **must be
  0 when no mask buffer is supplied**, and the buffer and the stride are produced
  by one helper so they cannot diverge. An all-unconstrained non-compact round
  (for example a traced all-greedy round, §8.5) is a real reachable case with
  `mask_words == nullptr`; the validator rejects `(None, non-zero)` and
  `(Some, 0)` alike (`lib.rs:18189-18197`).
- **`params` MUST point at device memory** holding `rows` consecutive 64 B
  blocks; the kernel dereferences `params[blockIdx.x]` on device. The residency
  requirement is pinned in `v41_sampling_gpu.h` and the product path uploads
  into `param_device` before launch (§5.5).
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
  (`constraints.rs:45`, `lib.rs:3358-3379`) gets **`mask_row = 0xFFFFFFFF` and
  `flags.bit2 NO_MASK`**; the validator requires the two to agree (§5.4). There
  is no all-ones fill in the native path (contrast the legacy `real_full`
  representation, `commands/real_full/constraint.rs:237-245`).
- **As-built constrained seam:** per-row `needs_mask` propagation is implemented
  by `State::prepare_verification_masks` (`constraints.rs:115`) and
  `State::prepare_verification_mask_row` (`constraints.rs:88`), both returning
  `Option<Vec<u32>>` (one entry per row; `None` = unconstrained). A row that
  needs no mask becomes `NO_MASK`, **never** an inherited or zeroed mask — the
  matcher's mask buffer is reused across rows, so dropping `fill_bitmask`'s
  return value would silently apply stale grammar.
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
2. Before upload, the host zeroes any bit `>= vocab` in the last word via
   **`ds41rt_v41_sampler_clear_remainder(words, vocab)`** (declared in
   `v41_sampling_gpu.h`; no-op when `vocab % 32 == 0`). This is enforced
   unconditionally, for every masked row, immediately before upload — it is
   **not** gated by `flags.bit2 NO_MASK` (§5.1). It makes a whole-word reader
   safe even if a future kernel reads 32 tokens per mask word
   (`apply_token_bitmask_f32_candidate_kernel` reads whole words,
   `sampling.cu:687-690`).
3. The kernel's `allowed(t)` is `mask_word(t/32) >> (t%32) & 1` evaluated only
   for `t < vocab`. A `NO_MASK` row skips the arena entirely and allows every
   `t < vocab`.

Today's CPU path never masks those bits and is harmless only because `allowed`
is per real token id (contract §6.1). The GPU path must not rely on that.
Pinned by §12's `vocab = 33` / `last word = u32::MAX` test and by the
`vocab ∈ {1, 32, 33, 100, 127, 129281}` device case
(`native/tests/v41_sampling_selftest.cu:760`, `:834`).

### 5.4 Outputs

```c
/* As built in native/cuda/kernels/v41_sampling_gpu.h and registered in
   native/include/ds41rt_native.h; mirrored by ds41rt-ffi/src/lib.rs. */
ds41rt_status_t ds41rt_cuda_v41_target_sample_async(
    const float* logits, size_t rows, size_t vocab, size_t logits_stride,
    const ds41rt_v41_sampler_row_t* params,     /* MUST be device memory */
    const uint32_t* mask_words,                 /* nullable */
    size_t mask_words_per_row,                  /* 0 iff mask_words == nullptr */
    uint32_t* out_indices,        /* rows u32, REQUIRED: the only production read */
    uint32_t* out_status,         /* rows u32, REQUIRED: hard-error channel */
    uint32_t* out_status_detail,  /* rows u32, REQUIRED: token id / width / NO_DETAIL */
    float*    out_scores,         /* rows f32, REQUIRED for greedy rows (scores.rs:61-66) */
    float*    out_total,          /* rows f32, nullable; DIAGNOSE only */
    uint32_t* out_nucleus_count,  /* rows u32, nullable; DIAGNOSE only */
    ds41rt_v41_sampler_scratch_t* scratch,      /* REQUIRED: rows * 64 B */
    void* cuda_stream);
/* plus the synchronizing wrapper with the same signature minus cuda_stream,
   matching sampling.cu:1321-1341's convention. */
```

Status codes: `0 OK`, `1 EMPTY_CANDIDATES`, `2 NONFINITE_LOGIT`,
`3 INVALID_TEMPERATURE`, `4 MASK_WIDTH`, `5 INTERNAL`.
**`out_status_detail` sentinel (as built):** the device writes
`DS41RT_V41_SAMPLER_NO_DETAIL = 0xFFFFFFFF` when there is no detail, because
token `0` is a real token id and a plain `0` could not be told from "no detail";
the host normalizes the sentinel to `0` before it is observable
(`SampledTargetRows`, `scores.rs:65-68`). Detail carries the offending token id
for `2` and the provided word count for `4`.

**Error-string status (as built, D1 preserved).** `check_status`
(`scores.rs:80-106`) renders:

| status | host message | relation to the CPU path |
| --- | --- | --- |
| `EMPTY_CANDIDATES` | `"grammar allows no target token"` | **byte-identical** to `scores.rs:317` |
| `INVALID_TEMPERATURE` | `"invalid target sampling parameter: temperature"` | **byte-identical** to `target_sampling.rs`'s `InvalidParameter("temperature")` rendering |
| `NONFINITE_LOGIT` | `"non-finite target logit at token {id}"` | deliberate **prefix-superset**: the CPU reports neither the id nor this phrasing |
| `MASK_WIDTH` | `"invalid grammar mask width: {width}"` | deliberate **prefix-superset**: the CPU does not report the actual width |

The two supersets share the CPU string's prefix, so no caller that matched on the
old text breaks. They are documented as supersets, not claimed identical.

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
- **The as-built FFI validator is stricter than the earlier §5.1 letter.** It
  rejects (`validate_v41_sampling_buffers`, `lib.rs:18151-18310`):
  `rows == 0`, `vocab == 0`, `vocab > u32::MAX`, `logits_stride < vocab`,
  `params.len() != rows`, a non-zero `mask_words_per_row` with no mask buffer, a
  positive `mask_words_per_row` that differs from `ceil(vocab/32)`, a
  `mask_words_per_row` of 0 with a mask buffer, temperature outside `[0, 2]`,
  `top_p` outside `(0, 1]`, `min_p` outside `[0, 1]`, `ln_min_p` that is not
  exactly `f32::ln(min_p)` (or not `-inf` when `min_p == 0`), non-zero reserved
  fields, unknown `flags` bits, `STRICT_FINITE` on a row that is neither greedy
  nor masked, `output_row >= rows`, `NO_MASK` without the `0xFFFFFFFF` sentinel,
  a masked row with no mask buffer, `mask_row >= rows`, and buffer extents.
  **`top_k` is not range-validated yet** — harmless in chunk 1 because K1 never
  selects with `k`; full `top_k ∈ {0} ∪ 1..=vocab` validation is a chunk-2+ item
  (§17).
- **Chunk-3a validators, and a deliberate trust-level choice.** The new top-k
  wrapper validates its own buffers (`validate_v41_topk_select_buffers` in
  `lib.rs` / `validate_topk_select_args` in the `.cu`): `rank_order_capacity` must
  be `0` (selection-only) or `>= max_r params[r].top_k`, capacity-bearing calls
  must supply both arenas, and the kernel re-checks `capacity >= total_retained`.
  **The raw-C `validate_topk_select_args` does NOT re-check `output_row` bounds
  while the FFI validator does.** That is a conscious **same-trust-level** choice:
  the C entry points are only called through the FFI wrapper, which has already
  bounds-checked `params` against `rows`; duplicating the check in C would buy no
  safety on the supported path. It is recorded here so **chunk 4 remembers to
  enforce `output_row` bounds at the FFI boundary** for any new caller (and to
  keep the two validators in step).

### 5.5 Buffer ownership

| Buffer | Owner | Lifetime |
| --- | --- | --- |
| logits `b(4)` | `TargetHeadWave` (`v41_target_head.rs:57-60`) | unchanged; read-only to the sampler |
| `param_pinned` + `param_device` | `TargetSamplingWave`, owned by `TargetHeadWave` | allocated once in `TargetHeadWeights::wave` (`:44-87`), freed with the wave |
| `mask_pinned` + `mask_device` (arena) | same | same |
| `scratch` (K1 per-row reductions, 64 B/row) | same | same |
| `bitmap` (top-k arena, chunk 3) and `histogram` (chunk 6) | same | allocated now; chunk 3a's K3/K4 use an **O(k) rank-ordered id list** inside this region's per-row stride, not a bitmap — see §4.4 and §11.1 |
| `ids`/`status`/`detail`/`scores` device + pinned | same | same |
| `total`/`nucleus` device (nullable outputs) | same | written only under `DIAGNOSE` |

The as-built field list is `TargetSamplingWave` (`v41_target_head.rs:80-112`).
Charged to `TargetHeadWave::device_bytes` (`:124-129`) so head budgeting stays
honest. **Zero per-call cudaMalloc**, matching the existing pattern
(`greedy_staging: HostAllocation::new(library, capacity * 8)`, `:83`;
`RowDownload::new`, `:85`).

**Residency requirement (as built).** `params` **must** point at device memory:
the kernel dereferences `params[blockIdx.x]` on device, so a pageable host
address is not device-addressable under CUDA's documented model. The requirement
is stated in `v41_sampling_gpu.h` next to the entry-point declarations, and the
product path stages the blocks in `param_pinned`, H2D-copies them into
`param_device`, and launches that buffer (`TargetSamplingWave::upload` →
`launch`), exactly as designed in §8.3.

**Platform observation, recorded as a hazard, not a property.** During chunk-1
investigation a pageable host `params` pointer *happened to work* on this box:
`cudaPointerGetAttributes` reported `cudaSuccess` (type 0, unmanaged/host) and
the launch returned success (`runs/chunk1-scratch/logs/host_pointer_probe.log`).
That is an ATS/HMM-class driver behaviour on this platform and must **not** be
relied on — it is precisely why the header now states the requirement explicitly
and why the validator/wave pass the uploaded device buffer. Any caller that
skips the upload is unsupported even if it appears to work locally.

**Chunk-3b closure of this defect class.** The chunk-3a K3/K4 wrappers and the
new chunk-3b K5 wrappers had reintroduced the host-pointer defect
(`params.as_ptr()` passed as a device pointer). All four FFI wrappers
(`ds41rt_cuda_v41_topk_select[_async]`, `ds41rt_cuda_v41_nucleus[_async]`) now
take a **validated `params_device: Ds41rtDeviceBuffer`** argument, call
`validate_v41_params_device` (non-null, `rows × 64` bytes) and launch
`params_device.ptr`, exactly the K1 pattern; the device-test helpers were fixed
to pass the uploaded `params_buffer`. Any caller passing a host `params.as_ptr()`
is now a **protocol violation** — it is not covered by the ABI and is unsupported
even where the driver tolerates it (`ADVERSARIAL-REVIEW-rev3`, mandatory item).
Any new entry point added later must follow the same rule.

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

**6.3b — `expf`: DECLARED residual (chunk-2 measured decision).**
Every weight `w_t = expf(scaled_t - max_scaled)` is evaluated by CUDA `expf`,
which is not bit-identical to glibc `expf` (what Rust `f32::exp` resolves to
here). Chunk 2 ran the experiment (§12.9a) and the decision is **keep CUDA
`expf` and declare the residual** — no tested path is bit-matching, and the
alternatives are unaffordable. MEASURED over every surviving
`delta = f32(logit·inv − max_scaled)` on the six phase-0 rows across the
fast-path grid (temperatures `{1e-4, 0.2, 0.7, 1.0, 2.0}` × `min_p
{0, 0.05, 0.5, 1.0}` = 120 cells):

| variant | weights differing / 7,029,396 | fraction | max ulp | tokens changed / 983,040 (sequential order) |
| --- | ---: | ---: | ---: | ---: |
| CUDA `expf` (shipped) | 1,089,047 | 15.4928% | 2 | 85 (8.647e-05) |
| double `exp` then round | 1,214 | 0.0173% | 1 | 0 |

Reasoning, with numbers:

1. **No tested path is bit-matching.** CUDA `expf` differs on 15.4928% of weights
   (max 2 ulp); the double-`exp`-then-round variant still differs on 1,214 weights
   (0.0173%, max 1 ulp), so it cannot be sold as a bit-pin either.
2. **The per-token effect of CUDA `expf` under the CPU's own sequential order is
   small**: 85/983,040 = 8.647e-05. The double variant changed 0 tokens in this
   sweep, so it would reduce — not provably eliminate — that slice.
3. **Double `exp` is unaffordable**: 1321.2 µs vs 392.5 µs at 4 rows (3.37×), and
   27.68 vs 8.43 µs/row at 48 rows.
4. `ln_min_p` stays host-precomputed and the device never calls `logf`, so `min_p`
   membership is untouched (6.3a).

Per-row and per-profile tables, plus the full 120-cell table, are in
`runs/chunk2-scratch/REPORT.md` §6 and `out/expf.json` (sha256 in that report).
`tied` and `min_p = 1.0` are exactly 0% because every weight is exactly
`expf(0) = 1`; `moderate` is ~0.01% because its support has large gaps; the wide
rows differ on 18–38% of weights by ≤2 ulp. Independent confirmation that glibc
`expf` is what Rust resolves to here: a host sequential port reproduces the
production CPU sampler's token on **0/983,040** draws. That is confirmation, not
proof of weight bit-identity, and it is documented as such.

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

**Measured (chunk 2, fast path).** MEASURED paired GPU-vs-CPU token mismatch over
120 cells × 8,192 seeded draws = 983,040 draws: **119,138 / 983,040 = 12.1193%**
(12.3603% before the chunk-2 fallback fix; that delta is the fallback correction,
not a distribution change). Per row (20 cells, 163,840 draws each, from
`runs/chunk2-scratch/out/mismatch.json`):

| row | mismatch / 163,840 | rate |
| --- | ---: | ---: |
| `peaked` | 3,307 | 2.0184% |
| `moderate` | 1,031 | 0.6293% |
| `near_uniform` | 94,618 | **57.7502%** |
| `tied` | 0 | 0.0000% |
| `long_tail` | 20,180 | 12.3169% |
| `trace_anchored` | 2 | 0.0012% |
| **all** | **119,138** | **12.1193%** |

The `near_uniform` figure lands in the O(0.1–1) band predicted above, exactly
because adjacent cumulative gaps there are ~`1/129,280` and a ~1e-5 relative
accumulation difference moves the crossing by one or two tokens; `tied` is 0%
because its weights are integer-exact.

**Distribution vs mapping.** Every cell's device histogram matches the
independent f64 analytic oracle within the phase-0 seeded-multinomial noise
reference, and `TV(device, oracle) ≈ TV(cpu, oracle) ≈ noise` in every cell.
`TV(device, cpu)` *below* the noise reference is the signature of a **mapping**
difference — adjacent-token shifts that share the same uniforms — not a
distribution error. The divergence is therefore a uniform→token mapping shift on
wide-support rows, not a wrong distribution, and the release wording must say
exactly that.

**The ordered path: per-token normalization makes the retained domain
token-exact (chunk-3b headline, MEASURED).** The original design evaluated the
nucleus with the algebraic rewrite `Σw >= threshold·total`
(`DS41RT_V41_K5_NORMALIZED_MASS=0`). That is equivalent in real arithmetic but
**not in f32**. Chunk 3b replaced it with the CPU's own per-token form
(`DS41RT_V41_K5_NORMALIZED_MASS` default **1**): accumulate `fl(w/total)` per
rank, exactly `target_sampling.rs:534-551`. Result on the 344,064-draw
ordered-path harness (`measure3b`; independently reproduced in
`ADVERSARIAL-REVIEW-rev3` §5):

| domain | `=1` (shipped, per-token) | `=0` (algebraic rewrite) |
| --- | ---: | ---: |
| retained (`top_k != 0`), 147,456 draws | **0 — 0.00% (0/18 cells)** | 16,834 — 11.4163% |
| survivor (`top_k = 0`), 196,608 draws | 50,927 — 25.9028% | 51,016 — 25.9481% |
| overall, 344,064 draws | **50,927 — 14.8016%** | 67,850 — 19.7202% |

Concrete mechanism: ten equal weights with `top_p = 0.8000000715255737f`
(`3f4cccce`) — CPU nucleus **8**, un-normalized GPU **9**
(`fl(top_p·10) = 8.000000953674316`), normalized GPU **8**
(`runs/chunk3b-scratch/verify/f5case.c`). The algebraic rewrite was only
real-arithmetic equivalent; the per-token division reproduces the CPU's f32
rounding order. This is a real behavioural change, not a harness artifact: the
counterfactual `=0` run on the same harness/CPU stream yields the 16,834 retained
mismatches, and the CPU's own code is `weights[r] /= total` then a left-to-right
prefix (`target_sampling.rs:534-551`).

**After this change the only remaining ordered-path residual is the
accumulation-association difference on the survivor domain**, with its two
labelled mechanisms:

- **Residual A — wide-flat total bias (O(n·ulp)).** On survivor-domain
  `near_uniform`/`tied` rows the CPU's sequential f32 normalization sum over
  ~116k–129k near-equal terms is systematically biased (measured device `total`
  129,188 vs CPU 129,280 = 7.1e-4 relative; exact f64 = 129,187.795), which moves
  the CPU's own boundary by +8 ranks on `near_uniform` and +140 on `tied`
  relative to the exact-double boundary; the GPU's balanced tree sum lands on the
  exact-f64 crossing.
- **Residual B — k-scale normalized-prefix drift (36–40 terms).** The CPU's
  per-token division plus sequential accumulation of only 36–40 quotients
  (`tied @ k40 @ top_p0.9`: CPU nucleus 37 vs GPU/oracle 36). **Residual B is now
  largely removed by the per-token division** — the retained domain is
  token-exact — so Residual A on the survivor domain is what remains.

**Numbers of record, and what a daemon-level sweep does NOT mean (chunk 4).** The
per-row-class rates are the published numbers, measured on the **real phase-0
fixtures** with the shipped kernels (independently reproduced in
`ADVERSARIAL-REVIEW` for chunks 3b/4a):

| class | rate | draws |
| --- | ---: | ---: |
| retained domain (`top_k != 0`, ordered) | **0 / 147,456 = 0.00%** | 18 cells |
| survivor domain (`top_k = 0`, ordered) | **50,927 / 196,608 = 25.9028%** | 24 cells |
| ordered overall | 50,927 / 344,064 = 14.8016% | 42 cells |
| fast path (K1+K2) | **121,047 / 983,040 = 12.3135%** | 120 cells |
| real `long_tail`, `temp0.7 + top_p0.9` | **1,368 / 8,192 = 16.70%** | 1 row |
| real `long_tail`, `temp1.0 + top_p0.5` | 2,936 / 8,192 = 35.84% | 1 row |
| real `near_uniform`, `temp0.7 + top_p0.9` | 8,110 / 8,192 = 99.00% | 1 row |

The daemon wiring's own device sweep measured **0/1024 (and 0/256 per profile) on a
`smooth_narrow` synthetic row**. That is a **wiring result**, not evidence that the
rates above were pessimistic, and it must never be published as the residual.
Mechanism: divergence scales with **nucleus width and mass-gap structure**. The
synthetic row has a nucleus of tens of ranks and a minimum adjacent normalized gap
of ≈`1e-4`; the real `long_tail` row has a **90,426-token nucleus** and a minimum
adjacent gap of ≈`3.3e-9`, so a 1–2 ulp `expf`/association difference can move the
crossing. The fixture was renamed **`smooth_narrow`** precisely so it cannot be
confused with the real `long_tail` row (it was mislabelled in an earlier revision
of the chunk-4 report). Re-run the real-fixture harnesses to describe the residual;
use the synthetic row only to prove routing and mask plumbing.

**Precedent already in-tree.** The CPU itself uses two different accumulation
orders for the same draw: the fast path accumulates in **token order**
(`target_sampling.rs:597-618`) and the ordered path in **rank order**
(`:534-570`). Accumulation-order sensitivity of the token is therefore already an
accepted property of the released sampler; this design adds a third, fixed,
per-row order and does not introduce a new class of behaviour.

**Required measurement.** Measured: the fast path at **12.3135%** after the
chunk-3b K2 rewrite (§4.2; it was 12.1193% before, not apples-to-apples), and the
ordered path at **14.8016%** overall with the retained domain at **0/147,456**.
It **must be re-measured in full after any change that alters the segment count,
the tree association, or the normalization form**, because such a change moves the
uniform→token mapping (see the §13.1 caveat). Bit-identity is not claimed for the
token; the mismatch rate is, and it is published in the release notes.

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
(the chunk-6/7 gate decides). The design keeps the CPU mapping intact and accepts a
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

**Measured outcome (chunk 2): ship the tree combine; the sequential combine is
not a candidate.**
Both variants are compiled (`DS41RT_V41_K2_SEQUENTIAL_COMBINE` in
`v41_sampling_gpu.cu:42-49`, default off). MEASURED at 4 rows: tree 392.5 µs vs
sequential 8251.6 µs (8.25 ms, ~2 ms/row — worse than the CPU sampler); at 64
rows the sequential cost is flat (8.35 ms) because each CTA serializes ~2 ×
129,280 `expf` and dependent adds while 255 threads idle. The sequential combine
does **not** buy bit-identity either: it removes the order term but leaves the
CUDA-`expf` term (§6.3b), still diverging on 85/983,040 draws
(0.0086%); tree+CUDA-`expf` diverges on 119,138 (12.1193%). Only
sequential+double-`exp` reached zero divergence in the sweep (0/983,040, model
plus a 1,536-draw GPU spot check), and that path is 3.37× slower than the tree
and still not weight-bit-identical. The chunk-2 default was therefore the tree
(the chunk-3b update below supersedes its cross-segment combine); the
sequential macro is retained as a build knob for diagnosis, never an ABI field.
The full-grid sequential rate is partly model-derived: the expf experiment's
sequential-order accumulation plus a GPU spot check (0/1,536), because running
the sequential GPU over 983,040 draws would take ~2.2 h — recorded as an honest
limit in §18.

**Update (chunk 3b).** The chunk-3b P0-1 fix replaced the tree combine with a
**consistent sequential segment fold** — `C(0) = 0`, `C(i+1) = fl(C(i) + local(i))`,
built by thread 0 in at most `kBlock = 256` steps (§4.2). The shipped
cross-segment association is therefore sequential now, but it is **not** the
one-thread-per-row `DS41RT_V41_K2_SEQUENTIAL_COMBINE` walk measured above (which
stays off at ~8.25 ms/row). This is why the fast-path divergence moved
12.1193% → **12.3135%** and latency improved 22.3%. The remaining K2 residual is
the in-segment walk association in the saturation case (§4.2), not the
cross-segment combine.

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
| 14 | NaN/Inf discipline: reject non-finite allowed logits; non-finite scaled max → invalid temperature; never NaN/panic (`:256-258`, `:434-448`) | K1 status precedence of §4.1; **mode-specific masked-non-finite rule**: greedy/constrained strict over the whole row (`scores.rs:407-420`), stochastic permissive on masked tokens (`target_sampling.rs:430-436`, `:1009-1024`); **the shipped host sets the per-row flag bit4 for greedy rows only** (`build_target_sampling_plan`, `scheduler.rs:725-735`) | `-f32::MAX` at `T=1e-5`; huge logit at tiny T; masked NaN per mode; strict-mode masked NaN must error; chunk-5b `boundary_eq`, the `raw_probe` permissive masked-non-finite case, `degenerate:nan_masked_out_permissive`, and the daemon test `masked_stochastic_rows_stay_permissive_and_greedy_rows_stay_strict` |
| 15 | Greedy throughput not poisoned by a stochastic peer (`scheduler.rs:397-407`, `:458-462`) | per-row params + per-row greedy branch; no whole-batch full download | mixed-lane test asserting zero full-row D2H and greedy id equality |

---

## 8. Integration plan

### 8.1 Seams to change

| Seam | Before | As built in chunk 1 (`9dffad0`) / planned |
| --- | --- | --- |
| `execute_logits` (`scheduler.rs:389-407`) | non-compact: `pass.execute` → `vec![0; logits.logits.bytes]` → blocking `lib.copy_d2h` (`:400-403`); compact: `execute_greedy` (`:397-398`) | a **fully-greedy** round (unconstrained and/or constrained) routes through the device terminal; any round containing a stochastic member keeps the CPU path and its full-row download. See §17 for the exact scope |
| `independent.rs:137-144` | compact: `execute_shared_greedy`; non-compact: `execute_shared` + `BatchScores::new(pass.download_logits(...).await?)` | same gate; the terminal is reached only when `use_sampled_terminal(P::SUPPORTS_SAMPLED_TERMINAL, round_is_fully_greedy(...))` is true (`scheduler.rs:519-545`) |
| `scores.rs` `BatchScores` | full `bytes: Vec<u8>` or compact `best: Vec<u32>`; `new` runs a CPU argmax per row (`:67-71`) | `SampledTargetRows` (+ `check_status`, `with_full_logits`) carries the device ids/status/detail/scores; `BatchScores::new` remains for the CPU path and the oracle |
| `scores.rs::row_logits` (`:11-17`) | called per sampled row (`:47`, `:119`) | still used by the CPU path; not called for a device-selected all-greedy round |
| `v41_target_head.rs` | `execute_block` (`:324-331`), `execute_block_cooperative` (`:334-348`), `execute_block_greedy` (`:350-377`) | `execute_block_sampled` (`:768-798`): `copy_block` → graph ensure/launch → sampler (`TargetSamplingWave::upload/launch`) → existing `wait()`/`synchronize()` → `sampling.output()`; plus `download_sampled_rows` (`:810-814`) for the trace and the CPU-path rows |
| `v41_target_pass.rs` | `execute_greedy`/`execute_shared_greedy` (`:177-189`) | `execute_shared_sampled` + `sampled_rows` + `download_sampled_rows` on the `VerificationTarget` trait, gated by `const SUPPORTS_SAMPLED_TERMINAL` (`verification.rs:32` false, `:64` true) |
| `constraints.rs` | `select_verification` (`:61-73`), `select_verification_sampled` (`:80-102`) select on the CPU from host rows | **as built**: `prepare_verification_masks` (`:115`) and `prepare_verification_mask_row` (`:88`) return `Option<Vec<u32>>` per row, preserving the CPU selection twins; device selection consumes the `NO_MASK`/arena form (§9.2) |
| `prepare_commit_lane` (`scheduler.rs:528-607`) | `finishing && next.has_full_logits()` retains host bytes (`:598-599`) | **as built**: `frontier_downloads: Vec<(usize, usize, Option<Vec<u32>>)>` (`:819`) — the retained frontier carries the per-row grammar mask, `None` for an unconstrained row |

### 8.2 What replaces the full-row D2H

For a **device-selected round**, a lane downloads `rows × 16 B` (ids, status,
detail, scores) instead of `rows × 517,120` B. ARITHMETIC: 48 rows × 16 B = 768 B
versus 24.8 MB before.

**Chunk-4 update — the chunk-1 qualifier is gone.** A round containing any
stochastic member is no longer pushed entirely onto the CPU: each row is routed
individually (§8.4). A **fallback** row still downloads its own full row (the CPU
re-sample needs it), and the retained frontier is still the only other full-row
transfer, gated per §10.4. The rounding is unchanged for the greedy compact lane.

### 8.3 How per-row parameters and masks reach the device

1. `prepare_decode_lane` already flattens members in order
   (`scheduler.rs:409-416`) and `selected` is `0..positions().len()`
   (`scheduler.rs:396`, `independent.rs:129`), so the row order is known.
2. Build `Vec<Ds41rtV41SamplerRow>` in that order: for member `m` with
   `input.len()` rows, row `i` gets `seed = params.seed()`,
   `position = request.generated + i` (`scheduler.rs:542`, `:516`),
   `temperature/top_p/min_p/top_k` from `request.job.sampling` (`:541`),
   `flags GREEDY` from `params.is_greedy()`, `ln_min_p` from the host `f32::ln`,
   and `mask_row = 0xFFFFFFFF` + `NO_MASK` for a row that needs no mask
   (`SamplingRound`/`TargetSamplingRowRequest`, `scheduler.rs:556-586`).
   **As built in chunk 4**, `build_target_sampling_plan` (`scheduler.rs:708`)
   produces the per-row `route`/`mask`/`params`/`position`, and a row the device
   cannot serve gets a **device-neutral greedy no-op** block
   (`fallback_sampling_row`, `:767`) while its real values are kept in the plan
   for the CPU re-sample.
3. Constrained members' masks are prepared once per step (§9.2) into
   `mask_pinned`; the `(mask buffer, mask_words_per_row)` pair is produced by one
   helper so the two cannot diverge (`v41_target_head.rs:114-120`).
4. `TargetSamplingWave::upload` fills the pinned staging and issues two
   `copy_h2d_async` calls (params and masks) on the head stream, then `launch`
   runs the kernel sequence, all before the existing drain. `require_complete()`
   guards overwriting staging on the next round (the pattern at
   `v41_memory.rs:75-78`).

### 8.4 Per-row routing and fallback (implemented in chunk 4)

**The gate is now per row, not per round**, replacing the chunk-1 whole-round
`round_is_fully_greedy` precondition. `sampling_route()` (`scheduler.rs:587`)
classifies each row from its **own request parameters**, mirroring the kernels'
own eligibility blocks:

| route | condition | kernels | selection source |
| --- | --- | --- | --- |
| compact lane (not built) | every member greedy + unconstrained + untraced | existing `execute_block_greedy` | untouched (chunk-1 path) |
| `DeviceGreedy` | `temperature < 1e-5` or `top_k == 1` | K1 | K1 `out_indices` |
| `DeviceFastPath` | `top_k == 0 && top_p >= 1.0` | K1 → K2 | K2 `out_indices` |
| `DeviceOrdered` | `top_k ∈ 1..=256`, or `top_k == 0 && top_p < 1.0` | K1 → K3 → K4 → K5 | K5 `out_indices` (K5's `out_status` read) |
| `CpuFallback` | the device cannot serve the row | none | CPU sampler, from the row's downloaded logits |

- **`MAX_RETAINED = 256`** (`DS41RT_V41_SAMPLING_MAX_RETAINED`,
  `v41_target_head.rs:45`) is pinned to K5's `kBlock = 256`.
  `top_k ∈ [257, survivor_count)` is **unsupported and loud** (INTERNAL); the host
  routes it to the CPU fallback **before** the launch rather than enqueueing a
  list K5 would refuse.
- `DeviceOrdered` deliberately does not split `top_k < survivor_count` from
  `top_k >= survivor_count`: K3/K4 are a per-row no-op in the second case and K5
  then treats the retained set as every survivor.
- **Greedy is not regressed:** the compact lane is byte-identical and measures
  **83.4 µs at 48 rows**; a greedy row that goes through the sampled route costs
  the K1 level (98 µs GPU / 103 µs round), never the stochastic level.
- **Per-row fallback, not whole-round, is deliberate.** A whole-round fallback
  would force greedy peers to download full logits, which contract §7.1.15
  forbids and which would defeat the acceptance gate. Per-row keeps the
  verification shape exact: a device row consumes the device id, a fallback row
  consumes the CPU id produced from the same parameters, mask and absolute
  position.
- **Fallback cases, all counted and logged at WARN on `ds41rt::sampling`:**
  (1) `top_k > 256`; (2) a parameter outside the validator's range
  (`temperature` not finite or outside `[0,2]`; `top_p` not finite or outside
  `(0,1]`; `min_p` not finite or outside `[0,1]`); (3) `min_p > 1.0` (defensive:
  K1 would report zero survivors while the CPU filter chain keeps the best token);
  (4) a row the device **refused after the launch** with `INTERNAL`.
- **One round-level fallback remains, deliberately:** a layout without the
  sampled terminal (`DistributedTargetPass`, `SUPPORTS_SAMPLED_TERMINAL = false`,
  `verification.rs:32`) keeps the whole round on the CPU, and so does a round in
  which every row is unservable (the logits must be downloaded anyway).
- **Never a silent fallback.** (a) A fallback row's device-facing block is the
  neutral no-op, so the batch validator accepts it and no kernel publishes a token
  for it. (b) `resolve_fallback_rows` (`scheduler.rs:967`) **stores** the CPU
  token in `BatchScores.best[row]`, and `select_routed` (`:1449`) consumes a row
  as stored when it is device-served **or** carries logits
  (`has_row_logits`); the commit path therefore never reads a stale device slot.
  (c) `store_sampled` (`scores.rs:329`) recomputes from the row's own logits for
  **every** parameter shape and errors if the row has no logits.
- **The finishing-frontier retention classifier is keyed on the parameters, not
  the route:** `frontier_retain(round, params) = round.is_some() &&
  !params.is_greedy()` (`scheduler.rs:1386`). In a device round a non-greedy
  `best` is a draw whoever produced it — K2, K5 or a CPU fallback — and a greedy
  `best` is an argmax whichever path produced it, so the route is deliberately not
  consulted.
- **Three wiring defects the chunk-4 reviews caught, recorded as the lesson:**
  the discarded fallback write-back (a post-launch refusal committed the stale
  device slot); the route-keyed frontier classifier (a planned-fallback
  stochastic frontier was classified `Checked` and its draw rejected as an argmax
  mismatch); and the `target_sampling` counters hidden under `host_metrics()`
  (§13.2 item 3). The invariant: **`best[row]` semantics across the fallback
  boundary must be keyed on the parameters, not the route.**
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
  needs full rows for `top_two`. **As built**, `SamplingRound::trace_rows` is
  every row when the trace target is `DEBUG` and the empty vector otherwise
  (`scheduler.rs:585`), and `download_sampled_rows` fetches exactly that
  selection (`:614-618`, `:636-640`). A traced all-greedy round therefore
  downloads each row; an untraced one downloads none. Do **not** add a per-row
  top-two reduction to the production kernel just for the trace.
- **The single-request blocking `cudaMemcpy` path (`ds41rt_native.cc:1625-1647`)
  is removed only for a device-selected all-greedy round**, because
  `single_lane_round` reaches the same `execute_block_sampled` terminal as the lane
  path (`scheduler.rs:471`, `layout.rs:33-43`) only under the §8.4 gate. A
  stochastic single-request round still uses the CPU path and its blocking copy.
  Removing the blocking copy on the stochastic path is chunk-4 wiring.

### 8.6 The CPU sampler as retained reference/fallback

- Retained as (i) the test oracle, via `reference_select`
  (`target_sampling.rs:1235-1399`) and `compare_to_reference` (`:1401-1450`);
  (ii) a **build- and env-gated diagnostic** fallback.
- Decision (D2): default is the GPU path for a device-selected round; the CPU
  path is what serves every stochastic round in chunk 1, and
  `DS41RT_TARGET_SAMPLER=cpu` (when wired) selects the CPU path for A/B
  diagnosis. It is *not* an automatic fallback: if the native symbol is missing,
  fail at startup, consistent with the existing missing-symbol diagnosis
  (`DEVELOPER.md:67-68`). A silent automatic fallback would make the
  reproducibility claims unverifiable, and the fallback must not silently
  reintroduce the full-row download on the default device path.

---

## 9. Constrained decoding on GPU

### 9.1 Matcher stays on the CPU

`fill_bitmask` targets a CPU `DLTensor` (`native/src/xgrammar_adapter.cc:349-352`),
and the FFI wrapper is CPU-side (`lib.rs:3358-3379`). Nothing in this design
moves xgrammar. Only mask storage and application move.

### 9.2 Mask rows for the speculative prefix, uploaded once per step

**As built (`9dffad0`)** the split is `State::prepare_verification_masks`
(`constraints.rs:115`) and `State::prepare_verification_mask_row` (`:88`), both
returning `Option<Vec<u32>>` — one entry per row, `None` meaning the grammar
allows every token at that row:

- `prepare_verification_masks(&self, input) -> Result<Vec<Option<Vec<u32>>>>` —
  fork the matcher once, then for each row `index`:
  `if index > 0 { branch.accept_token(input[index])? }` (error
  `"illegal verification draft token"`), `branch.fill_bitmask(&mut mask)`,
  record `needs_mask`, and keep `Some(mask)` only when it is true. Row 0's mask
  is the pre-draft state, exactly as today.
- `prepare_verification_mask_row(&self, input, row)` is the retention-time
  variant: only the accepted frontier's mask is needed, and it returns `None`
  when that row needs no mask.
- `select_verification` (`:61-73`) and `select_verification_sampled` (`:80-102`)
  remain the CPU selection twins and are unchanged, so the CPU path and the
  oracle keep exercising the same helper.
- The flat arena is uploaded once per step (§8.3). A row with `None` becomes
  `mask_row = 0xFFFFFFFF` **and** `flags.bit2 NO_MASK`; it is never an inherited
  or zeroed mask. A masked row's arena index is checked against `rows` and the
  mask buffer must exist (§5.4).

Preserved exactly: row 0 uses the mask before any draft accept; later rows use
the mask after accepting the preceding draft; `input[0]` is the committed anchor
already accepted by the authoritative matcher; the authoritative matcher still
advances only on emitted tokens.

**Chunk-4 as-built correction.** Chunk 4a claimed the K3/K4/K5 entry points "do
not consume a mask" for their retained search; that was **wrong** (4b REPORT §6
item 4). All four kernels already take `mask_words`/`mask_words_per_row` and K1
applies the mask before anything else, so no kernel change was needed. What chunk
4b added is **coverage**: a real-xgrammar per-prefix fork/rollback test
(`prepare_verification_mask_row` agrees with the batch form on every row without
mutating the authoritative matcher) and device sweeps proving masked stochastic
draws stay inside their own mask on the fast path, the ordered `top_p` path and
the ordered retained `top_k = 40` path (64/64 per profile), with a mixed round
proving no stale mask inheritance.

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

**Pre-chunk-4 defect (recorded; line numbers are pre-chunk-4).**
`scheduler.rs:597-603` (pre-4b) scheduled `frontier_downloads` whenever a request
was `finishing && !next.has_full_logits()`, and `commit_lane` performed it at
`:638-639`; `independent.rs:151-158` did the same. That happened **regardless of
whether prefix caching was enabled**, and only later did
`prefix.rs:177`/`:206` (now `:187`/`:216`) early-return when `bank.limit() == 0`,
so a finishing request in a cache-disabled deployment still paid a 517,120 B D2H.

**Implemented gate (chunk 4b, MEASURED).** The decision is now the pure function
`frontier_download(whole_batch_has_logits, row_has_logits, retain_enabled,
finishing, events_open)` (`scheduler.rs:1332`):

| condition | action | transfer |
| --- | --- | --- |
| not finishing, or the client disconnected | `Skip` | none |
| whole batch already on the host (CPU path / traced round) | `WholeBatch` | none (pre-4b behaviour preserved exactly) |
| this row already packed (a fallback row downloaded for the CPU re-sample) | `PackedRow` | **none** (was a second transfer on the independent lane) |
| retention enabled, row not on the host | `Device` | one `ROW_BYTES` D2H |
| retention disabled (`prefix_cache_entries == 0`) | `Skip` | **none** (was an unconditional 517,120 B per finishing request) |

- `PrefixCache::turn_bank_enabled()` (`prefix.rs:65`, built on the chunk-4b
  `prefix.rs` changes) reads `bank(Turn).limit() > 0`; `retire_request`
  (`scheduler.rs:119`) and `independent::retire` (`independent.rs:275`) also check it before
  requiring `next_after_commit`, so a cache-disabled deployment no longer logs a
  spurious `"finished request has no retained logits"`.
- **Correctness:** `next_after_commit`'s only readers are the gated retire paths
  and `PrefixCache::retain`/`queue_retain`, which early-return at `limit() == 0`
  before touching the host cache, so no snapshot can lose bytes it would have
  kept. A `PackedRow` is retained from bytes the round already holds
  (`retain_packed` for greedy, `retain` for a stochastic draw). The whole-batch
  path is byte-for-byte the pre-4b behaviour.
- **The saving is reported as TWO distinguishable counters, never a combined
  total** (§13.2): `frontier_gated_rows`/`frontier_gated_bytes` (both lanes; the
  gate removal — the primary saving) and
  `frontier_packed_rows`/`frontier_packed_saved_bytes` (independent lane only; the
  serial lane's `retain_from_device` already short-circuited on packed bytes, so
  the serial side deliberately does not count a saving).
- **`target_sampling` (including these counters) is published
  UNCONDITIONALLY** by `serving_stats(&PrefixCache)` (`scheduler.rs:173`);
  `host_cache`/`host_cache_config` are `null` without a bound cache. This was a
  4b review fix, **stated precisely**: the object was previously nested under
  `host_metrics()`, so it was published **only when a host cache was attached**
  and was invisible in the **no-host-cache** deployment. That is a *different
  condition* from the retention gate, which keys on the **turn bank**
  (`turn_bank_enabled()` = `bank(Turn).limit() > 0`, i.e. `prefix_cache_entries
  == 0` disables it, `prefix.rs:65`). The two knobs are **independent**: a host
  cache with a disabled turn bank means the gate is active while the counters
  were previously still visible; no host cache at all means the gate is inactive
  while the counters were previously hidden. The hidden configuration was the
  no-host-cache one, **not** the gated one — do not conflate them.
- **Measured cost of the removed transfer** (release, RTX PRO 6000, pageable
  `copy_d2h` of one row): one-row frontier D2H **29.2 µs / 517,120 B**; a
  representative 8-finishing-row round **233.9 µs / 4,136,960 B**.

---

## 11. Workspace, streams, and graph compatibility

### 11.1 Workspace, sized for maximum rows × vocab

Allocated once per `TargetHeadWave`:

| Region | Bytes | Formula |
| --- | ---: | --- |
| sampler params (device) | 5,120 | `capacity × 64`, staged in `param_pinned` |
| mask arena (device + pinned) | 2 × 1,292,800 | `capacity × ceil(vocab/32) × 4` at capacity 80 |
| scratch (per-row reductions) | `capacity × 64` | `ds41rt_v41_sampler_scratch_t`, exactly 64 B |
| top-k arena | 1,292,800 | `capacity × ceil(vocab/32) × 4`; **allocated now**. As built, chunk 3a's `ds41rt_cuda_v41_topk_select[_async]` writes an **O(k) rank-ordered id list** (4 B/id + 8 B u64 staging) whose per-row stride is `rank_order_capacity`; the §11.1 region is the production backing and its stride sizing / k-beyond-capacity fallback are **chunk-4 decisions** |
| ids/status/detail/scores | `capacity × 4 × 4` | device + pinned |
| diag (`total`, `nucleus_count`) | `capacity × 8` | **allocated now**, written only under `DIAGNOSE` |
| chunk-6 radix histogram | 655,360 | `capacity × 2048 × 4`; **allocated now**, unwritten until chunk 6 |

All regions above are allocated once in `TargetSamplingWave` at wave construction
(`v41_target_head.rs:80-112`); the chunk-3 rank-order arena, chunk-6 histogram
and the nullable diag outputs are reserved from the start so a later chunk adds
no allocation and no resize. Total ≈ 4.2 MB at capacity 80, well within the
existing head budget (`TargetHeadWave::device_bytes`, `v41_target_head.rs:124-129`).
As built, chunk 3a additionally takes an **O(k)** u32 `rank_order_ids`
(`rows × rank_order_capacity`) and an equal O(k) u64 staging buffer
(`rank_order_scratch`, 8 B/entry) — 25.6 KB + 51.2 KB at capacity 80, negligible
against the region above. Who allocates them and how the §11.1 top-k region is
carved into the per-row `rank_order_capacity` stride (plus the k-beyond-capacity
fallback) is a **chunk-4 decision**.
**Zero per-call allocations**: no `cudaMalloc`, no `Vec` growth, no `to_vec()` on
the hot path (contrast `download.rs:41`).

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

**As built**, chunk 1 added a dedicated CUDA test target,
`native/tests/v41_sampling_selftest.cu` (`native/CMakeLists.txt:1051-1059`),
with its own CPU oracle and a `SKIP_RETURN_CODE 77` convention. Later chunks
extend that file rather than `cuda_selftest.cc`; the style reference remains
`cuda_selftest.cc` (`:545-624`, `:2158-2300`).

- The CPU oracle `cpu_v41_target_sample(...)` must be a **faithful port of the
  production `reference_select`** (`target_sampling.rs:1235-1399`) — mask,
  temperature, min_p, top_k exact-k-lowest-id, inclusive top-p, SplitMix64 draw,
  status codes and `total`/`nucleus_count`. The reviewer must diff this port
  against `reference_select` line by line; it is the only oracle that can
  validate the new rules. Chunk 1's oracle covers the K1 subset; chunks 2–3b
  extend it.
- Chunk 1 landed `..._greedy_matches_ref`, `..._masked_matches_ref`,
  `..._status_precedence`, `..._vocab_remainder`, the mode-specific finiteness
  and `STRICT_FINITE` case, min_p boundary cases, the subnormal grid, batch
  independence and the 63-cell greedy parity grid. Chunks 2+ add
  `..._fast_path_matches_ref`, `..._ordered_matches_ref`, `..._rng_bit_equality`.
- Use the target's own conventions (mirroring `cuda_selftest.cc`):
  `device_buffer`, `copy_h2d`, `copy_d2h`, `assert_close`, `require_status`.
- **Warning learned in chunk 1:** a new device test target must set
  `CUDA_ARCHITECTURES` from `DS41RT_CUDA_ARCHITECTURES`; the chunk-1 target
  initially defaulted to `sm_75`, so its probe kernel never launched and its FTZ
  canary read zeros (§17.3 item 9).

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
- **Convention (chunk-2 decision): GPU-requiring FFI device tests are
  `#[ignore]`d and run explicitly**, matching the CUDA-ignored daemon tests
  (`v41_memory/download.rs:49`) and the ignored pattern above, so a plain
  `cargo test` stays green on a host without the native library. Documented
  explicit-run command:
  `DS41RT_NATIVE_LIB=/abs/path/libds41rt_native.so cargo test -p ds41rt-ffi --lib v41_sampler -- --ignored`
  (an explicit `--ignored` run asserts the library is present, so it cannot
  silently no-op).
- The loader used by those tests, `load_device_test_library`
  (`lib.rs:18535-18560`), **never skips**: on an explicit run with no locatable
  library it returns an error, so the test **FAILS rather than passing unrun**.
  Chunk 2 landed `v41_sampler_device_fast_path_matches_cpu_oracle`
  (`lib.rs:19245` in the current tree, `:19215` in the reviewed revision; the
  name is the stable anchor as later chunks add lines)
  against the **production** CPU sampler (`ds41rt_core::TargetSamplingParams`),
  not a reimplementation; it pre-fills outputs with `0xDEADBEEF` so an unrun
  kernel cannot pass.

### 12.3 Full parameter grid × masks × boundary uniforms

Mirror `param_grid` (`target_sampling.rs:1496-1524`):

- temperatures `{1e-5, 1e-4, 0.2, 0.7, 2.0}`; **chunk-2 decision: the
  fast-path temperature grid includes the greedy epsilon boundary `1e-5`**
  (`GREEDY_TEMPERATURE_EPS`, `target_sampling.rs:49`; `is_greedy` is a strict
  `<` at `:172-174`, so `T = 1e-5` is stochastic and is the highest-value
  boundary to cover). Chunk 2's device fast-path grid ran
  `{1e-4, 0.2, 0.7, 2.0}`; chunk 5's full grid must include `1e-5`;
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

**As built in chunk 3a (stronger than the original wording).** The K3/K4 tie
tests assert the **exact id multiset and the exact rank order**, not counts, and
they do so on thousands-deep and flat tie groups:

- 2048 survivors at 1.0 and 2048 at 0.5 with `k = 3000` → the cut keeps exactly
  the 952 lowest **odd** ids;
- three leaders over a 4093-token flat group with `k = 2000` → retained ids are
  exactly `0..1999`, **asserted independently of the oracle helper**;
- all-tied vocab 1024, `k ∈ {2,40,64,257,1000}` → ids `0..k-1`, also asserted
  independently, plus a 256-token `-inf` tie group below one finite leader;
- `-0.0` tie groups publish `+0.0`, matching `descending_radix_key`.

**Mutation evidence (chunk 3a review).** Two deliberate mutants were built and
run: a **keep-all-ties** K4 (vLLM/FlashInfer rule) and the **old `hi - lo < 3`
termination**. **Both fail the suite**, so the exact-k lowest-id rule and the
length-2 final probe are pinned by tests rather than by inspection
(`runs/review3a/mutA_test`, `mutB_test`; `runs/chunk3a-scratch/REPORT.md` §2, §5).

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
  - stochastic rows: legal and unread (`target_sampling.rs:430-436`,
    `:1009-1024`);
  - greedy and constrained rows: an **error**, matching `scores.rs::argmax`'s
    `ensure!(value.is_finite())` that runs *before* the mask test
    (`scores.rs:407-420`). The strict-mode test must use a masked non-finite
    token and assert `NONFINITE_LOGIT`, so the constrained path is never silently
    loosened.
- **What the shipped plan builds (chunk 5b).** `build_target_sampling_plan` sets
  `STRICT_FINITE` for **greedy rows only** (`if greedy`, `scheduler.rs:725-735`),
  so a masked *stochastic* row now reaches the permissive branch on the served
  device path. The previous strictness was the host `needs_mask || greedy`
  condition, **not** the `BatchScores::new` pre-validation (which lives on the
  whole-round CPU path). The chunk-5b validation pins it with the
  `boundary_eq`/`raw_probe` masked-non-finite cases, the
  `degenerate:nan_masked_out_permissive` cell (32/32 `OK`, 0 mismatches vs the
  CPU oracle) and the daemon test
  `masked_stochastic_rows_stay_permissive_and_greedy_rows_stay_strict`.
- **Unmasked** non-finite logits still fail `NONFINITE_LOGIT` on every path, and
  greedy is unchanged.
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
the harness rows in the repo as part of chunk 6/7 or assert their sha256
(§16 D4).

The same 8,192-draw sweep must also record the **GPU-vs-CPU token mismatch count
and rate per cell** (§6.3c). That number is a required chunk-6/7 deliverable and
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

**DONE (chunk 2).** Outcome (ii): CUDA `expf` kept, residual declared, with the
full table in §6.3b and the report in `runs/chunk2-scratch/REPORT.md` §6 /
`out/expf.json`. The prerequisite for chunk 3a/3b is therefore satisfied.

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
- Additionally: the `reference_select` port in
  `native/tests/v41_sampling_selftest.cu` (chunk 1; `cuda_selftest.cc` for the
  legacy samplers) must be reviewed as a separate artifact from the kernel, by a
  reviewer who does not read the kernel first — otherwise a shared misreading of
  the contract passes both. This is the same reason the phase-0 oracle was built
  independently of the sampler (`REPORT.md:106-111`).

### 12.12 Test-credibility corrections (chunk 3b) and the mutant suite

The chunk-3b external review found that the earlier green K5 suite **overstated
K5 coverage**. Recorded here so no later chunk repeats the pattern:

- **Two tests named as K5 boundary/fallback regressions used `top_k = 0`,
  `top_p = 1`** — i.e. the K2 fast path — so they executed **no K5 code at all**
  and their "0 reached the normalizing shortfall" line was the reference's own
  early return.
- **The one-rank exception never bounded `abs(device_nucleus_count - expected)`**
  and hard-coded `max_nucleus_delta = 1`.
- **The wide profiles disabled `strict_nucleus`**, so `nucleus_mismatch` and the
  rank bounds were not collected, yet the summary printed "nucleus mismatches 0"
  (unmeasured), and device-nucleus membership was never asserted.

**Fixes (all landed in chunk 3b, `runs/chunk3b-scratch/REPORT.md` §4,
`ADVERSARIAL-REVIEW-rev4` §7):**

1. The boundary test now uses `top_k = 2` (an ordered row) and asserts
   `compared == 1`/`exact == 1`, so K5 genuinely runs.
2. The fallback tests use the 13-equal/`top_k = 12` and 71-equal/`top_k = 71`
   fixtures and assert `compared == 2` and **`fallback_rows > 0`** (a real
   `expect`, satisfied by the retained fixture).
3. The exception requires `|delta| == 1` **and** a genuine **f32** prefix
   comparison (`host_cumulative_through_f32`), with `max_nucleus_delta` recording
   the true maximum before the exception.
4. Counters (`nucleus_mismatch`, `max_nucleus_delta`) are collected **outside** the
   `strict_nucleus` gate; device-nucleus membership, `KP_CROSSING_WINDOW`, a
   measured `1e-3` CDF-error bound and a measured wide-row nucleus-delta bound of
   **64 (observed 28)** are always asserted.
5. Device coverage for the raw-C shapes: `test_k2_zero_weight_invariant`,
   `test_k1_only_entry_loud_status` and `test_k5_scattered_row_identity_guard`.

**Mutant evidence — the sensitivity test of the suite.** Six final-source
single-defect mutants all fail (M1 survivor rank keyed by token order, M2 crossing
returns `lo`, M3 empty-prefix acceptance, M4 tree-association retained prefix, M5
silent status, M6 the pre-fix K2 walk), plus auxiliary **M7** (conditional
identity guard) and **M8** (K1 branches removed). The older rev3-based mutant
files abort first on the new P0-1 test purely as a **test-ordering artifact**
(all embed the pre-fix K2 walk and P0-1 runs first) — verified by removing that
call, after which each fails at its intended test.

### 12.13 Verification semantics across the fallback boundary (chunk 4, settled)

Recorded so they are not re-litigated:

- **A stochastic device frontier is NOT argmax-cross-checked**, and that is
  faithful: the CPU path already skipped that check for stochastic rows. Only a
  greedy frontier takes `FrontierRetain::Checked`
  (`retain_packed`/`retain_downloaded_with`); a non-greedy frontier takes
  `RecordedSample` (`retain_recorded_from_device`).
- **The classifier is keyed on the PARAMETERS alone:**
  `frontier_retain(round, params) = round.is_some() && !params.is_greedy()`
  (`scheduler.rs:1386`). In a device round a non-greedy row's `best` is a draw
  whichever producer selected it — K2, K5 or a CPU fallback — and a greedy row's
  `best` is an argmax whichever path produced it, so the **route is deliberately
  not consulted**. Keying on the route was the delta-review P1 defect: a
  planned-fallback stochastic frontier was classified `Checked`, its stored draw
  was recomputed as an argmax, and the lane failed with `"GPU and retained CPU
  greedy selection differ"`.
- **`store_sampled` recomputes from the row's own logits for every parameter
  shape** (`argmax` with the mask for greedy, `select_token` for stochastic) and
  errors if the row has no logits, so **no stale device slot can ever be
  committed** — hardened rather than merely documented after two consecutive
  latent-to-real bugs in this area.
- **A whole-CPU round never reaches the classifier:** `has_full_logits()`
  short-circuits to the trusted `WholeBatch` retain path, and the CPU-sampled
  token never occupies the device `best` slot in that shape.
- **Lesson, stated once:** `best[row]` semantics across the fallback boundary
  must be keyed on the **parameters**, not the route. The three wiring defects the
  chunk-4 reviews caught — the discarded fallback write-back, the route-keyed
  frontier classifier, and the counters hidden under `host_metrics()` — are all
  instances of getting that boundary wrong.

### 12.14 Validation-harness power (chunk 5b): the first campaign could not fail

The chunk-5 campaign's first revision could pass on a deliberately broken kernel.
Chunk 5b hardened it (`runs/chunk5-scratch/REPORT-5b.md` §R1/§R11,
`REPORT-5b-fixes.md`, `review5b/REVIEW-5b.md` §3):

- **Exactness domains hard-fail.** `Campaign::enforce_cells(mode, &stats)` runs
  after every artifact: `mismatches > 0` **or** any undeclared status is a hard
  error with a non-zero exit and a printed `EXACT FAILURE … first_divergence`
  record (`class/cell/policy/seed/position/route/device_id/device_status/cpu_ok/
  cpu_id_or_error`). Greedy, `ordered_retained`, the retained/top-k grid, masks
  and the status-discipline probes are all exact domains.
- **Residual domains use a per-cell deterministic bound with NO clamp:**
  `allowed = baseline + 0.5 / device_draws`, where the baseline records
  `{rate, draws}` and is byte-identical to the pre-hardening baseline on all
  **4,296** cells. `0.5/n` lies strictly between the adjacent achievable rates
  `k/n` and `(k+1)/n`, so **one extra mismatch fails**. Under this policy a
  **1.25×** rate regression fails **43 of 46** non-zero-baseline cells (a 1.5×
  fails 44/46); the three that survive are baseline-saturated data ceilings
  (`baseline = 1.0`), not policy escapes; and all **4,250** zero-baseline cells —
  which the old policy allowed +0.02 — fail on a single new mismatch.
  **4,293 of 4,296** cells have `allowed < 1.0`.
- **The documented runner path was stale.** `rust/target/release/chunk5-campaign`
  was a pre-hardening binary and produced none of the recorded numbers; every
  post-hardening figure comes from
  `runs/chunk5-scratch/harness/target/release/chunk5-campaign`
  (`REPORT-5b.md` §R11).
- **Mutant matrix and its forensic correction.** The first review's "K2 fast-path"
  and "K5 draw" mutants were the **same file byte-for-byte**
  (`c2eb7310…`), both editing line `:742` inside the K2 sequential kernel that is
  **compiled out** unless `DS41RT_V41_K2_SEQUENTIAL_COMBINE=1`; their
  `runs/chunk5-scratch/review/mutants/mutC.cu:1806` is
  `const float target = uniform * nucleus_mass;` and contains no `<=` at all. Its
  "1 of 3 caught" table is therefore **not evidence**. The corrected matrix
  (`REPORT-5b.md` §R9) is:

  | mutant | site | broad campaign | `boundary_eq` | verdict |
  | --- | --- | --- | --- | --- |
  | `mutA` (greedy tie `>`→`>=`) | K1 `:316` | `greedy` exit 1 (80/1,744) | exit 0 | caught (greedy class) |
  | `mutB_seq` (sequential `<=`→`<`) | K2 `:742`, not launched | all 0 (dead code) | exit 0 | not caught — **dead code** |
  | `mutB_seq_active` + the macro | K2 `:742` | — | exit 1 (dev 4 vs 3) | caught |
  | `mutB_shipped` (real K2 `<=`→`<`) | K2 `:668` | `fast` exit 0 (983,040 draws blind) | exit 1 (dev 2 vs 3) | caught |
  | `mutC_k5` (K5 retained `>=`→`>`) | `:1812`, `:1819` | `ordered` exit 1 (2 exact mismatches) | exit 1 (dev 4 vs 3) | caught twice |
  | `mutC_k5_survivor` (`k5_mass_satisfies`) | `:1336` | `ordered` exit 0 | exit 1 (dev 1 vs 0) | caught |

  So the **real** K2 comparison (`:668`), the K5 retained comparisons
  (`:1812`/`:1819`) and the survivor predicate (`:1336`) all now fail. The K5
  equality is **reachable naturally** — the retained mutant produced 2
  exact-equality mismatches in `ordered` mode at seed 20260922 / position 519 —
  which is why the broad ordered mode catches it; the K2 equality did not arise in
  983,040 draws, so only the constructed `k2_segment_boundary` case catches the
  shipped K2 flip. The exact-boundary cases (`k2_fast_path`,
  `k2_segment_boundary`, `k5_survivor`, `k5_retained`) are built by inverting
  SplitMix64 to place the uniform exactly on `k/2^24`.

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
  here, so the chunk-2 decision is made on data. **Done for the fast path in
  chunk 2** (MEASURED, see below).
- **Proposed gate (original):** the sampler+D2H must add ≤ 0.25 ms to a 48-row
  ordered round and ≤ 0.10 ms to a 4-row round. If it does not, chunk 6/7
  (radix histogram) is mandatory before any E2E claim.

**Chunk-2 measured result (fast path only; MEASURED).** Blocking C-ABI entry
point plus ids D2H, 5 warmups, 50 timed calls, `peaked` row
(`runs/chunk2-scratch/out/latency_tree_cuda.json`):

| rows | K1 only (µs) | K1+K2 tree, CUDA `expf` (µs) | K2 delta (µs) | sequential (µs) | double `exp` (µs) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 172.3 | 382.6 | 210.3 | 8,175.8 | 1,316.3 |
| 4 | 177.1 | 392.5 | 215.4 | 8,251.6 | 1,321.2 |
| 16 | 183.2 | 404.4 | 221.1 | 8,353.5 | 1,327.0 |
| 48 | 183.3 | 404.5 | 221.2 | 8,354.1 | 1,328.5 |
| 64 | 183.3 | 404.7 | 221.5 | 8,352.6 | 1,328.6 |

- **The proposed ≤0.10 ms/4-row gate is EXCEEDED and is now a chunk-6/7 item.**
  The tree K2 adds ~215 µs at 4 rows (and K1 alone ~177 µs), against a 100 µs
  gate. Recorded honestly; chunk 2's acceptance item was the measured comparison,
  not the gate.
- **The excess is per-launch critical-path-bound, not launch-count-bound and not
  row-count-bound.** The 4-row and 48-row calls differ by 12.0 µs (3.1%), and it
  is not bandwidth-bound (48 rows × 517 KB × ~4 reads ≈ 99 MB ≈ 55 µs at HBM
  speed). **As measured in chunk 2**, the critical path was two full passes over
  the row, each a ~`vocab / kBlock ≈ 505`-iteration serial dependent `expf`+add
  chain per thread, plus the step-3 owner walk (~33 µs/launch) and two
  Hillis-Steele barrier sets. **The chunk-3b rewrite removed the owner walk and the
  Hillis-Steele scan** (see the re-measurement below).
- **Cheapest levers, in order:** (1) raise the per-row thread count from 256 to
  512/1024, halving/quartering the per-thread serial chain — a launch-geometry
  change under contract §7.2.6 with no semantic change, and the largest lever;
  (2) fold the owner walk into an existing pass or replace the tree scan with a
  cheaper fixed-shape combine; (3) the chunk-6 radix histogram, which attacks the
  pass count but not the per-thread dependent chain. **Chunk 3b already
  implemented (2) as part of the P0-1 fix, for a −22.3% net gain.**
- **Caveat on any lever that changes the segment count:** the tree association
  changes, hence the uniform→token mapping changes. Determinism is preserved and
  it is contract-§7.2.6-legal, but the published GPU-vs-CPU mismatch rate **must
  be re-measured afterwards** (the 8,192-draw protocol of §12.9); the mismatch
  figures in §6.3c are valid only for the shipped configuration.

**Chunk-3b re-measurement after the K2 rewrite (MEASURED).** Same harness and
protocol as above; "before" is the shipped chunk-2 (rev3) kernel, "after" is the
chunk-3b rewrite (§4.2; `runs/chunk3b-scratch/k2_latency_{before,after}.log`):

| rows | before µs | after µs | delta | K2-non-applicable baseline before → after |
| ---: | ---: | ---: | ---: | --- |
| 1 | 382.6 | 298.0 | −84.6 (−22.1%) | — |
| **4** | 392.6 | **305.0** | **−87.6 (−22.3%)** | 176.9 → 176.3 µs |
| 16 | 404.6 | 314.4 | −90.2 (−22.3%) | — |
| **48** | 404.9 | **314.8** | **−90.1 (−22.3%)** | 183.1 → 182.6 µs |
| 64 | 405.1 | 314.7 | −90.4 (−22.3%) | — |

The independent review re-ran the same harness and got 305.8/315.0/315.2 µs
(run-to-run noise; −22.1% vs −22.3%). The K2-non-applicable baseline is
unchanged, so the whole gain is in K2: the ≤256-step serial fold replaces the
tree scan, `total = C(kBlock)` deletes the owner re-walk, and pass 2 early-outs
segments with `C(i) >= target`. **The ≤0.10 ms/4-row and ≤0.25 ms/48-row gates
are still missed** (K2 delta ~128 µs at 4 rows against a 100 µs gate), so
chunk 6/7 still owns the pass budget. The fast-path GPU-vs-CPU divergence on the
same 983,040-draw harness moved **12.1193% → 12.3135%** (§4.2, §6.3c) — a
mapping shift from the same fix, not invalidity (0/983,040 invalid selections).
**The number to publish is 12.3135%.**

### 13.2 D2H bytes and synchronizations before/after

Count from code plus a device-side byte counter test:

| Item | Before | After (chunk 4) |
| --- | --- | --- |
| device-served round D2H | `rows × 517,120 B` (`scheduler.rs:401-403`, `independent.rs:143`) | `rows × 16 B` (ids, status, detail, scores) for device rows |
| stochastic / mixed round D2H | `rows × 517,120 B` | `rows × 16 B` for device rows; a **fallback** row keeps its own full row (the CPU re-sample needs it) |
| per-row host materialization | `rows × 517,120 B` host Vec (`scores.rs:47`, `:119`) | 0 for a device row; 1 row for a fallback row |
| finishing frontier | `1 × 517,120 B`, unconditional (`scores.rs:88`, `independent.rs:153`) | **`0 B` when the turn bank is disabled or the row is already packed; `1 × 517,120 B` otherwise** (§10.4) |
| per-step syncs | head `wait()`/`synchronize()` + one pinned async D2H; single-lane adds a blocking pageable `cudaMemcpy` | device round: head `wait()`/`synchronize()` + one pinned async D2H; the single-lane blocking copy for stochastic rounds is removed |
| greedy compact round | `2 × rows × 4 B` (`v41_target_head.rs:367-369`) | unchanged (83.4 µs at 48 rows) |

**Chunk-4 measured removals — TWO counters, never a combined total (MEASURED).**

| counter | scope | meaning |
| --- | --- | --- |
| `frontier_gated_rows` / `frontier_gated_bytes` | both lanes | frontiers the gate skipped because no snapshot could consume them (cache disabled, or client disconnected) — the primary saving |
| `frontier_packed_rows` | serial lane (a reuse, not a saving) | frontiers retained in place from bytes the round already held |
| `frontier_packed_saved_bytes` | **independent lane only** | the removed second transfer of an already-packed fallback row; the serial lane deliberately does not count it |

Measured cost of the removed transfer (release, RTX PRO 6000, pageable `copy_d2h`
of one 517,120 B row): **29.2 µs / row**; a representative 8-finishing-row round
**233.9 µs / 4,136,960 B**. The counters are published **unconditionally** under
the stats JSON's `target_sampling` key (`serving_stats(&PrefixCache)`,
`scheduler.rs:173`), with `host_cache`/`host_cache_config` `null` when no cache is
bound. Stated precisely: the pre-fix nesting under `host_metrics()` published the
object **only when a host cache was attached**, so it was invisible in the
**no-host-cache** deployment. The retention gate is a **separate** knob keyed on
the **turn bank** (`turn_bank_enabled()`, `prefix.rs:65`) — a deployment can have
a host cache with a disabled turn bank (gate active, counters previously
*visible*) or no host cache at all (gate inactive, counters previously *hidden*).
The two conditions are independent; the hidden one was not the gated one.

**Chunk-4 measured sampler latency and the honest crossover (MEASURED).** Same
protocol as §13.1, release, 1× RTX PRO 6000, vocab 129,280. **Only the device
columns are load-independent**; the CPU column is a host-throughput measurement
and must be quoted as a range:

| batch | device round | GPU | CPU (D2H + sample) | speedup |
| ---: | ---: | ---: | ---: | ---: |
| 4 rows, ordered (`0.7 / top_p 0.9 / top_k 40 / min_p 0.05`) | **2,190 µs** | 2,185 µs | 1,460–1,470 µs | **0.67× — the device LOSES** |
| 48 rows, ordered (same profile) | **2,264 µs** | 2,258 µs | 17.3–22.1 ms | **7.65×–9.54×** |
| 4 rows, fast path (K1+K2) | — | **343 µs** | — | — |
| 48 rows, fast path (K1+K2) | — | **356 µs** | — | — |
| 48 rows, greedy compact argmax (chunk-1 fast path) | — | **83.4 µs** | — | — |
| 48 rows, greedy sampled route (K1) | 103 µs | 98 µs | — | — |

- The ordered device path pays K3's up-to-21 dual-probe passes over the
  vocabulary (~1.9 ms of K3/K4/K5 over 48 rows), so below the crossover it is
  slower than the CPU sampler. The measured 4-row and 48-row points **bracket**
  the crossover; contract §3.5 caps a request at ≤6 rows, so the practically
  relevant ordered small-batch regime sits at or just below it — the design
  records the crossover as **≈6 rows** and gives its reduction to chunk 6/7.
  An adaptive "ordered rows + small row count → CPU sampler" route is a
  legitimate chunk-6 option.
- The fast path is already 343 µs at 4 rows, so it is never the loser.
- Greedy is unchanged: the compact lane is byte-identical (83.4 µs), and a greedy
  row on the sampled route costs 98 µs GPU / 103 µs round — the K1 level.
- The device round is essentially all GPU time (1.5–1.9 µs host staging), so the
  cost is the kernels, not the staging.

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
- Constrained (JSON schema) runs at both concurrencies: assert error strings are
  byte-identical where §5.4 says identical, and prefix-supersets otherwise; no
  status change (D1 preserves today's worker error).

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

**Rationale for the split (kernel-first, wiring-last).** The three chunk-1 review
rounds produced a clear signal: the kernel-side artifacts — the kernel, the ABI,
the validator, the device tests — were merge-quality on their first review, while
**every** defect found lived in the daemon wiring: two P0 breaks, a trace
regression, a host-pointer-as-device-pointer launch, a distributed-layout gap, a
stale-mask inheritance, and a launch-shape mismatch. The remaining work is
therefore split so that chunks 2–3b touch only `native/` and the device tests (a
review defect there cannot be confounded with wiring), and **all** daemon changes
land in one consolidated wiring chunk (4) whose gate includes the executed
end-to-end test that chunk 1 declared missing. Kernel chunks are gated on kernel
evidence; the wiring chunk is gated on wiring evidence plus the four §17.5 gaps.

| # | Chunk | Kind | Contents | Acceptance gate | Review |
| --- | --- | --- | --- | --- | --- |
| 0 | **Design gate** | doc | this document | adversarial review passes; corrected facts §2 accepted | required |
| 1 ✓ | **ABI + GPU prepare/greedy — DELIVERED (`9dffad0`)** | kernel + wiring | as planned: `ds41rt_v41_sampler_row_t` + helpers in `v41_sampling_gpu.h`, mask arena, `TargetSamplingWave`, C-ABI registration, FFI wrappers + `validate_v41_sampling_buffers`, `execute_block_sampled`, scheduler/pass wiring, K1 with the greedy/constrained branch. Passed three adversarial review rounds (two found wiring defects, one found a third; all fixed before commit) | **met for the delivered scope**: ABI validated end to end; device greedy/constrained-greedy parity vs the CPU oracle (81 device cases / 6552 assertions; 63-cell greedy parity grid); compact greedy path behaviourally unchanged; no full-row D2H for greedy rows; stochastic path behaviourally unchanged. Four coverage gaps carried to chunk 4 (§17.5) | required |
| 2 ✓ | **KERNEL-ONLY: K2 fast path + device RNG + accumulation-order/`expf` evaluation — DELIVERED (`f0e7902`, on `9dffad0`)** | kernel only | as planned: K1 stochastic reductions with host `ln_min_p`; the K2 fast path; the accumulation-order evaluation (sequential vs parallel segmented scan); the §12.9a `expf` experiment; per-cell mismatch rates at 8,192 seeded draws; device tests plus the FFI-level oracle test against the production CPU sampler. **No daemon/scheduler/scores changes** | **met and reviewed.** RNG **345/345 bit-equal** (extended grid incl. `2^63`, `u64::MAX`); `expf` **declared residual** with numbers (§6.3b: CUDA 15.4928% of weights / max 2 ulp, double 0.0173% / max 1 ulp and 3.37× slower); mismatch rates recorded per cell (§6.3c: 12.1193% overall, `near_uniform` 57.7502%, `tied` 0.0000%); distribution matches the analytic oracle within noise in every cell; device selftest **90 cases / 7,329 assertions** incl. the fast-path grid, seeded-replay, min_p-threshold and fallback regression cases; FFI oracle green (`v41_sampler_device_fast_path_matches_cpu_oracle`); both arches compile. **Honest limits (§18.4):** no daemon/E2E path; `sm_121` compile-only; the sequential full-grid rate is partly model-derived; the design's ≤0.10 ms/4-row gate is **missed** and is now a chunk-6/7 item. **The gate was kernel-only and deliberately did not close the §17.5 gaps** (chunk 4's) | required |
| 3a ✓ | **KERNEL-ONLY: K3 + K4 — DELIVERED (`90f745c`, on `4277898`)** | kernel only | K3 general-k pivot selection (true bound 21 passes, §4.3) and K4 exact-k lowest-id tie prefix plus the **rank-ordered id list** the §4.5 K5 contract consumes (§4.4), with device tests against the CPU oracle | **met and reviewed (mergeable).** Device selftest **219 cases / 44,780 assertions** (MEASURED) incl. the k-grid over 4 vocabularies, thousands-tied/all-tied/`-inf`/mask cases and the length-2 regression; **K3 max 21 passes** of the 32 cap (host recurrence, randomized simulation, device measurement); FFI **set-and-order** oracle green; **mutation-tested** — keep-all-ties and old-termination mutants both fail; both arches compile; purely additive diffs (0 deletions). **Honest limits (§19.4):** token-level end-to-end vs the production sampler deferred to 3b; no daemon/E2E; `sm_121` compile-only; large-k rank placement deferred to chunk 6 | required |
| 3b ✓ | **KERNEL-ONLY: K5 nucleus + rank-order draw — DELIVERED (`ad9ea72`, on `ddae9bc`)** | kernel only | K5 inclusive-prefix top-p (largest-key boundary, no `top_p = 1.0` shortcut, full-`S` fallback) and the rank-order draw with the `nucleus_count - 1` fallback, consuming the chunk-3a rank-ordered id list through the §4.5 entry point; the chunk-3b fix round adds `out_status`, the unconditional row-identity guard, the fixed prefix association and the K2 rewrite | **met and reviewed (mergeable after two documentation/test-message fixes, applied).** Device selftest **401 cases / 47,767 assertions** (the reviewed rev-4 tree measured 400/47,754; the commit added one case); six final-source single-defect mutants all fail (M1–M6) plus auxiliary M7/M8; K5 ordered path **344,064 draws = 14.8016%** overall with the retained domain at **0/147,456 (0.00%)**; `sm_120` + `sm_121` compile clean; per-domain divergence published. **Honest limits (§21.4):** P0-3 portability cannot be exercised on this HMM/ATS host; `sm_121` compile-only; no daemon end-to-end; the non-monotone K2 saturation residual (§4.2); wide-row nucleus delta 28 against a 64 bound; the 71-fixture's kernel `!p_found` branch not directly pinned | required |
| 4 ✓ | **WIRING — DELIVERED (4a `3572101`, 4b `b38e910`)** | wiring | stochastic rows on the sampled terminal with **per-row routing** (§8.4); per-row CPU fallback (planned and post-launch refused) with the token stored and committed; capability gate for layouts without the terminal; constrained mask plumbing (already kernel-complete) now executed-tested; retention gating (§10.4) and the two removed full-row D2H; the executed `upload → launch → output` end-to-end tests; the single-lane blocking stochastic copy removed. Honoured the §20 interface changes | **met for the deliverable scope.** GPU `v41_device` **10 passed / 0 failed**; sampling CPU families **39 passed / 0 failed / 12 ignored**; both arches compile; kernels frozen at the chunk-3b hashes; greedy unchanged (compact 83.4 µs at 48 rows); constrained stochastic draws in-mask on three routes (64/64 each) and constrained-greedy device round token-exact; D2H saving measured as two separate counters (29.2 µs / 517,120 B per gated row). **Honest limits (§22.5):** there is no loaded `TargetPass`/`BlockOutput` fixture (it needs the official head weights plus a backbone output, and that fixture family needs torch/triton, which the container lacks), so `TargetHeadWave::copy_block`, the head graph and `execute_block_sampled` itself are **UNEXECUTED** and the lane-level gate is not run end-to-end; `sm_121` compile-only; residual rates not re-measured (kernels unchanged); full-suite counts are host-dependent and must always be quoted with their command | required |
| 5 ✓ | **Full correctness validation + independent review — DELIVERED (`714fc1d`, evidence-only)** | validation | the complete matrix vs the CPU oracle: token agreement by class, boundaries, ties, masks; seeded replay across batch sizes, row indices, orderings, peers and rejected drafts; distribution (top-64 verdict); speculation; constrained speculation; greedy non-regression. The only product change is the chunk-5b finiteness alignment (`scheduler.rs` `ed870795…`, `v41_target_head.rs` `78cdf67b…`, the latter test-only) | **met and independently reviewed.** 16 shipped modes exit 0; greedy 0/1,264 (+0/480 band); retained domain token-exact at `top_k` 2/40/256 on every row; residual rates recorded per profile; boundary grid 85,050 draws / 0 mismatches; headline figures reproduced (retained 0/147,456, survivor 25.9028%, overall 14.8016%, fast 12.3135%); replay bit-identical; top-64 binned distribution passes all cells; speculation rule confirmed with the correction counterfactual; constrained 20,000/20,000 in-mask; CPU sampler byte-identical to v11. **The harness was hardened so it can fail** (§12.14: per-cell `0.5/draws` bound, 43/46 cells catch a 1.25× regression; the earlier mutant table was forensically withdrawn). **Honest limits (§22.7):** no loaded `TargetPass`/`BlockOutput` fixture (so `copy_block`, the head graph and `execute_block_sampled` remain unexecuted and the lane-level retention gate is not run end-to-end); `sm_121` compile-only; no real-model or HTTP end-to-end run; draft-logit generation unexercised; full-suite counts host-dependent and must be quoted with their command. **Release-note item:** the masked-non-finite alignment is consumer-visible (error → CPU token, §12.6/D9) | required |
| 6/7 | **Pass-budget optimisation (conditional), then measurement — NEXT** | kernel/measurement | **now triggered for the launch-critical-path cost measured in chunk 2** (§13.1): try the levers in order — wider per-row blocks (256→512/1024), fold the owner walk into an existing pass, then the deterministic integer radix histogram (≤2048 buckets, fixed-order combine, u64 fixed-point mass) — and re-measure the per-cell mismatch rate after any segment-count change; then fixed-logit latency vs CPU and the published per-cell mismatch-rate measurement | pass budget ≤ the §13.1 gate after the chosen lever; mismatch rate re-measured for the shipped configuration; §13.7 criteria 3, 4 and 7 | required |
| Phase 3–4 | **E2E campaign + release** | release | end-to-end campaign on 1× RTX + 4× Spark with dSpark (§13.3–§13.5), per-round timing (§13.6), README five-profile measurement update (`:214-260`), `docs/release-v11-performance.md` and release notes (§6.4), and the upstream/placement decision memo (former chunk 8) | §13.7 all criteria; workload identity; provenance/identity files; validated campaign | required |

**Current position:** chunks 1 (`9dffad0`, §17), 2 (`f0e7902`, §18), 3a
(`90f745c`, §19), 3b (`ad9ea72`, §21), **4 (4a `3572101`, 4b `b38e910`, §22)** and
**5 / 5b (`714fc1d`, §22.7)** are all delivered and reviewed; the design gate
(chunk 0) is this document. **Chunks 6/7 — pass-budget optimisation and the
published measurement — are NEXT**, followed by the phase-3–4 campaign; those
remain design-only and still require the adversarial review of §14. The four
chunk-1 coverage gaps (§17.5) are **closed at the test/function-chain level** by
chunk 4b, with the `TargetPass`-level portion explicitly unexecuted (§22.5 and
§22.7).

Do not start a later ordered-path chunk before chunk 2's RNG **and** `expf` gates
pass exactly: the RNG equality test is the only cheap way to separate a
draw-stream bug from a filter bug, and the `expf` experiment is what tells the
implementer whether the remaining token differences are the expected residual or
a real defect. **Chunk 2 satisfied both gates, so chunk 3a was correctly started
and is delivered (§19); chunk 3b inherits the same settled `expf`/accumulation
decisions and must not re-litigate them.**

---

## 15. Risk register

| ID | Risk | Why it bites | Mitigation / evidence |
| --- | --- | --- | --- |
| R1 | **Provenance / pinning of reference algorithms** | FlashInfer is an unpinned `.venv` tree (no submodule, no lock; only a cache-identity hash at `scripts/kernel-cache-identity.py:159-170`). Copying code without recording provenance creates an unverifiable derivation | Port the *mechanism* as new code, keep a `.cuda` header comment naming each FlashInfer anchor, and record the anchors here. If a header must be vendored later, add a lock entry |
| R2 | **Tie-rule drift** | FlashInfer/vLLM keep all k-th ties (`sampling.cuh:945`); ds41rt keeps exactly k lowest-id (`target_sampling.rs:30-33`). A naive port silently changes every tied row | K3/K4 implement the total-order rule explicitly; §12.4 pins it; the reviewer diffs the `reference_select` port |
| R3 | **top-p tail-mass vs inclusive prefix, and boundary direction** | FlashInfer accepts the strict tail-mass criterion (`sampling.cuh:1072`), excluding the boundary tie; ds41rt is inclusive (`target_sampling.rs:544-554`). A monotonicity slip (`M` decreases as the key gets *better*) would select the whole vocabulary | K5 uses the **largest key with `M >= top_p`** = the last rank of the prefix; §4.5 states the direction and the bisection invariant; no `top_p = 1.0` shortcut; §12.3 boundary-uniform probes and the `top_k + top_p = 1.0` overshoot oracle |
| R4 | **Mask-width bug / out-of-vocab id** | The native path never masks the last word's high bits; a whole-word GPU reader could select `id >= vocab` on a non-divisible vocabulary | Bounded loops + host remainder-mask + kernel re-check (§5.3); vocab-33/127/129281 tests |
| R5 | **FP non-bit-identity** | `expf` is not bit-identical (1–2 ulp per weight) and the reduction order differs, so any comparison against an f32 accumulation can move the token — including the top-p membership, which shifts the draw by a rank; in the wide-nucleus case the mismatch rate can be O(0.1–1), not ULP-rare. **Realized and measured** (chunk 2): fast-path 12.1193% overall, `near_uniform` 57.7502% | `min_p` made exact by host `ln` (§6.3a); **DECLARED** in chunk 2 (§6.3b: CUDA `expf` kept, double variant 0.0173% residual at 3.37× cost); measured per-cell rates published (§6.3c, §18.3); Gumbel-max alternative recorded and out of scope (§6.5.2); re-measure after any segment-count change (§13.1) |
| R6 | **Graph-capture issues** | Kernel-node index patching is fragile (`sampling.cu:1203,1268`); captured row count is a key (`v41_target_head.rs:226`) | Sampler launched outside the graph; parameters from device memory; no cooperative launch; §11.3 |
| R7 | **Greedy regression** | Mixed-lane heterogeneity, an extra kernel launch, or per-row parameters could slow greed; forcing a lane-wide full download would be catastrophic | Compact path untouched; per-row params; §13.7 criterion 1; mixed-lane test |
| R8 | **Arch 120/121 build and AOT packaging** | Coordinator builds `CUDA_ARCH=120` (rewritten to `120f` only for the FlashInfer packed-MLA source, `native/CMakeLists.txt:317-319`) and Spark `121` (`build.sh:427`, `:558`); only one native arch per image | New plain `.cu` in `DS41RT_NATIVE_SOURCES` needs no AOT exporter; build and selftest on both images; verify the `120f` rewrite is not accidentally triggered |
| R9 | **Draft RNG co-existence** | `v41_dspark.cu:245-266` already has a Philox stream; mixing the target into it would break domain separation (`target_sampling.rs:182-186`) | **Verified**: the draft Gumbel branch is temperature-gated (`if (temperature != 0)` at `v41_dspark.cu:246`, `:262`) and production passes `0.0` (`speculative.rs:511-512`, `:613`), so no draft draw is consumed today. Target uses only SplitMix64 on `(seed, position)`; draft reservation semantics (`dspark_rng.rs:34-49`) untouched |
| R10 | **Evidence hygiene: `runs/` is gitignored** (`.gitignore:48`) | The phase-0 report, harness rows and campaign artifacts are not in the release tree, so a design or release claim can rest on an artifact a reviewer cannot fetch | Land the fixed-logit rows and the harness (or their sha256 manifest) in the repo before gating on them (§16 D4); keep release-page hashes as the fallback |
| R11 | **Pass-count blow-up** | The ordered path's ternary searches can reach ~74 row reads worst case (and `top_k40` now also pays the top-p search), which could cost more than the 0.55–1.79 ms it removes | §13.1 gate; chunk 6 histogram; §13.7 criteria 3 and 4 |
| R20 | **Masked-non-finite mode drift** | The strict (`scores.rs:407-420`) and permissive (`target_sampling.rs:430-436`) rules differ, so a careless unification silently changes one of them. **This risk materialized** as the chunk-5b F1 finding: the shipped host built the strict flag for masked stochastic rows too (`needs_mask \|\| greedy`), which made the device reject a masked-out non-finite value the CPU accepts | Per-row `STRICT_FINITE` flag (bit4) set per call site; chunk 5b aligned the host to `if greedy` (`scheduler.rs:725-735`) and pinned it with `masked_stochastic_rows_stay_permissive_and_greedy_rows_stay_strict`; §4.1 records the delta; §12.6 pins both; D9 records the choice and the release-note requirement. The greedy/constrained path is never loosened |
| R12 | **Reduction nondeterminism** | f32 atomics or batch-dependent combine orders would break claim (b) | No f32 atomics; fixed contiguous segments and fixed tree combines; chunk 6's histogram uses integer/fixed-point counts |
| R13 | **`EmptyCandidates` status change** | `is_bad_request` has no consumer today, so mapping to 400 would be a behavior change | **Decided (D1): preserve today's worker error (500)**; the 400 mapping is a separate follow-up item. The device status is unaffected either way |
| R14 | **Mask upload pressure** | `rows × 16,160 B` per step (1.3 MB at 48 rows) on the head stream could add latency | One async H2D per step, not per row; measure in §13.1; masks only exist for constrained rows |
| R15 | **Loss of logit-trace observability** | Removing the full-row download would silently break `ds41rt::logit_trace` (`scheduler.rs:565-576`) | **Implemented**: `trace_rows` is all rows under `DEBUG` and empty otherwise (`scheduler.rs:585`), and `download_sampled_rows` fetches exactly that selection (§8.5); untraced all-greedy rounds download nothing |
| R16 | **Frontier retention regression** | Device-side retention or a missing row would break exact-frontier restore (`scores.rs:195-206`, `prefix.rs:248-…`) | Keep the host `TokenScores` and the one-row download (§10); retention tests stay green |
| R17 | **CPU fallback divergence** | If the CPU path can serve the same seeded request, two token streams exist in one build | **Decided (D2): oracle + explicit env-gated A/B fallback only**, never automatic; fail closed when the symbol is missing; the default path never probes the CPU sampler on a miss |
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
in §12.6. **The device plan was aligned to the CPU arbiter in chunk 5b**
(`build_target_sampling_plan`, `if greedy`, `scheduler.rs:725-735`), so the
shipped host now sets bit4 for greedy rows only.

**Served consequence, stated plainly (release-note item).** Pre-change, a
masked-out non-finite logit on a served **stochastic** device-selected row was
admitted by `admit_device_rows` (`scheduler.rs:1075`) and then **hard-errored** in
`check_status` (`scores.rs:80-90`, `"non-finite target logit at token {id}"`),
failing the active requests through `decode_round`'s error fan-out
(`scheduler.rs:396-402`) with **no CPU retry**; stochastic frontier retention
(`retain_from_bytes`, `scores.rs:199-203`) performs no finiteness check either.
Post-change the same request completes and returns **the CPU sampler's token** —
direction **error → token**, matching the CPU arbiter (`target_sampling.rs:430-436`).
The whole-round CPU path still rejects it: `BatchScores::new` is `scores.rs:246-248`
(not `:33-35`, which is `TokenScores::new`), its `argmax` `ensure!` is
`scores.rs:407-420` (`:415`), and its non-test call sites are the whole-round CPU
paths — `execute_logits` (`scheduler.rs:1119`) and the independent lane's
equivalent branch (`independent.rs:161`) — reached only for a compact all-greedy
round, a round with no device-servable row, or a layout without the terminal.
That is why the change is **device-path-only**. Unmasked non-finite logits still
fail `NONFINITE_LOGIT`, and greedy is unchanged.

**The alternative was considered and rejected:** keep the strict bit on masked
stochastic rows and document a device-vs-CPU divergence as accepted. It preserves
the old served behaviour but leaves the device stricter than the reference the
design names as normative, so the shipping choice is the loosening, and **it must
be in the release notes**; the revert (restore `needs_mask || greedy`) remains a
one-line host change. The change is pinned by
`masked_stochastic_rows_stay_permissive_and_greedy_rows_stay_strict`, so it
cannot silently flip back.

**Implementation note (chunk 4) — verification semantics are settled, not open.**
The wiring settled them; the detail and the three defects they caught are in
§12.13. In one line: a stochastic device frontier is **not** argmax-cross-checked
(faithful to the CPU path), the finishing-frontier classifier is
`frontier_retain(round, params)` keyed on the **parameters** and deliberately not
on the route, `store_sampled` recomputes from the row's own logits for every
parameter shape so no stale device slot can be committed, and a whole-CPU round
never reaches the classifier because `has_full_logits()` takes the trusted path.
No further review is needed on these points; a change here is a new defect.

---

## 17. Chunk 1 as-built record (commit `9dffad0`, 2026-09-22)

This section records what chunk 1 actually shipped and reconciles the normative
sections above with it. Commit
`9dffad0236cae8b5398a41bea973b1ff21f2da88` ("Add GPU target sampler chunk 1:
ABI, prepare kernel, greedy device path"), 16 files, `+4673/-71`, on `dev`.
It passed three adversarial review rounds: the first two found wiring defects,
the third one more; all were fixed before commit. Chunk 2 is next.

### 17.1 Files

| File | Role |
| --- | --- |
| `native/cuda/kernels/v41_sampling_gpu.h` | the 64 B ABI, flag/status constants, scratch layout, `ds41rt_v41_sampler_clear_remainder`, `ds41rt_v41_sampler_mask_words`, residency requirement, entry-point declarations |
| `native/cuda/kernels/v41_sampling_gpu.cu` | K1: mask-first predicate, finiteness discipline, temperature scaling, scaled max, `min_p` survivor count, greedy/constrained-greedy argmax |
| `native/tests/v41_sampling_selftest.cu` | the device selftest and its CPU oracle |
| `native/include/ds41rt_native.h` | the `extern "C"` declarations |
| `rust/crates/ds41rt-ffi/src/lib.rs` | wrappers, `Ds41rtV41SamplerRow` mirror, `validate_v41_sampling_buffers`, ABI `sizeof`/offset pins |
| `rust/crates/ds41rt-daemon/src/v41_target_head.rs` | `TargetSamplingWave`, `execute_block_sampled`, `download_sampled_rows` |
| `rust/crates/ds41rt-daemon/src/v41_target_pass.rs`, `.../v41_target_pass/verification.rs` | trait methods + `SUPPORTS_SAMPLED_TERMINAL` |
| `.../v41_native_serve/{scheduler.rs,scheduler/independent.rs,scores.rs,constraints.rs}` | round gating, per-row params/masks, `SampledTargetRows`, status mapping, mask preparation |
| `native/CMakeLists.txt` | source registration, selftest target, arch propagation |
| `.gitignore`, `rust/Cargo.lock` | local build tree + SQLite sidecars ignored; lockfile |

### 17.2 Scope, recorded verbatim

**All-greedy rounds — unconstrained and/or constrained — are device-selected and
greedy rows no longer download full logits. Any round containing a stochastic
member stays entirely on the CPU path, and mixed greedy+stochastic rounds are NOT
wired.** `VerificationTarget::SUPPORTS_SAMPLED_TERMINAL` is an associated const
(default `false`, `TargetPass` `true`) so `DistributedTargetPass` keeps the CPU
fallback and has no sampled terminal yet — a documented limitation for a later
chunk (§8.4).

### 17.3 Deviations from the first-revision design, and why

1. **`flags.bit2` is `NO_MASK`, not "mask remainder-masked"** (§5.1). The
   design's §5.2 wanted `mask_row = 0xFFFFFFFF` for an unconstrained row and
   §5.3 said there is no all-ones fill, so nothing told the kernel to skip the
   arena. `needs_mask == false` is the production signal. A `NO_MASK` row must
   carry `mask_row == 0xFFFFFFFF`, and the validator enforces the
   correspondence. The remainder rule is separate and unconditional, enforced
   host-side immediately before upload by
   `ds41rt_v41_sampler_clear_remainder` (§5.3).
2. **`out_status_detail` uses `DS41RT_V41_SAMPLER_NO_DETAIL = 0xFFFFFFFF` on
   device and is normalized to `0` host-side** (§5.4). A plain `0` could not be
   told from "token 0".
3. **The validator is stricter than §5.1's letter** (§5.4): it rejects `NO_MASK`
   without the sentinel, `mask_row >= rows`, `mask_words_per_row != 0` when no
   mask buffer is supplied, `STRICT_FINITE` on a row that is neither greedy nor
   masked, and non-zero reserved fields, among the rest listed in §5.4.
   **`top_k` is not range-validated yet** — harmless in chunk 1 because K1 never
   selects with `k`; it is a chunk-2+ item.
4. **`params` MUST point at device memory**, and the residency requirement is
   now pinned in `v41_sampling_gpu.h` (§5.1, §5.5). A pageable host pointer
   *happened* to work on this box (ATS/HMM-class:
   `cudaPointerGetAttributes` accepts unregistered pageable memory here), which
   is exactly why the requirement is explicit. Recorded as a portability hazard,
   never a property to rely on; the product path passes the uploaded
   `param_device` buffer.
5. **Launch-shape rule:** `mask_words_per_row` MUST be `0` when no mask buffer is
   supplied; the buffer and the stride are produced by one helper so they cannot
   diverge (§5.1, `v41_target_head.rs:114-120`). An all-unconstrained
   non-compact round (for example a traced all-greedy round) is a real reachable
   case.
6. **`STRICT_FINITE` is functional and additive** (§4.1): a strict stochastic row
   keeps the permissive pass's results and gets an additional whole-row
   finiteness pre-scan. The host sets it only for greedy and constrained rows.
7. **Per-row `needs_mask` propagation** changed the daemon helper signatures:
   `prepare_verification_masks` / `prepare_verification_mask_row` return
   `Option<Vec<u32>>`, and `frontier_downloads` is
   `Vec<(usize, usize, Option<Vec<u32>>)>` (`scheduler.rs:819`). A row that
   needs no mask becomes `NO_MASK`, never an inherited or zeroed mask (§5.2,
   §9.2, §8.1).
8. **Error-string status** (§5.4): `EMPTY_CANDIDATES` and
   `INVALID_TEMPERATURE` are byte-identical to the daemon CPU path;
   `NONFINITE_LOGIT` and `MASK_WIDTH` are deliberate prefix-supersets (they add
   the token id / actual width) and are documented as supersets, not claimed
   identical.
9. **Test-target architecture fix:** the new device-selftest target had been
   defaulting to `sm_75`, so its probe kernel never launched and its FTZ canary
   read zeros. `CUDA_ARCHITECTURES` is now set from
   `DS41RT_CUDA_ARCHITECTURES` on that target (`native/CMakeLists.txt:1051-1058`;
   `arch.log` shows `arch 120 exit=0`, `arch 121 exit=0`). General lesson for any
   new device test target: propagate the configured architecture explicitly.
10. **`.gitignore`** now covers `native/build-sampling/` and `data/*.sqlite-*`.

### 17.4 Evidence and results (MEASURED)

- The K1 kernel was verified against the CPU oracle with **81 device cases /
  6552 assertions** and a **63-cell greedy parity grid**
  (`native/tests/v41_sampling_selftest.cu`;
  `runs/chunk1-scratch/logs/device_selftest.log` —
  `ds41rt_v41_sampling_selftest passed: 81 cases, 6552 assertions`). The counter
  line is the selftest's own output (`v41_sampling_selftest.cu:1253-1254`).
- The stochastic path is **behaviourally unchanged**: it is untouched and still
  serves every stochastic round.
- The compact greedy path is **behaviourally unchanged**.
- Trace behaviour is as designed: `trace_rows` is all rows when
  `ds41rt::logit_trace` is `DEBUG` and empty otherwise, so traced all-greedy
  rounds download exactly those rows and untraced ones download nothing (§8.5).
- Arch checks were run for both built arches (`arch.log`).
- `runs/` is gitignored, so these evidence paths are local; the committed
  normative evidence is the selftest source and its output counter (risk R10).

### 17.5 Honest gaps still open after chunk 1

1. **The `upload → launch → output` join has no hardware-executed test.** The
   individual pieces are covered and the validator/wave paths are unit-tested,
   but no test drives a real `TargetSamplingWave::upload` followed by a launch
   and an `output()` on hardware.
2. **No daemon-level GPU end-to-end constrained-greedy round.** The device
   masked-argmax parity is covered at the kernel/selftest level, but no daemon
   test runs a constrained all-greedy round through the whole terminal and commit
   path.
3. **The distributed sampled terminal is not implemented.**
   `DistributedTargetPass` has `SUPPORTS_SAMPLED_TERMINAL = false` and keeps the
   CPU path.
4. **Mixed greedy+stochastic rounds are unwired.** Any stochastic member keeps
   the whole round on the CPU path, so those rounds still download full rows for
   every member.

These four gaps are carried into the **chunk-4 gate** (§14), because every one of
them is a wiring defect rather than a kernel gap; chunk 2 is kernel-only and
deliberately does not gate on them. None of them is a correctness defect in the
delivered scope; they are unwired or untested surface.

### 17.6 What the later chunks inherit unchanged

The normative text of §4.0 (no fast-math/FTZ), §4.1 (K1 semantics, mode-specific
finiteness), §5 (ABI, masks, outputs, ownership), §6 (RNG and the residual),
§8.4 (the gating rule), §9.2 (mask preparation) and §11 (streams, workspace,
graph legality) now matches the as-built code above and is the baseline chunks
2–3b (kernel-only) and chunk 4 (wiring) must preserve.

---

## 18. Chunk 2 as-built record (kernel-only, reviewed; committed as `f0e7902`)

Chunk 2 (K2 fast path) passed adversarial review and is recorded here **before
chunk 3a/3b start**, because §12.9a makes the `expf` outcome a prerequisite for
the ordered path. It is committed as `f0e7902` on base revision `9dffad0`; no
daemon, scheduler or scores file was touched, and the kernel `.cu`/`.h` plus the
FFI diff are purely additive (`0` deleted lines).

### 18.1 Files and content hashes (hashes of the committed `f0e7902` blobs)

| File | sha256 | Change |
| --- | --- | --- |
| `native/cuda/kernels/v41_sampling_gpu.h` | `a5286f336abe4cd15e70d0759736b2079f5e7dbd74a3a65c4e44568e1ed06b74` | adds `ds41rt_v41_target_uniform` (`:198`) and `ds41rt_v41_target_clamp_uniform` (`:210`) |
| `native/cuda/kernels/v41_sampling_gpu.cu` | `b313fd82b15ba5f011dce284ed6fa4fd3919df7185ffae9cfc9ba9231584bc4f` | adds `v41_sample_categorical_kernel` (`:475`), `k2_applicable` (`:418`), `k2_weight` (`:447`), `tree_max_u32` (`:142`), the launch switch (`:738-744`) and the two build macros (`:42-57`) |
| `native/tests/v41_sampling_selftest.cu` | `48b8cb0b99d195a0fddabe3d8b0eaae55cc55149210c2a0db166f1e34a2716f8` | the K2 test set; existing K1 assertions untouched |
| `rust/crates/ds41rt-ffi/src/lib.rs` | `e9652d0c56bea7d7bc1f010d086e2d73642610de02dd9fdd34a884792ddeb583` | adds the device-test loader (`:18535`) and the production-oracle test (`:19245` now; `:19215` in the reviewed revision) |

**Hash provenance.** These are the blobs actually committed as chunk 2. The
chunk-2 report (`runs/chunk2-scratch/REPORT.md` §1) lists an **earlier** revision
of three of them (`.cu` `0f79fa3b…`, selftest `aac44c9e…`, `lib.rs` `97135f68…`;
the `.h` matches). The committed `f0e7902` is the reviewed behaviour plus the
final test additions, so the commit, not the report table, is the authoritative
chunk-2 revision; line numbers in this section are from it and drift as later
chunks add lines.

Build macros are `DS41RT_V41_K2_SEQUENTIAL_COMBINE` (default 0) and
`DS41RT_V41_K2_WEIGHT_DOUBLE` (default 0), compile-time switches only — neither
is an ABI field, so the 64-byte struct, the validator and the daemon are
unchanged.

### 18.2 Decisions settled by chunk 2

1. **`expf`: declared residual** (§6.3b) — keep CUDA `expf`; no tested path is
   bit-matching and double-`exp` is 3.37× slower for 0.0173% residual.
2. **Accumulation order: ship the parallel segmented scan** (§6.5.3); the
   strictly sequential token-order combine is 8.25 ms at 4 rows and is not a
   candidate. It is retained as a compile knob for diagnosis.
3. **K2's no-crossing fallback is unreachable on both sides** (§4.2 step 3). The
   device derives `total` from the crossing walk's own accumulation, never from
   the tree prefix. The pre-fix device reached the fallback because the tree
   prefix exceeded the walk cumulative by ~3e-5 on a wide row.
4. **Latent sentinel bug fixed** (§4.2 step 4): the per-segment last-survivor
   sentinel was `NO_DETAIL` (`0xFFFFFFFF`), the max-u32, so it won
   `tree_max_u32` whenever trailing segments were empty; it is now `0`.
5. **`ln_min_p` unchanged**: still host-precomputed, no device `logf` (6.3a).
6. **Mismatch rates are measured, not asserted** (§6.3c): 12.1193% overall,
   `near_uniform` 57.7502%, `tied` 0.0000%, with distribution-vs-mapping argued
   from `TV`.
7. **Test conventions** (§12.2, §12.3): GPU-requiring FFI tests are `#[ignore]`d
   with a documented explicit-run command and a loader that fails rather than
   skips; the fast-path temperature grid includes the `1e-5` greedy boundary.

### 18.3 Evidence (MEASURED)

- RNG bit-equality: **345/345** device-vs-host cases including `2^63` and
  `u64::MAX`; `MAX_UNIFORM` clamp pinned separately.
- Device selftest: **90 cases / 7,329 assertions** (K1's 81 cases unchanged and
  green; the count was corrected from an earlier 7,172 draft figure), including
  the fast-path grid, seeded replay, min_p-threshold-token cases, tied/edge rows,
  `k2_applicable == false` rows, and the fallback regression.
- FFI: **7/7** `v41_sampler` tests green, including the production-oracle
  fast-path test; both arches (`sm_120`, `sm_121`) compile.
- Latency and `expf`/mismatch measurements: `runs/chunk2-scratch/out/*.json`
  (sha256 in the report); the six phase-0 logits rows still match
  `out/logits-manifest.json`.
- `runs/` is gitignored (risk R10): the committed evidence is the selftest
  source, the FFI test and the report's hash table.

### 18.4 Honest limits

1. **No daemon or end-to-end path.** Chunk 2 is kernel-only; nothing serves a
   request through K2 yet. That is chunk-4 work, and the §17.5 gaps remain open.
2. **`sm_121` is compile-only** for this chunk; execution was on the coordinator
   (`sm_120`). The Spark arch is not exercised until the phase-3 campaign.
3. **The sequential full-grid mismatch rate is partly model-derived**: the
   `expf` experiment's sequential-order accumulation plus a GPU spot check
   (0/1,536), because a full 983,040-draw sequential GPU run would take ~2.2 h.
   The shipped tree figures are all GPU-measured.
4. **The design's own ≤0.10 ms/4-row gate is missed** (~215 µs K2 delta, ~392 µs
   total); it is now an explicit chunk-6/7 item with the lever list in §13.1.
5. **The mismatch figures are valid only for the shipped 256-thread layout**;
   any segment-count change requires re-measurement (§13.1 caveat).

### 18.5 What chunks 3a/3b inherit

The K2 entry-point placement, mask handling, `NO_MASK`/`STRICT_FINITE` semantics,
the 64-byte ABI and the §4.0 no-fast-math/FTZ rule are unchanged. K3/K4/K5 must
reuse the same `order_key`/total-order primitive (§4.3–§4.5), keep `total` out of
the tree for any categorical walk, and follow the §12.2 test conventions. The
`expf` and accumulation-order decisions of §6.3b/§6.5.3 are settled and are not
re-litigated in the ordered path.

---

## 19. Chunk 3a as-built record (K3 + K4, kernel-only, reviewed; committed as `90f745c`)

Chunk 3a passed adversarial review as **mergeable**, verified by execution and by
mutation testing. It is committed as `90f745c` on base `4277898` (chunks 1 and 2
committed). It adds new **additive** entry points; the chunk-1/chunk-2
`ds41rt_cuda_v41_target_sample[_async]` path, the 64-byte ABI struct and the
daemon are byte-for-byte untouched.

### 19.1 Files and content hashes (hashes of the committed `90f745c` blobs)

| File | sha256 | Change |
| --- | --- | --- |
| `native/cuda/kernels/v41_sampling_gpu.h` | `f9cc3fc2a2829628e20d71f03e3ff58b6af0bfbb9a2798c8848ee06898e458f8` | adds `ds41rt_v41_order_key` / `ds41rt_v41_ordered_value`, the rank-order contract, `DS41RT_V41_TOPK_MAX_PIVOT_STEPS = 32`, and `ds41rt_cuda_v41_topk_select[_async]` |
| `native/cuda/kernels/v41_sampling_gpu.cu` | `deca8344017865cf0c7f3baac810106090c68c4749c3f0c6ea1cd65abb2bf393` | adds `tree_add_u32_pair`, `tree_inclusive_scan_u32`, `k3_scaled`/`k3_survivor`/`k3_eligible`, `v41_sample_topk_pivot_kernel` (K3), `v41_sample_topk_membership_kernel` (K4), `validate_topk_select_args`, and the two entry points |
| `native/tests/v41_sampling_selftest.cu` | `08276847bcce9fb8c43f12233a032315bfdc55f1adaba1b9b573af7285c96716` | adds the CPU top-k oracle, `run_topk_case`, the five K3/K4 tests and the pass-budget print |
| `rust/crates/ds41rt-ffi/src/lib.rs` | `bf640ef0a5e86d1774496e95097f795b33191c852ef206ba9d93382120dfab75` | adds the two K3/K4 FFI wrappers, `validate_v41_topk_select_buffers`, the `TopkRun`/`v41_expected_topk` helpers and the FFI set-and-order oracle test |

`git diff --numstat` for the four paths is `457/0`, `111/0`, `849/0`, `736/0` —
purely additive, **0 deleted lines** in each
(`runs/chunk3a-scratch/REPORT.md` §1; note the committed selftest carries 8 more
lines than the report's `0095b9e8…` / `841` snapshot, so the commit is the
authoritative chunk-3a revision).

### 19.2 Decisions and corrections settled by chunk 3a

1. **K3 worst case is 21 passes, not "≤ 20"** (§4.3), confirmed by a host
   interval recurrence, a randomized host simulation, and device measurement
   against the 32-step defensive cap.
2. **Termination bug fixed** (§4.3): the old `hi - lo < 3` stop could return
   `kth+1` from a length-2 interval (261/20,000 random cases), cutting the tie set
   short. The loop now probes while `hi - lo >= 2` with a load-bearing
   `hi - lo == 2` final probe; seven concrete keys are pinned by
   `test_k3_k4_length_two_interval`.
3. **K4 emits a rank-ordered id list, not a bitmap** (§4.4) — descending
   `order_key`, ties ascending id, `-0.0` → `+0.0`, exactly the CPU's total-order
   permutation; O(k) memory; rank placement is O(k²/256) counting.
4. **K5's entry point is fixed** (§4.5): `ds41rt_cuda_v41_topk_select_async(...)`
   with the exact parameter list, `rank 0` best, `capacity == 0` =
   selection-only, capacity `>= max top_k` enforced by FFI and re-checked in the
   kernel, `kth_value_bits`/`above_count` published, and CPU-identical
   eligibility/no-op rules.
5. **Tie tests assert exact id multiset and order**, not counts, with the
   all-tied case independent of the oracle (§12.4).
6. **Mutation evidence**: a keep-all-ties K4 mutant and the old-termination
   mutant both fail the suite (`runs/review3a/mutA_test`, `mutB_test`).
7. **Validator trust-level note** (§5.4): the raw-C `validate_topk_select_args`
   does not re-check `output_row` bounds while the FFI validator does; chunk 4
   must enforce it at the FFI boundary.

### 19.3 Evidence (MEASURED)

- **Device selftest: 219 cases / 44,780 assertions** — the executed binary's own
  output, matching the committed selftest blob
  `08276847bcce9fb8c43f12233a032315bfdc55f1adaba1b9b573af7285c96716` and
  `runs/chunk3a-scratch/REPORT.md` §4. *(Resolved provenance: an earlier chunk-3a
  revision reported 219 cases / 39,710 assertions; the reviewer asked for one more
  coverage assertion — eligible rows must leave arena entries `[k, capacity)` at
  the sentinel — the implementer added it, and the re-run gave 44,780. The 39,710
  figure was therefore superseded, and 44,780 is the reviewed, committed count.)*
- **K3 pass budget: max 21 of the 32 cap**, host worst-case bound 21 — printed by
  the selftest.
- The chunk-1/chunk-2 suite is independently reproduced unchanged at
  **90 cases / 7,329 assertions** by rebuilding the `HEAD` revision
  (`runs/chunk3a-scratch/selftest_head.cu`), so the additive claim is verified,
  not asserted.
- **FFI set-and-order oracle green**; the default sweep is 5 passed / 3 ignored,
  and the explicit device run is 3 passed (greedy, fast-path, top-k).
- Both arches compile (`sm_120`, `sm_121`).
- `runs/` is gitignored (risk R10): the committed evidence is the selftest and
  FFI test source; the scratch report carries the artifact hashes.

### 19.4 Honest limits

1. **No token-level end-to-end comparison against the production sampler yet.**
   K3/K4 produce the retained set; the first end-to-end token comparison is
   deliberately deferred to chunk 3b, when K5 completes the ordered path.
2. **No daemon or E2E path** — chunk 3a is kernel-only; the §17.5 gaps stay open
   for chunk 4.
3. **`sm_121` is compile-only**; execution was on `sm_120`.
4. **Large-k rank placement is deferred.** O(k²/256) rank counting is fine at the
   served `k = 40` and to `k ≈ 1000`, but sluggish by `k ≈ 4093` and unusable near
   `k ≈ vocab`; wide-k coverage here is selection-only, and the chunk-6 histogram
   owns the large-k case.
5. **Arena sizing and the k-beyond-capacity fallback are undecided** (chunk 4).

### 19.5 What chunk 3b inherits

K5 must consume `rank_order_ids` through the §4.5 entry point, iterate ranks
`0..k-1` recomputing `scaled`/`w` per rank as `sample_from_ranked` does, keep the
largest-key boundary direction and the `top_p = 1.0` strict-prefix rule (§4.5),
and add the token-level oracle comparison that chunk 3a deferred. The `order_key`
canonicalization, the exact-k lowest-id tie rule and the 21-pass K3 bound are
settled and must be pinned unchanged by K5's tests.

---

## 20. Chunk-4 hand-off: interface changes, and their chunk-4 disposition

Chunk 4 (wiring) is the first chunk that calls K3/K4/K5 from the daemon. It must
honour every interface change the kernel chunks settled; this list is the
authoritative hand-off (`runs/chunk3b-scratch/REPORT.md` §8).

1. **K5 `out_status` parameter.** `ds41rt_cuda_v41_nucleus[_async]` takes
   `uint32_t* out_status` after `out_indices`; pass K1's status buffer. A K5-class
   row that cannot produce a token writes `INTERNAL` there **and** to
   `scratch.status`; the entry point still returns `OK` (per-row failure), so the
   daemon must read `out_status`, not the return code.
2. **`params[r].output_row == block_row` is REQUIRED for every K5-class row.**
   The FFI validator rejects a non-identity batch and the kernel reports
   `INTERNAL` unconditionally otherwise (before reading `rank_retained_count`).
   K1–K4 still support scatter; only K5's block-row consumption needs identity.
3. **K1 now writes `INTERNAL`** for a non-finite `top_p` and for a non-greedy
   zero-survivor row, so the K1+K2-only `ds41rt_cuda_v41_target_sample[_async]`
   cannot return OK with an unwritten index. The FFI validator rejects both shapes
   on the host; a raw-C caller sees the per-row status. Chunk 4 must propagate it.
4. **`params_device` on the K3/K4/K5 FFI wrappers** (after `params`): upload the
   64-byte blocks and pass the `Ds41rtDeviceBuffer`; the wrapper validates it and
   launches that pointer (§5.5). Do not pass a host `params.as_ptr()`.
5. **Re-published fast-path divergence: 12.3135%** (was 12.1193%), and the
   ordered-path rate 14.8016% with the retained domain at 0. Publish these beside
   the new numbers in the release notes; they are the §6.3c figures of record.
6. **`top_k ∈ [257, survivor_count)` remains unsupported by K5** and stays **loud**
   (`INTERNAL`); chunk 4 still needs either a wider retained list or an explicit
   CPU fallback for that band.
7. **The four §17.5 gaps and the executed `upload → launch → output` end-to-end
   test remain chunk-4's acceptance gate**, as does the daemon-level
   constrained-greedy round.

**Chunk-4 disposition — what chunk 5 and the campaign now inherit (all seven
items verified).** Items 1–5 are honoured in the shipped wiring: K5's
`out_status` is K1's buffer; the `output_row == block_row` rule has both the FFI
validator and an **unconditional kernel guard**; K1's loud INTERNAL statuses for
a non-finite `top_p` and a non-greedy zero-survivor row propagate to the caller;
all four K3/K4/K5 wrappers take a validated `params_device`; and the published
fast-path divergence for the campaign is **12.3135%** (§6.3c). Item 6 is resolved
as a **counted CPU fallback**: `top_k > 256` is routed to the CPU before the
launch and counted (`planned_fallback_rows`, plus `refused_fallback_rows` for a
post-launch refusal) and logged at WARN — it is no longer an unservable shape and
no longer silent; a raw-C caller of K5 directly still gets a loud `INTERNAL` for
`top_k ∈ [257, survivor_count)`. Item 7 is **partially** closed: the executed
`upload → launch → output` tests, the constrained-greedy device round and the
finishing-fallback-frontier end-to-end test landed in chunk 4b, but no loaded
`TargetPass`/`BlockOutput` fixture could be run, so `execute_block_sampled`
itself and the lane-level retention gate remain unexecuted (§22.5). Chunk 5's
validation matrix and the phase-3 campaign inherit the §22.1 hashes and the
§6.3c/§13.2 figures of record.

---

## 21. Chunk 3b as-built record (K5 nucleus + draw, kernel-only, reviewed; committed as `ad9ea72`)

Chunk 3b passed three adversarial review rounds and is mergeable after the two
documentation/test-message fixes those rounds required (applied). It is committed
as `ad9ea72` on base `ddae9bc` (chunks 1, 2 and 3a committed).
K3/K4/K5 kernel math is unchanged by the fix round — the rev3→rev4 diff contains
only the K1 status branches, the K2 rewrite, the K5 identity guard and
additive/test hunks.

### 21.1 Files and content hashes (hashes of the committed `ad9ea72` blobs)

| File | sha256 | diff vs `HEAD` |
| --- | --- | --- |
| `native/cuda/kernels/v41_sampling_gpu.cu` | `637db1e8c7d8aa51a8a866a65843f1c418a38e9275f24ccfeeb6f8cd8754035c` | +911 / −61 |
| `native/cuda/kernels/v41_sampling_gpu.h` | `0972d432a81dc075d237174dd1628aca41d44717141927fb05f64ae488054d77` | +133 / −5 |
| `native/tests/v41_sampling_selftest.cu` | `6e915ea919313b6970ef85fc6ffa3af42f05ad40d3e3a17bf85e9d251873bda3` | +2015 / −0 (the rev-4 tree's was +1876) |
| `rust/crates/ds41rt-ffi/src/lib.rs` | `9cbcf2677f1c6a056c185f6d6b20b8e70674fe0f30c602dd51a7e149d484b1b8` | +804 / −2 |

The 68 deleted lines are fully accounted for (`ADVERSARIAL-REVIEW-rev4`,
"Deleted-line accounting"): the old K2 tree-combine body/owner walk, the obsolete
`DS41RT_V41_K2_NO_WALK_TOTAL` comment, the old K2 header paragraph, and the two
host-`params` launches. No K1/K3/K4/K5 functional line or arch guard is deleted.

### 21.2 What chunk 3b built and fixed

- **K5 (`v41_sample_nucleus_kernel`)** implements the §4.5 ordered path: the
  inclusive-prefix top-p boundary (largest-key, no `top_p = 1.0` shortcut, full-`S`
  fallback) and the rank-order draw with the `nucleus_count - 1` fallback, over
  the chunk-3a rank-ordered id list.
- **Normalization (headline).** The nucleus mass now accumulates the CPU's
  per-token `fl(w/total)` (`DS41RT_V41_K5_NORMALIZED_MASS` default 1) instead of
  the algebraic `Σw >= threshold·total`; the retained domain becomes
  **token-exact** against the production sampler (0/147,456). Full result and
  Residual A/B in §6.3c.
- **Verified fixes** (detail in §4.2/§4.5 and `REPORT.md`): the K2 zero-weight
  selection (P0-1) via the consistent sequential segment prefix and the
  `cumulative_before < target` rule; the unconditional K5 row-identity guard
  (P0-2); `params_device` on all four K3/K4/K5 wrappers (P0-3); K1's loud
  INTERNAL statuses (P1-4); the K5 test-credibility fixes (P1-5a/b/c, §12.12); and
  the mass-probe race, probe-dependent prefix and zero-uniform fixes.

### 21.3 Evidence (MEASURED)

- Device selftest: **401 cases / 47,767 assertions**, exit 0 as committed (the
  reviewed rev-4 tree measured 400/47,754; rev-3 was 395/47,371); the reviewer
  rebuilt and re-ran it independently.
- **Six final-source single-defect mutants all fail** (M1–M6) plus auxiliary M7
  (conditional identity guard) and M8 (K1 branches removed); the older
  rev3-based mutant files abort first on the new P0-1 test purely as a
  **test-ordering artifact** (verified by removing that call).
- **K5 ordered path: 50,927 / 344,064 = 14.8016%**, with the retained domain
  **0 / 147,456 = 0.00%** (18/18 cells) and the survivor domain
  50,927/196,608 = 25.9028%; the algebraic counterfactual was 19.7202% overall /
  11.4163% retained. Independently reproduced by the reviewer.
- **K2 latency after the rewrite: 305.0 µs at 4 rows / 314.8 µs at 48 rows**
  (−22.3%), baseline unchanged; fast-path divergence 12.1193% → **12.3135%**
  (§13.1). Validity scan: **0/983,040** invalid selections; `TV(device, oracle)
  <= TV(cpu, oracle)` + noise in all 120 cells.
- FFI: **108 passed / 1 pre-existing unrelated `v41_experts` failure / 10
  ignored**; the explicit ignored run is **5 passed**.
- **`sm_120` and `sm_121` compile clean** (rc=0, no warnings).
- `runs/` is gitignored (risk R10): the committed evidence is the selftest and FFI
  source; the scratch report and reviews carry the artifact hashes.

### 21.4 Honest limits

1. **P0-3 portability cannot be exercised here** — this is a unified-memory
   (HMM/ATS) box with no non-HMM host; the code path is now identical to K1's and
   the defect class is closed by construction, but the failing platform was not
   reproduced.
2. **`sm_121` is compile-only**; execution was on `sm_120`.
3. **No daemon/end-to-end run** — chunk-4 wiring is absent.
4. **The non-monotone K2 saturation residual** (§4.2): a below-threshold,
   non-monotone draw in the sub-half-ulp tail class; valid and positive-weight,
   bounded to the last few ulps, declared, with a cheap fix as a chunk-6 candidate.
5. **Wide-row K5 nucleus delta 28** against an asserted bound of 64 (each rank
   ≈7.7e-6 of the mass); token, rank, device-nucleus membership and CDF error are
   asserted.
6. **The 71-equal fixture's kernel `!p_found` branch is not directly pinned**
   (N3): the test asserts the resulting full-set nucleus, which the normal
   crossing would also produce.
7. **Known dead capacity:** `__shared__ float phase[4 * kBlock]` in K5 is 4×
   larger than the used region (~3 KB/block); harmless, noted not changed.

### 21.5 What chunk 4 inherits

Everything in §20. K5's entry point, `out_status`, the identity rule, the
`params_device` wrappers, the K1 statuses and the published divergence rates are
the contract chunk 4 wires to.

---

## 22. Chunk 4 as-built record (per-row routing, fallback, retention gating; 4a `3572101`, 4b `b38e910`)

Chunk 4 is split. **4a** (committed as `3572101`, base `dd457de`) wired
stochastic rows onto the sampled terminal with per-row routing and fallback.
**4b** (committed as `b38e910`, on `3572101`) added the executed constrained-mask
coverage, the retention gate, the remaining full-row D2H removals and the executed
end-to-end tests. The kernels and the FFI are **frozen at the chunk-3b hashes**
throughout. The two reviews that gated it are
`runs/chunk4a-scratch/ADVERSARIAL-REVIEW.md` + `DELTA-REVIEW-2.md` (no
`DELTA-REVIEW-3.md` exists) and `runs/chunk4b-scratch/ADVERSARIAL-REVIEW.md`.

### 22.1 Files and content hashes

**Frozen (unchanged by chunk 4):** `v41_sampling_gpu.cu` = `637db1e8…`,
`v41_sampling_gpu.h` = `0972d432…`, `v41_sampling_selftest.cu` = `6e915ea9…`,
`ds41rt-ffi/src/lib.rs` = `9cbcf267…` (all of the committed `ad9ea72`).

**Chunk-4b (7 daemon files; committed as `b38e910`, hashes verified from the
committed blobs):**

| file | sha256 | diff vs `3572101` |
| --- | --- | --- |
| `v41_native_serve/constraints.rs` | `c69fa646830cf00fe8498751e367649c2b3a7e30f217ff83c7f113ce8754c265` | +61 / −1 |
| `v41_native_serve/prefix.rs` | `cca8de9a6d6ab6f22e4b0f3cf14b5de63b495eaebf9794140af7ab17c008d889` | +10 / −0 |
| `v41_native_serve/scheduler.rs` | `02cb83ef17e5a453b05434262ba64f82e582fb63996500efafd5059dd80ceb1a` | +348 / −33 |
| `v41_native_serve/scheduler/independent.rs` | `1d82e1d4798afd569758b7783958e1fd5850430b00acb395b8926e5a39d0cb71` | +19 / −2 |
| `v41_native_serve/scheduler/layout.rs` | `6cea09116aaa1382808b796cd08bd3c3b318307d6af1aeaca8ae68f1f5608a63` | +1 / −1 |
| `v41_native_serve/scores.rs` | `536245293110f46645a91fc7bd35479d3d28a93c005172561a0edc9e2e877630` | +6 / −1 |
| `v41_target_head.rs` | `abbb5a49ffabf486091ca085010f3b6b28b904d57719d2f899d622d97e4e7a1b` | +549 / −0 |

`git diff --stat` for the **final revision** — the state recorded here — is
**`+994 / −38`** across the seven daemon files (which is exactly the per-file
column above), so the diffstat and the hashes describe the **same** revision. The
progression is `+843 / −24` (chunk-4b revision 1, the state the 4b
`ADVERSARIAL-REVIEW` §6 deletion accounting classified: 24 deletions) →
`+978 / −38` (revision 2, the five review fixes) → **`+994 / −38` (final; the two
extra revisions are the test-only scheduler extension described below). No
assertion, guard, error string or frozen-interface line is removed in any
revision. The 4a revision-3 hashes are in `runs/chunk4a-scratch/REPORT.md` §12.2.

**Resolved provenance for the drift (recorded, not hidden).** An earlier pass of
this section recorded the reviewed revision-2 blob for `scheduler.rs`
(`e7bffc6e…`) and flagged that the live tree had advanced to `b3a6ba76…`. That
drift is now resolved: it was the **authorized, test-only** extension of
`serving_stats_publish_sampling_counters_without_a_host_cache`, which now asserts
the full ten-counter key set for both `PrefixCache::new(0)` and
`PrefixCache::new(8)` inside `#[cfg(test)] mod sampling_tests` (`scheduler.rs`;
the +16 insertions over revision 2). The non-test build, the GPU path and the
frozen kernel/FFI blobs are unaffected; the current `scheduler.rs` hash
(`02cb83ef…`) and the `+994 / −38` diffstat are the recorded final values, and the
other six daemon blobs and the four frozen blobs are unchanged.

**Scope note.** The design document is committed **separately** from the daemon
code, so this file's own hash is not part of the `+994 / −38` daemon diffstat and
must not be folded into it.

### 22.2 What was built

- **Per-row routing and fallback** (§8.4), replacing the whole-round greedy gate:
  `DeviceGreedy` → K1, `DeviceFastPath` → K1→K2, `DeviceOrdered` → K1→K3→K4→K5,
  `CpuFallback` → CPU; per-row CPU re-sample for planned and post-launch-refused
  rows, stored into `best[row]`; `MAX_RETAINED = 256`; counted and WARN-logged.
- **Retention gating** (§10.4) and the **two removed full-row D2H transfers**,
  counted separately (`frontier_gated_*`, `frontier_packed_*`), with
  `target_sampling` published unconditionally.
- **Constrained masks:** the kernel path was already complete; 4b added the
  executed coverage (§9.2).
- **Executed end-to-end tests** (the 4b gate): mixed-batch
  `upload → launch → output` with device read-back of params/masks; the finishing
  planned-fallback frontier through `admit_device_rows` → `resolve_fallback_rows`
  → `frontier_retain` → `retain_packed_frontier` / `retain_recorded_from_device`;
  the constrained-greedy device round (chunk 1's outstanding gap); the four-route
  empty-mask hard error; masked stochastic in-mask sweeps on fast/ordered
  paths; and a real-xgrammar per-row fork/rollback test.
- **4a review fixes** (all applied): the fallback write-back (FIX 1), the
  route-keyed frontier classifier (delta-review P1), the counters (FIX 4), the
  mixed-round statistic (FIX 5), plus the latent `store_sampled` hardening.

### 22.3 Evidence (MEASURED)

- GPU `v41_device`: **10 passed / 0 failed** (4a was 5).
- Sampling CPU families: **39 passed / 0 failed / 12 ignored**.
- Native selftest: **401 cases / 47,767 assertions** (kernels frozen).
- FFI: **108 passed / 1 pre-existing unrelated / 10 ignored**.
- Both arches compile (`sm_120`, `sm_121`); all four frozen hashes match.
- Greedy compact **83.4 µs** at 48 rows, greedy sampled route 98 µs GPU / 103 µs
  round; fast path **343 / 356 µs**; ordered **2,190 / 2,264 µs**.
- Full-suite counts are **host-dependent** and must always be quoted with their
  command (this container `877 / 7 / 111`; 4a's host `871 / 7 / 106`; the 7 are
  pre-existing missing-`torch` fixtures; zero failures in this chunk's families).

### 22.4 Latency and the crossover

See §13.2. The ordered device path loses below the crossover (2,190 µs vs
1,460–1,470 µs at 4 rows) and wins 7.65×–9.54× at 48 rows; the crossover is
bracketed between 4 and 48 rows and is ≈6 rows at the served-shape boundary.
Reducing it is **chunk-6/7** work, including an adaptive small-batch CPU route
for the ordered class. Greedy and the fast path are never the losers.

### 22.5 Honest limits

1. **No loaded `TargetPass`/`BlockOutput` fixture.** It needs the official head
   weights plus a backbone output, and that fixture family needs `torch`/`triton`,
   which the container lacks. `TargetHeadWave::copy_block`, the head graph and
   `execute_block_sampled` itself are therefore **UNEXECUTED**, and the
   lane-level retention gate is not run end-to-end (it is covered by the pure
   policy table, `turn_bank_enabled`, the byte-export, publish-without-cache and
   `retain_packed_frontier` tests plus the 4a recording double).
2. **`sm_121` is compile-only.**
3. **No served HTTP / full-model round.**
4. **The residual rates were not re-measured** (kernels byte-identical): the
   §6.3c numbers remain those of record, and the device suite's `0/256` and
   `0/4096` on `smooth_narrow` are **wiring** results, not rates.
5. **The lane-level D2H saving is measured in isolation** (29.2 µs / 517,120 B
   per row); the in-situ number needs a served workload (`ds41rt::timing`).
6. **The counters are verified at the payload level**, not through a running
   `serve` loop.
7. **The distributed layout's retention gate is not exercised** (no sampled
   terminal).
8. **Two cosmetic counter inconsistencies are recorded, not fixed:** a
   single-lane `compact` round increments `cpu_fallback_rounds` even though
   nothing fell back (the counter really means "rounds that did not use the device
   terminal"), and the independent lane's `compact` branch records nothing. Neither
   claims device work; `device_rows` stays 0 on both.

### 22.6 What chunk 5 and the campaign inherit

The §20 disposition and §22.1/§22.3: `output_row == block_row` is mandatory for
K5-class rows (FFI validator plus an unconditional kernel guard); K1 writes loud
`INTERNAL` for a non-finite `top_p` and a non-greedy zero-survivor row; the
K3/K4/K5 FFI wrappers take a validated `params_device`; `top_k > 256` is a
**counted CPU fallback** (`planned_fallback_rows` / `refused_fallback_rows`,
WARN-logged), not an unservable shape; and the fast-path divergence published for
the campaign is **12.3135%** (ordered overall 14.8016%, retained domain 0.00%).

### 22.7 Chunk 5 / 5b validation record (evidence-only; product change `714fc1d`)

Chunk 5 is a **validation** chunk: it changed no kernel and no sampling math. Its
only product change is the chunk-5b alignment commit `714fc1d` ("Align the device
sampler's finiteness rule with the CPU reference"), 2 files — `scheduler.rs`
`ed870795f5731f4ed6d1dd2858c6001b4144775748a089d5ca9f2ad7963d4cbb` and
`v41_target_head.rs`
`78cdf67b0a308daffcbbafc6877b0fe2b66c880fa9c2777094214afd78ce5725` (the latter
the `#[cfg(test)]` recording double). Kernels and FFI stay frozen at chunk 3b:
`.cu 637db1e8…`, `.h 0972d432…`, selftest `6e915ea9…`, `lib.rs 9cbcf267…`. The
CPU oracle `target_sampling.rs` is `e7382400…`, byte-identical to `v11`, so the
oracle **is** the production sampler, not a port. Sources:
`runs/chunk5-scratch/REPORT.md`, `REPORT-5b.md`, `REPORT-5b-fixes.md`,
`review/REVIEW.md`, `review5b/REVIEW-5b.md`.

**Token agreement by class (MEASURED).**

| class | draws | mismatches | rate |
| --- | ---: | ---: | ---: |
| greedy, incl. epsilon band (1,264 pure + 480 band) | 1,744 | **0** | 0.000000 |
| retained `top_k 40` / `top_k 40 + top_p` (all masks) | 49,152 each | **0** | 0.000000 |
| `T0.2 + top_p0.95` (survivor) | 49,152 | 15,552 | 0.316406 |
| `T0.7 + top_p0.9` (survivor) | 49,152 | 17,131 | 0.348531 |
| `T0.7 + min_p0.05` (fast) | 49,152 | 8,029 | 0.163350 |
| `T0.7 + top_k40` | 49,152 | **0** | 0.000000 |

The whole retained/top-k domain is token-exact **at `top_k` 2/40/256 on every
row including `near_uniform`**. The §12.3 boundary grid runs **85,050 device
draws with 0 mismatches** (`DeviceGreedy` 12,150, `DeviceOrdered` 70,875,
`DeviceFastPath` 2,025). The four headline figures reproduce exactly: retained
**0 / 147,456 = 0.00%**; survivor **50,927 / 196,608 = 25.9028%**; overall
**50,927 / 344,064 = 14.8016%**; fast path **121,047 / 983,040 = 12.3135%**.

**Replay and determinism (MEASURED).** 128 identical rows in one launch all
identical; 50 repeated launches bit-identical; batch sizes 1–128 with the target
at first/middle/last (21 placements) give one id; forward vs reversed order with
interleaved peers is identical at every absolute position; a 48-row mixed wave
(masked + unmasked, five param seeds) is **bit-identical over 100 repeats**; a
different seed differs. The three reproducibility claims stay distinct: **(a)**
CPU bit-identity is asserted for the uniform stream, mask, `scaled`, finite
checks, `min_p` and every count-only decision but **not** the token; **(b)**
within-GPU replay is **asserted**; **(c)** spec/non-spec parity is **not
claimed**.

**Distribution (MEASURED).** The **top-64 binned TV is the verdict** and passes in
all 12 cells. The raw 129,280-support TV is now disclosed only as a
**support-coverage floor**, never a verdict: its null (what a perfectly correct
device produces) has mean ≈0.9386 on the wide rows, and the old bound
`noise·1.10 + 1/draws ≈ 1.03 > 1.0` could never fail. The f64 analytic reference's
own error is bounded only as "**no resolvable model error above the ~0.012
sampling floor at 65,536 draws** for the three checked retained cells"
(`out5b/reference_precision.json`), where the residual shrank as `1/√draws` and
device and CPU are token-identical. The wide-row **CPU** inexactness was verified
with production CPU primitives, independently of the device: `tied` CPU rank
−140 vs device +1, `near_uniform` −8 vs 0.

**Speculation (MEASURED).** The shipped **emit-the-target-draw** rule preserves
the target distribution with exact sample-and-match, so **no full-draft
correction is required**; this is the §12.9/§6.5.2 disposition confirmed on
device. Acceptance: point mass 0.4615 (45,017/97,545) vs 0.4592 theory (~1.4σ);
non-point-mass 0.0317 (3,170/99,999) vs 0.0345 theory (~4.7σ — a small excess
over the independence model, recorded as such, not a comfortable match). The
counterfactual delimits when a correction **would** be needed: a standard
publish-the-draft sampler measures `TV 0.0569` uncorrected vs `0.0026` corrected
(noise 0.0027). Honest limits: the synthetic setup precomputes the target sequence
and cycles a fixed 64-token draft, so **draft-logit generation, the model/context
path and the real rejection context remain unproven**, and the
`emitted == sequential` assertion is near-tautological by construction (the
non-tautological content is the acceptance-rate and distribution measurement).

**Constrained speculation (MEASURED).** 20,000/20,000 device draws inside their
own per-row mask with no inheritance across 16 distinct masks; fork/commit/
rollback enforced (5,715 rounds, 2,857 rollbacks, next anchor = last committed
token); an all-zero mask remains a hard error on **every** route.

**Greedy non-regression and the oracle.** 0 mismatches, lowest-id ties, compact
argmax **83.10–83.41 µs** at 48 rows (v11 recorded 83.4 µs), greedy sampled route
97.5 µs. The CPU sampler is byte-identical to the v11 release revision
(`target_sampling.rs` `e7382400…`; `git log v11..HEAD` on it is empty).

**Harness power, and the mutant-forensics correction.** See §12.14. In one line:
exact domains hard-fail, residual cells use a per-cell `baseline + 0.5/draws`
bound with no clamp (a 1.25× regression fails 43/46 non-zero-baseline cells and
all 4,250 zero-baseline cells fail on a single new mismatch), and the earlier
review's "K2/K5" mutants were the **same dead sequential-combine file**, so its
"1 of 3 caught" table is not evidence; the corrected matrix catches the real K2
comparison (`:668`), the K5 retained comparisons (`:1812`/`:1819`) and the
survivor predicate (`:1336`), with the K5 equality reached naturally at seed
20260922 / position 519.

**The release-note item.** The masked-non-finite alignment is
**consumer-visible** — a masked-out non-finite logit on a served stochastic
device-selected row goes from hard request failure to the CPU-agreeing token
(§12.6, D9) — and **must be in the release notes**.

**Runner-path note.** The path documented in the original script,
`rust/target/release/chunk5-campaign`, was a **stale pre-hardening binary**;
every recorded post-hardening number comes from
`runs/chunk5-scratch/harness/target/release/chunk5-campaign`
(`REPORT-5b.md` §R11).

**Honest limits.** (1) No loaded `TargetPass`/`BlockOutput` fixture, so
`copy_block`, the head graph and `execute_block_sampled` remain **unexecuted** and
the lane-level retention gate is not run end-to-end. (2) `sm_121` is
compile-only. (3) No real-model or HTTP end-to-end run. (4) Draft-logit
generation is unexercised. (5) The full-suite counts are **host-dependent** and
must be quoted with their command. (6) The error→token change is established from
the shipped control flow and the CPU arbiter, not observed on a live request.
(7) The 3 saturated `baseline = 1.0` cells are arithmetically unfailable (a data
ceiling, not a clamp).

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

For K2 the tree scan produces only the per-segment carries; **`total` must come
from the crossing walk's own accumulation at the last survivor, never from the
tree's inclusive prefix** (§4.2 step 3). That is what makes the device no-crossing
fallback unreachable exactly as on the CPU: the CPU's `total` and crossing
`cumulative` are the same left-to-right accumulation, so at the last allowed token
`cumulative == total` and the clamped uniform cannot overshoot it. Deriving
`total` from the tree introduces a device-only artifact (measured ~3e-5 on a wide
row pre-fix).

Deterministic for any grid shape; order-exact within a segment; the cross-segment
tree is one of the two accumulation-order differences from the CPU (the other is
`expf`) — §6.3c. Any change to the segment count changes the tree association and
therefore the uniform→token mapping; determinism is preserved and it is
contract-§7.2.6-legal, but the published GPU-vs-CPU mismatch rate must be
re-measured afterwards (§13.1).
