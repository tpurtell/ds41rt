# GPU target-sampler contract (derived from the v11 CPU reference)

**Read-only analysis.** No code, config, cluster or git state was modified. Every
claim below cites the source as it exists in this checkout. Where the code is
ambiguous the ambiguity is stated together with the check that would resolve it.

**Scope.** The production sampling hot path is the `serve-native` full-vocabulary
sampler: `ds41rt_core::TargetSamplingParams` (`rust/crates/ds41rt-core/src/target_sampling.rs`)
driven from the native scheduler (`rust/crates/ds41rt-daemon/src/v41_native_serve/scheduler.rs`,
`.../scheduler/independent.rs`) with materialised logit rows from
`rust/crates/ds41rt-daemon/src/v41_native_serve/scores.rs`. A GPU implementation
must reproduce *this* module's semantics. There is a pre-existing, older GPU
sampler (`ds41rt_cuda_logits_sample_topk_topp_f32*`, `native/cuda/kernels/sampling.cu:598`)
used by the legacy `real-ds4-full` path; it is **not** semantically compatible
with the v11 contract (see §4.6) and must not be promoted as-is.

---

## 0. Where the release semantics are frozen

| Fact | Anchor |
| --- | --- |
| Order: mask → temperature → `min_p` → `top_k` → `top_p` | `target_sampling.rs:9-25` |
| Greedy epsilon `1e-5` | `target_sampling.rs:49`, `172-174` |
| Top-k tie deviation from vLLM (keep exactly k, lowest id wins) | `target_sampling.rs:30-33` |
| Determinism: draw is a pure function of `(seed, position)` | `target_sampling.rs:35-38` |
| Domain separation constant | `target_sampling.rs:182-199` |
| Served parameter resolution (defaults, `top_k` -1/0, signed seed) | `rust/crates/ds41rt-api/src/native_v41.rs:139-188` |
| v11 release wording for the above | `docs/release-v11-notes.md:28-43` |
| "same logits + same execution path" guarantee; no spec/non-spec bitwise parity | `docs/release-v11-performance.md:228-249` |
| Full filter chain honoured under speculation (min_p claim) | `docs/release-v11-notes.md:81-86`, `docs/release-v11-performance.md:245-247` |

---

## 1. Filter pipeline contract

Entry points: `TargetSamplingParams::select_token` (`target_sampling.rs:202-209`) →
`select_token_with_uniform` (`:213-243`) → greedy `argmax_allowed` (`:246-265`) or
`sample_allowed_internals` (`:421-523`) → `sample_from_ranked` (`:527-571`) /
`sample_categorical` (`:590-619`).

### 1.1 Exact order of operations

1. **Empty vocabulary** → `EmptyVocabulary` (`:219-221`).
2. **Mask width validation** — required words = `vocab.div_ceil(32)`; mismatch →
   `MaskWidth` (`:222-230`; oracle `:1245-1253`).
3. **Greedy short-circuit** — `is_greedy()` (`:234-236`) *before* the uniform
   finiteness check. `is_greedy()` = `temperature < 1e-5 || top_k == Some(1)`
   (`:172-174`). Greedy is exactly `argmax_allowed`: highest logit, first
   occurrence wins on ties (`:259-262`), non-finite logit → `NonFiniteLogit`
   (`:256-258`), no allowed token → `EmptyCandidates` (`:264`).
   **Greedy never reads the draw**, so a NaN uniform is accepted in greedy mode
   and rejected only in stochastic mode (test `:779-797`).
4. **Draw finiteness** (stochastic only) — `!uniform.is_finite()` →
   `InvalidParameter("uniform")` (`:237-241`).
5. **Temperature scaling** — `inv_temperature = 1.0 / temperature`
   (`:427`); every allowed logit is scaled as `logit * inv_temperature`
   (`:438`, `:457`, `:485`, `:507`). *Multiplication by the reciprocal, not
   division.* Non-finite logit anywhere allowed → `NonFiniteLogit` (`:434-436`).
6. **`max_scaled`** = max of scaled allowed logits (`:428`, `:438`). If the max
   is not finite → `InvalidParameter("temperature")` (`:445-448`); a grammar with
   no allowed token is reported as `EmptyCandidates` *before* that check
   (`:440-444`).
7. **`min_p`** — `min_scaled = max_scaled + ln(min_p)` when `min_p > 0`, else
   `-inf` (`:449-454`); survivor test `allowed(token) && logit*inv_temperature >= min_scaled`
   (`:456-458`). `min_p = 0` keeps `-inf`, i.e. the whole allowed row even when a
   scaled logit is `-inf` (test `:1748-1764`).
8. **`top_k`** — `None` disables. `Some(k)` with `k >= survivor_count` truncates
   nothing and falls through (`:477-501`). Otherwise a bounded capacity-`k`
   min-heap keeps the k best survivors (`:479-497`).
9. **`top_p`** — see §1.3.
10. **Draw** — see §1.3 / §1.4.

### 1.2 Renormalization behavior

Two distinct paths, both *bit-equivalent* to the verbatim pre-optimization
oracle `reference_select` (`:1235-1399`), asserted by
`optimized_paths_are_bit_identical_to_reference_on_adversarial_rows`
(`:1527-1571`).

- **Fast path** (`top_k.is_none() && top_p >= 1.0`, `:463-466`):
  `sample_categorical` computes `total = Σ exp(scaled - max_scaled)` in token
  order, floors `total` at `1e-20` (`:603`), computes `target = uniform * total`
  and walks the same token order accumulating `exp(scaled - max_scaled)`
  (`:597-618`). No intermediate normalization (the oracle's fast path is
  identical, `:1308-1342`). The returned `total` is a diagnostic only.
- **Ordered path** (any `top_k` or `top_p < 1.0`): weights
  `w_i = exp(scaled_i - max_scaled)` over the ranked list (`:534-537`),
  `total = Σ w_i` floored at `1e-20` (`:538-539`), then **normalized**
  `w_i /= total` (`:540-542`); `top_p` is applied to the normalized weights and
  the draw is on `target = uniform * nucleus_mass` within
  `[0, nucleus_mass]` (`:544-565`). The oracle divides at the same place
  (`:1361-1365`) and reports the same `total`.

Consequence the GPU kernel must respect: for `top_p < 1.0` the nucleus boundary
comparison happens on **renormalized** weights, and the final
`out_scores = prob / nucleus_mass` (`native/cuda/kernels/sampling.cu:684`) is a
*third*, purely diagnostic quantity that neither the token nor the oracle uses
(`RankedSample.total`/`nucleus_count` are `dead_code` outside tests,
`target_sampling.rs:403-410`).

### 1.3 `top_p` details

- `top_p` is `clamp(1e-6, 1.0)` in the ordered path (`:545`) and in the oracle
  (`:1366`). `top_p = 1.0` is the disabled bound (`has_filters`, `:177-179`;
  served default `:98`), but note the fast path is selected by `top_p >= 1.0`
  (`:463`), so the clamp is only reachable for `top_p < 1.0` — i.e. only for
  values in `(0, 1)` after validation (`:118-120` allows `(0, 1]`).
- The nucleus is the smallest descending prefix whose normalized mass is
  **`>= top_p`** (`:548-554`, oracle `:1369-1375`); at least one token is always
  included. The draw selects the first rank with `target <= cumulative`
  (`:559-565`), so an exact-boundary uniform belongs to the **earlier** rank.
- Promotion evidence for the ordered path: `docs/release-v11-performance.md:154-190`.

### 1.4 `top_k` spellings and boundary behavior

- **Disable spellings, two layers:**
  - API/raw-body: `top_k` absent, `null`, `0`, or `-1` → `None`; `< -1` → 400;
    non-integer → 400 (`native_v41.rs:162-176`). Pinned by
    `explicit_filters_and_signed_seed_are_resolved`
    (`rust/crates/ds41rt-api/src/tests/upstream_native_v41.rs:166-191`) and
    `out_of_range_top_k_is_accepted_as_a_no_op` (`:193-207`).
  - Core: `None` disables; `Some(0)` is an invalid parameter
    (`target_sampling.rs:121-123`, test `:819-835`).
- `k >= survivor_count` is a **no-op** (`:499-501`); `k >= vocab` is accepted and
  tested at `:682-699`.
- **Exact ties at the k-th boundary:** ds41rt keeps exactly `k` candidates with
  the lowest token id winning; vLLM keeps every token equal to the k-th value.
  This is a deliberate, documented deviation (`:30-33`). The comparison sort
  uses `partial_cmp` with ascending-id tie-break (`:323-330`); the radix path
  relies on stable LSD over a total key so equal keys keep ascending ids
  (`:362-401`); the heap comparator is the same ranking (`:283-319`). Any GPU
  ordering must reproduce the order for **distinct** scaled values; for exactly
  tied scaled values within the survivors it must be deterministic and, if it is
  to match the CPU, id-ascending.

### 1.5 Zero mass, `-inf`, NaN prevention

- `total` is floored at `1e-20` in both paths (`:539`, `:555`, `:603`) so a
  single dominant token cannot produce `0/0`.
- A pathologically tiny temperature with a huge logit scales to `±inf`; the
  scaled maximum then fails `is_finite()` and is reported as
  `InvalidParameter("temperature")`, explicitly *never* NaN and never a panic
  (`:445-448`, test `:755-776`).
- Non-finite allowed logits are always rejected (`:434-436`, `:256-258`); a
  **masked** non-finite token is never read (test `:1009-1024`).
- `-inf` scaled logits are legal survivors when `min_p = 0`, and every ordered
  branch must match the oracle on them (test `:1748-1764`).
- Ties at the maximum are all retained by `min_p = 1` (`linf` threshold equality,
  test `:737-753`).

### 1.6 Draw representation and the signed-seed convention

- The uniform is the **top 24 bits** of a SplitMix64-style mix, scaled by
  `2^-24`, i.e. `[0, 1)` (`:187-199`). It is clamped for the draw with
  `MAX_UNIFORM = 0.999_999_94` and `max(0.0)` in `sample_allowed_internals`
  (`:51-52`, `:459`); the clamp matters for how a near-1 draw resolves against a
  final cumulative boundary. **Ambiguity:** in the ordered path the *clamped*
  local is used for `target`, while the oracle's ordered branch uses the raw
  `uniform` after the same clamp (`:1307` vs `:1383`); the fast path clamps too
  (`:1307`, `:1316`). The comparison is identical, but a device kernel must apply
  the same clamp or document it.
- `seed_from_i64(seed) = seed as u64` — full two's-complement mapping, so
  **`-1` is a real deterministic seed** (`:161-167`, test `:701-725`, release
  wording `docs/release-v11-notes.md:33-35`). This is an intentional difference
  from vLLM, where `-1` means "unseeded".
- `seed = 0` is likewise a real seed, never a sentinel (`:727-735`).
- Absent/`null` seed → time-based seed mixed with a global counter
  (`native_v41.rs:129-137`, `:177-185`).

### 1.7 Greedy parity guarantees

- `TargetSamplingParams::greedy()` = `temperature 0, top_p 1, top_k None, min_p 0,
  seed 0` (`:96-104`) and is the `Default` (`:88-92`).
- Served default is greedy unless `temperature` is present and non-zero
  (`native_v41.rs:145-159`; tests `upstream_native_v41.rs:154-163`).
- `TokenScores::sample` and `BatchScores::sample` short-circuit greedy straight
  to the existing argmax (`scores.rs:44-46`, `:112-114`), so a greedy request
  never materialises an `f32` row and never consults the draw.
- Device greedy tie-break is lowest id (`native/cuda/kernels/sampling.cu:568`,
  `:583-587` with `CheckFinite`), matching `argmax_allowed`. The scheduler
  additionally cross-checks the downloaded compact id against a CPU argmax of
  the same row when a frontier is retained (`scores.rs:61-66`, error
  `"GPU and retained CPU greedy selection differ"`).

---

## 2. Randomness contract

### 2.1 How a target draw is keyed

`random_uniform(seed, position)` (`target_sampling.rs:187-199`):

```
mixed = seed.wrapping_add(0x7f4a_7c15_9e37_79b9)      // domain
            .wrapping_add(position.wrapping_mul(0x9e37_79b9_7f4a_7c15))
            .wrapping_add(0x9e37_79b9_7f4a_7c15)
mixed = (mixed ^ (mixed >> 30)) * 0xbf58_476d_1ce4_e5b9
mixed = (mixed ^ (mixed >> 27)) * 0x94d0_49bb_1331_11eb
mixed ^= mixed >> 31
uniform = (mixed >> 40) as u32 as f32 * (1.0 / 16_777_216.0)
```

The key is exactly `(request seed, absolute emitted-token position)`. `position`
is **never** a batch-local row index — this is stated in the module contract
(`:35-38`) and pinned by `uniform_depends_on_position_not_on_batch_layout`
(`:998-1007`) and by the scheduler test `sample_target_rows_keys_draws_on_absolute_position`
(`scheduler.rs:666-691`), which also asserts that row-keyed draws differ from
emitted-index draws.

### 2.2 Position plumbing

- The first generated token is emitted-token index `0` (`scheduler.rs:323-324`).
- Per round, `base_position = request.generated` (`scheduler.rs:542`) and row
  `index` uses `base_position + index` (`scheduler.rs:508-518`, constrained twin
  `constraints.rs:80-102`).
- `request.generated` advances **only** in `emit_one` (`scheduler.rs:77-108`,
  `:82`), once per actually emitted token, after the whole commit decision is
  made (`scheduler.rs:478-486`, `independent.rs:190-212`). Therefore
  `base_position + index` is the true absolute emitted index, and a round that
  emits fewer tokens than it verified simply leaves the later positions unused
  rather than shifting them.
- Prefill/first-token sampling uses position `0` (`scheduler.rs:324`).

### 2.3 Target-versus-draft domain separation

- **Target**: `seed` = served request seed (signed→u64), domain constant
  `TARGET_SAMPLING_DOMAIN = 0x7f4a_7c15_9e37_79b9` folded into the SplitMix64
  mix (`target_sampling.rs:188-193`). Uniform, stateless, host-side.
- **Draft (dSpark)**: a per-request Philox subsequence stream,
  `DsparkRng::new(seed)` with `SUBSEQUENCES_PER_DRAFT = 5 * 256`
  (`rust/crates/ds41rt-core/src/dspark_rng.rs:5-42`), seeded from the **daemon
  request id** (`rust/crates/ds41rt-daemon/src/v41_native_serve/speculative.rs:104-108`),
  reserved per draft width (`speculative.rs:507-513`,
  `rust/crates/ds41rt-daemon/src/v41_experts/dspark/terminal.rs:148-175`) and
  uploaded as `(seed, first_subsequence)` pairs. Draft temperature is passed as
  a separate array (`terminal.rs:169-172`); the native serve call passes `0.0`
  (`speculative.rs:511-512`).
- **Why separation is required:** the draft and the target may share the same
  integer seed space (draft = request id, target = request seed), and the target
  uniform is a pure hash. Without the domain constant a shared integer would
  correlate the acceptance draw with the proposal draw. The comment states this
  explicitly (`target_sampling.rs:182-186`).
- **Precedent not to copy:** the legacy path's equivalent draw omits the domain
  constant entirely (`rust/crates/ds41rt-api/src/real_full.rs:67-78`: `mixed =
  seed + step*K + K`, no domain term), so it produces a *different* draw for the
  same `(seed, position)`. It also takes `decode_step_index` from the batch
  (`real_full.rs:100-116`, `rust/crates/ds41rt-daemon/src/commands/real_full/entry.rs:100-116`),
  which is a layout-dependent key. A GPU kernel that copies that pattern silently
  changes every seeded request.

### 2.4 Why a rejected draft must not shift future draws

Because the draw is keyed on `(seed, absolute position)` and not on a consumed
counter, a rejected row consumes nothing. The loop only advances
`position += decision.emitted.len()` (`target_sampling.rs:1199-1201`), and the
pinned test `speculative_sample_match_equals_sequential_sampling` (`:1100-1208`)
asserts over 24 positions, constrained and unconstrained, with deliberate
mismatches on odd rounds, that the speculative sequence equals sequential
sampling at the same absolute positions. Any device RNG that advances a mutable
per-request state on rejection (or on every verified row) violates this
invariant.

---

## 3. dSpark verification contract

Implementation: `rust/crates/ds41rt-core/src/dspark_verify.rs`. Production caller:
`prepare_commit_lane` (`scheduler.rs:528-607`, call at `:562-564`).

### 3.1 Exact sample-and-match semantics

`verify_dspark_greedy(inputs, target_next, eos=1, remaining)` (`dspark_verify.rs:18-52`):

- `inputs[0]` is the already-emitted **anchor** (not re-emitted);
  `inputs[1..]` are draft proposals. `target_next[i]` is the target selection for
  row `i`. Lengths must be equal, `1..=8` (`:24-26`), `MAX_DSPARK_PROPOSALS = 7`
  (`:2`).
- The match test is `inputs.get(i+1) == Some(&token)` (`:39-40`): "the target's
  selection for row `i` equals the draft token carried into row `i+1`".
- `target_next[i]` is **always** appended to `emitted` (`:41`) — a mismatch emits
  the target's own selection (the correction), a match emits a draft token that
  equals the target selection, and an all-match round emits the final bonus token
  that has not itself been evaluated as an input (`GreedyVerification` doc,
  `:6-8`).
- `accepted_inputs` starts at 1 (the anchor) and increments per match (`:33-34`,
  `:42-44`); i.e. it is the number of input rows safe to publish to KV.
- The loop stops at the first mismatch, at `token == eos` (token `1`), or at the
  output-length limit (`:45-49`). Length-limit and EOS flags are reported
  (`:45-46`).
- Validation: non-empty, ≤8, equal lengths, `remaining > 0`, `inputs[0] != eos`,
  and every id `< 129280` (`:24-32`).

### 3.2 Acceptance probability equals `p(draft token)`

For a stochastic request the "target argmax" is replaced by the **target sample**
per row: unconstrained at `scheduler.rs:556-561` (`sample_target_rows`), constrained
at `constraints.rs:80-102` (`select_verification_sampled`). Because the emitted
token in every row is the target's own draw, the effective rule is
"accept while the draft equals this row's target sample, else emit the target
sample". With the draft treated as a point mass (`q = δ_draft`) the acceptance
probability is `p(draft token)`; this is the release's documented exact
rejection-correction scheme (`docs/release-v11-performance.md:228-249`,
`docs/release-v11-notes.md:35-39`). The verifier itself is unchanged greedy code
— the exactness comes from *what is passed as `target_next`*, not from the
verifier. Comment at `scheduler.rs:543-547`.

`min_p`/`top_k`/`top_p` are honored under speculation because the full chain runs
on the verified row before the verifier sees it (release claim
`docs/release-v11-performance.md:245-247`; code path `scores.rs:104-124`).

### 3.3 Rejection resampling

There is **no separate rejection-resampling draw**. The correction token is the
target draw already computed for the mismatching row, and it is used as-is
(`dspark_verify.rs:41`, `:47-49`); the next round's leftmost row is that token
(`scheduler.rs:597` frontier; `inputs[0]` on the next round,
`scheduler.rs:432-433`). Consequently the draw count per emitted token stays
one, and rejected rows must not consume a draw (see §2.4).

### 3.4 Inputs the verifier needs, and where they come from

- The batch carries, per request, `input.len()` target rows laid out
  contiguously in member order (`prepare_decode_lane`, `scheduler.rs:409-416`;
  `offset` accumulation at `scheduler.rs:534`, `:604`).
- Row `i` is the target distribution at the state *after* `inputs[0..=i]` — i.e.
  the causal verification row for that prefix — produced by one target pass over
  the speculative batch (`execute_logits`, `scheduler.rs:389-407`).
- For a constrained request the rows must additionally carry the grammar mask of
  the **hypothetical** prefix, per row (below).
- `remaining = max_tokens - generated` (`scheduler.rs:563`) and `eos = 1`
  (`scheduler.rs:562-563`; token `1` is stop, `scheduler.rs:87`, `:93`).

### 3.5 Rows per step

- `MAX_DSPARK_PROPOSALS = 7` → at most 8 rows per request per round
  (`dspark_verify.rs:2`, `:24`).
- Scheduler lane cap: 8 requests (`scheduler.rs:424`, `independent.rs:56`).
- Target head capacity `1..=80` rows (`rust/crates/ds41rt-daemon/src/v41_target_head.rs:124-130`);
  scheduler reserves `(max_tokens - generated).min(draft.max_verify_rows())`
  (`scheduler.rs:355`, `independent.rs:63-67`), with `draft_limit = 5` and
  `max_verify_rows() = draft_limit + 1` (`speculative.rs:76`, `:111`).
- Draft width 5 in production (`speculative.rs:76`), so a round is normally
  ≤6 rows per request, ≤48 rows per lane.

### 3.6 Must be preserved exactly vs may differ

**Must be preserved exactly**

1. `inputs[0]` is the emitted anchor; `target_next[i]` corresponds to the state
   after `inputs[0..=i]`; equality test is `inputs[i+1] == target_next[i]`.
2. Every row's target selection is emitted, including the mismatch row.
3. `accepted_inputs = 1 + (number of leading matches)`; rows beyond the first
   mismatch are never published to KV.
4. Stop at the first mismatch, at EOS, and at the length limit; report
   `eos` / `length_limit`.
5. The target selection that feed rows is produced by the **same** filter chain
   as the non-speculative path (mask first, temperature, `min_p`, `top_k`,
   `top_p`, then draw) at the same absolute positions.
6. A rejected row consumes no draw and does not advance the position stream.
7. Constrained rows use the hypothetical-prefix mask, not the committed-state
   mask.

**May legitimately differ (with documentation)**

- The *algorithm* that produces each row's selection (device radix sort vs
  comparison sort, different tie walk) provided the resulting token is identical
  for distinct probabilities and deterministic for exact ties.
- The target draw values, if and only if the seeded-output mapping changes and is
  documented (see §7).
- Diagnostic side outputs (probabilities, nucleus counts, top-two traces) — they
  are not consumed by the verifier (`scheduler.rs:565-576` is trace-gated;
  `scores.rs:127-142` is diagnostic-only).
- The number of rows executed per wave, the launch geometry, the split between
  greedy and stochastic lanes (see §5), and the draft-side confidence/adaptive
  policy, which were unchanged in v11 (`docs/release-v11-notes.md:38-39`).
- Bitwise parity between speculative and non-speculative token streams is
  **explicitly not guaranteed** (`docs/release-v11-notes.md:85-86`,
  `docs/release-v11-performance.md:249`).

---

## 4. GPU sampler interface

### 4.1 What it must accept

| Input | Shape / layout | Source of truth |
| --- | --- | --- |
| Logits rows | `f32`, `vocab` per row (129,280), `rows ∈ 1..=80`, contiguous, already final-norm + vocabulary-projected | `scores.rs:7-17`, `v41_target_head.rs:124-130`, `:387-404` |
| Vocabulary width | discovered from the row / buffer extent, not hard-coded | `target_sampling.rs:27-28`; `scores.rs:7` is the checkpoint value |
| Per-row parameters | `temperature`, `top_p`, `top_k` (`None` = disabled), `min_p`, `seed` (u64), **per row**, not per launch | `TargetSamplingParams` `:79-134`; per-request at `scheduler.rs:541` |
| Greedy flag | per row, `temperature < 1e-5 \|\| top_k == Some(1)` | `:172-174` |
| Constraint masks | packed `u32`, `words_per_row = ceil(vocab/32)`, row-major, bit `t%32` of word `t/32` = token `t`; absent row / `needs_mask=false` = unconstrained | `scores.rs:154-169`, `constraints.rs:44-46`, `:61-102` |
| Draw key | `(seed, absolute emitted position)`, per row — see §4.2 | `:187-199`, `scheduler.rs:508-518` |
| EOS / length budget | `1` / `max_tokens - generated` (verifier inputs) | `scheduler.rs:562-563` |

Validation the interface must enforce (mirroring `TargetSamplingParams::new`
`:108-134` and the FFI validators `rust/crates/ds41rt-ffi/src/lib.rs:17859-17896`):
`temperature` finite in `[0, 2]`; `top_p` finite in `(0, 1]`; `top_k != Some(0)`;
`min_p` finite in `[0, 1]`; rows > 0; vocab ≤ `u32::MAX`; mask extent exactly
`rows * ceil(vocab/32)`.

### 4.2 Draw generation — two acceptable options

- **Option A (preserves v11 seeded output exactly).** Compute
  `uniform[r] = splitmix64(seed_r, position_r)` on the host (or identically on
  device) and pass `rows` `f32` uniforms. This is exactly what the existing GPU
  `*_f32_async` samplers already accept (`rust/crates/ds41rt-ffi/src/lib.rs:14567-14586`,
  `Ds41rtDeviceBuffer random_uniforms`), and it makes batch independence trivial
  because each row carries its own draw.
- **Option B (device RNG).** Upload `(seed_r, position_r)` `u64` pairs (they are
  fixed for the round) and derive the uniform on device. This **is** a
  seeded-output mapping change unless the device reproduces §2.1 bit-for-bit
  (including the `>> 40` truncation to 24 bits, `target_sampling.rs:197-198`).
  Requires the §7 documentation.

Either way the kernel must clamp the uniform to `[0.0, 0.999_999_94]` before the
CDF comparison (`:51-52`, `:459`) and reject non-finite draws in stochastic mode
(`:237-241`).

### 4.3 What it must return

**Required:** one `u32` token id per row (`out_indices`), the only value any
production caller reads (`scores.rs:48-51`, `:120-123`, `scheduler.rs:556-561`,
`constraints.rs:94-99`).

**Optional / diagnostic only:** a per-row probability or normalized score
(`out_scores` in the legacy kernel, `native/cuda/kernels/sampling.cu:684`), a
nucleus count, and the top-two diagnostic (`scores.rs:127-142`). `.total` and
`.nucleus_count` are `dead_code` outside tests (`target_sampling.rs:403-410`).
A GPU implementation only needs them if an equivalence oracle or a trace target
consumes them; do not make them part of the wire contract.

**Also required by the surrounding machinery, though not by the sampler
itself:** the compact greedy `(id, score)` pair for the retained-frontier
consistency check (`scores.rs:61-66`, `v41_target_head.rs:378-386`), and a
*retained full row* only when a finishing/cacheable request must snapshot a
frontier (`scores.rs:80-90`, `scheduler.rs:597-603`).

### 4.4 What must NOT be downloaded to the host

1. **Full-vocabulary logits for a normal decode round.** The stochastic path
   currently violates this: `execute_logits` downloads the whole logit batch
   whenever `compact == false` (`scheduler.rs:397-404`), and `compact` is a
   lane-wide conjunction (`scheduler.rs:458-462`, `independent.rs:131-135`).
   Regardless of how the GPU sampler is wired, the hot path must not copy
   `rows × 129,280 × 4 B` (**517,120 B ≈ 0.52 MB per row**; ≈ 3.1 MB for a 6-row
   request, ≈ 24.8 MB for 8 requests × 6 rows) per round.
2. **Greedy rows.** They must stay on the device argmax path
   (`v41_target_head.rs:350-386`, `execute_greedy`/`execute_shared_greedy`
   `rust/crates/ds41rt-daemon/src/v41_target_pass.rs:177-189`), which copies only
   `2 × rows × 4 B`.
3. **Logits for rows whose result is discarded.** Rows past the first mismatch
   are never published (§3.6.3); only the finishing frontier may be copied
   (`scores.rs:143-151`).
4. The **draft RNG reservation staging** and adaptive-gate weights stay device
   side; only tokens/confidence trace cross (`speculative.rs:507-513`, `:596-620`).

### 4.5 Per-row parameters in one launch

The legacy kernel takes **one** `temperature`/`top_k`/`top_p` for the whole
launch and requires `top_k ∈ 1..=min(vocab, 64)`
(`native/cuda/kernels/sampling.cu:598-606`, `rust/crates/ds41rt-ffi/src/lib.rs:74`,
`:17880-17883`; `validate_lm_head_sampler_options`
`.../coordinator_kernels/sampling.rs:916-948`). That interface cannot express the
served contract, where two requests in one lane may have different parameters
(`scheduler.rs:541`). The GPU sampler must take per-row parameter arrays (or
accept only rows sharing parameters and be launched per homogeneous group).

### 4.6 Semantics the existing GPU kernel does *not* meet (do not reuse as-is)

- No `min_p` at all (`native/cuda/kernels/sampling.cu:598-685`).
- `top_k` is mandatory and capped at 64 (`kMaxSampleTopK`), whereas the served
  contract allows `None` and `k >= vocab` (`target_sampling.rs:477-501`).
- Ranks raw logits, then scales the chosen top-k (`sampling.cu:623-641`), whereas
  the CPU reference scales first and filters on the scaled values
  (`target_sampling.rs:427-458`). Ranking is scale-invariant, but the retained
  *values* and the `min_p` comparison are not, so an equivalence check that
  compares adjusted-logit bits would fail.
- `out_scores` divides by `nucleus_mass` (`sampling.cu:684`) — a different
  diagnostic scale from `RankedSample.total`.
- No domain separation and no `(seed, position)` key (§2.3).

---

## 5. Batch contract

1. **Per-request parameters inside a batched execution.** Every row must carry
   the parameters of the request it belongs to. A device launch must not take
   one shared `(temperature, top_p, top_k, min_p)`; the current code resolves
   parameters per request (`scheduler.rs:541`) and the API resolves them per
   HTTP request (`native_v41.rs:149-188`).
2. **Mixed greedy / stochastic / constrained batches.** A lane is compact only
   when *all* its members are greedy *and* unconstrained
   (`scheduler.rs:458-462`, `independent.rs:131-135`). This is a performance
   contract, not a correctness one, and it must **not regress greedy
   throughput**: the invariant is that a greedy row's work stays on the device
   argmax path and is unaffected by unrelated stochastic rows in the same batch
   (`v41_target_pass.rs:177-189`; the copy of only `id`+`score` at
   `v41_target_head.rs:366-369`). A GPU sampler that forces a whole-batch full
   download, or that reprocesses greedy rows through the filter chain, breaks it.
   Mixed batches should be split so greedy members use the compact path and only
   stochastic members go through the GPU sampler.
3. **Draw independence.** A row's draw depends only on its own request's
   `(seed, position)`. It must not depend on batch membership, lane ordinal,
   member order, or the number of rows in the wave (test
   `uniform_depends_on_position_not_on_batch_layout`, `target_sampling.rs:998-1007`;
   scheduler test `scheduler.rs:666-691`).
4. **No cross-row state.** Because draws are position-keyed, the sampler must be
   stateless per call; a single batch-wide RNG stream advanced per row would
   violate §2.4 for every request whose earlier rounds rejected differently.
5. **Constrained members** add a per-row mask dimension; a batch mixing
   constrained and unconstrained rows must express "no mask" per row. In the
   legacy `real_full` representation this is done by filling all-ones words for
   rows where `needs_mask` is false (`.../real_full/constraint.rs:237-245`); in
   the native path an absent mask is `None` (`scores.rs:30-34`).

---

## 6. Constrained-decoding contract

### 6.1 Mask semantics

- Representation: packed `u32`, little-endian bit order, `words_per_row =
  vocab.div_ceil(32)`, token `t` allowed iff `words[t/32] & (1 << (t%32)) != 0`
  (`scores.rs:154-169`, `target_sampling.rs:231-233`). Vocabulary: 129,280
  (`scores.rs:7`).
- Width: `mask.len()` must equal `vocab.div_ceil(32)`; otherwise `MaskWidth`
  (`target_sampling.rs:222-230`) / `"invalid grammar mask width"`
  (`scores.rs:154-157`).
- `needs_mask == false` from `fill_bitmask` means unconstrained, and the native
  path passes `None` (`constraints.rs:44-46`, `rust/crates/ds41rt-ffi/src/lib.rs:3358-3379`).
  The legacy path instead fills all-ones and masks off the vocabulary remainder
  (`.../real_full/constraint.rs:240-245`). **Ambiguity:** the native path never
  masks the unused high bits of the last word. This is harmless only because
  `allowed()` is evaluated per real token id; a GPU kernel that reads whole
  words must either ignore bits ≥ vocab or receive a remainder-masked row. The
  check that resolves it: a unit test on the GPU mask reader with the last word
  set to `u32::MAX` and `vocab % 32 != 0` (129,280 % 32 == 0 for the official
  checkpoint, so this needs a synthetic width).
- Mask is applied **first**, before temperature and every filter
  (`target_sampling.rs:9-11`, test `mask_is_applied_before_sampling_filters`
  `:938-952`), and masked logits are never read, so a masked NaN is legal
  (test `:1009-1024`).

### 6.2 Per-position versus per-prefix mask preparation

- **Per-position (next token).** The authoritative matcher has accepted every
  emitted token (`State::accept`, `constraints.rs:47-50`, called from
  `emit_one`, `scheduler.rs:79`); `State::mask()` calls `fill_bitmask` once
  (`constraints.rs:44-46`) and the first token of a request is sampled with that
  mask (`scheduler.rs:322-324`).
- **Per-prefix rows (verification).** `select_verification_sampled`
  (`constraints.rs:80-102`) forks the authoritative matcher, then for each row
  `index`: row 0 uses the mask *before* accepting any draft token, and later rows
  first accept `input[index]` into the fork (`:92`), then fill the mask
  (`:93`). This yields the grammar state that would hold if the preceding drafts
  had been emitted — the mask for the hypothetical prefix. `needs_mask=false`
  rows pass `None` to the sampler (`:96-97`). The greedy twin is
  `select_verification` (`:61-73`); the legacy equivalent is `masks_for_draft`
  (`.../real_full/constraint.rs:219-272`), which additionally asserts that the
  draft token was allowed by the row mask and that the matcher accepts it
  (`:246-264`).

### 6.3 Fork / commit / rollback

- `fork()` clones the matcher (`rust/crates/ds41rt-ffi/src/lib.rs:3344-3356`).
- The fork is scratch only: it advances through *hypothetical* draft tokens and
  is dropped (`constraints.rs:62-72`, `:88-101`). The authoritative
  `State.matcher` is never mutated by verification.
- Proposals are pre-trimmed against a fork as well: `truncate_proposal`
  (`constraints.rs:51-60`) truncates the proposal at the first illegal token and
  at grammar completion (keeping the completing token). Called before the target
  pass in both schedulers (`scheduler.rs:434-440`, `independent.rs:89-95`).
- Commit advances the authoritative matcher through the *emitted* tokens only:
  `active.emit_one` → `constraint.accept(token)` (`scheduler.rs:77-82`), and a
  rejected token is a hard error `"emitted token violates request grammar"`
  (`constraints.rs:47-50`). An illegal draft is rejected by
  `"illegal verification draft token"` (`constraints.rs:68`, `:92`).
- `is_completed` is checked in the proposal path so nothing is accepted after the
  grammar completes (`constraints.rs:54`, `:57`; legacy `:174-187`, `:290-295`).
- **Rollback contract:** because the fork is scratch and the authoritative state
  only advances on emit, a round that emits fewer tokens than it verified
  requires no explicit rollback — the grammar simply never saw the rejected rows.
  A GPU-side grammar state machine, if introduced, must preserve this: the
  committed-state matcher advances exactly `accepted_inputs` positions, and the
  remaining speculative rows' effects are discarded.

### 6.4 Empty-mask error path

- A grammar allowing no token yields an all-zero (or all-unmasked-bit-clear) mask
  and `EmptyCandidates` (`target_sampling.rs:264`, `:440-444`), never a silent
  fallback to the unmasked distribution (module contract `:23-25`), in every mode
  including the fast categorical path (test `empty_mask_is_reported_as_empty_candidates_in_every_mode`
  `:1026-1048`).
- The batch/compact path reports its own error before that:
  `"grammar allows no target token"` from the CPU argmax (`scores.rs:168`) or
  `"grammar requires full logits"` when a compact (greedy-only) batch is asked
  for a masked selection (`scores.rs:95-98`).
- `fill_bitmask` itself guarantees a non-empty buffer (`rust/crates/ds41rt-ffi/src/lib.rs:3359`),
  so "empty" means an all-zero mask, not a zero-length one.
- Propagation: `TargetSamplingError` is a library error; `is_bad_request`
  classifies `MaskWidth`/`InvalidParameter` as caller faults
  (`target_sampling.rs:68-76`) but is **not** wired into any daemon HTTP-status
  mapping in this checkout (grep shows no consumer outside the module), so a
  grammar that admits nothing surfaces as a worker error (HTTP 500) rather than
  a 400 today. **Ambiguity / check:** grep for `is_bad_request` across
  `rust/crates` returns only its definition — confirm whether a GPU path should
  route `EmptyCandidates`/`MaskWidth` to `NativeFailure::BadRequest`, which would
  change observable status codes.

---

## 7. MUST NOT CHANGE vs MAY CHANGE WITH DOCUMENTATION

### 7.1 MUST NOT CHANGE

1. Filter order and mask-first placement: mask → temperature → `min_p` → `top_k`
   → `top_p` → draw (`target_sampling.rs:9-25`, `:430-465`).
2. Greedy epsilon semantics: `temperature < 1e-5` or `top_k == Some(1)` is exact
   argmax and consumes no draw; `top_k == Some(1)` is *still* greedy even at a
   sampling temperature (`:172-174`).
3. Greedy argmax tie-break (first/lowest id wins) and the greedy device path
   staying compact (`:259-262`, `scores.rs:44-46`, `v41_target_head.rs:349-386`).
4. Served defaults: no `temperature` → greedy; unset filters disabled
   (`native_v41.rs:145-159`) — pinned by API tests
   (`upstream_native_v41.rs:138-163`).
5. `top_k` disable spellings (`absent`, `null`, `0`, `-1`) and `k >= vocab` as a
   no-op (`native_v41.rs:162-176`, `target_sampling.rs:499-501`).
6. `min_p` in logit space, relative to the scaled max, inclusive `>=`,
   `min_p = 0` disabled (`:449-458`).
7. `top_p` inclusive `>=` nucleus prefix, at least one token, clamp `1e-6..=1.0`,
   `top_p = 1.0` disabled (`:544-554`).
8. The filter chain can never empty the candidate set; the best token survives
   (`:23-25`, test `:954-973`); an all-masked row is a hard error, never a
   fallback (`:1026-1048`).
9. Draw keying on `(request seed, absolute emitted-token position)` and batch
   independence; rejected drafts consume no draw
   (`:35-38`, `:187-199`, `:998-1007`, `:1100-1208`).
10. `seed_from_i64` two's-complement mapping (`-1` deterministic), `seed = 0` a
    real seed, unset seed generated per request (`:161-167`, `:727-735`,
    `native_v41.rs:129-137`).
11. Target/draft RNG domain separation (`:182-186`).
12. Verifier semantics of §3.1–3.4: sample-and-match, emit-every-row, accepted
    prefix, stop rules, hypothetical-prefix masks (`dspark_verify.rs:18-52`,
    `scheduler.rs:539-604`, `constraints.rs:61-102`).
13. Mask bit layout, word count, mask-first application, and the hard error on an
    empty allowed set (`scores.rs:154-169`, `constraints.rs:44-46`).
14. NaN/Inf discipline: reject non-finite allowed logits; report a non-finite
    scaled maximum as an invalid temperature; never emit NaN or panic
    (`:256-258`, `:434-448`).
15. Greedy throughput: the compact path must remain device-side and must not be
    forced to download full logits because of a stochastic peer
    (`scheduler.rs:397-407`, `:458-462`; §5.2).

### 7.2 MAY CHANGE WITH DOCUMENTATION

1. **The draw's numeric value / the seeded-output mapping.** Replacing SplitMix64
   with a device RNG, or changing the domain constant, the mix, or the 24-bit
   truncation, changes every seeded token stream for `(seed, position)`. This is
   permissible only as a documented mapping change (§7.3).
2. The uniformization strategy (host-precomputed uniforms vs device RNG vs
   `(seed, position)` upload), provided §7.1.9 holds.
3. The ordered-path algorithm: device radix/heap/bitonic ordering instead of the
   comparison sort, provided the ranked order for distinct scaled values matches
   and equal values are ordered deterministically by ascending id if bit-parity
   with the CPU is desired (`target_sampling.rs:277-401`; the CPU itself already
   has three branches, `:463-522`).
4. The top-k boundary tie policy: keeping every token equal to the k-th value
   (vLLM behavior) instead of exactly k. This is currently a documented deviation
   (`:30-33`); changing it is a semantic change and must be documented as such.
5. Diagnostic outputs (`total`, `nucleus_count`, per-row probability,
   top-two trace). None are read in production (`:403-410`, `scheduler.rs:565-576`,
   `scores.rs:127-142`); if a GPU kernel publishes them, their scale/definition
   must match the CPU if any oracle compares them.
6. Launch geometry, per-row vectorization, chunking, and kernel fusion — subject
   to §7.1.14 (determinism is per `(seed, position)`, not per layout) and the
   requirement that each row carries its own parameters (§5.1).
7. Splitting lanes into homogeneous greedy/stochastic/constrained groups, and
   the batch-size caps (currently 8 requests/lane, `scheduler.rs:424`,
   `independent.rs:56`; target-head 80 rows, `v41_target_head.rs:124-130`).
8. Retained-frontier download strategy (which rows, when), provided a cacheable
   finishing request still retains a full row so a later grammar can re-select
   (`scores.rs:78-90`, `scheduler.rs:597-603`).

### 7.3 Reproducibility versus bit-identity, and what the release notes must state

These are **three distinct claims**; the release must not conflate them.

- **(a) Bit-identity against the current CPU sampler.** With the same logits, the
  same parameters, the same mask and the same absolute position, the GPU
  sampler returns the same token id *and* (if compared) the same
  `total`/`nucleus_count` bits as `target_sampling.rs`. This is the only claim
  that allows "no semantic change" wording; the existing GPU kernels do **not**
  satisfy it (§4.6), so a new kernel must be validated against the in-tree
  oracle (`reference_select`, `target_sampling.rs:1235-1399`;
  `compare_to_reference`, `:1401-1450`) plus the boundary sweeps
  (`:1527-1571`, `:1573-1603`, `:1705-1743`).
- **(b) Deterministic replay within the GPU implementation.** Same seed, same
  position, same logits, same parameters ⇒ same token, every run, independent of
  batch composition, peer requests, lane, or warmup. This is weaker than (a) but
  is the invariant the scheduler and the seed contract actually rely on
  (`:35-38`, `:998-1007`, `:1100-1208`). It must hold even if the device RNG
  replaces SplitMix64.
- **(c) Bitwise parity between speculative and non-speculative execution.** Not
  claimed today and must not be claimed (`docs/release-v11-notes.md:85-86`,
  `docs/release-v11-performance.md:249`).

**If the seeded-output mapping changes** (Option B in §4.2, or any change to the
mix/domain/truncation), the release documentation must state, explicitly:

1. That seeded outputs are **not** bit-identical to v11 CPU-sampler seed
   `(seed, position)` streams, and that the mapping is version-scoped.
2. The new mapping's exact definition (domain constant, RNG algorithm, integer
   range, truncation, clamping) and the version/commit at which it took effect,
   so a replay can be reproduced.
3. That unseeded requests (absent seed → time-based seed,
   `native_v41.rs:129-137`) remain non-reproducible by design.
4. That query-level reproducibility holds only for *the same logits and the same
   execution path within the same build* — the wording already used at
   `docs/release-v11-performance.md:249` and `docs/release-v11-notes.md:39-41`.
5. That greedy requests are unaffected (greedy consumes no draw).
6. Whether the change is per-release global or gated, and how a caller can obtain
   the old stream (there is no compatibility knob today).
7. The qualification evidence for the new path: bit-identity against the in-tree
   oracle for the *filter chain*, plus seeded replay equality against the new
   documented mapping, plus the boundary/`-inf`/tie cases already covered
   (`target_sampling.rs:1526-1763`).

---

## 8. Open ambiguities and the checks that would resolve them

1. **`MAX_UNIFORM` clamp in the ordered path.** The clamped local is used in
   `sample_allowed_internals` (`:459`) but the oracle's ordered branch uses the
   raw `uniform` (`:1383`) after clamping in its fast path only (`:1307`).
   *Check:* add an oracle case with `uniform = 1.0` and a `top_p < 1.0` nucleus
   whose final cumulative boundary is exactly in `(MAX_UNIFORM, 1.0]`.
2. **Last-word remainder bits on a masked GPU read.** §6.1. *Check:* synthetic
   vocab with `vocab % 32 != 0` and a `u32::MAX` last word.
3. **`EmptyCandidates` / `MaskWidth` HTTP status.** §6.4. *Check:* grep for
   `is_bad_request` consumers (none found) and decide the GPU-path mapping.
4. **Who owns `(seed, position)` for the finishing/bonus token.** Position is the
   emitted index at round start, and the bonus token shares the round's
   `base_position + index` slot; because the bonus token is emitted *before* the
   next round's leftmost row (which is that bonus token,
   `scheduler.rs:597`), the mapping is consistent — but there is no test that
   asserts the bonus-token case specifically under sampling. *Check:* extend
   `speculative_sample_match_equals_sequential_sampling` (or add a target-sampling
   test) asserting the bonus token equals a sequential draw at that same absolute
   position.
5. **Draft temperature is `0.0` in native serve** (`speculative.rs:511-512`,
   `terminal.rs:169-172`) while the draft RNG is reserved per width. *Check:*
   confirm the production draft is greedy-only and that the reservation
   (`SUBSEQUENCES_PER_DRAFT`) already accounts for it; no sampling-side change
   needed, but a GPU draft sampler would have to keep the reservation semantics
   (cancellation must not recycle draws, `dspark_rng.rs:64-74`).
6. **Vocabulary constant.** `scores.rs:7` hard-codes 129,280 while
   `target_sampling.rs:27-28` discovers width from the row. *Check:* keep the GPU
   kernel width-driven; do not bake 129,280 into the sampler interface.

---

## 9. Reference file map

| Concern | File |
| --- | --- |
| CPU sampler, oracle, boundary tests | `rust/crates/ds41rt-core/src/target_sampling.rs` |
| Verification rule | `rust/crates/ds41rt-core/src/dspark_verify.rs` |
| Draft RNG reservations | `rust/crates/ds41rt-core/src/dspark_rng.rs` |
| Host logit materialisation / compact rows / retained frontier | `rust/crates/ds41rt-daemon/src/v41_native_serve/scores.rs` |
| Row sampling, commit decision, batching | `rust/crates/ds41rt-daemon/src/v41_native_serve/scheduler.rs` |
| Independent-lane path (same contract) | `rust/crates/ds41rt-daemon/src/v41_native_serve/scheduler/independent.rs` |
| Grammar masks, per-prefix rows | `rust/crates/ds41rt-daemon/src/v41_native_serve/constraints.rs` |
| Greedy device path, head capacity | `rust/crates/ds41rt-daemon/src/v41_target_head.rs` |
| Request/parameter resolution, status mapping | `rust/crates/ds41rt-api/src/native_v41.rs` |
| API sampling pins | `rust/crates/ds41rt-api/src/tests/upstream_native_v41.rs`, `.../tests/upstream_sampler.rs` |
| Legacy GPU sampler + device argmax | `native/cuda/kernels/sampling.cu`, `rust/crates/ds41rt-ffi/src/lib.rs` |
| Legacy `real_full` sampling plumbing | `rust/crates/ds41rt-daemon/src/commands/real_full/{target_sampling.rs,sampling/lm_head*.rs,constraint.rs}`, `rust/crates/ds41rt-api/src/real_full.rs` |
| Release claims | `docs/release-v11-notes.md`, `docs/release-v11-performance.md`, `docs/research/vllm-sampling.md` |

---

## Corrections applied (adversarial review)

Applied against repo revision `e1f5d49` on 2026-09-22. No measured number, table
value or evidence hash in this contract was changed; the only edit is a
mislabeled unit:

- **Item 4.** In §4.4 the logits-row size was corrected from "≈ 3.1 MB/row" to
  **517,120 B ≈ 0.52 MB per row**, with ≈ 3.1 MB identified as a 6-row request
  (the ≈ 24.8 MB batch figure is unchanged). Buffer-sizing arithmetic is now
  unambiguous.

