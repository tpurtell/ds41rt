# TP×EP greedy-quality adjudication (supplemental protocol)

Status: **CPU-only design and evidence review. No GPU/remote action, no production
edit, no runner change.** The frozen runner and its `passed=false` results stand;
nothing here replaces or weakens the existing gate.

Scope: adjudicate the deterministic-greedy failures in the two completed matched
runs — G3 `tp2ep2` (`g3-full-candidate.json`) and TP4 `tp4ep1`
(`tp4-full-candidate.json`) — and propose one defensible **supplemental**
evaluation protocol for TP2-vs-baseline quality that does not require bit-exact
text.

Shared controls (both arms): corpus `823f203c…`, runner `e197146a…`, N20,
`--kv-pool-size 5037542400`, temperature 0 / greedy, same diag-v2 binaries,
4 Sparks, no config/default change.

## 1. Consolidated raw facts

372 rows per arm, zero runtime/SSE errors. `index` is the concurrent slot; each
case is a single corpus prompt, so rows at different `C`/`index` are the same
prompt under different concurrency, not distinct prompts.

| Arm | case | rows | objective pass | finish | tokens min/med/max | C1 groups (inconsistent) | C2+ groups (inconsistent) | indices varying across C |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: |
| G3 TP2EP2 | counting-1-64 | 93 | 93 | stop 93 | 191/191/191 | 1 (0) | 30 (0) | 0 |
| G3 TP2EP2 | code-merge-intervals | 93 | 93 | stop 93 | 239/290/467 | 1 (1) | 30 (27) | 15 |
| G3 TP2EP2 | topic-virtual-memory | 93 | 93 | stop 23, length 70 | 266/384/384 | 1 (1) | 30 (30) | 16 |
| G3 TP2EP2 | mixed-code-fix | 93 | 93 | stop 93 | 79/84/92 | 1 (1) | 30 (24) | 15 |
| TP4 TP4EP1 | counting-1-64 | 93 | 93 | stop 93 | 191/191/191 | 1 (0) | 30 (0) | 0 |
| TP4 TP4EP1 | code-merge-intervals | 93 | 92 | stop 92, length 1 | 239/263/512 | 1 (0) | 30 (9) | 10 |
| TP4 TP4EP1 | topic-virtual-memory | 93 | 93 | stop 12, length 81 | 358/384/384 | 1 (0) | 30 (27) | 15 |
| TP4 TP4EP1 | mixed-code-fix | 93 | 93 | stop 93 | 80/84/93 | 1 (0) | 30 (5) | 7 |

Readings:

1. **`counting-1-64` is deterministic in both arms** (0 inconsistent groups at
   C1 and C2+, identical 191 tokens, no cross-C variation). The runner and this
   analysis detect determinism correctly; counting is a valid control.
2. **Repeat drift is concentrated at C2+ in both arms.** Baseline variation
   therefore exists in both arms, but the candidate's rates are markedly
   elevated (code 27/30 vs 9/30; mixed 24/30 vs 5/30). That excess is
   **unadjudicated and requires disclosure, not exoneration**; neither arm is
   fully quality-qualified.
3. **C1 differs by topology**: G3 TP2 drifts at C1 for code/topic/mixed; TP4 C1
   is exact in this run. That matches the native numeric asymmetry below.
4. **Topic is length-bound** (70/93 and 81/93 rows hit `max_tokens`), so topic
   text drift is partly truncation, not content divergence.
5. **Every non-truncated row passes the frozen objective checks** in both arms.
   The only objective failure is one TP4 code row.

### The single TP4 content failure is truncation, not a code error

`code-merge-intervals`, C=8, repeat 1, index 5: `completion_tokens=512`,
`finish_reason=length`; the text is one opening ```` ```python ```` fence with
**no closing fence** (1 fence total) and is cut mid-`assert`
(`assert merge_intervals([(1, 4), (4, 5), (`). The checker condition
(`python_structure`: "response is not exactly one Python code block") fails
because the block is unclosed. The visible implementation is well-formed up to
the cut. **Classification: budget truncation / format, not a semantic code
error** (checker data and text inspection only; nothing executed).

## 2. What each criterion actually measures

- **Within-`(C,index)` repeats**: same prompt, same slot, but the daemon derives
  the scheduler tie seed from a per-encode monotonic `request_id`, so repeats are
  *not* the same scheduling identity. This gate conflates request-id-derived
  ownership change with any other nondeterminism.
- **Cross-`C` differences**: same prompt at a different concurrency; also changes
  batch shape and the id sequence, so it is not an independent content signal.
- **C1 vs C2+**: C1 is single-stream (one request, simple batch); C2+ adds batch
  shape and concurrent id interleaving, which is where most drift appears.
- **Objective checks**: structural/AST only; they do not compare text across runs.
- **Finish reasons / length**: `length` marks budget-bounded output; it must be
  separated from semantic quality and it confounds TPS.

### Native numeric context (component-level, no E2E claim)

Component replay (`runs/tp-ep-native/tie-seed-replay/gpu/results.md`): same
request id ×3 is bit-identical per rank and final; a different request id changes
owners at TP2EP2 (61/64 ids in 1..64) and changes the net FFN hidden result by
rel_L2 0.0031246 / cosine 0.9999960 (all four rank planes), while TP4EP1 is
bit-identical. So the seed changes the per-rank pre-compaction sum grouping and
the BF16 partition rounding. This is consistent with deterministic-per-id but
different-across-id text; it does **not** prove the E2E text cause (FFN hidden is
not logits).

## 3. Supplemental protocol (does not replace the frozen gate)

**Name:** TP×EP greedy-stability and quality adjudication v1. It is reported
alongside the frozen result, which remains `passed=false` and is **not renamed**
as a pass.

This task's scope is performance plumbing and quality adjudication, **not**
proving universal baseline batch determinism.

- **S0 — C1 repeatability prerequisite.** One logical request, C=1, ≥3 repeats:
  the decoded **text/token stream** must be identical per case. This is an
  HTTP-observed text/token identity check, **not** a bit-exact logits claim.
  *Current build:* passes for TP4EP1, fails for TP2EP2.
- **S1 — Per-row objective quality.** 100% of rows with `finish_reason != length`
  must pass the frozen objective checks. Rows ending in `length` are reported as
  *budget-inconclusive* and excluded from quality (not hidden). Acceptance:
  zero objective failures among non-`length` rows in both arms.
- **S2 — Matched fixed-output performance.** TPS is comparable only on paired
  rows with identical `completion_tokens`, or on a dedicated harness that forces
  a fixed output length. Acceptance: no TPS claim from unmatched lengths; report
  paired deltas with per-C sample counts, otherwise mark indicative-only.
- **S3 — Content-equivalence review (supporting evidence).** For inconsistent
  code groups, classify the differing texts (automated AST/objective data plus a
  bounded manual sample) as formatting, equivalent-valid, or semantic error.
  Acceptance: zero semantic errors; topic measured above truncation.

### Adopted supplemental package (parent decision)

The following are adopted for reporting **alongside** the frozen result:

1. C1 repeatability as HTTP text/token identity (S0).
2. Zero runtime/SSE errors across all rows.
3. Frozen objective checks must pass on every non-truncated row (S1).
4. Truncated rows (`finish_reason=length`) are explicitly labelled
   budget-inconclusive and excluded from content-quality conclusions.
5. Performance comparisons only at matched fixed output (S2).

C2+ greedy variation is documented as **baseline variation present in both
arms**. The candidate's elevated rates (code 27/30 vs 9/30; mixed 24/30 vs 5/30)
are **unadjudicated excess requiring disclosure, not exoneration**, and neither
arm is fully quality-qualified. The original frozen verdict is preserved
verbatim; the supplemental package does not replace it and no supplemental
runner is implemented yet.

### Optional future work (not on the critical path)

- A ≥5-run drift-rate study with confidence intervals is **optional** and not
  required; it should not start without a justified margin and sample plan.
- Topic re-run with a larger `max_tokens` to separate truncation from content
  drift.
- Automated AST/objective classification of inconsistent code groups.

## 4. Seed-layer option (implemented by 7b; not deployed here)

The layer-seed option is **implemented and covered by 7b's tests**, not
conceptual. It is a **stable h/layer seed**, and it is **not** a request-identity
seed: request ids come from a new encode counter, so deriving the seed from
request identity would not fix the current drift. Removing one variation source
does not guarantee bit-exactness, and it is not the sole possible fix. Default
dispatch is unchanged and nothing is deployed from this diagnostic.

## 5. If the strict original gate is genuinely required

The frozen cross-repeat greedy gate requires deterministic identical text for the
same request. The component evidence shows the request-id-derived tie seed can
change a TP2EP2 FFN hidden result by ~0.003 rel_L2, so image-level bit-exactness
is **not established**, and the stable layer seed removes only one variation
source rather than guaranteeing it. An EP1-only qualification would satisfy
neither the requested EP configurations, so it is not an acceptable route.
Therefore the frozen `passed=false` stands, and the supplemental package is a
quality/regression assessment only.

## 6. Still required for acceptance

- The three six-rank E2E cases: **1-RTX TP3EP2**, **2-RTX TP2EP3**, **2-RTX
  TP3EP2** (all six-rank; the current TP2EP2 evidence is a four-Spark world-4
  run, so "TP2EP2-six" is not a valid case).
- Real memory and profiling gates.

## 7. Explicit non-actions

- No runner/threshold/gate change; frozen FAIL preserved verbatim and not renamed.
- No supplemental runner implementation in this pass.
- No production or frozen-harness edit; no GPU, no remote execution.
- No claim that drift is benign, TP2-specific, or caused by the seed in E2E.
