# dSpark verification length: online bandwidth-balance policy (plan)

2026-09-23. Proposal only; nothing implemented. Replaces every existing adaptive
dSpark length selector on the native serving path with one calibration-free
model. Source facts below were read from the current `dev` tree.

## 1. What the cycle looks like today (C1)

- One request uses one lane. Each verification cycle is a serial stack:
  draft (always `draft_limit` proposals, ~2.9 ms) → 40 layers → commit.
- Every remote layer is a blocking round trip: RTX attention/router → post to
  four ranks over RoCE → shared expert on RTX overlapped with Spark compute →
  RTX waits → H2D + reduce + mHC. Sparks are idle during every attention stage;
  the RTX is idle during most of the remote wait. `v41_backbone_lane.rs:302-397`.
- Wire size is irrelevant: ~5.4 KB/row out, ~10.2 KB/row back per rank.
- Spark cost per layer is unique-expert bandwidth: each expert's 4.70 MB TP4
  slice is read once per group of 16 rows (`v41_route_plan.py`,
  `w4a8_v41_slice.py`); ~22 µs per unique expert at saturation (~210 GB/s), but
  ~160 µs for a 1-row/6-expert call (poor occupancy). Coordinator-observed
  end-to-end: ~36 µs per unique expert + ~6.6 µs per row + a per-layer floor.
- Local RTX layers (5 at single RTX, 20 at TP2 dual) use the same kernel family
  with 18.8 MB (TP1) or 9.4 MB (TP2) per unique expert at ~1.5 TB/s.
- The Spark measures its own kernel time with CUDA events but never returns it.
  The response header has 12 spare zeroed bytes (`protocol_v2/response.rs`).
- Per-layer `Instant` timings in `execute_tp4` are computed unconditionally;
  only the `tracing::debug!` is gated. No timing is aggregated at runtime.
- Routes per layer per row are already on the host before dispatch
  (`request.request().routes`) and captured into `route_capture`.

Back-of-envelope with the currently fitted coefficients (`cost-profile.json`,
`phase1-current-c1-profile.md`), C1, 35 remote layers, ~5.2 tokens per 42.6 ms
cycle (0.12 tok/ms):

| New experts per added row (per layer) | Marginal cost of one more draft row | Break-even cumulative acceptance |
|---:|---:|---:|
| 1.5 | ~2.6 ms | ~0.31 |
| 3.0 | ~4.5 ms | ~0.54 |
| 6.0 | ~8.2 ms | never |

The sharing forecast is the decisive input; the per-layer latency floor
(~160 µs × 35 ≈ 5.5 ms per cycle of pure idle) and per-round RTX cost
(~11–12 ms) are the fixed costs that push toward longer batches.

## 2. Objective: fixed plus wasted resource per committed token

Work that is paid exactly once per committed token, whichever cycle verifies
it (its attention rows, its activation traffic through the experts, its wire
bytes, its share of the reduce), is not a cost of the length decision. A token
verified now was otherwise verified next cycle. What the decision controls is:

- fixed cost per cycle: per-round RTX path, the per-layer transport/queue
  floor, and the weight bytes streamed for the cycle (shared across the rows
  that touch the same experts);
- wasted cost: per-token work and newly touched expert bytes for rows that are
  rejected.

So the objective is to minimize expected (fixed + wasted) resource per
committed token, with

```
E[tokens](K) = 1 + Σ_{j≤K} Π_{i≤j} p_i        p_i = sigmoid(raw confidence_i)
```

This is the same argmax as maximizing `E[tokens](K) / T(K)` with the full
cycle time (subtracting the once-per-token cost from the ratio's denominator
only shifts it by a constant), so the throughput view and the waste view agree.
Consequences: a row with p = 1 is always included, because it adds no waste
and amortizes the fixed cost; the per-row coefficient matters only through the
rejected rows; the decisive quantity is the newly touched expert bytes per
added row against that row's cumulative acceptance probability.

## 3. Cost in resource units, fitted scale factors only

Price everything in bytes of weight traffic on the resource that is busy, and
fit one scale per resource class rather than per-arrangement coefficients.
Bytes per unique expert are already known exactly for the running layout:
`resident_bytes / 384` per backend class (native MXFP4, NVFP4, EXL3 per-layer
arenas, any TP width) is what the Spark's own debug record uses
(`execution.rs:791`; loader geometry in `v41_expert_staging.rs`, `nvfp4.rs`,
`exl3.rs`). Changing quant or TP therefore changes the known byte count, not a
fitted number. Only the original native layout needs to be checked as the base.

For a candidate lane shape with N rows and the forecast newly-touched expert
set per layer:

```
T(shape) = A + B·N
         + Σ_{l local}  bytes_loc · U_l / BW_rtx
         + Σ_{l remote} ( L + b·N + bytes_spark · U_l / BW_spark )
```

Fitted scalars per serving instance: `A, B` (per-round non-expert path
including draft, prepare, sampling, commit), `BW_rtx`, `BW_spark` (effective
bandwidth including kernel efficiency, clocks, thermal state), `L` (per remote
layer floor: transport, queueing, H2D/D2H, reduce, low-occupancy ramp), `b`
(per-row remote cost). Occupancy differences between quants (EXL3 trellis
decode, tiny batches) land in `BW` and `L`; if residual stats later show a
systematic ramp, replace the constant `BW` with `BW_max·U/(U+U_half)` fitted
the same way. A `max(0, rows_of_busiest_expert−16)` regroup term is only
needed if C>1 residuals show it; C1 never exceeds 8 rows.

Two online regressions, both recursive least squares with a forgetting factor
and robust (Huber or trimmed) weighting to reject graph-capture and
prefill-interrupted outliers:

- R1, one sample per remote layer per cycle:
  `t_remote,l = L + b·N + (1/BW_spark)·bytes_spark·U_l_observed`, where
  `t_remote,l` is dispatch start to collect end (the critical-path span).
  `1/BW_spark` is identified from within-cycle variation of U across layers;
  `L`, `b` from across-cycle variation in N.
- R2, one sample per cycle:
  `T_cycle − Σ_l t_remote,l = A + B·N + (1/BW_rtx)·bytes_loc·Σ_{l local} U_l`.

Effective window ~500 layer samples for R1 (~15 cycles) and ~50 cycles for R2,
so clock, power-cap and thermal drift are tracked within seconds. Cold start:
run the fixed policy (`draft_limit`) until both estimators have a minimum
sample count; readiness warmup already runs cycles, so users never see it.
Physical priors for the first samples: 0.75 × device peak bandwidth from
`cudaDeviceProp` on the RTX, and the published 273 GB/s for GB10.

Known interaction at C>1: lanes share one tokio thread, so a lane's measured
remote wait includes the peer lane's RTX work. Keep two coefficient sets keyed
by "peer lane active during this cycle". C1 always uses the solo set.

## 4. Expert-sharing forecast as sets, unions computed incrementally

Keep the last W (32) committed positions' routes per layer per request as
384-bit bitsets (40 × 32 × 48 B = 61 KB per request). The stand-in for draft
position k of request r at layer l is the union of that request's last k
committed positions' routes, `S_r,l(k)`. Adding draft row k to a lane whose
current expert set at layer l is `E_l` costs

```
Δbytes_l = bytes_class(l) · | S_r,l(k) \ E_l |
```

summed over layers. Unions across requests are then exact for the stand-in
sets, so cross-request sharing of popular experts comes out of the data rather
than from a fitted ratio; at C1 it reduces to the marginal unique count of one
more consecutive token. Adjacent-token correlation is captured directly, and
the work is a few thousand word ORs per candidate step.

Refinement if the exported forecast error warrants it: average `|S_r,l(k)|`
over several recent windows of length k instead of using the single most
recent one. Seed a fresh request from a global per-layer profile (EMA across
requests); optionally (phase 3) from the prompt tail's prefill routes, which
pass through the same staging. Replaces `DsparkRouteHistory` (8 accepted rows).

## 5. Selection: forward incremental growth

Start every request at anchor + 0 drafts (or Kmin). Repeatedly pick, across
all requests in the lane, the next draft row with the best marginal ratio

```
gain_r(k) / Δcost_r(k),   gain_r(k) = Π_{i≤k} p_r,i,
Δcost_r(k) = b + waste-weighted per-row term + Σ_l Δbytes_l / BW_class(l)
```

and add it while the marginal ratio beats the current average
`E[tokens] / T(shape)`. Because cumulative acceptance falls with k and newly
touched bytes per row do not fall faster, the ratio is unimodal per request,
so this reaches the optimum for C1 in at most Kmax steps and is a good greedy
for lanes with several requests. Keep the existing selector's trick of
continuing through a temporary loss and returning the best visited shape, to
cross a 16-row group boundary at C>1.

- Kmax = min(draft_limit, output budget, grammar truncation) as today.
- Drop the 2 % hysteresis; graphs are retained for all shapes through 64 rows,
  so shape changes are free. Keep a tie-break toward the current shape.
- Allow Kmin = 0 (verify only the anchor) when the 1-row path is qualified; the
  current "voluntarily keep one draft" rule only existed for kernel-shape
  qualification.
- `select_dspark_prefixes_bounded` (backward suffix removal) is then deleted
  along with the rest of `dspark_policy.rs`.

## 6. Confidence

Use sigmoid(raw) as the conditional acceptance probability (the phase 1 audit
fitted slope ~0.95–0.98, intercept ~0). Add per-position decayed counters
(predicted mass, observed accepts) and export the reliability ratio. Apply a
bounded logit bias per position only if a quant/config shows drift; decide from
the exported stats, not offline.

## 7. Ideas beyond the ask

1. Spark-reported timing in the spare header bytes (`kernel_us`, `service_us`
   as u32; zero = not reported, so old workers keep working). Separates `a`
   from `L` cleanly, shows effective GB/s and throttling per rank, and exposes
   the slowest rank. Small, backward-compatible.
2. Publish the fitted coefficients, implied effective bandwidth, chosen-K
   histogram and predicted-vs-actual cycle error on `/v1/stats`. This replaces
   the offline fit scripts as the way to check the model.
3. Prefill seeding of the sharing profile (above).
4. Future, not now: trailing-row trimming at layer boundaries when observed U
   exceeds the forecast (shape changes mid-cycle; needs graph work).
5. The per-layer floor `L` is the largest fixed cost the policy sees; its live
   estimate is a direct diagnostic for transport work, which is separate.

## 8. Deletions (native serving path)

- `ds41rt-core/src/dspark_routes.rs` (route history, forecast, reuse selector).
- `v41_native_serve/speculative/cost.rs`, `cost-profile.json`,
  `cost-profile-nvfp4.json`; `DS41RT_ADAPTIVE_COST_MODE` / `_PROFILE` and
  their forwarding in `run.sh` and `scripts/run-tp-ep-native-candidate.sh`.
- In `speculative.rs`: `confidence_cutoff`, `reuse_floor`, `cost_model`,
  `select_reuse_prefixes`, `confidence_prefix`, `trace_cost_forecast`, the
  legacy `19864 + 803·rows + 636·unique` constants, three sigmoid copies.
- CLI: `--dspark-confidence-cutoff`, `--dspark-reuse-floor`, hidden
  `--dspark-adaptive`; all of `ds41rt-core/src/dspark_policy.rs` (suffix
  removal selector and confidence cutoff).
- Scripts: `fit-ds41-placement-costs.py`, `fit-ds41-adaptive-policy.py`,
  `compare-ds41-route-cost-models.py`, `summarize-ds41-route-policy.py`.
  Archive the phase1/phase2/release-v5 adaptive docs and fit JSONs.
- Re-anchor `ds41rt-core/tests/upstream_spec_acceptance.rs`.
- Keep: `--dspark-fixed` / `DSPARK_DRAFT_POLICY=full` as the A/B control,
  `draft_limit` sizing, `verify_dspark_greedy`, confidence kernel and download,
  grammar truncation, `route_capture` (now always on when dSpark is on).
- Out of scope unless bundled with the DS4 cleanup: the `commands/real_full`
  stack (its own SPS profile, calibrators, `DS41RT_REAL_FULL_DSPARK_*`), which
  is not on the native serving path.

## 9. Phases and validation

1. Estimator and stats in shadow mode. New pure module in `ds41rt-core`
   (`dspark_cost_estimate`): RLS with forgetting and robust weights; sharing
   forecaster; unit tests that recover synthetic coefficients. Feed the
   existing per-layer and per-round timings from `v41_backbone_lane.rs`,
   `scheduler.rs`, `independent.rs`. Export on `/v1/stats`. Log predicted vs
   actual. Replay the existing `ds41rt::cost_model` traces through the
   estimator offline and check it lands within ~10 % of the phase 2 fit.
2. Switch selection to the new model, delete the old modes, qualify: fixed K5,
   fixed K7, previous placement-adaptive image, new policy; code and mixed C1
   as the target, C2–C16 for non-regression, three runs each. Adaptation
   check: change the RTX power cap mid-run and watch `A`, `c_loc` move; the
   `/v1/stats` error must stay bounded.
3. Spark kernel-time header field; confidence bias if the stats show drift;
   prefill seeding.

Risks: with N held constant (shadow phase, or a long run at one K) `L` and `b`
are confounded; that is harmless for predicting that N and resolves once the
policy varies N. Kmin = 0 needs the 1-row verification path confirmed
qualified. Outlier cycles (graph capture, prefill interleave) need the robust
weighting from the start.
