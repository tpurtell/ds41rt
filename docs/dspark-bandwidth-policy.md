# dSpark verification length: online bandwidth-balance policy

Introduced in v13. It replaces every earlier adaptive length selector on the
native serving path: the joint suffix-removal search with its placement cost
profiles, the confidence cutoff, and the reuse-adjusted cutoff. It needs no
offline calibration, profile file, or per-configuration fit.
`--dspark-fixed` (`DSPARK_DRAFT_POLICY=full`) still verifies every available
draft and remains the control arm. Implementation:
`rust/crates/ds41rt-core/src/dspark_policy.rs` (pure policy) and
`rust/crates/ds41rt-daemon/src/v41_native_serve/speculative/policy.rs`
(placement, observations, stats export).

## What a verification round costs

At concurrency 1, a round drafts `draft_limit` proposals, then runs 40 layers.
Each remote layer is a blocking round trip. The RTX runs attention and the
router, posts rows to the four Spark ranks over RoCE, runs the shared expert
while the Sparks work, then waits, uploads, and reduces. The Sparks are idle
during attention and the RTX is idle during most of the wait. Wire bytes are
negligible (about 5.4 KB per row out and 10.2 KB back per rank).

The Spark cost is weight traffic. The grouped slice kernels read each routed
expert's slice once per 16-row group. On one GB10 a TP4 slice of the official
checkpoint is 4,700,160 bytes. The same kernel family runs resident RTX layers
with 18.8 MB (TP1) or 9.4 MB (TP2) slices.

## Objective

Work that each committed token needs anyway is paid once, whichever round
verifies it. So the decision controls fixed cost per round plus the cost of
rejected rows. The policy maximizes

```
E[tokens](lengths) / T(lengths),   E = sum_r (1 + sum_{j<=k_r} prod_{i<=j} p_{r,i})
```

where `p_{r,i}` is the sigmoid of the drafter's confidence for position `i`
(conditional on earlier positions being accepted). Subtracting the
once-per-token work from `T` shifts the ratio by a constant, so this argmax is
also the minimum of fixed plus wasted resource per committed token. A row with
certain acceptance is always verified.

## Cost model, fitted online

For a lane shape with `N` rows over `R` requests:

```
T = A + B*N + C*R                                        (round residual)
  + sum_{l>=1} ( alpha_c + beta_c*N + gamma_c * MB_l )   (per layer, class c)
MB_l = expert_bytes(l) * sum_e ceil(routes_{l,e} / 16) / 1e6
```

Classes are RTX-local and Spark-remote. `expert_bytes` comes from the
installed placement: `3 * 5120 * (2304 / tp) * (0.5 + 1/32)` for MXFP4
(`1/16` for NVFP4). Changing TP width or quantization changes these known byte
counts. `1/gamma_c` is the effective marginal bandwidth, which absorbs clocks,
thermal state, occupancy, and quantization-specific kernel efficiency.

Each lane stamps every layer's FFN completion while routes are captured.
Consecutive stamps give each layer's wall time, and routes give its groups.
One regression per class takes a sample per layer per round. A second
regression takes the round total minus the layer sum. Both use exponentially
forgotten sufficient statistics (per-sample decay 0.998 for layers, 0.98 for
rounds), Huber weighting at three residual scales, and nonnegative active-set
solves. Physical priors (1.2 TB/s RTX, 175 GB/s Spark) carry the weight of a
hundredth of a sample. They only resolve directions the data cannot, such as
the intercept/row split while every round has the same shape. Rows and
traffic are strongly correlated, so any stronger prior biases the fitted
bandwidth along that near-null direction; a unit test covers this.

Fits are kept separately for a lane running alone and for a lane whose peer
lane is active (the lanes share one scheduler thread). Until a regime's fit
has 120 layer samples per class and 12 rounds, the lane verifies every draft,
which also serves as the warmup.

## Expert-traffic forecast

Each request keeps the routes of its last 24 committed tokens. Draft row `k`
stands in for the `k`-th most recent committed token. The forecast averages
four shifted windows. The lane's expert counts are the exact multiset union
across its requests, so experts shared between requests are charged once and
the 16-row group boundary is priced exactly. Tokens without history use an
observed per-layer novelty rate. Recent-token routes are the stand-in because
adjacent tokens share experts far more than random tokens do.

## Selection

Start every request at its anchor. Repeatedly evaluate each request's next
draft row against the current joint shape and add the one with the best
resulting ratio, continuing through temporary losses to the full shape, then
return the best visited shape (longer on ties). One request needs at most
`draft_limit` steps. Unit tests check agreement with exhaustive search. A
selection takes under 10 µs. Anchor-only rounds are allowed.
Grammar-constrained requests are priced like any other; their proposals are
first truncated to the grammar.

## Monitoring

`/v1/stats` → `dspark_policy` exports the fitted coefficients and implied
bandwidth per class and regime, the round fit, prediction error, the draft
length histogram, acceptance totals, and per-position confidence reliability
(mean predicted versus observed acceptance where a position was reached).
`RUST_LOG=ds41rt::draft_policy=debug` logs each decision and observation.

## Evaluation against v12

See [release-v13-performance.md](release-v13-performance.md) for the
interleaved A/B against the v12 placement-cost policy and the release tables.
