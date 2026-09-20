# Decode-performance next steps: TP2×EP2 vs TP4×EP1 (CPU/local analysis, revision 2)

Prepared 2026-09-20, CPU/local only (no build, GPU, remote or service action; no
Rust edits). Supersedes revision 1, which contained structural errors corrected
below. The acceptance review remains frozen and untouched.

## 1. Matched evidence (counting is the only unconfounded arm)

Source: `runs/tp-ep-preflight/g2-live/COMPARISON-g3-vs-tp4.md` (TP2EP2 vs TP4EP1;
same corpus `823f203c…`, runner `e197146a…`, N20, KV `5,037,542,400`, same
diag-v2 binaries, `RUST_LOG=info`).

- Counting: every request 191 tokens, greedy repeat-consistent, 93/93 rows, both
  arms. Δ = TP4 − TP2EP2 TPS: `+0.6 / −1.8 / +15.6 / +20.2 / +40.4` at
  C1/C2/C4/C8/C16 → TP4 equal-to-~3 % faster, gap widening with concurrency.
- Code/topic/mixed: variable lengths and greedy drift in **both** arms; excluded
  from throughput inference.
- 191 identical output tokens does **not** imply identical speculative
  proposals/acceptance: the target logits differ between the two expert
  partitions, so `proposed`/`accepted`/`emitted` and `draft_us` must be measured,
  not assumed equal.

## 2. Corrected structural accounting (what is and is not invariant)

Whole-expert ownership never splits one expert's rows across groups: every route
of an expert is assigned to exactly one group, and that group's rank(s) receive
the expert's **full** histogram count. So a rank's owned rows are a whole-expert
subset of the histogram, not a per-expert halving.

| Quantity | TP4×EP1 (4 ranks) | TP2×EP2 (4 ranks) | Invariant? |
| --- | --- | --- | --- |
| Intermediate per TP rank | 2304/4 = 576, **kernel extent 640** | 2304/2 = 1152, extent 1152 | no; TP4 pays 640/576 = **+11.1 % padding** |
| Per-rank resident routed weights, per layer (whole 384 bank) | **2,005,401,600 B** (padded) | **3,609,722,880 B** | **no — TP2 is 1.80×** (20 layers: 40.1 GB vs 72.2 GB) |
| Per-expert packed shard | 5,222,400 B (padded) | 9,400,320 B | no |
| Active experts per rank | all active experts (÷4 shard) | ~half the active experts, **all rows of each** (÷2 shard) | scheduler-dependent |
| Wire to coordinator | 4 × BF16 [M,5120] planes | 4 × BF16 [M,5120] planes | **yes** |
| Reduction | `reduce` 4-plane + shared once | same `reduce` 4-plane + shared once | **yes** |

Active weight traffic is a whole-expert sum, so for a uniform histogram TP2
touches ≈half the experts at 2× shard bytes (≈0.5 × 9.40 MB vs 1 × 5.22 MB per
active expert) — roughly **10 % fewer weight bytes and no padding**, while
resident memory is 1.8× larger. With skew, the whole-expert assignment can
concentrate hot experts in one group, so per-group worst-case load and per-tile
occupancy (a group's width can be under-filled) matter; that is not a route-count
halving effect.

The logical-work argument (same top-6, same ideal 3×2304×5120 per route) does
**not** prove compute is irrelevant: it ignores the 640/576 padding, tile
occupancy, and per-expert activation distributions. Whether TP2's kernel
geometry is actually better per unit work is an open measurement.

## 3. Existing instrumentation (nested — do not sum naively)

All DEBUG-only and absent from the existing INFO runs (verified 0 lines in
`g2-live/raw/g3-coordinator.log` and `g2-live/tp4-actual/coordinator.raw`).

| Layer | Emitter | Fields |
| --- | --- | --- |
| Coordinator remote path | `v41_backbone_lane.rs:380` `ds41rt::timing` "target experts" | `routed_us`, `dispatch_us`, `shared_us`, `collect_us` |
| Coordinator collection | `v41_experts/coordinator.rs:700` "target collection" | `shared_copy_us`, `upload_us`, `receive_us`, `reduce_us` |
| Scheduler round (dSpark) | `v41_native_serve/scheduler.rs:483` "native scheduler round" | `proposed`, `accepted`, `emitted`, `draft_us`, `prepare_us`, `verify_us`, `total_us` |
| Worker expert kernel | `v41_experts/execution.rs:792` `ds41rt::expert_timing` (roles 1/5/6, 3 CUDA events) | `kernel_us`, `compact_us`, `upload_us`, `execution_host_us`, `download_us`, `active_experts`, `max_expert_rows`, `expert_rows_histogram`, `expert_rows_tail_routes`, `unique_expert_weight_bytes` |
| Transport | `DS41RT_PROTOCOL_V2_TCP_TIMING=1` | request/response capacity + span, `poll_recv_ms`, `parse_ms`, `execute_ms`, `encode_ms`, `send_ms`, `poll_send_ms` |
| RTX TP2 driver | `DS41RT_TP2_TIMING` | TP2 lane timing |

Interpretation limits: `dispatch_us` (enqueue) and `receive_us`/`collect_us`
overlap remote compute by design, so these intervals are **nested/overlapping**;
they must be compared per-phase, not summed, and a residual
(`total_us − measured`) is not evidence about the ownership planner.
`unique_expert_weight_bytes` is `resident/384 × active_experts`, a resident
share, not active weight traffic. No planner timer, per-tile/weight-traffic
counter, or pure link-latency isolation exists.

## 4. Three falsifiable hypotheses (replacing the tautological H1)

### H1 — Kernel geometry/padding: TP2's 1152 unpadded shard beats TP4's 576→640 padded shard per unit of work, enough to offset fixed costs in wall-clock
The only measured number is the end-to-end TPS delta; the kernel-level claim is
untested. Per rank, TP4 computes all active experts on a shard padded 576→640
(+11.1 %) while TP2 computes ~half the active experts (each with its full rows)
on a 2× wider, unpadded shard (≈10 % fewer active weight bytes for a uniform
histogram).
**Proof needed:** `kernel_us` and `compact_us` **normalized by logical work**, not
compared naively per owned route: per rank, logical work is
`Σ_active rows_e × (2304 / TP)` (plus a variant using the kernel extent 640/576
to expose padding), so the comparable metric is µs per (routed row × shard
column). The **primary** comparison is the **binned** histogram at the same
actual M and, per (topology, layer, M), the **worst-rank median** of kernel and
of compact (each metric picks its own worst rank). Worker timing lines carry no
request id, so this is a worst-rank median over a `(layer, rows)` bucket, **not**
a per-request critical path and not a per-route ratio across shard widths.
`active_experts`, the binned histogram and `max_expert_rows` travel with every
sample; the histogram cannot prove an expert-id-level match.
**Falsifier:** after shard-width normalization TP2's kernel/compact cost per
logical unit is equal or worse, or the padding-adjusted saving does not appear.
**Note:** `unique_expert_weight_bytes` is a resident-footprint estimate
(`resident/384 × active_experts`), **not measured traffic**; keep it out of the
traffic argument.

### H2 — Dispatch/overlap/fixed-cost: per-group wire/latency and coordinator work offset any kernel gain, and the nested timers hide where
TP2 doubles the number of groups producing planes for the same batch while the
4-plane wire/reduce contract is unchanged; per-group latency tails and
coordinator work per iteration can dominate at higher C.
**Proof needed:** per-phase `routed_us`, `dispatch_us`, `receive_us`,
`upload_us`, `reduce_us`, `collect_us` at C1 and C16 compared **phase by phase**
(overlaps acknowledged), plus worker `download_us`; scheduler iteration
`total_us` against matched work.
**Falsifier:** all remote-phase intervals are equal across arms and the delta is
entirely in `kernel_us`/tile distribution.

### H3 — dSpark/workload: proposal and acceptance counts differ between arms even at identical 191-token output, changing tokens/iteration
Different expert partitions change target logits, which can change dSpark draft
acceptance; identical final length does not imply identical speculative work.
**Proof needed:** `proposed`, `accepted`, `emitted`, `draft_us`, `prepare_us`,
`verify_us` per arm at matched C, plus accepted tokens per verify round; verify
acceptance equality before attributing TPS deltas to expert geometry.
**Falsifier:** proposal/acceptance distributions are statistically identical and
tokens/iteration match.

## 5. One matched profile experiment (for 915, after the transport package; not run)

Same-workload, matched, bounded: counting corpus only (fixed 191 tokens, greedy
clean), C1 and C16, 3 repeats, both arms, identical N20 / KV `5,037,542,400` and
the same diag-v2 binaries/corpus hashes as the matched comparison.

- Instrumentation: `RUST_LOG=info,ds41rt::timing=debug,ds41rt::expert_timing=debug`
  and `DS41RT_PROTOCOL_V2_TCP_TIMING=1`; capture coordinator plus all four worker
  logs per arm. This is the minimum set that reaches every existing subtimer.
- Per repeat record, at matched C: actual request rows **M**, per-rank
  `active_experts` and `expert_rows_histogram`/`max_expert_rows`, `kernel_us`,
  `compact_us`, host phases, coordinator phase timers, transport spans/timings,
  and dSpark `proposed`/`accepted`/`emitted`/`draft_us`/`verify_us`. Retain every
  raw observation alongside the aggregates.
- Analysis per hypothesis (no summing overlapping timers; medians/p95 with raw
  samples retained):
  - H1: shard-width-normalized kernel/compact cost per logical unit, the binned
    histogram at matched M, and the per-(layer, M) **worst-rank median** (not a
    per-request critical path).
  - H2: phase-by-phase delta at C1 vs C16 (overlap-aware), plus the worst-rank
    median of the remote phases.
  - H3: proposal/acceptance per arm and tokens per verify round.
- Decision: **H1, H2 and H3 can coexist**; the run should quantify their relative
  contribution rather than select exactly one. Only a measured dominant
  bottleneck justifies a concrete optimization; if none is visible, the next step
  is targeted instrumentation (planner timer / weight-traffic counter), not a
  kernel rewrite.

**Executed result:** the matched C1/C16 counting profile has since run; the
validated analysis (segment definition, kernel/coordinator/dSpark evidence and
hypothesis verdicts) is in
`runs/tp-ep-preflight/g2-live/profile/PROFILE-INTERPRETATION.md`. Measured
verdict: H3 is not supported, H1 is not supported as a TP2 win, and H2 has only
weak directional support (~1–2 % TP2 receive/shared-and-collect overhead); no
bottleneck is large enough to justify an optimization yet.

No speculative kernel change is proposed and no extra execution is performed by
this brief. Acceptance review stays frozen.
