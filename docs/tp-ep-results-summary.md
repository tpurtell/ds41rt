# TP×EP replicated-group results summary

Scope: official `deepseek-ai/DeepSeek-V4.1-Flash` only, opt-in replicated expert
groups. Detailed engineering record: `docs/tp-ep-implementation-plan.md`. Nothing
here changes a default or promotes a release.

## Acceptance categories (read this first)

| Category | State |
| --- | --- |
| Implementation | Present and opt-in; default unchanged |
| Native/component correctness | **Accepted** — 135 checks (all-rank) + native tie-seed replay (component) |
| E2E operational acceptance | **Accepted for executed scope** — every arm completed 372 requests; objective/content checks 372/372 except TP4 (371/372) |
| Quality/determinism acceptance | **NOT accepted** — strict greedy-repeat FAILs remain in every arm |

A completed arm is **not** a quality pass, **not** a speedup claim, and **not** a
release qualification.

## Deployments measured (five)

All five completed the **372-request** matrix (4 cases × 93); objective/content
checks passed 372/372 except TP4 (371/372). Four-Spark and dual-six pairs are
internally matched; cross-pair and single-RTX comparisons are **descriptive only**.

| # | Deployment | RTX | Sparks | Key actual settings | Quality |
| --- | --- | ---: | ---: | --- | --- |
| 1 | TP4×EP1 (baseline/control) | 2 | 4 | N20, KV `5,037,542,400`, draft 7, dispatch | 4 FAILs |
| 2 | TP2×EP2 (four-Spark) | 2 | 4 | N20, KV `5,037,542,400`, draft 7, dispatch | 3 FAILs |
| 3 | TP3×EP2 | 2 | 6 | N20, KV `5,037,542,400`, world 6, draft 7, dispatch | 3 FAILs |
| 4 | TP2×EP3 | 2 | 6 | N20, KV `5,037,542,400`, world 6, draft 7, dispatch | 3 FAILs |
| 5 | TP3×EP2 single-RTX all-40 | 1 | 6 | `--rtx-gpus 1`, `--rtx-expert-layers 0`, all ranks `--world 6 --first-layer 0`, auto KV, draft 5 | 3 FAILs |

## Counting decode across concurrency (tok/s)

Canonical counting-1-64 medians of three repeats (client-observed aggregate
decode; not per-stream or kernel). Read the whole curve, not only the endpoints.
The client metric is `sum(completion_tokens - 1) / (latest_finish - earliest_first_output)`
using absolute client timestamps and assuming the first chunk contains one token;
it approximates aggregate decode throughput, not isolated GPU execution.

| # | Deployment | C1 | C2 | C4 | C8 | C16 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | TP4×EP1 control | 229.9 | 345.6 | 532.1 | 814.4 | 1272.0 |
| 2 | TP2×EP2 four-Spark | 229.3 | 347.4 | 516.5 | 794.2 | 1231.6 |
| 3 | TP3×EP2 dual-6 | 245.7 | 359.0 | 588.5 | 954.0 | 1410.7 |
| 4 | TP2×EP3 dual-6 | 241.5 | 393.3 | 614.9 | 909.5 | 1407.4 |
| 5 | TP3×EP2 single-6 | 161.6 | 288.9 | 484.5 | 732.3 | 1200.5 |

- **Dual-6 pair (3 vs 4) is matched** (corpus `99034171…`, runner `e197146a…`):
  the curve crosses — TP2EP3 leads C2/C4 (393.3/614.9 vs 359.0/588.5), TP3EP2
  leads C8/C16 (954.0/1410.7 vs 909.5/1407.4), C1 near-equal. **No TP2/TP3
  speedup claim.**
- **Four-Spark pair (1 vs 2)** is close across the curve. Cross-pair and arm 5
  are **descriptive only**; **no statistical winner** (small repeats, no CI).
- **Arm 5 (single-RTX all-40) is standalone**, accepted after a retry via the
  system-wide `sync + drop_caches=1` helper as an **operational workaround, not
  admission-gate proof**.
- **Strict FAILs retained:** every arm drifts on `code-merge-intervals`,
  `topic-virtual-memory`, `mixed-code-fix`; `counting-1-64` is greedy-consistent;
  TP4×EP1 also had one code C8 content-check failure.

## Seed control (bounded; full cause unresolved)

Same-binary C1 control, tie-seed **dispatch** vs **layer**: `dispatch` reproduces
the frozen default drift; `layer` is **exact in all four C1 cases** but still has
**3 greedy failures at C2/C16** — a **C1 directional control, not a C2+ fix**.
Strict repeatability remains open. A bounded review of 20 non-truncated outputs
found no semantic errors in that sample (phrasing/formatting/equivalent code, not
executed), which does not override any strict FAIL. See
`runs/tp-ep-preflight/g2-live/supplemental-semantic-review.md`.

## Kernel timing (two separate 96-arm batches)

**First batch (archived; widths 64/128; no reuse).** 96/96 accepted, lease
released, in-window correctness 17. TP2EP2 was faster only in **w128 M1/M8/M16**
(1.122/1.100/1.117), parity elsewhere; six-Spark M8/M16 within 1 %. Single-rank
sequential projections on a synthetic no-reuse table — a stress comparison, not a
production predictor. **Kept separate**; its attestation gap remains.

**Remaining batch (w192 + M64) — independently ACCEPTED 96/96.** Frozen audit
`08a7d856…` (313 lines; **no blocking items**) after final review `c22a2673…`
(earlier `c4154a1d…`); validator **`d9632e81…`**: `ACCEPT validated=96 present=96
missing=0, errors []`, **35 selftests**, **97/97** per-file hashes, 96 `RUN_RC=0`.
Part A same-session w64 control at M8/M16: **w192 lower by 4.9–6.5 %**. Part B M64
(all four roles, widths 64/128/192): **w192 best in all 12 cells — 20–28 % below
w64, 6–17.5 % below w128**. The decision rests on **96/96 within-round paired
orderings** (48 Part A + 48 Part B, including the projected max-group), **not** on
an aggregate order-delta argument: those mixed-sign deltas (median |Δ| 4.7 µs, max
49 µs) are **comparable** to the smallest width gaps (~54–61 µs). **Decision:
retain tested cap80/w192 and cap1/w64 — no extra AOT build, no default-serving
change.** Caveats: JIT (**not AOT**); sequential single-shard **projected** groups;
diagnostic **LPT**; Part B = 384 routes / 48 distinct experts globally across EP
groups; direction only. Width 128 was **not tested at M8/M16 in this batch**;
there is **no speedup claim versus the current default**. **V2 caller repair VERIFIED** (`1dc81573…`;
current tool `ca8c9054…`, benchmark `ecf52e33…`): caller orchestration, truthful
shared labels and the early row guard are covered, focused run **80 passed / 5
CUDA skips / 0 failed** with CUDA hidden (executor superset 87 pass / 5 skip incl.
7 schedule tests). **Canonical digest `bce43420…`** (executor `589f69b5…`
superseded); raw arms + `master.log` unchanged; the frozen measurement sources
(`8255d1df…` harness, archived benchmark `2075cbf8…`) stay separate from the
current tool. Fleet **restored and verified** (original five, never recreated;
health/models/hashes/config PASS; no MOA/RHEA release) — see
`runs/tp-ep-preflight/FINAL-RESTORATION-REPORT.md`.

## Prefill validation (all five deployments)

Validated from `runs/tp-ep-preflight/prefill/COMPARISON-VALIDATED.md`: uncached
C1, `bases [0]`, sizes **512 / 2048 / 8192**, **3 repeats, no explicit warmup**,
`max_tokens=1`. **TTFT median ms** (one token). All **45/45** passed independent
gate re-derivation (`attempts == 1`, exact target, `hit == 0`, `miss == target`,
`completion == 1`, unique prompt sha, one fingerprint/file).

| Deployment | 512 tokens: TTFT ms | 2048 tokens: TTFT ms | 8192 tokens: TTFT ms |
| --- | ---: | ---: | ---: |
| TP4EP1 control (2 RTX + 4 Spark) | 162.4 | 308.8 | 969.1 |
| TP2EP2 four (2 RTX + 4 Spark) | 164.1 | 321.3 | 972.2 |
| TP3EP2 dual-6 (2 RTX + 6 Spark) | 180.4 | 302.1 | 957.1 |
| TP2EP3 dual-6 (2 RTX + 6 Spark) | 148.1 | 314.7 | 960.6 |
| TP3EP2 single-6 (1 RTX + 6 Spark) | 255.6 | 585.1 | 1283.8 |

- **Matched pairs** share prompts **9/9** (TP3EP2-vs-TP2EP3 dual-6;
  TP2EP2-vs-TP4EP1 four); single-RTX is **0/9**, not matched. **No formal winner:**
  3 repeats, no warmup, overlapping ranges; first-request effects included (up to
  1.64×). HTTP/SSE **proxies**, not kernel timings.

## Evidence

- Four-Spark pair: `runs/tp-ep-preflight/g2-live/COMPARISON-g3-vs-tp4.md`
- Six-node: `runs/tp-ep-preflight/g2-live/SIX-CONSOLIDATED.md` + independent audit
  `runs/tp-ep-preflight/g2-live/six-independent-review.md`
- Seed control: `runs/tp-ep-preflight/g2-live/SEED-dispatch-vs-layer.md`
- Kernel: `docs/tp-ep-kernel-timing-milestone.md` +
  `runs/tp-ep-kernel/timing/independent-review.md`; remaining batch
  `runs/tp-ep-kernel/timing/independent-review-remaining.md` +
  `results96-independent-analysis.json` / `results96-independent-tables.txt`
- Prefill: `runs/tp-ep-preflight/prefill/COMPARISON-VALIDATED.md`

## Remaining measurements (no performance speculation)

1. Strict repeatability FAIL (C2+ greedy drift) — unresolved.
2. ~~Width-192 M8/M16 + M64 reuse-vs-192~~ — **measured, independently accepted
   96/96; decision recorded** (cap80/w192, cap1/w64).
3. ~~Sampled uncached prefill for the other deployments~~ — **completed** (all five).
4. Whole-expert ownership at E2E scope — not asserted; component evidence only.
5. Direct-RC growth/reconnect — **passed at 2 and 4 ranks**; self-RC EP1 wire echo,
   not distributed GPU/EP proof.

## Status

**Scoped implementation, measurement, selection and restoration are complete**
— opt-in replicated expert groups implemented; native component correctness and
the four-Spark/dual-six/single-RTX E2E matrices executed; the kernel
measurement + independent audit accepted with the retention decision recorded;
the ledger closed (`bce43420…`, validator `d9632e81…`); V2 caller repair verified
(`1dc81573…`, review `08a7d856…`); and the fleet restored/verified against the
original five (`runs/tp-ep-preflight/FINAL-RESTORATION-REPORT.md`). **No release
qualification is claimed**, and **all strict quality FAILs and the six daemon
baseline failures are preserved**. Optional future hypotheses (non-default widths,
AOT adoption, distributed/DRAM measurement) remain **distinct from these
completion gates**, not open gates. See
`runs/tp-ep-preflight/final-rust-transport-regressions.md`.

No hardware, test or code change was made in producing this summary.
