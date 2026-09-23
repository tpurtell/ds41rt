# DS41RT v13 decode performance

V13 replaces every earlier adaptive dSpark length selector with the online
bandwidth-balance policy described in [dspark-bandwidth-policy.md](dspark-bandwidth-policy.md).
The image source is `00b25e7025410cdb1dadc7960b8e94ae9b89db48` (SparkInfer
`4b0954148523b5a2e93813f963d483ffd350b9c9`, universal Spark roles
`tp2;tp3;tp6`). All measurements use the official DeepSeek V4.1 Flash
checkpoint `dba1be0a40aa45a94ad051997016db3960a90277`, four Spark workers
(ostrich, dodo, emu, kiwi, TP4), and RTX PRO 6000 Blackwell cards at 400 W with
standard memory clocks (13,365 MHz loaded). The coordinator local image ID is
`sha256:f0a7c245be0224ee6d472bc57c6e42eb578ba6ba8b5b2195afe558937c5282de`; the
Spark local image ID is
`sha256:2ce1637492b251e9c3a33393bef16181f3d0d9448ee9e58a78a861b4db9b4f62` on
all four hosts. The launch configuration is `ds41rt.build-v13.config`
(SHA-256 `db7224a6f1bc2bf033447899d18f782092fcb3fe38f21da136ec35850c0c053c`).

## Evaluation against the v12 adaptive policy

Each layout ran twelve sessions in the interleaved order v12, v13, v13-w5,
v13-w5, v13, v12, v13, v13-w5, v12, v12, v13-w5, v13, where v13-w5 is the
final configuration (drafting the trained five-token block) and v13 is the
same build loading the seven-token block with per-round width choice. Every
session relaunches its arm's exact image pair through `run.sh`, discards a
greedy battery and a code C1–C16 sweep as warmup, then runs identical paired
workloads with fixed nonce seeds: three greedy nine-case batteries (plus
counting), two T0.7/top-p 0.9 batteries (seed 20260923), code and topic C1–C16,
and a mixed C1–C16 sweep. Sessions are the independent units. Ratios are the
geometric mean of the arm's session scores; intervals are Welch 95% intervals
on the session log-ratios. Bold marks intervals that exclude 1. No session had
a quality failure. Greedy outputs matched a v12 output for the same prompt on
23–27 of 30 keys; the remainder differ through batch-shape numerics, not
correctness (every token is target-verified).

| Family | 1 RTX: v13-w5 / v12 | 1 RTX: v13 per-round width / v13-w5 | 2 RTX: v13-w5 / v12 | 2 RTX: v13 per-round width / v13-w5 |
|---|---:|---:|---:|---:|
| Greedy C1, weighted battery | 1.007 (0.986–1.028) | 0.985 (0.969–1.001) | 1.014 (0.981–1.049) | 0.997 (0.971–1.024) |
| Greedy C1, per content case | **1.021 (1.003–1.040)** | 0.997 (0.986–1.009) | **1.035 (1.000–1.072)** | 1.012 (0.985–1.041) |
| T0.7/top-p 0.9 C1, weighted | **1.033 (1.013–1.054)** | 0.992 (0.976–1.009) | 1.017 (0.988–1.048) | 0.987 (0.957–1.017) |
| T0.7/top-p 0.9 C1, per case | **1.023 (1.003–1.043)** | 1.003 (0.985–1.022) | **1.048 (1.017–1.080)** | 0.995 (0.968–1.024) |
| Code C1 (concurrency harness) | 1.003 (0.994–1.012) | **0.972 (0.951–0.994)** | **0.979 (0.972–0.986)** | **0.981 (0.970–0.992)** |
| Topic C1 | **0.952 (0.941–0.963)** | **0.970 (0.947–0.993)** | **1.021 (1.014–1.029)** | 0.987 (0.961–1.014) |
| Mixed C1 | 0.995 (0.974–1.015) | 0.984 (0.958–1.010) | 0.998 (0.979–1.017) | 1.007 (0.974–1.040) |
| Code C2–C16 | 1.005 (0.999–1.012) | **0.949 (0.933–0.965)** | **1.026 (1.007–1.046)** | **0.961 (0.937–0.985)** |
| Topic C2–C16 | 1.022 (0.998–1.048) | 0.973 (0.942–1.006) | **1.027 (1.005–1.050)** | 1.000 (0.976–1.026) |
| Mixed C2–C16 | 0.986 (0.954–1.019) | 0.983 (0.945–1.023) | **1.046 (1.022–1.069)** | 0.993 (0.974–1.013) |

With the default five-token block on two RTX, v13 beats v12 by 2–5% on
per-case greedy and sampled C1 decode, topic C1, and code, topic and mixed
C2–C16; the weighted batteries and mixed C1 tie. On one RTX it beats v12 by
2–3% on per-case greedy and sampled C1 decode and ties elsewhere except topic C1. Losses: topic C1 on one RTX
(−4.8%), code C1 in the concurrency harness on two RTX (−2.1%), and counting
C1 on two RTX, where v12 drafted seven tokens by default (195 versus 221 tok/s
paired medians; counting is the weight-zero diagnostic). Per-round width choice
recovers counting (219 on two RTX, 174 versus 159 on one RTX) but loses to the
five-token block under concurrency, so it is opt-in with
`--dspark-draft-limit 7`.

The policy passed through four builds before this result, each evaluated the
same way against v12:

1. Uncalibrated, five-token block (ten sessions, one RTX): ties everywhere
   (weighted greedy 0.998, sampled 1.023, concurrency 0.99–1.03, every
   interval spanning 1).
2. Seven-token block with per-position offset calibration: positions 6 and 7
   reported about 0.95–0.99 confidence against 0.78–0.86 observed acceptance.
   A single logit offset saturated at its bound, which motivated per-position
   Platt calibration.
3. Fixed seven-token block with Platt calibration (three arms): the seven-token
   draft costs about 0.45 ms per round whether or not the extra rows are
   verified and lost 5–9% under concurrency and on topic C1.
4. Per-round 5/7 width choice (above).

Raw sessions and analyses are under `runs/v13-release/ab/` (ignored); the
evidence package carries them.

## Release decode tables

Campaign on the final images, 2026-09-24. Each layout was launched with the
default configuration, warmed with one discarded nine-case battery and one code
C1–C16 sweep, then measured: the nine-case battery (three repeats, nonce seed
61001, counting included), per-case acceptance (three repeats per case, nonce
seed 62001, from `/v1/stats` deltas), concurrency (counting, code, topic,
C1–C16, three repeats, nonce 20260914), three mixed sweeps (nonce seed
56001), and retained-context decode (three repeats). A fresh launch measured
the 2K retained control, and a `--no-dspark` launch measured target-only
decode. The retained-context source is
`~/.cache/ds41rt-v7-bench/release-context-source.md` (SHA-256
`1881a1d1349b22d7ea71eab2aa0347c35bb8ce4000931bf6c0b5ca6fcf5d55f8`), the
v7-era file; the v6 source is no longer available, so retained rows are not
comparable to v6 row for row. Weighted dSpark repeats were
91.1, 89.91, 92.66 (1 RTX) and 108.91, 109.51, 107.8 (2 RTX);
target-only repeats were 46.95, 47.46, 47.3 and 50.1, 50.33, 51.45.

Every measured workload passed its quality and cache checks. Two harness
artifacts are disclosed: the counting-only acceptance run reports
`passed=false` because it has no weighted case (all three samples passed), and
the first schema-JSON acceptance run reused natural JSON's prompt under the
same nonce seed and was served from the prompt cache. It was rerun after a
fresh warm launch with its own nonce seed (63001), and the table uses the rerun.

**Headline decode.**

| Measurement | Official 1x | Official 2x |
|---|---:|---:|
| Counting decode | 159.35 | 192.56 |
| Weighted decode | 91.10 | 108.91 |
| C1 code decode | 134.01 | 161.20 |

**Content-type decode.** Median tokens/s of three repeats.

| Case | 1 RTX target | 1 RTX dSpark | 2 RTX target | 2 RTX dSpark |
|---|---:|---:|---:|---:|
| Code | 49.22 | 134.01 | 53.41 | 161.20 |
| Code with reasoning | 47.62 | 99.72 | 50.67 | 122.97 |
| Math | 45.80 | 116.55 | 48.65 | 133.95 |
| Fable | 44.77 | 57.11 | 48.12 | 64.03 |
| Hello | 44.71 | 65.56 | 47.81 | 94.79 |
| Topic | 46.49 | 73.89 | 49.17 | 88.55 |
| Natural JSON | 48.44 | 117.65 | 51.71 | 133.82 |
| Schema JSON | 48.59 | 121.34 | 50.50 | 141.18 |
| Multilingual | 45.98 | 75.29 | 49.66 | 83.83 |
| Counting 1–200 | 49.41 | 159.35 | 53.74 | 192.56 |
| Weighted (eight-weight corpus) | 47.30 | 91.10 | 50.33 | 108.91 |

**dSpark acceptance.** Accepted/verified drafts, and mean emitted tokens per
verification round in parentheses, from `/v1/stats` over three requests per case.
Unlike earlier releases, these include the policy's trimming: rows it declines
to verify are neither accepted nor rejected.

| Content | 1 RTX | 2 RTX |
|---|---:|---:|
| Code | 89.74% (5.26) | 86.97% (5.18) |
| Code with reasoning | 74.48% (3.76) | 71.20% (3.80) |
| Math | 78.67% (4.61) | 75.45% (4.46) |
| Fable | 55.36% (1.95) | 57.64% (2.25) |
| Hello | 56.73% (2.20) | 45.45% (2.20) |
| Topic | 58.88% (2.57) | 55.02% (2.63) |
| Natural JSON | 75.42% (3.47) | 66.18% (3.81) |
| Schema JSON | 96.00% (5.27) | 89.47% (4.64) |
| Multilingual | 58.51% (2.61) | 54.51% (2.69) |
| Counting 1–200 | 98.37% (5.78) | 99.01% (5.92) |

**Concurrency scaling.** Median aggregate tokens/s from earliest first output
to final completion, including admission gaps.

| Concurrency | 1 RTX counting | 2 RTX counting | 1 RTX code | 2 RTX code | 1 RTX topic | 2 RTX topic |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 160.76 | 193.37 | 131.47 | 164.77 | 75.89 | 91.70 |
| 2 | 251.60 | 303.43 | 213.17 | 274.43 | 125.82 | 153.29 |
| 4 | 451.44 | 542.63 | 368.85 | 470.06 | 228.43 | 286.43 |
| 8 | 745.95 | 895.77 | 615.86 | 786.29 | 353.81 | 476.53 |
| 16 | 1,267.07 | 1,520.41 | 1,022.48 | 1,282.02 | 595.52 | 776.60 |

**Mixed traffic.** Code/fable/topic mix; aggregate tokens/s median and range
across three sweeps.

| Concurrency | 1 RTX | 2 RTX |
|---|---:|---:|
| 1 | 108.94 (107.12–131.60) | 153.05 (151.78–158.25) |
| 2 | 105.76 (103.31–113.80) | 140.11 (133.53–143.55) |
| 4 | 149.58 (145.37–155.34) | 182.99 (181.09–185.72) |
| 8 | 168.19 (157.19–178.00) | 207.66 (201.77–210.68) |
| 16 | 193.34 (182.91–199.67) | 298.15 (278.97–301.43) |

**Decode over retained context.** Weighted nine-category dSpark tokens/s with
verified prefix reuse.

| Retained base | 1 RTX | 2 RTX |
|---|---:|---:|
| 0K | 89.03 | 106.08 |
| 2K | 88.80 | 107.06 |
| 32K | 88.10 | 102.14 |
| 64K | 89.91 | 101.92 |
| 128K | 86.03 | 98.56 |
| 256K | 84.27 | 97.68 |

Across the campaign the policy's round-time prediction error was 6.5% (one RTX)
and 5.3% (two RTX) mean absolute, with mean error under 50 µs.
