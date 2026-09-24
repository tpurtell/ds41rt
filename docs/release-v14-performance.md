# DS41RT v14 decode performance

V14 shortens the verification cycle (see [release notes](release-v14-notes.md)).
The image source is `af9d96923f70af1cd385960e28450e5ec0653ff8` (SparkInfer
`7fcc094edcc93af61fdfbe14300100e3204363ea`, universal Spark roles
`tp2;tp3;tp6`). All measurements use the official DeepSeek V4.1 Flash
checkpoint `dba1be0a40aa45a94ad051997016db3960a90277`, four Spark workers
(ostrich, dodo, emu, kiwi, TP4), and RTX PRO 6000 Blackwell cards at 400 W with
standard memory clocks (13,365 MHz loaded). The coordinator local image ID is
`sha256:c49493737243adb693023d02d0dabbe0b64c56668ada7e55b3f70657811bf75e`; the
Spark local image ID is
`sha256:6720bdb57173e5f88185e1d8c3b41d7285bed255d2301256859fab8cf6301589` on
all four hosts. The launch configuration is `ds41rt.build-v14.config`
(SHA-256 `d7c3f898bbb8e4c500672ee406e5a4a016feec599c163eb75ccc23387f893fd3`).

V14 is faster than v13 on every measured family on both layouts. C1 decode
gains 12–19%, and two-RTX sampled decode gains 35–38% because sampling now
stays on the GPU. Concurrent traffic gains 6–17%. The only key that did not
improve is two-RTX code at C16 (1,312 versus 1,299 tok/s, −1.0%). Target-only
decode, which runs the same chained verification pass without drafts, rises
from about 47 to 60 tok/s on one RTX and from about 50 to 61 tok/s on two.

## Evaluation against v13

Each layout ran eight sessions in the interleaved order v13, v14, v14, v13,
v13, v14, v14, v13. Every session relaunches its arm's exact image pair through
`run.sh` from that arm's own checkout, discards a greedy battery and a code
C1–C16 sweep as warmup, then runs identical paired workloads with fixed nonce
seeds: three greedy nine-case batteries (plus counting), two T0.7/top-p 0.9
batteries (seed 20260924), code and topic C1–C16, and a mixed C1–C16 sweep.
Sessions are the independent units. Ratios are the geometric mean of the arm's
session scores; intervals are Welch 95% intervals on the session log-ratios.
Bold marks intervals that exclude 1. No session had a quality failure.

| Family | 1 RTX: v14 / v13 | 2 RTX: v14 / v13 |
|---|---:|---:|
| Greedy C1, weighted battery | **1.166 (1.135–1.197)** | **1.182 (1.177–1.186)** |
| Greedy C1, per content case | **1.156 (1.127–1.186)** | **1.182 (1.175–1.188)** |
| T0.7/top-p 0.9 C1, weighted | **1.164 (1.140–1.189)** | **1.377 (1.369–1.385)** |
| T0.7/top-p 0.9 C1, per case | **1.157 (1.129–1.185)** | **1.346 (1.338–1.354)** |
| Code C1 (concurrency harness) | **1.123 (1.116–1.130)** | **1.133 (1.126–1.140)** |
| Topic C1 | **1.164 (1.125–1.204)** | **1.185 (1.164–1.205)** |
| Mixed C1 | **1.144 (1.114–1.174)** | **1.151 (1.130–1.172)** |
| Code C2–C16 | **1.077 (1.062–1.091)** | **1.072 (1.048–1.097)** |
| Topic C2–C16 | **1.105 (1.073–1.138)** | **1.172 (1.124–1.222)** |
| Mixed C2–C16 | **1.064 (1.028–1.102)** | **1.136 (1.090–1.184)** |

Greedy outputs matched a v13 output for the same prompt on 25 of 30 keys on
one RTX and 9 of 30 on two RTX. The remainder differ through batch-shape
numerics and, on two RTX, the TP2 route-sum addition order; every token is
target-verified and every objective check passed. Within the v14 build,
chained and host-drained passes produce identical greedy outputs on both
layouts (fixed draft policy).

The cycle-time program measured each change on one RTX before adopting it; the
per-step log is in the v14 release evidence. Raw sessions and analyses are
under `runs/v14-release/ab/` (ignored); the evidence package carries them.

## Release decode tables

Campaign on the final images, 2026-09-24, with the v13 protocol. Each layout
was launched with the default configuration, warmed with one discarded
nine-case battery and one code C1–C16 sweep, then measured: the nine-case
battery (three repeats, nonce seed 61001, counting included), per-case
acceptance (three repeats per case, nonce seed 62001, from `/v1/stats`
deltas), concurrency (counting, code, topic, C1–C16, three repeats, nonce
20260914), three mixed sweeps (nonce seed 56001), and retained-context decode
(three repeats). A fresh launch measured the 2K retained control, and a
`--no-dspark` launch measured target-only decode. The retained-context source
is `~/.cache/ds41rt-v7-bench/release-context-source.md` (SHA-256
`1881a1d1349b22d7ea71eab2aa0347c35bb8ce4000931bf6c0b5ca6fcf5d55f8`), the same
file as v13. Weighted dSpark repeats were 104.52, 104.69, 107.81 (1 RTX) and
131.32, 125.77, 129.67 (2 RTX); target-only repeats were 59.62, 59.96, 59.84
and 60.67, 60.64, 60.67.

Every measured workload passed its quality and cache checks. The same two
harness artifacts as v13 are disclosed: the counting-only acceptance run
reports `passed=false` because it has no weighted case (all samples passed),
and the schema-JSON acceptance run reused natural JSON's prompt under the same
nonce seed and was served from the prompt cache. It was rerun after a fresh
warm launch with its own nonce seed (63001), and the table uses the rerun.

**Headline decode.**

| Measurement | Official 1x | Official 2x |
|---|---:|---:|
| Counting decode | 182.39 | 219.86 |
| Weighted decode | 104.69 | 129.67 |
| C1 code decode | 146.12 | 179.86 |

**Content-type decode.** Median tokens/s of three repeats.

| Case | 1 RTX target | 1 RTX dSpark | 2 RTX target | 2 RTX dSpark |
|---|---:|---:|---:|---:|
| Code | 60.63 | 146.12 | 61.76 | 179.86 |
| Code with reasoning | 59.87 | 115.78 | 60.60 | 138.99 |
| Math | 59.79 | 129.15 | 60.91 | 155.34 |
| Fable | 59.36 | 67.59 | 60.56 | 84.98 |
| Hello | 58.37 | 77.20 | 60.43 | 113.79 |
| Topic | 60.00 | 84.05 | 60.50 | 102.32 |
| Natural JSON | 59.72 | 133.54 | 60.48 | 157.00 |
| Schema JSON | 59.12 | 135.55 | 60.17 | 159.10 |
| Multilingual | 59.18 | 87.46 | 60.24 | 106.52 |
| Counting 1–200 | 60.81 | 182.39 | 61.64 | 219.86 |
| Weighted (eight-weight corpus) | 59.84 | 104.69 | 60.67 | 129.67 |

**dSpark acceptance.** Accepted drafts per verified draft, with emitted tokens
per request round in parentheses, from `/v1/stats` deltas over each case's
three repeats.

| Content | 1 RTX | 2 RTX |
|---|---:|---:|
| Code | 91.10% (5.37) | 88.05% (5.19) |
| Code with reasoning | 75.41% (3.72) | 70.88% (3.61) |
| Math | 78.54% (4.50) | 73.51% (4.49) |
| Fable | 54.94% (1.86) | 56.63% (2.14) |
| Hello | 53.97% (2.15) | 44.23% (2.31) |
| Topic | 61.29% (2.54) | 57.82% (2.62) |
| Natural JSON | 71.21% (2.96) | 74.07% (3.61) |
| Schema JSON | 96.39% (5.21) | 82.29% (4.29) |
| Multilingual | 57.55% (2.45) | 56.30% (2.73) |
| Counting 1–200 | 99.27% (5.90) | 99.47% (5.94) |

**Concurrency scaling.** Median aggregate tokens/s from earliest first output
to final completion, including admission gaps.

| Concurrency | 1 RTX counting | 2 RTX counting | 1 RTX code | 2 RTX code | 1 RTX topic | 2 RTX topic |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 183.05 | 218.43 | 143.36 | 187.76 | 86.28 | 108.02 |
| 2 | 268.07 | 325.51 | 221.85 | 284.08 | 131.75 | 168.36 |
| 4 | 480.50 | 595.51 | 381.69 | 516.72 | 251.27 | 311.28 |
| 8 | 768.33 | 992.72 | 610.86 | 861.79 | 382.46 | 487.66 |
| 16 | 1,326.87 | 1,622.69 | 1,057.35 | 1,394.33 | 605.29 | 834.23 |

**Mixed traffic.** Code/fable/topic mix; aggregate tokens/s median and range
across three sweeps.

| Concurrency | 1 RTX | 2 RTX |
|---|---:|---:|
| 1 | 128.34 (127.87–146.76) | 171.68 (171.23–180.94) |
| 2 | 119.97 (116.85–128.93) | 167.00 (162.91–172.04) |
| 4 | 160.51 (142.95–179.96) | 214.11 (197.18–218.97) |
| 8 | 171.70 (167.87–172.92) | 247.43 (240.65–252.11) |
| 16 | 212.53 (203.73–213.00) | 314.35 (308.25–319.88) |

**Decode over retained context.** Weighted nine-category dSpark tokens/s with
verified prefix reuse.

| Retained base | 1 RTX | 2 RTX |
|---|---:|---:|
| 0K | 102.79 | 124.91 |
| 2K | 108.50 | 129.83 |
| 32K | 100.37 | 118.29 |
| 64K | 99.76 | 123.70 |
| 128K | 99.13 | 119.95 |
| 256K | 94.70 | 108.22 |

Across the campaign the policy's round-time prediction error was 7.1% (one RTX)
and 5.4% (two RTX) mean absolute, with mean error under 70 µs.
