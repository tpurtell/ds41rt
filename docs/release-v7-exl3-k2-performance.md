# DS41RT v7 performance - diffbot DeepSeek-V4.1-Flash-EXL3 (2.0 bpw, K=2)

Checkpoint `diffbot/DeepSeek-V4.1-Flash-EXL3-2.0bpw-2x-RTX-PRO-6000`. Uniform K=2 trellis with a two-tier kernel family; the paired 3.25 bpw publication uses the second tier.

Measured with the v7 release protocol: three samples per cell, 400 W per card, standard memory speed, reasoning code at high effort, all other throughput cases with thinking disabled, dSpark speculative decoding enabled for every decode cell.
Each report table is generated from the raw bench output in the v7 package by `scripts/render-ds41-v7-quant-reports.py`; an em dash is a measurement not yet taken.

## Configurations

- **2x RTX PRO 6000, no Spark** — All 40 routed-expert layers resident as TP2 on the pair; no Spark worker.

## Headline

Tokens/s.

| Measurement | 2x RTX PRO 6000, no Spark |
|---|---:|
| Best prefill | — |
| Counting decode | 337.35 |
| Weighted decode | 145.10 |
| C1 code decode | 222.06 |

## Content-type decode

Median tokens/s.

| Case | 2x RTX PRO 6000, no Spark |
|---|---:|
| Code | 222.06 |
| Code with reasoning | 160.72 |
| Math | 196.75 |
| Fable | 78.69 |
| Hello | 114.88 |
| Topic | 98.94 |
| Natural JSON | 159.55 |
| Schema JSON | 161.40 |
| Multilingual | 100.93 |
| Counting 1-200 | 337.35 |

## Prefill

Median effective tokens/s after shape warmup, by retained context and added tokens.

**2x RTX PRO 6000, no Spark**

_Not measured yet._

## Not yet measured

- 1x compact profile (32 GiB card simulated, 2 Sparks) - not implemented; needs TP2 sharding on the expert side
- 2x tool-call evaluation runs
