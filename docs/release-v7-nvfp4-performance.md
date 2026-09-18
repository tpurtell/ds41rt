# DS41RT v7 performance - NVIDIA DeepSeek-V4.1-Flash-NVFP4 (W4A4)

Checkpoint `nvidia/DeepSeek-V4.1-Flash-NVFP4`. Weights are ModelOpt NVFP4 (W4A4): E2M1 payload with E4M3 K16 block scales, activations quantized in-kernel from BF16 rows.

The target v7 release protocol uses three samples per cell, 400 W per card, standard memory speed, reasoning code at high effort, all other throughput cases with thinking disabled, dSpark speculative decoding enabled for every decode cell.
Each report table is generated from the raw bench output in the v7 package by `scripts/render-ds41-v7-quant-reports.py`; an em dash means no usable qualifying result is available in the selected package, not necessarily that a measurement was never attempted. Best prefill requires a completed, passing campaign; incomplete matrices are provisional.

## Configurations

- **1x RTX PRO 6000 + 4x Spark** — 4 full-width TP1 layers resident on the card; layers 4-39 on the Sparks.
- **2x RTX PRO 6000 + 4x Spark** — 20 TP2 layers resident across the pair; layers 20-39 on the Sparks.

## Headline

Tokens/s.

| Measurement | 1x RTX PRO 6000 + 4x Spark | 2x RTX PRO 6000 + 4x Spark | Change |
|---|---:|---:|---:|
| Best prefill | — | 7,432 | — |
| Counting decode | 113.32 | 152.25 | +34.4% |
| Weighted decode | 66.60 | 80.12 | +20.3% |
| C1 code decode | 88.77 | 109.20 | +23.0% |

## Decode completion checks

- **1x RTX PRO 6000 + 4x Spark**: 30/30 sample checks explicitly passed; 0 failed; 0 unknown/unreported. These completion checks do not establish a full benchmark pass.
- **2x RTX PRO 6000 + 4x Spark**: 30/30 sample checks explicitly passed; 0 failed; 0 unknown/unreported. These completion checks do not establish a full benchmark pass.

## Content-type decode

Median tokens/s.

| Case | 1x RTX PRO 6000 + 4x Spark | 2x RTX PRO 6000 + 4x Spark |
|---|---:|---:|
| Code | 88.77 | 109.20 |
| Code with reasoning | 72.83 | 93.50 |
| Math | 84.15 | 105.48 |
| Fable | 42.27 | 45.74 |
| Hello | 54.48 | 57.48 |
| Topic | 52.02 | 59.55 |
| Natural JSON | 83.75 | 98.55 |
| Schema JSON | 83.96 | 113.19 |
| Multilingual | 53.43 | 64.43 |
| Counting 1-200 | 113.32 | 152.25 |

## Prefill

Median effective tokens/s after shape warmup, by retained context and added tokens.

**1x RTX PRO 6000 + 4x Spark**

_Completed, passing prefill status has not been established. Any available partial measurements below are provisional and excluded from Best prefill._

_No usable prefill cells available in the selected package._

**2x RTX PRO 6000 + 4x Spark**

| Retained | +1K | +2K | +4K | +8K | +16K | +32K |
|---|---:|---:|---:|---:|---:|---:|
| 0K | 2,994 | 4,477 | 6,183 | 6,994 | 7,332 | 7,432 |
| 32K | 2,635 | 3,962 | 5,306 | 5,945 | 6,362 | 6,495 |
| 64K | 2,398 | 3,595 | 4,746 | 5,356 | 5,777 | 5,906 |
| 128K | 1,967 | 3,004 | 3,952 | 4,550 | 4,936 | 5,080 |
| 256K | 1,413 | 2,225 | 2,926 | 3,444 | 3,785 | 3,935 |

## Outstanding measurements and qualification

- 1x best prefill (campaign interrupted by a service restart)
- 1x and 2x tool-call evaluation runs
- 1x RTX PRO 6000 + 4x Spark full prefill: no completed, passing campaign result available; any partial results are provisional
