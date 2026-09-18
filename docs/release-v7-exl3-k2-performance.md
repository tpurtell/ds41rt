# DS41RT v7 performance - diffbot DeepSeek-V4.1-Flash-EXL3 (2.0 bpw, K=2)

Checkpoint `diffbot/DeepSeek-V4.1-Flash-EXL3-2.0bpw-2x-RTX-PRO-6000`. Uniform K=2 trellis with a two-tier kernel family; the paired 3.25 bpw publication uses the second tier.

The target v7 release protocol uses three samples per cell, 400 W per card, standard memory speed, reasoning code at high effort, all other throughput cases with thinking disabled, dSpark speculative decoding enabled for every decode cell.
Each report table is generated from the raw bench output in the v7 package by `scripts/render-ds41-v7-quant-reports.py`; an em dash means no usable qualifying result is available in the selected package, not necessarily that a measurement was never attempted. Best prefill requires a completed, passing campaign; incomplete matrices are provisional.

## Configurations

- **1x RTX PRO 6000 (32 GiB budget) + 2x Spark** — Compact RTX 5090 memory simulation: total device ceiling includes 2 GiB runtime headroom; layer 0 full-width on RTX and layers 1–39 as TP2 on exactly two Sparks; prefill batches capped at 256 and global KV pool 2 GiB. See [residency and qualification](release-v7-exl3-compact.md). This is not a measurement on RTX 5090 hardware.
- **2x RTX PRO 6000, no Spark** — All 40 routed-expert layers resident as TP2 on the pair; no Spark worker.

EXL3 Change compares 2x RTX PRO 6000 with no Sparks against 1x RTX PRO 6000 capped at 32 GiB plus 2x Spark, not isolated second-GPU scaling.

## Headline

Tokens/s.

| Measurement | 1x RTX PRO 6000 (32 GiB budget) + 2x Spark | 2x RTX PRO 6000, no Spark | Change |
|---|---:|---:|---:|
| Best prefill | 2,015 | 5,702 | +182.9% |
| Counting decode | 163.55 | 337.35 | +106.3% |
| Weighted decode | 88.10 | 145.10 | +64.7% |
| C1 code decode | 123.54 | 222.06 | +79.7% |

## Decode completion checks

- **1x RTX PRO 6000 (32 GiB budget) + 2x Spark**: 29/30 sample checks explicitly passed; 1 failed; 0 unknown/unreported. These completion checks do not establish a full benchmark pass.
  - code-reasoning, repeat 3: finish reason `length`; final response empty. Throughput above includes this sample; it is not a successful quality result.
- **2x RTX PRO 6000, no Spark**: 28/30 sample checks explicitly passed; 2 failed; 0 unknown/unreported. These completion checks do not establish a full benchmark pass.
  - code-reasoning, repeat 1: finish reason `length`; final response empty. Throughput above includes this sample; it is not a successful quality result.
  - code-reasoning, repeat 3: finish reason `length`; final response empty. Throughput above includes this sample; it is not a successful quality result.

## Content-type decode

Median tokens/s.

| Case | 1x RTX PRO 6000 (32 GiB budget) + 2x Spark | 2x RTX PRO 6000, no Spark |
|---|---:|---:|
| Code | 123.54 | 222.06 |
| Code with reasoning | 92.58 | 160.72 |
| Math | 118.03 | 196.75 |
| Fable | 55.62 | 78.69 |
| Hello | 68.54 | 114.88 |
| Topic | 68.49 | 98.94 |
| Natural JSON | 96.63 | 159.55 |
| Schema JSON | 101.28 | 161.40 |
| Multilingual | 71.69 | 100.93 |
| Counting 1-200 | 163.55 | 337.35 |

## Prefill

Median effective tokens/s after shape warmup, by retained context and added tokens.

**1x RTX PRO 6000 (32 GiB budget) + 2x Spark**

| Retained | +1K | +2K | +4K | +8K | +16K | +32K |
|---|---:|---:|---:|---:|---:|---:|
| 0K | 1,622 | 1,871 | 1,952 | 1,989 | 1,977 | 2,015 |
| 32K | 1,511 | 1,769 | 1,856 | 1,903 | 1,911 | 1,943 |
| 64K | 1,447 | 1,721 | 1,822 | 1,887 | 1,907 | 1,941 |
| 128K | 1,309 | 1,608 | 1,763 | 1,843 | 1,889 | 1,935 |
| 256K | 1,057 | 1,386 | 1,570 | 1,707 | 1,759 | 1,813 |

**2x RTX PRO 6000, no Spark**

| Retained | +1K | +2K | +4K | +8K | +16K | +32K |
|---|---:|---:|---:|---:|---:|---:|
| 0K | 3,567 | 4,144 | 5,636 | 5,702 | 5,671 | 5,553 |
| 32K | 3,008 | 3,656 | 4,258 | 4,655 | 4,848 | 4,921 |
| 64K | 2,643 | 3,315 | 3,919 | 4,284 | 4,501 | 4,582 |
| 128K | 2,178 | 2,829 | 3,346 | 3,751 | 3,961 | 4,051 |
| 256K | 1,527 | 2,106 | 2,564 | 2,960 | 3,196 | 3,294 |

## Outstanding measurements and qualification

- 1x and 2x tool-call evaluation runs
- RTX 5090 hardware performance (the compact profile simulates its memory budget on RTX PRO 6000)
