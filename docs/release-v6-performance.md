# DS41RT v6 performance

These are the native full-checkpoint measurements used to qualify v6. Historical EXL3 results are linked below and are not v6 measurements.

Engine `619be005af10c506df3c261d99e9f282fdc846ae`; SparkInfer `63e2140e4a32a977faa777c172b86679344fdc6a`; official checkpoint `deepseek-ai/DeepSeek-V4.1-Flash@dba1be0a40aa45a94ad051997016db3960a90277`. The [structured summary](release-v6-performance.json) records commands, controls, artifact hashes, deployment capacity, and telemetry.

RTX measurements use **400 W per card and standard memory speed, without a memory overclock**. Each performance cell has three samples. Reasoning code uses high-effort thinking; other throughput cases disable thinking. All new TP2 switches are off.

**Headlines.** Tokens/s. Changes compare two RTX cards with one. Counting is outside the weighted score.

| Measurement | 1 RTX | 2 RTX | Change |
|---|---:|---:|---:|
| Best median prefill | 7,823.90 | 8,355.22 | +6.8% |
| Counting target-only decode | 49.36 | 49.53 | +0.3% |
| Counting dSpark decode | 161.58 | 221.64 | +37.2% |
| Weighted nine-category target-only decode | 47.23 | 49.95 | +5.8% |
| Weighted nine-category dSpark decode | 92.00 | 109.44 | +19.0% |
| C16 code aggregate | 1,069.16 | 1,195.37 | +11.8% |
| C16 topic aggregate | 601.90 | 622.81 | +3.5% |
| C16 counting aggregate | 1,267.07 | 1,508.81 | +19.1% |
| C16 mixed aggregate | 191.75 | 284.69 | +48.5% |

**Content-type decode.** Median tokens/s. Official Flash values are the historical one-shot reference, including its prior fable wording; they were not rerun.

| Case | 1 RTX target | 1 RTX dSpark | 2 RTX target | 2 RTX dSpark | Historical official Flash |
|---|---:|---:|---:|---:|---:|
| Code | 49.08 | 130.41 | 51.67 | 155.70 | 345.90 |
| Code with reasoning | 47.51 | 100.27 | 50.74 | 126.08 | — |
| Math | 45.77 | 134.22 | 48.45 | 134.13 | 285.33 |
| Fable | 44.72 | 56.49 | 48.25 | 67.88 | 123.63 |
| Hello | 44.75 | 61.51 | 48.39 | 98.04 | 141.10 |
| Topic | 46.28 | 73.57 | 48.59 | 87.10 | 169.24 |
| Natural JSON | 47.37 | 103.83 | 50.07 | 133.90 | 175.33 |
| Schema JSON | 48.50 | 105.79 | 50.84 | 102.92 | HTTP 400 |
| Multilingual | 45.84 | 74.10 | 49.50 | 82.97 | 183.61 |
| Counting 1–200 | 49.36 | 161.58 | 49.53 | 221.64 | 427.29 |

**1 RTX prefill matrix.** Median effective tokens/s after shape warmup and verified parent reuse.

| Retained base | +1K | +2K | +4K | +8K | +16K | +32K |
|---|---:|---:|---:|---:|---:|---:|
| 0K | 3,005 | 4,031 | 6,950 | 7,493 | 7,801 | 7,824 |
| 32K | 2,463 | 3,326 | 5,379 | 6,140 | 6,598 | 6,863 |
| 64K | 2,242 | 3,103 | 4,961 | 5,665 | 6,151 | 6,361 |
| 128K | 1,928 | 2,718 | 4,205 | 4,914 | 5,343 | 5,530 |
| 256K | 1,463 | 2,122 | 3,195 | 3,797 | 4,156 | 4,306 |

**2 RTX prefill matrix.** Median effective tokens/s after shape warmup and verified parent reuse.

| Retained base | +1K | +2K | +4K | +8K | +16K | +32K |
|---|---:|---:|---:|---:|---:|---:|
| 0K | 4,725 | 6,369 | 7,825 | 8,185 | 8,355 | 8,290 |
| 32K | 3,683 | 5,153 | 6,373 | 6,930 | 7,207 | 7,251 |
| 64K | 3,221 | 4,626 | 5,673 | 6,226 | 6,507 | 6,567 |
| 128K | 2,588 | 3,678 | 4,655 | 5,202 | 5,486 | 5,556 |
| 256K | 1,770 | 2,697 | 3,336 | 3,847 | 4,126 | 4,229 |

**Decode over retained context.** Weighted nine-category dSpark tokens/s with verified prefix reuse.

| Retained base | 1 RTX | 2 RTX | Change |
|---|---:|---:|---:|
| 0K | 86.85 | 104.21 | +20.0% |
| 2K | 85.97 | 103.81 | +20.7% |
| 32K | 83.28 | 100.72 | +20.9% |
| 64K | 82.64 | 102.15 | +23.6% |
| 128K | 86.67 | 98.40 | +13.5% |
| 256K | 80.02 | 92.78 | +15.9% |

**Concurrency scaling.** Median aggregate tokens/s from earliest first output to final completion, including admission gaps.

| Concurrency | 1 RTX counting | 2 RTX counting | 1 RTX code | 2 RTX code | 1 RTX topic | 2 RTX topic |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 159.32 | 211.83 | 137.43 | 171.96 | 74.63 | 96.94 |
| 2 | 252.73 | 314.71 | 218.48 | 279.69 | 125.39 | 163.79 |
| 4 | 454.48 | 586.05 | 384.41 | 476.04 | 225.99 | 260.73 |
| 8 | 736.37 | 973.21 | 625.30 | 749.73 | 333.06 | 447.59 |
| 16 | 1,267.07 | 1,508.81 | 1,069.16 | 1,195.37 | 601.90 | 622.81 |

**Mixed traffic.** Code/fable/topic mix; aggregate tokens/s median and range across three sweeps.

| Concurrency | 1 RTX | 2 RTX |
|---|---:|---:|
| 1 | 104.01 (97.36–130.92) | 156.41 (144.89–160.71) |
| 2 | 107.10 (100.71–114.91) | 140.72 (136.36–141.02) |
| 4 | 139.42 (135.51–161.87) | 184.62 (171.39–186.38) |
| 8 | 147.90 (145.71–164.74) | 211.97 (206.42–215.89) |
| 16 | 191.75 (183.09–193.95) | 284.69 (281.08–296.28) |

**Deployment and cache capacity.** RAM bytes include staging and snapshot overhead. Combined tokens count each logical source once; active requests must fit the GPU pool.

| Configuration | RTX expert layers | Spark resident / active layers / budget | Global FP4 source pool | RAM cache | Combined usable tokens | Prompt / turn retention |
|---|---:|---:|---:|---:|---:|---:|
| 1 RTX | 0–4, TP1 | 40 / 35 / 100 GiB each | 16.701 GB / 18,736,128 usable + 28,672 private-tail tokens | 3.25 GiB pinned / 2,235,904 logical tokens | 20,972,032 | 20 / 20 |
| 2 RTX | 0–19, TP2 | 20 / 20 / 100 GiB each | 13.091 GB / 14,680,064 usable + 28,672 private-tail tokens | 6.75 GiB pinned / 6,291,968 logical tokens | 20,972,032 | 20 / 20 |

**Startup.** Standard launcher to API readiness, including orchestration; one observation per layout.

| Configuration | Seconds |
|---|---:|
| 1 RTX | 58.63 |
| 2 RTX | 31.71 |

**Memory after readiness.** GPU allocations include weights, KV and workspaces; later graph capture can consume additional memory.

| Configuration | Logical RTX | Loaded MiB | Free MiB | Planned runtime reserve MiB |
|---|---:|---:|---:|---:|
| 1 RTX | 0 | 94,590.00 | 2,661.00 | 2,048.00 |
| 2 RTX | 0 | 95,132.00 | 2,119.00 | 800.00 |
| 2 RTX | 1 | 95,738.00 | 1,510.00 | 800.00 |

**Historical native draft acceptance.** V5 measurements, not rerun for v6: C1, three requests per content type. Accepted/verified percentage and mean emitted tokens per nonterminal cycle in parentheses. Schema JSON is grammar-constrained; reasoning code includes reasoning and final output. Adaptive selection omits unverified drafts, so these are serving rates, not fixed-history agreement.

| Content | Historical 1 RTX | Historical 2 RTX |
|---|---:|---:|
| Code | 90.67% (5.26) | 74.89% (5.82) |
| Code with reasoning | 78.90% (3.74) | 68.83% (3.92) |
| Math | 90.97% (5.55) | 69.25% (5.21) |
| Fable | 52.86% (1.79) | 40.70% (1.77) |
| Hello | 65.12% (2.75) | 35.33% (2.43) |
| Topic | 61.93% (2.47) | 49.38% (2.47) |
| Natural JSON | 82.14% (4.17) | 62.14% (4.00) |
| Schema JSON | 75.40% (4.65) | 54.76% (4.29) |
| Multilingual | 57.33% (2.22) | 50.37% (2.38) |

**Historical native tool calling.** The three v4 full-checkpoint campaigns retained in v5; not rerun for v6. High-effort thinking was enabled, and failures remain in the scores.

| Run | Basic | Hard | Total |
|---|---:|---:|---:|
| 2026-09-16T04-12-34.961137Z_2214ecfb | 124/138 | 31/38 | 155/176 |
| 2026-09-16T04-17-40.551606Z_43911d6a | 126/138 | 34/38 | 160/176 |
| 2026-09-16T04-22-39.602726Z_81eaa393 | 123/138 | 33/38 | 156/176 |

Historical EXL3 performance, acceptance and quantization analysis are preserved in the [v5 performance report](https://github.com/tpurtell/ds41rt/blob/v5/docs/release-v5-performance.md); they are not v6 measurements.
