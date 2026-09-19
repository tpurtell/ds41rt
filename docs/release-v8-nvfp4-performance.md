# DS41RT v8 performance - NVIDIA DeepSeek-V4.1-Flash-NVFP4 (W4A4)

Checkpoint `nvidia/DeepSeek-V4.1-Flash-NVFP4`. Weights are ModelOpt NVFP4 (W4A4): E2M1 payload with E4M3 K16 block scales, activations quantized in-kernel from BF16 rows.

The release protocol specifies **400 W per RTX card and standard memory speed, without a memory overclock**. Reported throughput cells use three samples. Recorded requests use temperature zero, high-effort thinking for reasoning code and thinking disabled for the other cases. Decode uses C1 dSpark; reasoning throughput includes reasoning and final-answer tokens.

These are new v8 quant campaigns, not the historical official-image [v6 measurements](release-v6-performance.md) shown below the headline in the [README](../README.md#performance). V5 EXL3 results use a different checkpoint and are not substituted here.

**Provenance.** Raw records are under `~/.cache/ds41rt-v7-published/performance/`; the filenames and SHA-256 digests below identify the selected inputs. The campaigns were served from the published release images (`ghcr.io/tpurtell/ds41rt-coordinator:v7` sha256:d85608bb, `ghcr.io/tpurtell/ds41rt-spark-expert:v7` sha256:aa477ff1, both engine revision `0107d01e3d35d22b1dbc5de70c4e1a32d32d165f`), and each decode campaign reuses the nonce seed its recorded predecessor used so the two are directly comparable. The records preserve request controls, timestamps, corpus/tokenizer hashes and individual samples, but their `model` is an API alias, not proof of the quant snapshot. They do not bind each campaign to engine/SparkInfer revisions or power/clock telemetry. Those missing bindings remain owed; the protocol above is not a claim that every hardware control is independently verified by these JSON files.

Decode cells are medians of `observed_decode_tokens_per_second` by case; weighted decode is `median_weighted_observed_decode_tokens_per_second`: the median of repeat-level weighted token/time ratios, not an average of the case medians (nine categories; natural/schema JSON weight 0.5 each, other categories 1, counting excluded). Best prefill is the maximum `median_effective_prefill_tokens_per_second` over a completed, passing matrix. Change is `(2x / 1x - 1) × 100`, calculated before display rounding.
Each report table is generated from the raw bench output in the v8 package by `scripts/render-ds41-v7-quant-reports.py`; an em dash means no usable qualifying result is available in the selected package, not necessarily that a measurement was never attempted. Best prefill requires a completed, passing campaign; incomplete matrices are provisional.

- `single-nvfp4-dspark.json` — SHA-256 `46534507b1169673cbf44bc134f178f1047c09d6f12ff1a900beac1f3aab23a8`
- `single-nvfp4-prefill.json` — SHA-256 `febc290a03cb44c84ba72434b502958317813cc6f5f29f73bb2556cca5a7a52d`
- `dual-nvfp4-dspark.json` — SHA-256 `77b7e60a70fc4d5b5f8b5d651938e83664673be2946d698226d61be8d1e226a6`
- `dual-nvfp4-prefill.json` — SHA-256 `be735bb88272efef854b0005c6fd45ea616c975663291fb1e875fe1206e09ca1`

## Configurations

- **1x RTX PRO 6000 + 4x Spark** — 4 full-width TP1 layers resident on the card; layers 4-39 on the Sparks.
- **2x RTX PRO 6000 + 4x Spark** — 20 TP2 layers resident across the pair; layers 20-39 on the Sparks.

## Headlines

Tokens/s. Prefill is the best cell median; decode is C1 dSpark. Weighted decode excludes counting. Change compares the two configurations defined above.

| Measurement | 1 RTX | 2 RTX | Change |
|---|---:|---:|---:|
| Best prefill | 5,237 | 7,371 | +40.7% |
| Counting decode | 144.66 | 192.42 | +33.0% |
| Weighted decode | 81.74 | 100.78 | +23.3% |
| C1 code decode | 112.55 | 145.96 | +29.7% |

## Content-type decode

Median C1 dSpark tokens/s; three samples per case. Schema JSON is grammar-constrained. See Quality and evaluation below for failed completion checks.

| Case | 1 RTX dSpark | 2 RTX dSpark |
|---|---:|---:|
| Code | 112.55 | 145.96 |
| Code with reasoning | 96.82 | 113.78 |
| Math | 105.44 | 114.70 |
| Fable | 53.76 | 59.35 |
| Hello | 68.35 | 70.73 |
| Topic | 65.47 | 75.82 |
| Natural JSON | 105.04 | 127.19 |
| Schema JSON | 104.41 | 124.82 |
| Multilingual | 71.82 | 79.28 |
| Counting 1–200 | 144.66 | 192.42 |

## Prefill

Median effective tokens/s: uncached suffix tokens divided by client time to first content, not isolated GPU prefill time. Completed matrices contain 30 cells, each with one excluded shape warmup and three measured samples with verified parent reuse. K = 1,024 tokens.

**1x RTX PRO 6000 + 4x Spark**

| Retained base | +1K | +2K | +4K | +8K | +16K | +32K |
|---|---:|---:|---:|---:|---:|---:|
| 0K | 2,044 | 2,590 | 4,520 | 4,899 | 5,146 | 5,237 |
| 32K | 1,781 | 2,488 | 3,771 | 4,251 | 4,717 | 4,874 |
| 64K | 1,674 | 2,340 | 3,559 | 4,123 | 4,503 | 4,651 |
| 128K | 1,473 | 2,126 | 3,279 | 3,743 | 4,094 | 4,247 |
| 256K | 1,155 | 1,728 | 2,615 | 3,073 | 3,401 | 3,542 |

**2x RTX PRO 6000 + 4x Spark**

| Retained base | +1K | +2K | +4K | +8K | +16K | +32K |
|---|---:|---:|---:|---:|---:|---:|
| 0K | 3,377 | 4,898 | 6,434 | 7,051 | 7,324 | 7,371 |
| 32K | 2,877 | 4,269 | 4,910 | 5,666 | 6,200 | 6,396 |
| 64K | 2,622 | 3,855 | 4,443 | 5,173 | 5,656 | 5,839 |
| 128K | 2,149 | 3,226 | 3,765 | 4,414 | 4,845 | 5,024 |
| 256K | 1,519 | 2,312 | 2,809 | 3,371 | 3,731 | 3,898 |

## Quality and evaluation

**Decode completion checks.** These check response completion and limited objective structure, not general prose quality, tool calling or fixed-history quant agreement. Failed samples remain in throughput and in the denominators.

- **1x RTX PRO 6000 + 4x Spark**: 30/30 sample checks explicitly passed; 0 failed; 0 unknown/unreported. These completion checks do not establish a full benchmark pass.
- **2x RTX PRO 6000 + 4x Spark**: 30/30 sample checks explicitly passed; 0 failed; 0 unknown/unreported. These completion checks do not establish a full benchmark pass.

**Tool-call evaluation.** High-effort thinking enabled. Scores move with sampling, so every run is listed rather than averaged; per-scenario outcomes stay in the raw records.

| Configuration | Run | Basic | Hard | Total |
|---|---|---:|---:|---:|
| 1x RTX PRO 6000 + 4x Spark | 2026-09-19T09-42-31.434133Z_5a3ea8fe | 124/138 | 32/38 | 156/176 |
| 1x RTX PRO 6000 + 4x Spark | 2026-09-19T09-50-48.855152Z_e329b0e4 | 125/138 | 32/38 | 157/176 |
| 1x RTX PRO 6000 + 4x Spark | 2026-09-19T09-59-01.715734Z_0caa364e | 119/138 | 33/38 | 152/176 |
| 2x RTX PRO 6000 + 4x Spark | 2026-09-19T14-02-19.781567Z_38afa854 | 122/138 | 34/38 | 156/176 |
| 2x RTX PRO 6000 + 4x Spark | 2026-09-19T14-09-30.217177Z_71eb5879 | 122/138 | 35/38 | 157/176 |
| 2x RTX PRO 6000 + 4x Spark | 2026-09-19T14-16-19.589120Z_3c1f7edc | 123/138 | 34/38 | 157/176 |


## Startup and memory

Container start to readiness, including orchestration; memory is sampled once after readiness and excludes later graph capture. `Coordinator` is the API-ready moment, `full` also waits for the last Spark expert.

| Configuration | Coordinator s | Full s | GPU used MiB |
|---|---:|---:|---|
| 1x RTX PRO 6000 + 4x Spark | 9.0 | 171.5 | 0: 89,724, 1: 12 |
| 2x RTX PRO 6000 + 4x Spark | 33.0 | 33.0 | 0: 94,128, 1: 96,482 |

## Decode over retained context

Weighted nine-category dSpark tokens/s with verified prefix reuse.

| Retained base | 1x | 2x | Change |
|---|---:|---:|---:|
| 0K | 83 | 103 | +25.0% |
| 2K | 81 | 101 | +25.3% |
| 32K | 77 | 100 | +30.0% |
| 64K | 80 | 94 | +17.0% |
| 128K | 76 | 95 | +24.7% |
| 256K | 72 | 85 | +17.9% |

## Concurrency scaling

Median aggregate tokens/s from earliest first output to final completion, including admission gaps.

| Concurrency | 1x counting | 1x code | 1x topic | 2x counting | 2x code | 2x topic |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 142 | 114 | 62 | 193 | 150 | 81 |
| 2 | 207 | 163 | 89 | 270 | 227 | 112 |
| 4 | 332 | 221 | 139 | 410 | 321 | 171 |
| 8 | 500 | 305 | 179 | 635 | 472 | 225 |
| 16 | 861 | 449 | 226 | 977 | 641 | 329 |

## Mixed traffic

Code/fable/topic mix; aggregate tokens/s by concurrency level.

| Concurrency | 1x | 2x | Change |
|---|---:|---:|---:|
| 4 | 107 | 143 | +34.0% |
| 16 | 143 | 209 | +46.9% |

## Target-only decode

dSpark disabled: median C1 tokens/s per case.

| Case | 1x | 2x | Change |
|---|---:|---:|---:|
| Code | 46.87 | 49.85 | +6.4% |
| Counting 1–200 | 47.01 | 50.04 | +6.5% |
## Outstanding measurements and qualification

- Full battery (tool-call evaluation, retained-context decode with its 2K control, counting/code/topic concurrency scaling, mixed traffic and target-only decode) is deferred pending the W4A4 optimization: that work changes the kernel family and the activation wire, so measuring the battery now would have to be redone
- Cache-capacity qualification; adaptive draft acceptance and fixed-history quant agreement: not replaced by historical official-image or v5 EXL3 results
- Per-campaign engine/SparkInfer revisions, quant snapshot and binary/launch identity, KV/PLE and TP2 controls, plus power/clock evidence (including stock-memory settings)
- v8 release images are built and published (ghcr.io/tpurtell/ds41rt-coordinator:v8 sha256:08c2d6df, ghcr.io/tpurtell/ds41rt-spark-expert:v8 sha256:99079839); physical RTX 5090 validation remains owed
