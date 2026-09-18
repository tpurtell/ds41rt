# DS41RT v7 performance - NVIDIA DeepSeek-V4.1-Flash-NVFP4 (W4A4)

Checkpoint `nvidia/DeepSeek-V4.1-Flash-NVFP4`. Weights are ModelOpt NVFP4 (W4A4): E2M1 payload with E4M3 K16 block scales, activations quantized in-kernel from BF16 rows.

The release protocol specifies **400 W per RTX card and standard memory speed, without a memory overclock**. Reported throughput cells use three samples. Recorded requests use temperature zero, high-effort thinking for reasoning code and thinking disabled for the other cases. Decode uses C1 dSpark; reasoning throughput includes reasoning and final-answer tokens.

These are new v7 quant campaigns, not the historical official-image [v6 measurements](release-v6-performance.md) shown below the headline in the [README](../README.md#performance). V5 EXL3 results use a different checkpoint and are not substituted here.

**Provenance.** Raw records are under `~/.cache/ds41rt-v7-package/performance/`; the filenames and SHA-256 digests below identify the selected inputs. They preserve request controls, timestamps, corpus/tokenizer hashes and individual samples, but their `model` is an API alias, not proof of the quant snapshot. They do not bind each campaign to engine/SparkInfer revisions, release-image hashes or power/clock telemetry. Those missing bindings remain owed; the protocol above is not a claim that every hardware control is independently verified by these JSON files. V7 release Docker images have not been built or published.

Decode cells are medians of `observed_decode_tokens_per_second` by case; weighted decode is `median_weighted_observed_decode_tokens_per_second`: the median of repeat-level weighted token/time ratios, not an average of the case medians (nine categories; natural/schema JSON weight 0.5 each, other categories 1, counting excluded). Best prefill is the maximum `median_effective_prefill_tokens_per_second` over a completed, passing matrix. Change is `(2x / 1x - 1) × 100`, calculated before display rounding.
Each report table is generated from the raw bench output in the v7 package by `scripts/render-ds41-v7-quant-reports.py`; an em dash means no usable qualifying result is available in the selected package, not necessarily that a measurement was never attempted. Best prefill requires a completed, passing campaign; incomplete matrices are provisional.

- `single-nvfp4-dspark.json` — SHA-256 `75268bb542ccf2434a057d02d87bba1b535606fdc73f203f74a6b2028c0c22d5`
- `single-nvfp4-prefill.json` — SHA-256 `0991246754392b6178b8b8693ba99093998f9b3bbdeedf69cfdc126f503b5983`
- `dual-nvfp4-dspark.json` — SHA-256 `061f6c5a6a6370b37e92f17f426c09c3139714b0480b0aa1cd76ec762208a222`
- `dual-nvfp4-prefill.json` — SHA-256 `b12117c59be026db1c320fcd12c5461cda2e99626d29533bd0cdb476afbc51bd`

## Configurations

- **1x RTX PRO 6000 + 4x Spark** — 4 full-width TP1 layers resident on the card; layers 4-39 on the Sparks.
- **2x RTX PRO 6000 + 4x Spark** — 20 TP2 layers resident across the pair; layers 20-39 on the Sparks.

## Headlines

Tokens/s. Prefill is the best cell median; decode is C1 dSpark. Weighted decode excludes counting. Change compares the two configurations defined above.

| Measurement | 1 RTX | 2 RTX | Change |
|---|---:|---:|---:|
| Best prefill | 4,131 | 7,432 | +79.9% |
| Counting decode | 113.32 | 152.25 | +34.4% |
| Weighted decode | 66.60 | 80.12 | +20.3% |
| C1 code decode | 88.77 | 109.20 | +23.0% |

## Content-type decode

Median C1 dSpark tokens/s; three samples per case. Schema JSON is grammar-constrained. See Quality and evaluation below for failed completion checks.

| Case | 1 RTX dSpark | 2 RTX dSpark |
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
| Counting 1–200 | 113.32 | 152.25 |

## Prefill

Median effective tokens/s: uncached suffix tokens divided by client time to first content, not isolated GPU prefill time. Completed matrices contain 30 cells, each with one excluded shape warmup and three measured samples with verified parent reuse. K = 1,024 tokens.

**1x RTX PRO 6000 + 4x Spark**

| Retained base | +1K | +2K | +4K | +8K | +16K | +32K |
|---|---:|---:|---:|---:|---:|---:|
| 0K | 1,688 | 2,208 | 3,418 | 3,832 | 4,016 | 4,131 |

**2x RTX PRO 6000 + 4x Spark**

| Retained base | +1K | +2K | +4K | +8K | +16K | +32K |
|---|---:|---:|---:|---:|---:|---:|
| 0K | 2,994 | 4,477 | 6,183 | 6,994 | 7,332 | 7,432 |
| 32K | 2,635 | 3,962 | 5,306 | 5,945 | 6,362 | 6,495 |
| 64K | 2,398 | 3,595 | 4,746 | 5,356 | 5,777 | 5,906 |
| 128K | 1,967 | 3,004 | 3,952 | 4,550 | 4,936 | 5,080 |
| 256K | 1,413 | 2,225 | 2,926 | 3,444 | 3,785 | 3,935 |

## Quality and evaluation

**Decode completion checks.** These check response completion and limited objective structure, not general prose quality, tool calling or fixed-history quant agreement. Failed samples remain in throughput and in the denominators.

- **1x RTX PRO 6000 + 4x Spark**: 30/30 sample checks explicitly passed; 0 failed; 0 unknown/unreported. These completion checks do not establish a full benchmark pass.
- **2x RTX PRO 6000 + 4x Spark**: 30/30 sample checks explicitly passed; 0 failed; 0 unknown/unreported. These completion checks do not establish a full benchmark pass.

Tool-call evaluation, adaptive draft acceptance and fixed-history quant agreement have no completed results published here. Historical official or v5 EXL3 quality scores are not evidence for these new quants.

The 1x prefill campaign was interrupted by a service restart ([campaign record](release-v7-plan.md)). Its raw file retains partial samples, but no finalized cell summaries or completed, passing matrix; no best is estimated.

## Outstanding measurements and qualification

- 1x and 2x completed tool-call evaluation runs
- Fresh target-only decode; retained-context decode including the separate 2K control; counting/code/topic concurrency scaling and mixed-traffic sweeps: not qualified here to the v5/v6 scope
- Per-layout startup, memory and cache-capacity qualification; adaptive draft acceptance and fixed-history quant agreement: not replaced by historical official-image or v5 EXL3 results
- Per-campaign engine/SparkInfer revisions, quant snapshot and binary/launch identity, KV/PLE and TP2 controls, plus power/clock evidence (including stock-memory settings)
- Producer-built v7 release Docker images and publication: neither completed
