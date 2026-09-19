# DS41RT v7 performance - NVIDIA DeepSeek-V4.1-Flash-NVFP4 (W4A4)

Checkpoint `nvidia/DeepSeek-V4.1-Flash-NVFP4`. Weights are ModelOpt NVFP4 (W4A4): E2M1 payload with E4M3 K16 block scales, activations quantized in-kernel from BF16 rows.

The release protocol specifies **400 W per RTX card and standard memory speed, without a memory overclock**. Reported throughput cells use three samples. Recorded requests use temperature zero, high-effort thinking for reasoning code and thinking disabled for the other cases. Decode uses C1 dSpark; reasoning throughput includes reasoning and final-answer tokens.

These are new v7 quant campaigns, not the historical official-image [v6 measurements](release-v6-performance.md) shown below the headline in the [README](../README.md#performance). V5 EXL3 results use a different checkpoint and are not substituted here.

**Provenance.** Raw records are under `~/.cache/ds41rt-v7-published/performance/`; the filenames and SHA-256 digests below identify the selected inputs. The campaigns were served from the published release images (`ghcr.io/tpurtell/ds41rt-coordinator:v7` sha256:d85608bb, `ghcr.io/tpurtell/ds41rt-spark-expert:v7` sha256:aa477ff1, both engine revision `0107d01e3d35d22b1dbc5de70c4e1a32d32d165f`), and each decode campaign reuses the nonce seed its recorded predecessor used so the two are directly comparable. The records preserve request controls, timestamps, corpus/tokenizer hashes and individual samples, but their `model` is an API alias, not proof of the quant snapshot. They do not bind each campaign to engine/SparkInfer revisions or power/clock telemetry. Those missing bindings remain owed; the protocol above is not a claim that every hardware control is independently verified by these JSON files.

Decode cells are medians of `observed_decode_tokens_per_second` by case; weighted decode is `median_weighted_observed_decode_tokens_per_second`: the median of repeat-level weighted token/time ratios, not an average of the case medians (nine categories; natural/schema JSON weight 0.5 each, other categories 1, counting excluded). Best prefill is the maximum `median_effective_prefill_tokens_per_second` over a completed, passing matrix. Change is `(2x / 1x - 1) × 100`, calculated before display rounding.
Each report table is generated from the raw bench output in the v7 package by `scripts/render-ds41-v7-quant-reports.py`; an em dash means no usable qualifying result is available in the selected package, not necessarily that a measurement was never attempted. Best prefill requires a completed, passing campaign; incomplete matrices are provisional.

- `single-nvfp4-dspark.json` — SHA-256 `1a775b4475dfb733ac70dc4e3d4fab48690e6e399775feb36287de101cb03407`
- `single-nvfp4-prefill.json` — SHA-256 `069de6988a59ee87cdf1619525aad384e0cd50c34a7dbe7cb77ca9b3dc0a6a60`
- `dual-nvfp4-dspark.json` — SHA-256 `7e93a6a58854e683074a0e0c4d5c368c599be8cb23a249d8f1a631db6555fe72`
- `dual-nvfp4-prefill.json` — SHA-256 `fd762e7e79602b79f4e20bab2702a309a8ea629da31b57e2dc5c97646d028f88`

## Configurations

- **1x RTX PRO 6000 + 4x Spark** — 4 full-width TP1 layers resident on the card; layers 4-39 on the Sparks.
- **2x RTX PRO 6000 + 4x Spark** — 20 TP2 layers resident across the pair; layers 20-39 on the Sparks.

## Headlines

Tokens/s. Prefill is the best cell median; decode is C1 dSpark. Weighted decode excludes counting. Change compares the two configurations defined above.

| Measurement | 1 RTX | 2 RTX | Change |
|---|---:|---:|---:|
| Best prefill | 4,141 | 7,370 | +78.0% |
| Counting decode | 114.06 | 149.98 | +31.5% |
| Weighted decode | 66.81 | 78.57 | +17.6% |
| C1 code decode | 86.58 | 106.38 | +22.9% |

## Content-type decode

Median C1 dSpark tokens/s; three samples per case. Schema JSON is grammar-constrained. See Quality and evaluation below for failed completion checks.

| Case | 1 RTX dSpark | 2 RTX dSpark |
|---|---:|---:|
| Code | 86.58 | 106.38 |
| Code with reasoning | 73.03 | 91.92 |
| Math | 83.64 | 103.38 |
| Fable | 42.72 | 44.76 |
| Hello | 53.54 | 56.76 |
| Topic | 52.41 | 58.30 |
| Natural JSON | 84.84 | 96.65 |
| Schema JSON | 84.62 | 112.30 |
| Multilingual | 53.71 | 63.76 |
| Counting 1–200 | 114.06 | 149.98 |

## Prefill

Median effective tokens/s: uncached suffix tokens divided by client time to first content, not isolated GPU prefill time. Completed matrices contain 30 cells, each with one excluded shape warmup and three measured samples with verified parent reuse. K = 1,024 tokens.

**1x RTX PRO 6000 + 4x Spark**

| Retained base | +1K | +2K | +4K | +8K | +16K | +32K |
|---|---:|---:|---:|---:|---:|---:|
| 0K | 1,659 | 2,241 | 3,488 | 3,822 | 4,022 | 4,141 |
| 32K | 1,514 | 2,104 | 3,257 | 3,593 | 3,852 | 3,957 |
| 64K | 1,415 | 2,007 | 3,058 | 3,427 | 3,704 | 3,832 |
| 128K | 1,259 | 1,819 | 2,779 | 3,152 | 3,426 | 3,506 |
| 256K | 1,030 | 1,515 | 2,302 | 2,670 | 2,941 | 3,066 |

**2x RTX PRO 6000 + 4x Spark**

| Retained base | +1K | +2K | +4K | +8K | +16K | +32K |
|---|---:|---:|---:|---:|---:|---:|
| 0K | 2,994 | 4,502 | 6,085 | 6,896 | 7,255 | 7,370 |
| 32K | 2,635 | 3,997 | 4,714 | 5,528 | 6,106 | 6,354 |
| 64K | 2,402 | 3,622 | 4,327 | 5,046 | 5,591 | 5,820 |
| 128K | 1,974 | 3,033 | 3,625 | 4,321 | 4,796 | 4,999 |
| 256K | 1,425 | 2,275 | 2,756 | 3,327 | 3,694 | 3,879 |

## Quality and evaluation

**Decode completion checks.** These check response completion and limited objective structure, not general prose quality, tool calling or fixed-history quant agreement. Failed samples remain in throughput and in the denominators.

- **1x RTX PRO 6000 + 4x Spark**: 30/30 sample checks explicitly passed; 0 failed; 0 unknown/unreported. These completion checks do not establish a full benchmark pass.
- **2x RTX PRO 6000 + 4x Spark**: 30/30 sample checks explicitly passed; 0 failed; 0 unknown/unreported. These completion checks do not establish a full benchmark pass.

Tool-call evaluation, adaptive draft acceptance and fixed-history quant agreement have no completed results published here. Historical official or v5 EXL3 quality scores are not evidence for these new quants.

## Outstanding measurements and qualification

- 1x and 2x completed tool-call evaluation runs
- Fresh target-only decode; retained-context decode including the separate 2K control; counting/code/topic concurrency scaling and mixed-traffic sweeps: not qualified here to the v5/v6 scope
- Per-layout startup, memory and cache-capacity qualification; adaptive draft acceptance and fixed-history quant agreement: not replaced by historical official-image or v5 EXL3 results
- Per-campaign engine/SparkInfer revisions, quant snapshot and binary/launch identity, KV/PLE and TP2 controls, plus power/clock evidence (including stock-memory settings)
- v7 release images are built and published (ghcr.io/tpurtell/ds41rt-coordinator:v7 sha256:d85608bb, ghcr.io/tpurtell/ds41rt-spark-expert:v7 sha256:aa477ff1); physical RTX 5090 validation remains owed
