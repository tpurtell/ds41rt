# DS41RT v4 performance

Measured September 16, 2026 on one/two RTX PRO 6000 Workstation cards and four DGX Spark workers. The complete final-image performance campaign passed; this report preserves measured regressions and the scope of subsequent controls.

The standard launcher selects two feasible peer-connected RTX cards and falls back to one. Two-RTX mode hosts encoder expert layers 0–19 TP2 on the RTX pair; one-RTX mode uses bottom-up local placement. Both measured layouts use adaptive K5, C16 capacity, and 24 prompt/completed-turn cache entries. Throughput tests use temperature zero and thinking disabled; tool evaluation uses high-effort thinking.

Each performance cell has three measured samples. The main dSpark campaign runs direct decode, code/topic/counting concurrency, three mixed sweeps, retained-context decode, then prefill on the same serving process. The separate 2K retained-base test follows; target-only decode uses a fresh launch. Counting is a low-entropy reference; weighted code/prose workloads and mixed traffic are the useful serving comparisons.

RTX power limit: **400 W per card**, with **standard memory speed** (no memory overclock). Loaded memory clock is typically 13,365 MHz; monitored maximum is at most 14,001 MHz.

**Decode headlines.** Median tokens/s; counting is outside the weighted real-content score.

| Measurement | 1 RTX | 2 RTX | 2 RTX change |
|---|---:|---:|---:|
| Best median prefill | 7,720.75 | 8,372.36 | +8.4% |
| Counting target-only decode | 49.58 | 53.85 | +8.6% |
| Counting dSpark decode | 161.71 | 196.70 | +21.6% |
| Weighted eight-type target-only decode | 47.78 | 50.20 | +5.1% |
| Weighted eight-type dSpark decode | 78.97 | 97.79 | +23.8% |
| C16 code aggregate | 1,042.00 | 1,298.72 | +24.6% |
| C16 topic aggregate | 554.17 | 734.40 | +32.5% |
| C16 counting aggregate | 1,258.23 | 1,494.85 | +18.8% |
| C16 mixed aggregate | 198.31 | 308.19 | +55.4% |

**Change from v3.** Matched prompts and three measured samples per mode. Numerical changes may change generated text and draft acceptance.

| Measurement | v3 1 RTX | v4 1 RTX | Change | v3 2 RTX | v4 2 RTX | Change |
|---|---:|---:|---:|---:|---:|---:|
| Best median prefill | 7,878.39 | 7,720.75 | -2.0% | 8,453.54 | 8,372.36 | -1.0% |
| Counting target-only decode | 44.95 | 49.58 | +10.3% | 48.41 | 53.85 | +11.2% |
| Counting dSpark decode | 156.08 | 161.71 | +3.6% | 181.31 | 196.70 | +8.5% |
| Weighted eight-type target-only decode | 43.04 | 47.78 | +11.0% | 46.14 | 50.20 | +8.8% |
| Weighted eight-type dSpark decode | 76.72 | 78.97 | +2.9% | 79.33 | 97.79 | +23.3% |
| C16 code aggregate | 992.78 | 1,042.00 | +5.0% | 1,181.49 | 1,298.72 | +9.9% |
| C16 topic aggregate | 514.96 | 554.17 | +7.6% | 596.14 | 734.40 | +23.2% |
| C16 counting aggregate | 1,161.12 | 1,258.23 | +8.4% | 1,333.57 | 1,494.85 | +12.1% |
| C16 mixed aggregate | 196.46 | 198.31 | +0.9% | 309.06 | 308.19 | -0.3% |

**Eight content types and counting.** Local results have three samples. Official Flash is the preserved one-shot prior reference, including its prior fable wording; it was not rerun. Code, math and JSON have objective checks; open prose is unscored.

| Case | 1 RTX target | 1 RTX dSpark | 2 RTX target | 2 RTX dSpark | Official Flash |
|---|---:|---:|---:|---:|---:|
| Code | 49.42 | 129.47 | 53.69 | 161.05 | 345.90 |
| Math | 49.19 | 125.76 | 49.80 | 143.55 | 285.33 |
| Fable | 45.71 | 55.39 | 48.61 | 66.50 | 123.63 |
| Hello | 45.92 | 67.31 | 48.51 | 94.59 | 141.10 |
| Topic | 47.92 | 74.02 | 48.72 | 91.22 | 169.24 |
| Natural JSON | 48.83 | 103.06 | 51.56 | 118.23 | 175.33 |
| Schema JSON | 48.60 | 105.39 | 52.04 | 129.41 | HTTP 400 |
| Multilingual | 47.55 | 72.63 | 49.33 | 83.32 | 183.61 |
| Counting 1–200 | 49.58 | 161.71 | 53.85 | 196.70 | 427.29 |

**One-RTX prefill matrix.** Median effective tokens/s after one shape warmup, with three samples per cell and verified parent reuse.

| Retained base | +1K | +2K | +4K | +8K | +16K | +32K |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 2,907 | 3,940 | 6,857 | 7,403 | 7,681 | 7,721 |
| 32K | 2,120 | 3,094 | 4,931 | 5,839 | 6,392 | 6,708 |
| 64K | 1,936 | 2,808 | 4,652 | 5,487 | 6,014 | 6,249 |
| 128K | 1,735 | 2,438 | 3,958 | 4,689 | 5,243 | 5,464 |
| 256K | 1,340 | 1,957 | 2,978 | 3,635 | 4,032 | 4,256 |

**Two-RTX prefill matrix.** Median effective tokens/s after one shape warmup, with three samples per cell and verified parent reuse.

| Retained base | +1K | +2K | +4K | +8K | +16K | +32K |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 4,770 | 6,346 | 7,853 | 8,223 | 8,372 | 8,300 |
| 32K | 3,704 | 5,188 | 6,416 | 6,987 | 7,224 | 7,262 |
| 64K | 3,282 | 4,610 | 5,716 | 6,313 | 6,543 | 6,597 |
| 128K | 2,618 | 3,764 | 4,748 | 5,235 | 5,498 | 5,586 |
| 256K | 1,806 | 2,721 | 3,356 | 3,841 | 4,160 | 4,270 |

**Single-RTX short-prefill controls.** Separate three-sample cells with identical prompt hashes. Fresh v4/v3/v4 deployments and matched direct/code/topic/counting plus three mixed sweeps do not reproduce the large historical short-prefill loss. The full matrix additionally follows long-context decode; these controls do not isolate that broader state effect or replace the matrix above.

| Base | Suffix | v3 fresh | v4 fresh A | v4 fresh B | v3 after decode history | v4 after decode history | History change |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0K | +1K | 2,970.56 | 3,037.52 | 3,132.66 | 3,009.48 | 3,112.90 | +3.4% |
| 0K | +2K | 4,072.13 | 4,047.54 | 4,128.23 | 3,966.32 | 4,144.64 | +4.5% |
| 0K | +8K | 7,007.19 | 7,472.72 | 7,550.20 | 7,014.61 | 7,134.49 | +1.7% |
| 32K | +1K | 2,510.09 | 2,513.77 | 2,595.95 | 2,366.91 | 2,494.10 | +5.4% |
| 32K | +2K | 3,396.69 | 3,435.85 | 3,501.40 | 3,169.73 | 3,355.03 | +5.8% |
| 32K | +8K | 6,133.57 | 6,066.18 | 6,085.77 | 5,882.58 | 5,800.05 | -1.4% |

**Decode over retained context.** Three samples for each of eight content types per base. Changes compare matched v3 retained-context runs. The 2K row is a separate realistic-context measurement without a matching v3 baseline; it is not substituted for the original zero-context protocol.

| Retained base | 1 RTX weighted dSpark | Change from v3 | 2 RTX weighted dSpark | Change from v3 | 1 RTX completed/cache-valid | 2 RTX completed/cache-valid |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 76.69 | -0.1% | 96.89 | +3.5% | 24/24 | 24/24 |
| 2K | 77.44 | — | 98.30 | — | 24/24 | 24/24 |
| 32K | 76.27 | +6.0% | 93.35 | +8.8% | 24/24 | 24/24 |
| 64K | 76.22 | +7.8% | 92.83 | +6.9% | 24/24 | 24/24 |
| 128K | 74.81 | +9.5% | 92.28 | +9.6% | 24/24 | 24/24 |
| 256K | 72.21 | +7.2% | 84.70 | +8.5% | 24/24 | 24/24 |

**Concurrency scaling.** Median aggregate tokens/s across three runs, timed from earliest first content to last completion, including admission gaps.

| Concurrency | 1 RTX counting | 2 RTX counting | 1 RTX code | 2 RTX code | 1 RTX topic | 2 RTX topic |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 159.22 | 192.61 | 134.44 | 159.82 | 72.49 | 98.38 |
| 2 | 252.00 | 299.40 | 214.52 | 268.86 | 124.23 | 167.53 |
| 4 | 445.55 | 541.46 | 372.87 | 444.58 | 215.39 | 289.90 |
| 8 | 741.35 | 912.15 | 621.51 | 789.43 | 336.89 | 461.62 |
| 16 | 1,258.23 | 1,494.85 | 1,042.00 | 1,298.72 | 554.17 | 734.40 |

**Mixed traffic.** Fixed code/fable/topic mix with simultaneous admission and nonce seed 56001. Three-sweep ranges preserve workload and scheduling variation.

| Concurrency | 1 RTX median (range) | 2 RTX median (range) |
|---|---:|---:|
| 1 | 110.24 (109.16–128.20) | 151.94 (150.39–158.23) |
| 2 | 111.83 (106.00–113.52) | 145.15 (142.36–151.23) |
| 4 | 133.43 (131.78–152.20) | 187.18 (182.92–192.44) |
| 8 | 155.36 (147.94–171.33) | 226.39 (204.87–226.82) |
| 16 | 198.31 (189.82–200.35) | 308.19 (307.01–313.51) |

**Deployment and cache capacity.** Actual launch controls and startup pool reservation. Prompt/completed retention counts are cache entries.

| Layout | RTX expert layers | Spark residency / budget | Global FP4 source pool | Prompt / completed retention |
|---|---:|---:|---:|---:|
| 1 RTX | 0–4, TP1 | 40 layers / 100 GiB each | 16.681 GB / 18,710,016 logical + 32,768 private-tail tokens | 24 / 24 |
| 2 RTX | 0–19, TP2 | 20 layers / 100 GiB each | 13.094 GB / 14,680,064 logical + 32,768 private-tail tokens | 24 / 24 |

**Startup.** Measured standard-script startup to API readiness, including deployment orchestration. Each entry is one launch, not a three-run median.

| Layout | v3 seconds | v4 seconds | Change |
|---|---:|---:|---:|
| 1 RTX | 58.05 | 60.66 | +4.5% |
| 2 RTX | 46.00 | 52.71 | +14.6% |

**Memory after readiness.** Measured immediately after the standard dSpark launch: the initial single-RTX launch and the final restored dual-RTX launch. Includes all GPU allocations, not just KV; free memory is the reported driver value.

| Layout | GPU | Loaded MiB | Free MiB | Runtime headroom policy MiB |
|---|---:|---:|---:|---:|
| 1 RTX | 0 | 94,568.00 | 2,683.00 | 2,048.00 |
| 2 RTX | 0 | 95,080.00 | 2,171.00 | 800.00 |
| 2 RTX | 1 | 95,320.00 | 1,928.00 | 800.00 |

**Observed GPU peaks.** Sampled device usage across each performance phase, including idle monitored cards. Used memory includes weights, KV and workspaces; it is not KV capacity. Peaks need not occur simultaneously.

| Phase | GPU UUID | Peak used GiB | Minimum free GiB | Power W | Memory MHz |
|---|---:|---:|---:|---:|---:|
| dual-dspark | GPU-95f8f212-9131-df99-fd53-7535965197d7 | 94.26 | 0.71 | 401.38 | 13,365.00 |
| dual-dspark | GPU-fe5b6dd0-a77c-c8fb-6360-e1b9d9918ac0 | 93.90 | 1.07 | 419.85 | 13,365.00 |
| dual-retained_2k | GPU-95f8f212-9131-df99-fd53-7535965197d7 | 94.26 | 0.71 | 252.30 | 13,365.00 |
| dual-retained_2k | GPU-fe5b6dd0-a77c-c8fb-6360-e1b9d9918ac0 | 93.90 | 1.07 | 232.80 | 13,365.00 |
| dual-target | GPU-95f8f212-9131-df99-fd53-7535965197d7 | 84.63 | 10.34 | 210.03 | 13,365.00 |
| dual-target | GPU-fe5b6dd0-a77c-c8fb-6360-e1b9d9918ac0 | 92.73 | 2.24 | 191.12 | 13,365.00 |
| single-dspark | GPU-95f8f212-9131-df99-fd53-7535965197d7 | 0.01 | 94.96 | 22.13 | 405.00 |
| single-dspark | GPU-fe5b6dd0-a77c-c8fb-6360-e1b9d9918ac0 | 94.88 | 0.10 | 415.37 | 13,365.00 |
| single-retained_2k | GPU-95f8f212-9131-df99-fd53-7535965197d7 | 0.01 | 94.96 | 19.98 | 405.00 |
| single-retained_2k | GPU-fe5b6dd0-a77c-c8fb-6360-e1b9d9918ac0 | 94.88 | 0.10 | 246.32 | 13,365.00 |
| single-target | GPU-95f8f212-9131-df99-fd53-7535965197d7 | 0.01 | 94.96 | 20.02 | 405.00 |
| single-target | GPU-fe5b6dd0-a77c-c8fb-6360-e1b9d9918ac0 | 90.59 | 4.38 | 231.66 | 13,365.00 |

**Tool calling.** Exactly three completed campaigns, thinking enabled at high effort, C16, and a 4,096-token response cap including reasoning. These runs used the earlier clean integration image; later graph-lifetime changes receive focused checks. See the tool-eval report for image provenance and failures.

| Run | Basic | Hard | Total |
|---|---:|---:|---:|
| 2026-09-16T04-12-34.961137Z_2214ecfb | 124/138 | 31/38 | 155/176 |
| 2026-09-16T04-17-40.551606Z_43911d6a | 126/138 | 34/38 | 160/176 |
| 2026-09-16T04-22-39.602726Z_81eaa393 | 123/138 | 33/38 | 156/176 |

**Performance review.** Against published v3, weighted dSpark decode rises from 76.72 to 78.97 tok/s on one RTX (+2.9%) and from 79.33 to 97.79 on two (+23.3%). At 32K–256K retained context, weighted decode improves 6.0–9.5% and 6.9–9.6%, respectively. Single-RTX topic C4/C8 medians fall 3.3%/6.0%; generated answers differ across versions and some repeats, so those records do not isolate kernel speed or draft acceptance.

The full single-RTX matrix shows 10–17% lower throughput on retained +1K/+2K prefills than the historical v3 table. Preceding request history was not controlled across that historical comparison. Separate fresh v4/v3/v4 and matched decode-history controls use identical prompt hashes: the short-prefill loss does not reproduce there, while v3 itself slows after decode history. These controls do not include the full matrix's preceding long-context decode campaign, and do not isolate the internal cause of state sensitivity. The original matrix and its losses remain above; the better control results do not replace them.

The final single-RTX campaign completes with 99 MiB minimum free after accumulated decode and prefill, retaining the original 16.681 GB / 18,710,016-logical-token pool. In-place query RoPE saves exactly 512 MiB at readiness. Standard restarts measured 60.66/52.71 seconds for one/two RTX, versus v3's 58.05/46.00; these include deployment orchestration and are single observations, not repeated startup medians.

The [single-topic review](sparkinfer-upstream-v4-single-topic-review-20260916.json), [fresh prefill control](sparkinfer-upstream-short-prefill-focused-20260916.json), and [matched-history control](sparkinfer-upstream-short-prefill-history-20260916.json) preserve sample ranges, prompt/output hashes, image identities and raw evidence. Their tables are also reproduced in the README.

## Source and evidence

The benchmarked coordinator and all four Spark workers were built from engine revision `39242276302b24e6e07313152cc532e43d8ed6ca`, source manifest `bf22e0f27f0368db99fbe43c966b3be771a14ead09f2d1428d872521d98b0394`, and merged SparkInfer `4e31d0a10ea7ca24ac48d522fb65eaafbc27ad30`. Later release commits update documentation and report tooling; they do not change these benchmarked runtime artifacts. The [clean build record](sparkinfer-upstream-v4-final-build-20260916.json) records image labels, packaged binary checksums and startup.

[Machine-readable results](release-v4-performance.json) contain exact summaries, commands, configuration, GPU UUIDs, source/corpus/binary hashes, memory telemetry and evidence hashes. The [complete matrix archive](evidence/upstream-v4-final-matrix-20260916.tar.gz) preserves every raw sample and the campaign runners. A harness log-name collision stopped the first attempt before any target-only request; the resumed campaign kept completed phases and ran only unfinished phases. Both attempt records are preserved.

To regenerate the numerical summary from the extracted archive, run `scripts/summarize-ds41-upstream-release.py --input <archive>/performance --build-record <archive>/v4-final-build-record.json --output <new-summary.json>`. The shared table renderer is `scripts/render-ds41-upstream-release-tables.py`; it also reads the committed v3/API references and the two focused-control reports.

Quality evidence is separate from throughput scoring. Exactly [three high-thinking tool campaigns](sparkinfer-upstream-tool-eval-20260916.md) completed all 264 scenarios, averaging 157/176 points; their report identifies the earlier clean integration image and preserves failures. The subsequent graph-lifetime and in-place query changes passed real-weight parity and [focused retained-context/vision/branch checks](sparkinfer-upstream-query-inplace-focused-20260916.json), including cold/exact 1.04M retrieval after accumulated single-RTX traffic. The [integration analysis](sparkinfer-upstream-integration-20260916.md) records component comparisons and numerical choices. The preserved official Flash reference was not rerun.
