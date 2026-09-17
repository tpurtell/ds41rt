# DS41RT v5 performance and quant analysis

Measured September 17, 2026 on one or two RTX PRO 6000 Blackwell 96 GB coordinator GPUs and four DGX Spark workers. The candidate uses engine revision `3841117183a399a30bfc2f227b77cacbbefebc38` and SparkInfer revision `e685f48c3b941208a8dc5915a742738a3c2de706`.

Each performance cell has three measured samples. The dSpark campaigns run nine-category direct decode, code/topic/counting concurrency, three mixed sweeps, retained-context decode, prefill, and a separate realistic 2K retained-base check. Target-only decode uses a fresh launch. The four prefill matrices each cover five retained bases by six suffix sizes after shape warmup and verified parent reuse. Tool evaluation uses high-effort thinking; ordinary throughput cases disable thinking except the explicitly labeled reasoning-code case, which enables high effort.

The full and EXL3 performance matrices use FP8 PLE. FP4 PLE is separately covered by tool, focused-quality, quant-shape, and fixed-history top-1 checks. All reported RTX measurements use a 400 W power limit per card and stock memory speed.

RTX measurements use **400 W per card** and **standard memory speed, without a memory overclock**. All performance tables use FP8 PLE. Reasoning code uses thinking enabled at high effort; its throughput includes reasoning and final-answer tokens.

**Headlines.** Median tokens/s. Changes compare two RTX cards with one for the same checkpoint. Counting is outside the weighted content score.

| Measurement | Full 1 RTX | Full 2 RTX | Change | EXL3 1 RTX | EXL3 2 RTX | Change |
|---|---:|---:|---:|---:|---:|---:|
| Best median prefill | 7,852.77 | 8,366.20 | +6.5% | 4,460.01 | 5,149.28 | +15.5% |
| Counting target-only decode | 49.56 | 53.76 | +8.5% | 49.55 | 57.34 | +15.7% |
| Counting dSpark decode | 161.70 | 219.75 | +35.9% | 167.68 | 247.83 | +47.8% |
| Weighted nine-category target-only decode | 48.36 | 51.97 | +7.5% | 48.46 | 55.05 | +13.6% |
| Weighted nine-category dSpark decode | 92.79 | 109.88 | +18.4% | 93.66 | 119.55 | +27.6% |
| C16 code aggregate | 1,074.96 | 1,205.02 | +12.1% | 1,153.15 | 1,294.55 | +12.3% |
| C16 topic aggregate | 595.79 | 603.62 | +1.3% | 549.34 | 756.30 | +37.7% |
| C16 counting aggregate | 1,253.43 | 1,511.84 | +20.6% | 1,366.95 | 1,690.30 | +23.7% |
| C16 mixed aggregate | 200.11 | 295.71 | +47.8% | 223.10 | 332.48 | +49.0% |

**Full content-type decode.** Three samples per case. The nine-category score retains non-thinking code separately from reasoning code. Historical official results, when shown, were not rerun and include different prior fable wording; no reasoning-code reference exists.

| Case | 1 RTX target | 1 RTX dSpark | 2 RTX target | 2 RTX dSpark | Historical official Flash |
|---|---:|---:|---:|---:|---:|
| Code | 49.41 | 132.81 | 53.36 | 155.69 | 345.90 |
| Code with reasoning | 48.58 | 100.61 | 52.21 | 126.34 | — |
| Math | 49.23 | 140.02 | 53.39 | 136.26 | 285.33 |
| Fable | 46.25 | 57.41 | 49.75 | 68.30 | 123.63 |
| Hello | 46.07 | 61.57 | 48.84 | 98.06 | 141.10 |
| Topic | 48.50 | 73.63 | 51.74 | 87.72 | 169.24 |
| Natural JSON | 48.79 | 104.04 | 53.19 | 129.97 | 175.33 |
| Schema JSON | 48.58 | 107.52 | 51.84 | 107.45 | HTTP 400 |
| Multilingual | 47.44 | 74.74 | 51.42 | 83.12 | 183.61 |
| Counting 1–200 | 49.56 | 161.70 | 53.76 | 219.75 | 427.29 |

**EXL3 content-type decode.** Three samples per case. The nine-category score retains non-thinking code separately from reasoning code. Historical official results, when shown, were not rerun and include different prior fable wording; no reasoning-code reference exists.

| Case | 1 RTX target | 1 RTX dSpark | 2 RTX target | 2 RTX dSpark | Historical official Flash |
|---|---:|---:|---:|---:|---:|
| Code | 49.30 | 132.59 | 56.90 | 165.83 | 345.90 |
| Code with reasoning | 48.94 | 107.19 | 55.72 | 142.87 | — |
| Math | 48.97 | 113.57 | 54.12 | 155.99 | 285.33 |
| Fable | 46.10 | 57.61 | 51.75 | 71.87 | 123.63 |
| Hello | 45.25 | 75.16 | 50.81 | 86.59 | 141.10 |
| Topic | 48.28 | 71.74 | 54.01 | 88.75 | 169.24 |
| Natural JSON | 47.75 | 98.26 | 55.41 | 122.22 | 175.33 |
| Schema JSON | 48.59 | 92.76 | 54.55 | 128.62 | HTTP 400 |
| Multilingual | 47.21 | 74.69 | 54.06 | 90.72 | 183.61 |
| Counting 1–200 | 49.55 | 167.68 | 57.34 | 247.83 | 427.29 |

**Full 1 RTX prefill matrix.** Median effective tokens/s, three samples per cell after shape warmup and verified parent reuse.

| Retained base | +1K | +2K | +4K | +8K | +16K | +32K |
|---|---:|---:|---:|---:|---:|---:|
| 0K | 3,119 | 4,165 | 6,987 | 7,510 | 7,703 | 7,853 |
| 32K | 2,473 | 3,215 | 5,128 | 6,000 | 6,549 | 6,789 |
| 64K | 2,213 | 3,080 | 4,727 | 5,598 | 6,094 | 6,328 |
| 128K | 1,905 | 2,699 | 4,121 | 4,855 | 5,285 | 5,528 |
| 256K | 1,446 | 2,098 | 3,131 | 3,778 | 4,130 | 4,298 |

**Full 2 RTX prefill matrix.** Median effective tokens/s, three samples per cell after shape warmup and verified parent reuse.

| Retained base | +1K | +2K | +4K | +8K | +16K | +32K |
|---|---:|---:|---:|---:|---:|---:|
| 0K | 4,755 | 6,367 | 7,875 | 8,206 | 8,366 | 8,289 |
| 32K | 3,666 | 5,118 | 6,423 | 6,941 | 7,228 | 7,264 |
| 64K | 3,272 | 4,595 | 5,702 | 6,239 | 6,510 | 6,581 |
| 128K | 2,622 | 3,744 | 4,699 | 5,225 | 5,496 | 5,568 |
| 256K | 1,831 | 2,674 | 3,349 | 3,859 | 4,149 | 4,251 |

**EXL3 1 RTX prefill matrix.** Median effective tokens/s, three samples per cell after shape warmup and verified parent reuse.

| Retained base | +1K | +2K | +4K | +8K | +16K | +32K |
|---|---:|---:|---:|---:|---:|---:|
| 0K | 2,531 | 2,965 | 4,084 | 4,296 | 4,431 | 4,460 |
| 32K | 2,125 | 2,495 | 3,728 | 4,068 | 4,229 | 4,292 |
| 64K | 1,939 | 2,408 | 3,523 | 3,850 | 4,033 | 4,092 |
| 128K | 1,699 | 2,164 | 3,151 | 3,503 | 3,695 | 3,777 |
| 256K | 1,331 | 1,777 | 2,584 | 2,892 | 3,095 | 3,170 |

**EXL3 2 RTX prefill matrix.** Median effective tokens/s, three samples per cell after shape warmup and verified parent reuse.

| Retained base | +1K | +2K | +4K | +8K | +16K | +32K |
|---|---:|---:|---:|---:|---:|---:|
| 0K | 3,124 | 3,761 | 4,981 | 5,106 | 5,149 | 5,112 |
| 32K | 2,710 | 3,363 | 3,939 | 4,323 | 4,560 | 4,631 |
| 64K | 2,468 | 3,112 | 3,637 | 4,033 | 4,265 | 4,344 |
| 128K | 2,096 | 2,745 | 3,229 | 3,582 | 3,803 | 3,872 |
| 256K | 1,577 | 2,180 | 2,524 | 2,881 | 3,104 | 3,184 |

**Decode over retained context.** Weighted nine-category dSpark tokens/s with verified retained-prefix reuse; three samples per case and base.

| Retained base | Full 1 RTX | Full 2 RTX | Change | EXL3 1 RTX | EXL3 2 RTX | Change |
|---|---:|---:|---:|---:|---:|---:|
| 0K | 87.51 | 105.77 | +20.9% | 88.99 | 116.77 | +31.2% |
| 2K | 86.58 | 106.66 | +23.2% | 91.79 | 119.13 | +29.8% |
| 32K | 87.46 | 101.74 | +16.3% | 87.39 | 111.07 | +27.1% |
| 64K | 85.54 | 103.45 | +20.9% | 88.49 | 111.17 | +25.6% |
| 128K | 85.05 | 102.39 | +20.4% | 87.41 | 115.55 | +32.2% |
| 256K | 79.68 | 96.22 | +20.8% | 82.67 | 105.92 | +28.1% |

**Full concurrency scaling.** Median aggregate tokens/s across three runs, from earliest first output to final completion, including admission gaps.

| Concurrency | 1 RTX counting | 2 RTX counting | 1 RTX code | 2 RTX code | 1 RTX topic | 2 RTX topic |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 159.25 | 211.23 | 137.57 | 171.33 | 75.98 | 97.59 |
| 2 | 250.57 | 315.17 | 218.00 | 278.81 | 126.12 | 162.21 |
| 4 | 450.17 | 584.14 | 380.88 | 458.27 | 223.88 | 263.55 |
| 8 | 739.81 | 971.54 | 620.52 | 755.73 | 340.93 | 437.76 |
| 16 | 1,253.43 | 1,511.84 | 1,074.96 | 1,205.02 | 595.79 | 603.62 |

**EXL3 concurrency scaling.** Median aggregate tokens/s across three runs, from earliest first output to final completion, including admission gaps.

| Concurrency | 1 RTX counting | 2 RTX counting | 1 RTX code | 2 RTX code | 1 RTX topic | 2 RTX topic |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 165.25 | 239.59 | 133.36 | 183.56 | 78.12 | 106.50 |
| 2 | 264.62 | 356.64 | 212.78 | 279.61 | 126.47 | 151.12 |
| 4 | 486.65 | 627.96 | 408.17 | 523.56 | 227.15 | 252.39 |
| 8 | 769.25 | 1,029.43 | 689.48 | 844.61 | 365.61 | 461.30 |
| 16 | 1,366.95 | 1,690.30 | 1,153.15 | 1,294.55 | 549.34 | 756.30 |

**Mixed traffic.** Matched code/fable/topic mix; median and range across three sweeps.

| Concurrency | Full 1 RTX | Full 2 RTX | EXL3 1 RTX | EXL3 2 RTX |
|---|---:|---:|---:|---:|
| 1 | 105.86 (104.60–131.07) | 158.21 (154.04–160.00) | 115.90 (114.83–132.69) | 157.12 (154.97–167.05) |
| 2 | 105.98 (103.43–113.89) | 141.17 (138.63–145.29) | 112.31 (107.78–114.26) | 132.20 (128.75–133.41) |
| 4 | 151.57 (135.82–155.87) | 182.83 (170.15–186.70) | 130.48 (128.88–157.58) | 187.68 (187.04–193.28) |
| 8 | 160.90 (154.58–175.47) | 199.02 (196.41–209.78) | 172.01 (155.78–176.85) | 256.15 (242.50–257.24) |
| 16 | 200.11 (199.43–202.68) | 295.71 (292.81–298.34) | 223.10 (218.31–230.36) | 332.48 (320.02–333.40) |

**Adaptive draft acceptance by content.** C1, three requests per category. Each cell shows accepted/verified draft percentage and mean emitted tokens per observed nonterminal verification cycle in parentheses. Terminal cycles are excluded; schema JSON uses grammar-constrained targets. Reasoning code includes both reasoning and final-answer generation. Adaptive selection censors unverified drafts, and model continuations differ: these are serving acceptance measurements, not teacher-forced quant agreement. Instrumented timings are excluded from TPS tables.

| Content | Full 1 RTX | Full 2 RTX | EXL3 1 RTX | EXL3 2 RTX |
|---|---:|---:|---:|---:|
| Code | 90.67% (5.26) | 74.89% (5.82) | 88.75% (5.10) | 73.46% (5.80) |
| Code with reasoning | 78.90% (3.74) | 68.83% (3.92) | 78.29% (3.71) | 66.20% (3.82) |
| Math | 90.97% (5.55) | 69.25% (5.21) | 68.29% (4.04) | 66.17% (5.09) |
| Fable | 52.86% (1.79) | 40.70% (1.77) | 51.59% (1.77) | 41.67% (1.81) |
| Hello | 65.12% (2.75) | 35.33% (2.43) | 63.33% (2.78) | 41.09% (3.04) |
| Topic | 61.93% (2.47) | 49.38% (2.47) | 64.16% (2.51) | 53.10% (2.53) |
| Natural JSON | 82.14% (4.17) | 62.14% (4.00) | 80.00% (4.17) | 68.79% (4.59) |
| Schema JSON (grammar-constrained) | 75.40% (4.65) | 54.76% (4.29) | 71.31% (4.48) | 59.01% (4.52) |
| Multilingual | 57.33% (2.22) | 50.37% (2.38) | 63.01% (2.43) | 51.99% (2.34) |

**Deployment and cache capacity.** Measured placement and pool reservation. Spark resident layers can include unused weights below the active range; retention values count cache entries.

| Configuration | RTX expert layers | Spark resident / active layers / budget | Global FP4 source pool | Prompt / completed retention |
|---|---:|---:|---:|---:|
| Full 1 RTX | 0–4, TP1 | 40 resident / 35 active / 100 GiB each | 16.681 GB / 18,710,016 logical + 32,768 private-tail tokens | 24 / 24 |
| Full 2 RTX | 0–19, TP2 | 20 resident / 20 active / 100 GiB each | 13.094 GB / 14,680,064 logical + 32,768 private-tail tokens | 24 / 24 |
| EXL3 1 RTX | 0–5, TP1 | 40 resident / 34 active / 100 GiB each | 16.681 GB / 18,710,016 logical + 32,768 private-tail tokens | 24 / 24 |
| EXL3 2 RTX | 0–24, TP2 | 20 resident / 15 active / 100 GiB each | 13.094 GB / 14,680,064 logical + 32,768 private-tail tokens | 24 / 24 |

**Startup.** Standard-script launch to API readiness, including orchestration; one launch per configuration.

| Configuration | Seconds |
|---|---:|
| Full 1 RTX | 59.78 |
| Full 2 RTX | 64.00 |
| EXL3 1 RTX | 63.65 |
| EXL3 2 RTX | 59.37 |

**Memory after readiness.** All GPU allocations, including weights, KV and workspace.

| Configuration | GPU | Loaded MiB | Free MiB | Runtime reserve MiB |
|---|---:|---:|---:|---:|
| Full 1 RTX | 0 | 94,568.00 | 2,683.00 | 2,048.00 |
| Full 2 RTX | 0 | 95,138.00 | 2,113.00 | 800.00 |
| Full 2 RTX | 1 | 95,726.00 | 1,522.00 | 800.00 |
| EXL3 1 RTX | 0 | 92,020.00 | 5,231.00 | 2,048.00 |
| EXL3 2 RTX | 0 | 93,772.00 | 3,479.00 | 800.00 |
| EXL3 2 RTX | 1 | 92,626.00 | 4,622.00 | 800.00 |

**Tool calling.** Three campaigns per checkpoint variant with high-effort thinking. Full/FP8-PLE rows preserve the clean v4 campaigns for the unchanged tool and schema path; both EXL3 variants are fresh v5 campaigns. Failures remain in the scores; see each campaign report for its engine/image provenance and response cap.

| Checkpoint | Run | Basic | Hard | Total |
|---|---:|---:|---:|---:|
| Full / FP8 PLE | 2026-09-16T04-12-34.961137Z_2214ecfb | 124/138 | 31/38 | 155/176 |
| Full / FP8 PLE | 2026-09-16T04-17-40.551606Z_43911d6a | 126/138 | 34/38 | 160/176 |
| Full / FP8 PLE | 2026-09-16T04-22-39.602726Z_81eaa393 | 123/138 | 33/38 | 156/176 |
| EXL3 / FP8 PLE | 2026-09-17T05-08-20.536955Z_1a1eb46f | 121/138 | 35/38 | 156/176 |
| EXL3 / FP8 PLE | 2026-09-17T05-13-17.171884Z_5c9247f1 | 125/138 | 32/38 | 157/176 |
| EXL3 / FP8 PLE | 2026-09-17T05-39-22.611937Z_9d81067b | 121/138 | 36/38 | 157/176 |
| EXL3 / FP4 PLE | 2026-09-17T05-54-27.625385Z_944f32ed | 123/138 | 34/38 | 157/176 |
| EXL3 / FP4 PLE | 2026-09-17T06-01-25.437889Z_b5d60f29 | 118/138 | 35/38 | 153/176 |
| EXL3 / FP4 PLE | 2026-09-17T06-08-18.487070Z_66d59e17 | 120/138 | 35/38 | 155/176 |

### Quant analysis

**Checkpoint and tensor payload sizes.** GiB uses 2^30 bytes. Tensor columns exclude safetensors headers. Both EXL3 checkpoints share identical routed experts; FP4 PLE changes only the lookup tables.

| Checkpoint | Shards | Checkpoint GiB | Size vs full | Routed-expert GiB | PLE GiB |
|---|---:|---:|---:|---:|---:|
| Full / FP8 PLE | 48 | 475.25 | — | 275.67 | 188.83 |
| EXL3 / FP8 PLE | 52 | 411.05 | -13.5% | 211.46 | 188.83 |
| EXL3 / FP4 PLE | 52 | 325.22 | -31.6% | 211.46 | 103.00 |

**EXL3 routed projection tiers.** Target layers plus one MTP layer: 384 experts each. The aggregate is 3.25 nominal bpw and 3.2601 bpw including scales and metadata.

| Projection | Logical shape | 3-bit tensors | 4-bit tensors | 4-bit share |
|---|---:|---:|---:|---:|
| w1 | 5,120 × 2,304 | 13,530 | 2,214 | 14.06% |
| w2 | 2,304 × 5,120 | 9,840 | 5,904 | 37.50% |
| w3 | 5,120 × 2,304 | 12,054 | 3,690 | 23.44% |

**PLE table geometry.** The FP4-PLE clone hard-links 48 unchanged shards and replaces 4 shards.

| Variant | Stored tensor geometry | Payload GiB |
|---|---:|---:|
| FP8 PLE | F8_E4M3 384,006,168×256 (1×); F8_E4M3 384,016,682×256 (1×); F8_E8M0 384,006,168×8 (1×); F8_E8M0 384,016,682×8 (1×) | 188.83 |
| FP4 PLE | F32 scalar (2×); F8_E4M3 384,006,168×16 (1×); F8_E4M3 384,016,682×16 (1×); U8 384,006,168×128 (1×); U8 384,016,682×128 (1×) | 103.00 |

**Fixed-history top-1 agreement.** Exact unconstrained target argmax IDs at C1 with dSpark disabled. All checkpoints use the same 20-layer TP2 placement, byte-identical teacher-forced assistant prefixes, and corpus SHA-256 `2c30b0c840160ac02a859e177fc0d7e0dad3e6dcab1d67c57bc5b2d6d26c6e45`. Runtime includes launch and collection. This isolates next-token argmax preservation; it is not a long-form generation-quality score.

| Checkpoint | Samples | Matches | Wilson 95% CI | Seconds |
|---|---:|---:|---:|---:|
| Full baseline | 132 | Reference | — | 66.4 |
| EXL3 / FP8 PLE | 132 | 95/132 (71.97%) | 63.77–78.93% | 77.8 |
| EXL3 / FP4 PLE | 132 | 99/132 (75.00%) | 66.98–81.61% | 59.6 |

**Top-1 agreement by category.** Each category uses twelve fixed continuations; percentages retain exact token-ID denominators.

| Category | EXL3 / FP8 PLE | EXL3 / FP4 PLE |
|---|---:|---:|
| Creative | 10/12 (83.3%) | 9/12 (75.0%) |
| Factual | 8/12 (66.7%) | 8/12 (66.7%) |
| Instructions | 10/12 (83.3%) | 10/12 (83.3%) |
| Long Context | 10/12 (83.3%) | 10/12 (83.3%) |
| Math | 11/12 (91.7%) | 11/12 (91.7%) |
| Multilingual | 5/12 (41.7%) | 5/12 (41.7%) |
| Python | 7/12 (58.3%) | 10/12 (83.3%) |
| Reasoning | 8/12 (66.7%) | 8/12 (66.7%) |
| Science | 8/12 (66.7%) | 9/12 (75.0%) |
| Structured | 8/12 (66.7%) | 9/12 (75.0%) |
| Systems | 10/12 (83.3%) | 10/12 (83.3%) |

## Review

The EXL3 checkpoint changes weighted dSpark decode from 92.79 to 93.66 tok/s on one RTX (+0.9%) and from 109.88 to 119.55 tok/s on two RTX cards (+8.8%). High-effort reasoning-code decode changes from 100.61 to 107.19 tok/s and from 126.34 to 142.87 tok/s. Its two-RTX best prefill is 5,149.28 tok/s versus 8,366.20 for the full checkpoint (-38.5%), so the official full checkpoint remains the standard launcher default and EXL3 remains opt-in.

The v5 qualification-evidence release asset contains the raw samples, client results, server traces, GPU telemetry, launch metadata, validation scripts, and recursive checksums used to reproduce these tables.
