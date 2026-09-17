# DS41RT

DS41RT serves the official [DeepSeek V4.1 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) checkpoint across one or two RTX PRO 6000 Blackwell coordinator GPUs and four DGX Spark expert workers. It combines native target execution, local dSpark speculative decoding, token-level prefix reuse, tools, constrained output, and vision in one OpenAI-compatible service.

All reported RTX measurements use an enforced **400 W power limit** and **standard 14,001 MHz maximum memory speed with no memory overclock**. The loaded memory clock reached 13,365 MHz; the RTX driver was 595.91.07. The four GB10 workers used driver 580.159.03.

[![DS41RT native execution across RTX coordinators and four expert workers](docs/native-path-execution.svg)](docs/native-path-execution.svg)

## Performance

The standard launcher selects two feasible peer-connected RTX cards and falls back to one. The official full checkpoint remains the default; the EXL3 3.25 bpw checkpoint is an opt-in configuration. Every performance cell contains three measured samples. Counting is retained as a low-entropy reference, while weighted content, reasoning code, and mixed traffic are the primary serving measurements.

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

**Performance review.** The EXL3 checkpoint changes weighted dSpark decode from 92.79 to 93.66 tok/s on one RTX (+0.9%) and from 109.88 to 119.55 tok/s on two RTX cards (+8.8%). High-effort reasoning-code decode changes from 100.61 to 107.19 tok/s and from 126.34 to 142.87 tok/s. Its two-RTX best prefill is 5,149.28 tok/s versus 8,366.20 for the full checkpoint (-38.5%), so the official full checkpoint remains the standard launcher default and EXL3 remains opt-in.

The [v5 performance report](docs/release-v5-performance.md) includes the measurement protocol and the same complete table set. [Exact structured results](docs/release-v5-performance.json) preserve the summaries and qualification provenance. The downloadable v5 qualification-evidence archive contains raw samples, traces, telemetry, launch records, and validators.

## Getting started

The release topology requires:

- one Linux amd64 host with one or two RTX PRO 6000 Blackwell 96 GB GPUs;
- four ARM64 DGX Spark hosts reachable over passwordless SSH;
- Docker with the NVIDIA Container Toolkit on all five hosts;
- IP connectivity for the expert fabric and `/dev/infiniband` access for the qualified RoCE path;
- the pinned model snapshot in the same `HF_HOME` layout on every host.

Weights are mounted read-only from the host Hugging Face cache and are not included in the repository or images.

Clone the source and initialize its pinned dependencies:

```bash
git clone --recurse-submodules https://github.com/tpurtell/ds41rt.git
cd ds41rt
git submodule update --init --recursive
```

Edit [`ds41rt.config`](ds41rt.config) for the deployment. At minimum, verify the four `SPARK_*_HOST` names and `SPARK_*_LANE_A` addresses. Configure all four `LANE_B` values for the qualified secondary rail or leave all four empty. Pin `COORDINATOR_GPU_UUID` and `COORDINATOR_GPU_PCI_BUS_ID` on multi-GPU hosts. Keep `MODEL_REVISION` synchronized with the snapshot installed on every machine.

The official full checkpoint remains the default. To opt into the routed-expert EXL3 checkpoint while retaining the normal FP8 PLE tables, set:

```bash
MODEL_ID=wrldsuksgo2mars/DeepSeek-V4.1-EXL3-K3.25-v1
MODEL_REVISION=cfd4ca1d1934a8e81dd2d7515598d4ce288e8b88
EXL3_PAIRED_TP4=on
```

Install that exact snapshot on the coordinator and all four Sparks. The FP4-PLE variant is separately analyzed and tool-qualified; the release performance tables use the FP8-PLE checkpoint above.

To use the published images, pull the coordinator image locally and the Spark image on each worker:

```bash
docker pull ghcr.io/tpurtell/ds41rt-coordinator:v5
for host in ostrich dodo emu kiwi; do
  ssh "$host" docker pull ghcr.io/tpurtell/ds41rt-spark-expert:v5
done
./run.sh --dry-run
./run.sh
```

To build from the checked-out source instead, run:

```bash
./build.sh
./run.sh --dry-run
./run.sh
```

`build.sh` compiles amd64 coordinator artifacts locally, compiles ARM64 expert artifacts natively on the first Spark, verifies source and dependency identity, and distributes the Spark image to all configured workers. A subset such as `./build.sh --spark-hosts ostrich,dodo` limits build and distribution only; serving still needs all four ranks.

The standard launch listens on port **8000**. Replace an existing deployment with `./run.sh --restart`; stop it with `./stop.sh`. A basic request is:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-ai/DeepSeek-V4.1-Flash",
    "messages": [{"role": "user", "content": "Write a CUDA haiku."}],
    "stream": false
  }'
```

Thinking is enabled at **high** effort when a request omits thinking controls. Pass `"reasoning_effort":"low"`, `"high"`, or `"max"` to select an effort, `"reasoning_effort":"none"` to turn it off, or `"thinking":{"type":"disabled"}` to disable it explicitly. Responses keep reasoning and final content separate.

## Runtime options

Command-line values override [`ds41rt.config`](ds41rt.config) for one launch:

| Option | Default | Purpose |
|---|---:|---|
| `--listen HOST:PORT` | `0.0.0.0:8000` | API bind address |
| `--rtx-gpus auto\|1\|2` | `auto` | Select two feasible peer GPUs automatically, or force a layout |
| `--concurrency N` | `16` | Active requests, 1–16 |
| `--kv-pool-size SIZE` | automatic | Exact global KV/index pool; B, MB, GB, MiB, or GiB |
| `--host-cache-bytes auto\|SIZE` | `auto` | Pinned RAM paging sized from GPU capacity, retention slots and context limit; `0` disables |
| `--memory-reservation SIZE` | device plan | Total GPU occupancy ceiling as bytes or a percentage |
| `--prefix-cache-entries N` | `20` | Independent limits for retained completed turns and prompt snapshots |
| `--max-context-tokens N` | `1,048,576` | Per-request context maximum |
| `--max-output-tokens N` | `393,216` | Model output maximum |
| `--prefill-batch-tokens N` | `2,048` | Prefill step size, 80–4,096 |
| `--dspark` / `--no-dspark` | enabled | Enable or disable speculative decoding |
| `--restart` | off | Replace the running five-host deployment |
| `--dry-run` | off | Validate configuration, images, hosts, model, and devices without starting services |

Automatic RTX selection checks physical UUIDs, bidirectional peer reads, the
requested cache or memory ceiling, and available memory. With `--restart`, it
adds back only memory owned by the coordinator container being replaced; other
GPU processes still count against feasibility. The chosen UUID order is passed
through unchanged as logical RTX0/RTX1. Dual mode publishes its live memory-derived RTX layer boundary before starting
Spark workers, so only the remaining routed layers load there. Single mode keeps
all 40 layers on every Spark. Restarting the same coordinator container preserves
its acknowledged boundary; `./run.sh --restart` replans the deployment.

With no explicit pool setting, single-RTX mode uses a **16.681 GB global pool
for 18,710,016 tokens plus 32,768 private-tail tokens**. Dual-RTX mode targets a
**13.094 GB global pool with 14,712,832 source-token positions**, including
32,768 tail/COW positions. The 20 completed-turn and prompt-snapshot limits
remain independent of the aggregate token budget. `--kv-pool-size` selects the
exact global pool. The recipe defaults to `--host-cache-bytes auto`, which keeps
usable GPU plus RAM token capacity above `prefix-cache-entries × max-context-tokens`;
snapshot overhead and restore staging are reserved separately. Explicit RAM sizes
and `0` override that automatic capacity guarantee. `--memory-reservation` caps total planned device occupancy;
when both are present, the exact pool must fit under the ceiling. A reservation
without an explicit KV size uses remaining capacity for KV before filling extra
RTX expert layers; specify both to preserve a chosen pool under a tighter ceiling. Smaller values
are useful for side-by-side development servers:

```bash
./run.sh --listen 0.0.0.0:18000 --concurrency 2 \
  --max-context-tokens 65536 --max-output-tokens 8192 \
  --kv-pool-size 2GiB --prefix-cache-entries 3 --no-dspark
```

The API supports incremental SSE, cancellation, tools, parallel tool calls, JSON and supported JSON Schema constraints, and up to sixteen images in a prompt. Image input accepts data URLs and bounded HTTP/HTTPS URLs. See the [thinking](docs/release-v1-thinking.md), [tool serving](docs/release-v1-tool-serving.md), [output constraints](docs/release-v1-response-constraints.md), and [vision](docs/release-v1-vision-serving.md) qualification records.

## Engineering

Single-RTX dSpark defaults to adaptive K5 with placement-aware costs calibrated
for the 400 W RTX and four Spark configuration. The binary embeds the profile
and selects costs from the expert layers actually installed on each backend.
Native serving accepts `--dspark-draft-limit 7` for K7;
`DS41RT_ADAPTIVE_COST_PROFILE=legacy` restores the previous cost formula.
The [development comparison](docs/phase2-adaptive-verification.md#single-rtx-default-comparison)
records the policy tradeoffs; the release performance tables above measure the selected K5 default.

The coordinator owns attention, mHC residuals, embeddings, mapped Engram lookup, routers, shared experts, vision, all three dSpark stages, the vocabulary head, sampling, cache ownership, and the API. Dual mode splits this work by dependency across the RTX pair, uses TP2 for all shared experts and encoder routed experts, and partitions vocabulary rows for deterministic parallel greedy selection. Four Sparks retain only decoder routed experts in dual mode and all routed experts in single mode.

Compressed global KV uses FP4 E2M1 values with group-16 E4M3 scales. The 128-token sliding windows remain FP8, and the independent selection index uses its own FP4 format. A token radix shares immutable pages, uses copy-on-write for divergent suffixes, and restores retained target/dSpark state. Partial matches replay no more than the final 128 encoder tokens; exact hits can reuse saved first-token logits. The default keeps 24 completed turns and 24 prompt snapshots under LRU eviction.

The [engineering report](docs/ENGINEERING.md) covers the final kernels, execution lanes, cache transactions, prefix policy, transport, dSpark, vision, constrained decoding, memory planning, and startup path. [`architecture.md`](architecture.md) gives the compact ownership contract.

## Qualification

The clean v5 candidate passed the scoped release qualification:

- matched one/two-RTX target-only and dSpark throughput for nine content categories plus counting, with three samples per cell;
- four one/two-RTX full/EXL3 prefill matrices covering 30 retained-base/suffix cells each, plus retained-prefix decode through 256K and a separate 2K control;
- counting, code, and topic at C1, C2, C4, C8, and C16, with three mixed-traffic sweeps for every checkpoint/layout combination;
- instrumented adaptive draft acceptance for nine categories across all four full/EXL3 one/two-RTX deployments;
- exactly three high-effort tool campaigns for EXL3 FP8 PLE and three for EXL3 FP4 PLE, alongside the preserved three-run full-checkpoint reference;
- focused vision and 32K/1.04M needle checks for EXL3, plus bounded exact target top-1 comparison against the full checkpoint;
- clean five-host candidate builds and standard-script deployment smokes, with binary, image, checkpoint, and raw-evidence provenance.

The performance report and downloadable qualification-evidence archive preserve failures and measured limitations alongside passes. The [Phase 2 engineering log](docs/phase2-dual-rtx.md) covers the earlier dual-RTX implementation sequence.

## RDMA tools for DGX Spark and RoCE PCs

Move model weights, checkpoints, datasets, and container images between your local AI machines with [Local AI Tap](https://github.com/tpurtell/local-ai-tap):

- **`rdmasync`** — rsync-style file synchronization with RDMA bulk transfers.
- **`rdmapipe`** — stream command output over RDMA into a remote command, using SSH for authentication and orchestration.

Native **ARM64 and AMD64 binary bottles** are available for Linux, including DGX Spark. Homebrew installs the dependencies automatically.

**Install on both endpoints**, with [Homebrew](https://brew.sh/) already installed:

```bash
brew tap tpurtell/local-ai https://github.com/tpurtell/local-ai-tap.git

if brew commands | grep -qx trust; then
  brew trust --tap tpurtell/local-ai
fi

brew install tpurtell/local-ai/rdmapipe tpurtell/local-ai/rdmasync
```

Replace `spark` below with your machine’s SSH hostname or alias.

**Copy model files over RDMA:**

```bash
rdmasync -a --rdma=required \
  --rsync-path=/home/linuxbrew/.linuxbrew/bin/rdmasync \
  ./models/ spark:~/models/
```

**Stream an ARM64 container image directly into a Spark:**

```bash
docker image save --platform linux/arm64 my-ai-image:latest |
  rdmapipe \
    --remote-path=/home/linuxbrew/.linuxbrew/bin/rdmapipe \
    spark -- docker image load
```

The Docker example requires an ARM64 image locally and Docker access on both machines.

Use these tools on a trusted, configured RDMA/RoCE fabric with working Linux drivers and SSH access. Bulk RDMA traffic is not encrypted.

[Installation guide and documentation →](https://github.com/tpurtell/local-ai-tap#readme)

## Thanks

DS41RT builds on the official DeepSeek V4.1 Flash model and reference implementation, NVIDIA CUDA and DGX Spark, [SparkInfer](https://github.com/efeslab/SparkInfer), [XGrammar](https://github.com/mlc-ai/xgrammar), Hugging Face, and the open-source Rust, Python, and CUDA ecosystems.

DS41RT is available under the [MIT License](LICENSE). Dependency licenses and notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
