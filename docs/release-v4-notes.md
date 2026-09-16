# DS41RT v4

V4 integrates upstream SparkInfer's DeepSeek V4.1 Flash support with DS41RT's
native serving engine, improving decode on the standard one/two-RTX and four-Spark deployment.

- Faster packed-FP4 attention, native lagged mHC, and selected narrow-projection kernels.
- Weighted dSpark decode improves **23.3% on two RTX cards** and **2.9% on one** versus v3; retained-context decode improves approximately **6–10%** across 32K–256K contexts.
- CUDA graph reuse remains responsive across mixed traffic; in-place query RoPE saves **512 MiB** without reducing KV capacity or resident expert layers.
- OpenAI-compatible streaming, tools, structured output, vision, retained-turn snapshots, and bounded prefix replay remain available through the standard launcher on port 8000.
- Expanded performance reporting includes code/topic/counting concurrency, mixed traffic, full prefill matrices, 2K retained-context decode, memory, and startup.
- Three high-thinking tool campaigns completed all 264 scenarios, averaging **157/176 points**; focused checks cover 1.04M retrieval, vision, cache branching, cancellation and numerical parity.

All measurements use a **400 W limit per RTX and standard memory speed**.
The [performance report](release-v4-performance.md) includes every table and
observed regressions: single-topic C4/C8 are lower, startup is a few seconds
longer, and historical short-prefill differences depend on preceding workloads.
Separate fresh and matched decode-history controls do not reproduce the large
short-prefill loss; they do not replace the full-matrix results.

The [tool-evaluation report](sparkinfer-upstream-tool-eval-20260916.md) preserves
partial and failed cases and its earlier-image provenance. The
[integration analysis](sparkinfer-upstream-integration-20260916.md) records
component comparisons, numerical choices and focused checks after graph changes.

The images and binary assets use engine revision `3924227` and merged SparkInfer
`4e31d0a1`; later release commits contain documentation and report-tool updates.
Routed EXL3/Trellis and parallel attention/projection remain future work.
