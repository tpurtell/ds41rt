# DS41RT v5

V5 adds optimized mixed-projection EXL3 routed experts to DS41RT while preserving the official DeepSeek V4.1 Flash checkpoint as the standard launcher default.

- Generic K2–K5 EXL3 routed-expert loading and native execution in source, with the published AOT packages tuned and qualified for the 3.25 bpw K3/K4 mixed-projection checkpoint.
- Paired four-Spark ownership balances 3-bit and 4-bit projection work dynamically without changing the full-model path.
- Placement-aware dSpark profiles for the qualified one- and two-RTX layouts, including separate RTX and Spark expert costs.
- Automatic memory planning uses the quant checkpoint’s smaller routed experts to place additional complete bottom-up layers on RTX while preserving the release KV pools and runtime reserve.
- The official full checkpoint remains the default; `wrldsuksgo2mars/DeepSeek-V4.1-EXL3-K3.25-v1` is an opt-in FP8-PLE configuration.
- The FP4-PLE variant receives separate tool, focused-quality, and fixed-history top-1 analysis. Normal performance tables use FP8 PLE.
- Release reporting compares full and EXL3 checkpoints across one/two RTX cards, content types, concurrency, mixed traffic, retained context, prefill, capacity, startup, and memory.
- OpenAI-compatible streaming, high-effort reasoning by default, tools, structured output, vision, retained-turn snapshots, and bounded replay remain available through `run.sh` on port 8000.

All reported RTX measurements use a **400 W power limit per card and standard memory speed**. See the v5 performance report for exact results, methodology, raw evidence, quant geometry, agreement results, and observed limitations.

The official full checkpoint remains the launcher default after qualification: EXL3 improved weighted decode by +0.9% on one RTX and +8.8% on two, while its best two-RTX prefill was -38.5% versus full.
