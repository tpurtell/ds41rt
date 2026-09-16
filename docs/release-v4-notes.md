# DS41RT v4

Draft: final performance measurements and publication are pending.

V4 integrates upstream SparkInfer's DeepSeek V4.1 Flash support with DS41RT's
native serving engine. It preserves the official model checkpoint, automatic
one/two-RTX placement, four DGX Spark workers, and OpenAI-compatible API.

- Updated sparse attention operates directly on the packed FP4 compressed
  cache, with FP8 decode arithmetic and BF16 QK / FP8 PV prefill arithmetic.
- Native lagged mHC and selected narrow-projection kernels reduce coordinator
  work while preserving lane-owned execution and CUDA graph replay.
- Bounded sorting of selected index positions improves locality without
  changing which positions are selected.
- FP8 sliding-window storage, bounded prefix replay, completed-turn snapshots,
  tool calling, structured output, and vision remain supported.
- Standard launch controls remain available for RTX layout, concurrency,
  memory reservation, KV capacity, and retained entries on port 8000.

The [integration analysis](sparkinfer-upstream-integration-20260916.md) records
the comparisons and implementation choices. The [clean-image evidence](
sparkinfer-upstream-clean-serving-20260916.json) covers vision, 1.04M-token
retrieval, exact and partial cache reuse, concurrent branching, and cancellation.

Release performance tables, final tool-evaluation results, image digests, and
downloadable assets will be added after qualification finishes.
