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
- Decode retains bounded CUDA graph shapes across mixed traffic and batch
  restarts, avoiding the persistent slowdown caused by repeated index and
  cache-producer graph rebuilding.
- In-place query RoPE removes 512 MiB of redundant workspace across the default
  two serving lanes, preserving KV capacity and resident expert placement.
- FP8 sliding-window storage, bounded prefix replay, completed-turn snapshots,
  tool calling, structured output, and vision remain supported.
- Standard launch controls remain available for RTX layout, concurrency,
  memory reservation, KV capacity, and retained entries on port 8000.

The [integration analysis](sparkinfer-upstream-integration-20260916.md) records
the comparisons and implementation choices. The [clean-image evidence](
sparkinfer-upstream-clean-serving-20260916.json) covers vision, 1.04M-token
retrieval, exact and partial cache reuse, concurrent branching, and cancellation.

The [three-run high-thinking tool evaluation](sparkinfer-upstream-tool-eval-20260916.md)
completed all 264 scenarios with a mean of 157/176 points. The report states
the benchmark output cap and preserves every partial and failed result.

The earlier clean dual-RTX candidate increased weighted eight-type throughput
from 79.33 to 97.55 tokens/s against v3 (+23.0%). C16 code increases
from 1,181 to 1,296 aggregate tokens/s and topic from 596 to 749. C16 mixed
traffic is approximately flat (309 to 305). These are three-sample,
prompt-matched results at 400 W per RTX with stock memory clocks; the
[raw evidence and comparisons](sparkinfer-upstream-v4-dual-decode-20260916.json)
preserve ranges and individual cases. Final-image measurements remain pending.

The three tool campaigns and vision checks used the clean candidate before the final
graph-reuse correction. That correction changes graph lifetime, not kernel
arithmetic; it separately passed prefix equivalence, all 16 divergent cache
branches, a CUDA graph lifetime test, and cold/exact 1.04M-token retrieval.
The [correction evidence](sparkinfer-upstream-index-graph-reuse-20260916.json)
records the tested binaries explicitly.

Release performance tables, image digests, and downloadable assets will be
added after qualification finishes.
