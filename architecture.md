# DS41RT architecture

This document describes the qualified V4.1 release architecture. Detailed
implementation rationale is in [docs/ENGINEERING.md](docs/ENGINEERING.md). The
[v6 release plan](docs/release-v6-plan.md) links the current evidence.

## Ownership

| Component | Owner |
| --- | --- |
| API, tokenizer, admission, scheduling, sampling, request history | Coordinator process |
| Text embeddings and layer-0 input | Logical RTX0 |
| Vocabulary head | RTX0 in single mode; vocabulary-row TP2 in dual mode |
| CED attention, compression, indexers, cache | Layer owner; dual mode keeps source-dependent ranges together |
| Native vision encoder, spatial aligner, image embeddings | Coordinator RTX owner |
| Engram mapped weights/scales and background page prefetch | Host storage and I/O workers |
| Gathered Engram dequantization, projection, gate/residual update | Owning coordinator RTX |
| Backbone routers | Owning coordinator RTX |
| Backbone shared experts | RTX0 in single mode; TP2 in dual mode |
| Bottom-up resident backbone routed experts | Coordinator; TP1 in single mode, TP2 in dual mode |
| Remaining backbone routed experts | Four Spark intermediate-dimension TP ranks (or, in the opt-in pure `TP6xEP1` layout, six unreplicated intermediate slices — see below) |
| Three-stage dSpark drafter | Coordinator; normally RTX1 in dual mode, with optional routed-expert TP2 |
| Retained inactive snapshots | Pinned host RAM; active requests remain in the GPU pool |

## Backbone and cache

The official backbone has 40 layers and width 5120, arranged as a 20-layer causal
encoder followed by a 20-layer decoder; routed experts use intermediate width 2304,
384 experts per layer, and top-6 routing.

Every layer has a 128-token local window; layers 2–19 add ratio-2 compressed global
attention, and layers 20–39 share ratio-1 global KV produced at the encoder/decoder
boundary.
Global KV source layers are 2, 8, 14, and 20, and index-selection sources are
2, 8, 14, 20, 24, 28, 32, and 36.
The first decoder indexer selects candidate blocks for later decoder indexers.

Window KV, compressed KV, and indexer keys have distinct native quantization
contracts; their layouts and scales must not be treated interchangeably.
Each request owns its cache references, source selections, candidate blocks,
compression tail, and speculative transaction state.

Partial prefix reuse rebuilds no more than the last 128 encoder tokens needed
for CED decoder SWA state. Exact hits restore retained windows and first-token
logits; compressed pages remain shared through copy-on-write ownership.
Prompt and completed-turn banks independently retain 20 entries by default.
Automatic pinned-RAM sizing makes logical GPU-plus-host capacity exceed 20 times
the configured maximum context. Host snapshots do not expand the active GPU set.

## Engram and speculative execution

Engram modules at layers 1 and 14 hash 2-, 3-, and 4-grams with eight heads per order.
Normalized token IDs determine table addresses before hidden-state computation,
allowing both modules' page prefetch to start as soon as the input IDs are known.
Image spans break n-gram history and receive no engram residual contribution.

Mapped table pages are advised by bounded background workers and gathered into
bounded staging buffers, then dequantized and projected on the GPU.
Prefill and verification batches must preserve row order, image masks, and request
identity; rejecting speculative tokens must not advance committed history.

The dSpark drafter has three stages with 128 routed experts and top-3 routing per
stage, uses the backbone's embedding and output head, and conditions on incoming
residual-stream means at target layers 37, 38, and 39.
It runs inside the coordinator while target verification still traverses backbone
AFD. In the selected dual default its transformer remains on RTX1 and the
vocabulary head is split. Optional routed-expert TP2 remains disabled by default.

## Serving and qualification

The target topology is four expert TP ranks and one coordinator process using
one or two RTX cards, with 16-request admission and two independent execution
lanes around remote expert boundaries. The coordinator reserves KV and runtime
headroom, then fills remaining device space with complete routed-expert layers
from the bottom up. Dual local expert layers use TP2, and every Spark starts at
the coordinator's published boundary.
Resident workspaces are prepared before readiness. Request-shape CUDA graphs
are captured lazily, then reused while their owner and binding identities match.
Readiness must verify checkpoint/dependency identity, weight residency, mapped
table access, transport, numerical startup probes, and prepared graph shapes.

The official checkpoint is the default release weight source. EXL3 remains an
optional compatibility path; its historical performance is outside the v6
native qualification tables.
See [docs/ds41-architecture-audit.md](docs/ds41-architecture-audit.md) for pinned
source evidence.
