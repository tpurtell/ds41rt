# DeepSeek V4.1 EXL3 K3.25

Target: `wrldsuksgo2mars/DeepSeek-V4.1-EXL3-K3.25-v1`.

The GPTQModel starting revision is the remote fork's main HEAD observed on
2026-09-14: `339cccf0c4b62440b572440a7d06608ed9e4a165`. This is a new
submodule checkout, not the older GLMRT vendor pin. V4 support exists in this
fork; V4.1 support still needs implementation and numerical qualification.

The local complete source snapshot is
`dba1be0a40aa45a94ad051997016db3960a90277`. Its config has 40 main blocks
(384 experts) and three dSpark blocks (128 experts), hidden size 5120 and
expert intermediate size 2304. Quantize only routed projections. Preserve
vision, shared experts, attention, embeddings, heads, and other source tensors.

Allocate K4 upgrades from scored K3 candidates with gate:up:down ratio 3:5:8.
Per main block the quotas are 54/90/144; per dSpark block they are 18/30/48.
Replay each actual mixed block before calibrating its successor. Bind the
calibration corpus, source, implementation, recipe, and resume records by hash.
Reuse the latest calibration texts and attest tokenization for this model.

PLE tables at layers 1 and 14 must remain file-backed and reclaimable. Export
each table with its scales in its own shard, separated from other tensors.
The source shards currently also contain the small engram projection tensors,
so simply hardlinking those entire shards does not meet this separation.
Use bounded gathers and measured prefetch; do not allocate full tables as
Torch parameters or retain their pages through activation references.

Use NVMe for rolling activation/replay frontiers and durable projection
checkpoints. Aim to export directly into the standard Hugging Face cache,
then bind that same payload to the published commit without redownload.
Recheck peak space including K3 candidates, selected K4, export and both PLE
tables before launch; scratch is available if a full intermediate is necessary.

Current state: no quantizer running. The user stopped the ds41rt server and
released both GPUs on 2026-09-14; free memory was verified at approximately
95 GiB per GPU. Build this repository's own workflow and documentation;
the sister projects are implementation references, not this run's artifacts.
No calibrated block or final ETA exists yet. After several durable blocks and
stable memory measurements, monitor the live job every 30 minutes and report
completed blocks, memory, throughput, and estimated final completion time.

Completion requires full main+dSpark quantization, exact tier/payload audit,
quality and recovery verification, separate PLE files, successful Hub upload,
remote object verification, and the matching local standard-cache snapshot.

## Implementation evidence (2026-09-14)

The user authorized both RTX GPUs and all four Sparks. Activation generation
and causal replay stay on the RTX GPUs; distribute independent trellis searches
using the fork's authenticated `exl3_remote` scheduler. Remote-worker support
exists but needs qualification for this run before launch.

Transformers is pinned to the unmerged V4.1 text implementation from PR 48721,
revision `62d7ebd7de4938e072b7aaeb881593b79dc56835`, under
`third_party/transformers`. It reports `5.18.0.dev0` and imports with the
current GPTQModel fork after upgrading the quantization Python environment's
tokenizers to 0.23.1. It omits dSpark and vision execution. Preserve vision
source tensors; implement dSpark in our separate GPTQModel V4.1 definition.
No production auto-dispatch registration yet: stateful layerwise capture,
bounded source loading, and complete dSpark support are prerequisites.

The separate `deepseek_v41.py` currently provides unfused routed experts and
a parameter-free mapped PLE embedding. Tests pass for:

- exact five-layer FP32 logits before/after expert conversion, with eager
  expert dispatch on both sides (default grouped dispatch differs by 1.8e-7);
- exact expert outputs against the checkpoint's unchanged `Expert` body in
  FP32 and BF16 on CPU and each RTX GPU;
- exact mapped FP8+E8M0 embedding gathers against the Transformers lookup on
  CPU and both RTX GPUs, including chunking, repeats, and empty inputs;
- mapping lifetime, explicit prefetch/release, owned rows, and bounds.
- five-layer explicit replay with owned CPU boundaries, carrying both mHC
  pre-mix and CSA2 shared state, matching full-forward logits exactly and
  reproducing each layer when replayed twice from the same input state.

`gptqmodel.utils.v41_replay.V41ReplayBatch` now implements that replay contract
for full-prompt text batches. It drops consumed PLE gathers at their layer.
Durable serialization, source-layer loading, dSpark replay and integration
with the quantizer's layer loop still remain to be implemented and tested.
TileLang 0.1.14 was built in the development container for direct checkpoint
reference tests; the production dependency lock is still pending.

`gptqmodel.utils.v41_source.V41Source` now reads source tensors individually
and can assemble one decoded main block on a chosen device. Loading real
block 0 completed in 6.02 seconds: 1,174 tensors and 25.62 GiB allocated on
GPU 0. The process exited successfully and GPU allocation returned to idle.
This decoded representation is for diagnostics and trellis weights, not yet
the calibration forward baseline: native FP8 activation quantization still
needs integration. No full source rewrite was made.

The source header test validates every runtime parameter/buffer name and
logical shape for all 40 main blocks against the actual 48-shard checkpoint,
with exact coverage of all non-scale, non-PLE main-block tensors. The two
PLE full-read rejection cases also pass. dSpark source mapping/loading is
still pending and is not covered by this test.

These are component gates, not full official-reference backbone parity or
quantization qualification. The checkpoint's `inference/model.py` remains the
architecture oracle. Preserve necessary arithmetic while measuring allocation
and conversion costs; reference implementation inefficiencies are not required.

Development container: `ds41rt-quant-dev`, based on local image
`sha256:6213ea40c79617373562d7f2d3cc5fa25ca9d03e27dd213361b216c3e315e9f4`.
It mounts this checkout at `/workspace`, source HF cache read-only at `/hf`,
and uses both vendored Python source trees through PYTHONPATH. It is a development
environment, not yet a reproducible production image. Run component gates with:

```bash
docker exec -w /workspace ds41rt-quant-dev python -m unittest discover -s quantization/tests -v
```
