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
