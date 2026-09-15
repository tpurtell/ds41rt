# NVFP4 PLE-only variant

Target: `wrldsuksgo2mars/DeepSeek-V4.1-EXL3-K3.25-FP4PLE-v1` (public).
Completed public revision: `04cada4d3f38584f069e0a7debc53720832d738e`.
Base: `wrldsuksgo2mars/DeepSeek-V4.1-EXL3-K3.25-v1`, pinned revision
`cfd4ca1d1934a8e81dd2d7515598d4ce288e8b88`.

User constraints: CPU only; GPUs are serving and must not be touched. No
activation calibration, whole-model numerical validation, model replay, or expert
requantization. User clarified that a tiny individual-row numerical check is
desired: packing, block/global scales, and reconstruction error only. Basic
file/header/index and publication integrity checks remain.
Do not scan/hash/upload unchanged base weights. Copy their existing blob
references server-side into the new repository; locally hardlink them.

Representation: one-dimensional blocks of 16 consecutive embedding dimensions,
packed E2M1 (even element in low nibble), FP8 E4M3 block scales, and one FP32
global scale per PLE table. Weight-only round-to-nearest-even conversion,
global scale = table absolute maximum / (6 * 448); block scale is block maximum
divided by (6 * global), clamped to [2^-9, 448] and rounded to E4M3. Zero blocks
use scale 1. This follows NVIDIA ModelOpt's default dynamic NVFP4 weight recipe:
https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt/torch/quantization/qtensor/nvfp4_tensor.py

The maximum pass is required to derive weight scales, not a validation pass.
Read the source packed FP8/E8M0 data in bounded chunks, compute maximum FP8 code
per source exponent, then convert with small lookup tables. This avoids full
FP32 table allocations, GPU work, and expensive floating-point work per value.
Source weights/scales and all base artifacts remain immutable.

Keep 52 globally numbered shards: 1–48 unchanged, 49/51 each hold one PLE's
`weight_scale` [rows,16] E4M3 and `weight_scale_2` scalar F32, 50/52 hold packed
`weight` [rows,128] U8. Tensor names retain the original `layers.N.engram.embed`
prefix. Document the representation in both config metadata and the model card;
no claim of existing inference-engine support or quality validation.

Use detached CPU-only execution, durable per-chunk resume records, source
fingerprints and small-code identity hashes. Only new PLE shards and metadata
are uploaded. New cache snapshot must reuse unchanged base inodes, be readable
as tj, and point to the acknowledged public commit. Preserve the original repo,
export, snapshots, and receipts.

## Detached execution

Implementation: `fp4ple.py` with the eight-thread `ple_nvfp4_cpu.cpp` helper.
The synthetic 256-value microcheck passed exact packing and E4M3-scale comparison
against independent NumPy arithmetic/Torch FP8 casting. Each actual table also
checks one middle row using its final global scale; no whole-table error pass.

Run root: `/home/tj/.cache/ds41rt/quantization/deepseek-v41-exl3-k325-fp4ple-v1`.
Export: `/home/tj/.cache/huggingface/ds41rt-exports/DeepSeek-V4.1-EXL3-K3.25-FP4PLE-v1`.
Progress: `events.jsonl`, `ple-{1,14}/stats.json`, `conversion.json`, `complete.json`.
Statistics derive the required global scale; conversion checkpoints are synced
after every 262,144 rows (64 MiB source weights). A completed table is not redone.
Source fingerprints and small-code hashes bind resume identity. Upload progress
is durable in `preuploaded/`; existing prepared files reuse their receipts.
The original export, base source receipts, and inference server are untouched.

Launched detached container `ds41rt-fp4ple-cpu` with no GPU device requests,
`CUDA_VISIBLE_DEVICES=` and `NVIDIA_VISIBLE_DEVICES=void`, eight CPUs and 8 GiB
memory, uid/gid 1000. Image:
`sha256:bdd15949d70120fa42e4f9188da727f288ca0617e58b04f4ec97fdc5aa6f88f9`.
Exact entrypoint `/opt/glmrt/quant-venv/bin/python /code/fp4ple.py --stage all`.
Mount `quantization/` read-only at `/code`, and mount host HF home and
`/home/tj/.cache/ds41rt/quantization` at their identical absolute paths.
Environment: `PYTHONPATH=/code`, `OMP_NUM_THREADS=8`, `OPENBLAS_NUM_THREADS=1`,
`HF_TOKEN_PATH=/home/tj/.cache/huggingface/token`; do not log credential contents.

Inspect `docker inspect ds41rt-fp4ple-cpu` and `docker logs --tail 30
ds41rt-fp4ple-cpu` on reconnection. No automatic restart. On failure, diagnose
the terminal state before explicitly resuming the same script/identity. No
numerical-quality claim follows from completion; the requested check is only
an individual row. Publication/cache receipts are separate from conversion
completion and must exist before treating the target repo as finished.

Launch note: the first attempt stopped before conversion because the existing
`ds41rt-exports` parent is root-owned. A separate CPU/no-network setup container
created only the new target directory and assigned it to 1000:1000; the same
terminal job was then explicitly restarted. The run root was already owned by
tj. Do not change ownership of the whole HF home or other model repositories.

## Conversion completed; publication in progress

Both tables converted in the CPU-only job. The independent middle-row checks
matched packed bytes and block scales exactly: layer 1 relative L2 error
0.08876550, layer 14 0.09830786 (256 values each). This does not assess whole
model quality. No whole-model replay, calibration or quality check was run.
All 94 variant files are assembled; tensor payload is 349,178,352,752 bytes.
86 unchanged files (including all 48 base weight shards) are local hardlinks
and planned server-side cross-repository copies. Only eight changed files are
uploaded: four PLE shards, index, two configs, and model card. The first new PLE
scale shard transferred at approximately 85 MB/s; publication/cache completion
is not established until the final receipts exist.

## Publication complete — 2026-09-16 Taipei

The detached CPU-only coordinator exited 0, with no OOM and no GPU device
requests. Public repository:
https://huggingface.co/wrldsuksgo2mars/DeepSeek-V4.1-EXL3-K3.25-FP4PLE-v1

Revision: `04cada4d3f38584f069e0a7debc53720832d738e`.
Anonymous Hub inspection confirms this head, `private=false`, and 94 files.
All eight changed files were uploaded and committed together with 86 unchanged
server-side copies from the pinned base. Base weight upload bytes: **zero**.
New file content submitted totals 110,627,438,017 bytes; Xet may deduplicate
parts of that new content. No additional full weight-hash pass was performed.

The standard local HF cache is complete at:
`/home/tj/.cache/huggingface/hub/models--wrldsuksgo2mars--DeepSeek-V4.1-EXL3-K3.25-FP4PLE-v1/snapshots/04cada4d3f38584f069e0a7debc53720832d738e`.
`ds41rt-fp4ple-offline-audit` exited 0 with networking disabled as uid 1000:
all 94 files resolve through `main`, are readable, and share inodes with the
export; all 86 reused files retain the original remote blob IDs and local
hardlinks. Every base export fingerprint is unchanged. This audit only checks
file identity/metadata and reads eight bytes per file; no tensor computation.

Run-root evidence: `publication-complete.json`, `upload-complete.json`,
`cache-complete.json`, `offline-cache-audit.json`, `artifact.json`, per-table
`complete.json`/`row-check.json`, and `events.jsonl`. Docker logs preserve the
full upload history. The source model, base quantization, old snapshots and
inference server were left intact. Only the requested micro-scale numerical
checks were run; whole-model quality remains untested by design.
