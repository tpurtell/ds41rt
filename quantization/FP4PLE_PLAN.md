# NVFP4 PLE-only variant

Target: `wrldsuksgo2mars/DeepSeek-V4.1-EXL3-K3.25-FP4PLE-v1` (public).
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
