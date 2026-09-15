# DeepSeek V4.1 routed EXL3 K3.25

The authoritative recipe, user constraints and qualification evidence are in
[PLAN.md](PLAN.md). This directory contains our implementation; sister-project
scripts are references, not production dependencies.

## Completed v1 artifact

Public model: https://huggingface.co/wrldsuksgo2mars/DeepSeek-V4.1-EXL3-K3.25-v1

Published revision: `076c0dcf88436e1d6f69ca50f3557c307439f2ef`.
The standard local cache snapshot is
`/home/tj/.cache/huggingface/hub/models--wrldsuksgo2mars--DeepSeek-V4.1-EXL3-K3.25-v1/snapshots/076c0dcf88436e1d6f69ca50f3557c307439f2ef`.
All 40 main and three dSpark blocks are complete: 47,232 routed projections,
11,808 at K4. All 94 published files (52 safetensors shards) are materialized
with hardlinks and were verified offline as uid 1000. No weight redownload,
second payload copy, full final weight-hash pass, or final activation replay
was used. Inference integration and behavioral validation remain deferred.

`model.safetensors.index.json` maps each tensor name to its shard filename.
The `model-*`, `ple-1-*`, and `ple-14-*` files form one indexed checkpoint,
not separate models. Each PLE has one scale tensor (~3.07 GB) and one encoded
weight tensor (~98.3 GB), each in its own file. The 5 GB shard target is soft:
individual tensors are not split. This file split is packaging, not a GPU
upload requirement. Swap both files for a PLE, retaining a matching index and
representation metadata; never modify shared hardlinked payloads in place.

## Detached one-shot entry point

After preparing a manifest and qualifying its immutable coordinator image:

```bash
python3 quantization/launch_quantization.py /absolute/run-root/production-manifest.json \
  --image sha256:QUALIFIED_IMAGE_ID \
  --hf-home /home/tj/.cache/huggingface
```

The launcher returns the container ID and persistent log path. Closing the chat
or shell does not stop the container. The image executes quantization, structural
export checks, public upload to
`wrldsuksgo2mars/DeepSeek-V4.1-EXL3-K3.25-v1`, then standard HF cache materialization.
There is no private staging repository, final replay or behavioral evaluation.
Inference-engine integration and behavioral/tool-call testing happen later.

The manifest must enable `publication`, pin both RTX UUIDs/image/preflight IDs
and all four Spark endpoints, and point to the passed source/input attestations
and unchanged calibration corpus. Put manifest, corpus, attestations, worker
token, journal and export state under `run_root`. Store the export under
`HF_HOME/ds41rt-exports/`, outside `HF_HOME/hub`; both share one container mount
so materialization can hard-link the weights. The source model's cache directory
is overmounted read-only. The existing HF credential is mounted read-only and is
never embedded in the manifest or printed. Do not edit hard-linked model files
in place after materialization.

## Monitoring and recovery

- `run_root/runtime-events.jsonl`: durable progress, block timing and memory.
- `run_root/attempts/<id>.log`: complete attempt stdout/stderr, including errors.
- `run_root/attempts/<id>-launch.json`: exact launch intent and pinned image.
- `run_root/attempts/<id>-container.json`: returned Docker container ID.
- `run_root/attempts/<id>-exit.json`: normal completion or caught exception.
  A kill/OOM can prevent this last file; inspect Docker state as the authority.
- `export_state/publication-complete.json`: verified upload commit and local
  snapshot path, written only after all publication stages succeed.

On failure, inspect the log and Docker exit/OOM state, fix the cause, and rerun
the same launch command with `--resume`. No automatic job restart is configured.
Exception: the completed v1 needed the publication-only recovery below; do not
resume its original coordinator after that metadata revision.
The launcher refuses to create another coordinator while a prior attempt is
nonterminal. An ambiguous launch error leaves the exact container name in its
durable launch intent; inspect that name before retrying. Old containers/logs
are retained, never automatically removed.

Recovery identities deliberately reject unreviewed changes to the corpus, model,
recipe, topology or qualified code. A code/image change during recovery requires
an explicit compatibility review and any necessary identity migration; do not
simply bypass the checks. Completed layer inputs and intermediate payloads are
retired only after durable downstream commitment. Final main states survive only
until dSpark input handoff, and final dSpark states are not kept for replay.

After several production layers commit with stable memory, monitor every
30 minutes and report current progress plus completion ETA.

The continuous-search repair uses the explicit
`ds41rt-continuous-search-recovery-v1` manifest authorization documented in
`PLAN.md`. Keep its referenced previous manifest and qualification report.
Old assignment files and completed candidates remain intact; new assignments
are recorded separately in `search-assignments-continuous-v1.json`. Only
unfinished searches may be recomputed. Resume the same repaired manifest with
`--resume`; do not replace the old assignment directory or relabel old results.

## v1 publication-only recovery record

The original coordinator exited after the public commit because Hugging Face
appended two JSON LFS rules to `.gitattributes`. All other 93 remote files
matched the upload receipts exactly. `recover_publication.py` verified the
pinned public commit, accepted only those exact appended rules, updated the
small local attributes file, revalidated the export, and materialized the cache.
It performs no GPU work, weight hashing, retransmission, or remote writes.
Future exports predeclare the rules and assign cache ownership to the run user.

The executed recovery command (already completed; not a pending step) was:

```bash
docker run --name ds41rt-publication-recovery --network host \
  -v /home/tj/Developer/ds41rt/quantization:/recovery-code:ro \
  -v /home/tj/.cache/ds41rt/quantization/deepseek-v41-exl3-k325-v1:/home/tj/.cache/ds41rt/quantization/deepseek-v41-exl3-k325-v1 \
  -v /home/tj/.cache/huggingface:/home/tj/.cache/huggingface \
  -e HF_TOKEN_PATH=/home/tj/.cache/huggingface/token \
  -e PYTHONPATH=/recovery-code:/opt/ds41rt/third_party/gptqmodel \
  --entrypoint /opt/glmrt/quant-venv/bin/python \
  sha256:bdd15949d70120fa42e4f9188da727f288ca0617e58b04f4ec97fdc5aa6f88f9 \
  /recovery-code/recover_publication.py \
  /home/tj/.cache/ds41rt/quantization/deepseek-v41-exl3-k325-v1/production-continuous-manifest.json \
  --commit 076c0dcf88436e1d6f69ca50f3557c307439f2ef
```

Under that run root, `export-state/hub-json-lfs-recovery-v1/` holds the explicit
authorization, revised export plan and passing structural validation. Original
plans and preupload receipts remain intact. Standard `upload-complete.json`,
`cache-complete.json` and `publication-complete.json` are in `export-state/`.
`reports/publication-recovery.log`, `reports/final-cache-audit.log` and
`reports/final-component-tests.log` record successful recovery, the network-free
94-file hardlink/readability audit, and 81 passing component tests respectively.
