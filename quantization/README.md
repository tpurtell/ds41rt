# DeepSeek V4.1 routed EXL3 K3.25

The authoritative recipe, user constraints and qualification evidence are in
[PLAN.md](PLAN.md). This directory contains our implementation; sister-project
scripts are references, not production dependencies.

## Completed v1 artifact

Public model: https://huggingface.co/wrldsuksgo2mars/DeepSeek-V4.1-EXL3-K3.25-v1

Current published revision: `cfd4ca1d1934a8e81dd2d7515598d4ce288e8b88`.
The standard local cache snapshot is
`/home/tj/.cache/huggingface/hub/models--wrldsuksgo2mars--DeepSeek-V4.1-EXL3-K3.25-v1/snapshots/cfd4ca1d1934a8e81dd2d7515598d4ce288e8b88`.
All 40 main and three dSpark blocks are complete: 47,232 routed projections,
11,808 at K4. All 94 published files (52 safetensors shards) are materialized
with hardlinks and were verified offline as uid 1000. No weight redownload,
second payload copy, full final weight-hash pass, or final activation replay
was used. Inference integration and behavioral validation remain deferred.

`model.safetensors.index.json` maps each tensor name to its shard filename.
All shards use `model-00001-of-00052.safetensors` through
`model-00052-of-00052.safetensors`. Shards 49–50 contain PLE 1 (scale, weight),
and 51–52 contain PLE 14 (scale, weight), isolated from other tensors.
Each PLE has one scale tensor (~3.07 GB) and one encoded
weight tensor (~98.3 GB), each in its own file. The 5 GB shard target is soft:
individual tensors are not split. This file split is packaging, not a GPU
upload requirement. Swap both files for a PLE, retaining a matching index and
representation metadata; never modify shared hardlinked payloads in place.

The original revision `076c0dcf88436e1d6f69ca50f3557c307439f2ef` remains intact.
The numbered revision was published with server-side shard copies/deletions and
one updated 13,398,375-byte index. No weight uploads or weight hashing occurred.
Both local snapshots reuse the same weight inodes. The numbered export is
`/home/tj/.cache/huggingface/ds41rt-exports/DeepSeek-V4.1-EXL3-K3.25-v1-numbered`.
The current receipts and revised plan are under the original run root's
`export-state/numbered-shards-v1/`; the older top-level receipts describe the
original revision and are deliberately preserved. The export planner now uses
global numbering for future runs too.

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

## Numbered-shard revision

`rename_shards.py` is the publication-only migration used for the current
revision. It prepares a separate hardlinked export, changes only the index,
validates all headers/configs/quotas and PLE isolation, and compares original
local fingerprints and remote blob receipts. Without `--publish`, it stops
after preparation. With `--publish`, it commits server-side copies, old-name
deletions, and the new index together against the pinned parent, then advances
the cache ref only after materializing the complete new snapshot. An interrupted
commit can be recovered only if the remote inventory matches exactly.

Arguments used with the same qualified image/mounts as the recovery above:

```text
/recovery-code/rename_shards.py
  /home/tj/.cache/ds41rt/quantization/deepseek-v41-exl3-k325-v1/production-continuous-manifest.json
  --source-state /home/tj/.cache/ds41rt/quantization/deepseek-v41-exl3-k325-v1/export-state/hub-json-lfs-recovery-v1
  --state /home/tj/.cache/ds41rt/quantization/deepseek-v41-exl3-k325-v1/export-state/numbered-shards-v1
  --output /home/tj/.cache/huggingface/ds41rt-exports/DeepSeek-V4.1-EXL3-K3.25-v1-numbered
  --parent 076c0dcf88436e1d6f69ca50f3557c307439f2ef
  --publish
```

The detached `ds41rt-numbered-shards-publish` container exited 0. Its Docker
logs retain the commit/upload record. `ds41rt-numbered-shards-offline-audit`
also exited 0: all 94 files resolve offline through `main` as uid 1000, all
52 weights share inodes/blob IDs with the old revision, and every old snapshot
fingerprint is unchanged. The updated component suite passed 83 tests.
