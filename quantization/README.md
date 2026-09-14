# DeepSeek V4.1 routed EXL3 K3.25

The authoritative recipe, user constraints and qualification evidence are in
[PLAN.md](PLAN.md). This directory contains our implementation; sister-project
scripts are references, not production dependencies.

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
