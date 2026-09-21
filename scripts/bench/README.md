# DS41RT benchmark and report scaffolding

This directory holds the CPU-only plumbing that turns recorded bench output into
reports without ever inventing a number. It owns **no** engine code: the
launch, model and kernel sources belong to their own owners.

## Pieces

| File | Role |
| --- | --- |
| `v9-report-manifest.schema.json` | JSON Schema for a report manifest: binds every arm to its exact checkpoint, image digests, config hash, runtime flags and raw files. |
| `official-v9-campaign.md` | The command matrix and provenance checklist the campaign follows. |
| `tp6-candidate-flags.md` | Candidate TP6xEP1 flags matched to the Official baseline, cost-policy isolation, live network/provisioning findings. |
| `candidate-tp6-{1x,2x}-official-match.config` | CPU-only candidate configs: 1x 5-local/35-remote and 2x 20/20, KV auto, six ranks, `LANE_A` only. |
| `../tests/test_v9_candidate_configs.py` | CPU-only tests that load the candidate configs through `release-common.sh` and check the resolved geometry/dials. |
| `run-v9-campaign.sh` | **Planner / checker / renderer only — not a benchmark executor.** `plan` prints the planned arm matrix (a plan, not a measurement); `check` validates a manifest and artifact presence; `render` writes a report from already-recorded raw files. It never launches a model, container, Docker, SSH or benchmark. |
| `../render-ds41-v9-tp6-reports.py` | Renders a markdown report from a manifest. Missing file/field/arm renders as an em dash. |
| `../tests/test_v9_tp6_reports.py` | CPU-only regression tests for the renderer (no Docker, GPU or network). |

Of the planned arms, only the two published-v8 TP4 official arms (`official-v8-1x-tp4`,
`official-v8-2x-tp4`) have been measured. Every other row in the plan is unmeasured
and must not be described as performed.

## Rules the renderer enforces

1. **No invented values.** Every cell comes from a named raw bench JSON. An
   absent file, missing key or unmeasured arm renders as `—`.
2. **No silent omissions.** A "Missing-data index" lists every declared artifact
   that was not present, and negative results from the manifest's `failures`
   list are rendered verbatim. Absence of a failure is explicitly *not* a pass.
3. **Identity is mandatory.** Checkpoint revision, both image digests, both image
   revisions, network rail speed and topology, per-arm config SHA-256 and
   runtime flags are required; a manifest missing them is rejected (exit 2).
4. **No delta column in headlines.** Arms are reported side by side with their
   own identity; within-arm repeat spread is shown separately.
5. **Component vs end-to-end stay separate.** The FFN kernel tiling table states
   it is a component microbenchmark, not serving throughput.

## Provenance fields to capture per arm

- checkpoint `model_id` + `revision` (the **model quant**), and the **container
  release tag + digest + engine revision** — these are different things and both
  are required.
- topology (`TP4xEP1`, `TP6xEP1`, `TP3xEP2`, ...), RTX count, Spark count.
- the resolved launch geometry read back from argv/log, not the intended config:
  RTX expert layers, first remote dispatch layer, remote layer count, global KV
  pool bytes, source pages, runtime headroom, dSpark draft width, cost profile.
- tiling sweep dimensions actually measured (width, capacity, rows) for both
  decode-sized and prefill-sized batches.
- warmup policy, repeats, nonce seed; the per-sample raw values are kept.
- the network rail actually used and its negotiated link speed, with the
  captured evidence path.

## Thresholds

Set `thresholds.basis` to where the numbers come from. The thresholds are **not**
a universal constant and must be re-stated per campaign from that campaign's own
repeat spread.

Observed spread on the published-v8 official baseline (median-of-3, weighted
nine-category decode): **1x 4.7%, 2x 7.1%**. Per-content-type three-sample
spreads reach **13–26%** (2x math 27.1%, hello 32.1%). Historical `v6→v7`
official-regression deltas of +1.7–3.0% were called noise, but that is a
different campaign on different days.

Consequence: a single `<5%` band is **not** defensible across layouts — the 2x
arm's own weighted spread already exceeds it. Report the measured spread beside
every claim and treat `5–10%` as unresolved, not "noise", when the within-arm
spread is of the same size. A single content-type claim needs `>15–20%`.
Counting is the most repeatable anchor (observed 5.0–5.5%).

## Usage

```bash
scripts/render-ds41-v9-tp6-reports.py \
  --manifest /path/to/manifest.json \
  --package /path/to/package-dir \
  --output docs/release-v9-tp6-performance.md
```

`--package` is the directory that relative `raw.*` paths resolve against
(defaults to the manifest's directory). The renderer is read-only apart from the
single `--output` file.
