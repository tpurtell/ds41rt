# v10 TP3 campaign: command matrix and report manifest

This is the command matrix behind the two v10 TP3 reports:

- [`docs/release-v10-tp3-official-1x-3spark.md`](../../docs/release-v10-tp3-official-1x-3spark.md)
  — one RTX + three Sparks, official native checkpoint, explicit `TP3xEP1`.
- [`docs/release-v10-tp3-exl3-compact-1x-3spark.md`](../../docs/release-v10-tp3-exl3-compact-1x-3spark.md)
  — one RTX under the compact 32 GiB reservation + three Sparks, checkpoint-native
  EXL3 K3.25 (`k34`), implicit compact TP3.
- [`docs/release-v10-tp3-campaign-status.md`](../../docs/release-v10-tp3-campaign-status.md)
  — the campaign status index.

Every number that reaches a report must come from a raw file named in a report
manifest (`scripts/bench/v10-tp3-report-manifest.schema.json`) and rendered by
`scripts/render-ds41-v10-tp3-reports.py`; nothing is transcribed by hand.

Scope of the tooling: `scripts/bench/run-v10-tp3-campaign.sh` is a **planner,
checker and renderer only**. It does not execute this matrix, launch a model or
touch hardware, Docker, SSH or any benchmark; its `plan` output is a plan, not a
record of measurements. Hardware execution is a separate, explicitly owned step.

## Status

**NOT EXECUTED.** The v10 image pair
(`ghcr.io/tpurtell/ds41rt-coordinator:v10` / `ds41rt-spark-expert:v10`) is not
published, so no arm has been launched and no raw manifest exists. Every family
in both reports renders as `PENDING` until raw manifests arrive. This document
describes the matrix that will produce them.

## Rules

- **Identity, not labels.** Record the model quant (`model_id` + `revision`, plus
  the EXL3 `family`/`bits`) and the container identity (`tag` + digest + engine
  revision) separately. "Official" means the official checkpoint quant, not a
  release tag.
- **Frozen launch.** Run every arm from an isolated git worktree or a frozen
  script copy, with a captured config whose SHA-256 is recorded, so concurrent
  source edits cannot silently change an arm.
- **Two arms, two identities.** The compact EXL3 arm's 32 GiB reservation, 2 GiB
  KV pool and 256-token prefill bound are part of its identity. Never compare its
  throughput to the uncapped native arm as though only the quant differed.
- **Rails.** The native release path uses `LANE_A` only; `LANE_B` is not read by
  `run.sh`. Record the resolved peer addresses/GIDs actually used, not the config
  label.
- **No cross-RTX speed attribution.** A 1-RTX arm and a 2-RTX arm are different
  deployments; compare each against its own baseline only.
- **Warmup then repeats.** One shape warmup, then three timed samples per cell;
  keep every sample and report median and spread.
- **Thresholds are per campaign, not universal.** Every claim states the observed
  within-arm spread it was compared against.
- **Completed is not passed.** Report timed samples, serving-completed samples and
  `objective passed / objective assessed` separately. An unassessed sample is not
  a pass.
- **Pending is not zero.** A family with no raw file is `PENDING`, never a zero
  and never an inferred number.
- **Keep failures.** A rejected candidate, an incomplete matrix, a failed
  completion check and a misconfigured arm are recorded, never dropped.

## Arms

| Arm | Quant | Topology | RTX | Sparks | Config |
| --- | --- | --- | ---: | ---: | --- |
| `v10-native-tp3ep1` | official | `TP3xEP1` (explicit) | 1 | 3 | `examples/configs/tp3ep1-native.config` |
| `v10-exl3-compact-tp3` | exl3 `k34` | `TP3` (implicit) | 1 | 3 | `examples/configs/exl3-compact-tp3.config` |

The native arm pins `RTX_EXPERT_LAYERS=5`, so the RTX holds the first five routed
layers and each Spark starts at layer 5 to hold the other 35. The compact EXL3
arm leaves the boundary at the published compact default and requires the
`k34`-family `tp3-rank{0,1,2}` packages in the Spark image.

## Measured families and the derived record ledger

The expected counts below are **derived at check/render time** from the live
harness scripts' constants and argparse defaults, not asserted in prose. The
renderer prints the arithmetic next to the comparison.

| Family | Raw file(s) | Records | Derivation |
| --- | --- | ---: | --- |
| headline decode (dSpark) | `raw/decode.json` | 30 | 3 repeats x (9 weighted cases + counting) |
| prefill matrix | `raw/prefill.json` | 90 | 30 cells (5 bases x 6 suffixes) x 3 timed |
| retained prime contexts | `raw/retained-ctx-<n>.json`, `raw/retained.json` | 135 | 5 contexts x 9 cases x 3 repeats |
| retained 2K control | `raw/retained-control-2k.json` | 27 | 1 control context (2048) x 9 cases x 3 repeats |
| concurrency counting/code/topic | `raw/concurrency-<case>.json` | 45 | 3 cases x 5 levels (C1,C2,C4,C8,C16) x 3 repeats |
| mixed traffic | `raw/mixed.json` | 2 | adaptive-mixed batches at C4 + C16 |
| target-only decode | `raw/target-only.json` | 30 | 3 repeats x (9 + counting), dSpark off |
| **performance total** | | **359** | |
| startup / memory | `raw/memory.json` | — | diagnostics; excluded from the total |
| kernel / tile diagnostics | `raw/kernel-tp3.json` | — | component microbenchmark; excluded from the total |

`359 = 30 + 90 + 135 + 27 + 45 + 2 + 30`. Note the retained split: the
mechanical enumerator reports the prime contexts (5) and the 2K control (1)
separately, `135 + 27 = 162`, which is the same six-context total the native
lane's `summarize.py` reports as a single `162` (`6 contexts x 9 cases x 3
repeats`). Both decompositions agree; the split form is preferred here because a
failure at one context cannot then be hidden inside a larger number.

Tool-eval is **not** part of the 359: it is `3 runs x 88 scenarios = 264
scenario-runs` per arm, counted separately.

Warmups (`raw/warmup-*.json`), primes and lifecycle probes are stored and
excluded from the ledger.

### Expected-counts interoperability

The lanes already carry their own mechanically-derived expectation documents
(`runs/v10-exl3-tp3/enumerate-expected.py` → `expected-counts.json`). The
renderer can ingest one and compare it shape by shape against its own derivation,
and can emit its derivation in the same shape:

```bash
scripts/render-ds41-v10-tp3-reports.py --manifest M.json --package DIR \
  --check --expected-counts runs/v10-exl3-tp3/expected-counts.json
scripts/render-ds41-v10-tp3-reports.py --manifest M.json \
  --check --write-expected-counts runs/v10-tp3/expected-counts.json
```

A shape that differs is printed as `DISCREPANCY` and, in `--check`, makes the
reconciliation an error rather than being hidden behind a matching total. The
`retained` shape is compared on the basis the source document declares (its own
`contexts` list): a six-context lump (162) and the split five prime contexts
(135) plus the 2K control (27) reconcile to the same total and are **not**
reported as a disagreement.

### The 359 reconciliation

The requirement carried a stated breakdown of
`decode 30, prefill 90, retained 162, concurrency 45, mixed 2, target-only 30 =
359`. The mechanical derivation reproduces that total exactly:
`30 + 90 + 135 + 27 + 45 + 2 + 30 = 359`, where `135 + 27 = 162` is the same
retained work expressed as prime contexts plus the separate 2K control. There is
**no arithmetic discrepancy**: the two forms differ only in whether the 2K
control is lumped into the retained family or reported as its own family. This
report splits it, because a control failure must not be averaged into a
long-context number. The renderer states both forms and the total rather than
asserting a bare `359`.

All five correctness/provenance gates (placement, memory headroom, graph/replay,
service API, model identity) are recorded separately from the 359 and are not
performance records.

## Exact commands

Common: `SNAP=$HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277`,
`PY=./.venv/bin/python`, `URL=http://127.0.0.1:8000`, `RAW=runs/v10-tp3/<arm>`.

### 0. Frozen worktree and captured config

```bash
git worktree add ~/.cache/ds41rt/builds/v10-tp3-<arm> -b v10-tp3-<arm> dev
sha256sum <captured>.config        # -> manifest arms[].config_sha256
```

### 1. Launch an arm (identity captured before any number)

```bash
# native TP3xEP1
./run.sh --config examples/configs/tp3ep1-native.config
# compact EXL3 TP3 (implicit: SPARK_COUNT=3, no SPARK_TP/SPARK_EP)
./run.sh --config examples/configs/exl3-compact-tp3.config
```

Capture: `docker inspect` argv for coordinator and every worker; the coordinator
placement/residency/pool log lines; `docker exec <coordinator> nvidia-smi
--query-gpu=index,name,memory.total,power.limit,clocks.max.memory --format=csv`;
`docker image inspect` digests and the `org.opencontainers.image.revision` /
`io.ds41rt.sparkinfer.revision` labels for both roles; each worker's loaded layer
count (remember `run.sh` does **not** forward `RUST_LOG` to workers).

### 2. Correctness before numbers

```bash
scripts/api-smoke.sh "$URL"
scripts/api-constrained-smoke.sh "$URL"     # sets thinking disabled; see its header note
"$PY" scripts/qualify-ds41-native-api.py --base-url "$URL" --output "$RAW/native-api.json"
```

### 3. Headline decode (nine weighted categories + counting)

```bash
"$PY" scripts/bench-ds41-release-decode.py --base-url "$URL" \
  --model deepseek-ai/DeepSeek-V4.1-Flash --tokenizer "$SNAP/tokenizer.json" \
  --label <arm> --repeats 3 --nonce-seed 79001 --include-counting \
  --output "$RAW/decode.json"
```

### 4. Prefill matrix (30 cells)

```bash
"$PY" scripts/bench-ds41-release-prefill-matrix.py --base-url "$URL" \
  --tokenizer "$SNAP/tokenizer.json" --context-file "$CTX" \
  --label <arm> --repeats 3 --warmups 1 --output "$RAW/prefill.json"
```

`--context-file` is **required**; the v10 planning note that omitted it is stale.

### 5. Retained-context decode and its separate 2K control

```bash
for context in 0 32768 65536 131072 262144; do
  "$PY" scripts/bench-ds41-release-retained-decode.py --base-url "$URL" \
    --tokenizer "$SNAP/tokenizer.json" --context-file "$CTX" --label <arm> \
    --context-tag <arm> --context "$context" --repeats 3 \
    --output "$RAW/retained-ctx-$context.json"
done
"$PY" scripts/bench-ds41-release-retained-decode.py --base-url "$URL" \
  --tokenizer "$SNAP/tokenizer.json" --context-file "$CTX" --label <arm>-control-2k \
  --context 2048 --repeats 3 --output "$RAW/retained-control-2k.json"
```

The retained flag is `--context` (repeatable), not `--prompt-tokens`. Each
context is its own invocation so a failure at one context cannot suppress
another.

### 6. Concurrency scaling C1..C16

```bash
for case in counting code topic; do
  "$PY" scripts/bench-ds41-concurrent-api.py --base-url "$URL" \
    --output "$RAW/concurrency-$case.json" --concurrency 1 2 4 8 16 \
    --repeats 3 --label <arm> --case "$case" --nonce "v10-tp3-$case"
done
```

### 7. Mixed traffic and target-only control

```bash
"$PY" scripts/bench-ds41-adaptive-mixed.py --base-url "$URL" \
  --tokenizer "$SNAP/tokenizer.json" --nonce-seed 79002 \
  --concurrency 4 16 --output "$RAW/mixed.json"

# target-only: stop, relaunch the same config with dSpark disabled, re-measure
./stop.sh --config <captured>.config
./run.sh --config <captured>.config --no-dspark
"$PY" scripts/bench-ds41-release-decode.py --base-url "$URL" \
  --model deepseek-ai/DeepSeek-V4.1-Flash --tokenizer "$SNAP/tokenizer.json" \
  --label <arm>-target --repeats 3 --nonce-seed 79001 --include-counting \
  --output "$RAW/target-only.json"
```

The accepted battery uses `scripts/bench-ds41-adaptive-mixed.py`
(`--concurrency`, `--nonce-seed`). `scripts/bench-real-full-mixed-concurrency.py`
has `--warmups`, no `--deadline-seconds`, and takes `--max-tokens` as a comma
list; the planning note that used the latter is stale.

### 8. Startup / memory evidence

```bash
docker logs ds41rt-coordinator > "$LOGS/coordinator.log" 2>&1
grep -iE 'device_occupied|device_budget|reservation|kv|headroom|pool' \
  "$LOGS/coordinator.log" | tail -40
for host in ostrich dodo emu; do
  ssh -o BatchMode=yes "$host" "docker logs ds41rt-spark-expert-$host-19441 2>&1" \
    > "$LOGS/spark-$host.log" 2>&1
done
```

Record the resolved `MEMORY_RESERVATION`, `KV_POOL_SIZE` and
`PREFILL_BATCH_TOKENS` for the compact arm, plus the actual reservation,
occupancy and KV figures the services report. Weight-only admission arithmetic
is admission arithmetic, not a runtime-fit claim.

### 9. FFN kernel / tile diagnostics (component, not serving throughput)

```bash
python3 python/tools/bench_tp_ep_kernel.py --operands checkpoint --snapshot "$SNAP" \
  --expect-revision dba1be0a40aa45a94ad051997016db3960a90277 --layer 20 \
  --topologies tp3 --ep-degree 1 --widths 64,128,192 \
  --rows 1,8,16,64,256,1024,4096 --capacity 80 --experts 384 --lpt \
  --weight-cost 1.0 --tile-cost 0 --tile-rows 16 --samples 30 --replays 10 \
  --warmup 5 --cold --flush-bytes 268435456 --amortised --max-group \
  --output "$RAW/kernel-tp3.json"
```

TP3 slice width 192 is the built role; cover both decode-sized (1/8/16/64) and
prefill-sized (256/1024/4096) batches and report the selection criterion and
every losing cell.

### 10. Tool-call evaluation (three runs per arm, 176 scored items each)

```bash
"$PY" scripts/qualify-ds41-tool-eval.py \
  --base-url "$URL" \
  --output-dir "$RAW/tool-eval" \
  --runs 3 --parallel 16 \
  --reference-date 2026-09-21
```

**Corrected flags.** The v9 planning note's
`--thinking high --evidence-dir <dir>` does not exist. Thinking and effort are
fixed **enabled / high inside the script**; it is not a flag. The real surface is
`--base-url`, `--output-dir` (required), `--runs`, `--parallel`, `--timeout`,
`--max-turns`, `--max-tokens`, `--reference-date` (required), `--label` and
`--collect-only`. `--runs 3` is the qualified-release setting. Each run writes
`run-NN/tool-eval.json` plus its SQLite traces, and the script exports
`run-NN/summary.json` and the combined `summaries.json`; the report reads those
for per-run scores and for the non-pass scenario detail.

### 11. Provenance counters (evidence only, perturbs timing)

```bash
RUST_LOG='info,ds41rt::timing=debug,ds41rt::cost_model=debug' \
  ./run.sh --config <captured>.config
```

Debug logging perturbs timing: this pass is evidence-only and must not be
reported as performance. Record the resolved cost mode/profile explicitly rather
than inferring it from the `profile=None` input path.

## Report manifest

Build one manifest per campaign with one entry per arm:

```json
{
  "release": "v10",
  "checkpoint": {
    "model_id": "deepseek-ai/DeepSeek-V4.1-Flash",
    "revision": "dba1be0a40aa45a94ad051997016db3960a90277",
    "quant": "official"
  },
  "accounting": {"repeats": 3, "prefill_warmups": 1, "tool_eval_runs": 3},
  "images": {
    "coordinator": {"tag": "ghcr.io/tpurtell/ds41rt-coordinator:v10", "digest": "sha256:<64 hex>", "revision": "<40 hex>"},
    "spark_expert": {"tag": "ghcr.io/tpurtell/ds41rt-spark-expert:v10", "digest": "sha256:<64 hex>", "revision": "<40 hex>"}
  },
  "network": {"spark_link_mbps": 100000, "rails": "A-only", "evidence": "<path>"},
  "thresholds": {"noise_percent": 5, "weak_percent": 10, "credible_percent": 10, "basis": "<where they come from>"},
  "arms": [
    {
      "id": "v10-native-tp3ep1",
      "quant": "official",
      "topology": "TP3xEP1",
      "topology_form": "explicit",
      "rtx": 1,
      "sparks": 3,
      "config_path": "examples/configs/tp3ep1-native.config",
      "config_sha256": "<64 hex>",
      "ceiling": {"memory_reservation": null, "kv_pool_size": null, "prefill_batch_tokens": null},
      "runtime_flags": {"rtx_expert_layers": 5, "remote_dispatch_layers": 35},
      "warmup": "shape warmup",
      "repeats": 3,
      "raw": {
        "decode": "raw/decode.json",
        "prefill": "raw/prefill.json",
        "retained": ["raw/retained-ctx-0.json", "raw/retained-ctx-32768.json"],
        "retained_control_2k": "raw/retained-control-2k.json",
        "concurrency_counting": "raw/concurrency-counting.json",
        "concurrency_code": "raw/concurrency-code.json",
        "concurrency_topic": "raw/concurrency-topic.json",
        "mixed": "raw/mixed.json",
        "target_only": "raw/target-only.json",
        "startup_memory": "raw/memory.json",
        "kernel_tiling": "raw/kernel-tp3.json"
      },
      "tool_eval": {"summaries": "raw/tool-eval/summaries.json", "dir": "raw/tool-eval", "runs": ["raw/tool-eval/run-01", "raw/tool-eval/run-02", "raw/tool-eval/run-03"]},
      "failures": [
        {"what": "api-constrained-smoke", "observed": "RC=1 with empty content under default thinking", "evidence": "logs/21-api-constrained.log"}
      ]
    },
    {
      "id": "v10-exl3-compact-tp3",
      "quant": "exl3",
      "topology": "TP3",
      "topology_form": "implicit",
      "rtx": 1,
      "sparks": 3,
      "config_path": "examples/configs/exl3-compact-tp3.config",
      "config_sha256": "<64 hex>",
      "ceiling": {
        "memory_reservation": "32GiB",
        "kv_pool_size": "2GiB",
        "prefill_batch_tokens": 256,
        "evidence": "logs/coordinator-memory.txt"
      },
      "raw": {"decode": "raw/decode.json"},
      "tool_eval": {"summaries": "raw/tool-eval/summaries.json", "dir": "raw/tool-eval"}
    }
  ]
}
```

`raw` paths may be relative to `--package` or absolute / `~`-prefixed; they are
expanded before use. A `null` or omitted entry renders as a `PENDING` cell.

Render and gate:

```bash
scripts/bench/run-v10-tp3-campaign.sh render --manifest manifest.json --package <dir> --output <report>.md
scripts/bench/run-v10-tp3-campaign.sh check  --manifest manifest.json --package <dir>
scripts/bench/run-v10-tp3-campaign.sh check  --manifest manifest.json --package <dir> --strict
```

`--strict` is the release gate. It fails (exit 2) while any measured family is
pending or short of its derived expected record count, or while the tool-eval
run/scenario-runs totals are short. A missing file that the manifest *declares*
is an error (exit 2) in every mode.

## Provenance checklist per arm

- [ ] git revision of the launcher and the engine revision label of both images
- [ ] both image digests, checkpoint `model_id` + `revision` (and EXL3 `family`/`bits`)
- [ ] captured config + SHA-256, resolved argv for coordinator and workers
- [ ] topology form recorded: explicit `TP3xEP1` vs implicit compact `TP3`
- [ ] RTX local layers, first remote dispatch layer, remote layer count and the
      worker-allocated layer count (separately)
- [ ] compact arm: resolved memory reservation / KV pool / prefill batch and the
      service-reported occupancy, not just the config intent
- [ ] global KV pool bytes, source pages, runtime headroom, host cache bytes
- [ ] dSpark draft width, resolved cost mode/profile
- [ ] promoted/accelerated counters (proposed/accepted/emitted) where claimed
- [ ] warmup policy, repeats, nonce seed, corpus/tokenizer SHA-256
- [ ] network rail(s), negotiated link speed, resolved peers/GIDs, evidence path
- [ ] tool-eval: three runs, per-run score, every non-pass scenario retained
- [ ] failures and incomplete matrices, retained verbatim

## Report rules

- A family with no raw file renders as `PENDING` with no number. A family that is
  partially present renders as `SHORT` with the observed count.
- Every failed or non-passing raw sample is listed with its family, case/context,
  repeat and reason; a capacity-gated record is marked as such and is never
  counted as a pass.
- The tool-eval section prints every run's own score and the non-pass scenario
  detail, and never averages the runs into one number.
- The expected/actual table prints the derivation of every expected count and the
  SHA-256 of every harness script the expectation was derived from.
