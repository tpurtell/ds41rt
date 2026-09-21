# Official v9 benchmark campaign: command matrix and report manifest

This is the command matrix behind the v9 official-image / TP6 report. It covers
the measured published-v8 TP4 baseline, the candidate-v9 TP4 bridge and the TP6
arms, plus the FFN kernel sweep. Every number that reaches a report must come
from a raw file named in a report manifest
(`scripts/bench/v9-report-manifest.schema.json`) and rendered by
`scripts/render-ds41-v9-tp6-reports.py`; nothing is transcribed by hand.

Scope of the tooling: `scripts/bench/run-v9-campaign.sh` is a **planner, checker
and renderer only**. It does not execute this matrix, launch a model or touch
hardware, Docker, SSH or any benchmark; its `plan` output is a plan, not a record
of measurements. Of the arms below, only `official-v8-1x-tp4` and
`official-v8-2x-tp4` have been measured. Hardware execution is a separate,
explicitly owned step.

## Rules

- **Identity, not labels.** Record the model quant (`model_id` + `revision`) and
  the container identity (`tag` + digest + engine revision) separately. "Official"
  means the official checkpoint quant, not a release tag.
- **Frozen launch.** Run every arm from an isolated git worktree or a frozen
  script copy, with a captured config whose SHA-256 is recorded, so concurrent
  source edits cannot silently change an arm. `runs/v8/serve.sh` is such a frozen
  launcher for 4-Spark arms.
- **Rails.** The native release path uses `LANE_A` only; `LANE_B` is not read by
  `run.sh`. Record the resolved peer addresses/GIDs actually used, not the config
  label. Do not assume dual lanes help: verify the negotiated per-rail link speed
  and whether A and B share one L2/uplink.
- **No cross-RTX speed attribution.** A 1-RTX arm and a 2-RTX arm are different
  deployments; compare each against its own baseline only.
- **Warmup then repeats.** One shape warmup, then three timed samples per cell;
  keep every sample and report median and spread. For paired A/B use an
  interleaved (ABBA/BAAB) order.
- **Thresholds are per campaign, not universal.** Do not repeat a blanket `<5%`
  noise claim. The published-v8 official baseline's own weighted-decode repeat
  spread was **4.7% (1x)** and **7.1% (2x)**; per-content-type three-sample
  spreads reached 13–26%. So `5–10%` is *unresolved*, not "noise", whenever the
  within-arm spread is comparable, and a single content-type claim needs
  `>15–20%`. Every claim states the spread it was compared against. Counting is
  the most repeatable anchor (5.0–5.5%).
- **Completed is not passed.** Report timed samples, serving-completed samples and
  `objective passed / objective assessed` separately. An unassessed sample is not
  a pass: an arm with 30 timed samples and objective checks on 15 of them is
  "30 completed, 15/15 assessed passed", never "30/30 passed".
- **Keep failures.** A rejected candidate, an incomplete matrix, a failed
  completion check and a misconfigured arm are recorded, never dropped.

## Arms

| Arm | Quant | Topology | RTX | Sparks | Purpose |
| --- | --- | --- | ---: | ---: | --- |
| `official-v8-1x-tp4` | official | TP4xEP1 | 1 | 4 | measured baseline |
| `official-v8-2x-tp4` | official | TP4xEP1 | 2 | 4 | measured baseline |
| `official-v9-1x-tp4` | official | TP4xEP1 | 1 | 4 | bridge: attributes code revision at fixed rank count |
| `official-v9-2x-tp4` | official | TP4xEP1 | 2 | 4 | bridge |
| `official-v9-1x-tp6` | official | TP6xEP1 | 1 | 6 | candidate |
| `official-v9-2x-tp6` | official | TP6xEP1 | 2 | 6 | candidate |
| `kernel-tp4-vs-tp6` | official | TP4xEP1 / TP6xEP1 | 1 | 4 / 6 | FFN component sweep |

The TP6 arms must reproduce the primary 1-RTX placement exactly (5 local / 35
remote at the historical geometry) unless `auto` with identical flags provably
produces a different documented boundary; an all-40-remote side arm is a
separate, clearly labelled optimisation, never a silent replacement.

## Exact commands

Common: `SNAP=$HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277`, `PY=./.venv/bin/python`, `URL=http://127.0.0.1:8000`.

### 1. Launch an arm (frozen worktree, captured config)

```bash
# 4-Spark published-image launcher (boundary discovered, not hardcoded):
runs/v8/serve.sh <config> start        # config selects quant; for official use the release run.sh
# 4/6-Spark release launcher from a frozen worktree:
./run.sh --config <captured>.config    # SPARK_TP/SPARK_EP select TP6xEP1 once supported
```

Capture before measuring: `docker inspect` argv for coordinator and every worker,
the coordinator placement/residency/pool log lines, `nvidia-smi` per GPU, and the
four worker logs (remember `run.sh` does **not** forward `RUST_LOG` to workers;
start a worker directly with `-e RUST_LOG=info` if the allocated layer count is
needed).

### 2. Correctness before numbers

```bash
scripts/api-smoke.sh "$URL"
scripts/api-constrained-smoke.sh "$URL"     # sets thinking disabled; see its header note
scripts/qualify-ds41-native-api.py          # lifecycle/stream/tools as required
```

### 3. Nine-category + counting decode (headline source)

```bash
"$PY" scripts/bench-ds41-release-decode.py --base-url "$URL" \
  --model deepseek-ai/DeepSeek-V4.1-Flash --tokenizer "$SNAP/tokenizer.json" \
  --label <arm> --repeats 3 --nonce-seed 61001 --include-counting --output <raw>/decode.json
```

### 4. Prefill matrix (30 cells, best-prefill source)

```bash
"$PY" scripts/bench-ds41-release-prefill-matrix.py --base-url "$URL" \
  --tokenizer "$SNAP/tokenizer.json" --label <arm> --repeats 3 --output <raw>/prefill.json
```

### 5. Retained-context decode and its 2K control

```bash
"$PY" scripts/bench-ds41-release-retained-decode.py --base-url "$URL" \
  --tokenizer "$SNAP/tokenizer.json" --label <arm> --output <raw>/retained.json
```

### 6. Concurrency scaling C1..C16

```bash
for case in counting code topic; do
  "$PY" scripts/bench-ds41-concurrent-api.py --base-url "$URL" \
    --output <raw>/concurrency-$case.json --concurrency 1 2 4 8 16 \
    --repeats 3 --label <arm> --case "$case"
done
```

### 7. Mixed traffic and target-only control

```bash
"$PY" scripts/bench-real-full-mixed-concurrency.py --base-url "$URL" --output <raw>/mixed.json ...
"$PY" scripts/bench-ds41-release-decode.py ... --case code ...   # with --no-dspark launch for target-only
```

### 8. FFN kernel tiling sweep (component, not serving throughput)

```bash
python python/tools/bench_tp_ep_kernel.py \
  --operands checkpoint --snapshot "$SNAP" --expect-revision dba1be0a40aa45a94ad051997016db3960a90277 \
  --layer 20 --topologies tp4 --ep-degree 1 --widths 64,128,192 \
  --rows 1,8,16,64 --capacity 80 --experts 384 --lpt --weight-cost 1.0 \
  --tile-cost 0 --tile-rows 16 --samples 30 --replays 10 --warmup 5 --cold \
  --flush-bytes 268435456 --amortised --max-group --output <raw>/kernel-tp4.json
```

Valid dimensions (observed): rows 1/8/16/64 for decode-sized batches and
256/1024/4096 for prefill-sized; capacity 1 with M1, otherwise the AOT capacities
1/16/80/256/1024/4096; widths 64/128/192, plus the TP6 storage width that the
exporter actually emits. Cover both decode and prefill batch sizes, and report the
selection criterion and every losing cell.

### 9. Acceptance / proposal counts (evidence, not throughput)

```bash
RUST_LOG='info,ds41rt::timing=debug,ds41rt::cost_model=debug' ./run.sh --config <captured>.config
# then drive a short counting request and read "native scheduler round"
# proposed=/accepted=/emitted= and the "placement-aware adaptive costs loaded" line.
```

Debug logging perturbs timing: this pass is evidence-only and must not be
reported as performance. Record the resolved cost mode/profile explicitly rather
than inferring it from the `profile=None` input path.

## Report manifest

Build a manifest with one entry per arm:

```json
{
  "release": "v9",
  "checkpoint": {"model_id": "deepseek-ai/DeepSeek-V4.1-Flash", "revision": "<40 hex>", "quant": "official"},
  "images": {
    "coordinator": {"tag": "ghcr.io/tpurtell/ds41rt-coordinator:v9", "digest": "sha256:<64 hex>", "revision": "<40 hex>"},
    "spark_expert": {"tag": "ghcr.io/tpurtell/ds41rt-spark-expert:v9", "digest": "sha256:<64 hex>", "revision": "<40 hex>"}
  },
  "network": {"spark_link_mbps": 100000, "rails": "A-only", "evidence": "<path>", "dual_lane_note": "<what was and was not tested>"},
  "thresholds": {"noise_percent": 5, "weak_percent": 10, "credible_percent": 10, "basis": "<where they come from>"},
  "arms": [
    {"id": "official-v8-1x-tp4", "quant": "official", "topology": "TP4xEP1", "rtx": 1, "sparks": 4,
     "config_sha256": "<64 hex>", "runtime_flags": {"rtx_expert_layers": 5, "remote_dispatch_layers": 35},
     "warmup": "shape warmup", "repeats": 3,
     "raw": {"decode": "raw/decode.json", "prefill": null},
     "failures": [{"what": "api-constrained-smoke", "observed": "RC=1 with empty content under default thinking", "evidence": "raw/api-constrained-smoke.txt"}]}
  ]
}
```

Render:

```bash
scripts/render-ds41-v9-tp6-reports.py --manifest manifest.json --package <dir> --output <report>.md
```

## Provenance checklist per arm

- [ ] git revision of the launcher and the engine revision label of both images
- [ ] both image digests, checkpoint `model_id` + `revision`
- [ ] captured config + SHA-256, resolved argv for coordinator and workers
- [ ] RTX local layers, first remote dispatch layer, remote layer count, and the
      worker-allocated layer count (separately)
- [ ] global KV pool bytes, source pages, runtime headroom, host cache bytes
- [ ] dSpark draft width, resolved cost mode/profile
- [ ] promoted/accelerated counters (proposed/accepted/emitted) where claimed
- [ ] warmup policy, repeats, nonce seed, corpus/tokenizer SHA-256
- [ ] network rail(s), negotiated link speed, resolved peers/GIDs, evidence path
- [ ] failures and incomplete matrices, retained verbatim
