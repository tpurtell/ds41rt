# TP4EP1 vs TP2EP2 end-to-end qualification protocol

Status: **protocol + corpus + CPU-validated runner only. Not executed.** The
author ran no GPU, service, build or network step. The only executed parts are
CPU-only: the corpus schema check and the adversarial test suite
(`scripts/tests/test_tp_ep_e2e_protocol.py`, 26 passed).

Scope: **four Spark ranks only**, `TP4EP1` (control) vs `TP2EP2` (candidate),
both `2 RTX + 4 Spark`. This primary comparison is unchanged. `rhea` (global
rank 4) and `moa` (global rank 5) are now connected; per the user's latest
permission they may be used freely for benchmarking and development (the earlier
full-six-only reservation is superseded), and no images or official checkpoint
are present on them yet. Six-rank `TP3EP2`/`TP2EP3` remain available but
**unqualified**; the full-six extension has its own corpus, provisioned
images/checkpoint and matched metadata (section 12, `docs/cluster-hosts.md`).

Build environment: never build or stage under `/mnt/scratch` (read-only NTFS
driver bug). Use a unique `/home/tj/.cache/ds41rt/builds/<task>` path on
raptor's root NVMe and run `python3 scripts/assert-build-filesystem.py PATH...`
before direct builds; the release/WIP artifact helpers enforce the same check.
This protocol builds nothing.

## 1. Why the two arms are not expected to be bit-identical

`TP4EP1` sums four 576-wide rank shards of one group; `TP2EP2` has each route
computed by one TP2 group, so the ordered FP32 sum order and the per-rank FP8
intermediate quantization granularity differ. The in-tree kernel test states
this directly: cross-topology comparison is tolerance-based and never bit-exact
(`python/tests/test_v41_sentinel_masking.py`). Text divergence is therefore
legitimate and handled by the review gate in section 8 — never by weakening a
content check, and never auto-qualified.

## 2. Prerequisites and arm launch

```bash
# Read-only identity / dry-run for both arms (no service change):
./run.sh --config examples/configs/tp4ep1-explicit-native.config --dry-run
./run.sh --config examples/configs/tp2ep2-native.config --dry-run

# Launch one arm at a time (same commands without --dry-run), then:
scripts/api-smoke.sh http://127.0.0.1:8000
```

`--dry-run` prints the release identity, resolved RTX layout, Spark rank map,
resolved boundary, weight-only admission and prefill/expert capacity. It does
**not** prove feasibility (`examples/configs/README.md`).

## 3. Structured matched identity

The engine probe's `system_fingerprint` proves only that both arms are the same
engine build. It says nothing about the RTX boundary, KV pool, speculation or
power. Capture a structured `startup_metadata` JSON per arm from real artifacts
(image labels, the expert TP manifest written by
`scripts/write-v41-expert-tp-manifest.py`, and the resolved launcher values), and
pass it to `run`. Use the tracked template
`scripts/fixtures/tp-ep-e2e-startup-metadata.template.json`; it records where
each field comes from. Required fields and rules:

- **Source evidence:** `source_commit` (image label
  `org.opencontainers.image.revision`), `source_artifact` (coordinator/expert
  image digests plus `io.ds41rt.source-manifest.sha256`; a full manifest string
  is acceptable), `source_snapshots` (`daemon`, `native` — must match),
  `sparkinfer_revision` (`io.ds41rt.sparkinfer.revision`).
- **Expert families:** `expert_tp_manifests` maps a family (`spark`,
  `spark_tp2`, ...) to `{tp_degree, intermediate, native_library_sha256}`. Every
  baseline family must appear in the candidate with an **equal**
  `native_library_sha256`; the candidate may add the family its role selects
  (TP2EP2 adds `spark_tp2`). Names come from `io.ds41rt.v41.spark_tp_roles`.
- **Applied controls:** `coordinator_argv` is the single applied coordinator
  argv; `worker_argv` is one argv per physical rank. `compare` parses the
  required flags and compares **values**, not just presence:
  `--prefill-batch-tokens`, `--rtx-gpus`, `--prefix-cache-entries` and
  `--spark-tp`/`--spark-ep` against the metadata; `--dspark` presence against
  the `dspark` value; the effective draft limit (`--dspark-draft-limit`, or the
  5/7 daemon default derived from `--rtx-gpus`); `--kv-pool-size` when present;
  and `--rtx-expert-layers` when present must equal
  `resolved_rtx_expert_layers`. Each worker rank must have
  `--world == spark_tp * spark_ep`, a valid `--rank`, and `--capacity` equal to
  `worker_capacity`; `worker_argv` must have exactly `world` entries whose ranks
  are unique and cover `0..TP*EP-1` (duplicate or extra rank argvs are
  rejected); `--first-layer` / `--device-budget-bytes` must match the metadata. Duplicate known flags are rejected. A config that was read but never
  passed therefore cannot be reported as matched, and a mismatched value such as
  `--prefill-batch-tokens 4096` against metadata `2048` fails tier A.
- **Controls matched:** `rtx_gpus`, `resolved_rtx_expert_layers`,
  `spark_first_layer`, `kv_pool`, `prefix_cache_entries`,
  `prefill_batch_tokens`, `worker_capacity`, `concurrency`,
  `max_context_tokens`, `max_output_tokens`, `dspark`, `dspark_draft_limit`,
  `spark_device_budget_bytes`, `spark_count`, `model`, `revision`,
  `power_caps_watts`. `concurrency`, `max_context_tokens` and
  `max_output_tokens` are compared by exact value against
  `--concurrency` / `--max-context-tokens` / `--max-output-tokens`, so two arms
  cannot record 16 while running 8. `host_cache_bytes`, `http_queue_depth` and
  `http_queue_wait_ms` are **recorded for manual comparison** until a later
  validator change gates them.
- **Requested vs resolved boundary:** `requested_rtx_expert_layers` is `"auto"`
  (when `--rtx-expert-layers` is absent) or an explicit integer, and is
  **recorded, not matched**. `resolved_rtx_expert_layers` is the **actual
  integer** from the coordinator `plan.json` and must match across arms;
  `spark_first_layer` must equal `min(resolved_rtx_expert_layers, 39)`. If the
  argument is present its value must equal the resolved actual. The literal
  `"auto"` is never accepted in the resolved field, so two arms that resolve to
  the same real boundary compare as matched even though both requested `auto`.
- **Worker capacity:** `worker_capacity` must be one of `1/16/80/256/1024/4096`,
  must equal every worker `--capacity` value, and must be at least the launcher
  capacity for `prefill_batch_tokens` (`<=80 → 80`, `<=256 → 256`,
  `<=1024 → 1024`, else `4096`), so an unsupported value or one below the
  admitted prefill bound cannot pass.
- **Recorded, not matched:** `spark_tp`, `spark_ep`, `spark_artifact_roles`
  (TP4EP1 uses the `spark` role, TP2EP2 adds `spark_tp2`) and
  `release_identity_full` (the launcher fingerprint embeds the topology). Both
  values are reported.

Not every field is a coordinator flag, and this document does not pretend
otherwise: `resolved_rtx_expert_layers`/`spark_first_layer` are the runtime
values from the published `plan.json`, `spark_device_budget_bytes` is the worker
`expertd-native --device-budget-bytes`, `power_caps_watts` is a host value, and
the source fields are image labels/manifest hashes.
`SPARK_REDUCTION_MIN_ROWS` is deliberately absent: it is a `run-wip`-only key
and is **not applied** on the release `serve-native` path. `source_snapshots` is
source-tree identity (`daemon`, `native`) and must match; it is **not** the
artifact-role-dependent manifest hashes, so equal source plus the family
superset rule is what admits the candidate's added `spark_tp2` family.

Also match: greedy sampling (`temperature=0`, `thinking.type=disabled`), the
fixed prompt nonce `tp-ep-e2e-v1`, `C1,2,4,8,16` x 3 measured repeats plus one
warmup, and the checkpoint/format (official native only). Power is sampled per
arm (section 9).

## 4. Corpus

`scripts/fixtures/tp-ep-e2e-corpus.jsonl` (tracked) is one JSON object per line,
self-contained and offline:

- `corpus` header and `controls`;
- `matrix` (`C1,2,4,8,16`, `repeats: 3`, `warmup: 1`);
- four decode `case` records (`counting-1-64`, `code-merge-intervals`,
  `topic-virtual-memory`, `mixed-code-fix`);
- two `prefill` records for the separate prefill tool;
- `gates` describing tier A and the divergence policy;
- `future_scope` recording the user-permitted, unqualified six-rank availability
  and the build-filesystem rule (`docs/cluster-hosts.md`).

Content checks are real: `throughput_case` calls
`scripts/release_throughput_checks.py` `check_output`; `exact_normalized`
compares the counting sequence; `nonempty` enforces a minimum length;
`code_block_and_prose` requires a fenced block and non-empty prose. The
`code-merge-intervals` prompt now explicitly requests every condition the
existing `code` check enforces (one fenced `merge_intervals` block, type hints on
the parameter and return, a docstring, at least three asserts) without weakening
`scripts/release_semantic_quality.py`; a canonical valid answer passes and a
complete-but-wrong answer fails.

Budgets were measured with the checkpoint tokenizer
(`.../dba1be0a.../tokenizer.json`, CPU only): the `code` canonical answer is 199
tokens against a 512 budget, and `counting-1-64` renders as 127 tokens
(no space), 190 tokens (comma+space) or 127 tokens (newline), so its budget is
256. The C1 pilot server returned 191 tokens and passed under the 256 budget;
the 256 fix is the frozen case budget for full arms.

## 5. Runner

`scripts/qualify-ds41-tp-ep-e2e.py` reuses the real client
`scripts/qualify-ds41-native-api.py` and the real content checks unchanged. It
exists because `scripts/bench-ds41-concurrent-api.py` accepts only its own
hardcoded cases and drops per-event timings.

```bash
# CPU only, allowed before any launch:
scripts/run-with-python-env.sh python scripts/qualify-ds41-tp-ep-e2e.py validate
scripts/run-with-python-env.sh python -m pytest -q scripts/tests/test_tp_ep_e2e_protocol.py

# Per arm, with that arm running and ready (requires GPU/service):
scripts/run-with-python-env.sh python scripts/qualify-ds41-tp-ep-e2e.py \
  run --arm tp4ep1 --base-url http://127.0.0.1:8000 \
  --startup-metadata runs/tp-ep-e2e/startup-tp4ep1.json \
  --output runs/tp-ep-e2e/tp4ep1.json
scripts/run-with-python-env.sh python scripts/qualify-ds41-tp-ep-e2e.py \
  run --arm tp2ep2 --base-url http://127.0.0.1:8000 \
  --startup-metadata runs/tp-ep-e2e/startup-tp2ep2.json \
  --output runs/tp-ep-e2e/tp2ep2.json

# CPU only, after both arms:
scripts/run-with-python-env.sh python scripts/qualify-ds41-tp-ep-e2e.py \
  compare --control runs/tp-ep-e2e/tp4ep1.json \
  --candidate runs/tp-ep-e2e/tp2ep2.json \
  --review-evidence runs/tp-ep-e2e/review.json \
  --kernel-evidence runs/tp-ep-e2e/kernel-tp2ep2.json \
  --output runs/tp-ep-e2e/compare.json
```

`run` performs the same readiness probe as the existing smoke (non-streaming
`What is 2 + 2?` must answer `4`), records `system_fingerprint`, attaches each
case before its first request and rewrites the output after every repeat (so an
interruption keeps the in-progress case and its completed rows), retains every
row's raw SSE `events` and `usage`, and records every error and timeout
verbatim.

## 6. Measurements (separate, correctly named)

- **TTFT** — `ttft_ms` (milliseconds), time to the first *content* delta; it
  **excludes reasoning** (`ttft_excludes_reasoning: true`). `first_output_ms`
  includes reasoning, and `reasoning_ms = first_content - first_output`. Every
  latency field is in milliseconds.
- **Inter-chunk latency** — `inter_chunk` (`mean_ms`, `median_ms`, `p95_ms`,
  milliseconds), the gap between consecutive *SSE content events*. It is **not**
  per-token latency: one event may carry several tokens (dSpark),
  `content_chunks` counts events and not tokens, and `usage.completion_tokens`
  carries no per-token timestamps.
- **Decode throughput** — `per_stream_decode_tps` and `aggregate_decode_tps`
  (tokens/s) over each repeat's own concurrent span, then summarized (`n=3`).
  The client rate is `(completion_tokens - 1) / (finish - first_output)`, which
  assumes exactly one token in the first output chunk; every row carries
  `observed_decode_tokens_per_second_approximate: true` plus a basis string. It
  is **approximate with unknown bias, not a bound**: with `k > 1` tokens in the
  first output chunk the formula overestimates the remaining-token rate
  `(N-k)/span`, and SSE exposes no per-token timestamps to correct it. Repeats
  are never merged into one multi-period span (that would include idle gaps and
  bias the rate down). Never called TTFT or inter-chunk latency.
- **Prefill** — separate tool and separate report section (section 7); decode
  numbers are never derived from a cold-prefill request.

## 7. Prefill (separate)

```bash
scripts/run-with-python-env.sh python scripts/bench-ds41-prefill-api.py \
  --tokenizer <tokenizer.json> --context-file <code or filler file> \
  --target-url http://127.0.0.1:8000 --modes target \
  --filler-tokens 3950 16384 --count-to 20 --max-output-tokens 64 \
  --output runs/tp-ep-e2e/prefill-tp4ep1.json
# repeat with --output runs/tp-ep-e2e/prefill-tp2ep2.json
```

`scripts/bench-ds41-release-prefill-matrix.py` covers the wider matrix; keep it
separate from decode as well.

## 8. Gates

- **Tier A (hard).** Both arms carry `startup_metadata` and match on every
  required identity field (section 3). Both arms cover the full corpus case set
  and the full `concurrency x repeat` matrix in both directions with no holes,
  duplicates or extra cases; the summed matched request count equals
  `sum(concurrency) * repeats * cases`. Engine fingerprints match. The readiness
  probe passes. Every request completes without error or timeout, passes its
  content check, and within-arm greedy output is identical across the three
  repeats.
- **Divergence.** Identical greedy text yields `gate_result: text_exact` — text
  equality, **not** runtime bit-exactness. Any divergent request yields
  `text_divergent_needs_review` and does **not** pass.
- **Review evidence.** A divergence is qualified only by a per-output review
  entry (`--review-evidence`) keyed to that exact
  `(case, concurrency, repeat, request)` with `kind` in
  `matched_logits | task_quality | manual_review`, `passed: true`, and non-empty
  `evidence`. If every divergent request has such an entry the gate is
  `text_divergent_review_qualified`; otherwise it stays
  `text_divergent_needs_review`.
- **Kernel evidence is context only.** `--kernel-evidence` records the SHA-256
  of the component-level numerics run (`rel_l2 < 0.01`, `cosine > 0.9999` from
  `python/tools/benchmark_v41_ep_groups.py`). It is reported as component-level
  context and can never qualify an end-to-end text divergence by itself.

Run the component oracle separately (context, not a gate substitute):

```bash
scripts/run-tp-ep-kernel-checks.sh test
scripts/run-tp-ep-kernel-checks.sh bench --phase split --tp-degree 4 --intermediate 576 \
  --output runs/tp-ep-e2e/kernel-tp4.json
scripts/run-tp-ep-kernel-checks.sh bench --phase split --tp-degree 2 --intermediate 1152 \
  --output runs/tp-ep-e2e/kernel-tp2ep2.json
# checkpoint slices: add --snapshot <official snapshot path>
```

## 9. Power and environment capture

```bash
nvidia-smi --query-gpu=timestamp,power.draw,temperature.gpu,clocks.sm \
  --format=csv,noheader,nounits -l 1 > runs/tp-ep-e2e/power-tp4ep1.csv
# repeat for tp2ep2
```

Reuse the host's existing thermal history if one is running; do not add a new
monitor for this protocol.

## 10. Evidence to keep

- `runs/tp-ep-e2e/startup-<arm>.json` and `runs/tp-ep-e2e/<arm>.json`
  (per-request `text`, `reasoning`, raw SSE `events`, raw `usage`, `ttft_ms`,
  `inter_chunk`, `content_chunks`, rate-approximation basis, content check,
  `error`/`partial`, probe identity). Fill the tracked template
  `scripts/fixtures/tp-ep-e2e-startup-metadata.template.json` per arm.
- `runs/tp-ep-e2e/review.json` (per-output review entries) and
  `runs/tp-ep-e2e/compare.json`.
- `runs/tp-ep-e2e/kernel-tp*.json` and the power CSVs.
- Both `run.sh --dry-run` transcripts and the corpus SHA-256. Frozen v3 for the
  full arms: `823f203ca1ce054b352a7e97f8acf6102116d5fa89b7226f167b26914778b750`.
  The earlier C1/C2 pilot used
  `43c8f361b392bd916a171432031d9ae3f977c465ba32ff7c94e7f118154d5709` (authoritative
  from `runs/tp-ep-preflight/g1-live/pilot-c1-rerun-tp2ep2.json`) and must be
  labelled `pilot`, never compared as the frozen corpus. The runner records the
  SHA it used.

## 11. Non-claims

- The four-Spark results above carry no six-rank claim. The full-six extension
  (section 12) is designed and CPU-tested but **not yet executed**; `rhea`/`moa`
  are connected and may be used freely for benchmarking and development per the
  user's latest permission, and remain unqualified with no images or checkpoint
  provisioned yet.
- A pass is an internal matched A/B qualification, not release throughput, broad
  quality, memory or readiness evidence.
- The component numerics oracle is tolerance-based context; it does not make
  cross-topology output bit-exact, and this protocol never asserts that it does.

## 12. Full-six (six-rank) extension

The six-rank objective is now **required**, not optional support. It is an
isolated extension so the frozen four-Spark runner and corpus are untouched:

- `scripts/fixtures/tp-ep-e2e-corpus-v4-six.jsonl` — sha256
  `99034171dba1fe8e6c64e41f9cbf07ab960d6c77932079f44a70c789cf836d85`; carries
  explicit `configs`/`comparisons`, common controls (no faked `SPARK_COUNT`),
  exactly one header/controls/matrix, and the four decode cases copied
  **verbatim** from the frozen v3 corpus (validated byte-for-byte against
  `823f203c…b750`). Validation reports the SHA of the corpus path it was given,
  not a hardcoded one.
- `scripts/qualify-ds41-tp-ep-e2e-six.py` — loads the frozen runner via runpy
  under a SHA-256 assertion (`e197146a…08c4`) and reuses its content checks, run
  loop and `compare_arms`; records the frozen runner SHA, the v3 base SHA and
  the v4 SHA in every output.
- `scripts/tests/test_tp_ep_e2e_six.py` — 18 CPU-only tests.

Only the three approved `(rtx_count, spark_count, spark_tp, spark_ep)` tuples
are accepted (`2,6,2,3`; `2,6,3,2`; `1,6,3,2`), so a permissive
`TP1×EP6`-style config is rejected even though it satisfies `TP*EP = count`.
Config field types must be real positive integers, not booleans or zeros.

Every arm must pass `--rtx-expert-layers` **explicitly** — `20` for the 2RTX
six-rank arms and `0` for the 1RTX arm — because an omitted flag defaults to
`auto` and reabsorbs the KV/experts. The common corpus controls are enforced
against the actual metadata, not just self-consistency: `concurrency 16`,
`prefill_batch_tokens 2048`, `worker_capacity 4096` (and at least the C16
admission need), `prefix_cache_entries 20`, `max_context_tokens 1048576`,
`max_output_tokens 393216`, and `model` = the frozen checkpoint. A run whose
metadata and argv both say `--concurrency 1` fails before any request, and
`compare` validates the corpus and verifies each arm's metadata against its
named config before qualifying. `dspark_draft_limit` is **per config** (7 for
2RTX, 5 for the 1RTX single) and must match the arm metadata; it is not a common
control.

Configs and allowed comparisons:

| Config | Hardware | Topology | World | RTX local | Class |
| --- | --- | --- | --- | --- | --- |
| `2rtx6-tp2ep3` | 2 RTX + 6 Spark | TP2×EP3 | 6 | 20 (remote 20) | paired |
| `2rtx6-tp3ep2` | 2 RTX + 6 Spark | TP3×EP2 | 6 | 20 (remote 20) | paired |
| `1rtx6-tp3ep2` | 1 RTX + 6 Spark | TP3×EP2 | 6 | 0 (all 40 remote) | standalone |

- Paired `2rtx6-tp2ep3` vs `2rtx6-tp3ep2`: speed and quality, requiring equal
  `rtx_count`, `spark_count`, `resolved_rtx_expert_layers`, `spark_first_layer`
  and `kv_class` (`g3-paired`). Both use the G3 KV class if qualified; the
  actual pool is **not** pinned to the 13.09 GB value.
- `1rtx6-tp3ep2`: standalone quality plus **actual memory/capacity evidence**.
  The coordinator argv must pass `--rtx-expert-layers 0` explicitly and must
  omit `--placement-directory`; KV is the initial `single-auto` pool.
- **Spark vs coordinator memory are separate domains and must not be conflated.**
  The Spark host reserve is **20 GiB (21474836480) for every config**: each of
  the six hosts supplies `mem_total_bytes`, `device_budget_bytes <=
  mem_total_bytes - 20 GiB`, `mem_available_bytes` between the reserve and
  `mem_total_bytes`, explicit `accounted_resident_bytes` /
  `accounted_workspace_bytes` / `accounted_ring_bytes` with
  `resident + workspace + ring <= device_budget_bytes`, and a `source_log`. The
  resident term must reach the exact per-TP-rank minimum (TP2 `3609722880`, TP3
  `2406481920` bytes per layer, times the remote-layer count: 72,194,457,600 /
  48,129,638,400 for the 2RTX arms and 96,259,276,800 for the 1RTX arm), so a
  zero or under-sized resident fails. The 2 GiB planner `runtime_headroom_bytes`
  is a **coordinator/RTX** domain recorded under `memory_evidence.coordinator`
  with `device_free_bytes` (RTX device free, never host free) and is never used
  as the Spark reserve; a 19 GiB Spark reserve is rejected. Evidence is
  machine-schema checked only: the gate reports
  `standalone_memory_schema_ok_review_required` and does not qualify until
  `logs_independently_reviewed` is set after an independent per-host log review;
  quality passing without memory evidence is
  `standalone_quality_pass_memory_review_required`. A single
  `observed_post_alloc` number is not accepted, because Spark UMA caches make
  total-minus-cudaFree meaningless as a resident measure. A valid single-RTX
  throughput report is allowed but is never a cross-hardware speedup claim.
- The current four-Spark run failing on an RTX graph OOM shows that the startup
  2 GiB / 800 MiB headroom is **not** a general qualification proof; the
  standalone arm therefore requires the per-host evidence above.
- `1rtx6` vs any `2rtx6` arm is a declared forbidden comparison and is rejected
  with its reason, not a generic identity mismatch.
- A missing control on a non-standalone candidate fails structurally as
  `comparison_not_allowed`; it never crashes.

**Canonical scope gate.** A qualifying arm must have run the exact
`C1/2/4/8/16 × 3` matrix over all four frozen cases, and its recorded
`corpus_sha256`/`frozen_runner_sha256` must match. The gate compares the arm
against the canonical v4 matrix/cases *before* the core coverage check, so a
pilot run with `--case`/`--concurrency` overrides (allowed for exploration)
cannot qualify; the frozen four-Spark runner has the same latent gap and its
in-flight runs will be verified manually (4 cases × 93 = 372 requests).

Six physical ranks must be present exactly once (world 6, ranks 0..5, no
duplicates or extra argvs). All TP families are built into the same native
library, so both arms record every `expert_tp_manifests` entry with identical
hashes and the per-arm difference lives in `active_topology`; the frozen family
gate therefore applies unchanged. No NVFP4/EXL3.

```bash
python scripts/qualify-ds41-tp-ep-e2e-six.py validate
python scripts/qualify-ds41-tp-ep-e2e-six.py selftest
python scripts/qualify-ds41-tp-ep-e2e-six.py run --arm 2rtx6-tp2ep3 \
  --startup-metadata runs/tp-ep-e2e/startup-2rtx6-tp2ep3.json \
  --output runs/tp-ep-e2e/2rtx6-tp2ep3.json
python scripts/qualify-ds41-tp-ep-e2e-six.py compare \
  --control runs/tp-ep-e2e/2rtx6-tp2ep3.json --candidate runs/tp-ep-e2e/2rtx6-tp3ep2.json \
  --output runs/tp-ep-e2e/compare-six-paired.json
# standalone: compare --candidate runs/tp-ep-e2e/1rtx6-tp3ep2.json (no --control)
```
