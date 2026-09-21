# TP6xEP1 candidate: exact live flags, cost isolation, provisioning

CPU-only preparation for the six-rank candidate. No hardware was used to produce
this file; every value is either read from the source, from the measured v8
baseline evidence, or from a live read-only inventory.

## 1. Exact candidate flags (matched to the Official TP4 baseline)

Files: `scripts/bench/candidate-tp6-1x-official-match.config` and
`candidate-tp6-2x-official-match.config`. They are loaded by the real
`scripts/release-common.sh` loader and unit-tested in
`scripts/tests/test_v9_candidate_configs.py` (CPU-only).

| Key | 1x candidate | 2x candidate | Baseline it matches |
| --- | --- | --- | --- |
| `MODEL_ID` / `MODEL_REVISION` | `deepseek-ai/DeepSeek-V4.1-Flash` / `dba1be0a…` | same | official quant, both baselines |
| `EXPERT_FORMAT` / `SPARKINFER_EXL3` | `native` / `disable` | same | both baselines |
| `RTX_GPUS` | `1` | `2` | both baselines |
| `RTX_EXPERT_LAYERS` | `5` | `20` | 1x measured 5 local / 35 remote; 2x measured 20/20 |
| `SPARK_COUNT` / `SPARK_TP` / `SPARK_EP` | `6` / `6` / `1` | same | new variable vs the 4-rank TP4xEP1 baseline |
| rank map | `0..5 -> group 0, tp_rank 0..5` | same | pure TP6, one replicated group |
| `DSPARK` | `on` | same | both baselines |
| `CONCURRENCY` / `PREFILL_BATCH_TOKENS` / `PREFIX_CACHE_ENTRIES` | `16` / `2048` / `20` | same | both baselines |
| `MAX_CONTEXT_TOKENS` / `MAX_OUTPUT_TOKENS` | `1048576` / `393216` | same | both baselines |
| `HOST_CACHE_BYTES` | `auto` | same | both baselines |
| `KV_POOL_SIZE` | unset (auto) → baseline 16,700,672,000 B | unset (auto) → baseline 13.091 GB / 14,680,064 device tokens | both baselines |
| `SPARK_DEVICE_BUDGET_BYTES` | `109119320064` | same | six-rank tested floor |
| `SPARK_5_LANE_A` | `10.55.0.252` | same | **live** moa fabric A (see §4) |
| `LANE_B` | all empty | all empty | native release path reads LANE_A only |

**KV pool warning.** The existing six-rank fixture
`scripts/fixtures/tp-ep-six/site-2rtx6-tp6ep1.config` pins
`KV_POOL_SIZE=5,037,542,400`. That is the tested six-rank candidate value, **not**
the official 2x pool. Using it for the matched comparison would confound the pool
size with the topology. Leave auto on both, or pin identical bytes on both arms.

**Placement handoff.** With an explicit `RTX_EXPERT_LAYERS` in 1..39 the release
launcher now publishes/waits for a placement plan (`placement_directory` is set
for 2 RTX always, and for 1 RTX when an explicit local count is given), so the
1x candidate gets a real 5-local boundary instead of the legacy no-handoff `0`.
Workers still receive the published first layer; their *allocated* layer count can
exceed the dispatched count (the v8 1x baseline dispatched 35 and each worker
allocated 40), so record both numbers.

## 2. Cost policy: default auto and the legacy isolation

`DS41RT_ADAPTIVE_COST_MODE` (source: `speculative/cost.rs`) accepts
`auto | legacy | builtin | profile` and is forwarded to both roles by `run.sh`.
The resolved model is logged at startup as `cost_model=`:

| Mode | Resolved `cost_model` | Notes |
| --- | --- | --- |
| `auto` (default) | `builtin-calibration` when the shipped table applies (legacy TP4 layout, `gpus=1`), otherwise `legacy-heuristic` | the shipped table only exists for the legacy TP4 layout |
| `legacy` | `legacy-heuristic` | forces the heuristic regardless |
| `builtin` | `builtin-calibration` | forces the builtin label |
| `profile` | `explicit-profile` (or `explicit-profile-missing` = config error, cannot serve) | requires `DS41RT_ADAPTIVE_COST_PROFILE` |

Two comparisons, per the plan:

1. **Primary (deployment truth):** run the baseline and the TP6 candidate with
   their normal defaults (`auto`) and record the resolved `cost_model=` line from
   each. This is what the deployment actually uses, but the two arms are expected
   to resolve differently (TP4 1x builtin vs TP6 heuristic), so it does not by
   itself isolate topology from cost policy.
2. **Attribution (controlled):** export
   `DS41RT_ADAPTIVE_COST_MODE=legacy` for **both** the baseline and the candidate.
   Both then resolve `legacy-heuristic`, and any remaining difference is not a
   cost-policy difference.

Do not infer the resolved mode from the `profile=None`/`nvfp4=false` input path;
use the `cost_model=` field. Record the mode and the captured log line in the
report manifest's `cost_policy` block (schema supports
`default_mode`/`legacy_mode`/`adaptive_mode_env`/`evidence`).

## 3. What the final report must contain

- **Actual decode and prefill measurements**, not a capacity proxy and not a
  component-only number. The kernel tiling table stays a component section.
- The FFN ~1.5x hypothesis is **unmeasured**; it must not appear as a result.
- Per-arm provenance: commit, both image digests, checkpoint revision, captured
  config SHA, resolved geometry (local/remote/allocated layers, pool bytes,
  pages, headroom), draft width, resolved `cost_model`, warmup/repeats, nonce
  seed, corpus/tokenizer SHA.
- Retained failures and incomplete matrices; per-campaign spread, never a
  universal noise band.

## 4. Network: moa — owner-verified, private site override

**Status: VERIFIED by the hardware lease owner at 2026-09-20T18:10:03Z (host
clock; 18:10:10Z owner clock).** Evidence:
`runs/tp6-validation/moa-identity-verification.md`.

| Host | Fabric A | Fabric B | Status |
| --- | --- | --- | --- |
| ostrich | `10.55.0.1` | `10.55.0.7` | matches record |
| dodo | `10.55.0.2` | `10.55.0.8` | matches record |
| emu | `10.55.0.3` | `10.55.0.9` | matches record |
| kiwi | `10.55.0.4` | `10.55.0.10` | matches record |
| rhea | `10.55.0.5` | `10.55.0.11` | matches record |
| **moa** | **`10.55.0.252`** | **`10.55.0.253`** | **verified override** |

Owner-recorded identity: hostname `moa`, FQDN `moa.birds.vimoin.com`, machine-id
`1f1364d8e14c488380f2e2ccb709ed58`, GB10 `GPU-52635ab1-1d3c-1360-58ea-a25501d3f0aa`
SM12.1, mgmt `enP7s7 172.22.2.6/16`. The recorded `10.55.0.6`/`.12` are stale and
unreachable, and the `moa` ssh alias still points at `.6`.

**Private override authorized by the owner.** The candidate configs use the
verified values:

- `SPARK_5_HOST=172.22.2.6` (management control path, because the alias is dead;
  `#SPARK_5_HOST=moa` is kept commented for when the alias is fixed)
- `SPARK_5_LANE_A=10.55.0.252` (LANE_B `.253` available, unused)

This is a **private benchmark override, not a shipped default**: it does not change
the user's `ds41rt.config`, `~/.ssh/config` or any NIC, and must not be turned into
a user-facing instruction. The user's default keeps the recorded value until the
owner decides otherwise.

**Open item — the wrong coordinator address was checked.** The owner's route line
targeted `10.55.0.1`, which is **ostrich** (`SPARK_0_LANE_A`), not the
coordinator. raptor's own fabric address is `10.55.0.22` (its local
`ip -br -4 addr` shows `enp1s0np0 10.55.0.22/24`); the route from moa to
`10.55.0.22` still needs to be recorded. Also note the A/B asymmetry: if moa's
lane is its A address, replies egress B unless a host policy route forces A.

Earlier unverified observation of the same facts is retained for provenance at
`~/.cache/ds41rt-v9-baseline/provenance/moa-address-observation-UNVERIFIED.md`
(including a local/UTC label error this session made: `02:05Z` was actually
~`18:05Z`).

## 5. Checkpoint provisioning (read-only verification)

The official snapshot `models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a…`
is present on raptor and all six Sparks: 59 entries with the final shard
(`model-00048-of-00048.safetensors`) present — verified read-only on ostrich,
rhea and moa, and already present on dodo/emu/kiwi.

The release launcher mounts `$HOME/.cache/huggingface` (HOME is the user's, not
`/root`), so the earlier `/root/.cache/huggingface` permission-denied reading is
**not** a blocker and no root access is required for the checkpoint. What rhea and
moa still lack is the **ds41rt container image**, not the model weights.

## 6. Build/provisioning sequence to think ahead

1. Prepare the candidate source, then **freeze it either way**: an immutable copy
   of the current dirty tree with a generated source manifest is acceptable for
   measurement before the commit lands, and a final clean commit/worktree is
   produced later for the release. What must not happen is measuring a tree that
   changes underneath the run; the manifest plus the captured digests are what
   make a dirty-tree freeze honest. A worktree at `HEAD` alone would not contain
   the uncommitted TP6 work.
2. `./build.sh` must export roles `tp2;tp3;tp6` plus the default TP4 into the
   Spark image (`io.ds41rt.v41.spark_tp_roles`), and the coordinator image must
   carry the same engine revision.
3. The **hardware lease owner provisions all six ranks once the image is built**
   (rhea and moa currently have no ds41rt image, and the four original Sparks have
   only the v8 image), including the candidate library/daemon on every rank.
4. Re-verify fabric lanes (moa override) and RDMA device/GID per rank; record the
   route from moa to the coordinator at `10.55.0.22` (not `.1`).
5. Then, and only then, the E2E/perf campaign in
   `scripts/bench/official-v9-campaign.md`.
6. The global Docker wipe and clean `./build.sh` remain a later release gate,
   after all raw evidence is durable.

A Python-only qualifier fix does not change `libds41rt_native.so`, so native
artifact digests stay comparable across such a fix; state this explicitly when
mixing a pre-fix and post-fix source revision in one campaign.

**Separate qualification record.** The six-rank qualifier (v5 six-arm corpus and
its fixed `KV_POOL_SIZE=5,037,542,400`) remains its own qualification artifact and
keeps its own maintained report; the official-match configs here are a different,
full-system campaign and do not replace or renumber that record.
