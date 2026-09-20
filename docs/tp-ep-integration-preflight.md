# TP×EP replicated Spark expert groups — integration preflight

Status: **preflight only, no measurement and no mutation.** Nothing in this
document is a performance result. Every figure is either a read-only
observation, a source-verified fact, or a recorded test exit status.

Owner: integration/preflight agent (`91570601-c759-48c8-b28e-180746e8482c`).
Artifacts: this file and the disposable `runs/tp-ep-preflight/` directory only.
No implementation file, GPU setting, service, container, checkpoint or artifact
was modified. Nothing was committed or pushed.

Companion documents: `docs/tp-ep-implementation-plan.md` (planning agent, the
consumer of this preflight), `docs/tp-ep-architecture-audit.md` and
`docs/tp-ep-kernel-audit.md` (read-only audits; the kernel audit existed at
finalization, the architecture audit had not yet landed).

Capture time: `2026-09-20T02:56:05Z` on `raptor`. Evidence snapshots:
`runs/tp-ep-preflight/logs/baseline-snapshot.txt`,
`runs/tp-ep-preflight/logs/spark-inventory.txt`.

---

## 1. Scope and method

Read-only inspection of source, git state, containers, GPU state and logs;
focused CPU-only test execution in an isolated target directory; no CUDA build,
no GPU kernel, no benchmark, no service restart, no GPU-setting change, no
commit. Read-only `ssh`/`docker exec` inspection only. Tests were run in
`runs/tp-ep-preflight/` with an isolated `CARGO_TARGET_DIR` and did not read or
write any served artifact.

Hardware assumptions to validate later (not yet validate-able): config A is the
only physically testable layout (`raptor` 2× RTX + `ostrich,dodo,emu,kiwi`).
Configs B/C (six Sparks) have no connected hardware.

---

## 2. Current baseline source identity

| Item | Value |
| --- | --- |
| Repo | `/home/tj/Developer/ds41rt` (primary worktree) |
| Branch / HEAD | `dev` @ `bec5fcc45381b9383dfe6ec5d395626b91343f32` |
| Upstream | `origin/dev` == HEAD (no ahead/behind) |
| Tracked tree | clean at first snapshot; later only other agents' in-flight files (see §2.2) |
| Worktrees | `dev` primary; `/home/tj/Developer/ds41rt-v6-integration` @ `af863be` `[work/v6-hostcache]`; a prunable detached checkout `/home/tj/.cache/ds41rt-upstream-20260916/release-candidate-source` @ `8380890` |

### 2.1 Submodule pins (recorded in index and matching checked-out HEAD)

| Submodule | Pin | State |
| --- | --- | --- |
| `third_party/sparkinfer` | `4b0954148523b5a2e93813f963d483ffd350b9c9` | detached, clean |
| `third_party/xgrammar` | `557becfb64c503ae9c04344b0047661f43f44320` | detached, clean |
| `third_party/gptqmodel` | `5340775df1370b8812f7cd91a907afe97d420784` | detached, clean |
| `third_party/transformers` | `62d7ebd7de4938e072b7aaeb881593b79dc56835` | detached, clean |
| `third_party/xgrammar/3rdparty/cpptrace` | `6689d14c203eed390ae7bb64f56a983cfd7dff9c` | clean |
| `third_party/xgrammar/3rdparty/dlpack` | `bbd2f4d32427e548797929af08cfe2a9cbb3cf12` | clean |
| `third_party/xgrammar/3rdparty/googletest` | `df1544bcee0c7ce35cd5ea0b3eb8cc81855a4140` | clean |

The release-image provenance records agree with these pins:
`.ds41rt-release-image/SPARKINFER_PROVENANCE.json` revision
`4b095414…` and `XGRAMMAR_PROVENANCE.json` revision `557becf…`. The published
v8 images in use were therefore built from, and validate, this source identity.

### 2.2 Clean/dirty assessment excluding agents' new documents

At first capture the whole tracked tree was clean. During this session other
agents began work, so at `02:56Z` `git status` was:

```
## dev...origin/dev
 M rust/crates/ds41rt-core/src/lib.rs
?? docs/tp-ep-implementation-plan.md
?? rust/crates/ds41rt-core/src/replicated_expert_schedule.rs
```

These belong to the planning agent, the CPU scheduler agent and a kernel/loader
agent, not to this preflight; they are excluded from the baseline identity. By
the time this report was finalized the agent-authored set was:
`docs/tp-ep-implementation-plan.md`, `docs/tp-ep-kernel-audit.md`,
`docs/tp-ep-scheduler.md`, `rust/crates/ds41rt-core/src/replicated_expert_schedule.rs`,
and unstaged edits to `rust/crates/ds41rt-core/src/lib.rs` and
`rust/crates/ds41rt-loader/src/v41_expert_staging.rs`
(`docs/tp-ep-architecture-audit.md` was still absent). The
preflight-observable baseline is: **tracked tree clean at `bec5fcc`, all
submodules clean at the pins above, plus untracked/unstaged TP×EP agent files.**

---

## 3. Host, device and service inventory (read-only)

### 3.1 raptor (x86_64, coordinator)

| Item | Observation |
| --- | --- |
| CPU / RAM | 64 cores; 183 GiB total, ~166 GiB available; load 1.75 |
| GPU0 | RTX PRO 6000 Blackwell 96 GB, UUID `GPU-fe5b6dd0-a77c-c8fb-6360-e1b9d9918ac0`, PCI `00000000:11:00.0` |
| GPU1 | RTX PRO 6000 Blackwell 96 GB, UUID `GPU-95f8f212-9131-df99-fd53-7535965197d7`, PCI `00000000:E1:00.0` |
| Power / clocks | both 400 W limit; memory clock 405 MHz idle, max 14001 MHz; persistence Enabled; no overclock |
| GPU0 usage | 94,700 MiB used by host PID 3477239 (`ds41rt`); 0 % util at capture |
| GPU1 usage | 12 MiB used; effectively free |
| Port 8000 | LISTEN on `0.0.0.0:8000` |
| Port 19441 | not listening locally (expert port is on the Sparks) |

### 3.2 Sparks

| Host | Role | GPU | Release expert container (image `sha256:881f51cc1e86…`) | GPU holder |
| --- | --- | --- | --- | --- |
| `ostrich` | rank 0 / world 4 | GB10 SM121 | `ds41rt-spark-expert-ostrich-19441`, up ~15 min | PID 2994059 `ds41rt`, 76,889 MiB |
| `dodo` | rank 1 / world 4 | GB10 SM121 | `ds41rt-spark-expert-dodo-19441`, `--rank 1 --world 4` | release expert `ds41rt` resident |
| `emu` | rank 2 / world 4 | GB10 SM121 | `ds41rt-spark-expert-emu-19441`, `--rank 2 --world 4` | release expert `ds41rt` resident |
| `kiwi` | rank 3 / world 4 | GB10 SM121 | `ds41rt-spark-expert-kiwi-19441`, `--rank 3 --world 4` | release expert `ds41rt` resident |

Each Spark listens on `0.0.0.0:19441`. Spark load was ≤ 0.04. `nvidia-smi`
memory fields on GB10 report `N/A`; the compute-app query is the authoritative
holder check and it shows a live `ds41rt` expert per Spark.

**Consequence:** the Spark GPUs are *not* idle. An "exclusive ostrich GPU
lease" is impossible while this service is resident, and a WIP container run
would contend with ~77 GiB of resident expert memory on the same physical GPU.

---

## 4. Currently served checkpoint format, config and layer placement

This is the live v8 release deployment, verified from `docker inspect`,
`nvidia-smi`, and the coordinator log — not assumed.

### 4.1 Identity

| Item | Value |
| --- | --- |
| Checkpoint | official `deepseek-ai/DeepSeek-V4.1-Flash`, snapshot `dba1be0a40aa45a94ad051997016db3960a90277` |
| Format | native routed-expert format (`EXPERT_FORMAT=native`, log `nvfp4=false`); no EXL3 |
| Coordinator image | `ghcr.io/tpurtell/ds41rt-coordinator:v8` digest `sha256:08c2d6df9a0a6a365eff2c014172478b40d9f39d06437a1c9244c566181b9e40` |
| Spark image | `sha256:881f51cc1e86e114e2bd054e6b9f01f45ab7a7fb6f7e5aaed58e2967e71e51ab` |
| Config fingerprint | `DS41RT_RELEASE_CONFIG_SHA256=6a1e29d3412418b4b09f5b6a7182508b862f73cecb0f22c41aaae6abacde817c` |
| API | `/health` and `/v1/models` OK; model id `deepseek-ai/DeepSeek-V4.1-Flash` |

### 4.2 Exact live topology

- Coordinator: `--rtx-gpus 1` (single RTX, GPU0), `--dspark`, `--concurrency 16`,
  `--prefill-batch-tokens 2048`, `--prefix-cache-entries 20`, `--host-cache-bytes auto`.
- Sparks: 4 peers `10.55.0.1..4:19441`, each `expertd-native --rank i --world 4
  --capacity 4096 --device-budget-bytes 107374182400 --first-layer 0`.
- Coordinator log: `native serving residency ready rtx_layers=5
  first_remote_dispatch_layer=5 remote_dispatch_layers=35 spark_world=4`.
- Placement vector: 5 × `rtx_local_shared1` (layers 0–4, local TP1 on one card),
  35 × `spark_tp4_shared1` (layers 5–39).
- Memory: `global_bytes=16,700,672,000`, `device_occupied_bytes=99,175,628,800`,
  `device_budget_bytes=101,973,491,712`, `runtime_headroom_bytes=2,147,483,648`,
  `bottom-up RTX expert placement layers=5 resident_bytes=36,097,228,800`,
  snapshot `slots=42 bytes=122,766,336`, host cache `3,489,660,928 B`,
  dSpark `lanes=2 lane_requests=8 draft_width=5`.

**Note for matched comparison.** The *served* baseline is RTX1/TP1 with 5 local
layers (automatic placement). The v6 performance record describes the dual-RTX
TP4×EP1 architecture with 20 local layers TP2 (`docs/release-v6-performance.md`
§Deployment and cache capacity). A TP4×EP1 reference arm for the campaign must
re-measure whichever layout it claims; legacy v6/v7/v8 tables are background
evidence only (matching the plan's rule).

---

## 5. Existing WIP slots, artifacts and shared ownership

### 5.1 WIP slots present (read-only listing; untouched)

| Container | Slots |
| --- | --- |
| `ds41rt-coordinator-wip` (idle `sleep infinity`, Up 40 h, image `ds41rt-coordinator-dev` `f318ec18…`) | `cpuopt`, `expertenv`, `expertevent`, `expertevent2`, `expertsleep`, `v7-k2homog`, `v7q-a1` |
| `ds41rt-coordinator-wip-dual` (idle, Up 31 h, same image) | `v7-k2homog`, `v7q-a1`, `wip-slots-copy` |
| Spark `ds41rt-spark-expert-wip` on all four hosts (idle `sleep infinity`, Up 40 h, image `ds41rt-spark-expert-dev` `dab2b979…`) | one per host |

All slots are prefixed `v7*`, `cpuopt`, or `expertenv`/`expertevent*`/`expertsleep`
and carry the standard native `ds41rt.config` with `TP2_*=off`. **No existing
slot is TP×EP-related.** Fingerprints observed: `cpuopt`/`expertenv`/
`expertevent`/`expertevent2`/`expertsleep` → `28db4740884ee1d2…`;
`v7-k2homog`/`v7q-a1` → `24f2522929eb27dd…`.

### 5.2 Safe isolated baseline and candidate build slots

- **Baseline (no new build needed):** the running release service is the
  current, provenance-verified baseline and must be left alone. Its exact
  restore recipe is recorded in §8.
- **Candidate build slot:** a single, new, unique slot name that does not
  collide with the table above — e.g. `tp-ep-preflight` (dry-run only) and later
  `tp-ep-a` for the A/B campaign. Never reuse or overwrite `v7*`, `cpuopt`,
  `expertenv`, `expertevent*`, `expertsleep`, or `wip-slots-copy`.
- **Candidate arm slots:** exactly one WIP slot per arm at a time; both arms must
  freeze to the same engine/submodule/checkpoint identity.

### 5.3 Shared resources that slots do **not** isolate

Per `AGENT_DEV_HINTS.md`: slots isolate artifacts, not GPUs, ports, build
staging or containers. Specifically shared:

- **GPU0** (`fe5b6dd0…`) and both RTX cards; **all four Spark GPUs** with the
  resident release experts.
- **Port 8000** and **port 19441** on every host.
- **WIP containers** (all five, plus the dev images) and `/wip/build`,
  `/wip/incoming`, `/wip/output`, `/wip/cache`.
- **Build staging**: `/wip/source`, `.ds41rt-wip/source-staging`, `dist/`,
  `native/build-*` (e.g. `native/build-nvfp4-share`, 58 MB).
- **Docker build caches and images**. `wip.sh --recreate` destroys all five WIP
  containers and their build caches — an action that interferes with other
  agents and must be coordinated.

### 5.4 Dev-container toolchain and its staleness (build dependency)

- `ds41rt-coordinator-dev` (`f318ec18…`): Rust/cargo, CUDA 13.2, PyTorch;
  `DS41RT_CUDA_ARCH=120`, `DS41RT_TARGET_PLATFORM=linux/amd64`.
- `ds41rt-spark-expert-dev` (`dab2b979…`): arm64 CUDA 13.2, cargo at
  `/opt/cargo/bin`, Triton tool paths; `DS41RT_CUDA_ARCH=121`.
- Source is **copied**, not bind-mounted: the only bind mounts are
  `/home/tj/.cache/huggingface` → `…:ro` (twice). Work happens in `/wip/source`
  and `/wip/slots/<name>/workspace`.
- **Staleness:** both dev images and all five containers record
  `DS41RT_SPARKINFER_COMMIT=63e2140e4a32a977faa777c172b86679344fdc6a`, while
  the current pin is `4b0954148523b5a2e93813f963d483ffd350b9c9`. `wip.sh`'s
  `preflight_existing_container_images` rejects containers whose image differs
  from the current dev image, and the standard remedy is `--recreate`
  (destructive). A clean WIP build therefore requires either a coordinated
  `--recreate` window or a rebuilt dev image; this is a Phase-3 prerequisite and
  a serialization hazard, not something any one agent may do unilaterally.

---

## 6. Readiness and build dependencies

### 6.1 Source-verified blockers for TP×EP (no GPU needed to see them)

These are the concrete places a TP2×EP2 / TP3×EP2 / TP2×EP3 topology meets
hard-coded TP4×EP1 assumptions. They corroborate the plan's §4 and §14 and are
listed here as verified preflight facts (audits own the full enumeration):

| # | Blocker | Evidence |
| --- | --- | --- |
| B1 | Coordinator rejects any peer count other than 2 or 4 | `rust/crates/ds41rt-daemon/src/v41_native_serve.rs:41` `ensure!(matches!(args.peers.len(), 2 \| 4), …)` |
| B2 | A 2-peer Spark world requires an EXL3 checkpoint; native + TP2 is disallowed | `v41_native_serve.rs:200-201`, `v41_experts/service.rs:133-137` |
| B3 | Spark executor ID namespaces are TP2 (`5..=6`) and TP4 (`1..=4`) only | `rust/crates/ds41rt-transport/src/v41_expert.rs:29-33` |
| B4 | Rank planes are fixed-size `[u64; 4]` / `planes[4]`; TP2 has a separate `planes[2]`; no 3-plane or N-plane path | `v41_expert.rs:254,275,352`; `v41_expert/{tcp,roce,chunks}.rs`; `native/include/ds41rt_v41_experts.h:73-110` |
| B5 | Native coordinator wave requires world 2 or 4 | `rust/crates/ds41rt-daemon/src/v41_experts/coordinator.rs:120` |
| B6 | Launcher accepts `SPARK_COUNT` only `0, 2, 4` | `scripts/release-common.sh:255-262` |
| B7 | Only `SPARK_0..3` host/lane keys exist | `ds41rt.config:73-86`; allowlist `scripts/release-common.sh:104` |
| B8 | `EXPERT_HOSTS` is `[&str; 4]` and `DS4_EXPERT_TP_WORLD_SIZE = 4` | `rust/crates/ds41rt-core/src/constants.rs:6-7` |
| B9 | Placement labels only `rtx_local_shared1`, `spark_tp2`, `spark_tp4_shared1`; cost model selects TP2/TP4 by world | `v41_native_serve/speculative/cost.rs:32,42` |
| B10 | Native target expert TP is a compile-time 4 | `commands/real_full/coordinator_kernels/target_attention.rs:56` |
| B11 | Intermediate sharding asserts exactly 4 shards | `commands/real_full/intermediate_sharding.rs` (plan refs 136-145, 301-302, 452-453) |
| B12 | Executor-set/planes types cannot represent `(group, tp_rank)` for 6 ranks | `v41_expert.rs` `ExpertProtocolV2ExecutorSet::Tp4`; `planes()` returns `[&[u8];4]` |

Positive: the protocol header already permits a Spark collective part count of
`2..=15` (`EXPERT_PROTOCOL_V2_MAX_SPARK_COLLECTIVE_PARTS = 15`), so a 6-rank
*part count* is not itself the blocker.

### 6.2 What is genuinely unknown (cannot be resolved read-only)

- Whether config A closes its Spark budget at the v6 matched boundary
  (`L_r = 20`, TP2: 20 × 3,609,722,880 B = 72.19 GB weights per rank) inside the
  100 GiB budget **including measured workspace/staging/headroom**. Needs a
  runtime startup report; the resident release expert already uses 76,889 MiB on
  the 100 GiB budget, which is suggestive but is TP4, not TP2.
- Measured Spark workspace per (TP, capacity) and per-layer load time for TP2/TP3.
- Exactly-once assembly and route-slot skipping behavior (kernel audit).
- Six-Spark behavior of any kind (no hardware).

---

## 7. Focused CPU-only test execution

All commands below were run from `/home/tj/Developer/ds41rt`. Rust tests used
`CARGO_TARGET_DIR=/home/tj/Developer/ds41rt/runs/tp-ep-preflight/target` and
`--offline`; Python tests used the project `.venv`. No test touched the GPU,
served artifacts, containers, build staging or the network beyond loopback.

### 7.1 Result summary

| Suite | Command | Result |
| --- | --- | --- |
| `ds41rt-core` | `cargo test -p ds41rt-core --offline` | **PASS** — 156 unit (155 passed, 1 ignored) + 25 integration, 0 failed |
| `ds41rt-transport` at HEAD | `cargo test -p ds41rt-transport --offline` | **FAIL (pre-existing, compile)** — test target does not compile |
| `ds41rt-transport` production | `cargo check -p ds41rt-transport --offline` | **PASS** in 4.36 s (production code compiles; only the test cfg is broken) |
| `ds41rt-transport` probe | patched disposable copy (see §7.3) | **PASS** — 165 unit (162 passed, 3 ignored) + 14 integration, 0 failed |
| launcher/topology/contract (Python) | `PYTHONPATH=python/reference .venv/bin/python -m pytest -q scripts/tests/test_native_release_launcher.py scripts/tests/test_compact_spark_topology.py scripts/tests/test_placement_handoff.py scripts/tests/test_release_ab_controls.py scripts/tests/test_release_gpu_selection.py scripts/tests/test_concurrency_cases.py python/tests/test_v41_expert_launch_contract.py python/tests/test_ds4_spark_aot_profiles.py` | **PASS** — 61 passed + 50 subtests, 7.89 s |
| scheduler/reduction/identity (Python) | `PYTHONPATH=python/reference .venv/bin/python -m pytest -q scripts/tests/test_placement_cost_holdout.py scripts/tests/test_deepseek_reduction_replay.py scripts/tests/test_top1_agreement.py scripts/tests/test_wip_expert_runtime_identity.py scripts/tests/test_v7_configurations.py scripts/tests/test_release_throughput_checks.py python/tests/test_native_experts.py` | **PASS** — 31 passed, 0.19 s |

Logs: `runs/tp-ep-preflight/logs/cargo-transport-core.log`,
`…/cargo-transport-probe.log`, `…/pytest-launcher-topology.log`,
`…/pytest-scheduler-reduction.log`.

### 7.2 Pre-existing failure (must be fixed before any transport test gate)

`cargo test -p ds41rt-transport` aborts at compile time:

```
error: expected one of `!`, `.`, `::`, `;`, `?`, `{`, `}`, or an operator, found `:`
  --> crates/ds41rt-transport/src/v41_expert/tcp_tests.rs:37:43
   |
37 | fn config() -> TcpTransportConfig { timing: false,
```

This is committed at HEAD (`git show HEAD:…` reproduces it) and was introduced
by commit `429a970` ("Resolve protocol-v2 transport timing once at startup
instead of per chunk"), which added a `timing` field to `TcpTransportConfig`
(`rust/crates/ds41rt-transport/src/lib.rs:96-102`) and left a stray
`{ timing: false,` fragment in the test helper. **Every transport test —
including `spark_executor_namespaces_reject_other_world_responses` and the TP2
RoCE/verbs topology tests — is currently unrunnable at HEAD.** A scan found no
other instance of this syntax pattern.

### 7.3 Disposable probe (not an implementation change)

To separate "one syntax error" from "the suite is broken", the whole `rust/`
source tree was copied (16 MB, `--exclude=target`) to
`runs/tp-ep-preflight/rust-probe/` and exactly one line was repaired there:

```diff
-fn config() -> TcpTransportConfig { timing: false,
+fn config() -> TcpTransportConfig {
     TcpTransportConfig { timing: false,
```

With only that repair, all 165 transport unit tests and 14 integration tests
pass. **The real repository was not edited.** Topology-relevant tests that pass
in the probe include `spark_executor_namespaces_reject_other_world_responses`,
`tp2_constructor_validates_real_peers_and_identities`,
`tp2_chunks_require_both_exact_rank_planes`,
`tp4_assembly_checks_identity_geometry_duplicates_and_arrival_order`,
`rdma_chunks_preserve_rank_rows_and_reject_stale_or_reordered_data`,
`paired_chunked_responses_complete_all_ranks_without_request_flag`,
`tp2_reset_retains_only_two_actual_peer_slots`, and
`protocol_v2_tcp_expert_request_bridge_matches_inproc_route_executor`.
Three tests are `#[ignore]`d because they require four idle RoCE expert workers
or a CUDA native library; those remain a GPU-lease dependency.

The probe directory is disposable and must be deleted before any build that
stages `rust/`; it is not tracked and not part of the deliverable.

### 7.4 Not run (deliberately)

- `ds41rt-daemon` Rust tests (908 tests) — the daemon crate pulls heavy
  dependencies (pyo3, turbojpeg/cmake, image, tokenizers, axum) and a full build
  was judged a "giant expensive build" for this preflight. The relevant daemon
  facts (`v41_native_serve.rs:41,200`; `service.rs:133-137`;
  `coordinator.rs:120`) were verified by reading the source instead. The
  `spark_world_is_format_and_rank_checked` unit test run is **SKIPPED**.
- Any CUDA/native build, any `native/tests/` run, any SparkInfer test, any GPU
  kernel, any benchmark, any service start/stop.

---

## 8. Proposed serialized full-model A/B plan (for the parent to authorize)

This uses only existing tools. It is a *plan*; no part has been executed. It
assumes config A (TP2×EP2, 2 RTX + 4 Spark) passes its CPU/loader gates first.

### 8.1 Preconditions

1. Transport test syntax error fixed in the real tree; `cargo test -p
   ds41rt-transport` green.
2. TP×EP implementation, launcher/config and loader/admission gates from the
   plan (Phases 2–5) pass; the kernel audit confirms route-slot skipping.
3. An authorized maintenance window with a single exclusive GPU/build lease
   (currently nobody has one; the release service is live).
4. A coordinated WIP dev-image refresh or `--recreate` window, because the
   current dev containers are pinned to SparkInfer `63e2140e…` while the tree is
   at `4b095414…`.

### 8.2 Matched conditions (topology is the only intended variable)

- Same engine commit and submodule pins in both arms; record `git rev-parse HEAD`
  and `git submodule status` per arm.
- **Same native checkpoint source**: official `DeepSeek-V4.1-Flash` snapshot
  `dba1be0a40aa45a94ad051997016db3960a90277`, `EXPERT_FORMAT=native`,
  `MODEL_REVISION` pinned. No EXL3, no NVFP4, no re-quantization.
- **Fixed RTX residency**: set `RTX_EXPERT_LAYERS` to an explicit `L_r` (never
  `auto`) equal in both arms, and require each arm's startup log to report that
  same `rtx_layers` / `first_remote_dispatch_layer`. If the budget forces a
  different `L_r`, disclose it and treat the comparison as un-matched.
- **Fixed KV budget**: pin explicit `KV_POOL_SIZE` and `MEMORY_RESERVATION`
  (identical bytes in both arms) so KV and headroom are not an independent
  variable. Note the current served default is `auto`/`auto`; the campaign must
  pin, not inherit.
- Same `CONCURRENCY=16`, `PREFILL_BATCH_TOKENS=2048`, `PREFIX_CACHE_ENTRIES=20`,
  `MAX_CONTEXT_TOKENS`, `MAX_OUTPUT_TOKENS`, `DSPARK=on`, power cap 400 W,
  memory clocks stock, same corpus/prompts/nonces.
- Both arms launched through `./scripts/run-wip.sh --wip-slot <slot> --config
  <frozen config>`; verify with `--dry-run` first.

### 8.3 Serialized execution

One arm at a time, one WIP slot at a time, no overlapping builds or performance
runs (`AGENT_DEV_HINTS.md`). Between arms: stop, confirm GPU memory returns to
baseline, rebuild/reload, then start the next arm. Order: (1) TP4×EP1 reference,
(2) TP2×EP2 candidate. Correctness before timing.

### 8.4 Workload matrix

| Dimension | Cells |
| --- | --- |
| Decode concurrency | C1, C2, C4, C8, C16 |
| Content | `code`, `topic`, `mixed` (primary); `counting` secondary |
| Prefill | representative cell `0 base + 32K new` (headline best), plus one retained cell (e.g. `64K + 16K`) |
| Speculation | `DSPARK=on` primary; target-only control where a claim depends on dSpark |
| Repeats | 1 warmup + **3 timed samples** per cell; keep every sample, report median and variance |

Existing tools:

- `scripts/bench-ds41-concurrent-api.py --concurrency 1 2 4 8 16 --case code|topic|counting`
  (use `--nonce` and `--max-tokens` for controlled comparison);
  `scripts/bench-real-full-mixed-concurrency.py` for mixed traffic.
- `scripts/bench-ds41-release-decode.py --case … --repeats 3 --nonce-seed …`
  for the nine-category decode corpus.
- `scripts/bench-ds41-release-prefill-matrix.py --base 0 --suffix 32768 …`
  (and the retained cell) for prefill.

### 8.5 Acceptance, timing and memory measurements

- **Acceptance** (separate from TPS): `scripts/collect-ds41-content-acceptance.py`
  against an otherwise idle server with `RUST_LOG=info,ds41rt::draft_policy=debug,
  ds41rt::timing=debug,ds41rt::lane_schedule=debug`, capturing the trace to
  `runs/tp-ep-preflight/`. Record unconstrained and grammar-constrained
  acceptance, verified/accepted drafts, zero-acceptance cycles, width counts.
- **Correctness gates before numbers**: `scripts/api-smoke.sh`,
  `scripts/api-constrained-smoke.sh`, retained-prefix/cache-counter checks, and
  output/acceptance equality (or a stated bound) against the reference arm.
- **Timing**: median and variance of every timed sample, per cell; startup to
  API ready and per-rank expert load time; graph capture/replay timings;
  per-layer group dispatch/reduce/assembly time and achieved imbalance where the
  implementation reports them. Component timing must not be reported as serving
  throughput.
- **Memory**: `runs/v8/capture-startup-memory.sh` (post-readiness RSS and
  per-device residency), `nvidia-smi` before/after each arm, coordinator RSS,
  and the startup log fields `device_occupied_bytes`, `reserved`,
  `runtime_headroom_bytes`, `resident_bytes` per rank.

### 8.6 Failure handling and evidence retention

Failures, timeouts and rejected candidates are kept and reported, not dropped.
Each reported number carries: source commit, submodule revisions, checkpoint
revision, config, exact launch command, corpus/input hashes, arm identity, and
whether it is synthetic, component or end-to-end.

---

## 9. Restoring the original service and preserving provenance

- **Exact current service is recorded** in
  `runs/tp-ep-preflight/logs/baseline-snapshot.txt` (coordinator and Spark
  commands, image digests, config fingerprint, placement/memory log lines).
- **Restore path**: `./run.sh --config ds41rt.config` deploys the released v8
  images and validates their source/dependency identity. If a byte-exact
  restore is required, replay the recorded commands verbatim (coordinator image
  `08c2d6df…`, Spark image `881f51cc1e86…`, native snapshot `dba1be0a…`,
  `--rtx-gpus 1 --dspark`, four peers `10.55.0.1..4:19441`). Verify afterwards
  with `/health`, `/v1/models`, `DS41RT_RELEASE_CONFIG_SHA256`, the
  `rtx_layers=5` / `spark_world=4` log lines and `nvidia-smi` memory.
- `./stop.sh` stops release and WIP services but retains images, slots and
  caches; it must not be run except inside the authorized window.
- **Provenance to preserve for the campaign**: engine commit `bec5fcc…` (or the
  implementation commit), submodule pins (§2.1), checkpoint revision
  `dba1be0a…`, image digests, `DS41RT_RELEASE_CONFIG_SHA256`, and the frozen
  per-slot config. Do not commit experiment artifacts, caches or slots into
  release images (`AGENT_DEV_HINTS.md`).
- No default or release change: `ds41rt.config` and the release launchers must
  resolve to TP4×EP1 after the work.

---

## 10. Readiness verdict (pre-maintenance snapshot)

> **Historical snapshot at 2026-09-20T02:56Z, before the maintenance
> authorization.** Superseded by §12 for current status. Kept for history: the
> "documentation-only" label for six-Spark configs, the absence of a GPU lease,
> and the transport compile defect were true at capture and have since changed.

| Area | Verdict |
| --- | --- |
| Source/submodule identity | **Established** — clean `bec5fcc` + pinned, provenance-matching submodules |
| Hardware presence/health | **Established** — 2× RTX 96 GB (400 W), 4× GB10; 64 cores, 183 GiB RAM |
| CPU test readiness | **Established** — all focused CPU suites pass; one pre-existing committed test-compile defect identified |
| Served-checkpoint identity | **Established** — live native FP4 RTX1 + Spark TP4, image/config hashes recorded |
| Feature/build readiness | **Unknown / blocked** — no group/TP contract in transport, launcher, loader or reducer; world 6 unsupported; native+TP2 disallowed |
| GPU/build lease readiness | **Not established** — the v8 release service is live on GPU0 and all four Spark GPUs; exclusive leases require an authorized maintenance window |
| Six-Spark configs (B/C) | **Hardware-unqualified** — no hardware connected; documentation-only |

### Concrete prerequisites to move from "unknown" to "ready"

1. Resolve B1–B12 in the real tree with the plan's Phases 2–3 gates, keeping the
   TP4×EP1 default byte-identical.
2. Fix `tcp_tests.rs:37` so the transport suite (and its topology tests) can run.
3. Authorize a maintenance window and a single exclusive GPU/build lease;
   record the restore recipe (§9) before stopping anything.
4. Coordinate the WIP dev-image refresh/`--recreate` (currently pinned to a
   stale SparkInfer `63e2140e…`).
5. Obtain the runtime per-rank Spark budget report to answer the `L_r = 20` TP2
   budget hinge (plan open question 1).

---

## 11. Evidence index (`runs/tp-ep-preflight/`)

| Path | Contents |
| --- | --- |
| `logs/baseline-snapshot.txt` | git/submodule/docker/GPU/port/health/coordinator-command snapshot |
| `logs/spark-inventory.txt` | per-Spark containers, release expert commands/ranks, idle WIP top |
| `logs/cargo-transport-core.log` | core pass, transport pre-existing compile failure |
| `logs/cargo-transport-probe.log` | patched-copy transport suite pass (165 + 14) |
| `logs/pytest-launcher-topology.log` | 61 passed + 50 subtests |
| `logs/pytest-scheduler-reduction.log` | 31 passed |
| `rust-probe/` | disposable patched copy used only for §7.3 (not a deliverable; delete before staging) |
| `target/` | isolated cargo target dir |

Nothing here was committed or pushed.

---

## 12. Current status — maintenance window and leases

Appended `2026-09-20` after the user granted full GPU use. This section is the
current operational status; §10 remains a historical snapshot.

### 12.1 Changes since the snapshot

| Item | Snapshot (§10) | Current |
| --- | --- | --- |
| Transport test defect | `cargo test -p ds41rt-transport` blocked at compile | **Fixed in the real tree**; suite now 180 passed + 14 integration, 0 failed |
| Implementation foundations | absent | **Landing**: native CMake/TP2/TP3 sources, reducer planes selftest, transport native-group module, scheduler, loader staging, config/launcher edits, `examples/` |
| Six-Spark configs B/C | documentation-only | **Not docs-only**: code + CPU microbench planned; hardware E2E still unqualified (no six-Spark hardware) |
| Default behavior | TP4×EP1 default | **Preserved by behavior**, not by a byte-identity manifest |
| GPU authorization | none; fleet live | **Full GPU use authorized**; service preservation captured |

### 12.2 Service preservation and exact restore

All five release containers were captured read-only in
`runs/tp-ep-preflight/service-snapshot/`. Restore is
`docker start` of the **same** containers (never remove/replace), because the
served binaries are host bind-mounts that the images do not contain:

- raptor coordinator: `/tmp/ds41-opt/bin/opt1/ds41rt`, sha256 `e0505e1d…`
- each Spark expert: `/home/tj/ds41rt-expert-expertevent2`, sha256 `39f8458b…`
  (identical on all four hosts)

Both are backed up under `/mnt/scratch/ds41rt-tpep/preserved/`. `./run.sh` or a
re-created container would substitute the image binary and is **not** an exact
restore.

Scripts (in `runs/tp-ep-preflight/`): `stop-release-fleet.sh`,
`restore-release-fleet.sh`, `verify-release-service.sh`,
`sync-source-to-sparks.sh`, `source-manifest.sh`.

### 12.3 Isolated environments (no shared WIP touched, no `--recreate`)

| Host(s) | Container | Source | Scratch |
| --- | --- | --- | --- |
| raptor | `ds41rt-tpep-dev` | live repo → `/workspace/ds41rt` | `/mnt/scratch/ds41rt-tpep` → `/scratch` |
| ostrich/dodo/emu/kiwi | `ds41rt-tpep-dev` | `/home/tj/ds41rt-tpep/source` → `/workspace/ds41rt` | `/home/tj/ds41rt-tpep` → `/scratch` |

Toolchain verified current in every container (cargo/rustc 1.98.1, CUDA 13.2,
cmake 3.31.6, ninja 1.13.0, python 3.12.3); the old
`DS41RT_SPARKINFER_COMMIT=63e2140e…` label is metadata only, not a broken
toolchain, so no image rebuild was performed. Ostrich import check passed:
torch `2.12.0a0+5aff3928d8.nv26.05`, cutlass `4.6.2`, `cute` OK, triton `3.7.0`,
`V41SlicePipeline` present.

### 12.4 Lease status (registry: `runs/tp-ep-preflight/leases.md`)

| Lease | Device | Agent | Status |
| --- | --- | --- | --- |
| L1 reducer/native | raptor RTX1 `GPU-95f8f212…` | `f6d0` | **GRANTED** |
| L2 Spark phase0 | ostrich GB10 (exclusive) | `fdcd9d82` | **ACTIVE** (release fleet stopped) |
| L4 minimal official TP2/TP3 AOT | dodo GB10 | this coordinator | **ACTIVE** |
| L3 E2E TP×EP | both RTX + four GB10 | this coordinator | Not scheduled |

L1, L2 and L4 may run concurrently (distinct physical devices; no coupled timing).

### 12.5 L4 dodo AOT preflight (2026-09-20)

Frozen bundle `bec5fcc…`, manifest `fc21d22e…`, SparkInfer verified
`eca542dd…`. Quick proof: TP2 capacity-1 export emitted a real SM121
`v41_spark_tp2_m1.o` with Cutlass/CuTe `V41SlicePipeline` symbols
(`spark_tp_degree=2`, `intermediate=1152`, `capability=[12,1]`). The full
`tp2;tp3` all-capacity build completed with NVFP4/EXL3/coordinator AOT OFF
(§12.6).

Dodo memory (exact bytes, no system changes): MemTotal
`130,594,156,544` (121.6253 GiB), MemAvailable `126,027,001,856`,
CUDA total `130,594,156,544` (identical to MemTotal), CUDA free
`122,619,883,520`; candidate unified-app budget `MemTotal − 20 GiB =
109,119,320,064` B (101.6253 GiB). "128 GB" is not 128 GiB: the measured total
is `0.9502 × 128 GiB`.

### 12.6 L4 dodo AOT result and 137 observation (2026-09-20)

**L4 build (official kernels only, dodo SM121).** Frozen bundle engine
`bec5fcc…`, manifest `fc21d22e…`, SparkInfer `eca542dd…` (lock value), nvcc 13.2.
CMake with `DS41RT_V41_SPARK_TP_ROLES="tp2;tp3"`, NVFP4/EXL3/coordinator/W8A16
AOT OFF, XGRAMMAR OFF, RDMA+NCCL ON. Produced `libds41rt_native.so`
(16,418,544 B, sha256 `d453812c5c53e916fe8ea0b54d72a51c19b0df8e4db8ce9b7c1ea5b967b1f513`),
only `v41_spark_tp2_experts` and `v41_spark_tp3_experts` role directories (no
`exl3/`, no `nvfp4/`), and 6 exported capacity symbols per role
(`m1,m16,m80,m256,m1024,m4096`). The role manifest was written by the current
fa33 writer with `--native-library` (frozen writer hash `4b1f2a84…` equals the
live-root writer); it reports `symbols_verified=true` via `nm`,
`verified_artifacts=16` per role.

Emitted variants (actual `core_scratch_nbytes`; TP2 intermediate 1152, TP3 768):

| Cap | Width | TP2 scratch | TP3 scratch | ABI / kind |
| ---: | ---: | ---: | ---: | --- |
| 1 | 64 | 2,352,160 | 1,614,880 | 2 / fp32_routes |
| 16 | 192 | 13,925,776 | 9,993,616 | 2 / fp32_routes |
| 80 | 192 | 69,598,096 | 49,937,296 | 2 / fp32_routes |
| 256 | 192 | 7,738,912 | 7,738,912 | 3 / fp32_tokens |
| 1024 | 192 | 30,932,512 | 30,932,512 | 3 / fp32_tokens |
| 4096 | 192 | 123,706,912 | 123,706,912 | 3 / fp32_tokens |

Runtime self-tests executed on genuine dodo SM121 under lease L4:
`ds41rt_v41_route_reduce_planes_selftest` → ok (exit 0);
`ds41rt_v41_expert_pack_tp3_selftest` → ok, `intermediate=768`,
`3932160+245760+1966080+122880 bytes` (exit 0).

**Container 137 observation — cause unknown.** On `ostrich` and `dodo` the
isolated `ds41rt-tpep-dev` and the shared `ds41rt-spark-expert-wip` show
`ExitCode=137`, `OOMKilled=false`, `RestartCount=0`; ostrich pair finished
`2026-09-20T05:43:20Z` (18 ms apart) and dodo pair `05:44:50Z` (10 ms apart);
emu/kiwi were unaffected. `OOMKilled=false` only excludes the *recorded
container-cgroup* OOM; kernel OOM history was inaccessible and current memory
readings cannot prove the absence of an earlier host-level OOM. The ~90 s
per-host correlation is an observation, **not** a causal verdict. Metadata
retained in `runs/tp-ep-preflight/service-snapshot/137/`. The isolated ostrich
container was later restarted by `docker start` of the same container.
