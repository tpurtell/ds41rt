# Opt-in Spark TP x EP replicated expert groups

Status: **packaged and launchable; still not a qualified release.** The native
`--spark-tp`/`--spark-ep` parsing and wire geometry landed and have run on real
hardware: the manual G1 flow (four Sparks, N20 boundary) and two completed
six-rank dual arms (`2rtx6-tp3ep2`, `2rtx6-tp2ep3`) that an independent audit
accepted for the canonical 372-row scope **with strict quality FAIL** (known
greedy-drift class; see §3a). Packaging and qualification are separate claims:
the published v9 Spark image is **universal** and advertises
`io.ds41rt.v41.spark_tp_roles=tp2;tp3;tp6` alongside the default TP4 shard
([release-v9-notes.md](release-v9-notes.md)), so one published pair serves every
approved topology and `run.sh` selects the role from `SPARK_TP`. Bounded
final-image functional checks passed for TP6 and the implicit TP4 layout on one
and two RTX; that is **not** coverage of all five topologies, the canonical
frozen six-rank qualifier was not run, and no final-image performance number is
claimed. The default `ds41rt.config` serving selection is unchanged.

Related: [tp-ep-implementation-plan.md](tp-ep-implementation-plan.md) (design
and phase gates), `examples/configs/README.md` (runnable example files),
[six-independent-review.md](../runs/tp-ep-preflight/g2-live/six-independent-review.md)
(independent audit),
[SIX-CONSOLIDATED.md](../runs/tp-ep-preflight/g2-live/SIX-CONSOLIDATED.md)
(consolidated results),
[final-acceptance-review.md](../runs/tp-ep-preflight/final-acceptance-review.md)
(final acceptance review).

## 1. What is configurable today

`scripts/release-common.sh` accepts two new **optional** configuration keys:

| Key | Meaning | Allowed |
| --- | --- | --- |
| `SPARK_TP` | tensor-parallel degree inside one expert group | `2`, `3`, `4`, `6` |
| `SPARK_EP` | number of replicated expert groups (each holds every expert) | `1`, `2`, `3` |

Rules enforced while loading a configuration:

- The keys are **all-or-none**. Only one of them is an error.
- `SPARK_TP * SPARK_EP == SPARK_COUNT`.
- `SPARK_COUNT` is now `0`, `2`, `4` or `6`. `6` requires explicit keys.
- Only the approved native official layouts are accepted:
  `TP2xEP2`, `TP3xEP2`, `TP2xEP3`, the explicit legacy `TP4xEP1`, and the
  **pure unreplicated `TP6xEP1`** (see §14).
- Explicit keys require `EXPERT_FORMAT=native`, `SPARKINFER_EXL3=disable` and
  `EXL3_PAIRED_TP4=off`. EXL3 and NVFP4 keep their existing, separate behavior;
  they are only rejected when an explicit replicated topology is requested.
- `RTX_GPUS` is **not** restricted by the explicit-topology validator:
  `release-common.sh` checks only `SPARK_COUNT` (4 or 6), native format, EXL3
  off and the approved `2x2|3x2|2x3|4x1|6x1` set. Both RTX counts are exercised —
  the tested `six-1rtx6-tp3ep2` and `six-2rtx6-tp3ep2` configs use the same
  TP3xEP2 topology with 1 and 2 RTX respectively.
- `SPARK_0`..`SPARK_5` host and `LANE_A` keys are validated for every active
  rank. `LANE_B` remains optional but must be present for **all** active ranks
  or for none, exactly as before.

The rank map is frozen and group-major:

```
group   = global_rank / SPARK_TP
tp_rank = global_rank % SPARK_TP
```

| Layout | Ranks | RTX | Spark map |
| --- | --- | --- | --- |
| default / explicit `TP4xEP1` | 4 | 2 | `0:(0,0) 1:(0,1) 2:(0,2) 3:(0,3)` |
| `TP2xEP2` | 4 | 2 | `0:(0,0) 1:(0,1) 2:(1,0) 3:(1,1)` |
| `TP3xEP2` | 6 | 1 | `0..2:(0,0..2) 3..5:(1,0..2)` |
| `TP2xEP3` | 6 | 2 | `0,1:(0,0/1) 2,3:(1,0/1) 4,5:(2,0/1)` |
| `TP6xEP1` (pure) | 6 | 1 or 2 | `0..5:(0,0..5)` — one unreplicated group |

`release_spark_rank_map` in `scripts/release-common.sh` is the single launcher
source of that map and is unit-tested for every approved layout.

## 2. Launcher / daemon CLI contract

The launcher emits the new flags **only** for an explicit topology. The legacy
default emits exactly the previous argument vector.

| Process | Legacy args (unchanged) | Added when explicit |
| --- | --- | --- |
| coordinator (`serve-native`) | `--peers <all rank addresses>`, `--rtx-gpus`, ... | `--spark-tp N --spark-ep M` |
| Spark worker (`expertd-native`) | `--rank <global rank> --world <TP*EP>` | `--spark-tp N --spark-ep M` |

`--rank` stays the physical global rank and `--world` stays the number of
physical Spark ranks, so existing daemon semantics are retained. The worker
derives `group` and `tp_rank` from the same two topology flags as the
coordinator; no dummy hosts or ranks are synthesized. The coordinator peers
list always contains all `SPARK_TP * SPARK_EP` real rank addresses.

`SPARK_TP`/`SPARK_EP` are configuration-only; `run.sh` has no CLI override for
them, matching the other topology-shaping keys.

> The daemon-side `--spark-tp` / `--spark-ep` parsing, wire fields, TP3 3-plane
> reduction and six-rank EP assembly have **landed** and been exercised on real
> hardware. Do not convert that into release support or a qualification claim:
> the completed replicated-group runs are audited experiments with a strict
> quality FAIL. Packaging is a separate claim from qualification: the published
> universal v9 Spark image carries the `tp2`/`tp3`/`tp6` roles (§1), so these
> layouts are launchable from the published pair, and only TP6 and the implicit
> TP4 layout have bounded final-image functional checks.
>
> Entry points differ and do **not** all support the same surface:
> `run.sh` (release) takes topology from configuration only and hands off the
> resolved boundary on one RTX as well as two for an explicit local count in
> 1..=39 (§3); `scripts/run-wip.sh` is the legacy four-Spark WIP path and
> **rejects** explicit `SPARK_TP`/`SPARK_EP`;
> `scripts/run-tp-ep-native-candidate.sh` is the candidate six-rank path, which
> requires explicit topology, accepts 1 or 2 RTX, and is single-rail A-only.

## 3. Admission: weight-only, never a feasibility claim

`run.sh` derives the Spark request from the **actual resolved RTX/Spark
boundary**, not a hardcoded 20:

- `spark_first_layer` is the explicit `RTX_EXPERT_LAYERS` when given, `0` for a
  single-RTX layout, and the coordinator-published `plan.json` value for an
  automatic dual-RTX launch.
- `release_spark_remote_layers` = `40 - spark_first_layer`.
- `release_spark_layer_bytes` returns the exact native per-TP-rank routed weight
  for one layer:
  `TP2 = 3,609,722,880 B`, `TP3 = 2,406,481,920 B`,
  `TP4 = 2,005,401,600 B` (576 padded to a 640 kernel extent),
  `TP6 = 1,203,240,960 B` (384, already 128-aligned, no padding).
- `release_validate_spark_weight_admission` compares
  `remote_layers * layer_bytes` with `SPARK_DEVICE_BUDGET_BYTES`. Overflow is a
  hard failure before any service change; a fit prints the residual and
  `workspace_accounted=no`.

Consequences at the tested/global-minimum candidate budget
`109,119,320,064 B` (used by all five tested configs in §5 and by the G1 and
six-rank runs):

| TP | Zero-workspace ceiling | Meaning |
| --- | --- | --- |
| 2 | 30 remote layers | **at least 10 RTX-local layers** are required for the weights alone; workspace needs more |
| 3 | 40 remote layers (96.26 GB) | all layers can be remote on weights alone |
| 4 | 40 remote layers (80.2 GB padded) | release geometry |
| 6 | 40 remote layers (48.13 GB) | pure TP6; even the 100 GiB default fallback admits all 40 remote layers |

The default `ds41rt.config` fallback remains 100 GiB (`107,374,182,400 B`) and is
unchanged; the tested configurations raise it to the candidate minimum above.

**Admission and the serve gate are different checks.** The launcher admission is
weight-only. The runtime serve gate is **CUDA-free based** (`cuda_free_bytes`),
not host RSS: CUDA free can be below the serve peak and the gate then refuses.
The 1-RTX + 6-Spark TP3xEP2 arm loaded **all 40** routed layers after the
execution agent used the user-authorized installed `sync; drop_caches=1` helper
(system-wide clean page-cache reclaim). See the run-attributed evidence in
`runs/tp-ep-preflight/g2-live/six/singleRtx-success-memory.md`. This operational
workaround does not establish whether the earlier CUDA-free-only gate was
necessary: the rejected attempts never reached weight allocation. Do not lower the
configured value on a transient undercount, and do not assume 128 GiB from an
advertised 128 GB part.

For an automatic dual-RTX launch the boundary is unknown at dry-run time, so
`run.sh` prints `PENDING` and re-runs the same check after it reads the
coordinator's `plan.json`, before any Spark worker starts. The plan carries the
coordinator's **actual** `rtx_gpus` (1 or 2); the launcher requires it to match
the layout it selected and derives `spark_first_layer` from the real dynamic
boundary. A dry-run pass is explicitly labelled **not a launch-feasibility
claim**: SparkInfer workspace, load staging, replicated activation buffers and
runtime headroom are only known from the expert service startup report.

### 3a. Current results (audited experiments, strict quality FAIL)

- Two six-rank dual arms completed the canonical 372-row v4 scope:
  `2rtx6-tp3ep2` and `2rtx6-tp2ep3`. The independent audit accepted both as
  correctly matched with reproducible headline numbers
  ([six-independent-review.md](../runs/tp-ep-preflight/g2-live/six-independent-review.md),
  [SIX-CONSOLIDATED.md](../runs/tp-ep-preflight/g2-live/SIX-CONSOLIDATED.md)).
- **Strict quality FAIL is preserved:** exactly three quality failures (code /
  topic / mixed greedy drift), with `counting-1-64` greedy-consistent; the audit
  classifies these as the known greedy-drift class, not a new six-rank defect.
- The audit's §7 also accepts the single-RTX TP3xEP2 arm as an executed standalone
  372-request experiment: actual argv confirms one RTX, all 40 layers remote,
  no placement directory, auto KV and draft 5. It retains the same three strict
  repeatability failures. Its hardware/settings differ from the dual arms;
  no matched topology-speed claim follows from comparing them.
- See [tp-ep-results-summary.md](tp-ep-results-summary.md) for verified numbers
  across all five deployments and the bounded supplemental semantic review.
- Qualification status: **not qualified**. These are experiment results, not
  release support.

Single-RTX placement in the **release** launcher (`run.sh`) is **implemented**
for an explicit local count: with `RTX_EXPERT_LAYERS` in 1..=39 and an explicit
topology, `run.sh` passes `--placement-directory` on one RTX, the coordinator
publishes its real boundary before anything waits on it (the Spark transport
connects lazily, so the old deadlock rationale is gone), the launcher reads the
plan, starts every worker at that layer and only then acknowledges readiness.
`auto` and `0` on one RTX have no local boundary to hand off and keep the
historical no-handoff launch: every worker starts at layer 0 and loads all 40
routed layers, and only an explicit `0` guarantees that all 40 are also
*dispatched* remotely — under `auto` the coordinator still places local layers
that those workers have already reserved.
That is the shape the tested `six-1rtx6-tp3ep2` arm ran (`--rtx-expert-layers 0`,
no placement directory) and the shape
`examples/configs/tp3ep2-native.config` pins; `RTX_EXPERT_LAYERS=auto` there
would let the coordinator keep local layers the workers already reserve.

Daemon-level `TP2EP1` remains not launcher-selectable and stays a low-level
diagnostic that must be driven manually, so no user-facing customization is
implied there. The three-rank `SPARK_COUNT=3` gate opened for v10 as an opt-in
launcher selection, and the two three-rank forms are deliberately distinct:
explicit `SPARK_TP=3 SPARK_EP=1` is the native `TP3EP1` expert-group topology
(`examples/configs/tp3ep1-native.config`), while `SPARK_COUNT=3` with **no**
`SPARK_TP`/`SPARK_EP` keys is the implicit compact EXL3 TP3 layout
(`examples/configs/exl3-compact-tp3.config`), never a native one. Both target
the v10 pair and carry examples-only status; this paragraph records launcher
selectability and nothing about qualification.

## 4. Identity

The release fingerprint now includes `spark-topology=<TP>x<EP>:explicit=<0|1>`
so two arms that differ only by topology produce different deployment
identities. The legacy WIP launcher binds the resolved topology into its expert
runtime identity and exports `DS41RT_SPARK_TP` / `DS41RT_SPARK_EP` alongside
`DS41RT_SPARK_HOSTS`.

## 5. Examples and tested configurations

Standalone `--config` files live in `examples/configs/` (unchanged): the
user-facing overlays `tp4ep1-explicit-native.config`, `tp2ep2-native.config`,
`tp3ep2-native.config` and `tp2ep3-native.config`. The five configurations used
for the current results live under `runs/tp-ep-preflight/`:

| Config | Layout | RTX | Status |
| --- | --- | --- | --- |
| `runs/tp-ep-preflight/baseline-tp4ep1.config` | TP4 x EP1, 4 ranks | 2 | control arm |
| `runs/tp-ep-preflight/candidate-tp2ep2.config` | TP2 x EP2, 4 ranks | 2 | completed 372 requests; strict quality FAIL; not release-qualified |
| `runs/tp-ep-preflight/six-1rtx6-tp3ep2.config` | TP3 x EP2, 6 ranks | 1 | completed all 40 remote; independently audited; strict quality FAIL |
| `runs/tp-ep-preflight/six-2rtx6-tp3ep2.config` | TP3 x EP2, 6 ranks | 2 | completed; audited — strict quality FAIL (greedy drift) |
| `runs/tp-ep-preflight/six-2rtx6-tp2ep3.config` | TP2 x EP3, 6 ranks | 2 | completed; audited — strict quality FAIL (greedy drift) |

All five set `SPARK_DEVICE_BUDGET_BYTES=109119320064`. The shipped
`examples/configs/{tp2ep2,tp3ep2,tp2ep3}-native.config` overlays now set the same
tested floor `109119320064` (replacing the earlier illustrative
`109186371584`); the control `tp4ep1-explicit-native.config` carries no budget
override and keeps the 100 GiB default fallback.

The six-rank configs use `rhea` (global rank 4) and `moa` (rank 5) from
[cluster-hosts.md](cluster-hosts.md) and leave `LANE_B` unset. The candidate CLI
is single-rail A-only, so the legacy `LANE_B` default range is unused and there
is no active secondary-rail collision to work around. TP3×EP2 and TP2×EP3 are
connected and have completed runs, but **not qualified**: no memory,
correctness, performance or readiness claim is attached, the dual arms have a
strict quality FAIL, and no gate may be marked passed from them.

## 6. Build plumbing and the role-manifest trust boundary

The optional CMake switch `DS41RT_V41_SPARK_TP_ROLES` (`tp2`/`tp3`/`tp6`, empty
by default) is passed through by both build helpers:

| Build | Env override | Default |
| --- | --- | --- |
| `build.sh` → `scripts/build-release-artifacts.sh` | `DS41RT_RELEASE_SPARK_TP_ROLES` | derived from an explicit `SPARK_TP` (2→`tp2`, 3→`tp3`, 6→`tp6`); empty for the legacy default |
| `wip.sh` → `scripts/build-wip-artifacts.sh` | `DS41RT_WIP_SPARK_TP_ROLES` | same derivation |
| WIP official-only scope | `DS41RT_WIP_EXL3_AOT`, `DS41RT_WIP_NVFP4_AOT` | both `ON`; `OFF` skips the quantization AOT and its package/verify |

`build.sh --dry-run` and `wip.sh --dry-run` validate the configuration, the
six-host build/distribution plan and the selected roles without touching
Docker, SSH, submodules or any image.

Every release/WIP expert artifact gets a mandatory `V41_EXPERT_TP_AOT.json`
(also written for the empty-role default) that records the built roles and is
copied into the image (`COPY .../V41_EXPERT_TP_AOT.json`) and advertised as the
label `io.ds41rt.v41.spark_tp_roles`. `scripts/write-v41-expert-tp-manifest.py`
derives it from the AOT export manifests CMake actually produced and validates:

- `schema == 1` and `capability == [12, 1]`;
- geometry: experts 384, hidden 5120, intermediate 1152 (tp2) / 768 (tp3),
  `kernel_intermediate == intermediate`, topk 6;
- non-empty `variants` covering capacities 1/16/80/256/1024/4096;
- every file in `artifact_sha256` exists and hashes to its declared value
  (including at least one compiled object);
- the built `libds41rt_native.so` hash, and — when `nm -D` or `readelf` is
  available — the role's dynamic `_info`/`_launch` symbols.

**Trust boundary.** The manifest binds the validated export contents and the
built library hash, and symbol verification proves the role module was linked
into that library. It does not execute kernels, prove numerics or prove device
compatibility. Those remain hardware-qualification gates. `run.sh` additionally
requires the requested role to be advertised by the Spark image label and fails
before any service is stopped; a prebuilt legacy v8 image with no label keeps
working for the default `TP4xEP1` path.

## 7. What is explicitly unchanged

- `ds41rt.config` is untouched and still resolves to `SPARK_COUNT=4`,
  `TP4xEP1`, native official V4.1 Flash, v8 images, 100 GiB Spark fallback.
- Absent `SPARK_TP`/`SPARK_EP` means the legacy geometry for `SPARK_COUNT`
  `0`, `2` and `4`, including the `SPARK_COUNT=2` EXL3 compact path and its
  32 GiB ceiling.
- The default launch argument vector, peer list, images, fingerprint inputs
  other than the added topology and role tags, and serving selection are
  unchanged; a prebuilt v8 image without any role label still launches the
  default.
- EXL3 and NVFP4 launch paths are unchanged when no explicit topology is set.
- The legacy phase0 Spark backend is not expanded: it keeps its fixed
  four-plane TP4 wire and four-host gate.

## 8. WIP path

`scripts/run-wip.sh` remains the **legacy** four-Spark WIP entry: it requires
`SPARK_COUNT=4`, iterates the active `SPARK_0..5` bounds, binds the resolved
topology into its expert identity, and **rejects** explicit `SPARK_TP`/
`SPARK_EP` because `scripts/phase0-spark-tcp-bench.sh` runs the legacy
protocol-v2 `expertd` backend, which does not implement the replicated TP×EP
wire. Advertising that path as support would be wrong.

The correct native WIP route is to reuse the native entry (`run.sh` /
`serve-native` + `expertd-native`) with candidate artifacts once the integration
workstream (`91570601`) provides the artifact override; the daemon takes the
topology as CLI flags (`--spark-tp`/`--spark-ep`) and derives group/tp from the
global rank, with no env parsing. The WIP build helpers already build the role
artifacts and record them in the slot `META.json` (`spark_tp_roles`,
`v41_expert_tp_manifest_sha256`) from the hash-verified artifact.

## 9. Native candidate launcher (isolated experiment)

Status: **the isolated runs are hardware-exercised through the manual/candidate
flow; the launcher's own automated start gates are still CPU-tested only.** The
coordinator x86-64 and Spark AArch64 artifacts are staged on all six ranks, and
the frozen daemon/libraries ran the G1 four-Spark flow plus the six-rank dual
arms (`2rtx6-tp3ep2`, `2rtx6-tp2ep3`; strict quality FAIL per §3a). A `plan`
pass is not a readiness claim.

`scripts/run-tp-ep-native-candidate.sh` runs the frozen candidate daemon +
native libraries in the isolated `ds41rt-tpep-nvme-dev` (raptor) and
`ds41rt-tpep-dev` (Sparks) containers. It never creates, renames or removes a
container, never touches a production name and never builds an image. The
official snapshot is confirmed present in all six Spark containers (four
original Sparks plus rhea/moa) at
`/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277`.

- `plan` (default / `--dry-run`) prints every command vector; `start` runs them
  and requires `DS41RT_TPEP_L3_GRANT=1`; `status`/`stop` act only on the
  `ds41rt-tpep-candidate-*` processes and fail closed on unreachable hosts.
- Native CLI: `serve-native`/`expertd-native` with `--spark-tp`, `--spark-ep`,
  global `--rank`, `--world == TP*EP`, and the matched release controls
  (prefill batch, concurrency, prefix cache, context/output limits, HTTP queue,
  dSpark, host cache, KV pool, memory reservation, TP2 switches, placement).
- 2 RTX: coordinator starts, the plan is read for the **actual** `rtx_gpus` and
  `spark_first_layer` before any worker, then the plan is acknowledged. Any
  `--first-layer` must exactly equal the published boundary.
- 1 RTX: no handoff, workers load from layer 0, `--first-layer` must be 0.
- Each run uses a unique placement directory; a reused directory is refused
  unless `--restart` explicitly clears this run's directory.
- Runtime env is explicit per `docker exec`: `DS41RT_NATIVE_LIB=<role lib>`
  (the transport's only native-library path; without it verbs falls back to a
  relative `native/build*` candidate) matching `--native-lib`, and
  `DS41RT_WIP_RUNTIME_ROOT=/scratch/candidate/run` so candidate PIDs/logs live
  under the artifact bind. Optional RDMA/native keys are forwarded only when the
  operator sets them (`DS41RT_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP`,
  `DS41RT_VERBS_APP_IB_PORT_NUM`,
  `DS41RT_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES`, `DS41RT_PYTHON`). Release
  quant/FP4 env is never cloned. The frozen binary is expected to resolve
  libpython from the container's system path/RPATH; set `DS41RT_PYTHON` only if
  that resolution needs an override. Release images carry
  `ENV DS41RT_NATIVE_LIB=/opt/ds41rt/lib/libds41rt_native.so`, but the isolated
  `ds41rt-tpep-*` dev containers do not, so the candidate launcher always passes
  it explicitly.
- Provenance is fail-closed: `--qualified` requires coordinator/Spark library
  and daemon sha256, the per-role manifest hash, and `readelf` architecture
  (x86-64 coordinator, AArch64 Spark).
- Readiness is staged and real: workers start, then every rank must log the
  current process's `native local RoCE expert worker ready` line with its
  `rank`/`world`/`first_layer` before the plan is acknowledged; only then does
  the API `/health` + native model + every-rank-process gate run. A worker that
  dies before that line fails fast. The wait is bounded by
  `DS41RT_TPEP_WORKER_READY_TIMEOUT_SECONDS` (default 900). Because
  `wip-process.sh` appends to its log, the byte size is captured before each
  worker starts and only bytes after that offset are searched, so a previous
  run's line cannot satisfy readiness; use
  `--wip-runtime-root /scratch/candidate/run/<run-id>` to isolate a manual run.
  The real tracing log inserts ANSI dim/italic codes between a field name and
  its `=`, so the launcher strips CSI sequences before matching
  `rank`/`world`/`first_layer` (verified against the raw G1
  `runs/tp-ep-preflight/g1-live/raw/dodo-expert.log` line).
- `RUST_LOG=${RUST_LOG:-info}` is forwarded to both roles: the daemon's
  `EnvFilter::from_default_env()` is `ERROR` when unset, which hides the
  readiness/placement `tracing::info` lines. A user filter that hides those
  lines makes the readiness waits time out (nonzero) — the intended fail-closed
  behaviour.
- The script is single-rail A-only by design and does not fabricate a dual-rail
  endpoint. `TP4EP1` is accepted as the matched baseline reference arm (no extra
  role manifest entry).
- The release fleet is currently stopped by the TP×EP preflight leases. This
  launcher preserves the containers but does not restore the baseline; that is
  the separate explicit operation
  `runs/tp-ep-preflight/restore-release-fleet.sh`.

## 10. Fabric addresses (2026-09-20 audit)

Actual addresses from [fabric-inventory.md](../runs/tp-ep-preflight/fabric-inventory.md):

| Host | Fabric A | Fabric B |
| --- | --- | --- |
| raptor | 10.55.0.22 | — |
| ostrich | 10.55.0.1 | 10.55.0.7 |
| dodo | 10.55.0.2 | 10.55.0.8 |
| emu | 10.55.0.3 | 10.55.0.9 |
| kiwi | 10.55.0.4 | 10.55.0.10 |
| rhea | 10.55.0.5 | 10.55.0.11 |
| moa | 10.55.0.6 | 10.55.0.12 |

There is no duplicate address. The **default `ds41rt.config` secondary rail is
stale** — its `SPARK_0..3_LANE_B = .5/.6/.7/.8` now names rhea/moa fabric A and
ostrich/dodo fabric B — and defaults are deliberately unchanged. This is not an
active TP/EP blocker: the candidate CLI is single-rail (A only) by design, the
tested six-rank configs leave `LANE_B` unset, and the four-Spark candidate
examples that set a secondary rail use the corrected `.7/.8/.9/.10`. Flag the
stale default to any user intending a four-Spark secondary-rail launch, and do
not use the default (topology-less) config with the candidate launcher, which
requires explicit `SPARK_TP`/`SPARK_EP`.

## 11. Integration status

| Requirement | Owner | Status |
| --- | --- | --- |
| Native candidate launcher (`scripts/run-tp-ep-native-candidate.sh`) | this workstream | script + CPU tests landed; automated start gates not live-exercised |
| ARM64 candidate daemon build (`expertd-native` for the Sparks) | integration `91570601` | built, staged and exercised on hardware (G1 and six-rank arms; artifact `5162bc8a…`, Spark lib `d453812c…`) |
| Uniform per-rank library stage (`/scratch/candidate/…`) | integration `91570601` | staged on all six ranks (four original Sparks + rhea/moa) |
| Native WIP artifact override on `run.sh` (replaces legacy phase0) | integration `91570601` | manual G1/six-rank native flow used instead; WIP override PENDING |
| Daemon `--spark-tp` / `--spark-ep` parsing and wire geometry | daemon `6ba87` | landed and exercised on hardware (G1 N20; six-rank dual arms) |
| Candidate budget | memory `885b` / integration `91570601` | used: `109,119,320,064 B` in all five tested configs; serve gate is CUDA-free based |
| Single-RTX placement boot-ordering (then re-enable the release 1-RTX handoff) | daemon/integration | **landed**: `run.sh` opens the handoff on one RTX for an explicit local count in 1..=39 (§3); `auto`/`0` keep the historical no-handoff launch, which is how the candidate 1-RTX arm ran |
| Release pair that serves every approved topology | build/publish | **landed in v9**: the published Spark image is universal (`io.ds41rt.v41.spark_tp_roles=tp2;tp3;tp6`) and `run.sh` selects/verifies the role per topology; see [release-v9-notes.md](release-v9-notes.md) |
| Matched TP4 vs TP2×EP2 decode/prefill campaign | performance | PENDING |
| Six-rank RDMA GID/interface selection (rhea/moa multi-homing) | integration | exercised in the six-rank runs; explicit per-rail selection still required |
| Independent audit of the completed six-rank dual arms | review | done — accept for the canonical 372-row scope, strict quality FAIL preserved |

## 12. Test coverage

`scripts/tests/test_tp_ep_configuration.py` (CPU-only, no Docker/SSH/GPU):

- default `ds41rt.config` still resolves to `TP4xEP1` and is not mutated;
- rank map for `TP2xEP2`, `TP3xEP2`, `TP2xEP3`, `TP4xEP1`;
- invalid combinations, product mismatch, six-host-without-topology, unknown
  keys, all-or-none secondary rail; any `RTX_GPUS` parses and infeasibility is
  a budget result, not a topology hardcode;
- native-only rejection (EXL3 / NVFP4 wrapper flags);
- exact layer bytes and admission thresholds (`TP2` needs ≥ 11 RTX layers at the
  100 GiB default fallback, ≥ 10 at the tested `109,119,320,064 B` budget),
  including a dynamic-boundary verdict flip;
- every example config parses with its approved geometry;
- the real `run.sh` startup block with process stubs: four/six workers start,
  the coordinator receives the same topology flags and all real peers, and the
  legacy launch receives none;
- the manifest writer rejects stale/partial exports (schema, capability,
  geometry, capacity coverage, artifact hash, missing library) with exit 2;
- `build.sh --dry-run` and `wip.sh --dry-run` cover a six-host, role-selected,
  official-only plan without invoking Docker/SSH;
- every example config names the published release pair `ds41rt.config` names
  (no per-topology local tag can reach a host that only pulled the published
  images), and the pair carries one release tag on both roles;
- the real `run.sh` per-host Spark preflight runs through an OpenSSH-style shim
  that elides empty arguments, so the host name survives an empty EXL3 family
  tag and a failing check names its host;
- the real `run.sh` role gate: the universal label satisfies `tp2`/`tp3`/`tp6`,
  a subset or legacy unlabeled image is refused before any service change, and
  the default TP4 launch never probes it;
- `run-wip.sh` rejects the replicated wire on the legacy backend;
- candidate launcher: matched control flags present, coordinator-only GPU pin
  validation, `DS41RT_NATIVE_LIB`/`RUST_LOG`/runtime-root env before the
  container and matching `--native-lib`, 2-RTX
  plan→workers→worker-ready→ack→API-ready ordering, six-rank single-RTX
  experts-first ordering, TP4EP1 reference, explicit-topology requirement,
  corrected secondary rail, and CPU stubs that simulate a stale placement
  directory, a stale pre-offset worker log (times out), a wrong-rank/split-line
  readiness line with numeric boundaries, a fatal SSH/stat log-size error
  (never a silent 0), worker death before ready, coordinator child exit and
  unreachable-host `stop`.

## 13. Artifact identity names

| Surface | Name |
| --- | --- |
| Config keys | `SPARK_TP`, `SPARK_EP` |
| Release worker/coordinator CLI | `--spark-tp`, `--spark-ep` (legacy `--rank`, `--world` retained) |
| Build env (release) | `DS41RT_RELEASE_SPARK_TP_ROLES`, `DS41RT_RELEASE_EXL3_PAIRED_TP4` (existing) |
| Build env (WIP) | `DS41RT_WIP_SPARK_TP_ROLES`, `DS41RT_WIP_EXL3_AOT`, `DS41RT_WIP_NVFP4_AOT` |
| CMake | `DS41RT_V41_SPARK_TP_ROLES` |
| Image label | `io.ds41rt.v41.spark_tp_roles` |
| Artifact manifest | `V41_EXPERT_TP_AOT.json` |
| Release fingerprint tag | `spark-topology=<TP>x<EP>:explicit=<0|1>`, `v41-spark-tp-roles=<role>` |

## 14. Pure unreplicated `TP6xEP1` (opt-in, UNQUALIFIED)

**Status: configuration, runtime, native-role, admission and CPU-test plumbing
are implemented and locally validated (Rust unit/integration tests plus
CPU-only script gates). No hardware run has been performed for TP6, and nothing
in this section is a memory, correctness, readiness, quality or performance
claim.** It does not change `ds41rt.config`, the release images or the default
serving selection.

### What it is, and what it is not

`TP6xEP1` is **one unreplicated group of six Spark ranks**. Each rank holds a
disjoint `2304 / 6 = 384` column slice of **every** routed expert of every
remote layer, and the coordinator sums the six compact BF16 `[M, 5120]` rank
planes once. It is therefore *not* an expert-parallel layout:

| | `TP2xEP3` / `TP3xEP2` (replicated groups) | `TP6xEP1` (pure) |
| --- | --- | --- |
| Copies of each expert | `EP` copies, one per group | exactly one |
| Per-rank intermediate | 1152 / 768 | 384 |
| Kernel storage padding | none | none (384 is already 128-aligned) |
| Wire planes per batch | `TP * EP` physical planes | 6 physical planes |
| Coordinator reduction | sum `TP*EP` planes | sum 6 planes (same generic entry point) |
| Route ownership word | one-hot group bit per expert | always group 0 |

Because no expert is duplicated, TP6 uses six ranks to make each rank's resident
weight set **smaller** (`1,203,240,960 B` per rank per layer versus `2,005,401,600 B`
at TP4 and `3,609,722,880 B` at TP2), at the cost of one more wire plane and one
more term in the coordinator reduction than TP4. It is a resident-footprint and
shard-width experiment, not an expert-duplication strategy.

### Implemented surfaces (unqualified)

| Surface | Where |
| --- | --- |
| Topology + disjoint executor namespace `27..=32` | `rust/crates/ds41rt-transport/src/v41_expert/native_group.rs` |
| Six-plane compact reduction (`ranks=6`) | `native/cuda/kernels/v41_route_reduce.cu`, FFI `V41CompactReducer::reduce_planes` |
| Worker shard geometry (`world: 6` → native role 7) | `rust/crates/ds41rt-daemon/src/v41_experts.rs`, `rust/crates/ds41rt-daemon/src/v41_spark_topology.rs` |
| CLI parsing (`--spark-tp 6 --spark-ep 1`) | `rust/crates/ds41rt-daemon/src/cli.rs` |
| Checkpoint staging (384-column W2 slice, 32-group aligned) | `rust/crates/ds41rt-loader/src/v41_expert_staging.rs` |
| Role AOT export (`spark_tp6`, intermediate 384) | `python/tools/export_b12x_v41_{slices,experts}_aot.py`, `native/cmake/v41_spark_tp_experts.cmake` |
| Launch config | `examples/configs/tp6ep1-native.config`, `scripts/fixtures/tp-ep-six/site-{1,2}rtx6-tp6ep1.config` |
| Qualification gate + corpus | `scripts/qualify-ds41-tp-ep-e2e-six.py`, `scripts/fixtures/tp-ep-e2e-corpus-v5-six.jsonl` (v5), startup templates `scripts/fixtures/tp-ep-six/startup-{1,2}rtx6-tp6ep1.template.json` |
| CPU tests | `scripts/tests/test_tp_ep_six_readiness_tp6.py`, transport `pure_tp6_*`, loader TP6 staging coverage, FFI role-7 geometry |

The **v4 six-rank corpus is unchanged** (`sha256 99034171…`) and its accepted
results keep binding that revision. v5 adds the two TP6 arms; its header records
that it is a new, unexecuted revision, its `future_scope` names the pure TP6
family, and the approved v5 arm set is a strict superset of v4 (nothing was
removed or renumbered). The qualifier's `report["six"]["schema"]` is derived from
the corpus header, so a v5 run never reports itself as v4.

`--corpus` defaults to **v4** for compatibility with the accepted arms. A TP6
arm must name v5 explicitly:

```bash
# CPU-only schema/identity preflight for the six-rank corpus
python3 scripts/qualify-ds41-tp-ep-e2e-six.py validate \
  --corpus scripts/fixtures/tp-ep-e2e-corpus-v5-six.jsonl

# Drive one TP6 arm (requires the hardware lease and a filled startup metadata)
python3 scripts/qualify-ds41-tp-ep-e2e-six.py run \
  --corpus scripts/fixtures/tp-ep-e2e-corpus-v5-six.jsonl \
  --arm 2rtx6-tp6ep1 \
  --startup-metadata runs/tp-ep-six/startup-2rtx6-tp6ep1.json \
  --output runs/tp-ep-six/2rtx6-tp6ep1.json

# Compare the pure TP6 candidate against the TP3EP2 control
python3 scripts/qualify-ds41-tp-ep-e2e-six.py compare \
  --corpus scripts/fixtures/tp-ep-e2e-corpus-v5-six.jsonl \
  --control runs/tp-ep-six/2rtx6-tp3ep2.json \
  --candidate runs/tp-ep-six/2rtx6-tp6ep1.json \
  --output runs/tp-ep-six/compare-tp6-vs-tp3ep2.json
```

Because the default corpus is v4, `run --arm 2rtx6-tp6ep1` without `--corpus`
fails with "--arm is not a config in <v4 corpus>"; that is the intended
fail-closed behavior, not a missing arm.

### Build and launch

```bash
./build.sh --config examples/configs/tp6ep1-native.config --dry-run
DS41RT_RELEASE_SPARK_TP_ROLES=tp6 ./build.sh --config examples/configs/tp6ep1-native.config
```

The Spark image must advertise the `spark_tp6` role in
`io.ds41rt.v41.spark_tp_roles`; a v8 image without it is rejected before any
service change. `run.sh` derives `tp6` from an explicit `SPARK_TP=6` the same way
it derives `tp2`/`tp3`.

### Measurement confounds that must be controlled before any A/B

1. **Verification-cost model.** `builtin_profile_applies` deliberately returns
   false for every explicit topology, so by default a TP6 arm runs the legacy
   heuristic while a legacy `TP4xEP1` control runs the shipped calibrated table.
   A topology A/B under the defaults compares two different cost models.
   `DS41RT_ADAPTIVE_COST_MODE=legacy|builtin|profile` pins the model explicitly
   (`auto` is the unchanged default, and `builtin` *fails loudly* if the shipped
   TP4 table does not cover the placement). The resolved mode is logged at
   startup as `cost_model=`; every report must record it.
2. **dSpark draft policy/limit** must be identical across arms; the per-arm
   draft limit is a control, not a topology property.
3. **KV class/pool and boundary** must be matched (`g3-paired` N20 for the
   2-RTX arms; the 1-RTX arm is standalone and cannot be compared across
   hardware).
4. **Moa rail address**: the staged peer list uses rhea fabric A `.5` + moa
   fabric B `.12`. That mix follows the documented route analysis
   ([tp-ep-six-readiness.md](tp-ep-six-readiness.md) §4) and remains
   **PROVISIONAL** pending live A↔B reachability; moa fabric A is `.6`. Do not
   describe `.12` as fabric A.

### Runtime role proof (v5 gate)

A startup template's `expert_tp_manifests` block describes what the **artifact**
contains; `spark_artifact_roles` lists the role labels the image advertises.
Neither proves that a running rank loaded its topology's shard family. v5
therefore requires `worker_runtime_role`: one entry per physical rank with the
`role` and `intermediate` its readiness line actually reported, plus a
`source_log`. For TP6 that is role `7`, intermediate `384`, world `6`. An image
that advertises `spark_tp6` in a manifest but never initializes role 7 fails this
gate instead of failing at the first request. The gate is versioned
(`WORKER_RUNTIME_ROLE_SINCE = 5`), so accepted v4 results are not
retro-invalidated.

Weight-only perspective for the record: at 100 GiB, TP3 (96.26 GB), TP4
(80.2 GB padded) and TP6 (48.13 GB) all fit 40 remote layers' weights; TP6 is the
roomiest, not the only one. Workspace, staging and ring headroom remain
unmeasured for all of them.

### What is still required

A qualified TP6 claim needs: the `spark_tp6` role in the image, a live
per-rank memory capture (20 GiB reserve + workspace + ring versus the device
budget), the six-plane reduction numerical check on the built library, the
canonical v5 matrix, a matched cost model and draft policy against the TP3EP2
control, and independent review of the logs and per-host inventory.
