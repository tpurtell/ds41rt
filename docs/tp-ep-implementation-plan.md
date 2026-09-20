# TP×EP replicated Spark expert groups — implementation plan

Status: **plan only, objective revision 4 — full six-node E2E qualification**
(on the revision 3.2 + L2/L4/L5/L6 + candidate + cluster body; frozen
single-level N-plane baseline, live-path scope, sentinel masking). **The full G2
candidate failed (165/372) on a decode graph-instantiation OOM; G3 then passed
the objective (372/372, 0 runtime errors, near-plateau not leak proof); M1
matched timing is complete with quality FAIL in both arms and no TP2 speedup;
M3 native tie-seed replay is complete as a component.** Current consolidated
state: **§3.9–§3.11**. This document contains no measured serving result. Every
budget figure below is either (a) arithmetic derived from the official
checkpoint geometry and the native quantization format, or (b) a runtime value
reported by an owning agent that passes no gate until verified.
CPU subgates are marked `PASS (CPU, reported)` or `PARTIAL`; component builds,
selftests and timing pilots are **not E2E qualification**. **Native component
GPU correctness is CLOSED (135 pass)**, candidate binaries are **built and
staged** (manifest `d63d8b50…`, x86 `9a4bf37b…`, ARM `08cd9a1f…`, superset
TP4/2/3), and the **candidate is LIVE** with the fleet leased to `91570601-…`.
**G1 live ran on 20260920T071305Z-446871 (`N=20`, sparks at layer 20, smoke
PASS); corrected pilots are in §3.6, but memory is not closed and no completion
is declared** — see §3.5–§3.6. **G1 memory closure, kernel
selection/optimization, integrated runtime GPU correctness and every
end-to-end gate remain open**, and the **TP4 baseline has not been run, so no
performance-improvement claim is made.**

Owner: planning/validation agent (this file only). No code, build, service, GPU
or workload action was taken while producing it. Nothing was committed or pushed.

## 1. Objective and scope

**Objective revision 4 (parent-confirmed): full six-node E2E qualification now,
not mere support.** Deliver, behind an opt-in switch that preserves the TP4×EP1
default, a **replicated expert-group** execution mode for the backbone routed
experts of the official `deepseek-ai/DeepSeek-V4.1-Flash` checkpoint
(`MODEL_REVISION=dba1be0a40aa45a94ad051997016db3960a90277`), and qualify the
**four case matrix** end-to-end with matched topology comparisons where the
hardware allows (staged gates in §3.7):

| Config | Hardware | Spark layout | RTX local experts | Status |
| --- | --- | --- | --- | --- |
| **A** | 2 RTX + 4 Spark (connected) | TP2 × EP2 = 4 ranks | TP2 across 2 RTX | **case (1)**, first; existing primary vs TP4×EP1 |
| **B** | 1 RTX + 6 Spark (`rhea`,`moa` ranks 4/5) | TP3 × EP2 = 6 ranks | TP1 | **case (2)**; single-RTX all-40-remote budget model pending measurement |
| **C** | 2 RTX + 6 Spark (`rhea`,`moa` ranks 4/5) | TP2 × EP3 = 6 ranks | TP2 across 2 RTX | **case (3)**; six-Spark E2E pending provisioning/RDMA |
| **D** | 2 RTX + 6 Spark (`rhea`,`moa` ranks 4/5) | TP3 × EP2 = 6 ranks | TP2 across 2 RTX | **case (4)** matched-topology partner of C (same placement/KV/power/dSpark) |

Six-Spark hardware is now **connected** — `rhea` (global rank 4) and `moa`
(rank 5), GB10/SM121, per `docs/cluster-hosts.md`. At inventory time they had
**no docker images, no official checkpoint snapshot and unqualified RDMA
providers**; reachability is **not** an E2E pass, and `rhea`/`moa` may need
devtools/container/model-artifact preparation. Six-Spark end-to-end proceeds
**after the four-node qualification**.

> **HISTORIC restriction (now SUPERSEDED, kept for the record):** an earlier
> allocation note reserved `rhea`/`moa` **only** for full-fleet six-node runs and
> excluded independent builds, standalone microbenchmarks, spare capacity or
> replacement of a four-Spark member. The direct human has since clarified that
> **`rhea`/`moa` may be used for benchmarks/development/anything**, and that the
> original four Sparks are merely *preferred* for actual 4-node deployments. The
> superseding rule is the one stated in the next paragraph.

**Current rule (revision 4 clarification):** `rhea`/`moa` are general-purpose
benchmark/development nodes once prepared; an **independent new-node kernel
lease is allowed as soon as preparation finishes and does not interfere with
G2**. Only a **full six-GPU run** is coordinated and kept exclusive under
`91570601-…` while it is running. Group-major assignment is fixed:
TP3×EP2 = `(ostrich,dodo,emu)` and `(kiwi,rhea,moa)`; TP2×EP3 =
`(ostrich,dodo)`, `(emu,kiwi)`, `(rhea,moa)`.

**Frozen baseline decisions (parent, latest):**

1. The first implementation performs **no group-local (Spark→Spark) reduction**.
   All physical rank planes return directly to the coordinator, which performs
   one **ordered FP32 N-plane sum** (`N = TP·EP ∈ {2,3,4,6}`), adds the shared
   expert **once**, and applies a single final BF16 rounding. A two-level
   TP-then-EP scheme is a later optional alternative only.
2. `group = rank / TP`, `local_rank = rank % TP`.
3. Masking uses a native flag/route **owner word** with a worker sentinel
   `384 + 0` weight for unassigned slots and a fixed top-6 layout that permits
   0..6 owned routes per token; the kernel truly skips active computation for
   masked slots via `rows == 0` metadata while the launch grid keeps its fixed
   upper bound.
4. Only the live `serve-native` path (`v41_native_serve`, `v41_experts`,
   `v41_expert` transport, shared loader, native ABI) is in scope. Legacy
   `commands/real_full` constants are not overhauled unless traced reachable.

Non-goals and hard constraints:

- No default change and no release/default promotion. TP4×EP1 stays the default.
- Official checkpoint only. EXL3 / NVFP4 / any other quant is out of scope for
  these configs, and kernel selection must stay on the official native path.
- No new checkpoint format, no re-quantization, no padding of the checkpoint.
- The two independent target execution lanes must remain independent.
- All implementation and benchmarks are performed by DeepSeek 4.1 Flash Max
  agents under parent review. This planning/validation agent implements nothing.

## 2. Terminology: replicated expert groups (EP) vs disjoint expert sharding

These are different mechanisms and must not be conflated in code, docs or
evidence.

- **Replicated expert group (EP in this plan).** Every group owns a *complete
  copy* of all 384 routed experts (each group internally tensor-parallel
  sharded). A route `(row, expert_id, gate_weight)` is assigned to exactly one
  group; that group's TP ranks each compute their own shard partial, and **every
  physical rank returns its own partial plane directly to the coordinator**.
  There is no group-local reduction: the coordinator sums all `N = TP · EP`
  planes once, adds the shared expert once, and rounds once. EP therefore buys
  compute/latency distribution and wire-shaping at the cost of **replicated
  weight storage**; it does **not** reduce per-rank weight memory.
- **Disjoint expert sharding (classic MoE EP).** Each rank owns a *distinct
  subset* of experts; tokens must be all-to-all dispatched to the owning rank by
  `expert_map` and combined across ranks. SparkInfer already ships this as
  `b12x.moe.ep_moe` (`third_party/sparkinfer/tests/moe/test_ep_moe.py`,
  `test_ep_moe_api.py`). DS41RT's plan here does **not** use that contract;
  it is recorded as a distinct alternative so later readers do not "reuse"
  `ep_moe` and silently change the memory and reduction semantics.

Additional distinct terms:

- **TP degree** (`TP`): intermediate-dimension split within a group
  (TP2 → 1152 per rank; TP3 → 768; TP4 → 576 padded to 640).
- **EP degree** (`EP`): number of replicated groups. `TP × EP =` number of Spark
  ranks. Current production is `TP=4, EP=1`.
- **RTX local group**: the coordinator-owned bottom-up routed layers, TP1 (1 RTX)
  or TP2 (2 RTX). Local layers never span the Spark boundary.
- **Native role vs FFI interface**: the new native roles **5/6** (Spark SM121
  TP3/TP2 geometry) are **not** the same numbering as FFI interfaces **8/9**.
  Keep role ids and FFI interface ids distinct in code, manifests and tests.

## 3. Task ownership (dynamic)

| Agent id | Owns | Mode | Reported state |
| --- | --- | --- | --- |
| `7b81d9c6-6270-40da-a1b9-6333bea6a9af` | `docs/tp-ep-architecture-audit.md` | read-only discovery | **done** (772 lines); superseded on several live-path hardcode claims by current source — see §4 |
| `fdcd9d82-fe6e-494c-9872-b34922fd7e4b` | `docs/tp-ep-kernel-audit.md` (**DONE, 765 lines**); mask/replay tests + benchmark implementation | audit + test/bench implementation | not waiting on its own audit; scaffold review findings are **inputs to kernel selection/optimization (pending), no mandatory rewrite**; awaiting exclusive GPU environment from `91570601-…` |
| `91570601-c759-48c8-b28e-180746e8482c` | `docs/tp-ep-integration-preflight.md`, `runs/tp-ep-preflight`; now **sole service/build coordinator** | read-only + CPU; build/service coordination | user approved full GPUs on `raptor` + Sparks for maintenance and restoration; leases L1 RTX1 / L2 `ostrich` / L4 `dodo`; **original service STOPPED, containers retained, restore pending**; no destructive shared WIP `--recreate` |
| `afd3de8a-e48b-4bd9-a94b-2275d84e7a7e` | `rust/crates/ds41rt-core/src/replicated_expert_schedule.rs`, minimal `core/lib.rs` export, `docs/tp-ep-scheduler.md` | CPU-only implementation | **DONE (CPU, reported): 15 unit + full core; reserve-Vec review fix passes 16 focused regression tests** |
| `885b74a4-ed45-416f-bd05-7ad9f61ad11e` | `rust/crates/ds41rt-loader/src/v41_expert_staging.rs`, unique new loader tests, `docs/tp-ep-loader.md` | CPU-only implementation (native-only generic `BackboneTp {layer, expert, rank, world}` 2/3/4 staging; preserves existing variants; existing safe reads; no catalog edits) | **DONE (CPU, reported): 82 lib + 5 new + 11 integration, including an independent raw real-checkpoint check** |
| `d12965bd-8f8f-4453-b256-6d9b15aaa1ee` | `rust/crates/ds41rt-transport/**`, `docs/tp-ep-transport.md` | implementation | **DONE (CPU, reported): pre-canonical validation before encode + topology flag fail-closed; 180 lib + 14 integration incl. 6-rank loopback; topology set TP2EP1/TP3EP1/TP4EP1/TP2EP2/TP3EP2/TP2EP3** |
| `fa33b227-5aef-4d31-a13e-efb46ceb6da4` | config/launcher scripts, new examples, tests, `docs/tp-ep-configuration.md`; now also `build.sh`, `wip.sh`, phase-0 script, helpers | implementation; **no `ds41rt.config`/default changes; no release/default promotion** | **stage 1 DONE (reported 257 pass / 1 skip)**; approved `2x2\|3x2\|2x3\|4x1`, `SPARK_COUNT=6` needs explicit topology, hosts `SPARK_0..5`; **build/WIP environment only, not yet the native CLI** |
| `f6d0b154-c2a6-465e-9343-f82c1a4e1881` | `native/**`, `rust/crates/ds41rt-ffi/**`, `python/tools/export_b12x_v41_{experts,slices}_aot.py`, `docs/tp-ep-native.md` | implementation; Spark SM121 TP2/TP3 static roles, native packer, **2/3/4/6-plane ordered FP32 reducer** | stage 1 reported 101 FFI + 18 Python; **RTX1 reducer GPU selftest PASS** (`…/reducer_selftest.meta`, exit 0, `route_reduce_sha256 0c12ff6f…`); **TP3 packer PASS synthetic (768, not real checkpoint)**; GPU compile/real weights pending |
| `6ba87b75-6d77-4068-a8a6-8cf9ec278ad8` | `rust/crates/ds41rt-daemon/**`, `docs/tp-ep-runtime.md` | daemon integration | **wiring present** (`new_topology`/scheduler/masks/`reduce_planes`, roles 5/6, `rank % tp`); **compile/tests pending**; runtime doc not yet written |
| planning/validation (this file) | `docs/tp-ep-implementation-plan.md` | read-only inspection + plan | this document |
| remaining | kernel selection/optimization + mask/replay/benchmark GPU run, service build/WIP + restore, integration/E2E | DeepSeek 4.1 Flash Max under parent review | **pending exclusive GPU environment / measurement** |

Status vocabulary: `DONE (CPU, reported)` means the owning agent reports passing
CPU tests; **this planning agent did not execute them** and has not independently
verified the counts. Counts are **historical reported values**: the scheduler
reserve-Vec and transport pre-canonical/fail-closed review fixes now report
passing, while native review fixes (encoder atomic validation, reducer negative
tests) remain pending; counts are kept as reported and are **not overwritten
until a rerun**. These are verified foundational fixes, **not GPU
qualification**. Every GPU, budget, live-service and end-to-end gate remains
`PENDING`. No further discovery subagents are being spawned; this plan is the
durable coordination record.

Current implementation activity: the CPU scheduler module, generic loader
staging, the transport contract, the configuration/launcher/building plumbing,
the native geometry ABI/roles and the daemon runtime wiring. Both audits are done
(architecture 772 lines, kernel 765 lines). Native **component** GPU correctness
is CLOSED (135 pass) and the RTX1 reducer selftest passes; **integrated runtime
GPU correctness** and kernel **selection/optimization** remain pending (no
mandatory rewrite is assumed). The service is stopped pending restore, and
budget closure (G1) plus all live phases still wait on measurement.

Do not edit the audit/preflight documents from this plan; they are owned and
complete. This plan consumes them, corrects their superseded live-path claims
from current source (`§4`), and distinguishes **pending current artifact** from
**reported CPU tests** and from **standalone GPU component checks**.

Recorded design gates from the orchestrating parent that this plan incorporates:

1. TP×EP rank identity and per-group outputs are separate concepts; the
   assembler must not confuse rank planes with group outputs.
2. Six-rank/group reduction is not supported today (reducer is 2-plane or
   4-plane). New reduction support is an explicit deliverable.
3. Scheduler masking must truly skip compute. Dispatching all top-6 routes to
   every group with zero gate weights is **not acceptable** as a performance
   design.
4. All legacy release history (v6/v7/v8 tables, TP4 numbers) is background
   evidence only and must be clearly separated from any fresh TP×EP baseline.

### 3.1 Durable progress log (reported, CPU/deterministic work)

This is the authoritative progress record between planning rounds. Counts are
the owning agents' reports; this planning agent did not run them.

| Workstream | Reported deliverable | Reported result | Environment status |
| --- | --- | --- | --- |
| Scheduler | `replicated_expert_schedule.rs` | 15 unit + full core; reserve-Vec review fix: 16 focused regression tests pass | CPU complete (foundational fix, not GPU qualification) |
| Loader | generic `BackboneTp` world 2/3/4 staging | 82 lib + 5 new + 11 integration, incl. an independent raw real-checkpoint check | CPU complete |
| Transport | rank/group wire + collector | 180 lib + 14 integration, incl. 6-rank loopback; review fix: pre-canonical validation before encode + topology flag fail-closed | CPU complete (foundational fix, not GPU qualification) |
| Config/launcher | opt-in `SPARK_TP`/`SPARK_EP` topology plumbing | stage 1: 257 pass / 1 skip; approved set `2x2\|3x2\|2x3\|4x1`, `SPARK_COUNT=6` needs explicit topology, hosts `SPARK_0..5` | **build/WIP environment only, not yet the native CLI**; a reviewer must run the actual command |
| Native ABI | SM121 TP2/TP3 roles, packer, 2/3/4/6-plane reducer | L4 all six caps + `nm`/manifest PASS (lib sha `d453812c…`); **native COMPONENT GPU correctness CLOSED 135 pass**; **L6 coordinator SM120 lib `1160095c…` + native/reduce/xgrammar selftests PASS**; prod peer-ordering caller contract PASS (`7b81d9c6-…`); M1 pilot warm numbers (six-Spark projected only) | **integrated runtime GPU correctness pending** (not generic "expert correctness"); **warm 62→155 µs unproven: conditional L2 working-set vs CTA-occupancy — neither proven**; conditional **L3 grant already issued**; not E2E |
| Daemon | `daemon/**` integration | wiring present (`new_topology`/scheduler/masks/`reduce_planes`, roles 5/6, `rank % tp`); **835 pass / 6 baseline-proven fails / 101 ignore** (transport's separate 184 pass); review `885b74a4-…` A/B/C PASS, teardown **D conservative live pending** | **candidate binaries already built + staged** (manifest `d63d8b50…` with final ring fix) — **freeze no longer awaits the RAII patch**; `docs/tp-ep-runtime.md` not yet written |
| Kernel | audit + mask/replay tests + benchmark implementation | **audit DONE** (`docs/tp-ep-kernel-audit.md`, 765 lines) | scaffold review findings are **inputs to kernel selection/optimization (pending)**; **no mandatory rewrite assumed**; native **component** correctness CLOSED 135 pass, **integrated runtime correctness pending** |
| Integration | service/build coordination, isolated env | L4 `dodo` completed/released; L2 `ostrich` restarted GPU-debug-ready; L4 evidence → `runs/tp-ep-preflight/l4-dodo`; **original service STOPPED, containers retained** | leases L1/L5 active (`dodo` L5 for native FFN); no destructive shared WIP `--recreate`; **restore pending** |

Corrections applied this round:

- The replicated-group definition now matches the frozen single-level design:
  each physical rank returns its own partial to the coordinator; there is no
  group-local reduction.
- The wire stores a **one-hot owner bitmap at bits 9..11**, exactly bit
  `9 + group` set for group 0..2; `source` identifies the physical rank.
  The implemented field supports at most three replicated groups.
- Native roles 5/6 are recorded as distinct from FFI interfaces 8/9.
- The historical v6 boundary is not treated as a current default; `RTX_GPUS=auto`
  means the dual split is selected at runtime.
- The kernel audit is **done** (`docs/tp-ep-kernel-audit.md`, 765 lines); the
  kernel workstream's only dependency is the exclusive GPU environment, not the
  audit.
- Scheduler/transport review fixes now **report passing** (scheduler reserve-Vec:
  16 focused regression tests; transport pre-canonical validation before encode
  + fail-closed topology flag: 180 lib + 14 integration). Native review fixes
  (encoder atomic validation, reducer negative tests) remain pending. All counts
  stay historical reported values until rerun, and none of this is GPU
  qualification.

L4/L5 + cluster delta (rev 3.2 + delta):

- **L4 `dodo` completed and released:** native TP2/TP3 **all six capacities**
  built with `nm`/manifest PASS, library sha
  `d453812c5c53e916fe8ea0b54d72a51c19b0df8e4db8ce9b7c1ea5b967b1f513`, and
  emitted scratch **matches the memory-review formulas**; SM121 reducer + TP3
  pack selftests PASS. This is **not full-FFN qualification**.
- **L5 `dodo` granted to native `f6d0b154-…`** for native FFN qualification with
  a new helper; **L2 `ostrich` restarted and GPU-debug-ready**.
- Actual min `MemTotal` **130,594,156,544 B (121.6253 GiB)**; conservative budget
  **109,119,320,064 B (101.6253 GiB)** after the 20 GiB reserve. With the
  **~970.9 MiB two-endpoint ring**, **TP2 30 layers fail by ~469 MiB**, **29 fit
  the known model but are unmeasured**, and **31 cannot fit**; TP3 all 40 is
  ~90.4 GiB + rings. **G1 still pending readiness/load peaks.**
- Local L4 evidence is copied to `runs/tp-ep-preflight/l4-dodo`.
- WIP now **explicitly rejects unsupported legacy invocations** instead of
  falsely forwarding environment support.
- **Cluster/hardware (urgent commit `92e29c7`, `docs/cluster-hosts.md`):**
  `rhea` (rank 4) and `moa` (rank 5) are connected GB10/SM121 Sparks; they have
  **no images/checkpoint** and their **RDMA userspace providers are unqualified**
  (link state only), so six-Spark E2E follows four-node qualification. The
  earlier "six Sparks not connected" assumption is **obsolete**, and the
  full-fleet-only reservation is **SUPERSEDED** (they may be used for
  benchmarks/development/anything once prepared; see §1).
- **Build filesystem:** never build on `/mnt/scratch` (read-only NTFS bug);
  builds live at `~/.cache/ds41rt/builds/<task>` on root NVMe, with
  `scripts/assert-build-filesystem.py` as the guard; the raptor replacement
  container is `ds41rt-tpep-nvme-dev` and the fresh daemon target is
  `~/.cache/ds41rt/builds/daemon-tp-ep-target` (old NTFS target has read errors
  and is not reused).

Consequence for gate reporting: CPU subgates that these workstreams cover may be
recorded as `PASS (CPU, reported)` or `PARTIAL`; **G1 budget closure, GPU
correctness and every end-to-end gate remain `PENDING`.**

### 3.2 Actual-state snapshot (rev 3.2, reported unless marked inspected)

- **Memory/hardware:** the user now allows **>100 GB per Spark** with a **20 GB
  Linux reserve**. Measured minimum `MemTotal` is **130,594,156,544 B =
  121.6253 GiB**; minus 20 GiB the conservative budget is **109,119,320,064 B =
  101.6253 GiB**. With the newly viewed **~97/0.9 MiB ring at capacity 4096 ×
  two endpoints**, TP2 at 30 remote layers **fails the 20 GiB-reserve budget by
  ~469 MiB**; **29 layers fit the known model but remain unmeasured**; at a
  20-decimal-GB budget (110,594,156,544 B) 30 layers would leave ~0.915 GiB
  margin (conditional) and 31 cannot fit the declared reserve. TP3 all 40 is
  ~90.4 GiB plus the known rings. The shipped `SPARK_DEVICE_BUDGET_BYTES=100 GiB`
  and every other default are **unchanged**.
- **Build filesystem safety (mandatory):** **never run Cargo or any build from
  `/mnt/scratch`** — the NTFS driver is buggy and the user locked it read-only;
  do not remount, probe writes or use a container alias to build there. Use a
  unique `~/.cache/ds41rt/builds/<task>` path on raptor's root NVMe (ext4,
  `/dev/nvme0n1p2`) and run `scripts/assert-build-filesystem.py PATH...` before
  direct builds; the checker resolves symlinks via `findmnt` and fails closed on
  unknown/read-only filesystems, so a name alone does not identify the backing
  filesystem.
- **TP/EP maintenance relocation (urgent commit `92e29c7`):** preserved
  artifacts/source/serving-binary backups moved from `/mnt/scratch/ds41rt-tpep/`
  to **`/home/tj/.cache/ds41rt/builds/tp-ep/`**; old copies left read-only. The
  replacement raptor container is **`ds41rt-tpep-nvme-dev`** (maps `/scratch` to
  the NVMe path); the old `ds41rt-tpep-dev` (NTFS `/scratch`) **must remain
  stopped**. Fresh daemon target is
  **`/home/tj/.cache/ds41rt/builds/daemon-tp-ep-target`**; the old
  `/mnt/scratch/ds41rt-daemon-tp-ep-target` has filesystem read errors and is
  **not reused**. No shared WIP or serving container is recreated/restarted.
- **Hardware:** `rhea`/`moa` are now connected as Spark ranks 4/5 for the full
  six set only; no images/checkpoint on them yet, `rdma link` is link-state only
  and the userspace provider selection is unqualified — resolve it before
  claiming RDMA readiness. `nvidia-smi memory.total` reports N/A on these
  unified-memory GPUs; use actual RAM plus runtime CUDA measurements, not
  advertised capacity.
- **GPU leases:** L1 native RTX1 (reducer selftest); **L4 `dodo` completed and
  released** — native TP2/TP3 **all six capacities** built, `nm`/manifest PASS,
  library sha `d453812c5c53e916fe8ea0b54d72a51c19b0df8e4db8ce9b7c1ea5b967b1f513`,
  emitted scratch matches the memory-review formulas; SM121 reducer + TP3 pack
  selftests PASS too, but this is **not full-FFN qualification**. **L5 `dodo`
  granted to the native agent `f6d0b154-…`** for native FFN qualification with a
  new helper. L2 `ostrich` container restarted and **GPU debug ready**.
- **L4 evidence:** integration copies it to `runs/tp-ep-preflight/l4-dodo`.
- **Service:** the original service is **STOPPED**, the same containers are
  **retained**; **restore is pending**, and the fleet is now leased to
  `91570601-…` with the **candidate LIVE** and all other GPU work stopped
  (§3.5).
- **Spark staging / dependency:** the broad staging tar excludes docs and
  dependency docs (fixed sync); a **separate verified SparkInfer pin**
  `/scratch/sparkinfer-verified` with hash `eca542dd…` is accepted by the
  verifier (reported on the build host; not present in this checkout).
- **Kernel scaffold:** parent-review findings (wrong TP assembly, unused
  checkpoint, uninitialized outputs, stage signatures, scale axes, padding,
  import path) are **inputs, not a scheduled rewrite**; no mandatory rewrite is
  assumed. Kernel **selection/optimization is pending**. Native **component**
  correctness is CLOSED 135 pass; **integrated runtime GPU correctness is
  pending**.
- **Daemon runtime wiring:** `new_topology`, scheduler, masks and
  `reduce_planes` are present in source; **compile/tests pending**.
- **Build/WIP:** environment only — **not yet the native CLI** — so any review
  must run the actual command rather than trust a wrapper. WIP now **explicitly
  rejects unsupported legacy invocations instead of falsely forwarding env
  support**.
- **Single-RTX placement handoff:** not ready; boot connects before the plan.
  TP3 all-40 remote is a conditional layout that must **retain layer 0 on the
  RTX until tested**; no dangerous handoff is attempted.

### 3.3 Latest completed gates (reported) and current dependencies

Append-only checkpoint; no CPU/GPU work performed by this planning agent.

| Gate | Reported result | Harness |
| --- | --- | --- |
| **L6** coordinator SM120 | official-only lib built (sha `1160095c…`); native/reduce/xgrammar selftests **PASS** | official-only, no GPU claim beyond selftests |
| **L6 peer/copy** | `cuda_peer_selftest` **PASS**; `v41_peer_copy_selftest` pitched split/gather **FAIL 3/3** → **L3 held** for `f6d0b154-…` triage | dual-GPU selftests |
| **L5** DODO real checkpoint | **108 PASS** (first pass); now **native component GPU correctness CLOSED 135 pass (all-rank scope)** — see §3.4 | superseded by §3.4 |
| **L2** synthetic | **9 PASS**: graph fixed runtime capture stream; assembly 4 layouts on **one** Spark (not network) | broad width + real checkpoint ongoing, then conditional timing authorized |
| **CPU daemon** | 833 pass / 6 legacy fails; baseline verification ongoing; **now 835 pass / 6 baseline-proven fails / 101 ignore** (transport's separate 184 pass) | **candidate already built + staged** (manifest `d63d8b50…` with final ring fix); freeze no longer awaits the RAII patch |
| **E2E runner** | 21 CPU tests, tracked paths, strict comparison | **no actual run** |
| Cache/registry | network slow; `915` copy healthy; ROOT cargo registry then offline | provisioning |

Current dependencies:

1. **L3 conditional grant already issued** to `91570601-…`, effective once OST is
   free, the `fa33b227-…` `CUDA_VISIBLE` UUID pin is present and all PIDs are
   idle; the prod peer-ordering caller contract already **PASSES**
   (`7b81d9c6-…`, no fix).
2. **Candidate freeze is no longer blocked:** the candidate binaries are already
   built and staged (manifest `d63d8b50…` with the final ring fix).
3. **Memory G1** has a **conditional grant already issued** to `91570601-…`:
   after OST is released, the `fa33b227-…` UUID pin is set and all PIDs are idle,
   the coordinator auto-derives `N`; if `N < 11` it **stops and records without
   forcing**, and if `N ≥ 11` it brings up **4 workers plus one 8-token memory
   probe** for ring/readiness values. **Not yet launched.**
4. **Fabric (current):** A.1–.6, B.7–.12, raptor `.22`. The **DEFAULT config B is
   stale — do not change it**; the candidate is **explicit A only**, and
   restore must preserve the A-only release arguments.
5. **Six-Spark E2E** remains after four-node qualification and `rhea`/`moa`
   provisioning (images, pinned checkpoint, RDMA providers).

### 3.4 Bounded checkpoint (reported; append-only)

- **L5 native COMPONENT GPU correctness CLOSED — 135 pass (all-rank scope)**:
  TP2 ranks 0/1 and TP3 ranks 0/1/2,
  layers 11/20/39, capacities 1/80/256/4096, M1/80/256, including a **changed
  activation with a new oracle** (max relative `0.0017589`, cos min
  `0.99999845`); pack **byte 0** difference; native compaction ABI2/3 max
  relative `0.0027429`, cos `0.99999619`; **exact inactive zero**; allocation
  delta **0**. This closes the L5 "activation change / all TP shards" gap.
- **Ring-budget / transport CPU closed: 184 pass** (distinct from the daemon
  count below).
- **Daemon:** 835 pass, 6 baseline-proven fails, 101 ignore; review
  `885b74a4-…` A/B/C **PASS** with normal teardown, **D conservative live
  pending**.
- **Prod peer ordering:** caller-contract **PASS** (`7b81d9c6-…`), no fix; the
  earlier `v41_peer_copy` split/gather failure is superseded for the production
  ordering contract.
- **Candidate binaries built and staged — freeze no longer blocked on the RAII
  patch:** manifest `d63d8b50…` **already includes the final ring fix**, staged on
  raptor + 4 Sparks; x86 `9a4bf37b…`, ARM `08cd9a1f…`; native libs unchanged
  (`1160095c…` / `d453812c…`), a **superset of TP4/TP2/TP3**; `ldd` 0 missing;
  all containers respond to `help`.
- **E2E CPU:** 40 tests PASS + 37 config tests (the stale SyntaxError is
  obsolete). Still **no E2E serving run**.
- **Native M1 timing pilot** (E384, capacity 1, width 64, direct 200-launch
  graph): TP4 a6 `156.37 µs` vs TP2 a3 `155.11 µs`; TP3 a3 `62.27 µs` vs TP2 a2
  `62.17 µs`; six-Spark figures are **projected only, not E2E**. Cold 32 MiB,
  5 samples: `225.98` / `229.12 µs`.
- **Warm 62→155 µs: two competing hypotheses, neither proven.** (a) An **L2
  working-set/cache-capacity confound** — **the ~24 MiB GB10 L2 figure is an
  unverified source property** (`f6d0b154-…` could not locate a source): **if**
  L2 ≈ 24 MiB **then** TP2 a2 ≈ 18.8 MB fits, TP2 a3 ≈ 28.2 MB would not, TP3 a3
  ≈ 18.8 MB fits and TP4 a6 ≈ 31.3 MB would not — a conditional statement, not
  a fact. (b) A **CTA-occupancy cliff** (`54` vs `36`) suggested by source
  counts. The resource census (REG 122, block 128, supporting a 4-CTA/SM
  register upper bound) means the simplistic "1 wave → 2 waves" story is **not
  supported**, and the cache hypothesis is likewise **not proof**. The cold 32
  MiB run shows no 62 plateau, so the **real 29-layer stream cold** case matters
  more. The
  read-only resource census by `f6d0b154-…` is pending; **do not claim a wave
  proof**.
- **Leases/fabric:** DODO now **released**; OST finishing the E384 width matrix;
  the **conditional L3 grant is already issued to `91570601-…`** once OST is
  free, the `fa33b227-…` UUID pin is set and all PIDs are idle.
- **Budget candidate:** `109,119,320,064` B (A only); the secondary blank was
  corrected; overlay defaults untouched.

Bounded dependencies: the conditional G1 grant was subsequently exercised — see
§3.5. Six-Spark E2E waits on four-node qualification. No commit was made.

### 3.5 G1 live checkpoint — actual result (20260920T071305Z-446871)

Append-only, bounded; this supersedes the "not yet launched" note in §3.4. All
figures are the reported run output, not a planning claim.

- **Run:** RTX layout auto with **`N = 20`** local expert layers, **Spark first
  layer 20**; all **4 Sparks ready ~24 s**; **coordinator ready 95.7 s**. Raw
  evidence: `runs/tp-ep-preflight/g1-live/raw`.
- **Initial integrated smoke PASS:** SSE probe returned `ready`, finish
  `stop`, usage 2.
- **NOT full G1 memory closure** — the initial read saw only **one** per-rank
  observation (the ring count is **corrected in §3.6**: C2 produced two records
  per rank):
  - lane 0 ring **128 MiB** (actual 8 MiB slots × depth 8) against a
    **1,024,851,968 B** allowance;
  - CUDA free **1–2.5 GB** vs Host Avail **49–50 GB** supported, with
    **Cached 47–50 GB** (a reclaimable meminfo discrepancy);
  - **second lane, large prefill and reconnect are pending.**
- **Modeled 20 remote layers** (not a measured constant): weights
  `72,194,457,600` B, serve `73,520,889,296` B, slack `35,598,430,768` B
  (**33.15 GiB**).
- **C1 counting dSpark:** 3 repeats consistent at **~226–230 tok/s**; the
  content check failed at **length 160 / truncation 54**, diagnosed as a
  **corpus bug, not a quality result**. `afd3de8a-…` is fixing the canonical
  count budget plus a code prompt/check mismatch (3 asserts).
- **Native E384 JIT** partial: the **EP > 1 harness fails** and was
  **CPU-diagnosed** to a last-group mask vs first-oracle mismatch; 7 duplicate
  shadow definitions were removed and 9 CPU tests added — **no GPU rerun**. The
  native **135 all-rank artifact is unaffected**.
- **Launcher:** 329 tests plus UUID-pin `info`/readiness offset gates done; the
  live path was **run manually, not through the automated flow**.
  `fa33b227-…` is checking the real ANSI ready line.
- **Fleet:** the entire fleet is **leased to `91570601-…`**, the candidate is
  **LIVE**, and all other GPU work is stopped.
- **No performance-improvement claim:** the baseline TP4 comparison has **not**
  been run yet.

**G1 remains open:** the run establishes a live 20-layer topology and a smoke
pass, but not memory closure (second lane, large prefill, reconnect) or a
matched TP4 baseline.

### 3.6 G1 corrected pilots, ring records and diag plan (append-only)

Bounded update; **G1 is not closed and no completion is declared.** Plan metadata
below is labelled **PLAN (not observed)** wherever it describes a future arm.

- **Corrected C1 counting (dSpark):** 3/3 PASS, **191 completion tokens**, TPS
  median **226.50** — `runs/tp-ep-preflight/g1-live/pilot-c1-rerun-tp2ep2.json`.
- **C2 (six requests) PASS:** aggregates `248.38 / 342.32 / 347.79`, median
  **342.32** — `pilot-c2-counting-tp2ep2.json`. This is a **dSpark cached
  pilot, not a TP4 speedup**.
- **Uncached prefill probe:** 604-token actual prompt, 1 completion, PASS,
  elapsed **10.28 s** (`prefill-600-sse.txt`); **not a throughput claim**.
- **Ring records corrected (`NOTES-rings.md`):** C2 used all four workers with
  **two ring records**, second `budget_used = budget_peak = 268,435,456` B each.
  `execution_lane=0` is **hardcoded** in `connect_local -> connect_impl(..., 0,
  ...)` for **both** independent `NativeTp4Wave` clients (D129 source) — it is
  **not** a decode/prefill identity, so the earlier **lane-1 requirement is
  retracted**; no wire lane 1 is expected.
- **Capacity nuance:** coordinator CAP is **2048** (workers 4096); ordinary
  two-endpoint max ≈ `512,491,520` B against the `1,024,851,968` B allowance;
  the **response** slot first grows at **819** rows and the **request** slot at
  **1556** rows; the **request-2048 growth test is pending** and the baseline cap
  2048 is kept.
- **Memory still not closed:** needs new post-allocation diagnostic logs; the
  earlier "memory closure" reading is withdrawn.
- **Diagnostics (`6ba87b75-…`, 3 Rust files, CPU check PASS 836/6 known
  baseline fails/101 ignored):** adds INFO bounded-memory snapshots and DEBUG
  `ExpertTiming` for roles 1/5/6; **no kernel changes**.
- **Builds:** both architecture diag-v2 incremental **root-NVMe** builds are
  ongoing against source manifest
  `517c86202a02a9306979e39ed92bb7c12ff5ebbaf4bc26eff395675a0ab03c8e` with new
  version paths; the original candidate is left **idle and preserved**.
- **Frozen full corpus v3** SHA
  `823f203ca1ce054b352a7e97f8acf6102116d5fa89b7226f167b26914778b75047` tests
  **PASS**.
- **PLAN (not observed):** full arms will use **explicit**
  `--kv-pool-size 13090775040 --rtx-expert-layers 20` **BOTH**; architecture
  verified **28,728 groups** with cache `[7,877,077,888, 5,258,201,728]`, exact
  fail-closed, matching the G1-resolved layout. Pilot auto-captured metadata uses
  the older SHA `43c8f…` and is **separate**.
- **Artifacts:** new hashes pending; the **matched baseline has NOT yet run**.
- **Conditional `91570601-…` restart grant:** only after all build `0` / hashes /
  `ldd` / `help` are ready, for a **G2 new-diag pilot and memory only**, then
  review of large prefill / full matrix.

**Status: G1 = initial smoke plus corrected C1/C2 pilots; memory and the matched
TP4 baseline remain open.** Cited evidence: `runs/tp-ep-preflight/g1-live/`
(`NOTES-rings.md`, `NOTES-corrections.md`, `pilot-c1-rerun-tp2ep2.json`,
`pilot-c2-counting-tp2ep2.json`, `prefill-600-sse.txt`, `worker-rings.txt`,
`matched-tp2ep2.plan.json`, `matched-tp4ep1.plan.json`).

**Current checkpoint (append):** `fa33b227-…` launcher/scripts CPU regression —
**338 pass / 1 skip / 119 subtests in 31.78 s**, config **44 pass / 35
subtests**, all shell syntax and `git diff --check` clean, `ds41rt.config`
defaults **unchanged**; log `runs/tp-ep-launcher/cpu-final.log` SHA-256
`e210e5a4eba6bfbd730928ee211f262c6f8f9aa842e183918b72471fe4cc2468`. **No
further plan updates until G2.**

### 3.7 Objective revision 4 — full six-node qualification, staged gates

Bounded scope expansion. The human objective was updated to **qualify all cases
end-to-end with six Sparks connected**, which supersedes the "support only"
reading. Nothing here is a measured result and **`TP2×EP3` being faster is a
hypothesis, not a claim.**

| Stage | Case / action | Concrete gate |
| --- | --- | --- |
| **S1** | (1) 2 RTX + 4 Spark **TP2×EP2** vs existing **TP4×EP1** | Matched placement/KV/power/dSpark corpus; 3 repeats; numerical + request/constraint/cancellation; existing 4-Spark frozen corpus preserved unchanged |
| **S2** | Six-node bring-up on `rhea`/`moa` | Devtools/container/model artifacts prepared on both; **no NTFS builds**; RDMA provider selection qualified. `rhea`/`moa` are **general-purpose benchmark/development nodes** once prepared (the old full-fleet-only restriction is **SUPERSEDED**); an **independent new-node kernel lease is allowed after prep without interfering with G2** |
| **S3** | (2) 1 RTX + 6 Spark **TP3×EP2** | Single-RTX all-40-remote **TP3 budget 97.565 GB model** must be qualified by measurement under the actual `MemTotal − 20 GiB` reserve |
| **S4** | (3) 2 RTX + 6 Spark **TP2×EP3** | Six-node E2E; explicit matrix host count; runner extension is a **separate version** |
| **S5** | (4) 2 RTX + 6 Spark **TP3×EP2** vs (3) **TP2×EP3** | Matched topology comparison with **same placement/KV/power/dSpark**; 3 repeats; no speedup claim before measurement |

Staged constraints and evidence rules:

- **Six-node E2E runner extension is a separate version** with an **explicit
  matrix host-count identity** that **prevents a fake 4-rank result** from being
  recorded as six-node.
- **Preserve the current 4-Spark frozen corpus evidence**; do not overwrite it.
- **Single-RTX all-40-remote TP3** model `97.565 GB` sits under
  `actual MemTotal − 20 GiB` and therefore **requires measured qualification**
  before any claim.
- **Comparison fairness:** on 2 RTX + 6 Spark, TP3 remote-20 is ~**48 GB** of
  weights vs TP2 ~**72 GB**; the two are **not** the same remote-layer count, so
  a TP2/TP3 pairing at 40 remote layers is **unmatched** and must not be
  presented as a like-for-like result.
- **Superseded restriction (kept for the record):** the earlier
  "`rhea`/`moa` only for full-fleet six-node runs" rule no longer applies; they
  may be used for benchmarks/development/anything, and the original four Sparks
  are merely *preferred* for actual 4-node deployments.
- **Scheduling:** a **full six-GPU run** is coordinated and kept **exclusive
  under the fleet owner `91570601-…`** while it runs; independent CPU
  planning/tests, new-host preparation and a prepared new-node kernel lease are
  allowed in parallel; **the existing 4-run (G2) must not be interrupted**.
- **`/mnt/scratch` Cargo/NTFS ban is UNCHANGED** — root-NVMe build paths only.
- **Parallel work now allowed:** `fa33b227-…` inventories the new nodes;
  `f6d0b154-…` does a CPU real-operand loader task and then a possible GPU
  new-node **17-correctness + matched-widths** run — all while the current
  4-node **G2 run stays uninterrupted**.
- Only plan/doc files change; no code, build, service, GPU or workload action is
  taken by this planning agent.

**Plan was frozen** after that clarification; the material G2 failure evidence
below reopens a scoped update only.

### 3.8 G2 full-candidate failure and G3 remediation (actual evidence)

Scoped update from `runs/tp-ep-preflight/g2-live/FAILURE-EVIDENCE.md`,
`NOTES-memory-attribution.md` and `g2-full-candidate-summary.json`. History in
§3.5–§3.7 is preserved; this section does not overwrite it.

**Failure shape.** Arm `tp2ep2`, corpus SHA `823f203c…`, runner SHA
`e197146a…`: `passed=false`, **165 failures = 163 incomplete-SSE + 2 repeat
drift (code, topic)** over **372 attempted**. Coverage: counting 93/93, code
93/93, topic 23/93, mixed 0/93. Mixed and most topic requests end **before the
first content token** (`done=False`, `finish`/`usage` absent; one initial role
event, no content events).

**First error is a decode OOM, not admission:** at `07:35:23.816Z`,
`ds41rt_cuda_graph_end_capture` → `cudaGraphInstantiate failed: out of memory`
(status 6); the first admission failure follows at `07:35:38.925Z` (56 of 100
admission failures are OOM, window to `07:37:21.102Z`). Spark side shows only
`native RoCE peer removed` WARNs (24 on ostrich) and **0 explicit worker
errors**. GPU1 `nvidia-smi` showed **`97,244 MiB` used and `4 MiB` free** at
capture (the reported `97,887` total does not sum with those two because
~`639 MiB` is reserved; use the `used`/`free` pair and do not present the
fraction as arithmetic).

**Root cause is open.** The `6ba87b75-…` source review finds the reachable graph
caches **bounded but large** (`LayerGraphs ≤ 65/layer/bank`; dual tails
64/layer/2 lanes) with some instantiate-before-evict +1 paths; there is **no
existing retention env**. **Failed instantiate destroys the temporary graph; no
failed-capture leak was found in the inspected path.** The only identified gap is
the **reserve aggregate cache vs 800 MiB**. **No leak is proven or ruled out
globally**; a plateau requires measurement.
`7b81d9c6-…` owns the RTX headroom-control math.

**Ring correction (kept from the notes).** `402,653,184 B = 3 × 134,217,728 B`
is **three small (8 MiB-slot) endpoint reservations transiently live from
old/new sessions of the same coordinator PID**, not large-frame growth:
`request_capacity`/`response_capacity` stay `8,388,608`, `depth=8`. **No
large-frame growth is proven**, and the **actual prefill wire-M issue is
unresolved**. Only the ring budget `268,435,456 B` is **exactly attributable**;
the `cuda_free` delta of `308,891,648 B` is **not** proof of per-connection cost.

**Arbitrated process state.** Coordinator gracefully stopped (worker PIDs
retained: ostrich `12802`, dodo `8801`, emu `21008`, kiwi `8909`, all
`--world 4 --spark-tp 2 --spark-ep 2 --first-layer 20`, diag-v2). RTX after stop:
GPU0 `2 MiB`, GPU1 `12 MiB` used. No further GPU queries were made.

**G3 remediation (AUTHORIZED).** `N=20`, **same binaries and worker PIDs**, with
exact KV **5,037,542,400**, cache `[3,045,138,304, 2,036,908,672]`, free
4.5/3 GiB (vs G2 KV `13,090,775,040`). **INFO-only** initial reconnect plus
bounded all-case probes; **full 372 only if healthy AND the EMU image export is
finished**; **stop at the first OOM**. Both future TP4/TP2 arms must use the same
G3 KV, and there must be **no G2 comparison**. `N=19` is **not authorized** as an
alternative; if needed, 3.362 GiB/card with Spark 21 layers (75.8 GB) fits.

**Control-scope capacity note.** The control scope is **capacity**, not a
`16 × 1M` device promise; no plan statement should be read as guaranteeing
sixteen 1M-token device-resident requests.

**Other current state.** New hosts' source/ARM/harness are verified; `fa33b227-…`
image export is granted with **CDI import on MOA first** (rhea MGMT cache copy
running ~2h42; moa cache exists). The kernel GPU lease is pending container
static checks. The six-node wrapper is being corrected by `afd3de8a-…` for
memory arithmetic and compare gates — **not accepted yet, so do not freeze the
SHA now**.

**Evidence root:** `runs/tp-ep-preflight/g2-live/` (`FAILURE-EVIDENCE.md`,
`NOTES-memory-attribution.md`, `g2-full-candidate-summary.json`,
`startup-tp2ep2.observed.json`, ring before/after files).

**Objective conformance:** this is now a **full six-node qualification**
objective, not merely replicated-group support.

### 3.9 Current status after G3 + MOA correctness + six-node prep (append-only)

Old failures in §3.5–§3.8 are kept. No runtime or plumbing edit was made by this
planning agent.

- **G3 full matrix (canonical 372) — objective PASS.** `passed=false` with
  **3 failures, all `greedy output changed across repeats`** (code-merge-intervals,
  topic-virtual-memory, mixed-code-fix); **372/372 rows passed, 0 runtime/SSE
  errors, 0 `cudaGraphInstantiate` failures**. Frozen identity: corpus
  `823f203c…`, runner `e197146a…`, metadata `2898eb23…`, x86 daemon `82c96f24…`,
  arm64 `5162bc8a…`.
- **Near-plateau, not a leak proof.** Final RTX free `5339`/`3038 MiB` (GPU0/1);
  burst then a **202 s near-plateau** (−0.22/−0.25 MiB/s; 15/33 and 14/33 flat
  10 s steps). `used+free` is ~636/639 MiB below nominal (driver-reserved).
  Reconnect credit behaved: released endpoint re-reserved
  (`134,217,728 → 268,435,456 B`) while the historical peak `402,653,184 B` was
  retained. **No per-graph attribution** and no universal reserve claim.
- **TP4 matched control GRANTED** to `91570601-…`: same `N=20` / KV
  `5,037,542,400`; reload pending. **No speedup claim.**
- **New hosts (`rhea`/`moa`).** User-loaded caches are complete on both (48
  shards, `refs/main=dba1be0a…`, `repo_bytes=510,380,950,382`); Fabric A/B MTU
  **9000** verified as link/MTU metadata only (not RDMA/GID/E2E). The completed
  rsync moved **0 files** (destination already in sync) — **no 510 GB transfer**.
  **RHEA image is absent** (0 images/0 containers); rhea image import comes from
  moa after the fdcd lease.
- **MOA correctness.** Compat **SM 12.1 / 48 SMs**, **L2 measured `25,165,824 B`
  (24 MiB)**, 17 GPU tests reported pass. Real smoke: the unloaded-route fixture
  bug is fixed and the real **unmasked** arm is nonzero at **~0.0017 relative
  L2**; the **masked** arm fails with a CPU/CUDA **device mismatch**
  (`torch.where` cpu vs cuda:0) — **repeats held pending that fix, no timing**.
- **Tie-seed (CPU).** 10 tests: **61/64 owners vary under TP2 vs 0 under TP4**;
  native is pending (see the method report).
- **Active leases:** `91570601-…` original 4 + RTX matched control; `f6d0b154-…`
  MOA correctness; `fa33b227-…` metadata + RHEA image after MOA release.

**Remaining gates made explicit.**

1. **Large-frame actual-M test** — exercise the real prefill wire-M frame size
   (capacity fields `> 8,388,608`), which no run has yet done; the observed
   `402,653,184 B` peak is still only three 8 MiB-slot endpoints.
2. **All-4-credit test** — prove the endpoint/ring credit across **all four**
   Spark ranks (only rank 0 has evidence today; ranks 1–3 logs are empty).
3. **Six-node runtime** — after the rhea image import + CDI-create + static
   checks, run the six-node matrix with the separate-version host-count identity
   (G18/G19/G20).
4. Still open from prior sections: G1 memory closure, G3-vs-TP4 matched
   comparison without a G2 comparison, kernel selection/optimization, and the
   objective-revision-4 cases (2)–(4).

**Evidence:** `runs/tp-ep-preflight/g2-live/g3-FULL-SUMMARY.md`,
`runs/tp-ep-preflight/g3-memory-review.md`, `runs/tp-ep-preflight/six-node-prep.md`,
`runs/tp-ep-kernel/moa-correctness/MOA-STATUS.json` and `moa-real-smoke{2,3}.log`.

### 3.10 Milestone execution board (concise ownership)

Consolidated board replacing step-by-step coordination. Agents own routine fixes
and end-to-end work **within their package**; the parent reviews consolidated
milestones and scope/architecture changes only.

| Milestone | Owner | Scope | Gate / dependency |
| --- | --- | --- | --- |
| **M1** four-Spark matched comparison | `91570601-…` | TP4×EP1 vs TP2×EP2 at `N=20` / KV `5,037,542,400`; then reconnect/ring checks | **comparison corrected: 372 completed both; G3 372 content, TP4 371 (truncated code 512, not an established semantic error); C2+ candidate excess unadjudicated**; no TP2 speedup; memory unblocked; large-frame growth not a required gate, direct-RC (`d12965bd-…`) awaits the `915` quiet window |
| **M2** kernel correctness/provenance + timing | `fdcd9d82-…` | official-only kernel correctness, provenance, matched-width timing | **milestone COMPLETE: 96/96 accepted, 0 rejected, lease released** (`tp-ep-kernel-timing-milestone.md`; independent audit ACCEPT). Four-Spark TP2EP2 faster only in w128 M1/M8/M16; six-Spark M8/M16 parity within 1%; **width-192/M64 NOT measured, no AOT recommendation**; native **135 correctness COMPLETE** |
| **M3** MOA native replay | `f6d0b154-…` | native replay on MOA, then release | **COMPLETE (component); independent APPROVE `seed-mode-review.md`; width-map plumbing APPROVE (`width-map-review.md`)** |
| **M4** RHEA image / CDI / RDMA prep | `fa33b227-…` | RHEA image import, CDI container create, RDMA prep | **both new hosts runtime READY; six-node-prep fully consolidated**; single-RTX retry authorized, cause unresolved (§3.15) |
| **M5** six-node runtime protocol/evidence | `afd3de8a-…` / `91570601-…` | six-node runtime protocol + evidence, execution under `915` | **launcher APPROVE by f6: 28 tests, full 387/1 skip; dual six-node arms EXECUTED** (TP3EP2 + TP2EP3: counting exact, 3 greedy-drift failures each, no TP2 win); **single-RTX all-40 `1rtx6-tp3ep2` retry authorized after two pre-allocation gate rejections** |
| **M6** seed implementation | `7b81d9c6-…` / `91570601-…` | opt-in vs layer-default dispatch; `915` new-coordinator-ONLY build (root NVMe) + same binary, C1 triplicates on four cases after the frozen profile | **DONE for C1**: `layer` exact in all four cases vs `dispatch` drift; **not a C2+ fix** (3 greedy failures remain). x86 build `6abe8ad2…`; core 178/0/1, focused 10, full 838/6/101 |

Rules: **GPU leases are serialized**; **no default/release change**; **root-NVMe
builds only** (`/mnt/scratch` NTFS ban unchanged).

**MOA note (authoritative: updated `MOA-STATUS.json` + `fdcd9d82-…` consolidated
provenance).** The earlier "smoke4 / 4 pass" shorthand was ambiguous. The
**passing jobs are `bash-96` and `bash-99`, each `4 passed / 0 failed`** (26.16 s
and 63.78 s), against source `test_tp_ep_real_smoke.py` sha
`98634c712787e164928769bf7050217d1dc24e8850f4e1d516e84ca2994bd47d` (logs
`moa-real-smoke5` plus final confirm). It is **not** one passing case inside a
3-failed test. The intermediate masked-arm failures were **test-harness
device-consistency bugs, not kernel/numeric defects** (route table on CPU vs
weights/outputs on CUDA). Final MOA state: **PASS, lease released, GPU idle**,
SM 12.1 / 48 SMs, L2 `25,165,824 B`.
- Unmasked rel-L2 ≈ `0.0016072 / 0.0016287 / 0.0016622` (cosine ≈ `0.9999986`).
- Masked rel-L2 ≈ `0.0017096 / 0.0016179 / 0.0016709`; all-unowned arm is an
  **exact zero**.
- The mask is a **fixture unit mask** (`expert % tp_degree`), **not** the
  production runtime group mapping and not a scheduler result.
- **Benchmark hash discrepancy:** the frozen
  `benchmark_v41_ep_groups.py 7cc83b5a…` vs the agent-reported `2075cbf…`
  (earlier written `2075f…`); `fdcd9d82-…` is resolving provenance and the
  harness is **not assumed frozen** until settled. The frozen benchmark itself was
  untouched; the smoke wrapper is run-local and never part of the frozen harness.

**Plan frozen until milestone results; no further incremental status edits.**

### 3.11 Consolidated milestone update (accepted facts, single edit)

Historical append notes above remain; this supersedes the intermediate drafts and
records only accepted current facts.

- **M3 — native tie-seed replay: COMPLETE (component, not E2E root cause).**
  Audited **20 raw planes, all hashes match** (`ALL_OK: true`, 10,240 B each;
  little-endian BF16, `-0.0` not conflated). `runs/tp-ep-native/tie-seed-replay/gpu/results.md`:
  TP2×EP2 cross-id final **rel-L2 `0.0031246`**, cosine `0.9999960`, owner ids
  differ 61/64, every rank plane differs; **TP4×EP1 cross-id is exact** (`rel_L2
  0.0`, 0/64 owner change) as the tie-seed-insensitive control. Same-id is
  bit-identical in both. **Not claimed:** logits/greedy flip, or the exact G3 C1
  code-drift root cause.
- **M1 — matched timing: COMPLETE; quality FAIL both arms.**
  `runs/tp-ep-preflight/g2-live/COMPARISON-g3-vs-tp4.md`, same frozen controls
  (corpus `823f203c…`, runner `e197146a…`, N20, KV `5,037,542,400`, diag-v2,
  4 Sparks). Counting `191` tokens clean: **TP4 equal-to-3% faster**, i.e. **no
  TP2 speedup**. G3 failures = 3, TP4 = 4 (both arms drift on code/topic/mixed;
  TP4 additionally one code C8 content-check failure). **C1 G3 non-counting
  drift where TP4 was exact** (targeted repeat, not a qualification claim);
  C2+ drift in both. Both arms 0 runtime/SSE errors, 0 graph OOM.
- **Large-frame / all-4-credit:** **large-frame growth is not a required gate**;
  the ring/credit behavior is addressed by the **`d12965bd-…` test-only
  direct-RC plan**, which **awaits the `915` quiet window** (no GPU compute).
- **M4 — RHEA image now streaming; `f6d0b154-…` released.** `fdcd9d82-…`
  full real-operand timing package **approved after the image transfer ends**,
  including **TP2×EP3 vs TP3×EP2**; readiness docs updated and the **`2075cbf…`
  hash ledger resolved**.
- **M6 (new) — seed diagnostic:** `7b81d9c6-…` opt-in
  dispatch-vs-layer default dispatch, **source/tests only**, root-NVMe CPU window
  granted, **no deploy**.
- **Quality supplemental is a PROPOSAL** from `f6d0b154-…` that preserves the
  frozen gates; **not accepted**.
- **Performance brief (`6ba87b75-…`) is under revision and NOT accepted** — its
  resident-weight and row-split claims are incorrect.
- **Six-node readiness:** `afd3de8a-…` fixes the executable candidate helper and
  minimum budget **`109,119,320,064`**; `d12965bd-…` network review pending;
  **MOA `.6`/`.12` are provisional**.

No obsolete blocking-cache note remains; the board above is the current state.

### 3.12 Consolidated accepted milestones (single update; execution priority)

- **Seed implementation accepted.** `7b81d9c6-…` core **178/0/1-ignore**, focused
  **10 pass**, full **838 / 6 known-baseline / 101 ignored** (exit 101);
  independent `f6d0b154-…` **APPROVE** (`seed-mode-review.md`). Parent authorized
  `91570601-…` a **new coordinator ONLY** build (root NVMe, same binary) for
  **dispatch/layer C1 triplicates on four cases after the frozen profile** —
  **no default/release/native/worker change**.
- **M1 comparison corrected.** Both arms completed **372**; **G3 372 content,
  TP4 371** (the truncated code-512 row is **not** an established semantic
  error); **C2+ candidate excess is unadjudicated**; **no TP2 speedup**.
- **Supplemental quality adopted as a separate report** with **strict FAIL
  unchanged** (docs quality-adjudication, 171 lines); **no five-run study on the
  critical path**.
- **Performance v2 corrected and accepted**; **profile execution authorized** —
  counting C1/C16 ×3 **after transport**; `6ba87b75-…` analysis owns **no extra
  GPU**; **no TP2 speedup**.
- **Network:** DF8972 **bidirectional PASS** on `.22 ↔ .5/.11/.6/.12`; NIC
  **9000** vs verbs **4096** verified; **IP is not RC proof**;
  **`HOST_DEVICE_MAP` required**.
- **Launcher:** the fragile Python path was **retracted/deleted**; the replacement
  reuses the existing candidate plan/start with an **ordered 7-process ACK mock
  (371 pass / 1 skip)**; last budget/single-mock pending for `afd3de8a-…`.
- **Memory:** `885` six-node **static review now** (owner `fdcd9d82-…`).
- **Runtime — single-RTX retry authorized, cause unresolved (not goal-blocked):**
  the single-RTX all-40 attempt **failed twice at the pre-allocation CUDA-free-only
  gate** (latest ostrich `cuda_free=75,634,618,368` vs serve need
  `97,565,310,416`, `MemAvailable` ~125 GB, no active CUDA holders). Allocation was
  **never attempted**; do **not** treat the actual allocation as infeasible or the
  page-cache cause as proven. The **installed helper is a system-wide
  `sync + drop_caches=1`** (cleans page cache globally), **not** a targeted
  `POSIX_FADV`; the user authorizes it **on all 7 hosts whenever needed**. Do not
  relax admission. Workers stopped, RTX idle at this milestone.

**Priority is execution, not more proposals.** Plan frozen after this update; no
ongoing transient edits.

### 3.13 HISTORICAL snapshot (superseded by §3.14; kept for the record)

Pre-§3.14 snapshot. In particular, the six-launcher **REJECT** recorded below is
**superseded by the f6 APPROVE** in §3.14; the board there is authoritative.

- **New hosts:** both runtime **READY**; `fa33b227-…` six-node-prep fully
  consolidated with **no import/cache block**.
- **Four-Spark memory unblocked:** targeted checkpoint-blob
  `POSIX_FADV_DONTNEED` (**no global drop**), free **108–120 GB**. *(Historical
  wording, superseded: the installed helper is a **system-wide
  `sync + drop_caches=1`**, not per-blob FADV; the user authorizes it on all 7
  hosts whenever needed.)*
- **Transport HTTP probe COMPLETE** at all **M128** (actual **1032 / 2008**
  tokens) with **no 819+ growth proof**; `d12965bd-…` has a **test-only
  direct-RC plan + CPU 1 pass**, **live NOT RUN** pending the window.
- **Profile C1/C16 ×3 collected for both arms** (`g2-live/profile/tp2`,
  `tp4`); `6ba87b75-…` analysis running — **not a new performance claim**;
  DEBUG shows C16 spread.
- **Seed:** independent review **accepted**; new-coordinator build + C1
  experiment **in progress** under `91570601-…`.
- **Six-node launcher:** independent review **REJECT** (confirms **GID
  preconnect deadlock**, **missing timing-forward**, **numeric guard**);
  `afd3de8a-…` fixing, **no six-node start yet**; conditional six execution
  granted to `915` after review + MOA lease.
- **MOA kernel checkpoint:** 17 same-window pass; initial 8/96 partial
  reporting bug; M8 **restore unpadded→padded fixed** + **22 CPU tests**
  verification pass; new harness **`452d6eb6…`**; batch 115 cancelled; **full 96
  restarting (~1.5–2 h)**; **RC1 arms not accepted**; essential current
  **width-192 M8/M16 baseline + 32 extra authorized if needed**, **no production
  default change**.
- **Native 135 correctness COMPLETE, unchanged.**
- **Kernel integration (`7b81d9c6-…` CPU doc):** same **JIT class, not AOT
  exact**; scope only **changed role / cap 80**, **not** an all-cap/RTX retune.
- **Memory review `885` final:** fits all 3 configurations, **slack 10.76 GiB
  at all-40**, with allocation unknowns explicit.

**Plan state — explicit.**

- **COMPLETE / accepted:** native component correctness (135); M3 native
  tie-seed replay (component); M1 matched **timing** (quality FAIL both, no TP2
  speedup); seed implementation + independent review; MOA new-host runtime
  readiness; memory 885 static fit; four-Spark memory unblock; transport HTTP
  probe at M128; profile C1/C16 ×3 raw collection.
- **PENDING / not accepted:** objective is **NOT complete**. Still open —
  live direct-RC (d129, awaiting the `915` quiet window), six-node
  runtime/selection/regression/restoration (G18–G20/G25), MOA full-96/RC1
  acceptance (M2), and performance v2 execution after transport (no TP2 speedup
  yet). **G15 (default/release) is OUT OF SCOPE by user direction; G23
  (forcing actual-M 819+ growth) is not a required gate** — direct-RC covers the
  ring behavior separately.

**No new proposals. Plan frozen.**

### 3.14 Final board refresh — accepted milestones and scope corrections

One refresh of the existing current board; history above is not rewritten.

- **Six-node launcher: APPROVE.** `f6d0b154-…` review at
  `runs/tp-ep-six/launcher-review.md`, **28 independent tests, full 387 / 1
  skip**. Conditional six execution stays with `91570601-…`.
- **Profile interpretation: COMPLETE, accepted** (160 lines,
  `runs/tp-ep-preflight/g2-live/profile/PROFILE-INTERPRETATION.md`; parser 13
  tests; corrected math and unverified joins). Kernels/dSpark are **materially
  equal in aggregate — no causal TP2 win**; the **timestamp-less transport is not
  windowable**.
- **Seed:** x86 build **COMPLETE, exit 0**, binary
  `6abe8ad2f41becdc0154aa1b799e136d6f022145fb717df7c9354cbea71acd23`; the **C1
  experiment under `915` is in progress** (launch record
  `seeddispatch-20260920T082837Z`), **results not yet available**.
- **RC test: f6 APPROVE** bounded execution after the doc/IP fix; `d12965bd-…`
  awaits the `915` quiet window; uses **CUDA pinned host buffers, no GPU
  compute**; **two first, then four**; **not a prerequisite for six-model E2E**.
- **Kernel:** **96-invocation run healthy; 10/96 completed at last checkpoint**
  (harness `452d6eb6…`);
  **release MOA after 96**; width-192 baseline + 32 later lease pending; cap-80
  optional; focused **M64 real-C16** and reuse fixture pending `6ba` CPU analysis.
  Integration doc final: `docs/tp-ep-kernel-selection-integration.md`.
- **Memory review:** reclaim is the **system-wide `sync + drop_caches=1` helper**
  (installed; user-authorized on all 7 whenever needed), **not** a targeted
  `POSIX_FADV`; the historical **108–120 GB free is not a guarantee**; **live
  six-node is pending**.
- **Change-scope audit:** `runs/tp-ep-preflight/change-scope-audit.md` current;
  last stale sentence fixed; **default unchanged**; **135 inactive-zero/replay
  component closed**.
- **Wire correction (source-verified):** bits 9..11 are a **one-hot owner bitmap**.
  Earlier binary-index wording was erroneous; source/build provenance confirms
  the measured coordinator and workers used this same one-hot encoding.
- **Completion-gate scope:** the user explicitly excludes default
  serving/release, so **G15 is OUT OF SCOPE / not required** rather than a
  pending completion item. **G23 (actual-M 819+ E2E growth) is not required as a
  completion gate** — it must not be provoked by altering prefill solely to test
  it; **direct-RC proof addresses ring behavior separately**.

**Preserved:** every strict quality **FAIL** remains recorded, and the remaining
open work is **all three six-node runtime/selection/regression/restoration**
items. The objective is **not complete**.

**Plan frozen.**

### 3.15 Consolidated remaining milestones and execution order

One consolidation of the newly available milestone evidence; no new broad scope.

**Accepted now.**

- **Dual six-node E2E executed** (`SIX-CONSOLIDATED.md`, both arms): **2rtx6-tp3ep2**
  and **2rtx6-tp2ep3** each `passed=False` with **3 failures** — counting exact,
  greedy drift on code/topic/mixed. Counting C1/C16 TPS: TP3EP2 `245.7/1410.7`
  and TP2EP3 `241.5/1407.4` — **counting parity, no TP2 win**. Seed coordinator
  `6abe8ad2…`, ARM workers `5162bc8a…`/lib `d453812c…`, N20, KV 5037542400, world
  6, corpus v4 `99034171…`. Memory values are **stage samples**, not serve
  headroom. RC GID evidence recorded.
- **Single-RTX arm `1rtx6-tp3ep2` — retry authorized, root cause unresolved.**
  It **failed twice at the pre-allocation CUDA-free-only gate**, not in
  allocation: the latest run reports ostrich `cuda_free=75,634,618,368` against a
  serve need of `97,565,310,416`, while **`MemAvailable` is ~125 GB** and there are
  **no active CUDA holders**. **Allocation was never attempted.** Do **not**
  classify the actual allocation as infeasible and do **not** treat the page-cache
  cause as proven — possible over-conservative gate, not established. The old
  RHEA permission failure is **historical**. Retry authorized under `915`.
- **Seed `dispatch` vs `layer`** (`SEED-dispatch-vs-layer.md`): C1 four cases ×3 —
  **layer mode is exact in all four** where dispatch drifts; **layer mode is a C1
  directional control, not a C2+ fix** (3 greedy failures remain at C2/C16).
  Dispatch reproduces the frozen `82c96` behavior. The **installed cache helper is
  a system-wide `sync + drop_caches=1`** (cleans page cache globally), **not** a
  targeted `POSIX_FADV`; the user authorizes it **on all 7 hosts whenever needed**,
  so there is no "required FADV before each launch" step. Only the coordinator uses
  the x86 seed binary.
- **Kernel timing milestone complete** (`tp-ep-kernel-timing-milestone.md`):
  **96/96 accepted, 0 rejected; lease released**. Four-Spark (TP4EP1 vs TP2EP2)
  and six-Spark (TP3EP2 vs TP2EP3) are **separate decisions**: TP2EP2 is
  materially faster only in **w128 M1/M8/M16** (1.122/1.100/1.117) and at parity
  elsewhere; the six-Spark M8/M16 ratios are **within 1 %** (w64 M1 favours TP2EP3
  ~13 %). **Width 192 (production comparator) and M64 are NOT measured**; the
  synthetic route table has **no reuse**, so this is a **stress comparison, not a
  production predictor**; **no AOT recommendation**. Independent audit **ACCEPT**;
  the **gate bookkeeping is FIXED** — **42 CPU tests, 96 accepted unchanged** per
  the latest consolidation + archive manifest. The **attestation gap remains**
  (arm JSONs carry no harness hash; archived `c741d34a…` is labelled historical).
- **Width-map plumbing APPROVE** (`width-map-review.md`): re-export invalidation
  and exporter CLI validation **executed**; **real AOT object content and timing
  unproved**.

**Still to execute (completion gates).**

| # | Remaining item | Status |
| --- | --- | --- |
| R-a | **Single-RTX all-40 E2E** (`1rtx6-tp3ep2`) under `915` | **RETRY AUTHORIZED, root cause unresolved** — failed twice at the **pre-allocation CUDA-free-only gate** (ostrich free `75,634,618,368` vs serve `97,565,310,416`, `MemAvailable` ~125 GB, no CUDA holders); allocation never attempted; not proven infeasible or cache-caused |
| R-b | **Width-192 M8/M16 comparator** + **M64 reuse against 192** on `scripts/fixtures/tp-ep-reuse-m64-e384.json` | **NOT done** — needs a new lease |
| R-c | **Sampled uncached prefill comparisons** (decode TTFT is not prefill) | **NOT done** |
| R-d | **Strict repeatability FAIL** (C2+ greedy drift; layer mode fixes C1 only) | **UNRESOLVED** |
| R-e | **Direct-RC test** | **PENDING, optional**; awaits `915` quiet window |
| R-f | **Final regressions / restoration** | **PENDING** |

**Execution/dependency order:** **system-wide `sync + drop_caches=1` helper**
(installed; user-authorized on all 7 whenever needed) → single-RTX all-40 E2E
under `915` → new lease for width-192 M8/M16 comparator + M64 reuse-vs-192 →
sampled uncached prefill comparisons → direct-RC (optional, quiet window) → final
regressions/restoration. **No Cargo, no `/mnt/scratch`, no defaults/release.**
Goals remain within the active 512 limit.

**Final status: objective NOT complete; R-a–R-f above are the remaining work.**

## 4. Current baseline — read-only facts (no measurement)

Verified by inspection in this planning pass. **Scope of change:** only the live
`ds41rt serve-native` path (`rust/crates/ds41rt-daemon/src/v41_native_serve/`,
`.../v41_experts/`, `v41_backbone_execution/`, `ds41rt-transport/src/v41_expert*`,
the shared loader and the native ABI) is in scope. The `commands/real_full/*`
tree is a different command; its constants are **legacy** and must not be
"overhauled" by this work. A legacy hardcode is only listed as a blocker if it
is first traced reachable from the live path; a broad unrelated TP refactor is
explicitly out of scope.

| Fact | Reachability | Evidence |
| --- | --- | --- |
| Default topology 2 RTX + 4 Spark, TP4×EP1 | live/config | `ds41rt.config`, `architecture.md`, `docs/ENGINEERING.md` |
| `RTX_GPUS=auto` in the shipped config (dual is *selected*, not guaranteed) | live/config | `ds41rt.config:26` |
| `SPARK_COUNT` default 4; launcher accepts `0,2,4,6`; `SPARK_COUNT=6` requires explicit `SPARK_TP`/`SPARK_EP` | current launcher | `scripts/release-common.sh:149,258,268-269` |
| Opt-in `SPARK_TP`/`SPARK_EP`, approved `2x2\|3x2\|2x3\|4x1`; hosts `SPARK_0..5` | current launcher/config | `scripts/release-common.sh:104,152-153,430-456` |
| Live coordinator accepts **world 2/3/4/6**; reducer entry fixed 2/4 plus generic 3/6 | **current — supersedes the rev-2 "2 or 4 only" reading** | `v41_experts/coordinator.rs:256,530` |
| Live worker resolves `V41SparkTopology`, maps `rank % tp`, world = tp, roles 5/6 | **current — supersedes the rev-2 `BackboneTp2 only` reading** | `v41_experts/service.rs:13-27,278-300`; `v41_spark_topology.rs:18-68` |
| Transport topology set `TP2EP1, TP3EP1, TP4EP1, TP2EP2, TP3EP2, TP2EP3`; RoCE rank counts 2/3/4/6; canonical executor ids per layout | **current — supersedes the fixed-4 collector reading for the new path** | `ds41rt-transport/src/v41_expert/native_group.rs:63-67,170-178`; `v41_expert/roce.rs:50-96` |
| Live serve constructs the topology-bound transport | **current** | `v41_native_serve.rs:189-195` |
| Owner wire: 12-byte route, bits 0..8 expert id, bits 9..11 **one-hot owner bitmap**, exactly bit `9 + group`; transport `source` identifies rank | **current** | `native_group.rs:184-228`; `docs/tp-ep-final-integration-review.md` |
| dSpark cost labels the topology (`spark_tp2ep2`, …) instead of assuming `spark_tp4` | **current — supersedes the rev-2 cost reading** | `v41_native_serve/speculative/cost.rs:10-23,155-164` |
| Reducer ABI now `planes[6]` with `ranks ∈ {2,3,4,6}`; legacy 2/4 entry points preserved and test-equivalent | **current — supersedes the rev-2 "no 3/6 path" reading** | `native/include/ds41rt_v41_experts.h`; `native/cuda/kernels/v41_route_reduce.cu:87-135` |
| `EXPERT_HOSTS [;4]` / `DS4_EXPERT_TP_WORLD_SIZE=4` and `expert_format.rs` shard math | shared/legacy generic helper; the new path uses loader `BackboneTp {world}` staging instead, so this is not a live blocker | `constants.rs:6-7`; `expert_format.rs:294-306` |
| Per-expert native FP4 staging is 18,800,640 bytes | live loader | `rust/crates/ds41rt-loader/src/v41_catalog.rs:1100,1118` |
| Spark device budget 100 GiB (`107374182400`) | live/config | `ds41rt.config:60` |
| RTX-local boundary 0–19 TP2, Spark 20/20 at 100 GiB | **HISTORICAL (v6 campaign), not the current default guarantee** | `docs/release-v6-performance.md:91-94` |
| Two independent execution lanes around the remote expert boundary | live | `architecture.md`, `docs/ENGINEERING.md:11-14` |
| SparkInfer has an `ep_moe` disjoint-sharding op (different contract) | third-party reference | `third_party/sparkinfer/tests/moe/test_ep_moe.py` |
| `commands/real_full/*` asserts `== 4`, `TARGET_EXPERT_TP = 4`, `expert_parallel == false` | **LEGACY — not on the `serve-native` path; do not refactor unless traced reachable** | `commands/real_full/intermediate_sharding.rs:136-145,301-302,452-453`; `.../coordinator_kernels/target_attention.rs:56,127`; `.../scheduler/execution/progression.rs:284-397` |

No GPU query, workload, build, service or performance run was performed **by
this planning agent**. GPU results quoted elsewhere in this document belong to
the named owning agents and are marked as reported.

## 5. Phase 1 — architecture and exact official weight/runtime budgets

### 5.1 Official geometry (from the checkpoint config)

Source: `rust/crates/ds41rt-loader/src/official-v41-config.json` (validated by
`v41_config.rs` tests).

| Quantity | Value |
| --- | ---: |
| Hidden size H | 5120 |
| Routed intermediate I | 2304 |
| Backbone layers | 40 |
| Routed experts/layer | 384 |
| Top-k | 6 |
| Shared experts | 1 |
| Native routed format | FP4 E2M1 packed pairs + E8M0 K32 scales |
| dSpark | 3 blocks × 128 experts, top-3 |

### 5.2 Exact per-expert and per-layer weight arithmetic (derivation, not measurement)

Native routed tensors per expert, from `expert_format.rs` and the packed
shapes in `docs/release-v7-plan.md`:

| Projection | Logical shape | FP4 payload B | E8M0 scales B | Total B |
| --- | --- | ---: | ---: | ---: |
| w1 gate | [2304, 5120] | 2304·5120/2 = 5,898,240 | 2304·5120/32 = 368,640 | 6,266,880 |
| w3 up | [2304, 5120] | 5,898,240 | 368,640 | 6,266,880 |
| w2 down | [5120, 2304] | 5120·2304/2 = 5,898,240 | 5120·2304/32 = 368,640 | 6,266,880 |
| **per expert** | | | | **18,800,640** |

Per layer (×384): **7,219,445,760 B** = 6.7236 GiB. All 40 layers:
**288,777,830,400 B** = 268.94 GiB. The 18,800,640 figure matches the loader's
own staging test, which is the only cross-check used here.

Per-rank raw routed weight per layer by TP degree (exact tensor windows; no
kernel padding):

| TP | Intermediate/rank | Bytes/layer/rank | GiB/layer/rank | Kernel extent | Padding factor |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2304 | 7,219,445,760 | 6.7236 | 2304 | 1.000 |
| 2 | 1152 | 3,609,722,880 | 3.3618 | 1152 | 1.000 |
| 3 | 768 | 2,406,481,920 | 2.2412 | 768 | 1.000 |
| 4 | 576 | 1,804,861,440 | 1.6809 | 640 | 1.111 |

TP2 and TP3 shard widths are already multiples of 128, so they take **no**
kernel padding. TP4 pads 576→640, the documented 11.1% penalty
(`docs/ds41-expert-tp-efficiency.md`). Derived padded TP4 resident weight is
therefore 2,005,401,600 B/layer/rank. These are arithmetic; the loader-reported
resident byte count is the authoritative value and is a Phase 3 gate.

### 5.3 Runtime budget model and dynamic RTX placement

For each Spark rank on a remote-owned layer set of size `L_s = 40 − L_r`:

```
peak = L_s · resident_layer(TP) + max(load_transients, serve_transients) + reserve
fits <=> peak <= budget
budget = min(SPARK_DEVICE_BUDGET_BYTES (default 100 GiB, unchanged),
             actual MemTotal − reserve)      # conservative parent policy
```

**Memory policy (rev 3.2 + delta):** the user now allows **>100 GB per Spark**
with a **20 GB Linux reserve**. Measured minimum `MemTotal` is
**130,594,156,544 B = 121.6253 GiB**; minus 20 GiB the conservative budget is
**109,119,320,064 B = 101.6253 GiB**. The shipped
`SPARK_DEVICE_BUDGET_BYTES=100 GiB` and the default config are **unchanged**.
Two predicates must be reported, never just one: the launcher is **weight-only**
while the worker requires `resident + workspace ≤ budget`.

With the newly viewed **~970.9 MiB ring per rank at capacity 4096 across two
endpoints**: TP2 at **30 remote layers fails** the 20 GiB-reserve budget by
**~469 MiB**; **29 layers fit the known model but are unmeasured**. At a
20-decimal-GB budget (110,594,156,544 B) 30 layers would leave **~0.915 GiB**
margin and is conditional; **31 cannot fit the declared reserve**. TP3 all 40 is
**~90.4 GiB plus known rings**.

Arithmetic bounds from the memory review (`docs/tp-ep-memory-review.md`, current
source, not measured):

| TP | Weight-only max remote layers | With the worker workspace/transients | Config |
| ---: | ---: | --- | --- |
| 2 | 29 | 29 only if the Spark-side reserve is ≤ ~2.23 GiB and the unaccounted transients are admitted (28 at 3 GiB) | A/C: RTX minimum 11; historical 20-boundary leaves ~9 layers of slack |
| 3 | 40 (44 by weight, capped) | 40 at reserve 0/2/3/10 GiB (~10.35 GiB slack; 39 at 12 GiB) | B: memory is not the constraint |
| 4 (reference) | 40 (53 by weight) | 40 at every reserve shown | shipped TP4×EP1 |

Placement rule (mirrors the existing runtime plan, `run.sh:302-317`):

1. Reserve RTX fixed owners, KV pool, workspaces and runtime headroom.
2. Fill RTX routed layers from layer 0 upward at the local TP degree until the
   next layer would not fit.
3. Publish `rtx_expert_layers = L_r` and `spark_first_layer = min(L_r, 39)`.
4. Spark workers load exactly `[spark_first_layer, 40)` on every rank; each rank
   holds its TP shard of all 384 experts.

RTX per-card routed weight is `L_r · per_layer_rank(local_TP)`:
`L_r=20` is 72.19 GB at TP2 or 144.39 GB at TP1 (TP1 on one 96 GiB card cannot
reach 20; config B must therefore place far fewer local layers). The
**single-RTX placement handoff is not ready**: boot must connect before the plan
is published. A TP3 all-40-remote layout is conditional and must **retain
layer 0 on the RTX until tested**; no dangerous handoff is attempted. The
shipped `RTX_GPUS=auto` means the current dual split is selected at runtime; it
is read from the launch, never assumed.

### 5.4 Phase 1 deliverables and gates

| Deliverable | Gate | Status |
| --- | --- | --- |
| Architecture record for configs A/B/C: rank→(group,local_rank) map, local vs remote layer split, per-layer weight tables | `docs/tp-ep-architecture-audit.md` (done, 772 lines) **plus current-source corrections in §4**; disagreements resolved before implementation | **PARTIAL** — audit done, but several live-path hardcode claims are superseded by current source; §4 is authoritative |
| Exact per-config budget tables at the chosen `L_r` | Startup reports per-rank resident weight, workspace, staging, headroom; each ≤ budget with zero fallback; `L_r` republished and matched to the loaded first layer | **PARTIAL** — arithmetic bounds recorded (TP2 29 / TP3 40); measured values pending (G1) |
| Dynamic placement uses the actual per-config workspace | No silent shrink; infeasible plan fails startup (existing contract); single-RTX handoff boots before publishing | **PARTIAL** — handoff not ready; TP3 all-40 conditional retains layer 0 until tested |
| EP semantics documented as replication (not expert_map dispatch) | Reviewer sign-off; `ep_moe` not used | **PASS (design, reported)** — single-level N-plane contract frozen; `ep_moe` not used |

## 6. Phase 2 — topology, rank/shard/transport contracts and the scheduler

### 6.1 Rank, group and shard contract (freeze before coding)

- `EP` groups × `TP` ranks; the frozen formula is
  `group = rank_global / TP`, `local_rank = rank_global % TP`
  (equivalently `rank_global = group · TP + local_rank`, group-major so each
  group is a contiguous peer range).
- Expert sharding inside a group is the native FP4 window on the intermediate
  dimension: `intermediate_start = (2304/TP) · local_rank`; gate/up rows and down
  columns and their K/32 scales shift together. The live path now resolves this
  through `V41SparkTopology` (`rank % tp`, world = tp) and the loader's generic
  `BackboneTp { world: 2|3|4 }` staging; the older
  `DS4_EXPERT_TP_WORLD_SIZE=4` generic helper in `expert_format.rs` is not the
  new-path source of truth.
- Layer ownership is unchanged: a layer is either fully RTX-local or fully
  Spark-remote; the Spark set starts at the published boundary.
- Route assignment (the EP degree of freedom): every route
  `(row, expert_id, gate_weight)` is assigned to exactly one group by the
  scheduler; all TP ranks of that group receive it. The chosen representation is
  a per-route **owner word** plus a fixed top-6 slot layout per token that
  permits 0..6 owned routes; unassigned slots carry a `sentinel 384 + 0` weight.
  A token therefore can contribute to zero, some, or all six slots.
- Transport: the coordinator addresses each group's TP ranks directly; the wire
  contract carries `(group, local_rank, world, layer)` explicitly and must not
  infer ranks from array positions or sized `[T;4]` types
  (`ds41rt-transport/src/v41_expert.rs:272,299,352` is the live fixed-4
  collector to replace).
- **Wire encoding (source/build verified):** a **one-hot owner bitmap at bits
  9..11**, exactly `1 << (9 + owner)`; the API exposes a group index but the
  wire carries a bitmap. Zero/multiple bits and owners outside the configured
  EP are rejected. The field supports groups 0..2 (EP ≤ 3), not EP ≤ 8.
  Transport namespace `source` identifies the physical rank. See the final
  integration review for the matching measured-artifact source provenance.

### 6.2 Reduction/assembly contract (frozen baseline)

The first implementation performs **no group-local (Spark→Spark) reduction**.
Every physical rank plane returns directly to the coordinator:

1. All `N = TP · EP` rank planes (`N ∈ {2,3,4,6}`) arrive at the coordinator as
   complete FP32 `[rows, 5120]` partials.
2. The coordinator performs one **ordered FP32 N-plane sum** in a fixed rank
   order (group-major: `group · TP + local_rank`).
3. The shared expert is added **once**.
4. A single final BF16 rounding.

This preserves the existing per-route arithmetic contract (FP32 rank
sum, one shared add, one BF16 rounding) and makes exactly-once trivial to audit
because there is no intermediate group result to double count. A two-level
TP-then-EP reduction is a **later optional alternative only**, not the chosen
path, and is not required for any Phase 1–6 gate.

Non-owning ranks/groups must contribute nothing (see the sentinel in §6.1), and
only owned routes are dispatched. The canonical route order must be fixed so
results are reproducible; PENDING gate: bit-exact or explicitly bounded
comparison against the TP4 result for the same batch.

Required native work (`f6d0b154-…`, `docs/tp-ep-native.md`): the ordered FP32
**generic 2/3/4/6-plane** reducer (landed) and the TP2/TP3 SM121 static roles; the
existing `planes[4]` / `planes[2]` reducers cannot express 3 or 6 ranks
(`native/include/ds41rt_v41_experts.h:73-110`). Kernel masking details are
specified by the completed `docs/tp-ep-kernel-audit.md` (765 lines).

### 6.3 Expert-aware greedy scheduler vs simple baselines (initial scope)

The CPU-only scheduler module is already assigned (`afd3de8a-…`,
`replicated_expert_schedule.rs`): deterministic, preallocated, greedy LPT over
whole experts with a calibrated cost API, 1/2/3 groups. The **initial**
comparison stays deliberately small so it can gate Phase 2 quickly:

| Candidate | Definition |
| --- | --- |
| B1 round-robin | route index round-robin across groups |
| B2 route-count | a simple static cost/route-count split (no expert awareness) |
| S1 greedy LPT | the assigned scheduler, whole-expert LPT over measured frequencies |

Wider baselines (whole-expert modulo/range, hash(row,expert), route-level
greedy) and richer cost models are **deferred**: only add them if S1 does not
already dominate B1/B2 on imbalance and exact-once, or if Phase 6 timing shows
the imbalance bound is the limiting factor. Do not grow the baseline matrix
before that evidence exists.

Metrics: group route-count max/mean, imbalance ratio, predicted vs measured
group time, and weight-reuse proxy (distinct experts touched per group). The
sharing-vs-balancing trade-off is explicit: concentrating a hot expert's routes
in one group maximizes resident-weight reuse but unbalances compute; spreading
them balances but lowers reuse. Report both, decide nothing without Phase 6
timing.

### 6.4 Phase 2 gates

| Gate | Status |
| --- | --- |
| One frozen rank→(group,local_rank) function shared by wire, shard and reduce; unit-tested for TP∈{2,3,4}, EP∈{1,2,3} | **PASS (CPU, reported)** — scheduler `replicated_expert_schedule.rs` 15 unit + full core (reserve-Vec fix: 16 focused); loader generic `BackboneTp` 82 lib + 5 new + 11 integration; transport 180 lib + 14 integration incl. 6-rank loopback (pre-canonical validation + fail-closed topology flag) |
| Shard math generalized to runtime world and rejects misalignment (see risk R1) | **PASS (CPU, reported)** — loader stage for world 2/3/4 incl. independent raw real-checkpoint check; GPU execution still pending under G7 |
| Scheduler deterministic and preallocated; identical inputs → identical assignment | **PASS (CPU, reported)** (15 unit + full core; reserve-Vec review fix 16 focused) |
| S1 greedy LPT does not lose to B1/B2 on imbalance on held-out real route banks, with no exactly-once violation | **PARTIAL** — scheduler CPU complete; the held-out baseline comparison is not yet reported as a closed artifact |
| Sentinel `384 + 0` weights and `rows == 0` metadata prove unowned slots skip active computation while the launch grid keeps its fixed upper bound | **PENDING** — mask/replay tests + benchmark assigned (`fdcd9d82-…`); GPU execution awaits the exclusive environment |

## 7. Phase 3 — opt-in configuration, launch, loader/admission plumbing

Preserve TP4×EP1 and every existing default.

- Config: add opt-in `SPARK_GROUPS` (EP) and `SPARK_TP` (TP), or a single
  `SPARK_TOPOLOGY=tp2ep2`. Validate `SPARK_TP · SPARK_GROUPS == SPARK_COUNT`
  and `SPARK_TP · SPARK_GROUPS ∈ {4,6}` against `SPARK_COUNT ∈ {0,2,4,6}`.
  Default must resolve to `TP=4, EP=1`. Owned by `fa33b227-…`
  (`docs/tp-ep-configuration.md`): launcher/config scripts, examples and tests
  are in scope, but **no `ds41rt.config` default change and no `build.sh`/`wip.sh`
  edit yet**.
- Launcher (`run.sh`, `scripts/release-common.sh`):
  - extend `SPARK_COUNT` to `6` and `SPARK_4/SPARK_5` host + `LANE_A/B` keys
    (allowlist `release-common.sh:104`), with all-or-none rail validation;
  - pass `--group`, `--local-rank` (or `--tp`), `--world` to each worker; keep
    the group-major `rank = group·TP + local_rank` formula in one place;
  - keep the dual-RTX placement handoff (`plan.json` → `ready.json`) and extend
    the plan schema with `expert_groups`, `spark_tp`, `per_group_rank_base`;
  - fingerprint the new fields (`run.sh:238`) so A/B differs only by topology.
- Loader/admission:
  - generic staging `BackboneTp {layer, expert, rank, world}` for world 2/3/4,
    preserving every existing variant and reading through existing safe paths
    (`docs/tp-ep-loader.md` `[agent 885b74a4-…]`); no catalog edits;
  - runtime world validation on the **live** path replaces the `matches!(world, 2|4)`
    gates (`v41_experts/coordinator.rs:119-120`, `v41_experts/service.rs:133-135`)
    and the 2/4-only reducer dispatch (`coordinator.rs:368-388`); the legacy
    `commands/real_full` `== 4` asserts are not the target;
  - per-rank weight budget check precedes allocation (existing admission
    contract, `DEVELOPER.md:80-84`); reject before allocating on shortfall;
  - EP replication is accounted as replicated bytes, not extra logical tokens
    (mirrors the TP2-attention replica accounting precedent, v6 plan
    `Experimental serving switch and replicated pool sizing`).
- Preserve TP4: `expert_format.rs`, reducer selection and shard math must keep
  the TP4 path byte-identical in arithmetic and defaults.

### Phase 3 gates

| Gate | Status |
| --- | --- |
| `run.sh --dry-run` and launch tests accept `tp2ep2` on 2 RTX + 4 Spark and reject inconsistent combinations | **PASS (CPU, reported)** — config/launcher stage 1: 257 pass / 1 skip |
| Default config still resolves to TP4×EP1; `scripts/tests/test_native_release_launcher.py`, `test_compact_spark_topology.py`, `test_placement_handoff.py`, `test_release_ab_controls.py` pass unchanged-or-extended | **PASS (CPU, reported)** — stage 1 includes launcher/config tests; no `ds41rt.config` default change |
| Six-Spark config parses and dry-runs through production plumbing (not docs-only), with six-Spark E2E kept qualified-pending (hardware connected; provisioning/RDMA unqualified) | **PARTIAL — CPU pass reported** (stage 1; approved set `2x2\|3x2\|2x3\|4x1`). Build/WIP is **environment only, not yet the native CLI** (`fa33b227-…`); six-Spark E2E remains unqualified |
| Loader admits/rejects on real per-rank numbers; startup logs report exact resident/workspace/staging/headroom per rank | **PARTIAL — loader CPU pass reported** incl. independent raw real-checkpoint check; live startup budget is G1 |
| TP4 native arithmetic and defaults unchanged (regression suite) | **PENDING** — native GPU regression not yet run |

## 8. Phase 4 — concurrent group dispatch, reduction and assembly semantics

Required behavior and failure coverage:

1. **Concurrent group dispatch.** Both independent lanes dispatch to all groups
   concurrently; per-lane and per-group buffers/streams/events are owned, never
   shared mutably.
2. **Direct N-plane assembly.** All `N = TP · EP` physical rank planes return to
   the coordinator and are summed in one ordered FP32 N-plane reduce
   (`N ∈ {2,3,4,6}`), with the shared expert added once and a single final BF16
   rounding. No group-local reduction. The assembler proves contribution counts:
   each route's rank contributions == TP, its owning-group contributions == 1,
   and no non-owning rank contributes.
3. **Route weights and exactly-once.** `gate_weight` is applied once per route,
   in the group that owns it; assembly must not scale or double count. A route
   delivered to a non-owning group is a hard error, and the sentinel
   `384 + 0` slot must never enter the reduce as a real contribution.
4. **Empty ranks/groups.** A group with zero owned routes still participates in
   ready handshake and reduction with a defined empty result; no hang, no stale
   buffer reuse. Empty TP ranks (if the scheduler ever yields zero routes for a
   rank) produce zeros, not garbage.
5. **Repeated experts.** Many rows routed to the same expert must be batched
   per group without changing per-route arithmetic.
6. **Balance.** Per-group route/row counts stay within the scheduler's declared
   bound; the runtime records the achieved imbalance.
7. **Cancellation/failure.** Cancelling or failing one group/lane drains or
   revokes only its owners; the other lane keeps progressing; no partial
   publication.
8. **Graph lifetimes.** Per-group captured graphs key on group identity, layer,
   capacity, lane, request partition and cache identity; a changed assignment
   either replays a compatible graph or recaptures after draining, never
   allocates inside replay.

### Phase 4 gates

| Gate | Status |
| --- | --- |
| Rust/CUDA tests for exactly-once over adversarial assignments (repeated experts, empty group, single-route group) | **PARTIAL — CPU contract pass reported** (scheduler/loader/transport) and the reducer's **RTX1 GPU selftest PASS**; full assembly on real expert weights pending |
| Empty-group and empty-rank progress tests | **PARTIAL — CPU contract coverage reported**; daemon/live pending |
| Cancellation/failure isolation: one lane's group failure cannot block the other | **PENDING** |
| Independent-lane trace check (`scripts/verify-ds41-independent-lane-trace.py`) still passes | **PENDING** |
| Graph replay with changed assignment/inputs performs no allocation and no stale read | **PENDING** — native GPU pending |

## 9. Phase 5 — official-only TP2/TP3/TP4 kernel selection

- Exports: add SM121 Spark AOT variants for TP2 (1152) and TP3 (768); keep TP4
  (576→640) unchanged; keep SM120 RTX `rtx_tp2` (1152) and `rtx_backbone`
  (2304). Role geometry comes from `python/tools/export_b12x_v41_experts_aot.py`
  (`experts, intermediate, topk = (384, 576, 6)` for Spark today).
- Small-M top-6 split across groups. With route ownership, a decode row's six
  routes may be split across groups. Required design:
  - fixed top-6 slot layout with 0..6 owned routes per token;
  - unassigned slots carry the worker **sentinel `384 + 0` weight** so a masked
    slot can never be confused with a real expert or a real weight;
  - the kernel performs a **true early skip of active computation** for masked
    slots via `rows == 0` metadata, while the launch grid keeps its fixed
    **upper-bound** shape (so graphs stay shape-stable);
  - a group is given only its owned routes. Broadcasting all top-6 with zero
    weights is rejected as a performance design.
  The completed `docs/tp-ep-kernel-audit.md` (765 lines) specifies the exact
  skipping semantics; mask/replay tests and benchmark implementation are
  assigned and wait only on the exclusive GPU environment.
- TP3 kernel measurement is supported on a **single connected Spark** (one SM121
  rank, width 768); six-Spark end-to-end serving follows four-node qualification
  and the `rhea`/`moa` provisioning/RDMA gaps. The single-rank measurement is a
  Phase 5 deliverable.
- Correctness ladder:
  1. synthetic fixtures (per-width partials vs an unsplit oracle);
  2. real official weights (TP2/TP3 shards of one shared tensor, not
     independently generated shards);
  3. changed-input graph replay without allocation;
  4. complete FFN (routed + shared + residual) against the TP4 reference;
  5. communication overhead measured **separately** from compute.
- Qualification inputs to reuse: `python/tools/qualify_v41_slice_native.py`,
  `qualify_v41_expert_slices.py`, `qualify_v41_spark_native.py`,
  `qualify_v41_tp2_numerics.py`, `qualify_v41_tp2_reduction.py`,
  `compare_v41_expert_upstream.py`; SparkInfer
  `tests/moe/test_v41_tp4.py`, `test_v41_grouped_slices.py`,
  `benchmarks/benchmark_v41_grouped_slices.py`.

### Phase 5 gates

| Gate | Status |
| --- | --- |
| TP2/TP3/TP4 shard numerics match the unsplit oracle at rows 1/16/80/(capacity); tolerance stated | **PARTIAL — native** roles/packer/reducer present (101 FFI + 18 Python reported); reducer **RTX1 GPU selftest PASS**; TP3 packer **synthetic PASS only**. Real-checkpoint packer, GPU compile and expert numerics pending |
| Real-official-weight (not synthetic) shard correctness | **PENDING** — GPU |
| Changed-input graph replay, no allocation | **PENDING** — GPU |
| Complete FFN matches TP4 reference within stated tolerance or bit-exact | **PENDING** |
| Wire/communication bytes and time quantified separately per layer | **PENDING** — transport CPU contract done, timing pending |
| Masked/inactive routes proven to skip compute | **PENDING** — mask/replay tests + benchmark assigned (`fdcd9d82-…`); GPU execution awaits the exclusive environment |

## 10. Phase 6 — isolated live comparison TP4 vs TP2×EP2 (config A)

Serialized and isolated: one WIP slot at a time
(`./wip.sh --slot NAME --role both`), builds and performance runs never overlap
(`AGENT_DEV_HINTS.md:5,16`), services restored to the default afterwards.

- **Matched conditions.** Same `L_r` and first layer, same KV pool, same
  concurrency, same prefill batch, same checkpoint/revision, same power cap and
  memory clocks, same corpus and prompts. Topology is the only changed variable;
  if `L_r` must differ because of the budget, that difference is disclosed.
- **Workloads.** Code / topic / mixed decode at C1, 2, 4, 8, 16 and a
  representative prefill matrix; target-only and dSpark controls.
- **Repeats.** Three final repeats per cell after warmup; record every sample,
  median and variance. Failures kept.
- **Recorded per arm:** wall time, decode tokens/s, prefill tokens/s,
  acceptance/completion checks, peak and post-readiness memory per device,
  startup until ready and expert load time, graph capture/replay timings,
  per-layer group dispatch/reduce/assembly time, achieved imbalance.
- **Numerical/behavioral checks:** output/acceptance equality or stated bound,
  request/cancellation behavior, constraints/tools smoke, retained-prefix
  reuse, long-context needle for affected paths.
- Legacy v6/v7/v8 numbers are **not** used as the fresh baseline; the TP4 arm is
  re-measured in the same campaign. The v6 "RTX 0–19 / Spark 20" boundary is
  **historical**: the shipped config is `RTX_GPUS=auto`, so the current dual
  split is selected at runtime and must be read from the launch, not assumed.

### Phase 6 gates

| Gate | Status |
| --- | --- |
| Matched-config TP4 baseline and TP2×EP2 candidate measured in one campaign | PENDING |
| Correctness/acceptance pass before any performance number is reported | PENDING |
| Startup, memory and graph timing reported separately from request execution | PENDING |
| Three-sample medians with variance; every timed sample retained | PENDING |
| No default changed; services restored | PENDING |
| Configs B/C: production plumbing + CPU validation + single-Spark TP3 kernel measurement complete; **six-Spark E2E** follows four-node qualification once `rhea`/`moa` have matching images, the pinned checkpoint and qualified RDMA providers | PENDING |

## 11. Phase 7 — regression, reproducibility, config plumbing

- Regression review: TP4×EP1 defaults, single/dual/zero-Spark launches,
  validation suites, `run.sh --dry-run`, image/launcher identity checks. Only
  live-path code is in scope; no broad legacy `real_full` refactor.
- Reproducibility evidence: source commit, SparkInfer revision, checkpoint
  revision, config, exact launch command, corpus hash, raw samples, artifact
  hashes — the existing evidence conventions (`DEVELOPER.md:96-105`).
- **Six-Spark configs B and C are production plumbing, not documentation-only**
  (`fa33b227-…`): opt-in config, launcher support, examples and CPU validation
  are real deliverables, and the TP3 width-768 kernel has a single-Spark
  measurement path. Six-Spark hardware is now connected (`rhea`/`moa`), but
  **six-Spark E2E is qualification-pending** — no images/checkpoint on those
  hosts and RDMA providers unqualified at inventory — and it follows four-node
  qualification. Every B/C artifact must say exactly that, and no E2E
  throughput/memory/correctness claim may be attached before it runs.
- No release branch, no image publish, no default promotion, no `ds41rt.config`
  default change.

### Phase 7 gates

| Gate | Status |
| --- | --- |
| Regression suites and default-path checks pass | **PARTIAL** — launcher/config stage 1 reported 257 pass / 1 skip; native/service regression pending |
| Evidence bundle reproducible from the recorded commands and hashes | **PENDING** |
| B/C config+launcher+CPU validation and single-Spark TP3 measurement delivered; six-Spark E2E follows four-node qualification (provisioning/RDMA pending) | **PARTIAL — plumbing/CPU reported** (config stage 1 + native roles 5/6). Single-Spark TP3 GPU measurement pending |
| No default/release change in the working tree | **PARTIAL** — constraint honored so far; final end-state check pending |

## 12. Acceptance gates — consolidated honest status

Status legend: `PASS (CPU, reported)` = owning CPU workstream reports passing
tests; **not independently re-run by this planning agent**. `PARTIAL` = the CPU
subgate is reported done but a GPU/daemon/live subgate is not. `PENDING` = no
gate evidence.

| # | Gate | Status |
| --- | --- | --- |
| G1 | Config A budget closes at a measured `L_r` with zero fallback | **PARTIAL — live run + corrected pilots, memory NOT closed; no completion declared.** Live `N=20`/spark-20, 4 Sparks ~24 s, coordinator 95.7 s, SSE smoke PASS. Corrected C1 counting 3/3 PASS 191 tokens TPS median **226.50**; C2 six-request PASS median **342.32** (dSpark cached, not TP4); 604-token uncached prefill probe PASS 10.28 s (not throughput). Rings: two records/rank, second `268,435,456 B`, allowance `1,024,851,968 B`; **lane-1 requirement retracted** (`execution_lane=0` hardcoded for both wave clients). Coordinator CAP 2048; request-2048 growth test pending. Needs new post-alloc diag logs; matched TP4 baseline **not run**. See §3.5, §3.6 |
| G2 | Rank/group/shard/wire contract frozen and unit-tested | **PARTIAL — CPU pass reported**: scheduler 15 + full core; loader 82 lib + 5 new + 11 integration (incl. raw real-checkpoint); transport 180 lib + 14 integration (incl. 6-rank loopback); **daemon 835 pass / 6 baseline-proven fails / 101 ignore**; E2E CPU 40 tests + 37 config tests; review 885b A/B/C PASS (teardown D live pending). E2E serving not run |
| G3 | TP4 defaults byte-unchanged; opt-in only | **PARTIAL — CPU pass reported**: launcher/config stage 1 257 pass / 1 skip. Native TP4 arithmetic regression pending |
| G4 | Loader admission on real per-rank numbers | **PARTIAL — CPU pass reported** incl. raw real-checkpoint loader check; memory-review peak model `L·resident + max(load,serve) + reserve` documented. Live startup budget pending under G1 |
| G5 | Exactly-once assembly incl. empty/repeated/cancel cases | **PARTIAL** — CPU contract pass reported; reducer RTX1 GPU selftest PASS; **native component correctness CLOSED 135 (all-rank) with exact inactive zero and allocation delta 0**; full E2E exactly-once assembly still pending |
| G6 | Group dispatch truly skips unowned routes | **PENDING** — kernel scaffold review findings are inputs to pending selection/optimization (no mandatory rewrite); native component correctness CLOSED, integrated runtime correctness not yet accepted |
| G7 | TP2/TP3/TP4 official-weight numerics + graph replay | **PASS (native component, reported)** — **component correctness CLOSED 135 (all-rank)** (TP2 r0/1, TP3 r0/1/2; layers 11/20/39; caps 1/80/256/4096; changed-activation oracle max rel `0.0017589`/cos `0.99999845`; pack byte 0; compaction max rel `0.0027429`/cos `0.99999619`; exact inactive zero). **Integrated runtime** graph replay in serving still pending |
| G8 | Complete-FFN equivalence to TP4 | **PENDING** |
| G9 | Matched TP4 vs TP2×EP2 decode/prefill, 3 repeats | **PENDING** — candidate is LIVE but the TP4 baseline has **not** been run; no performance-improvement claim; the C1 counting numbers (~226–230 tok/s) are not a matched comparison |
| G10 | Startup/memory/graph timings separated | **PARTIAL** — native M1 pilot warm numbers (TP4 a6 `156.37 µs` vs TP2 a3 `155.11 µs`; TP3 a3 `62.27 µs` vs TP2 a2 `62.17 µs`) and cold 32 MiB `225.98`/`229.12 µs`, but these are **component pilots, not serving**, and the warm 62→155 µs cause is **unproven** (L2 cache-capacity vs CTA-occupancy); six-Spark figures projected only. Startup/memory separation pending |
| G11 | Request/constraint/cancellation checks | **PENDING** |
| G12 | Independent-lane preservation | **PENDING** |
| G13 | Reproducibility bundle | **PARTIAL** — reducer selftest meta, L4/L5 result records and the **built+staged** candidate manifest `d63d8b50…` (x86 `9a4bf37b…`, ARM `08cd9a1f…`, libs `1160095c…`/`d453812c…`) exist; a single consolidated bundle pending |
| G14 | B/C production plumbing + CPU validation + single-Spark TP3 kernel measurement done; six-Spark E2E qualified after four-node qualification | **PARTIAL** — plumbing/CPU reported; **native component correctness CLOSED 135 includes TP3 ranks 0/1/2**; candidate binaries staged on raptor + 4 Sparks, `ldd` 0 missing; six-Spark E2E pending (hardware connected, provisioning/RDMA unqualified; M1 six-Spark timing is projected only) |
| G15 | No default/release promotion | **OUT OF SCOPE / NOT REQUIRED** — the user explicitly excludes default serving and release promotion, so this is a scope boundary, not a pending completion item. Default remains unchanged |
| G16 | Case (1) 2 RTX + 4 Spark TP2×EP2 vs TP4×EP1 matched E2E | **PARTIAL — comparison corrected: 372 completed both; G3 372 content, TP4 371 (truncated code 512, not an established semantic error); C2+ candidate excess unadjudicated**; no TP2 speedup; ring/credit behavior via `d12965bd-…` direct-RC awaiting the `915` quiet window (G23 not required) |
| G17 | Case (2) 1 RTX + 6 Spark TP3×EP2 E2E, incl. **measured** qualification of the 97.565 GB single-RTX all-40-remote model | **PENDING** — S3 gate |
| G18 | Case (3) 2 RTX + 6 Spark TP2×EP3 E2E with explicit six-node host-count identity | **PENDING** — S4 gate; runner extension is a separate version |
| G19 | Case (4) 2 RTX + 6 Spark TP3×EP2 vs TP2×EP3 matched topology (same placement/KV/power/dSpark) | **PENDING** — S5 gate; no TP2EP3-faster claim |
| G20 | Six-node matrix integrity: no fake 4-rank result, no unmatched TP2/TP3 remote-layer comparison presented as like-for-like, full-fleet GPU runs sequential under `91570601-…`, existing 4-run uninterrupted | **PENDING** — S2/S4/S5 evidence rule |
| G21 | G2 failure root cause (decode graph-instantiation OOM at GPU1 4 MiB free) triaged; graph-cache growth/leak/cleanup bounded; failed instantiate destroys its temporary graph (no failed-capture leak in the inspected path); only the reserve aggregate cache vs 800 MiB is identified | **PENDING** — `6ba87b75-…` source review + `7b81d9c6-…` headroom math. Gating = **G3 bounded probes healthy, then monitor the full run for a plateau**; do not block the authorized trial on a pre-run plateau proof (the full run warms the cache) |
| G22 | G3 remediation valid: `N=20`, same binaries/PIDs, exact KV 5,037,542,400, INFO-only reconnect + bounded probes, full 372 only if healthy AND EMU image export finished, stop at first OOM; **no G2 comparison**; both future TP4/TP2 arms at the same G3 KV | **PARTIAL — G3 ran and passed the objective (372/372, 0 runtime errors, 3 greedy-repeat failures), near-plateau not leak proof**; TP4 matched control granted, reload pending; no speedup claim. See §3.9 |
| G23 | Large-frame actual-M 819+ growth | **NOT REQUIRED as a completion gate** — the E2E growth path is unexercised, but it must **not** be provoked by altering prefill solely to test it. Direct-RC proof addresses ring behavior separately (G24-adjacent, `d12965bd-…` test-only plan) |
| G24 | All-4-credit / ring-behavior evidence across all four Spark ranks | **PENDING, optional** — direct-RC (`d12965bd-…`, CUDA pinned host buffers, no GPU compute, two then four) awaits the `915` quiet window |
| G25 | Six-node runtime (cases (2)–(4)) | **PARTIAL — dual six-node arms executed** (TP3EP2 and TP2EP3, counting exact, 3 greedy-drift failures each, no TP2 win); **single-RTX all-40 `1rtx6-tp3ep2` retry authorized** after two pre-allocation CUDA-free-only gate rejections (cause unresolved; allocation never attempted) |
| G26 | Width-192 M8/M16 production comparator + M64 reuse-vs-192 (same `tp-ep-reuse-m64-e384.json` fixture) | **PENDING** — not measured; needs a new lease; no AOT recommendation until then |
| G27 | Sampled uncached prefill comparisons (decode TTFT is not prefill) | **PENDING** |
| G28 | Strict repeatability (C2+ greedy drift; `layer` seed fixes C1 only) | **PENDING/UNRESOLVED** |
| G29 | Final regressions / restoration | **PENDING** |

**Open:** G1, G6–G13, G16–G20, G24 (optional), G25–G29 and the GPU/E2E portions
of G2–G5, G7, G14 remain `PENDING`. **G15 is OUT OF SCOPE** (no default/release by
user direction), and **G23 is NOT REQUIRED** (do not alter prefill solely to force
actual-M growth; direct-RC covers ring behavior). No performance, memory,
numerical or end-to-end number in this document is a result, and none of the
reported CPU counts above has been independently verified by the planning agent.

## 13. Test inventory and mapping

Existing suites (counted read-only; no suite was executed):

| Suite | Count | Location |
| --- | ---: | --- |
| Rust unit/integration `#[test]` | 1793 | `rust/crates/**` |
| Rust async `#[tokio::test]` | 165 | `rust/crates/**` |
| Repo Python tests | 35 | `python/tests/` |
| Repo launcher/quality tests | 35 | `scripts/tests/` |
| Native selftests | 22 | `native/tests/` |
| SparkInfer tensor/MoE tests | ~57 in `tests/moe/` | `third_party/sparkinfer/tests/moe/` |

Mapping of required categories to existing assets (and gaps to add):

| Category | Existing | Gap to add (CPU-only where possible) |
| --- | --- | --- |
| Topology/rank map | `scripts/tests/test_compact_spark_topology.py`, `test_placement_handoff.py`, core `tests/expert_batch.rs` (`expert_host_batch_set_replicates_routes_for_intermediate_shards`) | rank↔(group,local_rank) for TP∈{2,3,4}/EP∈{1,2,3}; 6-rank host/lane validation; reject mismatched `TP·EP` |
| Shard geometry | `expert_format.rs` tests, `python/tests/test_ds4_spark_aot_profiles.py`, `native/tests/v41_nvfp4_padding_selftest.py` | runtime-world shard windows incl. TP3 768 and misalignment rejection; TP4 unchanged |
| Reduction/assembly | `qualify_v41_tp2_reduction.py`, `qualify_v41_local_reduction.py`, `qualify_v41_native_route...`, SparkInfer `test_tp_moe_reference.py`; native `v41_route_reduce_planes_selftest.cc` (RTX1 PASS) | generic 2/3/4/6-plane ordered FP32 reducer CPU contract tests; direct-to-coordinator exactly-once; bit-exactness vs TP4 |
| Scheduler | `src/replicated_expert_schedule.rs` (new), `expert_route_plan.rs`, `python/tools/compare-ds41-route-cost-models.py` | deterministic assignment tests; initial S1 vs B1/B2 on held-out banks; imbalance bound; wider baselines only if needed |
| Dispatch/protocol | `rust/crates/ds41rt-transport/src/protocol_v2*.rs`, `host_batch_set.rs` | group/tp fields on the wire; wrong-group route rejection; 6-peer transport contract tests |
| Launcher/config | `scripts/tests/test_native_release_launcher.py`, `test_release_ab_controls.py`, `test_compact_spark_topology.py` | `SPARK_COUNT=6`, `SPARK_TOPOLOGY`, fingerprint covers topology, default stays TP4 |
| Loader/admission | `v41_catalog.rs`, `v41_expert_staging.rs`, admission tests; new generic `BackboneTp` world 2/3/4 CPU tests (`docs/tp-ep-loader.md`) | per-rank budget with EP replication; infeasible fails startup |
| Numerics/graphs | `native/tests/v41_nvfp4_*`, `qualify_v41_slice_native.py`, SparkInfer `test_v41_tp4.py`, `test_v41_grouped_slices.py` | TP2/TP3 real-weight ladders; small-M route-slot skipping; changed-input replay per group |
| Serving comparison | `scripts/bench-ds41-release-decode.py`, `bench-ds41-concurrent-api.py`, `bench-ds41-release-prefill-matrix.py`, `bench-real-full-*` | matched TP4 vs TP2×EP2 harness; group timing/imbalance counters |
| Quality/constraints | `scripts/tests/test_content_acceptance.py`, `test_concurrency_cases.py`, `qualify-ds41-independent-constraints.py`, `qualify-ds41-tool-eval.py` | cancellation/empty-group serving cases; affected-path retained/needle checks |

Recommended **new CPU-only tests** for the implementation agents (no GPU needed,
fast, and they can be written before kernels exist):

1. `rank_group_map` exhaustive over TP∈{2,3,4}×EP∈{1,2,3}: bijection, group
   contiguity, round-trip.
2. `shard_window` for TP3: exact row/column/scale offsets and byte windows; and
   rejection of a hypothetical misaligned intermediate.
3. Reducer contract stubs: the generic 2/3/4/6-plane N-plane reducer accepts/rejects
   the right plane counts, nulls, aliasing and row bounds; TP4/TP2 unchanged.
4. Exact-once assembler: adversarial route assignments (duplicate route,
   non-owned route, sentinel `384 + 0` slot, empty group, repeated experts) must
   fail or sum once.
5. Scheduler determinism + initial S1 vs B1/B2 comparison on a frozen histogram
   fixture; imbalance bound assertion.
6. Launcher/config matrix: `TP·EP` validation, default TP4, 6-Spark dry-run,
   fingerprint sensitivity to topology only.
7. Budget planner: given synthetic per-rank components, pick `L_r` and assert
   the boundary equality `spark_first_layer == min(L_r, 39)`.

## 14. Risk register

| Id | Risk | Why it matters here | Mitigation / gate |
| --- | --- | --- | --- |
| R1 | **TP3 quantization shard alignment** | 2304/3 = 768; K/32 and 128 granularity must hold for gate/up rows, down columns and scales; a wrong split silently corrupts numerics | Generalize alignment from `world·32`; explicit TP3 window tests; real-weight oracle (G7) |
| R2 | **Native format geometry/padding** | TP4 pads 576→640 (+11.1%); TP2/TP3 do not; exports and resident accounting must not reuse TP4 constants | Per-TP kernel extent in the manifest; loader reports padded resident bytes; Phase 1/3 budget gate |
| R3 | **Transport/topology rank identity** | Largely resolved in current source: coordinator accepts world 2/3/4/6, reducer has a generic 3/6 path, transport topology set is `TP2EP1/TP3EP1/TP4EP1/TP2EP2/TP3EP2/TP2EP3`, owner word bits 9..11, and the serve path builds the topology transport. Remaining: `EXPERT_HOSTS [;4]`/`expert_format.rs` generic helper is not the new path; keep shard/wire/daemon using one `V41SparkTopology` | Runtime world validation on the live path; wrong-size rejection tests; Phase 2/3. No broad legacy refactor |
| R4 | **Group kernel masked-route compute waste** | Broadcasting top-6 with zero weights costs EP× compute and hides the benefit; EP must use the grouped pipeline (compact pipeline launches route CTAs and writes zeros) | Sentinel `384 + 0` masked slots + `rows == 0` metadata for true active-compute skip, fixed upper-bound grid for graph stability; dispatch only owned routes; reject zero-weight broadcast (G6) |
| R5 | **Static graph shape vs dynamic allocation** | Group assignment changes row/route counts per batch; graphs are shape-stable | Graph key includes group + route-count bucket; recapture after drain; no replay allocation (G5/G7) |
| R6 | **Load transients** | TP2/TP3 per-rank shards are larger than TP4; per-layer load peak is `resident + staging + pinned + read` and the admission checks omit pinned/read/exchange/row-index bytes; GB10 has one unified pool and page cache can hold ~97 GiB | Count pinned/read/exchange/row-index in the Spark budget; add `SPARK_RUNTIME_HEADROOM`; the installed reclaim helper is a **system-wide `sync + drop_caches=1`** (not per-file FADV); keep `L_r` feasible with headroom (G10) |
| R7 | **Compute vs wire latency** | EP can trade wire for compute; TP3 changes partial sizes | Measure compute, reduction and wire separately per layer; do not infer from whole-serving (G10) |
| R8 | **Preserving two independent lanes** | Shared group state or a single completion event would serialize lanes | Lane-owned streams/events/buffers; cancellation isolation; independent-lane trace (G12) |
| R9 | **Sharing weight reuse vs balancing** | Concentrating hot-expert routes maximizes reuse but unbalances; spreading balances but loses reuse | Scheduler reports both; decide only on matched timing (G9) |
| R10 | **Exactly-once double count** | Group assignment plus shared-expert addition is easy to double count; zero *owned routes* is not zero token rows, so empty groups must still return a full `[M,H]` zero plane | Contribution-count proof; adversarial tests (G5) |
| R11 | **Legacy evidence contamination** | v6/v7/v8 tables are different topologies/quant formats; the memory review shows historical `V41_EXPERT_AOT.json`/qualification tables disagree with current source and must not size TP2/TP3 | Fresh matched TP4 baseline; regenerate manifests and read `info.scratch_bytes`; legacy numbers labelled background only (G9) |
| R12 | **Over-fitting the budget to zero workspace** | Launcher is weight-only; worker requires `resident + workspace`, and transients/pinned/page-cache are unaccounted; default 100 GiB mask the real `MemTotal` pool | Report both predicates; conservative `MemTotal − 20 GiB reserve`; measured readiness values required (G1/G4) |
| R13 | **Six-Spark qualification gap** | `rhea`/`moa` (ranks 4/5) are connected but have no matching images or checkpoint and unqualified RDMA userspace; objective revision 4 now **requires** six-node qualification, so the gap is on the critical path, not optional | Provision devtools/images/verified source/pinned checkpoint and qualify RDMA providers; run cases (2)–(4) after case (1) with the separate-version host-count-identity runner; attach no E2E claim before it runs (G16–G20) |
| R14 | **Kernel scaffold findings** | Parent review found wrong TP assembly, unused checkpoint, uninitialized outputs, wrong stage signatures, scale axes, padding and import path; these are **inputs, not a scheduled rewrite** | No mandatory rewrite assumed; treat as kernel **selection/optimization** inputs (pending). Native component correctness CLOSED 135; integrated runtime correctness not accepted (G6/G7) |
| R15 | **Hardware budget uncertainty** | User allows >100 GB/Spark with 20 GB Linux reserve; measured min `MemTotal` 130,594,156,544 B gives a 101.6253 GiB conservative budget. The ~970.9 MiB two-endpoint ring makes **TP2 30 layers fail by ~469 MiB**; 29 fit the known model but are unmeasured; 31 cannot fit | Measure `cudaMemGetInfo` at readiness + per-layer load peak with role-5/6 weights and execution live; keep the shipped 100 GiB default until then (G1) |
| R16 | **Launcher vs daemon approved sets** | Launcher approves `2x2\|3x2\|2x3\|4x1`; the daemon/topology type also accepts `TP2EP1`/`TP3EP1` | Both reject unambiguously, so not a correctness hole; reconcile or document the difference (foundation review open item 2) |
| R17 | **Service state / restore** | The original service is stopped with containers retained; the fleet is now leased to `91570601-…` with the candidate **LIVE** and all other GPU work stopped, so any restore is gated on that lease | Restore and verify before Phase 6; serialize with the integration coordinator `91570601-…` (§3.5) |
| R18 | **Build filesystem / stale-target collision** | `/mnt/scratch` is a buggy read-only NTFS drive; Cargo/native builds and container aliases must never target it, and the old daemon target has filesystem read errors | Use `~/.cache/ds41rt/builds/<task>`, run `scripts/assert-build-filesystem.py` (symlink/`findmnt`-aware, fails closed), use `ds41rt-tpep-nvme-dev` and the fresh `…/daemon-tp-ep-target`; keep the old NTFS container stopped (urgent commit `92e29c7`) |
| R19 | **Warm 62→155 µs cause unproven (not a wave proof)** | Two hypotheses: (a) an **L2 working-set/cache-capacity confound**, but the ~24 MiB GB10 L2 value is an **unverified source property** (no source located) — stated only conditionally ("if ~24 MiB then the working-set fits/does not"); (b) a **CTA-occupancy cliff** (`54` vs `36`) from source counts. The resource census (REG 122/block 128, 4-CTA/SM register upper bound) does **not support** the simplistic 1→2 wave story, and the cache hypothesis is **not proof** | Read-only resource census by `f6d0b154-…`; profile before changing width or claiming an occupancy cause; treat the **real 29-layer stream cold** case as the more important measurement (G10) |
| R20 | **Graph-cache growth / instantiate OOM** | Full G2 failed with a decode `cudaGraphInstantiate` OOM and GPU1 at 4 MiB free; reachable caches are bounded but large (`LayerGraphs ≤ 65/layer/bank`, dual tails 64/layer/2 lanes) with instantiate-before-evict +1. **Failed instantiate destroys the temporary graph; no failed-capture leak found in the inspected path.** Only gap: reserve aggregate cache vs 800 MiB. No leak proven or ruled out globally | Triage by `6ba87b75-…` + headroom math by `7b81d9c6-…`; G3 bounded probes must be healthy, then **monitor the full run for a plateau** (G21/G22) |
| R21 | **Prefill wire-M unresolved** | The actual prefill wire-M issue is unresolved and the observed `402,653,184 B` ring was three small endpoint reservations from overlapping sessions, **not** large-frame growth | Keep the 8 MiB-slot/depth-8 records as the only attributable ring bytes; resolve wire-M before any large-frame or prefill claim (G1/G21) |

## 15. Evidence and anti-invention rules

- Every reported number carries: source commit, SparkInfer revision, checkpoint
  revision, config, exact command, corpus/input hashes, and whether it is
  synthetic, real-weight, component or end-to-end.
- Derived arithmetic in this plan is labelled as derivation. It must be replaced
  by a runtime report before any gate is marked passed.
- No gate is marked passed from an audit document alone; audits constrain the
  design, they do not measure.
- Failures, timeouts, regressions and rejected candidates are retained.
- Component timing never substitutes for serving throughput, and vice versa.

## 16. Open questions for the parent

1. Budget hinge: under the conservative `20 GiB actual MemTotal − 20 GB reserve`
   policy (user now allows >100 GB/Spark), does the measured Spark workspace
   keep 29 remote TP2 layers, or does the reserve force 28? This decides whether
   Phase 6 is matched at the historical v6 boundary or at a new measured one.
   The shipped 100 GiB default is unchanged.
2. Scheduler granularity: is whole-expert LPT (S1) sufficient to dominate B1/B2
   on imbalance, or is route-level greedy required? Decide from the initial
   Phase 2 comparison; wider baselines are deferred until then.
3. Kernel scaffold: after the parent-review rewrites (TP assembly, checkpoint,
   outputs, stage sigs, scale axes, padding, import path), confirm the sentinel
   `384 + 0` + `rows == 0` skip and whether a compacted-route variant is needed.
   This waits on the rewrites and the exclusive GPU environment (`91570601-…`).
4. Reconciling the launcher approved set (`2x2\|3x2\|2x3\|4x1`) with the
   daemon/topology type also accepting `TP2EP1`/`TP3EP1` (foundation review open
   item 2) — reconcile or document deliberately.

Frozen decisions (no longer open): direct-to-coordinator ordered FP32 N-plane
reduction with no group-local reduction; `group = rank / TP`,
`local_rank = rank % TP`; six-Spark configs are production plumbing, hardware is
connected, and six-Spark E2E is qualification-pending after four-node
qualification; the 100 GiB default budget is unchanged until measured.

## 17. Review recommendations

1. **No mandatory kernel rewrite is assumed.** The audit is done (765 lines) and
   listed the contract; the parent review found the scaffold wrong on TP
   assembly, checkpoint use, output initialization, stage signatures, scale
   axes, padding and import path. Those are **inputs to kernel
   selection/optimization (pending)**. Do not accept any integrated runtime GPU
   correctness or
   timing until the rewrites land. It waits only on the exclusive GPU
   environment owned by `91570601-…`. Non-live `real_full` code is not a target.
2. **Make the budget closure (G1) the first hard gate**, using the conservative
   `MemTotal − 20 GiB reserve` policy and the worker predicate
   `resident + workspace`, not the launcher's weight-only number. Report both.
3. **Require the direct N-plane exact-once assembler and empty/duplicate/sentinel
   route tests before any live run** (G5). CPU contract coverage is reported and
   the reducer's RTX1 selftest passed; full assembly on real expert weights is
   still the gap. Remember zero *owned routes* is not zero token rows.
4. **Freeze the rank→(group,local_rank) map in one function** used by shard,
   wire, reducer and launcher; current source already uses `V41SparkTopology`
   (`rank % tp`) — keep it the single source.
5. **Restore the service before Phase 6.** It is stopped with containers
   retained; serialize with the integration coordinator and keep the no-recreate
   constraint. The configuration agent owns `build.sh`/`wip.sh`/phase-0/helpers,
   which are environment-only and not yet the native CLI.
6. **Six-Spark configs are now a qualification requirement, not just support**
   (objective revision 4, §3.7): deliver the single-Spark TP3 kernel
   measurement, provision `rhea`/`moa` (devtools, images, pinned checkpoint, RDMA
   provider selection), then run cases (2)–(4) after case (1), with the
   separate-version host-count-identity runner and the matched-topology
   comparison (G16–G20). The 97.565 GB single-RTX TP3 model needs measurement;
   TP2×EP3-faster is a hypothesis, not a claim.
7. **Never build from `/mnt/scratch`** (read-only NTFS bug): use
   `~/.cache/ds41rt/builds/<task>` and run
   `scripts/assert-build-filesystem.py` first; the replacement container is
   `ds41rt-tpep-nvme-dev` and the daemon target is the fresh NVMe path (R18).
8. **Do not treat reported CPU counts or the RTX1 reducer selftest as
   independently verified serving qualification.** The owner field at bits 9..11
   is a **binary index**, not a one-hot mask, and is not a capacity limit.
8. **Native roles 5/6 are not FFI interfaces 8/9**; keep the two numbering
   spaces distinct in manifests, tests and logs.

## Appendix A — derivation worked examples

- Per-expert: `3 · (2304·5120/2 + 2304·5120/32) = 18,800,640 B`.
- TP2 per rank per layer: `18,800,640/2 · 384 = 3,609,722,880 B`.
- TP3 per rank per layer: `18,800,640/3 · 384 = 2,406,481,920 B`.
- TP4 per rank per layer raw: `18,800,640/4 · 384 = 1,804,861,440 B`; padded
  `× 640/576 = 2,005,401,600 B`.
- All layers, full set: `18,800,640 · 384 · 40 = 288,777,830,400 B`.

## Appendix B — inspection references

- `AGENT_DEV_HINTS.md`, `DEVELOPER.md`, `architecture.md`, `docs/ENGINEERING.md`
- `ds41rt.config:5-11,19-22,60,73-86`; `run.sh:95-110,134-184,238,281-345`
- `scripts/release-common.sh:104,149,253-262,341-390`
- `rust/crates/ds41rt-core/src/constants.rs:10-11`
- `rust/crates/ds41rt-loader/src/expert_format.rs:289-356`
- `rust/crates/ds41rt-loader/src/v41_config.rs`, `official-v41-config.json`
- `rust/crates/ds41rt-loader/src/v41_catalog.rs:1100,1118`
- **Live path:** `rust/crates/ds41rt-daemon/src/main.rs:79`;
  `v41_native_serve/{scheduler,memory,speculative,distributed}.rs`;
  `v41_experts/coordinator.rs:119-120,368-388`, `v41_experts/service.rs:133-152`,
  `v41_experts/execution.rs`, `v41_experts/tp2*`;
  `v41_backbone_execution/distributed.rs`
- **Live transport:** `rust/crates/ds41rt-transport/src/v41_expert.rs:272,299,352`,
  `v41_expert/roce.rs:91,118`, `v41_expert/chunks.rs:125`
- **Legacy (different command; out of scope unless traced reachable):**
  `rust/crates/ds41rt-daemon/src/commands/real_full/intermediate_sharding.rs:136-145,301-302,452-453`;
  `.../coordinator_kernels/target_attention.rs:56,127,5660`;
  `.../scheduler/execution/progression.rs:284-397`
- `native/include/ds41rt_v41_experts.h:73-110`
- `docs/ds41-expert-tp-efficiency.md`, `docs/ds41-expert-grouped-slices.md`,
  `docs/ds41-native-route-reduction-qualification.md`, `docs/release-v6-plan.md`,
  `docs/release-v7-plan.md`, `docs/release-v8-notes.md`
- `third_party/sparkinfer/tests/moe/test_ep_moe.py`, `test_ep_moe_api.py`
- `python/tools/qualify_v41_slice_native.py`, `qualify_v41_expert_slices.py`,
  `qualify_v41_tp2_numerics.py`, `qualify_v41_tp2_reduction.py`,
  `export_b12x_v41_experts_aot.py`
