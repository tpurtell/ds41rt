# DS41RT example configurations

These files are **opt-in, standalone `--config` files**. They overlay the
launcher defaults in `scripts/release-common.sh`; every key they do not name
keeps its default. They are never selected automatically and they do not change
`ds41rt.config`, the release images or the default serving selection.

Select one explicitly:

```bash
./run.sh --config examples/configs/tp2ep2-native.config --dry-run
```

## Topology keys

| Key | Meaning |
| --- | --- |
| `SPARK_COUNT` | physical Spark ranks = `SPARK_TP * SPARK_EP` |
| `SPARK_TP` | tensor-parallel degree inside one replicated expert group |
| `SPARK_EP` | number of replicated expert groups (each group holds every expert) |

`SPARK_TP` and `SPARK_EP` are optional and **all-or-none**. When both are
absent the launcher keeps the legacy geometry (`TP = SPARK_COUNT`, `EP = 1`) for
the legacy counts `0`/`2`/`4`; count `3` with no keys is the **compact EXL3
TP3** layout (see below), never a native one — a native three-rank launch must
name both keys (`SPARK_TP=3 SPARK_EP=1`). A six-rank configuration has **no**
legacy geometry: `SPARK_COUNT=6` requires both keys explicitly and is rejected
without them. The rank map is group-major:

```
group   = global_rank / SPARK_TP
tp_rank = global_rank % SPARK_TP
```

Approved native official layouts:

| File | Hardware | Topology | Spark role | Status |
| --- | --- | --- | --- | --- |
| `tp4ep1-explicit-native.config` | 2 RTX + 4 Spark | TP4 x EP1 = 4 | none (legacy shard) | explicit form of the default; control arm |
| `tp2ep2-native.config` | 2 RTX + 4 Spark | TP2 x EP2 = 4 | `tp2` | experiment completed; quality not accepted; release not qualified |
| `tp3ep1-native.config` | 1 RTX + 3 Spark | TP3 x EP1 = 3 | `tp3` | v10 candidate; 5 RTX-local / 35 remote via the explicit-topology placement handoff; daemon support shipped since v9; this layout never qualified — see below |
| `tp3ep2-native.config` | 1 RTX + 6 Spark | TP3 x EP2 = 6 | `tp3` | experiment completed (all 40 remote); quality not accepted; release not qualified |
| `tp2ep3-native.config` | 2 RTX + 6 Spark | TP2 x EP3 = 6 | `tp2` | experiment completed; quality not accepted; release not qualified |
| `tp6ep1-native.config` | 2 RTX + 6 Spark | TP6 x EP1 = 6 | `tp6` | packaged; bounded final-image functional checks passed on 1 and 2 RTX; canonical six-rank qualifier not run |

All six native configurations have launcher support, including the dual-TP3
arms; the five six-rank/four-rank configurations actually run are recorded
under `runs/tp-ep-preflight/`.
These example files are **illustrative launch configurations, not the measured
artifacts**. See [docs/tp-ep-configuration.md](../../docs/tp-ep-configuration.md)
for the current results and the launcher limits (entry points differ; the
candidate launcher requires explicit topology and is single-rail A-only). No
release-support promise is made.

## v10 candidate profiles

`tp3ep1-native.config` (explicit native `TP3xEP1`, one unreplicated group of
three ranks) and `exl3-compact-tp3.config` (implicit compact EXL3 TP3:
`SPARK_COUNT=3` with **no** `SPARK_TP`/`SPARK_EP` keys, the checkpoint-native
`wrldsuksgo2mars/DeepSeek-V4.1-EXL3-K3.25-v1` mixed-projection staged-trellis
checkpoint — bit family `[3,4]`, launcher tag `k34` — on one RTX under the hard
32 GiB ceiling, non-paired disjoint packages, KV 2 GiB and prefill 256) are
**v10 candidate profiles**. They pin the not-yet-published `v10` pair, so
until the v10 release promotion they are deliberately exempt from the
published-pair equality below — their own test requires both names to be a
coherent `ghcr.io/tpurtell/...:v10` pair. Packaging is not qualification:
neither file carries a memory, correctness, performance or readiness claim.

The native profile pins `RTX_EXPERT_LAYERS=5`: the RTX holds the first five
routed layers and each Spark starts at layer 5 to hold the other 35. Because
that is an explicit local count on an explicit topology, the launcher takes the
single-RTX placement handoff (the coordinator publishes the boundary, workers
start at `--first-layer 5`) instead of the all-remote no-handoff shape. The
5-local / 35-remote single-RTX placement has prior TP6xEP1 functional evidence
on the final v9 images and matched the measured v8 baseline dispatch; that is
TP6 placement evidence, not TP3 qualification. Remote TP3 weight is
`35 * 2,406,481,920 = 84,226,867,200 B`, leaving `24,892,452,864 B` under the
`109,119,320,064 B` floor — admission arithmetic only, not a runtime fit.

The `v10` pair is built with the explicit build config `ds41rt.build-v10.config`
(`./build.sh --config ds41rt.build-v10.config`), which is `ds41rt.config` with
**only** the release image pair retagged to `v10`. It is a build-time target
override, not a runtime change: the runtime default `ds41rt.config` still names
the published `v9` pair and keeps serving it until promotion. A plain
`./build.sh` would otherwise derive the `v9` tag from `ds41rt.config` and
overwrite the live release.

Explicit topologies require the official native checkpoint. EXL3 and NVFP4
checkpoints are rejected before any service change; their existing non-topology
paths are unchanged.

## Images and roles

Every example other than the two v10 candidate profiles above names the **same
published release pair** that `ds41rt.config` names; the candidates pin the
coherent `v10` pair and are re-added to published-pair equality at release
promotion. The Spark image is universal: it carries the default TP4 shard plus
the `tp2`, `tp3` and `tp6` expert TP roles and advertises them as
`io.ds41rt.v41.spark_tp_roles=tp2;tp3;tp6` (see
[docs/release-v9-notes.md](../../docs/release-v9-notes.md)). One published pair
therefore serves every approved topology, and `run.sh` is what selects the mode:
it derives the needed role from `SPARK_TP`, requires that role in the image label
of **every** rank, and refuses before any service is stopped or replaced. A
topology that needs a role the image lacks is rejected even though the tag
resolves; the legacy default TP4 path never probes the label.

Do not pin a per-topology tag such as `ds41rt-coordinator:tp2ep3-candidate`.
`build.sh` takes the tag from the config it is given, so a name like that exists
only on a host where that exact file was built, and `run.sh` fails its image
check on every other host — the launcher cannot infer a tag from a topology.

Packaging is not qualification. The role check is a compatibility preflight: it
proves the image carries the shard family this topology asks for, nothing more.
Neither it nor a passing dry-run accepts the layout on quality, memory or
throughput; the per-file status above and
[docs/tp-ep-configuration.md](../../docs/tp-ep-configuration.md) carry that line.

To run a six-rank example, the Spark image must exist on all six ranks: a default
`./build.sh` builds and distributes to the four active ranks, while building with
one of the six-rank `--config` files selects all six. Otherwise pull the Spark
image on `rhea` and `moa` too. The coordinator image is x86_64 and runs only on
the RTX host.

## Fifth and sixth Sparks

The six-Spark files use `rhea` (global rank 4) and `moa` (rank 5), connected
2026-09-20; see [docs/cluster-hosts.md](../../docs/cluster-hosts.md). Under the
current permission they are ordinary benchmark/development hosts and are **not**
restricted to a single six-Spark serving set. These examples name a secondary
rail (`LANE_B`) on an isolated subnet with matching host numbers, but the release
transport still dials `LANE_A` and the candidate launcher is single-rail A-only
by design; `LANE_B` is unused unless a launch explicitly selects it. The six-rank
experiments have completed but are **not accepted as quality results and not
release-qualified**; no memory, correctness, performance or readiness claim is
attached.

## Admission

`run.sh` prints a Spark *weight-only* admission line derived from the actual
resolved RTX/Spark boundary. It is **not** a launch-feasibility claim: workspace,
load staging and runtime headroom are only known after the expert service
reports them. Weight-only overflow is rejected before any service change. See
[docs/tp-ep-configuration.md](../../docs/tp-ep-configuration.md).

The TP×EP overlays set `SPARK_DEVICE_BUDGET_BYTES=109119320064`, the tested
fleet-wide floor (minimum of `MemTotal - 20 GiB` across the six hosts); the
`tp4ep1` control carries no override and keeps the 100 GiB default fallback.
