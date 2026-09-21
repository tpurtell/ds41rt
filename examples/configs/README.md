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
the legacy counts `0`/`2`/`4`. A six-rank configuration has **no** legacy
geometry: `SPARK_COUNT=6` requires both keys explicitly and is rejected without
them. The rank map is group-major:

```
group   = global_rank / SPARK_TP
tp_rank = global_rank % SPARK_TP
```

Approved native official layouts:

| File | Hardware | Topology | Spark role | Status |
| --- | --- | --- | --- | --- |
| `tp4ep1-explicit-native.config` | 2 RTX + 4 Spark | TP4 x EP1 = 4 | none (legacy shard) | explicit form of the default; control arm |
| `tp2ep2-native.config` | 2 RTX + 4 Spark | TP2 x EP2 = 4 | `tp2` | experiment completed; quality not accepted; release not qualified |
| `tp3ep2-native.config` | 1 RTX + 6 Spark | TP3 x EP2 = 6 | `tp3` | experiment completed (all 40 remote); quality not accepted; release not qualified |
| `tp2ep3-native.config` | 2 RTX + 6 Spark | TP2 x EP3 = 6 | `tp2` | experiment completed; quality not accepted; release not qualified |
| `tp6ep1-native.config` | 2 RTX + 6 Spark | TP6 x EP1 = 6 | `tp6` | packaged; bounded final-image functional checks passed on 1 and 2 RTX; canonical six-rank qualifier not run |

All five TP×EP experiments have completed, including the dual-TP3 arms; the
five configurations actually run are recorded under `runs/tp-ep-preflight/`.
These example files are **illustrative launch configurations, not the measured
artifacts**. See [docs/tp-ep-configuration.md](../../docs/tp-ep-configuration.md)
for the current results and the launcher limits (entry points differ; the
candidate launcher requires explicit topology and is single-rail A-only). No
release-support promise is made.

Explicit topologies require the official native checkpoint. EXL3 and NVFP4
checkpoints are rejected before any service change; their existing non-topology
paths are unchanged.

## Images and roles

Every example names the **same published release pair** that `ds41rt.config`
names. The Spark image is universal: it carries the default TP4 shard plus the
`tp2`, `tp3` and `tp6` replicated-group roles and advertises them as
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
