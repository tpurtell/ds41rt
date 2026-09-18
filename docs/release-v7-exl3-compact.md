# EXL3 K2 compact serving: one 32 GiB RTX budget and two Sparks

## Design

The compact profile uses **Spark TP2**: both workers own all 384 experts in
each remote layer, with intermediate width 1152 per rank. The coordinator
broadcasts each remote request to exactly two peers, sums their compact BF16
planes in FP32, adds the shared expert once, and rounds once to BF16.
Bottom-up resident RTX layers use the existing full-width (2304) role.

The alternative, disjoint whole-layer ownership on two Sparks, was rejected.
Although a full-width RTX role exists, it is an SM120/FP32 export; there was no
SM121/BF16 full-width Spark artifact to reuse. Either design needs new Spark
artifacts. Layer ownership would additionally need per-layer peer selection,
bounded worker layer ranges, an independent executor identity, and a new
single-plane completion/reduction path. TP2 preserves the established broadcast
protocol and uses the existing H128-aligned TP2 weight slicing.

The older paths retain their contracts: native MXFP4 and NVFP4 Spark experts
are TP4; EXL3 TP4 remains available, including explicitly paired packages;
two-RTX EXL3 may keep all 40 routed layers local with zero Spark processes.
TP2 executor IDs are 5–6, disjoint from TP4 IDs 1–4, so cross-topology stale
worker responses are rejected. These IDs validate topology/rank, not checkpoint
identity; use the release launcher to deploy matching checkpoints/artifacts.

## Release configuration

**Release image publication is still pending:** this change has been built and
exercised in WIP, not published as new release images. The release producer must
package the updated library, EXL3 bridges, and Spark TP2 artifacts before the
release commands below are usable. After publication, select the published
diffbot checkpoint and:

```ini
EXPERT_FORMAT=exl3
SPARK_COUNT=2
RTX_GPUS=1
MEMORY_RESERVATION=32GiB
KV_POOL_SIZE=2GiB
PREFILL_BATCH_TOKENS=256
DSPARK=on
```

`SPARK_0_HOST`/`SPARK_1_HOST` and their lane addresses name the two workers.
The release launcher validates the EXL3 checkpoint, non-paired TP2 package
variants, GPU count, and absolute memory ceiling before replacing services.
Launch with `./run.sh --dry-run`, then `./run.sh` (or `--restart` to replace an
existing deployment). `scripts/run-wip.sh` is the unrelated legacy protocol
launcher and does not implement this topology.

Two-peer `serve-native` also defaults to an absolute 32 GiB ceiling and a
2 GiB global KV pool. Compact prefill batches are capped at 256 tokens with an
explicit startup notice when reducing a larger requested value. The inherited
2048-token setting rounds workspace capacity up to 4096 and was measured above
42 GiB before KV/local experts, so it cannot fit this profile. A larger or
percentage memory ceiling is rejected. A GPU exposing
less than 32 GiB to CUDA receives the smaller physical ceiling, never a larger
budget. KV accounting includes its additional cache/window allocations, not
just the global source-pool bytes. Mandatory weights, both execution lanes,
vision/draft/snapshot allocations, local expert workspaces and staging, and
**2 GiB runtime headroom** all count toward placement. The ready log reports
actual occupied device bytes and the effective ceiling.

An explicit KV pool matters: the ordinary reservation-only policy fills the
available ceiling with KV before expert placement and would leave no local
expert layers. An explicit larger KV pool can reduce the local-layer count;
it cannot increase the total ceiling.

The coordinator logs `rtx_layers`, `first_remote_dispatch_layer`, and
`remote_dispatch_layers`. Each Spark separately logs its loaded first layer,
layer count, resident bytes, workspace bytes, and budget. The release single-RTX
launcher conservatively loads every layer on each Spark; a manually selected
`--first-layer` may trim the unused RTX-resident prefix. Dispatch and loaded
residency are therefore deliberately reported separately.

## WIP verification harness

With the repository's `runs/v7q-a1/serve-diffbot.sh` harness and a completed
`./wip.sh --slot v7q-a1 --role both` build:

```bash
SPARK_COUNT=2 COORD_CONTAINER=ds41rt-coordinator-wip-dual \
  runs/v7q-a1/serve-diffbot.sh start-coordinator 1 --dspark
# Read the coordinator's bottom-up placement log, then use its layer count:
SPARK_COUNT=2 runs/v7q-a1/serve-diffbot.sh start-experts <first-remote-layer>
```

No service should be restarted during another benchmark. Runtime correctness,
residency evidence, and campaign results must be recorded before qualification.
The 32 GiB simulation measures RTX PRO 6000 performance, **not RTX 5090
performance**.

## AOT packaging and RTX 5090 compatibility

The updated non-paired Spark package build generates `tp2-rank0` and `tp2-rank1`
(width 1152, BF16, 384 experts, top-k 6) at every capacity 1/16/80/256/1024/4096,
for both k23 and k34. Existing TP4 variants remain. Paired packages remain TP4
only and are rejected by the compact launcher.

The EXL3 bridge formerly pinned the export GPU's SM count. It now requires the
same compute capability and caps the cooperative launch grid to the smaller
of the export/device SM counts times the compiled blocks-per-SM limit. It keeps
**all export-sized allocations and embedded barrier offsets unchanged**.
Merely exporting with a synthetic `sms=170` argument would be unsafe: upstream
subkernels still embed the export host's SM-dependent offsets.

The release producer must rebuild/package and publish these new bridges and
Spark TP2 artifacts once; publication is not part of this WIP verification.
After that, RTX 5090 users are intended to use the same SM120 coordinator image
without an end-user export or rebuild. Physical RTX 5090 hardware remains
untested. Exact-object reduced-grid numerical
and CUDA-graph tests are in `python/tests/test_exl3_grid_numerics.py`; actual
RTX 5090 hardware performance is outside the memory simulation.

## Qualification status

Hardware smoke passed on the rebuilt compact profile: the requested 64-token
`6*7` chat request returned `42` with `finish_reason=stop`. The rebuilt two-RTX
zero-Spark profile also returned `42`, with all 40 TP2 layers resident locally.
The requested three-repeat decode campaign completed: weighted 88.10 tok/s,
code 123.54 tok/s, counting 163.55 tok/s. **29/30 checks passed**. The third
high-effort code-reasoning sample reached its 4096-token output limit while
reasoning about the nonce and returned no final code; raw evidence retains this
failure and the throughput table includes it. This is not a fully passing
quality campaign. No nonce, token limit, or failed sample was silently changed.
The full prefill campaign **passed all 30 cells** (three measured repeats plus
one warmup per cell, 120 samples total), covering retained contexts through
256K and suffixes through 32K. Best median prefill is **2015.45 tok/s**.
Across 1,241 two-second whole-device telemetry samples spanning decode and
prefill, peak RTX occupancy was **29,568 MiB**. Even adding the full reserved
2,048 MiB runtime headroom gives **31,616 MiB < 32,768 MiB**. No budget was
relaxed.

Measured compact startup (prefill capacity 256, 16 concurrent slots, 20 retained
entries, dSpark width 5, 2 GiB global KV pool):

| Owner | Bytes |
|---|---:|
| Fixed coordinator occupancy before KV/local experts | 24,594,612,224 |
| KV/cache allocation (including window state) | 2,191,668,736 |
| One full-width routed layer | 3,433,037,824 |
| Two local EXL3 execution workspaces | 177,724,136 |
| Observed coordinator occupancy at ready (allocator/runtime included) | 30,430,986,240 |
| Reserved runtime headroom | 2,147,483,648 |
| Total ceiling | 34,359,738,368 |

Thus ready occupancy plus headroom is **32,578,469,888 bytes = 30.34 GiB**,
below the 32 GiB limit. The split is layer 0 local, layers 1–39 on both Sparks
as TP2. Each Spark holds **67,394,076,672 bytes** of expert weights plus
**56,007,284 bytes** of execution workspace at capacity 256 (62.82 GiB combined,
excluding transport/CUDA context). The initial capacity-4096 worker smoke used
799,442,308 bytes of workspace; the final compact launcher matches its workers
to capacity 256.

Exact-object grid tests passed bit-for-bit at export-grid 188 SMs versus reduced
170-SM grids for all **18 K2 coordinator variants** (three roles × six
capacities), including poisoned intermediate buffers and changed-input CUDA
graph replays. This establishes reduced-grid invariance on the RTX PRO 6000,
not performance or occupancy qualification on physical 5090 hardware.
The native GPU reduction oracle passed both TP2 and TP4, including shared/no-
shared outputs, FP32 accumulation, and exact shared/output aliasing.
See the generated [EXL3 performance report](release-v7-exl3-k2-performance.md)
for measured cells; missing measurements must not be inferred. The committed
[qualification evidence](release-v7-exl3-compact-evidence.json) binds the topology,
raw campaign SHA-256 hashes, exact tested binary/package hashes, residency logs,
and all 18 reduced-grid test identities. Full raw campaign JSONs remain in
`~/.cache/ds41rt-v7-package/performance/`.
