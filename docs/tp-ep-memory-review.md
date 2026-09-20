# TP×EP memory-budget review: native Spark TP2/TP3 (read-only, provisional)

Status: **read-only review. No code, build, export, GPU workload or service
change was made.** One read-only host query (`/proc/meminfo` on the four
Sparks) and one integration-agent CUDA probe were consumed. Every number here is
either (a) arithmetic from the **current source** of the
exporter/packer/daemon/transport, (b) a read-only host/probe value with its
timing stated, or (c) an explicit placeholder that still needs a runtime report.

Owner: this file only (`docs/tp-ep-memory-review.md`). No other file was
modified.

Revision note (direct human budget update): `SPARK_DEVICE_BUDGET_BYTES`
(default 100 GiB) is **not a hard ceiling**; "slightly over 100 GB/spark is
fine" and "20 GB should be plenty for the Linux OS". The user's "GB" is
ambiguous; the parent chose **20 GiB conservative**. §4 fixes the budget
semantics, §8 gives the TP2 29/30/31 thresholds. The shipped default config is
**not** changed by this document; any candidate budget is an explicit,
user-approved experimental override.

## 0. Review snapshot and superseded findings

The daemon workstream `6ba87b75-6d77-4068-a8a6-8cf9ec278ad8` was **actively
editing `rust/crates/ds41rt-daemon/**` during the review** and now reports the
ring fix final (§3.4). Findings it fixed before this document was written are
marked resolved; live/E2E items remain marked pending.

| earlier gap | current state |
| --- | --- |
| daemon could not select Spark TP2/TP3 roles | `ExpertLayer::BackboneReplicatedTp { layer, rank, world }`; `role()` 5/6, `info()`/`kernel()` dispatch (`v41_experts.rs:33-40,62-104`) |
| `validate_world` rejected world 3 / rank ≥ 4 | `V41SparkTopology` + `validate_topology` (`service.rs:13-27,203-230`); `service/local.rs:24` accepts `world ∈ {2,3,4,6}` |
| role 5/6 got no decode/small scratch or compact output | `compact_output_role(role) = role ∈ {1,5,6}` (`execution.rs:18-20,105-112,151`) |
| `NativeTp4Wave::device_bytes` undercounted N=6 by 2 planes | `device_bytes_for(capacity, ranks)`; callers pass `args.peers.len()` (`coordinator.rs:228-261`, `v41_native_serve.rs:336-357`) |
| transport ring bytes "unquantified" | quantified in §3.1; daemon mirror audited in §3.2 (four gaps: A/B/C now PASS on the final `6ba87` patch, D conservative) and status recorded in §3.4 |

Authority rule: `dist/spark-expert/V41_EXPERT_AOT.json` and
`docs/ds41-expert-aot-qualification.json` are **not** authoritative for the new
geometry (historical role-1 artifacts, mutually inconsistent, §2.3). TP2/TP3
scratch figures are **preliminary formulas from the current exporter**, not
measured artifacts.

Notation: `I` = logical intermediate, `K` = kernel intermediate, `H` = 5120,
`C` = export capacity, `topk` = 6, `experts` = 384, `routes = C·topk`,
`planes = ceil(K/width)` (width 64 at `C=1`, 192 otherwise). Units are always
labelled: `GiB = 2^30`, `GB = 10^9`.

## 1. What the Spark admission actually checks today [provisional]

`service.rs:60-125` (`load_weights`), per-layer bytes from
`v41_experts.rs:225-296` (`ExpertWeights::layout`):

```
resident = Σ_layers ExpertWeights::plan(...).resident_bytes
staging  = max_layers ExpertWeights::plan(...).device_staging_bytes
(1) resident + staging                <= device_budget
workspace = ExpertWeights::plan_execution(selection(0), capacity).total()
(2) resident + workspace              <= device_budget
then per layer: load(..., remaining -= layer.resident_bytes)
```

`device_budget` is `SPARK_DEVICE_BUDGET_BYTES` (default `107,374,182,400` =
100 GiB = 107.374 GB). `ExpertLoadBudget` also carries `pinned_host_bytes` and
`read_scratch_bytes` (`v41_experts.rs:197-206`), allocated as 16
`cudaHostAlloc(Portable|Mapped)` buffers plus 16 heap scratch vectors per layer
(`v41_experts.rs:326-333`), but **neither check uses them** — nor the RoCE rings
of §3.1. Only the RTX-side `LocalLayerPlan` consumes the full budget
(`v41_native_serve/memory.rs:363-384`).

### 1.1 Per-layer resident and pack-staging bytes (source-derived)

| TP | `I`/`K` | packed per expert | resident/layer (×384) | device staging | pinned host (×16) | read scratch (×64×16) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 1152 | 9,400,320 | 3,609,722,880 (3.3618 GiB) | 9,400,320 | 150,405,120 (143.44 MiB) | 1,179,648 |
| 3 | 768 | 6,266,880 | 2,406,481,920 (2.2412 GiB) | 6,266,880 | 100,270,080 (95.62 MiB) | 1,179,648 |
| 4 (shipped) | 576/640 | 5,222,400 | 2,005,401,600 (1.8677 GiB) | 4,700,160 | 75,202,560 (71.72 MiB) | 1,179,648 |

Packed strides are `padded·H`, `padded·H/16`, `H·padded/2`, `H·padded/32` with
`padded = align_up(I,128)` (`native/cuda/kernels/v41_expert_pack.cu:52-64`);
TP2/TP3 are unpadded. Each 16-expert group is read into pinned bytes, copied H2D
into one device staging buffer, then packed straight into the resident plane with
no packer-internal scratch (`v41_experts.rs:299-397`, `v41_expert_pack.cu:68-104`).

## 2. Emitted scratch formulas (current exporter source, preliminary)

`export_b12x_v41_slices_aot.py:115-141,217-279`; `atomic = C >= 256` (ABI 3
FP32 token accumulation, else ABI 2 route planes), widths 64 at `C=1` else 192.

| role | m1 w64 | m16 w192 | m80 w192 | m256 | m1024 | m4096 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Spark TP2 (`K=1152`) | 2,352,160 | 13,925,776 | 69,598,096 | 7,738,912 | 30,932,512 | 123,706,912 |
| Spark TP3 (`K=768`) | 1,614,880 | 9,993,616 | 49,937,296 | 7,738,912 | 30,932,512 | 123,706,912 |
| Spark TP4 (`K=640`, historical) | 1,369,120 | 9,993,616 | 49,937,296 | 7,738,912 | 30,932,512 | 123,706,912 |

At ABI ≥ 3 the scratch is degree-independent; the TP width only affects the
1/16/80 buckets (TP2 `planes=6` vs TP3/TP4 `planes=4`). The shipped
`dist/spark-expert/V41_EXPERT_AOT.json` is lower by exactly `routes·5120·4` at
every ABI-2 capacity (122,880 / 1,966,080 / 9,830,400), and
`docs/ds41-expert-aot-qualification.json` matches neither — do not size TP2/TP3
from either.

### 2.1 Daemon workspace for roles 5/6 (capacity 4096)

| component | TP2 (role 5) | TP3 (role 6) |
| --- | ---: | ---: |
| `scratch_bytes` (m4096) | 123,706,912 | 123,706,912 |
| `decode_scratch_bytes` (m1) | 2,352,160 | 1,614,880 |
| `small_scratch_bytes` (m80) | 69,598,096 | 49,937,296 |
| `hidden = C·5280` | 21,626,880 | 21,626,880 |
| `routing = C·48` | 196,608 | 196,608 |
| `output_and_shared = C·10240` | 41,943,040 | 41,943,040 |
| `ExpertExecutionBudget::total()` | **259,423,696 (247.41 MiB)** | **239,025,616 (227.95 MiB)** |

## 3. Bytes the Spark admission ignores

### 3.1 RoCE registered/mapped rings (quantified)

The live Spark worker uses the persistent verbs-host RC QP path
(`service/local.rs:1-32,78-176`). Ring layout
(`ds41rt-transport/src/verbs.rs:3979-4030,4806-4930`):

```
req_cap  = clamp(DS41RT_VERBS_HOST_RING_SLOT_BYTES, req_wire(R), max_frame_bytes)
resp_cap = clamp(DS41RT_VERBS_HOST_RING_SLOT_BYTES, resp_wire(R), max_frame_bytes)
req_wire(R)  ≈ HDR + R·(5280 + 6·4 + 6·4)      # FP8 K32 hidden + ids + weights
resp_wire(R) = HDR + R·4 + R·max(negotiated, 5280, 10240)   # BF16 partial plane
req_span     = depth · align_up(req_cap, 4096)   # verbs-host preferred_alignment
resp_span    = depth · align_up(resp_cap, 4096)
transport/rank = lanes · (req_span + resp_span)  # lanes = 2 (decode + prefill)
```

Defaults: slot floor `8 MiB` (`DS41RT_VERBS_HOST_RING_SLOT_BYTES`), depth `8`
(`DS41RT_VERBS_HOST_RING_DEPTH`), `--max-frame-bytes = 64 MiB` (`cli.rs:174`),
so the slot floor alone is 64 MiB per ring. `native/src/ds41rt_native.cc:3152-3175`
allocates **both** spans per endpoint with `cudaHostAlloc(Portable|Mapped)`:
pinned, device-visible, unified LPDDR on GB10.

| negotiated `R` | req span | resp span | per endpoint | per Spark rank (2 lanes) |
| ---: | ---: | ---: | ---: | ---: |
| ≤512 | 64.0 MiB | 64.0 MiB | 128.0 MiB | 256.0 MiB |
| 1024 | 64.0 MiB | 80.1 MiB | 144.1 MiB | 288.1 MiB |
| 2048 (shipped prefill) | 82.6 MiB | 160.1 MiB | 242.8 MiB | **485.5 MiB** |
| 4096 (full capacity) | 165.3 MiB | 320.2 MiB | 485.5 MiB | **970.9 MiB** |

Endpoint count per Spark rank is **2** (one persistent session per execution
lane; `v41_native_serve.rs:336-357`). `R` is the row count of the request that
created the session; a larger request reconnects with larger rings
(`verbs.rs:1613-1629`), and the size never shrinks. A device plane is delivered
in one slot only when it fits (`execution.rs` `execute_mapped_request`), so the
ring is sized to the largest batch the lane has served. Also ignored: the
per-connection `ExpertProtocolV2FrameBuffer` heap (`verbs.rs:2543`) and the
coordinator-side matching rings (separate budget).

### 3.2 Post-implementation audit of the daemon ring admission

`service.rs:207-295` now mirrors this rule for the native compact-BF16 shape:
`spark_ring_bytes = endpoints · depth · (align_up(clamp(req_wire)) + align_up(clamp(resp_wire)))`,
with the protocol constants matching the transport (`REQUEST`/`RESPONSE_HEADER_LEN`
= 96, `ROW_DESCRIPTOR_LEN` = 40, `ROUTE_ENTRY_LEN` = 12; `wire_stats` is
`header + rows·40 + routes·12 + hidden`, so `req_wire = 96 + R·5392`; the client's
`verbs_host_expected_response_wire_bytes` takes the BF16 max, so
`resp_wire = 96 + R·10244`). `spark_admission_budget` (`service.rs:305-333`) forms
`load_peak = resident + staging + pinned + read + headroom` and
`serve_peak = resident + workspace + exchange + row_indices + rings + headroom`,
checks both against the budget and, fail-closed, against `cuda_memory_info()`
(`service.rs:112-158`). The `max(load, serve)` split is **valid**: `local.rs:36-40`
creates weights, execution workspace and exchange before the listener and the
rings, so nothing ring-shaped is live during a layer load, and the exchange is
not live during per-layer loads. Four gaps found:

**A - critical: `DS41RT_SPARK_RDMA_ENDPOINTS` is not a transport setting.**
It appears only in `service.rs:285` (and the daemon doc); the transport never
reads it. The live path opens exactly **2** persistent sessions per Spark rank
(decode + prefill; `v41_native_serve.rs:339,358`). At the shipped capacity 4096
(`rows = config.capacity`, `service.rs:243,288`) one endpoint is
8·(22,089,728 + 41,963,520) = 512,425,984 B (488.69 MiB); `endpoints=2` gives
1,024,851,968 B (977.38 MiB) but `endpoints=1` gives 512,425,984 B. Setting 1
therefore lets 30 TP2 layers **pass** admission (model serve peak 109,105,692,112 B
<= 109,119,320,064 B) while the two real endpoints need 109,618,118,096 B: the
model undercounts the rings by one endpoint, 512,425,984 B (488.69 MiB), which
is a **498,798,032 B (475.69 MiB) overcommit relative to the budget**. Fix: derive the endpoint
count from the lane topology (always 2 today) or fail closed when the env
differs. **Closed in the final patch:** `parse_rdma_endpoints`
(`service.rs:284-298`) bails unless the variable is unset or exactly `2`, so a
stale `ENDPOINTS=1` now fails startup instead of under-reserving.

**B - the Spark does not size its own rings (closed and verified in the final patch).**
The server validates and allocates the wire values the coordinator sent
(`verbs/local.rs:62-76` `from_wire`, `create_from_wire_bytes_mapped_on_device`);
`DS41RT_VERBS_HOST_RING_SLOT_BYTES` and `DEPTH` set on the Spark worker are read
only by the client path (`verbs.rs:4913-4925`). Pre-patch, a coordinator
advertising 64 MiB slots while the Spark admission assumed the 8 MiB default
would have pinned 8·64 MiB = 512 MiB per ring, i.e. **1,073,741,824 B (1 GiB) for
one endpoint's two rings - already above the planned 1,024,851,968 B (977.38 MiB)
allowance**; two endpoints (one per execution lane on the same peer) would be
2,147,483,648 B (2 GiB) per peer, but that state is unreachable now. With
`RingBudget` (`verbs/local.rs:140-176`) the **first** such accept is rejected:
`reserve(1,073,741,824)` exceeds the limit by 48,889,856 B (46.62 MiB) and other
peers are untouched. Depth > 8 and a frame-budget mismatch still fail closed at
accept (`local.rs:46-54`).

**C - the 16-connection cap is bounded in bytes by the credit (closed and
verified in the final patch).** `local.rs:106-112` pushes every accepted connection while
the vector is below 16, with no per-peer dedup; removal happens only after a poll
error (`:169-174`), so churn could leave old + new endpoints (`reset_connections`
runs on failure and between scheduler layouts: `coordinator.rs:1103`,
`v41_native_serve/scheduler/layout.rs`). Pre-patch the worst case at capacity
4096 was 16 × 512,425,984 = 8,198,815,744 B (7.636 GiB) versus 1,024,851,968 B
admitted. The shared `RingBudget` now caps the aggregate reserved across pending
bootstrap + live connections + reconnect at the planned 1,024,851,968 B, so the
16-connection cap no longer becomes unbounded memory. Residual: at full-capacity
negotiation the limit is exactly two endpoints, so a replacement accept fails
closed until the old connection is reaped - a liveness, not an overcommit,
property.

**D - conservatism that costs one layer (not a safety gap).** The admission sizes
the rings with `rows = config.capacity` (4096), while the transport negotiates
with the live request rows, which the shipped `PREFILL_BATCH_TOKENS=2048` caps
at 2048. Real 2048-row rings are 512,491,520 B (488.75 MiB), so the model
over-reserves 512,360,448 B (488.63 MiB) and rejects 30 TP2 layers that the real
traffic would fit (see §8).

**Teardown (final patch).** `accept_with_budget` stores the reservation as
the connection's last field, so the endpoint is destroyed before the credit
returns. Native `destroy_rdma_rc_endpoint` (`ds41rt_native.cc:474-527`) has no
early return and the normal path frees both `cudaHostAlloc` buffers and the
endpoint; but the `ibv_*` and `cudaFreeHost` return codes are ignored, so a
`cudaFreeHost` failure would leak. Phrase it as freed on the normal path, **not**
an unconditional guarantee once API errors are ignored.

### 3.3 Other uncharged transients

| item | bytes (C=4096) | note |
| --- | ---: | --- |
| pinned host staging | TP2 150,405,120 (143.44 MiB); TP3 100,270,080 (95.62 MiB) | `cudaHostAlloc` page-locked; load phase only |
| loader read scratch | 1,179,648 (1.125 MiB) | heap; load phase only |
| `HostExpertExchange` | 42,139,648 (40.19 MiB) | `execution.rs:561-575`; serve phase |
| `row_indices` | 16,384 | `service/local.rs:39`; serve phase |
| cudaMalloc granularity | **unmeasured estimate; not charged** | driver rounding per allocation |
| CUDA context + driver + module | **inside the OS reserve** | not an app-budget line |
| checkpoint page cache | up to `resident`, reclaimable | buffered reads + `FADV_WILLNEED` |

### 3.4 Final ring-fix status (agent-reported, source-reviewed)

Agent `6ba87` reports the patch final and tested: transport **184 pass / 0 failed
/ 3 ignored**, daemon **835 pass / 6 baseline-proven failures / 101 ignored**
(counts are the owning agent's report; they were not re-run here because this
review takes no build). The claims were verified in the final source:

- `RingBudget::reserve` (`local.rs:47-70`) is a CAS loop with `checked_add`
  (overflow rejected) and a strict `next <= limit` check;
  `RingReservation::drop` (`:88-92`) returns the credit exactly once.
- `validated_ring_geometry` (`local.rs:222-244`) runs `from_wire` on both rings
  before any charge; the unit test rejects a bad stride, a bad span, zero
  capacity and depth 9 without reserving (`local.rs:709-743`).
- `accept_with_budget` (`local.rs:144-180`) validates -> charges -> allocates, and
  stores the reservation as the connection's last field, so teardown precedes the
  credit return.
- Concurrency: `concurrent_reservations_never_exceed_the_limit` (`local.rs:745-778`)
  runs 8 threads requesting 16 bytes from a 64-byte limit behind a barrier and
  asserts peak <= limit, at least one rejection, and `used == 0` after release;
  genuine `checked_add` overflow is exercised with a `usize::MAX` limit holding
  1 byte then requesting `usize::MAX` (`local.rs:673-683`).

Status: **A/B/C PASS in normal operation; D conservative (capacity-sized rings);
live reconnect E2E pending.**

## 4. Budget semantics, measurement, and units

`SPARK_DEVICE_BUDGET_BYTES` is a **user-configured CUDA/device budget**, not the
physical RAM size and not a cgroup limit. The remote Spark container runs with
`--gpus all --network host --ipc host --ulimit memlock=-1:-1` and **no
`--memory` cap** (`run.sh` remote `docker run`), so the worker sees host
`/proc/meminfo` and can pin unlimited memory.

Read-only `/proc/meminfo` (2026-09-20) and the integration CUDA probe relayed by
the parent:

| quantity | bytes | GiB | GB |
| --- | ---: | ---: | ---: |
| Spark `MemTotal` (= CUDA total), dodo/ostrich | 130,594,156,544 | 121.6253 | 130.5942 |
| Spark `MemTotal`, emu/kiwi | 130,596,184,064 | 121.6272 | 130.5962 |
| `MemAvailable` at probe | 126,027,001,856 | 117.3742 | 126.0270 |
| CUDA free at probe | 122,619,883,520 | 114.2010 | 122.6199 |
| SwapTotal, all four | 17,179,865,088 | 16.0000 | 17.1799 |

**Timing caveat.** `MemAvailable` and CUDA free are point-in-time probes of a
specific device state, not a global fit: the CUDA-free value is ~7.4 GiB below
`MemTotal` because some allocations/context were already live at probe time, and
`MemAvailable` moves with page cache. Only `MemTotal` is a stable capacity and
is what the budget should be derived from.

Budget rule (human update, parent's choice of 20 GiB conservative):

| quantity | bytes | GiB | GB |
| --- | ---: | ---: | ---: |
| 20 GiB OS reserve (chosen) | 21,474,836,480 | 20.0000 | 21.4748 |
| 20 GB decimal reserve (user's literal "20 GB") | 20,000,000,000 | 18.6265 | 20.0000 |
| candidate app budget, `MemTotal − 20 GiB` | **109,119,320,064** | **101.6253** | **109.1193** |
| candidate app budget, `MemTotal − 20 GB` | 110,594,156,544 | 102.9988 | 110.5942 |
| old default budget (unchanged) | 107,374,182,400 | 100.0000 | 107.3742 |

The two reserve readings differ by **1,474,836,480 B = 1.3740 GiB = 1.4748 GB**.
`MemTotal − 20 GiB = 101.6253 GiB` is "slightly over 100 GB" as the user said,
and is **not** 108 GiB. Rules:

- `OS_reserve` is the **only** subtraction. The app footprint (resident weights +
  pinned + workspace + exchange + transport rings) must fit `MemTotal −
  OS_reserve`; no second 20 GiB is reserved inside the app budget.
- Driver/CUDA context is inside the OS reserve, not an app line.
- The candidate is an experimental override; the shipped default stays
  `107374182400`.

## 5. Lifetime / phase model (what overlaps what)

Order in `service/local.rs`: `load_weights` → `weights.execution(...)` →
`HostExpertExchange::new(...)` → listener → transport endpoints on accept.

| phase | live allocations | released at |
| --- | --- | --- |
| A (per-layer load) | resident weights so far + device staging + 16 pinned hosts + 16 read scratch | end of each `ExpertWeights::load` |
| B (serve) | all resident weights + execution workspace + `HostExpertExchange` + `row_indices` + 2 RoCE endpoints (req+resp spans) + frame buffers | worker exit |

`peak = max(A, B)`, not the sum. Pinned staging never overlaps the workspace,
the exchange or the rings. Server-side reconnect is the one transient: the old
connection lingers until the poll loop drops it (`service/local.rs:169-174`
`swap_remove`), so a reconnected rank can briefly hold old + new rings.

## 6. Consistency gaps that remain

1. **Launcher is weight-only; worker is not.** `run.sh:178-183` derives the
   minimum RTX boundary from
   `SPARK_DEVICE_BUDGET_BYTES / release_spark_layer_bytes(TP)` (no workspace, no
   pinned, no rings), while `service.rs` requires `resident + workspace`. Any
   budget revision must update the launcher arithmetic too, or the boundary and
   the worker admission disagree. `release_validate_spark_weight_admission`
   labels itself `workspace_accounted=no` (`release-common.sh:810-826`).
2. **No Spark-side runtime-headroom constant.** The OS reserve now covers
   driver/context; pinned + rings must still be inside the app footprint.
3. **Role/world/geometry coupling.** `ExpertWeights::layout` asserts
   `info.role == layer.role()` and
   `info.logical_intermediate == first.intermediate_size()`; role 5/6, loader
   `BackboneTp { world }` and native geometry must stay in lockstep.

## 7. RTX N-plane reservation across the two lanes, capacity 4096

`coordinator.rs:228-261`: `device_bytes_for(C,n) = C·(n·10,240 + 2·10,240)`.

| N | per lane | 2 lanes |
| ---: | ---: | ---: |
| 2 | 160.0 MiB | 0.312 GiB |
| 3 | 200.0 MiB | 0.391 GiB |
| 4 | 240.0 MiB | 0.469 GiB |
| 6 | 320.0 MiB | 0.625 GiB |

The N=6 undercount is fixed and both serve waves pass
`device_bytes_for(capacity, args.peers.len())`. The check is still
self-referential (the reservation is its own `available_bytes`); safety comes
from `cuda_memory_info()` sampling after both waves, vision, snapshots and draft
are live, charged into `PoolPlan`. Do not sum it as a fit bound without doubling
for two lanes.

## 8. TP2 remote-layer thresholds on the actual GB10

Post-implementation the daemon admission charges the rings of §3.2 itself
(`spark_ring_bytes`), i.e. it uses `rows = config.capacity = 4096` for both
directions. Per-rank peaks:

```
serve = L·3,609,722,880 + 259,423,696 (workspace) + 42,139,648 (exchange)
        + 16,384 (row indices) + rings + headroom
rings = 1,024,851,968 B (977.38 MiB)   # capacity-sized (4096), endpoints = 2
      =   512,491,520 B (488.75 MiB)   # if sized to the shipped 2048 prefill rows
load  = L·3,609,722,880 + 9,400,320 + 150,405,120 + 1,179,648   # load phase
```

Budget = `MemTotal − 20 GiB` = 109,119,320,064 B = 101.6253 GiB = 109.1193 GB
(the 20 GB decimal alternative is 110,594,156,544 B = 102.9988 GiB).

| L | model serve peak (rings@4096, admission) | 20 GiB | rings@2048 (live-sized) | 20 GiB |
| ---: | ---: | --- | ---: | --- |
| 29 | 106,008,395,216 B (98.7280 GiB) | fits, 3,110,924,848 B (2.897 GiB) slack | 105,496,034,768 B (98.2508 GiB) | fits, 3,623,285,296 B (3.375 GiB) |
| 30 | 109,618,118,096 B (102.0898 GiB) | **rejected** by 498,798,032 B (475.69 MiB) | 109,105,757,648 B (101.6127 GiB) | fits, 13,562,416 B (12.93 MiB) |
| 31 | 113,227,840,976 B (105.4516 GiB) | rejected; weights + non-ring overhead alone are 112,202,989,008 B (104.4970 GiB) | - | rejected |

The table is a **known-footprint model**, not a validated fit. Actual startup,
reconnect, CUDA-context and frame-buffer measurements are pending (§9).

Conclusions for this fleet (`MemTotal = 121.6253–121.6272 GiB`):

- **29 remote TP2 layers: fits the known model** under either ring sizing
  (2.897 GiB / 3,110,924,848 B slack capacity-sized, 3.375 GiB live-sized). This
  is model arithmetic, **not a qualified startup result**; startup, reconnect,
  context and frame-buffer measurement is pending.
- **30 remote TP2 layers: rejected by the current admission**, which sizes rings
  at capacity 4096 (serve peak 109,618,118,096 B exceeds the 109,119,320,064 B
  budget by 498,798,032 B = 475.69 MiB). The real 2048-row traffic would need only
  512,491,520 B of rings and would fit with 13,562,416 B (12.93 MiB); the maximum
  ring row count that still admits 30 is **2102**. So 30 is today a startup
  rejection, **not** a silent overcommit - but only while `endpoints=2`, the
  coordinator's ring settings match the model, and the server holds exactly two
  connections. `DS41RT_SPARK_RDMA_ENDPOINTS=1` (gap A), a coordinator-side
  slot/depth larger than the model (gap B), or reconnect churn (gap C) each turn
  the same configuration into an overcommit of up to 498,798,032 B (475.69 MiB)
  for A, 1,122,631,680 B (1.045 GiB) for B, and 7.636 GiB of endpoints for C.
- **31 remote TP2 layers: does not fit under the declared 20 GiB or 20 GB OS
  reserve regardless of rings, but this is a reserve-policy outcome, not a
  hardware limit.** Weights plus non-ring serve overhead alone are
  112,202,989,008 B = 104.4970 GiB, above the 101.6253 GiB budget but below the
  121.6253 GiB MemTotal. Without any OS reserve, 31 fits with 18,391,167,536 B
  (17.130 GiB) of hardware headroom; the 20 GiB reserve instead needs
  `MemTotal ≥ 133,677,825,488 B = 124.4971 GiB` before any rings.
- **TP3: fits the known model.** 40 layers need 96,259,276,800 + 281,181,648
  (workspace + exchange + row indices) + 1,024,851,968 (capacity-sized rings) =
  97,565,310,416 B = 90.8652 GiB, leaving 11,554,009,648 B (10.760 GiB) under
  the 20 GiB reserve and 13,028,846,128 B (12.134 GiB) under 20 GB. Model
  arithmetic only; kernel/daemon wiring and startup measurement remain open.
- **Gap C is byte-bounded by the final credit**: reconnect churn can no
  longer hold 16 unbounded endpoints (pre-patch 7.636 GiB at capacity 4096); the
  aggregate `RingBudget` caps all reserved rings at the planned 1,024,851,968 B.
  The residual is liveness only - at full-capacity negotiation a replacement
  accept fails closed until the old connection is reaped - so an E2E 4096-row
  reconnect stress must show the old endpoint is eventually dropped.
- **Do not read the default as "no silent overcommit".** `endpoints=2` only
  describes the steady two-lane footprint. The final 6ba87 RAII aggregate
  credit is what closes A (obsolete env rejected), B (oversized wire spans
  rejected before allocation) and C (aggregate byte bound); any extra reconnect
  credit must **not** be invented - if it is added it belongs in `serve_peak` and
  in this memory review, charged like every other byte.

## 9. Pending GPU/measurement items (no claim made)

1. Regenerated `v41_spark_tp2_m*`/`v41_spark_tp3_m*` manifests: actual
   `info.scratch_bytes` per capacity, to confirm §2.
2. `cudaMemGetInfo` at worker readiness per rank with role-5/6 weights and
   execution live, replacing the footprint model with measured bytes. The probe
   numbers in §4 are a point-in-time snapshot, not a fit.
3. Per-layer load peak (pinned + scratch + staging) and readiness time.
4. Observed ring spans at accept from the
   `protocol_v2_verbs_persistent_server_connect` / `..._ring_budget` log lines at
   the real prefill row count, to confirm §3.1/§3.2 and the 30-layer margin.
5. **Gap A (PASS, normal operation)**: `parse_rdma_endpoints` rejects every stale
   value (`service.rs:284-298`); no launch path may export the env.
6. **Gap B (PASS, normal operation)**: the first oversized advertisement is
   rejected before any native allocation; the launch must still keep the
   coordinator/Spark ring envs equal.
7. **Gap C (PASS, normal operation)**: aggregate reservation returns to zero after
   all connections drop (unit-covered in §3.4); live connection counting remains
   desirable for the E2E run.
8. **Reconnect E2E stress (PENDING)**: with 4096-row negotiation, force a
   reconnect and prove the old endpoint is eventually reaped and the lane
   recovers. Fail-closed reconnect is acceptable correctness; it must still make
   progress under sustained churn.
9. **Gap D**: decide whether the admission sizes rings from `config.capacity`
   (conservative, rejects 30) or from the configured prefill row count (fits 30
   with 12.93 MiB). Any reconnect reserve added here must be charged in
   `serve_peak` and reflected in this document - no unbudgeted credit.
10. Operator decision: decimal 20 GB vs binary 20 GiB (the 30-layer answer depends
    on it; parent chose 20 GiB conservative).
11. `cudaMalloc` granularity and CUDA context baseline (unmeasured estimates; not
    charged in §8).

G1 (budget closure) remains `PENDING` for the measurement items 1-4. Gaps A-C are
**PASS in normal operation** on the final `6ba87` patch (RAII aggregate byte
credit reserved before any native pinned allocation, covering pending bootstrap
+ live connections + reconnect, endpoint count fixed at 2 rather than read from
the env), with the unit evidence in §3.4. **D is conservative** (rings sized at
`config.capacity` rather than the live prefill rows). **Live 4096-row reconnect
E2E is pending** and is the item that must show the old endpoint is eventually
reaped; integration agent `915` currently has refresh/stage-copy scope only, no
L3 build. No reconnect headroom may be added outside `serve_peak`. No number in
this document is a measured GPU result.
