# Customizable Spark routed-FFN TP with fully replicated expert groups (EP)

Status: read-only architecture/memory discovery audit for the requested topologies.
This file is the sole artifact written by this phase; no source, config, service or
default was modified. No GPU benchmark or build was run.

Requested targets:

| # | RTX | Sparks | Layout |
| --- | ---: | ---: | --- |
| a | 2 | 4 | routed-FFN TP2 x EP2 |
| b | 1 | 6 | routed-FFN TP3 x EP2 |
| c | 2 | 6 | routed-FFN TP2 x EP3 |

Constraints taken as given: only the official `deepseek-ai/DeepSeek-V4.1-Flash`
checkpoint (revision `dba1be0a40aa45a94ad051997016db3960a90277`), never NVFP4/EXL3;
existing TP4 remains supported; no default serving-config change and no release.

Evidence legend used throughout:

- **[measured]** live occupancy/serving log or checkpoint header fact.
- **[code]** read directly from source; path and line given.
- **[arith]** derived arithmetic from measured/code constants.
- **[assumption]** not decidable from the repository; stated as such.

---

## 0. Chosen contract (orchestrator decision)

The first implementation is fixed as follows and is what this report targets:

- **All physical rank planes return to the coordinator.** The coordinator sums
  every rank's BF16 partial plane in a fixed global-rank order in FP32 and applies
  **one** final BF16 rounding, then adds the shared-expert contribution **exactly
  once**. There is no Spark-Spark reduction and no intermediate BF16 rounding
  between groups.
- **Group identity:** `global_rank = 0 .. TP*EP-1`, `group = global_rank / TP`,
  `tp_rank = global_rank % TP`. Every group holds a complete replica of all 384
  routed experts, split along the intermediate dimension by TP.
- **Ownership metadata rides the existing route word.** A new native request flag
  (`1<<18`) carries an expert->group ownership bitmap encoded in the canonical
  top-6 route entries (same 12-byte route entry). The implemented representation is
  **one-hot**: `bit 9 + group`, exactly one bit set per active expert, expert ids in
  bits 0..8, up to three group bits 9..11. A 2-bit group *index* would also encode
  groups 0..2; one-hot is a representation choice (now implemented), not a
  necessity forced by bit width. A worker decodes the word, and for every route
  whose expert is not owned by its group substitutes the sentinel expert id 384
  with zero routing weight. The b12x grouped slice pipeline never owns an
  out-of-range id, maps it to `inverse = -1`, and `V41SliceReduce` emits zero, so
  the masked route contributes nothing and the token still carries six route slots.
- **Config:** explicit `SPARK_TP` in {2,3,4} and `SPARK_EP` in {1,2,3}, with
  `SPARK_COUNT = TP*EP`. Absent legacy fields infer today's behavior unchanged.
- **Whole-expert group assignment first; hot-expert splitting is a later
  measurement.** Static top-3 cannot express whole-expert global balancing.

Alternative B (group leaders reduce locally, coordinator reduces 2/3 group
partials) is explicitly rejected for the first implementation because it inserts a
second BF16 rounding boundary. It remains the bandwidth fallback to evaluate if
6-plane wire cost proves unacceptable (**future measurement**, not this phase).

---

## 1. Checkpoint and native storage facts

**[measured]** `config.json` of the pinned revision: 40 hidden layers, hidden 5120,
`moe_intermediate_size` 2304, `vocab_size` 129280, `n_routed_experts` 384,
`num_experts_per_tok` 6, `n_shared_experts` 1, `num_nextn_predict_layers` 3,
dSpark 128 experts / top-3, block 5, target layers 37/38/39, index head 128,
index heads 32, index topk 512. There is **no dense `intermediate_size` and no
`first_k_dense_replace`** anywhere in the text config — **all 40 layers are MoE**
(384 routed + 1 shared each). Do not invent a dense-layer schedule.
`quantization_config = {quant_method: fp8, weight_block_size: [32,32],
scale_fmt: ue8m0, expert_dtype: fp4}`.

**[measured]** A routed expert is `w1 [2304,2560] I8 + scale [2304,160] F8_E8M0`,
`w3` the same, `w2 [5120,1152] I8 + scale [5120,72]`. FP4 packs two values per byte
and E8M0 adds one scale byte per 32-element block, so the native byte factor is
`0.5 + 1/32 = 0.53125` B/weight.

```text
full expert payload          = 3 * 5120 * 2304 * 0.53125        = 18,800,640 B
full layer (384 experts)     = 384 * 18,800,640                 = 7,219,445,760 B (6.7236 GiB)
all 40 layers                = 40 * 7,219,445,760               = 288,777,830,400 B (268.945 GiB)
```

Per-rank native checkpoint share after the intermediate split is `full/world`; the
TP4 share is the documented `72,194,457,600 B`.

**[code]** The native packer `native/cuda/kernels/v41_expert_pack.cu:52-62` accepts
only `intermediate in {576, 1152, 2304}` and pads
`padded = (intermediate + 127) / 128 * 128`, producing four per-expert strides
`[padded*hidden, padded*hidden/16, hidden*padded/2, hidden*padded/32]`.

| TP width | logical I/rank | kernel I/rank | packed bytes/expert | packed bytes/layer (384) |
| ---: | ---: | ---: | ---: | ---: |
| 1 (full/RTX local) | 2304 | 2304 | 18,800,640 | 7,219,445,760 (6.7236 GiB) |
| 2 | 1152 | 1152 | 9,400,320 | 3,609,722,880 (3.3618 GiB) |
| 3 | 768 | 768 (no path today) | 6,266,880 | 2,406,481,920 (2.2412 GiB) |
| 4 (Spark) | 576 | 640 | 5,222,400 | 2,005,401,600 (1.8677 GiB) |

**[code]** `rust/crates/ds41rt-ffi/src/v41_experts.rs:479-491` is the authoritative
native ABI geometry table: role 1 Spark TP4 `(384, 576, 640, 6)`, role 2 RTX local
`(384, 2304, 2304, 6)`, role 3 TP2 `(384, 1152, 1152, 6)`, role 0/4 dSpark. **There
is no 768 / TP3 role, symbol, AOT package or selector at the audited HEAD.**
The parallel kernel discovery (`docs/tp-ep-kernel-audit.md`) confirms 768 is
alignment-safe (768 = 6x128, padding-free at both storage and compute), that the
packer accept-list `{576,1152,2304}` must grow to include 768, and that a Spark
TP2/TP3 export is a new geometry set; implementation agents have begun adding
TP2/TP3 Spark artifacts in parallel. Kernel geometry, export and microbenchmark
details are owned by that document, not repeated here.

**[measured]** Native TP4 residency per Spark is `80,216,064,000 B` (74.707 GiB);
padding above the checkpoint share is `8,021,606,400 B` (`+1/9`, `+11.11%`).
**[code]** `Service` admission uses those same packer strides, so the padding is
already inside the 80.2 GB figure.

**Audit note (arithmetic correction).** A prior draft computed the RTX
`BackboneFull` (2304) layer as `4x` the Spark padded layer (`8,021,606,400 B`).
That is wrong because `4*640 = 2560 != 2304`; the full-width layer has no padding.
The measured single-RTX log proves the correct value:
`docs/phase1-async-counting.json:522` reports
`bottom-up RTX expert placement layers=5 resident_bytes=36097228800`, and
`5 * 7,219,445,760 = 36,097,228,800`. Use **7,219,445,760 B (6.7236 GiB)** per
single-RTX `BackboneFull` layer.

**[code]** `docs/ds41-checkpoint-inventory.md:11` labels `18,750,160,200` as
"coordinator excluding dSpark"; the correct reading (architecture audit) is
coordinator **including** dSpark (`7,932,874,632`). Coordinator non-dSpark is
`10,817,285,568 B`.

---

## 2. Memory budgets per topology

The Spark budget is `SPARK_DEVICE_BUDGET_BYTES = 107,374,182,400` (100 GiB)
**[code]** `scripts/release-common.sh:145`, `ds41rt.config`. Spark admission is
**[code]** `rust/crates/ds41rt-daemon/src/v41_experts/service.rs:41-118`:

```text
resident(layers first_layer..40) + max_layer(device_staging_bytes) <= device_budget
resident(layers first_layer..40) + execution_workspace(capacity)  <= device_budget
```

`first_layer` is supplied externally (the coordinator's published boundary); there
is no Spark-side truncating planner — overflow is a hard error. A Spark therefore
holds `40 - b` layers where `b` is the coordinator's local RTX expert boundary.

### 2.1 Per-Spark weight-only maximum layers

Let `W` = execution workspace + staging + safety (GiB). Weight-only fitting:

| TP | bytes/layer | all-40 GiB | layers fitting `(100-W)` |
| ---: | ---: | ---: | --- |
| 2 | 3,609,722,880 | 134.473 | 29 @W=0, 28 @W=4, 27 @W=6..8, 26 @W=10 |
| 3 | 2,406,481,920 | 89.648 | 40 @W<=10.35; 44 by weight alone |
| 4 (current) | 2,005,401,600 | 74.707 | 40, with 25.3 GiB of slack |

**Consequence.** EP replication does **not** multiply per-rank memory; the TP width
does. A fully replicated group stores the complete expert set at its TP width, so
TP2 needs a layer split and TP3 does not (within the current budget).

> Naming caution: "TP2 can keep ~11 layers on the Sparks" that appeared in a
> parallel note is the RTX **minimum**, not the Spark maximum. The Spark maxima are
> **29** (TP2) and **40** (TP3); the corresponding RTX minima are 11 and 0.

**Workspace (TP4 only measured; TP2/TP3 pending an emitted artifact).** The shipped
TP4 Spark AOT (`dist/spark-expert/V41_EXPERT_AOT.json`, capacity 4096) reports
`info.scratch_bytes = 123,706,912 B` (118 MiB); `plan_execution` adds decode scratch
(cap 1, 1,246,240 B), small scratch (cap 80, 40,106,896 B), hidden =
`capacity * input_row_bytes` (21.6-41.9 MB), routing (196,608 B) and the
output/shared plane (41,943,040 B), for a TP4 workspace of about 0.21-0.23 GiB plus
~4.7 MB of reused device staging. **Do not project that number onto TP2/TP3.** For
TP2/TP3 the scratch, per-capacity variant set, staging and graph policy must come
from an emitted manifest and the actual expert service at startup; only a candidate
estimate is available now (exporter formula scales the slice planes with the wider
per-rank intermediate). Until that artifact exists, the `W` allowance in the fit
table is a placeholder and the budgets are **conditional**, not a fit proof. The
in-flight launcher's `release_validate_spark_weight_admission`
(`scripts/release-common.sh:813-826`) is weight-only and labels its output
`workspace_accounted=no`, which is the correct honest gate for this state.

### 2.2 RTX layer placement (dynamic, bottom-up)

**[code]** `rust/crates/ds41rt-daemon/src/v41_native_serve/memory.rs:353-384`
(`LocalLayerPlan::new`) selects the longest bottom-up prefix `layers 0,1,2,...`
with `cumulative_resident + workspace + max_staging <= available`, where
`available = ceiling - currently_occupied - RUNTIME_HEADROOM`. There is **no fixed
layer-20 boundary**; 20 is only the historical fallback minimum. The boundary is a
live function of the KV reservation, dSpark/workspace allocations and headroom, so
**the modern default dual boundary is not known from this audit** and must be read
from the runtime `dual RTX bottom-up expert placement expert_layers=` line.

**[measured]** era-specific anchors (each tied to a stated config, not a default):

| condition | RTX local layers | per-layer evidence |
| --- | ---: | --- |
| single RTX native, qualified default cache | 5 | `docs/phase1-async-counting.json:522`: `layers=5 resident_bytes=36097228800` |
| dual RTX native, historical release-candidate config | 20 | `docs/release-v6-plan.md:1427`; `phase2-memory-breakdown.md:24` measured 67.266 GiB/card at that config |
| dual RTX, 80 GiB ceiling, no KV cap | 8 | `docs/release-v6-plan.md` live placement table |
| dual RTX, 80 GiB ceiling, 13.09 GB KV | 15 | same |
| dual RTX, 0 Sparks (all local) | 40 | `runs/v7q-a1/dual-regression.log:23` |

**[measured]** with the historical 20-layer dual-RTX TP2 config the machine is at
`94.146 / 94.032 GiB` occupied with `0.824 / 0.935 GiB` unallocated and
800 MiB/GPU headroom (`docs/phase2-memory-breakdown.md:22-42`); single-RTX runtime
headroom is 2 GiB, dual 800 MiB, and the dual expert setup headroom is 64 MiB
(`EXPERT_SETUP_HEADROOM`, `docs/release-v7-published-memory-evidence.log:35`). This
is a measured point at a configured boundary, not a claim about today's default.

> Caveat **[measured, era]**: the dual-RTX rows in `phase2-memory-breakdown.md`
> are self-described "intermediate configuration, not release qualification"; the
> qualified owner/arena table is `docs/release-v1-memory.json` (1 RTX). The
> default dual KV pool also differs by era: 13,094,420,480 B in
> `phase2-adaptive-verification.md` vs the older 14,960,885,760 B table.

### 2.3 Per-topology fit (weight-only; conditional on measured workspace)

`b` = RTX local layers, `N = TP*EP` return planes. The table is arithmetic from the
packer, not a fit proof: the TP2/TP3 workspace is an unmeasured placeholder and the
modern default `b` is unknown.

| topology | RTX | TP x EP | per-Spark layer | weight-only condition | status |
| --- | ---: | --- | ---: | --- | --- |
| a | 2 | 2 x 2 | 3.3618 GiB | fits if `b >= 11..14` (depending on `W`) | conditional: default `b` unmeasured |
| b | 1 | 3 x 2 | 2.2412 GiB | weight-only fits at `b = 0` if `W <= 10.35` | conditional: TP3 workspace pending; no TP3 kernel yet |
| c | 2 | 2 x 3 | 3.3618 GiB | same as (a) | conditional: default `b` unmeasured |

**Verdicts (weight-only arithmetic, not a launch-feasibility proof):**

- **(a) TP2 x EP2 on 4 Sparks** needs the RTX to hold at least ~11-14 routed layers
  so that 29 or fewer TP2 layers remain per Spark. The historical release-candidate
  config chose 20, which would satisfy this, but the modern default boundary must be
  read from the runtime placement log rather than assumed.
- **(b) TP3 x EP2 on 6 Sparks** is weight-only feasible even with `b = 0`
  (89.65 GiB, ~10.35 GiB of workspace slack); the measured single-RTX `b = 5` would
  leave 35 layers at 78.44 GiB. Both are conditional on the TP3 workspace and on the
  TP3 kernel existing.
- **(c) TP2 x EP3 on 6 Sparks** has the same weight-only condition as (a): per-rank
  memory depends on TP, not EP, so the third group costs no additional per-rank
  weight bytes.
- **EP is a load-balancing and TP-width choice, not a wire or reduction saving.**
  Under the chosen contract every physical rank plane returns over the same wire and
  the coordinator performs the same ordered FP32 sum with one final BF16 rounding, so
  EP reduces neither wire volume nor reduction work. Its purpose is to assign each
  expert to exactly one replicated group (load balance) and to allow a narrower TP
  width; the isolated diagnostic (`docs/ds41-expert-tp-efficiency.md`) shows better
  kernel scaling at TP2 than TP4, but that is a component result, not an end-to-end
  serving claim.

### 2.4 Coordinator (RTX) side at N planes

**[code]** `NativeTp4Wave::device_bytes(capacity) = capacity * (4*10240 + 2*10240)`
(TP4 upper bound), and all rank planes live on the single transport GPU.

| N | plane bytes at capacity 4096 | two lanes |
| ---: | ---: | ---: |
| 4 | 4096 * 6 * 10240 = 251,658,240 B (240 MiB) | 480 MiB |
| 6 | 4096 * 8 * 10240 = 335,544,320 B (320 MiB) | 640 MiB |

The N=6 delta is `+160 MiB` total. This is not a memory risk.

Wire volume per wave (R rows): egress `N*(5280+72)*R`, ingress `N*10240*R`.
For `R = 2048`, ingress is ~84 MB at N=4 and ~126 MB at N=6. That +50% ingress is
the real cost of Option A and is the item reserved for future measurement.

---

## 3. Current execution trace

This section describes the pre-change baseline (HEAD `bec5fcc`) that the memory and
interface audits must respect. The in-flight implementation deltas are in sections 4
and 5 and in the parallel `docs/tp-ep-kernel-audit.md` /
`docs/tp-ep-implementation-plan.md`; where the working tree already implements an
item, that is stated in section 4/5 rather than here.

### 3.1 Config -> launch -> rank identity

**[code]** One parser only: `release_load_config()` in `scripts/release-common.sh`
(lines 113-289). It hard-codes the Spark count at parse time:
whitelist `SPARK_[0-3]_*` (line 104), defaults for indices 0..3 (156-160),
`SPARK_COUNT=4` (149), allow-list `0|2|4` (255-263). `run.sh` starts each Spark
with `expertd-native --rank $i --world $SPARK_COUNT` (run.sh:323-327) and passes
`--peers` built from the LANE_A list (run.sh:234-236, 283).

**[code]** Rust identity: CLI `rank 0..4`, `world 2..=4`
(`ds41rt-daemon/src/cli.rs:157-162`); `validate_world` requires `world in {2,4}`
and `world == 4 || exl3` (`v41_experts/service.rs:133-137`); the local service also
asserts `config.rank < 4` (`service/local.rs:14`). Each worker's wire identity is
`v41_spark_executor_id(world, rank)` (`service/local.rs:71`), defined at
`ds41rt-transport/src/v41_expert.rs:29-33`: TP4 -> ids `1..4`, TP2 -> ids `5..6`,
disjoint namespaces. There is no explicit rank handshake; the coordinator maps the
response `executor_id` by position in its expected list
(`v41_expert.rs:322-348`), and the RoCE/TCP receivers re-check
`rank == chunk.stream_id`.

**[code]** Two coordinator paths:
`v41_native_serve.rs::worker` (1 RTX) accepts `peers.len() in {2,4}` and maps to
executor namespaces at `:167-177`; `v41_native_serve/distributed.rs:353-357`
(2 RTX) hard-codes `V41Tp4Roce::new(args.peers.try_into() /* "four Spark peers
required" */, [1,2,3,4], ...)`. So today only Spark TP4 (or EXL3 TP2 on 1 RTX) is
reachable.

### 3.2 Load / shard

**[code]** Selection enum `V41ExpertSelection` (`ds41rt-loader/src/v41_expert_staging.rs:6-26`)
has `Backbone{layer,expert,rank}` (TP4, rank<4), `BackboneTp2{..rank}` (rank<2),
`BackboneFull`, `Dspark`, `DsparkTp2`. The canonical shard model is
`native_fp4_tp_projection` (`ds41rt-loader/src/expert_format.rs:289-356`):
`alignment = world*32`, `local = intermediate/world`,
`intermediate_start = local*rank`; W1/W3 split rows, W2 splits packed columns with
byte offset `intermediate_start/2` and E8M0 offset `intermediate_start/32`.
**[code]** `v41_catalog.rs:147-150` additionally hard-codes `rank < 4` and
`byte_length / 4`. Every Spark rank stores **all 384 experts** and only its
intermediate slice; there is no expert->rank ownership in the native path.

### 3.3 Coordinator routing and dispatch

**[code]** The lane builds one canonical top-6 request and sends the *same* request
to every rank: `v41_backbone_lane.rs:356-365` (`routed.expert_request(...)`,
`transport.prepare_remote_request(...)`, `transport.dispatch_ffn(...)`).
`prepare_remote_request` (`v41_experts/coordinator.rs:32-37`) is where the EXL3
paired ownership encoding is applied today, and is the natural hook for native
group-owner encoding. The route-word bit layout is already assumed by trace code:
`v41_backbone_lane.rs:385` reads `r.expert_id & 511`.

**[code]** `V41BackboneRequest::validate_owned` (`v41_expert.rs:144`) requires
six routes per row, `expert_id < 384`, distinct ids per row, finite non-negative
gate weights, and rejects any flag other than `DEBUG_CHECKSUM|V41_COMPACT_BF16`
(shared `validate_canonical`, `v41_expert.rs:36`). The paired path
(`validate_paired`, `v41_expert.rs:103`) strips its own flag and decodes the route
word before canonical validation — the exact template for a replicated-groups
validator. `copy_routes_into` is at `v41_expert.rs:173`.

### 3.4 Transport, wire layout and reduction

**[code]** Wire revision 3: `MAGIC=b"DS41RTE3"`, version 3 (`protocol_v2.rs:6-7`);
request/response headers 96 B (128 with debug SHA-256), row descriptor 40 B, route
entry 12 B (`row_index u32 | expert_id u32 | gate_weight f32`). The response
`executor_id` is a `u64` at header offset 80; there is no rank/group/world field.
Dtype code `6 = F32`; the native rank plane is dtype BF16, dim 5120, stride 10240.
Frame limit is 64 MiB, and chunk coverage requires contiguous ascending row indices,
no overlap/skip/reorder, and `more_chunks == (end < rows)` — an early or missing
final marker is rejected (`chunks.rs`).

**[code]** `V41Executors { Tp2([u64;2]), Tp4([u64;4]) }`,
`V41Tp4Planes { planes: [Option<&[u8]>; 4] }`, and
`V41Tp4ChunkReceiver { received:[u32;4], finished:[bool;4] }` are fixed at two or
four ranks (`v41_expert.rs:251-355`, `chunks.rs:105-176`). `V41Tp4Planes::complete`
and `planes()` iterate/unwrap all four slots regardless of the twin-case length, so
the non-chunked TP2 path is already a latent hazard and must not be reused for N=6.
`V41Tp4Roce::dispatch` (`roce.rs:105-130`) does call `from_owned_ranks` for any
world != 4, and the TCP fanout loops are count-generic, so the receive-side framing
is closer to N-rank than the fixed arrays suggest; `V41Tp4Roce::new`/`with_clients`
(`roce.rs:56`) and `V41Tp4Tcp` (`tcp.rs:10-50`) still require 2 or 4.

**[code]** Coordinator reduction `v41_experts/coordinator.rs:373-392` dispatches to
only two native entry points, `reduce_tp2` (2) and `reduce` (4), with
`world_size in {2,4}` enforced at `:120`. The native kernels
(`native/cuda/kernels/v41_route_reduce.cu`) are templated on rank count:

```text
reduce_compact<Ranks>(p0, p1, [p2, p3], shared, output):
    value = fp32(p0); value += fp32(p1);
    if Ranks == 4 { value += fp32(p2); value += fp32(p3); }
    if shared { value = fadd_rn(value, fp32(shared)); }
    output = float2bfloat16_rn(value);     // one rounding
```

The shared expert is added **once**, after the routed sum, and the per-rank BF16
partials are what cross the wire. Option A is therefore numerically the existing
TP4 contract with `N` planes instead of four: each rank's local partial is rounded
to BF16 once, the coordinator accumulates in FP32, and there is **one** final BF16
rounding per token element. No new hierarchical boundary is introduced. This is the
central correctness argument for Option A over Option B.

### 3.5 Spark kernel path, masking and graph lifetimes

**[code]** The worker consumes the wire request in
`v41_experts/execution.rs:619-738`: `request.copy_routes_into(&mut exchange.ids,
&mut exchange.routing)`, then `copy_h2d` of ids/routing, launch, then a compact
reducer (`Fp32Tokens` / `Bf16Routes` / `Fp32Routes`) into a BF16 token-major plane.
This host-side `ids`/`routing` exchange is where the group mask is applied: for each
route, if `(owners >> group) & 1 == 0`, write `ids = 384` and `routing = 0.0`. No
extra device read or D2H is required.

**[code]** `third_party/sparkinfer/b12x/moe/_shared/kernels/v41_route_plan.py`
documents and implements the sentinel contract: *"Invalid expert IDs produce
inverse=-1"* (line 6, pack kernel line 60/67 only matches `ids[pair] == expert` for
`expert in 0..experts`), and `V41SliceReduce` writes `Float32(0)` when
`inverse[route] < 0` (lines 161-165). The `V41RoutePlan` expert count is a
constructor argument defaulting to 384, so sentinel 384 is outside the matched
range.

**[measured]** The native b12x consumer honours out-of-range ids as zero:
`docs/ds41-expert-intermediate-split-probe.json` (b12x revision `6adffce7cec2`)
records, for both BF16 and pre-quantized FP8 K32 inputs at M1/M2/M6/M80, a graph
replay after `ids.fill_(-1)` that asserts `torch.count_nonzero(result) == 0`, plus
changed-input replay and stable allocation. The parallel kernel audit
(`docs/tp-ep-kernel-audit.md`, "Sentinel behaviour, both paths") verifies the same
for sentinel `384`: the shipped grouped slice pipeline's `pack` predicate
`ids[pair] == expert` for `expert < experts` means a sentinel is never owned, so
the group is skipped (no weight staging, no MMA), `inverse = -1`, and
`V41SliceReduce` writes zero; `third_party/sparkinfer/tests/moe/test_v41_expert_numerics.py:143-146`
exercises the all-invalid replay through the public binding. The shipped Spark AOT
uses the grouped `slices` implementation for all capacities
(`dist/spark-expert/V41_EXPERT_AOT.json`), so the mask is a real early skip, not a
zero-out. This closes the sentinel question the earlier draft left open; the
coordinator must still apply the mask before the kernel consumes the route ids.

**[code]** Graph lifetime on the Spark: `ExpertExecution` owns
`graph: Option<(*mut c_void, u32, bool)>` and its `Drop` destroys the graph before
releasing allocations (`execution.rs:510-525`); `execution_state(rows)` selects a
graph by live row count. A TP-width change changes captured intermediate extents,
so graphs must be re-captured and the AOT capacity set must cover the new width.

---

## 4. Target interfaces and changes (implemented status noted)

### 4.1 Config fields and launcher

New keys (absent legacy keys keep today's behavior):

```text
SPARK_TP=2|3|4        # ranks per replicated group
SPARK_EP=1|2|3        # fully replicated expert groups
SPARK_COUNT=TP*EP     # derived; must stay consistent with the host list
```

**[code]** The in-flight launcher already implements the keys: key whitelist and
defaults `scripts/release-common.sh:104,147-160`; count allow-list now `0|2|4|6`
(`:255-275`); `SPARK_TP in {2,3,4}`, `SPARK_EP in {1,2,3}` set together or omitted
together, `SPARK_TP*SPARK_EP == SPARK_COUNT`, native-only, EXL3/paired excluded
(`:387-460`); approved set `TP2EP2, TP3EP2, TP2EP3, TP4EP1`, with `RTX_GPUS=1`
allowed only for TP3EP2; host keys extended to `SPARK_[0-5]_*`; weight-only
admission `release_validate_spark_weight_admission` (`:813-826`) and
`release_spark_layer_bytes` (`:787-794`). `run.sh` passes `--world` and the
group-major rank map. Do not change `ds41rt.config` defaults.

`StartupPlacement` still hard-codes `rtx_gpus: 2`
(`v41_native_serve/placement.rs:55,86`); the 1-RTX topology needs a real value or a
separate plan contract.

### 4.2 Rank / group interface (implemented split)

The topology is now expressed by the implemented types rather than a new core type:

- **`ds41rt-core::replicated_expert_schedule`** owns the group *assignment*,
  TP-agnostic: `ReplicatedExpertScheduler`, `ReplicatedExpertGroupPlan`,
  `ReplicatedExpertGroupId`, `MAX_REPLICATED_EXPERT_GROUPS = 3`,
  `INACTIVE_REPLICATED_EXPERT_GROUP = 255`. It maps experts to group indices and
  reports per-group load; it does not know TP.
- **`ds41rt-transport::v41_expert::native_group`** owns the *topology and identity*:
  `V41SparkTopology` with `NATIVE_TP2_EP2`/`NATIVE_TP3_EP2`/`NATIVE_TP2_EP3`/
  `NATIVE_TP4_EP1`, `group(global_rank) = global_rank / TP`,
  `tp_rank(global_rank) = global_rank % TP`,
  `global_rank(group, tp_rank) = group*TP + tp_rank`, and
  `executor_id`/`executor_ids`/`rank_of_executor` in disjoint per-topology
  namespaces (so TP3xEP2 and TP2xEP3 cannot collide). It also owns the request flag
  and the owner route word.
- The loader's `V41ExpertSelection::BackboneTp { layer, expert, rank, world }`
  (native only, `world in {2,3,4}`) preserves all existing variants. Slice rule:
  `local = intermediate/world`, W1/W3 rows window, W2 packed-column window,
  alignment `world*32`; require `intermediate % (world*32) == 0` (2304 -> 1152/768/576
  all pass). Kernel padding stays owned by the packer, never the loader.
- Remaining daemon-side work: construct these types from `SPARK_TP`/`SPARK_EP`,
  widen `validate_world` (`service.rs:133-137`) to `world in {2,3,4,6}` with
  `world == tp*ep`, stop requiring EXL3 for native TP2/TP3, and replace the
  hard-coded `[1,2,3,4]` / `matches!(world, 2|4)` sites in the coordinator paths.

### 4.3 Wire changes (Option A)

**[code]** The wire is revision 3 (`MAGIC=b"DS41RTE3"`, `protocol_v2.rs:6-7`); there is
no v3 module. Request/response headers are 96 B (128 with debug checksum), the row
descriptor is 40 B, and the route entry stays **12 B**
(`row_index u32 | expert_id u32 | gate_weight f32`, `protocol_v2.rs:588-606`).
Output dtype `6 = F32` exists; the native partial plane is dtype BF16 with stride
10240. Response `executor_id` is at offset 80 and is validated only positionally
against the expected list; there is **no rank/group/world field**.

1. **Flag (implemented).** `V41_NATIVE_GROUP_REQUEST_FLAG = 1 << 18`
   (`v41_expert/native_group.rs`). Occupied request-relevant bits are 0, 1, 4-9,
   **12-15** (striped-collective part count, allowed for requests), 16 and 17
   (`protocol_v2.rs:17-33,474-539`); free request bits are 2, 3, 10, 18-31. The
   native-group flag must be added to the allowed-flag set in `validate_flags`, and
   exactly one of paired/native-group may be set.
2. **Route word (implemented).** `V41NativeOwnerRouteWord` is a **one-hot**
   owning-group bitmap at bits 9..11 (`bit 9 + group`, exactly one bit set), expert
   id in bits 0..8, reserved bits 12..31 zero, with
   `V41_MAX_NATIVE_GROUPS = 3`. A 2-bit group *index* would also encode groups 0..2;
   one-hot is the chosen representation (a single set-bit check expresses the
   exactly-one-owner invariant and it parallels the EXL3 owner word), not a
   bit-width requirement. The paired word keeps bits 9..10 and `WORD_MASK = 2047`;
   the request flag selects which decoder is legal.
3. **Validator/builder (implemented).** `V41NativeOwnerRouteWord::decode(word,
   group_count)` and `observe(...)` enforce batch-consistent ownership;
   `with_native_group_owners` builds the request, and the native parser rejects
   owner overflow, reserved bits, duplicate ids and bad weights. Remaining work is
   only to admit the flag through `V41BackboneRequest` validation and route it to
   this decoder (exactly one of paired/native-group set).
4. **Worker mask.** In the host exchange, per route: keep when
   `(owners >> group) & 1 == 1`, else `ids = 384`, `routing = 0.0`. The wire request
   stays canonical; the masked form is local and is not re-validated as canonical.
5. **Identity (implemented).** `V41SparkTopology` defines
   `NATIVE_TP2_EP2`/`NATIVE_TP3_EP2`/`NATIVE_TP2_EP3`/`NATIVE_TP4_EP1` with
   `global_rank = group*TP + tp_rank`, `group = global_rank/TP`,
   `tp_rank = global_rank%TP`, and `executor_id`/`executor_ids`/`rank_of_executor`
   in disjoint per-topology namespaces; tests assert TP3xEP2 and TP2xEP3 do not
   collide and unsupported topologies are rejected without dummy ranks. Remaining:
   construct the coordinator and Spark service from this topology and replace the
   hard-coded `[1,2,3,4]` / `matches!(world, 2|4)` sites.
6. **Transport N (partially implemented).** `V41Tp4ChunkReceiver` is now
   `received: [u32; 6]`, `finished: [bool; 6]` with `new_ranks`, `world_size`, and
   `received_rows_slice`; TCP fanout and `V41Tp4Roce::dispatch` (`from_owned_ranks`)
   are count-generic. Remaining: `V41Tp4Roce::new`/`with_clients` still require
   `matches!(len, 2|4)` (`roce.rs:56`), `V41Tp4Tcp` (`tcp.rs:10-50`) is fixed-4, and
   `V41Tp4Planes::complete()`/`planes()` must use the world length rather than four
   slots.
7. **Reducer (kernel + FFI done; coordinator dispatch pending).** The native
   `ds41rt_v41_reduce_compact_bf16_planes_async` with `reduce_compact<2|3|4|6>` and
   `valid_compact_planes`, and the FFI
   `V41CompactReducer::reduce_planes([*const u16; 6], ranks, ...)` accepting
   `ranks in {2,3,4,6}`, are in place. Remaining: `enqueue_reduce_planes`
   (`v41_experts/coordinator.rs:377`) still matches only `2|4` and must call
   `reduce_planes`; document the global-rank summation order and keep the shared
   contribution added exactly once. The native header documents the ordered-FP32 +
   single-BF16 contract (`native/include/ds41rt_v41_experts.h:100-110`); the old
   raw-route reducer has no ranks==3 path, which is irrelevant to the chosen
   compact-plane contract.
8. **Coordinator plane budget and constructor.** `NativeTp4Wave::device_bytes`
   still computes the TP4 upper bound and `NativeTp4Wave::new` still rejects a world
   other than 2 or 4; both must use `N = TP*EP`.

Note: the legacy `real_full` stack (`DS4_FLASH_TP=4`, hidden 4096/7168,
`sparse_block.rs`, `intermediate_sharding.rs`, `rdma_reduction.rs`) has its own
four-shard reduction arenas and a row-scoped striped collective (bits 12-15) that
distributes rows, not experts. The three target topologies must be built on the
native `v41_native_serve` path (hidden 5120), not the legacy `real_full` path; do
not extend the row-sharded collective for expert ownership.

### 4.4 Expert-aware scheduling strategy

Implement in a new `ds41rt-core/src/replicated_expert_schedule.rs` (owning agent
already assigned):

- Input: per-layer 384 routed experts, per-expert route count from the batch
  histogram (computed lane-locally, no D2H — the same pattern as
  `PairedAssignment::encode`, `v41_experts/paired.rs:102-134`), and per-expert
  resident weight bytes.
- Cost: `weight_bytes(expert) + ceil(rows(expert)/tile)*tile` in a caller-chosen
  common integer unit; assign experts to the least-loaded group with deterministic
  ties (LPT: descending cost, then ascending expert id, then ascending group id).
- Output: borrowed `owners: [u8; 384]` (one group bit per active expert, `255`
  inactive) plus per-group cost vector; no allocation on the success path; the
  scratch is lane-owned and reusable.
- Whole-expert baseline first. Hot-expert splitting (splitting a single expert's
  route tiles across groups) is a later measured change and must not be assumed.
- Integration: called from `transport.prepare_remote_request` (or the lane just
  before it) so the route word is encoded once, request-owned, and exactly
  reproducible for replay/rollback.

Scheduling invariants:

1. Every active expert is assigned to **exactly one** group per batch.
2. All TP ranks of a group receive identical ownership bytes for the batch.
3. Deterministic output for identical input (batch order, tie seed, geometry).
4. No starvation across the two independent execution lanes (the planner is
   lane-local and does not share mutable state).
5. Per-route contribution is counted exactly once across the union of returned
   planes.

### 4.5 Correctness invariants

- **Reduction precision.** Each rank rounds its local partial to BF16 once (wire
  format); the coordinator sums all N planes in FP32 in global-rank order, adds the
  shared expert once, and rounds to BF16 once. This must be asserted by an oracle
  test against the full-width expert reference.
- **Contribution exactly once.** For a route owned by group `g`, exactly the `TP`
  ranks of `g` contribute and every other rank contributes zero via the sentinel.
  A route's token output must equal the full-width expert output within tolerance.
- **Ownership batch consistency.** Within a request, all occurrences of an expert
  carry the same owner bitmap; conflicting ownership is rejected before mutation.
- **Canonical wire vs masked kernel.** Wire validation is canonical (`< 384`,
  distinct); the masked descriptor is local. The sentinel must never be written
  back to the wire or to committed history.
- **Admission before allocation.** Weight/workspace admission
  (`resident + staging`, `resident + workspace`) precedes payload reads on every
  Spark; the boundary handoff is acknowledged before dispatch.
- **Graph lifetime.** Captured graphs are destroyed before their captured
  allocations; a TP-width change invalidates and re-captures graphs and requires
  matching AOT capacities.
- **Determinism.** Summation order, LPT tie-breaking and route-word packing are
  fixed, so identical inputs replay bit-identically.

---

### 4.6 b12x / SparkInfer constraints (third_party/sparkinfer/AGENTS.md)

Any TP3 kernel work touches `third_party/sparkinfer`; that submodule has its own
rules and they are not optional:

- b12x owns the planner, the policy layer and the 576->640 padded-extent decision.
  Integrations supply metadata and capacity limits; do not duplicate b12x policy or
  apply an engine-side packing heuristic.
- Live token/row/expert/occupancy counts must stay runtime scalar launch arguments;
  compile/cache keys may include only static geometry, planned capacity, device
  identity and toolchain identity. Precompile capacities before graph capture.
- CUDA graph capture/replay, warmup, stable allocation and preplanned workspace are
  serving requirements.
- Correctness gates precede any performance claim; benchmark the real target path.
- Push the fork change before updating its pin and lock, and re-run
  `scripts/verify-sparkinfer-source.py`.

The native AOT export currently instantiates `experts=384, intermediate=576, topk=6`
for the Spark role (`python/tools/export_b12x_v41_experts_aot.py:273`), requires
`route_planner == "internal"`, and derives `kernel_intermediate` from b12x. A TP3 or
TP2 export is the same call with `intermediate_size = 2304/world`; the padded
extent must come from `moe._dynamic_kernel_intermediate_size`, never from Rust.

---

## 5. Independently implementable tasks (non-overlapping ownership)

### 5.0 In-flight state (former "collisions" now resolved)

The working tree already contains uncommitted parallel work (loader, scheduler,
config/launcher, transport, kernel agents). The kernel discovery
(`docs/tp-ep-kernel-audit.md`) and the sequencing/ownership plan
(`docs/tp-ep-implementation-plan.md`) are the authorities for kernel geometry and
task order. Two items that looked like blockers during discovery were partial edits
and are now resolved in the tree:

1. **Spark TP2/TP3 AOT role ids and FFI interface ids are distinct.** The native
   Spark roles are 5 (`spark_tp2`) and 6 (`spark_tp3`), while the FFI interfaces are
   **8/9** (`ds41rt_v41_spark_tp2_expert_info` / `..._spark_tp3_expert_info`);
   interfaces 5/6 remain the NVFP4 family and the geometry table now recognizes
   roles 5/6. The packer accepts `{576,768,1152,2304}`. No collision remains.
2. **The N-plane reducer is bound.** FFI
   `V41CompactReducer::reduce_planes([*const u16; 6], ranks, shared, output, rows,
   stream)` accepts `ranks in {2,3,4,6}` and dispatches
   `ds41rt_v41_reduce_compact_bf16_planes_async`. The only remaining coordinator
   work is to call it from `enqueue_reduce_planes` and size the wave for `N`.

Neither item changes the memory budget or the chosen contract.

### 5.1 Task ownership

| # | Task | Owned files (only these) | Depends on | Tests |
| --- | --- | --- | --- | --- |
| 1 | Native expert schedule planner (in progress) | `rust/crates/ds41rt-core/src/replicated_expert_schedule.rs`, `lib.rs` export, `docs/tp-ep-scheduler.md` | none | pure Rust unit tests: LPT determinism, exactly-once, ties, overflow, inactive experts, 1/2/3 groups |
| 2 | Generic native shard selector (in progress) | `rust/crates/ds41rt-loader/src/v41_expert_staging.rs`, new loader test file, `docs/tp-ep-loader.md` | none | `cargo test -p ds41rt-loader`; real checkpoint header reads for world 2/3/4; byte-window equality vs `native_fp4_tp_projection` |
| 3 | Transport topology + flag + N-rank client (**mostly done**) | `rust/crates/ds41rt-transport/**` | 1 (types only) | implemented: `native_group.rs` (flag, one-hot word, `V41SparkTopology` + disjoint executor ids), `reduce_planes` FFI, `V41Tp4ChunkReceiver [;6]`/`new_ranks`; remaining: `V41Tp4Roce::new`/`with_clients` 2\|4 guard, `V41Tp4Tcp` fixed-4, `V41Tp4Planes::complete/planes()` world length |
| 4 | Config + launcher (in progress) | config/launcher scripts, new examples/tests (not `ds41rt.config`, not Rust) | none | shell syntax checks, `scripts/tests/*` launcher tests for TP2/TP3 x EP2/EP3, fingerprint, invalid combos |
| 5 | **Native TP3 ABI** (**mostly done**) | `native/src/v41_experts.cc`/`v41_tp2_experts.cc` + `native/include/ds41rt_v41_experts.h`, `native/cuda/kernels/v41_expert_pack.cu`, AOT export script, `rust/crates/ds41rt-ffi/src/v41_experts.rs` | none of 1-4 | implemented: packer `{576,768,1152,2304}`, native roles 5/6, FFI interfaces 8/9; remaining: CMake role branch/variant translation units and GB10 native self-tests |
| 6 | **Coordinator N-plane reduction** (kernel + FFI done) | `rust/crates/ds41rt-daemon/src/v41_experts/coordinator.rs`, `native/cuda/kernels/v41_route_reduce.cu` | 3, 5 | implemented: `reduce_compact<2\|3\|4\|6>`, `valid_compact_planes`, `reduce_planes` FFI; remaining: `enqueue_reduce_planes` call + `NativeTp4Wave` N world/`device_bytes`; M16 oracle N=6 equals full-width, cancellation/reuse |
| 7 | **Daemon topology + Spark mask + service** | `rust/crates/ds41rt-daemon/src/v41_experts/service.rs`, `service/local.rs`, `v41_experts.rs` (`ExpertLayer`), `v41_native_serve.rs`, `v41_native_serve/distributed.rs`, `v41_native_serve/placement.rs` | 2, 3, 4, 5, 6 | admission tests for layer splits, world {2,3,4,6}, sentinel masking unit tests, boundary handoff for 1-RTX |
| 8 | Kernel sentinel audit (**done**, parallel) | `docs/tp-ep-kernel-audit.md` | none | sentinel 384 / any out-of-range id is never owned by `V41RoutePlan.pack`, so the group is skipped and `V41SliceReduce` writes zero; `tests/moe/test_v41_expert_numerics.py:143-146` proves the all-invalid replay; shipped AOT uses the grouped `slices` path at every capacity |
| 9 | End-to-end oracle + docs | test files + `docs/tp-ep-*.md` | all | full-width expert oracle at N=4/6; TP2xEP2, TP3xEP2, TP2xEP3 numerics; existing TP4 regression |

The remaining critical path is task 7 (daemon topology, Spark mask, service) plus
the task-6 coordinator dispatch/plane-budget wiring; task 5 needs only the CMake
role branch and device qualification, and task 3 only the RoCE/TCP constructor and
`V41Tp4Planes` world-length fixes. Keep `third_party/sparkinfer` changes behind its
own pin/lock and the `verify-sparkinfer-source.py` gate.

---

## 6. Tests and commands (no GPU benchmarks in this phase)

Static / unit:

```bash
git diff --check
bash -n build.sh run.sh wip.sh scripts/release-common.sh
cargo test --manifest-path rust/Cargo.toml -p ds41rt-core
cargo test --manifest-path rust/Cargo.toml -p ds41rt-loader
cargo test --manifest-path rust/Cargo.toml -p ds41rt-transport
cargo test --manifest-path rust/Cargo.toml -p ds41rt-ffi
DS41RT_PYTHON=.venv/bin/python scripts/run-with-python-env.sh \
  cargo build --manifest-path rust/Cargo.toml -p ds41rt-daemon
python3 scripts/verify-sparkinfer-source.py \
  --source third_party/sparkinfer --lock third_party/sparkinfer.lock.json
PYTHONPATH=python/reference .venv/bin/python -m pytest -q PATH_TO_TEST
```

Contract-level (read-only metadata, allowed):

```bash
cargo run --manifest-path rust/Cargo.toml -p ds41rt-loader --example v41_catalog -- SNAPSHOT
```

Numerical / GPU qualification (owned by later phases, listed for completeness):
the existing `scripts/qualify-ds41-*.py` components and the four-Spark TP4
qualification must be extended with N=6 masks and a full-width oracle; the
one-BF16-rounding invariant is the acceptance gate.

---

## 7. Unknowns and open decisions

1. **[assumption, resolved by orchestrator]** "Fully replicated expert groups"
   means each group stores all 384 experts at its TP width and experts are assigned
   to groups for load balance. Confirmed by the orchestrator; one earlier discovery
   pass read EP as expert-sharded instead. If the intent ever changes to
   expert-sharded EP, per-rank memory becomes `full/(TP*EP)` and the plan changes,
   but the wire/flag design is unchanged.
2. **[resolved by parallel work]** TP3 had no native ABI at the audited HEAD: no
   768 role, no symbol, no AOT package, packer rejected 768
   (`v41_expert_pack.cu:54`). The kernel discovery confirms 768 = 6x128 is
   padding-free and alignment-safe, and the in-flight tree already extends the
   packer to `{576,768,1152,2304}` and adds `spark_tp3` slice roles. Remaining
   unknown: whether the TP2/TP3 `w2` packed-column slice boundary is byte-aligned
   for the N256/K128 layout (`docs/tp-ep-kernel-audit.md`, open questions).
3. **[resolved]** The AOT sentinel predicate: the shipped grouped `slices` pipeline
   never owns an out-of-range id, `inverse = -1`, and `V41SliceReduce` writes zero;
   `tests/moe/test_v41_expert_numerics.py:143-146` exercises the all-invalid replay.
   See `docs/tp-ep-kernel-audit.md` and the §3.5 note. The mask must still be
   applied before the kernel consumes the ids.
4. **[partially resolved]** Exact TP4 Spark execution workspace: measured from the
   shipped AOT manifests at capacity 4096 (see §2.1), about 0.21-0.23 GiB plus
   staging, so weights dominate the fit. TP2/TP3 `kernel_intermediate` (1152/768,
   both already 128-aligned so padding-free) and their scratch sizes are derived
   from the exporter formula, not from an emitted artifact
   (`docs/tp-ep-kernel-audit.md`, open questions); no exporter was run in this
   phase.
5. **[unknown, gates the fit]** The modern default dual-RTX `b` is unmeasured in this
   audit. Historical anchors are 8/15/20 for the stated historical configs, not a
   current default. Every budget in §2.3 remains **conditional** until the runtime
   `dual RTX bottom-up expert placement expert_layers=` line is read; do not treat
   `b = 20` as known.
6. **[unknown]** Whether native TP2 should also be exposed on Sparks (the loader
   and FFI role 3 exist; `validate_world` currently requires EXL3 for world 2).
   This matters for a possible TP2 x EP1 variant but not for the three targets.
7. **[measured but stale]** Dual-RTX component rows and default KV-pool sizes differ
   by document era; pick one era per topology and state it.
8. **[code hazard]** `V41Tp4Planes::complete()`/`planes()` iterate four slots
   regardless of executor count; any N != 4 path must avoid them until fixed.
9. **[code hazard]** `COORDINATOR_GPU_HEADROOM_GIB` is validated but never used;
   `MEMORY_RESERVATION` plus `RUNTIME_HEADROOM` are the effective controls.
10. **[assumption]** `SPARK_EP` groups need no RTX-side per-group reduction root
    under Option A; only N-plane summation. Confirm no lane/scheduler assumption of
    "one local owner per layer" breaks when groups hold replicas (the local RTX
    layer path is independent of Spark groups).
11. **[unknown]** Producer and meaning of `placement_version` (an opaque `u64`
    passed into `expert_request`, `v41_backbone_router.rs:766-769`); it is the only
    existing per-wave field that could carry a topology id without a wire revision.
12. **[representation, not necessity]** The implemented native owner field is a
    one-hot 3-bit bitmap at bits 9..11 (`bit 9 + group`, exactly one bit set;
    `V41_MAX_NATIVE_GROUPS = 3`). A 2-bit group index also encodes groups 0..2, so
    three bits are a representation choice that makes the exactly-one-owner
    invariant a single set-bit test and parallels the EXL3 owner word. The paired
    word keeps bits 9..10 (`WORD_MASK = 2047`); bits 12..31 stay reserved and request
    bits 12-15 remain the striped-collective part count. The request flag selects
    which decoder is legal, so the two encodings do not collide.
13. **[unknown]** Which stack the target topologies will actually serve. The native
    `v41_native_serve` path (hidden 5120) and the legacy `real_full` path (hidden
    4096/7168 with four-shard arenas and a row-scoped striped collective) share the
    wire but not the reduction graph. This audit assumes the native path.

---

## 8. Source index

- Config/launch: `ds41rt.config`, `run.sh`, `scripts/release-common.sh`,
  `scripts/run-wip.sh`, `wip.sh`.
- Loader/shard: `rust/crates/ds41rt-loader/src/{v41_catalog.rs,v41_expert_staging.rs,expert_format.rs,v41_exl3.rs}`.
- Core: `rust/crates/ds41rt-core/src/{constants.rs,placement.rs,expert_host_batch.rs,exl3_tp4_ownership.rs}`.
- FFI: `rust/crates/ds41rt-ffi/src/v41_experts.rs`.
- Transport: `rust/crates/ds41rt-transport/src/{protocol_v2.rs,v41_expert.rs}`, `v41_expert/{chunks.rs,roce.rs,tcp.rs,paired.rs}`.
- Daemon: `rust/crates/ds41rt-daemon/src/v41_experts.rs`, `v41_experts/{service.rs,service/local.rs,coordinator.rs,execution.rs,tp2.rs,tp2_ffn.rs,local.rs,paired.rs}`, `v41_native_serve.rs`, `v41_native_serve/{distributed.rs,memory.rs,placement.rs}`, `v41_backbone_lane.rs`, `cli.rs`.
- Native/kernel: `native/cuda/kernels/{v41_expert_pack.cu,v41_route_reduce.cu}`, `native/include/ds41rt_v41_experts.h`, `native/src/v41_experts.cc`, `third_party/sparkinfer/b12x/moe/_shared/kernels/v41_route_plan.py`.
- Evidence: `docs/phase2-memory-breakdown.md`, `docs/phase2-adaptive-verification.md`, `docs/release-v1-memory.{md,json}`, `docs/release-v6-plan.md`, `docs/release-v7-published-memory-evidence.log`, `docs/phase1-async-counting.json`, `docs/ds41-expert-tp-efficiency.md`, `docs/ds41-native-expert-packing-qualification.md`, `docs/ds41-architecture-audit.md`.
