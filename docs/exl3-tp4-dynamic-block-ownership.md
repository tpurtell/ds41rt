# EXL3 TP4: balance whole rotation blocks per expert

Proposed TP4 optimization. The CPU ownership planner and explicit paired loader
layout are implemented and tested. Native paired kernels and worker dispatch
are implemented, but coordinator integration and distributed qualification
remain incomplete. This is not a
release default and has no measured serving benefit yet.
Keep H128 rotations local while distributing the two extra blocks of each
18-block expert across the four Sparks. TP6 remains a separate goal.

## Resident layout

Number the expert's 2,304 intermediate channels as blocks 0–17, each 128
channels wide. Each pair stores one duplicated boundary block:

| Spark rank | Always computes | Optional block stored locally | Partner |
| --- | --- | --- | --- |
| 0 | 0–3 | 4 | 1 |
| 1 | 5–8 | 4 | 0 |
| 2 | 9–12 | 13 | 3 |
| 3 | 14–17 | 13 | 2 |

For each active expert, exactly one member of each pair computes its boundary
block. All 18 blocks contribute exactly once. Every Spark stores five blocks,
or 640 channels, rather than today's 640/640/512/512 allocation. Aggregate
expert storage increases from 18 to 20 blocks: approximately 11.1% for the
projection payloads. This is in-memory duplication; checkpoint files remain
shared, unchanged inputs. Exact metadata, workspace and startup budgets still
need measurement.

Gate, up and down projections must use the same channel ownership. Their
rotation/scaling tables follow the corresponding global channel indices.
Boundary blocks must stay whole through rotations and activation. Each Spark
still returns its ordinary full-hidden-width partial for the existing TP4 sum.
Changing down-projection summation groups can change floating-point rounding;
mathematical equivalence does not imply bitwise equivalence to the old split.

## Admission and ownership

Build one expert reuse histogram per lane's layer batch. Assign each active
expert one ownership bit per pair, and retain that assignment for every row
using that expert throughout the batch. No decision waits for the other lane.

Balance estimated incremental boundary-block cost, starting with distinct
expert weight bytes and adding a measured routed-row/reuse term. Token count
alone is insufficient: repeated use of one expert can reuse streamed weights,
while extra packed row groups still add computation. Mixed projection K3/K4
rates also change the bytes associated with each expert.

A deterministic largest-cost-first assignment to the lighter member of each
pair is a reasonable first candidate. Tie breaking can depend on layer/expert
identity without introducing a shared cross-lane decision. The coordinator
must publish one authoritative assignment in batch-owned storage; workers must
not independently infer ownership from possibly different execution order.
The placement planner should execute where route information is already
available, avoiding a new synchronous GPU-to-host read in the decode loop.

For six equal-cost experts, each Spark computes 24 mandatory block-expert
units plus three optional units: 27 each. The current arrangement computes
30/30/24/24. That illustrative case reduces the maximum streamed work by 10%;
it is not a measured kernel or serving speedup. Unequal costs and discrete
assignments leave residual imbalance.

## Fused execution requirements

### Current implementation and integration points

`rust/crates/ds41rt-core/src/exl3_tp4_ownership.rs` provides a reusable,
allocation-free-on-success planner. It accepts measured marginal block costs
for each pair and mandatory costs for each rank, rather than hard-coding a
bandwidth/reuse model. Each pair sorts active experts by descending cost and
assigns the next block to its lighter rank. Layer/expert identity can seed
deterministic ties. Inactive experts carry an invalid ownership sentinel.
The returned borrowed assignment prevents planner reuse while that view lives;
transport must still retain its own batch-owned assignment through completion.

The daemon's `v41_experts/paired.rs` adapter now builds these costs from
per-expert weight-streaming and per-routed-row coefficients, using fixed
384-expert histogram/cost scratch and a lane-owned planner. Its current model
uses the same marginal cost for each of an expert's four mandatory blocks
within a pair. It validates the ordinary request and all cost arithmetic before
encoding ownership, retaining route order and exact routing weights. A request
owns its encoded decisions independently of later scratch reuse. Single- and dual-RTX serving now install this adapter when
`DS41RT_EXL3_PAIRED_COST_PROFILE` names a JSON file. The schema is
`ds41rt.exl3-paired-cost.v1`; `layers` contains 40 arrays of 384 objects, each
with two-element integer arrays `weight` and `per_row` (one cost per pair).
Startup rejects incomplete, zero-cost or overflowing profiles and rejects
activation on a non-EXL3 checkpoint. The profile is shared read-only while every
lane owns separate scratch. Assignment happens after original route capture,
immediately before remote dispatch; local RTX expert execution is unaffected.
The profile must be used with matching paired Spark artifacts. No calibrated
profile is shipped yet, and this mode remains opt-in pending live qualification.

Four focused CPU tests cover all four ownership combinations (each of the 18
global blocks exactly once), the six-expert 27/27/27/27 example, unequal costs
and initial rank loads, inactive experts, scratch reuse, invalid extents and
overflow recovery. These prove planner/layout properties, not GPU equivalence.

The resident channel intervals are contiguous: `[0,640)`, `[512,1152)`,
`[1152,1792)`, `[1664,2304)`. Odd ranks have the optional block at their
**beginning**, even ranks at their end. The planner exposes an active local
block range of `0..5`, `0..4`, or `1..5`; kernels must retain the physical
five-block stride. Contiguous resident slices avoid requiring scattered file
reads, but loading time remains to be measured.

`v41_backbone_router.rs::expert_request` already obtains completed route IDs
from the router's pinned staging via `download_request`, then constructs the
Spark request. `v41_backbone_lane.rs` invokes this for remote dispatch. This is
the candidate histogram/planning insertion point without a new D2H operation.
The request protocol needs an explicit ownership contract before wiring it in.
The loader's `V41Exl3Partition::PairedTp4` now propagates the paired ranges
through projection slicing, staging reads, residency destinations and byte
budgets. Existing APIs keep `Disjoint` as their behavior. Paired mode rejects
other world sizes and intermediate dimensions; general K2–K5 disjoint support
is unchanged. The inspection example accepts `--paired-tp4` alongside `--read`.

Loader validation: 77 library tests passed, three environment-dependent tests
were ignored. The published-checkpoint projection-read test was then explicitly
run and passed: K3/K4 gate/up/down payloads, both rotations and MCG bytes match
full-checkpoint slices for disjoint TP2/TP4 and paired TP4. Synthetic K2–K5 tests
also exercise paired reads beyond 2 GiB, while residency tests check complete
buffer coverage without overlapping destinations. This does not qualify GPU
computation or loading performance.

For actual backbone layer 30 in the published FP8-PLE EXL3 checkpoint, the
initial paired weight-layout inspection (before adding the runtime ownership
row) recorded these logical resident allocations:

| Spark rank | Disjoint bytes | Paired bytes | Added bytes |
| --- | ---: | ---: | ---: |
| 0 | 1,560,096,784 | 1,560,096,784 | 0 |
| 1 | 1,560,096,784 | 1,560,096,784 | 0 |
| 2 | 1,252,798,480 | 1,560,096,784 | 307,298,304 |
| 3 | 1,252,798,480 | 1,560,096,784 | 307,298,304 |

The additional allocation is 293.0625 MiB per smaller rank for this layer.
Staging grows from 1,310,720 to 1,638,400 bytes on these ranks; minimum read
scratch stays 18,432 bytes. These figures include resident metadata but exclude
allocator alignment and execution workspace. They are a layer-specific budget,
not a complete Spark memory total. Exact results are recorded in
`release-v5-exl3-paired-residency.json`.

The paired native contract now adds one zero-initialized int32 descriptor row:
768 slots, or 3,072 bytes per rank for this two-tier checkpoint. Thus the
current paired total for this layer is 1,560,099,856 bytes on every rank;
the original disjoint allocation is unchanged. Ownership is populated into
this preallocated row per batch, including zeroing inactive/padded slots.

### GPU work still required

The next checkpoint gate has now passed: `qualify_v41_exl3_paired.py` loads
all 384 experts from actual layer 30 at hidden size 5120 and compares the sum
of four paired rank partials with four existing disjoint rank partials. All 24
checks passed across live row counts 1/8/16/24/64, low/high expert reuse and four
ownership patterns, including changed-input graph replay and poisoned unused
intermediates. Maximum normalized absolute error was 0.0036843 and maximum
relative L2 error was 0.0023791, within the predeclared 0.006/0.003 gates.
Changing BF16 partial summation groups is not bitwise equivalent; each paired
rank's graph replay did match its own eager execution exactly. This comparison
ran the eight partials on one Spark, so it does not qualify distributed
transport, an independent dequantization oracle, or performance. Results and
the exact runner are in `release-v5-exl3-paired-checkpoint.json` and
`evidence/v5-exl3-paired-checkpoint.tar.gz`.

A two-tier CuTe prototype is now on the SparkInfer fork (`ce9bfcec`), in an
isolated checkout; the engine's vendor lock has not advanced to it. It adds a
fourth int32 descriptor row containing per-expert local ownership. The physical
first/last boundary is an immutable compile option, while ownership remains
mutable graph input. On an unowned block, FC1 omits the gate/up output tiles and
FC2 reduces only the retained 512 channels, keeping the physical 640-channel
strides. Whole-tile scheduling is required so no split-K participant is removed.

Four GPU tests passed on GB10: first/last boundary crossed with K64/K128 tiles,
using live row counts 1, 3 and 8 and five successive ownership patterns under
graph replay. The tests poison unused FC1 intermediates and check graph/eager
agreement. This initial synthetic test uses hidden size 128 and compares with
the existing full-width kernel whose omitted block's gate post-rotation scales
are zeroed. It is not the independent, real-checkpoint four-rank oracle, and
does not establish reduced measured traffic or a performance gain. Evidence is
in `release-v5-exl3-paired-kernel.json` and the matching evidence archive.

Native export/metadata validation, transport lifetime handling and distributed
serving integration remain required; the prototype currently covers only two
tiers. One-versus-two blocks/SM and tiling remain separate performance choices
to measure on the mixed 512/640 active widths after correctness.

### Compact wire contract

`v41_expert/paired.rs` implements the ownership-word codec with explicit
paired admission. The existing 12-byte route entry is retained:
expert-word bits 0–8 hold ID 0–383; bits 9–10 hold the two pair-owner selections;
bits 11–31 must be zero. There are no additional payload bytes or messages.
Repeated routes for an expert must carry identical ownership throughout the
batch. Fixed-size batch scratch validates consistency and expands ownership
to the kernel's int32 row, clearing inactive and padded slots on reuse.

Request flag bit 17 identifies this contract. Generic framing accepts it only
with native compact-response semantics and optional debug checksums; legacy
streaming/compression flags cannot be combined with it. Ordinary native parsing
rejects paired frames; explicit paired parsing requires the flag and validates
decoded IDs, unique routes per row, and consistent ownership across the batch.
The worker service selects parsing from its loaded execution layout, checking
paired/disjoint agreement in **both** directions before execution. Even an
all-zero owner selection requires the paired flag; it must never silently run
on a disjoint or NVFP4 worker. The original model retains its equal 576-channel
TP4 split, ordinary expert IDs and existing request/response behavior.

### Native artifact admission

The exporter accepts an explicit `--paired-boundary first|last` for the
two-tier, 640-channel, top-k-six candidate. Paired DSOs reject the old
16-word `ds41rt_exl3_info` query and expose an 18-word
`ds41rt_exl3_paired_info` contract: version 3, explicit first/last boundary,
and four descriptor rows. Ordinary exports keep the original version-2 query.
This prevents an older engine from silently loading a paired kernel as disjoint.

Rust's existing `V41Exl3Kernel::load` still requires disjoint layout. The new
`load_with_layout` compares the reported layout with the caller's explicit
expectation before initializing the CUDA module. Malformed paired metadata,
wrong boundaries, and paired/disjoint mismatches are rejected.

Three native information tests passed in Rust. On GB10, disjoint, paired-first
and paired-last DSOs exported, linked, answered the expected queries and
completed CUDA create/destroy. Both paired exports used two blocks/SM at
capacity 80. This checks artifact construction and initialization, not native
compute launch correctness or performance. Evidence is in
`release-v5-exl3-paired-export.json` and its archive. These initial checks did
not exercise worker launch or paired frame admission.

The exported C compute path has subsequently passed real-checkpoint comparison
with B12x JIT for disjoint, paired-first and paired-last artifacts. Each used
six layer-30 experts and hidden size 5120, with live row counts 1, 3, 79 and 80;
all outputs were bitwise equal. The paired artifacts used two blocks/SM and
also passed four ownership changes through a captured native graph, including
poisoned unused FC1 storage and route metadata. The qualifier validates the
version-3 information contract and the old-query rejection before execution.
These are exported-kernel tests, not Rust-worker or transport tests. Evidence:
`release-v5-exl3-paired-native.json` and its archive. Paired fixture emission is
explicitly rejected until fixtures carry ownership metadata.

Rust execution setup now cross-checks the manifest's boundary, descriptor-row
count and native-info version against every resident layer and the loaded DSO.
Descriptor extents must match exactly. The explicit paired launch accepts an
already validated device ownership row and queues its copy into the prebound
fourth descriptor row on the launch stream. Ordinary launches reject paired
kernels, and paired launches reject disjoint kernels. Destination pointers are
resolved during setup, with no per-launch allocation or tensor-name lookup.
The caller must preserve ownership-buffer lifetime and exclusive descriptor
use through stream/graph completion.

Worker request decoding is now connected to this explicit launch. A paired
worker preallocates an ID-upload buffer with an additional ownership row,
unpacks routes into its prefix and writes ownership immediately after the live
IDs. The existing ID H2D transfer carries both, and the launch-stream device
copy moves ownership into the bound descriptor. There is no extra network
message, H2D call, per-request allocation or cross-lane decision. The extra GPU
input capacity is 3,072 bytes for the current two-tier checkpoint; it is included
in workspace planning. Ordinary workers retain their existing input sizes and
route-copy path. Responses strip the request-only paired flag and retain the
existing compact BF16 format.

All 157 transport library tests passed (three live/environment tests ignored),
including paired frame length equality, exact decoded IDs/FP32 weights,
per-pair ownership coverage, repeated-expert conflict rejection, ordinary versus
paired admission, and unchanged response flags. The daemon compiles with the
new worker path. This is not yet Rust-worker GPU or distributed-serving
qualification. Worker startup now selects the resident partition from the
explicit artifact manifest and validates matching boundary orientation across
all required capacities before reading weights. Paired plans use the existing
banked loader with exact arena-rounded residency and staging budgets; ordinary
RTX, dSpark and disjoint Spark callers retain the default layout. A focused
CPU test covers all ranks, wrong boundaries and mixed capacity packages.
Coordinator assignment/encoding, exporting matching serving packages, and live
rejection/cancellation/recovery tests remain required before deployment.

Coordinator response collection now validates explicitly flagged paired requests
for RoCE and TCP dispatch. Indexed response chunks strip the request-only
ownership flag, just as full responses do. The transport suite passes 158 tests
(three environment tests ignored), including two chunks per rank across all
four ranks and rejection of conflicting ownership before dispatch. This is
host-side framing evidence; live RoCE/GPU qualification remains pending.

The current resident layout and mixed kernel assume one intermediate width
for every expert in a launch. Supporting this proposal requires a real
per-expert active extent and boundary-block selection inside the fused path.

- Store a fixed five-block physical layout, with explicit global block mapping.
  Gate/up tile scheduling must omit an unowned optional block, and down must
  omit its K contribution. Reading all five blocks and zeroing the fifth does
  not deliver the intended streaming reduction.
- Preserve correct physical strides when the active extent is four blocks.
  Active width must not silently change the stride of the next expert or tier.
- Keep four- and five-block experts in the same cooperative launch. Splitting
  them into separate launches is only a control experiment, not the intended
  fused implementation.
- Preallocate ownership metadata, histograms and scratch for each lane/worker
  owner. Bind them to the request lifetime, including cancellation and error
  draining. Keep graph addresses stable and live counts out of compile keys.
- Extend native export, runtime metadata and transport validation together.
  Reject mismatched ownership contracts rather than silently using the fixed
  partition on one worker.

## Experiments and acceptance

1. Verify exact coverage of all 18 blocks across all four ranks for every
   ownership combination, including repeated experts and uneven row counts.
2. Compare the four-rank sum with an independent fixed-partition/reference
   computation on real mixed-projection checkpoint weights. Exercise both
   ownership choices, boundary rotations, clipping, masked/padded routes,
   changed-input graph replay, and cancellation/reuse of metadata storage.
3. Measure actual weight traffic or tightly controlled native kernel timings
   to establish that unowned blocks are skipped. Profile both small and large
   expert sets and reuse distributions; include the planner/transport cost.
4. Compare fixed placement, static per-expert placement, and dynamic ownership
   on code, topic and mixed serving workloads. Preserve loading speed, the RTX
   KV budget, and independent lanes. Inspect prefill separately.
5. Integrate only after the complete path demonstrates a useful gain without
   a clear regression. Retune adaptive costs after choosing the kernel/layout.
   The final quant agreement and release qualification requirements still apply.
