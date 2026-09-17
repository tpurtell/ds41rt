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

### Paired package export

`package_v41_exl3_aot.py build --role spark --paired-tp4` exports last-boundary
kernels for ranks 0/2 and first-boundary kernels for ranks 1/3, all with physical
width 640. Each capacity retains the pinned compiler's normal tile policy. The
package records `paired_tp4: true` and each variant's boundary; verification
checks rank orientation, width, tier count and descriptor ABI and rejects
paired artifacts labelled as ordinary packages. CPU package tests pass. The SM121 export also completed and verified all 24
rank/capacity variants (1, 16, 80, 256, 1024 and 4096 rows). This proves package
compilation, not live Rust-worker correctness. Evidence:
`release-v5-exl3-paired-package.json`.

### Rust worker GPU check

The paired Rust worker test passed on GB10 for layer 30 at 80 rows, emulating
ranks 0 and 1 sequentially with all 384 experts resident and six experts routed.
Both first/last boundary orientations matched the Python reference byte-for-byte
through mapped output and multi-chunk host responses; mapped-buffer guard bytes
remained intact. The initial harness launch lacked `/opt/ds41rt/lib` in its
library path; correcting that allowed the test to run and pass in 1.77 seconds.
This verifies worker loading, ownership upload, native launch and response
formatting at that shape. It does not establish live four-Spark execution,
capacity transitions, changing ownership across Rust worker launches or speed.
Evidence: `release-v5-exl3-paired-worker-gpu.json`.

The extended worker check also passed ownership inversion followed by restoration
and row counts 80→1→16→17→80 on both boundary orientations. Restored 80-row
outputs were bitwise equal; smaller-capacity prefixes passed the predeclared
relative-max ≤0.006 and relative-L2 ≤0.003 gates for different BF16 tile schedules.
This checks stale state across m1/m16/m80, not every large-prefill transition.
All four Sparks now have verified candidate packages and identical ARM binary
hashes staged separately from their running services. Evidence:
`release-v5-exl3-paired-worker-transitions.json`.

### First live paired serving smoke

The paired coordinator and all four Spark workers served code and topic requests
at C1/C8/C16 successfully, including objective output checks and warm-cache
assertions. The run retained 25 RTX expert layers, K7 and the 14M-token pool;
coordinator readiness was 27.833 seconds. Initial ownership used packed weight
bytes only, without calibrated row costs. Single-sample throughput was mixed
and does not establish a performance benefit. Repeated matched comparisons,
large-prefill and lifecycle qualification remain pending. Original services
were restored. Raw requests/results and logs are archived alongside
`release-v5-exl3-paired-serving-smoke.json`.

### Matched live comparison

A disjoint→paired→paired→disjoint comparison used identical binaries and pinned
source, 25 RTX expert layers, K7, 14M KV, stock memory clocks and 400 W RTX limits.
Each arm had a warmup and three measured samples per code/topic C1/C8/C16 point,
giving six samples per layout. Median paired changes were code −2.50%/+2.39%/+0.62%
and topic +4.19%/+3.62%/−1.89%. Code C16 produced identical continuations; other
points sometimes differed in output and length, especially topic, so the results
do not isolate kernel cost. Paired mode remains opt-in. Fixed-input timing using
production tile policies and the same weight-byte planner costs is next. All
original services were restored. Evidence and raw outputs:
`release-v5-exl3-paired-comparison.json` and its adjacent archive.

### Fixed-input production-tile timing

The m80 comparison used real layer30 weights, identical routed inputs and the
same weight-byte ownership policy on one GB10, timing each rank sequentially.
It retained production tiles, including the different 512-wide disjoint tile.
All 60 sum/reference checks passed across one- and two-block paired runs. With
30–384 distinct experts, paired one-block slowest-rank times improved roughly
5–7%; two-block paired times improved 11–13% versus their disjoint controls.
Six-expert cases benefited more from two blocks, but these were still m80
executions, not measurements of the m1 serving policy. The next candidate uses
two blocks only for paired m80. No serving-default change is justified yet.
Evidence: `release-v5-exl3-paired-fixed-input.json` and its raw archive.

### Paired m80 residency serving comparison

The one→two→two→one block comparison changed only the four m80 libraries and
manifests; all other capacity files were bitwise identical. With six measured
samples per configuration, two-block C16 changes were code +2.79%, topic +2.52%
and mixed traffic +2.81%. Topic C8 was +0.18%; mixed C4 was −0.94%. Raw code C8
was −2.52%, with more alternate continuations in the two-block samples. A
frequency-selected same-output diagnostic showed approximately +2.9% at code
C8 (four versus three qualifying samples), but does not replace the raw result.
Two-block m80 is retained as the preferred paired candidate, not a default
release policy. Paired mode remains opt-in pending the remaining calibration
and qualification. All original services were restored. Evidence:
`release-v5-exl3-paired-residency-comparison.json` and the adjacent raw archive.

### Preferred candidate lifecycle

The paired two-block m80 candidate passed mixed C4/C16, a 32,815-token needle
prompt, full prompt-cache reuse, and retained-turn continuation (32,821 cached
tokens). A 16-request batch cancelled eight requests after content while all
eight survivors completed the exact requested sequence; a subsequent recovery
request also passed. This covers larger prefill-to-decode transitions and basic
cache/cancellation behavior for the dual-RTX candidate, not the entire release
quality suite or 1M-context behavior. Original services were restored. Evidence:
`release-v5-exl3-paired-lifecycle.json` and the adjacent archive.

### Fixed-K7 acceptance baseline

The preferred paired candidate completed an instrumented fixed-K7 collection
with code and topic at C1/C8/C16 plus mixed C4/C16. Excluding terminal/constrained
observations, code accepted 4.468 of 7 verified drafts on average (4,602/7,210;
63.83%), while topic accepted 1.735 (4,924/19,866; 24.79%). Each content cohort
contains 26 requests including warmup and concurrency drain. These pooled,
short-context observations are not a full-model quantization comparison.
Conditional confidence calibration stops at the first mismatch; unverified
suffixes are never assigned labels.

The summary now supports K7 and independent-lane scheduler records. Ten focused
analysis tests pass. Placement-cost fitting keeps topic entirely held out, in
addition to even draft widths on code/mixed. The remaining width collection and
adaptive serving validation are pending. Debug-instrumented throughput must not
be used in release performance tables. Evidence and reproducible analysis:
`release-v5-exl3-paired-k7-acceptance.json` and its adjacent archive.

### High-effort reasoning-code acceptance

The new ninth content category uses the same merge-intervals coding task with
thinking enabled and high effort. A fixed-K7 dual-RTX paired run passed code
checks at C1/C8/C16, retaining both reasoning and final answers. Across 26
requests including warmup, 6,334 nonterminal observations accepted 20,395 of
44,338 verified draft tokens: **46.00%, or 3.220 drafts per cycle**. This cohort
covers both reasoning and answer generation. The corresponding non-thinking
code baseline was 4.468 drafts per cycle; topic was 1.735. This is a content
comparison, not a measurement of quantization loss.

Throughput accounting now times all completion tokens from the first reasoning
or answer delta; answer latency remains separately recorded. These runs enable
debug instrumentation and do not supply final release throughput. Original
services were restored. Evidence: `release-v5-exl3-reasoning-code-acceptance.json`
and its archive, including all draft observations, scheduler rounds, client
responses and collector sources. Per-layer timing logs are excluded from this
acceptance archive; the complete original log hash is recorded.

### Placement-aware adaptive cost fit

Fixed K1–K7 collection produced 12,004 complete verification rounds including
the separately held-out reasoning-code run. The fit uses 3,031 warm odd-width
code/mixed rounds. Even widths validate code/mixed; every topic and reasoning
observation is held out. Median absolute verification-time prediction errors:

| Held-out workload | Legacy formula | Placement fit | Fit p90 error |
| --- | ---: | ---: | ---: |
| Code, K2/K4/K6 | 34.30% | 8.06% | 21.74% |
| Mixed, K2/K4/K6 | 15.35% | 7.49% | 19.10% |
| Topic, K2/K4/K6 | 35.93% | 8.11% | 23.12% |
| Reasoning code, K7 | 42.32% | 10.90% | 26.09% |

The fit separates TP2 RTX and TP4 Spark layer costs, plus non-expert round cost.
Nonnegative hinge coefficients converged to zero; the affine and hinge fits
therefore agree. These predictions use observed routes and instrumented elapsed
times. They do not establish forecast accuracy or a serving throughput gain.
An uninstrumented adaptive legacy/calibrated/calibrated/legacy comparison is
required before choosing defaults. This profile covers paired dual-RTX EXL3
only; single-RTX and full-model calibration remain pending.

The K5 trace exposed a logging defect: disabled route capture left old decode
rows available to a same-sized prefill. Logging now checks that capture is
active. The two identified old prefill batches are recorded in a trace-hash
bound exclusion manifest; the parser still rejects unexplained leftovers and
forbids excluding any completed verification round. Corrected K2/K4/K6 traces
need no exclusions. Evidence: `release-v5-exl3-paired-adaptive-cost-fit.json` and
its archive. Filtered archived cost traces reproduce every original parsed
observation exactly; source hashes and the explicit exclusion are retained.

### Confidence audit and reasoning-budget incident

An offline audit fits one affine logit correction on warm odd-width code/mixed
conditional labels, holding out even widths plus all topic and reasoning code.
The fitted intercept/slope are −0.1106 / 0.9412. Raw confidence already predicts
mean emitted tokens closely: held-out code actual/raw = 4.00/4.07, mixed =
2.72/2.73, and reasoning code = 4.21/4.20. The correction slightly improves code
Brier score but worsens mixed, topic and reasoning scores and their mean-emission
bias. **Keep raw confidence; no correction is adopted.** Archived filtered
observations reproduce the fit and every cohort exactly. Evidence:
`release-v5-exl3-confidence-audit.json` and its archive.

The first calibrated serving arm stopped at reasoning-code C8 when at least one
request exhausted its 4,096-token budget without a final answer. The old client
reported this as incomplete SSE and failed to retain that response. The client
now distinguishes a completed reasoning-only stream from transport failure and
preserves all responses in failed batches while keeping code checks failed.
A fresh calibrated repetition passed all code/topic/reasoning C1/C8/C16 and
mixed C4/C16 samples without reproducing the exhaustion. This does not erase
the original incident or establish a cause. All four complete comparison arms are now available below; the reasoning
budget is unchanged.


### Adaptive cost profile: completed serving comparison

**Keep the affine placement profile experimental.** The legacy/calibrated/
calibrated/legacy comparison completed with three samples per arm, yielding six
samples per mode at each point. Both modes use paired m80 two-block packages,
25 RTX-resident expert layers, K7, a 14M-token pool, and info-level logging.
The first complete fresh calibrated repetition replaces the aborted attempt
in these medians; the original attempt remains in the evidence archive.

| Workload | Concurrency | Legacy TPS | Calibrated TPS | Change |
| --- | ---: | ---: | ---: | ---: |
| Code | 1 | 186.66 | 182.88 | −2.03% |
| Code | 8 | 775.24 | 777.29 | +0.27% |
| Code | 16 | 1303.42 | 1330.95 | +2.11% |
| Topic | 1 | 104.34 | 100.64 | −3.54% |
| Topic | 8 | 472.12 | 490.09 | +3.81% |
| Topic | 16 | 769.38 | 768.42 | −0.12% |
| Reasoning code | 1 | 144.16 | 132.51 | −8.08% |
| Reasoning code | 8 | 618.12 | 572.38 | −7.40% |
| Reasoning code | 16 | 822.34 | 949.12 | +15.42% |
| Mixed traffic | 4 | 188.76 | 190.59 | +0.97% |
| Mixed traffic | 16 | 308.40 | 312.78 | +1.42% |

Code C1 produces the identical 231-token answer in every arm, so output variation
does not explain its loss. Median whole-response latency also rises from
1.23337 to 1.25903 seconds (+2.08%), while first-output latency moves only from
1.226 to 1.369 milliseconds. Excluding first-output time from the decode metric
therefore does not explain the regression. Reasoning C1 changes from 978 to 1,529 completion
tokens, and concurrent reasoning outputs also vary; those throughput changes
are serving observations, not isolated kernel speedups. No workload-specific
or concurrency-specific switching is adopted from this small comparison.
Single-RTX calibration is proceeding separately.

Evidence: `release-v5-exl3-adaptive-serving-comparison.json` and its archive.
The archived summarizer reproduces all reported metrics after relocation;
all successful and aborted collection arms are retained. Same-output diagnostics
match reasoning plus answer, select common outputs separately per concurrency
by frequency, and do not replace the all-sample table.

An offline `capped16` timing diagnostic allows a lower positive per-row slope
above 16 rows while keeping total predicted cost monotonic. Using the existing
traces, held-out median absolute error falls from 8.06% to 6.47% for code,
7.49% to 7.26% for mixed, 8.11% to 6.17% for even-width topic, and 10.90% to
7.79% for reasoning code. Reasoning p90 error worsens from 26.09% to 26.97%.
This basis cannot be exported to the current runtime schema and is not enabled
in serving. Forecast accuracy and prefix decisions still require analysis;
improving an observed-route fit alone is insufficient. The archived filtered
calibration traces reproduce all coefficients and evaluations exactly.
Evidence: `release-v5-exl3-capped-cost-diagnostic.json`.


### Separating route forecasts from timing error

Reanalysis of the earlier **disjoint** dual-RTX EXL3 adaptive trace finds little
average bias in forecast unique-expert counts: predicted/observed means range
from 0.990 to 0.998 across RTX/Spark backends and code/topic C1/C8. Individual
rounds still have 7.75–10.69% mean absolute relative error; near-unbiased averages
do not imply exact route prediction. In the same trace the legacy formula
predicts verification times 1.19–1.43 times the observed times on average.
This separates count-forecast bias from timing-model bias in that older run.

These observations do not diagnose the paired candidate's C1 regression:
the kernels differ, and only selected prefixes are observed. Rejected prefix
alternatives have no counterfactual timings in this trace. The next policy
analysis must retain that distinction. No runtime change follows from this
audit. `summarize-ds41-decode-profile.py` now validates forecast shapes and
reports backend-specific count errors. Every pre-existing summary metric was
reproduced unchanged; both input files match the published evidence archive
byte for byte. Evidence: `release-v5-exl3-route-forecast-audit.json`.

### Reproducing calibration evidence

`scripts/archive-ds41-calibration.py` packages completed fixed-width runs with
both timing and acceptance records. It requires passing code, topic, reasoning
code and mixed collections plus a recorded coordinator restoration. Filtering
retains cost-model, draft-observation and lane-round records in every content
window, remaps byte offsets, and records the original and filtered trace hashes.
Before writing the archive it compares every parsed cost record and every
parsed acceptance observation/round against the original, then checks saved
acceptance summaries. The archive includes clients' results, run configuration,
parsers and a file-hash manifest. A single-RTX K5 trial reproduced all 3,089 cost
rounds and all content acceptance summaries exactly. This validates evidence
preservation; instrumented timings remain unsuitable for release TPS tables.

### Single-RTX EXL3 calibration completed

Fixed K1–K5 collections passed code, high-effort reasoning code, topic at
C1/C8/C16, and mixed traffic at C4/C16. This configuration keeps six routed
expert layers on one RTX, with paired TP4 Spark ownership and the m80 two-block
package. There are 21,932 complete cost rounds, of which 21,772 pass the warm
shape filter; 2,451 odd-width code/mixed rounds train the fit. All topic and
reasoning-code rounds are held out, as are even-width code/mixed rounds.

| Held-out workload | Legacy formula median error | Affine fit median error | Affine p90 error |
| --- | ---: | ---: | ---: |
| Code, K2/K4 | 13.77% | 8.32% | 17.95% |
| Mixed, K2/K4 | 16.02% | 6.16% | 15.01% |
| Topic, K2/K4 | 13.94% | 8.73% | 18.70% |
| Reasoning code, K2/K4 | 13.44% | 6.30% | 17.21% |
| Reasoning code, K1/K3/K5 | 12.69% | 8.24% | 18.38% |

The reference column is the legacy affine formula, **not** the existing
single-RTX placement profile. These observed-route fits do not qualify a new
serving default. The capped16 diagnostic improves most held-out median errors
further but remains non-exportable; mixed even-width median error increases
from 6.16% to 6.52%. All variants and cohorts are retained in
`release-v5-exl3-single-cost-fit.json`.

The K5 acceptance slice includes 26 requests per content type, including warmup
and concurrency drain. Terminal and constrained observations are excluded.
Reasoning-code counts include both reasoning and final-answer generation.

| Content, fixed K5 | Accepted / verified drafts | Acceptance | Mean emitted tokens/cycle | Zero-acceptance cycles |
| --- | ---: | ---: | ---: | ---: |
| Code | 4,422 / 5,450 | 81.14% | 5.06 | 4.40% |
| Reasoning code | 26,602 / 42,740 | 62.24% | 4.11 | 15.72% |
| Topic | 3,876 / 10,990 | 35.27% | 2.76 | 24.84% |

These rates cannot be compared directly with the earlier dual-RTX K7 slice as
a quant-quality change: verification length differs. The evidence retains all
five widths for later matched-width full/EXL3 comparisons. Instrumentation also
precludes using these runs as final throughput tables.

Evidence: `release-v5-exl3-single-calibration.json` and its archive. Every cost
record, acceptance observation and lane round is reproduced exactly after
filtering, and all fitted coefficients/evaluations match the originals. Startup
logs retain the actual residency and pool reservation alongside run commands.
Full-model calibration on two RTX cards has started; its one-RTX runs follow.

### Full-model dual-RTX reference calibration

All K1–K7 fixed-width collections passed the same code, topic, high-effort
reasoning-code and mixed-traffic checks using the full model with FP8 PLE,
20 RTX expert layers, and the native NVFP4 expert path. There are 26,488 complete
cost rounds, 26,264 warm rounds, and 2,908 odd-width code/mixed training rounds.

| Held-out workload | Legacy formula median error | Affine fit median error | Affine p90 error |
| --- | ---: | ---: | ---: |
| Code, K2/K4/K6 | 23.91% | 12.76% | 28.79% |
| Mixed, K2/K4/K6 | 10.13% | 11.18% | 21.69% |
| Topic, K2/K4/K6 | 28.42% | 13.37% | 30.01% |
| Reasoning code, K2/K4/K6 | 20.14% | 7.51% | 27.43% |

The mixed-traffic median error worsens, even though other cohorts improve.
This fit remains experimental; no full-model serving default is changed.
The capped16 diagnostic also leaves mixed median error at 11.18%. All raw
cost/acceptance records reproduce exactly from the filtered archive in
`release-v5-full-dual-calibration.json` and its adjacent archive. The fit report
is `release-v5-full-dual-cost-fit.json`.

The first matched fixed-K7 content comparison is now available:

| Content, dual RTX / K7 | Full accepted / verified | Full acceptance | EXL3 accepted / verified | EXL3 acceptance |
| --- | ---: | ---: | ---: | ---: |
| Code | 4,520 / 7,364 | 61.38% | 4,602 / 7,210 | 63.83% |
| Reasoning code | 25,310 / 54,544 | 46.40% | 20,395 / 44,338 | 46.00% |
| Topic | 4,038 / 14,882 | 27.13% | 4,924 / 19,866 | 24.79% |

Each cohort has the same prompt, generation limit and 26-request C1/C8/C16
schedule including warmup. Code/topic corpus hashes differ because reasoning
code was added later; their original prompts and disabled-thinking payloads
were checked unchanged. Reasoning code uses the same corpus hash and high-effort
thinking with a 4,096-token cap. Full/EXL3 residency is 20/25 RTX expert layers;
generated outputs and cycle counts can differ. These are draft-acceptance
observations, not teacher-forced top-1 agreement or final throughput evidence.
EXL3 sources are the earlier paired K7 and reasoning-code acceptance archives.

### Reproducible paired residency build option

The package builder now accepts explicit paired-capacity residency choices:
`--paired-tp4 --residency 80=2` requests the tested two-block m80 candidate for
both boundary orientations and leaves every other capacity unchanged. This
replaces the experiment's Python monkey-patch with an ordinary build argument.
It introduces no automatic device heuristic or default change: B12X still owns
kernel resource validation and compilation. Overrides require a selected
capacity, one or two blocks/SM, and the paired Spark package; duplicate or
non-paired requests fail before importing CUDA.

New manifests retain requested overrides and per-variant compiled residency.
Verification checks both against kernel metadata, including all four rank
copies. Five focused packaging tests pass, and the updated verifier accepts
all 24 variants of the existing measured candidate package. A fresh GPU build
through this option, exact source-pin promotion, and normal clean-image
qualification remain pending; they will run after the measurement queue.

### Full-model single-RTX reference calibration

All K1–K5 collections passed with five RTX expert layers and 35 Spark expert
layers. The same code/topic/reasoning C1/C8/C16 and mixed C4/C16 schedule yielded
20,674 complete cost rounds, 20,514 warm rounds, and 2,405 odd-width code/mixed
training rounds. This completes fixed-width calibration for full/EXL3 on both
one and two RTX cards.

| Held-out workload | Legacy formula median error | Affine fit median error | Affine p90 error |
| --- | ---: | ---: | ---: |
| Code, K2/K4 | 9.31% | 10.50% | 26.51% |
| Mixed, K2/K4 | 28.32% | 6.64% | 19.51% |
| Topic, K2/K4 | 8.87% | 10.10% | 26.53% |
| Reasoning code, K2/K4 | 8.24% | 9.09% | 26.58% |

The improved mixed fit trades against worse errors elsewhere. No profile is
promoted. As with the EXL3 single-RTX fit, the reference here is the legacy
formula, not the embedded placement profile currently used by default.
The queued EXL3 serving comparison explicitly uses that embedded profile as
its baseline, with its bytes checked against the tested binary's build source.

| Content, single RTX / K5 | Full accepted / verified | Full acceptance | EXL3 accepted / verified | EXL3 acceptance |
| --- | ---: | ---: | ---: | ---: |
| Code | 4,472 / 5,200 | 86.00% | 4,422 / 5,450 | 81.14% |
| Reasoning code | 19,662 / 34,780 | 56.53% | 26,602 / 42,740 | 62.24% |
| Topic | 4,290 / 10,800 | 39.72% | 3,876 / 10,990 | 35.27% |

Prompts, corpus hashes, thinking/effort controls, generation limits, concurrency
points and repeat counts match for each content pair. Each includes 26 requests
with warmup and drain. Both target and draft weights differ between checkpoints;
this does not isolate draft-only quantization or measure answer quality. Actual
outputs and cycle counts differ. Final adaptive acceptance and release TPS
remain separate measurements.

Evidence: `release-v5-full-single-calibration.json`, its archive, and
`release-v5-full-single-cost-fit.json`. Archive filtering reproduced every cost
observation and every acceptance/round record exactly, and checked the saved
acceptance summaries. Startup allocation logs and original source hashes remain
in the archive. All four calibration configurations are complete; this does not
complete serving policy selection, clean builds, release qualification or v5
publication.

The native CMake exporter now accepts explicit paired-package configuration:
`-DDS41RT_V41_EXL3_PAIRED_TP4=ON -DDS41RT_V41_EXL3_RESIDENCY=80=2`.
This forwards the package builder's existing options and records residency in
the package manifest. Configuration rejects coordinator targets, tier counts
other than two, duplicate or unselected capacities, and residency other than
one or two blocks per SM. B12X still validates actual kernel resource limits.
The default remains disjoint, with no residency override. CPU-only tests
configure the actual CMake rules and inspect generated export commands; they
do not substitute for a fresh GPU build. Updating the release source pin,
coordinator ownership selection and standard build/run defaults remains
required before publishing the paired package.
