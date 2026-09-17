# V6: dual-RTX parallelism and RAM-backed KV

Status: implementation and review. Baseline: v5 (`2b71d90`).

Increase useful dual-RTX serving throughput while preserving the winning
configuration as the default. Evaluate attention, projection, and dSpark TP2
independently; shipping supported opt-in paths is acceptable when they do not win.

## Implementation sequence

- [x] Review and port PR #4 onto current dev, preserving contributor attribution,
  current SparkInfer interfaces, and independent execution lanes.
- [x] Resolve issue #2: token-aware admission, pressure waiting, and safe progress
  when decode needs additional pages. Distinguish retained-prefix offload from
  active-request parking; host snapshots alone do not guarantee active capacity.
- [x] Resolve issue #3: bounded HTTP admission, default queue depth equal to
  concurrency, 25-second wait budget, 429 with Retry-After for overload and 503
  for shutdown. Bound waiting request memory and support cancellation.
- [ ] Map current retention/SWA controls and set the requested leading-edge
  default to 20. Compute logical device-plus-host capacity above
  slots * max_context_tokens, allowing for snapshot overhead and staging.
  Replicated device copies count once toward logical capacity.
- [ ] Replace the dual-RTX 20-layer minimum with budget-driven bottom-up placement;
  coordinate the chosen RTX/Spark boundary before loading and verify fast startup.
- [ ] Implement independently selectable TP2 projection and attention paths,
  considering head partitioning with replicated KV (DCP1-style). Account for
  reductions, replicated state, graph storage, scratch and host paging bandwidth.
- [ ] Evaluate TP2 dSpark independently, including its experts, attention,
  projections and shared embedding/head access. Preserve lane-owned workspaces.
- [ ] Measure combinations only after individual correctness and timing checks.
  Recalibrate adaptive drafting for any winning placement/timing configuration.

## Evidence and release

- Use the full checkpoint for performance. Prioritize weighted content, code,
  reasoning-code, topic and mixed concurrency; counting is a headline metric.
- Use three final samples; 400 W per RTX and standard memory speed. No cooldown
  requirement. Preserve failures and record source, model, layout and cache size.
- Validate cache restore, pressure progress, cancellation, tools and long-context
  needles for affected paths. EXL3 receives basic compatibility checks only.
- Change defaults only after end-to-end wins without material regressions;
  include startup and prefill alongside the decode-first assessment.
- At publication move historical EXL3 tables to a linked secondary page, retain
  clear full-checkpoint results in README, and document new memory allocations
  in bytes and logical tokens. Include a diagram below the goal introduction.
- Commit and push incremental work. Final publication includes notes, containers,
  and release/v6 at the exact final tagged commit after qualification.

## Initial findings

PR #4 targets release/v3 and introduces pinned-RAM retained snapshots despite
its disk-persistence title. It also includes protocol/loader fixes and tests.
Review its dual-device ownership and active-pressure behavior before enabling it.
The current dual worker explicitly requires 20–40 resident routed-expert layers;
the launcher also assumes Spark execution starts at layer 20. Both must agree
with flexible placement. Current documentation contains older single-RTX
ownership descriptions and must be updated from the implementation.

Sources: [PR #4](https://github.com/tpurtell/ds41rt/pull/4),
[issue #2](https://github.com/tpurtell/ds41rt/issues/2),
[issue #3](https://github.com/tpurtell/ds41rt/issues/3).

## Integration progress (September 17)

Merged and pushed to `dev` at `85e2be4`; GitHub records PR #4 as merged and issues
#2 and #3 as completed. Subsequent startup hardening validates host-cache options
before model loading and allocates the pinned pool before signalling API readiness.

The `work/v6-hostcache` branch merges PR #4's original commits onto v5,
preserving attribution. Its host cache remains disabled by default pending
complete serving validation. The daemon compiles; the host-cache suite passes
214 tests (two soak tests remain ignored). Native HTTP tests pass 28 cases,
including bounded queue waiting, cancellation, overload status and Retry-After.
The recipe and binary retention default is now 20; the existing separate prompt
and completed-turn banks each use that limit, which the RAM planner must include.

Dual-device review found that host restore incorrectly labelled dSpark prefixes
as GPU0-owned. Restored rings now allocate on their runtime device, and snapshot
ownership derives from the actual buffers. Two hardware tests pass on both RTX
cards: mixed-device RAM round trips through both batch and fallback copies, and
GPU1 draft ownership with both pooled and directly allocated storage.

Token-aware admission now checks prompt plus output allowances for the whole
active cohort against actual source-page capacity, preserving prefix sharing
and partial-page copy accounting. A blocked request keeps only its prepared host
input and retries after retirement. The lane wake policy prevents a waiting
request or nonempty HTTP queue from repeatedly stopping decode while the same
requests still occupy the pool; cancellation wakes admission. An individually
oversized request receives a clear 400 response instead of an execution failure.
Three CPU policy tests and a native source-page pressure test pass. Core, loader
and transport library tests also pass (390 cases, seven hardware-dependent cases
ignored). The one-off serving pressure test below passes. Performance qualification
and the final default RAM configuration remain outstanding.

### One-off small-pool torture test

Run on September 17 against `5f487c1`, using the full native checkpoint and v5
native kernels: dual RTX, 20 TP2 routed layers, dSpark enabled, C4, two retained
entries per bank, 6 MiB requested global KV (13 groups = 6,656 logical tokens,
5,923,840 allocated global bytes), 64 MiB pinned RAM in 8 MiB chunks, and an HTTP
queue of four with a 500 ms queue-space wait. These are deliberately restrictive
test settings, not proposed release defaults. RTX power limits and memory speed
were unchanged. This is not a throughput benchmark.

The deterministic randomized pressure phase lasted 120.42 seconds. Including
warm-up and recovery, 1,085 requests yielded 48 HTTP 200 responses, 992 retryable
429 responses, two explicit 400 responses for requests exceeding the GPU KV pool,
and 43 cancellations before headers. There were 45 intentional cancellations in
total (two after HTTP 200), no unexpected 5xx responses, stream errors, transport
exceptions or 40-second request timeouts. All three final recovery requests
finished successfully in 0.296, 0.182 and 0.184 seconds.

Cache telemetry recorded 43 completed RAM restores, 116 device evictions and 49
host evictions. All eleven `restore_failures` were logged no-device-room skips;
there were no CUDA copy failures or restore timeouts. The current metric combines
capacity skips with copy failures, so interpret it alongside the logs. The final
stats snapshot had 74 completed stores of 75 issued (publication can lag while
the scheduler waits for the next request). Normal v5 serving was restored after
the test. The local evidence bundle is `~/.cache/ds41rt-v6-torture/`; its script is
deliberately not part of recurring release qualification.

This admission policy reserves future GPU capacity; it does not implement active
request parking in RAM. RAM sizing, complete cache qualification, flexible expert
placement, all TP2 experiments and v6 publication remain required.

## Automatic RAM budget candidate

`--host-cache-bytes auto` is implemented in the binary and launcher (also
`HOST_CACHE_BYTES=auto`). Explicit integer byte counts remain compatible, and
unit sizes such as `8GiB` are accepted. The current default remains zero pending
startup and performance assessment; enabling the recipe's automatic policy is
still required before publication.

The planner subtracts private GPU page headroom and counts each logical source
once. It fills the shortfall above `retained_entries * max_context` with whole
512-token groups. An additional full-context store, boundary pages for both
retention banks, separate tail/draft slab chunks and chunk fragmentation are
reserved without advertising them as token capacity. At 14M usable device tokens,
20 retained entries and a 1M context limit, the candidate requests 6.75 GiB pinned
RAM: 6M + 512 logical host tokens plus these reserves. This describes storage
capacity after reclaiming redundant GPU snapshots, not a guarantee that every
cached conversation remains a hit while write-behind copies overlap.

Three budget tests pass, including an allocation test through the real slab-pool
implementation with a fake pinned-memory provider. CLI parsing and the existing
launcher checks pass. Comparative serving performance remains to be measured.

The full dual-RTX pilot at `9763941` confirmed the planned 7,247,757,312-byte
(6.75 GiB) allocation and 20,972,032 combined logical tokens. Pinned allocation
took 1.21 seconds; owners became ready at 26.44 seconds, compared with a preceding
v5 restart at 22.39 seconds. Only 1.21 seconds is directly attributable to the
allocation; these isolated launches do not establish a complete startup delta.
Two 1,060-token code requests completed, with identical 192-token continuations;
the second reused all 1,060 prompt tokens. This is a startup/reuse smoke check,
not comparative throughput qualification. Normal v5 serving was restored. Evidence:
`~/.cache/ds41rt-v6-auto-host/`.

## TP2 implementation constraints from current code

Backbone query B expands 1,280 features to 32,768 (64 heads); the output path
uses grouped WO-A followed by an 8,192-to-5,120 WO-B projection. A natural TP2
candidate keeps 32 heads and their output groups on each card, then reduces the
WO-B partial outputs. Attention-only and projection-only options need explicit
transfers at their boundaries, so their costs must be measured independently.
Current sparse attention buffers/FFI assume 64 heads and need a real head-range
contract; changing weight placement alone cannot implement this split.

The distributed dSpark path currently runs all three transformer stages on GPU1;
only its vocabulary terminal is already TP2. Draft expert and attention splitting
must preserve each lane's independent replay and cache ownership.

Flexible RTX expert placement also needs launcher coordination. Spark workers
currently load from layer 20. Lowering the coordinator's minimum without changing
worker loading would leave missing experts. Prefer a coordinator-produced memory
plan before deferred expert loading, letting the launcher start Spark ranks at
the actual boundary while RTX routed weights load. Avoid loading unused bottom
experts on every Spark merely to conceal this dependency.

### Explicit reduced-layer validation

The binary and launcher now accept explicit dual-RTX routed counts from 1 to 40
(`--rtx-expert-layers`, `RTX_EXPERT_LAYERS`). The launcher starts Spark workers at
the selected boundary; the existing nonempty worker requirement keeps layer 39
loaded when all 40 are local. Single-RTX workers retain their existing full range.
Automatic selection still uses the old minimum until runtime-plan negotiation is
implemented; GPU preflight also retains its conservative 20-layer memory estimate.

A full-checkpoint 17-layer run exposed a GPU1-only assumption in remote/shared FFN
completion for layers 17–19, whose attention still lives on GPU0. Shared reduction
now targets the transport GPU, collection is polled under that device's scope,
and the completed result returns to the block GPU in existing lane-owned storage.
The usual GPU1 path does not perform this peer return. No buffers are added.

The corrected run matched the 20-layer baseline on three concurrent API requests
(short response, code, and reference lookup), including complete reuse of an
804-token prompt. The real-weight independent-lane FFN test also passes, extended
to compare opposite-GPU shared reduction and return byte-for-byte for both input
GPUs and row-count transitions. These checks establish execution correctness, not
a performance win from moving experts back to Spark.

With the same 14M usable GPU-token pool and automatic RAM budget, measured device
occupancy fell from [99,737,665,536; 100,375,199,744] bytes at 20 layers to
[88,901,681,152; 89,539,215,360] at 17: approximately 10.1 GiB freed per RTX.
The corrected coordinator initialized in 23.23 seconds (26.44 for the preceding
20-layer automatic-RAM pilot; isolated launches, not a controlled timing claim).
Temporary workers on port 19442 were removed and normal v5 serving restored.
Evidence is in `~/.cache/ds41rt-v6-placement/`, including the initial failing run.

### Compact attention head kernel foundation

Added opt-in native 32-head bounded attention entry points. Queries and outputs
use `[rows,32,512]`, sinks contain the local 32 heads, and split scratch is
`[rows,parts,32,514]`. The caller will supply replicated local KV; this does not
yet implement replication, Rust serving integration or the TP2 scheduler. Existing
64-head entry points retain their launch geometry and AOT eligibility. The new
path currently uses WMMA, so a compact AOT backend remains needed for a competitive
decode experiment.

The expanded native selftest passes 288 cases on each RTX: closed-form output and
bit-for-bit comparison of two compact head halves with the full-head kernel,
covering FP4, legacy FP8, window-only attention, rows 1/6/128/256, sequential and
three-way split execution, private source overlays and valid/invalid replay bounds.
Queries and sinks vary across heads for the split comparison. Captured graphs
correctly observe changed bounds between replays, and undersized split scratch is
rejected. Compute Sanitizer memcheck reports zero errors. CUDA 13.3 SM120a
compilation passes both with and without the AOT feature macro; numerical tests
use the standalone WMMA build. Full native AOT integration and serving performance
are not established by these tests. Evidence: `~/.cache/ds41rt-v6-heads32/`.

Compact 32-head multi-request batch validation and launch are now implemented.
The existing 64-head batch API shares the same validation rules and retains its
original geometry. Additional six-row tests cover distinct request descriptors
for all three cache formats, compare both local head halves against full-head
batch output, and mutate descriptors between captured graph replays. Undersized
scratch and descriptor/output overlap are rejected. The expanded suite passes
on both RTX cards, and its memory-sanitized run reports zero errors. These are
native WMMA checks; Rust binding, compact AOT and replicated-cache serving remain
outstanding. Evidence: `batch-selftest*.log` and `batch-memcheck.log` in the same
local bundle.

### Optimized compact attention backend

SparkInfer now accepts static 32- or 64-head geometry in its direct V4.1 pointer
ABI, including the sink merge. The fork's master contains `63e2140e`; the source
lock and submodule are updated together. Head count participates in the compile
key, while live row counts remain runtime arguments. The existing producer oracle
still passes, and two compact halves match full-head AOT output bit-for-bit under
live row-count changes and graph replay with kernel resolution frozen. The tests
also retain invalid-row, private-overlay and recycled-page coverage. The producer
test passes on both RTX cards; the address-only high-page oracle passes on GPU1.

Native builds export both geometries into separate initialized modules. The
32-head bounded entry dispatches eligible aligned FP4 decode to AOT; a separate
AOT batch entry follows the existing prevalidation contract. Other inputs retain
WMMA fallback. The native selftest now includes ten-way split attention and aligned
AOT fixtures: 432 cases pass on each RTX, including bounded descriptor staging and
batch descriptor mutation during replay. Compute Sanitizer reports zero memory
errors for the exported native build. This used CUDA 13.3 and the current local
CUTLASS environment; full release-image build verification remains required.
Evidence: `aot-local-heads*.log`, `aot-native*.log`, `aot-memcheck.log`, and exported
manifests in `~/.cache/ds41rt-v6-heads32/`.

This completes the native kernel/backend foundation, not TP2 serving. Rust
bindings, local KV replicas and their lifetime/restore handling, projection
partitioning, lane integration and end-to-end measurements remain outstanding.
No serving defaults or performance claims change here.

Rust now exposes a separately prepared `v41_sparse_attention_heads32` binding,
with local-head query/sink/output validation and half-sized split scratch. Compact
single-request calls require explicit replay bounds (zero bounds permit the full
window); both WMMA and optional AOT batch symbols are selected before replay.
Batch graph identity includes local head geometry while preserving existing
64-head keys, and launch rejects a batch prepared for the other geometry. The
original 64-head constructor and static scratch helper remain compatible.
Two CPU contract tests pass for exact buffer extents, wrong-device rejection,
scratch limits and graph identity. The daemon compiles against the updated FFI.
These tests do not establish Rust-to-GPU execution or replicated KV correctness;
those require the upcoming serving ownership integration. Evidence:
`rust-tests.log` and `rust-daemon-check.log` in the compact-head evidence bundle.

### Compressed KV replica storage

Added an explicit peer `SourceReplica` owner that reuses the authoritative page
allocator's physical IDs. It allocates FP4 values/scales (288 bytes per compressed
row), page tables and committed lengths; index keys/scales remain on the source
owner. Replica allocation is allowed only before the source pool is populated,
so retained snapshots cannot silently acquire uninitialized peer storage. Logical
capacity is unchanged and the replica keeps no additional page references.

Append copies cover accepted row ranges and copy-on-write replacement tails,
then changed page-table entries, then lengths. The operations take an external
peer stream and perform no host synchronization or allocation; the caller must
order source writes, preserve reservations and drain both sides on errors.
Separate operations copy freshly RAM-restored pages and attach/reset slot metadata.
These are explicit unsafe ownership/ordering contracts, not yet serving hooks.

A hardware test passes in both GPU directions against the v5 native copy API:
255-row prefix, shared snapshot, append crossing into row 258 with copy-on-write,
unchanged retained owner, and a 300-row host-restored prefix. It checks peer value
and scale bytes, page tables and committed lengths. The CPU allocation-budget
check also passes. The temporary test executable was removed from the running
container. Evidence: `replica-hardware.log` and `replica-build.log` in the compact
attention evidence bundle. Independent-lane event ordering, cancellation handling,
SWA/proposal replication, memory-plan integration and serving remain required.

### Independent peer publication and DMA scheduling finding

A lane-owned `PeerPublication` preallocates a peer stream and two events. It
orders producer writes, peer copies and producer completion without host polling,
allocation or waiting during normal enqueue. The existing producer completion
and cancellation guard can then cover the round trip. Enqueue failures drain the
peer stream before releasing callback captures; each lane uses a separate owner.

The held-producer test exposed a hardware scheduling dependency with peer DMA:
with bidirectional peer access enabled, GPU1-to-GPU0 copies queued behind a stalled
lane prevented the other lane's copy from completing within three seconds.
Standalone CUDA probes reproduced the dependency; one-way peer access and
SM-issued peer reads did not reproduce it. This is an artificial dependency test,
not evidence of a three-second serving delay or an end-to-end throughput gain.

Added a preinitialized SM peer-copy kernel and Rust binding, and switched the new
compressed replica storage to that copy path. Its vectorized 16/4-byte paths and
byte fallback use 64-bit offsets; no DMA operation sits behind the lane-local wait.
The bidirectional hardware test now passes for independent-lane progress, injected
enqueue failure and dropping an outstanding publication. Replica append/COW/restore
checks also pass with producer completion covering peer publication. A native
48-case alignment/guard/graph-replay test passes under Compute Sanitizer with zero
errors. The test skips when fewer than two CUDA GPUs are present.

Rust hardware tests used a temporary CUDA 13.2 peer-kernel module linked to the
running v5 native library; the standalone sanitized test used CUDA 13.3. Temporary
container files were removed and the serving health check remains successful.
Evidence is in `publication-sm-hardware.log`, `replica-publication-sm-hardware.log`,
`peer-copy-memcheck-final.log`, and `publication-probe*.log` in the compact attention
bundle; failed DMA attempts are retained there. Full native release build and
serving integration remain outstanding. Existing non-replica peer DMA paths were
not changed; their sensitivity to this queue dependency is a profiling opportunity,
not grounds for changing defaults without measurements.

### Peer SWA ring storage

Added `WindowReplica` with peer FP8 value/scale rings and committed ends. It is
bound to the authoritative window owner and validates request generations.
Accepted commits copy only the changed suffix, capped at the last 128 rows;
wraparound needs at most two spans. End publication follows payload writes via
the SM-copy/event path. Restores copy only initialized ring positions, and empty
bounded replay publishes its end without copying stale ring contents. The shared
ring-span iterator now uses fixed inline storage instead of allocating a Vec.

The bidirectional GPU fixture passes short prefixes, wraparound at 128, an append
longer than the ring, retained-prefix restore to another slot, stale/foreign lease
rejection and an empty replay beginning at token 900. Poisoned untouched bytes
remain unchanged. The existing native prefix/slot-reuse test and CPU ring-span
checks also pass. Evidence: `window-replica-hardware.log`,
`window-prefix-hardware.log`, and `window-replica-build.log` in the compact attention
bundle. Temporary container fixtures were removed. This remains an explicit
storage component: private proposal replication, attention-wave integration,
startup budgeting and full serving qualification are still outstanding.

### Lane-owned private proposal replicas

Added preallocated `ProposalReplica` storage for FP8 windows and FP4 compressed
proposals. It preserves the producer's physical offsets and stride-one/two layout,
copying the bounded referenced span through SM peer reads. Stride gaps are copied
without changing metadata; attention still masks them through its original logical
indexing. Each producer wave should retain its own replica across consuming layers,
so this component neither duplicates committed history nor requires per-layer
full-cache copies. Empty spans enqueue no payload copy. Normal copying allocates
no buffers and uses the existing lane publication contract.

CPU checks cover capacity and overflow rejection. A bidirectional GPU test passes
for both formats, stride-two spans, narrow overwritten spans, empty spans, untouched
poisoned rows and stable destination addresses. Evidence: `proposal-replica-build-final.log`
and `proposal-replica-hardware.log` in the compact attention bundle. Temporary
container fixtures were removed. Binding these storage components to proposal
lifetimes, compact attention waves and graph replay is the next integration step;
there is still no TP2 serving performance claim.

### Compact attention execution wave

The execution wave now has compile-time 64-head and 32-head aliases. Existing
callers retain the 64-head alias; compact waves allocate half-sized query/output
and split scratch, retain the same descriptor/metadata capacities, and select the
compact Rust/native binding. Compact single-request execution always supplies
replay bounds, including zeros. Query-copy entry points reject mismatched head
extents before copying, and graph outputs use the wave's local-head row width.

A Rust-to-native hardware fixture passes on each RTX: compact workspace reservation,
a two-request batch with different row counts, graph capture and replay, exact
closed-form BF16 outputs, and changed proposal payloads observed by the same graph.
The unchanged 64-head reservation and committed-prefix/private-boundary hardware
checks pass too; CPU checks confirm the original 64-head memory formula exactly.
This fixture uses the new WMMA compact module linked to the v5 library; optimized
AOT was qualified separately above. Evidence: `compact-wave-hardware.log`,
`compact-wave-build.log`, `full-wave-reservation.log` and `full-wave-prefix.log` in
the compact attention bundle. Temporary container files were removed. Live model
query splitting, replica/proposal ownership binding and dual-wave scheduling remain
required before an end-to-end attention experiment.

### SM strided copies for attention boundaries

The preinitialized copy binding now supports pitched rows on either the local GPU
or a peer. This allows full-width query rows to be split into compact head rows
and compact outputs to be gathered for the existing full-width projection without
using peer DMA. Widths, pitches and offsets use 64-bit arithmetic with checked
buffer extents. Row count remains a runtime launch argument; capture performs no
allocation or kernel resolution.

Proposal replication now uses this primitive to skip stride-two gaps entirely,
superseding the bounding-span gap copies described above. Physical offsets and
metadata remain unchanged. The Rust GPU fixture confirms that gap rows retain
poison, narrow updates affect only their rows, and empty spans leave storage alone.
The native sanitized suite passes 72 cases, including two-way head split/gather,
local splitting, graph replay, byte/word/vector alignments, guards and invalid
pitch rejection. It includes 32,768-byte head halves at row counts 1, 7 and 64.
No memory errors are reported. Evidence: `peer-rows-memcheck-final.log`,
`proposal-strided-hardware.log` and `peer-rows-rust-build.log` in the same bundle.
Temporary container fixtures were removed. These are prerequisite copy operations;
live-model attention splitting/gathering and throughput measurement remain pending.

### Live query and window proposal adapters

Compact waves now accept an original full-width `AttentionQueryOutput`, validate
its layer, token order and selection origin, and enqueue a pitched head-half copy
into their own query storage before attention. This avoids fabricating a compact
query object whose other tensors still belong to the original producer. The copy
uses the existing preinitialized SM primitive and the wave's own stream; callers
must establish producer readiness with lane-local events.

Window replicas can now construct a peer proposal while retaining its original
request, lease, snapshot, offsets and token extent. The adapter validates the
authoritative lease, layer and committed range before substituting local storage.
Both adapters preserve the existing asynchronous owner-retention contract.
`cargo check -p ds41rt-daemon` passes (see `split-query-check.log` in the compact
attention evidence bundle). These adapters are not yet exercised by serving:
compressed-source and selection views, scheduler ownership/publication hooks,
full-model correctness and performance qualification remain outstanding.

### Compressed-source and selection peer views

Added an attention-only compressed proposal view that retains the authoritative
source lease, request, snapshot and strided metadata while using replica FP4
payloads. Index keys remain on the producer device. A selection peer view retains
the original query and source bindings and omits producer-only candidate blocks.
The CPU selection test verifies that the correct query remains valid and a new
snapshot at the same layer is rejected after creating the peer view.

Connecting these views exposed a proposal-copy extent mismatch: live producers
expose used rows, while replicas reserve maximum batch capacity. Copies now check
the referenced source extent rather than requiring the full reserved capacity.
The bidirectional GPU fixture passes both FP8 and FP4 with shortened producer
views, rejects one-byte truncation, preserves stride gaps and handles empty views.
Evidence: `peer-views-check.log`, `peer-views-tests.log` and
`proposal-used-rows-hardware.log` in the compact attention bundle. Temporary
container files were removed; serving health remains successful. End-to-end
publication/ownership integration and throughput qualification are still pending.

### Dual attention wave and output gathering

Added a lane-owned pair of compact attention waves, preinitialized copy kernels,
a peer completion event and full-width gathered output. Both halves are submitted
before cold preparation waits; warm execution uses a peer event and SM pitched
copy to join only this lane's two head halves. The original device receives the
64-head output for the existing projection. Proposal identities must match across
the halves. Cancellation drains both waves before owners can be reused, and CUDA
device scope is restored on each async poll. Budgets include both compact waves
and the full-width gathered output; cache replicas remain separately budgeted.

The hardware fixture passes with either GPU owning the output, four causal rows,
different sink weights per half, changed FP8 payloads, cancellation and subsequent
owner reuse. It checks closed-form BF16 results across every head and dimension.
This fixture directly submits native attention inputs: full query/proposal
enqueue, cold preparation and backbone integration still need combined coverage.
Evidence: `dual-wave-build.log`, `dual-wave-native-build.log`, and
`dual-wave-hardware.log` in the compact attention bundle. Temporary container
files were removed and serving remains healthy. No throughput claim or default
configuration change follows from this component check.

### Accepted SWA commits publish their peer replica

Window producer waves can now opt into a shared window replica with their own
publication stream/events. The existing accepted-commit path queues peer suffix
and end copies after native ring writes; its normal completion, abort and drop
paths therefore cover peer publication too. Different producer waves retain
independent publication owners. Foreign cache owners are rejected before writes.
Unconfigured waves retain the original path and allocate no publication objects.

The checkpoint-backed queued-window fixture passes both the unchanged path and
replicated commits. Two request slots use independent producer waves; after each
normal commit, peer FP8 values/scales and committed ends match the authoritative
cache. The fixture also checks mismatched acceptance, abort/revocation and the
other slot remaining valid. Evidence: `window-commit-replica-build.log` and
`window-commit-replica-hardware.log` in the compact attention bundle. Replica
creation/budgeting at serving startup, restore publication and the analogous
compressed-cache commit hooks remain required before enabling TP2 attention.

### Accepted compressed-cache commits publish replicas

Compressor waves now optionally retain the source replica and their own peer
publication owner. Native index/FP4 writes, copied shared tails and page-table
uploads precede replica publication; normal commit completion includes the peer
copies before applying page claims and advancing host slot versions. The original
commit path remains unchanged when no replica is configured. Replica owner and
producer device/library are validated before any writes.

The checkpoint-backed fixture passes original and replicated commits for layers
2 and 20 (both compression ratios), independent request slots, partial acceptance,
pending carry and abort/revocation. Peer FP4 values/scales, page IDs and committed
row counts match after the regular commit completes. The bidirectional replica
append/COW/host-restore fixture also passes. Evidence:
`source-commit-replica-build.log`, `source-commit-replica-hardware.log`, and
`source-commit-replica-cow.log` in the compact attention bundle. Serving startup,
restore/reset publication and the backbone dual-attention switch remain required;
these checks do not establish an end-to-end TP2 speedup.

### SWA replica reset and retained restore integration

Window state can now allocate/retain its replica before admission and share it
with producer waves. Fresh slot admission resets the peer end; bounded replay
publishes its empty frontier. Retained-prefix restore automatically republishes
initialized ring spans and the end on the caller's stream. Prefix streams on the
replica GPU copy directly; streams on the authoritative GPU use state-owned
publication events. Ordinary commit waves still use their own independent events.
A replicated state rejects commits from a wave not configured with its replica.

The GPU fixture passes both ownership directions and restores issued on either
device, plus wraparound, slot reuse, stale lease rejection and empty replay at
token 900. The checkpoint-backed original/replicated commit tests also pass with
state-owned replica configuration. Evidence: `window-restore-replica-build.log`,
`window-restore-replica-hardware.log`, and `window-state-replica-commit.log` in the
compact attention bundle. Temporary fixtures were removed and serving is healthy.
Compressed-cache restore/reset integration, startup budgeting/configuration and
backbone scheduling still remain before full-model TP2 qualification.

### Compressed replica reset, retained attach and RAM publication

Source caches can now own their replica and share it with producer waves. Reset
installs zero peer length; attaching a retained prefix installs peer page IDs and
length before publishing host references. Replicated states reject unconfigured
commit waves. Neither operation creates a second logical page pool.

The RAM-cache restore path publishes newly uploaded FP4 pages before exposing its
rebuilt prefix. `RestoreOutcome::Done` is emitted only after the host-cache engine
wait completes (`ds41rt-hostcache/src/cache.rs`); a preallocated peer stream then
copies and drains those pages, including on copy error. This is admission work,
not a decode-loop host wait. Retained GPU prefixes need only metadata attachment.

The bidirectional GPU fixture passes append/COW, retained attachment, reset and
publication of pages filled as a simulated host upload. Checkpoint-backed original
and replicated compressor commit tests also pass with state-owned configuration.
Evidence: `source-restore-replica-build.log`, `source-restore-replica-hardware.log`,
and `source-state-replica-commit.log` in the compact attention bundle. Actual RAM
eviction/restore with replicas still needs serving-level coverage once startup
and dual-attention scheduling are wired. Temporary fixtures were removed and the
running server remains healthy; defaults are unchanged.

### Replicated backbone cache allocation and producer configuration

The backbone bank now has a per-GPU replicated budget and constructor covering
all 40 FP8 windows and four FP4 source pools. Replica payloads and metadata are
charged to the opposite GPU; index keys and compression carry stay on their
original GPU. Logical source capacity is unchanged. Budget rejection precedes
allocation, and allocation failure drops the private partially built bank.

The bank can bind fresh producer waves to its replica owners. Distributed serving
now performs this binding for each lane after cache construction; with the current
nonreplicated constructor this allocates no publication owners. The replicated
constructor is not yet selected by serving: CLI/pool-sizing integration and dual
attention execution are still required together before enabling it.

GPU tests pass original and replicated banks with boundaries at layers 20 and 14,
under-budget rejection, preserved logical source capacity, two cycles of admission
and release, and zero ends on both GPUs after reset. Daemon build checks pass.
Evidence: `replicated-bank-build.log`, `replicated-bank-check.log`, and
`replicated-bank-hardware.log` in the compact attention bundle. Temporary fixtures
were removed and serving remains healthy. No performance result is implied.

### Committed-prefix peer views during follower append

Integration exposed a distinction missing from the initial peer adapter: replay
and encoder attention may use a committed-only source view whose query begins
before the committed frontier, including while a follower reserves an append.
Such views now retain an explicit committed-only marker. The peer adapter keeps
their original causal range and snapshot, uses replica backing for the empty
private overlay, and permits the same bounded committed-prefix reads as the
authoritative path. Ordinary private-proposal views still require an idle slot.

The checkpoint-backed compressor fixture passes both ratios with a peer committed
view created while the follower commit is pending. It checks unchanged binding,
metadata and row extent, peer-local empty-overlay buffers, rejection beyond the
query range, and continued rejection of ordinary views during the write. Existing
commit/abort checks also pass. Evidence: `peer-committed-build.log` and
`peer-committed-hardware.log` in the compact attention bundle. This validates the
adapter contract, not concurrent full-model attention results; combined dual-wave
and private-proposal integration remains outstanding.

### Lane-owned peer attention inputs

Added a reusable peer-input owner with window/source proposal buffers and selected
index storage. It constructs peer cache views from the original bank/proposals,
preserving request identities and token metadata. The peer request list uses fixed
stack storage. Source copies are reused across consumers only when all original
request bindings match; a new producing execution invalidates the copy key.
Window rows retain physical offsets, compressed rows retain their stride, and
committed-only source views require no private payload copy. All copy work uses
the caller's peer stream and its existing completion/lifetime contract.

A checkpoint-backed fixture passes layers 2 and 20 on opposite GPUs, both
compression ratios, byte-for-byte proposal payload comparison, preserved bindings
and metadata, repeated consumption and changed producer execution. Evidence:
`peer-inputs-check.log`, `peer-inputs-build.log`, and `peer-inputs-hardware.log` in
the compact attention bundle. Temporary fixtures were removed and serving remains
healthy. Combining these views with live selection and the dual wave is the next
step; this fixture does not establish full-model correctness or throughput.

### Combined cached dual-attention submission

The dual-wave owner now owns its peer proposal/selection buffers and sink half.
One guarded submission copies those inputs onto the peer attention stream, splits
the original query, and queues both head halves. Partial failure and cancellation
drain both streams before external owners can be released. Request views use fixed
stack storage on both sides, and device budgeting includes the new input buffers.

The checkpoint fixture now produces real queries and learned index selections,
then compares full-head attention with the combined dual path for layers 2 and 20.
It covers both GPU placements/compression ratios, cold preparation, warm replay,
changed producer inputs and cancellation followed by immediate reuse. Eight
comparisons match byte-for-byte using the WMMA kernels and a zero sink fixture.
This does not exercise the complete backbone, production sink tensors or optimized
AOT in the combined path. Evidence: `dual-cached-build.log`,
`dual-cached-native.log`, and `dual-cached-hardware.log` in the compact attention
bundle. Serving scheduling, projection continuation and end-to-end qualification
remain required before a performance/default decision.

### Projection continuation after dual attention

Dual completion now accepts a same-stream consumer before its final cooperative
wait. Projection can queue immediately behind the gathered output without a host
wait between attention and projection. The existing pending owner drains both
attention halves and downstream work on callback errors or cancellation. This is
the continuation mechanism; backbone FFN state integration remains outstanding.

The combined checkpoint fixture now compares both output-projection stages as
well as attention. All eight results match byte-for-byte across layers 2 and 20,
changed inputs and repeated execution. It also injects an error after projection
enqueue and verifies immediate reuse. Evidence: `dual-continuation-build.log`,
`dual-continuation-native.log`, and `dual-continuation-hardware.log` in the compact
attention bundle. The fixture uses WMMA attention with zero sink weights and the
checkpoint's FP8 projection weights. Temporary fixtures were removed and serving
remains healthy. Full-model serving, captured projection/FFN continuation and
performance qualification remain required.

### Backbone FFN handoff

The distributed backbone now routes optional dual attention through its existing
pending-FFN owner. The dual owner retains queued submission metadata until that
lane completes or cancels it, rejects duplicate submissions, and queues projection
plus block FFN preparation before its final wait. Full-head mode retains its
original captured-tail path. Enabling dual mode before execution replaces the
full-head workspace instead of keeping a second unused allocation; K7 reservation
and graph-shape configuration dispatch to the selected owner.

Daemon/test builds pass. The checkpoint attention/projection fixture now uses the
same owned-submission handoff and retains byte-for-byte results, duplicate-submit
rejection, callback-error cleanup and reuse. Evidence: `dual-backbone-build.log`,
`dual-backbone-check.log`, and `dual-backbone-hardware.log` in the compact attention
bundle. This fixture still stops after projection: the new FFN handoff and full
backbone need execution coverage. Serving does not yet select dual mode; startup
flags, replicated pool sizing and end-to-end correctness/performance remain.
Temporary fixtures were removed and the running server remains healthy.

### Experimental serving switch and replicated pool sizing

`--tp2-attention` now selects the combined attention path for `--rtx-gpus 2`.
Startup replaces each backbone lane's full-head workspace before K7 reservation
and final KV planning, then builds the replicated bank and binds both producer
lanes. One-RTX startup rejects the flag before loading the native library. The
flag is opt-in; launch defaults remain unchanged.

Both the expert-placement reservation and final pool planner use physical replica
bytes per GPU. Nominal `--kv-pool-size` and logical token capacity keep their prior
meaning; replicas do not add tokens. Automatic sizing may shrink to the tighter
card, while an explicit pool request fails instead of silently shrinking. Startup
logs identify whether attention replication is enabled.

Daemon build checks and all nine distributed memory-planner tests pass, including
unchanged nonreplicated sizing, constrained replicated sizing, physical-byte bounds
and explicit-size rejection/acceptance. Evidence: `tp2-startup-check.log` and
`tp2-startup-tests.log` in the compact attention bundle. This makes the experimental
path launchable in code, but no serving run or full-model qualification has yet
validated it. A current native build and actual startup/FFN execution checks are
next, before throughput experiments or any default decision.

### Current native build and checkpoint sinks

The coordinator native library now builds from current source with the release
feature set, including full/TP2 experts, FP8 projections, EXL3 modules and both
64/32-head attention AOT exports. The optimized Rust serving binary also builds.
This development build uses host CUDA 13.3; release-image construction and
qualification remain separate. Parallel exporters initially exhausted free GPU
memory beside the running v5 server; serial export/build completed successfully.

The combined checkpoint fixture now loads the actual learned attention sinks for
layers 2 and 20. Against the newly built native library, all eight attention and
projection comparisons remain byte-for-byte equal, including cancellation,
callback failure and immediate reuse. This verifies the current native artifact
with real sink weights, but does not establish every AOT dispatch shape or
full-model serving correctness/performance. Evidence in the compact attention
bundle: `native-configure.log`, `native-production-serial-build.log`,
`serving-release-build.log`, `checkpoint-sink-test-build.log`, and
`production-checkpoint-sinks.log`. Native build artifacts are outside the source
checkout at `~/.cache/ds41rt-v6-native`. Full-model startup/FFN execution and
retained-context checks are next.

### First full-model TP2 attention serving check

Full-model startup exposed an index handoff bug: layers 0 and 1 use SWA only,
but the experimental dual path tried to read an index selection for them. The
handoff now skips selection for these layers, matching the existing full-head
path, and explicitly requires an index lane for later layers.

With the current optimized binary/native build, TP2 attention now serves three
concurrent smoke requests correctly (READY, Python sum-of-squares, and retrieval
of MAPLE-7421). Repeating the retrieval request hits all 804 prompt tokens in the
GPU prefix cache. This exercises the full backbone/FFN and dSpark-enabled serving;
it does not yet verify RAM eviction/restore under replicated attention or establish
a throughput comparison. The host launch also needs `DS41RT_NATIVE_LIB` set for
RDMA discovery, in addition to the serving CLI argument.

At 20 bottom-up RTX expert layers, 2048 prefill rows, concurrency 16 and 20 retained
entries, automatic sizing provides 7,190,016 usable GPU-resident tokens. Automatic
RAM sizing allocates 13 GiB pinned (including staging/snapshot overhead), giving
20,972,032 combined logical tokens, just above 20 * 1,048,576. Physical replica
bytes remain excluded from logical capacity. The successful warm-filesystem
startup took 13.08 seconds; this is a smoke observation, not a qualified loading
comparison. The original v5 coordinator was restored after testing.

Evidence: `~/.cache/ds41rt-v6-serving-tp2/{launch.json,server.log,check.log,smoke.json}`;
`first-server.log` records the discovered index bug. No default changed and no
TPS claim is made from these short checks.

### Replicated attention RAM restore under small-pool pressure

A targeted full-model run used TP2 attention, dSpark, 20 RTX expert layers,
concurrency 2, two retained entries, a 6 MiB nominal GPU KV/index pool (13 source
page groups; 6,656 gross token positions before admission/COW allowances), and
256 MiB pinned RAM with 8 MiB chunks. Eight distinct 1,899-token documents forced
GPU eviction; revisiting documents 0, 1 and 2 returned the correct unique
identifiers with all 1,899 prompt tokens reused for each. Fresh counters confirm
three RAM restores, 24 device evictions, zero restore failures/timeouts and zero
failed stores. This exercises restoration/publication to both replicated caches
through full-model serving rather than only a cache fixture.

The test accounts for `/v1/stats`' one-second publication cadence by waiting and
waking the scheduler before reading counters. A preliminary 256 MiB host pool
with a single default-sized chunk could not cover the separate slab classes;
8 MiB chunks made the deliberately small test pool usable. The large automatic
production budget is unchanged. The original v5 coordinator was restored.
Evidence: `~/.cache/ds41rt-v6-replica-restore/{launch.json,server.log,check.log,smoke.json}`.
This targeted check is not another release-wide or randomized torture suite.
Throughput comparisons and replicated concurrent/cancellation coverage remain.

### First attention-only throughput comparison

Same current optimized Rust/native build, 20 RTX routed expert layers, K7 dSpark,
2048 prefill rows, concurrency limit 16, 20 retained entries, exact 5 GiB nominal
KV/index pool (6,003,200 usable tokens), and automatic RAM backing in both arms.
Only `--tp2-attention` changed. Both cards had 400 W power limits and standard
13365 MHz memory clocks. Warm release-corpus code/topic prompts used disabled
thinking and their existing output limits. Three measurements followed a warmup
at each workload/concurrency. Reference ran before candidate; this is exploratory,
not alternating final release qualification or retained-long-context coverage.

| Workload | Concurrency | Full-head decode tokens/s per request | TP2 attention | Change |
| --- | ---: | ---: | ---: | ---: |
| Code | 1 | 168.9 | 159.4 | -5.6% |
| Code | 4 | 119.4 | 109.7 | -8.1% |
| Topic | 1 | 96.1 | 90.2 | -6.2% |
| Topic | 4 | 68.9 | 63.5 | -7.8% |

Values are medians of each sample's median request decode rate. C4 aggregate
output/wall-clock throughput was 476.5 -> 439.3 tokens/s for code and
275.3 -> 252.4 for topic. Every paired response text and completion-token count
matched. No default is promoted. The dual path still launches projection/FFN
continuation outside the captured attention graph; investigate that known extra
launch work and peer-copy costs before deciding whether the attention split can
win. These timings do not isolate the cause of the regression.

Evidence: `~/.cache/ds41rt-v6-attention-perf/` contains both exact launch commands,
server logs, per-request SSE measurements, scripts and `summary.json`. Normal
v5 serving was restored after the experiment. TP2 projection and dSpark still
require independent implementation/evaluation.

### Capture the split-attention continuation

The dual owner now retains bounded per-layer projection/FFN continuation graphs,
keyed by the consumer storage/weight identity and row count. Warm replay refreshes
positions and host block state, then launches the graph after the gather on the
same owner stream. Cold tails warm, drain, restore unpublished block state,
capture, and replay before exposing FFN output. Cancellation drains both halves;
graph destruction occurs before consumer storage can be released. The full-head
path is unchanged, and no cross-lane dependency is introduced.

The optimized build passes, and full-model C2/C4 counting output parity plus
cancellation/replacement checks pass. Three warm code/topic measurements at C1/C4
also match the previous full-head response texts and completion-token counts.
Compared with the preceding uncaptured TP2 run, per-request median decode rates
changed: code C1 159.4 -> 161.5, code C4 109.7 -> 111.6, topic C1 90.2 -> 91.3,
topic C4 63.5 -> 64.9 tokens/s (roughly +1–2%). These separate short runs do not
establish a small gain beyond noise. They remain roughly 4–7% below the earlier
full-head arm, so TP2 attention remains off by default. The launch overhead was
not the complete explanation; peer-input/publication/gather costs and longer
retained-context attention still need investigation before a final judgment.

Evidence: `~/.cache/ds41rt-v6-tail-capture/` contains the exact launch, concurrency
results, performance requests/SSE results, server log and summary;
`~/.cache/ds41rt-v6-heads32/tail-capture-build.log` records the build. Normal
serving was restored after the experiment.

### Retained-context attention-only comparison

Using the same-build 20-layer/5 GiB logical-pool configuration above, the release
retained-context harness ran only code and topic at exact 32,768 and 131,072
parent tokens, three samples per case/arm. Both arms used identical request
messages, tokenizer, context source and corpus subset. All 24 samples completed,
reused the expected parent frontier, and passed applicable objective checks
(code); prose quality was not assessed.

| Retained context | Case | Full-head median decode tokens/s | Captured-tail TP2 | Observed change |
| ---: | --- | ---: | ---: | ---: |
| 32,768 | Code | 147.1 | 146.6 | -0.4% |
| 32,768 | Topic | 81.7 | 79.9 | -2.2% |
| 131,072 | Code | 148.4 | 136.8 | -7.8% |
| 131,072 | Topic | 85.4 | 76.1 | -10.8% |

Unlike the short-prompt trial, no paired output was text-identical, and topic
completion lengths differed materially (32K: reference 284/299/296 tokens,
TP2 242/217/176). Consequently these are observed workload rates, not an isolated
speedup measurement. They establish no reason to promote attention-only TP2.
Single cold-parent first-output observations were 4.79 -> 4.85 seconds at 32K and
17.91 -> 18.50 seconds at 128K; these are not qualified prefill medians.

Evidence: `~/.cache/ds41rt-v6-attention-context/` includes exact launches,
corpus subset, both complete harness reports and `summary.json`. Normal serving
was restored. Attention stays opt-in; next implementation work is independently
selectable query/output projection parallelism, followed by dSpark parallelism.
Splitting projection work could reduce weight streaming and full-head intermediate
traffic, but neither a gain nor a default change is presumed.

### Projection output-channel shard kernels

The coordinator FP8 export/build now includes query-B [1280 -> 16384] and
output-B [8192 -> 2560] kernels at all six native row capacities. These are
output-channel halves of the existing matrices: each retains the complete
reduction dimension, allowing concatenation without a cross-GPU partial-sum
reduction. Native scale packing accepts the two additional geometries. Existing
full-width kernels and serving defaults are unchanged.

`qualify-ds41-projection-shards.py` loads real layer 2/20 checkpoint weights,
compares full projections with concatenated output halves on GPU0/GPU1, and
replays captured graphs after changing input values. All 32 cases pass at rows
1, 6, 64 and 129 (capacities 1/16/80/256); 29 are exact and maximum absolute error
is 0.000030517578125 (absolute tolerance 0.0001, relative tolerance zero).
The complete native build also exports capacities 1024/4096, whose execution
coverage remains outstanding. Evidence: `projection-shards-build.log`,
`projection-shards-rebuild.log`, `projection-shards.log` and
`projection-shards.json` in the compact attention bundle.

This is a kernel prerequisite, not a serving implementation or performance claim.
Weight placement, lane-owned exchange/graph scheduling, query normalization and
output-stage integration, memory budgeting and independent runtime controls are
still required before projection TP2 can be evaluated end to end.

### Device-owned projection shard runtime

`v41_projection_tp2` now owns checkpoint output-channel shards for query-B and
output-B, with separate resident and load-peak budgets. The loader reads each
contiguous rank half directly and packs its local scales. The Rust FFI accepts
the two new matrix geometries. A lane wave preallocates each rank's input,
scratch and output storage plus the owner's gathered result; both rank launches
precede the owner's completion wait. Peer exchange uses SM copy kernels and
lane-owned streams/events. An interrupted submission drains both ranks before
storage reuse; no other lane participates in its completion decision.

The native-checkpoint hardware test runs both projection kinds, layers 2/20,
rows 1/6/16 with changed inputs, output placement on either GPU, two cooperative
lanes together, rejected insufficient budgets, and a one-poll drop followed by
immediate reuse. All 32 output comparisons against the original full-width FP8
projection pass at absolute tolerance 0.0001. Evidence in the compact attention
bundle: `projection-owner-check.log`, `projection-owner-test-build.log`, and
`projection-owner-hardware.log`.

The owner is not yet selected by serving. It currently launches projection and
exchange operations directly; graph integration, backbone query/output handoffs,
replacement of full-width weight allocations, startup budgeting/controls and
end-to-end performance measurements remain. No launch default changed.

### Query-A/norm to TP2 query-B and rotary handoff

The query wave now exposes a cooperative TP2 execution path: its normal
query-A/normalization prefix runs on the layer owner, both query-B ranks wait on
that producer event, and the gathered output receives rotary on the owner's
stream before the final completion wait. Prefix graphs are cached separately
from full-query graphs, so alternating modes cannot replay the wrong operations.
The projection wave supports an optional producer dependency and a same-stream
consumer callback; all dependency events remain lane-owned. Error/cancellation
cleanup drains queued producer and rank work before external owners can be reused.

Ten checkpoint cases (layers 2/20 on their respective devices, rows 1/6/16,
changed inputs/positions, warm prefix reuse, one-poll drop and immediate reuse)
match full query-B and rotated output exactly. Hidden/rank/position/frequency
buffers and published token identity also match their reference contracts. The
existing full-width query-preparation regression passes all 56 real-weight cases
bit-for-bit, including partial-producer failure and reuse. Build checks pass.
Evidence in the compact attention bundle: `query-tp2-check.log`,
`query-tp2-test-build.log`, `query-tp2-hardware.log`, and
`query-tp2-baseline-regression.log`.

This completes the component handoff, not serving selection. Full-width query-B
weights/workspace are still present in the fixture; they must be removed from
the split serving configuration. Backbone selection, output-B integration,
projection graph scheduling, startup budgets/options and performance evaluation
remain. No new throughput result or default change is claimed.

### Remove redundant full-width query-B storage

The split query configuration now loads only query-A and normalization weights,
with one packed-scale allocation and one projection scratch buffer. Query-B's
replacement shards remain owned by the TP2 projection runtime. Memory planning
uses the same split selection as loading; full-width callers keep their existing
allocation path. Rebinding between full and split waves is rejected, and the
full-width execution path rejects split weights.

All ten checkpoint query cases still match the reference exactly after removing
the redundant full-width weights, including warm prefix graphs, changed token
positions, cancellation and reuse. The full-width preparation regression also
reports all 56 real-weight cases bit-exact, including producer failure and reuse.
The daemon test targets compile. Evidence in the compact attention bundle:
`query-split-weights-build.log`, `query-split-weights-hardware.log`, and
`query-split-weights-baseline.log`.

This removes duplicate component storage; the distributed replacement shards
still consume memory. Serving selection and startup accounting remain to be
connected, so this is not a serving memory-saving or performance claim.

### Query-B TP2 serving selection

`serve-native --rtx-gpus 2 --tp2-query-projection` now selects checkpoint
query-B output-channel shards for all 40 backbone layers. This switch is
independent of `--tp2-attention`, does not require replicated KV, and defaults
to off. Query-A, normalization, rotary and the attention owner stay at their
existing placements. Both token entry and prepared-layer entry use the split
query handoff. Each target lane has separate rank streams, events and exchange
buffers; the immutable shard banks are shared. Output-B and dSpark are unchanged.

Startup omits full-width query-B weights and scratch, loads the replacement
shards, and creates all projection workspaces before measuring free memory for
expert and KV allocation. At capacity 2048 each layer-owner wave budgets
212,286,480 bytes on its output GPU and 78,068,752 bytes on its peer. There are
two such owners per target lane and two target lanes. These are gross replacement
workspace figures, not a net-memory comparison with the removed query-B scratch.

The serving smoke check passed three concurrent requests (READY, Python code,
and a retained identifier) and a repeat with all 804 prompt tokens reused. The
warm-filesystem startup reported 12.35 seconds; this single smoke observation is
not a load-speed qualification. With 20 local expert layers and the normal KV
setting, startup retained 14,680,064 usable GPU tokens plus 6,291,968 RAM tokens,
for 20,972,032 combined tokens, with 6.75 GiB pinned RAM. Evidence:
`~/.cache/ds41rt-v6-query-serving/{launch.json,server.log,smoke.json,check.log}`.
The daemon checks, test-target build and release build passed. Projection rank
and exchange operations still launch directly; their graph scheduling remains
an optimization candidate rather than a claimed completed optimization.

The first same-build query-only comparison used a fixed 5 GiB logical KV budget,
20 local expert layers, K7, capacity 2048, C16 serving, 20 retained slots,
automatic host cache, and stock 13,365 MHz memory with 400 W limits. Three warm
samples per point, thinking disabled, median per-request decode tokens/s:

| Workload | Concurrency | Full query-B | TP2 query-B, direct launches | Change |
|---|---:|---:|---:|---:|
| Code | 1 | 167.8 | 162.8 | -3.0% |
| Code | 4 | 118.0 | 115.7 | -1.9% |
| Topic | 1 | 94.5 | 95.1 | +0.7% |
| Topic | 4 | 67.2 | 68.3 | +1.8% |

Code and C1 topic outputs match exactly between arms. C4 topic varies in the
reference arm (259/264/274 output tokens versus 264 throughout the candidate),
so that row is not an isolated speedup measurement. The small topic differences
and variable C4 samples do not establish a performance winner; the option stays
off. Evidence: `~/.cache/ds41rt-v6-query-perf/` contains both launch commands,
server logs, per-request results, and the harness. Graphing the rank projection
and exchange work is the next query-path optimization to evaluate.

A second serving check exercised a 10,824-token prompt through capacity-2048
prefill chunks alongside the two short requests. All expected responses passed,
and the repeat reused all 10,824 prompt tokens. This exercises the large query
projection AOT path in serving, including chunked prefill and retained-context
reuse. Evidence: `~/.cache/ds41rt-v6-query-serving-wide/`. The standard coordinator
was restored after both smoke checks and the performance comparison.

### Graph replay for TP2 projection ranks and gather

Each projection rank now captures its FP8 plan against resident rank-local input,
output and scratch storage. External input copies and producer events remain
outside the graph, so changing the input allocation/device cannot replay a stale
pointer or dependency. The owner captures the two-half gather separately against
wave-owned storage. Cold execution completes once before capture records future
work; it does not replay the cold result. Graph caches retain small decode shapes
and one current large shape, and are drained and destroyed on the owning device
before their allocations are freed. Lane completion remains independent.

The 32 real-checkpoint projection comparisons pass with simultaneous lanes,
changed inputs, both output devices, cancellation/reuse, and assertions that the
same graph handles survive intervening decode shapes. All ten complete-query
comparisons remain exact. Test targets and the release binary build. Evidence:
`projection-graphs-build.log`, `projection-graphs-release-build.log`,
`projection-graphs-hardware.log`, and `projection-graphs-query.log` in
`~/.cache/ds41rt-v6-heads32/`.

The graph-enabled server also passed the concurrent short requests and the
10,824-token chunked-prefill identifier request; the repeat reused all 10,824
prompt tokens. Evidence: `~/.cache/ds41rt-v6-query-graphs-wide/`.

The repeated same-build comparison used the preceding query-only experiment's
fixed 5 GiB KV budget, launch settings and three warm samples per point:

| Workload | Concurrency | Full query-B | TP2 query-B with graphs | Change |
|---|---:|---:|---:|---:|
| Code | 1 | 166.3 | 165.6 | -0.5% |
| Code | 4 | 116.9 | 117.8 | +0.8% |
| Topic | 1 | 95.5 | 95.5 | +0.0% |
| Topic | 4 | 69.0 | 68.4 | -1.0% |

These are median per-request decode tokens/s with thinking disabled. Code and
C1 topic outputs match exactly. C4 topic varies slightly in the reference arm
(263/264 output tokens versus 264 in the candidate), so it remains a mixed timing
and output comparison. Observed power limits were 400 W, memory 13,365 MHz, and
neither GPU reported active hardware thermal slowdown at the sampled check.
Evidence: `~/.cache/ds41rt-v6-query-graphs-perf/` contains launch commands, logs,
and all request results. The candidate is now approximately tied with baseline;
this does not establish a default-worthy speedup. The earlier direct-launch
candidate was roughly 2–3% behind on code, but those separate trials are not a
paired direct-versus-graph estimate. Query TP2 stays opt-in. Output projection
and dSpark remain the next independent work items. The normal server was restored.

### Output-B TP2 serving path

`--tp2-output-projection` selects two output-channel halves of output-B,
independently of query projection and attention TP2. The inverse-rotary/grouped
output-A prefix stays on the attention owner and remains part of the attention
graph. Output-B ranks depend on that producer; their gathered result feeds the
owner's FFN continuation before the final projection completion wait. That
continuation has its own per-layer/shape graph bank. Each lane retains independent
streams, events and graph state. The legacy full-output path remains the default.

Split loading omits full-width output-B weights, scales and scratch. Resident
replacement shards and both target lanes' exchange workspaces are allocated
before expert/KV budgeting, as for query TP2. Full/split rebinds are rejected.

Ten complete-output checkpoint cases cover layers 2/20 on their respective GPUs,
rows 1/6/16, changed data/positions, graph reuse and dropped submission/reuse.
Input, grouped output-A, positions and frequencies match exactly. Seven final
output cases are exact; the other three differ at one or two elements by one
adjacent BF16 value. Investigation found the full M1 export uses split-K=2 while
the half-width M1 export uses split-K=1. The maximum absolute difference is
0.0078125 at a value around 1.1; this is one BF16 spacing, not an indexing or
producer mismatch. The check therefore requires finite values with absolute
error <=0.0001 or one adjacent BF16 value, while retaining exact intermediate
checks. Evidence: `output-split-hardware.log` and `output-split-diagnostic.log`
in `~/.cache/ds41rt-v6-heads32/`. Serving/performance validation follows below.

The output-only serving smoke passed the short concurrent requests and the
10,824-token chunked-prefill identifier request with complete prefix reuse.
After adding the FFN continuation graph, a second smoke enabled all three flags
(attention, query-B, output-B) together and passed the same requests/reuse.
Evidence: `~/.cache/ds41rt-v6-output-serving-wide/` and
`~/.cache/ds41rt-v6-projections-attention-wide/`. The combined run reported
6,077,440 usable GPU tokens plus 14,894,592 RAM tokens, totaling 20,972,032 tokens,
with 13.75 GiB pinned RAM. Its 13.72-second warm startup is a single smoke result,
not a load-speed qualification. At capacity 2048 each output projection wave
budgets 101,581,840 bytes on its output GPU and 80,610,320 on the peer; these are
gross workspace totals before accounting for removed full-width scratch.

The output-only same-build comparison retained the fixed 5 GiB KV budget,
20 local expert layers, K7, capacity 2048, C16 serving, 20 retained slots,
automatic host cache, 400 W limits and stock 13,365 MHz memory. Three warm
samples per point, thinking disabled, median per-request decode tokens/s:

| Workload | Concurrency | Full output-B | TP2 output-B + FFN graphs | Change |
|---|---:|---:|---:|---:|
| Code | 1 | 166.7 | 163.2 | -2.1% |
| Code | 4 | 118.1 | 115.2 | -2.5% |
| Topic | 1 | 97.1 | 93.8 | -3.5% |
| Topic | 4 | 69.6 | 68.5 | -1.6% |

All paired output texts and completion counts match in this comparison. No
performance winner is established; output TP2 stays off by default. Evidence:
`~/.cache/ds41rt-v6-output-perf/` contains both launch commands, server logs,
request results, GPU settings and harness. The standard coordinator was restored.
The daemon check, test-target build and release build pass. dSpark transformer
TP2 remains the next independent implementation/evaluation item.

### dSpark TP2 expert preparation and cross-device graph feasibility

The native draft has 128 routed experts per stage, top-3 routing and 2,304
intermediate channels. The first transformer split targets those routed weights:
1,152 intermediate channels per GPU with all expert IDs available on both ranks.
`V41ExpertSelection::DsparkTp2` reads contiguous W1/W3 row halves and W2 column
halves, with corresponding scale partitions and bounded caller-owned staging.
Each expert half stages 9,400,320 bytes with a 1,152-byte minimum row scratch.
Prefetch advises only the selected contiguous half or the full span needed by
strided column reads. Existing native/Spark paths remain unchanged.

The staging fixture passes both draft ranks at stage 2/expert 127, byte-for-byte
checks for all six tensors, offsets beyond 2 GiB, odd read batches, short-buffer
rejection with untouched sentinel storage, tail guards, and invalid stage/expert/
rank checks. The daemon still checks successfully. The AOT exporter now accepts
`--role dspark_tp2` with geometry 128 experts, 1,152 intermediate channels and
three routes. It consumes the existing FP8 K32 row representation and emits
ordered FP32 route planes. Capacities 1/16/80/256 compile at width 192; this is
export evidence, not an executed MoE or serving correctness/performance result.
Evidence: `dspark-tp2-staging-test.log`, `dspark-tp2-loader-check.log`, and
`dspark-tp2-export.log` in `~/.cache/ds41rt-v6-heads32/`, plus the manifest and
objects in `~/.cache/ds41rt-v6-dspark-tp2-aot/`.

A separate GPU experiment establishes a useful scheduling option:
`native/tests/v41_cross_device_graph_selftest.py` captures BF16 arithmetic on both
GPUs, event fork/join dependencies and bidirectional SM peer copies in one CUDA
graph. Three replays with changed input values produce exact expected results on
both GPUs. CUDA's ordinary `cudaMemcpyPeerAsync` rejected capture in the initial
probe; the existing SM peer-copy implementation succeeds. Evidence:
`~/.cache/ds41rt-v6-dspark-tp2-aot/cross-device-graph.json` (and the failed copy-API
probe in `cross_device_kernel_capture.json`). This validates the primitive graph
structure, not the complete draft transformer. The implementation should first
try retaining one captured draft-chain launch with GPU event dependencies,
rather than introducing host waits between the three draft stages. It also gives
a reason to revisit the separate projection launches after dSpark integration.

Native module/FFI ownership, two-rank weight loading, routing/input broadcast,
ordered reduction and draft-chain integration remain. The draft reduction must
sum rank contributions before the existing per-route BF16 rounding and top-3
sum/shared addition; the backbone's six-route reducer is not interchangeable.
No dSpark TP2 serving flag or performance claim exists yet.

### dSpark TP2 native interface and reduction verification

The dual-RTX native build now includes separate draft TP2 expert modules for
capacities 1/16/80/256/1024/4096 alongside the existing backbone TP2 modules.
Rust validates role 4 (128 experts, 1,152 intermediate channels, top-3, FP8 K32
input) and can load each draft half through the existing weight packer. EXL3
draft TP2 is explicitly unsupported; existing EXL3 paths remain unchanged.

The ordered route reducer now supports two ranks and three routes: sum FP32
rank contributions, round each route to BF16, then sum routes and add the
optional shared expert output before final BF16 rounding. The hardware test
initializes all six draft/backbone capacities on both GPUs and checks 108
changed-input CUDA graph replays across rows 1/7/16, shared output on/off, and
existing one-rank/top-3 and four-rank/top-6 paths. All results are exact against
the CPU rounding oracle; the fixture also distinguishes incorrect premature
per-rank rounding. The daemon test build passes. Evidence:
`~/.cache/ds41rt-v6-heads32/dspark-tp2-native-hardware.log` and
`dspark-tp2-native-test-build.log` in the same directory.

This verifies module initialization and reduction, not expert GEMM execution
or serving performance. Pair-owned weights/workspaces, input broadcast and
full draft-chain graph integration remain before a serving comparison.

The draft TP2 rank/pair implementation now checks successfully with the daemon.
It alternates rank weight loads within each stage, owns legacy weights under
explicit device scopes, and preallocates per-lane rank scratch. The pair quantizes
BF16 input once on RTX1, broadcasts FP8 rows plus top-3 IDs/weights to RTX0 using
SM peer copies, launches both expert halves, and joins their FP32 route planes
through GPU events before ordered reduction on RTX1. Enqueue performs no host
wait or allocation. The API permits capture in the containing draft graph and
requires its owner to drain external streams and destroy graphs before releasing
workspaces. Workspace budgeting accounts separately for both GPUs.

This is compiled implementation only: checkpoint expert execution comparison,
whole-chain graph integration and serving measurements remain unverified.
Evidence: `~/.cache/ds41rt-v6-heads32/dspark-tp2-pair-check.log`.

### dSpark TP2 checkpoint execution verified

The actual checkpoint expert halves now execute in a single cross-GPU CUDA
graph and pass comparison with the existing full dSpark expert path. The test
loads all three stages, exercises rows 1/7/16/65/112/1, toggles shared expert
addition, and replays each graph with three changed BF16 inputs and top-3 route
sets. All 108 comparisons pass a 0.0001 relative RMS bound. Worst observed
relative RMS is 0.0000171 (0.00171%); all single-row comparisons are exact.
Larger batches have only sparse rounding differences, with maximum absolute
difference 0.0078125. This verifies expert execution, packing, peer broadcast,
rank reduction and graph replay, but not the surrounding transformer or serving
acceptance/performance. Resident routed-expert weight bytes are 3,609,722,880
on each GPU (all three stages).

Evidence: `~/.cache/ds41rt-v6-heads32/dspark-tp2-checkpoint-hardware.log`; durable
test: `v41_experts::dspark::tp2::checkpoint_tests::cuda_dspark_tp2_checkpoint_experts_and_graph`.
The normal v5 server was stopped for GPU memory and restored after the test.
Whole-chain integration and independent lane/cancellation validation remain.

### dSpark TP2 transformer integration and numerical investigation

The draft transformer can now construct TP2 routed-expert stages. Attention,
shared experts and other draft projections remain on RTX1. The loader admits
both GPUs before reading weights; per-stage/per-lane workspace and distributed
chain budgets include peer allocations. The new backend uses the existing
three-stage capture path, so a captured transformer graph contains both ranks
without adding host waits or cross-lane dependencies. No serving flag/default
is enabled yet.

The K7 same-backend control passes exact tokens/logits/confidence for sequential
execution versus two concurrently executing lanes at C1/C3/C8/C16/C3, including
cold execution, graph replay, changed seed/cache/order, cancellation, reuse and
device restoration. Evidence: `~/.cache/ds41rt-v6-heads32/dspark-tp2-lanes-hardware.log`.

Comparison to full experts is NOT yet qualified. C1 is exact; at C3, logits
have relative RMS 0.000002067 and maximum absolute difference 0.00038147.
At C8 one lane has relative RMS 0.00008805, maximum logit difference 0.03756,
maximum probability change 0.00049077 and worst-row total variation 0.001909.
Token IDs still match at that point, but the 0.001 total-variation diagnostic
bound fails, before C16 is reached. The sequential/concurrent TP2 control points
toward a numerical split effect rather than lane scheduling; stage-level
comparison and acceptance measurement remain necessary. The diagnostic test
continues to fail rather than asserting full-reference equivalence. Evidence:
`dspark-tp2-chain-hardware.log` in the same directory. Normal v5 serving was
restored after each test. No serving speed or acceptance claim follows.

### Draft split trace and serving experiment entry point

Stage tracing locates the C8 difference in one of 286,720 stage-0 expert output
values (absolute 0.001953125) despite identical normalized input, expert IDs and
routing weights. Later stages retain identical expert IDs at C8, but quantized
projections amplify the changed values. At C16, stage 0 changes three of 573,440
expert output values (maximum 0.00390625); stage 2 eventually changes three of
336 expert IDs, and draft token IDs differ. Thus the split is not bit-exact
with full experts, even though independent TP2 lane execution is exact against
sequential TP2. This is evidence of numerical propagation from the partition,
not proof of equivalent draft acceptance. The full-reference diagnostic remains
failing; it now collects floating-point discrepancies through later batch sizes
while retaining strict token checks. The same-backend lane/cancellation test
remains the scheduling correctness check.

An experimental `--tp2-dspark-experts` flag selects the native split in dual-RTX
serving and requires `--dspark`. It rejects EXL3 through the loader, preserves
the existing full path by default, and admits weights/workspaces against both
GPUs' currently free memory before loading. The serving load reserves the main
context owner and two draft waves. Serving smoke, content acceptance and decode
measurements must establish whether this option is useful; stage tracing does
not establish those properties.

### dSpark serving smoke, initial decode comparison, and recipe RAM default

The release build and opt-in TP2 draft serving smoke pass. Concurrent READY,
code and 10,824-token identifier requests return the expected answers; repeating
the identifier prompt hits all 10,824 cached tokens. Startup takes 12.417 seconds.
With 20 RTX expert layers, the automatic usable GPU pool has 10,147,840 tokens;
automatic RAM sizing adds 10,824,192 tokens, for 20,972,032 combined tokens.
Pinned RAM is 11,274,289,152 bytes (10.5 GiB), including overhead/staging.
Evidence: `~/.cache/ds41rt-v6-dspark-serving-wide/`.

The first same-build comparison holds 20 expert layers, 5 GiB GPU KV allocation,
K7, automatic host cache and other switches fixed. Three warm samples per cell:

| Case | Concurrency | Full draft experts tok/s/request | TP2 draft experts tok/s/request | Change |
|---|---:|---:|---:|---:|
| Code | 1 | 165.90 | 166.39 | +0.29% |
| Code | 4 | 117.17 | 118.78 | +1.38% |
| Topic | 1 | 95.68 | 96.75 | +1.12% |
| Topic | 4 | 68.59 | 69.01 | +0.60% |

These small noisy changes do not establish a performance winner. Code outputs
and token counts match across all samples; topic C1 also matches, but topic C4
varies. TP2 draft remains opt-in. Evidence: `~/.cache/ds41rt-v6-dspark-perf/`.
A focused code/reasoning-code/topic acceptance comparison is still pending.

The standard recipe now defaults `HOST_CACHE_BYTES=auto`, alongside 20 retention
entries. This sizes RAM to keep combined usable token capacity above retention
entries times the configured context limit, with separate overhead/staging
reservation. Explicit byte sizes or zero override auto sizing. The direct binary
continues to accept explicit `--host-cache-bytes auto`; this change is to the
recipe default. Shell syntax checks and all 12 native release launcher tests
pass (`recipe-auto-tests.log` in `~/.cache/ds41rt-v6-heads32/`).
