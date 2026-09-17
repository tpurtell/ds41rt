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
