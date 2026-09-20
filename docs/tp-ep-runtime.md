# Live daemon integration: official-native Spark TP×EP replicated experts

Status: **CPU integration implemented; no GPU, native build, container, service
or workload action was performed for this document.** Every live, numerical,
memory and performance statement below is either marked `PENDING` or explicitly
derived from source. The only passed evidence is the CPU test list in §10.

Owner: daemon agent (`rust/crates/ds41rt-daemon/**`, this file). Related:
[tp-ep-implementation-plan.md](tp-ep-implementation-plan.md),
[tp-ep-transport.md](tp-ep-transport.md), [tp-ep-scheduler.md](tp-ep-scheduler.md),
[tp-ep-loader.md](tp-ep-loader.md), [tp-ep-native.md](tp-ep-native.md),
[tp-ep-memory-review.md](tp-ep-memory-review.md).

## 1. Scope and unchanged defaults

This change wires the already-implemented topology/transport/scheduler/loader/FFI
foundations into the two live daemons:

- coordinator: `ds41rt serve-native` (`v41_native_serve.rs`,
  `v41_native_serve/distributed.rs`, `v41_experts/coordinator.rs`);
- worker: `ds41rt expertd-native` (`v41_experts/service.rs`,
  `service/local.rs`, `service/backend.rs`, `v41_experts/execution.rs`).

Only the official `deepseek-ai/DeepSeek-V4.1-Flash` native checkpoint is
accepted for an explicit topology. EXL3 and NVFP4 keep their existing separate
paths and are rejected when `--spark-tp/--spark-ep` are given, before any
allocation or readiness publication. No default, release image, `ds41rt.config`
value, dSpark/attention/local-RTX parameter or legacy TP4×EP1/EXL3-TP2 launch
argument changed. No legacy `commands/real_full` code was touched.

## 2. CLI contract

Both processes gain two opt-in flags:

| Process | Flags | Values |
| --- | --- | --- |
| `serve-native` | `--spark-tp N --spark-ep M` | `N ∈ 2..=4`, `M ∈ 1..=3` |
| `expertd-native` | `--spark-tp N --spark-ep M` | `N ∈ 2..=4`, `M ∈ 1..=3` |

- The pair is **all-or-none** (enforced by clap `requires` and again by
  `v41_spark_topology::resolve`). Absent keys emit exactly the legacy argument
  vector and the legacy behavior.
- The approved combinations are exactly the six `V41SparkTopology` constants:
  `TP2EP1`, `TP3EP1`, `TP4EP1`, `TP2EP2`, `TP3EP2`, `TP2EP3`; anything else is
  rejected by `V41SparkTopology::new`.
- `TP * EP` must equal the physical rank count: the coordinator peers list
  length, the worker `--world`. `group = rank / TP`, `tp_rank = rank % TP` are
  taken from `V41SparkTopology`; the daemon never re-derives executor ids.
- Worker `--rank`/`--world` ranges were widened to admit six ranks
  (`--rank 0..6`, `--world 2..=6`). Legacy worlds 2 and 4 behave exactly as
  before.
- Explicit `TP4EP1` is **not** the legacy path: it is topology-bound, so the
  request carries the ownership flag with every owner `0`. The legacy default
  (no flags) remains the canonical request contract with executor ids `1..=4`.

## 3. Topology admission (fail before allocation/ready)

`v41_spark_topology.rs` owns the resolution:

```rust
resolve(tp, ep, world, component) -> Result<Option<V41SparkTopology>>
require_native(topology, &catalog)   // rejects EXL3/NVFP4 for an explicit topology
group_of / tp_rank_of                 // thin wrappers over the shared topology
```

Both workers additionally require, immediately after loading the native library
and catalog and before any model allocation:

```rust
library.v41_compact_reducer()?.require_rank_count(topology.world_size() as u32)?;
```

so a library that lacks the 3/6-plane reducer (or a legacy world-2 library that
lacks the optional TP2 reducer) fails at startup with a specific message rather
than at the first reduction. `NativeTp4Wave::new` repeats the check before it
allocates any plane, covering direct construction paths. Legacy two-rank
behavior is unaffected except that a missing optional entry point now fails
earlier.

## 4. Worker path

`NativeExpertServiceConfig` carries `Option<V41SparkTopology>` plus the existing
`world`, `rank`, `first_layer`, `capacity` and budget.

- **Resident shard.** For an explicit topology the worker selects
  `ExpertLayer::BackboneReplicatedTp { layer, rank: tp_rank, world: TP }`; for
  `TP == 2/3` this maps to `V41ExpertSelection::BackboneTp` (loader
  `world = TP`, `rank = local shard`), and to the FFI interfaces `8`/`9` with
  native roles `5`/`6`. `TP == 4` keeps the legacy `Backbone` layer/role `1`.
  RTX TP2 (`BackboneTp2`, role `3`) is untouched. Every replicated group loads
  **all 384 experts** for every remote layer with its own TP shard.
- **Execution.** `ExpertExecution::install_native_group(group)` binds the worker
  to `topology.group(rank)` once at startup. Route unpacking becomes
  `copy_native_group_routes_into(ids, weights, group)`: owned routes keep their
  true expert id and exact FP32 gate weight, every other route is the `384`
  sentinel with weight `0`, which the grouped kernel maps to inverse `-1` and
  skips. A request whose ownership contract disagrees with the worker's bound
  topology is rejected; a legacy canonical request never reaches a
  topology-bound worker.
- **Roles 5/6 accounting.** `compact_output_role(1 | 5 | 6)` now gates the
  compact output plane, the compact reducer, and the decode/small scratch
  variants, so a TP2/TP3 worker plans and allocates exactly the scratch the
  interface exports. The timing histogram ignores the out-of-range sentinel.
- **Admission.** Legacy checks are unchanged (`resident + staging ≤ budget`,
  `resident + workspace ≤ budget`). An explicit topology adds the known
  Spark-side transients (§6). `plan` and `load` use the same `selection(layer)`,
  and `plan_execution` is computed from the same selection, so a TP2/TP3 worker
  can never be budgeted with TP4 scratch.
- **Ready handshake.** No readiness is published until all of the above passes.

## 5. Coordinator path

One `ReplicatedGroupPlanner` is owned per independent lane (per
`NativeTp4Wave`), constructed before decode begins. It holds a preallocated
`ReplicatedExpertScheduler` (384 experts, `group_count = EP`), a `[u32; 384]`
histogram and a `[u8; 384]` owner scratch. There is no allocation in the steady
request path and no cross-lane state.

Per remote request, `prepare_remote_request` → `BoundExpertRequest::assign_native`:

1. prove the bound request still describes its block layer;
2. build the histogram from the router's host-side route IDs (no new D2H; the
   router already captures them);
3. plan with `tie_seed = replicated_expert_tie_seed(layer_id, request_id)`;
4. copy `assignment()[expert].encoded()` into the owner scratch;
5. re-encode the request with `with_native_group_owners(&owners, topology)`,
   preserving canonical order, the six unique experts per token and the exact
   FP32 gate weights, and setting `V41_NATIVE_GROUP_REQUEST_FLAG`.

The single remote funnel (`v41_backbone_lane::execute_tp4` →
`prepare_remote_request` → `dispatch_ffn`) covers decode, encoder/prefill and
verification, so every remote path carries ownership. Local RTX layers never
build a remote request and are unchanged. The same request is sent to every
physical rank; each rank unowns routes outside its group.

Ownership is deterministic for a fixed `(histogram, layer, request_id)` but is
**not** promised bit-identical to the legacy TP4 grouping: `tie_seed` can move
equal-cost experts between groups, which changes the BF16 partial grouping and
therefore the summation order. Numerical comparison to TP4 therefore needs a
stated tolerance or an inference-quality check, not bitwise equality.

### 5.1 Assembly and device budget

`NativeTp4Wave` keeps one BF16 `[M, 5120]` plane per physical rank
(`N ∈ {2,3,4,6}`) plus one shared input plane and one output plane.
`device_bytes_for(capacity, ranks)` is exact:

```
capacity * (ranks * V41_PARTIAL_ROW_BYTES + 2 * 5120 * 2)
```

`device_bytes(capacity)` remains the four-rank forecast for legacy callers and
tests, so `TP2EP2` (N=4) is unchanged while `TP3EP2`/`TP2EP3` (N=6) reserve two
extra planes instead of inheriting the four-rank underestimate. Both serve
callers pass the real peer count, and the dual-RTX planner charges one such wave
per lane (two on GPU1) into the KV reservation. `upload_frames` capacity is
`capacity * N`.

Reduction contract: ranks 2 and 4 keep the legacy fixed entry points
(`reduce_tp2`, `reduce`); ranks 3 and 6 use the generic
`reduce_planes([ptr; 6], ranks, shared, output, rows, stream)`. All ranks return
directly to the coordinator; there is no group-local reduction. The shared
expert is added exactly once and there is a single final BF16 rounding. An empty
group still returns its (zero) plane and still participates in coverage.

## 6. Memory admission and diagnostics

The human decision for the experimental Spark budget is
`MemTotal − ~20 GiB OS reserve`, with the existing `--device-budget-bytes` (the
launcher's `SPARK_DEVICE_BUDGET_BYTES`) as the override. The daemon does not
hardcode any ceiling; the OS reserve stays outside the budget, and the known
application transients stay inside it, so they are not double counted.

For an explicit topology only, admission adds the source-known transients
(`spark_admission_budget`):

```
load_peak  = resident + staging + pinned_host + read_scratch + headroom
serve_peak = resident + workspace + exchange + row_indices + rings + headroom
```

- `pinned_host` / `read_scratch` come from the loader plan
  (`ExpertLoadBudget`); they are pinned/allocated for one layer load at a time.
- `exchange = HostExpertExchange::bytes_for(capacity)` (ids + routing +
  partials) and `row_indices = capacity * 4` live while serving.
- `rings` are the registered/mapped RDMA rings the worker pins for its two
  persistent endpoints (decode and prefill), mirroring the transport's
  `verbs_host_persistent_rings` rule exactly for the native compact-BF16 frame
  shape:

  ```
  depth = DS41RT_VERBS_HOST_RING_DEPTH (default 8, 1..=8)
  slot  = DS41RT_VERBS_HOST_RING_SLOT_BYTES (default 8 MiB)
  request_wire  = 96 + rows * (40 + 6*12 + 5280)
  response_wire = 96 + rows * (4 + 5120*2)
  capacity      = min(max(slot, wire), max_frame_bytes)
  span          = align_up(capacity, verbs-host alignment 4096) * depth
  rings         = DS41RT_SPARK_RDMA_ENDPOINTS (default 2) * (request_span + response_span)
  ```

  A request/response frame that exceeds `--max-frame-bytes` fails admission
  instead of failing at connect. At capacity 4096 this is ~0.96 GiB for two
  endpoints (~256 MiB at small capacities, where the 8 MiB minimum slot
  dominates). The endpoint count is **fixed at two** (decode + prefill); the
  obsolete `DS41RT_SPARK_RDMA_ENDPOINTS` override is accepted only at the value
  `2` (with a deprecation warning) and any other value fails startup, so a stale
  launch cannot silently under-reserve.

  The admission figure is not just a model: the worker enforces it when a peer
  connects. `accept_with_budget` validates the peer's advertised ring geometry
  with the transport's own `from_wire` rules, then charges the exact registered
  span bytes to a shared `RingBudget` whose limit is the capacity-sized
  two-endpoint allowance already admitted, and only then lets the native mapped
  allocation run. The RAII reservation lives inside the connection (declared
  after the endpoint, so registered memory is destroyed before the credit is
  returned), which means pending connections in the admission channel and live
  connections both count, a rejected peer holds no credit, and a failed
  initialize rolls its credit back. A peer is rejected at accept only when its
  registered spans would push the aggregate past the budget: a peer advertising
  larger slots is rejected, and a third endpoint is rejected once the byte total
  is reached, while several small endpoints can still fit under the same
  capacity-sized allowance. Rejection never disconnects an existing good peer. At
  capacity 4096 the limit is exactly two full-size endpoints, so a reconnect can
  only be admitted after the old connection has been dropped and reaped; this is
  intentional and no unbudgeted credit is
  added.
- `headroom` is `DS41RT_SPARK_RUNTIME_HEADROOM_BYTES`, default `0`: no arbitrary
  reserve is invented. A measured CUDA-context/allocation-granularity reserve can
  be set from the field.
- Both peaks are also checked against the actual `cuda_memory_info()` free
  bytes at startup, so a configured budget above the real pool cannot admit an
  unmeetable plan. The query fails closed: a CUDA error aborts admission rather
  than silently skipping the real-availability check.

Memory diagnostics use one `/proc/meminfo` snapshot plus `cudaMemGetInfo` and
write to the `ds41rt::spark_memory` INFO target: at `worker startup`, after the
weights are resident, after the execution workspace and host exchange are
allocated, once when each accepted connection becomes owned (the mapped rings
already exist), and once per admission event after the first subsequent
successful request ("first success after the admission event", not a proof that
the request arrived on the newest connection). The `/proc` fields are
`MemTotal`, `MemFree`, `MemAvailable`, `Cached`, `Mlocked`, `Unevictable` and
`SReclaimable` in KiB, which separate host-available from UMA free without
another query. The serve-time calls are guarded by the same explicit target as
the event, so they cost nothing when that target is disabled; the ring
`used`/`peak` fields are point-in-time atomic samples (concurrent admission can
raise the peak), not reservations. Reclaimable page cache is reported, never
added to the permanent footprint; the loader's buffered reads are not forced out
(no `fadvise` change), and no global checkpoint-cache policy was altered.

Enabling `ds41rt::expert_timing` (DEBUG, default off) allocates three CUDA event
objects per worker for roles 1/5/6. Their footprint is an unmeasured
driver-object count; it is **not** asserted against the OS reserve or the device
budget, and no consumption claim is made.

**Still unmeasured.** The rings are admitted from the documented transport rule
and enforced byte-exactly at accept time, but no live run has yet confirmed the
negotiated frame sizes, the accepted ring spans, the reconnect timing, the CUDA
context baseline or the achieved free memory. `TP2EP2` at 29 remote layers
therefore has only a slim margin and **no claim that 29 layers is safe is made**
until that measurement exists; the user's experimental Spark budget
`MemTotal − 20 GiB` keeps the OS reserve outside this budget, and the existing
`--device-budget-bytes` override remains the launcher's contract. A live
4096-row acceptance on one worker's two endpoints (decode + prefill), together
with the same two-endpoint acceptance on the paired worker, and a
reconnect-after-reap test remain pending on the GPU lease.

This does not change legacy EXL3/TP4 admission, which keeps its two original
checks and the legacy `accept` path (no ring byte budget).

## 7. Placement handoff and the RTX boundary

`StartupPlacement::publish(directory, rtx_gpus, layers)` now records the actual
1/2 RTX count instead of an unconditional `2`, and the reader accepts 1 or 2.
The dual-RTX worker passes its real `--rtx-gpus`; the launcher's `plan.json`
filter still sees `rtx_gpus == 2` on that path, so the released dual-RTX
handshake is unchanged.

The dual-RTX plus explicit topology path uses a **dynamic** floor
(`minimum_expert_layers = 1` instead of the legacy `20`) so the published
boundary is the actual memory-driven split and is propagated to the Spark
`--first-layer`, never assumed to be 20. For an automatic dual-RTX launch the
existing `plan.json` → `ready.json` ordering already re-checks admission against
the published boundary.

Single-RTX boundary propagation is **not implemented**: `serve-native` still
requires `--placement-directory` to be absent for `--rtx-gpus 1`, and the
single-RTX worker connects the Spark transport before computing its local layer
plan, so publishing a boundary there would deadlock workers that wait for it.
Consequently `TP3EP2` currently launches Spark workers with `--first-layer 0`
and loads all 40 TP3 layers even when the coordinator keeps some layers local.
That is a memory redundancy, not a correctness error (TP3 fits 40 layers
weight-only with ~10 GiB slack); it must not be "fixed" by making the
coordinator wait after connecting the transport. A one-RTX handoff needs an
explicit boot-ordering change and test before it is enabled.

## 8. Cost model

`v41_native_serve/speculative/cost.rs` labels the remote backend from the actual
topology:

- explicit topology → `spark_tp{TP}ep{EP}` (so `TP2EP2` is never `spark_tp4`);
- legacy world 2 → `spark_tp2`; legacy world 4 → `spark_tp4`, unchanged.

The built-in TP4 calibration is applied only to the legacy single-RTX
TP4×EP1 layout. Any explicit topology returns `Ok(None)` (the legacy adaptive
heuristic) unless a caller supplies `DS41RT_ADAPTIVE_COST_PROFILE` with matching
truthful labels. No new coefficients were invented, and the placement is logged
with the topology.

## 9. Correctness properties carried by the daemon

- exactly-once: an active expert has exactly one owner; the shared expert is
  added once; the transport rejects conflicting/duplicate ownership.
- empty group: the same request reaches every rank; an empty group unowns every
  route and returns a zero plane, so coverage never loses a rank.
- ordered FP32 accumulation in rank order for N=2/3/4/6 with one BF16 rounding.
- route order, gate weights and the canonical six-expert rows are preserved by
  the ownership encode; masking is `384`/`0` only.
- cancellation/drain, pointer lifetimes, `reset_connections` and independent
  lanes are unchanged; the ownership scratch is lane-local and preallocated.

### 9.1 Request-ingestion ordering needs no source event

The normal Spark path (`execute_request_output`, roles 1/5/6) drains
`self.stream`, uploads `hidden`/`ids`/`routing` with `NativeLibrary::copy_h2d`,
then launches the expert kernel on the non-blocking execution stream.
`copy_h2d` does **not** hand a pageable pointer to the native copy: it first
copies the source bytes into a per-library pinned staging buffer
(`SyncH2DStagingBuffer`, allocated through `alloc_host_buffer` →
`cudaHostAlloc(Portable | Mapped)`), then calls native `ds41rt_copy_h2d`, which
is `cudaMemcpy(..., cudaMemcpyHostToDevice)` from that pinned pointer.

Per the CUDA API contract, a **pinned** host-to-device `cudaMemcpy` is
synchronous with respect to the host, so the destination holds the bytes before
the launch is enqueued. `cudaStreamNonBlocking` on the execution stream is
irrelevant here because the ordering comes from the completed host-blocking
copy, not from stream semantics. The pageable host-to-device caveat (the call
may return once staged, with the device DMA still pending) does not apply,
because the native call never receives a pageable pointer. The shared staging
buffer is mutex-guarded and each synchronous copy completes before it is reused.

No event, fence, or stream change is required; in particular, deliberately
making the execution stream blocking is not a fix and would only add
cross-stream serialization.

## 10. CPU test evidence

Focused command (root-NVMe isolated target directory; no GPU, no peers, no
service). Never build on the NTFS scratch mount:

```bash
CARGO_TARGET_DIR=/home/tj/.cache/ds41rt/builds/<unique> \
DS41RT_PYTHON=.venv/bin/python scripts/run-with-python-env.sh \
  cargo test --offline --manifest-path rust/Cargo.toml -p ds41rt-daemon
```

Coverage added by this change (all CPU):

| Area | Test |
| --- | --- |
| CLI | `cli::tests::spark_topology_flags_are_opt_in_all_or_none_and_range_checked` — both processes, absent keys identical to legacy, all-or-none, ranges, six-rank world/rank |
| Topology | `v41_spark_topology::tests::*` — six approved layouts, exact rank count, unapproved pairs, group/tp-rank mapping |
| Worker selection | `v41_experts::service::tests::explicit_topology_maps_every_physical_rank_to_its_local_shard` — physical rank → local shard, loader/staging pairing, roles 5/6 |
| Legacy selection | `...::legacy_selection_is_unchanged_and_topology_must_agree` |
| Budget transients | `...::explicit_admission_counts_every_known_transient_exactly`, `..._peak_overflow_is_rejected`, `host_exchange_allocation_matches_its_declared_extents`, `registered_ring_bytes_match_the_transport_sizing_rule`, `endpoint_count_is_fixed_at_two_and_stale_overrides_are_rejected`, `runtime_ring_budget_bounds_aggregate_endpoint_advertisements` |
| Ring budget (transport) | `ds41rt_transport::verbs::local::budget_tests::*` — reserve/release/peak/over-limit, checked_add overflow, unit RAII recovery after a failed reservation, concurrent CAS under a deterministic barrier, `from_wire` geometry rejection before any allocation |
| Timing roles (CPU) | `v41_experts::execution::timing_role_tests::routed_timing_is_enabled_for_the_compact_output_roles` — the DEBUG timing gate covers roles 1/5/6 and not 0/2/3/4 |
| Timing summarizer (CPU) | `scripts/tests/test_summarize_expert_timing.py` — EP1 legacy totals, EP2 owned-subset (2 rows, 7 owned), tail bin, empty group with null fractions, optional declared `owned_routes` agreement, and malformed over-count/bin/tail/fractional/non-finite rejection |
| Coordinator planes | `v41_experts::coordinator::replicated_tests::wave_reservation_scales_with_the_physical_rank_count` — exact 2/3/4/6 formulas, legacy alias, overflow |
| Ownership planning | `...::six_route_batches_split_evenly_across_replicated_groups` (3/3 and 2/2/2), `duplicate_experts_in_one_row_are_rejected_by_the_protocol`, `real_row_reuse_keeps_one_group_per_expert_and_masks_weights_exactly`, `an_empty_group_masks_every_route_and_returns_a_zero_plane`, `planning_is_reproducible_for_a_fixed_layer_and_request_id`, `single_group_topology_owns_every_active_expert` |
| Cost labels | `v41_native_serve::speculative::cost::tests::remote_backend_labels_report_the_actual_topology`, `builtin_tp4_profile_never_covers_an_explicit_topology` |
| Placement | `v41_native_serve::placement::tests::placement_records_and_validates_the_actual_rtx_gpu_count` and the updated round-trip tests |

Executed at commit `92e29c7` + this uncommitted tree, target
`/home/tj/.cache/ds41rt/builds/daemon-tp-ep-target`:

- `cargo check -p ds41rt-transport -p ds41rt-daemon`: PASS.
- `cargo test -p ds41rt-daemon --bins`: **835 passed, 6 failed, 101 ignored**.
  All 6 failures are pre-existing legacy `commands::real_full` tests unrelated to
  this change (see §10.1): five `target_attention` tests fail on a third-party
  `b12x.attention.dsa_indexer.SOURCE_LAYOUT_PAGED` AttributeError, and
  `real_checkpoint_nvfp4_decode_matches_python_fixture` fails on the absent
  `tests/fixtures/nvfp4/real_tensor_decode.json` fixture.
- `cargo test -p ds41rt-transport budget_tests`: **4 passed, 0 failed**.
- Focused filters: `replicated_tests` 8 passed; `v41_experts::service::tests` 8;
  `spark_topology` 6; `speculative::cost::tests` 6; `placement::tests` 4 (8
  ignored GPU tests); `cli::tests` 5. Zero failures.

The full daemon crate `cargo check` and the tests above are the only executed
evidence. No GPU, native, container, service or workload action was performed.

### 10.1 Baseline classification of the six failing legacy tests

The six failures in the full `--bins` run are **pre-existing at commit
`92e29c7`** and do not touch the TP×EP candidate. Two independent proofs were
produced: a source/interface proof and an isolated baseline run.

**A. Source proof (all HEAD-identical, none modified by this work).**

```bash
git ls-files tests/fixtures/nvfp4            # empty: the fixture is untracked
ls tests/fixtures                            # No such file or directory
git diff HEAD --stat -- \
  rust/crates/ds41rt-daemon/src/commands/real_full \
  python/reference/ds41rt_reference/deepseek_v4_attention_layer_capture.py
                                             # empty: identical to HEAD
git status --short third_party/sparkinfer    # clean at pinned 4b095414
```

- `commands::real_full::sparse_mlp::math::tests::real_checkpoint_nvfp4_decode_matches_python_fixture`
  reads `tests/fixtures/nvfp4/real_tensor_decode.json`. That directory does not
  exist in this checkout and the path is not tracked (`git ls-files` is empty),
  so the test panics on `fs::read` regardless of any Rust change.
- The five `commands::real_full::coordinator_kernels::target_attention::tests`
  failures come from `plan_deepseek_v4_target_device_storage`, which queries the
  tracked Python reference module
  `python/reference/ds41rt_reference/deepseek_v4_attention_layer_capture.py`.
  That module builds a
  `dsa_indexer.Caps(source_layout=dsa_indexer.SOURCE_LAYOUT_PAGED, ...)`
  (lines 1184/3672/4055/4421), but the pinned SparkInfer revision (`4b095414`)
  narrowed `b12x.attention.dsa_indexer` to a lazy/planned API whose
  `META.entry_points` do not export `SOURCE_LAYOUT_PAGED` (nor
  `INDEXER_SOURCE_LAYOUT_PAGED` at package level), so the lazy gate in
  `b12x/_lib/meta.py:79` raises `AttributeError`.

Direct reproduction against the pinned sources:

```bash
PYTHONPATH=python/reference:third_party/sparkinfer .venv/bin/python -c \
  "from b12x.attention import dsa_indexer; print(hasattr(dsa_indexer,'SOURCE_LAYOUT_PAGED'))"
# False
PYTHONPATH=python/reference:third_party/sparkinfer .venv/bin/python - <<'PY'
from ds41rt_reference.deepseek_v4_attention_layer_capture import deepseek_v4_c4_selector_scratch_nbytes
deepseek_v4_c4_selector_scratch_nbytes(variant="flash", mode="prefill",
    max_rows=2048, source_pages=64, max_page_table_width=64)
PY
# AttributeError: module 'b12x.attention.dsa_indexer' has no attribute 'SOURCE_LAYOUT_PAGED'
#   at b12x/_lib/meta.py:79
```

A sibling skew exists in the same package: calling
`qualify_deepseek_v4_attention_layer_contract(...)` raises
`ImportError: cannot import name 'BlockFP8LinearScratchPlan' from 'b12x.gemm._shared.block_fp8'`.
The tracked Python reference is simply behind the pinned SparkInfer submodule.

**B. Isolated baseline run at HEAD.** A fresh detached worktree was created at
`92e29c7` (main dirty tree untouched; the pinned submodule was symlinked
read-only because worktrees do not populate submodules) with its own root-NVMe
target:

```bash
git worktree add --detach /home/tj/.cache/ds41rt/baseline-worktree 92e29c7
CARGO_TARGET_DIR=/home/tj/.cache/ds41rt/builds/baseline-target \
DS41RT_PYTHON=.venv/bin/python scripts/run-with-python-env.sh \
  cargo test --offline --manifest-path rust/Cargo.toml -p ds41rt-daemon \
  --bin ds41rt coordinator_kernels::target_attention::tests
# test result: FAILED. 24 passed; 5 failed
#   AttributeError: module 'b12x.attention.dsa_indexer' has no attribute 'SOURCE_LAYOUT_PAGED' (x5)
# ... --bin ds41rt real_checkpoint_nvfp4_decode_matches_python_fixture
# test result: FAILED. 0 passed; 1 failed
#   reading .../tests/fixtures/nvfp4/real_tensor_decode.json: No such file or directory
```

Same six tests, same errors, on a tree that contains none of this change.

**C. Candidate impact: none.** The native candidate is served by
`v41_native_serve` / `v41_experts` (`serve-native`, `expertd-native`). Those
modules contain no `pyo3`, `Python::`, `python_graph_capture` or
`deepseek_v4_attention_layer_capture` reference, and the executor's Python
planner is reached only from the legacy `commands/real_full` path (and from
`commands/coordinator` via `initialize_coordinator_python_capture_from_env`,
which defaults off). The TP×EP changes are confined to the `v41_*` native path
and the shared transport/loader/FFI contracts, so the six failures cannot affect
the replicated-group binary, its admission or its execution.

**D. Legacy-impact note.** The legacy `ds41rt coordinator` command with Python
capture explicitly enabled (`DS41RT_B12X` truthy) would also hit the tracked
reference/SparkInfer skew, as the sibling `ImportError` shows. That is a
pre-existing Python-reference integration issue for its owner, is outside the
TP×EP candidate, and was deliberately not fixed here.

## 11. Pending GPU / native / live gates

Nothing below is passed. All require the coordinator and Spark hosts and belong
to the serialized build/GPU lease.

1. `libds41rt_native.so` exports interfaces `8`/`9` with roles `5`/`6`,
   `input_dtype == 7`, geometry `(384, 1152/768, 1152/768, 6)`; the 3/6-plane
   reducer symbol exists and `require_rank_count` passes.
2. Per-rank `info.scratch_bytes` at capacities 1/16/80/256/1024/4096 matches the
   memory review's formulas; the worker's `resident + workspace` and the new
   explicit `load/serve` peaks match `cudaMemGetInfo`.
3. Packer 768 real-weight round trip; TP2/TP3 shard numerics against an unsplit
   oracle; masked/inactive routes provably skip work.
4. Live `TP4EP1` explicit round trip (flag + owners all zero) and live
   `TP2EP2` round trip: same request to four ranks, three masks per row on
   average for EP2, ordered FP32 assembly with shared added once, empty-group
   zero plane.
5. Transport registered/mapped ring bytes and measured prefill/decode
   workspace; then the actual `L_r` / `spark_first_layer` budget closure for
   `TP2EP2` (weight-only allows 29 remote layers; the reserve decides).
6. Six-rank (`TP3EP2`, `TP2EP3`) hardware is not connected: end-to-end serving
   is hardware-unqualified. Transport contracts and CPU tests are the only
   evidence.
7. Matched TP4 vs TP2EP2 correctness/performance campaign with stated
   tolerance, startup/memory reported separately from execution.

## 12. Files changed

- `cli.rs` — `--spark-tp/--spark-ep`, widened worker rank/world ranges, CLI tests.
- `v41_spark_topology.rs` — new resolution/admission helper module.
- `v41_experts.rs` — `ExpertLayer::BackboneReplicatedTp`, role/interface mapping.
- `v41_experts/execution.rs` — `compact_output_role`, `install_native_group`,
  masked route unpack, role-5/6 scratch/output, `HostExpertExchange::bytes_for`.
- `v41_experts/service.rs` — topology config/validation/selection, explicit
  admission with known transients, ring budget and diagnostics, tests.
- `v41_experts/service/backend.rs`, `service/local.rs` — group install,
  topology-bound parse, executor id, `RingBudget` + `accept_with_budget` wiring.
- `v41_experts/coordinator.rs` — `ReplicatedGroupPlanner`, topology-aware wave,
  `device_bytes_for`, 3/6-plane reduction, ownership tests.
- `v41_experts/exl3.rs` — explicit rejection of replicated layers on EXL3.
- `v41_backbone_router.rs` — `BoundExpertRequest::assign_native`.
- `v41_native_serve.rs`, `v41_native_serve/distributed.rs` — topology transport,
  non-native rejection, rank-count check, N-plane budget, dynamic floor,
  actual `rtx_gpus` publication.
- `v41_native_serve/placement.rs` — actual GPU count.
- `v41_native_serve/speculative/cost.rs` — topology-truthful labels and guard.
- `scripts/summarize-ds41-expert-timing.py`,
  `scripts/tests/test_summarize_expert_timing.py` — histogram-inferred owned
  routes for replicated groups (Python-only).
- `docs/tp-ep-runtime.md` — this document.

Minimal additive transport interface (authorized for the launch-blocker fix,
coordinated with the transport owner):

- `ds41rt-transport/src/verbs/local.rs` — `RingBudget`/`RingReservation`,
  `accept_with_budget`, `validated_ring_geometry`, `read_persistent_start`;
  `accept`/`initialize` signatures and behavior unchanged.
- `ds41rt-transport/src/verbs.rs`, `src/lib.rs` — re-export the new API.

No file outside `rust/crates/ds41rt-daemon/**`,
`rust/crates/ds41rt-transport/src/{verbs.rs,lib.rs,verbs/local.rs}` and this
document was modified; no commit or push was made. The audit worktree target and
baseline worktree live under `/home/tj/.cache/ds41rt/` only.
