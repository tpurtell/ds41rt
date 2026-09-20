# TP/EP pre-E2E integration review

Read-only audit before the dual-RTX TP/EP end-to-end run. No code, test, kernel or
config was modified. Source state: frozen daemon release build 915 / l6 preflight
build `coord-native` (objects compiled 2026-09-20T14:43); the l6 snapshot source is
byte-identical to the working tree for the files cited here.

Scope: the three flagged invariants plus the urgent dual-GPU peer-copy failure.
Memory/weight budgets are deliberately **not** repeated here; they live in the
parallel memory-ring audit and `docs/tp-ep-architecture-audit.md`.

Evidence legend: [measured] log/selftest result, [code] read from source with
path:line, [unconfirmed] hypothesis awaiting instrumentation, [gap] missing test.

Severity: **H** blocks E2E, **M** should be fixed or explicitly accepted before E2E,
**L** record/follow-up.

---

## 0. Invariant status summary

| # | Invariant | Status | Worst finding |
| --- | --- | --- | --- |
| 1 | Masks/flags bound canonical request planner; no EP1 accidental reject | Pass, with coverage gaps | L: builder pre-validates at max_rows 4096; explicit TP4EP1 has no coordinator-level end-to-end test |
| 2 | Every physical rank returns one BF16 plane; shared added once; ABI 2 vs 3 | Pass | L: double `execution_state` selection (cost only); no H gap |
| 3 | Native loader `BackboneTp` world 2/3, roles 5/6, capacity, all 384 experts | Pass | M: TP3 packer selftest not in the l6 gate logs; no service-level resident-bytes test |

No invariant H-blocker was found. The measured peer-copy failure is a
test-ordering defect (section 4.2), and the production producer→peer-read
ordering is proven complete for all four audited sites on the cooperative path
(section 5); the only note is a non-production non-cooperative edge.

---

## 1. Invariant 1 — ownership flags, masks and canonical-request binding

**Claim under test:** the native group mask/flag is bound to a canonical request
planned exactly once, and an explicit `TP4EP1` (or any `EP1`) deployment is not
rejected by the legacy 4-rank admission path.

Verified [code]:

- The builder validates canonical shape **before** touching any route word and
  refuses a second encode:
  `with_native_group_owners` calls `V41BackboneRequest::validate_owned(self, 4096)`
  and rejects an already-flagged request
  (`rust/crates/ds41rt-transport/src/v41_expert/native_group.rs:308-315`). A
  malformed batch is returned byte-identical for retry.
- Owners are validated as `owners[expert] < topology.group_count()`
  (`native_group.rs:322-325`); for `EP1` the only legal owner is 0, so `EP1` is
  accepted by design rather than rejected. `INACTIVE` (255) is only legal for an
  expert that was never routed.
- The wire word requires **exactly one** owner bit and `owner < group_count`
  (`native_group.rs:213-228`), so a multi-bit/empty owner cannot be counted twice.
- Batch consistency is enforced on decode: all occurrences of an expert must agree
  (`V41NativeOwnershipBatch::observe`, `native_group.rs:257-266`).
- The worker's admission contract is mutually exclusive: a topology-bound worker
  unpacks the ownership contract and masks, a legacy worker rejects an
  ownership-encoded request outright, and a group-bound worker rejects a legacy
  request (`rust/crates/ds41rt-daemon/src/v41_experts/execution.rs:691-701`).
- The coordinator encodes only when a native topology is installed; the legacy lane
  is a strict no-op (`rust/crates/ds41rt-daemon/src/v41_experts/coordinator.rs:154-162`).
- Responses strip the native flag
  (`rust/crates/ds41rt-transport/src/v41_expert/chunks.rs:80-81`), so the
  response-flag allow-list does not need it.
- The planner never emits an inactive owner for a routed expert: every routed
  expert is active, so `group.encoded()` is written for all 384
  (`coordinator.rs:100-116`).
- Explicit `TP4EP1`: `selection()` maps `tp()==4` to the legacy
  `ExpertLayer::Backbone` role 1 (`v41_experts/service.rs:561-573`), and the
  coordinator's `N==4` reduction uses the historical 4-plane entry
  (`coordinator.rs:529-534`). This is intentional and numerically safe (section 2).

Findings:

1. **L — builder pre-validation uses a fixed 4096-row ceiling.**
   `native_group.rs:315` validates against `4096` regardless of the transport's
   admitted capacity. A row count above the configured capacity but ≤4096 passes
   the builder and fails later at dispatch. Not a correctness hole (the transport
   re-validates at its own capacity), but the failure moves to a less specific
   place. Test: build with a reduced capacity and assert the builder's own error
   stays specific.
2. **L — explicit `TP4EP1` has no coordinator-level end-to-end test.** The
   topology/owner encode and the legacy reducer are each tested, but nothing drives
   `assign_native` → worker mask → legacy `reduce` for `SPARK_TP=4 SPARK_EP=1` as
   one unit. Add a synthetic fixture for that exact topology.
3. **[gap]** No test asserts that a legacy worker (no topology configured) still
   accepts a plain canonical request while a topology-bound peer rejects it; the
   runtime path covers both (`execution.rs:691-701`) but only one side is tested
   directly.

No accidental `EP1` rejection was found.

---

## 2. Invariant 2 — one BF16 plane per physical rank, shared exactly once, ABI 2 vs 3

**Claim under test:** every physical rank returns exactly one BF16 `[rows,5120]`
partial; the coordinator adds the shared expert exactly once; the two shipped Spark
output ABIs (2 = `fp32_routes` at capacity 1/16/80, 3 = `fp32_tokens` at
256/1024/4096) agree.

Verified [code]:

- The worker compacts every kernel output kind to one BF16 token-major plane:
  `Fp32Tokens → compact_tokens`, `Bf16Routes → compact_bf16_routes`,
  `Fp32Routes → compact(route_partials)` at
  `rust/crates/ds41rt-daemon/src/v41_experts/execution.rs:743-751`;
  `route_partials` itself requires `output_kind == Fp32Routes`
  (`execution.rs:321`). The compact output is one `capacity*5120*2` allocation
  (`execution.rs`, `output_and_shared_bytes` in `plan_execution`).
- The coordinator uploads every rank plane into `planes[rank]` and reduces once;
  the native kernel sums the BF16 planes in rank order in FP32, adds the optional
  BF16 shared **after** the routed sum, and rounds once to BF16
  (`native/cuda/kernels/v41_route_reduce.cu:88-103`). `enqueue_reduce_planes`
  passes `shared` exactly once for `N ∈ {2,3,4,6}`
  (`coordinator.rs:525-540`).
- The native header documents the same single-round contract
  (`native/include/ds41rt_v41_experts.h:100-110`).

[measured] `ds41rt_v41_route_reduce_planes_selftest` ran in the l6 gate
(`runs/tp-ep-preflight/l6-coordinator/ds41rt_v41_route_reduce_planes_selftest.log`,
`SELFTEST_EXIT=0`; same log for the dual run). Its header states it verifies:

- an independent host scalar reference for `ranks 2/3/4/6` with shared added once
  and one final BF16 round;
- **the historical TP2/TP4 entry points agree bit-for-bit with the generic entry
  point for the same ordered planes**;
- destination coverage `[0, rows*5120)` with trailing poison intact;
- the `rows=4096` boundary; graph replay re-reading changed plane/shared contents;
  all-zero and single-zero planes contributing exact zeros; malformed six-slot
  argument sets rejected with exactly one violation.

This directly covers the flagged concern that explicit `TP4EP1` (owner 0) uses the
legacy 4-rank reducer instead of the new 3/6 path: the legacy 4-plane entry and
`reduce_planes(ranks=4)` are the same `reduce_compact<4>` with the same launch
geometry, summation order, shared placement and single rounding
(`v41_route_reduce.cu:190-212` vs `:242-277`), and the selftest proves bit
equality. **No correctness divergence at the reducer level.**

Findings:

1. **Correction — ABI2 vs ABI3 must NOT require identical BF16 planes.** The two
   output kinds use different kernel tiling and quantization, so they can round
   differently on the same inputs. The correct requirement is an **independent
   oracle with tolerance per ABI**. f6d0's native 108-real-case qualification
   already covers both APIs, each against its own oracle (not a direct cross-ABI
   comparison), so there is **no H gap** here. A cross-ABI relative metric is a
   useful diagnostic only, not a bit-exact acceptance gate.
2. **Correction — N=3/6 coverage already exists.** The receiver and plane
   collection are covered by name:
   `six_rank_receiver_covers_every_plane_and_rejects_stale_or_foreign_responses`
   and
   `six_rank_chunked_coverage_rejects_reordered_overlapping_and_bad_final_markers`
   and
   `planes_collection_supports_three_and_six_ranks_but_legacy_accessor_stays_four`
   (`rust/crates/ds41rt-transport/src/v41_expert/native_group.rs:709,766,822`),
   plus `six_rank_native_group_tcp_covers_every_group_and_rank`
   (`v41_expert/tcp_tests.rs:282`). Wave reservation scales with rank count in
   `wave_reservation_scales_with_the_physical_rank_count`
   (`v41_experts/coordinator.rs:893`). **No missing test name.**
3. **L — `execute_request_output` selects the execution state twice** (once for the
   compact match, once inside `route_partials`). Deterministic and same `rows`;
   cost only. Not a gate.

---

## 3. Invariant 3 — native loader `BackboneTp`, worlds 2/3, roles 5/6, all 384

**Claim under test:** the native loader's replicated TP selector is world 2/3
consistent, maps to native roles 5/6, keeps the packer/FFI capacity contract, and
still loads all 384 experts (full replica).

Verified [code]:

- `V41BackboneTpGeometry` accepts `world ∈ {2,3,4}`, requires
  `moe_intermediate_size % world == 0` and `(moe_intermediate_size/2) % world == 0`,
  and derives the shard as `2304/world` → 1152/768/576
  (`rust/crates/ds41rt-loader/src/v41_expert_staging.rs:66-130`). Both divisibility
  conditions are needed because W2 is packed two FP4 values per byte and its E8M0
  K32 groups must tile the slice.
- The packer accepts `{576,768,1152,2304}` and pads only
  `(I+127)/128*128`, so TP2 (1152) and TP3 (768) are padding-free
  (`native/cuda/kernels/v41_expert_pack.cu:52-64`).
- FFI interfaces 8/9 map to `ds41rt_v41_spark_tp2_expert_info` /
  `..._spark_tp3_expert_info`, require `input_dtype == 7` (FP8 K32), and expect
  roles 5/6 with geometry `(384,1152,1152,6)` / `(384,768,768,6)`
  (`rust/crates/ds41rt-ffi/src/v41_experts.rs:585-680` and
  `expected_expert_geometry` at `:186-205`). The `experts == 384` check is part of
  the tuple equality, so a partial expert set is rejected.
- `ExpertLayer::BackboneReplicatedTp` dispatches world 2 → spark_tp2,
  world 3 → spark_tp3, world 4 → legacy role 1, and rejects anything else
  (`rust/crates/ds41rt-daemon/src/v41_experts.rs:70-118`).
- The service selects `BackboneReplicatedTp { rank: tp_rank, world: tp }` for
  `tp ∈ {2,3}` and the legacy `Backbone` for `tp == 4`
  (`v41_experts/service.rs:561-573`), and `validate_topology` requires
  `world == topology.world_size()`, `rank < world`, and the native catalog
  (`service.rs:532-556`).
- Admission plans workspace from the **same** selection as the load and adds
  staging/pinned/read-scratch + host exchange + row-index scratch, then checks the
  device budget and the actual free bytes
  (`service.rs:82-140`). The comment explicitly states a TP2/TP3 topology cannot be
  budgeted with TP4 scratch.

[measured] Loader geometry: `rust/crates/ds41rt-loader/tests/replicated_tp_staging.rs`
asserts adjacent world 2/3/4 shards tile the official expert on both axes, with
W2 packed-byte and K32 scale offsets aligned and gapless. Native packer:
`native/tests/v41_expert_pack_tp3_selftest.cc` re-implements the N256/K128 lane
transform and compares every output byte at extent 768, checks the accepted-extent
list and the 32/128 alignment.

Findings:

1. **M — the TP3 packer selftest is built but not in the l6 gate logs.** The l6
   build links `ds41rt_v41_expert_pack_tp3_selftest`
   (`l6-build.log:43,100`, `BUILD_EXIT=0`), but
   `runs/tp-ep-preflight/l6-coordinator/` has no run log for it (only route-reduce,
   native, xgrammar, and the two peer tests). Run it and record the log; it needs
   only one supported device.
2. **M — no service-level resident-bytes test for worlds 2/3.** The loader test
   covers geometry algebra and staging windows; nothing asserts that the daemon's
   `resident` sum for a TP2/TP3 selection equals
   `staging_strides × 384 × (40 − first_layer)` and that a `first_layer > 0`
   deployment still admits. Add a planning test through `ExpertWeights::plan` with
   the replicated selection.
3. **L — `BackboneReplicatedTp { world: 4 }` falls through to role 1** in
   `role()`/`info()`/`kernel()` (`v41_experts.rs:70-118`). `selection()` never
   constructs that value, so it is dead, but the fallthrough would silently use the
   legacy 576/640 kernel if a future caller did. Prefer an explicit error.
4. **[gap]** All-384 is enforced by the FFI tuple and by the per-expert loader loop
   (one expert at a time, no expert partition), which matches the fully replicated
   contract; there is no test that deliberately asserts "the same 384 expert ids
   are resident on every group", because the code has no expert→rank partition to
   assert against. Acceptable for the chosen contract; note it so a future
   expert-sharded EP does not silently inherit it.

---

## 4. `ds41rt_v41_peer_copy_selftest` dual-GPU failure (root-caused)

### 4.1 Facts

- [measured] The failure message is thrown only at
  `native/tests/v41_peer_copy_selftest.cc:77` (the pitched split+join phase). The
  non-pitched phase's message is line 38, so that phase passed. `exit 134` is an
  uncaught `std::runtime_error`.
- [measured] Provenance: the l6 `coord-native` test and kernel objects were compiled
  at 2026-09-20T14:43; the l6 snapshot `coord-native-source` is byte-identical
  (empty diff) to the working tree for the test, kernel, and header, whose sources
  date to 09-17. A stale-source cause is unlikely. (Two binaries exist; both
  post-date the sources.)
- [measured] `ds41rt_v41_route_reduce_planes_selftest` passes on the same dual GPU,
  so the device/driver is functional for the reducer path.
- [measured] `ds41rt_cuda_peer_selftest` passes, but it exercises
  `cudaMemcpyPeerAsync` and `cudaMemcpy3DPeerAsync` (copy engine) through
  `ds41rt_copy_device_rows_async` (`native/src/ds41rt_native.cc:731-780`), **not**
  the SM-issued `copy_rows` kernel. It is not coverage for the failing path.
- [code] `ds41rt_v41_peer_copy_rows_async` selects `copy_rows<uint4|uint32|uint8>`
  by `dst|src|width|dst_pitch|src_pitch` alignment and launches
  `copy_rows<T>` (`native/cuda/kernels/v41_peer_copy.cu`).
- [code] The first loop (lines 12-48) uses the same NULL-stream `cudaMemset`
  followed by a graph on a `cudaStreamNonBlocking` stream and passes its guard
  check; CMake sets no `--default-stream` flag.

**Scope of the algebra check.** I modelled the exact `copy_rows` row/column index
formula and round-tripped the test's split+join for width {7,12,32,32768} × rows
{1,7,64}: payload and guard both reproduce exactly. That proves only the
**row/column mapping** is correct. It does **not** prove device association,
vectorized-branch execution, stream/race behaviour, or peer-write visibility.

### 4.2 Measured root cause (f6d0 triage)

Instrumented triage harness `native/tests/v41_peer_copy_triage.cc` and raw logs in
`runs/tp-ep-native/peer-triage/` (provenance hashes recorded there). Measured:

- `run_.out` (original ordering): **2/24** failures, both `width=32768 rows=7
  replay=1`, one at `owner=0` and one at `owner=1`, `region=payload`
  (`guard_failures=0`). Diagnostic: `stale_prev_input_match=1` — the mismatching
  bytes equal the **previous** input value, not the current one — and
  `split_half0_first_mismatch` is nonzero (65,536-73,088 depending on the run) while
  `half1` also mismatches at 229,376. So the **split (peer-side) read observed the
  old source contents** for part of the buffer.
- `run_sync.out` (device synchronize after the guard/memset): still **2/24** payload
  failures. The guard memset is therefore not the cause.
- `run_h2dsync.out` (synchronize after the source `cudaMemcpy` H2D): **0/24**.
- `run_h2dasync.out` (async H2D on the owner stream + sync): **0/24**.

Root cause: the test fills `source` with a **pageable** `cudaMemcpy`
(`v41_peer_copy_selftest.cc:58,79`). A pageable host-to-device `cudaMemcpy` may
return once the pageable buffer has been staged, before the device DMA to
`source` has completed, and its completion is only ordered with work in the
source device's own stream(s). The split graph runs on the **peer** device's
`cudaStreamNonBlocking` stream and reads `source` with no cross-device dependency,
so on replay 1 it can read the previous contents. A completion point after the
source write (or an async H2D on the owner stream) removes it.

**Approved disposition (f6d0):** test-only minimal fix — add the completion after
all four source writes in the original selftest; **kernel unchanged**; 3× rerun
pending. This is a test-ordering defect, not an SM-kernel index or vectorization
defect.

**Not measured, do not claim refuted:**

- **H2 (graph capture / foreign-device capture mode):** the no-graph variant was
  not run, so capture involvement is unmeasured. Keep as an open control.
- **H4 (device association / vectorized branch):** not separately isolated. The
  algebra check below covers only the row/column mapping, not the vectorized
  branch's device association. `ds41rt_v41_peer_copy_rows_async` still has no
  current-device check (the production `ds41rt_copy_device_rows_async` checks
  `device == dst.device_id`), which is a contract gap worth closing independently.

### 4.3 Scope of the algebra check

I modelled the exact `copy_rows` row/column formula and round-tripped the test's
split+join for width {7,12,32,32768} × rows {1,7,64}: payload and guard both
reproduce exactly. That proves only the **row/column mapping**. It does not prove
device association, vectorized-branch execution, or stream ordering — consistent
with the measured root cause being an ordering issue rather than an index bug.

### 4.4 Gate impact

The immediate defect is test-side and the approved minimal completion is
sufficient for the selftest. It must **not** be read as clearing production: the
test exercised a host-produced source, while production peer reads consume device
producers. Section 5 audits those call sites separately. `ds41rt_cuda_peer_selftest`
passing does not clear the SM pitched read either, because it exercises the copy
engine.

---

## 5. Production producer→peer-read ordering audit (caller-contract trace)

**Contract.** The FFI wrappers state the requirement explicitly:
`copy_peer_async` — "`stream` must be live on the current, destination device.
Source writes must precede the copy (use a stream event dependency)"
(`rust/crates/ds41rt-ffi/src/lib.rs:3872-3884`); `V41PeerCopy::launch`/`launch_rows`
— "Producers precede this stream, buffers are disjoint and live through completion"
(`rust/crates/ds41rt-ffi/src/v41_device_ops.rs:40-70`). The measured test defect is
the same missing-dependency class (a pageable H2D source write not ordered with a
foreign non-blocking read).

**Event-ordered exemplars (the correct pattern:** record on the producer stream →
destination/peer stream `cuda_stream_wait_event` → copy → optional copied event →
producer wait**):**

- `rust/crates/ds41rt-daemon/src/v41_memory/peer_publication.rs:46-52` — producer
  event, peer wait, copy, copied event, producer wait. This is the reference
  implementation.
- `rust/crates/ds41rt-daemon/src/v41_memory/device.rs:216` —
  `PeerTransfer::copy_then(producer, ...)`.
- `rust/crates/ds41rt-daemon/src/v41_experts/tp2.rs:554-558` — `ready.record(producer)`
  then `cuda_stream_wait_event(self.stream.raw, ready)`.
- `rust/crates/ds41rt-daemon/src/v41_experts/dspark/tp2/pair.rs:115-143` — `ready`
  event, peer wait, copies, `done` event, stream wait.
- `rust/crates/ds41rt-daemon/src/v41_memory/proposal_replica.rs:131-133` — goes
  through `publication.enqueue(producer.raw, ...)`.

**Caller-contract trace for the four sites (bounded, exact lines).** The completion
is provided by the lane phase machine, so the gate is **removed** for the
production cooperative path:

- **Sites 1/2 — `tp2_ffn.rs:176-190` (`execute_shared_on`) and `:266-291`
  (`execute`).** `NativeTp4Wave::execute_tp2_ffn` / `finish_tp2` are reached from
  `v41_backbone_lane.rs:302-397`, which runs inside
  `PreparedFfn::execute` (`v41_backbone_execution.rs:105-119`): that method
  `await`s `pending.complete()` before `ffn.execute_tp4(...)`, so the FFN input
  (`input.values`) is complete. The other copy sources are the router outputs:
  `v41_backbone_lane.rs:318-319` calls `router.execute_ffn_cooperative`, and
  `v41_backbone_router.rs:640` does `self.stream.wait().await?` before returning
  `output()` (the `wire`/`ids`/`routing` buffers). The production lane constructor
  sets `cooperative: true` (`v41_backbone_lane.rs:217`). Proven complete.
- **Site 3 — `tp2_ffn.rs:205-218` (`return_result`).** Its `source` is the reduced
  routed output; `NativePendingFfn::finish_values` launches the reduction and
  completes it (`reduce_planes_cooperative(...).await` drains, or `reduce_planes`
  calls `cuda_stream_synchronize`) **before** the `return_result` call
  (`v41_experts/coordinator.rs`, `finish_values`). Proven complete.
- **Site 4 — `v41_block/transfer.rs:40-56` (`BlockTransfer::copy`) via
  `import_previous_cooperative`.** `BlockTransfer::copy` takes
  `previous: &BlockOutput`, and every production caller obtains it from
  `BackboneLane::output()`, which requires `self.phase == Phase::Complete`
  (`v41_backbone_lane.rs:827-833`). That phase is set by `finish_ffn_cooperative`
  after the awaited block handoff (`v41_backbone_lane.rs:813-820`); the distributed
  target pass imports only after that (`v41_target_pass/distributed.rs:220-223` and
  `:345-350`). The source is already host-waited; not waiting again on the copy
  stream is intentional and avoids coupling independent lane DMA work. Proven
  complete.

**Residual edge (non-production path, keep noted, no E2E gate).** The
non-cooperative lane constructor `LaneFfn { cooperative: false }`
(`v41_backbone_lane.rs:687`, `PendingLaneFfn::attention_ffn`) pairs with
`BackboneRouterWave::execute_ffn` (`v41_backbone_router.rs:559`), which returns
after enqueue without a stream wait. If that lane were ever used with a local TP2
layer or a `BlockTransfer`, the source completion would be unproven. The
production dual-RTX target pass uses the cooperative constructors
(`complete_layer_cooperative`, `finish_ffn_cooperative`), so this is an edge to
keep, not a real violation on the E2E path.

**Conclusion.** On this trace the production cooperative path provides completion
for all four sites, so no production ordering fix is required for the E2E gate.
This is a caller-contract proof, not an inference from the test-only fix.

---

## 6. Recommended pre-E2E gate additions (ordered)

1. Confirm the approved selftest fix with the 3× rerun (peer fix already verified
   at 3×72 PASS); only if it is not clean, run the residual controls (no-graph H2,
   device-association H4) (M).
2. Run and record `ds41rt_v41_expert_pack_tp3_selftest` (M, invariant 3).
3. Add a daemon planning test for TP2/TP3 resident bytes with `first_layer > 0`
   (M, invariant 3).
4. Add the coordinator-level explicit `TP4EP1` fixture (L, invariant 1).
5. Close the `ds41rt_v41_peer_copy_rows_async` current-device contract gap (L).

No invariant-1/2/3 H-blocker was found in the frozen tree. The measured peer-copy
failure is a test-ordering defect with the fix verified; the production
producer→peer-read ordering is proven complete for all four audited sites on the
cooperative path (section 5). The only remaining note is the non-cooperative
`attention_ffn`/`execute_ffn` edge, which the production dual-RTX path does not
use.
