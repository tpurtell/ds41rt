# TP×EP foundation review: loader / native / transport

Read-only, CPU-only review of the newly implemented replicated expert-group
(`TP×EP`) foundation. No builds, no daemon/GPU/service action, no commits; I
edited none of the reviewed files. `docs/tp-ep-scheduler.md`,
`rust/crates/ds41rt-core/src/replicated_expert_schedule.rs` and this file are my
only outputs. Findings are anchored to the hashes below; the tree was edited by
sibling agents during the review and several findings were fixed in flight, as
noted.

| sha256 (16) | path |
| --- | --- |
| `0c12ff6ff6edc489` | `native/cuda/kernels/v41_route_reduce.cu` |
| `d4240644cecfcc29` | `native/include/ds41rt_v41_experts.h` |
| `5d390d0cb7e96e14` | `native/src/v41_experts.cc` |
| `254c1592f6bd90c9` | `native/cmake/v41_spark_tp_experts.cmake` |
| `6bf27a07310f7ad1` | `rust/crates/ds41rt-transport/src/v41_expert/native_group.rs` |
| `698a53b2f1a4cdd7` | `rust/crates/ds41rt-transport/src/v41_expert.rs` |
| `a2d7ca5d5140921a` | `rust/crates/ds41rt-transport/src/v41_expert/chunks.rs` |
| `373b03c350fbcb8c` | `rust/crates/ds41rt-loader/src/v41_expert_staging.rs` |
| `b4d580c63abc2674` | `rust/crates/ds41rt-ffi/src/v41_experts.rs` |
| `c8a4f9f4ee042094` | `rust/crates/ds41rt-daemon/src/v41_experts/coordinator.rs` |
| `3ca0b09716ca9bc0` | `rust/crates/ds41rt-daemon/src/v41_experts/execution.rs` |
| `772471123e93bc3f` | `rust/crates/ds41rt-daemon/src/v41_experts/service.rs` |
| `117a674e060ff8fc` | `rust/crates/ds41rt-daemon/src/v41_spark_topology.rs` |
| `02babff84d12d276` | `rust/crates/ds41rt-daemon/src/v41_native_serve.rs` |
| `bbb296a6ae30902f` | `run.sh` |
| `2daf14bc82f4cb11` | `scripts/release-common.sh` |
| `43a585668a68a911` | `scripts/tests/test_tp_ep_configuration.py` |

Reviewed through 2026-09-20T03:2xZ.

## Summary

| Risk | Verdict |
| --- | --- |
| 1. Native generic reducer preserves legacy FP32 order/aliases | **Correct.** Rank-ordered FP32 accumulation, one shared add, one BF16 round; legacy 2/4-plane arithmetic and aliasing rules preserved. The invalid-route-to-zero path is in-tree; GPU numerical proof is pending a new harness. |
| 2. Owner mask exact per-expert consistency source | **Exact and wired end to end** (coordinator producer, topology-bound transport, worker consumer, one shared topology source). One open item: the launcher and daemon approved-topology sets differ. |
| 3. Config forcing RTX count from Spark topology | **Was confirmed, fixed during review.** The hardcode is gone and the resolved weight-budget gate is now the only arbiter. |

## Risk 1 — native generic N-plane reducer

`reduce_compact<Ranks>` (`v41_route_reduce.cu:87-106`) accumulates `p0..p(N-1)`
in rank order with `__fadd_rn`, adds `shared` once, rounds once with
`__float2bfloat16_rn`. The `if constexpr` branches compile out, so `Ranks == 4`
and `Ranks == 2` reproduce the previous sequences exactly; the legacy entries
pass `nullptr` for the new `p4`/`p5` (`:205-208`, `:233-237`).
`valid_compact_planes` (`:111-135`) enforces ranks `{2,3,4,6}`, non-null/aligned
active planes, null inactive slots, plane/output non-overlap, exact
`shared == output` aliasing only, and the row cap; the kernel has no
`__restrict__`, so the in-place shared case is defined. This matches the header
(`ds41rt_v41_experts.h:147`) and the FFI `reduce_planes` safety contract.

**No zero-row gap.** `rows == 0` rejection (`:113`) is correct and unreachable
for a valid wave: zero *owned* routes is not zero *token* rows. Each replicated
group returns a full `[M, 5120]` plane for the wave's `M > 0` token rows, zero
where all its routes are masked (`copy_native_group_routes_into` writes the
unassigned id and zero weight; `v41_expert.rs:294`), so the coordinator always
reduces `rows = M > 0`. Empty groups must keep the `[M, H]` shape and return
zeros; there is no empty-group skip-response path.

**Invalid-route-to-zero source is in-tree.** The slice reduce that consumes the
sentinel is `third_party/sparkinfer/b12x/moe/_shared/kernels/v41_route_plan.py`:
`V41RoutePlan` (`:14`) writes `row = Int32(-1)` for unassigned slots, and
`V41SliceReduce` (`:121-122`) documents "Ordered FP32 slice sum into original
route order; invalid routes are zero"; `v41_slice_pipeline.py:12` wires both
into the pipeline. The transport's masked id (`384`) and zero weight therefore
land on a real, in-tree zero-contribution path. GPU numerical proof of sentinel
→ zero is still pending a new harness; that is the only remaining Risk 1 item.

## Risk 2 — owner mask exact per-expert consistency

The wire keeps the 12-byte route entry: bits `0..8` expert id, bits `9..11`
one-hot owner, `12..31` reserved (`native_group.rs:56-59`). `decode` (`:213-229`)
requires exactly one owner bit and `owner < group_count`, and
`V41NativeOwnershipBatch` (`:232-275`) rejects two owners for the same expert via
its 384-entry table. `validate_native_group` (`v41_expert.rs:122`) runs every
route through that table before canonical validation;
`copy_native_group_routes_into` (`:294`) preserves the true id and FP32 gate
weight only for the owned group and writes `384`/`0.0` otherwise. Ownership is
caller-provided from the scheduler; the transport never derives it.

End-to-end wiring is present:

- Coordinator: `ReplicatedGroupPlanner::encode` (`coordinator.rs:83-115`) builds
  the 384-entry histogram, seeds with `replicated_expert_tie_seed(layer_id,
  request_id)`, fills `owners[expert] = assignment[expert].encoded()`, and calls
  `with_native_group_owners`; `prepare_remote_request` (`:153-159`) applies it to
  the request that `dispatch_ffn` (`:353`) sends.
- Transport: `V41Tp4Roce/Tcp::new_topology` binds canonical executor ids, and the
  serve path now constructs it (`v41_native_serve.rs:195`); `NativeTp4Wave::new`
  derives the planner from `transport.topology()` (`coordinator.rs:269-272`), so
  flagged requests always meet a topology-bound transport.
- Worker: `execution.rs:681-691` accepts only (flagged request, installed group)
  or (canonical request, no group) and errors otherwise; the installed group and
  shard index come from the single `V41SparkTopology` (`service.rs:160-172`,
  `rank % tp`, `world = tp`).
- Reducer selection: `enqueue_reduce_planes` (`coordinator.rs:516-535`) keeps the
  fixed 2/4 entry points and uses the generic entry for 3/6 with a 6-slot array
  of nulls beyond `planes.len()`.

**Open — Finding 2b: launcher and daemon approved sets differ.**
`V41SparkTopology::new` (`native_group.rs:87-94`) and the daemon accept
`TP2EP1`/`TP3EP1` (`v41_spark_topology.rs:132`), while
`release_validate_spark_topology` approves only `2x2|3x2|2x3|4x1`
(`release-common.sh:453-456`) and dies for `SPARK_TP=2 SPARK_EP=1`. Both reject
unambiguously, so it is not a correctness hole; the sets should agree or the
difference should be documented (launcher = release config, daemon = internal
API).

**Non-issue (was Finding 2c): post-mutation failure cannot happen.**
`with_native_group_owners` fully validates canonical shape and owner range before
touching any route word (`native_group.rs:312-326`), so the re-encode and final
`self.validate()` are provably infallible absent a bug in that prevalidation; no
request clone or extra allocation is warranted. Negative byte preservation is
covered by `failed_owner_encoding_leaves_the_request_byte_identical`
(`:993-1043`), which checks flags, route words and the full encoded frame after
six rejection classes and then retries successfully on the same object.

## Risk 3 — config forcing RTX count from Spark topology (fixed)

The earlier `release_validate_spark_topology` hardcoded `3x2 ⇒ RTX_GPUS must be
1`, all others ⇒ must be 2, contradicting `run.sh:187-189` ("the Spark TP/EP
degree does NOT select the RTX layout ... instead of a topology-to-layout
hardcode"). It was fixed during the review: the current function
(`release-common.sh:430-456`) validates keys, all-or-none, count, format and the
approved topology only — there is no RTX case left — and
`test_tp_ep_configuration.py:138-151` now asserts
`test_config_accepts_any_rtx_gpus_and_defers_to_the_budget`, with the resolved
weight-budget gate (`run.sh:225-231`) rejecting infeasible combinations. Verified
against the hashes above; relayed to the configuration owner by the parent.

## Additional observations

- No native link blocker by inspection: the new `v41_spark_tp2/tp3_experts.cc`
  wrappers macro-rename the six entry points, the fixed-name quantizer symbols
  are correctly excluded for the new macros (`v41_experts.cc:247-249`), and
  `ds41rt_v41_initialize_scratch_storage_async` is only forward-declared there
  (defined in `v41_route_reduce.cu:279`). The CMake role option is empty by
  default, so release defaults are unchanged.
- Role/geometry agree across exporter, header and FFI: `spark_tp2 → role 5,
  intermediate 1152`, `spark_tp3 → role 6, 768`, packer accepts 768/1152 with a
  `% 32` guard.
- Loader geometry is thorough (world 2..4, 32-value alignment, overflow, bounded
  W2 column reads). Caution: `BackboneTp.world` must remain the group TP degree,
  not the physical `TP*EP` world; the daemon passes `rank % tp` correctly, but
  the loader cannot detect a caller that passes the physical rank/world.

## Open items

1. GPU numerical proof of sentinel → zero through the in-tree `V41RoutePlan` /
   `V41SliceReduce` path (Risk 1; new harness pending).
2. Reconcile the launcher and daemon approved-topology sets (Finding 2b).

## Not reviewed / out of scope

- Exporter/kernel AOT generation internals beyond the in-tree route/slice plan.
- CUDA-gated self-tests; not built or run.
- `docs/tp-ep-*` audit documents and other agents' summaries; source only.
- Daemon serve behavior beyond the owner/shape/reducer call sites above; no
  daemon, service or GPU action was taken.
