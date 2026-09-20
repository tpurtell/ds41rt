# Final integration review — replicated Spark TP/EP (source/CPU)

Verdict: **bounded source integration acceptance; no concrete correctness bug or
regression found.** This is a source-level integration acceptance only — **not** a
quality or release acceptance. The frozen E2E quality gates still report FAIL
(greedy drift), and **no release qualification is claimed**. The remaining
width-192/M64 kernel comparison has since been **measured and independently
accepted (96/96)** with a **retention decision recorded** (keep cap80/w192 and
cap1/w64; no extra AOT build; no default-serving change); AOT adoption remains
unapproved on its own. The ledger is independently closed
(`bce43420…`/validator `d9632e81…`), the **V2 caller repair is verified**
(`1dc81573…`; audit `08a7d856…`, no blocking items), and the fleet is restored and
verified against the original five containers. **Scoped implementation,
measurement, selection and restoration are complete**; only optional future
hypotheses (non-default widths, AOT adoption, distributed/DRAM measurement) remain,
and they are not completion gates. No implementation, build, Cargo, hardware,
checkpoint read or new run was performed; unrelated WIP is untouched.

## Files reviewed (sha256 prefixes)

| File | sha256 |
| --- | --- |
| `rust/crates/ds41rt-core/src/replicated_expert_schedule.rs` | `c257795cecee2117…` |
| `rust/crates/ds41rt-transport/src/v41_expert/native_group.rs` | `6bf27a07310f7ad1…` |
| `rust/crates/ds41rt-transport/src/v41_expert.rs` (parse/filter/copy) | reviewed functions |
| `rust/crates/ds41rt-transport/src/v41_expert/roce.rs` | `d5c00d3f85b79a64…` |
| `rust/crates/ds41rt-transport/src/v41_expert/chunks.rs` | `a2d7ca5d5140921a…` |
| `rust/crates/ds41rt-daemon/src/v41_experts/coordinator.rs` | `959a637dfd9e10b8…` |
| `rust/crates/ds41rt-daemon/src/v41_spark_topology.rs` | reviewed |
| `rust/crates/ds41rt-loader/src/v41_expert_staging.rs` | reviewed |
| `rust/crates/ds41rt-ffi/src/v41_experts.rs` | `ca88463496697f37…` |
| `native/cuda/kernels/v41_route_reduce.cu` | `0c12ff6ff6edc489…` |
| `native/src/v41_experts.cc` | `5d390d0cb7e96e14…` |

## Invariant chain (scheduler → wire → worker filter → rank reduction)

1. **Scheduler** (`replicated_expert_schedule.rs`, `coordinator.rs:128-165`):
   whole-expert LPT assignment; `owners[expert]` is the owning group for every
   assigned expert; every *routed* expert is validated `owner < group_count()`
   before any wire mutation. Default tie-seed mode is byte-identical to the
   historical seed (separate mode review).
2. **Wire** (`native_group.rs:56-59`, `287-339`): 9-bit expert field
   (`EXPERT_MASK=511`, covers 0..383), one-hot owner at bits 9-11
   (`OWNER_SHIFT=9`, `OWNER_MASK=0b111`), reserved bits rejected
   (`WORD_MASK=0xfff`); group flag `1<<18` distinct from the paired EXL3 `1<<17`
   and masked out of responses (`v41_expert.rs:338`). The re-encode preserves the
   true expert id and the exact FP32 gate weight bits, validates fully before
   mutating, and leaves the request byte-identical on failure (tested).
3. **Worker filtering** (`v41_expert.rs:294-322`): requires a topology-admitted
   request; `V41NativeOwnershipBatch::observe` enforces one owner per expert
   within the batch (conflicts rejected); owned route → exact expert id + exact
   `gate_weight`; non-owned → sentinel `384` + `0.0`. The sentinel is a
   worker-local kernel id and never appears as a wire word.
4. **Rank reduction** (`v41_route_reduce.cu:87-135`, `242-272`):
   `reduce_compact<Ranks>` starts at `p0` and applies ordered `__fadd_rn` through
   `p5`, adds `shared` once, then a single `__float2bfloat16_rn`. Dispatch accepts
   exactly 2/3/4/6; `valid_compact_planes` checks nulls, 2-byte alignment, uint64
   extent/overflow, plane↔output overlap, and that inactive slots are null. FFI
   `require_rank_count` (`v41_experts.rs:296-326`) gates 3/6 on the new symbol and
   is called at `NativeTp4Wave::new` before any allocation or readiness, so a
   missing artifact fails startup, not the first request.

## Runtime capacity selection and budget math

- Supported capacities are exactly `1|16|80|256|1024|4096`
  (`service/local.rs:28`); the AOT metadata is selected **by capacity** through
  the per-role native `ds41rt_v41_spark_tp2/tp3_expert_info` (FFI interfaces 8/9),
  validated `abi_version ∈ {2,3}` and `hidden_size == 5120`
  (`v41_experts.rs:572-607`). Capacities below the compile-time atomic threshold
  (256) use the FP32-route ABI; ≥256 use token accumulation.
- Budget: `NativeTp4Wave::device_bytes_for(capacity, ranks) =
  capacity * (ranks * V41_PARTIAL_ROW_BYTES + 2 * 5120 * 2)` with `checked_mul`,
  `ranks ∈ {2,3,4,6}`; the legacy accessor is exactly the 4-rank value
  (`coordinator.rs:279-297`). Six-rank layouts therefore reserve their own two
  extra planes rather than reusing the 4-rank reserve.

## Topology config consistency

`v41_spark_topology::resolve` is all-or-none, validates the pair through
`V41SparkTopology::new` (only TP2EP1/TP3EP1/TP4EP1/TP2EP2/TP3EP2/TP2EP3), and
requires `world_size == TP*EP`; `group = rank/TP`, `tp_rank = rank%TP`,
`global_rank = group*TP + tp_rank` (group-major, `native_group.rs:116-133`).
Executor namespaces are disjoint across layouts (TP4EP1 1-4, TP2EP1 5-6, TP3EP1
7-9, TP3EP2 11-16, TP2EP2 17-20, TP2EP3 21-26), so a response from another
topology is rejected by identity. The worker staging selection passes
`world = TP, rank = rank % TP` (asserted in `service.rs:460-471`), so each rank
loads the TP shard of every expert while ownership decides which experts it
serves; `require_native` rejects EXL3/NVFP4 for an explicit topology, and roles
5/6 are the new shard families.

## Test coverage of the changed files

- Transport `native_group.rs`: topology mapping/namespaces, unsupported layouts,
  owner-word round-trip and reserved bits, every supported topology
  encodes/owns exactly once, inactive group/expert sentinels, builder rejects
  missing/conflicting/oversized owners, parser rejects overflow/reserved/
  duplicates/bad weights, six-rank receiver and chunked-coverage rejection of
  stale/foreign/reordered planes, 3/6-plane collection, flag/world gating.
- Daemon `coordinator.rs`: six-route group split, seed-mode wiring, EP1,
  duplicate-route rejection, row-reuse weight masking, empty-group zero plane,
  reproducible planning, reservation scaling by rank count, missing-plane bounds.
- Native CTest: `v41_route_reduce_planes_selftest`, `v41_expert_pack_tp3_selftest`,
  `v41_peer_copy_selftest` are registered (`native/CMakeLists.txt:997-1006`).
- The established evidence (387+59 CPU, native 135, all five E2E executed) is not
  re-run or re-counted here.

## Bounded acceptance and limits

- **Accepted (bounded)**: the source-level integration contract above, for the
  requested official-checkpoint TP/EP layouts, with the invariants test-covered
  and no integrated owner map required.
- **Not accepted / not claimed**: E2E quality or release readiness (frozen gates
  still FAIL on greedy drift); **AOT adoption** (the width-192/M64 comparison is
  measured and independently accepted, but it is JIT rank-local projected data and
  does not justify AOT by itself); any end-to-end performance claim.
- **Closed since this review**: fixture relabel (correct **384 routes / 48 distinct
  experts globally across EP groups** description), V2 caller repair (`1dc81573…`;
  audit `08a7d856…`, no blocking items; focused 80 pass / 5 CUDA skips / 0 failed
  with CUDA hidden), ledger confirmation (`bce43420…`; validator `d9632e81…`), and
  fleet restoration/verification against the original five containers.
- **Optional future hypotheses (not completion gates)**: non-default widths, AOT
  adoption, and distributed/DRAM measurement.
- Minor notes (not bugs): `V41BackboneTpGeometry` bounds `world` to 2..=4 — that
  is the **TP degree**, so six-rank EP2/EP3 layouts are unaffected;
  `supports_rank_count(4)` is unconditional because the 4-plane entry point is the
  base path, while 3/6 are symbol-gated; the sentinel `384` fits the 9-bit expert
  field and stays worker-local.

## Artifact/source provenance (owner wire) — caveat

The one-hot owner encoding reviewed here is the encoding the executed artifacts
used; there was **no post-E2E wire change**.

- `native_group.rs` on disk is `6bf27a07…` (one-hot, `1 << (9+owner)`,
  `OWNER_MASK=0b111`) and its mtime is 03:13 UTC, before every build below; it is
  the only copy in the tree.
- `runs/tp-ep-preflight/g1-live/frozen-snapshots/pre-diag-v2/frozen-rust-files.txt`
  (captured 07:23 UTC) already lists
  `6bf27a07310f…  rust/crates/ds41rt-transport/src/v41_expert/native_group.rs`.
- `…/pre-diag-v2/diag-v2-identity.txt` records the diag-v2 build as differing from
  the pre-diag manifest only in `v41_experts/execution.rs`, `service.rs` and
  `service/local.rs` (native libs unchanged), so the ARM diag-v2 binary kept the
  same `native_group.rs`.
- `runs/tp-ep-preflight/logs/seed-build-x86.log` built `6abe8ad2…` at 08:28 UTC,
  after that file's last modification, and the seed-mode patch touched only
  `errors.rs`, `lib.rs`, `replicated_expert_schedule.rs` and `coordinator.rs`. The
  E2E reports tie `6abe8ad2…`/`5162bc8a…` to source manifests `517c8620…`
  (diag-v2) and `8afcb464…` (seed).

The earlier "binary owner index at bits 9..11 (not one-hot)" text in
`docs/tp-ep-implementation-plan.md` (lines 204, 877, 976, 1131) is **stale
documentation**, not a wire change; it even cites `native_group.rs:56-59,213-275`,
which implement the one-hot mask. `docs/tp-ep-architecture-audit.md:44,488` states
the one-hot bitmap consistently with the source. No rebuild or artifact gap.
