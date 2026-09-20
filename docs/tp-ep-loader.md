# TP/EP loader: generalized official-checkpoint expert shard staging

Scope: `ds41rt-loader` only. This document describes the explicit generic TP
shard selection added to the native MXFP4 expert reader, its validated
geometry, and the measured tests. It does not change checkpoint files, native
kernels, the daemon, or the release topology.

Source of record: `rust/crates/ds41rt-loader/src/v41_expert_staging.rs`
(tests: `rust/crates/ds41rt-loader/tests/replicated_tp_staging.rs`).

## Selection API

`V41ExpertSelection` gains one additive variant; every existing variant keeps
its exact bytes, sizing and meaning.

```rust
pub enum V41ExpertSelection {
    Backbone { layer: usize, expert: usize, rank: usize },        // implicit TP4
    BackboneFull { layer: usize, expert: usize },                 // whole expert
    BackboneTp2 { layer: usize, expert: usize, rank: usize },     // RTX pair, TP2
    BackboneTp { layer: usize, expert: usize, rank: usize, world: usize }, // NEW
    Dspark { stage: usize, expert: usize },
    DsparkTp2 { stage: usize, expert: usize, rank: usize },
}
```

`BackboneTp` stages one shard of one official native FP4/E8M0 backbone routed
expert for an explicit TP world:

- `world` is the shard count, restricted to `2..=4`; `rank` must be `< world`.
- The shard carries `moe_intermediate_size / world` intermediate values.
- `W1`, `W3` and both scale planes are **output-row slices** (axis 0).
- `W2` weight and scale are **input-column slices** (axis 1) read with
  caller-owned scratch.
- The six-region source order is unchanged: `W1, W3, W2, S1, S3, S2`.
- `plan.intermediate_size()` is the per-shard intermediate width;
  `plan.tp_world()` returns `Some(world)` for this selection;
  `plan.geometry()` returns the full validated geometry.
- `V41ExpertStaging::backbone_tp_geometry(moe_intermediate_size, hidden_size,
  world, rank)` exposes the same validation without a catalog, for callers that
  plan shards before touching the checkpoint.

Only the official `deepseek-ai/DeepSeek-V4.1-Flash` MXFP4 checkpoint (native
FP4 weights with E8M0 group-32 scales) is accepted. `expert_staging` rejects
`BackboneTp` when the catalog carries an EXL3 manifest or a ModelOpt NVFP4
contract, and `nvfp4_expert_staging` has no generic selection at all.

## Validated geometry

Official constants: `moe_intermediate_size = 2304`, `hidden_size = 5120`.

| world | shard `I` | W1 or W3 weight | w1/w3 scale | W2 weight | W2 scale | plan bytes | W2 scratch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 1152 | 2,949,120 | 184,320 | 2,949,120 | 184,320 | 9,400,320 | 1,152 |
| 3 | 768 | 1,966,080 | 122,880 | 1,966,080 | 122,880 | 6,266,880 | 1,152 |
| 4 | 576 | 1,474,560 | 92,160 | 1,474,560 | 92,160 | 4,700,160 | 1,152 |

`plan bytes = 8160 * I` because all three weight slots and all three scale
slots have equal per-shard extents. The read scratch is always one full packed
W2 source row (`2304 / 2 = 1152` bytes); larger scratch only coalesces more
physical rows per read.

Validation performed before any read (`V41BackboneTpGeometry::new` and the
per-tensor plan checks):

- `world in 2..=4`, `rank < world`, non-empty intermediate/hidden.
- `hidden_size % 32 == 0` (E8M0 group-32 scale plane tiles the hidden axis).
- `moe_intermediate_size % world == 0` (whole output rows per shard).
- `(moe_intermediate_size / world) % 32 == 0`: the W2 input-column slice starts
  on a group-32 scale boundary (`(moe/32) % world == 0`).
- `(moe_intermediate_size / 2) % world == 0`: the packed W2 row (two FP4 values
  per byte) splits on a byte boundary.
- `shard_intermediate * hidden_size` and every staging/file offset are
  `checked_mul`/`checked_add`; staging regions stay 16-byte aligned.
- Each tensor's declared `byte_length` must equal the geometry extent, and the
  catalog placement must be a backbone routed-expert tensor with native
  FP4/E8M0 storage (`I8` weights, `F8_E8M0` scales).

`moe_intermediate_size = 768` split three ways (`I768`, 256 per shard) satisfies
all of the above and is covered by test `tp3_covers_i768_shards`.

## Integration

```rust
let selection = V41ExpertSelection::BackboneTp { layer, expert, rank, world };
let plan = catalog.expert_staging(selection)?;
let mut staging = vec![0u8; plan.staging_bytes()];
let mut scratch = vec![0u8; plan.minimum_read_scratch_bytes()];
plan.prefetch()?;                       // advisory read ranges only
plan.read_into(&mut staging, &mut scratch)?;
// plan.tensor_ranges()[slot] locates the six regions in W1,W3,W2,S1,S3,S2 order.
```

Callers that own a device staging allocation use `tensor_ranges()` unchanged.
Daemon-side lane mapping (for example a `BackboneTp` lane next to the existing
`BackboneTp2` lane) is a daemon change and is out of scope for this file.

## Behavior preservation

- `BackboneTp { world: 4 }` reads byte-identical shards to `Backbone { rank }`.
- `BackboneTp { world: 2 }` reads byte-identical shards to `BackboneTp2 { rank }`.
- `BackboneFull` and `Dspark`/`DsparkTp2` plans are untouched.
- `device_tensor_bytes`/`read_device_tensor_into` keep their TP4-hardcoded
  meaning; the generic path does not call them. It re-derives every extent from
  checked metadata and reads directly through the catalog snapshot with its own
  row/column slicing, so release TP4 behavior cannot be perturbed by this
  change.

## Test evidence

Environment: commit `bec5fcc`, rustc 1.98.1, 2026-09-20, offline cargo
(`CARGO_TARGET_DIR` left at the shared `rust/target`, no isolated target
directory), no GPU. Official snapshot resolved through the Hugging Face ref
`refs/main -> dba1be0a40aa45a94ad051997016db3960a90277` (48 native shards).
`git diff --check` is clean.

```
cargo test --offline -p ds41rt-loader
  lib unittests:                 82 passed; 0 failed; 8 ignored
  tests/replicated_tp_staging:    5 passed; 0 failed
  tests/upstream_container_invariants: 11 passed; 0 failed

/usr/bin/time -v cargo test --offline -p ds41rt-loader --test replicated_tp_staging \
  generic_world_shards_match_raw_official_source_bytes
  test result: ok. 1 passed ... finished in 1.33s
  Maximum resident set size: 128,244 KB   File system inputs: 48
```

The five tests:

1. `explicit_world_2_3_4_shards_partition_the_official_expert_exactly` —
   catalog-free geometry: adjacent shards tile every plane exactly, adjacent W2
   byte/group column windows are disjoint and contiguous, offsets stay
   byte/group aligned.
2. `tp3_covers_i768_shards` — `I768` split three ways.
3. `explicit_tp_geometry_rejects_invalid_world_rank_and_shapes` — invalid world
   (`0,1,5,8,MAX`), invalid rank, non-tiling intermediate (`2300/2`), non-tiling
   hidden (`5119`), empty geometry, and extent overflow (`32 * (usize::MAX-31)`).
4. `generic_world_shards_match_raw_official_source_bytes` — real checkpoint,
   layer 0 expert 0: for `world` 2/3/4 and every rank, each of the six regions is
   compared against (a) an independent oracle that slices the raw safetensors
   bytes with test-local arithmetic, and (b) the pre-existing fixed selector
   (`Backbone` for `world=4`, `BackboneTp2` for `world=2`). The regions are then
   reassembled across ranks (concatenation on axis 0, row interleave on axis 1)
   and must equal the `BackboneFull` bytes. Plan sizes, scratch bounds, source
   order, short-buffer rejection, invalid selections and the unchanged legacy
   selector bounds are all asserted.
5. `generic_selection_is_rejected_for_exl3_and_nvfp4_publications` — loads the
   cached `diffbot/DeepSeek-V4.1-Flash-EXL3-2.0bpw-2x-RTX-PRO-6000` and
   `nvidia/DeepSeek-V4.1-Flash-NVFP4` catalogs and asserts `BackboneTp` (and
   NVFP4 staging) refuse them.

The real-checkpoint and compressed-publication tests soft-skip when the
corresponding snapshot is absent (`DS41RT_OFFICIAL_SNAPSHOT` can force the
official path); no test writes to the checkpoint cache.

## Assumptions and limits

- No checkpoint conversion, padding, requantization or repacking happens here;
  the native packer keeps handling any runtime padding elsewhere.
- Generic staging is only defined for the official native MXFP4 contract. It is
  not a general model loader: worlds other than 2/3/4, non-group-32 scales and
  non-packed weights are rejected rather than guessed.
- `read_into` on failure may leave partial staging; callers must not pack unless
  it succeeds (unchanged contract).
- Only layer-0/expert-0 was read end-to-end in the default real-checkpoint test;
  the geometry and offset proofs are shape-independent and all 40 layers/384
  experts share the same tensor shapes.
- A compact synthetic *catalog* fixture is not possible through the public API:
  `OfficialV41Catalog` has no public constructor and `read_official_v41_catalog`
  validates the complete 96,085-tensor official inventory against
  `model.safetensors.index.json`, so a hand-built snapshot cannot yield a
  catalog. The catalog-free `V41ExpertStaging::backbone_tp_geometry` tests cover
  the partition/alignment/overflow algebra instead, and the real-checkpoint test
  covers the actual reads. If a public test catalog constructor is added later,
  the geometry tests can be paired with synthetic byte-level reads.
