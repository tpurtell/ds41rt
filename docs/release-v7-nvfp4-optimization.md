# V7 NVFP4 optimization investigation

## Status and scope

Work starts from DS41RT `90451c8` on `dev`, with the user's existing README,
EXL3 report, report-renderer, and official-regression report changes left alone.
This is an investigation and opt-in kernel experiment, not a qualified release
or a claim of recovered tokens/s. Nothing was pushed or published.

The official regression check in `release-v7-official-regression.md` is not
contradicted: these are different execution paths, not evidence of a v7 MXFP4
regression. W4A4's smaller operands do not guarantee less executed work or lower
latency when scheduling, quantization, memory traffic, and reduction differ.

## Claim verdicts (source at investigation baseline)

### (a) CONFIRMED: different kernel family and sparse decode parallelism

- `python/tools/export_b12x_v41_experts_aot.py:243` redirects Spark `fp8_k32`
  exports to slices, width 64 at capacity 1 and 192 otherwise. BF16 fallback
  still exists; this is not a claim that every official exporter branch uses slices.
- `native/cmake/v41_local_experts.cmake:21` and
  `native/cmake/v41_tp2_experts.cmake:25` choose width-192 slices for RTX roles.
- `python/tools/export_b12x_v41_slices_aot.py:110` creates `V41SlicePipeline`;
  `third_party/sparkinfer/b12x/moe/_shared/kernels/v41_slice_pipeline.py:23`
  creates `V41FusedSliceKernel`.
- `python/tools/export_b12x_v41_nvfp4_aot.py` calls `_get_dynamic_kernel` with
  NVFP4, ordinary SiLU, fused execution, and deterministic output.
- Baseline `dynamic.py:3323` collapses the deterministic fused task's intermediate
  slices into one group; `dynamic.py:4387` publishes `total_pairs * groups` tasks.
  One valid row/top-6 thus exposes six useful compute items, NOT six physical
  launch blocks. The Spark intermediate576 swapped path actually traverses five
  N128 intermediate tiles, including a 64-column tail (`dynamic.py:2668`).
- `w4a8_v41_slice.py:57` computes ceil(576/64)=9 slices and line 80 launches
  `(slices, groups)`: six distinct valid routes give 54 CTAs, not nine total.

### (b) CONFIRMED: existing V4.1 and prequantized paths are W4A8-only

Baseline `dynamic.py:853` sets `is_v41` only for `silu_v41`, line 857 assigns
five shards only for direct V4.1, and line 865 rejects non-native W4A8, swapped,
materialized, or nondeterministic execution. `_impl.py:11270` further restricts
prequantized input to repacked, deterministic W4A8 with E384/K5120/N640/top6 and
no shared-input mode. This is stronger than merely choosing a new activation.

Do not remove the guard: V4.1 also changes gate/up BF16 rounding, intermediate
rounding, quantization floor, and FP32 output behavior. Current NVFP4 requires
BF16 per-route outputs and its existing SiLU/clamp semantics. The experiment
below introduces an independent scheduling parameter without relabeling SiLU.

### (c) CONFIRMED, for hidden request payload only

`native/cmake/v41_experts.cmake:10` selects `fp8_k32` for Spark.
`rust/crates/ds41rt-daemon/src/v41_backbone_router.rs:18` chooses BF16/10240
bytes for NVFP4 and FP8K32/5280 otherwise; transport formulas are at
`rust/crates/ds41rt-transport/src/protocol_v2.rs:79`. Header, IDs, and router
weights are additional traffic, and responses remain BF16.

The NVFP4 producer reads BF16 and calls FP4 quantization at baseline
`dynamic.py:4226`, followed by the swizzled scale writes at line 4264. Official
prequantized input instead copies payload/scales at line 4104.

### (d) PARTLY CONFIRMED: forced M16, but the official comparison is imprecise

The baseline NVFP4 CMake command omitted `--tile-m`; exporter default16 was
passed both to scratch configuration and `planned_tile_m`, suppressing automatic
selection. This is a real latent NVFP4 ladder issue.

However, the shipped official slice pipeline is not the generic dynamic M-tile
ladder. The generic `silu_v41` selector (`_impl.py:1812`) uses only M16/M32;
NVFP4 (`_impl.py:1867`) has crossovers at 15/48/96 routed rows per expert, not
16/48/96. It also falls back from M64 to M32 for swapped mid-atom geometry
(e.g. Spark576, `_impl.py:1879`). At 256 tokens/top6, NVFP4 automatic selection
still chooses M16. Selection uses planned capacity, not an assertion about the
live chunk. Do not infer a present speedup from this configuration gap.

## Additional findings and prioritization

1. Remove unused FP8 router quantization first: small, local, preserves actual
   NVFP4 inputs. `v41_backbone_router.rs:421` previously ran the FP8 producer
   unconditionally, then selected BF16 for transport. The flag is immutable
   through the weight reference, cross-format rebind is rejected at line 353,
   and graph keys include exact weight ownership (`v41_layer_graphs.rs:32`).
   TP2 still copies the unused FP8 plane (`v41_experts/tp2_ffn.rs:275`); this
   change does not claim to remove that transfer or buffer allocation.
2. Test independent output-column sharding before an FP4 wire. It targets the
   observed six-task decode geometry while preserving each column's full
   intermediate traversal and rounding order. It duplicates FC1 work; only
   GPU timing can establish whether this tradeoff wins.
3. Expose the upstream tile planner without duplicating its thresholds. Keep
   M16 as the default until cross-role correctness and performance qualification.
4. FP4 wire/prequantization is not a safe drop-in. The generic protocol has an
   FP4 dtype, but `transport/v41_expert.rs:59` accepts only BF16/FP8K32 for V4.1.
   `v41_experts/nvfp4.rs:262` loads per-expert activation scales, and
   `dynamic.py:4088` indexes them by expert. A common quantized row requires
   proven uniform scales or a new validated representation. Six individually
   quantized rows would be 17280 bytes, worse than a 10240-byte BF16 broadcast.
   A single compact FP4 row would be 2880 bytes, but that is arithmetic, not an
   implemented or measured wire. The existing prequant parity test references
   removed API arguments and cannot establish current support.

This ordering is based on source evidence and correctness risk, not measured
end-to-end speedup ranking. No FP4 transport or activation gate relaxation is
included. CPU staging copies, BF16 route reduction, cooperative barriers, and
repeated FC1 quantization remain profiling targets.

## Changes

DS41RT commits: `b1f4aad` removes the unused router quantizer; `d43137c`
adds opt-in tile/sharding export, tests, and the updated fork pin. Documentation
is committed separately. Release defaults remain tile16/shard1; the router
change is active but has no end-to-end qualification claim yet.

- `rust/crates/ds41rt-daemon/src/v41_backbone_router.rs`: omit the unused FP8
  quantizer only for NVFP4; official and EXL3 conditions remain unchanged.
- `native/cmake/v41_nvfp4_experts.cmake`: expose validated
  `DS41RT_V41_NVFP4_TILE_M=auto|16|32|64|128`, default16.
- `python/tools/export_b12x_v41_nvfp4_aot.py`: auto delegates to b12x, compile
  uses the resolved scratch plan tile, and each manifest variant records it.
  Experimental `--output-shards` affects direct NVFP4 variants only; default1.
- `native/tests/v41_nvfp4_numerics_selftest.py`: use variant-level resolved tile
  for the independent public oracle, with legacy manifest fallback.
- Added exporter/tile and router regression tests under `python/tests/`.
- Deliberate fork experiment: independent NVFP4 output-column sharding in
  `b12x/moe/_shared/kernels/dynamic.py`, `_impl.py`, and the SiLU wrapper.
  Each task retains the complete ordered intermediate traversal; disjoint
  output ranges include corresponding FC2 weight AND scale TMA offsets.
  No change to BF16 route shape, TP2 reduction, 44-slot bridge, K16 swizzle,
  or [up; gate] weight order. Default shard1 preserves existing scheduling.

Fork provenance follows `third_party/README.md`: commit source, recompute tree
hash, update `third_party/sparkinfer.lock.json`, and run the existing verifier.
The verifier is not bypassed. Fork commits are `bfacb979` (sharding) and
`2bcbe122` (SiLU factory forwarding and durable regression tests). Final lock
revision is `2bcbe122bf34d77fecbaf288df2f395b9c09e79e`, source hash
`8b9474c3c5c29097be1efdbf007caaa65817cc7e71fc33e3de18c00953c59f89`.
Full local source verification passes. These commits are local and were NOT
pushed; a remote clean checkout cannot fetch them until separately authorized.

## Verification and measurements

GPU observed idle before export: two RTX PRO 6000 GPUs, driver595.91.07,
400W limits, 2MiB/12MiB allocated. Container CUDA13.2 and Torch
2.12.0a0+5aff3928d8.nv26.05. Ostrich SSH and GB10 are accessible.

- Initial scripts: **234 passed, 1 skipped, 1 failed**, 78 subtests. Existing
  `test_v6_published_images_and_full_model_are_release_defaults` expects v6
  coordinator while committed `ds41rt.config` uses v7. Not changed to hide it.
- Initial Python: **413 passed, 2 skipped**, 27 subtests, one existing NumPy warning.
- Added tile/router targeted tests: **33 passed**.
- Final Python suite after lock refresh and shard-argument tests: **454 passed,
  2 skipped**, 27 subtests. Final scripts rerun is unchanged: **234 passed,
  1 skipped, 1 failed**, 78 subtests.
- Actual fork activation-factory tests with GPUs hidden: **37 passed**; all
  five default families construct, and SiLU shard validation covers swapped
  and unswapped geometry.
- Loader unit tests: **82 passed, 8 ignored**; integration **11 passed**;
  doc tests zero. Ignored real-checkpoint tests were not claimed as passes.
- Coordinator Rust release rebuild inside WIP container: passed, existing warnings.
- RTX TP2 auto export rows1,1024,4096: passed; resolved M16,M32,M64.
- GPU numerical selftest: **3/3 invocations passed**, exact public-oracle output
  initially and after input/routing mutation, finite/nonzero output and three
  graph replays per invocation.
- GPU launch-contract selftest: **3/3 invocations passed** at those capacities;
  exercises all44 null-slot rejections, scalar/grid guards, and graph replay.
- RTX TP2 shard5 capacity1: AOT export and isolated native link pass;
  **1/1 numerical and 1/1 launch-contract invocation pass**, including exact
  initial/mutated oracle output and graph replay. Total GPU selftest invocations
  across tile/shard experiments: **8 passed, 0 failed** after correcting the
  explicitly recorded setup/integration failures. The oracle uses synthetic
  nonzero weights, not a real-checkpoint or output-column-randomized corpus.

Commands for isolated qualification (inside `ds41rt-coordinator-wip`):

```sh
python3 /wip/source/python/tools/export_b12x_v41_nvfp4_aot.py \
  --role rtx_tp2 --rows 1,1024,4096 --tile-m auto \
  --output-dir /wip/nvfp4-optimization/tiles-auto
# Link the generated objects with native/src/v41_nvfp4_rtx_tp2_experts.cc,
# native/cuda/kernels/v41_route_reduce.cu, CUDA and libcute_dsl_runtime.
python3 /wip/source/native/tests/v41_nvfp4_numerics_selftest.py \
  /wip/nvfp4-optimization/tiles-auto/libnvfp4.so \
  /wip/nvfp4-optimization/tiles-auto/v41_nvfp4_experts.json --rows 1
# Repeat --rows 1024 and4096; run v41_nvfp4_launch_selftest.py with
# prefix ds41rt_v41_nvfp4_tp2_expert for each capacity.
```

Initial isolated linking failed without libcute_dsl_runtime; corrected linking
resolved it. Auto-manifest oracle initially failed because it used top-level
null tile; the resolved-variant change above fixed it. Initial shard export
found a missing SiLU wrapper keyword; this was not accepted as qualification.
A Python suite attempt during fork editing failed closed at provenance
collection; it must be rerun after the final lock update.

### End-to-end throughput: not measured in this investigation

The user-supplied historical NVFP4 1x weighted/counting/code figures are
66.60/113.32/88.77 tokens/s; 2x are80.12/152.25/109.20, prefill7432.
These are reference values, NOT a newly measured baseline. There is no after
measurement and no tokens/s delta to report. Microkernel correctness is not
an end-to-end benchmark.

### Concrete deployment blocker

Installed `:v7` coordinator and Spark images have unequal engine-revision
labels. `run.sh:205` requires equality. Coordinator image ID/digest is
`sha256:53af6703391f15e6789d93410261efe53c7b1207ecce885a02d8c643bb491050`,
engine `25b670e8a05d721ce62958b861b45621905a0a9e`. All four Spark hosts have
image ID `sha256:94e2815f1739b9828aa88ff9d03a068aa7a0862537693d951e71c956b071a024`,
repository digest `sha256:81918fe41e2c3dc9eb99386516af9ba919ff15e3eaf6cd6e471e6634f4b1389b`,
engine literal `37db31f`. Docker inspection succeeded on every host. This is a
source-verified preflight incompatibility, not a failed serving launch.
The SparkInfer labels agree with each other at `63e2140e`, but that
does not override the engine guard. After our intentional fork update, the
current source lock also differs from the images; `run.sh:190` correctly
rejects that combination independently. No manual launch, relabel, or weakened
verification is used. A matching verified baseline image pair or a separately
qualified source-built baseline is needed before a comparable serving A/B.

## Remaining unverified

- End-to-end before/after benchmark (`--repeats 3`, fresh nonce seed and output),
  deterministic token equivalence, and official/EXL3 serving nonregression.
- Full native release artifact rebuild/package on both architectures.
- Spark576 swapped-tail and RTX full-width sharding qualification, real
  checkpoint tensors with nonuniform scales, multi-row direct capacities,
  adversarial route distributions, sanitizer, and timing.
- Automatic tile performance, other capacities, Spark fallback, and deployment
  chunk/capacity effects. Default M16 remains intentional pending evidence.
- FP4 activation transport/prequantized input; no scale-uniformity proof obtained.
- Other native selftests are not implicitly covered by the eight passing invocations above.
