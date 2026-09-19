# V7 NVFP4 optimization investigation

## Published-pair follow-up (source c2ccf16)

The previous image-revision blocker is fixed. `run.sh --config
/tmp/ds41rt-nvfp4-ab.config --rtx-gpus 1 --dry-run` passes with both roles at
engine `0107d01e3d35d22b1dbc5de70c4e1a32d32d165f` and SparkInfer
`2bcbe122bf34d77fecbaf288df2f395b9c09e79e`. Historical status below is retained
as an investigation record, not the current deployment status.

### New published 1x baseline

Checkpoint: `nvidia/DeepSeek-V4.1-Flash-NVFP4` snapshot
`3431dde3247c13b5957f682b1e3c6fcae2566079`. One RTX GPU0 plus four Sparks,
dSpark on, prefill batch2048/capacity4096, concurrency16, prefix entries20.
Coordinator digest `sha256:d85608bbe14ce655a3bae4fd61655a0020099f056d357a40b8a3ac5115275c5e`;
expert digest `sha256:aa477ff1c74fe4e815734c1e57d3fad325c5126ddbb324c92fd58f6d02fe856b`.

The launcher preflight was used unchanged; the actual launch uses the user's
RDMA/capability/security shape. Exact launch commands and configuration are in
`measurements/nvfp4-v7-ab/ds41rt-nvfp4-baseline-launch.sh` and
`measurements/nvfp4-v7-ab/ds41rt-nvfp4-ab.config`. No image guard was bypassed.

```sh
bash /tmp/ds41rt-nvfp4-baseline-launch.sh
.venv/bin/python scripts/bench-ds41-release-decode.py \
  --base-url http://127.0.0.1:8000 \
  --tokenizer /home/tj/.cache/huggingface/hub/models--nvidia--DeepSeek-V4.1-Flash-NVFP4/snapshots/3431dde3247c13b5957f682b1e3c6fcae2566079/tokenizer.json \
  --label nvfp4-v7-published-1x-m16-shard1 --repeats 3 \
  --nonce-seed 198473621 --include-counting \
  --output /tmp/nvfp4-v7-baseline-198473621.json
```

| Metric (tok/s) | Repeat1 | Repeat2 | Repeat3 | Median |
| --- | ---: | ---: | ---: | ---: |
| Weighted decode | 65.820441 | 67.048520 | 64.711036 | 65.820441 |
| C1 code | 88.320296 | 88.625795 | 89.361311 | 88.625795 |
| Counting | 111.849745 | 113.749206 | 113.533224 | 113.533224 |

Benchmark exit0, **30/30 samples pass**, complete weighted corpus on all three
repeats. Raw requests, output, timing and pass/fail are retained in
`measurements/nvfp4-v7-ab/nvfp4-v7-baseline-198473621.json`; console log adjacent.
Prefill throughput is not measured by this decode workload.

### Opt-in A/B result: no demonstrated improvement

Both roles were fully rebuilt inside the WIP CUDA containers, serially, using
`scripts/build-release-artifacts.sh`, SM120 coordinator then SM121 expert.
Isolated `/wip/nvfp4-ab-source` differs only in its NVFP4 CMake policy:
`DS41RT_V41_NVFP4_TILE_M` default changed to `auto`, and exporter receives
`--output-shards 5`. The checked-in CMake/defaults are unchanged. This is an
export/build-time lever, not a serving environment variable. Exact command
record: `measurements/nvfp4-v7-ab/experiment-commands.sh`; both full build logs
and the isolated CMake file are adjacent. Both builds exit0.

Only the native library was copied into each stopped experiment container;
published daemon, model, launch options, hardware and EXL3 packages remained
unchanged. Full library rebuild is a qualification caveat versus a binary
patch of just the NVFP4 symbols. No container was published or retagged.

Native-library SHA256:
- Coordinator: `ed35a345a65e51859c425ca8b863df22624acf49ac4c99f2327a789611cc0f63`.
- Expert: `1e4031f49bd01023f80f78a2ad947fa6b936d945bc232b3122dcd9f7d08c935d`.

Retained manifests prove capacities1/16/80/256/1024/4096 resolve to tiles
16/16/16/16/32/64 for both RTX roles, 16/16/16/16/32/32 for Spark.
Only direct capacity1 uses shard5; grouped capacities retain shard1. Thus
this is a combined opt-in policy experiment, not separate attribution of
sharding and tiles, and decode does not exercise the larger tile ladder.

| Opt-in metric (tok/s) | Repeat1 | Repeat2 | Repeat3 | Median | vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| Weighted decode | 63.241250 | 64.272685 | 65.388733 | 64.272685 | -2.35% |
| C1 code | 87.389432 | 88.004415 | 89.918810 | 88.004415 | -0.70% |
| Counting | 114.107198 | 113.888325 | 113.508252 | 113.888325 | +0.31% |

Successful run seed198473623, repeats3, fresh output
`/tmp/nvfp4-v7-optin-198473623.json`, exit0, **30/30 pass**. Raw JSON/log
are committed adjacent to baseline. Nonce seeds differ deliberately; this
is the same corpus and controls, not token-identical requests or a proof
of deterministic real-checkpoint token equivalence. Three repeats do not
establish statistical significance. Counting's tiny gain is not evidence
of a useful optimization; weighted decode is worse. Keep defaults M16/shard1.

| Reference (tok/s) | Weighted | Counting | C1 code | Prefill |
| --- | ---: | ---: | ---: | ---: |
| Historical NVFP4 1x | 66.60 | 113.32 | 88.77 | not measured |
| New published baseline 1x | 65.82 | 113.53 | 88.63 | not measured |
| New opt-in 1x | 64.27 | 113.89 | 88.00 | not measured |
| Historical NVFP4 2x | 80.12 | 152.25 | 109.20 | 7432 |
| Official MXFP4 1x reference | 92.00 | 161.58 | 130.41 | 7824 |
| User's latest official 1x check | 93.57 | 166.05 | 134.38 | not supplied |

The opt-in 1x does not beat recorded v7 1x or the official 1x path. The 2x
row is historical context, not a same-topology A/B; no new 2x or prefill
measurement was made. No claim that these changes close the W4A8 gap.

### Qualification, failures and operational cleanup

- GPU numerical and launch-contract tests: **18/18 invocations passed**:
  three roles (RTX backbone, RTX TP2, Spark), capacities1/1024/4096,
  numerical + launch test each, device0. Exact initial/mutated public-oracle
  agreement, graph replay and launch guards pass. Logs retained. These are
  synthetic weights, not exhaustive/random checkpoint-scale qualification.
- Python suite: **454 passed, 2 skipped**, 27 subtests, existing NumPy warning.
- Scripts suite: **235 passed, 1 skipped**, 78 subtests. Both suites exit0.
- Host system Python initially lacked `tokenizers`; `.venv/bin/python`
  provides0.23.1 and ran both benchmark arms. No host CUDA build attempted.
- Initial opt-in run seed198473622: **0/30 pass**, exit1, every sample reports
  `IncompleteStreamError('incomplete SSE: done=False, first=None, finish=None, usage=None')`.
  `/v1/models` was available but did not establish inference readiness.
  Shortly afterward an eight-token streaming Hello probe completed through
  `[DONE]`; rerun used a fresh seed/path without any library/service change.
  Startup readiness is the likely explanation, not a proven root cause.
  Failed JSON/log retained; no throughput credited to that attempt.
- Retained `/wip/build` native libraries differed from published v7, so they
  were not reused for linking. Full clean builds avoided that provenance risk.
- All five experiment serving containers are stopped; original v7 native
  libraries restored. Three remote WIP containers lacked `expert-v7` outputs
  during initial restoration; copied the original from ostrich instead.
  RTX memory is back to2MiB/12MiB. Published images and source defaults untouched.

No FP4 wire, prequantized input, or `silu_v41` gate relaxation was implemented:
per-expert scale semantics and V4.1 rounding remain the concrete correctness
risks described below. This result does not justify broadening that scope
without independent numerical work. Official MXFP4/EXL3 serving was not
rerun; their source policies and installed published images were not changed,
but unchanged configuration alone is not a new performance-regression test.
Remaining work is prefill-specific A/B, 2x qualification, isolated lever
attribution and real-checkpoint equivalence. There is no image-revision blocker.

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

### Load-time padding investigation (September 19)

The native MXFP4 loader already pads Spark's logical 576-wide intermediate to
640 and repacks it for the W4A8 kernel. NVFP4 retained width 576, which selects
b12x's transposed (`swap_ab`) FC1 and disables its fused gate/up FC1 path.
Zero-padding each gate/up half and the down projection to 640 selects the
regular W4A4 path without changing checkpoint values or activation scales.

Preliminary **component** results, synthetic nonuniform FP4 payloads,
per-expert activation/weight scales, E=384/H=5120/top-k=6, capacity 16/live 8,
tile M16. Timings are CUDA-event medians of nine batches, each 200 captured
kernel executions. These are warm repeated routes, not serving throughput.

| Device / routing | Baseline µs | Candidate µs | Candidate |
| --- | ---: | ---: | --- |
| Spark, six shared experts | 581.04 | 293.48 | Identical weights zero-padded 576→640 |
| Spark, six shared experts | 293.48 | 191.22 | Padded 640 + grouped output shards 5 |
| RTX TP2, six shared experts | 368.73 | 169.23 | Grouped output shards 5 |
| RTX TP2, 48 distinct experts | 366.45 | 448.35 | Grouped output shards 5: regression |
| RTX TP2, 48 distinct experts | 365.62 | 349.71 | Grouped output shards 2 |

Every compared route output was bitwise identical, including captured replays.
Padding baseline/candidate were separate processes with the same seed 43;
sharding arms were timed in alternating order within one process. This is not
an independent mathematical oracle or real-checkpoint quality qualification.
Grouped sharding required removing its direct-routing-only guards in an
isolated b12x copy; those changes are not in the pinned submodule. Fixed
five-way sharding is unsuitable as a blanket default given low-reuse results.

Reproduction tool: `python/tools/bench_nvfp4_routing.py`, inside the appropriate
CUDA container. For padding, run `--intermediate 576 --capacity 16 --live 8
--save-output /path/original.pt`, then the same command with `--pad-to 640
--reference /path/original.pt` (omit `--save-output`). Both use the pinned
SparkInfer revision 2bcbe122 by default. `--routing distinct` probes low reuse.
`--source` selects an isolated kernel experiment; `--cases grouped:1,grouped:5`
requires that experiment's grouped-sharding support.

An opt-in native loader/AOT candidate now uses
`DS41RT_V41_NVFP4_PAD_INTERMEDIATE=ON`. Its metadata retains logical 576 and
publishes kernel 640; the loader derives allocation sizes from that metadata,
pads on GPU directly into resident planes, and swizzles scales in the same
load-time operation. No saved conversion or per-token repacking is needed.
Padding adds approximately 0.198 GiB per resident routed-expert layer per Spark
(7.91 GiB if all 40 layers reside there); budgets account for the actual width.
The default remains unchanged pending full serving and loading measurements.

The native padding byte-layout gate passes on SM120 and SM121 for 64→128,
576→640, and 1152→1152, including every plane, zero padding, source immutability,
output guards, and undersized-buffer rejection. Rust daemon `cargo check`
passes with Python 3.12; 41 focused Python tests pass. The padded Spark AOT
exports all six capacities 1/16/80/256/1024/4096 with logical 576/kernel 640.
Native Spark ABI/public-path comparisons at rows 1, 16, and 1024 pass exactly,
including mutated inputs/routing and three graph replays each. Both full
coordinator and Spark WIP builds pass. Serving qualification remains outstanding.

RTX-side activation quantization / NVFP4 transport are deferred at the user's
request; this phase focuses on layout, tiling and kernel parallelism.

### Padding serving A/B (September 19; development only)

The source-built candidate passes the targeted serving comparison. This is a
useful improvement, **not a release qualification**: code still trails the
previous official-checkpoint result of approximately 130 tok/s. Padding remains
opt-in. No release image or tag changed.

One RTX PRO 6000 at a 400 W power limit and standard memory settings, four
Spark ranks, dSpark enabled; identical coordinator/daemon and checkpoint in both
arms. Only the Spark AOT library changes logical/kernel width 576/576 to
576/640. Both arms retain the same current activation scaling and BF16 wire.
Source implementation: `5ba005e` (build frozen before final documentation and
manifest-only edits). Artifact hashes, launch configuration, raw results and the
failed trial are in [the evidence directory](measurements/nvfp4-padding/).

| Measurement (median of three) | Unpadded | Padded | Change |
| --- | ---: | ---: | ---: |
| C1 weighted nine-category decode, tok/s | 63.83 | 78.85 | +23.5% |
| C1 code, tok/s | 86.97 | 105.36 | +21.1% |
| C1 reasoning code, tok/s | 74.54 | 91.06 | +22.2% |
| C1 topic, tok/s | 50.88 | 64.25 | +26.3% |
| C1 counting, tok/s | 112.55 | 132.11 | +17.4% |
| C2 code aggregate, tok/s | 131.16 | 151.33 | +15.4% |
| C8 code aggregate, tok/s | 366.43 | 443.64 | +21.1% |
| C16 code aggregate, tok/s (512 output limit) | 398.42 | 615.76 | +54.5% |
| 32K fresh prefill, effective tok/s | 4038.40 | 5083.16 | +25.9% |

All 30 C1 requests per arm pass the corpus structural checks; all paired content
hashes **and reasoning strings** match exactly. This does not substitute for a
full tool/quality evaluation. C2/C8 use the original 320-token output limit.
The first padded C16 trial hit that limit in three responses before closing the
code fence; it remains recorded as failed. Both C16 arms were rerun with 512,
passing all three batches. Concurrency outputs may differ across schedules;
C16 ranges were 338.98–432.25 unpadded and 512.47–621.61 padded. Treat the
concurrency improvement as noisier than C1, not a universal speedup estimate.

Prefill uses the same frozen README text, tokenizer, 32768-token suffix and
zero base, with one excluded warmup and three measured runs per arm. Expert
startup (container start to loaded listener, one paired restart) had medians
41.40 s unpadded and 42.45 s padded across the four hosts. This small observed
increase is not a repeated loading qualification. The API model listing can
precede backend readiness: early warmup streams failed during expert loading;
subsequent complete warmup succeeded before measurement.

Reproduction: `bench-ds41-release-decode.py --repeats 3 --nonce-seed 198474001
--include-counting`; concurrency uses `--case code --repeats 3 --label
nvfp4-padding-ab --nonce padding-layout-fixed`, with `--concurrency 2 8 16`
initially and `--concurrency 16 --max-tokens 512` for the corrected paired C16.
Prefill uses `--base 0 --suffix 32768 --repeats 3 --warmups 1`. Pass the same
API URL, tokenizer and frozen context file to both arms; raw JSON records their
identities. This supersedes the outstanding serving item above only for this
NVFP4 1×RTX padding comparison; native nonregression, 2×RTX and release gates
remain unverified.

### Adaptive output splitting candidate (September 19)

Fork revision `4b095414` adds opt-in `nvfp4_output_shards=0`: after the existing
non-streaming routing barrier, each CTA reads the published task count and
chooses the largest divisor of 40 up to eight that fits one resident-grid wave.
Only the consumer work domain expands. The decision is GPU-local, depends on
actual routes, and adds no host feedback, lane join, workspace or compile key.
Default shard count stays one; native W4A8 keeps its existing policy.

`DS41RT_V41_NVFP4_OUTPUT_SHARDS=0` exposes this in native builds (default one).
The exporter records zero for adaptive direct and grouped variants; existing
positive values retain direct-only behavior. This candidate is not yet serving
qualified. Padding remains a separate option. Prioritize splitting and tiling;
materializing FC1 intermediates is deferred because additional memory traffic
could hurt Spark.

Component timings below use the same diagnostic geometry/method as above,
capacity 16/live 8/M16 unless marked direct (capacity/live one). No end-to-end
speedup is inferred. [Raw probe logs](measurements/nvfp4-adaptive/) record source
hashes; the committed kernel differs from the prototype only in comments and
validation error text.

| Probe | Shard one µs | Adaptive µs |
| --- | ---: | ---: |
| RTX N1152, six shared experts | 368.54 | 161.36 |
| RTX N1152, 48 distinct experts | 365.69 | 350.36 |
| RTX N2304, six shared experts | 709.16 | 286.00 |
| RTX N2304, 48 distinct experts | 725.78 | 679.16 |
| Spark padded 640, six shared experts | 296.49 | 200.88 |
| Spark padded 640, 48 distinct experts | 1330.55 | 1334.62 |
| RTX N2304, direct one row | 417.70 | 239.40 |
| Spark padded 640, direct one row | 198.68 | 173.87 |
| RTX N1152, capacity80/live33 distinct | 1800.03 | 1799.99 |
| Spark padded640, capacity80/live33 distinct | 5439.56 | 5443.91 |

Every comparison passes bit-exact route outputs. Updated diagnostic tooling
also mutates inputs and route sharing under the same captured graphs, and
`--check-live-counts` exercises already resolved kernels at live 1/33/80 with
three graph replays each. These pass on both architectures. Constructor tests
pass 55 cases; exporter/tile/router tests pass 50 cases. This remains a synthetic
kernel comparison, not an independent mathematical oracle. Native artifact
builds, ABI checks, serving A/B and native nonregression remain next.
