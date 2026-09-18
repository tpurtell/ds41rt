# V7 homogeneous K2 versus mixed K2/K3 experiment

## Scope and reachability

The diffbot uniform-K2 checkpoint can use SparkInfer's genuine homogeneous
kernel without changing the mixed compute kernels or their artifacts. This is
**not** a `[2]` flag for `export_b12x_v41_exl3_aot.py`, nor a duplicate-tier
`[2,2]` workaround.

The entry is `compile_w4a16_fused_moe()` → `W4A16FusedMoeKernel` in
`third_party/sparkinfer/b12x/moe/_shared/kernels/w4a16/kernel.py`, with
`trellis_t256`, `trellis_t256_proj`, K2/MCG, FP16 compute, BF16 rotation input,
full/intermediate rotation, and SwiGLU limit 10. A separate
`compile_w4a16_topk_sum()` provides the FP32 inverse-rotation/router sum.
The public trellis preparer dispatches MCG to the projection path; its other
uniform helper hardcodes N256, which does not divide the TP2 width 1152.
The low-level homogeneous compiler supports N128 and needs no vendor edit.

An isolated exporter/bridge is needed because the homogeneous core takes one
weight tuple, no projection descriptor, and two Int64 weight-element counts.
The mixed bridge accepts only Int32 scalars. The experimental bridge bakes the
fixed word counts (283,115,520 for W13; 141,557,760 for W2), maps pointer names
onto the existing Rust launch tables, and emits true `bits: [2]` metadata.
Homogeneous FC1/FC2 share a scratch allocation; its gate/up rotation buffers
are separate. The mixed scratch alias scheme is different and was not reused.
The existing checkpoint payload order—every gate expert followed by every up
expert—is directly compatible. No requantization or weight conversion occurs.

Two temporary Rust admission changes selected `[2]` only when
`DS41RT_EXL3_K2_HOMOG=1` and permitted that explicit family in native info.
With the variable unset/zero, the original `[2,3]` path remained selected.
No CMake, release-artifact script, `run.sh`, or mixed exporter was modified.
Only TP2 H5120/I1152/E384/top-k6/FP32 capacities 1,16,80,256,1024,4096 were
exported, with packed routing and the existing projection tile policy
(N128; K128 at capacity16, K64 otherwise). All selected one block/SM.

## Isolation and correctness

- Base engine checkout: `df73a79189a257be595ac7a69d49b3f70eaed811`, branch
  `work/v7-nvfp4-exl3`.
- Pinned SparkInfer: `63e2140e4a32a977faa777c172b86679344fdc6a`.
- Checkpoint: diffbot snapshot
  `28b7ab71ba2eb15569b08b91a8ea07df8eda8a75`.
- Both GPUs were idle, and a fresh CUDA allocation succeeded before export.
  Source was rsynced/docker-copied into `/wip/source`; the build guard verified
  exactly one `expert_format=nvfp4` occurrence in its `run.sh`.
- Experimental slot: `/wip/slots/v7-k2homog`, cloned from `v7q-a1`. Only the
  experimental Rust daemon and the new `exl3-k2/rtx-tp2` family were built.
  The full `wip.sh --role both` rebuild was deliberately avoided: it would
  unnecessarily regenerate protected mixed artifacts and unrelated Spark code.
- SHA-256 checks found all 566 protected release-staging files and all 96
  original WIP artifact files unchanged. No image build/push/tag occurred.
- Native-vs-native qualification loaded all 384 real layer-0 experts and
  compared the exact homogeneous DSO against the original mixed DSO. Capacity1
  used TP2 slice0; the other capacities used slice1152. Live rows included
  1,3,capacity−1,capacity where distinct, plus two changed-input/route graph
  replays, poisoned metadata/output, and stable-pointer checks. All **57
  numerical checks were bitwise equal**: 33 cross-native homogeneous/mixed
  comparisons and 24 same-arm graph/eager checks, all finite and nonzero;
  maximum absolute difference was zero. Small live counts do not claim all-expert route coverage.
- The service loaded all 40 routed-expert layers as TP2, with no Spark workers,
  and returned `42` to the arithmetic smoke request. Process maps verified the
  homogeneous arm actually loaded only `exl3-k2` modules, not dead-tier mixed.

## Measurement protocol

Two RTX PRO 6000 Blackwell Workstation Edition cards, 188 SM each, 400 W per
card; max memory setting 14,001 MHz. Candidate telemetry covered the complete
phase (205 samples per GPU), recorded a peak memory clock of 13,365 MHz, and
reported no thermal slowdown. Control telemetry likewise survived the phase
(193 samples per GPU), with the same cap, peak memory clock and no thermal
slowdown. Driver595 reports application-clock queries as
deprecated; that response is preserved rather than treated as a numeric
measurement.

Serving: two RTX, zero Spark, dSpark, prefill batch2048, concurrency16,
prefix-cache entries20, host cache auto, max context1,048,576 and max output
393,216; no TP2 attention/query/output/draft overrides. Native library bytes
are the original WIP bytes. The control uses the same experimental daemon
with the family variable zero, not a separately rebuilt mixed binary.
Both startups reserved the same KV/source-page pools and placed all 40 expert
layers locally. Homogeneous expert workspace was 1,290,910,212 bytes per GPU
per lane versus 1,152,498,180 for mixed (+138,412,032 bytes); single-tier is
not inherently smaller overall because the scratch alias schemes differ.

The requested commands were run with the checkpoint's tokenizer, label
`v7-dual-exl3-k2homog`, repeats3, decode seed79001 and counting enabled.
Prefill used `/home/tj/.cache/ds41rt-v7-bench/release-context-source.md`,
base0, repeats3 and one excluded warmup per suffix. All 30 homogeneous decode
completion checks and all 24 prefill requests passed. This is not a tool-call
quality qualification.

Weighted decode is the median of repeat-level weighted token/time ratios.
Code/counting are medians of three observed decode rates. Best prefill below
is the best of six base0 cell medians, **not** a newly measured full matrix.
A historical comparison uses different nonce seeds and has known completion
failures; it cannot by itself isolate the kernel effect.

## Requested comparison with shipped figures

Tokens/s; change is `(homogeneous / shipped − 1) × 100`.

| Measurement | Shipped mixed | Homogeneous K2 | Change |
|---|---:|---:|---:|
| Weighted decode | 145.10 | 149.95 | +3.34% |
| C1 code decode | 222.06 | 204.75 | −7.80% |
| Counting decode | 337.35 | 336.73 | −0.18% |
| Best base0 prefill | 5,702 | 5,336.26 | −6.41% |

Homogeneous repeat-level weighted rates: 152.45 / 144.00 / 149.95.
Code: 201.08 / 204.75 / 226.17. Counting: 305.92 / 336.73 / 339.29.

| Base0 suffix | Homogeneous median tok/s |
|---|---:|
| 1,024 | 3,054.74 |
| 2,048 | 4,023.36 |
| 4,096 | 5,261.02 |
| 8,192 | 5,336.26 |
| 16,384 | 5,320.75 |
| 32,768 | 5,269.38 |

## Same-seed current mixed control and decision

The historical weighted increase was not reproduced as a kernel benefit.
A second, freshly launched arm used the **same daemon/native library, same
seed79001 requests, same topology/cache settings and original mixed artifacts**.
All 30 completion checks passed, and all 30 outputs (including reasoning and
completion token counts) were identical between arms. Process maps confirmed
only `exl3-k23` in the control. Both base0 rows completed all 24 requests.

| Measurement | Current mixed control | Homogeneous K2 | Homogeneous change |
|---|---:|---:|---:|
| Weighted decode | 155.00 | 149.95 | **−3.26%** |
| C1 code decode | 206.80 | 204.75 | −1.00% |
| Counting decode | 323.65 | 336.73 | +4.04% |
| Best base0 prefill | 5,489.99 | 5,336.26 | **−2.80%** |

Control weighted repeats: 161.29 / 151.15 / 155.00. Homogeneous was slower
in every corresponding weighted repeat (−5.48%, −4.73%, −3.26%). Control code: 202.56 / 206.80 /
226.49; counting: 319.48 / 324.34 / 323.65. Control base0 medians for
1K/2K/4K/8K/16K/32K were 3,801.46 / 4,275.16 / 5,408.64 / 5,449.55 /
5,489.99 / 5,427.23 tok/s; homogeneous was slower at every suffix.

**Release decision: do not ship or select the candidate.** A real homogeneous
K2 export is reachable and numerically qualified, but this implementation is
not an overall serving improvement: weighted decode and every prefill cell
lose to the same-seed mixed control. Counting alone improves in this pair and
is approximately unchanged against the historical number. This does not prove
that every possible homogeneous schedule is slower, or quantify an isolated
GPU-kernel latency penalty. The experiment is sequential homogeneous→mixed,
three samples per cell, not a randomized/interleaved confidence study; clocks,
adaptive drafting and other end-to-end effects can still contribute. It is
sufficient evidence against replacing the current family on the premise that
removing an unused tier must be faster. No further tuning is required for the
release decision.

The old “empty tier is near-free” assertion should not be reinstated as a
mathematical claim: the measured answer is that the tested mixed serving path
wins the principal workloads. No changes to k23/k34 or published performance
figures are proposed. Temporary runtime/export code was archived outside the
repository, and the committed change is documentation only.

## Evidence and reproduction

Local evidence root: `~/.cache/ds41rt-v7-package/k2homog/`; raw campaigns are
in sibling `performance/`. This directory retains the separate exporter,
qualifier, `runtime.patch`, source/binary hashes, native qualification JSON,
family manifest, launch/build/benchmark scripts and complete candidate slot.
The exact executed qualifier is archived as
`qualify_v41_exl3_homogeneous_aot.executed.py`, SHA-256
`5d00351b5f16443aecdbdfcd0f0ecfcca2632771af54a5cf7e7e7cdb5ee2ad21`,
matching all six qualification records. The unsuffixed archived qualifier
(`789ef12b…`, also listed in `source-identity.json`) is a later variant adding
input-hash/routed-expert reporting; it was not the executed qualification
source. The executed bytes were recovered by reversing those three additions
and checking the complete recorded hash, not by substituting the later source.
The temporary implementation was removed from the release worktree; accepting
it later would require review and restoring the adapter/admission patch plus
explicit build/launcher integration. It is not silently shipped.

- `performance/dual-exl3-k2homog.json`: SHA-256
  `d6ebcf2eef598d4b6980c3ff23f887cf57c514d60352c7051ac2d2c1e50d0ad4`.
- `performance/dual-exl3-k2homog-prefill-base0.json`: SHA-256
  `0c82fa4fd06be29b57f67abcde5e40be946023883963b7529a5b0627cce55595`.
- `performance/dual-exl3-k23-control.json`: SHA-256
  `d070a5518f3c34d6d2c9197f0ceac3a2692e46347291e723621699e4ba6e99ff`.
- `performance/dual-exl3-k23-control-prefill-base0.json`: SHA-256
  `5035f5fb59516df2f940251a144d6a07724c3f1d01a0df282232c703b2717b24`.
- Historical raw SHA-256 values remain those in the
  [shipped EXL3 performance report](release-v7-exl3-k2-performance.md).

No compact/one-GPU profile was attempted, and no published table or image was
repointed at the candidate. Both experimental serving processes were stopped
by their PIDs inside the container; both GPUs were idle at completion. The
pre-existing untracked `.ds41rt-release-expert/` staging directory was preserved
byte-for-byte and locally excluded via `.git/info/exclude`, not deleted or
added to the documentation commit.
