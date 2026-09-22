# DS41RT v11 release notes

**STATUS: PUBLISHED.** The v11 pair is built from image source
`fb5115466a8c70c063280e25957577f284e903e3`, measured, re-qualified and published.
The registry digests below are read from the registry's own responses and
re-checked by anonymous pull on the matching architecture. The runtime default
names `:v11` and `./build.sh --dry-run` reports `v11`.

Status lifecycle, so this file is never both unfilled and required to be final:

- **DRAFT** - the unfilled template.
- **RELEASE-READY** - every non-registry fact filled from real evidence, with the
  registry digests and publication record still open. The pre-publication gate.
- **PUBLISHED** (now) - after the push and anonymous verification, the registry
  digests and the publication record are filled from the registry's own
  responses. This is the shippable release record.

This document is the PUBLISHED release record: every identity, measurement and
registry field is filled from real evidence, and the documented placeholder-count
gate over this file reports zero. [release-v11-checklist.md](release-v11-checklist.md)
§6 carries the publication gate.

Prepared from [the v10 notes](release-v10-notes.md); the ordered gate sheet is
[release-v11-checklist.md](release-v11-checklist.md).

## What v11 changes

- **Native non-greedy sampling.** The production native target path accepts
  `temperature > 0` and selects through the full-vocabulary host sampler
  (order-mask, temperature, `min_p`, `top_k`, `top_p`). Semantics to record here
  from the frozen source (`fb51154`): `temperature < 1e-5` samples greedily (the
  vLLM epsilon);
  `top_k` of 0 or -1 disables the filter and `k >= vocab` is a no-op; the seed is
  a signed i64 cast to u64, so `-1` is deterministic, unlike vLLM's `-1`
  (unseeded); and the no-temperature default remains greedy. dSpark stochastic
  selection is exact speculative-sampling rejection correction with the draft as
  a point mass (`q = delta`), exact for any draft and temperature - not the
  full-draft-distribution correction that was considered and rejected; the
  adaptive gate is unchanged. Seed reproducibility is conditional
  on the same logits and execution, not universal batch bitwise identity. Exact
  qualification and measured effect are recorded in
  [release-v11-performance.md](release-v11-performance.md); the runtime change is
  frozen at `fb51154` and independently reviewed.
- **Measured native deployment performance.** The v11 campaign measures the
  released pair on the real one-RTX-plus-four-Spark deployment: the headline
  nine-category decode table plus counting, five strict profiles, three repeats.
  No retained-turn and no concurrency-matrix campaign is required for v11. Both
  datasets are recorded in [release-v11-performance.md](release-v11-performance.md)
  with distinct provenance, and neither is published here as a single headline:

  | Profile | Baseline `9d9b3e0` | Final `fb51154` |
  | --- | ---: | ---: |
  | greedy | 94.41 | 95.46 |
  | temp0.2 + top_p0.95 | 47.35 | 80.58 |
  | temp0.7 + top_p0.9 | 47.52 | 83.29 |
  | temp0.7 + min_p0.05 | 88.26 | 88.99 |
  | temp0.7 + top_k40 | 47.14 | 87.74 |

  Both campaigns pass the campaign validator with zero failures, 15/15 raw
  reports and an identity deep-equal; evidence hashes are listed on the
  performance page. The final pair's optimization (`fb51154`) replaces the
  ordered sampler path's structured sort with bit-identical heap/radix
  selection, verified against a verbatim pre-change oracle.
  **Presentation rule (audit-mandated):** on the final dataset only **greedy** is
  clearly first (95.46, margin +6.47 to +14.88 over the others, whose spreads are
  0.97-5.73); the **four stochastic modes form one tight cluster and must not be
  ordered internally**, because min_p 88.99 versus top_k40 87.74 is +1.24 against
  a 2.00 spread, and top_p0.9 83.29 versus top_p0.95 80.58 is +2.70 against a
  5.73 spread - both within noise.
  **Workload identity:** weighted token counts are exactly equal between baseline
  and final for every profile (6608 / 7013 / 6067 / 6870 / 7166) and per-case
  median completion tokens are identical across every profile and case, so the
  entire gain is reduced elapsed time for the same output (top_k40 128.44 s ->
  69.37 s, top_p0.9 144.33 -> 82.47, top_p0.95 151.70 -> 88.11, with greedy and
  min_p at 0.989x / 0.988x) - not a workload artifact.
  **Re-qualification:** the final pair passed live qualification with **91 checks
  / 69 observations / 0 failures**, all replay checks exact and an empty
  check-outcome delta versus baseline.
  A baseline-image `--no-dspark` control found no profile faster without dSpark,
  so speculation is a net win at every vector and the adaptive gate is
  unchanged. Full-draft-distribution rejection correction was considered and
  rejected for this release (no gain at the benchmark temperatures; kept as a
  future high-temperature item), while our verifier applies the full filter
  chain to the verified target distribution, so `min_p` is honored under
  speculation. Bitwise speculative/non-speculative parity is explicitly not
  claimed, matching the upstream non-guarantee.
- **Launch selection.** The runtime default is promoted to `:v11`, so `run.sh`
  and `./build.sh` select the v11 pair directly; `ds41rt.build-v11.config` stays
  an explicit, identical v11 release **build** target. `run.sh` validates the
  image, source, dependency, model, host and device identity before starting. The
  build-to-benchmark handoff is specified in
  [release-v11-checklist.md](release-v11-checklist.md) §3. The frozen config's
  SHA-256 is the launch fingerprint `run.sh` records in-container as
  `DS41RT_RELEASE_CONFIG_SHA256`.
- **Does not switch the checkpoint.** v11 serves the same native
  `deepseek-ai/DeepSeek-V4.1-Flash` revision as v10 unless the frozen source says
  otherwise; the release is a runtime change plus a new image pair, not a new
  model or quantization.

## Images and source identity

| Artifact | Tag | Arch | Registry digest |
| --- | --- | --- | --- |
| Coordinator | `ghcr.io/tpurtell/ds41rt-coordinator:v11` | amd64 | `sha256:e0e5d631a54e84f36cd1cd99e2a7cf4e2092c15919c8c79403007ba0852d0d89` |
| Spark expert (all four) | `ghcr.io/tpurtell/ds41rt-spark-expert:v11` | arm64 | `sha256:fc50ac1cf7f309727d406962168fbe278807848566ae84abde0709b9ca5ba27a` |

Both column values are **registry manifest digests**, read from the registry's
own response and re-checked by an anonymous fresh pull on a host of the matching
architecture; they are not local image ids (see the identity rows below).

### Commit and identity separation

These are deliberately different commits; no row implies byte identity with
another. The tag annotation names the image build source, and the checkout at
that tag is **not** claimed to be byte-identical to the image build source.

| Role | Commit / value | Meaning |
| --- | --- | --- |
| Engine (sampling runtime) | `3f8ea80d8e728cf04832dd520c22a9bbf71f8827` | the reviewed runtime change |
| Baseline image build source | `9d9b3e0ce8bd85f6b8eced515d0ab00be015f2d0` | the pre-optimization candidate |
| Final image build source | `fb5115466a8c70c063280e25957577f284e903e3` | exports `org.opencontainers.image.revision` |
| Host tools (qualifier/validator) | `6f58206f70934c40515598e697b0fcd9813a548d` | committed before the live host run |
| Bench host | `ec7cfb5679696aa44d81486cf36e26d25595cfbc` | produced the baseline campaign |
| Qualification host | `8a7da7da9bd30d4ec797926ee4c5c8c6cef734d7` | produced the live qualification |
| Release docs / tag commit | the `v11` tag commit (this documentation commit) | `main`/`release/v11` point here |
| Image identity | local ids below + version `v11` | the measured artifacts |

- Final image build source commit
  **`fb5115466a8c70c063280e25957577f284e903e3`** (bare, no `-dirty-`), which
  carries the engine change `3f8ea80d8e728cf04832dd520c22a9bbf71f8827` and the
  sampler optimization; recorded from the build log header and asserted by
  `verify-release-artifacts.sh` on both roles. The baseline candidate was
  `9d9b3e0ce8bd85f6b8eced515d0ab00be015f2d0`.
- Version `v11`; role labels `coordinator` / `expert`; CUDA arch `120` / `121`;
  Spark expert roles `tp2;tp3;tp6` (universal); SparkInfer
  `4b0954148523b5a2e93813f963d483ffd350b9c9` and XGrammar
  `557becfb64c503ae9c04344b0047661f43f44320` at the frozen commit.
- **Built local image ids (not registry digests):** coordinator
  `sha256:e0e5d631a54e84f36cd1cd99e2a7cf4e2092c15919c8c79403007ba0852d0d89`;
  Spark expert on all four required hosts
  `sha256:f1233987c3b9ba13d468d621c96945052db15e7ede9a8feaf101f506aee85516`.
  Registry manifest digests are listed separately above; for the coordinator the
  local id and the manifest digest coincide, for the Spark expert they differ, as
  expected between an image config id and a manifest digest.
- **Shipped native libraries:** coordinator
  `a448f52c14246e6ded5f63512678215d8fd7869e925d9128c6dfa8459bf3da87`; Spark
  expert `2b4be85568109fcba89fc584ff9b3a6b8d845c7e69ad196dff5d115aafd8d148`;
  both equal the value in their dist `V41_EXPERT_TP_AOT.json`.
- Build result: `runs/v11-release/build-final/v11-build.rc` = `0`
  (06:48:33Z, source `fb51154`, the promoted pair; the baseline build kept its own
  `runs/v11-release/build/` evidence); artifact verification
  `10-verify-summary.txt` = **ALL CHECKS PASS** over the four required hosts and
  `dist/SHA256SUMS` (`10-dist-sha256sums.txt` `sha256sums_rc=0`), with EXL3
  package and SparkInfer provenance steps all `rc=0`.
- Build evidence root: `runs/v11-release/build-final/` (repo-ignored), with the build
  log, `v11-build.rc`, the verification artifacts named in
  [release-v11-checklist.md](release-v11-checklist.md) §2, and the
  `SHA256SUMS` manifest written at archive time.

### Registry publication

**Published.** `./push-containers.sh --config ds41rt.build-v11.config v11`
publishes the `v11` and `latest` tags for the two fixed repositories. After the
push, `scripts/release-digests.sh capture` records each registry digest from the
registry's own response and `scripts/release-digests.sh verify` re-checks it by
an anonymous fresh pull on the matching architecture. Record here, never from a
local image id or a config digest:

- coordinator `ghcr.io/tpurtell/ds41rt-coordinator:v11` (OCI index, linux/amd64):
  `sha256:e0e5d631a54e84f36cd1cd99e2a7cf4e2092c15919c8c79403007ba0852d0d89` -
  anonymous pull verified on `raptor`.
- spark-expert `ghcr.io/tpurtell/ds41rt-spark-expert:v11` (single manifest,
  linux/arm64):
  `sha256:fc50ac1cf7f309727d406962168fbe278807848566ae84abde0709b9ca5ba27a` -
  anonymous pull verified on `kiwi`.
- `latest` resolves to the same two manifest digests for the same roles
  (`post-push-latest.env`), captured separately after the push.

Each role is verified on a host of its own architecture: an amd64 host cannot
pull the arm64 Spark manifest, and the coordinator's amd64 blob set is not pulled
onto the workers (the same constraint recorded for v9 and v10).

The pre-push rollback baseline is captured before the push. At capture time
`latest` is expected to be the v10 pair and `v11` expected to return 404 in both
repositories, so the push creates the tag rather than moving it:

- pre-push `latest` rollback baseline:
  `sha256:2236d94317eb393cd78940efb117bcca14ae06e6d1113c6aeb188b0b22424689`
  (coordinator) and
  `sha256:98ddf9cd83626d92297169b04561b722d136de27ac8213a1a9f5702ebe40ec30`
  (spark) in `pre-push-latest-recheck.env`; equal to the published v10 pair in
  [release-v10-notes.md](release-v10-notes.md).
- `:v11` absence record: `v11-absence-recheck.txt` (both repositories returned
  absent before the push).
- GHCR visibility: the v10 packages are public and were re-verified anonymously
  during v11 preparation; re-verify at publication instead of assuming. Only if a
  package has gone private is a manual repository-owner action required.

## GitHub release and evidence package

The release is published at
[github.com/tpurtell/ds41rt/releases/tag/v11](https://github.com/tpurtell/ds41rt/releases/tag/v11)
and is neither a draft nor a prerelease. The annotated tag `v11` points at
`b6eceba7db7f46380ac7b74e093fb9ba748b0e73`; its annotation names the immutable
image build source `fb5115466a8c70c063280e25957577f284e903e3` and states that the
tag's checkout is not claimed byte-identical to it.

| Asset | Size (bytes) | SHA-256 of the asset |
| --- | ---: | --- |
| `v11-evidence.tar.gz` | 1,388,281 | `4e7058c41bec7b433128526e7259edc14b1cfed52a43b284f1edff34672c6629` |
| `SHA256SUMS` | 18,323 | `2ff225d2273b87f633b23f6b496b764e4950ce9f9ff864520ef04675b5c54670` |
| `v11-release-assets.sha256` | 402 | `5baa701810c10e88e6c2cab2e742a88eb5a1f2830248013945d577b9d1aba8da` |

**Scope of `SHA256SUMS` - do not conflate it with a manifest of the release
assets.** That asset is the *payload* manifest: it lists the **148 files inside
`v11-evidence.tar.gz`** and verifies them (`sha256sum -c` -> 148 OK, 0 failed). It
is not a manifest of the release assets, and its own SHA-256 (`2ff225d2...`) is a
different value from the tarball's (`4e7058c4...`). The additive
`v11-release-assets.sha256` asset lists those two asset hashes, so the asset-level
record is explicit and the two kinds of value are not merged.

Package contents: the 15 final raw reports, the baseline attempt-02 package, the
no-dSpark control package, the dSpark debug traces, the re-qualification
artifacts, the validator and aggregator outputs, the build and source-gate logs,
and the publication records. The package was secret-scanned before upload.

**Post-tag accuracy addendum.** This record and the README image-id labelling were
added after the `v11` tag was created, so they are not part of the tagged tree;
the tag remains at `b6eceba`. No measured number, digest or image reference was
changed by this addendum.

## Runtime default and rollback

- `ds41rt.config`, the `examples/configs/*` files and their README pair lines now
  name `:v11`, and `./build.sh --dry-run` reports `v11`.
  `ds41rt.build-v11.config` is therefore identical to `ds41rt.config`.
- The v10 images remain published as
  `ghcr.io/tpurtell/ds41rt-coordinator:v10` and
  `ghcr.io/tpurtell/ds41rt-spark-expert:v10`
  ([v10 notes and digests](release-v10-notes.md)). `latest` moved to the v11 pair
  at the v11 push.
- Promotion rollback is the promotion change reverted plus the `latest`
  re-point recorded in [release-v11-checklist.md](release-v11-checklist.md) §6.

## Known limitations

- **Non-greedy sampling scope is bounded by its own evidence.** The sampler
  qualification compares adjusted-logit bits, sampled tokens, graph replays and
  Philox sequencing on a fixture; that is not broad output-quality
  qualification. Any inherited Unicode objective failure and the exact coverage
  travel with the sampler record, not with this file.
- **The v11 performance campaign is the scope the coordinator set** (headline
  nine-category plus counting, five strict profiles, three repeats). It is not a
  retained-turn or concurrency-matrix qualification, and component timing still
  does not establish serving throughput.
- **No claim is made here about topologies, quantizations or profiles that the
  campaign did not measure.** The v10 TP3 profiles and reports keep their own
  status banners and remain informational; packaging is not qualification.

## Provenance requirements for every number

Record source commit, model revision, input shapes, concurrency, KV encoding,
sampling (including the non-greedy path), acceptance, warmup, every timed sample,
memory, graph setup, image identity (registry digest after publication) and the
exact launch configuration. Component timing alone does not establish serving
throughput, and an image measurement is not a checkpoint measurement.
