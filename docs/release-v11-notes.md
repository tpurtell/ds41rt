# DS41RT v11 release notes

**STATUS: DRAFT - PREPARED, NOT BUILT, NOT PUBLISHED.** Nothing in this file is a
measurement or a registry digest yet: every field below is an explicit
`PENDING` placeholder that is filled only from the named evidence file after the
corresponding step runs. The v11 image pair does not exist in GHCR, the runtime
default still names the published `:v10` pair, and `./build.sh --dry-run` still
reports `v10`.

Status lifecycle, so this file is never both unfilled and required to be final:

- **DRAFT** (now) - the template; facts still `PENDING`.
- **RELEASE-READY** - every non-registry fact is filled from real evidence
  (source revision, image ids, build verification, benchmark results) while the
  registry digests and the publication record remain `PENDING`. This state is
  the pre-publication gate.
- **PUBLISHED** - after the push and the anonymous verification, the registry
  digests and the publication record are filled from the registry's own
  responses. Only this state is the shippable release record.

**This template must not ship as the release record.** No `PENDING` placeholder
may remain when the status line moves to RELEASE-READY, except the registry and
publication fields that can only exist after the push; those are filled in the
same change that moves the status to PUBLISHED. [release-v11-checklist.md](release-v11-checklist.md)
§6 carries the gate.

Prepared from [the v10 notes](release-v10-notes.md); the ordered gate sheet is
[release-v11-checklist.md](release-v11-checklist.md).

## What v11 changes

- **Native non-greedy sampling.** The production native target path accepts
  `temperature > 0` and selects through the full-vocabulary host sampler
  (order-mask, temperature, `min_p`, `top_k`, `top_p`). Semantics to record here
  from the engine lead, and to confirm against the frozen source before this line
  is final: `temperature < 1e-5` samples greedily (the vLLM epsilon);
  `top_k` of 0 or -1 disables the filter and `k >= vocab` is a no-op; the seed is
  a signed i64 cast to u64, so `-1` is deterministic, unlike vLLM's `-1`
  (unseeded); and the no-temperature default remains greedy. dSpark stochastic
  selection is an exact sample-match given independent target draws, not a p/q
  rejection; the adaptive gate is unchanged. Seed reproducibility is conditional
  on the same logits and execution, not universal batch bitwise identity. Exact
  qualification and measured effect are the engine lead's and test agent's
  records; **PENDING** until the runtime change is frozen and independently
  reviewed.
- **Measured native deployment performance.** The v11 campaign measures the
  released pair on the real one-RTX-plus-four-Spark deployment: the headline
  nine-category decode table plus counting, five strict profiles, three repeats.
  No retained-turn and no concurrency-matrix campaign is required for v11.
  Numbers, raw package and limitations are **PENDING** in
  [release-v11-performance.md](release-v11-performance.md).
- **Launch selection before promotion.** Until the promotion change lands the
  runtime default names `:v10`, so the v11 candidate is launched with
  `--config ds41rt.build-v11.config`; `run.sh` then validates the v11 image,
  source, dependency, model, host and device identity before starting. The
  build-to-benchmark handoff and its launch overrides are specified in
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
| Coordinator | `ghcr.io/tpurtell/ds41rt-coordinator:v11` | amd64 | PENDING - §Registry publication |
| Spark expert (all four) | `ghcr.io/tpurtell/ds41rt-spark-expert:v11` | arm64 | PENDING - §Registry publication |

### Commit and identity separation

These are deliberately different commits; no row implies byte identity with
another. The tag annotation names the image build source, and the checkout at
that tag is **not** claimed to be byte-identical to the image build source.

| Role | Commit / value | Meaning |
| --- | --- | --- |
| Engine (sampling runtime) | `3f8ea80d8e728cf04832dd520c22a9bbf71f8827` | the reviewed runtime change |
| Image build source | `9d9b3e0ce8bd85f6b8eced515d0ab00be015f2d0` | exports `org.opencontainers.image.revision` |
| Host tools (qualifier/validator) | PENDING - separate evidence-tools commit | committed before the live host run |
| Release docs / tag commit | PENDING - final reviewed documentation commit | `main`/`release/v11` point here |
| Image identity | local ids below + version `v11` | the measured artifacts |

- Image build source commit **`9d9b3e0ce8bd85f6b8eced515d0ab00be015f2d0`**
  (bare, no `-dirty-`), which carries the engine change
  `3f8ea80d8e728cf04832dd520c22a9bbf71f8827`; recorded from the build log header
  and asserted by `verify-release-artifacts.sh` on both roles.
- Version `v11`; role labels `coordinator` / `expert`; CUDA arch `120` / `121`;
  Spark expert roles `tp2;tp3;tp6` (universal); SparkInfer
  `4b0954148523b5a2e93813f963d483ffd350b9c9` and XGrammar
  `557becfb64c503ae9c04344b0047661f43f44320` at the frozen commit.
- **Built local image ids (not registry digests):** coordinator
  `sha256:54c70eae326b7793bb2c6e34466179286b0861b991fd97ade0b99d69e79e4c41`;
  Spark expert on all four required hosts
  `sha256:b38647281b18f0a64c46b18417b74bf2027020583717fdceaba7a686be63369e`.
- **Shipped native libraries:** coordinator
  `11660155299d23b8fa14d68a74a34ae66ced497df442f84371626ce0a3aed701`; Spark
  expert `77c54a5a3c9b66ea43f362eac66dfde10e8de10694e41909bc6930fb598cdc35`;
  both equal the value in their dist `V41_EXPERT_TP_AOT.json`.
- Build result: `runs/v11-release/build/v11-build.rc` = `0`
  (05:23:43Z, source `9d9b3e0`); artifact verification
  `10-verify-summary.txt` = **ALL CHECKS PASS** over the four required hosts and
  `dist/SHA256SUMS` (`10-dist-sha256sums.txt` `sha256sums_rc=0`), with EXL3
  package and SparkInfer provenance steps all `rc=0`.
- Build evidence root: `runs/v11-release/build/` (repo-ignored), with the build
  log, `v11-build.rc`, the verification artifacts named in
  [release-v11-checklist.md](release-v11-checklist.md) §2, and the
  `SHA256SUMS` manifest written at archive time.

### Registry publication

**PENDING.** `./push-containers.sh --config ds41rt.build-v11.config v11`
publishes the `v11` and `latest` tags for the two fixed repositories. After the
push, `scripts/release-digests.sh capture` records each registry digest from the
registry's own response and `scripts/release-digests.sh verify` re-checks it by
an anonymous fresh pull on the matching architecture. Record here, never from a
local image id or a config digest:

- coordinator `ghcr.io/tpurtell/ds41rt-coordinator:v11` (OCI index, linux/amd64):
  `PENDING` - anonymous pull verified on `raptor`.
- spark-expert `ghcr.io/tpurtell/ds41rt-spark-expert:v11` (single manifest,
  linux/arm64): `PENDING` - anonymous pull verified on an arm64 worker.

Each role is verified on a host of its own architecture: an amd64 host cannot
pull the arm64 Spark manifest, and the coordinator's amd64 blob set is not pulled
onto the workers (the same constraint recorded for v9 and v10).

The pre-push rollback baseline is captured before the push. At capture time
`latest` is expected to be the v10 pair and `v11` expected to return 404 in both
repositories, so the push creates the tag rather than moving it:

- pre-push `latest` rollback baseline: `PENDING` (`pre-push-latest.env`); must
  equal the published v10 pair in [release-v10-notes.md](release-v10-notes.md).
- `:v11` absence record: `PENDING` (`v11-absence.txt`).
- GHCR visibility: the v10 packages are public and were re-verified anonymously
  during v11 preparation; re-verify at publication instead of assuming. Only if a
  package has gone private is a manual repository-owner action required.

## Runtime default and rollback

- Until the promotion change lands, `ds41rt.config` and the
  `examples/configs/*` files name `:v10`, and `./build.sh --dry-run` reports
  `v10`. `ds41rt.build-v11.config` is the explicit v11 **build** target and is
  intentionally not yet identical to `ds41rt.config`; the only intended
  difference is the release image pair.
- After publication, promotion moves `ds41rt.config`, `examples/configs/*` and
  their README pair lines to `:v11`; at that point `ds41rt.build-v11.config` is
  identical to `ds41rt.config` and its header comment is updated to say so.
- The v10 images remain published as
  `ghcr.io/tpurtell/ds41rt-coordinator:v10` and
  `ghcr.io/tpurtell/ds41rt-spark-expert:v10`
  ([v10 notes and digests](release-v10-notes.md)). `latest` is the v10 pair until
  the v11 push moves it.
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
