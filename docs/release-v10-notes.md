# DS41RT v10 release notes

**STATUS: MEASURED TP3 REPORTS PUBLISHED; REGISTRY PUBLICATION IN PROGRESS.** The
v10 images are built from the frozen source commit below and `ds41rt.config` names
the `:v10` pair. The registry push and its anonymous fresh-pull verification are
recorded in the publication section as they complete; until those boxes are
checked, this file carries no registry digest for `v10`. The release carries
measured TP3 reports, not a performance qualification: the official native arm is
incomplete and informational (**309 / 359** performance records, **88 / 264**
tool-eval scenario-runs), the compact EXL3 TP3 arm is complete and strict-**PASS**
(**359 / 359** performance records, **264 / 264** tool-eval scenario-runs), and the
EXL3 report's remaining network and change-threshold cells stay explicitly
pending. Publication ordering and rollback live in
[release-v10-checklist.md](release-v10-checklist.md).

## What v10 changes

- **Promotes the runtime default to the v10 pair.** `ds41rt.config`, the five
  older native `examples/configs/*` files and the `README.md` pull/pair lines name
  `:v10`. The two v10 TP3 profiles (`tp3ep1-native.config`,
  `exl3-compact-tp3.config`) were already at `:v10` and are no longer exempt from
  published-pair equality.
- **Does not switch the checkpoint.** v10 serves the same native
  `deepseek-ai/DeepSeek-V4.1-Flash` revision; the release change is the image pair
  and the runtime default that names it, not a new model or quantization.
- **Keeps the universal Spark role set.** The Spark image advertises
  `io.ds41rt.v41.spark_tp_roles=tp2;tp3;tp6` on top of the default TP4 shard, so
  one pair serves every approved native topology and `./run.sh` selects the role
  from `SPARK_TP`.
- **Ships the v10 TP3 profiles** described below. Packaging is not qualification:
  neither profile carries a memory, correctness, performance or readiness claim.

## Images and source identity

| Artifact | Tag | Arch | Local image id |
| --- | --- | --- | --- |
| Coordinator | `ghcr.io/tpurtell/ds41rt-coordinator:v10` | amd64 | `sha256:2236d94317eb393cd78940efb117bcca14ae06e6d1113c6aeb188b0b22424689` |
| Spark expert (all four) | `ghcr.io/tpurtell/ds41rt-spark-expert:v10` | arm64 | `sha256:d1b668cd7e87079b5b57e381bdd45dfb4858646533611f18f22061f58ae18dec` |

- Image source commit **`3dd9a4ac2be9fd17ecf4cb8b7746efdc900d38f0`** (bare, no
  `-dirty-`), version `v10`, role labels `coordinator` / `expert`, CUDA arch `120`
  / `121`, SparkInfer `4b0954148523b5a2e93813f963d483ffd350b9c9`, XGrammar
  `557becfb64c503ae9c04344b0047661f43f44320`.
- The identical Spark image id was verified on `ostrich`, `dodo`, `emu` and
  `kiwi` after RDMA distribution; the coordinator image is local to `raptor`
  only.
- Dist artifacts (verified inside the running containers): coordinator native lib
  `23eba1fe4c1e8e0dd7f92bd441771d14abde4845d63d01c28150db52762ed8f7`, TP AOT
  manifest `e37b1232556337c4907d5ccb117ca5a2ab4225852b7d8c938e1c923070df03d3`;
  Spark native lib `37837f9598206791816acdb4a97a4bfc10eb6d064673640639065d4aa7e87214`,
  TP AOT manifest `f6b9df3901708e315e75f0abc93f8c93e0f9fc9e95f7020e34b68a731cea0090`.
  EXL3 k23/k34 packages verify for both roles (18 coordinator / 54 Spark variants
  each); the TP3 packages carry `tp3-rank{0,1,2}` at `intermediate=768` for every
  contract capacity.
- Build evidence: `runs/v10-release/build/` (repo-ignored), including
  `RUN-SUMMARY.md`, `10-label-assertions.txt`,
  `10-local-coordinator-image.txt`, `10-fleet-spark-images.txt`,
  `10-exl3-tp3-verification.txt` and `11-supplementary-verification.txt`.

### Registry publication

The pre-push rollback baseline was captured from the registry before the push
(`runs/v10-release/publication/pre-push-latest.env`, repo-ignored evidence). At
capture time `latest` was still the v9 pair and `v10` returned 404 in both
repositories, so the push below creates the tag rather than moving it.

- pre-push `latest` rollback baseline — coordinator index
  `sha256:786d1d6704e4cdaaf12ae59f5324bb1a43ce2238c9ec79ff7884fd3c83e8eb7f`,
  spark-expert manifest
  `sha256:f0c67407adb4228200c1fcb97e3f7210501db120d1e1ee11697f66cfb1ca4ea9`
  (identical to the published v9 pair in [release-v9-notes.md](release-v9-notes.md)).
- v10 registry digests and the anonymous fresh-pull result are added here at the
  §2.2/§2.3 capture and verification steps; neither exists yet, so **no v10
  registry digest is claimed by this revision**.

`v10` and `latest` resolve to the same digest per role once pushed (both tags are
published from the same local image). Do not transcribe the local image id below,
or a config digest, into a registry field.

### Image identity as measured

The measured campaigns identify the image pair by its **local** image identity at
source commit `3dd9a4ac2be9fd17ecf4cb8b7746efdc900d38f0`; this is build-artifact
identity, not a registry digest:

- coordinator `ghcr.io/tpurtell/ds41rt-coordinator:v10`
  `sha256:2236d94317eb393cd78940efb117bcca14ae06e6d1113c6aeb188b0b22424689`
  (`runs/v10-release/build/10-local-coordinator-image.txt`,
  `runs/v10-exl3-tp3/evidence/20260921T1400Z-raptor/identity/coordinator-image.json`);
- Spark expert `ghcr.io/tpurtell/ds41rt-spark-expert:v10`
  `sha256:d1b668cd7e87079b5b57e381bdd45dfb4858646533611f18f22061f58ae18dec`
  on all four workers, with empty `repo_digests` because nothing was pushed
  (`runs/v10-release/build/10-fleet-spark-images.txt`); as in v9 this local id is
  the image config digest while the registry reports the manifest digest, so the
  Spark id and the post-push registry digest are not expected to be equal;
- both revisions are `3dd9a4ac2be9fd17ecf4cb8b7746efdc900d38f0`, and the Spark
  role label is `io.ds41rt.v41.spark_tp_roles=tp2;tp3;tp6`
  (`runs/v10-release/build/10-label-assertions.txt`).

A container `Id` captured from `docker inspect` is **not** an image identity: the
EXL3 lane's post-launch capture records the running container
`e7626ab1d9749709a5a241d87eedcbe94714face353e87c136d37c195a7c70e5`, whose own
`Image` field is the coordinator image id above. The published EXL3 report carries
the image id, not the container id.

## v10 TP3 profiles

- **`examples/configs/tp3ep1-native.config`** - explicit native `TP3xEP1`: one
  unreplicated group of three ranks (`SPARK_COUNT=3`, `SPARK_TP=3`, `SPARK_EP=1`),
  `RTX_EXPERT_LAYERS=5` (5 RTX-local / 35 remote via the placement handoff) and a
  pinned `KV_POOL_SIZE=12GiB`. **Not release-qualified.** The native TP3 campaign
  is reported for information only; read the status banner in
  [release-v10-tp3-official-1x-3spark.md](release-v10-tp3-official-1x-3spark.md)
  rather than any summary here. No publication may rest on it.
- **`examples/configs/exl3-compact-tp3.config`** - implicit compact EXL3 TP3
  (`SPARK_COUNT=3` with no `SPARK_TP`/`SPARK_EP`), checkpoint
  `wrldsuksgo2mars/DeepSeek-V4.1-EXL3-K3.25-v1` (bit family `[3,4]`, launcher tag
  `k34`) on one RTX under the hard 32 GiB ceiling. Report:
  [release-v10-tp3-exl3-compact-1x-3spark.md](release-v10-tp3-exl3-compact-1x-3spark.md);
  its battery is complete (**359 / 359** performance records, **264 / 264**
  tool-eval scenario-runs) and the report's strict accounting gate is a **PASS**.
  Pass is not a release-readiness claim: the report's own status banner and
  limitations govern, and its network and change-threshold cells stay pending.
- The native KV-pool sizing and the tool-eval admission failure that motivated it
  are recorded in
  [v10-native-tp3ep1-tool-eval-oom.md](v10-native-tp3ep1-tool-eval-oom.md).
- Campaign-level accounting for both arms:
  [release-v10-tp3-campaign-status.md](release-v10-tp3-campaign-status.md).

## Runtime default and rollback

- `ds41rt.config` (and the five older native examples) name the promoted pair.
  `ds41rt.build-v10.config` is retained as the explicit release **build** target
  and is now identical to `ds41rt.config`, including the pair.
- `./build.sh --dry-run` and
  `./build.sh --config ds41rt.build-v10.config --dry-run` both report release tag
  `v10`.
- The v9 images remain published as `ghcr.io/tpurtell/ds41rt-coordinator:v9` and
  `ghcr.io/tpurtell/ds41rt-spark-expert:v9`
  ([v9 notes and digests](release-v9-notes.md)). The pre-push `latest` digest is
  the v9 pair until the v10 push moves `latest`.
- Promotion rollback is this change reverted plus the `latest` re-point recorded
  in [release-v10-checklist.md](release-v10-checklist.md) §3.

## Known limitations

- **The v10 TP3 reports are measured, not a qualification.** The official native
  arm is incomplete and informational (**309 / 359** performance records,
  **88 / 264** tool-eval scenario-runs, measured on the pre-`KV_POOL_SIZE`-pin
  profile) and the compact EXL3 TP3 arm is complete and strict-**PASS**
  (**359 / 359** performance records, **264 / 264** tool-eval scenario-runs). Each
  report's own status banner and limitations travel with its numbers; the EXL3
  report's unresolved network and change-threshold cells remain **PENDING**. The
  two v10 TP3 profile files remain packaging support only.
- **No v10 performance campaign for the promoted default.** The headline table in
  `README.md` keeps the historical v6/v7/v8/v9 columns unchanged; v10 does not
  re-campaign the official default, and the TP3 numbers are a separate topology
  under its own ceiling rather than a replacement for any headline column.

## Provenance requirements for every number

Record source commit, model revision, input shapes, concurrency, KV encoding,
sampling, acceptance, warmup, every timed sample, memory, graph setup, image
identity and the exact launch configuration. Component timing alone does not
establish serving throughput, and an image measurement is not a checkpoint
measurement.
