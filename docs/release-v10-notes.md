# DS41RT v10 release notes

**STATUS: PREPARED - RUNTIME DEFAULT PROMOTED; REGISTRY PUBLICATION PENDING.** The
v10 images are built and locally verified from the frozen source commit below, and
`ds41rt.config` now names the `:v10` pair. **Nothing has been pushed.** No v10
registry digest exists yet, every registry value in this file is an explicit
placeholder, and no v10 performance number is claimed. Publication order, evidence
and rollback live in [release-v10-checklist.md](release-v10-checklist.md); the
publication summary and the real digests replace the placeholders here at that
point.

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

### Registry publication: PENDING (placeholders, not values)

No push, tag or publish has been performed. The coordinator's local repo digest is
its locally built manifest digest, and **no registry digest exists**.

- coordinator `ghcr.io/tpurtell/ds41rt-coordinator` index:
  `<PENDING: coordinator OCI index digest at v10 push>`
- spark-expert `ghcr.io/tpurtell/ds41rt-spark-expert` manifest:
  `<PENDING: spark manifest digest at v10 push>`
- pre-push `latest` rollback baseline: `<PENDING: latest coordinator index digest>`
  and `<PENDING: latest spark manifest digest>`
- anonymous fresh-pull verification: **NOT RUN**.

`v10` and `latest` are expected to resolve to the same digest per role once pushed
(both tags are published from the same local image). Do not transcribe a local
image id or a config digest into the registry fields above.

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
  the report's own status banner governs.
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

- **Registry publication pending.** Until the v10 push and anonymous verification
  complete, the runtime default names a tag that no registry serves. Do not treat
  this file as a publication record.
- **The v10 TP3 arms are informational.** The two profile files are packaging
  support only; each report's own status banner governs whether any result may be
  used, and the compact EXL3 TP3 arm may still be pending.
- **No v10 performance campaign.** The headline numbers in `README.md` remain the
  historical v6/v7/v8/v9 campaigns; v10 does not re-campaign the official
  default.

## Provenance requirements for every number

Record source commit, model revision, input shapes, concurrency, KV encoding,
sampling, acceptance, warmup, every timed sample, memory, graph setup, image
identity and the exact launch configuration. Component timing alone does not
establish serving throughput, and an image measurement is not a checkpoint
measurement.
