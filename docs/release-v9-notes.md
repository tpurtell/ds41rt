# DS41RT v9 release notes

**STATUS: RELEASE CANDIDATE. Images built and functionally validated; container
publication awaits the registry push.** No v9 container has been pushed yet, so
the pull-manifest digests are not known and are deliberately not stated. Model
weights are host-mounted and are never in the images.

## What v9 changes

- **Adds the pure unreplicated TP6EP1 Spark topology.** Six ranks form one group;
  each rank holds a disjoint `2304 / 6 = 384` column slice of every routed expert
  (`SPARK_COUNT=6`, `SPARK_TP=6`, `SPARK_EP=1`). This is tensor parallelism over
  the official intermediate, not expert parallelism and not a reuse of the
  TP2xEP3/TP3xEP2 replicated-group contract.
- **Keeps the default four-rank TP4EP1 path unchanged.** The release default
  remains `SPARK_COUNT=4`; TP4 needs no baked role label and is always exported.
- **Supports the approved explicit topologies** TP2EP2, TP3EP2, TP2EP3, TP4EP1 and
  TP6EP1. The v9 Spark role set is **tp2;tp3;tp6**, with the default TP4 path
  implicit.
- **Widens cleanup scope to every configured Spark host** (`stop.sh` /
  `release_select_stop_hosts`). Cleanup is a superset of the active ranks, so
  `stop.sh` reaches containers a previous six-rank run left on the fifth/sixth
  hosts even when the configuration selects four ranks. Launch and restart keep
  cleaning only the active ranks.
- **Names the fifth/sixth hosts in `ds41rt.config`** (`rhea`, `moa`). The default
  `SPARK_COUNT=4` selection does not launch them. The release and candidate
  launchers pass only `LANE_A`; the secondary `LANE_B` values are configuration
  annotations: on the **coordinator (`raptor`) both lanes share one physical port
  and HCA**, while **each Spark has two separate HCAs**. No dual-rail gain is
  claimed.

## Images and source identity

| Artifact | Tag | Arch | Local image id |
| --- | --- | --- | --- |
| Coordinator | `ghcr.io/tpurtell/ds41rt-coordinator:v9` | amd64 | `sha256:786d1d6704e4cdaaf12ae59f5324bb1a43ce2238c9ec79ff7884fd3c83e8eb7f` |
| Spark expert (all six) | `ghcr.io/tpurtell/ds41rt-spark-expert:v9` | arm64 | `sha256:9e8c8248b00366a493023687a42dea9c2544fe8eb7cc1a3c46656a91d9ea3ff9` |

- Engine revision **`5d0d209509bf1f26731bc588dfe8dc72d1373ad0`** (bare, no
  `-dirty-`), version `v9`, SparkInfer `4b0954148523b5a2e93813f963d483ffd350b9c9`,
  `io.ds41rt.v41.spark_tp_roles=tp2;tp3;tp6`.
- Dist binaries (verified inside the running containers): coordinator daemon
  `8c14fd85bb6eb532a196dc8ddd30e8de1ac7128a120e82ab9d75f0eedf416b35`, native lib
  `b14cac3ea246fe50369221d97aae984cafd72417ef536538813d7f653434b101`; Spark daemon
  `c176f2c8db03dfd36a179b28bf45e464ac2a41a1a57b34809eeb13914a32231d`, native lib
  `e1e71c51d1c74a6ef3ae38fb68e0c40a497ad28dfa14f54ddf3e901cb230520a`.
- **Registry publication: PUBLISHED and anonymous-verified (2026-09-21).** `v9` and
  `latest` resolve to the same digest per role:
  - coordinator `ghcr.io/tpurtell/ds41rt-coordinator` index
    `sha256:786d1d6704e4cdaaf12ae59f5324bb1a43ce2238c9ec79ff7884fd3c83e8eb7f`
    (linux/amd64 child `sha256:649e5c84b25d309457950d3e680b84cd2527434f156aecf8483076c4aa72807c`);
  - spark-expert `ghcr.io/tpurtell/ds41rt-spark-expert` manifest
    `sha256:f0c67407adb4228200c1fcb97e3f7210501db120d1e1ee11697f66cfb1ca4ea9`
    (arm64; local config id `sha256:9e8c8248…`).

  The coordinator's local `.Id` equals its index digest (buildkit stores the
  index+attestation locally); the Spark local `.Id` is its config digest while the
  registry reports the manifest digest — this is not a discrepancy. Both images were
  pulled anonymously with a fresh `DOCKER_CONFIG` and no host credentials (rc=0,
  matching digests). The GitHub release is a separate step and remains pending.

## Official quantization vs container distinction

- The **official quantization** is the checkpoint itself —
  `deepseek-ai/DeepSeek-V4.1-Flash` at
  `dba1be0a40aa45a94ad051997016db3960a90277`. Its published numbers are the
  historical official campaign; v9 does not re-campaign the official default for
  qualification purposes.
- The **container images** are the release artifacts built from the frozen source
  commit. They bundle binaries, Python tooling and AOT packages; they **do not
  bundle weights**. Weights are mounted read-only from the host Hugging Face cache
  and Engram tables are mapped from host storage.
- A container-image measurement therefore proves the shipped code, not the
  checkpoint license, and vice versa. Every number below names which of the two it
  belongs to.

## Final-image functional validation (PASS)

All four production geometries launched with the production `./run.sh` from the
clean build tree. The loaded container binaries equal the fresh dist artifacts
above, so no old or scratch binary was injected. The API and constrained smokes
return rc=0 on every geometry, and the strict-schema 400 rejection is genuinely
exercised (not forced): `json_schema=true`, `combined_response_tools=true`,
`strict_tool_stream=true`, `validation=true`.

| Geometry | Placement applied | Worker contract | api-smoke | constrained | Content |
| --- | --- | --- | --- | --- | --- |
| TP6, 1 RTX, 5/35 | RTX GPUs 1, RTX layers 5; Spark first layer 5, remote_layers=35 | world=6 role=7 intermediate=384 first_layer=5 layers=35 | rc=0 | rc=0 | 9 categories + counting, 3 repeats, 30/30 PASS (nonce 61001) |
| TP6, 2 RTX, 20/20 | RTX GPUs 2, RTX layers 20; Spark first layer 20, remote_layers=20 | world=6 role=7 intermediate=384 first_layer=20 layers=20 | rc=0 | rc=0 | 9 categories + counting, 1 each, 10/10 PASS |
| TP4 implicit default, 1 RTX | coordinator dispatch placement **not captured** | worker LOAD only: world=4 role=1 intermediate=640 first_layer=0 layers=40 | rc=0 | rc=0 | 10/10 PASS |
| TP4 implicit default, 2 RTX | RTX GPUs 2, RTX layers 20; Spark first layer 20 | world=4 role=1 intermediate=640 first_layer=20 layers=20 | rc=0 | rc=0 | 10/10 PASS |

**TP4 1 RTX placement.** The worker LOAD contract is `first_layer=0 layers=40`, but
a worker's load count does not say which layers the coordinator dispatches to it.
`start-tp4-1x.log` prints no placement line and the coordinator container logs were
not captured, so the dispatch split for this geometry is **unknown** and is
recorded as unknown rather than inferred. In particular it is **not** an
"all-40-remote" claim: the historical baseline also loads 40 while dispatching
5 local / 35 remote.

**Operational prerequisite.** GB10 unified memory counts clean page cache against
device-free bytes. The established `drop-page-cache` step on `kiwi` and `rhea`
restored ~113–117 GiB free, after which every geometry launched. This is recorded
as an operational prerequisite, not as tuning; no dial, config or threshold was
changed.

## Performance (warm candidate)

These are the v9 TP6 measurements. They were taken on the warm candidate binaries
(daemons `a41a76ba`/`884f4454`, libs `e9736c3e`/`c98b392d`), which are **distinct**
from the final clean-build dist binaries above. They are validated candidate
measurements; they are **not** final-image performance numbers. This headline table
is deliberately warm-candidate only. A final-image 1 RTX headline-shaped functional
run (30 samples) was performed and is preserved, but it is **not** used as a
performance replacement and is **not** paired against any other arm.

| Metric | 1 RTX + 6 TP6 | 2 RTX + 6 TP6 |
| --- | --- | --- |
| Weighted decode, median of three | 96.83 | 110.19 |
| Weighted decode, repeats | 92.90 / 97.91 / 96.83 | 110.19 / 110.70 / 109.20 |
| Counting decode, median | 174.54 | 231.51 |
| Prefill, best completed cell | 7,131 | 8,577 |

The headline protocol uses the fixed corpus `1972e572…`, tokenizer `c90dfa01…`
and nonce 61001, with 30 timed samples per arm and 15/15 objective checks passing.
Mixed traffic completes on both arms (`all_http_200`): median 175.84 token/s at
1 RTX and 188.47 at 2 RTX. **No causal speedup or qualification claim is made**:
the candidate throughput stands as measured, and "no speedup" means no causal
TP6-versus-TP4 qualification claim, not that these numbers are void.

## Baselines and comparison framing

- Published-v8 clean baselines: 1 RTX weighted **92.3024**, 2 RTX **112.5899**.
- The 1 RTX primary default comparison against published v8 is **cost-confounded**:
  the v8 baseline's `builtin-calibration` resolution is **inferred from source
  (`profile=None`)**, not observed in a captured resolved log, while the TP6
  launcher passes explicit `--spark-tp/--spark-ep` and resolves `legacy-heuristic`.
  The confound is stated, not erased.
- The 2 RTX primary comparison is **conservatively unverified** (the v8 2x resolved
  cost model is not observed), so no same-cost claim is made.
- A new **controlled-cost side arm** exists for context: TP4 1 RTX legacy
  `legacy-heuristic` **88.5130** versus TP6 1 RTX `legacy-heuristic` **96.8271**.
  This is a side arm, not a replacement for the default baseline.

## Component evidence (provenance, not numerical TP6 proof)

Earlier SM120/SM121 coordinator and expert packing builds and their FFI/mask
contract audits are preserved in the evidence archive
(`component-evidence/COMPONENT-EVIDENCE-INVENTORY.md`). They come from the same
source lineage but are **not** numerical TP6 test proof: build logs and the
`RealFullConstraintMasks` packing paths record compilation, not a TP6 numerical
result. Only the actual geometry selftests plus the qualifier numerical passes from
the earlier valid artifacts count. The recorded `ffi-symbol-shortfall` diagnostic is
**historical**: it was taken against an earlier broken library (`1c3`) and was
superseded by the full coordinator library `e9736c3e` and then the final
`b14cac3e`; it is **not** a current missing-ABI claim. Warm-candidate kernel/expert
tiling evidence is likewise distinct from the final dist binaries and does not
transfer as final-image proof.

## Known limitations

- **1 RTX retained 262144 strict full-parent reuse: FAILED.** prime64
  `code-reasoning` hit 262158 against the parent frontier 262290/291 (132 short),
  `cache_valid=False`. The lower contexts 0/32K/64K/128K all passed. This is a
  documented limitation, not an open work item.
- **2 RTX retained prime64 3-repeat sweep: not run** (accepted limitation). The 2x
  arm reports prime16 lower contexts plus a separate one-repeat 262144 diagnostic.
- The 2x old headline weighted **113.49 is INVALID** and retained for history only
  (same-process nonce contamination).
- **Reference equivalence: not performed.** No output-equivalence run against a
  reference implementation was made. The pure-TP6 expert path is numerically
  component-tested and passed bounded end-to-end functional checks, which is not a
  complete correctness qualification.
- The experimental canonical six-rank qualifier
  (`scripts/qualify-ds41-tp-ep-e2e-six.py`) was **not run** and is out of scope; it
  is recorded as not performed, not as a pass or a failure.

## Publication (completed)

1. `./push-containers.sh v9` was run from the clean build clone (HEAD
   `5d0d2095…`, `git status` clean) and published exactly the two images as `v9`
   and `latest`, the same digest per role (exit 0).
2. Registry digests were read back from the registry's own response and match:
   coordinator index `sha256:786d1d67…`, spark-expert manifest `sha256:f0c67407…`.
3. Anonymous verification used a fresh `DOCKER_CONFIG` with no host credentials:
   `docker pull …ds41rt-coordinator:v9` on raptor and `…ds41rt-spark-expert:v9` on
   ostrich both returned rc=0 with matching digests, created no auth file, and
   reused cached layers.

`ds41rt.config` is already at `:v9` (committed). The GitHub release remains pending
as a separate step. Rollback reference: the pre-push `latest` (v8) was coordinator
`sha256:08c2d6df9a0a6a365eff2c014172478b40d9f39d06437a1c9244c566181b9e40` and
spark-expert `sha256:9907983992916bb8e0f35ab869e12706cfe4613cc6dcd93f4ff53796c2d80bf6`.
Evidence: `v9-build/PUBLICATION-EVIDENCE.txt`.

## Provenance requirements for every number

Record source commit, model revision, input shapes, concurrency, KV encoding,
sampling, acceptance, warmup, every timed sample, memory, graph setup, image
identity and the exact launch configuration. Component timing alone does not
establish serving throughput, and an image measurement is not a checkpoint
measurement.
