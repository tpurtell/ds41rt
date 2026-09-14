# DS41RT containers

The V4.1 topology needs two native development/build roles:

| Development image | Platform | CUDA target | Role |
| --- | --- | --- | --- |
| `ds41rt-coordinator-dev` | `linux/amd64` | `sm_120` | Attention, vision, engram projections, shared experts, all dSpark, API and sampling |
| `ds41rt-spark-expert-dev` | `linux/arm64` | `sm_121` | Backbone routed-expert TP4 worker |

`docker-bake.hcl` contains only these two targets. Build each on its native
machine. There is no checkpoint conversion image: the V4.1 release uses the
official checkpoint's native representations.

## Build and provenance

Run `./build.sh` from the repository root. The workflow validates configuration
and source identity, builds coordinator artifacts locally, stages matching
source on the first selected Spark, builds ARM64 artifacts there, and
distributes the expert image. Exported artifacts live under ignored `dist/`.

Dirty checkouts receive an automatic source manifest under `.ds41rt-release/`.
The build checks local and staged source against that manifest, so keep source
files unchanged until it finishes. An existing manifest can be supplied via
`DS41RT_RELEASE_SOURCE_MANIFEST`; ordinary dirty-tree builds do not require
creating one manually.

For a subset of available build hosts:

```bash
./build.sh --spark-hosts ostrich,dodo
```

This changes build/distribution targets only; serving still requires four
Spark ranks. `./wip.sh --slot NAME` provides the fingerprinted incremental
workflow. Use each script's `--help` for its current options.

## Runtime

[`../ds41rt.config`](../ds41rt.config) selects the official V4.1 Flash
checkpoint and the v3 coordinator/worker images. `./run.sh` validates image,
source, dependency, model, host and device identity before starting all five
containers. The standard launch uses port 8000, concurrency 16, dSpark, the
official 1,048,576/393,216 context/output limits, and 24 retained turns.

Run `./run.sh --help` for per-launch concurrency, KV pool, total memory,
retention, context, output, prefill and dSpark controls. Command-line values
override the config without changing its defaults.

Weights are mounted read-only from the host Hugging Face cache; containers do
not bundle them. Preserve the cache's blob/snapshot symlink layout when
mounting a checkpoint. Engram tables are mapped from host storage.

Release artifacts identify the same engine revision, verified SparkInfer and
XGrammar source, native architecture and checkpoint revision across all five
hosts. The [clean build/run report](../docs/release-v1-build-run.md) records the
qualified workflow and artifact hashes.

## Quantization coordinator image

Quantization is separate from the serving images above. Its authoritative state,
qualification evidence and recovery policy are in [the quantization plan](../quantization/PLAN.md).
Build the bundled-code coordinator from its pinned local base:

```bash
docker tag sha256:6213ea40c79617373562d7f2d3cc5fa25ca9d03e27dd213361b216c3e315e9f4 ds41rt-quant-coordinator-base:6213ea40c796
docker build -f docker/Dockerfile.quant-coordinator -t ds41rt-quant-coordinator:integration-v1 .
```

The base must already be available; do not substitute an unrelated tag. The
overlay dependencies are pinned in `quantization/coordinator-overlay.lock`.
The image installs the vendored V4.1 Transformers version, includes our GPTQModel
fork and scripts, and requires no workspace mount. Model weights stay outside it.
Use the inspected image ID, not the mutable build tag, for qualification and runs.

Its entry point accepts a runtime manifest and optional `--resume`, and runs the
quantization/weight-export stage. It is **not yet the complete one-shot workflow**:
deployment, complete export validation, upload and HF-cache finalization are still
being integrated. Do not treat the stage's terminal status as a completed model.
