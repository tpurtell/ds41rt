# DS41RT development

DS41RT serves the official `deepseek-ai/DeepSeek-V4.1-Flash` checkpoint through
the native five-host path. [architecture.md](architecture.md) defines ownership
and execution contracts; [docs/ENGINEERING.md](docs/ENGINEERING.md) describes
the selected release implementation; the [release checklist](docs/release-v1-checklist.md)
links the full qualification evidence.

## Execution contract

- One x86-64 SM120 RTX coordinator owns attention, routing, shared experts,
  embeddings, mHC, sampling, vision and all three dSpark stages and experts.
- Four ARM64 SM121 Sparks execute intermediate-dimension TP4 slices of the
  backbone's routed experts. Each receives the same top-6 routes. dSpark
  routed experts stay on RTX and use their separate top-3 routing.
- The only release checkpoint is the official native V4.1 Flash snapshot.
  The target concurrency is 16 requests with measured wave scheduling.
- Compressed global KV uses FP4 E2M1 values with group-16 E4M3 scales. Local
  128-token windows use FP8, while index keys retain their separate FP4 format.
- Engram tables and scales stay memory-mapped with bounded background
  prefetch and GPU staging. Do not preload the entire tables onto a GPU.
- Request leases, execution bindings, source selections and accepted-prefix
  publication must remain consistent across every speculative transaction.

## Source and build

**Never build on `/mnt/scratch` (NTFS kernel-driver bug).** This includes Cargo
sources, target directories, caches, temporary build files and container bind
aliases. The drive is intentionally read-only; do not remount it. Use root-NVMe
storage, preferably a unique `~/.cache/ds41rt/builds/<task>` directory, and check
all paths before running Cargo:

```bash
export CARGO_TARGET_DIR="$HOME/.cache/ds41rt/builds/my-task/cargo-target"
python3 scripts/assert-build-filesystem.py "$PWD" "$CARGO_TARGET_DIR" \
  "${CARGO_HOME:-$HOME/.cargo}" "${TMPDIR:-/tmp}"
```

The artifact build helpers perform the same filesystem check, including NTFS
bind mounts. Direct Cargo commands still require the preflight above. Start a
fresh target if the old cache has filesystem errors; do not treat copied Cargo
artifacts as verified. See [cluster-hosts.md](docs/cluster-hosts.md) for current
build-environment paths and the restricted rhea/moa fifth/sixth-node inventory.

Inspect `git status --short --branch` before editing and preserve unrelated
changes. Initialize the two runtime dependencies with:

```bash
git submodule update --init --recursive
python3 scripts/verify-sparkinfer-source.py \
  --source third_party/sparkinfer --lock third_party/sparkinfer.lock.json
python3 scripts/verify-xgrammar-source.py \
  --source third_party/xgrammar --lock third_party/xgrammar.lock.json
```

The b12x fork uses `master`. Update each dependency pin and its source-tree
lock together. Runtime builds must use the verified source tree.

For the host daemon, select the project Python environment and its shared
library through the existing wrapper:

```bash
DS41RT_PYTHON=.venv/bin/python scripts/run-with-python-env.sh \
  cargo build --manifest-path rust/Cargo.toml -p ds41rt-daemon
```

Do not set `PYTHONHOME`. Native libraries must match the active device and
current FFI symbols; a missing symbol can indicate an older ignored build.

`./build.sh` builds amd64 locally and ARM64 natively on a Spark, exports
artifacts, and distributes the expert image. Dirty checkouts receive an
automatic source manifest. Keep source unchanged until the build finishes.
`./build.sh --spark-hosts ostrich,dodo` restricts build/distribution hosts;
serving still requires four ranks. See [docker/README.md](docker/README.md).

`./wip.sh --slot NAME` retains the fingerprinted iteration workflow. The
standard `./run.sh` path uses the qualified V4.1 release images and validates
their source and dependency identity before launch.

## Focused checks

Use `just --list` to inspect the command surface. Choose checks that exercise
the changed production component and its relevant failure paths:

```bash
git diff --check
bash -n build.sh run.sh wip.sh scripts/release-common.sh
PYTHONPATH=python/reference .venv/bin/python -m pytest -q PATH_TO_TEST
cargo test --manifest-path rust/Cargo.toml -p CRATE TEST_FILTER
```

The reusable `scripts/qualify-ds41-*.py` tools and `docs/ds41-*.json` records
cover V4.1 numerical components. Read each record's scope and source hashes:
synthetic fixtures, real-weight components, four-Spark transport, full-model
logits and end-to-end throughput prove different requirements. Repeat affected
CUDA graph and numerical checks after changing native storage or arithmetic.

Preserve graph pointer/shape/workspace lifetimes and drain queued work before
publishing results or releasing storage. Weight and workspace admission must
precede allocation. Measure startup I/O separately from graph setup and
request execution.

## Integration and release

Develop on `dev`; promote `main` and create `release/vX` only after the final
container pair is qualified and published and documentation is complete.
See the [operation manual](docs/OPERATIONS.md) for branch procedures and the
[first-release checklist](docs/release-v1-checklist.md) for the current scope.

Full integration must cover 40-layer CED execution, engram prefetch, vision,
request-safe cache transactions, target sampling, dSpark verification and
rollback, concurrency 16, API behavior and restart readiness.

The qualified v3 baseline is recorded in
[docs/release-v1-performance.md](docs/release-v1-performance.md). New performance
work must record source/model revisions, input shapes, concurrency, KV encoding,
sampling, acceptance, warmup, every timed sample, memory and graph setup.
Component timing alone does not establish serving throughput.

Keep weights, generated catalogs, build artifacts, benchmark runs and local
configuration out of Git. Small useful fixtures and caches under `/tmp` may
remain. [TO_DELETE_SCAFFOLDING.md](TO_DELETE_SCAFFOLDING.md) tracks temporary
committed code; reusable regression tools and qualification evidence stay.
