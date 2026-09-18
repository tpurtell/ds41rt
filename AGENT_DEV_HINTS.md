# DS41RT agent development hints

Keep this file short; inspect script `--help` and `ds41rt.config` for details.

## Hardware and network

- Coordinator `raptor` (`x86_64`, `172.22.2.12`, fabric `10.55.0.22`): 2x RTX PRO 6000 Blackwell 96 GB, SM120, 400 W caps, standard memory speed.
- Spark TP4 ranks: `ostrich`, `dodo`, `emu`, `kiwi` (`aarch64`, one GB10/121.6 GiB, SM121 each). Lane A is `10.55.0.1..4` in rank order.
- The second NICs currently report `10.55.0.7..10`; `ds41rt.config` still says `10.55.0.5..8`. Run `scripts/run-on-hosts.sh ostrich,dodo,emu,kiwi 'ip -br -4 addr'` before dual-rail work.
- API: `http://127.0.0.1:8000`; expert port: `19441`. Only one performance run/deployment at a time: GPUs, ports, WIP containers, and Sparks are shared.
- Model: `deepseek-ai/DeepSeek-V4.1-Flash@dba1be0a40aa45a94ad051997016db3960a90277`; cache is `${HF_HOME:-$HOME/.cache/huggingface}` on every host.

## Build and launch

Use one git worktree/branch and globally unique WIP slot per goal (`S=<goal>-<agent>`).

```bash
./scripts/doctor.sh                         # local coordinator
./build.sh                                  # only for base image/toolchain changes
./wip.sh --slot "$S" --role coordinator     # API/scheduler/RTX-only change
./wip.sh --slot "$S" --role expert          # Spark/expert-only change
./wip.sh --slot "$S" --role both            # shared Rust/CUDA/protocol change
ssh ostrich 'docker exec ds41rt-spark-expert-wip /wip/source/scripts/doctor.sh --role expert'
./run.sh --wip --wip-slot "$S" --dry-run
./run.sh --wip --wip-slot "$S" --restart
./stop.sh
```

`wip.sh` freezes source, builds incrementally in persistent containers, builds ARM64 on `ostrich`, then distributes expert artifacts. Use `--from-slot BASE` for cheap A/B clones; use `--recreate` only after a dev-image change. Standard release paths are `./build.sh`, `./run.sh --dry-run`, then `./run.sh`.

Shells/one-offs:

```bash
./scripts/ds41rt-dev.sh coordinator -- bash
ssh ostrich "docker exec ds41rt-spark-expert-wip bash -lc '<command>'"
./scripts/run-on-hosts.sh ostrich,dodo,emu,kiwi '<read-only command>'
docker exec ds41rt-coordinator-wip /wip/slots/$S/coordinator/workspace/scripts/wip-process.sh log coordinator-8000 200
ssh ostrich docker exec ds41rt-spark-expert-wip /wip/slots/$S/spark-expert/workspace/scripts/wip-process.sh log expert-19441 200
```

## Test and measure

```bash
./scripts/api-smoke.sh
./scripts/api-constrained-smoke.sh
./scripts/run-with-python-env.sh python scripts/bench-ds41-concurrent-api.py --help
./scripts/run-with-python-env.sh python scripts/bench-ds41-release-decode.py --help
./scripts/run-with-python-env.sh python scripts/bench-ds41-release-prefill-matrix.py --help
./scripts/run-with-python-env.sh python scripts/qualify-ds41-tool-eval.py --help
```

- Expert/kernel experiments: `scripts/phase0-spark-tcp-bench.sh`, `python/tools/`, and `third_party/sparkinfer/benchmarks/`.
- Profiles: enable only the relevant `DS41RT_*_TIMING`/`*_TRACE` variable; summarize with `scripts/summarize-ds41-*.py`. Tracing perturbs throughput.
- Thermal/power history: `~/Developer/nvidiatempgraph/gpu_monitor.db`. Check `nvidia-smi` before interpreting regressions.
- Performance loop: warm up; change one variable; use identical cached prompts; interleave baseline/candidate; use 3 runs for a final candidate. Prioritize mixed/code/topic and real workloads over counting. Record exact config, commit, temperatures/power, median, and variance.
- Validation order: focused unit/kernel check -> API smoke -> targeted correctness/acceptance -> targeted performance. Do not run the full qualification suite for exploration.

## Process

- Start clean and pull `origin/dev`; never overwrite another agent's branch, WIP slot, containers, or results.
- Preserve defaults unless a repeatable performance winner passes correctness. Avoid allocations, synchronization, and host round trips in steady-state loops; do not trade loading speed for marginal runtime gains.
- Submodules are pinned. For `third_party/sparkinfer`, commit and push the fork first, then update the superproject pointer and `third_party/sparkinfer.lock.json`; run `scripts/verify-sparkinfer-source.py`.
- Put disposable output under ignored `runs/<slot>/`; clean it when done. Commit durable evidence only when it supports a decision.
- Commit coherent checkpoints and push immediately. For performance commits, include approximate `before -> after` metrics and conditions in the commit message.
