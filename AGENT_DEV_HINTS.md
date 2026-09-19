# Agent development hints

- Hosts: `raptor` (local x86_64, 172.22.2.12 / fabric 10.55.0.22), 2× RTX PRO 6000 96 GB, SM120; 400 W caps, stock memory clocks. Spark TP4 ranks: `ostrich,dodo,emu,kiwi`, ARM64 GB10/SM121, fabric 10.55.0.1–4. SSH by hostname. Verify current NICs/power with `ip -br -4 addr` / `nvidia-smi`; secondary-rail config may be stale.
- Model/revision/topology: `ds41rt.config`. HF cache: `${HF_HOME:-$HOME/.cache/huggingface}`; reuse cached snapshots. API `localhost:8000`, expert port `19441`.
- Isolation: separate git worktree/branch + unique WIP slot per goal. **Slots isolate artifacts, not GPUs, ports, build staging or containers. Serialize WIP builds and performance runs; coordinate before restarting services.** Preserve others' files/results.

```bash
S=my-goal                         # unique across agents
./scripts/doctor.sh
./wip.sh --slot "$S" --role both  # freeze source; build RTX + ARM64; distribute
# Faster A/B: --from-slot BASE --role coordinator (or expert)
./scripts/run-wip.sh --wip-slot "$S" --dry-run
./scripts/run-wip.sh --wip-slot "$S" --restart
```

Use `scripts/run-wip.sh` directly: current `run.sh` does not dispatch `--wip`, despite older help text. `--config FILE` selects an experiment config; otherwise launch uses the slot's frozen config. `./build.sh` builds release images; `./run.sh --dry-run` / `./run.sh` deploys release images. WIP `--recreate` destroys shared containers/build caches.

```bash
./scripts/ds41rt-dev.sh coordinator -- bash  # disposable shell, checkout mounted; one RTX
# Persistent build/eval environments (source is copied, not a live checkout mount):
docker exec -it ds41rt-coordinator-wip bash
ssh -t ostrich docker exec -it ds41rt-spark-expert-wip bash
./scripts/run-on-hosts.sh ostrich,dodo,emu,kiwi 'nvidia-smi'
```

- WIP artifacts: `/wip/slots/$S/{coordinator,spark-expert}`; source beneath `workspace/`. Logs: that workspace's `scripts/wip-process.sh log coordinator-8000 200` (or `expert-19441`) inside the corresponding container. Run CUDA/PyTorch/kernel checks in the matching architecture's dev container; do not reuse host Python venvs there.
- API checks: `scripts/api-smoke.sh`, `scripts/api-constrained-smoke.sh`. Eval Python: `scripts/run-with-python-env.sh python scripts/<tool> --help`. Tools: `bench-ds41-release-decode.py`, `bench-ds41-concurrent-api.py`, `bench-ds41-release-prefill-matrix.py`, `qualify-ds41-tool-eval.py`. Kernel checks: `native/tests/`, `python/tools/`, `third_party/sparkinfer/{tests,benchmarks}/`.
- Correctness → warmup → identical-config A/B; three runs for final qualification, no full suite for exploration. Prioritize code/topic/mixed decode; counting is secondary. Record commits/config, acceptance, median/variance and power. Thermal history: `~/Developer/nvidiatempgraph/gpu_monitor.db`; no mandatory cooldown. Profiling perturbs timing.
- Keep defaults unless a repeatable winner passes correctness; preserve load speed. Read submodule `AGENTS.md`. Push SparkInfer fork changes before updating its pin + lock; run `scripts/verify-sparkinfer-source.py`.
- Disposable evidence: ignored `runs/$S/`; clean your own intermediates. Commit coherent changes and push; performance commit messages include approximate before → after and conditions. Never package experiments/caches into release images.
