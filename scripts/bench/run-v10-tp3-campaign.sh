#!/usr/bin/env bash
# CPU-only PLANNER / CHECKER / RENDERER for the v10 TP3 campaign.
#
# Scope limit: this script does NOT execute the campaign. It cannot launch a
# model, a container or a benchmark, and it never touches hardware, Docker, SSH
# or a GPU. It only prints the planned two-arm matrix (`plan`), validates a
# report manifest together with the presence and record accounting of its raw
# artifacts (`check`), and renders the reports from already-recorded raw files
# (`render`). Listing an arm in `plan` is a plan, not a measurement. Hardware
# execution is a separate, explicitly owned step.
#
#   scripts/bench/run-v10-tp3-campaign.sh plan
#   scripts/bench/run-v10-tp3-campaign.sh check  --manifest M.json --package DIR
#   scripts/bench/run-v10-tp3-campaign.sh check  --manifest M.json --package DIR --strict
#   scripts/bench/run-v10-tp3-campaign.sh render --manifest M.json --package DIR --output R.md
#
# `--strict` is the release gate: it fails while any measured family is pending
# or short of its derived expected record count.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
renderer="$repo_root/scripts/render-ds41-v10-tp3-reports.py"

usage() {
  cat <<'EOF'
Usage:
  scripts/bench/run-v10-tp3-campaign.sh plan
  scripts/bench/run-v10-tp3-campaign.sh check  --manifest M.json --package DIR [--strict]
  scripts/bench/run-v10-tp3-campaign.sh render --manifest M.json --package DIR --output R.md

CPU-only planner/checker/renderer. It does NOT execute benchmarks and never
touches hardware, Docker, SSH or a GPU; `plan` output is a plan, not a record of
measurements. --strict is the release gate and fails while any family is pending
or short of its derived expected record count.
EOF
}

command="${1:-plan}"
shift || true

case "$command" in
  plan)
    cat <<'EOF'
DS41RT v10 TP3 campaign PLANNED arm matrix (CPU plan only; nothing is executed or measured)

  #  id                      quant      topology   rtx sparks  config / status
  1  v10-native-tp3ep1       official   TP3xEP1    1   3       examples/configs/tp3ep1-native.config
  2  v10-exl3-compact-tp3    exl3       TP3        1   3       examples/configs/exl3-compact-tp3.config

Both arms are three-Spark, one-RTX deployments. They are NOT interchangeable:
  1. `v10-native-tp3ep1` names the topology explicitly (`SPARK_TP=3 SPARK_EP=1`,
     three ranks in ONE unreplicated group) and runs the official native
     checkpoint with `RTX_EXPERT_LAYERS=5` (5 RTX-local / 35 remote layers).
  2. `v10-exl3-compact-tp3` is the IMPLICIT compact layout (`SPARK_COUNT=3` with
     no `SPARK_TP`/`SPARK_EP`), runs the checkpoint-native EXL3 K3.25 (`k34`)
     publication under a hard 32 GiB memory reservation, KV 2 GiB and a bounded
     prefill of 256 tokens. Its resident bytes, allocation and ceiling are part
     of its identity.

Measured families, both arms (per family expected records derived at render time
from the live harness scripts, not asserted here):
  decode              10 cases x 3 repeats                          = 30
  prefill             30 cells (5 bases x 6 suffixes) x 3 timed      = 90
  retained prime      5 contexts x 9 cases x 3 repeats               = 135
  retained 2K control 1 context x 9 cases x 3 repeats                = 27
  concurrency         3 cases x 5 levels (C1..C16) x 3 repeats       = 45
  mixed               adaptive C4 + C16 batches                      = 2
  target-only         10 cases x 3 repeats, dSpark off               = 30
                                                                       ----
  performance total                                                    359
Diagnostics recorded but excluded from the total: startup/memory reservation
and device-budget lines, and the FFN kernel/tile microbenchmark (component
evidence, not serving throughput). Warmups, primes and lifecycle probes are
recorded but excluded from the 359.
Tool-eval is counted separately: 3 runs x 88 scenarios = 264 scenario-runs.

Status: NOT EXECUTED. The v10 image pair is not published, so no raw manifest
exists yet and every family renders as PENDING. This plan makes no performance,
memory, quality or readiness claim.

Per-arm order: capture identity -> correctness smokes -> placement/memory gates
-> warmup -> 3 timed samples per cell -> tool-eval x3 -> target-only relaunch.
See scripts/bench/v10-tp3-campaign.md for the exact commands, the corrected
tool-eval flags and the provenance checklist.
EOF
    ;;
  check)
    exec python3 "$renderer" --check "$@"
    ;;
  render)
    exec python3 "$renderer" "$@"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "unknown command: $command" >&2
    usage >&2
    exit 2
    ;;
esac
