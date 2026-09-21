#!/usr/bin/env bash
# CPU-only PLANNER / CHECKER / RENDERER for the v9 official / TP6 campaign.
#
# Scope limit: this script does NOT execute the campaign. It cannot launch a
# model, a container or a benchmark, and it performs no arm. It only prints the
# planned arm matrix (`plan`), validates a report manifest and the presence of
# its raw artifacts (`check`), and renders a report from already-recorded raw
# files (`render`). Listing an arm in `plan` is a plan, not a measurement.
# Hardware execution is a separate, explicitly owned step.
#
#   scripts/bench/run-v9-campaign.sh plan
#   scripts/bench/run-v9-campaign.sh check  --manifest M.json --package DIR [--strict]
#   scripts/bench/run-v9-campaign.sh render --manifest M.json --package DIR --output R.md
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
renderer="$repo_root/scripts/render-ds41-v9-tp6-reports.py"

usage() {
  cat <<'EOF'
Usage:
  scripts/bench/run-v9-campaign.sh plan
  scripts/bench/run-v9-campaign.sh check  --manifest M.json --package DIR [--strict]
  scripts/bench/run-v9-campaign.sh render --manifest M.json --package DIR --output R.md

CPU-only planner/checker/renderer. It does NOT execute benchmarks and never
touches hardware, Docker or SSH; `plan` output is a plan, not a record of
measurements.
EOF
}

command="${1:-plan}"
shift || true

case "$command" in
  plan)
    cat <<'EOF'
DS41RT v9 campaign PLANNED arm matrix (CPU plan only; nothing is executed or measured)
Only the two v8 TP4 arms below have been measured so far; every other row is a
plan, not a result.

  #  id                    quant      topology  rtx sparks  status / purpose
  1  official-v8-1x-tp4    official   TP4xEP1   1   4       MEASURED baseline
  2  official-v8-2x-tp4    official   TP4xEP1   2   4       MEASURED baseline
  3  official-v9-1x-tp4    official   TP4xEP1   1   4       planned: bridge, code revision at fixed rank count
  4  official-v9-2x-tp4    official   TP4xEP1   2   4       planned: bridge
  5  official-v9-1x-tp6    official   TP6xEP1   1   6       planned: candidate
  6  official-v9-2x-tp6    official   TP6xEP1   2   6       planned: candidate
  7  kernel-tp4-vs-tp6     official   TP4/TP6   1   4/6     planned: FFN component sweep (not serving throughput)

Comparisons (planned unless both sides are measured):
  - primary: 1x auto/default, labelled by resolved cost mode (builtin vs heuristic)
  - attribution side arm: controlled legacy-both within 1x
  - 4-Spark vs 6-Spark is a full-system comparison with its own campaign schema;
    the canonical six qualifier's fixed spark_count identity does not apply here
  - never attribute speed across different RTX counts

Per-arm order: capture identity -> correctness smokes -> warmup -> 3 timed
samples -> keep every sample and failure. See official-v9-campaign.md for the
exact commands and the provenance checklist.
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
