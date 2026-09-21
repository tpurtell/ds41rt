#!/usr/bin/env bash
# Six-rank qualification gate over the existing candidate launcher.
#
# Validates the staged fixture config (exactly six ranks, an approved replicated
# topology, and the global minimum-of-six device budget) and then delegates to
# scripts/run-tp-ep-native-candidate.sh. It never builds, never starts or stops
# anything itself, and it leaves 4-Spark runs and the generic launcher untouched:
# the budget ceiling is a six-rank qualification fixture input, not a runtime
# default. A rejected config exits 2 before the launcher runs, so no remote
# command is ever sent.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
launcher="$repo_root/scripts/run-tp-ep-native-candidate.sh"
max_budget=109119320064

usage() {
  echo "Usage: scripts/run-tp-ep-six-candidate.sh [plan|start|status|stop] --config FILE [launcher options]" >&2
}

mode="plan"
if [[ $# -gt 0 && "$1" =~ ^(plan|start|status|stop)$ ]]; then mode="$1"; shift; fi
pass=("$@")
config=""
for ((i = 0; i < ${#pass[@]}; i++)); do
  [[ "${pass[$i]}" == "--config" ]] && config="${pass[$((i + 1))]:-}"
done
[[ -n "$config" ]] || { usage; exit 2; }
[[ -f "$config" ]] || { echo "six-candidate: config not found: $config" >&2; exit 2; }

# Missing keys return empty (and stay fail-closed at exit 2) instead of aborting
# the pipeline with exit 1 under pipefail.
value() { grep -E "^$1=" "$config" 2>/dev/null | tail -1 | cut -d= -f2- || true; }
fail() { echo "six-candidate: $*" >&2; exit 2; }

count="$(value SPARK_COUNT)"
tp="$(value SPARK_TP)"
ep="$(value SPARK_EP)"
budget="$(value SPARK_DEVICE_BUDGET_BYTES)"
rtx="$(value RTX_GPUS)"
layers="$(value RTX_EXPERT_LAYERS)"
dspark="$(value DSPARK)"

[[ "$count" == "6" ]] || fail "SPARK_COUNT must be 6, got '${count}'"
[[ "$tp" =~ ^[0-9]+$ && "$ep" =~ ^[0-9]+$ ]] || fail "SPARK_TP/SPARK_EP must be integers, got '${tp}/${ep}'"
((tp * ep == 6)) || fail "SPARK_TP x SPARK_EP must be 6, got ${tp}x${ep}"
case "$rtx:$tp:$ep" in
  1:3:2 | 2:2:3 | 2:3:2 | 1:6:1 | 2:6:1) ;;
  *) fail "approved six-rank arms are 1rtx6-tp3ep2 (1:3:2), 2rtx6-tp2ep3 (2:2:3), 2rtx6-tp3ep2 (2:3:2), and the pure unreplicated 1rtx6-tp6ep1 (1:6:1) / 2rtx6-tp6ep1 (2:6:1); got ${rtx}:${tp}:${ep}" ;;
esac
[[ "$budget" =~ ^[0-9]+$ ]] || fail "SPARK_DEVICE_BUDGET_BYTES must be an integer, got '${budget}'"
# Length/string bound, never Bash arithmetic: a 2^64 digit string must not wrap.
canonical="$budget"
while [[ "$canonical" == 0* && ${#canonical} -gt 1 ]]; do canonical="${canonical#0}"; done
[[ "$canonical" != "0" ]] || fail "SPARK_DEVICE_BUDGET_BYTES must be positive"
if (( ${#canonical} > ${#max_budget} )) ||
  { (( ${#canonical} == ${#max_budget} )) && [[ "$canonical" > "$max_budget" ]]; }; then
  fail "SPARK_DEVICE_BUDGET_BYTES ${budget} exceeds the global minimum-of-six ${max_budget}"
fi
[[ "$rtx" == "1" || "$rtx" == "2" ]] || fail "RTX_GPUS must be 1 or 2, got '${rtx}'"
[[ "$dspark" == "on" ]] || fail "DSPARK must be on"
if [[ "$rtx" == "1" ]]; then
  # A 1-RTX arm either keeps no local routed layers (0, all 40 remote) or an
  # explicit local count in 1..=39 with a placement handoff, which is the current
  # official 1-RTX placement shape (5 local / 35 remote).
  if [[ "$layers" != "0" ]]; then
    [[ "$layers" =~ ^([1-9]|[1-3][0-9])$ ]] ||
      fail "a 1-RTX six-rank arm must set RTX_EXPERT_LAYERS=0 or an explicit local count in 1..39, got '${layers}'"
  fi
else
  [[ "$layers" == "20" ]] || fail "a 2-RTX six-rank arm must set RTX_EXPERT_LAYERS=20 explicitly, got '${layers}'"
fi
for index in 0 1 2 3 4 5; do
  [[ -n "$(value "SPARK_${index}_HOST")" ]] || fail "SPARK_${index}_HOST is missing"
  [[ -n "$(value "SPARK_${index}_LANE_A")" ]] || fail "SPARK_${index}_LANE_A is missing"
done

exec bash "$launcher" "$mode" "${pass[@]}"
