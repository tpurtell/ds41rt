#!/usr/bin/env bash
# Run the DS41RT replicated-expert-group tests or benchmark against the pinned
# SparkInfer source WITHOUT dirtying it or importing an unverified copy.
#
# Why this exists
# ---------------
# scripts/verify-sparkinfer-source.py hashes *every* file under
# third_party/sparkinfer, including tests/ and benchmarks/, and
# scripts/build-release-artifacts.sh aborts when that hash no longer matches
# third_party/sparkinfer.lock.json. Harness files therefore live in this
# repository, and the pinned submodule is imported read-only.
#
# Two shadowing hazards are handled explicitly rather than assumed away:
#   * python/reference/tests is a real package that WOULD shadow the pinned
#     tests.moe fixture if it came earlier on sys.path, so the pinned tree goes
#     FIRST here, contrary to the usual repo-paths-first convention;
#   * a stray installed `b12x` (host venv or site-packages) could shadow the
#     pinned kernels, so after import the runner verifies that the loaded
#     modules actually resolve inside the pinned tree and fails closed.
# A verified lock is a precondition for that check, never a substitute.
#
#   scripts/run-tp-ep-kernel-checks.sh test  [pytest args...]
#   scripts/run-tp-ep-kernel-checks.sh bench [benchmark args...]
#   DS41RT_NATIVE_BUILD_DIR=<dir> scripts/run-tp-ep-kernel-checks.sh gpu-check
#
# `test` runs the CPU contract tests (including the TP6 exporter/pack oracle)
# against the verified pinned tree. `gpu-check` runs the native CMake selftests
# on a real CUDA device with DS41RT_REQUIRE_CUDA=1, so a pass is `Passed` and a
# missing device is a failure rather than a skip.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pin="${DS41RT_SPARKINFER_SOURCE_DIR:-$repo_root/third_party/sparkinfer}"
lock="$repo_root/third_party/sparkinfer.lock.json"

[[ -d "$pin" ]] || { echo "SparkInfer checkout missing at $pin" >&2; exit 2; }
[[ -f "$lock" ]] || { echo "SparkInfer lock missing at $lock" >&2; exit 2; }
[[ -f "$pin/b12x/moe/_shared/kernels/v41_slice_pipeline.py" ]] || {
  echo "pinned tree lacks the V4.1 slice pipeline" >&2; exit 2; }

# Fail closed on source drift before importing anything. The release build runs
# this same check, so the harness stays on the same contract. Verification is
# never skipped and never weakened; to run against a different copy, point
# DS41RT_SPARKINFER_SOURCE_DIR at a complete, lock-matching tree.
echo "== verifying SparkInfer at $pin against $(basename "$lock")"
python3 "$repo_root/scripts/verify-sparkinfer-source.py" \
  --source "$pin" --lock "$lock"

# The bench/test modules import `_pinned_sparkinfer`, which verifies the pinned
# tree and lock, prepends that tree, and fails closed if b12x comes from
# anywhere else -- so python/tools must be importable and the pin is no longer
# load-bearing by convention. It is still listed first so tests.moe cannot be
# shadowed by python/reference/tests before the resolver runs.
export DS41RT_SPARKINFER_SOURCE_DIR="$pin"
export PYTHONPATH="$pin:$repo_root/python/tools:$repo_root/python/reference:$repo_root/python${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

mode="${1:-}"
shift || true

verify_imports() {
  python3 - "$pin" <<'PY'
import pathlib, sys
pin = pathlib.Path(sys.argv[1]).resolve()
import b12x.moe._shared.kernels.v41_slice_pipeline as kernel
import tests.moe.test_v41_expert_numerics as oracle
for module in (kernel, oracle):
    path = pathlib.Path(module.__file__).resolve()
    if not path.is_relative_to(pin):
        sys.exit(f"FAIL: {module.__name__} resolved to {path}, outside {pin}")
print(f"imports verified from {pin}")
PY
}

case "$mode" in
  test)
    verify_imports
    tests=(
      "$repo_root/python/tests/test_v41_sentinel_masking.py"
      "$repo_root/python/tests/test_v41_ep_algebra.py"
      "$repo_root/python/tests/test_v41_spark_tp6_contract.py"
      "$repo_root/python/tests/test_v41_spark_tp3_launch_geometry.py"
      "$repo_root/python/tests/test_bench_tp_ep_kernel_tp6.py"
    )
    for candidate in "${tests[@]}"; do
      [[ -f "$candidate" ]] || {
        echo "kernel-check test missing: $candidate" >&2; exit 2; }
    done
    exec "$repo_root/scripts/run-with-python-env.sh" \
      python -m pytest "${tests[@]}" "$@"
    ;;
  bench)
    [[ -f "$repo_root/python/tools/benchmark_v41_ep_groups.py" ]] || {
      echo "benchmark harness missing" >&2; exit 2; }
    verify_imports
    exec "$repo_root/scripts/run-with-python-env.sh" \
      python "$repo_root/python/tools/benchmark_v41_ep_groups.py" "$@"
    ;;
  gpu-check)
    # Device-level native checks. These MUST run on a CUDA host and must not
    # skip: DS41RT_REQUIRE_CUDA turns a missing device into a failure, so the
    # ctest result is "Passed", never "Skipped (77)".
    build_dir="${DS41RT_NATIVE_BUILD_DIR:-}"
    [[ -n "$build_dir" && -d "$build_dir" ]] || {
      echo "DS41RT_NATIVE_BUILD_DIR must name a configured native build dir" >&2
      exit 2; }
    export DS41RT_REQUIRE_CUDA=1
    # Run ctest in this shell (not via exec) so its status is the script's
    # status, and fail when the filter matches nothing instead of reporting a
    # vacuous success. ctest's "no tests matched" status varies by version, so
    # the selected-test count is checked explicitly rather than inferred.
    set +e
    output="$(ctest --test-dir "$build_dir" --output-on-failure \
      -R 'ds41rt_v41_(expert_pack_geometry|expert_pack_tp3|route_reduce_planes)_selftest' 2>&1)"
    status=$?
    set -e
    printf '%s\n' "$output"
    if grep -q "No tests were found" <<<"$output"; then
      echo "no native selftest matched the filter in $build_dir" >&2
      exit 2
    fi
    if [[ "$status" -eq 8 ]]; then
      echo "ctest reported no matching tests in $build_dir" >&2
      exit 2
    fi
    exit "$status"
    ;;
  *)
    echo "usage: $0 {test|bench|gpu-check} [args...]" >&2
    echo "  test      CPU contract tests (verified pinned SparkInfer)" >&2
    echo "  bench     standalone per-topology benchmark" >&2
    echo "  gpu-check native CMake selftests on a real CUDA device" >&2
    exit 2
    ;;
esac
