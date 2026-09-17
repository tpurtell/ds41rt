#!/usr/bin/env bash
set -euo pipefail

role="coordinator"
model_id="${DS41RT_MODEL_ID:-deepseek-ai/DeepSeek-V4.1-Flash}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --role)
      role="${2:?missing role}"
      shift 2
      ;;
    --model-id)
      model_id="${2:?missing model id}"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
model_cache="$hf_home/hub/models--${model_id//\//--}"

have() {
  command -v "$1" >/dev/null 2>&1
}

run_optional() {
  local label="$1"
  shift
  echo "## $label"
  if "$@" 2>&1; then
    true
  else
    echo "unavailable"
  fi
  echo
}

echo "# DS41RT doctor"
echo "role: $role"
echo "model_id: $model_id"
echo "hostname: $(hostname 2>/dev/null || true)"
echo "kernel: $(uname -srmo 2>/dev/null || true)"
echo "inside_container: $([[ -f /.dockerenv ]] && echo true || echo false)"
echo "cpu_arch: $(uname -m 2>/dev/null || true)"
echo "expected_cuda_arch: $([[ "$role" == "expert" ]] && echo sm_121 || echo sm_120)"
echo "hf_home: $hf_home"
echo "model_cache: $model_cache"
echo "model_cache_visible: $([[ -d "$model_cache" ]] && echo true || echo false)"
echo

if have nvidia-smi; then
  run_optional "nvidia-smi" nvidia-smi
  run_optional "gpu query" nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
else
  echo "## nvidia-smi"
  echo "missing"
  echo
fi

run_optional "python" python3 --version
run_optional "rust" rustc --version
run_optional "cargo" cargo --version
run_optional "cmake" cmake --version
run_optional "ninja" ninja --version
run_optional "rdma link" rdma link
run_optional "ibv_devices" ibv_devices
run_optional "ip route" ip route

if [[ -d "$model_cache/snapshots" ]]; then
  echo "## model snapshots"
  find "$model_cache/snapshots" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort | tail -n 5
  echo
fi
