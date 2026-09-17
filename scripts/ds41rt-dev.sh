#!/usr/bin/env bash
set -euo pipefail

role="${1:-coordinator}"
if [[ $# -gt 0 ]]; then
  shift
fi
if [[ "${1:-}" == "--" ]]; then
  shift
fi

case "$role" in
  coordinator)
    image="${DS41RT_COORDINATOR_DOCKER_DEV:-ds41rt-coordinator-dev}"
    ;;
  expert|spark)
    image="${DS41RT_SPARK_EXPERT_DOCKER_DEV:-ds41rt-spark-expert-dev}"
    ;;
  dev)
    image="${DS41RT_COORDINATOR_DOCKER_DEV:-ds41rt-coordinator-dev}"
    ;;
  *)
    echo "unknown role: $role" >&2
    exit 2
    ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
hf_home="${HF_HOME:-$HOME/.cache/huggingface}"

docker_args=(
  run --rm
  -v "$repo_root:/workspace/ds41rt"
  -v "$hf_home:$hf_home:ro"
  -v "$hf_home:/root/.cache/huggingface:ro"
  -e HF_HOME="$hf_home"
  -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  -e TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  -e DS41RT_MODEL_ID="${DS41RT_MODEL_ID:-deepseek-ai/DeepSeek-V4.1-Flash}"
  -e DS41RT_MODEL_REVISION="${DS41RT_MODEL_REVISION:-}"
)

if [[ "$role" == "expert" || "$role" == "spark" ]]; then
  docker_args+=(--gpus all)
else
  source "$repo_root/scripts/release-common.sh"
  release_load_config "${DS41RT_CONFIG:-$repo_root/ds41rt.config}"
  release_need nvidia-smi
  release_resolve_coordinator_gpu_identity
  docker_args+=(
    --gpus device="$RELEASE_COORDINATOR_GPU_UUID"
    -e CUDA_VISIBLE_DEVICES="$RELEASE_COORDINATOR_GPU_UUID"
    -e NVIDIA_VISIBLE_DEVICES="$RELEASE_COORDINATOR_GPU_UUID"
  )
fi

if [[ "$repo_root" != "/workspace/ds41rt" ]]; then
  docker_args+=(-v "$repo_root:$repo_root")
fi

if [[ -t 0 && -t 1 ]]; then
  docker_args+=(-it)
fi

if [[ "$role" == "expert" || "$role" == "spark" ]]; then
  docker_args+=(--net=host --ipc=host --ulimit memlock=-1:-1 --cap-add IPC_LOCK)
  if [[ -e /dev/infiniband ]]; then
    docker_args+=(--device=/dev/infiniband)
  fi
fi

if [[ "$role" == "coordinator" || "$role" == "dev" ]]; then
  if [[ "${DS41RT_DOCKER_HOST_NETWORK:-0}" == "1" ]]; then
    docker_args+=(--net=host --ipc=host --ulimit memlock=-1:-1 --cap-add IPC_LOCK)
    if [[ -e /dev/infiniband ]]; then
      docker_args+=(--device=/dev/infiniband)
    fi
  fi
fi

if [[ $# -eq 0 ]]; then
  set -- bash
fi

exec docker "${docker_args[@]}" "$image" "$@"
