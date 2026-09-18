#!/usr/bin/env bash
set -euo pipefail
snapshot=/root/.cache/huggingface/hub/models--nvidia--DeepSeek-V4.1-Flash-NVFP4/snapshots/3431dde3247c13b5957f682b1e3c6fcae2566079
hosts=(ostrich dodo emu kiwi)
for rank in 0 1 2 3; do
  ssh "${hosts[$rank]}" bash -s -- "$rank" "$snapshot" <<'REMOTE'
set -euo pipefail
docker run -d --name ds41rt-nvfp4-ab --gpus all --network host --ipc host --device /dev/infiniband:/dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1:-1 --security-opt seccomp=unconfined --security-opt label=disable -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" ghcr.io/tpurtell/ds41rt-spark-expert:v7 ds41rt expertd-native --snapshot "$2" --native-lib /opt/ds41rt/lib/libds41rt_native.so --rank "$1" --world 4 --capacity 4096 --device-budget-bytes 107374182400 --first-layer 0 --listen 0.0.0.0:19441
REMOTE
done
docker run -d --name ds41rt-nvfp4-ab --gpus device=0 --network host --ipc host --device /dev/infiniband:/dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1:-1 --security-opt seccomp=unconfined --security-opt label=disable -v /home/tj/.cache/huggingface:/root/.cache/huggingface:ro ghcr.io/tpurtell/ds41rt-coordinator:v7 ds41rt serve-native --snapshot "$snapshot" --native-lib /opt/ds41rt/lib/libds41rt_native.so --peers 10.55.0.1:19441,10.55.0.2:19441,10.55.0.3:19441,10.55.0.4:19441 --rtx-gpus 1 --listen 0.0.0.0:8000 --prefill-batch-tokens 2048 --concurrency 16 --prefix-cache-entries 20 --max-context-tokens 1048576 --max-output-tokens 393216 --host-cache-bytes auto --dspark
