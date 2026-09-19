#!/usr/bin/env bash
# Serve a v7 checkpoint from the PUBLISHED release images.
#
# Unlike runs/v7q-a1/serve-diffbot.sh this does not use the WIP slot dev build:
# it runs ghcr.io/tpurtell/ds41rt-coordinator:v8 and
# ghcr.io/tpurtell/ds41rt-spark-expert:v8, so measurements describe what ships.
#
# The RTX/Spark layer boundary is not hardcoded: the coordinator logs its
# bottom-up placement before it needs the experts, and that value is handed to
# the Spark experts as --first-layer. This mirrors how the recorded campaigns
# were configured without guessing per-config numbers.
#
# Usage:
#   serve-published.sh start <config>     # start coordinator, discover boundary, start experts
#   serve-published.sh stop <config>
#   serve-published.sh status <config>
#   serve-published.sh log <config> [coordinator|HOST]
set -euo pipefail

CONFIG="${1:-}"; shift || true
ACTION="${1:-}"; shift || true

COORD_IMAGE=ghcr.io/tpurtell/ds41rt-coordinator:v8
SPARK_IMAGE=ghcr.io/tpurtell/ds41rt-spark-expert:v8
COORD_NAME=ds41rt-v8-published
SPARK_NAME=ds41rt-spark-expert-published
EXPERT_PORT=19441
LANES=(10.55.0.1 10.55.0.2 10.55.0.3 10.55.0.4)
HOSTS=(ostrich dodo emu kiwi)
NVFP4_REL=hub/models--nvidia--DeepSeek-V4.1-Flash-NVFP4/snapshots/3431dde3247c13b5957f682b1e3c6fcae2566079
EXL3_REL=hub/models--diffbot--DeepSeek-V4.1-Flash-EXL3-2.0bpw-2x-RTX-PRO-6000/snapshots/28b7ab71ba2eb15569b08b91a8ea07df8eda8a75
RTX_UUID=GPU-fe5b6dd0-a77c-c8fb-6360-e1b9d9918ac0
RTX_UUID_2=GPU-95f8f212-9131-df99-fd53-7535965197d7
# DSPARK=0 serves the target path alone, which is how the v6 "target" phase was
# measured; the default is the dSpark path the headline uses.
DSPARK="${DSPARK:-1}"
dspark_args=()
[[ "$DSPARK" == 0 ]] || dspark_args=(--dspark)

# config: model | spark_count | rtx_gpus | prefill_batch | extra coordinator args
case "$CONFIG" in
  nvfp4-1x)  MODEL_REL="$NVFP4_REL"; SPARK_COUNT=4; RTX_GPUS=1; PREFILL_BATCH_TOKENS=2048; EXTRA=() ;;
  nvfp4-2x)  MODEL_REL="$NVFP4_REL"; SPARK_COUNT=4; RTX_GPUS=2; PREFILL_BATCH_TOKENS=2048; EXTRA=() ;;
  # 5090+2-spark: one card held to a 32 GiB budget including headroom, two TP2 Sparks.
  exl3-5090) MODEL_REL="$EXL3_REL"; SPARK_COUNT=2; RTX_GPUS=1; PREFILL_BATCH_TOKENS=256
             EXTRA=(--memory-reservation 32GiB --kv-pool-size 2GiB) ;;
  # 2x6000 0-spark: both cards, no Sparks at all.
  exl3-2x)   MODEL_REL="$EXL3_REL"; SPARK_COUNT=0; RTX_GPUS=2; PREFILL_BATCH_TOKENS=2048; EXTRA=() ;;
  *) echo "unknown config: $CONFIG" >&2; exit 2 ;;
esac

# Ad-hoc coordinator args for diagnostics, e.g.
#   EXTRA_ARGS="--memory-reservation 4GiB" runs/v8/serve.sh nvfp4-2x start
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  read -r -a _extra <<< "$EXTRA_ARGS"
  EXTRA+=("${_extra[@]}")
fi

SNAPSHOT="/root/.cache/huggingface/$MODEL_REL"
HOSTS=("${HOSTS[@]:0:SPARK_COUNT}")
LANES=("${LANES[@]:0:SPARK_COUNT}")

devices=(--device /dev/infiniband:/dev/infiniband --cap-add IPC_LOCK
         --ulimit memlock=-1:-1 --security-opt seccomp=unconfined --security-opt label=disable)
mounts=(-v /home/tj/.cache/huggingface:/root/.cache/huggingface:ro)

stop_all() {
  docker rm -f "$COORD_NAME" >/dev/null 2>&1 || true
  for host in "${HOSTS[@]}"; do
    ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" "docker rm -f $SPARK_NAME" >/dev/null 2>&1 || true
  done
}

start_experts() {
  local first_layer="$1" rank=0
  for host in "${HOSTS[@]}"; do
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" "docker rm -f $SPARK_NAME >/dev/null 2>&1; docker run -d --name $SPARK_NAME --network host --ipc host ${devices[*]} --gpus all ${mounts[*]} -e RUST_LOG=info $SPARK_IMAGE ds41rt expertd-native --snapshot $SNAPSHOT --native-lib /opt/ds41rt/lib/libds41rt_native.so --rank $rank --world $SPARK_COUNT --capacity 4096 --device-budget-bytes 107374182400 --first-layer $first_layer --listen 0.0.0.0:$EXPERT_PORT" >/dev/null
    echo "  $host rank $rank launched (first-layer $first_layer)"
    rank=$((rank + 1))
  done
}

peers() {
  local out=()
  if ((SPARK_COUNT == 0)); then
    echo "127.0.0.1:1,127.0.0.1:2,127.0.0.1:3,127.0.0.1:4"
  else
    for lane in "${LANES[@]}"; do out+=("$lane:$EXPERT_PORT"); done
    (IFS=,; echo "${out[*]}")
  fi
}

start_coordinator() {
  docker rm -f "$COORD_NAME" >/dev/null 2>&1 || true
  # Expose every card the layout claims: with only one visible the 2x path fails
  # its peer-capability query with "invalid device ordinal".
  local gpu_spec
  if ((RTX_GPUS == 2)); then
    gpu_spec="\"device=$RTX_UUID,$RTX_UUID_2\""
  else
    gpu_spec="\"device=$RTX_UUID\""
  fi
  docker run -d --name "$COORD_NAME" --network host --ipc host "${devices[@]}" \
    --gpus "$gpu_spec" ${mounts[*]} -e RUST_LOG=info "$COORD_IMAGE" \
    ds41rt serve-native --snapshot "$SNAPSHOT" \
      --native-lib /opt/ds41rt/lib/libds41rt_native.so \
      --peers "$(peers)" --rtx-gpus "$RTX_GPUS" --listen 0.0.0.0:8000 \
      --prefill-batch-tokens "$PREFILL_BATCH_TOKENS" --concurrency 16 \
      --prefix-cache-entries 20 --max-context-tokens 1048576 \
      --max-output-tokens 393216 --host-cache-bytes auto "${EXTRA[@]}" "${dspark_args[@]}" >/dev/null
}

case "$ACTION" in
  start)
    stop_all
    echo "starting $CONFIG coordinator (rtx-gpus $RTX_GPUS, sparks $SPARK_COUNT)"
    start_coordinator
    # The coordinator logs its planned RTX layer count before it needs experts:
    # single-card mode prints "bottom-up RTX expert placement layers=N", dual
    # mode prints "dual RTX bottom-up expert placement expert_layers=N".
    boundary=""
    for _ in $(seq 1 90); do
      boundary="$(docker logs "$COORD_NAME" 2>&1 | sed 's/\x1b\[[0-9;]*m//g' \
        | grep -o 'expert placement .*' | grep -oE 'expert_layers=[0-9]+|layers=[0-9]+' \
        | head -1 | cut -d= -f2 || true)"
      [[ -n "$boundary" ]] && break
      sleep 2
    done
    if [[ -z "$boundary" ]]; then
      echo "coordinator never reported its placement boundary" >&2
      docker logs "$COORD_NAME" 2>&1 | tail -20 >&2
      exit 1
    fi
    echo "  discovered RTX expert layers=$boundary"
    ((SPARK_COUNT > 0)) && start_experts "$boundary"
    # The coordinator reports its API ready before the experts finish loading
    # their layers, so wait on each expert separately.
    if ((SPARK_COUNT > 0)); then
      for host in "${HOSTS[@]}"; do
        for _ in $(seq 1 120); do
          if ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" \
             "docker logs $SPARK_NAME 2>&1" 2>/dev/null | grep -q "worker ready"; then
            echo "  $host experts ready"
            break
          fi
          sleep 2
        done
      done
    fi
    for _ in $(seq 1 120); do
      if docker logs "$COORD_NAME" 2>&1 | grep -q "target API ready"; then
        echo "  $CONFIG ready"
        exit 0
      fi
      sleep 3
    done
    echo "coordinator did not become ready" >&2
    docker logs "$COORD_NAME" 2>&1 | tail -20 >&2
    exit 1
    ;;
  stop) stop_all; echo "  $CONFIG stopped" ;;
  status)
    docker ps --filter "name=$COORD_NAME" --format 'coordinator {{.Status}}' || true
    curl -s -m 3 http://127.0.0.1:8000/v1/models | head -c 160; echo
    ;;
  log)
    target="${1:-coordinator}"
    if [[ "$target" == coordinator ]]; then
      docker logs "$COORD_NAME" 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | tail -40
    else
      ssh -o BatchMode=yes "$target" "docker logs $SPARK_NAME 2>&1" | tail -40
    fi
    ;;
  *) echo "usage: $0 <config> start|stop|status|log" >&2; exit 2 ;;
esac
