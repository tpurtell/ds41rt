#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$repo_root/scripts/release-common.sh"

usage() {
  cat <<'EOF'
Usage: ./run.sh [OPTIONS]

Starts native DeepSeek V4.1 on the RTX coordinator and four Spark TP ranks.
Command-line values override ds41rt.config for this launch.

  --config FILE                 alternate complete configuration
  --listen HOST:PORT            API address (default 0.0.0.0:8000)
  --rtx-gpus auto|1|2           select automatically or force a layout (default auto)
  --concurrency N               active requests, 1..16 (default 16)
  --kv-pool-size SIZE           exact KV/index pool, e.g. 22.5GB
  --memory-reservation SIZE     total GPU ceiling, e.g. 90% or 80GiB
  --prefix-cache-entries N      turn and prompt retention, 0..128 (default 24)
  --max-context-tokens N        context limit (default 1048576)
  --max-output-tokens N         output limit (default 393216)
  --prefill-batch-tokens N      prefill step, 80..4096 (default 2048)
  --dspark | --no-dspark        enable or disable native dSpark
  --restart                     replace the current release deployment
  --dry-run                     validate without changing services
EOF
}

config="$repo_root/ds41rt.config"
restart=0
dry_run=0
declare -A overrides=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) config="${2:?$1 requires FILE}"; shift 2 ;;
    --listen) overrides[ADDR]="${2:?$1 requires HOST:PORT}"; shift 2 ;;
    --rtx-gpus) overrides[RTX_GPUS]="${2:?$1 requires auto, 1, or 2}"; shift 2 ;;
    --concurrency) overrides[CONCURRENCY]="${2:?$1 requires N}"; shift 2 ;;
    --kv-pool-size) overrides[KV_POOL_SIZE]="${2:?$1 requires SIZE}"; shift 2 ;;
    --memory-reservation) overrides[MEMORY_RESERVATION]="${2:?$1 requires SIZE}"; shift 2 ;;
    --prefix-cache-entries) overrides[PREFIX_CACHE_ENTRIES]="${2:?$1 requires N}"; shift 2 ;;
    --max-context-tokens) overrides[MAX_CONTEXT_TOKENS]="${2:?$1 requires N}"; shift 2 ;;
    --max-output-tokens) overrides[MAX_OUTPUT_TOKENS]="${2:?$1 requires N}"; shift 2 ;;
    --prefill-batch-tokens) overrides[PREFILL_BATCH_TOKENS]="${2:?$1 requires N}"; shift 2 ;;
    --dspark) overrides[DSPARK]=on; shift ;;
    --no-dspark) overrides[DSPARK]=off; shift ;;
    --restart) restart=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) release_die "unknown run argument: $1" ;;
  esac
done

release_load_config "$config"
for name in "${!overrides[@]}"; do printf -v "$name" '%s' "${overrides[$name]}"; done
case "$DSPARK" in on|off) ;; *) release_die "DSPARK must be on or off" ;; esac
case "$RTX_GPUS" in auto|1|2) ;; *) release_die "RTX_GPUS must be auto, 1, or 2" ;; esac
[[ "$CONCURRENCY" =~ ^([1-9]|1[0-6])$ ]] || release_die "CONCURRENCY must be in 1..16"
[[ "$PREFIX_CACHE_ENTRIES" =~ ^([0-9]|[1-9][0-9]|1[01][0-9]|12[0-8])$ ]] || release_die "PREFIX_CACHE_ENTRIES must be in 0..128"
[[ "$MAX_CONTEXT_TOKENS" =~ ^[1-9][0-9]*$ ]] && ((MAX_CONTEXT_TOKENS <= 1048576)) || release_die "MAX_CONTEXT_TOKENS must be in 1..1048576"
[[ "$MAX_OUTPUT_TOKENS" =~ ^[1-9][0-9]*$ ]] && ((MAX_OUTPUT_TOKENS <= 393216)) || release_die "MAX_OUTPUT_TOKENS must be in 1..393216"
[[ "$PREFILL_BATCH_TOKENS" =~ ^[0-9]+$ ]] && ((PREFILL_BATCH_TOKENS >= 80 && PREFILL_BATCH_TOKENS <= 4096)) || release_die "PREFILL_BATCH_TOKENS must be in 80..4096"
[[ -z "$KV_POOL_SIZE" || "$KV_POOL_SIZE" =~ ^[0-9]+([.][0-9]{1,6})?(B|MB|GB|MiB|GiB)?$ ]] || release_die "KV_POOL_SIZE has an invalid unit"
[[ -z "$MEMORY_RESERVATION" || "$MEMORY_RESERVATION" =~ ^[0-9]+([.][0-9]{1,6})?((B|MB|GB|MiB|GiB)|%)$ ]] || release_die "MEMORY_RESERVATION has an invalid unit"
((restart == 0 || dry_run == 0)) || release_die "--restart and --dry-run are mutually exclusive"

for tool in docker ssh curl jq ss nvidia-smi sha256sum python3; do release_need "$tool"; done
docker info >/dev/null 2>&1 || release_die "local Docker daemon is unavailable"
docker image inspect "$COORDINATOR_DOCKER_INFERENCE" >/dev/null 2>&1 || release_die "coordinator image is missing: $COORDINATOR_DOCKER_INFERENCE (run ./build.sh)"
hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
release_resolve_local_model_revision "$hf_home"
release_resolve_coordinator_gpu_identity
snapshot_rel="hub/models--${RELEASE_MODEL_ID//\//--}/snapshots/$RELEASE_MODEL_REVISION"
model_is_exl3="$(jq -r '.quantization_config.quant_method == "exl3"' "$hf_home/$snapshot_rel/config.json")"
coordinator="$RELEASE_COORDINATOR_CONTAINER_NAME"

reclaim_args=()
if ((restart)) && docker container inspect "$coordinator" >/dev/null 2>&1; then
  while IFS= read -r pid; do
    [[ "$pid" =~ ^[0-9]+$ ]] && reclaim_args+=(--reclaim-pid "$pid")
  done < <(docker top "$coordinator" -eo pid 2>/dev/null | tail -n +2 || true)
fi
gpu_selection="$(python3 "$repo_root/scripts/select-release-gpus.py" \
  --mode "$RTX_GPUS" \
  --primary-uuid "$RELEASE_COORDINATOR_GPU_UUID" \
  --concurrency "$CONCURRENCY" \
  --max-context-tokens "$MAX_CONTEXT_TOKENS" \
  --retained-turns "$PREFIX_CACHE_ENTRIES" \
  --kv-pool-size "$KV_POOL_SIZE" \
  --memory-reservation "$MEMORY_RESERVATION" \
  "${reclaim_args[@]}")"
RELEASE_RTX_GPUS="$(jq -r '.count' <<<"$gpu_selection")"
mapfile -t release_gpu_uuids < <(jq -r '.gpus[].uuid' <<<"$gpu_selection")
mapfile -t release_gpu_indices < <(jq -r '.gpus[].index' <<<"$gpu_selection")
mapfile -t release_gpu_pci < <(jq -r '.gpus[].pci' <<<"$gpu_selection")
gpu_uuid_csv="$(IFS=,; echo "${release_gpu_uuids[*]}")"
gpu_index_csv="$(IFS=,; echo "${release_gpu_indices[*]}")"
gpu_pci_csv="$(IFS=,; echo "${release_gpu_pci[*]}")"
spark_first_layer=$((RELEASE_RTX_GPUS == 2 ? 20 : 0))
if ((RELEASE_RTX_GPUS == 2)); then
  gpu_request="\"device=$gpu_uuid_csv\""
else
  gpu_request="device=$gpu_uuid_csv"
fi

sparkinfer_commit="$(python3 "$repo_root/scripts/verify-sparkinfer-source.py" --source "$repo_root/third_party/sparkinfer" --lock "$repo_root/third_party/sparkinfer.lock.json" --print-revision)"
engine_commit="$(docker image inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$COORDINATOR_DOCKER_INFERENCE")"
image_sparkinfer="$(docker image inspect -f '{{index .Config.Labels "io.ds41rt.sparkinfer.revision"}}' "$COORDINATOR_DOCKER_INFERENCE")"
[[ -n "$engine_commit" && "$engine_commit" != '<no value>' ]] || release_die "coordinator image has no engine revision"
[[ "$image_sparkinfer" == "$sparkinfer_commit" ]] || release_die "coordinator image uses another SparkInfer revision (run ./build.sh)"

hosts=("$SPARK_0_HOST" "$SPARK_1_HOST" "$SPARK_2_HOST" "$SPARK_3_HOST")
lanes=("$SPARK_0_LANE_A" "$SPARK_1_LANE_A" "$SPARK_2_LANE_A" "$SPARK_3_LANE_A")
spark_exl3_identity=""
for host in "${hosts[@]}"; do
  spark_manifest="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" bash -s -- "$SPARK_EXPERT_DOCKER_INFERENCE" "$engine_commit" "$sparkinfer_commit" "$snapshot_rel" "$model_is_exl3" <<'REMOTE'
set -euo pipefail
image="$1"; engine="$2"; sparkinfer="$3"; snapshot_rel="$4"
docker info >/dev/null
test "$(docker image inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$image")" = "$engine"
test "$(docker image inspect -f '{{index .Config.Labels "io.ds41rt.sparkinfer.revision"}}' "$image")" = "$sparkinfer"
hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
test -d "$hf_home/$snapshot_rel"
! find "$hf_home/$snapshot_rel" -xtype l -print -quit | grep -q .
if [[ "$5" == true ]]; then
  docker run --rm --network none --entrypoint /bin/cat "$image" /opt/ds41rt/lib/exl3/manifest.json
fi
REMOTE
)"
  if [[ "$model_is_exl3" == true ]]; then
    identity="$(release_exl3_package_identity "$sparkinfer_commit" <<<"$spark_manifest")"
    [[ -z "$spark_exl3_identity" || "$spark_exl3_identity" == "$identity" ]] ||
      release_die "Spark EXL3 packages differ across hosts; rebuild/distribute matching images"
    spark_exl3_identity="$identity"
  fi
done

expert_capacity=4096
if ((PREFILL_BATCH_TOKENS <= 80)); then expert_capacity=80
elif ((PREFILL_BATCH_TOKENS <= 256)); then expert_capacity=256
elif ((PREFILL_BATCH_TOKENS <= 1024)); then expert_capacity=1024
fi
peers="${lanes[0]}:$EXPERT_PORT,${lanes[1]}:$EXPERT_PORT,${lanes[2]}:$EXPERT_PORT,${lanes[3]}:$EXPERT_PORT"
fingerprint="$(printf '%s\n' "$engine_commit" "$RELEASE_MODEL_ID" "$RELEASE_MODEL_REVISION" "$ADDR" "$RELEASE_RTX_GPUS" "$gpu_uuid_csv" "$gpu_pci_csv" "$CONCURRENCY" "$KV_POOL_SIZE" "$MEMORY_RESERVATION" "$PREFIX_CACHE_ENTRIES" "$MAX_CONTEXT_TOKENS" "$MAX_OUTPUT_TOKENS" "$PREFILL_BATCH_TOKENS" "$DSPARK" "$SPARK_DEVICE_BUDGET_BYTES" "$spark_first_layer" "$peers" "$spark_exl3_identity" | sha256sum | awk '{print $1}')"
spark_prefix="$RELEASE_SPARK_CONTAINER_PREFIX"

if ((dry_run)); then
  echo "Dry-run checks passed for native V4.1; no services changed."
  echo "  RTX layout: $RELEASE_RTX_GPUS GPU(s), host indices $gpu_index_csv"
  echo "  physical GPUs: $gpu_uuid_csv"
  echo "  Spark first routed layer: $spark_first_layer"
  [[ -z "$spark_exl3_identity" ]] || echo "  Spark EXL3 package: $spark_exl3_identity"
  exit 0
fi
if ((restart)); then
  release_stop_services "$coordinator" "$spark_prefix"
else
  docker inspect "$coordinator" >/dev/null 2>&1 && release_die "$coordinator already exists; use --restart"
  for i in 0 1 2 3; do
    remote="${spark_prefix}-${hosts[$i]}-${EXPERT_PORT}"
    ssh -o BatchMode=yes "${hosts[$i]}" "! docker inspect '$remote' >/dev/null 2>&1" || release_die "$remote already exists; use --restart"
  done
fi

ss -ltn "sport = :${ADDR##*:}" 2>/dev/null | tail -n +2 | grep -q . &&
  release_die "API port ${ADDR##*:} is already in use"
for i in 0 1 2 3; do
  ssh -o BatchMode=yes "${hosts[$i]}" \
    "! ss -ltn 'sport = :$EXPERT_PORT' 2>/dev/null | tail -n +2 | grep -q ." ||
    release_die "${hosts[$i]}:$EXPERT_PORT is already in use; stop the development worker first"
done

cleanup() {
  docker rm -f "$coordinator" >/dev/null 2>&1 || true
  for i in 0 1 2 3; do ssh -o BatchMode=yes "${hosts[$i]}" "docker rm -f '${spark_prefix}-${hosts[$i]}-${EXPERT_PORT}' >/dev/null 2>&1 || true" || true; done
}
trap cleanup EXIT

echo "== starting native Spark experts =="
pids=()
for i in 0 1 2 3; do
  host="${hosts[$i]}"; remote="${spark_prefix}-${host}-${EXPERT_PORT}"
  ssh -o BatchMode=yes "$host" bash -s -- "$SPARK_EXPERT_DOCKER_INFERENCE" "$remote" "$i" "$expert_capacity" "$SPARK_DEVICE_BUDGET_BYTES" "$EXPERT_PORT" "$snapshot_rel" "$fingerprint" "$spark_first_layer" <<'REMOTE' &
set -euo pipefail
image="$1"; name="$2"; rank="$3"; capacity="$4"; budget="$5"; port="$6"; snapshot_rel="$7"; fingerprint="$8"; first_layer="$9"
hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
docker run -d --name "$name" --restart no --gpus all --network host --ipc host --ulimit memlock=-1:-1 --device=/dev/infiniband -e "DS41RT_RELEASE_CONFIG_SHA256=$fingerprint" -v "$hf_home:/root/.cache/huggingface:ro" "$image" ds41rt expertd-native --snapshot "/root/.cache/huggingface/$snapshot_rel" --native-lib /opt/ds41rt/lib/libds41rt_native.so --rank "$rank" --capacity "$capacity" --device-budget-bytes "$budget" --first-layer "$first_layer" --listen "0.0.0.0:$port" >/dev/null
REMOTE
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

deadline=$((SECONDS + ${DS41RT_RELEASE_READY_TIMEOUT_SECONDS:-900}))
for i in 0 1 2 3; do
  host="${hosts[$i]}"; remote="${spark_prefix}-${host}-${EXPERT_PORT}"
  until ssh -o BatchMode=yes "$host" "test \"\$(docker inspect -f '{{.State.Status}}' '$remote' 2>/dev/null)\" = running && timeout 1 bash -c '</dev/tcp/127.0.0.1/$EXPERT_PORT'" 2>/dev/null; do
    ((SECONDS < deadline)) || { ssh "$host" "docker logs --tail 100 '$remote'" >&2 || true; release_die "$host native expert did not become ready"; }
    sleep 1
  done
done

echo "== starting native RTX coordinator =="
args=(serve-native --snapshot "/root/.cache/huggingface/$snapshot_rel" --native-lib /opt/ds41rt/lib/libds41rt_native.so --peers "$peers" --rtx-gpus "$RELEASE_RTX_GPUS" --listen "$ADDR" --prefill-batch-tokens "$PREFILL_BATCH_TOKENS" --concurrency "$CONCURRENCY" --prefix-cache-entries "$PREFIX_CACHE_ENTRIES" --max-context-tokens "$MAX_CONTEXT_TOKENS" --max-output-tokens "$MAX_OUTPUT_TOKENS")
[[ -z "$KV_POOL_SIZE" ]] || args+=(--kv-pool-size "$KV_POOL_SIZE")
[[ -z "$MEMORY_RESERVATION" ]] || args+=(--memory-reservation "$MEMORY_RESERVATION")
[[ "$DSPARK" != on ]] || args+=(--dspark)
[[ "$spark_exl3_identity" != paired:* ]] || args+=(--exl3-paired-tp4)
docker run -d --name "$coordinator" --restart no --gpus "$gpu_request" --network host --ipc host --ulimit memlock=-1:-1 --device=/dev/infiniband \
  -e "CUDA_VISIBLE_DEVICES=$gpu_uuid_csv" \
  -e "DS41RT_RELEASE_CONFIG_SHA256=$fingerprint" -e "RUST_LOG=${RUST_LOG:-info}" \
  -v "$hf_home:/root/.cache/huggingface:ro" "$COORDINATOR_DOCKER_INFERENCE" ds41rt "${args[@]}" >/dev/null
api_url="http://127.0.0.1:${ADDR##*:}"
until curl -fsS "$api_url/health" >/dev/null 2>&1 &&
  release_api_advertises_native_model "$api_url" "$RELEASE_MODEL_ID"; do
  [[ "$(docker inspect -f '{{.State.Status}}' "$coordinator" 2>/dev/null)" == running ]] || { docker logs --tail 200 "$coordinator" >&2 || true; release_die "native coordinator exited during startup"; }
  ((SECONDS < deadline)) || { docker logs --tail 200 "$coordinator" >&2 || true; release_die "native API did not become ready"; }
  sleep 1
done
trap - EXIT
echo "DS41RT native API is ready at $api_url/v1/"
echo "  model: $RELEASE_MODEL_ID@$RELEASE_MODEL_REVISION"
echo "  RTX layout: $RELEASE_RTX_GPUS GPU(s), host indices $gpu_index_csv ($gpu_uuid_csv)"
echo "  cache: FP4 compressed source, FP8 SWA, FP4 index"
echo "  concurrency: $CONCURRENCY; retained turns: $PREFIX_CACHE_ENTRIES"
echo "  context/output: $MAX_CONTEXT_TOKENS/$MAX_OUTPUT_TOKENS; dSpark: $DSPARK"
