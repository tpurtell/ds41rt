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
  --rtx-expert-layers auto|N    bottom-up routed layers (dual explicit: 1..40)
  --concurrency N               active requests, 1..16 (default 16)
  --http-queue-depth N          buffered jobs and maximum extra waiters (default concurrency)
  --http-queue-wait-ms N        queue-space wait budget (default 25000)
  --kv-pool-size SIZE           exact KV/index pool, e.g. 22.5GB
  --host-cache-bytes auto|SIZE  pinned RAM cache; 0 disables (default auto)
  --memory-reservation SIZE     total GPU ceiling, e.g. 90% or 80GiB
  --prefix-cache-entries N      turn and prompt retention, 0..128 (default 20)
  --max-context-tokens N        context limit (default 1048576)
  --max-output-tokens N         output limit (default 393216)
  --prefill-batch-tokens N      prefill step, 80..4096 (default 2048)
  --dspark | --no-dspark        enable or disable native dSpark
  --tp2-attention               split attention heads; replicate KV (default off)
  --tp2-query-projection        split query-B projection (default off)
  --tp2-output-projection       split output-B projection (default off)
  --tp2-dspark-experts          split native draft routed experts (default off)
  --no-tp2-<option>             disable the corresponding configured TP2 option
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
    --http-queue-depth) overrides[HTTP_QUEUE_DEPTH]="${2:?$1 requires N}"; shift 2 ;;
    --http-queue-wait-ms) overrides[HTTP_QUEUE_WAIT_MS]="${2:?$1 requires N}"; shift 2 ;;
    --rtx-expert-layers) overrides[RTX_EXPERT_LAYERS]="${2:?$1 requires auto or N}"; shift 2 ;;
    --concurrency) overrides[CONCURRENCY]="${2:?$1 requires N}"; shift 2 ;;
    --host-cache-bytes) overrides[HOST_CACHE_BYTES]="${2:?$1 requires auto or SIZE}"; shift 2 ;;
    --kv-pool-size) overrides[KV_POOL_SIZE]="${2:?$1 requires SIZE}"; shift 2 ;;
    --memory-reservation) overrides[MEMORY_RESERVATION]="${2:?$1 requires SIZE}"; shift 2 ;;
    --prefix-cache-entries) overrides[PREFIX_CACHE_ENTRIES]="${2:?$1 requires N}"; shift 2 ;;
    --max-context-tokens) overrides[MAX_CONTEXT_TOKENS]="${2:?$1 requires N}"; shift 2 ;;
    --max-output-tokens) overrides[MAX_OUTPUT_TOKENS]="${2:?$1 requires N}"; shift 2 ;;
    --prefill-batch-tokens) overrides[PREFILL_BATCH_TOKENS]="${2:?$1 requires N}"; shift 2 ;;
    --dspark) overrides[DSPARK]=on; shift ;;
    --no-dspark) overrides[DSPARK]=off; shift ;;
    --tp2-attention) overrides[TP2_ATTENTION]=on; shift ;;
    --no-tp2-attention) overrides[TP2_ATTENTION]=off; shift ;;
    --tp2-query-projection) overrides[TP2_QUERY_PROJECTION]=on; shift ;;
    --no-tp2-query-projection) overrides[TP2_QUERY_PROJECTION]=off; shift ;;
    --tp2-output-projection) overrides[TP2_OUTPUT_PROJECTION]=on; shift ;;
    --no-tp2-output-projection) overrides[TP2_OUTPUT_PROJECTION]=off; shift ;;
    --tp2-dspark-experts) overrides[TP2_DSPARK_EXPERTS]=on; shift ;;
    --no-tp2-dspark-experts) overrides[TP2_DSPARK_EXPERTS]=off; shift ;;
    --restart) restart=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) release_die "unknown run argument: $1" ;;
  esac
done

release_load_config "$config"
for name in "${!overrides[@]}"; do printf -v "$name" '%s' "${overrides[$name]}"; done
release_validate_tp2_options
[[ -z "$HTTP_QUEUE_DEPTH" || ( "$HTTP_QUEUE_DEPTH" =~ ^[1-9][0-9]*$ && "$HTTP_QUEUE_DEPTH" -le 4096 ) ]] || release_die "HTTP_QUEUE_DEPTH must be in 1..4096"
[[ "$HTTP_QUEUE_WAIT_MS" =~ ^[0-9]+$ ]] || release_die "HTTP_QUEUE_WAIT_MS must be non-negative"
[[ "$HOST_CACHE_BYTES" == auto || "$HOST_CACHE_BYTES" =~ ^[0-9]+([.][0-9]{1,6})?(B|MB|GB|MiB|GiB)?$ ]] || release_die "HOST_CACHE_BYTES must be auto, 0, or a byte size"
case "$DSPARK" in on|off) ;; *) release_die "DSPARK must be on or off" ;; esac
case "$RTX_GPUS" in auto|1|2) ;; *) release_die "RTX_GPUS must be auto, 1, or 2" ;; esac
[[ "$RTX_EXPERT_LAYERS" == auto || "$RTX_EXPERT_LAYERS" =~ ^([0-9]|[1-3][0-9]|40)$ ]] || release_die "RTX_EXPERT_LAYERS must be auto or 0..40"
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
# Multi-family EXL3 images ship exl3-kXX packages per decoder-tier family.
# The deployed checkpoint's resident tiers are [floor(bits), floor(bits)+1]
# for both raw integer-bit publications and staged fractional-bit snapshots.
exl3_family_tag=""
if [[ "$model_is_exl3" == true ]]; then
  # Family resolution is best-effort: without readable bits the remote
  # manifest read falls back to the legacy single-package location.
  exl3_bits="$(jq -er '.bits' "$hf_home/$snapshot_rel/quantize_config.json" 2>/dev/null || jq -er '.quantization_config.bits' "$hf_home/$snapshot_rel/config.json" 2>/dev/null || true)"
  if [[ "$exl3_bits" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    exl3_family_base="${exl3_bits%%.*}"
    exl3_family_tag="k${exl3_family_base}$((exl3_family_base + 1))"
  fi
fi
coordinator="$RELEASE_COORDINATOR_CONTAINER_NAME"
[[ "$TP2_DSPARK_EXPERTS" != on || "$DSPARK" == on ]] || release_die "TP2_DSPARK_EXPERTS requires dSpark"
[[ "$TP2_DSPARK_EXPERTS" != on || "$model_is_exl3" != true ]] || release_die "TP2_DSPARK_EXPERTS requires native expert weights"

reclaim_args=()
if ((restart)) && docker container inspect "$coordinator" >/dev/null 2>&1; then
  while IFS= read -r pid; do
    [[ "$pid" =~ ^[0-9]+$ ]] && reclaim_args+=(--reclaim-pid "$pid")
  done < <(docker top "$coordinator" -eo pid 2>/dev/null | tail -n +2 || true)
fi
minimum_expert_layers=1
[[ "$RTX_EXPERT_LAYERS" == auto || "$RTX_EXPERT_LAYERS" == 0 ]] || minimum_expert_layers="$RTX_EXPERT_LAYERS"
expert_format=native
if [[ "$model_is_exl3" == true ]]; then
  case "$exl3_family_tag" in
    k23) expert_format=exl3-k23 ;;
    k34) expert_format=exl3-k34 ;;
    # Unresolvable families keep the conservative native constants; the
    # dual fit model is only exercised for two-RTX launches.
    *) expert_format=native ;;
  esac
fi
gpu_selection="$(python3 "$repo_root/scripts/select-release-gpus.py" \
  --mode "$RTX_GPUS" \
  --minimum-expert-layers "$minimum_expert_layers" \
  --primary-uuid "$RELEASE_COORDINATOR_GPU_UUID" \
  --concurrency "$CONCURRENCY" \
  --max-context-tokens "$MAX_CONTEXT_TOKENS" \
  --retained-turns "$PREFIX_CACHE_ENTRIES" \
  --kv-pool-size "$KV_POOL_SIZE" \
  --memory-reservation "$MEMORY_RESERVATION" \
  --expert-format "$expert_format" \
  "${reclaim_args[@]}")"
RELEASE_RTX_GPUS="$(jq -r '.count' <<<"$gpu_selection")"
if release_tp2_enabled; then
  ((RELEASE_RTX_GPUS == 2)) || release_die "TP2 options require two selected RTX GPUs; use --rtx-gpus 2"
fi
mapfile -t release_gpu_uuids < <(jq -r '.gpus[].uuid' <<<"$gpu_selection")
mapfile -t release_gpu_indices < <(jq -r '.gpus[].index' <<<"$gpu_selection")
mapfile -t release_gpu_pci < <(jq -r '.gpus[].pci' <<<"$gpu_selection")
gpu_uuid_csv="$(IFS=,; echo "${release_gpu_uuids[*]}")"
gpu_index_csv="$(IFS=,; echo "${release_gpu_indices[*]}")"
gpu_pci_csv="$(IFS=,; echo "${release_gpu_pci[*]}")"
spark_first_layer="$(release_spark_first_layer "$RELEASE_RTX_GPUS" "$RTX_EXPERT_LAYERS")"
if ((RELEASE_RTX_GPUS == 2)) && [[ "$RTX_EXPERT_LAYERS" == auto ]]; then spark_first_layer=runtime-plan; fi
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

hosts=()
lanes=()
if ((SPARK_COUNT > 0)); then
  hosts=("$SPARK_0_HOST" "$SPARK_1_HOST" "$SPARK_2_HOST" "$SPARK_3_HOST")
  lanes=("$SPARK_0_LANE_A" "$SPARK_1_LANE_A" "$SPARK_2_LANE_A" "$SPARK_3_LANE_A")
fi
spark_exl3_identity=""
for host in "${hosts[@]}"; do
  spark_manifest="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" bash -s -- "$SPARK_EXPERT_DOCKER_INFERENCE" "$engine_commit" "$sparkinfer_commit" "$snapshot_rel" "$model_is_exl3" "$exl3_family_tag" <<'REMOTE'
set -euo pipefail
image="$1"; engine="$2"; sparkinfer="$3"; snapshot_rel="$4"
docker info >/dev/null
test "$(docker image inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$image")" = "$engine"
test "$(docker image inspect -f '{{index .Config.Labels "io.ds41rt.sparkinfer.revision"}}' "$image")" = "$sparkinfer"
hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
test -d "$hf_home/$snapshot_rel"
! find "$hf_home/$snapshot_rel" -xtype l -print -quit | grep -q .
if [[ "$5" == true ]]; then
  docker run --rm --network none --entrypoint /bin/sh "$image" -c \
    'if [ -f "/opt/ds41rt/lib/exl3/exl3-'"$6"'/manifest.json" ]; then cat "/opt/ds41rt/lib/exl3/exl3-'"$6"'/manifest.json"; else cat /opt/ds41rt/lib/exl3/manifest.json; fi'
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
# Zero-Spark deployments hold every routed layer on the RTX pair; the daemon
# still requires four peer addresses but never connects to them.
if ((SPARK_COUNT == 0)); then
  peers="127.0.0.1:1,127.0.0.1:2,127.0.0.1:3,127.0.0.1:4"
else
  peers="${lanes[0]}:$EXPERT_PORT,${lanes[1]}:$EXPERT_PORT,${lanes[2]}:$EXPERT_PORT,${lanes[3]}:$EXPERT_PORT"
fi
fingerprint="$(printf '%s\n' "$engine_commit" "$RELEASE_MODEL_ID" "$RELEASE_MODEL_REVISION" "$ADDR" "$RELEASE_RTX_GPUS" "$gpu_uuid_csv" "$gpu_pci_csv" "$CONCURRENCY" "${HTTP_QUEUE_DEPTH:-$CONCURRENCY}" "$HTTP_QUEUE_WAIT_MS" "$HOST_CACHE_BYTES" "$RTX_EXPERT_LAYERS" "$KV_POOL_SIZE" "$MEMORY_RESERVATION" "$PREFIX_CACHE_ENTRIES" "$MAX_CONTEXT_TOKENS" "$MAX_OUTPUT_TOKENS" "$PREFILL_BATCH_TOKENS" "$DSPARK" "$TP2_ATTENTION" "$TP2_QUERY_PROJECTION" "$TP2_OUTPUT_PROJECTION" "$TP2_DSPARK_EXPERTS" "$SPARK_DEVICE_BUDGET_BYTES" "$spark_first_layer" "$peers" "$spark_exl3_identity" | sha256sum | awk '{print $1}')"
spark_prefix="$RELEASE_SPARK_CONTAINER_PREFIX"

if ((dry_run)); then
  echo "Dry-run checks passed for native V4.1; no services changed."
  echo "  RTX layout: $RELEASE_RTX_GPUS GPU(s), host indices $gpu_index_csv"
  echo "  physical GPUs: $gpu_uuid_csv"
  echo "  TP2 attention/query/output/draft experts: $TP2_ATTENTION/$TP2_QUERY_PROJECTION/$TP2_OUTPUT_PROJECTION/$TP2_DSPARK_EXPERTS"
  echo "  Spark first routed layer: $spark_first_layer"
  [[ -z "$spark_exl3_identity" ]] || echo "  Spark EXL3 package: $spark_exl3_identity"
  exit 0
fi
if ((restart)); then
  release_stop_services "$coordinator" "$spark_prefix"
else
  docker inspect "$coordinator" >/dev/null 2>&1 && release_die "$coordinator already exists; use --restart"
  for i in "${!hosts[@]}"; do
    remote="${spark_prefix}-${hosts[$i]}-${EXPERT_PORT}"
    ssh -o BatchMode=yes "${hosts[$i]}" "! docker inspect '$remote' >/dev/null 2>&1" || release_die "$remote already exists; use --restart"
  done
fi

ss -ltn "sport = :${ADDR##*:}" 2>/dev/null | tail -n +2 | grep -q . &&
  release_die "API port ${ADDR##*:} is already in use"
for i in "${!hosts[@]}"; do
  ssh -o BatchMode=yes "${hosts[$i]}" \
    "! ss -ltn 'sport = :$EXPERT_PORT' 2>/dev/null | tail -n +2 | grep -q ." ||
    release_die "${hosts[$i]}:$EXPERT_PORT is already in use; stop the development worker first"
done

cleanup() {
  docker rm -f "$coordinator" >/dev/null 2>&1 || true
  for i in "${!hosts[@]}"; do ssh -o BatchMode=yes "${hosts[$i]}" "docker rm -f '${spark_prefix}-${hosts[$i]}-${EXPERT_PORT}' >/dev/null 2>&1 || true" || true; done
}
trap cleanup EXIT

placement_directory=
((RELEASE_RTX_GPUS != 2)) || placement_directory=/run/ds41rt-placement
start_coordinator() {
echo "== starting native RTX coordinator =="
local -a args=(serve-native --snapshot "/root/.cache/huggingface/$snapshot_rel" --native-lib /opt/ds41rt/lib/libds41rt_native.so --peers "$peers" --rtx-gpus "$RELEASE_RTX_GPUS" --listen "$ADDR" --prefill-batch-tokens "$PREFILL_BATCH_TOKENS" --concurrency "$CONCURRENCY" --prefix-cache-entries "$PREFIX_CACHE_ENTRIES" --max-context-tokens "$MAX_CONTEXT_TOKENS" --max-output-tokens "$MAX_OUTPUT_TOKENS")
args+=(--http-queue-depth "${HTTP_QUEUE_DEPTH:-$CONCURRENCY}" --http-queue-wait-ms "$HTTP_QUEUE_WAIT_MS")
[[ "$RTX_EXPERT_LAYERS" == auto ]] || args+=(--rtx-expert-layers "$RTX_EXPERT_LAYERS")
[[ "$HOST_CACHE_BYTES" == 0 ]] || args+=(--host-cache-bytes "$HOST_CACHE_BYTES")
[[ -z "$KV_POOL_SIZE" ]] || args+=(--kv-pool-size "$KV_POOL_SIZE")
[[ -z "$MEMORY_RESERVATION" ]] || args+=(--memory-reservation "$MEMORY_RESERVATION")
[[ "$DSPARK" != on ]] || args+=(--dspark)
[[ "$TP2_ATTENTION" != on ]] || args+=(--tp2-attention)
[[ "$TP2_QUERY_PROJECTION" != on ]] || args+=(--tp2-query-projection)
[[ "$TP2_OUTPUT_PROJECTION" != on ]] || args+=(--tp2-output-projection)
[[ "$TP2_DSPARK_EXPERTS" != on ]] || args+=(--tp2-dspark-experts)
[[ "$spark_exl3_identity" != paired:* ]] || args+=(--exl3-paired-tp4)
[[ -z "$placement_directory" ]] || args+=(--placement-directory "$placement_directory")
docker run -d --name "$coordinator" --restart no --gpus "$gpu_request" --network host --ipc host --ulimit memlock=-1:-1 --device=/dev/infiniband \
  -e "CUDA_VISIBLE_DEVICES=$gpu_uuid_csv" \
  -e "DS41RT_RELEASE_CONFIG_SHA256=$fingerprint" -e "RUST_LOG=${RUST_LOG:-info}" \
  -v "$hf_home:/root/.cache/huggingface:ro" "$COORDINATOR_DOCKER_INFERENCE" ds41rt "${args[@]}" >/dev/null
}
deadline=$((SECONDS + ${DS41RT_RELEASE_READY_TIMEOUT_SECONDS:-900}))
if ((RELEASE_RTX_GPUS == 2)); then
  start_coordinator
  until placement_plan="$(docker exec "$coordinator" cat "$placement_directory/plan.json" 2>/dev/null)"; do
    [[ "$(docker inspect -f '{{.State.Status}}' "$coordinator" 2>/dev/null)" == running ]] || { docker logs --tail 100 "$coordinator" >&2 || true; release_die "coordinator exited before placement publication"; }
    ((SECONDS < deadline)) || release_die "timed out waiting for coordinator placement"
    sleep 1
  done
  spark_first_layer="$(jq -er '
    select(.version == 1 and .rtx_gpus == 2)
    | select((.nonce | type) == "string" and (.nonce | length) > 0)
    | select((.rtx_expert_layers | type) == "number")
    | select(.rtx_expert_layers == (.rtx_expert_layers | floor) and .rtx_expert_layers >= 1 and .rtx_expert_layers <= 40)
    | select(.spark_first_layer == ([.rtx_expert_layers, 39] | min))
    | .spark_first_layer' <<<"$placement_plan")" || release_die "invalid coordinator placement plan"
  echo "  runtime placement: RTX layers $(jq -r '.rtx_expert_layers' <<<"$placement_plan"); Spark first layer $spark_first_layer"
fi

echo "== starting native Spark experts =="
pids=()
for i in "${!hosts[@]}"; do
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

for i in "${!hosts[@]}"; do
  host="${hosts[$i]}"; remote="${spark_prefix}-${host}-${EXPERT_PORT}"
  until ssh -o BatchMode=yes "$host" "test \"\$(docker inspect -f '{{.State.Status}}' '$remote' 2>/dev/null)\" = running && timeout 1 bash -c '</dev/tcp/127.0.0.1/$EXPERT_PORT'" 2>/dev/null; do
    ((SECONDS < deadline)) || { ssh "$host" "docker logs --tail 100 '$remote'" >&2 || true; release_die "$host native expert did not become ready"; }
    sleep 1
  done
done

if ((RELEASE_RTX_GPUS == 2)); then
  docker exec "$coordinator" sh -c 'cp "$1/plan.json" "$1/.ready-pending" && mv "$1/.ready-pending" "$1/ready.json"' sh "$placement_directory"
else
  start_coordinator
fi
api_url="http://127.0.0.1:${ADDR##*:}"
until curl -fsS "$api_url/health" >/dev/null 2>&1 &&
  release_api_advertises_native_model "$api_url" "$RELEASE_NATIVE_API_MODEL_ID"; do
  [[ "$(docker inspect -f '{{.State.Status}}' "$coordinator" 2>/dev/null)" == running ]] || { docker logs --tail 200 "$coordinator" >&2 || true; release_die "native coordinator exited during startup"; }
  ((SECONDS < deadline)) || { docker logs --tail 200 "$coordinator" >&2 || true; release_die "native API did not become ready"; }
  sleep 1
done
trap - EXIT
echo "DS41RT native API is ready at $api_url/v1/"
echo "  API model: $RELEASE_NATIVE_API_MODEL_ID"
echo "  checkpoint: $RELEASE_MODEL_ID@$RELEASE_MODEL_REVISION"
echo "  RTX layout: $RELEASE_RTX_GPUS GPU(s), host indices $gpu_index_csv ($gpu_uuid_csv)"
echo "  cache: FP4 compressed source, FP8 SWA, FP4 index"
echo "  concurrency: $CONCURRENCY; retained turns: $PREFIX_CACHE_ENTRIES"
echo "  context/output: $MAX_CONTEXT_TOKENS/$MAX_OUTPUT_TOKENS; dSpark: $DSPARK"
