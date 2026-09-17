#!/usr/bin/env bash

# The native API uses the recipe model identity for every supported checkpoint.
# MODEL_ID selects Hugging Face storage and may name a routed-only quant.
RELEASE_NATIVE_API_MODEL_ID=deepseek-ai/DeepSeek-V4.1-Flash
RELEASE_COORDINATOR_CONTAINER_NAME=ds41rt-coordinator
RELEASE_SPARK_CONTAINER_PREFIX=ds41rt-spark-expert

release_die() {
  echo "ds41rt release: $*" >&2
  exit 2
}

release_need() {
  command -v "$1" >/dev/null 2>&1 || release_die "required command not found: $1"
}

# Inference images verify package payload hashes when built. Compare their
# immutable manifests before changing services so all peers select one layout.
release_exl3_package_identity() {
  local revision="$1" manifest layout digest
  manifest="$(cat)"
  layout="$(jq -er --arg revision "$revision" '
    if .schema == "ds41rt.exl3-package.v1" and .role == "spark"
      and .sparkinfer_revision == $revision
      and ((has("paired_tp4") | not) or (.paired_tp4 | type) == "boolean")
    then (if .paired_tp4 == true then "paired" else "disjoint" end)
    else error("invalid Spark EXL3 package identity") end
  ' <<<"$manifest")" || release_die "invalid Spark EXL3 package identity"
  digest="$(printf '%s' "$manifest" | sha256sum | awk '{print $1}')"
  printf '%s:%s\n' "$layout" "$digest"
}

release_model_list_matches() {
  local model_id="$1"
  local full_model_id="${model_id}-full"
  jq -e \
    --arg model_id "$model_id" \
    --arg full_model_id "$full_model_id" \
    '.object == "list"
      and (.data | type == "array")
      and ([.data[]? | select(.id == $model_id)] | length == 1)
      and ([.data[]? | select(.id == $full_model_id)] | length == 1)' \
    >/dev/null 2>&1
}

release_api_advertises_model() {
  local url="$1"
  local model_id="$2"
  curl -fsS "$url/v1/models" 2>/dev/null |
    release_model_list_matches "$model_id"
}

release_native_model_list_matches() {
  local model_id="$1"
  jq -e \
    --arg model_id "$model_id" \
    '.object == "list"
      and (.data | type == "array")
      and ([.data[]? | select(.id == $model_id)] | length == 1)' \
    >/dev/null 2>&1
}

release_api_advertises_native_model() {
  local url="$1"
  local model_id="$2"
  curl -fsS "$url/v1/models" 2>/dev/null |
    release_native_model_list_matches "$model_id"
}

release_validate_model_list_file() {
  local path="$1"
  local model_id="$2"
  release_model_list_matches "$model_id" <"$path" ||
    release_die "API model list does not advertise exact configured identities: $model_id and ${model_id}-full"
}

release_trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

release_known_key() {
  case "$1" in
    EXL3_PAIRED_TP4) return 0 ;;
    HTTP_QUEUE_DEPTH|HTTP_QUEUE_WAIT_MS|MODEL_ID|MODEL_VARIANT|MODEL_REVISION|EXPERT_FORMAT|DSPARK|DSPARK_DRAFT_POLICY|RTX_GPUS|COORDINATOR_GPU|COORDINATOR_GPU_UUID|COORDINATOR_GPU_PCI_BUS_ID|COORDINATOR_GPU_HEADROOM_GIB|KV_POOL_TOKENS|KV_POOL_SIZE|HOST_CACHE_BYTES|MEMORY_RESERVATION|MAX_CONTEXT_TOKENS|MAX_OUTPUT_TOKENS|CONCURRENCY|PREFIX_CACHE_ENTRIES|PREFILL_BATCH_TOKENS|SPARK_DEVICE_BUDGET_BYTES|SPARK_REDUCTION_MIN_ROWS|SPARKINFER_EXL3|ADDR|EXPERT_PORT|SPARK_[0-3]_HOST|SPARK_[0-3]_LANE_A|SPARK_[0-3]_LANE_B|COORDINATOR_DOCKER_DEV|COORDINATOR_DOCKER_INFERENCE|SPARK_EXPERT_DOCKER_DEV|SPARK_EXPERT_DOCKER_INFERENCE)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

release_load_config() {
  local config="$1"
  [[ -f "$config" ]] || release_die "configuration file not found: $config"

  local default_model_id=deepseek-ai/DeepSeek-V4.1-Flash
  MODEL_ID="$default_model_id"
  MODEL_VARIANT=flash
  MODEL_REVISION=dba1be0a40aa45a94ad051997016db3960a90277
  EXPERT_FORMAT=native
  DSPARK=on
  DSPARK_DRAFT_POLICY=adaptive
  RTX_GPUS=auto
  COORDINATOR_GPU=0
  COORDINATOR_GPU_UUID=
  COORDINATOR_GPU_PCI_BUS_ID=
  COORDINATOR_GPU_HEADROOM_GIB=8
  KV_POOL_TOKENS=
  KV_POOL_SIZE=
  HOST_CACHE_BYTES=0
  MEMORY_RESERVATION=
  MAX_CONTEXT_TOKENS=1048576
  MAX_OUTPUT_TOKENS=393216
  HTTP_QUEUE_DEPTH=
  HTTP_QUEUE_WAIT_MS=25000
  CONCURRENCY=16
  PREFIX_CACHE_ENTRIES=20
  PREFILL_BATCH_TOKENS=2048
  SPARK_DEVICE_BUDGET_BYTES=107374182400
  SPARK_REDUCTION_MIN_ROWS=16
  SPARKINFER_EXL3=disable
  EXL3_PAIRED_TP4=off
  ADDR=0.0.0.0:8000
  EXPERT_PORT=19441
  COORDINATOR_DOCKER_DEV=ds41rt-coordinator-dev
  COORDINATOR_DOCKER_INFERENCE=ds41rt-coordinator
  SPARK_EXPERT_DOCKER_DEV=ds41rt-spark-expert-dev
  SPARK_EXPERT_DOCKER_INFERENCE=ds41rt-spark-expert
  for release_i in 0 1 2 3; do
    printf -v "SPARK_${release_i}_HOST" '%s' ""
    printf -v "SPARK_${release_i}_LANE_A" '%s' ""
    printf -v "SPARK_${release_i}_LANE_B" '%s' ""
  done

  local raw line key value model_id_explicit=0 model_revision_explicit=0
  while IFS= read -r raw || [[ -n "$raw" ]]; do
    line="$(release_trim "${raw%%#*}")"
    [[ -n "$line" ]] || continue
    [[ "$line" == *=* ]] || release_die "invalid configuration line: $raw"
    key="$(release_trim "${line%%=*}")"
    value="$(release_trim "${line#*=}")"
    release_known_key "$key" || release_die "unknown configuration key: $key"
    if [[ "$value" == \"*\" && "$value" == *\" ]]; then
      value="${value:1:${#value}-2}"
    elif [[ "$value" == \'*\' && "$value" == *\' ]]; then
      value="${value:1:${#value}-2}"
    elif [[ "$value" == *[[:space:]]* ]]; then
      release_die "unquoted whitespace is not allowed for $key"
    fi
    printf -v "$key" '%s' "$value"
    [[ "$key" != MODEL_ID ]] || model_id_explicit=1
    [[ "$key" != MODEL_REVISION ]] || model_revision_explicit=1
  done <"$config"

  # A model override without its own revision must never inherit the pinned
  # calibrated-release commit from the defaults.
  if ((model_id_explicit && !model_revision_explicit)) &&
    [[ "$MODEL_ID" != "$default_model_id" ]]; then
    MODEL_REVISION=
  fi

  case "$MODEL_VARIANT" in flash|pro) ;; *) release_die "MODEL_VARIANT must be flash or pro" ;; esac
  case "$EXPERT_FORMAT" in native|exl3) ;; *) release_die "EXPERT_FORMAT must be native or exl3" ;; esac
  case "$EXL3_PAIRED_TP4" in on|off) ;; *) release_die "EXL3_PAIRED_TP4 must be on or off" ;; esac
  [[ "$MODEL_VARIANT" != pro || "$EXPERT_FORMAT" == exl3 ]] ||
    release_die "DeepSeek V4 Pro requires EXPERT_FORMAT=exl3"
  case "$DSPARK" in on|off) ;; *) release_die "DSPARK must be on or off" ;; esac
  case "$DSPARK_DRAFT_POLICY" in
    full|adaptive) ;;
    *) release_die "DSPARK_DRAFT_POLICY must be full or adaptive" ;;
  esac
  case "$SPARKINFER_EXL3" in
    auto|disable|force) ;;
    *) release_die "SPARKINFER_EXL3 must be auto, disable, or force" ;;
  esac
  [[ "$SPARKINFER_EXL3" != disable || "$EXPERT_FORMAT" == native ]] ||
    release_die "SPARKINFER_EXL3=disable requires EXPERT_FORMAT=native"
  [[ "$SPARKINFER_EXL3" != force || "$EXPERT_FORMAT" == exl3 ]] ||
    release_die "SPARKINFER_EXL3=force requires EXPERT_FORMAT=exl3"
  case "$RTX_GPUS" in auto|1|2) ;; *) release_die "RTX_GPUS must be auto, 1, or 2" ;; esac
  [[ "$COORDINATOR_GPU" =~ ^[0-9]+$ ]] || release_die "COORDINATOR_GPU must be a non-negative host GPU index"
  [[ -z "$COORDINATOR_GPU_UUID" || "$COORDINATOR_GPU_UUID" =~ ^GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$ ]] ||
    release_die "COORDINATOR_GPU_UUID must be empty or a physical NVIDIA GPU UUID"
  [[ -z "$COORDINATOR_GPU_PCI_BUS_ID" || "$COORDINATOR_GPU_PCI_BUS_ID" =~ ^[0-9A-Fa-f]{8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]$ ]] ||
    release_die "COORDINATOR_GPU_PCI_BUS_ID must be empty or a full PCI bus ID"
  [[ -z "$COORDINATOR_GPU_UUID" && -z "$COORDINATOR_GPU_PCI_BUS_ID" ]] ||
    [[ -n "$COORDINATOR_GPU_UUID" && -n "$COORDINATOR_GPU_PCI_BUS_ID" ]] ||
    release_die "COORDINATOR_GPU_UUID and COORDINATOR_GPU_PCI_BUS_ID must be set together"
  [[ "$CONCURRENCY" =~ ^([1-9]|1[0-6])$ ]] || release_die "CONCURRENCY must be in 1..16"
  [[ "$PREFIX_CACHE_ENTRIES" =~ ^([0-9]|[1-9][0-9]|1[01][0-9]|12[0-8])$ ]] ||
    release_die "PREFIX_CACHE_ENTRIES must be in 0..128"
  [[ "$PREFILL_BATCH_TOKENS" =~ ^[0-9]+$ ]] &&
    ((PREFILL_BATCH_TOKENS >= 80 && PREFILL_BATCH_TOKENS <= 4096)) ||
    release_die "PREFILL_BATCH_TOKENS must be in 80..4096"
  [[ "$SPARK_DEVICE_BUDGET_BYTES" =~ ^[1-9][0-9]*$ ]] ||
    release_die "SPARK_DEVICE_BUDGET_BYTES must be a positive integer"
  [[ "$SPARK_REDUCTION_MIN_ROWS" =~ ^[1-9][0-9]*$ ]] ||
    release_die "SPARK_REDUCTION_MIN_ROWS must be a positive integer"
  [[ "$EXPERT_PORT" =~ ^[0-9]+$ ]] && ((EXPERT_PORT >= 1 && EXPERT_PORT <= 65535)) || release_die "EXPERT_PORT must be in 1..65535"
  [[ "$COORDINATOR_GPU_HEADROOM_GIB" =~ ^[0-9]+([.][0-9]+)?$ ]] || release_die "COORDINATOR_GPU_HEADROOM_GIB must be non-negative"
  for release_integer_name in KV_POOL_TOKENS MAX_CONTEXT_TOKENS MAX_OUTPUT_TOKENS; do
    value="${!release_integer_name}"
    [[ -z "$value" || "$value" =~ ^[1-9][0-9]*$ ]] || release_die "$release_integer_name must be a positive integer"
  done
  [[ -z "$MAX_CONTEXT_TOKENS" ]] || ((MAX_CONTEXT_TOKENS <= 1048576)) ||
    release_die "MAX_CONTEXT_TOKENS must be in 1..1048576"
  [[ -z "$MAX_OUTPUT_TOKENS" ]] || ((MAX_OUTPUT_TOKENS <= 393216)) ||
    release_die "MAX_OUTPUT_TOKENS must be in 1..393216"
  if [[ -n "$KV_POOL_TOKENS" ]]; then
    ((KV_POOL_TOKENS % 64 == 0)) || release_die "KV_POOL_TOKENS must be a multiple of 64"
  fi
  [[ -z "$KV_POOL_SIZE" || "$KV_POOL_SIZE" =~ ^[0-9]+([.][0-9]{1,6})?(B|MB|GB|MiB|GiB)?$ ]] ||
    release_die "KV_POOL_SIZE must use B, MB, GB, MiB or GiB"
  [[ "$HOST_CACHE_BYTES" == auto || "$HOST_CACHE_BYTES" =~ ^[0-9]+([.][0-9]{1,6})?(B|MB|GB|MiB|GiB)?$ ]] ||
    release_die "HOST_CACHE_BYTES must be auto, 0, or a byte size"
  [[ -z "$MEMORY_RESERVATION" || "$MEMORY_RESERVATION" =~ ^[0-9]+([.][0-9]{1,6})?((B|MB|GB|MiB|GiB)|%)$ ]] ||
    release_die "MEMORY_RESERVATION must be a byte size or percentage"
  [[ "$ADDR" == *:* ]] || release_die "ADDR must be HOST:PORT"
  [[ -n "$MODEL_ID" && "$MODEL_ID" == */* && "$MODEL_ID" != *[[:space:]]* ]] ||
    release_die "MODEL_ID must be a Hugging Face repository ID"
  [[ -z "$MODEL_REVISION" || "$MODEL_REVISION" =~ ^[0-9a-f]{40,64}$ ]] ||
    release_die "MODEL_REVISION must be empty or a 40..64 lowercase hex revision"

  local missing_b=0 present_b=0
  for release_i in 0 1 2 3; do
    local host_name="SPARK_${release_i}_HOST"
    local lane_a_name="SPARK_${release_i}_LANE_A"
    local lane_b_name="SPARK_${release_i}_LANE_B"
    [[ -n "${!host_name}" ]] || release_die "$host_name must not be empty"
    [[ "${!lane_a_name}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || release_die "$lane_a_name must be an IPv4 address"
    if [[ -n "${!lane_b_name}" ]]; then
      ((present_b += 1))
    else
      ((missing_b += 1))
    fi
  done
  ((present_b == 0 || missing_b == 0)) || release_die "secondary Spark rail must provide all four LANE_B values or none"

  for release_image_name in COORDINATOR_DOCKER_DEV COORDINATOR_DOCKER_INFERENCE SPARK_EXPERT_DOCKER_DEV SPARK_EXPERT_DOCKER_INFERENCE; do
    [[ -n "${!release_image_name}" && "${!release_image_name}" != *[[:space:]]* ]] || release_die "$release_image_name must be a Docker image reference"
  done

  RELEASE_CONFIG="$(realpath "$config")"
  RELEASE_MODEL_ID="$MODEL_ID"
  RELEASE_MODEL_REVISION="$MODEL_REVISION"
}

release_resolve_coordinator_gpu_identity() {
  local selector gpu_line observed_index observed_uuid observed_pci
  selector="${COORDINATOR_GPU_PCI_BUS_ID:-$COORDINATOR_GPU}"
  gpu_line="$(
    nvidia-smi --id="$selector" \
      --query-gpu=index,uuid,pci.bus_id \
      --format=csv,noheader,nounits
  )" || release_die "cannot resolve coordinator GPU selector: $selector"
  [[ -n "$gpu_line" && "$gpu_line" != *$'\n'* ]] ||
    release_die "coordinator GPU selector did not resolve to exactly one device: $selector"
  IFS=, read -r observed_index observed_uuid observed_pci <<<"$gpu_line"
  observed_index="$(release_trim "$observed_index")"
  observed_uuid="$(release_trim "$observed_uuid")"
  observed_pci="$(release_trim "$observed_pci")"
  [[ "$observed_index" =~ ^[0-9]+$ ]] ||
    release_die "coordinator GPU has an invalid host index: $observed_index"
  [[ "$observed_uuid" =~ ^GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$ ]] ||
    release_die "coordinator GPU has an invalid physical UUID: $observed_uuid"
  [[ "$observed_pci" =~ ^[0-9A-Fa-f]{8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]$ ]] ||
    release_die "coordinator GPU has an invalid PCI identity: $observed_pci"
  if [[ -n "$COORDINATOR_GPU_UUID" ]]; then
    [[ "$observed_uuid" == "$COORDINATOR_GPU_UUID" && "$observed_pci" == "$COORDINATOR_GPU_PCI_BUS_ID" ]] ||
      release_die "coordinator GPU PCI selector resolved to another physical device"
  else
    [[ "$observed_index" == "$COORDINATOR_GPU" ]] ||
      release_die "coordinator GPU ordinal resolved to an unexpected device"
  fi
  COORDINATOR_GPU_UUID="$observed_uuid"
  COORDINATOR_GPU_PCI_BUS_ID="$observed_pci"
  RELEASE_COORDINATOR_GPU_UUID="$observed_uuid"
  RELEASE_COORDINATOR_GPU_HOST_INDEX="$observed_index"
  RELEASE_COORDINATOR_GPU_PCI_BUS_ID="$observed_pci"
}

release_resolve_local_model_revision() {
  local hf_home="$1"
  local model_root="$hf_home/hub/models--${RELEASE_MODEL_ID//\//--}"
  if [[ -z "$RELEASE_MODEL_REVISION" ]]; then
    [[ -s "$model_root/refs/main" ]] ||
      release_die "MODEL_REVISION is empty and $RELEASE_MODEL_ID has no local refs/main"
    RELEASE_MODEL_REVISION="$(<"$model_root/refs/main")"
  fi
  [[ "$RELEASE_MODEL_REVISION" =~ ^[0-9a-f]{40,64}$ ]] ||
    release_die "resolved model revision is not 40..64 lowercase hex: $RELEASE_MODEL_REVISION"
  [[ -d "$model_root/snapshots/$RELEASE_MODEL_REVISION" ]] ||
    release_die "model snapshot is missing: $RELEASE_MODEL_ID@$RELEASE_MODEL_REVISION"
}

release_hosts_csv() {
  printf '%s,%s,%s,%s' "$SPARK_0_HOST" "$SPARK_1_HOST" "$SPARK_2_HOST" "$SPARK_3_HOST"
}

release_lane_a_csv() {
  printf '%s,%s,%s,%s' "$SPARK_0_LANE_A" "$SPARK_1_LANE_A" "$SPARK_2_LANE_A" "$SPARK_3_LANE_A"
}

release_lane_b_csv() {
  if [[ -z "$SPARK_0_LANE_B" ]]; then
    return
  fi
  printf '%s,%s,%s,%s' "$SPARK_0_LANE_B" "$SPARK_1_LANE_B" "$SPARK_2_LANE_B" "$SPARK_3_LANE_B"
}

release_expert_hosts_csv() {
  printf '%s=%s:%s,%s=%s:%s,%s=%s:%s,%s=%s:%s' \
    "spark-0" "$SPARK_0_LANE_A" "$EXPERT_PORT" \
    "spark-1" "$SPARK_1_LANE_A" "$EXPERT_PORT" \
    "spark-2" "$SPARK_2_LANE_A" "$EXPERT_PORT" \
    "spark-3" "$SPARK_3_LANE_A" "$EXPERT_PORT"
}

release_stop_local_container() {
  local container="$1"
  if ! docker container inspect "$container" >/dev/null 2>&1; then
    return
  fi
  if [[ "$(docker inspect -f '{{.State.Running}}' "$container")" == true ]]; then
    echo "  coordinator: stopping $container"
    docker stop -t 30 "$container" >/dev/null
  else
    echo "  coordinator: removing stopped $container"
  fi
  docker rm -f "$container" >/dev/null
}

release_stop_host_api() {
  local addr="$1"
  local port="${addr##*:}"
  local pids
  pids="$(
    ss -ltnp "sport = :$port" 2>/dev/null |
      sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' |
      sort -u
  )"
  [[ -n "$pids" ]] || return 0
  local pid command
  for pid in $pids; do
    command="$(ps -p "$pid" -o args= 2>/dev/null || true)"
    [[ "$command" == *ds41rt*coordinator* ]] ||
      release_die "port $port is owned by a non-DS41RT process: pid=$pid $command"
    echo "  coordinator: stopping host API pid=$pid"
    kill -TERM "$pid"
  done
  for _ in $(seq 1 300); do
    ss -ltn "sport = :$port" 2>/dev/null | tail -n +2 | grep -q . || return 0
    sleep 0.1
  done
  release_die "host API did not exit within 30 seconds"
}

release_stop_remote_containers() {
  local host="$1"
  local release_container="$2"
  local legacy_container="$3"
  ssh -o BatchMode=yes "$host" bash -s -- \
    "$host" "$release_container" "$legacy_container" <<'REMOTE'
set -euo pipefail
host="$1"
shift
for container in "$@"; do
  if ! docker container inspect "$container" >/dev/null 2>&1; then
    continue
  fi
  if [[ "$(docker inspect -f '{{.State.Running}}' "$container")" == true ]]; then
    echo "  $host: stopping $container"
    docker stop -t 30 "$container" >/dev/null
  else
    echo "  $host: removing stopped $container"
  fi
  docker rm -f "$container" >/dev/null
done
REMOTE
}

release_stop_services() {
  local coordinator_container="$1"
  local spark_container_prefix="$2"
  release_stop_local_container "$coordinator_container"
  release_stop_host_api "$ADDR"

  local host release_container legacy_container
  local failed=0
  local -a stop_hosts=()
  local -a stop_pids=()
  for host in "$SPARK_0_HOST" "$SPARK_1_HOST" "$SPARK_2_HOST" "$SPARK_3_HOST"; do
    release_container="${spark_container_prefix}-${host}-${EXPERT_PORT}"
    legacy_container="ds41rt-phase0-tcp-expertd-${host}-${EXPERT_PORT}"
    release_stop_remote_containers \
      "$host" "$release_container" "$legacy_container" &
    stop_hosts+=("$host")
    stop_pids+=("$!")
  done
  local index
  for index in "${!stop_pids[@]}"; do
    if ! wait "${stop_pids[$index]}"; then
      echo "  ${stop_hosts[$index]}: failed to stop one or more DS41RT containers" >&2
      failed=1
    fi
  done
  ((failed == 0))
}

release_stop_persistent_local_container() {
  local container="$1"
  if ! docker container inspect "$container" >/dev/null 2>&1; then
    return
  fi
  if [[ "$(docker inspect -f '{{.State.Running}}' "$container")" == true ]]; then
    echo "  coordinator: stopping persistent $container"
    docker stop -t 30 "$container" >/dev/null
  else
    echo "  coordinator: persistent $container is already stopped"
  fi
}

release_stop_persistent_remote_container() {
  local host="$1"
  local container="$2"
  ssh -o BatchMode=yes "$host" bash -s -- "$host" "$container" <<'REMOTE'
set -euo pipefail
host="$1"
container="$2"
if ! docker container inspect "$container" >/dev/null 2>&1; then
  exit 0
fi
if [[ "$(docker inspect -f '{{.State.Running}}' "$container")" == true ]]; then
  echo "  $host: stopping persistent $container"
  docker stop -t 30 "$container" >/dev/null
else
  echo "  $host: persistent $container is already stopped"
fi
REMOTE
}

release_stop_wip_containers() {
  local coordinator_container="${1:-ds41rt-coordinator-wip}"
  local spark_container="${2:-ds41rt-spark-expert-wip}"
  local failed=0

  release_stop_persistent_local_container "$coordinator_container" || failed=1

  local host
  local -a hosts=() pids=()
  for host in "$SPARK_0_HOST" "$SPARK_1_HOST" "$SPARK_2_HOST" "$SPARK_3_HOST"; do
    release_stop_persistent_remote_container "$host" "$spark_container" &
    hosts+=("$host")
    pids+=("$!")
  done
  local index
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      echo "  ${hosts[$index]}: failed to stop persistent $spark_container" >&2
      failed=1
    fi
  done
  ((failed == 0))
}

release_stop_wip_process_in_container() {
  local container="$1"
  local process_name="$2"
  docker exec -i "$container" bash -s -- "$process_name" <<'CONTAINER'
set -euo pipefail
name="$1"
pid_file="/wip/run/$name.pid"
identity_file="/wip/run/$name.identity"
[ -f "$pid_file" ] || exit 0
pid="$(<"$pid_file")"
if ! [[ "$pid" =~ ^[0-9]+$ ]] || ! kill -0 "$pid" 2>/dev/null; then
  rm -f "$pid_file" "$identity_file"
  exit 0
fi
command_line="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)"
case "$command_line" in
  *wip-process.sh*run*"$name"*) ;;
  *) echo "refusing to stop stale WIP pid $pid for $name: $command_line" >&2; exit 2 ;;
esac
kill -TERM "$pid"
for _ in $(seq 1 300); do
  kill -0 "$pid" 2>/dev/null || { rm -f "$pid_file" "$identity_file"; exit 0; }
  sleep 0.1
done
echo "WIP process did not stop within 30 seconds: $name pid=$pid" >&2
exit 2
CONTAINER
}

release_stop_wip_coordinator() {
  local coordinator_process="${1:-coordinator-${ADDR##*:}}"
  local coordinator_container=ds41rt-coordinator-wip

  if docker container inspect "$coordinator_container" >/dev/null 2>&1 &&
    [[ "$(docker inspect -f '{{.State.Running}}' "$coordinator_container")" == true ]]; then
    echo "  coordinator: stopping WIP process $coordinator_process"
    release_stop_wip_process_in_container \
      "$coordinator_container" "$coordinator_process"
  fi
}

release_stop_wip_services() {
  local coordinator_process="${1:-coordinator-${ADDR##*:}}"
  local expert_process="${2:-expert-$EXPERT_PORT}"
  local coordinator_container=ds41rt-coordinator-wip
  local spark_container=ds41rt-spark-expert-wip
  local failed=0

  release_stop_wip_coordinator "$coordinator_process" || failed=1

  local host
  local -a hosts=() pids=()
  for host in "$SPARK_0_HOST" "$SPARK_1_HOST" "$SPARK_2_HOST" "$SPARK_3_HOST"; do
    (
      if ! ssh -o BatchMode=yes "$host" \
        "test \"\$(docker inspect -f '{{.State.Running}}' '$spark_container' 2>/dev/null || true)\" = true"; then
        exit 0
      fi
      ssh -o BatchMode=yes "$host" docker exec -i "$spark_container" \
        bash -s -- "$expert_process" <<'CONTAINER'
set -euo pipefail
name="$1"
pid_file="/wip/run/$name.pid"
identity_file="/wip/run/$name.identity"
[ -f "$pid_file" ] || exit 0
pid="$(<"$pid_file")"
if ! [[ "$pid" =~ ^[0-9]+$ ]] || ! kill -0 "$pid" 2>/dev/null; then
  rm -f "$pid_file" "$identity_file"
  exit 0
fi
command_line="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)"
case "$command_line" in
  *wip-process.sh*run*"$name"*) ;;
  *) echo "refusing to stop stale WIP pid $pid for $name: $command_line" >&2; exit 2 ;;
esac
kill -TERM "$pid"
for _ in $(seq 1 300); do
  kill -0 "$pid" 2>/dev/null || { rm -f "$pid_file" "$identity_file"; exit 0; }
  sleep 0.1
done
echo "WIP process did not stop within 30 seconds: $name pid=$pid" >&2
exit 2
CONTAINER
    ) &
    hosts+=("$host")
    pids+=("$!")
  done
  local index
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      echo "  ${hosts[$index]}: failed to stop WIP process $expert_process" >&2
      failed=1
    fi
  done
  ((failed == 0))
}

# Build availability is independent of the four-rank runtime topology.
release_select_build_hosts() {
  local requested="${1:-}" host configured found prior
  local -a configured_hosts=("$SPARK_0_HOST" "$SPARK_1_HOST" "$SPARK_2_HOST" "$SPARK_3_HOST")
  RELEASE_BUILD_HOSTS=()
  if [[ -z "$requested" ]]; then
    RELEASE_BUILD_HOSTS=("${configured_hosts[@]}")
    return 0
  fi
  [[ "$requested" != ,* && "$requested" != *, && "$requested" != *,,* ]] ||
    release_die "--spark-hosts contains an empty host"
  local -a requested_hosts
  IFS=, read -r -a requested_hosts <<< "$requested"
  for host in "${requested_hosts[@]}"; do
    found=0
    for configured in "${configured_hosts[@]}"; do
      [[ "$host" != "$configured" ]] || found=1
    done
    ((found)) || release_die "build host is not configured: $host"
    for prior in "${RELEASE_BUILD_HOSTS[@]}"; do
      [[ "$host" != "$prior" ]] || release_die "duplicate build host: $host"
    done
    RELEASE_BUILD_HOSTS+=("$host")
  done
}
