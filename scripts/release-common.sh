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

release_validate_exl3_tp2_variants() {
  local capacity="$1" family="$2"
  [[ "$family" =~ ^k([23])([34])$ && ( "$family" == k23 || "$family" == k34 ) ]] || release_die "cannot resolve compact EXL3 package bit tiers"
  local low="${BASH_REMATCH[1]}" high="${BASH_REMATCH[2]}"
  jq -e --argjson capacity "$capacity" --argjson low "$low" --argjson high "$high" '
    . as $manifest | .compute == [12,1] and
    all([1,16,80,256,1024,4096][] | select(. <= $capacity); . as $m |
      all([0,1][]; . as $rank |
        [$manifest.variants[]? |
          select(.directory == ("tp2-rank" + ($rank|tostring) + "/m" + ($m|tostring))) |
          select(.capacity == $m and .intermediate == 1152 and .experts == 384
                 and .top_k == 6 and .output_dtype == "bf16"
                 and .bits == [$low,$high])] | length == 1))
  ' >/dev/null || release_die "Spark EXL3 package lacks required TP2 rank/capacity/shape/bit variants"
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
    EXL3_PAIRED_TP4|TP2_ATTENTION|TP2_QUERY_PROJECTION|TP2_OUTPUT_PROJECTION|TP2_DSPARK_EXPERTS) return 0 ;;
    HTTP_QUEUE_DEPTH|HTTP_QUEUE_WAIT_MS|MODEL_ID|MODEL_VARIANT|MODEL_REVISION|EXPERT_FORMAT|DSPARK|DSPARK_DRAFT_POLICY|RTX_GPUS|RTX_EXPERT_LAYERS|COORDINATOR_GPU|COORDINATOR_GPU_UUID|COORDINATOR_GPU_PCI_BUS_ID|COORDINATOR_GPU_HEADROOM_GIB|KV_POOL_TOKENS|KV_POOL_SIZE|HOST_CACHE_BYTES|MEMORY_RESERVATION|MAX_CONTEXT_TOKENS|MAX_OUTPUT_TOKENS|CONCURRENCY|PREFIX_CACHE_ENTRIES|PREFILL_BATCH_TOKENS|SPARK_DEVICE_BUDGET_BYTES|SPARK_REDUCTION_MIN_ROWS|SPARKINFER_EXL3|SPARK_COUNT|SPARK_TP|SPARK_EP|ADDR|EXPERT_PORT|SPARK_[0-5]_HOST|SPARK_[0-5]_LANE_A|SPARK_[0-5]_LANE_B|COORDINATOR_DOCKER_DEV|COORDINATOR_DOCKER_INFERENCE|SPARK_EXPERT_DOCKER_DEV|SPARK_EXPERT_DOCKER_INFERENCE)
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
  TP2_ATTENTION=off
  TP2_QUERY_PROJECTION=off
  TP2_OUTPUT_PROJECTION=off
  TP2_DSPARK_EXPERTS=off
  DSPARK_DRAFT_POLICY=adaptive
  RTX_EXPERT_LAYERS=auto
  RTX_GPUS=auto
  COORDINATOR_GPU=0
  COORDINATOR_GPU_UUID=
  COORDINATOR_GPU_PCI_BUS_ID=
  COORDINATOR_GPU_HEADROOM_GIB=8
  KV_POOL_TOKENS=
  KV_POOL_SIZE=
  HOST_CACHE_BYTES=auto
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
  SPARK_COUNT=4
  # Optional explicit replicated expert-group topology. Absent means the legacy
  # geometry (TP = SPARK_COUNT, EP = 1). See docs/tp-ep-configuration.md.
  SPARK_TP=
  SPARK_EP=
  ADDR=0.0.0.0:8000
  EXPERT_PORT=19441
  COORDINATOR_DOCKER_DEV=ds41rt-coordinator-dev
  COORDINATOR_DOCKER_INFERENCE=ds41rt-coordinator
  SPARK_EXPERT_DOCKER_DEV=ds41rt-spark-expert-dev
  SPARK_EXPERT_DOCKER_INFERENCE=ds41rt-spark-expert
  for release_i in 0 1 2 3 4 5; do
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
  [[ "$RTX_EXPERT_LAYERS" == auto || "$RTX_EXPERT_LAYERS" =~ ^([0-9]|[1-3][0-9]|40)$ ]] ||
    release_die "RTX_EXPERT_LAYERS must be auto or 0..40"
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

  release_validate_tp2_options
  release_validate_spark_topology

  case "$SPARK_COUNT" in
    0)
      [[ "$RTX_EXPERT_LAYERS" == 40 ]] || release_die "SPARK_COUNT=0 requires RTX_EXPERT_LAYERS=40 (every routed layer must fit the RTX layout)"
      [[ "$RTX_GPUS" != 1 ]] || release_die "SPARK_COUNT=0 requires two RTX GPUs"
      ;;
    2) release_validate_compact_tp2 ;;
    4) ;;
    6)
      release_spark_topology_explicit ||
        release_die "SPARK_COUNT=6 requires explicit SPARK_TP and SPARK_EP (six Sparks are only approved as a replicated native topology)"
      ;;
    *) release_die "SPARK_COUNT must be 0, 2, 4, or 6" ;;
  esac

  local missing_b=0 present_b=0 spark_required="$SPARK_COUNT"
  for ((release_i = 0; release_i < spark_required; release_i++)); do
    local host_name="SPARK_${release_i}_HOST"
    local lane_a_name="SPARK_${release_i}_LANE_A"
    local lane_b_name="SPARK_${release_i}_LANE_B"
    if ((release_i < spark_required)); then
      [[ -n "${!host_name}" ]] || release_die "$host_name must not be empty"
      [[ "${!lane_a_name}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || release_die "$lane_a_name must be an IPv4 address"
    fi
    if [[ -n "${!lane_b_name}" ]]; then
      ((present_b += 1))
    else
      ((missing_b += 1))
    fi
  done
  ((present_b == 0 || missing_b == 0)) || release_die "secondary Spark rail must provide all $SPARK_COUNT active LANE_B values or none"

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

# Two Spark ranks are an opt-in compact EXL3 topology, not RTX tensor
# parallelism. Keep the coordinator ceiling explicit; never raise it to fit.
release_validate_compact_tp2() {
  [[ "$SPARK_COUNT" == 2 ]] || return 0
  [[ "$EXPERT_FORMAT" == exl3 ]] || release_die "SPARK_COUNT=2 requires EXPERT_FORMAT=exl3"
  [[ "$RTX_GPUS" != 2 ]] || release_die "SPARK_COUNT=2 requires a single RTX GPU"
  [[ "$EXL3_PAIRED_TP4" == off ]] || release_die "SPARK_COUNT=2 is incompatible with EXL3_PAIRED_TP4"
  MEMORY_RESERVATION="${MEMORY_RESERVATION:-32GiB}"
  KV_POOL_SIZE="${KV_POOL_SIZE:-2GiB}"
  PREFILL_BATCH_TOKENS="${PREFILL_BATCH_TOKENS:-256}"
  [[ "$PREFILL_BATCH_TOKENS" =~ ^[1-9][0-9]*$ ]] &&
    ((PREFILL_BATCH_TOKENS >= 80 && PREFILL_BATCH_TOKENS <= 4096)) ||
    release_die "PREFILL_BATCH_TOKENS must be in 80..4096"
  if ((PREFILL_BATCH_TOKENS > 256)); then
    echo "ds41rt release: compact Spark TP2 caps PREFILL_BATCH_TOKENS=$PREFILL_BATCH_TOKENS to 256 to fit the 32GiB ceiling" >&2
    PREFILL_BATCH_TOKENS=256
  fi
  python3 - "$MEMORY_RESERVATION" <<'PY' || release_die "SPARK_COUNT=2 requires a positive absolute MEMORY_RESERVATION no greater than 32GiB (percentages are not allowed)"
import re
import sys
from decimal import Decimal
match = re.fullmatch(r'([0-9]+(?:\.[0-9]{1,6})?)(B|MB|GB|MiB|GiB)', sys.argv[1])
if not match:
    sys.exit(1)
scale = {'B': 1, 'MB': 10**6, 'GB': 10**9, 'MiB': 2**20, 'GiB': 2**30}
size = Decimal(match[1]) * scale[match[2]]
sys.exit(0 if 1 <= size <= 32 * 2**30 else 1)
PY
}

release_spark_values() {
  local field="$1" i name
  for ((i = 0; i < SPARK_COUNT; i++)); do
    name="SPARK_${i}_${field}"
    printf '%s\n' "${!name}"
  done
}

# ---------------------------------------------------------------------------
# Opt-in replicated expert-group topology (SPARK_TP x SPARK_EP = SPARK_COUNT).
#
# The default configuration sets neither key and keeps the legacy geometry:
# every Spark rank is one TP rank of a single replicated group. Explicit keys
# are all-or-none and only valid for the approved native official topologies.
# The rank map is group-major: group = rank / TP and tp_rank = rank % TP.
# ---------------------------------------------------------------------------

release_spark_topology_explicit() {
  [[ -n "$SPARK_TP" || -n "$SPARK_EP" ]]
}

release_spark_tp() {
  if [[ -n "$SPARK_TP" ]]; then printf '%s\n' "$SPARK_TP"; else printf '%s\n' "$SPARK_COUNT"; fi
}

release_spark_ep() {
  if [[ -n "$SPARK_EP" ]]; then printf '%s\n' "$SPARK_EP"; else printf '1\n'; fi
}

release_spark_group() {
  local rank="$1" tp="$2"
  [[ "$rank" =~ ^[0-9]+$ && "$tp" =~ ^[1-9][0-9]*$ ]] ||
    release_die "invalid Spark rank/TP for group resolution: rank=$rank tp=$tp"
  printf '%s\n' "$((rank / tp))"
}

release_spark_tp_rank() {
  local rank="$1" tp="$2"
  [[ "$rank" =~ ^[0-9]+$ && "$tp" =~ ^[1-9][0-9]*$ ]] ||
    release_die "invalid Spark rank/TP for tp-rank resolution: rank=$rank tp=$tp"
  printf '%s\n' "$((rank % tp))"
}

# Print "rank group tp_rank" for every configured physical Spark rank.
release_spark_rank_map() {
  local tp rank
  tp="$(release_spark_tp)"
  for ((rank = 0; rank < SPARK_COUNT; rank++)); do
    printf '%s %s %s\n' "$rank" "$(release_spark_group "$rank" "$tp")" "$(release_spark_tp_rank "$rank" "$tp")"
  done
}

release_validate_spark_topology() {
  case "$SPARK_TP" in
    ""|2|3|4) ;;
    *) release_die "SPARK_TP must be 2, 3, or 4" ;;
  esac
  case "$SPARK_EP" in
    ""|1|2|3) ;;
    *) release_die "SPARK_EP must be 1, 2, or 3" ;;
  esac
  [[ -n "$SPARK_TP" && -n "$SPARK_EP" || -z "$SPARK_TP" && -z "$SPARK_EP" ]] ||
    release_die "SPARK_TP and SPARK_EP must be set together or omitted together"
  release_spark_topology_explicit || return 0

  [[ "$SPARK_COUNT" == 4 || "$SPARK_COUNT" == 6 ]] ||
    release_die "explicit SPARK_TP/SPARK_EP requires SPARK_COUNT=4 or 6"
  [[ "$EXPERT_FORMAT" == native ]] ||
    release_die "explicit SPARK_TP/SPARK_EP requires EXPERT_FORMAT=native"
  [[ "$EXL3_PAIRED_TP4" == off ]] ||
    release_die "explicit SPARK_TP/SPARK_EP is a native topology; EXL3_PAIRED_TP4 must be off"
  [[ "$SPARKINFER_EXL3" == disable ]] ||
    release_die "explicit SPARK_TP/SPARK_EP requires SPARKINFER_EXL3=disable"
  ((SPARK_TP * SPARK_EP == SPARK_COUNT)) ||
    release_die "SPARK_TP(${SPARK_TP}) * SPARK_EP(${SPARK_EP}) must equal SPARK_COUNT(${SPARK_COUNT})"
  case "${SPARK_TP}x${SPARK_EP}" in
    2x2|3x2|2x3|4x1) ;;
    *) release_die "unsupported native Spark topology TP${SPARK_TP}EP${SPARK_EP}; approved: TP2EP2, TP3EP2, TP2EP3, TP4EP1" ;;
  esac
}

release_hosts_csv() { release_spark_values HOST | paste -sd, -; }
release_lane_a_csv() { release_spark_values LANE_A | paste -sd, -; }
release_lane_b_csv() {
  [[ -n "$SPARK_0_LANE_B" ]] || return 0
  release_spark_values LANE_B | paste -sd, -
}

# Optional RDMA/verbs tuning values shared by run.sh and the isolated candidate
# launcher. They are forwarded to both roles only when the operator sets them;
# an empty value keeps the transport's own device selection. The device map is
# `local-ip=device` comma-separated and must name unique IPv4 sources, which is
# what multi-homed six-rank hosts (rhea/moa) need to pin a rail.
release_validate_verbs_device_map() {
  local map="${DS41RT_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP:-}"
  [[ -n "$map" ]] || return 0
  local -a entries=() seen_ips=()
  local entry ip dev prior
  IFS=',' read -ra entries <<<"$map"
  for entry in "${entries[@]}"; do
    [[ "$entry" == *=* ]] ||
      release_die "invalid DS41RT_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP entry: '$entry' (expected local-ip=device)"
    ip="${entry%%=*}"
    dev="${entry#*=}"
    [[ "$ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ && -n "$dev" ]] ||
      release_die "invalid DS41RT_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP entry: '$entry' (expected local-ip=device)"
    for prior in ${seen_ips[@]+"${seen_ips[@]}"}; do
      [[ "$ip" != "$prior" ]] ||
        release_die "duplicate DS41RT_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP ip: $ip"
    done
    seen_ips+=("$ip")
  done
}

release_expert_hosts_csv() {
  local i lane separator=
  for ((i = 0; i < SPARK_COUNT; i++)); do
    lane="SPARK_${i}_LANE_A"
    printf '%sspark-%s=%s:%s' "$separator" "$i" "${!lane}" "$EXPERT_PORT"
    separator=,
  done
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
  local -a active_hosts=()
  mapfile -t active_hosts < <(release_spark_values HOST)
  for host in "${active_hosts[@]}"; do
    [[ -n "$host" ]] || continue
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
  local -a active_hosts=()
  mapfile -t active_hosts < <(release_spark_values HOST)
  for host in "${active_hosts[@]}"; do
    [[ -n "$host" ]] || continue
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
  local -a active_hosts=()
  mapfile -t active_hosts < <(release_spark_values HOST)
  for host in "${active_hosts[@]}"; do
    [[ -n "$host" ]] || continue
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
  local requested="${1:-}" host configured found prior name
  local -a configured_hosts=()
  if [[ -n "${SPARK_COUNT:-}" ]]; then
    mapfile -t configured_hosts < <(release_spark_values HOST)
  else
    # Callers that have not loaded a configuration may still set the host keys
    # directly; fall back to the bounded rank keys and keep only filled ones.
    for name in SPARK_0_HOST SPARK_1_HOST SPARK_2_HOST SPARK_3_HOST SPARK_4_HOST SPARK_5_HOST; do
      [[ -n "${!name:-}" ]] && configured_hosts+=("${!name}")
    done
  fi
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

# The Spark interval must cover every routed layer the coordinator delegates.
# Auto currently guarantees at least 20 RTX layers for the dual layout; an
# explicit boundary can be lower while the coordinator-produced plan is pending.
release_spark_first_layer() {
  local layout="$1" layers="$2"
  if [[ "$layout" == 1 ]]; then printf '0\n'; return; fi
  [[ "$layout" == 2 ]] || release_die "invalid RTX layout"
  if [[ "$layers" == auto ]]; then printf '20\n'; return; fi
  [[ "$layers" =~ ^([1-9]|[1-3][0-9]|40)$ ]] || release_die "dual RTX expert layers must be 1..40"
  # The existing expert service requires a nonempty interval even when every
  # routed layer is local. Keep its last layer as an unused transport endpoint.
  if [[ "$layers" == 40 ]]; then printf '39\n'; return; fi
  printf '%s\n' "$layers"
}

# Native per-TP-rank routed weight for one 40-layer backbone layer, in bytes.
# These are the exact tensor windows from the official checkpoint geometry:
# 1,804,861,440 B/rank at TP4 raw, padded to a 640-wide kernel extent
# (2,005,401,600 B); TP2 (1152) and TP3 (768) need no padding. They are weight
# arithmetic only and do not include workspace, staging or runtime headroom.
release_spark_layer_bytes() {
  case "$1" in
    2) printf '%s\n' 3609722880 ;;
    3) printf '%s\n' 2406481920 ;;
    4) printf '%s\n' 2005401600 ;;
    *) release_die "unsupported Spark TP degree: $1 (expected 2, 3, or 4)" ;;
  esac
}

release_spark_remote_layers() {
  local first_layer="$1"
  [[ "$first_layer" =~ ^([0-9]|[1-3][0-9]|40)$ ]] ||
    release_die "Spark first layer must be 0..40: $first_layer"
  ((first_layer <= 39)) ||
    release_die "Spark first layer must be 0..39: $first_layer"
  printf '%s\n' "$((40 - first_layer))"
}

release_spark_remote_weight_bytes() {
  local first_layer="$1" tp="$2"
  printf '%s\n' "$(($(release_spark_remote_layers "$first_layer") * $(release_spark_layer_bytes "$tp")))"
}

# Weight-only admission for the resolved dynamic RTX/Spark boundary.
#
# IMPORTANT: a successful check means the *weights* fit the Spark budget. It is
# not a launch-feasibility claim: SparkInfer workspace, load staging, replicated
# activation buffers and runtime headroom are only known after the expert
# service reports them at startup. run.sh prints the residual and labels it
# explicitly. Weight-only overflow is a hard failure before any service change.
release_validate_spark_weight_admission() {
  local first_layer="$1" tp="$2" budget="$3" remote_layers weight margin
  [[ "$budget" =~ ^[1-9][0-9]*$ ]] || release_die "Spark device budget must be a positive integer"
  remote_layers="$(release_spark_remote_layers "$first_layer")"
  weight="$(release_spark_remote_weight_bytes "$first_layer" "$tp")"
  ((weight <= budget)) ||
    release_die "Spark TP${tp} weight-only admission fails: ${remote_layers} remote layers need ${weight} B > ${budget} B budget (lower the RTX boundary first layer)"
  margin=$((budget - weight))
  printf 'remote_layers=%s per_rank_weight_bytes=%s weight_margin_bytes=%s budget_bytes=%s workspace_accounted=no\n' \
    "$remote_layers" "$weight" "$margin" "$budget"
}

# Validate booleans independently of resolved GPU selection. Called again after
# command-line overrides, before launcher operations that change services.
release_validate_tp2_options() {
  local name
  for name in TP2_ATTENTION TP2_QUERY_PROJECTION TP2_OUTPUT_PROJECTION TP2_DSPARK_EXPERTS; do
    case "${!name}" in on|off) ;; *) release_die "$name must be on or off" ;; esac
  done
}

release_tp2_enabled() {
  [[ "$TP2_ATTENTION" == on || "$TP2_QUERY_PROJECTION" == on ||
     "$TP2_OUTPUT_PROJECTION" == on || "$TP2_DSPARK_EXPERTS" == on ]]
}
