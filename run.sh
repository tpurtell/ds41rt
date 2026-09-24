#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$repo_root/scripts/release-common.sh"

usage() {
  cat <<'EOF'
Usage: ./run.sh [OPTIONS]

Starts native DeepSeek V4.1 on the RTX coordinator and configured Spark ranks.
SPARK_COUNT=2 or SPARK_COUNT=3 (compact EXL3, no SPARK_TP/SPARK_EP keys)
requires EXL3 on one RTX, with a hard 32GiB GPU ceiling.
An optional expert-group topology is selected in the configuration with
SPARK_TP and SPARK_EP (all-or-none, native checkpoint only); at SPARK_COUNT=3
the explicit SPARK_TP=3 SPARK_EP=1 form is one unreplicated three-rank group.
See docs/tp-ep-configuration.md.
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
  --dspark-draft-limit N        draft width per request, 1..7 (default 5)
  --tp2-attention               split attention heads; replicate KV (default off)
  --tp2-query-projection        split query-B projection (default off)
  --tp2-output-projection       split output-B projection (default off)
  --tp2-dspark-experts          split native draft routed experts (default off)
  --no-tp2-<option>             disable the corresponding configured TP2 option
  --restart                     replace the current release deployment
  --dry-run                     validate without changing services

Optional RDMA tuning env values are forwarded to both roles only when set:
  DS41RT_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP (local-ip=device,...),
  DS41RT_VERBS_APP_IB_PORT_NUM, DS41RT_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES.
This is how a multi-homed six-rank launch pins the rail per host.

Every remote step shares one SSH option set with the release build scripts:
  DS41RT_RELEASE_SSH_CONFIG       ssh config file to use (default empty: stock
                                  OpenSSH resolution; BatchMode is always forced so
                                  a poll or cleanup can never wait on a prompt).
                                  /dev/null discards a broken system include but
                                  also drops your ~/.ssh/config host aliases, so
                                  prefer a file containing 'Include ~/.ssh/config'.
EOF
}

config="$repo_root/ds41rt.config"
restart=0
dry_run=0
dspark_draft_limit=""
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
    --dspark-draft-limit) dspark_draft_limit="${2:?$1 requires N}"; shift 2 ;;
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
# RTX_GPUS can be overridden above, so re-check the explicit topology layout.
release_validate_spark_topology
topology_explicit=0
release_spark_topology_explicit && topology_explicit=1
spark_tp="$(release_spark_tp)"
spark_ep="$(release_spark_ep)"
# Extra SM121 expert roles the image must have been built with. Legacy TP4EP1
# and the shipped default require none, so a prebuilt v8 image without any role
# label keeps working unchanged.
spark_tp_roles_required=""
if ((topology_explicit)); then
  case "$spark_tp" in
    2) spark_tp_roles_required=tp2 ;;
    3) spark_tp_roles_required=tp3 ;;
    6) spark_tp_roles_required=tp6 ;;
  esac
fi
[[ -z "$HTTP_QUEUE_DEPTH" || ( "$HTTP_QUEUE_DEPTH" =~ ^[1-9][0-9]*$ && "$HTTP_QUEUE_DEPTH" -le 4096 ) ]] || release_die "HTTP_QUEUE_DEPTH must be in 1..4096"
[[ "$HTTP_QUEUE_WAIT_MS" =~ ^[0-9]+$ ]] || release_die "HTTP_QUEUE_WAIT_MS must be non-negative"
[[ "$HOST_CACHE_BYTES" == auto || "$HOST_CACHE_BYTES" =~ ^[0-9]+([.][0-9]{1,6})?(B|MB|GB|MiB|GiB)?$ ]] || release_die "HOST_CACHE_BYTES must be auto, 0, or a byte size"
case "$DSPARK" in on|off) ;; *) release_die "DSPARK must be on or off" ;; esac
[[ -z "$dspark_draft_limit" || "$dspark_draft_limit" =~ ^[1-7]$ ]] ||
  release_die "--dspark-draft-limit must be in 1..7"
[[ -z "$dspark_draft_limit" || "$DSPARK" == on ]] ||
  release_die "--dspark-draft-limit requires dSpark"
release_validate_verbs_device_map
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
release_validate_compact_spark
if ((SPARK_COUNT == 0)); then
  [[ "$RTX_EXPERT_LAYERS" == 40 && "$RTX_GPUS" != 1 ]] ||
    release_die "SPARK_COUNT=0 requires two RTX GPUs and RTX_EXPERT_LAYERS=40"
fi

# Resolve the one SSH option set used by every remote step below: preflight reads,
# expert launch, readiness polling and the EXIT cleanup. Placed after argument,
# config and value validation so a mistyped DS41RT_RELEASE_SSH_CONFIG fails before
# any host is contacted, and before the daemon/image checks so --dry-run and every
# action share one answer. Stock resolution stays the default.
release_configure_ssh_transport

for tool in docker ssh curl jq ss nvidia-smi sha256sum python3; do release_need "$tool"; done
docker info >/dev/null 2>&1 || release_die "local Docker daemon is unavailable"
docker image inspect "$COORDINATOR_DOCKER_INFERENCE" >/dev/null 2>&1 ||
  release_die "coordinator image is missing: $COORDINATOR_DOCKER_INFERENCE (pull the published pair, or build it with a config whose COORDINATOR_DOCKER_INFERENCE names this tag)"
hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
release_resolve_local_model_revision "$hf_home"
release_resolve_coordinator_gpu_identity
snapshot_rel="hub/models--${RELEASE_MODEL_ID//\//--}/snapshots/$RELEASE_MODEL_REVISION"
model_is_exl3="$(jq -r '.quantization_config.quant_method == "exl3"' "$hf_home/$snapshot_rel/config.json")"
if release_spark_compact_active; then
  [[ "$model_is_exl3" == true ]] || release_die "SPARK_COUNT=$SPARK_COUNT requires an EXL3 checkpoint"
fi
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
model_is_nvfp4="$(jq -r '.quantization_config.moe_quant_algo // empty' "$hf_home/$snapshot_rel/config.json" 2>/dev/null || true)"
[[ "$model_is_nvfp4" == "NVFP4" ]] && expert_format=nvfp4
# The explicit replicated topology is approved for the official native
# checkpoint only. Reject a routed quant before any service change; EXL3 and
# NVFP4 keep their existing non-topology behavior.
if ((topology_explicit)); then
  [[ "$model_is_exl3" != true ]] ||
    release_die "explicit SPARK_TP/SPARK_EP requires the native official checkpoint; EXL3 is not supported"
  [[ "$model_is_nvfp4" != "NVFP4" ]] ||
    release_die "explicit SPARK_TP/SPARK_EP requires the native official checkpoint; NVFP4 is not supported"
fi
if [[ "$model_is_exl3" == true ]]; then
  case "$exl3_family_tag" in
    k23) expert_format=exl3-k23 ;;
    k34) expert_format=exl3-k34 ;;
    # Unresolvable families keep the conservative native constants; the
    # dual fit model is only exercised for two-RTX launches.
    *) expert_format=native ;;
  esac
fi
# Auto placement must not pick an RTX boundary so low that the remote Spark
# weights cannot fit the device budget. The floor is derived from the actual
# budget and the resolved TP degree, not a hardcoded 20.
if ((topology_explicit)) && [[ "$RTX_EXPERT_LAYERS" == auto ]]; then
  remote_capacity=$((SPARK_DEVICE_BUDGET_BYTES / $(release_spark_layer_bytes "$spark_tp")))
  ((remote_capacity > 40)) && remote_capacity=40
  topology_min_layers=$((40 - remote_capacity))
  ((topology_min_layers < 1)) && topology_min_layers=1
  ((minimum_expert_layers >= topology_min_layers)) || minimum_expert_layers="$topology_min_layers"
fi
gpu_selection_mode="$RTX_GPUS"
# Auto must not turn the compact topology into a two-RTX launch.
# The Spark TP/EP degree does NOT select the RTX layout: RTX_GPUS (auto or an
# explicit 1/2) decides, and an infeasible combination is rejected below by the
# resolved weight budget instead of a topology-to-layout hardcode.
compact_selection_args=()
if release_spark_compact_active; then
  gpu_selection_mode=1
  compact_selection_args+=(--compact-spark)
fi
gpu_selection="$(python3 "$repo_root/scripts/select-release-gpus.py" \
  --mode "$gpu_selection_mode" \
  --minimum-expert-layers "$minimum_expert_layers" \
  --primary-uuid "$RELEASE_COORDINATOR_GPU_UUID" \
  --concurrency "$CONCURRENCY" \
  --max-context-tokens "$MAX_CONTEXT_TOKENS" \
  --retained-turns "$PREFIX_CACHE_ENTRIES" \
  --kv-pool-size "$KV_POOL_SIZE" \
  --memory-reservation "$MEMORY_RESERVATION" \
  --expert-format "$expert_format" \
  "${compact_selection_args[@]}" "${reclaim_args[@]}")"
RELEASE_RTX_GPUS="$(jq -r '.count' <<<"$gpu_selection")"
if release_spark_compact_active; then
  ((RELEASE_RTX_GPUS == 1)) || release_die "SPARK_COUNT=$SPARK_COUNT requires one selected RTX GPU"
fi
((SPARK_COUNT != 0 || RELEASE_RTX_GPUS == 2)) || release_die "SPARK_COUNT=0 requires two selected RTX GPUs"
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
# A dual-RTX auto boundary, and a single-RTX explicit topology with an explicit
# local count, are published by the coordinator; the weight-only admission below
# must not pretend a boundary it has not read yet.
if { ((RELEASE_RTX_GPUS == 2)) && [[ "$RTX_EXPERT_LAYERS" == auto ]]; } ||
   { ((RELEASE_RTX_GPUS == 1)) && [[ "$RTX_EXPERT_LAYERS" =~ ^([1-9]|[1-3][0-9])$ ]]; }; then
  spark_first_layer=runtime-plan
fi
# Admission against the resolved boundary. Weight-only is all the launcher can
# know before the expert service reports its workspace; a pass is not a
# launch-feasibility claim. An auto dual boundary is published by the
# coordinator and re-checked after plan.json is read below.
spark_admission="not-applicable"
if ((topology_explicit)); then
  if [[ "$spark_first_layer" == runtime-plan ]]; then
    spark_admission="PENDING (coordinator placement plan not yet published; weight-only check uses the real dynamic boundary)"
  else
    spark_admission="$(release_validate_spark_weight_admission "$spark_first_layer" "$spark_tp" "$SPARK_DEVICE_BUDGET_BYTES")"
  fi
fi
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

expert_capacity=4096
if ((PREFILL_BATCH_TOKENS <= 80)); then expert_capacity=80
elif ((PREFILL_BATCH_TOKENS <= 256)); then expert_capacity=256
elif ((PREFILL_BATCH_TOKENS <= 1024)); then expert_capacity=1024
fi
mapfile -t hosts < <(release_spark_values HOST)
mapfile -t lanes < <(release_spark_values LANE_A)
spark_exl3_identity=""
for host in "${hosts[@]}"; do
  # Every argument below is non-empty on purpose. OpenSSH joins the command
  # arguments into one remote shell line, so an empty argument is elided and
  # every later positional shifts; the optional EXL3 family tag therefore
  # travels as a sentinel and the host name sits ahead of it.
  spark_manifest="$(release_ssh -o ConnectTimeout=10 "$host" bash -s -- "$SPARK_EXPERT_DOCKER_INFERENCE" "$engine_commit" "$sparkinfer_commit" "$snapshot_rel" "$host" "$model_is_exl3" "${exl3_family_tag:-__none__}" <<'REMOTE'
set -euo pipefail
image="$1"; engine="$2"; sparkinfer="$3"; snapshot_rel="$4"; host="$5"
exl3="$6"; exl3_family="${7:-__none__}"
[[ "$exl3_family" == __none__ ]] && exl3_family=
# Every failure names the host and the check: this block runs over SSH, so a
# bare nonzero exit would otherwise surface as an unexplained transport error.
# Diagnostics go to stderr; stdout stays the EXL3 manifest alone.
die() { echo "spark preflight on $host: $*" >&2; exit 1; }
docker info >/dev/null 2>&1 || die "the Docker daemon is unavailable"
docker image inspect "$image" >/dev/null 2>&1 || die "inference image is missing: $image (pull or distribute it)"
[[ "$(docker image inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$image")" == "$engine" ]] ||
  die "$image has another engine revision"
[[ "$(docker image inspect -f '{{index .Config.Labels "io.ds41rt.sparkinfer.revision"}}' "$image")" == "$sparkinfer" ]] ||
  die "$image uses another SparkInfer revision"
hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
[[ -d "$hf_home/$snapshot_rel" ]] || die "model snapshot is missing: $snapshot_rel"
if find "$hf_home/$snapshot_rel" -xtype l -print -quit | grep -q .; then
  die "model snapshot has dangling links: $snapshot_rel"
fi
if [[ "$exl3" == true ]]; then
  docker run --rm --network none --entrypoint /bin/sh "$image" -c \
    'if [ -f "/opt/ds41rt/lib/exl3/exl3-'"$exl3_family"'/manifest.json" ]; then cat "/opt/ds41rt/lib/exl3/exl3-'"$exl3_family"'/manifest.json"; else cat /opt/ds41rt/lib/exl3/manifest.json; fi'
fi
REMOTE
)" || release_die "Spark host preflight failed on $host (see the messages above)"
  if [[ "$model_is_exl3" == true ]]; then
    identity="$(release_exl3_package_identity "$sparkinfer_commit" <<<"$spark_manifest")"
    if release_spark_compact_active; then
      [[ "$identity" != paired:* ]] ||
        release_die "SPARK_COUNT=$SPARK_COUNT requires disjoint EXL3 packages, not paired TP4"
      release_validate_exl3_compact_variants "$expert_capacity" "$exl3_family_tag" "$SPARK_COUNT" <<<"$spark_manifest"
    fi
    [[ -z "$spark_exl3_identity" || "$spark_exl3_identity" == "$identity" ]] ||
      release_die "Spark EXL3 packages differ across hosts; rebuild/distribute matching images"
    spark_exl3_identity="$identity"
  fi
done

# An explicit TP2/TP3/TP6 topology needs the matching SM121 expert role baked
# into the Spark image. The published universal release pair advertises every
# extra role at once (label `io.ds41rt.v41.spark_tp_roles=tp2;tp3;tp6`), so one
# published image pair serves all approved topologies and this launch selects the
# role it needs. A prebuilt legacy image carries no role label and keeps working
# for the default TP4EP1 path; an explicit topology is refused here, before any
# service is stopped or replaced.
spark_advertised_roles=""
if [[ -n "$spark_tp_roles_required" ]]; then
  for host in "${hosts[@]}"; do
    advertised_roles="$(
      release_ssh -o ConnectTimeout=10 "$host" \
        "docker image inspect -f '{{index .Config.Labels \"io.ds41rt.v41.spark_tp_roles\"}}' '$SPARK_EXPERT_DOCKER_INFERENCE'"
    )" || release_die "$host cannot report the role label of $SPARK_EXPERT_DOCKER_INFERENCE; is the Spark image present on that host?"
    [[ ";$advertised_roles;" == *";$spark_tp_roles_required;"* ]] ||
      release_die "$host Spark image does not advertise required expert role $spark_tp_roles_required (advertised: ${advertised_roles:-<none>}); refusing an unbuilt TP$spark_tp topology: use the published universal release pair, or rebuild with DS41RT_RELEASE_SPARK_TP_ROLES=$spark_tp_roles_required"
    [[ -n "$spark_advertised_roles" ]] || spark_advertised_roles="$advertised_roles"
  done
fi

# Zero-Spark deployments hold every routed layer on the RTX pair; the daemon
# still requires four peer addresses but never connects to them.
if ((SPARK_COUNT == 0)); then
  peers="127.0.0.1:1,127.0.0.1:2,127.0.0.1:3,127.0.0.1:4"
else
  peer_addresses=()
  for lane in "${lanes[@]}"; do peer_addresses+=("$lane:$EXPERT_PORT"); done
  peers="$(IFS=,; echo "${peer_addresses[*]}")"
fi
fingerprint="$(printf '%s\n' "$engine_commit" "$RELEASE_MODEL_ID" "$RELEASE_MODEL_REVISION" "$ADDR" "$RELEASE_RTX_GPUS" "$gpu_uuid_csv" "$gpu_pci_csv" "$CONCURRENCY" "${HTTP_QUEUE_DEPTH:-$CONCURRENCY}" "$HTTP_QUEUE_WAIT_MS" "$HOST_CACHE_BYTES" "$RTX_EXPERT_LAYERS" "$KV_POOL_SIZE" "$MEMORY_RESERVATION" "$PREFIX_CACHE_ENTRIES" "$MAX_CONTEXT_TOKENS" "$MAX_OUTPUT_TOKENS" "$PREFILL_BATCH_TOKENS" "$DSPARK" "$DSPARK_DRAFT_POLICY" "${dspark_draft_limit:-auto}" "$TP2_ATTENTION" "$TP2_QUERY_PROJECTION" "$TP2_OUTPUT_PROJECTION" "$TP2_DSPARK_EXPERTS" "${DS41RT_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP:-}" "${DS41RT_VERBS_APP_IB_PORT_NUM:-}" "${DS41RT_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES:-}" "$SPARK_DEVICE_BUDGET_BYTES" "$spark_first_layer" "$SPARK_COUNT" "$(release_hosts_csv)" "$peers" "$spark_exl3_identity" "spark-topology=${spark_tp}x${spark_ep}:explicit=${topology_explicit}" "v41-spark-tp-roles=${spark_tp_roles_required}" | sha256sum | awk '{print $1}')"
spark_prefix="$RELEASE_SPARK_CONTAINER_PREFIX"

if ((dry_run)); then
  echo "Dry-run checks passed for native V4.1; no services changed."
  echo "  RTX layout: $RELEASE_RTX_GPUS GPU(s), host indices $gpu_index_csv"
  echo "  physical GPUs: $gpu_uuid_csv"
  echo "  TP2 attention/query/output/draft experts: $TP2_ATTENTION/$TP2_QUERY_PROJECTION/$TP2_OUTPUT_PROJECTION/$TP2_DSPARK_EXPERTS"
  echo "  Spark ranks: $SPARK_COUNT; hosts: $(release_hosts_csv)"
  echo "  Spark peers: $peers"
  echo "  Spark first routed layer: $spark_first_layer"
  if ((topology_explicit)); then
    echo "  Spark topology: TP=$spark_tp EP=$spark_ep (group=rank/TP, tp_rank=rank%TP; no dummy ranks)"
    while read -r global_rank group_index tp_rank; do
      echo "    global=$global_rank group=$group_index tp_rank=$tp_rank"
    done < <(release_spark_rank_map)
    echo "  Spark weight admission (workspace/staging NOT accounted; not a feasibility claim): $spark_admission"
    [[ -z "$spark_tp_roles_required" ]] ||
      echo "  Spark image must advertise V41 expert role: $spark_tp_roles_required (selected rank advertises: ${spark_advertised_roles:-<unreadable>})"
  fi
  echo "  coordinator memory reservation: ${MEMORY_RESERVATION:-runtime default}"
  echo "  prefill batch tokens: $PREFILL_BATCH_TOKENS; expert capacity: $expert_capacity"
  echo "  dSpark draft: policy=$DSPARK_DRAFT_POLICY limit=${dspark_draft_limit:-5}"
  [[ -z "${DS41RT_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP:-}" ]] || echo "  RDMA device map: $DS41RT_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP"
  [[ -z "${DS41RT_VERBS_APP_IB_PORT_NUM:-}" ]] || echo "  RDMA IB port: $DS41RT_VERBS_APP_IB_PORT_NUM"
  [[ -z "${DS41RT_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES:-}" ]] || echo "  RDMA execution lanes: $DS41RT_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES"
  echo "  release identity: $fingerprint"
  [[ -z "$spark_exl3_identity" ]] || echo "  Spark EXL3 package: $spark_exl3_identity"
  exit 0
fi
if ((restart)); then
  release_stop_services "$coordinator" "$spark_prefix"
else
  docker inspect "$coordinator" >/dev/null 2>&1 && release_die "$coordinator already exists; use --restart"
  for i in "${!hosts[@]}"; do
    remote="${spark_prefix}-${hosts[$i]}-${EXPERT_PORT}"
    release_ssh "${hosts[$i]}" "! docker inspect '$remote' >/dev/null 2>&1" || release_die "$remote already exists; use --restart"
  done
fi

ss -ltn "sport = :${ADDR##*:}" 2>/dev/null | tail -n +2 | grep -q . &&
  release_die "API port ${ADDR##*:} is already in use"
for i in "${!hosts[@]}"; do
  release_ssh "${hosts[$i]}" \
    "! ss -ltn 'sport = :$EXPERT_PORT' 2>/dev/null | tail -n +2 | grep -q ." ||
    release_die "${hosts[$i]}:$EXPERT_PORT is already in use; stop the development worker first"
done

cleanup() {
  docker rm -f "$coordinator" >/dev/null 2>&1 || true
  for i in "${!hosts[@]}"; do release_ssh "${hosts[$i]}" "docker rm -f '${spark_prefix}-${hosts[$i]}-${EXPERT_PORT}' >/dev/null 2>&1 || true" || true; done
}
trap cleanup EXIT

placement_directory=
# A placement plan is needed whenever the coordinator keeps local routed experts
# the workers must not also reserve. The dual-RTX path always does; an explicit
# topology on ONE RTX does when it is given an explicit local count in 1..=39.
# `auto` and `0` have no local boundary to publish on 1 RTX and keep the legacy
# no-handoff launch, so the legacy TP4 default and the 2-RTX auto handoff are
# unchanged. The daemon computes the real boundary from the local device only; it
# does not read or connect the remote transport to do so.
release_local_expert_count=""
if [[ "$RTX_EXPERT_LAYERS" =~ ^([1-9]|[1-3][0-9])$ ]]; then release_local_expert_count="$RTX_EXPERT_LAYERS"; fi
if ((RELEASE_RTX_GPUS == 2)) || { ((topology_explicit)) && [[ -n "$release_local_expert_count" ]]; }; then
  placement_directory=/run/ds41rt-placement
fi

# Optional RDMA tuning values travel to both roles only when the operator sets
# them, so a multi-homed six-rank launch can pin the rail without changing any
# default. Values were format-checked above by release_validate_verbs_device_map.
rdma_env_args=()
# Diagnostic switch: DS41RT_STAGE_CHAIN=0 restores host-drained target stages.
[[ -z "${DS41RT_STAGE_CHAIN:-}" ]] || rdma_env_args+=(-e "DS41RT_STAGE_CHAIN=$DS41RT_STAGE_CHAIN")
for rdma_env_name in DS41RT_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP DS41RT_VERBS_APP_IB_PORT_NUM DS41RT_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES; do
  [[ -n "${!rdma_env_name:-}" ]] && rdma_env_args+=(-e "$rdma_env_name=${!rdma_env_name}")
done

start_coordinator() {
echo "== starting native RTX coordinator =="
local -a args=(serve-native --snapshot "/root/.cache/huggingface/$snapshot_rel" --native-lib /opt/ds41rt/lib/libds41rt_native.so --peers "$peers" --rtx-gpus "$RELEASE_RTX_GPUS" --listen "$ADDR" --prefill-batch-tokens "$PREFILL_BATCH_TOKENS" --concurrency "$CONCURRENCY" --prefix-cache-entries "$PREFIX_CACHE_ENTRIES" --max-context-tokens "$MAX_CONTEXT_TOKENS" --max-output-tokens "$MAX_OUTPUT_TOKENS")
args+=(--http-queue-depth "${HTTP_QUEUE_DEPTH:-$CONCURRENCY}" --http-queue-wait-ms "$HTTP_QUEUE_WAIT_MS")
[[ "$RTX_EXPERT_LAYERS" == auto ]] || args+=(--rtx-expert-layers "$RTX_EXPERT_LAYERS")
[[ "$HOST_CACHE_BYTES" == 0 ]] || args+=(--host-cache-bytes "$HOST_CACHE_BYTES")
[[ -z "$KV_POOL_SIZE" ]] || args+=(--kv-pool-size "$KV_POOL_SIZE")
[[ -z "$MEMORY_RESERVATION" ]] || args+=(--memory-reservation "$MEMORY_RESERVATION")
[[ "$DSPARK" != on ]] || args+=(--dspark)
[[ "$DSPARK" != on || "$DSPARK_DRAFT_POLICY" != full ]] || args+=(--dspark-fixed)
[[ -z "$dspark_draft_limit" ]] || args+=(--dspark-draft-limit "$dspark_draft_limit")
[[ "$TP2_ATTENTION" != on ]] || args+=(--tp2-attention)
[[ "$TP2_QUERY_PROJECTION" != on ]] || args+=(--tp2-query-projection)
[[ "$TP2_OUTPUT_PROJECTION" != on ]] || args+=(--tp2-output-projection)
[[ "$TP2_DSPARK_EXPERTS" != on ]] || args+=(--tp2-dspark-experts)
[[ "$spark_exl3_identity" != paired:* ]] || args+=(--exl3-paired-tp4)
# Explicit replicated topology is described to the coordinator once, and each
# worker derives group=rank/TP and tp_rank=rank%TP from the same two flags.
if ((topology_explicit)); then
  args+=(--spark-tp "$spark_tp" --spark-ep "$spark_ep")
fi
[[ -z "$placement_directory" ]] || args+=(--placement-directory "$placement_directory")
docker run -d --name "$coordinator" --restart no --gpus "$gpu_request" --network host --ipc host --ulimit memlock=-1:-1 --device=/dev/infiniband \
  -e "CUDA_VISIBLE_DEVICES=$gpu_uuid_csv" \
  -e "DS41RT_RELEASE_CONFIG_SHA256=$fingerprint" -e "RUST_LOG=${RUST_LOG:-info}" \
  "${rdma_env_args[@]}" \
  -v "$hf_home:/root/.cache/huggingface:ro" "$COORDINATOR_DOCKER_INFERENCE" ds41rt "${args[@]}" >/dev/null
}
deadline=$((SECONDS + ${DS41RT_RELEASE_READY_TIMEOUT_SECONDS:-900}))
if [[ -n "$placement_directory" ]]; then
  start_coordinator
  until placement_plan="$(docker exec "$coordinator" cat "$placement_directory/plan.json" 2>/dev/null)"; do
    [[ "$(docker inspect -f '{{.State.Status}}' "$coordinator" 2>/dev/null)" == running ]] || { docker logs --tail 100 "$coordinator" >&2 || true; release_die "coordinator exited before placement publication"; }
    ((SECONDS < deadline)) || release_die "timed out waiting for coordinator placement"
    sleep 1
  done
  # Accept the coordinator's actual RTX count (1 or 2) and require it to match
  # the layout this launch selected.
  spark_first_layer="$(jq -er --argjson gpus "$RELEASE_RTX_GPUS" '
    select(.version == 1)
    | select((.rtx_gpus | type) == "number" and .rtx_gpus == $gpus)
    | select((.nonce | type) == "string" and (.nonce | length) > 0)
    | select((.rtx_expert_layers | type) == "number")
    | select(.rtx_expert_layers == (.rtx_expert_layers | floor) and .rtx_expert_layers >= 1 and .rtx_expert_layers <= 40)
    | select(.spark_first_layer == ([.rtx_expert_layers, 39] | min))
    | .spark_first_layer' <<<"$placement_plan")" || release_die "invalid coordinator placement plan"
  echo "  runtime placement: RTX GPUs $(jq -r '.rtx_gpus' <<<"$placement_plan"), RTX layers $(jq -r '.rtx_expert_layers' <<<"$placement_plan"); Spark first layer $spark_first_layer"
  if ((topology_explicit)); then
    # Re-check against the boundary the coordinator actually published; this is
    # the dynamic value, not the auto placeholder.
    spark_admission="$(release_validate_spark_weight_admission "$spark_first_layer" "$spark_tp" "$SPARK_DEVICE_BUDGET_BYTES")"
    echo "  runtime Spark weight admission (workspace/staging NOT accounted): $spark_admission"
  fi
fi

echo "== starting native Spark experts =="
pids=()
for i in "${!hosts[@]}"; do
  host="${hosts[$i]}"; remote="${spark_prefix}-${host}-${EXPERT_PORT}"
  # ssh joins its arguments into one remote command line, which drops empty
  # arguments and shifts every later position; quote each one explicitly.
  remote_args=("$SPARK_EXPERT_DOCKER_INFERENCE" "$remote" "$i" "$expert_capacity" "$SPARK_DEVICE_BUDGET_BYTES" "$EXPERT_PORT" "$snapshot_rel" "$fingerprint" "$spark_first_layer" "$SPARK_COUNT" "$topology_explicit" "$spark_tp" "$spark_ep" "${DS41RT_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP:-}" "${DS41RT_VERBS_APP_IB_PORT_NUM:-}" "${DS41RT_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES:-}" "${RUST_LOG:-info}")
  release_ssh "$host" "bash -s -- $(printf '%q ' "${remote_args[@]}")" <<'REMOTE' &
set -euo pipefail
image="$1"; name="$2"; rank="$3"; capacity="$4"; budget="$5"; port="$6"; snapshot_rel="$7"; fingerprint="$8"; first_layer="$9"; world="${10}"
# Defaults keep a legacy invocation (ten positional arguments) valid.
topology_explicit="${11:-0}"; topology_tp="${12:-}"; topology_ep="${13:-}"
topology_args=()
[[ "$topology_explicit" != 1 ]] || topology_args=(--spark-tp "$topology_tp" --spark-ep "$topology_ep")
# Optional RDMA tuning, forwarded only when the operator set it.
rdma_env="${14:-}"; ib_port="${15:-}"; execution_lanes="${16:-}"
# The operator's RUST_LOG reaches the worker too (a quoted heredoc does not
# see the caller's environment).
rust_log="${17:-info}"
rdma_args=()
[[ -z "$rdma_env" ]] || rdma_args+=(-e "DS41RT_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP=$rdma_env")
[[ -z "$ib_port" ]] || rdma_args+=(-e "DS41RT_VERBS_APP_IB_PORT_NUM=$ib_port")
[[ -z "$execution_lanes" ]] || rdma_args+=(-e "DS41RT_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES=$execution_lanes")
hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
# A fixed INFO default makes the worker's structured startup evidence
# (`rank`/`world`/`role`/`intermediate`) observable; without it EnvFilter is
# ERROR and the readiness line never reaches the container log. This adds no
# positional argument, so the worker argument contract is unchanged.
docker run -d --name "$name" --restart no --gpus all --network host --ipc host --ulimit memlock=-1:-1 --device=/dev/infiniband -e "DS41RT_RELEASE_CONFIG_SHA256=$fingerprint" -e "RUST_LOG=$rust_log" "${rdma_args[@]}" -v "$hf_home:/root/.cache/huggingface:ro" "$image" ds41rt expertd-native --snapshot "/root/.cache/huggingface/$snapshot_rel" --native-lib /opt/ds41rt/lib/libds41rt_native.so --rank "$rank" --world "$world" --capacity "$capacity" --device-budget-bytes "$budget" --first-layer "$first_layer" --listen "0.0.0.0:$port" "${topology_args[@]}" >/dev/null
REMOTE
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

for i in "${!hosts[@]}"; do
  host="${hosts[$i]}"; remote="${spark_prefix}-${host}-${EXPERT_PORT}"
  until release_ssh "$host" "test \"\$(docker inspect -f '{{.State.Status}}' '$remote' 2>/dev/null)\" = running && timeout 1 bash -c '</dev/tcp/127.0.0.1/$EXPERT_PORT'" 2>/dev/null; do
    ((SECONDS < deadline)) || { release_ssh "$host" "docker logs --tail 100 '$remote'" >&2 || true; release_die "$host native expert did not become ready"; }
    sleep 1
  done
done

# Acknowledge the published boundary for every handoff launch, including the
# single-RTX explicit-topology case the coordinator now publishes. Only a launch
# with no handoff starts the coordinator at this point.
if [[ -n "$placement_directory" ]]; then
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
