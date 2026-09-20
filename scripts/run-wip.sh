#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repo_root/scripts/release-common.sh"
launcher_started_ns="$(date +%s%N)"
launcher_phase_started_ns="$launcher_started_ns"

report_wip_startup_phase() {
  local stage="$1" now_ns elapsed_ms total_ms
  now_ns="$(date +%s%N)"
  elapsed_ms=$(((now_ns - launcher_phase_started_ns) / 1000000))
  total_ms=$(((now_ns - launcher_started_ns) / 1000000))
  echo "wip_launcher_startup_phase stage=$stage elapsed_ms=$elapsed_ms total_ms=$total_ms" >&2
  launcher_phase_started_ns="$now_ns"
}

usage() {
  cat <<'EOF'
Usage: ./run.sh --wip [--wip-slot NAME] [--config FILE] [--restart] [--dry-run]
                    [--dspark-off | --dspark-shadow-trace]
                    [--execution-lanes N]
                    [--allow-development-unqualified-exl3]

Runs a named slot inside the five persistent WIP development containers.
Without --config, the configuration frozen into the coordinator slot is used.
No source synchronization, compilation, image creation, or container
recreation is performed here; use wip.sh for those operations.
With --restart, exact fingerprint-matched resident Spark experts are retained;
otherwise all affected processes are restarted.
The development-unqualified override is WIP-only and must be stated on every
launch; release launchers never accept it.
The dSpark shadow trace disables active speculation for the coordinator and
records all five proposal positions against the unchanged target trajectory.
The dSpark-off override runs only the target model without changing the slot's
frozen production configuration.
The execution-lane override narrows coordinator admission for lane-local
correctness and performance diagnostics without changing Spark TP placement.
EOF
}

config="$repo_root/ds41rt.config"
config_explicit=0
slot=current
restart=0
dry_run=0
dspark_off=0
dspark_shadow_trace=0
execution_lanes_override=
allow_development_unqualified_exl3=0
wip_api_ready_timeout_secs="${DS41RT_WIP_API_READY_TIMEOUT_SECS:-900}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --wip)
      shift
      ;;
    --wip-slot)
      slot="${2:?--wip-slot requires a name}"
      shift 2
      ;;
    --config)
      config="${2:?$1 requires a configuration file}"
      config_explicit=1
      shift 2
      ;;
    --restart)
      restart=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --dspark-shadow-trace)
      dspark_shadow_trace=1
      shift
      ;;
    --dspark-off)
      dspark_off=1
      shift
      ;;
    --execution-lanes)
      execution_lanes_override="${2:?--execution-lanes requires a value}"
      shift 2
      ;;
    --allow-development-unqualified-exl3)
      allow_development_unqualified_exl3=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      release_die "unknown WIP run argument: $1"
      ;;
  esac
done

[[ "$slot" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] ||
  release_die "invalid WIP slot name: $slot"
((restart == 0 || dry_run == 0)) ||
  release_die "--restart and --dry-run are mutually exclusive"
((dspark_off == 0 || dspark_shadow_trace == 0)) ||
  release_die "--dspark-off and --dspark-shadow-trace are mutually exclusive"
if [[ -n "$execution_lanes_override" ]] &&
  [[ ! "$execution_lanes_override" =~ ^[1-8]$ ]]; then
  release_die "--execution-lanes must be an integer in 1..8"
fi
[[ "$wip_api_ready_timeout_secs" =~ ^[1-9][0-9]*$ ]] ||
  release_die "DS41RT_WIP_API_READY_TIMEOUT_SECS must be a positive integer"

release_need docker
release_need ssh
release_need jq
release_need curl
release_need ss
release_need nvidia-smi
release_need sha256sum
release_need python3
docker info >/dev/null 2>&1 || release_die "local Docker daemon is unavailable"

coordinator_container=ds41rt-coordinator-wip
spark_container=ds41rt-spark-expert-wip
docker container inspect "$coordinator_container" >/dev/null 2>&1 ||
  release_die "persistent coordinator WIP container is missing; run ./wip.sh --slot '$slot'"
[[ "$(docker inspect -f '{{.State.Running}}' "$coordinator_container")" == true ]] ||
  release_die "persistent coordinator WIP container is stopped; run ./wip.sh --slot '$slot'"

# Bootstrap topology from the operator configuration, then prefer the exact
# configuration frozen with the slot unless the caller selected one explicitly.
release_load_config "$config"
state_dir="$repo_root/.ds41rt-wip/run"
mkdir -p "$state_dir"
if ((config_explicit == 0)); then
  slot_config="$state_dir/${slot}.config"
  docker exec "$coordinator_container" \
    cat "/wip/slots/$slot/coordinator/workspace/ds41rt.config" >"$slot_config" ||
    release_die "coordinator WIP slot is missing its frozen ds41rt.config: $slot"
  config="$slot_config"
  release_load_config "$config"
fi
[[ "$SPARK_COUNT" != 2 ]] || release_die "legacy WIP launcher does not support the EXL3 compact Spark TP2 path; use run.sh or runs/v7q-a1/serve-diffbot.sh"
[[ "$SPARK_COUNT" == 4 ]] ||
  release_die "legacy WIP launcher runs exactly four persistent Spark experts; SPARK_COUNT=$SPARK_COUNT requires the release launcher (run.sh)"
# The legacy phase0 expert backend implements the fixed four-plane TP4 wire, not
# the replicated TP×EP contract. An explicit topology must go through the
# native path (run.sh with candidate artifacts), never this legacy launcher.
if release_spark_topology_explicit; then
  release_die "run-wip.sh's legacy phase0 backend does not implement the replicated TP×EP wire; launch the native path with run.sh (TP$SPARK_TP EP$SPARK_EP), not the legacy WIP launcher"
fi
report_wip_startup_phase bootstrap

hosts_csv="$(release_hosts_csv)"
lane_a_csv="$(release_lane_a_csv)"
lane_b_csv="$(release_lane_b_csv)"
expert_hosts_csv="$(release_expert_hosts_csv)"
mapfile -t wip_hosts < <(release_spark_values HOST)
wip_spark_tp="$(release_spark_tp)"
wip_spark_ep="$(release_spark_ep)"
hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$hf_home"
release_resolve_local_model_revision "$hf_home"
release_resolve_coordinator_gpu_identity
coordinator_workspace="/wip/slots/$slot/coordinator/workspace"
expert_workspace="/wip/slots/$slot/spark-expert/workspace"
coordinator_process="coordinator-${ADDR##*:}"
expert_process="expert-$EXPERT_PORT"

validate_local_slot() {
  local image_id
  image_id="$(docker image inspect -f '{{.Id}}' "$COORDINATOR_DOCKER_DEV" 2>/dev/null || true)"
  [[ -n "$image_id" ]] || release_die "missing development image $COORDINATOR_DOCKER_DEV; run ./build.sh"
  [[ "$(docker inspect -f '{{.Image}}' "$coordinator_container")" == "$image_id" ]] ||
    release_die "$coordinator_container uses an old development image; run ./wip.sh --recreate"
  docker exec -i "$coordinator_container" bash -s -- \
    "$slot" coordinator "$image_id" <<'CONTAINER'
set -euo pipefail
slot="$1"
role="$2"
image_id="$3"
root="/wip/slots/$slot/$role"
test -s "$root/META.json"
test -s "$root/FINGERPRINT"
test -x "$root/workspace/.ds41rt-wip/ds41rt"
test -s "$root/workspace/.ds41rt-wip/libds41rt_native.so"
actual_fingerprint="$(sha256sum "$root/META.json" | awk '{print $1}')"
test "$actual_fingerprint" = "$(<"$root/FINGERPRINT")"
python3 "$root/workspace/scripts/verify-release-source-manifest.py" \
  --source "$root/workspace" --manifest "$root/SOURCE_SHA256SUMS" >&2
(
  cd "$root/workspace/.ds41rt-wip"
  sha256sum -c ARTIFACT_SHA256SUMS >&2
)
python3 - "$root/META.json" "$image_id" "$root" <<'PY'
import hashlib
import json
import pathlib
import sys

meta_path, image_id, root = sys.argv[1:]
meta = json.loads(pathlib.Path(meta_path).read_text())
assert meta["schema"] == 1
assert meta["base_image_id"] == image_id, (meta["base_image_id"], image_id)
source_sum = hashlib.sha256(pathlib.Path(root, "SOURCE_SHA256SUMS").read_bytes()).hexdigest()
artifact_sum = hashlib.sha256(pathlib.Path(root, "workspace/.ds41rt-wip/ARTIFACT_SHA256SUMS").read_bytes()).hexdigest()
assert meta["source_manifest_sha256"] == source_sum
assert meta["artifact_manifest_sha256"] == artifact_sum
PY
cat "$root/FINGERPRINT"
CONTAINER
}

validate_remote_slot() {
  local host="$1"
  ssh -o BatchMode=yes "$host" bash -s -- \
    "$spark_container" "$SPARK_EXPERT_DOCKER_DEV" "$slot" \
    "$RELEASE_MODEL_ID" "$RELEASE_MODEL_REVISION" "$EXPERT_FORMAT" \
    "$allow_development_unqualified_exl3" <<'REMOTE'
set -euo pipefail
container="$1"
image="$2"
slot="$3"
model_id="$4"
model_revision="$5"
expert_format="$6"
allow_development_unqualified_exl3="$7"
test "$(docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null || true)" = true
image_id="$(docker image inspect -f '{{.Id}}' "$image")"
container_image_id="$(docker inspect -f '{{.Image}}' "$container")"
test "$container_image_id" = "$image_id"
slot_fingerprint="$(docker exec -i "$container" bash -s -- "$slot" "$image_id" <<'CONTAINER'
set -euo pipefail
slot="$1"
image_id="$2"
root="/wip/slots/$slot/spark-expert"
test -s "$root/META.json"
test -s "$root/FINGERPRINT"
test -x "$root/workspace/.ds41rt-wip/ds41rt"
test -s "$root/workspace/.ds41rt-wip/libds41rt_native.so"
actual_fingerprint="$(sha256sum "$root/META.json" | awk '{print $1}')"
test "$actual_fingerprint" = "$(<"$root/FINGERPRINT")"
python3 "$root/workspace/scripts/verify-release-source-manifest.py" \
  --source "$root/workspace" --manifest "$root/SOURCE_SHA256SUMS" >&2
(
  cd "$root/workspace/.ds41rt-wip"
  sha256sum -c ARTIFACT_SHA256SUMS >&2
)
python3 - "$root/META.json" "$image_id" "$root" <<'PY'
import hashlib
import json
import pathlib
import sys

meta_path, image_id, root = sys.argv[1:]
meta = json.loads(pathlib.Path(meta_path).read_text())
assert meta["schema"] == 1
assert meta["base_image_id"] == image_id, (meta["base_image_id"], image_id)
source_sum = hashlib.sha256(pathlib.Path(root, "SOURCE_SHA256SUMS").read_bytes()).hexdigest()
artifact_sum = hashlib.sha256(pathlib.Path(root, "workspace/.ds41rt-wip/ARTIFACT_SHA256SUMS").read_bytes()).hexdigest()
assert meta["source_manifest_sha256"] == source_sum
assert meta["artifact_manifest_sha256"] == artifact_sum
PY
cat "$root/FINGERPRINT"
CONTAINER
)"
hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
model_root="$hf_home/hub/models--${model_id//\//--}"
if [[ -z "$model_revision" ]]; then
  test -s "$model_root/refs/main"
  model_revision="$(<"$model_root/refs/main")"
fi
[[ "$model_revision" =~ ^[0-9a-f]{40,64}$ ]]
model_snapshot="$model_root/snapshots/$model_revision"
test -d "$model_snapshot"
! find "$model_snapshot" -xtype l -print -quit | grep -q .
if [[ "$expert_format" == exl3 ]]; then
  validator="/wip/slots/$slot/spark-expert/workspace/python/tools/validate_ds4_staged_snapshot.py"
  validator_args=(python3 "$validator" \
    --checkpoint "$model_snapshot" \
    --model-id "$model_id" \
    --revision "$model_revision" \
    --startup-contract-only)
  if [[ "$allow_development_unqualified_exl3" == 1 ]]; then
    validator_args+=(--allow-development-unqualified)
  fi
  docker exec "$container" "${validator_args[@]}" >/dev/null
fi
printf '%s %s\n' "$slot_fingerprint" "$model_revision"
REMOTE
}

echo "== validating persistent WIP containers and slot $slot =="
validation_dir="$(mktemp -d "$state_dir/slot-validation.XXXXXX")"
validation_labels=(coordinator)
validation_pids=()
validate_local_slot >"$validation_dir/0.out" 2>"$validation_dir/0.err" &
validation_pids+=("$!")
validation_index=0
for host in "${wip_hosts[@]}"; do
  validation_index=$((validation_index + 1))
  validation_labels+=("$host")
  validate_remote_slot "$host" \
    >"$validation_dir/$validation_index.out" \
    2>"$validation_dir/$validation_index.err" &
  validation_pids+=("$!")
done
validation_failed=0
for validation_index in "${!validation_pids[@]}"; do
  if ! wait "${validation_pids[$validation_index]}"; then
    echo "${validation_labels[$validation_index]} WIP slot validation failed: $slot" >&2
    validation_failed=1
  fi
  cat "$validation_dir/$validation_index.err" >&2
done
if ((validation_failed)); then
  rm -rf "$validation_dir"
  release_die "one or more WIP slot validations failed: $slot"
fi
coordinator_slot_fingerprint="$(<"$validation_dir/0.out")"
expert_slot_fingerprint=
expert_model_revision=
validation_index=0
for host in "${wip_hosts[@]}"; do
  validation_index=$((validation_index + 1))
  read -r remote_fingerprint remote_model_revision extra \
    <"$validation_dir/$validation_index.out" ||
    release_die "$host returned no expert slot identity"
  [[ -z "${extra:-}" && "$remote_fingerprint" =~ ^[0-9a-f]{64}$ &&
    "$remote_model_revision" =~ ^[0-9a-f]{40,64}$ ]] ||
    release_die "$host returned an invalid expert slot identity"
  if [[ -z "$expert_slot_fingerprint" ]]; then
    expert_slot_fingerprint="$remote_fingerprint"
    expert_model_revision="$remote_model_revision"
  else
    [[ "$remote_fingerprint" == "$expert_slot_fingerprint" ]] ||
      release_die "$host has a different Spark WIP slot fingerprint"
    [[ "$remote_model_revision" == "$expert_model_revision" ]] ||
      release_die "$host has a different cached text-model revision"
  fi
  echo "  $host: persistent container and expert slot ready"
done
rm -rf "$validation_dir"
coordinator_engine_commit="wip-${slot}-${coordinator_slot_fingerprint:0:12}-${expert_slot_fingerprint:0:12}"
coordinator_image_id="$(docker image inspect -f '{{.Id}}' "$COORDINATOR_DOCKER_DEV")"
report_wip_startup_phase slot-validation

settings_args=(
  --repo-root "$coordinator_workspace"
  --model-id "$MODEL_ID"
  --model-variant "$MODEL_VARIANT"
  --expert-format "$EXPERT_FORMAT"
  --dspark "$DSPARK"
  --dspark-draft-policy "$DSPARK_DRAFT_POLICY"
  --coordinator-gpu "$COORDINATOR_GPU"
  --coordinator-gpu-uuid "$RELEASE_COORDINATOR_GPU_UUID"
  --coordinator-gpu-pci-bus-id "$RELEASE_COORDINATOR_GPU_PCI_BUS_ID"
  --headroom-gib "$COORDINATOR_GPU_HEADROOM_GIB"
  --concurrency "$CONCURRENCY"
  --spark-reduction-min-rows "$SPARK_REDUCTION_MIN_ROWS"
  --dry-run
)
[[ -z "$KV_POOL_TOKENS" ]] || settings_args+=(--kv-pool-tokens "$KV_POOL_TOKENS")
[[ -z "$MAX_CONTEXT_TOKENS" ]] || settings_args+=(--max-context-tokens "$MAX_CONTEXT_TOKENS")
[[ -z "$MAX_OUTPUT_TOKENS" ]] || settings_args+=(--max-output-tokens "$MAX_OUTPUT_TOKENS")

resolved_json="$state_dir/resolved-settings.json"
docker exec \
  -e PYTHONPATH="$coordinator_workspace/third_party/sparkinfer:$coordinator_workspace/python/reference/ds41rt_reference:$coordinator_workspace/python/reference:/opt/ds41rt/third_party/sparkinfer" \
  -w "$coordinator_workspace" \
  "$coordinator_container" \
  python3 "$coordinator_workspace/python/tools/resolve_serve_settings.py" \
  "${settings_args[@]}" >"$resolved_json"
jq -e . "$resolved_json" >/dev/null || release_die "settings resolver returned invalid JSON"
resolved_dspark_draft_policy="$(
  jq -er '.dspark_draft_policy | select(type == "string")' "$resolved_json"
)" || release_die "WIP slot settings resolver did not report its dSpark draft policy; rebuild the slot"
[[ "$resolved_dspark_draft_policy" == "$DSPARK_DRAFT_POLICY" ]] ||
  release_die "WIP slot settings resolver draft policy mismatch: requested $DSPARK_DRAFT_POLICY, resolved $resolved_dspark_draft_policy"
resolved_fixed_drafts="$(
  jq -r '.environment.DS41RT_REAL_FULL_DSPARK_FIXED_DRAFTS // ""' "$resolved_json"
)"
if [[ "$DSPARK:$DSPARK_DRAFT_POLICY" == on:full ]]; then
  [[ "$resolved_fixed_drafts" == 5 ]] ||
    release_die "WIP slot full dSpark policy did not resolve exactly five proposals; rebuild the slot"
else
  [[ -z "$resolved_fixed_drafts" ]] ||
    release_die "WIP slot disabled/adaptive dSpark unexpectedly resolved a fixed proposal width"
fi
blockers="$(jq -r '.blockers[]?' "$resolved_json")"
[[ -z "$blockers" ]] || release_die "launch blockers:\n$blockers"

config_sha256="$(sha256sum "$config" | awk '{print $1}')"
expert_identity_settings=()
for wip_index in "${!wip_hosts[@]}"; do
  wip_host="${wip_hosts[$wip_index]}"
  wip_lane_a_name="SPARK_${wip_index}_LANE_A"
  wip_lane_b_name="SPARK_${wip_index}_LANE_B"
  expert_identity_settings+=(
    --setting "spark_${wip_index}=${wip_host},${!wip_lane_a_name},${!wip_lane_b_name}"
  )
done
# Bind the expert identity to the resolved topology, not just the host list.
expert_identity_settings+=(
  --setting "spark_tp=$(release_spark_tp)"
  --setting "spark_ep=$(release_spark_ep)"
)
expert_runtime_fingerprint="$(
  python3 "$repo_root/scripts/wip-expert-runtime-identity.py" \
    --resolved-settings "$resolved_json" \
    --expert-slot-fingerprint "$expert_slot_fingerprint" \
    --setting "model_id=$RELEASE_MODEL_ID" \
    --setting "model_revision=$expert_model_revision" \
    --setting "model_variant=$MODEL_VARIANT" \
    --setting "expert_format=$EXPERT_FORMAT" \
    --setting "allow_historical_exl3_control=$allow_development_unqualified_exl3" \
    --setting "sparkinfer_exl3=$SPARKINFER_EXL3" \
    --setting "expert_port=$EXPERT_PORT" \
    --setting "expert_image=$SPARK_EXPERT_DOCKER_DEV" \
    --setting "runtime_cache=/wip/cache" \
    --setting "transport=verbs-host" \
    "${expert_identity_settings[@]}"
)"
deployment_fingerprint="$({
  printf 'ds41rt-wip-deployment-v2\n'
  jq -S . "$resolved_json"
  printf '%s\n' \
    "$config_sha256" "$coordinator_slot_fingerprint" \
    "$expert_runtime_fingerprint" "$ADDR" "$coordinator_engine_commit"
} | sha256sum | awk '{print $1}')"
report_wip_startup_phase settings-resolution

process_status_local() {
  docker exec "$coordinator_container" \
    "$coordinator_workspace/scripts/wip-process.sh" status "$coordinator_process" 2>/dev/null || echo stopped
}

process_identity_local() {
  docker exec "$coordinator_container" \
    "$coordinator_workspace/scripts/wip-process.sh" identity "$coordinator_process" \
    2>/dev/null || true
}

inspect_remote_service_state() {
  local host="$1"
  local release_container="${RELEASE_SPARK_CONTAINER_PREFIX}-${host}-${EXPERT_PORT}"
  local legacy_container="ds41rt-phase0-tcp-expertd-${host}-${EXPERT_PORT}"
  ssh -o BatchMode=yes "$host" bash -s -- \
    "$release_container" "$legacy_container" "$spark_container" \
    "$expert_workspace/scripts/wip-process.sh" "$expert_process" <<'REMOTE'
set -euo pipefail
release_container="$1"
legacy_container="$2"
wip_container="$3"
wip_process_script="$4"
expert_process="$5"
standard_active=0
for container in "$release_container" "$legacy_container"; do
  [[ "$(docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null || true)" != true ]] ||
    standard_active=1
done
wip_active=0
wip_identity=-
if [[ "$(docker inspect -f '{{.State.Running}}' "$wip_container" 2>/dev/null || true)" == true ]]; then
  wip_status="$(docker exec "$wip_container" "$wip_process_script" status "$expert_process")"
  case "$wip_status" in
    running\ *)
      wip_active=1
      wip_identity="$(docker exec "$wip_container" "$wip_process_script" identity "$expert_process" 2>/dev/null || true)"
      [[ -n "$wip_identity" ]] || wip_identity=-
      ;;
    stopped) ;;
    *) echo "invalid WIP process status for $expert_process: $wip_status" >&2; exit 2 ;;
  esac
fi
printf '%s %s %s\n' "$standard_active" "$wip_active" "$wip_identity"
REMOTE
}

standard_services=0
[[ "$(docker inspect -f '{{.State.Running}}' "$RELEASE_COORDINATOR_CONTAINER_NAME" 2>/dev/null || true)" != true ]] || standard_services=1
wip_coordinator_running=0
[[ "$(process_status_local)" == running\ * ]] && wip_coordinator_running=1
wip_experts_running=0
expert_fingerprints_match=1
service_state_dir="$(mktemp -d "$state_dir/service-state.XXXXXX")"
service_state_hosts=()
service_state_pids=()
for host in "${wip_hosts[@]}"; do
  inspect_remote_service_state "$host" \
    >"$service_state_dir/$host.out" 2>"$service_state_dir/$host.err" &
  service_state_hosts+=("$host")
  service_state_pids+=("$!")
done
service_state_failed=0
for service_state_index in "${!service_state_pids[@]}"; do
  host="${service_state_hosts[$service_state_index]}"
  if ! wait "${service_state_pids[$service_state_index]}"; then
    echo "$host service-state inspection failed" >&2
    service_state_failed=1
  elif ! read -r remote_standard_active remote_wip_active remote_wip_identity extra \
    <"$service_state_dir/$host.out"; then
    echo "$host returned no service-state data" >&2
    service_state_failed=1
  else
    if [[ -n "${extra:-}" || ! "$remote_standard_active" =~ ^[01]$ || ! "$remote_wip_active" =~ ^[01]$ ]] ||
      [[ "$remote_wip_identity" != - && ! "$remote_wip_identity" =~ ^[0-9a-f]{64}$ ]]; then
      echo "$host returned invalid service-state data" >&2
      service_state_failed=1
    else
      [[ "$remote_standard_active" == 0 ]] || standard_services=1
      [[ "$remote_wip_active" == 0 ]] || ((wip_experts_running += 1))
      [[ "$remote_wip_identity" == "$expert_runtime_fingerprint" ]] ||
        expert_fingerprints_match=0
    fi
  fi
  cat "$service_state_dir/$host.err" >&2
done
rm -rf "$service_state_dir"
((service_state_failed == 0)) || release_die "one or more Spark service-state inspections failed"
services_active=$((standard_services + wip_coordinator_running + wip_experts_running))
reuse_spark_experts=0

if ((services_active)); then
  if ((!restart)); then
    current_fingerprint="$(process_identity_local)"
    if ((standard_services == 0 && wip_coordinator_running == 1 && wip_experts_running == SPARK_COUNT && expert_fingerprints_match)) &&
      [[ "$current_fingerprint" == "$deployment_fingerprint" ]] &&
      release_api_advertises_model \
        "http://127.0.0.1:${ADDR##*:}" "$RELEASE_MODEL_ID"; then
      echo "All five WIP services already match slot '$slot' and the selected configuration."
      exit 0
    fi
    release_die "partial, standard, or configuration-mismatched service state is active; use --restart"
  fi
  if ((standard_services == 0 && wip_experts_running == SPARK_COUNT)); then
    reuse_spark_experts="$expert_fingerprints_match"
  fi
  if ((reuse_spark_experts)); then
    echo "== reusing four fingerprint-matched resident WIP Spark experts =="
    release_stop_wip_coordinator "$coordinator_process"
  else
    echo "== stopping existing WIP and release services =="
    release_stop_wip_services "$coordinator_process" "$expert_process"
    ((standard_services == 0)) ||
      release_stop_services "$RELEASE_COORDINATOR_CONTAINER_NAME" "$RELEASE_SPARK_CONTAINER_PREFIX"
  fi
fi
report_wip_startup_phase service-reconciliation

check_model_cache_local() {
  local model_id="$1" revision="${2:-}"
  local root="$hf_home/hub/models--${model_id//\//--}"
  [[ -n "$revision" || ! -f "$root/refs/main" ]] || revision="$(<"$root/refs/main")"
  [[ -n "$revision" && -d "$root/snapshots/$revision" ]] ||
    release_die "coordinator model snapshot is missing: $model_id${revision:+@$revision}"
  find "$root/snapshots/$revision" -xtype l -print -quit | grep -q . &&
    release_die "coordinator model snapshot has unresolved blobs: $model_id@$revision"
  if [[ "$EXPERT_FORMAT" == exl3 ]]; then
    validator_args=(python3 "$repo_root/python/tools/validate_ds4_staged_snapshot.py" \
      --checkpoint "$root/snapshots/$revision" \
      --model-id "$model_id" \
      --revision "$revision" \
      --startup-contract-only)
    if ((allow_development_unqualified_exl3)); then
      validator_args+=(--allow-development-unqualified)
    fi
    "${validator_args[@]}" >/dev/null ||
      release_die "coordinator EXL3 snapshot is not a qualified immutable publication: $model_id@$revision"
  fi
  return 0
}

echo "== checking model snapshots =="
check_model_cache_local "$RELEASE_MODEL_ID" "$RELEASE_MODEL_REVISION"
report_wip_startup_phase model-snapshots

echo "== checking launch headroom =="
available_kib="$(awk '/MemAvailable:/{print $2}' /proc/meminfo)"
((available_kib >= 8 * 1024 * 1024)) || release_die "coordinator has less than 8 GiB available system memory"
total_mib=0
free_mib=0
for _ in $(seq 1 50); do
  # Query the coordinator GPU explicitly. With two visible GPUs, piping the
  # multi-line result through `head` intermittently SIGPIPEs nvidia-smi under
  # `set -o pipefail` and aborts a restart after the old service is stopped.
  gpu_line="$(
    nvidia-smi --id="$RELEASE_COORDINATOR_GPU_PCI_BUS_ID" \
      --query-gpu=memory.total,memory.free --format=csv,noheader,nounits
  )"
  IFS=, read -r total_mib free_mib <<<"$gpu_line"
  total_mib="$(release_trim "$total_mib")"
  free_mib="$(release_trim "$free_mib")"
  ((free_mib >= 80 * 1024)) && break
  sleep 0.1
done
((free_mib >= 80 * 1024)) || release_die "coordinator GPU has only ${free_mib} MiB free; 80 GiB is required"
echo "  coordinator: RAM $((available_kib / 1024)) MiB available; GPU ${free_mib}/${total_mib} MiB free"

resource_check_dir="$(mktemp -d "$state_dir/resource-check.XXXXXX")"
resource_check_hosts=()
resource_check_pids=()
for host in "${wip_hosts[@]}"; do
  if ((reuse_spark_experts)); then
    echo "  $host: reusing resident fingerprint-matched expert; launch headroom check skipped"
    continue
  fi
  ssh -o BatchMode=yes "$host" bash -s -- "$spark_container" \
    >"$resource_check_dir/$host.out" 2>"$resource_check_dir/$host.err" <<'REMOTE' &
set -euo pipefail
container="$1"
min_kib=$((105 * 1024 * 1024))
available_kib=0
for _ in $(seq 1 50); do
  # A stopped CUDA process can disappear before Linux has finished returning
  # its unified-memory pages to MemAvailable.  Match the coordinator GPU
  # teardown grace period so a native/EXL3 switch does not fail on that race.
  available_kib="$(awk '/MemAvailable:/{print $2}' /proc/meminfo)"
  ((available_kib >= min_kib)) && break
  sleep 0.1
done
((available_kib >= min_kib)) || { echo "only $((available_kib / 1024)) MiB unified memory is available" >&2; exit 2; }
gpu_line="$(docker exec "$container" nvidia-smi --id=0 --query-gpu=memory.total,memory.free --format=csv,noheader,nounits)"
echo "RAM $((available_kib / 1024)) MiB available; GPU $gpu_line"
REMOTE
  resource_check_hosts+=("$host")
  resource_check_pids+=("$!")
done
resource_check_failed=0
for resource_check_index in "${!resource_check_pids[@]}"; do
  host="${resource_check_hosts[$resource_check_index]}"
  if ! wait "${resource_check_pids[$resource_check_index]}"; then
    echo "$host failed the 105 GiB WIP launch headroom check" >&2
    resource_check_failed=1
  else
    echo "  $host: $(<"$resource_check_dir/$host.out")"
  fi
  cat "$resource_check_dir/$host.err" >&2
done
rm -rf "$resource_check_dir"
((resource_check_failed == 0)) || release_die "one or more Spark launch headroom checks failed"
report_wip_startup_phase launch-headroom
echo "  WIP slot: $slot"
echo "  SparkInfer EXL3 mode: $SPARKINFER_EXL3"
echo "  coordinator GPU: $RELEASE_COORDINATOR_GPU_UUID ($RELEASE_COORDINATOR_GPU_PCI_BUS_ID)"
if ((dry_run)); then
  echo "WIP dry-run checks passed."
  exit 0
fi

eval "$(jq -r '.environment | to_entries[] | "\(.key)=\(.value | @sh); export \(.key)"' "$resolved_json")"
if ((allow_development_unqualified_exl3)); then
  export DS41RT_WIP_ALLOW_HISTORICAL_EXL3_CONTROL=1
else
  unset DS41RT_WIP_ALLOW_HISTORICAL_EXL3_CONTROL || true
fi
export DS41RT_SPARK_HOSTS="$hosts_csv"
export DS41RT_SPARK_TP="$wip_spark_tp"
export DS41RT_SPARK_EP="$wip_spark_ep"
export DS41RT_REAL_FULL_SERVE_EXPERT_HOSTS="$expert_hosts_csv"
export DS41RT_SPARK_IMAGE="$SPARK_EXPERT_DOCKER_DEV"
export DS41RT_SPARK_EXISTING_CONTAINER="$spark_container"
export DS41RT_SPARK_RUNTIME_CACHE_DIR=/wip/cache
export DS41RT_SPARK_WORKDIR="$expert_workspace"
export DS41RT_SPARK_PREBUILT=1
export DS41RT_MODEL_REVISION="$RELEASE_MODEL_REVISION"
export DS41RT_SPARK_PREBUILT_BIN="$expert_workspace/.ds41rt-wip/ds41rt"
export DS41RT_SPARK_PREBUILT_NATIVE_LIB="$expert_workspace/.ds41rt-wip/libds41rt_native.so"
export DS41RT_SPARK_SKIP_STAGE=1
export DS41RT_RELEASE_CONFIG_SHA256="$expert_runtime_fingerprint"
export DS41RT_SPARK_EXPERT_PORT="$EXPERT_PORT"
export DS41RT_SPARK_EXPERT_TRANSPORT=verbs-host
export DS41RT_SPARK_KEEP_EXPERTS=1
export DS41RT_SPARK_EXPERT_REAL_LAYER=all
export DS41RT_PHASE0_SPARK_SKIP_BENCH=1
export DS41RT_EXPERT_INTERMEDIATE_RDMA_PEERS="$lane_a_csv"
if [[ -n "$lane_b_csv" ]]; then
  export DS41RT_EXPERT_INTERMEDIATE_RDMA_ADDITIONAL_PEERS="$lane_b_csv"
else
  unset DS41RT_EXPERT_INTERMEDIATE_RDMA_ADDITIONAL_PEERS || true
fi
spark_start_pid=
if ((reuse_spark_experts)); then
  echo "== retaining WIP Spark expert processes =="
else
  echo "== starting WIP Spark expert processes =="
  "$repo_root/scripts/phase0-spark-tcp-bench.sh" &
  spark_start_pid=$!
fi
report_wip_startup_phase spark-dispatch

env_file="$state_dir/coordinator.env"
jq -r '.environment | to_entries[] | "\(.key)=\(.value)"' "$resolved_json" >"$env_file"
{
  echo "ADDR=$ADDR"
  echo "DS41RT_REAL_FULL_SERVE_EXPERT_HOSTS=$expert_hosts_csv"
  echo "DS41RT_SPARK_HOSTS=$hosts_csv"
  echo "DS41RT_SPARK_TP=$wip_spark_tp"
  echo "DS41RT_SPARK_EP=$wip_spark_ep"
  echo "DS41RT_SPARK_EXPERT_PORT=$EXPERT_PORT"
  echo "DS41RT_SPARKINFER_EXL3=$SPARKINFER_EXL3"
  echo "DS41RT_MODEL_REVISION=$RELEASE_MODEL_REVISION"
  echo "DS41RT_REAL_FULL_SERVE_START_EXPERTS=0"
  echo "DS41RT_REAL_FULL_SERVE_BUILD_DAEMON=0"
  echo "DS41RT_REAL_FULL_SERVE_BUILD_NATIVE=0"
  echo "DS41RT_REAL_FULL_SERVE_REQUIRE_CUDA=1"
  echo "DS41RT_REAL_FULL_SERVE_EXPERT_WARMUP_STATUS_FILE=/wip/run/expert-warmup.status"
  echo "DS41RT_BIN=$coordinator_workspace/.ds41rt-wip/ds41rt"
  echo "DS41RT_NATIVE_LIB=$coordinator_workspace/.ds41rt-wip/libds41rt_native.so"
  echo "DS41RT_ENGINE_COMMIT=$coordinator_engine_commit"
  echo "DS41RT_RELEASE_CONFIG_SHA256=$deployment_fingerprint"
  echo "DS41RT_KERNEL_CACHE_BASE=/wip/cache/kernels"
  echo "DS41RT_KERNEL_CACHE_ENVIRONMENT_ID=$coordinator_image_id"
  echo "DS41RT_RUNTIME_CATALOG_CACHE_DIR=/wip/cache/catalogs"
  if ((allow_development_unqualified_exl3)); then
    echo "DS41RT_WIP_ALLOW_HISTORICAL_EXL3_CONTROL=1"
  fi
  echo "HF_HOME=$hf_home"
  echo "PYTHONPATH=$coordinator_workspace/third_party/sparkinfer:$coordinator_workspace/python/reference/ds41rt_reference:$coordinator_workspace/python/reference:/opt/ds41rt/third_party/sparkinfer"
} >>"$env_file"

if [[ -n "$execution_lanes_override" ]]; then
  echo "DS41RT_REAL_FULL_MAX_EXECUTION_LANES=$execution_lanes_override" >>"$env_file"
fi

if ((dspark_off)); then
  {
    echo "DS41RT_REAL_FULL_DSPARK=0"
    echo "DS41RT_REAL_FULL_DSPARK_SHADOW=0"
  } >>"$env_file"
elif ((dspark_shadow_trace)); then
  {
    echo "DS41RT_REAL_FULL_DSPARK=0"
    echo "DS41RT_REAL_FULL_DSPARK_SHADOW=1"
    echo "DS41RT_REAL_FULL_DSPARK_TRACE=1"
  } >>"$env_file"
fi

# Keep graph-capture tracing opt-in and launcher-scoped. The persistent WIP
# container otherwise receives only the resolved production environment, which
# made an A/B graph-identity audit require manually editing its generated env
# file between restarts.
if [[ -n "${DS41RT_REAL_FULL_GRAPH_CAPTURE_TRACE:-}" ]]; then
  echo "DS41RT_REAL_FULL_GRAPH_CAPTURE_TRACE=$DS41RT_REAL_FULL_GRAPH_CAPTURE_TRACE" \
    >>"$env_file"
fi

# Coordinator-only kernel candidates are WIP launch controls. They do not
# affect Spark identity, and remain absent from production settings until the
# corresponding hardware qualification promotes them.
if [[ -n "${DS41RT_DS4_FLASH_ROUTER_SHORTLIST:-}" ]]; then
  echo "DS41RT_DS4_FLASH_ROUTER_SHORTLIST=$DS41RT_DS4_FLASH_ROUTER_SHORTLIST" \
    >>"$env_file"
fi

if [[ -n "${DS41RT_REAL_FULL_BF16_HIDDEN_READBACK_OVERLAP:-}" ]]; then
  echo "DS41RT_REAL_FULL_BF16_HIDDEN_READBACK_OVERLAP=$DS41RT_REAL_FULL_BF16_HIDDEN_READBACK_OVERLAP" \
    >>"$env_file"
fi

# Native activation and exact row-route capture are likewise explicit WIP
# diagnostics. Keep them out of release settings and expert identity: only the
# coordinator writes them, and callers must request a restart when enabling or
# disabling the capture environment.
for diagnostic_name in \
  B12X_COMPILE_DISK_CACHE \
  B12X_PAGED_INDEX_SUPERTILE_K \
  CUDA_LAUNCH_BLOCKING \
  DS41RT_REAL_FULL_B12X_PACKED_HIDDEN_EXCHANGE \
  DS41RT_REAL_FULL_DIAGNOSTIC_LAYER_DUMP_DIR \
  DS41RT_REAL_FULL_DIAGNOSTIC_LAYER_DUMP_LAYER \
  DS41RT_REAL_FULL_DSPARK_TRACE \
  DS41RT_REAL_FULL_DSPARK_CONFIDENCE_POLICY \
  DS41RT_REAL_FULL_DSPARK_FIXED_DRAFTS \
  DS41RT_REAL_FULL_DSPARK_PROFILE_AT_STARTUP \
  DS41RT_REAL_FULL_SERVE_PREFIX_PREFILL_PROBE \
  DS41RT_REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_REPEATS \
  DS41RT_REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_PREFIX_ROWS \
  DS41RT_REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_NEW_ROWS \
  DS41RT_REAL_FULL_REQUEST_TIMING \
  DS41RT_REAL_FULL_REQUEST_PREFILL_CHUNK_TOKENS \
  DS41RT_REAL_FULL_SCHEDULER_TIMING \
  DS41RT_REAL_FULL_SCHEDULER_SUMMARY_TIMING \
  DS41RT_REAL_FULL_ADMISSION_STAGE_TIMING \
  DS41RT_REAL_FULL_ATTENTION_CUDA_TIMING \
  DS41RT_REAL_FULL_ROLLING_SPARSE_PACKS \
  DS41RT_REAL_FULL_SPARSE_TCP_STAGE_TIMING \
  DS41RT_REAL_FULL_NVFP4_ROUTE_TIMING \
  DS41RT_REAL_FULL_NVFP4_ROUTE_CUDA_EVENT_TIMING \
  DS41RT_REAL_FULL_PROTOCOL_V2_EXECUTOR_TIMING \
  DS41RT_REAL_FULL_TERMINAL_SAMPLE_VALIDATE \
  DS41RT_PROTOCOL_V2_TCP_TIMING \
  DS41RT_PROTOCOL_V2_EXPERT_QUEUE_STATS \
  DS41RT_PROTOCOL_V2_EXPERT_QUEUE_ROW_ROUTES; do
  diagnostic_value="${!diagnostic_name:-}"
  if [[ -n "$diagnostic_value" ]]; then
    printf '%s=%s\n' "$diagnostic_name" "$diagnostic_value" >>"$env_file"
  fi
done

echo "== starting coordinator process in $coordinator_container =="
if ! docker exec -d --env-file "$env_file" -w "$coordinator_workspace" \
  "$coordinator_container" \
  "$coordinator_workspace/scripts/wip-process.sh" run "$coordinator_process" \
  "$coordinator_workspace/scripts/real-full-tcp-serve.sh"; then
  kill "$spark_start_pid" >/dev/null 2>&1 || true
  wait "$spark_start_pid" >/dev/null 2>&1 || true
  release_die "failed to start WIP coordinator process"
fi
report_wip_startup_phase coordinator-dispatch

if [[ -n "$spark_start_pid" ]] && ! wait "$spark_start_pid"; then
  docker exec "$coordinator_container" \
    "$coordinator_workspace/scripts/wip-process.sh" log "$coordinator_process" 200 >&2 || true
  docker exec "$coordinator_container" \
    "$coordinator_workspace/scripts/wip-process.sh" stop "$coordinator_process" || true
  release_die "one or more WIP Spark experts failed during startup"
fi

docker exec "$coordinator_container" \
  "$coordinator_workspace/scripts/wip-process.sh" bind-identity \
  "$coordinator_process" "$deployment_fingerprint"
if ((!reuse_spark_experts)); then
  for host in "${wip_hosts[@]}"; do
    ssh -o BatchMode=yes "$host" \
      "docker exec '$spark_container' '$expert_workspace/scripts/wip-process.sh' bind-identity '$expert_process' '$expert_runtime_fingerprint'"
  done
fi

deadline=$((SECONDS + wip_api_ready_timeout_secs))
until release_api_advertises_model \
  "http://127.0.0.1:${ADDR##*:}" "$RELEASE_MODEL_ID"; do
  coordinator_state="$(process_status_local)"
  if [[ "$coordinator_state" != running\ * ]]; then
    docker exec "$coordinator_container" \
      "$coordinator_workspace/scripts/wip-process.sh" log "$coordinator_process" 200 >&2 || true
    release_die "WIP coordinator process exited during startup"
  fi
  ((SECONDS < deadline)) || {
    docker exec "$coordinator_container" \
      "$coordinator_workspace/scripts/wip-process.sh" log "$coordinator_process" 200 >&2 || true
    release_die "WIP API did not become ready within ${wip_api_ready_timeout_secs} seconds"
  }
  sleep 0.25
done

curl -fsS "http://127.0.0.1:${ADDR##*:}/v1/models" >"$state_dir/models.json"
release_validate_model_list_file "$state_dir/models.json" "$RELEASE_MODEL_ID"
report_wip_startup_phase api-ready
echo "DS41RT WIP server is ready at http://127.0.0.1:${ADDR##*:}/v1/"
echo "  slot:        $slot"
echo "  KV cache:    fp8"
echo "  model:       $RELEASE_MODEL_ID"
echo "  variant:     $MODEL_VARIANT"
echo "  experts:     $EXPERT_FORMAT"
if ((dspark_off)); then
  echo "  dSpark:      off (WIP override)"
elif ((dspark_shadow_trace)); then
  echo "  dSpark:      shadow trace (WIP override)"
else
  if [[ "$DSPARK" == on ]]; then
    echo "  dSpark:      on ($DSPARK_DRAFT_POLICY proposals)"
  else
    echo "  dSpark:      off"
  fi
fi
echo "  EXL3 kernel: $SPARKINFER_EXL3"
echo "  concurrency: $CONCURRENCY"
if [[ -n "$execution_lanes_override" ]]; then
  echo "  exec lanes:  $execution_lanes_override (WIP override)"
fi
echo "  containers:  persistent $coordinator_container + $SPARK_COUNT $spark_container"
echo "  Spark reuse: $([[ "$reuse_spark_experts" == 1 ]] && echo yes || echo no)"
