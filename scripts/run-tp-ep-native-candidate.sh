#!/usr/bin/env bash
# Standalone native TP×EP candidate launcher (isolated experiment only).
#
# STATUS: NOT LAUNCH-READY. Plan/dry-run and the CPU tests may be used now. A
# live start additionally needs the ARM64 `expertd-native` daemon build, the
# uniform `/scratch/candidate/libds41rt_native.so` staged on every rank, a clear
# uniform-artifact copy, and an L3 grant. Do not treat a plan pass as readiness.
#
# Confirmed official snapshot inside every Spark container:
# /root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277
#
# Runs already-built, frozen `ds41rt` daemon + native libraries inside the
# isolated `ds41rt-tpep-*` development containers. It never creates, renames or
# removes a container, never touches a production name
# (`ds41rt-coordinator` / `ds41rt-spark-expert-*`) and never builds an image.
#
# The v8 release fleet is currently stopped by the TP×EP preflight leases; this
# launcher does not restore it. Baseline restoration is the separate explicit
# operation `runs/tp-ep-preflight/restore-release-fleet.sh` on the same original
# containers, not a side effect of this launcher.
#
# Contract:
#   * explicit SPARK_TP/SPARK_EP only, native official checkpoint only;
#   * native CLI: `serve-native` / `expertd-native` with `--spark-tp`,
#     `--spark-ep`, global `--rank` and `--world == TP*EP`;
#   * 2 RTX: the coordinator's plan is published and read (actual rtx_gpus and
#     the real spark_first_layer) BEFORE any worker starts; a stale placement
#     directory is refused unless --restart explicitly clears this run's dir;
#   * 1 RTX: no handoff, workers load from layer 0;
#   * fail closed on filesystem, topology, budget, provenance, arch, stale
#     placement, daemon process death and unreachable hosts;
#   * readiness is the real API health + native model advertisement plus a
#     running candidate process on every rank, never just a live container;
#   * `plan` (default / --dry-run) prints the exact command vectors and changes
#     nothing; `start` additionally requires DS41RT_TPEP_L3_GRANT=1.
#
# Requires scripts/release-common.sh for config parsing and admission only.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repo_root/scripts/release-common.sh"
source "$repo_root/scripts/tp-ep-native-candidate-plan.sh"

# Optional MGMT source binding for the new hosts: when set, every ssh call is
# pinned to that local address (fabric names stay usable when it is empty).
ssh_bind_ip=""
ssh() {
  if [[ -n "$ssh_bind_ip" ]]; then command ssh -b "$ssh_bind_ip" "$@"; else command ssh "$@"; fi
}

usage() {
  cat <<'EOF'
Usage: scripts/run-tp-ep-native-candidate.sh [plan|start|status|stop] [OPTIONS]

plan (default, same as --dry-run) prints the exact commands and exits.
start runs them; it fails closed unless DS41RT_TPEP_L3_GRANT=1 is exported.

  --config FILE                    configuration (default ds41rt.config)
  --rtx-gpus 1|2                   RTX layout (default: config RTX_GPUS when 1|2)
  --first-layer N                  expected Spark first layer; 2 RTX must match
                                   the published plan, 1 RTX must be 0
  --run-id ID                      unique run id (default UTC time + pid)
  --restart                        explicitly clear this run's placement dir
  --qualified                      require daemon and library sha256 + arch
  --api-port N                     candidate API port (default 18000)
  --expert-port N                  candidate expert port (default 29441)
  --listen HOST                    API bind host (default 127.0.0.1)
  --coordinator-container NAME     default ds41rt-tpep-nvme-dev
  --spark-container NAME           default ds41rt-tpep-dev
  --daemon-bin PATH                in-container ds41rt binary
  --coordinator-native-lib PATH    in-container coordinator native library
  --spark-native-lib PATH          in-container Spark native library
  --spark-lib-source PATH          in-container built Spark library to stage
  --spark-lib-source-host PATH     host path of the built Spark library
  --spark-lib-stage-host PATH      host destination for the uniform Spark lib
  --expect-coordinator-lib-sha256 HEX  fail-closed coordinator library hash
  --expect-spark-lib-sha256 HEX        fail-closed Spark library hash
  --expect-coordinator-daemon-sha256 HEX  fail-closed coordinator daemon hash
  --expect-spark-daemon-sha256 HEX        fail-closed Spark daemon hash
  --role-manifest FILE             {"tp2":{"sha256":...}} Spark role hashes
  --snapshot PATH                  in-container official checkpoint snapshot
  --wip-process PATH               in-container wip-process.sh (coordinator)
  --spark-wip-process PATH         in-container wip-process.sh (Spark)
  --wip-runtime-root PATH          candidate PID/log root (default /scratch/candidate/run)
  --coordinator-cuda-visible-devices UUID,UUID
                                   ordered coordinator GPU UUIDs; coordinator
                                   container only. Env: DS41RT_TPEP_COORDINATOR_CUDA_VISIBLE_DEVICES
  --placement-root PATH            candidate placement root
  --host-artifact-root PATH        local ext4 artifact root for the fs guard
  --dspark-draft-limit N           coordinator dSpark draft limit (default 7 for
                                   2 RTX, 5 for 1 RTX; must match corpus config)
  --host-device-map VALUE          DS41RT_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP for
                                   both roles (all-local-IP union is accepted;
                                   the native parser selects this host's IP)
  --spark-ssh RANK=DEST            repeatable: ssh destination for one rank
                                   (e.g. 4=172.22.2.5 for rhea over MGMT)
  --spark-ssh-bind IP              pin every ssh source to this local address
  --tcp-timing VALUE               startup diagnostic (forwarded to BOTH roles as
                                   DS41RT_PROTOCOL_V2_TCP_TIMING) that makes the
                                   transport emit the selected device/GID line.
                                   No production default: unset means no RC proof
                                   line and the six-rank GID capture fails closed.
  --capacity N                     expert capacity (default from prefill batch)
  --require-arch on|off            verify native library architecture (default on)

Artifact paths are per-container `/scratch` paths by design: the coordinator
container maps /scratch to the NVMe artifact root and each Spark container maps
its own /scratch. Use the overrides above for any other layout.
EOF
}

mode="plan"
case "${1:-}" in
  plan|start|status|stop) mode="$1"; shift ;;
  --dry-run|-n) mode=plan; shift ;;
  -h|--help) usage; exit 0 ;;
  "") ;;
  *) release_die "unknown mode: $1 (expected plan, start, status or stop)" ;;
esac

config="$repo_root/ds41rt.config"
rtx_gpus_override=""
first_layer=""
run_id=""
restart=0
qualified=0
api_port=18000
expert_port=29441
listen_host=127.0.0.1
coordinator_container="ds41rt-tpep-nvme-dev"
spark_container="ds41rt-tpep-dev"
daemon_bin="/scratch/candidate/ds41rt"
coordinator_native_lib="/scratch/coord-native/libds41rt_native.so"
# Uniform per-rank contract: every Spark container reads its own
# /scratch/candidate/libds41rt_native.so (Spark /scratch is the local
# /home/tj/ds41rt-tpep bind). The dodo L4 build output is staged from here.
spark_native_lib="/scratch/candidate/libds41rt_native.so"
spark_lib_source="/scratch/spark-tp-aot-dodo/native/libds41rt_native.so"
spark_lib_source_host="/home/tj/ds41rt-tpep/l4/spark-tp-aot-dodo/native/libds41rt_native.so"
spark_lib_stage_host="/home/tj/ds41rt-tpep/candidate/libds41rt_native.so"
snapshot_override=""
role_manifest=""
expect_coordinator_lib_sha256=""
expect_spark_lib_sha256=""
expect_coordinator_daemon_sha256=""
expect_spark_daemon_sha256=""
wip_process=""
spark_wip_process=""
# Per-container writable run root for candidate PIDs/logs, so the ops logs live
# under the artifact bind instead of the container's /wip/run.
wip_runtime_root="/scratch/candidate/run"
placement_root="/scratch/candidate/placement"
host_artifact_root="/home/tj/.cache/ds41rt/builds/tp-ep"
# Optional coordinator-only GPU pin. The config's COORDINATOR_GPU is a single
# index and cannot express the ordered RTX pair, so the candidate takes the
# exact ordered UUID CSV explicitly; it is never inferred from a hidden lookup.
coordinator_cuda_visible_devices="${DS41RT_TPEP_COORDINATOR_CUDA_VISIBLE_DEVICES:-}"
capacity_override=""
dspark_draft_limit=""
host_device_map=""
tcp_timing=""
spark_ssh_specs=()
require_arch=on

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) config="${2:?$1 requires FILE}"; shift 2 ;;
    --rtx-gpus) rtx_gpus_override="${2:?$1 requires 1 or 2}"; shift 2 ;;
    --first-layer) first_layer="${2:?$1 requires N}"; shift 2 ;;
    --run-id) run_id="${2:?$1 requires ID}"; shift 2 ;;
    --restart) restart=1; shift ;;
    --qualified) qualified=1; shift ;;
    --api-port) api_port="${2:?$1 requires N}"; shift 2 ;;
    --expert-port) expert_port="${2:?$1 requires N}"; shift 2 ;;
    --listen) listen_host="${2:?$1 requires HOST}"; shift 2 ;;
    --coordinator-container) coordinator_container="${2:?$1 requires NAME}"; shift 2 ;;
    --spark-container) spark_container="${2:?$1 requires NAME}"; shift 2 ;;
    --daemon-bin) daemon_bin="${2:?$1 requires PATH}"; shift 2 ;;
    --coordinator-native-lib) coordinator_native_lib="${2:?$1 requires PATH}"; shift 2 ;;
    --spark-native-lib) spark_native_lib="${2:?$1 requires PATH}"; shift 2 ;;
    --spark-lib-source) spark_lib_source="${2:?$1 requires PATH}"; shift 2 ;;
    --spark-lib-source-host) spark_lib_source_host="${2:?$1 requires PATH}"; shift 2 ;;
    --spark-lib-stage-host) spark_lib_stage_host="${2:?$1 requires PATH}"; shift 2 ;;
    --snapshot) snapshot_override="${2:?$1 requires PATH}"; shift 2 ;;
    --role-manifest) role_manifest="${2:?$1 requires FILE}"; shift 2 ;;
    --expect-coordinator-lib-sha256) expect_coordinator_lib_sha256="${2:?$1 requires HEX}"; shift 2 ;;
    --expect-spark-lib-sha256) expect_spark_lib_sha256="${2:?$1 requires HEX}"; shift 2 ;;
    --expect-coordinator-daemon-sha256) expect_coordinator_daemon_sha256="${2:?$1 requires HEX}"; shift 2 ;;
    --expect-spark-daemon-sha256) expect_spark_daemon_sha256="${2:?$1 requires HEX}"; shift 2 ;;
    --wip-process) wip_process="${2:?$1 requires PATH}"; shift 2 ;;
    --spark-wip-process) spark_wip_process="${2:?$1 requires PATH}"; shift 2 ;;
    --wip-runtime-root) wip_runtime_root="${2:?$1 requires PATH}"; shift 2 ;;
    --coordinator-cuda-visible-devices) coordinator_cuda_visible_devices="${2:?$1 requires an ordered GPU UUID CSV}"; shift 2 ;;
    --placement-root) placement_root="${2:?$1 requires PATH}"; shift 2 ;;
    --host-artifact-root) host_artifact_root="${2:?$1 requires PATH}"; shift 2 ;;
    --dspark-draft-limit) dspark_draft_limit="${2:?$1 requires N}"; shift 2 ;;
    --host-device-map) host_device_map="${2:?$1 requires VALUE}"; shift 2 ;;
    --tcp-timing) tcp_timing="${2:?$1 requires VALUE}"; shift 2 ;;
    --spark-ssh) spark_ssh_specs+=("${2:?$1 requires RANK=DEST}"); shift 2 ;;
    --spark-ssh-bind) ssh_bind_ip="${2:?$1 requires IP}"; shift 2 ;;
    --capacity) capacity_override="${2:?$1 requires N}"; shift 2 ;;
    --require-arch) require_arch="${2:?$1 requires on or off}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) release_die "unknown candidate launcher argument: $1" ;;
  esac
done

for port_name in api_port expert_port; do
  value="${!port_name}"
  [[ "$value" =~ ^[0-9]+$ ]] && ((value >= 1 && value <= 65535)) ||
    release_die "$port_name must be in 1..65535, got: $value"
done
case "$require_arch" in on|off) ;; *) release_die "--require-arch must be on or off" ;; esac
case "$rtx_gpus_override" in ""|1|2) ;; *) release_die "--rtx-gpus must be 1 or 2" ;; esac
[[ -z "$first_layer" || "$first_layer" =~ ^([0-9]|[1-3][0-9])$ ]] ||
  release_die "--first-layer must be 0..39"
[[ -z "$run_id" || "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] ||
  release_die "--run-id must be a short [A-Za-z0-9._-] token"
if [[ -n "$capacity_override" ]]; then
  [[ "$capacity_override" =~ ^[1-9][0-9]*$ ]] || release_die "--capacity must be a positive integer"
fi
if [[ "$qualified" == 1 ]]; then
  [[ -n "$role_manifest" || -n "$expect_spark_lib_sha256" ]] ||
    release_die "--qualified requires --role-manifest or --expect-spark-lib-sha256"
  [[ -n "$expect_coordinator_lib_sha256" ]] || release_die "--qualified requires --expect-coordinator-lib-sha256"
  [[ -n "$expect_coordinator_daemon_sha256" ]] || release_die "--qualified requires --expect-coordinator-daemon-sha256"
  [[ -n "$expect_spark_daemon_sha256" ]] || release_die "--qualified requires --expect-spark-daemon-sha256"
  [[ -n "$coordinator_cuda_visible_devices" ]] ||
    release_die "--qualified requires --coordinator-cuda-visible-devices (ordered physical GPU UUIDs)"
  [[ "$require_arch" == on ]] || release_die "--qualified requires --require-arch on"
fi

# Reject the read-only NTFS scratch tree by name and by backing filesystem.
[[ "$wip_runtime_root" == /* && "$wip_runtime_root" != /mnt/scratch* ]] ||
  release_die "--wip-runtime-root must be an absolute non-/mnt/scratch path: $wip_runtime_root"
for candidate_path in "$host_artifact_root" "$coordinator_native_lib" "$spark_native_lib" \
  "$spark_lib_source" "$daemon_bin" "$placement_root" "$wip_process" "$wip_runtime_root"; do
  [[ -z "$candidate_path" || "$candidate_path" != /mnt/scratch* ]] ||
    release_die "refusing a /mnt/scratch path (NTFS read-only): $candidate_path"
done
if [[ -f "$repo_root/scripts/assert-build-filesystem.py" && "$host_artifact_root" == /* ]]; then
  if ! python3 "$repo_root/scripts/assert-build-filesystem.py" "$host_artifact_root" >/dev/null 2>&1; then
    if [[ "$mode" == start ]]; then
      release_die "artifact root rejected by the build-filesystem guard: $host_artifact_root"
    fi
    echo "WARN artifact root not verified by the build-filesystem guard: $host_artifact_root" >&2
  fi
fi

release_load_config "$config"
release_spark_topology_explicit ||
  release_die "the candidate launcher requires explicit SPARK_TP and SPARK_EP (native replicated topology only)"
spark_tp="$SPARK_TP"
spark_ep="$SPARK_EP"
spark_world=$((spark_tp * spark_ep))
[[ "$spark_world" == "$SPARK_COUNT" ]] ||
  release_die "SPARK_TP x SPARK_EP must equal SPARK_COUNT"

rtx_gpus="$rtx_gpus_override"
if [[ -z "$rtx_gpus" ]]; then
  case "$RTX_GPUS" in 1|2) rtx_gpus="$RTX_GPUS" ;; esac
fi
[[ "$rtx_gpus" == 1 || "$rtx_gpus" == 2 ]] ||
  release_die "set --rtx-gpus 1|2, or RTX_GPUS=1|2 in $config"
# dSpark draft limit is a coordinator-only flag; 7 is the dual-RTX corpus value
# and 5 the single-RTX one. It is passed explicitly so the arm cannot silently
# run the daemon default and drift from the qualified metadata.
if [[ -z "$dspark_draft_limit" ]]; then
  if [[ "$rtx_gpus" == 2 ]]; then dspark_draft_limit=7; else dspark_draft_limit=5; fi
fi
[[ "$dspark_draft_limit" == 5 || "$dspark_draft_limit" == 7 ]] ||
  release_die "--dspark-draft-limit must be 5 (1 RTX) or 7 (2 RTX), got: $dspark_draft_limit"


# Ordered coordinator GPU UUID pin: comma-separated, every entry a physical
# NVIDIA GPU UUID, no duplicates, and exactly one per selected RTX. Numeric
# indices are rejected: the container's device mapping must be reproducible.
if [[ -n "$coordinator_cuda_visible_devices" ]]; then
  IFS=',' read -ra coordinator_gpu_uuids <<<"$coordinator_cuda_visible_devices"
  ((${#coordinator_gpu_uuids[@]} == rtx_gpus)) ||
    release_die "--coordinator-cuda-visible-devices needs exactly $rtx_gpus UUID(s), got ${#coordinator_gpu_uuids[@]}"
  for coordinator_gpu_uuid in "${coordinator_gpu_uuids[@]}"; do
    [[ "$coordinator_gpu_uuid" =~ ^GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$ ]] ||
      release_die "coordinator CUDA_VISIBLE_DEVICES entry is not a physical GPU UUID: $coordinator_gpu_uuid"
  done
  printf '%s\n' "${coordinator_gpu_uuids[@]}" | sort | uniq -d | grep -q . &&
    release_die "coordinator CUDA_VISIBLE_DEVICES contains a duplicate UUID"
fi

mapfile -t hosts < <(release_spark_values HOST)
mapfile -t lanes < <(release_spark_values LANE_A)
((${#hosts[@]} == spark_world)) ||
  release_die "configured Spark hosts (${#hosts[@]}) must equal TP x EP ($spark_world)"
# Per-rank ssh destination override (e.g. MGMT IPs for the new hosts). The
# override is the ssh target only; the config host name stays the logical id.
for spark_ssh_spec in ${spark_ssh_specs[@]+"${spark_ssh_specs[@]}"}; do
  [[ "$spark_ssh_spec" == *=* ]] || release_die "--spark-ssh expects RANK=DEST, got: $spark_ssh_spec"
  ssh_rank="${spark_ssh_spec%%=*}"; ssh_dest="${spark_ssh_spec#*=}"
  [[ "$ssh_rank" =~ ^[0-5]$ && -n "$ssh_dest" ]] || release_die "invalid --spark-ssh: $spark_ssh_spec"
  hosts[$ssh_rank]="$ssh_dest"
done
if [[ -n "$ssh_bind_ip" ]]; then
  [[ "$ssh_bind_ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] ||
    release_die "--spark-ssh-bind must be an IPv4 address, got: $ssh_bind_ip"
fi
[[ -z "$host_device_map" ]] || export DS41RT_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP="$host_device_map"
# Explicit startup diagnostic only; never a production default. When set it is
# forwarded to both roles so the transport emits the client/server GID lines.
[[ -z "$tcp_timing" ]] || export DS41RT_PROTOCOL_V2_TCP_TIMING="$tcp_timing"

if [[ -z "$wip_process" ]]; then
  [[ "$coordinator_container" == "ds41rt-tpep-nvme-dev" ]] ||
    release_die "coordinator container '$coordinator_container' is not the known layout; pass --wip-process explicitly (no silent fallback)"
  wip_process="/scratch/coord-native-source/scripts/wip-process.sh"
fi
if [[ -z "$spark_wip_process" ]]; then
  # The Spark containers mount the staged source at /workspace/ds41rt; the
  # frozen coordinator source is not present there.
  [[ "$spark_container" == "ds41rt-tpep-dev" ]] ||
    release_die "Spark container '$spark_container' is not the known layout; pass --spark-wip-process explicitly (no silent fallback)"
  spark_wip_process="/workspace/ds41rt/scripts/wip-process.sh"
fi

expert_capacity="${capacity_override}"
if [[ -z "$expert_capacity" ]]; then
  expert_capacity=4096
  if ((PREFILL_BATCH_TOKENS <= 80)); then expert_capacity=80
  elif ((PREFILL_BATCH_TOKENS <= 256)); then expert_capacity=256
  elif ((PREFILL_BATCH_TOKENS <= 1024)); then expert_capacity=1024
  fi
fi

snapshot="${snapshot_override:-/root/.cache/huggingface/hub/models--${RELEASE_MODEL_ID//\//--}/snapshots/${RELEASE_MODEL_REVISION}}"
peer_addresses=()
for lane in "${lanes[@]}"; do peer_addresses+=("$lane:$expert_port"); done
peers="$(IFS=,; echo "${peer_addresses[*]}")"
coordinator_process="candidate-coordinator-$api_port"
expert_process="candidate-expert-$expert_port"
declare -A worker_log_offsets=()

# Unique placement directory per run. A reused directory can carry an old nonce
# and layer boundary; start refuses it unless --restart explicitly clears it.
run_id="${run_id:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
placement_dir="$placement_root/$run_id"

# Optional role manifest: {"tp2":{"sha256":"...","arch":"aarch64"},...}.
if [[ -n "$role_manifest" ]]; then
  [[ -f "$role_manifest" ]] || release_die "role manifest not found: $role_manifest"
  role_key="tp${spark_tp}"
  manifest_sha="$(python3 -c '
import json, sys
document = json.load(open(sys.argv[1]))
entry = document.get(sys.argv[2])
if not isinstance(entry, dict):
    raise SystemExit(1)
print(entry.get("sha256", ""))
' "$role_manifest" "$role_key" 2>/dev/null || true)"
  [[ -n "$manifest_sha" ]] || {
    # TP4EP1 is the matched legacy reference arm: there is no tp4 extra role,
    # so its library identity comes from --expect-spark-lib-sha256 directly.
    [[ "$spark_tp" == 4 ]] ||
      release_die "role manifest $role_manifest has no sha256 for $role_key"
    manifest_sha="$expect_spark_lib_sha256"
  }
  if [[ -n "$manifest_sha" ]]; then
    [[ -z "$expect_spark_lib_sha256" || "$expect_spark_lib_sha256" == "$manifest_sha" ]] ||
      release_die "--expect-spark-lib-sha256 disagrees with $role_manifest for $role_key"
    expect_spark_lib_sha256="$manifest_sha"
  fi
fi
if [[ "$qualified" == 1 && -z "$expect_spark_lib_sha256" ]]; then
  release_die "--qualified requires a Spark library hash (--expect-spark-lib-sha256 or a matching role-manifest entry)"
fi

# ---------------------------------------------------------------------------
# Admission. Weight-only against the resolved boundary; never a feasibility
# claim. A 2-RTX automatic boundary is published by the coordinator and is
# re-checked after the plan is read, before any worker starts.
# ---------------------------------------------------------------------------
resolved_first_layer="$first_layer"
# A placement plan exists whenever the coordinator keeps local routed experts the
# workers must not also reserve: always on two RTX, and on one RTX for an explicit
# topology given an explicit local count in 1..=39. `auto`/`0` on one RTX has no
# local boundary to publish and keeps the legacy no-handoff launch. The daemon
# computes the boundary from the local device only, before any remote transport.
candidate_placement_handoff=0
# The candidate launcher always requires an explicit topology (checked above), so
# the local-count condition only has to confirm an explicit layer count.
if [[ "$RTX_EXPERT_LAYERS" =~ ^([1-9]|[1-3][0-9])$ ]]; then
  [[ "$rtx_gpus" != 1 || ( -n "$spark_tp" && -n "$spark_ep" ) ]] && candidate_placement_handoff=1
fi
[[ "$rtx_gpus" == 2 ]] && candidate_placement_handoff=1
admission="not-applicable"
if [[ "$candidate_placement_handoff" == 0 && "$rtx_gpus" == 1 && -z "$resolved_first_layer" ]]; then
  resolved_first_layer=0
fi
if [[ -n "$resolved_first_layer" ]]; then
  admission="$(release_validate_spark_weight_admission "$resolved_first_layer" "$spark_tp" "$SPARK_DEVICE_BUDGET_BYTES")"
else
  admission="PENDING (placement plan not published; weight-only check runs after the real boundary is read)"
fi

render() { printf '%q ' "$@"; }

# NUL-separated `-e KEY=VALUE` args carrying the candidate runtime environment.
# DS41RT_NATIVE_LIB is the transport's only native-library path: verbs falls
# back to a relative native/build* candidate otherwise, which would load a
# wrong or missing RDMA library. It must match --native-lib exactly. Only the
# native/RDMA runtime keys are forwarded; release quant env is not cloned.
candidate_env_args() {
  local native_lib="$1" forward
  printf '%s\0' -e "DS41RT_NATIVE_LIB=$native_lib"
  printf '%s\0' -e "DS41RT_WIP_RUNTIME_ROOT=$wip_runtime_root"
  # main.rs uses EnvFilter::from_default_env(), which is ERROR when unset, so
  # the readiness/placement tracing::info lines are invisible without this.
  # A user filter that hides them makes the readiness waits time out (nonzero).
  printf '%s\0' -e "RUST_LOG=${RUST_LOG:-info}"
  for forward in \
    DS41RT_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP \
    DS41RT_VERBS_APP_IB_PORT_NUM \
    DS41RT_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES \
    DS41RT_ADAPTIVE_COST_MODE \
    DS41RT_PROTOCOL_V2_TCP_TIMING \
    DS41RT_SPARKINFER_SOURCE_DIR \
    PYTHONPATH \
    DS41RT_PYTHON; do
    [[ -z "${!forward:-}" ]] || printf '%s\0' -e "$forward=${!forward}"
  done
}

# d129 device map: `local-ip=device` entries, required on BOTH roles so the
# verbs host layer binds the intended HCA instead of guessing. Fail closed on a
# malformed or duplicate local IP rather than forwarding a broken map.
if [[ -n "${DS41RT_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP:-}" ]]; then
  declare -A device_map_seen=()
  IFS=',' read -ra device_map_entries <<<"$DS41RT_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP"
  for entry in "${device_map_entries[@]}"; do
    [[ "$entry" == *"="* && -n "${entry%%=*}" && -n "${entry#*=}" ]] ||
      release_die "invalid DS41RT_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP entry: '$entry' (expected local-ip=device)"
    device_map_ip="${entry%%=*}"
    [[ "$device_map_ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] ||
      release_die "invalid device-map local IP: '$device_map_ip'"
    [[ -z "${device_map_seen[$device_map_ip]:-}" ]] ||
      release_die "duplicate device-map local IP: '$device_map_ip'"
    device_map_seen["$device_map_ip"]=1
  done
fi

# Command builders emit NUL-separated argv so callers can either render them
# for the plan or exec them directly, with no eval and no re-quoting.
#
# The coordinator control flags mirror run.sh's selected native controls so the
# candidate is matched against the release arm. Parsing a config key and not
# forwarding it would silently run different prefill/queue/cache/dSpark
# behaviour, and an undersized expert capacity.
coordinator_argv() {
  local -a env_args=()
  mapfile -d '' -t env_args < <(candidate_env_args "$coordinator_native_lib")
  # Coordinator-only ordered GPU pin; the Spark containers must never receive it.
  [[ -z "$coordinator_cuda_visible_devices" ]] ||
    env_args+=(-e "CUDA_VISIBLE_DEVICES=$coordinator_cuda_visible_devices")
  local -a args=(
    docker exec -d
    "${env_args[@]}"
    "$coordinator_container"
    "$wip_process" run "$coordinator_process"
    "$daemon_bin" serve-native
    --snapshot "$snapshot"
    --native-lib "$coordinator_native_lib"
    --peers "$peers"
    --rtx-gpus "$rtx_gpus"
    --listen "$listen_host:$api_port"
    --prefill-batch-tokens "$PREFILL_BATCH_TOKENS"
    --concurrency "$CONCURRENCY"
    --prefix-cache-entries "$PREFIX_CACHE_ENTRIES"
    --max-context-tokens "$MAX_CONTEXT_TOKENS"
    --max-output-tokens "$MAX_OUTPUT_TOKENS"
    --http-queue-depth "${HTTP_QUEUE_DEPTH:-$CONCURRENCY}"
    --http-queue-wait-ms "$HTTP_QUEUE_WAIT_MS"
    --spark-tp "$spark_tp"
    --spark-ep "$spark_ep"
  )
  [[ "$RTX_EXPERT_LAYERS" == auto ]] || args+=(--rtx-expert-layers "$RTX_EXPERT_LAYERS")
  [[ "$HOST_CACHE_BYTES" == 0 ]] || args+=(--host-cache-bytes "$HOST_CACHE_BYTES")
  [[ -z "$KV_POOL_SIZE" ]] || args+=(--kv-pool-size "$KV_POOL_SIZE")
  [[ -z "$MEMORY_RESERVATION" ]] || args+=(--memory-reservation "$MEMORY_RESERVATION")
  [[ "$DSPARK" != on ]] || args+=(--dspark --dspark-draft-limit "$dspark_draft_limit")
  [[ "$TP2_ATTENTION" != on ]] || args+=(--tp2-attention)
  [[ "$TP2_QUERY_PROJECTION" != on ]] || args+=(--tp2-query-projection)
  [[ "$TP2_OUTPUT_PROJECTION" != on ]] || args+=(--tp2-output-projection)
  [[ "$TP2_DSPARK_EXPERTS" != on ]] || args+=(--tp2-dspark-experts)
  [[ "$candidate_placement_handoff" == 0 ]] || args+=(--placement-directory "$placement_dir")
  printf '%s\0' "${args[@]}"
}

expert_argv() {
  local rank="$1" first="$2"
  local -a env_args=()
  mapfile -d '' -t env_args < <(candidate_env_args "$spark_native_lib")
  local -a args=(
    docker exec -d
    "${env_args[@]}"
    "$spark_container"
    "$spark_wip_process" run "$expert_process"
    "$daemon_bin" expertd-native
    --snapshot "$snapshot"
    --native-lib "$spark_native_lib"
    --rank "$rank"
    --world "$spark_world"
    --spark-tp "$spark_tp"
    --spark-ep "$spark_ep"
    --capacity "$expert_capacity"
    --device-budget-bytes "$SPARK_DEVICE_BUDGET_BYTES"
    --first-layer "$first"
    --listen "0.0.0.0:$expert_port"
  )
  printf '%s\0' "${args[@]}"
}

coordinator_command_string() {
  local -a argv=()
  mapfile -d '' -t argv < <(coordinator_argv)
  render "${argv[@]}"
}

expert_command_string() {
  local rank="$1" host="$2" first="$3"
  local -a argv=()
  mapfile -d '' -t argv < <(expert_argv "$rank" "$first")
  printf 'ssh -o BatchMode=yes %q ' "$host"
  render "${argv[@]}"
}

start_coordinator() {
  local -a argv=()
  mapfile -d '' -t argv < <(coordinator_argv)
  "${argv[@]}"
}

start_expert() {
  local rank="$1" host="$2" first="$3"
  local -a argv=()
  mapfile -d '' -t argv < <(expert_argv "$rank" "$first")
  ssh -o BatchMode=yes "$host" "$(render "${argv[@]}")"
}

# ---------------------------------------------------------------------------
# Live helpers. Only reachable through an explicit L3 grant.
# ---------------------------------------------------------------------------
require_l3_grant() {
  [[ "${DS41RT_TPEP_L3_GRANT:-0}" == "1" ]] ||
    release_die "live start is gated: export DS41RT_TPEP_L3_GRANT=1 after the parent records an L3 lease"
}

# Print running|stopped|absent|unreachable for a container on a host. An absent
# container is normal maintenance state; an unreachable host is a failure.
container_state() {
  local host="$1" container="$2" state
  if [[ "$host" == "$(hostname)" || "$host" == "localhost" ]]; then
    state="$(docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null || true)"
  else
    state="$(ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" \
      "docker inspect -f '{{.State.Running}}' '$container' 2>/dev/null || echo __unreachable__" 2>/dev/null || echo __unreachable__)"
    [[ "$state" != "__unreachable__" ]] || { printf 'unreachable\n'; return; }
  fi
  case "$state" in
    true) printf 'running\n' ;;
    false) printf 'stopped\n' ;;
    *) printf 'absent\n' ;;
  esac
}

# Never start or stop a production/shared container, even if an override points
# at one. Read-only plan/status may still name them for inspection.
require_candidate_containers() {
  local container
  for container in "$coordinator_container" "$spark_container"; do
    case "$container" in
      ds41rt-coordinator|ds41rt-coordinator-wip|ds41rt-spark-expert|ds41rt-spark-expert-wip|ds41rt-spark-expert-*)
        release_die "container '$container' is a production/shared name; the candidate launcher refuses to start or stop it" ;;
    esac
  done
}

# Candidate process liveness via wip-process, not container liveness. The run
# root env must match the one used to start the process or status/log look in
# the wrong directory.
candidate_process_running() {
  local host="$1" container="$2" process="$3" wip="$4" status
  if [[ "$host" == "$(hostname)" || "$host" == "localhost" ]]; then
    status="$(docker exec -e "DS41RT_WIP_RUNTIME_ROOT=$wip_runtime_root" "$container" "$wip" status "$process" 2>/dev/null || true)"
  else
    status="$(ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" \
      "docker exec -e 'DS41RT_WIP_RUNTIME_ROOT=$wip_runtime_root' '$container' '$wip' status '$process'" 2>/dev/null || true)"
  fi
  [[ "$status" == running\ * ]]
}

candidate_log_tail() {
  local host="$1" container="$2" process="$3" wip="$4"
  if [[ "$host" == "$(hostname)" || "$host" == "localhost" ]]; then
    docker exec -e "DS41RT_WIP_RUNTIME_ROOT=$wip_runtime_root" "$container" "$wip" log "$process" 80 2>/dev/null || true
  else
    ssh -o BatchMode=yes "$host" \
      "docker exec -e 'DS41RT_WIP_RUNTIME_ROOT=$wip_runtime_root' '$container' '$wip' log '$process' 80" 2>/dev/null || true
  fi
}

verify_container_artifact() {
  local host="$1" container="$2" path="$3" expected_sha="$4" arch="$5" label="$6"
  local script
  script=$(
    cat <<REMOTE
set -euo pipefail
test -s "$path" || { echo "missing $path" >&2; exit 2; }
if [ -n "$expected_sha" ]; then
  actual="\$(sha256sum "$path" | awk '{print \$1}')"
  [ "\$actual" = "$expected_sha" ] || { echo "sha mismatch $path: \$actual" >&2; exit 2; }
fi
if [ "$require_arch" = on ]; then
  command -v readelf >/dev/null || { echo "readelf missing for arch check" >&2; exit 2; }
  machine="\$(readelf -h "$path" | awk '/Machine:/{print \$2}')"
  case "$arch" in
    x86_64) [ "\$machine" = "Advanced" ] || { echo "expected x86-64, got \$machine" >&2; exit 2; } ;;
    aarch64) [ "\$machine" = "AArch64" ] || { echo "expected AArch64, got \$machine" >&2; exit 2; } ;;
    *) echo "unknown arch $arch" >&2; exit 2 ;;
  esac
fi
REMOTE
  )
  if [[ "$host" == "$(hostname)" || "$host" == "localhost" ]]; then
    docker exec "$container" bash -c "$script" || release_die "$label artifact check failed on $host"
  else
    ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" \
      "docker exec '$container' bash -c $(printf '%q' "$script")" ||
      release_die "$label artifact check failed on $host"
  fi
}

verify_container_file() {
  local host="$1" container="$2" path="$3" label="$4"
  if [[ "$host" == "$(hostname)" || "$host" == "localhost" ]]; then
    docker exec "$container" test -s "$path" || release_die "$label missing in $container: $path"
  else
    ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" "docker exec '$container' test -s '$path'" ||
      release_die "$label missing on $host ($container): $path"
  fi
}

# Refuse to inherit another run's plan; --restart clears only this run's dir
# Worker log epoch helpers. wip-process.sh appends to the log, so a whole-log
# grep can match a previous run. The byte size is captured before the worker
# starts and only bytes after that offset are searched for this run's line.
worker_log_file() {
  printf '%s\n' "$wip_runtime_root/$expert_process.log"
}

worker_log_offset() {
  local host="$1" log output probe
  log="$(worker_log_file)"
  # One probe distinguishes a missing log (0) from a stat/SSH/container error,
  # which must be fatal. A bare `stat ... || echo 0` would silently re-expose a
  # previous run's log after a transient failure.
  probe='if [ -e "$1" ]; then stat -c %s "$1"; else echo 0; fi'
  if [[ "$host" == "$(hostname)" || "$host" == "localhost" ]]; then
    output="$(docker exec -e "DS41RT_WIP_RUNTIME_ROOT=$wip_runtime_root" "$spark_container" \
      sh -c "$probe" sh "$log")" ||
      release_die "failed to read worker log size on $host (container/stat error)"
  else
    output="$(ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" \
      "docker exec -e 'DS41RT_WIP_RUNTIME_ROOT=$wip_runtime_root' '$spark_container' sh -c '$probe' sh '$log'")" ||
      release_die "failed to read worker log size on $host (ssh/container/stat error)"
  fi
  [[ "$output" =~ ^[0-9]+$ ]] ||
    release_die "invalid worker log size on $host: '${output}'"
  printf '%s\n' "$output"
}

worker_log_since() {
  local host="$1" offset="$2" log
  log="$(worker_log_file)"
  if [[ "$host" == "$(hostname)" || "$host" == "localhost" ]]; then
    docker exec -e "DS41RT_WIP_RUNTIME_ROOT=$wip_runtime_root" "$spark_container" \
      tail -c "+$((offset + 1))" "$log" 2>/dev/null || true
  else
    ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" \
      "docker exec -e 'DS41RT_WIP_RUNTIME_ROOT=$wip_runtime_root' '$spark_container' tail -c '+$((offset + 1))' '$log'" 2>/dev/null || true
  fi
}

# The default tracing formatter writes ANSI dim/italic codes between a field
# name and its `=`, so a raw `grep rank=1` misses the real log. Strip CSI
# sequences before matching the readiness fields.
strip_ansi() {
  sed $'s/\033\\[[0-9;]*[a-zA-Z]//g'
}

# True when one trimmed line carries the readiness message and this rank's exact
# numeric fields. All three fields must be on the SAME line and use numeric
# boundaries, so `rank=1` cannot match `rank=10` and `first_layer=2` cannot
# match `first_layer=20`.
worker_ready_line_matches() {
  local rank="$1" world="$2" first="$3" line
  while IFS= read -r line; do
    [[ "$line" == *"native local RoCE expert worker ready"* ]] || continue
    grep -Eq "(^|[^0-9])rank=$rank([^0-9]|$)" <<<"$line" || continue
    grep -Eq "(^|[^0-9])world=$world([^0-9]|$)" <<<"$line" || continue
    grep -Eq "(^|[^0-9])first_layer=$first([^0-9]|$)" <<<"$line" || continue
    return 0
  done <<<"$4"
  return 1
}

# Block until every rank logs the current process's readiness line (rank, world,
# first_layer) after the pre-start byte offset, with a bounded timeout and early
# process-death detection.
wait_for_worker_ready() {
  local resolved="$1"
  local deadline=$((SECONDS + ${DS41RT_TPEP_WORKER_READY_TIMEOUT_SECONDS:-900}))
  local -a pending=()
  local rank host slice_text
  for rank in "${!hosts[@]}"; do pending+=("$rank"); done
  local announced=0
  while ((${#pending[@]})); do
    local -a still=()
    for rank in "${pending[@]}"; do
      host="${hosts[$rank]}"
      if ! candidate_process_running "$host" "$spark_container" "$expert_process" "$spark_wip_process"; then
        candidate_log_tail "$host" "$spark_container" "$expert_process" "$spark_wip_process" >&2 || true
        release_die "$host candidate expert (rank $rank) exited before reporting ready"
      fi
      slice_text="$(worker_log_since "$host" "${worker_log_offsets[$rank]}" | strip_ansi)"
      if worker_ready_line_matches "$rank" "$spark_world" "$resolved" "$slice_text"; then
        echo "  rank $rank ($host): worker ready"
      else
        still+=("$rank")
      fi
    done
    pending=("${still[@]}")
    ((${#pending[@]})) || break
    if ((!announced)); then
      echo "== waiting for candidate worker readiness ($spark_world rank(s), timeout ${DS41RT_TPEP_WORKER_READY_TIMEOUT_SECONDS:-900}s) =="
      announced=1
    fi
    ((SECONDS < deadline)) || {
      for rank in "${pending[@]}"; do
        candidate_log_tail "${hosts[$rank]}" "$spark_container" "$expert_process" "$spark_wip_process" >&2 || true
      done
      release_die "candidate worker readiness timed out (set RUST_LOG to include daemon info, or raise DS41RT_TPEP_WORKER_READY_TIMEOUT_SECONDS)"
    }
    sleep 1
  done
}


# Capture the actually selected RDMA device/GID per rank. An `ip route` match
# is not RC proof, so a rank with no GID/device line fails closed before ACK.
capture_gid_binding() {
  local resolved="$1" rank host slice out file
  file="$repo_root/runs/tp-ep-six/gid-binding-${run_id}.txt"
  mkdir -p "$(dirname "$file")"
  : >"$file"
  for rank in "${!hosts[@]}"; do
    host="${hosts[$rank]}"
    slice="$(worker_log_since "$host" "${worker_log_offsets[$rank]}" | strip_ansi)"
    out="$(grep -E 'gid_index=|client_gid=|server_gid=|RDMA RC endpoint created' <<<"$slice" | head -5 || true)"
    if [[ -z "$out" ]]; then
      # Six-rank qualification is fail-closed: a rank with no GID/device line is
      # not RC-proven. The generic 4-Spark path only warns so its contract is
      # unchanged (and the reviewer's "keep original 4 untouched" holds).
      if [[ "$SPARK_COUNT" == "6" ]]; then
        candidate_log_tail "$host" "$spark_container" "$expert_process" "$spark_wip_process" >&2 || true
        release_die "rank $rank ($host) logged no RDMA GID/device binding; IP reachability alone is not RC proof"
      fi
      echo "WARN rank $rank ($host) logged no RDMA GID/device line; recorded as MISSING" >&2
      out="MISSING"
    fi
    echo "rank $rank ($host) first_layer=$resolved: $out" >>"$file"
  done
  echo "  captured RDMA GID/device binding for $spark_world ranks: $file"
}

# Refuse to inherit another run's plan; --restart clears only this run's dir
# under the candidate placement root.
prepare_placement_directory() {
  # Same predicate as the handoff itself: a single-RTX explicit-topology launch
  # with local routed layers uses the directory and needs the stale-state guard.
  [[ "$candidate_placement_handoff" == 1 ]] || return 0
  case "$placement_dir" in
    "$placement_root"/*) ;;
    *) release_die "placement directory must be under $placement_root: $placement_dir" ;;
  esac
  if docker exec "$coordinator_container" test -e "$placement_dir" 2>/dev/null; then
    if ((restart)); then
      echo "== clearing this run's stale placement directory: $placement_dir =="
      docker exec "$coordinator_container" rm -rf "$placement_dir" ||
        release_die "failed to clear $placement_dir"
    else
      release_die "placement directory already exists: $placement_dir (use --run-id or --restart after confirming no candidate coordinator owns it)"
    fi
  fi
}

wait_for_candidate_ready() {
  local deadline=$((SECONDS + ${DS41RT_TPEP_READY_TIMEOUT_SECONDS:-900}))
  local probe_host="$listen_host"
  [[ "$probe_host" != "0.0.0.0" ]] || probe_host=127.0.0.1
  local api_url="http://$probe_host:$api_port"
  until curl -fsS "$api_url/health" >/dev/null 2>&1 &&
    release_api_advertises_native_model "$api_url" "$RELEASE_NATIVE_API_MODEL_ID"; do
    candidate_process_running localhost "$coordinator_container" "$coordinator_process" "$wip_process" ||
      release_die "candidate coordinator process exited during startup"
    ((SECONDS < deadline)) ||
      release_die "candidate API did not become ready within ${DS41RT_TPEP_READY_TIMEOUT_SECONDS:-900}s"
    sleep 1
  done
  local rank host
  for rank in "${!hosts[@]}"; do
    host="${hosts[$rank]}"
    candidate_process_running "$host" "$spark_container" "$expert_process" "$spark_wip_process" ||
      release_die "$host candidate expert (rank $rank) is not running at readiness"
  done
  echo "candidate ready: api=$api_url (health + native model) and $spark_world expert processes running"
}

first_bounded_request() {
  local probe_host="$listen_host" api_url payload
  [[ "$probe_host" != "0.0.0.0" ]] || probe_host=127.0.0.1
  api_url="http://$probe_host:$api_port"
  payload="{\"model\":\"$RELEASE_NATIVE_API_MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"1+1\"}],\"max_tokens\":1,\"temperature\":0,\"stream\":false}"
  curl -fsS -m "${DS41RT_TPEP_FIRST_REQUEST_TIMEOUT_SECONDS:-120}" -o /dev/null \
    -H 'Content-Type: application/json' -d "$payload" "$api_url/v1/chat/completions" ||
    release_die "bounded first request failed; the lazy RDMA connections were not initialized"
  echo "  first bounded request accepted (lazy RDMA connections initialized)"
}

start_candidate() {
  require_l3_grant
  require_candidate_containers
  release_need docker
  release_need ssh
  release_need jq
  release_need curl

  local state
  state="$(container_state localhost "$coordinator_container")"
  [[ "$state" == running ]] ||
    release_die "coordinator container is $state (expected running): $coordinator_container"
  local host
  for host in "${hosts[@]}"; do
    state="$(container_state "$host" "$spark_container")"
    [[ "$state" == running ]] ||
      release_die "$host Spark container is $state (expected running): $spark_container"
  done

  verify_container_artifact localhost "$coordinator_container" "$daemon_bin" \
    "$expect_coordinator_daemon_sha256" x86_64 "coordinator daemon"
  verify_container_artifact localhost "$coordinator_container" "$coordinator_native_lib" \
    "$expect_coordinator_lib_sha256" x86_64 "coordinator native lib"
  for host in "${hosts[@]}"; do
    verify_container_artifact "$host" "$spark_container" "$daemon_bin" \
      "$expect_spark_daemon_sha256" aarch64 "$host daemon"
    verify_container_artifact "$host" "$spark_container" "$spark_native_lib" \
      "$expect_spark_lib_sha256" aarch64 "$host native lib"
  done
  # Per-role runtime paths must exist BEFORE anything is launched.
  verify_container_file localhost "$coordinator_container" "$wip_process" "coordinator wip-process"
  for host in "${hosts[@]}"; do
    verify_container_file "$host" "$spark_container" "$spark_wip_process" "$host wip-process"
  done

  local resolved="$resolved_first_layer"
  if [[ "$candidate_placement_handoff" == 1 ]]; then
    prepare_placement_directory
    echo "== starting candidate coordinator (plan handshake, run $run_id) =="
    start_coordinator
    local deadline=$((SECONDS + ${DS41RT_TPEP_PLAN_TIMEOUT_SECONDS:-300})) plan
    while ! plan="$(docker exec "$coordinator_container" cat "$placement_dir/plan.json" 2>/dev/null)"; do
      if ! candidate_process_running localhost "$coordinator_container" "$coordinator_process" "$wip_process"; then
        candidate_log_tail localhost "$coordinator_container" "$coordinator_process" "$wip_process" >&2 || true
        release_die "candidate coordinator process exited before publishing placement"
      fi
      ((SECONDS < deadline)) || {
        candidate_log_tail localhost "$coordinator_container" "$coordinator_process" "$wip_process" >&2 || true
        release_die "timed out waiting for candidate placement plan"
      }
      sleep 1
    done
    resolved="$(jq -er --argjson gpus "$rtx_gpus" '
      select(.version == 1)
      | select((.rtx_gpus | type) == "number" and .rtx_gpus == $gpus)
      | select((.nonce | type) == "string" and (.nonce | length) > 0)
      | select((.rtx_expert_layers | type) == "number")
      | select(.rtx_expert_layers == (.rtx_expert_layers | floor) and .rtx_expert_layers >= 1 and .rtx_expert_layers <= 40)
      | select(.spark_first_layer == ([.rtx_expert_layers, 39] | min))
      | .spark_first_layer' <<<"$plan")" || release_die "invalid candidate placement plan"
    [[ -z "$first_layer" || "$first_layer" == "$resolved" ]] ||
      release_die "--first-layer $first_layer does not match the published plan boundary $resolved"
    release_validate_spark_weight_admission "$resolved" "$spark_tp" "$SPARK_DEVICE_BUDGET_BYTES" >/dev/null
    echo "  placement: RTX GPUs $(jq -r '.rtx_gpus' <<<"$plan"), Spark first layer $resolved"
  else
    resolved=0
    [[ -z "$first_layer" || "$first_layer" == 0 ]] ||
      release_die "--first-layer must be 0 for a single-RTX launch with no local expert boundary"
  fi

  echo "== starting candidate Spark experts =="
  local rank failed=0
  for rank in "${!hosts[@]}"; do
    worker_log_offsets["$rank"]="$(worker_log_offset "${hosts[$rank]}")"
  done
  local -a started_experts=()
  for rank in "${!hosts[@]}"; do
    if ! start_expert "$rank" "${hosts[$rank]}" "$resolved"; then
      release_die "candidate expert rank $rank (${hosts[$rank]}) failed to start; NO further ranks were launched. Durable started set: coordinator=$coordinator_process (dual layout starts it first) experts=[${started_experts[*]:-}]. Stop exactly those with: $0 stop --config <the same config>"
    fi
    started_experts+=("$rank")
  done

  # Workers accept connections only after the full weight load, so the plan is
  # acknowledged only after every rank logs its current-process readiness line.
  wait_for_worker_ready "$resolved"

  # The acknowledgement must track the handoff predicate, not the RTX count: a
  # single-RTX explicit-topology launch with local routed layers publishes a plan
  # too, and gating this on `rtx_gpus == 2` would start the coordinator a second
  # time and never acknowledge the plan.
  if [[ "$candidate_placement_handoff" == 1 ]]; then
    docker exec "$coordinator_container" sh -c \
      "cp \"$placement_dir/plan.json\" \"$placement_dir/.ready-pending\" && mv \"$placement_dir/.ready-pending\" \"$placement_dir/ready.json\"" ||
      release_die "failed to acknowledge the candidate placement plan"
  else
    echo "== starting candidate coordinator (single RTX, no handoff) =="
    start_coordinator
  fi
  wait_for_candidate_ready
  # The RC endpoint (and its device/GID line) is created per connection, and the
  # coordinator opens none before the ACK, so one bounded request is sent first
  # to initialize the lazy connections; only then is the real RC GID captured.
  first_bounded_request
  capture_gid_binding "$resolved"
  echo "candidate started: api=$listen_host:$api_port coordinator=$coordinator_process experts=$expert_process"
  echo "the release fleet is untouched and still stopped; stop only the candidate processes with: $0 stop"
  echo "baseline restoration is the separate explicit operation runs/tp-ep-preflight/restore-release-fleet.sh"
}

process_action() {
  local action="$1" failed=0 state
  release_need docker
  release_need ssh
  require_candidate_containers
  state="$(container_state localhost "$coordinator_container")"
  case "$state" in
    running)
      docker exec -e "DS41RT_WIP_RUNTIME_ROOT=$wip_runtime_root" "$coordinator_container" "$wip_process" "$action" "$coordinator_process" || failed=1 ;;
    absent|stopped)
      echo "coordinator $coordinator_container: $state (no candidate process to $action)" ;;
    unreachable)
      echo "coordinator container state unreachable" >&2; failed=1 ;;
  esac
  local host
  for host in "${hosts[@]}"; do
    state="$(container_state "$host" "$spark_container")"
    case "$state" in
      running)
        if [[ "$host" == "$(hostname)" || "$host" == "localhost" ]]; then
          docker exec -e "DS41RT_WIP_RUNTIME_ROOT=$wip_runtime_root" "$spark_container" "$spark_wip_process" "$action" "$expert_process" || failed=1
        else
          ssh -o BatchMode=yes "$host" \
            "docker exec -e 'DS41RT_WIP_RUNTIME_ROOT=$wip_runtime_root' '$spark_container' '$spark_wip_process' '$action' '$expert_process'" || failed=1
        fi ;;
      absent|stopped)
        echo "$host $spark_container: $state (no candidate process to $action)" ;;
      unreachable)
        echo "$host unreachable" >&2; failed=1 ;;
    esac
  done
  ((failed == 0)) || release_die "candidate $action had one or more failures"
}

case "$mode" in
  plan) candidate_print_plan ;;
  start) start_candidate ;;
  status) process_action status ;;
  stop) process_action stop ;;
esac
