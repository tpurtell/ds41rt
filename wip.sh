#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$repo_root/scripts/release-common.sh"

usage() {
  cat <<'EOF'
Usage: ./wip.sh [--slot NAME] [--role coordinator|expert|both]
                [--from-slot NAME] [--config FILE] [--recreate] [--dry-run]

Synchronizes the current checkout into persistent development containers and
incrementally builds a named WIP slot. The coordinator container builds and
runs coordinator slots. The first configured Spark builds Spark slots, which
are copied directly and concurrently to the other persistent Spark WIP
containers.
An explicit SPARK_TP=2/3 topology builds the matching opt-in SM121 expert role;
DS41RT_WIP_SPARK_TP_ROLES=tp2;tp3 overrides that selection. The default
configuration builds no extra role and keeps the historical Spark TP4 shard.
--dry-run prints the resolved hosts, role plan and build invocations without
touching Docker, SSH or any container.

--from-slot NAME first clones an existing slot, then rebuilds the selected
role. This is useful for coordinator-only or expert-only A/B candidates.
For a coordinator-only clone, wip.sh stops only the coordinator process and
keeps resident Spark experts available for fingerprint-checked reuse.
--recreate discards all WIP containers and their build caches before
creating them again from the configured development images.
EOF
}

config="$repo_root/ds41rt.config"
slot=current
role=both
from_slot=
recreate=0
dry_run=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --slot)
      slot="${2:?--slot requires a name}"
      shift 2
      ;;
    --role)
      role="${2:?--role requires coordinator, expert, or both}"
      shift 2
      ;;
    --from-slot)
      from_slot="${2:?--from-slot requires a name}"
      shift 2
      ;;
    --config)
      config="${2:?$1 requires a configuration file}"
      shift 2
      ;;
    --recreate)
      recreate=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      release_die "unknown WIP argument: $1"
      ;;
  esac
done

[[ "$slot" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] ||
  release_die "invalid WIP slot name: $slot"
[[ -z "$from_slot" || "$from_slot" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] ||
  release_die "invalid source WIP slot name: $from_slot"
[[ "$slot" != "$from_slot" ]] || release_die "--from-slot must differ from --slot"
case "$role" in coordinator|expert|both) ;; *) release_die "--role must be coordinator, expert, or both" ;; esac

release_load_config "$config"
release_need docker
release_need ssh
release_need rsync
release_need python3
release_need sha256sum
release_need install
release_need nvidia-smi

mapfile -t wip_hosts < <(release_spark_values HOST)
((${#wip_hosts[@]})) || release_die "configuration has no active Spark hosts"
[[ "${#wip_hosts[@]}" == "$SPARK_COUNT" ]] ||
  release_die "SPARK_COUNT=$SPARK_COUNT does not match ${#wip_hosts[@]} configured Spark hosts"
seed_host="${wip_hosts[0]}"
wip_target_hosts=("${wip_hosts[@]:1}")

# Opt-in replicated-group Spark expert roles for the WIP slot. The default and
# explicit TP4xEP1 build no extra role.
wip_spark_tp_roles="${DS41RT_WIP_SPARK_TP_ROLES:-}"
if [[ -z "$wip_spark_tp_roles" ]] && release_spark_topology_explicit; then
  case "$SPARK_TP" in
    2) wip_spark_tp_roles=tp2 ;;
    3) wip_spark_tp_roles=tp3 ;;
    4) wip_spark_tp_roles= ;;
  esac
fi
if [[ -n "$wip_spark_tp_roles" ]]; then
  IFS=';' read -ra wip_spark_tp_role_list <<<"$wip_spark_tp_roles"
  for wip_spark_tp_role in "${wip_spark_tp_role_list[@]}"; do
    case "$wip_spark_tp_role" in
      tp2|tp3) ;;
      *) release_die "DS41RT_WIP_SPARK_TP_ROLES accepts only tp2 and tp3, got: $wip_spark_tp_role" ;;
    esac
  done
  unset wip_spark_tp_role wip_spark_tp_role_list
fi

if ((dry_run)); then
  echo "WIP dry-run passed; no container, image, SSH or build operation was performed."
  echo "  config: $RELEASE_CONFIG"
  echo "  slot: $slot (role $role)"
  echo "  Spark hosts (${#wip_hosts[@]}): $(IFS=,; echo "${wip_hosts[*]}")"
  echo "  seed host: $seed_host"
  echo "  topology: tp=$(release_spark_tp) ep=$(release_spark_ep) explicit=$(release_spark_topology_explicit && echo 1 || echo 0)"
  echo "  V41 Spark expert roles: ${wip_spark_tp_roles:-<legacy TP4 only>}"
  echo "  EXL3 AOT: ${DS41RT_WIP_EXL3_AOT:-ON}; NVFP4 AOT: ${DS41RT_WIP_NVFP4_AOT:-ON}"
  exit 0
fi

docker info >/dev/null 2>&1 || release_die "local Docker daemon is unavailable"
release_resolve_coordinator_gpu_identity

coordinator_container=ds41rt-coordinator-wip
spark_container=ds41rt-spark-expert-wip
seed_host="$SPARK_0_HOST"
state_dir="$repo_root/.ds41rt-wip"
staging_dir="$state_dir/source-staging"
hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$state_dir" "$hf_home"

snapshot_args=(
  -a --delete --delete-excluded
  --exclude .git --exclude '.venv*/' --exclude .mypy_cache/
  --exclude .pytest_cache/ --exclude .ruff_cache/ --exclude __pycache__/
  --exclude '*.pyc' --exclude '*.pyo' --exclude .ds41rt-cache/
  --exclude .ds41rt-release/ --exclude .ds41rt-release-image/
  --exclude .ds41rt-wip/ --exclude dist/ --exclude rust/target/
  --exclude 'native/build*/'
)

echo "== freezing current checkout for WIP slot $slot =="
mkdir -p "$staging_dir"
rsync "${snapshot_args[@]}" "$repo_root/" "$staging_dir/"
# The selected complete configuration is part of the slot, even when the
# caller chose a file other than the repository's default ds41rt.config.
install -m 0644 "$RELEASE_CONFIG" "$staging_dir/ds41rt.config"
python3 "$staging_dir/scripts/verify-sparkinfer-source.py" \
  --source "$staging_dir/third_party/sparkinfer" \
  --lock "$staging_dir/third_party/sparkinfer.lock.json"
python3 "$staging_dir/scripts/verify-xgrammar-source.py" \
  --source "$staging_dir/third_party/xgrammar" \
  --lock "$staging_dir/third_party/xgrammar.lock.json"

ensure_local_image() {
  docker image inspect "$COORDINATOR_DOCKER_DEV" >/dev/null 2>&1 ||
    release_die "missing coordinator development image $COORDINATOR_DOCKER_DEV; run ./build.sh"
}

ensure_seed_image() {
  ssh -o BatchMode=yes "$seed_host" \
    "docker image inspect '$SPARK_EXPERT_DOCKER_DEV' >/dev/null" ||
    release_die "$seed_host lacks development image $SPARK_EXPERT_DOCKER_DEV; run ./build.sh"
}

distribute_spark_dev_image() {
  local seed_id
  seed_id="$(ssh -o BatchMode=yes "$seed_host" "docker image inspect -f '{{.Id}}' '$SPARK_EXPERT_DOCKER_DEV'")"
  local -a targets=()
  local host remote_id
  for host in "${wip_target_hosts[@]}"; do
    remote_id="$(ssh -o BatchMode=yes "$host" "docker image inspect -f '{{.Id}}' '$SPARK_EXPERT_DOCKER_DEV' 2>/dev/null || true")"
    [[ "$remote_id" == "$seed_id" ]] || targets+=("$host")
  done
  ((${#targets[@]})) || return 0
  for host in "$seed_host" "${targets[@]}"; do
    ssh -o BatchMode=yes "$host" "command -v rdmapipe >/dev/null" ||
      release_die "rdmapipe is required to distribute the WIP development image ($host)"
  done
  echo "== concurrently distributing Spark development image from $seed_host =="
  local -a pids=()
  for host in "${targets[@]}"; do
    (
      set -o pipefail
      ssh -o BatchMode=yes "$seed_host" \
        "docker image save '$SPARK_EXPERT_DOCKER_DEV' | rdmapipe --send" |
        ssh -o BatchMode=yes "$host" 'rdmapipe --recv | docker image load'
    ) &
    pids+=("$!")
  done
  local failed=0 pid
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  ((failed == 0)) || release_die "Spark development image distribution failed"
}

remove_wip_containers() {
  docker rm -f "$coordinator_container" >/dev/null 2>&1 || true
  local host
  for host in "${wip_hosts[@]}"; do
    ssh -o BatchMode=yes "$host" "docker rm -f '$spark_container' >/dev/null 2>&1 || true" &
  done
  wait
}

if ((recreate)); then
  echo "== discarding persistent WIP containers and build caches =="
  remove_wip_containers
fi

ensure_local_image
ensure_seed_image

preflight_existing_container_images() {
  local expected actual host coordinator_device
  expected="$(docker image inspect -f '{{.Id}}' "$COORDINATOR_DOCKER_DEV")"
  if docker container inspect "$coordinator_container" >/dev/null 2>&1; then
    actual="$(docker inspect -f '{{.Image}}' "$coordinator_container")"
    [[ "$actual" == "$expected" ]] ||
      release_die "$coordinator_container uses an old development image; rerun ./wip.sh --recreate"
    coordinator_device="$(
      docker inspect -f \
        '{{range .HostConfig.DeviceRequests}}{{range .DeviceIDs}}{{println .}}{{end}}{{end}}' \
        "$coordinator_container"
    )"
    [[ "$coordinator_device" == "$RELEASE_COORDINATOR_GPU_UUID" ]] ||
      release_die "$coordinator_container is not bound to the configured physical GPU; rerun ./wip.sh --recreate"
  fi
  expected="$(ssh -o BatchMode=yes "$seed_host" "docker image inspect -f '{{.Id}}' '$SPARK_EXPERT_DOCKER_DEV'")"
  for host in "${wip_hosts[@]}"; do
    actual="$(ssh -o BatchMode=yes "$host" "docker inspect -f '{{.Image}}' '$spark_container' 2>/dev/null || true")"
    [[ -z "$actual" || "$actual" == "$expected" ]] ||
      release_die "$host $spark_container uses an old development image; rerun ./wip.sh --recreate"
  done
}

preflight_existing_container_images
distribute_spark_dev_image

ensure_local_container() {
  local image_id container_id
  image_id="$(docker image inspect -f '{{.Id}}' "$COORDINATOR_DOCKER_DEV")"
  if docker container inspect "$coordinator_container" >/dev/null 2>&1; then
    container_id="$(docker inspect -f '{{.Image}}' "$coordinator_container")"
    [[ "$container_id" == "$image_id" ]] ||
      release_die "$coordinator_container uses an old development image; rerun ./wip.sh --recreate"
    [[ "$(docker inspect -f '{{.State.Running}}' "$coordinator_container")" == true ]] ||
      docker start "$coordinator_container" >/dev/null
    docker exec "$coordinator_container" mkdir -p /wip/build /wip/output /wip/slots /wip/incoming /wip/run /wip/cache
    return
  fi
  local -a args=(
    run -d --name "$coordinator_container" --restart no
    --gpus device="$RELEASE_COORDINATOR_GPU_UUID"
    --net=host --ipc=host --security-opt seccomp=unconfined
    --ulimit memlock=-1:-1 --cap-add IPC_LOCK
    -v "$hf_home:$hf_home:ro" -v "$hf_home:/root/.cache/huggingface:ro"
    -e HF_HOME="$hf_home"
    -e CUDA_VISIBLE_DEVICES="$RELEASE_COORDINATOR_GPU_UUID"
    -e NVIDIA_VISIBLE_DEVICES="$RELEASE_COORDINATOR_GPU_UUID"
  )
  [[ ! -e /dev/infiniband ]] || args+=(--device=/dev/infiniband)
  docker "${args[@]}" "$COORDINATOR_DOCKER_DEV" sleep infinity >/dev/null
  docker exec "$coordinator_container" mkdir -p /wip/build /wip/output /wip/slots /wip/incoming /wip/run /wip/cache
}

ensure_remote_container() {
  local host="$1"
  ssh -o BatchMode=yes "$host" bash -s -- \
    "$spark_container" "$SPARK_EXPERT_DOCKER_DEV" <<'REMOTE'
set -euo pipefail
container="$1"
image="$2"
image_id="$(docker image inspect -f '{{.Id}}' "$image")"
if docker container inspect "$container" >/dev/null 2>&1; then
  container_id="$(docker inspect -f '{{.Image}}' "$container")"
  if [ "$container_id" != "$image_id" ]; then
    echo "$container uses an old development image; rerun ./wip.sh --recreate" >&2
    exit 2
  fi
  [ "$(docker inspect -f '{{.State.Running}}' "$container")" = true ] || docker start "$container" >/dev/null
  docker exec "$container" mkdir -p /wip/build /wip/output /wip/slots /wip/incoming /wip/run /wip/cache
  exit 0
fi
hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
args=(
  run -d --name "$container" --restart no
  --gpus all --net=host --ipc=host --security-opt seccomp=unconfined
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK
  -v "$hf_home:$hf_home:ro" -v "$hf_home:/root/.cache/huggingface:ro"
  -e HF_HOME="$hf_home"
)
[ ! -e /dev/infiniband ] || args+=(--device=/dev/infiniband)
docker "${args[@]}" "$image" sleep infinity >/dev/null
docker exec "$container" mkdir -p /wip/build /wip/output /wip/slots /wip/incoming /wip/run /wip/cache
REMOTE
}

echo "== ensuring persistent WIP development containers =="
ensure_local_container
remote_pids=()
for host in "${wip_hosts[@]}"; do
  ensure_remote_container "$host" &
  remote_pids+=("$!")
done
remote_failed=0
for pid in "${remote_pids[@]}"; do wait "$pid" || remote_failed=1; done
((remote_failed == 0)) || release_die "one or more Spark WIP containers could not be prepared"

wip_coordinator_processes_active() {
  docker exec "$coordinator_container" bash -lc '
for pid_file in /wip/run/*.pid; do
  [ -f "$pid_file" ] || continue
  pid="$(<"$pid_file")"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && exit 0
done
exit 1
'
}

wip_expert_processes_active() {
  local host
  for host in "${wip_hosts[@]}"; do
    if ssh -o BatchMode=yes "$host" docker exec -i "$spark_container" bash -s <<'CONTAINER'
for pid_file in /wip/run/*.pid; do
  [ -f "$pid_file" ] || continue
  pid="$(<"$pid_file")"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && exit 0
done
exit 1
CONTAINER
    then
      return 0
    fi
  done
  return 1
}

if [[ "$role" == coordinator && -n "$from_slot" ]]; then
  release_stop_wip_coordinator
  wip_coordinator_processes_active &&
    release_die "a coordinator WIP process remains active; stop it before building"
  if wip_expert_processes_active; then
    echo "== preserving resident Spark experts during coordinator-only build =="
  fi
elif wip_coordinator_processes_active || wip_expert_processes_active; then
  release_die "a WIP DS41RT process is active; stop it before synchronizing or building"
fi

if [[ -n "$from_slot" ]]; then
  echo "== cloning WIP slot $from_slot -> $slot =="
  docker exec -i "$coordinator_container" bash -s -- "$from_slot" "$slot" <<'CONTAINER'
set -euo pipefail
from="$1"
to="$2"
test -d "/wip/slots/$from/coordinator"
test ! -e "/wip/slots/$to"
CONTAINER
  ssh -o BatchMode=yes "$seed_host" docker exec -i "$spark_container" bash -s -- "$from_slot" "$slot" <<'CONTAINER'
set -euo pipefail
from="$1"
to="$2"
test -d "/wip/slots/$from/spark-expert"
test ! -e "/wip/slots/$to"
CONTAINER
  docker exec -i "$coordinator_container" bash -s -- "$from_slot" "$slot" <<'CONTAINER'
set -euo pipefail
from="$1"
to="$2"
mkdir -p "/wip/slots/$to"
cp -a "/wip/slots/$from/coordinator" "/wip/slots/$to/coordinator"
CONTAINER
  ssh -o BatchMode=yes "$seed_host" docker exec -i "$spark_container" bash -s -- "$from_slot" "$slot" <<'CONTAINER'
set -euo pipefail
from="$1"
to="$2"
test -d "/wip/slots/$from/spark-expert"
mkdir -p "/wip/slots/$to"
cp -a "/wip/slots/$from/spark-expert" "/wip/slots/$to/spark-expert"
CONTAINER
fi

sync_local_source() {
  docker exec "$coordinator_container" rm -rf /wip/source.next
  docker cp "$staging_dir/." "$coordinator_container:/wip/source.next"
  docker exec "$coordinator_container" bash -lc 'rm -rf /wip/source && mv /wip/source.next /wip/source'
}

sync_seed_source() {
  local remote_staging
  remote_staging="$(ssh -o BatchMode=yes "$seed_host" 'printf "%s/.ds41rt-wip-source-staging" "$HOME"')"
  local sync=rsync
  if command -v rdmasync >/dev/null 2>&1 && ssh -o BatchMode=yes "$seed_host" 'command -v rdmasync >/dev/null'; then
    sync=rdmasync
  fi
  ssh -o BatchMode=yes "$seed_host" "mkdir -p '$remote_staging'"
  if [[ "$sync" == rdmasync ]]; then
    rdmasync -a --delete --rdma=required --rdma-show-config "$staging_dir/" "$seed_host:$remote_staging/"
  else
    rsync -a --delete "$staging_dir/" "$seed_host:$remote_staging/"
  fi
  ssh -o BatchMode=yes "$seed_host" \
    "docker exec '$spark_container' rm -rf /wip/source.next && docker cp '$remote_staging/.' '$spark_container:/wip/source.next' && docker exec '$spark_container' bash -lc 'rm -rf /wip/source && mv /wip/source.next /wip/source'"
}

build_coordinator() {
  echo "== incrementally building coordinator slot $slot =="
  sync_local_source
  local image_id
  image_id="$(docker image inspect -f '{{.Id}}' "$COORDINATOR_DOCKER_DEV")"
  docker exec \
    -e "DS41RT_WIP_EXL3_AOT=${DS41RT_WIP_EXL3_AOT:-ON}" \
    -e "DS41RT_WIP_NVFP4_AOT=${DS41RT_WIP_NVFP4_AOT:-ON}" \
    "$coordinator_container" \
    /wip/source/scripts/build-wip-artifacts.sh \
    /wip/source coordinator 120 /wip/build/coordinator /wip/output/coordinator
  docker exec "$coordinator_container" \
    /wip/source/scripts/finalize-wip-slot.sh \
    /wip/source coordinator "$slot" /wip/output/coordinator \
    "$COORDINATOR_DOCKER_DEV" "$image_id"
}

build_expert() {
  echo "== incrementally building Spark expert slot $slot on $seed_host =="
  sync_seed_source
  local image_id
  image_id="$(ssh -o BatchMode=yes "$seed_host" "docker image inspect -f '{{.Id}}' '$SPARK_EXPERT_DOCKER_DEV'")"
  # The role list and build-scope opt-ins travel inside a single quoted remote
  # command so a `tp2;tp3` value is never split by the remote shell.
  ssh -o BatchMode=yes "$seed_host" \
    "docker exec -e 'DS41RT_WIP_SPARK_TP_ROLES=$wip_spark_tp_roles' -e 'DS41RT_WIP_EXL3_AOT=${DS41RT_WIP_EXL3_AOT:-ON}' -e 'DS41RT_WIP_NVFP4_AOT=${DS41RT_WIP_NVFP4_AOT:-ON}' '$spark_container' /wip/source/scripts/build-wip-artifacts.sh /wip/source expert 121 /wip/build/expert /wip/output/expert"
  ssh -o BatchMode=yes "$seed_host" docker exec "$spark_container" \
    /wip/source/scripts/finalize-wip-slot.sh \
    /wip/source spark-expert "$slot" /wip/output/expert \
    "$SPARK_EXPERT_DOCKER_DEV" "$image_id"
}

case "$role" in
  coordinator) build_coordinator ;;
  expert) build_expert ;;
  both)
    build_coordinator &
    coordinator_build_pid=$!
    if ! build_expert; then
      wait "$coordinator_build_pid" || true
      release_die "Spark WIP build failed"
    fi
    wait "$coordinator_build_pid" || release_die "coordinator WIP build failed"
    ;;
esac

# A named slot is launchable only when both role artifacts are present. A
# role-specific rebuild must retain the cloned counterpart even if its build
# helper stages or replaces only the selected role.
if [[ -n "$from_slot" && "$role" == expert ]]; then
  docker exec -i "$coordinator_container" bash -s -- "$from_slot" "$slot" <<'CONTAINER'
set -euo pipefail
from="$1"
to="$2"
if [ ! -d "/wip/slots/$to/coordinator" ]; then
  test -d "/wip/slots/$from/coordinator"
  mkdir -p "/wip/slots/$to"
  cp -a "/wip/slots/$from/coordinator" "/wip/slots/$to/coordinator"
fi
CONTAINER
elif [[ -n "$from_slot" && "$role" == coordinator ]]; then
  ssh -o BatchMode=yes "$seed_host" docker exec -i "$spark_container" bash -s -- "$from_slot" "$slot" <<'CONTAINER'
set -euo pipefail
from="$1"
to="$2"
if [ ! -d "/wip/slots/$to/spark-expert" ]; then
  test -d "/wip/slots/$from/spark-expert"
  mkdir -p "/wip/slots/$to"
  cp -a "/wip/slots/$from/spark-expert" "/wip/slots/$to/spark-expert"
fi
CONTAINER
fi

docker exec -i "$coordinator_container" bash -s -- "$slot" <<'CONTAINER'
set -euo pipefail
slot="$1"
test -s "/wip/slots/$slot/coordinator/FINGERPRINT"
test -s "/wip/slots/$slot/coordinator/workspace/ds41rt.config"
CONTAINER
ssh -o BatchMode=yes "$seed_host" docker exec -i "$spark_container" bash -s -- "$slot" <<'CONTAINER'
set -euo pipefail
slot="$1"
test -s "/wip/slots/$slot/spark-expert/FINGERPRINT"
test -s "/wip/slots/$slot/spark-expert/workspace/ds41rt.config"
CONTAINER

distribute_expert_slot() {
  ssh -o BatchMode=yes "$seed_host" \
    "docker exec '$spark_container' test -s '/wip/slots/$slot/spark-expert/FINGERPRINT'" ||
    return 0
  echo "== concurrently distributing Spark WIP slot $slot from $seed_host =="
  local -a pids=()
  local host
  for host in "${wip_target_hosts[@]}"; do
    (
      set -o pipefail
      ssh -o BatchMode=yes "$host" \
        "docker exec '$spark_container' bash -lc 'rm -rf /wip/incoming/$slot.spark-expert && mkdir -p /wip/incoming/$slot.spark-expert'"
      ssh -o BatchMode=yes "$seed_host" \
        "docker exec '$spark_container' tar -C '/wip/slots/$slot/spark-expert' -cf - . | rdmapipe --send" |
        ssh -o BatchMode=yes "$host" \
          "rdmapipe --recv | docker exec -i '$spark_container' tar -C '/wip/incoming/$slot.spark-expert' -xf -"
      ssh -o BatchMode=yes "$host" \
        "docker exec '$spark_container' bash -lc 'mkdir -p /wip/slots/$slot; rm -rf /wip/slots/$slot/spark-expert; mv /wip/incoming/$slot.spark-expert /wip/slots/$slot/spark-expert'"
    ) &
    pids+=("$!")
  done
  local failed=0 pid
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  ((failed == 0)) || release_die "Spark WIP slot distribution failed"
  local expected actual
  expected="$(ssh -o BatchMode=yes "$seed_host" "docker exec '$spark_container' cat '/wip/slots/$slot/spark-expert/FINGERPRINT'")"
  for host in "${wip_target_hosts[@]}"; do
    actual="$(ssh -o BatchMode=yes "$host" "docker exec '$spark_container' cat '/wip/slots/$slot/spark-expert/FINGERPRINT'")"
    [[ "$actual" == "$expected" ]] || release_die "$host received a mismatched WIP slot fingerprint"
  done
}

distribute_expert_slot
echo "WIP slot '$slot' is ready. Launch it with: ./run.sh --wip --wip-slot '$slot' --restart"
