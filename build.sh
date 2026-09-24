#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$repo_root/scripts/release-common.sh"

usage() {
  cat <<'EOF'
Usage: ./build.sh [--config FILE] [--spark-hosts HOST,...] [--dry-run]

Builds the coordinator image locally and the Spark image natively over SSH on
the first configured Spark. It exports both release artifact sets to dist/
and distributes the Spark inference image to all configured Spark hosts.
Use --spark-hosts ostrich,dodo to build and distribute only on available hosts;
this does not change the serving topology.
Release images are universal by default: the ARM64 (SM121) Spark expert image
carries the TP2, TP3 and TP6 replicated-group shards on top of the always-built
TP4 shard, so one pair serves every approved native topology and ./run.sh selects
the mode with SPARK_TP/SPARK_EP. The x86_64 coordinator image needs no Spark role.
Set DS41RT_RELEASE_SPARK_TP_ROLES to an explicit subset (for example tp6, or empty
for the historical TP4-only shard) for a bounded topology A/B or a legacy rebuild.
--dry-run validates the configuration, host set and role plan without touching
Docker, SSH, submodules or any image.
Set DS41RT_RELEASE_SSH_CONFIG to an ssh config file that every remote step should
use (default empty: stock OpenSSH resolution, so a build host's ~/.ssh/config keeps
working, with BatchMode forced either way). Pass /dev/null to discard a system
ssh_config that OpenSSH refuses to read - but -F replaces the whole config chain, so
that also drops your own host aliases; prefer a file containing just
    Include ~/.ssh/config
and pass that instead. The setting reaches ssh, rsync and rdmasync alike, and is
exported to child build scripts.
Set DS41RT_RELEASE_BUILD_ROOT to one unique absolute path, on a real writable
filesystem with room for a source copy plus objects, to hold the container build
roots of both roles instead of their default /tmp scratch. The same path is bound
into each container, created and filesystem-guarded on the coordinator host and the
seed Spark before any compile, and it is per-task: never share one between builds.
DS41RT_RELEASE_REMOTE_BUILD_DIR selects where the Spark seed host stages the source
tree and builds the expert images (default: a ds41rt-release-build directory in the
seed host's own home). Use a fresh one per release so a previous staging tree cannot
be reused.
DS41RT_RELEASE_SSH_CONFIG, DS41RT_RELEASE_BUILD_ROOT and
DS41RT_RELEASE_REMOTE_BUILD_DIR must each be a canonical absolute path built from
letters, digits, dot, underscore, plus and minus - no spaces, dot segments, trailing
slashes or shell metacharacters - because they reach remote shells and bind mounts.

Images are labelled with the checkout's Git revision (HEAD), even when the
tree has local changes. Dirty checkouts still get an automatic source manifest
under .ds41rt-release/ so local and remote inventories can be verified; keep
source files unchanged during the build. DS41RT_RELEASE_SOURCE_MANIFEST can
supply an existing manifest. Source archives without .git must provide
DS41RT_RELEASE_ENGINE_REVISION (a 40-hex Git revision).
EOF
}

config="$repo_root/ds41rt.config"
build_hosts_csv=""
dry_run=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      config="${2:?$1 requires a configuration file}"
      shift 2
      ;;
    --spark-hosts)
      build_hosts_csv="${2:?--spark-hosts requires a comma-separated host list}"
      shift 2
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
      release_die "unknown build argument: $1"
      ;;
  esac
done

release_load_config "$config"
release_select_build_hosts "$build_hosts_csv"
release_need docker
release_need ssh
release_need rsync
release_need sha256sum
release_need python3
release_need install

release_version="${COORDINATOR_DOCKER_INFERENCE##*:}"
spark_release_version="${SPARK_EXPERT_DOCKER_INFERENCE##*:}"
[[ "$release_version" != "$COORDINATOR_DOCKER_INFERENCE" &&
  "$release_version" =~ ^[A-Za-z0-9_][A-Za-z0-9._-]{0,127}$ ]] ||
  release_die "coordinator inference image must use a valid release tag"
[[ "$spark_release_version" == "$release_version" ]] ||
  release_die "coordinator and Spark inference image release tags must match"

# Spark expert roles are universal by default: the ARM64 (SM121) image carries the
# TP2, TP3 and TP6 replicated-group shards on top of the always-built TP4 shard,
# so one published pair serves every approved native topology (TP4EP1, TP2EP2,
# TP2EP3, TP3EP2, TP6EP1) and ./run.sh selects the mode. The x86_64 coordinator
# needs no Spark role: roles are expert-only, so it is topology-independent.
# DS41RT_RELEASE_SPARK_TP_ROLES is an explicit SUBSET override for a bounded
# topology A/B or a legacy TP4-only rebuild (empty).
# release-spark-tp-roles:start
release_spark_tp_roles_canonical() {
  # Echo a role list sorted and de-duplicated, or die. An unknown token, an empty
  # element, an embedded newline or a duplicate would advertise a topology the
  # image cannot serve. The wholly empty list is the explicit legacy TP4-only
  # request. $2 names the source in diagnostics: the build override or the label.
  local raw="$1" source_name="${2:-DS41RT_RELEASE_SPARK_TP_ROLES}"
  local entry prior
  local -a parts=() selected=()
  [[ -n "$raw" ]] || return 0
  [[ "$raw" != *";;"* && "$raw" != ";"* && "$raw" != *";" && "$raw" != *$'\n'* ]] ||
    release_die "$source_name is not a ';'-separated role list: $raw"
  IFS=';' read -ra parts <<<"$raw"
  for entry in "${parts[@]}"; do
    case "$entry" in
      tp2|tp3|tp6) ;;
      *) release_die "$source_name accepts only tp2, tp3 and tp6, got: $entry" ;;
    esac
    for prior in ${selected[@]+"${selected[@]}"}; do
      [[ "$prior" != "$entry" ]] ||
        release_die "$source_name lists $entry more than once"
    done
    selected+=("$entry")
  done
  printf '%s\n' "${selected[@]}" | sort | paste -sd';' -
}

release_universal_spark_tp_roles="tp2;tp3;tp6"
# `${VAR-default}`, not `:-`, so an explicitly empty override stays the legacy
# TP4-only request rather than falling back to the universal set.
spark_tp_roles="$(release_spark_tp_roles_canonical \
  "${DS41RT_RELEASE_SPARK_TP_ROLES-$release_universal_spark_tp_roles}")"
[[ "$spark_tp_roles" == "$release_universal_spark_tp_roles" ]] ||
  echo "== NON-UNIVERSAL expert role subset '${spark_tp_roles:-<none>}'; this pair cannot serve every approved native topology =="
# Repeated by --dry-run and the build summary so a subset build is never mistaken
# for the universal default.
spark_tp_roles_note='universal, covers every approved native topology'
[[ "$spark_tp_roles" == "$release_universal_spark_tp_roles" ]] ||
  spark_tp_roles_note='explicit subset, not universal'
# release-spark-tp-roles:end

# release-build-transport:start
# The SSH option set and the canonical-path validator live in
# scripts/release-common.sh, so a release build and the Spark-facing helpers cannot
# drift apart: stock resolution unless DS41RT_RELEASE_SSH_CONFIG names a config file,
# BatchMode forced always, and a bad value refused here before any host is reached.
# What stays here is the release-only setting: where a container leg keeps its
# writable build root.
release_configure_ssh_transport

# Optional unique writable build root for the release container legs. Unset keeps
# the historical in-container /tmp scratch; a set value must be the identical path
# inside the container and on the host, because the artifact compiler guards the
# path it writes and that guard resolves the filesystem behind the string it is
# given. It is per-task: two concurrent builds must never share one root.
release_build_root="${DS41RT_RELEASE_BUILD_ROOT:-}"
release_validate_path_setting DS41RT_RELEASE_BUILD_ROOT "$release_build_root"
if [[ -n "$release_build_root" ]]; then
  # A root inside the source tree would be staged into its own copy and then
  # guarded as if it were the source, so the two must stay disjoint. The remote
  # value is still unknown here; the leg that knows it repeats this check.
  release_path_within "$release_build_root" "$repo_root" &&
    release_die "DS41RT_RELEASE_BUILD_ROOT must not be $repo_root or inside it"
  release_build_root_args=(
    -v "$release_build_root:$release_build_root"
    -e "DS41RT_RELEASE_BUILD_ROOT=$release_build_root"
  )
else
  release_build_root_args=()
fi

# Create and probe the build root on the machine that will run the container.
# HOST is empty for the local leg; SCRIPT_DIR is the tree holding
# assert-build-filesystem.py there (the local checkout or the staged remote copy).
# Both operands are %q-quoted, so a validated path cannot be reinterpreted by the
# shell that runs the command.
release_prepare_build_root() {
  [[ -n "$release_build_root" ]] || return 0
  local host="$1" script_dir="$2" quoted_root quoted_dir command
  printf -v quoted_root '%q' "$release_build_root"
  printf -v quoted_dir '%q' "$script_dir"
  command="mkdir -p $quoted_root && python3 $quoted_dir/scripts/assert-build-filesystem.py $quoted_root"
  if [[ -n "$host" ]]; then
    release_ssh "$host" "$command" ||
      release_die "$host release build root is not a safe writable filesystem: $release_build_root"
  else
    bash -c "$command" ||
      release_die "release build root is not a safe writable filesystem: $release_build_root"
  fi
}
# release-build-transport:end

if ((dry_run)); then
  echo "Build dry-run passed; no image, container, SSH or submodule was touched."
  echo "  config: $RELEASE_CONFIG"
  echo "  build hosts (${#RELEASE_BUILD_HOSTS[@]}): $(IFS=,; echo "${RELEASE_BUILD_HOSTS[*]}")"
  echo "  seed host: ${RELEASE_BUILD_HOSTS[0]:-}"
  echo "  release tag: $release_version"
  echo "  V41 Spark expert roles: ${spark_tp_roles:-<legacy TP4 only>} ($spark_tp_roles_note)"
  echo "  coordinator image: $COORDINATOR_DOCKER_INFERENCE"
  echo "  spark image: $SPARK_EXPERT_DOCKER_INFERENCE"
  echo "  ssh config: ${release_ssh_config:-<stock>}"
  echo "  release build root: ${release_build_root:-<container /tmp>}"
  exit 0
fi

prepare_pinned_source_dependencies() {
  local git_root=""
  if command -v git >/dev/null 2>&1; then
    git_root="$(git -C "$repo_root" rev-parse --show-toplevel 2>/dev/null || true)"
  fi

  if [[ -n "$git_root" &&
    "$(cd "$git_root" && pwd -P)" == "$(cd "$repo_root" && pwd -P)" ]]; then
    echo "== preparing pinned source dependencies =="
    git -C "$repo_root" submodule sync -- \
      third_party/sparkinfer third_party/xgrammar
    git -C "$repo_root" submodule update --init --checkout -- \
      third_party/sparkinfer third_party/xgrammar
    git -C "$repo_root/third_party/xgrammar" submodule sync -- \
      3rdparty/dlpack
    git -C "$repo_root/third_party/xgrammar" submodule update --init --checkout -- \
      3rdparty/dlpack
  fi

  [[ -f "$repo_root/third_party/sparkinfer/b12x/__init__.py" ]] ||
    release_die "SparkInfer source is missing; initialize third_party/sparkinfer"
  [[ -f "$repo_root/third_party/xgrammar/include/xgrammar/compiler.h" ]] ||
    release_die "XGrammar source is missing; initialize third_party/xgrammar"
  [[ -f "$repo_root/third_party/xgrammar/3rdparty/dlpack/include/dlpack/dlpack.h" ]] ||
    release_die "XGrammar DLPack source is missing; initialize third_party/xgrammar/3rdparty/dlpack"
}

prepare_pinned_source_dependencies

docker info >/dev/null 2>&1 || release_die "local Docker daemon is unavailable"
detected_engine_commit="$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || true)"
engine_source_dirty=0
if [[ -n "$detected_engine_commit" &&
  -n "$(git -C "$repo_root" status --porcelain 2>/dev/null || true)" ]]; then
  engine_source_dirty=1
fi
source_manifest="${DS41RT_RELEASE_SOURCE_MANIFEST:-}"
source_manifest_sha256=""
if [[ -z "$source_manifest" ]] && ((engine_source_dirty)); then
  # Build provenance is generated by the build, not an extra manual prerequisite.
  # This directory is excluded from both the inventory and remote source sync.
  mkdir -p "$repo_root/.ds41rt-release/source-manifests"
  source_manifest="$(mktemp "$repo_root/.ds41rt-release/source-manifests/source.XXXXXXXX.sha256")"
  python3 "$repo_root/scripts/verify-release-source-manifest.py" \
    --source "$repo_root" --write "$source_manifest"
  echo "== recorded current checkout: $source_manifest =="
fi
if [[ -n "$source_manifest" ]]; then
  source_manifest="$(
    python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' \
      "$source_manifest"
  )"
  [[ -f "$source_manifest" ]] ||
    release_die "release source manifest not found: $source_manifest"
  source_manifest_sha256="$(sha256sum "$source_manifest" | awk '{print $1}')"
fi

verify_local_source_manifest() {
  [[ -n "$source_manifest" ]] || return 0
  local current_source_manifest_sha256
  current_source_manifest_sha256="$(
    sha256sum "$source_manifest" | awk '{print $1}'
  )"
  [[ "$current_source_manifest_sha256" == "$source_manifest_sha256" ]] ||
    release_die "release source manifest changed during the build"
  python3 "$repo_root/scripts/verify-release-source-manifest.py" \
    --source "$repo_root" \
    --manifest "$source_manifest" ||
    release_die "release source differs from $source_manifest"
}

verify_remote_source_manifest() {
  [[ -n "$source_manifest" ]] || return 0
  local current_source_manifest_sha256
  current_source_manifest_sha256="$(
    sha256sum "$source_manifest" | awk '{print $1}'
  )"
  [[ "$current_source_manifest_sha256" == "$source_manifest_sha256" ]] ||
    release_die "release source manifest changed during the build"
  local remote_dir_quoted
  printf -v remote_dir_quoted '%q' "$remote_dir"
  release_ssh "$seed_host" \
    "cd $remote_dir_quoted && python3 scripts/verify-release-source-manifest.py --source . --manifest -" \
    <"$source_manifest" ||
    release_die "$seed_host staged source differs from $source_manifest"
}

verify_local_source_manifest
sparkinfer_commit="$(
  python3 "$repo_root/scripts/verify-sparkinfer-source.py" \
    --source "$repo_root/third_party/sparkinfer" \
    --lock "$repo_root/third_party/sparkinfer.lock.json" \
    --print-revision
)"
python3 "$repo_root/scripts/verify-xgrammar-source.py" \
  --source "$repo_root/third_party/xgrammar" \
  --lock "$repo_root/third_party/xgrammar.lock.json"
engine_revision_override="${DS41RT_RELEASE_ENGINE_REVISION:-}"
if [[ -n "$engine_revision_override" ]]; then
  [[ "$engine_revision_override" =~ ^[0-9a-f]{40}(-dirty-[0-9a-f]{12})?$ ]] ||
    release_die "DS41RT_RELEASE_ENGINE_REVISION must be REVISION or REVISION-dirty-MANIFEST12"
  [[ -n "$source_manifest_sha256" ]] ||
    release_die "DS41RT_RELEASE_ENGINE_REVISION requires DS41RT_RELEASE_SOURCE_MANIFEST"
  engine_commit="$engine_revision_override"
elif [[ -z "$detected_engine_commit" ]]; then
  release_die "source snapshot has no Git metadata; set DS41RT_RELEASE_ENGINE_REVISION and DS41RT_RELEASE_SOURCE_MANIFEST"
elif ((engine_source_dirty)); then
  # Label the Git revision; the build does not encode local changes.
  echo "== note: building $detected_engine_commit with uncommitted local changes =="
  engine_commit="$detected_engine_commit"
else
  engine_commit="$detected_engine_commit"
fi
if [[ "$engine_commit" == *-dirty-* ]]; then
  [[ -n "$source_manifest_sha256" ]] ||
    release_die "dirty engine revision requires DS41RT_RELEASE_SOURCE_MANIFEST"
  [[ "$engine_commit" == *"-dirty-${source_manifest_sha256:0:12}" ]] ||
    release_die "dirty engine revision suffix does not match the source manifest"
fi
release_source_label_args=()
if [[ -n "$source_manifest_sha256" ]]; then
  release_source_label_args+=(
    --label "io.ds41rt.source-manifest.sha256=$source_manifest_sha256"
  )
fi

hosts_csv="$(IFS=,; echo "${RELEASE_BUILD_HOSTS[*]}")"
seed_host="${RELEASE_BUILD_HOSTS[0]}"
remote_dir="${DS41RT_RELEASE_REMOTE_BUILD_DIR:-}"
artifact_dir="$repo_root/.ds41rt-release-image"

echo "== validating native Spark build hosts =="
for host in "${RELEASE_BUILD_HOSTS[@]}"; do
  release_ssh -o ConnectTimeout=10 "$host" bash -s <<'REMOTE'
set -euo pipefail
command -v docker >/dev/null
docker info >/dev/null
test "$(uname -m)" = "aarch64"
REMOTE
  echo "  $host: ssh/docker/aarch64 ready"
done

release_sync_program=rsync
if command -v rdmasync >/dev/null 2>&1 &&
  release_ssh "$seed_host" "command -v rdmasync >/dev/null 2>&1"; then
  release_sync_program=rdmasync
  echo "== using RDMA source/artifact synchronization =="
fi

release_sync() {
  # Transport options live in one place: rsync and rdmasync both accept the
  # remote shell as a single --rsh value, so the ssh config override reaches the
  # source and artifact copies as well as the explicit ssh calls above. A bare
  # "-F" argument here would instead be their own --filter=dir-merge option.
  if [[ "$release_sync_program" == rdmasync ]]; then
    rdmasync -a --rdma=required --rdma-show-config --rsh="$release_rsh" "$@"
  else
    rsync -a --rsh="$release_rsh" "$@"
  fi
}

if [[ -z "$remote_dir" ]]; then
  remote_dir="$(
    release_ssh "$seed_host" \
      'printf "%s/ds41rt-release-build" "$HOME"'
  )"
fi

# The Spark staging directory is embedded in remote shell command strings and in
# Docker bind sources, so it gets the same canonical treatment as every other
# setting; the default derived from the seed host's own $HOME is checked too rather
# than assumed to be simple.
release_validate_path_setting DS41RT_RELEASE_REMOTE_BUILD_DIR "$remote_dir"
if [[ -n "$release_build_root" ]]; then
  # The remote source tree and the remote scratch must stay disjoint: a build root
  # inside the staged source would be copied into its own build directory, and a
  # source tree inside the scratch would be deleted with it.
  release_path_within "$release_build_root" "$remote_dir" &&
    release_die "DS41RT_RELEASE_BUILD_ROOT must not be $remote_dir or inside it"
  release_path_within "$remote_dir" "$release_build_root" &&
    release_die "the Spark staging directory must not be inside DS41RT_RELEASE_BUILD_ROOT"
fi

local_free_kib="$(df -Pk "$repo_root" | awk 'NR==2 {print $4}')"
((local_free_kib >= 60 * 1024 * 1024)) || release_die "local build needs at least 60 GiB free"
remote_free_kib="$(
  release_ssh "$seed_host" bash -s <<'REMOTE'
df -Pk "$HOME" | awk 'NR == 2 { print $4 }'
REMOTE
)"
((remote_free_kib >= 60 * 1024 * 1024)) || release_die "$seed_host build needs at least 60 GiB free"

# A relocated build root is its own filesystem decision: the two checks above
# cover the source tree and the Spark home, not the scratch the container writes.
# The remote mkdir is where the directory starts to exist, so the same path can be
# bind-mounted into the container without Docker creating it as root-owned later.
if [[ -n "$release_build_root" ]]; then
  # Create first: `df` on a path that does not exist yields no capacity, which an
  # arithmetic comparison would silently read as zero and report as a space problem.
  mkdir -p "$release_build_root" ||
    release_die "cannot create release build root: $release_build_root"
  local_root_free_kib="$(df -Pk "$release_build_root" | awk 'NR==2 {print $4}')"
  ((local_root_free_kib >= 60 * 1024 * 1024)) ||
    release_die "release build root $release_build_root needs at least 60 GiB free"
  printf -v release_build_root_quoted '%q' "$release_build_root"
  remote_root_free_kib="$(
    release_ssh "$seed_host" \
      "mkdir -p $release_build_root_quoted && df -Pk $release_build_root_quoted | awk 'NR == 2 { print \$4 }'"
  )"
  ((remote_root_free_kib >= 60 * 1024 * 1024)) ||
    release_die "$seed_host release build root $release_build_root needs at least 60 GiB free"
fi

echo "== building coordinator development image: $COORDINATOR_DOCKER_DEV =="
docker build \
  --build-arg DS41RT_ROLE=coordinator \
  --build-arg CUDA_ARCH=120 \
  --build-arg TARGET_PLATFORM=linux/amd64 \
  --build-arg DS41RT_SPARKINFER_COMMIT="$sparkinfer_commit" \
  -f "$repo_root/docker/Dockerfile.dev" \
  -t "$COORDINATOR_DOCKER_DEV" \
  "$repo_root"

echo "== compiling coordinator release artifacts in GPU-enabled development container =="
mkdir -p "$artifact_dir"
release_prepare_build_root "" "$repo_root"
docker run --rm \
  --gpus device=0 \
  --ipc=host \
  --ulimit memlock=-1:-1 \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  ${release_build_root_args[@]+"${release_build_root_args[@]}"} \
  -v "$repo_root:/source:ro" \
  -v "$artifact_dir:/output" \
  "$COORDINATOR_DOCKER_DEV" \
  /source/scripts/build-release-artifacts.sh /source coordinator 120 /output

echo "== building coordinator inference image: $COORDINATOR_DOCKER_INFERENCE =="
docker build \
  "${release_source_label_args[@]}" \
  --build-arg DS41RT_ROLE=coordinator \
  --build-arg CUDA_ARCH=120 \
  --build-arg DS41RT_ENGINE_COMMIT="$engine_commit" \
  --build-arg DS41RT_SPARKINFER_COMMIT="$sparkinfer_commit" \
  --build-arg DS41RT_RELEASE_VERSION="$release_version" \
  --build-arg DS41RT_V41_SPARK_TP_ROLES= \
  -f "$repo_root/docker/Dockerfile.release" \
  -t "$COORDINATOR_DOCKER_INFERENCE" \
  "$repo_root"

echo "== staging native Spark build on $seed_host:$remote_dir =="
printf -v remote_dir_quoted '%q' "$remote_dir"
release_ssh "$seed_host" "mkdir -p $remote_dir_quoted"
release_sync --delete \
  --exclude '.git' \
  --exclude '.venv*/' \
  --exclude '.mypy_cache/' \
  --exclude '.pytest_cache/' \
  --exclude '.ruff_cache/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '*.pyo' \
  --exclude '.ds41rt-cache/' \
  --exclude '.ds41rt-release/' \
  --exclude '.ds41rt-release-image/' \
  --exclude '.ds41rt-wip/' \
  --exclude 'dist/' \
  --exclude 'rust/target/' \
  --exclude 'native/build*/' \
  "$repo_root/" "$seed_host:$remote_dir/"
# The broad staging sync protects excluded paths from deletion. Reconcile the
# pinned source separately so bytecode left by an earlier build cannot survive
# merely because it is now excluded.
release_sync --delete --delete-excluded \
  --exclude '.git' \
  --exclude '.venv*/' \
  --exclude '.mypy_cache/' \
  --exclude '.pytest_cache/' \
  --exclude '.ruff_cache/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '*.pyo' \
  "$repo_root/third_party/sparkinfer/" \
  "$seed_host:$remote_dir/third_party/sparkinfer/"
release_sync --delete --delete-excluded \
  --exclude '.git' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '*.pyo' \
  "$repo_root/third_party/xgrammar/" \
  "$seed_host:$remote_dir/third_party/xgrammar/"
verify_remote_source_manifest

# Prepared before the leg below so the quoted heredoc region stays exactly the
# argument transport it is tested as: creating and probing the root is a host-side
# step, not part of what the remote shell receives.
release_prepare_build_root "$seed_host" "$remote_dir"
echo "== building Spark development and inference images natively on $seed_host =="
release_ssh "$seed_host" bash -s -- \
  "$remote_dir" "$SPARK_EXPERT_DOCKER_DEV" "$SPARK_EXPERT_DOCKER_INFERENCE" \
  "$engine_commit" "$sparkinfer_commit" "$release_version" \
  "$EXL3_PAIRED_TP4" "${source_manifest_sha256:-__legacy__}" "${spark_tp_roles//;/,}" \
  "${release_build_root:-__legacy__}" <<'REMOTE'
set -euo pipefail
remote_dir="$1"
dev_image="$2"
inference_image="$3"
engine_commit="$4"
sparkinfer_commit="$5"
release_version="$6"
exl3_paired_tp4="$7"
# SSH reconstructs a shell command and can omit an empty argument, so every
# optional trailing value is passed as a non-empty sentinel and decoded here.
# Both source_manifest_sha256 and spark_tp_roles are optional: sentinels keep
# the two from shifting into each other's position.
source_manifest_sha256="${8-__legacy__}"
# The role list travels as a comma list so a remote shell cannot split it at a
# semicolon; it is restored to the CMake semicolon list here.
spark_tp_roles="${9-__legacy__}"
# The relocated build root is optional and travels last, for the same reason.
# It is already created and filesystem-guarded by the caller, on this host.
release_build_root="${10-__legacy__}"
[[ "$source_manifest_sha256" != "__legacy__" ]] || source_manifest_sha256=
[[ "$spark_tp_roles" != "__legacy__" ]] || spark_tp_roles=
[[ "$release_build_root" != "__legacy__" ]] || release_build_root=
spark_tp_roles="${spark_tp_roles//,/;}"
release_build_root_args=()
if [[ -n "$release_build_root" ]]; then
  release_build_root_args=(
    -v "$release_build_root:$release_build_root"
    -e "DS41RT_RELEASE_BUILD_ROOT=$release_build_root"
  )
fi
release_source_label_args=()
if [[ -n "$source_manifest_sha256" ]]; then
  release_source_label_args+=(
    --label "io.ds41rt.source-manifest.sha256=$source_manifest_sha256"
  )
fi
cd "$remote_dir"
python3 scripts/verify-sparkinfer-source.py \
  --source third_party/sparkinfer \
  --lock third_party/sparkinfer.lock.json \
  --require-no-python-cache
docker build \
  --build-arg DS41RT_ROLE=expert \
  --build-arg CUDA_ARCH=121 \
  --build-arg TARGET_PLATFORM=linux/arm64 \
  --build-arg DS41RT_SPARKINFER_COMMIT="$sparkinfer_commit" \
  -f docker/Dockerfile.dev \
  -t "$dev_image" .
mkdir -p .ds41rt-release-image
docker run --rm \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1:-1 \
  -e "DS41RT_RELEASE_EXL3_PAIRED_TP4=$exl3_paired_tp4" \
  -e "DS41RT_RELEASE_SPARK_TP_ROLES=$spark_tp_roles" \
  ${release_build_root_args[@]+"${release_build_root_args[@]}"} \
  -v "$remote_dir:/source:ro" \
  -v "$remote_dir/.ds41rt-release-image:/output" \
  "$dev_image" \
  /source/scripts/build-release-artifacts.sh /source expert 121 /output
docker build \
  "${release_source_label_args[@]}" \
  --build-arg DS41RT_ROLE=expert \
  --build-arg CUDA_ARCH=121 \
  --build-arg DS41RT_ENGINE_COMMIT="$engine_commit" \
  --build-arg DS41RT_SPARKINFER_COMMIT="$sparkinfer_commit" \
  --build-arg DS41RT_RELEASE_VERSION="$release_version" \
  --build-arg DS41RT_V41_SPARK_TP_ROLES="$spark_tp_roles" \
  -f docker/Dockerfile.release \
  -t "$inference_image" .
REMOTE
verify_remote_source_manifest

echo "== exporting release binaries =="
mkdir -p "$repo_root/dist/coordinator" "$repo_root/dist/spark-expert"
find "$repo_root/dist/coordinator" "$repo_root/dist/spark-expert" \
  -mindepth 1 -delete
coordinator_container="$(docker create "$COORDINATOR_DOCKER_INFERENCE")"
trap 'docker rm -f "$coordinator_container" >/dev/null 2>&1 || true' EXIT
docker cp "$coordinator_container:/opt/ds41rt/bin/ds41rt" "$repo_root/dist/coordinator/ds41rt"
docker cp "$coordinator_container:/opt/ds41rt/lib/libds41rt_native.so" "$repo_root/dist/coordinator/libds41rt_native.so"
docker cp "$coordinator_container:/opt/ds41rt/lib/exl3" "$repo_root/dist/coordinator/exl3"
docker cp "$coordinator_container:/opt/ds41rt/share/V41_EXPERT_AOT.json" "$repo_root/dist/coordinator/V41_EXPERT_AOT.json"
docker cp "$coordinator_container:/opt/ds41rt/share/V41_EXPERT_TP_AOT.json" "$repo_root/dist/coordinator/V41_EXPERT_TP_AOT.json"
docker cp "$coordinator_container:/opt/ds41rt/share/V41_FP8_AOT.json" "$repo_root/dist/coordinator/V41_FP8_AOT.json"
docker cp \
  "$coordinator_container:/opt/ds41rt/share/THIRD_PARTY_NOTICES.md" \
  "$repo_root/dist/coordinator/THIRD_PARTY_NOTICES.md"
docker cp \
  "$coordinator_container:/opt/ds41rt/share/SPARKINFER_PROVENANCE.json" \
  "$repo_root/dist/coordinator/SPARKINFER_PROVENANCE.json"
docker cp \
  "$coordinator_container:/opt/ds41rt/share/licenses/sparkinfer/LICENSE" \
  "$repo_root/dist/coordinator/SPARKINFER_LICENSE"
docker cp \
  "$coordinator_container:/opt/ds41rt/share/SPARKINFER_SHA256SUMS" \
  "$repo_root/dist/coordinator/SPARKINFER_SHA256SUMS"
docker cp \
  "$coordinator_container:/opt/ds41rt/share/XGRAMMAR_PROVENANCE.json" \
  "$repo_root/dist/coordinator/XGRAMMAR_PROVENANCE.json"
docker cp \
  "$coordinator_container:/opt/ds41rt/share/licenses/xgrammar/LICENSE" \
  "$repo_root/dist/coordinator/XGRAMMAR_LICENSE"
docker cp \
  "$coordinator_container:/opt/ds41rt/share/XGRAMMAR_SHA256SUMS" \
  "$repo_root/dist/coordinator/XGRAMMAR_SHA256SUMS"
docker rm "$coordinator_container" >/dev/null
trap - EXIT

release_ssh "$seed_host" bash -s -- \
  "$SPARK_EXPERT_DOCKER_INFERENCE" "$remote_dir/dist/spark-expert" <<'REMOTE'
set -euo pipefail
image="$1"
destination="$2"
mkdir -p "$destination"
find "$destination" -mindepth 1 -delete
container="$(docker create "$image")"
trap 'docker rm -f "$container" >/dev/null 2>&1 || true' EXIT
docker cp "$container:/opt/ds41rt/bin/ds41rt" "$destination/ds41rt"
docker cp "$container:/opt/ds41rt/lib/libds41rt_native.so" "$destination/libds41rt_native.so"
docker cp "$container:/opt/ds41rt/lib/exl3" "$destination/exl3"
docker cp "$container:/opt/ds41rt/share/V41_EXPERT_AOT.json" "$destination/V41_EXPERT_AOT.json"
docker cp "$container:/opt/ds41rt/share/V41_EXPERT_TP_AOT.json" "$destination/V41_EXPERT_TP_AOT.json"
docker cp "$container:/opt/ds41rt/share/V41_FP8_AOT.json" "$destination/V41_FP8_AOT.json"
docker cp \
  "$container:/opt/ds41rt/share/THIRD_PARTY_NOTICES.md" \
  "$destination/THIRD_PARTY_NOTICES.md"
docker cp \
  "$container:/opt/ds41rt/share/SPARKINFER_PROVENANCE.json" \
  "$destination/SPARKINFER_PROVENANCE.json"
docker cp \
  "$container:/opt/ds41rt/share/licenses/sparkinfer/LICENSE" \
  "$destination/SPARKINFER_LICENSE"
docker cp \
  "$container:/opt/ds41rt/share/SPARKINFER_SHA256SUMS" \
  "$destination/SPARKINFER_SHA256SUMS"
docker cp \
  "$container:/opt/ds41rt/share/XGRAMMAR_PROVENANCE.json" \
  "$destination/XGRAMMAR_PROVENANCE.json"
docker cp \
  "$container:/opt/ds41rt/share/licenses/xgrammar/LICENSE" \
  "$destination/XGRAMMAR_LICENSE"
docker cp \
  "$container:/opt/ds41rt/share/XGRAMMAR_SHA256SUMS" \
  "$destination/XGRAMMAR_SHA256SUMS"
docker rm "$container" >/dev/null
trap - EXIT
REMOTE
release_sync --delete \
  "$seed_host:$remote_dir/dist/spark-expert/" \
  "$repo_root/dist/spark-expert/"
dist_source_manifest=()
if [[ -n "$source_manifest" ]]; then
  install -m 0644 "$source_manifest" "$repo_root/dist/SOURCE_SHA256SUMS"
  dist_source_manifest+=(SOURCE_SHA256SUMS)
fi
for role in coordinator spark-expert; do
  python3 "$repo_root/scripts/sparkinfer-release-provenance.py" \
    --source "$repo_root/third_party/sparkinfer" \
    --lock "$repo_root/third_party/sparkinfer.lock.json" \
    --license "$repo_root/dist/$role/SPARKINFER_LICENSE" \
    --notices "$repo_root/dist/$role/THIRD_PARTY_NOTICES.md" \
    --verify "$repo_root/dist/$role/SPARKINFER_PROVENANCE.json"
  (
    python3 "$repo_root/python/tools/package_v41_exl3_aot.py" verify --package "$repo_root/dist/$role/exl3" --sparkinfer-revision "$sparkinfer_commit"
    cd "$repo_root/dist/$role"
    sha256sum -c SPARKINFER_SHA256SUMS
    sha256sum -c XGRAMMAR_SHA256SUMS
  )
done
# EXL3 manifests live either flat in the exl3 root (legacy single family) or
# nested per decoder-tier family (multi-family release images). Collect whichever
# layout the build produced so the checksum list never names a missing path.
dist_exl3_manifests=()
for role in coordinator spark-expert; do
  if [[ -f "$repo_root/dist/$role/exl3/manifest.json" ]]; then
    dist_exl3_manifests+=("$role/exl3/manifest.json")
  else
    for manifest in "$repo_root/dist/$role"/exl3/exl3-*/manifest.json; do
      [[ -f "$manifest" ]] || continue
      dist_exl3_manifests+=("${manifest#"$repo_root/dist/"}")
    done
  fi
done
(
  cd "$repo_root/dist"
  sha256sum \
    coordinator/ds41rt coordinator/libds41rt_native.so coordinator/V41_EXPERT_AOT.json coordinator/V41_EXPERT_TP_AOT.json coordinator/V41_FP8_AOT.json \
    coordinator/THIRD_PARTY_NOTICES.md \
    coordinator/SPARKINFER_PROVENANCE.json \
    coordinator/SPARKINFER_LICENSE \
    coordinator/SPARKINFER_SHA256SUMS \
    coordinator/XGRAMMAR_PROVENANCE.json \
    coordinator/XGRAMMAR_LICENSE \
    coordinator/XGRAMMAR_SHA256SUMS \
    spark-expert/ds41rt spark-expert/libds41rt_native.so spark-expert/V41_EXPERT_AOT.json spark-expert/V41_EXPERT_TP_AOT.json spark-expert/V41_FP8_AOT.json \
    spark-expert/THIRD_PARTY_NOTICES.md \
    spark-expert/SPARKINFER_PROVENANCE.json \
    spark-expert/SPARKINFER_LICENSE \
    spark-expert/SPARKINFER_SHA256SUMS \
    spark-expert/XGRAMMAR_PROVENANCE.json \
    spark-expert/XGRAMMAR_LICENSE \
    spark-expert/XGRAMMAR_SHA256SUMS \
    "${dist_exl3_manifests[@]}" \
    "${dist_source_manifest[@]}" >SHA256SUMS
  sha256sum -c SHA256SUMS
)
verify_local_source_manifest
verify_remote_source_manifest

echo "== distributing fresh Spark inference image =="
for host in "${RELEASE_BUILD_HOSTS[@]:1}"; do
  # A stopped expert container can still reference the prior image ID. Force
  # removal only untags that image while preserving the referenced layers, so
  # ensure_image cannot mistake the stale tag for the fresh seed image.
  release_ssh "$host" "docker image rm --force '$SPARK_EXPERT_DOCKER_INFERENCE' >/dev/null 2>&1 || true"
done
rdmapipe_ready=1
for host in "${RELEASE_BUILD_HOSTS[@]}"; do
  if ! release_ssh "$host" "command -v rdmapipe >/dev/null 2>&1"; then
    rdmapipe_ready=0
    break
  fi
done
if ((rdmapipe_ready)); then
  echo "== concurrently distributing Spark image over RDMA =="
  image_copy_pids=()
  image_copy_hosts=()
  printf -v spark_image_quoted '%q' "$SPARK_EXPERT_DOCKER_INFERENCE"
  for host in "${RELEASE_BUILD_HOSTS[@]:1}"; do
    (
      set -o pipefail
      echo "== RDMA image copy $seed_host -> $host =="
      release_ssh "$seed_host" \
        "set -o pipefail; docker image save $spark_image_quoted | rdmapipe --send" |
        release_ssh "$host" \
          "set -o pipefail; rdmapipe --recv | docker image load"
      echo "== RDMA image copy $seed_host -> $host complete =="
    ) &
    image_copy_pids+=("$!")
    image_copy_hosts+=("$host")
  done
  image_copy_failed=0
  for index in "${!image_copy_pids[@]}"; do
    if ! wait "${image_copy_pids[$index]}"; then
      echo "RDMA image copy to ${image_copy_hosts[$index]} failed" >&2
      image_copy_failed=1
    fi
  done
  ((image_copy_failed == 0)) || release_die "concurrent RDMA Spark image distribution failed"
else
  echo "== rdmapipe unavailable on one or more Sparks; using serial netcat image distribution =="
  DS41RT_SPARK_HOSTS="$hosts_csv" \
  DS41RT_SPARK_IMAGE="$SPARK_EXPERT_DOCKER_INFERENCE" \
  DS41RT_SPARK_IMAGE_SEED_HOST="$seed_host" \
  DS41RT_SPARK_IMAGE_COPY_METHOD=spark-netcat \
  DS41RT_SPARK_IMAGE_ONLY=1 \
  DS41RT_SPARK_SKIP_STAGE=1 \
  RSYNC_RSH="$release_rsh" \
  DS41RT_RELEASE_SSH_CONFIG="$release_ssh_config" \
  "$repo_root/scripts/phase0-spark-tcp-bench.sh"
fi

coordinator_revision="$(
  docker image inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' \
    "$COORDINATOR_DOCKER_INFERENCE"
)"
[[ "$coordinator_revision" == "$engine_commit" ]] || release_die "coordinator image revision mismatch"
coordinator_version="$(
  docker image inspect -f '{{index .Config.Labels "org.opencontainers.image.version"}}' \
    "$COORDINATOR_DOCKER_INFERENCE"
)"
[[ "$coordinator_version" == "$release_version" ]] || release_die "coordinator image version mismatch"
coordinator_sparkinfer_revision="$(
  docker image inspect -f '{{index .Config.Labels "io.ds41rt.sparkinfer.revision"}}' \
    "$COORDINATOR_DOCKER_INFERENCE"
)"
[[ "$coordinator_sparkinfer_revision" == "$sparkinfer_commit" ]] ||
  release_die "coordinator image SparkInfer revision mismatch"
if [[ -n "$source_manifest_sha256" ]]; then
  coordinator_source_manifest="$(
    docker image inspect -f '{{index .Config.Labels "io.ds41rt.source-manifest.sha256"}}' \
      "$COORDINATOR_DOCKER_INFERENCE"
  )"
  [[ "$coordinator_source_manifest" == "$source_manifest_sha256" ]] ||
    release_die "coordinator image source manifest mismatch"
fi
for host in "${RELEASE_BUILD_HOSTS[@]}"; do
  revision="$(
    release_ssh "$host" \
      "docker image inspect -f '{{index .Config.Labels \"org.opencontainers.image.revision\"}}' '$SPARK_EXPERT_DOCKER_INFERENCE'"
  )"
  [[ "$revision" == "$engine_commit" ]] || release_die "$host Spark image revision mismatch: $revision"
  spark_version="$(
    release_ssh "$host" \
      "docker image inspect -f '{{index .Config.Labels \"org.opencontainers.image.version\"}}' '$SPARK_EXPERT_DOCKER_INFERENCE'"
  )"
  [[ "$spark_version" == "$release_version" ]] ||
    release_die "$host Spark image version mismatch: $spark_version"
  spark_sparkinfer_revision="$(
    release_ssh "$host" \
      "docker image inspect -f '{{index .Config.Labels \"io.ds41rt.sparkinfer.revision\"}}' '$SPARK_EXPERT_DOCKER_INFERENCE'"
  )"
  [[ "$spark_sparkinfer_revision" == "$sparkinfer_commit" ]] ||
    release_die "$host Spark image SparkInfer revision mismatch: $spark_sparkinfer_revision"
  if [[ -n "$source_manifest_sha256" ]]; then
    spark_source_manifest="$(
      release_ssh "$host" \
        "docker image inspect -f '{{index .Config.Labels \"io.ds41rt.source-manifest.sha256\"}}' '$SPARK_EXPERT_DOCKER_INFERENCE'"
    )"
    [[ "$spark_source_manifest" == "$source_manifest_sha256" ]] ||
      release_die "$host Spark image source manifest mismatch: $spark_source_manifest"
  fi
  # The advertised role set must equal what was requested and actually built;
  # this is the identity the release launcher verifies before an explicit
  # topology launch, so a mismatch is a hard build failure.
  spark_role_label="$(
    release_ssh "$host" \
      "docker image inspect -f '{{index .Config.Labels \"io.ds41rt.v41.spark_tp_roles\"}}' '$SPARK_EXPERT_DOCKER_INFERENCE'"
  )"
  [[ "$spark_role_label" != "<no value>" ]] || spark_role_label=
  # release-spark-tp-roles-postcheck:start
  # One canonicalizer for both sides of the comparison, so a permutation or a
  # stray separator cannot make an equal set look unequal (or the reverse).
  [[ "$(release_spark_tp_roles_canonical "$spark_role_label" \
    "io.ds41rt.v41.spark_tp_roles")" == "$spark_tp_roles" ]] ||
    release_die "$host Spark image advertises expert roles '$spark_role_label', expected exactly '$spark_tp_roles'"
  # release-spark-tp-roles-postcheck:end
done

echo "Build complete."
echo "  revision:    $engine_commit"
echo "  SparkInfer:  $sparkinfer_commit"
if [[ -n "$source_manifest_sha256" ]]; then
  echo "  source:      $source_manifest_sha256"
fi
echo "  expert roles: ${spark_tp_roles:-<none>} ($spark_tp_roles_note)"
echo "  coordinator: $COORDINATOR_DOCKER_INFERENCE"
echo "  spark:       $SPARK_EXPERT_DOCKER_INFERENCE"
echo "  artifacts:   $repo_root/dist"
