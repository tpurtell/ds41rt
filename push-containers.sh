#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$repo_root/scripts/release-common.sh"

usage() {
  cat <<'EOF'
Usage: ./push-containers.sh TAG

Tags and pushes the current coordinator and Spark inference images to GHCR.
The supplied release tag and latest are published for both images. The
coordinator image is local; the Spark image is published from SPARK_0_HOST.
The Spark image must advertise the V41 expert roles it carries (./build.sh bakes
the universal tp2;tp3;tp6 set by default); a role-less legacy build is rejected.

Example:
  ./push-containers.sh v4
EOF
}

if [[ $# -eq 1 && ("$1" == -h || "$1" == --help) ]]; then
  usage
  exit 0
fi
[[ $# -eq 1 ]] || {
  usage >&2
  exit 2
}

tag="$1"
[[ "$tag" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$ ]] ||
  release_die "invalid Docker tag: $tag"
[[ "$tag" != latest ]] ||
  release_die "provide a version tag; latest is published automatically"

release_load_config "$repo_root/ds41rt.config"
release_need docker
release_need ssh

coordinator_repository="ghcr.io/tpurtell/ds41rt-coordinator"
spark_repository="ghcr.io/tpurtell/ds41rt-spark-expert"
spark_host="$SPARK_0_HOST"
expected_source="https://github.com/tpurtell/ds41rt"

# The published pair is universal: ./build.sh bakes the TP2/TP3/TP6 Spark expert
# shards by default on top of the always-built TP4 shard, and the release launcher
# refuses an explicit SPARK_TP/SPARK_EP topology without the matching role.
# push-universal-role-guard:start
push_require_universal_roles() {
  local advertised="$1" image="$2" role
  for role in tp2 tp3 tp6; do
    [[ ";$advertised;" == *";$role;"* ]] ||
      release_die "$image does not advertise Spark expert role '$role' (advertised: '${advertised:-<none>}'); refusing to publish a legacy or subset build as the universal release pair. Rebuild it with ./build.sh, whose default is tp2;tp3;tp6."
  done
}
# push-universal-role-guard:end

docker info >/dev/null 2>&1 ||
  release_die "local Docker daemon is unavailable"
docker image inspect "$COORDINATOR_DOCKER_INFERENCE" >/dev/null 2>&1 ||
  release_die "coordinator image is missing: $COORDINATOR_DOCKER_INFERENCE"

ssh -o BatchMode=yes -o ConnectTimeout=10 "$spark_host" bash -s -- \
  "$SPARK_EXPERT_DOCKER_INFERENCE" <<'REMOTE'
set -euo pipefail
image="$1"
docker info >/dev/null
docker image inspect "$image" >/dev/null
REMOTE

coordinator_revision="$(
  docker image inspect \
    -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' \
    "$COORDINATOR_DOCKER_INFERENCE"
)"
spark_revision="$(
  ssh -o BatchMode=yes "$spark_host" bash -s -- \
    "$SPARK_EXPERT_DOCKER_INFERENCE" <<'REMOTE'
set -euo pipefail
docker image inspect \
  -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$1"
REMOTE
)"
coordinator_source="$(
  docker image inspect \
    -f '{{index .Config.Labels "org.opencontainers.image.source"}}' \
    "$COORDINATOR_DOCKER_INFERENCE"
)"
spark_source="$(
  ssh -o BatchMode=yes "$spark_host" bash -s -- \
    "$SPARK_EXPERT_DOCKER_INFERENCE" <<'REMOTE'
set -euo pipefail
docker image inspect \
  -f '{{index .Config.Labels "org.opencontainers.image.source"}}' "$1"
REMOTE
)"
[[ -n "$coordinator_revision" && "$coordinator_revision" != "<no value>" ]] ||
  release_die "coordinator image has no engine revision label"
[[ -n "$spark_revision" && "$spark_revision" != "<no value>" ]] ||
  release_die "$spark_host Spark image has no engine revision label"
[[ "$coordinator_revision" == "$spark_revision" ]] ||
  release_die "image revision mismatch: coordinator=$coordinator_revision spark=$spark_revision"
[[ "$coordinator_source" == "$expected_source" ]] ||
  release_die "coordinator image is not linked to the release repository: $coordinator_source"
[[ "$spark_source" == "$expected_source" ]] ||
  release_die "$spark_host image is not linked to the release repository: $spark_source"

spark_roles="$(
  ssh -o BatchMode=yes "$spark_host" bash -s -- \
    "$SPARK_EXPERT_DOCKER_INFERENCE" <<'REMOTE'
set -euo pipefail
docker image inspect \
  -f '{{index .Config.Labels "io.ds41rt.v41.spark_tp_roles"}}' "$1"
REMOTE
)"
[[ "$spark_roles" != "<no value>" ]] || spark_roles=
push_require_universal_roles "$spark_roles" "$SPARK_EXPERT_DOCKER_INFERENCE"

echo "Publishing DS41RT containers"
echo "  revision:    $coordinator_revision"
echo "  expert roles: $spark_roles"
echo "  coordinator: $coordinator_repository:$tag"
echo "  spark:       $spark_repository:$tag (from $spark_host)"

docker tag "$COORDINATOR_DOCKER_INFERENCE" "$coordinator_repository:$tag"
ssh -o BatchMode=yes "$spark_host" bash -s -- \
  "$SPARK_EXPERT_DOCKER_INFERENCE" "$spark_repository:$tag" <<'REMOTE'
set -euo pipefail
docker tag "$1" "$2"
REMOTE

docker push "$coordinator_repository:$tag"
ssh -o BatchMode=yes "$spark_host" docker push "$spark_repository:$tag"

docker tag "$COORDINATOR_DOCKER_INFERENCE" "$coordinator_repository:latest"
ssh -o BatchMode=yes "$spark_host" bash -s -- \
  "$SPARK_EXPERT_DOCKER_INFERENCE" "$spark_repository:latest" <<'REMOTE'
set -euo pipefail
docker tag "$1" "$2"
REMOTE

docker push "$coordinator_repository:latest"
ssh -o BatchMode=yes "$spark_host" docker push "$spark_repository:latest"

echo "Published both $tag and latest:"
echo "  docker pull $coordinator_repository:$tag"
echo "  docker pull $spark_repository:$tag"
