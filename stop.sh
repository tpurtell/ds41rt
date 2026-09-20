#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$repo_root/scripts/release-common.sh"

usage() {
  cat <<'EOF'
Usage: ./stop.sh [--config FILE]

Gracefully stops release and WIP DS41RT processes on the coordinator and every
configured Spark rank (SPARK_COUNT; four or six) selected by the configuration.
Release containers are removed; persistent WIP development containers are
stopped but retained. WIP slots, build caches, images, model caches, and
unrelated containers are left untouched.

--config FILE selects an entire alternate
configuration file.
EOF
}

config="$repo_root/ds41rt.config"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      config="${2:?$1 requires a configuration file}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      release_die "unknown stop argument: $1"
      ;;
  esac
done

release_load_config "$config"
release_need docker
release_need ssh
release_need ss
release_need ps

docker info >/dev/null 2>&1 ||
  release_die "local Docker daemon is unavailable"

echo "== stopping DS41RT release services =="
wip_failed=0
release_stop_wip_services || wip_failed=1
release_stop_wip_containers || wip_failed=1
((wip_failed == 0)) ||
  release_die "one or more WIP DS41RT processes or containers could not be stopped"
release_stop_services \
  "$RELEASE_COORDINATOR_CONTAINER_NAME" \
  "$RELEASE_SPARK_CONTAINER_PREFIX" ||
  release_die "one or more remote DS41RT services could not be stopped"
echo "DS41RT release and WIP services are stopped."
