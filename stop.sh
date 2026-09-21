#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$repo_root/scripts/release-common.sh"

usage() {
  cat <<'EOF'
Usage: ./stop.sh [--config FILE]

Gracefully stops release and WIP DS41RT processes on the coordinator and every
configured Spark rank, regardless of SPARK_COUNT. Cleanup is a superset of the
active ranks: a previous six-rank run can leave release or WIP containers on the
fifth/sixth hosts even when the configuration currently selects a smaller
serving set, and the default configuration names those hosts.
Release containers are removed; persistent WIP development containers are
stopped but retained. WIP slots, build caches, images, model caches, and
unrelated containers are left untouched.

Every configured host and every cleanup phase is attempted even if one host or
phase fails; the script exits nonzero if anything could not be stopped.

--config FILE selects an entire alternate configuration file. Its Spark host
keys define the cleanup scope. Only syntax, known keys, value domains and host
tokens are validated: a file that is incomplete or invalid for launching (for
example six hosts without SPARK_TP/SPARK_EP) still stops every host it names.
EOF
}

config="$repo_root/ds41rt.config"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      [[ $# -ge 2 && -n "$2" ]] || release_die "--config requires a configuration file"
      config="$2"
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

release_load_config "$config" stop
# Widen cleanup past the active ranks before any stop helper runs. The load
# above validates the configuration first (syntax, known keys, value domains),
# so a missing, malformed or unsafe file fails before any container or process
# is touched. Stop mode skips launch-only topology/readiness rules, so a file
# that is incomplete for launching (for example six hosts without SPARK_TP/
# SPARK_EP) still cleans every host it names.
release_select_stop_hosts
release_need docker
release_need ssh
release_need ss
release_need ps

docker info >/dev/null 2>&1 ||
  release_die "local Docker daemon is unavailable"

echo "== stopping DS41RT release services =="
echo "  Spark cleanup hosts: $(release_stop_hosts | paste -sd, -)"
failed=0
release_stop_wip_services || failed=1
release_stop_wip_containers || failed=1
release_stop_services \
  "$RELEASE_COORDINATOR_CONTAINER_NAME" \
  "$RELEASE_SPARK_CONTAINER_PREFIX" || failed=1
((failed == 0)) ||
  release_die "one or more DS41RT services or containers could not be stopped"
echo "DS41RT release and WIP services are stopped."
