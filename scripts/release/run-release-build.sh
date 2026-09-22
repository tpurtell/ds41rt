#!/usr/bin/env bash
# Foreground release-build runner for one number release, with evidence capture.
#
# Owned by the release executor. This does NOT publish: it runs the documented
# `./build.sh` pipeline for a vN build config from a frozen source tree and writes
# the command line, environment, source identity and the complete log under an
# evidence directory so the run survives an interrupted session.
#
# Run it from a MANAGED background job (the harness job manager), not from a
# detached `nohup`:
#   ./scripts/release/run-release-build.sh --config ds41rt.build-v11.config \
#     --source /home/tj/.cache/ds41rt/builds/v11-release-clone \
#     --evidence runs/v11-release/build
#
# A single `git worktree` cannot host the coordinator build leg: build.sh bind
# mounts the tree at /source and a linked worktree's relative submodule gitdir
# does not resolve there (v10 RUN-SUMMARY section 1). Use a standalone clone at
# the frozen commit with third_party/sparkinfer, third_party/xgrammar and
# xgrammar/3rdparty/dlpack materialized.
#
# Preconditions refused up front: a dirty source tree, a missing build config, an
# unsafe build root (never /mnt/scratch), or an evidence directory that already
# holds a completed run.
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage: scripts/release/run-release-build.sh --config FILE --source DIR --evidence DIR
                                            [--roles tp2;tp3;tp6] [--allow-dirty]
                                            [--label NAME] [--help]

  --config FILE   release build config (release tag derives from its image pair)
  --source DIR    frozen standalone clone to build from (default: repo root)
  --evidence DIR  evidence directory (required); logs and identity files go here
  --roles LIST    Spark expert roles; default tp2;tp3;tp6 (universal)
  --allow-dirty   build a dirty tree (records the automatic source manifest)
  --label NAME    evidence prefix (default: the release tag)

Build root (root NVMe, never /mnt/scratch):
  DS41RT_RELEASE_BUILD_ROOT         (default ~/.cache/ds41rt/builds/<tag>-build-root)
  DS41RT_RELEASE_REMOTE_BUILD_DIR   (default ~/ds41rt-release-build-<tag>)
  DS41RT_RELEASE_SSH_CONFIG         unset by default so ~/.ssh/config aliases resolve
EOF
}

die() { echo "run-release-build: $*" >&2; exit 2; }

config=""
source_dir="$repo_root"
evidence=""
roles="tp2;tp3;tp6"
allow_dirty=0
label=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)      [[ $# -ge 2 ]] || die "--config requires a file";  config="$2"; shift 2 ;;
    --source)      [[ $# -ge 2 ]] || die "--source requires a dir";   source_dir="$2"; shift 2 ;;
    --evidence)    [[ $# -ge 2 ]] || die "--evidence requires a dir"; evidence="$2"; shift 2 ;;
    --roles)       [[ $# -ge 2 ]] || die "--roles requires a list";   roles="$2"; shift 2 ;;
    --label)       [[ $# -ge 2 ]] || die "--label requires a name";   label="$2"; shift 2 ;;
    --allow-dirty) allow_dirty=1; shift ;;
    -h|--help)     usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "$config" ]] || { usage >&2; exit 2; }
[[ -f "$config" ]] || die "build config not found: $config"
[[ -d "$source_dir" ]] || die "source directory not found: $source_dir"

# Canonicalize before any `cd`: a relative --config or --source is resolved
# against the caller's cwd, not against the build source directory, and the
# build later runs from inside the source tree.
config="$(realpath "$config")" || die "cannot canonicalize config path: $config"
source_dir="$(realpath "$source_dir")" || die "cannot canonicalize source path: $source_dir"
[[ -f "$config" ]] || die "build config not found after canonicalization: $config"
[[ -f "$source_dir/build.sh" ]] || die "source directory is not a DS41RT tree (no build.sh): $source_dir"

# Resolve the release tag exactly as build.sh does.
source "$repo_root/scripts/release-common.sh"
release_load_config "$config"
release_version="${COORDINATOR_DOCKER_INFERENCE##*:}"
[[ -n "$label" ]] || label="$release_version"
[[ -n "$evidence" ]] || { usage >&2; exit 2; }
# Canonicalize before anything else. The build runs from inside the source clone,
# so a relative evidence path would be created and logged there instead of in the
# repository's runs/ tree, and the operator's cwd must not decide where evidence
# lands.
evidence="$(realpath -m "$evidence")" || die "cannot canonicalize evidence path: $evidence"
mkdir -p "$evidence" || die "cannot create evidence directory: $evidence"

# DS41RT_RELEASE_REMOTE_BUILD_DIR is validated by build.sh against
# release_canonical_path: it must be an absolute path. Default it under $HOME on
# the seed Spark rather than a bare name, or build.sh refuses to start.
build_root="${DS41RT_RELEASE_BUILD_ROOT:-$HOME/.cache/ds41rt/builds/${release_version}-build-root}"
remote_dir="${DS41RT_RELEASE_REMOTE_BUILD_DIR:-$HOME/ds41rt-release-build-${release_version}}"
export DS41RT_RELEASE_BUILD_ROOT="$build_root"
export DS41RT_RELEASE_REMOTE_BUILD_DIR="$remote_dir"
export DS41RT_RELEASE_SPARK_TP_ROLES="$roles"

log="$evidence/${label}-build.log"
rc_file="$evidence/${label}-build.rc"

if [[ -s "$rc_file" && "$(cat "$rc_file")" == "0" ]]; then
  die "evidence already holds a completed build ($rc_file); use a fresh evidence directory"
fi

# Filesystem safety: the build root and the source tree must not sit on the
# read-only NTFS scratch drive. build.sh repeats this, but a clear early refusal
# beats a half-started build.
python3 "$repo_root/scripts/assert-build-filesystem.py" "$build_root" "$source_dir" \
  "${CARGO_HOME:-$HOME/.cargo}" "${TMPDIR:-/tmp}" \
  || die "build filesystem preflight failed (never build under /mnt/scratch)"

if [[ -d "$source_dir/.git" || -f "$source_dir/.git" ]]; then
  if [[ "$allow_dirty" != 1 ]]; then
    dirty="$(git -C "$source_dir" status --porcelain=v1 -uall | head -20)"
    [[ -z "$dirty" ]] || {
      echo "dirty source tree in $source_dir:" >&2
      echo "$dirty" >&2
      die "refusing a dirty source tree without --allow-dirty"
    }
  fi
  source_revision="$(git -C "$source_dir" rev-parse HEAD 2>/dev/null || echo unknown)"
  submodule_state="$(git -C "$source_dir" submodule status --recursive 2>/dev/null | tr '\n' '|' || true)"
else
  source_revision="unknown (no git metadata in $source_dir)"
  submodule_state=""
fi

# Frozen-source preflight, host side. The v10 pipeline verified the staged source
# and the dependency locks before building (DEVELOPER.md), and build.sh repeats
# it inside the container; doing it here turns a bad pin into an early refusal
# instead of a failed long build. Skipped only when the source carries no
# submodules to check.
preflight_note=""
if [[ -f "$source_dir/third_party/sparkinfer.lock.json" ]]; then
  source_manifest="$(
    PYTHONPATH="$source_dir/python/reference${PYTHONPATH:+:$PYTHONPATH}" \
      python3 "$source_dir/scripts/verify-sparkinfer-source.py" \
        --source "$source_dir/third_party/sparkinfer" \
        --lock "$source_dir/third_party/sparkinfer.lock.json" 2>&1
  )" || die "source preflight failed (SparkInfer pin): $source_manifest"
  preflight_note="$preflight_note SparkInfer:$source_manifest"
fi
if [[ -f "$source_dir/third_party/xgrammar.lock.json" ]]; then
  xgrammar_out="$(
    PYTHONPATH="$source_dir/python/reference${PYTHONPATH:+:$PYTHONPATH}" \
      python3 "$source_dir/scripts/verify-xgrammar-source.py" \
        --source "$source_dir/third_party/xgrammar" \
        --lock "$source_dir/third_party/xgrammar.lock.json" 2>&1
  )" || die "source preflight failed (XGrammar pin): $xgrammar_out"
  xgrammar_rev="$(sed -n 's/.*revision=\([0-9a-f]\{40\}\).*/\1/p' <<<"$xgrammar_out" | head -1)"
  preflight_note="$preflight_note XGrammar:${xgrammar_rev:-verified}"
fi

{
  echo "===== ${label} release build start $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
  echo "host=$(hostname)"
  echo "repo_root=$repo_root"
  echo "source_dir=$source_dir"
  echo "source_revision=$source_revision"
  echo "submodules=$submodule_state"
  echo "dependency_preflight=$preflight_note"
  echo "config=$config"
  echo "release_version=$release_version"
  echo "DS41RT_RELEASE_SPARK_TP_ROLES=$DS41RT_RELEASE_SPARK_TP_ROLES"
  echo "DS41RT_RELEASE_BUILD_ROOT=$DS41RT_RELEASE_BUILD_ROOT"
  echo "DS41RT_RELEASE_REMOTE_BUILD_DIR=$DS41RT_RELEASE_REMOTE_BUILD_DIR"
  echo "DS41RT_RELEASE_SSH_CONFIG=${DS41RT_RELEASE_SSH_CONFIG-<unset: stock OpenSSH>}"
  echo "----- ./build.sh --config $config -----"
} | tee -a "$log"

cd "$source_dir" || die "cannot enter source directory: $source_dir"
./build.sh --config "$config" 2>&1 | tee -a "$log"
rc=${PIPESTATUS[0]}
echo "$rc" >"$rc_file"
{
  echo "===== ${label} release build end $(date -u +%Y-%m-%dT%H:%M:%SZ) rc=$rc ====="
} | tee -a "$log"

exit "$rc"
