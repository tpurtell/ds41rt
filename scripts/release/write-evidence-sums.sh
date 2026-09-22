#!/usr/bin/env bash
# Evidence-directory integrity helper for one release.
#
# Writes `<evidence>/SHA256SUMS` covering every regular file in the evidence
# directory except the manifest and the already-checked publication logs' own
# scratch area, then optionally verifies it. Publication evidence for v10 was
# kept under runs/vN-release/publication/SHA256SUMS; this makes that step
# repeatable instead of hand-written.
#
# Usage:
#   scripts/release/write-evidence-sums.sh --evidence DIR        # write manifest
#   scripts/release/write-evidence-sums.sh --evidence DIR --check
#
# The evidence directory is normally repo-ignored (runs/ is in .gitignore), so
# the manifest is a durable local record, not a committed artifact.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/release/write-evidence-sums.sh --evidence DIR [--check] [--quiet]

  --evidence DIR  evidence directory holding the recorded raw files (required)
  --check         verify the existing manifest instead of rewriting it
  --quiet         do not list every hashed file
EOF
}

die() { echo "write-evidence-sums: $*" >&2; exit 2; }

evidence=""
check=0
quiet=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --evidence) [[ $# -ge 2 ]] || die "--evidence requires a dir"; evidence="$2"; shift 2 ;;
    --check)    check=1; shift ;;
    --quiet)    quiet=1; shift ;;
    -h|--help)  usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "$evidence" ]] || { usage >&2; exit 2; }
[[ -d "$evidence" ]] || die "evidence directory not found: $evidence"

manifest="$evidence/SHA256SUMS"
cd "$evidence" || die "cannot enter $evidence"

if ((check)); then
  [[ -f "$manifest" ]] || die "no manifest to check: $manifest"
  sha256sum -c SHA256SUMS
  echo "write-evidence-sums: $manifest verified"
  exit 0
fi

mapfile -t files < <(find . -type f ! -name SHA256SUMS -printf '%P\n' | LC_ALL=C sort)
((${#files[@]} > 0)) || die "no evidence files under $evidence"
if ((quiet)); then
  sha256sum -- "${files[@]}" >SHA256SUMS
else
  sha256sum -- "${files[@]}" | tee SHA256SUMS
fi
echo "write-evidence-sums: wrote $manifest over ${#files[@]} file(s)"
