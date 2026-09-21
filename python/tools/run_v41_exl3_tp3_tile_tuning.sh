#!/usr/bin/env bash
# Serialize the authorized EXL3 TP3 tile-tuning matrix on ONE SM121 lane.
#
# This runner launches the COMMITTED, review-accepted harness
# (ds41rt 309e9e817975d5eab51576ffae84303d786cc99b, bench_v41_exl3_tiles.py)
# and enforces the freeze conditions that are deliberately NOT in the harness
# itself:
#   * primary matrix: ranks slice 0/768/1536 x capacity 16/80, one cell at a
#     time (fully serial, lane-locked);
#   * 3 matched runs per cell, family DERIVED (no --tiers anywhere on the
#     primary matrix, so the [3,4] requirement is proven, not declared);
#   * abort before the next run whenever the finished run failed (nonzero
#     exit, correctness/invariant gate, or any recorded throttle-reason
#     bitmask that is not 0x0 — the harness records clocks before/after each
#     timing window; this runner turns that audit data into a hard stop);
#   * ALWAYS writes a manifest — on abort it records status=aborted with the
#     reason, the cell that died, tool hashes, and every partial evidence
#     file, so the offline analyzer still sees commit provenance;
#   * optional legacy paired-width regression ONLY after all primaries.
#
# Evidence layout under $EVIDENCE/v10-exl3-tp3/tile-tuning/<utc>-<host>/:
# per-run JSONs (harness --output), manifest.json, lane.log.
set -euo pipefail

: "${SNAPSHOT:?set SNAPSHOT to the absolute cfd4ca1d... snapshot directory}"
: "${EVIDENCE:?set EVIDENCE to a unique root-NVMe run directory}"
case "$EVIDENCE" in
  /mnt/scratch*|*/mnt/scratch*) echo "REFUSED: EVIDENCE must be a root-NVMe path, not /mnt/scratch" >&2; exit 2;;
esac
[[ "$SNAPSHOT" == */cfd4ca1d1934a8e81dd2d7515598d4ce288e8b88 ]] || {
  echo "REFUSED: SNAPSHOT must be the exact cfd4ca1d... revision directory" >&2; exit 2; }

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
COMMIT=$(git -C "$REPO" rev-parse HEAD)
[[ "$COMMIT" == 309e9e817975d5eab51576ffae84303d786cc99b* ]] ||
  echo "WARNING: HEAD $COMMIT is not the authorized 309e9e8 base" >&2
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$(hostname -s)"
OUT="$EVIDENCE/v10-exl3-tp3/tile-tuning/$RUN_ID"
mkdir -p "$OUT"
exec 9>"$OUT/.lane.lock"; flock -x 9          # one tuning lane at a time
PY="$REPO/.venv/bin/python"; TO="$REPO/python/tools/bench_v41_exl3_tiles.py"
CKPT="wrldsuksgo2mars/DeepSeek-V4.1-EXL3-K3.25-v1"
echo "lane start $RUN_ID commit=$COMMIT host=$(hostname) out=$OUT" | tee "$OUT/lane.log"

throttle_clean() {  # every recorded clock sample in one run JSON must show 0x0
  "$PY" - "$1" <<'THROTTLE_EOF'
import json, sys
record = json.load(open(sys.argv[1]))
ok = True
for result in record.get('results', []):
    for key in ('clocks_before', 'clocks_after'):
        line = result.get(key)
        if not line:
            ok = False; continue
        try:
            throttled = int(line.split(',')[-1].strip(), 16) != 0
        except ValueError:
            throttled = True          # unparseable = unprovenanced = not clean
        ok = ok and not throttled
print('clean' if ok else 'THROTTLED-or-unprovenanced')
sys.exit(0 if ok else 1)
THROTTLE_EOF
}

write_manifest() {  # write_manifest <status> [reason] [cell]
  "$PY" - "$OUT" "$COMMIT" "$SNAPSHOT" "$REPO" "${1:-complete}" "${2:-}" "${3:-}" <<'MANIFEST_EOF'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
out, commit, snap, repo, status, reason, cell = Path(sys.argv[1]), sys.argv[2], sys.argv[3], \
    Path(sys.argv[4]), sys.argv[5], sys.argv[6], sys.argv[7]
tools = {}
for rel in ('python/tools/bench_v41_exl3_tiles.py', 'python/tools/v41_exl3_family.py',
            'python/tools/run_v41_exl3_tp3_tile_tuning.sh',
            'python/tools/analyze_v41_exl3_tp3_tile_tuning.py'):
    path = repo / rel
    if path.is_file():
        tools[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
files = sorted(p for p in out.iterdir() if p.suffix == '.json' and p.name != 'manifest.json')
manifest = dict(schema='ds41rt.tp3-tile-tuning-lane-v1',
                generated_at=datetime.now(timezone.utc).isoformat(),
                status=status, abort_reason=reason or None, aborted_cell=cell or None,
                ds41rt_commit=commit, tools_sha256=tools, snapshot=snap,
                checkpoint='wrldsuksgo2mars/DeepSeek-V4.1-EXL3-K3.25-v1',
                tier_mode='derived-no-override', repetitions_per_cell=3, serialized=True,
                files=[dict(name=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                       for p in files])
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(f'manifest ({status}) written:', out / 'manifest.json', f'({len(files)} run files)')
MANIFEST_EOF
}

run_cell() {  # run_cell <label> <width> <start> <capacity>
  local label=$1 width=$2 start=$3 cap=$4
  for rep in 1 2 3; do
    local file="$OUT/${label}_rep${rep}.json"
    echo "=== $label rep$rep $(date -u +%FT%TZ)" | tee -a "$OUT/lane.log"
    if ! "$PY" "$TO" --snapshot "$SNAPSHOT" --intermediate "$width" \
      --slice-start "$start" --capacity "$cap" --checkpoint "$CKPT" \
      --output "$file" 2>&1 | tee -a "$OUT/lane.log"; then
      write_manifest aborted "harness nonzero exit (correctness gate, invariant, or crash)" "$label rep$rep"
      echo "ABORT: harness failed in $file — lane stopped" | tee -a "$OUT/lane.log"; exit 1
    fi
    if ! throttle_clean "$file" | tee -a "$OUT/lane.log"; then
      write_manifest aborted "throttle provenance failed (nonzero clock-event mask or unprovenanced bracket)" "$label rep$rep"
      echo "ABORT: throttle provenance failed in $file — lane stopped" | tee -a "$OUT/lane.log"; exit 3
    fi
  done
}

# --- primary matrix: disjoint TP3 rank slices, family derived, no override ---
for cap in 16 80; do
  for start in 0 768 1536; do
    run_cell "tp3-w768-s${start}-c${cap}" 768 "$start" "$cap"
  done
done

# --- optional legacy paired-width regression, ONLY after the primaries ---
if [[ "${1:-}" == --with-legacy ]]; then
  for cap in 16 80; do
    run_cell "legacy-w512-s0-c${cap}"  512 0   "$cap"
    run_cell "legacy-w640-s640-c${cap}" 640 640 "$cap"
  done
fi

write_manifest complete
echo "lane complete $RUN_ID" | tee -a "$OUT/lane.log"
