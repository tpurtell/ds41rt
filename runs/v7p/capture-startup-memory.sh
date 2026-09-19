#!/usr/bin/env bash
# Record startup latency and post-readiness GPU memory for one served
# configuration into the published package's startup-memory.json.
#
# Two latencies are recorded because they are not the same thing:
#   coordinator_seconds - container start to the coordinator's "target API ready"
#   spark_seconds       - container start to the last Spark "worker ready"
#   full_seconds        - the later of the two, i.e. all experts serving
# The coordinator logs API readiness before the Spark experts have finished
# loading their layers, so a coordinator-only figure would understate startup for
# every configuration that uses Sparks. v6's startup section measured readiness
# including orchestration, which corresponds to full_seconds.
#
# Memory is sampled once after readiness; later graph capture is excluded, and
# the record says so rather than implying a peak.
set -uo pipefail

CONFIG="${1:?config: nvfp4-1x|nvfp4-2x|exl3-5090|exl3-2x}"
OUT=/home/tj/.cache/ds41rt-v7-published/startup-memory.json
SPARK_NAME=ds41rt-spark-expert-published
case "$CONFIG" in
  nvfp4-1x|nvfp4-2x) SPARK_HOSTS=(ostrich dodo emu kiwi) ;;
  exl3-5090)         SPARK_HOSTS=(ostrich dodo) ;;
  *)                 SPARK_HOSTS=() ;;
esac

started="$(docker inspect ds41rt-v7-published --format '{{.State.StartedAt}}' 2>/dev/null)"
[[ -n "$started" ]] || { echo "no running ds41rt-v7-published container" >&2; exit 1; }

ready="$(docker logs ds41rt-v7-published 2>&1 | sed 's/\x1b\[[0-9;]*m//g' \
  | grep 'target API ready' | tail -1 | awk '{print $1}')"
[[ -n "$ready" ]] || { echo "coordinator never logged API readiness" >&2; exit 1; }

spark_ready=""
for host in "${SPARK_HOSTS[@]}"; do
  stamp="$(ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" \
    "docker logs $SPARK_NAME 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep 'worker ready' | tail -1 | awk '{print \$1}'" 2>/dev/null)"
  [[ -n "$stamp" ]] && spark_ready+="$stamp "
done

memory="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | paste -sd';' -)"
python3 - "$CONFIG" "$started" "$ready" "$memory" "$OUT" "$spark_ready" <<'PY'
import datetime, json, sys
from pathlib import Path

config, started, ready, memory, out, spark_ready = sys.argv[1:7]

def parse(value):
    return datetime.datetime.fromisoformat(value.replace('Z', '+00:00'))

def seconds(value):
    return (parse(value) - parse(started)).total_seconds()

gpus = []
for entry in memory.split(';'):
    if entry.strip():
        index, used = entry.split(',')
        gpus.append({'index': int(index), 'used_mib': int(used)})

coordinator = seconds(ready)
stamps = [seconds(s) for s in spark_ready.split() if s.strip()]
record = {
    'coordinator_seconds': round(coordinator, 3),
    'gpu_memory_used_mib': gpus,
    'scope': 'container start to readiness; memory sampled once after readiness, '
             'excludes later graph capture',
}
if stamps:
    record['spark_seconds'] = round(max(stamps), 3)
    record['full_seconds'] = round(max(coordinator, max(stamps)), 3)
else:
    record['full_seconds'] = round(coordinator, 3)

path = Path(out)
store = json.loads(path.read_text()) if path.is_file() else {}
store[config] = record
path.write_text(json.dumps(store, indent=2, sort_keys=True) + '\n')
print(f"{config}: coordinator {coordinator:.1f}s, full {record['full_seconds']:.1f}s, "
      + ', '.join(f"gpu{g['index']}={g['used_mib']}MiB" for g in gpus))
PY
