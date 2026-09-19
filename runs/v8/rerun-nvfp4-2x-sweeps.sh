#!/usr/bin/env bash
# Re-run the three NVFP4 2x phases that died with IncompleteStreamError during
# the main battery (concurrency code, concurrency topic, mixed traffic). The
# phases that ran after them - the three-run evaluation and target-only decode -
# completed normally, so this was a transient outage rather than a bad build.
# The failed records are deleted first: they hold an error and no summaries, and
# leaving them would let a reader mistake an empty table for a measured one.
set -uo pipefail

cd /home/tj/Developer/ds41rt
SERVE=runs/v8/serve.sh
P=/home/tj/.cache/ds41rt-v8-package/performance
PY=./.venv/bin/python

bash "$SERVE" nvfp4-2x start > /tmp/v8-rerun-2x-serve.log 2>&1 || {
  echo "FAILED TO SERVE"; tail -5 /tmp/v8-rerun-2x-serve.log; exit 1; }

for case in code topic; do
  rm -f "$P/dual-nvfp4-concurrency-$case.json"
  $PY scripts/bench-ds41-concurrent-api.py --base-url http://127.0.0.1:8000 \
    --output "$P/dual-nvfp4-concurrency-$case.json" --concurrency 1 2 4 8 16 \
    --repeats 3 --label v8-nvfp4-dual --case "$case" > "/tmp/v8-rerun-2x-conc-$case.log" 2>&1
  echo "concurrency $case exit=$?"
done

rm -f "$P/dual-nvfp4-mixed.json"
$PY scripts/bench-ds41-adaptive-mixed.py --base-url http://127.0.0.1:8000 \
  --tokenizer /home/tj/.cache/huggingface/hub/models--nvidia--DeepSeek-V4.1-Flash-NVFP4/snapshots/3431dde3247c13b5957f682b1e3c6fcae2566079/tokenizer.json \
  --nonce-seed 71001 --output "$P/dual-nvfp4-mixed.json" > /tmp/v8-rerun-2x-mixed.log 2>&1
echo "mixed exit=$?"

bash "$SERVE" nvfp4-2x stop >/dev/null 2>&1
echo "RERUN DONE"
