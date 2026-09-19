#!/usr/bin/env bash
# Finish the NVFP4 2x cells that the memory exhaustion blocked, with the KV pool
# cut by ~500MB more (1GiB -> 512MiB) to leave device headroom. Placement stays
# at the documented 20 RTX expert layers (0-19) and first-layer 20.
set -uo pipefail

cd /home/tj/Developer/ds41rt
SERVE=runs/v8/serve.sh
P=/home/tj/.cache/ds41rt-v8-package/performance
E=/home/tj/.cache/ds41rt-v8-package/tool-eval
TOK=/home/tj/.cache/huggingface/hub/models--nvidia--DeepSeek-V4.1-Flash-NVFP4/snapshots/3431dde3247c13b5957f682b1e3c6fcae2566079/tokenizer.json
PY=./.venv/bin/python
export EXTRA_ARGS="--kv-pool-size 512MiB"

bash "$SERVE" nvfp4-2x stop >/dev/null 2>&1
bash "$SERVE" nvfp4-2x start > /tmp/v8-kv512-serve.log 2>&1 || { echo "FAILED TO SERVE"; tail -4 /tmp/v8-kv512-serve.log; exit 1; }
grep -oE "discovered RTX expert layers=[0-9]+|first-layer [0-9]+" /tmp/v8-kv512-serve.log | sort -u | tr '\n' ' '; echo

for case in code topic; do
  rm -f "$P/dual-nvfp4-concurrency-$case.json"
  $PY scripts/bench-ds41-concurrent-api.py --base-url http://127.0.0.1:8000 \
    --output "$P/dual-nvfp4-concurrency-$case.json" --concurrency 1 2 4 8 16 \
    --repeats 3 --label v8-nvfp4-dual --case "$case" > "/tmp/v8-kv512-conc-$case.log" 2>&1
  echo "concurrency $case exit=$?"
done

rm -f "$P/dual-nvfp4-mixed.json"
$PY scripts/bench-ds41-adaptive-mixed.py --base-url http://127.0.0.1:8000 --tokenizer "$TOK" \
  --nonce-seed 71001 --output "$P/dual-nvfp4-mixed.json" > /tmp/v8-kv512-mixed.log 2>&1
echo "mixed exit=$?"

rm -rf "$E/nvfp4-dual"
$PY scripts/qualify-ds41-tool-eval.py --base-url http://127.0.0.1:8000 \
  --output-dir "$E/nvfp4-dual" --runs 3 --parallel 8 --reference-date 2026-09-19 \
  > /tmp/v8-kv512-eval.log 2>&1
echo "eval exit=$?"

bash "$SERVE" nvfp4-2x stop >/dev/null 2>&1
echo "KV512 COLLECTION DONE"
