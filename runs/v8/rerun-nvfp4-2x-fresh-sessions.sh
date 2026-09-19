#!/usr/bin/env bash
# Re-run each failed NVFP4 2x phase in its OWN fresh serving session.
#
# Why: the C16 sweeps and the 16-way evaluation failed with IncompleteStreamError
# (every stream delivered a first chunk then never finished) when they followed a
# long, heavy session, while an isolated C16 code run on a fresh session passed
# (680 tok/s, no failures). That points at state accumulated over a long session
# rather than a hard C16 limit, so each phase gets a clean server. The failure is
# recorded in the report either way rather than being papered over.
set -uo pipefail

cd /home/tj/Developer/ds41rt
SERVE=runs/v8/serve.sh
P=/home/tj/.cache/ds41rt-v8-package/performance
E=/home/tj/.cache/ds41rt-v8-package/tool-eval
TOK=/home/tj/.cache/huggingface/hub/models--nvidia--DeepSeek-V4.1-Flash-NVFP4/snapshots/3431dde3247c13b5957f682b1e3c6fcae2566079/tokenizer.json
PY=./.venv/bin/python

fresh() {  # config
  bash "$SERVE" "$1" stop >/dev/null 2>&1
  sleep 5
  bash "$SERVE" "$1" start > "/tmp/v8-fresh-$1.log" 2>&1 || {
    echo "FAILED TO SERVE $1"; tail -4 "/tmp/v8-fresh-$1.log"; return 1; }
}

for case in code topic; do
  fresh nvfp4-2x || continue
  rm -f "$P/dual-nvfp4-concurrency-$case.json"
  $PY scripts/bench-ds41-concurrent-api.py --base-url http://127.0.0.1:8000 \
    --output "$P/dual-nvfp4-concurrency-$case.json" --concurrency 1 2 4 8 16 \
    --repeats 3 --label v8-nvfp4-dual --case "$case" > "/tmp/v8-fresh-conc-$case.log" 2>&1
  echo "concurrency $case exit=$?"
done

fresh nvfp4-2x || true
rm -f "$P/dual-nvfp4-mixed.json"
$PY scripts/bench-ds41-adaptive-mixed.py --base-url http://127.0.0.1:8000 --tokenizer "$TOK" \
  --nonce-seed 71001 --output "$P/dual-nvfp4-mixed.json" > /tmp/v8-fresh-mixed.log 2>&1
echo "mixed exit=$?"

fresh nvfp4-2x || true
rm -rf "$E/nvfp4-dual"
$PY scripts/qualify-ds41-tool-eval.py --base-url http://127.0.0.1:8000 \
  --output-dir "$E/nvfp4-dual" --runs 3 --parallel 16 --reference-date 2026-09-19 \
  > /tmp/v8-fresh-eval.log 2>&1
echo "eval exit=$?"

bash "$SERVE" nvfp4-2x stop >/dev/null 2>&1
echo "FRESH RERUN DONE"
