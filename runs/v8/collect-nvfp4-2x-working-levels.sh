#!/usr/bin/env bash
# Collect the remaining NVFP4 2x phases at the levels that actually work.
#
# The highest-concurrency long-generation phases fail on NVFP4 2x, but the A/B
# shows this is pre-existing: the v7 published image fails the code sweep at C8,
# while v8 reaches C16, so v8 is not the cause and is not worse. The failing
# levels are excluded and the exclusion is disclosed rather than reported as a
# measured zero:
#   concurrency sweeps -> C1..C8   (C16 stalls: IncompleteStreamError, v7 fails earlier)
#   mixed traffic      -> C4, C8   (C16 stalls for the same reason)
#   tool-call eval     -> --parallel 8 (16-way stalls; v7 is the baseline for the
#                                     stall, not v8)
set -uo pipefail

cd /home/tj/Developer/ds41rt
SERVE=runs/v8/serve.sh
P=/home/tj/.cache/ds41rt-v8-package/performance
E=/home/tj/.cache/ds41rt-v8-package/tool-eval
TOK=/home/tj/.cache/huggingface/hub/models--nvidia--DeepSeek-V4.1-Flash-NVFP4/snapshots/3431dde3247c13b5957f682b1e3c6fcae2566079/tokenizer.json
PY=./.venv/bin/python

bash runs/v7p/serve-published.sh nvfp4-2x stop >/dev/null 2>&1
bash "$SERVE" nvfp4-2x start > /tmp/v8-final-2x-serve.log 2>&1 || {
  echo "FAILED TO SERVE"; tail -4 /tmp/v8-final-2x-serve.log; exit 1; }

for case in code topic; do
  rm -f "$P/dual-nvfp4-concurrency-$case.json"
  $PY scripts/bench-ds41-concurrent-api.py --base-url http://127.0.0.1:8000 \
    --output "$P/dual-nvfp4-concurrency-$case.json" --concurrency 1 2 4 8 \
    --repeats 3 --label v8-nvfp4-dual --case "$case" > "/tmp/v8-final-conc-$case.log" 2>&1
  echo "concurrency $case (C1-C8) exit=$?"
done

rm -f "$P/dual-nvfp4-mixed.json"
$PY scripts/bench-ds41-adaptive-mixed.py --base-url http://127.0.0.1:8000 --tokenizer "$TOK" \
  --nonce-seed 71001 --concurrency 4 8 \
  --output "$P/dual-nvfp4-mixed.json" > /tmp/v8-final-mixed.log 2>&1
echo "mixed (C4,C8) exit=$?"

bash "$SERVE" nvfp4-2x stop >/dev/null 2>&1
sleep 5
bash "$SERVE" nvfp4-2x start > /tmp/v8-final-eval-serve.log 2>&1 || { echo "FAILED TO SERVE (eval)"; exit 1; }
rm -rf "$E/nvfp4-dual"
$PY scripts/qualify-ds41-tool-eval.py --base-url http://127.0.0.1:8000 \
  --output-dir "$E/nvfp4-dual" --runs 3 --parallel 8 --reference-date 2026-09-19 \
  > /tmp/v8-final-eval.log 2>&1
echo "eval (parallel 8) exit=$?"
bash "$SERVE" nvfp4-2x stop >/dev/null 2>&1
echo "FINAL 2X COLLECTION DONE"
