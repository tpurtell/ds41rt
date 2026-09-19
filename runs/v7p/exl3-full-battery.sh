#!/usr/bin/env bash
# Full v7 campaign battery for the two EXL3 2 bpw configurations, measured on
# the PUBLISHED images:
#   exl3-2x     = 2x RTX PRO 6000, no Spark      (README: EXL3 2x6000 0-spark)
#   exl3-5090   = 1 RTX PRO 6000 at a 32 GiB budget + 2 TP2 Spark
#                                              (README: EXL3 5090+2-spark)
#
# Decode and prefill are already recorded for both (seed-matched); this driver
# adds the rest of the v5/v6 scope: the three-run tool-call evaluation, the
# retained-context decode with its separate 2K control, counting/code/topic
# concurrency sweeps, mixed traffic, and the target-only decode.
#
# NVFP4 is deliberately excluded: that quant is pending optimization, so its
# battery is deferred rather than measured twice.
set -uo pipefail

cd /home/tj/Developer/ds41rt
SERVE=runs/v7p/serve-published.sh
P=/home/tj/.cache/ds41rt-v7-published/performance
E=/home/tj/.cache/ds41rt-v7-published/tool-eval
CTX=/home/tj/.cache/ds41rt-v7-bench/release-context-source.md
EXL3_TOK=/home/tj/.cache/huggingface/hub/models--diffbot--DeepSeek-V4.1-Flash-EXL3-2.0bpw-2x-RTX-PRO-6000/snapshots/28b7ab71ba2eb15569b08b91a8ea07df8eda8a75/tokenizer.json
PY=./.venv/bin/python

battery() {  # config stem eval-name seed
  local config="$1" stem="$2" evalname="$3" seed="$4"
  echo "########## $config ($stem) ##########"
  bash "$SERVE" "$config" start > "/tmp/battery-$config-serve.log" 2>&1 || {
    echo "$config FAILED TO SERVE"; tail -5 "/tmp/battery-$config-serve.log"; return 1; }

  echo "--- tool-call evaluation (3 runs) ---"
  rm -rf "$E/$evalname"
  $PY scripts/qualify-ds41-tool-eval.py --base-url http://127.0.0.1:8000 \
    --output-dir "$E/$evalname" --runs 3 --parallel 16 --reference-date 2026-09-19 \
    > "/tmp/battery-$config-eval.log" 2>&1
  echo "eval exit=$?"

  echo "--- retained-context decode ---"
  $PY scripts/bench-ds41-release-retained-decode.py --base-url http://127.0.0.1:8000 \
    --tokenizer "$EXL3_TOK" --context-file "$CTX" --label "v7-published-$config" \
    --repeats 3 --context-tag "v7-published-$config" \
    --output "$P/$stem-exl3-retained.json" > "/tmp/battery-$config-retained.log" 2>&1
  echo "retained exit=$?"

  echo "--- retained-context 2K control ---"
  $PY scripts/bench-ds41-release-retained-decode.py --base-url http://127.0.0.1:8000 \
    --tokenizer "$EXL3_TOK" --context-file "$CTX" --label "v7-published-$config" \
    --repeats 3 --context-tag "v7-published-$config" --context 2048 \
    --output "$P/$stem-exl3-retained-2k.json" > "/tmp/battery-$config-retained2k.log" 2>&1
  echo "retained-2k exit=$?"

  echo "--- concurrency sweeps ---"
  for case in counting code topic; do
    $PY scripts/bench-ds41-concurrent-api.py --base-url http://127.0.0.1:8000 \
      --output "$P/$stem-exl3-concurrency-$case.json" --concurrency 1 2 4 8 16 \
      --repeats 3 --label "v7-published-$config" --case "$case" \
      > "/tmp/battery-$config-conc-$case.log" 2>&1
    echo "concurrency $case exit=$?"
  done

  echo "--- mixed traffic ---"
  $PY scripts/bench-ds41-adaptive-mixed.py --base-url http://127.0.0.1:8000 \
    --tokenizer "$EXL3_TOK" --nonce-seed "$seed" \
    --output "$P/$stem-exl3-mixed.json" > "/tmp/battery-$config-mixed.log" 2>&1
  echo "mixed exit=$?"

  echo "--- target-only decode (DSPARK=0) ---"
  bash "$SERVE" "$config" stop >/dev/null 2>&1
  DSPARK=0 bash "$SERVE" "$config" start > "/tmp/battery-$config-target-serve.log" 2>&1 || {
    echo "$config target FAILED TO SERVE"; return 1; }
  $PY scripts/bench-ds41-release-decode.py --base-url http://127.0.0.1:8000 \
    --tokenizer "$EXL3_TOK" --label "v7-published-$config-target" \
    --repeats 3 --nonce-seed "$seed" --include-counting \
    --output "$P/$stem-exl3-target.json" > "/tmp/battery-$config-target.log" 2>&1
  echo "target exit=$?"
  bash "$SERVE" "$config" stop >/dev/null 2>&1
}

battery exl3-2x   dual   exl3-dual   75001
battery exl3-5090 single exl3-compact 77001
echo "EXL3 FULL BATTERY DONE"
