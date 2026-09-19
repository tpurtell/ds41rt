#!/usr/bin/env bash
# Full v8 campaign battery for the two NVFP4 W4A4 configurations, measured on
# the PUBLISHED v8 images:
#   nvfp4-1x = 1 RTX PRO 6000 + 4x Spark
#   nvfp4-2x = 2 RTX PRO 6000 + 4x Spark
#
# Covers the scope the v7 report deferred: decode, prefill, retained-context
# decode with its separate 2K control, counting/code/topic concurrency scaling,
# mixed traffic, target-only decode, startup/memory and a three-run tool-call
# evaluation. Each decode reuses the nonce seed the v7 measurement used
# (73001 for 1x, 71001 for 2x) so the optimisation can be compared like for like.
set -uo pipefail

cd /home/tj/Developer/ds41rt
SERVE=runs/v8/serve.sh
P=/home/tj/.cache/ds41rt-v8-package/performance
E=/home/tj/.cache/ds41rt-v8-package/tool-eval
CTX=/home/tj/.cache/ds41rt-v7-bench/release-context-source.md
TOK=/home/tj/.cache/huggingface/hub/models--nvidia--DeepSeek-V4.1-Flash-NVFP4/snapshots/3431dde3247c13b5957f682b1e3c6fcae2566079/tokenizer.json
PY=./.venv/bin/python

decode() {  # output label seed
  $PY scripts/bench-ds41-release-decode.py --base-url http://127.0.0.1:8000 --tokenizer "$TOK" \
    --label "$2" --repeats 3 --nonce-seed "$3" --include-counting --output "$1"
}

retained() {  # output extra-args...
  local output="$1"; shift
  $PY scripts/bench-ds41-release-retained-decode.py --base-url http://127.0.0.1:8000 \
    --tokenizer "$TOK" --context-file "$CTX" --label "v8-nvfp4" \
    --repeats 3 --context-tag v8-nvfp4 "$@" --output "$output"
}

battery() {  # config stem seed evalname
  local config="$1" stem="$2" seed="$3" evalname="$4"
  echo "########## $config ($stem) ##########"
  bash "$SERVE" "$config" start > "/tmp/v8-$config-serve.log" 2>&1 || {
    echo "$config FAILED TO SERVE"; tail -5 "/tmp/v8-$config-serve.log"; return 1; }
  bash runs/v8/capture-startup-memory.sh "$config" || echo "  startup capture failed"

  echo "--- decode (dSpark, seed $seed) ---"
  decode "$P/$stem-nvfp4-dspark.json" "v8-nvfp4-$stem" "$seed" > "/tmp/v8-$config-decode.log" 2>&1
  echo "decode exit=$?"

  echo "--- prefill matrix ---"
  $PY scripts/bench-ds41-release-prefill-matrix.py --base-url http://127.0.0.1:8000 \
    --tokenizer "$TOK" --context-file "$CTX" --label "v8-nvfp4-$stem" --repeats 3 --warmups 1 \
    --output "$P/$stem-nvfp4-prefill.json" > "/tmp/v8-$config-prefill.log" 2>&1
  echo "prefill exit=$?"

  echo "--- retained-context decode ---"
  retained "$P/$stem-nvfp4-retained.json" > "/tmp/v8-$config-retained.log" 2>&1
  echo "retained exit=$?"

  echo "--- retained-context 2K control ---"
  retained "$P/$stem-nvfp4-retained-2k.json" --context 2048 > "/tmp/v8-$config-retained2k.log" 2>&1
  echo "retained-2k exit=$?"

  echo "--- concurrency sweeps ---"
  for case in counting code topic; do
    $PY scripts/bench-ds41-concurrent-api.py --base-url http://127.0.0.1:8000 \
      --output "$P/$stem-nvfp4-concurrency-$case.json" --concurrency 1 2 4 8 16 \
      --repeats 3 --label "v8-nvfp4-$stem" --case "$case" > "/tmp/v8-$config-conc-$case.log" 2>&1
    echo "concurrency $case exit=$?"
  done

  echo "--- mixed traffic ---"
  $PY scripts/bench-ds41-adaptive-mixed.py --base-url http://127.0.0.1:8000 --tokenizer "$TOK" \
    --nonce-seed "$seed" --output "$P/$stem-nvfp4-mixed.json" > "/tmp/v8-$config-mixed.log" 2>&1
  echo "mixed exit=$?"

  echo "--- tool-call evaluation (3 runs) ---"
  rm -rf "$E/$evalname"
  $PY scripts/qualify-ds41-tool-eval.py --base-url http://127.0.0.1:8000 \
    --output-dir "$E/$evalname" --runs 3 --parallel 16 --reference-date 2026-09-19 \
    > "/tmp/v8-$config-eval.log" 2>&1
  echo "eval exit=$?"

  echo "--- target-only decode (DSPARK=0) ---"
  bash "$SERVE" "$config" stop >/dev/null 2>&1
  DSPARK=0 bash "$SERVE" "$config" start > "/tmp/v8-$config-target-serve.log" 2>&1 || {
    echo "$config target FAILED TO SERVE"; return 1; }
  decode "$P/$stem-nvfp4-target.json" "v8-nvfp4-$stem-target" "$seed" > "/tmp/v8-$config-target.log" 2>&1
  echo "target exit=$?"
  bash "$SERVE" "$config" stop >/dev/null 2>&1
}

battery nvfp4-1x single 73001 nvfp4-single
battery nvfp4-2x dual   71001 nvfp4-dual
echo "NVFP4 V8 BATTERY DONE"
