#!/usr/bin/env bash
# Re-run the published-image decode campaigns with the nonce seed each recorded
# v7 campaign used, so the published numbers are directly comparable to the
# recorded ones. A first pass used one shared seed for every config, which made
# the EXL3 2x comparison look like a 7.5% regression until the seed was matched.
set -uo pipefail

cd /home/tj/Developer/ds41rt
P=/home/tj/.cache/ds41rt-v7-published/performance
SERVE=runs/v7p/serve-published.sh
NVFP4_TOK=/home/tj/.cache/huggingface/hub/models--nvidia--DeepSeek-V4.1-Flash-NVFP4/snapshots/3431dde3247c13b5957f682b1e3c6fcae2566079/tokenizer.json
EXL3_TOK=/home/tj/.cache/huggingface/hub/models--diffbot--DeepSeek-V4.1-Flash-EXL3-2.0bpw-2x-RTX-PRO-6000/snapshots/28b7ab71ba2eb15569b08b91a8ea07df8eda8a75/tokenizer.json

run() {  # config tokenizer seed previous_config output
  local config="$1" tok="$2" seed="$3" previous="$4" output="$5"
  echo "=== $config (seed $seed) ==="
  [[ -n "$previous" ]] && bash "$SERVE" "$previous" stop >/dev/null 2>&1
  if ! bash "$SERVE" "$config" start > "/tmp/seed-$config-start.log" 2>&1; then
    echo "$config FAILED TO SERVE"; tail -5 "/tmp/seed-$config-start.log"; return 1
  fi
  ./.venv/bin/python scripts/bench-ds41-release-decode.py \
    --base-url http://127.0.0.1:8000 --tokenizer "$tok" \
    --label "v7-published-$config-seed$seed" --repeats 3 --nonce-seed "$seed" \
    --include-counting --output "$output" > "/tmp/seed-$config-decode.log" 2>&1
  echo "$config decode exit=$? (1 means a sample failed a check; the record is still written)"
}

run nvfp4-1x "$NVFP4_TOK" 73001 exl3-2x   "$P/single-nvfp4-dspark-seed73001.json"
run nvfp4-2x "$NVFP4_TOK" 71001 nvfp4-1x  "$P/dual-nvfp4-dspark-seed71001.json"
run exl3-5090 "$EXL3_TOK" 77001 nvfp4-2x  "$P/single-exl3-dspark-seed77001.json"
echo "ALL DONE"
