#!/usr/bin/env bash
set -euo pipefail

FILE=${1:?Usage: ask-file.sh FILE SUFFIX [MAX_TOKENS]}
SUFFIX=${2:?Usage: ask-file.sh FILE SUFFIX [MAX_TOKENS]}
# Keep the default bounded: this helper is commonly used for quick realistic
# latency checks, and an accidental 4K generation changes the decode context
# regime during the same request. Pass an explicit larger value for long-output
# tests.
MAX_TOKENS=${3:-256}
MODEL=${MODEL:-deepseek-ai/DeepSeek-V4.1-Flash}

[[ "$MAX_TOKENS" =~ ^[1-9][0-9]*$ ]] || {
  echo "MAX_TOKENS must be a positive integer" >&2
  exit 2
}

{
  cat "$FILE"
  printf '\n\n\n%s' "$SUFFIX"
} | jq -Rs \
  --arg model "$MODEL" \
  --argjson max_tokens "$MAX_TOKENS" \
  '{
    model: $model,
    messages: [{role: "user", content: .}],
    stream: true,
    max_tokens: $max_tokens
  }' |
curl -sN http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @- |
while IFS= read -r line; do
  [[ $line == data:\ * ]] || continue
  json=${line#data: }
  [[ $json == "[DONE]" ]] && continue

  jq -j '.choices[0].delta.content // empty' <<<"$json"

  if jq -e 'has("metrics")' >/dev/null <<<"$json"; then
    printf '\n\n--- performance ---\n'
    jq '.metrics' <<<"$json"
  fi
done
