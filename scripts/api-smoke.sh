#!/usr/bin/env bash
set -euo pipefail

url="${1:-http://127.0.0.1:8000}"
model="${2:-deepseek-ai/DeepSeek-V4.1-Flash}"

curl -fsS "$url/health" | jq .
curl -fsS "$url/v1/models" | jq .
curl -fsS "$url/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in five words.\"}],\"max_tokens\":16}" | jq .
