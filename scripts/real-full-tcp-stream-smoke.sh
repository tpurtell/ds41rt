#!/usr/bin/env bash
set -euo pipefail

url="${1:-${URL:-http://127.0.0.1:8000}}"
model="${2:-${MODEL:-deepseek-ai/DeepSeek-V4.1-Flash}}"
max_tokens="${3:-${MAX_TOKENS:-1}}"
prompt="${PROMPT:-hi}"
first_event_timeout_s="${DS41RT_REAL_FULL_TCP_STREAM_SMOKE_FIRST_EVENT_TIMEOUT_S:-10}"
total_timeout_s="${DS41RT_REAL_FULL_TCP_STREAM_SMOKE_TOTAL_TIMEOUT_S:-900}"
require_content="${DS41RT_REAL_FULL_TCP_STREAM_SMOKE_REQUIRE_CONTENT:-0}"
require_done="${DS41RT_REAL_FULL_TCP_STREAM_SMOKE_REQUIRE_DONE:-0}"

need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "required command not found: $1" >&2
    exit 2
  fi
}

require_bool_flag() {
  local name="$1"
  local value="$2"
  case "$value" in
    0|1) ;;
    *)
      echo "$name must be 0 or 1" >&2
      exit 2
      ;;
  esac
}

require_positive_int() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[0-9]+$ ]] || [ "$value" -lt 1 ]; then
    echo "$name must be a positive integer" >&2
    exit 2
  fi
}

first_data_line() {
  local path="$1"
  grep -m1 '^data: ' "$path" 2>/dev/null || true
}

content_data_line() {
  local path="$1"
  while IFS= read -r line; do
    case "$line" in
      data:\ \[DONE\]) continue ;;
      data:\ *)
        if printf '%s\n' "${line#data: }" |
          jq -e '(.choices[0].delta.content // "") | length > 0' >/dev/null 2>&1; then
          printf '%s\n' "$line"
          return 0
        fi
        ;;
    esac
  done <"$path"
}

wait_for_pattern() {
  local label="$1"
  local attempts="$2"
  local check_fn="$3"
  local path="$4"
  local observed
  for _ in $(seq 1 "$attempts"); do
    observed="$("$check_fn" "$path" || true)"
    if [ -n "$observed" ]; then
      printf '%s\n' "$observed"
      return 0
    fi
    sleep 0.1
  done
  echo "stream smoke timed out waiting for ${label}" >&2
  return 1
}

wait_for_done() {
  local path="$1"
  if grep -q '^data: \[DONE\]$' "$path" 2>/dev/null; then
    printf '%s\n' 'data: [DONE]'
  fi
}

need curl
need jq
need grep
need mktemp
require_positive_int MAX_TOKENS "$max_tokens"
require_positive_int DS41RT_REAL_FULL_TCP_STREAM_SMOKE_FIRST_EVENT_TIMEOUT_S "$first_event_timeout_s"
require_positive_int DS41RT_REAL_FULL_TCP_STREAM_SMOKE_TOTAL_TIMEOUT_S "$total_timeout_s"
require_bool_flag DS41RT_REAL_FULL_TCP_STREAM_SMOKE_REQUIRE_CONTENT "$require_content"
require_bool_flag DS41RT_REAL_FULL_TCP_STREAM_SMOKE_REQUIRE_DONE "$require_done"

health="$(curl -fsS --connect-timeout 5 --max-time "$first_event_timeout_s" "${url}/health")"
models="$(curl -fsS --connect-timeout 5 --max-time "$first_event_timeout_s" "${url}/v1/models")"
echo "$health" | jq -e '.backend == "real-ds4-full" and .transport == "tcp"' >/dev/null
echo "$models" | jq -e --arg model "$model" '.data | map(.id) | index($model) != null' >/dev/null

payload="$(
  jq -cn \
    --arg model "$model" \
    --arg content "$prompt" \
    --argjson max_tokens "$max_tokens" \
    '{model:$model,stream:true,messages:[{role:"user",content:$content}],max_tokens:$max_tokens}'
)"

stream_tmp="$(mktemp)"
stderr_tmp="$(mktemp)"
status_tmp="$(mktemp)"
cleanup() {
  if [ -n "${curl_pid:-}" ] && kill -0 "$curl_pid" 2>/dev/null; then
    kill "$curl_pid" 2>/dev/null || true
    wait "$curl_pid" 2>/dev/null || true
  fi
  rm -f "$stream_tmp" "$stderr_tmp" "$status_tmp"
}
trap cleanup EXIT

(
  set +e
  curl -fsS -N --no-buffer \
    --connect-timeout 5 \
    --max-time "$total_timeout_s" \
    "${url}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$payload" \
    >"$stream_tmp" 2>"$stderr_tmp"
  printf '%s\n' "$?" >"$status_tmp"
) &
curl_pid="$!"

attempts=$((first_event_timeout_s * 10))
first_event="$(wait_for_pattern "assistant role SSE frame" "$attempts" first_data_line "$stream_tmp" || true)"
if [ -z "$first_event" ]; then
  cat "$stderr_tmp" >&2 || true
  sed -n '1,40p' "$stream_tmp" >&2 || true
  exit 1
fi

first_payload="${first_event#data: }"
if ! printf '%s\n' "$first_payload" | jq -e '.choices[0].delta.role == "assistant"' >/dev/null; then
  echo "first SSE data frame was not an assistant role frame:" >&2
  printf '%s\n' "$first_event" >&2
  exit 1
fi

content_seen=false
done_seen=false
if [ "$require_content" = "1" ]; then
  content_event="$(wait_for_pattern "content SSE frame" "$((total_timeout_s * 10))" content_data_line "$stream_tmp" || true)"
  if [ -z "$content_event" ]; then
    cat "$stderr_tmp" >&2 || true
    sed -n '1,80p' "$stream_tmp" >&2 || true
    exit 1
  fi
  content_seen=true
fi

if [ "$require_done" = "1" ]; then
  done_event="$(wait_for_pattern "[DONE] SSE frame" "$((total_timeout_s * 10))" wait_for_done "$stream_tmp" || true)"
  if [ -z "$done_event" ]; then
    cat "$stderr_tmp" >&2 || true
    sed -n '1,120p' "$stream_tmp" >&2 || true
    exit 1
  fi
  done_seen=true
fi

if [ "$require_content" = "1" ] || [ "$require_done" = "1" ]; then
  wait "$curl_pid" || {
    status="$(cat "$status_tmp" 2>/dev/null || true)"
    echo "curl streaming request failed with status ${status:-unknown}" >&2
    cat "$stderr_tmp" >&2 || true
    exit 1
  }
  curl_pid=""
else
  kill "$curl_pid" 2>/dev/null || true
  wait "$curl_pid" 2>/dev/null || true
  curl_pid=""
fi

echo "real_full_tcp_stream_smoke first_role=true content_seen=${content_seen} done_seen=${done_seen} max_tokens=${max_tokens} model=${model}" >&2
