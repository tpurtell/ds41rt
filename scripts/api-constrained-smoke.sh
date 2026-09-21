#!/usr/bin/env bash
# Constrained-output / tool-stream serving smoke.
#
# The generation requests set `thinking: {type: "disabled"}` explicitly. Without
# it the server defaults to high-effort thinking, which spends the whole 96-token
# budget on reasoning and returns finish_reason=length with empty content, so the
# exact-output assertions below fail even though constrained decoding is working.
# That behaviour was observed on the published v8 images for both the 1x and 2x
# official layouts (retained as a negative result in the v9 baseline evidence,
# ~/.cache/ds41rt-v9-baseline/raw/CONTAMINATED/ and clean-*/api-constrained-smoke.txt).
# The checks themselves are unchanged: exact structured output, exact streamed
# tool arguments and the strict-schema 400 rejection are all still enforced.
set -euo pipefail

url="${1:-${URL:-http://127.0.0.1:8000}}"
model="${2:-${MODEL:-deepseek-ai/DeepSeek-V4.1-Flash}}"

need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "required command not found: $1" >&2
    exit 2
  }
}

need curl
need jq
need grep
need mktemp
need sed

smoke_tmp="$(mktemp -d)"
cleanup() {
  rm -rf -- "$smoke_tmp"
}
trap cleanup EXIT

schema_payload="$(
  jq -cn --arg model "$model" '{
    model: $model,
    thinking: {type: "disabled"},
    messages: [{role: "user", content: "Return Taipei, 27, and sunny."}],
    temperature: 0,
    max_tokens: 96,
    response_format: {
      type: "json_schema",
      json_schema: {
        name: "weather",
        strict: true,
        schema: {
          type: "object",
          properties: {
            city: {type: "string", enum: ["Taipei"]},
            temperature: {type: "integer", enum: [27]},
            condition: {type: "string", enum: ["sunny"]}
          },
          required: ["city", "temperature", "condition"],
          additionalProperties: false
        }
      }
    }
  }'
)"
curl -fsS --max-time 300 "$url/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "$schema_payload" >"$smoke_tmp/schema.json"
jq -e '
  .choices[0].finish_reason == "stop" and
  (.choices[0].message.content | fromjson) == {
    city: "Taipei", temperature: 27, condition: "sunny"
  }
' "$smoke_tmp/schema.json" >/dev/null

combined_payload="$(
  jq -cn --arg model "$model" '{
    model: $model,
    thinking: {type: "disabled"},
    messages: [
      {role: "user", content: "Look up the Taipei weather, then return the requested JSON."},
      {
        role: "assistant",
        content: null,
        tool_calls: [{
          id: "call_weather",
          type: "function",
          function: {
            name: "lookup_weather",
            arguments: "{\"city\":\"Taipei\",\"units\":\"metric\"}"
          }
        }]
      },
      {
        role: "tool",
        tool_call_id: "call_weather",
        name: "lookup_weather",
        content: "{\"city\":\"Taipei\",\"temperature\":27,\"condition\":\"sunny\"}"
      }
    ],
    temperature: 0,
    max_tokens: 96,
    response_format: {
      type: "json_schema",
      json_schema: {
        name: "weather",
        strict: true,
        schema: {
          type: "object",
          properties: {
            city: {type: "string", enum: ["Taipei"]},
            temperature: {type: "integer", enum: [27]},
            condition: {type: "string", enum: ["sunny"]}
          },
          required: ["city", "temperature", "condition"],
          additionalProperties: false
        }
      }
    },
    tools: [{
      type: "function",
      function: {
        name: "lookup_weather",
        description: "Look up weather",
        strict: true,
        parameters: {
          type: "object",
          properties: {
            city: {type: "string"},
            units: {type: "string", enum: ["metric", "imperial"]}
          },
          required: ["city", "units"],
          additionalProperties: false
        }
      }
    }],
    tool_choice: "auto",
    parallel_tool_calls: false
  }'
)"
curl -fsS --max-time 300 "$url/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "$combined_payload" >"$smoke_tmp/combined.json"
jq -e '
  .choices[0].finish_reason == "stop" and
  (.choices[0].message.content | fromjson) == {
    city: "Taipei", temperature: 27, condition: "sunny"
  }
' "$smoke_tmp/combined.json" >/dev/null

tool_payload="$(
  jq -cn --arg model "$model" '{
    model: $model,
    thinking: {type: "disabled"},
    messages: [{role: "user", content: "Call lookup_weather for Taipei in metric units."}],
    temperature: 0,
    max_tokens: 96,
    stream: true,
    response_format: {type: "text"},
    tools: [{
      type: "function",
      function: {
        name: "lookup_weather",
        description: "Look up weather",
        strict: true,
        parameters: {
          type: "object",
          properties: {
            city: {type: "string"},
            units: {type: "string", enum: ["metric", "imperial"]}
          },
          required: ["city", "units"],
          additionalProperties: false
        }
      }
    }],
    tool_choice: {type: "function", function: {name: "lookup_weather"}},
    parallel_tool_calls: false
  }'
)"
curl -fsS -N --no-buffer --max-time 300 "$url/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "$tool_payload" >"$smoke_tmp/tool.sse"
grep -q '^data: \[DONE\]$' "$smoke_tmp/tool.sse"
sed -n 's/^data: //p' "$smoke_tmp/tool.sse" |
  grep -v '^\[DONE\]$' |
  jq -s -e '
    ([.[] | .choices[0].delta.tool_calls[0].function.arguments? // empty]
      | join("") | fromjson) as $arguments |
    $arguments == {city: "Taipei", units: "metric"} and
    (.[-1].choices[0].finish_reason == "tool_calls")
  ' >/dev/null

invalid_payload="$(
  jq -cn --arg model "$model" '{
    model: $model,
    messages: [{role: "user", content: "Return x."}],
    response_format: {
      type: "json_schema",
      json_schema: {
        name: "invalid_strict_schema",
        strict: true,
        schema: {
          type: "object",
          properties: {x: {type: "string"}},
          additionalProperties: false
        }
      }
    }
  }'
)"
invalid_status="$(
  curl -sS --max-time 30 -o "$smoke_tmp/invalid.json" -w '%{http_code}' \
    "$url/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$invalid_payload"
)"
[[ "$invalid_status" == 400 ]]
jq -e '
  .error.type == "invalid_request_error" and
  .error.param == "response_format.json_schema.schema"
' "$smoke_tmp/invalid.json" >/dev/null

echo "api_constrained_smoke json_schema=true combined_response_tools=true strict_tool_stream=true validation=true model=$model"
