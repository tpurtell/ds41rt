#!/usr/bin/env bash
set -euo pipefail

url="${1:-${URL:-http://127.0.0.1:8000}}"
model="${2:-${MODEL:-deepseek-ai/DeepSeek-V4.1-Flash}}"
max_tokens="${3:-${MAX_TOKENS:-1}}"
prompt="${PROMPT:-Use real full.}"
strict="${STRICT:-${DS41RT_REAL_FULL_TCP_SMOKE_STRICT:-0}}"
prompt_repeat_token="${PROMPT_REPEAT_TOKEN:-${DS41RT_REAL_FULL_TCP_SMOKE_PROMPT_REPEAT_TOKEN:-}}"
prompt_repeat_count="${PROMPT_REPEAT_COUNT:-${DS41RT_REAL_FULL_TCP_SMOKE_PROMPT_REPEAT_COUNT:-0}}"
min_prefill_chunks="${MIN_PREFILL_CHUNKS:-${DS41RT_REAL_FULL_TCP_SMOKE_MIN_PREFILL_CHUNKS:-0}}"
require_runtime_summary="${REQUIRE_RUNTIME_SUMMARY:-${DS41RT_REAL_FULL_TCP_SMOKE_REQUIRE_RUNTIME_SUMMARY:-0}}"
require_real_nvfp4="${REQUIRE_REAL_NVFP4:-${DS41RT_REAL_FULL_TCP_SMOKE_REQUIRE_REAL_NVFP4:-0}}"
expert_mode="${DS41RT_PHASE0_SPARK_EXPERT_MODE:-real}"
expected_transport="${EXPECTED_TRANSPORT:-${DS41RT_REAL_FULL_SMOKE_TRANSPORT:-tcp}}"

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

if ! [[ "$max_tokens" =~ ^[0-9]+$ ]] || [ "$max_tokens" -lt 1 ]; then
  echo "MAX_TOKENS must be a positive integer" >&2
  exit 2
fi
if ! [[ "$prompt_repeat_count" =~ ^[0-9]+$ ]]; then
  echo "PROMPT_REPEAT_COUNT must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "$min_prefill_chunks" =~ ^[0-9]+$ ]]; then
  echo "MIN_PREFILL_CHUNKS must be a non-negative integer" >&2
  exit 2
fi
require_bool_flag "REQUIRE_RUNTIME_SUMMARY" "$require_runtime_summary"
require_bool_flag "REQUIRE_REAL_NVFP4" "$require_real_nvfp4"
case "$expert_mode" in
  real|synthetic) ;;
  *)
    echo "DS41RT_PHASE0_SPARK_EXPERT_MODE must be real or synthetic, got: ${expert_mode}" >&2
    exit 2
    ;;
esac
if [ -n "$prompt_repeat_token" ] && [ "$prompt_repeat_count" -gt 0 ]; then
  prompt="$(
    printf '%s\n' "$prompt_repeat_token" | awk -v count="$prompt_repeat_count" '
      { token = $0 }
      END {
        for (i = 0; i < count; i++) {
          if (i > 0) {
            printf " "
          }
          printf "%s", token
        }
      }'
  )"
fi

require_jq_field() {
  local json="$1"
  local filter="$2"
  local message="$3"
  if ! echo "$json" | jq -e "$filter" >/dev/null; then
    echo "$message" >&2
    exit 1
  fi
}

jq_value_or_unknown() {
  local json="$1"
  local filter="$2"
  echo "$json" | jq -r "(${filter}) as \$value | if \$value == null then \"unknown\" else \$value end"
}

diagnostic_field() {
  local content="$1"
  local key="$2"
  printf '%s\n' "$content" | grep -o "${key}=[^ ]*" | head -n1 | cut -d= -f2- || true
}

diagnostic_blocker() {
  local content="$1"
  printf '%s\n' "$content" | sed -n 's/.* blocker=\(.*\) failed=\[\(.*\)\]$/blocker=\1 failed=[\2]/p'
}

require_min_prefill_chunks() {
  local observed="$1"
  if [ "$min_prefill_chunks" -eq 0 ]; then
    return 0
  fi
  if ! [[ "$observed" =~ ^[0-9]+$ ]]; then
    echo "request_prefill_chunks=${observed} is not numeric; required at least ${min_prefill_chunks}" >&2
    exit 1
  fi
  if [ "$observed" -lt "$min_prefill_chunks" ]; then
    echo "request_prefill_chunks=${observed}, required at least ${min_prefill_chunks}" >&2
    exit 1
  fi
}

require_runtime_summary_evidence() {
  local observed="$1"
  if [ "$require_runtime_summary" != "1" ]; then
    return 0
  fi
  if [ "$observed" != "true" ]; then
    echo "request_scheduler_summary_runtime_reported=${observed}; required true" >&2
    exit 1
  fi
}

require_real_nvfp4_evidence() {
  local sparse_passed="$1"
  local all_real="$2"
  local consumed_by_residual="$3"
  if [ "$require_real_nvfp4" != "1" ]; then
    return 0
  fi
  if [ "$sparse_passed" != "true" ]; then
    echo "scheduler_sparse_tcp_dispatch_passed=${sparse_passed}; required true" >&2
    exit 1
  fi
  if [ "$all_real" != "true" ]; then
    echo "scheduler_sparse_tcp_dispatch_all_responses_real_nvfp4=${all_real}; required true" >&2
    exit 1
  fi
  if [ "$consumed_by_residual" != "true" ]; then
    echo "scheduler_sparse_tcp_dispatch_consumed_by_residual=${consumed_by_residual}; required true" >&2
    exit 1
  fi
}

require_full_scheduler_layer_evidence() {
  local request_layerwaves="$1"
  local sparse_layers="$2"
  local sparse_iterations="$3"
  local sparse_batches="$4"
  local numeric_passed="$5"
  local attention_complete="$6"
  if [ "$require_runtime_summary" != "1" ] && [ "$require_real_nvfp4" != "1" ]; then
    return 0
  fi
  if ! [[ "$request_layerwaves" =~ ^[0-9]+$ ]] || [ "$request_layerwaves" -lt 43 ]; then
    echo "request_layerwaves=${request_layerwaves}; required at least 43 for the full DeepSeek V4 Flash layer stack" >&2
    exit 1
  fi
  if ! [[ "$sparse_layers" =~ ^[0-9]+$ ]] || [ "$sparse_layers" -ne 43 ]; then
    echo "scheduler_sparse_tcp_dispatch_sparse_layers=${sparse_layers}; required all 43 sparse layers" >&2
    exit 1
  fi
  if ! [[ "$sparse_iterations" =~ ^[0-9]+$ ]] || [ "$sparse_iterations" -lt 1 ]; then
    echo "scheduler_sparse_tcp_dispatch_iterations_per_sparse_layer=${sparse_iterations}; required at least 1" >&2
    exit 1
  fi
  if ! [[ "$sparse_batches" =~ ^[0-9]+$ ]]; then
    echo "scheduler_sparse_tcp_dispatch_batches=${sparse_batches}; required numeric" >&2
    exit 1
  fi
  if [ "$numeric_passed" != "true" ]; then
    echo "request_numeric_progression_passed=${numeric_passed}; required true" >&2
    exit 1
  fi
  local expected_sparse_batches=$((sparse_layers * sparse_iterations))
  if [ "$sparse_batches" -ne "$expected_sparse_batches" ]; then
    echo "scheduler_sparse_tcp_dispatch_batches=${sparse_batches}; expected ${expected_sparse_batches} from sparse_layers=${sparse_layers} iterations=${sparse_iterations}" >&2
    exit 1
  fi
  if [ "$expert_mode" != "synthetic" ]; then
    if [ "$attention_complete" != "true" ]; then
      echo "scheduler_full_context_device_attention_complete=${attention_complete}; required true" >&2
      exit 1
    fi
  fi
}

health="$(curl -fsS "${url}/health")"
models="$(curl -fsS "${url}/v1/models")"
completion_tmp="$(mktemp)"
cleanup_completion_tmp() {
  rm -f "$completion_tmp"
}
trap cleanup_completion_tmp EXIT
completion_start_ns="$(date +%s%N)"
completion_status="$(
  curl -sS -o "$completion_tmp" -w '%{http_code}' "${url}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(jq -cn \
      --arg model "$model" \
      --arg content "$prompt" \
      --argjson max_tokens "$max_tokens" \
      '{model:$model,messages:[{role:"user",content:$content}],max_tokens:$max_tokens}')"
)"
completion_end_ns="$(date +%s%N)"
completion="$(cat "$completion_tmp")"
completion_elapsed_ms="$(
  awk -v start="$completion_start_ns" -v end="$completion_end_ns" \
    'BEGIN { printf "%.3f", (end - start) / 1000000.0 }'
)"
completion_tokens_for_tps="$(
  echo "$completion" | jq -r '.usage.completion_tokens // 0' 2>/dev/null || echo 0
)"
decode_tokens_per_second="$(
  awk -v tokens="$completion_tokens_for_tps" -v elapsed_ms="$completion_elapsed_ms" \
    'BEGIN { if (elapsed_ms > 0) printf "%.6f", tokens * 1000.0 / elapsed_ms; else printf "0.000000" }'
)"
completion="$(
  echo "$completion" \
    | jq \
        --argjson elapsed_ms "$completion_elapsed_ms" \
        --argjson decode_tps "$decode_tokens_per_second" \
        '.metrics.client_completion_elapsed_ms = $elapsed_ms
         | .metrics.client_decode_tokens_per_second = $decode_tps' \
        2>/dev/null \
    || printf '%s' "$completion"
)"

echo "$health" | jq .
echo "$models" | jq .
echo "$completion" | jq .

if ! [[ "$completion_status" =~ ^[0-9][0-9][0-9]$ ]]; then
  echo "completion request returned non-numeric HTTP status: ${completion_status}" >&2
  exit 1
fi
if [ "$completion_status" -lt 200 ] || [ "$completion_status" -ge 300 ]; then
  message="$(echo "$completion" | jq -r '.error.message // empty' 2>/dev/null || true)"
  code="$(echo "$completion" | jq -r '.error.code // empty' 2>/dev/null || true)"
  echo "completion request returned HTTP ${completion_status} code=${code:-unknown} message=${message:-unknown}" >&2
  exit 1
fi

require_jq_field "$health" '.backend == "real-ds4-full"' \
  "coordinator health backend is not real-ds4-full"
if ! echo "$health" | jq -e --arg expected_transport "$expected_transport" \
  '.transport == $expected_transport' >/dev/null; then
  echo "coordinator health transport is not ${expected_transport}" >&2
  exit 1
fi
if ! echo "$models" | jq -e --arg model "$model" '.data | map(.id) | index($model) != null' >/dev/null; then
  echo "model list does not include ${model}" >&2
  exit 1
fi
require_jq_field "$completion" '.metrics.backend_mode == "real-ds4-full"' \
  "completion did not use real-ds4-full backend"
if ! echo "$completion" | jq -e --arg expected_transport "$expected_transport" \
  '.metrics.transport_backend == $expected_transport' >/dev/null; then
  echo "completion did not report ${expected_transport} transport" >&2
  exit 1
fi

content="$(echo "$completion" | jq -r '.choices[0].message.content // ""')"
if [ -z "$content" ]; then
  echo "completion content was empty" >&2
  exit 1
fi

if [[ "$content" != real\ ds4\ full\ status=* ]]; then
  completion_tokens="$(echo "$completion" | jq -r '.usage.completion_tokens')"
  if ! [[ "$completion_tokens" =~ ^[0-9]+$ ]] || [ "$completion_tokens" -ne "$max_tokens" ]; then
    echo "real-full TCP returned non-diagnostic content but completion_tokens=${completion_tokens}; expected ${max_tokens}" >&2
    exit 1
  fi
  layerwave_decode_rows="$(echo "$completion" | jq -r '.metrics.layerwave_decode_rows // "unknown"')"
  if ! [[ "$layerwave_decode_rows" =~ ^[0-9]+$ ]] || [ "$layerwave_decode_rows" -ne "$max_tokens" ]; then
    echo "real-full TCP returned layerwave_decode_rows=${layerwave_decode_rows}; expected ${max_tokens}" >&2
    exit 1
  fi
  sample_status="$(jq_value_or_unknown "$completion" '.metrics.real_full.scheduler_terminal_lm_head_sample_status')"
  attention_complete="$(jq_value_or_unknown "$completion" '.metrics.real_full.scheduler_full_context_device_attention_complete')"
  sparse_status="$(jq_value_or_unknown "$completion" '.metrics.real_full.scheduler_sparse_tcp_dispatch_status')"
  sparse_passed="$(jq_value_or_unknown "$completion" '.metrics.real_full.scheduler_sparse_tcp_dispatch_passed')"
  all_real="$(jq_value_or_unknown "$completion" '.metrics.real_full.scheduler_sparse_tcp_dispatch_all_responses_real_nvfp4')"
  consumed_by_residual="$(jq_value_or_unknown "$completion" '.metrics.real_full.scheduler_sparse_tcp_dispatch_consumed_by_residual')"
  runtime_summary="$(jq_value_or_unknown "$completion" '.metrics.real_full.request_scheduler_summary_runtime_reported')"
  request_prefill_tokens="$(echo "$completion" | jq -r '.metrics.real_full.request_prefill_tokens // "unknown"')"
  request_prefill_chunks="$(echo "$completion" | jq -r '.metrics.real_full.request_prefill_chunks // "unknown"')"
  request_layerwaves="$(echo "$completion" | jq -r '.metrics.real_full.request_layerwaves // "unknown"')"
  sparse_layers="$(echo "$completion" | jq -r '.metrics.real_full.scheduler_sparse_tcp_dispatch_sparse_layers // "unknown"')"
  sparse_iterations="$(echo "$completion" | jq -r '.metrics.real_full.scheduler_sparse_tcp_dispatch_iterations_per_sparse_layer // "unknown"')"
  sparse_batches="$(echo "$completion" | jq -r '.metrics.real_full.scheduler_sparse_tcp_dispatch_batches // "unknown"')"
  numeric_passed="$(jq_value_or_unknown "$completion" '.metrics.real_full.request_numeric_progression_passed')"
  require_min_prefill_chunks "$request_prefill_chunks"
  require_runtime_summary_evidence "$runtime_summary"
  require_real_nvfp4_evidence "$sparse_passed" "$all_real" "$consumed_by_residual"
  require_full_scheduler_layer_evidence "$request_layerwaves" "$sparse_layers" "$sparse_iterations" "$sparse_batches" "$numeric_passed" "$attention_complete"
  echo "real_full_tcp_smoke result=generated completion_tokens=${completion_tokens} layerwave_decode_rows=${layerwave_decode_rows} completion_elapsed_ms=${completion_elapsed_ms} decode_tokens_per_second=${decode_tokens_per_second} sparse_tcp=${sparse_status} sparse_passed=${sparse_passed} all_real_nvfp4=${all_real} consumed_by_residual=${consumed_by_residual} terminal_sample=${sample_status} attention_complete=${attention_complete} runtime_scheduler_summary=${runtime_summary} request_layerwaves=${request_layerwaves} sparse_layers=${sparse_layers} sparse_iterations=${sparse_iterations} request_prefill_tokens=${request_prefill_tokens} request_prefill_chunks=${request_prefill_chunks} content=${content}" >&2
  exit 0
fi

if echo "$completion" | jq -e '.metrics.real_full != null' >/dev/null; then
  status="$(jq_value_or_unknown "$completion" '.metrics.real_full.status')"
  sparse_status="$(jq_value_or_unknown "$completion" '.metrics.real_full.scheduler_sparse_tcp_dispatch_status')"
  sparse_passed="$(jq_value_or_unknown "$completion" '.metrics.real_full.scheduler_sparse_tcp_dispatch_passed')"
  all_real="$(jq_value_or_unknown "$completion" '.metrics.real_full.scheduler_sparse_tcp_dispatch_all_responses_real_nvfp4')"
  consumed_by_residual="$(jq_value_or_unknown "$completion" '.metrics.real_full.scheduler_sparse_tcp_dispatch_consumed_by_residual')"
  sparse_batches="$(echo "$completion" | jq -r '.metrics.real_full.scheduler_sparse_tcp_dispatch_batches // "unknown"')"
  sparse_rows="$(echo "$completion" | jq -r '.metrics.real_full.scheduler_sparse_tcp_dispatch_global_rows // "unknown"')"
  sample_status="$(jq_value_or_unknown "$completion" '.metrics.real_full.scheduler_terminal_lm_head_sample_status')"
  sample_passed="$(jq_value_or_unknown "$completion" '.metrics.real_full.scheduler_terminal_lm_head_sample_passed')"
  attention_complete="$(jq_value_or_unknown "$completion" '.metrics.real_full.scheduler_full_context_device_attention_complete')"
  runtime_summary="$(jq_value_or_unknown "$completion" '.metrics.real_full.request_scheduler_summary_runtime_reported')"
  request_prefill_tokens="$(echo "$completion" | jq -r '.metrics.real_full.request_prefill_tokens // "unknown"')"
  request_prefill_chunks="$(echo "$completion" | jq -r '.metrics.real_full.request_prefill_chunks // "unknown"')"
  request_layerwaves="$(echo "$completion" | jq -r '.metrics.real_full.request_layerwaves // "unknown"')"
  sparse_layers="$(echo "$completion" | jq -r '.metrics.real_full.scheduler_sparse_tcp_dispatch_sparse_layers // "unknown"')"
  sparse_iterations="$(echo "$completion" | jq -r '.metrics.real_full.scheduler_sparse_tcp_dispatch_iterations_per_sparse_layer // "unknown"')"
  numeric_passed="$(jq_value_or_unknown "$completion" '.metrics.real_full.request_numeric_progression_passed')"
  require_min_prefill_chunks "$request_prefill_chunks"
  require_runtime_summary_evidence "$runtime_summary"
  require_real_nvfp4_evidence "$sparse_passed" "$all_real" "$consumed_by_residual"
  require_full_scheduler_layer_evidence "$request_layerwaves" "$sparse_layers" "$sparse_iterations" "$sparse_batches" "$numeric_passed" "$attention_complete"
  failed="$(echo "$completion" | jq -r '.metrics.real_full.failed_requirements // [] | join(",")')"
  blocker_text="$(echo "$completion" | jq -r '.metrics.real_full.blocker // ""')"
  if [ -z "$blocker_text" ] && [ -z "$failed" ]; then
    echo "real-full structured diagnostic did not include blocker or failed requirements" >&2
    exit 1
  fi
  blocker="blocker=${blocker_text:-none} failed=[${failed}]"
  echo "real_full_tcp_smoke result=blocked status=${status} completion_elapsed_ms=${completion_elapsed_ms} sparse_tcp=${sparse_status} sparse_passed=${sparse_passed} all_real_nvfp4=${all_real} consumed_by_residual=${consumed_by_residual} sparse_batches=${sparse_batches} sparse_rows=${sparse_rows} terminal_sample=${sample_status} terminal_sample_passed=${sample_passed} attention_complete=${attention_complete} runtime_scheduler_summary=${runtime_summary} request_layerwaves=${request_layerwaves} sparse_layers=${sparse_layers} sparse_iterations=${sparse_iterations} request_prefill_tokens=${request_prefill_tokens} request_prefill_chunks=${request_prefill_chunks} ${blocker}" >&2
else
  if [ "$require_runtime_summary" = "1" ] || [ "$require_real_nvfp4" = "1" ]; then
    echo "structured metrics.real_full fields are required when runtime-summary or real-NVFP4 evidence is required" >&2
    exit 1
  fi
  status="$(diagnostic_field "$content" status)"
  sparse_status="$(diagnostic_field "$content" scheduler_sparse_tcp_dispatch_status)"
  sample_status="$(diagnostic_field "$content" scheduler_terminal_lm_head_sample_status)"
  attention_complete="$(diagnostic_field "$content" scheduler_full_context_device_attention_complete)"
  request_prefill_tokens="$(diagnostic_field "$content" request_prefill_tokens)"
  request_prefill_chunks="$(diagnostic_field "$content" request_prefill_chunks)"
  require_min_prefill_chunks "${request_prefill_chunks:-unknown}"
  blocker="$(diagnostic_blocker "$content")"

  if [ -z "$blocker" ]; then
    echo "real-full diagnostic did not include blocker/failed fields" >&2
    exit 1
  fi
  echo "real_full_tcp_smoke result=blocked status=${status:-unknown} completion_elapsed_ms=${completion_elapsed_ms} sparse_tcp=${sparse_status:-unknown} terminal_sample=${sample_status:-unknown} attention_complete=${attention_complete:-unknown} request_prefill_tokens=${request_prefill_tokens:-unknown} request_prefill_chunks=${request_prefill_chunks:-unknown} ${blocker}" >&2
fi

if [ "$strict" = "1" ]; then
  exit 3
fi
