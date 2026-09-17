#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

addr="${ADDR:-127.0.0.1:8073}"
base_port="${BASE_PORT:-9181}"
catalog="${CATALOG:-.ds41rt-cache/model-artifacts/diagnostic/model_catalog.json}"
model="${MODEL:-deepseek-ai/DeepSeek-V4.1-Flash}"
log_dir="${LOG_DIR:-reports/phase0_artifacts/logs}"
log_prefix="${LOG_PREFIX:-real-slice-tcp-smoke-local}"
bin="${DS41RT_BIN:-$repo_root/rust/target/debug/ds41rt}"

if ! [[ "$base_port" =~ ^[0-9]+$ ]] || [ "$base_port" -lt 1 ] || [ "$base_port" -gt 65532 ]; then
  echo "BASE_PORT must be an integer in 1..65532" >&2
  exit 2
fi

mkdir -p "$log_dir"

owners=(ostrich dodo emu kiwi)
pids=()

cleanup() {
  if ((${#pids[@]})); then
    kill "${pids[@]}" 2>/dev/null || true
    wait "${pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

wait_for_port() {
  local host="$1"
  local port="$2"
  local label="$3"
  local attempts="${4:-160}"
  for _ in $(seq 1 "$attempts"); do
    if (echo >"/dev/tcp/${host}/${port}") >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.1
  done
  echo "${label} did not become ready on ${host}:${port}" >&2
  return 1
}

wait_for_health() {
  local url="$1"
  local pid="$2"
  for _ in $(seq 1 720); do
    if curl -fsS "${url}/health" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "coordinator exited early" >&2
      return 1
    fi
    sleep 0.25
  done
  echo "coordinator did not become ready at ${url}" >&2
  return 1
}

echo "building ds41rt daemon" >&2
cargo build --manifest-path rust/Cargo.toml -p ds41rt-daemon \
  >"${log_dir}/${log_prefix}-build.log" 2>&1

targets=()
for idx in "${!owners[@]}"; do
  owner="${owners[$idx]}"
  port=$((base_port + idx))
  targets+=("${owner}=127.0.0.1:${port}")
  RUST_LOG=ds41rt_transport=info "$bin" expertd \
    --synthetic-weights \
    --transport tcp \
    --listen "127.0.0.1:${port}" \
    >"${log_dir}/${log_prefix}-expertd-${owner}.log" 2>&1 &
  pids+=("$!")
done

for idx in "${!owners[@]}"; do
  owner="${owners[$idx]}"
  port=$((base_port + idx))
  wait_for_port 127.0.0.1 "$port" "expertd ${owner}"
done

expert_hosts="$(IFS=,; echo "${targets[*]}")"
"$bin" coordinator \
  --backend real-ds4-slice \
  --transport tcp \
  --listen "$addr" \
  --model-id "$model" \
  --catalog "$catalog" \
  --expert-hosts "$expert_hosts" \
  >"${log_dir}/${log_prefix}-coordinator.log" 2>&1 &
coordinator_pid="$!"
pids+=("$coordinator_pid")

url="http://${addr}"
if ! wait_for_health "$url" "$coordinator_pid"; then
  sed -n '1,200p' "${log_dir}/${log_prefix}-coordinator.log" >&2 || true
  exit 1
fi

health="$(curl -fsS "${url}/health")"
models="$(curl -fsS "${url}/v1/models")"
completion="$(
  curl -fsS "${url}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(jq -cn \
      --arg model "$model" \
      '{model:$model,messages:[{role:"user",content:"Say hello in five words."}],max_tokens:16}')"
)"

echo "$health" | jq .
echo "$models" | jq .
echo "$completion" | jq .

echo "$health" | jq -e '.backend == "real-ds4-slice" and .transport == "tcp"' >/dev/null
echo "$completion" | jq -e '
  .metrics.backend_mode == "real-ds4-slice"
  and .metrics.transport_backend == "tcp"
  and .metrics.prefill_chunk_count >= 1
  and .metrics.layerwave_prefill_rows == 6
  and .metrics.layerwave_decode_rows == 1
  and .metrics.prefill_ms >= 0
  and .metrics.time_to_first_token_ms >= .metrics.prefill_ms
  and .metrics.prefill_tokens_per_sec > 0
' >/dev/null

content="$(echo "$completion" | jq -r '.choices[0].message.content')"
if [[ "$content" != *"prefill_dispatch transport=tcp"* ]]; then
  echo "completion did not include real prefill TCP dispatch summary" >&2
  exit 1
fi
if [[ "$content" != *"prefill_mlp_input_dispatch transport=tcp"* ]]; then
  echo "completion did not include real prefill MLP-input TCP dispatch summary" >&2
  exit 1
fi
if [[ "$content" != *"prefill_attention_mlp_input_dispatch transport=tcp"* ]]; then
  echo "completion did not include real prefill attention MLP-input TCP dispatch summary" >&2
  exit 1
fi
if [[ "$content" != *"prefill_next_layer_attention_mlp_input_dispatch transport=tcp"* ]]; then
  echo "completion did not include real prefill next-layer attention MLP-input TCP dispatch summary" >&2
  exit 1
fi
if [[ "$content" != *"dispatch_probe transport=tcp"* ]]; then
  echo "completion did not include real router TCP dispatch summary" >&2
  exit 1
fi

for owner in "${owners[@]}"; do
  if ! rg -q 'synthetic expert request received' "${log_dir}/${log_prefix}-expertd-${owner}.log"; then
    echo "expert log for ${owner} did not record a request" >&2
    exit 1
  fi
done
