#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [ -n "${LOADPLAN:-}" ]; then
  echo "LOADPLAN is not supported by strict DeepSeek V4 TP4 serving; every route executes on all four Spark ranks" >&2
  exit 2
fi

addr="${ADDR:-127.0.0.1:8000}"
url="http://${addr}"
model_id="${DS41RT_MODEL_ID:-deepseek-ai/DeepSeek-V4.1-Flash}"
model="${MODEL:-${model_id}-full}"
max_tokens="${MAX_TOKENS:-1}"
hosts_csv="${DS41RT_SPARK_HOSTS:-ostrich,dodo,emu,kiwi}"
expert_port="${DS41RT_SPARK_EXPERT_PORT:-9100}"
expert_link_suffix="${DS41RT_REAL_FULL_TCP_EXPERT_LINK_SUFFIX:-${DS41RT_SPARK_EXPERT_LINK_SUFFIX:-}}"
expert_hosts="${DS41RT_REAL_FULL_TCP_EXPERT_HOSTS:-}"
catalog="${CATALOG:-.ds41rt-cache/model-artifacts/diagnostic/model_catalog.json}"
coordinator_transport="${DS41RT_REAL_FULL_SMOKE_TRANSPORT:-tcp}"
if [ -n "${DS41RT_NATIVE_LIB:-}" ]; then
  native_lib="$DS41RT_NATIVE_LIB"
elif [ "$coordinator_transport" = "verbs-host" ]; then
  native_lib="$repo_root/native/build-cuda-rdma/libds41rt_native.so"
else
  native_lib="$repo_root/native/build-cuda/libds41rt_native.so"
fi
log_dir="${LOG_DIR:-reports/phase0_artifacts/logs}"
artifact_dir="${ARTIFACT_DIR:-reports/phase0_artifacts/smoke}"
log_prefix="${LOG_PREFIX:-real-full-tcp-live-smoke}"
build_daemon="${DS41RT_REAL_FULL_TCP_SMOKE_BUILD_DAEMON:-1}"
check_experts="${DS41RT_REAL_FULL_TCP_SMOKE_CHECK_EXPERTS:-1}"
require_cuda="${DS41RT_REAL_FULL_TCP_SMOKE_REQUIRE_CUDA:-1}"
bin="${DS41RT_BIN:-$repo_root/rust/target/debug/ds41rt}"
warmup_experts="${DS41RT_REAL_FULL_TCP_SMOKE_WARMUP_EXPERTS:-1}"
warmup_layer_id="${DS41RT_REAL_FULL_TCP_SMOKE_WARMUP_LAYER_ID:-${DS41RT_PHASE0_TCP_LAYER_ID:-3}}"
warmup_iterations="${DS41RT_REAL_FULL_TCP_SMOKE_WARMUP_ITERATIONS:-1}"
warmup_rows="${DS41RT_REAL_FULL_TCP_SMOKE_WARMUP_ROWS:-1}"
warmup_timeout_ms="${DS41RT_REAL_FULL_TCP_SMOKE_WARMUP_TIMEOUT_MS:-120000}"
warmup_measured_timeout_ms="${DS41RT_REAL_FULL_TCP_SMOKE_WARMUP_MEASURED_TIMEOUT_MS:-5000}"
warmup_roundtrip_rows="${DS41RT_REAL_FULL_TCP_SMOKE_WARMUP_ROUNDTRIP_ROWS:-1}"
warmup_prefill_roundtrip_rows="${DS41RT_REAL_FULL_TCP_SMOKE_WARMUP_PREFILL_ROUNDTRIP_ROWS:-16,256,512}"
warmup_prefill_chain_rows="${DS41RT_REAL_FULL_TCP_SMOKE_WARMUP_PREFILL_CHAIN_ROWS:-16,256,512}"
warmup_hidden_dim="${DS41RT_REAL_FULL_TCP_SMOKE_WARMUP_HIDDEN_DIM:-}"
warmup_routes_per_row="${DS41RT_REAL_FULL_TCP_SMOKE_WARMUP_ROUTES_PER_ROW:-}"
expert_mode="${DS41RT_PHASE0_SPARK_EXPERT_MODE:-real}"
dry_run="${DS41RT_REAL_FULL_TCP_SMOKE_DRY_RUN:-0}"
protocol_v2_timeout_ms="${DS41RT_REAL_FULL_PROTOCOL_V2_TIMEOUT_MS:-120000}"
intermediate_shards="${DS41RT_EXPERT_INTERMEDIATE_SHARDS:-4}"
intermediate_reduction="${DS41RT_EXPERT_INTERMEDIATE_REDUCTION:-}"
if [ -z "$intermediate_reduction" ]; then
  if [ "$coordinator_transport" = "verbs-host" ]; then
    intermediate_reduction=spark-rdma
  else
    intermediate_reduction=coordinator
  fi
fi
intermediate_reduction_dtype="${DS41RT_EXPERT_INTERMEDIATE_REDUCTION_DTYPE:-fp8}"
if [ "${DS41RT_EXPERT_INTERMEDIATE_ROW_SHARDED_REDUCTION+x}" ]; then
  intermediate_row_sharded_reduction="$DS41RT_EXPERT_INTERMEDIATE_ROW_SHARDED_REDUCTION"
else
  case "$intermediate_reduction" in
    spark-rdma|spark-rdma-hybrid) intermediate_row_sharded_reduction=1 ;;
    *) intermediate_row_sharded_reduction=0 ;;
  esac
fi

case "$expert_mode" in
  real)
    default_warmup_expected_executor="protocol-v2-real-nvfp4-checkpoint-executor"
    default_require_real_nvfp4=1
    ;;
  synthetic)
    default_warmup_expected_executor="protocol-v2-synthetic-route-dependent-executor"
    default_require_real_nvfp4=0
    ;;
  *)
    echo "DS41RT_PHASE0_SPARK_EXPERT_MODE must be real or synthetic, got: ${expert_mode}" >&2
    exit 2
    ;;
esac

warmup_expected_executor="${DS41RT_REAL_FULL_TCP_SMOKE_WARMUP_EXPECTED_EXECUTOR:-$default_warmup_expected_executor}"

export DS41RT_REAL_FULL_TCP_SMOKE_REQUIRE_RUNTIME_SUMMARY="${DS41RT_REAL_FULL_TCP_SMOKE_REQUIRE_RUNTIME_SUMMARY:-1}"
export DS41RT_REAL_FULL_TCP_SMOKE_REQUIRE_REAL_NVFP4="${DS41RT_REAL_FULL_TCP_SMOKE_REQUIRE_REAL_NVFP4:-$default_require_real_nvfp4}"

if ! [[ "$max_tokens" =~ ^[0-9]+$ ]] || [ "$max_tokens" -lt 1 ]; then
  echo "MAX_TOKENS must be a positive integer" >&2
  exit 2
fi

if ! [[ "$expert_port" =~ ^[0-9]+$ ]] || [ "$expert_port" -lt 1 ] || [ "$expert_port" -gt 65535 ]; then
  echo "DS41RT_SPARK_EXPERT_PORT must be an integer in 1..65535" >&2
  exit 2
fi
case "$warmup_experts" in
  0|1) ;;
  *)
    echo "DS41RT_REAL_FULL_TCP_SMOKE_WARMUP_EXPERTS must be 0 or 1" >&2
    exit 2
    ;;
esac
case "$dry_run" in
  0|1) ;;
  *)
    echo "DS41RT_REAL_FULL_TCP_SMOKE_DRY_RUN must be 0 or 1" >&2
    exit 2
    ;;
esac
case "$coordinator_transport" in
  tcp|verbs-host) ;;
  *)
    echo "DS41RT_REAL_FULL_SMOKE_TRANSPORT must be tcp or verbs-host, got: ${coordinator_transport}" >&2
    exit 2
    ;;
esac
if [ "$intermediate_shards" != "4" ]; then
  echo "strict DS4 smoke requires DS41RT_EXPERT_INTERMEDIATE_SHARDS=4" >&2
  exit 2
fi
export DS41RT_EXPERT_INTERMEDIATE_SHARDS=4
export DS41RT_EXPERT_INTERMEDIATE_REDUCTION="$intermediate_reduction"
export DS41RT_EXPERT_INTERMEDIATE_REDUCTION_DTYPE="$intermediate_reduction_dtype"
export DS41RT_EXPERT_INTERMEDIATE_ROW_SHARDED_REDUCTION="$intermediate_row_sharded_reduction"
if ! [[ "$warmup_layer_id" =~ ^[0-9]+$ ]]; then
  echo "DS41RT_REAL_FULL_TCP_SMOKE_WARMUP_LAYER_ID must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "$warmup_iterations" =~ ^[0-9]+$ ]] || [ "$warmup_iterations" -lt 1 ]; then
  echo "DS41RT_REAL_FULL_TCP_SMOKE_WARMUP_ITERATIONS must be a positive integer" >&2
  exit 2
fi
if ! [[ "$warmup_rows" =~ ^[0-9]+$ ]] || [ "$warmup_rows" -lt 1 ]; then
  echo "DS41RT_REAL_FULL_TCP_SMOKE_WARMUP_ROWS must be a positive integer" >&2
  exit 2
fi
if ! [[ "$warmup_timeout_ms" =~ ^[0-9]+$ ]] || [ "$warmup_timeout_ms" -lt 1 ]; then
  echo "DS41RT_REAL_FULL_TCP_SMOKE_WARMUP_TIMEOUT_MS must be a positive integer" >&2
  exit 2
fi
if ! [[ "$warmup_measured_timeout_ms" =~ ^[0-9]+$ ]] || [ "$warmup_measured_timeout_ms" -lt 1 ]; then
  echo "DS41RT_REAL_FULL_TCP_SMOKE_WARMUP_MEASURED_TIMEOUT_MS must be a positive integer" >&2
  exit 2
fi
if ! [[ "$protocol_v2_timeout_ms" =~ ^[0-9]+$ ]] || [ "$protocol_v2_timeout_ms" -lt 1 ]; then
  echo "DS41RT_REAL_FULL_PROTOCOL_V2_TIMEOUT_MS must be a positive integer" >&2
  exit 2
fi
export DS41RT_REAL_FULL_PROTOCOL_V2_TIMEOUT_MS="$protocol_v2_timeout_ms"

need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "required command not found: $1" >&2
    exit 2
  fi
}

add_host_python_libdir() {
  python_libs=()
  if [ -n "${DS41RT_PYTHON_LIBDIR:-}" ]; then
    python_libs+=("$DS41RT_PYTHON_LIBDIR")
  fi
  if command -v python3 >/dev/null 2>&1; then
    python_sysconfig_lib="$(python3 - <<'PY' 2>/dev/null || true
import sysconfig
print(sysconfig.get_config_var("LIBDIR") or "")
PY
)"
    if [ -n "$python_sysconfig_lib" ]; then
      python_libs+=("$python_sysconfig_lib")
    fi
  fi
  uv_python_lib="$HOME/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/lib"
  if [ -d "$uv_python_lib" ]; then
    python_libs+=("$uv_python_lib")
  fi
  if [ "${#python_libs[@]}" -gt 0 ]; then
    python_path="$(IFS=:; printf '%s' "${python_libs[*]}")"
    export LD_LIBRARY_PATH="$python_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
}

configure_pinned_sparkinfer() {
  local configured_python="${DS41RT_PYTHON:-${PYO3_PYTHON:-}}"
  if [ -z "$configured_python" ] && [ -x "$repo_root/.venv/bin/python" ]; then
    configured_python="$repo_root/.venv/bin/python"
  fi
  if [ -z "$configured_python" ]; then
    configured_python="$(command -v python3 || true)"
  fi
  if [ -z "$configured_python" ] || [ ! -x "$configured_python" ]; then
    echo "Python interpreter not found; set DS41RT_PYTHON" >&2
    exit 2
  fi
  configured_python="$(realpath -s "$configured_python")"
  export DS41RT_PYTHON="$configured_python"
  export PYO3_PYTHON="${PYO3_PYTHON:-$configured_python}"

  local sparkinfer_source="$repo_root/third_party/sparkinfer"
  local sparkinfer_lock="$repo_root/third_party/sparkinfer.lock.json"
  export PYTHONPATH="$sparkinfer_source${PYTHONPATH:+:$PYTHONPATH}"
  "$configured_python" "$repo_root/scripts/verify-sparkinfer-source.py" \
    --source "$sparkinfer_source" \
    --lock "$sparkinfer_lock" \
    --assert-import-source
}

need curl
need jq
need timeout
need xargs
if [ "$build_daemon" = "1" ]; then
  need cargo
fi

test -f "$catalog" || {
  echo "catalog not found: $catalog" >&2
  exit 2
}
warmup_hidden_dim="${warmup_hidden_dim:-$(jq -er '.facts.hidden_size' "$catalog")}"
warmup_routes_per_row="${warmup_routes_per_row:-$(jq -er '.facts.top_k' "$catalog")}"
if ! [[ "$warmup_hidden_dim" =~ ^[0-9]+$ ]] || [ "$warmup_hidden_dim" -lt 1 ]; then
  echo "DS41RT_REAL_FULL_TCP_SMOKE_WARMUP_HIDDEN_DIM must be a positive integer" >&2
  exit 2
fi
if ! [[ "$warmup_routes_per_row" =~ ^[0-9]+$ ]] || [ "$warmup_routes_per_row" -lt 1 ]; then
  echo "DS41RT_REAL_FULL_TCP_SMOKE_WARMUP_ROUTES_PER_ROW must be a positive integer" >&2
  exit 2
fi

if [ "$require_cuda" = "1" ]; then
  test -f "$native_lib" || {
    cat >&2 <<EOF
CUDA native library not found: $native_lib
Build it with:
  cmake -S native -B native/build-cuda -G Ninja -DDS41RT_ENABLE_CUDA=ON -DDS41RT_ENABLE_RDMA=OFF -DDS41RT_ENABLE_NCCL=ON -DDS41RT_CUDA_ARCHITECTURES=120
  cmake --build native/build-cuda
For verbs-host, build the RDMA-enabled native library:
  cmake -S native -B native/build-cuda-rdma -G Ninja -DDS41RT_ENABLE_CUDA=ON -DDS41RT_ENABLE_RDMA=ON -DDS41RT_ENABLE_NCCL=ON -DDS41RT_CUDA_ARCHITECTURES=120
  cmake --build native/build-cuda-rdma
or set DS41RT_REAL_FULL_TCP_SMOKE_REQUIRE_CUDA=0 for a diagnostic-only local check.
EOF
    exit 2
  }
  export DS41RT_REAL_FULL_CUDA_REFERENCE_KERNELS="${DS41RT_REAL_FULL_CUDA_REFERENCE_KERNELS:-1}"
  export DS41RT_NATIVE_LIB="$native_lib"
  export DS41RT_B12X="${DS41RT_B12X:-1}"
fi
add_host_python_libdir

IFS=',' read -r -a hosts <<< "$hosts_csv"
if [ -z "$expert_hosts" ]; then
  targets=()
  for host in "${hosts[@]}"; do
    host="$(echo "$host" | xargs)"
    [ -n "$host" ] || continue
    targets+=("${host}=${host}${expert_link_suffix}:${expert_port}")
  done
  expert_hosts="$(IFS=,; echo "${targets[*]}")"
fi

if [ -z "$expert_hosts" ]; then
  echo "no expert targets configured; set DS41RT_SPARK_HOSTS or DS41RT_REAL_FULL_TCP_EXPERT_HOSTS" >&2
  exit 2
fi

if [ "$dry_run" = "1" ]; then
  printf 'DS41RT_PHASE0_SPARK_EXPERT_MODE=%s\n' "$expert_mode"
  printf 'DS41RT_REAL_FULL_TCP_EXPERT_HOSTS=%s\n' "$expert_hosts"
  printf 'DS41RT_REAL_FULL_TCP_SMOKE_WARMUP_EXPECTED_EXECUTOR=%s\n' "$warmup_expected_executor"
  printf 'DS41RT_REAL_FULL_TCP_SMOKE_REQUIRE_REAL_NVFP4=%s\n' "$DS41RT_REAL_FULL_TCP_SMOKE_REQUIRE_REAL_NVFP4"
  printf 'DS41RT_REAL_FULL_TCP_SMOKE_REQUIRE_RUNTIME_SUMMARY=%s\n' "$DS41RT_REAL_FULL_TCP_SMOKE_REQUIRE_RUNTIME_SUMMARY"
  printf 'DS41RT_REAL_FULL_TCP_SMOKE_WARMUP_EXPERTS=%s\n' "$warmup_experts"
  printf 'DS41RT_REAL_FULL_TCP_SMOKE_WARMUP_HIDDEN_DIM=%s\n' "$warmup_hidden_dim"
  printf 'DS41RT_REAL_FULL_TCP_SMOKE_WARMUP_ROUTES_PER_ROW=%s\n' "$warmup_routes_per_row"
  printf 'DS41RT_REAL_FULL_TCP_SMOKE_BUILD_DAEMON=%s\n' "$build_daemon"
  printf 'DS41RT_REAL_FULL_TCP_SMOKE_REQUIRE_CUDA=%s\n' "$require_cuda"
  printf 'DS41RT_REAL_FULL_SMOKE_TRANSPORT=%s\n' "$coordinator_transport"
  printf 'DS41RT_REAL_FULL_PROTOCOL_V2_TIMEOUT_MS=%s\n' "$DS41RT_REAL_FULL_PROTOCOL_V2_TIMEOUT_MS"
  printf 'DS41RT_NATIVE_LIB=%s\n' "${DS41RT_NATIVE_LIB:-$native_lib}"
  printf 'DS41RT_B12X=%s\n' "${DS41RT_B12X:-}"
  exit 0
fi

configure_pinned_sparkinfer

mkdir -p "$log_dir" "$artifact_dir"

coordinator_pid=""
cleanup() {
  if [ -n "$coordinator_pid" ]; then
    kill "$coordinator_pid" 2>/dev/null || true
    wait "$coordinator_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

target_addr() {
  local entry="$1"
  if [[ "$entry" == *=* ]]; then
    echo "${entry#*=}"
  else
    echo "$entry"
  fi
}

target_owner() {
  local entry="$1"
  if [[ "$entry" == *=* ]]; then
    echo "${entry%%=*}"
    return
  fi
  local addr_part
  addr_part="$(target_addr "$entry")"
  local host_part
  host_part="$(target_host "$addr_part")"
  if [ -n "$expert_link_suffix" ] && [[ "$host_part" == *"$expert_link_suffix" ]]; then
    echo "${host_part%"$expert_link_suffix"}"
  else
    echo "$host_part"
  fi
}

target_host() {
  local addr="$1"
  echo "${addr%:*}"
}

target_port() {
  local addr="$1"
  if [[ "$addr" == *:* ]]; then
    echo "${addr##*:}"
  else
    echo "$expert_port"
  fi
}

wait_for_tcp() {
  local host="$1"
  local port="$2"
  local label="$3"
  local attempts="${4:-30}"
  for _ in $(seq 1 "$attempts"); do
    if timeout 2 bash -c ":</dev/tcp/${host}/${port}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "$label did not become reachable at ${host}:${port}" >&2
  return 1
}

wait_for_health() {
  local attempts="${1:-720}"
  for _ in $(seq 1 "$attempts"); do
    if curl -fsS "${url}/health" >/dev/null 2>&1; then
      return 0
    fi
    if [ -n "$coordinator_pid" ] && ! kill -0 "$coordinator_pid" 2>/dev/null; then
      echo "real-full coordinator exited early" >&2
      return 1
    fi
    sleep 0.5
  done
  echo "real-full coordinator did not become ready at ${url}" >&2
  return 1
}

warmup_expert_ids_for_rank() {
  local count="$1"
  local routed_experts
  routed_experts="$(jq -er '.facts.routed_experts' "$catalog")"
  if [ "$count" -gt "$routed_experts" ]; then
    echo "cannot infer ${count} identical TP4 warmup routes from ${routed_experts} experts" >&2
    exit 2
  fi
  local -a expert_ids=()
  local expert_id
  for ((expert_id = 0; expert_id < count; expert_id++)); do
    expert_ids+=("$expert_id")
  done
  (IFS=,; echo "${expert_ids[*]}")
}

warmup_protocol_v2_experts() {
  if [ "$warmup_experts" != "1" ]; then
    echo "== skipping ProtocolV2 expert warmup ==" >&2
    return
  fi

  local wire_contract="bf16-in/bf16-out"
  local -a wire_args=()
  local -a warmup_pids=()
  local -a warmup_rank_labels=()
  local -a warmup_logs=()

  echo "== warming Spark experts with binary ProtocolV2 precompile frames transport=${coordinator_transport} wire=${wire_contract} ==" >&2
  IFS=',' read -r -a entries <<< "$expert_hosts"
  for entry in "${entries[@]}"; do
    local rank_label
    local addr_part
    local expert_id
    local expert_ids
    local warmup_log
    rank_label="$(target_owner "$entry")"
    addr_part="$(target_addr "$entry")"
    if [[ "$addr_part" != *:* ]]; then
      addr_part="${addr_part}:${expert_port}"
    fi
    # Every strict-TP4 rank receives the same global routes. The endpoint label
    # identifies a TP rank and never assigns ownership of an expert.
    expert_ids="$(warmup_expert_ids_for_rank "$warmup_routes_per_row")"
    expert_id="${expert_ids%%,*}"
    warmup_log="${artifact_dir}/${log_prefix}-warmup-${rank_label}.log"
    echo "protocol_v2_warmup rank_label=${rank_label} addr=${addr_part} layer_id=${warmup_layer_id} hidden_dim=${warmup_hidden_dim} expert_id=${expert_id} expert_ids=${expert_ids} routes_per_row=${warmup_routes_per_row} warmup_iterations=${warmup_iterations} warmup_timeout_ms=${warmup_timeout_ms} measured_timeout_ms=${warmup_measured_timeout_ms}" >&2
    {
      echo "protocol_v2_warmup rank_label=${rank_label} addr=${addr_part} layer_id=${warmup_layer_id} hidden_dim=${warmup_hidden_dim} expert_id=${expert_id} expert_ids=${expert_ids} routes_per_row=${warmup_routes_per_row} warmup_iterations=${warmup_iterations} warmup_timeout_ms=${warmup_timeout_ms} measured_timeout_ms=${warmup_measured_timeout_ms}"
      "$bin" bench-protocol-v2-tcp \
        --addr "$addr_part" \
        --transport "$coordinator_transport" \
        --target "real-full-live-smoke-warmup-${rank_label}" \
        --hops 1 \
        --iterations 1 \
        --large-iterations 1 \
        --warmup-iterations "$warmup_iterations" \
        --warmup-rows "$warmup_rows" \
        --warmup-timeout-ms "$warmup_timeout_ms" \
        --warmup-only \
        --roundtrip-rows "$warmup_roundtrip_rows" \
        --prefill-roundtrip-rows "$warmup_prefill_roundtrip_rows" \
        --prefill-chain-rows "$warmup_prefill_chain_rows" \
        --hidden-dim "$warmup_hidden_dim" \
        --layer-id "$warmup_layer_id" \
        --expert-id "$expert_id" \
        --expert-ids "$expert_ids" \
        --routes-per-row "$warmup_routes_per_row" \
        --expected-executor "$warmup_expected_executor" \
        --require-expected-executor \
        "${wire_args[@]}" \
        --timeout-ms "$warmup_measured_timeout_ms"
    } >"$warmup_log" 2>&1 &
    warmup_pids+=("$!")
    warmup_rank_labels+=("$rank_label")
    warmup_logs+=("$warmup_log")
  done
  local warmup_status=0
  local warmup_index
  for warmup_index in "${!warmup_pids[@]}"; do
    if ! wait "${warmup_pids[$warmup_index]}" \
      || grep -q '^Error:' "${warmup_logs[$warmup_index]}"; then
      cat "${warmup_logs[$warmup_index]}" >&2 || true
      warmup_status=1
    else
      echo "protocol_v2_warmup status=ok rank_label=${warmup_rank_labels[$warmup_index]}" \
        >>"${warmup_logs[$warmup_index]}"
    fi
  done
  return "$warmup_status"
}

if [ "$check_experts" = "1" ]; then
  IFS=',' read -r -a entries <<< "$expert_hosts"
  for entry in "${entries[@]}"; do
    addr_part="$(target_addr "$entry")"
    wait_for_tcp "$(target_host "$addr_part")" "$(target_port "$addr_part")" "expert target ${entry}" 10
  done
fi

if [ "$build_daemon" = "1" ]; then
  echo "== building ds41rt daemon ==" >&2
  cargo build --manifest-path rust/Cargo.toml -p ds41rt-daemon \
    >"${log_dir}/${log_prefix}-build.log" 2>&1
fi

warmup_protocol_v2_experts

echo "== starting real-full ${coordinator_transport} coordinator at ${addr} ==" >&2
echo "expert_hosts=${expert_hosts}" >&2
"$bin" coordinator \
  --backend real-ds4-full \
  --transport "$coordinator_transport" \
  --listen "$addr" \
  --model-id "$model_id" \
  --catalog "$catalog" \
  --expert-hosts "$expert_hosts" \
  >"${log_dir}/${log_prefix}-coordinator.log" 2>&1 &
coordinator_pid="$!"

if ! wait_for_health; then
  sed -n '1,220p' "${log_dir}/${log_prefix}-coordinator.log" >&2 || true
  exit 1
fi

echo "== running real-full structured TCP smoke ==" >&2
set +e
DS41RT_REAL_FULL_SMOKE_TRANSPORT="$coordinator_transport" \
  scripts/real-full-tcp-smoke.sh "$url" "$model" "$max_tokens" \
  > >(tee "${artifact_dir}/${log_prefix}-response.json") \
  2> >(tee "${artifact_dir}/${log_prefix}-summary.txt" >&2)
smoke_status=$?
set -e

cp "${log_dir}/${log_prefix}-coordinator.log" "${artifact_dir}/${log_prefix}-coordinator.log"
if [ "$smoke_status" -ne 0 ]; then
  echo "real-full TCP live smoke failed with status ${smoke_status}" >&2
  exit "$smoke_status"
fi

echo "real_full_tcp_live_smoke artifacts=${artifact_dir}/${log_prefix}-*.{json,txt,log}" >&2
