set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

model_id := env_var_or_default("DS41RT_MODEL_ID", "deepseek-ai/DeepSeek-V4.1-Flash")
expert_roles := env_var_or_default("DS41RT_EXPERT_HOSTS", "spark-0,spark-1,spark-2,spark-3")
spark_hosts := env_var_or_default("DS41RT_SPARK_HOSTS", "ostrich,dodo,emu,kiwi")
base_image := env_var_or_default("DS41RT_CONTAINER_BASE", "nvcr.io/nvidia/pytorch:26.05-py3")
spark_catalog := env_var_or_default("DS41RT_PHASE0_SPARK_CATALOG", ".ds41rt-cache/model-artifacts/diagnostic/model_catalog.json")
spark_loadplan_dir := env_var_or_default("DS41RT_PHASE0_SPARK_LOADPLAN_DIR", ".ds41rt-cache/model-artifacts/diagnostic")
HOSTS := env_var_or_default("HOSTS", "")
MODE := env_var_or_default("MODE", "")

default:
    @just --list

doctor-host:
    scripts/doctor.sh --role coordinator --model-id "{{ model_id }}"

doctor:
    scripts/ds41rt doctor --role coordinator --model-id "{{ model_id }}"

doctor-container-coordinator:
    scripts/ds41rt-dev.sh coordinator ds41rt doctor --role coordinator --model-id "{{ model_id }}"

doctor-hosts HOSTS=spark_hosts:
    scripts/run-on-hosts.sh "{{ HOSTS }}" 'cd {{ justfile_directory() }} && scripts/doctor.sh --role expert --model-id "{{ model_id }}"'

build-rust:
    cargo build --manifest-path rust/Cargo.toml --workspace

test-rust:
    scripts/run-with-python-env.sh \
      cargo test --manifest-path rust/Cargo.toml --workspace

test-rust-fast:
    RUSTFLAGS="${RUSTFLAGS:--Awarnings}" \
      DS41RT_DISABLE_NATIVE_AUTO_DISCOVERY=1 \
      scripts/run-with-python-env.sh \
      cargo test --manifest-path rust/Cargo.toml --workspace --exclude ds41rt-daemon
    RUSTFLAGS="${RUSTFLAGS:--Awarnings}" \
      env -u DS41RT_NATIVE_LIB -u DS41RT_REAL_FULL_CUDA_REFERENCE_KERNELS -u DS41RT_B12X \
      DS41RT_DISABLE_NATIVE_AUTO_DISCOVERY=1 \
      scripts/run-with-python-env.sh \
      cargo test --manifest-path rust/Cargo.toml -p ds41rt-daemon -- \
        --skip real_checkpoint \
        --skip when_available \
        --skip when_cuda_available \
        --skip when_cuda_enabled \
        --skip native_available \
        --skip real_full_preflight \
        --skip real_full_runtime \
        --skip real_full_info_from_report \
        --skip uses_coord_dense_graph_slot \
        --skip uses_coord_sparse_a_graph_slot \
        --skip replays_same_bucket_when_rows_change \
        --skip cuda_graph \
        --skip b12x \
        --skip triton

build-native-coordinator-test:
    python="{{ justfile_directory() }}/.venv/bin/python"; \
      test -x "$python"; \
      cmake -S native -B native/build-cuda-rdma-coordinator-aot -G Ninja \
        -U DS41RT_ENABLE_B12X_AOT \
        -U DS41RT_ENABLE_B12X_COORDINATOR_AOT \
        -DDS41RT_ENABLE_CUDA=ON \
        -DDS41RT_ENABLE_RDMA=ON \
        -DDS41RT_ENABLE_SPARKINFER_AOT=OFF \
        -DDS41RT_ENABLE_SPARKINFER_COORDINATOR_AOT=ON \
        -DDS41RT_ENABLE_W8A16_AOT=ON \
        -DDS41RT_SPARKINFER_SOURCE_DIR="{{ justfile_directory() }}/third_party/sparkinfer" \
        -DDS41RT_SPARKINFER_LOCK_FILE="{{ justfile_directory() }}/third_party/sparkinfer.lock.json" \
        -DDS41RT_ENABLE_NCCL=OFF \
        -DPython3_EXECUTABLE="$python" \
        -DDS41RT_CUDA_ARCHITECTURES=120
    cmake --build native/build-cuda-rdma-coordinator-aot -j 16
    @echo "coordinator CUDA test library: {{ justfile_directory() }}/native/build-cuda-rdma-coordinator-aot/libds41rt_native.so"

test-rust-phase0a-focused NATIVE_LIB="native/build-cuda-rdma-coordinator-aot/libds41rt_native.so": build-native-coordinator-test
    native_lib="{{ NATIVE_LIB }}"; \
      if [[ "$native_lib" != /* ]]; then native_lib="{{ justfile_directory() }}/$native_lib"; fi; \
      test -f "$native_lib"; \
      printf 'DS41RT_NATIVE_LIB=%s\n' "$native_lib"; \
      python_lib="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")')"; \
      export LD_LIBRARY_PATH="$python_lib:${LD_LIBRARY_PATH:-}"; \
      export RUSTFLAGS="${RUSTFLAGS:--Awarnings}"; \
      export DS41RT_NATIVE_LIB="$native_lib"; \
      export DS41RT_REAL_FULL_CUDA_REFERENCE_KERNELS=1; \
      cargo test --manifest-path rust/Cargo.toml -p ds41rt-core graph -- --test-threads=1; \
      cargo test --manifest-path rust/Cargo.toml -p ds41rt-core kv_cache -- --test-threads=1; \
      cargo test --manifest-path rust/Cargo.toml -p ds41rt-daemon bench_cuda_kernels::tests -- --test-threads=1; \
      cargo test --manifest-path rust/Cargo.toml -p ds41rt-daemon real_full_nvfp4_kv_accounting_uses_targeted_dry_runs -- --test-threads=1; \
      cargo test --manifest-path rust/Cargo.toml -p ds41rt-daemon device_kv_execution_mirror_writes_nvfp4_projected_mla_kv_a -- --test-threads=1; \
      cargo test --manifest-path rust/Cargo.toml -p ds41rt-daemon uses_coord -- --test-threads=1; \
      cargo test --manifest-path rust/Cargo.toml -p ds41rt-daemon replays_same_bucket_when_rows_change -- --test-threads=1; \
      cargo test --manifest-path rust/Cargo.toml -p ds41rt-daemon uses_triton_graph_when_python_enabled -- --test-threads=1; \
      cargo test --manifest-path rust/Cargo.toml -p ds41rt-daemon coordinator_cuda_graph -- --test-threads=1

test-rust-cuda-graphs NATIVE_LIB="native/build-cuda-rdma-coordinator-aot/libds41rt_native.so": build-native-coordinator-test
    ctest --test-dir native/build-cuda-rdma-coordinator-aot --output-on-failure
    native_lib="{{ NATIVE_LIB }}"; \
      if [[ "$native_lib" != /* ]]; then native_lib="{{ justfile_directory() }}/$native_lib"; fi; \
      python_lib="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")')"; \
      test -f "$native_lib"; \
      printf 'DS41RT_NATIVE_LIB=%s\n' "$native_lib"; \
      LD_LIBRARY_PATH="$python_lib:${LD_LIBRARY_PATH:-}" DS41RT_NATIVE_LIB="$native_lib" DS41RT_REAL_FULL_CUDA_REFERENCE_KERNELS=1 cargo test --manifest-path rust/Cargo.toml -p ds41rt-daemon coordinator_cuda_graph
    native_lib="{{ NATIVE_LIB }}"; \
      if [[ "$native_lib" != /* ]]; then native_lib="{{ justfile_directory() }}/$native_lib"; fi; \
      python_lib="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")')"; \
      LD_LIBRARY_PATH="$python_lib:${LD_LIBRARY_PATH:-}" DS41RT_NATIVE_LIB="$native_lib" DS41RT_REAL_FULL_CUDA_REFERENCE_KERNELS=1 cargo test --manifest-path rust/Cargo.toml -p ds41rt-daemon real_checkpoint_layer_ordered_execution_probe_when_available

test-rust-full-attention NATIVE_LIB="native/build-cuda-rdma-coordinator-aot/libds41rt_native.so": build-native-coordinator-test
    native_lib="{{ NATIVE_LIB }}"; \
      if [[ "$native_lib" != /* ]]; then native_lib="{{ justfile_directory() }}/$native_lib"; fi; \
      python_lib="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")')"; \
      test -f "$native_lib"; \
      printf 'DS41RT_NATIVE_LIB=%s\n' "$native_lib"; \
      LD_LIBRARY_PATH="$python_lib:${LD_LIBRARY_PATH:-}" DS41RT_NATIVE_LIB="$native_lib" DS41RT_REAL_FULL_CUDA_REFERENCE_KERNELS=1 cargo test --manifest-path rust/Cargo.toml -p ds41rt-daemon real_checkpoint_layer_ordered_full_output_mla_rope_attention_mlp_probe_when_available -- --ignored

test-python:
    cd python && ../scripts/run-with-python-env.sh \
      uv run --frozen pytest reference/tests ../scripts/tests

test-native:
    cmake -S native -B native/build -G Ninja -DDS41RT_ENABLE_CUDA=OFF -DDS41RT_ENABLE_RDMA=OFF
    cmake --build native/build
    ctest --test-dir native/build --output-on-failure
    native_lib="{{ justfile_directory() }}/native/build/libds41rt_native.so"; \
      for test_name in \
        tests::native_version_call \
        tests::error_propagation \
        tests::allocate_copy_free_roundtrip \
        tests::rdma_device_info_and_host_buffer_plan \
        tests::pro_router_ffi_requires_exact_production_geometry; do \
        DS41RT_NATIVE_LIB="$native_lib" cargo test \
          --manifest-path rust/Cargo.toml -p ds41rt-ffi \
          "$test_name" -- --exact || exit; \
      done

test-native-rdma OUT="reports/phase0_artifacts/native_rdma_enabled_build_status.json":
    python python/tools/check_native_rdma_build.py --clean --output "{{ OUT }}"

test-smoke: doctor-host build-rust test-rust

docker-build-coordinator:
    docker build \
      --platform linux/amd64 \
      --build-arg BASE_IMAGE="{{ base_image }}" \
      --build-arg DS41RT_ROLE=coordinator \
      --build-arg CUDA_ARCH=120 \
      --build-arg TARGET_PLATFORM=linux/amd64 \
      -f docker/Dockerfile.dev \
      -t ds41rt-coordinator-dev .

docker-build-spark HOSTS=spark_hosts:
    DS41RT_SPARK_HOSTS="{{ HOSTS }}" \
      DS41RT_PHASE0_SPARK_EXPERT_MODE=synthetic \
      DS41RT_SPARK_IMAGE_COPY_METHOD=none \
      DS41RT_SPARK_BUILD_IMAGE=1 \
      DS41RT_SPARK_FORCE_BUILD_IMAGE=1 \
      DS41RT_SPARK_IMAGE_ONLY=1 \
      scripts/phase0-spark-tcp-bench.sh

docker-gpu-check IMAGE="ds41rt-coordinator-dev":
    DS41RT_DOCKER_GPU_VERIFY_IMAGE="{{ IMAGE }}" scripts/configure-docker-nvidia-runtime.sh --verify-only

docker-configure-nvidia-runtime:
    sudo scripts/configure-docker-nvidia-runtime.sh

docker-shell-coordinator *ARGS:
    scripts/ds41rt-dev.sh coordinator {{ ARGS }}

docker-shell-spark *ARGS:
    scripts/ds41rt-dev.sh expert {{ ARGS }}

inspect-model:
    scripts/ds41rt inspect-model --model-id "{{ model_id }}" --out .ds41rt-cache/model-artifacts/diagnostic/model_catalog.json --summary .ds41rt-cache/model-artifacts/diagnostic/tensor_summary.md

make-loadplan POLICY="modulo":
    scripts/ds41rt make-loadplan --catalog .ds41rt-cache/model-artifacts/diagnostic/model_catalog.json --policy "{{ POLICY }}" --hosts "{{ expert_roles }}" --out .ds41rt-cache/model-artifacts/diagnostic/loadplan.json

api-smoke MODEL=model_id URL="http://127.0.0.1:8000":
    scripts/api-smoke.sh "{{ URL }}" "{{ MODEL }}"

api-prefill-smoke MODEL="ds41rt-synthetic-ds4-layer" PROMPT_TOKENS="16" URL="http://127.0.0.1:8000":
    scripts/api-prefill-smoke.sh "{{ URL }}" "{{ MODEL }}" "{{ PROMPT_TOKENS }}"

real-slice-tcp-smoke ADDR="127.0.0.1:8073" BASE_PORT="9181":
    ADDR="{{ ADDR }}" BASE_PORT="{{ BASE_PORT }}" scripts/real-slice-tcp-smoke.sh

real-full-tcp-smoke URL="http://127.0.0.1:8000" MODEL=(model_id + "-full") MAX_TOKENS="1":
    scripts/real-full-tcp-smoke.sh "{{ URL }}" "{{ MODEL }}" "{{ MAX_TOKENS }}"

real-full-tcp-smoke-multi-token URL="http://127.0.0.1:8000" MODEL=(model_id + "-full") MAX_TOKENS="2":
    scripts/real-full-tcp-smoke.sh "{{ URL }}" "{{ MODEL }}" "{{ MAX_TOKENS }}"

real-full-tcp-smoke-long-prefill URL="http://127.0.0.1:8000" MODEL=(model_id + "-full") MAX_TOKENS="1":
    DS41RT_REAL_FULL_TCP_SMOKE_PROMPT_REPEAT_TOKEN=chunk DS41RT_REAL_FULL_TCP_SMOKE_PROMPT_REPEAT_COUNT=600 DS41RT_REAL_FULL_TCP_SMOKE_MIN_PREFILL_CHUNKS=2 DS41RT_REAL_FULL_TCP_SMOKE_REQUIRE_RUNTIME_SUMMARY=1 scripts/real-full-tcp-smoke.sh "{{ URL }}" "{{ MODEL }}" "{{ MAX_TOKENS }}"

real-full-tcp-stream-smoke URL="http://127.0.0.1:8000" MODEL=(model_id + "-full") MAX_TOKENS="1":
    scripts/real-full-tcp-stream-smoke.sh "{{ URL }}" "{{ MODEL }}" "{{ MAX_TOKENS }}"

real-full-tcp-live-smoke:
    scripts/real-full-tcp-live-smoke.sh

real-full-tcp-live-smoke-synthetic:
    DS41RT_PHASE0_SPARK_EXPERT_MODE=synthetic LOG_PREFIX=real-full-tcp-live-smoke-synthetic scripts/real-full-tcp-live-smoke.sh

real-full-tcp-live-smoke-multi-token:
    LOG_PREFIX=real-full-tcp-live-smoke-multi-token MAX_TOKENS=2 scripts/real-full-tcp-live-smoke.sh

real-full-tcp-live-smoke-long-prefill:
    LOG_PREFIX=real-full-tcp-live-smoke-long-prefill DS41RT_REAL_FULL_REQUEST_PREFILL_CHUNK_TOKENS=64 DS41RT_REAL_FULL_TCP_SMOKE_PROMPT_REPEAT_TOKEN=chunk DS41RT_REAL_FULL_TCP_SMOKE_PROMPT_REPEAT_COUNT=130 DS41RT_REAL_FULL_TCP_SMOKE_MIN_PREFILL_CHUNKS=2 DS41RT_REAL_FULL_TCP_SMOKE_WARMUP_PREFILL_ROUNDTRIP_ROWS=64,128 DS41RT_REAL_FULL_TCP_SMOKE_WARMUP_PREFILL_CHAIN_ROWS=64,128 scripts/real-full-tcp-live-smoke.sh

serve-real-full-tcp:
    scripts/real-full-tcp-serve.sh

transport-capabilities BENCHMARK_JSONL="reports/phase0_artifacts/benchmarks/phase0_results.jsonl" OUT="reports/phase0_artifacts/transport_capabilities.json":
    scripts/ds41rt transport-capabilities --benchmark-jsonl "{{ BENCHMARK_JSONL }}" --out "{{ OUT }}"

scheduler-smoke:
    scripts/ds41rt scheduler-smoke

start-coordinator MODE="tiny" TRANSPORT="inproc" ADDR="127.0.0.1:8000":
    scripts/ds41rt coordinator --backend "{{ MODE }}" --transport "{{ TRANSPORT }}" --listen "{{ ADDR }}" --model-id "{{ model_id }}" --expert-hosts "{{ expert_roles }}"

start-experts-tcp HOSTS=HOSTS MODE=MODE:
    hosts="${HOSTS:-}"; \
    mode="${MODE:-}"; \
    positional=0; \
    for arg in "{{ HOSTS }}" "{{ MODE }}"; do \
      [ -n "$arg" ] || continue; \
      case "$arg" in \
        HOSTS=*) hosts="${arg#HOSTS=}" ;; \
        MODE=*) mode="${arg#MODE=}" ;; \
        real|synthetic) mode="$arg" ;; \
        *) \
          if [ "$positional" -eq 0 ]; then \
            hosts="$arg"; \
            positional=1; \
          else \
            mode="$arg"; \
          fi \
          ;; \
      esac; \
    done; \
    hosts="${hosts:-{{ spark_hosts }}}"; \
    mode="${mode:-synthetic}"; \
    case "$mode" in \
      synthetic) \
        scripts/start-spark-experts-tcp.sh \
          --hosts "$hosts" \
          --mode synthetic \
        ;; \
      real) \
        scripts/start-spark-experts-tcp.sh \
          --hosts "$hosts" \
          --mode real \
          --catalog "{{ spark_catalog }}" \
          --loadplan-dir "{{ spark_loadplan_dir }}" \
        ;; \
      *) \
        echo 'MODE must be synthetic or real' >&2; \
        exit 2 \
        ;; \
    esac

bench-rdma HOST_A HOST_B:
    scripts/bench-rdma-pair.sh "{{ HOST_A }}" "{{ HOST_B }}"

bench-verbs-app HOST_A HOST_B:
    scripts/bench-verbs-app-pair.sh "{{ HOST_A }}" "{{ HOST_B }}"

bench-verbs-app-coordinator HOSTS=spark_hosts:
    scripts/bench-verbs-app-coordinator-links.sh "{{ HOSTS }}"

bench-phase0-spark-tcp HOSTS=spark_hosts MODE="real":
    DS41RT_SPARK_HOSTS="{{ HOSTS }}" DS41RT_PHASE0_SPARK_EXPERT_MODE="{{ MODE }}" scripts/phase0-spark-tcp-bench.sh
