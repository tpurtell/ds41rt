#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: build-wip-artifacts.sh SOURCE_DIR ROLE CUDA_ARCH BUILD_DIR OUTPUT_DIR" >&2
  exit 2
}

[[ $# -eq 5 ]] || usage
source_dir="$(realpath "$1")"
role="$2"
cuda_arch="$3"
build_dir="$(realpath -m "$4")"
output_dir="$(realpath -m "$5")"

case "$role" in
  coordinator)
    sparkinfer_aot=OFF
    coordinator_aot=ON
    w8a16_aot=ON
    nccl=OFF
    xgrammar=ON
    ;;
  expert)
    sparkinfer_aot=ON
    coordinator_aot=OFF
    w8a16_aot=OFF
    nccl=ON
    xgrammar=OFF
    ;;
  *)
    echo "ROLE must be coordinator or expert" >&2
    exit 2
    ;;
esac
[[ "$cuda_arch" =~ ^[0-9]+$ ]] || {
  echo "CUDA_ARCH must be numeric" >&2
  exit 2
}
[[ -f "$source_dir/rust/Cargo.toml" && -f "$source_dir/native/CMakeLists.txt" ]] || {
  echo "SOURCE_DIR is not a DS41RT source tree: $source_dir" >&2
  exit 2
}

python3 "$source_dir/scripts/verify-sparkinfer-source.py" \
  --source "$source_dir/third_party/sparkinfer" \
  --lock "$source_dir/third_party/sparkinfer.lock.json"
if [[ "$xgrammar" == ON ]]; then
  python3 "$source_dir/scripts/verify-xgrammar-source.py" \
    --source "$source_dir/third_party/xgrammar" \
    --lock "$source_dir/third_party/xgrammar.lock.json"
fi

mkdir -p "$build_dir" "$output_dir"
export PYO3_PYTHON=python3
export PYTHONPATH="$source_dir/third_party/sparkinfer:$source_dir/python/reference/ds41rt_reference:$source_dir/python/reference${PYTHONPATH:+:$PYTHONPATH}"
export CARGO_TARGET_DIR="$build_dir/cargo-target"

cargo build \
  --quiet \
  --manifest-path "$source_dir/rust/Cargo.toml" \
  -p ds41rt-daemon \
  --release

cmake \
  -S "$source_dir/native" \
  -B "$build_dir/native" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DDS41RT_ENABLE_CUDA=ON \
  -DDS41RT_ENABLE_V41_EXPERT_AOT=ON \
  -DDS41RT_ENABLE_V41_EXL3_AOT=ON \
  -DDS41RT_V41_EXL3_BITS="${DS41RT_WIP_EXL3_BITS:-2;3}" \
  -DDS41RT_ENABLE_V41_FP8_AOT="$coordinator_aot" \
  -DDS41RT_ENABLE_V41_ATTENTION_AOT="$coordinator_aot" \
  -DDS41RT_ENABLE_V41_HC_LAGGED_AOT="$coordinator_aot" \
  -DDS41RT_ENABLE_V41_NARROW_AOT="$coordinator_aot" \
  -DDS41RT_ENABLE_RDMA=ON \
  -DDS41RT_ENABLE_SPARKINFER_AOT="$sparkinfer_aot" \
  -DDS41RT_ENABLE_SPARKINFER_COORDINATOR_AOT="$coordinator_aot" \
  -DDS41RT_ENABLE_DS4_FLASH_AOT=ON \
  -DDS41RT_ENABLE_W8A16_AOT="$w8a16_aot" \
  -DDS41RT_SPARKINFER_SOURCE_DIR="$source_dir/third_party/sparkinfer" \
  -DDS41RT_SPARKINFER_LOCK_FILE="$source_dir/third_party/sparkinfer.lock.json" \
  -DDS41RT_ENABLE_NCCL="$nccl" \
  -DDS41RT_ENABLE_XGRAMMAR="$xgrammar" \
  -DDS41RT_XGRAMMAR_SOURCE_DIR="$source_dir/third_party/xgrammar" \
  -DDS41RT_XGRAMMAR_LOCK_FILE="$source_dir/third_party/xgrammar.lock.json" \
  -DPython3_EXECUTABLE="$(command -v python3)" \
  -DDS41RT_CUDA_ARCHITECTURES="$cuda_arch"
cmake --build "$build_dir/native"

install -m 0755 "$CARGO_TARGET_DIR/release/ds41rt" "$output_dir/ds41rt"
install -m 0755 "$build_dir/native/libds41rt_native.so" "$output_dir/libds41rt_native.so"
python3 "$source_dir/python/tools/package_v41_exl3_aot.py" install \
  --package "$build_dir/native/exl3" --output "$output_dir/exl3"
python3 "$source_dir/python/tools/package_v41_exl3_aot.py" verify \
  --package "$output_dir/exl3" --role "$role"
install -m 0644 "$build_dir/native/v41_experts/v41_experts.json" "$output_dir/V41_EXPERT_AOT.json"
if [[ "$coordinator_aot" == ON ]]; then
  install -m 0644 "$build_dir/native/v41_fp8/v41_fp8.json" "$output_dir/V41_FP8_AOT.json"
else
  printf '%s\n' '{"schema":1,"role":"expert","enabled":false}' >"$output_dir/V41_FP8_AOT.json"
fi
(
  cd "$output_dir"
  sha256sum ds41rt libds41rt_native.so V41_EXPERT_AOT.json V41_FP8_AOT.json >ARTIFACT_SHA256SUMS
  sha256sum -c ARTIFACT_SHA256SUMS
)
