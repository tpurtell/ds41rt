#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: build-release-artifacts.sh SOURCE_DIR ROLE CUDA_ARCH OUTPUT_DIR" >&2
  exit 2
}

[[ $# -eq 4 ]] || usage
source_dir="$(realpath "$1")"
role="$2"
cuda_arch="$3"
output_dir="$(realpath -m "$4")"
# Where the writable build root is created. Unset keeps the historical in-container
# /tmp scratch; build.sh relocates it to a unique per-task path on the host, mounted
# at the identical path in the container, so a release build never depends on how
# large a container's default /tmp happens to be. mktemp is given an absolute
# template, so honoring this variable is what makes the relocation real: a mere
# TMPDIR export would be ignored by the template.
build_root_parent="${DS41RT_RELEASE_BUILD_ROOT:-/tmp}"
# Reject unsafe output/cache filesystems before staging or invoking Cargo.
# SOURCE_DIR is a read-only input: the release container mounts it `/source:ro`
# and this script stages a writable copy into the build root below, so probing
# the source here would only trip the guard's fail-closed read-only rule. The
# cargo target is guarded because it is where the daemon is written, and the build
# root parent because the default target lives beneath it. Each filesystem is
# probed once: with no target override the parent already covers it, and the parent
# is added separately only when a relocated target points elsewhere. With the hook
# unset this is the historical output, /tmp, cargo-home probe.
build_root_probe=()
if [[ -n "${CARGO_TARGET_DIR:-}" && "$CARGO_TARGET_DIR" != "$build_root_parent" ]]; then
  build_root_probe=("$build_root_parent")
fi
python3 "$(dirname "$0")/assert-build-filesystem.py" \
  "$output_dir" "${CARGO_TARGET_DIR:-$build_root_parent}" "${CARGO_HOME:-$HOME/.cargo}" \
  ${build_root_probe[@]+"${build_root_probe[@]}"}

# Report the real cause here instead of an opaque mktemp failure: a requested build
# root the container cannot write means the bind mount is missing, or it exists on
# the host as a user the container does not run as.
if ! test -w "$build_root_parent"; then
  echo "release build root is not writable inside this container: $build_root_parent" >&2
  echo "it must exist on the host and be bind-mounted at the identical path (see build.sh --help)" >&2
  exit 2
fi

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
exl3_paired_tp4=OFF
exl3_residency=""
# Opt-in replicated-group Spark expert roles. Empty is the default and keeps
# the historical Spark TP4 shard (and every release default) byte-identical.
spark_tp_roles="${DS41RT_RELEASE_SPARK_TP_ROLES:-}"
spark_tp_role_list=()
if [[ -n "$spark_tp_roles" ]]; then
  IFS=';' read -ra spark_tp_role_list <<<"$spark_tp_roles"
  for spark_tp_role in "${spark_tp_role_list[@]}"; do
    case "$spark_tp_role" in
      tp2|tp3|tp6) ;;
      *) echo "DS41RT_RELEASE_SPARK_TP_ROLES accepts only tp2, tp3 and tp6, got: $spark_tp_role" >&2; exit 2 ;;
    esac
  done
  [[ "$role" == expert ]] ||
    { echo "DS41RT_RELEASE_SPARK_TP_ROLES is only valid for the expert role" >&2; exit 2; }
fi
# v7 ships both EXL3 decoder families by default: the uniform K=2 raw
# publication family (2,3) and the staged K3.25 family (3,4). Paired TP4
# builds remain single-family and stay on the v5 (3,4) family.
exl3_bit_families="${DS41RT_RELEASE_EXL3_BIT_FAMILIES:-2,3;3,4}"
case "${DS41RT_RELEASE_EXL3_PAIRED_TP4:-off}" in
  on)
    [[ "$role" == expert ]] || { echo "Paired EXL3 package requires expert role" >&2; exit 2; }
    exl3_paired_tp4=ON
    exl3_residency=80=2
    exl3_bit_families="${DS41RT_RELEASE_EXL3_BIT_FAMILIES:-3,4}"
    ;;
  off) ;;
  *) echo "DS41RT_RELEASE_EXL3_PAIRED_TP4 must be on or off" >&2; exit 2 ;;
esac
[[ "$cuda_arch" =~ ^[0-9]+$ ]] || {
  echo "CUDA_ARCH must be numeric" >&2
  exit 2
}
[[ -f "$source_dir/rust/Cargo.toml" && -f "$source_dir/native/CMakeLists.txt" ]] || {
  echo "SOURCE_DIR is not a DS41RT source tree: $source_dir" >&2
  exit 2
}
[[ -f "$source_dir/THIRD_PARTY_NOTICES.md" ]] || {
  echo "SOURCE_DIR is missing THIRD_PARTY_NOTICES.md" >&2
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

build_root="$(mktemp -d "$build_root_parent/ds41rt-release-build.XXXXXX")"
trap 'rm -rf "$build_root"' EXIT
mkdir -p "$build_root/source"
# The cargo target and the install source must agree. Default to the writable
# staged copy; an explicit external override is honored for both so a caller
# that relocates the target cannot leave the install pointing at the old path.
cargo_target_dir="${CARGO_TARGET_DIR:-$build_root/source/rust/target}"
tar \
  -C "$source_dir" \
  --exclude=.git \
  --exclude='*/.git' \
  --exclude='.venv*' \
  --exclude='*/.venv*' \
  --exclude=.mypy_cache \
  --exclude='*/.mypy_cache' \
  --exclude=.pytest_cache \
  --exclude='*/.pytest_cache' \
  --exclude=.ruff_cache \
  --exclude='*/.ruff_cache' \
  --exclude=__pycache__ \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  --exclude=.ds41rt-cache \
  --exclude=.ds41rt-release \
  --exclude=.ds41rt-release-image \
  --exclude=.ds41rt-wip \
  --exclude=dist \
  --exclude=rust/target \
  --exclude='native/build*' \
  -cf - . |
  tar -C "$build_root/source" -xf -

python3 "$build_root/source/scripts/verify-sparkinfer-source.py" \
  --source "$build_root/source/third_party/sparkinfer" \
  --lock "$build_root/source/third_party/sparkinfer.lock.json" \
  --require-no-python-cache
if [[ "$xgrammar" == ON ]]; then
  python3 "$build_root/source/scripts/verify-xgrammar-source.py" \
    --source "$build_root/source/third_party/xgrammar" \
    --lock "$build_root/source/third_party/xgrammar.lock.json"
fi

export PYO3_PYTHON=python3
CARGO_TARGET_DIR="$cargo_target_dir" cargo build \
  --manifest-path "$build_root/source/rust/Cargo.toml" \
  -p ds41rt-daemon \
  --release

# Release images serve the native V4.1 path. The DS4 Flash/Pro AOT bridge is
# retained for development commands, but must not add legacy generated kernels
# or ABI coupling to the release artifact.
cmake \
  -S "$build_root/source/native" \
  -B "$build_root/native" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DDS41RT_ENABLE_CUDA=ON \
  -DDS41RT_ENABLE_V41_EXPERT_AOT=ON \
  -DDS41RT_V41_SPARK_TP_ROLES="$spark_tp_roles" \
  -DDS41RT_ENABLE_V41_NVFP4_AOT="${DS41RT_RELEASE_NVFP4_AOT:-ON}" \
  -DDS41RT_ENABLE_V41_EXL3_AOT=ON \
  -DDS41RT_V41_EXL3_BIT_FAMILIES="$exl3_bit_families" \
  -DDS41RT_V41_EXL3_PAIRED_TP4="$exl3_paired_tp4" \
  -DDS41RT_V41_EXL3_RESIDENCY="$exl3_residency" \
  -DDS41RT_ENABLE_V41_LOCAL_EXPERT_AOT="$coordinator_aot" \
  -DDS41RT_ENABLE_V41_TP2_EXPERT_AOT="$coordinator_aot" \
  -DDS41RT_ENABLE_V41_FP8_AOT="$coordinator_aot" \
  -DDS41RT_ENABLE_V41_ATTENTION_AOT="$coordinator_aot" \
  -DDS41RT_ENABLE_V41_HC_LAGGED_AOT="$coordinator_aot" \
  -DDS41RT_ENABLE_V41_NARROW_AOT="$coordinator_aot" \
  -DDS41RT_ENABLE_RDMA=ON \
  -DDS41RT_ENABLE_SPARKINFER_AOT="$sparkinfer_aot" \
  -DDS41RT_ENABLE_SPARKINFER_COORDINATOR_AOT="$coordinator_aot" \
  -DDS41RT_ENABLE_DS4_FLASH_AOT=OFF \
  -DDS41RT_ENABLE_W8A16_AOT="$w8a16_aot" \
  -DDS41RT_SPARKINFER_SOURCE_DIR="$build_root/source/third_party/sparkinfer" \
  -DDS41RT_SPARKINFER_LOCK_FILE="$build_root/source/third_party/sparkinfer.lock.json" \
  -DDS41RT_ENABLE_NCCL="$nccl" \
  -DDS41RT_ENABLE_XGRAMMAR="$xgrammar" \
  -DDS41RT_XGRAMMAR_SOURCE_DIR="$build_root/source/third_party/xgrammar" \
  -DDS41RT_XGRAMMAR_LOCK_FILE="$build_root/source/third_party/xgrammar.lock.json" \
  -DPython3_EXECUTABLE="$(command -v python3)" \
  -DDS41RT_CUDA_ARCHITECTURES="$cuda_arch"
cmake --build "$build_root/native"

install -d "$output_dir"
install -m 0755 "$cargo_target_dir/release/ds41rt" "$output_dir/ds41rt"
install -m 0755 "$build_root/native/libds41rt_native.so" "$output_dir/libds41rt_native.so"
exl3_family_tags=()
IFS=';' read -ra exl3_family_list <<<"$exl3_bit_families"
for exl3_family in "${exl3_family_list[@]}"; do
  exl3_family_tags+=("k${exl3_family//,/}")
done
# Families nest under exl3/ so images keep one well-known EXL3 root; the
# daemon resolves exl3/exl3-kXX by checkpoint tiers and treats a direct
# layout child of exl3/ as the legacy single-family package.
for exl3_tag in "${exl3_family_tags[@]}"; do
  python3 "$build_root/source/python/tools/package_v41_exl3_aot.py" install \
    --package "$build_root/native/exl3-$exl3_tag" --output "$output_dir/exl3/exl3-$exl3_tag"
  python3 "$build_root/source/python/tools/package_v41_exl3_aot.py" verify \
    --package "$output_dir/exl3/exl3-$exl3_tag" --role "$role"
done
install -m 0644 "$build_root/native/v41_experts/v41_experts.json" "$output_dir/V41_EXPERT_AOT.json"
# Always write the built-role manifest, including the empty-role default, so
# the release Dockerfile can COPY it unconditionally. Every listed role is
# derived from the AOT export manifest CMake actually produced.
python3 "$build_root/source/scripts/write-v41-expert-tp-manifest.py" \
  --role "$role" \
  --requested "$spark_tp_roles" \
  --native-build-dir "$build_root/native" \
  --native-library "$build_root/native/libds41rt_native.so" \
  --output "$output_dir/V41_EXPERT_TP_AOT.json"
if [[ "$coordinator_aot" == ON ]]; then
  # Automatic RTX placement requires the full local-expert ABI in release images.
  python3 - "$output_dir/libds41rt_native.so" <<'PY_CHECK'
import ctypes
import sys
library = ctypes.CDLL(sys.argv[1])
getattr(library, "ds41rt_v41_local_expert_info")
getattr(library, "ds41rt_v41_tp2_expert_info")
PY_CHECK
  install -m 0644 "$build_root/native/v41_fp8/v41_fp8.json" "$output_dir/V41_FP8_AOT.json"
else
  printf '%s\n' '{"schema":1,"role":"expert","enabled":false}' >"$output_dir/V41_FP8_AOT.json"
fi
install -m 0644 \
  "$build_root/source/THIRD_PARTY_NOTICES.md" \
  "$output_dir/THIRD_PARTY_NOTICES.md"
install -m 0644 \
  "$build_root/source/third_party/sparkinfer/LICENSE" \
  "$output_dir/SPARKINFER_LICENSE"
install -m 0644 \
  "$build_root/source/third_party/xgrammar/LICENSE" \
  "$output_dir/XGRAMMAR_LICENSE"
install -m 0644 \
  "$build_root/source/third_party/xgrammar.lock.json" \
  "$output_dir/XGRAMMAR_PROVENANCE.json"
python3 "$build_root/source/scripts/sparkinfer-release-provenance.py" \
  --source "$build_root/source/third_party/sparkinfer" \
  --lock "$build_root/source/third_party/sparkinfer.lock.json" \
  --license "$output_dir/SPARKINFER_LICENSE" \
  --notices "$output_dir/THIRD_PARTY_NOTICES.md" \
  --write "$output_dir/SPARKINFER_PROVENANCE.json"
(
  cd "$output_dir"
  sha256sum \
    THIRD_PARTY_NOTICES.md \
    SPARKINFER_PROVENANCE.json \
    SPARKINFER_LICENSE >SPARKINFER_SHA256SUMS
  sha256sum -c SPARKINFER_SHA256SUMS
  sha256sum \
    THIRD_PARTY_NOTICES.md \
    XGRAMMAR_PROVENANCE.json \
    XGRAMMAR_LICENSE >XGRAMMAR_SHA256SUMS
  sha256sum -c XGRAMMAR_SHA256SUMS
)
test -x "$output_dir/ds41rt"
test -s "$output_dir/libds41rt_native.so"
test -s "$output_dir/V41_EXPERT_TP_AOT.json"
test -s "$output_dir/THIRD_PARTY_NOTICES.md"
test -s "$output_dir/SPARKINFER_PROVENANCE.json"
test -s "$output_dir/SPARKINFER_LICENSE"
test -s "$output_dir/SPARKINFER_SHA256SUMS"
test -s "$output_dir/XGRAMMAR_PROVENANCE.json"
test -s "$output_dir/XGRAMMAR_LICENSE"
test -s "$output_dir/XGRAMMAR_SHA256SUMS"
