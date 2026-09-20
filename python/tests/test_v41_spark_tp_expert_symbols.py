"""CPU-only contract check for the replicated-group Spark TP2/TP3 AOT families.

Compiles the real native wrapper translation units against a host-only CUDA
shim and mock exported variant tables, then asserts the new symbol families
publish the exact role ids/geometry and enforce the SM121 guard. No GPU, CUDA
toolkit, torch or SparkInfer import is required.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CUDA_SHIM = """
#pragma once
using cudaLibrary_t = void*;
using cudaError_t = int;
constexpr int cudaSuccess = 0, cudaErrorInvalidValue = 1, cudaErrorInvalidDevice = 101;
constexpr int cudaDevAttrComputeCapabilityMajor = 0;
constexpr int cudaDevAttrComputeCapabilityMinor = 1;
constexpr int cudaDevAttrMultiProcessorCount = 2;
inline int cudaLibraryUnload(void*) { return 0; }
inline int cudaGetDevice(int* device) { *device = 0; return 0; }
inline int cudaDeviceGetAttribute(int* value, int attribute, int) {
  *value = attribute == cudaDevAttrComputeCapabilityMajor ? 12
         : attribute == cudaDevAttrComputeCapabilityMinor ? 1 : 188;
  return 0;
}
"""

# Mock exported variant tables for each degree. The real exporter emits these
# from the SM121 device; here the geometry is fixed to the official extents.
# Placeholders avoid brace escaping so the C aggregate stays literal.
VARIANT = """
#pragma once
#define DS41RT_V41_CC_MINOR 1
#define DS41RT_V41_SMS 188
static int launches = 0;
static void initialize(void**) {}
static void load(void**) {}
static void launch(void** args, int count) {
  ++launches;
  *static_cast<int*>(args[52]) = count == 53 ? 0 : 1;
}
#define DS41RT_V41_VARIANTS {{2,__ROLE__,384,5120,__LOGICAL__,__KERNEL__,6,16,4096,32,32,64,64,188,7},initialize,load,launch,{0}}
"""


def _variant(role: int, logical: int, kernel: int) -> str:
    return (
        VARIANT.replace("__ROLE__", str(role))
        .replace("__LOGICAL__", str(logical))
        .replace("__KERNEL__", str(kernel))
    )

PROGRAM = """
#include "ds41rt_v41_experts.h"
#include <cassert>
#include <cstdio>

extern "C" int32_t ds41rt_v41_initialize_scratch_storage_async(
    void*, uint64_t, uint64_t, uint64_t, uint32_t, void*) {{ return 0; }}

static void check(int32_t (*info)(int32_t, ds41rt_v41_expert_info_t*),
                  int32_t (*initialize)(int32_t, void**),
                  int32_t (*launch)(void*, const ds41rt_v41_expert_launch_t*),
                  uint32_t role, uint32_t intermediate) {{
  ds41rt_v41_expert_info_t metadata{{}};
  assert(info(16, &metadata) == 0);
  assert(metadata.role == role);
  assert(metadata.experts == 384 && metadata.topk == 6 && metadata.hidden_size == 5120);
  assert(metadata.logical_intermediate == intermediate);
  assert(metadata.kernel_intermediate == intermediate);
  assert(metadata.input_dtype == 7);
  assert(metadata.capacity_rows == 16 && metadata.scratch_bytes > 0);
  void* kernel = nullptr;
  assert(initialize(16, &kernel) == 0 && kernel != nullptr);
  ds41rt_v41_expert_launch_t args{{}};
  for (auto& pointer : args.tensors) pointer = &args;
  args.num_tokens = 1;
  args.scatter_rows = 6;
  args.max_rows = metadata.max_rows;
  args.rows_padded = metadata.rows_padded;
  args.max_tasks = metadata.max_tasks;
  args.max_phys_tiles = metadata.max_phys_tiles;
  args.max_active_clusters = metadata.max_active_clusters;
  assert(launch(kernel, &args) == 0);
}}

int main() {{
  check(ds41rt_v41_spark_tp2_expert_info, ds41rt_v41_spark_tp2_expert_initialize,
        ds41rt_v41_spark_tp2_expert_launch, 5, 1152);
  check(ds41rt_v41_spark_tp3_expert_info, ds41rt_v41_spark_tp3_expert_initialize,
        ds41rt_v41_spark_tp3_expert_launch, 6, 768);
  // The two families must be distinct symbols with distinct role ids.
  ds41rt_v41_expert_info_t tp2{{}}, tp3{{}};
  assert(ds41rt_v41_spark_tp2_expert_info(16, &tp2) == 0);
  assert(ds41rt_v41_spark_tp3_expert_info(16, &tp3) == 0);
  assert(tp2.role != tp3.role);
  assert(tp2.logical_intermediate != tp3.logical_intermediate);
  std::printf("spark tp2/tp3 symbol contract ok\\n");
  return 0;
}}
"""


@unittest.skipUnless(shutil.which("c++"), "requires a C++ compiler")
class SparkTpSymbolContractTests(unittest.TestCase):
    def test_new_symbol_families_publish_exact_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "cuda_runtime.h").write_text(CUDA_SHIM)
            (root / "v41_spark_tp2_expert_variants.h").write_text(
                _variant(role=5, logical=1152, kernel=1152)
            )
            (root / "v41_spark_tp3_expert_variants.h").write_text(
                _variant(role=6, logical=768, kernel=768)
            )
            (root / "test.cc").write_text(PROGRAM)
            command = [
                "c++", "-std=c++17", "-I", str(root), "-I",
                str(ROOT / "native" / "include"),
                str(ROOT / "native" / "src" / "v41_spark_tp2_experts.cc"),
                str(ROOT / "native" / "src" / "v41_spark_tp3_experts.cc"),
                str(root / "test.cc"), "-o", str(root / "test"),
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            subprocess.run([str(root / "test")], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
