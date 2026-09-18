"""Exercise the real shared launch guard with a host-only CUDA/variant fixture."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(shutil.which("c++"), "requires a C++ compiler")
class ExpertLaunchContractTests(unittest.TestCase):
    def test_cluster_policy_and_pointer_guards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "cuda_runtime.h").write_text("""
#pragma once
using cudaLibrary_t = void*;
using cudaError_t = int;
constexpr int cudaSuccess=0, cudaErrorInvalidValue=1, cudaErrorInvalidDevice=101;
constexpr int cudaDevAttrComputeCapabilityMajor=0, cudaDevAttrComputeCapabilityMinor=1,
              cudaDevAttrMultiProcessorCount=2;
inline int cudaLibraryUnload(void*) { return 0; }
inline int cudaGetDevice(int* p) { *p=0; return 0; }
inline int cudaDeviceGetAttribute(int* p, int attr, int) {
  *p = attr == 0 ? 12 : attr == 1 ? 0 : 188; return 0;
}
""")
            (root / "v41_local_expert_variants.h").write_text("""
#pragma once
#define DS41RT_V41_CC_MINOR 0
#define DS41RT_V41_SMS 188
static int launches=0;
static void initialize(void**) {}
static void load(void**) {}
static void launch(void** args, int count) {
  ++launches;
  *static_cast<int*>(args[52]) = count == 53 ? 0 : 1;
}
#define DS41RT_V41_VARIANTS {{2,3,384,5120,1152,1152,6,16,4096,32,32,64,64,188,1},initialize,load,launch,{0}}
""")
            (root / "test.cc").write_text(f"""
#define DS41RT_V41_LOCAL_EXPERTS 1
#ifdef TEST_NVFP4
#define DS41RT_V41_NVFP4_VARIANTS_HEADER "v41_local_expert_variants.h"
#endif
#include "{ROOT / 'native/src/v41_experts.cc'}"
#include <cassert>
extern "C" int32_t ds41rt_v41_initialize_scratch_storage_async(
    void*, uint64_t, uint64_t, uint64_t, uint32_t, void*) {{ return 0; }}
int main() {{
  void* kernel=nullptr;
  assert(ds41rt_v41_expert_initialize(16, &kernel) == 0);
  ds41rt_v41_expert_launch_t args{{}};
  for (auto& p : args.tensors) p = &args;
  args.num_tokens=1; args.scatter_rows=6;
  args.max_rows=32; args.rows_padded=32;
  args.max_tasks=64; args.max_phys_tiles=64;
  for (int cap : {{1,188,376}}) {{
    args.max_active_clusters=cap;
    assert(ds41rt_v41_expert_launch(kernel, &args) == 0);
  }}
  int before=launches;
  for (int cap : {{-2,0,377}}) {{
    args.max_active_clusters=cap;
    assert(ds41rt_v41_expert_launch(kernel, &args) == 1);
  }}
  assert(launches == before);
  args.max_active_clusters=-1;
  assert(ds41rt_v41_expert_launch(kernel, &args) == 1);
  args.max_active_clusters=188;
  before=launches;
  for (auto& p : args.tensors) {{
    p=nullptr;
    assert(ds41rt_v41_expert_launch(kernel, &args) == 1);
    p=&args;
  }}
  for (auto* scalar : {{&args.max_rows, &args.rows_padded,
                        &args.max_tasks, &args.max_phys_tiles, &args.scatter_rows}}) {{
    ++*scalar;
    assert(ds41rt_v41_expert_launch(kernel, &args) == 1);
    --*scalar;
  }}
  for (int rows : {{-1,0,17}}) {{
    args.num_tokens=rows; args.scatter_rows=rows*6;
    assert(ds41rt_v41_expert_launch(kernel, &args) == 1);
  }}
  assert(launches == before);
}}
""")
            for nvfp4 in (False, True):
                with self.subTest(nvfp4=nvfp4):
                    command = ["c++", "-std=c++17", "-I", str(root), "-I",
                               str(ROOT / "native/include"), str(root / "test.cc"),
                               "-o", str(root / "test")]
                    if nvfp4:
                        command.append("-DTEST_NVFP4")
                    subprocess.run(command, check=True, capture_output=True, text=True)
                    subprocess.run([str(root / "test")], check=True)


if __name__ == "__main__":
    unittest.main()
