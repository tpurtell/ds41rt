#include "ds41rt_v41_attention_aot_internal.h"
#include "v41_attention.h"
#include <atomic>
#include <mutex>
namespace {
ds41rt_v41_attention_Kernel_Module_t module{};
std::mutex initialization_mutex;
std::atomic<bool> initialized{false};
bool loaded[32]{};
}
extern "C" int32_t ds41rt_v41_attention_aot_initialize() {
  std::lock_guard<std::mutex> guard(initialization_mutex);
  int device = -1;
  auto status = cudaGetDevice(&device);
  if (status != cudaSuccess) return status;
  if (device < 0 || device >= 32) return cudaErrorInvalidDevice;
  if (loaded[device]) return cudaSuccess;
  cudaDeviceProp properties{};
  status = cudaGetDeviceProperties(&properties, device);
  if (status != cudaSuccess) return status;
  if (properties.major != 12 || properties.minor != 0) return cudaErrorInvalidDevice;
  auto* library = &module.module;
  int result = 0;
  if (!module.module) {
    void* args[] = {&library, &result};
    _mlir_ds41rt_v41_attention_cuda_init(args);
    if (result) return result;
  }
  void* args[] = {&library, &device, &result};
  _mlir_ds41rt_v41_attention_cuda_load_to_device(args);
  if (!result) {
    loaded[device] = true;
    initialized.store(true, std::memory_order_release);
  }
  return result;
}
extern "C" int32_t ds41rt_v41_attention_aot_launch(const void* query, const void* descriptors,
    const void* metadata, const void* selected, const void* bounds, const void* sink,
    void* partials, void* lses, void* output, int32_t rows, void* stream) {
  // Initialization is required on each owning device before graph capture.
  if (!initialized.load(std::memory_order_acquire)) return cudaErrorInitializationError;
  return cute_dsl_ds41rt_v41_attention_wrapper(&module,
      const_cast<void*>(query), const_cast<void*>(descriptors), const_cast<void*>(metadata),
      const_cast<void*>(selected), const_cast<void*>(bounds), const_cast<void*>(sink),
      partials, lses, output, rows, static_cast<cudaStream_t>(stream));
}
