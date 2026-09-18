#include "ds41rt_v41_experts.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <mutex>
#ifdef DS41RT_V41_NVFP4_VARIANTS_HEADER
/* W4A4 (ModelOpt NVFP4) variants carry their own bridge and consume BF16
 * hidden rows, so the FP8 input quantizer is not part of this family. */
#include DS41RT_V41_NVFP4_VARIANTS_HEADER
#elif defined(DS41RT_V41_DSPARK_TP2_EXPERTS)
#include "v41_dspark_tp2_expert_variants.h"
#elif defined(DS41RT_V41_TP2_EXPERTS)
#include "v41_tp2_expert_variants.h"
#elif defined(DS41RT_V41_LOCAL_EXPERTS)
#include "v41_local_expert_variants.h"
#else
#include "v41_expert_variants.h"
#include "v41_input_quant_dispatch.h"
#endif
#ifndef DS41RT_V41_OUTPUT_KIND
#define DS41RT_V41_OUTPUT_KIND(capacity) 0
#endif

static_assert(sizeof(ds41rt_v41_expert_info_t) == 64);
static_assert(sizeof(ds41rt_v41_expert_launch_t) == 392);
static_assert(offsetof(ds41rt_v41_expert_launch_t, stream) == 384);

namespace {
using ModuleFn = void (*)(void**);
using LaunchFn = void (*)(void**, int32_t);
struct Variant {
  ds41rt_v41_expert_info_t info;
  ModuleFn initialize;
  ModuleFn load;
  LaunchFn launch;
  uint64_t scratch_offsets[DS41RT_V41_EXPERT_POINTERS];
  cudaLibrary_t library = nullptr;
  int device = -1;
  ~Variant() { if (library) cudaLibraryUnload(library); }
};
Variant variants[] = {DS41RT_V41_VARIANTS};
#ifdef DS41RT_V41_TP2_EXPERTS
// The dual coordinator exposes exactly two selected GPUs as CUDA devices 0/1.
// Each owns a separate loaded module and immutable handle per capacity.
Variant peer_variants[] = {DS41RT_V41_VARIANTS};
#endif
std::mutex initialization_mutex;
Variant* by_handle(void* handle) {
  for (auto& variant : variants) if (&variant == handle) return &variant;
#ifdef DS41RT_V41_TP2_EXPERTS
  for (auto& variant : peer_variants) if (&variant == handle) return &variant;
#endif
  return nullptr;
}
bool valid_scratch(Variant* variant, void* storage, uint64_t bytes) {
  return variant && variant->device >= 0 && storage &&
    reinterpret_cast<uintptr_t>(storage) % 16 == 0 && bytes >= variant->info.scratch_bytes &&
    reinterpret_cast<uintptr_t>(storage) <= UINTPTR_MAX - variant->info.scratch_bytes;
}
Variant* by_capacity(int32_t capacity) {
#ifdef DS41RT_V41_TP2_EXPERTS
  int device = -1;
  if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device > 1) return nullptr;
  if (device == 1) {
    for (auto& variant : peer_variants)
      if (variant.info.capacity_rows == static_cast<uint32_t>(capacity)) return &variant;
    return nullptr;
  }
#endif
  for (auto& variant : variants)
    if (variant.info.capacity_rows == static_cast<uint32_t>(capacity)) return &variant;
  return nullptr;
}
}

extern "C" int32_t ds41rt_v41_initialize_scratch_storage_async(
    void*, uint64_t, uint64_t, uint64_t, uint32_t, void*);

extern "C" int32_t ds41rt_v41_expert_bind_scratch(void* kernel, void* storage,
    uint64_t bytes, void* tensors[DS41RT_V41_EXPERT_POINTERS]) {
  auto* variant = by_handle(kernel);
  if (!valid_scratch(variant, storage, bytes) || !tensors) return cudaErrorInvalidValue;
#ifdef DS41RT_V41_TP2_EXPERTS
  int device = -1;
  const auto status = cudaGetDevice(&device);
  if (status != cudaSuccess) return status;
  if (device != variant->device) return cudaErrorInvalidDevice;
#endif
  for (int slot = 0; slot < DS41RT_V41_EXPERT_POINTERS; ++slot)
    if (variant->scratch_offsets[slot] != UINT64_MAX)
      tensors[slot] = static_cast<char*>(storage) + variant->scratch_offsets[slot];
  return cudaSuccess;
}

extern "C" int32_t ds41rt_v41_expert_initialize_scratch_async(void* kernel,
    void* storage, uint64_t bytes, void* stream) {
  auto* variant = by_handle(kernel);
  if (!valid_scratch(variant, storage, bytes)) return cudaErrorInvalidValue;
  int device = -1;
  auto status = cudaGetDevice(&device);
  if (status != cudaSuccess) return status;
  if (device != variant->device) return cudaErrorInvalidDevice;
  return ds41rt_v41_initialize_scratch_storage_async(storage, variant->info.scratch_bytes,
      variant->scratch_offsets[37], variant->scratch_offsets[40],
      variant->info.experts, stream);
}

extern "C" int32_t ds41rt_v41_expert_info(int32_t capacity, ds41rt_v41_expert_info_t* out) {
  auto* variant = by_capacity(capacity);
  if (!variant || !out) return cudaErrorInvalidValue;
  *out = variant->info;
  return cudaSuccess;
}

extern "C" int32_t ds41rt_v41_expert_output_kind(int32_t capacity, uint32_t* out) {
  auto* variant = by_capacity(capacity);
  if (!variant || !out) return cudaErrorInvalidValue;
  *out = DS41RT_V41_OUTPUT_KIND(capacity);
  return cudaSuccess;
}

namespace {
// Same contract as v41_fp8.cc: the AOT export pins compute capability and SM count of the
// GPU that ran export_b12x_v41_experts_aot.py. Name the mismatch instead of returning a
// bare cudaErrorInvalidDevice (101).
int32_t reject_expert_device(const char* what, int device, int major, int minor, int sms) {
  std::fprintf(stderr,
               "ds41rt: %s AOT kernels were exported for compute 12.%d with %d SMs, but device "
               "%d is compute %d.%d with %d SMs; rebuild the AOT export on the target GPU "
               "(cudaErrorInvalidDevice)\n",
               what, int(DS41RT_V41_CC_MINOR), int(DS41RT_V41_SMS), device, major, minor, sms);
  return cudaErrorInvalidDevice;
}
}  // namespace
extern "C" int32_t ds41rt_v41_expert_initialize(int32_t capacity, void** out) {
  if (!out) return cudaErrorInvalidValue;
  *out = nullptr;
  auto* variant = by_capacity(capacity);
  if (!variant) return cudaErrorInvalidValue;
  int device = -1, major = 0, minor = 0, sms = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) return status;
  status = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  if (status != cudaSuccess) return status;
  status = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
  if (status != cudaSuccess) return status;
  status = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  if (status != cudaSuccess) return status;
  if (major != 12 || minor != DS41RT_V41_CC_MINOR || sms != DS41RT_V41_SMS)
    return reject_expert_device("v41 expert", device, major, minor, sms);
  std::lock_guard<std::mutex> lock(initialization_mutex);
  if (variant->device >= 0) {
    if (variant->device != device) return cudaErrorInvalidDevice;
    *out = variant;
    return cudaSuccess;
  }
  auto* module_owner = variant;
#ifdef DS41RT_V41_TP2_EXPERTS
  // Generated launch symbols are shared process-wide. Keep one library per
  // exported variant and configure it on both devices; handles remain distinct.
  for (auto& entry : variants)
    if (entry.info.capacity_rows == variant->info.capacity_rows) { module_owner = &entry; break; }
#endif
  auto* library_ptr = &module_owner->library;
  const bool existing = module_owner->library != nullptr;
  void* init_args[] = {&library_ptr, &status};
  if (!existing) variant->initialize(init_args);
  if (status != cudaSuccess) {
    if (!existing) {
      if (module_owner->library) cudaLibraryUnload(module_owner->library);
      module_owner->library = nullptr;
    }
    return status;
  }
  void* load_args[] = {&library_ptr, &device, &status};
  variant->load(load_args);
  if (status != cudaSuccess) {
    if (!existing) {
      cudaLibraryUnload(module_owner->library);
      module_owner->library = nullptr;
    }
    return status;
  }
  variant->device = device;
  *out = variant;
  return cudaSuccess;
}

extern "C" int32_t ds41rt_v41_expert_launch(void* kernel, const ds41rt_v41_expert_launch_t* args) {
  Variant* variant = by_handle(kernel);
  if (!variant || !args || variant->device < 0) return cudaErrorInvalidValue;
  const auto& info = variant->info;
  if (args->num_tokens <= 0 || static_cast<uint32_t>(args->num_tokens) > info.capacity_rows ||
      args->scatter_rows != args->num_tokens * static_cast<int32_t>(info.topk) ||
      args->max_rows != info.max_rows || args->rows_padded != info.rows_padded ||
      args->max_tasks != info.max_tasks || args->max_phys_tiles != info.max_phys_tiles ||
      args->max_active_clusters <= 0 || args->max_active_clusters > 2 * DS41RT_V41_SMS)
    return cudaErrorInvalidValue;
  for (auto* pointer : args->tensors) if (!pointer) return cudaErrorInvalidValue;
  int device = -1;
  auto status = cudaGetDevice(&device);
  if (status != cudaSuccess) return status;
  if (device != variant->device) return cudaErrorInvalidDevice;
  void* pointers[44];
  std::copy(args->tensors, args->tensors + 44, pointers);
  int32_t scalars[] = {args->num_tokens, args->max_rows, args->scatter_rows,
    args->rows_padded, args->max_tasks, args->max_phys_tiles, args->max_active_clusters};
  void* stream = args->stream;
  int32_t result = 0;
  void* parameters[53];
  for (int i = 0; i < 44; ++i) parameters[i] = &pointers[i];
  for (int i = 0; i < 7; ++i) parameters[44 + i] = &scalars[i];
  parameters[51] = &stream;
  parameters[52] = &result;
  variant->launch(parameters, 53);
  return result;
}

#ifndef DS41RT_V41_LOCAL_EXPERTS
namespace {
struct InputQuantModule {
  cudaLibrary_t library = nullptr;
  int device = -1;
  ~InputQuantModule() { if (library) cudaLibraryUnload(library); }
} input_quant, peer_input_quant;
}
extern "C" int32_t ds41rt_v41_expert_input_quant_initialize(void** out) {
  if (!out) return cudaErrorInvalidValue;
  *out = nullptr;
  int device, major, minor, sms;
  auto status = cudaGetDevice(&device); if (status) return status;
  status = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device); if (status) return status;
  status = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device); if (status) return status;
  status = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device); if (status) return status;
  if (major != 12 || minor != DS41RT_V41_CC_MINOR || sms != DS41RT_V41_SMS)
    return reject_expert_device("v41 expert input-quant", device, major, minor, sms);
  std::lock_guard<std::mutex> lock(initialization_mutex);
  auto* owner = input_quant.device == device ? &input_quant :
      (peer_input_quant.device == device ? &peer_input_quant :
       (input_quant.device < 0 ? &input_quant : (peer_input_quant.device < 0 ? &peer_input_quant : nullptr)));
  if (!owner) return cudaErrorInvalidDevice;
  if (owner->device == device) { *out=owner;return cudaSuccess; }
  // One canonical library owns process-global generated launch symbols.
  const bool existing = input_quant.library != nullptr;
  auto* library = &input_quant.library;
  void* init[] = {&library, &status};
  if (!existing) _mlir_ds41rt_v41_expert_input_quant_cuda_init(init);
  if (!status) {
    void* load[] = {&library, &device, &status};
    _mlir_ds41rt_v41_expert_input_quant_cuda_load_to_device(load);
  }
  if (status) {
    if (!existing) { if (input_quant.library) cudaLibraryUnload(input_quant.library); input_quant.library = nullptr; }
    return status;
  }
  owner->device = device;
  *out = owner;
  return cudaSuccess;
}
extern "C" int32_t ds41rt_v41_expert_input_quantize_async(void* kernel,
    const uint16_t* input, uint8_t* output, uint32_t rows, void* stream) {
  auto* owner = kernel == &input_quant ? &input_quant : (kernel == &peer_input_quant ? &peer_input_quant : nullptr);
  if (!owner || owner->device < 0 || !input || !output ||
      rows == 0 || rows > 4096) return cudaErrorInvalidValue;
  const auto a = reinterpret_cast<uintptr_t>(input), b = reinterpret_cast<uintptr_t>(output);
  const uint64_t an = uint64_t(rows) * 10240, bn = uint64_t(rows) * 5280;
  if (a % 16 || b % 16 || a > UINTPTR_MAX-an || b > UINTPTR_MAX-bn ||
      (a <= b ? b-a < an : a-b < bn)) return cudaErrorInvalidValue;
  int device;
  auto status = cudaGetDevice(&device); if (status) return status;
  if (device != owner->device) return cudaErrorInvalidDevice;
  void* source = const_cast<uint16_t*>(input);
  void* values = output;
  void* scales = output + 5120;
  void* unused_mma = output; // wire specialization does not write MMA scales
  int32_t m = rows, grid = ds41rt_v41_input_quant_grids[rows-1];
  int32_t result = 0;
  void* args[] = {&source, &values, &scales, &unused_mma, &m, &grid, &stream, &result};
  DS41RT_V41_INPUT_QUANT_ENTRY(args, 8);
  return result;
}

#endif // DS41RT_V41_LOCAL_EXPERTS
