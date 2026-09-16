#include "ds41rt_v41_exl3_wire.h"
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <new>

namespace {
struct Context { int device; };
__global__ void decode_wire(const uint8_t* input, __nv_bfloat16* output, uint64_t elements) {
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < elements; i += uint64_t(gridDim.x) * blockDim.x) {
    const uint64_t row = i / 5120, col = i % 5120;
    __nv_fp8_e4m3 value; value.__x = input[row * 5280 + col];
    __nv_fp8_e8m0 scale; scale.__x = input[row * 5280 + 5120 + col / 32];
    output[i] = __float2bfloat16_rn(__fmul_rn(static_cast<float>(value), static_cast<float>(scale)));
  }
}
}

extern "C" int32_t ds41rt_v41_exl3_wire_initialize(void** out) {
  if (!out) return cudaErrorInvalidValue;
  *out = nullptr;
  auto* context = new(std::nothrow) Context;
  if (!context) return cudaErrorMemoryAllocation;
  auto status = cudaGetDevice(&context->device);
  cudaFuncAttributes attributes{};
  if (status == cudaSuccess) status = cudaFuncGetAttributes(&attributes, decode_wire);
  if (status != cudaSuccess) { delete context; return status; }
  *out = context; return cudaSuccess;
}
extern "C" void ds41rt_v41_exl3_wire_destroy(void* handle) {
  delete static_cast<Context*>(handle);
}
extern "C" int32_t ds41rt_v41_exl3_wire_decode(void* handle, const uint8_t* input,
    uint64_t input_bytes, uint16_t* output, uint64_t output_bytes, uint32_t rows, void* stream) {
  if (!handle || !input || !output || rows < 1 || rows > 4096) return cudaErrorInvalidValue;
  const uint64_t in_size = uint64_t(rows) * 5280, out_size = uint64_t(rows) * 5120 * 2;
  const auto in = reinterpret_cast<uintptr_t>(input), out = reinterpret_cast<uintptr_t>(output);
  if (in % 16 || out % 16 || input_bytes < in_size || output_bytes < out_size ||
      in > UINTPTR_MAX - in_size || out > UINTPTR_MAX - out_size ||
      (in < out + out_size && out < in + in_size)) return cudaErrorInvalidValue;
  int device = -1; auto status = cudaGetDevice(&device);
  if (status != cudaSuccess) return status;
  if (device != static_cast<Context*>(handle)->device) return cudaErrorInvalidDevice;
  const uint64_t elements = uint64_t(rows) * 5120;
  const unsigned blocks = static_cast<unsigned>((elements + 255) / 256);
  decode_wire<<<blocks < 1024 ? blocks : 1024, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      input, reinterpret_cast<__nv_bfloat16*>(output), elements);
  return cudaGetLastError();
}
