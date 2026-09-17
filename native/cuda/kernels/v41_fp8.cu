#include <cuda_runtime.h>
#include <stdint.h>
#include <cuda_bf16.h>
#include "ds41rt_v41_fp8.h"
namespace {
__global__ void pack_scales(const uint8_t* source, uint8_t* output, uint64_t bytes, uint32_t scale_k) {
  const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= bytes) return;
  const uint64_t k4 = i % 4, row4 = (i / 4) % 4;
  const uint64_t tile = i / 512, nk = tile / (scale_k / 4), kk = tile % (scale_k / 4);
  output[i] = source[(nk * 4 + row4) * scale_k + kk * 4 + k4];
}
__global__ void alpha_one(float* alpha) { *alpha = 1.0f; }
__global__ void shared_swiglu(const __nv_bfloat16* gate, const __nv_bfloat16* up,
    __nv_bfloat16* output, uint64_t count) {
  const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const float g = fminf(__bfloat162float(gate[i]), 10.0f);
  const float u = fminf(fmaxf(__bfloat162float(up[i]), -10.0f), 10.0f);
  output[i] = __float2bfloat16_rn((g / (1.0f + expf(-g))) * u);
}
}
static int32_t shared_swiglu_width(const uint16_t* gate, const uint16_t* up,
    uint16_t* output, int32_t rows, uint32_t width, void* stream) {
  if (rows < 1 || rows > 4096) return cudaErrorInvalidValue;
  const uint64_t count = uint64_t(rows) * width, bytes = count * 2;
  const uintptr_t g = reinterpret_cast<uintptr_t>(gate), u = reinterpret_cast<uintptr_t>(up),
      o = reinterpret_cast<uintptr_t>(output);
  if (!g || !u || !o || (g | u | o) % 2 || g > UINTPTR_MAX - bytes ||
      u > UINTPTR_MAX - bytes || o > UINTPTR_MAX - bytes ||
      (o < g + bytes && g < o + bytes) || (o < u + bytes && u < o + bytes))
    return cudaErrorInvalidValue;
  shared_swiglu<<<(count + 255) / 256, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(gate), reinterpret_cast<const __nv_bfloat16*>(up),
      reinterpret_cast<__nv_bfloat16*>(output), count);
  return cudaGetLastError();
}
extern "C" int32_t ds41rt_v41_shared_swiglu(const uint16_t* gate, const uint16_t* up,
    uint16_t* output, int32_t rows, void* stream) {
  return shared_swiglu_width(gate, up, output, rows, 2304, stream);
}
extern "C" int32_t ds41rt_v41_shared_tp2_swiglu(const uint16_t* gate, const uint16_t* up,
    uint16_t* output, int32_t rows, void* stream) {
  return shared_swiglu_width(gate, up, output, rows, 1152, stream);
}
extern "C" int32_t ds41rt_v41_fp8_matrix_pack_scales(
    const uint8_t* source, uint8_t* destination, int32_t k, int32_t n, void* stream) {
  if (!((k == 6144 && n == 25600) || (k == 5120 && n == 2304) || (k == 2304 && n == 5120) ||
        (k == 5120 && n == 1152) || (k == 1152 && n == 5120) ||
        (k == 1280 && n == 16384) || (k == 8192 && n == 2560) ||
        (k == 15360 && n == 5120) || (k == 5120 && n == 1280) || (k == 1280 && n == 32768) ||
        (k == 1280 && n == 4096) || (k == 5120 && n == 512) || (k == 8192 && n == 5120) || (k == 32768 && n == 8192)))
    return cudaErrorInvalidValue;
  // WO-A has eight independent groups, each [1024,4096]. The checkpoint
  // stores groups consecutively, so ordinary packing applies to [8192,4096].
  if (k == 32768 && n == 8192) k = 4096;
  const uint64_t src_bytes = uint64_t(k) * n / 1024, dst_bytes = src_bytes * 32;
  auto a = reinterpret_cast<uintptr_t>(source), b = reinterpret_cast<uintptr_t>(destination);
  if (!a || !b || a > UINTPTR_MAX - src_bytes || b > UINTPTR_MAX - dst_bytes ||
      (a < b + dst_bytes && b < a + src_bytes)) return cudaErrorInvalidValue;
  pack_scales<<<(dst_bytes + 255) / 256, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      source, destination, dst_bytes, k / 32);
  return cudaGetLastError();
}
extern "C" int32_t ds41rt_v41_fp8_pack_scales(const uint8_t* source, uint8_t* destination, void* stream) {
  return ds41rt_v41_fp8_matrix_pack_scales(source, destination, 6144, 25600, stream);
}
extern "C" int32_t ds41rt_v41_fp8_initialize_storage(void* scratch, uint64_t bytes, float* alpha, void* stream) {
  auto status = cudaMemsetAsync(scratch, 0, bytes, reinterpret_cast<cudaStream_t>(stream));
  if (status != cudaSuccess) return status;
  alpha_one<<<1, 1, 0, reinterpret_cast<cudaStream_t>(stream)>>>(alpha);
  return cudaGetLastError();
}

namespace {
__global__ void reduce_splits(const float* partials, __nv_bfloat16* output, uint64_t elements, int slices) {
  const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= elements) return;
  float sum = 0;
  for (int split = 0; split < slices; ++split) sum += partials[uint64_t(split) * elements + i];
  output[i] = __float2bfloat16_rn(sum);
}
}
extern "C" int32_t ds41rt_v41_fp8_reduce_splits(const float* partials, uint16_t* output,
    int32_t rows, int32_t columns, int32_t slices, void* stream) {
  if (rows < 1 || rows > 4096 || columns < 1 || (slices != 2 && slices != 4))
    return cudaErrorInvalidValue;
  const uint64_t elements = uint64_t(rows) * columns;
  reduce_splits<<<(elements + 255) / 256,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(
      partials,reinterpret_cast<__nv_bfloat16*>(output),elements,slices);
  return cudaGetLastError();
}

namespace {
__global__ void grouped_output_rows(const uint16_t* input,uint16_t* output,uint64_t rows) {
  const uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=rows*8192)return;
  const uint64_t row=i/8192,group=(i%8192)/1024,col=i%1024;
  output[i]=input[(group*rows+row)*1024+col];
}
}
extern "C" int32_t ds41rt_v41_fp8_grouped_output(const uint16_t* input,uint16_t* output,
    int32_t rows,void* stream) {
  if(rows<1 || rows>4096)return cudaErrorInvalidValue;
  grouped_output_rows<<<(uint64_t(rows)*8192+255)/256,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(input,output,rows);
  return cudaGetLastError();
}
