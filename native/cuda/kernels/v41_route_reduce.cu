#include "ds41rt_v41_experts.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace {
constexpr uint64_t hidden = 5120;
__global__ void initialize_global_scales(float* input, float* down, uint32_t count) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) { input[index] = 1.0f; down[index] = 1.0f; }
}
template<int Ranks, int TopK>
__global__ void reduce_routes(const float* p0, const float* p1,
    const float* p2, const float* p3, const __nv_bfloat16* shared,
    __nv_bfloat16* output, uint64_t count) {
  const float* planes[] = {p0, p1, p2, p3};
  for (uint64_t offset = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       offset < count; offset += uint64_t(gridDim.x) * blockDim.x) {
    const uint64_t row = offset / hidden, col = offset % hidden;
    float total = 0;
#pragma unroll
    for (int route = 0; route < TopK; ++route) {
      const uint64_t index = (row * TopK + route) * hidden + col;
      float value = planes[0][index];
#pragma unroll
      for (int rank = 1; rank < Ranks; ++rank)
        value = __fadd_rn(value, planes[rank][index]);
      total = __fadd_rn(total, __bfloat162float(__float2bfloat16_rn(value)));
    }
    if (shared) total = __fadd_rn(total, __bfloat162float(shared[offset]));
    output[offset] = __float2bfloat16_rn(total);
  }
}

bool overlaps(const void* a, uint64_t a_bytes, const void* b, uint64_t b_bytes) {
  const auto av = reinterpret_cast<uintptr_t>(a);
  const auto bv = reinterpret_cast<uintptr_t>(b);
  // Subtraction avoids overflow at the upper end of the address space.
  return av <= bv ? bv - av < a_bytes : av - bv < b_bytes;
}

template<int Routes>
__global__ void compact_routes(const float* routes, __nv_bfloat16* output,
    uint64_t count) {
  for (uint64_t offset = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       offset < count; offset += uint64_t(gridDim.x) * blockDim.x) {
    const uint64_t base = (offset / hidden) * Routes * hidden + offset % hidden;
    float value = routes[base];
#pragma unroll
    for (int route = 1; route < Routes; ++route)
      value = __fadd_rn(value, routes[base + route * hidden]);
    output[offset] = __float2bfloat16_rn(value);
  }
}

__global__ void reduce_compact(const __nv_bfloat16* p0,
    const __nv_bfloat16* p1, const __nv_bfloat16* p2,
    const __nv_bfloat16* p3, const __nv_bfloat16* shared,
    __nv_bfloat16* output, uint64_t count) {
  for (uint64_t offset = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       offset < count; offset += uint64_t(gridDim.x) * blockDim.x) {
    float value = __bfloat162float(p0[offset]);
    value = __fadd_rn(value, __bfloat162float(p1[offset]));
    value = __fadd_rn(value, __bfloat162float(p2[offset]));
    value = __fadd_rn(value, __bfloat162float(p3[offset]));
    if (shared) value = __fadd_rn(value, __bfloat162float(shared[offset]));
    output[offset] = __float2bfloat16_rn(value);
  }
}
}

extern "C" int32_t ds41rt_v41_compact_routes_bf16_async(const float* routes,
    uint16_t* output, uint32_t rows, void* stream) {
  const uint64_t count = uint64_t(rows) * hidden;
  if (!rows || !routes || !output || reinterpret_cast<uintptr_t>(routes) % 4 ||
      reinterpret_cast<uintptr_t>(output) % 2 ||
      overlaps(routes, count * 6 * 4, output, count * 2))
    return cudaErrorInvalidValue;
  const unsigned blocks = static_cast<unsigned>(count / 256 < 4096 ? count / 256 : 4096);
  compact_routes<6><<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      routes, reinterpret_cast<__nv_bfloat16*>(output), count);
  return cudaGetLastError();
}

extern "C" int32_t ds41rt_v41_compact_tokens_bf16_async(const float* tokens,
    uint16_t* output, uint32_t rows, void* stream) {
  const uint64_t count = uint64_t(rows) * hidden;
  if (!rows || rows > 4096 || !tokens || !output ||
      reinterpret_cast<uintptr_t>(tokens) % 4 || reinterpret_cast<uintptr_t>(output) % 2 ||
      reinterpret_cast<uintptr_t>(tokens) > UINTPTR_MAX - count * 4 ||
      reinterpret_cast<uintptr_t>(output) > UINTPTR_MAX - count * 2 ||
      overlaps(tokens, count * 4, output, count * 2)) return cudaErrorInvalidValue;
  const unsigned blocks = static_cast<unsigned>(count / 256 < 4096 ? count / 256 : 4096);
  compact_routes<1><<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      tokens, reinterpret_cast<__nv_bfloat16*>(output), count);
  return cudaGetLastError();
}

extern "C" int32_t ds41rt_v41_reduce_compact_bf16_async(
    const uint16_t* const planes[4], const uint16_t* shared, uint16_t* output,
    uint32_t rows, void* stream) {
  const uint64_t count = uint64_t(rows) * hidden;
  if (!rows || !planes || !output || reinterpret_cast<uintptr_t>(output) % 2)
    return cudaErrorInvalidValue;
  for (int rank = 0; rank < 4; ++rank)
    if (!planes[rank] || reinterpret_cast<uintptr_t>(planes[rank]) % 2 ||
        overlaps(planes[rank], count * 2, output, count * 2))
      return cudaErrorInvalidValue;
  if (shared && (reinterpret_cast<uintptr_t>(shared) % 2 ||
      (shared != output && overlaps(shared, count * 2, output, count * 2))))
    return cudaErrorInvalidValue;
  const unsigned blocks = static_cast<unsigned>(count / 256 < 4096 ? count / 256 : 4096);
  reduce_compact<<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(planes[0]),
      reinterpret_cast<const __nv_bfloat16*>(planes[1]),
      reinterpret_cast<const __nv_bfloat16*>(planes[2]),
      reinterpret_cast<const __nv_bfloat16*>(planes[3]),
      reinterpret_cast<const __nv_bfloat16*>(shared),
      reinterpret_cast<__nv_bfloat16*>(output), count);
  return cudaGetLastError();
}

extern "C" int32_t ds41rt_v41_initialize_scratch_storage_async(void* storage,
    uint64_t bytes, uint64_t input_offset, uint64_t down_offset,
    uint32_t experts, void* stream) {
  auto cuda_stream = static_cast<cudaStream_t>(stream);
  auto status = cudaMemsetAsync(storage, 0, bytes, cuda_stream);
  if (status != cudaSuccess) return status;
  auto* base = static_cast<char*>(storage);
  initialize_global_scales<<<(experts + 255) / 256, 256, 0, cuda_stream>>>(
      reinterpret_cast<float*>(base + input_offset),
      reinterpret_cast<float*>(base + down_offset), experts);
  return cudaGetLastError();
}

extern "C" int32_t ds41rt_v41_reduce_routes_async(const float* const planes[4],
    const uint16_t* shared, uint16_t* output, uint32_t rows,
    uint32_t ranks, uint32_t topk, void* stream) {
  if (!planes || !output || !rows ||
      !(((ranks == 1 || ranks == 2) && topk == 3) || (ranks == 4 && topk == 6)))
    return cudaErrorInvalidValue;
  const uint64_t count = uint64_t(rows) * hidden;
  for (uint32_t rank = 0; rank < 4; ++rank) {
    if (rank >= ranks) {
      if (planes[rank]) return cudaErrorInvalidValue;
    } else if (!planes[rank] || reinterpret_cast<uintptr_t>(planes[rank]) % 4 ||
        overlaps(output, count * 2, planes[rank], count * topk * 4)) {
      return cudaErrorInvalidValue;
    }
  }
  if (reinterpret_cast<uintptr_t>(output) % 2 ||
      (shared && (reinterpret_cast<uintptr_t>(shared) % 2 ||
        (shared != output && overlaps(output, count * 2, shared, count * 2)))))
    return cudaErrorInvalidValue;
  const unsigned blocks = static_cast<unsigned>(count / 256 < 4096 ? count / 256 : 4096);
  auto cuda_stream = static_cast<cudaStream_t>(stream);
  if (ranks == 1)
    reduce_routes<1, 3><<<blocks, 256, 0, cuda_stream>>>(planes[0], nullptr,
        nullptr, nullptr, reinterpret_cast<const __nv_bfloat16*>(shared),
        reinterpret_cast<__nv_bfloat16*>(output), count);
  else if (ranks == 2)
    reduce_routes<2, 3><<<blocks, 256, 0, cuda_stream>>>(planes[0], planes[1],
        nullptr, nullptr, reinterpret_cast<const __nv_bfloat16*>(shared),
        reinterpret_cast<__nv_bfloat16*>(output), count);
  else
    reduce_routes<4, 6><<<blocks, 256, 0, cuda_stream>>>(planes[0], planes[1],
        planes[2], planes[3], reinterpret_cast<const __nv_bfloat16*>(shared),
        reinterpret_cast<__nv_bfloat16*>(output), count);
  return cudaGetLastError();
}

namespace {
__global__ void add_tp2_shared(const __nv_bfloat16* a, const __nv_bfloat16* b,
    __nv_bfloat16* output, uint64_t count) {
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += uint64_t(gridDim.x) * blockDim.x)
    output[i] = __float2bfloat16_rn(__fadd_rn(__bfloat162float(a[i]), __bfloat162float(b[i])));
}
template<int Routes>
__global__ void reduce_tp2(const float* rank0, const float* rank1,
    __nv_bfloat16* output, uint64_t count) {
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += uint64_t(gridDim.x) * blockDim.x) {
    const uint64_t base = (i / hidden) * Routes * hidden + i % hidden;
    float value = __fadd_rn(rank0[base], rank1[base]);
#pragma unroll
    for (int route = 1; route < Routes; ++route) {
      const auto index = base + route * hidden;
      value = __fadd_rn(value, __fadd_rn(rank0[index], rank1[index]));
    }
    output[i] = __float2bfloat16_rn(value);
  }
}

template<int Routes>
__global__ void finish_local(const float* routed, const __nv_bfloat16* shared,
    __nv_bfloat16* output, uint64_t count) {
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += uint64_t(gridDim.x) * blockDim.x) {
    const uint64_t base = (i / hidden) * Routes * hidden + i % hidden;
    float value = routed[base];
#pragma unroll
    for (int route = 1; route < Routes; ++route)
      value = __fadd_rn(value, routed[base + route * hidden]);
    // Preserve the compact routed-output boundary before adding shared FFN.
    value = __bfloat162float(__float2bfloat16_rn(value));
    if (shared) value = __fadd_rn(value, __bfloat162float(shared[i]));
    output[i] = __float2bfloat16_rn(value);
  }
}
}
extern "C" int32_t ds41rt_v41_add_tp2_shared_async(const uint16_t* a,
    const uint16_t* b, uint16_t* output, size_t count, void* stream) {
  if (!count || count > uint64_t(4096) * hidden || !a || !b || !output ||
      reinterpret_cast<uintptr_t>(a) % 2 || reinterpret_cast<uintptr_t>(b) % 2 ||
      reinterpret_cast<uintptr_t>(output) % 2) return cudaErrorInvalidValue;
  const auto bytes = count * 2;
  if (reinterpret_cast<uintptr_t>(a) > UINTPTR_MAX - bytes ||
      reinterpret_cast<uintptr_t>(b) > UINTPTR_MAX - bytes ||
      reinterpret_cast<uintptr_t>(output) > UINTPTR_MAX - bytes ||
      overlaps(a, bytes, output, bytes) || overlaps(b, bytes, output, bytes))
    return cudaErrorInvalidValue;
  const auto blocks = static_cast<unsigned>((count + 255) / 256 < 4096 ? (count + 255) / 256 : 4096);
  add_tp2_shared<<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(a), reinterpret_cast<const __nv_bfloat16*>(b),
      reinterpret_cast<__nv_bfloat16*>(output), count);
  return cudaGetLastError();
}
extern "C" int32_t ds41rt_v41_reduce_tp2_experts_async(const float* rank0,
    const float* rank1, uint16_t* output, uint32_t rows, uint32_t token_sums,
    void* stream) {
  if (!rows || rows > 4096 || token_sums > 1 || !rank0 || !rank1 || !output ||
      reinterpret_cast<uintptr_t>(rank0) % 4 || reinterpret_cast<uintptr_t>(rank1) % 4 ||
      reinterpret_cast<uintptr_t>(output) % 2) return cudaErrorInvalidValue;
  const uint64_t count = uint64_t(rows) * hidden;
  const uint64_t input_bytes = count * (token_sums ? 1 : 6) * 4;
  const uint64_t output_bytes = count * 2;
  if (reinterpret_cast<uintptr_t>(rank0) > UINTPTR_MAX - input_bytes ||
      reinterpret_cast<uintptr_t>(rank1) > UINTPTR_MAX - input_bytes ||
      reinterpret_cast<uintptr_t>(output) > UINTPTR_MAX - output_bytes ||
      overlaps(rank0, input_bytes, output, output_bytes) ||
      overlaps(rank1, input_bytes, output, output_bytes)) return cudaErrorInvalidValue;
  const unsigned blocks = static_cast<unsigned>(count / 256 < 4096 ? count / 256 : 4096);
  if (token_sums)
    reduce_tp2<1><<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(rank0, rank1,
        reinterpret_cast<__nv_bfloat16*>(output), count);
  else
    reduce_tp2<6><<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(rank0, rank1,
        reinterpret_cast<__nv_bfloat16*>(output), count);
  return cudaGetLastError();
}

extern "C" int32_t ds41rt_v41_finish_local_experts_async(const float* routed,
    const uint16_t* shared, uint16_t* output, uint32_t rows,
    uint32_t token_sums, void* stream) {
  const uint64_t count = uint64_t(rows) * hidden;
  const uint64_t input_bytes = count * (token_sums ? 1 : 6) * 4;
  const uint64_t output_bytes = count * 2;
  if (!rows || rows > 4096 || token_sums > 1 || !routed || !output ||
      reinterpret_cast<uintptr_t>(routed) % 4 || reinterpret_cast<uintptr_t>(output) % 2 ||
      reinterpret_cast<uintptr_t>(routed) > UINTPTR_MAX - input_bytes ||
      reinterpret_cast<uintptr_t>(output) > UINTPTR_MAX - output_bytes ||
      overlaps(routed, input_bytes, output, output_bytes) ||
      (shared && (reinterpret_cast<uintptr_t>(shared) % 2 ||
        reinterpret_cast<uintptr_t>(shared) > UINTPTR_MAX - output_bytes ||
        (shared != output && overlaps(shared, output_bytes, output, output_bytes)))))
    return cudaErrorInvalidValue;
  const unsigned blocks = static_cast<unsigned>(count / 256 < 4096 ? count / 256 : 4096);
  if (token_sums)
    finish_local<1><<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(routed,
        reinterpret_cast<const __nv_bfloat16*>(shared), reinterpret_cast<__nv_bfloat16*>(output), count);
  else
    finish_local<6><<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(routed,
        reinterpret_cast<const __nv_bfloat16*>(shared), reinterpret_cast<__nv_bfloat16*>(output), count);
  return cudaGetLastError();
}
