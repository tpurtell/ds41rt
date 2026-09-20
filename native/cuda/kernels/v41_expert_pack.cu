#include "ds41rt_v41_experts.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace {
constexpr uint32_t hidden = 5120;

// N256/K128 lane-major representation consumed by b12x W4A8.
template<bool Gated, bool Scales>
__global__ void pack(const uint8_t* first, const uint8_t* second, uint8_t* output,
    uint32_t n, uint32_t k, uint32_t n_pad, uint64_t count) {
  const uint32_t k_tiles = (k + 127) / 128;
  constexpr uint32_t tile_elements = Scales ? 1024 : 4096;
  for (uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += uint64_t(gridDim.x) * blockDim.x) {
    const uint64_t tile = index / tile_elements;
    const uint32_t local = index % tile_elements;
    const uint32_t nt = tile / k_tiles, kt = tile % k_tiles;
    uint32_t row, col;
    if constexpr (Scales) {
      row = nt * 256 + local / 4;
      col = kt * 4 + local % 4;
    } else {
      const uint32_t n8i = local & 3;
      const uint32_t combined = (local >> 2) & 31;
      const uint32_t n8c = (local >> 7) & 7;
      const uint32_t k32 = local >> 10;
      row = nt * 256 + n8c * 32 + n8i * 8 + (combined >> 2);
      col = kt * 16 + k32 * 4 + (combined & 3);
    }
    const uint8_t* source = first;
    if constexpr (Gated) {
      if (row >= n_pad) { row -= n_pad; source = second; }
    }
    constexpr uint32_t divisor = Scales ? 32 : 8;
    const bool valid = row < n && col < k / divisor;
    if constexpr (Scales) {
      output[index] = valid ? source[uint64_t(row) * (k / divisor) + col] : 0;
    } else {
      reinterpret_cast<uint32_t*>(output)[index] = valid
        ? reinterpret_cast<const uint32_t*>(source)[uint64_t(row) * (k / divisor) + col] : 0;
    }
  }
}

bool overlaps(const void* a, uint64_t an, const void* b, uint64_t bn) {
  const auto av = reinterpret_cast<uintptr_t>(a), bv = reinterpret_cast<uintptr_t>(b);
  return av <= bv ? bv - av < an : av - bv < bn;
}
}

extern "C" int32_t ds41rt_v41_expert_packed_sizes(uint32_t intermediate,
    uint64_t bytes[4]) {
  // Official native extents: 576 (Spark TP4, padded to 640), 768 (Spark TP3),
  // 1152 (Spark/RTX TP2), 2304 (full). Every value is a multiple of 32 so the
  // K/32 scale axis is exact; the 128 padding below is storage-only.
  if (!bytes || intermediate % 32 != 0 ||
      (intermediate != 576 && intermediate != 768 && intermediate != 1152 && intermediate != 2304))
    return cudaErrorInvalidValue;
  const uint64_t padded = (intermediate + 127) / 128 * 128;
  bytes[0] = padded * hidden;
  bytes[1] = padded * hidden / 16;
  bytes[2] = hidden * padded / 2;
  bytes[3] = hidden * padded / 32;
  return cudaSuccess;
}

extern "C" int32_t ds41rt_v41_pack_expert_async(const uint8_t* const sources[6],
    uint8_t* const destinations[4], uint32_t intermediate, void* stream) {
  uint64_t sizes[4];
  if (!sources || !destinations || ds41rt_v41_expert_packed_sizes(intermediate, sizes))
    return cudaErrorInvalidValue;
  const uint64_t weight_bytes = uint64_t(intermediate) * hidden / 2;
  const uint64_t source_sizes[] = {weight_bytes, weight_bytes, weight_bytes,
    weight_bytes / 16, weight_bytes / 16, weight_bytes / 16};
  for (int i = 0; i < 6; ++i)
    if (!sources[i] || (i < 3 && reinterpret_cast<uintptr_t>(sources[i]) % 4))
      return cudaErrorInvalidValue;
  for (int i = 0; i < 4; ++i) {
    if (!destinations[i] || reinterpret_cast<uintptr_t>(destinations[i]) % 16)
      return cudaErrorInvalidValue;
    for (int j = 0; j < 6; ++j)
      if (overlaps(destinations[i], sizes[i], sources[j], source_sizes[j])) return cudaErrorInvalidValue;
    for (int j = 0; j < i; ++j)
      if (overlaps(destinations[i], sizes[i], destinations[j], sizes[j])) return cudaErrorInvalidValue;
  }
  const uint32_t padded = (intermediate + 127) / 128 * 128;
  auto cuda_stream = static_cast<cudaStream_t>(stream);
  // First half is up (W3), second half gate (W1), padded independently.
  pack<true, false><<<256, 256, 0, cuda_stream>>>(sources[1], sources[0], destinations[0],
      intermediate, hidden, padded, sizes[0] / 4);
  auto status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  pack<true, true><<<256, 256, 0, cuda_stream>>>(sources[4], sources[3], destinations[1],
      intermediate, hidden, padded, sizes[1]);
  status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  pack<false, false><<<256, 256, 0, cuda_stream>>>(sources[2], nullptr, destinations[2],
      hidden, intermediate, 0, sizes[2] / 4);
  status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  pack<false, true><<<256, 256, 0, cuda_stream>>>(sources[5], nullptr, destinations[3],
      hidden, intermediate, 0, sizes[3]);
  return cudaGetLastError();
}
