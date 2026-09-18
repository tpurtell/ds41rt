//! NVFP4 (ModelOpt) block-scale plane swizzle.
//!
//! Kept independent of the Spark-only b12x AOT translation unit so the
//! coordinator can re-lay checkpoint scales for its resident experts.
#include "common.h"

#include <cstdint>
#include <limits>

namespace {

bool nvfp4_buffer_has_bytes(ds41rt_device_buffer_t buffer, size_t required) {
  return buffer.ptr != nullptr && buffer.bytes >= required;
}

// NVFP4 (ModelOpt) block-scale planes are consumed by the block-scaled MMA
// as 128x4 scale-factor atoms. The checkpoint stores a plain [rows, cols]
// E4M3 plane (cols = row width / 16), so the pack step re-swizzles it:
//   off(r, j) = (r / 128) * (cp * 128) + (j / 4) * 512
//             + (r % 32) * 16 + ((r / 32) % 4) * 4 + (j % 4)
// with rp = ceil128(rows), cp = ceil4(cols); padding bytes are zero.
__global__ void swizzle_nvfp4_scale_kernel(const uint8_t* __restrict__ source,
                                           size_t rows, size_t cols,
                                           uint8_t* __restrict__ destination,
                                           size_t rows_padded, size_t columns_padded) {
  const size_t total = rows_padded * columns_padded;
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  const size_t row = index / columns_padded;
  const size_t column = index % columns_padded;
  uint8_t value = 0;
  if (row < rows && column < cols) {
    value = source[row * cols + column];
  }
  const size_t offset = (row / 128) * (columns_padded * 128) + (column / 4) * 512 +
                        (row % 32) * 16 + ((row / 32) % 4) * 4 + (column % 4);
  destination[offset] = value;
}

}  // namespace

extern "C" ds41rt_status_t ds41rt_cuda_nvfp4_swizzle_scale_async(
    ds41rt_device_buffer_t source, ds41rt_device_buffer_t destination, size_t rows,
    size_t cols, void* cuda_stream) {
  if (rows == 0 || cols == 0) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  const size_t rows_padded = ((rows + 127) / 128) * 128;
  const size_t columns_padded = ((cols + 3) / 4) * 4;
  if (rows > std::numeric_limits<size_t>::max() / cols ||
      rows_padded > std::numeric_limits<size_t>::max() / columns_padded) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  const size_t source_bytes = rows * cols;
  const size_t destination_bytes = rows_padded * columns_padded;
  if (!nvfp4_buffer_has_bytes(source, source_bytes) ||
      !nvfp4_buffer_has_bytes(destination, destination_bytes)) {
    return DS41RT_STATUS_BUFFER_TOO_SMALL;
  }
  constexpr size_t threads = 256;
  const size_t blocks = (destination_bytes + threads - 1) / threads;
  swizzle_nvfp4_scale_kernel<<<static_cast<unsigned int>(blocks), threads, 0,
                               reinterpret_cast<cudaStream_t>(cuda_stream)>>>(
      static_cast<const uint8_t*>(source.ptr), rows, cols,
      static_cast<uint8_t*>(destination.ptr), rows_padded, columns_padded);
  return status_from_cuda(cudaGetLastError());
}
