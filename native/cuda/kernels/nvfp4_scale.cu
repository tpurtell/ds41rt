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

namespace {
// Load-time padding only: preserve every checkpoint byte and insert exact
// zeros. FC1's gate half must move independently of its up half. Scale planes
// are swizzled directly into their final resident layout.
template <bool Fc1, bool Scale>
__global__ void pad_nvfp4_plane(const uint8_t* source, uint8_t* destination,
                               size_t source_n, size_t kernel_n) {
  constexpr size_t hidden = 5120;
  constexpr size_t divisor = Scale ? 16 : 2;
  const size_t rows = Fc1 ? 2 * kernel_n : hidden;
  const size_t cols = (Fc1 ? hidden : kernel_n) / divisor;
  const size_t source_cols = (Fc1 ? hidden : source_n) / divisor;
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= rows * cols) return;
  const size_t row = index / cols;
  const size_t col = index % cols;
  const size_t half_row = Fc1 ? row % kernel_n : row;
  const size_t source_row = Fc1 ? half_row + (row / kernel_n) * source_n : row;
  const bool valid = (!Fc1 || half_row < source_n) && col < source_cols;
  const size_t offset = Scale
      ? (row / 128) * (cols * 128) + (col / 4) * 512 +
            (row % 32) * 16 + ((row / 32) % 4) * 4 + col % 4
      : index;
  destination[offset] = valid ? source[source_row * source_cols + col] : 0;
}
}  // namespace

extern "C" ds41rt_status_t ds41rt_cuda_nvfp4_pad_expert_async(
    const ds41rt_device_buffer_t* sources, const ds41rt_device_buffer_t* destinations,
    size_t source_n, size_t kernel_n, void* cuda_stream) {
  // Bounded model geometry also keeps all byte/grid arithmetic representable.
  if (!sources || !destinations || source_n == 0 || source_n % 64 != 0 ||
      kernel_n < source_n || kernel_n % 128 != 0 || kernel_n > 8192) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  const size_t source_bytes[] = {2 * source_n * 2560, 2 * source_n * 320,
                                 5120 * source_n / 2, 5120 * source_n / 16};
  const size_t destination_bytes[] = {2 * kernel_n * 2560, 2 * kernel_n * 320,
                                      5120 * kernel_n / 2, 5120 * kernel_n / 16};
  for (int i = 0; i < 4; ++i) {
    if (!nvfp4_buffer_has_bytes(sources[i], source_bytes[i]) ||
        !nvfp4_buffer_has_bytes(destinations[i], destination_bytes[i])) {
      return DS41RT_STATUS_BUFFER_TOO_SMALL;
    }
  }
  auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
#define PAD_PLANE(index, fc1, scale) \
  pad_nvfp4_plane<fc1, scale><<<static_cast<unsigned>((destination_bytes[index] + 255) / 256), 256, 0, stream>>>( \
      static_cast<const uint8_t*>(sources[index].ptr), \
      static_cast<uint8_t*>(destinations[index].ptr), source_n, kernel_n); \
  if (auto error = cudaGetLastError(); error != cudaSuccess) return status_from_cuda(error)
  PAD_PLANE(0, true, false);
  PAD_PLANE(1, true, true);
  PAD_PLANE(2, false, false);
  PAD_PLANE(3, false, true);
#undef PAD_PLANE
  return DS41RT_STATUS_OK;
}

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
