#include "common.h"

#include "moe_tp4_w4a16_decode_m1.h"
#include "moe_tp4_w4a16_decode_m1_fused_sum.h"
#include "moe_tp4_w4a16_m1_parity_m2_topk8.h"
#include "moe_tp4_w4a16_m1_parity_m3_topk8.h"
#include "moe_tp4_w4a16_m1_parity_m4_topk8.h"
#include "moe_tp4_w4a16_m1_parity_m5_topk8.h"
#include "moe_tp4_w4a16_m1_parity_m6_topk8.h"
#include "moe_tp4_w4a16_m1_parity_m7_topk8.h"
#include "moe_tp4_w4a16_m1_parity_m8_topk8.h"
#include "moe_tp4_w4a16_m1_parity_grouped_m2_topk8.h"
#include "moe_tp4_w4a16_m1_parity_grouped_m3_topk8.h"
#include "moe_tp4_w4a16_m1_parity_grouped_m4_topk8.h"
#include "moe_tp4_w4a16_m1_parity_grouped_m5_topk8.h"
#include "moe_tp4_w4a16_m1_parity_grouped_m6_topk8.h"
#include "moe_tp4_w4a16_m1_parity_grouped_m7_topk8.h"
#include "moe_tp4_w4a16_m1_parity_grouped_m8_topk8.h"
#include "moe_tp4_w4a16_m1_parity_grouped_wide_m2_topk8.h"
#include "moe_tp4_w4a16_m1_parity_grouped_wide_m3_topk8.h"
#include "moe_tp4_w4a16_m1_parity_grouped_wide_m4_topk8.h"
#include "moe_tp4_w4a16_m1_parity_grouped_wide_m5_topk8.h"
#include "moe_tp4_w4a16_m1_parity_grouped_wide_m6_topk8.h"
#include "moe_tp4_w4a16_m1_parity_grouped_wide_m7_topk8.h"
#include "moe_tp4_w4a16_m1_parity_grouped_wide_m8_topk8.h"
#include "moe_tp4_w4a16_prefill_m2_topk8.h"
#include "moe_tp4_w4a16_prefill_m4_topk8.h"
#include "moe_tp4_w4a16_prefill_m8_topk8.h"
#include "moe_tp4_w4a16_prefill_m16_topk8.h"
#include "moe_tp4_w4a16_prefill_m32_topk8.h"
#include "moe_tp4_w4a16_prefill_m64_topk8.h"
#include "moe_tp4_w4a16_prefill_m128_topk8.h"
#include "moe_tp4_w4a16_prefill_m256_topk8.h"
#include "moe_tp4_w4a16_prefill_m1024_topk8.h"
#include "moe_tp4_w4a16_prefill_m2048_topk8.h"
#include "moe_tp4_w4a16_prefill_m512_topk8.h"
#include "moe_tp4_w4a16_top1_m1.h"
#include "moe_tp4_w4a16_top1_m128.h"
#include "moe_tp4_w4a16_top1_m16.h"
#include "moe_tp4_w4a16_top1_m2.h"
#include "moe_tp4_w4a16_top1_m256.h"
#include "moe_tp4_w4a16_top1_m32.h"
#include "moe_tp4_w4a16_top1_m4.h"
#include "moe_tp4_w4a16_top1_m64.h"
#include "moe_tp4_w4a16_top1_m8.h"
#include "b12x_spark_moe_aot_config.h"
#include "b12x_spark_w4a16_m1_parity_aot_config.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdlib>
#include <mutex>

namespace {

constexpr size_t kB12xW4a16MaxRows = 2048;
constexpr size_t kB12xHidden = 6144;
constexpr size_t kB12xTp4Intermediate = 512;
constexpr size_t kB12xOutput = 6144;
constexpr size_t kB12xExperts = 256;
constexpr size_t kB12xTopK = 8;
ds41rt_b12x_moe_tp4_w4a16_decode_m1_Kernel_Module_t moe_tp4_w4a16_decode_m1_module;
ds41rt_b12x_moe_tp4_w4a16_decode_m1_fused_sum_Kernel_Module_t
    moe_tp4_w4a16_decode_m1_fused_sum_module;
#define DS41RT_DEFINE_W4A16_M1_PARITY_MODULE(M)                                  \
  ds41rt_b12x_moe_tp4_w4a16_m1_parity_m##M##_topk8_Kernel_Module_t              \
      moe_tp4_w4a16_m1_parity_m##M##_topk8_module;
DS41RT_DEFINE_W4A16_M1_PARITY_MODULE(2)
DS41RT_DEFINE_W4A16_M1_PARITY_MODULE(3)
DS41RT_DEFINE_W4A16_M1_PARITY_MODULE(4)
DS41RT_DEFINE_W4A16_M1_PARITY_MODULE(5)
DS41RT_DEFINE_W4A16_M1_PARITY_MODULE(6)
DS41RT_DEFINE_W4A16_M1_PARITY_MODULE(7)
DS41RT_DEFINE_W4A16_M1_PARITY_MODULE(8)
#undef DS41RT_DEFINE_W4A16_M1_PARITY_MODULE
#define DS41RT_DEFINE_W4A16_M1_PARITY_GROUPED_MODULE(M)                          \
  ds41rt_b12x_moe_tp4_w4a16_m1_parity_grouped_m##M##_topk8_Kernel_Module_t      \
      moe_tp4_w4a16_m1_parity_grouped_m##M##_topk8_module;
DS41RT_DEFINE_W4A16_M1_PARITY_GROUPED_MODULE(2)
DS41RT_DEFINE_W4A16_M1_PARITY_GROUPED_MODULE(3)
DS41RT_DEFINE_W4A16_M1_PARITY_GROUPED_MODULE(4)
DS41RT_DEFINE_W4A16_M1_PARITY_GROUPED_MODULE(5)
DS41RT_DEFINE_W4A16_M1_PARITY_GROUPED_MODULE(6)
DS41RT_DEFINE_W4A16_M1_PARITY_GROUPED_MODULE(7)
DS41RT_DEFINE_W4A16_M1_PARITY_GROUPED_MODULE(8)
#undef DS41RT_DEFINE_W4A16_M1_PARITY_GROUPED_MODULE
#define DS41RT_DEFINE_W4A16_M1_PARITY_GROUPED_WIDE_MODULE(M)                    \
  ds41rt_b12x_moe_tp4_w4a16_m1_parity_grouped_wide_m##M##_topk8_Kernel_Module_t \
      moe_tp4_w4a16_m1_parity_grouped_wide_m##M##_topk8_module;
DS41RT_DEFINE_W4A16_M1_PARITY_GROUPED_WIDE_MODULE(2)
DS41RT_DEFINE_W4A16_M1_PARITY_GROUPED_WIDE_MODULE(3)
DS41RT_DEFINE_W4A16_M1_PARITY_GROUPED_WIDE_MODULE(4)
DS41RT_DEFINE_W4A16_M1_PARITY_GROUPED_WIDE_MODULE(5)
DS41RT_DEFINE_W4A16_M1_PARITY_GROUPED_WIDE_MODULE(6)
DS41RT_DEFINE_W4A16_M1_PARITY_GROUPED_WIDE_MODULE(7)
DS41RT_DEFINE_W4A16_M1_PARITY_GROUPED_WIDE_MODULE(8)
#undef DS41RT_DEFINE_W4A16_M1_PARITY_GROUPED_WIDE_MODULE
ds41rt_b12x_moe_tp4_w4a16_prefill_m2_topk8_Kernel_Module_t
    moe_tp4_w4a16_prefill_m2_topk8_module;
ds41rt_b12x_moe_tp4_w4a16_prefill_m4_topk8_Kernel_Module_t
    moe_tp4_w4a16_prefill_m4_topk8_module;
ds41rt_b12x_moe_tp4_w4a16_prefill_m8_topk8_Kernel_Module_t
    moe_tp4_w4a16_prefill_m8_topk8_module;
ds41rt_b12x_moe_tp4_w4a16_prefill_m16_topk8_Kernel_Module_t
    moe_tp4_w4a16_prefill_m16_topk8_module;
ds41rt_b12x_moe_tp4_w4a16_prefill_m32_topk8_Kernel_Module_t
    moe_tp4_w4a16_prefill_m32_topk8_module;
ds41rt_b12x_moe_tp4_w4a16_prefill_m64_topk8_Kernel_Module_t
    moe_tp4_w4a16_prefill_m64_topk8_module;
ds41rt_b12x_moe_tp4_w4a16_prefill_m128_topk8_Kernel_Module_t
    moe_tp4_w4a16_prefill_m128_topk8_module;
ds41rt_b12x_moe_tp4_w4a16_prefill_m256_topk8_Kernel_Module_t
    moe_tp4_w4a16_prefill_m256_topk8_module;
ds41rt_b12x_moe_tp4_w4a16_prefill_m1024_topk8_Kernel_Module_t
    moe_tp4_w4a16_prefill_m1024_topk8_module;
ds41rt_b12x_moe_tp4_w4a16_prefill_m2048_topk8_Kernel_Module_t
    moe_tp4_w4a16_prefill_m2048_topk8_module;
ds41rt_b12x_moe_tp4_w4a16_prefill_m512_topk8_Kernel_Module_t
    moe_tp4_w4a16_prefill_m512_topk8_module;
ds41rt_b12x_moe_tp4_w4a16_top1_m1_Kernel_Module_t moe_tp4_w4a16_top1_m1_module;
ds41rt_b12x_moe_tp4_w4a16_top1_m2_Kernel_Module_t moe_tp4_w4a16_top1_m2_module;
ds41rt_b12x_moe_tp4_w4a16_top1_m4_Kernel_Module_t moe_tp4_w4a16_top1_m4_module;
ds41rt_b12x_moe_tp4_w4a16_top1_m8_Kernel_Module_t moe_tp4_w4a16_top1_m8_module;
ds41rt_b12x_moe_tp4_w4a16_top1_m16_Kernel_Module_t moe_tp4_w4a16_top1_m16_module;
ds41rt_b12x_moe_tp4_w4a16_top1_m32_Kernel_Module_t moe_tp4_w4a16_top1_m32_module;
ds41rt_b12x_moe_tp4_w4a16_top1_m64_Kernel_Module_t moe_tp4_w4a16_top1_m64_module;
ds41rt_b12x_moe_tp4_w4a16_top1_m128_Kernel_Module_t moe_tp4_w4a16_top1_m128_module;
ds41rt_b12x_moe_tp4_w4a16_top1_m256_Kernel_Module_t moe_tp4_w4a16_top1_m256_module;
std::once_flag b12x_module_init_once;
ds41rt_status_t b12x_module_init_status = DS41RT_STATUS_OK;
constexpr size_t kB12xW4a16LockElements = 48 * 4 + 2;
constexpr int kB12xW4a16DecodeMaxGridX =
    static_cast<int>((kB12xW4a16LockElements - 2) / 2);
constexpr int kB12xW4a16DecodeResidentGridX = 48;
constexpr int kB12xW4a16Top1M1GridX = 32;
constexpr int kB12xW4a16Top1GridX = 48;

int w4a16_decode_grid_x() {
  static const int grid_x = [] {
    const char* raw = std::getenv("DS41RT_B12X_SPARK_W4A16_DECODE_GRID_X");
    if (raw == nullptr || *raw == '\0') {
      return DS41RT_B12X_W4A16_DECODE_M1_GRID_X;
    }
    char* end = nullptr;
    const long parsed = std::strtol(raw, &end, 10);
    if (end == raw || *end != '\0' || parsed <= 0 ||
        parsed > kB12xW4a16DecodeMaxGridX) {
      return DS41RT_B12X_W4A16_DECODE_M1_GRID_X;
    }
    return static_cast<int>(parsed);
  }();
  return grid_x;
}

bool w4a16_m1_fused_sum_enabled() {
  static const bool enabled = [] {
    const char* raw =
        std::getenv("DS41RT_B12X_SPARK_W4A16_M1_FUSED_SUM");
    return raw != nullptr && *raw != '\0' &&
           !(raw[0] == '0' && raw[1] == '\0');
  }();
  return enabled;
}

bool buffer_has_bytes(ds41rt_device_buffer_t buffer, size_t required) {
  return buffer.ptr != nullptr && buffer.bytes >= required;
}

__global__ void dequantize_nvfp4_row_payload_bf16_kernel(const uint8_t* payload,
                                                          uint16_t* output,
                                                          size_t hidden_dim) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= hidden_dim) {
    return;
  }
  const size_t packed_bytes = hidden_dim / 2;
  const uint8_t packed = payload[index / 2];
  const uint8_t code = index % 2 == 0 ? packed & 0x0f : packed >> 4;
  const float scale = f8e4m3_to_f32(payload[packed_bytes + index / 16]);
  output[index] = f32_to_bf16(nvfp4_e2m1_code_value(code) * scale);
}

__global__ void dequantize_nvfp4_row_payloads_bf16_kernel(
    const uint8_t* payload, size_t row_stride_bytes, uint16_t* output,
    size_t rows, size_t hidden_dim) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t values = rows * hidden_dim;
  if (index >= values) {
    return;
  }
  const size_t row = index / hidden_dim;
  const size_t col = index % hidden_dim;
  const uint8_t* source = payload + row * row_stride_bytes;
  const size_t packed_bytes = hidden_dim / 2;
  const uint8_t packed = source[col / 2];
  const uint8_t code = (col & 1) == 0 ? packed & 0x0f : packed >> 4;
  const float scale = f8e4m3_to_f32(source[packed_bytes + col / 16]);
  output[index] = f32_to_bf16(nvfp4_e2m1_code_value(code) * scale);
}

__global__ void sum_w4a16_topk_bf16_kernel(const uint16_t* routed,
                                             uint16_t* output, size_t rows,
                                             size_t hidden_dim) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t values = rows * hidden_dim;
  if (index >= values) {
    return;
  }
  const size_t row = index / hidden_dim;
  const size_t col = index % hidden_dim;
  float sum = 0.0f;
#pragma unroll
  for (size_t route = 0; route < kB12xTopK; ++route) {
    sum += bf16_to_f32(routed[(row * kB12xTopK + route) * hidden_dim + col]);
  }
  output[index] = f32_to_bf16(sum);
}

__global__ void sum_w4a16_topk_bf16_to_fp8_row_scaled_kernel(
    const uint16_t* routed, uint8_t* output, size_t rows,
    size_t output_row_stride_bytes) {
  constexpr int kBlock = 256;
  __shared__ uint16_t rounded_values[kB12xHidden];
  __shared__ float maxima[kBlock];
  __shared__ float row_scale;
  const size_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  float maximum = 0.0f;
  for (size_t col = threadIdx.x; col < kB12xHidden; col += blockDim.x) {
    float sum = 0.0f;
#pragma unroll
    for (size_t route = 0; route < kB12xTopK; ++route) {
      sum += bf16_to_f32(
          routed[(row * kB12xTopK + route) * kB12xHidden + col]);
    }
    const uint16_t rounded = f32_to_bf16(sum);
    const float value = bf16_to_f32(rounded);
    rounded_values[col] = rounded;
    maximum = fmaxf(maximum, isfinite(value) ? fabsf(value) : 0.0f);
  }
  maxima[threadIdx.x] = maximum;
  __syncthreads();
  for (int stride = kBlock / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      maxima[threadIdx.x] =
          fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  uint8_t* row_output = output + row * output_row_stride_bytes;
  if (threadIdx.x == 0) {
    row_scale = maxima[0] > 0.0f ? maxima[0] / 448.0f : 1.0f;
    *reinterpret_cast<float*>(row_output + kB12xHidden) = row_scale;
  }
  __syncthreads();
  for (size_t col = threadIdx.x; col < kB12xHidden; col += blockDim.x) {
    const float value = bf16_to_f32(rounded_values[col]);
    row_output[col] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(
        isfinite(value) ? value / row_scale : 0.0f, __NV_SATFINITE,
        __NV_E4M3));
  }
}

__global__ void gather_nvfp4_rows_bf16_kernel(
    const uint8_t* payload, size_t source_rows, size_t source_row_stride_bytes,
    const uint32_t* row_indices, uint16_t* output, size_t rows, size_t hidden_dim) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t values = rows * hidden_dim;
  if (index >= values) {
    return;
  }
  const size_t row = index / hidden_dim;
  const size_t col = index % hidden_dim;
  const size_t source_row = row_indices[row];
  if (source_row >= source_rows) {
    output[index] = 0;
    return;
  }
  const uint8_t* source = payload + source_row * source_row_stride_bytes;
  const size_t packed_bytes = hidden_dim / 2;
  const uint8_t packed = source[col / 2];
  const uint8_t code = (col & 1) == 0 ? packed & 0x0f : packed >> 4;
  const float scale = f8e4m3_to_f32(source[packed_bytes + col / 16]);
  output[index] = f32_to_bf16(nvfp4_e2m1_code_value(code) * scale);
}

__global__ void pack_w4a16_weight_kernel(const uint8_t* source, uint32_t* destination,
                                          size_t size_k, size_t source_size_k,
                                          size_t source_start_k, size_t size_n,
                                          size_t row_rotation) {
  const size_t output_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t k_tiles = size_k / 16;
  const size_t n_tiles = size_n / 64;
  const size_t output_words = k_tiles * n_tiles * 128;
  if (output_index >= output_words) {
    return;
  }
  const size_t packed_position = output_index % 128;
  const size_t tile_index = output_index / 128;
  const size_t n_tile = tile_index % n_tiles;
  const size_t k_tile = tile_index / n_tiles;
  const size_t thread_group = packed_position / 4;
  const size_t warp_column = packed_position % 4;
  const size_t tensor_column = thread_group / 4;
  const size_t tensor_row = (thread_group % 4) * 2;
  constexpr int element_offsets[4] = {0, 1, 8, 9};
  constexpr int pack_order[8] = {0, 2, 4, 6, 1, 3, 5, 7};
  uint32_t result = 0;
  for (int slot = 0; slot < 8; ++slot) {
    const int source_slot = pack_order[slot];
    const int element_slot = source_slot & 3;
    const size_t element = tensor_row + element_offsets[element_slot];
    const size_t k_half = element / 8;
    const size_t nibble = element % 8;
    const size_t column_base = warp_column * 16 + tensor_column;
    const size_t packed_row = n_tile * 64 + column_base + (source_slot >= 4 ? 8 : 0);
    const size_t source_row = (packed_row + row_rotation) % size_n;
    const size_t source_word = source_start_k / 8 + k_tile * 2 + k_half;
    const uint32_t word = reinterpret_cast<const uint32_t*>(source)[
        source_row * (source_size_k / 8) + source_word];
    result |= ((word >> (nibble * 4)) & 0x0fU) << (slot * 4);
  }
  destination[output_index] = result;
}

__global__ void pack_w4a16_scale_kernel(const uint8_t* source, uint8_t* destination,
                                         size_t size_k, size_t source_size_k,
                                         size_t source_start_k, size_t size_n,
                                         size_t row_rotation, float scale_factor) {
  const size_t output_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t values = (size_k / 16) * size_n;
  if (output_index >= values) {
    return;
  }
  const size_t k_block = output_index / size_n;
  const size_t output_row = output_index % size_n;
  constexpr int swap_four[4] = {0, 2, 1, 3};
  const size_t swapped = (output_row & ~size_t{3}) + swap_four[output_row & 3];
  const size_t group_base = (swapped / 64) * 64;
  const size_t group_offset = swapped % 64;
  const size_t permuted_row = group_base + group_offset / 8 + 8 * (group_offset % 8);
  const size_t source_row = (permuted_row + row_rotation) % size_n;
  const float source_scale = f8e4m3_to_f32(
      source[source_row * (source_size_k / 16) + source_start_k / 16 + k_block]);
  const float adjusted = source_scale * scale_factor * 128.0f;
  if (adjusted < 2.0f) {
    destination[output_index] = 0;
    return;
  }
  const __half_raw encoded = __float2half_rn(adjusted);
  destination[output_index] = static_cast<uint8_t>((encoded.x >> 7) & 0xffU);
}

__global__ void initialize_w4a16_top1_routes_kernel(
    int32_t* packed_route_indices, int32_t* block_expert_ids,
    int32_t* packed_route_count, float* topk_weights, size_t rows,
    size_t capacity_rows, uint32_t expert_id, bool direct_topk) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < capacity_rows) {
    topk_weights[index] = 1.0f;
  }
  if (direct_topk) {
    if (index < rows) {
      packed_route_indices[index] = static_cast<int32_t>(expert_id);
    }
    return;
  }
  const size_t padded_rows = ((rows + 7) / 8) * 8;
  if (index < padded_rows) {
    packed_route_indices[index] =
        index < rows ? static_cast<int32_t>(index) : static_cast<int32_t>(rows);
  }
  if (index < padded_rows / 8) {
    block_expert_ids[index] = static_cast<int32_t>(expert_id);
  }
  if (index == 0) {
    packed_route_count[0] = static_cast<int32_t>(padded_rows);
  }
}

void initialize_b12x_modules() {
  ds41rt_b12x_moe_tp4_w4a16_decode_m1_Kernel_Module_Load(
      &moe_tp4_w4a16_decode_m1_module);
  ds41rt_b12x_moe_tp4_w4a16_decode_m1_fused_sum_Kernel_Module_Load(
      &moe_tp4_w4a16_decode_m1_fused_sum_module);
#define DS41RT_LOAD_W4A16_M1_PARITY_MODULE(M)                                    \
  ds41rt_b12x_moe_tp4_w4a16_m1_parity_m##M##_topk8_Kernel_Module_Load(           \
      &moe_tp4_w4a16_m1_parity_m##M##_topk8_module);
  DS41RT_LOAD_W4A16_M1_PARITY_MODULE(2)
  DS41RT_LOAD_W4A16_M1_PARITY_MODULE(3)
  DS41RT_LOAD_W4A16_M1_PARITY_MODULE(4)
  DS41RT_LOAD_W4A16_M1_PARITY_MODULE(5)
  DS41RT_LOAD_W4A16_M1_PARITY_MODULE(6)
  DS41RT_LOAD_W4A16_M1_PARITY_MODULE(7)
  DS41RT_LOAD_W4A16_M1_PARITY_MODULE(8)
#undef DS41RT_LOAD_W4A16_M1_PARITY_MODULE
#define DS41RT_LOAD_W4A16_M1_PARITY_GROUPED_MODULE(M)                           \
  ds41rt_b12x_moe_tp4_w4a16_m1_parity_grouped_m##M##_topk8_Kernel_Module_Load(  \
      &moe_tp4_w4a16_m1_parity_grouped_m##M##_topk8_module);
  DS41RT_LOAD_W4A16_M1_PARITY_GROUPED_MODULE(2)
  DS41RT_LOAD_W4A16_M1_PARITY_GROUPED_MODULE(3)
  DS41RT_LOAD_W4A16_M1_PARITY_GROUPED_MODULE(4)
  DS41RT_LOAD_W4A16_M1_PARITY_GROUPED_MODULE(5)
  DS41RT_LOAD_W4A16_M1_PARITY_GROUPED_MODULE(6)
  DS41RT_LOAD_W4A16_M1_PARITY_GROUPED_MODULE(7)
  DS41RT_LOAD_W4A16_M1_PARITY_GROUPED_MODULE(8)
#undef DS41RT_LOAD_W4A16_M1_PARITY_GROUPED_MODULE
#define DS41RT_LOAD_W4A16_M1_PARITY_GROUPED_WIDE_MODULE(M)                      \
  ds41rt_b12x_moe_tp4_w4a16_m1_parity_grouped_wide_m##M##_topk8_Kernel_Module_Load( \
      &moe_tp4_w4a16_m1_parity_grouped_wide_m##M##_topk8_module);
  DS41RT_LOAD_W4A16_M1_PARITY_GROUPED_WIDE_MODULE(2)
  DS41RT_LOAD_W4A16_M1_PARITY_GROUPED_WIDE_MODULE(3)
  DS41RT_LOAD_W4A16_M1_PARITY_GROUPED_WIDE_MODULE(4)
  DS41RT_LOAD_W4A16_M1_PARITY_GROUPED_WIDE_MODULE(5)
  DS41RT_LOAD_W4A16_M1_PARITY_GROUPED_WIDE_MODULE(6)
  DS41RT_LOAD_W4A16_M1_PARITY_GROUPED_WIDE_MODULE(7)
  DS41RT_LOAD_W4A16_M1_PARITY_GROUPED_WIDE_MODULE(8)
#undef DS41RT_LOAD_W4A16_M1_PARITY_GROUPED_WIDE_MODULE
  ds41rt_b12x_moe_tp4_w4a16_prefill_m2_topk8_Kernel_Module_Load(
      &moe_tp4_w4a16_prefill_m2_topk8_module);
  ds41rt_b12x_moe_tp4_w4a16_prefill_m4_topk8_Kernel_Module_Load(
      &moe_tp4_w4a16_prefill_m4_topk8_module);
  ds41rt_b12x_moe_tp4_w4a16_prefill_m8_topk8_Kernel_Module_Load(
      &moe_tp4_w4a16_prefill_m8_topk8_module);
  ds41rt_b12x_moe_tp4_w4a16_prefill_m16_topk8_Kernel_Module_Load(
      &moe_tp4_w4a16_prefill_m16_topk8_module);
  ds41rt_b12x_moe_tp4_w4a16_prefill_m32_topk8_Kernel_Module_Load(
      &moe_tp4_w4a16_prefill_m32_topk8_module);
  ds41rt_b12x_moe_tp4_w4a16_prefill_m64_topk8_Kernel_Module_Load(
      &moe_tp4_w4a16_prefill_m64_topk8_module);
  ds41rt_b12x_moe_tp4_w4a16_prefill_m128_topk8_Kernel_Module_Load(
      &moe_tp4_w4a16_prefill_m128_topk8_module);
  ds41rt_b12x_moe_tp4_w4a16_prefill_m256_topk8_Kernel_Module_Load(
      &moe_tp4_w4a16_prefill_m256_topk8_module);
  ds41rt_b12x_moe_tp4_w4a16_prefill_m1024_topk8_Kernel_Module_Load(
      &moe_tp4_w4a16_prefill_m1024_topk8_module);
  ds41rt_b12x_moe_tp4_w4a16_prefill_m2048_topk8_Kernel_Module_Load(
      &moe_tp4_w4a16_prefill_m2048_topk8_module);
  ds41rt_b12x_moe_tp4_w4a16_prefill_m512_topk8_Kernel_Module_Load(
      &moe_tp4_w4a16_prefill_m512_topk8_module);
  ds41rt_b12x_moe_tp4_w4a16_top1_m1_Kernel_Module_Load(
      &moe_tp4_w4a16_top1_m1_module);
  ds41rt_b12x_moe_tp4_w4a16_top1_m2_Kernel_Module_Load(
      &moe_tp4_w4a16_top1_m2_module);
  ds41rt_b12x_moe_tp4_w4a16_top1_m4_Kernel_Module_Load(
      &moe_tp4_w4a16_top1_m4_module);
  ds41rt_b12x_moe_tp4_w4a16_top1_m8_Kernel_Module_Load(
      &moe_tp4_w4a16_top1_m8_module);
  ds41rt_b12x_moe_tp4_w4a16_top1_m16_Kernel_Module_Load(
      &moe_tp4_w4a16_top1_m16_module);
  ds41rt_b12x_moe_tp4_w4a16_top1_m32_Kernel_Module_Load(
      &moe_tp4_w4a16_top1_m32_module);
  ds41rt_b12x_moe_tp4_w4a16_top1_m64_Kernel_Module_Load(
      &moe_tp4_w4a16_top1_m64_module);
  ds41rt_b12x_moe_tp4_w4a16_top1_m128_Kernel_Module_Load(
      &moe_tp4_w4a16_top1_m128_module);
  ds41rt_b12x_moe_tp4_w4a16_top1_m256_Kernel_Module_Load(
      &moe_tp4_w4a16_top1_m256_module);
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    ds41rt_set_last_error_message(cudaGetErrorString(error));
    b12x_module_init_status = DS41RT_STATUS_INTERNAL_ERROR;
  }
}

ds41rt_status_t check_aot_launch(int result, const char* label) {
  if (result == 0) {
    return DS41RT_STATUS_OK;
  }
  ds41rt_set_last_error_message(label);
  return DS41RT_STATUS_INTERNAL_ERROR;
}

#define DS41RT_DEFINE_W4A16_LAUNCH(function_name, prefix, module_name, default_grid_x)          \
  int function_name##_grid(const ds41rt_b12x_spark_w4a16_moe_buffers_t* buffers,                \
                           size_t active_m, int grid_x, cudaStream_t stream) {                  \
    prefix##_Tensor_fc1_bf16_flat_t fc1{buffers->fc1_output.ptr};                              \
    prefix##_Tensor_activated_bf16_flat_t activated{buffers->activated.ptr};                   \
    prefix##_Tensor_fc2_bf16_flat_t fc2{buffers->output.ptr};                                  \
    prefix##_Tensor_packed_route_indices_t packed_routes{buffers->packed_route_indices.ptr};   \
    prefix##_Tensor_block_expert_ids_t block_experts{buffers->block_expert_ids.ptr};           \
    prefix##_Tensor_packed_route_count_t route_count{buffers->packed_route_count.ptr};          \
    prefix##_Tensor_activation_amax_flat_t activation_amax{buffers->w13_global_scale.ptr};      \
    prefix##_Tensor_fc1_c_tmp_f32_flat_t fc1_scratch{buffers->fc1_scratch.ptr};                \
    prefix##_Tensor_fc2_c_tmp_f32_flat_t fc2_scratch{buffers->fc2_scratch.ptr};                \
    prefix##_Tensor_locks_i32_flat_t locks{buffers->locks.ptr};                                \
    return cute_dsl_##prefix##_wrapper(                                                         \
        &module_name, buffers->input.ptr, buffers->input.ptr, buffers->input.ptr,               \
        buffers->w13_weight.ptr, buffers->w2_weight.ptr,                                        \
        static_cast<int64_t>(buffers->w13_weight.bytes / sizeof(int32_t)),                       \
        static_cast<int64_t>(buffers->w2_weight.bytes / sizeof(int32_t)),                        \
        &fc1, &activated, &fc2,                                                                  \
        buffers->w13_scale.ptr, buffers->w2_scale.ptr, buffers->w13_global_scale.ptr,           \
        buffers->w2_global_scale.ptr,                                                           \
        &packed_routes, &block_experts, &route_count, &activation_amax, 0,                     \
        buffers->topk_weights.ptr, &fc1_scratch, &fc2_scratch, &locks,                         \
        buffers->w13_global_scale.ptr, buffers->w13_global_scale.ptr,                           \
        buffers->w13_global_scale.ptr, buffers->packed_route_indices.ptr,                      \
        buffers->w13_scale.ptr, buffers->w2_scale.ptr,                                         \
        static_cast<int32_t>(kB12xExperts), 0,                                                  \
        static_cast<int32_t>(active_m), static_cast<int32_t>(grid_x), stream);                  \
  }                                                                                            \
  int function_name(const ds41rt_b12x_spark_w4a16_moe_buffers_t* buffers, size_t active_m,      \
                    cudaStream_t stream) {                                                     \
    return function_name##_grid(buffers, active_m, static_cast<int>(default_grid_x), stream);  \
  }

DS41RT_DEFINE_W4A16_LAUNCH(
    launch_w4a16_decode_m1, ds41rt_b12x_moe_tp4_w4a16_decode_m1,
    moe_tp4_w4a16_decode_m1_module, w4a16_decode_grid_x())
DS41RT_DEFINE_W4A16_LAUNCH(
    launch_w4a16_decode_m1_fused_sum,
    ds41rt_b12x_moe_tp4_w4a16_decode_m1_fused_sum,
    moe_tp4_w4a16_decode_m1_fused_sum_module,
    DS41RT_B12X_W4A16_DECODE_M1_FUSED_SUM_GRID_X)
#define DS41RT_DEFINE_W4A16_M1_PARITY_LAUNCH(M)                                  \
  DS41RT_DEFINE_W4A16_LAUNCH(                                                    \
      launch_w4a16_m1_parity_m##M##_topk8,                                     \
      ds41rt_b12x_moe_tp4_w4a16_m1_parity_m##M##_topk8,                         \
      moe_tp4_w4a16_m1_parity_m##M##_topk8_module,                             \
      DS41RT_B12X_W4A16_M1_PARITY_M##M##_TOPK8_GRID_X)
DS41RT_DEFINE_W4A16_M1_PARITY_LAUNCH(2)
DS41RT_DEFINE_W4A16_M1_PARITY_LAUNCH(3)
DS41RT_DEFINE_W4A16_M1_PARITY_LAUNCH(4)
DS41RT_DEFINE_W4A16_M1_PARITY_LAUNCH(5)
DS41RT_DEFINE_W4A16_M1_PARITY_LAUNCH(6)
DS41RT_DEFINE_W4A16_M1_PARITY_LAUNCH(7)
DS41RT_DEFINE_W4A16_M1_PARITY_LAUNCH(8)
#undef DS41RT_DEFINE_W4A16_M1_PARITY_LAUNCH
#define DS41RT_DEFINE_W4A16_M1_PARITY_GROUPED_LAUNCH(M)                         \
  DS41RT_DEFINE_W4A16_LAUNCH(                                                   \
      launch_w4a16_m1_parity_grouped_m##M##_topk8,                            \
      ds41rt_b12x_moe_tp4_w4a16_m1_parity_grouped_m##M##_topk8,                \
      moe_tp4_w4a16_m1_parity_grouped_m##M##_topk8_module,                    \
      DS41RT_B12X_W4A16_M1_PARITY_GROUPED_M##M##_TOPK8_GRID_X)
DS41RT_DEFINE_W4A16_M1_PARITY_GROUPED_LAUNCH(2)
DS41RT_DEFINE_W4A16_M1_PARITY_GROUPED_LAUNCH(3)
DS41RT_DEFINE_W4A16_M1_PARITY_GROUPED_LAUNCH(4)
DS41RT_DEFINE_W4A16_M1_PARITY_GROUPED_LAUNCH(5)
DS41RT_DEFINE_W4A16_M1_PARITY_GROUPED_LAUNCH(6)
DS41RT_DEFINE_W4A16_M1_PARITY_GROUPED_LAUNCH(7)
DS41RT_DEFINE_W4A16_M1_PARITY_GROUPED_LAUNCH(8)
#undef DS41RT_DEFINE_W4A16_M1_PARITY_GROUPED_LAUNCH
#define DS41RT_DEFINE_W4A16_M1_PARITY_GROUPED_WIDE_LAUNCH(M)                    \
  DS41RT_DEFINE_W4A16_LAUNCH(                                                   \
      launch_w4a16_m1_parity_grouped_wide_m##M##_topk8,                       \
      ds41rt_b12x_moe_tp4_w4a16_m1_parity_grouped_wide_m##M##_topk8,           \
      moe_tp4_w4a16_m1_parity_grouped_wide_m##M##_topk8_module,               \
      DS41RT_B12X_W4A16_M1_PARITY_GROUPED_WIDE_M##M##_TOPK8_GRID_X)
DS41RT_DEFINE_W4A16_M1_PARITY_GROUPED_WIDE_LAUNCH(2)
DS41RT_DEFINE_W4A16_M1_PARITY_GROUPED_WIDE_LAUNCH(3)
DS41RT_DEFINE_W4A16_M1_PARITY_GROUPED_WIDE_LAUNCH(4)
DS41RT_DEFINE_W4A16_M1_PARITY_GROUPED_WIDE_LAUNCH(5)
DS41RT_DEFINE_W4A16_M1_PARITY_GROUPED_WIDE_LAUNCH(6)
DS41RT_DEFINE_W4A16_M1_PARITY_GROUPED_WIDE_LAUNCH(7)
DS41RT_DEFINE_W4A16_M1_PARITY_GROUPED_WIDE_LAUNCH(8)
#undef DS41RT_DEFINE_W4A16_M1_PARITY_GROUPED_WIDE_LAUNCH
DS41RT_DEFINE_W4A16_LAUNCH(
    launch_w4a16_prefill_m2_topk8,
    ds41rt_b12x_moe_tp4_w4a16_prefill_m2_topk8,
    moe_tp4_w4a16_prefill_m2_topk8_module,
    DS41RT_B12X_W4A16_PREFILL_M2_TOPK8_GRID_X)
DS41RT_DEFINE_W4A16_LAUNCH(
    launch_w4a16_prefill_m4_topk8,
    ds41rt_b12x_moe_tp4_w4a16_prefill_m4_topk8,
    moe_tp4_w4a16_prefill_m4_topk8_module,
    DS41RT_B12X_W4A16_PREFILL_M4_TOPK8_GRID_X)
DS41RT_DEFINE_W4A16_LAUNCH(
    launch_w4a16_prefill_m8_topk8,
    ds41rt_b12x_moe_tp4_w4a16_prefill_m8_topk8,
    moe_tp4_w4a16_prefill_m8_topk8_module,
    DS41RT_B12X_W4A16_PREFILL_M8_TOPK8_GRID_X)
DS41RT_DEFINE_W4A16_LAUNCH(
    launch_w4a16_prefill_m16_topk8,
    ds41rt_b12x_moe_tp4_w4a16_prefill_m16_topk8,
    moe_tp4_w4a16_prefill_m16_topk8_module,
    DS41RT_B12X_W4A16_PREFILL_M16_TOPK8_GRID_X)
DS41RT_DEFINE_W4A16_LAUNCH(
    launch_w4a16_prefill_m32_topk8,
    ds41rt_b12x_moe_tp4_w4a16_prefill_m32_topk8,
    moe_tp4_w4a16_prefill_m32_topk8_module,
    DS41RT_B12X_W4A16_PREFILL_M32_TOPK8_GRID_X)
DS41RT_DEFINE_W4A16_LAUNCH(
    launch_w4a16_prefill_m64_topk8,
    ds41rt_b12x_moe_tp4_w4a16_prefill_m64_topk8,
    moe_tp4_w4a16_prefill_m64_topk8_module,
    DS41RT_B12X_W4A16_PREFILL_M64_TOPK8_GRID_X)
DS41RT_DEFINE_W4A16_LAUNCH(
    launch_w4a16_prefill_m128_topk8,
    ds41rt_b12x_moe_tp4_w4a16_prefill_m128_topk8,
    moe_tp4_w4a16_prefill_m128_topk8_module,
    DS41RT_B12X_W4A16_PREFILL_M128_TOPK8_GRID_X)
DS41RT_DEFINE_W4A16_LAUNCH(
    launch_w4a16_prefill_m256_topk8,
    ds41rt_b12x_moe_tp4_w4a16_prefill_m256_topk8,
    moe_tp4_w4a16_prefill_m256_topk8_module,
    DS41RT_B12X_W4A16_PREFILL_M256_TOPK8_GRID_X)
DS41RT_DEFINE_W4A16_LAUNCH(
    launch_w4a16_prefill_m512_topk8,
    ds41rt_b12x_moe_tp4_w4a16_prefill_m512_topk8,
    moe_tp4_w4a16_prefill_m512_topk8_module,
    DS41RT_B12X_W4A16_PREFILL_M512_TOPK8_GRID_X)
DS41RT_DEFINE_W4A16_LAUNCH(
    launch_w4a16_prefill_m1024_topk8,
    ds41rt_b12x_moe_tp4_w4a16_prefill_m1024_topk8,
    moe_tp4_w4a16_prefill_m1024_topk8_module,
    DS41RT_B12X_W4A16_PREFILL_M1024_TOPK8_GRID_X)
DS41RT_DEFINE_W4A16_LAUNCH(
    launch_w4a16_prefill_m2048_topk8,
    ds41rt_b12x_moe_tp4_w4a16_prefill_m2048_topk8,
    moe_tp4_w4a16_prefill_m2048_topk8_module,
    DS41RT_B12X_W4A16_PREFILL_M2048_TOPK8_GRID_X)
DS41RT_DEFINE_W4A16_LAUNCH(
    launch_w4a16_top1_m1, ds41rt_b12x_moe_tp4_w4a16_top1_m1,
    moe_tp4_w4a16_top1_m1_module, kB12xW4a16Top1M1GridX)
DS41RT_DEFINE_W4A16_LAUNCH(
    launch_w4a16_top1_m2, ds41rt_b12x_moe_tp4_w4a16_top1_m2,
    moe_tp4_w4a16_top1_m2_module, kB12xW4a16Top1GridX)
DS41RT_DEFINE_W4A16_LAUNCH(
    launch_w4a16_top1_m4, ds41rt_b12x_moe_tp4_w4a16_top1_m4,
    moe_tp4_w4a16_top1_m4_module, kB12xW4a16Top1GridX)
DS41RT_DEFINE_W4A16_LAUNCH(
    launch_w4a16_top1_m8, ds41rt_b12x_moe_tp4_w4a16_top1_m8,
    moe_tp4_w4a16_top1_m8_module, kB12xW4a16Top1GridX)
DS41RT_DEFINE_W4A16_LAUNCH(
    launch_w4a16_top1_m16, ds41rt_b12x_moe_tp4_w4a16_top1_m16,
    moe_tp4_w4a16_top1_m16_module, kB12xW4a16Top1GridX)
DS41RT_DEFINE_W4A16_LAUNCH(
    launch_w4a16_top1_m32, ds41rt_b12x_moe_tp4_w4a16_top1_m32,
    moe_tp4_w4a16_top1_m32_module, kB12xW4a16Top1GridX)
DS41RT_DEFINE_W4A16_LAUNCH(
    launch_w4a16_top1_m64, ds41rt_b12x_moe_tp4_w4a16_top1_m64,
    moe_tp4_w4a16_top1_m64_module, kB12xW4a16Top1GridX)
DS41RT_DEFINE_W4A16_LAUNCH(
    launch_w4a16_top1_m128, ds41rt_b12x_moe_tp4_w4a16_top1_m128,
    moe_tp4_w4a16_top1_m128_module, kB12xW4a16Top1GridX)
DS41RT_DEFINE_W4A16_LAUNCH(
    launch_w4a16_top1_m256, ds41rt_b12x_moe_tp4_w4a16_top1_m256,
    moe_tp4_w4a16_top1_m256_module, kB12xW4a16Top1GridX)

#undef DS41RT_DEFINE_W4A16_LAUNCH

ds41rt_status_t validate_w4a16_moe_buffers(
    const ds41rt_b12x_spark_w4a16_moe_buffers_t* buffers, size_t capacity_rows,
    size_t top_k) {
  constexpr size_t w13_weight_bytes =
      kB12xExperts * 2 * kB12xTp4Intermediate * kB12xHidden / 2;
  constexpr size_t w13_scale_bytes =
      kB12xExperts * 2 * kB12xTp4Intermediate * kB12xHidden / 16;
  constexpr size_t w2_weight_bytes =
      kB12xExperts * kB12xOutput * kB12xTp4Intermediate / 2;
  constexpr size_t w2_scale_bytes =
      kB12xExperts * kB12xOutput * kB12xTp4Intermediate / 16;
  constexpr size_t expert_scalars_bytes = kB12xExperts * sizeof(float);
  constexpr size_t max_packed_route_slots =
      DS41RT_B12X_W4A16_PREFILL_M2048_TOPK8_PACKED_ROUTE_SLOTS;
  constexpr size_t max_route_blocks =
      DS41RT_B12X_W4A16_PREFILL_M2048_TOPK8_MAX_M_BLOCKS;
  constexpr size_t max_scratch_elements = 1572864;
  if (buffers == nullptr || capacity_rows == 0 || capacity_rows > kB12xW4a16MaxRows ||
      (top_k != 1 && top_k != kB12xTopK)) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  const size_t output_rows =
      top_k == kB12xTopK ? capacity_rows * top_k : capacity_rows;
  const bool valid =
      buffer_has_bytes(buffers->input, capacity_rows * kB12xHidden * sizeof(uint16_t)) &&
      buffer_has_bytes(buffers->w13_weight, w13_weight_bytes) &&
      buffer_has_bytes(buffers->w2_weight, w2_weight_bytes) &&
      buffer_has_bytes(buffers->fc1_output,
                       capacity_rows * top_k * 2 * kB12xTp4Intermediate *
                           sizeof(uint16_t)) &&
      buffer_has_bytes(buffers->activated,
                       capacity_rows * top_k * kB12xTp4Intermediate *
                           sizeof(uint16_t)) &&
      buffer_has_bytes(buffers->output, output_rows * kB12xOutput * sizeof(uint16_t)) &&
      buffer_has_bytes(buffers->w13_scale, w13_scale_bytes) &&
      buffer_has_bytes(buffers->w2_scale, w2_scale_bytes) &&
      buffer_has_bytes(buffers->w13_global_scale, expert_scalars_bytes) &&
      buffer_has_bytes(buffers->w2_global_scale, expert_scalars_bytes) &&
      buffer_has_bytes(buffers->packed_route_indices,
                       max_packed_route_slots * sizeof(int32_t)) &&
      buffer_has_bytes(buffers->block_expert_ids, max_route_blocks * sizeof(int32_t)) &&
      buffer_has_bytes(buffers->packed_route_count, sizeof(int32_t)) &&
      buffer_has_bytes(buffers->topk_weights, capacity_rows * top_k * sizeof(float)) &&
      buffer_has_bytes(buffers->fc1_scratch, max_scratch_elements * sizeof(float)) &&
      buffer_has_bytes(buffers->fc2_scratch, max_scratch_elements * sizeof(float)) &&
      buffer_has_bytes(buffers->locks, kB12xW4a16LockElements * sizeof(int32_t));
  return valid ? DS41RT_STATUS_OK : DS41RT_STATUS_BUFFER_TOO_SMALL;
}

ds41rt_status_t reset_w4a16_locks_async(
    const ds41rt_b12x_spark_w4a16_moe_buffers_t* buffers, cudaStream_t stream) {
  return status_from_cuda(
      cudaMemsetAsync(buffers->locks.ptr, 0, kB12xW4a16LockElements * sizeof(int32_t), stream));
}

using W4A16LaunchFn = int (*)(const ds41rt_b12x_spark_w4a16_moe_buffers_t*, size_t,
                              cudaStream_t);
using W4A16GridLaunchFn = int (*)(const ds41rt_b12x_spark_w4a16_moe_buffers_t*, size_t,
                                  int, cudaStream_t);

W4A16LaunchFn w4a16_m1_parity_launcher(size_t rows) {
  switch (rows) {
    case 2:
      return &launch_w4a16_m1_parity_m2_topk8;
    case 3:
      return &launch_w4a16_m1_parity_m3_topk8;
    case 4:
      return &launch_w4a16_m1_parity_m4_topk8;
    case 5:
      return &launch_w4a16_m1_parity_m5_topk8;
    case 6:
      return &launch_w4a16_m1_parity_m6_topk8;
    case 7:
      return &launch_w4a16_m1_parity_m7_topk8;
    case 8:
      return &launch_w4a16_m1_parity_m8_topk8;
    default:
      return nullptr;
  }
}

W4A16LaunchFn w4a16_m1_parity_grouped_launcher(size_t rows) {
  switch (rows) {
    case 2:
      return &launch_w4a16_m1_parity_grouped_m2_topk8;
    case 3:
      return &launch_w4a16_m1_parity_grouped_m3_topk8;
    case 4:
      return &launch_w4a16_m1_parity_grouped_m4_topk8;
    case 5:
      return &launch_w4a16_m1_parity_grouped_m5_topk8;
    case 6:
      return &launch_w4a16_m1_parity_grouped_m6_topk8;
    case 7:
      return &launch_w4a16_m1_parity_grouped_m7_topk8;
    case 8:
      return &launch_w4a16_m1_parity_grouped_m8_topk8;
    default:
      return nullptr;
  }
}

W4A16LaunchFn w4a16_m1_parity_grouped_wide_launcher(size_t rows) {
  switch (rows) {
    case 2:
      return &launch_w4a16_m1_parity_grouped_wide_m2_topk8;
    case 3:
      return &launch_w4a16_m1_parity_grouped_wide_m3_topk8;
    case 4:
      return &launch_w4a16_m1_parity_grouped_wide_m4_topk8;
    case 5:
      return &launch_w4a16_m1_parity_grouped_wide_m5_topk8;
    case 6:
      return &launch_w4a16_m1_parity_grouped_wide_m6_topk8;
    case 7:
      return &launch_w4a16_m1_parity_grouped_wide_m7_topk8;
    case 8:
      return &launch_w4a16_m1_parity_grouped_wide_m8_topk8;
    default:
      return nullptr;
  }
}

W4A16LaunchFn w4a16_top1_launcher(size_t capacity_rows) {
  switch (capacity_rows) {
    case 1:
      return &launch_w4a16_top1_m1;
    case 2:
      return &launch_w4a16_top1_m2;
    case 4:
      return &launch_w4a16_top1_m4;
    case 8:
      return &launch_w4a16_top1_m8;
    case 16:
      return &launch_w4a16_top1_m16;
    case 32:
      return &launch_w4a16_top1_m32;
    case 64:
      return &launch_w4a16_top1_m64;
    case 128:
      return &launch_w4a16_top1_m128;
    case 256:
      return &launch_w4a16_top1_m256;
    default:
      return nullptr;
  }
}

W4A16GridLaunchFn w4a16_top1_grid_launcher(size_t capacity_rows) {
  switch (capacity_rows) {
    case 1:
      return &launch_w4a16_top1_m1_grid;
    case 2:
      return &launch_w4a16_top1_m2_grid;
    case 4:
      return &launch_w4a16_top1_m4_grid;
    case 8:
      return &launch_w4a16_top1_m8_grid;
    case 16:
      return &launch_w4a16_top1_m16_grid;
    case 32:
      return &launch_w4a16_top1_m32_grid;
    case 64:
      return &launch_w4a16_top1_m64_grid;
    case 128:
      return &launch_w4a16_top1_m128_grid;
    case 256:
      return &launch_w4a16_top1_m256_grid;
    default:
      return nullptr;
  }
}

W4A16GridLaunchFn w4a16_prefill_grid_launcher(size_t capacity_rows) {
  switch (capacity_rows) {
    case 2:
      return &launch_w4a16_prefill_m2_topk8_grid;
    case 4:
      return &launch_w4a16_prefill_m4_topk8_grid;
    case 8:
      return &launch_w4a16_prefill_m8_topk8_grid;
    case 16:
      return &launch_w4a16_prefill_m16_topk8_grid;
    case 32:
      return &launch_w4a16_prefill_m32_topk8_grid;
    case 64:
      return &launch_w4a16_prefill_m64_topk8_grid;
    case 128:
      return &launch_w4a16_prefill_m128_topk8_grid;
    case 256:
      return &launch_w4a16_prefill_m256_topk8_grid;
    case 512:
      return &launch_w4a16_prefill_m512_topk8_grid;
    case 1024:
      return &launch_w4a16_prefill_m1024_topk8_grid;
    case 2048:
      return &launch_w4a16_prefill_m2048_topk8_grid;
    default:
      return nullptr;
  }
}

}  // namespace

extern "C" ds41rt_status_t ds41rt_cuda_b12x_spark_aot_available(int* out_available) {
  if (out_available == nullptr) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  *out_available = 1;
  return DS41RT_STATUS_OK;
}

extern "C" ds41rt_status_t ds41rt_cuda_b12x_spark_aot_init(void) {
  std::call_once(b12x_module_init_once, initialize_b12x_modules);
  return b12x_module_init_status;
}

extern "C" ds41rt_status_t ds41rt_cuda_b12x_w4a16_pack_weight_async(
    ds41rt_device_buffer_t source, ds41rt_device_buffer_t destination, size_t size_k,
    size_t size_n, size_t row_rotation, void* cuda_stream) {
  if (size_k == 0 || size_n == 0 || size_k % 16 != 0 || size_n % 64 != 0 ||
      row_rotation >= size_n || size_n > std::numeric_limits<size_t>::max() / size_k) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  const size_t bytes = size_n * size_k / 2;
  if (!buffer_has_bytes(source, bytes) || !buffer_has_bytes(destination, bytes)) {
    return DS41RT_STATUS_BUFFER_TOO_SMALL;
  }
  const size_t words = bytes / sizeof(uint32_t);
  constexpr size_t threads = 256;
  const size_t blocks = (words + threads - 1) / threads;
  pack_w4a16_weight_kernel<<<static_cast<unsigned int>(blocks), threads, 0,
                               reinterpret_cast<cudaStream_t>(cuda_stream)>>>(
      static_cast<const uint8_t*>(source.ptr), static_cast<uint32_t*>(destination.ptr),
      size_k, size_k, 0, size_n, row_rotation);
  return status_from_cuda(cudaGetLastError());
}

extern "C" ds41rt_status_t ds41rt_cuda_b12x_w4a16_pack_weight_strided_async(
    ds41rt_device_buffer_t source, ds41rt_device_buffer_t destination, size_t size_k,
    size_t source_size_k, size_t source_start_k, size_t size_n, size_t row_rotation,
    void* cuda_stream) {
  if (size_k == 0 || source_size_k == 0 || size_n == 0 || size_k % 16 != 0 ||
      source_size_k % 16 != 0 || source_start_k % 16 != 0 ||
      source_start_k > source_size_k || size_k > source_size_k - source_start_k ||
      size_n % 64 != 0 || row_rotation >= size_n ||
      size_n > std::numeric_limits<size_t>::max() / source_size_k ||
      size_n > std::numeric_limits<size_t>::max() / size_k) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  const size_t source_bytes = size_n * source_size_k / 2;
  const size_t destination_bytes = size_n * size_k / 2;
  if (!buffer_has_bytes(source, source_bytes) ||
      !buffer_has_bytes(destination, destination_bytes)) {
    return DS41RT_STATUS_BUFFER_TOO_SMALL;
  }
  const size_t words = destination_bytes / sizeof(uint32_t);
  constexpr size_t threads = 256;
  const size_t blocks = (words + threads - 1) / threads;
  pack_w4a16_weight_kernel<<<static_cast<unsigned int>(blocks), threads, 0,
                               reinterpret_cast<cudaStream_t>(cuda_stream)>>>(
      static_cast<const uint8_t*>(source.ptr), static_cast<uint32_t*>(destination.ptr),
      size_k, source_size_k, source_start_k, size_n, row_rotation);
  return status_from_cuda(cudaGetLastError());
}

extern "C" ds41rt_status_t ds41rt_cuda_b12x_w4a16_pack_scale_async(
    ds41rt_device_buffer_t source, ds41rt_device_buffer_t destination, size_t size_k,
    size_t size_n, size_t row_rotation, float scale_factor, void* cuda_stream) {
  if (size_k == 0 || size_n == 0 || size_k % 16 != 0 || size_n % 64 != 0 ||
      row_rotation >= size_n || !isfinite(scale_factor) || scale_factor <= 0.0f ||
      size_n > std::numeric_limits<size_t>::max() / (size_k / 16)) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  const size_t bytes = size_n * (size_k / 16);
  if (!buffer_has_bytes(source, bytes) || !buffer_has_bytes(destination, bytes)) {
    return DS41RT_STATUS_BUFFER_TOO_SMALL;
  }
  constexpr size_t threads = 256;
  const size_t blocks = (bytes + threads - 1) / threads;
  pack_w4a16_scale_kernel<<<static_cast<unsigned int>(blocks), threads, 0,
                              reinterpret_cast<cudaStream_t>(cuda_stream)>>>(
      static_cast<const uint8_t*>(source.ptr), static_cast<uint8_t*>(destination.ptr),
      size_k, size_k, 0, size_n, row_rotation, scale_factor);
  return status_from_cuda(cudaGetLastError());
}

extern "C" ds41rt_status_t ds41rt_cuda_b12x_w4a16_pack_scale_strided_async(
    ds41rt_device_buffer_t source, ds41rt_device_buffer_t destination, size_t size_k,
    size_t source_size_k, size_t source_start_k, size_t size_n, size_t row_rotation,
    float scale_factor, void* cuda_stream) {
  if (size_k == 0 || source_size_k == 0 || size_n == 0 || size_k % 16 != 0 ||
      source_size_k % 16 != 0 || source_start_k % 16 != 0 ||
      source_start_k > source_size_k || size_k > source_size_k - source_start_k ||
      size_n % 64 != 0 || row_rotation >= size_n || !isfinite(scale_factor) ||
      scale_factor <= 0.0f ||
      size_n > std::numeric_limits<size_t>::max() / (source_size_k / 16) ||
      size_n > std::numeric_limits<size_t>::max() / (size_k / 16)) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  const size_t source_bytes = size_n * (source_size_k / 16);
  const size_t destination_bytes = size_n * (size_k / 16);
  if (!buffer_has_bytes(source, source_bytes) ||
      !buffer_has_bytes(destination, destination_bytes)) {
    return DS41RT_STATUS_BUFFER_TOO_SMALL;
  }
  constexpr size_t threads = 256;
  const size_t blocks = (destination_bytes + threads - 1) / threads;
  pack_w4a16_scale_kernel<<<static_cast<unsigned int>(blocks), threads, 0,
                              reinterpret_cast<cudaStream_t>(cuda_stream)>>>(
      static_cast<const uint8_t*>(source.ptr), static_cast<uint8_t*>(destination.ptr),
      size_k, source_size_k, source_start_k, size_n, row_rotation, scale_factor);
  return status_from_cuda(cudaGetLastError());
}

extern "C" ds41rt_status_t ds41rt_cuda_b12x_gather_nvfp4_rows_bf16_async(
    ds41rt_device_buffer_t payload, size_t source_rows, size_t source_row_stride_bytes,
    ds41rt_device_buffer_t row_indices, ds41rt_device_buffer_t output, size_t rows,
    size_t hidden_dim, void* cuda_stream) {
  const size_t logical_row_bytes = hidden_dim / 2 + hidden_dim / 16;
  if (source_rows == 0 || rows == 0 || rows > source_rows || hidden_dim == 0 ||
      hidden_dim % 16 != 0 || source_row_stride_bytes < logical_row_bytes) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  if (!buffer_has_bytes(payload, source_rows * source_row_stride_bytes) ||
      !buffer_has_bytes(row_indices, rows * sizeof(uint32_t)) ||
      !buffer_has_bytes(output, rows * hidden_dim * sizeof(uint16_t))) {
    return DS41RT_STATUS_BUFFER_TOO_SMALL;
  }
  const size_t values = rows * hidden_dim;
  constexpr size_t threads = 256;
  const size_t blocks = (values + threads - 1) / threads;
  gather_nvfp4_rows_bf16_kernel<<<static_cast<unsigned int>(blocks), threads, 0,
                                    reinterpret_cast<cudaStream_t>(cuda_stream)>>>(
      static_cast<const uint8_t*>(payload.ptr), source_rows, source_row_stride_bytes,
      static_cast<const uint32_t*>(row_indices.ptr), static_cast<uint16_t*>(output.ptr), rows,
      hidden_dim);
  return status_from_cuda(cudaGetLastError());
}

extern "C" ds41rt_status_t ds41rt_cuda_b12x_dequantize_nvfp4_rows_bf16_async(
    ds41rt_device_buffer_t payload, size_t row_stride_bytes,
    ds41rt_device_buffer_t output, size_t rows, size_t hidden_dim,
    void* cuda_stream) {
  const size_t logical_row_bytes = hidden_dim / 2 + hidden_dim / 16;
  if (rows == 0 || hidden_dim == 0 || hidden_dim % 16 != 0 ||
      row_stride_bytes < logical_row_bytes ||
      rows > std::numeric_limits<size_t>::max() / row_stride_bytes ||
      rows > std::numeric_limits<size_t>::max() / hidden_dim ||
      rows * hidden_dim > std::numeric_limits<size_t>::max() / sizeof(uint16_t)) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  if (!buffer_has_bytes(payload, rows * row_stride_bytes) ||
      !buffer_has_bytes(output, rows * hidden_dim * sizeof(uint16_t))) {
    return DS41RT_STATUS_BUFFER_TOO_SMALL;
  }
  const size_t values = rows * hidden_dim;
  constexpr size_t threads = 256;
  const size_t blocks = (values + threads - 1) / threads;
  dequantize_nvfp4_row_payloads_bf16_kernel<<<
      static_cast<unsigned int>(blocks), threads, 0,
      reinterpret_cast<cudaStream_t>(cuda_stream)>>>(
      static_cast<const uint8_t*>(payload.ptr), row_stride_bytes,
      static_cast<uint16_t*>(output.ptr), rows, hidden_dim);
  return status_from_cuda(cudaGetLastError());
}

ds41rt_status_t launch_w4a16_decode_m1_nvfp4_grid(
    const ds41rt_b12x_spark_w4a16_moe_buffers_t* buffers,
    ds41rt_device_buffer_t input_payload, size_t input_payload_stride_bytes,
    ds41rt_device_buffer_t topk_ids, int grid_x, void* cuda_stream) {
  constexpr size_t input_payload_bytes = kB12xHidden / 2 + kB12xHidden / 16;
  if (grid_x < 0 || grid_x > kB12xW4a16DecodeResidentGridX) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  const ds41rt_status_t valid = validate_w4a16_moe_buffers(buffers, 1, kB12xTopK);
  if (valid != DS41RT_STATUS_OK) {
    return valid;
  }
  if (input_payload_stride_bytes < input_payload_bytes ||
      !buffer_has_bytes(input_payload, input_payload_stride_bytes) ||
      !buffer_has_bytes(topk_ids, kB12xTopK * sizeof(int32_t))) {
    return DS41RT_STATUS_BUFFER_TOO_SMALL;
  }
  const ds41rt_status_t initialized = ds41rt_cuda_b12x_spark_aot_init();
  if (initialized != DS41RT_STATUS_OK) {
    return initialized;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  ds41rt_status_t status = reset_w4a16_locks_async(buffers, stream);
  if (status != DS41RT_STATUS_OK) {
    return status;
  }
  constexpr int threads = 256;
  constexpr int blocks = static_cast<int>((kB12xHidden + threads - 1) / threads);
  dequantize_nvfp4_row_payload_bf16_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const uint8_t*>(input_payload.ptr),
      static_cast<uint16_t*>(buffers->input.ptr), kB12xHidden);
  status = status_from_cuda(cudaGetLastError());
  if (status != DS41RT_STATUS_OK) {
    return status;
  }
  ds41rt_b12x_spark_w4a16_moe_buffers_t launch_buffers = *buffers;
  launch_buffers.packed_route_indices = topk_ids;
  const int launch_status =
      grid_x == 0
          ? launch_w4a16_decode_m1(&launch_buffers, 1, stream)
          : launch_w4a16_decode_m1_grid(&launch_buffers, 1, grid_x, stream);
  status = check_aot_launch(
      launch_status, "B12X Spark packed W4A16 decode M1 launch failed");
  if (status != DS41RT_STATUS_OK) {
    return status;
  }
  sum_w4a16_topk_bf16_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const uint16_t*>(buffers->output.ptr),
      static_cast<uint16_t*>(buffers->output.ptr), 1, kB12xHidden);
  return status_from_cuda(cudaGetLastError());
}

extern "C" ds41rt_status_t ds41rt_cuda_b12x_spark_w4a16_decode_m1_nvfp4_async(
    const ds41rt_b12x_spark_w4a16_moe_buffers_t* buffers,
    ds41rt_device_buffer_t input_payload, size_t input_payload_stride_bytes,
    ds41rt_device_buffer_t topk_ids, void* cuda_stream) {
  return launch_w4a16_decode_m1_nvfp4_grid(
      buffers, input_payload, input_payload_stride_bytes, topk_ids,
      0, cuda_stream);
}

ds41rt_status_t launch_w4a16_decode_m1_fused_sum_nvfp4(
    const ds41rt_b12x_spark_w4a16_moe_buffers_t* buffers,
    ds41rt_device_buffer_t input_payload, size_t input_payload_stride_bytes,
    ds41rt_device_buffer_t topk_ids, void* cuda_stream) {
  constexpr size_t input_payload_bytes = kB12xHidden / 2 + kB12xHidden / 16;
  const ds41rt_status_t valid =
      validate_w4a16_moe_buffers(buffers, 1, kB12xTopK);
  if (valid != DS41RT_STATUS_OK) {
    return valid;
  }
  if (input_payload_stride_bytes < input_payload_bytes ||
      !buffer_has_bytes(input_payload, input_payload_stride_bytes) ||
      !buffer_has_bytes(topk_ids, kB12xTopK * sizeof(int32_t))) {
    return DS41RT_STATUS_BUFFER_TOO_SMALL;
  }
  const ds41rt_status_t initialized = ds41rt_cuda_b12x_spark_aot_init();
  if (initialized != DS41RT_STATUS_OK) {
    return initialized;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  ds41rt_status_t status = reset_w4a16_locks_async(buffers, stream);
  if (status != DS41RT_STATUS_OK) {
    return status;
  }
  constexpr int threads = 256;
  constexpr int blocks =
      static_cast<int>((kB12xHidden + threads - 1) / threads);
  dequantize_nvfp4_row_payload_bf16_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const uint8_t*>(input_payload.ptr),
      static_cast<uint16_t*>(buffers->input.ptr), kB12xHidden);
  status = status_from_cuda(cudaGetLastError());
  if (status != DS41RT_STATUS_OK) {
    return status;
  }
  ds41rt_b12x_spark_w4a16_moe_buffers_t launch_buffers = *buffers;
  launch_buffers.packed_route_indices = topk_ids;
  return check_aot_launch(
      launch_w4a16_decode_m1_fused_sum(&launch_buffers, 1, stream),
      "B12X Spark packed W4A16 decode M1 fused-sum launch failed");
}

extern "C" ds41rt_status_t
ds41rt_cuda_b12x_spark_w4a16_decode_m1_fused_sum_nvfp4_async(
    const ds41rt_b12x_spark_w4a16_moe_buffers_t* buffers,
    ds41rt_device_buffer_t input_payload, size_t input_payload_stride_bytes,
    ds41rt_device_buffer_t topk_ids, void* cuda_stream) {
  return launch_w4a16_decode_m1_fused_sum_nvfp4(
      buffers, input_payload, input_payload_stride_bytes, topk_ids,
      cuda_stream);
}

extern "C" ds41rt_status_t
ds41rt_cuda_b12x_spark_w4a16_m1_parity_m2_8_nvfp4_async(
    const ds41rt_b12x_spark_w4a16_moe_buffers_t* buffers,
    ds41rt_device_buffer_t input_payload, size_t input_payload_stride_bytes,
    ds41rt_device_buffer_t topk_ids, size_t rows, void* cuda_stream) {
  constexpr size_t input_payload_bytes = kB12xHidden / 2 + kB12xHidden / 16;
  W4A16LaunchFn launcher = w4a16_m1_parity_launcher(rows);
  if (launcher == nullptr || input_payload_stride_bytes < input_payload_bytes ||
      !buffer_has_bytes(input_payload, rows * input_payload_stride_bytes) ||
      !buffer_has_bytes(topk_ids, rows * kB12xTopK * sizeof(int32_t))) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  const ds41rt_status_t valid = validate_w4a16_moe_buffers(buffers, rows, kB12xTopK);
  if (valid != DS41RT_STATUS_OK) {
    return valid;
  }
  const ds41rt_status_t initialized = ds41rt_cuda_b12x_spark_aot_init();
  if (initialized != DS41RT_STATUS_OK) {
    return initialized;
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  ds41rt_status_t status = reset_w4a16_locks_async(buffers, stream);
  if (status != DS41RT_STATUS_OK) {
    return status;
  }
  constexpr size_t threads = 256;
  const size_t values = rows * kB12xHidden;
  const size_t blocks = (values + threads - 1) / threads;
  dequantize_nvfp4_row_payloads_bf16_kernel<<<static_cast<unsigned int>(blocks),
                                                threads, 0, stream>>>(
      static_cast<const uint8_t*>(input_payload.ptr), input_payload_stride_bytes,
      static_cast<uint16_t*>(buffers->input.ptr), rows, kB12xHidden);
  status = status_from_cuda(cudaGetLastError());
  if (status != DS41RT_STATUS_OK) {
    return status;
  }

  ds41rt_b12x_spark_w4a16_moe_buffers_t launch_buffers = *buffers;
  launch_buffers.packed_route_indices = topk_ids;
  status = check_aot_launch(
      launcher(&launch_buffers, rows, stream),
      "B12X Spark ordered direct-top-k W4A16 M=2..8 launch failed");
  if (status != DS41RT_STATUS_OK) {
    return status;
  }
  sum_w4a16_topk_bf16_kernel<<<static_cast<unsigned int>(blocks), threads, 0,
                                stream>>>(
      static_cast<const uint16_t*>(buffers->output.ptr),
      static_cast<uint16_t*>(buffers->input.ptr), rows, kB12xHidden);
  return status_from_cuda(cudaGetLastError());
}

extern "C" ds41rt_status_t
ds41rt_cuda_b12x_spark_w4a16_m1_parity_grouped_m2_8_nvfp4_async(
    const ds41rt_b12x_spark_w4a16_moe_buffers_t* buffers,
    ds41rt_device_buffer_t input_payload, size_t input_payload_stride_bytes,
    size_t rows, void* cuda_stream) {
  constexpr size_t input_payload_bytes = kB12xHidden / 2 + kB12xHidden / 16;
  W4A16LaunchFn launcher = w4a16_m1_parity_grouped_launcher(rows);
  if (launcher == nullptr || input_payload_stride_bytes < input_payload_bytes ||
      !buffer_has_bytes(input_payload, rows * input_payload_stride_bytes)) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  const ds41rt_status_t valid =
      validate_w4a16_moe_buffers(buffers, rows, kB12xTopK);
  if (valid != DS41RT_STATUS_OK) {
    return valid;
  }
  const ds41rt_status_t initialized = ds41rt_cuda_b12x_spark_aot_init();
  if (initialized != DS41RT_STATUS_OK) {
    return initialized;
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  ds41rt_status_t status = reset_w4a16_locks_async(buffers, stream);
  if (status != DS41RT_STATUS_OK) {
    return status;
  }
  constexpr size_t threads = 256;
  const size_t values = rows * kB12xHidden;
  const size_t blocks = (values + threads - 1) / threads;
  dequantize_nvfp4_row_payloads_bf16_kernel<<<static_cast<unsigned int>(blocks),
                                                threads, 0, stream>>>(
      static_cast<const uint8_t*>(input_payload.ptr),
      input_payload_stride_bytes,
      static_cast<uint16_t*>(buffers->input.ptr), rows, kB12xHidden);
  status = status_from_cuda(cudaGetLastError());
  if (status != DS41RT_STATUS_OK) {
    return status;
  }

  status = check_aot_launch(
      launcher(buffers, rows, stream),
      "B12X Spark grouped block-8 W4A16 M=2..8 launch failed");
  if (status != DS41RT_STATUS_OK) {
    return status;
  }
  sum_w4a16_topk_bf16_kernel<<<static_cast<unsigned int>(blocks), threads, 0,
                                stream>>>(
      static_cast<const uint16_t*>(buffers->output.ptr),
      static_cast<uint16_t*>(buffers->input.ptr), rows, kB12xHidden);
  return status_from_cuda(cudaGetLastError());
}

extern "C" ds41rt_status_t
ds41rt_cuda_b12x_spark_w4a16_m1_parity_grouped_wide_m2_8_nvfp4_async(
    const ds41rt_b12x_spark_w4a16_moe_buffers_t* buffers,
    ds41rt_device_buffer_t input_payload, size_t input_payload_stride_bytes,
    size_t rows, void* cuda_stream) {
  constexpr size_t input_payload_bytes = kB12xHidden / 2 + kB12xHidden / 16;
  W4A16LaunchFn launcher = w4a16_m1_parity_grouped_wide_launcher(rows);
  if (launcher == nullptr || input_payload_stride_bytes < input_payload_bytes ||
      !buffer_has_bytes(input_payload, rows * input_payload_stride_bytes)) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  const ds41rt_status_t valid =
      validate_w4a16_moe_buffers(buffers, rows, kB12xTopK);
  if (valid != DS41RT_STATUS_OK) {
    return valid;
  }
  const ds41rt_status_t initialized = ds41rt_cuda_b12x_spark_aot_init();
  if (initialized != DS41RT_STATUS_OK) {
    return initialized;
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  ds41rt_status_t status = reset_w4a16_locks_async(buffers, stream);
  if (status != DS41RT_STATUS_OK) {
    return status;
  }
  constexpr size_t threads = 256;
  const size_t values = rows * kB12xHidden;
  const size_t blocks = (values + threads - 1) / threads;
  dequantize_nvfp4_row_payloads_bf16_kernel<<<static_cast<unsigned int>(blocks),
                                                threads, 0, stream>>>(
      static_cast<const uint8_t*>(input_payload.ptr),
      input_payload_stride_bytes,
      static_cast<uint16_t*>(buffers->input.ptr), rows, kB12xHidden);
  status = status_from_cuda(cudaGetLastError());
  if (status != DS41RT_STATUS_OK) {
    return status;
  }

  status = check_aot_launch(
      launcher(buffers, rows, stream),
      "B12X Spark grouped-wide W4A16 M=2..8 launch failed");
  if (status != DS41RT_STATUS_OK) {
    return status;
  }
  sum_w4a16_topk_bf16_kernel<<<static_cast<unsigned int>(blocks), threads, 0,
                                stream>>>(
      static_cast<const uint16_t*>(buffers->output.ptr),
      static_cast<uint16_t*>(buffers->input.ptr), rows, kB12xHidden);
  return status_from_cuda(cudaGetLastError());
}

extern "C" ds41rt_status_t
ds41rt_cuda_b12x_spark_w4a16_decode_m1_nvfp4_grid_candidate_async(
    const ds41rt_b12x_spark_w4a16_moe_buffers_t* buffers,
    ds41rt_device_buffer_t input_payload, size_t input_payload_stride_bytes,
    ds41rt_device_buffer_t topk_ids, int grid_x, void* cuda_stream) {
  if (grid_x <= 0 || grid_x > kB12xW4a16DecodeResidentGridX) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  return launch_w4a16_decode_m1_nvfp4_grid(
      buffers, input_payload, input_payload_stride_bytes, topk_ids, grid_x,
      cuda_stream);
}

static ds41rt_status_t launch_w4a16_prefill_topk8_nvfp4(
    const ds41rt_b12x_spark_w4a16_moe_buffers_t* buffers,
    ds41rt_device_buffer_t input_payload, size_t input_payload_stride_bytes,
    size_t rows, ds41rt_device_buffer_t output_fp8,
    size_t output_fp8_row_stride_bytes, bool fuse_fp8_response,
    int grid_x, void* cuda_stream) {
  size_t capacity_rows = 2;
  while (capacity_rows < rows && capacity_rows < kB12xW4a16MaxRows) {
    capacity_rows *= 2;
  }
  constexpr size_t input_payload_bytes = kB12xHidden / 2 + kB12xHidden / 16;
  if (rows == 0 || rows > capacity_rows ||
      input_payload_stride_bytes < input_payload_bytes ||
      !buffer_has_bytes(input_payload, rows * input_payload_stride_bytes) ||
      (fuse_fp8_response &&
       (output_fp8_row_stride_bytes < kB12xHidden + sizeof(float) ||
        !buffer_has_bytes(output_fp8,
                          rows * output_fp8_row_stride_bytes)))) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  if (rows == 1 && w4a16_m1_fused_sum_enabled()) {
    const ds41rt_status_t fused_status =
        launch_w4a16_decode_m1_fused_sum_nvfp4(
            buffers, input_payload, input_payload_stride_bytes,
            buffers->packed_route_indices, cuda_stream);
    if (fused_status != DS41RT_STATUS_OK) {
      return fused_status;
    }
    if (fuse_fp8_response) {
      return ds41rt_cuda_bf16_rows_to_fp8_e4m3_row_scaled_async(
          static_cast<const uint16_t*>(buffers->output.ptr),
          static_cast<uint8_t*>(output_fp8.ptr), 1, kB12xHidden,
          output_fp8_row_stride_bytes, cuda_stream);
    }
    return status_from_cuda(cudaMemcpyAsync(
        buffers->input.ptr, buffers->output.ptr,
        kB12xHidden * sizeof(uint16_t), cudaMemcpyDeviceToDevice,
        reinterpret_cast<cudaStream_t>(cuda_stream)));
  }
  const ds41rt_status_t valid =
      validate_w4a16_moe_buffers(buffers, capacity_rows, kB12xTopK);
  if (valid != DS41RT_STATUS_OK) {
    return valid;
  }
  const ds41rt_status_t initialized = ds41rt_cuda_b12x_spark_aot_init();
  if (initialized != DS41RT_STATUS_OK) {
    return initialized;
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  ds41rt_status_t status = reset_w4a16_locks_async(buffers, stream);
  if (status != DS41RT_STATUS_OK) {
    return status;
  }
  constexpr size_t threads = 256;
  const size_t values = rows * kB12xHidden;
  const size_t blocks = (values + threads - 1) / threads;
  dequantize_nvfp4_row_payloads_bf16_kernel<<<static_cast<unsigned int>(blocks), threads, 0,
                                                stream>>>(
      static_cast<const uint8_t*>(input_payload.ptr), input_payload_stride_bytes,
      static_cast<uint16_t*>(buffers->input.ptr), rows, kB12xHidden);
  status = status_from_cuda(cudaGetLastError());
  if (status != DS41RT_STATUS_OK) {
    return status;
  }
  W4A16LaunchFn launcher = &launch_w4a16_prefill_m2048_topk8;
  if (capacity_rows == 2) {
    launcher = &launch_w4a16_prefill_m2_topk8;
  } else if (capacity_rows == 4) {
    launcher = &launch_w4a16_prefill_m4_topk8;
  } else if (capacity_rows == 8) {
    launcher = &launch_w4a16_prefill_m8_topk8;
  } else if (capacity_rows == 16) {
    launcher = &launch_w4a16_prefill_m16_topk8;
  } else if (capacity_rows == 32) {
    launcher = &launch_w4a16_prefill_m32_topk8;
  } else if (capacity_rows == 64) {
    launcher = &launch_w4a16_prefill_m64_topk8;
  } else if (capacity_rows == 128) {
    launcher = &launch_w4a16_prefill_m128_topk8;
  } else if (capacity_rows == 256) {
    launcher = &launch_w4a16_prefill_m256_topk8;
  } else if (capacity_rows == 512) {
    launcher = &launch_w4a16_prefill_m512_topk8;
  } else if (capacity_rows == 1024) {
    launcher = &launch_w4a16_prefill_m1024_topk8;
  }
  const W4A16GridLaunchFn grid_launcher =
      grid_x > 0 ? w4a16_prefill_grid_launcher(capacity_rows) : nullptr;
  if (grid_x > 0 && grid_launcher == nullptr) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  const int launch_status = grid_launcher != nullptr
                                ? grid_launcher(buffers, rows, grid_x, stream)
                                : launcher(buffers, rows, stream);
  status = check_aot_launch(
      launch_status, "B12X Spark packed W4A16 prefill top-k=8 launch failed");
  if (status != DS41RT_STATUS_OK) {
    return status;
  }
  if (fuse_fp8_response) {
    sum_w4a16_topk_bf16_to_fp8_row_scaled_kernel<<<
        static_cast<unsigned int>(rows), threads, 0, stream>>>(
        static_cast<const uint16_t*>(buffers->output.ptr),
        static_cast<uint8_t*>(output_fp8.ptr), rows,
        output_fp8_row_stride_bytes);
  } else {
    sum_w4a16_topk_bf16_kernel<<<static_cast<unsigned int>(blocks), threads, 0,
                                  stream>>>(
        static_cast<const uint16_t*>(buffers->output.ptr),
        static_cast<uint16_t*>(buffers->input.ptr), rows, kB12xHidden);
  }
  return status_from_cuda(cudaGetLastError());
}

extern "C" ds41rt_status_t ds41rt_cuda_b12x_spark_w4a16_prefill_topk8_nvfp4_async(
    const ds41rt_b12x_spark_w4a16_moe_buffers_t* buffers,
    ds41rt_device_buffer_t input_payload, size_t input_payload_stride_bytes,
    size_t rows, void* cuda_stream) {
  return launch_w4a16_prefill_topk8_nvfp4(
      buffers, input_payload, input_payload_stride_bytes, rows,
      ds41rt_device_buffer_t{}, 0, false, 0, cuda_stream);
}

extern "C" ds41rt_status_t
ds41rt_cuda_b12x_spark_w4a16_prefill_topk8_nvfp4_grid_candidate_async(
    const ds41rt_b12x_spark_w4a16_moe_buffers_t* buffers,
    ds41rt_device_buffer_t input_payload, size_t input_payload_stride_bytes,
    size_t rows, int grid_x, void* cuda_stream) {
  if (grid_x <= 0 || grid_x > kB12xW4a16DecodeMaxGridX) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  return launch_w4a16_prefill_topk8_nvfp4(
      buffers, input_payload, input_payload_stride_bytes, rows,
      ds41rt_device_buffer_t{}, 0, false, grid_x, cuda_stream);
}

extern "C" ds41rt_status_t
ds41rt_cuda_b12x_spark_w4a16_prefill_topk8_nvfp4_fp8_async(
    const ds41rt_b12x_spark_w4a16_moe_buffers_t* buffers,
    ds41rt_device_buffer_t input_payload, size_t input_payload_stride_bytes,
    size_t rows, ds41rt_device_buffer_t output_fp8,
    size_t output_fp8_row_stride_bytes, void* cuda_stream) {
  return launch_w4a16_prefill_topk8_nvfp4(
      buffers, input_payload, input_payload_stride_bytes, rows, output_fp8,
      output_fp8_row_stride_bytes, true, 0, cuda_stream);
}

extern "C" ds41rt_status_t ds41rt_cuda_b12x_spark_sum_topk8_bf16_async(
    ds41rt_device_buffer_t routed_bf16, ds41rt_device_buffer_t output_bf16,
    size_t rows, void* cuda_stream) {
  const size_t routed_values = rows * kB12xTopK * kB12xHidden;
  const size_t output_values = rows * kB12xHidden;
  if (rows == 0 || !buffer_has_bytes(routed_bf16, routed_values * sizeof(uint16_t)) ||
      !buffer_has_bytes(output_bf16, output_values * sizeof(uint16_t))) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  constexpr int threads = 256;
  const size_t blocks = (output_values + threads - 1) / threads;
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  sum_w4a16_topk_bf16_kernel<<<static_cast<unsigned int>(blocks), threads, 0,
                                stream>>>(
      static_cast<const uint16_t*>(routed_bf16.ptr),
      static_cast<uint16_t*>(output_bf16.ptr), rows, kB12xHidden);
  return status_from_cuda(cudaGetLastError());
}

extern "C" ds41rt_status_t
ds41rt_cuda_b12x_spark_sum_topk8_bf16_to_fp8_async(
    ds41rt_device_buffer_t routed_bf16, ds41rt_device_buffer_t output_fp8,
    size_t rows, size_t output_row_stride_bytes, void* cuda_stream) {
  const size_t routed_values = rows * kB12xTopK * kB12xHidden;
  const size_t minimum_output_row_bytes = kB12xHidden + sizeof(float);
  if (rows == 0 || output_row_stride_bytes < minimum_output_row_bytes ||
      !buffer_has_bytes(routed_bf16, routed_values * sizeof(uint16_t)) ||
      !buffer_has_bytes(output_fp8, rows * output_row_stride_bytes)) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  constexpr int threads = 256;
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  sum_w4a16_topk_bf16_to_fp8_row_scaled_kernel<<<
      static_cast<unsigned int>(rows), threads, 0, stream>>>(
      static_cast<const uint16_t*>(routed_bf16.ptr),
      static_cast<uint8_t*>(output_fp8.ptr), rows,
      output_row_stride_bytes);
  return status_from_cuda(cudaGetLastError());
}

extern "C" ds41rt_status_t ds41rt_cuda_b12x_spark_w4a16_top1_async(
    const ds41rt_b12x_spark_w4a16_moe_buffers_t* buffers, size_t rows,
    size_t capacity_rows, uint32_t expert_id, void* cuda_stream) {
  if (rows == 0 || rows > capacity_rows || expert_id >= kB12xExperts) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  const W4A16LaunchFn launcher = w4a16_top1_launcher(capacity_rows);
  if (launcher == nullptr) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  const ds41rt_status_t valid = validate_w4a16_moe_buffers(buffers, capacity_rows, 1);
  if (valid != DS41RT_STATUS_OK) {
    return valid;
  }
  const ds41rt_status_t initialized = ds41rt_cuda_b12x_spark_aot_init();
  if (initialized != DS41RT_STATUS_OK) {
    return initialized;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  ds41rt_status_t status = reset_w4a16_locks_async(buffers, stream);
  if (status != DS41RT_STATUS_OK) {
    return status;
  }
  constexpr size_t threads = 256;
  const size_t init_values = capacity_rows;
  const size_t blocks = (init_values + threads - 1) / threads;
  initialize_w4a16_top1_routes_kernel<<<static_cast<unsigned int>(blocks), threads, 0,
                                          stream>>>(
      static_cast<int32_t*>(buffers->packed_route_indices.ptr),
      static_cast<int32_t*>(buffers->block_expert_ids.ptr),
      static_cast<int32_t*>(buffers->packed_route_count.ptr),
      static_cast<float*>(buffers->topk_weights.ptr), rows, capacity_rows, expert_id,
      capacity_rows <= 8);
  status = status_from_cuda(cudaGetLastError());
  if (status != DS41RT_STATUS_OK) {
    return status;
  }
  return check_aot_launch(
      launcher(buffers, rows, stream),
      "B12X Spark packed W4A16 top-k=1 launch failed");
}

extern "C" ds41rt_status_t ds41rt_cuda_b12x_spark_w4a16_top1_grid_candidate_async(
    const ds41rt_b12x_spark_w4a16_moe_buffers_t* buffers, size_t rows,
    size_t capacity_rows, uint32_t expert_id, int grid_x, void* cuda_stream) {
  if (rows == 0 || rows > capacity_rows || expert_id >= kB12xExperts || grid_x <= 0 ||
      grid_x > kB12xW4a16DecodeMaxGridX) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  const W4A16GridLaunchFn launcher = w4a16_top1_grid_launcher(capacity_rows);
  if (launcher == nullptr) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  const ds41rt_status_t valid = validate_w4a16_moe_buffers(buffers, capacity_rows, 1);
  if (valid != DS41RT_STATUS_OK) {
    return valid;
  }
  const ds41rt_status_t initialized = ds41rt_cuda_b12x_spark_aot_init();
  if (initialized != DS41RT_STATUS_OK) {
    return initialized;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  ds41rt_status_t status = reset_w4a16_locks_async(buffers, stream);
  if (status != DS41RT_STATUS_OK) {
    return status;
  }
  constexpr size_t threads = 256;
  const size_t blocks = (capacity_rows + threads - 1) / threads;
  initialize_w4a16_top1_routes_kernel<<<static_cast<unsigned int>(blocks), threads, 0,
                                          stream>>>(
      static_cast<int32_t*>(buffers->packed_route_indices.ptr),
      static_cast<int32_t*>(buffers->block_expert_ids.ptr),
      static_cast<int32_t*>(buffers->packed_route_count.ptr),
      static_cast<float*>(buffers->topk_weights.ptr), rows, capacity_rows, expert_id,
      capacity_rows <= 8);
  status = status_from_cuda(cudaGetLastError());
  if (status != DS41RT_STATUS_OK) {
    return status;
  }
  return check_aot_launch(
      launcher(buffers, rows, grid_x, stream),
      "B12X Spark packed W4A16 top-k=1 grid candidate launch failed");
}
