#include "common.h"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cmath>

namespace {
constexpr int kEngramDim = 5120;
constexpr int kEngramCopies = 4;
constexpr int kEngramThreads = 256;
constexpr int kEngramItems = kEngramDim / kEngramThreads;

__device__ float warp_sum(float value) {
  for (int offset = 16; offset; offset >>= 1)
    value += __shfl_down_sync(0xffffffffu, value, offset);
  return value;
}

// One block per (token, residual stream), keeping the residual in registers.
__global__ void engram_gate_kernel(const uint16_t* x, const uint16_t* kv,
    const uint16_t* q_weight, const uint16_t* k_weight, const uint8_t* text_mask,
    uint16_t* out) {
  const size_t row = blockIdx.x;
  const size_t copy = blockIdx.y;
  const int tid = threadIdx.x;
  const size_t x_base = (row * kEngramCopies + copy) * kEngramDim;
  if (text_mask && !text_mask[row]) {
    for (int col = tid; col < kEngramDim; col += kEngramThreads)
      out[x_base + col] = x[x_base + col];
    return;
  }
  const size_t key_base = (row * (kEngramCopies + 1) + copy) * kEngramDim;
  const size_t value_base = (row * (kEngramCopies + 1) + kEngramCopies) * kEngramDim;
  float residual[kEngramItems];
  float xx = 0, kk = 0, dot = 0;
#pragma unroll
  for (int i = 0; i < kEngramItems; ++i) {
    const int col = tid + i * kEngramThreads;
    const float h = bf16_to_f32(x[x_base + col]);
    const float k = bf16_to_f32(kv[key_base + col]);
    const float weight = bf16_to_f32(q_weight[copy * kEngramDim + col]) *
                         bf16_to_f32(k_weight[copy * kEngramDim + col]);
    residual[i] = h;
    xx = __fadd_rn(xx, __fmul_rn(h, h));
    kk = __fadd_rn(kk, __fmul_rn(k, k));
    dot = __fadd_rn(dot, __fmul_rn(__fmul_rn(h, weight), k));
  }
  xx = warp_sum(xx); kk = warp_sum(kk); dot = warp_sum(dot);
  __shared__ float sums[3][8];
  const int lane = tid % 32, warp = tid / 32;
  if (lane == 0) { sums[0][warp] = xx; sums[1][warp] = kk; sums[2][warp] = dot; }
  __syncthreads();
  if (warp == 0) {
    xx = warp_sum(lane < 8 ? sums[0][lane] : 0);
    kk = warp_sum(lane < 8 ? sums[1][lane] : 0);
    dot = warp_sum(lane < 8 ? sums[2][lane] : 0);
    if (lane == 0) {
      const float rstd = rsqrtf(xx / kEngramDim + 1e-20f) * rsqrtf(kk / kEngramDim + 1e-20f);
      dot = __fmul_rn(__fmul_rn(dot, rstd), rsqrtf(float(kEngramDim)));
      sums[0][0] = sigmoid_f32(copysignf(sqrtf(fmaxf(fabsf(dot), 1e-6f)), dot));
    }
  }
  __syncthreads();
  const float gate = sums[0][0];
#pragma unroll
  for (int i = 0; i < kEngramItems; ++i) {
    const int col = tid + i * kEngramThreads;
    const float value = bf16_to_f32(kv[value_base + col]);
    out[x_base + col] = __bfloat16_as_ushort(__float2bfloat16_rn(
        __fadd_rn(residual[i], __fmul_rn(gate, value))));
  }
}
}  // namespace

extern "C" ds41rt_status_t ds41rt_cuda_engram_gate_bf16_async(
    const uint16_t* x, const uint16_t* kv, const uint16_t* q_weight,
    const uint16_t* k_weight, const uint8_t* text_mask, uint16_t* out,
    int rows, void* cuda_stream) {
  if (!x || !kv || !q_weight || !k_weight || !out || rows <= 0) {
    ds41rt_set_last_error_message("engram gate requires non-null tensors and positive rows");
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  engram_gate_kernel<<<dim3(rows, kEngramCopies), kEngramThreads, 0,
      reinterpret_cast<cudaStream_t>(cuda_stream)>>>(x, kv, q_weight, k_weight, text_mask, out);
  return status_from_cuda(cudaGetLastError());
}

namespace {
__global__ void engram_dequant_kernel(const uint8_t* weights, const uint8_t* scales, uint16_t* out) {
  const size_t row = blockIdx.x;
  const size_t col = threadIdx.x;
  __nv_fp8_e4m3 value;
  value.__x = weights[row * 256 + col];
  const uint8_t exponent = scales[row * 8 + col / 32];
  const float scale = exponent == 255 ? CUDART_NAN_F :
      __int_as_float(exponent == 0 ? 0x00400000u : uint32_t(exponent) << 23);
  out[row * 256 + col] = __bfloat16_as_ushort(__float2bfloat16_rn(static_cast<float>(value) * scale));
}
}
extern "C" ds41rt_status_t ds41rt_cuda_engram_dequant_bf16_async(
    const uint8_t* weights, const uint8_t* scales, uint16_t* out, int hash_rows, void* cuda_stream) {
  if (!weights || !scales || !out || hash_rows <= 0) {
    ds41rt_set_last_error_message("engram dequant requires non-null tensors and positive rows");
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  engram_dequant_kernel<<<hash_rows, 256, 0, reinterpret_cast<cudaStream_t>(cuda_stream)>>>(weights, scales, out);
  return status_from_cuda(cudaGetLastError());
}

namespace {
__global__ void engram_nvfp4_dequant_kernel(const uint8_t* weights, const uint8_t* scales,
    float global_scale, uint16_t* out) {
  const size_t row = blockIdx.x;
  const size_t col = threadIdx.x;
  const uint8_t packed = weights[row * 128 + col / 2];
  const uint8_t code = (packed >> ((col & 1) * 4)) & 15;
  const float magnitudes[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  const float value = (code & 8) ? -magnitudes[code & 7] : magnitudes[code & 7];
  __nv_fp8_e4m3 scale;
  scale.__x = scales[row * 16 + col / 16];
  out[row * 256 + col] = __bfloat16_as_ushort(__float2bfloat16_rn(
      __fmul_rn(__fmul_rn(value, static_cast<float>(scale)), global_scale)));
}
}
extern "C" ds41rt_status_t ds41rt_cuda_engram_nvfp4_dequant_bf16_async(
    const uint8_t* weights, const uint8_t* scales, float global_scale,
    uint16_t* out, int hash_rows, void* cuda_stream) {
  if (!weights || !scales || !out || hash_rows <= 0 || !std::isfinite(global_scale) || global_scale <= 0.f) {
    ds41rt_set_last_error_message("NVFP4 engram dequant requires tensors, positive rows and a finite positive global scale");
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  engram_nvfp4_dequant_kernel<<<hash_rows, 256, 0,
      reinterpret_cast<cudaStream_t>(cuda_stream)>>>(weights, scales, global_scale, out);
  return status_from_cuda(cudaGetLastError());
}
