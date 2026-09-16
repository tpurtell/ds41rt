#include <cuda_runtime.h>
#include <math_constants.h>
#include <curand_kernel.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "ds41rt_v41_dspark.h"
namespace {
#if defined(DS41RT_V41_VOCAB_ROW_EXPERIMENT)
// Upstream's vocabulary row-reduction strategy, retaining native FP32 logits.
// One block per vocabulary row; only selected for a single input row.
__global__ void vocabulary_row(const __nv_bfloat16* input,
    const __nv_bfloat16* weight, float* output) {
  const uint64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  float sum = 0;
  #pragma unroll
  for (int col = tid; col < 5120; col += 256)
    sum = fmaf(__bfloat162float(input[col]),
               __bfloat162float(weight[row * 5120 + col]), sum);
  for (int offset = 16; offset; offset >>= 1)
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  __shared__ float partials[8];
  if (tid % 32 == 0) partials[tid / 32] = sum;
  __syncthreads();
  if (tid < 32) {
    sum = tid < 8 ? partials[tid] : 0;
    for (int offset = 16; offset; offset >>= 1)
      sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (tid == 0) output[row] = sum;
  }
}
#endif
__device__ float warp_sum(float value) {
  for (int offset = 16; offset; offset >>= 1)
    value += __shfl_down_sync(0xffffffffu, value, offset);
  return value;
}
// Fuse concatenation and BF16 -> FP32 promotion into the projection. The
// reference promotes checkpoint confidence weights and both inputs to FP32.
__global__ void confidence(const __nv_bfloat16* hidden,
    const __nv_bfloat16* markov, const __nv_bfloat16* weight, float* output) {
  const uint64_t row = blockIdx.x;
  const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
  float sum = 0;
  for (int col = tid; col < 5376; col += 256) {
    const float x = __bfloat162float(col < 5120 ? hidden[row * 5120 + col]
        : markov[row * 256 + col - 5120]);
    sum = fmaf(x, __bfloat162float(weight[col]), sum);
  }
  sum = warp_sum(sum);
  __shared__ float partials[8];
  if (lane == 0) partials[warp] = sum;
  __syncthreads();
  if (warp == 0) {
    sum = warp_sum(lane < 8 ? partials[lane] : 0);
    if (lane == 0) output[row] = sum;
  }
}
bool span(const void* p, uint64_t bytes, uint64_t alignment) {
  const auto address = reinterpret_cast<uintptr_t>(p);
  return address && address % alignment == 0 && address <= UINTPTR_MAX - bytes;
}
bool disjoint(const void* a, uint64_t na, const void* b, uint64_t nb) {
  const auto x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
  return x + na <= y || y + nb <= x;
}
}
extern "C" int32_t ds41rt_v41_dspark_confidence(const uint16_t* hidden,
    const uint16_t* markov, const uint16_t* weight, float* output,
    int32_t rows, void* stream) {
  if (rows <= 0 || rows > 4096) return cudaErrorInvalidValue;
  const uint64_t h = uint64_t(rows) * 10240, m = uint64_t(rows) * 512;
  const uint64_t w = 10752, o = uint64_t(rows) * 4;
  if (!span(hidden, h, 2) || !span(markov, m, 2) || !span(weight, w, 2) ||
      !span(output, o, 4) || !disjoint(hidden, h, output, o) ||
      !disjoint(markov, m, output, o) || !disjoint(weight, w, output, o))
    return cudaErrorInvalidValue;
  confidence<<<rows, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(hidden),
      reinterpret_cast<const __nv_bfloat16*>(markov),
      reinterpret_cast<const __nv_bfloat16*>(weight), output);
  return cudaGetLastError();
}

#include <cublas_v2.h>
#include <new>
namespace {
struct MarkovHandle { cublasHandle_t blas; int device; void* workspace; int width; int max_rows; int vocab_rows; };
constexpr uint64_t kMarkovWorkspace = 4 * 1024 * 1024;
int32_t blas_status(cublasStatus_t status) { return status == CUBLAS_STATUS_SUCCESS ? 0 : -int32_t(status); }
}
static int32_t create_head(void* workspace, uint64_t bytes, void** output, int width, int max_rows, int vocab_rows = 129280) {
  if (!output) return cudaErrorInvalidValue;
  *output = nullptr;
  if (vocab_rows < 1 || vocab_rows > 129280) return cudaErrorInvalidValue;
  if (bytes < kMarkovWorkspace || !span(workspace, kMarkovWorkspace, 256)) return cudaErrorInvalidValue;
  auto* handle = new (std::nothrow) MarkovHandle{};
  if (!handle) return cudaErrorMemoryAllocation;
  auto status = cudaGetDevice(&handle->device);
  if (status != cudaSuccess) { delete handle; return status; }
  auto created = cublasCreate(&handle->blas);
  if (created != CUBLAS_STATUS_SUCCESS) { delete handle; return blas_status(created); }
  handle->workspace = workspace;
  handle->width = width;
  handle->max_rows = max_rows;
  handle->vocab_rows = vocab_rows;
  *output = handle;
  return 0;
}
extern "C" int32_t ds41rt_v41_markov_create(void* workspace, uint64_t bytes, void** output) {
  return create_head(workspace, bytes, output, 256, 16);
}
extern "C" int32_t ds41rt_v41_vocabulary_head_create(void* workspace, uint64_t bytes, void** output) {
  return create_head(workspace, bytes, output, 5120, 128);
}
extern "C" int32_t ds41rt_v41_vocabulary_shard_create(void* workspace, uint64_t bytes,
    int32_t vocab_rows, void** output) {
  return create_head(workspace, bytes, output, 5120, 128, vocab_rows);
}
extern "C" int32_t ds41rt_v41_markov_destroy(void* opaque) {
  if (!opaque) return cudaErrorInvalidValue;
  auto* handle = static_cast<MarkovHandle*>(opaque);
  const auto status = cublasDestroy(handle->blas);
  delete handle;
  return blas_status(status);
}
static int32_t launch_head(void* opaque, const uint16_t* embedding,
    const uint16_t* weight, float* logits, int32_t rows, void* stream, int width) {
  if (!opaque || rows < 1) return cudaErrorInvalidValue;
  auto* handle = static_cast<MarkovHandle*>(opaque);
  if (handle->width != width || rows > handle->max_rows) return cudaErrorInvalidValue;
  int device;
  auto status = cudaGetDevice(&device);
  if (status != cudaSuccess) return status;
  if (device != handle->device) return cudaErrorInvalidDevice;
  const uint64_t e = uint64_t(rows) * width * 2, w = uint64_t(handle->vocab_rows) * width * 2;
  const uint64_t o = uint64_t(rows) * handle->vocab_rows * 4;
  if (!span(embedding, e, 2) || !span(weight, w, 2) || !span(logits, o, 4) ||
      !disjoint(logits, o, embedding, e) || !disjoint(logits, o, weight, w) ||
      !disjoint(handle->workspace, kMarkovWorkspace, embedding, e) ||
      !disjoint(handle->workspace, kMarkovWorkspace, weight, w) ||
      !disjoint(handle->workspace, kMarkovWorkspace, logits, o)) return cudaErrorInvalidValue;
#if defined(DS41RT_V41_VOCAB_ROW_EXPERIMENT)
  if (width == 5120 && rows == 1) {
    vocabulary_row<<<handle->vocab_rows, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const __nv_bfloat16*>(embedding),
        reinterpret_cast<const __nv_bfloat16*>(weight), logits);
    return cudaGetLastError();
  }
#endif
  auto result = cublasSetStream(handle->blas, reinterpret_cast<cudaStream_t>(stream));
  if (result != CUBLAS_STATUS_SUCCESS) return blas_status(result);
  // SetStream resets the workspace: rebind this wave's owned storage afterward.
  result = cublasSetWorkspace(handle->blas, handle->workspace, kMarkovWorkspace);
  if (result != CUBLAS_STATUS_SUCCESS) return blas_status(result);
  const float alpha = 1, beta = 0;
  // Vocabulary logits follow the reference's FP32-promoted projection. The
  // default BF16 tensor-op path exceeds its error bound on real head weights;
  // require pedantic FP32 accumulation while retaining BF16 resident storage.
  const auto compute = width == 5120 ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F;
  const auto algorithm = width == 5120 ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  return blas_status(cublasGemmEx(handle->blas, CUBLAS_OP_T, CUBLAS_OP_N,
      handle->vocab_rows, rows, width, &alpha, weight, CUDA_R_16BF, width,
      embedding, CUDA_R_16BF, width, &beta, logits, CUDA_R_32F, handle->vocab_rows,
      compute, algorithm));
}

extern "C" int32_t ds41rt_v41_markov_launch(void* handle, const uint16_t* input,
    const uint16_t* weight, float* output, int32_t rows, void* stream) {
  return launch_head(handle, input, weight, output, rows, stream, 256);
}
extern "C" int32_t ds41rt_v41_vocabulary_head_launch(void* handle, const uint16_t* input,
    const uint16_t* weight, float* output, int32_t rows, void* stream) {
  return launch_head(handle, input, weight, output, rows, stream, 5120);
}

namespace {
__global__ void vocabulary_merge_greedy(const uint32_t* ids0, const float* scores0,
    const uint32_t* ids1, const float* scores1, uint32_t* ids, float* scores,
    int rows, uint32_t split) {
  const int row = threadIdx.x;
  if (row >= rows) return;
  const auto a = ids0[row], b = ids1[row];
  const float x = scores0[row], y = scores1[row];
  if (a >= split || b >= 129280u - split || !isfinite(x) || !isfinite(y)) {
    ids[row] = UINT32_MAX;
    scores[row] = CUDART_NAN_F;
  } else {
    const bool first = x >= y;
    ids[row] = first ? a : split + b;
    scores[row] = first ? x : y;
  }
}
}
extern "C" int32_t ds41rt_v41_vocabulary_merge_greedy(const uint32_t* ids0,
    const float* scores0, const uint32_t* ids1, const float* scores1,
    uint32_t* ids, float* scores, int32_t rows, int32_t split, void* stream) {
  if (rows < 1 || rows > 128 || split < 1 || split >= 129280) return cudaErrorInvalidValue;
  const uint64_t bytes = uint64_t(rows) * 4;
  const void* buffers[] = {ids0, scores0, ids1, scores1, ids, scores};
  for (int i = 0; i < 6; ++i) {
    if (!span(buffers[i], bytes, 4)) return cudaErrorInvalidValue;
    if (i >= 4) for (int j = 0; j < i; ++j)
      if (!disjoint(buffers[i], bytes, buffers[j], bytes)) return cudaErrorInvalidValue;
  }
  vocabulary_merge_greedy<<<1, 128, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      ids0, scores0, ids1, scores1, ids, scores, rows, uint32_t(split));
  return cudaGetLastError();
}

namespace {
// Position-local logits remain raw for verification. Sampling uses log-space
// exponential racing, equivalent in distribution to softmax(logits/T)/Exp(1).
constexpr int kDraftVocab = 129280;
constexpr int kDraftTile = 512;
constexpr int kDraftTiles = (kDraftVocab + kDraftTile - 1) / kDraftTile;
static_assert(kDraftTiles <= 256 && 2 * kDraftTiles < kDraftTile);

__device__ void draft_argmax(float& best, uint32_t& id) {
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  for (int offset = 16; offset; offset >>= 1) {
    const float score = __shfl_down_sync(0xffffffffu, best, offset);
    const uint32_t candidate = __shfl_down_sync(0xffffffffu, id, offset);
    if (score > best || (score == best && candidate < id)) { best = score; id = candidate; }
  }
  __shared__ float scores[8];
  __shared__ uint32_t indices[8];
  if (lane == 0) { scores[warp] = best; indices[warp] = id; }
  __syncthreads();
  if (warp == 0) {
    best = lane < 8 ? scores[lane] : -CUDART_INF_F;
    id = lane < 8 ? indices[lane] : UINT32_MAX;
    for (int offset = 16; offset; offset >>= 1) {
      const float score = __shfl_down_sync(0xffffffffu, best, offset);
      const uint32_t candidate = __shfl_down_sync(0xffffffffu, id, offset);
      if (score > best || (score == best && candidate < id)) { best = score; id = candidate; }
    }
  }
}

__global__ void draft_step_tiles(const float* shared, const float* bias,
    const uint64_t* rng, const float* temperatures, float* adjusted, int position) {
  const int row = blockIdx.y, tile = blockIdx.x, tid = threadIdx.x;
  const float temperature = temperatures[row];
  curandStatePhilox4_32_10_t state;
  if (temperature != 0) {
    // Preserve the original 256 logical RNG lanes: each tile skips exactly
    // the uint32 draws consumed by earlier columns in that lane.
    curand_init(rng[uint64_t(row)*2], rng[uint64_t(row)*2+1] + uint64_t(position)*256 + tid,
        uint64_t(tile) * (kDraftTile / 256), &state);
  }
  float best = -CUDART_INF_F;
  uint32_t id = UINT32_MAX;
  for (uint32_t col = tile * kDraftTile + tid;
       col < kDraftVocab && col < (tile + 1) * kDraftTile; col += 256) {
    const uint64_t offset = uint64_t(row) * kDraftVocab + col;
    const float value = shared[offset] + bias[offset];
    // Reserve a small prefix for partial argmax results. The finish kernel
    // restores its raw logits after reading all partials; no extra allocation.
    if (col >= 2 * kDraftTiles) adjusted[offset] = value;
    float score = value;
    if (temperature != 0) {
      const float uniform = (float(curand(&state) >> 9) + 0.5f) * 0x1p-23f;
      const float exponential = -logf(uniform);
      score = value / fmaxf(temperature, 1e-5f) - logf(exponential);
    }
    if (score > best || (score == best && col < id)) { best = score; id = col; }
  }
  draft_argmax(best, id);
  if (tid == 0) {
    const uint64_t base = uint64_t(row) * kDraftVocab;
    adjusted[base + 2 * tile] = best;
    adjusted[base + 2 * tile + 1] = __uint_as_float(id);
  }
}

__global__ void draft_step_finish(const float* shared, const float* bias,
    float* adjusted, uint32_t* tokens) {
  const int row = blockIdx.x, tid = threadIdx.x;
  const uint64_t base = uint64_t(row) * kDraftVocab;
  float best = tid < kDraftTiles ? adjusted[base + 2 * tid] : -CUDART_INF_F;
  uint32_t id = tid < kDraftTiles
      ? __float_as_uint(adjusted[base + 2 * tid + 1]) : UINT32_MAX;
  // The block barrier inside argmax ensures every prefix read finishes before
  // any thread restores it. Both kernels run on the caller's ordered stream.
  draft_argmax(best, id);
  if (tid == 0) tokens[row] = id;
  for (int col = tid; col < 2 * kDraftTiles; col += 256)
    adjusted[base + col] = shared[base + col] + bias[base + col];
}

}
extern "C" int32_t ds41rt_v41_draft_step_rng(const float* shared, const float* bias,
    const uint64_t* rng, const float* temperatures, float* adjusted, uint32_t* tokens,
    int32_t rows, int32_t position, void* stream) {
  if (rows < 1 || rows > 16 || position < 0 || position > 6) return cudaErrorInvalidValue;
  const uint64_t logits = uint64_t(rows) * 129280 * 4, small = uint64_t(rows) * 4;
  const void* pointers[] = {shared, bias, rng, temperatures, adjusted, tokens};
  const uint64_t bytes[] = {logits, logits, small*4, small, logits, small};
  for (int i = 0; i < 6; ++i) if (!span(pointers[i], bytes[i], i == 2 ? 8 : 4)) return cudaErrorInvalidValue;
  for (int i = 4; i < 6; ++i)
    for (int j = 0; j < i; ++j)
      if (!disjoint(pointers[i], bytes[i], pointers[j], bytes[j])) return cudaErrorInvalidValue;
  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  draft_step_tiles<<<dim3(kDraftTiles, rows), 256, 0, cuda_stream>>>(
      shared, bias, rng, temperatures, adjusted, position);
  auto status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  draft_step_finish<<<rows, 256, 0, cuda_stream>>>(shared, bias, adjusted, tokens);
  return cudaGetLastError();
}
