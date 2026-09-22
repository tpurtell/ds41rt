/* ds41rt v4.1 GPU target-sampler, chunk 1: mask application, finiteness
 * discipline, temperature scaling, scaled maximum, min_p survivor count and the
 * greedy / constrained-greedy device argmax (K1).
 *
 * See `v41_sampling_gpu.h` for the device ABI and
 * `docs/gpu-sampling-design.md` §4.0-§4.1, §5, §11, §12.1-§12.2, §14 chunk 1 for
 * the contract. The CPU sampler in
 * `rust/crates/ds41rt-core/src/target_sampling.rs` is the correctness oracle and
 * is deliberately untouched.
 *
 * Shared invariants honoured here (design §4.0):
 *   - one CTA per row, `kBlock` threads, intra-CTA `__syncthreads()` and fixed
 *     binary-tree combines only: no grid sync, no cooperative launch, no host
 *     callback, no per-call allocation, so the launch is graph-capture legal;
 *   - the mask is applied FIRST, as a predicate, without mutating `b(4)`. No
 *     filter ever reads a masked-out token, so a masked non-finite logit is
 *     legal for a stochastic row. The logits buffer stays byte-identical,
 *     which the retained-frontier download depends on;
 *   - temperature is a multiply by the reciprocal, never a division;
 *   - `min_p` is evaluated in logit space against the host-precomputed
 *     `ln_min_p`, so the device performs exactly one f32 add and never calls
 *     `logf`. This is a hard requirement of design §6.3a;
 *   - every loop is bounded by `vocab`, so no code path can emit an id
 *     `>= vocab`, independent of the mask's unused high bits (§5.3 rule 1).
 *
 * Compilation: this source is NOT given `-use_fast_math` and must not be.
 * Subnormal scaled values are load-bearing (the `negpos_zero_subnormal` oracle
 * row). The default `native/CMakeLists.txt` flags apply.
 */

#include "common.h"
#include "v41_sampling_gpu.h"

#include <cuda_runtime_api.h>
#include <math_constants.h>

#include <cstdint>
#include <limits>

namespace {

/* `kBlock` (256) comes from `common.h`, the tree-wide sampler block size. */

/* One CTA per row. Per-thread state lives in registers; the reductions use
 * these shared arrays and a fixed binary-tree combine. */
struct Shared {
  float score[kBlock];       /* greedy argmax score */
  uint32_t id[kBlock];       /* greedy argmax token id */
  float scaled[kBlock];      /* stochastic scaled maximum */
  float inv_temperature[kBlock];
  uint32_t allowed[kBlock];  /* stochastic allowed-token count */
  uint32_t survivors[kBlock];/* stochastic min_p survivor count */
  uint32_t nonfinite[kBlock];/* lowest offending token id, or NO_DETAIL */
};

__device__ __forceinline__ bool row_allowed(const ds41rt_v41_sampler_row_t& row,
                                            const uint32_t* mask_words,
                                            size_t mask_words_per_row, size_t mask_rows,
                                            size_t token) {
  /* A row is unconstrained when it says so (`FLAG_NO_MASK`), when it carries the
   * design's `0xFFFFFFFF` sentinel, or when no arena was supplied at all.
   *
   * `mask_row >= mask_rows` is also treated as unconstrained. The ABI has no
   * arena-length field, so `rows` (the batch row count, which is the arena's row
   * count by construction) is the only bound the kernel can check; out-of-range
   * values cannot be reported through the frozen status codes, so the kernel
   * chooses the memory-safe fallback instead of reading `mask_row * words` past
   * the arena. The host-side validator still rejects such a row outright. */
  if ((row.flags & DS41RT_V41_SAMPLER_FLAG_NO_MASK) != 0u ||
      row.mask_row == DS41RT_V41_SAMPLER_NO_MASK_ROW || mask_words == nullptr ||
      static_cast<size_t>(row.mask_row) >= mask_rows) {
    return true;
  }
  const size_t index = static_cast<size_t>(row.mask_row) * mask_words_per_row + token / 32u;
  return ((mask_words[index] >> (token % 32u)) & 1u) != 0u;
}

/* Fixed binary-tree reductions. Every row pays exactly the same number of
 * steps, so a row's result cannot depend on the wave, the lane or a peer. */
__device__ __forceinline__ float tree_max(float* values, int tid) {
  for (int stride = kBlock / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      values[tid] = fmaxf(values[tid], values[tid + stride]);
    }
    __syncthreads();
  }
  return values[0];
}

__device__ __forceinline__ uint32_t tree_add_u32(uint32_t* values, int tid) {
  for (int stride = kBlock / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      values[tid] += values[tid + stride];
    }
    __syncthreads();
  }
  return values[0];
}

__device__ __forceinline__ uint32_t tree_min_u32(uint32_t* values, int tid) {
  for (int stride = kBlock / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      const uint32_t other = values[tid + stride];
      if (other < values[tid]) {
        values[tid] = other;
      }
    }
    __syncthreads();
  }
  return values[0];
}

/* Greedy argmax combine: strict `>` with the lowest id winning an exact tie,
 * matching the CPU's `argmax_allowed` (`target_sampling.rs:259-262`) and the
 * legacy device argmax (`sampling.cu:583-587`). */
__device__ __forceinline__ void tree_argmax(float* scores, uint32_t* ids, int tid) {
  for (int stride = kBlock / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      const float other_score = scores[tid + stride];
      const uint32_t other_id = ids[tid + stride];
      if (other_score > scores[tid] ||
          (other_score == scores[tid] && other_id < ids[tid])) {
        scores[tid] = other_score;
        ids[tid] = other_id;
      }
    }
    __syncthreads();
  }
}

/* K1. Grid: `rows` CTAs; block: `kBlock`; the mask is the first predicate.
 *
 * Greedy rows (including constrained greedy) take the strict whole-row branch:
 * finiteness is checked over every token BEFORE the mask test, exactly as
 * `scores.rs::argmax` does, so a masked non-finite logit still errors. A greedy
 * row never computes `1.0f / temperature`, never calls `expf` and consumes no
 * draw, so `temperature = 0.0` -- the normal greedy case -- cannot produce
 * `inf`/`NaN`.
 *
 * Stochastic rows take the permissive branch: only allowed tokens are read, so a
 * masked non-finite logit is legal and unread. Status precedence is the CPU's:
 * non-finite beats EMPTY_CANDIDATES, which beats an invalid temperature.
 */
__global__ void v41_sample_prepare_kernel(
    const float* logits, size_t rows, size_t vocab, size_t logits_stride,
    const ds41rt_v41_sampler_row_t* params, const uint32_t* mask_words,
    size_t mask_words_per_row, uint32_t* out_indices, uint32_t* out_status,
    uint32_t* out_status_detail, float* out_scores, float* out_total,
    uint32_t* out_nucleus_count, ds41rt_v41_sampler_scratch_t* scratch) {
  __shared__ Shared shared;

  const size_t block_row = blockIdx.x;
  const int tid = threadIdx.x;
  if (block_row >= rows) {
    return;
  }

  const ds41rt_v41_sampler_row_t row = params[block_row];
  const size_t output_row = static_cast<size_t>(row.output_row);
  const float* row_logits = logits + block_row * logits_stride;

  /* Greedy is re-derived here and ORed with the host bit, so a host bug cannot
   * silently turn a stochastic row greedy. `top_k == 1` is greedy even at a
   * sampling temperature (contract §7.1.2). */
  const bool greedy = (row.temperature < 1e-5f) || (row.top_k == 1u) ||
      ((row.flags & DS41RT_V41_SAMPLER_FLAG_GREEDY) != 0u);
  const float inv_temperature = greedy ? 0.0f : (1.0f / row.temperature);
  /* `STRICT_FINITE` adds a whole-row finiteness precondition to a row that
   * would otherwise take the permissive stochastic branch. It is an *addition*,
   * not a replacement: the permissive pass still runs, so the stochastic
   * scratch outputs are computed exactly as they are without the flag. That
   * matters for the shared reduction below, which selects
   * `greedy ? argmax_allowed : allowed_count` — a flag-driven second writer of
   * `argmax_allowed` on a stochastic row would reduce a zero count. */
  const bool strict = (row.flags & DS41RT_V41_SAMPLER_FLAG_STRICT_FINITE) != 0u;

  /* Mask width is validated once, before any logit is read, mirroring
   * `target_sampling.rs:222-230` and the FFI validator. */
  uint32_t mask_status = DS41RT_V41_SAMPLER_STATUS_OK;
  if (mask_words == nullptr) {
    if (mask_words_per_row != 0u) {
      mask_status = DS41RT_V41_SAMPLER_STATUS_MASK_WIDTH;
    }
  } else if (mask_words_per_row != ((vocab + 31u) / 32u)) {
    mask_status = DS41RT_V41_SAMPLER_STATUS_MASK_WIDTH;
  }
  if (mask_status != DS41RT_V41_SAMPLER_STATUS_OK) {
    if (tid == 0) {
      ds41rt_v41_sampler_scratch_t value = {};
      value.status = mask_status;
      value.status_detail = static_cast<uint32_t>(mask_words_per_row);
      value.nonfinite_token = DS41RT_V41_SAMPLER_NO_DETAIL;
      scratch[block_row] = value;
      out_status[output_row] = mask_status;
      out_status_detail[output_row] = static_cast<uint32_t>(mask_words_per_row);
    }
    return;
  }

  float best_score = -CUDART_INF_F;
  uint32_t best_id = 0u;
  float max_scaled = -CUDART_INF_F;
  uint32_t allowed_count = 0u;   /* stochastic only; stays 0 on a greedy row */
  /* Greedy rows do not publish `allowed_count` (the design's scratch field is a
   * stochastic quantity), but `scores.rs::argmax` reports an empty allowed set
   * as EMPTY_CANDIDATES, so the argmax branch still counts the allowed tokens
   * separately. */
  uint32_t argmax_allowed = 0u;
  uint32_t nonfinite = DS41RT_V41_SAMPLER_NO_DETAIL;

  if (greedy) {
    /* Strict whole-row scan: finiteness first, then the mask test. A greedy or
     * constrained row must reject a non-finite logit even when masked out,
     * mirroring `scores.rs::argmax`. */
    for (size_t token = static_cast<size_t>(tid); token < vocab;
         token += static_cast<size_t>(blockDim.x)) {
      const float logit = row_logits[token];
      if (!isfinite(logit)) {
        const uint32_t candidate = static_cast<uint32_t>(token);
        if (candidate < nonfinite) {
          nonfinite = candidate;
        }
      }
      if (row_allowed(row, mask_words, mask_words_per_row, rows, token)) {
        ++argmax_allowed;
        const uint32_t token_id = static_cast<uint32_t>(token);
        if (logit > best_score || (logit == best_score && token_id < best_id)) {
          best_score = logit;
          best_id = token_id;
        }
      }
    }
    /* `allowed_count` stays zero on this branch: the greedy argmax does not need
     * it and the host does not read it for greedy rows. */
  } else {
    if (strict) {
      /* Whole-row finiteness precondition for a stochastic row that opted in.
       * It does not touch `argmax_allowed` or `allowed_count`; the permissive
       * pass below still computes every stochastic output. */
      for (size_t token = static_cast<size_t>(tid); token < vocab;
           token += static_cast<size_t>(blockDim.x)) {
        if (!isfinite(row_logits[token])) {
          const uint32_t candidate = static_cast<uint32_t>(token);
          if (candidate < nonfinite) {
            nonfinite = candidate;
          }
        }
      }
    }
    for (size_t token = static_cast<size_t>(tid); token < vocab;
         token += static_cast<size_t>(blockDim.x)) {
      if (!row_allowed(row, mask_words, mask_words_per_row, rows, token)) {
        continue;
      }
      const float logit = row_logits[token];
      if (!isfinite(logit)) {
        const uint32_t candidate = static_cast<uint32_t>(token);
        if (candidate < nonfinite) {
          nonfinite = candidate;
        }
        continue;
      }
      ++allowed_count;
      max_scaled = fmaxf(max_scaled, logit * inv_temperature);
    }
  }

  shared.score[tid] = best_score;
  shared.id[tid] = best_id;
  shared.scaled[tid] = max_scaled;
  shared.inv_temperature[tid] = inv_temperature;
  shared.allowed[tid] = greedy ? argmax_allowed : allowed_count;
  shared.survivors[tid] = 0u;
  shared.nonfinite[tid] = nonfinite;
  __syncthreads();

  tree_argmax(shared.score, shared.id, tid);
  const float row_max_scaled = tree_max(shared.scaled, tid);
  /* Exact allowed-token count for both branches: the stochastic count feeds
   * EMPTY_CANDIDATES, and the greedy count mirrors `scores.rs::argmax`. */
  const uint32_t row_allowed_count = tree_add_u32(shared.allowed, tid);
  const uint32_t lowest_nonfinite = tree_min_u32(shared.nonfinite, tid);

  uint32_t survivor_count = 0u;
  if (!greedy) {
    const float min_scaled = (row.min_p > 0.0f)
        ? row_max_scaled + row.ln_min_p
        : -CUDART_INF_F;
    for (size_t token = static_cast<size_t>(tid); token < vocab;
         token += static_cast<size_t>(blockDim.x)) {
      if (!row_allowed(row, mask_words, mask_words_per_row, rows, token)) {
        continue;
      }
      if (row_logits[token] * shared.inv_temperature[0] >= min_scaled) {
        ++survivor_count;
      }
    }
    shared.survivors[tid] = survivor_count;
    __syncthreads();
    survivor_count = tree_add_u32(shared.survivors, tid);
  }

  if (tid == 0) {
    ds41rt_v41_sampler_scratch_t value = {};
    value.max_scaled = row_max_scaled;
    value.inv_temperature = shared.inv_temperature[0];
    value.allowed_count = greedy ? 0u : row_allowed_count;
    value.survivor_count = survivor_count;
    value.nonfinite_token = lowest_nonfinite;

    uint32_t status = DS41RT_V41_SAMPLER_STATUS_OK;
    uint32_t detail = DS41RT_V41_SAMPLER_NO_DETAIL;
    if (lowest_nonfinite != DS41RT_V41_SAMPLER_NO_DETAIL) {
      status = DS41RT_V41_SAMPLER_STATUS_NONFINITE_LOGIT;
      detail = lowest_nonfinite;
    } else if (row_allowed_count == 0u) {
      /* A grammar that allows no token is reported as such, not as a
       * temperature error from the -inf maximum (`target_sampling.rs:440-444`). */
      status = DS41RT_V41_SAMPLER_STATUS_EMPTY_CANDIDATES;
    } else if (!greedy && !isfinite(row_max_scaled)) {
      status = DS41RT_V41_SAMPLER_STATUS_INVALID_TEMPERATURE;
    }
    value.status = status;
    value.status_detail = detail;
    scratch[block_row] = value;
    out_status[output_row] = status;
    out_status_detail[output_row] = detail;
    if (status == DS41RT_V41_SAMPLER_STATUS_OK) {
      if (greedy) {
        out_indices[output_row] = shared.id[0];
        /* The raw maximum logit, so the retained-frontier cross-check in
         * `scores.rs::from_greedy` still sees a finite score. */
        out_scores[output_row] = shared.score[0];
      }
      if ((row.flags & DS41RT_V41_SAMPLER_FLAG_DIAGNOSE) != 0u) {
        if (out_total != nullptr) {
          out_total[output_row] = row_max_scaled;
        }
        if (out_nucleus_count != nullptr) {
          out_nucleus_count[output_row] = survivor_count;
        }
      }
    }
  }
}

ds41rt_status_t validate_args(const float* logits, size_t rows, size_t vocab,
                              size_t logits_stride,
                              const ds41rt_v41_sampler_row_t* params,
                              const uint32_t* mask_words, size_t mask_words_per_row,
                              uint32_t* out_indices, uint32_t* out_status,
                              uint32_t* out_status_detail, float* out_scores,
                              ds41rt_v41_sampler_scratch_t* scratch) {
  if (logits == nullptr || params == nullptr || out_indices == nullptr ||
      out_status == nullptr || out_status_detail == nullptr || out_scores == nullptr ||
      scratch == nullptr) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  if (rows == 0 || vocab == 0 ||
      rows > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      vocab > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  if (logits_stride < vocab) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  if (mask_words != nullptr && mask_words_per_row == 0) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  return DS41RT_STATUS_OK;
}

}  // namespace

extern "C" ds41rt_status_t ds41rt_cuda_v41_target_sample_async(
    const float* logits, size_t rows, size_t vocab, size_t logits_stride,
    const ds41rt_v41_sampler_row_t* params, const uint32_t* mask_words,
    size_t mask_words_per_row, uint32_t* out_indices, uint32_t* out_status,
    uint32_t* out_status_detail, float* out_scores, float* out_total,
    uint32_t* out_nucleus_count, ds41rt_v41_sampler_scratch_t* scratch,
    void* cuda_stream) {
  const ds41rt_status_t valid = validate_args(
      logits, rows, vocab, logits_stride, params, mask_words, mask_words_per_row,
      out_indices, out_status, out_status_detail, out_scores, scratch);
  if (valid != DS41RT_STATUS_OK) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  v41_sample_prepare_kernel<<<static_cast<unsigned int>(rows), kBlock, 0, stream>>>(
      logits, rows, vocab, logits_stride, params, mask_words, mask_words_per_row,
      out_indices, out_status, out_status_detail, out_scores, out_total,
      out_nucleus_count, scratch);
  return status_from_cuda(cudaGetLastError());
}

extern "C" ds41rt_status_t ds41rt_cuda_v41_target_sample(
    const float* logits, size_t rows, size_t vocab, size_t logits_stride,
    const ds41rt_v41_sampler_row_t* params, const uint32_t* mask_words,
    size_t mask_words_per_row, uint32_t* out_indices, uint32_t* out_status,
    uint32_t* out_status_detail, float* out_scores, float* out_total,
    uint32_t* out_nucleus_count, ds41rt_v41_sampler_scratch_t* scratch) {
  const ds41rt_status_t status = ds41rt_cuda_v41_target_sample_async(
      logits, rows, vocab, logits_stride, params, mask_words, mask_words_per_row,
      out_indices, out_status, out_status_detail, out_scores, out_total,
      out_nucleus_count, scratch, nullptr);
  if (status != DS41RT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}
