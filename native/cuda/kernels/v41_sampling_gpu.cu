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

/* Chunk 2 selects the K2 cross-segment combine. The shipped default is the
 * fixed-tree segmented scan of design §4.2/§4.8; defining
 * `DS41RT_V41_K2_SEQUENTIAL_COMBINE=1` compiles/launches the strictly sequential
 * token-order combine of §6.5.3 instead. Both kernels are always compiled; the
 * macro only chooses which one the entry point launches, so the chunk-2
 * measurement can build the same source both ways. This is a build-time knob,
 * never an ABI field: the parameter block stays exactly 64 B and the flag bits
 * are unchanged. */
#ifndef DS41RT_V41_K2_SEQUENTIAL_COMBINE
#define DS41RT_V41_K2_SEQUENTIAL_COMBINE 0
#endif

/* Chunk 2's expf experiment also measures a double-precision `exp`-then-round
 * weight. It is a *measurement* variant only: the shipped default is CUDA
 * `expf`. `DS41RT_V41_K2_WEIGHT_DOUBLE=1` selects the double path so its token
 * and latency can be compared against the host's glibc `expf` (design §6.3b). */
#ifndef DS41RT_V41_K2_WEIGHT_DOUBLE
#define DS41RT_V41_K2_WEIGHT_DOUBLE 0
#endif

/* Measurement-only: `DS41RT_V41_K2_NO_WALK_TOTAL=1` reinstates the pre-fix K2
 * `total` (the tree scan's inclusive prefix) so the chunk-2 latency harness can
 * isolate the cost of the owner walk that derives the walk-consistent total.
 * It reintroduces the spurious-fallback bug and MUST NOT be shipped; the shipped
 * default is 0. */
#ifndef DS41RT_V41_K2_NO_WALK_TOTAL
#define DS41RT_V41_K2_NO_WALK_TOTAL 0
#endif

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

__device__ __forceinline__ uint32_t tree_max_u32(uint32_t* values, int tid) {
  for (int stride = kBlock / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      const uint32_t other = values[tid + stride];
      if (other > values[tid]) {
        values[tid] = other;
      }
    }
    __syncthreads();
  }
  return values[0];
}

/* Fixed-shape Hillis-Steele inclusive scan over `kBlock` f32 values, in place.
 * The combine tree is the same for every row, so a row's result cannot depend on
 * the wave, the lane or a peer (design §4.0/§4.8). `values` must be initialised
 * and every thread must have reached the call before any read; callers insert a
 * `__syncthreads()` after the initial write. */
__device__ __forceinline__ float tree_inclusive_scan_f32(float* values, int tid) {
  for (int offset = 1; offset < kBlock; offset <<= 1) {
    float addend = 0.0f;
    if (tid >= offset) {
      addend = values[tid - offset];
    }
    __syncthreads();
    if (tid >= offset) {
      values[tid] += addend;
    }
    __syncthreads();
  }
  return values[kBlock - 1];
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

/* ====================================================================== */
/* K2: fast-path categorical draw (design §4.2, §4.6, §6.1, §6.5.3)       */
/* ====================================================================== */

/* K2 handles exactly the CPU's `sample_categorical` branch: `top_k` disabled
 * (`Option::None`, encoded as 0) and `top_p >= 1.0`, on a non-greedy row whose
 * K1 status is OK. A `top_k >= survivor_count` row whose top-p search still runs
 * on the CPU (`target_sampling.rs:499-522`) is deliberately NOT a fast-path row,
 * because with `top_p = 1.0` the CPU nucleus can still be a strict f32 prefix. */
__device__ __forceinline__ bool k2_applicable(const ds41rt_v41_sampler_row_t& row,
                                              const ds41rt_v41_sampler_scratch_t& state) {
  if (state.status != DS41RT_V41_SAMPLER_STATUS_OK || state.survivor_count == 0u) {
    return false;
  }
  const bool greedy = (row.temperature < 1e-5f) || (row.top_k == 1u) ||
      ((row.flags & DS41RT_V41_SAMPLER_FLAG_GREEDY) != 0u);
  return !greedy && row.top_k == 0u && row.top_p >= 1.0f;
}

/* The CPU's survivor predicate (`target_sampling.rs:456-458`):
 * `allowed(t) && logits[t] * inv >= min_scaled`. `logits[t] * inv` is a single
 * f32 multiply, so it is bit-identical to the CPU and to K1's own pass. */
__device__ __forceinline__ bool k2_survivor(const ds41rt_v41_sampler_row_t& row,
                                            const uint32_t* mask_words,
                                            size_t mask_words_per_row, size_t mask_rows,
                                            const float* row_logits, size_t token,
                                            float inv_temperature, float min_scaled) {
  return row_allowed(row, mask_words, mask_words_per_row, mask_rows, token) &&
      row_logits[token] * inv_temperature >= min_scaled;
}

/* One CPU weight: `(logit * inv - max_scaled).exp()` (`target_sampling.rs:604`).
 *
 * `__fmul_rn`/`__fsub_rn` are required, not stylistic: nvcc's default
 * `-fmad=true` may contract `logit * inv - max_scaled` into a single fma, while
 * Rust/LLVM does not contract by default, so a contracted device expression
 * would differ from the CPU for reasons unrelated to `expf`. The explicit
 * intrinsics make the device perform the same two roundings as the host. */
__device__ __forceinline__ float k2_weight(const float* row_logits, size_t token,
                                           float inv_temperature, float max_scaled) {
  const float delta = __fsub_rn(__fmul_rn(row_logits[token], inv_temperature), max_scaled);
#if DS41RT_V41_K2_WEIGHT_DOUBLE
  return static_cast<float>(exp(static_cast<double>(delta)));
#else
  return expf(delta);
#endif
}

/* K2, tree combine (shipped default): `w_t = expf(scaled_t - max_scaled)` over
 * survivors, a fixed-shape segmented sum, the inclusive ascending-token-order
 * crossing scan, and the CPU's no-crossing last-survivor fallback.
 *
 * The row is cut into `kBlock` contiguous segments. Each thread sums its segment
 * sequentially in ascending token order (so the *within-segment* order is the
 * CPU's), then the segment totals are combined by a fixed Hillis-Steele tree.
 * This is the accumulation-order residual of design §6.3c: a cross-segment tree
 * association differs from the CPU's strict left-to-right sum, so identical
 * uniforms can select different tokens. The sequential variant below removes
 * that source at a measured cost.
 *
 * The crossing scan is the design's "minimum reporting token": every segment
 * whose final cumulative reaches `target` reports its first crossing token, and
 * the minimum over reports is exactly the first crossing token *because* the
 * earliest reporting segment is the one that contains the crossing (weights are
 * non-negative, so earlier segments have final cumulative < target). The
 * comparison is inclusive (`target <= cumulative`). */
[[maybe_unused]] __global__ void v41_sample_categorical_kernel(
    const float* logits, size_t rows, size_t vocab, size_t logits_stride,
    const ds41rt_v41_sampler_row_t* params, const uint32_t* mask_words,
    size_t mask_words_per_row, uint32_t* out_indices, float* out_total,
    ds41rt_v41_sampler_scratch_t* scratch) {
  __shared__ float inclusive[kBlock];
  __shared__ uint32_t hits[kBlock];
  __shared__ uint32_t lasts[kBlock];
  __shared__ float walk_total;

  const size_t block_row = blockIdx.x;
  const int tid = threadIdx.x;
  if (block_row >= rows) {
    return;
  }
  const ds41rt_v41_sampler_row_t row = params[block_row];
  const ds41rt_v41_sampler_scratch_t state = scratch[block_row];
  if (!k2_applicable(row, state)) {
    return;
  }
  const size_t output_row = static_cast<size_t>(row.output_row);
  const float* row_logits = logits + block_row * logits_stride;
  const float inv_temperature = state.inv_temperature;
  const float max_scaled = state.max_scaled;
  /* The device never calls `logf`; `ln_min_p` is host-precomputed (design
   * §6.3a), so this is exactly K1's `min_scaled` and exactly the CPU's. */
  const float min_scaled = (row.min_p > 0.0f)
      ? max_scaled + row.ln_min_p
      : -CUDART_INF_F;

  /* The draw: the same `(seed, position)`-keyed SplitMix64 uniform as the CPU,
   * then the same `[0, MAX_UNIFORM]` clamp (design §6.1/§4.8). */
  const float uniform = ds41rt_v41_target_clamp_uniform(
      ds41rt_v41_target_uniform(row.seed, row.position));

  /* Contiguous `[begin, end)` segment for this thread, in the fixed row shape. */
  const size_t per_thread = (vocab + kBlock - 1) / kBlock;
  const size_t begin = static_cast<size_t>(tid) * per_thread;
  size_t end = begin + per_thread;
  if (end > vocab) {
    end = vocab;
  }

  /* Pass 1: sequential in-segment weight sum, and the highest survivor in this
   * segment.
   *
   * The CPU's `sample_categorical` accumulates `total` and the crossing
   * `cumulative` with the *same* left-to-right f32 walk
   * (`target_sampling.rs:600-615`), so its `total` is exactly the cumulative at
   * the last survivor and its `last` fallback is unreachable:
   * `uniform <= MAX_UNIFORM = 0.99999994 < 1` gives
   * `target <= total == cumulative_at_last_survivor`, and the inclusive
   * `target <= cumulative` fires first. A tree sum is a *different*
   * accumulation, so using the scan's inclusive prefix as `total` (as chunk 2
   * originally did) could make `target` exceed the walk's own final cumulative
   * and fire the defensive fallback spuriously. `total` below is therefore
   * derived from the walk itself, exactly as the CPU derives it. */
  float local = 0.0f;
  /* 0 (not `NO_DETAIL`) marks a segment with no survivor: `NO_DETAIL` is the
   * maximum u32 and would win the `tree_max_u32` reduction below whenever any
   * trailing segment is empty, which is almost every vocabulary. Every real
   * token id is >= 0 and `k2_applicable` guarantees at least one survivor, so
   * the maximum over `lasts` is the true last survivor. */
  uint32_t segment_last_survivor = 0u;
  for (size_t token = begin; token < end; ++token) {
    if (k2_survivor(row, mask_words, mask_words_per_row, rows, row_logits, token,
                    inv_temperature, min_scaled)) {
      segment_last_survivor = static_cast<uint32_t>(token);
      local += k2_weight(row_logits, token, inv_temperature, max_scaled);
    }
  }
  inclusive[tid] = local;
  lasts[tid] = segment_last_survivor;
  __syncthreads();
  tree_inclusive_scan_f32(inclusive, tid);
  /* Highest survivor over the whole row in ascending token order;
   * `k2_applicable` guarantees at least one survivor. */
  const uint32_t global_last_survivor = tree_max_u32(lasts, tid);

  /* `total` = the segmented walk's cumulative at `global_last_survivor`, i.e.
   * the value the crossing pass below actually reaches there. Its owner segment
   * walks up to that token; the other threads idle at the barrier. The added
   * cost is bounded by one segment (<= ceil(vocab / kBlock) tokens). */
#if DS41RT_V41_K2_NO_WALK_TOTAL
  /* Measurement-only (see the macro at the top of this file): the pre-fix total
   * from the tree scan, so the latency harness can isolate the owner walk. This
   * path is never shipped. */
  const float total = fmaxf(inclusive[kBlock - 1], 1.0e-20f);
#else
  if (tid == 0) {
    walk_total = 0.0f;
  }
  __syncthreads();
  {
    const size_t owner = static_cast<size_t>(global_last_survivor) / per_thread;
    if (owner == static_cast<size_t>(tid)) {
      const size_t owner_begin = owner * per_thread;
      float running = (owner == 0u) ? 0.0f : inclusive[owner - 1];
      for (size_t token = owner_begin; token <= global_last_survivor; ++token) {
        if (k2_survivor(row, mask_words, mask_words_per_row, rows, row_logits, token,
                        inv_temperature, min_scaled)) {
          running += k2_weight(row_logits, token, inv_temperature, max_scaled);
        }
      }
      walk_total = running;
    }
  }
  __syncthreads();
  const float total = fmaxf(walk_total, 1.0e-20f);
#endif
  const float target = uniform * total;

  /* Pass 2: walk the segment from its exclusive prefix and report the first
   * token whose inclusive cumulative reaches `target`. Because `total` is the
   * walk's own value at `global_last_survivor`, its owner thread reaches
   * `total >= target` there, so some thread always reports: the CPU's `last`
   * fallback is unreachable on device too, exactly as on the CPU. The sentinel
   * branch is kept for a malformed raw-C call only. */
  float cumulative = (tid == 0) ? 0.0f : inclusive[tid - 1];
  uint32_t first_hit = DS41RT_V41_SAMPLER_NO_DETAIL;
  for (size_t token = begin; token < end; ++token) {
    if (!k2_survivor(row, mask_words, mask_words_per_row, rows, row_logits, token,
                     inv_temperature, min_scaled)) {
      continue;
    }
    cumulative += k2_weight(row_logits, token, inv_temperature, max_scaled);
    if (first_hit == DS41RT_V41_SAMPLER_NO_DETAIL && target <= cumulative) {
      first_hit = static_cast<uint32_t>(token);
    }
  }
  hits[tid] = first_hit;
  __syncthreads();
  const uint32_t crossing = tree_min_u32(hits, tid);
  const uint32_t selected = (crossing != DS41RT_V41_SAMPLER_NO_DETAIL)
      ? crossing
      : global_last_survivor;

  if (tid == 0 && selected != DS41RT_V41_SAMPLER_NO_DETAIL) {
    out_indices[output_row] = selected;
    if ((row.flags & DS41RT_V41_SAMPLER_FLAG_DIAGNOSE) != 0u && out_total != nullptr) {
      out_total[output_row] = total;
    }
  }
}

/* K2, strictly sequential token-order combine (design §6.5.3 option, selected
 * only by `DS41RT_V41_K2_SEQUENTIAL_COMBINE=1`). One thread walks the whole row
 * twice in ascending token order with the exact CPU expression, so with
 * bit-identical weights it reproduces the CPU's cumulative comparison bit for
 * bit and `expf` (§6.3b) is the only remaining residual. This exists to be
 * measured against the tree combine; it is deliberately serial. */
[[maybe_unused]] __global__ void v41_sample_categorical_sequential_kernel(
    const float* logits, size_t rows, size_t vocab, size_t logits_stride,
    const ds41rt_v41_sampler_row_t* params, const uint32_t* mask_words,
    size_t mask_words_per_row, uint32_t* out_indices, float* out_total,
    ds41rt_v41_sampler_scratch_t* scratch) {
  const size_t block_row = blockIdx.x;
  if (block_row >= rows || threadIdx.x != 0) {
    return;
  }
  const ds41rt_v41_sampler_row_t row = params[block_row];
  const ds41rt_v41_sampler_scratch_t state = scratch[block_row];
  if (!k2_applicable(row, state)) {
    return;
  }
  const size_t output_row = static_cast<size_t>(row.output_row);
  const float* row_logits = logits + block_row * logits_stride;
  const float inv_temperature = state.inv_temperature;
  const float max_scaled = state.max_scaled;
  const float min_scaled = (row.min_p > 0.0f)
      ? max_scaled + row.ln_min_p
      : -CUDART_INF_F;
  const float uniform = ds41rt_v41_target_clamp_uniform(
      ds41rt_v41_target_uniform(row.seed, row.position));

  float total = 0.0f;
  for (size_t token = 0; token < vocab; ++token) {
    if (k2_survivor(row, mask_words, mask_words_per_row, rows, row_logits, token,
                    inv_temperature, min_scaled)) {
      total += k2_weight(row_logits, token, inv_temperature, max_scaled);
    }
  }
  total = fmaxf(total, 1.0e-20f);
  const float target = uniform * total;
  float cumulative = 0.0f;
  uint32_t selected = DS41RT_V41_SAMPLER_NO_DETAIL;
  uint32_t last_survivor = DS41RT_V41_SAMPLER_NO_DETAIL;
  for (size_t token = 0; token < vocab; ++token) {
    if (!k2_survivor(row, mask_words, mask_words_per_row, rows, row_logits, token,
                     inv_temperature, min_scaled)) {
      continue;
    }
    last_survivor = static_cast<uint32_t>(token);
    cumulative += k2_weight(row_logits, token, inv_temperature, max_scaled);
    if (target <= cumulative) {
      selected = static_cast<uint32_t>(token);
      break;
    }
  }
  if (selected == DS41RT_V41_SAMPLER_NO_DETAIL) {
    selected = last_survivor;
  }
  if (selected != DS41RT_V41_SAMPLER_NO_DETAIL) {
    out_indices[output_row] = selected;
    if ((row.flags & DS41RT_V41_SAMPLER_FLAG_DIAGNOSE) != 0u && out_total != nullptr) {
      out_total[output_row] = total;
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
  const ds41rt_status_t prepare = status_from_cuda(cudaGetLastError());
  if (prepare != DS41RT_STATUS_OK) {
    return prepare;
  }
  /* K2 reads K1's `scratch` and, for fast-path rows only, publishes
   * `out_indices` (and the diagnostic `out_total`). Same stream, so K1's
   * scratch write is visible; no host synchronization is added. */
#if DS41RT_V41_K2_SEQUENTIAL_COMBINE
  v41_sample_categorical_sequential_kernel<<<static_cast<unsigned int>(rows), kBlock, 0,
                                             stream>>>(
      logits, rows, vocab, logits_stride, params, mask_words, mask_words_per_row,
      out_indices, out_total, scratch);
#else
  v41_sample_categorical_kernel<<<static_cast<unsigned int>(rows), kBlock, 0, stream>>>(
      logits, rows, vocab, logits_stride, params, mask_words, mask_words_per_row,
      out_indices, out_total, scratch);
#endif
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
