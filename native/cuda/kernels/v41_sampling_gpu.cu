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

/* Measurement-only, now VESTIGIAL: `DS41RT_V41_K2_NO_WALK_TOTAL=1` used to
 * reinstate the pre-fix K2 `total` (the tree scan's inclusive prefix) so the
 * chunk-2 latency harness could isolate the owner walk. Chunk 3b's fix removes
 * the owner walk and the tree scan together and derives `total` from the
 * sequential segment prefix `C(kBlock)`, so the macro no longer selects any
 * code path. It is kept defined only so existing measurement build scripts that
 * pass `-DDS41RT_V41_K2_NO_WALK_TOTAL=1` still compile; the shipped default is
 * and always was 0. */
#ifndef DS41RT_V41_K2_NO_WALK_TOTAL
#define DS41RT_V41_K2_NO_WALK_TOTAL 0
#endif

/* Chunk-3b mass arithmetic selector. The **shipped default is 1**: K5 performs
 * the CPU's literal `sample_from_ranked` arithmetic, one f32 division `w / total`
 * per retained/survivor token accumulated in the kernel's own order, with the
 * un-scaled thresholds (`top_p`, and `uniform * nucleus_mass`). The chunk-3b
 * measurement over the 344,064-draw harness is decisive: the retained domain's
 * token mismatch rate against the production sampler falls from 11.42 % to
 * **0.00 %** (16,834 -> 0 of 147,456) and the survivor domain from 25.948 % to
 * 25.903 %, because the retained prefix becomes bit-identical to the CPU's
 * `nucleus_mass` accumulation.
 *
 * Defining `DS41RT_V41_K5_NORMALIZED_MASS=0` selects the algebraic rewrite
 * instead: the search predicates run on the **un-normalized** weight sums and
 * the threshold is `fl(top_p * total)`, so no probe performs a per-token
 * division. The two forms are equal in real arithmetic but not in f32; the 0
 * variant is kept for the report's A/B measurement and for a caller that wants
 * the division-free form. It is a build-time knob, never an ABI field. */
#ifndef DS41RT_V41_K5_NORMALIZED_MASS
#define DS41RT_V41_K5_NORMALIZED_MASS 1
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
 * `__syncthreads()` after the initial write.
 *
 * K2 no longer uses this (its segment prefix is now a single sequential fold so
 * the exclusive prefix is the crossing walk's own association); it is kept as a
 * fixed-tree reference and for the M6 pre-fix mutant. */
[[maybe_unused]] __device__ __forceinline__ float tree_inclusive_scan_f32(float* values, int tid) {
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
    } else if (!greedy && !isfinite(row.top_p)) {
      /* `top_p = NaN` (or any non-finite value) makes K2's `top_p >= 1.0` and
       * K5's `top_p < 1.0` both false, so without this the row would keep the
       * caller's sentinel while the entry point returned OK -- the K1+K2-only
       * `ds41rt_cuda_v41_target_sample` cannot otherwise tell "sampled" from
       * "never ran". The FFI validator rejects a non-finite `top_p` on the host;
       * this closes the raw-C gap for every entry point that launches K1. */
      status = DS41RT_V41_SAMPLER_STATUS_INTERNAL;
    } else if (!greedy && survivor_count == 0u) {
      /* A non-greedy row with no surviving token cannot be sampled by any stage
       * (K2 and K5 both require a survivor). On a valid row the best allowed
       * token always survives (`min_p <= 1`), so this is the raw-C shape; make
       * it loud rather than leaving `out_indices` at the caller's sentinel with
       * an OK status. */
      status = DS41RT_V41_SAMPLER_STATUS_INTERNAL;
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
 * CPU's); the segment totals are then folded into ONE non-decreasing segment
 * prefix `C(i+1) = fl(C(i) + local(i))` by a single sequential scan. This is the
 * accumulation-order residual of design §6.3c: a cross-segment association
 * differs from the CPU's strict left-to-right sum, so identical uniforms can
 * select different tokens. The sequential variant below removes that source at a
 * measured cost.
 *
 * The crossing scan is the design's "minimum reporting token": every segment
 * whose start prefix is strictly below `target` reports its first crossing
 * token, the minimum over reports is the first crossing token, and a token is
 * only reported when the cumulative *before* it was below `target` -- the
 * minimality rule that structurally forbids a zero-weight selection. The
 * comparison is inclusive (`target <= cumulative`), matching the CPU. */
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

  /* Consistent sequential prefix over the segment masses. `inclusive[i]` becomes
   * the EXCLUSIVE prefix `C(i) = fl(C(i-1) + local(i-1))` with `C(0) = 0`, and
   * `walk_total` becomes `C(kBlock)`, the single non-decreasing segment-level
   * cumulative both the crossing and the total are read from. This replaces the
   * Hillis-Steele tree scan. The tree's association made the prefix handed to
   * segment `i` differ from the value the crossing walk itself reaches at the
   * end of segment `i-1`: with a tail weight below half an ulp of 1.0 the
   * segment-local sum keeps the tails while a running fold does not, so the next
   * segment could start at a prefix already past `target` and "cross" at its
   * first survivor -- a token whose weight is exactly zero. The scan is at most
   * `kBlock` sequential shared-memory adds by one thread (~256 steps); pass 1's
   * per-segment sums and every min/max reduction below stay parallel.
   *
   * The old owner-segment re-walk that derived `total` from the tree prefix is
   * gone: `C(kBlock)` is the walk's own segment-level total, `target <= total`
   * for the clamped uniform, and no second accumulation is needed. */
  if (tid == 0) {
    float running = 0.0f;
    for (int segment = 0; segment < kBlock; ++segment) {
      const float segment_mass = inclusive[segment];
      inclusive[segment] = running;
      running += segment_mass;
    }
    walk_total = running;
  }
  __syncthreads();
  /* Highest survivor over the whole row in ascending token order;
   * `k2_applicable` guarantees at least one survivor. */
  const uint32_t global_last_survivor = tree_max_u32(lasts, tid);
  const float total = fmaxf(walk_total, 1.0e-20f);
  const float target = uniform * total;

  /* Pass 2: walk the segment from its exclusive prefix `C(tid)` and report the
   * first survivor whose cumulative reaches `target`. Three rules make the
   * reported token the minimum-index survivor `t` with `W(t) >= target`:
   *
   *   - a segment may report only if its start prefix is strictly below
   *     `target` (`may_report`), so a segment whose boundary already jumped past
   *     `target` cannot "cross" at a zero-weight first survivor;
   *   - a token is reported only if the cumulative BEFORE it was strictly below
   *     `target`, the minimality rule: if `W(t) == W(t-1)` then `t-1` is also a
   *     survivor index that satisfies. With `fl(before + w) >= target > before`
   *     this forces `w > 0` in f32, so a zero-weight token is structurally
   *     unreachable;
   *   - `target == 0` (uniform 0) still selects the first survivor, exactly as
   *     the CPU's loop does by adding a weight before testing `target <=
   *     cumulative`.
   *
   * If the segment whose mass brackets `target` (`C(tid) < target <= C(tid+1)`)
   * cannot reach `target` because its own walk association rounds the in-segment
   * tail away, it reports its FIRST positive-weight survivor -- the minimal-index
   * candidate in the bracketing segment, which never overruns `target` and, like
   * every report, has a strictly positive weight. That segment always contains a
   * positive weight (otherwise its mass could not span `target`), and only the
   * earliest such segment reports, so the minimum over reports is still the first
   * crossing. The sentinel branch stays for a malformed raw-C call. */
  const float start = inclusive[tid];
  const float end_prefix = (tid + 1 < kBlock) ? inclusive[tid + 1] : walk_total;
  const bool zero_target = (target == 0.0f);
  const bool may_report = zero_target || (start < target);
  float cumulative = start;
  uint32_t first_hit = DS41RT_V41_SAMPLER_NO_DETAIL;
  uint32_t first_positive = DS41RT_V41_SAMPLER_NO_DETAIL;
  if (may_report) {
    for (size_t token = begin; token < end; ++token) {
      if (!k2_survivor(row, mask_words, mask_words_per_row, rows, row_logits, token,
                       inv_temperature, min_scaled)) {
        continue;
      }
      const float weight = k2_weight(row_logits, token, inv_temperature, max_scaled);
      const float before = cumulative;
      cumulative += weight;
      if (weight > 0.0f && first_positive == DS41RT_V41_SAMPLER_NO_DETAIL) {
        first_positive = static_cast<uint32_t>(token);
      }
      if (first_hit == DS41RT_V41_SAMPLER_NO_DETAIL && target <= cumulative &&
          (zero_target || before < target)) {
        first_hit = static_cast<uint32_t>(token);
      }
    }
    if (first_hit == DS41RT_V41_SAMPLER_NO_DETAIL && !zero_target &&
        start < target && end_prefix >= target) {
      first_hit = first_positive;
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

/* ====================================================================== */
/* K3/K4: general-k pivot selection + exact-k lowest-id membership        */
/* (design §4.3-§4.4, Appendix A.1; contract §1.4 / risk R2)              */
/* ====================================================================== */

/* Two independent u32 counts at once, in the fixed binary-tree shape of the
 * other reductions. K3 needs both probes' counts per pass; combining them in
 * one stride loop halves the barrier count relative to two `tree_add_u32`
 * calls. Integer addition is exact and associative, so the fixed combine order
 * is only about determinism (§4.0/§4.8), which the fixed shape gives. */
__device__ __forceinline__ void tree_add_u32_pair(uint32_t* first, uint32_t* second,
                                                  int tid) {
  for (int stride = kBlock / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      first[tid] += first[tid + stride];
      second[tid] += second[tid + stride];
    }
    __syncthreads();
  }
}

/* Fixed-shape Hillis-Steele inclusive scan over `kBlock` u32 values, in place.
 * The integer analogue of `tree_inclusive_scan_f32`; used by K4's per-segment
 * exclusive counts (the tie prefix and the compaction offsets). Callers insert a
 * `__syncthreads()` after the initial write. */
__device__ __forceinline__ void tree_inclusive_scan_u32(uint32_t* values, int tid) {
  for (int offset = 1; offset < kBlock; offset <<= 1) {
    uint32_t addend = 0u;
    if (tid >= offset) {
      addend = values[tid - offset];
    }
    __syncthreads();
    if (tid >= offset) {
      values[tid] += addend;
    }
    __syncthreads();
  }
}

/* `-0.0`-canonicalizing scaled value of one token, the single f32 multiply of
 * design §4.0 (`logit * inv_temperature`, never a division). */
__device__ __forceinline__ float k3_scaled(const float* row_logits, size_t token,
                                           float inv_temperature) {
  return __fmul_rn(row_logits[token], inv_temperature);
}

/* The survivor predicate shared by K3 and K4, byte-identical to K1's and to
 * `target_sampling.rs:456-458`: `allowed(t) && scaled(t) >= min_scaled`. A
 * masked-out token is never read, so a masked non-finite logit stays legal on a
 * stochastic row (§4.0/§12.6). */
__device__ __forceinline__ bool k3_survivor(const ds41rt_v41_sampler_row_t& row,
                                            const uint32_t* mask_words,
                                            size_t mask_words_per_row, size_t mask_rows,
                                            const float* row_logits, size_t token,
                                            float inv_temperature, float min_scaled) {
  return row_allowed(row, mask_words, mask_words_per_row, mask_rows, token) &&
      k3_scaled(row_logits, token, inv_temperature) >= min_scaled;
}

/* K3/K4 row eligibility, exactly the CPU's `top_k` branch
 * (`target_sampling.rs:477-501`):
 *   - K1 must have reported `OK` with at least one survivor;
 *   - the row must not be greedy (`temperature < 1e-5 || top_k == 1`);
 *   - `top_k == 0` is `Option::None` (disabled) and never enters K3;
 *   - `top_k >= survivor_count` truncates nothing and is a no-op, so K5 must
 *     treat the retained set as all survivors there.
 * A greedy row never reaches here: the CPU short-circuits it at `:234-236`. */
__device__ __forceinline__ bool k3_eligible(const ds41rt_v41_sampler_row_t& row,
                                            const ds41rt_v41_sampler_scratch_t& state) {
  if (state.status != DS41RT_V41_SAMPLER_STATUS_OK || state.survivor_count == 0u) {
    return false;
  }
  const bool greedy = (row.temperature < 1e-5f) || (row.top_k == 1u) ||
      ((row.flags & DS41RT_V41_SAMPLER_FLAG_GREEDY) != 0u);
  return !greedy && row.top_k != 0u && row.top_k < state.survivor_count;
}

/* K3, `v41_sample_topk_pivot_kernel` (design §4.3, Appendix A.1).
 *
 * Finds the k-th largest order key by bisection on the monotone count
 * `C_gt(v) = #{survivors : order_key(scaled) > v}`. `C_gt` is non-increasing in
 * `v`, so the predicate `C_gt(v) < k` is upward-closed and the k-th value is its
 * **smallest** satisfying key. The invariant is
 *   `C_gt(lo) >= k` (lo rejected), `C_gt(hi) < k` (hi accepted), answer in `(lo, hi]`.
 * It starts at `lo = 0` (below every real key: `C_gt(0) = survivor_count`, and
 * `k < survivor_count` is the eligibility condition, so `lo` is rejected) and
 * `hi = 0xFFFFFFFF` (above every real key, so `C_gt(hi) = 0 < k`). This also
 * covers the all `-inf`-survivor corner: `key(-inf) = 0x007FFFFF > 0`, so the
 * bisection walks down to it instead of needing a special case.
 *
 * Two probes per pass (`m0 < m1`) make it a **ternary** search: a satisfying
 * `m0` keeps `[lo, m0]`, an unsatisfied `m0` with a satisfying `m1` keeps
 * `[m0, m1]`, and neither satisfying keeps `[m1, hi]`. The interval shrinks to
 * at most ~1/3 + O(1) per pass, so `2^32` converges in well under the
 * `DS41RT_V41_TOPK_MAX_PIVOT_STEPS` (32) defensive cap; hitting the cap is
 * reported as `INTERNAL` and materializes nothing.
 *
 * The ternary split needs `hi - lo >= 3`; the search therefore finishes with a
 * **single extra probe at `m0 = lo + 1` when `hi - lo == 2`**. That final probe
 * is load-bearing, not cosmetic: the ternary rule can land on a length-2
 * interval (e.g. from a length-4 interval's middle branch) whose *lower* value is
 * the true k-th key, and stopping at `hi - lo < 3` would then return `hi = kth+1`
 * — a key no survivor carries, which silently empties the tie cut. The shipped
 * loop probes while `hi - lo >= 2` and stops at `hi - lo == 1`, so the returned
 * `hi` is exactly the smallest accepted key. (Adversarial single-key regression:
 * `native/tests/v41_sampling_selftest.cu`, "K3 length-2 interval".)
 *
 * The final `hi` is a real survivor's key (the count can only drop at a present
 * key, and `C_gt(hi-1) >= k`), and `above` tracks `C_gt(hi)` through the passes,
 * so no extra count pass is needed. */
__global__ void v41_sample_topk_pivot_kernel(
    const float* logits, size_t rows, size_t vocab, size_t logits_stride,
    const ds41rt_v41_sampler_row_t* params, const uint32_t* mask_words,
    size_t mask_words_per_row, uint32_t* out_pivot_passes,
    ds41rt_v41_sampler_scratch_t* scratch) {
  __shared__ uint32_t probe0[kBlock];
  __shared__ uint32_t probe1[kBlock];

  const size_t block_row = blockIdx.x;
  const int tid = threadIdx.x;
  if (block_row >= rows) {
    return;
  }
  const ds41rt_v41_sampler_row_t row = params[block_row];
  const ds41rt_v41_sampler_scratch_t state = scratch[block_row];
  if (!k3_eligible(row, state)) {
    if (tid == 0 && out_pivot_passes != nullptr) {
      out_pivot_passes[row.output_row] = 0u;
    }
    return;
  }
  const uint32_t k = row.top_k;
  const float inv_temperature = state.inv_temperature;
  const float min_scaled = (row.min_p > 0.0f)
      ? state.max_scaled + row.ln_min_p
      : -CUDART_INF_F;
  const float* row_logits = logits + block_row * logits_stride;

  uint32_t lo = 0u;          /* rejected: C_gt(lo) >= k */
  uint32_t hi = 0xFFFFFFFFu; /* accepted: C_gt(hi) < k (vacuously, above all) */
  uint32_t above = 0u;       /* C_gt(hi) for the current hi */
  uint32_t passes = 0u;
  for (; passes < DS41RT_V41_TOPK_MAX_PIVOT_STEPS && (hi - lo) >= 2u; ++passes) {
    uint32_t m0 = 0u;
    uint32_t m1 = 0u;
    if (hi - lo == 2u) {
      /* Final single probe: for a length-2 interval the only candidate below
       * `hi` is `lo + 1`, and probing `hi` again is harmless because it is
       * known accepted. */
      m0 = lo + 1u;
      m1 = hi;
    } else {
      const uint32_t third = (hi - lo) / 3u;
      m0 = lo + third;
      m1 = hi - third;
    }
    uint32_t count0 = 0u;
    uint32_t count1 = 0u;
    for (size_t token = static_cast<size_t>(tid); token < vocab;
         token += static_cast<size_t>(blockDim.x)) {
      if (!k3_survivor(row, mask_words, mask_words_per_row, rows, row_logits, token,
                       inv_temperature, min_scaled)) {
        continue;
      }
      const uint32_t key =
          ds41rt_v41_order_key(k3_scaled(row_logits, token, inv_temperature));
      if (key > m0) {
        ++count0;
        if (key > m1) {
          ++count1;
        }
      }
    }
    probe0[tid] = count0;
    probe1[tid] = count1;
    __syncthreads();
    tree_add_u32_pair(probe0, probe1, tid);
    const uint32_t total0 = probe0[0];
    const uint32_t total1 = probe1[0];
    /* Every thread reads the same reduce result before any thread can overwrite
     * the shared arrays on the next pass. */
    __syncthreads();
    if (total0 < k) {
      hi = m0;
      above = total0;
    } else if (total1 < k) {
      lo = m0;
      hi = m1;
      above = total1;
    } else {
      lo = m1;
    }
  }
  const bool converged = (hi - lo) == 1u;
  if (tid == 0) {
    ds41rt_v41_sampler_scratch_t value = state;
    if (!converged) {
      value.status = DS41RT_V41_SAMPLER_STATUS_INTERNAL;
      value.kth_value_bits = 0u;
      value.above_count = 0u;
    } else {
      value.kth_value_bits = __float_as_uint(ds41rt_v41_ordered_value(hi));
      value.above_count = above;
    }
    scratch[block_row] = value;
    if (out_pivot_passes != nullptr) {
      out_pivot_passes[row.output_row] = passes;
    }
  }
}

/* K4, `v41_sample_topk_membership_kernel` (design §4.4, Appendix A.2).
 *
 * Membership is ds41rt's exact-k rule (`target_sampling.rs:30-33`):
 *   `{order_key > kth} ∪ {the lowest-id (k - above_count) survivors with
 *    order_key == kth}`.
 * A single pass over `[0, vocab)` in `kBlock` contiguous segments gives each
 * thread the `above` and `equal` counts of its segment; a fixed-tree exclusive
 * scan of the equal counts yields `eq_before(t)` — the number of equal-key
 * survivors earlier in token order — and each thread admits the equal-key
 * tokens whose running index is `< target_tie = k - above_count`. That is
 * correct for any multiplicity, including thousands of tokens sharing the k-th
 * value and the whole row tied (then `above = 0`, `target_tie = k`, and the
 * first `k` tokens in token order are admitted, i.e. ids `0..k-1`).
 *
 * The retained set is then compacted in token order into `rank_order_scratch`
 * as u64 `(order_key << 32) | id`, stable over token order. A second pass
 * assigns each entry its exact rank by counting the entries that are strictly
 * better under the u64 total order (larger key, then lower id) and writes the
 * id at that rank into `rank_order_ids`. Stability of the compaction plus the
 * unique id make the placement exact: the result is `ranked[..k]` in the CPU's
 * comparison-sort order, which the CPU's stable LSD radix sort also produces
 * (`target_sampling.rs:323-344`). No sentinel carrier is involved (contrast the
 * chunk-2 `tree_max_u32` bug): every decision is an integer count or a scan.
 *
 * `rank_order_capacity == 0`, or a row whose `k` exceeds it, is a
 * selection-only call: the membership state is computed but nothing is written.
 * The FFI wrapper rejects a non-zero capacity below the batch's maximum `k`, so
 * a materializing call is always in bounds; the in-kernel check is the
 * raw-C-caller safety net. */
__global__ void v41_sample_topk_membership_kernel(
    const float* logits, size_t rows, size_t vocab, size_t logits_stride,
    const ds41rt_v41_sampler_row_t* params, const uint32_t* mask_words,
    size_t mask_words_per_row, uint32_t* rank_order_ids,
    uint64_t* rank_order_scratch, size_t rank_order_capacity,
    uint32_t* out_retained_count, ds41rt_v41_sampler_scratch_t* scratch) {
  __shared__ uint32_t equal_counts[kBlock];
  __shared__ uint32_t above_counts[kBlock];
  __shared__ uint32_t retained_counts[kBlock];

  const size_t block_row = blockIdx.x;
  const int tid = threadIdx.x;
  if (block_row >= rows) {
    return;
  }
  const ds41rt_v41_sampler_row_t row = params[block_row];
  const ds41rt_v41_sampler_scratch_t state = scratch[block_row];
  if (!k3_eligible(row, state)) {
    if (tid == 0 && out_retained_count != nullptr) {
      out_retained_count[row.output_row] = 0u;
    }
    return;
  }
  const uint32_t k = row.top_k;
  const float inv_temperature = state.inv_temperature;
  const float min_scaled = (row.min_p > 0.0f)
      ? state.max_scaled + row.ln_min_p
      : -CUDART_INF_F;
  const uint32_t kth_key = ds41rt_v41_order_key(__uint_as_float(state.kth_value_bits));
  const float* row_logits = logits + block_row * logits_stride;

  /* Contiguous `[begin, end)` segment, the shape the tie prefix's token order
   * needs (design A.2). */
  const size_t per_thread = (vocab + kBlock - 1) / kBlock;
  const size_t begin = static_cast<size_t>(tid) * per_thread;
  size_t end = begin + per_thread;
  if (end > vocab) {
    end = vocab;
  }

  uint32_t local_above = 0u;
  uint32_t local_equal = 0u;
  for (size_t token = begin; token < end; ++token) {
    if (!k3_survivor(row, mask_words, mask_words_per_row, rows, row_logits, token,
                     inv_temperature, min_scaled)) {
      continue;
    }
    const uint32_t key =
        ds41rt_v41_order_key(k3_scaled(row_logits, token, inv_temperature));
    if (key > kth_key) {
      ++local_above;
    } else if (key == kth_key) {
      ++local_equal;
    }
  }
  equal_counts[tid] = local_equal;
  above_counts[tid] = local_above;
  __syncthreads();
  tree_inclusive_scan_u32(equal_counts, tid);
  const uint32_t total_above = tree_add_u32(above_counts, tid);
  const uint32_t equal_before = (tid == 0u) ? 0u : equal_counts[tid - 1];
  /* `total_above == C_gt(kth) < k` by construction; the guard is defensive for a
   * malformed K3 result and keeps `target_tie` from wrapping. */
  const uint32_t target_tie = (total_above < k) ? (k - total_above) : 0u;
  const uint32_t admit_room = (equal_before < target_tie) ? (target_tie - equal_before) : 0u;
  const uint32_t admitted = (admit_room < local_equal) ? admit_room : local_equal;
  retained_counts[tid] = local_above + admitted;
  __syncthreads();
  tree_inclusive_scan_u32(retained_counts, tid);
  const uint32_t retained_before = (tid == 0u) ? 0u : retained_counts[tid - 1];
  const uint32_t total_retained = retained_counts[kBlock - 1];

  const bool materialize = rank_order_capacity > 0u &&
      rank_order_capacity >= static_cast<size_t>(total_retained) &&
      rank_order_ids != nullptr && rank_order_scratch != nullptr;
  if (materialize) {
    uint64_t* const row_entries =
        rank_order_scratch + block_row * rank_order_capacity;
    uint32_t* const row_ids = rank_order_ids + block_row * rank_order_capacity;
    /* Compaction, token order: retained tokens land at
     * `retained_before + local_index`, so equal keys keep ascending ids. */
    uint32_t write = retained_before;
    uint32_t equal_running = equal_before;
    for (size_t token = begin; token < end; ++token) {
      if (!k3_survivor(row, mask_words, mask_words_per_row, rows, row_logits, token,
                       inv_temperature, min_scaled)) {
        continue;
      }
      const uint32_t key =
          ds41rt_v41_order_key(k3_scaled(row_logits, token, inv_temperature));
      if (key > kth_key) {
        row_entries[write] = (static_cast<uint64_t>(key) << 32) |
                             static_cast<uint32_t>(token);
        ++write;
      } else if (key == kth_key) {
        if (equal_running < target_tie) {
          row_entries[write] = (static_cast<uint64_t>(key) << 32) |
                               static_cast<uint32_t>(token);
          ++write;
        }
        ++equal_running;
      }
    }
    __syncthreads();
    /* Rank placement: the number of retained entries strictly better under the
     * u64 total order is the entry's rank. */
    for (uint32_t index = static_cast<uint32_t>(tid); index < total_retained;
         index += static_cast<uint32_t>(blockDim.x)) {
      const uint64_t entry = row_entries[index];
      const uint32_t key = static_cast<uint32_t>(entry >> 32);
      const uint32_t id = static_cast<uint32_t>(entry);
      uint32_t rank = 0u;
      for (uint32_t other = 0u; other < total_retained; ++other) {
        const uint64_t candidate = row_entries[other];
        const uint32_t candidate_key = static_cast<uint32_t>(candidate >> 32);
        if (candidate_key > key ||
            (candidate_key == key && static_cast<uint32_t>(candidate) < id)) {
          ++rank;
        }
      }
      row_ids[rank] = id;
    }
  }

  if (tid == 0) {
    ds41rt_v41_sampler_scratch_t value = state;
    /* K4 recomputes `C_gt(kth)` from its own pass, so the published
     * `above_count` is self-consistent with the materialized membership even if
     * K3's probe count and this pass were to disagree. */
    value.above_count = total_above;
    scratch[block_row] = value;
    if (out_retained_count != nullptr) {
      out_retained_count[row.output_row] = total_retained;
    }
  }
}

/* ====================================================================== */
/* K5: inclusive-prefix top-p nucleus + rank-order draw                    */
/* (design §4.5, §4.6, Appendix A.1; contract §1.3)                        */
/* ====================================================================== */

/* K5's weight primitive, byte-identical to K2's `k2_weight`
 * (`target_sampling.rs:536`): `expf(scaled - max_scaled)` with the scaled value
 * formed by the single f32 multiply of §4.0 and the subtraction explicitly
 * rounded so nvcc's `-fmad` cannot contract the expression into an fma the host
 * does not perform. K5 is the first chunk that needs weights on the ordered
 * path, so it reuses the chunk-2 primitive rather than adding a second
 * spelling. */
__device__ __forceinline__ float k5_weight(const float* row_logits, size_t token,
                                           float inv_temperature, float max_scaled) {
  return k2_weight(row_logits, token, inv_temperature, max_scaled);
}

/* K5 row eligibility, the ordered subset of §4.6:
 *   - K1 must have reported `OK` with at least one survivor;
 *   - the row must not be greedy (`temperature < 1e-5 || top_k == 1`); a greedy
 *     row never consumes a draw and K1's argmax already wrote `out_indices`;
 *   - `top_k != 0 || top_p < 1.0`: `top_k != 0` is the ordered top-k path (the
 *     CPU's bounded-heap and full-ordering branches, `:477-522`) and
 *     `top_k == 0 && top_p < 1.0` is the ordered top-p-only branch. The only
 *     row left out is the disjoint K2 fast path (`top_k == 0 && top_p >= 1.0`,
 *     `target_sampling.rs:463-466`), which K2 already sampled. */
__device__ __forceinline__ bool k5_ordered(const ds41rt_v41_sampler_row_t& row,
                                           const ds41rt_v41_sampler_scratch_t& state) {
  if (state.status != DS41RT_V41_SAMPLER_STATUS_OK || state.survivor_count == 0u) {
    return false;
  }
  const bool greedy = (row.temperature < 1.0e-5f) || (row.top_k == 1u) ||
      ((row.flags & DS41RT_V41_SAMPLER_FLAG_GREEDY) != 0u);
  return !greedy && (row.top_k != 0u || row.top_p < 1.0f);
}

/* The retained domain no longer needs a multi-bucket mass probe. Its boundary
 * and draw run on the fixed-association inclusive prefix built once from the
 * shared weight table (see `v41_sample_nucleus_kernel`), which is what makes
 * `W(rank)` a single monotone predicate: the previous `tree_add_f32_4` /
 * `k5_probe_retained` pair split each thread's rank segment at the *probe*
 * bounds, so the same rank had different f32 associations under different
 * search states. */

/* K5's survivor-domain **key-domain** mass probe (design §4.5 "Boundary
 * primitive"). For `top_k == 0` and `top_k >= survivor_count` K3/K4 are no-ops
 * and the CPU's ordered set is the *full* survivor list sorted by its total
 * order
 *
 *   key(t) = (order_key(scaled_t) << 32) | ~id_t      (larger key = better rank,
 *                                                      exact ties to the lower id)
 *
 * so a rank prefix is exactly `{t in S : key(t) >= K}` and its mass is
 * `M(K) = Sum_{key(t) >= K} w_t`. `M` is non-increasing in `K`, so every
 * boundary K5 needs is the *largest* key with `M(K) >= threshold` -- the worst
 * rank still inside the prefix, which is the CPU's crossing rank itself.
 *
 * The probe reports `M((v << 32) | ~j)`: the mass of survivors whose
 * `order_key` is greater than `v`, plus those with `order_key == v` and token id
 * `<= j`. `(v, j) = (0u, ~0u)` is every survivor, `(v, ~0u)` is `M(order_key >=
 * v)`, and `(v, 0u)` is `M` at the boundary of the tie group. Weights are
 * recomputed per pass from the logits -- there is no O(vocab) table on this
 * path -- and every pass uses the same fixed tree reduction, so the association
 * is row-independent and reproducible (design §4.0/§4.8).
 *
 * Ids are 32-bit in the key, but only ids `< vocab` exist, so the tie search
 * only ever probes ids below `vocab`. */
__device__ __forceinline__ float k5_key_mass(
    const ds41rt_v41_sampler_row_t& row, const uint32_t* mask_words,
    size_t mask_words_per_row, size_t mask_rows, const float* row_logits, size_t vocab,
    float inv_temperature, float max_scaled, float min_scaled, uint32_t value,
    uint32_t id_bound, float divisor, float* phase, uint32_t* count_scratch,
    uint32_t* count_out, int tid) {
  float local = 0.0f;
  uint32_t local_count = 0u;
  for (size_t token = static_cast<size_t>(tid); token < vocab;
       token += static_cast<size_t>(blockDim.x)) {
    if (!k3_survivor(row, mask_words, mask_words_per_row, mask_rows, row_logits, token,
                     inv_temperature, min_scaled)) {
      continue;
    }
    const uint32_t key =
        ds41rt_v41_order_key(k3_scaled(row_logits, token, inv_temperature));
    if (key > value || (key == value && static_cast<uint32_t>(token) <= id_bound)) {
      const float weight = k5_weight(row_logits, token, inv_temperature, max_scaled);
#if DS41RT_V41_K5_NORMALIZED_MASS
      /* The CPU's per-token normalization (`target_sampling.rs:540-542`): one f32
       * division per in-set token. The caller passes `divisor == 1.0f` for the
       * pass that *derives* the total and `divisor == total` for every later
       * pass, so a mass is always `Sum (w_t / total)`. */
      local += weight / divisor;
#else
      (void)divisor;
      local += weight;
#endif
      ++local_count;
    }
  }
  phase[tid] = local;
  if (count_scratch != nullptr) {
    count_scratch[tid] = local_count;
  }
  __syncthreads();
  for (int stride = kBlock / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      phase[tid] += phase[tid + stride];
      if (count_scratch != nullptr) {
        count_scratch[tid] += count_scratch[tid + stride];
      }
    }
    __syncthreads();
  }
  /* Snapshot the reduced value into every thread's local, then barrier before
   * any thread can overwrite `phase`/`count_scratch` on the next probe. The
   * reduction's last barrier makes the result visible, but it does NOT stop a
   * fast thread from entering the next probe and writing `phase[tid]` before a
   * slow thread has read `phase[0]`. `k5_key_mass` is called in a search loop
   * with a fresh write at its head, so the read/return below must be followed by
   * a barrier -- exactly the hazard K3 documents and fences in
   * `v41_sample_topk_pivot_kernel` ("Every thread reads the same reduce result
   * before any thread can overwrite the shared arrays on the next pass"). */
  const float reduced = phase[0];
  const uint32_t reduced_count = (count_scratch != nullptr) ? count_scratch[0] : 0u;
  __syncthreads();
  if (count_out != nullptr && tid == 0) {
    *count_out = reduced_count;
  }
  return reduced;
}

/* One pass that reports the survivor key range: `min_key` and `max_key` over the
 * rows' survivors. A malformed row with no survivors leaves both at `value` (the
 * caller never reaches this pass in that case: K1 publishes
 * `survivor_count == 0`). */
__device__ __forceinline__ void k5_key_range(
    const ds41rt_v41_sampler_row_t& row, const uint32_t* mask_words,
    size_t mask_words_per_row, size_t mask_rows, const float* row_logits, size_t vocab,
    float inv_temperature, float min_scaled, uint32_t* min_max, int tid) {
  uint32_t local_min = 0xFFFFFFFFu;
  uint32_t local_max = 0u;
  uint32_t local_worst_id = 0u;
  for (size_t token = static_cast<size_t>(tid); token < vocab;
       token += static_cast<size_t>(blockDim.x)) {
    if (!k3_survivor(row, mask_words, mask_words_per_row, mask_rows, row_logits, token,
                     inv_temperature, min_scaled)) {
      continue;
    }
    const uint32_t key =
        ds41rt_v41_order_key(k3_scaled(row_logits, token, inv_temperature));
    if (key < local_min) {
      local_min = key;
      local_worst_id = static_cast<uint32_t>(token);
    } else if (key == local_min && static_cast<uint32_t>(token) > local_worst_id) {
      local_worst_id = static_cast<uint32_t>(token);
    }
    if (key > local_max) {
      local_max = key;
    }
  }
  __shared__ uint32_t range_min[kBlock];
  __shared__ uint32_t range_max[kBlock];
  __shared__ uint32_t range_id[kBlock];
  range_min[tid] = local_min;
  range_max[tid] = local_max;
  range_id[tid] = local_worst_id;
  __syncthreads();
  for (int stride = kBlock / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      if (range_min[tid + stride] < range_min[tid]) {
        range_min[tid] = range_min[tid + stride];
        range_id[tid] = range_id[tid + stride];
      } else if (range_min[tid + stride] == range_min[tid] &&
                 range_id[tid + stride] > range_id[tid]) {
        range_id[tid] = range_id[tid + stride];
      }
      range_max[tid] = (range_max[tid + stride] > range_max[tid])
          ? range_max[tid + stride]
          : range_max[tid];
    }
    __syncthreads();
  }
  if (tid == 0) {
    min_max[0] = range_min[0];
    min_max[1] = range_max[0];
    min_max[2] = range_id[0];
  }
  __syncthreads();
}

/* `M >= threshold` alone accepts an EMPTY set when `threshold == 0`, because
 * every mass is `>= 0.0`. The draw's `uniform == 0` case reaches exactly that
 * (`target = 0 * nucleus_mass`), and the search would then return token id 0
 * even when token 0 is masked out or is not a survivor -- a non-member draw.
 * The CPU never does that: its loop adds a weight *before* testing
 * `target <= cumulative`, so at `target = 0` it still selects the first rank of
 * the ordered set. Requiring a strictly positive mass is the same statement:
 * for `threshold > 0` it is implied by `M >= threshold` (all weights are
 * non-negative, and a satisfying `M` is positive), and for `threshold == 0` it
 * skips every empty prefix and lands on the best actual survivor (the highest
 * key, lowest id among its ties) -- the CPU's rank 0. */
__device__ __forceinline__ bool k5_mass_satisfies(float mass, float threshold) {
  return mass > 0.0f && mass >= threshold;
}

/* The largest key with `M(key) >= threshold`, as `(value, id)`: the design's
 * two-level search (§4.5), a **binary** search on the 32-bit ordered value and
 * then a binary search on the token id inside the boundary tie group. Both
 * predicates are monotone, so termination is structural and the pass count is
 * `2 + ceil(log2(value_hi - value_lo + 1)) + ceil(log2(id_hi + 1))`.
 *
 * `value_lo` must satisfy `M(order_key >= value_lo) >= threshold` (the caller
 * passes a proven bound); `value_hi` is an upper bound on the search and may or
 * may not satisfy it. `found == false` means even the whole `value_hi` tie group
 * falls short, which for the top-p search is the design's "no key satisfies"
 * fallback and for the draw is unreachable (`M(K_p) = nucleus_mass >= target`,
 * and both are read from the same accumulation). */
__device__ __forceinline__ void k5_largest_key_with_mass(
    const ds41rt_v41_sampler_row_t& row, const uint32_t* mask_words,
    size_t mask_words_per_row, size_t mask_rows, const float* row_logits, size_t vocab,
    float inv_temperature, float max_scaled, float min_scaled, float threshold,
    float divisor, uint32_t value_lo, uint32_t value_hi, uint32_t id_hi, uint32_t tie_value,
    uint32_t tie_id_hi, float* phase, uint32_t* count_scratch, int tid,
    uint32_t* out_value, uint32_t* out_id, bool* out_found) {
  /* Value level. */
  uint32_t lo = value_lo;
  uint32_t hi = value_hi;
  if (k5_mass_satisfies(
          k5_key_mass(row, mask_words, mask_words_per_row, mask_rows, row_logits, vocab,
                      inv_temperature, max_scaled, min_scaled, hi, 0xFFFFFFFFu, divisor,
                      phase, nullptr, nullptr, tid),
          threshold)) {
    lo = hi;
  } else {
    while (hi - lo > 1u) {
      const uint32_t mid = lo + (hi - lo) / 2u;
      const float mass = k5_key_mass(row, mask_words, mask_words_per_row, mask_rows,
                                     row_logits, vocab, inv_temperature, max_scaled,
                                     min_scaled, mid, 0xFFFFFFFFu, divisor, phase, nullptr,
                                     nullptr, tid);
      if (k5_mass_satisfies(mass, threshold)) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
  }
  const uint32_t value = lo;
  /* Tie level: the smallest id in the tie group whose inclusive mass reaches the
   * threshold. `id_hi` is the largest id still allowed by the caller's interval
   * (the full id range, or the nucleus crossing's id when the value level chose
   * the same value). */
  uint32_t id = 0u;
  bool found = true;
  const uint32_t final_id_hi = (value == tie_value) ? tie_id_hi : id_hi;
  if (k5_mass_satisfies(
          k5_key_mass(row, mask_words, mask_words_per_row, mask_rows, row_logits, vocab,
                      inv_temperature, max_scaled, min_scaled, value, 0u, divisor, phase,
                      nullptr, nullptr, tid),
          threshold)) {
    id = 0u;
  } else {
    uint32_t low = 0u;
    uint32_t high = final_id_hi;
    if (!k5_mass_satisfies(
            k5_key_mass(row, mask_words, mask_words_per_row, mask_rows, row_logits,
                        vocab, inv_temperature, max_scaled, min_scaled, value, high,
                        divisor, phase, nullptr, nullptr, tid),
            threshold)) {
      found = false;
    } else {
      while (high - low > 1u) {
        const uint32_t mid = low + (high - low) / 2u;
        const float mass = k5_key_mass(row, mask_words, mask_words_per_row, mask_rows,
                                       row_logits, vocab, inv_temperature, max_scaled,
                                       min_scaled, value, mid, divisor, phase, nullptr,
                                       nullptr, tid);
        if (k5_mass_satisfies(mass, threshold)) {
          high = mid;
        } else {
          low = mid;
        }
      }
      id = high;
    }
  }
  *out_value = value;
  *out_id = id;
  *out_found = found;
}


/* K5's loud per-row failure. A row that reached the K5 row class but cannot
 * produce a defined token (a retained list wider than `kBlock`, a capacity-0
 * selection-only K3/K4, a `top_p` that makes K2 and K5 both inapplicable, or a
 * zero-survivor row) must not leave `out_indices` at the caller's sentinel while
 * the entry point returns OK. `scratch[block_row].status` was already the
 * documented channel; `out_status[output_row]` is the SAME channel K1 uses, so a
 * caller that only reads the returned status now sees the failure too. */
__device__ __forceinline__ void k5_mark_internal(
    ds41rt_v41_sampler_scratch_t* scratch, size_t block_row, size_t output_row,
    uint32_t* out_status, int tid) {
  if (tid == 0) {
    ds41rt_v41_sampler_scratch_t value = scratch[block_row];
    value.status = DS41RT_V41_SAMPLER_STATUS_INTERNAL;
    scratch[block_row] = value;
    if (out_status != nullptr) {
      out_status[output_row] = DS41RT_V41_SAMPLER_STATUS_INTERNAL;
    }
  }
}

/* K5, `v41_sample_nucleus_kernel` (design §4.5, §4.6).
 *
 * Weights `w_r = expf(scaled_r - max_scaled)` over `S`; `total = max(Σ w_r,
 * 1e-20f)`; the top-p boundary and the draw both run on the CPU's per-token
 * normalized masses `p_r = w_r / total` (`target_sampling.rs:540-542`). The
 * default build (`DS41RT_V41_K5_NORMALIZED_MASS=1`) accumulates those quotients
 * directly; the `=0` build instead uses the algebraically equivalent
 * un-normalized form `Σ w_r >= threshold * total` so no probe divides. The two
 * are equal in real arithmetic only; the report's A/B measurement is why the
 * normalized form is the default (retained-domain token mismatch 11.42 % -> 0).
 *
 * `S` is K3/K4's rank-ordered retained list when it exists (K4's ids are the
 * CPU's `ranked[..k]` exactly), and the full survivor set otherwise
 * (`top_k >= survivor_count` and `top_k == 0`), enumerated in ascending token
 * order exactly as the CPU's full-ordering branches enumerate it.
 *
 * There is deliberately **no `top_p >= 1.0` special case**. `top_p = 1.0` is
 * reachable on the ordered path (the served `temperature 0.7 + top_k 40`
 * profile sets it with a finite `top_k`, and `target_sampling.rs:463-466` fails
 * on `top_k.is_some()`), and the f32 running prefix can reach `1.0` before the
 * last survivor, so the nucleus can be a strict prefix. The boundary search
 * always runs.
 *
 * Both searches are **binary searches on monotone predicates**, so termination
 * is structural: `hi - lo` halves every pass and the loops stop at an interval
 * of length one -- at most `ceil(log2(count))` passes, far inside
 * `DS41RT_V41_NUCLEUS_MAX_PASSES` for any vocabulary up to `2^32`.
 *
 *   - `crossing` is the smallest rank whose inclusive mass reaches the clamped
 *     `top_p`, i.e. the CPU's `if nucleus_mass >= top_p { break }` crossing; the
 *     nucleus is ranks `0..crossing`. `crossing == count` means no rank crossed:
 *     the CPU consumed every weight without breaking (reachable in the
 *     normalized build when the f32 prefix rounds below `top_p`), and the
 *     nucleus is the full set. The draw still runs over the whole set there --
 *     the CPU's `selected = nucleus_count - 1` is only its initial value, not a
 *     hardcoded last rank.
 *   - `chosen` is the smallest rank `<= nucleus_last` whose inclusive mass
 *     reaches `target`. `mass(nucleus_last) = nucleus_mass >= target` (the clamp
 *     keeps `uniform <= MAX_UNIFORM < 1`), so a crossing always exists, and
 *     `target` is read from the same fixed accumulation the crossing searched.
 *
 * Outputs: `out_indices[output_row]` is the selected token for every ordered
 * row; `out_total[output_row] = total` and `out_nucleus_count[output_row] =
 * nucleus_last + 1` are written under `DIAGNOSE` only and may be null (design
 * D3). */
__global__ void v41_sample_nucleus_kernel(
    const float* logits, size_t rows, size_t vocab, size_t logits_stride,
    const ds41rt_v41_sampler_row_t* params, const uint32_t* mask_words,
    size_t mask_words_per_row, const uint32_t* rank_order_ids,
    size_t rank_order_capacity, const uint32_t* rank_retained_count,
    uint32_t* out_indices, uint32_t* out_status, float* out_total,
    uint32_t* out_nucleus_count, ds41rt_v41_sampler_scratch_t* scratch) {
  __shared__ float weights[kBlock];
  __shared__ float prefix[kBlock];
  __shared__ float phase[4 * kBlock];
  __shared__ uint32_t rank_counts[kBlock];
  __shared__ float shared_f32;

  const size_t block_row = blockIdx.x;
  const int tid = threadIdx.x;
  if (block_row >= rows) {
    return;
  }
  const ds41rt_v41_sampler_row_t row = params[block_row];
  const ds41rt_v41_sampler_scratch_t state = scratch[block_row];
  const size_t output_row = static_cast<size_t>(row.output_row);
  const bool greedy = (row.temperature < 1.0e-5f) || (row.top_k == 1u) ||
      ((row.flags & DS41RT_V41_SAMPLER_FLAG_GREEDY) != 0u);
  if (greedy || state.status != DS41RT_V41_SAMPLER_STATUS_OK) {
    /* Not a K5 row: K1 (or K2) already produced the token or the error status,
     * and K5 must not touch it. */
    return;
  }
  if (!isfinite(row.top_p)) {
    /* `top_p = NaN` makes `top_p >= 1.0` (K2) and `top_p < 1.0` (K5) both false,
     * so without this guard the row would silently keep the caller's sentinel.
     * The CPU rejects a non-finite `top_p` at construction and the FFI validator
     * rejects it on the host; this is the raw-C loud path. It also covers
     * `top_k != 0`, where the `fminf/fmaxf` clamp below would otherwise turn the
     * NaN into `1e-6` and sample a nucleus the CPU never would. */
    k5_mark_internal(scratch, block_row, output_row, out_status, tid);
    return;
  }
  if (state.survivor_count == 0u) {
    /* K1 reported OK with no surviving token: every stage (K2/K3/K4/K5) skips
     * and the caller would see an unwritten index. On a valid row the best
     * allowed token always survives (`min_p <= 1`), so this is a raw-C-only
     * shape; make it loud. */
    k5_mark_internal(scratch, block_row, output_row, out_status, tid);
    return;
  }
  if (!k5_ordered(row, state)) {
    /* The disjoint K2 fast path (`top_k == 0 && top_p >= 1.0`); K2 owns the
     * token and K5 must not touch it. */
    return;
  }
  if (output_row != block_row) {
    /* Unconditional row-identity guard, BEFORE any retained count is read. K5
     * publishes every output by `output_row` but consumes K3/K4's
     * `rank_retained_count` and rank-order arena by BLOCK row, so a non-identity
     * mapping pairs a row with another row's retained count. The pre-fix guard
     * tested identity only inside the `retained_mode &&` conjunction, so a
     * foreign count of 0 flipped `retained_mode` false and silently routed the
     * row through the survivor domain -- exactly the shape the check exists to
     * stop. The FFI validator rejects a non-identity batch on the host; this is
     * the raw-C loud path. */
    k5_mark_internal(scratch, block_row, output_row, out_status, tid);
    return;
  }
  const float* row_logits = logits + block_row * logits_stride;
  const float inv_temperature = state.inv_temperature;
  const float max_scaled = state.max_scaled;
  const float min_scaled = (row.min_p > 0.0f)
      ? max_scaled + row.ln_min_p
      : -CUDART_INF_F;
  const float top_p = fminf(fmaxf(row.top_p, 1.0e-6f), 1.0f);
  const float uniform = ds41rt_v41_target_clamp_uniform(
      ds41rt_v41_target_uniform(row.seed, row.position));

  /* Domain selection. K3/K4 publish `out_retained_count[r] == top_k` exactly
   * when they materialized a rank-ordered list; a no-op row (and every
   * `top_k == 0` row) leaves it 0, so the survivor domain is the CPU's
   * full-ordering branch. `rank_retained_count` and `rank_order_ids` are keyed
   * by BLOCK row (K4's arena layout), which is only coherent while
   * `output_row == block_row`; the unconditional identity guard above already
   * rejected every non-identity row, and the FFI validator rejects a
   * non-identity batch on the host (see the chunk-3b report, finding F4). */
  const uint32_t retained = rank_retained_count[block_row];
  const bool retained_mode = retained > 0u;
  const uint32_t count = retained_mode ? retained : state.survivor_count;
  if (retained_mode && (rank_order_ids == nullptr ||
                        static_cast<size_t>(retained) > rank_order_capacity ||
                        retained > kBlock)) {
    /* Raw-C shape violation (the FFI wrapper rejects both on the host) or a
     * retained list wider than the shared weight table. Either way K5 cannot
     * produce a defined token, so it reports INTERNAL on the caller-visible
     * status channel rather than sampling from a partial set. `retained > kBlock`
     * IS reachable through the FFI: the validator only requires
     * `rank_order_capacity >= max_r params[r].top_k`, so a `top_k` in
     * `[kBlock + 1, survivor_count)` (e.g. 300) validates, K3/K4 materialize it,
     * and each such row gets `INTERNAL` on `out_status` and an unwritten
     * `out_indices` -- no longer a silent OK. The same holds for a capacity-0
     * (selection-only) K3/K4 where K4 still publishes `out_retained_count =
     * top_k` for eligible rows. */
    k5_mark_internal(scratch, block_row, output_row, out_status, tid);
    return;
  }
  const uint32_t* const rank_ids =
      retained_mode ? rank_order_ids + block_row * rank_order_capacity : nullptr;

  /* The retained rank table, `w_r = expf(scaled_r - max_scaled)`. Filled once,
   * AND the fixed-association inclusive prefix `prefix[r] = W(r)` is built from
   * it once, so no search pass recomputes a weight and no rank's mass depends on
   * the search bounds that query it. */
  if (retained_mode) {
    for (uint32_t rank = static_cast<uint32_t>(tid); rank < count;
         rank += static_cast<uint32_t>(blockDim.x)) {
      weights[rank] = k5_weight(row_logits, static_cast<size_t>(rank_ids[rank]),
                                inv_temperature, max_scaled);
    }
    __syncthreads();
  }

  if (!retained_mode) {
    /* ---------------------------------------------------------------------
     * Survivor domain (`top_k == 0`, or `top_k >= survivor_count`): K3/K4 are
     * no-ops, so the CPU's ordered set is the full survivor list sorted by
     * `key = (order_key(scaled) << 32) | ~id` (design §4.5 "Boundary
     * primitive"). Both boundaries are "largest key with `M(key) >=
     * threshold`", found by the two-level search in
     * `k5_largest_key_with_mass`.
     * ------------------------------------------------------------------- */
    k5_key_range(row, mask_words, mask_words_per_row, rows, row_logits, vocab,
                 inv_temperature, min_scaled, rank_counts, tid);
    const uint32_t max_key = rank_counts[1];
    const uint32_t min_key = rank_counts[0];
    (void)min_key; /* read only by the normalized-mass fallback below */
    const uint32_t worst_id = rank_counts[2];
    /* The pass that derives the total never divides (`divisor == 1.0f`); every
     * later pass divides by it when `DS41RT_V41_K5_NORMALIZED_MASS` is set. */
    const float raw_total =
        k5_key_mass(row, mask_words, mask_words_per_row, rows, row_logits, vocab,
                    inv_temperature, max_scaled, min_scaled, 0u, 0xFFFFFFFFu, 1.0f,
                    phase, nullptr, nullptr, tid);
    if (tid == 0) {
      shared_f32 = fmaxf(raw_total, 1.0e-20f);
    }
    __syncthreads();
    const float total = shared_f32;
#if DS41RT_V41_K5_NORMALIZED_MASS
    const float top_p_mass = top_p;
    const float divisor = total;
#else
    const float top_p_mass = top_p * total;
    const float divisor = 1.0f;
#endif
    const uint32_t id_top = static_cast<uint32_t>(vocab - 1u);

    uint32_t p_value = 0u;
    uint32_t p_id = 0u;
    bool p_found = false;
    k5_largest_key_with_mass(row, mask_words, mask_words_per_row, rows, row_logits,
                             vocab, inv_temperature, max_scaled, min_scaled, top_p_mass,
                             divisor, 0u, max_key, id_top, 0xFFFFFFFFu, 0u, phase, nullptr,
                             tid, &p_value, &p_id, &p_found);
    if (!p_found) {
#if DS41RT_V41_K5_NORMALIZED_MASS
      /* Reachable in the normalized variant: the CPU's normalized survivor mass can
       * round below `clamp(top_p)`, in which case the CPU's nucleus loop consumes
       * every weight and the nucleus is the FULL survivor set. `K_p` is then the
       * worst key (the full-set rank), and the draw below still runs over the whole
       * set -- exactly `selected = nucleus_count - 1` being overwritten by the
       * genuine draw. */
      p_value = min_key;
      p_id = worst_id;
      p_found = true;
#else
      /* DEAD CODE on valid rows, kept only as a defined fallback for a raw-C
       * caller. `total >= w_best = expf(0) = 1` (the best survivor attains
       * `max_scaled`), and `fl(top_p * total) <= total` for any `top_p <= 1`, so
       * the largest key always satisfies `M(K) >= threshold` and the search's
       * own predicate is already true at its lower bound. The CPU's
       * *consumed-every-weight* case is therefore handled by the NORMAL path
       * below (the crossing is the last rank of the search interval), not here;
       * this branch's answer (the worst key) would NOT match the CPU's genuine
       * full-set draw, which is why it must stay unreachable rather than merely
       * unlikely. */
      if (tid == 0) {
        out_indices[output_row] = worst_id;
        if ((row.flags & DS41RT_V41_SAMPLER_FLAG_DIAGNOSE) != 0u) {
          if (out_total != nullptr) {
            out_total[output_row] = total;
          }
          if (out_nucleus_count != nullptr) {
            out_nucleus_count[output_row] = state.survivor_count;
          }
        }
      }
      return;
#endif
    }

    uint32_t nucleus_count = 0u;
    const float nucleus_mass = k5_key_mass(
        row, mask_words, mask_words_per_row, rows, row_logits, vocab, inv_temperature,
        max_scaled, min_scaled, p_value, p_id, divisor, phase, rank_counts, &nucleus_count,
        tid);
    if (tid == 0) {
      shared_f32 = uniform * fmaxf(nucleus_mass, 1.0e-20f);
    }
    __syncthreads();
    const float target = shared_f32;

    /* The draw's answer is at least as good as `K_p`, so its ordered value is at
     * least `p_value` and, if it is exactly `p_value`, its id is at most
     * `p_id`; those are the search's lower bounds (design §4.5 "Reuse the
     * value/tie intervals"). */
    uint32_t s_value = p_value;
    uint32_t s_id = p_id;
    bool s_found = false;
    k5_largest_key_with_mass(row, mask_words, mask_words_per_row, rows, row_logits,
                             vocab, inv_temperature, max_scaled, min_scaled, target,
                             divisor, p_value, max_key, id_top, p_value, p_id, phase,
                             nullptr, tid, &s_value, &s_id, &s_found);
    if (!s_found) {
      /* Unreachable by construction (`M(K_p) = nucleus_mass >= target`, and both
       * are read from the same accumulation), but if a malformed row made it
       * reachable the CPU's own fallback is `selected = nucleus_count - 1`, the
       * worst key of the nucleus, which is `K_p` itself. */
      s_value = p_value;
      s_id = p_id;
    }
    if (tid == 0) {
      out_indices[output_row] = s_id;
      if ((row.flags & DS41RT_V41_SAMPLER_FLAG_DIAGNOSE) != 0u) {
        if (out_total != nullptr) {
          out_total[output_row] = total;
        }
        if (out_nucleus_count != nullptr) {
          out_nucleus_count[output_row] = nucleus_count;
        }
      }
    }
    (void)s_value;
    return;
  }

  /* -----------------------------------------------------------------------
   * Retained domain: K3/K4 materialized the exact CPU rank order, so the
   * nucleus and the draw are rank-domain searches over that list.
   *
   * `prefix[r]` is the inclusive mass through rank `r`, built ONCE by a single
   * sequential walk of the shared weight table. It has ONE fixed f32
   * association, so `W(rank)` returns the same bits whatever search state
   * queries it. The previous bucket-split probe made the association a function
   * of `(lo, m0, m1)`: a rank could read `fl(1 + 2^-24 + 2^-24) =
   * 1.0000001192092896` when it was the middle bound and `1.0` when it was the
   * lower bound, so the binary search's "one fixed monotone predicate"
   * assumption did not hold bit-exactly, and the crossing search and the draw
   * could read different masses for the same rank. With
   * `DS41RT_V41_K5_NORMALIZED_MASS` the walk is the CPU's own `Sigma (w/total)`
   * in rank order, which makes the prefix bit-identical to
   * `sample_from_ranked`'s `nucleus_mass` accumulation.
   * --------------------------------------------------------------------- */
  if (tid == 0) {
    float raw_total = 0.0f;
    for (uint32_t rank = 0; rank < count; ++rank) {
      raw_total += weights[rank];
    }
    raw_total = fmaxf(raw_total, 1.0e-20f);
    float running = 0.0f;
#if DS41RT_V41_K5_NORMALIZED_MASS
    for (uint32_t rank = 0; rank < count; ++rank) {
      running += weights[rank] / raw_total;
      prefix[rank] = running;
    }
#else
    for (uint32_t rank = 0; rank < count; ++rank) {
      running += weights[rank];
      prefix[rank] = running;
    }
#endif
    shared_f32 = raw_total; /* the CPU's `RankedSample.total` */
  }
  __syncthreads();
  const float total = shared_f32;
#if DS41RT_V41_K5_NORMALIZED_MASS
  const float top_p_mass = top_p;
#else
  const float top_p_mass = top_p * total;
#endif

  /* Top-p crossing: the smallest rank with `W(rank) >= top_p_mass`. Every thread
   * reads the same fixed `prefix`, so the answer is identical in every thread and
   * no snapshot/barrier round trip is needed. The un-normalized prefix always
   * reaches `top_p_mass` at `count - 1` (`prefix[count-1] == total` and
   * `fl(top_p * total) <= total`); the normalized prefix can fall short, which is
   * exactly the CPU's *consumed every weight* case and must run the draw over the
   * full set rather than short-circuiting to the last rank. */
  uint32_t crossing = count;
  if (prefix[0] >= top_p_mass) {
    crossing = 0u;
  } else if (prefix[count - 1u] >= top_p_mass) {
    uint32_t lo = 0u;
    uint32_t hi = count - 1u;
    while (hi - lo > 1u) {
      const uint32_t mid = lo + (hi - lo) / 2u;
      if (prefix[mid] >= top_p_mass) {
        hi = mid;
      } else {
        lo = mid;
      }
    }
    crossing = hi;
  }
  const uint32_t nucleus_last = (crossing < count) ? crossing : (count - 1u);
  /* `nucleus_mass` is the SAME `prefix[nucleus_last]` the crossing search read
   * (or the fixed full-set value when no rank crossed), so the draw's target can
   * never disagree with the boundary that produced it. */
  const float nucleus_mass = fmaxf(prefix[nucleus_last], 1.0e-20f);
  const float target = uniform * nucleus_mass;

  /* Draw crossing: the smallest rank `s <= nucleus_last` with `W(s) >= target`.
   * `prefix[nucleus_last] = nucleus_mass >= target` because `uniform < 1`, so a
   * crossing always exists. */
  uint32_t chosen = nucleus_last;
  if (prefix[0] >= target) {
    chosen = 0u;
  } else {
    uint32_t lo = 0u;
    uint32_t hi = nucleus_last;
    while (hi - lo > 1u) {
      const uint32_t mid = lo + (hi - lo) / 2u;
      if (prefix[mid] >= target) {
        hi = mid;
      } else {
        lo = mid;
      }
    }
    chosen = hi;
  }

  if (tid == 0) {
    out_indices[output_row] = rank_ids[chosen];
    if ((row.flags & DS41RT_V41_SAMPLER_FLAG_DIAGNOSE) != 0u) {
      if (out_total != nullptr) {
        out_total[output_row] = total;
      }
      if (out_nucleus_count != nullptr) {
        out_nucleus_count[output_row] = nucleus_last + 1u;
      }
    }
  }
}

ds41rt_status_t validate_topk_select_args(
    const float* logits, size_t rows, size_t vocab, size_t logits_stride,
    const ds41rt_v41_sampler_row_t* params, const uint32_t* mask_words,
    size_t mask_words_per_row, uint32_t* rank_order_ids,
    uint64_t* rank_order_scratch, size_t rank_order_capacity,
    uint32_t* out_retained_count, uint32_t* out_pivot_passes,
    ds41rt_v41_sampler_scratch_t* scratch) {
  if (logits == nullptr || params == nullptr || out_retained_count == nullptr ||
      out_pivot_passes == nullptr || scratch == nullptr) {
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
  if (rank_order_capacity > 0u &&
      (rank_order_ids == nullptr || rank_order_scratch == nullptr)) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  return DS41RT_STATUS_OK;
}

ds41rt_status_t validate_nucleus_args(
    const float* logits, size_t rows, size_t vocab, size_t logits_stride,
    const ds41rt_v41_sampler_row_t* params, const uint32_t* mask_words,
    size_t mask_words_per_row, const uint32_t* rank_order_ids,
    size_t rank_order_capacity, const uint32_t* rank_retained_count,
    uint32_t* out_indices, ds41rt_v41_sampler_scratch_t* scratch) {
  if (logits == nullptr || params == nullptr || out_indices == nullptr ||
      rank_retained_count == nullptr || scratch == nullptr) {
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
  if (rank_order_capacity > 0u && rank_order_ids == nullptr) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  return DS41RT_STATUS_OK;
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

/* Chunk 3a entry points: K3 then K4 on the same stream, reading K1's `scratch`
 * exactly as K2 does. Purely additive — the K1/K2 entry points above are
 * unchanged, so the chunk-1/chunk-2 production path and its daemon caller do not
 * move. See the rank-order contract in `v41_sampling_gpu.h`. */
extern "C" ds41rt_status_t ds41rt_cuda_v41_topk_select_async(
    const float* logits, size_t rows, size_t vocab, size_t logits_stride,
    const ds41rt_v41_sampler_row_t* params, const uint32_t* mask_words,
    size_t mask_words_per_row, uint32_t* rank_order_ids,
    uint64_t* rank_order_scratch, size_t rank_order_capacity,
    uint32_t* out_retained_count, uint32_t* out_pivot_passes,
    ds41rt_v41_sampler_scratch_t* scratch, void* cuda_stream) {
  const ds41rt_status_t valid = validate_topk_select_args(
      logits, rows, vocab, logits_stride, params, mask_words, mask_words_per_row,
      rank_order_ids, rank_order_scratch, rank_order_capacity, out_retained_count,
      out_pivot_passes, scratch);
  if (valid != DS41RT_STATUS_OK) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  v41_sample_topk_pivot_kernel<<<static_cast<unsigned int>(rows), kBlock, 0, stream>>>(
      logits, rows, vocab, logits_stride, params, mask_words, mask_words_per_row,
      out_pivot_passes, scratch);
  const ds41rt_status_t pivot = status_from_cuda(cudaGetLastError());
  if (pivot != DS41RT_STATUS_OK) {
    return pivot;
  }
  v41_sample_topk_membership_kernel<<<static_cast<unsigned int>(rows), kBlock, 0,
                                      stream>>>(
      logits, rows, vocab, logits_stride, params, mask_words, mask_words_per_row,
      rank_order_ids, rank_order_scratch, rank_order_capacity, out_retained_count,
      scratch);
  return status_from_cuda(cudaGetLastError());
}

extern "C" ds41rt_status_t ds41rt_cuda_v41_topk_select(
    const float* logits, size_t rows, size_t vocab, size_t logits_stride,
    const ds41rt_v41_sampler_row_t* params, const uint32_t* mask_words,
    size_t mask_words_per_row, uint32_t* rank_order_ids,
    uint64_t* rank_order_scratch, size_t rank_order_capacity,
    uint32_t* out_retained_count, uint32_t* out_pivot_passes,
    ds41rt_v41_sampler_scratch_t* scratch) {
  const ds41rt_status_t status = ds41rt_cuda_v41_topk_select_async(
      logits, rows, vocab, logits_stride, params, mask_words, mask_words_per_row,
      rank_order_ids, rank_order_scratch, rank_order_capacity, out_retained_count,
      out_pivot_passes, scratch, nullptr);
  if (status != DS41RT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

/* Chunk 3b entry points: K5 on the same stream after K1 and K3/K4, reading
 * K1's `scratch` and K3/K4's rank-ordered retained list exactly as K2 reads
 * K1's scratch. Purely additive -- K1, K2 and K3/K4 above are unchanged, so the
 * chunk-1..3a paths and their daemon caller do not move. See the K5 contract in
 * `v41_sampling_gpu.h`. */
extern "C" ds41rt_status_t ds41rt_cuda_v41_nucleus_async(
    const float* logits, size_t rows, size_t vocab, size_t logits_stride,
    const ds41rt_v41_sampler_row_t* params, const uint32_t* mask_words,
    size_t mask_words_per_row, const uint32_t* rank_order_ids,
    size_t rank_order_capacity, const uint32_t* rank_retained_count,
    uint32_t* out_indices, uint32_t* out_status, float* out_total,
    uint32_t* out_nucleus_count, ds41rt_v41_sampler_scratch_t* scratch,
    void* cuda_stream) {
  const ds41rt_status_t valid = validate_nucleus_args(
      logits, rows, vocab, logits_stride, params, mask_words, mask_words_per_row,
      rank_order_ids, rank_order_capacity, rank_retained_count, out_indices, scratch);
  if (valid != DS41RT_STATUS_OK) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  v41_sample_nucleus_kernel<<<static_cast<unsigned int>(rows), kBlock, 0, stream>>>(
      logits, rows, vocab, logits_stride, params, mask_words, mask_words_per_row,
      rank_order_ids, rank_order_capacity, rank_retained_count, out_indices, out_status,
      out_total, out_nucleus_count, scratch);
  return status_from_cuda(cudaGetLastError());
}

extern "C" ds41rt_status_t ds41rt_cuda_v41_nucleus(
    const float* logits, size_t rows, size_t vocab, size_t logits_stride,
    const ds41rt_v41_sampler_row_t* params, const uint32_t* mask_words,
    size_t mask_words_per_row, const uint32_t* rank_order_ids,
    size_t rank_order_capacity, const uint32_t* rank_retained_count,
    uint32_t* out_indices, uint32_t* out_status, float* out_total,
    uint32_t* out_nucleus_count, ds41rt_v41_sampler_scratch_t* scratch) {
  const ds41rt_status_t status = ds41rt_cuda_v41_nucleus_async(
      logits, rows, vocab, logits_stride, params, mask_words, mask_words_per_row,
      rank_order_ids, rank_order_capacity, rank_retained_count, out_indices, out_status,
      out_total, out_nucleus_count, scratch, nullptr);
  if (status != DS41RT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}
