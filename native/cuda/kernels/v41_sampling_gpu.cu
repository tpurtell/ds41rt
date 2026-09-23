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
 * sequential segment prefix `C(kSamplerBlock)`, so the macro no longer selects any
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

/* `kSamplerBlock` is the sampler's per-row CTA width; the knob, its shipped
 * default and the reason for it live in `v41_sampling_gpu.h`
 * (`DS41RT_V41_SAMPLER_BLOCK`) so the device selftest's host model reads the
 * same definition. */
constexpr int kSamplerBlock = DS41RT_V41_SAMPLER_CTA;
static_assert(kSamplerBlock % 32 == 0, "sampler CTA width must be a warp multiple");

/* One CTA per row. Per-thread state lives in registers; the reductions use
 * these shared arrays and a fixed binary-tree combine. */
struct Shared {
  float score[kSamplerBlock];       /* greedy argmax score */
  uint32_t id[kSamplerBlock];       /* greedy argmax token id */
  float scaled[kSamplerBlock];      /* stochastic scaled maximum */
  float inv_temperature[kSamplerBlock];
  uint32_t allowed[kSamplerBlock];  /* stochastic allowed-token count */
  uint32_t survivors[kSamplerBlock];/* stochastic min_p survivor count */
  uint32_t nonfinite[kSamplerBlock];/* lowest offending token id, or NO_DETAIL */
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
  for (int stride = kSamplerBlock / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      values[tid] = fmaxf(values[tid], values[tid + stride]);
    }
    __syncthreads();
  }
  return values[0];
}

__device__ __forceinline__ uint32_t tree_add_u32(uint32_t* values, int tid) {
  for (int stride = kSamplerBlock / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      values[tid] += values[tid + stride];
    }
    __syncthreads();
  }
  return values[0];
}

__device__ __forceinline__ uint32_t tree_min_u32(uint32_t* values, int tid) {
  for (int stride = kSamplerBlock / 2; stride > 0; stride >>= 1) {
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
  for (int stride = kSamplerBlock / 2; stride > 0; stride >>= 1) {
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

/* Fixed-shape Hillis-Steele inclusive scan over `kSamplerBlock` f32 values, in place.
 * The combine tree is the same for every row, so a row's result cannot depend on
 * the wave, the lane or a peer (design §4.0/§4.8). `values` must be initialised
 * and every thread must have reached the call before any read; callers insert a
 * `__syncthreads()` after the initial write.
 *
 * K2 no longer uses this (its segment prefix is now a single sequential fold so
 * the exclusive prefix is the crossing walk's own association); it is kept as a
 * fixed-tree reference and for the M6 pre-fix mutant. */
[[maybe_unused]] __device__ __forceinline__ float tree_inclusive_scan_f32(float* values, int tid) {
  for (int offset = 1; offset < kSamplerBlock; offset <<= 1) {
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
  return values[kSamplerBlock - 1];
}

/* Greedy argmax combine: strict `>` with the lowest id winning an exact tie,
 * matching the CPU's `argmax_allowed` (`target_sampling.rs:259-262`) and the
 * legacy device argmax (`sampling.cu:583-587`). */
__device__ __forceinline__ void tree_argmax(float* scores, uint32_t* ids, int tid) {
  for (int stride = kSamplerBlock / 2; stride > 0; stride >>= 1) {
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

/* K1. Grid: `rows` CTAs; block: `kSamplerBlock`; the mask is the first predicate.
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
 * The row is cut into `kSamplerBlock` contiguous segments. Each thread sums its segment
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
  __shared__ float inclusive[kSamplerBlock];
  __shared__ uint32_t hits[kSamplerBlock];
  __shared__ uint32_t lasts[kSamplerBlock];
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
  const size_t per_thread = (vocab + kSamplerBlock - 1) / kSamplerBlock;
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
   * `walk_total` becomes `C(kSamplerBlock)`, the single non-decreasing segment-level
   * cumulative both the crossing and the total are read from. This replaces the
   * Hillis-Steele tree scan. The tree's association made the prefix handed to
   * segment `i` differ from the value the crossing walk itself reaches at the
   * end of segment `i-1`: with a tail weight below half an ulp of 1.0 the
   * segment-local sum keeps the tails while a running fold does not, so the next
   * segment could start at a prefix already past `target` and "cross" at its
   * first survivor -- a token whose weight is exactly zero. The scan is at most
   * `kSamplerBlock` sequential shared-memory adds by one thread; pass 1's
   * per-segment sums and every min/max reduction below stay parallel.
   *
   * The old owner-segment re-walk that derived `total` from the tree prefix is
   * gone: `C(kSamplerBlock)` is the walk's own segment-level total, `target <= total`
   * for the clamped uniform, and no second accumulation is needed. */
  if (tid == 0) {
    float running = 0.0f;
    for (int segment = 0; segment < kSamplerBlock; ++segment) {
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
  const float end_prefix = (tid + 1 < kSamplerBlock) ? inclusive[tid + 1] : walk_total;
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

/* Fixed-shape Hillis-Steele inclusive scan over `kSamplerBlock` u32 values, in place.
 * The integer analogue of `tree_inclusive_scan_f32`; used by K4's per-segment
 * exclusive counts (the tie prefix and the compaction offsets). Callers insert a
 * `__syncthreads()` after the initial write. */
__device__ __forceinline__ void tree_inclusive_scan_u32(uint32_t* values, int tid) {
  for (int offset = 1; offset < kSamplerBlock; offset <<= 1) {
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

/* Exactly the `k3_survivor` predicate K1, K2, K4 and K5 use. */
__device__ __forceinline__ uint32_t k3_masked_key(const float* row_logits, size_t token,
                                                  float inv_temperature) {
  return ds41rt_v41_order_key(k3_scaled(row_logits, token, inv_temperature));
}

/* One histogram pass over the row's survivors, binning `(key >> shift)`.
 *
 * `bins` is zeroed first by the CTA. Every token lands in exactly one bin and
 * every bin add is an integer atomic, so the counts are exact and independent of
 * thread scheduling: the histogram is a pure function of the row (§4.0/§4.8
 * determinism). The trailing barrier is load-bearing -- the localization below
 * reads these shared counts in every thread and the next pass re-zeroes them.
 *
 * `lo`/`hi` are an inclusive-exclusive key interval: only survivors whose key is
 * inside it are binned, which is what lets a later pass refine the interval. */
__device__ __forceinline__ void k3_hist_pass(
    const ds41rt_v41_sampler_row_t& row, const uint32_t* mask_words,
    size_t mask_words_per_row, size_t mask_rows, const float* row_logits, size_t vocab,
    float inv_temperature, float min_scaled, uint32_t lo, uint32_t hi, uint32_t shift,
    uint32_t bin_mask, uint32_t* bins, int tid) {
  for (uint32_t b = static_cast<uint32_t>(tid); b <= bin_mask;
       b += static_cast<uint32_t>(blockDim.x)) {
    bins[b] = 0u;
  }
  __syncthreads();
  for (size_t token = static_cast<size_t>(tid); token < vocab;
       token += static_cast<size_t>(blockDim.x)) {
    if (!k3_survivor(row, mask_words, mask_words_per_row, mask_rows, row_logits, token,
                     inv_temperature, min_scaled)) {
      continue;
    }
    const uint32_t key =
        ds41rt_v41_order_key(k3_scaled(row_logits, token, inv_temperature));
    if (key < lo || key >= hi) {
      continue;
    }
    atomicAdd(&bins[(key >> shift) & bin_mask], 1u);
  }
  __syncthreads();
}

/* Descending walk of a pass histogram: the bin `m` whose running count from the
 * top first reaches `target = k - above` holds the k-th best survivor's key
 * prefix. Sections are visited in descending key order, exactly the rank order.
 *
 * `above = C_gt(hi) < k` is the caller's invariant, so `target >= 1` and a
 * bucket always reaches it. A bucket is an interval, not a key: `section_lo` is
 * only guaranteed to be `<= kth`. That is fine because each pass narrows by the
 * key's OWN bits and the last pass bins the low 8 bits, whose buckets are one
 * key wide, so its bucket index IS the k-th key's remaining bits. */
__device__ __forceinline__ bool k3_hist_localize(
    const uint32_t* bins, uint32_t bin_count, uint32_t k, uint32_t above, uint32_t lo,
    uint32_t hi, uint32_t bin_shift, uint32_t* out_lo, uint32_t* out_hi,
    uint32_t* out_above) {
  if (above >= k) {
    return false;
  }
  const uint32_t target = k - above;
  uint32_t cum = 0u;
  for (int bin = static_cast<int>(bin_count) - 1; bin >= 0; --bin) {
    cum += bins[bin];
    if (cum >= target) {
      const uint32_t b = static_cast<uint32_t>(bin);
      const uint32_t section_hi = lo + ((b + 1u) << bin_shift);
      const uint32_t section_lo = lo + (b << bin_shift);
      *out_above = above + (cum - bins[bin]);
      *out_lo = section_lo;
      *out_hi = (section_hi < hi) ? section_hi : hi;
      return *out_above < k;
    }
  }
  return false;
}

/* Chunk-6 K3: three histogram passes instead of up to 21 dual-probe pivots.
 *
 * The k-th largest `order_key` is localized by the key's own bits:
 *   pass 1  bins `key >> 20` (12 bits, 4096 bins) over [0, 2^32)
 *   pass 2  bins `(key >> 8) & 0xFFF` over the pass-1 interval (2^8 wide)
 *   pass 3  bins `key & 0xFF` over the pass-2 interval (one key per bucket)
 * Each pass carries `above = C_gt(hi)`, so `target = k - above >= 1` holds by
 * construction. Passes 2 and 3 are skipped when the interval is already one key
 * wide, so the pass count is 1..3 (0 for a no-op row).
 *
 * EXACTNESS. After pass 3 every bucket is exactly one key, so the bucket `m`
 * whose running count reaches `target` is the k-th value: `kth = lo + m`. The
 * number of survivors sharing that value is `bins[m]`, so the exact tie cut is
 * `above_count = k - bins[m]`, which is `C_gt(kth)` by the definition of `m`.
 * The shipped bisection needed a final extra probe to resolve a length-2
 * interval; here a one-key interval is resolved by the bucket index directly,
 * which is why this form is both cheaper and free of that termination corner.
 *
 * This is EXACT and general-k with no sort: every bound is an exact key and all
 * decisions are integer counts. The all-tied row is handled structurally: its
 * keys share one bucket at every level, so `m` is that bucket's index and the
 * tie cut admits the lowest-id equals. */
__global__ void v41_sample_topk_pivot_kernel(
    const float* logits, size_t rows, size_t vocab, size_t logits_stride,
    const ds41rt_v41_sampler_row_t* params, const uint32_t* mask_words,
    size_t mask_words_per_row, uint32_t* out_pivot_passes,
    ds41rt_v41_sampler_scratch_t* scratch) {
  __shared__ uint32_t bins[4096];

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

  uint32_t lo = 0u;
  uint32_t hi = 0xFFFFFFFFu;
  uint32_t above = 0u;
  uint32_t passes = 0u;
  uint32_t kth_key = 0xFFFFFFFFu;
  bool ok = true;
  bool exact = false;

  k3_hist_pass(row, mask_words, mask_words_per_row, rows, row_logits, vocab,
               inv_temperature, min_scaled, lo, hi, 20u, 4095u, bins, tid);
  ++passes;
  {
    uint32_t nlo = 0u, nhi = 0u, nabove = 0u;
    ok = k3_hist_localize(bins, 4096u, k, above, lo, hi, 20u, &nlo, &nhi, &nabove);
    if (ok) { lo = nlo; hi = nhi; above = nabove; }
    __syncthreads();
  }
  if (ok && hi - lo > 1u) {
    k3_hist_pass(row, mask_words, mask_words_per_row, rows, row_logits, vocab,
                 inv_temperature, min_scaled, lo, hi, 8u, 4095u, bins, tid);
    ++passes;
    uint32_t nlo = 0u, nhi = 0u, nabove = 0u;
    ok = k3_hist_localize(bins, 4096u, k, above, lo, hi, 8u, &nlo, &nhi, &nabove);
    if (ok) { lo = nlo; hi = nhi; above = nabove; }
    __syncthreads();
  }
  if (ok && hi - lo > 1u) {
    k3_hist_pass(row, mask_words, mask_words_per_row, rows, row_logits, vocab,
                 inv_temperature, min_scaled, lo, hi, 0u, 255u, bins, tid);
    ++passes;
    uint32_t nlo = 0u, nhi = 0u, nabove = 0u;
    ok = k3_hist_localize(bins, 256u, k, above, lo, hi, 0u, &nlo, &nhi, &nabove);
    if (ok) {
      const uint32_t m = nlo - lo;
      const uint32_t share = (m < 256u) ? bins[m] : 0u;
      kth_key = lo + m;
      /* `nabove` from the walk IS `C_gt(kth)`: it is `C_gt(hi)` at the pass-2
       * bound plus every survivor strictly above this one-key bucket. That is
       * exactly the rank-order contract's `above_count` (v41_sampling_gpu.h).
       * K4 independently re-derives the same count from the keys before
       * publishing, but the FFI's selection-only path reads K3's value, so it
       * must be `C_gt(kth)` and not `k - share`: those coincide only when
       * `C_ge(kth) == k`, and `k - share` under-counts by the size of a partial
       * tie group at the cut otherwise. */
      above = nabove;
      exact = (share > 0u) && (kth_key >= lo) && (kth_key < hi) && (above < k);
    }
    __syncthreads();
  } else if (ok && hi - lo == 1u) {
    kth_key = lo;
    exact = (above < k);
  }

  if (tid == 0) {
    ds41rt_v41_sampler_scratch_t value = state;
    if (!ok || !exact || kth_key == 0xFFFFFFFFu) {
      value.status = DS41RT_V41_SAMPLER_STATUS_INTERNAL;
      value.kth_value_bits = 0u;
      value.above_count = 0u;
    } else {
      value.kth_value_bits = __float_as_uint(ds41rt_v41_ordered_value(kth_key));
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
 * A single pass over `[0, vocab)` in `kSamplerBlock` contiguous segments gives each
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
  __shared__ uint32_t equal_counts[kSamplerBlock];
  __shared__ uint32_t above_counts[kSamplerBlock];
  __shared__ uint32_t retained_counts[kSamplerBlock];

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
  const size_t per_thread = (vocab + kSamplerBlock - 1) / kSamplerBlock;
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
  const uint32_t total_retained = retained_counts[kSamplerBlock - 1];

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
  for (int stride = kSamplerBlock / 2; stride > 0; stride >>= 1) {
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
  __shared__ uint32_t range_min[kSamplerBlock];
  __shared__ uint32_t range_max[kSamplerBlock];
  __shared__ uint32_t range_id[kSamplerBlock];
  range_min[tid] = local_min;
  range_max[tid] = local_max;
  range_id[tid] = local_worst_id;
  __syncthreads();
  for (int stride = kSamplerBlock / 2; stride > 0; stride >>= 1) {
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

/* Fixed-point mass is used only for the full-survivor radix search. A weight is
 * in [0, 1], so Q32 mass cannot overflow u64 for a u32-sized vocabulary. All
 * atomics are integer additions: scheduling cannot change the result. */
__device__ __forceinline__ unsigned long long k5_fixed_weight(float weight) {
  return static_cast<unsigned long long>(static_cast<double>(weight) * 4294967296.0 + 0.5);
}

/* Combine equal histogram bins within each warp before touching shared memory.
 * Three 16-bit reductions reconstruct an exact u64 Q32 sum without overflow:
 * each reduction contains at most 32 * 65535. All lanes participate, including
 * those with no eligible token (the sentinel bin never receives an atomic). */
__device__ __forceinline__ void k5_warp_add_mass(
    unsigned long long* bins, uint32_t bin, unsigned long long mass) {
  const uint32_t peers = __match_any_sync(0xFFFFFFFFu, bin);
  const uint32_t lo = __reduce_add_sync(peers, static_cast<uint32_t>(mass & 0xFFFFull));
  const uint32_t mid = __reduce_add_sync(peers, static_cast<uint32_t>((mass >> 16) & 0xFFFFull));
  const uint32_t hi = __reduce_add_sync(peers, static_cast<uint32_t>(mass >> 32));
  if (bin != 0xFFFFFFFFu && (threadIdx.x & 31) == (__ffs(peers) - 1)) {
    atomicAdd(&bins[bin], static_cast<unsigned long long>(lo) +
                           (static_cast<unsigned long long>(mid) << 16) +
                           (static_cast<unsigned long long>(hi) << 32));
  }
}

__device__ __forceinline__ void k5_warp_add_count(
    unsigned long long* bins, uint32_t bin) {
  const uint32_t peers = __match_any_sync(0xFFFFFFFFu, bin);
  const uint32_t count = __reduce_add_sync(peers, bin == 0xFFFFFFFFu ? 0u : 1u);
  if (bin != 0xFFFFFFFFu && (threadIdx.x & 31) == (__ffs(peers) - 1)) {
    atomicAdd(&bins[bin], static_cast<unsigned long long>(count));
  }
}

/* Select the value whose descending cumulative mass first reaches target.
 * The first pass also measures the fixed-point total for a top-p search. The
 * The three radix digits use an 11/11/10 split. `state` and `bins` are
 * shared across the CTA; every exit is uniform. */
__device__ __forceinline__ bool k5_radix_value(
    const ds41rt_v41_sampler_row_t& row, const uint32_t* mask_words,
    size_t mask_words_per_row, size_t rows, const float* row_logits, size_t vocab,
    float inv_temperature, float max_scaled, float min_scaled, float top_p,
    unsigned long long requested_target, unsigned long long* bins,
    unsigned long long* state, int tid, uint32_t* out_value,
    unsigned long long* out_above, unsigned long long* out_target,
    unsigned long long* out_total) {
  uint32_t prefix = 0u;
  unsigned long long above = 0ull;
  for (int pass = 0; pass < 3; ++pass) {
    const int shift = (pass == 0) ? 21 : ((pass == 1) ? 10 : 0);
    const uint32_t width = (pass == 2) ? 1024u : 2048u;
    for (uint32_t b = static_cast<uint32_t>(tid); b < width; b += blockDim.x) {
      bins[b] = 0ull;
    }
    __syncthreads();
    for (size_t base = 0; base < vocab; base += blockDim.x) {
      const size_t token = base + static_cast<size_t>(tid);
      uint32_t bin = 0xFFFFFFFFu;
      unsigned long long mass = 0ull;
      if (token < vocab &&
          k3_survivor(row, mask_words, mask_words_per_row, rows, row_logits, token,
                      inv_temperature, min_scaled)) {
        const uint32_t key = k3_masked_key(row_logits, token, inv_temperature);
        if (pass == 0 || (pass == 1 ? (key >> 21) : (key >> 10)) == prefix) {
          mass = k5_fixed_weight(k5_weight(row_logits, token, inv_temperature, max_scaled));
          if (mass != 0ull) bin = (key >> shift) & (width - 1u);
        }
      }
      k5_warp_add_mass(bins, bin, mass);
    }
    __syncthreads();
    if (tid == 0) {
      unsigned long long total = 0ull;
      for (uint32_t b = 0; b < width; ++b) total += bins[b];
      if (pass == 0) {
        state[2] = total;
        state[3] = (top_p >= 0.0f)
            ? static_cast<unsigned long long>(ceil(static_cast<double>(top_p) * total))
            : requested_target;
      }
      const unsigned long long target = state[3];
      unsigned long long cumulative = above;
      bool found = false;
      uint32_t selected = 0u;
      for (int b = static_cast<int>(width) - 1; b >= 0; --b) {
        const unsigned long long mass = bins[b];
        cumulative += mass;
        if (mass != 0ull && cumulative >= target) {
          selected = static_cast<uint32_t>(b);
          cumulative -= mass;
          found = true;
          break;
        }
      }
      state[0] = (static_cast<unsigned long long>(prefix) << ((pass == 2) ? 10 : 11)) | selected;
      state[1] = cumulative;
      state[4] = found ? 1ull : 0ull;
    }
    __syncthreads();
    if (state[4] == 0ull) return false;
    prefix = static_cast<uint32_t>(state[0]);
    above = state[1];
  }
  *out_value = prefix;
  *out_above = above;
  *out_target = state[3];
  *out_total = state[2];
  return true;
}

/* All tokens with one scaled value have the same weight. Select the nth one in
 * ascending token-id order with three count histograms, including sparse masks
 * and arbitrary vocab widths. */
__device__ __forceinline__ bool k5_radix_tie_id(
    const ds41rt_v41_sampler_row_t& row, const uint32_t* mask_words,
    size_t mask_words_per_row, size_t rows, const float* row_logits, size_t vocab,
    float inv_temperature, float min_scaled, uint32_t value, unsigned long long nth,
    unsigned long long* bins, unsigned long long* state, int tid, uint32_t* out_id) {
  uint32_t prefix = 0u;
  for (int pass = 0; pass < 3; ++pass) {
    const int shift = (pass == 0) ? 21 : ((pass == 1) ? 10 : 0);
    const uint32_t width = (pass == 2) ? 1024u : 2048u;
    for (uint32_t b = static_cast<uint32_t>(tid); b < width; b += blockDim.x) {
      bins[b] = 0ull;
    }
    __syncthreads();
    for (size_t base = 0; base < vocab; base += blockDim.x) {
      const size_t token = base + static_cast<size_t>(tid);
      uint32_t bin = 0xFFFFFFFFu;
      if (token < vocab &&
          (pass == 0 ||
           (pass == 1 ? (token >> 21) : (token >> 10)) == prefix) &&
          k3_survivor(row, mask_words, mask_words_per_row, rows, row_logits, token,
                      inv_temperature, min_scaled) &&
          k3_masked_key(row_logits, token, inv_temperature) == value) {
        bin = (static_cast<uint32_t>(token) >> shift) & (width - 1u);
      }
      k5_warp_add_count(bins, bin);
    }
    __syncthreads();
    if (tid == 0) {
      unsigned long long cumulative = 0ull;
      bool found = false;
      uint32_t selected = 0u;
      for (uint32_t b = 0; b < width; ++b) {
        cumulative += bins[b];
        if (bins[b] != 0ull && cumulative >= nth) {
          selected = b;
          nth -= cumulative - bins[b];
          found = true;
          break;
        }
      }
      state[0] = (static_cast<unsigned long long>(prefix) << ((pass == 2) ? 10 : 11)) | selected;
      state[3] = nth;
      state[4] = found ? 1ull : 0ull;
    }
    __syncthreads();
    if (state[4] == 0ull) return false;
    prefix = static_cast<uint32_t>(state[0]);
    nth = state[3];
  }
  *out_id = prefix;
  return prefix < vocab;
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
 * produce a defined token (a retained list wider than `kSamplerBlock`, a capacity-0
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
 * `top_p = 1.0` remains a real boundary search. It is reachable on the
 * retained path (`temperature 0.7 + top_k 40`), where the exact f32 prefix
 * may reach 1.0 before the last token. A full-survivor p=1 row uses the old
 * float search for that same saturation behavior; p<1 uses fixed-point radix.
 *
 * Full-survivor p<1 selection uses three mass-radix passes over the scaled
 * value and three count-radix passes over tied IDs for each boundary. The
 * retained list uses fixed-prefix binary search. Both have bounded passes.
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
  union K5Workspace {
    struct {
      float weights[kSamplerBlock];
      float prefix[kSamplerBlock];
    } retained;
    struct {
      float phase[kSamplerBlock];
      uint32_t rank_counts[kSamplerBlock];
      unsigned long long radix_bins[2048];
    } survivor;
  };
  __shared__ K5Workspace workspace;
  float* const weights = workspace.retained.weights;
  float* const prefix = workspace.retained.prefix;
  float* const phase = workspace.survivor.phase;
  uint32_t* const rank_counts = workspace.survivor.rank_counts;
  unsigned long long* const radix_bins = workspace.survivor.radix_bins;
  __shared__ float shared_f32;
  __shared__ unsigned long long radix_state[5];

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
                        retained > kSamplerBlock)) {
    /* Raw-C shape violation (the FFI wrapper rejects both on the host) or a
     * retained list wider than the shared weight table. Either way K5 cannot
     * produce a defined token, so it reports INTERNAL on the caller-visible
     * status channel rather than sampling from a partial set. `retained > kSamplerBlock`
     * IS reachable through the FFI: the validator only requires
     * `rank_order_capacity >= max_r params[r].top_k`, so a `top_k` in
     * `[kSamplerBlock + 1, survivor_count)` (e.g. 300) validates, K3/K4 materialize it,
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
    /* Survivor domain: K3/K4 are no-ops. For p<1, fixed-point mass radix
     * selection finds each scaled-value boundary, then count radix selection
     * finds the lowest-id tie cutoff. Integer atomics give a deterministic sum.
     * The float search remains for p=1, whose prefix may saturate early. */
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
    /* At p=1 the CPU's f32 prefix can saturate before the last survivor. The
     * integer sum deliberately does not saturate, so retain the original
     * fixed-association search for this boundary case. Served p<1 nucleus
     * profiles use the radix path below. */
    if (top_p >= 1.0f) {
      const uint32_t id_top = static_cast<uint32_t>(vocab - 1u);
      uint32_t p_value = 0u, p_id = 0u;
      bool p_found = false;
      k5_largest_key_with_mass(row, mask_words, mask_words_per_row, rows,
          row_logits, vocab, inv_temperature, max_scaled, min_scaled,
          top_p_mass, divisor, 0u, max_key, id_top, 0xFFFFFFFFu, 0u,
          phase, nullptr, tid, &p_value, &p_id, &p_found);
      if (!p_found) {
        p_value = min_key;
        p_id = worst_id;
      }
      uint32_t nucleus_count = 0u;
      const float nucleus_mass = k5_key_mass(row, mask_words,
          mask_words_per_row, rows, row_logits, vocab, inv_temperature,
          max_scaled, min_scaled, p_value, p_id, divisor, phase,
          rank_counts, &nucleus_count, tid);
      if (tid == 0) shared_f32 = uniform * fmaxf(nucleus_mass, 1.0e-20f);
      __syncthreads();
      uint32_t s_value = p_value, s_id = p_id;
      bool s_found = false;
      k5_largest_key_with_mass(row, mask_words, mask_words_per_row, rows,
          row_logits, vocab, inv_temperature, max_scaled, min_scaled,
          shared_f32, divisor, p_value, max_key, id_top, p_value, p_id,
          phase, nullptr, tid, &s_value, &s_id, &s_found);
      if (!s_found) s_id = p_id;
      if (tid == 0) {
        out_indices[output_row] = s_id;
        if ((row.flags & DS41RT_V41_SAMPLER_FLAG_DIAGNOSE) != 0u) {
          if (out_total != nullptr) out_total[output_row] = total;
          if (out_nucleus_count != nullptr) out_nucleus_count[output_row] = nucleus_count;
        }
      }
      return;
    }
    uint32_t p_value = 0u;
    uint32_t p_id = 0u;
    unsigned long long p_above = 0ull;
    unsigned long long p_target = 0ull;
    unsigned long long fixed_total = 0ull;
    bool p_found = k5_radix_value(
        row, mask_words, mask_words_per_row, rows, row_logits, vocab,
        inv_temperature, max_scaled, min_scaled, top_p, 0ull, radix_bins,
        radix_state, tid, &p_value, &p_above, &p_target, &fixed_total);
    if (!p_found) {
      /* On a valid row the best survivor has weight 1, and p<1 leaves an
       * integer threshold below the positive fixed-point total. Reaching this
       * branch means the row or scratch was corrupted; never publish a token. */
      k5_mark_internal(scratch, block_row, output_row, out_status, tid);
      return;
    }

    const unsigned long long p_weight = k5_fixed_weight(
        expf(ds41rt_v41_ordered_value(p_value) - max_scaled));
    const unsigned long long p_need = p_target > p_above ? p_target - p_above : 0ull;
    const unsigned long long p_nth = p_weight == 0ull ? 0ull :
        max(1ull, (p_need + p_weight - 1ull) / p_weight);
    if (p_nth == 0ull || !k5_radix_tie_id(
            row, mask_words, mask_words_per_row, rows, row_logits, vocab,
            inv_temperature, min_scaled, p_value, p_nth, radix_bins,
            radix_state, tid, &p_id)) {
      k5_mark_internal(scratch, block_row, output_row, out_status, tid);
      return;
    }
    uint32_t nucleus_count = 0u;
    const float nucleus_mass = k5_key_mass(
        row, mask_words, mask_words_per_row, rows, row_logits, vocab, inv_temperature,
        max_scaled, min_scaled, p_value, p_id, divisor, phase, rank_counts, &nucleus_count,
        tid);
    const unsigned long long fixed_nucleus_mass = p_above + p_nth * p_weight;
    const unsigned long long target = static_cast<unsigned long long>(
        ceil(static_cast<double>(uniform) * fixed_nucleus_mass));

    /* The draw's answer is at least as good as `K_p`, so its ordered value is at
     * least `p_value` and, if it is exactly `p_value`, its id is at most
     * `p_id`; those are the search's lower bounds (design §4.5 "Reuse the
     * value/tie intervals"). */
    uint32_t s_value = 0u;
    uint32_t s_id = 0u;
    unsigned long long s_above = 0ull;
    unsigned long long s_target = 0ull;
    unsigned long long s_total = 0ull;
    if (!k5_radix_value(row, mask_words, mask_words_per_row, rows, row_logits,
                        vocab, inv_temperature, max_scaled, min_scaled, -1.0f,
                        target, radix_bins, radix_state, tid, &s_value,
                        &s_above, &s_target, &s_total)) {
      k5_mark_internal(scratch, block_row, output_row, out_status, tid);
      return;
    }
    const unsigned long long s_weight = k5_fixed_weight(
        expf(ds41rt_v41_ordered_value(s_value) - max_scaled));
    const unsigned long long s_need = s_target > s_above ? s_target - s_above : 0ull;
    const unsigned long long s_nth = s_weight == 0ull ? 0ull :
        max(1ull, (s_need + s_weight - 1ull) / s_weight);
    if (s_nth == 0ull || !k5_radix_tie_id(
            row, mask_words, mask_words_per_row, rows, row_logits, vocab,
            inv_temperature, min_scaled, s_value, s_nth, radix_bins,
            radix_state, tid, &s_id)) {
      k5_mark_internal(scratch, block_row, output_row, out_status, tid);
      return;
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
    (void)nucleus_mass;
    (void)fixed_total;
    (void)s_total;
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
  v41_sample_prepare_kernel<<<static_cast<unsigned int>(rows), kSamplerBlock, 0, stream>>>(
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
  v41_sample_categorical_sequential_kernel<<<static_cast<unsigned int>(rows), kSamplerBlock, 0,
                                             stream>>>(
      logits, rows, vocab, logits_stride, params, mask_words, mask_words_per_row,
      out_indices, out_total, scratch);
#else
  v41_sample_categorical_kernel<<<static_cast<unsigned int>(rows), kSamplerBlock, 0, stream>>>(
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
  v41_sample_topk_pivot_kernel<<<static_cast<unsigned int>(rows), kSamplerBlock, 0, stream>>>(
      logits, rows, vocab, logits_stride, params, mask_words, mask_words_per_row,
      out_pivot_passes, scratch);
  const ds41rt_status_t pivot = status_from_cuda(cudaGetLastError());
  if (pivot != DS41RT_STATUS_OK) {
    return pivot;
  }
  v41_sample_topk_membership_kernel<<<static_cast<unsigned int>(rows), kSamplerBlock, 0,
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
  v41_sample_nucleus_kernel<<<static_cast<unsigned int>(rows), kSamplerBlock, 0, stream>>>(
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
