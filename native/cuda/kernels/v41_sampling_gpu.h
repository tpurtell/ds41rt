/* GPU target-sampler device ABI (ds41rt v4.1, `serve-native`).
 *
 * This header is the single definition of the sampler's device-side contract:
 * the 64-byte per-row parameter block of `docs/gpu-sampling-design.md` §5.1,
 * the packed constraint-mask layout of §5.2/§5.3, the per-row scratch layout,
 * the per-row status codes of §5.4, and the C ABI entry-point declarations.
 *
 * It is included by `native/cuda/kernels/v41_sampling_gpu.cu` (the kernel) and
 * by `native/include/ds41rt_native.h` (the public ABI), so the two can never
 * disagree; it is mirrored by `rust/crates/ds41rt-ffi/src/lib.rs`, which
 * test-pins `sizeof`/offsets/alignment of `ds41rt_v41_sampler_row_t`.
 *
 * Algorithm provenance: the mask predicate, the scale-then-compare order, the
 * `min_p` threshold and the lowest-id tie rule are ports of the frozen CPU
 * filter chain (`rust/crates/ds41rt-core/src/target_sampling.rs`), which is the
 * correctness oracle. The argmax/block-reduce structure is ported from the
 * legacy device argmax (`native/cuda/kernels/sampling.cu`,
 * `logits_argmax_f32_kernel`), which itself cites TRT-LLM; the FlashInfer
 * headers are a reference for the later chunks (K2/K3/K5) only, never a build
 * dependency.
 *
 * Chunk 1 implements K1 only: mask application first, finiteness discipline,
 * temperature scaling, scaled maximum, the `min_p` survivor count and the
 * greedy / constrained-greedy device argmax. `k`-selection and top-p land in
 * later chunks; the scratch and arena regions they need are already allocated.
 *
 * Chunk 2 adds K2 (`v41_sample_categorical_kernel`): the fast-path categorical
 * draw for rows with `top_k` disabled and `top_p >= 1.0` (the temperature-only
 * and `min_p`-only profiles), the device port of the `(seed, position)` draw,
 * the **consistent sequential segment prefix** (chunk-3b `P0-1`: one
 * non-decreasing fold `C(i+1) = fl(C(i) + local(i))` over the per-segment
 * masses, replacing the fixed-tree inclusive scan whose cross-segment
 * association could hand a segment a start prefix already past `target`), the
 * ascending-token-order crossing scan with the `cumulative_before < target`
 * minimality rule, and the CPU's no-crossing `last`-survivor fallback. Both
 * accumulation orders are compiled: the shipped default is the sequential
 * segment prefix above, and `DS41RT_V41_K2_SEQUENTIAL_COMBINE` selects the
 * strictly sequential token-order combine evaluated by the chunk-2 measurement.
 */
#ifndef DS41RT_V41_SAMPLING_GPU_H
#define DS41RT_V41_SAMPLING_GPU_H

#include "ds41rt_native.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Per-row flags (§5.1, plus the resolved D9/§5.3 unconstrained bit) ---- */
/* bit0: greedy row. The host resolves `temperature < 1e-5 || top_k == 1`; the
 *       kernel re-derives it and ORs the two, so a host bug cannot silently
 *       turn a stochastic row greedy. */
#define DS41RT_V41_SAMPLER_FLAG_GREEDY 0x1u
/* bit1: diagnose. Optional `out_total`/`out_nucleus_count` are written. */
#define DS41RT_V41_SAMPLER_FLAG_DIAGNOSE 0x2u
/* bit2: the mask row has no meaningful bits -- an unconstrained row
 *       (`fill_bitmask` reported `needs_mask == false`, or no mask at all).
 *       The kernel then treats every token `t < vocab` as allowed and never
 *       reads the mask arena.
 *
 *       APPROVED design deviation (recorded in the chunk-1 report so the design
 *       doc can be updated). The design's §5.1 assigns bit2 to "mask
 *       remainder-masked", but §5.2 wants `mask_row = 0xFFFFFFFF` for an
 *       unconstrained row and §5.3 says there is no all-ones fill in the native
 *       path, so nothing told the kernel to skip mask reads. `needs_mask ==
 *       false` is the production signal (`constraints.rs`). The remainder rule
 *       stays unconditional and host-enforced: the host always zeroes bits
 *       `>= vocab` of the final word before upload
 *       (`ds41rt_v41_sampler_clear_remainder`), so a whole-word reader is safe
 *       regardless of this bit. */
#define DS41RT_V41_SAMPLER_FLAG_NO_MASK 0x4u
/* bit3: CPU-oracle cross-check (diagnostic only; unused in production). */
#define DS41RT_V41_SAMPLER_FLAG_ORACLE_CROSSCHECK 0x8u
/* bit4: strict whole-row finiteness for a row that would otherwise take the
 *       permissive stochastic branch. Greedy rows are strict unconditionally
 *       (the kernel derives strictness from `greedy`), so the host sets this bit
 *       on greedy and constrained rows only and leaves a stochastic row
 *       permissive. It mirrors `scores.rs::argmax`'s
 *       `ensure!(value.is_finite())`, which runs before the mask test for
 *       greedy/constrained rows, so such a row rejects a non-finite logit even
 *       when that token is masked out. */
#define DS41RT_V41_SAMPLER_FLAG_STRICT_FINITE 0x10u

/* ---- Per-row status codes (§5.4). Host-side rank = the integer value. ---- */
#define DS41RT_V41_SAMPLER_STATUS_OK 0u
#define DS41RT_V41_SAMPLER_STATUS_EMPTY_CANDIDATES 1u
#define DS41RT_V41_SAMPLER_STATUS_NONFINITE_LOGIT 2u
#define DS41RT_V41_SAMPLER_STATUS_INVALID_TEMPERATURE 3u
#define DS41RT_V41_SAMPLER_STATUS_MASK_WIDTH 4u
#define DS41RT_V41_SAMPLER_STATUS_INTERNAL 5u

/* `out_status_detail` for status 2 carries the offending token id; the design
 * text writes a plain `0` when there is none, but token 0 is a real token, so a
 * caller could not tell "no detail" from "token 0". The kernel writes the
 * UINT32_MAX sentinel instead and the host normalizes it to 0 before it is
 * observable. APPROVED design deviation (recorded in the chunk-1 report so the
 * design doc can be updated). */
#define DS41RT_V41_SAMPLER_NO_DETAIL 0xFFFFFFFFu

/* `mask_row` sentinel: an unconstrained row. The kernel treats this sentinel as
 * unconstrained even when `FLAG_NO_MASK` is absent, so a raw-C caller that only
 * follows `docs/gpu-sampling-design.md` §5.2 (which names the sentinel but not
 * the flag) still cannot index the arena out of bounds. The host-side validator
 * is stricter and requires the flag to agree with the sentinel. */
#define DS41RT_V41_SAMPLER_NO_MASK_ROW 0xFFFFFFFFu

/* ---- 64-byte per-row parameter block (§5.1), natural alignment ---- */
typedef struct ds41rt_v41_sampler_row_s {
  uint64_t seed;        /* +0  served request seed, two's complement */
  uint64_t position;    /* +8  absolute emitted-token index */
  float    temperature; /* +16 validated range 0..=2 */
  float    top_p;       /* +20 validated (0,1]; >= 1.0 means disabled */
  float    min_p;       /* +24 validated [0,1]; 0.0 means disabled */
  uint32_t top_k;       /* +28 0 = Option::None (disabled); 1 = greedy;
                               k > vocab = no-op */
  uint32_t mask_row;    /* +32 index into the mask arena; 0xFFFFFFFF =
                               unconstrained */
  uint32_t flags;       /* +36 DS41RT_V41_SAMPLER_FLAG_* */
  uint32_t output_row;  /* +40 row index into logits/out_* */
  float    ln_min_p;    /* +44 HOST-precomputed ln(min_p); -inf when min_p == 0.
                               Hard requirement: the device never calls logf. */
  uint32_t reserved0;   /* +48 must be 0 */
  uint32_t reserved1;   /* +52 must be 0 */
  uint64_t reserved2;   /* +56 must be 0 */
} ds41rt_v41_sampler_row_t; /* exactly 64 B */

/* ---- `params` residency ---- */
/* `params` MUST point at device memory holding `rows` consecutive
 * `ds41rt_v41_sampler_row_t` blocks: the kernel dereferences
 * `params[blockIdx.x]` on device. A pageable host address is not
 * device-addressable under CUDA's documented model, even on a platform whose
 * driver happens to expose it, so the host must H2D-copy the block first. The
 * shipped `TargetSamplingWave` uploads into `param_device` and launches that
 * buffer; the FFI wrapper takes the device buffer separately from the host
 * slice it validates. */

/* ---- Per-row K1 scratch, 64-byte stride (§11.1) ---- */
typedef struct ds41rt_v41_sampler_scratch_s {
  float    max_scaled;        /* +0  max over ALLOWED tokens (stochastic) */
  float    inv_temperature;   /* +4  1.0f / temperature, exactly as the CPU */
  uint32_t allowed_count;     /* +8  allowed tokens (0 -> EMPTY_CANDIDATES) */
  uint32_t survivor_count;    /* +12 allowed tokens with scaled >= min_scaled */
  uint32_t kth_value_bits;    /* +16 K3, later chunk */
  uint32_t above_count;       /* +20 K3, later chunk */
  uint32_t nonfinite_token;   /* +24 lowest offending id, or NO_DETAIL */
  uint32_t status;            /* +28 one of DS41RT_V41_SAMPLER_STATUS_* */
  uint32_t status_detail;     /* +32 token id / actual width / NO_DETAIL */
  uint32_t reserved0;         /* +36 must be 0 */
  uint64_t reserved1;         /* +40 must be 0 */
  uint32_t reserved2;         /* +48 must be 0 */
  uint32_t reserved3;         /* +52 must be 0 */
  uint64_t reserved4;         /* +56 must be 0 */
} ds41rt_v41_sampler_scratch_t; /* exactly 64 B */

/* Byte layout of one row's scratch region. Exposed as macros so the Rust
 * planner and the C ABI test can pin the same numbers. */
#define DS41RT_V41_SAMPLER_SCRATCH_BYTES 64u
#define DS41RT_V41_SAMPLER_PARAM_BYTES 64u

/* Number of packed u32 mask words for a vocabulary: `ceil(vocab / 32)` (§5.2).
 * The host must additionally zero the bits `>= vocab` of the final word before
 * upload (§5.3 rule 2); every kernel loop is bounded by `vocab` (rule 1), so a
 * token id `>= vocab` can never be produced (rule 3). */
static inline size_t ds41rt_v41_sampler_mask_words(size_t vocab) {
  return (vocab + 31u) / 32u;
}

/* Apply §5.3 rule 2 to a host mask row in place: clear every bit `>= vocab` of
 * the final word. `words` must be `ds41rt_v41_sampler_mask_words(vocab)`. */
static inline void ds41rt_v41_sampler_clear_remainder(uint32_t* words, size_t vocab) {
  const size_t remainder = vocab % 32u;
  if (remainder == 0u || vocab == 0u) {
    return;
  }
  words[(vocab - 1u) / 32u] &= (uint32_t{1} << remainder) - 1u;
}

/* ---- Device draw: SplitMix64, ported bit-identically (design §6.1) ----
 *
 * The target draw is `TargetSamplingParams::random_uniform`
 * (`rust/crates/ds41rt-core/src/target_sampling.rs:187-199`) reproduced exactly,
 * so the `(seed, position)` -> uniform mapping is unchanged from the CPU. The
 * mantissa is truncated to 24 bits and scaled by the exact `2^-24`, so the
 * product is a single exact f32 operation and the device value is bit-identical
 * to the host for every `(seed, position)`.
 *
 * `DS41RT_V41_SAMPLER_MAX_UNIFORM_BITS` is the `0x3F7FFFFF` (= 0.99999994) clamp
 * of `target_sampling.rs:51-52`, applied after the draw and before the CDF
 * comparison (`:459`). It is a bit pattern, never a decimal literal.
 *
 * These live in the header (under `__CUDACC__`, so host C++ that includes
 * `ds41rt_native.h` never sees them) because the device tests must call the
 * *shipped* function from a probe kernel, not a copy of it.
 */
#define DS41RT_V41_SAMPLER_MAX_UNIFORM_BITS 0x3F7FFFFFu
#define DS41RT_V41_SAMPLER_RNG_DOMAIN 0x7f4a7c159e3779b9ull
#define DS41RT_V41_SAMPLER_RNG_MUL 0x9e3779b97f4a7c15ull

#if defined(__CUDACC__)
__device__ __forceinline__ float ds41rt_v41_target_uniform(uint64_t seed, uint64_t position) {
  uint64_t mixed = seed + DS41RT_V41_SAMPLER_RNG_DOMAIN +
                   position * DS41RT_V41_SAMPLER_RNG_MUL + DS41RT_V41_SAMPLER_RNG_MUL;
  mixed = (mixed ^ (mixed >> 30)) * 0xbf58476d1ce4e5b9ull;
  mixed = (mixed ^ (mixed >> 27)) * 0x94d049bb133111ebull;
  mixed ^= mixed >> 31;
  const uint32_t mantissa = static_cast<uint32_t>(mixed >> 40);
  return static_cast<float>(mantissa) * (1.0f / 16777216.0f);
}

/* The clamp of design §4.8 / §6.1, in the exact CPU order
 * (`MAX_UNIFORM.min(uniform.max(0.0))`). */
__device__ __forceinline__ float ds41rt_v41_target_clamp_uniform(float uniform) {
  return fminf(fmaxf(uniform, 0.0f), __uint_as_float(DS41RT_V41_SAMPLER_MAX_UNIFORM_BITS));
}

/* ---- Chunk 3a: the order-key primitive (design §4.3) ----
 *
 * `order_key(x)` is the standard IEEE total-order map onto u32 with **larger =
 * better**: `bits ^ 0x80000000` for a non-negative value, `~bits` for a negative
 * one, after canonicalizing `-0.0` to `+0.0` (the CPU comparator's
 * `partial_cmp` treats the two zeroes as equal, and `Ranked::better_than` then
 * breaks the tie by id). It is the bitwise complement of
 * `target_sampling.rs:335-344`'s `descending_radix_key`, i.e. the *ascending*
 * form the design's `C_gt(v) < k` acceptance reads from (Appendix A.1).
 *
 * The full total order K4 materializes is the u64 key
 * `(order_key(scaled) << 32) | ~id`: larger is better and an exact scaled tie
 * goes to the **lowest token id**, matching `Ranked::better_than`
 * (`target_sampling.rs:283-291`) and `sampling.cu:28-33`.
 *
 * These live in the header (under `__CUDACC__`) because the device tests and
 * any later chunk (K5) must call the *shipped* primitive, never a copy. */
__device__ __forceinline__ uint32_t ds41rt_v41_order_key(float scaled) {
  /* Canonicalize -0.0 so both zeroes map to one key, exactly as
   * `descending_radix_key` does (`target_sampling.rs:336`). */
  const float value = (scaled == 0.0f) ? 0.0f : scaled;
  const uint32_t bits = __float_as_uint(value);
  return ((bits & 0x80000000u) != 0u) ? ~bits : (bits ^ 0x80000000u);
}

/* Inverse of `ds41rt_v41_order_key` on the canonical f32 bit patterns: the float
 * whose order key is `key`. Used only to publish `scratch.kth_value_bits` in the
 * design's §4.3 spelling (the *value bits* of the k-th value) and to feed K4's
 * tie predicate; `order_key(ordered_value(key)) == key` for every key produced
 * by `order_key`, and the k-th value is always a survivor's scaled value. */
__device__ __forceinline__ float ds41rt_v41_ordered_value(uint32_t key) {
  const uint32_t bits =
      ((key & 0x80000000u) != 0u) ? (key ^ 0x80000000u) : ~key;
  return __uint_as_float(bits);
}
#endif

/* Lower `16 * DS41RT_V41_SAMPLER_ROW_PARAM_OFFSET_<field>` is the field's byte
 * offset inside `ds41rt_v41_sampler_row_t`; the Rust ABI test pins these. */
#define DS41RT_V41_SAMPLER_ROW_PARAM_OFFSET_SEED 0u
#define DS41RT_V41_SAMPLER_ROW_PARAM_OFFSET_POSITION 8u
#define DS41RT_V41_SAMPLER_ROW_PARAM_OFFSET_TEMPERATURE 16u
#define DS41RT_V41_SAMPLER_ROW_PARAM_OFFSET_TOP_P 20u
#define DS41RT_V41_SAMPLER_ROW_PARAM_OFFSET_MIN_P 24u
#define DS41RT_V41_SAMPLER_ROW_PARAM_OFFSET_TOP_K 28u
#define DS41RT_V41_SAMPLER_ROW_PARAM_OFFSET_MASK_ROW 32u
#define DS41RT_V41_SAMPLER_ROW_PARAM_OFFSET_FLAGS 36u
#define DS41RT_V41_SAMPLER_ROW_PARAM_OFFSET_OUTPUT_ROW 40u
#define DS41RT_V41_SAMPLER_ROW_PARAM_OFFSET_LN_MIN_P 44u
#define DS41RT_V41_SAMPLER_ROW_PARAM_OFFSET_RESERVED0 48u
#define DS41RT_V41_SAMPLER_ROW_PARAM_OFFSET_RESERVED1 52u
#define DS41RT_V41_SAMPLER_ROW_PARAM_OFFSET_RESERVED2 56u

/* ---- C ABI entry points (§5.4) ----
 *
 * `logits` is `rows` contiguous f32 rows of `vocab` values with row stride
 * `logits_stride` floats (>= vocab). `params` is `rows` 64-byte blocks.
 * `mask_words` is nullable; when non-null it is `rows * mask_words_per_row`
 * packed u32 words and `mask_words_per_row` must equal
 * `ds41rt_v41_sampler_mask_words(vocab)`.
 *
 * `out_indices`, `out_status`, `out_status_detail`, `out_scores` and
 * `scratch` are required; `out_total` and `out_nucleus_count` may be null.
 * Each is `rows` elements (scratch is `rows * 64` bytes).
 *
 * `out_status_detail` reports the offending token id for
 * `NONFINITE_LOGIT`, the provided word count for `MASK_WIDTH`, and the
 * `DS41RT_V41_SAMPLER_NO_DETAIL` sentinel otherwise.
 *
 * The kernel is graph-capture legal: one CTA per row, intra-CTA
 * `__syncthreads()` only, no grid sync, no host callback, no per-call
 * allocation. It must be compiled without `-use_fast_math` and without FTZ.
 */
ds41rt_status_t ds41rt_cuda_v41_target_sample_async(
    const float* logits, size_t rows, size_t vocab, size_t logits_stride,
    const ds41rt_v41_sampler_row_t* params, const uint32_t* mask_words,
    size_t mask_words_per_row, uint32_t* out_indices, uint32_t* out_status,
    uint32_t* out_status_detail, float* out_scores, float* out_total,
    uint32_t* out_nucleus_count, ds41rt_v41_sampler_scratch_t* scratch,
    void* cuda_stream);
ds41rt_status_t ds41rt_cuda_v41_target_sample(
    const float* logits, size_t rows, size_t vocab, size_t logits_stride,
    const ds41rt_v41_sampler_row_t* params, const uint32_t* mask_words,
    size_t mask_words_per_row, uint32_t* out_indices, uint32_t* out_status,
    uint32_t* out_status_detail, float* out_scores, float* out_total,
    uint32_t* out_nucleus_count, ds41rt_v41_sampler_scratch_t* scratch);

/* ====================================================================== */
/* Chunk 3a: K3 pivot selection + K4 exact-k membership (design §4.3-§4.4) */
/* ====================================================================== */

/* ---- The rank-order contract K5 (chunk 3b) consumes ----
 *
 * K4 materializes the retained set into `rank_order_ids` in the CPU's **exact
 * rank order**: descending `order_key(scaled)`, with an exact `scaled` tie
 * broken by ascending token id (`Ranked::better_than`,
 * `target_sampling.rs:283-291`; the same total order the CPU's comparison sort
 * and stable LSD radix sort both produce, `:323-344`).
 *
 * Arena layout: `rank_order_ids` is `rows * rank_order_capacity` u32 with the
 * **block row** `b` (the `params[b]` / `logits + b * logits_stride` row) at
 * `rank_order_ids + b * rank_order_capacity`; `rank_order_scratch` is the same
 * layout in u64. When K3 ran on row `b`, exactly `params[b].top_k` ids are
 * written, `rank 0` being the best. (The small `out_retained_count` /
 * `out_pivot_passes` outputs are indexed by `output_row` like the other ABI
 * outputs; the daemon makes the two equal.) The buffer is O(k) per row, not
 * O(vocab): the design's §11.1 top-k bitmap arena (capacity × ceil(vocab/32)
 * u32) is the production region once chunk 4 plumbs it, and
 * `rank_order_capacity` is its per-row id stride.
 *
 * `rank_order_scratch` is the matching `rows * rank_order_capacity` u64 staging
 * (token-order entries packed as `(order_key << 32) | id`); it is scratch, never
 * observable. `rank_order_capacity == 0` selects a **selection-only** call: K3
 * publishes `scratch.kth_value_bits` / `scratch.above_count`, K4 computes the
 * membership state, but nothing is materialized and both arena pointers may be
 * null. A non-zero capacity must be `>= max_r params[r].top_k`; the FFI wrapper
 * enforces that on the host.
 *
 * Row eligibility is exactly the CPU's (`target_sampling.rs:477-501`): K3/K4
 * run only when K1 reported `OK`, the row is **not** greedy
 * (`temperature < 1e-5 || top_k == 1`), `top_k != 0` and
 * `top_k < survivor_count`. Every other row is a no-op: `top_k == 0` is
 * `Option::None` (disabled) and never enters K3; `top_k >= survivor_count`
 * truncates nothing and falls through to the CPU's full ordering branches, so
 * K5 must treat the retained set as *all* survivors there. For a no-op row both
 * outputs are 0 and the arena region is untouched.
 *
 * `out_retained_count[r]` is `params[r].top_k` for an eligible row and 0
 * otherwise (so a caller can tell a real selection from a no-op).
 * `out_pivot_passes[r]` is the number of K3 bisection passes that row used
 * (`<= DS41RT_V41_TOPK_MAX_PIVOT_STEPS`, 0 for a no-op row). A bisection that
 * cannot converge inside the cap writes `DS41RT_V41_SAMPLER_STATUS_INTERNAL`
 * into `scratch[r].status` and materializes nothing.
 *
 * `scratch[r].kth_value_bits` receives the f32 bits of the k-th largest scaled
 * value and `scratch[r].above_count` receives `C_gt(kth) = #{survivors :
 * order_key(scaled) > order_key(kth)}`. Membership is then exactly
 * `{order_key > kth} ∪ {the lowest-id (k - above_count) survivors whose
 * order_key == kth}` — ds41rt's documented deviation from vLLM/FlashInfer,
 * which keep every k-th-value tie (`target_sampling.rs:30-33`; risk R2).
 *
 * These entry points are additive: `ds41rt_cuda_v41_target_sample[_async]` is
 * byte-for-byte unchanged, so the chunk-1/chunk-2 production path and its
 * daemon caller are untouched. K5 lands in chunk 3b on top of this contract.
 */
#define DS41RT_V41_TOPK_MAX_PIVOT_STEPS 32u

ds41rt_status_t ds41rt_cuda_v41_topk_select_async(
    const float* logits, size_t rows, size_t vocab, size_t logits_stride,
    const ds41rt_v41_sampler_row_t* params, const uint32_t* mask_words,
    size_t mask_words_per_row, uint32_t* rank_order_ids,
    uint64_t* rank_order_scratch, size_t rank_order_capacity,
    uint32_t* out_retained_count, uint32_t* out_pivot_passes,
    ds41rt_v41_sampler_scratch_t* scratch, void* cuda_stream);
ds41rt_status_t ds41rt_cuda_v41_topk_select(
    const float* logits, size_t rows, size_t vocab, size_t logits_stride,
    const ds41rt_v41_sampler_row_t* params, const uint32_t* mask_words,
    size_t mask_words_per_row, uint32_t* rank_order_ids,
    uint64_t* rank_order_scratch, size_t rank_order_capacity,
    uint32_t* out_retained_count, uint32_t* out_pivot_passes,
    ds41rt_v41_sampler_scratch_t* scratch);

/* ====================================================================== */
/* Chunk 3b: K5 inclusive-prefix top-p nucleus + rank-order draw          */
/* (design §4.5, §4.6, Appendix A.1)                                      */
/* ====================================================================== */

/* K5 is a THIRD kernel on the same stream after K1 and K3/K4. It consumes
 * K1's `scratch` and the chunk-3a rank-ordered retained list and writes the
 * final sampled token for every **ordered** row.
 *
 * Ordered row (the "K5 row class" of design §4.6): K1 `status == OK`, not
 * greedy (`temperature < 1e-5 || top_k == 1`), and `top_k != 0 || top_p < 1.0`.
 * Three sub-cases, all of which the CPU routes through the ordered
 * `sample_from_ranked` tail (`target_sampling.rs:503-522`, `:527-571`):
 *   1. `top_k != 0 && top_k < survivor_count` — K3/K4 materialized exactly
 *      `top_k` rank-ordered ids, and the set `S` is that list;
 *   2. `top_k >= survivor_count` — K3/K4 were a no-op, `S` is every survivor;
 *   3. `top_k == 0` (disabled) with `top_p < 1.0` — `S` is every survivor.
 * The disjoint K2 fast path (`top_k == 0 && top_p >= 1.0`) is untouched: those
 * rows are not touched by K5 and keep K2's id.
 *
 * `rank_retained_count[r]` is K3/K4's `out_retained_count` for block row `r`
 * (0 for a no-op row). When it is non-zero K5 uses the rank-ordered list at
 * `rank_order_ids + r * rank_order_capacity` -- entries are indexed by BLOCK
 * row, exactly as K3/K4 write them; every other output is indexed by
 * `output_row` like the rest of the ABI. `rank_order_ids` may be null when no
 * row materialized a list. K4 publishes `out_retained_count` by `output_row`
 * while this kernel consumes it (and the arena) by block row, so the nucleus
 * entry point requires the identity mapping `params[r].output_row == r`; the FFI
 * validator rejects a non-identity batch and the kernel reports `INTERNAL` for
 * ANY K5-class raw-C caller whose `output_row != block_row` -- unconditionally,
 * BEFORE it interprets `rank_retained_count`, so a foreign count of 0 cannot
 * flip `retained_mode` false and slip past the check. The chunk-4 integrator
 * must keep the mapping identity (the daemon already asserts it in
 * `v41_target_head.rs`).
 *
 * `out_total[r]` and `out_nucleus_count[r]` are written only when
 * `params[r].flags` has `DS41RT_V41_SAMPLER_FLAG_DIAGNOSE`; both may be null.
 * `out_total` is the CPU's `RankedSample.total` (the floored, un-normalized
 * weight total over `S`), `out_nucleus_count` is the CPU's `nucleus_count`.
 *
 * `out_status` is the SAME per-row status channel K1 uses and may be null. K5
 * writes `INTERNAL` there, for a K5-class row that cannot produce a defined
 * token: a retained list wider than `kBlock` (256) or not materialized
 * (`rank_order_ids == nullptr`), ANY non-identity `output_row` (checked
 * unconditionally before the retained count is read), a non-finite `top_p`, or a
 * `survivor_count == 0` row. `scratch[r].status` carries the same value; the
 * entry point's own return value stays OK because the failure is per row. A
 * caller that passes `out_status` therefore cannot mistake a silent no-token for
 * success. **K1 additionally reports `INTERNAL` for a non-finite `top_p` and for
 * a non-greedy zero-survivor row**, so even the K1+K2-only
 * `ds41rt_cuda_v41_target_sample[_async]` entry point -- which never launches
 * K3/K4/K5 -- cannot return OK with an unwritten `out_indices` for those two
 * shapes (the FFI validator rejects them on the host; this is the raw-C path).
 *
 * Both boundaries find the CPU's boundaries. The mass arithmetic defaults to the
 * CPU's literal per-token normalization (`DS41RT_V41_K5_NORMALIZED_MASS=1`): one
 * f32 division `w_t / total` per token, accumulated in the kernel's own order,
 * with the `top_p` / `uniform * nucleus_mass` thresholds. The `=0` build uses
 * the algebraically equivalent `Sum w >= threshold * total` form instead. The
 * two differ in f32 rounding; the chunk-3b report measures 11.42 % -> 0.00 %
 * retained-domain token mismatch in favor of the normalized default.
 *
 * Retained domain (case 1): K4's list is already in exact CPU rank order. The
 * inclusive prefix `W(rank)` is built ONCE per row into a fixed array, so a rank
 * has one f32 association and the monotone-predicate binary searches are
 * probe-invariant; the crossing and the draw read the same bits. `k <= kBlock`
 * is required because the per-rank weights live in one shared table; a wider
 * retained list is reported as `INTERNAL` rather than sampled from a partial
 * set.
 *
 * Survivor domain (cases 2 and 3): K3/K4 are no-ops, so the boundary is
 * `K_p` = largest total-order key `(order_key(scaled) << 32) | ~id` with
 * `M(K_p) >= threshold` (design §4.5 "Boundary primitive"). It is found in two
 * binary searches -- the ordered value first, then the token id inside the
 * boundary tie group -- so a boundary costs
 * `<= 2 + ceil(log2(value range)) + ceil(log2(id range))` passes and the draw
 * reuses the top-p search's `(value, id)` as lower bounds. The mass predicate
 * additionally requires a strictly positive mass, so a zero-uniform draw
 * (`target == 0`) still selects the best actual survivor instead of an empty
 * prefix. The no-crossing fallback reports the full survivor set.
 *
 * `DS41RT_V41_NUCLEUS_MAX_PASSES` is the documented cap for the *retained*
 * domain's `ceil(log2(k))` search; the wider survivor-domain key search is
 * bounded by `2 + 32 + 32` passes per boundary and is the cost the design's
 * chunk-6/7 pass-reduction work targets. `rank_order_capacity`/
 * `rank_retained_count` are never written.
 *
 * The design's "no key satisfies `M(K) >= top_p`" fallback is DEAD CODE in the
 * un-normalized build: every `w_t > 0`, the best survivor attains `max_scaled`
 * so `w_best = expf(0) = 1` and `total >= 1`, and `fl(top_p * total) <= total`
 * for `top_p <= 1`, so the whole-set key always satisfies the predicate. In the
 * normalized build the CPU's `Sigma (w/total)` can round below `top_p`, which is
 * exactly the CPU's *consumed every weight* case: the nucleus is the full set
 * and the draw still runs over it.
 *
 * K5 writes `scratch[r].status`/`out_status[r] = INTERNAL` for a raw-C call whose
 * `rank_retained_count[r]` exceeds `rank_order_capacity` or `kBlock`, whose
 * retained arena is null, whose `output_row != r` (any K5-class row, checked
 * before the count is read), whose `top_p` is non-finite, or whose row has no
 * survivor. Every other status is K1's, unchanged. K1 itself writes INTERNAL
 * for a non-finite `top_p` or a non-greedy zero-survivor row, so the K1+K2-only
 * entry point is loud for those shapes too.
 *
 * Like K3/K4 these entry points are additive: K1, K2 and K3/K4 are
 * byte-for-byte unchanged.
 */
#define DS41RT_V41_NUCLEUS_MAX_PASSES 32u

ds41rt_status_t ds41rt_cuda_v41_nucleus_async(
    const float* logits, size_t rows, size_t vocab, size_t logits_stride,
    const ds41rt_v41_sampler_row_t* params, const uint32_t* mask_words,
    size_t mask_words_per_row, const uint32_t* rank_order_ids,
    size_t rank_order_capacity, const uint32_t* rank_retained_count,
    uint32_t* out_indices, uint32_t* out_status, float* out_total,
    uint32_t* out_nucleus_count, ds41rt_v41_sampler_scratch_t* scratch,
    void* cuda_stream);
ds41rt_status_t ds41rt_cuda_v41_nucleus(
    const float* logits, size_t rows, size_t vocab, size_t logits_stride,
    const ds41rt_v41_sampler_row_t* params, const uint32_t* mask_words,
    size_t mask_words_per_row, const uint32_t* rank_order_ids,
    size_t rank_order_capacity, const uint32_t* rank_retained_count,
    uint32_t* out_indices, uint32_t* out_status, float* out_total,
    uint32_t* out_nucleus_count, ds41rt_v41_sampler_scratch_t* scratch);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* DS41RT_V41_SAMPLING_GPU_H */
