#include "common.h"

#include <cub/cub.cuh>

namespace {

constexpr size_t kParallelSampleTopK = 8;
using MaskedTopKBlockSort =
    cub::BlockRadixSort<uint64_t, kBlock, static_cast<int>(kParallelSampleTopK)>;
using MaskedArgmaxBlockReduce = cub::BlockReduce<uint64_t, kBlock>;

struct MaxUint64 {
  __device__ uint64_t operator()(uint64_t lhs, uint64_t rhs) const {
    return lhs > rhs ? lhs : rhs;
  }
};

__device__ uint32_t float_to_ordered_bits(float value) {
  value = value == 0.0f ? 0.0f : value;
  const uint32_t bits = __float_as_uint(value);
  const uint32_t mask = static_cast<uint32_t>(-static_cast<int32_t>(bits >> 31)) | 0x80000000U;
  return bits ^ mask;
}

__device__ float ordered_bits_to_float(uint32_t ordered) {
  const uint32_t mask = (ordered >> 31) == 0 ? 0xffffffffU : 0x80000000U;
  return __uint_as_float(ordered ^ mask);
}

__device__ uint64_t topk_sort_key(float logit, uint32_t token_id) {
  // Descending integer order gives descending logits and ascending IDs on exact ties.
  return (static_cast<uint64_t>(float_to_ordered_bits(logit)) << 32) |
         static_cast<uint64_t>(~token_id);
}

__device__ void insert_topk_candidate(float logit, uint32_t token_id, float* best_logits,
                                      uint32_t* best_indices, size_t top_k) {
  for (size_t rank = 0; rank < top_k; ++rank) {
    if (logit > best_logits[rank] ||
        (logit == best_logits[rank] && token_id < best_indices[rank])) {
      for (size_t shift = top_k - 1; shift > rank; --shift) {
        best_logits[shift] = best_logits[shift - 1];
        best_indices[shift] = best_indices[shift - 1];
      }
      best_logits[rank] = logit;
      best_indices[rank] = token_id;
      break;
    }
  }
}

__device__ void write_topk_topp_sample(const float* best_logits, const uint32_t* best_indices,
                                       const float* random_uniforms, uint32_t* out_indices,
                                       float* out_scores, size_t row, size_t top_k,
                                       float temperature, float top_p) {
  float scaled[kMaxSampleTopK];
  float max_scaled = -CUDART_INF_F;
  for (size_t rank = 0; rank < top_k; ++rank) {
    scaled[rank] = best_logits[rank] / temperature;
    max_scaled = fmaxf(max_scaled, scaled[rank]);
  }

  float probs[kMaxSampleTopK];
  float total = 0.0f;
  for (size_t rank = 0; rank < top_k; ++rank) {
    probs[rank] = expf(scaled[rank] - max_scaled);
    total += probs[rank];
  }
  total = fmaxf(total, 1.0e-20f);
  for (size_t rank = 0; rank < top_k; ++rank) {
    probs[rank] /= total;
  }

  const float top_p_clamped = fminf(fmaxf(top_p, 1.0e-6f), 1.0f);
  float nucleus_mass = 0.0f;
  size_t nucleus_count = 0;
  for (size_t rank = 0; rank < top_k; ++rank) {
    nucleus_mass += probs[rank];
    nucleus_count = rank + 1;
    if (nucleus_mass >= top_p_clamped) {
      break;
    }
  }
  nucleus_mass = fmaxf(nucleus_mass, 1.0e-20f);

  const float target = fminf(fmaxf(random_uniforms[row], 0.0f), 0.99999994f) * nucleus_mass;
  float cumulative = 0.0f;
  size_t selected_rank = nucleus_count - 1;
  for (size_t rank = 0; rank < nucleus_count; ++rank) {
    cumulative += probs[rank];
    if (target <= cumulative) {
      selected_rank = rank;
      break;
    }
  }

  out_indices[row] = best_indices[selected_rank];
  out_scores[row] = probs[selected_rank] / nucleus_mass;
}

__device__ void write_topk_topp_sample_small(const float* best_logits, const uint32_t* best_indices,
                                             const float* random_uniforms, uint32_t* out_indices,
                                             float* out_scores, size_t row, size_t top_k,
                                             float temperature, float top_p) {
  float scaled[kParallelSampleTopK];
  float max_scaled = -CUDART_INF_F;
  for (size_t rank = 0; rank < top_k; ++rank) {
    scaled[rank] = best_logits[rank] / temperature;
    max_scaled = fmaxf(max_scaled, scaled[rank]);
  }

  float probs[kParallelSampleTopK];
  float total = 0.0f;
  for (size_t rank = 0; rank < top_k; ++rank) {
    probs[rank] = expf(scaled[rank] - max_scaled);
    total += probs[rank];
  }
  total = fmaxf(total, 1.0e-20f);
  for (size_t rank = 0; rank < top_k; ++rank) {
    probs[rank] /= total;
  }

  const float top_p_clamped = fminf(fmaxf(top_p, 1.0e-6f), 1.0f);
  float nucleus_mass = 0.0f;
  size_t nucleus_count = 0;
  for (size_t rank = 0; rank < top_k; ++rank) {
    nucleus_mass += probs[rank];
    nucleus_count = rank + 1;
    if (nucleus_mass >= top_p_clamped) {
      break;
    }
  }
  nucleus_mass = fmaxf(nucleus_mass, 1.0e-20f);

  const float target = fminf(fmaxf(random_uniforms[row], 0.0f), 0.99999994f) * nucleus_mass;
  float cumulative = 0.0f;
  size_t selected_rank = nucleus_count - 1;
  for (size_t rank = 0; rank < nucleus_count; ++rank) {
    cumulative += probs[rank];
    if (target <= cumulative) {
      selected_rank = rank;
      break;
    }
  }

  out_indices[row] = best_indices[selected_rank];
  out_scores[row] = probs[selected_rank] / nucleus_mass;
}

__device__ void compute_serial_lm_head_topk(const uint16_t* row_hidden, const uint16_t* lm_head,
                                            float* best_logits, uint32_t* best_indices,
                                            size_t hidden_dim, size_t vocab, size_t top_k) {
  for (size_t rank = 0; rank < top_k; ++rank) {
    best_logits[rank] = -CUDART_INF_F;
    best_indices[rank] = 0;
  }
  for (size_t token = 0; token < vocab; ++token) {
    const uint16_t* weight_row = lm_head + token * hidden_dim;
    float logit = 0.0f;
    for (size_t col = 0; col < hidden_dim; ++col) {
      logit += bf16_to_f32(row_hidden[col]) * bf16_to_f32(weight_row[col]);
    }
    insert_topk_candidate(logit, static_cast<uint32_t>(token), best_logits, best_indices, top_k);
  }
}

__global__ void lm_head_argmax_bf16_kernel(const uint16_t* hidden, const uint16_t* lm_head,
                                           uint32_t* out_indices, float* out_scores,
                                           size_t rows, size_t hidden_dim, size_t vocab) {
  __shared__ float shared_scores[kBlock];
  __shared__ uint32_t shared_indices[kBlock];

  const size_t row = blockIdx.x;
  const size_t tid = threadIdx.x;
  if (row >= rows) {
    return;
  }

  const uint16_t* row_hidden = hidden + row * hidden_dim;
  float best_score = -CUDART_INF_F;
  uint32_t best_index = 0;
  for (size_t token = tid; token < vocab; token += blockDim.x) {
    const uint16_t* weight_row = lm_head + token * hidden_dim;
    float score = 0.0f;
    for (size_t col = 0; col < hidden_dim; ++col) {
      score += bf16_to_f32(row_hidden[col]) * bf16_to_f32(weight_row[col]);
    }
    const uint32_t token_id = static_cast<uint32_t>(token);
    if (score > best_score || (score == best_score && token_id < best_index)) {
      best_score = score;
      best_index = token_id;
    }
  }

  shared_scores[tid] = best_score;
  shared_indices[tid] = best_index;
  __syncthreads();
  for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      const float other_score = shared_scores[tid + stride];
      const uint32_t other_index = shared_indices[tid + stride];
      if (other_score > shared_scores[tid] ||
          (other_score == shared_scores[tid] && other_index < shared_indices[tid])) {
        shared_scores[tid] = other_score;
        shared_indices[tid] = other_index;
      }
    }
    __syncthreads();
  }
  if (tid == 0) {
    out_indices[row] = shared_indices[0];
    out_scores[row] = shared_scores[0];
  }
}

__global__ void lm_head_sample_topk_topp_bf16_kernel(
    const uint16_t* hidden, const uint16_t* lm_head, const float* random_uniforms,
    uint32_t* out_indices, float* out_scores, size_t rows, size_t hidden_dim, size_t vocab,
    float temperature, size_t top_k, float top_p) {
  const size_t row = blockIdx.x;
  const size_t tid = threadIdx.x;
  if (row >= rows) {
    return;
  }

  const uint16_t* row_hidden = hidden + row * hidden_dim;
  __shared__ float shared_logits[kBlock * kParallelSampleTopK];
  __shared__ uint32_t shared_indices[kBlock * kParallelSampleTopK];
  float local_logits[kParallelSampleTopK];
  uint32_t local_indices[kParallelSampleTopK];
  for (size_t rank = 0; rank < kParallelSampleTopK; ++rank) {
    local_logits[rank] = -CUDART_INF_F;
    local_indices[rank] = 0;
  }

  for (size_t token = tid; token < vocab; token += blockDim.x) {
    const uint16_t* weight_row = lm_head + token * hidden_dim;
    float logit = 0.0f;
    for (size_t col = 0; col < hidden_dim; ++col) {
      logit += bf16_to_f32(row_hidden[col]) * bf16_to_f32(weight_row[col]);
    }
    insert_topk_candidate(logit, static_cast<uint32_t>(token), local_logits, local_indices, top_k);
  }

  const size_t shared_base = tid * kParallelSampleTopK;
  for (size_t rank = 0; rank < kParallelSampleTopK; ++rank) {
    shared_logits[shared_base + rank] = local_logits[rank];
    shared_indices[shared_base + rank] = local_indices[rank];
  }
  __syncthreads();

  if (tid == 0) {
    float best_logits[kParallelSampleTopK];
    uint32_t best_indices[kParallelSampleTopK];
    for (size_t rank = 0; rank < kParallelSampleTopK; ++rank) {
      best_logits[rank] = -CUDART_INF_F;
      best_indices[rank] = 0;
    }
    for (size_t thread = 0; thread < blockDim.x; ++thread) {
      const size_t candidate_base = thread * kParallelSampleTopK;
      for (size_t rank = 0; rank < top_k; ++rank) {
        insert_topk_candidate(shared_logits[candidate_base + rank],
                              shared_indices[candidate_base + rank], best_logits, best_indices,
                              top_k);
      }
    }
    write_topk_topp_sample_small(best_logits, best_indices, random_uniforms, out_indices,
                                 out_scores, row, top_k, temperature, top_p);
  }
}

__global__ void lm_head_sample_topk_topp_bf16_serial_kernel(
    const uint16_t* hidden, const uint16_t* lm_head, const float* random_uniforms,
    uint32_t* out_indices, float* out_scores, size_t rows, size_t hidden_dim, size_t vocab,
    float temperature, size_t top_k, float top_p) {
  const size_t row = blockIdx.x;
  if (row >= rows || threadIdx.x != 0) {
    return;
  }
  float best_logits[kMaxSampleTopK];
  uint32_t best_indices[kMaxSampleTopK];
  const uint16_t* row_hidden = hidden + row * hidden_dim;
  compute_serial_lm_head_topk(row_hidden, lm_head, best_logits, best_indices, hidden_dim, vocab,
                              top_k);
  write_topk_topp_sample(best_logits, best_indices, random_uniforms, out_indices, out_scores, row,
                         top_k, temperature, top_p);
}

__global__ void lm_head_argmax_sample_topk_topp_bf16_kernel(
    const uint16_t* hidden, const uint16_t* lm_head, const float* random_uniforms,
    uint32_t* out_argmax_indices, float* out_argmax_scores, uint32_t* out_sample_indices,
    float* out_sample_scores, size_t rows, size_t hidden_dim, size_t vocab, float temperature,
    size_t top_k, float top_p) {
  const size_t row = blockIdx.x;
  const size_t tid = threadIdx.x;
  if (row >= rows) {
    return;
  }

  const uint16_t* row_hidden = hidden + row * hidden_dim;
  __shared__ float shared_logits[kBlock * kParallelSampleTopK];
  __shared__ uint32_t shared_indices[kBlock * kParallelSampleTopK];
  float local_logits[kParallelSampleTopK];
  uint32_t local_indices[kParallelSampleTopK];
  for (size_t rank = 0; rank < kParallelSampleTopK; ++rank) {
    local_logits[rank] = -CUDART_INF_F;
    local_indices[rank] = 0;
  }

  for (size_t token = tid; token < vocab; token += blockDim.x) {
    const uint16_t* weight_row = lm_head + token * hidden_dim;
    float logit = 0.0f;
    for (size_t col = 0; col < hidden_dim; ++col) {
      logit += bf16_to_f32(row_hidden[col]) * bf16_to_f32(weight_row[col]);
    }
    insert_topk_candidate(logit, static_cast<uint32_t>(token), local_logits, local_indices, top_k);
  }

  const size_t shared_base = tid * kParallelSampleTopK;
  for (size_t rank = 0; rank < kParallelSampleTopK; ++rank) {
    shared_logits[shared_base + rank] = local_logits[rank];
    shared_indices[shared_base + rank] = local_indices[rank];
  }
  __syncthreads();

  if (tid == 0) {
    float best_logits[kParallelSampleTopK];
    uint32_t best_indices[kParallelSampleTopK];
    for (size_t rank = 0; rank < kParallelSampleTopK; ++rank) {
      best_logits[rank] = -CUDART_INF_F;
      best_indices[rank] = 0;
    }
    for (size_t thread = 0; thread < blockDim.x; ++thread) {
      const size_t candidate_base = thread * kParallelSampleTopK;
      for (size_t rank = 0; rank < top_k; ++rank) {
        insert_topk_candidate(shared_logits[candidate_base + rank],
                              shared_indices[candidate_base + rank], best_logits, best_indices,
                              top_k);
      }
    }
    out_argmax_indices[row] = best_indices[0];
    out_argmax_scores[row] = best_logits[0];
    write_topk_topp_sample_small(best_logits, best_indices, random_uniforms, out_sample_indices,
                                 out_sample_scores, row, top_k, temperature, top_p);
  }
}

__global__ void lm_head_argmax_sample_topk_topp_bf16_serial_kernel(
    const uint16_t* hidden, const uint16_t* lm_head, const float* random_uniforms,
    uint32_t* out_argmax_indices, float* out_argmax_scores, uint32_t* out_sample_indices,
    float* out_sample_scores, size_t rows, size_t hidden_dim, size_t vocab, float temperature,
    size_t top_k, float top_p) {
  const size_t row = blockIdx.x;
  if (row >= rows || threadIdx.x != 0) {
    return;
  }
  float best_logits[kMaxSampleTopK];
  uint32_t best_indices[kMaxSampleTopK];
  const uint16_t* row_hidden = hidden + row * hidden_dim;
  compute_serial_lm_head_topk(row_hidden, lm_head, best_logits, best_indices, hidden_dim, vocab,
                              top_k);
  out_argmax_indices[row] = best_indices[0];
  out_argmax_scores[row] = best_logits[0];
  write_topk_topp_sample(best_logits, best_indices, random_uniforms, out_sample_indices,
                         out_sample_scores, row, top_k, temperature, top_p);
}

__global__ void lm_head_logits_bf16_kernel(const uint16_t* hidden, const uint16_t* lm_head,
                                           float* logits, size_t rows, size_t hidden_dim,
                                           size_t vocab) {
  __shared__ float scratch[kBlock];
  const size_t row_token = blockIdx.x;
  const size_t row = row_token / vocab;
  const size_t token = row_token % vocab;
  const size_t tid = threadIdx.x;
  if (row >= rows) {
    return;
  }

  const uint16_t* row_hidden = hidden + row * hidden_dim;
  const uint16_t* weight_row = lm_head + token * hidden_dim;
  float score = 0.0f;
  for (size_t col = tid; col < hidden_dim; col += blockDim.x) {
    score = fmaf(bf16_to_f32(row_hidden[col]), bf16_to_f32(weight_row[col]), score);
  }
  scratch[tid] = score;
  __syncthreads();
  for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    logits[row * vocab + token] = scratch[0];
  }
}

constexpr size_t kDsparkProposalTokens = 5;

// Slot storage stays request-major because every request owns a stable KV and
// residual page. The terminal batch is transposed to position-major so each
// Markov dependency step sees one contiguous batch of active requests.
__global__ void dspark_gather_normalized_bf16_kernel(
    const uint16_t* normalized_hidden_by_slot, const uint32_t* active_slot_ids,
    uint16_t* compact_normalized_hidden, size_t active_requests, size_t max_slots,
    size_t hidden_dim) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t rows = active_requests * kDsparkProposalTokens;
  const size_t total = rows * hidden_dim;
  if (idx >= total) {
    return;
  }
  const size_t compact_row = idx / hidden_dim;
  const size_t col = idx % hidden_dim;
  const size_t position = compact_row / active_requests;
  const size_t request = compact_row % active_requests;
  const size_t slot = active_slot_ids[request];
  if (slot >= max_slots) {
    compact_normalized_hidden[idx] = 0;
    return;
  }
  const size_t slot_row = slot * kDsparkProposalTokens + position;
  compact_normalized_hidden[idx] =
      normalized_hidden_by_slot[slot_row * hidden_dim + col];
}

__global__ void dspark_initialize_anchor_tokens_kernel(
    const uint32_t* anchor_token_ids, uint32_t* output_token_ids,
    size_t active_requests) {
  const size_t request = blockIdx.x * blockDim.x + threadIdx.x;
  if (request < active_requests) {
    // Position-major [anchor + five proposals, active requests].
    output_token_ids[request] = anchor_token_ids[request];
  }
}

__global__ void dspark_gather_markov_embedding_bf16_kernel(
    const uint16_t* markov_w1, const uint32_t* output_token_ids,
    uint16_t* markov_embeddings, size_t active_requests, size_t vocab,
    size_t markov_rank, size_t position) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = active_requests * markov_rank;
  if (idx >= total) {
    return;
  }
  const size_t request = idx / markov_rank;
  const size_t col = idx % markov_rank;
  const uint32_t token_id = output_token_ids[position * active_requests + request];
  const size_t output_offset =
      (position * active_requests + request) * markov_rank + col;
  markov_embeddings[output_offset] = token_id < vocab
      ? markov_w1[static_cast<size_t>(token_id) * markov_rank + col]
      : 0;
}

__global__ void dspark_add_markov_argmax_f32_kernel(
    float* shared_logits, const float* markov_logits,
    uint32_t* output_token_ids, size_t active_requests, size_t vocab,
    size_t position) {
  __shared__ float shared_scores[kBlock];
  __shared__ uint32_t shared_indices[kBlock];
  const size_t request = blockIdx.x;
  const size_t tid = threadIdx.x;
  if (request >= active_requests) {
    return;
  }
  float* base =
      shared_logits + (position * active_requests + request) * vocab;
  const float* bias = markov_logits + request * vocab;
  float best_score = -CUDART_INF_F;
  uint32_t best_index = 0;
  for (size_t token = tid; token < vocab; token += blockDim.x) {
    const float score = base[token] + bias[token];
    base[token] = score;
    const uint32_t token_id = static_cast<uint32_t>(token);
    if (score > best_score || (score == best_score && token_id < best_index)) {
      best_score = score;
      best_index = token_id;
    }
  }
  shared_scores[tid] = best_score;
  shared_indices[tid] = best_index;
  __syncthreads();
  for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      const float other_score = shared_scores[tid + stride];
      const uint32_t other_index = shared_indices[tid + stride];
      if (other_score > shared_scores[tid] ||
          (other_score == shared_scores[tid] &&
           other_index < shared_indices[tid])) {
        shared_scores[tid] = other_score;
        shared_indices[tid] = other_index;
      }
    }
    __syncthreads();
  }
  if (tid == 0) {
    output_token_ids[(position + 1) * active_requests + request] =
        shared_indices[0];
  }
}

__global__ void dspark_confidence_bf16_kernel(
    const uint16_t* collapsed_hidden_by_slot,
    const uint16_t* markov_embeddings, const uint16_t* confidence_weight,
    const uint32_t* active_slot_ids, float* conditional_confidence,
    size_t active_requests, size_t max_slots, size_t hidden_dim,
    size_t markov_rank) {
  __shared__ float scratch[kBlock];
  const size_t compact_row = blockIdx.x;
  const size_t tid = threadIdx.x;
  const size_t position = compact_row / active_requests;
  const size_t request = compact_row % active_requests;
  const size_t slot = active_slot_ids[request];
  if (slot >= max_slots) {
    if (tid == 0) {
      conditional_confidence[compact_row] = 0.5f;
    }
    return;
  }
  const uint16_t* hidden = collapsed_hidden_by_slot +
      (slot * kDsparkProposalTokens + position) * hidden_dim;
  const uint16_t* markov =
      markov_embeddings + compact_row * markov_rank;
  float partial = 0.0f;
  for (size_t col = tid; col < hidden_dim + markov_rank;
       col += blockDim.x) {
    const float input = col < hidden_dim
        ? bf16_to_f32(hidden[col])
        : bf16_to_f32(markov[col - hidden_dim]);
    partial = fmaf(input, bf16_to_f32(confidence_weight[col]), partial);
  }
  scratch[tid] = partial;
  __syncthreads();
  for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    conditional_confidence[compact_row] = sigmoid_f32(scratch[0]);
  }
}

template <bool CheckFinite>
__global__ void logits_argmax_f32_kernel(const float* logits, uint32_t* out_indices,
                                         float* out_scores, size_t rows, size_t vocab) {
  __shared__ float shared_scores[kBlock];
  __shared__ uint32_t shared_indices[kBlock];
  __shared__ int shared_invalid[kBlock];

  const size_t row = blockIdx.x;
  const size_t tid = threadIdx.x;
  if (row >= rows) {
    return;
  }

  const float* row_logits = logits + row * vocab;
  float best_score = -CUDART_INF_F;
  uint32_t best_index = 0;
  int invalid = 0;
  for (size_t col = tid; col < vocab; col += blockDim.x) {
    const float score = row_logits[col];
    if constexpr (CheckFinite) invalid |= !isfinite(score);
    const uint32_t token_id = static_cast<uint32_t>(col);
    if (score > best_score || (score == best_score && token_id < best_index)) {
      best_score = score;
      best_index = token_id;
    }
  }

  shared_scores[tid] = best_score;
  shared_indices[tid] = best_index;
  if constexpr (CheckFinite) shared_invalid[tid] = invalid;
  __syncthreads();
  for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      if constexpr (CheckFinite) shared_invalid[tid] |= shared_invalid[tid + stride];
      const float other_score = shared_scores[tid + stride];
      const uint32_t other_index = shared_indices[tid + stride];
      if (other_score > shared_scores[tid] ||
          (other_score == shared_scores[tid] && other_index < shared_indices[tid])) {
        shared_scores[tid] = other_score;
        shared_indices[tid] = other_index;
      }
    }
    __syncthreads();
  }
  if (tid == 0) {
    out_indices[row] = shared_indices[0];
    if constexpr (CheckFinite) out_scores[row] = shared_invalid[0] ? CUDART_NAN_F : shared_scores[0];
    else out_scores[row] = shared_scores[0];
  }
}

// Row-parallel argmax for decode-sized batches. One 1024-thread block per row
// keeps many vectorized loads in flight; the original 256-thread scalar scan was
// memory-latency bound (about 171 us for six 129280-entry rows on SM120). The
// packed key orders by logit and then by ascending token ID, so ties and the
// NaN/-inf behaviour match `logits_argmax_f32_kernel` exactly: NaN is never
// selected and an all-(-inf) row yields token 0.
constexpr int kWideArgmaxBlock = 1024;

template <bool CheckFinite>
__global__ void __launch_bounds__(kWideArgmaxBlock) logits_argmax_f32_wide_kernel(
    const float* logits, uint32_t* out_indices, float* out_scores, size_t vocab) {
  __shared__ uint64_t warp_best[kWideArgmaxBlock / 32];
  __shared__ int warp_invalid[kWideArgmaxBlock / 32];
  const size_t row = blockIdx.x;
  const float4* row4 = reinterpret_cast<const float4*>(logits + row * vocab);
  const uint32_t vectors = static_cast<uint32_t>(vocab / 4);
  uint64_t best = topk_sort_key(-CUDART_INF_F, 0);
  int invalid = 0;
  auto consider = [&](float score, uint32_t token) {
    if constexpr (CheckFinite) invalid |= !isfinite(score);
    if (!isnan(score)) {
      const uint64_t key = topk_sort_key(score, token);
      best = key > best ? key : best;
    }
  };
#pragma unroll 4
  for (uint32_t i = threadIdx.x; i < vectors; i += kWideArgmaxBlock) {
    const float4 v = __ldg(row4 + i);
    const uint32_t token = i * 4;
    consider(v.x, token);
    consider(v.y, token + 1);
    consider(v.z, token + 2);
    consider(v.w, token + 3);
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    const uint64_t other = __shfl_down_sync(0xffffffffU, best, offset);
    best = other > best ? other : best;
    if constexpr (CheckFinite) invalid |= __shfl_down_sync(0xffffffffU, invalid, offset);
  }
  const int warp = threadIdx.x / 32;
  if ((threadIdx.x & 31) == 0) {
    warp_best[warp] = best;
    if constexpr (CheckFinite) warp_invalid[warp] = invalid;
  }
  __syncthreads();
  if (warp == 0) {
    best = warp_best[threadIdx.x];
    if constexpr (CheckFinite) invalid = warp_invalid[threadIdx.x];
    for (int offset = 16; offset > 0; offset >>= 1) {
      const uint64_t other = __shfl_down_sync(0xffffffffU, best, offset);
      best = other > best ? other : best;
      if constexpr (CheckFinite) invalid |= __shfl_down_sync(0xffffffffU, invalid, offset);
    }
    if (threadIdx.x == 0) {
      out_indices[row] = ~static_cast<uint32_t>(best);
      const float score = ordered_bits_to_float(static_cast<uint32_t>(best >> 32));
      if constexpr (CheckFinite) out_scores[row] = invalid ? CUDART_NAN_F : score;
      else out_scores[row] = score;
    }
  }
}

bool wide_argmax_eligible(const float* logits, size_t vocab) {
  return vocab % 4 == 0 && vocab / 4 <= std::numeric_limits<uint32_t>::max() &&
         reinterpret_cast<uintptr_t>(logits) % 16 == 0;
}

__global__ void logits_sample_topk_topp_f32_kernel(const float* logits,
                                                   const float* random_uniforms,
                                                   uint32_t* out_indices, float* out_scores,
                                                   size_t rows, size_t vocab, float temperature,
                                                   size_t top_k, float top_p) {
  const size_t row = blockIdx.x;
  if (row >= rows || threadIdx.x != 0) {
    return;
  }

  float best_logits[kMaxSampleTopK];
  uint32_t best_indices[kMaxSampleTopK];
  for (size_t rank = 0; rank < top_k; ++rank) {
    best_logits[rank] = -CUDART_INF_F;
    best_indices[rank] = 0;
  }

  const float* row_logits = logits + row * vocab;
  for (size_t col = 0; col < vocab; ++col) {
    const float logit = row_logits[col];
    const uint32_t token_id = static_cast<uint32_t>(col);
    for (size_t rank = 0; rank < top_k; ++rank) {
      if (logit > best_logits[rank] ||
          (logit == best_logits[rank] && token_id < best_indices[rank])) {
        for (size_t shift = top_k - 1; shift > rank; --shift) {
          best_logits[shift] = best_logits[shift - 1];
          best_indices[shift] = best_indices[shift - 1];
        }
        best_logits[rank] = logit;
        best_indices[rank] = token_id;
        break;
      }
    }
  }

  float scaled[kMaxSampleTopK];
  float max_scaled = -CUDART_INF_F;
  for (size_t rank = 0; rank < top_k; ++rank) {
    scaled[rank] = best_logits[rank] / temperature;
    max_scaled = fmaxf(max_scaled, scaled[rank]);
  }

  float probs[kMaxSampleTopK];
  float total = 0.0f;
  for (size_t rank = 0; rank < top_k; ++rank) {
    probs[rank] = expf(scaled[rank] - max_scaled);
    total += probs[rank];
  }
  total = fmaxf(total, 1.0e-20f);
  for (size_t rank = 0; rank < top_k; ++rank) {
    probs[rank] /= total;
  }

  const float top_p_clamped = fminf(fmaxf(top_p, 1.0e-6f), 1.0f);
  float nucleus_mass = 0.0f;
  size_t nucleus_count = 0;
  for (size_t rank = 0; rank < top_k; ++rank) {
    nucleus_mass += probs[rank];
    nucleus_count = rank + 1;
    if (nucleus_mass >= top_p_clamped) {
      break;
    }
  }
  nucleus_mass = fmaxf(nucleus_mass, 1.0e-20f);

  const float target = fminf(fmaxf(random_uniforms[row], 0.0f), 0.99999994f) * nucleus_mass;
  float cumulative = 0.0f;
  size_t selected_rank = nucleus_count - 1;
  for (size_t rank = 0; rank < nucleus_count; ++rank) {
    cumulative += probs[rank];
    if (target <= cumulative) {
      selected_rank = rank;
      break;
    }
  }

  out_indices[row] = best_indices[selected_rank];
  out_scores[row] = probs[selected_rank] / nucleus_mass;
}

__global__ void apply_token_bitmask_f32_candidate_kernel(
    const float* logits, const uint32_t* token_bitmask, float* masked_logits, size_t rows,
    size_t vocab, size_t mask_words) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t values = rows * vocab;
  if (index >= values) {
    return;
  }
  const size_t row = index / vocab;
  const size_t token = index % vocab;
  const uint32_t word = token_bitmask[row * mask_words + token / 32];
  const bool allowed = (word & (uint32_t{1} << static_cast<uint32_t>(token % 32))) != 0;
  masked_logits[index] = allowed ? logits[index] : -CUDART_INF_F;
}

__global__ void masked_logits_argmax_partial_f32_candidate_kernel(
    const float* logits, const uint32_t* token_bitmask, uint64_t* partial_keys, size_t rows,
    size_t vocab, size_t mask_words, size_t blocks_per_row) {
  __shared__ typename MaskedArgmaxBlockReduce::TempStorage reduce_storage;
  const size_t row = blockIdx.x / blocks_per_row;
  const size_t shard = blockIdx.x % blocks_per_row;
  if (row >= rows) {
    return;
  }

  const float* row_logits = logits + row * vocab;
  const uint32_t* row_mask = token_bitmask + row * mask_words;
  float best_score = -CUDART_INF_F;
  uint32_t best_index = 0;
  for (size_t token = shard * blockDim.x + threadIdx.x; token < vocab;
       token += blocks_per_row * blockDim.x) {
    const uint32_t word = row_mask[token / 32];
    if ((word & (uint32_t{1} << static_cast<uint32_t>(token % 32))) == 0) {
      continue;
    }
    const float score = row_logits[token];
    const uint32_t token_id = static_cast<uint32_t>(token);
    if (score > best_score || (score == best_score && token_id < best_index)) {
      best_score = score;
      best_index = token_id;
    }
  }
  const uint64_t best_key = MaskedArgmaxBlockReduce(reduce_storage).Reduce(
      topk_sort_key(best_score, best_index), MaxUint64{}
  );
  if (threadIdx.x == 0) {
    partial_keys[blockIdx.x] = best_key;
  }
}

__global__ void masked_logits_argmax_finalize_f32_candidate_kernel(
    const uint64_t* partial_keys, uint32_t* out_indices, float* out_scores, size_t rows,
    size_t blocks_per_row) {
  __shared__ typename MaskedArgmaxBlockReduce::TempStorage reduce_storage;
  const size_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  uint64_t best_key = topk_sort_key(-CUDART_INF_F, 0);
  for (size_t shard = threadIdx.x; shard < blocks_per_row; shard += blockDim.x) {
    best_key = max(best_key, partial_keys[row * blocks_per_row + shard]);
  }
  best_key = MaskedArgmaxBlockReduce(reduce_storage).Reduce(best_key, MaxUint64{});
  if (threadIdx.x == 0) {
    out_indices[row] = ~static_cast<uint32_t>(best_key);
    out_scores[row] = 1.0f;
  }
}

__global__ void masked_logits_topk_partial_f32_candidate_kernel(
    const float* logits, const uint32_t* token_bitmask, uint64_t* partial_keys, size_t rows,
    size_t vocab, size_t mask_words, size_t blocks_per_row) {
  __shared__ typename MaskedTopKBlockSort::TempStorage sort_storage;
  const size_t row = blockIdx.x / blocks_per_row;
  const size_t shard = blockIdx.x % blocks_per_row;
  if (row >= rows) {
    return;
  }

  const float* row_logits = logits + row * vocab;
  const uint32_t* row_mask = token_bitmask + row * mask_words;
  float local_logits[kParallelSampleTopK];
  uint32_t local_indices[kParallelSampleTopK];
  for (size_t rank = 0; rank < kParallelSampleTopK; ++rank) {
    local_logits[rank] = -CUDART_INF_F;
    local_indices[rank] = 0;
  }
  for (size_t token = shard * blockDim.x + threadIdx.x; token < vocab;
       token += blocks_per_row * blockDim.x) {
    const uint32_t word = row_mask[token / 32];
    if ((word & (uint32_t{1} << static_cast<uint32_t>(token % 32))) == 0) {
      continue;
    }
    insert_topk_candidate(
        row_logits[token], static_cast<uint32_t>(token), local_logits, local_indices,
        kParallelSampleTopK
    );
  }

  uint64_t sort_keys[kParallelSampleTopK];
  for (size_t rank = 0; rank < kParallelSampleTopK; ++rank) {
    sort_keys[rank] = topk_sort_key(local_logits[rank], local_indices[rank]);
  }
  MaskedTopKBlockSort(sort_storage).SortDescending(sort_keys);
  if (threadIdx.x == 0) {
    const size_t output_base = blockIdx.x * kParallelSampleTopK;
    for (size_t rank = 0; rank < kParallelSampleTopK; ++rank) {
      partial_keys[output_base + rank] = sort_keys[rank];
    }
  }
}

__global__ void masked_logits_topk_finalize_f32_candidate_kernel(
    const uint64_t* partial_keys, const float* random_uniforms, uint32_t* out_indices,
    float* out_scores, size_t rows, size_t blocks_per_row, float temperature, size_t top_k,
    float top_p) {
  __shared__ typename MaskedTopKBlockSort::TempStorage sort_storage;
  const size_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const uint64_t placeholder = topk_sort_key(-CUDART_INF_F, 0);
  uint64_t sort_keys[kParallelSampleTopK];
  for (size_t rank = 0; rank < kParallelSampleTopK; ++rank) {
    sort_keys[rank] = threadIdx.x < blocks_per_row
                          ? partial_keys[((row * blocks_per_row + threadIdx.x) *
                                          kParallelSampleTopK) +
                                         rank]
                          : placeholder;
  }
  MaskedTopKBlockSort(sort_storage).SortDescending(sort_keys);
  if (threadIdx.x == 0) {
    float best_logits[kParallelSampleTopK];
    uint32_t best_indices[kParallelSampleTopK];
    for (size_t rank = 0; rank < kParallelSampleTopK; ++rank) {
      const uint64_t key = sort_keys[rank];
      best_logits[rank] = ordered_bits_to_float(static_cast<uint32_t>(key >> 32));
      best_indices[rank] = ~static_cast<uint32_t>(key);
    }
    write_topk_topp_sample_small(
        best_logits, best_indices, random_uniforms, out_indices, out_scores, row, top_k,
        temperature, top_p
    );
  }
}

__global__ void logits_sample_topk_topp_f32_cub_fill_indices_offsets_kernel(
    uint32_t* indices, int* segment_offsets, size_t rows, size_t vocab) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = rows * vocab;
  if (idx < total) {
    indices[idx] = static_cast<uint32_t>(idx % vocab);
  }
  if (idx <= rows) {
    segment_offsets[idx] = static_cast<int>(idx * vocab);
  }
}

__global__ void logits_sample_topk_topp_f32_cub_finalize_kernel(
    const float* sorted_logits, const uint32_t* sorted_indices, const float* random_uniforms,
    uint32_t* out_indices, float* out_scores, size_t rows, size_t vocab, float temperature,
    size_t top_k, float top_p) {
  const size_t row = blockIdx.x;
  if (row >= rows || threadIdx.x != 0) {
    return;
  }

  const size_t row_offset = row * vocab;
  float scaled[kMaxSampleTopK];
  float max_scaled = -CUDART_INF_F;
  for (size_t rank = 0; rank < top_k; ++rank) {
    scaled[rank] = sorted_logits[row_offset + rank] / temperature;
    max_scaled = fmaxf(max_scaled, scaled[rank]);
  }

  float probs[kMaxSampleTopK];
  float total = 0.0f;
  for (size_t rank = 0; rank < top_k; ++rank) {
    probs[rank] = expf(scaled[rank] - max_scaled);
    total += probs[rank];
  }
  total = fmaxf(total, 1.0e-20f);
  for (size_t rank = 0; rank < top_k; ++rank) {
    probs[rank] /= total;
  }

  const float top_p_clamped = fminf(fmaxf(top_p, 1.0e-6f), 1.0f);
  float nucleus_mass = 0.0f;
  size_t nucleus_count = 0;
  for (size_t rank = 0; rank < top_k; ++rank) {
    nucleus_mass += probs[rank];
    nucleus_count = rank + 1;
    if (nucleus_mass >= top_p_clamped) {
      break;
    }
  }
  nucleus_mass = fmaxf(nucleus_mass, 1.0e-20f);

  const float target = fminf(fmaxf(random_uniforms[row], 0.0f), 0.99999994f) * nucleus_mass;
  float cumulative = 0.0f;
  size_t selected_rank = nucleus_count - 1;
  for (size_t rank = 0; rank < nucleus_count; ++rank) {
    cumulative += probs[rank];
    if (target <= cumulative) {
      selected_rank = rank;
      break;
    }
  }

  out_indices[row] = sorted_indices[row_offset + selected_rank];
  out_scores[row] = probs[selected_rank] / nucleus_mass;
}

ds41rt_status_t validate_lm_head_argmax_bf16_args(const uint16_t* hidden, const uint16_t* lm_head,
                                                 const uint32_t* out_indices,
                                                 const float* out_scores, size_t rows,
                                                 size_t hidden_dim, size_t vocab) {
  if (hidden == nullptr || lm_head == nullptr || out_indices == nullptr || out_scores == nullptr) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  if (rows == 0 || hidden_dim == 0 || vocab == 0 ||
      rows > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      vocab > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  size_t ignored = 0;
  if (!checked_mul(rows, hidden_dim, &ignored) || !checked_mul(vocab, hidden_dim, &ignored)) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  return DS41RT_STATUS_OK;
}

ds41rt_status_t validate_dspark_terminal_greedy_bf16_args(
    const uint16_t* normalized_hidden_by_slot,
    const uint16_t* collapsed_hidden_by_slot, const uint16_t* shared_head,
    const uint16_t* markov_w1, const uint16_t* markov_w2,
    const uint16_t* confidence_weight, const uint32_t* active_slot_ids,
    const uint32_t* anchor_token_ids, const uint16_t* compact_normalized_hidden,
    const float* shared_logits, const uint16_t* markov_embeddings,
    const float* markov_logits, const uint32_t* output_token_ids,
    const float* conditional_confidence, size_t active_requests,
    size_t max_slots, size_t hidden_dim, size_t vocab, size_t markov_rank) {
  if (normalized_hidden_by_slot == nullptr ||
      collapsed_hidden_by_slot == nullptr || shared_head == nullptr ||
      markov_w1 == nullptr || markov_w2 == nullptr ||
      confidence_weight == nullptr || active_slot_ids == nullptr ||
      anchor_token_ids == nullptr || compact_normalized_hidden == nullptr ||
      shared_logits == nullptr || markov_embeddings == nullptr ||
      markov_logits == nullptr || output_token_ids == nullptr ||
      conditional_confidence == nullptr || active_requests == 0 ||
      active_requests > max_slots || hidden_dim == 0 || vocab == 0 ||
      markov_rank == 0 ||
      active_requests > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      vocab > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  size_t ignored = 0;
  if (!checked_mul(max_slots, kDsparkProposalTokens, &ignored) ||
      !checked_mul(ignored, hidden_dim, &ignored) ||
      !checked_mul(active_requests, kDsparkProposalTokens, &ignored) ||
      !checked_mul(ignored, hidden_dim, &ignored) ||
      !checked_mul(vocab, hidden_dim, &ignored) ||
      !checked_mul(vocab, markov_rank, &ignored) ||
      !checked_mul(active_requests, kDsparkProposalTokens, &ignored) ||
      !checked_mul(ignored, vocab, &ignored) ||
      !checked_mul(active_requests, kDsparkProposalTokens, &ignored) ||
      !checked_mul(ignored, markov_rank, &ignored) ||
      !checked_mul(active_requests, vocab, &ignored) ||
      !checked_mul(active_requests, kDsparkProposalTokens + 1, &ignored) ||
      !checked_add(hidden_dim, markov_rank, &ignored)) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  return DS41RT_STATUS_OK;
}

ds41rt_status_t validate_lm_head_sample_topk_topp_bf16_args(
    const uint16_t* hidden, const uint16_t* lm_head, const float* random_uniforms,
    const uint32_t* out_indices, const float* out_scores, size_t rows, size_t hidden_dim,
    size_t vocab, float temperature, size_t top_k, float top_p) {
  if (hidden == nullptr || lm_head == nullptr || random_uniforms == nullptr ||
      out_indices == nullptr || out_scores == nullptr) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  if (rows == 0 || hidden_dim == 0 || vocab == 0 ||
      rows > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      vocab > static_cast<size_t>(std::numeric_limits<uint32_t>::max()) || top_k == 0 ||
      top_k > vocab || top_k > kMaxSampleTopK || temperature <= 0.0f ||
      !std::isfinite(temperature) || top_p <= 0.0f || top_p > 1.0f || !std::isfinite(top_p)) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  size_t ignored = 0;
  if (!checked_mul(rows, hidden_dim, &ignored) || !checked_mul(vocab, hidden_dim, &ignored)) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  return DS41RT_STATUS_OK;
}

ds41rt_status_t validate_lm_head_argmax_sample_topk_topp_bf16_staged_args(
    const uint16_t* hidden, const uint16_t* lm_head, const float* random_uniforms,
    const uint32_t* out_argmax_indices, const float* out_argmax_scores,
    const uint32_t* out_sample_indices, const float* out_sample_scores, const float* logits,
    size_t rows, size_t hidden_dim, size_t vocab, float temperature, size_t top_k, float top_p) {
  if (hidden == nullptr || lm_head == nullptr || random_uniforms == nullptr ||
      out_argmax_indices == nullptr || out_argmax_scores == nullptr ||
      out_sample_indices == nullptr || out_sample_scores == nullptr || logits == nullptr) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  if (rows == 0 || hidden_dim == 0 || vocab == 0 ||
      rows > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      vocab > static_cast<size_t>(std::numeric_limits<uint32_t>::max()) || top_k == 0 ||
      top_k > vocab || top_k > kMaxSampleTopK || temperature <= 0.0f ||
      !std::isfinite(temperature) || top_p <= 0.0f || top_p > 1.0f || !std::isfinite(top_p)) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  size_t hidden_values = 0;
  size_t weight_values = 0;
  size_t logits_values = 0;
  if (!checked_mul(rows, hidden_dim, &hidden_values) ||
      !checked_mul(vocab, hidden_dim, &weight_values) ||
      !checked_mul(rows, vocab, &logits_values) ||
      logits_values > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  return DS41RT_STATUS_OK;
}

ds41rt_status_t validate_logits_argmax_args(const float* logits, const uint32_t* out_indices,
                                           const float* out_scores, size_t rows, size_t vocab) {
  if (logits == nullptr || out_indices == nullptr || out_scores == nullptr) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  if (rows == 0 || vocab == 0 || rows > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      vocab > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  size_t ignored = 0;
  if (!checked_mul(rows, vocab, &ignored)) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  return DS41RT_STATUS_OK;
}

ds41rt_status_t validate_logits_sample_topk_topp_args(const float* logits,
                                                     const float* random_uniforms,
                                                     const uint32_t* out_indices,
                                                     const float* out_scores, size_t rows,
                                                     size_t vocab, float temperature,
                                                     size_t top_k, float top_p) {
  if (logits == nullptr || random_uniforms == nullptr || out_indices == nullptr ||
      out_scores == nullptr) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  if (rows == 0 || vocab == 0 || rows > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      vocab > static_cast<size_t>(std::numeric_limits<uint32_t>::max()) || top_k == 0 ||
      top_k > vocab || top_k > kMaxSampleTopK || temperature <= 0.0f ||
      !std::isfinite(temperature) || top_p <= 0.0f || top_p > 1.0f || !std::isfinite(top_p)) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  size_t ignored = 0;
  if (!checked_mul(rows, vocab, &ignored)) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  return DS41RT_STATUS_OK;
}

ds41rt_status_t validate_masked_logits_sample_topk_topp_candidate_args(
    const float* logits, const uint32_t* token_bitmask, const float* random_uniforms,
    const uint32_t* out_indices, const float* out_scores, size_t rows, size_t vocab,
    size_t mask_words, float temperature, size_t top_k, float top_p) {
  const ds41rt_status_t valid = validate_logits_sample_topk_topp_args(
      logits, random_uniforms, out_indices, out_scores, rows, vocab, temperature, top_k, top_p
  );
  if (valid != DS41RT_STATUS_OK) {
    return valid;
  }
  if (token_bitmask == nullptr || top_k > kParallelSampleTopK || mask_words < (vocab + 31) / 32) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  size_t ignored = 0;
  if (!checked_mul(rows, mask_words, &ignored)) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  return DS41RT_STATUS_OK;
}

ds41rt_status_t validate_logits_sample_topk_topp_cub_args(
    const float* logits, const float* random_uniforms, float* sorted_logits,
    uint32_t* unsorted_indices, uint32_t* sorted_indices, int* segment_offsets,
    uint32_t* out_indices, float* out_scores, void* cub_temp_storage,
    size_t cub_temp_storage_bytes, size_t rows, size_t vocab, float temperature, size_t top_k,
    float top_p) {
  const ds41rt_status_t valid = validate_logits_sample_topk_topp_args(
      logits, random_uniforms, out_indices, out_scores, rows, vocab, temperature, top_k, top_p);
  if (valid != DS41RT_STATUS_OK) {
    return valid;
  }
  if (sorted_logits == nullptr || unsorted_indices == nullptr || sorted_indices == nullptr ||
      segment_offsets == nullptr || cub_temp_storage == nullptr || cub_temp_storage_bytes == 0) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  size_t logits_values = 0;
  if (!checked_mul(rows, vocab, &logits_values) ||
      logits_values > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      rows > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  return DS41RT_STATUS_OK;
}

ds41rt_status_t validate_lm_head_sample_topk_topp_bf16_cub_args(
    const uint16_t* hidden, const uint16_t* lm_head, const float* random_uniforms,
    float* logits_workspace, float* sorted_logits, uint32_t* unsorted_indices,
    uint32_t* sorted_indices, int* segment_offsets, uint32_t* out_indices, float* out_scores,
    void* cub_temp_storage, size_t cub_temp_storage_bytes, size_t rows, size_t hidden_dim,
    size_t vocab, float temperature, size_t top_k, float top_p) {
  const ds41rt_status_t valid = validate_lm_head_sample_topk_topp_bf16_args(
      hidden, lm_head, random_uniforms, out_indices, out_scores, rows, hidden_dim, vocab,
      temperature, top_k, top_p);
  if (valid != DS41RT_STATUS_OK) {
    return valid;
  }
  if (logits_workspace == nullptr || sorted_logits == nullptr || unsorted_indices == nullptr ||
      sorted_indices == nullptr || segment_offsets == nullptr || cub_temp_storage == nullptr ||
      cub_temp_storage_bytes == 0) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  size_t logits_values = 0;
  if (!checked_mul(rows, vocab, &logits_values) ||
      logits_values > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      rows > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  return DS41RT_STATUS_OK;
}

ds41rt_status_t validate_bf16_graph_lm_head_argmax_buffers(
    ds41rt_device_buffer_t hidden, ds41rt_device_buffer_t lm_head,
    ds41rt_device_buffer_t out_indices, ds41rt_device_buffer_t out_scores, size_t rows,
    size_t hidden_dim, size_t vocab) {
  const ds41rt_status_t valid = validate_lm_head_argmax_bf16_args(
      static_cast<const uint16_t*>(hidden.ptr), static_cast<const uint16_t*>(lm_head.ptr),
      static_cast<const uint32_t*>(out_indices.ptr), static_cast<const float*>(out_scores.ptr),
      rows, hidden_dim, vocab);
  if (valid != DS41RT_STATUS_OK) {
    return valid;
  }
  size_t hidden_values = 0;
  size_t weight_values = 0;
  if (!checked_mul(rows, hidden_dim, &hidden_values) ||
      !checked_mul(vocab, hidden_dim, &weight_values)) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  size_t hidden_bytes = 0;
  size_t weight_bytes = 0;
  size_t index_bytes = 0;
  size_t score_bytes = 0;
  if (!checked_mul(hidden_values, sizeof(uint16_t), &hidden_bytes) ||
      !checked_mul(weight_values, sizeof(uint16_t), &weight_bytes) ||
      !checked_mul(rows, sizeof(uint32_t), &index_bytes) ||
      !checked_mul(rows, sizeof(float), &score_bytes)) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  if (hidden.bytes < hidden_bytes || lm_head.bytes < weight_bytes ||
      out_indices.bytes < index_bytes || out_scores.bytes < score_bytes) {
    return DS41RT_STATUS_BUFFER_TOO_SMALL;
  }
  return DS41RT_STATUS_OK;
}

ds41rt_status_t validate_bf16_graph_lm_head_sample_topk_topp_buffers(
    ds41rt_device_buffer_t hidden, ds41rt_device_buffer_t lm_head,
    ds41rt_device_buffer_t random_uniforms, ds41rt_device_buffer_t out_indices,
    ds41rt_device_buffer_t out_scores, size_t rows, size_t hidden_dim, size_t vocab,
    float temperature, size_t top_k, float top_p) {
  const ds41rt_status_t valid = validate_lm_head_sample_topk_topp_bf16_args(
      static_cast<const uint16_t*>(hidden.ptr), static_cast<const uint16_t*>(lm_head.ptr),
      static_cast<const float*>(random_uniforms.ptr), static_cast<const uint32_t*>(out_indices.ptr),
      static_cast<const float*>(out_scores.ptr), rows, hidden_dim, vocab, temperature, top_k,
      top_p);
  if (valid != DS41RT_STATUS_OK) {
    return valid;
  }
  size_t hidden_values = 0;
  size_t weight_values = 0;
  if (!checked_mul(rows, hidden_dim, &hidden_values) ||
      !checked_mul(vocab, hidden_dim, &weight_values)) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  size_t hidden_bytes = 0;
  size_t weight_bytes = 0;
  size_t random_bytes = 0;
  size_t index_bytes = 0;
  size_t score_bytes = 0;
  if (!checked_mul(hidden_values, sizeof(uint16_t), &hidden_bytes) ||
      !checked_mul(weight_values, sizeof(uint16_t), &weight_bytes) ||
      !checked_mul(rows, sizeof(float), &random_bytes) ||
      !checked_mul(rows, sizeof(uint32_t), &index_bytes) ||
      !checked_mul(rows, sizeof(float), &score_bytes)) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  if (hidden.bytes < hidden_bytes || lm_head.bytes < weight_bytes ||
      random_uniforms.bytes < random_bytes || out_indices.bytes < index_bytes ||
      out_scores.bytes < score_bytes) {
    return DS41RT_STATUS_BUFFER_TOO_SMALL;
  }
  return DS41RT_STATUS_OK;
}

}  // namespace

extern "C" ds41rt_status_t ds41rt_cuda_graph_update_lm_head_argmax_bf16_node(
    void* cuda_graph, void* cuda_graph_exec, size_t kernel_node_index,
    ds41rt_device_buffer_t hidden, ds41rt_device_buffer_t lm_head,
    ds41rt_device_buffer_t out_indices, ds41rt_device_buffer_t out_scores, size_t rows,
    size_t hidden_dim, size_t vocab) {
  if (cuda_graph == nullptr || cuda_graph_exec == nullptr) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  const ds41rt_status_t valid = validate_bf16_graph_lm_head_argmax_buffers(
      hidden, lm_head, out_indices, out_scores, rows, hidden_dim, vocab);
  if (valid != DS41RT_STATUS_OK) {
    return valid;
  }

  cudaGraphNode_t node = nullptr;
  const ds41rt_status_t node_status = find_kernel_node_by_index(cuda_graph, kernel_node_index, &node);
  if (node_status != DS41RT_STATUS_OK) {
    return node_status;
  }

  cudaKernelNodeParams existing = {};
  cudaError_t err = cudaGraphKernelNodeGetParams(node, &existing);
  if (err != cudaSuccess) {
    return DS41RT_STATUS_INTERNAL_ERROR;
  }
  if (existing.func != reinterpret_cast<void*>(lm_head_argmax_bf16_kernel)) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }

  const uint16_t* hidden_ptr = static_cast<const uint16_t*>(hidden.ptr);
  const uint16_t* lm_head_ptr = static_cast<const uint16_t*>(lm_head.ptr);
  uint32_t* out_indices_ptr = static_cast<uint32_t*>(out_indices.ptr);
  float* out_scores_ptr = static_cast<float*>(out_scores.ptr);
  void* args[] = {
      &hidden_ptr,
      &lm_head_ptr,
      &out_indices_ptr,
      &out_scores_ptr,
      &rows,
      &hidden_dim,
      &vocab,
  };

  cudaKernelNodeParams params = {};
  params.func = reinterpret_cast<void*>(lm_head_argmax_bf16_kernel);
  params.gridDim = dim3(static_cast<unsigned int>(rows), 1, 1);
  params.blockDim = dim3(kBlock, 1, 1);
  params.sharedMemBytes = 0;
  params.kernelParams = args;
  params.extra = nullptr;

  err = cudaGraphKernelNodeSetParams(node, &params);
  if (err != cudaSuccess) {
    return DS41RT_STATUS_INTERNAL_ERROR;
  }
  err = cudaGraphExecKernelNodeSetParams(reinterpret_cast<cudaGraphExec_t>(cuda_graph_exec), node,
                                         &params);
  if (err != cudaSuccess) {
    return DS41RT_STATUS_INTERNAL_ERROR;
  }
  return DS41RT_STATUS_OK;
}

extern "C" ds41rt_status_t ds41rt_cuda_graph_update_lm_head_sample_topk_topp_bf16_node(
    void* cuda_graph, void* cuda_graph_exec, size_t kernel_node_index,
    ds41rt_device_buffer_t hidden, ds41rt_device_buffer_t lm_head,
    ds41rt_device_buffer_t random_uniforms, ds41rt_device_buffer_t out_indices,
    ds41rt_device_buffer_t out_scores, size_t rows, size_t hidden_dim, size_t vocab,
    float temperature, size_t top_k, float top_p) {
  if (cuda_graph == nullptr || cuda_graph_exec == nullptr) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  const ds41rt_status_t valid = validate_bf16_graph_lm_head_sample_topk_topp_buffers(
      hidden, lm_head, random_uniforms, out_indices, out_scores, rows, hidden_dim, vocab,
      temperature, top_k, top_p);
  if (valid != DS41RT_STATUS_OK) {
    return valid;
  }

  cudaGraphNode_t node = nullptr;
  const ds41rt_status_t node_status = find_kernel_node_by_index(cuda_graph, kernel_node_index, &node);
  if (node_status != DS41RT_STATUS_OK) {
    return node_status;
  }

  cudaKernelNodeParams existing = {};
  cudaError_t err = cudaGraphKernelNodeGetParams(node, &existing);
  if (err != cudaSuccess) {
    return DS41RT_STATUS_INTERNAL_ERROR;
  }
  if (existing.func != reinterpret_cast<void*>(lm_head_sample_topk_topp_bf16_kernel)) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }

  const uint16_t* hidden_ptr = static_cast<const uint16_t*>(hidden.ptr);
  const uint16_t* lm_head_ptr = static_cast<const uint16_t*>(lm_head.ptr);
  const float* random_uniforms_ptr = static_cast<const float*>(random_uniforms.ptr);
  uint32_t* out_indices_ptr = static_cast<uint32_t*>(out_indices.ptr);
  float* out_scores_ptr = static_cast<float*>(out_scores.ptr);
  void* args[] = {
      &hidden_ptr,
      &lm_head_ptr,
      &random_uniforms_ptr,
      &out_indices_ptr,
      &out_scores_ptr,
      &rows,
      &hidden_dim,
      &vocab,
      &temperature,
      &top_k,
      &top_p,
  };

  cudaKernelNodeParams params = {};
  params.func = reinterpret_cast<void*>(lm_head_sample_topk_topp_bf16_kernel);
  params.gridDim = dim3(static_cast<unsigned int>(rows), 1, 1);
  params.blockDim = dim3(kBlock, 1, 1);
  params.sharedMemBytes = 0;
  params.kernelParams = args;
  params.extra = nullptr;

  err = cudaGraphKernelNodeSetParams(node, &params);
  if (err != cudaSuccess) {
    return DS41RT_STATUS_INTERNAL_ERROR;
  }
  err = cudaGraphExecKernelNodeSetParams(reinterpret_cast<cudaGraphExec_t>(cuda_graph_exec), node,
                                         &params);
  if (err != cudaSuccess) {
    return DS41RT_STATUS_INTERNAL_ERROR;
  }
  return DS41RT_STATUS_OK;
}

extern "C" ds41rt_status_t ds41rt_cuda_lm_head_argmax_bf16_async(
    const uint16_t* hidden, const uint16_t* lm_head, uint32_t* out_indices, float* out_scores,
    size_t rows, size_t hidden_dim, size_t vocab, void* cuda_stream) {
  const ds41rt_status_t valid = validate_lm_head_argmax_bf16_args(
      hidden, lm_head, out_indices, out_scores, rows, hidden_dim, vocab);
  if (valid != DS41RT_STATUS_OK) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  lm_head_argmax_bf16_kernel<<<static_cast<int>(rows), kBlock, 0, stream>>>(
      hidden, lm_head, out_indices, out_scores, rows, hidden_dim, vocab);
  return status_from_cuda(cudaGetLastError());
}

extern "C" ds41rt_status_t ds41rt_cuda_lm_head_argmax_bf16(
    const uint16_t* hidden, const uint16_t* lm_head, uint32_t* out_indices, float* out_scores,
    size_t rows, size_t hidden_dim, size_t vocab) {
  const ds41rt_status_t status = ds41rt_cuda_lm_head_argmax_bf16_async(
      hidden, lm_head, out_indices, out_scores, rows, hidden_dim, vocab, nullptr);
  if (status != DS41RT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" ds41rt_status_t ds41rt_cuda_lm_head_sample_topk_topp_bf16_async(
    const uint16_t* hidden, const uint16_t* lm_head, const float* random_uniforms,
    uint32_t* out_indices, float* out_scores, size_t rows, size_t hidden_dim, size_t vocab,
    float temperature, size_t top_k, float top_p, void* cuda_stream) {
  const ds41rt_status_t valid = validate_lm_head_sample_topk_topp_bf16_args(
      hidden, lm_head, random_uniforms, out_indices, out_scores, rows, hidden_dim, vocab,
      temperature, top_k, top_p);
  if (valid != DS41RT_STATUS_OK) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  if (top_k <= kParallelSampleTopK) {
    lm_head_sample_topk_topp_bf16_kernel<<<static_cast<int>(rows), kBlock, 0, stream>>>(
        hidden, lm_head, random_uniforms, out_indices, out_scores, rows, hidden_dim, vocab,
        temperature, top_k, top_p);
  } else {
    lm_head_sample_topk_topp_bf16_serial_kernel<<<static_cast<int>(rows), 1, 0, stream>>>(
        hidden, lm_head, random_uniforms, out_indices, out_scores, rows, hidden_dim, vocab,
        temperature, top_k, top_p);
  }
  return status_from_cuda(cudaGetLastError());
}

extern "C" ds41rt_status_t ds41rt_cuda_lm_head_sample_topk_topp_bf16(
    const uint16_t* hidden, const uint16_t* lm_head, const float* random_uniforms,
    uint32_t* out_indices, float* out_scores, size_t rows, size_t hidden_dim, size_t vocab,
    float temperature, size_t top_k, float top_p) {
  const ds41rt_status_t status = ds41rt_cuda_lm_head_sample_topk_topp_bf16_async(
      hidden, lm_head, random_uniforms, out_indices, out_scores, rows, hidden_dim, vocab,
      temperature, top_k, top_p, nullptr);
  if (status != DS41RT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" ds41rt_status_t ds41rt_cuda_lm_head_argmax_sample_topk_topp_bf16_staged_async(
    const uint16_t* hidden, const uint16_t* lm_head, const float* random_uniforms,
    uint32_t* out_argmax_indices, float* out_argmax_scores, uint32_t* out_sample_indices,
    float* out_sample_scores, float* logits_workspace, size_t rows, size_t hidden_dim,
    size_t vocab, float temperature, size_t top_k, float top_p, void* cuda_stream) {
  const ds41rt_status_t valid = validate_lm_head_argmax_sample_topk_topp_bf16_staged_args(
      hidden, lm_head, random_uniforms, out_argmax_indices, out_argmax_scores, out_sample_indices,
      out_sample_scores, logits_workspace, rows, hidden_dim, vocab, temperature, top_k, top_p);
  if (valid != DS41RT_STATUS_OK) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  if (top_k <= kParallelSampleTopK) {
    lm_head_argmax_sample_topk_topp_bf16_kernel<<<static_cast<int>(rows), kBlock, 0, stream>>>(
        hidden, lm_head, random_uniforms, out_argmax_indices, out_argmax_scores,
        out_sample_indices, out_sample_scores, rows, hidden_dim, vocab, temperature, top_k, top_p);
  } else {
    lm_head_argmax_sample_topk_topp_bf16_serial_kernel<<<static_cast<int>(rows), 1, 0, stream>>>(
        hidden, lm_head, random_uniforms, out_argmax_indices, out_argmax_scores,
        out_sample_indices, out_sample_scores, rows, hidden_dim, vocab, temperature, top_k, top_p);
  }
  return status_from_cuda(cudaGetLastError());
}

extern "C" ds41rt_status_t ds41rt_cuda_lm_head_argmax_sample_topk_topp_bf16_staged(
    const uint16_t* hidden, const uint16_t* lm_head, const float* random_uniforms,
    uint32_t* out_argmax_indices, float* out_argmax_scores, uint32_t* out_sample_indices,
    float* out_sample_scores, float* logits_workspace, size_t rows, size_t hidden_dim,
    size_t vocab, float temperature, size_t top_k, float top_p) {
  const ds41rt_status_t status = ds41rt_cuda_lm_head_argmax_sample_topk_topp_bf16_staged_async(
      hidden, lm_head, random_uniforms, out_argmax_indices, out_argmax_scores, out_sample_indices,
      out_sample_scores, logits_workspace, rows, hidden_dim, vocab, temperature, top_k, top_p,
      nullptr);
  if (status != DS41RT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" ds41rt_status_t ds41rt_cuda_lm_head_sample_topk_topp_bf16_cub_async(
    const uint16_t* hidden, const uint16_t* lm_head, const float* random_uniforms,
    float* logits_workspace, float* sorted_logits, uint32_t* unsorted_indices,
    uint32_t* sorted_indices, int* segment_offsets, uint32_t* out_indices, float* out_scores,
    void* cub_temp_storage, size_t cub_temp_storage_bytes, size_t rows, size_t hidden_dim,
    size_t vocab, float temperature, size_t top_k, float top_p, void* cuda_stream) {
  const ds41rt_status_t valid = validate_lm_head_sample_topk_topp_bf16_cub_args(
      hidden, lm_head, random_uniforms, logits_workspace, sorted_logits, unsorted_indices,
      sorted_indices, segment_offsets, out_indices, out_scores, cub_temp_storage,
      cub_temp_storage_bytes, rows, hidden_dim, vocab, temperature, top_k, top_p);
  if (valid != DS41RT_STATUS_OK) {
    return valid;
  }
  size_t logits_values = 0;
  if (!checked_mul(rows, vocab, &logits_values) ||
      logits_values > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  lm_head_logits_bf16_kernel<<<static_cast<int>(logits_values), kBlock, 0, stream>>>(
      hidden, lm_head, logits_workspace, rows, hidden_dim, vocab);
  ds41rt_status_t status = status_from_cuda(cudaGetLastError());
  if (status != DS41RT_STATUS_OK) {
    return status;
  }
  return ds41rt_cuda_logits_sample_topk_topp_f32_cub_async(
      logits_workspace, random_uniforms, sorted_logits, unsorted_indices, sorted_indices,
      segment_offsets, out_indices, out_scores, cub_temp_storage, cub_temp_storage_bytes, rows,
      vocab, temperature, top_k, top_p, cuda_stream);
}

extern "C" ds41rt_status_t ds41rt_cuda_lm_head_sample_topk_topp_bf16_cub(
    const uint16_t* hidden, const uint16_t* lm_head, const float* random_uniforms,
    float* logits_workspace, float* sorted_logits, uint32_t* unsorted_indices,
    uint32_t* sorted_indices, int* segment_offsets, uint32_t* out_indices, float* out_scores,
    void* cub_temp_storage, size_t cub_temp_storage_bytes, size_t rows, size_t hidden_dim,
    size_t vocab, float temperature, size_t top_k, float top_p) {
  const ds41rt_status_t status = ds41rt_cuda_lm_head_sample_topk_topp_bf16_cub_async(
      hidden, lm_head, random_uniforms, logits_workspace, sorted_logits, unsorted_indices,
      sorted_indices, segment_offsets, out_indices, out_scores, cub_temp_storage,
      cub_temp_storage_bytes, rows, hidden_dim, vocab, temperature, top_k, top_p, nullptr);
  if (status != DS41RT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" ds41rt_status_t ds41rt_cuda_logits_argmax_f32_async(
    const float* logits, uint32_t* out_indices, float* out_scores, size_t rows, size_t vocab,
    void* cuda_stream) {
  const ds41rt_status_t valid =
      validate_logits_argmax_args(logits, out_indices, out_scores, rows, vocab);
  if (valid != DS41RT_STATUS_OK) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  if (wide_argmax_eligible(logits, vocab)) {
    logits_argmax_f32_wide_kernel<false><<<static_cast<int>(rows), kWideArgmaxBlock, 0, stream>>>(
        logits, out_indices, out_scores, vocab);
  } else {
    logits_argmax_f32_kernel<false><<<static_cast<int>(rows), kBlock, 0, stream>>>(
        logits, out_indices, out_scores, rows, vocab);
  }
  return status_from_cuda(cudaGetLastError());
}

extern "C" ds41rt_status_t ds41rt_cuda_logits_argmax_checked_f32_async(
    const float* logits, uint32_t* out_indices, float* out_scores, size_t rows, size_t vocab,
    void* cuda_stream) {
  const ds41rt_status_t valid =
      validate_logits_argmax_args(logits, out_indices, out_scores, rows, vocab);
  if (valid != DS41RT_STATUS_OK) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  if (wide_argmax_eligible(logits, vocab)) {
    logits_argmax_f32_wide_kernel<true><<<static_cast<int>(rows), kWideArgmaxBlock, 0, stream>>>(
        logits, out_indices, out_scores, vocab);
  } else {
    logits_argmax_f32_kernel<true><<<static_cast<int>(rows), kBlock, 0, stream>>>(
        logits, out_indices, out_scores, rows, vocab);
  }
  return status_from_cuda(cudaGetLastError());
}

extern "C" ds41rt_status_t ds41rt_cuda_logits_argmax_f32(const float* logits,
                                                       uint32_t* out_indices, float* out_scores,
                                                       size_t rows, size_t vocab) {
  const ds41rt_status_t status =
      ds41rt_cuda_logits_argmax_f32_async(logits, out_indices, out_scores, rows, vocab, nullptr);
  if (status != DS41RT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" ds41rt_status_t ds41rt_cuda_dspark_terminal_greedy_bf16_async(
    const uint16_t* normalized_hidden_by_slot,
    const uint16_t* collapsed_hidden_by_slot, const uint16_t* shared_head,
    const uint16_t* markov_w1, const uint16_t* markov_w2,
    const uint16_t* confidence_weight, const uint32_t* active_slot_ids,
    const uint32_t* anchor_token_ids, uint16_t* compact_normalized_hidden,
    float* shared_logits, uint16_t* markov_embeddings, float* markov_logits,
    uint32_t* output_token_ids, float* conditional_confidence,
    size_t active_requests, size_t max_slots, size_t hidden_dim, size_t vocab,
    size_t markov_rank, void* cuda_stream) {
  const ds41rt_status_t valid = validate_dspark_terminal_greedy_bf16_args(
      normalized_hidden_by_slot, collapsed_hidden_by_slot, shared_head,
      markov_w1, markov_w2, confidence_weight, active_slot_ids,
      anchor_token_ids, compact_normalized_hidden, shared_logits,
      markov_embeddings, markov_logits, output_token_ids,
      conditional_confidence, active_requests, max_slots, hidden_dim, vocab,
      markov_rank);
  if (valid != DS41RT_STATUS_OK) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const int threads = 256;
  const size_t compact_rows = active_requests * kDsparkProposalTokens;
  const size_t hidden_values = compact_rows * hidden_dim;
  const size_t hidden_blocks = (hidden_values + threads - 1) / threads;
  const size_t request_blocks = (active_requests + threads - 1) / threads;
  const size_t markov_values = active_requests * markov_rank;
  const size_t markov_blocks = (markov_values + threads - 1) / threads;
  if (hidden_blocks > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      request_blocks > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      markov_blocks > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      compact_rows > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }

  dspark_gather_normalized_bf16_kernel<<<static_cast<int>(hidden_blocks),
      threads, 0, stream>>>(normalized_hidden_by_slot, active_slot_ids,
                           compact_normalized_hidden, active_requests,
                           max_slots, hidden_dim);
  ds41rt_status_t status = status_from_cuda(cudaGetLastError());
  if (status != DS41RT_STATUS_OK) {
    return status;
  }
  dspark_initialize_anchor_tokens_kernel<<<static_cast<int>(request_blocks),
      threads, 0, stream>>>(anchor_token_ids, output_token_ids,
                           active_requests);
  status = status_from_cuda(cudaGetLastError());
  if (status != DS41RT_STATUS_OK) {
    return status;
  }
  status = ds41rt_cuda_linear_bf16_f32_cublas_async(
      compact_normalized_hidden, shared_head, shared_logits, compact_rows,
      hidden_dim, vocab, cuda_stream);
  if (status != DS41RT_STATUS_OK) {
    return status;
  }

  // The host loop only enqueues fixed graph nodes. Device dependencies remain
  // position -> sampled token -> next position, while every launch covers all
  // active requests together.
  for (size_t position = 0; position < kDsparkProposalTokens; ++position) {
    dspark_gather_markov_embedding_bf16_kernel<<<
        static_cast<int>(markov_blocks), threads, 0, stream>>>(
        markov_w1, output_token_ids, markov_embeddings, active_requests,
        vocab, markov_rank, position);
    status = status_from_cuda(cudaGetLastError());
    if (status != DS41RT_STATUS_OK) {
      return status;
    }
    const uint16_t* position_embedding =
        markov_embeddings + position * active_requests * markov_rank;
    status = ds41rt_cuda_linear_bf16_f32_cublas_async(
        position_embedding, markov_w2, markov_logits, active_requests,
        markov_rank, vocab, cuda_stream);
    if (status != DS41RT_STATUS_OK) {
      return status;
    }
    dspark_add_markov_argmax_f32_kernel<<<static_cast<int>(active_requests),
        threads, 0, stream>>>(shared_logits, markov_logits,
                             output_token_ids, active_requests, vocab,
                             position);
    status = status_from_cuda(cudaGetLastError());
    if (status != DS41RT_STATUS_OK) {
      return status;
    }
  }

  dspark_confidence_bf16_kernel<<<static_cast<int>(compact_rows), threads, 0,
      stream>>>(collapsed_hidden_by_slot, markov_embeddings,
                confidence_weight, active_slot_ids, conditional_confidence,
                active_requests, max_slots, hidden_dim, markov_rank);
  return status_from_cuda(cudaGetLastError());
}

extern "C" ds41rt_status_t ds41rt_cuda_logits_sample_topk_topp_f32_async(
    const float* logits, const float* random_uniforms, uint32_t* out_indices, float* out_scores,
    size_t rows, size_t vocab, float temperature, size_t top_k, float top_p, void* cuda_stream) {
  const ds41rt_status_t valid = validate_logits_sample_topk_topp_args(
      logits, random_uniforms, out_indices, out_scores, rows, vocab, temperature, top_k, top_p);
  if (valid != DS41RT_STATUS_OK) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  logits_sample_topk_topp_f32_kernel<<<static_cast<int>(rows), 1, 0, stream>>>(
      logits, random_uniforms, out_indices, out_scores, rows, vocab, temperature, top_k, top_p);
  return status_from_cuda(cudaGetLastError());
}

extern "C" ds41rt_status_t ds41rt_cuda_logits_sample_topk_topp_f32(
    const float* logits, const float* random_uniforms, uint32_t* out_indices, float* out_scores,
    size_t rows, size_t vocab, float temperature, size_t top_k, float top_p) {
  const ds41rt_status_t status = ds41rt_cuda_logits_sample_topk_topp_f32_async(
      logits, random_uniforms, out_indices, out_scores, rows, vocab, temperature, top_k, top_p,
      nullptr);
  if (status != DS41RT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" ds41rt_status_t
ds41rt_cuda_logits_apply_bitmask_sample_topk_topp_f32_candidate_async(
    const float* logits, const uint32_t* token_bitmask, const float* random_uniforms,
    float* masked_logits, uint32_t* out_indices, float* out_scores, size_t rows, size_t vocab,
    size_t mask_words, float temperature, size_t top_k, float top_p, void* cuda_stream) {
  const ds41rt_status_t valid = validate_masked_logits_sample_topk_topp_candidate_args(
      logits, token_bitmask, random_uniforms, out_indices, out_scores, rows, vocab, mask_words,
      temperature, top_k, top_p
  );
  if (valid != DS41RT_STATUS_OK || masked_logits == nullptr) {
    return valid == DS41RT_STATUS_OK ? DS41RT_STATUS_INVALID_ARGUMENT : valid;
  }
  size_t values = 0;
  if (!checked_mul(rows, vocab, &values)) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  const size_t blocks = (values + kBlock - 1) / kBlock;
  if (blocks > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  apply_token_bitmask_f32_candidate_kernel<<<static_cast<unsigned int>(blocks), kBlock, 0, stream>>>(
      logits, token_bitmask, masked_logits, rows, vocab, mask_words
  );
  ds41rt_status_t status = status_from_cuda(cudaGetLastError());
  if (status != DS41RT_STATUS_OK) {
    return status;
  }
  logits_sample_topk_topp_f32_kernel<<<static_cast<unsigned int>(rows), 1, 0, stream>>>(
      masked_logits, random_uniforms, out_indices, out_scores, rows, vocab, temperature, top_k,
      top_p
  );
  return status_from_cuda(cudaGetLastError());
}

extern "C" ds41rt_status_t ds41rt_cuda_logits_masked_sample_topk_topp_f32_grid_candidate_async(
    const float* logits, const uint32_t* token_bitmask, const float* random_uniforms,
    uint64_t* partial_keys, uint32_t* out_indices, float* out_scores, size_t rows, size_t vocab,
    size_t mask_words, size_t blocks_per_row, float temperature, size_t top_k, float top_p,
    void* cuda_stream) {
  const ds41rt_status_t valid = validate_masked_logits_sample_topk_topp_candidate_args(
      logits, token_bitmask, random_uniforms, out_indices, out_scores, rows, vocab, mask_words,
      temperature, top_k, top_p
  );
  if (valid != DS41RT_STATUS_OK || partial_keys == nullptr || blocks_per_row == 0 ||
      blocks_per_row > kBlock) {
    return valid == DS41RT_STATUS_OK ? DS41RT_STATUS_INVALID_ARGUMENT : valid;
  }
  size_t total_blocks = 0;
  if (!checked_mul(rows, blocks_per_row, &total_blocks) ||
      total_blocks > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  if (top_k == 1) {
    masked_logits_argmax_partial_f32_candidate_kernel<<<
        static_cast<unsigned int>(total_blocks), kBlock, 0, stream>>>(
        logits, token_bitmask, partial_keys, rows, vocab, mask_words, blocks_per_row
    );
    ds41rt_status_t status = status_from_cuda(cudaGetLastError());
    if (status != DS41RT_STATUS_OK) {
      return status;
    }
    masked_logits_argmax_finalize_f32_candidate_kernel<<<static_cast<unsigned int>(rows), kBlock,
                                                         0, stream>>>(
        partial_keys, out_indices, out_scores, rows, blocks_per_row
    );
  } else {
    masked_logits_topk_partial_f32_candidate_kernel<<<static_cast<unsigned int>(total_blocks),
                                                      kBlock, 0, stream>>>(
        logits, token_bitmask, partial_keys, rows, vocab, mask_words, blocks_per_row
    );
    ds41rt_status_t status = status_from_cuda(cudaGetLastError());
    if (status != DS41RT_STATUS_OK) {
      return status;
    }
    masked_logits_topk_finalize_f32_candidate_kernel<<<static_cast<unsigned int>(rows), kBlock, 0,
                                                       stream>>>(
        partial_keys, random_uniforms, out_indices, out_scores, rows, blocks_per_row, temperature,
        top_k, top_p
    );
  }
  return status_from_cuda(cudaGetLastError());
}

extern "C" ds41rt_status_t ds41rt_cuda_logits_sample_topk_topp_f32_cub_async(
    const float* logits, const float* random_uniforms, float* sorted_logits,
    uint32_t* unsorted_indices, uint32_t* sorted_indices, int* segment_offsets,
    uint32_t* out_indices, float* out_scores, void* cub_temp_storage,
    size_t cub_temp_storage_bytes, size_t rows, size_t vocab, float temperature, size_t top_k,
    float top_p, void* cuda_stream) {
  const ds41rt_status_t valid = validate_logits_sample_topk_topp_cub_args(
      logits, random_uniforms, sorted_logits, unsorted_indices, sorted_indices, segment_offsets,
      out_indices, out_scores, cub_temp_storage, cub_temp_storage_bytes, rows, vocab, temperature,
      top_k, top_p);
  if (valid != DS41RT_STATUS_OK) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const size_t logits_values = rows * vocab;
  const size_t fill_values = std::max(logits_values, rows + 1);
  const size_t fill_blocks = (fill_values - 1) / kBlock + 1;
  if (fill_blocks > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  logits_sample_topk_topp_f32_cub_fill_indices_offsets_kernel<<<
      static_cast<unsigned int>(fill_blocks), kBlock, 0, stream>>>(unsorted_indices,
                                                                   segment_offsets, rows, vocab);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return status_from_cuda(err);
  }
  err = cub::DeviceSegmentedRadixSort::SortPairsDescending(
      cub_temp_storage, cub_temp_storage_bytes, logits, sorted_logits, unsorted_indices,
      sorted_indices, static_cast<int>(logits_values), static_cast<int>(rows), segment_offsets,
      segment_offsets + 1, 0, sizeof(float) * 8, stream);
  if (err != cudaSuccess) {
    return status_from_cuda(err);
  }
  logits_sample_topk_topp_f32_cub_finalize_kernel<<<static_cast<unsigned int>(rows), 1, 0,
                                                    stream>>>(
      sorted_logits, sorted_indices, random_uniforms, out_indices, out_scores, rows, vocab,
      temperature, top_k, top_p);
  return status_from_cuda(cudaGetLastError());
}

extern "C" ds41rt_status_t ds41rt_cuda_logits_sample_topk_topp_f32_cub(
    const float* logits, const float* random_uniforms, float* sorted_logits,
    uint32_t* unsorted_indices, uint32_t* sorted_indices, int* segment_offsets,
    uint32_t* out_indices, float* out_scores, void* cub_temp_storage,
    size_t cub_temp_storage_bytes, size_t rows, size_t vocab, float temperature, size_t top_k,
    float top_p) {
  const ds41rt_status_t status = ds41rt_cuda_logits_sample_topk_topp_f32_cub_async(
      logits, random_uniforms, sorted_logits, unsorted_indices, sorted_indices, segment_offsets,
      out_indices, out_scores, cub_temp_storage, cub_temp_storage_bytes, rows, vocab, temperature,
      top_k, top_p, nullptr);
  if (status != DS41RT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}
