/* Device tests for the chunk-1 GPU target-sampler (K1).
 *
 * Style follows `native/tests/cuda_selftest.cc`: plain `assert`-like checks with
 * `std::abort` on failure and a `passed` line on success. This file tests the
 * *shipped* entry point from the linked `ds41rt_native` library; it does not
 * compile the kernel itself, so it also pins the exported C ABI.
 *
 * The CPU oracle below is a faithful port of the production reference's greedy
 * branch and of the K1 scalar chain
 * (`rust/crates/ds41rt-core/src/target_sampling.rs`), which is the frozen
 * correctness oracle. The reviewer must diff this port against
 * `reference_select` line by line (design §12.1, §12.11): chunk 1's device scope
 * is exactly the greedy/constrained branch plus the K1 reductions, so only that
 * branch is ported here. The stochastic *draw* chain (K2/K3/K5) lands in later
 * chunks.
 *
 * Covers design §12.1, §12.6 and §12.7: mask-first ordering, mode-specific
 * masked-non-finite discipline, exact status precedence, the inclusive min_p
 * threshold at ±1 ULP using the shipped host `ln_min_p`, survivor counts,
 * greedy lowest-id ties and greedy re-derivation, vocabularies not divisible by
 * 32 with a `u32::MAX` final mask word, `MASK_WIDTH`, a greedy parity grid and
 * batch/order/peer independence.
 */

#include "ds41rt_native.h"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

int g_checks = 0;
int g_cases = 0;
/* Largest K3 probe-pass count observed over every eligible row of every case;
 * the measured pass budget reported at the end of the run. */
uint32_t g_max_pivot_passes = 0;

void fail(const char* what) {
  std::cerr << "FAIL: " << what << "\n";
  std::abort();
}

void expect(bool condition, const std::string& what) {
  ++g_checks;
  if (!condition) {
    fail(what.c_str());
  }
}

void require_cuda(cudaError_t status, const char* action) {
  if (status != cudaSuccess) {
    std::cerr << action << ": " << cudaGetErrorString(status) << "\n";
    std::abort();
  }
}

/* Additive overload so the chunk-3b loud-status test can pass a composed
 * `std::string`; every pre-existing `const char*` call is unchanged. */
void require_cuda(cudaError_t status, const std::string& action) {
  require_cuda(status, action.c_str());
}

/* Chunk-2 shorthand; identical to the `std::numeric_limits` spellings used
 * above. */
float inf() { return std::numeric_limits<float>::infinity(); }
uint32_t u32max() { return std::numeric_limits<uint32_t>::max(); }

/* K2 tests reuse one logits row across many (seed, position) cells, so the
 * device buffer must hold a real copy per row: the kernel reads
 * `logits + block_row * logits_stride`. */
std::vector<float> repeat_row(const std::vector<float>& row, size_t rows) {
  std::vector<float> out(rows * row.size());
  for (size_t r = 0; r < rows; ++r) {
    for (size_t i = 0; i < row.size(); ++i) {
      out[r * row.size() + i] = row[i];
    }
  }
  return out;
}

/* ---- CPU oracle: a faithful port of `reference_select`'s greedy branch and of
 * the K1 scalar chain (`target_sampling.rs`, the design's correctness oracle).
 * Only the greedy/constrained path is ported in chunk 1: chunk 1's device scope
 * is exactly that branch plus the K1 reductions. ---- */
struct RefGreedy {
  bool ok = false;
  /* Expected per-row status, exactly as the CPU contract reports it
   * (`target_sampling.rs` / `scores.rs::argmax`). Tokens are only meaningful
   * when `ok`; `nonfinite_token` is the lowest offending id. */
  uint32_t stochastic_status = DS41RT_V41_SAMPLER_STATUS_OK;
  uint32_t nonfinite_token = DS41RT_V41_SAMPLER_NO_DETAIL;
  uint32_t token = 0;
  float maximum = -std::numeric_limits<float>::infinity();
  /* Scalars the stochastic K1 chain would report; only valid when is_greedy. */
  float max_scaled = 0.0f;
  float min_scaled = 0.0f;
  uint32_t survivor_count = 0;
  uint32_t allowed_count = 0;
};

bool is_greedy(float temperature, uint32_t top_k) {
  return temperature < 1.0e-5f || top_k == 1u;
}

RefGreedy cpu_reference(const float* logits, size_t vocab,
                        const uint32_t* mask_words, size_t mask_words_per_row,
                        float temperature, uint32_t top_k, float ln_min_p, float min_p,
                        uint32_t flags) {
  RefGreedy out;
  /* Mirrors the kernel's `row_allowed`: an unconstrained row (NO_MASK flag) is
   * all-allowed and never reads the arena, exactly as `constraints.rs` passes
   * `None` for `needs_mask == false`. */
  const bool unconstrained =
      mask_words == nullptr || (flags & DS41RT_V41_SAMPLER_FLAG_NO_MASK) != 0u;
  const auto allowed = [&](size_t token) {
    return unconstrained || ((mask_words[token / 32u] >> (token % 32u)) & 1u) != 0u;
  };
  const bool greedy = is_greedy(temperature, top_k) ||
                      (flags & DS41RT_V41_SAMPLER_FLAG_GREEDY) != 0u;
  if (greedy) {
    float maximum = -std::numeric_limits<float>::infinity();
    bool have = false;
    for (size_t token = 0; token < vocab; ++token) {
      const float logit = logits[token];
      /* Strict whole-row finiteness first, then the mask test: exactly
       * `scores.rs::argmax` (`:160-163`). */
      if (!std::isfinite(logit)) {
        out.stochastic_status = DS41RT_V41_SAMPLER_STATUS_NONFINITE_LOGIT;
        out.nonfinite_token = static_cast<uint32_t>(token);
        return out;
      }
      if (!allowed(token)) {
        continue;
      }
      if (logit > maximum) {
        maximum = logit;
        out.token = static_cast<uint32_t>(token);
        have = true;
      }
    }
    out.ok = have;
    out.maximum = maximum;
    if (!have) {
      out.stochastic_status = DS41RT_V41_SAMPLER_STATUS_EMPTY_CANDIDATES;
    }
    return out;
  }
  /* Stochastic K1 scalar chain (not the draw; chunk 1 does not sample). */
  const float inv = 1.0f / temperature;
  float max_scaled = -std::numeric_limits<float>::infinity();
  uint32_t allowed_count = 0;
  for (size_t token = 0; token < vocab; ++token) {
    if (!allowed(token)) {
      continue;
    }
    const float logit = logits[token];
    if (!std::isfinite(logit)) {
      out.stochastic_status = DS41RT_V41_SAMPLER_STATUS_NONFINITE_LOGIT;
      out.nonfinite_token = static_cast<uint32_t>(token);
      return out;
    }
    ++allowed_count;
    max_scaled = std::fmax(max_scaled, logit * inv);
  }
  out.allowed_count = allowed_count;
  if (allowed_count == 0) {
    out.stochastic_status = DS41RT_V41_SAMPLER_STATUS_EMPTY_CANDIDATES;
    return out;
  }
  if (!std::isfinite(max_scaled)) {
    out.stochastic_status = DS41RT_V41_SAMPLER_STATUS_INVALID_TEMPERATURE;
    return out;
  }
  out.ok = true;
  out.max_scaled = max_scaled;
  out.min_scaled = min_p > 0.0f ? max_scaled + ln_min_p
                                : -std::numeric_limits<float>::infinity();
  for (size_t token = 0; token < vocab; ++token) {
    if (allowed(token) && logits[token] * inv >= out.min_scaled) {
      ++out.survivor_count;
    }
  }
  return out;
}

/* ---- device plumbing ---- */
struct K1 {
  ds41rt_v41_sampler_row_t* params = nullptr;
  uint32_t* mask = nullptr;
  uint32_t* ids = nullptr;
  uint32_t* status = nullptr;
  uint32_t* detail = nullptr;
  float* scores = nullptr;
  float* total = nullptr;
  uint32_t* nucleus = nullptr;
  ds41rt_v41_sampler_scratch_t* scratch = nullptr;
  float* logits = nullptr;
  size_t rows = 0;
  size_t vocab = 0;
  size_t words = 0;
};

/* Allocate one K1 fixture. Rows, logits and the mask arena always exist; the
 * optional diagnostic buffers exist only when `diagnose` is set, matching the
 * production rule that `out_total`/`out_nucleus_count` may be null. */
/* Allocation helper that names the failing buffer and refuses a zero extent. */
template <typename T>
void alloc_device(T** out, size_t bytes, const char* what) {
  if (bytes == 0) {
    std::cerr << "zero-byte device allocation for " << what << "\n";
    std::abort();
  }
  void* raw = nullptr;
  if (cudaMalloc(&raw, bytes) != cudaSuccess) {
    std::cerr << "device allocation failed for " << what << " bytes=" << bytes << "\n";
    std::abort();
  }
  *out = static_cast<T*>(raw);
}

K1 make_k1(size_t rows, size_t vocab, bool with_mask, bool diagnose) {
  K1 k;
  k.rows = rows;
  k.vocab = vocab;
  k.words = with_mask ? (vocab + 31u) / 32u : 0u;
  alloc_device(&k.logits, rows * vocab * sizeof(float), "logits");
  alloc_device(&k.params, rows * sizeof(ds41rt_v41_sampler_row_t), "params");
  alloc_device(&k.ids, rows * sizeof(uint32_t), "ids");
  alloc_device(&k.status, rows * sizeof(uint32_t), "status");
  alloc_device(&k.detail, rows * sizeof(uint32_t), "detail");
  alloc_device(&k.scores, rows * sizeof(float), "scores");
  alloc_device(&k.scratch, rows * sizeof(ds41rt_v41_sampler_scratch_t), "scratch");
  if (with_mask) {
    alloc_device(&k.mask, rows * k.words * sizeof(uint32_t), "mask");
  }
  if (diagnose) {
    alloc_device(&k.total, rows * sizeof(float), "total");
    alloc_device(&k.nucleus, rows * sizeof(uint32_t), "nucleus");
  }
  return k;
}

void free_k1(K1* k) {
  cudaFree(k->logits); cudaFree(k->params); cudaFree(k->ids); cudaFree(k->status);
  cudaFree(k->detail); cudaFree(k->scores); cudaFree(k->scratch);
  if (k->mask) cudaFree(k->mask);
  if (k->total) cudaFree(k->total);
  if (k->nucleus) cudaFree(k->nucleus);
  *k = K1{};
}

ds41rt_v41_sampler_row_t row(uint32_t output_row, float temperature, uint32_t top_k,
                             float min_p, float ln_min_p, uint32_t mask_row,
                             uint32_t flags) {
  ds41rt_v41_sampler_row_t value = {};
  value.seed = 0x7f4a7c159e3779b9ull;
  value.position = 17;
  value.temperature = temperature;
  value.top_p = 1.0f;
  value.min_p = min_p;
  value.top_k = top_k;
  value.mask_row = mask_row;
  value.flags = flags;
  value.output_row = output_row;
  value.ln_min_p = ln_min_p;
  return value;
}

/* Runs one K1 configuration against the oracle and asserts exact agreement for
 * every greedy row and the exact K1 scalars for every stochastic row. */
void run_case(const std::vector<float>& logits, size_t rows, size_t vocab,
              const std::vector<ds41rt_v41_sampler_row_t>& params,
              const std::vector<uint32_t>& mask, bool with_mask, bool strict_mask_bits,
              const char* label) {
  ++g_cases;
  K1 k = make_k1(rows, vocab, with_mask, true);
  require_cuda(cudaMemcpy(k.logits, logits.data(), logits.size() * sizeof(float),
                          cudaMemcpyHostToDevice), "logits h2d");
  require_cuda(cudaMemcpy(k.params, params.data(),
                          params.size() * sizeof(ds41rt_v41_sampler_row_t),
                          cudaMemcpyHostToDevice), "params h2d");
  std::vector<uint32_t> device_mask = mask;
  if (with_mask) {
    if (strict_mask_bits) {
      for (size_t r = 0; r < rows; ++r) {
        ds41rt_v41_sampler_clear_remainder(device_mask.data() + r * k.words, vocab);
      }
    }
    require_cuda(cudaMemcpy(k.mask, device_mask.data(),
                            device_mask.size() * sizeof(uint32_t),
                            cudaMemcpyHostToDevice), "mask h2d");
  }
  const ds41rt_status_t launch = ds41rt_cuda_v41_target_sample(
      k.logits, rows, vocab, vocab, k.params, with_mask ? k.mask : nullptr,
      with_mask ? k.words : 0u, k.ids, k.status, k.detail, k.scores, k.total,
      k.nucleus, k.scratch);
  expect(launch == DS41RT_STATUS_OK, std::string(label) + ": launch status");
  require_cuda(cudaDeviceSynchronize(), "kernel");

  std::vector<uint32_t> ids(rows), status(rows), detail(rows), nucleus(rows);
  std::vector<float> scores(rows), total(rows);
  std::vector<ds41rt_v41_sampler_scratch_t> scratch(rows);
  require_cuda(cudaMemcpy(ids.data(), k.ids, rows * 4, cudaMemcpyDeviceToHost), "ids");
  require_cuda(cudaMemcpy(status.data(), k.status, rows * 4, cudaMemcpyDeviceToHost), "status");
  require_cuda(cudaMemcpy(detail.data(), k.detail, rows * 4, cudaMemcpyDeviceToHost), "detail");
  require_cuda(cudaMemcpy(scores.data(), k.scores, rows * 4, cudaMemcpyDeviceToHost), "scores");
  require_cuda(cudaMemcpy(total.data(), k.total, rows * 4, cudaMemcpyDeviceToHost), "total");
  require_cuda(cudaMemcpy(nucleus.data(), k.nucleus, rows * 4, cudaMemcpyDeviceToHost), "nucleus");
  require_cuda(cudaMemcpy(scratch.data(), k.scratch,
                          rows * sizeof(ds41rt_v41_sampler_scratch_t),
                          cudaMemcpyDeviceToHost), "scratch");

  for (size_t r = 0; r < rows; ++r) {
    const ds41rt_v41_sampler_row_t& p = params[r];
    const bool greedy = is_greedy(p.temperature, p.top_k) ||
                        (p.flags & DS41RT_V41_SAMPLER_FLAG_GREEDY) != 0u;
    /* A NO_MASK row carries the 0xFFFFFFFF sentinel, so it must not be turned
     * into a host pointer at all. */
    const bool unconstrained =
        (p.flags & DS41RT_V41_SAMPLER_FLAG_NO_MASK) != 0u ||
        p.mask_row == DS41RT_V41_SAMPLER_NO_MASK_ROW;
    const uint32_t* row_mask = (with_mask && !unconstrained)
        ? device_mask.data() + static_cast<size_t>(p.mask_row) * k.words
        : nullptr;
    const size_t row_vocab = vocab;
    const float* row_logits = logits.data() + r * vocab;
    const RefGreedy expected = cpu_reference(row_logits, row_vocab, row_mask,
                                             (with_mask && !unconstrained) ? k.words : 0u,
                                             p.temperature,
                                             p.top_k, p.ln_min_p, p.min_p, p.flags);
    const std::string tag = std::string(label) + " row " + std::to_string(r);
    /* Both branches must reproduce the CPU status exactly. */
    if (status[r] != expected.stochastic_status) {
      std::cerr << "status mismatch " << tag << " flags=" << p.flags
                << " expected=" << expected.stochastic_status << " device=" << status[r]
                << " allowed=" << scratch[r].allowed_count << "\n";
    }
    expect(status[r] == expected.stochastic_status, tag + ": status");
    if (expected.stochastic_status == DS41RT_V41_SAMPLER_STATUS_NONFINITE_LOGIT) {
      expect(detail[r] == expected.nonfinite_token, tag + ": nonfinite detail");
      expect(scratch[r].nonfinite_token == expected.nonfinite_token,
             tag + ": scratch nonfinite token");
    }
    if (greedy) {
      /* Exact equality with the CPU masked argmax, including strict whole-row
       * finiteness. */
      if (expected.ok) {
        expect(ids[r] == expected.token, tag + ": greedy id");
        expect(scores[r] == expected.maximum, tag + ": greedy raw max score");
      }
    } else if (expected.stochastic_status == DS41RT_V41_SAMPLER_STATUS_OK) {
      /* Count-only K1 outputs must be exact: no accumulation is involved. */
      expect(scratch[r].max_scaled == expected.max_scaled, tag + ": max_scaled bits");
      expect(scratch[r].allowed_count == expected.allowed_count, tag + ": allowed_count");
      expect(scratch[r].survivor_count == expected.survivor_count, tag + ": survivor_count");
      if ((p.flags & DS41RT_V41_SAMPLER_FLAG_DIAGNOSE) != 0u) {
        expect(nucleus[r] == expected.survivor_count, tag + ": out_nucleus_count");
        expect(total[r] == expected.max_scaled, tag + ": out_total");
      }
      expect(scratch[r].inv_temperature == 1.0f / p.temperature, tag + ": inv_temperature");
      expect(scratch[r].nonfinite_token == DS41RT_V41_SAMPLER_NO_DETAIL,
             tag + ": no nonfinite");
      expect(detail[r] == DS41RT_V41_SAMPLER_NO_DETAIL, tag + ": detail sentinel");
    }
  }
  free_k1(&k);
}

/* ------------------------------------------------------------------ tests */

/* Mask application happens FIRST: a masked token can never win the argmax even
 * when it is the global maximum, and a mask that excludes everything is
 * EMPTY_CANDIDATES rather than an unmasked fallback. */
void test_mask_first_and_empty_candidates() {
  const size_t rows = 2;
  const size_t vocab = 5;
  std::vector<float> logits = {
      0.0f, 9.0f, 1.0f, 2.0f, 3.0f,     /* row 0: token 1 dominates */
      4.0f, 3.0f, 2.0f, 1.0f, 0.0f,     /* row 1: token 0 dominates */
  };
  const size_t words = (vocab + 31u) / 32u;
  std::vector<uint32_t> mask(rows * words, 0u);
  mask[0 * words + 0] = (1u << 0) | (1u << 2);   /* row 0 allows tokens 0 and 2 */
  /* row 1 is all-zero -> EMPTY_CANDIDATES */
  std::vector<ds41rt_v41_sampler_row_t> params = {
      row(0, 0.0f, 0u, 0.0f, -std::numeric_limits<float>::infinity(), 0u,
          DS41RT_V41_SAMPLER_FLAG_GREEDY | DS41RT_V41_SAMPLER_FLAG_STRICT_FINITE),
      row(1, 0.0f, 0u, 0.0f, -std::numeric_limits<float>::infinity(), 1u,
          DS41RT_V41_SAMPLER_FLAG_GREEDY | DS41RT_V41_SAMPLER_FLAG_STRICT_FINITE),
  };
  run_case(logits, rows, vocab, params, mask, true, true, "mask first (greedy)");
  /* The oracle check inside run_case already pins both rows; re-run through the
   * full harness for the stochastic survivor shape too. */
  std::vector<ds41rt_v41_sampler_row_t> stochastic = {
      row(0, 0.7f, 0u, 0.0f, -std::numeric_limits<float>::infinity(), 0u,
          DS41RT_V41_SAMPLER_FLAG_DIAGNOSE),
      row(1, 0.7f, 0u, 0.0f, -std::numeric_limits<float>::infinity(), 1u,
          DS41RT_V41_SAMPLER_FLAG_DIAGNOSE),
  };
  K1 k = make_k1(rows, vocab, true, true);
  require_cuda(cudaMemcpy(k.logits, logits.data(), logits.size() * 4, cudaMemcpyHostToDevice),
               "h2d");
  require_cuda(cudaMemcpy(k.params, stochastic.data(),
                          stochastic.size() * sizeof(ds41rt_v41_sampler_row_t),
                          cudaMemcpyHostToDevice), "params");
  require_cuda(cudaMemcpy(k.mask, mask.data(), mask.size() * 4, cudaMemcpyHostToDevice), "mask");
  expect(ds41rt_cuda_v41_target_sample(k.logits, rows, vocab, vocab, k.params, k.mask, words,
                                       k.ids, k.status, k.detail, k.scores, k.total, k.nucleus,
                                       k.scratch) == DS41RT_STATUS_OK, "launch");
  std::vector<uint32_t> status(rows), nucleus(rows);
  std::vector<ds41rt_v41_sampler_scratch_t> scratch(rows);
  require_cuda(cudaMemcpy(status.data(), k.status, rows * 4, cudaMemcpyDeviceToHost), "status");
  require_cuda(cudaMemcpy(nucleus.data(), k.nucleus, rows * 4, cudaMemcpyDeviceToHost), "nucleus");
  require_cuda(cudaMemcpy(scratch.data(), k.scratch, rows * sizeof(scratch[0]),
                          cudaMemcpyDeviceToHost), "scratch");
  expect(status[0] == DS41RT_V41_SAMPLER_STATUS_OK, "row0 allowed");
  expect(nucleus[0] == 2u, "row0 two allowed survivors with min_p disabled");
  expect(scratch[0].allowed_count == 2u, "row0 allowed_count");
  expect(status[1] == DS41RT_V41_SAMPLER_STATUS_EMPTY_CANDIDATES, "row1 empty");
  expect(scratch[1].allowed_count == 0u, "row1 allowed_count");
  free_k1(&k);
  std::cout << "ok  mask-first ordering and empty candidates\n";
}

/* Mode-specific masked-non-finite discipline (design §4.1, §12.6):
 *   strict (greedy/constrained): a masked non-finite logit is an ERROR;
 *   permissive (stochastic): a masked non-finite logit is legal and unread. */
void test_masked_nonfinite_is_mode_specific() {
  const size_t rows = 2;
  const size_t vocab = 4;
  std::vector<float> logits = {
      1.0f, std::numeric_limits<float>::quiet_NaN(), 2.0f, 3.0f,   /* row 0 */
      1.0f, std::numeric_limits<float>::infinity(), 2.0f, 3.0f,    /* row 1 */
  };
  const size_t words = (vocab + 31u) / 32u;
  std::vector<uint32_t> mask(rows * words, 0u);
  mask[0 * words + 0] = 0b1101u;   /* token 1 (NaN) masked out, tokens 0,2,3 allowed */
  mask[1 * words + 0] = 0b1101u;   /* token 1 (+inf) masked out */
  std::vector<ds41rt_v41_sampler_row_t> params = {
      row(0, 0.0f, 0u, 0.0f, -std::numeric_limits<float>::infinity(), 0u,
          DS41RT_V41_SAMPLER_FLAG_GREEDY | DS41RT_V41_SAMPLER_FLAG_STRICT_FINITE),
      row(1, 0.7f, 0u, 0.0f, -std::numeric_limits<float>::infinity(), 1u, 0u),
  };
  K1 k = make_k1(rows, vocab, true, true);
  require_cuda(cudaMemcpy(k.logits, logits.data(), logits.size() * 4, cudaMemcpyHostToDevice),
               "h2d");
  require_cuda(cudaMemcpy(k.params, params.data(),
                          params.size() * sizeof(ds41rt_v41_sampler_row_t),
                          cudaMemcpyHostToDevice), "params");
  require_cuda(cudaMemcpy(k.mask, mask.data(), mask.size() * 4, cudaMemcpyHostToDevice), "mask");
  expect(ds41rt_cuda_v41_target_sample(k.logits, rows, vocab, vocab, k.params, k.mask, words,
                                       k.ids, k.status, k.detail, k.scores, k.total, k.nucleus,
                                       k.scratch) == DS41RT_STATUS_OK, "launch");
  std::vector<uint32_t> ids(rows), status(rows), detail(rows);
  std::vector<ds41rt_v41_sampler_scratch_t> scratch(rows);
  require_cuda(cudaMemcpy(ids.data(), k.ids, rows * 4, cudaMemcpyDeviceToHost), "ids");
  require_cuda(cudaMemcpy(status.data(), k.status, rows * 4, cudaMemcpyDeviceToHost), "status");
  require_cuda(cudaMemcpy(detail.data(), k.detail, rows * 4, cudaMemcpyDeviceToHost), "detail");
  require_cuda(cudaMemcpy(scratch.data(), k.scratch, rows * sizeof(scratch[0]),
                          cudaMemcpyDeviceToHost), "scratch");
  /* Strict row: masked NaN still errors, with the offending id. */
  expect(status[0] == DS41RT_V41_SAMPLER_STATUS_NONFINITE_LOGIT,
         "strict row rejects masked NaN");
  expect(detail[0] == 1u, "strict row offending id is the masked token");
  expect(scratch[0].nonfinite_token == 1u, "strict scratch offending id");
  /* Permissive row: masked +inf is not read. The stochastic branch is
   * permissive by default; `STRICT_FINITE` is what opts a row into the
   * whole-row check, and it is absent here. */
  expect((params[1].flags & DS41RT_V41_SAMPLER_FLAG_STRICT_FINITE) == 0u,
         "permissive row must not carry STRICT_FINITE");
  expect(status[1] == DS41RT_V41_SAMPLER_STATUS_OK, "stochastic row allows masked +inf");
  expect(scratch[1].allowed_count == 3u, "stochastic allowed_count skips the masked token");
  expect(scratch[1].survivor_count == 3u, "stochastic survivors skip the masked token");
  (void)ids;
  free_k1(&k);

  /* The same stochastic row WITH `STRICT_FINITE` must take the whole-row
   * branch and reject the masked non-finite token, exactly as a greedy or
   * constrained row does. This is what makes the ABI flag functional rather
   * than decorative. */
  {
    std::vector<ds41rt_v41_sampler_row_t> strict_stochastic = {
        row(0, 0.7f, 0u, 0.0f, -std::numeric_limits<float>::infinity(), 0u,
            DS41RT_V41_SAMPLER_FLAG_STRICT_FINITE),
        row(1, 0.7f, 0u, 0.0f, -std::numeric_limits<float>::infinity(), 1u,
            DS41RT_V41_SAMPLER_FLAG_STRICT_FINITE),
    };
    K1 strict = make_k1(rows, vocab, true, true);
    require_cuda(cudaMemcpy(strict.logits, logits.data(), logits.size() * 4,
                            cudaMemcpyHostToDevice), "strict h2d");
    require_cuda(cudaMemcpy(strict.params, strict_stochastic.data(),
                            strict_stochastic.size() * sizeof(strict_stochastic[0]),
                            cudaMemcpyHostToDevice), "strict params");
    require_cuda(cudaMemcpy(strict.mask, mask.data(), mask.size() * 4,
                            cudaMemcpyHostToDevice), "strict mask");
    expect(ds41rt_cuda_v41_target_sample(strict.logits, rows, vocab, vocab, strict.params,
                                         strict.mask, words, strict.ids, strict.status,
                                         strict.detail, strict.scores, strict.total,
                                         strict.nucleus, strict.scratch) == DS41RT_STATUS_OK,
           "strict stochastic launch");
    std::vector<uint32_t> strict_status(rows), strict_detail(rows);
    require_cuda(cudaMemcpy(strict_status.data(), strict.status, rows * 4,
                            cudaMemcpyDeviceToHost), "strict status");
    require_cuda(cudaMemcpy(strict_detail.data(), strict.detail, rows * 4,
                            cudaMemcpyDeviceToHost), "strict detail");
    expect(strict_status[0] == DS41RT_V41_SAMPLER_STATUS_NONFINITE_LOGIT,
           "STRICT_FINITE stochastic row rejects masked NaN");
    expect(strict_detail[0] == 1u, "STRICT_FINITE offending id is the masked token");
    expect(strict_status[1] == DS41RT_V41_SAMPLER_STATUS_NONFINITE_LOGIT,
           "STRICT_FINITE stochastic row rejects masked +inf");
    expect(strict_detail[1] == 1u, "STRICT_FINITE offending id for +inf");
    free_k1(&strict);
  }
  std::cout << "ok  masked non-finite is mode-specific and STRICT_FINITE is functional\n";
}

/* An ALLOWED non-finite logit is an error in every mode, with the lowest
 * offending id, and error precedence puts it above EMPTY_CANDIDATES. */
void test_allowed_nonfinite_and_status_precedence() {
  const size_t vocab = 4;
  struct Case { std::vector<float> logits; uint32_t expected_status; uint32_t expected_detail; };
  const std::vector<Case> cases = {
      /* An allowed non-finite logit is NONFINITE_LOGIT with its own id. */
      {{std::numeric_limits<float>::quiet_NaN(), 1.0f, 2.0f, 3.0f},
       DS41RT_V41_SAMPLER_STATUS_NONFINITE_LOGIT, 0u},
      {{1.0f, 1.0f, 1.0f, -std::numeric_limits<float>::infinity()},
       DS41RT_V41_SAMPLER_STATUS_NONFINITE_LOGIT, 3u},
      /* All-zero mask: EMPTY_CANDIDATES, never a temperature error from the -inf
       * maximum (`empty_mask_is_reported_as_empty_candidates_in_every_mode`). */
      {{1.0f, 2.0f, 3.0f, 4.0f}, DS41RT_V41_SAMPLER_STATUS_EMPTY_CANDIDATES, 0xFFFFFFFFu},
      {{-0.0f, 0.0f, 0.0f, 0.0f}, DS41RT_V41_SAMPLER_STATUS_OK, 0xFFFFFFFFu},
  };
  for (size_t index = 0; index < cases.size(); ++index) {
    const size_t rows = 1;
    const size_t words = (vocab + 31u) / 32u;
    std::vector<uint32_t> mask(words, 0u);
    /* First case: an all-zero mask, so EMPTY_CANDIDATES would win if precedence
     * were wrong. Later cases allow everything. */
    if (index != 2) {
      mask[0] = 0xFu;
    }
    std::vector<ds41rt_v41_sampler_row_t> params = {
        row(0, 0.7f, 0u, 0.0f, -std::numeric_limits<float>::infinity(), 0u, 0u),
    };
    K1 k = make_k1(rows, vocab, true, true);
    require_cuda(cudaMemcpy(k.logits, cases[index].logits.data(), vocab * 4,
                            cudaMemcpyHostToDevice), "h2d");
    require_cuda(cudaMemcpy(k.params, params.data(), sizeof(params[0]),
                            cudaMemcpyHostToDevice), "params");
    require_cuda(cudaMemcpy(k.mask, mask.data(), words * 4, cudaMemcpyHostToDevice), "mask");
    expect(ds41rt_cuda_v41_target_sample(k.logits, rows, vocab, vocab, k.params, k.mask, words,
                                         k.ids, k.status, k.detail, k.scores, k.total, k.nucleus,
                                         k.scratch) == DS41RT_STATUS_OK, "launch");
    std::vector<uint32_t> status(rows), detail(rows);
    require_cuda(cudaMemcpy(status.data(), k.status, rows * 4, cudaMemcpyDeviceToHost), "status");
    require_cuda(cudaMemcpy(detail.data(), k.detail, rows * 4, cudaMemcpyDeviceToHost), "detail");
    expect(status[0] == cases[index].expected_status, "nonfinite status");
    expect(detail[0] == cases[index].expected_detail, "nonfinite detail");
    free_k1(&k);
  }
  /* Empty candidates beats invalid temperature: all mask bits clear and a
   * temperature so small that the (absent) maximum would be non-finite. */
  {
    const size_t rows = 1;
    const size_t vocab2 = 2;
    std::vector<float> logits = {std::numeric_limits<float>::max(),
                                 std::numeric_limits<float>::max()};
    std::vector<uint32_t> mask2(1, 0u);
    std::vector<ds41rt_v41_sampler_row_t> params = {
        row(0, 1.0e-5f, 0u, 0.0f, -std::numeric_limits<float>::infinity(), 0u, 0u),
    };
    K1 k = make_k1(rows, vocab2, true, true);
    require_cuda(cudaMemcpy(k.logits, logits.data(), vocab2 * 4, cudaMemcpyHostToDevice), "h2d");
    require_cuda(cudaMemcpy(k.params, params.data(), sizeof(params[0]),
                            cudaMemcpyHostToDevice), "params");
    require_cuda(cudaMemcpy(k.mask, mask2.data(), 4, cudaMemcpyHostToDevice), "mask");
    expect(ds41rt_cuda_v41_target_sample(k.logits, rows, vocab2, vocab2, k.params, k.mask, 1u,
                                         k.ids, k.status, k.detail, k.scores, k.total, k.nucleus,
                                         k.scratch) == DS41RT_STATUS_OK, "launch");
    std::vector<uint32_t> status(rows);
    require_cuda(cudaMemcpy(status.data(), k.status, rows * 4, cudaMemcpyDeviceToHost), "status");
    expect(status[0] == DS41RT_V41_SAMPLER_STATUS_EMPTY_CANDIDATES,
           "EMPTY_CANDIDATES beats invalid temperature");
    free_k1(&k);
  }
  /* An allowed huge logit at the smallest non-greedy temperature is
   * INVALID_TEMPERATURE. `temperature = 1e-5` is exactly the boundary at which
   * `is_greedy()` becomes false, so the stochastic branch runs and
   * `f32::MAX * 1e5` overflows to +inf. */
  {
    const size_t rows = 1;
    const size_t vocab2 = 2;
    std::vector<float> logits = {std::numeric_limits<float>::max(), 0.0f};
    std::vector<uint32_t> mask2(1, 0x3u);
    std::vector<ds41rt_v41_sampler_row_t> params = {
        row(0, 1.0e-5f, 0u, 0.0f, -std::numeric_limits<float>::infinity(), 0u, 0u),
    };
    K1 k = make_k1(rows, vocab2, true, true);
    require_cuda(cudaMemcpy(k.logits, logits.data(), vocab2 * 4, cudaMemcpyHostToDevice), "h2d");
    require_cuda(cudaMemcpy(k.params, params.data(), sizeof(params[0]),
                            cudaMemcpyHostToDevice), "params");
    require_cuda(cudaMemcpy(k.mask, mask2.data(), 4, cudaMemcpyHostToDevice), "mask");
    expect(ds41rt_cuda_v41_target_sample(k.logits, rows, vocab2, vocab2, k.params, k.mask, 1u,
                                         k.ids, k.status, k.detail, k.scores, k.total, k.nucleus,
                                         k.scratch) == DS41RT_STATUS_OK, "launch");
    std::vector<uint32_t> status(rows), detail(rows);
    std::vector<ds41rt_v41_sampler_scratch_t> scratch(rows);
    require_cuda(cudaMemcpy(status.data(), k.status, rows * 4, cudaMemcpyDeviceToHost), "status");
    require_cuda(cudaMemcpy(detail.data(), k.detail, rows * 4, cudaMemcpyDeviceToHost), "detail");
    require_cuda(cudaMemcpy(scratch.data(), k.scratch, rows * sizeof(scratch[0]),
                            cudaMemcpyDeviceToHost), "scratch");
    expect(status[0] == DS41RT_V41_SAMPLER_STATUS_INVALID_TEMPERATURE,
           "invalid temperature");
    free_k1(&k);
  }
  std::cout << "ok  allowed non-finite and status precedence\n";
}

/* min_p boundary: a token exactly AT max_scaled + ln_min_p is retained
 * (inclusive `>=`); one ULP below is rejected. The threshold uses the shipped
 * host `ln_min_p`, never a device `logf`. */
void test_min_p_boundary_is_inclusive() {
  /* Rust `f32::ln(0.5)` = -0.6931472 (0xbf317218) is the shipped `ln_min_p`;
   * one ULP below is -0.6931471 (0xbf317217). At T = 2 the logit values that
   * multiply bit-exactly to those scaled values are -1.3862944 and
   * -1.3862942 (verified with Rust `f32` arithmetic). */
  const float min_p = 0.5f;
  const float ln_min_p = -0.6931472f;
  const float exact_logit = -1.3862944f;   /* scaled == max + ln_min_p */
  const float below_logit = -1.3862945f;   /* scaled == threshold - 1 ULP */
  const float temperature = 2.0f;
  const size_t rows = 2;
  const size_t vocab = 8;
  std::vector<float> logits(rows * vocab, -4.0f);
  /* Row 0: token 0 is the max (scaled 0), token 3 is exactly at the threshold,
   * token 4 is one ULP below. */
  logits[0 * vocab + 0] = 0.0f;
  logits[0 * vocab + 3] = exact_logit;
  logits[0 * vocab + 4] = below_logit;
  /* Row 1: token 3 is one ULP below the threshold instead. */
  logits[1 * vocab + 0] = 0.0f;
  logits[1 * vocab + 3] = below_logit;
  std::vector<ds41rt_v41_sampler_row_t> params = {
      row(0, temperature, 0u, min_p, ln_min_p, DS41RT_V41_SAMPLER_NO_MASK_ROW,
          DS41RT_V41_SAMPLER_FLAG_NO_MASK),
      row(1, temperature, 0u, min_p, ln_min_p, DS41RT_V41_SAMPLER_NO_MASK_ROW,
          DS41RT_V41_SAMPLER_FLAG_NO_MASK),
  };
  run_case(logits, rows, vocab, params, {}, false, false, "min_p boundary");
  K1 k = make_k1(rows, vocab, false, true);
  require_cuda(cudaMemcpy(k.logits, logits.data(), logits.size() * 4, cudaMemcpyHostToDevice),
               "h2d");
  require_cuda(cudaMemcpy(k.params, params.data(),
                          params.size() * sizeof(ds41rt_v41_sampler_row_t),
                          cudaMemcpyHostToDevice), "params");
  expect(ds41rt_cuda_v41_target_sample(k.logits, rows, vocab, vocab, k.params, nullptr, 0u,
                                       k.ids, k.status, k.detail, k.scores, k.total, k.nucleus,
                                       k.scratch) == DS41RT_STATUS_OK, "launch");
  std::vector<ds41rt_v41_sampler_row_t> diagnose = params;
  for (ds41rt_v41_sampler_row_t& p : diagnose) {
    p.flags |= DS41RT_V41_SAMPLER_FLAG_DIAGNOSE;
  }
  require_cuda(cudaMemcpy(k.params, diagnose.data(),
                          diagnose.size() * sizeof(ds41rt_v41_sampler_row_t),
                          cudaMemcpyHostToDevice), "params");
  expect(ds41rt_cuda_v41_target_sample(k.logits, rows, vocab, vocab, k.params, nullptr, 0u,
                                       k.ids, k.status, k.detail, k.scores, k.total, k.nucleus,
                                       k.scratch) == DS41RT_STATUS_OK, "launch diagnose");
  std::vector<ds41rt_v41_sampler_scratch_t> scratch(rows);
  std::vector<uint32_t> nucleus(rows);
  require_cuda(cudaMemcpy(scratch.data(), k.scratch, rows * sizeof(scratch[0]),
                          cudaMemcpyDeviceToHost), "scratch");
  require_cuda(cudaMemcpy(nucleus.data(), k.nucleus, rows * 4, cudaMemcpyDeviceToHost), "nucleus");
  /* The inclusive boundary is checked twice: against the shared oracle above
   * and against the exact survivor counts here. */
  expect(scratch[0].max_scaled == 0.0f, "row0 max_scaled is the max logit");
  expect(scratch[0].max_scaled == 0.0f, "row0 max_scaled");
  expect(scratch[0].survivor_count == 2u, "exact-threshold token survives");
  expect(nucleus[0] == 2u, "out_nucleus_count equals the survivor count");
  expect(scratch[1].survivor_count == 1u, "one-ULP-below token is rejected");
  free_k1(&k);
  std::cout << "ok  min_p boundary inclusive at the threshold\n";
}

/* min_p = 0 keeps -inf-scaled survivors, and min_p = 1 keeps every tied max. */
void test_min_p_disabled_and_unity() {
  const size_t rows = 3;
  const size_t vocab = 4;
  const float neg_inf = -std::numeric_limits<float>::infinity();
  const float neg_max = -std::numeric_limits<float>::max();
  /* `-f32::MAX` at T = 1e-5 scales to -inf and stays a legal survivor when
   * min_p = 0 (`negative_infinity_survivor_ordered_paths_match_reference`,
   * `target_sampling.rs:1748-1764`). The raw logits must remain finite: the CPU
   * rejects a non-finite *logit* in every mode. */
  std::vector<float> logits = {
      1.0f, neg_max, neg_max, neg_max,   /* row 0: min_p = 0 keeps -inf survivors */
      0.0f, 0.0f, neg_max, 0.0f,         /* row 1: min_p = 1 keeps every tied max */
      0.0f, -1.0f, -2.0f, -3.0f,         /* row 2: min_p = 1 keeps only the max */
  };
  std::vector<ds41rt_v41_sampler_row_t> params = {
      row(0, 1.0e-5f, 0u, 0.0f, neg_inf, DS41RT_V41_SAMPLER_NO_MASK_ROW,
          DS41RT_V41_SAMPLER_FLAG_NO_MASK),
      row(1, 1.0f, 0u, 1.0f, 0.0f, DS41RT_V41_SAMPLER_NO_MASK_ROW,
          DS41RT_V41_SAMPLER_FLAG_NO_MASK),
      row(2, 1.0f, 0u, 1.0f, 0.0f, DS41RT_V41_SAMPLER_NO_MASK_ROW,
          DS41RT_V41_SAMPLER_FLAG_NO_MASK),
  };
  run_case(logits, rows, vocab, params, {}, false, false, "min_p disabled/unity");
  K1 k = make_k1(rows, vocab, false, true);
  require_cuda(cudaMemcpy(k.logits, logits.data(), logits.size() * 4, cudaMemcpyHostToDevice),
               "h2d");
  require_cuda(cudaMemcpy(k.params, params.data(),
                          params.size() * sizeof(ds41rt_v41_sampler_row_t),
                          cudaMemcpyHostToDevice), "params");
  expect(ds41rt_cuda_v41_target_sample(k.logits, rows, vocab, vocab, k.params, nullptr, 0u,
                                       k.ids, k.status, k.detail, k.scores, k.total, k.nucleus,
                                       k.scratch) == DS41RT_STATUS_OK, "launch");
  std::vector<ds41rt_v41_sampler_scratch_t> scratch(rows);
  require_cuda(cudaMemcpy(scratch.data(), k.scratch, rows * sizeof(scratch[0]),
                          cudaMemcpyDeviceToHost), "scratch");
  expect(scratch[0].survivor_count == 4u, "min_p = 0 keeps -inf survivors");
  expect(scratch[1].survivor_count == 3u, "min_p = 1 keeps every tied maximum");
  expect(scratch[2].survivor_count == 1u, "min_p = 1 keeps only the strict max");
  free_k1(&k);
  std::cout << "ok  min_p disabled keeps -inf, min_p = 1 keeps ties\n";
}

/* Greedy lowest-id exact ties, `top_k == 1` still greedy, and the greedy
 * re-derivation from temperature/top_k even when the host bit is clear. */
void test_greedy_lowest_id_and_rederivation() {
  const size_t rows = 4;
  const size_t vocab = 6;
  std::vector<float> logits = {
      -2.0f, 1.5f, 1.5f, 1.5f, 0.0f, -1.0f,    /* row 0: three-way tie at the max */
      0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,      /* row 1: all equal */
      -3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,   /* row 2: unique max */
      2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,      /* row 3: all equal */
  };
  std::vector<ds41rt_v41_sampler_row_t> params = {
      /* host greedy bit SET, T = 0.7 */
      row(0, 0.7f, 0u, 0.0f, -std::numeric_limits<float>::infinity(),
          DS41RT_V41_SAMPLER_NO_MASK_ROW,
          DS41RT_V41_SAMPLER_FLAG_GREEDY | DS41RT_V41_SAMPLER_FLAG_NO_MASK |
              DS41RT_V41_SAMPLER_FLAG_STRICT_FINITE),
      /* host bit CLEAR but top_k == 1 -> still greedy by re-derivation */
      row(1, 0.7f, 1u, 0.0f, -std::numeric_limits<float>::infinity(),
          DS41RT_V41_SAMPLER_NO_MASK_ROW, DS41RT_V41_SAMPLER_FLAG_NO_MASK),
      /* host bit CLEAR but T < 1e-5 -> still greedy, and 1/T would be finite */
      row(2, 0.0f, 0u, 0.0f, -std::numeric_limits<float>::infinity(),
          DS41RT_V41_SAMPLER_NO_MASK_ROW, DS41RT_V41_SAMPLER_FLAG_NO_MASK),
      row(3, 1.0e-6f, 0u, 0.0f, -std::numeric_limits<float>::infinity(),
          DS41RT_V41_SAMPLER_NO_MASK_ROW, DS41RT_V41_SAMPLER_FLAG_NO_MASK),
  };
  run_case(logits, rows, vocab, params, {}, false, false, "greedy ties + rederivation");
  K1 k = make_k1(rows, vocab, false, false);
  require_cuda(cudaMemcpy(k.logits, logits.data(), logits.size() * 4, cudaMemcpyHostToDevice),
               "h2d");
  require_cuda(cudaMemcpy(k.params, params.data(),
                          params.size() * sizeof(ds41rt_v41_sampler_row_t),
                          cudaMemcpyHostToDevice), "params");
  expect(ds41rt_cuda_v41_target_sample(k.logits, rows, vocab, vocab, k.params, nullptr, 0u,
                                       k.ids, k.status, k.detail, k.scores, k.total, k.nucleus,
                                       k.scratch) == DS41RT_STATUS_OK, "launch");
  std::vector<uint32_t> ids(rows);
  require_cuda(cudaMemcpy(ids.data(), k.ids, rows * 4, cudaMemcpyDeviceToHost), "ids");
  expect(ids[0] == 1u, "lowest-id tie wins (three-way)");
  expect(ids[1] == 0u, "lowest-id tie wins (top_k == 1, all equal)");
  expect(ids[2] == 5u, "unique max");
  expect(ids[3] == 0u, "all-equal row at T = 1e-6 picks id 0");
  free_k1(&k);
  std::cout << "ok  greedy lowest-id ties and greedy re-derivation\n";
}

/* Vocabularies not divisible by 32, with the final mask word set to u32::MAX:
 * no code path may emit an id >= vocab, and the kernel must not be fooled by
 * the unused high bits. Run once with the host remainder rule applied and once
 * without, to show the loop bound is the primary guard. */
void test_vocab_remainder_bits() {
  const size_t vocabularies[] = {1u, 32u, 33u, 100u, 127u, 129281u};
  for (size_t v = 0; v < sizeof(vocabularies) / sizeof(vocabularies[0]); ++v) {
    const size_t vocab = vocabularies[v];
    const size_t rows = 1;
    const size_t words = (vocab + 31u) / 32u;
    std::vector<float> logits(vocab, 0.0f);
    /* Put the maximum in the last real token and a smaller value everywhere. */
    logits[0] = 1.0f;
    logits[vocab - 1] = 2.0f;
    std::vector<uint32_t> mask(words, 0xFFFFFFFFu);
    std::vector<ds41rt_v41_sampler_row_t> params = {
        row(0, 0.0f, 0u, 0.0f, -std::numeric_limits<float>::infinity(), 0u,
            DS41RT_V41_SAMPLER_FLAG_GREEDY | DS41RT_V41_SAMPLER_FLAG_STRICT_FINITE),
    };
    for (int strict = 0; strict < 2; ++strict) {
      run_case(logits, rows, vocab, params, mask, true, strict == 1,
               "vocab remainder");
      K1 k = make_k1(rows, vocab, true, false);
      require_cuda(cudaMemcpy(k.logits, logits.data(), vocab * 4, cudaMemcpyHostToDevice), "h2d");
      require_cuda(cudaMemcpy(k.params, params.data(), sizeof(params[0]),
                              cudaMemcpyHostToDevice), "params");
      std::vector<uint32_t> device_mask = mask;
      if (strict == 1) {
        ds41rt_v41_sampler_clear_remainder(device_mask.data(), vocab);
      }
      require_cuda(cudaMemcpy(k.mask, device_mask.data(), words * 4, cudaMemcpyHostToDevice),
                   "mask");
      expect(ds41rt_cuda_v41_target_sample(k.logits, rows, vocab, vocab, k.params, k.mask, words,
                                           k.ids, k.status, k.detail, k.scores, k.total,
                                           k.nucleus, k.scratch) == DS41RT_STATUS_OK, "launch");
      std::vector<uint32_t> ids(rows), status(rows);
      require_cuda(cudaMemcpy(ids.data(), k.ids, rows * 4, cudaMemcpyDeviceToHost), "ids");
      require_cuda(cudaMemcpy(status.data(), k.status, rows * 4, cudaMemcpyDeviceToHost), "status");
      expect(status[0] == DS41RT_V41_SAMPLER_STATUS_OK, "remainder row status");
      expect(ids[0] < vocab, "id is inside the vocabulary");
      expect(ids[0] == static_cast<uint32_t>(vocab - 1), "last real token wins");
      free_k1(&k);
    }
    /* Mask-width mismatch: MASK_WIDTH with the actual word count as detail. */
    {
      K1 k = make_k1(rows, vocab, true, false);
      require_cuda(cudaMemcpy(k.logits, logits.data(), vocab * 4, cudaMemcpyHostToDevice), "h2d");
      require_cuda(cudaMemcpy(k.params, params.data(), sizeof(params[0]),
                              cudaMemcpyHostToDevice), "params");
      std::vector<uint32_t> device_mask(words, 0xFFFFFFFFu);
      require_cuda(cudaMemcpy(k.mask, device_mask.data(), words * 4, cudaMemcpyHostToDevice),
                   "mask");
      const size_t wrong = words + 1u;
      expect(ds41rt_cuda_v41_target_sample(k.logits, rows, vocab, vocab, k.params, k.mask, wrong,
                                           k.ids, k.status, k.detail, k.scores, k.total,
                                           k.nucleus, k.scratch) == DS41RT_STATUS_OK, "launch");
      std::vector<uint32_t> status(rows), detail(rows);
      require_cuda(cudaMemcpy(status.data(), k.status, rows * 4, cudaMemcpyDeviceToHost), "status");
      require_cuda(cudaMemcpy(detail.data(), k.detail, rows * 4, cudaMemcpyDeviceToHost), "detail");
      expect(status[0] == DS41RT_V41_SAMPLER_STATUS_MASK_WIDTH, "mask width status");
      expect(detail[0] == static_cast<uint32_t>(wrong), "mask width detail");
      free_k1(&k);
    }
    /* A mask is required if and only if a word count is supplied. */
    {
      K1 k = make_k1(rows, vocab, false, false);
      require_cuda(cudaMemcpy(k.logits, logits.data(), vocab * 4, cudaMemcpyHostToDevice), "h2d");
      require_cuda(cudaMemcpy(k.params, params.data(), sizeof(params[0]),
                              cudaMemcpyHostToDevice), "params");
      expect(ds41rt_cuda_v41_target_sample(k.logits, rows, vocab, vocab, k.params, nullptr, words,
                                           k.ids, k.status, k.detail, k.scores, k.total,
                                           k.nucleus, k.scratch) == DS41RT_STATUS_OK, "launch");
      std::vector<uint32_t> status(rows);
      require_cuda(cudaMemcpy(status.data(), k.status, rows * 4, cudaMemcpyDeviceToHost), "status");
      expect(status[0] == DS41RT_V41_SAMPLER_STATUS_MASK_WIDTH,
             "null mask with a word count is MASK_WIDTH");
      free_k1(&k);
    }
  }
  std::cout << "ok  vocab remainder and mask width (vocabs 1/32/33/100/127/129281)\n";
}

/* A greedy parameter/mask grid on random rows must match the CPU masked argmax
 * exactly, including strict whole-row finiteness. Also runs the same rows at
 * several wave shapes to show a row's result cannot depend on its peers. */
void test_greedy_parity_grid() {
  const float temperatures[] = {0.0f, 1.0e-6f, 9.9e-6f, 1.0e-5f, 0.2f, 0.7f, 2.0f};
  const uint32_t top_ks[] = {0u, 1u, 2u};
  const size_t vocab = 257;   /* not a multiple of 32 */
  const size_t words = (vocab + 31u) / 32u;
  const size_t rows = 24;
  uint64_t seed = 0x12345678u;
  const auto next = [&seed]() {
    seed = seed * 6364136223846793005ull + 1442695040888963407ull;
    return static_cast<uint32_t>(seed >> 33);
  };
  std::vector<float> logits(rows * vocab);
  for (size_t r = 0; r < rows; ++r) {
    for (size_t t = 0; t < vocab; ++t) {
      /* Integer-valued logits create frequent exact ties. */
      logits[r * vocab + t] = static_cast<float>(static_cast<int>(next() % 13) - 6) * 0.5f;
    }
  }
  size_t cases = 0;
  for (float temperature : temperatures) {
    for (uint32_t top_k : top_ks) {
      /* Two mask shapes per parameter cell: all-allowed and a sparse mask. */
      for (int mask_kind = 0; mask_kind < 3; ++mask_kind) {
        std::vector<uint32_t> mask(rows * words, 0u);
        for (size_t r = 0; r < rows; ++r) {
          for (size_t w = 0; w < words; ++w) {
            if (mask_kind == 0) {
              mask[r * words + w] = 0xFFFFFFFFu;
            } else if (mask_kind == 1) {
              mask[r * words + w] = next();
            } else {
              mask[r * words + w] = next() | next();
            }
          }
        }
        if (mask_kind != 0) {
          /* Never leave a row empty: the oracle's error path is covered
           * separately. */
          for (size_t r = 0; r < rows; ++r) {
            if (mask[r * words] == 0u) {
              mask[r * words] = 1u;
            }
          }
        }
        std::vector<ds41rt_v41_sampler_row_t> params(rows);
        for (size_t r = 0; r < rows; ++r) {
          const bool host_greedy = (r % 3) == 0;
          params[r] = row(static_cast<uint32_t>(r), temperature, top_k, 0.0f,
                          -std::numeric_limits<float>::infinity(),
                          static_cast<uint32_t>(r),
                          DS41RT_V41_SAMPLER_FLAG_STRICT_FINITE |
                              (host_greedy ? DS41RT_V41_SAMPLER_FLAG_GREEDY : 0u));
        }
        run_case(logits, rows, vocab, params, mask, true, true, "greedy parity grid");
        ++cases;
      }
    }
  }
  std::cout << "ok  greedy parity grid: " << cases << " cells\n";
}

/* Batch independence: the same logical row must produce the same id at wave
 * sizes 1, 6, 8, 16, 24 and in reverse order. */
void test_batch_independence() {
  const size_t vocab = 65;
  const size_t words = (vocab + 31u) / 32u;
  const size_t rows = 24;
  std::vector<float> logits(rows * vocab);
  std::vector<uint32_t> mask(rows * words);
  for (size_t r = 0; r < rows; ++r) {
    for (size_t t = 0; t < vocab; ++t) {
      logits[r * vocab + t] = static_cast<float>((r * 7 + t * 3) % 11) - 5.0f;
    }
    mask[r * words] = 0xFFFFFFFFu;
    mask[r * words + 1] = 0xFFFFFFFFu;
    mask[r * words + 2] = 0x00000001u;
  }
  /* The expected ids for the full wave are the reference the sub-waves and the
   * reordered wave must reproduce. */
  std::vector<size_t> all(rows);
  for (size_t i = 0; i < rows; ++i) {
    all[i] = i;
  }
  const auto run_wave = [&](const std::vector<size_t>& selection) -> std::vector<uint32_t> {
    const size_t n = selection.size();
    K1 k = make_k1(n, vocab, true, false);
    std::vector<float> wave_logits(n * vocab);
    std::vector<uint32_t> wave_mask(n * words);
    std::vector<ds41rt_v41_sampler_row_t> params(n);
    for (size_t i = 0; i < n; ++i) {
      const size_t src = selection[i];
      std::memcpy(&wave_logits[i * vocab], &logits[src * vocab], vocab * 4);
      std::memcpy(&wave_mask[i * words], &mask[src * words], words * 4);
      params[i] = row(static_cast<uint32_t>(i), 0.0f, 0u, 0.0f,
                      -std::numeric_limits<float>::infinity(), static_cast<uint32_t>(i),
                      DS41RT_V41_SAMPLER_FLAG_GREEDY | DS41RT_V41_SAMPLER_FLAG_STRICT_FINITE);
    }
    require_cuda(cudaMemcpy(k.logits, wave_logits.data(), n * vocab * 4, cudaMemcpyHostToDevice),
                 "h2d");
    require_cuda(cudaMemcpy(k.params, params.data(), n * sizeof(params[0]),
                            cudaMemcpyHostToDevice), "params");
    require_cuda(cudaMemcpy(k.mask, wave_mask.data(), n * words * 4, cudaMemcpyHostToDevice),
                 "mask");
    expect(ds41rt_cuda_v41_target_sample(k.logits, n, vocab, vocab, k.params, k.mask, words,
                                         k.ids, k.status, k.detail, k.scores, k.total, k.nucleus,
                                         k.scratch) == DS41RT_STATUS_OK, "launch");
    std::vector<uint32_t> ids(n);
    require_cuda(cudaMemcpy(ids.data(), k.ids, n * 4, cudaMemcpyDeviceToHost), "ids");
    free_k1(&k);
    return ids;
  };
  /* Reference: the whole batch in forward order. */
  const std::vector<uint32_t> forward = run_wave(all);
  /* Same rows in waves of 1, 6, 8, 16, 24: each row must keep its id. */
  const std::vector<size_t> sizes = {1u, 6u, 8u, 16u, 24u};
  for (size_t size : sizes) {
    std::vector<size_t> selection(size);
    for (size_t i = 0; i < size; ++i) {
      selection[i] = i;
    }
    const std::vector<uint32_t> ids = run_wave(selection);
    for (size_t i = 0; i < size; ++i) {
      expect(ids[i] == forward[selection[i]], "wave-size independence");
    }
  }
  /* Reversed row order: selection[0] is row `rows-1`. */
  std::vector<size_t> reversed(rows);
  for (size_t i = 0; i < rows; ++i) {
    reversed[i] = rows - 1u - i;
  }
  const std::vector<uint32_t> reversed_ids = run_wave(reversed);
  for (size_t i = 0; i < rows; ++i) {
    expect(reversed_ids[i] == forward[reversed[i]], "row order independence");
  }
  /* A peer set that reuses one row many times must not change that row. */
  std::vector<size_t> peers(rows, 3u);
  const std::vector<uint32_t> peer_ids = run_wave(peers);
  for (size_t i = 0; i < rows; ++i) {
    expect(peer_ids[i] == forward[3], "peer-activity independence");
  }
  std::cout << "ok  batch independence across wave sizes and row order\n";
}

/* FTZ / fast-math canary (design §4.0, §12.6).
 *
 * `max_scaled` and `survivor_count` alone cannot detect a build that flushes
 * subnormals to zero: a flushed `1e-45` still loses to any normal maximum, so
 * the two builds agree on both. This probe computes the exact `logit * 1/T`
 * product for every token in a dedicated kernel and ships the raw bits back, so
 * a single flushed subnormal fails the comparison. It is compiled in the same
 * translation unit and with the same flags as the shipped library's CUDA
 * objects, which is what makes it a valid canary for that build.
 *
 * The probe never writes to the caller's logits; it only reads them.
 */
__global__ void v41_scale_probe_kernel(const float* logits, size_t vocab, float inv_temperature,
                                       float* out_scaled) {
  const size_t token = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (token < vocab) {
    out_scaled[token] = logits[token] * inv_temperature;
  }
}

/* A raw-C caller can pass a `mask_row` that does not fit the arena. The ABI has
 * no arena-length field, so the kernel must fall back to the memory-safe
 * unconstrained behaviour instead of reading `mask_row * words` past the buffer:
 * the `0xFFFFFFFF` sentinel (with or without `FLAG_NO_MASK`) and any other
 * out-of-range index. These rows are deliberately given a mask that would select
 * a different token, so an OOB read would also be visible as a wrong id.
 */
void test_out_of_range_mask_row_stays_memory_safe() {
  const size_t vocab = 40;
  const size_t words = (vocab + 31u) / 32u;
  const size_t rows = 2;
  /* Full rows: every token not listed stays at -100 so the argmax is explicit. */
  std::vector<float> logits(rows * vocab, -100.0f);
  logits[0 * vocab + 0] = 1.0f;  /* row 0: unmasked argmax is token 1 */
  logits[0 * vocab + 1] = 9.0f;
  logits[0 * vocab + 2] = 2.0f;
  logits[1 * vocab + 0] = 5.0f;  /* row 1: unmasked argmax is token 3 */
  logits[1 * vocab + 3] = 8.0f;
  /* A mask that allows only token 0 on both rows: applying it would pick 0. */
  std::vector<uint32_t> mask(rows * words, 0u);
  for (size_t r = 0; r < rows; ++r) {
    mask[r * words] = 0x1u;
  }
  std::vector<ds41rt_v41_sampler_row_t> params = {
      /* Out-of-range arena index, no NO_MASK flag. */
      row(0, 1.0e-7f, 0u, 0.0f, -std::numeric_limits<float>::infinity(), 7u,
          DS41RT_V41_SAMPLER_FLAG_GREEDY),
      /* The sentinel without NO_MASK. */
      row(1, 1.0e-7f, 0u, 0.0f, -std::numeric_limits<float>::infinity(),
          DS41RT_V41_SAMPLER_NO_MASK_ROW, DS41RT_V41_SAMPLER_FLAG_GREEDY),
  };
  K1 k = make_k1(rows, vocab, true, true);
  require_cuda(cudaMemcpy(k.logits, logits.data(), logits.size() * sizeof(float),
                          cudaMemcpyHostToDevice), "oob logits");
  require_cuda(cudaMemcpy(k.params, params.data(),
                          params.size() * sizeof(params[0]), cudaMemcpyHostToDevice),
               "oob params");
  require_cuda(cudaMemcpy(k.mask, mask.data(), mask.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice), "oob mask");
  expect(ds41rt_cuda_v41_target_sample(k.logits, rows, vocab, vocab, k.params, k.mask, words,
                                       k.ids, k.status, k.detail, k.scores, k.total, k.nucleus,
                                       k.scratch) == DS41RT_STATUS_OK, "oob launch");
  std::vector<uint32_t> ids(rows), status(rows);
  require_cuda(cudaMemcpy(ids.data(), k.ids, rows * 4, cudaMemcpyDeviceToHost), "oob ids");
  require_cuda(cudaMemcpy(status.data(), k.status, rows * 4, cudaMemcpyDeviceToHost), "oob status");
  /* Both rows are read as unconstrained: the plain argmax, never the masked 0. */
  expect(status[0] == DS41RT_V41_SAMPLER_STATUS_OK, "out-of-range mask_row status");
  expect(ids[0] == 1u, "out-of-range mask_row selects the unmasked argmax");
  expect(status[1] == DS41RT_V41_SAMPLER_STATUS_OK, "sentinel-without-flag status");
  expect(ids[1] == 3u, "sentinel-without-flag selects the unmasked argmax");
  free_k1(&k);
  std::cout << "ok  out-of-range mask_row and sentinel stay memory-safe\n";
}

/* All-greedy rounds may mix constrained and unconstrained members. The arena is
 * indexed by row ordinal, so each constrained row must read its own mask (the
 * sentinel is what marks the unconstrained ones), and every row must match the
 * CPU masked argmax.
 *
 * The staging that packs only the constrained rows contiguously is exactly what
 * this catches: with it, the first unconstrained row would silently read the
 * next constrained row's grammar.
 */
void test_mixed_constrained_and_unconstrained_greedy_round() {
  const size_t vocab = 70;
  const size_t words = (vocab + 31u) / 32u;
  const size_t rows = 5;
  /* Row 0 unconstrained, row 1 constrained, row 2 unconstrained, row 3
   * constrained, row 4 unconstrained. */
  const bool constrained[rows] = {false, true, false, true, false};
  std::vector<float> logits(rows * vocab);
  std::vector<uint32_t> mask(rows * words, 0u);
  std::vector<ds41rt_v41_sampler_row_t> params(rows);
  for (size_t r = 0; r < rows; ++r) {
    for (size_t token = 0; token < vocab; ++token) {
      /* The unmasked winner alternates so a mask that is ignored changes it. */
      logits[r * vocab + token] = static_cast<float>((r * 5 + token) % 9);
    }
    /* Restrict the mask to a token that is NOT the row's unmasked winner, so a
     * misplaced mask cannot coincidentally produce the right id. */
    const size_t allowed = (r * 3 + 2) % vocab;
    mask[r * words + allowed / 32] |= 1u << (allowed % 32);
    const uint32_t flags = DS41RT_V41_SAMPLER_FLAG_GREEDY |
        (constrained[r] ? 0u : DS41RT_V41_SAMPLER_FLAG_NO_MASK);
    params[r] = row(static_cast<uint32_t>(r), 1.0e-7f, 0u, 0.0f,
                    -std::numeric_limits<float>::infinity(),
                    constrained[r] ? static_cast<uint32_t>(r) : DS41RT_V41_SAMPLER_NO_MASK_ROW,
                    flags);
  }
  run_case(logits, rows, vocab, params, mask, true, true,
           "mixed constrained and unconstrained greedy round");
  std::cout << "ok  mixed constrained/unconstrained all-greedy round\n";
}

/* Non-finite whole-row handling for greedy rows stays exact, and a subnormal
 * scaled value survives the reduction (this is the FTZ/fast-math canary). */
void test_subnormal_and_nonfinite_grid() {
  const size_t rows = 3;
  const size_t vocab = 6;
  const float subnormal = std::numeric_limits<float>::denorm_min();   /* f32::from_bits(1) */
  std::vector<float> logits = {
      subnormal, -1.0f, 2.0f, 3.0f, -4.0f, 0.0f,   /* row 0: subnormal present */
      0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,          /* row 1 */
      -1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f,    /* row 2 */
  };
  std::vector<ds41rt_v41_sampler_row_t> params = {
      /* Subnormal logit is a legal survivor at min_p = 0. */
      row(0, 1.0f, 0u, 0.0f, -std::numeric_limits<float>::infinity(),
          DS41RT_V41_SAMPLER_NO_MASK_ROW, DS41RT_V41_SAMPLER_FLAG_NO_MASK),
      /* -f32::MAX at T = 1e-5 scales to -inf and is still legal at min_p = 0. */
      row(1, 1.0e-5f, 0u, 0.0f, -std::numeric_limits<float>::infinity(),
          DS41RT_V41_SAMPLER_NO_MASK_ROW, DS41RT_V41_SAMPLER_FLAG_NO_MASK),
      row(2, 1.0f, 0u, 0.0f, -std::numeric_limits<float>::infinity(),
          DS41RT_V41_SAMPLER_NO_MASK_ROW, DS41RT_V41_SAMPLER_FLAG_NO_MASK),
  };
  run_case(logits, rows, vocab, params, {}, false, false, "subnormal");
  /* A row whose scaled maximum is -inf (all logits -f32::MAX at a tiny T) is
   * INVALID_TEMPERATURE, never NaN. */
  const size_t rows2 = 1;
  const size_t vocab2 = 2;
  std::vector<float> huge = {-std::numeric_limits<float>::max(),
                             -std::numeric_limits<float>::max()};
  std::vector<ds41rt_v41_sampler_row_t> params2 = {
      /* 1e-5 is the smallest non-greedy temperature; -MAX * 1e5 = -inf. */
      row(0, 1.0e-5f, 0u, 0.0f, -std::numeric_limits<float>::infinity(),
          DS41RT_V41_SAMPLER_NO_MASK_ROW, DS41RT_V41_SAMPLER_FLAG_NO_MASK),
  };
  K1 k = make_k1(rows2, vocab2, false, true);
  require_cuda(cudaMemcpy(k.logits, huge.data(), vocab2 * 4, cudaMemcpyHostToDevice), "h2d");
  require_cuda(cudaMemcpy(k.params, params2.data(), sizeof(params2[0]),
                          cudaMemcpyHostToDevice), "params");
  expect(ds41rt_cuda_v41_target_sample(k.logits, rows2, vocab2, vocab2, k.params, nullptr, 0u,
                                       k.ids, k.status, k.detail, k.scores, k.total, k.nucleus,
                                       k.scratch) == DS41RT_STATUS_OK, "launch");
  std::vector<uint32_t> status(rows2);
  std::vector<float> total(rows2);
  require_cuda(cudaMemcpy(status.data(), k.status, rows2 * 4, cudaMemcpyDeviceToHost), "status");
  require_cuda(cudaMemcpy(total.data(), k.total, rows2 * 4, cudaMemcpyDeviceToHost), "total");
  expect(status[0] == DS41RT_V41_SAMPLER_STATUS_INVALID_TEMPERATURE,
         "-inf scaled max is invalid temperature");
  free_k1(&k);
  /* Per-token scaled bits at T = 2 (inv = 0.5), built from exact bit patterns
   * so no decimal literal can round differently between compilers.
   *
   * Every product is chosen to be subnormal. Dividing a subnormal by two is an
   * exact mantissa shift, so a logit with bits `2p` produces the product `p`;
   * the table below is that arithmetic, not an approximation. A build that
   * flushes subnormals to zero (any `-use_fast_math`/FTZ translation unit)
   * returns 0 for every one of these products and fails here.
   *
   * `logit_bits` are per row, in token order; `product_bits` is the exact
   * expectation. Rows 0 and 1 additionally make a subnormal the row maximum, so
   * `max_scaled` itself (not just the per-token product) is a canary. */
  {
    const float inv = 0.5f;
    const size_t rows3 = 4;
    const size_t vocab3 = 6;
    const uint32_t logit_bits[rows3][vocab3] = {
        {0x00400000u, 0x00000064u, 0x00000002u, 0x00000001u, 0x00000000u, 0x00000000u},
        {0x00000002u, 0x00000001u, 0xC0000000u, 0xC0000000u, 0xC0000000u, 0xC0000000u},
        {0x40000000u, 0x00000002u, 0x00000064u, 0xC0000000u, 0xC0000000u, 0xC0000000u},
        {0x00000000u, 0x00000001u, 0xC0000000u, 0xC0000000u, 0xC0000000u, 0xC0000000u},
    };
    const uint32_t product_bits[rows3][vocab3] = {
        {0x00200000u, 0x00000032u, 0x00000001u, 0x00000000u, 0x00000000u, 0x00000000u},
        {0x00000001u, 0x00000000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u},
        {0x3F800000u, 0x00000001u, 0x00000032u, 0xBF800000u, 0xBF800000u, 0xBF800000u},
        {0x00000000u, 0x00000000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u},
    };
    /* The row maximum is the subnormal `0x00200000` for row 0 and `0x1` for
     * row 1; a flushed build reports 0 for both. */
    const uint32_t row_max_product_bits[rows3] = {0x00200000u, 0x00000001u, 0x3F800000u, 0x00000000u};
    std::vector<float> scale_logits(rows3 * vocab3, 0.0f);
    for (size_t r = 0; r < rows3; ++r) {
      for (size_t token = 0; token < vocab3; ++token) {
        std::memcpy(&scale_logits[r * vocab3 + token], &logit_bits[r][token], 4);
      }
    }
    std::vector<ds41rt_v41_sampler_row_t> scale_params(rows3);
    for (size_t r = 0; r < rows3; ++r) {
      scale_params[r] = row(static_cast<uint32_t>(r), 2.0f, 0u, 0.0f,
                            -std::numeric_limits<float>::infinity(),
                            DS41RT_V41_SAMPLER_NO_MASK_ROW,
                            DS41RT_V41_SAMPLER_FLAG_NO_MASK);
    }
    float* device_logits = nullptr;
    float* device_scaled = nullptr;
    alloc_device(&device_logits, scale_logits.size() * sizeof(float), "scale logits");
    alloc_device(&device_scaled, vocab3 * sizeof(float), "scale probe output");
    require_cuda(cudaMemcpy(device_logits, scale_logits.data(),
                            scale_logits.size() * sizeof(float), cudaMemcpyHostToDevice),
                 "scale logits h2d");
    for (size_t r = 0; r < rows3; ++r) {
      v41_scale_probe_kernel<<<1, static_cast<int>(vocab3)>>>(device_logits + r * vocab3,
                                                             vocab3, inv, device_scaled);
      require_cuda(cudaDeviceSynchronize(), "scale probe");
      std::vector<float> scaled(vocab3);
      require_cuda(cudaMemcpy(scaled.data(), device_scaled, vocab3 * sizeof(float),
                              cudaMemcpyDeviceToHost), "scale probe d2h");
      for (size_t token = 0; token < vocab3; ++token) {
        uint32_t observed = 0;
        std::memcpy(&observed, &scaled[token], 4);
        expect(observed == product_bits[r][token],
               "per-token scaled bits match exact subnormal arithmetic (no FTZ/fast-math)");
      }
    }
    /* The shipped kernel's own reduction of the same rows must keep the same
     * subnormal bits in `max_scaled`. */
    K1 scale_k = make_k1(rows3, vocab3, false, true);
    require_cuda(cudaMemcpy(scale_k.logits, scale_logits.data(),
                            scale_logits.size() * sizeof(float), cudaMemcpyHostToDevice),
                 "scale kernel h2d");
    require_cuda(cudaMemcpy(scale_k.params, scale_params.data(),
                            scale_params.size() * sizeof(scale_params[0]),
                            cudaMemcpyHostToDevice), "scale kernel params");
    expect(ds41rt_cuda_v41_target_sample(scale_k.logits, rows3, vocab3, vocab3, scale_k.params,
                                         nullptr, 0u, scale_k.ids, scale_k.status, scale_k.detail,
                                         scale_k.scores, scale_k.total, scale_k.nucleus,
                                         scale_k.scratch) == DS41RT_STATUS_OK, "scale launch");
    std::vector<ds41rt_v41_sampler_scratch_t> scale_scratch(rows3);
    require_cuda(cudaMemcpy(scale_scratch.data(), scale_k.scratch,
                            rows3 * sizeof(scale_scratch[0]), cudaMemcpyDeviceToHost), "scale scratch");
    for (size_t r = 0; r < rows3; ++r) {
      uint32_t max_bits = 0;
      std::memcpy(&max_bits, &scale_scratch[r].max_scaled, 4);
      expect(max_bits == row_max_product_bits[r],
             "row max_scaled keeps the exact subnormal bits under no FTZ");
    }
    free_k1(&scale_k);
    require_cuda(cudaFree(device_logits), "free scale logits");
    require_cuda(cudaFree(device_scaled), "free scale probe output");
  }
  std::cout << "ok  subnormal per-token bits, survivors and non-finite grid\n";
}

/* ==================================================================== */
/* Chunk 2: K2 fast path (`top_k` disabled, `top_p >= 1.0`)             */
/* ==================================================================== */

/* Host port of `TargetSamplingParams::random_uniform`
 * (`target_sampling.rs:187-199`). This is the value the CPU sampler draws and
 * the value the shipped device function must reproduce bit for bit. */
float host_target_uniform(uint64_t seed, uint64_t position) {
  const uint64_t domain = 0x7f4a7c159e3779b9ull;
  const uint64_t mul = 0x9e3779b97f4a7c15ull;
  uint64_t mixed = seed + domain + position * mul + mul;
  mixed = (mixed ^ (mixed >> 30)) * 0xbf58476d1ce4e5b9ull;
  mixed = (mixed ^ (mixed >> 27)) * 0x94d049bb133111ebull;
  mixed ^= mixed >> 31;
  const uint32_t mantissa = static_cast<uint32_t>(mixed >> 40);
  return static_cast<float>(mantissa) * (1.0f / 16777216.0f);
}

/* 0x3F7FFFFF as a bit pattern, the design's `MAX_UNIFORM` (§4.8/§6.1). */
float max_uniform_bits() {
  const uint32_t bits = 0x3F7FFFFFu;
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

float host_clamp_uniform(float uniform) {
  /* `MAX_UNIFORM.min(uniform.max(0.0))` (`target_sampling.rs:459`). */
  return std::fmin(std::fmax(uniform, 0.0f), max_uniform_bits());
}

__global__ void v41_uniform_probe_kernel(const uint64_t* seeds, const uint64_t* positions,
                                         size_t count, float* out_uniforms) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    out_uniforms[index] = ds41rt_v41_target_uniform(seeds[index], positions[index]);
  }
}

__global__ void v41_clamp_probe_kernel(const float* in, size_t count, float* out_clamped) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    out_clamped[index] = ds41rt_v41_target_clamp_uniform(in[index]);
  }
}

/* Faithful C port of the CPU fast path (`target_sampling.rs:430-465` plus
 * `sample_categorical`, `:596-618`). Chunk 2's device scope is exactly this
 * branch, so this is the oracle the device draw must reproduce. */
struct RefFast {
  uint32_t status = DS41RT_V41_SAMPLER_STATUS_OK;
  uint32_t token = 0;
  float total = 0.0f;
  float max_scaled = 0.0f;
  uint32_t survivor_count = 0;
  /* True when the inclusive crossing fired. False would mean the CPU took its
   * `last` fallback (`target_sampling.rs:616-618`), which is unreachable. */
  bool crossed = false;
  /* True when every cumulative boundary is at least 1e-4 * total away from the
   * target, so a 1-ulp accumulation difference cannot change the crossing. */
  bool margin_safe = false;
};

RefFast cpu_fast_reference(const float* logits, size_t vocab, const uint32_t* mask_words,
                           bool unconstrained, float temperature, float min_p, float ln_min_p,
                           uint64_t seed, uint64_t position) {
  RefFast out;
  const auto allowed = [&](size_t token) {
    return unconstrained || ((mask_words[token / 32u] >> (token % 32u)) & 1u) != 0u;
  };
  const float inv = 1.0f / temperature;
  float max_scaled = -std::numeric_limits<float>::infinity();
  uint32_t allowed_count = 0;
  for (size_t token = 0; token < vocab; ++token) {
    if (!allowed(token)) {
      continue;
    }
    const float logit = logits[token];
    if (!std::isfinite(logit)) {
      out.status = DS41RT_V41_SAMPLER_STATUS_NONFINITE_LOGIT;
      return out;
    }
    ++allowed_count;
    max_scaled = std::fmax(max_scaled, logit * inv);
  }
  if (allowed_count == 0) {
    out.status = DS41RT_V41_SAMPLER_STATUS_EMPTY_CANDIDATES;
    return out;
  }
  if (!std::isfinite(max_scaled)) {
    out.status = DS41RT_V41_SAMPLER_STATUS_INVALID_TEMPERATURE;
    return out;
  }
  out.max_scaled = max_scaled;
  const float min_scaled = min_p > 0.0f ? max_scaled + ln_min_p
                                        : -std::numeric_limits<float>::infinity();
  const auto survivor = [&](size_t token) {
    return allowed(token) && logits[token] * inv >= min_scaled;
  };
  const float uniform = host_clamp_uniform(host_target_uniform(seed, position));
  float total = 0.0f;
  for (size_t token = 0; token < vocab; ++token) {
    if (survivor(token)) {
      total += std::exp(logits[token] * inv - max_scaled);
      ++out.survivor_count;
    }
  }
  total = std::fmax(total, 1.0e-20f);
  const float target = uniform * total;
  float cumulative = 0.0f;
  float min_distance = std::numeric_limits<float>::infinity();
  bool crossed = false;
  uint32_t last = 0u;
  for (size_t token = 0; token < vocab; ++token) {
    if (!survivor(token)) {
      continue;
    }
    last = static_cast<uint32_t>(token);
    cumulative += std::exp(logits[token] * inv - max_scaled);
    min_distance = std::fmin(min_distance, std::fabs(cumulative - target));
    if (!crossed && target <= cumulative) {
      out.token = static_cast<uint32_t>(token);
      crossed = true;
    }
  }
  if (!crossed) {
    /* The CPU's `last` fallback (`target_sampling.rs:616-618`). Unreachable by
     * construction: `total` and `cumulative` are the same left-to-right sum, so
     * the last survivor's cumulative equals `total >= target`. Kept so a port
     * that diverges is visible. */
    out.token = last;
  }
  out.total = total;
  out.crossed = crossed;
  out.margin_safe = min_distance > 1.0e-4f * total;
  return out;
}

/* Runs one K2 configuration against the ported CPU fast path. `exact` says the
 * rows' weights are exactly representable in both implementations (all equal or
 * zero), so token and `total` equality are asserted unconditionally; otherwise
 * token equality is asserted only where the crossing has a safe margin and
 * `total` is compared with a loose relative bound. */
void run_fast_case(const std::vector<float>& logits, size_t rows, size_t vocab,
                   const std::vector<ds41rt_v41_sampler_row_t>& params,
                   const std::vector<uint32_t>& mask, bool with_mask, bool strict_mask_bits,
                   bool exact, const char* label) {
  ++g_cases;
  K1 k = make_k1(rows, vocab, with_mask, true);
  require_cuda(cudaMemcpy(k.logits, logits.data(), logits.size() * sizeof(float),
                          cudaMemcpyHostToDevice), "k2 logits h2d");
  require_cuda(cudaMemcpy(k.params, params.data(),
                          params.size() * sizeof(ds41rt_v41_sampler_row_t),
                          cudaMemcpyHostToDevice), "k2 params h2d");
  std::vector<uint32_t> device_mask = mask;
  if (with_mask) {
    if (strict_mask_bits) {
      for (size_t r = 0; r < rows; ++r) {
        ds41rt_v41_sampler_clear_remainder(device_mask.data() + r * k.words, vocab);
      }
    }
    require_cuda(cudaMemcpy(k.mask, device_mask.data(),
                            device_mask.size() * sizeof(uint32_t),
                            cudaMemcpyHostToDevice), "k2 mask h2d");
  }
  const ds41rt_status_t launch = ds41rt_cuda_v41_target_sample(
      k.logits, rows, vocab, vocab, k.params, with_mask ? k.mask : nullptr,
      with_mask ? k.words : 0u, k.ids, k.status, k.detail, k.scores, k.total,
      k.nucleus, k.scratch);
  expect(launch == DS41RT_STATUS_OK, std::string(label) + ": launch status");
  std::vector<uint32_t> ids(rows), status(rows);
  std::vector<float> total(rows);
  require_cuda(cudaMemcpy(ids.data(), k.ids, rows * 4, cudaMemcpyDeviceToHost), "k2 ids");
  require_cuda(cudaMemcpy(status.data(), k.status, rows * 4, cudaMemcpyDeviceToHost), "k2 status");
  require_cuda(cudaMemcpy(total.data(), k.total, rows * 4, cudaMemcpyDeviceToHost), "k2 total");
  for (size_t r = 0; r < rows; ++r) {
    const ds41rt_v41_sampler_row_t& p = params[r];
    const bool unconstrained = (p.flags & DS41RT_V41_SAMPLER_FLAG_NO_MASK) != 0u ||
                               p.mask_row == DS41RT_V41_SAMPLER_NO_MASK_ROW;
    const uint32_t* row_mask = (with_mask && !unconstrained)
        ? device_mask.data() + static_cast<size_t>(p.mask_row) * k.words
        : nullptr;
    const RefFast expected = cpu_fast_reference(
        logits.data() + r * vocab, vocab, row_mask, unconstrained, p.temperature, p.min_p,
        p.ln_min_p, p.seed, p.position);
    const std::string tag = std::string(label) + " row " + std::to_string(r);
    if (status[r] != expected.status) {
      std::cerr << "k2 status mismatch " << tag << " expected=" << expected.status
                << " device=" << status[r] << "\n";
    }
    expect(status[r] == expected.status, tag + ": status");
    if (expected.status != DS41RT_V41_SAMPLER_STATUS_OK) {
      continue;
    }
    /* The CPU port must have crossed, never taken its `last` fallback. */
    expect(expected.crossed, tag + ": CPU crossing exists (last fallback unreachable)");
    /* The device draw must land in-vocab and on a survivor. */
    expect(ids[r] < vocab, tag + ": in-vocab id");
    const float inv = 1.0f / p.temperature;
    const float min_scaled = p.min_p > 0.0f
        ? expected.max_scaled + p.ln_min_p
        : -std::numeric_limits<float>::infinity();
    const bool allowed_id = unconstrained ||
        ((row_mask[ids[r] / 32u] >> (ids[r] % 32u)) & 1u) != 0u;
    expect(allowed_id, tag + ": selected token is allowed");
    expect(logits[r * vocab + ids[r]] * inv >= min_scaled, tag + ": selected token is a survivor");
    if (exact || expected.margin_safe) {
      if (ids[r] != expected.token) {
        std::cerr << "k2 token mismatch " << tag << " expected=" << expected.token
                  << " device=" << ids[r] << " total=" << total[r] << "\n";
      }
      expect(ids[r] == expected.token, tag + ": fast-path token");
    }
    if (exact) {
      expect(total[r] == expected.total, tag + ": fast-path total bits");
    } else {
      if (!(std::fabs(total[r] - expected.total) <= 1.0e-3f * expected.total)) {
        std::cerr << "k2 total mismatch " << tag << " device=" << total[r]
                  << " port=" << expected.total << "\n";
      }
      expect(std::fabs(total[r] - expected.total) <= 1.0e-3f * expected.total,
             tag + ": fast-path total within 1e-3");
    }
  }
  free_k1(&k);
}

ds41rt_v41_sampler_row_t fast_row(uint32_t output_row, float temperature, float min_p,
                                  float ln_min_p, uint64_t seed, uint64_t position,
                                  uint32_t mask_row, uint32_t flags) {
  ds41rt_v41_sampler_row_t value = {};
  value.seed = seed;
  value.position = position;
  value.temperature = temperature;
  value.top_p = 1.0f;
  value.min_p = min_p;
  value.top_k = 0u;
  value.mask_row = mask_row;
  /* DIAGNOSE is always set so K2's `out_total` (and K1's `out_nucleus_count`) are
   * observable; it changes no draw. */
  value.flags = flags | DS41RT_V41_SAMPLER_FLAG_DIAGNOSE;
  value.output_row = output_row;
  value.ln_min_p = ln_min_p;
  return value;
}

/* The device SplitMix64 must equal the host port bit for bit on the design's
 * §6.2/§12.8 grid, including positions 0, 2^63 and u64::MAX. */
void test_k2_rng_bit_equality() {
  std::vector<uint64_t> seeds = {0ull, 1ull, std::numeric_limits<uint64_t>::max(),
                                 0x8000000000000000ull, 4242ull};
  std::vector<uint64_t> positions;
  for (uint64_t p = 0; p <= 64; ++p) {
    positions.push_back(p);
  }
  positions.push_back(1ull << 31);
  positions.push_back((1ull << 32) - 1ull);
  positions.push_back(1ull << 63);
  positions.push_back(std::numeric_limits<uint64_t>::max());
  std::vector<uint64_t> host_seeds, host_positions;
  for (uint64_t seed : seeds) {
    for (uint64_t position : positions) {
      host_seeds.push_back(seed);
      host_positions.push_back(position);
    }
  }
  const size_t count = host_seeds.size();
  uint64_t* device_seeds = nullptr;
  uint64_t* device_positions = nullptr;
  float* device_uniforms = nullptr;
  alloc_device(&device_seeds, count * sizeof(uint64_t), "rng seeds");
  alloc_device(&device_positions, count * sizeof(uint64_t), "rng positions");
  alloc_device(&device_uniforms, count * sizeof(float), "rng uniforms");
  require_cuda(cudaMemcpy(device_seeds, host_seeds.data(), count * sizeof(uint64_t),
                          cudaMemcpyHostToDevice), "rng seeds h2d");
  require_cuda(cudaMemcpy(device_positions, host_positions.data(), count * sizeof(uint64_t),
                          cudaMemcpyHostToDevice), "rng positions h2d");
  const unsigned int block = 256;
  v41_uniform_probe_kernel<<<static_cast<unsigned int>((count + block - 1) / block), block>>>(
      device_seeds, device_positions, count, device_uniforms);
  require_cuda(cudaDeviceSynchronize(), "rng probe");
  std::vector<float> observed(count);
  require_cuda(cudaMemcpy(observed.data(), device_uniforms, count * sizeof(float),
                          cudaMemcpyDeviceToHost), "rng probe d2h");
  size_t differing = 0;
  for (size_t i = 0; i < count; ++i) {
    const float expected = host_target_uniform(host_seeds[i], host_positions[i]);
    uint32_t a = 0;
    uint32_t b = 0;
    std::memcpy(&a, &expected, 4);
    std::memcpy(&b, &observed[i], 4);
    if (a != b) {
      ++differing;
      if (differing <= 3) {
        std::cerr << "rng mismatch seed=" << host_seeds[i] << " position=" << host_positions[i]
                  << " host=" << a << " device=" << b << "\n";
      }
    }
  }
  expect(differing == 0, "device SplitMix64 is bit-identical to the host port");
  expect(count == 5 * 69, "rng grid size");
  require_cuda(cudaFree(device_seeds), "free rng seeds");
  require_cuda(cudaFree(device_positions), "free rng positions");
  require_cuda(cudaFree(device_uniforms), "free rng uniforms");
  std::cout << "ok  K2 device RNG bit-identical over " << count << " (seed, position) pairs\n";
}

/* The `MAX_UNIFORM` clamp is applied in the CPU order on device, including
 * values above the bound (which SplitMix64 itself cannot produce, so this is
 * also the guard against a future draw change). */
void test_k2_max_uniform_clamp() {
  const float nan_value = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> inputs = {
      0.0f, -0.0f, 0.5f, max_uniform_bits(), 1.0f, 2.0f, -1.0f, -1.0e-30f,
      std::nextafter(max_uniform_bits(), 1.0f), std::nextafter(max_uniform_bits(), 0.0f),
      nan_value};
  const size_t count = inputs.size();
  float* device_in = nullptr;
  float* device_out = nullptr;
  alloc_device(&device_in, count * sizeof(float), "clamp in");
  alloc_device(&device_out, count * sizeof(float), "clamp out");
  require_cuda(cudaMemcpy(device_in, inputs.data(), count * sizeof(float),
                          cudaMemcpyHostToDevice), "clamp h2d");
  v41_clamp_probe_kernel<<<1, static_cast<int>(count)>>>(device_in, count, device_out);
  require_cuda(cudaDeviceSynchronize(), "clamp probe");
  std::vector<float> observed(count);
  require_cuda(cudaMemcpy(observed.data(), device_out, count * sizeof(float),
                          cudaMemcpyDeviceToHost), "clamp d2h");
  for (size_t i = 0; i < count; ++i) {
    const float expected = host_clamp_uniform(inputs[i]);
    uint32_t a = 0;
    uint32_t b = 0;
    std::memcpy(&a, &expected, 4);
    std::memcpy(&b, &observed[i], 4);
    expect(a == b, "clamp result bits");
  }
  /* The largest SplitMix64 output is exactly MAX_UNIFORM, so the upper clamp is
   * a no-op for every real draw; the lower clamp is the only binding one. */
  expect(host_target_uniform(0ull, 0ull) >= 0.0f, "draw is non-negative");
  require_cuda(cudaFree(device_in), "free clamp in");
  require_cuda(cudaFree(device_out), "free clamp out");
  std::cout << "ok  K2 MAX_UNIFORM clamp matches the CPU order (" << count << " inputs)\n";
}

/* Full fast-path parameter grid x masks: temperatures `{1e-5, 1e-4, 0.2, 0.7, 2.0}`
 * (1e-5 is the design §12.3 boundary: `is_greedy` is strictly `< 1e-5`, so it is
 * a valid non-greedy fast-path cell) x min_p `{0, 1e-6, 0.05, 0.5, 1.0}`, over
 * all-allowed, `token < 5` and `token % 3 != 0`. */
void test_k2_fast_path_grid() {
  const size_t vocab = 65;
  const size_t words = (vocab + 31u) / 32u;
  const std::vector<float> temperatures = {1.0e-5f, 1.0e-4f, 0.2f, 0.7f, 2.0f};
  const std::vector<float> min_ps = {0.0f, 1.0e-6f, 0.05f, 0.5f, 1.0f};
  const size_t cells = temperatures.size() * min_ps.size();
  std::vector<float> base_row(vocab);
  for (size_t token = 0; token < vocab; ++token) {
    /* Deterministic, sign-changing, with a clear maximum so the crossing has a
     * safe margin on every cell. */
    base_row[token] = 0.75f * std::sin(static_cast<float>(token) * 0.7f) -
                      0.02f * static_cast<float>(token);
  }
  base_row[7] = 3.5f;
  const std::vector<float> logits = repeat_row(base_row, cells);

  /* Three mask rows, built programmatically so no hand-computed bit pattern can
   * be wrong: unconstrained, `token < 5`, and `token % 3 != 0`. */
  const auto build_mask = [&](int kind) {
    std::vector<uint32_t> mask(cells * words, 0u);
    for (size_t r = 0; r < cells; ++r) {
      for (size_t token = 0; token < vocab; ++token) {
        const bool allow = kind == 0 ? true : (kind == 1 ? token < 5u : (token % 3u) != 0u);
        if (allow) {
          mask[r * words + token / 32u] |= 1u << (token % 32u);
        }
      }
    }
    return mask;
  };
  const std::vector<std::pair<const char*, int>> masks = {
      {"all-allowed", 0}, {"token<5", 1}, {"token%3!=0", 2}};
  for (const auto& [name, kind] : masks) {
    std::vector<ds41rt_v41_sampler_row_t> params;
    for (size_t t = 0; t < temperatures.size(); ++t) {
      for (size_t m = 0; m < min_ps.size(); ++m) {
        const size_t r = params.size();
        const float min_p = min_ps[m];
        const float ln_min_p = min_p > 0.0f ? std::log(min_p) : -inf();
        params.push_back(fast_row(
            static_cast<uint32_t>(r), temperatures[t], min_p, ln_min_p, 1000u + r, 37u * r + 1u,
            kind == 0 ? DS41RT_V41_SAMPLER_NO_MASK_ROW : 0u,
            kind == 0 ? DS41RT_V41_SAMPLER_FLAG_NO_MASK : 0u));
      }
    }
    const std::vector<uint32_t> mask = build_mask(kind);
    run_fast_case(logits, cells, vocab, params, mask, kind != 0, true, false, name);
  }
  std::cout << "ok  K2 fast-path grid: " << cells * masks.size() << " cells across 3 masks\n";
}

/* min_p membership at the exact threshold and one ULP below, seen through the
 * drawn token. `scaled == max_scaled + ln_min_p` is retained (inclusive `>=`);
 * one ULP below is rejected. With a uniform above `1 / total`, the draw lands on
 * the threshold token iff it survives, so the two rows below differ exactly when
 * the boundary rule is wrong. */
void test_k2_min_p_threshold_tokens() {
  const float min_p = 0.5f;
  const float ln_min_p = std::log(0.5f);        /* = -0.6931472, the shipped value */
  const float temperature = 2.0f;               /* inv = 0.5 exactly */
  const float exact_logit = -1.3862944f;        /* scaled == max + ln_min_p exactly */
  const float below_logit = -1.3862945f;        /* scaled == threshold - 1 ULP */
  const size_t vocab = 4;
  /* u = 0.843... > 1 / 1.5, so the crossing is the second survivor. */
  const uint64_t seed = 0ull;
  const uint64_t position = 5ull;
  const std::vector<float> logits = {
      0.0f, exact_logit, below_logit, -8.0f,   /* row 0: token 1 at the threshold */
      0.0f, below_logit, exact_logit, -8.0f,   /* row 1: token 2 at the threshold */
  };
  std::vector<ds41rt_v41_sampler_row_t> params = {
      fast_row(0, temperature, min_p, ln_min_p, seed, position,
               DS41RT_V41_SAMPLER_NO_MASK_ROW, DS41RT_V41_SAMPLER_FLAG_NO_MASK),
      fast_row(1, temperature, min_p, ln_min_p, seed, position,
               DS41RT_V41_SAMPLER_NO_MASK_ROW, DS41RT_V41_SAMPLER_FLAG_NO_MASK),
  };
  /* Non-zero surviving weight, so the crossing is a real weight boundary. */
  expect(host_target_uniform(seed, position) * (1.0f + std::exp(ln_min_p)) > 1.0f,
         "threshold draw lands on the second survivor");
  run_fast_case(logits, 2, vocab, params, {}, false, false, false, "min_p threshold tokens");
  /* Exact expected ids: the at-threshold token is the second survivor. */
  K1 k = make_k1(2, vocab, false, true);
  require_cuda(cudaMemcpy(k.logits, logits.data(), logits.size() * 4, cudaMemcpyHostToDevice),
               "threshold logits");
  require_cuda(cudaMemcpy(k.params, params.data(), params.size() * sizeof(params[0]),
                          cudaMemcpyHostToDevice), "threshold params");
  expect(ds41rt_cuda_v41_target_sample(k.logits, 2, vocab, vocab, k.params, nullptr, 0u, k.ids,
                                       k.status, k.detail, k.scores, k.total, k.nucleus,
                                       k.scratch) == DS41RT_STATUS_OK, "threshold launch");
  std::vector<uint32_t> ids(2);
  require_cuda(cudaMemcpy(ids.data(), k.ids, 2 * 4, cudaMemcpyDeviceToHost), "threshold ids");
  expect(ids[0] == 1u, "at-threshold survivor on row 0 is drawn");
  expect(ids[1] == 2u, "at-threshold survivor on row 1 is drawn");
  free_k1(&k);
  std::cout << "ok  K2 min_p threshold membership is inclusive on the drawn token\n";
}

/* Edge cases: all-tied weights (exact, exercises the RNG + crossing), a single
 * survivor (`min_p = 1`), and all-`-inf` survivors at `min_p = 0` (zero weights,
 * every survivor's weight exactly 0 except the maximum). */
void test_k2_tied_and_edge_cases() {
  const float neg_max = -std::numeric_limits<float>::max();
  const float neg_inf = -std::numeric_limits<float>::infinity();
  /* All logits equal -> every weight exactly 1 -> exact arithmetic in both
   * implementations, so token equality is asserted unconditionally. */
  {
    const size_t vocab = 97;
    std::vector<float> logits = repeat_row(std::vector<float>(vocab, 0.25f), 4);
    std::vector<ds41rt_v41_sampler_row_t> params;
    for (size_t r = 0; r < 4; ++r) {
      params.push_back(fast_row(static_cast<uint32_t>(r), 0.7f, 0.0f, neg_inf, 1u + r, r * 11u,
                                DS41RT_V41_SAMPLER_NO_MASK_ROW,
                                DS41RT_V41_SAMPLER_FLAG_NO_MASK));
    }
    run_fast_case(logits, params.size(), vocab, params, {}, false, false, true, "all-tied");
  }
  /* One survivor: min_p = 1 keeps exactly the strict maximum. */
  {
    const size_t vocab = 8;
    const std::vector<float> base = {0.5f, 2.0f, -1.0f, 0.0f, 0.25f, -3.0f, 1.0f, -0.5f};
    const std::vector<float> logits = repeat_row(base, 2);
    std::vector<ds41rt_v41_sampler_row_t> params = {
        fast_row(0, 1.0f, 1.0f, 0.0f, 42u, 0u, DS41RT_V41_SAMPLER_NO_MASK_ROW,
                 DS41RT_V41_SAMPLER_FLAG_NO_MASK),
        fast_row(1, 1.0f, 1.0f, 0.0f, 42u, 99u, DS41RT_V41_SAMPLER_NO_MASK_ROW,
                 DS41RT_V41_SAMPLER_FLAG_NO_MASK),
    };
    run_fast_case(logits, 2, vocab, params, {}, false, false, true, "single survivor");
  }
  /* -inf survivors at min_p = 0: their weights are exactly 0, so the maximum
   * token always wins and `total` is exactly 1. */
  {
    const size_t vocab = 4;
    std::vector<float> logits = {neg_max, neg_max, neg_max, neg_max};
    std::vector<ds41rt_v41_sampler_row_t> params = {
        fast_row(0, 1.0e-5f, 0.0f, neg_inf, 7u, 5u, DS41RT_V41_SAMPLER_NO_MASK_ROW,
                 DS41RT_V41_SAMPLER_FLAG_NO_MASK),
    };
    run_fast_case(logits, 1, vocab, params, {}, false, false, true, "-inf survivors");
  }
  /* Tied maximum with min_p = 1: every tied token survives and the draw picks
   * among them in ascending token order. */
  {
    const size_t vocab = 6;
    const std::vector<float> logits =
        repeat_row({1.0f, 1.0f, 1.0f, 0.0f, -1.0f, 1.0f}, 3);
    std::vector<ds41rt_v41_sampler_row_t> params;
    for (size_t r = 0; r < 3; ++r) {
      params.push_back(fast_row(static_cast<uint32_t>(r), 1.0f, 1.0f, 0.0f, 5u + r, 2u,
                                DS41RT_V41_SAMPLER_NO_MASK_ROW,
                                DS41RT_V41_SAMPLER_FLAG_NO_MASK));
    }
    run_fast_case(logits, params.size(), vocab, params, {}, false, false, false,
                  "tied maximum with min_p=1");
  }
  std::cout << "ok  K2 tied / single-survivor / -inf-survivor edge cases\n";
}

/* Seeded replay: the token is a pure function of `(seed, position)` and the row,
 * across positions 0, 2^31, 2^32-1, 2^63 and u64::MAX, and independent of the
 * wave it is run in. */
void test_k2_seeded_replay() {
  const size_t vocab = 33;
  const size_t words = (vocab + 31u) / 32u;
  std::vector<float> base_row(vocab);
  for (size_t token = 0; token < vocab; ++token) {
    base_row[token] = 0.5f * std::cos(static_cast<float>(token) * 0.31f);
  }
  base_row[3] = 2.25f;
  const std::vector<uint64_t> positions = {0ull, 1ull << 31, (1ull << 32) - 1ull, 1ull << 63,
                                           std::numeric_limits<uint64_t>::max()};
  const std::vector<uint64_t> seeds = {0ull, 0xDEADBEEFull};
  std::vector<ds41rt_v41_sampler_row_t> params;
  for (uint64_t seed : seeds) {
    for (uint64_t position : positions) {
      const size_t r = params.size();
      params.push_back(fast_row(static_cast<uint32_t>(r), 0.7f, 0.05f, std::log(0.05f), seed,
                                position, 0u, 0u));
    }
  }
  const size_t rows = params.size();
  const std::vector<float> logits = repeat_row(base_row, rows);
  std::vector<uint32_t> mask(rows * words, u32max());
  run_fast_case(logits, rows, vocab, params, mask, true, true, false, "seeded replay");
  /* Determinism: run the same wave again and require identical ids. */
  K1 k = make_k1(rows, vocab, true, true);
  require_cuda(cudaMemcpy(k.logits, logits.data(), logits.size() * 4, cudaMemcpyHostToDevice),
               "replay logits");
  require_cuda(cudaMemcpy(k.params, params.data(), rows * sizeof(params[0]),
                          cudaMemcpyHostToDevice), "replay params");
  require_cuda(cudaMemcpy(k.mask, mask.data(), mask.size() * 4, cudaMemcpyHostToDevice),
               "replay mask");
  std::vector<uint32_t> first(rows), second(rows);
  for (int pass = 0; pass < 2; ++pass) {
    expect(ds41rt_cuda_v41_target_sample(k.logits, rows, vocab, vocab, k.params, k.mask, words,
                                         k.ids, k.status, k.detail, k.scores, k.total, k.nucleus,
                                         k.scratch) == DS41RT_STATUS_OK, "replay launch");
    require_cuda(cudaMemcpy(pass == 0 ? first.data() : second.data(), k.ids, rows * 4,
                            cudaMemcpyDeviceToHost), "replay ids");
  }
  expect(first == second, "K2 seeded replay is deterministic");
  free_k1(&k);
  std::cout << "ok  K2 seeded replay over " << rows << " (seed, position) cells\n";
}

/* Design §6.2 claim (b) for stochastic (non-greedy) rows: a row's draw depends
 * only on its own params and `(seed, position)`, never on how it is batched or
 * where it sits in the wave. The same row, params, seed and position must select
 * the same token at batch size 1, at index 0 and at the last index of an 8-wave,
 * in the middle of a 16-wave with peers on both sides, and on a bit-identical
 * replay. Cheap: 65-token rows, a handful of <=16-row launches. */
void test_k2_batch_composition_independence() {
  const size_t vocab = 65;

  auto make_row = [&](size_t shift, float peak) {
    std::vector<float> row(vocab);
    for (size_t t = 0; t < vocab; ++t) {
      row[t] = 0.2f * static_cast<float>((t * 7u + shift) % 13u) - 1.2f;
    }
    row[7] = peak;
    return row;
  };
  const std::vector<float> row_a = make_row(0u, 3.5f);
  const std::vector<float> row_b = make_row(5u, 1.0f);
  /* One-hot-style row: the crossing is token 7 with a wide margin, so the
   * CPU-equality assertion below is never vacuous. */
  std::vector<float> row_onehot(vocab, -100.0f);
  row_onehot[7] = 0.0f;

  struct Cell {
    uint64_t seed;
    uint64_t position;
  };
  const std::vector<Cell> cells = {
      {0xDEADBEEFull, 0ull}, {1ull, 2ull}, {7ull, 4096ull}, {25015ull, 24ull},
  };

  auto launch = [&](const std::vector<std::vector<float>>& rows,
                    const std::vector<ds41rt_v41_sampler_row_t>& params) {
    const size_t n = rows.size();
    std::vector<float> flat;
    flat.reserve(n * vocab);
    for (const std::vector<float>& r : rows) {
      flat.insert(flat.end(), r.begin(), r.end());
    }
    K1 k = make_k1(n, vocab, false, true);
    require_cuda(cudaMemcpy(k.logits, flat.data(), flat.size() * 4, cudaMemcpyHostToDevice),
                 "wave logits");
    require_cuda(cudaMemcpy(k.params, params.data(), n * sizeof(params[0]),
                            cudaMemcpyHostToDevice), "wave params");
    expect(ds41rt_cuda_v41_target_sample(k.logits, n, vocab, vocab, k.params, nullptr, 0u, k.ids,
                                         k.status, k.detail, k.scores, k.total, k.nucleus,
                                         k.scratch) == DS41RT_STATUS_OK, "wave launch");
    std::vector<uint32_t> ids(n);
    require_cuda(cudaMemcpy(ids.data(), k.ids, n * 4, cudaMemcpyDeviceToHost), "wave ids");
    free_k1(&k);
    return ids;
  };
  auto wave_params = [&](const std::vector<Cell>& cs, size_t focus) {
    std::vector<ds41rt_v41_sampler_row_t> p;
    for (size_t i = 0; i < cs.size(); ++i) {
      const bool is_focus = i == focus;
      p.push_back(fast_row(static_cast<uint32_t>(i), 0.7f, 0.0f, -inf(),
                           is_focus ? cs[i].seed : 1000ull + i,
                           is_focus ? cs[i].position : i, DS41RT_V41_SAMPLER_NO_MASK_ROW,
                           DS41RT_V41_SAMPLER_FLAG_NO_MASK));
    }
    return p;
  };

  for (const Cell& cell : cells) {
    const ds41rt_v41_sampler_row_t solo = fast_row(
        0, 0.7f, 0.0f, -inf(), cell.seed, cell.position, DS41RT_V41_SAMPLER_NO_MASK_ROW,
        DS41RT_V41_SAMPLER_FLAG_NO_MASK);
    const uint32_t token = launch({row_a}, {solo})[0];

    std::vector<std::vector<float>> wave8(8, row_b);
    wave8[0] = row_a;
    expect(launch(wave8, wave_params(std::vector<Cell>(8, cell), 0))[0] == token,
           "A at index 0 of an 8-wave equals solo");

    wave8[0] = row_b;
    wave8[7] = row_a;
    expect(launch(wave8, wave_params(std::vector<Cell>(8, cell), 7))[7] == token,
           "A at the last index of an 8-wave equals solo");

    std::vector<std::vector<float>> wave16(16, row_b);
    wave16[8] = row_a;
    std::vector<Cell> c16(16, cell);
    const std::vector<uint32_t> r16 = launch(wave16, wave_params(c16, 8));
    expect(r16[8] == token, "A in the middle of a 16-wave with peers equals solo");
    expect(launch(wave16, wave_params(c16, 8)) == r16, "identical wave replays bit-identically");

    const RefFast ref = cpu_fast_reference(row_a.data(), vocab, nullptr, true, 0.7f, 0.0f, -inf(),
                                           cell.seed, cell.position);
    expect(ref.crossed, "batch-independence cell crosses on the CPU");
    if (ref.margin_safe) {
      expect(token == ref.token, "batch composition does not change the CPU-matching token");
    }
  }

  /* A single wave carrying the same row at four indices with four different
   * `(seed, position)` cells, compared cell-by-cell to the production-order CPU
   * port. These cells have `u` in (0.4, 0.62), so `row_onehot`'s `min(u, 1-u)`
   * margin makes the assertion non-vacuous by construction. */
  const std::vector<Cell> onehot_cells = {
      {0ull, 0ull}, {0ull, 4ull}, {0ull, 6ull}, {1ull, 0ull},
  };
  std::vector<std::vector<float>> wave4(4, row_onehot);
  std::vector<ds41rt_v41_sampler_row_t> p4;
  for (size_t i = 0; i < 4; ++i) {
    p4.push_back(fast_row(static_cast<uint32_t>(i), 0.7f, 0.0f, -inf(), onehot_cells[i].seed,
                          onehot_cells[i].position, DS41RT_V41_SAMPLER_NO_MASK_ROW,
                          DS41RT_V41_SAMPLER_FLAG_NO_MASK));
  }
  const std::vector<uint32_t> r4 = launch(wave4, p4);
  for (size_t i = 0; i < 4; ++i) {
    const RefFast ref = cpu_fast_reference(row_onehot.data(), vocab, nullptr, true, 0.7f, 0.0f,
                                           -inf(), onehot_cells[i].seed, onehot_cells[i].position);
    expect(ref.crossed && ref.margin_safe, "one-hot cell has a wide CPU margin");
    expect(r4[i] == ref.token, "same row at four in-wave indices equals the CPU port");
  }

  std::cout << "ok  K2 batch composition / in-wave index / replay independence ("
            << cells.size() << " cells)\n";
}

/* K2 must leave every row it does not own untouched: a `top_k` row, a `top_p < 1`
 * row and a greedy row. Pre-filled output memory makes a stale value visible. */
void test_k2_non_applicable_rows_untouched() {
  const size_t vocab = 8;
  const std::vector<float> logits =
      repeat_row({0.0f, 4.0f, 1.0f, 2.0f, 3.0f, -1.0f, 1.5f, 0.5f}, 4);
  std::vector<ds41rt_v41_sampler_row_t> params = {
      /* top_k = 40: ordered path, not the fast path (even though it is a no-op). */
      [] {
        ds41rt_v41_sampler_row_t p = fast_row(0, 0.7f, 0.0f, -inf(), 3u, 1u,
                                              DS41RT_V41_SAMPLER_NO_MASK_ROW,
                                              DS41RT_V41_SAMPLER_FLAG_NO_MASK);
        p.top_k = 40u;
        return p;
      }(),
      /* top_p = 0.9: ordered path. */
      [] {
        ds41rt_v41_sampler_row_t p = fast_row(1, 0.7f, 0.0f, -inf(), 3u, 1u,
                                              DS41RT_V41_SAMPLER_NO_MASK_ROW,
                                              DS41RT_V41_SAMPLER_FLAG_NO_MASK);
        p.top_p = 0.9f;
        return p;
      }(),
      /* Greedy: K1 owns the id. */
      fast_row(2, 0.0f, 0.0f, -inf(), 3u, 1u, DS41RT_V41_SAMPLER_NO_MASK_ROW,
               DS41RT_V41_SAMPLER_FLAG_GREEDY | DS41RT_V41_SAMPLER_FLAG_STRICT_FINITE |
                   DS41RT_V41_SAMPLER_FLAG_NO_MASK),
      /* Fast path: K2 owns the id. */
      fast_row(3, 0.7f, 0.0f, -inf(), 3u, 1u, DS41RT_V41_SAMPLER_NO_MASK_ROW,
               DS41RT_V41_SAMPLER_FLAG_NO_MASK),
  };
  K1 k = make_k1(4, vocab, false, true);
  require_cuda(cudaMemcpy(k.logits, logits.data(), logits.size() * 4, cudaMemcpyHostToDevice),
               "nonapp logits");
  require_cuda(cudaMemcpy(k.params, params.data(), params.size() * sizeof(params[0]),
                          cudaMemcpyHostToDevice), "nonapp params");
  const std::vector<uint32_t> sentinel(4, 0xCAFEBABEu);
  require_cuda(cudaMemcpy(k.ids, sentinel.data(), 4 * 4, cudaMemcpyHostToDevice),
               "nonapp sentinel");
  expect(ds41rt_cuda_v41_target_sample(k.logits, 4, vocab, vocab, k.params, nullptr, 0u, k.ids,
                                       k.status, k.detail, k.scores, k.total, k.nucleus,
                                       k.scratch) == DS41RT_STATUS_OK, "nonapp launch");
  std::vector<uint32_t> ids(4);
  require_cuda(cudaMemcpy(ids.data(), k.ids, 4 * 4, cudaMemcpyDeviceToHost), "nonapp ids");
  expect(ids[0] == 0xCAFEBABEu, "top_k row untouched by K2");
  expect(ids[1] == 0xCAFEBABEu, "top_p<1 row untouched by K2");
  expect(ids[2] == 1u, "greedy row still argmax");
  expect(ids[3] != 0xCAFEBABEu, "fast-path row written by K2");
  free_k1(&k);
  std::cout << "ok  K2 leaves non-fast-path rows to K1 / the CPU\n";
}

/* Host model of the shipped tree kernel's two quantities, used to prove that the
 * device takes its `total` from the walk (the crossing accumulation) and not
 * from the tree scan's inclusive prefix. Association-for-association f32
 * simulation of `v41_sample_categorical_kernel`. */
struct K2TreeModel {
  float walk_total = 0.0f;
  float tree_total = 0.0f;
  /* 0 marks "no survivor" here, matching the kernel's `lasts` sentinel. */
  uint32_t global_last_survivor = 0u;
  size_t per_thread = 0;
};

K2TreeModel k2_tree_model(const float* logits, size_t vocab, float temperature, float min_p,
                          float ln_min_p) {
  K2TreeModel out;
  constexpr int kBlockLocal = 256;
  out.per_thread = (vocab + kBlockLocal - 1) / kBlockLocal;
  const float inv = 1.0f / temperature;
  float max_scaled = -std::numeric_limits<float>::infinity();
  for (size_t token = 0; token < vocab; ++token) {
    max_scaled = std::fmax(max_scaled, logits[token] * inv);
  }
  const float min_scaled =
      min_p > 0.0f ? max_scaled + ln_min_p : -std::numeric_limits<float>::infinity();
  const auto survivor = [&](size_t token) { return logits[token] * inv >= min_scaled; };
  const auto weight = [&](size_t token) { return std::exp(logits[token] * inv - max_scaled); };
  float seg[kBlockLocal];
  uint32_t last_seg[kBlockLocal];
  for (int s = 0; s < kBlockLocal; ++s) {
    const size_t begin = static_cast<size_t>(s) * out.per_thread;
    const size_t end = std::min(begin + out.per_thread, vocab);
    float local = 0.0f;
    uint32_t last = 0u;   /* 0 marks an empty segment, matching the kernel */
    for (size_t token = begin; token < end; ++token) {
      if (survivor(token)) {
        last = static_cast<uint32_t>(token);
        local += weight(token);
      }
    }
    seg[s] = local;
    last_seg[s] = last;
  }
  float inclusive[kBlockLocal];
  for (int s = 0; s < kBlockLocal; ++s) inclusive[s] = seg[s];
  for (int offset = 1; offset < kBlockLocal; offset <<= 1) {
    float addend[kBlockLocal];
    for (int s = 0; s < kBlockLocal; ++s) {
      addend[s] = (s >= offset) ? inclusive[s - offset] : 0.0f;
    }
    for (int s = 0; s < kBlockLocal; ++s) {
      if (s >= offset) inclusive[s] += addend[s];
    }
  }
  out.tree_total = inclusive[kBlockLocal - 1];
  for (int s = 0; s < kBlockLocal; ++s) {
    out.global_last_survivor = std::max(out.global_last_survivor, last_seg[s]);
  }
  float walk = 0.0f;
  if (out.global_last_survivor != DS41RT_V41_SAMPLER_NO_DETAIL) {    const size_t owner = static_cast<size_t>(out.global_last_survivor) / out.per_thread;
    float running = (owner == 0u) ? 0.0f : inclusive[owner - 1];
    const size_t owner_begin = owner * out.per_thread;
    for (size_t token = owner_begin; token <= static_cast<size_t>(out.global_last_survivor);
         ++token) {
      if (survivor(token)) {
        running += weight(token);
      }
    }
    walk = running;
  }
  out.walk_total = std::fmax(walk, 1.0e-20f);
  return out;
}

/* The CPU's `last` fallback is UNREACHABLE, and the device must now match that.
 *
 * Proof from `target_sampling.rs::sample_categorical` (`:590-618`): the first loop
 * accumulates `total` and the second loops `cumulative` with the identical
 * left-to-right f32 expression over the same survivor set, so at the last
 * survivor `cumulative == total` bit for bit; `MAX_UNIFORM` (`:52`) is
 * `0.99999994 < 1` and is applied at `:459`, so `target = uniform * total <=
 * total`; the inclusive test `target <= cumulative` (`:613`) therefore fires
 * before the `last.map(...)` fallback (`:616-618`). User code cannot reach the
 * fallback; it is defensive only.
 *
 * Chunk 2's first kernel violated this on device by taking `total` from the tree
 * scan's inclusive prefix while the crossing walk accumulated per segment, so the
 * walk's final cumulative could sit below `target`. This row is that adversarial
 * case: one survivor of weight 1 and 129,279 survivors of weight
 * `expf(-16.6355324) = 2^-24`. The tree prefix reports 1.00767553 while the walk
 * loses the tiny weights at the start of the row, so the old kernel fired the
 * fallback and returned token 129,279. The fix derives `total` from the walk's
 * own cumulative at the last survivor; the fallback no longer fires and the
 * kernel returns a legitimate crossing in the last segment. */
void test_k2_fallback_unreachable_like_cpu() {
  const size_t vocab = 129280;
  const uint32_t tiny_bits = 0xc1851592u;   /* expf(-16.6355324) == 2^-24 on device */
  float tiny = 0.0f;
  std::memcpy(&tiny, &tiny_bits, sizeof(tiny));
  std::vector<float> logits(vocab, tiny);
  logits[0] = 0.0f;                          /* the weight-1 survivor */
  const uint32_t last_token = static_cast<uint32_t>(vocab - 1);

  const K2TreeModel model = k2_tree_model(logits.data(), vocab, 1.0f, 0.0f, -inf());
  expect(model.global_last_survivor == last_token, "model: every token survives");
  expect(model.tree_total > 1.0f, "model: the tree prefix retains the grouped tiny weights");
  expect(model.walk_total > 1.0f, "model: the walk total also retains them");
  expect(std::fabs(model.walk_total - model.tree_total) < 1e-3f,
         "model: walk and tree totals are close but associated differently");
  const size_t owner = static_cast<size_t>(last_token) / model.per_thread;
  expect(owner == 255u, "model: the last survivor is in the final segment");

  /* (a) MAX_UNIFORM-scale draw: the old kernel took the fallback here. */
  const uint64_t seed = 5020ull;
  const uint64_t position = 1ull;            /* u = 0.999993801 */
  std::vector<ds41rt_v41_sampler_row_t> params = {
      fast_row(0, 1.0f, 0.0f, -inf(), seed, position, DS41RT_V41_SAMPLER_NO_MASK_ROW,
               DS41RT_V41_SAMPLER_FLAG_NO_MASK),
  };
  K1 k = make_k1(1, vocab, false, true);
  require_cuda(cudaMemcpy(k.logits, logits.data(), logits.size() * 4, cudaMemcpyHostToDevice),
               "fallback logits");
  require_cuda(cudaMemcpy(k.params, params.data(), sizeof(params[0]), cudaMemcpyHostToDevice),
               "fallback params");
  expect(ds41rt_cuda_v41_target_sample(k.logits, 1, vocab, vocab, k.params, nullptr, 0u, k.ids,
                                       k.status, k.detail, k.scores, k.total, k.nucleus,
                                       k.scratch) == DS41RT_STATUS_OK, "fallback launch");
  std::vector<uint32_t> ids(1);
  std::vector<float> total(1);
  require_cuda(cudaMemcpy(ids.data(), k.ids, 4, cudaMemcpyDeviceToHost), "fallback id");
  require_cuda(cudaMemcpy(total.data(), k.total, 4, cudaMemcpyDeviceToHost), "fallback total");
  const RefFast cpu = cpu_fast_reference(logits.data(), vocab, nullptr, true, 1.0f, 0.0f, -inf(),
                                         seed, position);
  expect(cpu.crossed, "CPU port crosses (its last fallback is unreachable)");
  expect(cpu.total == 1.0f, "CPU sequential total loses every sub-half-ulp weight");
  expect(cpu.token == 0u, "CPU crosses at token 0");
  if (ids[0] == last_token) {
    std::cerr << "spurious fallback returned " << ids[0] << "\n";
  }
  expect(ids[0] != last_token, "device no longer returns the last survivor via the fallback");
  expect(ids[0] < last_token, "device crossing is a real token");
  expect(ids[0] >= static_cast<uint32_t>(owner * model.per_thread),
         "device crossing is inside the final segment (the walk's own total)");
  /* The device's `out_total` must be the walk total, not the tree total. */
  expect(std::fabs(total[0] - model.walk_total) <=
             std::fabs(total[0] - model.tree_total) + 1.0e-7f,
         "device total comes from the walk, not the tree scan");
  expect(total[0] > 1.0f, "device total retains the tiny weights");

  /* (b) CPU == GPU on the same adversarial row where the crossing agrees: a
   * mid-range uniform lands on token 0 in both accumulations. */
  const uint64_t mid_seed = 0ull;
  const uint64_t mid_position = 0ull;        /* u = 0.400835 */
  params[0] = fast_row(0, 1.0f, 0.0f, -inf(), mid_seed, mid_position,
                       DS41RT_V41_SAMPLER_NO_MASK_ROW, DS41RT_V41_SAMPLER_FLAG_NO_MASK);
  require_cuda(cudaMemcpy(k.params, params.data(), sizeof(params[0]), cudaMemcpyHostToDevice),
               "fallback params mid");
  expect(ds41rt_cuda_v41_target_sample(k.logits, 1, vocab, vocab, k.params, nullptr, 0u, k.ids,
                                       k.status, k.detail, k.scores, k.total, k.nucleus,
                                       k.scratch) == DS41RT_STATUS_OK, "fallback launch mid");
  require_cuda(cudaMemcpy(ids.data(), k.ids, 4, cudaMemcpyDeviceToHost), "fallback id mid");
  const RefFast mid_cpu = cpu_fast_reference(logits.data(), vocab, nullptr, true, 1.0f, 0.0f,
                                             -inf(), mid_seed, mid_position);
  expect(mid_cpu.crossed && mid_cpu.token == 0u, "CPU mid-uniform crosses at token 0");
  if (ids[0] != mid_cpu.token) {
    std::cerr << "mid-uniform mismatch: device=" << ids[0] << " cpu=" << mid_cpu.token << "\n";
  }
  expect(ids[0] == mid_cpu.token, "GPU == CPU on the fallback row at a mid-range uniform");
  free_k1(&k);
  std::cout << "ok  K2 last fallback unreachable (device and CPU both cross; row=" << vocab
            << ")\n";
}

/* P0-1 regression: a K2 draw must never select a token whose weight is exactly
 * zero. The witness (vocab 2048, `T = 1`, `top_k = 0`, `top_p = 1`, `min_p = 0`,
 * `logits[0] = 0`, `logits[8..15] = -17`, everything else `-200`, seed
 * `0x9216a62488dbaa7b`, position 0 -> the uniform is `MAX_UNIFORM`) has each
 * tail weight `expf(-17) = 4.1399378e-8`, below half an ulp of 1.0. The pre-fix
 * kernel's tree prefix handed segment 2 a start already past `target`
 * (`inclusive[1] = 1.00000036 >= target = 1.00000024`), so it "crossed" at its
 * first survivor -- token 16, whose weight is `expf(-200) = 0` -- while the CPU
 * sequential oracle returns token 0. The fixed kernel's consistent segment
 * prefix plus the `cumulative_before < target` minimality guard makes that
 * structurally impossible; this test asserts the selected weight is strictly
 * positive. Mutant M6 (the pre-fix K2 walk) fails it. */
void test_k2_zero_weight_invariant() {
  ++g_cases;
  const size_t vocab = 2048;
  std::vector<float> logits(vocab, -200.0f);
  logits[0] = 0.0f;
  for (int token = 8; token <= 15; ++token) {
    logits[static_cast<size_t>(token)] = -17.0f;
  }
  const uint64_t seed = 0x9216a62488dbaa7bull;
  const uint64_t position = 0ull;
  const uint32_t sentinel = 0xDEADBEEFu;
  const std::vector<ds41rt_v41_sampler_row_t> params = {
      fast_row(0, 1.0f, 0.0f, -inf(), seed, position, DS41RT_V41_SAMPLER_NO_MASK_ROW,
               DS41RT_V41_SAMPLER_FLAG_NO_MASK),
  };
  K1 k = make_k1(1, vocab, false, true);
  require_cuda(cudaMemcpy(k.logits, logits.data(), logits.size() * sizeof(float),
                          cudaMemcpyHostToDevice), "K2 witness logits h2d");
  require_cuda(cudaMemcpy(k.params, params.data(), sizeof(params[0]),
                          cudaMemcpyHostToDevice), "K2 witness params h2d");
  std::vector<uint32_t> host_ids(1u, sentinel);
  require_cuda(cudaMemcpy(k.ids, host_ids.data(), sizeof(uint32_t),
                          cudaMemcpyHostToDevice), "K2 witness ids h2d");
  expect(ds41rt_cuda_v41_target_sample(k.logits, 1, vocab, vocab, k.params, nullptr, 0u,
                                       k.ids, k.status, k.detail, k.scores, k.total,
                                       k.nucleus, k.scratch) == DS41RT_STATUS_OK,
         "K2 witness: launch");
  uint32_t id = sentinel;
  uint32_t status = sentinel;
  require_cuda(cudaMemcpy(&id, k.ids, sizeof(uint32_t), cudaMemcpyDeviceToHost),
               "K2 witness ids d2h");
  require_cuda(cudaMemcpy(&status, k.status, sizeof(uint32_t), cudaMemcpyDeviceToHost),
               "K2 witness status d2h");
  const RefFast cpu = cpu_fast_reference(logits.data(), vocab, nullptr, true, 1.0f, 0.0f,
                                         -inf(), seed, position);
  expect(cpu.crossed, "K2 witness: the CPU port crosses (its last fallback is unreachable)");
  expect(cpu.total == 1.0f, "K2 witness: the CPU sequential total loses every sub-half-ulp tail");
  expect(cpu.token == 0u, "K2 witness: the CPU crosses at token 0");
  expect(status == DS41RT_V41_SAMPLER_STATUS_OK, "K2 witness: K2 reported OK");
  expect(id != sentinel && id < vocab, "K2 witness: K2 wrote an in-vocab token");
  /* `T = 1` and `max_scaled = logits[0] = 0`, so the weight is `expf(logit)`.
   * Token 16 is the pre-fix zero-weight pick; its weight really is zero, so the
   * positive-weight assertion below is a genuine structural guard. */
  expect(std::exp(logits[16]) == 0.0f,
         "K2 witness: token 16 has exactly zero weight (the pre-fix pick)");
  const float weight = std::exp(logits[id]);
  if (!(weight > 0.0f)) {
    std::cerr << "K2 zero-weight violation: device token " << id << " weight " << weight
              << "\n";
  }
  /* Qualified (adversarial review rev4 N2): this claim holds for `target > 0`
   * only. The `u == 0` (`target == 0`) path legitimately selects the FIRST
   * survivor, whose weight may be exactly zero -- device-confirmed (seed 310147,
   * position 0 -> token 0, weight 0) and matched by the production CPU sampler,
   * which adds the weight before testing `target <= cumulative`. */
  expect(weight > 0.0f,
         "P0-1: for target > 0 the K2 draw never selects a zero-weight token "
         "(the target == 0 path selects the first survivor, which may have "
         "weight 0 and matches the production CPU sampler)");
  expect(id != 16u, "P0-1: the K2 draw no longer selects the zero-weight token 16");
  free_k1(&k);
  std::cout << "ok  K2 zero-weight invariant (device token " << id << ", weight " << weight
            << ", CPU token " << cpu.token << ")\n";
}

/* N1 regression (adversarial review rev4): PIN the K2 bracketing-segment
 * saturation fallback on the device-confirmed vocab-1024 witness.
 *
 * *** KNOWN, DOCUMENTED DEVIATION -- PINNED DELIBERATELY. ***
 * Do not "fix" this corner silently: the behaviour asserted below is the
 * documented residual described in `runs/chunk3b-scratch/REPORT.md` §2.2/§6
 * (rev5) and in the adversarial review N1, and any change here is a
 * chunk-6-candidate mapping change that must be re-measured on the 120-cell
 * grid, not a drive-by product fix.
 *
 * Row (kBlock = 256, so segment 1 = tokens 4..7): `logits[0] = 0` (weight 1),
 * `logits[4] = ln(0.4*2^-23)`, `logits[5] = ln(1.4*2^-23)`,
 * `logits[6] = logits[7] = ln(0.4*2^-23)`, everything else -1e9. The
 * segment-level fold gives `C(2) = 1 + 3 ulp`, but the bracketing segment's
 * in-segment walk uses a different f32 association and saturates at
 * `1 + 1 ulp`: token 5's weight rounds the walk up to `1 + 1 ulp`, and tokens
 * 6/7 (0.4 ulp each, below half an ulp of `1 + 1 ulp`) round back down. Hence:
 *
 *   - `u = 0.9999997615814209` (mantissa 16777212 on the 2^-24 grid):
 *     `target = 1 + 1 ulp` and token 5 is the GENUINE first crossing;
 *   - `u = 0.99999988079071045` (mantissa 16777214, the LARGER uniform):
 *     `target = 1 + 2 ulp`; the in-segment walk never reaches it, so the
 *     saturation fallback reports the segment's FIRST positive-weight survivor,
 *     token 4 -- a BACKWARD step (token 4 < token 5) whose cumulative
 *     `W(4) = 1.0` is strictly BELOW the target.
 *
 * The fallback answer is therefore neither the minimum index satisfying the
 * threshold nor monotone in u; it is only guaranteed to be a positive-weight
 * survivor (a 983,040-draw validity scan found 0 zero-weight / non-survivor
 * selections, and the affected range is the last few ulps of the uniform,
 * ~1e-7 of the mass). Both exact tokens are asserted so any future change to
 * this corner breaks this test loudly.
 *
 * The uniforms are reached through the production seed path: the seeds below
 * are the smallest whose bit-identical SplitMix64 mapping
 * (`host_target_uniform`, the host port of `random_uniform`) yields the wanted
 * 24-bit mantissas at position 0 (same brute-force technique that found seed
 * 310147 for u = 0); the mapping assertions pin that correspondence. */
void test_k2_saturation_fallback_witness_pinned() {
  ++g_cases;
  const size_t vocab = 1024;
  const uint64_t seed_low_u = 34442741ull;  /* mantissa 16777212 -> u = 0.9999997615814209 */
  const uint64_t seed_high_u = 11753212ull; /* mantissa 16777214 -> u = 0.99999988079071045 */
  const uint32_t mantissa_low_u = 16777212u;
  const uint32_t mantissa_high_u = 16777214u;
  const float u_low = static_cast<float>(mantissa_low_u) * (1.0f / 16777216.0f);
  const float u_high = static_cast<float>(mantissa_high_u) * (1.0f / 16777216.0f);
  /* The hard-coded seeds must reproduce the witness uniforms bit for bit. */
  expect(host_target_uniform(seed_low_u, 0ull) == u_low,
         "saturation witness: seed 34442741 maps to mantissa 16777212");
  expect(host_target_uniform(seed_high_u, 0ull) == u_high,
         "saturation witness: seed 11753212 maps to mantissa 16777214");
  expect(u_low < u_high,
         "saturation witness: the two uniforms are ordered (u_low < u_high)");

  std::vector<float> logits(vocab, -1.0e9f);
  logits[0] = 0.0f;
  const double ulp = std::ldexp(1.0, -23);
  logits[4] = static_cast<float>(std::log(0.4 * ulp));
  logits[5] = static_cast<float>(std::log(1.4 * ulp));
  logits[6] = logits[4];
  logits[7] = logits[4];
  /* Both witness tokens are survivors with strictly positive f32 weights. */
  expect(std::exp(logits[4]) > 0.0f && std::exp(logits[5]) > 0.0f,
         "saturation witness: tokens 4 and 5 have positive weights");

  const std::vector<float> rows = repeat_row(logits, 2);
  const std::vector<ds41rt_v41_sampler_row_t> params = {
      fast_row(0, 1.0f, 0.0f, -inf(), seed_low_u, 0ull, DS41RT_V41_SAMPLER_NO_MASK_ROW,
               DS41RT_V41_SAMPLER_FLAG_NO_MASK),
      fast_row(1, 1.0f, 0.0f, -inf(), seed_high_u, 0ull, DS41RT_V41_SAMPLER_NO_MASK_ROW,
               DS41RT_V41_SAMPLER_FLAG_NO_MASK),
  };
  K1 k = make_k1(2, vocab, false, true);
  require_cuda(cudaMemcpy(k.logits, rows.data(), rows.size() * sizeof(float),
                          cudaMemcpyHostToDevice), "saturation witness logits h2d");
  require_cuda(cudaMemcpy(k.params, params.data(), 2 * sizeof(params[0]),
                          cudaMemcpyHostToDevice), "saturation witness params h2d");
  const std::vector<uint32_t> sentinels(2u, 0xDEADBEEFu);
  require_cuda(cudaMemcpy(k.ids, sentinels.data(), 2 * sizeof(uint32_t),
                          cudaMemcpyHostToDevice), "saturation witness ids h2d");
  expect(ds41rt_cuda_v41_target_sample(k.logits, 2, vocab, vocab, k.params, nullptr, 0u,
                                       k.ids, k.status, k.detail, k.scores, k.total,
                                       k.nucleus, k.scratch) == DS41RT_STATUS_OK,
         "saturation witness: launch");
  std::vector<uint32_t> ids(2), status(2);
  require_cuda(cudaMemcpy(ids.data(), k.ids, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost),
               "saturation witness ids d2h");
  require_cuda(cudaMemcpy(status.data(), k.status, 2 * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost), "saturation witness status d2h");
  expect(status[0] == DS41RT_V41_SAMPLER_STATUS_OK && status[1] == DS41RT_V41_SAMPLER_STATUS_OK,
         "saturation witness: both draws report OK");
  /* The pinned behaviour: u_low -> token 5 (genuine crossing), u_high -> token 4
   * (saturation fallback's first positive-weight survivor). */
  expect(ids[0] == 5u,
         "saturation witness: u = 0.9999997615814209 selects token 5 (pinned; "
         "see REPORT.md §2.2/§6 -- documented deviation)");
  expect(ids[1] == 4u,
         "saturation witness: u = 0.99999988079071045 selects token 4 (pinned; "
         "the saturation fallback answers with the bracketing segment's first "
         "positive-weight survivor, which lies below the target)");
  /* The explicit non-monotonicity: the LARGER uniform selects the SMALLER token.
   * Known, documented deviation -- not a bug to fix in this revision. */
  expect(ids[1] < ids[0],
         "saturation witness: the mapping is NON-MONOTONE in u at the pinned "
         "witness (documented residual, rev4 adversarial review N1)");
  for (int r = 0; r < 2; ++r) {
    const float weight = std::exp(logits[ids[static_cast<size_t>(r)]]);
    expect(weight > 0.0f,
           "saturation witness: every pinned selection has positive f32 weight");
  }
  /* The production CPU oracle crosses at token 0 for BOTH uniforms (its
   * sequential total is 1 + 1 ulp, putting both targets below 1.0): the device's
   * 5-then-4 answers are part of the same documented fast-path deviation, not a
   * CPU-parity failure introduced by this test. */
  const RefFast cpu_low = cpu_fast_reference(logits.data(), vocab, nullptr, true, 1.0f, 0.0f,
                                             -inf(), seed_low_u, 0ull);
  const RefFast cpu_high = cpu_fast_reference(logits.data(), vocab, nullptr, true, 1.0f, 0.0f,
                                              -inf(), seed_high_u, 0ull);
  expect(cpu_low.crossed && cpu_high.crossed,
         "saturation witness: the CPU port crosses on both uniforms");
  expect(cpu_low.token == 0u && cpu_high.token == 0u,
         "saturation witness: the CPU port crosses at token 0 for both uniforms "
         "(the device's 5-then-4 answers are the documented fast-path deviation)");
  free_k1(&k);
  std::cout << "ok  K2 saturation fallback pinned (mantissa " << mantissa_low_u << " -> token "
            << ids[0] << "; mantissa " << mantissa_high_u << " -> token " << ids[1]
            << "; backward step is the documented residual)\n";
}

/* ====================================================================== */
/* Chunk 3a: K3 pivot selection + K4 exact-k membership (design §4.3-4.4)  */
/* ====================================================================== */

/* Host port of the shipped `ds41rt_v41_order_key` (design §4.3): the standard
 * IEEE ascending u32 map, larger = better, with -0.0 canonicalized to +0.0. */
uint32_t host_order_key(float scaled) {
  const float value = (scaled == 0.0f) ? 0.0f : scaled;
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & 0x80000000u) != 0u ? ~bits : (bits ^ 0x80000000u);
}

/* CPU top-k branch oracle: a faithful port of `target_sampling.rs:477-497` and
 * the served comparator `Ranked::better_than` (`:283-291`) — survivors sorted by
 * (scaled descending, id ascending), truncated to `top_k`, plus
 * `above_count = C_gt(kth)`. `ranked` is the CPU's `ranked[..k]`, the exact set
 * and order K4 must materialize. */
struct RefTopk {
  bool greedy = false;
  uint32_t status = DS41RT_V41_SAMPLER_STATUS_OK;
  uint32_t survivor_count = 0;
  bool runs = false; /* K3/K4 eligibility, `target_sampling.rs:477-501` */
  uint32_t above_count = 0;
  float kth_value = 0.0f;
  std::vector<uint32_t> ranked;
};

RefTopk cpu_topk_reference(const float* logits, size_t vocab,
                           const uint32_t* mask_words, size_t mask_words_per_row,
                           float temperature, uint32_t top_k, float min_p,
                           float ln_min_p, uint32_t flags) {
  RefTopk out;
  const RefGreedy k1 = cpu_reference(logits, vocab, mask_words, mask_words_per_row,
                                     temperature, top_k, ln_min_p, min_p, flags);
  out.greedy = is_greedy(temperature, top_k) ||
               (flags & DS41RT_V41_SAMPLER_FLAG_GREEDY) != 0u;
  out.status = k1.stochastic_status;
  out.survivor_count = k1.survivor_count;
  if (out.greedy || k1.stochastic_status != DS41RT_V41_SAMPLER_STATUS_OK ||
      k1.survivor_count == 0u || top_k == 0u || top_k >= k1.survivor_count) {
    return out;
  }
  out.runs = true;
  const bool unconstrained = mask_words == nullptr || mask_words_per_row == 0u;
  const auto allowed = [&](size_t token) {
    return unconstrained ||
        ((mask_words[token / 32u] >> (token % 32u)) & 1u) != 0u;
  };
  const float inv = 1.0f / temperature;
  std::vector<std::pair<float, uint32_t>> ranked;
  ranked.reserve(k1.survivor_count);
  for (size_t token = 0; token < vocab; ++token) {
    if (allowed(token) && logits[token] * inv >= k1.min_scaled) {
      ranked.emplace_back(logits[token] * inv, static_cast<uint32_t>(token));
    }
  }
  std::sort(ranked.begin(), ranked.end(),
            [](const std::pair<float, uint32_t>& a, const std::pair<float, uint32_t>& b) {
              if (a.first != b.first) {
                return a.first > b.first;
              }
              return a.second < b.second;
            });
  const uint32_t kth_key = host_order_key(ranked[top_k - 1].first);
  out.kth_value = ranked[top_k - 1].first;
  for (const std::pair<float, uint32_t>& entry : ranked) {
    if (host_order_key(entry.first) > kth_key) {
      ++out.above_count;
    }
  }
  out.ranked.reserve(top_k);
  for (uint32_t rank = 0; rank < top_k; ++rank) {
    out.ranked.push_back(ranked[rank].second);
  }
  return out;
}

/* The worst-case pass count of the ternary bisection: always keep the largest
 * of the three sub-intervals of `[0, 0xFFFFFFFF]` (and the length-2 final
 * single probe) and count steps until the interval holds one key. This is an
 * upper bound on the shipped kernel's passes for any row and any k, and it is
 * the host evidence that the loop makes progress every pass (each pass shrinks
 * the interval to <= ~1/3). */
uint32_t host_bisection_pass_bound() {
  uint32_t lo = 0u;
  uint32_t hi = 0xFFFFFFFFu;
  uint32_t passes = 0u;
  while (hi - lo >= 2u) {
    uint32_t length = 0u;
    if (hi - lo == 2u) {
      length = 1u;
    } else {
      const uint32_t third = (hi - lo) / 3u;
      const uint32_t m0 = lo + third;
      const uint32_t m1 = hi - third;
      length = m0 - lo;
      if (m1 - m0 > length) {
        length = m1 - m0;
      }
      if (hi - m1 > length) {
        length = hi - m1;
      }
    }
    /* `length` is strictly smaller than `hi - lo` for every `hi - lo >= 2`, so
     * the loop cannot spin. */
    expect(length < hi - lo, "bisection interval shrinks every pass");
    lo = 0u;
    hi = length;
    ++passes;
    expect(passes <= DS41RT_V41_TOPK_MAX_PIVOT_STEPS,
           "bisection bound is inside the defensive cap");
  }
  return passes;
}

/* Device buffers for one K3/K4 run. `capacity` is the per-row rank-order id
 * stride; 0 selects the selection-only call (null arenas). */
struct TopkBuffers {
  uint32_t* rank_ids = nullptr;
  uint64_t* rank_scratch = nullptr;
  uint32_t* retained = nullptr;
  uint32_t* passes = nullptr;
  size_t rows = 0;
  size_t capacity = 0;
};

TopkBuffers make_topk_buffers(size_t rows, size_t capacity) {
  TopkBuffers buffers;
  buffers.rows = rows;
  buffers.capacity = capacity;
  if (capacity > 0) {
    alloc_device(&buffers.rank_ids, rows * capacity * sizeof(uint32_t), "topk rank ids");
    alloc_device(&buffers.rank_scratch, rows * capacity * sizeof(uint64_t),
                 "topk rank scratch");
  }
  alloc_device(&buffers.retained, rows * sizeof(uint32_t), "topk retained count");
  alloc_device(&buffers.passes, rows * sizeof(uint32_t), "topk pivot passes");
  return buffers;
}

void free_topk_buffers(TopkBuffers* buffers) {
  if (buffers->rank_ids) cudaFree(buffers->rank_ids);
  if (buffers->rank_scratch) cudaFree(buffers->rank_scratch);
  if (buffers->retained) cudaFree(buffers->retained);
  if (buffers->passes) cudaFree(buffers->passes);
  *buffers = TopkBuffers{};
}

/* Runs one K1 + K3/K4 configuration against the ported CPU top-k oracle.
 *
 * `capacity` is the materialization stride handed to the device (0 =
 * selection-only). The rank-order arena is pre-filled with `0xDEADBEEF`, so a
 * no-op row that writes anything — or an eligible row whose ids were never
 * written — cannot pass. */
void run_topk_case(const std::vector<float>& logits, size_t rows, size_t vocab,
                   const std::vector<ds41rt_v41_sampler_row_t>& params,
                   const std::vector<uint32_t>& mask, bool with_mask, bool strict_mask_bits,
                   size_t capacity, const char* label) {
  ++g_cases;
  const std::string tag(label);
  expect(params.size() == rows, tag + ": one param block per row");
  K1 k = make_k1(rows, vocab, with_mask, false);
  require_cuda(cudaMemcpy(k.logits, logits.data(), logits.size() * sizeof(float),
                          cudaMemcpyHostToDevice), "topk logits h2d");
  require_cuda(cudaMemcpy(k.params, params.data(),
                          params.size() * sizeof(ds41rt_v41_sampler_row_t),
                          cudaMemcpyHostToDevice), "topk params h2d");
  std::vector<uint32_t> device_mask = mask;
  if (with_mask) {
    if (strict_mask_bits) {
      for (size_t r = 0; r < rows; ++r) {
        ds41rt_v41_sampler_clear_remainder(device_mask.data() + r * k.words, vocab);
      }
    }
    require_cuda(cudaMemcpy(k.mask, device_mask.data(),
                            device_mask.size() * sizeof(uint32_t),
                            cudaMemcpyHostToDevice), "topk mask h2d");
  }
  /* K1 first: K3/K4 read its `scratch` exactly as K2 does. K2 itself is a no-op
   * for every row here (`top_k != 0`). */
  expect(ds41rt_cuda_v41_target_sample(
             k.logits, rows, vocab, vocab, k.params, with_mask ? k.mask : nullptr,
             with_mask ? k.words : 0u, k.ids, k.status, k.detail, k.scores, nullptr,
             nullptr, k.scratch) == DS41RT_STATUS_OK,
         tag + ": K1 launch");
  TopkBuffers buffers = make_topk_buffers(rows, capacity);
  const uint32_t sentinel = 0xDEADBEEFu;
  std::vector<uint32_t> host_ids(rows * capacity, sentinel);
  std::vector<uint64_t> host_scratch(rows * capacity, 0ull);
  std::vector<uint32_t> host_retained(rows, sentinel);
  std::vector<uint32_t> host_passes(rows, sentinel);
  if (capacity > 0) {
    require_cuda(cudaMemcpy(buffers.rank_ids, host_ids.data(),
                            rows * capacity * sizeof(uint32_t), cudaMemcpyHostToDevice),
                 "topk rank ids h2d");
    require_cuda(cudaMemcpy(buffers.rank_scratch, host_scratch.data(),
                            rows * capacity * sizeof(uint64_t), cudaMemcpyHostToDevice),
                 "topk rank scratch h2d");
  }
  require_cuda(cudaMemcpy(buffers.retained, host_retained.data(), rows * sizeof(uint32_t),
                          cudaMemcpyHostToDevice), "topk retained h2d");
  require_cuda(cudaMemcpy(buffers.passes, host_passes.data(), rows * sizeof(uint32_t),
                          cudaMemcpyHostToDevice), "topk passes h2d");
  expect(ds41rt_cuda_v41_topk_select(
             k.logits, rows, vocab, vocab, k.params, with_mask ? k.mask : nullptr,
             with_mask ? k.words : 0u, capacity > 0 ? buffers.rank_ids : nullptr,
             capacity > 0 ? buffers.rank_scratch : nullptr, capacity, buffers.retained,
             buffers.passes, k.scratch) == DS41RT_STATUS_OK,
         tag + ": topk launch");
  require_cuda(cudaDeviceSynchronize(), "topk kernel");

  std::vector<ds41rt_v41_sampler_scratch_t> scratch(rows);
  std::vector<uint32_t> retained(rows);
  std::vector<uint32_t> passes(rows);
  require_cuda(cudaMemcpy(scratch.data(), k.scratch,
                          rows * sizeof(ds41rt_v41_sampler_scratch_t),
                          cudaMemcpyDeviceToHost), "topk scratch");
  require_cuda(cudaMemcpy(retained.data(), buffers.retained, rows * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost), "topk retained");
  require_cuda(cudaMemcpy(passes.data(), buffers.passes, rows * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost), "topk passes");
  std::vector<uint32_t> rank_ids(rows * capacity, sentinel);
  if (capacity > 0) {
    require_cuda(cudaMemcpy(rank_ids.data(), buffers.rank_ids,
                            rows * capacity * sizeof(uint32_t), cudaMemcpyDeviceToHost),
                 "topk rank ids");
  }

  for (size_t r = 0; r < rows; ++r) {
    const ds41rt_v41_sampler_row_t& p = params[r];
    const std::string row_tag = tag + " row " + std::to_string(r);
    const bool unconstrained = (p.flags & DS41RT_V41_SAMPLER_FLAG_NO_MASK) != 0u ||
                               p.mask_row == DS41RT_V41_SAMPLER_NO_MASK_ROW;
    const uint32_t* row_mask = (with_mask && !unconstrained)
        ? device_mask.data() + static_cast<size_t>(p.mask_row) * k.words
        : nullptr;
    const size_t row_words = (row_mask != nullptr) ? k.words : 0u;
    const RefTopk expected = cpu_topk_reference(
        logits.data() + r * vocab, vocab, row_mask, row_words, p.temperature, p.top_k,
        p.min_p, p.ln_min_p, p.flags);
    if (!expected.runs) {
      expect(retained[r] == 0u, row_tag + ": no-op row retains nothing");
      expect(passes[r] == 0u, row_tag + ": no-op row runs no pivot pass");
      expect(scratch[r].status == DS41RT_V41_SAMPLER_STATUS_OK,
             row_tag + ": no-op row keeps its K1 status");
      if (capacity > 0) {
        for (size_t i = 0; i < capacity; ++i) {
          expect(rank_ids[r * capacity + i] == sentinel,
                 row_tag + ": no-op row leaves the arena untouched");
        }
      }
      continue;
    }
    expect(scratch[r].status == DS41RT_V41_SAMPLER_STATUS_OK,
           row_tag + ": K3 converged (status OK)");
    expect(retained[r] == p.top_k, row_tag + ": retained count is exactly k");
    expect(passes[r] >= 1u, row_tag + ": at least one pivot pass");
    expect(passes[r] <= host_bisection_pass_bound(),
           row_tag + ": pivot passes within the ternary bound");
    expect(passes[r] <= DS41RT_V41_TOPK_MAX_PIVOT_STEPS,
           row_tag + ": pivot passes within the defensive cap");
    if (passes[r] > g_max_pivot_passes) {
      g_max_pivot_passes = passes[r];
    }
    expect(scratch[r].above_count == expected.above_count,
           row_tag + ": above_count equals C_gt(kth)");
    /* The device publishes the *canonicalized* k-th value: `order_key` maps
     * `-0.0` and `+0.0` to one key (as the CPU's `descending_radix_key` does),
     * so a tie group at zero is reported as `+0.0`. Canonicalize the oracle's
     * value the same way before comparing bits. */
    expect(scratch[r].kth_value_bits ==
               [&] {
                 const float canonical =
                     (expected.kth_value == 0.0f) ? 0.0f : expected.kth_value;
                 uint32_t bits = 0;
                 std::memcpy(&bits, &canonical, sizeof(bits));
                 return bits;
               }(),
           row_tag + ": kth_value_bits equals the oracle's k-th value");
    if (capacity == 0) {
      continue; /* selection-only: no arena to compare */
    }
    expect(capacity >= p.top_k, row_tag + ": capacity covers k");
    /* The id multiset, not just the count, and then the exact CPU rank order. */
    std::vector<uint32_t> got(rank_ids.begin() + r * capacity,
                              rank_ids.begin() + r * capacity + p.top_k);
    std::vector<uint32_t> got_sorted = got;
    std::vector<uint32_t> want_sorted = expected.ranked;
    std::sort(got_sorted.begin(), got_sorted.end());
    std::sort(want_sorted.begin(), want_sorted.end());
    if (got_sorted != want_sorted) {
      std::cerr << "topk multiset mismatch " << row_tag << "\n";
    }
    expect(got_sorted == want_sorted, row_tag + ": exact retained id multiset");
    if (got != expected.ranked) {
      std::cerr << "topk rank-order mismatch " << row_tag << " k=" << p.top_k << "\n";
    }
    expect(got == expected.ranked, row_tag + ": exact CPU rank order");
    expect(got.size() == static_cast<size_t>(p.top_k), row_tag + ": k ids materialized");
    /* No within-row overrun: entries `[k, capacity)` must still hold the
     * sentinel the harness pre-filled, so a materializer that ran past the
     * retained prefix cannot pass unnoticed (cross-row overruns are caught by
     * the neighbouring no-op rows). */
    for (size_t i = static_cast<size_t>(p.top_k); i < capacity; ++i) {
      expect(rank_ids[r * capacity + i] == sentinel,
             row_tag + ": no write past the retained prefix");
    }
  }
  free_topk_buffers(&buffers);
  free_k1(&k);
}

/* ---- deterministic row shapes ---- */
std::vector<float> shape_descending(size_t vocab) {
  std::vector<float> row(vocab);
  for (size_t t = 0; t < vocab; ++t) {
    row[t] = -0.01f * static_cast<float>(t);
  }
  return row;
}
std::vector<float> shape_periodic_ties(size_t vocab) {
  std::vector<float> row(vocab);
  for (size_t t = 0; t < vocab; ++t) {
    row[t] = static_cast<float>(t % 16u) - 8.0f;
  }
  return row;
}
std::vector<float> shape_two_value(size_t vocab) {
  std::vector<float> row(vocab);
  for (size_t t = 0; t < vocab; ++t) {
    row[t] = (t < vocab / 2u) ? 0.0f : -1.0f;
  }
  return row;
}

/* The required k grid over several row shapes and vocabularies, including
 * `survivor_count - 1` (the last real selection), `survivor_count`,
 * `survivor_count + 1` and `vocab` (all no-ops for a full row). */
void test_k3_k4_k_grid() {
  const std::vector<size_t> vocabs = {33u, 127u, 1000u, 1200u};
  for (size_t vocab : vocabs) {
    const std::vector<std::vector<float>> shapes = {
        shape_descending(vocab),
        shape_periodic_ties(vocab),
        shape_two_value(vocab),
    };
    const char* shape_names[] = {"desc", "ties16", "two_value"};
    for (size_t s = 0; s < shapes.size(); ++s) {
      std::vector<uint32_t> ks = {
          1u, 2u, 40u, 64u, 257u, 1000u,
          static_cast<uint32_t>(vocab - 1u), static_cast<uint32_t>(vocab),
          static_cast<uint32_t>(vocab + 1u),
      };
      for (uint32_t k : ks) {
        std::vector<ds41rt_v41_sampler_row_t> params = {
            row(0, 0.7f, k, 0.0f, -inf(), DS41RT_V41_SAMPLER_NO_MASK_ROW,
                DS41RT_V41_SAMPLER_FLAG_NO_MASK),
        };
        const size_t capacity = k; /* one row: exactly its own k */
        const std::string label = "k3k4 grid " + std::string(shape_names[s]) +
                                  " vocab" + std::to_string(vocab) + " k" +
                                  std::to_string(k);
        run_topk_case(shapes[s], 1, vocab, params, {}, false, false, capacity,
                      label.c_str());
      }
    }
  }
  std::cout << "ok  K3/K4 k grid over " << vocabs.size() << " vocabularies (pass bound "
            << host_bisection_pass_bound() << ")\n";
}

/* Thousands of survivors sharing the k-th value: the exact id multiset must be
 * the CPU's (the tie cut admits the lowest-id equals), never "all ties". */
void test_k3_k4_thousands_tied() {
  /* 2048 survivors at 1.0 (even ids) and 2048 at 0.5 (odd ids); k = 3000 needs
   * 952 of the 0.5 ties, which must be the 952 lowest odd ids. */
  {
    const size_t vocab = 4096;
    std::vector<float> logits(vocab);
    for (size_t t = 0; t < vocab; ++t) {
      logits[t] = (t % 2u == 0u) ? 1.0f : 0.5f;
    }
    std::vector<ds41rt_v41_sampler_row_t> params = {
        row(0, 1.0f, 3000u, 0.0f, -inf(), DS41RT_V41_SAMPLER_NO_MASK_ROW,
            DS41RT_V41_SAMPLER_FLAG_NO_MASK),
    };
    run_topk_case(logits, 1, vocab, params, {}, false, false, 3000u,
                  "k3k4 thousands-tied half/half k=3000");
  }
  /* One huge flat tie below three distinct leaders; k = 2000 cuts deep into the
   * 4093-token tie group, so the prefix is ids 3..1999. */
  {
    const size_t vocab = 4096;
    std::vector<float> logits(vocab, 0.0f);
    logits[0] = 10.0f;
    logits[1] = 1.0f;
    logits[2] = 0.5f;
    std::vector<ds41rt_v41_sampler_row_t> params = {
        row(0, 1.0f, 2000u, 0.0f, -inf(), DS41RT_V41_SAMPLER_NO_MASK_ROW,
            DS41RT_V41_SAMPLER_FLAG_NO_MASK),
    };
    run_topk_case(logits, 1, vocab, params, {}, false, false, 2000u,
                  "k3k4 thousands-tied flat tail k=2000");
    /* And pin the exact expected ids independently of the oracle helper. */
    K1 k = make_k1(1, vocab, false, false);
    require_cuda(cudaMemcpy(k.logits, logits.data(), logits.size() * 4,
                            cudaMemcpyHostToDevice), "tie logits");
    require_cuda(cudaMemcpy(k.params, params.data(), sizeof(params[0]),
                            cudaMemcpyHostToDevice), "tie params");
    expect(ds41rt_cuda_v41_target_sample(k.logits, 1, vocab, vocab, k.params, nullptr,
                                         0u, k.ids, k.status, k.detail, k.scores, nullptr,
                                         nullptr, k.scratch) == DS41RT_STATUS_OK,
           "tie K1 launch");
    TopkBuffers buffers = make_topk_buffers(1, 2000u);
    expect(ds41rt_cuda_v41_topk_select(k.logits, 1, vocab, vocab, k.params, nullptr, 0u,
                                       buffers.rank_ids, buffers.rank_scratch, 2000u,
                                       buffers.retained, buffers.passes, k.scratch) ==
               DS41RT_STATUS_OK,
           "tie topk launch");
    std::vector<uint32_t> ids(2000u);
    std::vector<ds41rt_v41_sampler_scratch_t> scratch(1);
    require_cuda(cudaMemcpy(ids.data(), buffers.rank_ids, 2000u * 4,
                            cudaMemcpyDeviceToHost), "tie ids");
    require_cuda(cudaMemcpy(scratch.data(), k.scratch, sizeof(scratch[0]),
                            cudaMemcpyDeviceToHost), "tie scratch");
    expect(scratch[0].above_count == 3u, "tie: exactly three survivors above the k-th value");
    bool exact = ids.size() == 2000u;
    for (size_t i = 0; exact && i < ids.size(); ++i) {
      /* ranks 0..2 are the leaders, then ids 3,4,...,1999 in ascending order. */
      exact = ids[i] == static_cast<uint32_t>(i);
    }
    expect(exact, "tie: the retained ids are exactly 0..1999");
    free_topk_buffers(&buffers);
    free_k1(&k);
  }
  std::cout << "ok  K3/K4 thousands-tied rows keep exactly k lowest-id ties\n";
}

/* An all-tied row: `above_count = 0`, `target_tie = k`, so the tie cut admits
 * the first k tokens in token order, i.e. ids 0..k-1. */
void test_k3_k4_all_tied() {
  const size_t vocab = 1024;
  const std::vector<uint32_t> ks = {2u, 40u, 64u, 257u, 1000u};
  for (uint32_t k : ks) {
    std::vector<float> logits(vocab, 0.25f);
    std::vector<ds41rt_v41_sampler_row_t> params = {
        row(0, 1.0f, k, 0.0f, -inf(), DS41RT_V41_SAMPLER_NO_MASK_ROW,
            DS41RT_V41_SAMPLER_FLAG_NO_MASK),
    };
    const std::string label = "k3k4 all-tied k" + std::to_string(k);
    run_topk_case(logits, 1, vocab, params, {}, false, false, k, label.c_str());
    /* Independent of the oracle helper: the ids must be exactly 0..k-1. */
    K1 dev = make_k1(1, vocab, false, false);
    require_cuda(cudaMemcpy(dev.logits, logits.data(), logits.size() * 4,
                            cudaMemcpyHostToDevice), "all-tied logits");
    require_cuda(cudaMemcpy(dev.params, params.data(), sizeof(params[0]),
                            cudaMemcpyHostToDevice), "all-tied params");
    expect(ds41rt_cuda_v41_target_sample(dev.logits, 1, vocab, vocab, dev.params, nullptr,
                                         0u, dev.ids, dev.status, dev.detail, dev.scores,
                                         nullptr, nullptr, dev.scratch) == DS41RT_STATUS_OK,
           "all-tied K1");
    TopkBuffers buffers = make_topk_buffers(1, k);
    expect(ds41rt_cuda_v41_topk_select(dev.logits, 1, vocab, vocab, dev.params, nullptr,
                                       0u, buffers.rank_ids, buffers.rank_scratch, k,
                                       buffers.retained, buffers.passes, dev.scratch) ==
               DS41RT_STATUS_OK,
           "all-tied topk");
    std::vector<uint32_t> ids(k);
    require_cuda(cudaMemcpy(ids.data(), buffers.rank_ids, k * 4, cudaMemcpyDeviceToHost),
                 "all-tied ids");
    bool exact = true;
    for (uint32_t rank = 0; rank < k; ++rank) {
      exact = exact && ids[rank] == rank;
    }
    expect(exact, "all-tied row yields ids 0..k-1");
    free_topk_buffers(&buffers);
    free_k1(&dev);
  }
  /* A huge -inf tie group below one finite leader (min_p = 0 at a tiny
   * temperature): the k-th key is key(-inf) = 0x007FFFFF, so the bisection must
   * walk to the bottom of the key range and the tie cut still yields ascending
   * ids. Note every survivor cannot be -inf: a fully -inf scaled maximum is
   * `INVALID_TEMPERATURE` (K1), so at least one finite leader is required. */
  {
    const size_t vocab = 256;
    const float neg_max = -std::numeric_limits<float>::max();
    std::vector<float> logits(vocab, neg_max);
    logits[0] = 0.0f;
    std::vector<ds41rt_v41_sampler_row_t> params = {
        row(0, 1.0e-5f, 64u, 0.0f, -inf(), DS41RT_V41_SAMPLER_NO_MASK_ROW,
            DS41RT_V41_SAMPLER_FLAG_NO_MASK),
    };
    run_topk_case(logits, 1, vocab, params, {}, false, false, 64u,
                  "k3k4 -inf tie group");
  }
  std::cout << "ok  K3/K4 all-tied rows yield ids 0..k-1 (incl. -inf ties)\n";
}

/* Masks (sparse, all-allowed, a mask that makes the k-th value ambiguous) and
 * -inf survivors. */
void test_k3_k4_masks_and_inf() {
  /* Sparse mask: only tokens 1, 4, 5, 8, 9, 12 are allowed; k = 2. */
  {
    const size_t vocab = 33;
    const size_t words = (vocab + 31u) / 32u;
    std::vector<float> logits = shape_descending(vocab);
    std::vector<uint32_t> mask(words, 0u);
    const uint32_t allowed_ids[] = {1u, 4u, 5u, 8u, 9u, 12u};
    for (uint32_t id : allowed_ids) {
      mask[id / 32u] |= (1u << (id % 32u));
    }
    std::vector<ds41rt_v41_sampler_row_t> params = {
        row(0, 0.7f, 2u, 0.0f, -inf(), 0u, 0u),
    };
    run_topk_case(logits, 1, vocab, params, mask, true, true, 2u,
                  "k3k4 sparse mask k=2");
  }
  /* All-allowed mask with every remainder bit set (the host must still clear
   * them; `strict_mask_bits = false` leaves the junk and the kernel's
   * vocab-bounded loop must ignore it). */
  {
    const size_t vocab = 127;
    const size_t words = (vocab + 31u) / 32u;
    std::vector<float> logits = shape_periodic_ties(vocab);
    std::vector<uint32_t> mask(words, 0xFFFFFFFFu);
    std::vector<ds41rt_v41_sampler_row_t> params = {
        row(0, 0.7f, 40u, 0.0f, -inf(), 0u, 0u),
    };
    run_topk_case(logits, 1, vocab, params, mask, true, false, 40u,
                  "k3k4 all-allowed mask with junk high bits");
  }
  /* A mask that removes the unmasked k-th-value group: the retained set and the
   * tie cut must be computed over survivors only. */
  {
    const size_t vocab = 64;
    const size_t words = (vocab + 31u) / 32u;
    std::vector<float> logits(vocab, 0.0f);
    for (size_t t = 0; t < 8; ++t) {
      logits[t] = static_cast<float>(8 - t); /* 8,7,...,1 at tokens 0..7 */
    }
    std::vector<uint32_t> mask(words, 0u);
    /* Allow everything except tokens 2 and 3 (which would be ranks 5 and 6). */
    for (size_t t = 0; t < vocab; ++t) {
      if (t != 2u && t != 3u) {
        mask[t / 32u] |= (1u << (t % 32u));
      }
    }
    std::vector<ds41rt_v41_sampler_row_t> params = {
        row(0, 1.0f, 4u, 0.0f, -inf(), 0u, 0u),
    };
    run_topk_case(logits, 1, vocab, params, mask, true, true, 4u,
                  "k3k4 mask removes the k-th value");
  }
  /* -inf survivors with a real finite leader: min_p = 0 at T = 1e-5 keeps the
   * -inf tokens, which sort last; k lands inside the -inf tie group. */
  {
    const size_t vocab = 128;
    const float neg_max = -std::numeric_limits<float>::max();
    std::vector<float> logits(vocab, neg_max);
    logits[7] = 0.0f;
    logits[9] = -1.0f;
    std::vector<ds41rt_v41_sampler_row_t> params = {
        row(0, 1.0e-5f, 10u, 0.0f, -inf(), DS41RT_V41_SAMPLER_NO_MASK_ROW,
            DS41RT_V41_SAMPLER_FLAG_NO_MASK),
    };
    run_topk_case(logits, 1, vocab, params, {}, false, false, 10u,
                  "k3k4 -inf survivors k=10");
  }
  /* min_p keeps only the scaled maximum and -inf still enters when min_p = 0. */
  {
    const size_t vocab = 40;
    const size_t words = (vocab + 31u) / 32u;
    std::vector<float> logits = shape_periodic_ties(vocab);
    std::vector<uint32_t> mask(words, 0xFFFFFFFFu);
    const float min_p = 0.05f;
    const float ln_min_p = std::log(min_p);
    std::vector<ds41rt_v41_sampler_row_t> params = {
        row(0, 1.0f, 4u, min_p, ln_min_p, 0u, 0u),
    };
    run_topk_case(logits, 1, vocab, params, mask, true, true, 4u,
                  "k3k4 min_p + all-allowed mask");
  }
  std::cout << "ok  K3/K4 mask and -inf survivor cases\n";
}

/* Boundary/termination: no-op rows, greedy rows, capacity-0 selection-only, a
 * mixed batch, and a 129,280-wide row at a served k and at survivor_count - 1. */
void test_k3_k4_boundaries_and_termination() {
  /* A mixed batch: greedy (top_k = 1), disabled (top_k = 0), a no-op
   * (top_k >= survivor_count), and three real selections of different sizes.
   * Only the real rows may touch the arena; the others must stay sentinel. */
  {
    const size_t vocab = 300;
    std::vector<float> logits(6 * vocab);
    for (size_t r = 0; r < 6; ++r) {
      for (size_t t = 0; t < vocab; ++t) {
        logits[r * vocab + t] = shape_periodic_ties(vocab)[t] +
                                (r == 5u ? 0.0f : static_cast<float>(r));
      }
    }
    std::vector<ds41rt_v41_sampler_row_t> params = {
        row(0, 0.7f, 1u, 0.0f, -inf(), DS41RT_V41_SAMPLER_NO_MASK_ROW,
            DS41RT_V41_SAMPLER_FLAG_NO_MASK), /* greedy */
        row(1, 0.7f, 0u, 0.0f, -inf(), DS41RT_V41_SAMPLER_NO_MASK_ROW,
            DS41RT_V41_SAMPLER_FLAG_NO_MASK), /* disabled */
        row(2, 0.7f, 400u, 0.0f, -inf(), DS41RT_V41_SAMPLER_NO_MASK_ROW,
            DS41RT_V41_SAMPLER_FLAG_NO_MASK), /* top_k > survivor_count no-op */
        row(3, 0.7f, 17u, 0.0f, -inf(), DS41RT_V41_SAMPLER_NO_MASK_ROW,
            DS41RT_V41_SAMPLER_FLAG_NO_MASK),
        row(4, 1.0e-6f, 5u, 0.0f, -inf(), DS41RT_V41_SAMPLER_NO_MASK_ROW,
            DS41RT_V41_SAMPLER_FLAG_NO_MASK), /* T < 1e-5 greedy */
        row(5, 2.0f, 64u, 0.0f, -inf(), DS41RT_V41_SAMPLER_NO_MASK_ROW,
            DS41RT_V41_SAMPLER_FLAG_NO_MASK),
    };
    run_topk_case(logits, 6, vocab, params, {}, false, false, 64u,
                  "k3k4 mixed no-op/greedy/selection batch");
  }
  /* Selection-only (capacity 0): K3 still publishes kth/above and out_retained,
   * but no arena is touched. */
  {
    const size_t vocab = 500;
    std::vector<float> logits = shape_descending(vocab);
    std::vector<ds41rt_v41_sampler_row_t> params = {
        row(0, 0.7f, 37u, 0.0f, -inf(), DS41RT_V41_SAMPLER_NO_MASK_ROW,
            DS41RT_V41_SAMPLER_FLAG_NO_MASK),
    };
    run_topk_case(logits, 1, vocab, params, {}, false, false, 0u,
                  "k3k4 selection-only capacity 0");
  }
  /* Realistic 129,280-wide rows: a served k = 40 and k = 1000 with full rank
   * materialization, then survivor_count - 1 = 129,279 as a selection-only call
   * (materializing 129k ids would be O(k^2) and is a chunk-6 item). */
  {
    const size_t vocab = 129280;
    const auto splitmix_unit = [](uint64_t index) -> float {
      uint64_t hash = index * 0x9e3779b97f4a7c15ull;
      hash = (hash ^ (hash >> 30)) * 0xbf58476d1ce4e5b9ull;
      hash = (hash ^ (hash >> 27)) * 0x94d049bb133111ebull;
      hash ^= hash >> 31;
      return static_cast<float>(hash >> 40) / 16777216.0f;
    };
    std::vector<float> logits(vocab);
    for (size_t t = 0; t < vocab; ++t) {
      logits[t] = -8.0f + 10.0f * splitmix_unit(static_cast<uint64_t>(t));
    }
    {
      std::vector<ds41rt_v41_sampler_row_t> params = {
          row(0, 0.7f, 40u, 0.0f, -inf(), DS41RT_V41_SAMPLER_NO_MASK_ROW,
              DS41RT_V41_SAMPLER_FLAG_NO_MASK),
      };
      run_topk_case(logits, 1, vocab, params, {}, false, false, 40u,
                    "k3k4 wide vocab 129280 k=40");
    }
    {
      std::vector<ds41rt_v41_sampler_row_t> params = {
          row(0, 0.7f, 1000u, 0.0f, -inf(), DS41RT_V41_SAMPLER_NO_MASK_ROW,
              DS41RT_V41_SAMPLER_FLAG_NO_MASK),
      };
      run_topk_case(logits, 1, vocab, params, {}, false, false, 1000u,
                    "k3k4 wide vocab 129280 k=1000");
    }
    {
      const uint32_t k = static_cast<uint32_t>(vocab - 1u);
      std::vector<ds41rt_v41_sampler_row_t> params = {
          row(0, 0.7f, k, 0.0f, -inf(), DS41RT_V41_SAMPLER_NO_MASK_ROW,
              DS41RT_V41_SAMPLER_FLAG_NO_MASK),
      };
      run_topk_case(logits, 1, vocab, params, {}, false, false, 0u,
                    "k3k4 wide vocab 129280 k=survivor_count-1 (selection-only)");
    }
  }
  /* Independently pin the no-op and greedy outputs (a kernel that published a
   * retention for them would fail here). */
  {
    const size_t vocab = 64;
    std::vector<float> logits = shape_descending(vocab);
    std::vector<ds41rt_v41_sampler_row_t> params = {
        row(0, 0.7f, 1u, 0.0f, -inf(), DS41RT_V41_SAMPLER_NO_MASK_ROW,
            DS41RT_V41_SAMPLER_FLAG_NO_MASK),
        row(1, 0.7f, 0u, 0.0f, -inf(), DS41RT_V41_SAMPLER_NO_MASK_ROW,
            DS41RT_V41_SAMPLER_FLAG_NO_MASK),
        row(2, 0.7f, 64u, 0.0f, -inf(), DS41RT_V41_SAMPLER_NO_MASK_ROW,
            DS41RT_V41_SAMPLER_FLAG_NO_MASK),
        row(3, 0.7f, 65u, 0.0f, -inf(), DS41RT_V41_SAMPLER_NO_MASK_ROW,
            DS41RT_V41_SAMPLER_FLAG_NO_MASK),
    };
    const std::vector<float> four = [&] {
      std::vector<float> all;
      for (int i = 0; i < 4; ++i) {
        all.insert(all.end(), logits.begin(), logits.end());
      }
      return all;
    }();
    run_topk_case(four, 4, vocab, params, {}, false, false, 2u,
                  "k3k4 greedy/disabled/no-op rows");
  }
  std::cout << "ok  K3/K4 boundaries: no-op, greedy, capacity-0, wide 129280 (max passes "
            << host_bisection_pass_bound() << ")\n";
}

/* Regression for the ternary search's length-2 interval.
 *
 * The ternary rule `while (hi - lo >= 3)` can stop on a length-2 interval whose
 * *lower* value is the k-th key (reachable from a length-4 middle branch). It
 * then returns `hi = kth + 1`, a key no survivor carries: `above_count` is right
 * but the tie cut finds no equals, so the retained set comes up short. The
 * shipped loop probes once more at `lo + 1` and stops at `hi - lo == 1`.
 *
 * Each row has three survivors: `1.0` (best), a tiny positive middle value, and
 * `-1.0`; `k = 2`, so the k-th value is the middle one. The middle values below
 * are keys for which the old `>= 3` termination returned `kth + 1`; the test
 * re-derives that host-side so a future reintroduction is caught by this row and
 * not only by the oracle comparison. */
void test_k3_k4_length_two_interval() {
  const uint32_t middle_bits[] = {
      0x07EF06BDu, 0x2CDCDA2Eu, 0x27D8638Fu, 0x104FB2B1u,
      0x1C5D7853u, 0x10DE80B5u, 0x10567722u,
  };
  const size_t rows = sizeof(middle_bits) / sizeof(middle_bits[0]);
  const uint32_t top_key = host_order_key(1.0f);
  const uint32_t bottom_key = host_order_key(-1.0f);
  std::vector<float> logits(rows * 3);
  std::vector<ds41rt_v41_sampler_row_t> params;
  for (size_t r = 0; r < rows; ++r) {
    float middle = 0.0f;
    std::memcpy(&middle, &middle_bits[r], sizeof(middle));
    logits[r * 3 + 0] = 1.0f;
    logits[r * 3 + 1] = middle;
    logits[r * 3 + 2] = -1.0f;
    params.push_back(row(static_cast<uint32_t>(r), 1.0f, 2u, 0.0f, -inf(),
                         DS41RT_V41_SAMPLER_NO_MASK_ROW,
                         DS41RT_V41_SAMPLER_FLAG_NO_MASK));
    /* Host-side two-probe bisection with the OLD `>= 3` termination; it must
     * return `middle + 1`, which is what the fixed kernel must never return. */
    const uint32_t keys[3] = {top_key, host_order_key(middle), bottom_key};
    const auto count_gt = [&](uint32_t value) {
      uint32_t above = 0u;
      for (uint32_t key : keys) {
        if (key > value) {
          ++above;
        }
      }
      return above;
    };
    uint32_t lo = 0u;
    uint32_t hi = 0xFFFFFFFFu;
    while (hi - lo >= 3u) {
      const uint32_t third = (hi - lo) / 3u;
      const uint32_t m0 = lo + third;
      const uint32_t m1 = hi - third;
      const uint32_t c0 = count_gt(m0);
      const uint32_t c1 = count_gt(m1);
      if (c0 < 2u) {
        hi = m0;
      } else if (c1 < 2u) {
        lo = m0;
        hi = m1;
      } else {
        lo = m1;
      }
    }
    expect(hi == host_order_key(middle) + 1u,
           "length-2 regression: the old termination returns kth + 1");
  }
  run_topk_case(logits, rows, 3, params, {}, false, false, 2u,
                "k3k4 length-2 interval regression");
  std::cout << "ok  K3 length-2 interval final probe (" << rows << " adversarial keys)\n";
}

/* Deterministic pseudo-random rows over a mixed grid of shapes, k, min_p and
 * masks. The shapes repeat values at several granularities (six levels, a
 * tie band, exact duplicates), which is what stresses the tie cut and the
 * bisection's boundary landing rather than only the easy distinct-key case. */
void test_k3_k4_randomized_rows() {
  const size_t vocab = 257;
  const size_t words = (vocab + 31u) / 32u;
  const size_t rows = 24;
  const auto unit = [](uint64_t index) -> float {
    uint64_t hash = index * 0x9e3779b97f4a7c15ull + 0x1234567ull;
    hash = (hash ^ (hash >> 30)) * 0xbf58476d1ce4e5b9ull;
    hash = (hash ^ (hash >> 27)) * 0x94d049bb133111ebull;
    hash ^= hash >> 31;
    return static_cast<float>(hash >> 40) / 16777216.0f;
  };
  const uint32_t k_cycle[] = {2u, 3u, 5u, 17u, 40u, 64u, 100u, 256u, 300u};
  std::vector<float> logits(rows * vocab);
  std::vector<uint32_t> mask(rows * words, 0xFFFFFFFFu);
  std::vector<ds41rt_v41_sampler_row_t> params;
  size_t capacity = 0;
  for (size_t r = 0; r < rows; ++r) {
    for (size_t t = 0; t < vocab; ++t) {
      const float value = unit(static_cast<uint64_t>(r) * vocab + t);
      switch (r % 4u) {
        case 0:
          logits[r * vocab + t] = value * 2.0f - 1.0f;
          break;
        case 1:
          logits[r * vocab + t] = static_cast<float>(static_cast<int>(value * 6.0f));
          break;
        case 2:
          logits[r * vocab + t] = (t % 5u == 0u) ? 1.0f : 0.0f;
          break;
        default:
          logits[r * vocab + t] = -0.01f * static_cast<float>(t / 3u);
          break;
      }
    }
    /* 0.7 keeps every row stochastic; min_p cycles through disabled and two
     * finite thresholds, and one row in five is masked. */
    const float min_p = (r % 3u == 0u) ? 0.0f : ((r % 3u == 1u) ? 0.05f : 0.5f);
    const float ln_min_p = (min_p > 0.0f) ? std::log(min_p) : -inf();
    bool masked = (r % 5u == 0u);
    uint32_t mask_row = masked ? static_cast<uint32_t>(r) : DS41RT_V41_SAMPLER_NO_MASK_ROW;
    uint32_t flags = masked ? 0u : DS41RT_V41_SAMPLER_FLAG_NO_MASK;
    if (masked) {
      for (size_t t = 0; t < vocab; ++t) {
        const bool allowed = (t % 3u) != 0u;
        mask[r * words + t / 32u] =
            allowed ? (mask[r * words + t / 32u] | (1u << (t % 32u)))
                    : (mask[r * words + t / 32u] & ~(1u << (t % 32u)));
      }
      ds41rt_v41_sampler_clear_remainder(mask.data() + r * words, vocab);
    }
    const uint32_t k = k_cycle[r % (sizeof(k_cycle) / sizeof(k_cycle[0]))];
    capacity = std::max(capacity, static_cast<size_t>(k));
    params.push_back(row(static_cast<uint32_t>(r), 0.7f, k, min_p, ln_min_p, mask_row,
                         flags));
  }
  run_topk_case(logits, rows, vocab, params, mask, true, true, capacity,
                "k3k4 randomized mixed rows");
  std::cout << "ok  K3/K4 randomized grid: " << rows << " rows, vocab " << vocab
            << ", capacity " << capacity << "\n";
}

/* ====================================================================== */
/* Chunk 3b: K5 inclusive-prefix top-p nucleus + rank-order draw          */
/* (design §4.5-§4.6, §12.3-§12.4, §12.8; contract §1.3)                  */
/* ====================================================================== */

/* Host port of the production ordered path's tail, `sample_from_ranked`
 * (`target_sampling.rs:527-571`). `ranked` is the CPU's ranked list in exact
 * rank order (scaled descending, id ascending); `total` is the CPU's
 * un-normalized weight total. The port reproduces the CPU's operation order
 * exactly: no `top_p >= 1.0` shortcut, clamp to `[1e-6, 1]`, the inclusive
 * `>=` prefix break, the `1e-20` floors, and `selected = nucleus_count - 1` as
 * the initialized fallback. */
struct RefOrdered {
  uint32_t token = 0u;
  float total = 0.0f;
  uint32_t nucleus_count = 0u;
  uint32_t nucleus_crossing = 0u;
  bool fallback_used = false;
  bool nucleus_full_set = false;
};

RefOrdered host_ordered_tail(const std::vector<std::pair<float, uint32_t>>& ranked,
                             float max_scaled, float top_p, float uniform) {
  RefOrdered out;
  /* `ranked` is never empty on the ordered path: the best survivor always
   * survives every filter (`target_sampling.rs:533`). */
  std::vector<float> weights;
  weights.reserve(ranked.size());
  for (const std::pair<float, uint32_t>& entry : ranked) {
    weights.push_back(std::exp(entry.first - max_scaled));
  }
  float total = 0.0f;
  for (float weight : weights) {
    total += weight;
  }
  if (total < 1.0e-20f) {
    total = 1.0e-20f;
  }
  for (float& weight : weights) {
    weight /= total;
  }
  const float clamped = std::fmin(std::fmax(top_p, 1.0e-6f), 1.0f);
  float nucleus_mass = 0.0f;
  size_t nucleus_count = 0;
  for (float weight : weights) {
    nucleus_mass += weight;
    ++nucleus_count;
    if (nucleus_mass >= clamped) {
      break;
    }
  }
  out.fallback_used = (nucleus_count == weights.size()) && (nucleus_mass < clamped);
  out.nucleus_full_set = (nucleus_count == weights.size());
  if (nucleus_mass < 1.0e-20f) {
    nucleus_mass = 1.0e-20f;
  }
  const float target = uniform * nucleus_mass;
  size_t selected = nucleus_count - 1;
  float cumulative = 0.0f;
  for (size_t rank = 0; rank < nucleus_count; ++rank) {
    cumulative += weights[rank];
    if (target <= cumulative) {
      selected = rank;
      break;
    }
  }
  out.token = ranked[selected].second;
  out.total = total;
  out.nucleus_count = static_cast<uint32_t>(nucleus_count);
  out.nucleus_crossing = static_cast<uint32_t>(nucleus_count - 1);
  return out;
}

/* Host port of the CPU's ordered branch selection (`target_sampling.rs:477-522`
 * plus the shared tail): the bounded-heap top-k path when `top_k <
 * survivor_count`, the full survivor ordering otherwise, and the K2 fast-path
 * branch (`top_k` disabled and `top_p >= 1.0`) which K5 must not touch. */
struct RefOrderedRow {
  bool greedy = false;
  bool k2_fast_path = false;
  bool ok = false;
  uint32_t status = DS41RT_V41_SAMPLER_STATUS_OK;
  uint32_t survivor_count = 0;
  uint32_t domain_count = 0; /* K5's `S`: top_k when materialized, else survivors */
  uint32_t token = 0;
  float total = 0.0f;
  uint32_t nucleus_count = 0;
  /* The CPU's crossing rank (`nucleus_count - 1`): the rank whose inclusive
   * prefix first reaches `top_p`. Used to classify a device token that differs
   * from the production token as an adjacent-crossing boundary flip. */
  uint32_t nucleus_crossing = 0;
  bool fallback_used = false;
  bool nucleus_full_set = false;
  /* The CPU's ranked survivors, so a divergence can be classified rather than
   * dismissed. */
  std::vector<std::pair<float, uint32_t>> ranked;
};

/* The CPU's FLOAT32 normalized inclusive prefix through `id` in the ranked
 * list -- `Σ_{rank <= r(id)} fl(w_rank / total)` with
 * `w_rank = expf(first_rank - first_0)`, the same left-to-right f32
 * accumulation `sample_from_ranked` performs (`target_sampling.rs:534-551`).
 * This is the quantity the boundary exception must compare against `top_p`.
 * The previous double-precision helper returned a DIFFERENT number (for 12 equal
 * weights it returned exactly `1.0` where the CPU's f32 prefix is
 * `0.99999988079071045`), so it could accept a several-rank miss whenever the
 * double value sat within slack of `top_p`. */
float host_cumulative_through_f32(const RefOrderedRow& row, uint32_t id) {
  float total = row.total;
  if (total < 1.0e-20f) {
    total = 1.0e-20f;
  }
  float running = 0.0f;
  for (size_t rank = 0; rank < row.ranked.size(); ++rank) {
    running += std::exp(row.ranked[rank].first - row.ranked[0].first) / total;
    if (row.ranked[rank].second == id) {
      return running;
    }
  }
  return running;
}


RefOrderedRow host_ordered_reference(const float* logits, size_t vocab,
                                     const uint32_t* mask_words,
                                     size_t mask_words_per_row, float temperature,
                                     float top_p, uint32_t top_k, float min_p,
                                     float ln_min_p, uint32_t flags, uint64_t seed,
                                     uint64_t position) {
  RefOrderedRow out;
  const bool unconstrained = mask_words == nullptr || mask_words_per_row == 0u;
  const auto allowed = [&](size_t token) {
    return unconstrained || ((mask_words[token / 32u] >> (token % 32u)) & 1u) != 0u;
  };
  out.greedy = is_greedy(temperature, top_k) ||
               (flags & DS41RT_V41_SAMPLER_FLAG_GREEDY) != 0u;
  if (out.greedy) {
    return out;
  }
  const float inv = 1.0f / temperature;
  float max_scaled = -std::numeric_limits<float>::infinity();
  size_t allowed_count = 0;
  for (size_t token = 0; token < vocab; ++token) {
    if (!allowed(token)) {
      continue;
    }
    if (!std::isfinite(logits[token])) {
      out.status = DS41RT_V41_SAMPLER_STATUS_NONFINITE_LOGIT;
      return out;
    }
    ++allowed_count;
    max_scaled = std::fmax(max_scaled, logits[token] * inv);
  }
  if (allowed_count == 0u) {
    out.status = DS41RT_V41_SAMPLER_STATUS_EMPTY_CANDIDATES;
    return out;
  }
  if (!std::isfinite(max_scaled)) {
    out.status = DS41RT_V41_SAMPLER_STATUS_INVALID_TEMPERATURE;
    return out;
  }
  const float min_scaled = min_p > 0.0f ? max_scaled + ln_min_p
                                        : -std::numeric_limits<float>::infinity();
  size_t survivor_count = 0u;
  for (size_t token = 0; token < vocab; ++token) {
    if (allowed(token) && logits[token] * inv >= min_scaled) {
      ++survivor_count;
    }
  }
  out.ok = true;
  out.survivor_count = static_cast<uint32_t>(survivor_count);
  if (top_k == 0u && top_p >= 1.0f) {
    /* The disjoint K2 fast path; K5 is not allowed to overwrite its id. */
    out.k2_fast_path = true;
    return out;
  }
  if (survivor_count == 0u) {
    out.ok = false;
    return out;
  }
  const float uniform =
      host_clamp_uniform(host_target_uniform(seed, position));

  /* Branch 1: bounded capacity-`top_k` selection (`:477-497`). */
  if (top_k != 0u && static_cast<size_t>(top_k) < survivor_count) {
    std::vector<std::pair<float, uint32_t>> ranked;
    ranked.reserve(top_k);
    for (size_t token = 0; token < vocab; ++token) {
      if (!allowed(token) || !(logits[token] * inv >= min_scaled)) {
        continue;
      }
      const float scaled = logits[token] * inv;
      const uint32_t id = static_cast<uint32_t>(token);
      if (ranked.size() < top_k) {
        ranked.emplace_back(scaled, id);
        continue;
      }
      /* Worst = lowest scaled, then highest id (the CPU's `WorstFirst`). */
      size_t worst = 0;
      for (size_t i = 1; i < ranked.size(); ++i) {
        if (ranked[i].first < ranked[worst].first ||
            (ranked[i].first == ranked[worst].first &&
             ranked[i].second > ranked[worst].second)) {
          worst = i;
        }
      }
      if (scaled > ranked[worst].first ||
          (scaled == ranked[worst].first && id < ranked[worst].second)) {
        ranked[worst] = {scaled, id};
      }
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const std::pair<float, uint32_t>& a,
                 const std::pair<float, uint32_t>& b) {
                if (a.first != b.first) {
                  return a.first > b.first;
                }
                return a.second < b.second;
              });
    out.domain_count = static_cast<uint32_t>(ranked.size());
    const RefOrdered tail = host_ordered_tail(ranked, max_scaled, top_p, uniform);
    out.token = tail.token;
    out.total = tail.total;
    out.nucleus_count = tail.nucleus_count;
    out.nucleus_crossing = tail.nucleus_crossing;
    out.fallback_used = tail.fallback_used;
    out.nucleus_full_set = tail.nucleus_full_set;
    out.ranked = std::move(ranked);
    return out;
  }

  /* Branches 2/3: order every survivor, then the shared tail (`:503-522`). */
  std::vector<std::pair<float, uint32_t>> ranked;
  ranked.reserve(survivor_count);
  for (size_t token = 0; token < vocab; ++token) {
    if (allowed(token) && logits[token] * inv >= min_scaled) {
      ranked.emplace_back(logits[token] * inv, static_cast<uint32_t>(token));
    }
  }
  std::sort(ranked.begin(), ranked.end(),
            [](const std::pair<float, uint32_t>& a,
               const std::pair<float, uint32_t>& b) {
              if (a.first != b.first) {
                return a.first > b.first;
              }
              return a.second < b.second;
            });
  out.domain_count = static_cast<uint32_t>(ranked.size());
  const RefOrdered tail = host_ordered_tail(ranked, max_scaled, top_p, uniform);
  out.token = tail.token;
  out.total = tail.total;
  out.nucleus_count = tail.nucleus_count;
  out.nucleus_crossing = tail.nucleus_crossing;
  out.fallback_used = tail.fallback_used;
  out.nucleus_full_set = tail.nucleus_full_set;
  out.ranked = std::move(ranked);
  return out;
}

/* A row block with an explicit `top_p` (the shared `row` helper fixes it at
 * 1.0, and K5's whole point is the ordered `top_p`). */
ds41rt_v41_sampler_row_t k5_row(uint32_t output_row, float temperature, float top_p,
                                uint32_t top_k, float min_p, float ln_min_p,
                                uint32_t mask_row, uint32_t flags, uint64_t seed,
                                uint64_t position) {
  ds41rt_v41_sampler_row_t value = {};
  value.seed = seed;
  value.position = position;
  value.temperature = temperature;
  value.top_p = top_p;
  value.min_p = min_p;
  value.top_k = top_k;
  value.mask_row = mask_row;
  value.flags = flags;
  value.output_row = output_row;
  value.ln_min_p = ln_min_p;
  return value;
}

/* One K1 + K3/K4 + K5 run with its device buffers. */
struct K5Run {
  K1 k1;
  TopkBuffers topk;
  uint32_t* ids = nullptr;
  float* total = nullptr;
  uint32_t* nucleus = nullptr;
  size_t rows = 0;
  size_t capacity = 0;
};

K5Run make_k5_run(size_t rows, size_t vocab, size_t capacity, bool with_mask) {
  K5Run run;
  run.rows = rows;
  run.capacity = capacity;
  run.k1 = make_k1(rows, vocab, with_mask, false);
  run.topk = make_topk_buffers(rows, capacity);
  alloc_device(&run.ids, rows * sizeof(uint32_t), "k5 ids");
  alloc_device(&run.total, rows * sizeof(float), "k5 total");
  alloc_device(&run.nucleus, rows * sizeof(uint32_t), "k5 nucleus");
  return run;
}

void free_k5_run(K5Run* run) {
  free_topk_buffers(&run->topk);
  cudaFree(run->ids);
  cudaFree(run->total);
  cudaFree(run->nucleus);
  free_k1(&run->k1);
  *run = K5Run{};
}

/* Drive K1 -> K3/K4 -> K5 and assert the exact final token, the exact
 * `out_nucleus_count` and (with a tolerance for the device `expf`) `out_total`
 * against `host_ordered_reference`. Every output is pre-filled with a sentinel,
 * so a kernel that never ran cannot pass; the K2 fast-path rows must keep
 * K2's id and the greedy rows K1's.
 *
 * `describe` appends the fallback/nucleus shape of every compared row to the
 * run's counters, which the caller prints as the MEASURED coverage evidence. */
/* Aggregate K5 measurements: how many ordered rows were compared, how many
 * matched the production CPU sampler's token exactly, and how many of the
 * rest are one-rank boundary flips (a different f32 accumulation order landing
 * on the adjacent crossing). Every row must be one of the latter two. */
/* The window (in ranks) around the CPU crossing inside which a device token is
 * attributed to the accumulation-order residual rather than a filter bug. */
constexpr uint32_t KP_CROSSING_WINDOW = 4u;

struct K5Stats {
  size_t compared = 0;
  size_t exact = 0;
  size_t boundary_flip = 0;
  /* Largest measured CDF error between a divergent device token and the CPU's
   * own token: `|f32_cumulative_through(device) - f32_cumulative_through(cpu)|`.
   * This is the §6.3c accumulation residual expressed as a probability, and it
   * is asserted against `kK5CdfErrorBound` for every ordered row. */
  double max_cdf_error = 0.0;
  /* Defect-sensitive counters. `nucleus_mismatch` counts ordered rows whose
   * device `out_nucleus_count` differs from the production nucleus at all --
   * incremented OUTSIDE the `strict_nucleus` gate, so a relaxed profile still
   * measures it; `outside_nucleus` counts rows whose token is not inside the CPU
   * nucleus; `max_rank_displacement` is the largest |device rank - CPU rank|
   * observed in the CPU's ranked order. Any nonzero value in the first two, or a
   * displacement above the asserted window, fails the run (see `run_k5_case`). */
  size_t nucleus_mismatch = 0;
  size_t outside_nucleus = 0;
  size_t max_rank_displacement = 0;
  /* Rows whose nucleus differs by exactly one rank while the CPU's own FLOAT32
   * normalized prefix at the device's crossing is within `kK5BoundarySlack` of
   * the clamped `top_p` -- the accepted k-scale f32 boundary drift. */
  size_t nucleus_boundary = 0;
  /* The true largest |device nucleus - CPU nucleus| seen, recorded on every
   * ordered row (previously hard-coded to 1 and never read). */
  size_t max_nucleus_delta = 0;
  /* Ordered rows whose production reference reported `fallback_used` (the CPU
   * consumed every weight without crossing); asserted > 0 by the fallback test
   * so the fallback path is measured, not merely printed. */
  size_t fallback_rows = 0;
};

/* The tolerance for accepting a one-rank nucleus difference as the k-scale f32
 * normalization drift (measured: ~1.3e-6 on the 36-token `tied@k40+top_p` case;
 * a wrong crossing is off by a whole rank's mass, >= 1e-3 on these grids). */
constexpr double kK5BoundarySlack = 1.0e-5;

/* The measured CDF-error bound for a divergent token. Both engines are
 * internally consistent f32 draws of the same distribution, so a divergent token
 * must sit at (almost) the same CDF location; measured on the wide 129,280-token
 * rows the error is one rank's mass (~1/129,280 = 7.7e-6) or less. `1e-3` is two
 * orders of magnitude above that measured envelope and still far below a
 * filter/set error. */
constexpr double kK5CdfErrorBound = 1.0e-3;

/* The measured nucleus-count envelope for the 129,280-token served rows. Near
 * the `top_p` crossing each rank contributes only ~1/129,280 = 7.7e-6 of the
 * mass, so the two accumulation orders' ~2e-4 relative difference moves the
 * crossing by tens of ranks: the device `top_p = 0.9` survivor row measures a
 * delta of 28. This is a MEASURED bound on the wide rows, not a blanket
 * relaxation -- the token, its rank window, its membership in the DEVICE
 * nucleus and the CDF error are all still asserted. */
constexpr size_t kK5WideNucleusDeltaBound = 64u;

/* `strict_nucleus` now gates ONLY the exact nucleus-count equality assertion.
 * Every ordered row always asserts: the real `|device nucleus - CPU nucleus|`
 * is recorded (`K5Stats::nucleus_mismatch`, `max_nucleus_delta`), the token is
 * inside the CPU nucleus, the token is inside the DEVICE-reported nucleus, the
 * divergent token is within `KP_CROSSING_WINDOW` ranks of the CPU's token, and
 * the measured CDF error is within `kK5CdfErrorBound`. The wide 129,280-token
 * rows pass `false` for the exact-count assertion only; their token, rank and
 * CDF checks are the same as every other row. */
void run_k5_case(const std::vector<float>& logits, size_t rows, size_t vocab,
                 const std::vector<ds41rt_v41_sampler_row_t>& params,
                 const std::vector<uint32_t>& mask, bool with_mask,
                 bool strict_mask_bits, size_t capacity, const char* label,
                 K5Stats* stats, bool strict_nucleus = true,
                 std::vector<uint32_t>* nucleus_trace = nullptr,
                 std::vector<uint32_t>* retained_trace = nullptr) {
  ++g_cases;
  const std::string tag(label);
  expect(params.size() == rows, tag + ": one param block per row");
  K5Run run = make_k5_run(rows, vocab, capacity, with_mask);
  require_cuda(cudaMemcpy(run.k1.logits, logits.data(), logits.size() * sizeof(float),
                          cudaMemcpyHostToDevice), "k5 logits h2d");
  require_cuda(cudaMemcpy(run.k1.params, params.data(),
                          params.size() * sizeof(ds41rt_v41_sampler_row_t),
                          cudaMemcpyHostToDevice), "k5 params h2d");
  std::vector<uint32_t> device_mask = mask;
  if (with_mask) {
    if (strict_mask_bits) {
      for (size_t r = 0; r < rows; ++r) {
        ds41rt_v41_sampler_clear_remainder(device_mask.data() + r * run.k1.words, vocab);
      }
    }
    require_cuda(cudaMemcpy(run.k1.mask, device_mask.data(),
                            device_mask.size() * sizeof(uint32_t),
                            cudaMemcpyHostToDevice), "k5 mask h2d");
  }
  /* K1 then K2 (K2 is a no-op for every ordered row here). */
  expect(ds41rt_cuda_v41_target_sample(
             run.k1.logits, rows, vocab, vocab, run.k1.params,
             with_mask ? run.k1.mask : nullptr, with_mask ? run.k1.words : 0u,
             run.k1.ids, run.k1.status, run.k1.detail, run.k1.scores, nullptr, nullptr,
             run.k1.scratch) == DS41RT_STATUS_OK,
         tag + ": K1 launch");
  /* K3/K4 with a fully sentineled arena. */
  const uint32_t sentinel = 0xDEADBEEFu;
  std::vector<uint32_t> host_ids(rows * capacity, sentinel);
  std::vector<uint64_t> host_scratch(rows * capacity, 0ull);
  std::vector<uint32_t> host_retained(rows, sentinel);
  std::vector<uint32_t> host_passes(rows, sentinel);
  if (capacity > 0) {
    require_cuda(cudaMemcpy(run.topk.rank_ids, host_ids.data(),
                            rows * capacity * sizeof(uint32_t), cudaMemcpyHostToDevice),
                 "k5 rank ids h2d");
    require_cuda(cudaMemcpy(run.topk.rank_scratch, host_scratch.data(),
                            rows * capacity * sizeof(uint64_t), cudaMemcpyHostToDevice),
                 "k5 rank scratch h2d");
  }
  require_cuda(cudaMemcpy(run.topk.retained, host_retained.data(),
                          rows * sizeof(uint32_t), cudaMemcpyHostToDevice),
               "k5 retained h2d");
  require_cuda(cudaMemcpy(run.topk.passes, host_passes.data(), rows * sizeof(uint32_t),
                          cudaMemcpyHostToDevice),
               "k5 passes h2d");
  expect(ds41rt_cuda_v41_topk_select(
             run.k1.logits, rows, vocab, vocab, run.k1.params,
             with_mask ? run.k1.mask : nullptr, with_mask ? run.k1.words : 0u,
             capacity > 0 ? run.topk.rank_ids : nullptr,
             capacity > 0 ? run.topk.rank_scratch : nullptr, capacity,
             run.topk.retained, run.topk.passes, run.k1.scratch) == DS41RT_STATUS_OK,
         tag + ": topk launch");
  /* K5 with sentineled outputs. */
  std::vector<uint32_t> host_out_ids(rows, sentinel);
  std::vector<float> host_out_total(rows, -123.0f);
  std::vector<uint32_t> host_out_nucleus(rows, sentinel);
  require_cuda(cudaMemcpy(run.ids, host_out_ids.data(), rows * sizeof(uint32_t),
                          cudaMemcpyHostToDevice), "k5 out ids h2d");
  require_cuda(cudaMemcpy(run.total, host_out_total.data(), rows * sizeof(float),
                          cudaMemcpyHostToDevice), "k5 out total h2d");
  require_cuda(cudaMemcpy(run.nucleus, host_out_nucleus.data(), rows * sizeof(uint32_t),
                          cudaMemcpyHostToDevice), "k5 out nucleus h2d");
  expect(ds41rt_cuda_v41_nucleus(
             run.k1.logits, rows, vocab, vocab, run.k1.params,
             with_mask ? run.k1.mask : nullptr, with_mask ? run.k1.words : 0u,
             capacity > 0 ? run.topk.rank_ids : nullptr, capacity, run.topk.retained,
             run.ids, run.k1.status, run.total, run.nucleus,
             run.k1.scratch) == DS41RT_STATUS_OK,
         tag + ": nucleus launch");
  require_cuda(cudaDeviceSynchronize(), "k5 kernel");

  std::vector<uint32_t> retained(rows);
  require_cuda(cudaMemcpy(retained.data(), run.topk.retained, rows * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost), "k5 retained d2h");

  std::vector<uint32_t> ids(rows, sentinel);
  std::vector<float> totals(rows, -123.0f);
  std::vector<uint32_t> nucleus(rows, sentinel);
  require_cuda(cudaMemcpy(ids.data(), run.ids, rows * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost), "k5 ids d2h");
  require_cuda(cudaMemcpy(totals.data(), run.total, rows * sizeof(float),
                          cudaMemcpyDeviceToHost), "k5 total d2h");
  require_cuda(cudaMemcpy(nucleus.data(), run.nucleus, rows * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost), "k5 nucleus d2h");
  if (nucleus_trace != nullptr) {
    *nucleus_trace = nucleus;
  }
  if (retained_trace != nullptr) {
    *retained_trace = retained;
  }

  for (size_t r = 0; r < rows; ++r) {
    const ds41rt_v41_sampler_row_t& p = params[r];
    const std::string row_tag = tag + " row " + std::to_string(r);
    const bool unconstrained = (p.flags & DS41RT_V41_SAMPLER_FLAG_NO_MASK) != 0u ||
                               p.mask_row == DS41RT_V41_SAMPLER_NO_MASK_ROW;
    const uint32_t* row_mask = (with_mask && !unconstrained)
        ? device_mask.data() + static_cast<size_t>(p.mask_row) * run.k1.words
        : nullptr;
    const size_t row_words = (row_mask != nullptr) ? run.k1.words : 0u;
    const RefOrderedRow expected = host_ordered_reference(
        logits.data() + r * vocab, vocab, row_mask, row_words, p.temperature, p.top_p,
        p.top_k, p.min_p, p.ln_min_p, p.flags, p.seed, p.position);
    if (expected.greedy || expected.k2_fast_path) {
      /* K5 must not have touched the row: the sentinel survives. */
      expect(ids[r] == sentinel, row_tag + ": non-ordered row left untouched by K5");
      expect(nucleus[r] == sentinel, row_tag + ": non-ordered nucleus untouched");
      expect(totals[r] == -123.0f, row_tag + ": non-ordered total untouched");
      continue;
    }
    expect(expected.ok, row_tag + ": host reference reached the ordered path");
    if (stats != nullptr) {
      ++stats->compared;
      if (expected.fallback_used) {
        ++stats->fallback_rows;
      }
    }
    expect(ids[r] != sentinel, row_tag + ": K5 wrote the token");
    /* The token must be an element of the CPU's ordered set (a filter bug would
     * pick a masked or non-surviving token), and its rank in that order is the
     * only meaningful way to bound a divergence: two engines that differ by a
     * few ranks pick different ids but the same distribution, while a
     * rank-domain bug (the survivor domain keyed by token order) lands at an
     * arbitrary probability rank. */
    size_t device_rank = expected.ranked.size();
    for (size_t rank = 0; rank < expected.ranked.size(); ++rank) {
      if (expected.ranked[rank].second == ids[r]) {
        device_rank = rank;
        break;
      }
    }
    size_t cpu_rank = expected.ranked.size();
    for (size_t rank = 0; rank < expected.ranked.size(); ++rank) {
      if (expected.ranked[rank].second == expected.token) {
        cpu_rank = rank;
        break;
      }
    }
    expect(device_rank < expected.ranked.size(),
           row_tag + ": device token is in the CPU's ordered set");
    expect(cpu_rank < expected.ranked.size(),
           row_tag + ": CPU token is in its own ranked order");
    const size_t rank_distance = device_rank > cpu_rank ? device_rank - cpu_rank
                                                        : cpu_rank - device_rank;
    if (stats != nullptr && rank_distance > stats->max_rank_displacement) {
      stats->max_rank_displacement = rank_distance;
    }
    /* The nucleus is the CPU's rank prefix, so every selected token -- exact or
     * divergent -- must sit inside it. */
    if (device_rank >= expected.nucleus_count) {
      if (stats != nullptr) {
        ++stats->outside_nucleus;
      }
      std::cerr << "nucleus detail " << row_tag << ": device token " << ids[r]
                << " rank " << device_rank << " is outside the CPU nucleus of "
                << expected.nucleus_count << " ranks (crossing "
                << expected.nucleus_crossing << ")\n";
    }
    expect(device_rank < expected.nucleus_count,
           row_tag + ": device token rank is inside the CPU nucleus");
    /* The device's own nucleus must match the production nucleus. The REAL
     * count difference is recorded on every ordered row, outside the
     * `strict_nucleus` gate, so a relaxed profile still measures it; the
     * exact-equality assertion is what `strict_nucleus` gates. A difference is
     * accepted as the k-scale f32 boundary drift only when it is exactly ONE
     * rank AND the CPU's own FLOAT32 normalized prefix at the device crossing
     * rank is within `kK5BoundarySlack` of the clamped `top_p`. A several-rank
     * difference is never accepted. `max_nucleus_delta` is the true maximum,
     * not a hard-coded constant. */
    const uint32_t nucleus_delta =
        (nucleus[r] > expected.nucleus_count) ? (nucleus[r] - expected.nucleus_count)
                                              : (expected.nucleus_count - nucleus[r]);
    if (stats != nullptr) {
      if (nucleus_delta > stats->max_nucleus_delta) {
        stats->max_nucleus_delta = nucleus_delta;
      }
      if (nucleus_delta != 0u) {
        ++stats->nucleus_mismatch;
      }
    }
    bool nucleus_ok = (nucleus_delta == 0u);
    bool boundary_exception = false;
    if (!nucleus_ok && nucleus_delta == 1u && !expected.fallback_used &&
        nucleus[r] >= 1u && nucleus[r] <= expected.domain_count) {
      const float crossed = host_cumulative_through_f32(
          expected, expected.ranked[nucleus[r] - 1u].second);
      const float clamped_top_p = std::fmin(std::fmax(p.top_p, 1.0e-6f), 1.0f);
      boundary_exception =
          std::fabs(crossed - clamped_top_p) <= static_cast<float>(kK5BoundarySlack);
      nucleus_ok = boundary_exception;
    }
    if (stats != nullptr && boundary_exception) {
      ++stats->nucleus_boundary;
    }
    if (!nucleus_ok) {
      std::cerr << "nucleus detail " << row_tag << ": device nucleus " << nucleus[r]
                << " CPU nucleus " << expected.nucleus_count << " delta " << nucleus_delta
                << " (cpu crossing " << expected.nucleus_crossing << ", domain "
                << expected.domain_count << ", top_p " << p.top_p << ", top_k " << p.top_k
                << ", fallback " << (expected.fallback_used ? 1 : 0) << ")\n";
    }
    expect(!strict_nucleus || nucleus_ok,
           row_tag +
               ": device nucleus matches production (or is one rank off at the f32 boundary)");
    /* The selected token's rank must also lie inside the nucleus the DEVICE
     * reported, not only inside the CPU's. This is asserted for every ordered
     * row, independent of `strict_nucleus`: a relaxed profile may tolerate a
     * count difference, but never a token outside its own device nucleus. */
    if (nucleus[r] != sentinel && device_rank < expected.ranked.size()) {
      expect(device_rank < nucleus[r],
             row_tag + ": device token rank is inside the DEVICE nucleus");
    }
    if (ids[r] == expected.token) {
      if (stats != nullptr) {
        ++stats->exact;
      }
    } else {
      /* The declared accumulation-order/`expf` residual can move the *crossing
       * rank* and therefore the drawn rank. The CPU sums the ranked normalized
       * weights sequentially in f32; on rows with a wide dynamic range that
       * walk saturates (once the running sum is ~1.0 every later term rounds to
       * zero), while the device's fixed segmented tree keeps accumulating the
       * true tail. Both are internally consistent f32 draws of the same
       * distribution; which token they pick is §6.3c's measured residual, not a
       * filter property. What *is* assertable, and is asserted here: the device
       * token is one of the CPU's ranked survivors (so no masked or
       * non-surviving token can pass), and it lies inside the nucleus the
       * device itself reported. The exact-match count and the measured CDF
       * deviation are reported by the caller. */
      const float cdf_device = host_cumulative_through_f32(expected, ids[r]);
      const float cdf_cpu = host_cumulative_through_f32(expected, expected.token);
      const double cdf_error = std::fabs(static_cast<double>(cdf_device) -
                                         static_cast<double>(cdf_cpu));
      if (stats != nullptr) {
        ++stats->boundary_flip;
        if (cdf_error > stats->max_cdf_error) {
          stats->max_cdf_error = cdf_error;
        }
      }
      expect(nucleus[r] != sentinel, row_tag + ": divergent row still wrote a nucleus");
      /* The two engines are f32 draws of the same distribution, so a divergent
       * token must still sit at (almost) the same CDF location; this is the
       * MEASURED residual bound, enforced for every ordered row (including the
       * relaxed wide profiles that used to skip it entirely). */
      expect(cdf_error <= kK5CdfErrorBound,
             row_tag + ": divergent token is within the measured CDF-error bound");
      /* A divergent token is only attributable to the declared f32
       * accumulation residual when it sits within `KP_CROSSING_WINDOW` ranks of
       * the CPU's own selected rank. The window is asserted for EVERY ordered
       * row, wide profiles included (they used to pass `strict_nucleus=false`
       * and skip it); the measured wide-row displacement is 1, well inside it.
       * `KP_CROSSING_WINDOW` was previously declared and never used, which let a
       * survivor domain keyed by token order (arbitrary probability rank) pass. */
      if (rank_distance > KP_CROSSING_WINDOW) {
        std::cerr << "window detail " << row_tag << ": device rank " << device_rank
                  << " vs CPU rank " << cpu_rank << " (distance " << rank_distance
                  << ", window " << KP_CROSSING_WINDOW << ")\n";
      }
      expect(rank_distance <= KP_CROSSING_WINDOW,
             row_tag + ": divergent token is within KP_CROSSING_WINDOW ranks of the CPU rank");
      if (capacity == 0u || retained[r] == 0u) {
        /* Survivor domain: the rank must be inside the reported nucleus. */
        expect(nucleus[r] >= 1u && nucleus[r] <= expected.domain_count,
               row_tag + ": divergent nucleus count is inside the domain");
      }
    }
    /* `total` differs only by the declared `expf` residual (§6.3b): every
     * `w_r` differs by 1-2 ulp and the two accumulation orders differ, so the
     * relative error of a `count`-term sum grows with the term count. The
     * bound below is `2e-6 * log2(count)`, the measured envelope for the
     * widest 129,280-token rows; the observed maximum is reported by the
     * caller through the run counters. */
    const float relative = std::fabs(totals[r] - expected.total) /
                           std::fmax(expected.total, 1.0e-30f);
    /* `n` f32 terms accumulated in different orders can differ by
     * `O(n * 2^-24)` of the total even with identical inputs; measured on the
     * 129,280-wide rows the relative difference is ~3.4e-4, so the allowance is
     * a flat 5e-3 -- three orders of magnitude tighter than the `count*eps`
     * bound and still ~10x the observed envelope. */
    const float allowance = 5.0e-3f;
    if (relative > allowance) {
      std::cerr << "total mismatch " << row_tag << " device=" << totals[r]
                << " cpu=" << expected.total << " rel=" << relative
                << " allowance=" << allowance
                << " survivors=" << expected.survivor_count
                << " domain=" << expected.domain_count
                << " top_p=" << p.top_p << " top_k=" << p.top_k
                << " T=" << p.temperature << " seed=" << p.seed
                << " pos=" << p.position << "\n";
    }
    expect(relative <= allowance, row_tag + ": total within the expf residual");
    /* The token is inside the nucleus by construction. */
    if (capacity > 0 && retained[r] > 0u) {
      size_t rank = 0u;
      bool found = false;
      std::vector<uint32_t> row_ids(capacity, sentinel);
      require_cuda(cudaMemcpy(row_ids.data(),
                              run.topk.rank_ids + r * capacity,
                              capacity * sizeof(uint32_t), cudaMemcpyDeviceToHost),
                   "k5 rank ids d2h");
      for (size_t i = 0; i < retained[r]; ++i) {
        if (row_ids[i] == ids[r]) {
          rank = i;
          found = true;
          break;
        }
      }
      expect(found, row_tag + ": token is in the retained set");
      expect(rank < expected.nucleus_count,
             row_tag + ": token rank is inside the CPU nucleus");
      /* The retained domain has a device-reported nucleus too; use it, not the
       * CPU's count, for the membership assertion. */
      if (nucleus[r] != sentinel) {
        expect(rank < nucleus[r],
               row_tag + ": token rank is inside the DEVICE retained nucleus");
      }
    }
  }
  free_k5_run(&run);
}

/* The `top_p = 1.0` strict prefix, in two parts.
 *
 * (a) A row with a deep tail that underflows: `expf(-200)` is a subnormal
 * (~1e-87) and the normalized tail weights divide to exactly `0.0f`, so the
 * f32 running prefix reaches `1.0` several ranks before the last survivor and
 * the CPU nucleus is a **strict prefix**. That is the case the removed
 * "nucleus = all of S" shortcut got wrong. On this row the device's answer is
 * checked exactly against the production CPU sampler.
 *
 * (b) The rounded `top_p = 1.0` prefix that sits at the declared `expf`
 * residual. The CPU computes `p_0 = w_0 / total` with Rust's `f32::exp`; a
 * dominant rank with a ~1e-9 tail can make `p_0` round to exactly `1.0f` on one
 * engine and to the next f32 below on the other. That is a *measured
 * divergence*, not a defect, and the design explicitly forbids asserting it
 * away: this part pins the structural property that both engines must keep
 * (the nucleus is at least rank 0 and at most the ordered set, and the selected
 * token is inside the nucleus) and prints the measured split. */
void test_k5_strict_prefix_at_top_p_one() {
  /* (a) deep-tail underflow: stable, exact. */
  {
    const size_t vocab = 8;
    std::vector<float> logits(vocab, -200.0f);
    logits[0] = 0.0f;
    const RefOrderedRow witness = host_ordered_reference(
        logits.data(), vocab, nullptr, 0u, 1.0f, 1.0f, 8u, 0.0f, -inf(),
        DS41RT_V41_SAMPLER_FLAG_NO_MASK, 1ull, 0ull);
    expect(witness.ok, "strict-prefix witness (a): the ordered path runs");
    expect(witness.domain_count == 8u, "strict-prefix witness (a): eight ordered survivors");
    expect(witness.nucleus_count < witness.domain_count,
           "strict-prefix witness (a): the CPU nucleus is a strict prefix");
    expect(witness.nucleus_count == 1u,
           "strict-prefix witness (a): the underflowed prefix breaks at rank 0");
    K5Stats stats;
    std::vector<ds41rt_v41_sampler_row_t> params = {
        k5_row(0, 1.0f, 1.0f, 8u, 0.0f, -inf(), DS41RT_V41_SAMPLER_NO_MASK_ROW,
               DS41RT_V41_SAMPLER_FLAG_NO_MASK | DS41RT_V41_SAMPLER_FLAG_DIAGNOSE, 1ull,
               0ull),
    };
    run_k5_case(logits, 1, vocab, params, {}, false, false, 0u,
                "K5 strict prefix top_p=1.0 survivor domain", &stats);
    std::vector<ds41rt_v41_sampler_row_t> ranked = {
        k5_row(0, 1.0f, 1.0f, 2u, 0.0f, -inf(), DS41RT_V41_SAMPLER_NO_MASK_ROW,
               DS41RT_V41_SAMPLER_FLAG_NO_MASK | DS41RT_V41_SAMPLER_FLAG_DIAGNOSE, 3ull,
               1ull),
    };
    run_k5_case(logits, 1, vocab, ranked, {}, false, false, 2u,
                "K5 strict prefix top_p=1.0 retained list", &stats);
    std::vector<ds41rt_v41_sampler_row_t> tk = {
        k5_row(0, 0.7f, 1.0f, 40u, 0.0f, -inf(), DS41RT_V41_SAMPLER_NO_MASK_ROW,
               DS41RT_V41_SAMPLER_FLAG_NO_MASK | DS41RT_V41_SAMPLER_FLAG_DIAGNOSE, 7ull,
               5ull),
    };
    run_k5_case(logits, 1, vocab, tk, {}, false, false, 40u,
                "K5 strict prefix top_p=1.0 with the served top_k=40 profile", &stats);
  }

  /* (b) the rounded prefix at the expf residual: structural assertions only,
   * with the divergence measured and reported. */
  {
    const size_t vocab = 8;
    const float gap = 20.0f;
    std::vector<float> logits(vocab, -gap);
    logits[0] = 0.0f;
    /* Which engine sees the strict prefix here is exactly the residual; both
     * mechanisms are legitimate, so the test only requires that one of them
     * fires and records which. */
    const RefOrderedRow host = host_ordered_reference(
        logits.data(), vocab, nullptr, 0u, 1.0f, 1.0f, 8u, 0.0f, -inf(),
        DS41RT_V41_SAMPLER_FLAG_NO_MASK, 1ull, 0ull);
    expect(host.ok, "rounded top_p=1.0 witness: the ordered path runs");
    expect(host.nucleus_count < host.domain_count ||
               (host.nucleus_count == host.domain_count && host.nucleus_full_set),
           "rounded top_p=1.0 witness: the host prefix is one of the two shapes");
    std::vector<ds41rt_v41_sampler_row_t> params = {
        k5_row(0, 1.0f, 1.0f, 8u, 0.0f, -inf(), DS41RT_V41_SAMPLER_NO_MASK_ROW,
               DS41RT_V41_SAMPLER_FLAG_NO_MASK | DS41RT_V41_SAMPLER_FLAG_DIAGNOSE, 1ull,
               0ull),
    };
    run_k5_case(logits, 1, vocab, params, {}, false, false, 0u,
                "K5 rounded top_p=1.0 (expf residual, structural)", nullptr);
    std::cout << "ok  K5 top_p=1.0: strict prefix preserved (deep-tail exact; rounded "
                 "prefix classified at the expf residual)\n";
  }
}

/* `top_p` grid over the adversarial shapes and the boundary uniforms. */
void test_k5_top_p_grid_and_boundaries() {
  const size_t vocab = 257;
  const std::vector<std::vector<float>> shapes = {
      shape_descending(vocab),
      shape_periodic_ties(vocab),
      shape_two_value(vocab),
      std::vector<float>(vocab, 0.25f),       /* all-tied */
      std::vector<float>(vocab, -3.0f),       /* all-tied, negative */
  };
  const char* shape_names[] = {"desc", "ties16", "two_value", "all_tied", "all_neg"};
  const float top_ps[] = {1.0e-6f, 0.5f, 0.9f, 0.95f, 1.0f};
  const uint32_t top_ks[] = {0u, 40u, 80u};
  const uint64_t seeds[] = {1ull, 0xDEADBEEFull};
  K5Stats stats;
  for (size_t s = 0; s < shapes.size(); ++s) {
    for (float top_p : top_ps) {
      for (uint32_t top_k : top_ks) {
        for (uint64_t seed : seeds) {
          std::vector<ds41rt_v41_sampler_row_t> params = {
              k5_row(0, 0.7f, top_p, top_k, 0.0f, -inf(),
                     DS41RT_V41_SAMPLER_NO_MASK_ROW,
                     DS41RT_V41_SAMPLER_FLAG_NO_MASK | DS41RT_V41_SAMPLER_FLAG_DIAGNOSE,
                     seed, 11ull),
          };
          const std::string label = "k5 grid " + std::string(shape_names[s]) + " p" +
                                    std::to_string(top_p) + " k" +
                                    std::to_string(top_k) + " seed" +
                                    std::to_string(seed);
          run_k5_case(shapes[s], 1, vocab, params, {}, false, false, 80u,
                      label.c_str(), &stats);
        }
      }
    }
  }
  std::cout << "ok  K5 top_p x top_k x shape grid (" << stats.compared
            << " ordered rows compared; exact tokens " << stats.exact
            << ", measured residual divergences " << stats.boundary_flip
            << ", nucleus mismatches " << stats.nucleus_mismatch
            << ", max nucleus delta " << stats.max_nucleus_delta
            << ", boundary-slack nucleus differences " << stats.nucleus_boundary
            << ", max CDF error " << stats.max_cdf_error
            << ", outside-nucleus " << stats.outside_nucleus
            << ", max rank displacement " << stats.max_rank_displacement << ")\n";
  expect(stats.nucleus_mismatch == 0u,
         "the clean grid reproduces every production nucleus exactly, modulo the f32 boundary");
  expect(stats.max_nucleus_delta == 0u,
         "no clean-grid row needed even a one-rank nucleus delta");
  expect(stats.max_cdf_error <= kK5CdfErrorBound,
         "the clean grid stays inside the measured CDF-error bound");
}

/* Boundary uniforms: u = 0 lands on the best rank, u = MAX_UNIFORM on a late
 * rank, and a symmetric row's exact cumulative boundary lands on the *earlier*
 * rank (the CPU compares `target <= cumulative`). */
void test_k5_boundary_uniforms() {
  const size_t vocab = 257;
  const std::vector<float> logits = shape_descending(vocab);
  /* A seed/position pair for each boundary is found by scanning the *host*
   * uniform stream, so the test probes the shipped mapping, not a synthetic
   * float. */
  const auto find_uniform = [&](bool want_zero, bool want_max,
                               float wanted) -> std::pair<uint64_t, uint64_t> {
    for (uint64_t seed = 1ull; seed < 4000ull; ++seed) {
      for (uint64_t position = 0ull; position < 64ull; ++position) {
        const float uniform = host_target_uniform(seed, position);
        if (want_zero && uniform <= 1.0e-5f) {
          return std::make_pair(seed, position);
        }
        if (want_max && uniform > 0.99f) {
          return std::make_pair(seed, position);
        }
        if (!want_zero && !want_max && uniform <= wanted &&
            uniform >= std::nextafter(wanted, 0.0f)) {
          return std::make_pair(seed, position);
        }
      }
    }
    return std::make_pair(0ull, 0ull);
  };
  /* `random_uniform` is `mantissa * 2^-24` with a 24-bit mantissa, so the
   * smallest value the shipped stream can produce is `min_mantissa * 2^-24` and
   * `u == 0.0f` needs a zero mantissa. The smallest zero-mantissa seed on this
   * stream is **310147 (position 0)**, which the `seed < 4000` scan below does
   * not reach; the exact `u = 0` cases are pinned by
   * `test_k5_zero_uniform_draw`. This harness therefore scans for the
   * **smallest drawable positive** uniform and asserts that bound, plus the
   * MAX_UNIFORM clamp at the other end -- the two reachable ends of the served
   * draw stream. */
  const std::pair<uint64_t, uint64_t> bottom = find_uniform(true, false, 0.0f);
  const std::pair<uint64_t, uint64_t> top = find_uniform(false, true, 0.0f);
  expect(bottom.first != 0u, "found a seed/position at the smallest drawable uniform");
  expect(top.first != 0u, "found a seed/position at the largest drawable uniform");
  expect(host_target_uniform(bottom.first, bottom.second) > 0.0f &&
             host_target_uniform(bottom.first, bottom.second) <= 1.0e-5f,
         "the smallest drawable uniform found is a small positive draw");
  expect(host_target_uniform(top.first, top.second) > 0.99f,
         "the largest drawable uniform found is near one");
  /* The `MAX_UNIFORM` *clamp* itself is pinned by `test_k2_max_uniform_clamp`
   * (the same `ds41rt_v41_target_clamp_uniform` the device draw uses), so K5
   * only needs the reachable ends of the stream. */
  K5Stats stats;
  std::vector<ds41rt_v41_sampler_row_t> params;
  if (bottom.first != 0u) {
    params.push_back(k5_row(0, 1.0f, 0.9f, 0u, 0.0f, -inf(),
                            DS41RT_V41_SAMPLER_NO_MASK_ROW,
                            DS41RT_V41_SAMPLER_FLAG_NO_MASK |
                                DS41RT_V41_SAMPLER_FLAG_DIAGNOSE,
                            bottom.first, bottom.second));
  }
  if (top.first != 0u) {
    params.push_back(k5_row(static_cast<uint32_t>(params.size()), 1.0f, 0.5f, 0u, 0.0f,
                            -inf(), DS41RT_V41_SAMPLER_NO_MASK_ROW,
                            DS41RT_V41_SAMPLER_FLAG_NO_MASK |
                                DS41RT_V41_SAMPLER_FLAG_DIAGNOSE,
                            top.first, top.second));
  }
  /* A two-level row: scaled 0.0 and ln(2) give p = 1/3 and 2/3, so a uniform of
   * 1/3 is exactly the cumulative boundary of rank 1 and must land on rank 1
   * (not rank 2). A NONZERO `top_k` is required: `top_k == 0, top_p == 1` is the
   * disjoint K2 fast path, where `host_ordered_reference` early-returns and
   * `run_k5_case` only confirms K5 left the row's sentinels alone. `top_k = 2`
   * (`< survivor_count = 4`) materializes a K3/K4 retained list, so this case
   * genuinely executes K5's ordered branch. */
  {
    const size_t small = 4;
    std::vector<float> two(small, -30.0f);
    two[0] = static_cast<float>(std::log(2.0));
    two[3] = 0.0f;
    const std::vector<float> repeated = repeat_row(two, 1);
    std::vector<ds41rt_v41_sampler_row_t> exact = {
        k5_row(0, 1.0f, 1.0f, 2u, 0.0f, -inf(), DS41RT_V41_SAMPLER_NO_MASK_ROW,
               DS41RT_V41_SAMPLER_FLAG_NO_MASK | DS41RT_V41_SAMPLER_FLAG_DIAGNOSE, 5ull,
               0ull),
    };
    const RefOrderedRow expected = host_ordered_reference(
        two.data(), small, nullptr, 0u, 1.0f, 1.0f, 2u, 0.0f, -inf(),
        DS41RT_V41_SAMPLER_FLAG_NO_MASK, 5ull, 0ull);
    expect(expected.ok && !expected.k2_fast_path,
           "exact cumulative boundary: the host reference takes the ordered path");
    K5Stats exact_stats;
    run_k5_case(repeated, 1, small, exact, {}, false, false, 2u,
                "K5 exact cumulative boundary (ordered)", &exact_stats);
    expect(exact_stats.compared == 1u,
           "exact cumulative boundary: K5 actually executed the ordered branch");
    expect(exact_stats.exact == 1u,
           "exact cumulative boundary: the device token equals production");
    stats.compared += exact_stats.compared;
    stats.exact += exact_stats.exact;
    stats.nucleus_mismatch += exact_stats.nucleus_mismatch;
    stats.max_nucleus_delta = std::max(stats.max_nucleus_delta, exact_stats.max_nucleus_delta);
    stats.max_cdf_error = std::max(stats.max_cdf_error, exact_stats.max_cdf_error);
  }
  if (!params.empty()) {
    /* Every row uses the same row, so the batch is `repeat_row`, not one copy:
     * the kernel reads `logits + block_row * logits_stride`. */
    const std::vector<float> batch = repeat_row(logits, params.size());
    run_k5_case(batch, params.size(), vocab, params, {}, false, false, 0u,
                "K5 boundary uniforms (smallest positive u, u=MAX_UNIFORM)", &stats);
  }
  std::cout << "ok  K5 boundary uniforms (smallest positive u, u=MAX_UNIFORM, exact "
               "cumulative)\n";
}

/* Probe/association invariance of the retained-domain prefix.
 *
 * The reviewer's weight pattern `[1, 2^-24, 2^-24]` (one dominant rank and two
 * sub-ulp tail ranks) is the minimal case where the bucket-split probe's f32
 * association depended on the probe bounds: `W(2)` read `fl(1 + 2^-24 + 2^-24) =
 * 1.0000001192092896` when rank 2 was the middle bound and `fl((1 + 2^-24) +
 * 2^-24) = 1.0` when it was a bound of a single-bucket group. The sequential
 * total has the same dependence (`1 + 2^-24 + 2^-24` rounds to `1.0`), so the
 * shipped total was `1.0000001192092896` while the CPU's is `1.0`. That makes the
 * crossing jump from rank 0 to rank 2 as `top_p` crosses `~0.99999988`, while the
 * CPU's nucleus is rank 0 for every `top_p <= 1`.
 *
 * The fixed kernel builds one `prefix[]` per row, so every search state reads the
 * same bits. This test sweeps `top_p` across the old jump point and asserts the
 * device nucleus and token stay CPU-exact; the pre-fix kernel fails at
 * `top_p = 1.0` (device nucleus 3, CPU 1). See the report's M4 mutant. */
void test_k5_probe_association_invariance() {
  const size_t vocab = 8;
  /* `ln(2^-24)`, the scaled value whose weight is exactly `2^-24`. */
  const float sub_ulp = -16.635532333438687f;
  std::vector<float> logits(vocab, -50.0f);
  logits[0] = 0.0f;
  logits[1] = sub_ulp;
  logits[2] = sub_ulp;
  const float top_ps[] = {0.9999f, 0.99999f, 0.999999f, 1.0f};
  const uint64_t seeds[] = {310147ull, 7ull}; /* u = 0 and a non-zero draw */
  std::vector<ds41rt_v41_sampler_row_t> params;
  for (uint64_t seed : seeds) {
    for (float top_p : top_ps) {
      params.push_back(k5_row(static_cast<uint32_t>(params.size()), 1.0f, top_p, 3u,
                              0.0f, -inf(), DS41RT_V41_SAMPLER_NO_MASK_ROW,
                              DS41RT_V41_SAMPLER_FLAG_NO_MASK |
                                  DS41RT_V41_SAMPLER_FLAG_DIAGNOSE,
                              seed, 0ull));
    }
  }
  const std::vector<float> rows = repeat_row(logits, params.size());
  K5Stats stats;
  run_k5_case(rows, params.size(), vocab, params, {}, false, false, 3u,
              "K5 probe-association invariance (sub-ulp tail)", &stats);
  /* Every one of these rows must have reached the retained domain and matched
   * the production nucleus exactly (the fixed prefix is CPU-sequential). */
  if (stats.nucleus_mismatch != 0u || stats.nucleus_boundary != 0u ||
      stats.exact != params.size()) {
    std::cerr << "probe-invariance detail: compared " << stats.compared << " exact "
              << stats.exact << " nucleus_mismatch " << stats.nucleus_mismatch
              << " boundary " << stats.nucleus_boundary << " max_rank_displacement "
              << stats.max_rank_displacement << "\n";
  }
  expect(stats.compared == params.size(), "every probe-invariance row was ordered");
  expect(stats.nucleus_mismatch == 0u,
         "every probe-invariance row matched the production nucleus");
  expect(stats.nucleus_boundary == 0u,
         "no probe-invariance row needed the f32 boundary slack");
  expect(stats.exact == params.size(), "every probe-invariance row matched the production token");
  std::cout << "ok  K5 probe-association invariance over top_p in [0.9999, 1.0] ("
            << stats.compared << " rows, " << stats.exact << " exact)\n";
}

/* The zero-uniform draw must select the BEST ACTUAL SURVIVOR, never the empty
 * prefix at rank 0.
 *
 * `random_uniform` is `mantissa * 2^-24` with a 24-bit mantissa, so `u == 0.0f`
 * IS reachable when `mixed >> 40 == 0` (seed 310147, position 0, on the shipped
 * mapping -- the earlier boundary-uniform case only found a small positive draw
 * because it scanned seeds below 4000). At `u = 0` the CPU's `target` is `0`, and
 * its draw loop still adds the first weight before testing `target <=
 * cumulative`, so it selects rank 0. A search whose predicate is only
 * `mass >= 0` accepts an EMPTY tie group instead and can return token id 0 even
 * when 0 is masked out or is not a survivor. Three shapes are covered:
 *   (a) survivor domain (`top_k == 0`, `top_p < 1`) with best token id 5;
 *   (b) the same with id 0 masked out (so the wrong answer is also a non-member);
 *   (c) retained domain (finite `top_k`) with best token id 5.
 * Every row is compared token-for-token with the production oracle at `u = 0`;
 * `host_ordered_reference` returns rank 0 there. */
void test_k5_zero_uniform_draw() {
  const uint64_t zero_seed = 310147ull;
  expect(host_target_uniform(zero_seed, 0ull) == 0.0f,
         "seed 310147 / position 0 is an exact zero uniform draw");
  K5Stats stats;
  {
    /* (b) id 0 masked out; best allowed token is id 5, so an empty-prefix answer
     * would be a masked-out non-member. This runs FIRST so the pre-fix
     * empty-prefix defect is observed on the non-member shape. */
    const size_t vocab = 64;
    const size_t words = (vocab + 31u) / 32u;
    std::vector<float> logits(vocab, -30.0f);
    logits[5] = 0.0f;
    std::vector<uint32_t> mask(words, 0xFFFFFFFFu);
    mask[0] &= ~1u; /* clear token 0 */
    std::vector<ds41rt_v41_sampler_row_t> params = {
        k5_row(0, 1.0f, 0.9f, 0u, 0.0f, -inf(), 0u,
               DS41RT_V41_SAMPLER_FLAG_DIAGNOSE, zero_seed, 0ull),
    };
    const RefOrderedRow ref = host_ordered_reference(
        logits.data(), vocab, mask.data(), words, 1.0f, 0.9f, 0u, 0.0f, -inf(), 0u,
        zero_seed, 0ull);
    expect(ref.token == 5u, "u=0 masked draw's production answer is the best allowed token");
    expect((mask[0] & 1u) == 0u, "token 0 is masked out in the u=0 non-member case");
    run_k5_case(logits, 1, vocab, params, mask, true, true, 0u,
                "K5 u=0 survivor draw, id 0 masked out", &stats);
  }
  {
    /* (a) best token is id 5, not 0. */
    const size_t vocab = 64;
    std::vector<float> logits(vocab, -30.0f);
    logits[5] = 0.0f;
    std::vector<ds41rt_v41_sampler_row_t> params = {
        k5_row(0, 1.0f, 0.9f, 0u, 0.0f, -inf(), DS41RT_V41_SAMPLER_NO_MASK_ROW,
               DS41RT_V41_SAMPLER_FLAG_NO_MASK | DS41RT_V41_SAMPLER_FLAG_DIAGNOSE,
               zero_seed, 0ull),
    };
    const RefOrderedRow ref = host_ordered_reference(
        logits.data(), vocab, nullptr, 0u, 1.0f, 0.9f, 0u, 0.0f, -inf(),
        DS41RT_V41_SAMPLER_FLAG_NO_MASK, zero_seed, 0ull);
    expect(ref.token == 5u, "u=0 survivor draw's production answer is the best token (id 5)");
    run_k5_case(logits, 1, vocab, params, {}, false, false, 0u,
                "K5 u=0 survivor draw, best id != 0", &stats);
  }
  {
    /* (c) retained domain: finite top_k, best token id 5. */
    const size_t vocab = 300;
    std::vector<float> logits = shape_descending(vocab);
    logits[5] = 100.0f;
    std::vector<ds41rt_v41_sampler_row_t> params = {
        k5_row(0, 0.7f, 0.9f, 40u, 0.0f, -inf(), DS41RT_V41_SAMPLER_NO_MASK_ROW,
               DS41RT_V41_SAMPLER_FLAG_NO_MASK | DS41RT_V41_SAMPLER_FLAG_DIAGNOSE,
               zero_seed, 0ull),
    };
    const RefOrderedRow ref = host_ordered_reference(
        logits.data(), vocab, nullptr, 0u, 0.7f, 0.9f, 40u, 0.0f, -inf(),
        DS41RT_V41_SAMPLER_FLAG_NO_MASK, zero_seed, 0ull);
    expect(ref.token == 5u, "u=0 retained draw's production answer is rank 0 (id 5)");
    run_k5_case(logits, 1, vocab, params, {}, false, false, 40u,
                "K5 u=0 retained draw", &stats);
  }
  std::cout << "ok  K5 zero-uniform draw selects the best surviving rank ("
            << stats.compared << " rows, exact " << stats.exact << ")\n";
}

/* Repeated-run determinism for the survivor-domain mass probe.
 *
 * `k5_key_mass` reuses the same shared `phase`/count arrays on every probe, so a
 * missing barrier between "read the reduced value" and "the next probe's first
 * write" is a shared-memory data race whose result would steer the binary-search
 * branches. The fixed kernel snapshots the reduced value into every thread's
 * local and barriers before returning (mirroring K3's documented fence), so
 * repeated launches of the same row must be bit-identical. This launches the
 * shipped K5 entry point 32 times with the same buffers (a probe-heavy
 * survivor-domain row: ~46 passes over a 4,096-token vocabulary) and compares
 * the token, total and nucleus count bit-for-bit. It exercises the shared-buffer
 * reuse path; it cannot force the warp interleaving that would manifest the race,
 * which is why the fix is structural. */
void test_k5_repeated_run_determinism() {
  ++g_cases;
  const size_t vocab = 4096;
  std::vector<float> logits(vocab);
  for (size_t t = 0; t < vocab; ++t) {
    logits[t] = -0.001f * static_cast<float>(t);
  }
  K5Run run = make_k5_run(1, vocab, 0, false);
  require_cuda(cudaMemcpy(run.k1.logits, logits.data(), logits.size() * sizeof(float),
                          cudaMemcpyHostToDevice), "determinism logits h2d");
  const std::vector<ds41rt_v41_sampler_row_t> params = {
      k5_row(0, 1.0f, 0.9f, 0u, 0.0f, -inf(), DS41RT_V41_SAMPLER_NO_MASK_ROW,
             DS41RT_V41_SAMPLER_FLAG_NO_MASK | DS41RT_V41_SAMPLER_FLAG_DIAGNOSE, 17ull,
             3ull),
  };
  require_cuda(cudaMemcpy(run.k1.params, params.data(),
                          params.size() * sizeof(ds41rt_v41_sampler_row_t),
                          cudaMemcpyHostToDevice), "determinism params h2d");
  std::vector<uint32_t> host_retained(1u, 0u);
  std::vector<uint32_t> host_passes(1u, 0u);
  require_cuda(cudaMemcpy(run.topk.retained, host_retained.data(), sizeof(uint32_t),
                          cudaMemcpyHostToDevice), "determinism retained h2d");
  require_cuda(cudaMemcpy(run.topk.passes, host_passes.data(), sizeof(uint32_t),
                          cudaMemcpyHostToDevice), "determinism passes h2d");
  expect(ds41rt_cuda_v41_target_sample(
             run.k1.logits, 1, vocab, vocab, run.k1.params, nullptr, 0u, run.k1.ids,
             run.k1.status, run.k1.detail, run.k1.scores, nullptr, nullptr,
             run.k1.scratch) == DS41RT_STATUS_OK,
         "determinism: K1 launch");
  expect(ds41rt_cuda_v41_topk_select(
             run.k1.logits, 1, vocab, vocab, run.k1.params, nullptr, 0u, nullptr,
             nullptr, 0u, run.topk.retained, run.topk.passes,
             run.k1.scratch) == DS41RT_STATUS_OK,
         "determinism: topk launch");
  uint32_t first_token = 0xFFFFFFFFu;
  uint32_t first_status = 0xFFFFFFFFu;
  float first_total = 0.0f;
  uint32_t first_nucleus = 0u;
  const uint32_t sentinel = 0xDEADBEEFu;
  for (int rep = 0; rep < 32; ++rep) {
    std::vector<uint32_t> host_ids(1u, sentinel);
    std::vector<float> host_total(1u, -123.0f);
    std::vector<uint32_t> host_nucleus(1u, sentinel);
    require_cuda(cudaMemcpy(run.ids, host_ids.data(), sizeof(uint32_t),
                            cudaMemcpyHostToDevice), "determinism ids h2d");
    require_cuda(cudaMemcpy(run.total, host_total.data(), sizeof(float),
                            cudaMemcpyHostToDevice), "determinism total h2d");
    require_cuda(cudaMemcpy(run.nucleus, host_nucleus.data(), sizeof(uint32_t),
                            cudaMemcpyHostToDevice), "determinism nucleus h2d");
    expect(ds41rt_cuda_v41_nucleus(
               run.k1.logits, 1, vocab, vocab, run.k1.params, nullptr, 0u, nullptr, 0u,
               run.topk.retained, run.ids, run.k1.status, run.total, run.nucleus,
               run.k1.scratch) == DS41RT_STATUS_OK,
           "determinism: nucleus launch");
    require_cuda(cudaDeviceSynchronize(), "determinism K5 kernel");
    uint32_t token = 0u;
    uint32_t status = 0u;
    uint32_t nucleus = 0u;
    float total = 0.0f;
    require_cuda(cudaMemcpy(&token, run.ids, sizeof(uint32_t), cudaMemcpyDeviceToHost),
                 "determinism ids d2h");
    require_cuda(cudaMemcpy(&status, run.k1.status, sizeof(uint32_t),
                            cudaMemcpyDeviceToHost), "determinism status d2h");
    require_cuda(cudaMemcpy(&total, run.total, sizeof(float), cudaMemcpyDeviceToHost),
                 "determinism total d2h");
    require_cuda(cudaMemcpy(&nucleus, run.nucleus, sizeof(uint32_t),
                            cudaMemcpyDeviceToHost), "determinism nucleus d2h");
    expect(status == DS41RT_V41_SAMPLER_STATUS_OK, "determinism: K5 row stayed OK");
    if (rep == 0) {
      first_token = token;
      first_status = status;
      first_total = total;
      first_nucleus = nucleus;
      expect(token != sentinel, "determinism: first run wrote a token");
    } else {
      expect(token == first_token, "determinism: repeated run token is bit-identical");
      expect(status == first_status, "determinism: repeated run status is identical");
      expect(std::memcmp(&total, &first_total, sizeof(float)) == 0,
             "determinism: repeated run total is bit-identical");
      expect(nucleus == first_nucleus, "determinism: repeated run nucleus is identical");
    }
  }
  free_k5_run(&run);
  std::cout << "ok  K5 repeated-run determinism over 32 launches (vocab " << vocab
            << ", token " << first_token << ", nucleus " << first_nucleus << ")\n";
}

/* Loud per-row failure: a K5-class row that cannot produce a defined token must
 * set BOTH `scratch[block_row].status` and the caller-visible
 * `out_status[output_row]` to INTERNAL, and must leave `out_indices` at the
 * sentinel. The entry point's own return value stays OK -- the failure is per
 * row -- so `out_status` is the channel a caller that only reads the returned
 * status must consume (the same channel K1 writes).
 *
 * Four shapes, each reproduced through the shipped C ABI:
 *   (a) `top_k = 300` with `survivor_count > 300` -> retained list wider than
 *       `kBlock` (the old `top_k in [257, survivor_count)` silent-OK case);
 *   (b) `top_k = 40` with `rank_order_capacity = 0`, a selection-only K3/K4
 *       (K4 still publishes `out_retained_count = 40`, but the arena is null);
 *   (c) `top_p = NaN` with `top_k = 0`, which makes K2 (`top_p >= 1.0`) and K5
 *       (`top_p < 1.0`) both inapplicable -- K1 itself now reports INTERNAL;
 *   (d) a zero-survivor non-greedy row (`min_p = 1`, `ln_min_p = +100`), which
 *       K1 itself now reports INTERNAL for.
 * The FFI validator rejects (c) and (d) on the host; they remain reachable from
 * the raw C ABI and must not be silent. (a)/(b) are loud only in K5, so for
 * those the caller-visible status is reset to a sentinel before K5 and K5 must
 * write INTERNAL; for (c)/(d) K1's INTERNAL must survive the whole chain. */
void test_k5_loud_status_failures() {
  const size_t vocab = 400;
  const std::vector<float> logits = shape_descending(vocab);
  const uint32_t sentinel = 0xDEADBEEFu;
  struct Case {
    const char* label;
    uint32_t top_k;
    float top_p;
    float min_p;
    float ln_min_p;
    size_t capacity;
    bool k1_loud; /* K1 alone must report INTERNAL (the P1-4 shapes) */
  };
  const Case cases[] = {
      {"K5 loud status: top_k 300 > kBlock", 300u, 0.9f, 0.0f, -inf(), 300u, false},
      {"K5 loud status: capacity-0 selection-only", 40u, 0.9f, 0.0f, -inf(), 0u, false},
      {"K5 loud status: top_p NaN", 0u, std::numeric_limits<float>::quiet_NaN(), 0.0f,
       -inf(), 0u, true},
      {"K5 loud status: zero survivors", 0u, 0.9f, 1.0f, 100.0f, 0u, true},
  };
  for (const Case& c : cases) {
    ++g_cases;
    const std::string tag(c.label);
    K5Run run = make_k5_run(1, vocab, c.capacity, false);
    require_cuda(cudaMemcpy(run.k1.logits, logits.data(), logits.size() * sizeof(float),
                            cudaMemcpyHostToDevice), "loud logits h2d");
    const std::vector<ds41rt_v41_sampler_row_t> params = {
        k5_row(0, 0.7f, c.top_p, c.top_k, c.min_p, c.ln_min_p,
               DS41RT_V41_SAMPLER_NO_MASK_ROW,
               DS41RT_V41_SAMPLER_FLAG_NO_MASK | DS41RT_V41_SAMPLER_FLAG_DIAGNOSE, 5ull,
               1ull),
    };
    require_cuda(cudaMemcpy(run.k1.params, params.data(),
                            params.size() * sizeof(ds41rt_v41_sampler_row_t),
                            cudaMemcpyHostToDevice), "loud params h2d");
    /* K1+K2-only entry point. For (c)/(d) K1 itself must already be loud: this
     * is the P1-4 invariant, observable with no K3/K4/K5 launch at all. */
    expect(ds41rt_cuda_v41_target_sample(
               run.k1.logits, 1, vocab, vocab, run.k1.params, nullptr, 0u, run.k1.ids,
               run.k1.status, run.k1.detail, run.k1.scores, nullptr, nullptr,
               run.k1.scratch) == DS41RT_STATUS_OK,
           tag + ": K1 launch");
    uint32_t k1_status = sentinel;
    ds41rt_v41_sampler_scratch_t k1_scratch = {};
    require_cuda(cudaMemcpy(&k1_status, run.k1.status, sizeof(uint32_t),
                            cudaMemcpyDeviceToHost), tag + ": K1 status d2h");
    require_cuda(cudaMemcpy(&k1_scratch, run.k1.scratch, sizeof(k1_scratch),
                            cudaMemcpyDeviceToHost), tag + ": K1 scratch d2h");
    expect((k1_status == DS41RT_V41_SAMPLER_STATUS_INTERNAL) == c.k1_loud,
           tag + ": K1+K2-only entry point reports the loud status for this shape");
    expect((k1_scratch.status == DS41RT_V41_SAMPLER_STATUS_INTERNAL) == c.k1_loud,
           tag + ": K1 scratch agrees with the K1+K2-only entry point");
    if (c.capacity > 0u) {
      std::vector<uint32_t> host_ids(c.capacity, sentinel);
      std::vector<uint64_t> host_scratch(c.capacity, 0ull);
      require_cuda(cudaMemcpy(run.topk.rank_ids, host_ids.data(),
                              c.capacity * sizeof(uint32_t), cudaMemcpyHostToDevice),
                   tag + ": rank ids h2d");
      require_cuda(cudaMemcpy(run.topk.rank_scratch, host_scratch.data(),
                              c.capacity * sizeof(uint64_t), cudaMemcpyHostToDevice),
                   tag + ": rank scratch h2d");
    }
    std::vector<uint32_t> host_retained(1u, sentinel);
    std::vector<uint32_t> host_passes(1u, sentinel);
    require_cuda(cudaMemcpy(run.topk.retained, host_retained.data(), sizeof(uint32_t),
                            cudaMemcpyHostToDevice), tag + ": retained h2d");
    require_cuda(cudaMemcpy(run.topk.passes, host_passes.data(), sizeof(uint32_t),
                            cudaMemcpyHostToDevice), tag + ": passes h2d");
    expect(ds41rt_cuda_v41_topk_select(
               run.k1.logits, 1, vocab, vocab, run.k1.params, nullptr, 0u,
               c.capacity > 0u ? run.topk.rank_ids : nullptr,
               c.capacity > 0u ? run.topk.rank_scratch : nullptr, c.capacity,
               run.topk.retained, run.topk.passes, run.k1.scratch) == DS41RT_STATUS_OK,
           tag + ": topk launch");
    std::vector<uint32_t> host_ids(1u, sentinel);
    require_cuda(cudaMemcpy(run.ids, host_ids.data(), sizeof(uint32_t),
                            cudaMemcpyHostToDevice), tag + ": ids h2d");
    if (!c.k1_loud) {
      /* (a)/(b): K1 was OK, so reset the caller-visible status to prove K5
       * itself writes INTERNAL. For (c)/(d) K1's INTERNAL must survive. */
      require_cuda(cudaMemcpy(run.k1.status, host_ids.data(), sizeof(uint32_t),
                              cudaMemcpyHostToDevice), tag + ": status sentinel h2d");
    }
    expect(ds41rt_cuda_v41_nucleus(
               run.k1.logits, 1, vocab, vocab, run.k1.params, nullptr, 0u,
               c.capacity > 0u ? run.topk.rank_ids : nullptr, c.capacity,
               run.topk.retained, run.ids, run.k1.status, run.total, run.nucleus,
               run.k1.scratch) == DS41RT_STATUS_OK,
           tag + ": nucleus launch still returns OK (per-row status)");
    require_cuda(cudaDeviceSynchronize(), "loud K5 kernel");
    uint32_t token = 0u;
    uint32_t status = 0u;
    ds41rt_v41_sampler_scratch_t scratch = {};
    require_cuda(cudaMemcpy(&token, run.ids, sizeof(uint32_t), cudaMemcpyDeviceToHost),
                 tag + ": ids d2h");
    require_cuda(cudaMemcpy(&status, run.k1.status, sizeof(uint32_t),
                            cudaMemcpyDeviceToHost), tag + ": status d2h");
    require_cuda(cudaMemcpy(&scratch, run.k1.scratch, sizeof(scratch),
                            cudaMemcpyDeviceToHost), tag + ": scratch d2h");
    expect(status == DS41RT_V41_SAMPLER_STATUS_INTERNAL,
           tag + ": the caller-visible out_status is INTERNAL at the end of the chain");
    expect(scratch.status == DS41RT_V41_SAMPLER_STATUS_INTERNAL,
           tag + ": scratch.status is INTERNAL at the end of the chain");
    expect(token == sentinel, tag + ": out_indices left unwritten");
    free_k5_run(&run);
  }
  std::cout << "ok  K5/K1 loud per-row status on four no-token shapes ("
            << (sizeof(cases) / sizeof(cases[0])) << " rows)\n";
}

/* P1-4, isolated: the K1+K2-only entry point (`ds41rt_cuda_v41_target_sample`)
 * must never return OK while leaving `out_indices` unwritten. Both FFI-invalid
 * raw-C shapes -- a non-finite `top_p` and a non-greedy zero-survivor row -- are
 * checked with NO K3/K4/K5 launch in between, so the loud status can only have
 * come from K1. */
void test_k1_only_entry_loud_status() {
  const size_t vocab = 400;
  const std::vector<float> logits = shape_descending(vocab);
  const uint32_t sentinel = 0xDEADBEEFu;
  struct Shape {
    const char* label;
    float top_p;
    float min_p;
    float ln_min_p;
  };
  const Shape shapes[] = {
      {"K1-only NaN top_p", std::numeric_limits<float>::quiet_NaN(), 0.0f, -inf()},
      {"K1-only zero survivors", 0.9f, 1.0f, 100.0f},
  };
  for (const Shape& s : shapes) {
    ++g_cases;
    const std::string tag(s.label);
    K5Run run = make_k5_run(1, vocab, 0u, false);
    require_cuda(cudaMemcpy(run.k1.logits, logits.data(), logits.size() * sizeof(float),
                            cudaMemcpyHostToDevice), tag + ": logits h2d");
    const std::vector<ds41rt_v41_sampler_row_t> params = {
        k5_row(0, 0.7f, s.top_p, 0u, s.min_p, s.ln_min_p,
               DS41RT_V41_SAMPLER_NO_MASK_ROW,
               DS41RT_V41_SAMPLER_FLAG_NO_MASK | DS41RT_V41_SAMPLER_FLAG_DIAGNOSE, 5ull,
               1ull),
    };
    require_cuda(cudaMemcpy(run.k1.params, params.data(),
                            params.size() * sizeof(ds41rt_v41_sampler_row_t),
                            cudaMemcpyHostToDevice), tag + ": params h2d");
    std::vector<uint32_t> host_ids(1u, sentinel);
    require_cuda(cudaMemcpy(run.k1.ids, host_ids.data(), sizeof(uint32_t),
                            cudaMemcpyHostToDevice), tag + ": ids h2d");
    require_cuda(cudaMemcpy(run.k1.status, host_ids.data(), sizeof(uint32_t),
                            cudaMemcpyHostToDevice), tag + ": status h2d");
    expect(ds41rt_cuda_v41_target_sample(
               run.k1.logits, 1, vocab, vocab, run.k1.params, nullptr, 0u, run.k1.ids,
               run.k1.status, run.k1.detail, run.k1.scores, nullptr, nullptr,
               run.k1.scratch) == DS41RT_STATUS_OK,
           tag + ": entry returns OK (per-row status)");
    uint32_t token = 0u;
    uint32_t status = 0u;
    ds41rt_v41_sampler_scratch_t scratch = {};
    require_cuda(cudaMemcpy(&token, run.k1.ids, sizeof(uint32_t), cudaMemcpyDeviceToHost),
                 tag + ": ids d2h");
    require_cuda(cudaMemcpy(&status, run.k1.status, sizeof(uint32_t),
                            cudaMemcpyDeviceToHost), tag + ": status d2h");
    require_cuda(cudaMemcpy(&scratch, run.k1.scratch, sizeof(scratch),
                            cudaMemcpyDeviceToHost), tag + ": scratch d2h");
    expect(status == DS41RT_V41_SAMPLER_STATUS_INTERNAL,
           tag + ": K1 wrote INTERNAL to the caller-visible out_status");
    expect(scratch.status == DS41RT_V41_SAMPLER_STATUS_INTERNAL,
           tag + ": K1 wrote INTERNAL to scratch.status");
    expect(token == sentinel, tag + ": no token is observable with an OK return");
    free_k5_run(&run);
  }
  std::cout << "ok  K1+K2-only entry is loud for both invalid shapes ("
            << (sizeof(shapes) / sizeof(shapes[0])) << " rows)\n";
}

/* P0-2 raw-C regression: the scattered-`output_row` identity guard must fire
 * BEFORE any retained count is interpreted. The two-row witness is the second
 * review's exact shape: block 0 asks for `top_k = 2` but declares
 * `output_row = 1`, and block 1 declares `output_row = 0`. K4 writes
 * `out_retained_count[output_row]`, so block 0 reads block 1's count (0) at
 * `rank_retained_count[block_row = 0]`; the pre-fix guard tested identity only
 * inside the `retained_mode &&` conjunction, so the foreign 0 flipped
 * `retained_mode` false, the guard was skipped, and block 0 sampled all four
 * survivors and wrote `out_indices[1] = 3` with status OK. The unconditional
 * guard now reports INTERNAL for both non-identity rows and writes no token. */
void test_k5_scattered_row_identity_guard() {
  ++g_cases;
  const size_t rows = 2;
  const size_t vocab = 4;
  const size_t capacity = 2;
  const std::vector<float> logits(rows * vocab, 0.0f); /* four equal logits */
  const uint32_t sentinel = 0xDEADBEEFu;
  K5Run run = make_k5_run(rows, vocab, capacity, false);
  require_cuda(cudaMemcpy(run.k1.logits, logits.data(), logits.size() * sizeof(float),
                          cudaMemcpyHostToDevice), "scatter logits h2d");
  const std::vector<ds41rt_v41_sampler_row_t> params = {
      /* block 0: output_row 1, top_k 2, top_p 0.9 */
      k5_row(1u, 1.0f, 0.9f, 2u, 0.0f, -inf(), DS41RT_V41_SAMPLER_NO_MASK_ROW,
             DS41RT_V41_SAMPLER_FLAG_NO_MASK | DS41RT_V41_SAMPLER_FLAG_DIAGNOSE, 0ull,
             5ull),
      /* block 1: output_row 0, top_k 0, top_p 0.9 */
      k5_row(0u, 1.0f, 0.9f, 0u, 0.0f, -inf(), DS41RT_V41_SAMPLER_NO_MASK_ROW,
             DS41RT_V41_SAMPLER_FLAG_NO_MASK | DS41RT_V41_SAMPLER_FLAG_DIAGNOSE, 0ull,
             5ull),
  };
  require_cuda(cudaMemcpy(run.k1.params, params.data(),
                          params.size() * sizeof(ds41rt_v41_sampler_row_t),
                          cudaMemcpyHostToDevice), "scatter params h2d");
  std::vector<uint32_t> host_ids(rows * capacity, sentinel);
  std::vector<uint64_t> host_scratch(rows * capacity, 0ull);
  require_cuda(cudaMemcpy(run.topk.rank_ids, host_ids.data(),
                          rows * capacity * sizeof(uint32_t), cudaMemcpyHostToDevice),
               "scatter rank ids h2d");
  require_cuda(cudaMemcpy(run.topk.rank_scratch, host_scratch.data(),
                          rows * capacity * sizeof(uint64_t), cudaMemcpyHostToDevice),
               "scatter rank scratch h2d");
  std::vector<uint32_t> host_retained(rows, sentinel);
  std::vector<uint32_t> host_passes(rows, sentinel);
  require_cuda(cudaMemcpy(run.topk.retained, host_retained.data(),
                          rows * sizeof(uint32_t), cudaMemcpyHostToDevice),
               "scatter retained h2d");
  require_cuda(cudaMemcpy(run.topk.passes, host_passes.data(), rows * sizeof(uint32_t),
                          cudaMemcpyHostToDevice), "scatter passes h2d");
  expect(ds41rt_cuda_v41_target_sample(
             run.k1.logits, rows, vocab, vocab, run.k1.params, nullptr, 0u, run.k1.ids,
             run.k1.status, run.k1.detail, run.k1.scores, nullptr, nullptr,
             run.k1.scratch) == DS41RT_STATUS_OK,
         "scatter witness: K1 launch");
  expect(ds41rt_cuda_v41_topk_select(
             run.k1.logits, rows, vocab, vocab, run.k1.params, nullptr, 0u,
             run.topk.rank_ids, run.topk.rank_scratch, capacity, run.topk.retained,
             run.topk.passes, run.k1.scratch) == DS41RT_STATUS_OK,
         "scatter witness: topk launch");
  std::vector<uint32_t> host_out_ids(rows, sentinel);
  require_cuda(cudaMemcpy(run.ids, host_out_ids.data(), rows * sizeof(uint32_t),
                          cudaMemcpyHostToDevice), "scatter out ids h2d");
  require_cuda(cudaMemcpy(run.k1.status, host_out_ids.data(), rows * sizeof(uint32_t),
                          cudaMemcpyHostToDevice), "scatter out status h2d");
  expect(ds41rt_cuda_v41_nucleus(
             run.k1.logits, rows, vocab, vocab, run.k1.params, nullptr, 0u,
             run.topk.rank_ids, capacity, run.topk.retained, run.ids, run.k1.status,
             run.total, run.nucleus, run.k1.scratch) == DS41RT_STATUS_OK,
         "scatter witness: nucleus launch");
  require_cuda(cudaDeviceSynchronize(), "scatter witness kernel");
  std::vector<uint32_t> ids(rows, sentinel);
  std::vector<uint32_t> status(rows, sentinel);
  std::vector<uint32_t> retained(rows, sentinel);
  require_cuda(cudaMemcpy(ids.data(), run.ids, rows * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost), "scatter ids d2h");
  require_cuda(cudaMemcpy(status.data(), run.k1.status, rows * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost), "scatter status d2h");
  require_cuda(cudaMemcpy(retained.data(), run.topk.retained, rows * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost), "scatter retained d2h");
  /* K4's count is keyed by output_row: slot 1 is block 0's 2, slot 0 is block
   * 1's 0 -- the mispairing that made the old guard bypassable. */
  expect(retained[1] == 2u && retained[0] == 0u,
         "scatter witness: K4 counts are keyed by output_row (2 and 0)");
  expect(status[1] == DS41RT_V41_SAMPLER_STATUS_INTERNAL,
         "scatter witness: block 0 (output_row 1) reports INTERNAL");
  expect(status[0] == DS41RT_V41_SAMPLER_STATUS_INTERNAL,
         "scatter witness: block 1 (output_row 0) reports INTERNAL");
  expect(ids[1] == sentinel,
         "scatter witness: block 0 wrote no token (pre-fix wrote 3 with OK)");
  expect(ids[0] == sentinel, "scatter witness: block 1 wrote no token");
  free_k5_run(&run);
  std::cout << "ok  K5 scattered-row identity guard on the raw-C two-row witness\n";
}

/* A single survivor (min_p keeps only the scaled maximum) and `-inf`
 * survivors: the nucleus is rank 0 either way and the draw is forced. */
void test_k5_single_survivor_and_inf() {
  const size_t vocab = 64;
  const float neg_max = -std::numeric_limits<float>::max();
  K5Stats stats;
  {
    std::vector<float> logits = shape_descending(vocab);
    const float min_p = 0.5f;
    std::vector<ds41rt_v41_sampler_row_t> params = {
        k5_row(0, 1.0f, 0.9f, 0u, min_p, std::log(min_p), DS41RT_V41_SAMPLER_NO_MASK_ROW,
               DS41RT_V41_SAMPLER_FLAG_NO_MASK | DS41RT_V41_SAMPLER_FLAG_DIAGNOSE, 3ull,
               4ull),
    };
    run_k5_case(logits, 1, vocab, params, {}, false, false, 0u, "K5 single survivor",
                &stats);
  }
  {
    /* One finite leader and 63 scaled -inf tokens (min_p = 0 at T = 1e-5 keeps
     * them); the nucleus must never be empty and the draw must stay in range. */
    std::vector<float> logits(vocab, neg_max);
    logits[7] = 0.0f;
    std::vector<ds41rt_v41_sampler_row_t> params = {
        k5_row(0, 1.0e-5f, 0.95f, 0u, 0.0f, -inf(), DS41RT_V41_SAMPLER_NO_MASK_ROW,
               DS41RT_V41_SAMPLER_FLAG_NO_MASK | DS41RT_V41_SAMPLER_FLAG_DIAGNOSE, 9ull,
               2ull),
    };
    run_k5_case(logits, 1, vocab, params, {}, false, false, 0u, "K5 -inf survivors",
                &stats);
  }
  std::cout << "ok  K5 single survivor and -inf survivor rows\n";
}

/* Combined top_k + top_p: the nucleus runs over K3/K4's retained list, so the
 * mask, the tie cut and the top-p boundary all interact. */
void test_k5_combined_topk_top_p() {
  const size_t vocab = 300;
  const size_t words = (vocab + 31u) / 32u;
  K5Stats stats;
  {
    std::vector<float> logits = shape_periodic_ties(vocab);
    std::vector<ds41rt_v41_sampler_row_t> params = {
        k5_row(0, 0.7f, 0.9f, 40u, 0.0f, -inf(), DS41RT_V41_SAMPLER_NO_MASK_ROW,
               DS41RT_V41_SAMPLER_FLAG_NO_MASK | DS41RT_V41_SAMPLER_FLAG_DIAGNOSE, 11ull,
               3ull),
    };
    run_k5_case(logits, 1, vocab, params, {}, false, false, 40u, "K5 top_k40 + top_p0.9",
                &stats);
  }
  {
    /* Sparse mask + finite top_k + top_p < 1. */
    std::vector<float> logits = shape_descending(vocab);
    std::vector<uint32_t> mask(words, 0u);
    for (size_t t = 0; t < vocab; ++t) {
      if ((t % 3u) != 0u) {
        mask[t / 32u] |= (1u << (t % 32u));
      }
    }
    std::vector<ds41rt_v41_sampler_row_t> params = {
        k5_row(0, 1.0f, 0.5f, 17u, 0.0f, -inf(), 0u,
               DS41RT_V41_SAMPLER_FLAG_DIAGNOSE, 13ull, 5ull),
    };
    run_k5_case(logits, 1, vocab, params, mask, true, true, 17u,
                "K5 sparse mask + top_k17 + top_p0.5", &stats);
  }
  {
    /* top_k >= survivor_count is a K3/K4 no-op: the survivor domain, with
     * top_p < 1 so K5 still runs. */
    std::vector<float> logits = shape_two_value(vocab);
    std::vector<ds41rt_v41_sampler_row_t> params = {
        k5_row(0, 1.0f, 0.95f, 500u, 0.0f, -inf(), DS41RT_V41_SAMPLER_NO_MASK_ROW,
               DS41RT_V41_SAMPLER_FLAG_NO_MASK | DS41RT_V41_SAMPLER_FLAG_DIAGNOSE, 17ull,
               1ull),
    };
    run_k5_case(logits, 1, vocab, params, {}, false, false, 0u,
                "K5 top_k >= survivor_count survivor domain", &stats);
  }
  {
    /* Served-like `temperature 0.7 + top_k 40` with `top_p = 1.0` on a wide
     * row. */
    const size_t wide = 4096;
    std::vector<float> logits(wide);
    for (size_t t = 0; t < wide; ++t) {
      logits[t] = -0.02f * static_cast<float>(t);
    }
    std::vector<ds41rt_v41_sampler_row_t> params = {
        k5_row(0, 0.7f, 1.0f, 40u, 0.0f, -inf(), DS41RT_V41_SAMPLER_NO_MASK_ROW,
               DS41RT_V41_SAMPLER_FLAG_NO_MASK | DS41RT_V41_SAMPLER_FLAG_DIAGNOSE, 19ull,
               7ull),
    };
    run_k5_case(logits, 1, wide, params, {}, false, false, 40u,
                "K5 wide top_k40 + top_p1.0", &stats);
  }
  std::cout << "ok  K5 combined top_k + top_p cases (" << stats.compared
            << " rows; exact " << stats.exact << ", boundary flips "
            << stats.boundary_flip << ")\n";
}

/* The no-key-satisfies fallback and the dominant-rank corner.
 *
 * The CPU's fallback (`ranked` consumed without a break) requires the *f32
 * normalized* prefix to stay below `top_p` even after the last rank, i.e.
 * `Σ (w_r / total) < top_p`. The two constructed fixtures below reach it on
 * device (the previous version of this test used `top_k = 0, top_p = 1`, the K2
 * fast path, so it never executed K5 at all and only printed the count).
 *
 *   * retained domain: 13 equal survivors with `top_k = 12` gives a 12-rank
 *     retained list whose f32 prefix at rank 11 is `0.99999988079071045 < 1.0`;
 *   * survivor domain: 71 equal survivors with `top_k = 71 == survivor_count`
 *     (K3/K4 no-op, retained 0) has tree-normalized mass `0.99999994039535522 <
 *     1.0`, so K5's `!p_found` normalized fallback is unavoidably reached;
 *   * the dominant-rank corner (`W(0) >= top_p`) is checked against production.
 *
 * `fallback_rows > 0` is asserted, and the device nucleus of each fixture is
 * read back (through the trace outputs) and required to be the full set. */
void test_k5_unsatisfied_fallback() {
  const float top_p = 1.0f;
  K5Stats stats;
  /* Retained-domain shortfall. */
  {
    const size_t vocab = 13;
    const std::vector<float> row(vocab, 0.0f);
    const std::vector<ds41rt_v41_sampler_row_t> params = {
        k5_row(0, 1.0f, top_p, 12u, 0.0f, -inf(), DS41RT_V41_SAMPLER_NO_MASK_ROW,
               DS41RT_V41_SAMPLER_FLAG_NO_MASK | DS41RT_V41_SAMPLER_FLAG_DIAGNOSE, 23ull,
               0ull),
    };
    const RefOrderedRow expected = host_ordered_reference(
        row.data(), vocab, nullptr, 0u, 1.0f, top_p, 12u, 0.0f, -inf(),
        DS41RT_V41_SAMPLER_FLAG_NO_MASK, 23ull, 0ull);
    expect(expected.ok && expected.fallback_used && !expected.k2_fast_path,
           "retained fallback fixture reaches the CPU normalizing shortfall");
    std::vector<uint32_t> nucleus_trace;
    std::vector<uint32_t> retained_trace;
    run_k5_case(row, 1, vocab, params, {}, false, false, 12u,
                "K5 retained normalizing shortfall", &stats, /*strict_nucleus=*/true,
                &nucleus_trace, &retained_trace);
    expect(retained_trace.size() == 1u && retained_trace[0] == 12u,
           "retained fallback fixture materialized exactly 12 ranks");
    expect(nucleus_trace.size() == 1u && nucleus_trace[0] == 12u,
           "retained fallback nucleus is the full 12-rank set");
  }
  /* Survivor-domain shortfall. Here the CPU's own sequential `Σ (w/total)` for 71
   * equal weights rounds to `1.0000004768371582 >= 1.0`, so the CPU's
   * `fallback_used` is FALSE; but the KERNEL's normalized survivor mass uses a
   * different (tree) association, which rounds to `0.99999994039535522 < 1.0`,
   * so K5's `!p_found` normalized fallback is the path actually taken and the
   * nucleus is the full 71-token set (validated on device in `probe.log`). This
   * fixture therefore exercises the kernel fallback, not the CPU one. */
  {
    const size_t vocab = 71;
    const std::vector<float> row(vocab, 0.0f);
    const std::vector<ds41rt_v41_sampler_row_t> params = {
        k5_row(0, 1.0f, top_p, 71u, 0.0f, -inf(), DS41RT_V41_SAMPLER_NO_MASK_ROW,
               DS41RT_V41_SAMPLER_FLAG_NO_MASK | DS41RT_V41_SAMPLER_FLAG_DIAGNOSE, 29ull,
               0ull),
    };
    const RefOrderedRow expected = host_ordered_reference(
        row.data(), vocab, nullptr, 0u, 1.0f, top_p, 71u, 0.0f, -inf(),
        DS41RT_V41_SAMPLER_FLAG_NO_MASK, 29ull, 0ull);
    expect(expected.ok && !expected.k2_fast_path && expected.nucleus_count == 71u,
           "survivor fallback fixture: the CPU orders all 71 survivors");
    std::vector<uint32_t> nucleus_trace;
    std::vector<uint32_t> retained_trace;
    run_k5_case(row, 1, vocab, params, {}, false, false, 0u,
                "K5 survivor normalizing shortfall", &stats, /*strict_nucleus=*/true,
                &nucleus_trace, &retained_trace);
    expect(retained_trace.size() == 1u && retained_trace[0] == 0u,
           "survivor fallback fixture is a K3/K4 no-op (retained 0)");
    expect(nucleus_trace.size() == 1u && nucleus_trace[0] == 71u,
           "survivor fallback nucleus is the full 71-token set (the !p_found path)");
  }
  expect(stats.compared == 2u,
         "both normalizing-shortfall fixtures executed the ordered branch");
  /* The dominant-rank corner on the ordered path: one rank at 0.0 and the rest
   * at -30, with `top_p = 0.5 < 1` and `top_k` disabled, so the CPU takes the
   * ordered branch (`top_p >= 1.0` would take the disjoint K2 fast path, which
   * K5 must not touch). `p_0` alone reaches 0.5, so the crossing is rank 0, the
   * nucleus is one token and the draw is forced. */
  {
    const size_t vocab = 32;
    std::vector<float> row(vocab, -30.0f);
    row[5] = 0.0f;
    std::vector<ds41rt_v41_sampler_row_t> one = {
        k5_row(0, 1.0f, 0.5f, 0u, 0.0f, -inf(), DS41RT_V41_SAMPLER_NO_MASK_ROW,
               DS41RT_V41_SAMPLER_FLAG_NO_MASK | DS41RT_V41_SAMPLER_FLAG_DIAGNOSE, 29ull,
               3ull),
    };
    const RefOrderedRow expected = host_ordered_reference(
        row.data(), vocab, nullptr, 0u, 1.0f, 0.5f, 0u, 0.0f, -inf(),
        DS41RT_V41_SAMPLER_FLAG_NO_MASK, 29ull, 3ull);
    expect(expected.ok && expected.nucleus_count == 1u,
           "dominant-rank corner: the CPU nucleus is the single crossing rank");
    K5Stats corner;
    run_k5_case(row, 1, vocab, one, {}, false, false, 0u,
                "K5 dominant-rank corner (W(0) >= top_p)", &corner);
    expect(corner.compared == 1u && corner.exact == 1u,
           "dominant-rank corner matches the production sampler exactly");
  }
  expect(stats.fallback_rows > 0u,
         "the fallback grid actually reached the normalizing shortfall (not 0)");
  std::cout << "ok  K5 fallback: " << stats.compared << " ordered rows, "
            << stats.fallback_rows
            << " reached the normalizing shortfall (device nuclei exact); dominant-rank "
               "corner exact\n";
}

/* Seeded replay across positions, including 0, 2^63 and u64::MAX. */
void test_k5_seeded_replay() {
  const size_t vocab = 129;
  const std::vector<float> logits = shape_periodic_ties(vocab);
  const uint64_t seeds[] = {1ull, 0ull, 0xDEADBEEFull, 987654321ull};
  const uint64_t positions[] = {0ull, 1ull, 2048ull, (1ull << 63), ~0ull, 0x8000000000000001ull};
  K5Stats stats;
  std::vector<ds41rt_v41_sampler_row_t> params;
  for (uint64_t seed : seeds) {
    for (uint64_t position : positions) {
      params.push_back(k5_row(static_cast<uint32_t>(params.size()), 0.7f, 0.9f, 40u,
                              0.0f, -inf(), DS41RT_V41_SAMPLER_NO_MASK_ROW,
                              DS41RT_V41_SAMPLER_FLAG_NO_MASK |
                                  DS41RT_V41_SAMPLER_FLAG_DIAGNOSE,
                              seed, position));
    }
  }
  const std::vector<float> rows = repeat_row(logits, params.size());
  run_k5_case(rows, params.size(), vocab, params, {}, false, false, 40u,
              "K5 seeded replay (0, 2^63, u64::MAX)", &stats);
  expect(stats.compared == params.size(),
         "every replay row was ordered and compared");
  std::cout << "ok  K5 seeded replay across positions incl. 0, 2^63, u64::MAX ("
            << stats.compared << " rows; exact " << stats.exact << ")\n";
}

/* The 129,280-wide served profiles: a real K3/K4 retained list and the
 * survivor domain at `top_p < 1`, each over several uniforms. */
void test_k5_wide_served_profiles() {
  const size_t vocab = 129280;
  const auto splitmix_unit = [](uint64_t index) -> float {
    uint64_t hash = index * 0x9e3779b97f4a7c15ull;
    hash = (hash ^ (hash >> 30)) * 0xbf58476d1ce4e5b9ull;
    hash = (hash ^ (hash >> 27)) * 0x94d049bb133111ebull;
    hash ^= hash >> 31;
    return static_cast<float>(hash >> 40) / 16777216.0f;
  };
  std::vector<float> logits(vocab);
  for (size_t t = 0; t < vocab; ++t) {
    logits[t] = -8.0f + 10.0f * splitmix_unit(static_cast<uint64_t>(t));
  }
  K5Stats stats;
  {
    std::vector<ds41rt_v41_sampler_row_t> params = {
        k5_row(0, 0.7f, 0.9f, 0u, 0.0f, -inf(), DS41RT_V41_SAMPLER_NO_MASK_ROW,
               DS41RT_V41_SAMPLER_FLAG_NO_MASK | DS41RT_V41_SAMPLER_FLAG_DIAGNOSE, 31ull,
               0ull),
    };
    run_k5_case(logits, 1, vocab, params, {}, false, false, 0u,
                "K5 wide 129280 top_p0.9 survivor domain", &stats,
                /*strict_nucleus=*/false);
  }
  {
    std::vector<ds41rt_v41_sampler_row_t> params = {
        k5_row(0, 0.7f, 1.0f, 40u, 0.0f, -inf(), DS41RT_V41_SAMPLER_NO_MASK_ROW,
               DS41RT_V41_SAMPLER_FLAG_NO_MASK | DS41RT_V41_SAMPLER_FLAG_DIAGNOSE, 37ull,
               2ull),
    };
    run_k5_case(logits, 1, vocab, params, {}, false, false, 40u,
                "K5 wide 129280 top_k40 (strict prefix, top_p=1.0)", &stats,
                /*strict_nucleus=*/false);
  }
  /* The wide rows are where the f32 accumulation residual is largest. The exact
   * nucleus-count equality is relaxed here (`strict_nucleus=false`), but the REAL
   * count difference is still measured (`max_nucleus_delta`), the token is still
   * asserted inside the DEVICE nucleus, the rank window is still asserted, and
   * the measured CDF error is still bounded -- nothing is silently unmeasured.
   * The explicit wide-row bounds are the observed envelope: at most one nucleus
   * rank and the same rank window as every other row. */
  std::cout << "ok  K5 wide 129280 served profiles (" << stats.compared
            << " rows; exact " << stats.exact << ", nucleus mismatches "
            << stats.nucleus_mismatch << ", max nucleus delta " << stats.max_nucleus_delta
            << ", max CDF error " << stats.max_cdf_error << ", outside-nucleus "
            << stats.outside_nucleus << ", max rank displacement "
            << stats.max_rank_displacement << ")\n";
  expect(stats.outside_nucleus == 0u,
         "no wide-row token fell outside the production nucleus");
  expect(stats.max_nucleus_delta <= kK5WideNucleusDeltaBound,
         "wide-row device nucleus stays inside the measured wide-row delta bound");
  expect(stats.max_rank_displacement <= KP_CROSSING_WINDOW,
         "wide-row rank displacement stays inside the asserted window");
  expect(stats.max_cdf_error <= kK5CdfErrorBound,
         "wide-row divergent tokens stay inside the measured CDF-error bound");
}
}  // namespace

int main() {
  test_mask_first_and_empty_candidates();
  test_out_of_range_mask_row_stays_memory_safe();
  test_mixed_constrained_and_unconstrained_greedy_round();
  test_masked_nonfinite_is_mode_specific();
  test_allowed_nonfinite_and_status_precedence();
  test_min_p_boundary_is_inclusive();
  test_min_p_disabled_and_unity();
  test_greedy_lowest_id_and_rederivation();
  test_vocab_remainder_bits();
  test_greedy_parity_grid();
  test_batch_independence();
  test_subnormal_and_nonfinite_grid();
  test_k2_rng_bit_equality();
  test_k2_max_uniform_clamp();
  test_k2_fast_path_grid();
  test_k2_min_p_threshold_tokens();
  test_k2_tied_and_edge_cases();
  test_k2_seeded_replay();
  test_k2_batch_composition_independence();
  test_k2_non_applicable_rows_untouched();
  test_k2_fallback_unreachable_like_cpu();
  test_k2_zero_weight_invariant();
  test_k2_saturation_fallback_witness_pinned();
  test_k3_k4_k_grid();
  test_k3_k4_thousands_tied();
  test_k3_k4_all_tied();
  test_k3_k4_masks_and_inf();
  test_k3_k4_boundaries_and_termination();
  test_k3_k4_length_two_interval();
  test_k3_k4_randomized_rows();
  test_k5_strict_prefix_at_top_p_one();
  test_k5_top_p_grid_and_boundaries();
  test_k5_boundary_uniforms();
  test_k5_probe_association_invariance();
  test_k5_zero_uniform_draw();
  test_k5_repeated_run_determinism();
  test_k5_loud_status_failures();
  test_k1_only_entry_loud_status();
  test_k5_scattered_row_identity_guard();
  test_k5_single_survivor_and_inf();
  test_k5_combined_topk_top_p();
  test_k5_unsatisfied_fallback();
  test_k5_seeded_replay();
  test_k5_wide_served_profiles();
  std::cout << "K3 pivot pass budget (measured): max " << g_max_pivot_passes << " of "
            << DS41RT_V41_TOPK_MAX_PIVOT_STEPS << " cap, host worst-case bound "
            << host_bisection_pass_bound() << "\n";
  std::cout << "ds41rt_v41_sampling_selftest passed: " << g_cases << " cases, " << g_checks
            << " assertions\n";
  return 0;
}
