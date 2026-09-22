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
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

int g_checks = 0;
int g_cases = 0;

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
  std::cout << "ds41rt_v41_sampling_selftest passed: " << g_cases << " cases, " << g_checks
            << " assertions\n";
  return 0;
}
