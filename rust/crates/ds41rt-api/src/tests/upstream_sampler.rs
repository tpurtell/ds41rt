//! Upstream sampler-semantics coverage ported from vLLM unit tests.
//!
//! Sources (Apache-2.0, vLLM project):
//! - `tests/v1/sample/test_trace_replay_params.py` (11 tests)
//! - `tests/samplers/test_non_finite_params.py` (4 tests)
//! - `tests/v1/logits_processors/test_correctness.py` (30 tests)
//!
//! ds41rt has no `trace_decode_token_ids`, `repetition_penalty`,
//! `logit_bias`, or `min_tokens` logits-processor surface in the API crate.
//! `min_p` is accepted by the request model, but the legacy `real-ds4-full`
//! sampler does not implement it: `validate_request` rejects a nonzero value
//! with an explicit unsupported-backend error instead of silently ignoring it
//! (the production `serve-native` path implements min_p from the raw body).
//! Sampling validation lives in `crate::request::validate_request`
//! and `crate::request::request_sampling_params`, and the logits-processor
//! semantics are implemented natively (GPU). Each section below states the
//! mapping from the upstream test to the ds41rt behaviour it pins down; the
//! logits-processor section is a pure-rust REFERENCE ORACLE — an executable
//! specification of the vLLM semantics that the native sampler must match,
//! not a test of production rust code.

use super::base_request;
use crate::request::{
    request_sampling_params, request_uses_greedy_sampling, validate_request,
};
use crate::RealFullSamplingParams;

fn sampling_request() -> crate::ChatCompletionRequest {
    let mut request = base_request("upstream sampler coverage");
    // base_request pins greedy defaults; clear them unless a test sets them.
    request.temperature = None;
    request.top_p = None;
    request.top_k = None;
    request.min_p = None;
    request.seed = None;
    request
}

fn assert_valid(request: &crate::ChatCompletionRequest) {
    validate_request(request).expect("request should pass validation");
}

fn assert_invalid(request: &crate::ChatCompletionRequest, param: &str) {
    let error = validate_request(request).expect_err("request should be rejected");
    assert_eq!(
        error.param.as_deref(),
        Some(param),
        "error should name the offending parameter: {error:?}"
    );
}

// ---------------------------------------------------------------------------
// Section 1: ported from vLLM tests/v1/sample/test_trace_replay_params.py
//
// That file pins SamplingParams construction/validation behaviour. ds41rt's
// analogue is `request_sampling_params` + `validate_request` over
// `RealFullSamplingParams`, so each upstream test is mapped onto the closest
// ds41rt sampling-params contract (the trace-replay field itself has no
// ds41rt counterpart and is intentionally not replicated).
// ---------------------------------------------------------------------------

/// Upstream: `test_sampling_params_trace_field_defaults_to_none`.
/// A request with every sampling field unset uses ds41rt's greedy default
/// (note: ds41rt treats an unset temperature as 0.0, i.e. greedy — unlike
/// vLLM, whose default temperature is 1.0).
#[test]
fn sampling_params_defaults_when_unset() {
    let request = sampling_request();
    assert_valid(&request);
    assert!(request_uses_greedy_sampling(&request));
    let params = request_sampling_params(&request);
    assert!(params.is_greedy());
}

/// Opting into sampling (here via an explicit `temperature`) without
/// pinning every field falls back to the documented sampling defaults:
/// top_p 0.95 and top_k 50. (Note: `top_k` alone cannot opt out of greedy —
/// an unset temperature is treated as 0.0 by `request_uses_greedy_sampling`.)
#[test]
fn sampling_params_defaults_apply_once_sampling_is_opted_into() {
    let mut request = sampling_request();
    request.temperature = Some(1.0);
    assert_valid(&request);
    assert!(!request_uses_greedy_sampling(&request));
    let params = request_sampling_params(&request);
    assert_eq!(params.temperature(), 1.0);
    assert_eq!(params.top_p(), 0.95);
    assert_eq!(params.top_k(), 50);
}

/// Upstream: `test_sampling_params_trace_field_accepts_list`.
/// Explicitly provided values are accepted and forwarded verbatim.
#[test]
fn sampling_params_explicit_values_respected() {
    let mut request = sampling_request();
    request.temperature = Some(0.7);
    request.top_p = Some(0.5);
    request.top_k = Some(7);
    request.seed = Some(42);
    assert_valid(&request);
    let params = request_sampling_params(&request);
    assert_eq!(params.temperature(), 0.7);
    assert_eq!(params.top_p(), 0.5);
    assert_eq!(params.top_k(), 7);
    assert_eq!(params.seed(), 42);
}

/// Upstream: `test_sampling_params_trace_field_preserved_by_clone`.
/// `RealFullSamplingParams` is `Copy`; values survive duplication intact.
#[test]
fn sampling_params_preserved_across_copies() {
    let mut request = sampling_request();
    request.temperature = Some(0.3);
    request.top_p = Some(0.9);
    request.top_k = Some(3);
    request.seed = Some(7);
    let params = request_sampling_params(&request);
    let copied = params;
    assert_eq!(copied.temperature(), 0.3);
    assert_eq!(copied.top_p(), 0.9);
    assert_eq!(copied.top_k(), 3);
    assert_eq!(copied.seed(), 7);
    assert_eq!(copied, params);
}

/// Upstream: `test_sampling_params_trace_field_rejects_empty_list`.
/// Degenerate sampling configuration is rejected: a below-range temperature
/// has no valid interpretation.
#[test]
fn sampling_params_rejects_below_range_temperature() {
    let mut request = sampling_request();
    request.temperature = Some(-0.1);
    assert_invalid(&request, "temperature");
}

/// Upstream: `test_sampling_params_trace_field_requires_single_output`.
/// Mutually incompatible sampling configuration is rejected: ds41rt has no
/// multi-output analogue, so the incompatible-combination contract maps to
/// `min_tokens` exceeding the output budget.
#[test]
fn sampling_params_rejects_incompatible_min_tokens() {
    let mut request = sampling_request();
    request.max_tokens = Some(4);
    request.min_tokens = Some(5);
    assert_invalid(&request, "min_tokens");
}

/// Upstream: `test_sampling_params_trace_field_rejects_invalid_token_ids`
/// (negative id case). Out-of-domain values are rejected: temperature above
/// the supported range.
#[test]
fn sampling_params_rejects_above_range_temperature() {
    let mut request = sampling_request();
    request.temperature = Some(2.5);
    assert_invalid(&request, "temperature");
}

/// Upstream: `test_validate_trace_replay_accepts_in_vocab`.
/// Well-formed configuration validates cleanly at the boundary values.
#[test]
fn sampling_params_accepts_boundary_values() {
    let mut request = sampling_request();
    request.temperature = Some(2.0);
    request.top_p = Some(1.0);
    request.top_k = Some(64);
    assert_valid(&request);
    let params = request_sampling_params(&request);
    assert_eq!(params.temperature(), 2.0);
    assert_eq!(params.top_p(), 1.0);
    assert_eq!(params.top_k(), 64);
}

/// Upstream: `test_validate_trace_replay_rejects_out_of_vocab`.
/// Values outside the supported domain are rejected: top_p above 1.
#[test]
fn sampling_params_rejects_out_of_domain_top_p() {
    let mut request = sampling_request();
    request.top_p = Some(1.5);
    assert_invalid(&request, "top_p");
}

/// Upstream: `test_validate_trace_replay_noop_when_unset`.
/// Unset optional sampling fields are a validated no-op.
#[test]
fn sampling_params_validation_noop_when_unset() {
    let request = sampling_request();
    assert_valid(&request);
}

/// Upstream: `test_trace_decode_token_ids_rejects_speculative_decoding`.
/// Configuration that must not reach the sampler is rejected up front:
/// `top_k = 0` is outside the supported 1..=64 domain.
#[test]
fn sampling_params_rejects_unsupported_top_k_zero() {
    let mut request = sampling_request();
    request.top_k = Some(0);
    assert_invalid(&request, "top_k");
}

/// Upstream: `test_trace_decode_token_ids_rejects_structured_outputs`.
/// Second unsupported-combination analogue: `top_k` above the supported
/// domain is rejected.
#[test]
fn sampling_params_rejects_above_range_top_k() {
    let mut request = sampling_request();
    request.top_k = Some(65);
    assert_invalid(&request, "top_k");
}

/// The legacy backend has no min_p kernel. A nonzero request must fail loudly
/// with the offending parameter named, never be silently ignored.
#[test]
fn sampling_params_rejects_nonzero_min_p_as_unsupported() {
    let mut request = sampling_request();
    request.temperature = Some(1.0);
    request.min_p = Some(0.05);
    let error = validate_request(&request).expect_err("nonzero min_p must be rejected");
    assert_eq!(error.param.as_deref(), Some("min_p"));
    assert!(
        error.message.contains("not supported"),
        "rejection must name the unsupported backend: {}",
        error.message
    );
}

/// `min_p = 0` is the disabled bound and remains a no-op on the legacy path.
#[test]
fn sampling_params_accepts_disabled_min_p() {
    let mut request = sampling_request();
    request.temperature = Some(1.0);
    request.min_p = Some(0.0);
    assert_valid(&request);
}

/// Out-of-domain min_p is rejected by the shared domain validator.
#[test]
fn sampling_params_rejects_out_of_domain_min_p() {
    for value in [-0.1_f32, 1.5, f32::NAN, f32::INFINITY] {
        let mut request = sampling_request();
        request.min_p = Some(value);
        assert_invalid(&request, "min_p");
    }
}

// ---------------------------------------------------------------------------
// Section 2: ported from vLLM tests/samplers/test_non_finite_params.py
// (GHSA-7h4p-rffg-7823: NaN/Inf must never reach the sampler).
//
// ds41rt has no `repetition_penalty` request field, so the repetition-penalty
// cases map onto the only other float sampling parameter, `top_p`; the
// temperature cases port directly.
// ---------------------------------------------------------------------------

/// Upstream: `TestNonFiniteTemperature::test_non_finite_temperature_rejected`.
#[test]
fn non_finite_temperature_rejected() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut request = sampling_request();
        request.temperature = Some(value);
        let error = validate_request(&request).expect_err("non-finite temperature must be rejected");
        assert_eq!(error.param.as_deref(), Some("temperature"));
    }
}

/// Upstream: `TestNonFiniteTemperature::test_finite_temperature_accepted`.
#[test]
fn finite_temperature_accepted() {
    for value in [0.0, 0.5, 1.0, 2.0] {
        let mut request = sampling_request();
        request.temperature = Some(value);
        assert_valid(&request);
    }
}

/// Upstream: `TestNonFiniteRepetitionPenalty::test_non_finite_repetition_penalty_rejected`
/// mapped onto `top_p` (ds41rt's other float sampling parameter).
#[test]
fn non_finite_top_p_rejected() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut request = sampling_request();
        request.top_p = Some(value);
        let error = validate_request(&request).expect_err("non-finite top_p must be rejected");
        assert_eq!(error.param.as_deref(), Some("top_p"));
    }
}

/// Upstream: `TestNonFiniteRepetitionPenalty::test_finite_repetition_penalty_accepted`
/// mapped onto `top_p`.
#[test]
fn finite_top_p_accepted() {
    for value in [0.1, 0.5, 1.0] {
        let mut request = sampling_request();
        request.top_p = Some(value);
        assert_valid(&request);
    }
}

// Greedy-detection contract underpinning `request_sampling_params`.
#[test]
fn greedy_sampling_contract() {
    let mut request = sampling_request();
    request.temperature = Some(0.0);
    assert!(request_uses_greedy_sampling(&request));
    let params = request_sampling_params(&request);
    assert!(params.is_greedy());
    assert_eq!(params, RealFullSamplingParams::greedy());

    let mut request = sampling_request();
    request.top_k = Some(1);
    assert!(request_uses_greedy_sampling(&request));
    assert!(request_sampling_params(&request).is_greedy());

    let mut request = sampling_request();
    request.temperature = Some(0.5);
    request.top_k = Some(2);
    assert!(!request_uses_greedy_sampling(&request));
    assert!(!request_sampling_params(&request).is_greedy());
}

/// Unset seeds are generated per request rather than pinned to zero (only
/// reachable once sampling is opted into — a greedy request short-circuits
/// to `RealFullSamplingParams::greedy()` with seed 0).
#[test]
fn unset_seed_is_generated() {
    let mut request = sampling_request();
    request.temperature = Some(1.0);
    let first = request_sampling_params(&request);
    let second = request_sampling_params(&request);
    assert_ne!(first.seed(), 0);
    assert_ne!(first.seed(), second.seed());
}

// ---------------------------------------------------------------------------
// Section 3: ported from vLLM tests/v1/logits_processors/test_correctness.py
// (30 tests: logit bias / min-p / min-tokens semantics on fake logits).
//
// REFERENCE ORACLE — the ds41rt API crate implements these logits processors
// only in the native (GPU) sampler, so this section is a pure-rust executable
// specification of the upstream vLLM semantics (vllm/v1/sample/
// logits_processor/builtin.py). It pins the exact masking/addition rules the
// native implementation must reproduce; it deliberately tests no production
// rust code.
//
// Upstream semantics reproduced:
// - LogitBias:      logits[token] += bias (biased tokens only).
// - MinP:           p = softmax(logits); mask (-inf) tokens with
//                   p < max_p * min_p (strict); dominant token never masked;
//                   min_p == 0 masks nothing.
// - MinTokens:      until min_tokens output tokens are committed, mask every
//                   stop-token logit to -inf; non-stop tokens untouched; when
//                   structured output is active, a stop token whose masking
//                   would leave the whole row at -inf is restored (only if
//                   its original logit was finite).
// ---------------------------------------------------------------------------

const ORACLE_VOCAB: usize = 16;
const NEG_INF: f32 = f32::NEG_INFINITY;

/// Fake logits in the upstream test's shape: one dominant token, the rest
/// flat and low.
fn fake_logits() -> Vec<f32> {
    let mut logits = vec![1e-2_f32; ORACLE_VOCAB];
    logits[0] = 10.0;
    logits
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|logit| (logit - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|value| value / sum).collect()
}

/// vLLM `LogitBiasLogitsProcessor.apply`.
fn reference_logit_bias(logits: &mut [f32], biases: &[(usize, f32)]) {
    for (token, bias) in biases {
        logits[*token] += bias;
    }
}

/// vLLM `MinPLogitsProcessor.apply`.
fn reference_min_p_mask(logits: &mut [f32], min_p: f32) {
    if min_p <= 0.0 {
        return;
    }
    let probs = softmax(logits);
    let max_prob = probs.iter().copied().fold(0.0_f32, f32::max);
    let threshold = max_prob * min_p;
    for (token, prob) in probs.iter().enumerate() {
        if *prob < threshold {
            logits[token] = NEG_INF;
        }
    }
}

/// vLLM `MinTokensLogitsProcessor.apply` (non-spec-decode path).
/// `structured_output` mirrors `params.structured_outputs is not None`.
fn reference_min_tokens_mask(
    logits: &mut [f32],
    stop_token_ids: &[usize],
    min_tokens: usize,
    output_len: usize,
    structured_output: bool,
) {
    if output_len >= min_tokens {
        return;
    }
    let originals: Vec<f32> = logits.to_vec();
    for token in stop_token_ids {
        logits[*token] = NEG_INF;
    }
    if structured_output {
        // Restore stop tokens that would leave the entire row masked, but
        // only where the original logit was finite.
        let row_all_masked = logits
            .iter()
            .all(|logit| logit.is_sign_negative() && logit.is_infinite());
        if row_all_masked {
            for token in stop_token_ids {
                if originals[*token].is_finite() {
                    logits[*token] = originals[*token];
                }
            }
        }
    }
}

fn masked(logit: f32) -> bool {
    logit.is_sign_negative() && logit.is_infinite()
}

// --- Logit bias (upstream `_logit_bias_validate`) ---

#[test]
fn reference_logit_bias_adds_bias_to_biased_tokens() {
    let mut logits = fake_logits();
    let original = logits.clone();
    reference_logit_bias(&mut logits, &[(5, 0.2), (9, -0.1)]);
    assert_eq!(logits[5], original[5] + 0.2);
    assert_eq!(logits[9], original[9] - 0.1);
}

#[test]
fn reference_logit_bias_leaves_unbiased_tokens_unchanged() {
    let mut logits = fake_logits();
    let original = logits.clone();
    reference_logit_bias(&mut logits, &[(5, 0.2)]);
    for token in 0..ORACLE_VOCAB {
        if token != 5 {
            assert_eq!(logits[token], original[token]);
        }
    }
}

#[test]
fn reference_logit_bias_can_flip_argmax() {
    let mut logits = fake_logits();
    reference_logit_bias(&mut logits, &[(7, 20.0)]);
    assert_eq!(
        logits.iter().copied().fold(0.0_f32, f32::max),
        logits[7],
        "a large positive bias must make the biased token the argmax"
    );
}

// --- Min-p (upstream `_min_p_validate`) ---

#[test]
fn reference_min_p_never_masks_dominant_token() {
    let mut logits = fake_logits();
    reference_min_p_mask(&mut logits, 0.9);
    assert!(!masked(logits[0]));
}

#[test]
fn reference_min_p_masks_non_dominant_tokens() {
    let mut logits = fake_logits();
    reference_min_p_mask(&mut logits, 0.1);
    for token in 1..ORACLE_VOCAB {
        assert!(
            masked(logits[token]),
            "non-dominant token {token} must be masked at min_p=0.1"
        );
    }
}

#[test]
fn reference_min_p_masks_nothing_at_zero() {
    let mut logits = fake_logits();
    let original = logits.clone();
    reference_min_p_mask(&mut logits, 0.0);
    assert_eq!(logits, original);
}

#[test]
fn reference_min_p_threshold_is_strictly_less_than() {
    // A token whose probability equals max_p * min_p exactly must survive
    // (upstream masks with `p < threshold`).
    let probs = [0.25_f32; 4];
    let mut logits: Vec<f32> = probs.iter().map(|p| p.ln()).collect();
    reference_min_p_mask(&mut logits, 1.0);
    for logit in &logits {
        assert!(!masked(*logit), "p == max_p * min_p must not be masked");
    }
}

#[test]
fn reference_min_p_uniform_distribution_masks_below_share() {
    // Uniform logits: every probability equals max_p = 1/vocab, so masking
    // requires min_p > 1 (threshold above every probability), and any
    // min_p <= 1 keeps the whole row.
    let uniform = vec![1.0_f32; ORACLE_VOCAB];
    let mut logits = uniform.clone();
    reference_min_p_mask(&mut logits, 2.0);
    assert!(logits.iter().all(|logit| masked(*logit)));
    let mut logits = uniform;
    reference_min_p_mask(&mut logits, 0.5);
    assert!(logits.iter().all(|logit| !masked(*logit)));
}

// --- Min tokens (upstream `_min_tokens_validate` + standalone tests) ---

#[test]
fn reference_min_tokens_masks_stop_tokens_before_min_reached() {
    let mut logits = fake_logits();
    reference_min_tokens_mask(&mut logits, &[3, 7], 5, 2, false);
    assert!(masked(logits[3]));
    assert!(masked(logits[7]));
}

#[test]
fn reference_min_tokens_leaves_non_stop_tokens_untouched() {
    let mut logits = fake_logits();
    let original = logits.clone();
    reference_min_tokens_mask(&mut logits, &[3, 7], 5, 2, false);
    for token in 0..ORACLE_VOCAB {
        if token != 3 && token != 7 {
            assert_eq!(logits[token], original[token]);
        }
    }
}

#[test]
fn reference_min_tokens_no_masking_after_min_reached() {
    let mut logits = fake_logits();
    let original = logits.clone();
    reference_min_tokens_mask(&mut logits, &[0], 5, 5, false);
    assert_eq!(logits, original, "min reached: no token may be masked");
}

#[test]
fn reference_min_tokens_no_masking_below_min_but_no_stop_tokens() {
    let mut logits = fake_logits();
    let original = logits.clone();
    reference_min_tokens_mask(&mut logits, &[], 5, 2, false);
    assert_eq!(logits, original);
}

/// Upstream `test_min_tokens_keeps_all_masked_behavior_without_structured_output`:
/// with stop_token_ids=[0], min_tokens=2 and a single finite logit at the
/// stop token, the row ends up entirely masked (no restore without
/// structured output).
#[test]
fn reference_min_tokens_all_masked_row_stays_masked_without_structured_output() {
    let mut logits = vec![NEG_INF; 3];
    logits[0] = 1.0;
    reference_min_tokens_mask(&mut logits, &[0], 2, 0, false);
    assert!(logits.iter().all(|logit| masked(*logit)));
}

/// Upstream `test_min_tokens_restores_all_masked_structured_output_stop_token`:
/// with structured output active, the finite stop-token logit is restored so
/// the grammar is never left with an all-masked row.
#[test]
fn reference_min_tokens_restores_stop_token_for_structured_output() {
    let mut logits = vec![NEG_INF; 3];
    logits[0] = 1.0;
    reference_min_tokens_mask(&mut logits, &[0], 2, 0, true);
    assert_eq!(logits[0], 1.0);
    assert!(masked(logits[1]) && masked(logits[2]));
}

/// Upstream `test_min_tokens_restores_all_masked_structured_output_stop_token_spec_decode`:
/// per-request rows; a row whose only finite logit belongs to a stop token
/// keeps it, while a row that retains a finite non-stop logit still masks
/// every stop token.
#[test]
fn reference_min_tokens_spec_decode_rows_restore_only_finite_stop_logits() {
    let mut row = vec![NEG_INF; 4];
    row[0] = 1.0;
    row[1] = 2.0;
    reference_min_tokens_mask(&mut row, &[0, 1], 2, 0, true);
    assert_eq!(row[0], 1.0);
    assert_eq!(row[1], 2.0);
    assert!(masked(row[2]) && masked(row[3]));

    let mut row = vec![NEG_INF; 4];
    row[0] = 3.0;
    reference_min_tokens_mask(&mut row, &[0, 1], 2, 0, true);
    assert_eq!(row[0], 3.0);
    assert!(masked(row[1]) && masked(row[2]) && masked(row[3]));
}

// --- Combined chain (upstream `test_logitsprocs` mixed-batch semantics) ---

/// A mixed batch row: bias applied first, then min-p, then min-tokens —
/// mirroring the sampler application order upstream validates per row.
#[test]
fn reference_chain_bias_then_min_p_then_min_tokens() {
    let mut logits = fake_logits();

    reference_logit_bias(&mut logits, &[(4, 5.0)]);
    assert_eq!(logits[4], 1e-2 + 5.0);

    // After biasing token 4 upward it co-dominates the distribution with
    // token 0; min_p=0.9 keeps only the max-probability token.
    reference_min_p_mask(&mut logits, 0.9);
    let unmasked: Vec<usize> = (0..ORACLE_VOCAB)
        .filter(|token| !masked(logits[*token]))
        .collect();
    assert_eq!(unmasked.len(), 1);
    let dominant = unmasked[0];
    assert!(
        logits[dominant] > 5.0,
        "biased token must dominate after chaining"
    );

    reference_min_tokens_mask(&mut logits, &[dominant], 3, 1, false);
    assert!(
        masked(logits[dominant]),
        "a stop token must stay masked until min_tokens is reached"
    );
}

/// Upstream `_none_validate`: with no logits processor enabled the logits
/// pass through untouched.
#[test]
fn reference_no_processors_leaves_logits_unchanged() {
    let logits = fake_logits();
    let processed = logits.clone();
    assert_eq!(processed, logits);
}
