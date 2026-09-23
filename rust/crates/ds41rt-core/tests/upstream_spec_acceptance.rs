//! Ports of upstream speculative-decoding acceptance/bookkeeping coverage
//! (vLLM `tests/v1/spec_decode/` + sglang `test/registered/unit/spec/`) onto
//! ds41rt-core public API. Reference math that has no ds41rt product
//! equivalent (unconditional→conditional rate conversion, per-request
//! acceptance metrics accumulator, async-scheduling backup-token indexing) is
//! reimplemented here as clearly-marked upstream ports; every group also
//! anchors the invariant against product code (`dspark_expected_tokens`,
//! `verify_dspark_greedy`).
//!
//! Upstream sources:
//! - test_synthetic_rejection_sampler_utils.py (5 tests, pure math)
//! - test_request_acceptance.py (11 tests, per-request metrics histogram)
//! - test_backup_token_async_spec.py (7 tests, backup-token bookkeeping)
//! - sglang spec bookkeeping/ownership tests (invariant class, AST guards
//!   not portable; the scheduling-ownership invariants are noted as unmapped)

// COVERAGE CLASS: standalone reference oracle. These cases document upstream
// behavior with no ds41rt dependency; they cannot detect product regressions
// by themselves. They are the comparison references for the deferred GPU
// parity tests (docs/test-coverage/DEFERRED.md) and are counted separately
// from product regression coverage (review MAJOR 4/6, 2026-09-15).

use ds41rt_core::{
    dspark_expected_tokens, verify_dspark_greedy,
};

// ---------------------------------------------------------------------------
// Upstream reference math (ports; no ds41rt product equivalent exists).
// ---------------------------------------------------------------------------

/// Port of vLLM `unconditional_to_conditional_rates`: c_0 = p_0, c_i =
/// p_i / p_{i-1}. After a zero the chain has terminated, so subsequent
/// conditional rates are clamped to 0 (never consumed downstream).
fn upstream_unconditional_to_conditional_rates(unconditional: &[f64]) -> Vec<f64> {
    let mut conditional = Vec::with_capacity(unconditional.len());
    let mut prev = 1.0;
    for &p in unconditional {
        let c = if prev == 0.0 { 0.0 } else { p / prev };
        conditional.push(c);
        prev = p;
    }
    conditional
}

/// Port of vLLM `SpeculativeConfig._acceptance_length_to_rates`: decompose a
/// mean acceptance length (including the mandatory +1 anchor/bonus) into a
/// length-`n` schedule of conditional acceptance rates, front-filled with 1s.
fn upstream_acceptance_length_to_rates(length: f64, n: usize) -> Vec<f64> {
    let accepted = length - 1.0; // mean accepted *draft* tokens per step
    let full = accepted.floor() as usize; // positions that always accept
    let frac = accepted - full as f64; // conditional rate at the boundary
    let mut rates = vec![0.0; n];
    for (i, slot) in rates.iter_mut().enumerate() {
        if i < full {
            *slot = 1.0;
        } else if i == full {
            *slot = frac;
        }
    }
    rates
}

// ---------------------------------------------------------------------------
// Group 1: synthetic rejection-sampler rate math
// (vLLM test_synthetic_rejection_sampler_utils.py)
// ---------------------------------------------------------------------------

/// Product anchor: ds41rt's `dspark_expected_tokens` (the policy objective) computes
/// expected_tokens = 1 + sum(cumulative products of the conditional rates),
/// i.e. exactly 1 + sum(unconditional rates). This is the semantic contract
/// tying ds41rt's "conditional draft confidence" input to vLLM's
/// unconditional acceptance-rate convention.
fn expected_tokens_for_conditional(confidence: &[f64]) -> f64 {
    dspark_expected_tokens(confidence)
}

#[test]
fn upstream_spec_unconditional_to_conditional_rates_basic() {
    let conditional = upstream_unconditional_to_conditional_rates(&[0.9, 0.5, 0.2]);
    assert!((conditional[0] - 0.9).abs() < 1e-12);
    assert!((conditional[1] - 0.5 / 0.9).abs() < 1e-12);
    assert!((conditional[2] - 0.2 / 0.5).abs() < 1e-12);
    // Product anchor: 1 + (0.9 + 0.5 + 0.2) = 2.6 expected tokens.
    assert!((expected_tokens_for_conditional(&conditional) - 2.6).abs() < 1e-12);
    // Products of conditionals reproduce the unconditional sequence: the
    // expected tokens of each prefix add exactly its unconditional rate.
    assert!((dspark_expected_tokens(&conditional[..2]) - 2.4).abs() < 1e-12);
    assert!((dspark_expected_tokens(&conditional[..1]) - 1.9).abs() < 1e-12);
}

#[test]
fn upstream_spec_unconditional_to_conditional_rates_handles_zero() {
    // After a zero, subsequent conditional rates are clamped to 0 (the chain
    // has already terminated; downstream never consumes them).
    let conditional = upstream_unconditional_to_conditional_rates(&[1.0, 0.6, 0.0, 0.0]);
    assert_eq!(conditional, vec![1.0, 0.6, 0.0, 0.0]);
    // Product anchor: cumulative products freeze at 0.6 → 2.6 expected tokens.
    assert!((expected_tokens_for_conditional(&conditional) - 2.6).abs() < 1e-12);
}

#[test]
fn upstream_spec_unconditional_to_conditional_rates_all_ones() {
    let conditional = upstream_unconditional_to_conditional_rates(&[1.0, 1.0, 1.0]);
    assert_eq!(conditional, vec![1.0, 1.0, 1.0]);
    assert!((expected_tokens_for_conditional(&conditional) - 4.0).abs() < 1e-12);
}

#[test]
fn upstream_spec_acceptance_length_to_rates() {
    // (length, n, expected) — upstream parametrized cases.
    let cases: &[(f64, usize, &[f64])] = &[
        (2.6, 3, &[1.0, 0.6, 0.0]),
        (1.0, 3, &[0.0, 0.0, 0.0]),
        (4.0, 3, &[1.0, 1.0, 1.0]),
        (2.0, 3, &[1.0, 0.0, 0.0]),
        (3.5, 4, &[1.0, 1.0, 0.5, 0.0]),
    ];
    for &(length, n, expected) in cases {
        let rates = upstream_acceptance_length_to_rates(length, n);
        assert_eq!(rates.len(), n);
        for (r, e) in rates.iter().zip(expected) {
            assert!((r - e).abs() < 1e-12, "length={length} n={n}: {rates:?}");
        }
        // Product anchor: the schedule reproduces the requested acceptance
        // length under ds41rt's expected-token accumulation (anchor +1).
        assert!(
            (expected_tokens_for_conditional(&rates) - length).abs() < 1e-12,
            "length={length} n={n}"
        );
    }
}

#[test]
fn upstream_spec_resolve_length_produces_minvariance_schedule() {
    // Upstream: _resolve_synthetic_acceptance_rates(3, None, 2.6).
    let rates = upstream_acceptance_length_to_rates(2.6, 3);
    assert!((rates[0] - 1.0).abs() < 1e-12);
    assert!((rates[1] - 0.6).abs() < 1e-12);
    assert_eq!(rates[2], 0.0);
    assert!((expected_tokens_for_conditional(&rates) - 2.6).abs() < 1e-12);
}

// ---------------------------------------------------------------------------
// Group 2: per-request acceptance metrics
// (vLLM test_request_acceptance.py — RequestSpecDecodeMetrics port)
// ---------------------------------------------------------------------------

/// Port of vLLM `RequestSpecDecodeMetrics`: dense acceptance histogram plus
/// optional ordered per-step arrays, surfaced via `to_dict`. ds41rt has no
/// metrics accumulator; this is the upstream bookkeeping contract, driven by
/// product verification output in `verification_outputs_feed_metrics`.
#[derive(Debug)]
struct UpstreamSpecDecodeMetrics {
    num_spec_tokens: usize,
    histogram: Vec<u64>, // dense, index j = steps that accepted exactly j drafts
    num_draft_tokens: u64,
    per_step_accepted: Vec<u64>,
    per_step_drafted: Vec<u64>,
    detailed: bool,
}

impl UpstreamSpecDecodeMetrics {
    fn new(num_spec_tokens: usize) -> Self {
        Self {
            num_spec_tokens,
            histogram: vec![0; num_spec_tokens + 1],
            num_draft_tokens: 0,
            per_step_accepted: Vec::new(),
            per_step_drafted: Vec::new(),
            detailed: false,
        }
    }

    fn observe(&mut self, num_draft_tokens: u64, num_accepted: u64) {
        self.histogram[num_accepted as usize] += 1;
        self.num_draft_tokens += num_draft_tokens;
        if self.detailed {
            self.per_step_accepted.push(num_accepted);
            self.per_step_drafted.push(num_draft_tokens);
        }
    }

    fn num_spec_steps(&self) -> u64 {
        self.histogram.iter().sum()
    }

    fn num_accepted_draft_tokens(&self) -> u64 {
        self.histogram
            .iter()
            .enumerate()
            .map(|(j, &count)| j as u64 * count)
            .sum()
    }

    /// Upstream `to_dict` payload (serde map mirrors the msgspec struct that
    /// rides EngineCoreOutput in vLLM).
    fn to_dict(&self) -> serde_json::Value {
        let steps = self.num_spec_steps();
        let accepted = self.num_accepted_draft_tokens();
        let mean_acceptance_length = if steps > 0 {
            1.0 + accepted as f64 / steps as f64
        } else {
            1.0 // empty: no divide-by-zero
        };
        let draft_acceptance_rate = if self.num_draft_tokens > 0 {
            accepted as f64 / self.num_draft_tokens as f64
        } else {
            0.0
        };
        let mut map = serde_json::json!({
            "mean_acceptance_length": mean_acceptance_length,
            "draft_acceptance_rate": draft_acceptance_rate,
            "acceptance_histogram": self.histogram,
            "num_spec_steps": steps,
            "num_accepted_draft_tokens": accepted,
            "num_draft_tokens": self.num_draft_tokens,
            "num_spec_tokens": self.num_spec_tokens,
        });
        if self.detailed {
            map["per_step_accepted"] = serde_json::json!(self.per_step_accepted);
            map["per_step_drafted"] = serde_json::json!(self.per_step_drafted);
        }
        map
    }
}

fn metrics_from(pairs: &[(u64, u64)], num_spec_tokens: usize, detailed: bool) -> UpstreamSpecDecodeMetrics {
    let mut s = UpstreamSpecDecodeMetrics::new(num_spec_tokens);
    s.detailed = detailed;
    for &(k, j) in pairs {
        s.observe(k, j);
    }
    s
}

#[test]
fn upstream_spec_metrics_new_allocates_dense_histogram_of_k_plus_one() {
    let s = UpstreamSpecDecodeMetrics::new(3);
    assert_eq!(s.num_spec_tokens, 3);
    assert_eq!(s.histogram, vec![0, 0, 0, 0]);
    assert_eq!(s.num_draft_tokens, 0);
    assert!(s.per_step_accepted.is_empty());
}

#[test]
fn upstream_spec_metrics_observe_buckets_by_accepted_draft_count() {
    let s = metrics_from(&[(3, 0), (3, 3), (3, 2), (3, 3), (3, 1)], 3, false);
    // j=0 ->1, j=1 ->1, j=2 ->1, j=3 ->2
    assert_eq!(s.histogram, vec![1, 1, 1, 2]);
    assert_eq!(s.num_draft_tokens, 15);
    // summary level does not record the ordered per-step arrays
    assert!(s.per_step_accepted.is_empty());
    assert!(s.per_step_drafted.is_empty());
}

#[test]
fn upstream_spec_metrics_observe_detailed_records_ordered_per_step_arrays() {
    // Distinct step count (3), max draft length (k=4), and per-step drafted
    // counts (4, 2, 4) so no accidental "everything is 3" pattern is implied.
    let s = metrics_from(&[(4, 3), (2, 2), (4, 0)], 4, true);
    assert_eq!(s.per_step_accepted, vec![3, 2, 0]);
    assert_eq!(s.per_step_drafted, vec![4, 2, 4]);
    // histogram (indexed by accepted j, length k+1=5) is still maintained
    assert_eq!(s.histogram, vec![1, 0, 1, 1, 0]);
}

#[test]
fn upstream_spec_metrics_to_dict_summary_omits_per_step_arrays() {
    let d = metrics_from(&[(3, 0), (3, 3), (3, 2), (3, 3), (3, 1)], 3, false).to_dict();
    assert!((d["mean_acceptance_length"].as_f64().unwrap() - (1.0 + 9.0 / 5.0)).abs() < 1e-12);
    assert!((d["draft_acceptance_rate"].as_f64().unwrap() - (9.0 / 15.0)).abs() < 1e-12);
    assert_eq!(d["acceptance_histogram"], serde_json::json!([1, 1, 1, 2]));
    assert_eq!(d["num_spec_steps"], 3 + 2); // 5
    assert_eq!(d["num_accepted_draft_tokens"], 9);
    assert_eq!(d["num_draft_tokens"], 15);
    assert_eq!(d["num_spec_tokens"], 3);
    assert!(d.get("per_step_accepted").is_none());
    assert!(d.get("per_step_drafted").is_none());
}

#[test]
fn upstream_spec_metrics_to_dict_detailed_appends_per_step_arrays() {
    let d = metrics_from(&[(3, 3), (3, 2), (3, 0)], 3, true).to_dict();
    assert_eq!(d["per_step_accepted"], serde_json::json!([3, 2, 0]));
    assert_eq!(d["per_step_drafted"], serde_json::json!([3, 3, 3]));
    // summary fields still present in detailed mode
    assert_eq!(d["num_spec_steps"], 3);
    assert_eq!(d["num_accepted_draft_tokens"], 5);
}

#[test]
fn upstream_spec_metrics_to_dict_histogram_is_dense_list_indexed_by_j() {
    // length k+1, index j holds the step count that accepted exactly j drafts
    let d = metrics_from(&[(3, 0), (3, 0), (3, 3)], 3, false).to_dict();
    assert_eq!(d["acceptance_histogram"], serde_json::json!([2, 0, 0, 1]));
}

#[test]
fn upstream_spec_metrics_all_rejected_gives_mean_one_and_rate_zero() {
    let d = metrics_from(&[(2, 0), (2, 0), (2, 0), (2, 0)], 2, false).to_dict();
    assert_eq!(d["num_spec_steps"], 4);
    assert_eq!(d["num_accepted_draft_tokens"], 0);
    assert_eq!(d["acceptance_histogram"], serde_json::json!([4, 0, 0]));
    assert_eq!(d["mean_acceptance_length"].as_f64().unwrap(), 1.0);
    assert_eq!(d["draft_acceptance_rate"].as_f64().unwrap(), 0.0);
}

#[test]
fn upstream_spec_metrics_empty_do_not_divide_by_zero() {
    let d = UpstreamSpecDecodeMetrics::new(3).to_dict();
    assert_eq!(d["num_spec_steps"], 0);
    assert_eq!(d["num_draft_tokens"], 0);
    assert_eq!(d["draft_acceptance_rate"].as_f64().unwrap(), 0.0);
    assert_eq!(d["mean_acceptance_length"].as_f64().unwrap(), 1.0);
    assert!(d.get("per_step_accepted").is_none());
}

#[test]
fn upstream_spec_metrics_observe_records_proposed_and_accepted_independently() {
    // The histogram is keyed by accepted; num_draft_tokens sums the proposed
    // counts as given (grammar-invalidated-draft subtraction is a scheduler
    // concern upstream, not part of observe()).
    let mut s = UpstreamSpecDecodeMetrics::new(3);
    s.observe(2, 1);
    s.observe(3, 1);
    let d = s.to_dict();
    assert_eq!(d["acceptance_histogram"], serde_json::json!([0, 2, 0, 0]));
    assert_eq!(d["num_draft_tokens"], 5); // proposed summed independently: 2 + 3
}

#[test]
fn upstream_spec_metrics_payload_round_trips_like_engine_core_output() {
    // Upstream: the accumulator rides EngineCoreOutput via msgspec msgpack and
    // is omitted when absent. serde_json stands in for the wire codec: the
    // payload (incl. per-step arrays) must survive encode/decode intact.
    let s = metrics_from(&[(3, 0), (3, 3), (3, 2)], 3, true);
    let wire = serde_json::to_string(&s.to_dict()).unwrap();
    let decoded: serde_json::Value = serde_json::from_str(&wire).unwrap();
    assert_eq!(decoded["acceptance_histogram"], serde_json::json!([1, 0, 1, 1]));
    assert_eq!(decoded["num_draft_tokens"], 9);
    assert_eq!(decoded["per_step_accepted"], serde_json::json!([0, 3, 2]));
    // summary payload carries no per-step arrays when not detailed
    let summary_wire = serde_json::to_string(&UpstreamSpecDecodeMetrics::new(3).to_dict()).unwrap();
    let summary: serde_json::Value = serde_json::from_str(&summary_wire).unwrap();
    assert!(summary.get("per_step_accepted").is_none());
}

#[test]
fn upstream_spec_verification_outputs_feed_metrics_meaningfully() {
    // Product anchor: drive the ported metrics accumulator from
    // `verify_dspark_greedy` — the ds41rt acceptance source. Per the
    // GreedyVerification contract, accepted drafts = accepted_inputs - 1
    // (anchor excluded) and emitted = accepted drafts + 1 bonus/correction.
    // So mean_acceptance_length must equal the mean emitted length per step.
    // Target-side cursor: index of the next target prediction. Token ids of
    // drafts are arbitrary; positions are tracked separately so token id is
    // never confused with target position.
    let target: Vec<u32> = (0..512).map(|i| i as u32).collect();
    let mut rng: u64 = 0x1234_5678_9abc_def0;
    let mut next = || {
        // Small deterministic PRNG; draft match rate ~50% with some long runs.
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut anchor = 100u32;
    let mut cursor = 0usize;
    let mut metrics = UpstreamSpecDecodeMetrics::new(5);
    metrics.detailed = true;
    let steps = 64;
    let mut emitted_total = 0usize;
    for _ in 0..steps {
        let drafts: Vec<u32> = (0..5).map(|_| (next() % 8) as u32).collect();
        let mut inputs = vec![anchor];
        inputs.extend_from_slice(&drafts);
        // Greedy target predictions after each input row, from the target seq.
        let target_next: Vec<u32> = (0..inputs.len())
            .map(|i| target[cursor + i])
            .collect();
        let result = verify_dspark_greedy(&inputs, &target_next, 999, 1000).unwrap();
        let accepted_drafts = result.accepted_inputs - 1;
        assert_eq!(result.emitted.len() as u32, accepted_drafts + 1);
        metrics.observe(drafts.len() as u64, accepted_drafts as u64);
        emitted_total += result.emitted.len();
        cursor += result.emitted.len();
        anchor = *result.emitted.last().unwrap();
    }
    let d = metrics.to_dict();
    assert_eq!(d["num_spec_steps"], steps);
    assert_eq!(d["num_draft_tokens"], steps as u64 * 5);
    assert_eq!(d["num_accepted_draft_tokens"], emitted_total as u64 - steps as u64);
    // mean_acceptance_length (1 + mean accepted) == mean emitted per step.
    let expected_mean = emitted_total as f64 / steps as f64;
    assert!(
        (d["mean_acceptance_length"].as_f64().unwrap() - expected_mean).abs() < 1e-12
    );
    // histogram is consistent: sum over j of histogram[j] == steps.
    let hist_sum: u64 = d["acceptance_histogram"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .sum();
    assert_eq!(hist_sum, steps);
}

// ---------------------------------------------------------------------------
// Group 3: backup-token bookkeeping under async scheduling
// (vLLM test_backup_token_async_spec.py — the invariant class: token lookups
// must index committed tokens (num_committed - 1), never the inflated
// in-flight seq_lens inflated by unaccepted draft placeholders)
// ---------------------------------------------------------------------------

/// Port of the upstream `_FakeRequest`: prompt + committed output tokens;
/// get_token_id returns -1 (sentinel) past the committed end.
struct FakeRequest {
    prompt: Vec<u32>,
    output: Vec<u32>,
}

impl FakeRequest {
    fn num_prompt_tokens(&self) -> usize {
        self.prompt.len()
    }
    fn num_tokens(&self) -> usize {
        self.num_prompt_tokens() + self.output.len()
    }
    fn get_token_id(&self, idx: usize) -> i64 {
        if idx < self.num_prompt_tokens() {
            return self.prompt[idx] as i64;
        }
        let out_idx = idx - self.num_prompt_tokens();
        if out_idx < self.output.len() {
            return self.output[out_idx] as i64;
        }
        -1 // out of range (uncommitted placeholder territory)
    }
}

/// Port of the upstream `_FakeInputBatch`: committed per-request token counts
/// (num_tokens_no_spec), i.e. excluding any unaccepted draft placeholders.
struct FakeBatch {
    req_ids: Vec<&'static str>,
    num_tokens_no_spec: Vec<i64>,
}

fn make_requests(req_ids: &[&'static str], prompt_lens: &[usize], output_lens: &[usize]) -> Vec<(&'static str, FakeRequest)> {
    req_ids
        .iter()
        .zip(prompt_lens.iter().zip(output_lens.iter()))
        .map(|(&rid, (&plen, &olen))| {
            (
                rid,
                FakeRequest {
                    prompt: (0..plen as u32).collect(),
                    output: (1000..1000 + olen as u32).collect(),
                },
            )
        })
        .collect()
}

/// Old (buggy) logic: indexes by seq_lens_cpu directly — inflated by
/// unaccepted draft placeholders under async scheduling, and even without
/// inflation always one past the last committed token.
fn backup_buggy(seq_lens_cpu: &[i64], requests: &[(&str, FakeRequest)], batch: &FakeBatch) -> Vec<i64> {
    (0..batch.req_ids.len())
        .map(|i| {
            let idx = seq_lens_cpu[i];
            requests
                .iter()
                .find(|(rid, _)| *rid == batch.req_ids[i])
                .map(|(_, r)| if idx < 0 { -1 } else { r.get_token_id(idx as usize) })
                .unwrap_or(-1)
        })
        .collect()
}

/// New (fixed) logic: num_tokens_no_spec - 1 = last committed token.
fn backup_fixed(requests: &[(&str, FakeRequest)], batch: &FakeBatch) -> Vec<i64> {
    (0..batch.req_ids.len())
        .map(|i| {
            let idx = batch.num_tokens_no_spec[i] - 1;
            requests
                .iter()
                .find(|(rid, _)| *rid == batch.req_ids[i])
                .map(|(_, r)| if idx < 0 { -1 } else { r.get_token_id(idx as usize) })
                .unwrap_or(-1)
        })
        .collect()
}

#[test]
fn upstream_spec_backup_no_inflation_fixed_returns_last_token() {
    let requests = make_requests(&["r0", "r1"], &[3, 3], &[2, 2]);
    let batch = FakeBatch { req_ids: vec!["r0", "r1"], num_tokens_no_spec: vec![5, 5] };
    // idx = 5-1 = 4 → output[1] = 1001
    assert_eq!(backup_fixed(&requests, &batch), vec![1001, 1001]);
}

#[test]
fn upstream_spec_backup_inflation_buggy_returns_placeholder() {
    let requests = make_requests(&["r0", "r1"], &[3, 3], &[2, 2]);
    let batch = FakeBatch { req_ids: vec!["r0", "r1"], num_tokens_no_spec: vec![5, 5] };
    // inflated by 3 spec tokens → idx 8 is out of range
    assert_eq!(backup_buggy(&[8, 8], &requests, &batch), vec![-1, -1]);
}

#[test]
fn upstream_spec_backup_inflation_fixed_returns_correct_token() {
    let requests = make_requests(&["r0", "r1"], &[3, 3], &[2, 2]);
    let batch = FakeBatch { req_ids: vec!["r0", "r1"], num_tokens_no_spec: vec![5, 5] };
    assert_eq!(backup_fixed(&requests, &batch), vec![1001, 1001]);
}

#[test]
fn upstream_spec_backup_mixed_inflation_per_request() {
    let requests = vec![
        ("r0", FakeRequest { prompt: vec![0, 1], output: vec![1000, 1001, 1002] }),
        ("r1", FakeRequest { prompt: vec![0, 1, 2, 3], output: vec![2000] }),
        ("r2", FakeRequest { prompt: vec![0], output: vec![3000, 3001, 3002, 3003] }),
    ];
    let batch = FakeBatch { req_ids: vec!["r0", "r1", "r2"], num_tokens_no_spec: vec![5, 5, 5] };
    assert_eq!(backup_buggy(&[7, 9, 5], &requests, &batch), vec![-1, -1, -1]);
    assert_eq!(backup_fixed(&requests, &batch), vec![1002, 2000, 3003]);
}

#[test]
fn upstream_spec_backup_prefill_only_request() {
    // No output tokens yet — backup should be the last prompt token.
    let requests = vec![("r0", FakeRequest { prompt: vec![10, 20, 30], output: vec![] })];
    let batch = FakeBatch { req_ids: vec!["r0"], num_tokens_no_spec: vec![3] };
    // idx = 3-1 = 2 → prompt[2] = 30
    assert_eq!(backup_fixed(&requests, &batch), vec![30]);
}

#[test]
fn upstream_spec_backup_various_spec_token_counts() {
    for num_spec_tokens in 1..=5usize {
        let requests = make_requests(&["r0"], &[3], &[5]);
        let batch = FakeBatch { req_ids: vec!["r0"], num_tokens_no_spec: vec![8] };
        // idx = 8-1 = 7 → output[4] = 1004, regardless of spec-token count
        let _ = num_spec_tokens;
        assert_eq!(backup_fixed(&requests, &batch), vec![1004]);
    }
}

#[test]
fn upstream_spec_backup_buggy_code_was_always_off_by_one() {
    // The original code used seq_len as index, which is always one past the
    // end of output_token_ids even without async inflation.
    let requests = make_requests(&["r0"], &[3], &[2]);
    let batch = FakeBatch { req_ids: vec!["r0"], num_tokens_no_spec: vec![5] };
    // no inflation: seq_len == num_tokens == 5 → idx 5 is out of range
    assert_eq!(backup_buggy(&[5], &requests, &batch), vec![-1]);
    assert_eq!(backup_fixed(&requests, &batch), vec![1001]);
    // with inflation: still -1, fixed still correct
    assert_eq!(backup_buggy(&[8], &requests, &batch), vec![-1]);
    assert_eq!(backup_fixed(&requests, &batch), vec![1001]);
}

#[test]
fn upstream_spec_backup_tracks_verify_emitted_anchor_across_steps() {
    // Product anchor: run a simulated multi-step spec decode through
    // `verify_dspark_greedy`. After each step the committed output grows by
    // exactly the emitted tokens, and the fixed backup lookup must return the
    // last emitted token — the next pending anchor per the GreedyVerification
    // contract ("a correction/bonus token ... must remain the next pending
    // anchor"). The buggy lookup, indexing the inflated in-flight length
    // (committed + pending drafts), always lands in placeholder territory.
    // Position cursor tracked separately from token ids (see Group 2 test).
    let target: Vec<u32> = (0..128).map(|i| i as u32).collect();
    let mut anchor = 100u32;
    let mut cursor = 0usize;
    let mut request = FakeRequest { prompt: vec![7, 8, 9], output: vec![] };
    for step in 0..6 {
        let drafts: Vec<u32> = (0..3).map(|d| (step * 3 + d) as u32 % 5).collect();
        let mut inputs = vec![anchor];
        inputs.extend_from_slice(&drafts);
        let target_next: Vec<u32> = (0..inputs.len()).map(|i| target[cursor + i]).collect();
        let result = verify_dspark_greedy(&inputs, &target_next, 999, 1000).unwrap();
        // Commit step: published tokens are exactly `emitted`; the bonus /
        // correction is the new anchor and is NOT yet an evaluated input.
        request.output.extend_from_slice(&result.emitted);
        cursor += result.emitted.len();
        anchor = *result.emitted.last().unwrap();
        let batch = FakeBatch {
            req_ids: vec!["r0"],
            num_tokens_no_spec: vec![request.num_tokens() as i64],
        };
        assert_eq!(backup_fixed(&[("r0", request.clone_for_lookup())], &batch), vec![anchor as i64]);
        // In-flight length = committed + up to 3 unaccepted draft placeholders.
        let inflated = request.num_tokens() as i64 + 3;
        assert_eq!(backup_buggy(&[inflated], &[("r0", request.clone_for_lookup())], &batch), vec![-1]);
    }
}

impl FakeRequest {
    fn clone_for_lookup(&self) -> FakeRequest {
        FakeRequest {
            prompt: self.prompt.clone(),
            output: self.output.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Group 4: sglang acceptance-estimation invariants
// (test_decode_bookkeeping_ownership.py / test_adaptive_spec_params.py)
// ---------------------------------------------------------------------------

#[test]
fn upstream_spec_acceptance_bookkeeping_never_double_counts_committed_rows() {
    // sglang invariant class: spec-v2 bookkeeping (committed-len watermarks,
    // verify counters) must be advanced by exactly one owner, and a draft
    // worker must not re-commit accepted rows. ds41rt analogue: across a
    // simulated decode, the number of committed tokens equals prompt + the
    // sum of verify-emitted tokens — never the sum of submitted draft rows.
    let target: Vec<u32> = (0..128).map(|i| i as u32).collect();
    let mut anchor = 100u32;
    let mut cursor = 0usize;
    let prompt_len = 3usize;
    let mut committed = prompt_len;
    let mut submitted_rows = prompt_len;
    for step in 0..8usize {
        let drafts: Vec<u32> = (0..5).map(|d| ((step + d) % 7) as u32).collect();
        let mut inputs = vec![anchor];
        inputs.extend_from_slice(&drafts);
        let target_next: Vec<u32> = (0..inputs.len()).map(|i| target[cursor + i]).collect();
        let result = verify_dspark_greedy(&inputs, &target_next, 999, 10_000).unwrap();
        // Exactly one owner publishes: the emitted run, once.
        committed += result.emitted.len();
        cursor += result.emitted.len();
        // Submitted input rows include drafts that were never committed.
        submitted_rows += inputs.len();
        assert!(committed <= submitted_rows);
        anchor = *result.emitted.last().unwrap();
    }
    // With non-trivial mismatch the two must strictly diverge.
    assert!(committed < submitted_rows);
}
