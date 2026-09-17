//! Upstream-ported stop-string / finish-reason coverage.
//!
//! Sources:
//! - vLLM `tests/detokenizer/test_check_stop_strings.py` (6 tests): which stop
//!   string wins when several match the text appended in one step.
//! - vLLM `tests/detokenizer/test_stop_string_while_stop_model_terminates.py`
//!   (1 test): a stop string present in the same final step as model
//!   termination must still truncate the output.
//!
//! Mapping notes (ds41rt differs from vLLM, so the invariants are ported onto
//! ds41rt's actual surface in `completion.rs`):
//! - ds41rt selects the stop string with the smallest *start* index
//!   (`content.find` + `min_by_key`), not the smallest *completion* index as
//!   vLLM's `check_stop_strings` does. Where the two semantics disagree the
//!   test is `#[ignore]`d as a BUG(candidate).
//! - ds41rt has no `include_in_output` surface: the output always excludes the
//!   stop string (equivalent to vLLM `include_in_output=False`). The
//!   include_in_output=true variants are therefore asserted as "truncation
//!   always excludes the stop text", and the -1 (no truncation) case maps to
//!   "stop at end of text still truncates at its start".
//! - ds41rt scans the whole completed content rather than an incremental
//!   per-step window, so `new_char_count` windowing has no direct equivalent;
//!   the tiny backend's deterministic content stands in for the decoded text.
//!
//! Deliberately skipped upstream files (no ds41rt surface — recorded as
//! unmapped invariants, not silently dropped):
//! - vLLM `tests/v1/core/test_repetition_detection.py` (19 tests): ds41rt-api
//!   has no repetition detection (no frequency/presence penalty, no
//!   repetition checker anywhere in the crate).
//! - llama.cpp `tests/test-reasoning-budget.cpp` (6 tests): ds41rt has a
//!   thinking on/off surface (`request_thinking_enabled`) but no reasoning
//!   *budget* parameter (no token budget, no forced end-sequence sampler, no
//!   UTF-8 boundary detector); `validate_request` knows no budget fields.

use super::{base_request, test_state};
use crate::completion::build_completion;
use crate::request::stop_strings;
use crate::{ApiBackend, ApiTransport, StopSpec};

/// Deterministic tiny-backend content for a non-"count" prompt
/// (`ds41rt_core::deterministic_tiny_completion`): words joined by single
/// spaces. Byte offsets used below:
/// `hello`=0..5, `from`=6..10, `ds41rt`=11..17, `tiny`=18..22, `backend`=23..30.
const TINY_CONTENT: &str = "hello from ds41rt tiny backend";

async fn tiny_output_with_stop(stop: StopSpec, max_tokens: Option<usize>) -> crate::CompletionOutput {
    let state = test_state(ApiBackend::Tiny, ApiTransport::Inproc);
    let mut request = base_request("Say hello.");
    request.stop = Some(stop);
    if let Some(max_tokens) = max_tokens {
        request.max_tokens = Some(max_tokens);
    }
    build_completion(&state, request).await.unwrap()
}

// ---- vLLM test_check_stop_strings.py ports ----

/// Port of `test_earliest_completing_stop_wins_regardless_of_list_order`
/// (both parametrized orders). With stops `ds41rt` (starts 11) and `tiny`
/// (starts 18) the earliest match wins no matter the list order; truncation
/// is at the winner's start, so the output is order-independent.
#[tokio::test]
async fn earliest_stop_wins_regardless_of_list_order() {
    for stop in [
        StopSpec::Many(vec!["ds41rt".to_owned(), "tiny".to_owned()]),
        StopSpec::Many(vec!["tiny".to_owned(), "ds41rt".to_owned()]),
    ] {
        let output = tiny_output_with_stop(stop, None).await;
        assert_eq!(
            output.content.as_deref(),
            Some("hello from "),
            "truncation must be at the earliest stop start (index 11)"
        );
        assert_eq!(output.finish_reason, "stop");
    }
}

/// Port of `test_ties_broken_by_list_order`. vLLM ties are same-completion
/// matches (`ab` vs `b`); under ds41rt's start-index selection the tie case
/// is a prefix pair sharing a start offset (`ds41rt` vs `ds41rt tiny`, both
/// at 11). `min_by_key` is stable so list order decides the winner; either
/// winner truncates at the shared start, so both orders must give identical
/// output.
#[tokio::test]
async fn same_start_ties_are_broken_by_list_order() {
    let mut outputs = Vec::new();
    for stop in [
        StopSpec::Many(vec!["ds41rt".to_owned(), "ds41rt tiny".to_owned()]),
        StopSpec::Many(vec!["ds41rt tiny".to_owned(), "ds41rt".to_owned()]),
    ] {
        outputs.push(tiny_output_with_stop(stop, None).await);
    }
    assert_eq!(outputs[0].content, outputs[1].content);
    assert_eq!(
        outputs[0].content.as_deref(),
        Some("hello from "),
        "both prefix-pair orders truncate at the shared start (index 11)"
    );
}

/// Port of vLLM's true same-completion tie (`ab` vs `b` in "...ab"): stops that
/// COMPLETE at the same position but start at different offsets. The
/// list-order tie-break decides the winner, so reversing the order must change
/// the truncation point. (Review 2026-09-15: the prefix-pair test above shares
/// a start and cannot detect a `<` vs `<=` regression.)
#[tokio::test]
async fn completion_ties_are_broken_by_list_order() {
    for (stops, expected) in [
        (vec!["from ds41rt".to_owned(), "ds41rt".to_owned()], "hello "),
        (vec!["ds41rt".to_owned(), "from ds41rt".to_owned()], "hello from "),
    ] {
        let output = tiny_output_with_stop(StopSpec::Many(stops), None).await;
        assert_eq!(output.content.as_deref(), Some(expected));
        assert_eq!(output.finish_reason, "stop");
    }
}

/// Port of `test_completion_position_not_start_position`.
/// `from ds41rt tiny` *starts* earlier (6) than `ds41rt` (11) but *completes*
/// later (22 vs 17). vLLM selects `ds41rt` (earliest completion) and
/// truncates at 11 -> "hello from ". ds41rt selects by earliest start and
/// truncates at 6 -> "hello ". Selection by start position instead of
/// completion position loses the stop that actually fires first under
/// incremental decoding.
#[tokio::test]
async fn completion_position_not_start_position() {
    let output = tiny_output_with_stop(
        StopSpec::Many(vec!["from ds41rt tiny".to_owned(), "ds41rt".to_owned()]),
        None,
    )
    .await;
    assert_eq!(
        output.content.as_deref(),
        Some("hello from "),
        "the stop that completes first (ds41rt, completion 17) must win over \
         the earlier-starting longer stop (completion 22)"
    );
}

/// Port of `test_single_stop_in_window_unchanged` +
/// `test_earliest_completing_stop_include_in_output` (include_in_output has
/// no ds41rt equivalent; the stop text is always excluded, truncation is at
/// the stop's start).
#[tokio::test]
async fn single_stop_mid_text_truncates_at_stop_start() {
    let output = tiny_output_with_stop(StopSpec::One("ds41rt".to_owned()), None).await;
    assert_eq!(output.content.as_deref(), Some("hello from "));
    assert_eq!(output.finish_reason, "stop");
}

/// Port of the `include_in_output=True` "stop completes at the very end ->
/// no truncation needed (-1)" case, inverted for ds41rt semantics: there is
/// no include surface, so a stop completing at the end of the text still
/// truncates at its start and drops it from the output.
#[tokio::test]
async fn stop_at_end_of_text_truncates_and_excludes_stop() {
    let output = tiny_output_with_stop(StopSpec::One("backend".to_owned()), None).await;
    assert_eq!(
        output.content.as_deref(),
        Some("hello from ds41rt tiny "),
        "ds41rt always excludes the stop string (no include_in_output surface)"
    );
    assert_eq!(output.finish_reason, "stop");
}

/// Port of `test_no_match_and_empty_inputs_return_none`: no match, an empty
/// stop string, and an empty stop list all leave the content untouched.
/// (ds41rt filters empty stop strings before matching.)
#[tokio::test]
async fn no_match_and_empty_stops_leave_content_unchanged() {
    let no_match = tiny_output_with_stop(StopSpec::One("zzz".to_owned()), None).await;
    assert_eq!(no_match.content.as_deref(), Some(TINY_CONTENT));
    assert_eq!(no_match.finish_reason, "stop");

    let empty_string = tiny_output_with_stop(StopSpec::One(String::new()), None).await;
    assert_eq!(
        empty_string.content.as_deref(),
        Some(TINY_CONTENT),
        "an empty stop string must be ignored, not match at index 0"
    );

    let empty_list = tiny_output_with_stop(StopSpec::Many(Vec::new()), None).await;
    assert_eq!(empty_list.content.as_deref(), Some(TINY_CONTENT));
}

// ---- vLLM test_stop_string_while_stop_model_terminates.py port ----

/// Port of `test_stop_string_while_stop_token_terminates` (both fixture
/// params collapse in ds41rt because truncation always excludes the stop
/// text). The model-terminate signals in ds41rt are the finish_reason
/// derivations "stop" (natural) and "length" (max_tokens reached). A stop
/// string present in the terminated output must still truncate the content
/// and own the finish_reason, even when the raw reason would have been
/// "length".
#[tokio::test]
async fn stop_string_wins_over_model_termination_signal() {
    // Natural termination + stop string mid-text.
    let natural = tiny_output_with_stop(StopSpec::One("ds41rt".to_owned()), None).await;
    assert_eq!(natural.content.as_deref(), Some("hello from "));
    assert_eq!(natural.finish_reason, "stop");

    // Length termination (5 words generated, max_tokens=5 => raw "length")
    // + stop string mid-text: the stop string overrides the reason.
    let length_override = tiny_output_with_stop(StopSpec::One("ds41rt".to_owned()), Some(5)).await;
    assert_eq!(length_override.content.as_deref(), Some("hello from "));
    assert_eq!(
        length_override.finish_reason, "stop",
        "a matched stop string must override the length finish_reason"
    );

    // Length termination + stop string at the exact tail of the truncated
    // content (max_tokens=3 => "hello from ds41rt").
    let boundary = tiny_output_with_stop(StopSpec::One("ds41rt".to_owned()), Some(3)).await;
    assert_eq!(boundary.content.as_deref(), Some("hello from "));
    assert_eq!(boundary.finish_reason, "stop");

    // No stop match under length termination: reason stays "length".
    let unmatched = tiny_output_with_stop(StopSpec::One("zzz".to_owned()), Some(5)).await;
    assert_eq!(unmatched.content.as_deref(), Some(TINY_CONTENT));
    assert_eq!(unmatched.finish_reason, "length");
}

// ---- StopSpec parsing (request surface) ----

/// The request-level `stop` parameter accepts one string, many strings, or
/// absence; `stop_strings` normalizes all three.
#[test]
fn stop_strings_normalizes_one_many_and_absent() {
    assert_eq!(
        stop_strings(Some(&StopSpec::One("ds41rt".to_owned()))),
        vec!["ds41rt".to_owned()]
    );
    assert_eq!(
        stop_strings(Some(&StopSpec::Many(vec!["a".to_owned(), "b".to_owned()]))),
        vec!["a".to_owned(), "b".to_owned()]
    );
    assert_eq!(stop_strings(None), Vec::<String>::new());
}
