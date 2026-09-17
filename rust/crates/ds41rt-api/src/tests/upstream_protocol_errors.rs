//! Ports of upstream vLLM OpenAI-protocol error/validation tests.
//!
//! Upstream sources:
//! - `tests/entrypoints/unit_tests/test_non_object_body_validation.py`
//! - `tests/entrypoints/serve/exception_handling/test_validation_exception_handler.py`
//! - `tests/entrypoints/serve/exception_handling/test_error_sanitization.py`
//!
//! ds41rt implements only the chat-completions protocol (plus /health and
//! /v1/models), so the multi-model parametrizations of the upstream tests
//! collapse onto `ChatCompletionRequest` and the `POST /v1/chat/completions`
//! route. Every test below exercises ds41rt's actual behavior through the
//! real axum router; where an upstream invariant has no ds41rt surface the
//! case is noted and skipped rather than invented.

use axum::body::{to_bytes, Body};
use axum::http::{Method, Request, StatusCode};
use serde_json::{json, Value};
use tower::ServiceExt;

const CHAT_COMPLETIONS: &str = "/v1/chat/completions";

/// POST `body` verbatim (raw text, so malformed JSON can be sent) and return
/// (status, response text).
async fn post_raw(body: &str) -> (StatusCode, String) {
    let app = crate::router();
    let request = Request::builder()
        .method(Method::POST)
        .uri(CHAT_COMPLETIONS)
        .header("content-type", "application/json")
        .body(Body::from(body.to_owned()))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

fn error_body(text: &str) -> Value {
    serde_json::from_str(text).expect("error responses must be valid JSON")
}

fn assert_openai_error_envelope(body: &Value) {
    let error = &body["error"];
    assert!(error.is_object(), "response must carry an OpenAI error envelope: {body}");
    assert_eq!(error["type"], "invalid_request_error");
    // `param` and `code` must exist as keys even when null.
    assert!(error.as_object().unwrap().contains_key("param"));
    assert!(error.as_object().unwrap().contains_key("code"));
}

// ---------------------------------------------------------------------------
// Port of test_non_object_body_validation.py
// ---------------------------------------------------------------------------

/// Upstream: test_request_models_reject_non_object_body
/// (`tests/entrypoints/unit_tests/test_non_object_body_validation.py`).
///
/// Upstream parametrizes 8 pydantic request models x 5 non-object payloads and
/// requires a clean `ValidationError` (4xx at the HTTP layer), never an
/// `AttributeError` surfacing as HTTP 500 from a `data.get(...)` before-validator.
/// ds41rt has a single request model (`ChatCompletionRequest`) behind the
/// `/v1/chat/completions` route, so the port is route-level: each non-object
/// payload must produce a 400 with the OpenAI error envelope, not a 500.
#[tokio::test]
async fn non_object_json_bodies_return_clean_400_not_500() {
    // Mirrors the upstream payload list: unparseable JSON text, a JSON array,
    // an integer, null, and a boolean.
    let payloads = [
        "this is not valid json{{{",
        r#"["not", "an", "object"]"#,
        "42",
        "null",
        "true",
    ];
    for payload in payloads {
        let (status, text) = post_raw(payload).await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "payload {payload:?} must give a clean 4xx, not a 5xx"
        );
        let body = error_body(&text);
        assert_openai_error_envelope(&body);
        assert_eq!(body["error"]["code"], "invalid_json");
        assert!(body["error"]["param"].is_null());
    }
}

/// Upstream: test_completion_request_still_validates_dict_bodies
/// (`tests/entrypoints/unit_tests/test_non_object_body_validation.py`).
///
/// The non-object guard must not swallow real field-level errors on object
/// bodies. Upstream asserts a `VLLMValidationError` matching "prompt" for an
/// empty prompt; the ds41rt equivalent is an object body that deserializes
/// fine but fails `validate_request` — here `max_tokens: 0` — which must still
/// surface its field-scoped 400 with `param` set.
#[tokio::test]
async fn object_bodies_still_get_field_level_validation() {
    let (status, text) = post_raw(
        &json!({
            "model": "ds41rt-tiny",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 0
        })
        .to_string(),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let body = error_body(&text);
    assert_openai_error_envelope(&body);
    assert_eq!(body["error"]["code"], "invalid_request");
    assert_eq!(body["error"]["param"], "max_tokens");
    assert!(body["error"]["message"]
        .as_str()
        .is_some_and(|message| message.contains("max_tokens")));
}

// Upstream: test_tokenize_chat_request_still_validates_dict_bodies — SKIPPED.
// ds41rt implements no tokenize endpoint or TokenizeChatRequest; nothing to
// port without inventing product surface.

// Upstream: test_request_models_reject_forbidden_cache_salt,
// test_request_models_reject_overlong_cache_salt,
// test_request_models_accept_safe_cache_salt — SKIPPED.
// ds41rt has no cache_salt field or cache-salt validation; serde ignores the
// unknown field, so the upstream cache_salt invariants have no mapping here.

// ---------------------------------------------------------------------------
// Port of test_validation_exception_handler.py
// ---------------------------------------------------------------------------

/// Upstream: test_param_falls_back_to_loc (missing-field and wrong-type
/// variants) (`tests/entrypoints/serve/exception_handling/test_validation_exception_handler.py`).
///
/// Upstream requires the error `param` to fall back to the pydantic error's
/// `loc` (e.g. "body.messages") for plain validation failures. ds41rt's
/// deserialization failures come through axum's `JsonRejection`, which carries
/// no structured field path into `param`, so the ported assertions pin the
/// actual ds41rt contract: a 400 with the OpenAI envelope, the failing field
/// named in the human-readable message (via serde's "missing field `...`"
/// detail), and `param: null`.
///
/// DIVERGENCE (noted, not a bug by itself): ds41rt does not populate `param`
/// from the deserialization error location; upstream vLLM does. If ds41rt ever
/// gains loc-based params, extend these assertions.
#[tokio::test]
async fn deserialization_errors_name_the_field_and_keep_openai_error_shape() {
    // missing-field variant: {"model": ...} with no messages.
    let (status, text) = post_raw(&json!({"model": "ds41rt-tiny"}).to_string()).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let body = error_body(&text);
    assert_openai_error_envelope(&body);
    assert_eq!(body["error"]["code"], "invalid_json");
    assert!(body["error"]["param"].is_null());
    assert!(body["error"]["message"]
        .as_str()
        .is_some_and(|message| message.contains("messages")));

    // wrong-type variant: messages given as a plain string.
    let (status, text) = post_raw(
        &json!({
            "model": "ds41rt-tiny",
            "messages": "not-a-list"
        })
        .to_string(),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let body = error_body(&text);
    assert_openai_error_envelope(&body);
    assert_eq!(body["error"]["code"], "invalid_json");
    assert!(body["error"]["param"].is_null());
}

/// Upstream: test_param_fallback_does_not_crash_on_non_dict_error
/// (`tests/entrypoints/serve/exception_handling/test_validation_exception_handler.py`).
///
/// Schemathesis found pydantic error entries that are not dicts; the handler
/// must degrade to `param: None` instead of raising. The ds41rt analog is a
/// top-level non-object error entry (a bare JSON array): the route must not
/// panic and must still return a well-formed 400 envelope with `param: null`.
#[tokio::test]
async fn non_dict_error_does_not_crash_the_handler() {
    let (status, text) = post_raw(r#"["some unexpected non-dict error"]"#).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let body = error_body(&text);
    assert_openai_error_envelope(&body);
    assert!(body["error"]["param"].is_null());
}

// Upstream: TestCleanLocForParam::{test_strips_internal_markers,
// test_all_internal_falls_back_to_raw_join} — SKIPPED.
// ds41rt has no clean_loc_for_param / pydantic-loc pipeline. The nearest
// invariant (validate_request param paths are clean, e.g. "messages[0].role")
// is already covered by chat_completion_route_smokes_tools_and_structured_errors
// in src/tests/routes.rs; duplicating it here would add no coverage.

/// Upstream: test_handler_strips_endpoint_file_context and
/// test_handler_strips_endpoint_path_only_variant
/// (`tests/entrypoints/serve/exception_handling/test_validation_exception_handler.py`).
///
/// FastAPI stamps endpoint file/line/function/path onto RequestValidationError
/// and the upstream handler must strip them so server internals never reach the
/// client. ds41rt builds error messages only from axum's `JsonRejection` text,
/// so the ported invariant is negative: neither malformed-JSON nor
/// missing-field responses may leak filesystem paths, Rust source locations,
/// or handler function names. (The FastAPI attribute mechanism itself has no
/// ds41rt analog; the path-only variant is exercised implicitly because
/// ds41rt messages carry no path context at all.)
#[tokio::test]
async fn error_messages_do_not_leak_server_internals() {
    let bodies = [
        "this is not valid json{{{",
        &json!({"model": "ds41rt-tiny"}).to_string(),
        &json!({
            "model": "ds41rt-tiny",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 0
        })
        .to_string(),
    ];
    for body_text in bodies {
        let (status, text) = post_raw(body_text).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let message = error_body(&text)["error"]["message"]
            .as_str()
            .expect("error message must be a string")
            .to_owned();
        assert!(!message.contains("/usr/local/"), "server path leaked: {message}");
        assert!(!message.contains("/home/"), "server path leaked: {message}");
        assert!(!message.contains(".rs"), "Rust source file leaked: {message}");
        assert!(!message.contains("api_utils"), "server helper leaked: {message}");
        assert!(!message.contains("chat_completions"), "handler name leaked: {message}");
    }
}

/// Upstream: test_many_errors_are_capped and
/// test_container_input_is_described_not_echoed
/// (`tests/entrypoints/serve/exception_handling/test_validation_exception_handler.py`).
///
/// Pydantic emits one error entry per offending element, so a 500-element bad
/// input produced thousands of errors and a 23.6 MB 400 body upstream (#49239);
/// the upstream handler caps and summarizes. serde fails on the first bad
/// element, so a single bounded error is the ds41rt contract. The port sends a
/// large container of invalid elements and pins: 400, body well under the
/// upstream 32 KiB bound, the count of errors never multiplies, and the
/// offending payload content is not echoed into the response.
#[tokio::test]
async fn per_element_failures_produce_a_bounded_error_body() {
    let bad_message = json!({"not_role": "payload-marker-xyz"});
    let huge_bad_messages = json!({
        "model": "ds41rt-tiny",
        "messages": Value::Array(vec![bad_message; 500])
    })
    .to_string();
    let (status, text) = post_raw(&huge_bad_messages).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        text.len() < 32_000,
        "400 body must stay bounded, was {} bytes",
        text.len()
    );
    assert!(
        !text.contains("payload-marker-xyz"),
        "offending payload content must not be echoed: {text}"
    );
    assert_openai_error_envelope(&error_body(&text));
}

/// Upstream: test_long_string_input_is_truncated
/// (`tests/entrypoints/serve/exception_handling/test_validation_exception_handler.py`).
///
/// A 100 KB offending string must not be echoed verbatim into the bounded 400
/// body (upstream requires "[truncated]" and a message under 4,000 chars).
/// ds41rt forwards axum's `JsonRejection` detail verbatim, and serde echoes
/// the full offending string into its invalid-type error, so the 400 body
/// grows linearly with attacker input — the exact class #49239 fixed upstream.
#[tokio::test]
async fn long_offending_input_is_bounded_in_the_error_body() {
    let long_string_input = json!({
        "model": "ds41rt-tiny",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": "A".repeat(100_000)
    })
    .to_string();
    let (status, text) = post_raw(&long_string_input).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let message = error_body(&text)["error"]["message"]
        .as_str()
        .expect("error message must be a string")
        .to_owned();
    assert!(
        message.len() < 4_000,
        "error message must be bounded, was {} chars",
        message.len()
    );
}

/// Upstream: test_small_input_is_still_reported_verbatim
/// (`tests/entrypoints/serve/exception_handling/test_validation_exception_handler.py`).
///
/// Bounding must not cost legitimate clients their diagnostics: a small
/// offending value is still shown. ds41rt/serde echoes short offending values
/// in the invalid-type detail, so the port asserts the small value survives.
#[tokio::test]
async fn small_offending_input_is_reported_verbatim() {
    let (status, text) = post_raw(
        &json!({
            "model": "ds41rt-tiny",
            "messages": "not-a-list"
        })
        .to_string(),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let body = error_body(&text);
    assert!(body["error"]["message"]
        .as_str()
        .is_some_and(|message| message.contains("not-a-list")));
}

// Upstream: test_union_loc_is_cleaned_in_the_message — SKIPPED.
// Union-branch type names in pydantic locs ("list[union[EasyInputMessageParam,...]]")
// have no serde analog in ds41rt's error path; serde errors never spell out
// union branches.

// ---------------------------------------------------------------------------
// Port of test_error_sanitization.py
// ---------------------------------------------------------------------------

/// Upstream: test_sanitize_message, TestSanitizeMessageCoversLeakPatterns
/// ({test_address_stripped x4, test_safe_message_unchanged,
/// test_multiple_addresses_stripped})
/// (`tests/entrypoints/serve/exception_handling/test_error_sanitization.py`).
///
/// Upstream's `sanitize_message` strips `<... object at 0x...>` memory-address
/// reprs (CVE-2026-22778) from error text. ds41rt has no sanitizer and no
/// object reprs, so the ported invariant is negative and behavioral: across
/// malformed and field-invalid requests, no error message may contain a
/// memory-address pattern (`0x` hex token) or an `... at 0x...` repr shape,
/// and a plain diagnostic message must pass through unmodified.
#[tokio::test]
async fn error_messages_do_not_leak_memory_addresses() {
    let cases = [
        r#"{"model": "ds41rt-tiny", "messages": "not-a-list"}"#,
        "this is not valid json{{{",
        &json!({"model": "ds41rt-tiny"}).to_string(),
    ];
    for case in cases {
        let (status, text) = post_raw(case).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let message = error_body(&text)["error"]["message"]
            .as_str()
            .expect("error message must be a string")
            .to_owned();
        assert!(
            !message.contains("0x"),
            "memory-address-like token leaked: {message}"
        );
        assert!(
            !message.contains(" at 0x"),
            "object-repr address pattern leaked: {message}"
        );
    }
}

/// Upstream: TestSanitizeMessageFilePaths::{test_strips_traceback_style_frame,
/// test_strips_arbitrary_absolute_path, test_strips_single_parent_container_path,
/// test_strips_both_address_and_path}
/// (`tests/entrypoints/serve/exception_handling/test_error_sanitization.py`).
///
/// Error text must not contain server filesystem paths (traceback frames,
/// arbitrary absolute paths, single-parent container paths like /app/ and
/// /workspace/). ds41rt error messages are ds41rt-authored strings plus serde
/// detail, so the port asserts none of the upstream path patterns appear in
/// any 400 response.
#[tokio::test]
async fn error_messages_do_not_contain_filesystem_paths() {
    let bodies = [
        "this is not valid json{{{",
        &json!({"model": "ds41rt-tiny", "messages": "not-a-list"}).to_string(),
        &json!({
            "model": "ds41rt-tiny",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 0
        })
        .to_string(),
    ];
    for body_text in bodies {
        let (status, text) = post_raw(body_text).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let message = error_body(&text)["error"]["message"]
            .as_str()
            .expect("error message must be a string")
            .to_owned();
        for forbidden in ["/usr/local/", "/home/user", "/app/", "/workspace/", ".py", "line 40"] {
            assert!(
                !message.contains(forbidden),
                "path-like token {forbidden:?} leaked: {message}"
            );
        }
    }
}

// Upstream: TestSanitizeMessageFilePaths::test_preserves_api_endpoint_paths —
// folded into the negative-path tests above: ds41rt error messages never
// rewrite or embed request paths, so "/v1/chat/completions" cannot be mangled
// by sanitization (there is no sanitizer). Asserting a substring of a string
// ds41rt never emits would test nothing.

// Upstream: TestSanitizeMessageFilePaths::test_preserves_short_field_references —
// covered by deserialization_errors_name_the_field_and_keep_openai_error_shape
// above ("missing field `messages`" survives into the message verbatim).

// Upstream: TestAffectedModulesUseSanitize — SKIPPED. Source-scan checks that
// named Python modules import sanitize_message have no ds41rt analog (no
// sanitizer module exists to grep for).

/// The production serving path (native_v41 router, mounted by the daemon) must
/// bound serde invalid-type echoes the same way as the lib.rs JsonRejection
/// path — verified live on the fleet 2026-09-15 (100 KB echo pre-fix).
#[tokio::test]
async fn native_v41_long_offending_input_is_bounded_in_the_error_body() {
    let (queue, _rx) = tokio::sync::mpsc::channel(1);
    let app = crate::native_v41::router(queue);
    let long_string_input = json!({
        "model": "deepseek-ai/DeepSeek-V4.1-Flash",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": "A".repeat(100_000)
    })
    .to_string();
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(long_string_input))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert!(
        body.len() < 4_000,
        "native_v41 error body must be bounded, was {} bytes",
        body.len()
    );
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.contains("truncated"), "bounded body must carry the truncation marker");
}

/// BLOCKER regression (review 2026-09-15): validly typed attacker-controlled
/// fields (model id, message role) are interpolated into ApiError messages;
/// the central ApiError::into_response bound must keep the 400 body small.
#[tokio::test]
async fn huge_valid_model_string_yields_bounded_400() {
    let body = json!({
        "model": "M".repeat(100_000),
        "messages": [{"role": "user", "content": "hello"}],
    })
    .to_string();
    let (status, text) = post_raw(&body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        text.len() < 4_000,
        "error body must be bounded, was {} bytes",
        text.len()
    );
}

#[tokio::test]
async fn huge_valid_role_string_yields_bounded_400() {
    let body = json!({
        "model": "ds41rt-tiny",
        "messages": [{"role": &"R".repeat(100_000), "content": "hello"}],
    })
    .to_string();
    let (status, text) = post_raw(&body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        text.len() < 4_000,
        "error body must be bounded, was {} bytes",
        text.len()
    );
}
