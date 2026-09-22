//! Upstream-ported protocol/streaming invariants against the PRODUCTION
//! serving router (`native_v41`, mounted by the daemon) — review MAJOR 7
//! (2026-09-15): the legacy-router suites do not cover the deployed path.
//! A fake worker queue feeds synthetic `InferenceChunk` events; no GPU or
//! model weights required.
//!
//! Coverage map (upstream source -> invariant pinned here):
//! - vllm `test_non_object_body_validation.py` -> non-object bodies are a
//!   clean 400 with a bounded body (via the bounded `error()` constructor).
//! - Review BLOCKER follow-up -> huge validly typed fields (role) stay
//!   bounded on THIS router too.
//! - OpenAI protocol contract -> wrong model rejects with a 400; greedy
//!   (`temperature` absent or 0) is the served default and non-greedy
//!   temperature/top_p/top_k/min_p/seed are resolved for the engine.
//! - llama.cpp `test_chat_completion.py` SSE patterns -> `data:` frames,
//!   JSON payload per frame, `data: [DONE]` terminator; non-stream bodies
//!   carry choices[].finish_reason.

use axum::body::{to_bytes, Body};
use axum::http::{Method, Request, StatusCode};
use deepseek_recipe::stream::{InferenceChunk, InferenceFinishReason};
use serde_json::{json, Value};
use tower::ServiceExt;

use crate::native_v41::{router, NativeRequest};

const MODEL: &str = "deepseek-ai/DeepSeek-V4.1-Flash";

fn app() -> axum::Router {
    let (queue, _rx) = tokio::sync::mpsc::channel(4);
    router(queue)
}

fn post_json(body: Value) -> Request<Body> {
    Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

fn valid_request() -> Value {
    json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 8,
        "temperature": 0,
    })
}

async fn response_json(response: axum::response::Response) -> (StatusCode, Value, usize) {
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let len = bytes.len();
    let value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, value, len)
}

/// Feed the synthetic worker events for the first queued request.
fn spawn_driver(
    mut rx: tokio::sync::mpsc::Receiver<NativeRequest>,
    chunks: Vec<Result<InferenceChunk, crate::native_v41::NativeFailure>>,
) {
    tokio::spawn(async move {
        let job = rx.recv().await.expect("worker receives the request");
        for chunk in chunks {
            job.events.send(chunk).await.expect("event accepted");
        }
    });
}

#[tokio::test]
async fn non_object_body_returns_clean_bounded_400() {
    let app = app();
    let response = app.oneshot(post_json(json!([1, 2, 3]))).await.unwrap();
    let (status, value, len) = response_json(response).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(len < 4_000, "error body must be bounded, was {len} bytes");
    assert!(value["error"]["message"].is_string());
}

#[tokio::test]
async fn huge_valid_role_string_is_bounded_on_the_native_router() {
    let app = app();
    let mut body = valid_request();
    body["messages"][0]["role"] = json!("R".repeat(100_000));
    let response = app.oneshot(post_json(body)).await.unwrap();
    let (status, _value, len) = response_json(response).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(len < 4_000, "error body must be bounded, was {len} bytes");
}

#[tokio::test]
async fn wrong_model_is_rejected() {
    let app = app();
    let mut wrong_model = valid_request();
    wrong_model["model"] = json!("someone-else");
    let response = app.oneshot(post_json(wrong_model)).await.unwrap();
    let (status, value, _) = response_json(response).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(value["error"]["message"].as_str().unwrap().contains("model must be"));
}

/// Drive one request through the production router and return the sampling
/// parameters the engine actually resolved (None when the request was rejected
/// before admission).
async fn post_and_capture_sampling(
    body: Value,
) -> (StatusCode, Option<ds41rt_core::TargetSamplingParams>) {
    let (queue, mut rx) = tokio::sync::mpsc::channel(4);
    let app = router(queue);
    let (tx, sampling_rx) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        if let Some(job) = rx.recv().await {
            let _ = tx.send(job.sampling);
            let _ = job
                .events
                .send(Ok(InferenceChunk::Ready {
                    system_fingerprint: None,
                    prompt_usage: Default::default(),
                }))
                .await;
            let _ = job
                .events
                .send(Ok(InferenceChunk::Finish {
                    finish_reason: InferenceFinishReason::Stop,
                }))
                .await;
        }
    });
    let response = app.oneshot(post_json(body)).await.unwrap();
    let status = response.status();
    let _ = to_bytes(response.into_body(), usize::MAX).await;
    (status, sampling_rx.await.ok())
}

#[tokio::test]
async fn stochastic_sampling_is_accepted_and_resolved() {
    // temperature=0.7 enables sampling; unspecified filters stay disabled so
    // the served distribution is never silently truncated.
    let mut body = valid_request();
    body["temperature"] = json!(0.7);
    let (status, sampling) = post_and_capture_sampling(body).await;
    assert_eq!(status, StatusCode::OK, "non-greedy sampling must be served");
    let sampling = sampling.expect("engine received the request");
    assert!(!sampling.is_greedy());
    assert_eq!(sampling.temperature(), 0.7);
    assert_eq!(sampling.top_p(), 1.0, "top_p default must not truncate");
    assert_eq!(sampling.top_k(), None, "top_k default must stay disabled");
    assert_eq!(sampling.min_p(), 0.0);
}

#[tokio::test]
async fn greedy_default_and_explicit_zero_stay_greedy() {
    let (status, sampling) = post_and_capture_sampling(valid_request()).await;
    assert_eq!(status, StatusCode::OK);
    assert!(sampling.unwrap().is_greedy());
    let mut zero = valid_request();
    zero["temperature"] = json!(0);
    let (_, sampling) = post_and_capture_sampling(zero).await;
    assert!(sampling.unwrap().is_greedy());
}

#[tokio::test]
async fn explicit_filters_and_signed_seed_are_resolved() {
    let mut body = valid_request();
    body["temperature"] = json!(0.8);
    body["top_p"] = json!(0.9);
    body["top_k"] = json!(7);
    body["min_p"] = json!(0.05);
    body["seed"] = json!(-7);
    let (status, sampling) = post_and_capture_sampling(body).await;
    assert_eq!(status, StatusCode::OK);
    let sampling = sampling.unwrap();
    assert_eq!(sampling.top_p(), 0.9);
    assert_eq!(sampling.top_k(), Some(7));
    assert_eq!(sampling.min_p(), 0.05);
    assert_eq!(
        sampling.seed(),
        ds41rt_core::TargetSamplingParams::seed_from_i64(-7)
    );
    // 0 and -1 both disable top_k.
    for disabled in [json!(0), json!(-1)] {
        let mut body = valid_request();
        body["temperature"] = json!(1.0);
        body["top_k"] = disabled;
        let (_, sampling) = post_and_capture_sampling(body).await;
        assert_eq!(sampling.unwrap().top_k(), None);
    }
}

#[tokio::test]
async fn out_of_range_top_k_is_accepted_as_a_no_op() {
    // k >= vocabulary must be accepted and behave as top-k disabled, not fail.
    for huge in [json!(129_280), json!(1_000_000), json!(u32::MAX as u64)] {
        let mut body = valid_request();
        body["temperature"] = json!(1.0);
        body["top_k"] = huge.clone();
        let (status, sampling) = post_and_capture_sampling(body).await;
        assert_eq!(status, StatusCode::OK, "top_k={huge} must be accepted");
        assert_eq!(
            sampling.unwrap().top_k(),
            Some(huge.as_u64().unwrap() as usize)
        );
    }
}

#[tokio::test]
async fn invalid_sampling_parameters_are_rejected_before_admission() {
    for (field, value) in [
        ("temperature", json!(2.5)),
        ("temperature", json!(-0.1)),
        ("top_p", json!(0.0)),
        ("top_p", json!(1.5)),
        ("top_k", json!(-2)),
        ("top_k", json!(1.5)),
        ("min_p", json!(-0.1)),
        ("min_p", json!(1.5)),
    ] {
        let mut body = valid_request();
        body[field] = value.clone();
        let (status, sampling) = post_and_capture_sampling(body).await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "{field}={value} must be rejected"
        );
        assert!(sampling.is_none(), "{field}={value} must not reach admission");
    }
}

#[tokio::test]
async fn streaming_emits_sse_frames_and_done_terminator() {
    let (queue, rx) = tokio::sync::mpsc::channel(4);
    let app = router(queue);
    spawn_driver(
        rx,
        vec![
            Ok(InferenceChunk::Ready { system_fingerprint: None, prompt_usage: Default::default() }),
            Ok(InferenceChunk::Text { content: "hello".to_owned(), content_tokens: 5 }),
            Ok(InferenceChunk::Finish { finish_reason: InferenceFinishReason::Stop }),
        ],
    );
    let mut body = valid_request();
    body["stream"] = json!(true);
    let response = app.oneshot(post_json(body)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get("content-type").unwrap(),
        "text/event-stream"
    );
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    let frames: Vec<&str> = text.split("\n\n").filter(|f| !f.is_empty()).collect();
    assert!(frames.iter().all(|f| f.starts_with("data: ")), "pure data: framing: {text:?}");
    assert_eq!(frames.last().unwrap(), &"data: [DONE]");
    let payload: Value = serde_json::from_str(frames[0].trim_start_matches("data: ")).unwrap();
    assert!(payload.get("choices").is_some() || payload.get("object").is_some());
}

#[tokio::test]
async fn non_stream_body_carries_content_and_finish_reason() {
    let (queue, rx) = tokio::sync::mpsc::channel(4);
    let app = router(queue);
    spawn_driver(
        rx,
        vec![
            Ok(InferenceChunk::Text { content: "hello".to_owned(), content_tokens: 5 }),
            Ok(InferenceChunk::Finish { finish_reason: InferenceFinishReason::Stop }),
        ],
    );
    let response = app.oneshot(post_json(valid_request())).await.unwrap();
    let (status, value, _) = response_json(response).await;
    assert_eq!(status, StatusCode::OK);
    let choice = &value["choices"][0];
    assert_eq!(choice["finish_reason"], "stop");
    // Native serving defaults thinking ON: pre-think-end text lands in
    // reasoning_content (the official conversion precedence).
    let text = choice["message"]["content"].as_str().unwrap_or("")
        .to_owned()
        + choice["message"]["reasoning_content"].as_str().unwrap_or("");
    assert!(text.contains("hello"), "processed text is carried: {value}");
}

#[tokio::test]
async fn native_queue_pressure_is_429_with_retry_after_and_stats() {
    let (tx, _rx) = tokio::sync::mpsc::channel(1);
    let held = tx.clone().reserve_owned().await.unwrap();
    let app = crate::native_v41::router_with_admission(tx,
        crate::native_v41::NativeLimits::default(),
        std::sync::Arc::new(std::sync::Mutex::new(Value::Null)),
        std::time::Duration::from_millis(1));
    let response = app.clone().oneshot(post_json(valid_request())).await.unwrap();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(response.headers()["retry-after"], "1");
    let (_, body, _) = response_json(response).await;
    assert!(body["error"]["message"].as_str().unwrap().contains("queue"));
    let response = app.oneshot(Request::get("/v1/stats").body(Body::empty()).unwrap()).await.unwrap();
    let (_, stats, _) = response_json(response).await;
    assert_eq!(stats["http_queue_waits"], 1);
    assert_eq!(stats["http_queue_rejects"], 1);
    drop(held);
}
