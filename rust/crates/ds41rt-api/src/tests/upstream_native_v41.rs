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
//! - OpenAI protocol contract -> wrong model / non-zero temperature reject
//!   with 400s; temperature=0 (greedy) is the documented native contract
//!   (resolves DEFERRED divergence #4 for native clients).
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
async fn wrong_model_and_nonzero_temperature_rejected() {
    let app = app();
    let mut wrong_model = valid_request();
    wrong_model["model"] = json!("someone-else");
    let response = app.clone().oneshot(post_json(wrong_model)).await.unwrap();
    let (status, value, _) = response_json(response).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(value["error"]["message"].as_str().unwrap().contains("model must be"));

    let mut warm = valid_request();
    warm["temperature"] = json!(0.7);
    let response = app.oneshot(post_json(warm)).await.unwrap();
    let (status, value, _) = response_json(response).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        value["error"]["message"].as_str().unwrap().contains("temperature=0"),
        "native contract pins greedy sampling: {}",
        value["error"]["message"]
    );
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
