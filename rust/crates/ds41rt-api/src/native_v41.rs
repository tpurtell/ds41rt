//! Official V4.1 protocol conversion and bounded handoff to a CUDA owner.
use axum::{
    body::Body,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
pub use deepseek_recipe::stream::{InferenceChunk, InferenceFinishReason, PromptUsage};
use deepseek_recipe::{
    openai::ChatCompletionRequest,
    request::{ConversionOptions, ProtocolRequest},
    response::ProtocolResponse,
    stream::StreamProcessor,
    util::append_delta::AppendDelta,
};
use deepseek_recipe_encoding::{v4::dsv41::DeepseekV41Encoding, PromptEncoding};
use ds41rt_core::TargetSamplingParams;
use futures::StreamExt;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

pub const MODEL: &str = "deepseek-ai/DeepSeek-V4.1-Flash";
mod limits;
mod admission;
mod constraints;
mod tools;
pub use constraints::NativeConstraint;
mod images;
#[cfg(test)]
mod unicode_tests;
pub use limits::{NativeLimits, MAX_CONTEXT_TOKENS, MAX_OUTPUT_TOKENS};
#[derive(Debug, Clone)]
pub enum NativeFailure {
    BadRequest(String),
    Worker(String),
}
impl std::fmt::Display for NativeFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self { Self::BadRequest(message) | Self::Worker(message) => f.write_str(message) }
    }
}
impl std::error::Error for NativeFailure {}
impl From<String> for NativeFailure {
    fn from(message: String) -> Self { Self::Worker(message) }
}
impl From<&str> for NativeFailure {
    fn from(message: &str) -> Self { Self::Worker(message.into()) }
}
pub struct NativeRequest {
    pub prompt: String,
    pub constraint: Option<NativeConstraint>,
    pub images: Vec<ds41rt_loader::V41Image>,
    pub max_tokens: usize,
    /// Resolved target-sampling parameters. `TargetSamplingParams::greedy()`
    /// keeps the legacy device-argmax route; anything else selects from the
    /// full vocabulary through the shared exact sampler.
    pub sampling: TargetSamplingParams,
    pub events: mpsc::Sender<Result<InferenceChunk, NativeFailure>>,
}
/// Serving statistics the CUDA owner publishes (a JSON object; `null` until the first publish).
pub type SharedStats = Arc<Mutex<Value>>;
#[derive(Clone)]
struct NativeState {
    queue: mpsc::Sender<NativeRequest>,
    limits: NativeLimits,
    images: images::ImageDecoder,
    stats: SharedStats,
    admission: admission::Admission,
}
pub fn router(queue: mpsc::Sender<NativeRequest>) -> Router {
    router_with_limits(queue, NativeLimits::default())
}
pub fn router_with_limits(queue: mpsc::Sender<NativeRequest>, limits: NativeLimits) -> Router {
    router_with_limits_and_stats(queue, limits, Arc::new(Mutex::new(Value::Null)))
}
pub fn router_with_limits_and_stats(queue: mpsc::Sender<NativeRequest>, limits: NativeLimits, stats: SharedStats) -> Router {
    router_with_admission(queue, limits, stats, std::time::Duration::from_secs(25))
}
pub fn router_with_admission(queue: mpsc::Sender<NativeRequest>, limits: NativeLimits,
    stats: SharedStats, wait: std::time::Duration) -> Router {
    let admission = admission::Admission::new(queue.max_capacity(), wait);
    let images = images::ImageDecoder::new(queue.max_capacity());
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/stats", get(stats_route))
        .route("/v1/chat/completions", post(chat))
        .layer(axum::extract::DefaultBodyLimit::max(images::BODY_BYTES))
        .with_state(NativeState { queue, limits, images, stats, admission })
}
async fn stats_route(State(state): State<NativeState>) -> Json<Value> {
    let mut value = state.stats.lock().map(|stats| stats.clone()).unwrap_or(Value::Null);
    if !value.is_object() { value = json!({}); }
    let object = value.as_object_mut().unwrap();
    object.extend(state.admission.metrics().as_object().unwrap().clone());
    object.insert("http_queue_len".into(), json!(state.queue.max_capacity() - state.queue.capacity()));
    Json(value)
}
async fn models(State(state): State<NativeState>) -> Json<Value> {
    Json(json!({"object":"list","data":[{"id":MODEL,"object":"model","owned_by":"deepseek-ai",
        "max_context_tokens":state.limits.context(),"max_output_tokens":state.limits.output()}]}))
}
async fn health(State(state): State<NativeState>) -> StatusCode {
    if state.queue.is_closed() {
        StatusCode::SERVICE_UNAVAILABLE
    } else {
        StatusCode::OK
    }
}
fn error(status: StatusCode, message: impl ToString) -> Response {
    // Bound upstream parse/validation details before they reach the response
    // body: serde invalid-type errors echo the full offending string (e.g. a
    // 100 KB string in a wrongly-typed field). Same class as the JsonRejection
    // echo fixed in lib.rs (upstream vLLM #49239). This router is the
    // production serving path (mounted by the daemon) — verified live on the
    // fleet 2026-09-15.
    let message = crate::error::bounded_error_detail(&message.to_string());
    (
        status,
        Json(json!({"error":{"message":message,"type":"native_v41_error"}})),
    )
        .into_response()
}

static NEXT_TARGET_SEED: AtomicU64 = AtomicU64::new(0);

fn generated_target_seed() -> u64 {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    now ^ NEXT_TARGET_SEED.fetch_add(0x9e37_79b9_7f4a_7c15, Ordering::Relaxed)
}

/// Resolve the served target-sampling parameters from the raw request body.
///
/// `top_k`, `min_p` and the signed `seed` are not part of the pinned
/// `deepseek-recipe` adapter, so they are read here. `top_p`/`temperature` are
/// also re-read so one validator owns the whole sampling contract.
///
/// Greedy is the served default: an absent or zero `temperature` keeps the
/// legacy argmax route and ignores the other filters. Once sampling is opted
/// into, unspecified filters are disabled (`top_p = 1`, no `top_k`,
/// `min_p = 0`) rather than silently inheriting a truncation.
fn request_target_sampling(body: &Value) -> Result<TargetSamplingParams, String> {
    let number = |name: &str| -> Result<Option<f64>, String> {
        match body.get(name) {
            None | Some(Value::Null) => Ok(None),
            Some(value) => value
                .as_f64()
                .map(Some)
                .ok_or_else(|| format!("{name} must be a finite number")),
        }
    };
    let temperature = number("temperature")?.unwrap_or(0.0) as f32;
    let top_p = number("top_p")?.unwrap_or(1.0) as f32;
    let min_p = number("min_p")?.unwrap_or(0.0) as f32;
    let top_k = match body.get("top_k") {
        None | Some(Value::Null) => None,
        Some(value) => {
            let k = value
                .as_i64()
                .ok_or_else(|| "top_k must be an integer".to_owned())?;
            if k == 0 || k == -1 {
                None
            } else if k < 0 {
                return Err("top_k must be -1, 0, or a positive integer".to_owned());
            } else {
                Some(k as usize)
            }
        }
    };
    let seed = match body.get("seed") {
        None | Some(Value::Null) => generated_target_seed(),
        Some(value) => {
            let seed = value
                .as_i64()
                .ok_or_else(|| "seed must be an integer".to_owned())?;
            TargetSamplingParams::seed_from_i64(seed)
        }
    };
    TargetSamplingParams::new(temperature, top_p, top_k, min_p, seed)
        .map_err(|error| error.to_string())
}
async fn chat(State(state): State<NativeState>, Json(mut body): Json<Value>) -> Response {
    let assistance = match body.get("tool_decoding_assistance") {
        None | Some(Value::Null) => true,
        Some(Value::Bool(value)) => *value,
        _ => return error(StatusCode::BAD_REQUEST, "tool_decoding_assistance must be boolean"),
    };
    let response_format = body.get("response_format").cloned().filter(|v| !v.is_null());
    // The adapter crate deserializes `response_format.json_schema` into a
    // fieldless "accepted and ignored" variant, so the schema is only available
    // in the raw body. Enforce the OpenAI strict-mode subset here, before any
    // backend admission and independently of the thinking mode, reusing the
    // same validator as the OpenAI-compat `validate_request` path.
    if let Some(format) = response_format.as_ref() {
        if format.get("type").and_then(Value::as_str) == Some("json_schema") {
            if let Some(definition) = format.get("json_schema") {
                let strict = match definition.get("strict") {
                    None | Some(Value::Null) => false,
                    Some(Value::Bool(strict)) => *strict,
                    _ => return error(
                        StatusCode::BAD_REQUEST,
                        "response_format.json_schema.strict must be boolean",
                    ),
                };
                if strict {
                    if let Some(schema) = definition.get("schema") {
                        if let Err(rejection) = crate::request::validate_strict_json_schema(
                            schema,
                            "response_format.json_schema.schema",
                        ) {
                            return rejection.into_response();
                        }
                    }
                }
            }
        }
    }
    // The recipe rejects its regex variant, while native XGrammar supports it.
    // Keep the original format for enforcement and render it as ordinary text.
    if response_format.as_ref().and_then(|v| v.get("type")).and_then(Value::as_str) == Some("regex") {
        body["response_format"] = json!({"type":"text"});
    }
    let sampling = match request_target_sampling(&body) {
        Ok(sampling) => sampling,
        Err(message) => return error(StatusCode::BAD_REQUEST, message),
    };
    // The adapter's `seed` field is `u64`; ds41rt keeps the signed convention,
    // so drop it after resolution to keep a negative seed from failing serde.
    if let Some(object) = body.as_object_mut() {
        object.remove("seed");
    }
    let mut parsed: ChatCompletionRequest = match serde_json::from_value(body) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let include_usage = parsed.include_usage();
    let parallel = parsed.parallel_tool_calls.unwrap_or(true);
    let selection = tools::Selection::extract(&mut parsed);
    // Native serving defaults to thinking at the adapter's high effort. Explicit
    // thinking/effort settings retain the official conversion precedence.
    let mut converted = match parsed.convert(ConversionOptions::default().with_default_thinking_mode(true)) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    if let Err(e) = selection.apply(&mut converted.conversation.tools) {
        return error(StatusCode::BAD_REQUEST, e);
    }
    if converted.model.as_deref() != Some(MODEL) {
        return error(StatusCode::BAD_REQUEST, format!("model must be {MODEL}"));
    }
    let max_tokens = match state.limits.requested_output(converted.inference_options.max_tokens) {
        Ok(limit) => limit,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let response_validator = match constraints::response_validator(response_format.as_ref()) {
        Ok(validator) => validator,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let tool_constraints = match tools::ToolConstraints::new(&converted.conversation.tools,
        converted.conversation.tool_choice, selection.required, parallel,
        assistance || response_format.as_ref().is_some_and(|v| v["type"] != "text")) {
        Ok(tools) => tools,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let constraint = match constraints::response_constraint(response_format, converted.conversation.thinking_mode, tool_constraints.as_ref()) {
        Ok(constraint) => constraint,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let mut validator = tools::CompletionValidator::new(response_validator, tool_constraints);
    let rendered = DeepseekV41Encoding::new().render_conversation(&converted.conversation);
    let streaming = converted.stream;
    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let generator = ChatCompletionRequest::chunk_generator(&converted, id.clone(), MODEL.into())
        .with_include_usage(!streaming || include_usage);
    let processor = StreamProcessor::new(generator, converted.parsing_options);
    // Rendered sources own the image payloads needed by preprocessing. Do not
    // retain another copy of their data URLs throughout the generated response.
    drop(converted.conversation);
    if rendered.image_sources.len() > ds41rt_loader::V41_MAX_IMAGES {
        return error(StatusCode::BAD_REQUEST, "at most 16 images are supported");
    }
    let permit = match state.admission.reserve(state.queue.clone()).await {
        Ok(permit) => permit,
        Err(admission::Rejected::Closed) => return error(StatusCode::SERVICE_UNAVAILABLE, "worker queue is closed"),
        Err(admission::Rejected::Overloaded) => {
            let mut response = error(StatusCode::TOO_MANY_REQUESTS, "request queue is full or its wait budget expired");
            response.headers_mut().insert(axum::http::header::RETRY_AFTER, axum::http::HeaderValue::from_static("1"));
            return response;
        }
    };
    let prepared = if rendered.image_sources.is_empty() { Vec::new() } else {
        // The queue permit bounds waiters while up to four decoders run. A C16
        // burst should wait here instead of imposing a hidden C4 image limit.
        let slot = match state.images.slots.clone().acquire_owned().await {
            Ok(slot) => slot,
            Err(_) => return error(StatusCode::SERVICE_UNAVAILABLE, "image preparation is closed"),
        };
        let decoder = state.images.clone();
        let result = tokio::task::spawn_blocking(move || {
            let _slot = slot;
            decoder.decode(rendered.image_sources)
        }).await;
        match result {
            Ok(Ok(images)) => images,
            Ok(Err(e)) => return error(StatusCode::BAD_REQUEST, format!("{e:#}")),
            Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, e),
        }
    };
    let (events, mut receive) = mpsc::channel(16);
    // Recipe 0.1.0 uses a protocol placeholder; the pinned model tokenizer
    // spells token 129264 differently. Preserve the text-only prompt verbatim.
    let prompt = if prepared.is_empty() { rendered.prompt }
        else { rendered.prompt.replace("<｜image｜>", "<｜deepseek_image｜>") };
    let job = NativeRequest {
        prompt,
        constraint,
        images: prepared,
        max_tokens,
        sampling,
        events,
    };
    permit.send(job);
    // Admission errors must retain their cause and HTTP status, including for
    // SSE, before a protocol processor can turn early EOF into a finish chunk.
    let first = match receive.recv().await {
        Some(Ok(chunk)) => chunk,
        Some(Err(NativeFailure::BadRequest(message))) => return error(StatusCode::BAD_REQUEST, message),
        Some(Err(message)) => return error(StatusCode::INTERNAL_SERVER_ERROR, message),
        None => return error(StatusCode::INTERNAL_SERVER_ERROR, "native worker ended without completion"),
    };
    // Never let a failed/disconnected backend be converted to a successful EOF.
    let failure = Arc::new(Mutex::new(None::<String>));
    let input_failure = failure.clone();
    let input = async_stream::stream! {
        let mut finished = matches!(first, InferenceChunk::Finish { .. });
        yield first;
        while !finished {
            let Some(event) = receive.recv().await else { break; };
            match event {
                Ok(chunk) => {
                    finished = matches!(chunk,InferenceChunk::Finish { .. });
                    yield chunk;
                    if finished { break; }
                }
                Err(message) => { *input_failure.lock().unwrap() = Some(message.to_string()); break; }
            }
        }
        if !finished {
            input_failure.lock().unwrap().get_or_insert_with(|| "native worker ended without completion".into());
        }
    };
    let chunks = processor.process(input);
    let chunks = async_stream::stream! {
        futures::pin_mut!(chunks);
        while let Some(chunk) = chunks.next().await {
            match chunk {
                Ok(chunk) => {
                    if validator.enabled() {
                        if let Err(e) = validator.observe(&serde_json::to_value(&chunk).unwrap()) {
                            yield Err(e); return;
                        }
                    }
                    yield Ok(chunk);
                }
                Err(e) => { yield Err(anyhow::anyhow!(e.to_string())); return; }
            }
        }
    };
    if streaming {
        let stream = async_stream::stream! {
            futures::pin_mut!(chunks);
            while let Some(chunk) = chunks.next().await {
                let failed = failure.lock().unwrap().clone();
                if let Some(message) = failed {
                    yield Err::<String,std::io::Error>(std::io::Error::other(message)); return;
                }
                match chunk {
                    Ok(chunk) => {
                        yield Ok(format!("data: {}\n\n",serde_json::to_string(&chunk).unwrap()));
                    },
                    Err(e) => { yield Err(std::io::Error::other(e.to_string())); return; }
                }
            }
            let failed = failure.lock().unwrap().clone();
            if let Some(message) = failed { yield Err(std::io::Error::other(message)); return; }
            yield Ok("data: [DONE]\n\n".to_owned());
        };
        return (
            [
                ("content-type", "text/event-stream"),
                ("cache-control", "no-cache"),
            ],
            Body::from_stream(stream),
        )
            .into_response();
    }
    type ChatResponse = <ChatCompletionRequest as ProtocolRequest>::Response;
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let mut response = ChatResponse::new(id, MODEL.into(), created, 0, 0);
    futures::pin_mut!(chunks);
    while let Some(chunk) = chunks.next().await {
        match chunk {
            Ok(chunk) => response.append(chunk),
            Err(e) => {
                let message = failure.lock().unwrap().clone().unwrap_or_else(|| e.to_string());
                return error(StatusCode::INTERNAL_SERVER_ERROR, message);
            }
        }
    }
    if let Some(message) = failure.lock().unwrap().clone() {
        return error(StatusCode::INTERNAL_SERVER_ERROR, message);
    }
    Json(response).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tower::ServiceExt;
    fn request(stream: bool) -> axum::http::Request<Body> {
        axum::http::Request::post("/v1/chat/completions").header("content-type","application/json").body(Body::from(json!({"model":MODEL,"messages":[{"role":"user","content":"What is 2 + 2? Answer with just the number."}],"thinking":{"type":"disabled"},"temperature":0,"max_tokens":16,"stream":stream}).to_string())).unwrap()
    }
    fn strict_schema_body(thinking_disabled: bool, schema: Value) -> Body {
        let mut body = json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "Return x."}],
            "temperature": 0,
            "max_tokens": 16,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "strict_probe", "strict": true, "schema": schema}}
        });
        if thinking_disabled {
            body["thinking"] = json!({"type": "disabled"});
        }
        Body::from(body.to_string())
    }

    #[tokio::test]
    async fn strict_schema_subset_is_rejected_before_admission_in_every_thinking_mode() {
        // `strict: true` without `required` is not a valid OpenAI strict schema.
        let invalid = json!({"type": "object", "properties": {"x": {"type": "string"}},
            "additionalProperties": false});
        for thinking_disabled in [false, true] {
            let (tx, _rx) = mpsc::channel::<NativeRequest>(1);
            let request = axum::http::Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(strict_schema_body(thinking_disabled, invalid.clone()))
                .unwrap();
            let response = router(tx).oneshot(request).await.unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST,
                "thinking_disabled={thinking_disabled} must still reject");
            let bytes = axum::body::to_bytes(response.into_body(), 1 << 20).await.unwrap();
            let value: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(value["error"]["type"], "invalid_request_error");
            assert_eq!(value["error"]["param"], "response_format.json_schema.schema");
        }
    }

    #[tokio::test]
    async fn valid_strict_schema_is_not_newly_rejected() {
        let valid = json!({"type": "object", "properties": {"x": {"type": "string"}},
            "required": ["x"], "additionalProperties": false});
        let (tx, mut rx) = mpsc::channel::<NativeRequest>(1);
        let worker = tokio::spawn(async move {
            let job = rx.recv().await.unwrap();
            assert!(job.constraint.is_some());
            job.events.send(Ok(InferenceChunk::Ready { system_fingerprint: None,
                prompt_usage: PromptUsage { prompt_tokens: 1, prompt_cache_hit_tokens: 0 } })).await.unwrap();
            job.events.send(Ok(InferenceChunk::Text {
                content: "{\"x\":\"a\"}".into(), content_tokens: 5 })).await.unwrap();
            job.events.send(Ok(InferenceChunk::Finish {
                finish_reason: InferenceFinishReason::Stop })).await.unwrap();
        });
        let request = axum::http::Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(strict_schema_body(false, valid)).unwrap();
        let response = router(tx).oneshot(request).await.unwrap();
        assert_ne!(response.status(), StatusCode::BAD_REQUEST);
        worker.await.unwrap();
    }

    #[tokio::test]
    async fn output_limits_reach_worker_and_model_metadata() {
        for (limits, requested, expected) in [
            (NativeLimits::default(), None, 393_216),
            (NativeLimits::default(), Some(8192), 8192),
            (NativeLimits::default(), Some(u32::MAX), 393_216),
            (NativeLimits::new(256, 128).unwrap(), None, 128),
            (NativeLimits::new(256, 128).unwrap(), Some(8192), 128),
            (NativeLimits::new(256, 128).unwrap(), Some(8), 8),
        ] {
            let (tx, mut rx) = mpsc::channel::<NativeRequest>(1);
            let app = router_with_limits(tx, limits);
            let response = app.clone().oneshot(axum::http::Request::get("/v1/models")
                .body(Body::empty()).unwrap()).await.unwrap();
            let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
            let value: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(value["data"][0]["max_context_tokens"], limits.context());
            assert_eq!(value["data"][0]["max_output_tokens"], limits.output());
            let worker = tokio::spawn(async move {
                let job = rx.recv().await.unwrap();
                assert_eq!(job.max_tokens, expected);
                job.events.send(Ok(InferenceChunk::Ready { system_fingerprint: None,
                    prompt_usage: PromptUsage { prompt_tokens: 1, prompt_cache_hit_tokens: 0 } })).await.unwrap();
                job.events.send(Ok(InferenceChunk::Finish {
                    finish_reason: InferenceFinishReason::Length,
                })).await.unwrap();
            });
            let body = json!({"model":MODEL,"messages":[{"role":"user","content":"Count."}],
                "max_tokens":requested});
            let response = app.oneshot(axum::http::Request::post("/v1/chat/completions")
                .header("content-type", "application/json").body(Body::from(body.to_string())).unwrap()).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            worker.await.unwrap();
        }
        let (tx, mut rx) = mpsc::channel::<NativeRequest>(1);
        let body = json!({"model":MODEL,"messages":[{"role":"user","content":"Count."}],"max_tokens":0});
        let response = router(tx).oneshot(axum::http::Request::post("/v1/chat/completions")
            .header("content-type", "application/json").body(Body::from(body.to_string())).unwrap()).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert!(rx.try_recv().is_err());
    }
    #[tokio::test]
    async fn thinking_defaults_high_and_honors_explicit_overrides() {
        for (options, score) in [
            (json!({}), Some(75)),
            (json!({"thinking":{"type":"enabled"}}), Some(75)),
            (json!({"reasoning_effort":"low"}), Some(50)),
            (json!({"reasoning_effort":"high"}), Some(75)),
            (json!({"reasoning_effort":"max"}), Some(100)),
            (json!({"reasoning_effort":"none"}), None),
            (json!({"thinking":{"type":"disabled"},"reasoning_effort":"max"}), None),
        ] {
            let (tx, mut rx) = mpsc::channel::<NativeRequest>(1);
            let worker = tokio::spawn(async move {
                let job = rx.recv().await.unwrap();
                if let Some(score) = score {
                    assert!(job.prompt.contains(&format!("Reasoning Effort: {score} (range 1-100")));
                    assert!(job.prompt.ends_with("<think>"));
                } else {
                    assert!(!job.prompt.contains("Reasoning Effort:"));
                    assert!(job.prompt.ends_with("</think>"));
                }
                job.events.send(Ok(InferenceChunk::Ready { system_fingerprint: None,
                    prompt_usage: PromptUsage { prompt_tokens: 1, prompt_cache_hit_tokens: 0 } })).await.unwrap();
                job.events.send(Ok(InferenceChunk::Text {
                    content: if score.is_some() { "Compute. </think>4" } else { "4" }.into(),
                    content_tokens: 1,
                })).await.unwrap();
                job.events.send(Ok(InferenceChunk::Finish {
                    finish_reason: InferenceFinishReason::Stop,
                })).await.unwrap();
            });
            let mut body = json!({"model":MODEL,"messages":[{"role":"user","content":"2+2?"}],
                "max_tokens":16,"stream":false});
            body.as_object_mut().unwrap().extend(options.as_object().unwrap().clone());
            let request = axum::http::Request::post("/v1/chat/completions")
                .header("content-type", "application/json").body(Body::from(body.to_string())).unwrap();
            let response = router(tx).oneshot(request).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
            let value: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(value["choices"][0]["message"]["content"], "4");
            if score.is_some() {
                assert_eq!(value["choices"][0]["message"]["reasoning_content"], "Compute. ");
            }
            worker.await.unwrap();
        }
    }
    #[tokio::test]
    async fn official_prompt_and_both_response_modes() {
        for streaming in [false, true] {
            let (tx, mut rx) = mpsc::channel::<NativeRequest>(1);
            let worker = tokio::spawn(async move {
                let job = rx.recv().await.unwrap();
                assert_eq!(job.prompt,"<｜begin▁of▁sentence｜><｜User｜>What is 2 + 2? Answer with just the number.<｜Assistant｜></think>");
                assert_eq!(job.max_tokens, 16);
                for event in [
                    InferenceChunk::Ready {
                        system_fingerprint: None,
                        prompt_usage: PromptUsage {
                            prompt_tokens: 18,
                            prompt_cache_hit_tokens: 0,
                        },
                    },
                    InferenceChunk::Text {
                        content: "4".into(),
                        content_tokens: 1,
                    },
                    InferenceChunk::Finish {
                        finish_reason: InferenceFinishReason::Stop,
                    },
                ] {
                    job.events.send(Ok(event)).await.unwrap();
                }
            });
            let response = router(tx).oneshot(request(streaming)).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
                .await
                .unwrap();
            if streaming {
                let text = std::str::from_utf8(&body).unwrap();
                assert!(text.contains("\"content\":\"4\""));
                assert!(text.ends_with("data: [DONE]\n\n"));
            } else {
                let value: Value = serde_json::from_slice(&body).unwrap();
                assert_eq!(value["choices"][0]["message"]["content"], "4");
                assert_eq!(value["usage"]["prompt_tokens"], 18);
            }
            worker.await.unwrap();
        }
    }
    #[tokio::test]
    async fn admission_errors_preserve_status_and_cause_before_json_or_sse() {
        for streaming in [false, true] {
            for bad_request in [false, true] {
                let (tx, mut rx) = mpsc::channel::<NativeRequest>(1);
                let worker = tokio::spawn(async move {
                    let job = rx.recv().await.unwrap();
                    let message = "required parameter has incompatible value constraints: nx".to_string();
                    job.events.send(Err(if bad_request { NativeFailure::BadRequest(message) }
                        else { NativeFailure::Worker(message) })).await.unwrap();
                });
                let body = json!({"model":MODEL,"messages":[{"role":"user","content":"Call lookup."}],
                    "tools":[{"type":"function","function":{"name":"lookup","strict":true,
                        "parameters":{"type":"object","properties":{"nx":{"const":1}},"required":["nx"]}}}],
                    "tool_choice":"required","stream":streaming});
                let request = axum::http::Request::post("/v1/chat/completions")
                    .header("content-type", "application/json").body(Body::from(body.to_string())).unwrap();
                let response = router(tx).oneshot(request).await.unwrap();
                assert_eq!(response.status(), if bad_request { StatusCode::BAD_REQUEST }
                    else { StatusCode::INTERNAL_SERVER_ERROR });
                let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
                let value: Value = serde_json::from_slice(&bytes).unwrap();
                assert_eq!(value["error"]["message"], "required parameter has incompatible value constraints: nx");
                worker.await.unwrap();
            }
        }
    }
    #[tokio::test]
    async fn late_worker_failure_is_not_replaced_by_required_tool_validation() {
        for streaming in [false, true] {
            let (tx, mut rx) = mpsc::channel::<NativeRequest>(1);
            tokio::spawn(async move {
                let job = rx.recv().await.unwrap();
                job.events.send(Ok(InferenceChunk::Ready { system_fingerprint: None,
                    prompt_usage: PromptUsage { prompt_tokens: 1, prompt_cache_hit_tokens: 0 } })).await.unwrap();
                job.events.send(Err(NativeFailure::Worker("late execution failure".into()))).await.unwrap();
            });
            let body = json!({"model":MODEL,"messages":[{"role":"user","content":"Call lookup."}],
                "tools":[{"type":"function","function":{"name":"lookup","strict":true,
                    "parameters":{"type":"object","properties":{"n":{"const":1}},"required":["n"]}}}],
                "tool_choice":"required","stream":streaming});
            let request = axum::http::Request::post("/v1/chat/completions")
                .header("content-type", "application/json").body(Body::from(body.to_string())).unwrap();
            let response = router(tx).oneshot(request).await.unwrap();
            if streaming {
                assert!(axum::body::to_bytes(response.into_body(), 1024 * 1024).await.is_err());
            } else {
                assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
                let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
                let value: Value = serde_json::from_slice(&bytes).unwrap();
                assert_eq!(value["error"]["message"], "late execution failure");
            }
        }
    }
    #[tokio::test]
    async fn worker_error_and_missing_finish_are_not_success() {
        for explicit_error in [false, true] {
            let (tx, mut rx) = mpsc::channel::<NativeRequest>(1);
            tokio::spawn(async move {
                let job = rx.recv().await.unwrap();
                if explicit_error {
                    job.events
                        .send(Err("execution failed".into()))
                        .await
                        .unwrap();
                }
            });
            let response = router(tx).oneshot(request(false)).await.unwrap();
            assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        }
    }
}
