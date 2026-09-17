pub mod native_v41;
use axum::extract::rejection::JsonRejection;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use ds41rt_core::ExpertRequest;
#[cfg(test)]
use ds41rt_core::DEFAULT_MODEL_ID;
use std::net::{SocketAddr, ToSocketAddrs};
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use uuid::Uuid;

mod backends;
mod completion;
mod config;
#[allow(dead_code)]
mod constrained;
mod continuation;
mod encoding;
mod error;
mod metrics;
mod openai;
mod real_full;
mod real_slice;
mod request;
mod streaming;
mod tooling;

use backends::try_real_ds4_full_streaming_response;
#[cfg(test)]
pub(crate) use completion::build_completion;
pub(crate) use completion::{
    build_validated_completion, validate_completion_request, CompletionOutput,
};
pub use config::*;
use error::{invalid_request, openai_error};
pub(crate) use error::{runtime_error, ApiError};
use metrics::BackendMetrics;
pub use metrics::CompletionMetrics;
pub use openai::*;
pub use real_full::*;
pub use real_slice::*;
use streaming::chat_stream_response;

// Long-context messages and large tool schemas can exceed Axum's 2 MiB JSON
// default. Keep the larger exception scoped to chat completions.
const CHAT_COMPLETIONS_BODY_LIMIT_BYTES: usize = 256 * 1024 * 1024;

#[derive(Debug)]
pub(crate) struct ApiState {
    pub(crate) config: ApiConfig,
    pub(crate) service_instance_id: String,
    pub(crate) next_request_id: AtomicU64,
    pub(crate) tool_continuations: Mutex<continuation::ToolContinuationCache>,
}

#[derive(Debug)]
pub(crate) struct BackendCompletion {
    pub(crate) content: String,
    pub(crate) reasoning_content: Option<String>,
    pub(crate) completion_tokens: Option<usize>,
    pub(crate) stream_chunks: Option<Vec<String>>,
    pub(crate) metrics: BackendMetrics,
}

pub fn router() -> Router {
    router_with_config(ApiConfig::default())
}

pub fn router_with_config(config: ApiConfig) -> Router {
    let state = Arc::new(ApiState {
        config,
        service_instance_id: Uuid::new_v4().to_string(),
        next_request_id: AtomicU64::new(1),
        tool_continuations: Mutex::new(continuation::ToolContinuationCache::default()),
    });
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route(
            "/v1/chat/completions",
            post(chat_completions).layer(DefaultBodyLimit::max(CHAT_COMPLETIONS_BODY_LIMIT_BYTES)),
        )
        .with_state(state)
}

async fn health(State(state): State<Arc<ApiState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        service: "ds41rt",
        backend: match state.config.backend {
            ApiBackend::Tiny => "tiny",
            ApiBackend::SyntheticDs4Layer => "synthetic-ds4-layer",
            ApiBackend::RealDs4Slice => "real-ds4-slice",
            ApiBackend::RealDs4Full => "real-ds4-full",
        },
        transport: state.config.transport.label(),
    })
}

async fn models(State(state): State<Arc<ApiState>>) -> Json<ModelList> {
    Json(ModelList {
        object: "list",
        data: vec![
            ModelInfo {
                id: "ds41rt-tiny".to_owned(),
                object: "model",
                owned_by: "ds41rt",
            },
            ModelInfo {
                id: "ds41rt-synthetic-ds4-layer".to_owned(),
                object: "model",
                owned_by: "ds41rt",
            },
            ModelInfo {
                id: format!("{}-slice", state.config.model_id),
                object: "model",
                owned_by: "ds41rt",
            },
            ModelInfo {
                id: format!("{}-full", state.config.model_id),
                object: "model",
                owned_by: "ds41rt",
            },
            ModelInfo {
                id: state.config.model_id.clone(),
                object: "model",
                owned_by: "ds41rt",
            },
        ],
    })
}

async fn chat_completions(
    State(state): State<Arc<ApiState>>,
    payload: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Response {
    let Json(request) = match payload {
        Ok(payload) => payload,
        Err(err) => {
            let detail = error::bounded_error_detail(&err.body_text());
            eprintln!("chat_completions_invalid_json error={detail}");
            return openai_error(
                StatusCode::BAD_REQUEST,
                format!("invalid JSON request: {detail}"),
                None,
                Some("invalid_json".to_owned()),
            );
        }
    };

    let stream_response = request.stream;
    let stream_include_usage = request
        .stream_options
        .as_ref()
        .is_some_and(|options| options.include_usage);
    if stream_response {
        match try_real_ds4_full_streaming_response(Arc::clone(&state), &request) {
            Ok(Some(response)) => return response,
            Ok(None) => {}
            Err(err) => return err.into_response(),
        }
    }
    let validated = match validate_completion_request(&state, request) {
        Ok(validated) => validated,
        Err(err) => return completion_error_response(err),
    };
    let completion = build_validated_completion(&state, validated).await;
    match completion {
        Ok(output) if stream_response => chat_stream_response(output, stream_include_usage),
        Ok(output) => Json(output.into_response_body()).into_response(),
        Err(err) => completion_error_response(err),
    }
}

fn completion_error_response(err: ApiError) -> Response {
    eprintln!(
        "chat_completions_error status={} code={} message={}",
        err.status,
        err.code.as_deref().unwrap_or("unknown"),
        error::bounded_error_detail(&err.message)
    );
    err.into_response()
}

pub(crate) async fn dispatch_expert_request(
    state: &ApiState,
    target: Option<&str>,
    request: &ExpertRequest,
) -> Result<ds41rt_core::ExpertResponse, ApiError> {
    match state.config.transport {
        ApiTransport::Inproc => ds41rt_transport::inproc_roundtrip(request)
            .await
            .map_err(runtime_error),
        ApiTransport::Tcp => {
            let target = target.ok_or_else(|| {
                runtime_error("TCP transport requires expert targets for synthetic-ds4-layer")
            })?;
            let addr = parse_expert_target(target)?;
            ds41rt_transport::tcp_protocol_v2_expert_request_roundtrip(
                addr,
                request,
                Default::default(),
            )
            .await
            .map_err(runtime_error)
        }
        ApiTransport::TcpDebugJson => {
            let target = target.ok_or_else(|| {
                runtime_error(
                    "TCP debug JSON transport requires expert targets for synthetic-ds4-layer",
                )
            })?;
            let addr = parse_expert_target(target)?;
            ds41rt_transport::debug_json_tcp_roundtrip(addr, request, Default::default())
                .await
                .map_err(runtime_error)
        }
        ApiTransport::VerbsHost => {
            let target = target.ok_or_else(|| {
                runtime_error(
                    "verbs-host transport requires expert targets for synthetic-ds4-layer",
                )
            })?;
            let addr = parse_expert_target(target)?;
            ds41rt_transport::verbs_host_protocol_v2_expert_request_roundtrip(
                addr,
                request,
                Default::default(),
            )
            .await
            .map_err(runtime_error)
        }
    }
}

pub(crate) fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

pub(crate) fn parse_expert_target(target: &str) -> Result<SocketAddr, ApiError> {
    let with_port = if target.contains(':') {
        target.to_owned()
    } else {
        format!("{target}:9100")
    };
    with_port
        .to_socket_addrs()
        .map_err(|err| {
            invalid_request(
                format!("expert target {with_port} could not be resolved: {err}"),
                Some("expert_hosts"),
            )
        })?
        .next()
        .ok_or_else(|| {
            invalid_request(
                format!("expert target {with_port} resolved to no addresses"),
                Some("expert_hosts"),
            )
        })
}

pub(crate) fn sum_partials(
    partials: &[Vec<f32>],
    expected_len: usize,
) -> Result<Vec<f32>, ApiError> {
    if partials.is_empty() {
        return Err(runtime_error("no expert partials were returned"));
    }
    let mut summed = vec![0.0_f32; expected_len];
    for partial in partials {
        if partial.len() != expected_len {
            return Err(runtime_error(format!(
                "partial length {} did not match expected size {}",
                partial.len(),
                expected_len
            )));
        }
        for (acc, value) in summed.iter_mut().zip(partial) {
            *acc += *value;
        }
    }
    Ok(summed)
}

#[cfg(test)]
mod tests;
