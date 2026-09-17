use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: ErrorBody,
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    message: String,
    #[serde(rename = "type")]
    error_type: &'static str,
    param: Option<String>,
    code: Option<String>,
}

#[derive(Debug)]
pub(crate) struct ApiError {
    pub(crate) status: StatusCode,
    pub(crate) message: String,
    pub(crate) param: Option<String>,
    pub(crate) code: Option<String>,
}

impl ApiError {
    pub(crate) fn into_response(self) -> Response {
        // Bound centrally: request-derived strings (model id, message role,
        // tool_call_id, ...) are interpolated into several ApiError messages
        // (request.rs unsupported-role, completion.rs unknown-model). Without
        // this bound a validly typed 100 KB string produces a ~100 KB response
        // body — the same unbounded-echo class as the JsonRejection fix
        // (upstream vLLM #49239).
        openai_error(
            self.status,
            bounded_error_detail(&self.message),
            self.param,
            self.code,
        )
    }
}

pub(crate) fn invalid_request(
    message: impl Into<String>,
    param: Option<impl Into<String>>,
) -> ApiError {
    ApiError {
        status: StatusCode::BAD_REQUEST,
        message: message.into(),
        param: param.map(Into::into),
        code: Some("invalid_request".to_owned()),
    }
}

pub(crate) fn runtime_error(message: impl std::fmt::Display) -> ApiError {
    ApiError {
        status: StatusCode::BAD_GATEWAY,
        message: message.to_string(),
        param: None,
        code: Some("backend_error".to_owned()),
    }
}

/// Bound an upstream parse/validation detail before it reaches the response
/// body or logs. Long attacker-controlled inputs (e.g. a 100 KB string in a
/// wrongly-typed field) must not be echoed back verbatim — the
/// unbounded-error-body class upstream vLLM fixed in #49239.
pub(crate) fn bounded_error_detail(detail: &str) -> String {
    const MAX_DETAIL_CHARS: usize = 512;
    let mut chars = detail.chars();
    let head: String = chars.by_ref().take(MAX_DETAIL_CHARS).collect();
    if chars.next().is_some() {
        format!("{head}... (truncated)")
    } else {
        head
    }
}

pub(crate) fn openai_error(
    status: StatusCode,
    message: String,
    param: Option<String>,
    code: Option<String>,
) -> Response {
    (
        status,
        Json(ErrorResponse {
            error: ErrorBody {
                message,
                error_type: "invalid_request_error",
                param,
                code,
            },
        }),
    )
        .into_response()
}
