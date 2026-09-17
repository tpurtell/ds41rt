use uuid::Uuid;

use std::path::Path;

use crate::backends::{
    real_ds4_full_completion, real_ds4_slice_completion, resolve_real_full_prompt_token_ids,
    synthetic_ds4_layer_completion, tiny_backend_completion,
};
use crate::constrained::{request_constraint, request_explicitly_requires_constraint};
use crate::metrics::CompletionMetrics;
use crate::request::{
    prompt_text, real_ds4_full_request_prompt_text, request_image_sources, request_max_tokens,
    request_sampling_params, rough_token_count, stop_strings, tool_calls_enabled, unix_timestamp,
    validate_request,
};
use crate::tooling::parse_ds4_tool_calls;
use crate::{
    ApiBackend, ApiError, ApiState, ApiTransport, AssistantMessage, ChatChoice,
    ChatCompletionRequest, ChatCompletionResponse, RealFullConstraint, ToolCall, Usage,
};
use std::sync::Arc;

#[derive(Debug)]
pub(crate) struct CompletionOutput {
    pub(crate) id: String,
    pub(crate) created: u64,
    pub(crate) model: String,
    pub(crate) content: Option<String>,
    pub(crate) reasoning_content: Option<String>,
    pub(crate) stream_chunks: Option<Vec<String>>,
    pub(crate) tool_calls: Option<Vec<ToolCall>>,
    pub(crate) finish_reason: String,
    pub(crate) usage: Usage,
    pub(crate) metrics: CompletionMetrics,
}

impl CompletionOutput {
    pub(crate) fn into_response_body(self) -> ChatCompletionResponse {
        ChatCompletionResponse {
            id: self.id,
            object: "chat.completion",
            created: self.created,
            model: self.model,
            choices: vec![ChatChoice {
                index: 0,
                message: AssistantMessage {
                    role: "assistant",
                    content: self.content,
                    reasoning_content: self.reasoning_content,
                    tool_calls: self.tool_calls,
                },
                finish_reason: self.finish_reason,
            }],
            usage: self.usage,
            metrics: self.metrics,
        }
    }
}

pub(crate) struct ValidatedCompletionRequest {
    pub(crate) request: ChatCompletionRequest,
    pub(crate) backend: ApiBackend,
    constraint: Option<Arc<RealFullConstraint>>,
}

#[cfg(test)]
pub(crate) async fn build_completion(
    state: &ApiState,
    request: ChatCompletionRequest,
) -> Result<CompletionOutput, ApiError> {
    let validated = validate_completion_request(state, request)?;
    build_validated_completion(state, validated).await
}

pub(crate) fn validate_completion_request(
    state: &ApiState,
    request: ChatCompletionRequest,
) -> Result<ValidatedCompletionRequest, ApiError> {
    validate_request(&request)?;
    let backend = selected_backend(state, &request)?;
    if backend != ApiBackend::RealDs4Full && request_explicitly_requires_constraint(&request) {
        return Err(crate::invalid_request(
            "constrained response formats and strict tool schemas require the real DS4 full backend",
            Some("model"),
        ));
    }
    let constraint = if backend == ApiBackend::RealDs4Full {
        request_constraint(&request)?
    } else {
        None
    };
    let image_sources = request_image_sources(&request)?;
    if !image_sources.is_empty() {
        return Err(crate::invalid_request(
            "image input is unsupported because DeepSeek V4 Flash and Pro are text-only models",
            Some("messages"),
        ));
    }
    Ok(ValidatedCompletionRequest {
        request,
        backend,
        constraint,
    })
}

pub(crate) async fn build_validated_completion(
    state: &ApiState,
    validated: ValidatedCompletionRequest,
) -> Result<CompletionOutput, ApiError> {
    let ValidatedCompletionRequest {
        request,
        backend,
        constraint,
    } = validated;
    let plain_prompt = prompt_text(&request.messages);
    let prompt = if backend == ApiBackend::RealDs4Full {
        real_ds4_full_request_prompt_text(&request)
    } else {
        plain_prompt
    };
    let max_tokens = request_max_tokens(&request);
    let tools_enabled = tool_calls_enabled(&request);
    let created = unix_timestamp();
    let id = format!("chatcmpl-{}", Uuid::new_v4());
    let resolved_prompt_token_ids = (backend == ApiBackend::RealDs4Full)
        .then(|| resolve_real_full_prompt_token_ids(state, &request, &prompt, backend))
        .flatten();
    let prompt_tokens = resolved_prompt_token_ids.as_ref().map_or_else(
        || prompt_token_count(state, backend, &prompt),
        |ids| ids.len(),
    );
    let transport_backend = transport_name(state.config.transport);

    let backend_completion = match backend {
        ApiBackend::Tiny => tiny_backend_completion(&prompt, prompt_tokens, max_tokens),
        ApiBackend::SyntheticDs4Layer => {
            synthetic_ds4_layer_completion(state, &prompt, prompt_tokens).await?
        }
        ApiBackend::RealDs4Slice => real_ds4_slice_completion(state).await?,
        ApiBackend::RealDs4Full => {
            real_ds4_full_completion(
                state,
                &prompt,
                prompt_tokens,
                resolved_prompt_token_ids,
                max_tokens,
                request.min_tokens.unwrap_or(0),
                request.ignore_eos.unwrap_or(false),
                request_sampling_params(&request),
                tools_enabled,
                constraint,
            )
            .await?
        }
    };
    let mut content = backend_completion.content;
    let reasoning_content = backend_completion.reasoning_content;
    let mut stream_chunks = backend_completion.stream_chunks;
    let mut completion_tokens = backend_completion.completion_tokens;
    let mut finish_reason = if completion_token_count(&content, completion_tokens) >= max_tokens {
        "length"
    } else {
        "stop"
    }
    .to_owned();
    // Stop selection follows vLLM's check_stop_strings semantics: the stop
    // string that *completes* earliest in the text wins (so the result matches
    // appending one token at a time under speculative decoding); ties are
    // broken by stop-list order.
    let mut matched_stop_start: Option<usize> = None;
    let mut matched_stop_end = usize::MAX;
    for stop in stop_strings(request.stop.as_ref())
        .iter()
        .filter(|stop| !stop.is_empty())
    {
        if let Some(idx) = content.find(stop) {
            let end = idx + stop.len();
            if end < matched_stop_end {
                matched_stop_start = Some(idx);
                matched_stop_end = end;
            }
        }
    }
    if let Some(stop_idx) = matched_stop_start {
        content.truncate(stop_idx);
        completion_tokens = None;
        stream_chunks = None;
        finish_reason = "stop".to_owned();
    }
    let completion_tokens = completion_token_count(&content, completion_tokens);
    let parsed_tools = tools_enabled.then(|| parse_ds4_tool_calls(&content));
    let (content, stream_chunks, tool_calls, finish_reason) = match parsed_tools {
        Some(parsed) if !parsed.tool_calls.is_empty() => (
            parsed.content,
            None,
            Some(parsed.tool_calls),
            "tool_calls".to_owned(),
        ),
        _ => (Some(content), stream_chunks, None, finish_reason),
    };
    let metrics = CompletionMetrics::from_backend(
        prompt_tokens,
        completion_tokens,
        backend_name(backend),
        transport_backend,
        backend_completion.metrics,
    );
    let usage = Usage::from_metrics(&metrics);
    Ok(CompletionOutput {
        id,
        created,
        model: request.model,
        content,
        reasoning_content,
        stream_chunks,
        tool_calls,
        finish_reason,
        usage,
        metrics,
    })
}

pub(crate) fn completion_token_count(
    content: &str,
    backend_completion_tokens: Option<usize>,
) -> usize {
    backend_completion_tokens.unwrap_or_else(|| rough_token_count(content))
}

pub(crate) fn selected_backend(
    state: &ApiState,
    request: &ChatCompletionRequest,
) -> Result<ApiBackend, ApiError> {
    let backend = if request.model == "ds41rt-tiny" {
        ApiBackend::Tiny
    } else if request.model == "ds41rt-synthetic-ds4-layer" {
        ApiBackend::SyntheticDs4Layer
    } else if request.model == format!("{}-slice", state.config.model_id) {
        ApiBackend::RealDs4Slice
    } else if request.model == format!("{}-full", state.config.model_id) {
        ApiBackend::RealDs4Full
    } else if request.model == state.config.model_id {
        state.config.backend
    } else {
        return Err(crate::invalid_request(
            format!(
                "model {} is not served by this DS41RT instance; query /v1/models for the exact identity",
                request.model
            ),
            Some("model"),
        ));
    };
    Ok(backend)
}

pub(crate) fn prompt_token_count(state: &ApiState, backend: ApiBackend, prompt: &str) -> usize {
    if backend != ApiBackend::RealDs4Full {
        return rough_token_count(prompt);
    }
    prompt_token_ids(state, backend, prompt)
        .map(|token_ids| token_ids.len())
        .unwrap_or_else(|| rough_token_count(prompt))
}

pub(crate) fn prompt_token_ids(
    state: &ApiState,
    backend: ApiBackend,
    prompt: &str,
) -> Option<Vec<usize>> {
    (backend == ApiBackend::RealDs4Full)
        .then_some(())
        .and_then(|_| state.config.real_full.as_ref())
        .and_then(|full| full.snapshot_path.as_deref())
        .and_then(|snapshot_path| {
            ds41rt_loader::encode_tokenizer_text(Path::new(snapshot_path), prompt, false).ok()
        })
        .map(|summary| {
            summary
                .token_ids
                .into_iter()
                .map(|token_id| token_id as usize)
                .collect()
        })
}

pub(crate) fn backend_name(backend: ApiBackend) -> &'static str {
    match backend {
        ApiBackend::Tiny => "tiny",
        ApiBackend::SyntheticDs4Layer => "synthetic-ds4-layer",
        ApiBackend::RealDs4Slice => "real-ds4-slice",
        ApiBackend::RealDs4Full => "real-ds4-full",
    }
}

pub(crate) fn transport_name(transport: ApiTransport) -> &'static str {
    transport.label()
}
