//! Upstream-ported SSE / streaming conformance tests.
//!
//! Ports the *streaming invariants* of three upstream suites onto ds41rt's
//! streaming surface (`crate::streaming` + `crate::completion`):
//!
//! - `vllm/tests/entrypoints/openai/responses/test_streaming_events.py`
//!   (7 tests: SSE event state machine, event types, delta splitting,
//!   lifecycle ordering). ds41rt's chat SSE surface is OpenAI-compatible
//!   rather than Responses-API-shaped, so the split_delta / state-machine
//!   invariants are asserted against ds41rt's equivalent property: every
//!   emitted chunk delta is *single-purpose* (exactly one of
//!   reasoning_content / content / tool_calls) and the chunk sequence is
//!   ordered reasoning -> content -> tool_calls -> finish.
//!   SKIPPED upstream tests (no ds41rt surface):
//!   - `test_browser_find_uses_responses_action_type` — ds41rt has no
//!     Responses API / browser-tool event surface at all.
//! - `vllm/tests/entrypoints/serve/utils/test_sse_keep_alive.py`
//!   (11 tests: SSE keep-alive wrapper semantics). ds41rt has *no*
//!   keep-alive wrapper (verified: no `keep_alive`/`keep-alive` symbol in
//!   ds41rt-api sources), so wrapper-semantics tests
//!   (`test_disabled_returns_same_generator`,
//!   `test_keep_alive_during_silent_start`,
//!   `test_no_keep_alive_when_streaming_is_fast`,
//!   `test_exception_propagates`,
//!   `test_upstream_cancelled_error_propagates`,
//!   `test_aclose_closes_upstream`,
//!   `test_client_disconnect_cancels_upstream`,
//!   `test_streaming_response_emits_comment`,
//!   `test_streaming_response_disconnect_closes_upstream`) have no surface
//!   to map onto and are intentionally not ported. The remaining mappable
//!   invariants — no data dropped/reordered by interleaving, empty input
//!   terminates promptly, and SSE framing purity (no stray comment lines) —
//!   are ported below.
//! - `llama.cpp/tools/server/tests/unit/test_chat_completion.py` — model-gated;
//!   only the streaming/SSE assertion patterns are ported as semantic
//!   references: chunk sequence, role-first ordering, stable stream id,
//!   `[DONE]` terminator, and usage-in-final-chunk under
//!   `stream_options.include_usage`.

use axum::body::to_bytes;
use axum::http::header::CONTENT_TYPE;
use serde_json::Value;

use super::*;
use crate::metrics::CompletionMetrics;
use crate::streaming::chat_stream_response;

/// Build a minimal `CompletionOutput` for streaming, mirroring the shape the
/// tiny backend produces but with explicit control over every field.
fn stream_output(
    content: Option<&str>,
    reasoning_content: Option<&str>,
    tool_calls: Option<Vec<ToolCall>>,
    stream_chunks: Option<Vec<String>>,
    prompt_tokens: usize,
    output_tokens: usize,
) -> CompletionOutput {
    let metrics = CompletionMetrics {
        queue_ms: 0.0,
        cache_load_ms: 0.0,
        prefill_ms: 1.0,
        time_to_first_token_ms: 2.0,
        decode_ms: 3.0,
        output_tokens,
        prompt_tokens,
        cached_prompt_tokens: 0,
        reasoning_tokens: 0,
        prefill_tokens_per_sec: None,
        transport_backend: "inproc",
        backend_mode: "tiny",
        prefill_chunk_count: 1,
        layerwave_prefill_rows: 1,
        layerwave_decode_rows: 1,
        real_full: None,
    };
    CompletionOutput {
        id: "chatcmpl-upstream-port".to_owned(),
        created: 1_700_000_000,
        model: "ds41rt-tiny".to_owned(),
        content: content.map(str::to_owned),
        reasoning_content: reasoning_content.map(str::to_owned),
        stream_chunks,
        tool_calls,
        finish_reason: "stop".to_owned(),
        usage: Usage::from_metrics(&metrics),
        metrics,
    }
}

fn sample_tool_call(name: &str, arguments: &str) -> ToolCall {
    ToolCall {
        id: format!("call_{name}"),
        tool_type: "function".to_owned(),
        function: ToolCallFunction {
            name: name.to_owned(),
            arguments: arguments.to_owned(),
        },
    }
}

/// Collect the SSE body of a `chat_stream_response` as `(event_name, data)`
/// pairs plus the raw body text. `event_name` is `None` for the default
/// (message) event, matching how axum frames unnamed events.
async fn collect_sse(
    output: CompletionOutput,
    include_usage: bool,
) -> (String, Vec<(Option<String>, String)>) {
    let response = chat_stream_response(output, include_usage);
    assert!(
        response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.starts_with("text/event-stream")),
        "stream response must use text/event-stream",
    );
    let bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("collecting SSE body");
    let text = String::from_utf8(bytes.to_vec()).expect("SSE body is utf-8");
    let mut events = Vec::new();
    for block in text.split("\n\n") {
        if block.is_empty() {
            continue;
        }
        let mut event_name = None;
        let mut data = String::new();
        for line in block.split('\n') {
            if let Some(rest) = line.strip_prefix("data: ") {
                if !data.is_empty() {
                    data.push('\n');
                }
                data.push_str(rest);
            } else if let Some(rest) = line.strip_prefix("event: ") {
                event_name = Some(rest.to_owned());
            } else {
                panic!("unexpected SSE line {line:?} in block {block:?}");
            }
        }
        events.push((event_name, data));
    }
    (text, events)
}

fn data_json(events: &[(Option<String>, String)], index: usize) -> Value {
    serde_json::from_str(&events[index].1).unwrap_or_else(|err| {
        panic!("event {index} is not JSON: {err}; data={:?}", events[index].1)
    })
}

/// Port of `vllm/.../responses/test_streaming_events.py::TestSplitDelta::test_all_three_fields`.
///
/// A delta carrying reasoning + content + tool calls must surface as three
/// *separate* single-purpose events upstream; ds41rt's equivalent guarantee
/// is that no emitted chunk delta is compound: every non-terminal chunk has
/// exactly one of `reasoning_content` / `content` / `tool_calls` set.
#[tokio::test]
async fn compound_delta_surfaces_as_single_purpose_chunks() {
    let output = stream_output(
        Some("answer"),
        Some("think"),
        Some(vec![sample_tool_call("f", "{}")]),
        Some(vec!["answer".to_owned()]),
        4,
        2,
    );
    let (_, events) = collect_sse(output, false).await;
    let finish_index = events.len() - 2; // last is [DONE]
    for (index, event) in events[..finish_index].iter().enumerate() {
        let chunk: Value = serde_json::from_str(&event.1).unwrap();
        let delta = &chunk["choices"][0]["delta"];
        if delta.get("role").is_some() {
            // Role-first chunk (llama.cpp role-first invariant): role only.
            assert!(
                delta.get("content").is_none()
                    && delta.get("reasoning_content").is_none()
                    && delta.get("tool_calls").is_none(),
                "role chunk must carry role alone: {delta}"
            );
            continue;
        }
        let set = [
            delta.get("reasoning_content").is_some(),
            delta.get("content").is_some(),
            delta.get("tool_calls").is_some(),
        ]
        .into_iter()
        .filter(|present| *present)
        .count();
        assert_eq!(
            set, 1,
            "chunk {index} must be single-purpose, got delta {delta}"
        );
    }
}

/// Port of `vllm/.../responses/test_streaming_events.py::TestSplitDelta::test_tool_calls_grouped_by_index`.
///
/// Tool calls at different indices must not be merged into one delta. ds41rt
/// emits one chunk per tool call carrying its own `index`.
#[tokio::test]
async fn tool_calls_at_distinct_indices_stay_in_separate_chunks() {
    let output = stream_output(
        None,
        None,
        Some(vec![
            sample_tool_call("f1", "{\"a\":1}"),
            sample_tool_call("f2", "{\"b\":2}"),
        ]),
        None,
        4,
        2,
    );
    let (_, events) = collect_sse(output, false).await;
    let tool_chunks: Vec<Value> = events
        .iter()
        .filter(|(_, data)| data != "[DONE]")
        .map(|(_, data)| serde_json::from_str::<Value>(data).unwrap())
        .filter(|chunk| chunk["choices"][0]["delta"].get("tool_calls").is_some())
        .collect();
    assert_eq!(tool_chunks.len(), 2, "expected one chunk per tool call");
    let indices: Vec<usize> = tool_chunks
        .iter()
        .map(|chunk| chunk["choices"][0]["delta"]["tool_calls"][0]["index"].as_u64().unwrap() as usize)
        .collect();
    assert_eq!(indices, vec![0, 1]);
}

/// Port of `vllm/.../responses/test_streaming_events.py::TestProcessorCompoundDeltas::test_all_three_states`.
///
/// Lifecycle ordering: reasoning deltas precede content deltas precede
/// tool-call deltas in the emitted chunk sequence.
#[tokio::test]
async fn reasoning_then_content_then_tool_call_ordering() {
    let output = stream_output(
        Some("c"),
        Some("r"),
        Some(vec![sample_tool_call("f", "{}")]),
        Some(vec!["c".to_owned()]),
        4,
        2,
    );
    let (_, events) = collect_sse(output, false).await;
    let kinds: Vec<&str> = events
        .iter()
        .filter(|(_, data)| data != "[DONE]")
        .map(|(_, data)| serde_json::from_str::<Value>(data).unwrap())
        .map(|chunk: Value| {
            let delta = &chunk["choices"][0]["delta"];
            if delta.get("reasoning_content").is_some() {
                "reasoning"
            } else if delta.get("content").is_some() {
                "content"
            } else if delta.get("tool_calls").is_some() {
                "tool_call"
            } else if chunk["choices"][0]["finish_reason"].is_null() {
                "role"
            } else {
                "finish"
            }
        })
        .collect();
    let reasoning_idx = kinds
        .iter()
        .position(|kind| *kind == "reasoning")
        .expect("reasoning chunk present");
    let content_idx = kinds
        .iter()
        .position(|kind| *kind == "content")
        .expect("content chunk present");
    let tool_call_idx = kinds
        .iter()
        .position(|kind| *kind == "tool_call")
        .expect("tool_call chunk present");
    assert!(
        reasoning_idx < content_idx && content_idx < tool_call_idx,
        "ordering broken: {kinds:?}"
    );
}

/// Port of `vllm/.../responses/test_streaming_events.py::TestProcessorCompoundDeltas::test_parallel_tool_calls`.
///
/// Two parallel tool calls produce two distinct tool-call chunks, each with a
/// single entry (never one chunk with two entries).
#[tokio::test]
async fn parallel_tool_calls_each_emit_own_chunk() {
    let output = stream_output(
        None,
        None,
        Some(vec![
            sample_tool_call("f1", "{\"a\":1}"),
            sample_tool_call("f2", "{\"b\":2}"),
        ]),
        None,
        4,
        2,
    );
    let (_, events) = collect_sse(output, false).await;
    let tool_chunks: Vec<Value> = events
        .iter()
        .filter(|(_, data)| data != "[DONE]")
        .map(|(_, data)| serde_json::from_str::<Value>(data).unwrap())
        .filter(|chunk| chunk["choices"][0]["delta"].get("tool_calls").is_some())
        .collect();
    assert_eq!(tool_chunks.len(), 2);
    for chunk in &tool_chunks {
        assert_eq!(chunk["choices"][0]["delta"]["tool_calls"].as_array().unwrap().len(), 1);
        assert_eq!(chunk["choices"][0]["delta"]["tool_calls"][0]["type"], "function");
        assert!(
            chunk["choices"][0]["delta"]["tool_calls"][0]["id"]
                .as_str()
                .unwrap()
                .starts_with("call_")
        );
    }
}

/// Port of `vllm/.../responses/test_streaming_events.py::TestProcessorCompoundDeltas::test_split_name_and_args_same_index`.
///
/// Regression: parsers emit name and arguments as separate same-index deltas;
/// the streamed result must carry the *full* function payload (name +
/// arguments together, plus id and type) in a single chunk entry.
#[tokio::test]
async fn tool_call_chunk_carries_full_function_payload() {
    let output = stream_output(
        None,
        None,
        Some(vec![sample_tool_call("get_weather", "{\"city\":\"SF\"}")]),
        None,
        4,
        2,
    );
    let (_, events) = collect_sse(output, false).await;
    let tool_chunk: Value = events
        .iter()
        .filter(|(_, data)| data != "[DONE]")
        .map(|(_, data)| serde_json::from_str::<Value>(data).unwrap())
        .find(|chunk: &Value| chunk["choices"][0]["delta"].get("tool_calls").is_some())
        .expect("one tool_call chunk");
    let entry = &tool_chunk["choices"][0]["delta"]["tool_calls"][0];
    assert_eq!(entry["function"]["name"], "get_weather");
    assert_eq!(entry["function"]["arguments"], "{\"city\":\"SF\"}");
    assert_eq!(entry["type"], "function");
    assert!(entry["id"].as_str().is_some_and(|id| !id.is_empty()));
}

/// Port of `vllm/.../responses/test_streaming_events.py::TestProcessorCompoundDeltas::test_reasoning_to_content_transition`.
///
/// A stream that begins in reasoning and transitions to content must emit
/// both delta types, reasoning strictly before content (the reasoning state
/// closes before the content state opens).
#[tokio::test]
async fn reasoning_to_content_transition_emits_both_delta_types_in_order() {
    let output = stream_output(
        Some("answer"),
        Some("think it through"),
        None,
        Some(vec!["answer".to_owned()]),
        4,
        2,
    );
    let (_, events) = collect_sse(output, false).await;
    let kinds: Vec<&str> = events
        .iter()
        .filter(|(_, data)| data != "[DONE]")
        .map(|(_, data)| serde_json::from_str::<Value>(data).unwrap())
        .map(|chunk: Value| {
            let delta = &chunk["choices"][0]["delta"];
            if delta.get("reasoning_content").is_some() {
                "reasoning"
            } else if delta.get("content").is_some() {
                "content"
            } else {
                "other"
            }
        })
        .collect();
    assert!(
        kinds.contains(&"reasoning") && kinds.contains(&"content"),
        "both delta types required: {kinds:?}"
    );
    let reasoning_idx = kinds.iter().position(|kind| *kind == "reasoning").unwrap();
    let content_idx = kinds.iter().position(|kind| *kind == "content").unwrap();
    assert!(reasoning_idx < content_idx, "reasoning must precede content: {kinds:?}");
}

/// Port of the framing invariant behind
/// `vllm/.../test_sse_keep_alive.py::test_keep_alive_between_chunks_does_not_drop_data`
/// and `test_no_keep_alive_when_streaming_is_fast`, for a server with no
/// keep-alive wrapper: explicitly provided stream chunks must be forwarded
/// verbatim, in order, with none dropped, none duplicated, none merged, and
/// no non-data framing lines injected between them.
#[tokio::test]
async fn provided_stream_chunks_forwarded_verbatim_in_order() {
    let chunks = vec!["alpha".to_owned(), " beta".to_owned(), " gamma".to_owned()];
    let output = stream_output(Some("ignored"), None, None, Some(chunks.clone()), 4, 3);
    let (text, events) = collect_sse(output, false).await;
    let forwarded: Vec<String> = events
        .iter()
        .filter(|(_, data)| data != "[DONE]")
        .map(|(_, data)| serde_json::from_str::<Value>(data).unwrap())
        .filter(|chunk: &Value| chunk["choices"][0]["delta"].get("content").is_some())
        .map(|chunk| chunk["choices"][0]["delta"]["content"].as_str().unwrap().to_owned())
        .collect();
    assert_eq!(forwarded, chunks, "chunk data must pass through untouched");
    // Framing purity: every non-empty line is a data line (no keep-alive
    // comments, no stray blank frames mid-stream beyond event separators).
    for line in text.lines() {
        assert!(
            line.is_empty() || line.starts_with("data: "),
            "unexpected SSE line {line:?}"
        );
    }
}

/// Port of `vllm/.../test_sse_keep_alive.py::test_empty_generator_returns_immediately`
/// onto ds41rt's non-wrapper surface: an empty completion must still yield a
/// well-formed minimal stream (role chunk, finish chunk, `[DONE]`) — the
/// stream terminates promptly rather than hanging or emitting nothing.
#[tokio::test]
async fn empty_content_still_yields_role_finish_and_done() {
    let output = stream_output(Some(""), None, None, Some(Vec::new()), 4, 0);
    let (_, events) = collect_sse(output, false).await;
    assert_eq!(events.len(), 3, "expected role + finish + [DONE], got {events:?}");
    let role: Value = serde_json::from_str(&events[0].1).unwrap();
    assert_eq!(role["choices"][0]["delta"]["role"], "assistant");
    let finish: Value = serde_json::from_str(&events[1].1).unwrap();
    assert_eq!(finish["choices"][0]["finish_reason"], "stop");
    assert_eq!(events[2].1, "[DONE]");
}

/// Port of the streaming assertion pattern in
/// `llama.cpp/.../unit/test_chat_completion.py::test_chat_completion_stream`:
/// the first chunk of a `stream=true` response carries
/// `delta.role == "assistant"` with no content and no finish_reason, and
/// every chunk in the stream shares the same completion id.
#[tokio::test]
async fn first_chunk_carries_role_only_and_stream_shares_one_id() {
    let output = stream_output(
        Some("answer"),
        None,
        None,
        Some(vec!["answer".to_owned()]),
        4,
        1,
    );
    let (_, events) = collect_sse(output, false).await;
    let first: Value = data_json(&events, 0);
    assert_eq!(first["choices"][0]["delta"]["role"], "assistant");
    assert!(
        first["choices"][0]["delta"].get("content").is_none(),
        "role-first chunk must not carry content: {}",
        first["choices"][0]["delta"]
    );
    assert!(first["choices"][0]["finish_reason"].is_null());
    let id = first["id"].as_str().unwrap().to_owned();
    assert!(id.starts_with("chatcmpl-"));
    for (index, (_, data)) in events.iter().enumerate() {
        if data == "[DONE]" {
            continue;
        }
        let chunk: Value = serde_json::from_str(data).unwrap();
        assert_eq!(chunk["id"].as_str().unwrap(), id, "chunk {index} changed id");
        assert_eq!(chunk["object"], "chat.completion.chunk");
        if index > 0 {
            assert!(
                chunk["choices"][0]["delta"].get("role").is_none(),
                "role must appear exactly once (chunk {index}): {}",
                chunk["choices"][0]["delta"]
            );
        }
    }
}

/// Port of the accumulation + terminator assertions in
/// `llama.cpp/.../unit/test_chat_completion.py::test_chat_completion_stream`:
/// concatenated content deltas reassemble the completion text, the finish
/// chunk has an empty delta and the expected finish_reason, and `[DONE]` is
/// the final SSE event.
#[tokio::test]
async fn content_reassembles_finish_chunk_is_empty_and_done_terminates() {
    let output = stream_output(
        Some("ignored"),
        None,
        None,
        Some(vec!["hello".to_owned(), " from".to_owned(), " ds41rt".to_owned()]),
        4,
        3,
    );
    let (_, events) = collect_sse(output, false).await;
    assert_eq!(events.last().unwrap().1, "[DONE]", "[DONE] must be the final event");
    let mut content = String::new();
    let mut finish_chunk: Option<Value> = None;
    for (_, data) in &events[..events.len() - 1] {
        let chunk: Value = serde_json::from_str(data).unwrap();
        let choice = &chunk["choices"][0];
        if let Some(piece) = choice["delta"].get("content") {
            content.push_str(piece.as_str().unwrap());
        }
        if !choice["finish_reason"].is_null() {
            assert_eq!(choice["finish_reason"], "stop");
            assert!(
                choice["delta"].get("content").is_none()
                    && choice["delta"].get("reasoning_content").is_none()
                    && choice["delta"].get("tool_calls").is_none(),
                "finish chunk delta must be empty: {}",
                choice["delta"]
            );
            assert!(finish_chunk.is_none(), "exactly one finish chunk");
            finish_chunk = Some(chunk.clone());
        }
    }
    assert_eq!(content, "hello from ds41rt");
    assert!(finish_chunk.is_some(), "stream must contain a finish chunk");
}

/// Port of the `stream_options.include_usage` usage-in-final-chunk pattern
/// from `llama.cpp/.../unit/test_chat_completion.py::test_chat_completion_with_timings_per_token`
/// (the usage-bearing chunk has empty choices and arrives after content):
/// when `include_usage` is set, ds41rt must emit a usage chunk with empty
/// choices after the finish chunk and before `[DONE]`, carrying correct
/// prompt/completion token counts.
#[tokio::test]
async fn include_usage_emits_usage_chunk_after_finish_before_done() {
    let output = stream_output(
        Some("answer"),
        None,
        None,
        Some(vec!["answer".to_owned()]),
        11,
        7,
    );
    let (_, events) = collect_sse(output, true).await;
    assert_eq!(events.last().unwrap().1, "[DONE]");
    let finish_index = events
        .iter()
        .position(|(_, data)| {
            data != "[DONE]"
                && serde_json::from_str::<Value>(data)
                    .map(|chunk: Value| !chunk["choices"][0]["finish_reason"].is_null())
                    .unwrap_or(false)
        })
        .expect("finish chunk present");
    let usage_index = events.len() - 2;
    assert!(
        finish_index < usage_index,
        "usage chunk must follow the finish chunk: {events:?}"
    );
    let usage_chunk: Value = serde_json::from_str(&events[usage_index].1).unwrap();
    assert_eq!(usage_chunk["choices"].as_array().unwrap().len(), 0, "usage chunk choices must be empty");
    assert_eq!(usage_chunk["usage"]["prompt_tokens"], 11);
    assert_eq!(usage_chunk["usage"]["completion_tokens"], 7);
    assert_eq!(usage_chunk["usage"]["total_tokens"], 18);
}

/// Inverse of the upstream `stream_options.include_usage` pattern
/// (`llama.cpp/.../unit/test_chat_completion.py::test_chat_completion_stream`
/// runs without stream_options): without `include_usage`, no chunk may
/// carry a `usage` field — usage is omitted from the stream entirely.
#[tokio::test]
async fn without_include_usage_no_chunk_carries_usage() {
    let output = stream_output(
        Some("answer"),
        None,
        None,
        Some(vec!["answer".to_owned()]),
        11,
        7,
    );
    let (_, events) = collect_sse(output, false).await;
    for (index, (_, data)) in events.iter().enumerate() {
        if data == "[DONE]" {
            continue;
        }
        let chunk: Value = serde_json::from_str(data).unwrap();
        assert!(
            chunk.get("usage").is_none(),
            "chunk {index} must not carry usage without include_usage: {chunk}"
        );
    }
}
