//! Upstream-ported tool-call semantics conformance tests (ds41rt component C12).
//!
//! Ports the tool-call invariant classes of four upstream suites onto ds41rt's
//! tool surface (`crate::tooling::{parse_ds4_tool_calls, Ds4ToolCallStreamParser}`,
//! `crate::completion` tool handling, and the response/stream payloads in
//! `crate::openai` / `crate::streaming`):
//!
//! - `vllm/tests/entrypoints/openai/test_tool_choice_content_none.py`
//!   (6 tests: named `tool_choice` with `content=None`; empty `tool_calls`
//!   omitted from the response *and* stream-chunk payloads). ds41rt has no
//!   `_extract_tool_calls(content=None, ...)` hook; the invariants map onto
//!   `parse_ds4_tool_calls` on empty/marker-free output plus the serde payload
//!   shapes of `AssistantMessage` and the stream delta.
//! - `vllm/tests/entrypoints/openai/test_tool_calls_serialization.py`
//!   (5 tests: `tool_calls` Iterable materialisation after `model_dump_json`).
//!   This was a Pydantic-v2 one-shot-iterator bug; serde has no lazy
//!   iterators, so the invariant maps to "deserialize -> serialize
//!   (model_dump_json analog) -> re-read yields the identical, un-consumed
//!   `tool_calls`", plus the OpenAI `arguments` shape coercions.
//! - `vllm/tests/tool_parsers/test_deepseekv4_tool_parser.py` (11 tests) and
//!   `vllm/tests/tool_parsers/test_deepseekv32_tool_parser.py` (48 tests).
//!   Only the invariant *classes* that map onto ds41rt's schema-free DSML
//!   parser are ported: no-tag-leak in streamed args, marker-split
//!   robustness, false partial markers, malformed-invoke recovery,
//!   typed/zero-arg/unicode parameter values, unique call ids. vllm-only
//!   classes (schema-driven type coercion, xgrammar structural tags,
//!   tokenizer interop) have no ds41rt surface and are noted per test.

use super::*;
use crate::streaming::chat_stream_response;
use crate::tooling::{parse_ds4_tool_calls, Ds4ToolCallStreamParser, Ds4ToolStreamDelta};

// ---------------------------------------------------------------------------
// DSML fixtures (same shape as the upstream `build_tool_call` helpers)
// ---------------------------------------------------------------------------

const TC_START: &str = "<｜DSML｜tool_calls>";
const TC_END: &str = "</｜DSML｜tool_calls>";
const INV_START: &str = "<｜DSML｜invoke name=\"";
const INV_END: &str = "</｜DSML｜invoke>";
const PARAM_START: &str = "<｜DSML｜parameter name=\"";
const PARAM_END: &str = "</｜DSML｜parameter>";

fn dsml_tool_call(func_name: &str, params: &[(&str, &str, bool)]) -> String {
    let param_strs = params
        .iter()
        .map(|(name, value, as_string)| {
            format!(
                "{PARAM_START}{name}\" string=\"{}\">{value}{PARAM_END}\n",
                if *as_string { "true" } else { "false" }
            )
        })
        .collect::<String>();
    format!("{TC_START}\n{INV_START}{func_name}\">\n{param_strs}{INV_END}\n{TC_END}")
}

/// ds41rt's *stream* parser anchors on `"\n\n<｜DSML｜tool_calls"` (the
/// marker always arrives preceded by a blank line in real decode output —
/// see `tooling.rs::DS4_STREAM_TOOL_CALLS_START`), unlike the non-streaming
/// parser and the vllm fixtures, which match the bare marker.
fn streamed_dsml_tool_call(func_name: &str, params: &[(&str, &str, bool)]) -> String {
    format!("\n\n{}", dsml_tool_call(func_name, params))
}

/// Drive a `Ds4ToolCallStreamParser` over `text` in `chunk_size` byte chunks
/// (snapped to char boundaries), then `finish()`, returning all deltas.
fn stream_in_chunks(text: &str, chunk_size: usize) -> Vec<Ds4ToolStreamDelta> {
    let mut parser = Ds4ToolCallStreamParser::new();
    let mut deltas = Vec::new();
    let mut start = 0;
    while start < text.len() {
        let mut end = (start + chunk_size).min(text.len());
        while end < text.len() && !text.is_char_boundary(end) {
            end += 1;
        }
        deltas.extend(parser.push(&text[start..end]));
        start = end;
    }
    deltas.extend(parser.finish());
    deltas
}

fn streamed_content(deltas: &[Ds4ToolStreamDelta]) -> String {
    deltas
        .iter()
        .filter_map(|delta| match delta {
            Ds4ToolStreamDelta::Content(chunk) => Some(chunk.clone()),
            _ => None,
        })
        .collect()
}

/// Reconstruct the per-index argument JSON fragments exactly like an
/// OpenAI-client reassembly of argument deltas.
fn streamed_arguments(deltas: &[Ds4ToolStreamDelta], index: usize) -> String {
    let mut arguments = String::new();
    for delta in deltas {
        if let Ds4ToolStreamDelta::ToolCall {
            index: tool_index,
            arguments: Some(fragment),
            ..
        } = delta
        {
            if *tool_index == index {
                arguments.push_str(fragment);
            }
        }
    }
    arguments
}

fn streamed_tool_names(deltas: &[Ds4ToolStreamDelta]) -> Vec<String> {
    let mut names = Vec::new();
    for delta in deltas {
        if let Ds4ToolStreamDelta::ToolCall {
            name: Some(name), ..
        } = delta
        {
            names.push(name.clone());
        }
    }
    names
}

// ---------------------------------------------------------------------------
// test_tool_choice_content_none.py: named tool_choice with content=None
// ---------------------------------------------------------------------------

/// Port of
/// `test_tool_choice_content_none.py::test_chat_completion_named_tool_choice_with_none_content`
/// (and its Responses-API sibling): a named `tool_choice` with model output
/// that carries no tool markers must yield *no* tool calls and leave the
/// content untouched. ds41rt has no `content=None` parser hook, so the
/// invariant is exercised end-to-end through `build_completion` with a named
/// `ToolChoice::Specific` and marker-free mock model output.
#[tokio::test]
async fn named_tool_choice_with_marker_free_output_yields_no_tool_calls() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let mut state = test_state(ApiBackend::RealDs4Full, ApiTransport::Inproc);
    state.config.real_full = Some(blocked_real_full_info());
    state.config.real_full_executor = Some(Arc::new(StepSamplingRealFullExecutor {
        requests: Arc::clone(&requests),
        base: ready_runtime_sample_real_full_info(42, "All done."),
        tokens: vec![(42, "All done.".to_owned())],
    }));

    let mut request = base_request("Use the lookup tool.");
    request.model = format!("{}-full", DEFAULT_MODEL_ID);
    request.max_tokens = Some(1);
    request.tools = Some(vec![lookup_tool()]);
    request.tool_choice = Some(ToolChoice::Specific {
        tool_type: "function".to_owned(),
        function: ToolChoiceFunction {
            name: "lookup".to_owned(),
        },
    });
    let output = build_completion(&state, request).await.unwrap();

    assert_eq!(output.tool_calls, None, "no tool markers -> no tool calls");
    assert_eq!(output.content.as_deref(), Some("All done."));
    assert_ne!(output.finish_reason, "tool_calls");
    let body = serde_json::to_value(output.into_response_body()).unwrap();
    let message = &body["choices"][0]["message"];
    assert!(
        message.get("tool_calls").is_none(),
        "empty tool_calls must be omitted from the response payload"
    );
}

/// Port of the payload-omission half of
/// `test_tool_choice_content_none.py::test_chat_completion_response_omits_empty_tool_calls_payload`
/// at the route level, with a model that produced *no* content at all (the
/// upstream `content=None` case): the serialized message must contain no
/// `tool_calls` key.
#[tokio::test]
async fn named_tool_choice_with_empty_model_output_omits_tool_calls_payload() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let mut state = test_state(ApiBackend::RealDs4Full, ApiTransport::Inproc);
    state.config.real_full = Some(blocked_real_full_info());
    state.config.real_full_executor = Some(Arc::new(StepSamplingRealFullExecutor {
        requests: Arc::clone(&requests),
        base: ready_runtime_sample_real_full_info(42, ""),
        tokens: vec![(42, String::new())],
    }));

    let (status, body) = request_json_with_config(
        state.config,
        Method::POST,
        "/v1/chat/completions",
        Some(json!({
            "model": format!("{}-full", DEFAULT_MODEL_ID),
            "messages": [{"role": "user", "content": "Use lookup."}],
            "tools": [{
                "type": "function",
                "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}}
            }],
            "tool_choice": {"type": "function", "function": {"name": "lookup"}},
            "max_tokens": 1
        })),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    let message = &body["choices"][0]["message"];
    assert!(
        message.get("tool_calls").is_none(),
        "empty tool_calls must be omitted from the response payload"
    );
    // MAPPING NOTE: upstream expects `content: null` here (the vllm parser
    // hooks receive content=None). ds41rt's real-full backend substitutes
    // placeholder text (`ds41rt-token:<id>`, see tests.rs:3561) for an empty
    // sampled token, so the response carries that placeholder instead of
    // null. The payload-omission invariant above is the part that maps.
    assert_eq!(message["content"], "ds41rt-token:42");
}

/// Port of
/// `test_tool_choice_content_none.py::test_chat_completion_response_keeps_non_empty_tool_calls_payload`
/// at the unit level: a response carrying tool calls must serialize the full
/// `tool_calls` payload (id, type, function.name, function.arguments).
#[test]
fn response_payload_keeps_non_empty_tool_calls() {
    let output = response_output(None, Some(vec![sample_tool_call(
        "get_weather",
        r#"{"city": "Beijing"}"#,
    )]));
    let body = serde_json::to_value(output.into_response_body()).unwrap();
    let tool_calls = body["choices"][0]["message"]["tool_calls"]
        .as_array()
        .expect("non-empty tool_calls must be serialized");
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0]["id"], "call_get_weather");
    assert_eq!(tool_calls[0]["type"], "function");
    assert_eq!(tool_calls[0]["function"]["name"], "get_weather");
    assert_eq!(
        tool_calls[0]["function"]["arguments"],
        r#"{"city": "Beijing"}"#
    );
}

// ---------------------------------------------------------------------------
// test_tool_choice_content_none.py: stream-chunk payload omission
// ---------------------------------------------------------------------------

fn response_output(content: Option<&str>, tool_calls: Option<Vec<ToolCall>>) -> CompletionOutput {
    let metrics = CompletionMetrics {
        queue_ms: 0.0,
        cache_load_ms: 0.0,
        prefill_ms: 1.0,
        time_to_first_token_ms: 2.0,
        decode_ms: 3.0,
        output_tokens: 1,
        prompt_tokens: 1,
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
        id: "chatcmpl-upstream-tool".to_owned(),
        created: 1_700_000_000,
        model: "ds41rt-tiny".to_owned(),
        content: content.map(str::to_owned),
        reasoning_content: None,
        stream_chunks: None,
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

async fn collect_stream_body(output: CompletionOutput) -> String {
    let response = chat_stream_response(output, false);
    let bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("collecting SSE body");
    String::from_utf8(bytes.to_vec()).expect("SSE body is utf-8")
}

/// Port of
/// `test_tool_choice_content_none.py::test_chat_completion_stream_response_omits_empty_tool_calls_payload`:
/// stream chunks for a completion without tool calls must not contain a
/// `tool_calls` key at all (no empty arrays either).
#[tokio::test]
async fn stream_chunks_omit_empty_tool_calls_payload() {
    let body = collect_stream_body(response_output(Some("done"), None)).await;
    for payload in body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|payload| *payload != "[DONE]")
    {
        let frame: Value = serde_json::from_str(payload).unwrap();
        let delta = &frame["choices"][0]["delta"];
        assert!(
            delta.get("tool_calls").is_none(),
            "stream delta must omit tool_calls when none exist: {payload}"
        );
    }
}

/// Port of
/// `test_tool_choice_content_none.py::test_chat_completion_stream_response_keeps_non_empty_tool_calls_payload`:
/// stream chunks for a completion with tool calls must carry the full delta
/// (index, id, type, function.name, function.arguments) and nothing else.
#[tokio::test]
async fn stream_chunks_keep_non_empty_tool_calls_payload() {
    let body = collect_stream_body(response_output(None, Some(vec![sample_tool_call(
        "get_weather",
        r#"{"city": "Beijing"}"#,
    )])))
    .await;
    let tool_frames = body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|payload| *payload != "[DONE]")
        .map(|payload| serde_json::from_str::<Value>(payload).unwrap())
        .filter(|frame| frame["choices"][0]["delta"].get("tool_calls").is_some())
        .collect::<Vec<_>>();
    assert_eq!(tool_frames.len(), 1, "exactly one tool_calls delta frame");
    let tool_call = &tool_frames[0]["choices"][0]["delta"]["tool_calls"][0];
    assert_eq!(tool_call["index"], 0);
    assert_eq!(tool_call["id"], "call_get_weather");
    assert_eq!(tool_call["type"], "function");
    assert_eq!(tool_call["function"]["name"], "get_weather");
    assert_eq!(
        tool_call["function"]["arguments"],
        r#"{"city": "Beijing"}"#
    );
    // The tool delta must be single-purpose (no content in the same delta).
    let delta = &tool_frames[0]["choices"][0]["delta"];
    assert!(delta.get("content").is_none());
}

// ---------------------------------------------------------------------------
// test_tool_calls_serialization.py: tool_calls materialisation / round-trips
// ---------------------------------------------------------------------------

fn tool_call_json(id: &str, name: &str, arguments: &str) -> Value {
    json!({
        "id": id,
        "type": "function",
        "function": {"name": name, "arguments": arguments}
    })
}

fn assistant_message_with_tool_calls(tool_calls: Value) -> Value {
    json!({
        "role": "assistant",
        "content": null,
        "tool_calls": tool_calls
    })
}

/// Port of
/// `test_tool_calls_serialization.py::test_tool_calls_list_preserved_after_model_dump`:
/// serializing the request (`model_dump_json` analog) must not consume or
/// mutate the assistant message's `tool_calls`; a re-read afterwards yields
/// the identical calls. (The upstream bug was a Pydantic one-shot iterator;
/// serde parses eagerly into `Vec<ToolCall>`, which this pins down.)
#[test]
fn tool_calls_preserved_after_serialization_round_trip() {
    let message = assistant_message_with_tool_calls(json!([tool_call_json(
        "call_abc123",
        "get_weather",
        r#"{"city": "Paris"}"#
    )]));
    let parsed: ChatMessage = serde_json::from_value(message).unwrap();
    let tool_calls = parsed.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].id, "call_abc123");
    assert_eq!(tool_calls[0].function.name, "get_weather");

    // model_dump_json analog, twice: must be byte-identical and lossless.
    let first_dump = serde_json::to_string(&parsed.tool_calls).unwrap();
    let second_dump = serde_json::to_string(&parsed.tool_calls).unwrap();
    assert_eq!(first_dump, second_dump, "serialization must not consume tool_calls");
    let reread: Vec<ToolCall> = serde_json::from_str(&second_dump).unwrap();
    assert_eq!(reread, *tool_calls, "re-read after dump must be identical");
}

/// Port of
/// `test_tool_calls_serialization.py::test_multiple_tool_calls_materialised`
/// (parametrized 1 and 3): every tool call in a single assistant message
/// survives a serialize -> deserialize round trip with ids and order intact.
#[test]
fn multiple_tool_calls_materialised_after_round_trip() {
    for count in [1_usize, 3] {
        let calls = (0..count)
            .map(|index| tool_call_json(&format!("call_{index}"), &format!("func_{index}"), &format!(r#"{{"arg": {index}}}"#)))
            .collect::<Vec<_>>();
        let parsed: ChatMessage =
            serde_json::from_value(assistant_message_with_tool_calls(json!(calls))).unwrap();
        assert_eq!(parsed.tool_calls.as_ref().unwrap().len(), count);

        let dump = serde_json::to_string(&parsed.tool_calls).unwrap();
        let reread: Vec<ToolCall> = serde_json::from_str(&dump).unwrap();
        assert_eq!(reread.len(), count);
        for (index, call) in reread.iter().enumerate() {
            assert_eq!(call.id, format!("call_{index}"));
            assert_eq!(call.function.name, format!("func_{index}"));
        }
    }
}

/// Port of
/// `test_tool_calls_serialization.py::test_messages_without_tool_calls_unaffected`:
/// messages that never carried `tool_calls` deserialize to `None` and
/// serialize with no `tool_calls` key injected.
#[test]
fn messages_without_tool_calls_unaffected() {
    let parsed: ChatMessage =
        serde_json::from_value(json!({"role": "assistant", "content": "Hi there!"})).unwrap();
    assert!(parsed.tool_calls.is_none());

    let message = AssistantMessage {
        role: "assistant",
        content: Some("Hi there!".to_owned()),
        reasoning_content: None,
        tool_calls: None,
    };
    let payload = serde_json::to_value(&message).unwrap();
    assert!(
        payload.get("tool_calls").is_none(),
        "tool_calls key must not be injected for plain messages"
    );
    assert_eq!(payload["content"], "Hi there!");
}

/// Port of the OpenAI-shape coercion half of
/// `test_tool_calls_serialization.py` (and the vllm request protocol):
/// `function.arguments` accepts a JSON string, a JSON object, explicit null,
/// or omission — all normalising to the string form used on the wire.
#[test]
fn tool_call_arguments_accept_openai_shapes() {
    let cases: Vec<Value> = vec![
        // Plain string (canonical wire form).
        tool_call_json("call_s", "fn", r#"{"a": 1}"#),
        // Object form (some clients send parsed objects).
        json!({"id": "call_o", "type": "function",
               "function": {"name": "fn", "arguments": {"a": 1}}}),
        // Explicit null -> empty object arguments.
        json!({"id": "call_n", "type": "function",
               "function": {"name": "fn", "arguments": null}}),
        // Omitted arguments -> empty object arguments.
        json!({"id": "call_m", "type": "function", "function": {"name": "fn"}}),
    ];
    let message = json!({"role": "assistant", "content": null, "tool_calls": cases});
    let parsed: ChatMessage = serde_json::from_value(message).unwrap();
    let arguments = parsed
        .tool_calls
        .unwrap()
        .iter()
        .map(|call| call.function.arguments.clone())
        .collect::<Vec<_>>();
    assert_eq!(
        arguments,
        vec![
            r#"{"a": 1}"#.to_owned(), // string form passes through verbatim
            r#"{"a":1}"#.to_owned(),  // object form normalises via to_string
            "{}".to_owned(),
            "{}".to_owned(),
        ]
    );

    // Non-object/non-string arguments (e.g. an array) must be rejected.
    let mut bad_function = serde_json::Map::new();
    bad_function.insert("name".to_owned(), json!("fn"));
    bad_function.insert("arguments".to_owned(), json!([1, 2]));
    let mut bad_call = serde_json::Map::new();
    bad_call.insert("id".to_owned(), json!("call_b"));
    bad_call.insert("type".to_owned(), json!("function"));
    bad_call.insert("function".to_owned(), Value::Object(bad_function));
    let bad = json!({"role": "assistant", "content": null, "tool_calls": [Value::Object(bad_call)]});
    assert!(
        serde_json::from_value::<ChatMessage>(bad).is_err(),
        "array-shaped arguments must not validate"
    );
}

// ---------------------------------------------------------------------------
// deepseekv4/deepseekv32 tool-parser invariant classes
// ---------------------------------------------------------------------------

/// Port of
/// `test_deepseekv4_tool_parser.py::test_no_dsml_closing_tag_leak_in_streamed_args`
/// (non-streaming half) and `test_non_streaming_extract_with_angle_brackets`:
/// parameter values containing `>` (shell redirects like `2>&1`) must parse
/// to the exact value with no DSML delimiter text captured into arguments.
#[test]
fn no_dsml_closing_tag_leak_in_arguments_with_angle_brackets() {
    let output = dsml_tool_call("run_command", &[("command", "git --version 2>&1", true)]);
    let parsed = parse_ds4_tool_calls(&output);
    assert_eq!(parsed.tool_calls.len(), 1);
    let arguments = &parsed.tool_calls[0].function.arguments;
    assert!(
        !arguments.contains("DSML"),
        "DSML delimiter leaked into arguments: {arguments}"
    );
    assert_eq!(
        serde_json::from_str::<Value>(arguments).unwrap(),
        json!({"command": "git --version 2>&1"})
    );
}

/// Port of the streaming half of
/// `test_deepseekv4_tool_parser.py::test_no_dsml_closing_tag_leak_in_streamed_args`:
/// the `'>'`-in-value case must hold at *every* chunk size — the streamed
/// argument reassembly must never contain marker text and must parse to the
/// exact expected JSON.
#[test]
fn no_dsml_leak_in_streamed_args_at_every_chunk_size() {
    let full_text = streamed_dsml_tool_call("run_command", &[("command", "git --version 2>&1", true)]);
    for chunk_size in 1..=full_text.len() {
        let deltas = stream_in_chunks(&full_text, chunk_size);
        let arguments = streamed_arguments(&deltas, 0);
        assert!(
            !arguments.is_empty(),
            "no arguments emitted at chunk_size={chunk_size}"
        );
        assert!(
            !arguments.contains("DSML"),
            "DSML marker leaked into args at chunk_size={chunk_size}: {arguments}"
        );
        assert_eq!(
            serde_json::from_str::<Value>(&arguments).unwrap(),
            json!({"command": "git --version 2>&1"}),
            "args mismatch at chunk_size={chunk_size}"
        );
    }
}

/// Port of `test_deepseekv32_tool_parser.py::{test_no_marker_leak_chunked,
/// test_no_marker_leak_with_prefix_chunked, test_no_marker_leak_char_by_char,
/// test_no_marker_leak_all_split_points}` (GitHub #40801): content before a
/// tool call must not include start-marker fragments, at any chunking.
#[test]
fn no_start_marker_leak_in_streamed_content_at_every_chunk_size() {
    let full_text = format!("Hello!{}", streamed_dsml_tool_call("fn", &[("k", "v", true)]));
    for chunk_size in 1..=full_text.len() {
        let deltas = stream_in_chunks(&full_text, chunk_size);
        let content = streamed_content(&deltas);
        assert_eq!(content, "Hello!", "leaked content at chunk_size={chunk_size}");
        assert!(
            !content.contains("<｜"),
            "marker fragment leaked at chunk_size={chunk_size}: {content}"
        );
        assert_eq!(
            serde_json::from_str::<Value>(&streamed_arguments(&deltas, 0)).unwrap(),
            json!({"k": "v"}),
            "args mismatch at chunk_size={chunk_size}"
        );
    }
}

/// Port of
/// `test_deepseekv32_tool_parser.py::test_false_partial_marker_emitted`: text
/// that merely *looks* like the start of a marker but is not one must still
/// be emitted as content once disambiguated.
#[test]
fn false_partial_marker_is_emitted_as_content() {
    let full_text = "<｜DSM some regular text";
    for chunk_size in 1..=full_text.len() {
        let deltas = stream_in_chunks(full_text, chunk_size);
        assert_eq!(
            streamed_content(&deltas),
            full_text,
            "false partial marker dropped at chunk_size={chunk_size}"
        );
    }
}

/// Mapping note for
/// `test_deepseekv4_tool_parser.py::test_streaming_emits_incremental_argument_chunks`
/// and `test_deepseekv32_tool_parser.py::test_emits_arguments_before_invoke_completes`:
/// vllm streams argument fragments incrementally; ds41rt's parser
/// deliberately withholds everything until the `</｜DSML｜invoke>` block
/// completes (see `tooling.rs::incremental_parser_withholds_incomplete_control_syntax`).
/// The closest ds41rt invariants are pinned here: no tool-call delta (and
/// certainly no partial arguments) is emitted before the invoke completes,
/// and completed calls arrive as exactly one delta with complete arguments.
#[test]
fn arguments_are_emitted_atomically_once_complete() {
    let full_text = streamed_dsml_tool_call("search", &[("query", "deepseek v4", true)]);
    // Everything up to (but excluding) the invoke end tag.
    let partial_end = full_text.find(INV_END).unwrap();
    let mut parser = Ds4ToolCallStreamParser::new();
    let mut deltas = parser.push(&full_text[..partial_end]);
    assert!(
        deltas
            .iter()
            .all(|delta| !matches!(delta, Ds4ToolStreamDelta::ToolCall { .. })),
        "no tool-call delta may be emitted before the invoke block completes"
    );
    deltas.extend(parser.push(&full_text[partial_end..]));
    deltas.extend(parser.finish());

    let names = streamed_tool_names(&deltas);
    assert_eq!(names, vec!["search".to_owned()]);
    // Exactly one arguments fragment, already complete JSON.
    let argument_fragments = deltas
        .iter()
        .filter_map(|delta| match delta {
            Ds4ToolStreamDelta::ToolCall {
                arguments: Some(fragment),
                ..
            } => Some(fragment.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        argument_fragments.len(),
        1,
        "ds41rt emits arguments atomically, not incrementally"
    );
    assert_eq!(
        serde_json::from_str::<Value>(&argument_fragments[0]).unwrap(),
        json!({"query": "deepseek v4"})
    );
}

/// Malformed-recovery class (v32 suite): an invoke block left unclosed at
/// end of stream must produce no tool call rather than a corrupt partial
/// one. (The non-streaming half — incomplete DSML remains content — is
/// already covered by `tooling.rs::malformed_or_incomplete_dsml_remains_content`.)
#[test]
fn unfinished_invoke_block_produces_no_tool_call_on_finish() {
    let full_text = streamed_dsml_tool_call("search", &[("query", "vllm", true)]);
    let truncated = &full_text[..full_text.find(INV_END).unwrap()];
    let deltas = stream_in_chunks(truncated, 3);
    assert!(
        deltas
            .iter()
            .all(|delta| !matches!(delta, Ds4ToolStreamDelta::ToolCall { .. })),
        "truncated invoke must not yield a tool-call delta"
    );
}

/// Port of
/// `test_deepseekv32_tool_parser.py::test_unique_tool_call_ids`: parallel
/// invokes in one block each get a distinct generated id.
#[test]
fn parallel_invokes_receive_unique_tool_call_ids() {
    let output = format!(
        "{TC_START}\n{INV_START}get_weather\">\n{PARAM_START}location\" string=\"true\">SF{PARAM_END}\n{INV_END}\n{INV_START}get_weather\">\n{PARAM_START}location\" string=\"true\">NYC{PARAM_END}\n{INV_END}\n{TC_END}"
    );
    let parsed = parse_ds4_tool_calls(&output);
    assert_eq!(parsed.tool_calls.len(), 2);
    let first = &parsed.tool_calls[0].id;
    let second = &parsed.tool_calls[1].id;
    assert_ne!(first, second, "tool call ids must be unique");
    for id in [first, second] {
        assert!(id.starts_with("call_"), "generated id shape: {id}");
    }
}

/// Port of
/// `test_deepseekv32_tool_parser.py::test_type_conversion_in_non_streaming`
/// (schema-free subset): `string="false"` values are parsed as typed JSON
/// (integer, boolean, array, object); `string="true"` values stay strings
/// even when they look numeric. vllm's schema-driven coercions have no
/// ds41rt surface (ds41rt is schema-free) and are intentionally not ported.
#[test]
fn typed_parameter_values_convert_to_json() {
    let output = format!(
        "{TC_START}\n{INV_START}plan_trip\">\n\
         {PARAM_START}days\" string=\"false\">3{PARAM_END}\n\
         {PARAM_START}flexible\" string=\"false\">false{PARAM_END}\n\
         {PARAM_START}cities\" string=\"false\">[\"Beijing\",\"Tokyo\"]{PARAM_END}\n\
         {PARAM_START}meta\" string=\"false\">{{\"k\":1}}{PARAM_END}\n\
         {PARAM_START}code\" string=\"true\">42{PARAM_END}\n\
         {INV_END}\n{TC_END}"
    );
    let parsed = parse_ds4_tool_calls(&output);
    assert_eq!(parsed.tool_calls.len(), 1);
    assert_eq!(
        serde_json::from_str::<Value>(&parsed.tool_calls[0].function.arguments).unwrap(),
        json!({
            "days": 3,
            "flexible": false,
            "cities": ["Beijing", "Tokyo"],
            "meta": {"k": 1},
            "code": "42"
        })
    );
}

/// Port of the unicode-value case in
/// `test_deepseekv4_tool_parser.py::test_streaming_emits_incremental_argument_chunks`
/// (value `靠窗座位`): multibyte argument values must survive streaming at
/// every char-boundary split. ds41rt emits them in one atomic delta (see
/// `arguments_are_emitted_atomically_once_complete`) instead of fragments.
#[test]
fn unicode_argument_values_survive_every_char_boundary_split() {
    let full_text = streamed_dsml_tool_call("plan_trip", &[("notes", "靠窗座位", true)]);
    let boundaries = full_text
        .char_indices()
        .map(|(index, _)| index)
        .chain(std::iter::once(full_text.len()))
        .collect::<Vec<_>>();
    for split in boundaries {
        let mut parser = Ds4ToolCallStreamParser::new();
        let mut deltas = parser.push(&full_text[..split]);
        deltas.extend(parser.push(&full_text[split..]));
        deltas.extend(parser.finish());
        assert_eq!(
            serde_json::from_str::<Value>(&streamed_arguments(&deltas, 0)).unwrap(),
            json!({"notes": "靠窗座位"}),
            "split={split}"
        );
    }
}

/// Malformed-recovery class: a structurally invalid invoke (here: a
/// duplicate parameter name, which `parse_ds4_arguments` rejects) must be
/// dropped without swallowing the valid invokes around it.
#[test]
fn malformed_invoke_is_dropped_but_valid_siblings_parse() {
    let output = format!(
        "{TC_START}\n{INV_START}good\">\n{PARAM_START}k\" string=\"true\">v1{PARAM_END}\n{INV_END}\n\
         {INV_START}broken\">\n{PARAM_START}k\" string=\"true\">a{PARAM_END}\n{PARAM_START}k\" string=\"true\">b{PARAM_END}\n{INV_END}\n\
         {INV_START}good2\">\n{PARAM_START}k\" string=\"true\">v2{PARAM_END}\n{INV_END}\n{TC_END}"
    );
    let parsed = parse_ds4_tool_calls(&output);
    let names = parsed
        .tool_calls
        .iter()
        .map(|call| call.function.name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        names,
        vec!["good", "good2"],
        "malformed invoke must be skipped without blocking its siblings"
    );
}

// ---------------------------------------------------------------------------
// Documented divergences (vllm repairs / recovers; ds41rt does not)
// ---------------------------------------------------------------------------

/// MAPPING DIVERGENCE — port of
/// `test_deepseekv4_tool_parser.py::test_extract_tool_calls_arguments_wrapper`
/// / v32 `test_arguments_wrapper_repaired`: vllm *repairs* a model-emitted
/// `<｜DSML｜parameter name="arguments" string="false">{...}` wrapper by
/// unwrapping the JSON into the call's arguments. ds41rt's schema-free
/// parser has no repair pass: it treats `arguments` as an ordinary parameter
/// name, so the call survives (name, validity) but the payload is nested one
/// level deep. This test pins the closest ds41rt behavior.
#[test]
fn arguments_wrapper_is_nested_parameter_not_repaired() {
    let output = format!(
        "{TC_START}{INV_START}get_weather\">{PARAM_START}arguments\" string=\"false\">{{\"location\":\"Beijing\"}}{PARAM_END}{INV_END}{TC_END}"
    );
    let parsed = parse_ds4_tool_calls(&output);
    assert_eq!(parsed.tool_calls.len(), 1, "the call itself must survive");
    assert_eq!(parsed.tool_calls[0].function.name, "get_weather");
    // ds41rt nests; vllm would repair to {"location": "Beijing"}.
    assert_eq!(
        serde_json::from_str::<Value>(&parsed.tool_calls[0].function.arguments).unwrap(),
        json!({"arguments": {"location": "Beijing"}})
    );
}

/// MAPPING DIVERGENCE — port of
/// `test_deepseekv4_tool_parser.py::test_missing_tool_calls_wrapper_is_recovered`
/// (regression for vllm #48931): vllm recovers an invoke whose
/// `<｜DSML｜tool_calls>` start marker the model omitted at long context.
/// ds41rt's `parse_ds4_tool_calls` anchors strictly on the wrapper and does
/// NOT recover: the marker-free output remains content with zero tool calls.
/// Closest ds41rt behavior pinned here.
#[test]
fn missing_tool_calls_wrapper_is_not_recovered_non_streaming() {
    let output = dsml_tool_call("search", &[("query", "vllm", true)])
        .replace(&format!("{TC_START}\n"), "");
    assert!(!output.contains(TC_START));
    let parsed = parse_ds4_tool_calls(&output);
    assert!(
        parsed.tool_calls.is_empty(),
        "ds41rt does not recover invokes without the tool_calls wrapper"
    );
    assert_eq!(
        parsed.content.as_deref(),
        Some(output.trim()),
        "unrecovered output must pass through as content"
    );
}

/// MAPPING DIVERGENCE — streaming half of
/// `test_deepseekv4_tool_parser.py::test_missing_tool_calls_wrapper_is_recovered`:
/// without the wrapper, ds41rt's stream parser emits the raw invoke syntax
/// into the content deltas (no tool call is recovered). vllm instead
/// recovers the call and streams it cleanly. Pinned as the closest ds41rt
/// behavior.
#[test]
fn missing_tool_calls_wrapper_stream_emits_invoke_syntax_as_content() {
    let output = streamed_dsml_tool_call("search", &[("query", "vllm", true)])
        .replace(&format!("{TC_START}\n"), "");
    for chunk_size in [1, 3, 7] {
        let deltas = stream_in_chunks(&output, chunk_size);
        assert!(
            deltas
                .iter()
                .all(|delta| !matches!(delta, Ds4ToolStreamDelta::ToolCall { .. })),
            "no tool call may be recovered without the wrapper"
        );
        let content = streamed_content(&deltas);
        assert_eq!(
            content, output,
            "wrapper-free invoke text is streamed verbatim as content"
        );
    }
}

/// MAPPING DIVERGENCE — port of
/// `test_deepseekv4_tool_parser.py::test_function_calls_wrapper_is_not_recognized`:
/// vllm's V4 parser does not terminate on the V3.2 `<｜DSML｜function_calls>`
/// wrapper but still parses the invoke inside it (tool calls anchor on the
/// invoke). ds41rt anchors strictly on the `tool_calls` wrapper, so the
/// V3.2-shaped block is treated as plain content with zero tool calls.
#[test]
fn function_calls_wrapper_is_not_recognized() {
    let output = dsml_tool_call("search", &[("query", "vllm", true)])
        .replace("tool_calls", "function_calls");
    let parsed = parse_ds4_tool_calls(&output);
    assert!(
        parsed.tool_calls.is_empty(),
        "ds41rt does not parse invokes inside a function_calls wrapper"
    );
    assert_eq!(
        parsed.content.as_deref(),
        Some(output.trim()),
        "wrapper text passes through as content"
    );
}
