use serde_json::Value;
use std::collections::HashSet;
use std::env;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::{invalid_request, ApiError};
use crate::{
    ChatCompletionRequest, ChatMessage, ChatTool, RealFullSamplingParams, ResponseFormat, StopSpec,
    ToolChoice,
};

const REAL_FULL_ENABLE_THINKING_ENV: &str = "DS41RT_REAL_FULL_ENABLE_THINKING";
const MAX_CONSTRAINT_SCHEMA_BYTES: usize = 1024 * 1024;
const MAX_TOOL_SCHEMA_BYTES: usize = 4 * MAX_CONSTRAINT_SCHEMA_BYTES;
const MAX_STRICT_SCHEMA_DEPTH: usize = 64;
const MAX_STRICT_SCHEMA_NODES: usize = 4096;

pub(crate) fn validate_request(request: &ChatCompletionRequest) -> Result<(), ApiError> {
    if request.model.trim().is_empty() {
        return Err(invalid_request("model must not be empty", Some("model")));
    }
    if request.messages.is_empty() {
        return Err(invalid_request(
            "messages must contain at least one item",
            Some("messages"),
        ));
    }
    let mut prior_tool_call_ids = HashSet::new();
    for (idx, message) in request.messages.iter().enumerate() {
        match message.role.as_str() {
            "system" | "developer" | "user" | "assistant" | "tool" | "latest_reminder" => {}
            _ => {
                return Err(invalid_request(
                    format!("unsupported message role {}", message.role),
                    Some(format!("messages[{idx}].role")),
                ));
            }
        }
        if message.tool_calls.is_some() && message.role != "assistant" {
            return Err(invalid_request(
                "tool_calls are only valid on assistant messages",
                Some(format!("messages[{idx}].tool_calls")),
            ));
        }
        if message.tool_call_id.is_some() && message.role != "tool" {
            return Err(invalid_request(
                "tool_call_id is only valid on tool messages",
                Some(format!("messages[{idx}].tool_call_id")),
            ));
        }
        if message.tools.is_some() && !matches!(message.role.as_str(), "system" | "developer") {
            return Err(invalid_request(
                "message-local tools are only valid on system or developer messages",
                Some(format!("messages[{idx}].tools")),
            ));
        }
        if message.response_format.is_some()
            && !matches!(message.role.as_str(), "system" | "developer")
        {
            return Err(invalid_request(
                "message-local response_format is only valid on system or developer messages",
                Some(format!("messages[{idx}].response_format")),
            ));
        }
        if message.wo_eos && message.role != "assistant" {
            return Err(invalid_request(
                "wo_eos is only valid on assistant messages",
                Some(format!("messages[{idx}].wo_eos")),
            ));
        }
        if let Some(task) = message.task.as_deref() {
            if !matches!(message.role.as_str(), "user" | "developer") {
                return Err(invalid_request(
                    "task is only valid on user or developer messages",
                    Some(format!("messages[{idx}].task")),
                ));
            }
            if !matches!(
                task,
                "action" | "query" | "authority" | "domain" | "title" | "read_url"
            ) {
                return Err(invalid_request(
                    "unsupported DeepSeek V4 task token",
                    Some(format!("messages[{idx}].task")),
                ));
            }
        }
        if let Some(tools) = message.tools.as_deref() {
            validate_tools(tools, &format!("messages[{idx}].tools"))?;
        }
        if let Some(response_format) = message.response_format.as_ref() {
            if !response_format.is_object() {
                return Err(invalid_request(
                    "message-local response_format must be a JSON Schema object",
                    Some(format!("messages[{idx}].response_format")),
                ));
            }
            validate_constraint_schema_size(
                response_format,
                format!("messages[{idx}].response_format"),
                MAX_CONSTRAINT_SCHEMA_BYTES,
            )?;
        }
        if let Some(tool_calls) = &message.tool_calls {
            for (call_idx, tool_call) in tool_calls.iter().enumerate() {
                let param = format!("messages[{idx}].tool_calls[{call_idx}]");
                if tool_call.id.trim().is_empty() {
                    return Err(invalid_request(
                        "tool call id must not be empty",
                        Some(format!("{param}.id")),
                    ));
                }
                if !prior_tool_call_ids.insert(tool_call.id.clone()) {
                    return Err(invalid_request(
                        format!("duplicate tool call id {}", tool_call.id),
                        Some(format!("{param}.id")),
                    ));
                }
                if tool_call.tool_type != "function" {
                    return Err(invalid_request(
                        "only function tool calls are supported",
                        Some(format!("{param}.type")),
                    ));
                }
                if tool_call.function.name.trim().is_empty() {
                    return Err(invalid_request(
                        "tool call function name must not be empty",
                        Some(format!("{param}.function.name")),
                    ));
                }
                if !matches!(
                    serde_json::from_str::<Value>(&tool_call.function.arguments),
                    Ok(Value::Object(_))
                ) {
                    return Err(invalid_request(
                        "tool call arguments must be a JSON object string",
                        Some(format!("{param}.function.arguments")),
                    ));
                }
            }
        }
        if message.role == "tool" {
            let Some(tool_call_id) = message
                .tool_call_id
                .as_deref()
                .filter(|tool_call_id| !tool_call_id.trim().is_empty())
            else {
                return Err(invalid_request(
                    "tool messages require tool_call_id",
                    Some(format!("messages[{idx}].tool_call_id")),
                ));
            };
            if !prior_tool_call_ids.contains(tool_call_id) {
                return Err(invalid_request(
                    format!("tool_call_id {tool_call_id} has no preceding assistant tool call"),
                    Some(format!("messages[{idx}].tool_call_id")),
                ));
            }
        }
    }
    if request.max_tokens == Some(0) {
        return Err(invalid_request(
            "max_tokens must be greater than zero",
            Some("max_tokens"),
        ));
    }
    if request.max_completion_tokens == Some(0) {
        return Err(invalid_request(
            "max_completion_tokens must be greater than zero",
            Some("max_completion_tokens"),
        ));
    }
    let max_tokens = request_max_tokens(request);
    if request
        .min_tokens
        .is_some_and(|minimum| minimum > max_tokens)
    {
        return Err(invalid_request(
            format!("min_tokens cannot exceed the requested output limit ({max_tokens})"),
            Some("min_tokens"),
        ));
    }
    if request
        .temperature
        .is_some_and(|value| !value.is_finite() || !(0.0..=2.0).contains(&value))
    {
        return Err(invalid_request(
            "temperature must be a finite number between 0 and 2",
            Some("temperature"),
        ));
    }
    if request
        .top_p
        .is_some_and(|value| !value.is_finite() || !(0.0 < value && value <= 1.0))
    {
        return Err(invalid_request(
            "top_p must be greater than 0 and at most 1",
            Some("top_p"),
        ));
    }
    if request
        .top_k
        .is_some_and(|value| !(1..=64).contains(&value))
    {
        return Err(invalid_request(
            "top_k must be between 1 and 64",
            Some("top_k"),
        ));
    }
    if let Some(effort) = request.reasoning_effort.as_ref() {
        let valid = effort.as_str().is_some_and(|effort| {
            matches!(
                effort.to_ascii_lowercase().as_str(),
                "none" | "off" | "disabled" | "low" | "high" | "max"
            )
        });
        if !valid {
            return Err(invalid_request(
                "reasoning_effort must be one of low, high, max, or disabled",
                Some("reasoning_effort"),
            ));
        }
    }
    if let Some(response_format) = request.response_format.as_ref() {
        match response_format {
            ResponseFormat::Text | ResponseFormat::JsonObject => {}
            ResponseFormat::JsonSchema { json_schema } => {
                validate_identifier(
                    &json_schema.name,
                    "response_format JSON Schema name",
                    "response_format.json_schema.name",
                )?;
                if !json_schema.schema.is_object() {
                    return Err(invalid_request(
                        "response_format JSON Schema must be an object",
                        Some("response_format.json_schema.schema"),
                    ));
                }
                validate_constraint_schema_size(
                    &json_schema.schema,
                    "response_format.json_schema.schema",
                    MAX_CONSTRAINT_SCHEMA_BYTES,
                )?;
                if json_schema.strict.unwrap_or(false) {
                    validate_strict_json_schema(
                        &json_schema.schema,
                        "response_format.json_schema.schema",
                    )?;
                }
            }
        }
    }
    if let Some(tools) = &request.tools {
        validate_tools(tools, "tools")?;
    }
    let tools = request.tools.as_deref().unwrap_or(&[]);
    match &request.tool_choice {
        Some(ToolChoice::Mode(mode)) if mode == "none" || mode == "auto" => {}
        Some(ToolChoice::Mode(mode)) if mode == "required" => {
            if tools.is_empty() {
                return Err(invalid_request(
                    "tool_choice required but no tools were provided",
                    Some("tool_choice"),
                ));
            }
        }
        Some(ToolChoice::Mode(mode)) => {
            return Err(invalid_request(
                format!("unsupported tool_choice mode {mode}"),
                Some("tool_choice"),
            ));
        }
        Some(ToolChoice::Specific {
            tool_type,
            function,
        }) => {
            if tool_type != "function" {
                return Err(invalid_request(
                    "specific tool_choice must have type function",
                    Some("tool_choice.type"),
                ));
            }
            if !tools.iter().any(|tool| tool.function.name == function.name) {
                return Err(invalid_request(
                    format!("requested tool {} was not provided", function.name),
                    Some("tool_choice.function.name"),
                ));
            }
        }
        None => {}
    }
    Ok(())
}

fn validate_tools(tools: &[ChatTool], parameter: &str) -> Result<(), ApiError> {
    if tools.len() > 128 {
        return Err(invalid_request(
            "tools must contain at most 128 functions",
            Some(parameter.to_owned()),
        ));
    }
    let mut tool_names = HashSet::new();
    let mut total_schema_bytes = 0_usize;
    for (idx, tool) in tools.iter().enumerate() {
        if tool.tool_type != "function" {
            return Err(invalid_request(
                "only function tools are supported",
                Some(format!("{parameter}[{idx}].type")),
            ));
        }
        validate_identifier(
            &tool.function.name,
            "function tool name",
            format!("{parameter}[{idx}].function.name"),
        )?;
        if !tool_names.insert(tool.function.name.as_str()) {
            return Err(invalid_request(
                format!("duplicate function tool name {}", tool.function.name),
                Some(format!("{parameter}[{idx}].function.name")),
            ));
        }
        if tool
            .function
            .parameters
            .as_ref()
            .is_some_and(|parameters| !parameters.is_object())
        {
            return Err(invalid_request(
                "function tool parameters must be a JSON Schema object",
                Some(format!("{parameter}[{idx}].function.parameters")),
            ));
        }
        if let Some(parameters) = tool.function.parameters.as_ref() {
            let schema_parameter = format!("{parameter}[{idx}].function.parameters");
            let schema_bytes = validate_constraint_schema_size(
                parameters,
                &schema_parameter,
                MAX_CONSTRAINT_SCHEMA_BYTES,
            )?;
            total_schema_bytes = total_schema_bytes
                .checked_add(schema_bytes)
                .ok_or_else(|| {
                    invalid_request("tool schemas are too large", Some(parameter.to_owned()))
                })?;
            if total_schema_bytes > MAX_TOOL_SCHEMA_BYTES {
                return Err(invalid_request(
                    format!("tool schemas must total at most {MAX_TOOL_SCHEMA_BYTES} bytes"),
                    Some(parameter.to_owned()),
                ));
            }
        }
        if tool.function.strict.unwrap_or(false) {
            let parameters = tool
                .function
                .parameters
                .as_ref()
                .cloned()
                .unwrap_or_else(crate::constrained::empty_parameters);
            validate_strict_json_schema(
                &parameters,
                format!("{parameter}[{idx}].function.parameters"),
            )?;
        }
    }
    Ok(())
}

fn validate_constraint_schema_size(
    schema: &Value,
    parameter: impl Into<String>,
    max_bytes: usize,
) -> Result<usize, ApiError> {
    let parameter = parameter.into();
    let bytes = serde_json::to_vec(schema)
        .map_err(|error| {
            invalid_request(
                format!("JSON Schema cannot be serialized: {error}"),
                Some(parameter.clone()),
            )
        })?
        .len();
    if bytes > max_bytes {
        return Err(invalid_request(
            format!("JSON Schema must be at most {max_bytes} bytes"),
            Some(parameter),
        ));
    }
    Ok(bytes)
}

fn validate_identifier(
    value: &str,
    label: &str,
    parameter: impl Into<String>,
) -> Result<(), ApiError> {
    let parameter = parameter.into();
    if value.is_empty()
        || value.len() > 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        return Err(invalid_request(
            format!("{label} must be 1-64 characters containing only letters, numbers, underscores, or hyphens"),
            Some(parameter),
        ));
    }
    Ok(())
}

/// Shared OpenAI strict-mode subset check. `pub(crate)` so the native
/// `serve-native` path can apply the same contract as `validate_request`
/// (the adapter crate deserializes `json_schema` into a fieldless variant).
pub(crate) fn validate_strict_json_schema(
    schema: &Value,
    parameter: impl Into<String>,
) -> Result<(), ApiError> {
    fn visit(
        schema: &Value,
        path: &str,
        depth: usize,
        visited_nodes: &mut usize,
    ) -> Result<(), String> {
        if depth > MAX_STRICT_SCHEMA_DEPTH {
            return Err(format!(
                "{path} exceeds the strict JSON Schema nesting limit of {MAX_STRICT_SCHEMA_DEPTH}"
            ));
        }
        *visited_nodes = visited_nodes
            .checked_add(1)
            .ok_or_else(|| format!("{path} has too many schema nodes"))?;
        if *visited_nodes > MAX_STRICT_SCHEMA_NODES {
            return Err(format!(
                "{path} exceeds the strict JSON Schema node limit of {MAX_STRICT_SCHEMA_NODES}"
            ));
        }
        let Some(object) = schema.as_object() else {
            return Err(format!("{path} must be a JSON Schema object"));
        };
        let is_object = object.get("type").is_some_and(|kind| {
            kind == "object"
                || kind
                    .as_array()
                    .is_some_and(|kinds| kinds.iter().any(|candidate| candidate == "object"))
        }) || object.contains_key("properties");
        if is_object {
            let properties = object
                .get("properties")
                .and_then(Value::as_object)
                .ok_or_else(|| format!("{path}.properties must be an object in strict mode"))?;
            if object.get("additionalProperties") != Some(&Value::Bool(false)) {
                return Err(format!(
                    "{path}.additionalProperties must be false in strict mode"
                ));
            }
            let required = object
                .get("required")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    format!("{path}.required must list every property in strict mode")
                })?;
            let required = required
                .iter()
                .map(|name| {
                    name.as_str()
                        .ok_or_else(|| format!("{path}.required entries must be strings"))
                })
                .collect::<Result<HashSet<_>, _>>()?;
            if required.len() != properties.len()
                || properties
                    .keys()
                    .any(|name| !required.contains(name.as_str()))
            {
                return Err(format!(
                    "{path}.required must contain every property exactly once in strict mode"
                ));
            }
            for (name, property) in properties {
                visit(
                    property,
                    &format!("{path}.properties.{name}"),
                    depth + 1,
                    visited_nodes,
                )?;
            }
        }
        if let Some(items) = object.get("items") {
            visit(items, &format!("{path}.items"), depth + 1, visited_nodes)?;
        }
        for keyword in ["anyOf", "oneOf", "allOf"] {
            if let Some(variants) = object.get(keyword).and_then(Value::as_array) {
                for (index, variant) in variants.iter().enumerate() {
                    visit(
                        variant,
                        &format!("{path}.{keyword}[{index}]"),
                        depth + 1,
                        visited_nodes,
                    )?;
                }
            }
        }
        for keyword in ["$defs", "definitions"] {
            if let Some(definitions) = object.get(keyword).and_then(Value::as_object) {
                for (name, definition) in definitions {
                    visit(
                        definition,
                        &format!("{path}.{keyword}.{name}"),
                        depth + 1,
                        visited_nodes,
                    )?;
                }
            }
        }
        Ok(())
    }

    let parameter = parameter.into();
    let root = schema.as_object().ok_or_else(|| {
        invalid_request(
            "strict JSON Schema root must be an object",
            Some(parameter.clone()),
        )
    })?;
    let root_is_object =
        root.get("type").is_some_and(|kind| kind == "object") || root.contains_key("properties");
    if !root_is_object {
        return Err(invalid_request(
            "strict JSON Schema root type must be object",
            Some(parameter),
        ));
    }
    let mut visited_nodes = 0;
    visit(schema, &parameter, 0, &mut visited_nodes)
        .map_err(|message| invalid_request(message, Some(parameter)))
}

pub(crate) fn request_uses_greedy_sampling(request: &ChatCompletionRequest) -> bool {
    request.temperature.unwrap_or(0.0) == 0.0 || request.top_k == Some(1)
}

pub(crate) fn request_sampling_params(request: &ChatCompletionRequest) -> RealFullSamplingParams {
    static NEXT_SEED: AtomicU64 = AtomicU64::new(0);
    if request_uses_greedy_sampling(request) {
        return RealFullSamplingParams::greedy();
    }
    let generated_seed = || {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        now ^ NEXT_SEED.fetch_add(0x9e37_79b9_7f4a_7c15, Ordering::Relaxed)
    };
    RealFullSamplingParams::new(
        request.temperature.unwrap_or(1.0),
        request.top_p.unwrap_or(0.95),
        request.top_k.unwrap_or(50),
        request.seed.map_or_else(generated_seed, |seed| seed as u64),
    )
}

pub(crate) fn request_max_tokens(request: &ChatCompletionRequest) -> usize {
    request
        .max_completion_tokens
        .or(request.max_tokens)
        .unwrap_or(16 * 1024)
}

pub(crate) fn tool_calls_enabled(request: &ChatCompletionRequest) -> bool {
    !matches!(request.tool_choice, Some(ToolChoice::Mode(ref mode)) if mode == "none")
        && (request
            .tools
            .as_ref()
            .is_some_and(|tools| !tools.is_empty())
            || request.messages.iter().any(|message| {
                message
                    .tools
                    .as_ref()
                    .is_some_and(|tools| !tools.is_empty())
            }))
}

pub(crate) fn prompt_text(messages: &[ChatMessage]) -> String {
    messages
        .iter()
        .map(|message| {
            let mut text = String::new();
            text.push_str(&message.role);
            text.push_str(": ");
            text.push_str(&message_content_text(&message.content));
            if let Some(name) = &message.name {
                text.push_str(" name=");
                text.push_str(name);
            }
            if let Some(tool_call_id) = &message.tool_call_id {
                text.push_str(" tool_call_id=");
                text.push_str(tool_call_id);
            }
            if let Some(tool_calls) = &message.tool_calls {
                text.push_str(" tool_calls=");
                text.push_str(&tool_calls.len().to_string());
            }
            text
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
pub(crate) fn real_ds4_full_prompt_text(messages: &[ChatMessage]) -> String {
    let request = ChatCompletionRequest {
        model: "deepseek-ai/DeepSeek-V4-Flash-0731".to_owned(),
        messages: messages.to_vec(),
        stream: false,
        stream_options: None,
        max_tokens: None,
        max_completion_tokens: None,
        min_tokens: None,
        ignore_eos: None,
        temperature: None,
        top_p: None,
        top_k: None,
        seed: None,
        stop: None,
        response_format: None,
        tools: None,
        tool_choice: None,
        parallel_tool_calls: None,
        tool_decoding_assistance: None,
        thinking: None,
        enable_thinking: None,
        reasoning_effort: None,
    };
    crate::encoding::render_ds4_request_prompt(&request)
}

pub(crate) fn real_ds4_full_request_prompt_text(request: &ChatCompletionRequest) -> String {
    crate::encoding::render_ds4_request_prompt(request)
}

pub(crate) fn request_thinking_enabled(request: &ChatCompletionRequest) -> bool {
    if let Some(thinking) = request.thinking.as_ref() {
        return matches!(
            thinking.thinking_type.trim().to_ascii_lowercase().as_str(),
            "enabled" | "enable" | "on" | "true"
        );
    }
    if let Some(enable_thinking) = request.enable_thinking {
        return enable_thinking;
    }
    if let Some(reasoning_effort) = request.reasoning_effort.as_ref() {
        return !matches!(
            reasoning_effort
                .as_str()
                .map(str::to_ascii_lowercase)
                .as_deref(),
            Some("none" | "off" | "disabled")
        );
    }
    env::var(REAL_FULL_ENABLE_THINKING_ENV)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::{
        real_ds4_full_request_prompt_text, request_image_sources, validate_request,
        ChatCompletionRequest, MAX_CONSTRAINT_SCHEMA_BYTES, MAX_STRICT_SCHEMA_DEPTH,
    };

    #[test]
    fn reasoning_effort_uses_the_deepseek_v4_high_or_max_contract() {
        let high: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}],
            "thinking": {"type": "enabled"},
            "reasoning_effort": "high"
        }))
        .unwrap();
        assert!(real_ds4_full_request_prompt_text(&high)
            .starts_with("<｜begin▁of▁sentence｜>Reasoning Effort: Absolute maximum"));

        let max: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}],
            "thinking": {"type": "enabled"},
            "reasoning_effort": "max"
        }))
        .unwrap();
        assert!(real_ds4_full_request_prompt_text(&max)
            .starts_with("<｜begin▁of▁sentence｜>Reasoning Effort: Beyond maximum"));
    }

    #[test]
    fn multipart_image_content_preserves_order_and_extracts_urls() {
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "before"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                    {"type": "text", "text": "after"}
                ]
            }]
        }))
        .unwrap();
        let prompt = real_ds4_full_request_prompt_text(&request);
        assert!(prompt.contains("<｜User｜>before<|begin_of_image|><|image|><|end_of_image|>after"));
        assert_eq!(
            request_image_sources(&request).unwrap(),
            vec!["data:image/png;base64,AA=="]
        );
    }

    #[test]
    fn constrained_schema_limits_bound_compile_input_and_validation_depth() {
        let oversized: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "oversized",
                    "schema": {
                        "type": "object",
                        "description": "x".repeat(MAX_CONSTRAINT_SCHEMA_BYTES)
                    }
                }
            }
        }))
        .unwrap();
        assert!(validate_request(&oversized).is_err());

        let mut nested = serde_json::json!({"type": "string"});
        for _ in 0..=MAX_STRICT_SCHEMA_DEPTH {
            nested = serde_json::json!({"type": "array", "items": nested});
        }
        let too_deep: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "too_deep",
                    "strict": true,
                    "schema": {
                        "type": "object",
                        "properties": {"value": nested},
                        "required": ["value"],
                        "additionalProperties": false
                    }
                }
            }
        }))
        .unwrap();
        assert!(validate_request(&too_deep).is_err());
    }
}

pub(crate) fn message_content_text(content: &Option<Value>) -> String {
    match content {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| {
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    return Some(text.to_owned());
                }
                image_part_source(part)
                    .is_some()
                    .then(|| "<|begin_of_image|><|image|><|end_of_image|>".to_owned())
            })
            .collect::<String>(),
        Some(Value::Null) | None => String::new(),
        Some(other) => other.to_string(),
    }
}

pub(crate) fn request_image_sources(
    request: &ChatCompletionRequest,
) -> Result<Vec<String>, ApiError> {
    let mut sources = Vec::new();
    for (message_index, message) in request.messages.iter().enumerate() {
        let Some(Value::Array(parts)) = message.content.as_ref() else {
            continue;
        };
        for (part_index, part) in parts.iter().enumerate() {
            let is_image = part
                .get("type")
                .and_then(Value::as_str)
                .is_some_and(|kind| matches!(kind, "image" | "image_url"));
            if !is_image {
                continue;
            }
            let source = image_part_source(part).ok_or_else(|| {
                invalid_request(
                    "image content requires image_url.url (or a string image_url)",
                    Some(format!(
                        "messages[{message_index}].content[{part_index}].image_url"
                    )),
                )
            })?;
            if source.trim().is_empty() {
                return Err(invalid_request(
                    "image URL must not be empty",
                    Some(format!(
                        "messages[{message_index}].content[{part_index}].image_url"
                    )),
                ));
            }
            sources.push(source.to_owned());
        }
    }
    Ok(sources)
}

fn image_part_source(part: &Value) -> Option<&str> {
    let kind = part.get("type")?.as_str()?;
    if !matches!(kind, "image" | "image_url") {
        return None;
    }
    for key in ["image_url", "image"] {
        let Some(value) = part.get(key) else {
            continue;
        };
        if let Some(source) = value.as_str() {
            return Some(source);
        }
        if let Some(source) = value.get("url").and_then(Value::as_str) {
            return Some(source);
        }
    }
    None
}

pub(crate) fn stop_strings(stop: Option<&StopSpec>) -> Vec<String> {
    match stop {
        Some(StopSpec::One(value)) => vec![value.clone()],
        Some(StopSpec::Many(values)) => values.clone(),
        None => Vec::new(),
    }
}

pub(crate) fn rough_token_count(text: &str) -> usize {
    text.split_whitespace().count()
}

pub(crate) fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}
