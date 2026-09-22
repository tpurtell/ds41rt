use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::CompletionMetrics;

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub service: &'static str,
    pub backend: &'static str,
    pub transport: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ModelList {
    pub object: &'static str,
    pub data: Vec<ModelInfo>,
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: &'static str,
    pub owned_by: &'static str,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<ChatCompletionStreamOptions>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    /// vLLM-compatible generation floor. Stop tokens are ignored until this
    /// many output tokens have been committed.
    #[serde(default)]
    pub min_tokens: Option<usize>,
    /// vLLM-compatible benchmark/diagnostic extension: ignore model stop
    /// tokens and continue until the requested output budget is exhausted.
    #[serde(default)]
    pub ignore_eos: Option<bool>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    /// vLLM-compatible minimum-probability filter. The production
    /// `serve-native` path resolves this from the raw body; the legacy
    /// `real-ds4-full` sampler does not implement it, so a nonzero value is
    /// rejected by `validate_request` rather than silently ignored.
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub stop: Option<StopSpec>,
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    #[serde(default)]
    pub tools: Option<Vec<ChatTool>>,
    #[serde(default)]
    pub tool_choice: Option<ToolChoice>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    /// DS41RT extension: enable automatic token-level structural assistance for
    /// ordinary (non-strict) tool calls. Defaults to true. Explicit strict
    /// tool schemas remain constrained regardless of this setting.
    #[serde(default)]
    pub tool_decoding_assistance: Option<bool>,
    #[serde(default)]
    pub thinking: Option<ThinkingConfig>,
    #[serde(default)]
    pub enable_thinking: Option<bool>,
    #[serde(default)]
    pub reasoning_effort: Option<Value>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ChatCompletionStreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<Value>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default, alias = "reasoning", alias = "reasoning_text")]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub tools: Option<Vec<ChatTool>>,
    #[serde(default)]
    pub response_format: Option<Value>,
    #[serde(default)]
    pub task: Option<String>,
    #[serde(default)]
    pub wo_eos: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ThinkingConfig {
    #[serde(rename = "type")]
    pub thinking_type: String,
    #[serde(default)]
    pub clear_thinking: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum StopSpec {
    One(String),
    Many(Vec<String>),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema { json_schema: JsonSchemaFormat },
}

#[derive(Debug, Clone, Deserialize)]
pub struct JsonSchemaFormat {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    pub schema: Value,
    #[serde(default)]
    pub strict: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatTool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ChatFunction,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatFunction {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Option<Value>,
    #[serde(default)]
    pub strict: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    Mode(String),
    Specific {
        #[serde(rename = "type")]
        tool_type: String,
        function: ToolChoiceFunction,
    },
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolChoiceFunction {
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCall {
    #[serde(default)]
    pub id: String,
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCallFunction {
    pub name: String,
    #[serde(
        default = "empty_json_object",
        deserialize_with = "deserialize_tool_arguments"
    )]
    pub arguments: String,
}

fn empty_json_object() -> String {
    "{}".to_owned()
}

fn deserialize_tool_arguments<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<Value>::deserialize(deserializer)?;
    match value {
        None => Ok(empty_json_object()),
        Some(Value::String(arguments)) => Ok(arguments),
        Some(Value::Object(arguments)) => {
            serde_json::to_string(&arguments).map_err(serde::de::Error::custom)
        }
        Some(other) => Err(serde::de::Error::custom(format!(
            "tool call arguments must be a JSON object string, object, or null; got {other}"
        ))),
    }
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
    pub metrics: CompletionMetrics,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: AssistantMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct AssistantMessage {
    pub role: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    pub prompt_tokens_details: PromptTokensDetails,
    pub completion_tokens_details: CompletionTokensDetails,
}

impl Usage {
    pub(crate) fn from_metrics(metrics: &CompletionMetrics) -> Self {
        Self {
            prompt_tokens: metrics.prompt_tokens,
            completion_tokens: metrics.output_tokens,
            total_tokens: metrics.prompt_tokens + metrics.output_tokens,
            prompt_tokens_details: PromptTokensDetails {
                cached_tokens: metrics.cached_prompt_tokens,
            },
            completion_tokens_details: CompletionTokensDetails {
                reasoning_tokens: metrics.reasoning_tokens,
            },
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PromptTokensDetails {
    pub cached_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompletionTokensDetails {
    pub reasoning_tokens: usize,
}
