//! Wire types for the OpenAI `chat/completions` surface: just enough of the
//! shape to round-trip messages, tool calls, and streaming deltas.
//! `reasoning_content` on both input and output is the de-facto extension
//! several OpenAI-compatible servers already use for `<think>` content.

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<ChatMessageIn>,
    #[serde(default)]
    pub tools: Vec<ToolIn>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub max_tokens: Option<usize>,
    pub seed: Option<u64>,
    #[serde(default)]
    pub stop: Option<StopSequences>,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum StopSequences {
    One(String),
    Many(Vec<String>),
}

impl StopSequences {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            Self::One(s) => vec![s],
            Self::Many(v) => v,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct ChatMessageIn {
    pub role: String,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<ToolCallIn>,
    /// Accepted but not otherwise used: the rendered protocol
    /// (`rocml::chat::render`) doesn't address tool results by id, only by
    /// their position in the transcript.
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ToolCallIn {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(rename = "type", default)]
    pub kind: Option<String>,
    pub function: FunctionCallIn,
}

#[derive(Debug, Deserialize)]
pub struct FunctionCallIn {
    pub name: String,
    /// OpenAI encodes tool-call arguments as a JSON string, not a nested
    /// object — this field is that string, parsed by `mapping::map_messages`.
    pub arguments: String,
}

#[derive(Debug, Deserialize)]
pub struct ToolIn {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionSpecIn,
}

#[derive(Debug, Deserialize)]
pub struct FunctionSpecIn {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub parameters: Value,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChoiceOut>,
    pub usage: UsageOut,
}

#[derive(Debug, Serialize)]
pub struct ChoiceOut {
    pub index: u32,
    pub message: ChatMessageOut,
    pub finish_reason: &'static str,
}

#[derive(Debug, Serialize, Default)]
pub struct ChatMessageOut {
    pub role: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCallOut>,
}

#[derive(Debug, Serialize, Clone)]
pub struct ToolCallOut {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub function: FunctionCallOut,
}

#[derive(Debug, Serialize, Clone)]
pub struct FunctionCallOut {
    pub name: String,
    /// A JSON-encoded string, mirroring `FunctionCallIn::arguments`.
    pub arguments: String,
}

#[derive(Debug, Serialize)]
pub struct UsageOut {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChoiceChunk>,
}

#[derive(Debug, Serialize)]
pub struct ChoiceChunk {
    pub index: u32,
    pub delta: DeltaOut,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize, Default)]
pub struct DeltaOut {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCallDeltaOut>,
}

#[derive(Debug, Serialize, Clone)]
pub struct ToolCallDeltaOut {
    pub index: usize,
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub function: FunctionCallOut,
}

pub fn chunk(
    id: &str,
    created: u64,
    model: &str,
    delta: DeltaOut,
    finish_reason: Option<&'static str>,
) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        choices: vec![ChoiceChunk {
            index: 0,
            delta,
            finish_reason,
        }],
    }
}
