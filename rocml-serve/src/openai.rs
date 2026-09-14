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
    /// OpenAI's `stream_options: {"include_usage": true}` — when set on a
    /// streaming request, every chunk carries a `usage` key (`null` until
    /// the final one) instead of omitting it entirely. Ignored for a
    /// non-streaming request (the JSON body always reports usage). Unknown
    /// sibling fields inside the object are silently ignored — `serde`'s
    /// default (no `deny_unknown_fields`) already does this, matching
    /// OpenAI's own forward-compatible posture.
    #[serde(default)]
    pub stream_options: Option<StreamOptionsIn>,
}

#[derive(Debug, Clone, Copy, Default, Deserialize)]
pub struct StreamOptionsIn {
    #[serde(default)]
    pub include_usage: bool,
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
    /// Always present (OpenAI includes this unconditionally nowadays, even
    /// when nothing was cached) — see [`PromptTokensDetailsOut`].
    pub prompt_tokens_details: PromptTokensDetailsOut,
}

#[derive(Debug, Serialize, Clone, Copy, Default)]
pub struct PromptTokensDetailsOut {
    /// Prompt tokens restored from a conversation-prefix snapshot rather
    /// than re-run through prefill this request — `TurnStats::cached_tokens`
    /// in `worker.rs`, sourced from `rocml::snapshot::turn::TurnOutcome::reused_prefix`.
    /// `0` on a snapshot miss (or when the snapshot layer is disabled/not
    /// applicable), never omitted.
    pub cached_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChoiceChunk>,
    /// `None` (field omitted) unless the request set `stream_options:
    /// {"include_usage": true}`, in which case every chunk carries this key
    /// — `null` here, the real [`UsageOut`] only on the dedicated trailing
    /// chunk `usage_chunk` builds. `Option<()>` rather than
    /// `Option<UsageOut>` because every non-final chunk's usage value is
    /// always exactly JSON `null`, never a partial object.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<()>,
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
    include_usage: bool,
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
        usage: include_usage.then_some(()),
    }
}

/// The extra trailing chunk `stream_options: {"include_usage": true}` asks
/// for: empty `choices` (OpenAI's own shape for this chunk) and the full
/// [`UsageOut`], sent after the `finish_reason` chunk and before `[DONE]`.
#[derive(Debug, Serialize)]
pub struct ChatCompletionUsageChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChoiceChunk>,
    pub usage: UsageOut,
}

pub fn usage_chunk(
    id: &str,
    created: u64,
    model: &str,
    usage: UsageOut,
) -> ChatCompletionUsageChunk {
    ChatCompletionUsageChunk {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        choices: Vec::new(),
        usage,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_out_always_serializes_prompt_tokens_details() {
        let usage = UsageOut {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
            prompt_tokens_details: PromptTokensDetailsOut { cached_tokens: 0 },
        };
        let v: Value = serde_json::to_value(&usage).unwrap();
        assert_eq!(v["prompt_tokens"], 10);
        assert_eq!(v["prompt_tokens_details"]["cached_tokens"], 0);
    }

    #[test]
    fn usage_out_reports_nonzero_cached_tokens() {
        let usage = UsageOut {
            prompt_tokens: 100,
            completion_tokens: 5,
            total_tokens: 105,
            prompt_tokens_details: PromptTokensDetailsOut { cached_tokens: 40 },
        };
        let v: Value = serde_json::to_value(&usage).unwrap();
        assert_eq!(v["prompt_tokens_details"]["cached_tokens"], 40);
        // Cached is a subset detail — prompt_tokens stays the full count.
        assert_eq!(v["prompt_tokens"], 100);
    }

    /// Default (no `stream_options`) chunks omit the `usage` key entirely —
    /// this is the "no behavior change" guarantee for existing clients.
    #[test]
    fn chunk_without_include_usage_omits_usage_key() {
        let c = chunk("id1", 0, "model", DeltaOut::default(), None, false);
        let v: Value = serde_json::to_value(&c).unwrap();
        assert!(
            v.as_object().unwrap().get("usage").is_none(),
            "usage key should be absent: {v}"
        );
    }

    /// `stream_options: {"include_usage": true}` makes every regular chunk
    /// carry `"usage": null`.
    #[test]
    fn chunk_with_include_usage_carries_null_usage() {
        let c = chunk("id1", 0, "model", DeltaOut::default(), None, true);
        let v: Value = serde_json::to_value(&c).unwrap();
        assert!(v.as_object().unwrap().contains_key("usage"));
        assert!(v["usage"].is_null());
    }

    /// The dedicated trailing chunk has empty `choices` and the full usage
    /// object.
    #[test]
    fn usage_chunk_has_empty_choices_and_full_usage() {
        let usage = UsageOut {
            prompt_tokens: 8,
            completion_tokens: 2,
            total_tokens: 10,
            prompt_tokens_details: PromptTokensDetailsOut { cached_tokens: 3 },
        };
        let c = usage_chunk("id1", 0, "model", usage);
        let v: Value = serde_json::to_value(&c).unwrap();
        assert_eq!(v["choices"].as_array().unwrap().len(), 0);
        assert_eq!(v["usage"]["prompt_tokens_details"]["cached_tokens"], 3);
    }

    #[test]
    fn stream_options_include_usage_parses_true() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"messages":[],"stream":true,"stream_options":{"include_usage":true}}"#,
        )
        .unwrap();
        assert!(req.stream_options.unwrap().include_usage);
    }

    #[test]
    fn stream_options_absent_by_default() {
        let req: ChatCompletionRequest =
            serde_json::from_str(r#"{"messages":[],"stream":true}"#).unwrap();
        assert!(req.stream_options.is_none());
    }

    /// Unknown sibling fields inside `stream_options` are ignored, not
    /// rejected — no `deny_unknown_fields` on `StreamOptionsIn`.
    #[test]
    fn stream_options_ignores_unknown_fields() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"messages":[],"stream":true,"stream_options":{"include_usage":true,"some_future_field":42}}"#,
        )
        .unwrap();
        assert!(req.stream_options.unwrap().include_usage);
    }
}
