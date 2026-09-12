//! Wire types for the Ornith chat protocol: messages, tool specs, and tool
//! calls. These are the Rust-side counterpart of the OpenAI-style JSON the
//! future server milestone maps requests onto — kept flat and serde-derived
//! rather than mirroring the template's nested `{type: "function", function:
//! {...}}` shape, which [`super::render`] builds on demand instead.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A conversation turn's author, per the `# Tools` / `<tool_call>` protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// One conversation turn.
///
/// `reasoning_content` mirrors the template's own optional field: when
/// `None`, [`super::render`] falls back to splitting an already-embedded
/// `<think>...</think>` block out of `content` (the shape history turns take
/// when an assistant reply is stored verbatim), matching the template's
/// `reasoning_content is string` branch exactly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

impl Message {
    fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            tool_calls: Vec::new(),
            reasoning_content: None,
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::new(Role::System, content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new(Role::User, content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(Role::Assistant, content)
    }

    /// A `tool` turn carrying one tool's result back to the model.
    pub fn tool_response(content: impl Into<String>) -> Self {
        Self::new(Role::Tool, content)
    }

    pub fn with_tool_calls(mut self, tool_calls: Vec<ToolCall>) -> Self {
        self.tool_calls = tool_calls;
        self
    }

    pub fn with_reasoning(mut self, reasoning: impl Into<String>) -> Self {
        self.reasoning_content = Some(reasoning.into());
        self
    }
}

/// A function call the assistant requested, in `<function=NAME><parameter=...>`
/// terms. `arguments` is always an object in the rendered protocol; other
/// JSON shapes render as zero parameters (see [`super::render`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}

impl ToolCall {
    pub fn new(name: impl Into<String>, arguments: Value) -> Self {
        Self {
            name: name.into(),
            arguments,
        }
    }
}

/// A function the model may call, declared in the `# Tools` system block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

impl Tool {
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }
}

/// Options that gate the tail of the rendered prompt.
#[derive(Debug, Clone, Copy, Default)]
pub struct RenderOpts {
    /// Append the `<|im_start|>assistant\n<think>...` priming tail.
    pub add_generation_prompt: bool,
    /// `None` leaves the template's own default (reasoning ON) in place;
    /// `Some(false)` pre-closes the think block (reasoning OFF); `Some(true)`
    /// is equivalent to `None` for this template.
    pub enable_thinking: Option<bool>,
    /// Issue #9: the raw `chat_template.jinja` does *not* strip a history
    /// assistant turn's `reasoning_content` on its own — whatever the
    /// caller passes for a past turn (an explicit `reasoning_content`
    /// field, or a `<think>...</think>` block embedded in `content`) is
    /// rendered verbatim, every turn, not just the newest one (verified
    /// directly against the template — see
    /// `rocml/tests/chat_fixtures.rs`'s
    /// `history_reasoning_is_stripped_by_default_but_available_via_opt_in`).
    /// But the Qwen3-family training recipe Ornith descends from removes
    /// prior-turn thinking from history at training time — feeding a past
    /// `<think>` block back as context is out of distribution and was
    /// observed to make the model loop, re-reasoning about an already
    /// resolved point forever. So `false` (the default, matching the
    /// training convention) strips every history assistant turn's
    /// reasoning; `true` opts back into raw-template-faithful behavior for
    /// a caller that specifically needs byte-for-byte template parity.
    pub keep_history_reasoning: bool,
}
