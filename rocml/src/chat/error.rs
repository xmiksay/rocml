//! Errors surfaced by chat-template rendering and assistant-output parsing.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ChatError {
    #[error("no messages provided")]
    NoMessages,

    #[error("system message must be at the beginning")]
    SystemMessageNotFirst,

    #[error("no user query found in messages")]
    NoUserQuery,

    /// A `<tool_call>` block's `<function=...>` name or parameter JSON body
    /// could not be recovered from the raw text. Carries the raw block so
    /// the caller can decide whether to retry generation.
    #[error("malformed tool call block: {reason}")]
    MalformedToolCall { raw: String, reason: String },

    /// A `<tool_call>` parameter value failed to parse as JSON where the
    /// protocol requires structured data (arrays/objects).
    #[error("malformed tool call arguments in {raw:?}: {source}")]
    MalformedToolCallArguments {
        raw: String,
        #[source]
        source: serde_json::Error,
    },
}
