//! Chat-template rendering and assistant-output parsing for Ornith-1.0-9B
//! (qwen35 family): a from-scratch Rust port of `chat_template.jinja`,
//! byte-identical to the official template (see
//! `rocml/tests/data/ornith_chat_fixtures.json` for the HF-rendered ground
//! truth and `rocml/tests/chat_fixtures.rs` for the fixture-parity tests).
//!
//! This is host-side string work only — no GPU, no model weights.

mod error;
mod parser;
mod render;
mod scanner;
mod types;

pub use error::ChatError;
pub use parser::{parse_assistant_output, AssistantOutput};
pub use render::render;
pub use scanner::{ScanEvent, StreamScanner};
pub use types::{Message, RenderOpts, Role, Tool, ToolCall};
