//! Hardcoded renderer for `chat_template.jinja` (Ornith-1.0-9B / qwen35).
//!
//! Not a Jinja engine: the template's control flow is fixed for this one
//! model family, so it's reproduced directly as Rust rather than
//! interpreted. Every literal string below was extracted byte-for-byte from
//! either the template file or the fixtures it produces (see
//! `rocml/tests/data/ornith_chat_fixtures.json` and
//! `rocml/tests/chat_fixtures.rs`).
//!
//! `serde_json`'s `preserve_order` feature is required: the template's
//! `tojson` filter walks caller-provided JSON objects (tool specs, tool-call
//! arguments) in their original key order, and only an order-preserving
//! `Value::Object` reproduces that byte-for-byte.

use serde_json::{Map, Value};

use super::error::ChatError;
use super::types::{Message, RenderOpts, Role, Tool, ToolCall};

// Extracted verbatim from chat_template.jinja lines 47 and 53.
const TOOLS_HEADER: &str = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
const TOOLS_FOOTER: &str = "</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

/// Render `messages`/`tools` into the exact prompt string
/// `AutoTokenizer::apply_chat_template_with_options` produces for this
/// template (see fixtures for byte-identical ground truth).
pub fn render(messages: &[Message], tools: &[Tool], opts: RenderOpts) -> Result<String, ChatError> {
    if messages.is_empty() {
        return Err(ChatError::NoMessages);
    }
    validate_system_position(messages)?;
    validate_user_query(messages)?;

    let mut out = String::new();
    let has_tools = !tools.is_empty();
    if has_tools {
        render_tools_header(&mut out, messages, tools);
    } else if messages[0].role == Role::System {
        out.push_str("<|im_start|>system\n");
        out.push_str(messages[0].content.trim());
        out.push_str("<|im_end|>\n");
    }

    for (index, message) in messages.iter().enumerate() {
        match message.role {
            // The system message's body was already emitted above (or
            // there's no system message); the main loop only ever needs to
            // check its position, which `validate_system_position` did.
            Role::System => {}
            Role::User => render_user_turn(&mut out, message),
            Role::Assistant => render_assistant_turn(&mut out, message, opts),
            Role::Tool => render_tool_turn(&mut out, messages, index),
        }
    }

    if opts.add_generation_prompt {
        out.push_str("<|im_start|>assistant\n");
        if opts.enable_thinking == Some(false) {
            out.push_str("<think>\n\n</think>\n\n");
        } else {
            out.push_str("<think>\n");
        }
    }
    Ok(out)
}

fn render_tools_header(out: &mut String, messages: &[Message], tools: &[Tool]) {
    out.push_str("<|im_start|>system\n");
    out.push_str(TOOLS_HEADER);
    for tool in tools {
        out.push('\n');
        out.push_str(&tojson(&tool_wire_value(tool)));
    }
    out.push('\n');
    out.push_str(TOOLS_FOOTER);
    if messages[0].role == Role::System {
        let content = messages[0].content.trim();
        if !content.is_empty() {
            out.push_str("\n\n");
            out.push_str(content);
        }
    }
    out.push_str("<|im_end|>\n");
}

/// Builds the `{"type": "function", "function": {name, description,
/// parameters}}` shape the template's `tool | tojson` expects — deliberately
/// independent of `Tool`'s own (flat) `Serialize` impl.
fn tool_wire_value(tool: &Tool) -> Value {
    let mut function = Map::new();
    function.insert("name".to_string(), Value::String(tool.name.clone()));
    function.insert(
        "description".to_string(),
        Value::String(tool.description.clone()),
    );
    function.insert("parameters".to_string(), tool.parameters.clone());

    let mut wire = Map::new();
    wire.insert("type".to_string(), Value::String("function".to_string()));
    wire.insert("function".to_string(), Value::Object(function));
    Value::Object(wire)
}

fn render_user_turn(out: &mut String, message: &Message) {
    out.push_str("<|im_start|>user\n");
    out.push_str(message.content.trim());
    out.push_str("<|im_end|>\n");
}

fn render_assistant_turn(out: &mut String, message: &Message, opts: RenderOpts) {
    let (reasoning, content) = split_reasoning(message);
    // Issue #9: every message in `messages` is, by construction, a
    // *completed* prior turn — the turn currently being generated is never
    // part of this slice, it's produced after `render` returns. So
    // "history" here means "every assistant turn", and the training
    // convention (see `RenderOpts::keep_history_reasoning`'s doc comment)
    // says none of them should carry their old thinking back in.
    let reasoning = if opts.keep_history_reasoning {
        reasoning
    } else {
        String::new()
    };
    out.push_str("<|im_start|>assistant\n<think>\n");
    out.push_str(reasoning.trim());
    out.push_str("\n</think>\n\n");
    out.push_str(&content);
    if !message.tool_calls.is_empty() {
        render_tool_calls(out, &content, &message.tool_calls);
    }
    out.push_str("<|im_end|>\n");
}

/// Mirrors the template's `reasoning_content` handling (lines 90-99): an
/// explicit field wins outright; otherwise a `<think>...</think>` block
/// already embedded in `content` (the shape a stored assistant turn takes)
/// is split out of it.
fn split_reasoning(message: &Message) -> (String, String) {
    let base = message.content.trim().to_string();
    if let Some(explicit) = &message.reasoning_content {
        return (explicit.clone(), base);
    }
    let Some(first_close) = base.find("</think>") else {
        return (String::new(), base);
    };
    let before_first_close = base[..first_close].trim_end_matches('\n');
    let reasoning_start = before_first_close
        .rfind("<think>")
        .map(|i| i + "<think>".len())
        .unwrap_or(0);
    let reasoning = before_first_close[reasoning_start..]
        .trim_start_matches('\n')
        .to_string();

    // `content.split('</think>')[-1]`: text after the LAST close tag, not
    // necessarily the same occurrence used for `reasoning`'s search above.
    let last_close = base.rfind("</think>").unwrap_or(first_close);
    let content = base[last_close + "</think>".len()..]
        .trim_start_matches('\n')
        .to_string();
    (reasoning, content)
}

fn render_tool_calls(out: &mut String, content: &str, tool_calls: &[ToolCall]) {
    for (i, call) in tool_calls.iter().enumerate() {
        if i == 0 {
            if content.trim().is_empty() {
                out.push_str("<tool_call>\n<function=");
            } else {
                out.push_str("\n\n<tool_call>\n<function=");
            }
        } else {
            out.push_str("\n<tool_call>\n<function=");
        }
        out.push_str(&call.name);
        out.push_str(">\n");
        if let Some(arguments) = call.arguments.as_object() {
            for (name, value) in arguments {
                out.push_str("<parameter=");
                out.push_str(name);
                out.push_str(">\n");
                out.push_str(&render_argument_value(value));
                out.push_str("\n</parameter>\n");
            }
        }
        out.push_str("</function>\n</tool_call>");
    }
}

/// `args_value | tojson if mapping/non-string-sequence else args_value |
/// string` (line 118). The scalar branch is Jinja's `string` filter applied
/// to whatever native type the caller's JSON produced — Python `str()`
/// semantics for bool/null are not exercised by any fixture (every fixture
/// tool argument is a string) and are reproduced here from the template's
/// documented behavior rather than verified byte-for-byte.
fn render_argument_value(value: &Value) -> String {
    match value {
        Value::Object(_) | Value::Array(_) => tojson(value),
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => (if *b { "True" } else { "False" }).to_string(),
        Value::Null => "None".to_string(),
    }
}

fn render_tool_turn(out: &mut String, messages: &[Message], index: usize) {
    // `loop.previtem and loop.previtem.role != "tool"` (line 128): no
    // previous item at all (index 0) is falsy, so a leading tool message
    // never opens a wrapper — matched here via the `index > 0` guard.
    let opens_wrapper = index > 0 && messages[index - 1].role != Role::Tool;
    if opens_wrapper {
        out.push_str("<|im_start|>user");
    }
    out.push_str("\n<tool_response>\n");
    out.push_str(messages[index].content.trim());
    out.push_str("\n</tool_response>");

    let is_last = index == messages.len() - 1;
    let next_is_tool = messages
        .get(index + 1)
        .is_some_and(|m| m.role == Role::Tool);
    if is_last || !next_is_tool {
        out.push_str("<|im_end|>\n");
    }
}

/// `raise_exception('System message must be at the beginning.')` (line 85):
/// any system message after index 0 aborts rendering.
fn validate_system_position(messages: &[Message]) -> Result<(), ChatError> {
    if messages.iter().skip(1).any(|m| m.role == Role::System) {
        return Err(ChatError::SystemMessageNotFirst);
    }
    Ok(())
}

/// `raise_exception('No user query found in messages.')` (lines 67-80): at
/// least one `user` turn must exist whose content isn't itself a
/// `<tool_response>...</tool_response>`-wrapped payload.
fn validate_user_query(messages: &[Message]) -> Result<(), ChatError> {
    let has_real_query = messages.iter().any(|m| {
        m.role == Role::User && {
            let c = m.content.trim();
            !(c.starts_with("<tool_response>") && c.ends_with("</tool_response>"))
        }
    });
    if has_real_query {
        Ok(())
    } else {
        Err(ChatError::NoUserQuery)
    }
}

/// Python-`json.dumps`-style rendering of the `tojson` filter: `", "` /
/// `": "` separators (not serde_json's compact defaults) plus HTML-safe
/// escaping of `< > & '`, matching Crane's `tojson` filter implementation
/// that produced the fixtures this module is verified against.
pub(super) fn tojson(value: &Value) -> String {
    struct PySeparators;
    impl serde_json::ser::Formatter for PySeparators {
        fn begin_array_value<W: ?Sized + std::io::Write>(
            &mut self,
            w: &mut W,
            first: bool,
        ) -> std::io::Result<()> {
            if first {
                Ok(())
            } else {
                w.write_all(b", ")
            }
        }
        fn begin_object_key<W: ?Sized + std::io::Write>(
            &mut self,
            w: &mut W,
            first: bool,
        ) -> std::io::Result<()> {
            if first {
                Ok(())
            } else {
                w.write_all(b", ")
            }
        }
        fn begin_object_value<W: ?Sized + std::io::Write>(
            &mut self,
            w: &mut W,
        ) -> std::io::Result<()> {
            w.write_all(b": ")
        }
    }

    let mut buf = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(&mut buf, PySeparators);
    // `Value`'s `Serialize` impl cannot fail for a well-formed JSON tree.
    if serde::Serialize::serialize(value, &mut serializer).is_err() {
        return String::new();
    }
    let raw = String::from_utf8(buf).unwrap_or_default();

    let mut escaped = String::with_capacity(raw.len());
    for c in raw.chars() {
        match c {
            '<' => escaped.push_str("\\u003c"),
            '>' => escaped.push_str("\\u003e"),
            '&' => escaped.push_str("\\u0026"),
            '\'' => escaped.push_str("\\u0027"),
            _ => escaped.push(c),
        }
    }
    escaped
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_messages_is_an_error() {
        assert!(matches!(
            render(&[], &[], RenderOpts::default()),
            Err(ChatError::NoMessages)
        ));
    }

    #[test]
    fn system_message_must_be_first() {
        let messages = vec![Message::user("hi"), Message::system("late")];
        assert!(matches!(
            render(&messages, &[], RenderOpts::default()),
            Err(ChatError::SystemMessageNotFirst)
        ));
    }

    #[test]
    fn no_user_query_is_rejected() {
        let messages = vec![Message::user("<tool_response>\nx\n</tool_response>")];
        assert!(matches!(
            render(&messages, &[], RenderOpts::default()),
            Err(ChatError::NoUserQuery)
        ));
    }

    #[test]
    fn tojson_uses_python_separators_and_html_escaping() {
        let value = serde_json::json!({"a": "<x>&'y'", "b": [1, 2]});
        assert_eq!(
            tojson(&value),
            "{\"a\": \"\\u003cx\\u003e\\u0026\\u0027y\\u0027\", \"b\": [1, 2]}"
        );
    }

    #[test]
    fn simple_system_user_render() {
        let messages = vec![Message::system("sys"), Message::user("hello")];
        let out = render(
            &messages,
            &[],
            RenderOpts {
                add_generation_prompt: true,
                enable_thinking: None,
                keep_history_reasoning: false,
            },
        )
        .unwrap();
        assert_eq!(
            out,
            "<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
    }

    // Issue #9's tool-result-hygiene tests (empty body, multiple consecutive
    // results) live in `rocml/tests/chat_fixtures.rs` instead of here — this
    // module is already at the 400-line cap, and both tests only exercise
    // the public `render` API, no private internals.
}
