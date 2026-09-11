//! Parses a generated assistant turn back into structured pieces: the
//! `<think>...</think>` reasoning block, the visible answer text, and any
//! `<tool_call>...</tool_call>` requests, per the protocol
//! `chat_template.jinja` documents (lines 47-53, 101-125).

use serde_json::Value;

use super::error::ChatError;
use super::types::ToolCall;

/// The result of parsing one assistant turn's raw decoded text.
#[derive(Debug, Clone, PartialEq)]
pub struct AssistantOutput {
    /// `Some` iff a `<think>...</think>` (or a bare `</think>` closing a
    /// block opened by the priming prompt) was found, even if empty.
    pub thinking: Option<String>,
    /// The visible answer text, with the thinking block and every
    /// `<tool_call>` block removed.
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
}

/// Parse a complete assistant turn. See module docs for the protocol.
pub fn parse_assistant_output(text: &str) -> Result<AssistantOutput, ChatError> {
    let (thinking, rest) = extract_thinking(text);
    let (content, tool_calls) = extract_tool_calls(&rest)?;
    Ok(AssistantOutput {
        thinking,
        content: content.trim().to_string(),
        tool_calls,
    })
}

/// Splits off a leading `<think>...</think>` (or, if the prompt already
/// primed the opening tag, a bare leading `...</think>`) from `text`.
fn extract_thinking(text: &str) -> (Option<String>, String) {
    let Some(close) = text.find("</think>") else {
        return (None, text.to_string());
    };
    let before = &text[..close];
    let after = &text[close + "</think>".len()..];
    let thinking = match before.rfind("<think>") {
        Some(open) => &before[open + "<think>".len()..],
        None => before,
    };
    (Some(thinking.trim().to_string()), after.to_string())
}

/// Extracts every `<tool_call>...</tool_call>` block from `text`, returning
/// the remaining text (blocks removed) alongside the parsed calls in order.
fn extract_tool_calls(text: &str) -> Result<(String, Vec<ToolCall>), ChatError> {
    let mut calls = Vec::new();
    let mut content = String::new();
    let mut rest = text;
    while let Some(start) = rest.find("<tool_call>") {
        content.push_str(&rest[..start]);
        let after = &rest[start + "<tool_call>".len()..];
        let Some(end) = after.find("</tool_call>") else {
            return Err(ChatError::MalformedToolCall {
                raw: after.to_string(),
                reason: "unterminated <tool_call> block".to_string(),
            });
        };
        calls.push(parse_one_tool_call_block(&after[..end])?);
        rest = &after[end + "</tool_call>".len()..];
    }
    content.push_str(rest);
    Ok((content, calls))
}

/// Parses one `<function=NAME>(<parameter=K>\nV\n</parameter>)*</function>`
/// block (the `</tool_call>` delimiters have already been stripped). Shared
/// with [`super::scanner::StreamScanner`], which buffers a `<tool_call>`
/// body whole and hands it here once the closing tag arrives.
pub(super) fn parse_one_tool_call_block(block: &str) -> Result<ToolCall, ChatError> {
    let raw = || block.to_string();
    let after_tag = block
        .find("<function=")
        .map(|i| &block[i + "<function=".len()..])
        .ok_or_else(|| ChatError::MalformedToolCall {
            raw: raw(),
            reason: "missing <function=...> tag".to_string(),
        })?;
    let name_end = after_tag
        .find('>')
        .ok_or_else(|| ChatError::MalformedToolCall {
            raw: raw(),
            reason: "unterminated <function=...> tag".to_string(),
        })?;
    let name = after_tag[..name_end].trim().to_string();

    let mut arguments = serde_json::Map::new();
    let mut rest = &after_tag[name_end + 1..];
    while let Some(p_start) = rest.find("<parameter=") {
        let after_p = &rest[p_start + "<parameter=".len()..];
        let Some(p_name_end) = after_p.find('>') else {
            break;
        };
        let pname = after_p[..p_name_end].trim().to_string();
        let value_region = &after_p[p_name_end + 1..];
        let Some(v_end) = value_region.find("</parameter>") else {
            break;
        };
        let value = parse_argument_value(value_region[..v_end].trim())?;
        arguments.insert(pname, value);
        rest = &value_region[v_end + "</parameter>".len()..];
    }

    Ok(ToolCall::new(name, Value::Object(arguments)))
}

/// The rendered protocol only JSON-encodes object/array argument values
/// (line 118); scalars are embedded as raw, unquoted text. So: recover a
/// structured value when the text parses as one, fall back to a plain
/// string otherwise — except when the text is unambiguously *trying* to be
/// a JSON object/array (starts with `{`/`[`) and fails to parse, which is
/// surfaced as a typed error carrying the raw text for the caller to retry.
fn parse_argument_value(raw: &str) -> Result<Value, ChatError> {
    let looks_like_container = raw.starts_with('{') || raw.starts_with('[');
    match serde_json::from_str::<Value>(raw) {
        Ok(value @ (Value::Object(_) | Value::Array(_))) => Ok(value),
        Ok(_) => Ok(Value::String(raw.to_string())),
        Err(source) if looks_like_container => Err(ChatError::MalformedToolCallArguments {
            raw: raw.to_string(),
            source,
        }),
        Err(_) => Ok(Value::String(raw.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_has_no_thinking_or_tool_calls() {
        let out = parse_assistant_output("just an answer").unwrap();
        assert_eq!(out.thinking, None);
        assert_eq!(out.content, "just an answer");
        assert!(out.tool_calls.is_empty());
    }

    #[test]
    fn splits_thinking_from_content() {
        let out =
            parse_assistant_output("<think>\nreasoning here\n</think>\n\nthe answer").unwrap();
        assert_eq!(out.thinking.as_deref(), Some("reasoning here"));
        assert_eq!(out.content, "the answer");
    }

    #[test]
    fn bare_closing_think_tag_from_a_primed_prompt() {
        let out = parse_assistant_output("still reasoning\n</think>\n\nanswer").unwrap();
        assert_eq!(out.thinking.as_deref(), Some("still reasoning"));
        assert_eq!(out.content, "answer");
    }

    #[test]
    fn parses_a_single_tool_call_with_string_arguments() {
        let text = "<tool_call>\n<function=get_weather>\n<parameter=location>\nPrague\n</parameter>\n</function>\n</tool_call>";
        let out = parse_assistant_output(text).unwrap();
        assert_eq!(out.content, "");
        assert_eq!(out.tool_calls.len(), 1);
        assert_eq!(out.tool_calls[0].name, "get_weather");
        assert_eq!(
            out.tool_calls[0].arguments,
            serde_json::json!({"location": "Prague"})
        );
    }

    #[test]
    fn parses_an_object_valued_argument() {
        let text = "<tool_call>\n<function=configure>\n<parameter=opts>\n{\"a\": 1, \"b\": 2}\n</parameter>\n</function>\n</tool_call>";
        let out = parse_assistant_output(text).unwrap();
        assert_eq!(
            out.tool_calls[0].arguments,
            serde_json::json!({"opts": {"a": 1, "b": 2}})
        );
    }

    #[test]
    fn malformed_container_argument_is_a_typed_error() {
        let text = "<tool_call>\n<function=configure>\n<parameter=opts>\n{not json\n</parameter>\n</function>\n</tool_call>";
        let err = parse_assistant_output(text).unwrap_err();
        assert!(matches!(err, ChatError::MalformedToolCallArguments { .. }));
    }

    #[test]
    fn unterminated_tool_call_is_a_typed_error() {
        let err = parse_assistant_output("<tool_call>\n<function=x>\n").unwrap_err();
        assert!(matches!(err, ChatError::MalformedToolCall { .. }));
    }

    #[test]
    fn missing_function_tag_is_a_typed_error() {
        let err = parse_assistant_output("<tool_call>\nnot a function\n</tool_call>").unwrap_err();
        assert!(matches!(err, ChatError::MalformedToolCall { .. }));
    }
}
