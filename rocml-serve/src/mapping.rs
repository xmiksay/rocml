//! Translates between the OpenAI wire shapes (`openai` module) and
//! `rocml::chat`'s types, plus the shared "apply one scanner event to an
//! in-progress assistant turn" logic both the streaming and non-streaming
//! response paths fold over.

use rocml::chat::{Message, Role, ScanEvent, Tool, ToolCall};
use rocml::SamplingParams;

use crate::error::ApiError;
use crate::openai::{ChatCompletionRequest, ChatMessageIn, ToolIn};

pub fn map_messages(input: &[ChatMessageIn]) -> Result<Vec<Message>, ApiError> {
    input.iter().map(map_message).collect()
}

fn map_message(m: &ChatMessageIn) -> Result<Message, ApiError> {
    let role = match m.role.as_str() {
        "system" => Role::System,
        "user" => Role::User,
        "assistant" => Role::Assistant,
        "tool" => Role::Tool,
        other => {
            return Err(ApiError::bad_request(format!(
                "unknown message role {other:?}"
            )))
        }
    };
    let mut tool_calls = Vec::with_capacity(m.tool_calls.len());
    for tc in &m.tool_calls {
        let arguments = serde_json::from_str(&tc.function.arguments).map_err(|e| {
            ApiError::bad_request(format!(
                "tool_calls[].function.arguments is not valid JSON: {e}"
            ))
        })?;
        tool_calls.push(ToolCall::new(tc.function.name.clone(), arguments));
    }
    Ok(Message {
        role,
        content: m.content.clone().unwrap_or_default(),
        tool_calls,
        reasoning_content: m.reasoning_content.clone(),
    })
}

pub fn map_tools(input: &[ToolIn]) -> Vec<Tool> {
    input
        .iter()
        .map(|t| {
            Tool::new(
                t.function.name.clone(),
                t.function.description.clone(),
                t.function.parameters.clone(),
            )
        })
        .collect()
}

pub fn map_sampling(req: &ChatCompletionRequest) -> SamplingParams {
    SamplingParams {
        // The engine's own default is greedy; that's also the safer choice
        // for tool-calling reliability, so an absent `temperature` stays
        // greedy rather than adopting OpenAI's own default of 1.0.
        temperature: req.temperature.unwrap_or(0.0),
        top_k: req.top_k,
        top_p: req.top_p,
        seed: req.seed.unwrap_or(0),
        ..SamplingParams::default()
    }
}

pub fn stop_strings(req: &ChatCompletionRequest) -> Vec<String> {
    req.stop
        .as_ref()
        .map(|s| s.clone().into_vec())
        .unwrap_or_default()
}

/// One in-progress assistant turn's accumulated pieces, built up by folding
/// `ScanEvent`s from a `StreamScanner` as generation output arrives — shared
/// by both the non-streaming (fold to completion, then respond once) and
/// streaming (fold and emit a delta per event) response paths.
#[derive(Debug, Default)]
pub struct AssistantAccumulator {
    pub content: String,
    pub thinking: String,
    pub tool_calls: Vec<ToolCall>,
}

impl AssistantAccumulator {
    pub fn apply(&mut self, event: ScanEvent) {
        match event {
            ScanEvent::TextDelta(s) => self.content.push_str(&s),
            ScanEvent::ThinkingDelta(s) => self.thinking.push_str(&s),
            ScanEvent::ToolCallStarted => {}
            ScanEvent::ToolCallComplete(call) => self.tool_calls.push(call),
        }
    }

    pub fn finish_reason(&self, generated_tokens: usize, max_new_tokens: usize) -> &'static str {
        if !self.tool_calls.is_empty() {
            "tool_calls"
        } else if generated_tokens >= max_new_tokens {
            "length"
        } else {
            "stop"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::{FunctionCallIn, FunctionSpecIn, ToolCallIn};
    use serde_json::json;

    #[test]
    fn maps_basic_roles_and_content() {
        let input = vec![
            ChatMessageIn {
                role: "system".to_string(),
                content: Some("be terse".to_string()),
                tool_calls: vec![],
                tool_call_id: None,
                reasoning_content: None,
            },
            ChatMessageIn {
                role: "user".to_string(),
                content: Some("hi".to_string()),
                tool_calls: vec![],
                tool_call_id: None,
                reasoning_content: None,
            },
        ];
        let mapped = map_messages(&input).unwrap();
        assert_eq!(mapped[0].role, Role::System);
        assert_eq!(mapped[0].content, "be terse");
        assert_eq!(mapped[1].role, Role::User);
    }

    #[test]
    fn unknown_role_is_a_bad_request() {
        let input = vec![ChatMessageIn {
            role: "narrator".to_string(),
            content: Some("x".to_string()),
            tool_calls: vec![],
            tool_call_id: None,
            reasoning_content: None,
        }];
        assert!(matches!(map_messages(&input), Err(ApiError::BadRequest(_))));
    }

    #[test]
    fn maps_tool_result_turn() {
        let input = vec![ChatMessageIn {
            role: "tool".to_string(),
            content: Some("71F and sunny".to_string()),
            tool_calls: vec![],
            tool_call_id: Some("call_1".to_string()),
            reasoning_content: None,
        }];
        let mapped = map_messages(&input).unwrap();
        assert_eq!(mapped[0].role, Role::Tool);
        assert_eq!(mapped[0].content, "71F and sunny");
    }

    #[test]
    fn round_trips_assistant_tool_calls() {
        let input = vec![ChatMessageIn {
            role: "assistant".to_string(),
            content: Some(String::new()),
            tool_calls: vec![ToolCallIn {
                id: Some("call_1".to_string()),
                kind: Some("function".to_string()),
                function: FunctionCallIn {
                    name: "get_weather".to_string(),
                    arguments: r#"{"location": "Prague"}"#.to_string(),
                },
            }],
            tool_call_id: None,
            reasoning_content: None,
        }];
        let mapped = map_messages(&input).unwrap();
        assert_eq!(mapped[0].tool_calls.len(), 1);
        assert_eq!(mapped[0].tool_calls[0].name, "get_weather");
        assert_eq!(
            mapped[0].tool_calls[0].arguments,
            json!({"location": "Prague"})
        );
    }

    #[test]
    fn malformed_tool_call_arguments_json_is_a_bad_request() {
        let input = vec![ChatMessageIn {
            role: "assistant".to_string(),
            content: None,
            tool_calls: vec![ToolCallIn {
                id: None,
                kind: None,
                function: FunctionCallIn {
                    name: "f".to_string(),
                    arguments: "{not json".to_string(),
                },
            }],
            tool_call_id: None,
            reasoning_content: None,
        }];
        assert!(matches!(map_messages(&input), Err(ApiError::BadRequest(_))));
    }

    #[test]
    fn maps_tool_spec_from_openai_function_shape() {
        let input = vec![ToolIn {
            kind: "function".to_string(),
            function: FunctionSpecIn {
                name: "get_weather".to_string(),
                description: "gets the weather".to_string(),
                parameters: json!({"type": "object", "properties": {}}),
            },
        }];
        let tools = map_tools(&input);
        assert_eq!(tools[0].name, "get_weather");
        assert_eq!(tools[0].description, "gets the weather");
    }

    #[test]
    fn finish_reason_prefers_tool_calls_then_length_then_stop() {
        let mut acc = AssistantAccumulator::default();
        assert_eq!(acc.finish_reason(5, 10), "stop");
        assert_eq!(acc.finish_reason(10, 10), "length");
        acc.tool_calls.push(ToolCall::new("f", json!({})));
        assert_eq!(acc.finish_reason(10, 10), "tool_calls");
    }
}
