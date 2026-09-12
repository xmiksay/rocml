//! Pure, GPU-free scoring logic: given a parsed [`AssistantOutput`] (from
//! `rocml::chat::parse_assistant_output`) and a scenario's expectation,
//! decide pass/fail and a one-line diagnosis. Kept separate from
//! `runner` (which drives the model) so every rule here is unit-testable
//! against canned outputs.

use rocml::chat::AssistantOutput;
use serde_json::{Map, Value};

use super::scenario::{ExpectedToolCall, SecondTurnExpectation};

pub struct Verdict {
    pub pass: bool,
    pub detail: String,
}

impl Verdict {
    fn pass() -> Self {
        Self {
            pass: true,
            detail: "ok".to_string(),
        }
    }

    fn fail(detail: impl Into<String>) -> Self {
        Self {
            pass: false,
            detail: detail.into(),
        }
    }
}

/// `tool_choice` (and a `multi_turn` first turn / tool-call second turn):
/// exactly one tool call, matching name and expected arguments.
pub fn score_tool_choice(expected: &ExpectedToolCall, output: &AssistantOutput) -> Verdict {
    if output.tool_calls.len() != 1 {
        return Verdict::fail(format!(
            "expected exactly 1 tool call, got {}",
            output.tool_calls.len()
        ));
    }
    let call = &output.tool_calls[0];
    if call.name != expected.name {
        return Verdict::fail(format!(
            "expected tool {:?}, got {:?}",
            expected.name, call.name
        ));
    }
    match args_match(&expected.arguments, &call.arguments) {
        Ok(()) => Verdict::pass(),
        Err(reason) => Verdict::fail(reason),
    }
}

/// `no_tool`: zero tool calls in the reply.
pub fn score_no_tool(output: &AssistantOutput) -> Verdict {
    if output.tool_calls.is_empty() {
        Verdict::pass()
    } else {
        Verdict::fail(format!(
            "expected no tool calls, got {} ({})",
            output.tool_calls.len(),
            output
                .tool_calls
                .iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ))
    }
}

/// `multi_turn`'s second turn: either another expected tool call, or a
/// final answer containing an expected substring (case-insensitive).
pub fn score_second_turn(expectation: &SecondTurnExpectation, output: &AssistantOutput) -> Verdict {
    match expectation {
        SecondTurnExpectation::ToolCall(expected) => score_tool_choice(expected, output),
        SecondTurnExpectation::FinalAnswer { contains } => {
            score_contains(&output.content, contains)
        }
    }
}

/// `long_context`: final answer contains the needle's expected substring
/// (case-insensitive).
pub fn score_long_context(expected_substring: &str, output: &AssistantOutput) -> Verdict {
    score_contains(&output.content, expected_substring)
}

fn score_contains(haystack: &str, needle: &str) -> Verdict {
    if haystack.to_lowercase().contains(&needle.to_lowercase()) {
        Verdict::pass()
    } else {
        Verdict::fail(format!("expected content to contain {needle:?}"))
    }
}

/// Every expected argument must be present in `actual` (an object) and
/// match per [`value_matches`]; extra arguments the model adds are not
/// penalized.
fn args_match(expected: &Map<String, Value>, actual: &Value) -> Result<(), String> {
    let actual_obj = actual
        .as_object()
        .ok_or_else(|| format!("actual arguments is not an object: {actual}"))?;
    for (key, expected_value) in expected {
        let actual_value = actual_obj
            .get(key)
            .ok_or_else(|| format!("missing argument {key:?}"))?;
        if !value_matches(expected_value, actual_value) {
            return Err(format!(
                "argument {key:?}: expected {expected_value}, got {actual_value}"
            ));
        }
    }
    Ok(())
}

/// The rendered `<parameter>` protocol only structures object/array
/// arguments — every scalar comes back from `rocml::chat::parser` as a
/// [`Value::String`] regardless of its logical type (see the parser's own
/// `parse_argument_value` doc comment), so scalar comparison always reads
/// the actual side as a string: numbers via numeric parse, everything else
/// via a trimmed string match. Booleans additionally accept "true"/"false"
/// case-insensitively.
fn value_matches(expected: &Value, actual: &Value) -> bool {
    match expected {
        Value::Number(n) => match (n.as_f64(), actual_as_f64(actual)) {
            (Some(e), Some(a)) => (e - a).abs() < 1e-6,
            _ => false,
        },
        Value::String(s) => actual_as_str(actual).is_some_and(|a| a.trim() == s.trim()),
        Value::Bool(b) => {
            actual_as_str(actual).is_some_and(|a| a.trim().eq_ignore_ascii_case(&b.to_string()))
        }
        Value::Null => {
            actual.is_null()
                || actual_as_str(actual).is_some_and(|a| {
                    let a = a.trim();
                    a.eq_ignore_ascii_case("none") || a.eq_ignore_ascii_case("null")
                })
        }
        Value::Object(_) | Value::Array(_) => actual == expected,
    }
}

fn actual_as_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse::<f64>().ok(),
        _ => None,
    }
}

fn actual_as_str(v: &Value) -> Option<&str> {
    match v {
        Value::String(s) => Some(s),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rocml::chat::ToolCall;
    use serde_json::json;

    fn output(content: &str, tool_calls: Vec<ToolCall>) -> AssistantOutput {
        AssistantOutput {
            thinking: None,
            content: content.to_string(),
            tool_calls,
        }
    }

    fn expected(name: &str, args: Value) -> ExpectedToolCall {
        ExpectedToolCall {
            name: name.to_string(),
            arguments: args.as_object().cloned().unwrap_or_default(),
        }
    }

    #[test]
    fn tool_choice_passes_on_exact_match() {
        let out = output(
            "",
            vec![ToolCall::new("get_weather", json!({"location": "Prague"}))],
        );
        let exp = expected("get_weather", json!({"location": "Prague"}));
        assert!(score_tool_choice(&exp, &out).pass);
    }

    #[test]
    fn tool_choice_fails_on_wrong_name() {
        let out = output("", vec![ToolCall::new("get_time", json!({}))]);
        let exp = expected("get_weather", json!({}));
        assert!(!score_tool_choice(&exp, &out).pass);
    }

    #[test]
    fn tool_choice_fails_when_zero_or_multiple_calls() {
        let exp = expected("get_weather", json!({}));
        assert!(!score_tool_choice(&exp, &output("", vec![])).pass);
        let two = vec![
            ToolCall::new("get_weather", json!({})),
            ToolCall::new("get_weather", json!({})),
        ];
        assert!(!score_tool_choice(&exp, &output("", two)).pass);
    }

    #[test]
    fn tool_choice_numeric_argument_compared_numerically() {
        // The parser always stores scalar arguments as strings; the
        // scenario's expected value is a real JSON number.
        let out = output(
            "",
            vec![ToolCall::new(
                "convert_currency",
                json!({"amount": "250", "from": "USD"}),
            )],
        );
        let exp = expected("convert_currency", json!({"amount": 250, "from": "USD"}));
        assert!(score_tool_choice(&exp, &out).pass);
    }

    #[test]
    fn tool_choice_string_argument_trims_whitespace() {
        let out = output(
            "",
            vec![ToolCall::new(
                "send_sms",
                json!({"message": "  hi there  "}),
            )],
        );
        let exp = expected("send_sms", json!({"message": "hi there"}));
        assert!(score_tool_choice(&exp, &out).pass);
    }

    #[test]
    fn tool_choice_extra_arguments_are_not_penalized() {
        let out = output(
            "",
            vec![ToolCall::new(
                "get_weather",
                json!({"location": "Prague", "unit": "celsius"}),
            )],
        );
        let exp = expected("get_weather", json!({"location": "Prague"}));
        assert!(score_tool_choice(&exp, &out).pass);
    }

    #[test]
    fn tool_choice_missing_argument_fails() {
        let out = output("", vec![ToolCall::new("get_weather", json!({}))]);
        let exp = expected("get_weather", json!({"location": "Prague"}));
        assert!(!score_tool_choice(&exp, &out).pass);
    }

    #[test]
    fn no_tool_passes_on_zero_calls() {
        assert!(score_no_tool(&output("Paris.", vec![])).pass);
    }

    #[test]
    fn no_tool_fails_when_a_call_was_made() {
        let out = output("", vec![ToolCall::new("get_weather", json!({}))]);
        assert!(!score_no_tool(&out).pass);
    }

    #[test]
    fn second_turn_final_answer_is_case_insensitive_substring() {
        let out = output("The temperature is 14 degrees.", vec![]);
        let exp = SecondTurnExpectation::FinalAnswer {
            contains: "14 DEGREES".to_string(),
        };
        assert!(score_second_turn(&exp, &out).pass);
    }

    #[test]
    fn second_turn_tool_call_variant_delegates_to_tool_choice() {
        let out = output(
            "",
            vec![ToolCall::new("get_weather", json!({"location": "Paris"}))],
        );
        let exp =
            SecondTurnExpectation::ToolCall(expected("get_weather", json!({"location": "Paris"})));
        assert!(score_second_turn(&exp, &out).pass);
    }

    #[test]
    fn long_context_substring_check_is_case_insensitive() {
        let out = output("the code is XJ-4471 as requested", vec![]);
        assert!(score_long_context("xj-4471", &out).pass);
        assert!(!score_long_context("ZZ-0000", &out).pass);
    }

    #[test]
    fn bool_argument_matches_case_insensitively() {
        let out = output(
            "",
            vec![ToolCall::new("control_light", json!({"on": "True"}))],
        );
        let exp = expected("control_light", json!({"on": true}));
        assert!(score_tool_choice(&exp, &out).pass);
    }
}
