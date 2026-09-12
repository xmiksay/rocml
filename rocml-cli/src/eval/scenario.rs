//! Scenario schema for the agentic quality-eval harness (issue #15):
//! deserializes `bench/eval/scenarios.json` into the four scenario kinds
//! `rocml-cli eval` drives — see the module doc on `super` for the overall
//! harness design and `scorer` for how each kind is graded.

use rocml::chat::Tool;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// A tool definition as it appears in scenario JSON — the flat shape
/// `rocml::chat::Tool` already uses, kept as its own type so scenario
/// parsing doesn't depend on the chat crate's `Serialize` impl matching by
/// coincidence.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

pub fn to_chat_tools(tools: &[ToolDef]) -> Vec<Tool> {
    tools
        .iter()
        .map(|t| Tool::new(t.name.clone(), t.description.clone(), t.parameters.clone()))
        .collect()
}

/// The tool call a passing turn must produce: exact function name, plus a
/// per-argument expected-value subset (`scorer::args_match`) — extra
/// arguments the model adds are not penalized, only missing/mismatched ones.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ExpectedToolCall {
    pub name: String,
    #[serde(default)]
    pub arguments: Map<String, Value>,
}

/// What a `multi_turn` scenario's second turn must produce.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SecondTurnExpectation {
    ToolCall(ExpectedToolCall),
    FinalAnswer {
        /// Case-insensitive substring the final answer text must contain.
        contains: String,
    },
}

/// One eval scenario. `kind` (the serde tag) picks the variant; every
/// variant carries its own `id`, used both in results output and to skip
/// already-scored scenarios on `--resume`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Scenario {
    /// A request that must trigger exactly one tool call with the expected
    /// name and arguments.
    ToolChoice {
        id: String,
        #[serde(default)]
        system: Option<String>,
        user: String,
        tools: Vec<ToolDef>,
        expected: ExpectedToolCall,
    },
    /// A request tools are available for but must not trigger — pass means
    /// zero tool calls in the reply.
    NoTool {
        id: String,
        #[serde(default)]
        system: Option<String>,
        user: String,
        tools: Vec<ToolDef>,
    },
    /// A two-turn exchange: turn 1 must produce `expected_first`; the
    /// harness then feeds back `tool_result` as a canned tool response and
    /// grades turn 2 against `second_turn`.
    MultiTurn {
        id: String,
        #[serde(default)]
        system: Option<String>,
        user: String,
        tools: Vec<ToolDef>,
        expected_first: ExpectedToolCall,
        tool_result: String,
        second_turn: SecondTurnExpectation,
    },
    /// A retrieval needle buried in deterministically-generated filler
    /// prose (`super::filler`) — `target_length_tokens` and
    /// `position_fraction` describe the filler, not the needle text itself,
    /// so the (large) filler never needs to be checked into JSON.
    LongContext {
        id: String,
        question: String,
        needle: String,
        /// Case-insensitive substring the final answer must contain to pass
        /// — usually the needle's distinguishing fact, not the whole
        /// sentence (fragile to ask a model to reproduce verbatim).
        expected_substring: String,
        /// Where in the filler (0.0 = start, 1.0 = end) the needle is
        /// spliced in.
        position_fraction: f32,
        target_length_tokens: usize,
        seed: u64,
    },
}

impl Scenario {
    pub fn id(&self) -> &str {
        match self {
            Scenario::ToolChoice { id, .. }
            | Scenario::NoTool { id, .. }
            | Scenario::MultiTurn { id, .. }
            | Scenario::LongContext { id, .. } => id,
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Scenario::ToolChoice { .. } => "tool_choice",
            Scenario::NoTool { .. } => "no_tool",
            Scenario::MultiTurn { .. } => "multi_turn",
            Scenario::LongContext { .. } => "long_context",
        }
    }
}

/// Loads and parses the scenario set. A non-empty, well-formed JSON array is
/// the only thing enforced here — kind-specific field validation happens
/// implicitly through serde (a missing required field is a parse error).
pub fn load_scenarios(path: &std::path::Path) -> Result<Vec<Scenario>, rocml::RocmlError> {
    let text = std::fs::read_to_string(path)?;
    let scenarios: Vec<Scenario> = serde_json::from_str(&text)?;
    if scenarios.is_empty() {
        return Err(rocml::RocmlError::Eval(format!(
            "{}: scenario file has no scenarios",
            path.display()
        )));
    }
    let mut seen = std::collections::HashSet::new();
    for s in &scenarios {
        if !seen.insert(s.id()) {
            return Err(rocml::RocmlError::Eval(format!(
                "duplicate scenario id {:?}",
                s.id()
            )));
        }
    }
    Ok(scenarios)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"[
        {
            "kind": "tool_choice",
            "id": "t1",
            "user": "What's the weather in Prague?",
            "tools": [
                {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}
            ],
            "expected": {"name": "get_weather", "arguments": {"location": "Prague"}}
        },
        {
            "kind": "no_tool",
            "id": "t2",
            "user": "What is the capital of France?",
            "tools": []
        },
        {
            "kind": "multi_turn",
            "id": "t3",
            "user": "Check the weather in London.",
            "tools": [],
            "expected_first": {"name": "get_weather", "arguments": {"location": "London"}},
            "tool_result": "{\"temperature_c\": 14}",
            "second_turn": {"type": "final_answer", "contains": "14"}
        },
        {
            "kind": "long_context",
            "id": "t4",
            "question": "What is the code?",
            "needle": "The code is 42.",
            "expected_substring": "42",
            "position_fraction": 0.5,
            "target_length_tokens": 4000,
            "seed": 7
        }
    ]"#;

    #[test]
    fn parses_all_four_kinds() {
        let scenarios: Vec<Scenario> = serde_json::from_str(SAMPLE).expect("valid schema");
        assert_eq!(scenarios.len(), 4);
        assert_eq!(scenarios[0].kind(), "tool_choice");
        assert_eq!(scenarios[1].kind(), "no_tool");
        assert_eq!(scenarios[2].kind(), "multi_turn");
        assert_eq!(scenarios[3].kind(), "long_context");
        assert_eq!(scenarios[0].id(), "t1");
    }

    #[test]
    fn multi_turn_second_turn_tool_call_variant_parses() {
        let json =
            r#"{"type": "tool_call", "name": "get_weather", "arguments": {"location": "Paris"}}"#;
        let expectation: SecondTurnExpectation = serde_json::from_str(json).expect("valid");
        match expectation {
            SecondTurnExpectation::ToolCall(call) => assert_eq!(call.name, "get_weather"),
            other => panic!("expected ToolCall variant, got {other:?}"),
        }
    }

    #[test]
    fn duplicate_ids_are_rejected() {
        let json = r#"[
            {"kind": "no_tool", "id": "dup", "user": "a", "tools": []},
            {"kind": "no_tool", "id": "dup", "user": "b", "tools": []}
        ]"#;
        let path =
            std::env::temp_dir().join(format!("rocml-eval-dup-test-{}.json", std::process::id()));
        std::fs::write(&path, json).unwrap();
        let err = load_scenarios(&path).unwrap_err();
        std::fs::remove_file(&path).ok();
        assert!(matches!(err, rocml::RocmlError::Eval(_)));
    }

    #[test]
    fn empty_scenario_list_is_rejected() {
        let path =
            std::env::temp_dir().join(format!("rocml-eval-empty-test-{}.json", std::process::id()));
        std::fs::write(&path, "[]").unwrap();
        let err = load_scenarios(&path).unwrap_err();
        std::fs::remove_file(&path).ok();
        assert!(matches!(err, rocml::RocmlError::Eval(_)));
    }

    /// The actual checked-in scenario set (`bench/eval/scenarios.json`):
    /// parses under this schema, has unique ids, and matches issue #15's
    /// requested mix (~8 tool_choice, ~3 no_tool, ~5 multi_turn, ~4
    /// long_context). A GPU-free regression test against drift in the data
    /// file itself, independent of any real eval run.
    #[test]
    fn checked_in_scenarios_file_parses_and_matches_the_requested_mix() {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../bench/eval/scenarios.json");
        let scenarios = load_scenarios(&path).expect("checked-in scenarios.json must parse");
        assert_eq!(scenarios.len(), 20);

        let mut counts = std::collections::HashMap::new();
        for s in &scenarios {
            *counts.entry(s.kind()).or_insert(0) += 1;
        }
        assert_eq!(counts.get("tool_choice"), Some(&8));
        assert_eq!(counts.get("no_tool"), Some(&3));
        assert_eq!(counts.get("multi_turn"), Some(&5));
        assert_eq!(counts.get("long_context"), Some(&4));
    }
}
