//! Round-trip test: parse a model-style assistant turn (thinking + a couple
//! of tool calls) back into structured data, then re-render it and check
//! the bytes match what the fixture originally contained. Exercises
//! `parse_assistant_output` and `render` together, end to end.
//!
//! The two `_byte_identical`/`_replay` tests below opt into
//! `RenderOpts::keep_history_reasoning: true` deliberately: they're testing
//! parse/render fidelity for a turn's *own* thinking, not issue #9's
//! history-stripping policy — that's covered separately by
//! `default_render_strips_the_reconstructed_turns_reasoning` below (and, for
//! the resolution against the official template, by
//! `rocml/tests/chat_fixtures.rs`'s
//! `history_reasoning_is_stripped_by_default_but_available_via_opt_in`).

use rocml::chat::{parse_assistant_output, render, Message, RenderOpts};

/// What the model actually generates for the `tool_call_then_response`
/// fixture's assistant turn — i.e. the text between `<|im_start|>assistant\n`
/// and `<|im_end|>\n` in `rocml/tests/data/ornith_chat_fixtures.json`.
const MODEL_OUTPUT: &str = "<think>\nI should call get_weather then run_command.\n</think>\n\nOn it.\n\n<tool_call>\n<function=get_weather>\n<parameter=location>\nPrague\n</parameter>\n<parameter=unit>\ncelsius\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=run_command>\n<parameter=cmd>\nls /tmp\n</parameter>\n</function>\n</tool_call>";

/// The corresponding fixture-verified rendered turn (same source), used as
/// the round-trip's ground truth when `keep_history_reasoning: true`.
const EXPECTED_TURN: &str = "<|im_start|>assistant\n<think>\nI should call get_weather then run_command.\n</think>\n\nOn it.\n\n<tool_call>\n<function=get_weather>\n<parameter=location>\nPrague\n</parameter>\n<parameter=unit>\ncelsius\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=run_command>\n<parameter=cmd>\nls /tmp\n</parameter>\n</function>\n</tool_call><|im_end|>\n";

/// Same turn, `keep_history_reasoning`'s default (`false`, issue #9's
/// training-convention strip): content and tool calls are unchanged, only
/// the `<think>` block is emptied.
const EXPECTED_TURN_STRIPPED: &str = "<|im_start|>assistant\n<think>\n\n</think>\n\nOn it.\n\n<tool_call>\n<function=get_weather>\n<parameter=location>\nPrague\n</parameter>\n<parameter=unit>\ncelsius\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=run_command>\n<parameter=cmd>\nls /tmp\n</parameter>\n</function>\n</tool_call><|im_end|>\n";

const FAITHFUL_OPTS: RenderOpts = RenderOpts {
    add_generation_prompt: false,
    enable_thinking: None,
    keep_history_reasoning: true,
};

#[test]
fn parsed_tool_calls_re_render_byte_identical() {
    let parsed = parse_assistant_output(MODEL_OUTPUT).expect("parse model output");
    assert_eq!(
        parsed.thinking.as_deref(),
        Some("I should call get_weather then run_command.")
    );
    assert_eq!(parsed.content, "On it.");
    assert_eq!(parsed.tool_calls.len(), 2);
    assert_eq!(parsed.tool_calls[0].name, "get_weather");
    assert_eq!(parsed.tool_calls[1].name, "run_command");

    let reconstructed = Message::assistant(parsed.content)
        .with_reasoning(parsed.thinking.expect("thinking block present"))
        .with_tool_calls(parsed.tool_calls);

    // A single unrelated user turn ahead of it, so `render` sees a valid
    // conversation without pulling in the tools system block (irrelevant to
    // assistant-turn rendering, which this test targets in isolation).
    let messages = vec![Message::user("placeholder query"), reconstructed];
    let rendered = render(&messages, &[], FAITHFUL_OPTS).expect("render");

    let assistant_turn_start = rendered
        .find("<|im_start|>assistant\n")
        .expect("assistant turn present");
    let assistant_turn = &rendered[assistant_turn_start..];
    assert_eq!(assistant_turn, EXPECTED_TURN);
}

#[test]
fn round_trips_through_a_full_conversation_replay() {
    // Parse, then feed the reconstructed assistant message straight back
    // into a `messages` slice shaped like the original fixture (user,
    // assistant, tool, tool) and confirm the assistant segment alone still
    // matches — i.e. the round trip holds even amid the tool-response turns
    // that follow it in a real agentic loop.
    let parsed = parse_assistant_output(MODEL_OUTPUT).expect("parse model output");
    let assistant = Message::assistant(parsed.content)
        .with_reasoning(parsed.thinking.expect("thinking block present"))
        .with_tool_calls(parsed.tool_calls);

    let messages = vec![
        Message::user("What's the weather in Prague, and list files in /tmp?"),
        assistant,
        Message::tool_response("{\"temperature_c\": 18, \"conditions\": \"cloudy\"}"),
        Message::tool_response("{\"entries\": [\"a.txt\", \"b.txt\"]}"),
    ];
    let rendered = render(
        &messages,
        &[],
        RenderOpts {
            add_generation_prompt: true,
            enable_thinking: None,
            keep_history_reasoning: true,
        },
    )
    .unwrap();

    let start = rendered
        .find("<|im_start|>assistant\n")
        .expect("assistant turn present");
    let end = rendered[start..]
        .find("<|im_end|>\n")
        .expect("assistant turn end")
        + "<|im_end|>\n".len();
    assert_eq!(&rendered[start..start + end], EXPECTED_TURN);
}

/// Issue #9's actual fix, pinned against this same reconstructed turn:
/// `RenderOpts::default()` (`keep_history_reasoning: false`) must drop the
/// reasoning this exact turn carries once it's rendered as history, leaving
/// content and tool calls untouched.
#[test]
fn default_render_strips_the_reconstructed_turns_reasoning() {
    let parsed = parse_assistant_output(MODEL_OUTPUT).expect("parse model output");
    let reconstructed = Message::assistant(parsed.content)
        .with_reasoning(parsed.thinking.expect("thinking block present"))
        .with_tool_calls(parsed.tool_calls);

    let messages = vec![Message::user("placeholder query"), reconstructed];
    let rendered = render(&messages, &[], RenderOpts::default()).expect("render");

    let assistant_turn_start = rendered
        .find("<|im_start|>assistant\n")
        .expect("assistant turn present");
    assert_eq!(&rendered[assistant_turn_start..], EXPECTED_TURN_STRIPPED);
}
