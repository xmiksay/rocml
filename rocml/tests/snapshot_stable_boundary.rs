//! Issue #12: conversation snapshots never hit for thinking-enabled models,
//! because both of `run_turn`'s pre-existing capture points land on
//! render-unstable positions once a real `<think>...</think>` block exists.
//!
//! This is a tokenizer-only regression test — no GPU, no model load, just
//! `BpeTokenizer` built from the real Qwen3.5-2B GGUF's own metadata (the
//! qwen35 family's actual 248k-vocab tokenizer, not an approximation).
//! Skips itself if the checkpoint isn't present, same as every other
//! checkpoint-backed test in this workspace.
//!
//! `matches_real_qwen35_tokenizer_json_on_tricky_inputs` (in
//! `rocml-core/tests/tokenizer_cross_validation.rs`) already proves this
//! `BpeTokenizer` matches HF's own byte-for-byte; this file builds on that
//! ground truth to reproduce and then fix the snapshot-miss bug at the
//! token level.

use rocml::chat::{render_with_boundary, Message, RenderOpts, Tool};
use rocml::snapshot::turn::stable_boundary_tokens;
use rocml_core::gguf::GgufFile;
use rocml_core::testpaths::checkpoint;
use rocml_core::tokenizer::BpeTokenizer;
use serde_json::json;

const GGUF_REL: &str = "Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf";

/// Thinking on, history reasoning stripped — exactly the server's own
/// `RenderOpts` for a thinking-enabled model (`rocml-serve/src/routes.rs`),
/// which is the configuration issue #12 was measured against.
const THINKING_ON: RenderOpts = RenderOpts {
    add_generation_prompt: true,
    enable_thinking: Some(true),
    keep_history_reasoning: false,
};

fn load_tokenizer(path: &std::path::Path) -> BpeTokenizer {
    let gguf = GgufFile::open(path).expect("gguf open failed");
    BpeTokenizer::from_gguf(&gguf).expect("tokenizer load failed")
}

/// The central regression: turn 1's full prefill-boundary prompt (ending
/// `...assistant\n<think>\n`, per issue #12's root-cause writeup) is NOT a
/// token-prefix of turn 2's re-rendered prompt — off by exactly the merged
/// `\n\n` at the cut, not by the whole reasoning block — while the
/// `render_with_boundary` cut (right after the last history message's
/// `<|im_end|>`) IS a common token-prefix of both, because it never lands
/// inside a text run that a later turn's content can extend.
#[test]
fn stable_boundary_survives_the_thinking_strip_that_breaks_the_old_boundary() {
    let Some(path) = checkpoint(GGUF_REL) else {
        return;
    };
    let tokenizer = load_tokenizer(&path);

    let turn1_messages = vec![Message::user("What is 2+2?")];
    let (turn1_prompt, boundary1_byte) =
        render_with_boundary(&turn1_messages, &[], THINKING_ON).expect("turn 1 render failed");
    let turn1_tokens = tokenizer.encode(&turn1_prompt);

    let boundary1_tokens =
        stable_boundary_tokens(&tokenizer, &turn1_prompt, boundary1_byte, &turn1_tokens)
            .expect("turn 1's stable boundary must be a genuine token prefix of its own prompt");

    let turn2_messages = vec![
        Message::user("What is 2+2?"),
        Message::assistant("4").with_reasoning("Let's see, 2 plus 2 equals 4."),
        Message::user("And what about 3+3?"),
    ];
    let (turn2_prompt, _boundary2_byte) =
        render_with_boundary(&turn2_messages, &[], THINKING_ON).expect("turn 2 render failed");
    let turn2_tokens = tokenizer.encode(&turn2_prompt);

    // Documents the bug: the old prefill-boundary snapshot (turn1_tokens in
    // full) is not reproduced by turn 2's history re-render, because turn 2
    // strips the real reasoning back out to nothing, changing what the text
    // run between the `<think>` and `</think>` specials BPE-merges into at
    // exactly the cut turn 1 made mid-run.
    assert!(
        !turn2_tokens.starts_with(turn1_tokens.as_slice()),
        "turn 1's full prompt should NOT be a token-prefix of turn 2's prompt \
         (if this now passes, the old bug is gone and this test should be simplified)"
    );

    // Proves the fix's invariant: the stable-boundary prefix — cut on the
    // `<|im_end|>` special token instead of mid-`<think>`-block — is a
    // common token-prefix of both turns' prompts, even though turn 2 is a
    // longer conversation with different (stripped) assistant content.
    let boundary1 = boundary1_tokens as usize;
    assert!(
        turn1_tokens.len() >= boundary1 && turn2_tokens.len() >= boundary1,
        "both prompts must be at least as long as the stable boundary"
    );
    assert_eq!(
        turn1_tokens[..boundary1],
        turn2_tokens[..boundary1],
        "the stable-boundary prefix must be identical in both turns' token sequences"
    );
}

/// General invariant `run_turn`'s `stable_boundary` plumbing relies on:
/// `render_with_boundary`'s offset always cuts on a token boundary, for any
/// well-formed render — not just the two-turn thinking case above. Exercises
/// a system message, tool calls in history, and thinking off, none of which
/// this fix's stability argument depends on the shape of.
#[test]
fn render_with_boundary_offset_is_always_a_token_prefix() {
    let Some(path) = checkpoint(GGUF_REL) else {
        return;
    };
    let tokenizer = load_tokenizer(&path);

    let weather_tool = Tool::new(
        "get_weather",
        "Get the current weather for a city",
        json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}),
    );

    let cases: Vec<(Vec<Message>, &[Tool])> = vec![
        (vec![Message::user("hi")], &[]),
        (vec![Message::system("be terse"), Message::user("hi")], &[]),
        (
            vec![
                Message::user("what's the weather in Prague?"),
                Message::assistant("").with_tool_calls(vec![rocml::chat::ToolCall::new(
                    "get_weather",
                    json!({"city": "Prague"}),
                )]),
                Message::tool_response("{\"temp_c\": 18}"),
            ],
            std::slice::from_ref(&weather_tool),
        ),
    ];

    for (messages, tools) in &cases {
        for opts in [
            THINKING_ON,
            RenderOpts {
                enable_thinking: Some(false),
                ..THINKING_ON
            },
        ] {
            let (prompt, boundary_byte) =
                render_with_boundary(messages, tools, opts).expect("render failed");
            let prompt_ids = tokenizer.encode(&prompt);
            assert!(
                stable_boundary_tokens(&tokenizer, &prompt, boundary_byte, &prompt_ids).is_some(),
                "render_with_boundary's offset must always tokenize to a genuine prefix \
                 (prompt: {prompt:?}, boundary_byte: {boundary_byte})"
            );
        }
    }
}
