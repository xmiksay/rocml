//! Drives one scenario end to end: renders messages through
//! `rocml::chat::render`, greedily decodes via `snapshot::turn::run_turn`
//! (with no snapshot store — every turn is a full, deterministic
//! reprocess), parses the reply with `rocml::chat::parse_assistant_output`,
//! and scores it (`super::scorer`). GPU-touching; not exercised by unit
//! tests (see `super::scorer`/`super::filler` for the pure logic those
//! cover instead).

use std::time::Instant;

use rocml::chat::{parse_assistant_output, AssistantOutput, Message, RenderOpts, Tool};
use rocml::snapshot::turn::run_turn;
use rocml::snapshot::{KvConfigStamp, ModelStamp};
use rocml::{RocmlError, SamplingParams};

use super::filler;
use super::results::{truncate_raw_output, ScenarioResult};
use super::scenario::{to_chat_tools, Scenario};
use super::scorer::{self, Verdict};
use crate::common::Loaded;

/// Thinking mode ON (Ornith's own default: `enable_thinking: None` leaves
/// the template's opening `<think>\n` tag in place) — issue #15 evaluates
/// the model as it's actually served, not with reasoning switched off.
const RENDER_OPTS: RenderOpts = RenderOpts {
    add_generation_prompt: true,
    enable_thinking: None,
    // Issue #9: a `multi_turn` scenario's turn-1 assistant message (below,
    // built with `.with_reasoning`) is re-rendered as history for turn 2 —
    // its thinking must not leak back in, matching how the model was
    // trained and how `rocml-serve` renders every request.
    keep_history_reasoning: false,
};

pub fn run_scenario(
    loaded: &mut Loaded,
    model_stamp: &ModelStamp,
    kv_config: &KvConfigStamp,
    max_gen_tokens: usize,
    scenario: &Scenario,
) -> Result<ScenarioResult, RocmlError> {
    let start = Instant::now();
    let (verdict, raw_output) = match scenario {
        Scenario::ToolChoice {
            system,
            user,
            tools,
            expected,
            ..
        } => {
            let messages = build_messages(system, user);
            let chat_tools = to_chat_tools(tools);
            let attempt = run_single_turn(
                loaded,
                model_stamp,
                kv_config,
                max_gen_tokens,
                &messages,
                &chat_tools,
            )?;
            score_attempt(attempt, |output| {
                scorer::score_tool_choice(expected, output)
            })
        }
        Scenario::NoTool {
            system,
            user,
            tools,
            ..
        } => {
            let messages = build_messages(system, user);
            let chat_tools = to_chat_tools(tools);
            let attempt = run_single_turn(
                loaded,
                model_stamp,
                kv_config,
                max_gen_tokens,
                &messages,
                &chat_tools,
            )?;
            score_attempt(attempt, scorer::score_no_tool)
        }
        Scenario::MultiTurn {
            system,
            user,
            tools,
            expected_first,
            tool_result,
            second_turn,
            ..
        } => run_multi_turn(
            loaded,
            model_stamp,
            kv_config,
            max_gen_tokens,
            system,
            user,
            tools,
            expected_first,
            tool_result,
            second_turn,
        )?,
        Scenario::LongContext {
            question,
            needle,
            expected_substring,
            position_fraction,
            target_length_tokens,
            seed,
            ..
        } => {
            let words = filler::words_for_tokens(*target_length_tokens);
            let context = filler::build_context(*seed, words, needle, *position_fraction);
            let user = format!("{context}\n\n{question}");
            let messages = vec![Message::user(user)];
            let attempt = run_single_turn(
                loaded,
                model_stamp,
                kv_config,
                max_gen_tokens,
                &messages,
                &[],
            )?;
            score_attempt(attempt, |output| {
                scorer::score_long_context(expected_substring, output)
            })
        }
    };

    Ok(ScenarioResult {
        id: scenario.id().to_string(),
        kind: scenario.kind().to_string(),
        pass: verdict.pass,
        detail: verdict.detail,
        raw_output: truncate_raw_output(&raw_output),
        wall_seconds: start.elapsed().as_secs_f64(),
    })
}

#[allow(clippy::too_many_arguments)]
fn run_multi_turn(
    loaded: &mut Loaded,
    model_stamp: &ModelStamp,
    kv_config: &KvConfigStamp,
    max_gen_tokens: usize,
    system: &Option<String>,
    user: &str,
    tools: &[super::scenario::ToolDef],
    expected_first: &super::scenario::ExpectedToolCall,
    tool_result: &str,
    second_turn: &super::scenario::SecondTurnExpectation,
) -> Result<(Verdict, String), RocmlError> {
    let chat_tools = to_chat_tools(tools);
    let mut messages = build_messages(system, user);

    let attempt1 = run_single_turn(
        loaded,
        model_stamp,
        kv_config,
        max_gen_tokens,
        &messages,
        &chat_tools,
    )?;
    // A malformed (unparseable) turn 1 can't be turned into an assistant
    // message for turn 2 to build on — the scenario fails outright, same as
    // a well-formed-but-wrong tool call would, just with a different detail.
    let turn1 = match attempt1 {
        TurnAttempt::Parsed(turn) => turn,
        TurnAttempt::Malformed { raw_text, reason } => {
            return Ok((
                Verdict::fail(format!("turn 1: malformed model output: {reason}")),
                raw_text,
            ));
        }
    };
    let verdict1 = scorer::score_tool_choice(expected_first, &turn1.output);
    if !verdict1.pass {
        return Ok((
            Verdict::fail(format!("turn 1: {}", verdict1.detail)),
            turn1.raw_text,
        ));
    }

    let mut assistant = Message::assistant(turn1.output.content.clone());
    if let Some(thinking) = &turn1.output.thinking {
        assistant = assistant.with_reasoning(thinking.clone());
    }
    if !turn1.output.tool_calls.is_empty() {
        assistant = assistant.with_tool_calls(turn1.output.tool_calls.clone());
    }
    messages.push(assistant);
    messages.push(Message::tool_response(tool_result.to_string()));

    let attempt2 = run_single_turn(
        loaded,
        model_stamp,
        kv_config,
        max_gen_tokens,
        &messages,
        &chat_tools,
    )?;
    let (verdict2, raw2) = score_attempt(attempt2, |output| {
        scorer::score_second_turn(second_turn, output)
    });
    let combined_raw = format!("[turn 1]\n{}\n\n[turn 2]\n{}", turn1.raw_text, raw2);
    let detail = format!("turn 1: ok; turn 2: {}", verdict2.detail);
    Ok((
        Verdict {
            pass: verdict2.pass,
            detail,
        },
        combined_raw,
    ))
}

struct TurnOutput {
    output: AssistantOutput,
    raw_text: String,
}

/// A model's raw decoded text either parses cleanly into structured pieces,
/// or it doesn't — an unterminated `<tool_call>` block, a malformed
/// `<parameter>` value, etc. The latter is a real, scoreable outcome (a
/// smaller/quantized model failing to hold the tool-call protocol together,
/// or hitting `max_gen_tokens` mid-block under a repetition loop that
/// greedy decoding can fall into), not a harness bug — so it's threaded
/// through as data instead of propagated as a `RocmlError` that would abort
/// the whole eval run over one bad scenario.
enum TurnAttempt {
    Parsed(TurnOutput),
    Malformed { raw_text: String, reason: String },
}

/// Scores a [`TurnAttempt`]: a parsed turn goes through `score`, a
/// malformed one is an automatic fail with the parse error as the detail.
/// Returns `(verdict, raw_text)` — the raw text always comes back so the
/// caller can record it for post-mortem regardless of which branch ran.
fn score_attempt(
    attempt: TurnAttempt,
    score: impl FnOnce(&AssistantOutput) -> Verdict,
) -> (Verdict, String) {
    match attempt {
        TurnAttempt::Parsed(turn) => (score(&turn.output), turn.raw_text),
        TurnAttempt::Malformed { raw_text, reason } => (
            Verdict::fail(format!("malformed model output: {reason}")),
            raw_text,
        ),
    }
}

fn run_single_turn(
    loaded: &mut Loaded,
    model_stamp: &ModelStamp,
    kv_config: &KvConfigStamp,
    max_gen_tokens: usize,
    messages: &[Message],
    tools: &[Tool],
) -> Result<TurnAttempt, RocmlError> {
    let prompt = rocml::chat::render(messages, tools, RENDER_OPTS)?;
    let prompt_ids = loaded.tokenizer.encode(&prompt);

    let mut raw_text = String::new();
    // `store: None` reduces `run_turn` to "reset, full prefill, greedy
    // decode" — no snapshot lookup/capture, so each scenario/turn is scored
    // against a fully independent, deterministic reprocess of its whole
    // conversation so far.
    run_turn(
        &mut loaded.model,
        &loaded.tokenizer,
        None,
        model_stamp,
        kv_config,
        &prompt_ids,
        max_gen_tokens,
        true,
        &SamplingParams::greedy(),
        &[],
        None,
        |chunk| raw_text.push_str(chunk),
    )?;

    Ok(match parse_assistant_output(&raw_text) {
        Ok(output) => TurnAttempt::Parsed(TurnOutput { output, raw_text }),
        Err(reason) => TurnAttempt::Malformed {
            raw_text,
            reason: reason.to_string(),
        },
    })
}

fn build_messages(system: &Option<String>, user: &str) -> Vec<Message> {
    let mut messages = Vec::new();
    if let Some(system) = system {
        messages.push(Message::system(system.clone()));
    }
    messages.push(Message::user(user.to_string()));
    messages
}
