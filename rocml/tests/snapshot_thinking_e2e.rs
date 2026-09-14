//! Issue #12's model-level acceptance gate: a thinking-enabled multi-turn
//! conversation must get a real snapshot hit on turn 2, and the
//! restore-then-resume-prefill path that hit takes must produce output
//! identical to a from-scratch prefill of the exact same turn-2 prompt.
//!
//! `rocml-serve/tests/server_e2e.rs`'s
//! `two_turn_conversation_matches_output_with_snapshots_disabled` proves the
//! same correctness property but needs two whole servers (two full model
//! loads) to compare a snapshot-enabled run against a snapshot-disabled one
//! — more VRAM than a dev GPU with another process already resident can
//! spare. This gate gets the same guarantee from one model instance: run
//! turn 2 once with the snapshot store populated (restore + resume), once
//! more with no store at all (full prefill), and diff the two turn-2
//! outputs.
//!
//! Real hardware + the real Qwen3.5-2B-Q8_0 checkpoint required; skips
//! itself if absent. Run via `make test-model` (`--release`).

use rocml::chat::{parse_assistant_output, render_with_boundary, Message, RenderOpts};
use rocml::snapshot::turn::{run_turn, stable_boundary_tokens};
use rocml::snapshot::{KvConfigStamp, ModelStamp, SnapshotStore};
use rocml::{KvCacheMode, LoadOptions, Model, SamplingParams};
use rocml_core::gguf::GgufFile;
use rocml_core::testpaths::checkpoint;
use rocml_core::tokenizer::BpeTokenizer;

const GGUF_REL: &str = "Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf";

/// Forces thinking on regardless of the registry preset (qwen3.5-2b's own
/// default is off) — issue #12 only reproduces when thinking is on, since
/// that's what puts a real `<think>...</think>` block in the way of the
/// pre-existing capture points.
const RENDER_OPTS: RenderOpts = RenderOpts {
    add_generation_prompt: true,
    enable_thinking: Some(true),
    keep_history_reasoning: false,
};

#[allow(clippy::too_many_arguments)]
fn run_one_turn(
    model: &mut Model,
    tokenizer: &BpeTokenizer,
    store: Option<&mut SnapshotStore>,
    model_stamp: &ModelStamp,
    kv_config: &KvConfigStamp,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    stable_boundary: Option<u32>,
) -> (rocml::snapshot::turn::TurnOutcome, String) {
    let mut raw_text = String::new();
    let outcome = run_turn(
        model,
        tokenizer,
        store,
        model_stamp,
        kv_config,
        prompt_ids,
        max_new_tokens,
        true,
        &SamplingParams::greedy(),
        &[],
        stable_boundary,
        |chunk| raw_text.push_str(chunk),
    )
    .expect("run_turn failed");
    (outcome, raw_text)
}

#[test]
fn thinking_enabled_second_turn_hits_the_stable_boundary_snapshot() {
    let Some(path) = checkpoint(GGUF_REL) else {
        return;
    };
    let ctx = 1024;
    let opts = LoadOptions::new(ctx).with_kv_cache(KvCacheMode::Fp16);
    let mut model = Model::load(&path, opts).expect("Model::load failed");
    let tokenizer = BpeTokenizer::from_gguf(&GgufFile::open(&path).expect("gguf open failed"))
        .expect("tokenizer load failed");
    let model_stamp = ModelStamp::from_path(&path).expect("stamp failed");
    let kv_config = KvConfigStamp {
        mode: KvCacheMode::Fp16,
        ctx,
    };
    let mut store = SnapshotStore::new(64 * 1024 * 1024, None, 0).expect("store failed");

    // Turn 1: cold, populates the snapshot store — including, thanks to
    // issue #12's fix, a mid-prefill capture at the render-stable boundary
    // (right after this turn's own last message), not just the two
    // pre-existing (and, under thinking, unstable) capture points.
    let turn1_messages = vec![Message::user(
        "What is 2+2? Answer with just the final number.",
    )];
    let (turn1_prompt, boundary1_byte) =
        render_with_boundary(&turn1_messages, &[], RENDER_OPTS).expect("turn 1 render failed");
    let turn1_ids = tokenizer.encode(&turn1_prompt);
    let boundary1 = stable_boundary_tokens(&tokenizer, &turn1_prompt, boundary1_byte, &turn1_ids)
        .expect("turn 1's stable boundary must be a genuine token prefix");

    let (outcome1, raw_text1) = run_one_turn(
        &mut model,
        &tokenizer,
        Some(&mut store),
        &model_stamp,
        &kv_config,
        &turn1_ids,
        96,
        Some(boundary1),
    );
    assert_eq!(outcome1.reused_prefix, 0, "turn 1 must be a cold miss");
    let turn1_output = parse_assistant_output(&raw_text1).expect("turn 1 output malformed");

    // Turn 2: history re-renders turn 1 with thinking stripped (matching
    // `rocml-serve/src/routes.rs`'s `keep_history_reasoning: false`) — the
    // exact shape that makes both pre-existing capture points miss.
    let turn2_messages = vec![
        Message::user("What is 2+2? Answer with just the final number."),
        Message::assistant(turn1_output.content.clone())
            .with_reasoning(turn1_output.thinking.clone().unwrap_or_default()),
        Message::user("Now what is 3+3? Answer with just the final number."),
    ];
    let (turn2_prompt, boundary2_byte) =
        render_with_boundary(&turn2_messages, &[], RENDER_OPTS).expect("turn 2 render failed");
    let turn2_ids = tokenizer.encode(&turn2_prompt);
    // Sanity check on the invariant this whole mechanism rests on (also
    // covered generically by `snapshot_stable_boundary.rs`) before relying
    // on it for the assertions below.
    stable_boundary_tokens(&tokenizer, &turn2_prompt, boundary2_byte, &turn2_ids)
        .expect("turn 2's stable boundary must be a genuine token prefix");

    let (outcome2, text2_with_snapshot) = run_one_turn(
        &mut model,
        &tokenizer,
        Some(&mut store),
        &model_stamp,
        &kv_config,
        &turn2_ids,
        64,
        None,
    );
    assert!(
        outcome2.reused_prefix > 0,
        "turn 2 must hit a snapshot despite thinking being on — this is issue #12's whole point"
    );
    assert_eq!(
        outcome2.reused_prefix, boundary1,
        "the only snapshot available to turn 2 is turn 1's own stable-boundary capture \
         (turn 1's old-style prefill-boundary and end-of-turn snapshots both still miss \
         under thinking, by design — see the module doc)"
    );

    // Correctness of the restore+resume path: turn 2 run fresh (no store at
    // all, full prefill) must produce byte-identical output to the
    // snapshot-restored run above.
    let (outcome3, text2_without_snapshot) = run_one_turn(
        &mut model,
        &tokenizer,
        None,
        &model_stamp,
        &kv_config,
        &turn2_ids,
        64,
        None,
    );
    assert_eq!(
        outcome3.reused_prefix, 0,
        "the no-store run must not reuse anything"
    );
    assert_eq!(
        text2_with_snapshot, text2_without_snapshot,
        "a snapshot-resumed turn 2 must match a from-scratch turn 2 byte-for-byte"
    );
}
