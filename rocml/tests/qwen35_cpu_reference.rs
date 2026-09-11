//! Part 1 of the qwen35 milestone: validate architectural understanding on
//! a pure-Rust f32 CPU reference *before* trusting any HIP kernel — isolates
//! "did I understand Gated Delta Net / the hybrid block layout" from "is the
//! GPU kernel right" (the GPU-vs-Crane check lives in
//! `qwen35_greedy_parity.rs`).
//!
//! Feeds the fixture's own recorded `prompt_ids` (not rocml's tokenizer —
//! see that file's module doc for why raw-prompt tokenization diverges from
//! Crane's fixture generator) and checks the first few greedy tokens match
//! Crane's `generated_ids`. Deliberately short (`NUM_TOKENS`): this is a
//! naive nested-loop CPU forward pass over a 2B-parameter model with no
//! BLAS, so a handful of tokens is what "reasonable time" buys — full-length
//! parity is the GPU test's job. Release-mode only (`make test-model`);
//! skips itself if the checkpoint isn't present.

mod support;

use rocml_core::testpaths::checkpoint;
use serde::Deserialize;
use support::qwen35_cpu::CpuModel;

const GGUF_REL: &str = "Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf";
const FIXTURES_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/data/qwen35_greedy_fixtures.json"
);
const NUM_TOKENS: usize = 6;

#[derive(Deserialize)]
struct Fixtures {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    prompt: String,
    prompt_ids: Vec<u32>,
    generated_ids: Vec<u32>,
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best_idx = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
}

#[test]
fn qwen35_cpu_reference_matches_crane_for_a_handful_of_tokens() {
    let Some(gguf_path) = checkpoint(GGUF_REL) else {
        return;
    };
    let fixtures: Fixtures =
        serde_json::from_str(&std::fs::read_to_string(FIXTURES_PATH).expect("read fixtures json"))
            .expect("parse fixtures json");

    // Only the first case: this loop is O(minutes) per case at 2B params,
    // naive f32 loops — enough to validate understanding, not a full sweep.
    let case = &fixtures.cases[0];
    eprintln!("qwen35 CPU reference: prompt {:?}", case.prompt);

    let mut model = CpuModel::load(&gguf_path);
    let mut logits = Vec::new();
    for &id in &case.prompt_ids {
        logits = model.forward_token(id);
    }

    let mut got = Vec::with_capacity(NUM_TOKENS);
    for _ in 0..NUM_TOKENS {
        let next = argmax(&logits);
        got.push(next);
        logits = model.forward_token(next);
    }

    assert_eq!(
        got,
        case.generated_ids[..NUM_TOKENS],
        "prompt {:?}: CPU-reference greedy ids diverge from Crane's fixture in the first \
         {NUM_TOKENS} tokens -- likely a misunderstanding of the GDN recurrence, gating, or \
         hybrid layer layout (check before debugging the GPU kernels)",
        case.prompt
    );
}
