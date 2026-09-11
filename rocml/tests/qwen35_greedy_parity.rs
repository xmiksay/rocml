//! Part 4 of the qwen35 milestone: GPU hybrid-model greedy decode must match
//! Crane's independent candle implementation token-for-token, from the same
//! GGUF, on real ROCm hardware.
//!
//! Run via `make test-model` (release: dequantizing the real ~2GB GGUF and
//! running a 24-layer x ~45-token decode loop is unbearably slow debug).
//! Skips itself if the checkpoint isn't present.
//!
//! **Known tokenizer discrepancy (documented, not a rocml bug):** Crane's
//! GGUF-embedded-tokenizer loader (`crane-core/src/utils/tokenizer_utils.rs`'s
//! `build_tokenizer_from_gguf`) always attaches the HF `tokenizers` crate's
//! *generic* `ByteLevelPreTokenizer::default()` (default `add_prefix_space =
//! true`, the plain GPT-2 split regex) and never reads `tokenizer.ggml.pre` —
//! so it never selects the qwen-specific split regex rocml's own tokenizer
//! implements (`rocml_core::tokenizer::PRETOKENIZE_PATTERN_QWEN35`, verified
//! byte-identical against the real Qwen3.5 `tokenizer.json`). For a *raw*
//! prompt (no chat-template text ahead of it to absorb the difference) this
//! shows up as: (1) the very first word gets Crane's synthetic leading space
//! (`"ĠThe"` instead of `"The"`), and (2) a punctuation character immediately
//! followed by a letter with no separating space (e.g. `"(n"` in
//! `"fibonacci(n):"`) merges into one Crane pre-token where the real Qwen
//! regex — and rocml's — keeps it two (`"("`, `"n"`). This is a latent
//! limitation of Crane's fixture generator, not a defect in either engine's
//! forward pass, so this test logs the prompt-id mismatch prominently but
//! does not fail on it; it feeds the fixture's own recorded `prompt_ids`
//! (not rocml's re-tokenization) into the model, which is what actually
//! isolates "does the hybrid GPU forward pass match Crane" — the thing this
//! test exists to check — from the unrelated tokenizer question.

use rocml::Model;
use rocml_core::gguf::GgufFile;
use rocml_core::testpaths::checkpoint;
use rocml_core::tokenizer::BpeTokenizer;
use serde::Deserialize;

const GGUF_REL: &str = "Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf";
const FIXTURES_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/data/qwen35_greedy_fixtures.json"
);
const NUM_TOKENS: usize = 40;
const NEAR_TIE_RELATIVE_GAP: f32 = 1e-2;
const MIN_EXACT_PREFIX: usize = 24;

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

fn top2(logits: &[f32]) -> (u32, f32, f32) {
    let (mut best_idx, mut best_val, mut second_val) = (0u32, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            second_val = best_val;
            best_val = v;
            best_idx = i as u32;
        } else if v > second_val {
            second_val = v;
        }
    }
    (best_idx, best_val, second_val)
}

fn generate_with_logit_gaps(
    model: &mut Model,
    prompt_ids: &[u32],
    n: usize,
) -> (Vec<u32>, Vec<(f32, f32)>) {
    let mut logits = Vec::new();
    for &id in prompt_ids {
        logits = model.forward_token(id).expect("forward_token failed");
    }
    let mut ids = Vec::with_capacity(n);
    let mut gaps = Vec::with_capacity(n);
    for _ in 0..n {
        let (best_idx, best_val, second_val) = top2(&logits);
        ids.push(best_idx);
        gaps.push((best_val, second_val));
        logits = model.forward_token(best_idx).expect("forward_token failed");
    }
    (ids, gaps)
}

#[test]
fn qwen35_2b_hybrid_greedy_matches_crane_gpu_reference() {
    let Some(gguf_path) = checkpoint(GGUF_REL) else {
        return;
    };
    let fixtures: Fixtures =
        serde_json::from_str(&std::fs::read_to_string(FIXTURES_PATH).expect("read fixtures json"))
            .expect("parse fixtures json");

    let gguf = GgufFile::open(&gguf_path).expect("open GGUF");
    let tokenizer = BpeTokenizer::from_gguf(&gguf).expect("build tokenizer");
    drop(gguf);

    let mut model = Model::load(&gguf_path).expect("load model");

    for case in &fixtures.cases {
        let own_ids = tokenizer.encode(&case.prompt);
        if own_ids != case.prompt_ids {
            eprintln!(
                "NOTE: prompt {:?}: rocml's tokenizer produced {own_ids:?}, Crane's fixture \
                 recorded {:?} -- see this file's module doc (known Crane GGUF-tokenizer \
                 limitation, not asserted here). Feeding the fixture's ids directly.",
                case.prompt, case.prompt_ids
            );
        }

        model.reset().expect("reset failed");
        let (ids, gaps) = generate_with_logit_gaps(&mut model, &case.prompt_ids, NUM_TOKENS);

        if ids == case.generated_ids {
            continue;
        }

        let i = ids
            .iter()
            .zip(&case.generated_ids)
            .position(|(a, b)| a != b)
            .unwrap_or(ids.len().min(case.generated_ids.len()));
        let (best, second) = gaps[i];
        let rel_gap = (best - second).abs() / best.abs().max(1.0);

        if rel_gap < NEAR_TIE_RELATIVE_GAP && i >= MIN_EXACT_PREFIX {
            eprintln!(
                "NOTE: prompt {:?} diverges from the Crane GPU reference at token {i} \
                 (relative top-2 logit gap {rel_gap:.6} -- a plausible f16-vs-f32 near-tie); \
                 asserting an exact match up to token {i} instead of the full {NUM_TOKENS}.",
                case.prompt
            );
            assert_eq!(
                &ids[..i],
                &case.generated_ids[..i],
                "prompt {:?}: prefix mismatch before the documented near-tie at token {i}",
                case.prompt
            );
        } else {
            panic!(
                "prompt {:?}: greedy decode diverges from the Crane GPU reference at token {i}, \
                 top-2 logits {best} vs {second} (relative gap {rel_gap:.6}) -- not a near-tie, \
                 likely a forward-pass bug (check GDN gating/recurrence, conv state ordering, \
                 partial-rope width, or the attention output gate). got {ids:?}, want {:?}",
                case.prompt, case.generated_ids
            );
        }
    }
}
