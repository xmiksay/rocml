//! End-to-end smoke test for Ornith-1.0-9B (qwen35 hybrid arch, Q6_K,
//! ~7.5GB): the checkpoint milestone 5b's quantized-weight loader exists to
//! make possible at all — dequant-to-f16 would need ~15GB just for weights,
//! which doesn't fit this card's 16GB VRAM alongside the KV/GDN cache and
//! scratch buffers. There is no independent CPU/GPU reference for this
//! checkpoint (unlike the candle/Crane parity suites), so this test checks
//! internal well-formedness instead: finite logits throughout, and greedy
//! decoding produces the exact same token sequence from two independent
//! model loads.
//!
//! Run via `make test-model` (release: a real ~7.4GB GGUF and a 32-layer x
//! 24-token decode loop, twice, is unbearably slow unoptimized). Skips
//! itself if the checkpoint isn't present on this machine.

use std::path::Path;

use rocml::Model;
use rocml_core::gguf::GgufFile;
use rocml_core::tokenizer::BpeTokenizer;

const GGUF_PATH: &str = "/mnt/nvme/miksa/checkpoints/Ornith-1.0-9B-GGUF/ornith-1.0-9b-Q6_K.gguf";
/// A raw continuation prompt (no chat template) — this test cares about
/// forward-pass well-formedness, not chat behavior.
const PROMPT: &str = "The capital of France is";
const NUM_TOKENS: usize = 24;

fn skip_if_missing(path: &str) -> bool {
    if !Path::new(path).exists() {
        eprintln!("skipping: {path} not present on this machine");
        return true;
    }
    false
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

/// Greedily decodes `n` tokens after `prompt_ids`, asserting every logits
/// vector along the way is entirely finite (a NaN/inf would silently corrupt
/// argmax rather than error, so this is checked explicitly rather than left
/// to show up as a garbled decode).
fn greedy_generate(model: &mut Model, prompt_ids: &[u32], n: usize) -> Vec<u32> {
    let mut logits = Vec::new();
    for &id in prompt_ids {
        logits = model.forward_token(id).expect("forward_token failed");
    }
    let mut ids = Vec::with_capacity(n);
    for _ in 0..n {
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "non-finite logit encountered during decode"
        );
        let best = argmax(&logits);
        ids.push(best);
        logits = model.forward_token(best).expect("forward_token failed");
    }
    ids
}

#[test]
fn ornith_9b_greedy_decode_is_well_formed_and_deterministic() {
    if skip_if_missing(GGUF_PATH) {
        return;
    }

    let gguf = GgufFile::open(GGUF_PATH).expect("open GGUF");
    let tokenizer = BpeTokenizer::from_gguf(&gguf).expect("build tokenizer");
    drop(gguf); // Model::load mmaps its own handle; no need to hold two.
    let prompt_ids = tokenizer.encode(PROMPT);
    assert!(!prompt_ids.is_empty(), "prompt tokenized to zero ids");

    let mut model_a = Model::load(GGUF_PATH).expect("load model (first run)");
    let ids_a = greedy_generate(&mut model_a, &prompt_ids, NUM_TOKENS);
    drop(model_a); // free VRAM before the second load

    let mut model_b = Model::load(GGUF_PATH).expect("load model (second run)");
    let ids_b = greedy_generate(&mut model_b, &prompt_ids, NUM_TOKENS);

    assert_eq!(
        ids_a, ids_b,
        "greedy decode is not deterministic across two independent loads"
    );

    let text = tokenizer.decode(&ids_a);
    assert!(!text.is_empty(), "decoded text is empty");
}
