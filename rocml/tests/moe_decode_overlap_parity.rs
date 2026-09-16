//! M4 lever 2's central correctness gate: the qwen35moe decode-overlap
//! pipeline (`qwen35::forward::moe_decode_overlap`, `LoadOptions::
//! moe_decode_overlap`/`--moe-decode-overlap`) must be numerically inert —
//! pipelining a cache miss's H2D copy on a non-blocking stream concurrently
//! with the previous expert's compute must produce exactly the same bytes a
//! run with overlap off would, not just near-tolerance ones. Mirrors
//! `moe_expert_cache_parity.rs`'s own methodology (a tiny cache capacity
//! forcing heavy eviction churn — the scenario most likely to expose a
//! stream/event ordering bug) and its own bit-identical-logits bar, per
//! this lever's own user-approved guard: "implement behind a flag, default
//! off, promoted only after a bit-identical parity gate passes."
//!
//! Real hardware + the real Ornith-1.5-35B-A3B-Q4_K_M checkpoint required;
//! skips itself if absent. Each `#[test]` loads two full ~35B models
//! sequentially within itself, but cargo's default parallel harness would
//! still run the two `#[test]` fns concurrently — `make test-model` runs
//! this file with `--test-threads=1` (measured: four concurrent ~35B
//! models on one card segfaults rather than erroring cleanly).

use rocml::{LoadOptions, Model};
use rocml_core::testpaths::checkpoint;

const GGUF_REL: &str = "Ornith-1.5-35B-A3B-GGUF/Ornith-1.5-35B-Q4_K_M.gguf";
const CTX: usize = 2048;
const PROMPT_LEN: usize = 16;
/// Same rationale as `moe_expert_cache_parity.rs`: every decode step visits
/// every MoE layer's 8-expert top-k, so a handful of steps already
/// exercises heavy eviction churn against a tiny cache.
const GREEDY_TOKENS: usize = 12;

fn synthetic_prompt(len: usize, vocab_size: u32) -> Vec<u32> {
    (0..len as u32).map(|i| (i * 41 + 7) % vocab_size).collect()
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .expect("logits must be non-empty")
}

fn run_greedy(model: &mut Model, prompt_ids: &[u32], n: usize) -> (Vec<Vec<f32>>, Vec<u32>) {
    model.reset().expect("reset failed");
    let mut all_logits = Vec::new();
    let mut logits = Vec::new();
    for &id in prompt_ids {
        logits = model.forward_token(id).expect("forward_token failed");
    }
    all_logits.push(logits.clone());
    let mut tokens = Vec::with_capacity(n);
    for _ in 0..n {
        let next = argmax(&logits);
        tokens.push(next);
        logits = model.forward_token(next).expect("forward_token failed");
        all_logits.push(logits.clone());
    }
    (all_logits, tokens)
}

/// `cache_slots`: `Some(3)` forces the same heavy-churn scenario
/// `moe_expert_cache_parity.rs` uses (fewer slots than a single layer's
/// top-k=8, so every layer evicts and reloads repeatedly — the timing
/// window most likely to expose a stream/event ordering bug in the
/// overlap pipeline); `None` uses the auto-sized (large) production cache,
/// where overlap mostly hits and only occasionally has a real miss to
/// pipeline, checking the lever doesn't misbehave in the common case
/// either.
fn run_with(
    path: &std::path::Path,
    cache_slots: Option<usize>,
    overlap: bool,
    prompt_ids: &[u32],
) -> (Vec<Vec<f32>>, Vec<u32>) {
    let opts = LoadOptions::new(CTX)
        .with_moe_cache_slots(cache_slots)
        .with_moe_decode_overlap(overlap);
    let mut model = Model::load(path, opts).expect("load model");
    run_greedy(&mut model, prompt_ids, GREEDY_TOKENS)
}

fn assert_bit_identical(label: &str, a: &(Vec<Vec<f32>>, Vec<u32>), b: &(Vec<Vec<f32>>, Vec<u32>)) {
    assert_eq!(
        a.1, b.1,
        "{label}: greedy continuation must be identical regardless of decode overlap"
    );
    assert_eq!(
        a.0.len(),
        b.0.len(),
        "{label}: same number of forward steps"
    );
    for (step, (x, y)) in a.0.iter().zip(b.0.iter()).enumerate() {
        assert_eq!(
            x, y,
            "{label}: step {step}: logits must be bit-identical between overlap on and off"
        );
    }
}

#[test]
fn overlap_on_and_off_produce_bit_identical_output_tiny_cache() {
    let Some(path) = checkpoint(GGUF_REL) else {
        eprintln!(
            "skipping overlap_on_and_off_produce_bit_identical_output_tiny_cache: \
             {GGUF_REL} not found"
        );
        return;
    };
    let vocab_size = {
        let m = Model::load(&path, LoadOptions::new(CTX).with_moe_cache_slots(Some(3)))
            .expect("load for vocab_size");
        m.vocab_size()
    };
    let prompt_ids = synthetic_prompt(PROMPT_LEN, vocab_size);

    let off = run_with(&path, Some(3), false, &prompt_ids);
    let on = run_with(&path, Some(3), true, &prompt_ids);
    assert_bit_identical("tiny cache (3 slots, heavy churn)", &off, &on);
}

#[test]
fn overlap_on_and_off_produce_bit_identical_output_production_cache() {
    let Some(path) = checkpoint(GGUF_REL) else {
        eprintln!(
            "skipping overlap_on_and_off_produce_bit_identical_output_production_cache: \
             {GGUF_REL} not found"
        );
        return;
    };
    let vocab_size = {
        let m = Model::load(&path, LoadOptions::new(CTX)).expect("load for vocab_size");
        m.vocab_size()
    };
    let prompt_ids = synthetic_prompt(PROMPT_LEN, vocab_size);

    let off = run_with(&path, None, false, &prompt_ids);
    let on = run_with(&path, None, true, &prompt_ids);
    assert_bit_identical("auto-sized production cache", &off, &on);
}
