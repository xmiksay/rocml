//! M2's central correctness gate: the VRAM-resident LRU expert cache
//! (`qwen35::forward::moe_cache::ExpertCache`) must be numerically inert —
//! a cache hit serves exactly the same bytes a fresh mmap copy would, so a
//! run with the cache disabled (`--moe-cache-slots 0`, M1's always-copy
//! path) and a run with a tiny cache (forcing constant eviction churn, the
//! worst case for a bug in the slot-reuse bookkeeping) must produce
//! bit-identical logits and greedy tokens, not just near-tolerance ones.
//!
//! Real hardware + the real Ornith-1.5-35B-A3B-Q4_K_M checkpoint required;
//! skips itself if absent. Loads the model twice *sequentially* (the
//! second only after the first is dropped), so this needs no
//! `--test-threads=1` the way `mixed_kv_chunked_prefill_parity.rs`'s
//! multiple-tests-loading-concurrently pattern does.

use rocml::{LoadOptions, Model};
use rocml_core::testpaths::checkpoint;

const GGUF_REL: &str = "Ornith-1.5-35B-A3B-GGUF/Ornith-1.5-35B-Q4_K_M.gguf";
const CTX: usize = 2048;
const PROMPT_LEN: usize = 16;
/// Each decode step visits every one of the model's ~40 MoE layers, 8
/// experts each — far more distinct `(layer, expert)` picks per step than
/// any small cache capacity below, so a handful of steps already exercises
/// heavy eviction churn.
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

/// Feeds `prompt_ids` token-serially, then greedily decodes `n` more steps,
/// returning every step's logits (prompt's last + every decode step's) and
/// chosen tokens.
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

#[test]
fn cached_and_uncached_runs_produce_bit_identical_logits_and_tokens() {
    let Some(path) = checkpoint(GGUF_REL) else {
        eprintln!(
            "skipping cached_and_uncached_runs_produce_bit_identical_logits_and_tokens: \
             {GGUF_REL} not found"
        );
        return;
    };

    let mut uncached = Model::load(&path, LoadOptions::new(CTX).with_moe_cache_slots(Some(0)))
        .expect("load uncached model");
    let vocab_size = uncached.vocab_size();
    let prompt_ids = synthetic_prompt(PROMPT_LEN, vocab_size);
    assert_eq!(
        uncached.as_hybrid().and_then(|m| m.moe_cache_stats()),
        None,
        "an explicit 0-slot override must disable the cache entirely"
    );
    let (uncached_logits, uncached_tokens) = run_greedy(&mut uncached, &prompt_ids, GREEDY_TOKENS);
    drop(uncached);

    // A tiny 3-slot cache: every decode step selects 8 experts per layer
    // across ~40 layers, so this thrashes constantly (heavy eviction, the
    // scenario most likely to expose a slot-reuse bug) while still
    // exercising real hits (the same expert re-selected before its slot is
    // evicted).
    let mut cached = Model::load(&path, LoadOptions::new(CTX).with_moe_cache_slots(Some(3)))
        .expect("load cached model");
    let (cached_logits, cached_tokens) = run_greedy(&mut cached, &prompt_ids, GREEDY_TOKENS);
    let (hits, misses, occupancy, capacity) = cached
        .as_hybrid()
        .and_then(|m| m.moe_cache_stats())
        .expect("a 3-slot override must yield Some cache");
    assert_eq!(capacity, 3);
    assert!(occupancy <= capacity);
    assert!(misses > 0, "a 3-slot cache must miss constantly");
    eprintln!(
        "3-slot cache over {GREEDY_TOKENS} decode steps: {hits} hits, {misses} misses \
         ({:.1}% hit rate)",
        100.0 * hits as f64 / (hits + misses).max(1) as f64
    );
    drop(cached);

    assert_eq!(
        uncached_tokens, cached_tokens,
        "greedy continuation must be identical regardless of caching"
    );
    assert_eq!(
        uncached_logits.len(),
        cached_logits.len(),
        "same number of forward steps"
    );
    for (step, (a, b)) in uncached_logits.iter().zip(cached_logits.iter()).enumerate() {
        assert_eq!(
            a, b,
            "step {step}: logits must be bit-identical between the uncached and cached paths"
        );
    }
}
