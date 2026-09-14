//! Issue #14 phase 2 step 3(a): the mixed-KV parity gates' own greedy-check
//! methodology (`mixed_kv_parity.rs`), run under the debug rotational
//! quantization simulation (`LoadOptions::kv_rot_sim`) instead of asserting
//! anything — this is explicitly informational (the issue's own protocol:
//! "expect some divergence at 2-3 bpw — record where"), not a new
//! correctness gate, and never touches `mixed_kv_parity.rs`'s existing
//! assertions.
//!
//! Compares real q8-mixed-KV greedy decode (K/V both real production Q8) to
//! the same load with `kv_rot_sim` set (V round-tripped through rotational
//! quantization at 2/3/4 bpw before every window eviction, K left alone) —
//! same checkpoint/prompt/methodology `mixed_kv_parity.rs` already
//! validated at production tolerances, just a different comparison target.
//! Uses Qwen3.5-2B (this crate's cheap dev/test model) rather than
//! Ornith-1.0-9B purely for turnaround time; the checked-in calibration
//! sidecar's `head_dim` (256) matches both checkpoints (qwen35's shared
//! attention head shape), confirmed by this test running at all rather
//! than hitting the sidecar's own head_dim mismatch error.
//!
//! `#[ignore]`d like every other diagnostic/measurement tool in this suite
//! (`kv_head_error_measure.rs`, `mmq_layer_diff.rs`) — two full model loads
//! per run, informational rather than a gate. Run via `make
//! rotational-kv-sim-parity`.

use rocml::{KvCacheMode, LoadOptions, Model};
use rocml_core::testpaths::checkpoint;

const GGUF_REL: &str = "Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf";
const CTX: usize = 4096;
const PROMPT_LEN: usize = 32;
/// Past `SINK_LEN(32) + WINDOW_LEN(128) = 160`, so at least one window
/// eviction (and therefore at least one rotational round-trip) actually
/// fires — mirrors `mixed_kv_parity.rs`'s own `GREEDY_TOKENS`.
const GREEDY_TOKENS: usize = 200;

fn synthetic_prompt(len: usize, vocab_size: u32) -> Vec<u32> {
    (0..len as u32).map(|i| (i * 37 + 5) % vocab_size).collect()
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .expect("logits must be non-empty")
}

fn run_greedy(model: &mut Model, prompt_ids: &[u32], n: usize) -> (Vec<f32>, Vec<u32>) {
    model.reset().expect("reset failed");
    let mut logits = Vec::new();
    for &id in prompt_ids {
        logits = model.forward_token(id).expect("forward_token failed");
    }
    let mut tokens = Vec::with_capacity(n);
    for _ in 0..n {
        let next = argmax(&logits);
        tokens.push(next);
        logits = model.forward_token(next).expect("forward_token failed");
    }
    (logits, tokens)
}

#[test]
#[ignore]
fn rotational_sim_vs_real_q8_greedy_divergence_by_bpw() {
    let Some(path) = checkpoint(GGUF_REL) else {
        return;
    };

    let probe = Model::load(&path, LoadOptions::new(CTX)).expect("probe load");
    let vocab_size = probe.vocab_size();
    drop(probe);
    let prompt_ids = synthetic_prompt(PROMPT_LEN, vocab_size);

    let mut model_q8 = Model::load(&path, LoadOptions::new(CTX).with_kv_cache(KvCacheMode::Q8))
        .expect("load real-q8 model");
    let (q8_logits, q8_tokens) = run_greedy(&mut model_q8, &prompt_ids, GREEDY_TOKENS);
    drop(model_q8);
    assert!(
        q8_logits.iter().all(|x| x.is_finite()),
        "real-q8 baseline logits must be finite"
    );

    for &bpw in &[2u8, 3, 4] {
        let opts = LoadOptions::new(CTX)
            .with_kv_cache(KvCacheMode::Q8)
            .with_kv_rot_sim(Some(bpw), false);
        let mut model_sim =
            Model::load(&path, opts).unwrap_or_else(|e| panic!("load {bpw}bpw sim model: {e}"));
        let (sim_logits, sim_tokens) = run_greedy(&mut model_sim, &prompt_ids, GREEDY_TOKENS);
        drop(model_sim);

        assert!(
            sim_logits.iter().all(|x| x.is_finite()),
            "{bpw}bpw sim logits must be finite (the whole point of the simulation is a real, \
             well-formed forward pass — a NaN/inf would mean the round-trip itself is broken, \
             not just lossy)"
        );

        let mut max_rel = 0.0f32;
        for (got, want) in sim_logits.iter().zip(&q8_logits) {
            max_rel = max_rel.max((got - want).abs() / want.abs().max(1.0));
        }
        let mut first_divergence = None;
        let mut divergence_count = 0;
        for (step, (q8_tok, sim_tok)) in q8_tokens.iter().zip(&sim_tokens).enumerate() {
            if q8_tok != sim_tok {
                divergence_count += 1;
                if first_divergence.is_none() {
                    first_divergence = Some(step);
                }
            }
        }
        eprintln!(
            "rotational V-sim @{bpw}bpw vs real-q8: max relative logit diff (decode step \
             {GREEDY_TOKENS}) {max_rel:.6}; greedy divergences {divergence_count}/{GREEDY_TOKENS} \
             (first at step {first_divergence:?})"
        );
        // Informational only — issue #14's own protocol expects real
        // divergence at 2-3 bpw; this is recorded, not gated. See
        // .claude/CLAUDE.md for the numbers this run produced.
    }
}
