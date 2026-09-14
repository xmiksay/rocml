//! Issue #2/#16's dense-architecture mixed-KV port — the `kv_dtype_parity`/
//! `mixed_kv_parity`-style correctness gates, extended to the dense `qwen3`
//! architecture (`rocml/src/forward/`).
//!
//! **Checkpoint note**: the task that requested this gate named "qwen3.5-2b
//! Q8_0" as the dense model, but that checkpoint's GGUF metadata is actually
//! `general.architecture = "qwen35"` (the hybrid Gated-Delta-Net +
//! full-attention arch — confirmed via `rocml-core`'s `inspect_gguf`
//! example: `qwen35.block_count: 24`, no `qwen3.*` keys at all), i.e. it
//! runs through `qwen35::forward::Model`, not `crate::forward::Model`. The
//! only genuinely dense (`general.architecture = "qwen3"`) checkpoint this
//! project's test suite resolves is `Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf`
//! (28 layers, confirmed `qwen3.block_count: 28`, no GDN layers at all —
//! see `greedy_parity.rs`), so that's what this file uses. Every layer here
//! is a full-attention layer (the dense config has no `layer_kinds` concept
//! at all — see `ModelConfig`), so `DenseAttnCache::new`'s boundary rule
//! (first + last layer stay dense fp16) leaves `28 - 2 = 26` layers eligible
//! for the mixed cache — far more than Qwen3.5-2B's 4, and with **no GDN
//! layers interleaved** to dampen compounding error between them (unlike
//! the hybrid architecture, where at most a handful of full-attention
//! layers are ever adjacent).
//!
//! **Decode-path-only, by design**: every test below drives both models
//! token-by-token via `forward_token` directly (never `forward_prompt`), so
//! it exercises the mixed-cache append/attend kernels identically regardless
//! of whether the dense architecture's prefill phase is token-serial or
//! chunked. As of issue #16's dense chunked-prefill port,
//! `crate::model::Model::forward_prompt`'s `Dense` arm *does* batch the
//! prompt into `PREFILL_CHUNK_SIZE`-token chunks (`forward::chunk_forward`,
//! reusing the same `AttnLayerCache`/mixed-cache dispatch this file's
//! `forward_token`-driven tests already cover) — see
//! `dense_chunked_prefill_parity.rs` for the chunked-vs-token-serial gate
//! itself, including for the mixed KV cache modes.
//!
//! **Methodology difference from `mixed_kv_parity.rs`, and why**: that
//! suite's q4-mixed gate asserts `divergence_count == 0` over 200 self-driven
//! greedy-decode steps — true for Qwen3.5-2B (24 layers, only 4 of them
//! mixed). On Qwen3-0.6B (26 consecutive mixed layers, no GDN interleaving,
//! and a much smaller model with correspondingly tighter top-1/top-2
//! margins), real measurement shows quantization-induced argmax flips *do*
//! happen well within 200 steps — confirmed to be genuine near-ties, not a
//! bug: the first divergence's top-2 gap was 6.4e-5 (q8) / comparably tiny
//! for q4, both far inside the existing `NEAR_TIE_RELATIVE_GAP` (1e-2) this
//! codebase already uses for the identical situation in `kv_dtype_parity.rs`
//! (also confirmed: fp16-vs-f32 KV storage alone shows *zero* argmax
//! divergence over the same 200 steps on this checkpoint, so the sensitivity
//! is specifically to the mixed cache's real quantization error compounding
//! over 26 consecutive layers, not an artifact of this being a small model
//! in general). Once a real divergence happens, the two self-driven
//! trajectories generate *different, unrelated* token sequences — comparing
//! their raw logit magnitudes past that point (as `mixed_kv_parity.rs` does
//! at a fixed late step) is not a meaningful quantization-error signal, so
//! these tests instead measure the logits relative diff at the *last step
//! both models were still driven by an identical token history* (i.e. right
//! up to and including the first divergent step, whose input history is
//! still shared) and assert only that: (a) that first divergence, if any,
//! is a documented near-tie (the real correctness gate — a *hard* flip with
//! a wide margin would indicate an actual bug); (b) the shared-history
//! logits relative diff is bounded. Anything after the first divergence is
//! reported, never asserted.
//!
//! Real hardware + the real Qwen3-0.6B-Q8_0 checkpoint required; every test
//! skips itself if absent.

use rocml::{KvCacheMode, LoadOptions, Model};
use rocml_core::testpaths::checkpoint;

const GGUF_REL: &str = "Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf";
const CTX: usize = 4096;
const PROMPT_LEN: usize = 32;
/// Past `SINK_LEN(32) + WINDOW_LEN(128) = 160`, so the bulk region actually
/// gets exercised (at least one full window eviction) on every one of the
/// 26 mixed-eligible layers — mirrors `mixed_kv_parity.rs`'s own reasoning.
const GREEDY_TOKENS: usize = 200;
/// Same threshold `kv_dtype_parity.rs`/`qwen35_chunked_prefill_parity.rs`
/// use for "close enough to be reduction-order/quantization noise, not a
/// real regression".
const NEAR_TIE_RELATIVE_GAP: f32 = 1e-2;

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

fn top2_gap(logits: &[f32]) -> f32 {
    let (mut best, mut second) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for &v in logits {
        if v > best {
            second = best;
            best = v;
        } else if v > second {
            second = v;
        }
    }
    (best - second) / best.abs().max(1.0)
}

fn max_rel_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs() / x.abs().max(1.0))
        .fold(0.0f32, f32::max)
}

/// Result of [`run_shared_history`]: the last pair of logits both models
/// were compared at (either the true final step, if the two never
/// disagreed, or the first divergent step — the last point where both
/// models had been fed an *identical* input history, so a direct logits
/// comparison is still meaningful) plus that step's own top-1/top-2 gaps,
/// and `divergence_step` (`None` if the two never disagreed through `n`
/// steps).
struct SharedHistoryResult {
    divergence_step: Option<usize>,
    gap_a: f32,
    gap_b: f32,
    logits_a: Vec<f32>,
    logits_b: Vec<f32>,
}

/// Feeds `prompt_ids`, then decodes up to `n` further steps with **shared**
/// history: at each step both models are fed the identical next token (`a`'s
/// own greedy choice, arbitrary once both agree) as long as their argmax
/// choices keep matching. Stops early (before feeding a further token) the
/// first time the two models' argmax choices differ, since from that point
/// on there is no single "next token" both models agree the shared history
/// should continue with.
fn run_shared_history(
    model_a: &mut Model,
    model_b: &mut Model,
    prompt_ids: &[u32],
    n: usize,
) -> SharedHistoryResult {
    model_a.reset().expect("reset failed");
    model_b.reset().expect("reset failed");
    let mut logits_a = Vec::new();
    let mut logits_b = Vec::new();
    for &id in prompt_ids {
        logits_a = model_a.forward_token(id).expect("forward_token failed");
        logits_b = model_b.forward_token(id).expect("forward_token failed");
    }

    for step in 0..n {
        let (tok_a, tok_b) = (argmax(&logits_a), argmax(&logits_b));
        if tok_a != tok_b {
            return SharedHistoryResult {
                divergence_step: Some(step),
                gap_a: top2_gap(&logits_a),
                gap_b: top2_gap(&logits_b),
                logits_a,
                logits_b,
            };
        }
        // Still agreeing — feed the (shared) chosen token to both models so
        // the next step's comparison is still against an identical history.
        logits_a = model_a.forward_token(tok_a).expect("forward_token failed");
        logits_b = model_b.forward_token(tok_a).expect("forward_token failed");
    }
    SharedHistoryResult {
        divergence_step: None,
        gap_a: top2_gap(&logits_a),
        gap_b: top2_gap(&logits_b),
        logits_a,
        logits_b,
    }
}

fn compare_against_fp16(mode: KvCacheMode, prompt_ids: &[u32]) -> SharedHistoryResult {
    let path = checkpoint(GGUF_REL).expect("checkpoint present (caller already checked)");
    let mut model_fp16 = Model::load(&path, LoadOptions::new(CTX)).expect("load fp16-KV model");
    let mut model_mixed = Model::load(&path, LoadOptions::new(CTX).with_kv_cache(mode))
        .unwrap_or_else(|e| panic!("load {mode:?}-KV model: {e}"));

    run_shared_history(&mut model_fp16, &mut model_mixed, prompt_ids, GREEDY_TOKENS)
}

/// Asserts the two behavioral gates every test below shares: (a) if the two
/// models' greedy choices ever diverge within `GREEDY_TOKENS`, that first
/// divergence must be a documented near-tie (a *hard* flip with a wide
/// margin this early would mean a real bug, not expected numeric noise);
/// (b) the shared-history logits (the true final step if no divergence
/// happened, else the divergence step itself — see [`SharedHistoryResult`])
/// stay within `tol`, a bound measured and justified per call site, never
/// guessed.
fn assert_shared_history_gate(label: &str, r: &SharedHistoryResult, tol: f32) {
    let max_rel = max_rel_diff(&r.logits_a, &r.logits_b);
    match r.divergence_step {
        None => eprintln!(
            "{label}: no greedy divergence over {GREEDY_TOKENS} steps, max_rel={max_rel:.6}"
        ),
        Some(step) => eprintln!(
            "{label}: first divergence at step {step}, shared-history max relative logit diff \
             {max_rel:.6}, gap_a={:.6} gap_b={:.6}",
            r.gap_a, r.gap_b
        ),
    }
    if let Some(step) = r.divergence_step {
        assert!(
            r.gap_a.abs() <= NEAR_TIE_RELATIVE_GAP || r.gap_b.abs() <= NEAR_TIE_RELATIVE_GAP,
            "{label}: greedy decode diverged at step {step} with neither gap near a tie \
             (gap_a={}, gap_b={}) — a flip this early with a wide margin would indicate a real \
             bug, not expected numeric noise",
            r.gap_a,
            r.gap_b
        );
    }
    assert!(
        max_rel <= tol,
        "{label}: shared-history max relative logit diff {max_rel} exceeds the \
         measured/justified bound {tol}"
    );
}

/// `kv_dtype_parity`-style gate, ported to the dense architecture: the
/// default f16 KV cache must be numerically close to the pre-issue-#3 f32
/// reference cache. Measured on this checkpoint: no greedy divergence over
/// 200 steps, final-step (step 200) max relative logit diff 0.0098 (vs
/// qwen35's own 1.4e-3 on its own checkpoint, at the prompt's logits only —
/// a real, checkpoint- and measurement-point-specific difference, not a
/// regression). `LOGITS_REL_TOL` is set with headroom above that
/// measurement rather than reused from the hybrid architecture's own
/// (smaller, different-checkpoint, different-measurement-point) figure.
#[test]
fn dense_fp16_kv_matches_f32_kv_logits_and_greedy_decode() {
    let Some(path) = checkpoint(GGUF_REL) else {
        eprintln!("skipping: {GGUF_REL} not found");
        return;
    };
    const LOGITS_REL_TOL: f32 = 1.5e-2;

    let mut model_f32 = Model::load(&path, LoadOptions::new(CTX).with_kv_cache(KvCacheMode::F32))
        .expect("load f32-KV model");
    let mut model_f16 = Model::load(&path, LoadOptions::new(CTX)).expect("load f16-KV model");
    let vocab_size = model_f32.vocab_size();
    let prompt_ids = synthetic_prompt(PROMPT_LEN, vocab_size);

    let result = run_shared_history(&mut model_f32, &mut model_f16, &prompt_ids, GREEDY_TOKENS);
    assert_shared_history_gate("dense fp16 vs f32", &result, LOGITS_REL_TOL);
}

/// `mixed_kv_parity`-style gate, ported to the dense architecture: q8-mixed
/// KV cache vs the fp16 default, over the 26 mixed-eligible layers
/// (`DenseAttnCache`'s boundary rule leaves layers 0 and 27 always fp16).
/// See this file's module doc for why this measures shared-history logits
/// at the first divergence rather than raw logits at a fixed late step.
/// Measured on real hardware: first divergence at step 161 (past
/// `SINK_LEN+WINDOW_LEN=160`, the first eviction), a documented near-tie
/// (both gaps `<1e-3`), shared-history max relative logit diff 0.041 —
/// `LOGITS_REL_TOL` set with headroom above that measurement.
#[test]
fn dense_q8_mixed_kv_vs_fp16_logits_and_greedy_stability() {
    let Some(path) = checkpoint(GGUF_REL) else {
        eprintln!("skipping: {GGUF_REL} not found");
        return;
    };
    const LOGITS_REL_TOL: f32 = 0.1;

    let probe = Model::load(&path, LoadOptions::new(CTX)).expect("probe load");
    let vocab_size = probe.vocab_size();
    drop(probe);
    let prompt_ids = synthetic_prompt(PROMPT_LEN, vocab_size);

    let result = compare_against_fp16(KvCacheMode::Q8, &prompt_ids);
    assert_shared_history_gate("dense q8-mixed vs fp16", &result, LOGITS_REL_TOL);
}

/// Like the above, at `q4-mixed` (the production-recommended default) — a
/// coarser quantization step than q8, so a looser (but still measured, not
/// guessed) bound. Measured: first divergence at step 161 (same step as q8
/// — driven by K's shared Q8-regardless-of-mode quantization, per this
/// file's module doc), a documented near-tie, shared-history max relative
/// logit diff 0.295.
#[test]
fn dense_q4_mixed_kv_vs_fp16_logits_and_greedy_stability() {
    let Some(path) = checkpoint(GGUF_REL) else {
        eprintln!("skipping: {GGUF_REL} not found");
        return;
    };
    const LOGITS_REL_TOL: f32 = 0.4;

    let probe = Model::load(&path, LoadOptions::new(CTX)).expect("probe load");
    let vocab_size = probe.vocab_size();
    drop(probe);
    let prompt_ids = synthetic_prompt(PROMPT_LEN, vocab_size);

    let result = compare_against_fp16(KvCacheMode::Q4Mixed, &prompt_ids);
    assert_shared_history_gate("dense q4-mixed vs fp16", &result, LOGITS_REL_TOL);
}
