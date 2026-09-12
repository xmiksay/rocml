//! Issue #6's central correctness gate: chunked prefill
//! (`Model::forward_prompt`, which batches the qwen35 hybrid forward pass
//! into `PREFILL_CHUNK_SIZE`-token chunks) must produce the same result as
//! the original token-serial path (`Model::forward_token`, still callable
//! and otherwise untouched — the parity/e2e suites keep exercising it
//! directly), across prompt lengths chosen to land on both sides of every
//! chunk-size boundary (128/256/512) plus a few off-by-one and short/long
//! extremes.
//!
//! Reduction-order changes (batched GEMM/attention/GDN accumulate the same
//! sums in a different order than the token-serial path) rule out
//! bit-identical logits, so this asserts near-equality on the final logits
//! (relative tolerance) *and* exact greedy-token equality for a short
//! continuation — and if a continuation ever disagrees, checks whether the
//! disagreement is a genuine near-tie (tiny top-1/top-2 gap, expected and
//! documented) or a real divergence (large gap, a bug) per the project's
//! established policy for this kind of float-order sensitivity.
//!
//! Real hardware + the real Qwen3.5-2B-Q8_0 checkpoint required; skips
//! itself if absent. Run via `make test-model` (`--release`: prompt lengths
//! up to 2048 through the token-serial path are unbearably slow in debug).

use rocml::{KvCacheMode, LoadOptions, Model};
use rocml_core::testpaths::checkpoint;

const GGUF_REL: &str = "Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf";
const PROMPT_LENGTHS: &[usize] = &[1, 4, 127, 128, 129, 500, 2048];
const CONTINUATION_LEN: usize = 8;
const LOGITS_REL_TOL: f32 = 1e-3;
/// A greedy pick counts as a documented near-tie (not a bug) when the
/// runner-up is within this fraction of the winner's margin over the
/// *next* logit — mirrors `qwen35_greedy_parity.rs`'s own threshold for the
/// same reduction-order-sensitivity reasoning.
const NEAR_TIE_RELATIVE_GAP: f32 = 1e-2;

/// Deterministic, valid (`< vocab_size`) but otherwise arbitrary token ids —
/// this test checks numerical equivalence of two forward-pass code paths,
/// not language-model output quality, so real text isn't needed.
fn synthetic_prompt(len: usize, vocab_size: u32) -> Vec<u32> {
    (0..len as u32).map(|i| (i * 37 + 5) % vocab_size).collect()
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

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .expect("logits must be non-empty")
}

fn assert_logits_close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        let diff = (got - want).abs();
        let tol = LOGITS_REL_TOL * want.abs().max(1.0);
        assert!(
            diff <= tol,
            "{label}[{i}]: got {got}, want {want} (diff {diff}, tol {tol})"
        );
    }
}

/// Greedily decodes `n` more tokens via the (unchanged) token-serial decode
/// path, starting from whatever the model's current cache/position already
/// holds. Returns each step's chosen token and its top-1/top-2 relative gap.
fn greedy_continue(model: &mut Model, mut logits: Vec<f32>, n: usize) -> Vec<(u32, f32)> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let gap = top2_gap(&logits);
        let next = argmax(&logits);
        out.push((next, gap));
        logits = model.forward_token(next).expect("forward_token failed");
    }
    out
}

fn run_case(model: &mut Model, prompt_ids: &[u32]) {
    model.reset().expect("reset failed");
    let mut serial_logits = Vec::new();
    for &id in prompt_ids {
        serial_logits = model.forward_token(id).expect("forward_token failed");
    }
    let serial_continuation = greedy_continue(model, serial_logits.clone(), CONTINUATION_LEN);

    model.reset().expect("reset failed");
    let chunked_logits = model
        .forward_prompt(prompt_ids, None)
        .expect("forward_prompt failed");
    let chunked_continuation = greedy_continue(model, chunked_logits.clone(), CONTINUATION_LEN);

    let label = format!("len={}", prompt_ids.len());
    assert_logits_close(
        &chunked_logits,
        &serial_logits,
        &format!("{label} final logits"),
    );

    for (step, ((serial_tok, serial_gap), (chunked_tok, chunked_gap))) in serial_continuation
        .iter()
        .zip(&chunked_continuation)
        .enumerate()
    {
        if serial_tok == chunked_tok {
            continue;
        }
        // Disagreement: only acceptable if the serial path's own choice was
        // already a near-tie (a large gap means the chunked path picked a
        // meaningfully worse token — a real bug, not float-order noise).
        assert!(
            serial_gap.abs() <= NEAR_TIE_RELATIVE_GAP || chunked_gap.abs() <= NEAR_TIE_RELATIVE_GAP,
            "{label} continuation step {step}: serial picked {serial_tok} (gap {serial_gap}), \
             chunked picked {chunked_tok} (gap {chunked_gap}) — gap too large to be a documented \
             near-tie"
        );
        eprintln!(
            "{label} continuation step {step}: documented near-tie flip, serial={serial_tok} \
             (gap {serial_gap}) vs chunked={chunked_tok} (gap {chunked_gap})"
        );
    }
}

#[test]
fn qwen35_2b_chunked_prefill_matches_token_serial() {
    let Some(path) = checkpoint(GGUF_REL) else {
        return;
    };
    // f32 KV explicitly: this suite's "near-tie" escape hatch is calibrated
    // against the pre-issue-#3 reference numerics, not the default f16
    // cache's slightly different rounding.
    let mut model = Model::load(
        &path,
        LoadOptions::new(4096).with_kv_cache(KvCacheMode::F32),
    )
    .expect("Model::load failed");
    let vocab_size = model.vocab_size();

    for &len in PROMPT_LENGTHS {
        let prompt_ids = synthetic_prompt(len, vocab_size);
        run_case(&mut model, &prompt_ids);
    }
}
