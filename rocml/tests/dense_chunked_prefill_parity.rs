//! Issue #16's central correctness gate for the dense chunked-prefill port:
//! chunked prefill (`Model::forward_prompt`, which now batches the dense
//! Qwen3 forward pass into `PREFILL_CHUNK_SIZE`-token chunks — see
//! `forward::chunk_forward`) must produce the same result as the original
//! token-serial path (`Model::forward_token`, still callable and otherwise
//! untouched), across prompt lengths chosen to land on both sides of the
//! chunk-size boundary (512) plus a few off-by-one and short/long extremes.
//! Mirrors `qwen35_chunked_prefill_parity.rs`'s design and tolerance
//! rationale exactly — see that file's module doc for why exact bit-
//! identity isn't the bar (reduction-order/WMMA precision differences
//! between the batched-GEMM and token-serial paths).
//!
//! Real hardware + the real Qwen3-0.6B-Q8_0 checkpoint required; skips
//! itself if absent. Run via `make test-model` (`--release`: prompt lengths
//! up to 2048 through the token-serial path are unbearably slow in debug).

use rocml::{KvCacheMode, LoadOptions, Model};
use rocml_core::testpaths::checkpoint;

const GGUF_REL: &str = "Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf";
const PROMPT_LENGTHS: &[usize] = &[1, 4, 127, 128, 129, 500, 512, 513, 1024, 2048];
/// Prompt lengths for the mixed-KV-cache variant below, chosen to land on
/// both sides of the sink/window boundaries (`SINK_LEN`=32, `WINDOW_LEN`=128)
/// — mirrors `mixed_kv_chunked_prefill_parity.rs`'s own prompt lengths.
const MIXED_PROMPT_LENGTHS: &[usize] = &[32, 128, 160, 500, 2048];
const CONTINUATION_LEN: usize = 8;
/// See `qwen35_chunked_prefill_parity.rs`'s identical constant for the
/// rationale (WMMA f16-matrix-core GEMM at `rows >= 128` vs the token-serial
/// path's scalar accumulation — a precision-mode comparison, not just a
/// reduction-order one).
const LOGITS_REL_TOL: f32 = 1e-2;
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

fn assert_logits_close(actual: &[f32], expected: &[f32], rel_tol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        let diff = (got - want).abs();
        let tol = rel_tol * want.abs().max(1.0);
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

/// Asserts a step's chosen token either matches outright or, on a
/// disagreement, that at least one side's own top-1/top-2 gap was already a
/// documented near-tie (a large gap on both sides means the two paths
/// picked meaningfully different tokens — a real bug, not float-order or
/// quantization noise).
fn assert_token_matches_or_near_tie(
    label: &str,
    step_label: &str,
    serial_tok: u32,
    serial_gap: f32,
    chunked_tok: u32,
    chunked_gap: f32,
) {
    if serial_tok == chunked_tok {
        return;
    }
    assert!(
        serial_gap.abs() <= NEAR_TIE_RELATIVE_GAP || chunked_gap.abs() <= NEAR_TIE_RELATIVE_GAP,
        "{label} {step_label}: serial picked {serial_tok} (gap {serial_gap}), chunked picked \
         {chunked_tok} (gap {chunked_gap}) — gap too large to be a documented near-tie"
    );
    eprintln!(
        "{label} {step_label}: documented near-tie flip, serial={serial_tok} (gap {serial_gap}) \
         vs chunked={chunked_tok} (gap {chunked_gap})"
    );
}

/// `logits_rel_tol`: `Some(tol)` asserts the final logits are element-wise
/// close (the dense fp16/f32-cache case, where both paths compute the exact
/// same math up to reduction order); `None` instead checks the final
/// logits' *argmax* the same near-tie-or-match way the continuation steps
/// are checked — needed for the mixed/quantized KV cache case, where
/// `dense_mixed_kv_parity.rs` already established that Qwen3-0.6B's 26
/// consecutive mixed layers (vs. the qwen35 hybrid's 4) compound enough
/// quantization noise that individual small-magnitude logits can show a
/// large *relative* deviation despite the argmax decision itself being
/// stable — the same "small-magnitude values dominate a relative metric"
/// pitfall `mixed_kv_chunked_prefill_parity.rs` documents for its own cache-
/// byte comparison.
fn run_case(model: &mut Model, prompt_ids: &[u32], logits_rel_tol: Option<f32>) {
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
    match logits_rel_tol {
        Some(tol) => assert_logits_close(
            &chunked_logits,
            &serial_logits,
            tol,
            &format!("{label} final logits"),
        ),
        None => assert_token_matches_or_near_tie(
            &label,
            "final logits argmax",
            argmax(&serial_logits),
            top2_gap(&serial_logits),
            argmax(&chunked_logits),
            top2_gap(&chunked_logits),
        ),
    }

    for (step, ((serial_tok, serial_gap), (chunked_tok, chunked_gap))) in serial_continuation
        .iter()
        .zip(&chunked_continuation)
        .enumerate()
    {
        assert_token_matches_or_near_tie(
            &label,
            &format!("continuation step {step}"),
            *serial_tok,
            *serial_gap,
            *chunked_tok,
            *chunked_gap,
        );
    }
}

#[test]
fn qwen3_0_6b_chunked_prefill_matches_token_serial() {
    let Some(path) = checkpoint(GGUF_REL) else {
        return;
    };
    // f32 KV explicitly, same reasoning as qwen35_chunked_prefill_parity.rs:
    // pins the reference numerics this test's near-tie escape hatch is
    // calibrated against, independent of the default f16 cache's rounding.
    let mut model = Model::load(
        &path,
        LoadOptions::new(4096).with_kv_cache(KvCacheMode::F32),
    )
    .expect("Model::load failed");
    let vocab_size = model.vocab_size();

    for &len in PROMPT_LENGTHS {
        let prompt_ids = synthetic_prompt(len, vocab_size);
        run_case(&mut model, &prompt_ids, Some(LOGITS_REL_TOL));
    }
}

/// Chunked-vs-token-serial parity over the KIVI-style mixed/quantized KV
/// cache (issue #2/#16's dense mixed-KV port), for both `Q8` and `Q4Mixed` —
/// the dense-architecture analogue of `mixed_kv_chunked_prefill_parity.rs`.
/// Lighter-weight than that suite: it checks the externally observable
/// behavior (final-logits argmax + exact-or-near-tie greedy continuation,
/// see `run_case`'s doc comment for why raw logit magnitude isn't the right
/// bound here) rather than also asserting internal `window_base` eviction-
/// bookkeeping exactness, since that bookkeeping is the same architecture-
/// generic `MixedLayout`/`MixedAttnPlane` code already proven correct by
/// `layout::chunk_plan`'s own pure-Rust fuzz tests and the hybrid
/// architecture's own gate — nothing here is dense-specific.
#[test]
fn qwen3_0_6b_chunked_prefill_matches_token_serial_mixed_kv() {
    let Some(path) = checkpoint(GGUF_REL) else {
        return;
    };
    for mode in [KvCacheMode::Q8, KvCacheMode::Q4Mixed] {
        let mut model = Model::load(&path, LoadOptions::new(4096).with_kv_cache(mode))
            .unwrap_or_else(|e| panic!("Model::load failed for {mode:?}: {e}"));
        let vocab_size = model.vocab_size();

        for &len in MIXED_PROMPT_LENGTHS {
            let prompt_ids = synthetic_prompt(len, vocab_size);
            run_case(&mut model, &prompt_ids, None);
        }
    }
}
