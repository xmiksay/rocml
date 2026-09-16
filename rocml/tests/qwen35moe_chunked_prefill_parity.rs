//! M3's central correctness gate: qwen35moe's grouped-by-expert batched
//! chunked-prefill FFN step (`qwen35::forward::moe_chunk::moe_ffn_chunk_step`)
//! must produce the same result as the pre-M3 per-row MoE step still used by
//! `Model::forward_token` (decode) *and*, via
//! `Model::forward_prompt_chunked_captured`, still reachable inside chunked
//! prefill itself (see `ffn_chunk_dispatch.rs`'s `LayerCapture`-diagnosed
//! fallback) — comparing against that captured path, not raw token-serial,
//! isolates exactly what M3 changed.
//!
//! **Why not compare straight against token-serial, the way
//! `qwen35_chunked_prefill_parity.rs` does for the dense/hybrid path**:
//! measured first, before settling on this design. This checkpoint (41
//! layers, wide GQA `group=8`) already shows real chunked-attention-vs-
//! token-serial numerical drift *before* any M3 change — isolated with a
//! 3-way comparison (serial vs. `forward_prompt_chunked_captured` [chunked
//! attention/GDN, *unchanged* per-row MoE] vs. this change's grouped MoE):
//! the captured (pre-M3) path already flips greedy tokens relative to
//! serial on its own. Comparing grouped against the *captured* path instead
//! removes that pre-existing attention-side drift from the measurement
//! entirely, since both run the identical attention/GDN kernels and differ
//! only in the MoE step itself — the actual thing this milestone changed.
//!
//! **Shared-history methodology** (mirrors `dense_mixed_kv_parity.rs`'s
//! `run_shared_history`, adapted to two forward-pass *variants* of the same
//! checkpoint instead of two KV dtypes): two separate `Model` instances load
//! the same checkpoint; one processes the prompt via
//! `forward_prompt_chunked_captured` (pre-M3), the other via
//! `forward_prompt` (M3's grouped path, `Model::forward_prompt` always
//! chunks). From there both decode with the *identical* shared token
//! history via the unchanged `forward_token` decode path (which always
//! runs the old per-row MoE step regardless of which prefill path produced
//! the state it's decoding from) — stopping the first time their argmax
//! choices disagree, since after that point the two would need genuinely
//! different next tokens and a raw logits comparison stops being
//! meaningful. This was necessary because a naive self-driven-continuation
//! comparison (each side feeding back its own greedy pick) showed both
//! sides racing off into unrelated generated text after the first flip,
//! making every subsequent step's "disagreement" an artifact of comparing
//! two different sentences, not a real signal.
//!
//! A max-relative-with-`.max(1.0)`-floor logits bound (the dense-path
//! gate's own metric) is dominated by near-zero-magnitude logit outliers on
//! this checkpoint (a real observed case: `got -0.829, want -0.293`, both
//! small and unremarkable, reports as 54% relative error) — the same
//! failure mode `docs/llama-diff.md` documents needing a "mean_abs, not
//! max_rel" fix for — so the logits-closeness half of this gate uses mean
//! absolute difference over the shared-history comparison point instead.
//!
//! Prompt lengths are chosen to land on both sides of `CHUNK_CAP`/
//! `PREFILL_CHUNK_SIZE` (512) — short (one expert group per used expert
//! within a single tiny chunk), exactly at the chunk boundary, and past it
//! (two chunks, exercising that per-chunk expert grouping resets and the
//! second chunk's `xn` is a fresh rmsnorm of that chunk's own rows, not a
//! continuation of the first's `MoeChunkHost` bucket state).
//!
//! Real hardware + the real Ornith-1.5-35B-A3B-Q4_K_M checkpoint required;
//! skips itself if absent. Run via `make test-model` (`--release`).

use rocml::qwen35::forward::layer_capture::LayerCapture;
use rocml::{KvCacheMode, LoadOptions, Model};
use rocml_core::testpaths::checkpoint;

const GGUF_REL: &str = "Ornith-1.5-35B-A3B-GGUF/Ornith-1.5-35B-Q4_K_M.gguf";
const PROMPT_LENGTHS: &[usize] = &[1, 4, 127, 128, 129, 511, 512, 513, 600];
const SHARED_HISTORY_STEPS: usize = 6;
/// Mean absolute difference over the full `[vocab_size]` logits vector, at
/// the shared-history comparison point (see the module doc) — calibrated
/// with headroom above this test's own measured maximum (~0.47, at the
/// `len=600` case whose first divergence lands three shared-history steps
/// in rather than at the prefill boundary itself) across every prompt
/// length below.
const MEAN_ABS_TOL: f32 = 0.6;
/// Wider than the dense/hybrid gates' own `1e-2` (measured: this deeper,
/// wider-GQA checkpoint's genuine first-divergence gaps ran up to ~0.016 on
/// the smaller of the two sides — still a real near-tie, not a confident
/// flip, just a looser one than the smaller models this constant was
/// originally picked against).
const NEAR_TIE_RELATIVE_GAP: f32 = 2e-2;

/// Deterministic, valid (`< vocab_size`) but otherwise arbitrary token ids —
/// this test checks numerical equivalence of two forward-pass code paths,
/// not language-model output quality.
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

fn mean_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len() as f32
}

/// See the module doc's "Shared-history methodology" section.
struct SharedHistoryResult {
    divergence_step: Option<usize>,
    gap_captured: f32,
    gap_grouped: f32,
    logits_captured: Vec<f32>,
    logits_grouped: Vec<f32>,
}

fn run_shared_history(
    model_captured: &mut Model,
    model_grouped: &mut Model,
    prompt_ids: &[u32],
    n: usize,
) -> SharedHistoryResult {
    model_captured.reset().expect("reset failed");
    model_grouped.reset().expect("reset failed");

    let mut cap = LayerCapture::new();
    let mut logits_captured = model_captured
        .as_hybrid_mut()
        .expect("qwen35moe checkpoint")
        .forward_prompt_chunked_captured(prompt_ids, None, &mut cap)
        .expect("forward_prompt_chunked_captured failed");
    let mut logits_grouped = model_grouped
        .forward_prompt(prompt_ids, None)
        .expect("forward_prompt failed");

    for step in 0..n {
        let (tok_c, tok_g) = (argmax(&logits_captured), argmax(&logits_grouped));
        if tok_c != tok_g {
            return SharedHistoryResult {
                divergence_step: Some(step),
                gap_captured: top2_gap(&logits_captured),
                gap_grouped: top2_gap(&logits_grouped),
                logits_captured,
                logits_grouped,
            };
        }
        // Still agreeing — feed the (shared) chosen token to both so the
        // next step's comparison is still against an identical history.
        logits_captured = model_captured
            .forward_token(tok_c)
            .expect("forward_token failed");
        logits_grouped = model_grouped
            .forward_token(tok_c)
            .expect("forward_token failed");
    }
    SharedHistoryResult {
        divergence_step: None,
        gap_captured: top2_gap(&logits_captured),
        gap_grouped: top2_gap(&logits_grouped),
        logits_captured,
        logits_grouped,
    }
}

fn assert_shared_history_gate(label: &str, r: &SharedHistoryResult) {
    let mean_abs = mean_abs_diff(&r.logits_captured, &r.logits_grouped);
    match r.divergence_step {
        None => eprintln!(
            "{label}: no greedy divergence over {SHARED_HISTORY_STEPS} steps, mean_abs={mean_abs:.6}"
        ),
        Some(step) => eprintln!(
            "{label}: first divergence at step {step}, shared-history mean abs logit diff \
             {mean_abs:.6}, gap_captured={:.6} gap_grouped={:.6}",
            r.gap_captured, r.gap_grouped
        ),
    }
    if let Some(step) = r.divergence_step {
        assert!(
            r.gap_captured.abs() <= NEAR_TIE_RELATIVE_GAP
                || r.gap_grouped.abs() <= NEAR_TIE_RELATIVE_GAP,
            "{label}: greedy decode diverged at step {step} with neither gap near a tie \
             (gap_captured={}, gap_grouped={}) — a flip this early with a wide margin would \
             indicate a real bug, not expected numeric noise",
            r.gap_captured,
            r.gap_grouped
        );
    }
    assert!(
        mean_abs <= MEAN_ABS_TOL,
        "{label}: shared-history mean abs logit diff {mean_abs} exceeds tolerance {MEAN_ABS_TOL}"
    );
}

#[test]
fn ornith_35b_moe_chunked_prefill_matches_token_serial() {
    let Some(path) = checkpoint(GGUF_REL) else {
        return;
    };
    // f32 KV, same reasoning as `qwen35_chunked_prefill_parity.rs`: pins the
    // attention-side numerics so this test isolates the MoE FFN grouping
    // change specifically, not also the default fp16 cache's own rounding.
    // Two model instances share one 16GB card (see the module doc's
    // "Shared-history methodology") — each capped to a modest expert-cache
    // slot count so both fit together; this test only needs a handful of
    // prompt lengths through a few layers, not decode throughput.
    let opts = || {
        LoadOptions::new(4096)
            .with_kv_cache(KvCacheMode::F32)
            .with_moe_cache_slots(Some(1024))
    };
    let mut model_captured = Model::load(&path, opts()).expect("load captured-path model");
    let mut model_grouped = Model::load(&path, opts()).expect("load grouped-path model");
    let vocab_size = model_captured.vocab_size();

    for &len in PROMPT_LENGTHS {
        let prompt_ids = synthetic_prompt(len, vocab_size);
        let result = run_shared_history(
            &mut model_captured,
            &mut model_grouped,
            &prompt_ids,
            SHARED_HISTORY_STEPS,
        );
        assert_shared_history_gate(&format!("len={len}"), &result);
    }
}
