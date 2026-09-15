//! Shared assertions for the snapshot-layer equivalence gates
//! (`snapshot_equivalence.rs`, `snapshot_rewind.rs`): near-exact logits plus
//! exact-or-documented-near-tie greedy continuations. See
//! `snapshot_equivalence.rs`'s module doc for why the logits bound is a
//! relative tolerance rather than bitwise equality.

use rocml::Model;

pub const CONTINUATION_LEN: usize = 8;

pub fn synthetic_prompt(len: usize, vocab_size: u32) -> Vec<u32> {
    (0..len as u32).map(|i| (i * 37 + 5) % vocab_size).collect()
}

pub fn top2_gap(logits: &[f32]) -> f32 {
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

pub fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .expect("logits must be non-empty")
}

pub fn assert_logits_close(actual: &[f32], expected: &[f32], rel_tol: f32, label: &str) {
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

/// Greedy-decodes `n` tokens from `logits`, returning each pick with its
/// top-1/top-2 relative gap (the near-tie evidence
/// [`assert_continuations_match`] needs).
pub fn greedy_continue(model: &mut Model, mut logits: Vec<f32>, n: usize) -> Vec<(u32, f32)> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let gap = top2_gap(&logits);
        let next = argmax(&logits);
        out.push((next, gap));
        logits = model.forward_token(next).expect("forward_token failed");
    }
    out
}

/// A disagreement only passes if the loser's own top-1/top-2 gap was already
/// within `near_tie_gap` (a real divergence has a much larger gap and fails).
pub fn assert_continuations_match(
    label: &str,
    full: &[(u32, f32)],
    restored: &[(u32, f32)],
    near_tie_gap: f32,
) {
    for (step, ((full_tok, full_gap), (restored_tok, restored_gap))) in
        full.iter().zip(restored).enumerate()
    {
        if full_tok == restored_tok {
            continue;
        }
        assert!(
            full_gap.abs() <= near_tie_gap || restored_gap.abs() <= near_tie_gap,
            "{label} continuation step {step}: full-prefill picked {full_tok} (gap {full_gap}), \
             restored picked {restored_tok} (gap {restored_gap}) — gap too large to be a \
             documented near-tie"
        );
        eprintln!(
            "{label} continuation step {step}: documented near-tie flip, full={full_tok} (gap \
             {full_gap}) vs restored={restored_tok} (gap {restored_gap})"
        );
    }
}
