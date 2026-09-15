//! Host-side sampling over a forward pass's f32 logits: temperature /
//! top-k / top-p, falling back to greedy argmax at `temperature <= 0`, plus
//! an optional repeat penalty over recently generated tokens. Runs on the
//! CPU after logits are copied back from the GPU — vocab is at most ~250k
//! entries, small enough that an `O(V log V)` sort per token is negligible
//! next to the GPU forward pass it follows.
//!
//! Determinism: sampling is driven by [`Rng`], a tiny inline xorshift64*
//! generator seeded directly from a `u64` — no `rand` dependency (workspace
//! policy: avoid extra deps for something this small), and no cryptographic
//! quality needed, just bit-for-bit repeatability across runs given the same
//! seed and inputs.

use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingParams {
    /// `<= 0.0` selects greedy argmax, ignoring every other field below.
    pub temperature: f32,
    pub top_k: Option<usize>,
    /// Nucleus sampling threshold in `(0, 1]`; `None` disables it.
    pub top_p: Option<f32>,
    pub seed: u64,
    /// `None` (the default) disables the repeat penalty entirely.
    pub repeat_penalty: Option<f32>,
    /// How many of the most recently generated tokens the penalty looks at.
    pub repeat_penalty_window: usize,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: None,
            top_p: None,
            seed: 0,
            repeat_penalty: None,
            repeat_penalty_window: 64,
        }
    }
}

impl SamplingParams {
    pub fn greedy() -> Self {
        Self::default()
    }

    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
    }
}

/// A tiny xorshift64* PRNG. Not cryptographically secure — it doesn't need
/// to be, only deterministic and cheap.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    /// Xorshift's state must never be all-zero (it would stay zero forever),
    /// so a zero seed is remapped to an arbitrary fixed nonzero constant.
    pub fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform float in `[0, 1)`: the top 24 bits of the generator's output
    /// give a value exactly representable as an `f32` mantissa.
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / (1u32 << 24) as f32
    }
}

/// Samples the next token id from `logits`, given the tokens generated so
/// far in `history` (used only by the repeat penalty). Returns 0 for an
/// empty `logits` slice — callers only reach that state once generation is
/// already ending (see `generate::generate`'s `!logits.is_empty()` guard),
/// so this never has an observable effect.
pub fn sample(logits: &[f32], history: &[u32], params: &SamplingParams, rng: &mut Rng) -> u32 {
    if logits.is_empty() {
        return 0;
    }
    let mut scored = logits.to_vec();
    if let Some(penalty) = params.repeat_penalty {
        apply_repeat_penalty(&mut scored, history, penalty, params.repeat_penalty_window);
    }
    if params.is_greedy() {
        return argmax(&scored);
    }

    let inv_temp = 1.0 / params.temperature;
    for v in scored.iter_mut() {
        *v *= inv_temp;
    }

    // `indices` narrows to the top-k candidates (unsorted) when top-k is
    // set, else stays the full vocab.
    let mut indices: Vec<u32> = (0..scored.len() as u32).collect();
    if let Some(k) = params.top_k {
        let k = k.clamp(1, indices.len());
        indices.select_nth_unstable_by(k - 1, |&a, &b| {
            cmp_desc(scored[a as usize], scored[b as usize])
        });
        indices.truncate(k);
    }

    let max_logit = indices
        .iter()
        .map(|&i| scored[i as usize])
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = indices
        .iter()
        .map(|&i| (scored[i as usize] - max_logit).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for p in probs.iter_mut() {
            *p /= sum;
        }
    }

    // `order` indexes into `indices`/`probs`; sampling itself doesn't need a
    // sorted order, only nucleus (top-p) truncation does, so the sort is
    // skipped entirely when top-p isn't requested.
    let mut order: Vec<usize> = (0..indices.len()).collect();
    let mut effective_total = sum;
    if let Some(top_p) = params.top_p {
        let top_p = top_p.clamp(0.0, 1.0);
        order.sort_unstable_by(|&a, &b| cmp_desc(probs[a], probs[b]));
        let mut cumulative = 0.0f32;
        let mut cutoff = order.len();
        for (rank, &idx) in order.iter().enumerate() {
            cumulative += probs[idx];
            if cumulative >= top_p {
                cutoff = rank + 1;
                break;
            }
        }
        cutoff = cutoff.max(1);
        order.truncate(cutoff);
        effective_total = order.iter().map(|&i| probs[i]).sum();
    }

    let threshold = rng.next_f32() * effective_total.max(f32::MIN_POSITIVE);
    let mut acc = 0.0f32;
    for &i in &order {
        acc += probs[i];
        if acc >= threshold {
            return indices[i];
        }
    }
    // Float rounding can leave `threshold` a hair above the accumulated
    // total; the highest-probability survivor is the reasonable fallback.
    indices[order[0]]
}

fn cmp_desc(a: f32, b: f32) -> Ordering {
    b.partial_cmp(&a).unwrap_or(Ordering::Equal)
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

/// Standard llama.cpp-style repetition penalty: a token seen within the last
/// `window` history entries has its logit divided (if positive) or
/// multiplied (if non-positive) by `penalty`, pushing it down either way for
/// `penalty > 1.0`.
fn apply_repeat_penalty(logits: &mut [f32], history: &[u32], penalty: f32, window: usize) {
    if penalty <= 0.0 || (penalty - 1.0).abs() < f32::EPSILON {
        return;
    }
    let start = history.len().saturating_sub(window);
    for &id in &history[start..] {
        if let Some(logit) = logits.get_mut(id as usize) {
            *logit = if *logit > 0.0 {
                *logit / penalty
            } else {
                *logit * penalty
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_logits(n: usize, peak: usize) -> Vec<f32> {
        let mut v: Vec<f32> = (0..n).map(|i| -(i as f32)).collect();
        v[peak] = 100.0;
        v
    }

    #[test]
    fn temperature_zero_is_argmax() {
        let logits = vec![0.1, 5.0, -3.0, 4.9];
        let params = SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        };
        let mut rng = Rng::new(42);
        assert_eq!(sample(&logits, &[], &params, &mut rng), 1);
    }

    #[test]
    fn top_k_one_matches_argmax() {
        let logits = make_logits(50, 7);
        let params = SamplingParams {
            temperature: 0.8,
            top_k: Some(1),
            ..SamplingParams::default()
        };
        let mut rng = Rng::new(123);
        assert_eq!(sample(&logits, &[], &params, &mut rng), 7);
    }

    #[test]
    fn sampled_token_is_within_top_k_set() {
        // Strictly increasing (all distinct, no ties) with one dominant
        // spike, so "the top 5 by value" is unambiguous.
        let mut logits: Vec<f32> = (0..200).map(|i| i as f32 * 0.01).collect();
        logits[20] = 1000.0;
        let params = SamplingParams {
            temperature: 1.0,
            top_k: Some(5),
            ..SamplingParams::default()
        };
        let mut top5: Vec<u32> = (0..logits.len() as u32).collect();
        top5.sort_unstable_by(|&a, &b| cmp_desc(logits[a as usize], logits[b as usize]));
        top5.truncate(5);

        for seed in 0..20u64 {
            let mut rng = Rng::new(seed);
            let picked = sample(&logits, &[], &params, &mut rng);
            assert!(top5.contains(&picked), "{picked} not in top-5 {top5:?}");
        }
    }

    #[test]
    fn sampled_token_respects_top_p_nucleus() {
        let mut logits = vec![0.0f32; 100];
        // Two dominant tokens carrying almost all probability mass; top-p
        // 0.5 should restrict sampling to just the single largest of them.
        logits[3] = 10.0;
        logits[9] = 9.0;
        let params = SamplingParams {
            temperature: 1.0,
            top_p: Some(0.5),
            ..SamplingParams::default()
        };
        for seed in 0..20u64 {
            let mut rng = Rng::new(seed);
            let picked = sample(&logits, &[], &params, &mut rng);
            assert_eq!(picked, 3);
        }
    }

    #[test]
    fn determinism_across_runs() {
        let logits = make_logits(500, 42);
        let params = SamplingParams {
            temperature: 0.9,
            top_k: Some(40),
            top_p: Some(0.95),
            seed: 7,
            ..SamplingParams::default()
        };
        let run = || {
            let mut rng = Rng::new(params.seed);
            (0..10)
                .map(|_| sample(&logits, &[], &params, &mut rng))
                .collect::<Vec<_>>()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn repeat_penalty_disabled_by_default() {
        let params = SamplingParams::default();
        assert!(params.repeat_penalty.is_none());
    }

    #[test]
    fn repeat_penalty_suppresses_recent_token() {
        let mut logits = vec![1.0f32; 10];
        logits[3] = 5.0; // clear favorite before any penalty
        let history = vec![3u32; 64]; // token 3 dominates recent history
        let params = SamplingParams {
            temperature: 0.0, // greedy, so the penalty alone decides
            repeat_penalty: Some(1.3),
            repeat_penalty_window: 64,
            ..SamplingParams::default()
        };
        let mut rng = Rng::new(1);
        let picked = sample(&logits, &history, &params, &mut rng);
        assert_ne!(picked, 3, "repeat penalty should have demoted token 3");
    }
}
