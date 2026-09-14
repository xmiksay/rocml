//! Attention-score/output perturbation metrics — issue #14's "the metric
//! that matters" for the tensor-level quality table, since raw per-vector
//! RMSE doesn't tell you how a K or V quantization error actually moves the
//! numbers attention produces. K only ever influences softmax *weights*
//! (via `Q . K`); V never does — it's only read back out through those
//! weights — so the two tensors get different derived metrics:
//!
//! - K: perturb the keys, keep the query exact, recompute softmax, RMSE the
//!   two weight vectors directly (both sum to 1, so a plain RMSE is
//!   meaningful without a "relative" denominator).
//! - V: keep the *true* softmax weights fixed (K is exact here — isolating
//!   V's own error), recompute the weighted-sum attention output with
//!   quantized V, and report the output vector's relative RMSE against the
//!   true output — the actual quantity a later layer would consume.
//!
//! No real captured query vectors exist in this codebase (the snapshot
//! layer captures K/V, not Q) — every caller here uses real captured K
//! vectors from a disjoint position range as a documented proxy query
//! (same residual stream, same per-head linear-projection-then-RoPE
//! structure as a real Q in this architecture), rather than a synthetic
//! random vector.

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn softmax(scores: &[f32]) -> Vec<f32> {
    let max = scores.iter().cloned().fold(f32::MIN, f32::max);
    let exps: Vec<f32> = scores.iter().map(|&s| (s - max).exp()).collect();
    let sum: f32 = exps.iter().sum::<f32>().max(1e-12);
    exps.iter().map(|&e| e / sum).collect()
}

fn scores_against(query: &[f32], keys: &[f32], n_keys: usize, head_dim: usize) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    (0..n_keys)
        .map(|i| dot(query, &keys[i * head_dim..(i + 1) * head_dim]) * scale)
        .collect()
}

/// Mean RMSE (over `queries`) between the true and perturbed-K softmax
/// weight vectors, for one (layer, head)'s key block.
pub fn k_attn_score_rmse(
    queries: &[f32],
    n_queries: usize,
    keys_true: &[f32],
    keys_approx: &[f32],
    n_keys: usize,
    head_dim: usize,
) -> f64 {
    let mut total = 0f64;
    for q in 0..n_queries {
        let query = &queries[q * head_dim..(q + 1) * head_dim];
        let w_true = softmax(&scores_against(query, keys_true, n_keys, head_dim));
        let w_approx = softmax(&scores_against(query, keys_approx, n_keys, head_dim));
        let mse: f64 = w_true
            .iter()
            .zip(&w_approx)
            .map(|(&a, &b)| {
                let e = (a - b) as f64;
                e * e
            })
            .sum::<f64>()
            / n_keys as f64;
        total += mse.sqrt();
    }
    total / n_queries as f64
}

/// Mean relative RMSE (over `queries`) of the attention output — true
/// softmax weights (computed from exact K) applied to true vs
/// perturbed-only V — for one (layer, head)'s key/value block.
pub fn v_attn_output_rel_rmse(
    queries: &[f32],
    n_queries: usize,
    keys_true: &[f32],
    n_keys: usize,
    head_dim: usize,
    v_true: &[f32],
    v_approx: &[f32],
) -> f64 {
    let mut total = 0f64;
    for q in 0..n_queries {
        let query = &queries[q * head_dim..(q + 1) * head_dim];
        let weights = softmax(&scores_against(query, keys_true, n_keys, head_dim));

        let mut out_true = vec![0f32; head_dim];
        let mut out_approx = vec![0f32; head_dim];
        for (i, &w) in weights.iter().enumerate() {
            let vt = &v_true[i * head_dim..(i + 1) * head_dim];
            let va = &v_approx[i * head_dim..(i + 1) * head_dim];
            for d in 0..head_dim {
                out_true[d] += w * vt[d];
                out_approx[d] += w * va[d];
            }
        }
        let (mut sum_sq_err, mut sum_sq_orig) = (0f64, 0f64);
        for (&t, &a) in out_true.iter().zip(&out_approx) {
            let e = (t - a) as f64;
            sum_sq_err += e * e;
            sum_sq_orig += (t as f64) * (t as f64);
        }
        total += if sum_sq_orig > 0.0 {
            (sum_sq_err / sum_sq_orig).sqrt()
        } else {
            0.0
        };
    }
    total / n_queries as f64
}
