//! Plain f32 math helpers, mirroring the HIP kernels' exact formulas
//! (`rocml-kernels/kernels/*.hip`) so this CPU reference validates
//! *understanding* of the architecture independently of the GPU kernels.

/// `y = mat * x`, `mat` row-major `[m, n]`.
pub fn matvec(mat: &[f32], x: &[f32], m: usize, n: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; m];
    for row in 0..m {
        let mat_row = &mat[row * n..(row + 1) * n];
        y[row] = mat_row.iter().zip(x).map(|(&a, &b)| a * b).sum();
    }
    y
}

/// In-place per-row RMS norm over `rows` rows of width `n`: `x /
/// sqrt(mean(x^2) + eps) * weight` (`weight` is `[n]`, shared by every row —
/// same contract as `rmsnorm_f32`).
pub fn rmsnorm_rows(x: &mut [f32], weight: &[f32], rows: usize, n: usize, eps: f32) {
    for row in 0..rows {
        let slice = &mut x[row * n..(row + 1) * n];
        let mean_sq: f32 = slice.iter().map(|&v| v * v).sum::<f32>() / n as f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();
        for (v, &w) in slice.iter_mut().zip(weight) {
            *v *= inv_rms * w;
        }
    }
}

/// L2 norm over `rows` rows of width `n`: `x / sqrt(sum(x^2) + eps)`,
/// optionally scaled by an extra constant afterward (used to fold in the
/// GDN recurrence's `1/sqrt(head_k_dim)` query scale).
pub fn l2norm_rows(x: &mut [f32], rows: usize, n: usize, eps: f32, extra_scale: f32) {
    for row in 0..rows {
        let slice = &mut x[row * n..(row + 1) * n];
        let sum_sq: f32 = slice.iter().map(|&v| v * v).sum();
        let inv = extra_scale / (sum_sq + eps).sqrt();
        for v in slice.iter_mut() {
            *v *= inv;
        }
    }
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// Softplus, matching the kernels' naive (non-overflow-guarded) formula.
pub fn softplus(x: f32) -> f32 {
    (1.0 + x.exp()).ln()
}

pub fn add_inplace(acc: &mut [f32], x: &[f32]) {
    for (a, &b) in acc.iter_mut().zip(x) {
        *a += b;
    }
}

/// Numerically-stable softmax over the whole slice, in place.
pub fn softmax_inplace(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

/// Partial NEOX rope over `x` viewed as `[heads, head_dim]` for one token at
/// absolute position `pos`: only the first `rot_dim` components of each head
/// rotate (paired `i` <-> `i + rot_dim/2`); the rest pass through unchanged.
pub fn rope_partial(
    x: &mut [f32],
    heads: usize,
    head_dim: usize,
    rot_dim: usize,
    pos: u32,
    theta_base: f32,
) {
    let half_rot = rot_dim / 2;
    for h in 0..heads {
        let base = h * head_dim;
        for i in 0..half_rot {
            let inv_freq = theta_base.powf(-2.0 * i as f32 / rot_dim as f32);
            let angle = pos as f32 * inv_freq;
            let (sin_a, cos_a) = angle.sin_cos();
            let x0 = x[base + i];
            let x1 = x[base + i + half_rot];
            x[base + i] = x0 * cos_a - x1 * sin_a;
            x[base + i + half_rot] = x0 * sin_a + x1 * cos_a;
        }
    }
}
