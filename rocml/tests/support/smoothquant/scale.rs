//! SmoothQuant per-input-channel-block scale: `s = amax_x^alpha /
//! amax_w^(1-alpha)`, quantized to per-32-channel-block granularity so it
//! folds against both the activation quantizer's block width and a Q4_K
//! sub-block's width (issue #17's design constraint — see the K-quant fold
//! logic in `q4k.rs`).

const EPS: f32 = 1e-6;

/// One block's SmoothQuant factor from that block's activation and weight
/// absmax (both already reduced to a single scalar per 32-channel block —
/// see [`per_block_max`]).
pub fn block_scale(x_amax: f32, w_amax: f32, alpha: f32) -> f32 {
    x_amax.max(EPS).powf(alpha) / w_amax.max(EPS).powf(1.0 - alpha)
}

/// Reduces a per-channel amax vector to one value per 32-channel block via
/// max — the conservative choice (no channel in the block is left
/// under-protected once `s` is applied uniformly across the block).
pub fn per_block_max(channel_amax: &[f32]) -> Vec<f32> {
    channel_amax
        .chunks(32)
        .map(|c| c.iter().cloned().fold(0f32, f32::max))
        .collect()
}

/// Per-input-channel (column) absmax of a row-major `[rows, cols]` weight
/// matrix — the "amax over all output rows" half of the SmoothQuant scale
/// formula, computed from a real CPU-dequantized weight.
pub fn weight_channel_amax(weight: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut amax = vec![0f32; cols];
    for r in 0..rows {
        let row = &weight[r * cols..(r + 1) * cols];
        for (a, &v) in amax.iter_mut().zip(row) {
            *a = a.max(v.abs());
        }
    }
    amax
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alpha_zero_moves_no_activation_magnitude() {
        // alpha=0: s = amax_w^-1 * amax_x^0 = 1/amax_w — depends only on
        // the weight side, matching the formula's own definition at the
        // boundary (not a special case in the implementation).
        let s = block_scale(10.0, 2.0, 0.0);
        assert!((s - 0.5).abs() < 1e-6);
    }

    #[test]
    fn alpha_half_is_geometric_mean_ratio() {
        let s = block_scale(16.0, 4.0, 0.5);
        assert!((s - 2.0).abs() < 1e-4, "got {s}");
    }
}
