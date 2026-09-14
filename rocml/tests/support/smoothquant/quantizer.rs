//! CPU reference for the per-32-block symmetric int8 activation quantizer
//! (`rocml-kernels/kernels/quantize_act_q8.hip`'s `quantize_act_q8_blk`) —
//! a plain-Rust mirror so issue #17's standalone measurement can quantize
//! calibration/captured activations without touching the GPU.

pub const ACT_BLOCK: usize = 32;

/// Per-32-block symmetric int8 quantization of one row: `d = amax/127` (or
/// `1.0` for an all-zero block), `code = round(v/d)` clamped to
/// `[-127, 127]` — bit-for-bit the same formula the HIP kernel uses.
/// Returns `(codes, scale_per_block)`; `row.len()` must be a multiple of
/// [`ACT_BLOCK`].
pub fn quantize_row_i8(row: &[f32]) -> (Vec<i8>, Vec<f32>) {
    assert!(row.len().is_multiple_of(ACT_BLOCK));
    let n_blocks = row.len() / ACT_BLOCK;
    let mut codes = vec![0i8; row.len()];
    let mut scale = vec![0f32; n_blocks];
    for (b, blk) in row.chunks_exact(ACT_BLOCK).enumerate() {
        let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = if amax > 0.0 { amax / 127.0 } else { 1.0 };
        scale[b] = d;
        for (i, &v) in blk.iter().enumerate() {
            let q = (v / d).round().clamp(-127.0, 127.0);
            codes[b * ACT_BLOCK + i] = q as i8;
        }
    }
    (codes, scale)
}

/// Dequantizes `(codes, scale)` back to f32 — `codes[i] as f32 *
/// scale[i/ACT_BLOCK]`, the same reconstruction the MMQ GEMM kernel
/// performs implicitly via its integer dot product times the per-block
/// scale.
pub fn dequant_row_i8(codes: &[i8], scale: &[f32]) -> Vec<f32> {
    codes
        .iter()
        .enumerate()
        .map(|(i, &c)| c as f32 * scale[i / ACT_BLOCK])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_is_within_one_lsb() {
        let row: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.37).collect();
        let (codes, scale) = quantize_row_i8(&row);
        let back = dequant_row_i8(&codes, &scale);
        for (a, b) in row.iter().zip(&back) {
            let tol = scale[0].max(scale[1]);
            assert!((a - b).abs() <= tol, "{a} vs {b} (tol {tol})");
        }
    }

    #[test]
    fn all_zero_block_does_not_divide_by_zero() {
        let row = vec![0f32; 32];
        let (codes, scale) = quantize_row_i8(&row);
        assert!(codes.iter().all(|&c| c == 0));
        assert_eq!(scale[0], 1.0);
    }
}
