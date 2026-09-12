//! Pure quantize/dequantize math for the KIVI-style mixed KV cache (issue
//! #2): K is per-channel (one scale per (kv head, head_dim channel) over a
//! whole evicted `WINDOW_LEN`-position block — outlier channels are
//! consistent across tokens, and K feeds softmax where error is
//! exponentially amplified); V is per-token (one scale per (kv head,
//! position) — outliers are distributed differently and partially cancel
//! once averaged through attention weights). This module is the CPU
//! reference both the host-side quantize-on-evict path and the fused HIP
//! kernels' dequant-on-load path are checked against — see
//! `rocml-kernels/tests/kv_quant.rs`.
//!
//! Layouts (all row-major, `[n_kv_heads, WINDOW_LEN, head_dim]` block
//! input):
//! - K codes: `i8`, same shape as the input block; scales: `[n_kv_heads,
//!   head_dim]` (one block's worth — the caller indexes a further
//!   evicted-block dimension on top of this).
//! - V codes (Q8): `i8`, same shape as the input block; scales:
//!   `[n_kv_heads, WINDOW_LEN]`.
//! - V codes (Q4): packed 2 codes/byte, `[n_kv_heads, WINDOW_LEN,
//!   head_dim/2]` (`head_dim` must be even); scales: `[n_kv_heads,
//!   WINDOW_LEN]`. Signed 4-bit range is `[-8, 7]`.

const Q8_MAX: f32 = 127.0;
const Q4_MAX: f32 = 7.0;

/// Per-channel K quantization: one scale per (kv head, channel), computed
/// from that channel's max-abs value across all `window_len` positions.
pub fn quantize_k_per_channel(
    block: &[f32],
    n_kv_heads: usize,
    window_len: usize,
    head_dim: usize,
) -> (Vec<i8>, Vec<f32>) {
    assert_eq!(block.len(), n_kv_heads * window_len * head_dim);
    let mut codes = vec![0i8; block.len()];
    let mut scales = vec![0f32; n_kv_heads * head_dim];
    for h in 0..n_kv_heads {
        for d in 0..head_dim {
            let mut max_abs = 0f32;
            for t in 0..window_len {
                max_abs = max_abs.max(block[(h * window_len + t) * head_dim + d].abs());
            }
            let scale = if max_abs > 0.0 { max_abs / Q8_MAX } else { 1.0 };
            scales[h * head_dim + d] = scale;
            for t in 0..window_len {
                let idx = (h * window_len + t) * head_dim + d;
                codes[idx] = (block[idx] / scale).round().clamp(-Q8_MAX, Q8_MAX) as i8;
            }
        }
    }
    (codes, scales)
}

pub fn dequantize_k_per_channel(
    codes: &[i8],
    scales: &[f32],
    n_kv_heads: usize,
    window_len: usize,
    head_dim: usize,
) -> Vec<f32> {
    assert_eq!(codes.len(), n_kv_heads * window_len * head_dim);
    assert_eq!(scales.len(), n_kv_heads * head_dim);
    let mut out = vec![0f32; codes.len()];
    for h in 0..n_kv_heads {
        for t in 0..window_len {
            for d in 0..head_dim {
                let idx = (h * window_len + t) * head_dim + d;
                out[idx] = codes[idx] as f32 * scales[h * head_dim + d];
            }
        }
    }
    out
}

/// Per-token V quantization at 8 bits: one scale per (kv head, position),
/// computed from that position's max-abs value across `head_dim`.
pub fn quantize_v_per_token_q8(
    block: &[f32],
    n_kv_heads: usize,
    window_len: usize,
    head_dim: usize,
) -> (Vec<i8>, Vec<f32>) {
    assert_eq!(block.len(), n_kv_heads * window_len * head_dim);
    let mut codes = vec![0i8; block.len()];
    let mut scales = vec![0f32; n_kv_heads * window_len];
    for h in 0..n_kv_heads {
        for t in 0..window_len {
            let row = &block[(h * window_len + t) * head_dim..(h * window_len + t + 1) * head_dim];
            let max_abs = row.iter().fold(0f32, |m, &v| m.max(v.abs()));
            let scale = if max_abs > 0.0 { max_abs / Q8_MAX } else { 1.0 };
            scales[h * window_len + t] = scale;
            for d in 0..head_dim {
                let idx = (h * window_len + t) * head_dim + d;
                codes[idx] = (block[idx] / scale).round().clamp(-Q8_MAX, Q8_MAX) as i8;
            }
        }
    }
    (codes, scales)
}

pub fn dequantize_v_per_token_q8(
    codes: &[i8],
    scales: &[f32],
    n_kv_heads: usize,
    window_len: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; codes.len()];
    for h in 0..n_kv_heads {
        for t in 0..window_len {
            let scale = scales[h * window_len + t];
            for d in 0..head_dim {
                let idx = (h * window_len + t) * head_dim + d;
                out[idx] = codes[idx] as f32 * scale;
            }
        }
    }
    out
}

/// Per-token V quantization at 4 bits, packed 2 codes/byte (`head_dim` must
/// be even): low nibble is the even channel, high nibble the odd one —
/// matches `quantize_evict_v_f16_to_q4`'s HIP kernel packing exactly (see
/// that kernel's own doc comment).
pub fn quantize_v_per_token_q4(
    block: &[f32],
    n_kv_heads: usize,
    window_len: usize,
    head_dim: usize,
) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(block.len(), n_kv_heads * window_len * head_dim);
    assert!(
        head_dim.is_multiple_of(2),
        "head_dim must be even for q4 packing"
    );
    let mut packed = vec![0u8; n_kv_heads * window_len * (head_dim / 2)];
    let mut scales = vec![0f32; n_kv_heads * window_len];
    for h in 0..n_kv_heads {
        for t in 0..window_len {
            let row = &block[(h * window_len + t) * head_dim..(h * window_len + t + 1) * head_dim];
            let max_abs = row.iter().fold(0f32, |m, &v| m.max(v.abs()));
            let scale = if max_abs > 0.0 { max_abs / Q4_MAX } else { 1.0 };
            scales[h * window_len + t] = scale;
            for pair in 0..head_dim / 2 {
                let lo = (row[2 * pair] / scale).round().clamp(-Q4_MAX - 1.0, Q4_MAX) as i8;
                let hi = (row[2 * pair + 1] / scale)
                    .round()
                    .clamp(-Q4_MAX - 1.0, Q4_MAX) as i8;
                let packed_idx = (h * window_len + t) * (head_dim / 2) + pair;
                packed[packed_idx] = pack_nibbles(lo, hi);
            }
        }
    }
    (packed, scales)
}

pub fn dequantize_v_per_token_q4(
    packed: &[u8],
    scales: &[f32],
    n_kv_heads: usize,
    window_len: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; n_kv_heads * window_len * head_dim];
    for h in 0..n_kv_heads {
        for t in 0..window_len {
            let scale = scales[h * window_len + t];
            for pair in 0..head_dim / 2 {
                let packed_idx = (h * window_len + t) * (head_dim / 2) + pair;
                let (lo, hi) = unpack_nibbles(packed[packed_idx]);
                let base = (h * window_len + t) * head_dim + 2 * pair;
                out[base] = lo as f32 * scale;
                out[base + 1] = hi as f32 * scale;
            }
        }
    }
    out
}

/// Packs two signed 4-bit values (`[-8, 7]`) into one byte: `lo` in bits
/// `[0,4)`, `hi` in bits `[4,8)`, both as their 4-bit two's-complement
/// representation.
fn pack_nibbles(lo: i8, hi: i8) -> u8 {
    ((lo as u8) & 0x0F) | (((hi as u8) & 0x0F) << 4)
}

/// Inverse of [`pack_nibbles`], sign-extending each nibble back to `i8`.
fn unpack_nibbles(byte: u8) -> (i8, i8) {
    let lo = sign_extend_nibble(byte & 0x0F);
    let hi = sign_extend_nibble((byte >> 4) & 0x0F);
    (lo, hi)
}

fn sign_extend_nibble(n: u8) -> i8 {
    if n >= 8 {
        (n as i8) - 16
    } else {
        n as i8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_block(
        n_kv_heads: usize,
        window_len: usize,
        head_dim: usize,
        seed: u32,
    ) -> Vec<f32> {
        (0..n_kv_heads * window_len * head_dim)
            .map(|i| {
                let x = (i as u32).wrapping_mul(2654435761).wrapping_add(seed);
                ((x >> 8) as f32 / u32::MAX as f32 - 0.5) * 4.0
            })
            .collect()
    }

    fn max_rel_err(a: &[f32], b: &[f32], scale_floor: f32) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs() / x.abs().max(scale_floor))
            .fold(0f32, f32::max)
    }

    fn mean_abs_err(a: &[f32], b: &[f32]) -> f32 {
        let sum: f32 = a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum();
        sum / a.len() as f32
    }

    #[test]
    fn nibble_pack_roundtrips_full_signed_range() {
        for lo in -8i8..=7 {
            for hi in -8i8..=7 {
                let (got_lo, got_hi) = unpack_nibbles(pack_nibbles(lo, hi));
                assert_eq!((got_lo, got_hi), (lo, hi));
            }
        }
    }

    #[test]
    fn k_per_channel_roundtrip_error_bounded_by_quant_step() {
        let (h, w, d) = (4usize, 128usize, 64usize);
        let block = synthetic_block(h, w, d, 1);
        let (codes, scales) = quantize_k_per_channel(&block, h, w, d);
        let deq = dequantize_k_per_channel(&codes, &scales, h, w, d);
        // Per-channel int8: worst-case rounding error is half a quant step
        // (scale/2), and scale = max_abs/127 for that channel — bounded by
        // the overall block's max value / 127 / 2 with slack for
        // clamp/round; assert a measured, comfortably-covering bound.
        let max_abs = block.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let bound = max_abs / 127.0; // one full quant step, generous
        for (got, want) in deq.iter().zip(&block) {
            assert!(
                (got - want).abs() <= bound,
                "got {got} want {want} bound {bound}"
            );
        }
        eprintln!(
            "K q8 per-channel: mean abs err {:.6}",
            mean_abs_err(&deq, &block)
        );
    }

    #[test]
    fn v_per_token_q8_roundtrip_error_bounded_by_quant_step() {
        let (h, w, d) = (4usize, 128usize, 64usize);
        let block = synthetic_block(h, w, d, 2);
        let (codes, scales) = quantize_v_per_token_q8(&block, h, w, d);
        let deq = dequantize_v_per_token_q8(&codes, &scales, h, w, d);
        let rel = max_rel_err(&block, &deq, 1e-3);
        // Per-token q8: same reasoning as K, just scoped per row instead of
        // per channel — the measured max relative error should sit well
        // under 1% for values not vanishingly close to zero.
        assert!(rel < 0.02, "measured max relative error {rel}");
        eprintln!("V q8 per-token: max rel err {rel:.6}");
    }

    #[test]
    fn v_per_token_q4_roundtrip_error_is_measured_and_bounded() {
        let (h, w, d) = (4usize, 128usize, 64usize);
        let block = synthetic_block(h, w, d, 3);
        let (packed, scales) = quantize_v_per_token_q4(&block, h, w, d);
        let deq = dequantize_v_per_token_q4(&packed, &scales, h, w, d);
        let mae = mean_abs_err(&block, &deq);
        let max_abs = block.iter().fold(0f32, |m, &v| m.max(v.abs()));
        // 4 bits has a much coarser step (1/7 of the per-row max vs 1/127
        // for q8) — measure and assert a bound reflecting that, not q8's.
        // Empirically this synthetic (roughly uniform [-2,2]) data lands
        // well under 10% of the row's own dynamic range.
        assert!(
            mae < 0.10 * max_abs,
            "measured mean abs error {mae} exceeds 10% of block max {max_abs}"
        );
        eprintln!("V q4 per-token: mean abs err {mae:.6} (block max {max_abs:.6})");
    }

    #[test]
    fn zero_block_quantizes_and_dequantizes_to_zero() {
        let (h, w, d) = (2usize, 8usize, 16usize);
        let block = vec![0f32; h * w * d];
        let (k_codes, k_scales) = quantize_k_per_channel(&block, h, w, d);
        assert!(k_codes.iter().all(|&c| c == 0));
        let k_deq = dequantize_k_per_channel(&k_codes, &k_scales, h, w, d);
        assert!(k_deq.iter().all(|&v| v == 0.0));

        let (v_codes, v_scales) = quantize_v_per_token_q8(&block, h, w, d);
        assert!(v_codes.iter().all(|&c| c == 0));
        let v_deq = dequantize_v_per_token_q8(&v_codes, &v_scales, h, w, d);
        assert!(v_deq.iter().all(|&v| v == 0.0));
    }
}
