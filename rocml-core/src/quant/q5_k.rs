//! Q5_K: like Q4_K but with an extra high-bit mask giving 5-bit codes.
//! Ported from ggml's `block_q5_K` / `dequantize_row_q5_K` (mirrored by
//! candle-core's `quantized/k_quants.rs::BlockQ5K`).

use super::common::{get_scale_min_k4, le_f16, K_SCALE_SIZE, QK_K};

const BLOCK_BYTES: usize = 176;

/// Documents the exact on-disk layout (see `q8_0.rs` for why dequant reads
/// bytes directly instead of transmuting this struct over the mmap).
#[repr(C)]
pub struct BlockQ5K {
    pub d: half::f16,
    pub dmin: half::f16,
    pub scales: [u8; K_SCALE_SIZE],
    pub qh: [u8; QK_K / 8],
    pub qs: [u8; QK_K / 2],
}
const _: () = assert!(std::mem::size_of::<BlockQ5K>() == BLOCK_BYTES);

pub(super) fn dequantize(bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() / BLOCK_BYTES * QK_K);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let d = le_f16(block, 0);
        let dmin = le_f16(block, 2);
        let scales = &block[4..4 + K_SCALE_SIZE];
        let qh = &block[4 + K_SCALE_SIZE..4 + K_SCALE_SIZE + QK_K / 8];
        let qs = &block[4 + K_SCALE_SIZE + QK_K / 8..BLOCK_BYTES];

        let mut is = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for j in (0..QK_K).step_by(64) {
            let ql = &qs[j / 2..j / 2 + 32];
            let (sc, m) = get_scale_min_k4(is, scales);
            let d1 = d * sc as f32;
            let m1 = dmin * m as f32;
            let (sc, m) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc as f32;
            let m2 = dmin * m as f32;

            for (&b, &h) in ql.iter().zip(qh) {
                let hi = if h & u1 != 0 { 16.0 } else { 0.0 };
                out.push(d1 * ((b & 0xF) as f32 + hi) - m1);
            }
            for (&b, &h) in ql.iter().zip(qh) {
                let hi = if h & u2 != 0 { 16.0 } else { 0.0 };
                out.push(d2 * ((b >> 4) as f32 + hi) - m2);
            }
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_is_176_bytes() {
        assert_eq!(std::mem::size_of::<BlockQ5K>(), 176);
    }
}
