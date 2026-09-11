//! Q4_K: 256-element super-block, `f16` scale + min, a packed 6-bit
//! (scale, min) table per 32-element sub-block, and 128 bytes of 4-bit
//! codes. Ported from ggml's `block_q4_K` / `dequantize_row_q4_K`
//! (mirrored by candle-core's `quantized/k_quants.rs::BlockQ4K`).

use super::common::{get_scale_min_k4, le_f16, K_SCALE_SIZE, QK_K};

const BLOCK_BYTES: usize = 144;

/// Documents the exact on-disk layout (see `q8_0.rs` for why dequant reads
/// bytes directly instead of transmuting this struct over the mmap).
#[repr(C)]
pub struct BlockQ4K {
    pub d: half::f16,
    pub dmin: half::f16,
    pub scales: [u8; K_SCALE_SIZE],
    pub qs: [u8; QK_K / 2],
}
const _: () = assert!(std::mem::size_of::<BlockQ4K>() == BLOCK_BYTES);

pub(super) fn dequantize(bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() / BLOCK_BYTES * QK_K);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let d = le_f16(block, 0);
        let dmin = le_f16(block, 2);
        let scales = &block[4..4 + K_SCALE_SIZE];
        let qs = &block[4 + K_SCALE_SIZE..BLOCK_BYTES];

        let mut is = 0usize;
        for j in (0..QK_K).step_by(64) {
            let q = &qs[j / 2..j / 2 + 32];
            let (sc, m) = get_scale_min_k4(is, scales);
            let d1 = d * sc as f32;
            let m1 = dmin * m as f32;
            let (sc, m) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc as f32;
            let m2 = dmin * m as f32;

            for &b in q {
                out.push(d1 * (b & 0xF) as f32 - m1);
            }
            for &b in q {
                out.push(d2 * (b >> 4) as f32 - m2);
            }
            is += 2;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_is_144_bytes() {
        assert_eq!(std::mem::size_of::<BlockQ4K>(), 144);
    }
}
