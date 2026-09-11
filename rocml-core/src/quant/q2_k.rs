//! Q2_K: 256-element super-block, 16 packed 4-bit (scale, min) pairs and
//! 64 bytes of 2-bit codes. Ported from ggml's `block_q2_K` /
//! `dequantize_row_q2_K` (mirrored by candle-core's
//! `quantized/k_quants.rs::BlockQ2K`).

use super::common::{le_f16, QK_K};

const BLOCK_BYTES: usize = 84;

/// Documents the exact on-disk layout (see `q8_0.rs` for why dequant reads
/// bytes directly instead of transmuting this struct over the mmap).
#[repr(C)]
pub struct BlockQ2K {
    pub scales: [u8; QK_K / 16],
    pub qs: [u8; QK_K / 4],
    pub d: half::f16,
    pub dmin: half::f16,
}
const _: () = assert!(std::mem::size_of::<BlockQ2K>() == BLOCK_BYTES);

pub(super) fn dequantize(bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() / BLOCK_BYTES * QK_K);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let scales = &block[0..16];
        let qs = &block[16..80];
        let d = le_f16(block, 80);
        let dmin = le_f16(block, 82);

        let mut is = 0usize;
        for outer in 0..2 {
            let qs_chunk = &qs[outer * 32..outer * 32 + 32];
            let mut shift = 0u32;
            for _ in 0..4 {
                let sc = scales[is];
                is += 1;
                let dl = d * (sc & 0xF) as f32;
                let ml = dmin * (sc >> 4) as f32;
                for &q in &qs_chunk[..16] {
                    out.push(dl * ((q >> shift) & 3) as f32 - ml);
                }

                let sc = scales[is];
                is += 1;
                let dl = d * (sc & 0xF) as f32;
                let ml = dmin * (sc >> 4) as f32;
                for &q in &qs_chunk[16..] {
                    out.push(dl * ((q >> shift) & 3) as f32 - ml);
                }
                shift += 2;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_is_84_bytes() {
        assert_eq!(std::mem::size_of::<BlockQ2K>(), 84);
    }
}
