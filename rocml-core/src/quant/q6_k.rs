//! Q6_K: 256-element super-block, 6-bit codes (4 low bits + 2 high bits)
//! and 16 signed 8-bit per-sub-block scales. Ported from ggml's
//! `block_q6_K` / `dequantize_row_q6_K` (mirrored by candle-core's
//! `quantized/k_quants.rs::BlockQ6K`).

use super::common::{le_f16, QK_K};

const BLOCK_BYTES: usize = 210;

/// Documents the exact on-disk layout (see `q8_0.rs` for why dequant reads
/// bytes directly instead of transmuting this struct over the mmap).
#[repr(C)]
pub struct BlockQ6K {
    pub ql: [u8; QK_K / 2],
    pub qh: [u8; QK_K / 4],
    pub scales: [i8; QK_K / 16],
    pub d: half::f16,
}
const _: () = assert!(std::mem::size_of::<BlockQ6K>() == BLOCK_BYTES);

pub(super) fn dequantize(bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() / BLOCK_BYTES * QK_K);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let ql_full = &block[0..128];
        let qh_full = &block[128..192];
        let sc_full = &block[192..208];
        let d = le_f16(block, 208);

        let mut y = [0f32; QK_K];
        for half_idx in 0..2usize {
            let ql = &ql_full[64 * half_idx..64 * half_idx + 64];
            let qh = &qh_full[32 * half_idx..32 * half_idx + 32];
            let sc = &sc_full[8 * half_idx..8 * half_idx + 8];
            let y = &mut y[128 * half_idx..128 * half_idx + 128];

            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i8 - 32;
                let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;
                y[l] = d * sc[is] as f32 * q1 as f32;
                y[l + 32] = d * sc[is + 2] as f32 * q2 as f32;
                y[l + 64] = d * sc[is + 4] as f32 * q3 as f32;
                y[l + 96] = d * sc[is + 6] as f32 * q4 as f32;
            }
        }
        out.extend_from_slice(&y);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_is_210_bytes() {
        assert_eq!(std::mem::size_of::<BlockQ6K>(), 210);
    }
}
