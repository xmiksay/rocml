//! Q3_K: 256-element super-block, a 32-byte high-bit mask, 64 bytes of
//! 2-bit low codes and a packed 6-bit scale table. Ported from ggml's
//! `block_q3_K` / `dequantize_row_q3_K` (mirrored by candle-core's
//! `quantized/k_quants.rs::BlockQ3K`).

use super::common::{le_f16, le_u32, QK_K};

const BLOCK_BYTES: usize = 110;
const KMASK1: u32 = 0x0303_0303;
const KMASK2: u32 = 0x0f0f_0f0f;

/// Documents the exact on-disk layout (see `q8_0.rs` for why dequant reads
/// bytes directly instead of transmuting this struct over the mmap).
#[repr(C)]
pub struct BlockQ3K {
    pub hmask: [u8; QK_K / 8],
    pub qs: [u8; QK_K / 4],
    pub scales: [u8; 12],
    pub d: half::f16,
}
const _: () = assert!(std::mem::size_of::<BlockQ3K>() == BLOCK_BYTES);

/// Unpacks the 12-byte packed 6-bit scale table into 16 signed scales,
/// ported bit-for-bit from ggml's inline unpacking in `dequantize_row_q3_K`.
fn unpack_scales(scales: &[u8]) -> [i8; 16] {
    let a0 = le_u32(scales, 0);
    let a1 = le_u32(scales, 4);
    let tmp = le_u32(scales, 8);

    let a2 = ((a0 >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
    let a3 = ((a1 >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
    let a0 = (a0 & KMASK2) | ((tmp & KMASK1) << 4);
    let a1 = (a1 & KMASK2) | (((tmp >> 2) & KMASK1) << 4);

    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&a0.to_le_bytes());
    bytes[4..8].copy_from_slice(&a1.to_le_bytes());
    bytes[8..12].copy_from_slice(&a2.to_le_bytes());
    bytes[12..16].copy_from_slice(&a3.to_le_bytes());
    std::array::from_fn(|i| bytes[i] as i8)
}

pub(super) fn dequantize(bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() / BLOCK_BYTES * QK_K);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let hmask = &block[0..32];
        let qs = &block[32..96];
        let scales = unpack_scales(&block[96..108]);
        let d_all = le_f16(block, 108);

        let mut m: u8 = 1;
        let mut is = 0usize;
        for outer in 0..2 {
            let qs_chunk = &qs[outer * 32..outer * 32 + 32];
            let mut shift = 0u32;
            for _ in 0..4 {
                for scale_index in 0..2usize {
                    let dl = d_all * (scales[is] as f32 - 32.0);
                    is += 1;
                    for i in 0..16 {
                        let idx = i + 16 * scale_index;
                        let hbit: i8 = if hmask[idx] & m == 0 { 4 } else { 0 };
                        let q = ((qs_chunk[idx] >> shift) & 3) as i8;
                        out.push(dl * (q - hbit) as f32);
                    }
                }
                shift += 2;
                m <<= 1;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_is_110_bytes() {
        assert_eq!(std::mem::size_of::<BlockQ3K>(), 110);
    }
}
