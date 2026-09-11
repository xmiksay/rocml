//! Shared constants/helpers for the K-quant block formats, ported from
//! ggml's `k_quants.c` (mirrored by candle-core's `quantized/utils.rs`,
//! which this crate uses as its correctness reference).

/// Super-block size shared by all K-quants: 256 elements per block.
pub(super) const QK_K: usize = 256;
/// Byte size of the packed 6-bit scale/min table in Q4_K and Q5_K blocks.
pub(super) const K_SCALE_SIZE: usize = 12;

pub(super) fn le_u16(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([bytes[at], bytes[at + 1]])
}

pub(super) fn le_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

/// Half-precision float stored little-endian at `at`, as ggml's `d`/`dmin`
/// scale fields are.
pub(super) fn le_f16(bytes: &[u8], at: usize) -> f32 {
    half::f16::from_bits(le_u16(bytes, at)).to_f32()
}

/// Unpacks the 6-bit (scale, min) pair for sub-block `j` out of the 12-byte
/// packed table used by Q4_K/Q5_K. Ported verbatim from ggml's
/// `get_scale_min_k4`.
pub(super) fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        let d = q[j] & 63;
        let m = q[j + 4] & 63;
        (d, m)
    } else {
        let d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}
