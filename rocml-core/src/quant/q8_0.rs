//! Q8_0: per-32-element block, one `f16` scale plus 32 signed 8-bit codes.
//! Ported from ggml's `block_q8_0` / `dequantize_row_q8_0`
//! (mirrored by candle-core's `quantized/k_quants.rs::BlockQ8_0`).

use super::common::le_f16;

pub(super) const QK8_0: usize = 32;

/// Documents the exact on-disk layout; dequantization reads bytes directly
/// (see module docs in `mod.rs` on why we don't transmute the mmap into
/// `&[BlockQ8_0]`), so this struct exists for the size assertion below.
#[repr(C)]
pub struct BlockQ8_0 {
    pub d: half::f16,
    pub qs: [i8; QK8_0],
}
const _: () = assert!(std::mem::size_of::<BlockQ8_0>() == 34);

/// Dequantizes a byte slice holding a whole number of Q8_0 blocks.
/// Caller (i.e. `quant::dequantize`) guarantees `bytes.len()` is a multiple
/// of 34.
pub(super) fn dequantize(bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() / 34 * QK8_0);
    for block in bytes.chunks_exact(34) {
        let d = le_f16(block, 0);
        for &q in &block[2..34] {
            out.push((q as i8) as f32 * d);
        }
    }
    out
}

/// Quantizes a row of `QK8_0`-aligned f32 values into ggml's Q8_0 layout.
/// Only used by tests to build synthetic data for the roundtrip check;
/// production code only ever dequantizes weights it received from a file.
#[cfg(test)]
pub(super) fn quantize(xs: &[f32]) -> Vec<u8> {
    assert!(xs.len().is_multiple_of(QK8_0));
    let mut out = Vec::with_capacity(xs.len() / QK8_0 * 34);
    for block in xs.chunks_exact(QK8_0) {
        let amax = block.iter().fold(0f32, |m, &x| m.max(x.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
        for &x in block {
            out.push((x * id).round() as i8 as u8);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_is_34_bytes() {
        assert_eq!(std::mem::size_of::<BlockQ8_0>(), 34);
    }

    #[test]
    fn quantize_dequantize_roundtrip_is_close() {
        // Synthetic data spanning a wide dynamic range, on a couple of
        // block boundaries (2 blocks of 32).
        let xs: Vec<f32> = (0..64)
            .map(|i| ((i as f32) - 32.0) * 0.37 + if i % 7 == 0 { 5.0 } else { 0.0 })
            .collect();
        let packed = quantize(&xs);
        assert_eq!(packed.len(), 2 * 34);
        let back = dequantize(&packed);
        assert_eq!(back.len(), xs.len());
        for block in xs.chunks_exact(QK8_0).zip(back.chunks_exact(QK8_0)) {
            let (orig_block, got_block) = block;
            let amax = orig_block.iter().fold(0f32, |m, &x| m.max(x.abs()));
            // Max rounding error is half the per-block quantization step;
            // the small additive term absorbs f16 rounding of `d` itself.
            let tol = amax / 127.0 / 2.0 + 1e-3;
            for (orig, got) in orig_block.iter().zip(got_block.iter()) {
                assert!(got.is_finite());
                assert!((orig - got).abs() <= tol, "orig={orig} got={got} tol={tol}");
            }
        }
    }
}
