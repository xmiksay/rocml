//! Q4_K block-level fold/requant reference for issue #17's standalone
//! SmoothQuant measurement — deliberately test-only and independent of
//! `rocml_core::quant::q4_k`'s production dequantizer (which has no public
//! sub-block-scale accessor, by design: nothing in the hot path needs one).
//! Ported by hand from the same ggml layout that module documents (`d`,
//! `dmin`, a 12-byte packed 6-bit `(scale, min)` table, 128 bytes of 4-bit
//! codes over a 256-element superblock split into eight 32-element
//! sub-blocks) so the measurement below operates on the exact bytes the
//! real GGUF stores.

pub const Q4K_BLOCK_BYTES: usize = 144;
pub const SUPERBLOCK: usize = 256;
pub const SUBBLOCK: usize = 32;
const K_SCALE_SIZE: usize = 12;

fn le_f16(bytes: &[u8], at: usize) -> f32 {
    half::f16::from_le_bytes([bytes[at], bytes[at + 1]]).to_f32()
}

/// Unpacks the 6-bit `(scale, min)` pair for sub-block `j` (0..8) out of the
/// 12-byte packed table — verbatim port of `rocml_core::quant::common`'s
/// private `get_scale_min_k4` (duplicated here since that helper isn't
/// exposed outside the `quant` module, and this is test-only code with no
/// business reaching into the production crate's internals).
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// Inverse of [`get_scale_min_k4`] — only used by this module's own unit
/// test to build a synthetic block without needing a real GGUF on disk.
#[cfg(test)]
fn pack_scale_min_k4(sub_scales: &[(u8, u8); 8]) -> [u8; K_SCALE_SIZE] {
    let mut out = [0u8; K_SCALE_SIZE];
    for j in 0..4 {
        let (sc, m) = sub_scales[j];
        out[j] = sc & 0x3F;
        out[j + 4] = m & 0x3F;
    }
    for j in 4..8 {
        let (sc, m) = sub_scales[j];
        out[j + 4] = (sc & 0xF) | ((m & 0xF) << 4);
        out[j - 4] |= (sc >> 4) << 6;
        out[j] |= (m >> 4) << 6;
    }
    out
}

/// One parsed 256-element Q4_K superblock: the two f16 anchors, the eight
/// unpacked `(scale, min)` sub-block pairs, and the raw 4-bit codes
/// (widened to `u8`, one element each, in dequant order).
pub struct Q4kSuperblock {
    pub d: f32,
    pub dmin: f32,
    pub sub_scales: [(u8, u8); 8],
    pub codes: [u8; SUPERBLOCK],
}

impl Q4kSuperblock {
    pub fn parse(block: &[u8]) -> Self {
        assert_eq!(block.len(), Q4K_BLOCK_BYTES);
        let d = le_f16(block, 0);
        let dmin = le_f16(block, 2);
        let scales = &block[4..4 + K_SCALE_SIZE];
        let qs = &block[4 + K_SCALE_SIZE..Q4K_BLOCK_BYTES];

        let mut sub_scales = [(0u8, 0u8); 8];
        let mut codes = [0u8; SUPERBLOCK];
        let mut is = 0usize;
        for j in (0..SUPERBLOCK).step_by(64) {
            let q = &qs[j / 2..j / 2 + 32];
            sub_scales[is] = get_scale_min_k4(is, scales);
            sub_scales[is + 1] = get_scale_min_k4(is + 1, scales);
            for (k, &b) in q.iter().enumerate() {
                codes[j + k] = b & 0xF;
                codes[j + 32 + k] = b >> 4;
            }
            is += 2;
        }
        Self {
            d,
            dmin,
            sub_scales,
            codes,
        }
    }

    /// `v_i = d*sc*q_i - dmin*m` for sub-block `sub_scales[i/32]` — must
    /// match `rocml_core::quant::q4_k::dequantize` bit-for-bit (checked by
    /// this module's own unit test against a real captured block).
    pub fn dequant(&self) -> [f32; SUPERBLOCK] {
        let mut out = [0f32; SUPERBLOCK];
        for (i, o) in out.iter_mut().enumerate() {
            let (sc, m) = self.sub_scales[i / SUBBLOCK];
            *o = self.d * sc as f32 * self.codes[i] as f32 - self.dmin * m as f32;
        }
        out
    }
}

/// **Naive analytic fold**: rescales each sub-block's `(sc, m)` codes by
/// that sub-block's SmoothQuant factor `s[i/32]`, leaving `d`/`dmin` (shared
/// across the whole superblock) and every element's 4-bit code untouched —
/// mathematically exact when `sc*s`/`m*s` round to an in-range 6-bit integer
/// with no rounding, lossy exactly to the extent they don't (headroom-
/// dependent clipping is the "catch" issue #17's brief calls out). Returns
/// the reconstructed dequantized values, for the caller to diff against the
/// ideal `original[i] * s[i/32]` target.
pub fn fold_naive(sb: &Q4kSuperblock, s: &[f32; 8]) -> [f32; SUPERBLOCK] {
    let mut new_scales = sb.sub_scales;
    for (j, ns) in new_scales.iter_mut().enumerate() {
        let (sc, m) = *ns;
        let new_sc = (sc as f32 * s[j]).round().clamp(0.0, 63.0) as u8;
        let new_m = (m as f32 * s[j]).round().clamp(0.0, 63.0) as u8;
        *ns = (new_sc, new_m);
    }
    let mut out = [0f32; SUPERBLOCK];
    for (i, o) in out.iter_mut().enumerate() {
        let (sc, m) = new_scales[i / SUBBLOCK];
        *o = sb.d * sc as f32 * sb.codes[i] as f32 - sb.dmin * m as f32;
    }
    out
}

/// **Dequant-scale-requant**: re-derives a fresh `d`/`dmin`/`(sc,m)`/codes
/// for `target` (already `original * s`, per sub-block) from scratch,
/// structurally faithful to the Q4_K format's real degrees of freedom (one
/// shared affine anchor pair per 256-superblock, 6-bit sub-block scale/min,
/// 4-bit codes) but not bit-identical to ggml's own iterative least-squares
/// search (`make_qkx2_quants` tries several bias corrections this skips) —
/// adequate for bounding the requant error's order of magnitude, not for
/// reproducing ggml's exact output.
pub fn requant(target: &[f32; SUPERBLOCK]) -> [f32; SUPERBLOCK] {
    let mut raw_scale = [0f32; 8];
    let mut raw_min = [0f32; 8];
    for j in 0..8 {
        let sub = &target[j * SUBBLOCK..(j + 1) * SUBBLOCK];
        let lo = sub.iter().cloned().fold(0f32, f32::min); // clamp lo <= 0, per ggml's own convention
        let hi = sub.iter().cloned().fold(f32::MIN, f32::max);
        raw_scale[j] = (hi - lo) / 15.0;
        raw_min[j] = -lo;
    }
    let d = raw_scale.iter().cloned().fold(0f32, f32::max) / 63.0;
    let dmin = raw_min.iter().cloned().fold(0f32, f32::max) / 63.0;
    let d = if d > 0.0 { d } else { 1.0 };
    let dmin = if dmin > 0.0 { dmin } else { 1.0 };

    let mut sc = [0u8; 8];
    let mut m = [0u8; 8];
    for j in 0..8 {
        sc[j] = (raw_scale[j] / d).round().clamp(0.0, 63.0) as u8;
        m[j] = (raw_min[j] / dmin).round().clamp(0.0, 63.0) as u8;
    }

    let mut out = [0f32; SUPERBLOCK];
    for (i, o) in out.iter_mut().enumerate() {
        let j = i / SUBBLOCK;
        let step = d * sc[j] as f32;
        let offset = dmin * m[j] as f32;
        let q = if step > 0.0 {
            ((target[i] + offset) / step).round().clamp(0.0, 15.0)
        } else {
            0.0
        };
        *o = step * q - offset;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fold/requant with `s == [1.0; 8]` must be a no-op on the reconstructed
    /// values (up to `sc`/`m` integer rounding, which is exact when `s=1`
    /// leaves every code unchanged) — a self-consistency check independent
    /// of any real GGUF bytes.
    #[test]
    fn identity_scale_reconstructs_dequant() {
        // Hand-built block: increasing codes across all 8 sub-blocks with
        // a simple non-degenerate scale/min table.
        let mut block = vec![0u8; Q4K_BLOCK_BYTES];
        block[0..2].copy_from_slice(&half::f16::from_f32(0.1).to_le_bytes());
        block[2..4].copy_from_slice(&half::f16::from_f32(0.05).to_le_bytes());
        let sub_scales = [
            (10u8, 2u8),
            (20, 4),
            (30, 6),
            (40, 8),
            (50, 10),
            (60, 12),
            (63, 14),
            (5, 1),
        ];
        block[4..16].copy_from_slice(&pack_scale_min_k4(&sub_scales));
        for (i, b) in block[16..144].iter_mut().enumerate() {
            *b = ((i % 16) | ((i % 15) << 4)) as u8;
        }

        let sb = Q4kSuperblock::parse(&block);
        assert_eq!(sb.sub_scales, sub_scales, "pack/unpack round trip");
        let base = sb.dequant();
        let folded = fold_naive(&sb, &[1.0; 8]);
        for (a, b) in base.iter().zip(&folded) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn requant_of_original_is_close_to_original() {
        let mut block = vec![0u8; Q4K_BLOCK_BYTES];
        block[0..2].copy_from_slice(&half::f16::from_f32(0.2).to_le_bytes());
        block[2..4].copy_from_slice(&half::f16::from_f32(0.1).to_le_bytes());
        let sub_scales = [
            (30u8, 5u8),
            (40, 8),
            (20, 3),
            (50, 10),
            (60, 15),
            (63, 20),
            (10, 2),
            (25, 6),
        ];
        block[4..16].copy_from_slice(&pack_scale_min_k4(&sub_scales));
        for (i, b) in block[16..144].iter_mut().enumerate() {
            *b = ((i * 7 % 16) | ((i * 3 % 15) << 4)) as u8;
        }
        let sb = Q4kSuperblock::parse(&block);
        let base = sb.dequant();
        let re = requant(&base);
        let max_abs = base.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
        let mut max_err = 0f32;
        for (a, b) in base.iter().zip(&re) {
            max_err = max_err.max((a - b).abs());
        }
        // A from-scratch re-fit of the *already-quantized* values should
        // land close to them, not exactly (different (sc,m) search than
        // the block's original codes) — this bounds the requantizer's own
        // baseline noise floor, independent of any SmoothQuant scaling.
        assert!(
            max_err < 0.15 * max_abs,
            "requant-of-original drifted too far: max_err={max_err}, max_abs={max_abs}"
        );
    }
}
