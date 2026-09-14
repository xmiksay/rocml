//! Rotational KV quantization reference implementation — issue #14 phase 2
//! (the decision gate): block-diagonal 2D Givens rotation
//! ([`rotate`]) + a precomputed Lloyd-Max scalar codebook ([`lloyd`]),
//! combined per tensor kind in [`sidecar`]. Pure CPU math, GPU-free, mirrors
//! `kv_quant::quant_math`'s role for the existing KIVI-style scalar
//! encoding: this is the reference every measurement and the debug
//! model-level simulation (`RotSimSpec`, `MixedAttnPlane`'s eviction hook)
//! is checked against.
//!
//! Deliberately **not** wired into the production quantize-evict kernels —
//! phase 2 is a quality-only decision gate (issue #14: "near-zero loss ->
//! proceed to phase 3 [fast kernels]; visible degradation -> stop here").
//! Bit-packing into an actual `ceil(bpw * head_dim / 8)`-byte wire format is
//! also out of scope for the same reason: every measurement here only needs
//! the *bpw* accounting (levels per coordinate) to compute error and project
//! bytes/token, not a real packed encoding a kernel could read.

pub mod lloyd;
pub mod rotate;
pub mod sidecar;

pub use lloyd::LloydMaxCodebook;
pub use rotate::{
    calibrate_angles, fixed_angles, forward_rotate, inverse_rotate, l2_norm, Pairing,
};
pub use sidecar::{default_sidecar, RotSidecar, TensorKind, TensorSidecar};

/// Debug-only model-level simulation config (issue #14 phase 2 step 3):
/// which bit width to round-trip V (and optionally K) through at the
/// mixed-KV-cache quantize-on-evict site, via the CPU reference above —
/// see `LoadOptions::kv_rot_sim`/`kv_rot_sim_k` and
/// `MixedAttnPlane`'s eviction hook. Performance is irrelevant on this path
/// (a full D2H/H2D round trip per evicted block); only the simulation's
/// numerical correctness matters, since this exists purely to measure
/// whether the technique's quality loss is small enough to justify a real
/// fast-kernel investment (phase 3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RotSimSpec {
    pub bpw: u8,
    pub apply_to_k: bool,
}

/// Bytes/token this technique would need at `bpw`, vs the KIVI-style scalar
/// encoding's own bytes/token — the projection issue #14's acceptance
/// criteria asks for. `norm_bits` accounts for the per-vector fp16 L2 norm
/// (16 bits, amortized over `head_dim` coordinates); real angles/codebook
/// bytes are calibration-time constants, not per-token cost, so they're
/// excluded here exactly like the KIVI encoding's own per-channel/per-token
/// scale amortization is (`kv_quant::quant_math`'s module doc).
pub fn rotational_bytes_per_vector(head_dim: usize, bpw: u8) -> f64 {
    let norm_bits = 16.0;
    (head_dim as f64 * bpw as f64 + norm_bits) / 8.0
}

/// The existing scalar KIVI encoding's bytes/vector for comparison: Q8 is 1
/// byte/coordinate + a per-token f32 scale amortized over `head_dim`; Q4 is
/// a half byte/coordinate + the same per-token scale. Mirrors
/// `budget::mixed_kv_bytes_per_token`'s own per-token accounting.
pub fn scalar_bytes_per_vector(head_dim: usize, bits: u8) -> f64 {
    let scale_bytes = 4.0;
    (head_dim as f64 * bits as f64 / 8.0) + scale_bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotational_bytes_beat_scalar_q4_at_3bpw_on_a_wide_head() {
        let head_dim = 256;
        let rot = rotational_bytes_per_vector(head_dim, 3);
        let q4 = scalar_bytes_per_vector(head_dim, 4);
        assert!(
            rot < q4,
            "rotational 3bpw ({rot} B/vec) should beat scalar q4 ({q4} B/vec) at head_dim {head_dim}"
        );
    }
}
