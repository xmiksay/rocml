//! Issue #14 phase 2 step 3's debug quality simulation — split out of
//! `mod.rs` purely for the 400-line file cap. See
//! `LoadOptions::kv_rot_sim`/`kv_rot_sim_k`'s doc comments for the feature
//! itself; this file is only [`MixedAttnPlane::apply_rot_sim`] and its
//! private D2H/round-trip/H2D helper.

use half::f16;
use rocml_hip::DeviceBuffer;

use super::MixedAttnPlane;
use crate::error::RocmlError;
use crate::kv_quant::rotational::{default_sidecar, RotSidecar, TensorKind};

impl MixedAttnPlane {
    /// When `rot_sim` is set, overwrites this plane's *whole* window (the
    /// exact block about to be evicted — sink/boundary layers never reach
    /// this call at all) with the CPU rotational-quantization round-trip's
    /// reconstruction, before the normal `quantize_evict_k`/`_v` GPU
    /// kernels run on it. The subsequent real Q8/Q4 evict then quantizes an
    /// *already rotationally-quantized* value — a deliberately minimal-risk
    /// design: it reuses every existing kernel/scratch-buffer/dequant path
    /// unchanged (no new kernel, no new bulk storage format), adding only
    /// the negligible extra error of Q8-encoding an already-lossy
    /// reconstruction (~0.5% relative, per `kv_quant::quant_math`'s own
    /// measured Q8 error) on top of the ~10%+ error the 2-4 bpw rotational
    /// step itself introduces — see `.claude/CLAUDE.md` for the measured
    /// comparison. No-op (and no sidecar load) when `rot_sim` is `None`,
    /// matching every other call site's behavior exactly.
    pub(super) fn apply_rot_sim(&mut self) -> Result<(), RocmlError> {
        let Some(spec) = self.rot_sim else {
            return Ok(());
        };
        let sidecar = default_sidecar()?;
        corrupt_plane_via_round_trip(
            &mut self.window_v,
            &sidecar,
            TensorKind::V,
            spec.bpw,
            self.n_kv_heads,
            self.window_len,
            self.head_dim,
        )?;
        if spec.apply_to_k {
            corrupt_plane_via_round_trip(
                &mut self.window_k,
                &sidecar,
                TensorKind::K,
                spec.bpw,
                self.n_kv_heads,
                self.window_len,
                self.head_dim,
            )?;
        }
        Ok(())
    }
}

/// Copies `buf` (one `[n_kv_heads, window_len, head_dim]` f16 plane) to
/// host, round-trips every `head_dim`-length (head, token) vector through
/// the rotational-quantization reference at `bpw`, and copies the result
/// back — the D2H/H2D pair `apply_rot_sim` needs for V and (optionally) K.
/// Never called when `rot_sim` is `None` (see that method), so this always
/// pays its full cost when it runs — deliberately, per this module's own
/// "performance is irrelevant" design note.
fn corrupt_plane_via_round_trip(
    buf: &mut DeviceBuffer<f16>,
    sidecar: &RotSidecar,
    kind: TensorKind,
    bpw: u8,
    n_kv_heads: u32,
    window_len: u32,
    head_dim: u32,
) -> Result<(), RocmlError> {
    let mut host = vec![f16::from_f32(0.0); buf.len()];
    buf.copy_to_host(&mut host)?;

    let tensor_sidecar = sidecar.for_kind(kind);
    let head_dim = head_dim as usize;
    let window_len = window_len as usize;
    let mut vector = vec![0f32; head_dim];
    for h in 0..n_kv_heads as usize {
        for t in 0..window_len {
            let base = h * window_len * head_dim + t * head_dim;
            let slot = &mut host[base..base + head_dim];
            for (dst, src) in vector.iter_mut().zip(slot.iter()) {
                *dst = src.to_f32();
            }
            let reconstructed = tensor_sidecar.round_trip(&vector, bpw)?;
            for (dst, &v) in slot.iter_mut().zip(&reconstructed) {
                *dst = f16::from_f32(v);
            }
        }
    }

    buf.copy_from_host(&host)?;
    Ok(())
}
