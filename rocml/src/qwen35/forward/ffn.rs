//! qwen35's SwiGLU FFN step — identical math to the dense Qwen3 FFN
//! (`crate::forward::ffn`), just against this arch's own weight/scratch
//! types.

use rocml_hip::DeviceBuffer;

use super::scratch::Scratch;
use crate::error::RocmlError;
use crate::forward::kernels::{offset, Kernels};
use crate::profile::{self, OpKind, Profiler};
use crate::qwen35::weights::FfnWeights;

#[allow(clippy::too_many_arguments)]
pub(crate) fn ffn_step(
    kernels: &Kernels,
    ffn: &FfnWeights,
    post_attention_norm: &DeviceBuffer<f32>,
    hidden: u32,
    ffn_dim: u32,
    rms_eps: f32,
    scratch: &mut Scratch,
    prof: Option<&Profiler>,
    layer_idx: Option<u32>,
) -> Result<(), RocmlError> {
    Profiler::scope(
        prof,
        layer_idx,
        OpKind::Norm,
        profile::norm_bytes(1, hidden),
        profile::norm_flops(1, hidden),
        || {
            kernels.rmsnorm(
                offset(&scratch.x, 0),
                offset(post_attention_norm, 0),
                offset(&scratch.xn, 0),
                1,
                hidden,
                rms_eps,
            )
        },
    )?;

    let gate_up_bytes = profile::matvec_bytes(ffn.gate.byte_size(), ffn_dim, hidden)
        + profile::matvec_bytes(ffn.up.byte_size(), ffn_dim, hidden);
    let gate_up_flops = profile::matvec_flops(ffn_dim, hidden) * 2;
    Profiler::scope(
        prof,
        layer_idx,
        OpKind::FfnGateUp,
        gate_up_bytes,
        gate_up_flops,
        || {
            ffn.gate.matvec(
                kernels,
                offset(&scratch.xn, 0),
                offset(&scratch.ffn_gate, 0),
                ffn_dim,
                hidden,
            )?;
            ffn.up.matvec(
                kernels,
                offset(&scratch.xn, 0),
                offset(&scratch.ffn_up, 0),
                ffn_dim,
                hidden,
            )?;
            // In place: silu_mul_f32 only ever reads gate[i]/up[i] before
            // writing out[i], so aliasing out == gate is safe per-element.
            kernels.silu_mul(
                offset(&scratch.ffn_gate, 0),
                offset(&scratch.ffn_up, 0),
                offset(&scratch.ffn_gate, 0),
                ffn_dim,
            )
        },
    )?;

    let down_bytes = profile::matvec_bytes(ffn.down.byte_size(), hidden, ffn_dim);
    let down_flops = profile::matvec_flops(hidden, ffn_dim);
    Profiler::scope(
        prof,
        layer_idx,
        OpKind::FfnDown,
        down_bytes,
        down_flops,
        || {
            ffn.down.matvec(
                kernels,
                offset(&scratch.ffn_gate, 0),
                offset(&scratch.ffn_out, 0),
                hidden,
                ffn_dim,
            )?;
            kernels.add_inplace(offset(&scratch.x, 0), offset(&scratch.ffn_out, 0), hidden)
        },
    )?;

    Ok(())
}

/// Analytical bytes/flops for one full ffn_step (see
/// `crate::forward::attention::attention_step_cost`'s doc comment).
pub(crate) fn ffn_step_cost(ffn: &FfnWeights, hidden: u32, ffn_dim: u32) -> (u64, u64) {
    let bytes = profile::norm_bytes(1, hidden)
        + profile::matvec_bytes(ffn.gate.byte_size(), ffn_dim, hidden)
        + profile::matvec_bytes(ffn.up.byte_size(), ffn_dim, hidden)
        + profile::matvec_bytes(ffn.down.byte_size(), hidden, ffn_dim);
    let flops = profile::norm_flops(1, hidden)
        + profile::matvec_flops(ffn_dim, hidden) * 2
        + profile::matvec_flops(hidden, ffn_dim);
    (bytes, flops)
}
