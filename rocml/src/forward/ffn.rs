//! One layer's SwiGLU FFN: rmsnorm -> gate/up projections -> silu*up ->
//! down projection -> residual add.

use super::kernels::{offset, Kernels};
use super::scratch::Scratch;
use crate::config::ModelConfig;
use crate::error::RocmlError;
use crate::profile::{self, OpKind, Profiler};
use crate::weights::LayerWeights;

pub(crate) fn ffn_step(
    kernels: &Kernels,
    config: &ModelConfig,
    layer: &LayerWeights,
    scratch: &mut Scratch,
    prof: Option<&Profiler>,
    layer_idx: Option<u32>,
) -> Result<(), RocmlError> {
    let hidden = config.embedding_length;
    let ffn = config.feed_forward_length;

    Profiler::scope(
        prof,
        layer_idx,
        OpKind::Norm,
        profile::norm_bytes(1, hidden),
        profile::norm_flops(1, hidden),
        || {
            kernels.rmsnorm(
                offset(&scratch.x, 0),
                offset(&layer.ffn_norm, 0),
                offset(&scratch.xn, 0),
                1,
                hidden,
                config.rms_eps,
            )
        },
    )?;

    let gate_up_bytes = profile::matvec_bytes(layer.ffn_gate.byte_size(), ffn, hidden)
        + profile::matvec_bytes(layer.ffn_up.byte_size(), ffn, hidden);
    let gate_up_flops = profile::matvec_flops(ffn, hidden) * 2;
    Profiler::scope(
        prof,
        layer_idx,
        OpKind::FfnGateUp,
        gate_up_bytes,
        gate_up_flops,
        || {
            layer.ffn_gate.matvec(
                kernels,
                offset(&scratch.xn, 0),
                offset(&scratch.gate, 0),
                ffn,
                hidden,
            )?;
            layer.ffn_up.matvec(
                kernels,
                offset(&scratch.xn, 0),
                offset(&scratch.up, 0),
                ffn,
                hidden,
            )?;
            // In place: silu_mul_f32 only ever reads gate[i]/up[i] before
            // writing out[i], so aliasing out == gate is safe per-element.
            kernels.silu_mul(
                offset(&scratch.gate, 0),
                offset(&scratch.up, 0),
                offset(&scratch.gate, 0),
                ffn,
            )
        },
    )?;

    let down_bytes = profile::matvec_bytes(layer.ffn_down.byte_size(), hidden, ffn);
    let down_flops = profile::matvec_flops(hidden, ffn);
    Profiler::scope(
        prof,
        layer_idx,
        OpKind::FfnDown,
        down_bytes,
        down_flops,
        || {
            layer.ffn_down.matvec(
                kernels,
                offset(&scratch.gate, 0),
                offset(&scratch.ffn_out, 0),
                hidden,
                ffn,
            )?;
            kernels.add_inplace(offset(&scratch.x, 0), offset(&scratch.ffn_out, 0), hidden)
        },
    )?;

    Ok(())
}

/// Analytical bytes/flops for one full ffn_step — see
/// `attention::attention_step_cost`'s doc comment for why this exists
/// (prefill's coarse per-layer instrumentation path).
pub(crate) fn ffn_step_cost(config: &ModelConfig, layer: &LayerWeights) -> (u64, u64) {
    let hidden = config.embedding_length;
    let ffn = config.feed_forward_length;
    let bytes = profile::norm_bytes(1, hidden)
        + profile::matvec_bytes(layer.ffn_gate.byte_size(), ffn, hidden)
        + profile::matvec_bytes(layer.ffn_up.byte_size(), ffn, hidden)
        + profile::matvec_bytes(layer.ffn_down.byte_size(), hidden, ffn);
    let flops = profile::norm_flops(1, hidden)
        + profile::matvec_flops(ffn, hidden) * 2
        + profile::matvec_flops(hidden, ffn);
    (bytes, flops)
}
