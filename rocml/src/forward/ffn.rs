//! One layer's SwiGLU FFN: rmsnorm -> gate/up projections -> silu*up ->
//! down projection -> residual add.

use super::kernels::{offset, Kernels};
use super::scratch::Scratch;
use crate::config::ModelConfig;
use crate::error::RocmlError;
use crate::weights::LayerWeights;

pub(crate) fn ffn_step(
    kernels: &Kernels,
    config: &ModelConfig,
    layer: &LayerWeights,
    scratch: &mut Scratch,
) -> Result<(), RocmlError> {
    let hidden = config.embedding_length;
    let ffn = config.feed_forward_length;

    kernels.rmsnorm(
        offset(&scratch.x, 0),
        offset(&layer.ffn_norm, 0),
        offset(&scratch.xn, 0),
        1,
        hidden,
        config.rms_eps,
    )?;
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
    // In place: silu_mul_f32 only ever reads gate[i]/up[i] before writing
    // out[i], so aliasing out == gate is safe per-element.
    kernels.silu_mul(
        offset(&scratch.gate, 0),
        offset(&scratch.up, 0),
        offset(&scratch.gate, 0),
        ffn,
    )?;
    layer.ffn_down.matvec(
        kernels,
        offset(&scratch.gate, 0),
        offset(&scratch.ffn_out, 0),
        hidden,
        ffn,
    )?;
    kernels.add_inplace(offset(&scratch.x, 0), offset(&scratch.ffn_out, 0), hidden)?;

    Ok(())
}
