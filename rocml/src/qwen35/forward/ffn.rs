//! qwen35's SwiGLU FFN step — identical math to the dense Qwen3 FFN
//! (`crate::forward::ffn`), just against this arch's own weight/scratch
//! types.

use rocml_hip::DeviceBuffer;

use super::scratch::Scratch;
use crate::error::RocmlError;
use crate::forward::kernels::{offset, Kernels};
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
) -> Result<(), RocmlError> {
    kernels.rmsnorm(
        offset(&scratch.x, 0),
        offset(post_attention_norm, 0),
        offset(&scratch.xn, 0),
        1,
        hidden,
        rms_eps,
    )?;
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
    // In place: silu_mul_f32 only ever reads gate[i]/up[i] before writing
    // out[i], so aliasing out == gate is safe per-element.
    kernels.silu_mul(
        offset(&scratch.ffn_gate, 0),
        offset(&scratch.ffn_up, 0),
        offset(&scratch.ffn_gate, 0),
        ffn_dim,
    )?;
    ffn.down.matvec(
        kernels,
        offset(&scratch.ffn_gate, 0),
        offset(&scratch.ffn_out, 0),
        hidden,
        ffn_dim,
    )?;
    kernels.add_inplace(offset(&scratch.x, 0), offset(&scratch.ffn_out, 0), hidden)?;

    Ok(())
}
