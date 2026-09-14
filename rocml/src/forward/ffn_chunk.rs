//! Chunked-prefill sibling of `ffn::ffn_step`: identical gated-FFN math,
//! batched over `chunk_len` tokens via `LinearWeight::matmul` instead of
//! `matvec` (same `config.activation` dispatch as `ffn::ffn_step` — issue
//! #16). Mirrors `qwen35::forward::ffn_chunk::ffn_chunk_step` exactly,
//! adapted to the dense `LayerWeights`' flat field layout (no nested
//! `FfnWeights`) and with no `LayerCapture` hook (that diagnostic is
//! qwen35-only, issue #10).

use super::chunk_scratch::ChunkScratch;
use super::kernels::{offset, Kernels};
use super::kernels_act::ActivationKernels;
use crate::config::{Activation, ModelConfig};
use crate::error::RocmlError;
use crate::profile::{self, OpKind, Profiler};
use crate::weights::LayerWeights;

#[allow(clippy::too_many_arguments)]
pub(crate) fn ffn_chunk_step(
    kernels: &Kernels,
    act_kernels: &ActivationKernels,
    config: &ModelConfig,
    layer: &LayerWeights,
    scratch: &mut ChunkScratch,
    chunk_len: u32,
    prof: Option<&Profiler>,
    layer_idx: Option<u32>,
) -> Result<(), RocmlError> {
    let hidden = config.embedding_length;
    let ffn = config.feed_forward_length;

    Profiler::scope(
        prof,
        layer_idx,
        OpKind::Norm,
        profile::norm_bytes(chunk_len, hidden),
        profile::norm_flops(chunk_len, hidden),
        || {
            kernels.rmsnorm(
                offset(&scratch.x, 0),
                offset(&layer.ffn_norm, 0),
                offset(&scratch.xn, 0),
                chunk_len,
                hidden,
                config.rms_eps,
            )
        },
    )?;

    let gate_up_bytes = profile::matvec_bytes(layer.ffn_gate.byte_size(), ffn, hidden)
        + profile::matvec_bytes(layer.ffn_up.byte_size(), ffn, hidden);
    let gate_up_flops = profile::matvec_flops(ffn, hidden) * 2 * chunk_len as u64;
    Profiler::scope(
        prof,
        layer_idx,
        OpKind::FfnGateUp,
        gate_up_bytes,
        gate_up_flops,
        || {
            layer.ffn_gate.matmul(
                kernels,
                offset(&scratch.xn, 0),
                offset(&scratch.ffn_gate, 0),
                chunk_len,
                ffn,
                hidden,
                scratch.mmq_scratch(),
                scratch.splitk_scratch(),
            )?;
            layer.ffn_up.matmul(
                kernels,
                offset(&scratch.xn, 0),
                offset(&scratch.ffn_up, 0),
                chunk_len,
                ffn,
                hidden,
                scratch.mmq_scratch(),
                scratch.splitk_scratch(),
            )?;
            match config.activation {
                Activation::SiLu => kernels.silu_mul(
                    offset(&scratch.ffn_gate, 0),
                    offset(&scratch.ffn_up, 0),
                    offset(&scratch.ffn_gate, 0),
                    chunk_len * ffn,
                ),
                Activation::Gelu => act_kernels.gelu_mul(
                    offset(&scratch.ffn_gate, 0),
                    offset(&scratch.ffn_up, 0),
                    offset(&scratch.ffn_gate, 0),
                    chunk_len * ffn,
                ),
            }
        },
    )?;

    let down_bytes = profile::matvec_bytes(layer.ffn_down.byte_size(), hidden, ffn);
    let down_flops = profile::matvec_flops(hidden, ffn) * chunk_len as u64;
    Profiler::scope(
        prof,
        layer_idx,
        OpKind::FfnDown,
        down_bytes,
        down_flops,
        || {
            layer.ffn_down.matmul(
                kernels,
                offset(&scratch.ffn_gate, 0),
                offset(&scratch.ffn_out, 0),
                chunk_len,
                hidden,
                ffn,
                scratch.mmq_scratch(),
                scratch.splitk_scratch(),
            )
        },
    )?;
    kernels.add_inplace(
        offset(&scratch.x, 0),
        offset(&scratch.ffn_out, 0),
        chunk_len * hidden,
    )?;

    Ok(())
}
