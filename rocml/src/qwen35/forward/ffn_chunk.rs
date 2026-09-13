//! Chunked-prefill sibling of `ffn::ffn_step`: identical SwiGLU math, batched
//! over `chunk_len` tokens via `LinearWeight::matmul` instead of `matvec`.

use rocml_hip::DeviceBuffer;

use super::chunk_scratch::ChunkScratch;
use super::layer_capture::LayerCapture;
use crate::error::RocmlError;
use crate::forward::kernels::{offset, Kernels};
use crate::profile::{self, OpKind, Profiler};
use crate::qwen35::weights::FfnWeights;

#[allow(clippy::too_many_arguments)]
pub(crate) fn ffn_chunk_step(
    kernels: &Kernels,
    ffn: &FfnWeights,
    post_attention_norm: &DeviceBuffer<f32>,
    hidden: u32,
    ffn_dim: u32,
    rms_eps: f32,
    scratch: &mut ChunkScratch,
    chunk_len: u32,
    prof: Option<&Profiler>,
    layer_idx: Option<u32>,
    mut capture: Option<&mut LayerCapture>,
) -> Result<(), RocmlError> {
    if let (Some(cap), Some(li)) = (capture.as_deref_mut(), layer_idx) {
        cap.record(li, "resid_pre_ffn", &scratch.x, chunk_len, hidden)?;
    }
    Profiler::scope(
        prof,
        layer_idx,
        OpKind::Norm,
        profile::norm_bytes(chunk_len, hidden),
        profile::norm_flops(chunk_len, hidden),
        || {
            kernels.rmsnorm(
                offset(&scratch.x, 0),
                offset(post_attention_norm, 0),
                offset(&scratch.xn, 0),
                chunk_len,
                hidden,
                rms_eps,
            )
        },
    )?;
    if let (Some(cap), Some(li)) = (capture.as_deref_mut(), layer_idx) {
        cap.record(li, "ffn_xn", &scratch.xn, chunk_len, hidden)?;
    }

    let gate_up_bytes = profile::matvec_bytes(ffn.gate.byte_size(), ffn_dim, hidden)
        + profile::matvec_bytes(ffn.up.byte_size(), ffn_dim, hidden);
    let gate_up_flops = profile::matvec_flops(ffn_dim, hidden) * 2 * chunk_len as u64;
    Profiler::scope(
        prof,
        layer_idx,
        OpKind::FfnGateUp,
        gate_up_bytes,
        gate_up_flops,
        || {
            ffn.gate.matmul(
                kernels,
                offset(&scratch.xn, 0),
                offset(&scratch.ffn_gate, 0),
                chunk_len,
                ffn_dim,
                hidden,
                scratch.mmq_scratch(),
                scratch.splitk_scratch(),
            )?;
            ffn.up.matmul(
                kernels,
                offset(&scratch.xn, 0),
                offset(&scratch.ffn_up, 0),
                chunk_len,
                ffn_dim,
                hidden,
                scratch.mmq_scratch(),
                scratch.splitk_scratch(),
            )?;
            kernels.silu_mul(
                offset(&scratch.ffn_gate, 0),
                offset(&scratch.ffn_up, 0),
                offset(&scratch.ffn_gate, 0),
                chunk_len * ffn_dim,
            )
        },
    )?;
    if let (Some(cap), Some(li)) = (capture.as_deref_mut(), layer_idx) {
        cap.record(li, "ffn_gate_silu", &scratch.ffn_gate, chunk_len, ffn_dim)?;
    }

    let down_bytes = profile::matvec_bytes(ffn.down.byte_size(), hidden, ffn_dim);
    let down_flops = profile::matvec_flops(hidden, ffn_dim) * chunk_len as u64;
    Profiler::scope(
        prof,
        layer_idx,
        OpKind::FfnDown,
        down_bytes,
        down_flops,
        || {
            ffn.down.matmul(
                kernels,
                offset(&scratch.ffn_gate, 0),
                offset(&scratch.ffn_out, 0),
                chunk_len,
                hidden,
                ffn_dim,
                scratch.mmq_scratch(),
                scratch.splitk_scratch(),
            )
        },
    )?;
    if let (Some(cap), Some(li)) = (capture, layer_idx) {
        cap.record(li, "ffn_down_out", &scratch.ffn_out, chunk_len, hidden)?;
    }
    kernels.add_inplace(
        offset(&scratch.x, 0),
        offset(&scratch.ffn_out, 0),
        chunk_len * hidden,
    )?;

    Ok(())
}
