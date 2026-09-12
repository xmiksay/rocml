//! One Gated Delta Net layer's chunked-prefill step: batched input
//! projections (`LinearWeight::matmul`), the batched causal conv1d+SiLU
//! kernel, the batched gate kernel, and the batched recurrence kernel (which
//! also fuses in the per-head L2 norm — see `kernels/gdn_chunk.hip`'s doc
//! comment) — the chunked sibling of `gdn::gdn_layer_step`, processing
//! `chunk_len` tokens in one pass through each kernel. See issue #6.
//!
//! `L2_NORM_EPS` mirrors `gdn.rs`'s own constant (Crane's hardcoded `1e-6`,
//! independent of `rms_eps`) — duplicated rather than shared across the two
//! sibling step modules to keep each one self-contained (a one-`const` diff
//! isn't worth a shared-constants module for).

use super::chunk_kernels::ChunkKernels;
use super::chunk_scratch::ChunkScratch;
use crate::error::RocmlError;
use crate::forward::kernels::{offset, Kernels};
use crate::profile::{self, OpKind, Profiler};
use crate::qwen35::cache::GdnLayerState;
use crate::qwen35::config::{GdnConfig, Qwen35Config};
use crate::qwen35::weights::GdnLayerWeights;

const L2_NORM_EPS: f32 = 1e-6;

#[allow(clippy::too_many_arguments)]
pub(crate) fn gdn_chunk_step(
    kernels: &Kernels,
    chunk: &ChunkKernels,
    config: &Qwen35Config,
    layer: &GdnLayerWeights,
    state: &mut GdnLayerState,
    scratch: &mut ChunkScratch,
    chunk_len: u32,
    prof: Option<&Profiler>,
    layer_idx: Option<u32>,
) -> Result<(), RocmlError> {
    let hidden = config.embedding_length;
    let gdn: &GdnConfig = &config.gdn;

    Profiler::scope(
        prof,
        layer_idx,
        OpKind::Norm,
        profile::norm_bytes(chunk_len, hidden),
        profile::norm_flops(chunk_len, hidden),
        || {
            kernels.rmsnorm(
                offset(&scratch.x, 0),
                offset(&layer.attn_norm, 0),
                offset(&scratch.xn, 0),
                chunk_len,
                hidden,
                config.rms_eps,
            )
        },
    )?;

    let (conv_bytes, conv_flops) = gdn_conv_chunk_cost(gdn, hidden, layer, chunk_len);
    Profiler::scope(
        prof,
        layer_idx,
        OpKind::GdnConv,
        conv_bytes,
        conv_flops,
        || {
            layer.attn_qkv.matmul(
                kernels,
                offset(&scratch.xn, 0),
                offset(&scratch.gdn_qkv, 0),
                chunk_len,
                gdn.conv_dim,
                hidden,
            )?;
            layer.attn_gate.matmul(
                kernels,
                offset(&scratch.xn, 0),
                offset(&scratch.gdn_z, 0),
                chunk_len,
                gdn.value_dim,
                hidden,
            )?;
            layer.ssm_alpha.matmul(
                kernels,
                offset(&scratch.xn, 0),
                offset(&scratch.gdn_a_raw, 0),
                chunk_len,
                gdn.num_v_heads,
                hidden,
            )?;
            layer.ssm_beta.matmul(
                kernels,
                offset(&scratch.xn, 0),
                offset(&scratch.gdn_b_raw, 0),
                chunk_len,
                gdn.num_v_heads,
                hidden,
            )?;

            chunk.gdn_conv1d_chunk(
                offset(&scratch.gdn_qkv, 0),
                offset(&state.conv_state, 0),
                offset(&layer.ssm_conv1d, 0),
                offset(&scratch.gdn_conv_out, 0),
                gdn.conv_dim,
                gdn.conv_kernel,
                chunk_len,
            )?;
            chunk.gdn_gate_chunk(
                offset(&scratch.gdn_a_raw, 0),
                offset(&scratch.gdn_b_raw, 0),
                offset(&layer.ssm_a, 0),
                offset(&layer.ssm_dt_bias, 0),
                offset(&scratch.gdn_beta, 0),
                offset(&scratch.gdn_g, 0),
                gdn.num_v_heads,
                chunk_len,
            )
        },
    )?;

    let recur_bytes = profile::gdn_recur_bytes(gdn.num_v_heads, gdn.head_k_dim, gdn.head_v_dim)
        + profile::norm_bytes(chunk_len * gdn.num_k_heads, gdn.head_k_dim) * 2;
    let recur_flops = profile::gdn_recur_flops(gdn.num_v_heads, gdn.head_k_dim, gdn.head_v_dim)
        * chunk_len as u64
        + profile::norm_flops(chunk_len * gdn.num_k_heads, gdn.head_k_dim) * 2;
    Profiler::scope(
        prof,
        layer_idx,
        OpKind::GdnRecur,
        recur_bytes,
        recur_flops,
        || {
            chunk.gdn_recurrence_chunk(
                offset(&state.state, 0),
                offset(&scratch.gdn_conv_out, 0),
                offset(&scratch.gdn_beta, 0),
                offset(&scratch.gdn_g, 0),
                offset(&scratch.gdn_y, 0),
                gdn.num_v_heads,
                gdn.num_k_heads,
                gdn.head_k_dim,
                gdn.head_v_dim,
                gdn.conv_dim,
                gdn.key_dim,
                chunk_len,
                L2_NORM_EPS,
            )
        },
    )?;

    let out_bytes = profile::norm_bytes(chunk_len * gdn.num_v_heads, gdn.head_v_dim)
        + profile::matvec_bytes(layer.ssm_out.byte_size(), hidden, gdn.value_dim);
    let out_flops = profile::norm_flops(chunk_len * gdn.num_v_heads, gdn.head_v_dim)
        + profile::matvec_flops(hidden, gdn.value_dim) * chunk_len as u64;
    Profiler::scope(
        prof,
        layer_idx,
        OpKind::GdnOut,
        out_bytes,
        out_flops,
        || {
            kernels.rmsnorm(
                offset(&scratch.gdn_y, 0),
                offset(&layer.ssm_norm, 0),
                offset(&scratch.gdn_y, 0),
                chunk_len * gdn.num_v_heads,
                gdn.head_v_dim,
                config.rms_eps,
            )?;
            kernels.silu_mul(
                offset(&scratch.gdn_z, 0),
                offset(&scratch.gdn_y, 0),
                offset(&scratch.gdn_y, 0),
                chunk_len * gdn.value_dim,
            )?;

            layer.ssm_out.matmul(
                kernels,
                offset(&scratch.gdn_y, 0),
                offset(&scratch.gdn_out, 0),
                chunk_len,
                hidden,
                gdn.value_dim,
            )?;
            kernels.add_inplace(
                offset(&scratch.x, 0),
                offset(&scratch.gdn_out, 0),
                chunk_len * hidden,
            )
        },
    )?;

    Ok(())
}

fn gdn_conv_chunk_cost(
    gdn: &GdnConfig,
    hidden: u32,
    layer: &GdnLayerWeights,
    chunk_len: u32,
) -> (u64, u64) {
    let bytes = profile::matvec_bytes(layer.attn_qkv.byte_size(), gdn.conv_dim, hidden)
        + profile::matvec_bytes(layer.attn_gate.byte_size(), gdn.value_dim, hidden)
        + profile::matvec_bytes(layer.ssm_alpha.byte_size(), gdn.num_v_heads, hidden)
        + profile::matvec_bytes(layer.ssm_beta.byte_size(), gdn.num_v_heads, hidden)
        + profile::gdn_conv_bytes(gdn.conv_dim, gdn.conv_kernel) * chunk_len as u64;
    let flops = (profile::matvec_flops(gdn.conv_dim, hidden)
        + profile::matvec_flops(gdn.value_dim, hidden)
        + profile::matvec_flops(gdn.num_v_heads, hidden) * 2)
        * chunk_len as u64
        + profile::gdn_conv_flops(gdn.conv_dim, gdn.conv_kernel) * chunk_len as u64;
    (bytes, flops)
}
