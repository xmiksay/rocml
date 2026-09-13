//! One Gated Delta Net layer's chunked-prefill step: batched input
//! projections (`LinearWeight::matmul`), the batched causal conv1d+SiLU
//! kernel, the batched gate kernel, and the chunkwise (blocked delta-rule)
//! recurrence pipeline (`gdn_chunkwise::gdn_chunkwise_step`, issue #6's
//! chunkwise rewrite — see that module's and `kernels/gdn_chunkwise.hip`'s
//! doc comments for the algebra and per-head L2-norm derivation) — the
//! chunked sibling of `gdn::gdn_layer_step`, processing `chunk_len` tokens
//! in one pass through each stage.

use super::chunk_kernels::ChunkKernels;
use super::chunk_scratch::ChunkScratch;
use super::gdn_chunkwise::gdn_chunkwise_step;
use super::gdn_chunkwise_kernels::GdnChunkwiseKernels;
use super::layer_capture::LayerCapture;
use crate::error::RocmlError;
use crate::forward::kernels::{offset, Kernels};
use crate::profile::{self, OpKind, Profiler};
use crate::qwen35::cache::GdnLayerState;
use crate::qwen35::config::{GdnConfig, Qwen35Config};
use crate::qwen35::weights::GdnLayerWeights;

#[allow(clippy::too_many_arguments)]
pub(crate) fn gdn_chunk_step(
    kernels: &Kernels,
    chunk: &ChunkKernels,
    cw: &GdnChunkwiseKernels,
    config: &Qwen35Config,
    layer: &GdnLayerWeights,
    state: &mut GdnLayerState,
    scratch: &mut ChunkScratch,
    chunk_len: u32,
    prof: Option<&Profiler>,
    layer_idx: Option<u32>,
    mut capture: Option<&mut LayerCapture>,
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
    if let (Some(cap), Some(li)) = (capture.as_deref_mut(), layer_idx) {
        cap.record(li, "gdn_xn", &scratch.xn, chunk_len, hidden)?;
    }

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
                scratch.mmq_scratch(),
            )?;
            if let (Some(cap), Some(li)) = (capture.as_deref_mut(), layer_idx) {
                cap.record(li, "gdn_qkv_raw", &scratch.gdn_qkv, chunk_len, gdn.conv_dim)?;
            }
            layer.attn_gate.matmul(
                kernels,
                offset(&scratch.xn, 0),
                offset(&scratch.gdn_z, 0),
                chunk_len,
                gdn.value_dim,
                hidden,
                scratch.mmq_scratch(),
            )?;
            layer.ssm_alpha.matmul(
                kernels,
                offset(&scratch.xn, 0),
                offset(&scratch.gdn_a_raw, 0),
                chunk_len,
                gdn.num_v_heads,
                hidden,
                scratch.mmq_scratch(),
            )?;
            layer.ssm_beta.matmul(
                kernels,
                offset(&scratch.xn, 0),
                offset(&scratch.gdn_b_raw, 0),
                chunk_len,
                gdn.num_v_heads,
                hidden,
                scratch.mmq_scratch(),
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
            // Must follow gdn_conv1d_chunk on the same stream — it reads
            // conv_state (untouched by the kernel above) and x to carry the
            // state forward; see both kernels' module doc in gdn_chunk.hip.
            chunk.gdn_conv1d_chunk_state_update(
                offset(&scratch.gdn_qkv, 0),
                offset(&state.conv_state, 0),
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

    // Cost formulas are quadratic in the tile size, so approximate with one
    // `GDN_RECUR_TILE`-sized tile's cost times the tile count rather than
    // `chunk_len` directly — exact when `chunk_len <= GDN_RECUR_TILE` (every
    // current caller), a slight over-count on a hypothetical bigger request.
    let tiles = chunk_len
        .div_ceil(super::chunk_scratch::GDN_RECUR_TILE)
        .max(1);
    let tile_len = chunk_len.min(super::chunk_scratch::GDN_RECUR_TILE);
    let recur_bytes =
        profile::gdn_chunkwise_bytes(gdn.num_v_heads, tile_len, gdn.head_k_dim, gdn.head_v_dim)
            * tiles as u64;
    let recur_flops =
        profile::gdn_chunkwise_flops(gdn.num_v_heads, tile_len, gdn.head_k_dim, gdn.head_v_dim)
            * tiles as u64;
    Profiler::scope(
        prof,
        layer_idx,
        OpKind::GdnRecur,
        recur_bytes,
        recur_flops,
        || gdn_chunkwise_step(cw, gdn, state, scratch, chunk_len),
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
            if let (Some(cap), Some(li)) = (capture.as_deref_mut(), layer_idx) {
                cap.record(li, "gdn_y_silu", &scratch.gdn_y, chunk_len, gdn.value_dim)?;
            }

            layer.ssm_out.matmul(
                kernels,
                offset(&scratch.gdn_y, 0),
                offset(&scratch.gdn_out, 0),
                chunk_len,
                hidden,
                gdn.value_dim,
                scratch.mmq_scratch(),
            )?;
            if let (Some(cap), Some(li)) = (capture, layer_idx) {
                cap.record(li, "gdn_ssm_out_raw", &scratch.gdn_out, chunk_len, hidden)?;
            }
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
