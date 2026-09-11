//! One layer's GQA decode attention: q/k/v projection, per-head RMS norm,
//! rope, KV-cache append, then per-head `scores -> softmax -> weighted V`
//! composed from the plain gemv/softmax kernels (no fused attention kernel
//! yet — batched prefill/fused attention is a later perf milestone).

use super::kernels::{offset, Kernels};
use crate::cache::KvCache;
use crate::config::ModelConfig;
use crate::error::RocmlError;
use crate::profile::{self, OpKind, Profiler};
use crate::weights::LayerWeights;

use super::scratch::Scratch;

#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_step(
    kernels: &Kernels,
    config: &ModelConfig,
    layer: &LayerWeights,
    cache: &mut KvCache,
    scratch: &mut Scratch,
    layer_idx: usize,
    pos: u32,
    prof: Option<&Profiler>,
) -> Result<(), RocmlError> {
    let hidden = config.embedding_length;
    let head_dim = config.head_dim;
    let n_heads = config.head_count;
    let n_kv_heads = config.head_count_kv;
    let group = config.kv_group_size();
    let cur_len = pos + 1;
    let layer_idx = layer_idx as u32;

    Profiler::scope(
        prof,
        Some(layer_idx),
        OpKind::Norm,
        profile::norm_bytes(1, hidden),
        profile::norm_flops(1, hidden),
        || {
            kernels.rmsnorm(
                offset(&scratch.x, 0),
                offset(&layer.attn_norm, 0),
                offset(&scratch.xn, 0),
                1,
                hidden,
                config.rms_eps,
            )
        },
    )?;

    let qkv_bytes = profile::matvec_bytes(layer.attn_q.byte_size(), config.q_dim(), hidden)
        + profile::matvec_bytes(layer.attn_k.byte_size(), config.kv_dim(), hidden)
        + profile::matvec_bytes(layer.attn_v.byte_size(), config.kv_dim(), hidden)
        + profile::norm_bytes(n_heads, head_dim)
        + profile::norm_bytes(n_kv_heads, head_dim);
    let qkv_flops = profile::matvec_flops(config.q_dim(), hidden)
        + profile::matvec_flops(config.kv_dim(), hidden) * 2
        + profile::norm_flops(n_heads, head_dim)
        + profile::norm_flops(n_kv_heads, head_dim);
    Profiler::scope(
        prof,
        Some(layer_idx),
        OpKind::Qkv,
        qkv_bytes,
        qkv_flops,
        || {
            layer.attn_q.matvec(
                kernels,
                offset(&scratch.xn, 0),
                offset(&scratch.q, 0),
                config.q_dim(),
                hidden,
            )?;
            layer.attn_k.matvec(
                kernels,
                offset(&scratch.xn, 0),
                offset(&scratch.k, 0),
                config.kv_dim(),
                hidden,
            )?;
            layer.attn_v.matvec(
                kernels,
                offset(&scratch.xn, 0),
                offset(&scratch.v, 0),
                config.kv_dim(),
                hidden,
            )?;

            // Per-head RMS norm: q/k viewed as [heads, head_dim] row-major (heads
            // outermost, contiguous per-head), the same shared weight vector
            // applied to every head's row — exactly rmsnorm_f32's (rows, n) shape.
            kernels.rmsnorm(
                offset(&scratch.q, 0),
                offset(&layer.attn_q_norm, 0),
                offset(&scratch.q, 0),
                n_heads,
                head_dim,
                config.rms_eps,
            )?;
            kernels.rmsnorm(
                offset(&scratch.k, 0),
                offset(&layer.attn_k_norm, 0),
                offset(&scratch.k, 0),
                n_kv_heads,
                head_dim,
                config.rms_eps,
            )?;

            kernels.rope(
                offset(&scratch.q, 0),
                1,
                n_heads,
                head_dim,
                pos,
                config.rope_freq_base,
            )?;
            kernels.rope(
                offset(&scratch.k, 0),
                1,
                n_kv_heads,
                head_dim,
                pos,
                config.rope_freq_base,
            )
        },
    )?;

    let score_bytes =
        profile::attn_score_bytes(n_heads, cur_len, head_dim) + config.kv_dim() as u64 * 2 * 4;
    let score_flops = profile::attn_score_flops(n_heads, cur_len, head_dim);
    Profiler::scope(
        prof,
        Some(layer_idx),
        OpKind::AttnScore,
        score_bytes,
        score_flops,
        || {
            cache.append(layer_idx as usize, pos, &scratch.k, &scratch.v)?;

            let max_seq = cache.max_seq();
            let k_buf = cache.k_buffer(layer_idx as usize)?;
            for h in 0..n_heads {
                let kvh = h / group;
                let k_plane = offset(k_buf, cache.head_plane_offset(kvh));
                let q_head = offset(&scratch.q, (h * head_dim) as usize);
                let score_row = offset(&scratch.scores, (h * max_seq) as usize);
                kernels.gemv_f32(k_plane, q_head, score_row, cur_len, head_dim)?;
            }

            let valid_len_host = vec![cur_len; n_heads as usize];
            scratch.valid_len.copy_from_host(&valid_len_host)?;
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            kernels.softmax_varlen(
                offset(&scratch.scores, 0),
                offset(&scratch.valid_len, 0),
                n_heads,
                max_seq,
                scale,
            )
        },
    )?;

    let out_bytes = profile::attn_out_bytes(n_heads, cur_len, head_dim)
        + profile::matvec_bytes(layer.attn_output.byte_size(), hidden, config.q_dim());
    let out_flops = profile::attn_out_flops(n_heads, cur_len, head_dim)
        + profile::matvec_flops(hidden, config.q_dim());
    Profiler::scope(
        prof,
        Some(layer_idx),
        OpKind::AttnOut,
        out_bytes,
        out_flops,
        || {
            let max_seq = cache.max_seq();
            let v_buf = cache.v_buffer(layer_idx as usize)?;
            for h in 0..n_heads {
                let kvh = h / group;
                let v_plane = offset(v_buf, cache.head_plane_offset(kvh));
                let probs = offset(&scratch.scores, (h * max_seq) as usize);
                let out_head = offset(&scratch.attn_concat, (h * head_dim) as usize);
                kernels.gemv_t_f32(v_plane, probs, out_head, cur_len, head_dim)?;
            }

            layer.attn_output.matvec(
                kernels,
                offset(&scratch.attn_concat, 0),
                offset(&scratch.attn_out, 0),
                hidden,
                config.q_dim(),
            )?;
            kernels.add_inplace(offset(&scratch.x, 0), offset(&scratch.attn_out, 0), hidden)
        },
    )?;

    Ok(())
}

/// Analytical bytes/flops for one full attention_step, used by the prefill
/// coarse (per-layer) instrumentation path in `forward::Model::forward_token_profiled`
/// — the same formulas the fine-grained decode path above uses, just summed
/// up-front instead of measured per sub-op.
pub(crate) fn attention_step_cost(
    config: &ModelConfig,
    layer: &LayerWeights,
    cur_len: u32,
) -> (u64, u64) {
    let hidden = config.embedding_length;
    let head_dim = config.head_dim;
    let n_heads = config.head_count;
    let n_kv_heads = config.head_count_kv;

    let bytes = profile::norm_bytes(1, hidden)
        + profile::matvec_bytes(layer.attn_q.byte_size(), config.q_dim(), hidden)
        + profile::matvec_bytes(layer.attn_k.byte_size(), config.kv_dim(), hidden)
        + profile::matvec_bytes(layer.attn_v.byte_size(), config.kv_dim(), hidden)
        + profile::norm_bytes(n_heads, head_dim)
        + profile::norm_bytes(n_kv_heads, head_dim)
        + profile::attn_score_bytes(n_heads, cur_len, head_dim)
        + profile::attn_out_bytes(n_heads, cur_len, head_dim)
        + profile::matvec_bytes(layer.attn_output.byte_size(), hidden, config.q_dim());
    let flops = profile::norm_flops(1, hidden)
        + profile::matvec_flops(config.q_dim(), hidden)
        + profile::matvec_flops(config.kv_dim(), hidden) * 2
        + profile::norm_flops(n_heads, head_dim)
        + profile::norm_flops(n_kv_heads, head_dim)
        + profile::attn_score_flops(n_heads, cur_len, head_dim)
        + profile::attn_out_flops(n_heads, cur_len, head_dim)
        + profile::matvec_flops(hidden, config.q_dim());
    (bytes, flops)
}
