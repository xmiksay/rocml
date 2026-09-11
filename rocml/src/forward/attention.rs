//! One layer's GQA decode attention: q/k/v projection, per-head RMS norm,
//! rope, KV-cache append, then per-head `scores -> softmax -> weighted V`
//! composed from the plain gemv/softmax kernels (no fused attention kernel
//! yet — batched prefill/fused attention is a later perf milestone).

use super::kernels::{offset, Kernels};
use crate::cache::KvCache;
use crate::config::ModelConfig;
use crate::error::RocmlError;
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
) -> Result<(), RocmlError> {
    let hidden = config.embedding_length;
    let head_dim = config.head_dim;
    let n_heads = config.head_count;
    let n_kv_heads = config.head_count_kv;
    let group = config.kv_group_size();
    let cur_len = pos + 1;

    kernels.rmsnorm(
        offset(&scratch.x, 0),
        offset(&layer.attn_norm, 0),
        offset(&scratch.xn, 0),
        1,
        hidden,
        config.rms_eps,
    )?;

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
    )?;

    cache.append(layer_idx, pos, &scratch.k, &scratch.v)?;

    let max_seq = cache.max_seq();
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let k_buf = cache.k_buffer(layer_idx)?;
    for h in 0..n_heads {
        let kvh = h / group;
        let k_plane = offset(k_buf, cache.head_plane_offset(kvh));
        let q_head = offset(&scratch.q, (h * head_dim) as usize);
        let score_row = offset(&scratch.scores, (h * max_seq) as usize);
        kernels.gemv_f32(k_plane, q_head, score_row, cur_len, head_dim)?;
    }

    let valid_len_host = vec![cur_len; n_heads as usize];
    scratch.valid_len.copy_from_host(&valid_len_host)?;
    kernels.softmax_varlen(
        offset(&scratch.scores, 0),
        offset(&scratch.valid_len, 0),
        n_heads,
        max_seq,
        scale,
    )?;

    let v_buf = cache.v_buffer(layer_idx)?;
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
    kernels.add_inplace(offset(&scratch.x, 0), offset(&scratch.attn_out, 0), hidden)?;

    Ok(())
}
