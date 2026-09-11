//! One full-attention layer's decode step: fused Q/gate projection, per-head
//! QK-norm, partial rope, GQA causal attention, sigmoid output gate, output
//! projection. Mirrors Crane's `FullAttention::forward`
//! (crane-core/src/models/qwen3_5/modeling.rs) for a single timestep.

use super::kernels::HybridKernels;
use super::scratch::Scratch;
use crate::error::RocmlError;
use crate::forward::kernels::{offset, Kernels};
use crate::qwen35::cache::AttnPlane;
use crate::qwen35::config::Qwen35Config;
use crate::qwen35::weights::AttnLayerWeights;

#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_step(
    kernels: &Kernels,
    hybrid: &HybridKernels,
    config: &Qwen35Config,
    layer: &AttnLayerWeights,
    plane: &mut AttnPlane,
    max_seq: u32,
    scratch: &mut Scratch,
    pos: u32,
) -> Result<(), RocmlError> {
    let hidden = config.embedding_length;
    let head_dim = config.head_dim;
    let n_heads = config.head_count;
    let n_kv_heads = config.head_count_kv;
    let group = config.kv_group_size();
    let cur_len = pos + 1;
    let q_dim = config.q_dim();
    let kv_dim = config.kv_dim();

    kernels.rmsnorm(
        offset(&scratch.x, 0),
        offset(&layer.attn_norm, 0),
        offset(&scratch.xn, 0),
        1,
        hidden,
        config.rms_eps,
    )?;

    let q_out = if layer.has_output_gate {
        2 * q_dim
    } else {
        q_dim
    };
    layer.attn_q.matvec(
        kernels,
        offset(&scratch.xn, 0),
        offset(&scratch.attn_q_raw, 0),
        q_out,
        hidden,
    )?;
    layer.attn_k.matvec(
        kernels,
        offset(&scratch.xn, 0),
        offset(&scratch.attn_k, 0),
        kv_dim,
        hidden,
    )?;
    layer.attn_v.matvec(
        kernels,
        offset(&scratch.xn, 0),
        offset(&scratch.attn_v, 0),
        kv_dim,
        hidden,
    )?;

    // Per-head extraction: HF/GGUF fuse `[query(head_dim) | gate(head_dim)]`
    // per head (stride `2*head_dim`), not a flat `[Q | gate]` split — see
    // `AttnLayerWeights::attn_q`'s doc comment.
    let Scratch {
        attn_q_raw,
        attn_q,
        attn_gate,
        ..
    } = scratch;
    let stride = if layer.has_output_gate {
        2 * head_dim
    } else {
        head_dim
    };
    for h in 0..n_heads {
        attn_q.copy_from_device(
            (h * head_dim) as usize,
            attn_q_raw,
            (h * stride) as usize,
            head_dim as usize,
        )?;
        if layer.has_output_gate {
            attn_gate.copy_from_device(
                (h * head_dim) as usize,
                attn_q_raw,
                (h * stride + head_dim) as usize,
                head_dim as usize,
            )?;
        }
    }

    kernels.rmsnorm(
        offset(&scratch.attn_q, 0),
        offset(&layer.attn_q_norm, 0),
        offset(&scratch.attn_q, 0),
        n_heads,
        head_dim,
        config.rms_eps,
    )?;
    kernels.rmsnorm(
        offset(&scratch.attn_k, 0),
        offset(&layer.attn_k_norm, 0),
        offset(&scratch.attn_k, 0),
        n_kv_heads,
        head_dim,
        config.rms_eps,
    )?;

    hybrid.rope_partial(
        offset(&scratch.attn_q, 0),
        1,
        n_heads,
        head_dim,
        config.rope_dim_count,
        pos,
        config.rope_freq_base,
    )?;
    hybrid.rope_partial(
        offset(&scratch.attn_k, 0),
        1,
        n_kv_heads,
        head_dim,
        config.rope_dim_count,
        pos,
        config.rope_freq_base,
    )?;

    plane.append(
        pos,
        max_seq,
        n_kv_heads,
        head_dim,
        &scratch.attn_k,
        &scratch.attn_v,
    )?;

    let scale = 1.0f32 / (head_dim as f32).sqrt();
    for h in 0..n_heads {
        let kvh = h / group;
        let k_plane = offset(
            plane.k_buffer(),
            plane.head_plane_offset(kvh, max_seq, head_dim),
        );
        let q_head = offset(&scratch.attn_q, (h * head_dim) as usize);
        let score_row = offset(&scratch.attn_scores, (h * max_seq) as usize);
        kernels.gemv_f32(k_plane, q_head, score_row, cur_len, head_dim)?;
    }

    let valid_len_host = vec![cur_len; n_heads as usize];
    scratch.attn_valid_len.copy_from_host(&valid_len_host)?;
    kernels.softmax_varlen(
        offset(&scratch.attn_scores, 0),
        offset(&scratch.attn_valid_len, 0),
        n_heads,
        max_seq,
        scale,
    )?;

    for h in 0..n_heads {
        let kvh = h / group;
        let v_plane = offset(
            plane.v_buffer(),
            plane.head_plane_offset(kvh, max_seq, head_dim),
        );
        let probs = offset(&scratch.attn_scores, (h * max_seq) as usize);
        let out_head = offset(&scratch.attn_concat, (h * head_dim) as usize);
        kernels.gemv_t_f32(v_plane, probs, out_head, cur_len, head_dim)?;
    }

    if layer.has_output_gate {
        hybrid.sigmoid_mul(
            offset(&scratch.attn_concat, 0),
            offset(&scratch.attn_gate, 0),
            offset(&scratch.attn_concat, 0),
            q_dim,
        )?;
    }

    layer.attn_output.matvec(
        kernels,
        offset(&scratch.attn_concat, 0),
        offset(&scratch.attn_out, 0),
        hidden,
        q_dim,
    )?;
    kernels.add_inplace(offset(&scratch.x, 0), offset(&scratch.attn_out, 0), hidden)?;

    Ok(())
}
