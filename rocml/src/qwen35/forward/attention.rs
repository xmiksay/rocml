//! One full-attention layer's decode step: fused Q/gate projection, per-head
//! QK-norm, partial rope, GQA causal attention, sigmoid output gate, output
//! projection. Mirrors Crane's `FullAttention::forward`
//! (crane-core/src/models/qwen3_5/modeling.rs) for a single timestep.

use super::kernels::HybridKernels;
use super::scratch::Scratch;
use crate::cache::KvDtype;
use crate::error::RocmlError;
use crate::forward::kernels::{attn_decode_splits, offset, Kernels};
use crate::profile::{self, OpKind, Profiler};
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
    prof: Option<&Profiler>,
    layer_idx: Option<u32>,
) -> Result<(), RocmlError> {
    let hidden = config.embedding_length;
    let head_dim = config.head_dim;
    let n_heads = config.head_count;
    let n_kv_heads = config.head_count_kv;
    let cur_len = pos + 1;
    let q_dim = config.q_dim();
    let kv_dim = config.kv_dim();

    Profiler::scope(
        prof,
        layer_idx,
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

    let q_out = if layer.has_output_gate {
        2 * q_dim
    } else {
        q_dim
    };
    let qkv_bytes = profile::matvec_bytes(layer.attn_q.byte_size(), q_out, hidden)
        + profile::matvec_bytes(layer.attn_k.byte_size(), kv_dim, hidden)
        + profile::matvec_bytes(layer.attn_v.byte_size(), kv_dim, hidden)
        + profile::norm_bytes(n_heads, head_dim)
        + profile::norm_bytes(n_kv_heads, head_dim);
    let qkv_flops = profile::matvec_flops(q_out, hidden)
        + profile::matvec_flops(kv_dim, hidden) * 2
        + profile::norm_flops(n_heads, head_dim)
        + profile::norm_flops(n_kv_heads, head_dim);
    Profiler::scope(prof, layer_idx, OpKind::Qkv, qkv_bytes, qkv_flops, || {
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
        )
    })?;

    let (n_splits, _) = attn_decode_splits(n_kv_heads, cur_len);
    let attn_bytes = profile::attn_decode_bytes(n_heads, n_kv_heads, cur_len, head_dim, n_splits)
        + kv_dim as u64 * 2 * 4;
    let attn_flops = profile::attn_decode_flops(n_heads, cur_len, head_dim);
    Profiler::scope(
        prof,
        layer_idx,
        OpKind::AttnDecode,
        attn_bytes,
        attn_flops,
        || {
            plane.append(
                kernels,
                pos,
                max_seq,
                n_kv_heads,
                head_dim,
                &scratch.attn_k,
                &scratch.attn_v,
            )?;

            let scale = 1.0f32 / (head_dim as f32).sqrt();
            let decode = match plane.dtype() {
                KvDtype::F16 => Kernels::attn_decode_f16,
                KvDtype::F32 => Kernels::attn_decode,
            };
            decode(
                kernels,
                offset(&scratch.attn_q, 0),
                plane.k_ptr(),
                plane.v_ptr(),
                offset(&scratch.attn_concat, 0),
                offset(&scratch.attn_partial_out, 0),
                offset(&scratch.attn_partial_m, 0),
                offset(&scratch.attn_partial_l, 0),
                n_heads,
                n_kv_heads,
                head_dim,
                max_seq,
                cur_len,
                scale,
            )
        },
    )?;

    let out_bytes = profile::matvec_bytes(layer.attn_output.byte_size(), hidden, q_dim);
    let out_flops = profile::matvec_flops(hidden, q_dim);
    Profiler::scope(
        prof,
        layer_idx,
        OpKind::AttnOut,
        out_bytes,
        out_flops,
        || {
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
            kernels.add_inplace(offset(&scratch.x, 0), offset(&scratch.attn_out, 0), hidden)
        },
    )?;

    Ok(())
}

/// Analytical bytes/flops for one full attention_step (see
/// `crate::forward::attention::attention_step_cost`'s doc comment).
pub(crate) fn attention_step_cost(
    config: &Qwen35Config,
    layer: &AttnLayerWeights,
    cur_len: u32,
) -> (u64, u64) {
    let hidden = config.embedding_length;
    let head_dim = config.head_dim;
    let n_heads = config.head_count;
    let n_kv_heads = config.head_count_kv;
    let q_dim = config.q_dim();
    let kv_dim = config.kv_dim();
    let q_out = if layer.has_output_gate {
        2 * q_dim
    } else {
        q_dim
    };

    // n_splits=1: this is prefill's coarse per-layer cost estimate (see the
    // dense analogue's doc comment), where cur_len grows one token at a
    // time and rarely reaches the depth the split-K heuristic kicks in at.
    let bytes = profile::norm_bytes(1, hidden)
        + profile::matvec_bytes(layer.attn_q.byte_size(), q_out, hidden)
        + profile::matvec_bytes(layer.attn_k.byte_size(), kv_dim, hidden)
        + profile::matvec_bytes(layer.attn_v.byte_size(), kv_dim, hidden)
        + profile::norm_bytes(n_heads, head_dim)
        + profile::norm_bytes(n_kv_heads, head_dim)
        + profile::attn_decode_bytes(n_heads, n_kv_heads, cur_len, head_dim, 1)
        + profile::matvec_bytes(layer.attn_output.byte_size(), hidden, q_dim);
    let flops = profile::norm_flops(1, hidden)
        + profile::matvec_flops(q_out, hidden)
        + profile::matvec_flops(kv_dim, hidden) * 2
        + profile::norm_flops(n_heads, head_dim)
        + profile::norm_flops(n_kv_heads, head_dim)
        + profile::attn_decode_flops(n_heads, cur_len, head_dim)
        + profile::matvec_flops(hidden, q_dim);
    (bytes, flops)
}
