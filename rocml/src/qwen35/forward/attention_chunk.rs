//! One full-attention layer's chunked-prefill step: batched Q/gate/K/V
//! projections (`LinearWeight::matmul`), batched per-head QK-norm and
//! partial rope, a batch KV-cache append, causal `attn_prefill` over the
//! whole chunk, the sigmoid output gate, and the output projection — the
//! chunked sibling of `attention::attention_step`, processing `chunk_len`
//! tokens (positions `pos_base..pos_base+chunk_len`) in one pass through
//! each kernel instead of one token at a time. See issue #6.

use super::chunk_kernels::ChunkKernels;
use super::chunk_scratch::ChunkScratch;
use super::kernels::HybridKernels;
use crate::error::RocmlError;
use crate::forward::kernels::{offset, Kernels};
use crate::profile::{self, OpKind, Profiler};
use crate::qwen35::cache::AttnPlane;
use crate::qwen35::config::Qwen35Config;
use crate::qwen35::weights::AttnLayerWeights;

#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_chunk_step(
    kernels: &Kernels,
    hybrid: &HybridKernels,
    chunk: &ChunkKernels,
    config: &Qwen35Config,
    layer: &AttnLayerWeights,
    plane: &mut AttnPlane,
    max_seq: u32,
    scratch: &mut ChunkScratch,
    pos_base: u32,
    chunk_len: u32,
    prof: Option<&Profiler>,
    layer_idx: Option<u32>,
) -> Result<(), RocmlError> {
    let hidden = config.embedding_length;
    let head_dim = config.head_dim;
    let n_heads = config.head_count;
    let n_kv_heads = config.head_count_kv;
    let q_dim = config.q_dim();
    let kv_dim = config.kv_dim();

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

    let q_out = if layer.has_output_gate {
        2 * q_dim
    } else {
        q_dim
    };
    let qkv_bytes = profile::matvec_bytes(layer.attn_q.byte_size(), q_out, hidden)
        + profile::matvec_bytes(layer.attn_k.byte_size(), kv_dim, hidden)
        + profile::matvec_bytes(layer.attn_v.byte_size(), kv_dim, hidden)
        + profile::norm_bytes(chunk_len * n_heads, head_dim)
        + profile::norm_bytes(chunk_len * n_kv_heads, head_dim);
    let qkv_flops = (profile::matvec_flops(q_out, hidden)
        + profile::matvec_flops(kv_dim, hidden) * 2)
        * chunk_len as u64
        + profile::norm_flops(chunk_len * n_heads, head_dim)
        + profile::norm_flops(chunk_len * n_kv_heads, head_dim);
    Profiler::scope(prof, layer_idx, OpKind::Qkv, qkv_bytes, qkv_flops, || {
        layer.attn_q.matmul(
            kernels,
            offset(&scratch.xn, 0),
            offset(&scratch.attn_q_raw, 0),
            chunk_len,
            q_out,
            hidden,
        )?;
        layer.attn_k.matmul(
            kernels,
            offset(&scratch.xn, 0),
            offset(&scratch.attn_k, 0),
            chunk_len,
            kv_dim,
            hidden,
        )?;
        layer.attn_v.matmul(
            kernels,
            offset(&scratch.xn, 0),
            offset(&scratch.attn_v, 0),
            chunk_len,
            kv_dim,
            hidden,
        )?;

        // Per-head extraction batched over the whole chunk (the decode path's
        // host-issued per-head loop doesn't scale to chunk_len*n_heads calls
        // — see `extract_heads_f32`'s doc comment).
        let stride = if layer.has_output_gate {
            2 * head_dim
        } else {
            head_dim
        };
        chunk.extract_heads(
            offset(&scratch.attn_q_raw, 0),
            offset(&scratch.attn_q, 0),
            chunk_len,
            n_heads,
            head_dim,
            stride,
            0,
        )?;
        if layer.has_output_gate {
            chunk.extract_heads(
                offset(&scratch.attn_q_raw, 0),
                offset(&scratch.attn_gate, 0),
                chunk_len,
                n_heads,
                head_dim,
                stride,
                head_dim,
            )?;
        }

        kernels.rmsnorm(
            offset(&scratch.attn_q, 0),
            offset(&layer.attn_q_norm, 0),
            offset(&scratch.attn_q, 0),
            chunk_len * n_heads,
            head_dim,
            config.rms_eps,
        )?;
        kernels.rmsnorm(
            offset(&scratch.attn_k, 0),
            offset(&layer.attn_k_norm, 0),
            offset(&scratch.attn_k, 0),
            chunk_len * n_kv_heads,
            head_dim,
            config.rms_eps,
        )?;

        hybrid.rope_partial(
            offset(&scratch.attn_q, 0),
            chunk_len,
            n_heads,
            head_dim,
            config.rope_dim_count,
            pos_base,
            config.rope_freq_base,
        )?;
        hybrid.rope_partial(
            offset(&scratch.attn_k, 0),
            chunk_len,
            n_kv_heads,
            head_dim,
            config.rope_dim_count,
            pos_base,
            config.rope_freq_base,
        )
    })?;

    let (attn_bytes, attn_flops) =
        attn_prefill_cost(n_heads, n_kv_heads, head_dim, pos_base, chunk_len);
    Profiler::scope(
        prof,
        layer_idx,
        OpKind::AttnDecode,
        attn_bytes,
        attn_flops,
        || {
            chunk.scatter_kv_chunk(
                offset(&scratch.attn_k, 0),
                offset(&scratch.attn_v, 0),
                offset(plane.k_buffer(), 0),
                offset(plane.v_buffer(), 0),
                n_kv_heads,
                head_dim,
                max_seq,
                chunk_len,
                pos_base,
            )?;

            let scale = 1.0f32 / (head_dim as f32).sqrt();
            kernels.attn_prefill(
                offset(&scratch.attn_q, 0),
                offset(plane.k_buffer(), 0),
                offset(plane.v_buffer(), 0),
                offset(&scratch.attn_concat, 0),
                n_heads,
                n_kv_heads,
                head_dim,
                max_seq,
                chunk_len,
                pos_base,
                scale,
            )
        },
    )?;

    let out_bytes = profile::matvec_bytes(layer.attn_output.byte_size(), hidden, q_dim);
    let out_flops = profile::matvec_flops(hidden, q_dim) * chunk_len as u64;
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
                    chunk_len * q_dim,
                )?;
            }

            layer.attn_output.matmul(
                kernels,
                offset(&scratch.attn_concat, 0),
                offset(&scratch.attn_out, 0),
                chunk_len,
                hidden,
                q_dim,
            )?;
            kernels.add_inplace(
                offset(&scratch.x, 0),
                offset(&scratch.attn_out, 0),
                chunk_len * hidden,
            )
        },
    )?;

    Ok(())
}

/// Analytical bytes/flops for the batched `attn_prefill` call: query row `i`
/// (global position `pos_base+i`) reads `pos_base+i+1` cached K/V positions,
/// so the total read scales with `sum_{i=0}^{chunk_len-1}(pos_base+i+1) =
/// chunk_len*pos_base + chunk_len*(chunk_len+1)/2` (closed form, no need to
/// loop). No split-K partial buffer here (unlike decode's `attn_decode`), so
/// just the K/V read plus the `[chunk_len, n_heads, head_dim]` output write.
fn attn_prefill_cost(
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    pos_base: u32,
    chunk_len: u32,
) -> (u64, u64) {
    let (cl, pb) = (chunk_len as u64, pos_base as u64);
    let sum_cur_len = cl * pb + cl * (cl + 1) / 2;
    let kv_read = 2 * n_kv_heads as u64 * head_dim as u64 * sum_cur_len * 4;
    let out_write = cl * n_heads as u64 * head_dim as u64 * 4;
    let bytes = kv_read + out_write;
    let flops = 4 * n_heads as u64 * head_dim as u64 * sum_cur_len;
    (bytes, flops)
}
