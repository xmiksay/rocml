//! `attention_chunk_step_mixed`: the KIVI-style mixed-KV-cache sibling of
//! `attention_chunk::attention_chunk_step` (issue #2's chunked-prefill
//! follow-up) — same batched Q/gate/K/V projections, QK-norm and partial
//! rope preamble, and output-gate/output-projection suffix, but appends
//! into and attends over a `MixedAttnPlane` instead of a dense `AttnPlane`.
//! Kept as its own file (mirroring why `attention.rs`/`attention_chunk.rs`
//! are already separate per-path files) rather than threading an enum
//! branch through the dense function, since the middle section — cache
//! append and attention kernel dispatch — is genuinely different code, not
//! a thin wrapper.
//!
//! Unlike the dense path, this always dispatches to the flash/split-K
//! kernel (`FlashPrefillMixedKernels::attn_prefill_flash_mixed`) regardless
//! of depth — see `kernels/attn_prefill_flash_mixed.hip`'s module doc for
//! why no separate shallow-depth (`attn_prefill.hip`-style) mixed kernel
//! exists.

use super::chunk_kernels::ChunkKernels;
use super::chunk_scratch::ChunkScratch;
use super::kernels::HybridKernels;
use super::kernels_flash_mixed::FlashPrefillMixedKernels;
use super::kernels_mixed::MixedKernels;
use crate::error::RocmlError;
use crate::forward::kernels::{offset, Kernels};
use crate::profile::{self, OpKind, Profiler};
use crate::qwen35::cache_mixed::MixedAttnPlane;
use crate::qwen35::config::Qwen35Config;
use crate::qwen35::weights::AttnLayerWeights;

#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_chunk_step_mixed(
    kernels: &Kernels,
    hybrid: &HybridKernels,
    chunk: &ChunkKernels,
    mixed: &MixedKernels,
    flash_mixed: &FlashPrefillMixedKernels,
    config: &Qwen35Config,
    layer: &AttnLayerWeights,
    plane: &mut MixedAttnPlane,
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
            scratch.mmq_scratch(),
            scratch.splitk_scratch(),
        )?;
        layer.attn_k.matmul(
            kernels,
            offset(&scratch.xn, 0),
            offset(&scratch.attn_k, 0),
            chunk_len,
            kv_dim,
            hidden,
            scratch.mmq_scratch(),
            scratch.splitk_scratch(),
        )?;
        layer.attn_v.matmul(
            kernels,
            offset(&scratch.xn, 0),
            offset(&scratch.attn_v, 0),
            chunk_len,
            kv_dim,
            hidden,
            scratch.mmq_scratch(),
            scratch.splitk_scratch(),
        )?;

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
        attn_prefill_mixed_cost(n_heads, n_kv_heads, head_dim, pos_base, chunk_len);
    Profiler::scope(
        prof,
        layer_idx,
        OpKind::AttnDecode,
        attn_bytes,
        attn_flops,
        || {
            // Append-before-attend, same order the dense chunked path and
            // the decode-per-token mixed path both use — see
            // `cache_mixed::chunk`'s module doc for why this ordering is
            // what makes the resulting cache state (and this chunk's own
            // causal reads of it) bit-identical to token-serial appends.
            plane.append_chunk(
                chunk,
                mixed,
                pos_base,
                chunk_len,
                &scratch.attn_k,
                &scratch.attn_v,
            )?;

            let ptrs = plane.ptrs();
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            flash_mixed.attn_prefill_flash_mixed(
                ptrs.v_bits,
                offset(&scratch.attn_q, 0),
                ptrs.sink_k,
                ptrs.sink_v,
                ptrs.window_k,
                ptrs.window_v,
                ptrs.bulk_k_codes,
                ptrs.bulk_k_scales,
                ptrs.bulk_v_codes,
                ptrs.bulk_v_scales,
                offset(&scratch.attn_concat, 0),
                offset(&scratch.attn_flash_partial_out, 0),
                offset(&scratch.attn_flash_partial_m, 0),
                offset(&scratch.attn_flash_partial_l, 0),
                n_heads,
                n_kv_heads,
                head_dim,
                ptrs.sink_len,
                ptrs.window_len,
                ptrs.window_base,
                ptrs.bulk_cap,
                ptrs.num_blocks_total,
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
                scratch.mmq_scratch(),
                scratch.splitk_scratch(),
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

/// Analytical bytes/flops for the mixed chunked-attention call — same
/// closed-form causal-read-count math as
/// `attention_chunk::attn_prefill_cost`, since every position in `[0,
/// pos_base+chunk_len)` still gets read regardless of which region (sink,
/// bulk, or window) it physically lives in; only the per-byte cost differs
/// (quantized codes are smaller than fp16), which this analytical formula
/// doesn't model precisely — matching this codebase's existing convention
/// of treating the profiler's byte/FLOP counters as a roofline estimate,
/// not an exact accounting (see `crate::profile::cost`'s module doc).
fn attn_prefill_mixed_cost(
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
