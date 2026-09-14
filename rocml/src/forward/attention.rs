//! One layer's GQA decode attention: q/k/v projection, per-head RMS norm,
//! rope, KV-cache append, then the fused `attn_decode` kernel (flash-decoding
//! style online-softmax causal attention — see `Kernels::attn_decode`'s doc
//! comment) in place of the old per-head gemv/softmax/gemv_t composition.
//! Batched (multi-token) prefill is a separate, later perf milestone —
//! today's token-serial prefill reuses this same decode-shaped step, so it
//! gets this kernel's win for free.
//!
//! Dense/Mixed per-layer dispatch (issue #2/#16's dense-architecture mixed-KV
//! port): `plane` is a `qwen35::cache::AttnLayerCache`, reused unchanged from
//! the hybrid path — this function's `AttnLayerCache::Mixed` arm mirrors
//! `qwen35::forward::attention::attention_step`'s own Mixed arm exactly
//! (same kernels, same call shape), since the mixed cache and its kernels
//! were already architecture-generic (`n_kv_heads`/`head_dim`/etc. are all
//! runtime arguments, never hybrid-specific config) — nothing here forks
//! kernel code, it only reuses it from a second call site.

use super::kernels::{attn_decode_splits, offset, Kernels};
use crate::cache::KvDtype;
use crate::config::ModelConfig;
use crate::error::RocmlError;
use crate::profile::{self, OpKind, Profiler};
use crate::qwen35::cache::AttnLayerCache;
use crate::qwen35::forward::kernels_mixed::MixedKernels;
use crate::weights::LayerWeights;

use super::scratch::Scratch;

#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_step(
    kernels: &Kernels,
    mixed: &MixedKernels,
    config: &ModelConfig,
    layer: &LayerWeights,
    plane: &mut AttnLayerCache,
    max_seq: u32,
    scratch: &mut Scratch,
    layer_idx: usize,
    pos: u32,
    prof: Option<&Profiler>,
) -> Result<(), RocmlError> {
    let hidden = config.embedding_length;
    let head_dim = config.head_dim;
    let n_heads = config.head_count;
    let n_kv_heads = config.head_count_kv;
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

    let (n_splits, _) = attn_decode_splits(n_kv_heads, cur_len);
    let attn_bytes = profile::attn_decode_bytes(n_heads, n_kv_heads, cur_len, head_dim, n_splits)
        + config.kv_dim() as u64 * 2 * 4;
    let attn_flops = profile::attn_decode_flops(n_heads, cur_len, head_dim);
    Profiler::scope(
        prof,
        Some(layer_idx),
        OpKind::AttnDecode,
        attn_bytes,
        attn_flops,
        || match plane {
            AttnLayerCache::Dense(plane) => {
                plane.append(
                    kernels, pos, max_seq, n_kv_heads, head_dim, &scratch.k, &scratch.v,
                )?;

                let scale = 1.0f32 / (head_dim as f32).sqrt();
                let decode = match plane.dtype() {
                    KvDtype::F16 => Kernels::attn_decode_f16,
                    KvDtype::F32 => Kernels::attn_decode,
                };
                decode(
                    kernels,
                    offset(&scratch.q, 0),
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
            }
            AttnLayerCache::Mixed(plane) => {
                plane.append(kernels, mixed, pos, &scratch.k, &scratch.v)?;

                let scale = 1.0f32 / (head_dim as f32).sqrt();
                let ptrs = plane.ptrs();
                let (n_splits, split_len) = attn_decode_splits(n_kv_heads, cur_len);
                mixed.attn_decode_partial_mixed(
                    ptrs.v_bits,
                    offset(&scratch.q, 0),
                    ptrs.sink_k,
                    ptrs.sink_v,
                    ptrs.window_k,
                    ptrs.window_v,
                    ptrs.bulk_k_codes,
                    ptrs.bulk_k_scales,
                    ptrs.bulk_v_codes,
                    ptrs.bulk_v_scales,
                    offset(&scratch.attn_partial_out, 0),
                    offset(&scratch.attn_partial_m, 0),
                    offset(&scratch.attn_partial_l, 0),
                    n_kv_heads,
                    n_heads / n_kv_heads,
                    head_dim,
                    ptrs.sink_len,
                    ptrs.window_len,
                    ptrs.window_base,
                    ptrs.bulk_cap,
                    ptrs.num_blocks_total,
                    cur_len,
                    split_len,
                    n_splits,
                    scale,
                )?;
                kernels.attn_decode_reduce(
                    offset(&scratch.attn_partial_out, 0),
                    offset(&scratch.attn_partial_m, 0),
                    offset(&scratch.attn_partial_l, 0),
                    offset(&scratch.attn_concat, 0),
                    n_heads,
                    head_dim,
                    n_splits,
                )
            }
        },
    )?;

    let out_bytes = profile::matvec_bytes(layer.attn_output.byte_size(), hidden, config.q_dim());
    let out_flops = profile::matvec_flops(hidden, config.q_dim());
    Profiler::scope(
        prof,
        Some(layer_idx),
        OpKind::AttnOut,
        out_bytes,
        out_flops,
        || {
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
/// up-front instead of measured per sub-op. Unaffected by the Dense/Mixed
/// cache dispatch above (the analytical cost model doesn't distinguish
/// them — see `qwen35::forward::attention::attention_step_cost`'s identical
/// choice).
pub(crate) fn attention_step_cost(
    config: &ModelConfig,
    layer: &LayerWeights,
    cur_len: u32,
) -> (u64, u64) {
    let hidden = config.embedding_length;
    let head_dim = config.head_dim;
    let n_heads = config.head_count;
    let n_kv_heads = config.head_count_kv;

    // n_splits=1: see the module doc for why this coarse prefill-only
    // estimate doesn't need the split-K occupancy heuristic.
    let bytes = profile::norm_bytes(1, hidden)
        + profile::matvec_bytes(layer.attn_q.byte_size(), config.q_dim(), hidden)
        + profile::matvec_bytes(layer.attn_k.byte_size(), config.kv_dim(), hidden)
        + profile::matvec_bytes(layer.attn_v.byte_size(), config.kv_dim(), hidden)
        + profile::norm_bytes(n_heads, head_dim)
        + profile::norm_bytes(n_kv_heads, head_dim)
        + profile::attn_decode_bytes(n_heads, n_kv_heads, cur_len, head_dim, 1)
        + profile::matvec_bytes(layer.attn_output.byte_size(), hidden, config.q_dim());
    let flops = profile::norm_flops(1, hidden)
        + profile::matvec_flops(config.q_dim(), hidden)
        + profile::matvec_flops(config.kv_dim(), hidden) * 2
        + profile::norm_flops(n_heads, head_dim)
        + profile::norm_flops(n_kv_heads, head_dim)
        + profile::attn_decode_flops(n_heads, cur_len, head_dim)
        + profile::matvec_flops(hidden, config.q_dim());
    (bytes, flops)
}
