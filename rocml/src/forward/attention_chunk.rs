//! One layer's chunked-prefill GQA attention for the dense Qwen3
//! architecture (issue #16's dense chunked-prefill port): batched Q/K/V
//! projections (`LinearWeight::matmul`), batched per-head RMS norm and full
//! (non-partial) rope, a batch KV-cache append, causal attention over the
//! whole chunk, and the output projection — the chunked sibling of
//! `attention::attention_step`, processing `chunk_len` tokens (positions
//! `pos_base..pos_base+chunk_len`) in one pass through each kernel instead
//! of one token at a time.
//!
//! Simpler than the qwen35 hybrid's `qwen35::forward::attention_chunk`: the
//! dense architecture has no fused Q+output-gate projection, so there's no
//! `extract_heads` step (a dense `attn_q` projection's output is already
//! `[chunk_len, n_heads, head_dim]`, exactly the shape the per-head norm/
//! rope calls need) and no partial-rotary rope (dense rotates the whole
//! `head_dim`, via the same `Kernels::rope` the decode step uses — that
//! kernel already batches over `tokens`, so no new kernel was needed here).
//!
//! Dense/Mixed per-layer dispatch (mirrors `attention::attention_step`'s own
//! `AttnLayerCache` match): the mixed cache and its chunked-append/
//! chunked-read kernels are architecture-generic (`n_kv_heads`/`head_dim`/
//! etc. are all runtime arguments), so this reuses
//! `qwen35::cache_mixed::MixedAttnPlane::append_chunk` and
//! `qwen35::forward::kernels_flash_mixed::FlashPrefillMixedKernels` exactly
//! as the hybrid path does, from a second call site.

use super::chunk_scratch::ChunkScratch;
use super::kernels::{offset, Kernels};
use crate::cache::KvDtype;
use crate::config::ModelConfig;
use crate::error::RocmlError;
use crate::profile::{self, OpKind, Profiler};
use crate::qwen35::cache::AttnLayerCache;
use crate::qwen35::forward::chunk_kernels::ChunkKernels;
use crate::qwen35::forward::kernels_flash_mixed::FlashPrefillMixedKernels;
use crate::qwen35::forward::kernels_mixed::MixedKernels;
use crate::weights::LayerWeights;

/// Below this many total KV positions (`pos_base + chunk_len`), the
/// single-pass `attn_prefill` kernel wins on pure launch overhead over the
/// flash/split-K design — same threshold and reasoning as the qwen35 hybrid
/// path's `USE_FLASH_PREFILL_MIN_DEPTH` (picked there by measurement; not
/// re-swept for the dense path since it's the same kernel and cost model).
const USE_FLASH_PREFILL_MIN_DEPTH: u32 = 512;

#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_chunk_step(
    kernels: &Kernels,
    chunk: &ChunkKernels,
    mixed: &MixedKernels,
    flash_mixed: &FlashPrefillMixedKernels,
    config: &ModelConfig,
    layer: &LayerWeights,
    plane: &mut AttnLayerCache,
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

    let qkv_bytes = profile::matvec_bytes(layer.attn_q.byte_size(), q_dim, hidden)
        + profile::matvec_bytes(layer.attn_k.byte_size(), kv_dim, hidden)
        + profile::matvec_bytes(layer.attn_v.byte_size(), kv_dim, hidden)
        + profile::norm_bytes(chunk_len * n_heads, head_dim)
        + profile::norm_bytes(chunk_len * n_kv_heads, head_dim);
    let qkv_flops = (profile::matvec_flops(q_dim, hidden)
        + profile::matvec_flops(kv_dim, hidden) * 2)
        * chunk_len as u64
        + profile::norm_flops(chunk_len * n_heads, head_dim)
        + profile::norm_flops(chunk_len * n_kv_heads, head_dim);
    Profiler::scope(prof, layer_idx, OpKind::Qkv, qkv_bytes, qkv_flops, || {
        layer.attn_q.matmul(
            kernels,
            offset(&scratch.xn, 0),
            offset(&scratch.attn_q, 0),
            chunk_len,
            q_dim,
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

        kernels.rope(
            offset(&scratch.attn_q, 0),
            chunk_len,
            n_heads,
            head_dim,
            pos_base,
            config.rope_freq_base,
        )?;
        kernels.rope(
            offset(&scratch.attn_k, 0),
            chunk_len,
            n_kv_heads,
            head_dim,
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
        || match plane {
            AttnLayerCache::Dense(plane) => {
                let (k_ptr, v_ptr) = (plane.k_ptr(), plane.v_ptr());
                match plane.dtype() {
                    KvDtype::F32 => chunk.scatter_kv_chunk(
                        offset(&scratch.attn_k, 0),
                        offset(&scratch.attn_v, 0),
                        k_ptr,
                        v_ptr,
                        n_kv_heads,
                        head_dim,
                        max_seq,
                        chunk_len,
                        pos_base,
                    ),
                    KvDtype::F16 => chunk.scatter_kv_chunk_f16(
                        offset(&scratch.attn_k, 0),
                        offset(&scratch.attn_v, 0),
                        k_ptr,
                        v_ptr,
                        n_kv_heads,
                        head_dim,
                        max_seq,
                        chunk_len,
                        pos_base,
                    ),
                }?;

                let scale = 1.0f32 / (head_dim as f32).sqrt();
                // Same overhead-crossover reasoning as the qwen35 hybrid
                // path's identical dispatch (see `USE_FLASH_PREFILL_MIN_DEPTH`).
                if pos_base + chunk_len >= USE_FLASH_PREFILL_MIN_DEPTH {
                    match plane.dtype() {
                        KvDtype::F16 => kernels.flash.attn_prefill_flash_f16(
                            offset(&scratch.attn_q, 0),
                            k_ptr,
                            v_ptr,
                            offset(&scratch.attn_concat, 0),
                            offset(&scratch.attn_flash_partial_out, 0),
                            offset(&scratch.attn_flash_partial_m, 0),
                            offset(&scratch.attn_flash_partial_l, 0),
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            max_seq,
                            chunk_len,
                            pos_base,
                            scale,
                        ),
                        KvDtype::F32 => kernels.flash.attn_prefill_flash(
                            offset(&scratch.attn_q, 0),
                            k_ptr,
                            v_ptr,
                            offset(&scratch.attn_concat, 0),
                            offset(&scratch.attn_flash_partial_out, 0),
                            offset(&scratch.attn_flash_partial_m, 0),
                            offset(&scratch.attn_flash_partial_l, 0),
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            max_seq,
                            chunk_len,
                            pos_base,
                            scale,
                        ),
                    }
                } else {
                    let prefill = match plane.dtype() {
                        KvDtype::F16 => Kernels::attn_prefill_f16,
                        KvDtype::F32 => Kernels::attn_prefill,
                    };
                    prefill(
                        kernels,
                        offset(&scratch.attn_q, 0),
                        k_ptr,
                        v_ptr,
                        offset(&scratch.attn_concat, 0),
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        max_seq,
                        chunk_len,
                        pos_base,
                        scale,
                    )
                }
            }
            AttnLayerCache::Mixed(plane) => {
                // Append-before-attend, same order the decode-per-token
                // mixed path and the hybrid chunked path both use.
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
                // No depth threshold: the mixed cache always dispatches to
                // the flash/split-K kernel regardless of depth, same as the
                // hybrid path — see `attn_prefill_flash_mixed.hip`'s module
                // doc for why no shallow-depth mixed kernel exists.
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
            }
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

/// Analytical bytes/flops for the batched attention call — identical
/// closed-form causal-read-count math as
/// `qwen35::forward::attention_chunk::attn_prefill_cost` (same attention
/// shape, just this architecture's own `n_heads`/`n_kv_heads`/`head_dim`).
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
