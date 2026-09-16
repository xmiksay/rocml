//! qwen35moe's chunked-prefill FFN step (M3): the grouped-by-expert batched
//! GEMM `super::moe::moe_ffn_step`'s own module doc named as still open
//! after M1/M2. `super::ffn_chunk_dispatch` routes here whenever a MoE
//! layer's chunk isn't being `LayerCapture`-diagnosed (see that module's
//! doc comment for why the diagnostic path stays on the old per-row
//! `moe::moe_ffn_step` loop instead — capture is never on a hot path, and
//! duplicating its per-row instrumentation into this batched design would
//! be real complexity for zero throughput benefit).
//!
//! Mirrors llama.cpp's `mul_mat_id` pattern: route the whole chunk at once
//! (one router matmul + one `moe_route_topk_f32` call, `rows = chunk_len`,
//! instead of M1/M2's one-row-at-a-time router step), bucket the resulting
//! `chunk_len * top_k` (row, expert, weight) assignments by expert on the
//! host, then for each *distinct* expert the chunk actually touches: gather
//! that expert's assigned rows out of the chunk's activations into a
//! contiguous buffer, run one batched `gemm_quant` per projection (instead
//! of `rows_for_this_expert` separate `gemv_quant` calls), and
//! scatter-accumulate the weighted result back into the chunk-wide output.
//! A 512-token chunk touches on the order of 186/256 experts (measured, see
//! `.claude/CLAUDE.md`'s M3 section) — this turns up to `chunk_len * top_k`
//! tiny GEMVs into a couple hundred batched GEMMs, each averaging tens of
//! rows.
//!
//! The one per-chunk (not per-row) host sync this design needs is reading
//! back `topk_idx`/`topk_weight` for the whole chunk — same architectural
//! reason M1/M2's per-row version needed one per row (the router's output
//! must reach the host before the host can decide which experts to stream),
//! just amortized over `chunk_len` rows instead of paid once per row.

use rocml_core::gguf::GgufFile;
use rocml_hip::DeviceBuffer;

use super::chunk_scratch::ChunkScratch;
use super::kernels_moe::MoeKernels;
use super::kernels_moe_chunk::MoeChunkKernels;
use super::moe_cache::ExpertCache;
use super::moe_chunk_scratch::MoeChunkScratch;
use super::moe_scratch::MoeScratch;
use crate::error::RocmlError;
use crate::forward::kernels::{offset, DevPtr, Kernels};
use crate::qwen35::config::{MoeConfig, MOE_WEIGHT_SUM_EPS};
use crate::qwen35::weights::MoeFfnWeights;

/// One (row, weight) assignment within an expert's bucket.
type Assignment = (u32, f32);

/// Host-side scratch reused across every chunk/layer's grouping pass —
/// kept off the hot per-call stack purely to avoid re-allocating
/// `expert_count` `Vec`s (and their contents) every single call.
pub(crate) struct MoeChunkHost {
    buckets: Vec<Vec<Assignment>>,
    idx_host: Vec<i32>,
    weight_host: Vec<f32>,
    row_idx_host: Vec<u32>,
    weight_group_host: Vec<f32>,
}

impl MoeChunkHost {
    pub(crate) fn new(cfg: &MoeConfig, chunk_cap: u32) -> Self {
        let cap = chunk_cap as usize;
        let top_k = cfg.expert_used_count as usize;
        Self {
            buckets: vec![Vec::new(); cfg.expert_count as usize],
            idx_host: vec![0i32; cap * top_k],
            weight_host: vec![0f32; cap * top_k],
            row_idx_host: vec![0u32; cap],
            weight_group_host: vec![0f32; cap],
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn moe_ffn_chunk_step(
    kernels: &Kernels,
    moe_kernels: &MoeKernels,
    moe_chunk_kernels: &MoeChunkKernels,
    gguf: &GgufFile,
    moe: &MoeFfnWeights,
    moe_cfg: &MoeConfig,
    post_attention_norm: &DeviceBuffer<f32>,
    hidden: u32,
    rms_eps: f32,
    chunk_scratch: &mut ChunkScratch,
    moe_scratch: &mut MoeScratch,
    moe_chunk_scratch: &mut MoeChunkScratch,
    host: &mut MoeChunkHost,
    layer_idx: u32,
    mut expert_cache: Option<&mut ExpertCache>,
    chunk_len: u32,
) -> Result<(), RocmlError> {
    let mmq_scratch = chunk_scratch.mmq_scratch();
    let splitk_scratch = chunk_scratch.splitk_scratch();
    let xn = &chunk_scratch.xn;

    kernels.rmsnorm(
        offset(&chunk_scratch.x, 0),
        offset(post_attention_norm, 0),
        offset(xn, 0),
        chunk_len,
        hidden,
        rms_eps,
    )?;

    // Router: softmax(W_router . xn) -> top-k -> renormalize, batched over
    // the whole chunk in one `route_topk` call (it already takes `rows`).
    moe_chunk_kernels.gemm_xwt_f32(
        offset(xn, 0),
        offset(&moe.router, 0),
        offset(&moe_chunk_scratch.router_logits, 0),
        chunk_len,
        moe_cfg.expert_count,
        hidden,
    )?;
    moe_kernels.route_topk(
        offset(&moe_chunk_scratch.router_logits, 0),
        offset(&moe_chunk_scratch.topk_idx, 0),
        offset(&moe_chunk_scratch.topk_weight, 0),
        chunk_len,
        moe_cfg.expert_count,
        moe_cfg.expert_used_count,
        MOE_WEIGHT_SUM_EPS,
    )?;

    // Shared expert: always-on, batched exactly like the dense FFN's own
    // chunked path (`ffn_chunk_step`) — the accumulator's first write.
    moe.shared.gate.matmul(
        kernels,
        offset(xn, 0),
        offset(&moe_chunk_scratch.shared_a, 0),
        chunk_len,
        moe_cfg.shared_ff_len,
        hidden,
        mmq_scratch,
        splitk_scratch,
    )?;
    moe.shared.up.matmul(
        kernels,
        offset(xn, 0),
        offset(&moe_chunk_scratch.shared_b, 0),
        chunk_len,
        moe_cfg.shared_ff_len,
        hidden,
        mmq_scratch,
        splitk_scratch,
    )?;
    kernels.silu_mul(
        offset(&moe_chunk_scratch.shared_a, 0),
        offset(&moe_chunk_scratch.shared_b, 0),
        offset(&moe_chunk_scratch.shared_a, 0),
        chunk_len * moe_cfg.shared_ff_len,
    )?;
    moe.shared.down.matmul(
        kernels,
        offset(&moe_chunk_scratch.shared_a, 0),
        offset(&moe_chunk_scratch.shared_out, 0),
        chunk_len,
        hidden,
        moe_cfg.shared_ff_len,
        mmq_scratch,
        splitk_scratch,
    )?;
    moe_chunk_kernels.gemm_xwt_f32(
        offset(xn, 0),
        offset(&moe.shared_gate, 0),
        offset(&moe_chunk_scratch.shared_gate_logit, 0),
        chunk_len,
        1,
        hidden,
    )?;
    moe_chunk_kernels.shared_gate_write_chunk(
        offset(&moe_chunk_scratch.shared_out, 0),
        offset(&moe_chunk_scratch.shared_gate_logit, 0),
        offset(&moe_chunk_scratch.accum, 0),
        chunk_len,
        hidden,
    )?;

    // The one per-chunk host sync this design needs — see the module doc.
    let top_k = moe_cfg.expert_used_count as usize;
    let n_assignments = chunk_len as usize * top_k;
    moe_chunk_scratch
        .topk_idx
        .copy_range_to_host(0, &mut host.idx_host[..n_assignments])?;
    moe_chunk_scratch
        .topk_weight
        .copy_range_to_host(0, &mut host.weight_host[..n_assignments])?;

    for bucket in host.buckets.iter_mut() {
        bucket.clear();
    }
    for row in 0..chunk_len as usize {
        for k in 0..top_k {
            let raw_idx = host.idx_host[row * top_k + k];
            let expert = u32::try_from(raw_idx).map_err(|_| {
                RocmlError::Config(format!(
                    "moe router produced an invalid expert index {raw_idx} (must be in \
                     [0, {}))",
                    moe_cfg.expert_count
                ))
            })?;
            if expert >= moe_cfg.expert_count {
                return Err(RocmlError::Config(format!(
                    "moe router selected expert {expert}, out of range for expert_count {}",
                    moe_cfg.expert_count
                )));
            }
            host.buckets[expert as usize].push((row as u32, host.weight_host[row * top_k + k]));
        }
    }

    for expert in 0..moe_cfg.expert_count as usize {
        let rows_e = host.buckets[expert].len();
        if rows_e == 0 {
            continue;
        }
        // Indexed (not `.iter()`) so each read of `host.buckets[expert][i]`
        // is its own statement, ending that borrow before the next
        // statement writes a different field (`row_idx_host`/
        // `weight_group_host`) — avoids holding an immutable borrow of one
        // field across a mutable borrow of another through the same
        // `RefMut`.
        for i in 0..rows_e {
            let (row, weight) = host.buckets[expert][i];
            host.row_idx_host[i] = row;
            host.weight_group_host[i] = weight;
        }
        let rows_e = rows_e as u32;
        moe_chunk_scratch
            .group_row_idx
            .copy_prefix_from_host(&host.row_idx_host[..rows_e as usize])?;
        moe_chunk_scratch
            .group_weight
            .copy_prefix_from_host(&host.weight_group_host[..rows_e as usize])?;

        moe_chunk_kernels.gather_rows(
            offset(xn, 0),
            offset(&moe_chunk_scratch.group_row_idx, 0),
            offset(&moe_chunk_scratch.group_x, 0),
            rows_e,
            hidden,
        )?;

        let (gate_ptr, up_ptr, down_ptr) = load_expert(
            gguf,
            moe,
            expert_cache.as_deref_mut(),
            moe_scratch,
            layer_idx,
            expert as u32,
        )?;

        kernels.gemm_quant(
            moe.gate.dtype,
            offset(&moe_chunk_scratch.group_x, 0),
            gate_ptr,
            offset(&moe_chunk_scratch.group_gate, 0),
            rows_e,
            moe_cfg.expert_ff_len,
            hidden,
            mmq_scratch,
            false,
            splitk_scratch,
        )?;
        kernels.gemm_quant(
            moe.up.dtype,
            offset(&moe_chunk_scratch.group_x, 0),
            up_ptr,
            offset(&moe_chunk_scratch.group_up, 0),
            rows_e,
            moe_cfg.expert_ff_len,
            hidden,
            mmq_scratch,
            false,
            splitk_scratch,
        )?;
        kernels.silu_mul(
            offset(&moe_chunk_scratch.group_gate, 0),
            offset(&moe_chunk_scratch.group_up, 0),
            offset(&moe_chunk_scratch.group_gate, 0),
            rows_e * moe_cfg.expert_ff_len,
        )?;
        kernels.gemm_quant(
            moe.down.dtype,
            offset(&moe_chunk_scratch.group_gate, 0),
            down_ptr,
            offset(&moe_chunk_scratch.group_down, 0),
            rows_e,
            hidden,
            moe_cfg.expert_ff_len,
            mmq_scratch,
            false,
            splitk_scratch,
        )?;
        moe_chunk_kernels.scatter_weighted_accum(
            offset(&moe_chunk_scratch.group_down, 0),
            offset(&moe_chunk_scratch.group_row_idx, 0),
            offset(&moe_chunk_scratch.group_weight, 0),
            offset(&moe_chunk_scratch.accum, 0),
            rows_e,
            hidden,
        )?;
    }

    kernels.add_inplace(
        offset(&chunk_scratch.x, 0),
        offset(&moe_chunk_scratch.accum, 0),
        chunk_len * hidden,
    )
}

/// Cache-hit-or-copy for one expert's gate/up/down bytes — identical
/// fallback shape to `moe::moe_ffn_step`'s own inline match, factored out
/// here only because this function's already-long per-expert loop body
/// benefited from the split.
fn load_expert(
    gguf: &GgufFile,
    moe: &MoeFfnWeights,
    expert_cache: Option<&mut ExpertCache>,
    moe_scratch: &mut MoeScratch,
    layer_idx: u32,
    expert: u32,
) -> Result<(DevPtr, DevPtr, DevPtr), RocmlError> {
    match expert_cache {
        Some(cache) => {
            cache.ensure_loaded(gguf, (layer_idx, expert), &moe.gate, &moe.up, &moe.down)
        }
        None => {
            let gate_bytes = moe.gate.expert_bytes(gguf, expert)?;
            let up_bytes = moe.up.expert_bytes(gguf, expert)?;
            let down_bytes = moe.down.expert_bytes(gguf, expert)?;
            moe_scratch.stage_gate.copy_prefix_from_host(gate_bytes)?;
            moe_scratch.stage_up.copy_prefix_from_host(up_bytes)?;
            moe_scratch.stage_down.copy_prefix_from_host(down_bytes)?;
            Ok((
                offset(&moe_scratch.stage_gate, 0),
                offset(&moe_scratch.stage_up, 0),
                offset(&moe_scratch.stage_down, 0),
            ))
        }
    }
}
