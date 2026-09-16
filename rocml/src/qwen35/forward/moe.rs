//! qwen35moe's mixture-of-experts FFN step. Always processes exactly one
//! token (`x_row`) at a time — `super::decode_forward` calls it once per
//! decode step, and `super::chunk_forward` loops it once per row of a
//! prefill chunk (see that file's dispatch) — so chunked prefill gets no
//! batched-GEMM speedup here (M1's own deferred item, still open; M2's
//! scope was the LRU expert cache below, not a grouped-GEMM rewrite).
//!
//! Per-token cost: one router matvec, one shared-expert SwiGLU FFN, then
//! for each of `expert_used_count` selected experts, a lookup into the
//! VRAM-resident LRU cache (`super::moe_cache::ExpertCache`, M2) keyed by
//! `(layer, expert)` — a hit skips straight to `gemv_quant` against the
//! slot the expert's bytes already occupy from a previous token; a miss
//! copies the expert's raw quantized gate/up/down bytes
//! (`ExpertTensorMeta::expert_bytes`, read straight out of the GGUF's
//! mmap — never uploaded wholesale, see `crate::qwen35::weights::moe`'s
//! module doc) into the slot the cache assigned it, then runs the same
//! fused dequant-GEMV kernels (`Kernels::gemv_quant`) every other quantized
//! weight in this codebase runs through. `expert_cache: None` (an explicit
//! `--moe-cache-slots 0` override, or no VRAM left after everything else —
//! see `Model::load`) falls back to M1's single-slot `MoeScratch::stage_*`
//! buffers, always re-copying. One `topk_idx` device-to-host copy per token
//! orchestrates which experts to fetch — this sync is inherent to the
//! architecture (layer L's routing decision depends on layer L-1's full
//! output, so it can't be pipelined across layers within one token) and is
//! unchanged from M1.

use rocml_core::gguf::GgufFile;
use rocml_hip::DeviceBuffer;

use super::kernels_moe::MoeKernels;
use super::layer_capture::LayerCapture;
use super::moe_cache::ExpertCache;
use super::moe_scratch::MoeScratch;
use crate::error::RocmlError;
use crate::forward::kernels::{offset, DevPtr, Kernels};
use crate::qwen35::config::{MoeConfig, MOE_WEIGHT_SUM_EPS};
use crate::qwen35::weights::MoeFfnWeights;

/// Issue #10-style per-row capture context for `moe_ffn_step`, threaded
/// only from `forward_prompt_chunked_captured`'s final-chunk loop (see
/// `ffn_chunk_dispatch.rs`) — every other caller passes `None`. Bundles
/// `LayerCapture` with which row of the chunk this call is processing,
/// since `LayerCapture::record_row`/`record_row_scalar` need both to
/// assemble a `[total_rows, cols]` tensor across `total_rows` separate
/// calls (one token's worth of MoE compute per call, unlike the dense
/// FFN's batched chunk-wide capture).
pub(crate) struct MoeRowCapture<'a> {
    pub capture: &'a mut LayerCapture,
    pub row: u32,
    pub total_rows: u32,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn moe_ffn_step(
    kernels: &Kernels,
    moe_kernels: &MoeKernels,
    gguf: &GgufFile,
    moe: &MoeFfnWeights,
    moe_cfg: &MoeConfig,
    post_attention_norm: &DeviceBuffer<f32>,
    hidden: u32,
    rms_eps: f32,
    x_row: DevPtr,
    scratch: &mut MoeScratch,
    layer_idx: u32,
    mut expert_cache: Option<&mut ExpertCache>,
    mut row_capture: Option<MoeRowCapture<'_>>,
) -> Result<(), RocmlError> {
    kernels.rmsnorm(
        x_row,
        offset(post_attention_norm, 0),
        offset(&scratch.xn, 0),
        1,
        hidden,
        rms_eps,
    )?;
    record_row(&mut row_capture, layer_idx, "ffn_xn", &scratch.xn, hidden)?;

    // Router: softmax(W_router . xn) -> top-k -> renormalize.
    kernels.gemv_f32(
        offset(&moe.router, 0),
        offset(&scratch.xn, 0),
        offset(&scratch.router_logits, 0),
        moe_cfg.expert_count,
        hidden,
    )?;
    moe_kernels.route_topk(
        offset(&scratch.router_logits, 0),
        offset(&scratch.topk_idx, 0),
        offset(&scratch.topk_weight, 0),
        1,
        moe_cfg.expert_count,
        moe_cfg.expert_used_count,
        MOE_WEIGHT_SUM_EPS,
    )?;

    // Shared expert: always-on, gated by sigmoid(w_shared_gate . xn). Its
    // write is the accumulator's first write (never a `+=`), so no zeroing
    // step is needed.
    moe.shared.gate.matvec(
        kernels,
        offset(&scratch.xn, 0),
        offset(&scratch.ffn_a, 0),
        moe_cfg.shared_ff_len,
        hidden,
    )?;
    moe.shared.up.matvec(
        kernels,
        offset(&scratch.xn, 0),
        offset(&scratch.ffn_b, 0),
        moe_cfg.shared_ff_len,
        hidden,
    )?;
    kernels.silu_mul(
        offset(&scratch.ffn_a, 0),
        offset(&scratch.ffn_b, 0),
        offset(&scratch.ffn_a, 0),
        moe_cfg.shared_ff_len,
    )?;
    // llama.cpp's "ffn_swiglu" node for a MoE layer is the shared expert's
    // own SiLU(gate)*up (there is no single such tensor for the routed
    // experts — they're 8 independent matmuls, not batched into one).
    record_row(
        &mut row_capture,
        layer_idx,
        "moe_shexp_swiglu",
        &scratch.ffn_a,
        moe_cfg.shared_ff_len,
    )?;
    moe.shared.down.matvec(
        kernels,
        offset(&scratch.ffn_a, 0),
        offset(&scratch.expert_out, 0),
        hidden,
        moe_cfg.shared_ff_len,
    )?;
    // The shared expert's raw down-projection output, before gating —
    // llama.cpp's "ffn_shexp". Must be captured now: `expert_out` is reused
    // as scratch for every routed expert's own down-projection below.
    record_row(
        &mut row_capture,
        layer_idx,
        "moe_shexp_down",
        &scratch.expert_out,
        hidden,
    )?;
    kernels.gemv_f32(
        offset(&moe.shared_gate, 0),
        offset(&scratch.xn, 0),
        offset(&scratch.shared_gate_logit, 0),
        1,
        hidden,
    )?;
    if let Some(ctx) = row_capture.as_mut() {
        let mut logit = [0f32; 1];
        scratch.shared_gate_logit.copy_to_host(&mut logit)?;
        ctx.capture.record_row_scalar(
            layer_idx,
            "moe_shared_gate",
            ctx.row,
            ctx.total_rows,
            logit[0],
        );
        let sigmoid = 1.0 / (1.0 + (-logit[0]).exp());
        ctx.capture.record_row_scalar(
            layer_idx,
            "moe_shared_gate_sigmoid",
            ctx.row,
            ctx.total_rows,
            sigmoid,
        );
    }
    moe_kernels.shared_gate_write(
        offset(&scratch.expert_out, 0),
        offset(&scratch.shared_gate_logit, 0),
        offset(&scratch.accum, 0),
        hidden,
    )?;
    // The accumulator's first write: shared_expert_out * sigmoid(gate) —
    // llama.cpp's "ffn_shexp_gated". Must be captured now, before any
    // routed expert accumulates on top of it below.
    record_row(
        &mut row_capture,
        layer_idx,
        "moe_shexp_gated",
        &scratch.accum,
        hidden,
    )?;
    if row_capture.is_some() {
        // Only materialized when capturing (see `MoeScratch::routed_sum`'s
        // doc comment) — zero it here so the routed-only accumulation below
        // starts clean regardless of what a previous token/row left in it.
        // A host-side zero-fill is fine: this branch only runs under the
        // diagnostic `LayerCapture` path, never on the hot decode/prefill
        // loop, so its cost is irrelevant.
        scratch
            .routed_sum
            .copy_from_host(&vec![0f32; hidden as usize])?;
    }

    // Which experts were selected — the one host sync this design needs
    // (see the module doc); the renormalized weights themselves stay
    // device-resident and are read by `weighted_accum` via a pointer
    // offset, never copied to host.
    let top_k = moe_cfg.expert_used_count as usize;
    let mut idx_host = vec![0i32; top_k];
    scratch.topk_idx.copy_to_host(&mut idx_host)?;

    // M4 lever 2: the decode-overlap pipeline only ever applies to a real
    // decode step (`row_capture.is_none()` — the `LayerCapture` diagnostic
    // prefill path always passes `Some`, see this module's doc comment and
    // `moe_decode_overlap`'s own module doc) on a cache built with the
    // overlap flag on. The `else` branch below is the original, unmodified
    // M1/M2 inline loop — bit-identical whenever the `if` isn't taken,
    // which is every call site until `LoadOptions::moe_decode_overlap`/
    // `--moe-decode-overlap` is set.
    let overlap_active = row_capture.is_none()
        && expert_cache
            .as_deref()
            .is_some_and(super::moe_cache::ExpertCache::overlap_enabled);
    if overlap_active {
        let cache = expert_cache
            .as_deref_mut()
            .expect("overlap_active implies expert_cache is Some (checked above)");
        super::moe_decode_overlap::routed_experts_overlap(
            kernels,
            moe_kernels,
            gguf,
            moe,
            moe_cfg,
            hidden,
            scratch,
            layer_idx,
            cache,
            &idx_host,
        )?;
    } else {
        for (k, &raw_idx) in idx_host.iter().enumerate() {
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

            let (gate_ptr, up_ptr, down_ptr) = match expert_cache.as_mut() {
                Some(cache) => {
                    cache.ensure_loaded(gguf, (layer_idx, expert), &moe.gate, &moe.up, &moe.down)?
                }
                None => {
                    let gate_bytes = moe.gate.expert_bytes(gguf, expert)?;
                    let up_bytes = moe.up.expert_bytes(gguf, expert)?;
                    let down_bytes = moe.down.expert_bytes(gguf, expert)?;
                    scratch.stage_gate.copy_prefix_from_host(gate_bytes)?;
                    scratch.stage_up.copy_prefix_from_host(up_bytes)?;
                    scratch.stage_down.copy_prefix_from_host(down_bytes)?;
                    (
                        offset(&scratch.stage_gate, 0),
                        offset(&scratch.stage_up, 0),
                        offset(&scratch.stage_down, 0),
                    )
                }
            };

            kernels.gemv_quant(
                moe.gate.dtype,
                gate_ptr,
                offset(&scratch.xn, 0),
                offset(&scratch.ffn_a, 0),
                moe_cfg.expert_ff_len,
                hidden,
            )?;
            kernels.gemv_quant(
                moe.up.dtype,
                up_ptr,
                offset(&scratch.xn, 0),
                offset(&scratch.ffn_b, 0),
                moe_cfg.expert_ff_len,
                hidden,
            )?;
            kernels.silu_mul(
                offset(&scratch.ffn_a, 0),
                offset(&scratch.ffn_b, 0),
                offset(&scratch.ffn_a, 0),
                moe_cfg.expert_ff_len,
            )?;
            kernels.gemv_quant(
                moe.down.dtype,
                down_ptr,
                offset(&scratch.ffn_a, 0),
                offset(&scratch.expert_out, 0),
                hidden,
                moe_cfg.expert_ff_len,
            )?;

            moe_kernels.weighted_accum(
                offset(&scratch.expert_out, 0),
                offset(&scratch.topk_weight, k),
                offset(&scratch.accum, 0),
                hidden,
            )?;
            if row_capture.is_some() {
                moe_kernels.weighted_accum(
                    offset(&scratch.expert_out, 0),
                    offset(&scratch.topk_weight, k),
                    offset(&scratch.routed_sum, 0),
                    hidden,
                )?;
            }
        }
    }
    // The routed-only sum, before adding the shared expert's contribution —
    // llama.cpp's "ffn_moe_out". `accum` (captured below as "ffn_down_out",
    // reusing the dense path's own key/mapping since both represent "the
    // FFN's combined pre-residual output") already equals llama's "ffn_out"
    // = ffn_moe_out + ffn_shexp_gated, confirmed against the real reference
    // dump (see the M2 report).
    record_row(
        &mut row_capture,
        layer_idx,
        "moe_routed_sum",
        &scratch.routed_sum,
        hidden,
    )?;
    record_row(
        &mut row_capture,
        layer_idx,
        "ffn_down_out",
        &scratch.accum,
        hidden,
    )?;

    kernels.add_inplace(x_row, offset(&scratch.accum, 0), hidden)
}

/// Shared helper for the repeated `if let Some(ctx) = ...` capture
/// boilerplate above — a plain no-op when `row_capture` is `None`.
fn record_row(
    row_capture: &mut Option<MoeRowCapture<'_>>,
    layer_idx: u32,
    tensor: &str,
    buf: &DeviceBuffer<f32>,
    cols: u32,
) -> Result<(), RocmlError> {
    if let Some(ctx) = row_capture.as_mut() {
        ctx.capture
            .record_row(layer_idx, tensor, ctx.row, ctx.total_rows, buf, cols)?;
    }
    Ok(())
}

/// Analytical bytes/flops for one `moe_ffn_step` call — approximate (unlike
/// `ffn::ffn_step_cost`'s exact `LinearWeight::byte_size()` accounting,
/// since a routed expert's weight bytes vary by which experts were
/// selected, not known until the router runs), used only by
/// `Profiler`'s coarse per-layer prefill path (see `decode_forward.rs`'s
/// `coarse_prefill` branch) — never on the hot decode path.
pub(crate) fn moe_ffn_step_cost(
    moe: &MoeFfnWeights,
    moe_cfg: &MoeConfig,
    hidden: u32,
) -> (u64, u64) {
    let router_bytes = moe_cfg.expert_count as u64 * hidden as u64 * 4;
    let router_flops = 2 * moe_cfg.expert_count as u64 * hidden as u64;
    let shared_bytes =
        crate::profile::matvec_bytes(moe.shared.gate.byte_size(), moe_cfg.shared_ff_len, hidden)
            + crate::profile::matvec_bytes(
                moe.shared.up.byte_size(),
                moe_cfg.shared_ff_len,
                hidden,
            )
            + crate::profile::matvec_bytes(
                moe.shared.down.byte_size(),
                hidden,
                moe_cfg.shared_ff_len,
            );
    let shared_flops = crate::profile::matvec_flops(moe_cfg.shared_ff_len, hidden) * 2
        + crate::profile::matvec_flops(hidden, moe_cfg.shared_ff_len);
    // One expert's worth of gate+up+down bytes, times how many are
    // selected — `max_expert_bytes` isn't known here without the full
    // weights table, so this approximates every expert as the `gate`
    // tensor's own per-expert size (gate/up/down are all the same order of
    // magnitude on this checkpoint).
    let one_expert_bytes = moe.gate.per_expert_bytes as u64 * 2 + moe.down.per_expert_bytes as u64;
    let one_expert_flops = crate::profile::matvec_flops(moe_cfg.expert_ff_len, hidden) * 2
        + crate::profile::matvec_flops(hidden, moe_cfg.expert_ff_len);
    let routed_bytes = one_expert_bytes * moe_cfg.expert_used_count as u64;
    let routed_flops = one_expert_flops * moe_cfg.expert_used_count as u64;
    (
        router_bytes + shared_bytes + routed_bytes,
        router_flops + shared_flops + routed_flops,
    )
}
