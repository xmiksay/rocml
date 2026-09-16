//! qwen35moe's mixture-of-experts FFN step (M1): the same per-layer slot
//! `ffn::ffn_step`/`ffn_chunk::ffn_chunk_step` fill for a dense SwiGLU FFN,
//! but routing through a softmax top-k gate plus one always-on shared
//! expert. Always processes exactly one token (`x_row`) at a time —
//! `super::decode_forward` calls it once per decode step, and
//! `super::chunk_forward` loops it once per row of a prefill chunk (see
//! that file's dispatch) — so chunked prefill gets no batched-GEMM speedup
//! in M1 (deferred to a later milestone alongside the LRU expert cache);
//! this milestone's bar is correctness, not throughput.
//!
//! Per-token cost: one router matvec, one shared-expert SwiGLU FFN, then
//! for each of `expert_used_count` selected experts, an H2D copy of that
//! expert's raw quantized gate/up/down bytes (`ExpertTensorMeta::expert_bytes`,
//! read straight out of the GGUF's mmap — never uploaded wholesale, see
//! `crate::qwen35::weights::moe`'s module doc) into a small reusable
//! staging buffer, followed by the same fused dequant-GEMV kernels
//! (`Kernels::gemv_quant`) every other quantized weight in this codebase
//! runs through. One `topk_idx` device-to-host copy per token orchestrates
//! which experts to stage — the one sync-per-token this design accepts for
//! M1's "correctness first, avoid a sync only if reasonably easy" brief.

use rocml_core::gguf::GgufFile;
use rocml_hip::DeviceBuffer;

use super::kernels_moe::MoeKernels;
use super::moe_scratch::MoeScratch;
use crate::error::RocmlError;
use crate::forward::kernels::{offset, DevPtr, Kernels};
use crate::qwen35::config::{MoeConfig, MOE_WEIGHT_SUM_EPS};
use crate::qwen35::weights::MoeFfnWeights;

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
) -> Result<(), RocmlError> {
    kernels.rmsnorm(
        x_row,
        offset(post_attention_norm, 0),
        offset(&scratch.xn, 0),
        1,
        hidden,
        rms_eps,
    )?;

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
    moe.shared.down.matvec(
        kernels,
        offset(&scratch.ffn_a, 0),
        offset(&scratch.expert_out, 0),
        hidden,
        moe_cfg.shared_ff_len,
    )?;
    kernels.gemv_f32(
        offset(&moe.shared_gate, 0),
        offset(&scratch.xn, 0),
        offset(&scratch.shared_gate_logit, 0),
        1,
        hidden,
    )?;
    moe_kernels.shared_gate_write(
        offset(&scratch.expert_out, 0),
        offset(&scratch.shared_gate_logit, 0),
        offset(&scratch.accum, 0),
        hidden,
    )?;

    // Which experts were selected — the one host sync this design needs
    // (see the module doc); the renormalized weights themselves stay
    // device-resident and are read by `weighted_accum` via a pointer
    // offset, never copied to host.
    let top_k = moe_cfg.expert_used_count as usize;
    let mut idx_host = vec![0i32; top_k];
    scratch.topk_idx.copy_to_host(&mut idx_host)?;

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

        let gate_bytes = moe.gate.expert_bytes(gguf, expert)?;
        let up_bytes = moe.up.expert_bytes(gguf, expert)?;
        let down_bytes = moe.down.expert_bytes(gguf, expert)?;
        scratch.stage_gate.copy_prefix_from_host(gate_bytes)?;
        scratch.stage_up.copy_prefix_from_host(up_bytes)?;
        scratch.stage_down.copy_prefix_from_host(down_bytes)?;

        kernels.gemv_quant(
            moe.gate.dtype,
            offset(&scratch.stage_gate, 0),
            offset(&scratch.xn, 0),
            offset(&scratch.ffn_a, 0),
            moe_cfg.expert_ff_len,
            hidden,
        )?;
        kernels.gemv_quant(
            moe.up.dtype,
            offset(&scratch.stage_up, 0),
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
            offset(&scratch.stage_down, 0),
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
    }

    kernels.add_inplace(x_row, offset(&scratch.accum, 0), hidden)
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
