//! One Gated Delta Net (linear-attention) layer's decode step: input
//! projections, causal conv1d + SiLU, per-head Q/K L2-norm, the beta/decay
//! gates, the gated delta-rule recurrence, gated RMSNorm, and the output
//! projection. Mirrors Crane's `GatedDeltaNet::forward`
//! (crane-core/src/ops/gdn/layer.rs) for a single timestep.
//!
//! `rmsnorm_f32(x, weight, out, rows, n, eps)` computes
//! `x / sqrt(mean(x^2) + eps) * weight`. Setting `weight` to the constant
//! `alpha = 1/sqrt(n)` and `eps' = eps/n` gives
//! `x / sqrt(sum(x^2) + eps) * alpha / alpha = x / sqrt(sum(x^2) + eps)`
//! — an exact L2 norm, not an approximation (same algebra Crane's
//! `l2_norm_fused` uses to reach the same kernel). `Scratch::l2_alpha_k`
//! carries this `alpha` for K; `Scratch::l2_alpha_q` additionally folds in
//! the recurrence's own `1/sqrt(head_k_dim)` query scale (`alpha^2` instead
//! of `alpha`), since `rmsnorm_f32`'s per-element weight multiply composes
//! with that scale for free.
//!
//! `L2_NORM_EPS` is Crane's hardcoded `1e-6` (crane-core/src/ops/gdn/backend.rs's
//! `l2_norm_fused` calls), independent of the model's own `rms_eps`.

use super::kernels::HybridKernels;
use super::scratch::Scratch;
use crate::error::RocmlError;
use crate::forward::kernels::{offset, Kernels};
use crate::qwen35::cache::GdnLayerState;
use crate::qwen35::config::{GdnConfig, Qwen35Config};
use crate::qwen35::weights::GdnLayerWeights;

const L2_NORM_EPS: f64 = 1e-6;

#[allow(clippy::too_many_arguments)]
pub(crate) fn gdn_layer_step(
    kernels: &Kernels,
    hybrid: &HybridKernels,
    config: &Qwen35Config,
    layer: &GdnLayerWeights,
    state: &mut GdnLayerState,
    scratch: &mut Scratch,
) -> Result<(), RocmlError> {
    let hidden = config.embedding_length;
    let gdn: &GdnConfig = &config.gdn;
    let l2_eps = (L2_NORM_EPS / gdn.head_k_dim as f64) as f32;

    kernels.rmsnorm(
        offset(&scratch.x, 0),
        offset(&layer.attn_norm, 0),
        offset(&scratch.xn, 0),
        1,
        hidden,
        config.rms_eps,
    )?;

    layer.attn_qkv.matvec(
        kernels,
        offset(&scratch.xn, 0),
        offset(&scratch.gdn_qkv, 0),
        gdn.conv_dim,
        hidden,
    )?;
    layer.attn_gate.matvec(
        kernels,
        offset(&scratch.xn, 0),
        offset(&scratch.gdn_z, 0),
        gdn.value_dim,
        hidden,
    )?;
    layer.ssm_alpha.matvec(
        kernels,
        offset(&scratch.xn, 0),
        offset(&scratch.gdn_a_raw, 0),
        gdn.num_v_heads,
        hidden,
    )?;
    layer.ssm_beta.matvec(
        kernels,
        offset(&scratch.xn, 0),
        offset(&scratch.gdn_b_raw, 0),
        gdn.num_v_heads,
        hidden,
    )?;

    hybrid.gdn_conv1d_decode(
        offset(&scratch.gdn_qkv, 0),
        offset(&state.conv_state, 0),
        offset(&layer.ssm_conv1d, 0),
        offset(&scratch.gdn_conv_out, 0),
        gdn.conv_dim,
        gdn.conv_kernel,
    )?;
    hybrid.gdn_gate(
        offset(&scratch.gdn_a_raw, 0),
        offset(&scratch.gdn_b_raw, 0),
        offset(&layer.ssm_a, 0),
        offset(&layer.ssm_dt_bias, 0),
        offset(&scratch.gdn_beta, 0),
        offset(&scratch.gdn_g, 0),
        gdn.num_v_heads,
    )?;

    // Per-head L2-norm Q and K in place (V is untouched); Q additionally
    // carries the recurrence's query scale via `l2_alpha_q` (see module doc).
    let q_off = offset(&scratch.gdn_conv_out, 0);
    kernels.rmsnorm(
        q_off,
        offset(&scratch.l2_alpha_q, 0),
        q_off,
        gdn.num_k_heads,
        gdn.head_k_dim,
        l2_eps,
    )?;
    let k_off = offset(&scratch.gdn_conv_out, gdn.key_dim as usize);
    kernels.rmsnorm(
        k_off,
        offset(&scratch.l2_alpha_k, 0),
        k_off,
        gdn.num_k_heads,
        gdn.head_k_dim,
        l2_eps,
    )?;
    let v_off = offset(&scratch.gdn_conv_out, 2 * gdn.key_dim as usize);

    hybrid.gdn_recurrence_decode(
        offset(&state.state, 0),
        q_off,
        k_off,
        v_off,
        offset(&scratch.gdn_beta, 0),
        offset(&scratch.gdn_g, 0),
        offset(&scratch.gdn_y, 0),
        gdn.num_v_heads,
        gdn.num_k_heads,
        gdn.head_k_dim,
        gdn.head_v_dim,
    )?;

    let y_off = offset(&scratch.gdn_y, 0);
    kernels.rmsnorm(
        y_off,
        offset(&layer.ssm_norm, 0),
        y_off,
        gdn.num_v_heads,
        gdn.head_v_dim,
        config.rms_eps,
    )?;
    kernels.silu_mul(offset(&scratch.gdn_z, 0), y_off, y_off, gdn.value_dim)?;

    layer.ssm_out.matvec(
        kernels,
        offset(&scratch.gdn_y, 0),
        offset(&scratch.gdn_out, 0),
        hidden,
        gdn.value_dim,
    )?;
    kernels.add_inplace(offset(&scratch.x, 0), offset(&scratch.gdn_out, 0), hidden)?;

    Ok(())
}
