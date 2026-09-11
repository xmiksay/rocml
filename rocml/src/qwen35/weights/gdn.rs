//! One Gated Delta Net (linear-attention) layer's weights.

use rocml_core::gguf::GgufFile;
use rocml_hip::DeviceBuffer;

use super::ffn::FfnWeights;
use crate::error::RocmlError;
use crate::qwen35::config::{GdnConfig, Qwen35Config};
use crate::weights::{load_matrix_f32, load_vector_f32, LinearWeight};

pub struct GdnLayerWeights {
    pub attn_norm: DeviceBuffer<f32>,
    /// (m=conv_dim, n=hidden): fused `[Q|K|V]` projection.
    pub attn_qkv: LinearWeight,
    /// (m=value_dim, n=hidden): the output silu-gate `Z`.
    pub attn_gate: LinearWeight,
    /// (m=num_v_heads, n=hidden): write-strength logit `B`.
    pub ssm_beta: LinearWeight,
    /// (m=num_v_heads, n=hidden): decay logit `A`.
    pub ssm_alpha: LinearWeight,
    /// gemv_f32 shape (m=conv_dim, n=conv_kernel): per-channel causal conv
    /// taps, oldest tap first. Kept f32 (small, feeds the recurrence's
    /// numerically sensitive gate path, matches Crane never quantizing it
    /// further than the GGUF's own storage).
    pub ssm_conv1d: DeviceBuffer<f32>,
    pub ssm_dt_bias: DeviceBuffer<f32>,
    /// `A_log`, one scalar per head.
    pub ssm_a: DeviceBuffer<f32>,
    /// Gated-RMSNorm weight, length `head_v_dim`.
    pub ssm_norm: DeviceBuffer<f32>,
    /// (m=hidden, n=value_dim).
    pub ssm_out: LinearWeight,
    pub post_attention_norm: DeviceBuffer<f32>,
    pub ffn: FfnWeights,
}

impl GdnLayerWeights {
    pub(crate) fn load(
        gguf: &GgufFile,
        cfg: &Qwen35Config,
        layer_idx: u32,
    ) -> Result<Self, RocmlError> {
        let p = format!("blk.{layer_idx}");
        let hidden = cfg.embedding_length;
        let gdn: &GdnConfig = &cfg.gdn;

        Ok(Self {
            attn_norm: load_vector_f32(gguf, &format!("{p}.attn_norm.weight"), hidden)?,
            attn_qkv: LinearWeight::load(
                gguf,
                &format!("{p}.attn_qkv.weight"),
                gdn.conv_dim,
                hidden,
            )?,
            attn_gate: LinearWeight::load(
                gguf,
                &format!("{p}.attn_gate.weight"),
                gdn.value_dim,
                hidden,
            )?,
            ssm_beta: LinearWeight::load(
                gguf,
                &format!("{p}.ssm_beta.weight"),
                gdn.num_v_heads,
                hidden,
            )?,
            ssm_alpha: LinearWeight::load(
                gguf,
                &format!("{p}.ssm_alpha.weight"),
                gdn.num_v_heads,
                hidden,
            )?,
            ssm_conv1d: load_matrix_f32(
                gguf,
                &format!("{p}.ssm_conv1d.weight"),
                gdn.conv_dim,
                gdn.conv_kernel,
            )?,
            ssm_dt_bias: load_vector_f32(gguf, &format!("{p}.ssm_dt.bias"), gdn.num_v_heads)?,
            ssm_a: load_vector_f32(gguf, &format!("{p}.ssm_a"), gdn.num_v_heads)?,
            ssm_norm: load_vector_f32(gguf, &format!("{p}.ssm_norm.weight"), gdn.head_v_dim)?,
            ssm_out: LinearWeight::load(
                gguf,
                &format!("{p}.ssm_out.weight"),
                hidden,
                gdn.value_dim,
            )?,
            post_attention_norm: load_vector_f32(
                gguf,
                &format!("{p}.post_attention_norm.weight"),
                hidden,
            )?,
            ffn: FfnWeights::load(gguf, &p, hidden, cfg.feed_forward_length)?,
        })
    }
}
