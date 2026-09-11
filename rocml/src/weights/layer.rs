//! Per-transformer-block weights.

use rocml_core::gguf::GgufFile;
use rocml_hip::DeviceBuffer;

use super::{load_matrix_f16, load_vector_f32};
use crate::config::ModelConfig;
use crate::error::RocmlError;

pub struct LayerWeights {
    pub attn_norm: DeviceBuffer<f32>,
    /// gemv_f16 shape (m=q_dim, n=hidden).
    pub attn_q: DeviceBuffer<half::f16>,
    /// gemv_f16 shape (m=kv_dim, n=hidden).
    pub attn_k: DeviceBuffer<half::f16>,
    /// gemv_f16 shape (m=kv_dim, n=hidden).
    pub attn_v: DeviceBuffer<half::f16>,
    /// Per-head RMS norm weight, length `head_dim` (shared by every q head).
    pub attn_q_norm: DeviceBuffer<f32>,
    /// Per-head RMS norm weight, length `head_dim` (shared by every kv head).
    pub attn_k_norm: DeviceBuffer<f32>,
    /// gemv_f16 shape (m=hidden, n=q_dim).
    pub attn_output: DeviceBuffer<half::f16>,
    pub ffn_norm: DeviceBuffer<f32>,
    /// gemv_f16 shape (m=feed_forward_length, n=hidden).
    pub ffn_gate: DeviceBuffer<half::f16>,
    /// gemv_f16 shape (m=feed_forward_length, n=hidden).
    pub ffn_up: DeviceBuffer<half::f16>,
    /// gemv_f16 shape (m=hidden, n=feed_forward_length).
    pub ffn_down: DeviceBuffer<half::f16>,
}

impl LayerWeights {
    pub(crate) fn load(
        gguf: &GgufFile,
        config: &ModelConfig,
        layer_idx: u32,
    ) -> Result<Self, RocmlError> {
        let p = format!("blk.{layer_idx}");
        let hidden = config.embedding_length;
        let q_dim = config.q_dim();
        let kv_dim = config.kv_dim();
        let ffn = config.feed_forward_length;
        let head_dim = config.head_dim;

        Ok(Self {
            attn_norm: load_vector_f32(gguf, &format!("{p}.attn_norm.weight"), hidden)?,
            attn_q: load_matrix_f16(gguf, &format!("{p}.attn_q.weight"), q_dim, hidden)?,
            attn_k: load_matrix_f16(gguf, &format!("{p}.attn_k.weight"), kv_dim, hidden)?,
            attn_v: load_matrix_f16(gguf, &format!("{p}.attn_v.weight"), kv_dim, hidden)?,
            attn_q_norm: load_vector_f32(gguf, &format!("{p}.attn_q_norm.weight"), head_dim)?,
            attn_k_norm: load_vector_f32(gguf, &format!("{p}.attn_k_norm.weight"), head_dim)?,
            attn_output: load_matrix_f16(gguf, &format!("{p}.attn_output.weight"), hidden, q_dim)?,
            ffn_norm: load_vector_f32(gguf, &format!("{p}.ffn_norm.weight"), hidden)?,
            ffn_gate: load_matrix_f16(gguf, &format!("{p}.ffn_gate.weight"), ffn, hidden)?,
            ffn_up: load_matrix_f16(gguf, &format!("{p}.ffn_up.weight"), ffn, hidden)?,
            ffn_down: load_matrix_f16(gguf, &format!("{p}.ffn_down.weight"), hidden, ffn)?,
        })
    }
}
