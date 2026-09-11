//! One full-attention layer's weights (every `full_attention_interval`-th
//! qwen35 block).

use rocml_core::gguf::GgufFile;
use rocml_hip::DeviceBuffer;

use super::ffn::FfnWeights;
use crate::error::RocmlError;
use crate::qwen35::config::Qwen35Config;
use crate::weights::{load_vector_f32, LinearWeight};

pub struct AttnLayerWeights {
    pub attn_norm: DeviceBuffer<f32>,
    /// (m=q_out, n=hidden). `q_out` is `2 * q_dim` when `has_output_gate`
    /// (Qwen3.5 always sets this): each head's row is `[query(head_dim) |
    /// gate(head_dim)]`, not a flat `[Q | gate]` split — see `attention.rs`'s
    /// per-head extraction.
    pub attn_q: LinearWeight,
    pub attn_q_norm: DeviceBuffer<f32>,
    /// (m=kv_dim, n=hidden).
    pub attn_k: LinearWeight,
    pub attn_k_norm: DeviceBuffer<f32>,
    /// (m=kv_dim, n=hidden).
    pub attn_v: LinearWeight,
    /// (m=hidden, n=q_dim).
    pub attn_output: LinearWeight,
    pub post_attention_norm: DeviceBuffer<f32>,
    pub ffn: FfnWeights,
    pub has_output_gate: bool,
}

impl AttnLayerWeights {
    pub(crate) fn load(
        gguf: &GgufFile,
        cfg: &Qwen35Config,
        layer_idx: u32,
    ) -> Result<Self, RocmlError> {
        let p = format!("blk.{layer_idx}");
        let hidden = cfg.embedding_length;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();

        // `shape()` is GGUF "ne" order (ne[0] innermost/n, ne[1] outermost/m)
        // — index 1 is the row count `matrix_dims`/`load_matrix_f16` treat
        // as `m`, not index 0.
        let q_name = format!("{p}.attn_q.weight");
        let q_rows = gguf
            .tensor(&q_name)?
            .shape()
            .get(1)
            .copied()
            .ok_or_else(|| {
                RocmlError::Config(format!(
                    "tensor {q_name:?}: must be 2D ([hidden, q_out] in ne order)"
                ))
            })?;
        let has_output_gate = match u32::try_from(q_rows) {
            Ok(rows) if rows == q_dim => false,
            Ok(rows) if rows == 2 * q_dim => true,
            Ok(rows) => {
                return Err(RocmlError::Config(format!(
                    "tensor {q_name:?}: {rows} rows matches neither q_dim {q_dim} nor \
                     2*q_dim {}",
                    2 * q_dim
                )))
            }
            Err(_) => {
                return Err(RocmlError::Config(format!(
                    "tensor {q_name:?}: row count {q_rows} overflows u32"
                )))
            }
        };
        let q_out = if has_output_gate { 2 * q_dim } else { q_dim };

        Ok(Self {
            attn_norm: load_vector_f32(gguf, &format!("{p}.attn_norm.weight"), hidden)?,
            attn_q: LinearWeight::load(gguf, &q_name, q_out, hidden)?,
            attn_q_norm: load_vector_f32(gguf, &format!("{p}.attn_q_norm.weight"), cfg.head_dim)?,
            attn_k: LinearWeight::load(gguf, &format!("{p}.attn_k.weight"), kv_dim, hidden)?,
            attn_k_norm: load_vector_f32(gguf, &format!("{p}.attn_k_norm.weight"), cfg.head_dim)?,
            attn_v: LinearWeight::load(gguf, &format!("{p}.attn_v.weight"), kv_dim, hidden)?,
            attn_output: LinearWeight::load(
                gguf,
                &format!("{p}.attn_output.weight"),
                hidden,
                q_dim,
            )?,
            post_attention_norm: load_vector_f32(
                gguf,
                &format!("{p}.post_attention_norm.weight"),
                hidden,
            )?,
            ffn: FfnWeights::load(gguf, &p, hidden, cfg.feed_forward_length)?,
            has_output_gate,
        })
    }
}
