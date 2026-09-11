//! Preallocated per-token scratch buffers, sized once from `ModelConfig` and
//! reused across every token and every layer — decode-style forward passes
//! one token at a time, so nothing here ever needs to grow.

use rocml_hip::DeviceBuffer;

use super::kernels::ATTN_DECODE_MAX_SPLITS;
use crate::config::ModelConfig;
use crate::error::RocmlError;

pub struct Scratch {
    /// The residual stream, updated in place across all layers for the
    /// current token.
    pub x: DeviceBuffer<f32>,
    /// Single-token id, uploaded fresh each step for the embedding lookup.
    pub token_id: DeviceBuffer<u32>,
    /// rmsnorm output, reused for both attn_norm and ffn_norm (never live
    /// across a layer boundary).
    pub xn: DeviceBuffer<f32>,
    pub q: DeviceBuffer<f32>,
    pub k: DeviceBuffer<f32>,
    pub v: DeviceBuffer<f32>,
    /// `attn_decode`'s split-K scratch: unnormalized per-(q head, split)
    /// weighted-V accumulator, `[n_heads, ATTN_DECODE_MAX_SPLITS, head_dim]`.
    pub attn_partial_out: DeviceBuffer<f32>,
    /// `attn_decode`'s per-(q head, split) running max/sum,
    /// `[n_heads, ATTN_DECODE_MAX_SPLITS]` each.
    pub attn_partial_m: DeviceBuffer<f32>,
    pub attn_partial_l: DeviceBuffer<f32>,
    /// Concatenated per-head attention output, `[n_heads, head_dim]`.
    pub attn_concat: DeviceBuffer<f32>,
    /// attn_output projection result, `[hidden]`.
    pub attn_out: DeviceBuffer<f32>,
    pub gate: DeviceBuffer<f32>,
    pub up: DeviceBuffer<f32>,
    /// ffn_down projection result, `[hidden]`.
    pub ffn_out: DeviceBuffer<f32>,
    pub logits: DeviceBuffer<f32>,
}

impl Scratch {
    pub fn new(config: &ModelConfig) -> Result<Self, RocmlError> {
        let hidden = config.embedding_length as usize;
        let q_dim = config.q_dim() as usize;
        let kv_dim = config.kv_dim() as usize;
        let ffn = config.feed_forward_length as usize;
        let n_heads = config.head_count as usize;
        let head_dim = config.head_dim as usize;
        let max_splits = ATTN_DECODE_MAX_SPLITS as usize;

        Ok(Self {
            x: DeviceBuffer::new(hidden)?,
            token_id: DeviceBuffer::new(1)?,
            xn: DeviceBuffer::new(hidden)?,
            q: DeviceBuffer::new(q_dim)?,
            k: DeviceBuffer::new(kv_dim)?,
            v: DeviceBuffer::new(kv_dim)?,
            attn_partial_out: DeviceBuffer::new(n_heads * max_splits * head_dim)?,
            attn_partial_m: DeviceBuffer::new(n_heads * max_splits)?,
            attn_partial_l: DeviceBuffer::new(n_heads * max_splits)?,
            attn_concat: DeviceBuffer::new(q_dim)?,
            attn_out: DeviceBuffer::new(hidden)?,
            gate: DeviceBuffer::new(ffn)?,
            up: DeviceBuffer::new(ffn)?,
            ffn_out: DeviceBuffer::new(hidden)?,
            logits: DeviceBuffer::new(config.vocab_size as usize)?,
        })
    }
}
