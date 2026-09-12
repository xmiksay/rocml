//! Preallocated per-chunk scratch buffers for the qwen35 hybrid forward
//! pass's chunked-prefill path (issue #6): every buffer holds up to
//! `CHUNK_CAP` tokens' worth of intermediate activations, reused across every
//! chunk and every layer — the batched analogue of `Scratch` (`scratch.rs`),
//! which stays exactly as-is for the still-token-serial decode path.

use rocml_hip::DeviceBuffer;

use super::super::config::Qwen35Config;
use crate::error::RocmlError;

/// Upper bound on tokens processed by one chunked-prefill launch. The
/// `generate` loop picks the actual per-call `chunk_len` (`<= CHUNK_CAP`) by
/// measurement among {128, 256, 512} — see `generate::prefill_chunk_size`.
pub const CHUNK_CAP: u32 = 512;

pub struct ChunkScratch {
    pub x: DeviceBuffer<f32>,
    pub xn: DeviceBuffer<f32>,
    pub token_ids: DeviceBuffer<u32>,
    /// Only the prompt's last token needs logits during prefill (issue #3's
    /// "prefill logits trap") — one row, not `[CHUNK_CAP, vocab]`.
    pub logits: DeviceBuffer<f32>,

    // GDN layer scratch, `[CHUNK_CAP, ...]`.
    pub gdn_qkv: DeviceBuffer<f32>,
    pub gdn_conv_out: DeviceBuffer<f32>,
    pub gdn_z: DeviceBuffer<f32>,
    pub gdn_a_raw: DeviceBuffer<f32>,
    pub gdn_b_raw: DeviceBuffer<f32>,
    pub gdn_beta: DeviceBuffer<f32>,
    pub gdn_g: DeviceBuffer<f32>,
    pub gdn_y: DeviceBuffer<f32>,
    pub gdn_out: DeviceBuffer<f32>,

    // Full-attention layer scratch, `[CHUNK_CAP, ...]`.
    pub attn_q_raw: DeviceBuffer<f32>,
    pub attn_q: DeviceBuffer<f32>,
    pub attn_gate: DeviceBuffer<f32>,
    pub attn_k: DeviceBuffer<f32>,
    pub attn_v: DeviceBuffer<f32>,
    pub attn_concat: DeviceBuffer<f32>,
    pub attn_out: DeviceBuffer<f32>,

    // FFN scratch, shared by both layer kinds, `[CHUNK_CAP, ffn_dim]`.
    pub ffn_gate: DeviceBuffer<f32>,
    pub ffn_up: DeviceBuffer<f32>,
    pub ffn_out: DeviceBuffer<f32>,
}

impl ChunkScratch {
    pub fn new(config: &Qwen35Config) -> Result<Self, RocmlError> {
        let cap = CHUNK_CAP as usize;
        let hidden = config.embedding_length as usize;
        let vocab = config.vocab_size as usize;
        let gdn = &config.gdn;
        let conv_dim = gdn.conv_dim as usize;
        let value_dim = gdn.value_dim as usize;
        let num_v_heads = gdn.num_v_heads as usize;
        let q_dim = config.q_dim() as usize;
        let kv_dim = config.kv_dim() as usize;
        let ffn = config.feed_forward_length as usize;

        Ok(Self {
            x: DeviceBuffer::new(cap * hidden)?,
            xn: DeviceBuffer::new(cap * hidden)?,
            token_ids: DeviceBuffer::new(cap)?,
            logits: DeviceBuffer::new(vocab)?,

            gdn_qkv: DeviceBuffer::new(cap * conv_dim)?,
            gdn_conv_out: DeviceBuffer::new(cap * conv_dim)?,
            gdn_z: DeviceBuffer::new(cap * value_dim)?,
            gdn_a_raw: DeviceBuffer::new(cap * num_v_heads)?,
            gdn_b_raw: DeviceBuffer::new(cap * num_v_heads)?,
            gdn_beta: DeviceBuffer::new(cap * num_v_heads)?,
            gdn_g: DeviceBuffer::new(cap * num_v_heads)?,
            gdn_y: DeviceBuffer::new(cap * value_dim)?,
            gdn_out: DeviceBuffer::new(cap * hidden)?,

            attn_q_raw: DeviceBuffer::new(cap * 2 * q_dim)?,
            attn_q: DeviceBuffer::new(cap * q_dim)?,
            attn_gate: DeviceBuffer::new(cap * q_dim)?,
            attn_k: DeviceBuffer::new(cap * kv_dim)?,
            attn_v: DeviceBuffer::new(cap * kv_dim)?,
            attn_concat: DeviceBuffer::new(cap * q_dim)?,
            attn_out: DeviceBuffer::new(cap * hidden)?,

            ffn_gate: DeviceBuffer::new(cap * ffn)?,
            ffn_up: DeviceBuffer::new(cap * ffn)?,
            ffn_out: DeviceBuffer::new(cap * hidden)?,
        })
    }
}
