//! Preallocated per-token scratch buffers for the qwen35 hybrid forward
//! pass, sized once from `Qwen35Config` and reused across every token and
//! every layer (decode-style forward, one token at a time — see
//! `crate::forward::scratch` for the dense analogue this mirrors).

use rocml_hip::DeviceBuffer;

use super::super::config::Qwen35Config;
use crate::error::RocmlError;

pub struct Scratch {
    pub x: DeviceBuffer<f32>,
    pub token_id: DeviceBuffer<u32>,
    pub xn: DeviceBuffer<f32>,
    pub logits: DeviceBuffer<f32>,

    // GDN layer scratch.
    /// Raw `attn_qkv` projection output, `[conv_dim]` = `[Q|K|V]`.
    pub gdn_qkv: DeviceBuffer<f32>,
    /// Post causal-conv1d + SiLU, `[conv_dim]` = `[Q|K|V]`; Q/K are then
    /// L2-normalized (and Q additionally recurrence-scaled) in place.
    pub gdn_conv_out: DeviceBuffer<f32>,
    /// Output silu-gate `Z`, `[value_dim]`.
    pub gdn_z: DeviceBuffer<f32>,
    /// Raw `A`/`B` gate-projection logits and the derived beta/g, each
    /// `[num_v_heads]`.
    pub gdn_a_raw: DeviceBuffer<f32>,
    pub gdn_b_raw: DeviceBuffer<f32>,
    pub gdn_beta: DeviceBuffer<f32>,
    pub gdn_g: DeviceBuffer<f32>,
    /// Recurrence readout / gated-norm output, `[value_dim]`.
    pub gdn_y: DeviceBuffer<f32>,
    /// `ssm_out` projection result, `[hidden]`.
    pub gdn_out: DeviceBuffer<f32>,
    /// Constant `1/sqrt(head_k_dim)` vector, `[head_k_dim]` — the "weight"
    /// that turns `rmsnorm_f32` into a plain L2 norm for K (see
    /// `forward::gdn`'s doc comment for the derivation).
    pub l2_alpha_k: DeviceBuffer<f32>,
    /// Constant `1/head_k_dim` vector, `[head_k_dim]` — same trick, but
    /// folding in the recurrence's extra `1/sqrt(head_k_dim)` query scale.
    pub l2_alpha_q: DeviceBuffer<f32>,

    // Full-attention layer scratch.
    /// Raw `attn_q` projection output, `[2 * q_dim]` when gated (per-head
    /// `[query(head_dim) | gate(head_dim)]`, see `forward::attention`).
    pub attn_q_raw: DeviceBuffer<f32>,
    /// Query and output-gate, extracted from `attn_q_raw` into compact
    /// `[q_dim]` per-head-contiguous layouts.
    pub attn_q: DeviceBuffer<f32>,
    pub attn_gate: DeviceBuffer<f32>,
    pub attn_k: DeviceBuffer<f32>,
    pub attn_v: DeviceBuffer<f32>,
    /// `[n_heads, max_seq]` raw/softmaxed attention scores.
    pub attn_scores: DeviceBuffer<f32>,
    pub attn_valid_len: DeviceBuffer<u32>,
    /// Concatenated per-head attention output, `[q_dim]`.
    pub attn_concat: DeviceBuffer<f32>,
    /// `attn_output` projection result, `[hidden]`.
    pub attn_out: DeviceBuffer<f32>,

    // FFN scratch, shared by both layer kinds.
    pub ffn_gate: DeviceBuffer<f32>,
    pub ffn_up: DeviceBuffer<f32>,
    pub ffn_out: DeviceBuffer<f32>,
}

impl Scratch {
    pub fn new(config: &Qwen35Config, max_seq: u32) -> Result<Self, RocmlError> {
        let hidden = config.embedding_length as usize;
        let vocab = config.vocab_size as usize;
        let gdn = &config.gdn;
        let conv_dim = gdn.conv_dim as usize;
        let value_dim = gdn.value_dim as usize;
        let num_v_heads = gdn.num_v_heads as usize;
        let head_k_dim = gdn.head_k_dim as usize;
        let q_dim = config.q_dim() as usize;
        let kv_dim = config.kv_dim() as usize;
        let n_heads = config.head_count as usize;
        let ffn = config.feed_forward_length as usize;

        let mut l2_alpha_k = DeviceBuffer::new(head_k_dim)?;
        let mut l2_alpha_q = DeviceBuffer::new(head_k_dim)?;
        let inv_sqrt = 1.0 / (head_k_dim as f32).sqrt();
        l2_alpha_k.copy_from_host(&vec![inv_sqrt; head_k_dim])?;
        l2_alpha_q.copy_from_host(&vec![inv_sqrt * inv_sqrt; head_k_dim])?;

        Ok(Self {
            x: DeviceBuffer::new(hidden)?,
            token_id: DeviceBuffer::new(1)?,
            xn: DeviceBuffer::new(hidden)?,
            logits: DeviceBuffer::new(vocab)?,

            gdn_qkv: DeviceBuffer::new(conv_dim)?,
            gdn_conv_out: DeviceBuffer::new(conv_dim)?,
            gdn_z: DeviceBuffer::new(value_dim)?,
            gdn_a_raw: DeviceBuffer::new(num_v_heads)?,
            gdn_b_raw: DeviceBuffer::new(num_v_heads)?,
            gdn_beta: DeviceBuffer::new(num_v_heads)?,
            gdn_g: DeviceBuffer::new(num_v_heads)?,
            gdn_y: DeviceBuffer::new(value_dim)?,
            gdn_out: DeviceBuffer::new(hidden)?,
            l2_alpha_k,
            l2_alpha_q,

            attn_q_raw: DeviceBuffer::new(2 * q_dim)?,
            attn_q: DeviceBuffer::new(q_dim)?,
            attn_gate: DeviceBuffer::new(q_dim)?,
            attn_k: DeviceBuffer::new(kv_dim)?,
            attn_v: DeviceBuffer::new(kv_dim)?,
            attn_scores: DeviceBuffer::new(n_heads * max_seq as usize)?,
            attn_valid_len: DeviceBuffer::new(n_heads)?,
            attn_concat: DeviceBuffer::new(q_dim)?,
            attn_out: DeviceBuffer::new(hidden)?,

            ffn_gate: DeviceBuffer::new(ffn)?,
            ffn_up: DeviceBuffer::new(ffn)?,
            ffn_out: DeviceBuffer::new(hidden)?,
        })
    }
}
