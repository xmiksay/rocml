//! Preallocated scratch for the qwen35moe chunked-prefill grouped-by-expert
//! FFN step (M3, `super::moe_chunk`). Sibling of `super::moe_scratch`'s
//! single-row `MoeScratch` (still used by decode and, when
//! `LayerCapture`-diagnosing, by chunked prefill's own per-row fallback —
//! see `super::moe_chunk`'s module doc), but every buffer here holds up to
//! `CHUNK_CAP` rows: the router/shared-expert pieces process the whole
//! chunk in one batched call each, and the `group_*` buffers are reused
//! across every distinct expert a chunk touches (worst case one expert
//! claims every row in the chunk, hence sized to `CHUNK_CAP` rather than the
//! model's average rows-per-expert).

use rocml_hip::DeviceBuffer;

use super::super::config::{MoeConfig, Qwen35Config};
use super::chunk_scratch::CHUNK_CAP;
use crate::error::RocmlError;

pub struct MoeChunkScratch {
    // Router + shared-expert, batched over the whole chunk.
    pub router_logits: DeviceBuffer<f32>, // [CHUNK_CAP, expert_count]
    pub topk_idx: DeviceBuffer<i32>,      // [CHUNK_CAP, top_k]
    pub topk_weight: DeviceBuffer<f32>,   // [CHUNK_CAP, top_k]
    pub shared_gate_logit: DeviceBuffer<f32>, // [CHUNK_CAP]
    pub shared_a: DeviceBuffer<f32>,      // [CHUNK_CAP, shared_ff_len]
    pub shared_b: DeviceBuffer<f32>,      // [CHUNK_CAP, shared_ff_len]
    pub shared_out: DeviceBuffer<f32>,    // [CHUNK_CAP, hidden]
    /// The FFN step's chunk-wide output accumulator — the shared expert's
    /// gated contribution is always the first write (see
    /// `super::moe_chunk`), each expert group's weighted contribution
    /// scatter-accumulates on top.
    pub accum: DeviceBuffer<f32>, // [CHUNK_CAP, hidden]

    // Reused across every distinct expert one chunk/layer touches.
    pub group_row_idx: DeviceBuffer<u32>, // [CHUNK_CAP]
    pub group_weight: DeviceBuffer<f32>,  // [CHUNK_CAP]
    pub group_x: DeviceBuffer<f32>,       // [CHUNK_CAP, hidden]
    pub group_gate: DeviceBuffer<f32>,    // [CHUNK_CAP, expert_ff_len]
    pub group_up: DeviceBuffer<f32>,      // [CHUNK_CAP, expert_ff_len]
    pub group_down: DeviceBuffer<f32>,    // [CHUNK_CAP, hidden]
}

impl MoeChunkScratch {
    pub fn new(cfg: &Qwen35Config, moe_cfg: &MoeConfig) -> Result<Self, RocmlError> {
        let cap = CHUNK_CAP as usize;
        let hidden = cfg.embedding_length as usize;
        let expert_count = moe_cfg.expert_count as usize;
        let top_k = moe_cfg.expert_used_count as usize;
        let shared_ff = moe_cfg.shared_ff_len as usize;
        let expert_ff = moe_cfg.expert_ff_len as usize;

        Ok(Self {
            router_logits: DeviceBuffer::new(cap * expert_count)?,
            topk_idx: DeviceBuffer::new(cap * top_k)?,
            topk_weight: DeviceBuffer::new(cap * top_k)?,
            shared_gate_logit: DeviceBuffer::new(cap)?,
            shared_a: DeviceBuffer::new(cap * shared_ff)?,
            shared_b: DeviceBuffer::new(cap * shared_ff)?,
            shared_out: DeviceBuffer::new(cap * hidden)?,
            accum: DeviceBuffer::new(cap * hidden)?,
            group_row_idx: DeviceBuffer::new(cap)?,
            group_weight: DeviceBuffer::new(cap)?,
            group_x: DeviceBuffer::new(cap * hidden)?,
            group_gate: DeviceBuffer::new(cap * expert_ff)?,
            group_up: DeviceBuffer::new(cap * expert_ff)?,
            group_down: DeviceBuffer::new(cap * hidden)?,
        })
    }
}
