//! Preallocated per-chunk scratch buffers for the dense Qwen3 architecture's
//! chunked-prefill path (issue #16 — the dense-architecture chunked-prefill
//! port; see `chunk_forward.rs`). Every buffer holds up to `CHUNK_CAP`
//! tokens' worth of intermediate activations, reused across every chunk and
//! every layer — the batched analogue of `Scratch` (`scratch.rs`), which
//! stays exactly as-is for the still-token-serial decode path.
//!
//! Deliberately smaller than `qwen35::forward::chunk_scratch::ChunkScratch`:
//! the dense architecture has no GDN layers (no `gdn_*` fields) and no
//! fused Q+output-gate projection (no `attn_q_raw`/`attn_gate` — a dense
//! `attn_q` projection's output is already `[chunk_len, n_heads, head_dim]`,
//! the exact shape the per-head norm/rope calls need, so no `extract_heads`
//! step is needed either).

use rocml_hip::DeviceBuffer;

use super::kernels::{offset, MmqScratch, SplitKScratch};
use super::kernels_flash::ATTN_PREFILL_FLASH_MAX_SPLITS;
use super::kernels_splitk::SPLITK_MAX_SPLITS;
use crate::config::ModelConfig;
use crate::error::RocmlError;

/// Upper bound on tokens processed by one chunked-prefill launch. Matches
/// the qwen35 hybrid path's `PREFILL_CHUNK_SIZE`/`CHUNK_CAP` (512) — every
/// batched kernel this touches (`gemm_xwt_wmma_q*`'s internal 128-row tiles,
/// the flash-prefill split-K design) is already sized/tuned for that chunk
/// width, and reusing the same constant keeps the two architectures'
/// chunking behavior easy to compare.
pub const CHUNK_CAP: u32 = 512;

pub struct ChunkScratch {
    pub x: DeviceBuffer<f32>,
    pub xn: DeviceBuffer<f32>,
    pub token_ids: DeviceBuffer<u32>,
    /// Only the prompt's last token needs logits during prefill (issue #3's
    /// "prefill logits trap") — one row, not `[CHUNK_CAP, vocab]`.
    pub logits: DeviceBuffer<f32>,

    pub attn_q: DeviceBuffer<f32>,
    pub attn_k: DeviceBuffer<f32>,
    pub attn_v: DeviceBuffer<f32>,
    pub attn_concat: DeviceBuffer<f32>,
    pub attn_out: DeviceBuffer<f32>,

    /// `attn_prefill_flash`'s split-K scratch — see
    /// `qwen35::forward::chunk_scratch::ChunkScratch`'s identical fields for
    /// the shape rationale.
    pub attn_flash_partial_out: DeviceBuffer<f32>,
    pub attn_flash_partial_m: DeviceBuffer<f32>,
    pub attn_flash_partial_l: DeviceBuffer<f32>,

    pub ffn_gate: DeviceBuffer<f32>,
    pub ffn_up: DeviceBuffer<f32>,
    pub ffn_out: DeviceBuffer<f32>,

    // int8 MMQ activation-quantization scratch — always allocated regardless
    // of `LoadOptions::use_mmq`, same uniform-call-site rationale as the
    // hybrid path's identical fields (`qwen35::forward::chunk_scratch`).
    pub mmq_x_codes: DeviceBuffer<i8>,
    pub mmq_x_scale: DeviceBuffer<f32>,
    pub mmq_x_sum: DeviceBuffer<f32>,

    // Split-K WMMA GEMM partial-sum scratch — see the hybrid path's
    // identical fields for the shape rationale.
    pub gemm_splitk_partial: DeviceBuffer<f32>,
    splitk_max_m: u32,
}

impl ChunkScratch {
    pub fn new(config: &ModelConfig) -> Result<Self, RocmlError> {
        let cap = CHUNK_CAP as usize;
        let hidden = config.embedding_length as usize;
        let vocab = config.vocab_size as usize;
        let q_dim = config.q_dim() as usize;
        let kv_dim = config.kv_dim() as usize;
        let ffn = config.feed_forward_length as usize;
        let n_heads = config.head_count as usize;
        let head_dim = config.head_dim as usize;
        let max_splits = ATTN_PREFILL_FLASH_MAX_SPLITS as usize;
        let mmq_dim = hidden.max(ffn);
        let mmq_blocks = mmq_dim.div_ceil(32);

        Ok(Self {
            x: DeviceBuffer::new(cap * hidden)?,
            xn: DeviceBuffer::new(cap * hidden)?,
            token_ids: DeviceBuffer::new(cap)?,
            logits: DeviceBuffer::new(vocab)?,

            attn_q: DeviceBuffer::new(cap * q_dim)?,
            attn_k: DeviceBuffer::new(cap * kv_dim)?,
            attn_v: DeviceBuffer::new(cap * kv_dim)?,
            attn_concat: DeviceBuffer::new(cap * q_dim)?,
            attn_out: DeviceBuffer::new(cap * hidden)?,

            attn_flash_partial_out: DeviceBuffer::new(cap * n_heads * max_splits * head_dim)?,
            attn_flash_partial_m: DeviceBuffer::new(cap * n_heads * max_splits)?,
            attn_flash_partial_l: DeviceBuffer::new(cap * n_heads * max_splits)?,

            ffn_gate: DeviceBuffer::new(cap * ffn)?,
            ffn_up: DeviceBuffer::new(cap * ffn)?,
            ffn_out: DeviceBuffer::new(cap * hidden)?,

            mmq_x_codes: DeviceBuffer::new(cap * mmq_dim)?,
            mmq_x_scale: DeviceBuffer::new(cap * mmq_blocks)?,
            mmq_x_sum: DeviceBuffer::new(cap * mmq_blocks)?,

            gemm_splitk_partial: DeviceBuffer::new(SPLITK_MAX_SPLITS as usize * cap * hidden)?,
            splitk_max_m: hidden as u32,
        })
    }

    /// See `qwen35::forward::chunk_scratch::ChunkScratch::mmq_scratch`'s
    /// doc — identical contract.
    pub fn mmq_scratch(&self) -> MmqScratch {
        MmqScratch {
            codes: offset(&self.mmq_x_codes, 0),
            scale: offset(&self.mmq_x_scale, 0),
            sum: offset(&self.mmq_x_sum, 0),
        }
    }

    /// See `qwen35::forward::chunk_scratch::ChunkScratch::splitk_scratch`'s
    /// doc — identical contract.
    pub fn splitk_scratch(&self) -> SplitKScratch {
        SplitKScratch {
            partial: offset(&self.gemm_splitk_partial, 0),
            max_m: self.splitk_max_m,
        }
    }
}
