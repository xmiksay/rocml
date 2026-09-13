//! Preallocated per-chunk scratch buffers for the qwen35 hybrid forward
//! pass's chunked-prefill path (issue #6): every buffer holds up to
//! `CHUNK_CAP` tokens' worth of intermediate activations, reused across every
//! chunk and every layer — the batched analogue of `Scratch` (`scratch.rs`),
//! which stays exactly as-is for the still-token-serial decode path.

use rocml_hip::DeviceBuffer;

use super::super::config::Qwen35Config;
use crate::error::RocmlError;
use crate::forward::kernels::{offset, MmqScratch, SplitKScratch};
use crate::forward::kernels_flash::ATTN_PREFILL_FLASH_MAX_SPLITS;
use crate::forward::kernels_splitk::SPLITK_MAX_SPLITS;

/// Upper bound on tokens processed by one chunked-prefill launch. The
/// `generate` loop picks the actual per-call `chunk_len` (`<= CHUNK_CAP`) by
/// measurement among {128, 256, 512} — see `generate::prefill_chunk_size`.
pub const CHUNK_CAP: u32 = 512;

/// Upper bound on one chunkwise-recurrence *tile* (`gdn_chunkwise.rs`
/// sub-chunks any `chunk_len > GDN_RECUR_TILE` into tiles this size, run in
/// sequence, carrying state between them) — capped at 128 because the
/// per-head triangular-inverse kernel needs a whole `tile x tile` f32
/// matrix in LDS, and gfx1101's dynamic shared memory budget is 64KB
/// (`128*128*4 == 65536`). In practice `chunk_len` is always
/// `PREFILL_CHUNK_SIZE` (128) or less already, so this sub-chunking loop
/// never actually iterates more than once — it exists so `forward_chunk`'s
/// documented `1..=CHUNK_CAP` contract stays true rather than silently
/// assuming its one current caller's chunk size forever.
pub const GDN_RECUR_TILE: u32 = 128;

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

    // Chunkwise gated-delta-rule recurrence scratch (`gdn_chunkwise.rs`),
    // sized for one `GDN_RECUR_TILE`-token tile — head-major (see
    // `kernels/gdn_chunkwise.hip`'s module doc for why): `[num_v_heads,
    // tile, head_k_dim]`/`[num_v_heads, tile, head_v_dim]` unless noted.
    pub gdn_cw_q_norm: DeviceBuffer<f32>,
    pub gdn_cw_k_norm: DeviceBuffer<f32>,
    pub gdn_cw_k_beta: DeviceBuffer<f32>,
    pub gdn_cw_g_cum: DeviceBuffer<f32>, // [num_v_heads, tile]
    pub gdn_cw_cum_decay_exp: DeviceBuffer<f32>, // [num_v_heads, tile]
    pub gdn_cw_kb: DeviceBuffer<f32>,    // [num_v_heads, tile, tile] (becomes Tinv in place)
    pub gdn_cw_kq: DeviceBuffer<f32>,    // [num_v_heads, tile, tile]
    pub gdn_cw_v_new: DeviceBuffer<f32>,

    // Full-attention layer scratch, `[CHUNK_CAP, ...]`.
    pub attn_q_raw: DeviceBuffer<f32>,
    pub attn_q: DeviceBuffer<f32>,
    pub attn_gate: DeviceBuffer<f32>,
    pub attn_k: DeviceBuffer<f32>,
    pub attn_v: DeviceBuffer<f32>,
    pub attn_concat: DeviceBuffer<f32>,
    pub attn_out: DeviceBuffer<f32>,

    // `attn_prefill_flash`'s split-K scratch (issue #6 flash-prefill
    // round): `[CHUNK_CAP, n_heads, ATTN_PREFILL_FLASH_MAX_SPLITS,
    // head_dim]` for the per-(row, head, split) online-softmax partial
    // output, `[CHUNK_CAP, n_heads, ATTN_PREFILL_FLASH_MAX_SPLITS]` for its
    // running max/sum — reused across every layer and every call, like the
    // rest of this struct's buffers. Never read back by the caller.
    pub attn_flash_partial_out: DeviceBuffer<f32>,
    pub attn_flash_partial_m: DeviceBuffer<f32>,
    pub attn_flash_partial_l: DeviceBuffer<f32>,

    // FFN scratch, shared by both layer kinds, `[CHUNK_CAP, ffn_dim]`.
    pub ffn_gate: DeviceBuffer<f32>,
    pub ffn_up: DeviceBuffer<f32>,
    pub ffn_out: DeviceBuffer<f32>,

    // int8 MMQ activation-quantization scratch (issue #6's
    // int8-MMQ-integration round, `forward::kernels_mmq`): sized for
    // `CHUNK_CAP` rows x `mmq_dim` columns, where `mmq_dim` is the largest
    // `n` any chunked-prefill `LinearWeight::matmul` call uses (`hidden`
    // for qkv/gate/ffn-gate-up projections, `feed_forward_length` for
    // ffn-down — whichever is bigger). Always allocated regardless of
    // `LoadOptions::use_mmq` (three small buffers; see
    // `forward::kernels_mmq::MmqScratch`'s doc for why keeping every
    // `matmul` call site uniform wins over an `Option`-gated allocation).
    // `quantize_act_q8_blk`'s fixed 32-element block width sizes
    // `mmq_x_scale`/`mmq_x_sum`.
    pub mmq_x_codes: DeviceBuffer<i8>,
    pub mmq_x_scale: DeviceBuffer<f32>,
    pub mmq_x_sum: DeviceBuffer<f32>,

    // Split-K WMMA GEMM partial-sum scratch (issue #6's split-K follow-up,
    // `forward::kernels_splitk`): `[SPLITK_MAX_SPLITS, CHUNK_CAP, hidden]`
    // f32. `hidden` is the widest output any narrow-grid (split-K-candidate)
    // projection ever produces in this architecture (ffn-down, attn-out,
    // ssm_out all project back to `hidden`; the FFN gate/up projections,
    // `m=feed_forward_length`, are always wide-grid already — see
    // `kernels_quant_dispatch.rs`'s `splitk_num_splits` for the `m <=
    // max_m` guard this sizing exists to make safe). Always allocated
    // regardless of whether any call ends up using split-K (same
    // uniform-call-site rationale as `mmq_x_*` above).
    pub gemm_splitk_partial: DeviceBuffer<f32>,
    /// This model's `hidden` — see `gemm_splitk_partial`'s doc comment and
    /// `splitk_scratch`'s `max_m`.
    splitk_max_m: u32,
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
        let head_k_dim = gdn.head_k_dim as usize;
        let head_v_dim = gdn.head_v_dim as usize;
        let tile = GDN_RECUR_TILE as usize;
        let q_dim = config.q_dim() as usize;
        let kv_dim = config.kv_dim() as usize;
        let ffn = config.feed_forward_length as usize;
        let n_heads = config.head_count as usize;
        let head_dim = config.head_dim as usize;
        let max_splits = ATTN_PREFILL_FLASH_MAX_SPLITS as usize;
        // Largest `n` any chunked-prefill `matmul` call uses — see the
        // `mmq_x_*` fields' doc comment. `quantize_act_q8_blk`'s block
        // width (32) must divide it evenly, which holds here since every
        // `LinearWeight::Quant` tensor's `n` is already a multiple of 32
        // (Q8_0) or 256 (Q4_K/Q5_K/Q6_K) by `LinearWeight::load`'s policy,
        // and `hidden`/`ffn` are themselves always multiples of those
        // block widths for any model this loader accepts.
        let mmq_dim = hidden.max(ffn);
        let mmq_blocks = mmq_dim.div_ceil(32);

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

            gdn_cw_q_norm: DeviceBuffer::new(tile * head_k_dim * num_v_heads)?,
            gdn_cw_k_norm: DeviceBuffer::new(tile * head_k_dim * num_v_heads)?,
            gdn_cw_k_beta: DeviceBuffer::new(tile * head_k_dim * num_v_heads)?,
            gdn_cw_g_cum: DeviceBuffer::new(tile * num_v_heads)?,
            gdn_cw_cum_decay_exp: DeviceBuffer::new(tile * num_v_heads)?,
            gdn_cw_kb: DeviceBuffer::new(tile * tile * num_v_heads)?,
            gdn_cw_kq: DeviceBuffer::new(tile * tile * num_v_heads)?,
            gdn_cw_v_new: DeviceBuffer::new(tile * head_v_dim * num_v_heads)?,

            attn_q_raw: DeviceBuffer::new(cap * 2 * q_dim)?,
            attn_q: DeviceBuffer::new(cap * q_dim)?,
            attn_gate: DeviceBuffer::new(cap * q_dim)?,
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

    /// Device pointers into this chunk's MMQ activation-quantization
    /// scratch, for `LinearWeight::matmul`'s `mmq_scratch` argument. Always
    /// safe to pass regardless of whether the call ends up using MMQ (see
    /// the `mmq_x_*` fields' doc comment).
    pub fn mmq_scratch(&self) -> MmqScratch {
        MmqScratch {
            codes: offset(&self.mmq_x_codes, 0),
            scale: offset(&self.mmq_x_scale, 0),
            sum: offset(&self.mmq_x_sum, 0),
        }
    }

    /// Device pointer + capacity for `LinearWeight::matmul`'s
    /// `splitk_scratch` argument. Always safe to pass regardless of whether
    /// the call ends up using split-K (see the `gemm_splitk_partial` field's
    /// doc comment); `max_m` is this model's `hidden` (embedding_length,
    /// captured at `ChunkScratch::new` time), read back by
    /// `kernels_quant_dispatch.rs`'s eligibility check.
    pub fn splitk_scratch(&self) -> SplitKScratch {
        SplitKScratch {
            partial: offset(&self.gemm_splitk_partial, 0),
            max_m: self.splitk_max_m,
        }
    }
}
