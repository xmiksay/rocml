//! `ffn_chunk_dispatch`: routes one layer's chunked-prefill FFN step to the
//! dense batched path or the qwen35moe per-token path. Split out of
//! `chunk_forward.rs` purely for the 400-line file cap.

use super::chunk_scratch::ChunkScratch;
use super::ffn_chunk::ffn_chunk_step;
use super::kernels_moe::MoeKernels;
use super::layer_capture::LayerCapture;
use super::moe::{self, MoeRowCapture};
use super::moe_cache::ExpertCache;
use super::moe_scratch::MoeScratch;
use crate::error::RocmlError;
use crate::forward::kernels::{offset, Kernels};
use crate::profile::Profiler;
use crate::qwen35::config::MoeConfig;
use crate::qwen35::weights::Ffn;
use rocml_core::gguf::GgufFile;
use rocml_hip::DeviceBuffer;

/// Chunked-prefill sibling of `decode_forward::ffn_dispatch` — the dense
/// path batches over the whole chunk via `ffn_chunk_step`; the qwen35moe
/// path (M1) has no batched grouped-GEMM yet (deferred to a later
/// milestone, see `moe::moe_ffn_step`'s module doc), so it just runs
/// `moe::moe_ffn_step` once per row of the chunk. `gguf`/`moe_cfg`/
/// `moe_scratch` are `None` only when every layer's `Ffn` is `Dense` — see
/// `decode_forward::ffn_dispatch`'s doc comment for the same invariant.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ffn_chunk_dispatch(
    kernels: &Kernels,
    moe_kernels: &MoeKernels,
    gguf: Option<&GgufFile>,
    moe_cfg: Option<&MoeConfig>,
    ffn_weights: &Ffn,
    post_attention_norm: &DeviceBuffer<f32>,
    hidden: u32,
    dense_ffn_dim: u32,
    rms_eps: f32,
    scratch: &mut ChunkScratch,
    moe_scratch: Option<&mut MoeScratch>,
    mut expert_cache: Option<&mut ExpertCache>,
    chunk_len: u32,
    prof: Option<&Profiler>,
    layer_idx: Option<u32>,
    mut capture: Option<&mut LayerCapture>,
) -> Result<(), RocmlError> {
    match ffn_weights {
        Ffn::Dense(w) => ffn_chunk_step(
            kernels,
            w,
            post_attention_norm,
            hidden,
            dense_ffn_dim,
            rms_eps,
            scratch,
            chunk_len,
            prof,
            layer_idx,
            capture,
        ),
        Ffn::Moe(w) => {
            let gguf =
                gguf.ok_or_else(|| RocmlError::Config("moe ffn layer with no gguf handle".into()))?;
            let moe_cfg = moe_cfg
                .ok_or_else(|| RocmlError::Config("moe ffn layer with no moe config".into()))?;
            let moe_scratch = moe_scratch
                .ok_or_else(|| RocmlError::Config("moe ffn layer with no moe scratch".into()))?;
            let layer_idx = layer_idx.ok_or_else(|| {
                RocmlError::Config("moe ffn layer with no layer_idx (internal bug)".into())
            })?;
            // "resid_pre_ffn" (llama.cpp's "attn_residual") is the whole
            // chunk's residual stream before this layer's FFN — unlike
            // every other MoE capture point below (computed per-row inside
            // `moe_ffn_step`, since that step has no chunk-wide buffer of
            // its own), `scratch.x` already holds every row contiguously,
            // exactly like the dense path's own one-shot capture in
            // `ffn_chunk_step`.
            if let Some(cap) = capture.as_mut() {
                cap.record(layer_idx, "resid_pre_ffn", &scratch.x, chunk_len, hidden)?;
            }
            for row in 0..chunk_len as usize {
                let x_row = offset(&scratch.x, row * hidden as usize);
                let row_capture = capture.as_deref_mut().map(|cap| MoeRowCapture {
                    capture: cap,
                    row: row as u32,
                    total_rows: chunk_len,
                });
                moe::moe_ffn_step(
                    kernels,
                    moe_kernels,
                    gguf,
                    w,
                    moe_cfg,
                    post_attention_norm,
                    hidden,
                    rms_eps,
                    x_row,
                    moe_scratch,
                    layer_idx,
                    expert_cache.as_deref_mut(),
                    row_capture,
                )?;
            }
            Ok(())
        }
    }
}
