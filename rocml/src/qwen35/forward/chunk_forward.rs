//! `Model::forward_chunk`: processes a whole prefill chunk of tokens through
//! the batched kernel paths (`attention_chunk`/`gdn_chunk`/`ffn_chunk`)
//! instead of `forward_token_profiled`'s one-token-at-a-time loop. See issue
//! #6's design and `generate::generate_core`'s caller for how a prompt gets
//! split into chunks.

use super::attention_chunk::attention_chunk_step;
use super::chunk_scratch::CHUNK_CAP;
use super::ffn_chunk::ffn_chunk_step;
use super::gdn_chunk::gdn_chunk_step;
use super::Model;
use crate::error::RocmlError;
use crate::forward::kernels::offset;
use crate::profile::{self, OpKind, Profiler};
use crate::qwen35::config::LayerKind;
use crate::qwen35::weights::LayerWeights;

/// Prompt-chunk size `generate` drives the hybrid forward pass with, chosen
/// by measurement among {128, 256, 512} on Qwen3.5-2B `bench --depth 2048`
/// (see issue #6's report): 246/241/239 tok/s respectively — a small but
/// consistent edge for the smallest chunk, since this GEMM/GDN-chunk-kernel
/// implementation isn't yet compute-bound enough at these depths for a
/// bigger chunk's better per-launch amortization to outweigh its slightly
/// larger constant-ish per-token kernel overhead.
pub const PREFILL_CHUNK_SIZE: u32 = 128;

impl Model {
    /// Splits `prompt_ids` (non-empty) into `PREFILL_CHUNK_SIZE`-token
    /// chunks and runs each through [`Self::forward_chunk`], requesting
    /// logits only from the final chunk. Advances `self.position()` by
    /// `prompt_ids.len()` overall, identically to `prompt_ids.len()` calls
    /// to `forward_token_profiled` (see `tests/qwen35_chunked_prefill_parity.rs`
    /// for the equivalence this is held to).
    pub fn forward_prompt_chunked(
        &mut self,
        prompt_ids: &[u32],
        prof: Option<&Profiler>,
    ) -> Result<Vec<f32>, RocmlError> {
        if prompt_ids.is_empty() {
            return Ok(Vec::new());
        }
        let chunk_size = (PREFILL_CHUNK_SIZE.min(CHUNK_CAP)) as usize;
        let mut logits = Vec::new();
        let mut i = 0;
        while i < prompt_ids.len() {
            let end = (i + chunk_size).min(prompt_ids.len());
            let is_last_chunk = end == prompt_ids.len();
            if let Some(l) = self.forward_chunk(&prompt_ids[i..end], is_last_chunk, prof)? {
                logits = l;
            }
            i = end;
        }
        Ok(logits)
    }

    /// Processes `token_ids` (`1..=CHUNK_CAP` tokens, positions
    /// `self.position()..self.position()+token_ids.len()`) through every
    /// layer in one batched pass each, advancing the cache/GDN state exactly
    /// as `token_ids.len()` calls to `forward_token_profiled` would. Returns
    /// logits for the chunk's last token only when `want_logits` is set
    /// (issue #3's "prefill logits trap": a full `[chunk_len, vocab]`
    /// intermediate is never materialized — the final norm+lm-head only ever
    /// run on the one row that's actually needed).
    pub fn forward_chunk(
        &mut self,
        token_ids: &[u32],
        want_logits: bool,
        prof: Option<&Profiler>,
    ) -> Result<Option<Vec<f32>>, RocmlError> {
        let chunk_len = token_ids.len() as u32;
        if chunk_len == 0 {
            return Ok(None);
        }
        if chunk_len > CHUNK_CAP {
            return Err(RocmlError::Config(format!(
                "forward_chunk: chunk_len {chunk_len} exceeds CHUNK_CAP {CHUNK_CAP}"
            )));
        }
        let pos_base = self.pos;
        let max_seq = self.cache.max_seq();
        if pos_base + chunk_len > max_seq {
            return Err(RocmlError::ContextOverflow {
                requested: pos_base + chunk_len,
                max_seq,
            });
        }
        let hidden = self.config.embedding_length;

        self.chunk_scratch
            .token_ids
            .copy_prefix_from_host(token_ids)?;
        Profiler::scope(
            prof,
            None,
            OpKind::Embed,
            profile::embed_bytes(hidden) * chunk_len as u64,
            0,
            || {
                self.kernels.embedding(
                    offset(&self.chunk_scratch.token_ids, 0),
                    offset(&self.weights.token_embd, 0),
                    offset(&self.chunk_scratch.x, 0),
                    chunk_len,
                    hidden,
                )
            },
        )?;

        for (layer_idx, layer) in self.weights.layers.iter().enumerate() {
            let layer_idx_u32 = layer_idx as u32;
            match (layer, self.config.layer_kinds[layer_idx]) {
                (LayerWeights::Gdn(gdn_weights), LayerKind::LinearAttention) => {
                    let state = self.cache.gdn_mut(layer_idx)?;
                    gdn_chunk_step(
                        &self.kernels,
                        &self.chunk_kernels,
                        &self.gdn_cw_kernels,
                        &self.config,
                        gdn_weights,
                        state,
                        &mut self.chunk_scratch,
                        chunk_len,
                        prof,
                        Some(layer_idx_u32),
                    )?;
                    ffn_chunk_step(
                        &self.kernels,
                        &gdn_weights.ffn,
                        &gdn_weights.post_attention_norm,
                        hidden,
                        self.config.feed_forward_length,
                        self.config.rms_eps,
                        &mut self.chunk_scratch,
                        chunk_len,
                        prof,
                        Some(layer_idx_u32),
                    )?;
                }
                (LayerWeights::Attention(attn_weights), LayerKind::FullAttention) => {
                    // Chunked prefill only ever runs when the cache has no
                    // mixed layers (see `Model::forward_prompt`'s doc
                    // comment) — attn_dense_mut errors clearly if that
                    // invariant is ever violated instead of silently
                    // misinterpreting a quantized plane as dense.
                    let plane = self.cache.attn_dense_mut(layer_idx)?;
                    attention_chunk_step(
                        &self.kernels,
                        &self.hybrid,
                        &self.chunk_kernels,
                        &self.config,
                        attn_weights,
                        plane,
                        max_seq,
                        &mut self.chunk_scratch,
                        pos_base,
                        chunk_len,
                        prof,
                        Some(layer_idx_u32),
                    )?;
                    ffn_chunk_step(
                        &self.kernels,
                        &attn_weights.ffn,
                        &attn_weights.post_attention_norm,
                        hidden,
                        self.config.feed_forward_length,
                        self.config.rms_eps,
                        &mut self.chunk_scratch,
                        chunk_len,
                        prof,
                        Some(layer_idx_u32),
                    )?;
                }
                _ => {
                    return Err(RocmlError::Config(format!(
                        "layer {layer_idx}: weight/kind mismatch (internal bug)"
                    )))
                }
            }
        }

        self.pos += chunk_len;

        if !want_logits {
            return Ok(None);
        }

        // Logits trap: normalize and project only the last row of the
        // chunk's residual stream, never the whole [chunk_len, hidden].
        let last_row = (chunk_len - 1) as usize * hidden as usize;
        Profiler::scope(
            prof,
            None,
            OpKind::Norm,
            profile::norm_bytes(1, hidden),
            profile::norm_flops(1, hidden),
            || {
                self.kernels.rmsnorm(
                    offset(&self.chunk_scratch.x, last_row),
                    offset(&self.weights.output_norm, 0),
                    offset(&self.chunk_scratch.xn, 0),
                    1,
                    hidden,
                    self.config.rms_eps,
                )
            },
        )?;
        let lm_head_bytes = profile::matvec_bytes(
            self.weights.output.byte_size(),
            self.config.vocab_size,
            hidden,
        );
        let lm_head_flops = profile::matvec_flops(self.config.vocab_size, hidden);
        Profiler::scope(
            prof,
            None,
            OpKind::LmHead,
            lm_head_bytes,
            lm_head_flops,
            || {
                self.weights.output.matvec(
                    &self.kernels,
                    offset(&self.chunk_scratch.xn, 0),
                    offset(&self.chunk_scratch.logits, 0),
                    self.config.vocab_size,
                    hidden,
                )
            },
        )?;

        let mut logits = vec![0.0f32; self.config.vocab_size as usize];
        self.chunk_scratch.logits.copy_to_host(&mut logits)?;
        Ok(Some(logits))
    }
}
