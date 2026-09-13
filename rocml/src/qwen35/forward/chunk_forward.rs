//! `Model::forward_chunk`: processes a whole prefill chunk of tokens through
//! the batched kernel paths (`attention_chunk`/`gdn_chunk`/`ffn_chunk`)
//! instead of `forward_token_profiled`'s one-token-at-a-time loop. See issue
//! #6's design and `generate::generate_core`'s caller for how a prompt gets
//! split into chunks.

use super::attention_chunk::attention_chunk_step;
use super::attention_chunk_mixed::attention_chunk_step_mixed;
use super::chunk_scratch::CHUNK_CAP;
use super::ffn_chunk::ffn_chunk_step;
use super::gdn_chunk::gdn_chunk_step;
use super::Model;
use crate::error::RocmlError;
use crate::forward::kernels::offset;
use crate::profile::{self, OpKind, Profiler};
use crate::qwen35::cache::AttnLayerCache;
use crate::qwen35::config::LayerKind;
use crate::qwen35::weights::LayerWeights;

/// Prompt-chunk size `generate` drives the hybrid forward pass with.
/// Originally chosen as 128 by measurement among {128, 256, 512} on
/// Qwen3.5-2B `bench --depth 2048` back when the GEMM path was still
/// scalar-FMA (issue #6's original report: 246/241/239 tok/s respectively —
/// not yet compute-bound enough for a bigger chunk's launch-overhead
/// amortization to outweigh its slightly larger per-token kernel overhead).
/// Re-swept on the WMMA-pipeline round: the WMMA GEMM kernel tiles rows
/// internally in fixed 128-row tiles regardless of chunk size (a bigger
/// chunk here just means more row-tiles per launch, not a kernel change),
/// and every batched kernel this touches (GDN conv/chunkwise-recurrence,
/// attention, FFN) was already sized/looped for up to `CHUNK_CAP` (512)
/// tokens (`chunk_scratch.rs`) — so growing this constant needed no other
/// code change. Measured end-to-end on ornith-9b (Q4_K_M) `bench --depth
/// {2048,8192}` (median of 3 runs, decode unaffected either way):
///   128 -> 256 -> 512 tok/s @ 2048: 466.3 -> 528.8 -> 587.0
///   128 -> 256 -> 512 tok/s @ 8192: 403.5 -> 450.1 -> 489.8
/// 512 wins outright at both depths (launch-overhead amortization keeps
/// paying off all the way to `CHUNK_CAP`, unlike the scalar-kernel-era
/// sweep above) with no VRAM cost (`ChunkScratch` was already allocated at
/// `CHUNK_CAP` regardless of this constant) and no GDN chunkwise-recurrence
/// code change needed (`gdn_chunkwise_step` already sub-chunks any
/// `chunk_len > GDN_RECUR_TILE`(128) into 128-token tiles, carrying state
/// between them — this was written for `forward_chunk`'s documented
/// `1..=CHUNK_CAP` contract from the start, just never previously exercised
/// beyond one tile per call).
pub const PREFILL_CHUNK_SIZE: u32 = 512;

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
                    // Boundary layers stay dense fp16 regardless of
                    // `KvCacheMode` (see `HybridCache::new`'s doc comment);
                    // layers strictly between them are `Mixed` whenever the
                    // cache is quantized. Both variants get a chunked-prefill
                    // path — see `attention_chunk_mixed.rs`'s module doc for
                    // why the mixed one is its own file rather than a branch
                    // inside `attention_chunk_step`.
                    match self.cache.attn_mut(layer_idx)? {
                        AttnLayerCache::Dense(plane) => attention_chunk_step(
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
                        )?,
                        AttnLayerCache::Mixed(plane) => attention_chunk_step_mixed(
                            &self.kernels,
                            &self.hybrid,
                            &self.chunk_kernels,
                            &self.mixed_kernels,
                            &self.flash_mixed_kernels,
                            &self.config,
                            attn_weights,
                            plane,
                            &mut self.chunk_scratch,
                            pos_base,
                            chunk_len,
                            prof,
                            Some(layer_idx_u32),
                        )?,
                    }
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
                // A plain `matvec` (single output row, never an MMQ-eligible
                // shape) — see "Logits trap" above: only the chunk's last
                // row ever reaches the lm-head, not a batched `matmul`.
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
