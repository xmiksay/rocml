//! Single-token decode-style forward pass for the qwen35 hybrid
//! architecture: embedding, N transformer blocks (each either a GDN or a
//! full-attention layer, per `Qwen35Config::layer_kinds`), final norm,
//! logits. Mirrors `crate::forward` (the dense Qwen3 path) structurally.

mod attention;
mod attention_chunk;
mod chunk_forward;
mod chunk_kernels;
mod chunk_scratch;
mod ffn;
mod ffn_chunk;
mod gdn;
mod gdn_chunk;
mod gdn_chunkwise;
mod gdn_chunkwise_kernels;
mod kernels;
pub(crate) mod kernels_mixed;
mod scratch;
mod snapshot;

use std::path::Path;

use chunk_kernels::ChunkKernels;
use chunk_scratch::ChunkScratch;
use gdn_chunkwise_kernels::GdnChunkwiseKernels;
use kernels::HybridKernels;
use kernels_mixed::MixedKernels;
use rocml_core::gguf::GgufFile;
use rocml_hip::{Device, MemoryInfo};
use scratch::Scratch;

use super::cache::HybridCache;
use super::config::{LayerKind, Qwen35Config};
use super::weights::{LayerWeights, ModelWeights};
use crate::budget::{mixed_kv_bytes_per_token, Budget, HIGH_USAGE_WARN_FRACTION};
use crate::error::RocmlError;
use crate::forward::kernels::{offset, Kernels};
use crate::load_opts::LoadOptions;
use crate::profile::{self, OpKind, Phase, Profiler};

pub struct Model {
    _device: Device,
    config: Qwen35Config,
    weights: ModelWeights,
    cache: HybridCache,
    kernels: Kernels,
    hybrid: HybridKernels,
    /// KIVI-style mixed KV cache kernels (issue #2) — loaded unconditionally
    /// like every other kernel set, even when `opts.kv_cache` is dense-only,
    /// since it's cheap and keeps `Model` shape independent of the chosen
    /// mode.
    mixed_kernels: MixedKernels,
    scratch: Scratch,
    /// Chunked-prefill-only kernels/scratch (issue #6) — see
    /// `chunk_forward::forward_chunk`. Loaded unconditionally at model load
    /// (the same way `scratch`/`hybrid` always are), since the public API
    /// always processes the prompt in chunks (see `crate::generate`).
    chunk_kernels: ChunkKernels,
    /// Chunkwise (blocked delta-rule) GDN recurrence kernels — see
    /// `gdn_chunkwise.rs`. Loaded unconditionally alongside `chunk_kernels`.
    gdn_cw_kernels: GdnChunkwiseKernels,
    chunk_scratch: ChunkScratch,
    pos: u32,
}

impl Model {
    /// See `crate::forward::Model::load`'s doc comment — mirrors that
    /// budget-check-then-allocate flow, with `n_cache_layers` counting only
    /// this architecture's full-attention layers (GDN layers' state is O(1)
    /// in context length, see `HybridCache`). Issue #2's quantized modes
    /// (`Q8`/`Q4Mixed`) are only implemented for this hybrid architecture,
    /// not the dense one — see `HybridCache::new`'s doc comment for the
    /// boundary-layer-skip rule this budget calculation must also account
    /// for (`mixed_kv_bytes_per_token` takes the boundary-layer count).
    pub fn load(gguf_path: impl AsRef<Path>, opts: LoadOptions) -> Result<Self, RocmlError> {
        let device = Device::new(0)?;
        let gguf = GgufFile::open(gguf_path)?;
        let config = Qwen35Config::from_gguf(&gguf)?;
        let weights = ModelWeights::load(&gguf, &config)?;

        let n_attn_layers = config
            .layer_kinds
            .iter()
            .filter(|k| **k == LayerKind::FullAttention)
            .count() as u32;
        // Boundary layers (first + last full-attention layer) always stay
        // dense fp16 regardless of mode — see HybridCache::new. A
        // single-attention-layer model has zero mixed layers (that layer is
        // both boundary positions at once).
        let n_boundary_layers = n_attn_layers.min(2);
        let n_mixed_layers = n_attn_layers.saturating_sub(n_boundary_layers);
        let v_bits: u8 = match opts.kv_cache {
            crate::load_opts::KvCacheMode::Q4Mixed => 4,
            _ => 8,
        };
        let (per_token, fixed_overhead) = mixed_kv_bytes_per_token(
            opts.kv_cache,
            n_boundary_layers,
            n_mixed_layers,
            config.head_count_kv,
            config.head_dim,
            v_bits,
        );
        let ctx = opts.ctx.min(config.context_length as usize).max(1);
        let mem = device.memory_info()?;
        let budget = Budget::for_loaded_weights_with_overhead(
            mem.total as u64,
            mem.free as u64,
            per_token,
            fixed_overhead,
        );
        if !budget.fits(ctx) {
            return Err(RocmlError::VramBudget {
                requested_ctx: ctx,
                suggested_max_ctx: budget.max_ctx(),
                breakdown: budget.breakdown(ctx),
            });
        }
        if budget.usage_fraction(ctx) > HIGH_USAGE_WARN_FRACTION {
            eprintln!(
                "warning: ctx {ctx} predicts {:.1}% VRAM usage ({})",
                budget.usage_fraction(ctx) * 100.0,
                budget.breakdown(ctx)
            );
        }

        let cache = HybridCache::new(&config, ctx, opts.kv_cache)?;
        let kernels = Kernels::load_all()?;
        let hybrid = HybridKernels::load_all()?;
        let mixed_kernels = MixedKernels::load_all()?;
        let scratch = Scratch::new(&config)?;
        let chunk_kernels = ChunkKernels::load_all()?;
        let gdn_cw_kernels = GdnChunkwiseKernels::load_all()?;
        let chunk_scratch = ChunkScratch::new(&config)?;

        Ok(Self {
            _device: device,
            config,
            weights,
            cache,
            kernels,
            hybrid,
            mixed_kernels,
            scratch,
            chunk_kernels,
            gdn_cw_kernels,
            chunk_scratch,
            pos: 0,
        })
    }

    pub fn config(&self) -> &Qwen35Config {
        &self.config
    }

    pub fn memory_info(&self) -> Result<MemoryInfo, RocmlError> {
        self._device.memory_info().map_err(Into::into)
    }

    pub fn position(&self) -> u32 {
        self.pos
    }

    /// Resets decode position and zeroes every GDN layer's conv/recurrence
    /// state (full-attention K/V planes need no explicit reset — see
    /// `HybridCache::reset`'s doc comment).
    pub fn reset(&mut self) -> Result<(), RocmlError> {
        self.pos = 0;
        self.cache.reset()
    }

    pub fn forward_token(&mut self, token_id: u32) -> Result<Vec<f32>, RocmlError> {
        self.forward_token_profiled(token_id, None)
    }

    /// Like [`Self::forward_token`], but instruments every op through `prof`
    /// when given — see `crate::forward::Model::forward_token_profiled`'s
    /// doc comment for the prefill-vs-decode granularity split, mirrored
    /// here.
    pub fn forward_token_profiled(
        &mut self,
        token_id: u32,
        prof: Option<&Profiler>,
    ) -> Result<Vec<f32>, RocmlError> {
        let pos = self.pos;
        let max_seq = self.cache.max_seq();
        if pos >= max_seq {
            return Err(RocmlError::ContextOverflow {
                requested: pos + 1,
                max_seq,
            });
        }
        let hidden = self.config.embedding_length;
        let cur_len = pos + 1;
        let coarse_prefill = prof.map(|p| p.phase() == Phase::Prefill).unwrap_or(false);

        self.scratch.token_id.copy_from_host(&[token_id])?;
        Profiler::scope(
            prof,
            None,
            OpKind::Embed,
            profile::embed_bytes(hidden),
            0,
            || {
                self.kernels.embedding(
                    offset(&self.scratch.token_id, 0),
                    offset(&self.weights.token_embd, 0),
                    offset(&self.scratch.x, 0),
                    1,
                    hidden,
                )
            },
        )?;

        for (layer_idx, layer) in self.weights.layers.iter().enumerate() {
            let layer_idx_u32 = layer_idx as u32;
            match (layer, self.config.layer_kinds[layer_idx]) {
                (LayerWeights::Gdn(gdn_weights), LayerKind::LinearAttention) => {
                    let state = self.cache.gdn_mut(layer_idx)?;
                    if coarse_prefill {
                        let (gdn_bytes, gdn_flops) =
                            gdn::gdn_layer_step_cost(&self.config, gdn_weights);
                        let (ffn_bytes, ffn_flops) = ffn::ffn_step_cost(
                            &gdn_weights.ffn,
                            hidden,
                            self.config.feed_forward_length,
                        );
                        Profiler::scope(
                            prof,
                            Some(layer_idx_u32),
                            OpKind::Layer,
                            gdn_bytes + ffn_bytes,
                            gdn_flops + ffn_flops,
                            || {
                                gdn::gdn_layer_step(
                                    &self.kernels,
                                    &self.hybrid,
                                    &self.config,
                                    gdn_weights,
                                    state,
                                    &mut self.scratch,
                                    None,
                                    None,
                                )?;
                                ffn::ffn_step(
                                    &self.kernels,
                                    &gdn_weights.ffn,
                                    &gdn_weights.post_attention_norm,
                                    hidden,
                                    self.config.feed_forward_length,
                                    self.config.rms_eps,
                                    &mut self.scratch,
                                    None,
                                    None,
                                )
                            },
                        )?;
                    } else {
                        gdn::gdn_layer_step(
                            &self.kernels,
                            &self.hybrid,
                            &self.config,
                            gdn_weights,
                            state,
                            &mut self.scratch,
                            prof,
                            Some(layer_idx_u32),
                        )?;
                        ffn::ffn_step(
                            &self.kernels,
                            &gdn_weights.ffn,
                            &gdn_weights.post_attention_norm,
                            hidden,
                            self.config.feed_forward_length,
                            self.config.rms_eps,
                            &mut self.scratch,
                            prof,
                            Some(layer_idx_u32),
                        )?;
                    }
                }
                (LayerWeights::Attention(attn_weights), LayerKind::FullAttention) => {
                    let plane = self.cache.attn_mut(layer_idx)?;
                    if coarse_prefill {
                        let (attn_bytes, attn_flops) =
                            attention::attention_step_cost(&self.config, attn_weights, cur_len);
                        let (ffn_bytes, ffn_flops) = ffn::ffn_step_cost(
                            &attn_weights.ffn,
                            hidden,
                            self.config.feed_forward_length,
                        );
                        Profiler::scope(
                            prof,
                            Some(layer_idx_u32),
                            OpKind::Layer,
                            attn_bytes + ffn_bytes,
                            attn_flops + ffn_flops,
                            || {
                                attention::attention_step(
                                    &self.kernels,
                                    &self.hybrid,
                                    &self.mixed_kernels,
                                    &self.config,
                                    attn_weights,
                                    plane,
                                    max_seq,
                                    &mut self.scratch,
                                    pos,
                                    None,
                                    None,
                                )?;
                                ffn::ffn_step(
                                    &self.kernels,
                                    &attn_weights.ffn,
                                    &attn_weights.post_attention_norm,
                                    hidden,
                                    self.config.feed_forward_length,
                                    self.config.rms_eps,
                                    &mut self.scratch,
                                    None,
                                    None,
                                )
                            },
                        )?;
                    } else {
                        attention::attention_step(
                            &self.kernels,
                            &self.hybrid,
                            &self.mixed_kernels,
                            &self.config,
                            attn_weights,
                            plane,
                            max_seq,
                            &mut self.scratch,
                            pos,
                            prof,
                            Some(layer_idx_u32),
                        )?;
                        ffn::ffn_step(
                            &self.kernels,
                            &attn_weights.ffn,
                            &attn_weights.post_attention_norm,
                            hidden,
                            self.config.feed_forward_length,
                            self.config.rms_eps,
                            &mut self.scratch,
                            prof,
                            Some(layer_idx_u32),
                        )?;
                    }
                }
                _ => {
                    return Err(RocmlError::Config(format!(
                        "layer {layer_idx}: weight/kind mismatch (internal bug)"
                    )))
                }
            }
        }

        Profiler::scope(
            prof,
            None,
            OpKind::Norm,
            profile::norm_bytes(1, hidden),
            profile::norm_flops(1, hidden),
            || {
                self.kernels.rmsnorm(
                    offset(&self.scratch.x, 0),
                    offset(&self.weights.output_norm, 0),
                    offset(&self.scratch.xn, 0),
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
                    offset(&self.scratch.xn, 0),
                    offset(&self.scratch.logits, 0),
                    self.config.vocab_size,
                    hidden,
                )
            },
        )?;

        let mut logits = vec![0.0f32; self.config.vocab_size as usize];
        self.scratch.logits.copy_to_host(&mut logits)?;

        self.pos += 1;
        Ok(logits)
    }

    /// Processes a whole prompt and returns its last token's logits — the
    /// hybrid architecture's public prompt-processing entry point (see
    /// `crate::model::Model::forward_prompt`). Issue #6's batched chunked
    /// path (`forward_prompt_chunked`, defined in `chunk_forward.rs`)
    /// doesn't support the mixed KV cache (issue #2) yet — quantize-on-evict
    /// and the fused mixed-KV read are only wired into the decode-style
    /// attention step (`attention::attention_step`), not the chunked one
    /// (`attention_chunk::attention_chunk_step`) — so a cache with any
    /// mixed layer falls back to the token-serial loop every architecture
    /// already has via `forward_token`, at a prefill-throughput cost this
    /// is a documented, deliberate scope cut for (decode is where issue #2's
    /// bandwidth win actually matters — see that issue's own review comment
    /// on why quantized KV speeds decode, not just capacity).
    pub fn forward_prompt(
        &mut self,
        prompt_ids: &[u32],
        prof: Option<&Profiler>,
    ) -> Result<Vec<f32>, RocmlError> {
        if self.cache.has_mixed_layers() {
            let mut logits = Vec::new();
            for &id in prompt_ids {
                logits = self.forward_token_profiled(id, prof)?;
            }
            Ok(logits)
        } else {
            self.forward_prompt_chunked(prompt_ids, prof)
        }
    }
}
