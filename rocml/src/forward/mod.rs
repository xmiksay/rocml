//! Single-token decode-style forward pass: embedding lookup, N transformer
//! layers (attention.rs + ffn.rs), final norm, logits. Prompt processing
//! reuses this exact path one token at a time — batched prefill (a fused
//! multi-token gemm path) is a later performance milestone, not this one.

mod attention;
mod ffn;
pub(crate) mod kernels;
pub(crate) mod kernels_flash;
pub(crate) mod kernels_kv;
pub(crate) mod kernels_mmq;
pub(crate) mod kernels_quant;
mod kernels_quant_dispatch;
mod scratch;

use std::path::Path;

use kernels::{offset, Kernels};
use rocml_core::gguf::GgufFile;
use rocml_hip::{Device, MemoryInfo};
use scratch::Scratch;

use crate::budget::{kv_bytes_per_token, Budget, HIGH_USAGE_WARN_FRACTION};
use crate::cache::KvCache;
use crate::config::ModelConfig;
use crate::error::RocmlError;
use crate::load_opts::LoadOptions;
use crate::profile::{self, OpKind, Phase, Profiler};
use crate::weights::ModelWeights;

pub struct Model {
    // Kept alive for the model's lifetime: HIP device selection is
    // per-thread global state, not scoped to this handle, but every other
    // field here assumes device 0 stays selected and its context alive.
    _device: Device,
    config: ModelConfig,
    weights: ModelWeights,
    cache: KvCache,
    kernels: Kernels,
    scratch: Scratch,
    pos: u32,
}

impl Model {
    /// Loads a dense Qwen3 GGUF onto the GPU: opens/mmaps the file, reads
    /// config, dequantizes and uploads every weight, runs the authoritative
    /// post-weights-load VRAM budget check (issue #3 — see `crate::budget`),
    /// then allocates the KV cache and scratch buffers and loads every
    /// kernel this forward pass needs.
    pub fn load(gguf_path: impl AsRef<Path>, opts: LoadOptions) -> Result<Self, RocmlError> {
        if opts.kv_cache.is_quantized() {
            return Err(RocmlError::Config(format!(
                "kv-cache mode {:?} is not yet implemented for the dense qwen3 architecture \
                 (issue #2 targets the qwen3.5 hybrid architecture's full-attention layers)",
                opts.kv_cache
            )));
        }
        let device = Device::new(0)?;
        let gguf = GgufFile::open(gguf_path)?;
        let config = ModelConfig::from_gguf(&gguf)?;
        let weights = ModelWeights::load(&gguf, &config)?;

        // Authoritative check: real hipMemGetInfo numbers now that weights
        // are actually resident, unlike registry::clamp_ctx's necessarily
        // approximate pre-load (file-size-proxied) estimate.
        let dtype = opts.kv_cache.dense_dtype();
        let per_token = kv_bytes_per_token(
            config.block_count,
            config.head_count_kv,
            config.head_dim,
            dtype,
        );
        let ctx = opts.ctx.min(config.context_length as usize).max(1);
        let mem = device.memory_info()?;
        let budget = Budget::for_loaded_weights(mem.total as u64, mem.free as u64, per_token);
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

        let cache = KvCache::new(&config, ctx, dtype)?;
        // `opts.use_mmq` is threaded through for consistency with the
        // qwen35 hybrid loader, but is inert here: the dense architecture
        // never chunks prefill (see `Model::forward_prompt`'s doc comment),
        // so `LinearWeight::matmul`/the MMQ dispatch it can reach are never
        // called on this path.
        let kernels = Kernels::load_all(opts.use_mmq)?;
        let scratch = Scratch::new(&config)?;

        Ok(Self {
            _device: device,
            config,
            weights,
            cache,
            kernels,
            scratch,
            pos: 0,
        })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Free/total VRAM on the current device, for reporting how much a load
    /// consumed (see the `generate` example).
    pub fn memory_info(&self) -> Result<MemoryInfo, RocmlError> {
        self._device.memory_info().map_err(Into::into)
    }

    /// Number of tokens already fed through `forward_token` (the next call
    /// writes cache position `position()`).
    pub fn position(&self) -> u32 {
        self.pos
    }

    /// Clears the cache position so the next `forward_token` starts a fresh
    /// sequence. Cheap: cache buffers are only ever read up to the current
    /// position, so stale bytes beyond it are never observed — no need to
    /// re-zero them.
    pub fn reset(&mut self) {
        self.pos = 0;
    }

    /// Runs one decode step for `token_id` at the current cache position and
    /// returns that step's vocab-sized logits (host-side, f32).
    pub fn forward_token(&mut self, token_id: u32) -> Result<Vec<f32>, RocmlError> {
        self.forward_token_profiled(token_id, None)
    }

    /// Like [`Self::forward_token`], but instruments every op through `prof`
    /// when given. During `Phase::Prefill`, per-layer sub-ops are *not*
    /// individually timed — see the `profile` module doc for why (bounding
    /// event count over a long token-serial prefill loop) — each layer is
    /// instead timed as a single [`OpKind::Layer`] span with analytically
    /// pre-summed bytes/flops.
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
            if coarse_prefill {
                let (attn_bytes, attn_flops) =
                    attention::attention_step_cost(&self.config, layer, cur_len);
                let (ffn_bytes, ffn_flops) = ffn::ffn_step_cost(&self.config, layer);
                Profiler::scope(
                    prof,
                    Some(layer_idx as u32),
                    OpKind::Layer,
                    attn_bytes + ffn_bytes,
                    attn_flops + ffn_flops,
                    || {
                        attention::attention_step(
                            &self.kernels,
                            &self.config,
                            layer,
                            &mut self.cache,
                            &mut self.scratch,
                            layer_idx,
                            pos,
                            None,
                        )?;
                        ffn::ffn_step(
                            &self.kernels,
                            &self.config,
                            layer,
                            &mut self.scratch,
                            None,
                            None,
                        )
                    },
                )?;
            } else {
                attention::attention_step(
                    &self.kernels,
                    &self.config,
                    layer,
                    &mut self.cache,
                    &mut self.scratch,
                    layer_idx,
                    pos,
                    prof,
                )?;
                ffn::ffn_step(
                    &self.kernels,
                    &self.config,
                    layer,
                    &mut self.scratch,
                    prof,
                    Some(layer_idx as u32),
                )?;
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
}
