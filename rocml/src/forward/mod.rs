//! Single-token decode-style forward pass: embedding lookup, N transformer
//! layers (attention.rs + ffn.rs), final norm, logits. Decode (and a
//! prompt's own last, possibly-partial chunk when it's short) can still run
//! one token at a time, but prompt processing now goes through
//! `chunk_forward`'s batched multi-token path (issue #16's dense
//! chunked-prefill port) — see `crate::model::Model::forward_prompt`'s doc
//! comment. The Dense/Mixed per-layer cache dispatch `attention::
//! attention_step` (decode) and `attention_chunk::attention_chunk_step`
//! (prefill) both reuse the same `AttnLayerCache` type and mixed-cache
//! kernels issue #2/#16's dense mixed-KV work landed.

mod attention;
mod attention_chunk;
mod chunk_forward;
mod chunk_scratch;
mod ffn;
mod ffn_chunk;
pub(crate) mod kernels;
pub(crate) mod kernels_act;
pub(crate) mod kernels_flash;
pub(crate) mod kernels_kv;
pub(crate) mod kernels_mmq;
pub(crate) mod kernels_quant;
mod kernels_quant_dispatch;
pub(crate) mod kernels_splitk;
mod scratch;

use std::path::Path;

use chunk_scratch::ChunkScratch;
use kernels::{offset, Kernels};
use kernels_act::ActivationKernels;
use rocml_core::gguf::GgufFile;
use rocml_hip::{Device, MemoryInfo};
use scratch::Scratch;

use crate::budget::{
    kv_bytes_per_token, mixed_kv_bytes_per_token, Budget, HIGH_USAGE_WARN_FRACTION,
};
use crate::cache::DenseAttnCache;
use crate::config::ModelConfig;
use crate::error::RocmlError;
use crate::kv_quant::validate_sink_window;
use crate::load_opts::LoadOptions;
use crate::profile::{self, OpKind, Phase, Profiler};
use crate::qwen35::forward::chunk_kernels::ChunkKernels;
use crate::qwen35::forward::kernels_flash_mixed::FlashPrefillMixedKernels;
use crate::qwen35::forward::kernels_mixed::MixedKernels;
use crate::weights::ModelWeights;

pub struct Model {
    // Kept alive for the model's lifetime: HIP device selection is
    // per-thread global state, not scoped to this handle, but every other
    // field here assumes device 0 stays selected and its context alive.
    _device: Device,
    config: ModelConfig,
    weights: ModelWeights,
    cache: DenseAttnCache,
    kernels: Kernels,
    /// KIVI-style mixed KV cache kernels (issue #2/#16) — loaded
    /// unconditionally like every other kernel set, even when `opts.kv_cache`
    /// is dense-only, mirroring `qwen35::forward::Model`'s own
    /// `mixed_kernels` field for the same reason (cheap, keeps `Model`
    /// shape independent of the chosen mode).
    mixed_kernels: MixedKernels,
    /// Chunked-prefill sibling of `mixed_kernels`' decode-attention kernel
    /// (issue #16, mirroring the qwen35 hybrid path's identical field) —
    /// see `kernels_flash_mixed.rs`'s module doc. Loaded unconditionally
    /// alongside `mixed_kernels` for the same reason.
    flash_mixed_kernels: FlashPrefillMixedKernels,
    /// GeGLU activation kernel (issue #16's config-driven-activation seam)
    /// — a separate field from `kernels` rather than nested inside it, so
    /// `kernels.rs` (already well past the 400-line cap) never grows to add
    /// it; see `kernels_act.rs`'s module doc.
    act: ActivationKernels,
    scratch: Scratch,
    /// Chunked-prefill-only kernels/scratch (issue #16's dense chunked-
    /// prefill port) — see `chunk_forward::forward_chunk`. `ChunkKernels`
    /// is reused as-is from the qwen35 hybrid path (architecture-generic,
    /// see `attention_chunk.rs`'s module doc), loaded unconditionally at
    /// model load like `scratch` always is.
    chunk_kernels: ChunkKernels,
    chunk_scratch: ChunkScratch,
    pos: u32,
}

impl Model {
    /// Loads a dense Qwen3 GGUF onto the GPU: opens/mmaps the file, reads
    /// config, dequantizes and uploads every weight, runs the authoritative
    /// post-weights-load VRAM budget check (issue #3 — see `crate::budget`),
    /// then allocates the KV cache and scratch buffers and loads every
    /// kernel this forward pass needs.
    ///
    /// `opts.kv_cache`'s quantized modes (issue #2, ported to this dense
    /// architecture per issue #16) are supported here the same way
    /// `qwen35::forward::Model::load` supports them: `DenseAttnCache::new`
    /// applies the generalized boundary-layer-skip rule (first + last layer
    /// stay dense fp16; see its own doc comment) and `mixed_kv_bytes_per_token`
    /// accounts for the mixed layers' VRAM cost the same way.
    pub fn load(gguf_path: impl AsRef<Path>, opts: LoadOptions) -> Result<Self, RocmlError> {
        let device = Device::new(0)?;
        let gguf = GgufFile::open(gguf_path)?;
        let config = ModelConfig::from_gguf(&gguf)?;
        let weights = ModelWeights::load(&gguf, &config)?;

        let ctx = opts.ctx.min(config.context_length as usize).max(1);
        if opts.kv_cache.is_quantized() {
            validate_sink_window(opts.kv_sink, opts.kv_window, ctx)?;
        }

        let dtype = opts.kv_cache.dense_dtype();
        let v_bits: u8 = match opts.kv_cache {
            crate::load_opts::KvCacheMode::Q4Mixed => 4,
            _ => 8,
        };
        // Boundary layers (first + last layer) always stay dense fp16
        // regardless of mode — see `DenseAttnCache::new`. A single-layer
        // model has zero mixed layers (that layer is both boundary
        // positions at once).
        let n_boundary_layers = config.block_count.min(2);
        let n_mixed_layers = config.block_count.saturating_sub(n_boundary_layers);
        let (per_token, fixed_overhead) = if opts.kv_cache.is_quantized() {
            mixed_kv_bytes_per_token(
                opts.kv_cache,
                n_boundary_layers,
                n_mixed_layers,
                config.head_count_kv,
                config.head_dim,
                v_bits,
                opts.kv_sink,
                opts.kv_window,
            )
        } else {
            (
                kv_bytes_per_token(
                    config.block_count,
                    config.head_count_kv,
                    config.head_dim,
                    dtype,
                ),
                0,
            )
        };

        // Authoritative check: real hipMemGetInfo numbers now that weights
        // are actually resident, unlike registry::clamp_ctx's necessarily
        // approximate pre-load (file-size-proxied) estimate.
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

        let rot_sim = opts
            .kv_rot_sim
            .map(|bpw| crate::kv_quant::rotational::RotSimSpec {
                bpw,
                apply_to_k: opts.kv_rot_sim_k,
            });
        let cache = DenseAttnCache::new(
            &config,
            ctx,
            opts.kv_cache,
            opts.kv_sink,
            opts.kv_window,
            rot_sim,
        )?;
        let kernels = Kernels::load_all(opts.use_mmq)?;
        let mixed_kernels = MixedKernels::load_all()?;
        let flash_mixed_kernels = FlashPrefillMixedKernels::load_all()?;
        let act = ActivationKernels::load_all()?;
        let scratch = Scratch::new(&config)?;
        let chunk_kernels = ChunkKernels::load_all()?;
        let chunk_scratch = ChunkScratch::new(&config)?;

        Ok(Self {
            _device: device,
            config,
            weights,
            cache,
            kernels,
            mixed_kernels,
            flash_mixed_kernels,
            act,
            scratch,
            chunk_kernels,
            chunk_scratch,
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
    /// sequence. Cheap for a dense plane (stale bytes past the current
    /// position are never read, see `crate::qwen35::cache`'s module doc);
    /// a mixed layer's eviction bookkeeping does need an explicit rewind
    /// (`DenseAttnCache::reset`'s doc comment), same as the hybrid cache.
    pub fn reset(&mut self) {
        self.pos = 0;
        self.cache.reset();
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
            let plane = self.cache.attn_mut(layer_idx)?;
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
                            &self.mixed_kernels,
                            &self.config,
                            layer,
                            plane,
                            max_seq,
                            &mut self.scratch,
                            layer_idx,
                            pos,
                            None,
                        )?;
                        ffn::ffn_step(
                            &self.kernels,
                            &self.act,
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
                    &self.mixed_kernels,
                    &self.config,
                    layer,
                    plane,
                    max_seq,
                    &mut self.scratch,
                    layer_idx,
                    pos,
                    prof,
                )?;
                ffn::ffn_step(
                    &self.kernels,
                    &self.act,
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
