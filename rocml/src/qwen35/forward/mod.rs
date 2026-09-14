//! Single-token decode-style forward pass for the qwen35 hybrid
//! architecture: embedding, N transformer blocks (each either a GDN or a
//! full-attention layer, per `Qwen35Config::layer_kinds`), final norm,
//! logits. Mirrors `crate::forward` (the dense Qwen3 path) structurally.

mod attention;
mod attention_chunk;
mod attention_chunk_mixed;
mod chunk_forward;
pub(crate) mod chunk_kernels;
mod chunk_scratch;
mod decode_forward;
mod ffn;
mod ffn_chunk;
mod gdn;
mod gdn_chunk;
mod gdn_chunkwise;
mod gdn_chunkwise_kernels;
mod gdn_chunkwise_kernels_wmma;
mod kernels;
pub(crate) mod kernels_flash_mixed;
pub(crate) mod kernels_mixed;
pub mod layer_capture;
mod scratch;
mod snapshot;

use std::path::Path;

use chunk_kernels::ChunkKernels;
use chunk_scratch::ChunkScratch;
use gdn_chunkwise_kernels::GdnChunkwiseKernels;
use gdn_chunkwise_kernels_wmma::GdnChunkwiseWmmaKernels;
use kernels::HybridKernels;
use kernels_flash_mixed::FlashPrefillMixedKernels;
use kernels_mixed::MixedKernels;
use rocml_core::gguf::GgufFile;
use rocml_hip::{Device, MemoryInfo};
use scratch::Scratch;

use super::cache::HybridCache;
use super::config::{LayerKind, Qwen35Config};
use super::weights::{LayerWeights, ModelWeights};
use crate::budget::{mixed_kv_bytes_per_token, Budget, HIGH_USAGE_WARN_FRACTION};
use crate::error::RocmlError;
use crate::forward::kernels::Kernels;
use crate::load_opts::LoadOptions;
use crate::profile::Profiler;

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
    /// Chunked-prefill sibling of `mixed_kernels`' decode-attention kernel
    /// (issue #2's chunked-prefill follow-up) — see `kernels_flash_mixed.rs`.
    /// Loaded unconditionally alongside `mixed_kernels` for the same reason.
    flash_mixed_kernels: FlashPrefillMixedKernels,
    scratch: Scratch,
    /// Chunked-prefill-only kernels/scratch (issue #6) — see
    /// `chunk_forward::forward_chunk`. Loaded unconditionally at model load
    /// (the same way `scratch`/`hybrid` always are), since the public API
    /// always processes the prompt in chunks (see `crate::generate`).
    chunk_kernels: ChunkKernels,
    /// Chunkwise (blocked delta-rule) GDN recurrence kernels — see
    /// `gdn_chunkwise.rs`. Loaded unconditionally alongside `chunk_kernels`.
    gdn_cw_kernels: GdnChunkwiseKernels,
    /// WMMA variants of stages B/F/G (gdn-wmma round) — see
    /// `gdn_chunkwise_kernels_wmma.rs`'s module doc for why this is a
    /// separate struct/field rather than folded into `gdn_cw_kernels`.
    /// Loaded unconditionally; `gdn_chunkwise::run_tile` picks per call
    /// based on whether `head_k_dim`/`head_v_dim` are multiples of 16.
    gdn_cw_wmma_kernels: GdnChunkwiseWmmaKernels,
    chunk_scratch: ChunkScratch,
    pos: u32,
}

impl Model {
    /// See `crate::forward::Model::load`'s doc comment — mirrors that
    /// budget-check-then-allocate flow, with `n_cache_layers` counting only
    /// this architecture's full-attention layers (GDN layers' state is O(1)
    /// in context length, see `HybridCache`). Issue #2's quantized modes
    /// (`Q8`/`Q4Mixed`) were ported to the dense `qwen3` architecture too
    /// (issue #2 leftovers/#16 prep, see `crate::cache::DenseAttnCache`) —
    /// see `HybridCache::new`'s doc comment for the boundary-layer-skip rule
    /// this budget calculation must also account for (`mixed_kv_bytes_per_token`
    /// takes the boundary-layer count).
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
        let ctx = opts.ctx.min(config.context_length as usize).max(1);
        if opts.kv_cache.is_quantized() {
            crate::kv_quant::validate_sink_window(opts.kv_sink, opts.kv_window, ctx)?;
        }
        let (per_token, fixed_overhead) = mixed_kv_bytes_per_token(
            opts.kv_cache,
            n_boundary_layers,
            n_mixed_layers,
            config.head_count_kv,
            config.head_dim,
            v_bits,
            opts.kv_sink,
            opts.kv_window,
        );
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
        let cache = HybridCache::new(
            &config,
            ctx,
            opts.kv_cache,
            opts.kv_sink,
            opts.kv_window,
            rot_sim,
        )?;
        let kernels = Kernels::load_all(opts.use_mmq)?;
        let hybrid = HybridKernels::load_all()?;
        let mixed_kernels = MixedKernels::load_all()?;
        let flash_mixed_kernels = FlashPrefillMixedKernels::load_all()?;
        let scratch = Scratch::new(&config)?;
        let chunk_kernels = ChunkKernels::load_all()?;
        let gdn_cw_kernels = GdnChunkwiseKernels::load_all()?;
        let gdn_cw_wmma_kernels = GdnChunkwiseWmmaKernels::load_all()?;
        let chunk_scratch = ChunkScratch::new(&config)?;

        Ok(Self {
            _device: device,
            config,
            weights,
            cache,
            kernels,
            hybrid,
            mixed_kernels,
            flash_mixed_kernels,
            scratch,
            chunk_kernels,
            gdn_cw_kernels,
            gdn_cw_wmma_kernels,
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

    /// Processes a whole prompt and returns its last token's logits — the
    /// hybrid architecture's public prompt-processing entry point (see
    /// `crate::model::Model::forward_prompt`). `forward_prompt_chunked`
    /// (`chunk_forward.rs`) now supports every `KvCacheMode`, including the
    /// mixed quantized cache (issue #2's chunked-prefill follow-up):
    /// `chunk_forward.rs`'s per-full-attention-layer dispatch picks
    /// `attention_chunk_step`/`attention_chunk_step_mixed` per
    /// `AttnLayerCache::{Dense,Mixed}`, so there is no longer a token-serial
    /// fallback here (there used to be one — see git history around issue
    /// #2's chunked-prefill work for the old `has_mixed_layers()` check this
    /// replaced, at ~17-30 tok/s vs chunked prefill's hundreds).
    pub fn forward_prompt(
        &mut self,
        prompt_ids: &[u32],
        prof: Option<&Profiler>,
    ) -> Result<Vec<f32>, RocmlError> {
        self.forward_prompt_chunked(prompt_ids, prof)
    }
}
