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
mod expert_lru;
mod ffn;
mod ffn_chunk;
mod ffn_chunk_dispatch;
mod gdn;
mod gdn_chunk;
mod gdn_chunkwise;
mod gdn_chunkwise_kernels;
mod gdn_chunkwise_kernels_wmma;
mod kernels;
pub(crate) mod kernels_flash_mixed;
pub(crate) mod kernels_mixed;
mod kernels_moe;
mod kernels_moe_chunk;
pub mod layer_capture;
mod moe;
mod moe_cache;
mod moe_chunk;
mod moe_chunk_scratch;
mod moe_scratch;
mod rewind;
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
use kernels_moe::MoeKernels;
use kernels_moe_chunk::MoeChunkKernels;
use moe_cache::ExpertCache;
use moe_chunk::MoeChunkHost;
use moe_chunk_scratch::MoeChunkScratch;
use moe_scratch::MoeScratch;
use rocml_core::gguf::GgufFile;
use rocml_hip::{Device, MemoryInfo};
use scratch::Scratch;

use super::cache::HybridCache;
use super::config::{LayerKind, Qwen35Config};
use super::weights::{LayerWeights, ModelWeights};
use crate::budget::{
    mixed_kv_bytes_per_token, rewind_points_bytes, Budget, HIGH_USAGE_WARN_FRACTION,
};
use crate::error::RocmlError;
use crate::forward::kernels::Kernels;
use crate::load_opts::LoadOptions;
use crate::profile::Profiler;

pub use rewind::RewindSlot;

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
    /// GPU-resident rewind points, the snapshot layer's hot tier — see
    /// `rewind.rs`. Allocated lazily on first save, never on load.
    rewind: rewind::RewindPoints,
    /// qwen35moe's mixture-of-experts router + accumulate kernels — loaded
    /// unconditionally like every other kernel set (cheap; keeps `Model`'s
    /// shape independent of whether this checkpoint is MoE).
    moe_kernels: MoeKernels,
    /// `Some` only for a `general.architecture = "qwen35moe"` checkpoint —
    /// the routed-expert tensors are never uploaded to the GPU (see
    /// `crate::qwen35::weights::moe`'s module doc), so every MoE FFN step
    /// needs the GGUF's mmap kept alive for the model's whole lifetime to
    /// read an expert's raw bytes on demand. `None` for a plain `"qwen35"`
    /// checkpoint, which drops its `GgufFile` at the end of `Self::load`
    /// like every architecture did before this milestone.
    gguf: Option<GgufFile>,
    /// `Some` exactly when `gguf`/`config.moe` are `Some` — see
    /// `moe_scratch::MoeScratch`.
    moe_scratch: Option<MoeScratch>,
    /// M2's VRAM-resident LRU expert cache (`moe_cache::ExpertCache`) —
    /// `Some` for a MoE checkpoint whenever any VRAM was left to cache
    /// experts in after every other allocation (see its construction at the
    /// end of `Self::load`, sized from real free VRAM at that point, not an
    /// upfront estimate); `None` for a non-MoE checkpoint, or a MoE one with
    /// an explicit `--moe-cache-slots 0` override, or (rare) a card with no
    /// headroom left — either way `moe::moe_ffn_step` falls back to
    /// `MoeScratch`'s single-slot staging buffers (M1's always-copy path).
    expert_cache: Option<ExpertCache>,
    /// M3's grouped-by-expert batched-GEMM kernels for chunked prefill — see
    /// `moe_chunk.rs`'s module doc. Loaded unconditionally like every other
    /// kernel set; `moe_chunk_scratch`/`moe_chunk_host` are the ones gated
    /// on `config.moe.is_some()`.
    moe_chunk_kernels: MoeChunkKernels,
    /// `Some` exactly when `moe_scratch` is — the chunk-batched scratch
    /// `moe_chunk::moe_ffn_chunk_step` needs alongside the still-present
    /// per-row `MoeScratch` (used by decode and by the `LayerCapture`
    /// diagnostic fallback — see `ffn_chunk_dispatch.rs`).
    moe_chunk_scratch: Option<MoeChunkScratch>,
    /// Host-side expert-bucketing scratch for the grouped chunked-prefill
    /// path — `Some` alongside `moe_chunk_scratch`.
    moe_chunk_host: Option<MoeChunkHost>,
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
        let moe_kernels = MoeKernels::load_all()?;
        let moe_chunk_kernels = MoeChunkKernels::load_all()?;
        let moe_scratch = config
            .moe
            .as_ref()
            .map(|moe_cfg| MoeScratch::new(&config, moe_cfg, &weights))
            .transpose()?;
        let moe_chunk_scratch = config
            .moe
            .as_ref()
            .map(|moe_cfg| MoeChunkScratch::new(&config, moe_cfg))
            .transpose()?;
        let moe_chunk_host = config
            .moe
            .as_ref()
            .map(|moe_cfg| MoeChunkHost::new(moe_cfg, chunk_scratch::CHUNK_CAP));
        // Only a qwen35moe checkpoint needs its GGUF's mmap kept alive past
        // this point — see the `gguf` field's doc comment.
        let gguf = if config.moe.is_some() {
            Some(gguf)
        } else {
            None
        };

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
        // Mirrors `budget::estimate_from_gguf`'s `"qwen35"` arm exactly: a
        // dense KV mode has no mixed layers, so it must take the plain
        // per-layer formula (the mixed one under-counts fp16 by nearly 2x
        // on Ornith's shape); the GPU rewind points (`rewind.rs`) are a
        // fixed cost reserved here so their lazy first allocation can't
        // fail on a budget that only just fit the KV cache.
        let n_mixed_layers = if opts.kv_cache.is_quantized() {
            n_mixed_layers
        } else {
            0
        };
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
                crate::budget::kv_bytes_per_token(
                    n_attn_layers,
                    config.head_count_kv,
                    config.head_dim,
                    opts.kv_cache.dense_dtype(),
                ),
                0,
            )
        };
        let fixed_overhead =
            fixed_overhead + rewind_points_bytes(&config, n_mixed_layers, opts.kv_window);
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

        // M2's expert cache is sized last, from the *actual* free VRAM left
        // after every other allocation above (weights, KV cache, every
        // kernel/scratch buffer, the rewind reservation) — deliberately not
        // part of the pre-KV `Budget` check above, since caching more or
        // fewer experts only changes decode throughput, never correctness
        // (a miss falls back to M1's copy-every-time path). See
        // `moe_cache::ExpertCache`'s doc comment for the slot-stride layout.
        let expert_cache = match config.moe.as_ref() {
            Some(moe_cfg) => {
                let (gate_stride, up_stride, down_stride) =
                    moe_scratch::per_tensor_max_bytes(&weights);
                let bytes_per_slot = gate_stride + up_stride + down_stride;
                let total_experts =
                    moe_scratch::total_distinct_experts(&weights, moe_cfg.expert_count);
                let capacity = if bytes_per_slot == 0 || total_experts == 0 {
                    0
                } else {
                    let mem_now = device.memory_info()?;
                    let usable = mem_now
                        .free
                        .saturating_sub(moe_cache::EXPERT_CACHE_SAFETY_MARGIN_BYTES);
                    let by_vram = usable / bytes_per_slot;
                    opts.moe_cache_slots
                        .unwrap_or(usize::MAX)
                        .min(by_vram)
                        .min(total_experts)
                };
                eprintln!(
                    "moe expert cache: {capacity}/{total_experts} slots ({:.0} MiB, {:.1} MiB/slot)",
                    (capacity * bytes_per_slot) as f64 / (1024.0 * 1024.0),
                    bytes_per_slot as f64 / (1024.0 * 1024.0),
                );
                ExpertCache::new(capacity, gate_stride, up_stride, down_stride)?
            }
            None => None,
        };

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
            rewind: rewind::RewindPoints::default(),
            moe_kernels,
            gguf,
            moe_scratch,
            expert_cache,
            moe_chunk_kernels,
            moe_chunk_scratch,
            moe_chunk_host,
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

    /// `(hits, misses, occupancy, capacity)` since load for M2's expert
    /// cache — `None` for a non-MoE checkpoint or a MoE one that ended up
    /// with no cache at all (see `expert_cache`'s doc comment). `bench`/
    /// tests use this to report the measured cache hit rate.
    pub fn moe_cache_stats(&self) -> Option<(u64, u64, usize, usize)> {
        self.expert_cache.as_ref().map(|c| {
            let (hits, misses) = c.stats();
            (hits, misses, c.occupancy(), c.capacity())
        })
    }

    /// Resets decode position and zeroes every GDN layer's conv/recurrence
    /// state (full-attention K/V planes need no explicit reset — see
    /// `HybridCache::reset`'s doc comment).
    pub fn reset(&mut self) -> Result<(), RocmlError> {
        self.pos = 0;
        // A fresh sequence rewrites positions from 0, so every GPU rewind
        // point's saved prefix stops describing the live cache — see
        // `rewind.rs`'s validity invariant.
        self.invalidate_rewind_points();
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
