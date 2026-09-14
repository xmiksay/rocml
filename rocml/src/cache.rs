//! KV cache storage for the dense (all full-attention) Qwen3 architecture.
//!
//! `KvDtype` (issue #3) is shared with `qwen35::cache`'s hybrid cache: `F16`
//! (default) halves the cache vs `F32` with negligible quality impact at
//! these scales (see the module doc on `attn_decode.hip`'s templated
//! `load_kv` seam for how the fused kernels read either dtype without ever
//! materializing a dequantized copy); `F32` exists purely so the parity
//! test suites can pin the pre-issue-#3 reference numerics exactly (see
//! `crate::load_opts::LoadOptions`).
//!
//! `DenseAttnCache` (issue #2/#16 — the dense-architecture mixed-KV port)
//! generalizes `qwen35::cache::HybridCache`'s boundary-layer-skip rule to a
//! model with no GDN layers at all: every layer here is a full-attention
//! layer, so a quantized `KvCacheMode` gives every layer *except* the first
//! and last the KIVI-style mixed cache
//! (`qwen35::cache_mixed::MixedAttnPlane`) — reused as-is, not forked: the
//! mixed cache and its kernels only ever take `n_kv_heads`/`head_dim`/
//! `max_seq`/`v_bits`/`sink_len`/`window_len`, no qwen35-specific config, so
//! nothing about them is hybrid-architecture-specific in the first place
//! (issue #16's "keep the kernels architecture-generic" constraint holds
//! for free here). `qwen35::cache::AttnPlane`/`AttnLayerCache` are reused
//! the same way for the dense-fp16/f32 and Dense/Mixed dispatch enum
//! respectively — see that module's doc comment for the per-layer storage
//! shape, unchanged by this reuse.
//!
//! No hardcoded cap on `max_seq` any more: issue #3 removed the old
//! `MAX_SEQ_CAP` constant in favor of an up-front VRAM budget check (see
//! `crate::budget` and `crate::registry::clamp_ctx`) — the caller supplies
//! whatever context length the budgeter approved, and `DenseAttnCache::new`
//! just allocates it.

use crate::config::ModelConfig;
use crate::error::RocmlError;
use crate::kv_quant::validate_sink_window;
use crate::load_opts::KvCacheMode;
use crate::qwen35::cache::{AttnLayerCache, AttnPlane};
use crate::qwen35::cache_mixed::MixedAttnPlane;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvDtype {
    F16,
    F32,
}

pub struct DenseAttnCache {
    max_seq: u32,
    layers: Vec<AttnLayerCache>,
}

impl DenseAttnCache {
    /// `ctx` is the caller's already-budgeted context length (see
    /// `crate::registry::clamp_ctx`) — clamped once more here against the
    /// model's own declared `context_length` as a final sanity bound,
    /// mirroring `qwen35::cache::HybridCache::new`.
    ///
    /// Boundary-layer skip (issue #2, generalized per issue #16): when
    /// `mode` is quantized, layer `0` and layer `block_count - 1` always get
    /// a dense fp16 plane regardless of `mode` — only the layers strictly
    /// between them get the mixed quantized layout. A single-layer model
    /// has no mixed layers at all (that one layer is both boundary
    /// positions at once), matching `HybridCache::new`'s same degenerate
    /// case.
    ///
    /// `sink_len`/`window_len` are issue #2 leftovers' `LoadOptions::kv_sink`/
    /// `kv_window` — validated here via `crate::kv_quant::validate_sink_window`
    /// (only when `mode.is_quantized()`; a non-quantized mode never
    /// allocates a mixed layer, so an out-of-range sink/window is inert and
    /// not worth rejecting).
    pub fn new(
        config: &ModelConfig,
        ctx: usize,
        mode: KvCacheMode,
        sink_len: u32,
        window_len: u32,
    ) -> Result<Self, RocmlError> {
        let max_seq = ctx.min(config.context_length as usize).max(1) as u32;
        let n_layers = config.block_count as usize;
        let dense_dtype = mode.dense_dtype();
        let v_bits: u8 = match mode {
            KvCacheMode::Q4Mixed => 4,
            _ => 8,
        };

        if mode.is_quantized() {
            validate_sink_window(sink_len, window_len, max_seq as usize)?;
        }

        let last = n_layers.saturating_sub(1);
        let mut layers = Vec::with_capacity(n_layers);
        for idx in 0..n_layers {
            let is_boundary = idx == 0 || idx == last;
            let layer = if mode.is_quantized() && !is_boundary {
                AttnLayerCache::Mixed(MixedAttnPlane::new(
                    config.head_count_kv,
                    config.head_dim,
                    max_seq,
                    v_bits,
                    sink_len,
                    window_len,
                )?)
            } else {
                AttnLayerCache::Dense(AttnPlane::new(
                    config.head_count_kv,
                    max_seq,
                    config.head_dim,
                    dense_dtype,
                )?)
            };
            layers.push(layer);
        }
        Ok(Self { max_seq, layers })
    }

    pub fn max_seq(&self) -> u32 {
        self.max_seq
    }

    /// Rewinds every mixed-layer's eviction bookkeeping for a fresh
    /// sequence — see `HybridCache::reset`'s doc comment for why this is
    /// necessary (unlike a dense plane's stale-bytes-never-read argument).
    /// No GDN state exists on this path to reset (the dense architecture
    /// has none).
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            if let AttnLayerCache::Mixed(plane) = layer {
                plane.reset();
            }
        }
    }

    pub fn attn_mut(&mut self, layer_idx: usize) -> Result<&mut AttnLayerCache, RocmlError> {
        self.layers.get_mut(layer_idx).ok_or_else(|| {
            RocmlError::Config(format!("cache: layer index {layer_idx} out of range"))
        })
    }
}
