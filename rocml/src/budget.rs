//! VRAM budget math for the KV cache (issue #3): pure, unit-tested
//! arithmetic, plus the two integration points that use it —
//! `crate::registry::clamp_ctx` (a cheap pre-load heuristic using the GGUF
//! file's on-disk size as a generous proxy for on-device weight bytes,
//! since the real number is only known once `LinearWeight::load` actually
//! uploads them) and `Model::load`'s own authoritative post-load check
//! (using the *real* `hipMemGetInfo` free-byte count once weights are
//! resident, via [`Budget::for_loaded_weights`]).
//!
//! Reference point from issue #3's design review, and this module's own
//! unit tests: Ornith-1.0-9B's 8 full-attention layers x 8 KV heads x 256
//! head_dim gives exactly the issue's "64 KB/token fp16 = 8 layers x 8
//! kv-heads x 512 x 2B" figure (512 = 2*head_dim, K+V); at 150K tokens that
//! matches the issue's "~9.8 GB fp16" reference point.

use std::path::Path;

use rocml_core::gguf::GgufFile;

use crate::cache::KvDtype;
use crate::config::ModelConfig;
use crate::error::RocmlError;
use crate::qwen35::config::{LayerKind, Qwen35Config};

/// Fixed scratch/activation headroom reserved on top of weights + KV bytes.
/// Measured-generous rather than exact: `Scratch`/`ChunkScratch` (the
/// forward pass's only other resident allocations) are a few tens of MB
/// even at the largest chunk cap this codebase uses (`CHUNK_CAP` = 512
/// tokens) — issue #3's own "never materialize logits for a whole prefill
/// chunk" rule keeps the single biggest transient (`forward_chunk` only
/// computing the last row's logits) from scaling with context at all. 768
/// MiB sits comfortably above every measured `Scratch`/`ChunkScratch`
/// footprint on the largest model this registry serves (Ornith-1.0-9B,
/// head_dim 256, vocab ~150K), with room left for driver overhead and
/// allocator fragmentation.
pub const ACTIVATION_HEADROOM_BYTES: u64 = 768 * 1024 * 1024;

/// Above this fraction of total VRAM, callers should warn even when the
/// requested context still technically fits.
pub const HIGH_USAGE_WARN_FRACTION: f64 = 0.90;

/// Bytes for one token's K+V across `n_cache_layers` layers holding growing
/// KV state (dense Qwen3: every layer; qwen35 hybrid: only the
/// full-attention layers — GDN state is O(1), see `HybridCache`) at
/// `n_kv_heads` heads x `head_dim` each, in `dtype`.
pub fn kv_bytes_per_token(
    n_cache_layers: u32,
    n_kv_heads: u32,
    head_dim: u32,
    dtype: KvDtype,
) -> u64 {
    let bytes_per_elem: u64 = match dtype {
        KvDtype::F16 => 2,
        KvDtype::F32 => 4,
    };
    // 2x for K and V, each its own [n_kv_heads, head_dim] plane per layer.
    2 * n_cache_layers as u64 * n_kv_heads as u64 * head_dim as u64 * bytes_per_elem
}

/// A VRAM budget snapshot: how much is free (net of weights, however they
/// were accounted for — see the two constructors), how much weights take
/// (for display only), how much one token of KV costs, and the fixed
/// activation headroom — enough to answer "does context length N fit" and
/// "what's the largest N that does" without re-deriving any of it at each
/// call site. Both constructors normalize `free_bytes` to *already exclude*
/// weights, so every method below has exactly one code path regardless of
/// which one built this `Budget`.
#[derive(Debug, Clone, Copy)]
pub struct Budget {
    pub total_bytes: u64,
    /// Free VRAM bytes, net of weights (already subtracted by whichever
    /// constructor built this `Budget`).
    pub free_bytes: u64,
    /// Weights bytes, for the human-readable breakdown only — already
    /// folded into `free_bytes`, never subtracted again.
    pub weights_bytes: u64,
    pub kv_bytes_per_token: u64,
}

impl Budget {
    /// Pre-load estimate: `free_bytes_before_weights` is `hipMemGetInfo`'s
    /// free count *before* any weights are uploaded, and `weights_bytes` is
    /// a generous proxy (the GGUF file's on-disk size — see
    /// `crate::registry::clamp_ctx`'s doc comment for why that's a safe
    /// over-estimate for every quant this registry serves).
    pub fn for_estimated_weights(
        total_bytes: u64,
        free_bytes_before_weights: u64,
        weights_bytes: u64,
        kv_bytes_per_token: u64,
    ) -> Self {
        Self {
            total_bytes,
            free_bytes: free_bytes_before_weights.saturating_sub(weights_bytes),
            weights_bytes,
            kv_bytes_per_token,
        }
    }

    /// Authoritative post-load check: `free_bytes` is `hipMemGetInfo`'s free
    /// count *after* weights are already resident, so it already excludes
    /// them — `weights_bytes` is derived (`total - free`) purely for the
    /// breakdown's display.
    pub fn for_loaded_weights(total_bytes: u64, free_bytes: u64, kv_bytes_per_token: u64) -> Self {
        Self {
            total_bytes,
            free_bytes,
            weights_bytes: total_bytes.saturating_sub(free_bytes),
            kv_bytes_per_token,
        }
    }

    fn usable_bytes(&self) -> u64 {
        self.free_bytes.saturating_sub(ACTIVATION_HEADROOM_BYTES)
    }

    /// Largest context length whose KV cache fits in `usable_bytes`.
    pub fn max_ctx(&self) -> usize {
        if self.kv_bytes_per_token == 0 {
            return usize::MAX;
        }
        (self.usable_bytes() / self.kv_bytes_per_token) as usize
    }

    pub fn kv_bytes(&self, ctx: usize) -> u64 {
        self.kv_bytes_per_token * ctx as u64
    }

    pub fn fits(&self, ctx: usize) -> bool {
        self.kv_bytes(ctx) + ACTIVATION_HEADROOM_BYTES <= self.free_bytes
    }

    /// Fraction of total VRAM predicted to be in use at `ctx` (weights,
    /// already reflected in `free_bytes`/`total_bytes`, plus this budget's
    /// own KV + activation share).
    pub fn usage_fraction(&self, ctx: usize) -> f64 {
        if self.total_bytes == 0 {
            return 0.0;
        }
        let used = self.total_bytes.saturating_sub(self.free_bytes)
            + self.kv_bytes(ctx)
            + ACTIVATION_HEADROOM_BYTES;
        used as f64 / self.total_bytes as f64
    }

    /// Human-readable breakdown for warnings/errors — what fits, and what
    /// to try instead (a smaller `--ctx` or a quantized `--kv-cache` mode).
    pub fn breakdown(&self, ctx: usize) -> String {
        format!(
            "VRAM: {:.2} GiB total, {:.2} GiB free, ~{:.2} GiB weights, {:.2} GiB KV @ ctx {} \
             ({:.1} KiB/token), {:.2} GiB activation headroom -> {:.1}% of total \
             (max ctx at this KV dtype: {})",
            gib(self.total_bytes),
            gib(self.free_bytes),
            gib(self.weights_bytes),
            gib(self.kv_bytes(ctx)),
            ctx,
            self.kv_bytes_per_token as f64 / 1024.0,
            gib(ACTIVATION_HEADROOM_BYTES),
            self.usage_fraction(ctx) * 100.0,
            self.max_ctx(),
        )
    }
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

/// Cheap pre-load budget estimate for `crate::registry::clamp_ctx`: opens
/// the GGUF just far enough to read the handful of fields the
/// KV-bytes-per-token formula needs (not the full architecture-specific
/// `ModelConfig`/`Qwen35Config` validation `Model::load` performs), and uses
/// the file's on-disk size as a generous proxy for on-device weight bytes.
/// That proxy never under-estimates for the quant configs this registry
/// serves: a raw-quant `LinearWeight` uploads byte-identical to its GGUF
/// bytes, an f16/bf16 source tensor uploads at the same 2 bytes/element,
/// and an f32 source tensor gets *halved* on upload (cast to f16) — so file
/// size is always >= real weight bytes, making this estimate safe in the
/// "clamp more than strictly necessary" direction. `Model::load`'s own
/// post-weights-load check (`Budget::for_loaded_weights`) is the
/// authoritative one; this is only a fast, no-upload preview so
/// `clamp_ctx` can act before spending seconds loading a model that won't
/// fit anyway.
///
/// Returns the estimated budget plus the model's own declared
/// `context_length` (a second, independent cap `clamp_ctx` also applies).
pub fn estimate_from_gguf(
    gguf_path: &Path,
    kv_dtype: KvDtype,
) -> Result<(Budget, usize), RocmlError> {
    let file_bytes = std::fs::metadata(gguf_path).map(|m| m.len()).unwrap_or(0);
    let gguf = GgufFile::open(gguf_path)?;
    let arch = gguf.get_str("general.architecture")?;
    let (n_cache_layers, n_kv_heads, head_dim, model_ctx_cap) = match arch {
        "qwen3" => {
            let c = ModelConfig::from_gguf(&gguf)?;
            (c.block_count, c.head_count_kv, c.head_dim, c.context_length)
        }
        "qwen35" => {
            let c = Qwen35Config::from_gguf(&gguf)?;
            let n_attn = c
                .layer_kinds
                .iter()
                .filter(|k| **k == LayerKind::FullAttention)
                .count() as u32;
            (n_attn, c.head_count_kv, c.head_dim, c.context_length)
        }
        other => {
            return Err(RocmlError::UnsupportedArchitecture {
                found: other.to_string(),
            })
        }
    };
    let per_token = kv_bytes_per_token(n_cache_layers, n_kv_heads, head_dim, kv_dtype);
    let device = rocml_hip::Device::new(0)?;
    let mem = device.memory_info()?;
    let budget =
        Budget::for_estimated_weights(mem.total as u64, mem.free as u64, file_bytes, per_token);
    Ok((budget, model_ctx_cap as usize))
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    #[test]
    fn kv_bytes_per_token_matches_issue_3_ornith_reference() {
        // Issue #3: "64 KB/token fp16 = 8 layers x 8 kv-heads x 512 x 2B"
        // (512 = 2*head_dim(256), i.e. K+V combined) for Ornith-1.0-9B's 8
        // full-attention layers.
        let bytes = kv_bytes_per_token(8, 8, 256, KvDtype::F16);
        assert_eq!(bytes, 64 * 1024);
    }

    #[test]
    fn kv_bytes_per_token_f32_is_double_f16() {
        let f16 = kv_bytes_per_token(8, 8, 256, KvDtype::F16);
        let f32 = kv_bytes_per_token(8, 8, 256, KvDtype::F32);
        assert_eq!(f32, f16 * 2);
    }

    #[test]
    fn max_ctx_matches_issue_3_150k_fp16_reference() {
        // Issue #3: "150K ctx = ~9.8 GB fp16" at Ornith's 64 KB/token. Give
        // this budget enough free room (11GB, i.e. a ~5GB weights load on a
        // 16GB card) that 150K tokens' ~9.8GB KV genuinely fits, to check
        // max_ctx's arithmetic rather than just its byte formula.
        let per_token = kv_bytes_per_token(8, 8, 256, KvDtype::F16);
        let budget = Budget::for_loaded_weights(16 * GIB, 11 * GIB, per_token);
        // 9.8 GB decimal ~= 9.8e9 bytes; confirm 150K tokens' worth is
        // within a couple percent of that, and that max_ctx round-trips.
        let bytes_150k = budget.kv_bytes(150_000);
        let expected = 150_000u64 * per_token;
        assert_eq!(bytes_150k, expected);
        assert!((expected as f64 - 9.83e9).abs() / 9.83e9 < 0.01);
        assert!(
            budget.max_ctx() >= 150_000,
            "expected 150K to fit in ~11GB usable"
        );
    }

    #[test]
    fn fits_and_max_ctx_agree() {
        let budget = Budget::for_loaded_weights(16 * GIB, 8 * GIB, 64 * 1024);
        let max = budget.max_ctx();
        assert!(budget.fits(max));
        assert!(!budget.fits(max + 1));
    }

    #[test]
    fn zero_free_bytes_yields_zero_max_ctx() {
        let budget = Budget::for_loaded_weights(16 * GIB, 0, 64 * 1024);
        assert_eq!(budget.max_ctx(), 0);
        assert!(!budget.fits(1));
    }

    #[test]
    fn usage_fraction_is_monotonic_in_ctx() {
        let budget = Budget::for_loaded_weights(16 * GIB, 8 * GIB, 64 * 1024);
        assert!(budget.usage_fraction(1000) < budget.usage_fraction(100_000));
    }

    #[test]
    fn for_estimated_weights_subtracts_weights_from_free() {
        // Pre-load: 16GB total, 15GB free (nothing loaded yet), a 7GB
        // weights estimate -> max_ctx should match a for_loaded_weights
        // budget built from the already-net free count (15GB - 7GB = 8GB).
        let pre = Budget::for_estimated_weights(16 * GIB, 15 * GIB, 7 * GIB, 64 * 1024);
        let post = Budget::for_loaded_weights(16 * GIB, 8 * GIB, 64 * 1024);
        assert_eq!(pre.max_ctx(), post.max_ctx());
        assert_eq!(pre.free_bytes, 8 * GIB);
    }
}
