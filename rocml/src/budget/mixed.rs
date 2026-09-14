//! KIVI-style mixed-cache-specific VRAM accounting (issue #2) — split out
//! of `budget/mod.rs` purely for the 400-line file cap: the plain-cache
//! formula (`kv_bytes_per_token`) and the `Budget` type itself stay there,
//! this file only adds the mixed layout's own bytes-per-token formula plus
//! the pre-load GGUF-based estimate that dispatches to either.

use std::path::Path;

use rocml_core::gguf::GgufFile;

use super::{kv_bytes_per_token, Budget};
use crate::cache::KvDtype;
use crate::config::ModelConfig;
use crate::error::RocmlError;
use crate::load_opts::KvCacheMode;
use crate::qwen35::config::{LayerKind, Qwen35Config};

/// VRAM accounting for the KIVI-style mixed KV cache (issue #2):
/// `n_boundary_layers` full-attention layers stay dense fp16 (the plain
/// [`kv_bytes_per_token`] formula); `n_mixed_layers` split into a per-token
/// bulk cost (K per-channel Q8 code + its amortized per-block scale; V
/// per-token Q8/Q4 code + its per-token scale) plus a fixed, ctx-independent
/// overhead (the sink + recent-window fp16 buffers, sized once regardless of
/// context length). `sink_len`/`window_len` are issue #2 leftovers'
/// `LoadOptions::kv_sink`/`kv_window` (defaulting to
/// `crate::kv_quant::{SINK_LEN, WINDOW_LEN}`). Returns `(bytes_per_token,
/// fixed_overhead_bytes)` — feed both into
/// [`Budget::for_loaded_weights_with_overhead`] /
/// [`Budget::for_estimated_weights_with_overhead`].
#[allow(clippy::too_many_arguments)]
pub fn mixed_kv_bytes_per_token(
    _mode: KvCacheMode,
    n_boundary_layers: u32,
    n_mixed_layers: u32,
    n_kv_heads: u32,
    head_dim: u32,
    v_bits: u8,
    sink_len: u32,
    window_len: u32,
) -> (u64, u64) {
    let boundary_per_token =
        kv_bytes_per_token(n_boundary_layers, n_kv_heads, head_dim, KvDtype::F16);

    let v_code_bytes: u64 = if v_bits == 8 {
        head_dim as u64
    } else {
        (head_dim as u64).div_ceil(2)
    };
    // K's per-channel scale (f32, one per (kv_head, channel) per evicted
    // block) amortized over that block's window_len tokens; V's per-token
    // scale (f32) is exact, not amortized.
    let k_scale_amortized_per_token = (head_dim as u64 * 4).div_ceil(window_len as u64);
    let mixed_bulk_per_layer =
        n_kv_heads as u64 * (head_dim as u64 + k_scale_amortized_per_token + v_code_bytes + 4);
    let mixed_per_token = n_mixed_layers as u64 * mixed_bulk_per_layer;

    // Sink + recent-window fp16 K/V buffers: fixed capacity regardless of
    // ctx, so this is a one-time cost per mixed layer, not per token.
    let sink_window_bytes_per_layer =
        n_kv_heads as u64 * (sink_len as u64 + window_len as u64) * head_dim as u64 * 2 * 2;
    let fixed_overhead = n_mixed_layers as u64 * sink_window_bytes_per_layer;

    (boundary_per_token + mixed_per_token, fixed_overhead)
}

/// Cheap pre-load budget estimate for `crate::registry::clamp_ctx`: opens
/// the GGUF just far enough to read the handful of fields the
/// KV-bytes-per-token formula needs (not the full architecture-specific
/// `ModelConfig`/`Qwen35Config` validation `Model::load` performs), and uses
/// the file's on-disk size as a generous proxy for on-device weight bytes.
/// This proxy is close but *not guaranteed* to be an over-estimate: most
/// tensors upload byte-identical to their GGUF bytes (raw-quant) or at the
/// same 2 bytes/element (f16/bf16 source) or *smaller* (an f32 source tensor
/// gets halved on upload, cast to f16) — but `token_embd` is the one
/// exception, always dequantized-and-f16-cast in VRAM regardless of its
/// source dtype (`ModelWeights::load`'s doc comment: "the embedding lookup
/// kernel only reads f16"), so a quantized embedding table's real VRAM
/// footprint can *exceed* its on-disk bytes (measured on Ornith-1.0-9B's
/// Q6_K checkpoint: pre-load estimate 6.85 GiB vs. 8.46 GiB actually
/// resident — see the task report). This is exactly why `Model::load`'s own
/// post-weights-load check (`Budget::for_loaded_weights`) is the
/// *authoritative* one and always runs regardless of what this estimate
/// said — this function is only a fast, no-upload preview so `clamp_ctx`
/// can act before spending seconds loading a model that obviously won't
/// fit, not a guarantee.
///
/// Returns the estimated budget plus the model's own declared
/// `context_length` (a second, independent cap `clamp_ctx` also applies).
///
/// `mode`'s quantized variants (issue #2, ported to the dense `qwen3`
/// architecture per issue #16 — see `crate::cache::DenseAttnCache`) apply
/// the same boundary-layer-skip accounting for both architectures now: the
/// dense arm below treats every layer as full-attention (there's no
/// `layer_kinds` split to filter, unlike `qwen35`) with `n_boundary =
/// block_count.min(2)`, mirroring `DenseAttnCache::new`'s own rule exactly.
pub fn estimate_from_gguf(
    gguf_path: &Path,
    mode: KvCacheMode,
    sink_len: u32,
    window_len: u32,
) -> Result<(Budget, usize), RocmlError> {
    let file_bytes = std::fs::metadata(gguf_path).map(|m| m.len()).unwrap_or(0);
    let gguf = GgufFile::open(gguf_path)?;
    let arch = gguf.get_str("general.architecture")?;
    let (per_token, fixed_overhead, model_ctx_cap) = match arch {
        "qwen3" => {
            let c = ModelConfig::from_gguf(&gguf)?;
            let (per_token, fixed_overhead) = if mode.is_quantized() {
                let n_boundary = c.block_count.min(2);
                let n_mixed = c.block_count.saturating_sub(n_boundary);
                let v_bits: u8 = if mode == KvCacheMode::Q4Mixed { 4 } else { 8 };
                mixed_kv_bytes_per_token(
                    mode,
                    n_boundary,
                    n_mixed,
                    c.head_count_kv,
                    c.head_dim,
                    v_bits,
                    sink_len,
                    window_len,
                )
            } else {
                (
                    kv_bytes_per_token(
                        c.block_count,
                        c.head_count_kv,
                        c.head_dim,
                        mode.dense_dtype(),
                    ),
                    0,
                )
            };
            (per_token, fixed_overhead, c.context_length)
        }
        "qwen35" => {
            let c = Qwen35Config::from_gguf(&gguf)?;
            let n_attn = c
                .layer_kinds
                .iter()
                .filter(|k| **k == LayerKind::FullAttention)
                .count() as u32;
            let (per_token, fixed_overhead) = if mode.is_quantized() {
                let n_boundary = n_attn.min(2);
                let n_mixed = n_attn.saturating_sub(n_boundary);
                let v_bits: u8 = if mode == KvCacheMode::Q4Mixed { 4 } else { 8 };
                mixed_kv_bytes_per_token(
                    mode,
                    n_boundary,
                    n_mixed,
                    c.head_count_kv,
                    c.head_dim,
                    v_bits,
                    sink_len,
                    window_len,
                )
            } else {
                (
                    kv_bytes_per_token(n_attn, c.head_count_kv, c.head_dim, mode.dense_dtype()),
                    0,
                )
            };
            (per_token, fixed_overhead, c.context_length)
        }
        other => {
            return Err(RocmlError::UnsupportedArchitecture {
                found: other.to_string(),
            })
        }
    };
    let device = rocml_hip::Device::new(0)?;
    let mem = device.memory_info()?;
    let budget = Budget::for_estimated_weights_with_overhead(
        mem.total as u64,
        mem.free as u64,
        file_bytes,
        per_token,
        fixed_overhead,
    );
    Ok((budget, model_ctx_cap as usize))
}
