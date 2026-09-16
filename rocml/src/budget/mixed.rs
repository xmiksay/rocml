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

/// qwen35moe's routed-expert tensors (`ffn_{gate,up,down}_exps.weight`) are
/// never uploaded to the GPU (see `crate::qwen35::weights::moe`'s module
/// doc) — at 18.16 GiB for Ornith-1.5-35B-A3B, treating the whole GGUF file
/// size as a proxy for uploaded weight bytes (the way every other
/// architecture's pre-load estimate does, see this module's own doc
/// comment) would make `registry::clamp_ctx` think a 16GB card can't fit
/// even the model's non-expert weights, silently clamping `ctx` down to a
/// tiny number for a model that in reality leaves most of its VRAM free.
/// This sums only the *non*-expert tensors' on-disk bytes, keeping the same
/// generous-proxy methodology (on-disk bytes as an upload-byte stand-in)
/// scoped to the tensors that actually get uploaded.
fn is_moe_expert_tensor(name: &str) -> bool {
    name.ends_with(".ffn_gate_exps.weight")
        || name.ends_with(".ffn_up_exps.weight")
        || name.ends_with(".ffn_down_exps.weight")
}

fn tensor_disk_bytes(t: &rocml_core::gguf::TensorInfo) -> u64 {
    let Some(elems) = t.n_elements() else {
        return 0;
    };
    let block_elems = t.dtype.block_elements() as u64;
    if block_elems == 0 {
        return 0;
    }
    (elems / block_elems) * t.dtype.block_bytes() as u64
}

/// Fixed VRAM the MoE per-expert staging pool costs (`MoeScratch`'s three
/// `stage_{gate,up,down}` buffers) — a few hundred KB to ~1MB each on this
/// checkpoint's shape, rounded generously up.
const MOE_STAGING_HEADROOM_BYTES: u64 = 8 * 1024 * 1024;

fn qwen35moe_non_expert_weight_bytes(gguf: &GgufFile) -> u64 {
    gguf.tensors()
        .iter()
        .filter(|t| !is_moe_expert_tensor(&t.name))
        .map(tensor_disk_bytes)
        .sum()
}

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

/// GPU rewind points `qwen35::forward::Model` keeps resident (see
/// `qwen35::forward::rewind`: one for the render-stable boundary, one for
/// the end of turn).
pub const REWIND_SLOTS: u64 = 2;

/// Fixed VRAM the hybrid model's GPU rewind points cost, so `Model::load`'s
/// budget check reserves it up front instead of the first save discovering
/// there was no room: `REWIND_SLOTS` x (every GDN layer's conv + recurrence
/// state in f32, plus every mixed-KV layer's fp16 recent-window K/V pair —
/// dense planes need no rewind storage at all, see
/// `qwen35::cache::rewind`). `n_mixed_layers` is 0 for a dense KV mode.
pub fn rewind_points_bytes(cfg: &Qwen35Config, n_mixed_layers: u32, window_len: u32) -> u64 {
    let n_gdn = cfg
        .layer_kinds
        .iter()
        .filter(|k| **k == LayerKind::LinearAttention)
        .count() as u64;
    let g = &cfg.gdn;
    let gdn_floats = g.conv_dim as u64 * (g.conv_kernel as u64).saturating_sub(1)
        + g.num_v_heads as u64 * g.head_k_dim as u64 * g.head_v_dim as u64;
    let window_bytes_per_layer =
        2 * cfg.head_count_kv as u64 * window_len as u64 * cfg.head_dim as u64 * 2;
    REWIND_SLOTS * (n_gdn * gdn_floats * 4 + n_mixed_layers as u64 * window_bytes_per_layer)
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
    // Every architecture but qwen35moe uploads (an approximation of) the
    // whole file as weights, so `file_bytes` is already the right proxy —
    // see `qwen35moe_non_expert_weight_bytes`'s own doc comment for why
    // that one architecture needs a different number.
    let weights_bytes = if arch == "qwen35moe" {
        qwen35moe_non_expert_weight_bytes(&gguf)
    } else {
        file_bytes
    };
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
        "qwen35" | "qwen35moe" => {
            let c = Qwen35Config::from_gguf(&gguf)?;
            let n_attn = c
                .layer_kinds
                .iter()
                .filter(|k| **k == LayerKind::FullAttention)
                .count() as u32;
            let n_boundary = n_attn.min(2);
            let n_mixed = if mode.is_quantized() {
                n_attn.saturating_sub(n_boundary)
            } else {
                0
            };
            let (per_token, fixed_overhead) = if mode.is_quantized() {
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
            let fixed_overhead = fixed_overhead
                + rewind_points_bytes(&c, n_mixed, window_len)
                + if c.moe.is_some() {
                    MOE_STAGING_HEADROOM_BYTES
                } else {
                    0
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
        weights_bytes,
        per_token,
        fixed_overhead,
    );
    Ok((budget, model_ctx_cap as usize))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen35::config::GdnConfig;

    /// Ornith-1.0-9B's shape: 24 GDN layers (conv_dim 8192, kernel 4, 32 v
    /// heads x 128 x 128) and, under a quantized mode, 6 mixed layers with
    /// 4 kv heads x 256 head_dim windows of 128.
    fn ornith_like() -> Qwen35Config {
        let mut layer_kinds = vec![LayerKind::LinearAttention; 32];
        for i in (3..32).step_by(4) {
            layer_kinds[i] = LayerKind::FullAttention;
        }
        Qwen35Config {
            block_count: 32,
            embedding_length: 4096,
            feed_forward_length: 12288,
            head_count: 16,
            head_count_kv: 4,
            head_dim: 256,
            rope_freq_base: 10_000_000.0,
            rope_dim_count: 64,
            rms_eps: 1e-6,
            context_length: 262_144,
            vocab_size: 151_936,
            layer_kinds,
            gdn: GdnConfig {
                conv_kernel: 4,
                num_k_heads: 16,
                num_v_heads: 32,
                head_k_dim: 128,
                head_v_dim: 128,
                key_dim: 2048,
                value_dim: 4096,
                conv_dim: 8192,
            },
            moe: None,
        }
    }

    #[test]
    fn rewind_points_bytes_matches_hand_count() {
        let cfg = ornith_like();
        let gdn_per_layer = (8192 * 3 + 32 * 128 * 128) * 4;
        let dense = rewind_points_bytes(&cfg, 0, 128);
        assert_eq!(dense, REWIND_SLOTS * 24 * gdn_per_layer);
        // ~53 MiB per slot on this shape, as the design notes claim.
        assert!((50 << 20..55 << 20).contains(&(dense / REWIND_SLOTS)));
        let window_per_layer = 2 * 4 * 128 * 256 * 2;
        assert_eq!(
            rewind_points_bytes(&cfg, 6, 128),
            dense + REWIND_SLOTS * 6 * window_per_layer
        );
    }
}
