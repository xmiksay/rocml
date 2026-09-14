//! KIVI-style quantized KV cache (issue #2, phase 1 — no rotation): the
//! pure host-side pieces (region layout/eviction bookkeeping, CPU-reference
//! quantize/dequant math) live here, shared by `qwen35::cache`'s
//! `MixedAttnPlane` and by the kernel-vs-CPU-reference unit tests in
//! `rocml-kernels/tests/kv_quant.rs`.

pub mod layout;
pub mod quant_math;
pub mod rotational;

pub use layout::{ChunkWindowSegment, MixedLayout, Region, SINK_LEN, WINDOW_LEN};

use crate::error::RocmlError;

/// HIP's grid-dimension-y/z limit (65535, unlike grid.x's much larger
/// 2^31-1) — `MixedKernels::quantize_evict_v`'s launch (`kernels_mixed.rs`)
/// puts `window_len` directly in `grid.y` (one warp per (kv_head, token) in
/// the evicted window), so a configured `--kv-window` above this would fail
/// the kernel launch itself, not just run slowly.
pub const HIP_GRID_YZ_MAX: u32 = 65_535;

/// Validates a configured `sink_len`/`window_len` pair (issue #2 leftovers,
/// `LoadOptions::kv_sink`/`kv_window`) against every invariant the mixed KV
/// cache's bookkeeping and kernels actually rely on — called once, at load
/// time (`HybridCache::new` and its dense-architecture counterpart), rather
/// than re-derived at each call site.
///
/// - `sink_len >= 1`, `window_len >= 1`: `MixedAttnPlane::new` divides by
///   `window_len` (`bulk_positions.div_ceil(window_len)`) and `MixedLayout`
///   divides by both when computing `region`/`evicted_blocks` — a zero
///   value would panic deep in that arithmetic instead of erroring cleanly
///   here.
/// - `window_len <= HIP_GRID_YZ_MAX`: see that constant's doc comment.
/// - `sink_len + window_len < ctx`: the mixed layout only exists to hold
///   *history* past the always-fp16 sink+window — a context budget that
///   doesn't even reach past them has no bulk region to quantize into, so
///   every mixed layer would silently behave as a plain dense fp16 layer
///   while still paying the mixed cache's fixed sink/window allocation.
///   That's a configuration the caller almost certainly didn't intend, so
///   it's rejected rather than silently accepted.
pub fn validate_sink_window(sink_len: u32, window_len: u32, ctx: usize) -> Result<(), RocmlError> {
    if sink_len == 0 {
        return Err(RocmlError::Config(
            "--kv-sink must be at least 1".to_string(),
        ));
    }
    if window_len == 0 {
        return Err(RocmlError::Config(
            "--kv-window must be at least 1".to_string(),
        ));
    }
    if window_len > HIP_GRID_YZ_MAX {
        return Err(RocmlError::Config(format!(
            "--kv-window {window_len} exceeds {HIP_GRID_YZ_MAX} (HIP's grid.y/z dimension \
             limit — the quantize-evict kernel launches one grid.y row per window position)"
        )));
    }
    let sink_plus_window = sink_len as usize + window_len as usize;
    if sink_plus_window >= ctx {
        return Err(RocmlError::Config(format!(
            "--kv-sink {sink_len} + --kv-window {window_len} = {sink_plus_window} must be \
             less than --ctx {ctx} (otherwise no history ever reaches the quantized bulk \
             region — every mixed layer would just be a fp16 layer with wasted overhead)"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_sink_window_accepts_the_defaults() {
        validate_sink_window(SINK_LEN, WINDOW_LEN, 4096).expect("defaults must validate");
    }

    #[test]
    fn validate_sink_window_accepts_a_non_default_config() {
        validate_sink_window(16, 256, 4096).expect("16/256 must validate");
    }

    #[test]
    fn validate_sink_window_rejects_zero_sink() {
        assert!(validate_sink_window(0, WINDOW_LEN, 4096).is_err());
    }

    #[test]
    fn validate_sink_window_rejects_zero_window() {
        assert!(validate_sink_window(SINK_LEN, 0, 4096).is_err());
    }

    #[test]
    fn validate_sink_window_rejects_window_past_hip_grid_limit() {
        assert!(validate_sink_window(SINK_LEN, HIP_GRID_YZ_MAX + 1, 1_000_000).is_err());
        assert!(validate_sink_window(SINK_LEN, HIP_GRID_YZ_MAX, 1_000_000).is_ok());
    }

    #[test]
    fn validate_sink_window_rejects_sink_plus_window_at_or_past_ctx() {
        assert!(validate_sink_window(32, 128, 160).is_err(), "== ctx");
        assert!(validate_sink_window(32, 128, 100).is_err(), "> ctx");
        assert!(validate_sink_window(32, 128, 161).is_ok(), "< ctx by one");
    }
}
