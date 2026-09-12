//! KIVI-style quantized KV cache (issue #2, phase 1 — no rotation): the
//! pure host-side pieces (region layout/eviction bookkeeping, CPU-reference
//! quantize/dequant math) live here, shared by `qwen35::cache`'s
//! `MixedAttnPlane` and by the kernel-vs-CPU-reference unit tests in
//! `rocml-kernels/tests/kv_quant.rs`.

pub mod layout;
pub mod quant_math;

pub use layout::{ChunkWindowSegment, MixedLayout, Region, SINK_LEN, WINDOW_LEN};
