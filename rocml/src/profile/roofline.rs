//! Single source of truth for the gfx1101 (RX 7800 XT, RDNA3, wave32)
//! hardware roofline constants used by [`super::report`]'s per-op bound-ness
//! math (issue #5). Every number here is either a measured figure from this
//! repo's own benchmarks or a vendor-published theoretical peak — never a
//! guess — and each constant says which.

/// Achievable HBM bandwidth, measured on this dev box with `rocml-cli bench`
/// against decode's memory-bound `gemv_q*`/`attn_decode` kernels at large
/// KV-cache depth (where those kernels' own `%BW-roofline` column saturates
/// near this figure — see `docs/profiling.md`'s worked example). Consistent
/// with the RX 7800 XT's advertised 624 GB/s (256-bit bus, 19.5 Gbps
/// effective GDDR6), so this is "vendor peak measured as reachable in
/// practice", not a separate lower estimate.
pub const BW_ROOFLINE_BYTES_PER_SEC: f64 = 624.0e9;

/// Theoretical peak f16 throughput of gfx1101's wave32 WMMA matrix cores
/// (`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`), from AMD's published
/// RDNA3 ISA specs for the RX 7800 XT (60 CUs x 2 WMMA-capable SIMDs x the
/// `16x16x16` fragment's FLOPs/cycle x clock). This is the ceiling the
/// `gemm_xwt_wmma_q*` kernels chase (see `../qwen35/forward`'s chunked
/// prefill doc comments) — not the number ordinary scalar-FMA kernels can
/// realistically be judged against, which is [`PRACTICAL_FLOPS_PER_SEC`]
/// below.
pub const WMMA_PEAK_FLOPS_PER_SEC: f64 = 75.0e12;

/// Practical FLOP/s ceiling this codebase's non-WMMA (scalar-FMA and mixed
/// scalar/vector) kernels actually reach at their best-measured shapes —
/// the midpoint of a measured 15-20 TFLOP/s range (see the WMMA follow-up's
/// own before/after numbers in `../qwen35/forward`'s module docs: scalar
/// GEMM topped out around 1.7-1.8 TFLOP/s per-kernel, but the *aggregate*
/// FLOP-bound ceiling across this repo's mix of matvec/norm/attention scalar
/// kernels — the ones WMMA doesn't touch — lands in this band). Used as the
/// FLOP-roofline denominator for every op, WMMA-eligible or not: an op that
/// already beats this (because it *did* dispatch to WMMA) shows up with
/// negative "wasted" time in [`super::report::AggRow`] rather than a
/// deceptively saturated 100%+ against the wrong ceiling — that's a
/// deliberate, documented signal, not a bug (see `docs/prefill-gap-analysis.md`).
pub const PRACTICAL_FLOPS_PER_SEC: f64 = 17.5e12;
