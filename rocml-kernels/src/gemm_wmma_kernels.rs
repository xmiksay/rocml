//! Batched prefill-path `gemm_xwt_*` code objects for GGUF-quantized
//! weights (scalar, WMMA default/narrow, WMMA split-K) — split out of
//! `lib.rs` purely for the 400-line file cap once the split-K round added a
//! fourth kernel family. Re-exported flat via `lib.rs`'s `pub use` so every
//! existing `rocml_kernels::GEMM_XWT_*` call site is unaffected.

/// `kernels/gemm_xwt_quant.hip`: batched prefill-path linear layer for
/// GGUF-quantized weights, `out[rows,m] = X[rows,n] * dequant(W)^T` — the
/// batched sibling of `gemv_q*` (which stays the fast path for rows==1
/// decode). One weight row shared by 8 output rows per workgroup; launch
/// with block = (32, 8, 1) and `256 * sizeof(f32)` bytes of dynamic shared
/// memory (see the kernel source's module doc for the tiling design). All
/// four dtypes share one code object.
pub const GEMM_XWT_Q8_0_HSACO: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/gemm_xwt_quant.hsaco"));
pub const GEMM_XWT_Q8_0_KERNEL: &str = "gemm_xwt_q8_0";
pub const GEMM_XWT_Q4_K_HSACO: &[u8] = GEMM_XWT_Q8_0_HSACO;
pub const GEMM_XWT_Q4_K_KERNEL: &str = "gemm_xwt_q4_k";
pub const GEMM_XWT_Q5_K_HSACO: &[u8] = GEMM_XWT_Q8_0_HSACO;
pub const GEMM_XWT_Q5_K_KERNEL: &str = "gemm_xwt_q5_k";
pub const GEMM_XWT_Q6_K_HSACO: &[u8] = GEMM_XWT_Q8_0_HSACO;
pub const GEMM_XWT_Q6_K_KERNEL: &str = "gemm_xwt_q6_k";

/// `kernels/gemm_xwt_quant_wmma.hip`: WMMA matrix-core version of
/// `gemm_xwt_q*` (issue #6's follow-up) — same `out[rows,m] = X[rows,n] *
/// dequant(W)^T` contract, but feeds gfx11's wave32 matrix unit instead of
/// scalar FMA. Launch with block = (32, 8, 1) and `(TILE_ROWS + TILE_M) *
/// K_STAGE * sizeof(f16)` bytes of dynamic shared memory (see the kernel
/// source's module doc for the tile sizes and fragment layout); grid =
/// `(ceil(m/TILE_M), ceil(rows/TILE_ROWS), 1)`. Requires `m` and `n`
/// multiples of 16 and `rows >= 16` to be worth it — the scalar
/// `gemm_xwt_q*` kernels above stay the dispatch fallback otherwise. All
/// four dtypes share one code object.
pub const GEMM_XWT_WMMA_Q8_0_HSACO: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/gemm_xwt_quant_wmma.hsaco"));
pub const GEMM_XWT_WMMA_Q8_0_KERNEL: &str = "gemm_xwt_wmma_q8_0";
pub const GEMM_XWT_WMMA_Q4_K_HSACO: &[u8] = GEMM_XWT_WMMA_Q8_0_HSACO;
pub const GEMM_XWT_WMMA_Q4_K_KERNEL: &str = "gemm_xwt_wmma_q4_k";
pub const GEMM_XWT_WMMA_Q5_K_HSACO: &[u8] = GEMM_XWT_WMMA_Q8_0_HSACO;
pub const GEMM_XWT_WMMA_Q5_K_KERNEL: &str = "gemm_xwt_wmma_q5_k";
pub const GEMM_XWT_WMMA_Q6_K_HSACO: &[u8] = GEMM_XWT_WMMA_Q8_0_HSACO;
pub const GEMM_XWT_WMMA_Q6_K_KERNEL: &str = "gemm_xwt_wmma_q6_k";

/// `kernels/gemm_xwt_quant_wmma_narrow.hip`: same contract as the
/// `GEMM_XWT_WMMA_Q*` kernels above, but built with the narrower
/// TILE_M=64/WARPS_M=8/WARPS_N=2 tile config (the original 1x2-fragment
/// layout, before the per-wave-efficiency round's TILE_M=128 rewrite) —
/// see that file's module doc for the shape-aware dispatch round's
/// measured `m`-crossover. `rocml/src/forward/kernels_quant.rs` dispatches
/// here for `m < 2048`.
pub const GEMM_XWT_WMMA_Q8_0_NARROW_HSACO: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/gemm_xwt_quant_wmma_narrow.hsaco"
));
pub const GEMM_XWT_WMMA_Q8_0_NARROW_KERNEL: &str = "gemm_xwt_wmma_q8_0_narrow";
pub const GEMM_XWT_WMMA_Q4_K_NARROW_HSACO: &[u8] = GEMM_XWT_WMMA_Q8_0_NARROW_HSACO;
pub const GEMM_XWT_WMMA_Q4_K_NARROW_KERNEL: &str = "gemm_xwt_wmma_q4_k_narrow";
pub const GEMM_XWT_WMMA_Q5_K_NARROW_HSACO: &[u8] = GEMM_XWT_WMMA_Q8_0_NARROW_HSACO;
pub const GEMM_XWT_WMMA_Q5_K_NARROW_KERNEL: &str = "gemm_xwt_wmma_q5_k_narrow";
pub const GEMM_XWT_WMMA_Q6_K_NARROW_HSACO: &[u8] = GEMM_XWT_WMMA_Q8_0_NARROW_HSACO;
pub const GEMM_XWT_WMMA_Q6_K_NARROW_KERNEL: &str = "gemm_xwt_wmma_q6_k_narrow";

/// `kernels/gemm_xwt_quant_wmma_micro.hip` (qwen35moe M4, lever 1): same
/// contract as the `GEMM_XWT_WMMA_Q*` kernels above, built with the
/// micro tile config (`TILE_ROWS`=16/`TILE_M`=64/`WARPS_M`=1/`WARPS_N`=4) for
/// MoE's grouped-by-expert batched GEMM, whose per-expert row groups
/// (typically 2-16 rows on a 512-token chunk) are far below the default/
/// narrow configs' `rows >= 128` floor. Opt-in only (`allow_micro` in
/// `rocml/src/forward/kernels_quant_dispatch.rs`) — see that module's doc.
pub const GEMM_XWT_WMMA_Q8_0_MICRO_HSACO: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/gemm_xwt_quant_wmma_micro.hsaco"));
pub const GEMM_XWT_WMMA_Q8_0_MICRO_KERNEL: &str = "gemm_xwt_wmma_q8_0_micro";
pub const GEMM_XWT_WMMA_Q4_K_MICRO_HSACO: &[u8] = GEMM_XWT_WMMA_Q8_0_MICRO_HSACO;
pub const GEMM_XWT_WMMA_Q4_K_MICRO_KERNEL: &str = "gemm_xwt_wmma_q4_k_micro";
pub const GEMM_XWT_WMMA_Q5_K_MICRO_HSACO: &[u8] = GEMM_XWT_WMMA_Q8_0_MICRO_HSACO;
pub const GEMM_XWT_WMMA_Q5_K_MICRO_KERNEL: &str = "gemm_xwt_wmma_q5_k_micro";
pub const GEMM_XWT_WMMA_Q6_K_MICRO_HSACO: &[u8] = GEMM_XWT_WMMA_Q8_0_MICRO_HSACO;
pub const GEMM_XWT_WMMA_Q6_K_MICRO_KERNEL: &str = "gemm_xwt_wmma_q6_k_micro";

/// `kernels/gemm_xwt_wmma_splitk.hip` (issue #6's split-K follow-up, for
/// narrow-grid shapes like ffn-down's `m=hidden, n=intermediate`): same
/// default `TILE_M`=128 tile config as `GEMM_XWT_WMMA_Q*` above, but a third
/// `blockIdx.z` grid dimension splits the K reduction into `num_splits`
/// independent partial sums, written to `[num_splits, rows, m]` scratch
/// instead of the final `[rows, m]` output — `GEMM_SPLITK_REDUCE_F32` sums
/// them. `rocml/src/forward/kernels_quant_dispatch.rs` dispatches here only
/// when the plain default-tile grid is too narrow to fill the GPU and `n`
/// divides evenly by `num_splits * K_STAGE`. All four dtypes share one code
/// object.
pub const GEMM_XWT_WMMA_SPLITK_Q8_0_HSACO: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/gemm_xwt_wmma_splitk.hsaco"));
pub const GEMM_XWT_WMMA_SPLITK_Q8_0_KERNEL: &str = "gemm_xwt_wmma_splitk_q8_0";
pub const GEMM_XWT_WMMA_SPLITK_Q4_K_HSACO: &[u8] = GEMM_XWT_WMMA_SPLITK_Q8_0_HSACO;
pub const GEMM_XWT_WMMA_SPLITK_Q4_K_KERNEL: &str = "gemm_xwt_wmma_splitk_q4_k";
pub const GEMM_XWT_WMMA_SPLITK_Q5_K_HSACO: &[u8] = GEMM_XWT_WMMA_SPLITK_Q8_0_HSACO;
pub const GEMM_XWT_WMMA_SPLITK_Q5_K_KERNEL: &str = "gemm_xwt_wmma_splitk_q5_k";
pub const GEMM_XWT_WMMA_SPLITK_Q6_K_HSACO: &[u8] = GEMM_XWT_WMMA_SPLITK_Q8_0_HSACO;
pub const GEMM_XWT_WMMA_SPLITK_Q6_K_KERNEL: &str = "gemm_xwt_wmma_splitk_q6_k";

/// `kernels/gemm_splitk_reduce.hip`: deterministic ascending-order sum of
/// the split-K GEMM's `[num_splits, rows, m]` partial buffer into the real
/// `[rows, m]` output. dtype-independent (f32 in, f32 out) — shared by
/// every `GEMM_XWT_WMMA_SPLITK_Q*` dtype.
pub const GEMM_SPLITK_REDUCE_F32_HSACO: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/gemm_splitk_reduce.hsaco"));
pub const GEMM_SPLITK_REDUCE_F32_KERNEL: &str = "gemm_splitk_reduce_f32";
