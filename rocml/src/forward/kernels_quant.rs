//! Fused dequant-GEMV kernels for GGUF-quantized linear weights: the model
//! keeps Q8_0/Q4_K/Q5_K/Q6_K tensors in their raw ggml block layout in VRAM,
//! and these kernels dequantize in-register during the matvec instead of
//! paying a CPU dequant + f16 upload at load time. See `rocml-kernels`'s
//! `gemv_q*` sources for the exact block layouts; `n` must be a multiple of
//! 32 for Q8_0 or 256 for the K-quants (enforced by `LinearWeight::load`'s
//! loader policy, not here).
//!
//! Split out of `kernels.rs` (which already sits close to the workspace's
//! 400-line file cap) into its own small kernel-owning struct, embedded as a
//! field of `Kernels` and reached through `Kernels::gemv_quant`.

use std::mem::size_of;

use rocml_core::quant::GgmlDType;
use rocml_hip::{kernel_params, LaunchConfig, Module};

use super::kernels::{load, DevPtr, REDUCE_BLOCK};
use super::kernels_mmq::{MmqKernels, MmqScratch};
use crate::error::RocmlError;

/// Reduction elements staged into LDS per outer iteration by every
/// `gemm_xwt_q*` kernel — fixes the dynamic shared memory request
/// (`TILE_ELEMS * sizeof(f32)`) regardless of dtype (see
/// `kernels/gemm_xwt_quant.hip`'s module doc).
const GEMM_QUANT_TILE_ELEMS: u32 = 256;
/// Warps per `gemm_xwt_q*` workgroup — fixes the launch's block.y. Must
/// equal `GEMM_QUANT_TILE_ELEMS / 32` (the kernel's cooperative dequant
/// stages one element per thread per outer iteration in a single pass).
const GEMM_QUANT_WARPS_PER_BLOCK: u32 = 8;
/// Output rows one warp carries in registers per weight tile (mirrors the
/// kernel's `ROWS_PER_WARP`).
const GEMM_QUANT_ROWS_PER_WARP: u32 = 8;
/// Output rows one `gemm_xwt_q*` workgroup shares a weight row across —
/// fixes the launch's grid.y.
const GEMM_QUANT_TILE_ROWS: u32 = GEMM_QUANT_WARPS_PER_BLOCK * GEMM_QUANT_ROWS_PER_WARP;

/// Output rows (`X` rows) a `gemm_xwt_wmma_q*` workgroup tile covers —
/// mirrors the kernel's `TILE_ROWS` (fixes the launch's grid.y and doubles
/// as the dispatch threshold below: it exactly matches
/// `qwen35::forward::chunk_forward::PREFILL_CHUNK_SIZE`, so every full
/// chunked-prefill chunk fills one row-tile exactly and only a short last
/// chunk falls back to the scalar kernel).
const GEMM_WMMA_TILE_ROWS: u32 = 128;
/// Output columns (`W` rows) a `gemm_xwt_wmma_q*` workgroup tile covers —
/// mirrors the kernel's `TILE_M` (fixes the launch's grid.x). 128 since the
/// per-wave-efficiency round — see `gemm_xwt_quant_wmma.hip`'s module doc.
const GEMM_WMMA_TILE_M: u32 = 128;
/// Reduction elements staged into LDS per outer iteration — mirrors the
/// kernel's `K_STAGE`; combined with `GEMM_WMMA_LDS_PAD` to size the LDS
/// request (the kernel's per-row stride is padded past `K_STAGE`).
const GEMM_WMMA_K_STAGE: u32 = 16;
/// LDS row-stride padding — must match the kernel's `LDS_PAD` (bank-conflict elimination).
const GEMM_WMMA_LDS_PAD: u32 = 8;
/// Warps per `gemm_xwt_wmma_q*` workgroup — fixes the launch's block.y.
const GEMM_WMMA_WARPS_PER_BLOCK: u32 = 16;

/// Output rows one `gemv_q*` workgroup owns — mirrors every `gemv_q*.hip`
/// kernel's `ROWS_PER_WG` (fixes the launch's `grid.x` and dynamic shared
/// memory request). See `gemv_q4_k.hip`'s module doc for why: at the
/// smallest served `m` (4096), one row per workgroup left each workgroup's
/// whole lifetime as a single iteration of uncovered memory latency, and 2
/// independent per-lane weight loads per iteration gives the scheduler
/// something to overlap that latency with instead — measured (not assumed)
/// as the best of {1,2,4,8} across both the isolated kernel microbench and
/// real end-to-end decode tok/s; 4 and 8 look better in isolation on the
/// largest shapes but cost more on the smallest ones (register pressure)
/// and lose overall.
const GEMV_ROWS_PER_WG: u32 = 2;

pub(crate) struct QuantKernels {
    _mod_q8_0: Module,
    q8_0_fn: rocml_hip::Function,
    _mod_q4_k: Module,
    q4_k_fn: rocml_hip::Function,
    _mod_q5_k: Module,
    q5_k_fn: rocml_hip::Function,
    _mod_q6_k: Module,
    q6_k_fn: rocml_hip::Function,
    _mod_gemm_q8_0: Module,
    gemm_q8_0_fn: rocml_hip::Function,
    _mod_gemm_q4_k: Module,
    gemm_q4_k_fn: rocml_hip::Function,
    _mod_gemm_q5_k: Module,
    gemm_q5_k_fn: rocml_hip::Function,
    _mod_gemm_q6_k: Module,
    gemm_q6_k_fn: rocml_hip::Function,
    _mod_gemm_wmma_q8_0: Module,
    gemm_wmma_q8_0_fn: rocml_hip::Function,
    _mod_gemm_wmma_q4_k: Module,
    gemm_wmma_q4_k_fn: rocml_hip::Function,
    _mod_gemm_wmma_q5_k: Module,
    gemm_wmma_q5_k_fn: rocml_hip::Function,
    _mod_gemm_wmma_q6_k: Module,
    gemm_wmma_q6_k_fn: rocml_hip::Function,
    mmq: MmqKernels,
    /// Load-time policy (`LoadOptions::with_mmq`, threaded down through
    /// `Kernels::load_all`): whether `gemm` may route an MMQ-eligible call
    /// through the int8 matrix-unit path at all. See this struct's `gemm`
    /// doc for the full dispatch policy and why this defaults to off.
    mmq_enabled: bool,
}

impl QuantKernels {
    pub(crate) fn load_all(mmq_enabled: bool) -> Result<Self, RocmlError> {
        let (_mod_q8_0, q8_0_fn) = load(
            rocml_kernels::GEMV_Q8_0_HSACO,
            rocml_kernels::GEMV_Q8_0_KERNEL,
        )?;
        let (_mod_q4_k, q4_k_fn) = load(
            rocml_kernels::GEMV_Q4_K_HSACO,
            rocml_kernels::GEMV_Q4_K_KERNEL,
        )?;
        let (_mod_q5_k, q5_k_fn) = load(
            rocml_kernels::GEMV_Q5_K_HSACO,
            rocml_kernels::GEMV_Q5_K_KERNEL,
        )?;
        let (_mod_q6_k, q6_k_fn) = load(
            rocml_kernels::GEMV_Q6_K_HSACO,
            rocml_kernels::GEMV_Q6_K_KERNEL,
        )?;
        let (_mod_gemm_q8_0, gemm_q8_0_fn) = load(
            rocml_kernels::GEMM_XWT_Q8_0_HSACO,
            rocml_kernels::GEMM_XWT_Q8_0_KERNEL,
        )?;
        let (_mod_gemm_q4_k, gemm_q4_k_fn) = load(
            rocml_kernels::GEMM_XWT_Q4_K_HSACO,
            rocml_kernels::GEMM_XWT_Q4_K_KERNEL,
        )?;
        let (_mod_gemm_q5_k, gemm_q5_k_fn) = load(
            rocml_kernels::GEMM_XWT_Q5_K_HSACO,
            rocml_kernels::GEMM_XWT_Q5_K_KERNEL,
        )?;
        let (_mod_gemm_q6_k, gemm_q6_k_fn) = load(
            rocml_kernels::GEMM_XWT_Q6_K_HSACO,
            rocml_kernels::GEMM_XWT_Q6_K_KERNEL,
        )?;
        let (_mod_gemm_wmma_q8_0, gemm_wmma_q8_0_fn) = load(
            rocml_kernels::GEMM_XWT_WMMA_Q8_0_HSACO,
            rocml_kernels::GEMM_XWT_WMMA_Q8_0_KERNEL,
        )?;
        let (_mod_gemm_wmma_q4_k, gemm_wmma_q4_k_fn) = load(
            rocml_kernels::GEMM_XWT_WMMA_Q4_K_HSACO,
            rocml_kernels::GEMM_XWT_WMMA_Q4_K_KERNEL,
        )?;
        let (_mod_gemm_wmma_q5_k, gemm_wmma_q5_k_fn) = load(
            rocml_kernels::GEMM_XWT_WMMA_Q5_K_HSACO,
            rocml_kernels::GEMM_XWT_WMMA_Q5_K_KERNEL,
        )?;
        let (_mod_gemm_wmma_q6_k, gemm_wmma_q6_k_fn) = load(
            rocml_kernels::GEMM_XWT_WMMA_Q6_K_HSACO,
            rocml_kernels::GEMM_XWT_WMMA_Q6_K_KERNEL,
        )?;
        let mmq = MmqKernels::load_all()?;

        Ok(Self {
            _mod_q8_0,
            q8_0_fn,
            _mod_q4_k,
            q4_k_fn,
            _mod_q5_k,
            q5_k_fn,
            _mod_q6_k,
            q6_k_fn,
            _mod_gemm_q8_0,
            gemm_q8_0_fn,
            _mod_gemm_q4_k,
            gemm_q4_k_fn,
            _mod_gemm_q5_k,
            gemm_q5_k_fn,
            _mod_gemm_q6_k,
            gemm_q6_k_fn,
            _mod_gemm_wmma_q8_0,
            gemm_wmma_q8_0_fn,
            _mod_gemm_wmma_q4_k,
            gemm_wmma_q4_k_fn,
            _mod_gemm_wmma_q5_k,
            gemm_wmma_q5_k_fn,
            _mod_gemm_wmma_q6_k,
            gemm_wmma_q6_k_fn,
            mmq,
            mmq_enabled,
        })
    }

    /// `gemv_<dtype>(w, x, y, m, n)`: y = W * x, W row-major m rows of raw
    /// GGUF blocks. `dtype` must be Q8_0/Q4_K/Q5_K/Q6_K —
    /// `LinearWeight::load` never constructs a `Quant` variant for any other
    /// dtype, so reaching the fallback arm here would be a loader bug;
    /// reported as an error rather than panicking, per the I/O-reachable
    /// panic ban.
    pub(crate) fn gemv(
        &self,
        dtype: GgmlDType,
        w: DevPtr,
        x: DevPtr,
        y: DevPtr,
        m: u32,
        n: u32,
    ) -> Result<(), RocmlError> {
        let function = match dtype {
            GgmlDType::Q8_0 => &self.q8_0_fn,
            GgmlDType::Q4_K => &self.q4_k_fn,
            GgmlDType::Q5_K => &self.q5_k_fn,
            GgmlDType::Q6_K => &self.q6_k_fn,
            other => {
                return Err(RocmlError::Config(format!(
                    "gemv_quant: {other:?} has no fused kernel (internal loader bug)"
                )))
            }
        };
        let cfg = LaunchConfig {
            grid: (m.div_ceil(GEMV_ROWS_PER_WG), 1, 1),
            block: (REDUCE_BLOCK, 1, 1),
            shared_mem_bytes: REDUCE_BLOCK * GEMV_ROWS_PER_WG * size_of::<f32>() as u32,
        };
        let mut params = kernel_params!(w, x, y, m, n);
        // SAFETY: params matches every gemv_<quant>'s signature (const
        // void*, const float*, float*, unsigned, unsigned); block size is
        // the required power of two, and grid/shared-mem match every
        // gemv_q*.hip kernel's `ROWS_PER_WG`-rows-per-workgroup contract.
        unsafe { function.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }

    /// `gemm_xwt_<dtype>(x, w, out, rows, m, n)`: `out[rows,m] = X[rows,n] *
    /// dequant(W)^T`, the batched prefill-path sibling of [`Self::gemv`].
    /// Same dtype restriction as `gemv` (Q8_0/Q4_K/Q5_K/Q6_K only).
    ///
    /// Dispatches to the WMMA matrix-core kernel
    /// (`gemm_xwt_quant_wmma.hip`, issue #6's follow-up) whenever the shape
    /// can fill at least one full tile cleanly — `rows >= TILE_ROWS`, `m >=
    /// TILE_M`, and `m`/`n` both multiples of 16 (WMMA's native fragment
    /// width) — and falls back to the scalar-FMA kernel above otherwise (a
    /// short last chunked-prefill chunk, a shape WMMA can't tile at all, or
    /// an `m` narrower than one `TILE_M`-wide output tile). `n` is already a
    /// multiple of 16 for every dtype this dispatches to (Q8_0/Q4_K/Q5_K/
    /// Q6_K block widths are 32/256/256/256), so the check below only ever
    /// turns on the fallback for `rows`/`m`.
    ///
    /// **The `m >= TILE_M` clause** (prefill-cleanup round, issue #6):
    /// without it, a real narrow-output projection — Ornith's GDN
    /// per-head alpha/beta gates, `m = num_v_heads = 32 < TILE_M(64)` —
    /// still passed the old `m.is_multiple_of(16)` check and dispatched to
    /// WMMA, whose grid is `(m.div_ceil(TILE_M), rows.div_ceil(TILE_ROWS))`:
    /// at `m=32` that's a **single** `grid.x` column, so the whole launch
    /// runs as one workgroup regardless of `rows` — one CU busy, the other
    /// 59 idle, serializing every `K_STAGE`-sized reduction step with none
    /// of WMMA's usual latency-hiding from concurrent tiles. The scalar
    /// kernel's grid is `(m, rows.div_ceil(GEMM_QUANT_TILE_ROWS))` instead —
    /// `m` itself is a grid dimension, so `m=32` still yields 32 (or 64 at
    /// `rows=128`) independent blocks — full occupancy despite lower
    /// per-FLOP throughput. Measured on ornith-9b @ depth 2048: these two
    /// projections alone cost ~294ms of a ~4.6s prefill (found by
    /// temporarily splitting the bundled `gdn-conv` profiler scope into its
    /// constituent kernels — the *conv1d* kernel itself, this round's other
    /// fix, was never the bottleneck the aggregate label suggested; see
    /// `.claude/CLAUDE.md`'s "Chunked prefill" section for the honest
    /// before/after numbers). This clause routes `m < TILE_M` shapes to
    /// scalar unconditionally — it does not special-case just this one
    /// tensor, so any future narrow projection gets the same fix for free.
    ///
    /// Known numeric consequence (measured, not a bug — see the kernel
    /// source's module doc and issue #6's synthetic correctness tests for
    /// why the WMMA kernel itself is verified correct against an
    /// f16-rounded reference): every quantized linear layer a chunked-
    /// prefill call routes through this path now rounds both operands to
    /// f16 before the matrix-core multiply. That's the expected ~5e-4
    /// relative per-layer input-rounding error the issue anticipated, but
    /// it compounds across a ~30-layer model measurably more than the
    /// *pre-WMMA* pure-f32 pipeline this dispatch replaced for `rows >=
    /// 128` shapes — `rocml/tests/qwen35_chunked_prefill_parity.rs`'s
    /// final-logits check (`LOGITS_REL_TOL = 1e-3`, no near-tie escape) and
    /// `rocml/tests/snapshot_equivalence.rs`'s (`1e-6`, calibrated for pure
    /// reduction-order differences, not a precision change) both fail at
    /// prompt lengths that engage a full 128-row WMMA chunk, with measured
    /// max relative logit error in the ~0.16%-0.55% range (up to ~21% of
    /// vocab logits past 1e-3 at some lengths) — while every *greedy-token*
    /// gate (which has a near-tie escape hatch) still passes, meaning
    /// argmax decisions are preserved even though raw logit magnitudes
    /// shift. Left enabled and documented here per the issue's honest-
    /// reporting instruction, not resolved unilaterally by loosening either
    /// test's tolerance.
    ///
    /// **int8 MMQ (int8-MMQ-integration round)**: when `mmq_enabled` was
    /// set at load time (`LoadOptions::with_mmq`, off by default — see this
    /// module's own top-level report/`.claude/CLAUDE.md` for the validation
    /// status that decides whether this ever flips to on by default) *and*
    /// the shape is WMMA-eligible (every weight kind this dispatch handles
    /// now has an MMQ kernel, so no further dtype gating is needed beyond
    /// `MmqKernels::supports`), this routes to the int8 integer-matrix-unit
    /// path instead of f16 WMMA — quantizing `x` on the fly
    /// (`kernels_mmq.rs`'s `MmqKernels::gemm`) rather than rounding it to
    /// f16. This is a strictly larger precision change than the WMMA
    /// rounding above (int8 activations, not f16), so it is gated
    /// separately and never turned on just because WMMA already is.
    /// `mmq_eligible=false` (`weights/linear.rs`'s `mmq_eligible_by_name`) forces WMMA/scalar.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gemm(
        &self,
        dtype: GgmlDType,
        x: DevPtr,
        w: DevPtr,
        out: DevPtr,
        rows: u32,
        m: u32,
        n: u32,
        mmq_scratch: MmqScratch,
        mmq_eligible: bool,
    ) -> Result<(), RocmlError> {
        let wmma_eligible = rows >= GEMM_WMMA_TILE_ROWS
            && m >= GEMM_WMMA_TILE_M
            && m.is_multiple_of(16)
            && n.is_multiple_of(16);
        if wmma_eligible && self.mmq_enabled && mmq_eligible && MmqKernels::supports(dtype) {
            self.mmq.gemm(dtype, x, w, out, rows, m, n, mmq_scratch)
        } else if wmma_eligible {
            self.gemm_wmma(dtype, x, w, out, rows, m, n)
        } else {
            self.gemm_scalar(dtype, x, w, out, rows, m, n)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm_scalar(
        &self,
        dtype: GgmlDType,
        x: DevPtr,
        w: DevPtr,
        out: DevPtr,
        rows: u32,
        m: u32,
        n: u32,
    ) -> Result<(), RocmlError> {
        let function = match dtype {
            GgmlDType::Q8_0 => &self.gemm_q8_0_fn,
            GgmlDType::Q4_K => &self.gemm_q4_k_fn,
            GgmlDType::Q5_K => &self.gemm_q5_k_fn,
            GgmlDType::Q6_K => &self.gemm_q6_k_fn,
            other => {
                return Err(RocmlError::Config(format!(
                    "gemm_quant: {other:?} has no fused kernel (internal loader bug)"
                )))
            }
        };
        let cfg = LaunchConfig {
            grid: (m, rows.div_ceil(GEMM_QUANT_TILE_ROWS), 1),
            block: (32, GEMM_QUANT_WARPS_PER_BLOCK, 1),
            shared_mem_bytes: GEMM_QUANT_TILE_ELEMS * size_of::<f32>() as u32,
        };
        let mut params = kernel_params!(x, w, out, rows, m, n);
        // SAFETY: params matches every gemm_xwt_<quant>'s signature (const
        // float*, const void*, float*, unsigned x3); block = (32, 8, 1)
        // matches the kernel's fixed warp-per-`ROWS_PER_WARP`-rows tiling.
        unsafe { function.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm_wmma(
        &self,
        dtype: GgmlDType,
        x: DevPtr,
        w: DevPtr,
        out: DevPtr,
        rows: u32,
        m: u32,
        n: u32,
    ) -> Result<(), RocmlError> {
        let function = match dtype {
            GgmlDType::Q8_0 => &self.gemm_wmma_q8_0_fn,
            GgmlDType::Q4_K => &self.gemm_wmma_q4_k_fn,
            GgmlDType::Q5_K => &self.gemm_wmma_q5_k_fn,
            GgmlDType::Q6_K => &self.gemm_wmma_q6_k_fn,
            other => {
                return Err(RocmlError::Config(format!(
                    "gemm_quant: {other:?} has no fused WMMA kernel (internal loader bug)"
                )))
            }
        };
        // x2: double-buffered LDS K-stage tiles (padded rows, see LDS_PAD).
        let shared_mem_bytes = 2
            * (GEMM_WMMA_TILE_ROWS + GEMM_WMMA_TILE_M)
            * (GEMM_WMMA_K_STAGE + GEMM_WMMA_LDS_PAD)
            * size_of::<u16>() as u32;
        let cfg = LaunchConfig {
            grid: (
                m.div_ceil(GEMM_WMMA_TILE_M),
                rows.div_ceil(GEMM_WMMA_TILE_ROWS),
                1,
            ),
            block: (32, GEMM_WMMA_WARPS_PER_BLOCK, 1),
            shared_mem_bytes,
        };
        let mut params = kernel_params!(x, w, out, rows, m, n);
        // SAFETY: params matches every gemm_xwt_wmma_<quant>'s signature
        // (const float*, const void*, float*, unsigned x3); block = (32, 8,
        // 1) and shared_mem_bytes match the kernel's fixed TILE_ROWS/TILE_M/
        // K_STAGE tiling (see the kernel source's module doc).
        unsafe { function.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }
}
