//! Batched prefill-path `gemm_xwt_<dtype>` dispatch for `QuantKernels`
//! (struct + `gemv`/loader in the sibling `kernels_quant` module — split
//! out purely for the 400-line file cap once the shape-aware dispatch round
//! below added a third kernel family to choose between).

use std::mem::size_of;

use rocml_core::quant::GgmlDType;
use rocml_hip::{kernel_params, Function, LaunchConfig};

use super::kernels::DevPtr;
use super::kernels_mmq::{MmqKernels, MmqScratch};
use super::kernels_quant::QuantKernels;
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
/// chunk falls back to the scalar kernel). Shared by both the default and
/// narrow WMMA tile configs — only `TILE_M`/`WARPS_M`/`WARPS_N` vary
/// between them, see `gemm_xwt_wmma_impl.h`'s module doc.
const GEMM_WMMA_TILE_ROWS: u32 = 128;
/// Output columns (`W` rows) the default `gemm_xwt_wmma_q*` workgroup tile
/// covers — mirrors the kernel's `TILE_M`. 128 since the per-wave-
/// efficiency round — see `gemm_xwt_quant_wmma.hip`'s module doc.
const GEMM_WMMA_TILE_M: u32 = 128;
/// Reduction elements staged into LDS per outer iteration — mirrors the
/// kernel's `K_STAGE`; combined with `GEMM_WMMA_LDS_PAD` to size the LDS
/// request (the kernel's per-row stride is padded past `K_STAGE`).
const GEMM_WMMA_K_STAGE: u32 = 16;
/// LDS row-stride padding — must match the kernel's `LDS_PAD` (bank-conflict elimination).
const GEMM_WMMA_LDS_PAD: u32 = 8;
/// Warps per WMMA workgroup — fixes the launch's block.y. Identical for the
/// default (`WARPS_M=4,WARPS_N=4`) and narrow (`WARPS_M=8,WARPS_N=2`) tile
/// configs (both multiply out to 16), so one constant covers both.
const GEMM_WMMA_WARPS_PER_BLOCK: u32 = 16;

/// Output columns the narrow `gemm_xwt_wmma_q*_narrow` workgroup tile
/// covers (the original 1x2-fragment/`TILE_M`=64 config, from before the
/// per-wave-efficiency round's `TILE_M`=128 rewrite) — see
/// `gemm_xwt_quant_wmma_narrow.hip`'s module doc for the full measured
/// crossover table this file's `GEMM_WMMA_NARROW_THRESHOLD_M` is based on.
const GEMM_WMMA_NARROW_TILE_M: u32 = 64;
/// Shape-aware dispatch round (issue #6, fixing the per-wave-efficiency
/// round's accepted `m=1024` regression): `m < 2048` routes to the narrow
/// (`TILE_M`=64) WMMA config instead of the default (`TILE_M`=128) one.
/// Measured via a standalone interleaved hipcc harness (rows=512, q4_k,
/// both configs including the vec4-X-load lever): narrow wins by 6.6%-30.9%
/// for every `m` in {512,768,1024,1280,1536,1792}, default wins by 1.5%-8.4%
/// for every `m` in {2048,3072,4096} — see `gemm_xwt_quant_wmma_narrow.hip`
/// for the full table. Ornith-9b's real WMMA-eligible `m`s are {1024 (attn_k/
/// attn_v), 2048, 4096, 8192, 12288} — this threshold routes exactly the
/// regressed 1024 shape to the narrow config and leaves every other real
/// shape on the default one.
const GEMM_WMMA_NARROW_THRESHOLD_M: u32 = 2048;

impl QuantKernels {
    /// `gemm_xwt_<dtype>(x, w, out, rows, m, n)`: `out[rows,m] = X[rows,n] *
    /// dequant(W)^T`, the batched prefill-path sibling of
    /// [`QuantKernels::gemv`]. Same dtype restriction as `gemv`
    /// (Q8_0/Q4_K/Q5_K/Q6_K only).
    ///
    /// Dispatches to a WMMA matrix-core kernel (`gemm_xwt_quant_wmma.hip`/
    /// `_narrow.hip`, issue #6's follow-ups) whenever the shape can fill at
    /// least one full tile cleanly — `rows >= GEMM_WMMA_TILE_ROWS`, `m >=
    /// GEMM_WMMA_NARROW_TILE_M` (64, the smaller of the two WMMA tile
    /// configs' minimums), and `m`/`n` both multiples of 16 (WMMA's native
    /// fragment width) — and falls back to the scalar-FMA kernel otherwise
    /// (a short last chunked-prefill chunk, a shape WMMA can't tile at all,
    /// or an `m` narrower than even the narrow config's tile). `n` is
    /// already a multiple of 16 for every dtype this dispatches to (Q8_0/
    /// Q4_K/Q5_K/Q6_K block widths are 32/256/256/256), so the check below
    /// only ever turns on the fallback for `rows`/`m`.
    ///
    /// **Shape-aware WMMA-width dispatch** (this round): among
    /// WMMA-eligible shapes, `m < GEMM_WMMA_NARROW_THRESHOLD_M` (2048)
    /// further routes to the narrow-tile kernel instead of the default one
    /// — see that constant's doc for the measured crossover. Below
    /// `GEMM_WMMA_NARROW_TILE_M` (64) even the narrow config can't fill a
    /// tile column, so the scalar kernel (whose grid makes `m` itself a
    /// grid dimension, giving `m=32` still 32+ independent blocks) stays
    /// the fallback — this is the same reasoning the prefill-cleanup
    /// round's original `m >= TILE_M` clause established for Ornith's GDN
    /// per-head alpha/beta gate projections (`m = num_v_heads = 32`).
    ///
    /// Known numeric consequence (measured, not a bug — see the kernel
    /// source's module doc and issue #6's synthetic correctness tests for
    /// why the WMMA kernel itself is verified correct against an
    /// f16-rounded reference): every quantized linear layer a chunked-
    /// prefill call routes through either WMMA path now rounds both
    /// operands to f16 before the matrix-core multiply. That's the expected
    /// ~5e-4 relative per-layer input-rounding error the issue anticipated,
    /// but it compounds across a ~30-layer model measurably more than the
    /// *pre-WMMA* pure-f32 pipeline this dispatch replaced for `rows >=
    /// 128` shapes — `rocml/tests/qwen35_chunked_prefill_parity.rs`'s and
    /// `rocml/tests/snapshot_equivalence.rs`'s final-logits tolerances were
    /// recalibrated to `1e-2` for it (measured max relative logit error
    /// ~0.16%-0.55%), while every *greedy-token* gate (with a near-tie
    /// escape hatch) still passes.
    ///
    /// **int8 MMQ (int8-MMQ-integration round)**: when `mmq_enabled` was
    /// set at load time (`LoadOptions::with_mmq`, off by default — see
    /// `.claude/CLAUDE.md` for the validation status that decides whether
    /// this ever flips to on by default) *and* the shape is WMMA-eligible
    /// (every weight kind this dispatch handles now has an MMQ kernel, so
    /// no further dtype gating is needed beyond `MmqKernels::supports`),
    /// this routes to the int8 integer-matrix-unit path instead of f16
    /// WMMA — quantizing `x` on the fly (`kernels_mmq.rs`'s
    /// `MmqKernels::gemm`) rather than rounding it to f16. This is a
    /// strictly larger precision change than the WMMA rounding above (int8
    /// activations, not f16), so it is gated separately and never turned on
    /// just because WMMA already is. `mmq_eligible=false`
    /// (`weights/linear.rs`'s `mmq_eligible_by_name`) forces WMMA/scalar.
    /// MMQ always uses the default (non-narrow) tile width — it was never
    /// swept against the narrow config, and MMQ stays off by default
    /// regardless (see its own validation-ladder writeup).
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
            && m >= GEMM_WMMA_NARROW_TILE_M
            && m.is_multiple_of(16)
            && n.is_multiple_of(16);
        if wmma_eligible && self.mmq_enabled && mmq_eligible && MmqKernels::supports(dtype) {
            self.mmq.gemm(dtype, x, w, out, rows, m, n, mmq_scratch)
        } else if wmma_eligible && m < GEMM_WMMA_NARROW_THRESHOLD_M {
            self.gemm_wmma_cfg(
                self.wmma_narrow_fn(dtype)?,
                GEMM_WMMA_NARROW_TILE_M,
                x,
                w,
                out,
                rows,
                m,
                n,
            )
        } else if wmma_eligible {
            self.gemm_wmma_cfg(
                self.wmma_fn(dtype)?,
                GEMM_WMMA_TILE_M,
                x,
                w,
                out,
                rows,
                m,
                n,
            )
        } else {
            self.gemm_scalar(dtype, x, w, out, rows, m, n)
        }
    }

    fn wmma_fn(&self, dtype: GgmlDType) -> Result<&Function, RocmlError> {
        match dtype {
            GgmlDType::Q8_0 => Ok(&self.gemm_wmma_q8_0_fn),
            GgmlDType::Q4_K => Ok(&self.gemm_wmma_q4_k_fn),
            GgmlDType::Q5_K => Ok(&self.gemm_wmma_q5_k_fn),
            GgmlDType::Q6_K => Ok(&self.gemm_wmma_q6_k_fn),
            other => Err(RocmlError::Config(format!(
                "gemm_quant: {other:?} has no fused WMMA kernel (internal loader bug)"
            ))),
        }
    }

    fn wmma_narrow_fn(&self, dtype: GgmlDType) -> Result<&Function, RocmlError> {
        match dtype {
            GgmlDType::Q8_0 => Ok(&self.gemm_wmma_q8_0_narrow_fn),
            GgmlDType::Q4_K => Ok(&self.gemm_wmma_q4_k_narrow_fn),
            GgmlDType::Q5_K => Ok(&self.gemm_wmma_q5_k_narrow_fn),
            GgmlDType::Q6_K => Ok(&self.gemm_wmma_q6_k_narrow_fn),
            other => Err(RocmlError::Config(format!(
                "gemm_quant: {other:?} has no fused narrow WMMA kernel (internal loader bug)"
            ))),
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

    /// Shared WMMA launch: `tile_m` selects which tile config `function`
    /// was compiled for (128 for the default kernels, 64 for the narrow
    /// ones) — both use `GEMM_WMMA_WARPS_PER_BLOCK`=16, so only the grid.x
    /// and shared-memory computation vary.
    #[allow(clippy::too_many_arguments)]
    fn gemm_wmma_cfg(
        &self,
        function: &Function,
        tile_m: u32,
        x: DevPtr,
        w: DevPtr,
        out: DevPtr,
        rows: u32,
        m: u32,
        n: u32,
    ) -> Result<(), RocmlError> {
        // x2: double-buffered LDS K-stage tiles (padded rows, see LDS_PAD).
        let shared_mem_bytes = 2
            * (GEMM_WMMA_TILE_ROWS + tile_m)
            * (GEMM_WMMA_K_STAGE + GEMM_WMMA_LDS_PAD)
            * size_of::<u16>() as u32;
        let cfg = LaunchConfig {
            grid: (m.div_ceil(tile_m), rows.div_ceil(GEMM_WMMA_TILE_ROWS), 1),
            block: (32, GEMM_WMMA_WARPS_PER_BLOCK, 1),
            shared_mem_bytes,
        };
        let mut params = kernel_params!(x, w, out, rows, m, n);
        // SAFETY: params matches every gemm_xwt_wmma_<quant>[_narrow]'s
        // signature (const float*, const void*, float*, unsigned x3);
        // block/grid/shared_mem_bytes match `function`'s compiled-in
        // TILE_ROWS/tile_m/K_STAGE tiling (caller picks the `function`/
        // `tile_m` pair that match each other).
        unsafe { function.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }
}
