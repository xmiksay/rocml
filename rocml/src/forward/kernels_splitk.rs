//! Split-K WMMA GEMM (issue #6's split-K follow-up, for narrow-grid shapes
//! like ffn-down's `m=hidden, n=intermediate` — see
//! `rocml-kernels/kernels/gemm_xwt_wmma_splitk_impl.h`'s module doc for the
//! full design/diagnosis). Split out of `kernels_quant.rs`/
//! `kernels_quant_dispatch.rs` purely for the 400-line file cap, mirroring
//! `kernels_mmq.rs`'s pattern: this file owns the split-K kernels' launch
//! mechanics (the GEMM launch plus the reduce-pass launch); `QuantKernels::
//! gemm` (`kernels_quant_dispatch.rs`) owns the WMMA-vs-split-K-vs-scalar
//! dispatch policy and eligibility/`num_splits` decision.

use std::mem::size_of;

use rocml_core::quant::GgmlDType;
use rocml_hip::{kernel_params, Function, LaunchConfig, Module};

use super::kernels::{load, DevPtr};
use crate::error::RocmlError;

/// Output rows (`X` rows) one workgroup tile covers — matches
/// `gemm_xwt_wmma_impl.h`'s `TILE_ROWS`, same as the plain default-tile WMMA
/// kernel (split-K reuses that exact tile config, see
/// `gemm_xwt_wmma_splitk.hip`'s module doc).
const TILE_ROWS: u32 = 128;
/// Output columns (`W` rows) one workgroup tile covers — matches the default
/// (non-narrow) plain WMMA kernel's `TILE_M`.
const TILE_M: u32 = 128;
/// Reduction elements staged into LDS per outer iteration — mirrors the
/// shared `gemm_xwt_wmma_impl.h`'s `K_STAGE`.
const K_STAGE: u32 = 16;
/// LDS row-stride padding — must match the shared header's `LDS_PAD`.
const LDS_PAD: u32 = 8;
/// Warps per split-K GEMM workgroup — fixes the launch's block.y (matches
/// the default plain-WMMA config's `WARPS_M(4) * WARPS_N(4)`).
const WARPS_PER_BLOCK: u32 = 16;
/// Threads per block for the reduce pass (a plain 1D elementwise kernel, no
/// shared-memory/warp-shape constraint — any power of two works).
const REDUCE_BLOCK: u32 = 256;

/// Fixed maximum number of K-splits ever launched — the only values
/// `QuantKernels::gemm`'s dispatch ever picks are `{1 (no split), 2, 4}`, all
/// `<= SPLITK_MAX_SPLITS`. `ChunkScratch::gemm_splitk_partial` is sized
/// against this constant, so raising it needs a matching scratch resize.
pub const SPLITK_MAX_SPLITS: u32 = 4;

/// Pre-allocated split-K partial-sum scratch: `[SPLITK_MAX_SPLITS, CHUNK_CAP,
/// max_m]` f32, owned by `qwen35::forward::chunk_scratch::ChunkScratch` and
/// passed to every [`LinearWeight::matmul`](crate::weights::LinearWeight::matmul)
/// call regardless of whether that call ends up using split-K (same
/// always-allocate-for-uniformity rationale as `MmqScratch`). `max_m` is
/// this model's `hidden` (embedding_length) — the widest output any
/// narrow-grid (split-K-candidate) projection ever has; `kernels_quant_
/// dispatch.rs`'s `splitk_num_splits` refuses to split any call whose `m`
/// exceeds it, so a wider-output call (e.g. the FFN gate/up projections,
/// `m=intermediate`) can never write past this buffer's allocated capacity
/// even though those shapes' grids are never narrow enough to request
/// split-K in practice.
#[derive(Clone, Copy)]
pub struct SplitKScratch {
    pub partial: DevPtr,
    pub max_m: u32,
}

pub(crate) struct SplitKKernels {
    _mod_q8_0: Module,
    q8_0_fn: Function,
    _mod_q4_k: Module,
    q4_k_fn: Function,
    _mod_q5_k: Module,
    q5_k_fn: Function,
    _mod_q6_k: Module,
    q6_k_fn: Function,
    _mod_reduce: Module,
    reduce_fn: Function,
}

impl SplitKKernels {
    pub(crate) fn load_all() -> Result<Self, RocmlError> {
        let (_mod_q8_0, q8_0_fn) = load(
            rocml_kernels::GEMM_XWT_WMMA_SPLITK_Q8_0_HSACO,
            rocml_kernels::GEMM_XWT_WMMA_SPLITK_Q8_0_KERNEL,
        )?;
        let (_mod_q4_k, q4_k_fn) = load(
            rocml_kernels::GEMM_XWT_WMMA_SPLITK_Q4_K_HSACO,
            rocml_kernels::GEMM_XWT_WMMA_SPLITK_Q4_K_KERNEL,
        )?;
        let (_mod_q5_k, q5_k_fn) = load(
            rocml_kernels::GEMM_XWT_WMMA_SPLITK_Q5_K_HSACO,
            rocml_kernels::GEMM_XWT_WMMA_SPLITK_Q5_K_KERNEL,
        )?;
        let (_mod_q6_k, q6_k_fn) = load(
            rocml_kernels::GEMM_XWT_WMMA_SPLITK_Q6_K_HSACO,
            rocml_kernels::GEMM_XWT_WMMA_SPLITK_Q6_K_KERNEL,
        )?;
        let (_mod_reduce, reduce_fn) = load(
            rocml_kernels::GEMM_SPLITK_REDUCE_F32_HSACO,
            rocml_kernels::GEMM_SPLITK_REDUCE_F32_KERNEL,
        )?;
        Ok(Self {
            _mod_q8_0,
            q8_0_fn,
            _mod_q4_k,
            q4_k_fn,
            _mod_q5_k,
            q5_k_fn,
            _mod_q6_k,
            q6_k_fn,
            _mod_reduce,
            reduce_fn,
        })
    }

    fn function(&self, dtype: GgmlDType) -> Result<&Function, RocmlError> {
        match dtype {
            GgmlDType::Q8_0 => Ok(&self.q8_0_fn),
            GgmlDType::Q4_K => Ok(&self.q4_k_fn),
            GgmlDType::Q5_K => Ok(&self.q5_k_fn),
            GgmlDType::Q6_K => Ok(&self.q6_k_fn),
            other => Err(RocmlError::Config(format!(
                "gemm_splitk: {other:?} has no fused split-K WMMA kernel (internal dispatch bug)"
            ))),
        }
    }

    /// Runs the split-K GEMM pass (writing `[num_splits, rows, m]` partial
    /// sums into `scratch.partial`) followed by the deterministic
    /// ascending-order reduce pass into `out`. Caller (`QuantKernels::gemm`)
    /// is responsible for shape eligibility (`m`/`n`/`rows` WMMA-legal, `n`
    /// evenly divisible by `num_splits * K_STAGE`, `rows * m` within
    /// `scratch`'s allocated capacity) — this always runs both passes once
    /// called.
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
        num_splits: u32,
        scratch: SplitKScratch,
    ) -> Result<(), RocmlError> {
        let function = self.function(dtype)?;
        let shared_mem_bytes =
            2 * (TILE_ROWS + TILE_M) * (K_STAGE + LDS_PAD) * size_of::<u16>() as u32;
        let cfg = LaunchConfig {
            grid: (m.div_ceil(TILE_M), rows.div_ceil(TILE_ROWS), num_splits),
            block: (32, WARPS_PER_BLOCK, 1),
            shared_mem_bytes,
        };
        let partial = scratch.partial;
        let mut params = kernel_params!(x, w, partial, rows, m, n, num_splits);
        // SAFETY: params matches every gemm_xwt_wmma_splitk_<quant>'s
        // signature (const float*, const void*, float*, unsigned x3,
        // unsigned); grid/block/shared_mem_bytes match this file's fixed
        // TILE_ROWS/TILE_M/K_STAGE tiling (the default plain-WMMA config).
        unsafe { function.launch(&cfg, &mut params, None) }?;

        let total = rows * m;
        let reduce_cfg = LaunchConfig {
            grid: (total.div_ceil(REDUCE_BLOCK), 1, 1),
            block: (REDUCE_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut reduce_params = kernel_params!(partial, out, rows, m, num_splits);
        // SAFETY: params matches gemm_splitk_reduce_f32's signature (const
        // float*, float*, unsigned x3); grid/block cover every `rows*m`
        // output element exactly once.
        unsafe { self.reduce_fn.launch(&reduce_cfg, &mut reduce_params, None) }.map_err(Into::into)
    }
}
