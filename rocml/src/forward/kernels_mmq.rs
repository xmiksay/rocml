//! int8 MMQ-style prefill GEMM (issue #6's WMMA-pipeline round, lever 2;
//! wired into the model's chunked-prefill dispatch in the
//! int8-MMQ-integration round): quantizes the f32 activation to Q8_1-style
//! per-32-element-block int8 codes (`quantize_act_q8_blk`,
//! `rocml-kernels/kernels/quantize_act_q8.hip`) and feeds RDNA3's integer
//! matrix unit directly with that plus the weight's native stored codes —
//! see `rocml-kernels/kernels/gemm_xwt_quant_mmq_q8_0.hip`'s module doc for
//! the full design.
//!
//! Split out of `kernels_quant.rs` (already close to the 400-line cap)
//! purely for that file-size reason, same rationale as `kernels_flash.rs`/
//! `kernels_kv.rs`. `QuantKernels::gemm` (`kernels_quant.rs`) owns the
//! WMMA-vs-MMQ-vs-scalar dispatch policy; this file only owns the MMQ
//! kernels' launch mechanics once that dispatch has already decided to use
//! them.

use std::mem::size_of;

use rocml_core::quant::GgmlDType;
use rocml_hip::{kernel_params, LaunchConfig, Module};

use super::kernels::{load, DevPtr};
use crate::error::RocmlError;

/// Output rows (`X` rows) one workgroup tile covers — matches every
/// `gemm_xwt_mmq_q*` kernel's `TILE_ROWS` exactly (same value as the WMMA
/// f16 kernel's `TILE_ROWS`, not a coincidence: both exist to fill one
/// `PREFILL_CHUNK_SIZE`-sized chunked-prefill row-tile per launch).
const TILE_ROWS: u32 = 128;
/// Output columns (`W` rows) one workgroup tile covers — mirrors the
/// kernel's `TILE_M`.
const TILE_M: u32 = 64;
/// Reduction elements staged into LDS per outer iteration — mirrors the
/// kernel's `K_STAGE` (also the activation quantizer's fixed block width).
const K_STAGE: u32 = 32;
/// Warps per `gemm_xwt_mmq_q*` workgroup — fixes the launch's block.y.
const WARPS_PER_BLOCK: u32 = 16;
/// Warps per `quantize_act_q8_blk` workgroup (one warp per (row, 32-block)).
const QUANTIZE_WARPS_PER_BLOCK: u32 = 8;

/// Pre-allocated int8 MMQ activation-quantization scratch, owned by
/// `qwen35::forward::chunk_scratch::ChunkScratch` and passed to every
/// [`LinearWeight::matmul`](crate::weights::LinearWeight::matmul) call
/// regardless of whether that particular call ends up dispatching to MMQ —
/// three raw pointers is cheap enough that keeping every call site uniform
/// (no `Option` threaded through every layer file) wins over skipping the
/// allocation. Sized for `CHUNK_CAP` rows by the largest `n` any
/// chunked-prefill matmul uses (`ChunkScratch::new`'s `mmq_dim`) — a matmul
/// with a smaller `n` just uses a prefix of it.
#[derive(Clone, Copy)]
pub struct MmqScratch {
    pub codes: DevPtr,
    pub scale: DevPtr,
    pub sum: DevPtr,
}

pub(crate) struct MmqKernels {
    _mod_quantize: Module,
    quantize_fn: rocml_hip::Function,
    _mod_q8_0: Module,
    q8_0_fn: rocml_hip::Function,
    _mod_q4_k: Module,
    q4_k_fn: rocml_hip::Function,
    _mod_q5_k: Module,
    q5_k_fn: rocml_hip::Function,
    _mod_q6_k: Module,
    q6_k_fn: rocml_hip::Function,
}

impl MmqKernels {
    pub(crate) fn load_all() -> Result<Self, RocmlError> {
        let (_mod_quantize, quantize_fn) = load(
            rocml_kernels::QUANTIZE_ACT_Q8_BLK_HSACO,
            rocml_kernels::QUANTIZE_ACT_Q8_BLK_KERNEL,
        )?;
        let (_mod_q8_0, q8_0_fn) = load(
            rocml_kernels::GEMM_XWT_MMQ_Q8_0_HSACO,
            rocml_kernels::GEMM_XWT_MMQ_Q8_0_KERNEL,
        )?;
        let (_mod_q4_k, q4_k_fn) = load(
            rocml_kernels::GEMM_XWT_MMQ_Q4_K_HSACO,
            rocml_kernels::GEMM_XWT_MMQ_Q4_K_KERNEL,
        )?;
        let (_mod_q5_k, q5_k_fn) = load(
            rocml_kernels::GEMM_XWT_MMQ_Q5_K_HSACO,
            rocml_kernels::GEMM_XWT_MMQ_Q5_K_KERNEL,
        )?;
        let (_mod_q6_k, q6_k_fn) = load(
            rocml_kernels::GEMM_XWT_MMQ_Q6_K_HSACO,
            rocml_kernels::GEMM_XWT_MMQ_Q6_K_KERNEL,
        )?;
        Ok(Self {
            _mod_quantize,
            quantize_fn,
            _mod_q8_0,
            q8_0_fn,
            _mod_q4_k,
            q4_k_fn,
            _mod_q5_k,
            q5_k_fn,
            _mod_q6_k,
            q6_k_fn,
        })
    }

    /// Every dtype this family supports has a fused MMQ kernel — kept as an
    /// explicit predicate (rather than assuming) so a future weight kind
    /// without one fails fast in `QuantKernels::gemm`'s dispatch instead of
    /// silently mis-routing.
    pub(crate) fn supports(dtype: GgmlDType) -> bool {
        matches!(
            dtype,
            GgmlDType::Q8_0 | GgmlDType::Q4_K | GgmlDType::Q5_K | GgmlDType::Q6_K
        )
    }

    /// `quantize_act_q8_blk(x, codes, scale, sum, rows, n)`: absmax-quantizes
    /// `x[rows, n]` into `[rows, n/32]` Q8_1-style int8 blocks. `n` must be a
    /// multiple of 32 (guaranteed here — every `Quant`-dtype weight's `n`
    /// already satisfies that per `LinearWeight::load`'s policy).
    fn quantize_activation(
        &self,
        x: DevPtr,
        codes: DevPtr,
        scale: DevPtr,
        sum: DevPtr,
        rows: u32,
        n: u32,
    ) -> Result<(), RocmlError> {
        let n_blocks = n / K_STAGE;
        let total_warps = rows * n_blocks;
        let cfg = LaunchConfig {
            grid: (total_warps.div_ceil(QUANTIZE_WARPS_PER_BLOCK), 1, 1),
            block: (32, QUANTIZE_WARPS_PER_BLOCK, 1),
            shared_mem_bytes: 0,
        };
        let mut params = kernel_params!(x, codes, scale, sum, rows, n);
        // SAFETY: params matches quantize_act_q8_blk's signature (const
        // float*, signed char*, float*, float*, unsigned, unsigned).
        unsafe { self.quantize_fn.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }

    /// Quantizes `x` into `mmq`'s scratch, then dispatches the matching
    /// `gemm_xwt_mmq_<dtype>` kernel. Caller (`QuantKernels::gemm`) is
    /// responsible for shape/dtype eligibility — this always runs the MMQ
    /// path once called.
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
        mmq: MmqScratch,
    ) -> Result<(), RocmlError> {
        self.quantize_activation(x, mmq.codes, mmq.scale, mmq.sum, rows, n)?;
        let cfg = LaunchConfig {
            grid: (m.div_ceil(TILE_M), rows.div_ceil(TILE_ROWS), 1),
            block: (32, WARPS_PER_BLOCK, 1),
            shared_mem_bytes: self.shared_mem_bytes(dtype),
        };
        match dtype {
            // Symmetric dtypes: no block-sum input needed (see
            // gemm_xwt_quant_mmq_q8_0.hip's/_q6_k.hip's module docs).
            GgmlDType::Q8_0 | GgmlDType::Q6_K => {
                let function = match dtype {
                    GgmlDType::Q8_0 => &self.q8_0_fn,
                    GgmlDType::Q6_K => &self.q6_k_fn,
                    _ => unreachable!(),
                };
                let mut params = kernel_params!(mmq.codes, mmq.scale, w, out, rows, m, n);
                // SAFETY: params matches gemm_xwt_mmq_q8_0/gemm_xwt_mmq_q6_k's
                // signature (const signed char*, const float*, const void*,
                // float*, unsigned x3); cfg matches this file's fixed
                // TILE_ROWS/TILE_M/K_STAGE tiling.
                unsafe { function.launch(&cfg, &mut params, None) }.map_err(Into::into)
            }
            // Affine dtypes: need the block-sum to cancel the min-offset
            // term (see gemm_xwt_quant_mmq_q4_k.hip's module doc).
            GgmlDType::Q4_K | GgmlDType::Q5_K => {
                let function = match dtype {
                    GgmlDType::Q4_K => &self.q4_k_fn,
                    GgmlDType::Q5_K => &self.q5_k_fn,
                    _ => unreachable!(),
                };
                let mut params = kernel_params!(mmq.codes, mmq.scale, mmq.sum, w, out, rows, m, n);
                // SAFETY: params matches gemm_xwt_mmq_q4_k/gemm_xwt_mmq_q5_k's
                // signature (const signed char*, const float*, const float*,
                // const void*, float*, unsigned x3); cfg as above.
                unsafe { function.launch(&cfg, &mut params, None) }.map_err(Into::into)
            }
            other => Err(RocmlError::Config(format!(
                "gemm_mmq: {other:?} has no fused MMQ kernel (internal dispatch bug)"
            ))),
        }
    }

    fn shared_mem_bytes(&self, dtype: GgmlDType) -> u32 {
        let f32_size = size_of::<f32>() as u32;
        let tiles = TILE_ROWS * K_STAGE + TILE_M * K_STAGE;
        match dtype {
            // One d_a per row, one d_w per column.
            GgmlDType::Q8_0 => tiles + TILE_ROWS * f32_size + TILE_M * f32_size,
            // One d_a + one block_sum per row, one d_w + one off_w per column.
            GgmlDType::Q4_K | GgmlDType::Q5_K => {
                tiles + TILE_ROWS * f32_size * 2 + TILE_M * f32_size * 2
            }
            // One d_a per row, but two d_w scales per column (16-wide scale
            // granularity within the 32-wide K_STAGE — see
            // gemm_xwt_quant_mmq_q6_k.hip's module doc).
            GgmlDType::Q6_K => tiles + TILE_ROWS * f32_size + TILE_M * f32_size * 2,
            // Unreachable via `QuantKernels::gemm`'s `Self::supports` guard;
            // 0 is a safe (if useless) fallback rather than panicking.
            _ => 0,
        }
    }
}
