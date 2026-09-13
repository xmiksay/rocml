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
//! field of `Kernels` and reached through `Kernels::gemv_quant`. The batched
//! prefill-path `gemm_xwt_*` dispatch logic (scalar/WMMA-wide/WMMA-narrow
//! selection) lives in the sibling `kernels_quant_dispatch` module — this
//! file's own struct fields are `pub(super)` so that module (a descendant of
//! `forward`, same as this one) can reach them without a getter per field.

use std::mem::size_of;

use rocml_core::quant::GgmlDType;
use rocml_hip::{kernel_params, LaunchConfig, Module};

use super::kernels::{load, DevPtr, REDUCE_BLOCK};
use super::kernels_mmq::MmqKernels;
use super::kernels_splitk::SplitKKernels;
use crate::error::RocmlError;

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
    pub(super) _mod_q8_0: Module,
    pub(super) q8_0_fn: rocml_hip::Function,
    pub(super) _mod_q4_k: Module,
    pub(super) q4_k_fn: rocml_hip::Function,
    pub(super) _mod_q5_k: Module,
    pub(super) q5_k_fn: rocml_hip::Function,
    pub(super) _mod_q6_k: Module,
    pub(super) q6_k_fn: rocml_hip::Function,
    pub(super) _mod_gemm_q8_0: Module,
    pub(super) gemm_q8_0_fn: rocml_hip::Function,
    pub(super) _mod_gemm_q4_k: Module,
    pub(super) gemm_q4_k_fn: rocml_hip::Function,
    pub(super) _mod_gemm_q5_k: Module,
    pub(super) gemm_q5_k_fn: rocml_hip::Function,
    pub(super) _mod_gemm_q6_k: Module,
    pub(super) gemm_q6_k_fn: rocml_hip::Function,
    pub(super) _mod_gemm_wmma_q8_0: Module,
    pub(super) gemm_wmma_q8_0_fn: rocml_hip::Function,
    pub(super) _mod_gemm_wmma_q4_k: Module,
    pub(super) gemm_wmma_q4_k_fn: rocml_hip::Function,
    pub(super) _mod_gemm_wmma_q5_k: Module,
    pub(super) gemm_wmma_q5_k_fn: rocml_hip::Function,
    pub(super) _mod_gemm_wmma_q6_k: Module,
    pub(super) gemm_wmma_q6_k_fn: rocml_hip::Function,
    // Narrow (TILE_M=64) WMMA variants — shape-aware dispatch round, see
    // `kernels_quant_dispatch.rs`'s module doc for the m<2048 crossover.
    pub(super) _mod_gemm_wmma_q8_0_narrow: Module,
    pub(super) gemm_wmma_q8_0_narrow_fn: rocml_hip::Function,
    pub(super) _mod_gemm_wmma_q4_k_narrow: Module,
    pub(super) gemm_wmma_q4_k_narrow_fn: rocml_hip::Function,
    pub(super) _mod_gemm_wmma_q5_k_narrow: Module,
    pub(super) gemm_wmma_q5_k_narrow_fn: rocml_hip::Function,
    pub(super) _mod_gemm_wmma_q6_k_narrow: Module,
    pub(super) gemm_wmma_q6_k_narrow_fn: rocml_hip::Function,
    pub(super) mmq: MmqKernels,
    /// Load-time policy (`LoadOptions::with_mmq`, threaded down through
    /// `Kernels::load_all`): whether `gemm` may route an MMQ-eligible call
    /// through the int8 matrix-unit path at all. See
    /// `kernels_quant_dispatch.rs`'s `gemm` doc for the full dispatch policy
    /// and why this defaults to off.
    pub(super) mmq_enabled: bool,
    /// Split-K WMMA GEMM (issue #6's split-K follow-up) — see
    /// `kernels_quant_dispatch.rs`'s `gemm` doc for the eligibility/
    /// `num_splits` policy.
    pub(super) splitk: SplitKKernels,
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
        let (_mod_gemm_wmma_q8_0_narrow, gemm_wmma_q8_0_narrow_fn) = load(
            rocml_kernels::GEMM_XWT_WMMA_Q8_0_NARROW_HSACO,
            rocml_kernels::GEMM_XWT_WMMA_Q8_0_NARROW_KERNEL,
        )?;
        let (_mod_gemm_wmma_q4_k_narrow, gemm_wmma_q4_k_narrow_fn) = load(
            rocml_kernels::GEMM_XWT_WMMA_Q4_K_NARROW_HSACO,
            rocml_kernels::GEMM_XWT_WMMA_Q4_K_NARROW_KERNEL,
        )?;
        let (_mod_gemm_wmma_q5_k_narrow, gemm_wmma_q5_k_narrow_fn) = load(
            rocml_kernels::GEMM_XWT_WMMA_Q5_K_NARROW_HSACO,
            rocml_kernels::GEMM_XWT_WMMA_Q5_K_NARROW_KERNEL,
        )?;
        let (_mod_gemm_wmma_q6_k_narrow, gemm_wmma_q6_k_narrow_fn) = load(
            rocml_kernels::GEMM_XWT_WMMA_Q6_K_NARROW_HSACO,
            rocml_kernels::GEMM_XWT_WMMA_Q6_K_NARROW_KERNEL,
        )?;
        let mmq = MmqKernels::load_all()?;
        let splitk = SplitKKernels::load_all()?;

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
            _mod_gemm_wmma_q8_0_narrow,
            gemm_wmma_q8_0_narrow_fn,
            _mod_gemm_wmma_q4_k_narrow,
            gemm_wmma_q4_k_narrow_fn,
            _mod_gemm_wmma_q5_k_narrow,
            gemm_wmma_q5_k_narrow_fn,
            _mod_gemm_wmma_q6_k_narrow,
            gemm_wmma_q6_k_narrow_fn,
            mmq,
            mmq_enabled,
            splitk,
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
}
