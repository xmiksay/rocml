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
use crate::error::RocmlError;

pub(crate) struct QuantKernels {
    _mod_q8_0: Module,
    q8_0_fn: rocml_hip::Function,
    _mod_q4_k: Module,
    q4_k_fn: rocml_hip::Function,
    _mod_q5_k: Module,
    q5_k_fn: rocml_hip::Function,
    _mod_q6_k: Module,
    q6_k_fn: rocml_hip::Function,
}

impl QuantKernels {
    pub(crate) fn load_all() -> Result<Self, RocmlError> {
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

        Ok(Self {
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
            grid: (m, 1, 1),
            block: (REDUCE_BLOCK, 1, 1),
            shared_mem_bytes: REDUCE_BLOCK * size_of::<f32>() as u32,
        };
        let mut params = kernel_params!(w, x, y, m, n);
        // SAFETY: params matches every gemv_<quant>'s signature (const
        // void*, const float*, float*, unsigned, unsigned); block size is
        // the required power of two.
        unsafe { function.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }
}
