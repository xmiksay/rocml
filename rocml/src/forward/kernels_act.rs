//! GeGLU-style gated-FFN activation kernel (`gelu_mul_f32`, same code object
//! as `silu_mul_f32` — see `kernels/silu_mul.hip`'s module doc), split into
//! its own tiny file rather than added to `kernels.rs` (already well past
//! the workspace's 400-line cap, and the hard rule against growing a file
//! already over the cap). Kept as a standalone kernel set, owned by
//! `forward::Model` alongside (not nested inside) `Kernels`, and passed as
//! its own `act_kernels: &ActivationKernels` parameter to `ffn::ffn_step`/
//! `ffn_chunk::ffn_chunk_step` — a plain extra argument rather than a field
//! on `Kernels` itself, specifically to avoid growing that file at all.
//!
//! Issue #16's config-driven-activation seam: `ModelConfig::activation`
//! (`Activation::SiLu`/`Activation::Gelu`) is what a future dense-family
//! loader (e.g. gemma, whose FFN uses `gelu_pytorch_tanh`) sets to reach
//! this kernel from `ffn::ffn_step`/`ffn_chunk::ffn_chunk_step` — no
//! architecture built by this loader selects `Gelu` yet, so this is
//! unreachable from any real model today, only from this module's own unit
//! test and `rocml-kernels/tests/gelu_mul.rs`'s GPU integration test.

use super::kernels::{load, DevPtr, LINEAR_BLOCK};
use crate::error::RocmlError;
use rocml_hip::{kernel_params, LaunchConfig, Module};

pub struct ActivationKernels {
    _mod_gelu_mul: Module,
    gelu_mul_fn: rocml_hip::Function,
}

impl ActivationKernels {
    pub fn load_all() -> Result<Self, RocmlError> {
        let (_mod_gelu_mul, gelu_mul_fn) = load(
            rocml_kernels::SILU_MUL_F32_HSACO,
            rocml_kernels::GELU_MUL_F32_KERNEL,
        )?;
        Ok(Self {
            _mod_gelu_mul,
            gelu_mul_fn,
        })
    }

    /// `gelu_mul_f32(gate, up, out, n)`: `out = gelu_tanh(gate) * up`, the
    /// GeGLU sibling of `Kernels::silu_mul`.
    pub fn gelu_mul(
        &self,
        gate: DevPtr,
        up: DevPtr,
        out: DevPtr,
        n: u32,
    ) -> Result<(), RocmlError> {
        let cfg = LaunchConfig {
            grid: (n.div_ceil(LINEAR_BLOCK), 1, 1),
            block: (LINEAR_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut params = kernel_params!(gate, up, out, n);
        // SAFETY: params matches gelu_mul_f32's signature (const float*,
        // const float*, float*, unsigned); no block-size constraint.
        unsafe { self.gelu_mul_fn.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }
}
