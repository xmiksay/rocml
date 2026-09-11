//! Loads every HIP kernel the forward pass needs once at model-load time,
//! and wraps each one in a typed launch helper so `attention.rs`/`ffn.rs`
//! don't touch raw `kernel_params!`/`LaunchConfig` directly. Every helper
//! takes plain device pointers (`DevPtr`) rather than `&DeviceBuffer<T>` so
//! the same helper serves both whole-buffer calls and offset sub-buffer
//! calls (e.g. one KV-cache head's plane inside a layer's full cache).

use std::ffi::c_void;
use std::mem::size_of;

use rocml_hip::{kernel_params, DeviceBuffer, LaunchConfig, Module};

use crate::error::RocmlError;

/// A raw device pointer, valid only as a kernel launch argument for as long
/// as the buffer it was derived from is alive. See [`offset`].
pub type DevPtr = *mut c_void;

/// Computes a device pointer `elem_offset` elements into `buf` — e.g. one
/// KV-cache head's plane, or one attention head's slice of a concatenated
/// q/k/v vector. `elem_offset` must be no larger than `buf.len()`,
/// unconditionally checked (not `debug_assert!`) because every caller here
/// derives it from this model's own fixed layer/head geometry, so a
/// violation is a real bug in that arithmetic, not untrusted input to
/// tolerate gracefully.
pub fn offset<T: Copy>(buf: &DeviceBuffer<T>, elem_offset: usize) -> DevPtr {
    assert!(
        elem_offset <= buf.len(),
        "offset {elem_offset} out of bounds for buffer of len {}",
        buf.len()
    );
    // SAFETY: `elem_offset <= buf.len()` was just checked, so the resulting
    // pointer lies within (or one-past-the-end of) `buf`'s live `hipMalloc`
    // allocation. It is only ever passed on as a kernel launch argument,
    // never dereferenced on the host.
    unsafe { (buf.device_ptr() as *mut u8).add(elem_offset * size_of::<T>()) as DevPtr }
}

/// Block size for every power-of-two-reduction kernel (`rmsnorm_f32`,
/// `gemv_f32`, `gemv_f16`, `softmax_varlen_f32`) — one value big enough to
/// stay efficient for the largest `n` this model uses (hidden=1024,
/// q_dim=2048, max_seq up to 4096) while still being a fine grid-stride
/// block for the smallest (head_dim=128).
const REDUCE_BLOCK: u32 = 128;
/// Block size for plain elementwise/grid-stride kernels with no
/// power-of-two constraint.
const LINEAR_BLOCK: u32 = 256;

pub struct Kernels {
    _mod_embedding: Module,
    embedding_fn: rocml_hip::Function,
    _mod_rmsnorm: Module,
    rmsnorm_fn: rocml_hip::Function,
    _mod_gemv_f32: Module,
    gemv_f32_fn: rocml_hip::Function,
    _mod_gemv_f16: Module,
    gemv_f16_fn: rocml_hip::Function,
    _mod_gemv_t_f32: Module,
    gemv_t_f32_fn: rocml_hip::Function,
    _mod_rope: Module,
    rope_fn: rocml_hip::Function,
    _mod_softmax: Module,
    softmax_fn: rocml_hip::Function,
    _mod_silu_mul: Module,
    silu_mul_fn: rocml_hip::Function,
    _mod_elementwise: Module,
    add_inplace_fn: rocml_hip::Function,
}

fn load(hsaco: &[u8], name: &str) -> Result<(Module, rocml_hip::Function), RocmlError> {
    let module = Module::load_from_bytes(hsaco)?;
    let function = module.get_function(name)?;
    Ok((module, function))
}

impl Kernels {
    pub fn load_all() -> Result<Self, RocmlError> {
        let (_mod_embedding, embedding_fn) = load(
            rocml_kernels::EMBEDDING_F16_F32_HSACO,
            rocml_kernels::EMBEDDING_F16_F32_KERNEL,
        )?;
        let (_mod_rmsnorm, rmsnorm_fn) = load(
            rocml_kernels::RMSNORM_F32_HSACO,
            rocml_kernels::RMSNORM_F32_KERNEL,
        )?;
        let (_mod_gemv_f32, gemv_f32_fn) = load(
            rocml_kernels::GEMV_F32_HSACO,
            rocml_kernels::GEMV_F32_KERNEL,
        )?;
        let (_mod_gemv_f16, gemv_f16_fn) = load(
            rocml_kernels::GEMV_F16_HSACO,
            rocml_kernels::GEMV_F16_KERNEL,
        )?;
        let (_mod_gemv_t_f32, gemv_t_f32_fn) = load(
            rocml_kernels::GEMV_T_F32_HSACO,
            rocml_kernels::GEMV_T_F32_KERNEL,
        )?;
        let (_mod_rope, rope_fn) = load(
            rocml_kernels::ROPE_NEOX_F32_HSACO,
            rocml_kernels::ROPE_NEOX_F32_KERNEL,
        )?;
        let (_mod_softmax, softmax_fn) = load(
            rocml_kernels::SOFTMAX_VARLEN_F32_HSACO,
            rocml_kernels::SOFTMAX_VARLEN_F32_KERNEL,
        )?;
        let (_mod_silu_mul, silu_mul_fn) = load(
            rocml_kernels::SILU_MUL_F32_HSACO,
            rocml_kernels::SILU_MUL_F32_KERNEL,
        )?;
        let (_mod_elementwise, add_inplace_fn) = load(
            rocml_kernels::ELEMENTWISE_HSACO,
            rocml_kernels::ADD_INPLACE_F32_KERNEL,
        )?;

        Ok(Self {
            _mod_embedding,
            embedding_fn,
            _mod_rmsnorm,
            rmsnorm_fn,
            _mod_gemv_f32,
            gemv_f32_fn,
            _mod_gemv_f16,
            gemv_f16_fn,
            _mod_gemv_t_f32,
            gemv_t_f32_fn,
            _mod_rope,
            rope_fn,
            _mod_softmax,
            softmax_fn,
            _mod_silu_mul,
            silu_mul_fn,
            _mod_elementwise,
            add_inplace_fn,
        })
    }

    /// `ids`/`table`/`out` per `embedding_f16_f32`: one block per token row.
    pub fn embedding(
        &self,
        ids: DevPtr,
        table: DevPtr,
        out: DevPtr,
        tokens: u32,
        dim: u32,
    ) -> Result<(), RocmlError> {
        let cfg = LaunchConfig {
            grid: (tokens, 1, 1),
            block: (LINEAR_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut params = kernel_params!(ids, table, out, tokens, dim);
        // SAFETY: params matches embedding_f16_f32's signature (const
        // unsigned*, const half*, float*, unsigned, unsigned); caller
        // guarantees the backing buffers outlive this launch.
        unsafe { self.embedding_fn.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }

    /// `rmsnorm_f32(x, weight, out, rows, n, eps)`, in place if `out == x`.
    pub fn rmsnorm(
        &self,
        x: DevPtr,
        weight: DevPtr,
        out: DevPtr,
        rows: u32,
        n: u32,
        eps: f32,
    ) -> Result<(), RocmlError> {
        let cfg = LaunchConfig {
            grid: (rows, 1, 1),
            block: (REDUCE_BLOCK, 1, 1),
            shared_mem_bytes: REDUCE_BLOCK * size_of::<f32>() as u32,
        };
        let mut params = kernel_params!(x, weight, out, rows, n, eps);
        // SAFETY: params matches rmsnorm_f32's signature (const float*,
        // const float*, float*, unsigned, unsigned, float); block size is
        // the required power of two.
        unsafe { self.rmsnorm_fn.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }

    /// `gemv_f32(mat, x, y, m, n)`: y = mat * x, mat row-major m x n f32.
    pub fn gemv_f32(
        &self,
        mat: DevPtr,
        x: DevPtr,
        y: DevPtr,
        m: u32,
        n: u32,
    ) -> Result<(), RocmlError> {
        let cfg = LaunchConfig {
            grid: (m, 1, 1),
            block: (REDUCE_BLOCK, 1, 1),
            shared_mem_bytes: REDUCE_BLOCK * size_of::<f32>() as u32,
        };
        let mut params = kernel_params!(mat, x, y, m, n);
        // SAFETY: params matches gemv_f32's signature (const float*, const
        // float*, float*, unsigned, unsigned); block size is the required
        // power of two.
        unsafe { self.gemv_f32_fn.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }

    /// `gemv_f16(w, x, y, m, n)`: y = W * x, W row-major m x n f16.
    pub fn gemv_f16(
        &self,
        w: DevPtr,
        x: DevPtr,
        y: DevPtr,
        m: u32,
        n: u32,
    ) -> Result<(), RocmlError> {
        let cfg = LaunchConfig {
            grid: (m, 1, 1),
            block: (REDUCE_BLOCK, 1, 1),
            shared_mem_bytes: REDUCE_BLOCK * size_of::<f32>() as u32,
        };
        let mut params = kernel_params!(w, x, y, m, n);
        // SAFETY: params matches gemv_f16's signature (const half*, const
        // float*, float*, unsigned, unsigned); block size is the required
        // power of two.
        unsafe { self.gemv_f16_fn.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }

    /// `gemv_t_f32(a, x, y, rows, n)`: y = A^T * x, A row-major rows x n f32.
    pub fn gemv_t_f32(
        &self,
        a: DevPtr,
        x: DevPtr,
        y: DevPtr,
        rows: u32,
        n: u32,
    ) -> Result<(), RocmlError> {
        let cfg = LaunchConfig {
            grid: (n.div_ceil(LINEAR_BLOCK), 1, 1),
            block: (LINEAR_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut params = kernel_params!(a, x, y, rows, n);
        // SAFETY: params matches gemv_t_f32's signature (const float*, const
        // float*, float*, unsigned, unsigned); no block-size constraint.
        unsafe { self.gemv_t_f32_fn.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }

    /// In-place NEOX rope over `x` viewed as `[tokens, heads, head_dim]`.
    pub fn rope(
        &self,
        x: DevPtr,
        tokens: u32,
        heads: u32,
        head_dim: u32,
        pos_base: u32,
        theta_base: f32,
    ) -> Result<(), RocmlError> {
        let half_dim = head_dim / 2;
        let total = tokens * heads * half_dim;
        let cfg = LaunchConfig {
            grid: (total.div_ceil(LINEAR_BLOCK), 1, 1),
            block: (LINEAR_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut params = kernel_params!(x, tokens, heads, head_dim, pos_base, theta_base);
        // SAFETY: params matches rope_neox_f32's signature (float*,
        // unsigned x3, unsigned, float); head_dim is even (validated by
        // ModelConfig — it's twice the rope half-dim by construction).
        unsafe { self.rope_fn.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }

    /// `softmax_varlen_f32(x, valid_len, rows, cols, scale)`, in place.
    pub fn softmax_varlen(
        &self,
        x: DevPtr,
        valid_len: DevPtr,
        rows: u32,
        cols: u32,
        scale: f32,
    ) -> Result<(), RocmlError> {
        let cfg = LaunchConfig {
            grid: (rows, 1, 1),
            block: (REDUCE_BLOCK, 1, 1),
            shared_mem_bytes: REDUCE_BLOCK * size_of::<f32>() as u32,
        };
        let mut params = kernel_params!(x, valid_len, rows, cols, scale);
        // SAFETY: params matches softmax_varlen_f32's signature (float*,
        // const unsigned*, unsigned, unsigned, float); block size is the
        // required power of two.
        unsafe { self.softmax_fn.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }

    /// `silu_mul_f32(gate, up, out, n)`.
    pub fn silu_mul(
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
        // SAFETY: params matches silu_mul_f32's signature (const float*,
        // const float*, float*, unsigned); no block-size constraint.
        unsafe { self.silu_mul_fn.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }

    /// `add_inplace_f32(acc, x, n)`: acc += x.
    pub fn add_inplace(&self, acc: DevPtr, x: DevPtr, n: u32) -> Result<(), RocmlError> {
        let cfg = LaunchConfig {
            grid: (n.div_ceil(LINEAR_BLOCK), 1, 1),
            block: (LINEAR_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut params = kernel_params!(acc, x, n);
        // SAFETY: params matches add_inplace_f32's signature (float*, const
        // float*, unsigned); no block-size constraint.
        unsafe { self.add_inplace_fn.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }
}
