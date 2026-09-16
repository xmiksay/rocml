//! Typed launch helpers for `kernels/moe_chunk.hip` (M3's grouped-by-expert
//! chunked-prefill batching). Kept separate from `kernels_moe.rs` purely for
//! the workspace's 400-line file cap, mirroring `kernels_flash_mixed.rs` vs
//! `kernels_mixed.rs`.

use rocml_hip::{kernel_params, LaunchConfig, Module};

use crate::error::RocmlError;
use crate::forward::kernels::{load, DevPtr};

/// Must match `MOE_CHUNK_TILE` in `kernels/moe_chunk.hip`.
const GEMM_F32_TILE: u32 = 16;
/// Grid-stride block size for the plain elementwise gather/scatter/gate
/// kernels below — mirrors `crate::forward::kernels`'s own (private)
/// constant of the same name.
const LINEAR_BLOCK: u32 = 256;

pub struct MoeChunkKernels {
    _mod_gemm: Module,
    gemm_fn: rocml_hip::Function,
    _mod_gather: Module,
    gather_fn: rocml_hip::Function,
    _mod_scatter: Module,
    scatter_fn: rocml_hip::Function,
    _mod_gate: Module,
    gate_fn: rocml_hip::Function,
}

impl MoeChunkKernels {
    pub fn load_all() -> Result<Self, RocmlError> {
        let (_mod_gemm, gemm_fn) = load(
            rocml_kernels::MOE_GEMM_XWT_F32_HSACO,
            rocml_kernels::MOE_GEMM_XWT_F32_KERNEL,
        )?;
        let (_mod_gather, gather_fn) = load(
            rocml_kernels::MOE_GATHER_ROWS_F32_HSACO,
            rocml_kernels::MOE_GATHER_ROWS_F32_KERNEL,
        )?;
        let (_mod_scatter, scatter_fn) = load(
            rocml_kernels::MOE_SCATTER_WEIGHTED_ACCUM_F32_HSACO,
            rocml_kernels::MOE_SCATTER_WEIGHTED_ACCUM_F32_KERNEL,
        )?;
        let (_mod_gate, gate_fn) = load(
            rocml_kernels::MOE_SHARED_GATE_WRITE_CHUNK_F32_HSACO,
            rocml_kernels::MOE_SHARED_GATE_WRITE_CHUNK_F32_KERNEL,
        )?;
        Ok(Self {
            _mod_gemm,
            gemm_fn,
            _mod_gather,
            gather_fn,
            _mod_scatter,
            scatter_fn,
            _mod_gate,
            gate_fn,
        })
    }

    /// `gemm_xwt_f32(x, w, out, rows, m, n)`: `out[rows,m] = X[rows,n] *
    /// W[m,n]^T`, both operands plain f32 — the router/shared-gate weights'
    /// own dtype (see `kernels/moe_chunk.hip`'s module doc for why neither
    /// `gemm_xwt_f16`/`gemm_quant` applies to them).
    pub fn gemm_xwt_f32(
        &self,
        x: DevPtr,
        w: DevPtr,
        out: DevPtr,
        rows: u32,
        m: u32,
        n: u32,
    ) -> Result<(), RocmlError> {
        let cfg = LaunchConfig {
            grid: (m.div_ceil(GEMM_F32_TILE), rows.div_ceil(GEMM_F32_TILE), 1),
            block: (GEMM_F32_TILE, GEMM_F32_TILE, 1),
            shared_mem_bytes: 0,
        };
        let mut params = kernel_params!(x, w, out, rows, m, n);
        // SAFETY: params matches gemm_xwt_f32's signature (const float* x3,
        // unsigned x3); block = (16, 16, 1) matches the kernel's fixed tile.
        unsafe { self.gemm_fn.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }

    /// `moe_gather_rows_f32(src, row_idx, dst, rows, cols)`: `dst[i,:] =
    /// src[row_idx[i],:]`.
    pub fn gather_rows(
        &self,
        src: DevPtr,
        row_idx: DevPtr,
        dst: DevPtr,
        rows: u32,
        cols: u32,
    ) -> Result<(), RocmlError> {
        let total = rows * cols;
        let cfg = LaunchConfig {
            grid: (total.div_ceil(LINEAR_BLOCK), 1, 1),
            block: (LINEAR_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut params = kernel_params!(src, row_idx, dst, rows, cols);
        // SAFETY: params matches moe_gather_rows_f32's signature (const
        // float*, const unsigned*, float*, unsigned, unsigned); no
        // block-size constraint.
        unsafe { self.gather_fn.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }

    /// `moe_scatter_weighted_accum_f32(src, row_idx, weight, dst, rows,
    /// cols)`: `dst[row_idx[i],:] += weight[i] * src[i,:]`.
    #[allow(clippy::too_many_arguments)]
    pub fn scatter_weighted_accum(
        &self,
        src: DevPtr,
        row_idx: DevPtr,
        weight: DevPtr,
        dst: DevPtr,
        rows: u32,
        cols: u32,
    ) -> Result<(), RocmlError> {
        let total = rows * cols;
        let cfg = LaunchConfig {
            grid: (total.div_ceil(LINEAR_BLOCK), 1, 1),
            block: (LINEAR_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut params = kernel_params!(src, row_idx, weight, dst, rows, cols);
        // SAFETY: params matches moe_scatter_weighted_accum_f32's signature
        // (const float*, const unsigned*, const float*, float*, unsigned,
        // unsigned); no block-size constraint.
        unsafe { self.scatter_fn.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }

    /// `moe_shared_gate_write_chunk_f32(y, gate_logit, out, rows, cols)`:
    /// `out[r,:] = y[r,:] * sigmoid(gate_logit[r])`.
    pub fn shared_gate_write_chunk(
        &self,
        y: DevPtr,
        gate_logit: DevPtr,
        out: DevPtr,
        rows: u32,
        cols: u32,
    ) -> Result<(), RocmlError> {
        let total = rows * cols;
        let cfg = LaunchConfig {
            grid: (total.div_ceil(LINEAR_BLOCK), 1, 1),
            block: (LINEAR_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut params = kernel_params!(y, gate_logit, out, rows, cols);
        // SAFETY: params matches moe_shared_gate_write_chunk_f32's signature
        // (const float*, const float*, float*, unsigned, unsigned); no
        // block-size constraint.
        unsafe { self.gate_fn.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }
}
