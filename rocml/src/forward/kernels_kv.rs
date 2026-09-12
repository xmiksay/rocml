//! f16-KV-cache kernels (issue #3's default dtype): the cast used when
//! appending a decode/prefill step's f32 K/V into f16 cache storage, and the
//! f16 siblings of `attn_decode_partial_f32`/`attn_prefill_f32`. Split out
//! of `kernels.rs` (already at the workspace's 400-line file cap) into its
//! own small kernel-owning struct, embedded as a field of `Kernels` and
//! reached through `Kernels::cast_f32_f16`/`attn_decode_f16`/
//! `attn_prefill_f16` — mirrors how `kernels_quant.rs`'s `QuantKernels`
//! already handles the same file-size pressure for the quantized-weight
//! kernels.

use std::mem::size_of;

use rocml_hip::{kernel_params, LaunchConfig, Module};

use super::kernels::{load, DevPtr, ATTN_DECODE_TILE_T, LINEAR_BLOCK};
use crate::error::RocmlError;

/// Consecutive query rows sharing one `attn_prefill_*` workgroup's K/V-tile
/// load — must match `ROW_TILE` in `kernels/attn_prefill.hip` exactly (a
/// launch-shape constant, not something read from shared memory sizing, but
/// the grid/block dims used by both `attn_prefill`/`attn_prefill_f16` are
/// meaningless if it drifts from the kernel). See that kernel's module doc
/// for the tuning rationale. Defined here rather than in `kernels.rs`
/// (already over this workspace's 400-line file cap) even though
/// `Kernels::attn_prefill`'s f32 launch also needs it.
pub(super) const ATTN_PREFILL_ROW_TILE: u32 = 4;

pub(crate) struct KvF16Kernels {
    _mod_cast: Module,
    cast_f32_f16_fn: rocml_hip::Function,
    _mod_attn_decode_partial_f16: Module,
    attn_decode_partial_f16_fn: rocml_hip::Function,
    _mod_attn_prefill_f16: Module,
    attn_prefill_f16_fn: rocml_hip::Function,
}

impl KvF16Kernels {
    pub(crate) fn load_all() -> Result<Self, RocmlError> {
        let (_mod_cast, cast_f32_f16_fn) = load(
            rocml_kernels::ELEMENTWISE_HSACO,
            rocml_kernels::CAST_F32_F16_KERNEL,
        )?;
        let (_mod_attn_decode_partial_f16, attn_decode_partial_f16_fn) = load(
            rocml_kernels::ATTN_DECODE_PARTIAL_F16_HSACO,
            rocml_kernels::ATTN_DECODE_PARTIAL_F16_KERNEL,
        )?;
        let (_mod_attn_prefill_f16, attn_prefill_f16_fn) = load(
            rocml_kernels::ATTN_PREFILL_F16_HSACO,
            rocml_kernels::ATTN_PREFILL_F16_KERNEL,
        )?;
        Ok(Self {
            _mod_cast,
            cast_f32_f16_fn,
            _mod_attn_decode_partial_f16,
            attn_decode_partial_f16_fn,
            _mod_attn_prefill_f16,
            attn_prefill_f16_fn,
        })
    }

    pub(crate) fn cast_f32_f16(
        &self,
        input: DevPtr,
        out: DevPtr,
        n: u32,
    ) -> Result<(), RocmlError> {
        let cfg = LaunchConfig {
            grid: (n.div_ceil(LINEAR_BLOCK), 1, 1),
            block: (LINEAR_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut params = kernel_params!(input, out, n);
        // SAFETY: params matches cast_f32_f16's signature (const float*,
        // half*, unsigned); no block-size constraint.
        unsafe { self.cast_f32_f16_fn.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }

    /// Launches only the partial (split-K) pass — `Kernels::attn_decode_f16`
    /// runs the (dtype-independent) reduce pass itself.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attn_decode_partial_f16(
        &self,
        q: DevPtr,
        k_layer: DevPtr,
        v_layer: DevPtr,
        partial_out: DevPtr,
        partial_m: DevPtr,
        partial_l: DevPtr,
        n_kv_heads: u32,
        group: u32,
        head_dim: u32,
        max_seq: u32,
        cur_len: u32,
        split_len: u32,
        n_splits: u32,
        scale: f32,
    ) -> Result<(), RocmlError> {
        let cfg = LaunchConfig {
            grid: (n_kv_heads, n_splits, 1),
            block: (32, group, 1),
            shared_mem_bytes: 2 * ATTN_DECODE_TILE_T * head_dim * size_of::<f32>() as u32,
        };
        let mut params = kernel_params!(
            q,
            k_layer,
            v_layer,
            partial_out,
            partial_m,
            partial_l,
            n_kv_heads,
            group,
            head_dim,
            max_seq,
            cur_len,
            split_len,
            n_splits,
            scale
        );
        // SAFETY: params matches attn_decode_partial_f16's signature (const
        // float*, two const half*, three float*, seven unsigned, float);
        // block = (32, group, 1) matches the kernel's warp-per-q-head design.
        unsafe {
            self.attn_decode_partial_f16_fn
                .launch(&cfg, &mut params, None)
        }
        .map_err(Into::into)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attn_prefill_f16(
        &self,
        q: DevPtr,
        k_layer: DevPtr,
        v_layer: DevPtr,
        out: DevPtr,
        n_kv_heads: u32,
        group: u32,
        head_dim: u32,
        max_seq: u32,
        chunk_len: u32,
        pos_base: u32,
        scale: f32,
    ) -> Result<(), RocmlError> {
        let cfg = LaunchConfig {
            grid: (n_kv_heads, chunk_len.div_ceil(ATTN_PREFILL_ROW_TILE), 1),
            block: (32, group, ATTN_PREFILL_ROW_TILE),
            shared_mem_bytes: 2 * ATTN_DECODE_TILE_T * head_dim * size_of::<f32>() as u32,
        };
        let mut params = kernel_params!(
            q, k_layer, v_layer, out, n_kv_heads, group, head_dim, max_seq, chunk_len, pos_base,
            scale
        );
        // SAFETY: params matches attn_prefill_f16's signature (const float*,
        // two const half*, float*, six unsigned, float); block =
        // (32, group, ATTN_PREFILL_ROW_TILE) as `attn_prefill`'s f32 sibling.
        unsafe { self.attn_prefill_f16_fn.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }
}
