//! Typed launch helpers for the KIVI-style mixed KV cache's kernels (issue
//! #2): quantize-on-evict (`kernels/kv_quant.hip`) and the fused
//! mixed-KV decode-attention kernel (`kernels/attn_decode_mixed.hip`). Kept
//! separate from `kernels.rs`/`crate::forward::kernels` purely for the
//! workspace's 400-line file cap, mirroring `chunk_kernels.rs`.

use std::mem::size_of;

use rocml_hip::{kernel_params, LaunchConfig, Module};

use crate::error::RocmlError;
use crate::forward::kernels::{load, DevPtr};

/// Must match `TILE_T` in `kernels/attn_decode_mixed.hip`.
const MIXED_TILE_T: u32 = 8;

pub struct MixedKernels {
    _mod_evict_k: Module,
    evict_k_fn: rocml_hip::Function,
    _mod_evict_v_q8: Module,
    evict_v_q8_fn: rocml_hip::Function,
    _mod_evict_v_q4: Module,
    evict_v_q4_fn: rocml_hip::Function,
    _mod_decode_q8: Module,
    decode_q8_fn: rocml_hip::Function,
    _mod_decode_q4: Module,
    decode_q4_fn: rocml_hip::Function,
}

impl MixedKernels {
    pub fn load_all() -> Result<Self, RocmlError> {
        let (_mod_evict_k, evict_k_fn) = load(
            rocml_kernels::QUANTIZE_EVICT_K_Q8_HSACO,
            rocml_kernels::QUANTIZE_EVICT_K_Q8_KERNEL,
        )?;
        let (_mod_evict_v_q8, evict_v_q8_fn) = load(
            rocml_kernels::QUANTIZE_EVICT_V_Q8_HSACO,
            rocml_kernels::QUANTIZE_EVICT_V_Q8_KERNEL,
        )?;
        let (_mod_evict_v_q4, evict_v_q4_fn) = load(
            rocml_kernels::QUANTIZE_EVICT_V_Q4_HSACO,
            rocml_kernels::QUANTIZE_EVICT_V_Q4_KERNEL,
        )?;
        let (_mod_decode_q8, decode_q8_fn) = load(
            rocml_kernels::ATTN_DECODE_PARTIAL_MIXED_Q8_HSACO,
            rocml_kernels::ATTN_DECODE_PARTIAL_MIXED_Q8_KERNEL,
        )?;
        let (_mod_decode_q4, decode_q4_fn) = load(
            rocml_kernels::ATTN_DECODE_PARTIAL_MIXED_Q4_HSACO,
            rocml_kernels::ATTN_DECODE_PARTIAL_MIXED_Q4_KERNEL,
        )?;
        Ok(Self {
            _mod_evict_k,
            evict_k_fn,
            _mod_evict_v_q8,
            evict_v_q8_fn,
            _mod_evict_v_q4,
            evict_v_q4_fn,
            _mod_decode_q8,
            decode_q8_fn,
            _mod_decode_q4,
            decode_q4_fn,
        })
    }

    /// `quantize_evict_k_f16_to_q8`: one block per kv_head, `head_dim`
    /// threads. See `kernels/kv_quant.hip`'s module doc for the layout.
    #[allow(clippy::too_many_arguments)]
    pub fn quantize_evict_k(
        &self,
        window_k: DevPtr,
        bulk_codes: DevPtr,
        bulk_scales: DevPtr,
        n_kv_heads: u32,
        window_len: u32,
        head_dim: u32,
        bulk_cap: u32,
        num_blocks_total: u32,
        block_idx: u32,
    ) -> Result<(), RocmlError> {
        let cfg = LaunchConfig {
            grid: (n_kv_heads, 1, 1),
            block: (head_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut params = kernel_params!(
            window_k,
            bulk_codes,
            bulk_scales,
            n_kv_heads,
            window_len,
            head_dim,
            bulk_cap,
            num_blocks_total,
            block_idx
        );
        // SAFETY: params matches quantize_evict_k_f16_to_q8's signature
        // (const half*, signed char*, float*, unsigned x5); block =
        // (head_dim, 1, 1) — one thread per channel.
        unsafe { self.evict_k_fn.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }

    /// `quantize_evict_v_f16_to_q8`/`_q4`: one warp per (kv_head, token).
    #[allow(clippy::too_many_arguments)]
    pub fn quantize_evict_v(
        &self,
        v_bits: u8,
        window_v: DevPtr,
        bulk_codes: DevPtr,
        bulk_scales: DevPtr,
        n_kv_heads: u32,
        window_len: u32,
        head_dim: u32,
        bulk_cap: u32,
        block_idx: u32,
    ) -> Result<(), RocmlError> {
        let cfg = LaunchConfig {
            grid: (n_kv_heads, window_len, 1),
            block: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut params = kernel_params!(
            window_v,
            bulk_codes,
            bulk_scales,
            n_kv_heads,
            window_len,
            head_dim,
            bulk_cap,
            block_idx
        );
        let function = if v_bits == 8 {
            &self.evict_v_q8_fn
        } else {
            &self.evict_v_q4_fn
        };
        // SAFETY: params matches quantize_evict_v_f16_to_q{8,4}'s signature
        // (const half*, {signed,unsigned} char*, float*, unsigned x4);
        // block = (32, 1, 1), one warp per (kv_head, token).
        unsafe { function.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }

    /// `attn_decode_partial_mixed_q8`/`_q4`: same split-K launch shape as
    /// `Kernels::attn_decode_f16`'s partial pass — the caller runs the
    /// (dtype-independent) reduce pass itself, exactly as
    /// `Kernels::attn_decode_f16` does.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode_partial_mixed(
        &self,
        v_bits: u8,
        q: DevPtr,
        sink_k: DevPtr,
        sink_v: DevPtr,
        window_k: DevPtr,
        window_v: DevPtr,
        bulk_k_codes: DevPtr,
        bulk_k_scales: DevPtr,
        bulk_v_codes: DevPtr,
        bulk_v_scales: DevPtr,
        partial_out: DevPtr,
        partial_m: DevPtr,
        partial_l: DevPtr,
        n_kv_heads: u32,
        group: u32,
        head_dim: u32,
        sink_len: u32,
        window_len: u32,
        window_base: u32,
        bulk_cap: u32,
        num_blocks_total: u32,
        cur_len: u32,
        split_len: u32,
        n_splits: u32,
        scale: f32,
    ) -> Result<(), RocmlError> {
        let cfg = LaunchConfig {
            grid: (n_kv_heads, n_splits, 1),
            block: (32, group, 1),
            shared_mem_bytes: 2 * MIXED_TILE_T * head_dim * size_of::<f32>() as u32,
        };
        let mut params = kernel_params!(
            q,
            sink_k,
            sink_v,
            window_k,
            window_v,
            bulk_k_codes,
            bulk_k_scales,
            bulk_v_codes,
            bulk_v_scales,
            partial_out,
            partial_m,
            partial_l,
            n_kv_heads,
            group,
            head_dim,
            sink_len,
            window_len,
            window_base,
            bulk_cap,
            num_blocks_total,
            cur_len,
            split_len,
            n_splits,
            scale
        );
        let function = if v_bits == 8 {
            &self.decode_q8_fn
        } else {
            &self.decode_q4_fn
        };
        // SAFETY: params matches attn_decode_partial_mixed_q{8,4}'s
        // signature exactly (see that kernel's own doc comment); block =
        // (32, group, 1) matches the warp-per-q-head design shared with
        // attn_decode_partial_f16.
        unsafe { function.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }
}
