//! Typed launch helpers for `kernels/attn_prefill_flash_mixed.hip` — the
//! chunked-prefill sibling of `kernels_mixed.rs`'s
//! `MixedKernels::attn_decode_partial_mixed`, reusing the dense chunked
//! prefill path's `attn_prefill_flash_splits` split-K sizing and
//! `attn_prefill_flash_reduce_f32` reduce kernel (both already loaded via
//! `crate::forward::kernels_flash`, dtype/source-independent — see that
//! module's doc comment). Split into its own file (mirroring why
//! `kernels_flash.rs`/`kernels_mixed.rs` are already split out) purely for
//! the workspace's 400-line file cap.

use std::mem::size_of;

use rocml_hip::{kernel_params, LaunchConfig, Module};

use crate::error::RocmlError;
use crate::forward::kernels::{load, DevPtr};
use crate::forward::kernels_flash::{
    attn_prefill_flash_br, attn_prefill_flash_splits, ATTN_PREFILL_FLASH_BC, ATTN_PREFILL_FLASH_BR,
};

pub(crate) struct FlashPrefillMixedKernels {
    _mod_partial_q8: Module,
    partial_q8_fn: rocml_hip::Function,
    _mod_partial_q4: Module,
    partial_q4_fn: rocml_hip::Function,
    /// `BR=4` siblings (`attn_prefill_flash_mixed_narrow.hip`) — see
    /// `crate::forward::kernels_flash::attn_prefill_flash_br`'s doc comment
    /// for when these run instead (this path has no shallow-depth,
    /// non-flash fallback to fall back to, unlike the dense path).
    _mod_partial_q8_br4: Module,
    partial_q8_br4_fn: rocml_hip::Function,
    _mod_partial_q4_br4: Module,
    partial_q4_br4_fn: rocml_hip::Function,
    _mod_reduce: Module,
    reduce_fn: rocml_hip::Function,
}

impl FlashPrefillMixedKernels {
    pub(crate) fn load_all() -> Result<Self, RocmlError> {
        let (_mod_partial_q8, partial_q8_fn) = load(
            rocml_kernels::ATTN_PREFILL_FLASH_PARTIAL_MIXED_Q8_HSACO,
            rocml_kernels::ATTN_PREFILL_FLASH_PARTIAL_MIXED_Q8_KERNEL,
        )?;
        let (_mod_partial_q4, partial_q4_fn) = load(
            rocml_kernels::ATTN_PREFILL_FLASH_PARTIAL_MIXED_Q4_HSACO,
            rocml_kernels::ATTN_PREFILL_FLASH_PARTIAL_MIXED_Q4_KERNEL,
        )?;
        let (_mod_partial_q8_br4, partial_q8_br4_fn) = load(
            rocml_kernels::ATTN_PREFILL_FLASH_PARTIAL_MIXED_Q8_BR4_HSACO,
            rocml_kernels::ATTN_PREFILL_FLASH_PARTIAL_MIXED_Q8_BR4_KERNEL,
        )?;
        let (_mod_partial_q4_br4, partial_q4_br4_fn) = load(
            rocml_kernels::ATTN_PREFILL_FLASH_PARTIAL_MIXED_Q4_BR4_HSACO,
            rocml_kernels::ATTN_PREFILL_FLASH_PARTIAL_MIXED_Q4_BR4_KERNEL,
        )?;
        // Same code object/entry point `crate::forward::kernels_flash`
        // already loads for the dense path — loaded again here as an
        // independent `Module`/`Function` handle (cheap, matches this
        // codebase's existing precedent of loading a shared hsaco's entry
        // point more than once, e.g. `MixedKernels`' own decode kernels).
        let (_mod_reduce, reduce_fn) = load(
            rocml_kernels::ATTN_PREFILL_FLASH_REDUCE_F32_HSACO,
            rocml_kernels::ATTN_PREFILL_FLASH_REDUCE_F32_KERNEL,
        )?;
        Ok(Self {
            _mod_partial_q8,
            partial_q8_fn,
            _mod_partial_q4,
            partial_q4_fn,
            _mod_partial_q8_br4,
            partial_q8_br4_fn,
            _mod_partial_q4_br4,
            partial_q4_br4_fn,
            _mod_reduce,
            reduce_fn,
        })
    }

    /// `attn_prefill_flash_partial_mixed_q8`/`_q4` + the shared
    /// `attn_prefill_flash_reduce_f32` reduce pass. `q` is `[chunk_len,
    /// n_heads, head_dim]`; the `sink_*`/`window_*`/`bulk_*` pointers and
    /// shape scalars mirror `MixedKernels::attn_decode_partial_mixed`
    /// exactly (same `MixedPtrs` fields, see `cache_mixed.rs`); `out` and
    /// the `partial_*` scratch reuse `ChunkScratch`'s existing
    /// `attn_concat`/`attn_flash_partial_{out,m,l}` buffers — already sized
    /// for `CHUNK_CAP * n_heads * ATTN_PREFILL_FLASH_MAX_SPLITS[*
    /// head_dim]`, the same upper bound this call needs.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attn_prefill_flash_mixed(
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
        out: DevPtr,
        partial_out: DevPtr,
        partial_m: DevPtr,
        partial_l: DevPtr,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        sink_len: u32,
        window_len: u32,
        window_base: u32,
        bulk_cap: u32,
        num_blocks_total: u32,
        chunk_len: u32,
        pos_base: u32,
        scale: f32,
    ) -> Result<(), RocmlError> {
        let group = n_heads / n_kv_heads;
        // See `attn_prefill_flash_br`'s doc comment — a wide-GQA checkpoint
        // (Ornith-1.5-35B-A3B's `group=8`) needs the narrower `BR=4` kernel.
        let br = attn_prefill_flash_br(group)?;
        let (n_splits, split_len) =
            attn_prefill_flash_splits(n_kv_heads, chunk_len, pos_base + chunk_len, br);
        let num_row_tiles = chunk_len.div_ceil(br);

        let partial_cfg = LaunchConfig {
            grid: (n_kv_heads, num_row_tiles, n_splits),
            block: (32, group, br),
            shared_mem_bytes: 2 * ATTN_PREFILL_FLASH_BC * head_dim * size_of::<f32>() as u32,
        };
        let mut partial_params = kernel_params!(
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
            chunk_len,
            pos_base,
            split_len,
            n_splits,
            scale
        );
        let partial_fn = match (v_bits, br == ATTN_PREFILL_FLASH_BR) {
            (8, true) => &self.partial_q8_fn,
            (8, false) => &self.partial_q8_br4_fn,
            (_, true) => &self.partial_q4_fn,
            (_, false) => &self.partial_q4_br4_fn,
        };
        // SAFETY: params matches attn_prefill_flash_partial_mixed_{q8,q4}
        // [_br4]'s signature exactly (see that kernel's own doc comment);
        // block = (32, group, br) matches whichever kernel `br` selected;
        // grid.z = n_splits.
        unsafe { partial_fn.launch(&partial_cfg, &mut partial_params, None) }?;

        let reduce_cfg = LaunchConfig {
            grid: (chunk_len, n_heads, 1),
            block: (head_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut reduce_params = kernel_params!(
            partial_out,
            partial_m,
            partial_l,
            out,
            n_heads,
            head_dim,
            n_splits
        );
        // SAFETY: params matches attn_prefill_flash_reduce_f32's signature
        // (three const float*, float*, three unsigned); dtype/source
        // independent — only reads the f32 partials the launch above wrote.
        unsafe { self.reduce_fn.launch(&reduce_cfg, &mut reduce_params, None) }.map_err(Into::into)
    }
}
