//! Flash-attention-style tiled prefill attention (issue #6 flash-prefill
//! round): host launcher for `kernels/attn_prefill_flash.hip`'s `BR`-row
//! query tile x `BC`-row KV tile design with split-K over the KV/depth axis
//! — see that kernel's module doc for the full design and why WMMA was
//! evaluated and rejected. Split into its own file for the same file-size
//! reason `kernels_kv.rs`/`kernels_quant.rs` already are (`kernels.rs` is
//! well past the workspace's 400-line cap) — `Kernels` holds a `pub(crate)
//! flash: FlashPrefillKernels` field and callers (`qwen35::forward`'s
//! `attention_chunk.rs`) reach these methods directly through it rather
//! than via pass-through wrappers on `Kernels` itself, to avoid growing
//! that file at all.

use std::mem::size_of;

use rocml_hip::{kernel_params, LaunchConfig, Module};

use super::kernels::{load, DevPtr};
use crate::error::RocmlError;

/// Query rows per workgroup, handled via `blockDim.z` (hardware threads,
/// the same mechanism `attn_prefill.hip`'s `ROW_TILE` uses — a
/// register-resident-loop first draft of this measured worse, see the
/// kernel source's module doc). Must match `BR` in
/// `kernels/attn_prefill_flash.hip`.
pub(crate) const ATTN_PREFILL_FLASH_BR: u32 = 8;
/// KV tile rows staged into LDS per outer iteration — must match `BC`.
pub(crate) const ATTN_PREFILL_FLASH_BC: u32 = 16;
/// Hard cap on split count, bounding `ChunkScratch`'s partial-buffer sizes
/// (`chunk_scratch.rs` allocates for this many splits up front, times
/// `CHUNK_CAP` rows times `n_heads` — see that file for the byte-budget
/// accounting). Smaller than `attn_decode`'s `ATTN_DECODE_MAX_SPLITS`(32):
/// a decode partial buffer is `n_heads * 32 * head_dim` floats (trivial),
/// but a prefill one is `CHUNK_CAP * n_heads * n_splits * head_dim` —
/// scales with the whole chunk, not one token, so this stays deliberately
/// small (32MB of scratch on Ornith's shape at this cap).
const MAX_SPLITS: u32 = 4;
/// Target total (kv_head, row_tile, split) workgroups at large depth —
/// same reasoning as `attn_decode`'s `ATTN_DECODE_TARGET_WORKGROUPS`, just
/// applied to `n_kv_heads * num_row_tiles` existing parallelism instead of
/// `n_kv_heads` alone (a prefill chunk's row-tiles are already independent
/// workgroups, unlike decode's single query).
const TARGET_WORKGROUPS: u32 = 240;
/// Below this many KV positions in a single split, splitting further isn't
/// worth the reduce kernel's extra per-split merge cost.
const MIN_SPLIT_LEN: u32 = 512;

/// Picks `(n_splits, split_len)` for one `attn_prefill_flash*` call, from
/// the *chunk's* deepest row (`pos_base + chunk_len`) — every row-tile in
/// the launch shares these same split boundaries; shallower row-tiles just
/// see most of their high-numbered splits contribute nothing (correct by
/// construction, see the kernel's module doc).
pub(crate) fn attn_prefill_flash_splits(
    n_kv_heads: u32,
    chunk_len: u32,
    deepest_end: u32,
) -> (u32, u32) {
    let num_row_tiles = chunk_len.div_ceil(ATTN_PREFILL_FLASH_BR).max(1);
    let by_occupancy = TARGET_WORKGROUPS.div_ceil((n_kv_heads * num_row_tiles).max(1));
    let by_min_len = deepest_end.div_ceil(MIN_SPLIT_LEN);
    let n_splits = by_occupancy.min(by_min_len).clamp(1, MAX_SPLITS);
    let split_len = deepest_end.div_ceil(n_splits);
    (n_splits, split_len)
}

/// Upper bound `ChunkScratch` must size its `attn_flash_partial_*` buffers
/// for: `CHUNK_CAP * n_heads * ATTN_PREFILL_FLASH_MAX_SPLITS[* head_dim]`.
pub(crate) const ATTN_PREFILL_FLASH_MAX_SPLITS: u32 = MAX_SPLITS;

pub(crate) struct FlashPrefillKernels {
    _mod_partial_f32: Module,
    partial_f32_fn: rocml_hip::Function,
    _mod_partial_f16: Module,
    partial_f16_fn: rocml_hip::Function,
    _mod_reduce: Module,
    reduce_fn: rocml_hip::Function,
}

impl FlashPrefillKernels {
    pub(crate) fn load_all() -> Result<Self, RocmlError> {
        let (_mod_partial_f32, partial_f32_fn) = load(
            rocml_kernels::ATTN_PREFILL_FLASH_PARTIAL_F32_HSACO,
            rocml_kernels::ATTN_PREFILL_FLASH_PARTIAL_F32_KERNEL,
        )?;
        let (_mod_partial_f16, partial_f16_fn) = load(
            rocml_kernels::ATTN_PREFILL_FLASH_PARTIAL_F16_HSACO,
            rocml_kernels::ATTN_PREFILL_FLASH_PARTIAL_F16_KERNEL,
        )?;
        let (_mod_reduce, reduce_fn) = load(
            rocml_kernels::ATTN_PREFILL_FLASH_REDUCE_F32_HSACO,
            rocml_kernels::ATTN_PREFILL_FLASH_REDUCE_F32_KERNEL,
        )?;
        Ok(Self {
            _mod_partial_f32,
            partial_f32_fn,
            _mod_partial_f16,
            partial_f16_fn,
            _mod_reduce,
            reduce_fn,
        })
    }

    /// f32-KV-cache variant (the parity-test reference path). `q` is
    /// `[chunk_len, n_heads, head_dim]`; `k_layer`/`v_layer` are one
    /// layer's whole KV-cache buffer; `out` is `[chunk_len, n_heads,
    /// head_dim]`; `partial_out`/`partial_m`/`partial_l` are caller-owned
    /// scratch sized for `CHUNK_CAP * n_heads * ATTN_PREFILL_FLASH_MAX_SPLITS[*
    /// head_dim]` (see `ChunkScratch`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attn_prefill_flash(
        &self,
        q: DevPtr,
        k_layer: DevPtr,
        v_layer: DevPtr,
        out: DevPtr,
        partial_out: DevPtr,
        partial_m: DevPtr,
        partial_l: DevPtr,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        max_seq: u32,
        chunk_len: u32,
        pos_base: u32,
        scale: f32,
    ) -> Result<(), RocmlError> {
        self.run(
            &self.partial_f32_fn,
            q,
            k_layer,
            v_layer,
            out,
            partial_out,
            partial_m,
            partial_l,
            n_heads,
            n_kv_heads,
            head_dim,
            max_seq,
            chunk_len,
            pos_base,
            scale,
        )
    }

    /// f16-KV-cache variant (issue #3's default dtype).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attn_prefill_flash_f16(
        &self,
        q: DevPtr,
        k_layer: DevPtr,
        v_layer: DevPtr,
        out: DevPtr,
        partial_out: DevPtr,
        partial_m: DevPtr,
        partial_l: DevPtr,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        max_seq: u32,
        chunk_len: u32,
        pos_base: u32,
        scale: f32,
    ) -> Result<(), RocmlError> {
        self.run(
            &self.partial_f16_fn,
            q,
            k_layer,
            v_layer,
            out,
            partial_out,
            partial_m,
            partial_l,
            n_heads,
            n_kv_heads,
            head_dim,
            max_seq,
            chunk_len,
            pos_base,
            scale,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        partial_fn: &rocml_hip::Function,
        q: DevPtr,
        k_layer: DevPtr,
        v_layer: DevPtr,
        out: DevPtr,
        partial_out: DevPtr,
        partial_m: DevPtr,
        partial_l: DevPtr,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        max_seq: u32,
        chunk_len: u32,
        pos_base: u32,
        scale: f32,
    ) -> Result<(), RocmlError> {
        let group = n_heads / n_kv_heads;
        let (n_splits, split_len) =
            attn_prefill_flash_splits(n_kv_heads, chunk_len, pos_base + chunk_len);
        let num_row_tiles = chunk_len.div_ceil(ATTN_PREFILL_FLASH_BR);

        let partial_cfg = LaunchConfig {
            grid: (n_kv_heads, num_row_tiles, n_splits),
            block: (32, group, ATTN_PREFILL_FLASH_BR),
            shared_mem_bytes: 2 * ATTN_PREFILL_FLASH_BC * head_dim * size_of::<f32>() as u32,
        };
        let mut partial_params = kernel_params!(
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
            chunk_len,
            pos_base,
            split_len,
            n_splits,
            scale
        );
        // SAFETY: params matches attn_prefill_flash_partial_{f32,f16}'s
        // signature (three const pointers, three float*, seven unsigned,
        // float); block = (32, group, ATTN_PREFILL_FLASH_BR) matches the
        // kernel's row-tiled design; grid.z = n_splits.
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
        // (three const float*, float*, three unsigned); grid = (chunk_len,
        // n_heads); dtype-independent (only reads the f32 partials the
        // launch above just wrote).
        unsafe { self.reduce_fn.launch(&reduce_cfg, &mut reduce_params, None) }.map_err(Into::into)
    }
}
