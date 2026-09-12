//! Typed launch helpers for the chunked-prefill-only kernels: the batched
//! GDN conv1d chunk kernel (`kernels/gdn_chunk.hip` — the recurrence's
//! chunk kernel that used to live here moved to `gdn_chunkwise_kernels.rs`,
//! see that module's doc comment) and the small reshape/broadcast kernels a
//! T-token chunk needs that a single decode step didn't — per-head
//! extraction, batched KV-cache append, and the GDN gate's per-token
//! broadcast (`kernels/chunk_reshape.hip`). Split out of `kernels.rs`
//! (which already sits close to the workspace's 400-line cap) so the
//! decode-step and chunk-step kernel sets stay in their own files.

use rocml_hip::{kernel_params, LaunchConfig, Module};

use crate::error::RocmlError;
use crate::forward::kernels::{load, DevPtr};

const LINEAR_BLOCK: u32 = 256;

pub struct ChunkKernels {
    _mod_conv_chunk: Module,
    conv_chunk_fn: rocml_hip::Function,
    _mod_extract_heads: Module,
    extract_heads_fn: rocml_hip::Function,
    _mod_scatter_kv: Module,
    scatter_kv_fn: rocml_hip::Function,
    _mod_scatter_kv_f16: Module,
    scatter_kv_f16_fn: rocml_hip::Function,
    _mod_gate_chunk: Module,
    gate_chunk_fn: rocml_hip::Function,
}

impl ChunkKernels {
    pub fn load_all() -> Result<Self, RocmlError> {
        let (_mod_conv_chunk, conv_chunk_fn) = load(
            rocml_kernels::GDN_CONV1D_CHUNK_F32_HSACO,
            rocml_kernels::GDN_CONV1D_CHUNK_F32_KERNEL,
        )?;
        let (_mod_extract_heads, extract_heads_fn) = load(
            rocml_kernels::EXTRACT_HEADS_F32_HSACO,
            rocml_kernels::EXTRACT_HEADS_F32_KERNEL,
        )?;
        let (_mod_scatter_kv, scatter_kv_fn) = load(
            rocml_kernels::SCATTER_KV_CHUNK_F32_HSACO,
            rocml_kernels::SCATTER_KV_CHUNK_F32_KERNEL,
        )?;
        let (_mod_scatter_kv_f16, scatter_kv_f16_fn) = load(
            rocml_kernels::SCATTER_KV_CHUNK_F16_HSACO,
            rocml_kernels::SCATTER_KV_CHUNK_F16_KERNEL,
        )?;
        let (_mod_gate_chunk, gate_chunk_fn) = load(
            rocml_kernels::GDN_GATE_CHUNK_F32_HSACO,
            rocml_kernels::GDN_GATE_CHUNK_F32_KERNEL,
        )?;

        Ok(Self {
            _mod_conv_chunk,
            conv_chunk_fn,
            _mod_extract_heads,
            extract_heads_fn,
            _mod_scatter_kv,
            scatter_kv_fn,
            _mod_scatter_kv_f16,
            scatter_kv_f16_fn,
            _mod_gate_chunk,
            gate_chunk_fn,
        })
    }

    /// `causal_conv1d_chunk_f32`: batched GDN causal-conv over a
    /// `chunk_len`-token chunk in one launch, mutating `conv_state` in
    /// place — the prefill-chunk sibling of `HybridKernels::gdn_conv1d_decode`.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_conv1d_chunk(
        &self,
        x: DevPtr,
        conv_state: DevPtr,
        weight: DevPtr,
        out: DevPtr,
        channels: u32,
        kernel_size: u32,
        chunk_len: u32,
    ) -> Result<(), RocmlError> {
        let cfg = LaunchConfig {
            grid: (channels.div_ceil(LINEAR_BLOCK), 1, 1),
            block: (LINEAR_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut params =
            kernel_params!(x, conv_state, weight, out, channels, kernel_size, chunk_len);
        // SAFETY: params matches causal_conv1d_chunk_f32's signature (const
        // float*, float*, const float*, float*, unsigned x3).
        unsafe { self.conv_chunk_fn.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }

    /// `extract_heads_f32(src, dst, tokens, heads, head_dim, src_stride,
    /// src_offset)`: pulls one `head_dim`-wide field out of a fused per-head
    /// `[..., src_stride]` projection row into a compact `[tokens, heads,
    /// head_dim]` buffer — see the kernel source's doc comment.
    #[allow(clippy::too_many_arguments)]
    pub fn extract_heads(
        &self,
        src: DevPtr,
        dst: DevPtr,
        tokens: u32,
        heads: u32,
        head_dim: u32,
        src_stride: u32,
        src_offset: u32,
    ) -> Result<(), RocmlError> {
        let total = tokens * heads * head_dim;
        let cfg = LaunchConfig {
            grid: (total.div_ceil(LINEAR_BLOCK), 1, 1),
            block: (LINEAR_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut params = kernel_params!(src, dst, tokens, heads, head_dim, src_stride, src_offset);
        // SAFETY: params matches extract_heads_f32's signature (const
        // float*, float*, unsigned x5).
        unsafe { self.extract_heads_fn.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }

    /// `scatter_kv_chunk_f32`: batch-appends a chunk's `[chunk_len,
    /// n_kv_heads, head_dim]` K/V into the `[n_kv_heads, max_seq, head_dim]`
    /// cache planes at `[pos_base, pos_base+chunk_len)` — the chunked
    /// sibling of `AttnPlane::append`'s per-token host loop.
    #[allow(clippy::too_many_arguments)]
    pub fn scatter_kv_chunk(
        &self,
        k_src: DevPtr,
        v_src: DevPtr,
        k_dst: DevPtr,
        v_dst: DevPtr,
        n_kv_heads: u32,
        head_dim: u32,
        max_seq: u32,
        chunk_len: u32,
        pos_base: u32,
    ) -> Result<(), RocmlError> {
        let total = chunk_len * n_kv_heads * head_dim;
        let cfg = LaunchConfig {
            grid: (total.div_ceil(LINEAR_BLOCK), 1, 1),
            block: (LINEAR_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut params = kernel_params!(
            k_src, v_src, k_dst, v_dst, n_kv_heads, head_dim, max_seq, chunk_len, pos_base
        );
        // SAFETY: params matches scatter_kv_chunk_f32's signature (four
        // const/mut float*, unsigned x5).
        unsafe { self.scatter_kv_fn.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }

    /// f16-cache sibling of [`Self::scatter_kv_chunk`] (issue #3's default
    /// KV dtype) — same launch shape, dispatching to `scatter_kv_chunk_f16`.
    #[allow(clippy::too_many_arguments)]
    pub fn scatter_kv_chunk_f16(
        &self,
        k_src: DevPtr,
        v_src: DevPtr,
        k_dst: DevPtr,
        v_dst: DevPtr,
        n_kv_heads: u32,
        head_dim: u32,
        max_seq: u32,
        chunk_len: u32,
        pos_base: u32,
    ) -> Result<(), RocmlError> {
        let total = chunk_len * n_kv_heads * head_dim;
        let cfg = LaunchConfig {
            grid: (total.div_ceil(LINEAR_BLOCK), 1, 1),
            block: (LINEAR_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut params = kernel_params!(
            k_src, v_src, k_dst, v_dst, n_kv_heads, head_dim, max_seq, chunk_len, pos_base
        );
        // SAFETY: params matches scatter_kv_chunk_f16's signature (two
        // const float*, two half*, unsigned x5).
        unsafe { self.scatter_kv_f16_fn.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }

    /// `gdn_gate_chunk_f32`: batched sibling of `HybridKernels::gdn_gate`
    /// (`a_raw`/`b_raw`/`beta_out`/`g_out` are `[chunk_len, heads]`,
    /// `a_log`/`dt_bias` stay `[heads]` and broadcast per token).
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_gate_chunk(
        &self,
        a_raw: DevPtr,
        b_raw: DevPtr,
        a_log: DevPtr,
        dt_bias: DevPtr,
        beta_out: DevPtr,
        g_out: DevPtr,
        heads: u32,
        chunk_len: u32,
    ) -> Result<(), RocmlError> {
        let total = heads * chunk_len;
        let cfg = LaunchConfig {
            grid: (total.div_ceil(LINEAR_BLOCK), 1, 1),
            block: (LINEAR_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut params =
            kernel_params!(a_raw, b_raw, a_log, dt_bias, beta_out, g_out, heads, total);
        // SAFETY: params matches gdn_gate_chunk_f32's signature (four const
        // float*, two float*, unsigned x2).
        unsafe { self.gate_chunk_fn.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }
}
