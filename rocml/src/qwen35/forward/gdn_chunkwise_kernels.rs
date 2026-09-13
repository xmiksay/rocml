//! Typed launch helpers for the chunkwise (blocked delta-rule) Gated Delta
//! Net recurrence kernels (`kernels/gdn_chunkwise.hip`) — split out of
//! `chunk_kernels.rs` purely for the 400-line cap, same reasoning as
//! `kernels_quant.rs`/`kernels_kv.rs` in the decode-path kernel set. See the
//! `.hip` file's module doc for the seven-kernel pipeline this wraps and
//! `gdn_chunkwise.rs` for the host-side orchestration (including the
//! `chunk_len > GDN_RECUR_TILE` sub-chunking loop).

use rocml_hip::{kernel_params, LaunchConfig, Module};

use crate::error::RocmlError;
use crate::forward::kernels::{load, DevPtr};

/// Must match `kernels/gdn_chunkwise.hip`'s `#define UT_BUILD_J_PER_BLOCK`.
const UT_BUILD_J_PER_BLOCK: u32 = 8;

pub struct GdnChunkwiseKernels {
    _mod_prep_point: Module,
    prep_point_fn: rocml_hip::Function,
    _mod_prep_cumsum: Module,
    prep_cumsum_fn: rocml_hip::Function,
    _mod_ut_build: Module,
    ut_build_fn: rocml_hip::Function,
    _mod_tinv: Module,
    tinv_fn: rocml_hip::Function,
    _mod_uv_vnew: Module,
    uv_vnew_fn: rocml_hip::Function,
    _mod_output: Module,
    output_fn: rocml_hip::Function,
    _mod_state: Module,
    state_fn: rocml_hip::Function,
}

impl GdnChunkwiseKernels {
    pub fn load_all() -> Result<Self, RocmlError> {
        let (_mod_prep_point, prep_point_fn) = load(
            rocml_kernels::GDN_CW_PREP_POINT_F32_HSACO,
            rocml_kernels::GDN_CW_PREP_POINT_F32_KERNEL,
        )?;
        let (_mod_prep_cumsum, prep_cumsum_fn) = load(
            rocml_kernels::GDN_CW_PREP_CUMSUM_F32_HSACO,
            rocml_kernels::GDN_CW_PREP_CUMSUM_F32_KERNEL,
        )?;
        let (_mod_ut_build, ut_build_fn) = load(
            rocml_kernels::GDN_CW_UT_BUILD_F32_HSACO,
            rocml_kernels::GDN_CW_UT_BUILD_F32_KERNEL,
        )?;
        let (_mod_tinv, tinv_fn) = load(
            rocml_kernels::GDN_CW_TINV_F32_HSACO,
            rocml_kernels::GDN_CW_TINV_F32_KERNEL,
        )?;
        let (_mod_uv_vnew, uv_vnew_fn) = load(
            rocml_kernels::GDN_CW_UV_VNEW_F32_HSACO,
            rocml_kernels::GDN_CW_UV_VNEW_F32_KERNEL,
        )?;
        let (_mod_output, output_fn) = load(
            rocml_kernels::GDN_CW_OUTPUT_F32_HSACO,
            rocml_kernels::GDN_CW_OUTPUT_F32_KERNEL,
        )?;
        let (_mod_state, state_fn) = load(
            rocml_kernels::GDN_CW_STATE_F32_HSACO,
            rocml_kernels::GDN_CW_STATE_F32_KERNEL,
        )?;
        Ok(Self {
            _mod_prep_point,
            prep_point_fn,
            _mod_prep_cumsum,
            prep_cumsum_fn,
            _mod_ut_build,
            ut_build_fn,
            _mod_tinv,
            tinv_fn,
            _mod_uv_vnew,
            uv_vnew_fn,
            _mod_output,
            output_fn,
            _mod_state,
            state_fn,
        })
    }

    /// `gdn_chunkwise_prep_point_f32`: grid = `(num_v_heads, tile_len, 1)`,
    /// block = `(32, 1, 1)` — one warp per (head, token) pair. Embarrassingly
    /// parallel per-token work (L2-norm(Q,K), `k_beta`), split out of the
    /// original fused `prep` kernel specifically to widen its grid past
    /// `num_v_heads` blocks — see the kernel source's module doc.
    #[allow(clippy::too_many_arguments)]
    pub fn prep_point(
        &self,
        conv_out: DevPtr,
        beta: DevPtr,
        q_norm: DevPtr,
        k_norm: DevPtr,
        k_beta: DevPtr,
        num_v_heads: u32,
        num_k_heads: u32,
        head_k_dim: u32,
        conv_dim: u32,
        key_dim: u32,
        tile_len: u32,
        l2_eps: f32,
    ) -> Result<(), RocmlError> {
        let cfg = LaunchConfig {
            grid: (num_v_heads, tile_len, 1),
            block: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut params = kernel_params!(
            conv_out,
            beta,
            q_norm,
            k_norm,
            k_beta,
            num_v_heads,
            num_k_heads,
            head_k_dim,
            conv_dim,
            key_dim,
            tile_len,
            l2_eps
        );
        // SAFETY: params matches gdn_chunkwise_prep_point_f32's signature
        // (two const float*, three float*, five unsigned, float).
        unsafe { self.prep_point_fn.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }

    /// `gdn_chunkwise_prep_cumsum_f32`: block size must equal `tile_len`
    /// exactly (the in-kernel Hillis-Steele g-cumsum scan needs every thread
    /// index `< tile_len` present and no more). Independent of
    /// [`Self::prep_point`] — no ordering requirement between the two.
    /// `state_decay` (per-wave-efficiency round, gdn) is `exp(g_last -
    /// g_cum[t])`, computed once here instead of redundantly inside
    /// `gdn_chunkwise_state_f32` — see that kernel's doc comment.
    pub fn prep_cumsum(
        &self,
        g: DevPtr,
        g_cum: DevPtr,
        cum_decay_exp: DevPtr,
        state_decay: DevPtr,
        num_v_heads: u32,
        tile_len: u32,
    ) -> Result<(), RocmlError> {
        let cfg = LaunchConfig {
            grid: (num_v_heads, 1, 1),
            block: (tile_len, 1, 1),
            shared_mem_bytes: tile_len * size_of_f32(),
        };
        let mut params =
            kernel_params!(g, g_cum, cum_decay_exp, state_decay, num_v_heads, tile_len);
        // SAFETY: params matches gdn_chunkwise_prep_cumsum_f32's signature
        // (one const float*, three float*, two unsigned); block == tile_len
        // as required.
        unsafe { self.prep_cumsum_fn.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }

    /// `gdn_chunkwise_ut_build_f32`: grid = `(num_v_heads, tile_len, 1)`,
    /// block = `(32, UT_BUILD_J_PER_BLOCK, 1)` — must match the kernel
    /// source's `#define UT_BUILD_J_PER_BLOCK` exactly (a warp per column,
    /// looped over `tile_len` columns; see that kernel's doc comment for
    /// why this replaced a one-thread-per-column launch shape).
    #[allow(clippy::too_many_arguments)]
    pub fn ut_build(
        &self,
        q_norm: DevPtr,
        k_norm: DevPtr,
        k_beta: DevPtr,
        g_cum: DevPtr,
        kb: DevPtr,
        kq: DevPtr,
        num_v_heads: u32,
        head_k_dim: u32,
        tile_len: u32,
    ) -> Result<(), RocmlError> {
        let cfg = LaunchConfig {
            grid: (num_v_heads, tile_len, 1),
            block: (32, UT_BUILD_J_PER_BLOCK, 1),
            shared_mem_bytes: 0,
        };
        let mut params = kernel_params!(
            q_norm,
            k_norm,
            k_beta,
            g_cum,
            kb,
            kq,
            num_v_heads,
            head_k_dim,
            tile_len
        );
        // SAFETY: params matches gdn_chunkwise_ut_build_f32's signature
        // (four const float*, two float*, three unsigned).
        unsafe { self.ut_build_fn.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }

    /// `gdn_chunkwise_tinv_f32`: one block per head, block = `(tile_len, 1,
    /// 1)`, dynamic shared memory `tile_len * tile_len *
    /// sizeof(f32)` bytes (<= 64KB at `tile_len <= 128`).
    pub fn tinv(&self, kb: DevPtr, num_v_heads: u32, tile_len: u32) -> Result<(), RocmlError> {
        let cfg = LaunchConfig {
            grid: (num_v_heads, 1, 1),
            block: (tile_len, 1, 1),
            shared_mem_bytes: tile_len * tile_len * size_of_f32(),
        };
        let mut params = kernel_params!(kb, num_v_heads, tile_len);
        // SAFETY: params matches gdn_chunkwise_tinv_f32's signature (float*,
        // two unsigned); shared_mem_bytes matches the kernel's dynamic LDS
        // array size exactly.
        unsafe { self.tinv_fn.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }

    /// `gdn_chunkwise_uv_vnew_f32` (fused D+E — see the kernel source's doc
    /// comment for why): grid = `(num_v_heads, tile_len, 1)`, block =
    /// `(max(head_k_dim, head_v_dim), 1, 1)`, dynamic shared memory
    /// `head_k_dim * sizeof(f32)` bytes.
    #[allow(clippy::too_many_arguments)]
    pub fn uv_vnew(
        &self,
        tinv: DevPtr,
        conv_out: DevPtr,
        beta: DevPtr,
        k_beta: DevPtr,
        cum_decay_exp: DevPtr,
        state: DevPtr,
        v_new: DevPtr,
        num_v_heads: u32,
        num_k_heads: u32,
        head_k_dim: u32,
        head_v_dim: u32,
        conv_dim: u32,
        key_dim: u32,
        tile_len: u32,
    ) -> Result<(), RocmlError> {
        let block = head_k_dim.max(head_v_dim);
        let cfg = LaunchConfig {
            grid: (num_v_heads, tile_len, 1),
            block: (block, 1, 1),
            shared_mem_bytes: head_k_dim * size_of_f32(),
        };
        let mut params = kernel_params!(
            tinv,
            conv_out,
            beta,
            k_beta,
            cum_decay_exp,
            state,
            v_new,
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            conv_dim,
            key_dim,
            tile_len
        );
        // SAFETY: params matches gdn_chunkwise_uv_vnew_f32's signature (six
        // const float*, float*, seven unsigned); shared_mem_bytes matches
        // the kernel's dynamic `kcd_row` LDS array size exactly.
        unsafe { self.uv_vnew_fn.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }

    /// `gdn_chunkwise_output_f32`: grid = `(num_v_heads, tile_len, 1)`,
    /// block = `(head_v_dim, 1, 1)`.
    #[allow(clippy::too_many_arguments)]
    pub fn output(
        &self,
        q_norm: DevPtr,
        cum_decay_exp: DevPtr,
        state: DevPtr,
        kq: DevPtr,
        v_new: DevPtr,
        y: DevPtr,
        num_v_heads: u32,
        head_k_dim: u32,
        head_v_dim: u32,
        tile_len: u32,
    ) -> Result<(), RocmlError> {
        let cfg = LaunchConfig {
            grid: (num_v_heads, tile_len, 1),
            block: (head_v_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut params = kernel_params!(
            q_norm,
            cum_decay_exp,
            state,
            kq,
            v_new,
            y,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            tile_len
        );
        // SAFETY: params matches gdn_chunkwise_output_f32's signature (five
        // const float*, float*, four unsigned).
        unsafe { self.output_fn.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }

    /// `gdn_chunkwise_state_f32`: grid = `(num_v_heads, head_k_dim, 1)`,
    /// block = `(head_v_dim, 1, 1)`. Mutates `state` in place. Takes
    /// `cum_decay_exp`/`state_decay` (both from [`Self::prep_cumsum`]) in
    /// place of the raw `g_cum` the kernel used to take and re-derive
    /// `chunk_decay`/`gdiff` from itself — see the kernel's doc comment.
    #[allow(clippy::too_many_arguments)]
    pub fn state_update(
        &self,
        k_norm: DevPtr,
        cum_decay_exp: DevPtr,
        state_decay: DevPtr,
        v_new: DevPtr,
        state: DevPtr,
        num_v_heads: u32,
        head_k_dim: u32,
        head_v_dim: u32,
        tile_len: u32,
    ) -> Result<(), RocmlError> {
        let cfg = LaunchConfig {
            grid: (num_v_heads, head_k_dim, 1),
            block: (head_v_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut params = kernel_params!(
            k_norm,
            cum_decay_exp,
            state_decay,
            v_new,
            state,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            tile_len
        );
        // SAFETY: params matches gdn_chunkwise_state_f32's signature (four
        // const float*, float*, four unsigned).
        unsafe { self.state_fn.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }
}

fn size_of_f32() -> u32 {
    std::mem::size_of::<f32>() as u32
}
