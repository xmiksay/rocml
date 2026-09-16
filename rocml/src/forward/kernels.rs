//! Loads every HIP kernel the forward pass needs once at model-load time,
//! and wraps each one in a typed launch helper so `attention.rs`/`ffn.rs`
//! don't touch raw `kernel_params!`/`LaunchConfig` directly. Every helper
//! takes plain device pointers (`DevPtr`) rather than `&DeviceBuffer<T>` so
//! the same helper serves both whole-buffer calls and offset sub-buffer
//! calls (e.g. one KV-cache head's plane inside a layer's full cache).

use std::ffi::c_void;
use std::mem::size_of;

use rocml_core::quant::GgmlDType;
use rocml_hip::{kernel_params, DeviceBuffer, LaunchConfig, Module};

use super::kernels_flash::FlashPrefillKernels;
use super::kernels_kv::{KvF16Kernels, ATTN_PREFILL_ROW_TILE};
pub(crate) use super::kernels_mmq::MmqScratch;
use super::kernels_quant::QuantKernels;
pub(crate) use super::kernels_splitk::SplitKScratch;
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
pub(super) const REDUCE_BLOCK: u32 = 128;
/// Block size for plain elementwise/grid-stride kernels with no
/// power-of-two constraint.
pub(super) const LINEAR_BLOCK: u32 = 256;

/// Rows of K/V staged into LDS at a time by `attn_decode_partial_f32` — must
/// match `TILE_T` in `kernels/attn_decode.hip` exactly, since this constant
/// is what sizes the dynamic shared memory the launch requests.
pub(super) const ATTN_DECODE_TILE_T: u32 = 8;
/// Below this many cached positions, decode attention runs as a single
/// split (`n_splits == 1`): the whole point of splitting the sequence axis
/// is to manufacture enough independent workgroups to fill gfx1101's 60 CUs
/// when `n_kv_heads` alone (as few as 2-8) isn't enough, and a shallow
/// decode doesn't have that occupancy problem in the first place — splitting
/// it anyway would only add the reduce kernel's (tiny but nonzero) overhead
/// for no benefit.
const ATTN_DECODE_MIN_SPLIT_LEN: u32 = 128;
/// Target total workgroups (`n_kv_heads * n_splits`) for the partial kernel
/// at large `cur_len` — comfortably above gfx1101's 60 CUs so there's enough
/// independent work to hide each workgroup's K/V global-memory latency
/// behind other resident waves, without over-splitting into workgroups so
/// small their own launch/reduce overhead starts to dominate again (the
/// exact failure mode this kernel replaces).
const ATTN_DECODE_TARGET_WORKGROUPS: u32 = 240;
/// Hard cap on split count: bounds the `partial_out`/`partial_m`/`partial_l`
/// scratch buffers' size (`Scratch` allocates for this many splits up
/// front) and the reduce kernel's per-head serial merge loop.
pub const ATTN_DECODE_MAX_SPLITS: u32 = 32;

/// Chooses `(n_splits, split_len)` for one `attn_decode` call — see the
/// constants above for the reasoning. `n_splits` never exceeds
/// `cur_len.div_ceil(ATTN_DECODE_MIN_SPLIT_LEN)`, which is always `<=
/// cur_len` for `cur_len >= 1`, so every split before the last is
/// non-empty (`split_len >= 1`) and no workgroup gets a `start >= cur_len`
/// range from this heuristic alone (`attn_decode_partial_f32` also handles
/// that case correctly regardless, for callers — e.g. kernel tests — that
/// pick `n_splits` directly).
pub(crate) fn attn_decode_splits(n_kv_heads: u32, cur_len: u32) -> (u32, u32) {
    let by_occupancy = ATTN_DECODE_TARGET_WORKGROUPS.div_ceil(n_kv_heads.max(1));
    let by_min_len = cur_len.div_ceil(ATTN_DECODE_MIN_SPLIT_LEN);
    let n_splits = by_occupancy
        .min(by_min_len)
        .clamp(1, ATTN_DECODE_MAX_SPLITS);
    let split_len = cur_len.div_ceil(n_splits);
    (n_splits, split_len)
}

pub struct Kernels {
    _mod_embedding: Module,
    embedding_fn: rocml_hip::Function,
    _mod_rmsnorm: Module,
    rmsnorm_fn: rocml_hip::Function,
    _mod_gemv_f32: Module,
    gemv_f32_fn: rocml_hip::Function,
    _mod_gemv_f16: Module,
    gemv_f16_fn: rocml_hip::Function,
    _mod_gemm_f16: Module,
    gemm_f16_fn: rocml_hip::Function,
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
    _mod_attn_decode_partial: Module,
    attn_decode_partial_fn: rocml_hip::Function,
    _mod_attn_decode_reduce: Module,
    attn_decode_reduce_fn: rocml_hip::Function,
    _mod_attn_prefill: Module,
    attn_prefill_fn: rocml_hip::Function,
    quant: QuantKernels,
    kv_f16: KvF16Kernels,
    /// `pub(crate)` (unlike the other kernel-owning fields above) so
    /// `qwen35::forward::attention_chunk` can call its methods directly
    /// instead of `kernels.rs` (already well past the 400-line cap)
    /// growing a full set of pass-through wrappers — see
    /// `kernels_flash.rs`'s module doc.
    pub(crate) flash: FlashPrefillKernels,
}

pub(crate) fn load(hsaco: &[u8], name: &str) -> Result<(Module, rocml_hip::Function), RocmlError> {
    let module = Module::load_from_bytes(hsaco)?;
    let function = module.get_function(name)?;
    Ok((module, function))
}

impl Kernels {
    pub fn load_all(mmq_enabled: bool) -> Result<Self, RocmlError> {
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
        let (_mod_gemm_f16, gemm_f16_fn) = load(
            rocml_kernels::GEMM_XWT_F16_HSACO,
            rocml_kernels::GEMM_XWT_F16_KERNEL,
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
        let (_mod_attn_decode_partial, attn_decode_partial_fn) = load(
            rocml_kernels::ATTN_DECODE_PARTIAL_F32_HSACO,
            rocml_kernels::ATTN_DECODE_PARTIAL_F32_KERNEL,
        )?;
        let (_mod_attn_decode_reduce, attn_decode_reduce_fn) = load(
            rocml_kernels::ATTN_DECODE_REDUCE_F32_HSACO,
            rocml_kernels::ATTN_DECODE_REDUCE_F32_KERNEL,
        )?;
        let (_mod_attn_prefill, attn_prefill_fn) = load(
            rocml_kernels::ATTN_PREFILL_F32_HSACO,
            rocml_kernels::ATTN_PREFILL_F32_KERNEL,
        )?;
        let quant = QuantKernels::load_all(mmq_enabled)?;
        let kv_f16 = KvF16Kernels::load_all()?;
        let flash = FlashPrefillKernels::load_all()?;

        Ok(Self {
            _mod_embedding,
            embedding_fn,
            _mod_rmsnorm,
            rmsnorm_fn,
            _mod_gemv_f32,
            gemv_f32_fn,
            _mod_gemv_f16,
            gemv_f16_fn,
            _mod_gemm_f16,
            gemm_f16_fn,
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
            _mod_attn_decode_partial,
            attn_decode_partial_fn,
            _mod_attn_decode_reduce,
            attn_decode_reduce_fn,
            _mod_attn_prefill,
            attn_prefill_fn,
            quant,
            kv_f16,
            flash,
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

    /// `y = W * x` where `W`'s rows are raw GGUF `dtype` blocks (Q8_0/Q4_K/
    /// Q5_K/Q6_K) — see [`LinearWeight`](crate::weights::LinearWeight),
    /// which is the only caller expected to hit this.
    pub fn gemv_quant(
        &self,
        dtype: GgmlDType,
        w: DevPtr,
        x: DevPtr,
        y: DevPtr,
        m: u32,
        n: u32,
    ) -> Result<(), RocmlError> {
        self.quant.gemv(dtype, w, x, y, m, n)
    }

    /// `gemm_xwt_f16(x, w, out, rows, m, n)`: `out[rows,m] = X[rows,n] * W^T`,
    /// the batched prefill-path sibling of [`Self::gemv_f16`].
    pub fn gemm_xwt_f16(
        &self,
        x: DevPtr,
        w: DevPtr,
        out: DevPtr,
        rows: u32,
        m: u32,
        n: u32,
    ) -> Result<(), RocmlError> {
        const TILE: u32 = 16;
        let cfg = LaunchConfig {
            grid: (m.div_ceil(TILE), rows.div_ceil(TILE), 1),
            block: (TILE, TILE, 1),
            shared_mem_bytes: 0,
        };
        let mut params = kernel_params!(x, w, out, rows, m, n);
        // SAFETY: params matches gemm_xwt_f16's signature (const float*,
        // const half*, float*, unsigned x3); block = (16, 16, 1) matches the
        // kernel's fixed TILE.
        unsafe { self.gemm_f16_fn.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }

    /// `out[rows,m] = X[rows,n] * dequant(W)^T` where `W`'s rows are raw
    /// GGUF `dtype` blocks — batched sibling of [`Self::gemv_quant`], see
    /// [`LinearWeight`](crate::weights::LinearWeight)`::matmul`. `allow_micro`
    /// (qwen35moe M4 lever 1) opts into the micro-tile WMMA fallback for
    /// shapes with `rows < 128` — see `kernels_quant_dispatch.rs`'s `gemm`
    /// doc. `LinearWeight::matmul` (every non-MoE caller) always passes
    /// `false`; only qwen35moe's grouped-by-expert batched GEMM
    /// (`qwen35::forward::moe_chunk`) passes `true`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_quant(
        &self,
        dtype: GgmlDType,
        x: DevPtr,
        w: DevPtr,
        out: DevPtr,
        rows: u32,
        m: u32,
        n: u32,
        mmq_scratch: MmqScratch,
        mmq_eligible: bool,
        splitk_scratch: SplitKScratch,
        allow_micro: bool,
    ) -> Result<(), RocmlError> {
        self.quant.gemm(
            dtype,
            x,
            w,
            out,
            rows,
            m,
            n,
            mmq_scratch,
            mmq_eligible,
            splitk_scratch,
            allow_micro,
        )
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

    /// Fused single-token causal attention (`kernels/attn_decode.hip`):
    /// replaces the old per-head `gemv_f32`+`softmax_varlen`+`gemv_t_f32`
    /// composition with two launches. `q` is `[n_heads, head_dim]`;
    /// `k_layer`/`v_layer` are one layer's whole KV-cache buffer, `[n_kv_heads,
    /// max_seq, head_dim]` (the layout `crate::cache::KvCache`/
    /// `qwen35::cache::AttnPlane` already use — a kv head's plane is
    /// `kvh * max_seq * head_dim` into the buffer, computed inside the
    /// kernel from `max_seq`, not passed per-head). `out` is `[n_heads,
    /// head_dim]`. `partial_out`/`partial_m`/`partial_l` are caller-owned
    /// scratch sized for `n_heads * ATTN_DECODE_MAX_SPLITS` (see
    /// `Scratch`) — reused across calls, never read back by the caller.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode(
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
        cur_len: u32,
        scale: f32,
    ) -> Result<(), RocmlError> {
        let group = n_heads / n_kv_heads;
        let (n_splits, split_len) = attn_decode_splits(n_kv_heads, cur_len);

        let partial_cfg = LaunchConfig {
            grid: (n_kv_heads, n_splits, 1),
            block: (32, group, 1),
            shared_mem_bytes: 2 * ATTN_DECODE_TILE_T * head_dim * size_of::<f32>() as u32,
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
            cur_len,
            split_len,
            n_splits,
            scale
        );
        // SAFETY: params matches attn_decode_partial_f32's signature (three
        // const float*, three float*, seven unsigned, float); block =
        // (32, group, 1) matches the kernel's warp-per-q-head design.
        unsafe {
            self.attn_decode_partial_fn
                .launch(&partial_cfg, &mut partial_params, None)
        }?;

        self.attn_decode_reduce(
            partial_out,
            partial_m,
            partial_l,
            out,
            n_heads,
            head_dim,
            n_splits,
        )
    }

    /// `attn_decode_reduce_f32`: merges per-split online-softmax partials
    /// into the final per-head output. Dtype-independent (only ever reads
    /// the f32 partial scratch buffers, never the cache itself) — shared by
    /// every KV storage scheme's partial kernel: [`Self::attn_decode`],
    /// [`Self::attn_decode_f16`], and the mixed-KV cache's
    /// `MixedKernels::attn_decode_partial_mixed` (issue #2), which calls
    /// this directly since it has no dtype-specific reduce pass of its own.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode_reduce(
        &self,
        partial_out: DevPtr,
        partial_m: DevPtr,
        partial_l: DevPtr,
        out: DevPtr,
        n_heads: u32,
        head_dim: u32,
        n_splits: u32,
    ) -> Result<(), RocmlError> {
        let reduce_cfg = LaunchConfig {
            grid: (n_heads, 1, 1),
            block: (head_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut reduce_params =
            kernel_params!(partial_out, partial_m, partial_l, out, head_dim, n_splits);
        // SAFETY: params matches attn_decode_reduce_f32's signature (three
        // const float*, float*, two unsigned); block = head_dim, one thread
        // per output element (head_dim <= 256 for every model this codebase
        // loads).
        unsafe {
            self.attn_decode_reduce_fn
                .launch(&reduce_cfg, &mut reduce_params, None)
        }
        .map_err(Into::into)
    }

    /// Batched causal attention for a prefill chunk (`kernels/attn_prefill.hip`):
    /// `chunk_len` new query rows (positions `pos_base..pos_base+chunk_len`)
    /// against the KV cache, which must already hold this chunk's own
    /// appended K/V (batch-append before calling this). `q`/`out` are
    /// `[chunk_len, n_heads, head_dim]`; `k_layer`/`v_layer` are one layer's
    /// whole KV-cache buffer, `[n_kv_heads, max_seq, head_dim]` — same
    /// layout and cache buffers `attn_decode` uses, so a layer can freely mix
    /// prefill chunks and decode steps against the same cache.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_prefill(
        &self,
        q: DevPtr,
        k_layer: DevPtr,
        v_layer: DevPtr,
        out: DevPtr,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        max_seq: u32,
        chunk_len: u32,
        pos_base: u32,
        scale: f32,
    ) -> Result<(), RocmlError> {
        let group = n_heads / n_kv_heads;
        let cfg = LaunchConfig {
            grid: (n_kv_heads, chunk_len.div_ceil(ATTN_PREFILL_ROW_TILE), 1),
            block: (32, group, ATTN_PREFILL_ROW_TILE),
            shared_mem_bytes: 2 * ATTN_DECODE_TILE_T * head_dim * size_of::<f32>() as u32,
        };
        let mut params = kernel_params!(
            q, k_layer, v_layer, out, n_kv_heads, group, head_dim, max_seq, chunk_len, pos_base,
            scale
        );
        // SAFETY: params matches attn_prefill_f32's signature (three const
        // float*, float*, six unsigned, float); block = (32, group,
        // ATTN_PREFILL_ROW_TILE) matches the kernel's row-tiled design;
        // shared_mem_bytes matches TILE_T(8) (ATTN_DECODE_TILE_T).
        unsafe { self.attn_prefill_fn.launch(&cfg, &mut params, None) }.map_err(Into::into)
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

    /// `cast_f32_f16(in, out, n)`: elementwise f32 -> f16, used by the
    /// default-dtype KV cache (issue #3) to cast a decode/prefill step's
    /// freshly projected f32 K/V into the cache's f16 storage — see
    /// [`KvF16Kernels`] (kept in its own small file; `kernels.rs` already
    /// sits at the workspace's 400-line file cap).
    pub fn cast_f32_f16(&self, input: DevPtr, out: DevPtr, n: u32) -> Result<(), RocmlError> {
        self.kv_f16.cast_f32_f16(input, out, n)
    }

    /// f16-KV-cache sibling of [`Self::attn_decode`] (issue #3's default
    /// dtype) — identical launch shape, dispatching to
    /// `attn_decode_partial_f16` instead; the reduce pass is shared
    /// unchanged (it never touches the cache, only the f32 partial scratch
    /// buffers).
    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode_f16(
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
        cur_len: u32,
        scale: f32,
    ) -> Result<(), RocmlError> {
        let group = n_heads / n_kv_heads;
        let (n_splits, split_len) = attn_decode_splits(n_kv_heads, cur_len);
        self.kv_f16.attn_decode_partial_f16(
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
            scale,
        )?;

        self.attn_decode_reduce(
            partial_out,
            partial_m,
            partial_l,
            out,
            n_heads,
            head_dim,
            n_splits,
        )
    }

    /// f16-KV-cache sibling of [`Self::attn_prefill`] (issue #3's default
    /// dtype).
    #[allow(clippy::too_many_arguments)]
    pub fn attn_prefill_f16(
        &self,
        q: DevPtr,
        k_layer: DevPtr,
        v_layer: DevPtr,
        out: DevPtr,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        max_seq: u32,
        chunk_len: u32,
        pos_base: u32,
        scale: f32,
    ) -> Result<(), RocmlError> {
        let group = n_heads / n_kv_heads;
        self.kv_f16.attn_prefill_f16(
            q, k_layer, v_layer, out, n_kv_heads, group, head_dim, max_seq, chunk_len, pos_base,
            scale,
        )
    }
}
