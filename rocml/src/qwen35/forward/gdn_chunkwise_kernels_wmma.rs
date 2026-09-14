//! Typed launch helpers for GDN chunkwise-recurrence stages B/F/G's WMMA
//! variants — a separate struct/file from `gdn_chunkwise_kernels.rs`'s
//! `GdnChunkwiseKernels` (not an extra `impl` block on that type) since its
//! fields are private to that module; `gdn_chunkwise.rs`'s `run_tile` holds
//! both and dispatches per call.
//!
//! **G** (`gdn_chunkwise_state_wmma_f32`, `kernels/gdn_chunkwise_wmma.hip`,
//! gdn-wmma round, issue #6): one dense matmul with minimal cross-block
//! operand reuse — won cleanly even with that file's naive, direct-from-
//! global-memory fragment loaders (186.6us/call -> 92.3us/call, ~2x on
//! ornith-9b's real shape).
//!
//! **B/F** (`ut_build_lds`/`output_lds`, `kernels/gdn_chunkwise_
//! {ut_build,output}_wmma_lds.hip`, gdn-wmma-lds round, the follow-up this
//! file's original doc flagged): the *naive* WMMA kernels for these two
//! stages (still in `gdn_chunkwise_wmma.hip`, correctness-tested by
//! `gdn_chunkwise_wmma.rs` but never wired in) measured 8.2x/2.4x *slower*
//! than scalar — each reads 2-3 shared operands (`k_norm`/`state`/`v_new`)
//! redundantly from global memory once per 16x16-output-tile block, with no
//! cross-block reuse. The LDS-staged rewrite wired in here fixes exactly
//! that: one workgroup per `head` stages every shared operand into LDS once
//! (`gdn_chunkwise_wmma_lds_common.h`'s module doc has the full design and
//! LDS-budget accounting), and *that* wins decisively — real-shape
//! `rocprofv3 --kernel-trace` measurement on ornith-9b (depth 8192, `n=1536`
//! calls each): `ut_build` scalar 197.1us/call -> LDS-staged WMMA
//! 84.5us/call (2.33x); `output` scalar 297.2us/call -> LDS-staged WMMA
//! 100.0us/call (2.97x). End-to-end (interleaved single-process A/B vs
//! origin/main, ornith-9b Q4_K_M): prefill 852.8->895.8 tok/s (+5.0%) @
//! depth 2048, 661.1->688.1 (+4.1%) @ 8192, 510.3->525.7 (+3.0%) @ 16384;
//! qwen3.5-2b 3276.9->3465.4 (+5.8%) @ 2048. Decode flat throughout (see
//! `.claude/CLAUDE.md`'s gdn-wmma-lds round entry for the full table).
//! Worklist @ depth 8192: `gdn-recur` wasted time 1898.7ms->1425.5ms
//! (-24.9%), 4.7%->6.2% efficient — still the largest remaining prefill
//! item (`tinv`'s serial forward-substitution and `uv_vnew`'s recurrent-
//! state-feeding chained matmuls are both still scalar, see below).
//!
//! **D+E** (`uv_vnew`) was evaluated again this round but still not
//! attempted: chains three dependent matmuls through a shared intermediate
//! that would need materializing into LDS as a fresh WMMA operand between
//! phases (this file's B/F design only ever re-stages *existing* global
//! operands, never a just-computed accumulator) — real additional
//! engineering risk on top of feeding the *recurrent* state carried across
//! every tile of a prompt, judged out of scope given B/F's own budget was
//! already spent proving out the harder-than-expected LDS-materialization
//! step for a from-global operand. Flagged for a future round if `tinv`
//! (this pipeline's now-largest single kernel, 372.99us/call, unchanged
//! since the `gdn-recur-occupancy` round evaluated and rejected a
//! block-recursive rewrite) or `uv_vnew` (294.55us/call) are ever revisited.

use rocml_hip::{kernel_params, LaunchConfig, Module};

use crate::error::RocmlError;
use crate::forward::kernels::{load, DevPtr};

/// Must match `kernels/gdn_chunkwise_wmma_lds_common.h`'s `LDS_FREE`/
/// `K_SLICE` constants — the LDS-staged kernels' shared-memory footprint is a
/// compile-time-fixed function of those two, not of the runtime `tile_len`/
/// `head_k_dim`/`head_v_dim` args (unlike `tinv`'s `tile_len`-sized dynamic
/// LDS array), so it's recomputed here from the same two numbers rather than
/// threaded through as a kernel argument.
const LDS_FREE: u32 = 128;
const K_SLICE: u32 = 64;
const LDS_TILE_BYTES: u32 = LDS_FREE * K_SLICE * 2; // f16

pub struct GdnChunkwiseWmmaKernels {
    _mod_state: Module,
    state_fn: rocml_hip::Function,
    _mod_output_lds: Module,
    output_lds_fn: rocml_hip::Function,
    _mod_ut_build_lds: Module,
    ut_build_lds_fn: rocml_hip::Function,
}

fn ceil_div16(n: u32) -> u32 {
    n.div_ceil(16)
}

impl GdnChunkwiseWmmaKernels {
    pub fn load_all() -> Result<Self, RocmlError> {
        let (_mod_state, state_fn) = load(
            rocml_kernels::GDN_CW_STATE_WMMA_F32_HSACO,
            rocml_kernels::GDN_CW_STATE_WMMA_F32_KERNEL,
        )?;
        let (_mod_output_lds, output_lds_fn) = load(
            rocml_kernels::GDN_CW_OUTPUT_WMMA_LDS_F32_HSACO,
            rocml_kernels::GDN_CW_OUTPUT_WMMA_LDS_F32_KERNEL,
        )?;
        let (_mod_ut_build_lds, ut_build_lds_fn) = load(
            rocml_kernels::GDN_CW_UT_BUILD_WMMA_LDS_F32_HSACO,
            rocml_kernels::GDN_CW_UT_BUILD_WMMA_LDS_F32_KERNEL,
        )?;
        Ok(Self {
            _mod_state,
            state_fn,
            _mod_output_lds,
            output_lds_fn,
            _mod_ut_build_lds,
            ut_build_lds_fn,
        })
    }

    /// `gdn_chunkwise_state_wmma_f32`: grid =
    /// `(num_v_heads, ceil(head_k_dim/16), ceil(head_v_dim/16))`, block =
    /// `(32, 1, 1)`. Mutates `state` in place.
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
            grid: (num_v_heads, ceil_div16(head_k_dim), ceil_div16(head_v_dim)),
            block: (32, 1, 1),
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
        // SAFETY: params matches gdn_chunkwise_state_wmma_f32's signature
        // (four const float*, float*, four unsigned).
        unsafe { self.state_fn.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }

    /// `gdn_chunkwise_output_wmma_lds_f32` (gdn-wmma-lds round): grid =
    /// `(num_v_heads, 1, 1)`, block = `(32, 16, 1)`, dynamic shared memory
    /// `2 * LDS_TILE_BYTES` (32KB — two operands staged at a time, see the
    /// kernel's module doc). Caller must ensure `head_k_dim`/`head_v_dim`
    /// are both multiples of 16 and `<= 128` (`gdn_chunkwise.rs`'s dispatch
    /// gate) — this staged design's correctness precondition, not just its
    /// perf tuning.
    #[allow(clippy::too_many_arguments)]
    pub fn output_lds(
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
            grid: (num_v_heads, 1, 1),
            block: (32, 16, 1),
            shared_mem_bytes: 2 * LDS_TILE_BYTES,
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
        // SAFETY: params matches gdn_chunkwise_output_wmma_lds_f32's
        // signature (five const float*, float*, four unsigned);
        // shared_mem_bytes matches the kernel's two staged LDS tiles.
        unsafe { self.output_lds_fn.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }

    /// `gdn_chunkwise_ut_build_wmma_lds_f32` (gdn-wmma-lds round): grid =
    /// `(num_v_heads, 1, 1)`, block = `(32, 16, 1)`, dynamic shared memory
    /// `3 * LDS_TILE_BYTES` (48KB — three operands staged at a time). Same
    /// `head_k_dim` eligibility bound as `output_lds` above.
    #[allow(clippy::too_many_arguments)]
    pub fn ut_build_lds(
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
            grid: (num_v_heads, 1, 1),
            block: (32, 16, 1),
            shared_mem_bytes: 3 * LDS_TILE_BYTES,
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
        // SAFETY: params matches gdn_chunkwise_ut_build_wmma_lds_f32's
        // signature (four const float*, two float*, three unsigned);
        // shared_mem_bytes matches the kernel's three staged LDS tiles.
        unsafe { self.ut_build_lds_fn.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }
}
