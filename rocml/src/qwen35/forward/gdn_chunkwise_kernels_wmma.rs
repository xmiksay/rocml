//! Typed launch helper for GDN chunkwise-recurrence stage G's WMMA variant
//! (`gdn_chunkwise_state_wmma_f32`, `kernels/gdn_chunkwise_wmma.hip`,
//! gdn-wmma round, issue #6) — a separate struct/file from
//! `gdn_chunkwise_kernels.rs`'s `GdnChunkwiseKernels` (not an extra `impl`
//! block on that type) since its fields are private to that module;
//! `gdn_chunkwise.rs`'s `run_tile` holds both and dispatches per call.
//!
//! Stages B (`ut_build`) and F (`output`) also have WMMA kernels in the
//! same `.hip` file, correctness-tested (`rocml-kernels/tests/
//! gdn_chunkwise_wmma.rs`) against the f64 reference — but **not** wired in
//! here: measured on ornith-9b's real shape (`rocprofv3 --kernel-trace`,
//! depth 8192), `ut_build_wmma` was 8.2x *slower* than the scalar kernel it
//! would replace (197.2us/call -> 1610.2us/call) and `output_wmma` was 2.4x
//! slower (294.0us/call -> 694.7us/call) — only `state_wmma` (this file)
//! won (186.6us/call -> 92.3us/call, ~2x). Root cause: unlike this file's
//! `state_update` (one dense matmul, minimal cross-block operand reuse),
//! `ut_build`/`output` each read 2-3 shared operands (`k_norm`/`state`/
//! `v_new`) redundantly from global memory once per 16x16-output-tile block
//! with no LDS staging or multi-warp fragment reuse (`gdn_chunkwise_wmma_
//! common.h`'s loaders are direct-from-global by design, sized for these
//! small once-per-tile operands, not for an operand a whole grid re-reads
//! many times) — correct, per `gdn_chunkwise_wmma.rs`'s test suite, but the
//! resulting memory traffic overwhelms the matrix-core throughput gain at
//! this size. A `gemm_xwt_wmma_impl.h`-style LDS-staged, multi-warp-per-
//! block rewrite might recover this, but is real additional engineering
//! risk/time out of this round's scope — kept as a correctness-proven,
//! perf-rejected prototype (same disposition this codebase already gives
//! the int8 MMQ GEMM prototype, `.claude/CLAUDE.md`'s WMMA-pipeline round).

use rocml_hip::{kernel_params, LaunchConfig, Module};

use crate::error::RocmlError;
use crate::forward::kernels::{load, DevPtr};

pub struct GdnChunkwiseWmmaKernels {
    _mod_state: Module,
    state_fn: rocml_hip::Function,
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
        Ok(Self {
            _mod_state,
            state_fn,
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
}
