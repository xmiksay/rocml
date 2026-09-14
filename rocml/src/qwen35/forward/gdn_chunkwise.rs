//! Host orchestration for the chunkwise (blocked delta-rule) Gated Delta Net
//! recurrence (issue #6's chunkwise rewrite) — the seven-kernel pipeline
//! `kernels/gdn_chunkwise.hip` implements, replacing
//! `gdn_recurrence_chunk_f32`'s token-serial-inside-chunk loop with O(tile^2)
//! parallel matmul-shaped kernels. See that file's module doc for the
//! algebra (a direct transcription of HF transformers'
//! `torch_chunk_gated_delta_rule` / llama.cpp's `build_delta_net_chunking`).
//!
//! `chunk_len` here can be up to `CHUNK_CAP` (`forward_chunk`'s documented
//! contract) even though every current caller passes `PREFILL_CHUNK_SIZE`
//! (128) or less: [`gdn_chunkwise_step`] sub-chunks any request bigger than
//! `GDN_RECUR_TILE` into `GDN_RECUR_TILE`-sized tiles, running the pipeline
//! once per tile and carrying `state` between them exactly like a
//! prompt-level chunk boundary already does — the tile cap comes from the
//! triangular-inverse kernel needing a whole `tile x tile` f32 matrix in
//! gfx1101's 64KB LDS budget (see `chunk_scratch.rs`'s `GDN_RECUR_TILE` doc).

use super::chunk_scratch::{ChunkScratch, GDN_RECUR_TILE};
use super::gdn_chunkwise_kernels::GdnChunkwiseKernels;
use super::gdn_chunkwise_kernels_wmma::GdnChunkwiseWmmaKernels;
use crate::error::RocmlError;
use crate::forward::kernels::offset;
use crate::qwen35::cache::GdnLayerState;
use crate::qwen35::config::GdnConfig;

const L2_NORM_EPS: f32 = 1e-6;

/// Per-model WMMA dispatch decisions, computed once per [`gdn_chunkwise_step`]
/// call (not per tile) from `GdnConfig`'s load-time-constant head dims — see
/// that function's doc comment for each flag's eligibility bound.
#[derive(Clone, Copy)]
struct WmmaDispatch {
    state: bool,
    output: bool,
    ut_build: bool,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gdn_chunkwise_step(
    cw: &GdnChunkwiseKernels,
    cw_wmma: &GdnChunkwiseWmmaKernels,
    gdn: &GdnConfig,
    state: &mut GdnLayerState,
    scratch: &mut ChunkScratch,
    chunk_len: u32,
) -> Result<(), RocmlError> {
    // gdn-wmma round: stage G (state update) routes through the matrix-core
    // kernel whenever this model's head dims are both multiples of 16 (a
    // per-model load-time constant, checked once per call rather than per
    // tile).
    let use_wmma = gdn.head_k_dim.is_multiple_of(16) && gdn.head_v_dim.is_multiple_of(16);
    // gdn-wmma-lds round: stages B (`ut_build`)/F (`output`) also have WMMA
    // kernels now, but the naive (`gdn_chunkwise_wmma.hip`) ones measured
    // slower than scalar — only the LDS-staged follow-up
    // (`gdn_chunkwise_{output,ut_build}_wmma_lds.hip`) is wired in below, and
    // only within its LDS design's correctness bound: `LDS_FREE`(128) caps
    // the free axis of every operand it stages (see
    // `gdn_chunkwise_wmma_lds_common.h`'s module doc) — `head_k_dim`/
    // `head_v_dim` bigger than that would silently leave part of the output
    // uncomputed, so this is a dispatch-time correctness gate, not a perf
    // tuning knob. `tile_len` itself is architecturally always <=128
    // (`GDN_RECUR_TILE` above) so it never needs its own check here.
    const LDS_MAX_DIM: u32 = 128;
    let dispatch = WmmaDispatch {
        state: use_wmma,
        output: use_wmma && gdn.head_k_dim <= LDS_MAX_DIM && gdn.head_v_dim <= LDS_MAX_DIM,
        ut_build: gdn.head_k_dim.is_multiple_of(16) && gdn.head_k_dim <= LDS_MAX_DIM,
    };
    let mut tile_start = 0u32;
    while tile_start < chunk_len {
        let tile_len = (chunk_len - tile_start).min(GDN_RECUR_TILE);
        run_tile(
            cw, cw_wmma, dispatch, gdn, state, scratch, tile_start, tile_len,
        )?;
        tile_start += tile_len;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_tile(
    cw: &GdnChunkwiseKernels,
    cw_wmma: &GdnChunkwiseWmmaKernels,
    dispatch: WmmaDispatch,
    gdn: &GdnConfig,
    state: &mut GdnLayerState,
    scratch: &mut ChunkScratch,
    tile_start: u32,
    tile_len: u32,
) -> Result<(), RocmlError> {
    let h = gdn.num_v_heads;
    let hk = gdn.num_k_heads;
    let sk = gdn.head_k_dim;
    let sv = gdn.head_v_dim;
    let conv_dim = gdn.conv_dim;
    let key_dim = gdn.key_dim;

    let conv_out = offset(&scratch.gdn_conv_out, (tile_start * conv_dim) as usize);
    let beta = offset(&scratch.gdn_beta, (tile_start * h) as usize);
    let g = offset(&scratch.gdn_g, (tile_start * h) as usize);
    let y = offset(&scratch.gdn_y, (tile_start * gdn.value_dim) as usize);

    let q_norm = offset(&scratch.gdn_cw_q_norm, 0);
    let k_norm = offset(&scratch.gdn_cw_k_norm, 0);
    let k_beta = offset(&scratch.gdn_cw_k_beta, 0);
    let g_cum = offset(&scratch.gdn_cw_g_cum, 0);
    let cum_decay_exp = offset(&scratch.gdn_cw_cum_decay_exp, 0);
    let state_decay = offset(&scratch.gdn_cw_state_decay, 0);
    let kb = offset(&scratch.gdn_cw_kb, 0);
    let kq = offset(&scratch.gdn_cw_kq, 0);
    let v_new = offset(&scratch.gdn_cw_v_new, 0);
    let state_ptr = offset(&state.state, 0);

    cw.prep_point(
        conv_out,
        beta,
        q_norm,
        k_norm,
        k_beta,
        h,
        hk,
        sk,
        conv_dim,
        key_dim,
        tile_len,
        L2_NORM_EPS,
    )?;
    cw.prep_cumsum(g, g_cum, cum_decay_exp, state_decay, h, tile_len)?;
    if dispatch.ut_build {
        cw_wmma.ut_build_lds(q_norm, k_norm, k_beta, g_cum, kb, kq, h, sk, tile_len)?;
    } else {
        cw.ut_build(q_norm, k_norm, k_beta, g_cum, kb, kq, h, sk, tile_len)?;
    }
    cw.tinv(kb, h, tile_len)?;
    cw.uv_vnew(
        kb, // now Tinv, in place
        conv_out,
        beta,
        k_beta,
        cum_decay_exp,
        state_ptr,
        v_new,
        h,
        hk,
        sk,
        sv,
        conv_dim,
        key_dim,
        tile_len,
    )?;
    if dispatch.output {
        cw_wmma.output_lds(
            q_norm,
            cum_decay_exp,
            state_ptr,
            kq,
            v_new,
            y,
            h,
            sk,
            sv,
            tile_len,
        )?;
    } else {
        cw.output(
            q_norm,
            cum_decay_exp,
            state_ptr,
            kq,
            v_new,
            y,
            h,
            sk,
            sv,
            tile_len,
        )?;
    }
    if dispatch.state {
        cw_wmma.state_update(
            k_norm,
            cum_decay_exp,
            state_decay,
            v_new,
            state_ptr,
            h,
            sk,
            sv,
            tile_len,
        )
    } else {
        cw.state_update(
            k_norm,
            cum_decay_exp,
            state_decay,
            v_new,
            state_ptr,
            h,
            sk,
            sv,
            tile_len,
        )
    }
}
