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
use crate::error::RocmlError;
use crate::forward::kernels::offset;
use crate::qwen35::cache::GdnLayerState;
use crate::qwen35::config::GdnConfig;

const L2_NORM_EPS: f32 = 1e-6;

#[allow(clippy::too_many_arguments)]
pub(crate) fn gdn_chunkwise_step(
    cw: &GdnChunkwiseKernels,
    gdn: &GdnConfig,
    state: &mut GdnLayerState,
    scratch: &mut ChunkScratch,
    chunk_len: u32,
) -> Result<(), RocmlError> {
    let mut tile_start = 0u32;
    while tile_start < chunk_len {
        let tile_len = (chunk_len - tile_start).min(GDN_RECUR_TILE);
        run_tile(cw, gdn, state, scratch, tile_start, tile_len)?;
        tile_start += tile_len;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_tile(
    cw: &GdnChunkwiseKernels,
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
    cw.ut_build(q_norm, k_norm, k_beta, g_cum, kb, kq, h, sk, tile_len)?;
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
