//! GPU integration tests for the LDS-staged WMMA variant of stages B/D+E/F
//! (`kernels/gdn_chunkwise_{ut_build,uv_vnew,output}_wmma_lds.hip`,
//! gdn-wmma-lds round + gdn-uvvnew round, issue #6) against the same f64 CPU
//! reference `gdn_chunkwise.rs`/`gdn_chunkwise_wmma.rs` use. Mirrors every
//! shape/tile-length case there, including head dims that are *not*
//! multiples of 16 (this LDS design's own correctness bound is
//! `LDS_FREE`(128) on each operand's free axis, not 16-alignment — see
//! `gdn_chunkwise_wmma_lds_common.h`'s module doc) to validate the kernels
//! standalone, independent of the real host dispatch's additional
//! (perf-motivated) multiple-of-16 gate
//! (`rocml/src/qwen35/forward/gdn_chunkwise.rs`).
//!
//! The `long_chain_*` tests below are `uv_vnew`'s own arbiter beyond the
//! single-tile cases above: its output (`v_new`) feeds the recurrent
//! `state` carried across every tile of a prompt, so an f16-rounding error
//! that's negligible in one tile could in principle compound across many —
//! these run 64 consecutive tiles feeding `state` forward and compare
//! against a single f64 reference run over the whole concatenated sequence.
#[path = "gdn_chunkwise_support/mod.rs"]
mod support;

use rocml_hip::Device;
use support::wmma_lds::{run_chunkwise_gpu_wmma_lds, ChunkwiseWmmaLdsKernels};
use support::{assert_close, reference_sequential};

#[allow(clippy::too_many_arguments)]
fn run_case_wmma_lds(
    num_heads: u32,
    num_k_heads: u32,
    head_k_dim: u32,
    head_v_dim: u32,
    chunk_len: u32,
) {
    let _device = Device::new(0).expect("failed to select device 0");
    let k = ChunkwiseWmmaLdsKernels::load();

    let (h, hk, sk, sv, t) = (
        num_heads as usize,
        num_k_heads as usize,
        head_k_dim as usize,
        head_v_dim as usize,
        chunk_len as usize,
    );
    let key_dim = hk * sk;
    let value_dim = h * sv;
    let conv_dim = 2 * key_dim + value_dim;

    let conv_out_f32: Vec<f32> = (0..t * conv_dim)
        .map(|i| ((i % 23) as f32) * 0.07 - 0.75)
        .collect();
    let beta_f32: Vec<f32> = (0..t * h).map(|i| 0.1 + (i as f32 % 7.0) * 0.09).collect();
    let g_f32: Vec<f32> = (0..t * h)
        .map(|i| -0.02 - (i as f32 % 9.0) * 0.025)
        .collect();
    let init_state_f32: Vec<f32> = (0..h * sk * sv)
        .map(|i| ((i % 13) as f32) * 0.04 - 0.24)
        .collect();

    let conv_out_f64: Vec<f64> = conv_out_f32.iter().map(|&v| v as f64).collect();
    let beta_f64: Vec<f64> = beta_f32.iter().map(|&v| v as f64).collect();
    let g_f64: Vec<f64> = g_f32.iter().map(|&v| v as f64).collect();
    let init_state_f64: Vec<f64> = init_state_f32.iter().map(|&v| v as f64).collect();

    let (expected_y, expected_state) = reference_sequential(
        h,
        hk,
        sk,
        sv,
        t,
        &conv_out_f64,
        &beta_f64,
        &g_f64,
        &init_state_f64,
    );

    let (actual_y, actual_state) = run_chunkwise_gpu_wmma_lds(
        &k,
        num_heads,
        num_k_heads,
        head_k_dim,
        head_v_dim,
        chunk_len,
        &conv_out_f32,
        &beta_f32,
        &g_f32,
        &init_state_f32,
    );

    assert_close(&actual_y, &expected_y, "y (wmma_lds)");
    assert_close(&actual_state, &expected_state, "state (wmma_lds)");
}

#[test]
fn wmma_lds_matches_f64_reference_ungrouped_small() {
    run_case_wmma_lds(2, 2, 8, 8, 5);
}

#[test]
fn wmma_lds_matches_f64_reference_grouped_heads() {
    run_case_wmma_lds(4, 2, 16, 16, 13);
}

#[test]
fn wmma_lds_matches_f64_reference_realistic_shape() {
    run_case_wmma_lds(8, 4, 32, 32, 37);
}

#[test]
fn wmma_lds_matches_f64_reference_degenerate_single_token() {
    run_case_wmma_lds(3, 3, 8, 8, 1);
}

#[test]
fn wmma_lds_matches_f64_reference_full_tile_lds_boundary() {
    run_case_wmma_lds(2, 1, 128, 128, 128);
}

#[test]
fn wmma_lds_matches_f64_reference_partial_last_chunk() {
    run_case_wmma_lds(2, 2, 16, 16, 127);
}

/// Runs `num_tiles` consecutive `tile_len`-token tiles through the LDS-staged
/// GPU pipeline, feeding `state` forward between calls exactly like
/// `gdn_chunkwise_step`'s own sub-chunking loop, and compares against a
/// single f64 `reference_sequential` call over the whole concatenated
/// sequence (a plain per-token recurrence — the chunkwise algebra is *exact*
/// against it, not an approximation, so any tile-count-dependent drift here
/// is real f16-rounding compounding, not a reference-side artifact).
#[allow(clippy::too_many_arguments)]
fn run_long_chain(
    num_heads: u32,
    num_k_heads: u32,
    head_k_dim: u32,
    head_v_dim: u32,
    tile_len: u32,
    num_tiles: u32,
) -> (Vec<f32>, Vec<f64>, Vec<f32>, Vec<f64>) {
    let _device = Device::new(0).expect("failed to select device 0");
    let k = ChunkwiseWmmaLdsKernels::load();

    let (h, hk, sk, sv, tl) = (
        num_heads as usize,
        num_k_heads as usize,
        head_k_dim as usize,
        head_v_dim as usize,
        tile_len as usize,
    );
    let full_len = tl * num_tiles as usize;
    let key_dim = hk * sk;
    let conv_dim = 2 * key_dim + h * sv;

    let conv_out_f32: Vec<f32> = (0..full_len * conv_dim)
        .map(|i| ((i % 23) as f32) * 0.07 - 0.75)
        .collect();
    let beta_f32: Vec<f32> = (0..full_len * h)
        .map(|i| 0.1 + (i as f32 % 7.0) * 0.09)
        .collect();
    let g_f32: Vec<f32> = (0..full_len * h)
        .map(|i| -0.02 - (i as f32 % 9.0) * 0.025)
        .collect();
    let init_state_f32: Vec<f32> = (0..h * sk * sv)
        .map(|i| ((i % 13) as f32) * 0.04 - 0.24)
        .collect();

    let to_f64 = |v: &[f32]| v.iter().map(|&x| x as f64).collect::<Vec<_>>();
    let (expected_y, expected_state) = reference_sequential(
        h,
        hk,
        sk,
        sv,
        full_len,
        &to_f64(&conv_out_f32),
        &to_f64(&beta_f32),
        &to_f64(&g_f32),
        &to_f64(&init_state_f32),
    );

    let mut state = init_state_f32;
    let mut actual_y = vec![0.0f32; full_len * h * sv];
    for tile in 0..num_tiles as usize {
        let t0 = tile * tl;
        let conv_slice = &conv_out_f32[t0 * conv_dim..(t0 + tl) * conv_dim];
        let beta_slice = &beta_f32[t0 * h..(t0 + tl) * h];
        let g_slice = &g_f32[t0 * h..(t0 + tl) * h];
        let (y_tile, new_state) = run_chunkwise_gpu_wmma_lds(
            &k,
            num_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            tile_len,
            conv_slice,
            beta_slice,
            g_slice,
            &state,
        );
        actual_y[t0 * h * sv..(t0 + tl) * h * sv].copy_from_slice(&y_tile);
        state = new_state;
    }
    (actual_y, expected_y, state, expected_state)
}

#[test]
fn wmma_lds_matches_f64_reference_long_chain_64_tiles() {
    let (actual_y, expected_y, actual_state, expected_state) =
        run_long_chain(4, 2, 32, 32, 128, 64);
    assert_close(&actual_y, &expected_y, "y (long chain)");
    assert_close(&actual_state, &expected_state, "state (long chain)");
}

/// Same long-chain check at Ornith's exact real shape (32 v-heads/16 k-heads,
/// head dims 128, `GDN_RECUR_TILE`(128) x 64 tiles = depth 8192) — `#[ignore]`d
/// for build/runtime cost, not correctness risk (the gate test above already
/// asserts `TOL`; this reports the measured figure the round writeup cites
/// as the compounding-across-64-tiles-at-depth-8192 arbiter).
#[test]
#[ignore]
fn measure_long_chain_max_rel_error_ornith_shape() {
    let (actual_y, expected_y, actual_state, expected_state) =
        run_long_chain(32, 16, 128, 128, 128, 64);
    let max_rel = |actual: &[f32], expected: &[f64]| -> f64 {
        actual
            .iter()
            .zip(expected)
            .map(|(&a, &e)| (a as f64 - e).abs() / e.abs().max(1.0))
            .fold(0.0, f64::max)
    };
    println!(
        "long-chain (64 tiles @ depth 8192) y max_rel_err={:.6e}  state max_rel_err={:.6e}  (TOL={:.1e})",
        max_rel(&actual_y, &expected_y),
        max_rel(&actual_state, &expected_state),
        support::TOL
    );
}

/// Diagnostic only (`#[ignore]`d, mirrors `gdn_chunkwise_wmma.rs`'s own
/// probe) — reports measured max relative error against the f64 reference at
/// Ornith's real shape, for the round writeup's precision table. Not a gate:
/// `wmma_lds_matches_f64_reference_full_tile_lds_boundary` above already
/// asserts this stays within `TOL` (3e-3) at this exact shape.
#[test]
#[ignore]
fn measure_max_rel_error_ornith_shape() {
    let _device = rocml_hip::Device::new(0).expect("failed to select device 0");
    let k = ChunkwiseWmmaLdsKernels::load();
    let (h, hk, sk, sv, t): (u32, u32, u32, u32, u32) = (32, 16, 128, 128, 128);
    let (hu, hku, sku, svu, tu) = (
        h as usize,
        hk as usize,
        sk as usize,
        sv as usize,
        t as usize,
    );
    let key_dim = hku * sku;
    let conv_dim = 2 * key_dim + hu * svu;

    let conv_out_f32: Vec<f32> = (0..tu * conv_dim)
        .map(|i| ((i % 23) as f32) * 0.07 - 0.75)
        .collect();
    let beta_f32: Vec<f32> = (0..tu * hu)
        .map(|i| 0.1 + (i as f32 % 7.0) * 0.09)
        .collect();
    let g_f32: Vec<f32> = (0..tu * hu)
        .map(|i| -0.02 - (i as f32 % 9.0) * 0.025)
        .collect();
    let init_state_f32: Vec<f32> = (0..hu * sku * svu)
        .map(|i| ((i % 13) as f32) * 0.04 - 0.24)
        .collect();
    let to_f64 = |v: &[f32]| v.iter().map(|&x| x as f64).collect::<Vec<_>>();
    let (expected_y, expected_state) = reference_sequential(
        hu,
        hku,
        sku,
        svu,
        tu,
        &to_f64(&conv_out_f32),
        &to_f64(&beta_f32),
        &to_f64(&g_f32),
        &to_f64(&init_state_f32),
    );
    let (actual_y, actual_state) = run_chunkwise_gpu_wmma_lds(
        &k,
        h,
        hk,
        sk,
        sv,
        t,
        &conv_out_f32,
        &beta_f32,
        &g_f32,
        &init_state_f32,
    );
    let max_rel = |actual: &[f32], expected: &[f64]| -> f64 {
        actual
            .iter()
            .zip(expected)
            .map(|(&a, &e)| (a as f64 - e).abs() / e.abs().max(1.0))
            .fold(0.0, f64::max)
    };
    println!(
        "y max_rel_err={:.6e}  state max_rel_err={:.6e}  (TOL={:.1e})",
        max_rel(&actual_y, &expected_y),
        max_rel(&actual_state, &expected_state),
        support::TOL
    );
}
