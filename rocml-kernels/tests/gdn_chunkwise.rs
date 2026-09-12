//! GPU integration tests for the chunkwise (blocked delta-rule) Gated Delta
//! Net recurrence pipeline (`kernels/gdn_chunkwise.hip`, issue #6's
//! chunkwise rewrite) against an f64 CPU reference (`gdn_chunkwise_support`)
//! — see that module's doc comment for why the ground truth is a plain
//! sequential recurrence rather than a literal re-derivation of the
//! chunking algebra: the chunkwise pipeline is an *exact* algebraic
//! reformulation of that recurrence, so any real bug in the new kernels'
//! algebra shows up as a mismatch here. `gdn_chunkwise_decode_check.rs`
//! adds an independent second ground truth (the actual GPU decode kernel
//! run sequentially).
#[path = "gdn_chunkwise_support/mod.rs"]
mod support;

use rocml_hip::Device;
use support::{assert_close, reference_sequential, run_chunkwise_gpu, ChunkwiseKernels};

#[allow(clippy::too_many_arguments)]
fn run_case(num_heads: u32, num_k_heads: u32, head_k_dim: u32, head_v_dim: u32, chunk_len: u32) {
    let _device = Device::new(0).expect("failed to select device 0");
    let k = ChunkwiseKernels::load();

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

    // Deterministic pseudo-random-ish fixtures (no external RNG dependency),
    // scaled to stay well away from 0/inf under L2-norm and cumulative exp.
    let conv_out_f32: Vec<f32> = (0..t * conv_dim)
        .map(|i| ((i % 23) as f32) * 0.07 - 0.75)
        .collect();
    let beta_f32: Vec<f32> = (0..t * h).map(|i| 0.1 + (i as f32 % 7.0) * 0.09).collect();
    // Negative log-decays only (`decay = exp(g) <= 1`, matching the "g <= 0"
    // contract `torch_chunk_gated_delta_rule`'s doc comment states).
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

    let (actual_y, actual_state) = run_chunkwise_gpu(
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

    assert_close(&actual_y, &expected_y, "y");
    assert_close(&actual_state, &expected_state, "state");
}

#[test]
fn chunkwise_matches_f64_reference_ungrouped_small() {
    run_case(2, 2, 8, 8, 5);
}

#[test]
fn chunkwise_matches_f64_reference_grouped_heads() {
    run_case(4, 2, 16, 16, 13);
}

#[test]
fn chunkwise_matches_f64_reference_realistic_shape() {
    run_case(8, 4, 32, 32, 37);
}

#[test]
fn chunkwise_matches_f64_reference_degenerate_single_token() {
    run_case(3, 3, 8, 8, 1);
}

#[test]
fn chunkwise_matches_f64_reference_full_tile_lds_boundary() {
    // tile_len == head_k_dim == head_v_dim == 128: the triangular-inverse
    // kernel's dynamic shared memory request is exactly gfx1101's 64KB
    // budget (128*128*4 == 65536) — the tightest case this pipeline runs.
    run_case(2, 1, 128, 128, 128);
}

#[test]
fn chunkwise_matches_f64_reference_partial_last_chunk() {
    // A prompt-length-127-style partial chunk one short of the 128 tile.
    run_case(2, 2, 16, 16, 127);
}
