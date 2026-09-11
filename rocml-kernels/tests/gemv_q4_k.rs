//! GPU integration tests for `gemv_q4_k`: fused dequant-GEMV that reads raw
//! GGUF Q4_K super-blocks directly from VRAM (dequantized in-register),
//! compared against rocml-core's proven-correct CPU dequant + a CPU dot
//! product. No real-tensor test here — the synthetic coverage below already
//! exercises the packed-scale/min unpack; `gemv_q6_k`/`gemv_q8_0` cover the
//! real-GGUF path.
mod common;

use common::{
    assert_close, expected_gemv, random_row_q4_k, run_gemv_kernel, Rng, K_BLOCK_ELEMS,
    Q4_K_BLOCK_BYTES,
};
use rocml_core::quant::GgmlDType;

fn run(m: u32, n: u32, seed: u32) {
    assert_eq!(n as usize % K_BLOCK_ELEMS, 0, "n must be a multiple of 256");
    let blocks_per_row = n as usize / K_BLOCK_ELEMS;
    let mut rng = Rng::new(seed);
    let mut w_bytes = Vec::with_capacity(m as usize * blocks_per_row * Q4_K_BLOCK_BYTES);
    for _ in 0..m {
        w_bytes.extend(random_row_q4_k(&mut rng, blocks_per_row));
    }
    let x: Vec<f32> = (0..n).map(|i| ((i % 13) as f32) * 0.2 - 1.0).collect();

    let expected = expected_gemv(
        GgmlDType::Q4_K,
        &w_bytes,
        &x,
        m as usize,
        blocks_per_row * Q4_K_BLOCK_BYTES,
    );
    let actual = run_gemv_kernel(
        rocml_kernels::GEMV_Q4_K_HSACO,
        rocml_kernels::GEMV_Q4_K_KERNEL,
        &w_bytes,
        &x,
        m,
        n,
    );
    assert_close(&actual, &expected, "gemv_q4_k y");
}

#[test]
fn gemv_q4_k_non_power_of_two_m() {
    run(17, 768, 11);
}

#[test]
fn gemv_q4_k_degenerate_single_row() {
    run(1, 256, 12);
}

#[test]
fn gemv_q4_k_large_shape() {
    run(256, 4096, 13);
}
