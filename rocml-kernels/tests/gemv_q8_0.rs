//! GPU integration tests for `gemv_q8_0`: fused dequant-GEMV that reads raw
//! GGUF Q8_0 blocks directly from VRAM (dequantized in-register), compared
//! against rocml-core's proven-correct CPU dequant + a CPU dot product.
mod common;

use common::{
    assert_close, expected_gemv, random_row_q8_0, run_gemv_kernel, Rng, Q8_0_BLOCK_BYTES,
    Q8_0_BLOCK_ELEMS,
};
use rocml_core::gguf::GgufFile;
use rocml_core::quant::GgmlDType;
use rocml_core::testpaths::checkpoint;

fn run_synthetic(m: u32, n: u32, seed: u32) {
    assert_eq!(
        n as usize % Q8_0_BLOCK_ELEMS,
        0,
        "n must be a multiple of 32"
    );
    let blocks_per_row = n as usize / Q8_0_BLOCK_ELEMS;
    let mut rng = Rng::new(seed);
    let mut w_bytes = Vec::with_capacity(m as usize * blocks_per_row * Q8_0_BLOCK_BYTES);
    for _ in 0..m {
        w_bytes.extend(random_row_q8_0(&mut rng, blocks_per_row));
    }
    let x: Vec<f32> = (0..n).map(|i| ((i % 13) as f32) * 0.2 - 1.0).collect();

    let expected = expected_gemv(
        GgmlDType::Q8_0,
        &w_bytes,
        &x,
        m as usize,
        blocks_per_row * Q8_0_BLOCK_BYTES,
    );
    let actual = run_gemv_kernel(
        rocml_kernels::GEMV_Q8_0_HSACO,
        rocml_kernels::GEMV_Q8_0_KERNEL,
        &w_bytes,
        &x,
        m,
        n,
    );
    assert_close(&actual, &expected, "gemv_q8_0 y");
}

#[test]
fn gemv_q8_0_non_power_of_two_m() {
    run_synthetic(17, 768, 1);
}

#[test]
fn gemv_q8_0_degenerate_single_row() {
    run_synthetic(1, 32, 2);
}

#[test]
fn gemv_q8_0_large_shape() {
    run_synthetic(256, 4096, 3);
}

#[test]
fn gemv_q8_0_real_tensor() {
    let Some(path) = checkpoint("Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf") else {
        return;
    };
    let gguf = GgufFile::open(&path).expect("failed to open gguf");
    // A modest attention projection, not the ~500M-element embedding table.
    let view = gguf
        .tensor("blk.0.attn_gate.weight")
        .expect("tensor not found");
    assert_eq!(view.dtype(), GgmlDType::Q8_0);
    let &[n, m] = view.shape() else {
        panic!("expected a 2D tensor, got shape {:?}", view.shape());
    };
    let (m, n) = (m as u32, n as u32);
    let w_bytes = view.data();

    let x: Vec<f32> = (0..n).map(|i| ((i % 29) as f32) * 0.037 - 0.5).collect();
    let blocks_per_row = n as usize / Q8_0_BLOCK_ELEMS;
    let expected = expected_gemv(
        GgmlDType::Q8_0,
        w_bytes,
        &x,
        m as usize,
        blocks_per_row * Q8_0_BLOCK_BYTES,
    );
    let actual = run_gemv_kernel(
        rocml_kernels::GEMV_Q8_0_HSACO,
        rocml_kernels::GEMV_Q8_0_KERNEL,
        w_bytes,
        &x,
        m,
        n,
    );
    assert_close(&actual, &expected, "gemv_q8_0 real-tensor y");
}
