//! GPU integration tests for the batched `gemm_xwt_q*` kernels
//! (`kernels/gemm_xwt_quant.hip`): `out[rows,m] = X[rows,n] * dequant(W)^T`
//! for GGUF-quantized weights, compared against rocml-core's proven-correct
//! CPU dequant + a CPU matmul. Shapes are chosen to cross the kernel's
//! internal tile boundaries (`ROWS_PER_WARP`=8 output rows per warp,
//! `TILE_ROWS`=64 output rows per workgroup, `TILE_ELEMS`=256 reduction
//! elements per outer iteration) and to include the degenerate `rows=1`
//! case, which must match `gemv_q*`'s single-vector semantics.
mod common;

use common::{
    expected_gemm, random_row_q4_k, random_row_q5_k, random_row_q6_k, random_row_q8_0,
    run_gemm_quant_kernel, Rng, Q4_K_BLOCK_BYTES, Q5_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES,
    Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMS,
};
use rocml_core::quant::GgmlDType;

/// Looser than `gemv_q*`'s 2e-3: a batched-row's dot product sums the same
/// number of terms as the single-row case, but across more output rows the
/// worst-case element is more likely to land near a cancellation-heavy sum —
/// still a fixed per-element tolerance, just budgeted a bit wider.
const REL_TOL: f32 = 4e-3;

fn assert_close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (got, want)) in actual.iter().zip(expected).enumerate() {
        let diff = (got - want).abs();
        let tol = REL_TOL * want.abs().max(1.0);
        assert!(
            diff <= tol,
            "{label}[{i}]: got {got}, want {want} (diff {diff}, tol {tol})"
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    dtype: GgmlDType,
    hsaco: &[u8],
    kernel: &str,
    block_bytes: usize,
    block_elems: usize,
    rows: u32,
    m: u32,
    n: u32,
    seed: u32,
) {
    assert_eq!(
        n as usize % block_elems,
        0,
        "n must be a multiple of the block width"
    );
    let blocks_per_row = n as usize / block_elems;
    let mut rng = Rng::new(seed);
    let mut w_bytes = Vec::with_capacity(m as usize * blocks_per_row * block_bytes);
    for _ in 0..m {
        let row = match dtype {
            GgmlDType::Q8_0 => random_row_q8_0(&mut rng, blocks_per_row),
            GgmlDType::Q4_K => random_row_q4_k(&mut rng, blocks_per_row),
            GgmlDType::Q5_K => random_row_q5_k(&mut rng, blocks_per_row),
            GgmlDType::Q6_K => random_row_q6_k(&mut rng, blocks_per_row),
            other => panic!("unsupported dtype for this test: {other:?}"),
        };
        w_bytes.extend(row);
    }
    let x: Vec<f32> = (0..(rows * n))
        .map(|i| ((i % 13) as f32) * 0.2 - 1.0)
        .collect();

    let expected = expected_gemm(dtype, &w_bytes, &x, rows as usize, m as usize, n as usize);
    let actual = run_gemm_quant_kernel(hsaco, kernel, &w_bytes, &x, rows, m, n);
    assert_close(&actual, &expected, &format!("gemm_xwt_{dtype:?}"));
}

macro_rules! quant_gemm_tests {
    ($mod_name:ident, $dtype:expr, $hsaco:expr, $kernel:expr, $block_bytes:expr, $block_elems:expr) => {
        mod $mod_name {
            use super::*;

            #[test]
            fn degenerate_single_row_matches_gemv_semantics() {
                run_case(
                    $dtype,
                    $hsaco,
                    $kernel,
                    $block_bytes,
                    $block_elems,
                    1,
                    5,
                    $block_elems as u32,
                    1,
                );
            }

            #[test]
            fn rows_crosses_tile_row_boundary() {
                // ROWS_PER_WARP=8: 17 rows exercises two full warps' worth
                // of rows, a partial third warp, and the tail-row-
                // out-of-range path within one workgroup (TILE_ROWS=64
                // still covers all 17 rows in a single grid.y block).
                run_case(
                    $dtype,
                    $hsaco,
                    $kernel,
                    $block_bytes,
                    $block_elems,
                    17,
                    11,
                    2 * $block_elems as u32,
                    2,
                );
            }

            #[test]
            fn rows_crosses_workgroup_tile_boundary() {
                // TILE_ROWS=64: 130 rows forces grid.y=3, exercising later
                // workgroups' row-base offsets and the last one's
                // out-of-range tail.
                run_case(
                    $dtype,
                    $hsaco,
                    $kernel,
                    $block_bytes,
                    $block_elems,
                    130,
                    9,
                    2 * $block_elems as u32,
                    5,
                );
            }

            #[test]
            fn n_crosses_reduction_tile_boundary() {
                // TILE_ELEMS=256: for Q8_0 (32-elem native block) this spans
                // multiple outer-loop groups per tile; for the K-quants
                // (256-elem native block) this spans multiple super-blocks.
                run_case(
                    $dtype,
                    $hsaco,
                    $kernel,
                    $block_bytes,
                    $block_elems,
                    9,
                    6,
                    3 * $block_elems as u32,
                    3,
                );
            }

            #[test]
            fn large_shape() {
                run_case(
                    $dtype,
                    $hsaco,
                    $kernel,
                    $block_bytes,
                    $block_elems,
                    64,
                    96,
                    8 * $block_elems as u32,
                    4,
                );
            }
        }
    };
}

quant_gemm_tests!(
    q8_0,
    GgmlDType::Q8_0,
    rocml_kernels::GEMM_XWT_Q8_0_HSACO,
    rocml_kernels::GEMM_XWT_Q8_0_KERNEL,
    Q8_0_BLOCK_BYTES,
    Q8_0_BLOCK_ELEMS
);
quant_gemm_tests!(
    q4_k,
    GgmlDType::Q4_K,
    rocml_kernels::GEMM_XWT_Q4_K_HSACO,
    rocml_kernels::GEMM_XWT_Q4_K_KERNEL,
    Q4_K_BLOCK_BYTES,
    256
);
quant_gemm_tests!(
    q5_k,
    GgmlDType::Q5_K,
    rocml_kernels::GEMM_XWT_Q5_K_HSACO,
    rocml_kernels::GEMM_XWT_Q5_K_KERNEL,
    Q5_K_BLOCK_BYTES,
    256
);
quant_gemm_tests!(
    q6_k,
    GgmlDType::Q6_K,
    rocml_kernels::GEMM_XWT_Q6_K_HSACO,
    rocml_kernels::GEMM_XWT_Q6_K_KERNEL,
    Q6_K_BLOCK_BYTES,
    256
);
