//! GPU integration tests for the micro-tile WMMA `gemm_xwt_wmma_q*_micro`
//! kernels (`kernels/gemm_xwt_quant_wmma_micro.hip`, qwen35moe M4 lever 1):
//! same `out[rows,m] = X[rows,n] * dequant(W)^T` contract as
//! `gemm_xwt_quant_wmma.rs`'s default/narrow kernels, compared against the
//! same CPU reference, but at this config's own tile sizes (`TILE_ROWS`=16,
//! `TILE_M`=64, `WARPS_PER_BLOCK`=4) and — the whole point of this kernel —
//! at the small `rows` values (1-20) MoE's grouped-by-expert batched GEMM
//! actually produces, which the default/narrow suite never exercises (its
//! smallest case is `degenerate_single_row` at `rows=1`, but every other
//! case there uses `rows >= 17`; this file's cases stay in the 1-16 range
//! that motivated adding this kernel at all).
mod common;

use common::{
    expected_gemm_wmma, random_row_q4_k, random_row_q5_k, random_row_q6_k, random_row_q8_0,
    run_gemm_wmma_kernel_cfg, Rng, Q4_K_BLOCK_BYTES, Q5_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES,
    Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMS,
};
use rocml_core::quant::GgmlDType;

/// Same tolerance as `gemm_xwt_quant_wmma.rs` — this is the identical
/// algorithm (`gemm_xwt_wmma_impl.h`), just instantiated at a smaller tile,
/// so the same f16-rounding-plus-reduction-order error profile applies.
const REL_TOL: f32 = 1.5e-2;

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
    assert_eq!(m % 16, 0, "WMMA requires m to be a multiple of 16");
    assert_eq!(n % 16, 0, "WMMA requires n to be a multiple of 16");
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

    let expected = expected_gemm_wmma(dtype, &w_bytes, &x, rows as usize, m as usize, n as usize);
    let actual = run_gemm_wmma_kernel_cfg(
        hsaco, kernel, &w_bytes, &x, rows, m, n, /* tile_rows */ 16, /* tile_m */ 64,
        /* warps_per_block */ 4,
    );
    assert_close(
        &actual,
        &expected,
        &format!("gemm_xwt_wmma_micro_{dtype:?}"),
    );
}

macro_rules! quant_gemm_wmma_micro_tests {
    ($mod_name:ident, $dtype:expr, $hsaco:expr, $kernel:expr, $block_bytes:expr, $block_elems:expr) => {
        mod $mod_name {
            use super::*;

            #[test]
            fn degenerate_single_row() {
                // The extreme end of what MoE's grouped GEMM can hand this
                // kernel: an expert with exactly one assigned row.
                run_case(
                    $dtype,
                    $hsaco,
                    $kernel,
                    $block_bytes,
                    $block_elems,
                    1,
                    64,
                    $block_elems as u32,
                    1,
                );
            }

            #[test]
            fn typical_moe_expert_group() {
                // The measured average expert-group size at CHUNK_CAP=512
                // (`.claude/CLAUDE.md`'s M3 section: ~2-16 rows/expert).
                run_case(
                    $dtype,
                    $hsaco,
                    $kernel,
                    $block_bytes,
                    $block_elems,
                    8,
                    64,
                    2 * $block_elems as u32,
                    2,
                );
            }

            #[test]
            fn rows_not_multiple_of_16() {
                // TILE_ROWS=16: 9 rows exercises the clamp-and-gate tail
                // inside one workgroup's row dimension.
                run_case(
                    $dtype,
                    $hsaco,
                    $kernel,
                    $block_bytes,
                    $block_elems,
                    9,
                    64,
                    2 * $block_elems as u32,
                    3,
                );
            }

            #[test]
            fn rows_crosses_tile_row_boundary() {
                // TILE_ROWS=16: 20 rows forces grid.y=2, exercising the
                // second workgroup's row-base offset and its out-of-range
                // tail (20 isn't a multiple of 16).
                run_case(
                    $dtype,
                    $hsaco,
                    $kernel,
                    $block_bytes,
                    $block_elems,
                    20,
                    64,
                    2 * $block_elems as u32,
                    4,
                );
            }

            #[test]
            fn cols_crosses_tile_col_boundary() {
                // TILE_M=64: m=80 forces grid.x=2 with a partial second
                // column tile.
                run_case(
                    $dtype,
                    $hsaco,
                    $kernel,
                    $block_bytes,
                    $block_elems,
                    10,
                    80,
                    2 * $block_elems as u32,
                    5,
                );
            }

            #[test]
            fn n_spans_multiple_k_stages() {
                run_case(
                    $dtype,
                    $hsaco,
                    $kernel,
                    $block_bytes,
                    $block_elems,
                    5,
                    64,
                    3 * $block_elems as u32,
                    6,
                );
            }

            #[test]
            fn larger_group_multiple_row_and_col_tiles() {
                // A busier-than-average expert (64 rows = 4 row tiles) at a
                // wider `m` (192 = 3 col tiles) — still far below the
                // default/narrow kernels' 128-row dispatch floor.
                run_case(
                    $dtype,
                    $hsaco,
                    $kernel,
                    $block_bytes,
                    $block_elems,
                    64,
                    192,
                    8 * $block_elems as u32,
                    7,
                );
            }
        }
    };
}

quant_gemm_wmma_micro_tests!(
    q8_0_micro,
    GgmlDType::Q8_0,
    rocml_kernels::GEMM_XWT_WMMA_Q8_0_MICRO_HSACO,
    rocml_kernels::GEMM_XWT_WMMA_Q8_0_MICRO_KERNEL,
    Q8_0_BLOCK_BYTES,
    Q8_0_BLOCK_ELEMS
);
quant_gemm_wmma_micro_tests!(
    q4_k_micro,
    GgmlDType::Q4_K,
    rocml_kernels::GEMM_XWT_WMMA_Q4_K_MICRO_HSACO,
    rocml_kernels::GEMM_XWT_WMMA_Q4_K_MICRO_KERNEL,
    Q4_K_BLOCK_BYTES,
    256
);
quant_gemm_wmma_micro_tests!(
    q5_k_micro,
    GgmlDType::Q5_K,
    rocml_kernels::GEMM_XWT_WMMA_Q5_K_MICRO_HSACO,
    rocml_kernels::GEMM_XWT_WMMA_Q5_K_MICRO_KERNEL,
    Q5_K_BLOCK_BYTES,
    256
);
quant_gemm_wmma_micro_tests!(
    q6_k_micro,
    GgmlDType::Q6_K,
    rocml_kernels::GEMM_XWT_WMMA_Q6_K_MICRO_HSACO,
    rocml_kernels::GEMM_XWT_WMMA_Q6_K_MICRO_KERNEL,
    Q6_K_BLOCK_BYTES,
    256
);
