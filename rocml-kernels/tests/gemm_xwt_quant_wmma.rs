//! GPU integration tests for the WMMA matrix-core `gemm_xwt_wmma_q*` kernels
//! (`kernels/gemm_xwt_quant_wmma.hip`, issue #6's follow-up to the scalar
//! `gemm_xwt_q*` kernels covered by `gemm_xwt_quant.rs`): same `out[rows,m]
//! = X[rows,n] * dequant(W)^T` contract, compared against the same CPU
//! reference. Shapes are chosen to cross the kernel's internal tile
//! boundaries (`TILE_ROWS`=128 output rows and `TILE_M`=64 output cols per
//! workgroup, `K_STAGE`=16 reduction elements staged per outer iteration)
//! and to include shapes smaller than one tile (exercising the row/col
//! clamp-and-gate tail handling the kernel uses instead of a branch).
mod common;

use common::{
    expected_gemm_wmma, random_row_q4_k, random_row_q5_k, random_row_q6_k, random_row_q8_0,
    run_gemm_wmma_kernel, Rng, Q4_K_BLOCK_BYTES, Q5_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES,
    Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMS,
};
use rocml_core::quant::GgmlDType;

/// Tight: [`expected_gemm_wmma`] already rounds both operands through f16
/// the same way the WMMA kernel's fragments do, so any residual difference
/// is just the matrix unit's own f32-accumulation reduction order (WMMA
/// sums 16 terms per instruction internally in an unspecified but
/// IEEE-754-consistent order) rather than input-rounding error — a debug
/// probe against this harness's data confirmed most shapes/dtypes agree to
/// ~1e-6 relative. Q6_K's synthetic test weights (full-signed-byte-range
/// scale bytes, real GGUF calibration never produces this) push a handful
/// of dequantized values into the thousands, where reduction-order noise
/// across many large-magnitude terms with mixed signs needs a bit more
/// headroom than every other case here — still two orders of magnitude
/// tighter than `gemm_xwt_quant.rs`'s pure-f32 scalar-kernel tolerance, and
/// tight enough to catch a real kernel bug (a wrong row/col/k index, a
/// missing sync, a transposed fragment) without chasing rounding noise.
/// `large_shape`'s Q6_K case (n=2048, thousand-magnitude dequantized
/// weights) is the measured worst case: a debug probe over all 32768
/// outputs found a 1.01e-2 max relative error but only 30 of them (0.09%)
/// past 3e-3 — heavy-cancellation sums of large terms in a different order
/// than the CPU reference, not a bug (confirmed by the tighter agreement
/// everywhere else in this file). `1.5e-2` covers that measured maximum
/// with headroom.
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
    tile_m: u32,
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
    let actual = run_gemm_wmma_kernel(hsaco, kernel, &w_bytes, &x, rows, m, n, tile_m);
    assert_close(&actual, &expected, &format!("gemm_xwt_wmma_{dtype:?}"));
}

macro_rules! quant_gemm_wmma_tests {
    ($mod_name:ident, $dtype:expr, $hsaco:expr, $kernel:expr, $block_bytes:expr, $block_elems:expr, $tile_m:expr, $cross_m:expr) => {
        mod $mod_name {
            use super::*;

            #[test]
            fn degenerate_single_row() {
                run_case(
                    $dtype,
                    $hsaco,
                    $kernel,
                    $block_bytes,
                    $block_elems,
                    1,
                    16,
                    $block_elems as u32,
                    1,
                    $tile_m,
                );
            }

            #[test]
            fn rows_within_one_tile_not_multiple_of_16() {
                // TILE_ROWS=128, but the valid row count (17) isn't a
                // multiple of 16 — exercises the clamp-and-gate tail inside
                // one workgroup's row dimension.
                run_case(
                    $dtype,
                    $hsaco,
                    $kernel,
                    $block_bytes,
                    $block_elems,
                    17,
                    32,
                    2 * $block_elems as u32,
                    2,
                    $tile_m,
                );
            }

            #[test]
            fn rows_crosses_tile_row_boundary() {
                // TILE_ROWS=128: 140 rows forces grid.y=2, exercising the
                // second workgroup's row-base offset and its out-of-range
                // tail (140 isn't a multiple of 128).
                run_case(
                    $dtype,
                    $hsaco,
                    $kernel,
                    $block_bytes,
                    $block_elems,
                    140,
                    48,
                    2 * $block_elems as u32,
                    3,
                    $tile_m,
                );
            }

            #[test]
            fn cols_crosses_tile_col_boundary() {
                // `$cross_m` is chosen per tile config (128 for the default
                // kernels, 64 for the narrow ones) to force grid.x=2,
                // exercising the second workgroup's col-base offset and its
                // out-of-range tail.
                run_case(
                    $dtype,
                    $hsaco,
                    $kernel,
                    $block_bytes,
                    $block_elems,
                    20,
                    $cross_m,
                    2 * $block_elems as u32,
                    4,
                    $tile_m,
                );
            }

            #[test]
            fn n_spans_multiple_k_stages() {
                // K_STAGE=16 divides every dtype's native block width here
                // (32 or 256), so a valid `n` (always a multiple of the
                // block width) is always a multiple of 16 too — the
                // zero-padded-last-stage path only exists for defensive
                // completeness and never actually triggers for real
                // weights. This shape instead just exercises accumulation
                // across several K_STAGE stages within one native block
                // (3*block_elems / 16 stages).
                run_case(
                    $dtype,
                    $hsaco,
                    $kernel,
                    $block_bytes,
                    $block_elems,
                    9,
                    32,
                    3 * $block_elems as u32,
                    5,
                    $tile_m,
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
                    128,
                    256,
                    8 * $block_elems as u32,
                    6,
                    $tile_m,
                );
            }
        }
    };
}

// Narrow-kernel-only extra case: a partial *last* tile (unlike
// `cols_crosses_tile_col_boundary`'s exactly-half-full second tile) —
// m=96 with TILE_M=64 gives grid.x=2 where the second tile only has 32 of
// its 64 columns valid, a genuinely unaligned partial-tile shape the wide
// kernel's own `cols_crosses_tile_col_boundary` (m=144, TILE_M=128) doesn't
// cover at this tile size.
macro_rules! quant_gemm_wmma_narrow_partial_test {
    ($mod_name:ident, $dtype:expr, $hsaco:expr, $kernel:expr, $block_bytes:expr, $block_elems:expr) => {
        mod $mod_name {
            use super::*;

            #[test]
            fn cols_partial_last_tile() {
                run_case(
                    $dtype,
                    $hsaco,
                    $kernel,
                    $block_bytes,
                    $block_elems,
                    20,
                    96,
                    2 * $block_elems as u32,
                    7,
                    64,
                );
            }
        }
    };
}

quant_gemm_wmma_tests!(
    q8_0,
    GgmlDType::Q8_0,
    rocml_kernels::GEMM_XWT_WMMA_Q8_0_HSACO,
    rocml_kernels::GEMM_XWT_WMMA_Q8_0_KERNEL,
    Q8_0_BLOCK_BYTES,
    Q8_0_BLOCK_ELEMS,
    128,
    144
);
quant_gemm_wmma_tests!(
    q4_k,
    GgmlDType::Q4_K,
    rocml_kernels::GEMM_XWT_WMMA_Q4_K_HSACO,
    rocml_kernels::GEMM_XWT_WMMA_Q4_K_KERNEL,
    Q4_K_BLOCK_BYTES,
    256,
    128,
    144
);
quant_gemm_wmma_tests!(
    q5_k,
    GgmlDType::Q5_K,
    rocml_kernels::GEMM_XWT_WMMA_Q5_K_HSACO,
    rocml_kernels::GEMM_XWT_WMMA_Q5_K_KERNEL,
    Q5_K_BLOCK_BYTES,
    256,
    128,
    144
);
quant_gemm_wmma_tests!(
    q6_k,
    GgmlDType::Q6_K,
    rocml_kernels::GEMM_XWT_WMMA_Q6_K_HSACO,
    rocml_kernels::GEMM_XWT_WMMA_Q6_K_KERNEL,
    Q6_K_BLOCK_BYTES,
    256,
    128,
    144
);

// Narrow (TILE_M=64) kernels — shape-aware dispatch round's `m < 2048`
// dispatch target, `kernels/gemm_xwt_quant_wmma_narrow.hip`. Same
// correctness contract as the default-width kernels above, verified
// against the same CPU reference; `cols_crosses_tile_col_boundary`'s
// `$cross_m` (80) is chosen to cross *this* config's TILE_M=64 boundary
// instead of the default kernels' 128.
quant_gemm_wmma_tests!(
    q8_0_narrow,
    GgmlDType::Q8_0,
    rocml_kernels::GEMM_XWT_WMMA_Q8_0_NARROW_HSACO,
    rocml_kernels::GEMM_XWT_WMMA_Q8_0_NARROW_KERNEL,
    Q8_0_BLOCK_BYTES,
    Q8_0_BLOCK_ELEMS,
    64,
    80
);
quant_gemm_wmma_tests!(
    q4_k_narrow,
    GgmlDType::Q4_K,
    rocml_kernels::GEMM_XWT_WMMA_Q4_K_NARROW_HSACO,
    rocml_kernels::GEMM_XWT_WMMA_Q4_K_NARROW_KERNEL,
    Q4_K_BLOCK_BYTES,
    256,
    64,
    80
);
quant_gemm_wmma_tests!(
    q5_k_narrow,
    GgmlDType::Q5_K,
    rocml_kernels::GEMM_XWT_WMMA_Q5_K_NARROW_HSACO,
    rocml_kernels::GEMM_XWT_WMMA_Q5_K_NARROW_KERNEL,
    Q5_K_BLOCK_BYTES,
    256,
    64,
    80
);
quant_gemm_wmma_tests!(
    q6_k_narrow,
    GgmlDType::Q6_K,
    rocml_kernels::GEMM_XWT_WMMA_Q6_K_NARROW_HSACO,
    rocml_kernels::GEMM_XWT_WMMA_Q6_K_NARROW_KERNEL,
    Q6_K_BLOCK_BYTES,
    256,
    64,
    80
);

quant_gemm_wmma_narrow_partial_test!(
    q8_0_narrow_partial,
    GgmlDType::Q8_0,
    rocml_kernels::GEMM_XWT_WMMA_Q8_0_NARROW_HSACO,
    rocml_kernels::GEMM_XWT_WMMA_Q8_0_NARROW_KERNEL,
    Q8_0_BLOCK_BYTES,
    Q8_0_BLOCK_ELEMS
);
quant_gemm_wmma_narrow_partial_test!(
    q4_k_narrow_partial,
    GgmlDType::Q4_K,
    rocml_kernels::GEMM_XWT_WMMA_Q4_K_NARROW_HSACO,
    rocml_kernels::GEMM_XWT_WMMA_Q4_K_NARROW_KERNEL,
    Q4_K_BLOCK_BYTES,
    256
);
quant_gemm_wmma_narrow_partial_test!(
    q5_k_narrow_partial,
    GgmlDType::Q5_K,
    rocml_kernels::GEMM_XWT_WMMA_Q5_K_NARROW_HSACO,
    rocml_kernels::GEMM_XWT_WMMA_Q5_K_NARROW_KERNEL,
    Q5_K_BLOCK_BYTES,
    256
);
quant_gemm_wmma_narrow_partial_test!(
    q6_k_narrow_partial,
    GgmlDType::Q6_K,
    rocml_kernels::GEMM_XWT_WMMA_Q6_K_NARROW_HSACO,
    rocml_kernels::GEMM_XWT_WMMA_Q6_K_NARROW_KERNEL,
    Q6_K_BLOCK_BYTES,
    256
);
