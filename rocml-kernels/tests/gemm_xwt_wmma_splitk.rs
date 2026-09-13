//! GPU integration tests for the split-K WMMA GEMM
//! (`kernels/gemm_xwt_wmma_splitk.hip` + `kernels/gemm_splitk_reduce.hip`,
//! issue #6's split-K follow-up): same `out[rows,m] = X[rows,n] *
//! dequant(W)^T` contract as the plain default-tile WMMA kernel
//! (`gemm_xwt_quant_wmma.hip`, covered by `gemm_xwt_quant_wmma.rs`),
//! compared against the same CPU reference. Exercises: an exact 4-way and
//! 2-way split, partial/unaligned row and column tiles under split-K,
//! `num_splits=1` (degenerate "no split" case — must equal the plain
//! kernel), and bit-identical determinism (same input launched twice).
mod common;

use std::ffi::c_void;

use common::{
    expected_gemm_wmma, random_row_q4_k, random_row_q6_k, Rng, Q4_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES,
};
use rocml_core::quant::GgmlDType;
use rocml_hip::{kernel_params, Device, DeviceBuffer, LaunchConfig, Module};

const TILE_ROWS: u32 = 128;
const TILE_M: u32 = 128;
const K_STAGE: u32 = 16;
const LDS_PAD: u32 = 8;
const WARPS_PER_BLOCK: u32 = 16;
const REDUCE_BLOCK: u32 = 256;

/// Uploads `w_bytes`/`x`, launches the split-K GEMM (`num_splits` grid.z
/// blocks writing `[num_splits, rows, m]` partials) followed by the reduce
/// pass, and downloads the final `[rows, m]` output — mirrors
/// `common::run_gemm_wmma_kernel` but for the two-kernel split-K pipeline.
#[allow(clippy::too_many_arguments)]
fn run_gemm_wmma_splitk(
    hsaco: &[u8],
    kernel_name: &str,
    w_bytes: &[u8],
    x: &[f32],
    rows: u32,
    m: u32,
    n: u32,
    num_splits: u32,
) -> Vec<f32> {
    let _device = Device::new(0).expect("failed to select device 0");
    let module = Module::load_from_bytes(hsaco).expect("module load failed");
    let function = module
        .get_function(kernel_name)
        .expect("kernel lookup failed");
    let reduce_hsaco = rocml_kernels::GEMM_SPLITK_REDUCE_F32_HSACO;
    let reduce_module = Module::load_from_bytes(reduce_hsaco).expect("reduce module load failed");
    let reduce_fn = reduce_module
        .get_function(rocml_kernels::GEMM_SPLITK_REDUCE_F32_KERNEL)
        .expect("reduce kernel lookup failed");

    let mut buf_x = DeviceBuffer::<f32>::new(x.len()).expect("hipMalloc x failed");
    let mut buf_w = DeviceBuffer::<u8>::new(w_bytes.len()).expect("hipMalloc w failed");
    let buf_partial = DeviceBuffer::<f32>::new((num_splits * rows * m) as usize)
        .expect("hipMalloc partial failed");
    let buf_out = DeviceBuffer::<f32>::new((rows * m) as usize).expect("hipMalloc out failed");
    buf_x.copy_from_host(x).expect("copy x failed");
    buf_w.copy_from_host(w_bytes).expect("copy w failed");

    let x_ptr: *mut c_void = buf_x.device_ptr();
    let w_ptr: *mut c_void = buf_w.device_ptr();
    let partial_ptr: *mut c_void = buf_partial.device_ptr();
    let out_ptr: *mut c_void = buf_out.device_ptr();

    let mut params = kernel_params!(x_ptr, w_ptr, partial_ptr, rows, m, n, num_splits);
    let shared_mem_bytes =
        2 * (TILE_ROWS + TILE_M) * (K_STAGE + LDS_PAD) * std::mem::size_of::<u16>() as u32;
    let cfg = LaunchConfig {
        grid: (m.div_ceil(TILE_M), rows.div_ceil(TILE_ROWS), num_splits),
        block: (32, WARPS_PER_BLOCK, 1),
        shared_mem_bytes,
    };
    // SAFETY: params matches every gemm_xwt_wmma_splitk_<type> kernel's
    // parameter list (const float*, const void*, float*, unsigned x3,
    // unsigned) in order; block/grid/shared_mem_bytes match the default
    // tile config (TILE_M=128/WARPS_M=4/WARPS_N=4) this file's constants
    // mirror, and every device buffer outlives the launch.
    unsafe { function.launch(&cfg, &mut params, None) }.expect("splitk kernel launch failed");

    let total = rows * m;
    let reduce_cfg = LaunchConfig {
        grid: (total.div_ceil(REDUCE_BLOCK), 1, 1),
        block: (REDUCE_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut reduce_params = kernel_params!(partial_ptr, out_ptr, rows, m, num_splits);
    // SAFETY: params matches gemm_splitk_reduce_f32's signature (const
    // float*, float*, unsigned x3); grid/block cover every output element.
    unsafe { reduce_fn.launch(&reduce_cfg, &mut reduce_params, None) }
        .expect("reduce kernel launch failed");

    let mut actual = vec![0.0f32; (rows * m) as usize];
    buf_out.copy_to_host(&mut actual).expect("copy out failed");
    actual
}

/// Same tolerance rationale as `gemm_xwt_quant_wmma.rs` (`REL_TOL` there):
/// [`expected_gemm_wmma`] already f16-rounds both operands, so the residual
/// gap is pure f32-accumulation reduction-order noise — split-K reorders the
/// K reduction across `num_splits` independent partial sums plus one more
/// reduce pass on top, which is *more* reduction-order freedom than the
/// plain kernel's single accumulation chain, but still well inside this
/// tolerance (verified empirically below; kept identical to the plain WMMA
/// suite's tolerance rather than loosened).
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
fn build_case(
    dtype: GgmlDType,
    block_bytes: usize,
    block_elems: usize,
    rows: u32,
    m: u32,
    n: u32,
    seed: u32,
) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    assert_eq!(n as usize % block_elems, 0, "n must divide the block width");
    assert_eq!(m % 16, 0, "WMMA requires m to be a multiple of 16");
    assert_eq!(n % 16, 0, "WMMA requires n to be a multiple of 16");
    let blocks_per_row = n as usize / block_elems;
    let mut rng = Rng::new(seed);
    let mut w_bytes = Vec::with_capacity(m as usize * blocks_per_row * block_bytes);
    for _ in 0..m {
        let row = match dtype {
            GgmlDType::Q4_K => random_row_q4_k(&mut rng, blocks_per_row),
            GgmlDType::Q6_K => random_row_q6_k(&mut rng, blocks_per_row),
            other => panic!("unsupported dtype for this test: {other:?}"),
        };
        w_bytes.extend(row);
    }
    let x: Vec<f32> = (0..(rows * n))
        .map(|i| ((i % 13) as f32) * 0.2 - 1.0)
        .collect();
    let expected = expected_gemm_wmma(dtype, &w_bytes, &x, rows as usize, m as usize, n as usize);
    (w_bytes, x, expected)
}

/// Exact 4-way split at ffn-down's real proportions (scaled down): `n`
/// divisible by `4*K_STAGE`.
#[test]
fn q4_k_four_way_split() {
    let (w, x, expected) = build_case(GgmlDType::Q4_K, Q4_K_BLOCK_BYTES, 256, 128, 128, 512, 1);
    let actual = run_gemm_wmma_splitk(
        rocml_kernels::GEMM_XWT_WMMA_SPLITK_Q4_K_HSACO,
        rocml_kernels::GEMM_XWT_WMMA_SPLITK_Q4_K_KERNEL,
        &w,
        &x,
        128,
        128,
        512,
        4,
    );
    assert_close(&actual, &expected, "q4_k_four_way_split");
}

#[test]
fn q6_k_four_way_split() {
    let (w, x, expected) = build_case(GgmlDType::Q6_K, Q6_K_BLOCK_BYTES, 256, 128, 128, 512, 2);
    let actual = run_gemm_wmma_splitk(
        rocml_kernels::GEMM_XWT_WMMA_SPLITK_Q6_K_HSACO,
        rocml_kernels::GEMM_XWT_WMMA_SPLITK_Q6_K_KERNEL,
        &w,
        &x,
        128,
        128,
        512,
        4,
    );
    assert_close(&actual, &expected, "q6_k_four_way_split");
}

/// 2-way split — the dispatch layer's fallback when `n` doesn't divide
/// `4*K_STAGE` but does divide `2*K_STAGE`.
#[test]
fn q4_k_two_way_split() {
    let (w, x, expected) = build_case(GgmlDType::Q4_K, Q4_K_BLOCK_BYTES, 256, 128, 128, 512, 3);
    let actual = run_gemm_wmma_splitk(
        rocml_kernels::GEMM_XWT_WMMA_SPLITK_Q4_K_HSACO,
        rocml_kernels::GEMM_XWT_WMMA_SPLITK_Q4_K_KERNEL,
        &w,
        &x,
        128,
        128,
        512,
        2,
    );
    assert_close(&actual, &expected, "q4_k_two_way_split");
}

/// `num_splits=1` is the degenerate "don't split" case (the dispatch layer
/// never actually calls the split-K kernel this way — `splitk_num_splits`
/// returns `1` to mean "fall through to the plain kernel" — but the kernel
/// itself must still behave correctly if it is, since nothing in the kernel
/// mechanics *requires* `num_splits > 1`). Confirms the whole grid.z=1
/// slice covers `k in [0, n)` exactly like the plain kernel.
#[test]
fn degenerate_one_split() {
    let (w, x, expected) = build_case(GgmlDType::Q4_K, Q4_K_BLOCK_BYTES, 256, 20, 32, 512, 4);
    let actual = run_gemm_wmma_splitk(
        rocml_kernels::GEMM_XWT_WMMA_SPLITK_Q4_K_HSACO,
        rocml_kernels::GEMM_XWT_WMMA_SPLITK_Q4_K_KERNEL,
        &w,
        &x,
        20,
        32,
        512,
        1,
    );
    assert_close(&actual, &expected, "degenerate_one_split");
}

/// Partial/unaligned tile shapes under split-K: `rows` not a multiple of
/// `TILE_ROWS` (140 crosses the row-tile boundary) and `m` not a multiple of
/// `TILE_M` (144 crosses the col-tile boundary) — the same clamp-and-gate
/// tail handling the plain kernel's own boundary tests exercise
/// (`gemm_xwt_quant_wmma.rs`'s `rows_crosses_tile_row_boundary`/
/// `cols_crosses_tile_col_boundary`), now combined with a real 4-way split.
#[test]
fn partial_tile_with_split() {
    let (w, x, expected) = build_case(GgmlDType::Q4_K, Q4_K_BLOCK_BYTES, 256, 140, 144, 1024, 5);
    let actual = run_gemm_wmma_splitk(
        rocml_kernels::GEMM_XWT_WMMA_SPLITK_Q4_K_HSACO,
        rocml_kernels::GEMM_XWT_WMMA_SPLITK_Q4_K_KERNEL,
        &w,
        &x,
        140,
        144,
        1024,
        4,
    );
    assert_close(&actual, &expected, "partial_tile_with_split");
}

/// Smallest `m` split-K's dispatch can ever reach in production
/// (`GEMM_WMMA_NARROW_TILE_M`, the overall WMMA-eligibility floor — the
/// split-K round dropped the `m >= GEMM_WMMA_NARROW_THRESHOLD_M` restriction
/// after measurement showed split-K also beats the narrow-tile kernel at
/// `m=1024`, see `kernels_quant_dispatch.rs`'s module doc): grid.x=1 (a
/// single, fully-valid column tile) crossed with a real 4-way split.
#[test]
fn smallest_eligible_m_with_split() {
    let (w, x, expected) = build_case(GgmlDType::Q4_K, Q4_K_BLOCK_BYTES, 256, 96, 64, 1024, 7);
    let actual = run_gemm_wmma_splitk(
        rocml_kernels::GEMM_XWT_WMMA_SPLITK_Q4_K_HSACO,
        rocml_kernels::GEMM_XWT_WMMA_SPLITK_Q4_K_KERNEL,
        &w,
        &x,
        96,
        64,
        1024,
        4,
    );
    assert_close(&actual, &expected, "smallest_eligible_m_with_split");
}

/// Determinism: the same input launched twice through the full split-K +
/// reduce pipeline must produce bit-identical output — the whole point of
/// the two-pass deterministic-reduce design over an atomicAdd accumulation
/// (see `gemm_xwt_wmma_splitk_impl.h`'s module doc).
#[test]
fn deterministic_repeat() {
    let (w, x, _expected) = build_case(GgmlDType::Q4_K, Q4_K_BLOCK_BYTES, 256, 128, 128, 512, 6);
    let run = || {
        run_gemm_wmma_splitk(
            rocml_kernels::GEMM_XWT_WMMA_SPLITK_Q4_K_HSACO,
            rocml_kernels::GEMM_XWT_WMMA_SPLITK_Q4_K_KERNEL,
            &w,
            &x,
            128,
            128,
            512,
            4,
        )
    };
    let a = run();
    let b = run();
    assert_eq!(a, b, "split-K output must be bit-identical across repeats");
}
