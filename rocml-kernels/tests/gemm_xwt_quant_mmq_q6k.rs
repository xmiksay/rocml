//! GPU integration test for `gemm_xwt_mmq_q6_k`
//! (`kernels/gemm_xwt_quant_mmq_q6_k.hip`, int8-MMQ-integration round):
//! Q6_K is symmetric like Q8_0 (no `x_sum` input), but — unlike every other
//! variant in this family — its scale granularity (16 elements) is finer
//! than the kernel's `K_STAGE`(32), so this test exercises the two-
//! accumulator-per-iteration structure that difference forces (see the
//! kernel source's module doc). Reference uses `rocml_core::quant::dequantize`
//! directly with f64 accumulation, same rationale as the Q4_K/Q5_K tests.
mod common;

use std::ffi::c_void;

use common::{random_row_q6_k, Rng, Q6_K_BLOCK_BYTES};
use rocml_core::quant::{dequantize, GgmlDType};
use rocml_hip::{kernel_params, Device, DeviceBuffer, LaunchConfig, Module};

const TILE_ROWS: u32 = 128;
const TILE_M: u32 = 64;
const K_STAGE: u32 = 32;
const WARPS_PER_BLOCK: u32 = 16;

fn quantize_row(x_row: &[f32]) -> (Vec<i8>, Vec<f32>) {
    let n_blocks = x_row.len() / 32;
    let mut codes = vec![0i8; x_row.len()];
    let mut scale = vec![0f32; n_blocks];
    for b in 0..n_blocks {
        let blk = &x_row[b * 32..b * 32 + 32];
        let amax = blk.iter().fold(0f32, |acc, &v| acc.max(v.abs()));
        let d = if amax > 0.0 { amax / 127.0 } else { 1.0 };
        for (e, &v) in blk.iter().enumerate() {
            codes[b * 32 + e] = (v / d).round().clamp(-127.0, 127.0) as i8;
        }
        scale[b] = d;
    }
    (codes, scale)
}

fn expected_mmq_q6_k(w_bytes: &[u8], x: &[f32], rows: usize, m: usize, n: usize) -> Vec<f32> {
    let row_bytes = w_bytes.len() / m;
    let deq_rows: Vec<Vec<f32>> = (0..m)
        .map(|col| {
            dequantize(
                GgmlDType::Q6_K,
                &w_bytes[col * row_bytes..(col + 1) * row_bytes],
            )
            .expect("cpu dequantize failed")
        })
        .collect();
    let mut out = vec![0f32; rows * m];
    for r in 0..rows {
        let x_row = &x[r * n..(r + 1) * n];
        let (a_codes, a_scale) = quantize_row(x_row);
        let a_deq: Vec<f64> = a_codes
            .iter()
            .enumerate()
            .map(|(e, &c)| a_scale[e / 32] as f64 * c as f64)
            .collect();
        for (col, deq) in deq_rows.iter().enumerate() {
            let acc: f64 = deq.iter().zip(&a_deq).map(|(&a, &b)| a as f64 * b).sum();
            out[r * m + col] = acc as f32;
        }
    }
    out
}

fn make_x(rows: u32, n: u32, seed: u32) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..rows * n)
        .map(|_| ((rng.next_u8() as f32) - 128.0) * 0.02)
        .collect()
}

fn run_case(rows: u32, m: u32, n: u32, seed: u32) {
    let blocks_per_row = (n / 256) as usize;
    let mut rng = Rng::new(seed);
    let mut w_bytes = Vec::with_capacity(m as usize * blocks_per_row * Q6_K_BLOCK_BYTES);
    for _ in 0..m {
        w_bytes.extend(random_row_q6_k(&mut rng, blocks_per_row));
    }
    let x = make_x(rows, n, seed + 1000);

    let want = expected_mmq_q6_k(&w_bytes, &x, rows as usize, m as usize, n as usize);

    let n_blocks = n / 32;
    let mut x_codes = vec![0i8; (rows * n) as usize];
    let mut x_scale = vec![0f32; (rows * n_blocks) as usize];
    for r in 0..rows as usize {
        let (codes, scale) = quantize_row(&x[r * n as usize..(r + 1) * n as usize]);
        x_codes[r * n as usize..(r + 1) * n as usize].copy_from_slice(&codes);
        x_scale[r * n_blocks as usize..(r + 1) * n_blocks as usize].copy_from_slice(&scale);
    }

    let _device = Device::new(0).expect("failed to select device 0");
    let module =
        Module::load_from_bytes(rocml_kernels::GEMM_XWT_MMQ_Q6_K_HSACO).expect("module load");
    let function = module
        .get_function(rocml_kernels::GEMM_XWT_MMQ_Q6_K_KERNEL)
        .expect("kernel lookup");

    let mut buf_x_codes = DeviceBuffer::<i8>::new(x_codes.len()).expect("hipMalloc x_codes");
    let mut buf_x_scale = DeviceBuffer::<f32>::new(x_scale.len()).expect("hipMalloc x_scale");
    let mut buf_w = DeviceBuffer::<u8>::new(w_bytes.len()).expect("hipMalloc w");
    let buf_out = DeviceBuffer::<f32>::new((rows * m) as usize).expect("hipMalloc out");
    buf_x_codes.copy_from_host(&x_codes).expect("copy x_codes");
    buf_x_scale.copy_from_host(&x_scale).expect("copy x_scale");
    buf_w.copy_from_host(&w_bytes).expect("copy w");

    let xc_ptr: *mut c_void = buf_x_codes.device_ptr();
    let xs_ptr: *mut c_void = buf_x_scale.device_ptr();
    let w_ptr: *mut c_void = buf_w.device_ptr();
    let out_ptr: *mut c_void = buf_out.device_ptr();
    let mut params = kernel_params!(xc_ptr, xs_ptr, w_ptr, out_ptr, rows, m, n);

    // Two `d_w` scales per column (see the kernel's module doc) doubles the
    // per-column LDS footprint relative to Q8_0's single-scale layout.
    let shared_mem_bytes = TILE_ROWS * K_STAGE + TILE_M * K_STAGE + TILE_ROWS * 4 + TILE_M * 4 * 2;
    let cfg = LaunchConfig {
        grid: (m.div_ceil(TILE_M), rows.div_ceil(TILE_ROWS), 1),
        block: (32, WARPS_PER_BLOCK, 1),
        shared_mem_bytes,
    };
    // SAFETY: params matches gemm_xwt_mmq_q6_k's parameter list (const
    // signed char*, const float*, const void*, float*, unsigned x3) in
    // order, and all device buffers outlive this launch.
    unsafe { function.launch(&cfg, &mut params, None) }.expect("launch failed");

    let mut got = vec![0f32; (rows * m) as usize];
    buf_out.copy_to_host(&mut got).expect("copy out");

    // Q6_K's synthetic test weights (full-signed-byte-range scale bytes,
    // real GGUF calibration never produces this) push some dequantized
    // values into the thousands, where reduction-order noise across many
    // large-magnitude mixed-sign terms needs more headroom than the other
    // variants here — same phenomenon `gemm_xwt_quant_wmma.rs`'s own Q6_K
    // tolerance note documents (measured up to 1.01e-2 relative there, at a
    // smaller n=2048). A debug sweep of this file's `large_shape` case
    // (n=4096, the largest reduction width here) across three seeds found
    // max relative error 1.16e-2/2.24e-2/1.57e-2 (2-3 outlier outputs out of
    // 524288 each time, everywhere else agreeing within ~6e-4) — heavy
    // cancellation on near-zero sums, not a bug, given the overwhelming
    // majority of outputs agree two orders of magnitude tighter. `3e-2`
    // covers the measured maximum with headroom.
    const REL_TOL: f32 = 3e-2;
    for i in 0..got.len() {
        let diff = (got[i] - want[i]).abs();
        let tol = REL_TOL * want[i].abs().max(1.0);
        assert!(
            diff <= tol,
            "[{i}]: got {}, want {} (diff {diff}, tol {tol}, seed {seed})",
            got[i],
            want[i]
        );
    }
}

#[test]
fn one_tile_exact() {
    run_case(128, 64, 4096, 700);
}

#[test]
fn rows_crosses_tile_boundary() {
    run_case(140, 64, 4096, 701);
}

#[test]
fn cols_crosses_tile_boundary() {
    run_case(128, 80, 4096, 702);
}

#[test]
fn degenerate_single_row() {
    run_case(1, 64, 4096, 703);
}

#[test]
fn large_shape() {
    run_case(128, 4096, 4096, 704);
}

#[test]
fn n_spans_many_superblocks() {
    run_case(128, 64, 12288, 705);
}

#[test]
fn large_shape_seed2() {
    run_case(128, 4096, 4096, 799);
}

#[test]
fn large_shape_seed3() {
    run_case(128, 4096, 4096, 850);
}
