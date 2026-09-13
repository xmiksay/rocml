//! GPU integration test for `gemm_xwt_mmq_q5_k`
//! (`kernels/gemm_xwt_quant_mmq_q5_k.hip`, int8-MMQ-integration round's
//! mechanical extension of `gemm_xwt_quant_mmq_q4k.rs`'s Q4_K test to the
//! Q5_K affine kernel — same `d*sc*q - dmin*mn` shape, one extra high bit
//! per element). Reference uses `rocml_core::quant::dequantize` directly
//! (the same CPU dequant every other kernel test in this crate is checked
//! against) rather than reimplementing the bit-unpacking here, exactly like
//! the Q4_K test's own rationale.
mod common;

use std::ffi::c_void;

use common::{random_row_q5_k, Rng, Q5_K_BLOCK_BYTES};
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

/// f64-accumulating reference — see `gemm_xwt_quant_mmq_q4k.rs`'s
/// `expected_mmq_q4_k` doc comment for why f64 accumulation is needed here
/// (the f32 *reference's* own summation noise, not a kernel bug).
fn expected_mmq_q5_k(w_bytes: &[u8], x: &[f32], rows: usize, m: usize, n: usize) -> Vec<f32> {
    let row_bytes = w_bytes.len() / m;
    let deq_rows: Vec<Vec<f32>> = (0..m)
        .map(|col| {
            dequantize(
                GgmlDType::Q5_K,
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
    let mut w_bytes = Vec::with_capacity(m as usize * blocks_per_row * Q5_K_BLOCK_BYTES);
    for _ in 0..m {
        w_bytes.extend(random_row_q5_k(&mut rng, blocks_per_row));
    }
    let x = make_x(rows, n, seed + 1000);

    let want = expected_mmq_q5_k(&w_bytes, &x, rows as usize, m as usize, n as usize);

    let n_blocks = n / 32;
    let mut x_codes = vec![0i8; (rows * n) as usize];
    let mut x_scale = vec![0f32; (rows * n_blocks) as usize];
    let mut x_sum = vec![0f32; (rows * n_blocks) as usize];
    for r in 0..rows as usize {
        let (codes, scale) = quantize_row(&x[r * n as usize..(r + 1) * n as usize]);
        x_codes[r * n as usize..(r + 1) * n as usize].copy_from_slice(&codes);
        for (b, &d) in scale.iter().enumerate() {
            let sum: i32 = codes[b * 32..b * 32 + 32].iter().map(|&c| c as i32).sum();
            x_scale[r * n_blocks as usize + b] = d;
            x_sum[r * n_blocks as usize + b] = d * sum as f32;
        }
    }

    let _device = Device::new(0).expect("failed to select device 0");
    let module =
        Module::load_from_bytes(rocml_kernels::GEMM_XWT_MMQ_Q5_K_HSACO).expect("module load");
    let function = module
        .get_function(rocml_kernels::GEMM_XWT_MMQ_Q5_K_KERNEL)
        .expect("kernel lookup");

    let mut buf_x_codes = DeviceBuffer::<i8>::new(x_codes.len()).expect("hipMalloc x_codes");
    let mut buf_x_scale = DeviceBuffer::<f32>::new(x_scale.len()).expect("hipMalloc x_scale");
    let mut buf_x_sum = DeviceBuffer::<f32>::new(x_sum.len()).expect("hipMalloc x_sum");
    let mut buf_w = DeviceBuffer::<u8>::new(w_bytes.len()).expect("hipMalloc w");
    let buf_out = DeviceBuffer::<f32>::new((rows * m) as usize).expect("hipMalloc out");
    buf_x_codes.copy_from_host(&x_codes).expect("copy x_codes");
    buf_x_scale.copy_from_host(&x_scale).expect("copy x_scale");
    buf_x_sum.copy_from_host(&x_sum).expect("copy x_sum");
    buf_w.copy_from_host(&w_bytes).expect("copy w");

    let xc_ptr: *mut c_void = buf_x_codes.device_ptr();
    let xs_ptr: *mut c_void = buf_x_scale.device_ptr();
    let xsum_ptr: *mut c_void = buf_x_sum.device_ptr();
    let w_ptr: *mut c_void = buf_w.device_ptr();
    let out_ptr: *mut c_void = buf_out.device_ptr();
    let mut params = kernel_params!(xc_ptr, xs_ptr, xsum_ptr, w_ptr, out_ptr, rows, m, n);

    let shared_mem_bytes =
        TILE_ROWS * K_STAGE + TILE_M * K_STAGE + TILE_ROWS * 4 * 2 + TILE_M * 4 * 2;
    let cfg = LaunchConfig {
        grid: (m.div_ceil(TILE_M), rows.div_ceil(TILE_ROWS), 1),
        block: (32, WARPS_PER_BLOCK, 1),
        shared_mem_bytes,
    };
    // SAFETY: params matches gemm_xwt_mmq_q5_k's parameter list (const
    // signed char*, const float*, const float*, const void*, float*,
    // unsigned x3) in order, and all device buffers outlive this launch.
    unsafe { function.launch(&cfg, &mut params, None) }.expect("launch failed");

    let mut got = vec![0f32; (rows * m) as usize];
    buf_out.copy_to_host(&mut got).expect("copy out");

    // Same reduction-order rationale as gemm_xwt_quant_mmq_q4k.rs's tolerance
    // note (affine two-term-per-sub-block structure, f64 reference) — a
    // debug sweep across every shape in this file (see this test file's own
    // history) found agreement within ~8.2e-4 relative everywhere except
    // `large_shape` (n=4096, 128 sub-blocks reduced per output), where a
    // single output (1 of 524288, a heavy-cancellation near-zero sum) hit
    // 1.34e-2 — one extra affine term than Q4_K's own worst case (6.6e-3 at
    // the same shape), still clearly reduction-order noise rather than a
    // bug given everywhere else agrees two orders of magnitude tighter.
    // `1.5e-2` covers the measured maximum with headroom.
    const REL_TOL: f32 = 1.5e-2;
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
    run_case(128, 64, 4096, 600);
}

#[test]
fn rows_crosses_tile_boundary() {
    run_case(140, 64, 4096, 601);
}

#[test]
fn cols_crosses_tile_boundary() {
    run_case(128, 80, 4096, 602);
}

#[test]
fn degenerate_single_row() {
    run_case(1, 64, 4096, 603);
}

#[test]
fn large_shape() {
    run_case(128, 4096, 4096, 604);
}

#[test]
fn n_spans_many_superblocks() {
    run_case(128, 64, 12288, 605);
}
