//! GPU integration test for `gemm_xwt_mmq_q8_0` (`kernels/gemm_xwt_quant_mmq.hip`,
//! WMMA-pipeline round's int8 MMQ prototype, issue #6 lever 2): checks the
//! int8-WMMA GEMM against a plain Rust reference that implements the exact
//! same algorithm (quantize the activation the same way
//! `quantize_act_q8_blk` does, then an integer dot product per native Q8_0
//! block scaled by `d_w * d_a`) — this isolates "does the kernel compute
//! the MMQ algorithm correctly" from "how much accuracy does int8
//! activation quantization cost" (a separate, real question answered in
//! the WMMA-pipeline round's report, not by this test).
mod common;

use std::ffi::c_void;

use common::{random_row_q8_0, Rng, Q8_0_BLOCK_BYTES};
use rocml_hip::{kernel_params, Device, DeviceBuffer, LaunchConfig, Module};

const TILE_ROWS: u32 = 128;
const TILE_M: u32 = 64;
const K_STAGE: u32 = 32;
const WARPS_PER_BLOCK: u32 = 16;

/// Absmax-quantizes one row into 32-wide int8 blocks — must match
/// `quantize_act_q8_blk` bit-for-bit (checked independently by
/// `quantize_act_q8.rs`).
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

fn expected_mmq_q8_0(w_bytes: &[u8], x: &[f32], rows: usize, m: usize, n: usize) -> Vec<f32> {
    let blocks_per_row = n / 32;
    let mut out = vec![0f32; rows * m];
    for r in 0..rows {
        let x_row = &x[r * n..(r + 1) * n];
        let (a_codes, a_scale) = quantize_row(x_row);
        for col in 0..m {
            let row_bytes = &w_bytes[col * blocks_per_row * Q8_0_BLOCK_BYTES
                ..(col + 1) * blocks_per_row * Q8_0_BLOCK_BYTES];
            let mut acc = 0f32;
            for blk in 0..blocks_per_row {
                let bptr = &row_bytes[blk * Q8_0_BLOCK_BYTES..(blk + 1) * Q8_0_BLOCK_BYTES];
                let d_w = half::f16::from_le_bytes([bptr[0], bptr[1]]).to_f32();
                let mut dot: i32 = 0;
                for e in 0..32 {
                    let qw = bptr[2 + e] as i8 as i32;
                    let qa = a_codes[blk * 32 + e] as i32;
                    dot += qw * qa;
                }
                acc += d_w * a_scale[blk] * dot as f32;
            }
            out[r * m + col] = acc;
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

#[allow(clippy::too_many_arguments)]
fn run_case(rows: u32, m: u32, n: u32, seed: u32) {
    let blocks_per_row = (n / 32) as usize;
    let mut rng = Rng::new(seed);
    let mut w_bytes = Vec::with_capacity(m as usize * blocks_per_row * Q8_0_BLOCK_BYTES);
    for _ in 0..m {
        w_bytes.extend(random_row_q8_0(&mut rng, blocks_per_row));
    }
    let x = make_x(rows, n, seed + 1000);

    let want = expected_mmq_q8_0(&w_bytes, &x, rows as usize, m as usize, n as usize);

    // Host-side activation quantization (mirrors quantize_act_q8_blk,
    // verified independently) — this test targets the GEMM kernel only.
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
        Module::load_from_bytes(rocml_kernels::GEMM_XWT_MMQ_Q8_0_HSACO).expect("module load");
    let function = module
        .get_function(rocml_kernels::GEMM_XWT_MMQ_Q8_0_KERNEL)
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

    let shared_mem_bytes = TILE_ROWS * K_STAGE + TILE_M * K_STAGE + TILE_ROWS * 4 + TILE_M * 4;
    let cfg = LaunchConfig {
        grid: (m.div_ceil(TILE_M), rows.div_ceil(TILE_ROWS), 1),
        block: (32, WARPS_PER_BLOCK, 1),
        shared_mem_bytes,
    };
    // SAFETY: params matches gemm_xwt_mmq_q8_0's parameter list (const
    // signed char*, const float*, const void*, float*, unsigned x3) in
    // order, and all device buffers outlive this launch.
    unsafe { function.launch(&cfg, &mut params, None) }.expect("launch failed");

    let mut got = vec![0f32; (rows * m) as usize];
    buf_out.copy_to_host(&mut got).expect("copy out");

    // The per-32-block integer dot product is exact (both the GPU's chained
    // WMMA-i32 accumulation and this file's scalar Rust loop do the same
    // associative integer arithmetic), so the only source of disagreement
    // is float summation order across `blocks_per_row` per-block terms
    // (GPU: parallel per-warp, converted+added once per outer K-stage
    // iteration; CPU: a plain sequential loop) — the same reduction-order
    // sensitivity `gemm_xwt_quant_wmma.rs`'s own tolerance note describes.
    // A debug sweep across every shape in this file found a 2.1e-3 max
    // relative error at the largest block counts (`n=12288`, 384 blocks);
    // `5e-3` covers that with headroom while still tight enough to catch a
    // real kernel bug (wrong row/col/k index, a missing sync, a transposed
    // fragment, a wrong sign flag).
    const REL_TOL: f32 = 5e-3;
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
    run_case(128, 64, 4096, 300);
}

#[test]
fn rows_crosses_tile_boundary() {
    run_case(140, 64, 4096, 301);
}

#[test]
fn cols_crosses_tile_boundary() {
    run_case(128, 80, 4096, 302);
}

#[test]
fn degenerate_single_row() {
    run_case(1, 64, 4096, 303);
}

#[test]
fn large_shape() {
    run_case(128, 4096, 4096, 304);
}

#[test]
fn n_spans_many_blocks() {
    run_case(128, 64, 12288, 305);
}
