//! GPU integration tests for the f16-weight dense-layer kernels: decode-path
//! `gemv_f16` (y = W*x) and prefill-path `gemm_xwt_f16` (out = x*W^T).
use std::ffi::c_void;

use half::f16;
use rocml_hip::{kernel_params, Device, DeviceBuffer, LaunchConfig, Module};

/// f16-weight kernels are compared against an f32 CPU reference computed
/// from the *f16-rounded* weights, so the only error left to tolerate is the
/// kernel's own reduction order — hence the looser 1e-3 tolerance.
const TOL: f32 = 1e-3;

fn assert_close(actual: &[f32], expected: &[f32], label: &str) {
    for (i, (got, want)) in actual.iter().zip(expected).enumerate() {
        let diff = (got - want).abs();
        let scale = want.abs().max(1.0);
        assert!(
            diff <= TOL * scale,
            "{label}[{i}]: got {got}, want {want} (diff {diff})"
        );
    }
}

fn run_gemv_f16(m: u32, n: u32) {
    let _device = Device::new(0).expect("failed to select device 0");
    let module =
        Module::load_from_bytes(rocml_kernels::GEMV_F16_HSACO).expect("module load failed");
    let function = module
        .get_function(rocml_kernels::GEMV_F16_KERNEL)
        .expect("kernel lookup failed");

    let w_f16: Vec<f16> = (0..(m * n))
        .map(|i| f16::from_f32(((i % 23) as f32) * 0.05 - 0.5))
        .collect();
    let x: Vec<f32> = (0..n).map(|i| ((i % 11) as f32) * 0.3 - 1.0).collect();

    let mut expected = vec![0.0f32; m as usize];
    for row in 0..m as usize {
        let mut sum = 0.0f32;
        for col in 0..n as usize {
            sum += w_f16[row * n as usize + col].to_f32() * x[col];
        }
        expected[row] = sum;
    }

    let mut buf_w = DeviceBuffer::<f16>::new(w_f16.len()).expect("hipMalloc w failed");
    let mut buf_x = DeviceBuffer::<f32>::new(x.len()).expect("hipMalloc x failed");
    let buf_y = DeviceBuffer::<f32>::new(m as usize).expect("hipMalloc y failed");
    buf_w.copy_from_host(&w_f16).expect("copy w failed");
    buf_x.copy_from_host(&x).expect("copy x failed");

    let w_ptr: *mut c_void = buf_w.device_ptr();
    let x_ptr: *mut c_void = buf_x.device_ptr();
    let y_ptr: *mut c_void = buf_y.device_ptr();
    let mut params = kernel_params!(w_ptr, x_ptr, y_ptr, m, n);

    let block = 128u32;
    let cfg = LaunchConfig {
        grid: (m, 1, 1),
        block: (block, 1, 1),
        shared_mem_bytes: block * std::mem::size_of::<f32>() as u32,
    };
    // SAFETY: params matches gemv_f16's parameter list (const __half*, const
    // float*, float*, unsigned, unsigned) in order, and all device buffers
    // outlive this launch.
    unsafe { function.launch(&cfg, &mut params, None) }.expect("kernel launch failed");

    let mut actual = vec![0.0f32; m as usize];
    buf_y.copy_to_host(&mut actual).expect("copy y failed");
    assert_close(&actual, &expected, "gemv_f16 y");
}

#[test]
fn gemv_f16_non_multiple_of_blocksize() {
    // n = 300 is not a multiple of the 128-thread block.
    run_gemv_f16(17, 300);
}

#[test]
fn gemv_f16_degenerate_single_row() {
    run_gemv_f16(1, 9);
}

fn run_gemm_xwt_f16(rows: u32, m: u32, n: u32) {
    let _device = Device::new(0).expect("failed to select device 0");
    let module =
        Module::load_from_bytes(rocml_kernels::GEMM_XWT_F16_HSACO).expect("module load failed");
    let function = module
        .get_function(rocml_kernels::GEMM_XWT_F16_KERNEL)
        .expect("kernel lookup failed");

    let x: Vec<f32> = (0..(rows * n))
        .map(|i| ((i % 13) as f32) * 0.2 - 1.0)
        .collect();
    let w_f16: Vec<f16> = (0..(m * n))
        .map(|i| f16::from_f32(((i % 19) as f32) * 0.07 - 0.6))
        .collect();

    let mut expected = vec![0.0f32; (rows * m) as usize];
    for r in 0..rows as usize {
        for j in 0..m as usize {
            let mut sum = 0.0f32;
            for k in 0..n as usize {
                sum += x[r * n as usize + k] * w_f16[j * n as usize + k].to_f32();
            }
            expected[r * m as usize + j] = sum;
        }
    }

    let mut buf_x = DeviceBuffer::<f32>::new(x.len()).expect("hipMalloc x failed");
    let mut buf_w = DeviceBuffer::<f16>::new(w_f16.len()).expect("hipMalloc w failed");
    let buf_out = DeviceBuffer::<f32>::new((rows * m) as usize).expect("hipMalloc out failed");
    buf_x.copy_from_host(&x).expect("copy x failed");
    buf_w.copy_from_host(&w_f16).expect("copy w failed");

    let x_ptr: *mut c_void = buf_x.device_ptr();
    let w_ptr: *mut c_void = buf_w.device_ptr();
    let out_ptr: *mut c_void = buf_out.device_ptr();
    let mut params = kernel_params!(x_ptr, w_ptr, out_ptr, rows, m, n);

    const TILE: u32 = 16;
    let cfg = LaunchConfig {
        grid: (m.div_ceil(TILE), rows.div_ceil(TILE), 1),
        block: (TILE, TILE, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: params matches gemm_xwt_f16's parameter list (const float*,
    // const __half*, float*, unsigned, unsigned, unsigned) in order, and all
    // device buffers outlive this launch. Block shape (16, 16, 1) matches
    // the kernel's fixed TILE.
    unsafe { function.launch(&cfg, &mut params, None) }.expect("kernel launch failed");

    let mut actual = vec![0.0f32; (rows * m) as usize];
    buf_out.copy_to_host(&mut actual).expect("copy out failed");
    assert_close(&actual, &expected, "gemm_xwt_f16 out");
}

#[test]
fn gemm_xwt_f16_non_multiple_of_tile() {
    // None of rows/m/n is a multiple of the fixed 16x16 tile.
    run_gemm_xwt_f16(37, 21, 45);
}

#[test]
fn gemm_xwt_f16_degenerate_single_row() {
    run_gemm_xwt_f16(1, 5, 7);
}
