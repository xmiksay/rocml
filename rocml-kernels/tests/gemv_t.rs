//! GPU integration test for `gemv_t_f32` (y = A^T * x), the decode-attention
//! "probs times V-cache" building block.
use std::ffi::c_void;

use rocml_hip::{kernel_params, Device, DeviceBuffer, LaunchConfig, Module};

const TOL: f32 = 1e-5;

fn assert_close(actual: &[f32], expected: &[f32], label: &str) {
    for (i, (got, want)) in actual.iter().zip(expected).enumerate() {
        let diff = (got - want).abs();
        assert!(
            diff <= TOL * want.abs().max(1.0),
            "{label}[{i}]: got {got}, want {want} (diff {diff})"
        );
    }
}

fn run_gemv_t(rows: u32, n: u32) {
    let _device = Device::new(0).expect("failed to select device 0");
    let module =
        Module::load_from_bytes(rocml_kernels::GEMV_T_F32_HSACO).expect("module load failed");
    let function = module
        .get_function(rocml_kernels::GEMV_T_F32_KERNEL)
        .expect("kernel lookup failed");

    let a: Vec<f32> = (0..(rows * n))
        .map(|i| ((i % 17) as f32) * 0.13 - 0.9)
        .collect();
    let x: Vec<f32> = (0..rows).map(|i| ((i % 7) as f32) * 0.4 - 1.1).collect();

    let mut expected = vec![0.0f32; n as usize];
    for col in 0..n as usize {
        let mut sum = 0.0f32;
        for row in 0..rows as usize {
            sum += a[row * n as usize + col] * x[row];
        }
        expected[col] = sum;
    }

    let mut buf_a = DeviceBuffer::<f32>::new(a.len()).expect("hipMalloc a failed");
    let mut buf_x = DeviceBuffer::<f32>::new(x.len()).expect("hipMalloc x failed");
    let buf_y = DeviceBuffer::<f32>::new(n as usize).expect("hipMalloc y failed");
    buf_a.copy_from_host(&a).expect("copy a failed");
    buf_x.copy_from_host(&x).expect("copy x failed");

    let a_ptr: *mut c_void = buf_a.device_ptr();
    let x_ptr: *mut c_void = buf_x.device_ptr();
    let y_ptr: *mut c_void = buf_y.device_ptr();
    let mut params = kernel_params!(a_ptr, x_ptr, y_ptr, rows, n);

    let block = 64u32;
    let cfg = LaunchConfig {
        grid: (n.div_ceil(block), 1, 1),
        block: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: params matches gemv_t_f32's parameter list (const float*,
    // const float*, float*, unsigned, unsigned) in order, and all device
    // buffers outlive this launch.
    unsafe { function.launch(&cfg, &mut params, None) }.expect("kernel launch failed");

    let mut actual = vec![0.0f32; n as usize];
    buf_y.copy_to_host(&mut actual).expect("copy y failed");
    assert_close(&actual, &expected, "gemv_t_f32 y");
}

#[test]
fn gemv_t_non_multiple_of_blocksize() {
    // n = 100 is not a multiple of the 64-thread block.
    run_gemv_t(37, 100);
}

#[test]
fn gemv_t_degenerate_single_row() {
    run_gemv_t(1, 9);
}

#[test]
fn gemv_t_degenerate_single_column() {
    run_gemv_t(23, 1);
}
