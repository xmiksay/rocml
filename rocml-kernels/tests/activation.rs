//! GPU integration tests for `silu_mul_f32` and `softmax_varlen_f32` — both
//! pure-f32 kernels, so tolerance is tight (1e-5).
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

fn run_silu_mul(n: u32) {
    let _device = Device::new(0).expect("failed to select device 0");
    let module =
        Module::load_from_bytes(rocml_kernels::SILU_MUL_F32_HSACO).expect("module load failed");
    let function = module
        .get_function(rocml_kernels::SILU_MUL_F32_KERNEL)
        .expect("kernel lookup failed");

    let gate: Vec<f32> = (0..n).map(|i| ((i % 23) as f32) * 0.1 - 1.2).collect();
    let up: Vec<f32> = (0..n).map(|i| ((i % 17) as f32) * 0.2 - 1.5).collect();
    let expected: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
        .collect();

    let mut buf_gate = DeviceBuffer::<f32>::new(gate.len()).expect("hipMalloc gate failed");
    let mut buf_up = DeviceBuffer::<f32>::new(up.len()).expect("hipMalloc up failed");
    let buf_out = DeviceBuffer::<f32>::new(n as usize).expect("hipMalloc out failed");
    buf_gate.copy_from_host(&gate).expect("copy gate failed");
    buf_up.copy_from_host(&up).expect("copy up failed");

    let gate_ptr: *mut c_void = buf_gate.device_ptr();
    let up_ptr: *mut c_void = buf_up.device_ptr();
    let out_ptr: *mut c_void = buf_out.device_ptr();
    let mut params = kernel_params!(gate_ptr, up_ptr, out_ptr, n);

    let block = 256u32;
    let cfg = LaunchConfig {
        grid: (n.div_ceil(block), 1, 1),
        block: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: params matches silu_mul_f32's parameter list (const float*,
    // const float*, float*, unsigned) in order, and all device buffers
    // outlive this launch.
    unsafe { function.launch(&cfg, &mut params, None) }.expect("kernel launch failed");

    let mut actual = vec![0.0f32; n as usize];
    buf_out.copy_to_host(&mut actual).expect("copy out failed");
    assert_close(&actual, &expected, "silu_mul out");
}

#[test]
fn silu_mul_non_multiple_of_blocksize() {
    // n = 1000 is not a multiple of the 256-thread block.
    run_silu_mul(1000);
}

#[test]
fn silu_mul_degenerate_single_element() {
    run_silu_mul(1);
}

fn run_softmax(rows: u32, cols: u32, valid_len: &[u32], scale: f32) {
    assert_eq!(valid_len.len(), rows as usize);
    let _device = Device::new(0).expect("failed to select device 0");
    let module = Module::load_from_bytes(rocml_kernels::SOFTMAX_VARLEN_F32_HSACO)
        .expect("module load failed");
    let function = module
        .get_function(rocml_kernels::SOFTMAX_VARLEN_F32_KERNEL)
        .expect("kernel lookup failed");

    let x: Vec<f32> = (0..(rows * cols))
        .map(|i| ((i % 41) as f32) * 0.07 - 1.3)
        .collect();

    let mut expected = vec![0.0f32; (rows * cols) as usize];
    for r in 0..rows as usize {
        let vlen = (valid_len[r] as usize).min(cols as usize);
        let row = &x[r * cols as usize..(r + 1) * cols as usize];
        if vlen == 0 {
            continue; // already zero-initialized
        }
        let row_max = row[..vlen]
            .iter()
            .map(|v| v * scale)
            .fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = row[..vlen]
            .iter()
            .map(|v| (v * scale - row_max).exp())
            .sum();
        for i in 0..vlen {
            expected[r * cols as usize + i] = (row[i] * scale - row_max).exp() / sum;
        }
    }

    let mut buf_x = DeviceBuffer::<f32>::new(x.len()).expect("hipMalloc x failed");
    let mut buf_vlen = DeviceBuffer::<u32>::new(valid_len.len()).expect("hipMalloc vlen failed");
    buf_x.copy_from_host(&x).expect("copy x failed");
    buf_vlen
        .copy_from_host(valid_len)
        .expect("copy vlen failed");

    let x_ptr: *mut c_void = buf_x.device_ptr();
    let vlen_ptr: *mut c_void = buf_vlen.device_ptr();
    let mut params = kernel_params!(x_ptr, vlen_ptr, rows, cols, scale);

    let block = 128u32;
    let cfg = LaunchConfig {
        grid: (rows, 1, 1),
        block: (block, 1, 1),
        shared_mem_bytes: block * std::mem::size_of::<f32>() as u32,
    };
    // SAFETY: params matches softmax_varlen_f32's parameter list (float*,
    // const unsigned*, unsigned, unsigned, float) in order, and all device
    // buffers outlive this launch.
    unsafe { function.launch(&cfg, &mut params, None) }.expect("kernel launch failed");

    let mut actual = vec![0.0f32; (rows * cols) as usize];
    buf_x.copy_to_host(&mut actual).expect("copy x back failed");
    assert_close(&actual, &expected, "softmax x");
}

#[test]
fn softmax_non_multiple_of_blocksize_with_mixed_valid_len() {
    // cols = 300 is not a multiple of the 128-thread block; valid_len covers
    // full rows, a short row, a single-entry row, and an empty row.
    run_softmax(5, 300, &[300, 150, 1, 299, 0], 0.5);
}

#[test]
fn softmax_degenerate_single_row() {
    run_softmax(1, 5, &[3], 1.0);
}
