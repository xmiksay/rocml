//! GPU integration tests for `rmsnorm_f32` and `rope_neox_f32` — both
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

fn run_rmsnorm(rows: u32, n: u32) {
    let _device = Device::new(0).expect("failed to select device 0");
    let module =
        Module::load_from_bytes(rocml_kernels::RMSNORM_F32_HSACO).expect("module load failed");
    let function = module
        .get_function(rocml_kernels::RMSNORM_F32_KERNEL)
        .expect("kernel lookup failed");

    let eps = 1e-5f32;
    let x: Vec<f32> = (0..(rows * n))
        .map(|i| ((i % 29) as f32) * 0.1 - 1.4)
        .collect();
    let weight: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.01).collect();

    let mut expected = vec![0.0f32; (rows * n) as usize];
    for r in 0..rows as usize {
        let row = &x[r * n as usize..(r + 1) * n as usize];
        let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / n as f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();
        for i in 0..n as usize {
            expected[r * n as usize + i] = row[i] * inv_rms * weight[i];
        }
    }

    let mut buf_x = DeviceBuffer::<f32>::new(x.len()).expect("hipMalloc x failed");
    let mut buf_w = DeviceBuffer::<f32>::new(weight.len()).expect("hipMalloc weight failed");
    let buf_out = DeviceBuffer::<f32>::new((rows * n) as usize).expect("hipMalloc out failed");
    buf_x.copy_from_host(&x).expect("copy x failed");
    buf_w.copy_from_host(&weight).expect("copy weight failed");

    let x_ptr: *mut c_void = buf_x.device_ptr();
    let w_ptr: *mut c_void = buf_w.device_ptr();
    let out_ptr: *mut c_void = buf_out.device_ptr();
    let mut params = kernel_params!(x_ptr, w_ptr, out_ptr, rows, n, eps);

    let block = 128u32;
    let cfg = LaunchConfig {
        grid: (rows, 1, 1),
        block: (block, 1, 1),
        shared_mem_bytes: block * std::mem::size_of::<f32>() as u32,
    };
    // SAFETY: params matches rmsnorm_f32's parameter list (const float*,
    // const float*, float*, unsigned, unsigned, float) in order, and all
    // device buffers outlive this launch.
    unsafe { function.launch(&cfg, &mut params, None) }.expect("kernel launch failed");

    let mut actual = vec![0.0f32; (rows * n) as usize];
    buf_out.copy_to_host(&mut actual).expect("copy out failed");
    assert_close(&actual, &expected, "rmsnorm out");
}

#[test]
fn rmsnorm_non_multiple_of_blocksize() {
    // n = 300 is not a multiple of the 128-thread block.
    run_rmsnorm(5, 300);
}

#[test]
fn rmsnorm_degenerate_single_row() {
    run_rmsnorm(1, 7);
}

fn run_rope(tokens: u32, heads: u32, head_dim: u32, pos_base: u32) {
    let _device = Device::new(0).expect("failed to select device 0");
    let module =
        Module::load_from_bytes(rocml_kernels::ROPE_NEOX_F32_HSACO).expect("module load failed");
    let function = module
        .get_function(rocml_kernels::ROPE_NEOX_F32_KERNEL)
        .expect("kernel lookup failed");

    let theta_base = 10000.0f32;
    let half_dim = head_dim / 2;
    let total_elems = (tokens * heads * head_dim) as usize;
    let x: Vec<f32> = (0..total_elems)
        .map(|i| ((i % 31) as f32) * 0.05 - 0.7)
        .collect();

    // CPU reference mirrors the kernel exactly: pair i with i + half_dim,
    // inv_freq = theta_base^(-2i/head_dim), position = pos_base + t.
    let mut expected = x.clone();
    for t in 0..tokens {
        let pos = pos_base + t;
        for h in 0..heads {
            let base = ((t * heads + h) * head_dim) as usize;
            for i in 0..half_dim {
                let inv_freq = theta_base.powf(-2.0 * i as f32 / head_dim as f32);
                let angle = pos as f32 * inv_freq;
                let (sin_a, cos_a) = angle.sin_cos();
                let x0 = x[base + i as usize];
                let x1 = x[base + (i + half_dim) as usize];
                expected[base + i as usize] = x0 * cos_a - x1 * sin_a;
                expected[base + (i + half_dim) as usize] = x0 * sin_a + x1 * cos_a;
            }
        }
    }

    let mut buf_x = DeviceBuffer::<f32>::new(total_elems).expect("hipMalloc x failed");
    buf_x.copy_from_host(&x).expect("copy x failed");

    let x_ptr: *mut c_void = buf_x.device_ptr();
    let mut params = kernel_params!(x_ptr, tokens, heads, head_dim, pos_base, theta_base);

    let total_threads = tokens * heads * half_dim;
    let block = 64u32;
    let cfg = LaunchConfig {
        grid: (total_threads.div_ceil(block), 1, 1),
        block: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: params matches rope_neox_f32's parameter list (float*,
    // unsigned, unsigned, unsigned, unsigned, float) in order, and buf_x
    // outlives this launch.
    unsafe { function.launch(&cfg, &mut params, None) }.expect("kernel launch failed");

    let mut actual = vec![0.0f32; total_elems];
    buf_x.copy_to_host(&mut actual).expect("copy x back failed");
    assert_close(&actual, &expected, "rope x");
}

#[test]
fn rope_non_multiple_of_blocksize() {
    // total = tokens*heads*half_dim = 10*3*8 = 240, not a multiple of the
    // 64-thread block.
    run_rope(10, 3, 16, 5);
}

#[test]
fn rope_degenerate_single_token() {
    run_rope(1, 2, 4, 0);
}
