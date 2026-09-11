//! End-to-end smoke test: loads the embedded code objects and launches each
//! kernel against a real GPU, checking results against a CPU reference.
use std::ffi::c_void;

use rocml_hip::{kernel_params, Device, DeviceBuffer, LaunchConfig, Module};

#[test]
fn vec_add_f32_matches_cpu() {
    let _device = Device::new(0).expect("failed to select device 0");
    let module =
        Module::load_from_bytes(rocml_kernels::VEC_ADD_F32_HSACO).expect("module load failed");
    let function = module
        .get_function(rocml_kernels::VEC_ADD_F32_KERNEL)
        .expect("kernel lookup failed");

    let n: u32 = 1 << 20;
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.5).collect();
    let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();

    let mut buf_a = DeviceBuffer::<f32>::new(n as usize).expect("hipMalloc a failed");
    let mut buf_b = DeviceBuffer::<f32>::new(n as usize).expect("hipMalloc b failed");
    let buf_out = DeviceBuffer::<f32>::new(n as usize).expect("hipMalloc out failed");
    buf_a.copy_from_host(&a).expect("copy a failed");
    buf_b.copy_from_host(&b).expect("copy b failed");

    let a_ptr: *mut c_void = buf_a.device_ptr();
    let b_ptr: *mut c_void = buf_b.device_ptr();
    let out_ptr: *mut c_void = buf_out.device_ptr();
    let mut params = kernel_params!(a_ptr, b_ptr, out_ptr, n);

    let block = 256u32;
    let grid = n.div_ceil(block);
    let cfg = LaunchConfig {
        grid: (grid, 1, 1),
        block: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: params was built by kernel_params! in the exact order/types of
    // vec_add_f32's parameter list (const float*, const float*, float*,
    // unsigned), and all three device buffers outlive this launch.
    unsafe { function.launch(&cfg, &mut params, None) }.expect("kernel launch failed");

    let mut actual = vec![0.0f32; n as usize];
    buf_out.copy_to_host(&mut actual).expect("copy out failed");
    assert_eq!(actual, expected);
}

#[test]
fn gemv_f32_matches_cpu() {
    let _device = Device::new(0).expect("failed to select device 0");
    let module =
        Module::load_from_bytes(rocml_kernels::GEMV_F32_HSACO).expect("module load failed");
    let function = module
        .get_function(rocml_kernels::GEMV_F32_KERNEL)
        .expect("kernel lookup failed");

    let m: u32 = 128;
    let n: u32 = 256;
    let mat: Vec<f32> = (0..(m * n))
        .map(|i| ((i % 17) as f32) * 0.1 - 0.5)
        .collect();
    let x: Vec<f32> = (0..n).map(|i| ((i % 13) as f32) * 0.2 - 1.0).collect();

    let mut expected = vec![0.0f32; m as usize];
    for row in 0..m as usize {
        let mut sum = 0.0f32;
        for col in 0..n as usize {
            sum += mat[row * n as usize + col] * x[col];
        }
        expected[row] = sum;
    }

    let mut buf_mat = DeviceBuffer::<f32>::new(mat.len()).expect("hipMalloc mat failed");
    let mut buf_x = DeviceBuffer::<f32>::new(x.len()).expect("hipMalloc x failed");
    let buf_y = DeviceBuffer::<f32>::new(m as usize).expect("hipMalloc y failed");
    buf_mat.copy_from_host(&mat).expect("copy mat failed");
    buf_x.copy_from_host(&x).expect("copy x failed");

    let mat_ptr: *mut c_void = buf_mat.device_ptr();
    let x_ptr: *mut c_void = buf_x.device_ptr();
    let y_ptr: *mut c_void = buf_y.device_ptr();
    let mut params = kernel_params!(mat_ptr, x_ptr, y_ptr, m, n);

    // Block size must be a power of two: gemv_f32's reduction halves the
    // shared-memory buffer down to index 0.
    let block = 128u32;
    let cfg = LaunchConfig {
        grid: (m, 1, 1),
        block: (block, 1, 1),
        shared_mem_bytes: block * std::mem::size_of::<f32>() as u32,
    };
    // SAFETY: params matches gemv_f32's parameter list (const float*, const
    // float*, float*, unsigned, unsigned) in order, and all device buffers
    // outlive this launch.
    unsafe { function.launch(&cfg, &mut params, None) }.expect("kernel launch failed");

    let mut actual = vec![0.0f32; m as usize];
    buf_y.copy_to_host(&mut actual).expect("copy y failed");

    for (row, (got, want)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            (got - want).abs() < 1e-5,
            "row {row}: got {got}, want {want} (diff {})",
            (got - want).abs()
        );
    }
}
