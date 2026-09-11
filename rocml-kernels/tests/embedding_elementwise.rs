//! GPU integration tests for `embedding_f16_f32`, `add_inplace_f32`, and the
//! f16<->f32 casts.
use std::ffi::c_void;

use half::f16;
use rocml_hip::{kernel_params, Device, DeviceBuffer, LaunchConfig, Module};

fn assert_close_f32(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    for (i, (got, want)) in actual.iter().zip(expected).enumerate() {
        let diff = (got - want).abs();
        assert!(
            diff <= tol * want.abs().max(1.0),
            "{label}[{i}]: got {got}, want {want} (diff {diff})"
        );
    }
}

fn run_embedding(t: u32, vocab: u32, dim: u32) {
    let _device = Device::new(0).expect("failed to select device 0");
    let module = Module::load_from_bytes(rocml_kernels::EMBEDDING_F16_F32_HSACO)
        .expect("module load failed");
    let function = module
        .get_function(rocml_kernels::EMBEDDING_F16_F32_KERNEL)
        .expect("kernel lookup failed");

    let ids: Vec<u32> = (0..t).map(|i| (i * 7) % vocab).collect();
    let table: Vec<f16> = (0..(vocab * dim))
        .map(|i| f16::from_f32(((i % 37) as f32) * 0.03 - 0.5))
        .collect();

    let mut expected = vec![0.0f32; (t * dim) as usize];
    for (row, &id) in ids.iter().enumerate() {
        for d in 0..dim as usize {
            expected[row * dim as usize + d] = table[id as usize * dim as usize + d].to_f32();
        }
    }

    let mut buf_ids = DeviceBuffer::<u32>::new(ids.len()).expect("hipMalloc ids failed");
    let mut buf_table = DeviceBuffer::<f16>::new(table.len()).expect("hipMalloc table failed");
    let buf_out = DeviceBuffer::<f32>::new((t * dim) as usize).expect("hipMalloc out failed");
    buf_ids.copy_from_host(&ids).expect("copy ids failed");
    buf_table.copy_from_host(&table).expect("copy table failed");

    let ids_ptr: *mut c_void = buf_ids.device_ptr();
    let table_ptr: *mut c_void = buf_table.device_ptr();
    let out_ptr: *mut c_void = buf_out.device_ptr();
    let mut params = kernel_params!(ids_ptr, table_ptr, out_ptr, t, dim);

    let cfg = LaunchConfig {
        grid: (t, 1, 1),
        block: (64, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: params matches embedding_f16_f32's parameter list (const
    // unsigned*, const __half*, float*, unsigned, unsigned) in order, and
    // all device buffers outlive this launch.
    unsafe { function.launch(&cfg, &mut params, None) }.expect("kernel launch failed");

    let mut actual = vec![0.0f32; (t * dim) as usize];
    buf_out.copy_to_host(&mut actual).expect("copy out failed");
    // f16 -> f32 widening is exact, so no rounding tolerance is needed.
    assert_eq!(actual, expected);
}

#[test]
fn embedding_non_multiple_of_blocksize() {
    // dim = 100 is not a multiple of the 64-thread block.
    run_embedding(7, 50, 100);
}

#[test]
fn embedding_degenerate_single_token() {
    run_embedding(1, 10, 8);
}

fn run_add_inplace(n: u32) {
    let _device = Device::new(0).expect("failed to select device 0");
    let module =
        Module::load_from_bytes(rocml_kernels::ELEMENTWISE_HSACO).expect("module load failed");
    let function = module
        .get_function(rocml_kernels::ADD_INPLACE_F32_KERNEL)
        .expect("kernel lookup failed");

    let acc: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 3.0).collect();
    let x: Vec<f32> = (0..n).map(|i| ((i % 7) as f32) * 0.1).collect();
    let expected: Vec<f32> = acc.iter().zip(&x).map(|(a, b)| a + b).collect();

    let mut buf_acc = DeviceBuffer::<f32>::new(acc.len()).expect("hipMalloc acc failed");
    let mut buf_x = DeviceBuffer::<f32>::new(x.len()).expect("hipMalloc x failed");
    buf_acc.copy_from_host(&acc).expect("copy acc failed");
    buf_x.copy_from_host(&x).expect("copy x failed");

    let acc_ptr: *mut c_void = buf_acc.device_ptr();
    let x_ptr: *mut c_void = buf_x.device_ptr();
    let mut params = kernel_params!(acc_ptr, x_ptr, n);

    let block = 256u32;
    let cfg = LaunchConfig {
        grid: (n.div_ceil(block), 1, 1),
        block: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: params matches add_inplace_f32's parameter list (float*, const
    // float*, unsigned) in order, and both buffers outlive this launch.
    unsafe { function.launch(&cfg, &mut params, None) }.expect("kernel launch failed");

    let mut actual = vec![0.0f32; n as usize];
    buf_acc
        .copy_to_host(&mut actual)
        .expect("copy acc back failed");
    assert_close_f32(&actual, &expected, 1e-5, "add_inplace acc");
}

#[test]
fn add_inplace_non_multiple_of_blocksize() {
    run_add_inplace(777);
}

#[test]
fn add_inplace_degenerate_single_element() {
    run_add_inplace(1);
}

fn run_casts(n: u32) {
    let _device = Device::new(0).expect("failed to select device 0");
    let module =
        Module::load_from_bytes(rocml_kernels::ELEMENTWISE_HSACO).expect("module load failed");
    let cast_up = module
        .get_function(rocml_kernels::CAST_F16_F32_KERNEL)
        .expect("cast_f16_f32 lookup failed");
    let cast_down = module
        .get_function(rocml_kernels::CAST_F32_F16_KERNEL)
        .expect("cast_f32_f16 lookup failed");

    let src_f16: Vec<f16> = (0..n)
        .map(|i| f16::from_f32(((i % 53) as f32) * 0.037 - 1.0))
        .collect();
    let src_f32: Vec<f32> = (0..n).map(|i| ((i % 61) as f32) * 0.041 - 1.2).collect();

    let mut buf_in16 = DeviceBuffer::<f16>::new(n as usize).expect("hipMalloc in16 failed");
    let buf_out32 = DeviceBuffer::<f32>::new(n as usize).expect("hipMalloc out32 failed");
    buf_in16.copy_from_host(&src_f16).expect("copy in16 failed");
    let in16_ptr: *mut c_void = buf_in16.device_ptr();
    let out32_ptr: *mut c_void = buf_out32.device_ptr();
    let mut up_params = kernel_params!(in16_ptr, out32_ptr, n);

    let mut buf_in32 = DeviceBuffer::<f32>::new(n as usize).expect("hipMalloc in32 failed");
    let buf_out16 = DeviceBuffer::<f16>::new(n as usize).expect("hipMalloc out16 failed");
    buf_in32.copy_from_host(&src_f32).expect("copy in32 failed");
    let in32_ptr: *mut c_void = buf_in32.device_ptr();
    let out16_ptr: *mut c_void = buf_out16.device_ptr();
    let mut down_params = kernel_params!(in32_ptr, out16_ptr, n);

    let block = 256u32;
    let cfg = LaunchConfig {
        grid: (n.div_ceil(block), 1, 1),
        block: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: up_params/down_params match cast_f16_f32's (const __half*,
    // float*, unsigned) and cast_f32_f16's (const float*, __half*, unsigned)
    // parameter lists respectively, and all device buffers outlive their launch.
    unsafe { cast_up.launch(&cfg, &mut up_params, None) }.expect("cast_f16_f32 launch failed");
    unsafe { cast_down.launch(&cfg, &mut down_params, None) }.expect("cast_f32_f16 launch failed");

    let mut actual32 = vec![0.0f32; n as usize];
    buf_out32
        .copy_to_host(&mut actual32)
        .expect("copy out32 failed");
    // f16 -> f32 widening is exact.
    let expected32: Vec<f32> = src_f16.iter().map(|v| v.to_f32()).collect();
    assert_eq!(actual32, expected32);

    let mut actual16 = vec![f16::from_f32(0.0); n as usize];
    buf_out16
        .copy_to_host(&mut actual16)
        .expect("copy out16 failed");
    // f32 -> f16 rounding must match bit-for-bit: both host (`half` crate)
    // and device (`__float2half`) round to nearest, ties to even.
    let expected16: Vec<f16> = src_f32.iter().map(|&v| f16::from_f32(v)).collect();
    assert_eq!(
        actual16.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        expected16.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}

#[test]
fn casts_non_multiple_of_blocksize() {
    run_casts(777);
}

#[test]
fn casts_degenerate_single_element() {
    run_casts(1);
}
