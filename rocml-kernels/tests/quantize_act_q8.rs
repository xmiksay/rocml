//! GPU integration test for `quantize_act_q8_blk` (`kernels/quantize_act_q8.hip`,
//! WMMA-pipeline round's int8 MMQ prototype, issue #6 lever 2): checks the
//! per-32-element-block `(code, scale, block_sum)` output against a plain
//! Rust reference implementing the exact same absmax-quantize algorithm.
mod common;

use std::ffi::c_void;

use common::Rng;
use rocml_hip::{kernel_params, Device, DeviceBuffer, LaunchConfig, Module};

const WARPS_PER_BLOCK: u32 = 8;

fn quantize_ref(x: &[f32], rows: u32, n: u32) -> (Vec<i8>, Vec<f32>, Vec<f32>) {
    let n_blocks = (n / 32) as usize;
    let mut codes = vec![0i8; (rows * n) as usize];
    let mut scale = vec![0f32; rows as usize * n_blocks];
    let mut block_sum = vec![0f32; rows as usize * n_blocks];
    for r in 0..rows as usize {
        for b in 0..n_blocks {
            let blk = &x[r * n as usize + b * 32..r * n as usize + b * 32 + 32];
            let amax = blk.iter().fold(0f32, |acc, &v| acc.max(v.abs()));
            let d = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            let mut sum = 0i32;
            for (e, &v) in blk.iter().enumerate() {
                let q = (v / d).round().clamp(-127.0, 127.0) as i32;
                codes[r * n as usize + b * 32 + e] = q as i8;
                sum += q;
            }
            scale[r * n_blocks + b] = d;
            block_sum[r * n_blocks + b] = d * sum as f32;
        }
    }
    (codes, scale, block_sum)
}

fn run_case(rows: u32, n: u32, seed: u32) {
    let mut rng = Rng::new(seed);
    // Wide dynamic range (including some exact zeros) so the amax==0 edge
    // case and typical activation magnitudes are both exercised.
    let x: Vec<f32> = (0..rows * n)
        .map(|i| {
            if i % 97 == 0 {
                0.0
            } else {
                ((rng.next_u8() as f32) - 128.0) * 0.037
            }
        })
        .collect();

    let (want_codes, want_scale, want_sum) = quantize_ref(&x, rows, n);

    let _device = Device::new(0).expect("failed to select device 0");
    let module =
        Module::load_from_bytes(rocml_kernels::QUANTIZE_ACT_Q8_BLK_HSACO).expect("module load");
    let function = module
        .get_function(rocml_kernels::QUANTIZE_ACT_Q8_BLK_KERNEL)
        .expect("kernel lookup");

    let n_blocks = n / 32;
    let mut buf_x = DeviceBuffer::<f32>::new(x.len()).expect("hipMalloc x");
    let buf_codes = DeviceBuffer::<i8>::new((rows * n) as usize).expect("hipMalloc codes");
    let buf_scale = DeviceBuffer::<f32>::new((rows * n_blocks) as usize).expect("hipMalloc scale");
    let buf_sum = DeviceBuffer::<f32>::new((rows * n_blocks) as usize).expect("hipMalloc sum");
    buf_x.copy_from_host(&x).expect("copy x");

    let x_ptr: *mut c_void = buf_x.device_ptr();
    let codes_ptr: *mut c_void = buf_codes.device_ptr();
    let scale_ptr: *mut c_void = buf_scale.device_ptr();
    let sum_ptr: *mut c_void = buf_sum.device_ptr();
    let mut params = kernel_params!(x_ptr, codes_ptr, scale_ptr, sum_ptr, rows, n);

    let total_warps = rows * n_blocks;
    let cfg = LaunchConfig {
        grid: (total_warps.div_ceil(WARPS_PER_BLOCK), 1, 1),
        block: (32, WARPS_PER_BLOCK, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: params matches quantize_act_q8_blk's parameter list (const
    // float*, signed char*, float*, float*, unsigned, unsigned) in order,
    // and all device buffers outlive this launch.
    unsafe { function.launch(&cfg, &mut params, None) }.expect("launch failed");

    let mut got_codes = vec![0i8; (rows * n) as usize];
    let mut got_scale = vec![0f32; (rows * n_blocks) as usize];
    let mut got_sum = vec![0f32; (rows * n_blocks) as usize];
    buf_codes.copy_to_host(&mut got_codes).expect("copy codes");
    buf_scale.copy_to_host(&mut got_scale).expect("copy scale");
    buf_sum.copy_to_host(&mut got_sum).expect("copy sum");

    assert_eq!(got_codes, want_codes, "codes mismatch (seed {seed})");
    for i in 0..got_scale.len() {
        assert!(
            (got_scale[i] - want_scale[i]).abs() < 1e-6,
            "scale[{i}]: got {}, want {}",
            got_scale[i],
            want_scale[i]
        );
        assert!(
            (got_sum[i] - want_sum[i]).abs() < 1e-3,
            "block_sum[{i}]: got {}, want {}",
            got_sum[i],
            want_sum[i]
        );
    }
}

#[test]
fn single_block_single_row() {
    run_case(1, 32, 1);
}

#[test]
fn multiple_blocks_and_rows() {
    run_case(37, 256, 2);
}

#[test]
fn large_shape_matches_chunked_prefill() {
    run_case(128, 4096, 3);
}

#[test]
fn all_zero_block_uses_scale_one() {
    // A row that's exactly zero for a whole 32-block must not divide by
    // zero: scale falls back to 1.0 and every code stays 0.
    let rows = 1u32;
    let n = 32u32;
    let x = vec![0.0f32; 32];
    let _device = Device::new(0).expect("failed to select device 0");
    let module =
        Module::load_from_bytes(rocml_kernels::QUANTIZE_ACT_Q8_BLK_HSACO).expect("module load");
    let function = module
        .get_function(rocml_kernels::QUANTIZE_ACT_Q8_BLK_KERNEL)
        .expect("kernel lookup");

    let mut buf_x = DeviceBuffer::<f32>::new(x.len()).expect("hipMalloc x");
    let buf_codes = DeviceBuffer::<i8>::new((rows * n) as usize).expect("hipMalloc codes");
    let buf_scale = DeviceBuffer::<f32>::new(1).expect("hipMalloc scale");
    let buf_sum = DeviceBuffer::<f32>::new(1).expect("hipMalloc sum");
    buf_x.copy_from_host(&x).expect("copy x");
    let x_ptr: *mut c_void = buf_x.device_ptr();
    let codes_ptr: *mut c_void = buf_codes.device_ptr();
    let scale_ptr: *mut c_void = buf_scale.device_ptr();
    let sum_ptr: *mut c_void = buf_sum.device_ptr();
    let mut params = kernel_params!(x_ptr, codes_ptr, scale_ptr, sum_ptr, rows, n);
    let cfg = LaunchConfig {
        grid: (1, 1, 1),
        block: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { function.launch(&cfg, &mut params, None) }.expect("launch failed");

    let mut got_scale = vec![0f32; 1];
    buf_scale.copy_to_host(&mut got_scale).expect("copy scale");
    assert_eq!(got_scale[0], 1.0);
}
