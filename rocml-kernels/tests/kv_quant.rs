//! GPU integration tests for the quantize-on-evict kernels
//! (`kernels/kv_quant.hip`) against a CPU reference. The reference here is
//! a standalone copy of `rocml::kv_quant::quant_math`'s logic (not an
//! import — `rocml` depends on `rocml-kernels`, not the reverse, and this
//! crate's existing kernel tests already follow the same
//! write-your-own-`cpu_reference` convention rather than a cross-crate
//! test-only dependency). Any drift between the two must be caught by
//! `rocml`'s own `kv_quant::quant_math` unit tests plus this file agreeing
//! on the exact same measured numbers for the same synthetic input.
use half::f16;
use rocml_hip::{kernel_params, Device, DeviceBuffer, LaunchConfig, Module};

fn load(hsaco: &[u8], name: &str) -> (Module, rocml_hip::Function) {
    let module = Module::load_from_bytes(hsaco).expect("module load failed");
    let function = module.get_function(name).expect("kernel lookup failed");
    (module, function)
}

fn synthetic_window(n_kv_heads: usize, window_len: usize, head_dim: usize, seed: u32) -> Vec<f32> {
    (0..n_kv_heads * window_len * head_dim)
        .map(|i| {
            let x = (i as u32).wrapping_mul(2654435761).wrapping_add(seed);
            ((x >> 8) as f32 / u32::MAX as f32 - 0.5) * 4.0
        })
        .collect()
}

fn cpu_quantize_k_per_channel(
    block: &[f32],
    n_kv_heads: usize,
    window_len: usize,
    head_dim: usize,
) -> (Vec<i8>, Vec<f32>) {
    let mut codes = vec![0i8; block.len()];
    let mut scales = vec![0f32; n_kv_heads * head_dim];
    for h in 0..n_kv_heads {
        for d in 0..head_dim {
            let mut max_abs = 0f32;
            for t in 0..window_len {
                max_abs = max_abs.max(block[(h * window_len + t) * head_dim + d].abs());
            }
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            scales[h * head_dim + d] = scale;
            for t in 0..window_len {
                let idx = (h * window_len + t) * head_dim + d;
                codes[idx] = (block[idx] / scale).round().clamp(-127.0, 127.0) as i8;
            }
        }
    }
    (codes, scales)
}

fn cpu_quantize_v_per_token_q8(
    block: &[f32],
    n_kv_heads: usize,
    window_len: usize,
    head_dim: usize,
) -> (Vec<i8>, Vec<f32>) {
    let mut codes = vec![0i8; block.len()];
    let mut scales = vec![0f32; n_kv_heads * window_len];
    for h in 0..n_kv_heads {
        for t in 0..window_len {
            let row = &block[(h * window_len + t) * head_dim..(h * window_len + t + 1) * head_dim];
            let max_abs = row.iter().fold(0f32, |m, &v| m.max(v.abs()));
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            scales[h * window_len + t] = scale;
            for d in 0..head_dim {
                let idx = (h * window_len + t) * head_dim + d;
                codes[idx] = (block[idx] / scale).round().clamp(-127.0, 127.0) as i8;
            }
        }
    }
    (codes, scales)
}

fn cpu_quantize_v_per_token_q4(
    block: &[f32],
    n_kv_heads: usize,
    window_len: usize,
    head_dim: usize,
) -> (Vec<u8>, Vec<f32>) {
    let mut packed = vec![0u8; n_kv_heads * window_len * (head_dim / 2)];
    let mut scales = vec![0f32; n_kv_heads * window_len];
    for h in 0..n_kv_heads {
        for t in 0..window_len {
            let row = &block[(h * window_len + t) * head_dim..(h * window_len + t + 1) * head_dim];
            let max_abs = row.iter().fold(0f32, |m, &v| m.max(v.abs()));
            let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
            scales[h * window_len + t] = scale;
            for pair in 0..head_dim / 2 {
                let lo = (row[2 * pair] / scale).round().clamp(-8.0, 7.0) as i8;
                let hi = (row[2 * pair + 1] / scale).round().clamp(-8.0, 7.0) as i8;
                let idx = (h * window_len + t) * (head_dim / 2) + pair;
                packed[idx] = ((lo as u8) & 0x0F) | (((hi as u8) & 0x0F) << 4);
            }
        }
    }
    (packed, scales)
}

#[test]
fn quantize_evict_k_matches_cpu_reference() {
    let _device = Device::new(0).expect("failed to select device 0");
    let (_m, function) = load(
        rocml_kernels::QUANTIZE_EVICT_K_Q8_HSACO,
        rocml_kernels::QUANTIZE_EVICT_K_Q8_KERNEL,
    );

    let (n_kv_heads, window_len, head_dim): (usize, usize, usize) = (4, 128, 256);
    let block = synthetic_window(n_kv_heads, window_len, head_dim, 11);
    let block_f16: Vec<f16> = block.iter().map(|&x| f16::from_f32(x)).collect();
    let block_roundtrip: Vec<f32> = block_f16.iter().map(|&x| x.to_f32()).collect();
    let (expected_codes, expected_scales) =
        cpu_quantize_k_per_channel(&block_roundtrip, n_kv_heads, window_len, head_dim);

    let bulk_cap = window_len * 3; // room for a few blocks, only block 1 written
    let num_blocks_total = 3u32;
    let block_idx = 1u32;

    let mut buf_window = DeviceBuffer::<f16>::new(block_f16.len()).expect("hipMalloc window");
    buf_window.copy_from_host(&block_f16).expect("copy window");
    let mut buf_codes =
        DeviceBuffer::<i8>::new(n_kv_heads * bulk_cap * head_dim).expect("hipMalloc codes");
    buf_codes
        .copy_from_host(&vec![0i8; n_kv_heads * bulk_cap * head_dim])
        .expect("zero codes");
    let mut buf_scales =
        DeviceBuffer::<f32>::new(n_kv_heads * num_blocks_total as usize * head_dim)
            .expect("hipMalloc scales");
    buf_scales
        .copy_from_host(&vec![
            0f32;
            n_kv_heads * num_blocks_total as usize * head_dim
        ])
        .expect("zero scales");

    let window_ptr = buf_window.device_ptr();
    let codes_ptr = buf_codes.device_ptr();
    let scales_ptr = buf_scales.device_ptr();
    let n_kv_heads_u32 = n_kv_heads as u32;
    let window_len_u32 = window_len as u32;
    let head_dim_u32 = head_dim as u32;
    let bulk_cap_u32 = bulk_cap as u32;
    let mut params = kernel_params!(
        window_ptr,
        codes_ptr,
        scales_ptr,
        n_kv_heads_u32,
        window_len_u32,
        head_dim_u32,
        bulk_cap_u32,
        num_blocks_total,
        block_idx
    );
    let cfg = LaunchConfig {
        grid: (n_kv_heads as u32, 1, 1),
        block: (head_dim as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { function.launch(&cfg, &mut params, None) }.expect("launch failed");

    let mut actual_codes = vec![0i8; n_kv_heads * bulk_cap * head_dim];
    buf_codes
        .copy_to_host(&mut actual_codes)
        .expect("copy codes");
    let mut actual_scales = vec![0f32; n_kv_heads * num_blocks_total as usize * head_dim];
    buf_scales
        .copy_to_host(&mut actual_scales)
        .expect("copy scales");

    for h in 0..n_kv_heads {
        for d in 0..head_dim {
            let want_scale = expected_scales[h * head_dim + d];
            let got_scale =
                actual_scales[(h * num_blocks_total as usize + block_idx as usize) * head_dim + d];
            assert!(
                (got_scale - want_scale).abs() <= 1e-6 * want_scale.abs().max(1.0),
                "scale[{h},{d}]: got {got_scale} want {want_scale}"
            );
            for t in 0..window_len {
                let want = expected_codes[(h * window_len + t) * head_dim + d];
                let got = actual_codes[h * bulk_cap * head_dim
                    + (block_idx as usize * window_len + t) * head_dim
                    + d];
                assert_eq!(got, want, "code[{h},{t},{d}]");
            }
        }
    }
}

#[test]
fn quantize_evict_v_q8_matches_cpu_reference() {
    let _device = Device::new(0).expect("failed to select device 0");
    let (_m, function) = load(
        rocml_kernels::QUANTIZE_EVICT_V_Q8_HSACO,
        rocml_kernels::QUANTIZE_EVICT_V_Q8_KERNEL,
    );

    let (n_kv_heads, window_len, head_dim): (usize, usize, usize) = (4, 128, 256);
    let block = synthetic_window(n_kv_heads, window_len, head_dim, 22);
    let block_f16: Vec<f16> = block.iter().map(|&x| f16::from_f32(x)).collect();
    let block_roundtrip: Vec<f32> = block_f16.iter().map(|&x| x.to_f32()).collect();
    let (expected_codes, expected_scales) =
        cpu_quantize_v_per_token_q8(&block_roundtrip, n_kv_heads, window_len, head_dim);

    let bulk_cap = window_len * 2;
    let block_idx = 1u32;

    let mut buf_window = DeviceBuffer::<f16>::new(block_f16.len()).expect("hipMalloc window");
    buf_window.copy_from_host(&block_f16).expect("copy window");
    let mut buf_codes =
        DeviceBuffer::<i8>::new(n_kv_heads * bulk_cap * head_dim).expect("hipMalloc codes");
    buf_codes
        .copy_from_host(&vec![0i8; n_kv_heads * bulk_cap * head_dim])
        .expect("zero codes");
    let mut buf_scales = DeviceBuffer::<f32>::new(n_kv_heads * bulk_cap).expect("hipMalloc scales");
    buf_scales
        .copy_from_host(&vec![0f32; n_kv_heads * bulk_cap])
        .expect("zero scales");

    let window_ptr = buf_window.device_ptr();
    let codes_ptr = buf_codes.device_ptr();
    let scales_ptr = buf_scales.device_ptr();
    let n_kv_heads_u32 = n_kv_heads as u32;
    let window_len_u32 = window_len as u32;
    let head_dim_u32 = head_dim as u32;
    let bulk_cap_u32 = bulk_cap as u32;
    let mut params = kernel_params!(
        window_ptr,
        codes_ptr,
        scales_ptr,
        n_kv_heads_u32,
        window_len_u32,
        head_dim_u32,
        bulk_cap_u32,
        block_idx
    );
    let cfg = LaunchConfig {
        grid: (n_kv_heads as u32, window_len as u32, 1),
        block: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { function.launch(&cfg, &mut params, None) }.expect("launch failed");

    let mut actual_codes = vec![0i8; n_kv_heads * bulk_cap * head_dim];
    buf_codes
        .copy_to_host(&mut actual_codes)
        .expect("copy codes");
    let mut actual_scales = vec![0f32; n_kv_heads * bulk_cap];
    buf_scales
        .copy_to_host(&mut actual_scales)
        .expect("copy scales");

    for h in 0..n_kv_heads {
        for t in 0..window_len {
            let want_scale = expected_scales[h * window_len + t];
            let got_scale = actual_scales[h * bulk_cap + block_idx as usize * window_len + t];
            assert!(
                (got_scale - want_scale).abs() <= 1e-6 * want_scale.abs().max(1.0),
                "scale[{h},{t}]: got {got_scale} want {want_scale}"
            );
            for d in 0..head_dim {
                let want = expected_codes[(h * window_len + t) * head_dim + d];
                let got = actual_codes[h * bulk_cap * head_dim
                    + (block_idx as usize * window_len + t) * head_dim
                    + d];
                assert_eq!(got, want, "code[{h},{t},{d}]");
            }
        }
    }
}

#[test]
fn quantize_evict_v_q4_matches_cpu_reference() {
    let _device = Device::new(0).expect("failed to select device 0");
    let (_m, function) = load(
        rocml_kernels::QUANTIZE_EVICT_V_Q4_HSACO,
        rocml_kernels::QUANTIZE_EVICT_V_Q4_KERNEL,
    );

    let (n_kv_heads, window_len, head_dim): (usize, usize, usize) = (4, 128, 256);
    let block = synthetic_window(n_kv_heads, window_len, head_dim, 33);
    let block_f16: Vec<f16> = block.iter().map(|&x| f16::from_f32(x)).collect();
    let block_roundtrip: Vec<f32> = block_f16.iter().map(|&x| x.to_f32()).collect();
    let (expected_packed, expected_scales) =
        cpu_quantize_v_per_token_q4(&block_roundtrip, n_kv_heads, window_len, head_dim);

    let bulk_cap = window_len * 2;
    let block_idx = 1u32;
    let half_dim = head_dim / 2;

    let mut buf_window = DeviceBuffer::<f16>::new(block_f16.len()).expect("hipMalloc window");
    buf_window.copy_from_host(&block_f16).expect("copy window");
    let mut buf_packed =
        DeviceBuffer::<u8>::new(n_kv_heads * bulk_cap * half_dim).expect("hipMalloc packed");
    buf_packed
        .copy_from_host(&vec![0u8; n_kv_heads * bulk_cap * half_dim])
        .expect("zero packed");
    let mut buf_scales = DeviceBuffer::<f32>::new(n_kv_heads * bulk_cap).expect("hipMalloc scales");
    buf_scales
        .copy_from_host(&vec![0f32; n_kv_heads * bulk_cap])
        .expect("zero scales");

    let window_ptr = buf_window.device_ptr();
    let packed_ptr = buf_packed.device_ptr();
    let scales_ptr = buf_scales.device_ptr();
    let n_kv_heads_u32 = n_kv_heads as u32;
    let window_len_u32 = window_len as u32;
    let head_dim_u32 = head_dim as u32;
    let bulk_cap_u32 = bulk_cap as u32;
    let mut params = kernel_params!(
        window_ptr,
        packed_ptr,
        scales_ptr,
        n_kv_heads_u32,
        window_len_u32,
        head_dim_u32,
        bulk_cap_u32,
        block_idx
    );
    let cfg = LaunchConfig {
        grid: (n_kv_heads as u32, window_len as u32, 1),
        block: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { function.launch(&cfg, &mut params, None) }.expect("launch failed");

    let mut actual_packed = vec![0u8; n_kv_heads * bulk_cap * half_dim];
    buf_packed
        .copy_to_host(&mut actual_packed)
        .expect("copy packed");
    let mut actual_scales = vec![0f32; n_kv_heads * bulk_cap];
    buf_scales
        .copy_to_host(&mut actual_scales)
        .expect("copy scales");

    for h in 0..n_kv_heads {
        for t in 0..window_len {
            let want_scale = expected_scales[h * window_len + t];
            let got_scale = actual_scales[h * bulk_cap + block_idx as usize * window_len + t];
            assert!(
                (got_scale - want_scale).abs() <= 1e-6 * want_scale.abs().max(1.0),
                "scale[{h},{t}]: got {got_scale} want {want_scale}"
            );
            for pair in 0..half_dim {
                let want = expected_packed[(h * window_len + t) * half_dim + pair];
                let got = actual_packed[h * bulk_cap * half_dim
                    + (block_idx as usize * window_len + t) * half_dim
                    + pair];
                assert_eq!(got, want, "packed[{h},{t},{pair}]");
            }
        }
    }
}
