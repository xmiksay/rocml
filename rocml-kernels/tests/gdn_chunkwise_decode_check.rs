//! Second, independent ground truth for the chunkwise (blocked delta-rule)
//! Gated Delta Net recurrence pipeline (`kernels/gdn_chunkwise.hip`):
//! `gdn_chunkwise.rs` checks it against an f64 CPU reference, this file
//! checks it against the actual GPU `gdn_recurrence_decode_f32` kernel run
//! `chunk_len` times sequentially (the same ground truth `gdn_chunk.rs`'s
//! old token-serial chunk-kernel test used) — catches a bug that happens to
//! also fool the f64 reference (e.g. a shared misreading of the decode
//! kernel's own math).
#[path = "gdn_chunkwise_support/mod.rs"]
mod support;

use std::ffi::c_void;

use rocml_hip::{kernel_params, Device, DeviceBuffer, LaunchConfig};
use support::{assert_close, load, run_chunkwise_gpu, ChunkwiseKernels, L2_EPS};

fn l2_norm_f32(raw: &[f32], extra_scale: f32) -> Vec<f32> {
    let sum_sq: f32 = raw.iter().map(|v| v * v).sum();
    let inv = (sum_sq + L2_EPS as f32).sqrt().recip() * extra_scale;
    raw.iter().map(|v| v * inv).collect()
}

#[test]
fn chunkwise_matches_sequential_decode_kernel() {
    let _device = Device::new(0).expect("failed to select device 0");
    let (_md, decode_fn) = load(
        rocml_kernels::GDN_RECURRENCE_DECODE_F32_HSACO,
        rocml_kernels::GDN_RECURRENCE_DECODE_F32_KERNEL,
    );
    let cw = ChunkwiseKernels::load();

    let (num_heads, num_k_heads, head_k_dim, head_v_dim, chunk_len) =
        (4u32, 2u32, 16u32, 16u32, 11u32);
    let (h, hk, sk, sv, t) = (
        num_heads as usize,
        num_k_heads as usize,
        head_k_dim as usize,
        head_v_dim as usize,
        chunk_len as usize,
    );
    let key_dim = hk * sk;
    let value_dim = h * sv;
    let conv_dim = 2 * key_dim + value_dim;
    let q_scale = 1.0f32 / (head_k_dim as f32).sqrt();

    let conv_out: Vec<f32> = (0..t * conv_dim)
        .map(|i| ((i % 19) as f32) * 0.06 - 0.55)
        .collect();
    let beta: Vec<f32> = (0..t * h).map(|i| 0.12 + (i as f32 % 6.0) * 0.08).collect();
    let g: Vec<f32> = (0..t * h)
        .map(|i| -0.03 - (i as f32 % 8.0) * 0.02)
        .collect();
    let init_state: Vec<f32> = (0..h * sk * sv)
        .map(|i| ((i % 17) as f32) * 0.05 - 0.4)
        .collect();

    // Reference: chunk_len sequential decode-kernel launches.
    let mut buf_state_seq = DeviceBuffer::<f32>::new(init_state.len()).unwrap();
    buf_state_seq.copy_from_host(&init_state).unwrap();
    let mut expected_y = vec![0.0f32; t * value_dim];
    for step in 0..t {
        let row = &conv_out[step * conv_dim..(step + 1) * conv_dim];
        let mut q_normed = vec![0.0f32; key_dim];
        let mut k_normed = vec![0.0f32; key_dim];
        for kh in 0..hk {
            let q_raw = &row[kh * sk..(kh + 1) * sk];
            let k_raw = &row[key_dim + kh * sk..key_dim + (kh + 1) * sk];
            q_normed[kh * sk..(kh + 1) * sk].copy_from_slice(&l2_norm_f32(q_raw, q_scale));
            k_normed[kh * sk..(kh + 1) * sk].copy_from_slice(&l2_norm_f32(k_raw, 1.0));
        }
        let v_row = &row[2 * key_dim..2 * key_dim + value_dim];
        let beta_t = &beta[step * h..(step + 1) * h];
        let g_t = &g[step * h..(step + 1) * h];

        let mut buf_q = DeviceBuffer::<f32>::new(key_dim).unwrap();
        let mut buf_k = DeviceBuffer::<f32>::new(key_dim).unwrap();
        let mut buf_v = DeviceBuffer::<f32>::new(value_dim).unwrap();
        let mut buf_beta = DeviceBuffer::<f32>::new(h).unwrap();
        let mut buf_g = DeviceBuffer::<f32>::new(h).unwrap();
        let buf_y = DeviceBuffer::<f32>::new(value_dim).unwrap();
        buf_q.copy_from_host(&q_normed).unwrap();
        buf_k.copy_from_host(&k_normed).unwrap();
        buf_v.copy_from_host(v_row).unwrap();
        buf_beta.copy_from_host(beta_t).unwrap();
        buf_g.copy_from_host(g_t).unwrap();

        let state_ptr: *mut c_void = buf_state_seq.device_ptr();
        let q_ptr: *mut c_void = buf_q.device_ptr();
        let k_ptr: *mut c_void = buf_k.device_ptr();
        let v_ptr: *mut c_void = buf_v.device_ptr();
        let beta_ptr: *mut c_void = buf_beta.device_ptr();
        let g_ptr: *mut c_void = buf_g.device_ptr();
        let y_ptr: *mut c_void = buf_y.device_ptr();
        let block = head_k_dim.max(head_v_dim);
        let cfg = LaunchConfig {
            grid: (num_heads, 1, 1),
            block: (block, 1, 1),
            shared_mem_bytes: 2 * (head_k_dim + head_v_dim) * 4,
        };
        let mut params = kernel_params!(
            state_ptr,
            q_ptr,
            k_ptr,
            v_ptr,
            beta_ptr,
            g_ptr,
            y_ptr,
            num_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim
        );
        unsafe { decode_fn.launch(&cfg, &mut params, None) }.expect("decode launch failed");
        buf_y
            .copy_to_host(&mut expected_y[step * value_dim..(step + 1) * value_dim])
            .unwrap();
    }
    let mut expected_state = vec![0.0f32; init_state.len()];
    buf_state_seq.copy_to_host(&mut expected_state).unwrap();

    let (actual_y, actual_state) = run_chunkwise_gpu(
        &cw,
        num_heads,
        num_k_heads,
        head_k_dim,
        head_v_dim,
        chunk_len,
        &conv_out,
        &beta,
        &g,
        &init_state,
    );

    let expected_y_f64: Vec<f64> = expected_y.iter().map(|&v| v as f64).collect();
    let expected_state_f64: Vec<f64> = expected_state.iter().map(|&v| v as f64).collect();
    assert_close(&actual_y, &expected_y_f64, "y vs sequential decode");
    assert_close(
        &actual_state,
        &expected_state_f64,
        "state vs sequential decode",
    );
}
