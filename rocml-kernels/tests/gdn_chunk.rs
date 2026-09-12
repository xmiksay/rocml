//! GPU integration tests for the Gated Delta Net prefill-chunk kernels
//! (`causal_conv1d_chunk_f32`, `gdn_recurrence_chunk_f32`) against their
//! `_decode_f32` siblings run `chunk_len` times sequentially — the chunk
//! kernel must produce the *exact same state evolution* as T decode steps
//! (same math, just fewer launches), per issue #6's correctness gate.
use std::ffi::c_void;

use rocml_hip::{kernel_params, Device, DeviceBuffer, LaunchConfig, Module};

const TOL: f32 = 1e-5;
const L2_EPS: f32 = 1e-6;

fn assert_close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (got, want)) in actual.iter().zip(expected).enumerate() {
        let diff = (got - want).abs();
        assert!(
            diff <= TOL * want.abs().max(1.0),
            "{label}[{i}]: got {got}, want {want} (diff {diff})"
        );
    }
}

fn load(hsaco: &[u8], name: &str) -> (Module, rocml_hip::Function) {
    let module = Module::load_from_bytes(hsaco).expect("module load failed");
    let function = module.get_function(name).expect("kernel lookup failed");
    (module, function)
}

// ── causal_conv1d_chunk_f32 vs T x causal_conv1d_decode_f32 ────────────────

fn run_conv_chunk(channels: u32, kernel_size: u32, chunk_len: u32) {
    let _device = Device::new(0).expect("failed to select device 0");
    let (_md, decode_fn) = load(
        rocml_kernels::GDN_CONV1D_DECODE_F32_HSACO,
        rocml_kernels::GDN_CONV1D_DECODE_F32_KERNEL,
    );
    let (_mc, chunk_fn) = load(
        rocml_kernels::GDN_CONV1D_CHUNK_F32_HSACO,
        rocml_kernels::GDN_CONV1D_CHUNK_F32_KERNEL,
    );

    let hist_len = (kernel_size - 1) as usize;
    let cl = chunk_len as usize;
    let x: Vec<f32> = (0..cl * channels as usize)
        .map(|i| ((i % 13) as f32) * 0.1 - 0.6)
        .collect();
    let init_state: Vec<f32> = (0..channels as usize * hist_len)
        .map(|i| ((i % 11) as f32) * 0.05 - 0.25)
        .collect();
    let weight: Vec<f32> = (0..channels as usize * kernel_size as usize)
        .map(|i| ((i % 7) as f32) * 0.1 - 0.3)
        .collect();

    let mut buf_w = DeviceBuffer::<f32>::new(weight.len()).expect("hipMalloc weight failed");
    buf_w.copy_from_host(&weight).expect("copy weight failed");
    let w_ptr: *mut c_void = buf_w.device_ptr();

    // Reference: chunk_len sequential decode-kernel launches over the same
    // running conv_state, one token at a time.
    let mut buf_state_seq =
        DeviceBuffer::<f32>::new(init_state.len()).expect("hipMalloc state_seq failed");
    buf_state_seq
        .copy_from_host(&init_state)
        .expect("copy state_seq failed");
    let mut expected_out = vec![0.0f32; cl * channels as usize];
    for t in 0..cl {
        let mut buf_xt = DeviceBuffer::<f32>::new(channels as usize).expect("hipMalloc xt failed");
        buf_xt
            .copy_from_host(&x[t * channels as usize..(t + 1) * channels as usize])
            .expect("copy xt failed");
        let buf_out_t =
            DeviceBuffer::<f32>::new(channels as usize).expect("hipMalloc out_t failed");
        let x_ptr: *mut c_void = buf_xt.device_ptr();
        let state_ptr: *mut c_void = buf_state_seq.device_ptr();
        let out_ptr: *mut c_void = buf_out_t.device_ptr();
        let mut params = kernel_params!(x_ptr, state_ptr, w_ptr, out_ptr, channels, kernel_size);
        let block = 64u32;
        let cfg = LaunchConfig {
            grid: (channels.div_ceil(block), 1, 1),
            block: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: params matches causal_conv1d_decode_f32's parameter list;
        // buffers outlive this launch.
        unsafe { decode_fn.launch(&cfg, &mut params, None) }.expect("decode launch failed");
        buf_out_t
            .copy_to_host(&mut expected_out[t * channels as usize..(t + 1) * channels as usize])
            .expect("copy out_t failed");
    }
    let mut expected_state = vec![0.0f32; init_state.len()];
    buf_state_seq
        .copy_to_host(&mut expected_state)
        .expect("copy final state_seq failed");

    // Chunk kernel: one launch over the whole chunk from the same initial state.
    let mut buf_x = DeviceBuffer::<f32>::new(x.len()).expect("hipMalloc x failed");
    buf_x.copy_from_host(&x).expect("copy x failed");
    let mut buf_state_chunk =
        DeviceBuffer::<f32>::new(init_state.len()).expect("hipMalloc state_chunk failed");
    buf_state_chunk
        .copy_from_host(&init_state)
        .expect("copy state_chunk failed");
    let buf_out_chunk =
        DeviceBuffer::<f32>::new(cl * channels as usize).expect("hipMalloc out_chunk failed");

    let x_ptr: *mut c_void = buf_x.device_ptr();
    let state_ptr: *mut c_void = buf_state_chunk.device_ptr();
    let out_ptr: *mut c_void = buf_out_chunk.device_ptr();
    let mut params = kernel_params!(
        x_ptr,
        state_ptr,
        w_ptr,
        out_ptr,
        channels,
        kernel_size,
        chunk_len
    );
    let block = 64u32;
    let cfg = LaunchConfig {
        grid: (channels.div_ceil(block), 1, 1),
        block: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: params matches causal_conv1d_chunk_f32's parameter list (const
    // float*, float*, const float*, float*, unsigned x3); buffers outlive
    // this launch.
    unsafe { chunk_fn.launch(&cfg, &mut params, None) }.expect("chunk launch failed");

    let mut actual_out = vec![0.0f32; cl * channels as usize];
    buf_out_chunk
        .copy_to_host(&mut actual_out)
        .expect("copy out_chunk failed");
    assert_close(&actual_out, &expected_out, "conv chunk out");

    let mut actual_state = vec![0.0f32; init_state.len()];
    buf_state_chunk
        .copy_to_host(&mut actual_state)
        .expect("copy final state_chunk failed");
    assert_close(&actual_state, &expected_state, "conv chunk state");
}

#[test]
fn conv_chunk_matches_sequential_decode() {
    run_conv_chunk(37, 4, 9);
}

#[test]
fn conv_chunk_degenerate_single_token() {
    run_conv_chunk(16, 4, 1);
}

#[test]
fn conv_chunk_kernel_size_one() {
    run_conv_chunk(8, 1, 5);
}

// ── gdn_recurrence_chunk_f32 vs T x gdn_recurrence_decode_f32 ──────────────

/// L2-norms `raw` (sum-of-squares, `+ l2_eps`) and applies `extra_scale` —
/// exactly the math `gdn_recurrence_chunk_f32` does in-kernel, and what
/// `forward::gdn`'s decode path does via a separate `rmsnorm_f32` call
/// before invoking the decode kernel (see that module's doc comment).
fn l2_norm(raw: &[f32], extra_scale: f32) -> Vec<f32> {
    let sum_sq: f32 = raw.iter().map(|v| v * v).sum();
    let inv = (sum_sq + L2_EPS).sqrt().recip() * extra_scale;
    raw.iter().map(|v| v * inv).collect()
}

#[allow(clippy::too_many_arguments)]
fn run_recurrence_chunk(
    num_heads: u32,
    num_k_heads: u32,
    head_k_dim: u32,
    head_v_dim: u32,
    chunk_len: u32,
) {
    let _device = Device::new(0).expect("failed to select device 0");
    let (_md, decode_fn) = load(
        rocml_kernels::GDN_RECURRENCE_DECODE_F32_HSACO,
        rocml_kernels::GDN_RECURRENCE_DECODE_F32_KERNEL,
    );
    let (_mc, chunk_fn) = load(
        rocml_kernels::GDN_RECURRENCE_CHUNK_F32_HSACO,
        rocml_kernels::GDN_RECURRENCE_CHUNK_F32_KERNEL,
    );

    let (h, hk, kd, vd, cl) = (
        num_heads as usize,
        num_k_heads as usize,
        head_k_dim as usize,
        head_v_dim as usize,
        chunk_len as usize,
    );
    let key_dim = hk * kd;
    let value_dim = h * vd;
    let conv_dim = 2 * key_dim + value_dim;
    let q_scale = 1.0f32 / (head_k_dim as f32).sqrt();

    // conv_out row t: [Q(key_dim) | K(key_dim) | V(value_dim)], raw
    // (pre-L2-norm) — the chunk kernel's actual input contract.
    let conv_out: Vec<f32> = (0..cl * conv_dim)
        .map(|i| ((i % 23) as f32) * 0.07 - 0.7)
        .collect();
    let beta: Vec<f32> = (0..cl * h).map(|i| 0.15 + (i as f32 % 5.0) * 0.1).collect();
    let g: Vec<f32> = (0..cl * h)
        .map(|i| -0.05 - (i as f32 % 5.0) * 0.03)
        .collect();
    let init_state: Vec<f32> = (0..h * kd * vd)
        .map(|i| ((i % 17) as f32) * 0.05 - 0.4)
        .collect();

    // Reference: chunk_len sequential decode-kernel launches, each fed this
    // token's L2-normed Q/K (computed on the CPU with the exact formula the
    // chunk kernel applies in-kernel) and raw V.
    let mut buf_state_seq =
        DeviceBuffer::<f32>::new(init_state.len()).expect("hipMalloc state_seq failed");
    buf_state_seq
        .copy_from_host(&init_state)
        .expect("copy state_seq failed");
    let mut expected_y = vec![0.0f32; cl * value_dim];
    for t in 0..cl {
        let row = &conv_out[t * conv_dim..(t + 1) * conv_dim];
        let mut q_normed = vec![0.0f32; key_dim];
        let mut k_normed = vec![0.0f32; key_dim];
        for kh in 0..hk {
            let q_raw = &row[kh * kd..(kh + 1) * kd];
            let k_raw = &row[key_dim + kh * kd..key_dim + (kh + 1) * kd];
            q_normed[kh * kd..(kh + 1) * kd].copy_from_slice(&l2_norm(q_raw, q_scale));
            k_normed[kh * kd..(kh + 1) * kd].copy_from_slice(&l2_norm(k_raw, 1.0));
        }
        let v_row = &row[2 * key_dim..2 * key_dim + value_dim];
        let beta_t = &beta[t * h..(t + 1) * h];
        let g_t = &g[t * h..(t + 1) * h];

        let mut buf_q = DeviceBuffer::<f32>::new(key_dim).expect("hipMalloc q failed");
        let mut buf_k = DeviceBuffer::<f32>::new(key_dim).expect("hipMalloc k failed");
        let mut buf_v = DeviceBuffer::<f32>::new(value_dim).expect("hipMalloc v failed");
        let mut buf_beta = DeviceBuffer::<f32>::new(h).expect("hipMalloc beta failed");
        let mut buf_g = DeviceBuffer::<f32>::new(h).expect("hipMalloc g failed");
        let buf_y = DeviceBuffer::<f32>::new(value_dim).expect("hipMalloc y failed");
        buf_q.copy_from_host(&q_normed).expect("copy q failed");
        buf_k.copy_from_host(&k_normed).expect("copy k failed");
        buf_v.copy_from_host(v_row).expect("copy v failed");
        buf_beta.copy_from_host(beta_t).expect("copy beta failed");
        buf_g.copy_from_host(g_t).expect("copy g failed");

        let state_ptr: *mut c_void = buf_state_seq.device_ptr();
        let q_ptr: *mut c_void = buf_q.device_ptr();
        let k_ptr: *mut c_void = buf_k.device_ptr();
        let v_ptr: *mut c_void = buf_v.device_ptr();
        let beta_ptr: *mut c_void = buf_beta.device_ptr();
        let g_ptr: *mut c_void = buf_g.device_ptr();
        let y_ptr: *mut c_void = buf_y.device_ptr();
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
        let block = head_k_dim.max(head_v_dim);
        let cfg = LaunchConfig {
            grid: (num_heads, 1, 1),
            block: (block, 1, 1),
            shared_mem_bytes: 2 * (head_k_dim + head_v_dim) * std::mem::size_of::<f32>() as u32,
        };
        // SAFETY: params matches gdn_recurrence_decode_f32's parameter list;
        // block size is max(head_k_dim, head_v_dim); buffers outlive launch.
        unsafe { decode_fn.launch(&cfg, &mut params, None) }.expect("decode launch failed");
        buf_y
            .copy_to_host(&mut expected_y[t * value_dim..(t + 1) * value_dim])
            .expect("copy y failed");
    }
    let mut expected_state = vec![0.0f32; init_state.len()];
    buf_state_seq
        .copy_to_host(&mut expected_state)
        .expect("copy final state_seq failed");

    // Chunk kernel: one launch per head over the whole chunk's raw conv_out.
    let mut buf_conv = DeviceBuffer::<f32>::new(conv_out.len()).expect("hipMalloc conv_out failed");
    let mut buf_beta_all = DeviceBuffer::<f32>::new(beta.len()).expect("hipMalloc beta_all failed");
    let mut buf_g_all = DeviceBuffer::<f32>::new(g.len()).expect("hipMalloc g_all failed");
    let mut buf_state_chunk =
        DeviceBuffer::<f32>::new(init_state.len()).expect("hipMalloc state_chunk failed");
    let buf_y_chunk = DeviceBuffer::<f32>::new(cl * value_dim).expect("hipMalloc y_chunk failed");
    buf_conv
        .copy_from_host(&conv_out)
        .expect("copy conv_out failed");
    buf_beta_all
        .copy_from_host(&beta)
        .expect("copy beta_all failed");
    buf_g_all.copy_from_host(&g).expect("copy g_all failed");
    buf_state_chunk
        .copy_from_host(&init_state)
        .expect("copy state_chunk failed");

    let state_ptr: *mut c_void = buf_state_chunk.device_ptr();
    let conv_ptr: *mut c_void = buf_conv.device_ptr();
    let beta_ptr: *mut c_void = buf_beta_all.device_ptr();
    let g_ptr: *mut c_void = buf_g_all.device_ptr();
    let y_ptr: *mut c_void = buf_y_chunk.device_ptr();
    let conv_dim_u32 = conv_dim as u32;
    let key_dim_u32 = key_dim as u32;
    let l2_eps = L2_EPS;
    let mut params = kernel_params!(
        state_ptr,
        conv_ptr,
        beta_ptr,
        g_ptr,
        y_ptr,
        num_heads,
        num_k_heads,
        head_k_dim,
        head_v_dim,
        conv_dim_u32,
        key_dim_u32,
        chunk_len,
        l2_eps
    );
    let block = head_k_dim.max(head_v_dim);
    let cfg = LaunchConfig {
        grid: (num_heads, 1, 1),
        block: (block, 1, 1),
        shared_mem_bytes: 2 * (head_k_dim + head_v_dim) * std::mem::size_of::<f32>() as u32,
    };
    // SAFETY: params matches gdn_recurrence_chunk_f32's parameter list
    // (float*, const float* x3, float*, six unsigned, unsigned, float);
    // block size is max(head_k_dim, head_v_dim); buffers outlive this launch.
    unsafe { chunk_fn.launch(&cfg, &mut params, None) }.expect("chunk launch failed");

    let mut actual_y = vec![0.0f32; cl * value_dim];
    buf_y_chunk
        .copy_to_host(&mut actual_y)
        .expect("copy y_chunk failed");
    assert_close(&actual_y, &expected_y, "recurrence chunk y");

    let mut actual_state = vec![0.0f32; init_state.len()];
    buf_state_chunk
        .copy_to_host(&mut actual_state)
        .expect("copy final state_chunk failed");
    assert_close(&actual_state, &expected_state, "recurrence chunk state");
}

#[test]
fn recurrence_chunk_matches_sequential_decode() {
    run_recurrence_chunk(3, 3, 8, 8, 6);
}

#[test]
fn recurrence_chunk_grouped_key_heads() {
    // num_k_heads < num_heads, Ornith-1.0-9B-shaped tiled broadcast.
    run_recurrence_chunk(4, 2, 5, 7, 5);
}

#[test]
fn recurrence_chunk_degenerate_single_token() {
    run_recurrence_chunk(2, 2, 4, 4, 1);
}
