//! GPU integration tests for the Gated Delta Net decode-step kernels:
//! `causal_conv1d_decode_f32`, `gdn_gate_f32`, `gdn_recurrence_decode_f32`.
use std::ffi::c_void;

use rocml_hip::{kernel_params, Device, DeviceBuffer, LaunchConfig, Module};

const TOL: f32 = 1e-4;

fn assert_close(actual: &[f32], expected: &[f32], label: &str) {
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

// ── causal_conv1d_decode_f32 ────────────────────────────────────────────

fn run_conv(channels: u32, kernel_size: u32) {
    let _device = Device::new(0).expect("failed to select device 0");
    let (_m, function) = load(
        rocml_kernels::GDN_CONV1D_DECODE_F32_HSACO,
        rocml_kernels::GDN_CONV1D_DECODE_F32_KERNEL,
    );

    let hist_len = (kernel_size - 1) as usize;
    let x_new: Vec<f32> = (0..channels)
        .map(|i| ((i % 13) as f32) * 0.1 - 0.6)
        .collect();
    let conv_state: Vec<f32> = (0..channels as usize * hist_len)
        .map(|i| ((i % 11) as f32) * 0.05 - 0.25)
        .collect();
    let weight: Vec<f32> = (0..channels as usize * kernel_size as usize)
        .map(|i| ((i % 7) as f32) * 0.1 - 0.3)
        .collect();

    // CPU reference: mirrors the kernel exactly (oldest tap/history first,
    // SiLU on the raw accumulator, then shift-and-append).
    let mut expected_out = vec![0.0f32; channels as usize];
    let mut expected_state = conv_state.clone();
    for c in 0..channels as usize {
        let hist = &conv_state[c * hist_len..(c + 1) * hist_len];
        let w = &weight[c * kernel_size as usize..(c + 1) * kernel_size as usize];
        let mut acc = 0.0f32;
        for j in 0..hist_len {
            acc += hist[j] * w[j];
        }
        acc += x_new[c] * w[hist_len];
        expected_out[c] = acc / (1.0 + (-acc).exp());

        let new_hist = &mut expected_state[c * hist_len..(c + 1) * hist_len];
        for j in 0..hist_len.saturating_sub(1) {
            new_hist[j] = new_hist[j + 1];
        }
        if hist_len > 0 {
            new_hist[hist_len - 1] = x_new[c];
        }
    }

    let mut buf_x = DeviceBuffer::<f32>::new(x_new.len()).expect("hipMalloc x_new failed");
    let mut buf_state =
        DeviceBuffer::<f32>::new(conv_state.len()).expect("hipMalloc conv_state failed");
    let mut buf_w = DeviceBuffer::<f32>::new(weight.len()).expect("hipMalloc weight failed");
    let buf_out = DeviceBuffer::<f32>::new(channels as usize).expect("hipMalloc out failed");
    buf_x.copy_from_host(&x_new).expect("copy x_new failed");
    buf_state
        .copy_from_host(&conv_state)
        .expect("copy conv_state failed");
    buf_w.copy_from_host(&weight).expect("copy weight failed");

    let x_ptr: *mut c_void = buf_x.device_ptr();
    let state_ptr: *mut c_void = buf_state.device_ptr();
    let w_ptr: *mut c_void = buf_w.device_ptr();
    let out_ptr: *mut c_void = buf_out.device_ptr();
    let mut params = kernel_params!(x_ptr, state_ptr, w_ptr, out_ptr, channels, kernel_size);

    let block = 64u32;
    let cfg = LaunchConfig {
        grid: (channels.div_ceil(block), 1, 1),
        block: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: params matches causal_conv1d_decode_f32's parameter list
    // (const float*, float*, const float*, float*, unsigned, unsigned) in
    // order; all device buffers outlive this launch.
    unsafe { function.launch(&cfg, &mut params, None) }.expect("kernel launch failed");

    let mut actual_out = vec![0.0f32; channels as usize];
    buf_out
        .copy_to_host(&mut actual_out)
        .expect("copy out failed");
    assert_close(&actual_out, &expected_out, "conv1d out");

    let mut actual_state = vec![0.0f32; conv_state.len()];
    buf_state
        .copy_to_host(&mut actual_state)
        .expect("copy conv_state back failed");
    assert_close(&actual_state, &expected_state, "conv1d state");
}

#[test]
fn conv1d_non_multiple_of_blocksize() {
    // channels = 100 is not a multiple of the 64-thread block.
    run_conv(100, 4);
}

#[test]
fn conv1d_degenerate_kernel_size_one() {
    // kernel_size = 1: zero history slots, no shift, pure pointwise + SiLU.
    run_conv(5, 1);
}

#[test]
fn conv1d_degenerate_single_channel() {
    run_conv(1, 4);
}

// ── gdn_gate_f32 ─────────────────────────────────────────────────────────

fn run_gate(n: u32) {
    let _device = Device::new(0).expect("failed to select device 0");
    let (_m, function) = load(
        rocml_kernels::GDN_GATE_F32_HSACO,
        rocml_kernels::GDN_GATE_F32_KERNEL,
    );

    let a_raw: Vec<f32> = (0..n).map(|i| ((i % 9) as f32) * 0.2 - 0.9).collect();
    let b_raw: Vec<f32> = (0..n).map(|i| ((i % 7) as f32) * 0.3 - 1.0).collect();
    let a_log: Vec<f32> = (0..n).map(|i| ((i % 5) as f32) * 0.15 - 0.4).collect();
    let dt_bias: Vec<f32> = (0..n).map(|i| ((i % 3) as f32) * 0.1).collect();

    let expected_beta: Vec<f32> = b_raw.iter().map(|&b| 1.0 / (1.0 + (-b).exp())).collect();
    let expected_g: Vec<f32> = (0..n as usize)
        .map(|i| -a_log[i].exp() * (1.0 + (a_raw[i] + dt_bias[i]).exp()).ln())
        .collect();

    let mut buf_a = DeviceBuffer::<f32>::new(n as usize).expect("hipMalloc a_raw failed");
    let mut buf_b = DeviceBuffer::<f32>::new(n as usize).expect("hipMalloc b_raw failed");
    let mut buf_alog = DeviceBuffer::<f32>::new(n as usize).expect("hipMalloc a_log failed");
    let mut buf_dt = DeviceBuffer::<f32>::new(n as usize).expect("hipMalloc dt_bias failed");
    let buf_beta = DeviceBuffer::<f32>::new(n as usize).expect("hipMalloc beta failed");
    let buf_g = DeviceBuffer::<f32>::new(n as usize).expect("hipMalloc g failed");
    buf_a.copy_from_host(&a_raw).expect("copy a_raw failed");
    buf_b.copy_from_host(&b_raw).expect("copy b_raw failed");
    buf_alog.copy_from_host(&a_log).expect("copy a_log failed");
    buf_dt
        .copy_from_host(&dt_bias)
        .expect("copy dt_bias failed");

    let a_ptr: *mut c_void = buf_a.device_ptr();
    let b_ptr: *mut c_void = buf_b.device_ptr();
    let alog_ptr: *mut c_void = buf_alog.device_ptr();
    let dt_ptr: *mut c_void = buf_dt.device_ptr();
    let beta_ptr: *mut c_void = buf_beta.device_ptr();
    let g_ptr: *mut c_void = buf_g.device_ptr();
    let mut params = kernel_params!(a_ptr, b_ptr, alog_ptr, dt_ptr, beta_ptr, g_ptr, n);

    let block = 32u32;
    let cfg = LaunchConfig {
        grid: (n.div_ceil(block), 1, 1),
        block: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: params matches gdn_gate_f32's parameter list (four const
    // float*, two float*, unsigned) in order; buffers outlive this launch.
    unsafe { function.launch(&cfg, &mut params, None) }.expect("kernel launch failed");

    let mut actual_beta = vec![0.0f32; n as usize];
    let mut actual_g = vec![0.0f32; n as usize];
    buf_beta
        .copy_to_host(&mut actual_beta)
        .expect("copy beta failed");
    buf_g.copy_to_host(&mut actual_g).expect("copy g failed");
    assert_close(&actual_beta, &expected_beta, "gate beta");
    assert_close(&actual_g, &expected_g, "gate g");
}

#[test]
fn gate_non_multiple_of_blocksize() {
    run_gate(37);
}

#[test]
fn gate_degenerate_single_head() {
    run_gate(1);
}

// ── gdn_recurrence_decode_f32 ────────────────────────────────────────────

fn run_recurrence(num_heads: u32, num_k_heads: u32, head_k_dim: u32, head_v_dim: u32) {
    let _device = Device::new(0).expect("failed to select device 0");
    let (_m, function) = load(
        rocml_kernels::GDN_RECURRENCE_DECODE_F32_HSACO,
        rocml_kernels::GDN_RECURRENCE_DECODE_F32_KERNEL,
    );

    let (h, hk, kd, vd) = (
        num_heads as usize,
        num_k_heads as usize,
        head_k_dim as usize,
        head_v_dim as usize,
    );
    let state: Vec<f32> = (0..h * kd * vd)
        .map(|i| ((i % 17) as f32) * 0.05 - 0.4)
        .collect();
    let q: Vec<f32> = (0..hk * kd)
        .map(|i| ((i % 13) as f32) * 0.1 - 0.6)
        .collect();
    let k: Vec<f32> = (0..hk * kd)
        .map(|i| ((i % 11) as f32) * 0.1 - 0.5)
        .collect();
    let v: Vec<f32> = (0..h * vd).map(|i| ((i % 9) as f32) * 0.1 - 0.4).collect();
    let beta: Vec<f32> = (0..h).map(|i| 0.2 + (i as f32) * 0.1).collect();
    let g: Vec<f32> = (0..h).map(|i| -0.1 - (i as f32) * 0.05).collect();

    // CPU reference: per value head, decay -> kv_mem (from decayed state) ->
    // delta -> write S += outer(k, delta) -> y = S^T . q (from updated
    // state), with q/k taken from key head `head % num_k_heads` — a tiled
    // broadcast matching llama.cpp's GGUF V-head reorder (see the kernel's
    // doc comment). Mirrors Crane's portable `gated_delta_rule_recurrence`
    // for one step in the `num_k_heads == num_heads` case.
    let mut expected_state = state.clone();
    let mut expected_y = vec![0.0f32; h * vd];
    for head in 0..h {
        let key_head = head % hk;
        let decay = g[head].exp();
        let s = &mut expected_state[head * kd * vd..(head + 1) * kd * vd];
        for e in s.iter_mut() {
            *e *= decay;
        }
        let q_h = &q[key_head * kd..(key_head + 1) * kd];
        let k_h = &k[key_head * kd..(key_head + 1) * kd];
        let v_h = &v[head * vd..(head + 1) * vd];
        let mut delta = vec![0.0f32; vd];
        for (vi, delta_v) in delta.iter_mut().enumerate() {
            let kv_mem: f32 = (0..kd).map(|ki| s[ki * vd + vi] * k_h[ki]).sum();
            *delta_v = beta[head] * (v_h[vi] - kv_mem);
        }
        for ki in 0..kd {
            for vi in 0..vd {
                s[ki * vd + vi] += k_h[ki] * delta[vi];
            }
        }
        let y_h = &mut expected_y[head * vd..(head + 1) * vd];
        for (vi, y_v) in y_h.iter_mut().enumerate() {
            *y_v = (0..kd).map(|ki| s[ki * vd + vi] * q_h[ki]).sum();
        }
    }

    let mut buf_state = DeviceBuffer::<f32>::new(state.len()).expect("hipMalloc state failed");
    let mut buf_q = DeviceBuffer::<f32>::new(q.len()).expect("hipMalloc q failed");
    let mut buf_k = DeviceBuffer::<f32>::new(k.len()).expect("hipMalloc k failed");
    let mut buf_v = DeviceBuffer::<f32>::new(v.len()).expect("hipMalloc v failed");
    let mut buf_beta = DeviceBuffer::<f32>::new(beta.len()).expect("hipMalloc beta failed");
    let mut buf_g = DeviceBuffer::<f32>::new(g.len()).expect("hipMalloc g failed");
    let buf_y = DeviceBuffer::<f32>::new(h * vd).expect("hipMalloc y failed");
    buf_state.copy_from_host(&state).expect("copy state failed");
    buf_q.copy_from_host(&q).expect("copy q failed");
    buf_k.copy_from_host(&k).expect("copy k failed");
    buf_v.copy_from_host(&v).expect("copy v failed");
    buf_beta.copy_from_host(&beta).expect("copy beta failed");
    buf_g.copy_from_host(&g).expect("copy g failed");

    let state_ptr: *mut c_void = buf_state.device_ptr();
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
    // SAFETY: params matches gdn_recurrence_decode_f32's parameter list
    // (float*, four const float*, const float*, float*, four unsigned) in
    // order; block size is max(head_k_dim, head_v_dim) as the kernel
    // requires; buffers outlive this launch.
    unsafe { function.launch(&cfg, &mut params, None) }.expect("kernel launch failed");

    let mut actual_y = vec![0.0f32; h * vd];
    buf_y.copy_to_host(&mut actual_y).expect("copy y failed");
    assert_close(&actual_y, &expected_y, "recurrence y");

    let mut actual_state = vec![0.0f32; state.len()];
    buf_state
        .copy_to_host(&mut actual_state)
        .expect("copy state back failed");
    assert_close(&actual_state, &expected_state, "recurrence state");
}

#[test]
fn recurrence_square_heads() {
    run_recurrence(3, 3, 8, 8);
}

#[test]
fn recurrence_asymmetric_key_value_dims() {
    // head_k_dim != head_v_dim, and neither is a power of two.
    run_recurrence(2, 2, 5, 7);
}

#[test]
fn recurrence_degenerate_single_head_single_dim() {
    run_recurrence(1, 1, 1, 1);
}

#[test]
fn recurrence_grouped_key_heads() {
    // num_k_heads < num_heads: tiled broadcast, so value heads {0,2} share
    // key head 0 and {1,3} share key head 1 (`head % num_k_heads`) — the
    // Ornith-1.0-9B-shaped case (16 key / 32 value).
    run_recurrence(4, 2, 5, 7);
}
