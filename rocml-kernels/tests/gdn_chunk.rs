//! GPU integration test for the Gated Delta Net prefill-chunk causal-conv1d
//! kernel (`causal_conv1d_chunk_f32`) against `causal_conv1d_decode_f32` run
//! `chunk_len` times sequentially — the chunk kernel must produce the
//! *exact same state evolution* as T decode steps (same math, just fewer
//! launches). The recurrence's chunk kernel used to be tested here the same
//! way (`gdn_recurrence_chunk_f32`, now removed) — its chunkwise
//! replacement is tested in `gdn_chunkwise.rs` against both an f64 CPU
//! reference and the same T-sequential-decode-launches ground truth.
use std::ffi::c_void;

use rocml_hip::{kernel_params, Device, DeviceBuffer, LaunchConfig, Module};

const TOL: f32 = 1e-5;

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
