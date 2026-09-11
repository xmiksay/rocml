//! Informational perf probe for the fused dequant-GEMV kernels — not a
//! correctness gate (see `gemv_q4_k.rs`/`gemv_q6_k.rs` for that), just a
//! manually-run measurement of the memory-bandwidth win these kernels exist
//! for. Run with `cargo test -p rocml-kernels --test gemv_quant_perf --
//! --ignored --nocapture`.
mod common;

use std::ffi::c_void;
use std::time::Instant;

use common::{random_row_q4_k, random_row_q6_k, Rng, Q4_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES};
use rocml_hip::{kernel_params, Device, DeviceBuffer, LaunchConfig, Module, Stream};

const M: u32 = 4096;
const N: u32 = 4096;
const WARMUP_ITERS: u32 = 10;
const TIMED_ITERS: u32 = 200;
const K_BLOCK_ELEMS: usize = 256;

/// Times `TIMED_ITERS` launches of `kernel_name` at the fixed M x N probe
/// shape and prints effective GB/s (weight bytes read per pass, ignoring the
/// comparatively tiny x/y traffic) and GFLOP/s (2 flops per element: one
/// multiply, one add).
fn perf_probe(hsaco: &[u8], kernel_name: &str, w_bytes: &[u8], x: &[f32], block_bytes: usize) {
    let _device = Device::new(0).expect("failed to select device 0");
    let module = Module::load_from_bytes(hsaco).expect("module load failed");
    let function = module
        .get_function(kernel_name)
        .expect("kernel lookup failed");
    let stream = Stream::new().expect("failed to create stream");

    let mut buf_w = DeviceBuffer::<u8>::new(w_bytes.len()).expect("hipMalloc w failed");
    let mut buf_x = DeviceBuffer::<f32>::new(x.len()).expect("hipMalloc x failed");
    let buf_y = DeviceBuffer::<f32>::new(M as usize).expect("hipMalloc y failed");
    buf_w.copy_from_host(w_bytes).expect("copy w failed");
    buf_x.copy_from_host(x).expect("copy x failed");

    let w_ptr: *mut c_void = buf_w.device_ptr();
    let x_ptr: *mut c_void = buf_x.device_ptr();
    let y_ptr: *mut c_void = buf_y.device_ptr();
    let m = M;
    let n = N;
    let mut params = kernel_params!(w_ptr, x_ptr, y_ptr, m, n);

    let block = 128u32;
    let cfg = LaunchConfig {
        grid: (M, 1, 1),
        block: (block, 1, 1),
        shared_mem_bytes: block * std::mem::size_of::<f32>() as u32,
    };

    for _ in 0..WARMUP_ITERS {
        // SAFETY: params matches gemv_<type>'s parameter list (const void*,
        // const float*, float*, unsigned, unsigned) in order, and all device
        // buffers outlive every launch in this loop.
        unsafe { function.launch(&cfg, &mut params, Some(&stream)) }.expect("launch failed");
    }
    stream.synchronize().expect("sync failed");

    let start = Instant::now();
    for _ in 0..TIMED_ITERS {
        // SAFETY: same contract as the warmup loop above.
        unsafe { function.launch(&cfg, &mut params, Some(&stream)) }.expect("launch failed");
    }
    stream.synchronize().expect("sync failed");
    let elapsed = start.elapsed().as_secs_f64();

    let bytes_per_pass = (M as u64) * (N as u64 / K_BLOCK_ELEMS as u64) * block_bytes as u64;
    let flops_per_pass = 2.0 * M as f64 * N as f64;
    let gb_per_s = bytes_per_pass as f64 * TIMED_ITERS as f64 / elapsed / 1e9;
    let gflop_per_s = flops_per_pass * TIMED_ITERS as f64 / elapsed / 1e9;
    println!(
        "{kernel_name}: {TIMED_ITERS} iters in {elapsed:.4}s -> {gb_per_s:.1} GB/s, {gflop_per_s:.1} GFLOP/s"
    );
}

#[test]
#[ignore]
fn perf_probe_gemv_q4_k() {
    let blocks_per_row = N as usize / K_BLOCK_ELEMS;
    let mut rng = Rng::new(101);
    let mut w_bytes = Vec::with_capacity(M as usize * blocks_per_row * Q4_K_BLOCK_BYTES);
    for _ in 0..M {
        w_bytes.extend(random_row_q4_k(&mut rng, blocks_per_row));
    }
    let x: Vec<f32> = (0..N).map(|i| ((i % 13) as f32) * 0.2 - 1.0).collect();
    perf_probe(
        rocml_kernels::GEMV_Q4_K_HSACO,
        rocml_kernels::GEMV_Q4_K_KERNEL,
        &w_bytes,
        &x,
        Q4_K_BLOCK_BYTES,
    );
}

#[test]
#[ignore]
fn perf_probe_gemv_q6_k() {
    let blocks_per_row = N as usize / K_BLOCK_ELEMS;
    let mut rng = Rng::new(102);
    let mut w_bytes = Vec::with_capacity(M as usize * blocks_per_row * Q6_K_BLOCK_BYTES);
    for _ in 0..M {
        w_bytes.extend(random_row_q6_k(&mut rng, blocks_per_row));
    }
    let x: Vec<f32> = (0..N).map(|i| ((i % 13) as f32) * 0.2 - 1.0).collect();
    perf_probe(
        rocml_kernels::GEMV_Q6_K_HSACO,
        rocml_kernels::GEMV_Q6_K_KERNEL,
        &w_bytes,
        &x,
        Q6_K_BLOCK_BYTES,
    );
}
