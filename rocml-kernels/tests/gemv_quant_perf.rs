//! Informational perf probe for the fused dequant-GEMV kernels — not a
//! correctness gate (see `gemv_q4_k.rs`/`gemv_q6_k.rs` for that), just a
//! manually-run measurement of the memory-bandwidth win these kernels exist
//! for. Run with `cargo test --release -p rocml-kernels --test
//! gemv_quant_perf -- --ignored --nocapture`.
//!
//! Covers the square 4096x4096 probe shape for all four quant types plus two
//! real model shapes (Ornith-1.0-9B, Q6_K): the lm-head/embedding-table shape
//! (m=248320, n=4096) and an FFN-down shape (m=4096, n=12288) — see
//! `rocml-kernels/kernels/gemv_q6_k.hip`'s doc comment for why these two
//! matter beyond the square shape.
mod common;

use std::ffi::c_void;
use std::time::Instant;

use common::{
    random_row_q4_k, random_row_q5_k, random_row_q6_k, random_row_q8_0, Rng, Q4_K_BLOCK_BYTES,
    Q5_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMS,
};
use rocml_hip::{kernel_params, Device, DeviceBuffer, LaunchConfig, Module, Stream};

const WARMUP_ITERS: u32 = 10;
const TIMED_ITERS: u32 = 200;
const K_BLOCK_ELEMS: usize = 256;
/// Matches `rocml`'s `REDUCE_BLOCK` (the fixed launch config every
/// `gemv_quant` caller uses; not exposed to this crate, so duplicated here)
/// — the probe must launch exactly what production does or the numbers
/// don't mean anything.
const BLOCK: u32 = 128;
/// Number of distinct rows generated per shape before cycling — the lm-head
/// shape's weight buffer is ~800MB, and generating that much through the
/// host-side xorshift RNG byte-by-byte would dominate probe setup time for
/// no benefit (the GPU still reads every byte of the full m x row_bytes
/// buffer; it doesn't care that the pattern repeats every 256 rows).
const ROW_POOL: usize = 256;

/// Builds an `m`-row weight buffer by generating `min(m, ROW_POOL)` unique
/// rows via `row_fn` and cycling through them for the rest.
fn build_w_bytes(
    mut row_fn: impl FnMut(&mut Rng, usize) -> Vec<u8>,
    rng: &mut Rng,
    blocks_per_row: usize,
    row_bytes: usize,
    m: usize,
) -> Vec<u8> {
    let pool_rows = ROW_POOL.min(m).max(1);
    let mut pool = Vec::with_capacity(pool_rows * row_bytes);
    for _ in 0..pool_rows {
        pool.extend(row_fn(rng, blocks_per_row));
    }
    let mut out = Vec::with_capacity(m * row_bytes);
    for i in 0..m {
        let start = (i % pool_rows) * row_bytes;
        out.extend_from_slice(&pool[start..start + row_bytes]);
    }
    out
}

fn make_x(n: u32) -> Vec<f32> {
    (0..n).map(|i| ((i % 13) as f32) * 0.2 - 1.0).collect()
}

/// Times `TIMED_ITERS` launches of `kernel_name` at shape `m x n` and prints
/// effective GB/s (weight bytes read per pass, ignoring the comparatively
/// tiny x/y traffic) and GFLOP/s (2 flops per element: one multiply, one
/// add). `label` distinguishes multiple shapes probed against the same
/// kernel in the printed output.
#[allow(clippy::too_many_arguments)]
fn perf_probe(
    label: &str,
    hsaco: &[u8],
    kernel_name: &str,
    w_bytes: &[u8],
    x: &[f32],
    block_bytes: usize,
    k_block_elems: usize,
    m: u32,
    n: u32,
) {
    let _device = Device::new(0).expect("failed to select device 0");
    let module = Module::load_from_bytes(hsaco).expect("module load failed");
    let function = module
        .get_function(kernel_name)
        .expect("kernel lookup failed");
    let stream = Stream::new().expect("failed to create stream");

    let mut buf_w = DeviceBuffer::<u8>::new(w_bytes.len()).expect("hipMalloc w failed");
    let mut buf_x = DeviceBuffer::<f32>::new(x.len()).expect("hipMalloc x failed");
    let buf_y = DeviceBuffer::<f32>::new(m as usize).expect("hipMalloc y failed");
    buf_w.copy_from_host(w_bytes).expect("copy w failed");
    buf_x.copy_from_host(x).expect("copy x failed");

    let w_ptr: *mut c_void = buf_w.device_ptr();
    let x_ptr: *mut c_void = buf_x.device_ptr();
    let y_ptr: *mut c_void = buf_y.device_ptr();
    let mut params = kernel_params!(w_ptr, x_ptr, y_ptr, m, n);

    let cfg = LaunchConfig {
        grid: (m, 1, 1),
        block: (BLOCK, 1, 1),
        shared_mem_bytes: BLOCK * std::mem::size_of::<f32>() as u32,
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

    let bytes_per_pass = (m as u64) * (n as u64 / k_block_elems as u64) * block_bytes as u64;
    let flops_per_pass = 2.0 * m as f64 * n as f64;
    let gb_per_s = bytes_per_pass as f64 * TIMED_ITERS as f64 / elapsed / 1e9;
    let gflop_per_s = flops_per_pass * TIMED_ITERS as f64 / elapsed / 1e9;
    println!(
        "{label}: {TIMED_ITERS} iters in {elapsed:.4}s -> {gb_per_s:.1} GB/s, {gflop_per_s:.1} GFLOP/s"
    );
}

#[test]
#[ignore]
fn perf_probe_gemv_q8_0() {
    let (m, n) = (4096u32, 4096u32);
    let blocks_per_row = n as usize / Q8_0_BLOCK_ELEMS;
    let mut rng = Rng::new(100);
    let w_bytes = build_w_bytes(
        random_row_q8_0,
        &mut rng,
        blocks_per_row,
        blocks_per_row * Q8_0_BLOCK_BYTES,
        m as usize,
    );
    perf_probe(
        "gemv_q8_0 (square 4096x4096)",
        rocml_kernels::GEMV_Q8_0_HSACO,
        rocml_kernels::GEMV_Q8_0_KERNEL,
        &w_bytes,
        &make_x(n),
        Q8_0_BLOCK_BYTES,
        Q8_0_BLOCK_ELEMS,
        m,
        n,
    );
}

#[test]
#[ignore]
fn perf_probe_gemv_q4_k() {
    let (m, n) = (4096u32, 4096u32);
    let blocks_per_row = n as usize / K_BLOCK_ELEMS;
    let mut rng = Rng::new(101);
    let w_bytes = build_w_bytes(
        random_row_q4_k,
        &mut rng,
        blocks_per_row,
        blocks_per_row * Q4_K_BLOCK_BYTES,
        m as usize,
    );
    perf_probe(
        "gemv_q4_k (square 4096x4096)",
        rocml_kernels::GEMV_Q4_K_HSACO,
        rocml_kernels::GEMV_Q4_K_KERNEL,
        &w_bytes,
        &make_x(n),
        Q4_K_BLOCK_BYTES,
        K_BLOCK_ELEMS,
        m,
        n,
    );
}

#[test]
#[ignore]
fn perf_probe_gemv_q5_k() {
    let (m, n) = (4096u32, 4096u32);
    let blocks_per_row = n as usize / K_BLOCK_ELEMS;
    let mut rng = Rng::new(103);
    let w_bytes = build_w_bytes(
        random_row_q5_k,
        &mut rng,
        blocks_per_row,
        blocks_per_row * Q5_K_BLOCK_BYTES,
        m as usize,
    );
    perf_probe(
        "gemv_q5_k (square 4096x4096)",
        rocml_kernels::GEMV_Q5_K_HSACO,
        rocml_kernels::GEMV_Q5_K_KERNEL,
        &w_bytes,
        &make_x(n),
        Q5_K_BLOCK_BYTES,
        K_BLOCK_ELEMS,
        m,
        n,
    );
}

#[test]
#[ignore]
fn perf_probe_gemv_q6_k() {
    let (m, n) = (4096u32, 4096u32);
    let blocks_per_row = n as usize / K_BLOCK_ELEMS;
    let mut rng = Rng::new(102);
    let w_bytes = build_w_bytes(
        random_row_q6_k,
        &mut rng,
        blocks_per_row,
        blocks_per_row * Q6_K_BLOCK_BYTES,
        m as usize,
    );
    perf_probe(
        "gemv_q6_k (square 4096x4096)",
        rocml_kernels::GEMV_Q6_K_HSACO,
        rocml_kernels::GEMV_Q6_K_KERNEL,
        &w_bytes,
        &make_x(n),
        Q6_K_BLOCK_BYTES,
        K_BLOCK_ELEMS,
        m,
        n,
    );
}

/// Ornith-1.0-9B's `output.weight`/`token_embd.weight` shape (Q6_K):
/// m=248320 (vocab), n=4096 (hidden). Dominates prefill/prompt-eval and the
/// very first decode step; also the shape most likely to be split-k
/// territory since a single 128-lane workgroup per row already has plenty
/// of independent rows (248320) to hide latency across 60 CUs even before
/// any occupancy fix.
#[test]
#[ignore]
fn perf_probe_gemv_q6_k_lm_head() {
    let (m, n) = (248_320u32, 4096u32);
    let blocks_per_row = n as usize / K_BLOCK_ELEMS;
    let mut rng = Rng::new(104);
    let w_bytes = build_w_bytes(
        random_row_q6_k,
        &mut rng,
        blocks_per_row,
        blocks_per_row * Q6_K_BLOCK_BYTES,
        m as usize,
    );
    perf_probe(
        "gemv_q6_k (lm_head 248320x4096)",
        rocml_kernels::GEMV_Q6_K_HSACO,
        rocml_kernels::GEMV_Q6_K_KERNEL,
        &w_bytes,
        &make_x(n),
        Q6_K_BLOCK_BYTES,
        K_BLOCK_ELEMS,
        m,
        n,
    );
}

/// Ornith-1.0-9B's `blk.*.ffn_down.weight` shape (Q6_K): m=4096 (hidden),
/// n=12288 (ffn intermediate) — the "wide" FFN transpose direction, 48
/// superblocks per row instead of the square shape's 16.
#[test]
#[ignore]
fn perf_probe_gemv_q6_k_ffn_down() {
    let (m, n) = (4096u32, 12288u32);
    let blocks_per_row = n as usize / K_BLOCK_ELEMS;
    let mut rng = Rng::new(105);
    let w_bytes = build_w_bytes(
        random_row_q6_k,
        &mut rng,
        blocks_per_row,
        blocks_per_row * Q6_K_BLOCK_BYTES,
        m as usize,
    );
    perf_probe(
        "gemv_q6_k (ffn_down 4096x12288)",
        rocml_kernels::GEMV_Q6_K_HSACO,
        rocml_kernels::GEMV_Q6_K_KERNEL,
        &w_bytes,
        &make_x(n),
        Q6_K_BLOCK_BYTES,
        K_BLOCK_ELEMS,
        m,
        n,
    );
}
