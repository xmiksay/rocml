//! Informational perf probe for the WMMA matrix-core `gemm_xwt_wmma_q*`
//! kernels (`kernels/gemm_xwt_quant_wmma.hip`, issue #6's follow-up to
//! `gemm_xwt_quant_perf.rs`'s scalar-kernel probe) — not a correctness gate
//! (see `gemm_xwt_quant_wmma.rs` for that), just a manually-run measurement
//! of chunked-prefill GEMM throughput. Run with `cargo test --release -p
//! rocml-kernels --test gemm_xwt_quant_wmma_perf -- --ignored --nocapture`.
//!
//! Same shapes as `gemm_xwt_quant_perf.rs` (issue #6's acceptance shapes:
//! `rows=128` against `n=4096` x `m in {4096, 8192, 12288}`) so the two
//! probes' printed GFLOP/s are directly comparable before/after.
mod common;

use std::ffi::c_void;
use std::time::Instant;

use common::{random_row_q4_k, random_row_q6_k, Rng, Q4_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES};
use rocml_hip::{kernel_params, Device, DeviceBuffer, LaunchConfig, Module, Stream};

const WARMUP_ITERS: u32 = 20;
const TIMED_ITERS: u32 = 300;
const ROWS: u32 = 128;
const K_BLOCK_ELEMS: usize = 256;
/// Mirrors `gemm_xwt_quant_wmma.hip`'s fixed tile sizes (not exposed to this
/// crate, so duplicated here) — the probe must launch exactly what
/// production does or the numbers don't mean anything.
const TILE_ROWS: u32 = 128;
const TILE_M: u32 = 64;
const K_STAGE: u32 = 16;
const WARPS_PER_BLOCK: u32 = 16;

fn make_x(rows: u32, n: u32) -> Vec<f32> {
    (0..rows * n)
        .map(|i| ((i % 13) as f32) * 0.2 - 1.0)
        .collect()
}

/// Builds an `m`-row weight buffer by generating `min(m, ROW_POOL)` unique
/// rows via `row_fn` and cycling — same rationale as `gemv_quant_perf`'s
/// `build_w_bytes` (avoids the host-side RNG dominating probe setup time).
fn build_w_bytes(
    mut row_fn: impl FnMut(&mut Rng, usize) -> Vec<u8>,
    rng: &mut Rng,
    blocks_per_row: usize,
    row_bytes: usize,
    m: usize,
) -> Vec<u8> {
    const ROW_POOL: usize = 256;
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

#[allow(clippy::too_many_arguments)]
fn perf_probe(
    label: &str,
    hsaco: &[u8],
    kernel_name: &str,
    w_bytes: &[u8],
    x: &[f32],
    block_bytes: usize,
    k_block_elems: usize,
    rows: u32,
    m: u32,
    n: u32,
) {
    let _device = Device::new(0).expect("failed to select device 0");
    let module = Module::load_from_bytes(hsaco).expect("module load failed");
    let function = module
        .get_function(kernel_name)
        .expect("kernel lookup failed");
    let stream = Stream::new().expect("failed to create stream");

    let mut buf_x = DeviceBuffer::<f32>::new(x.len()).expect("hipMalloc x failed");
    let mut buf_w = DeviceBuffer::<u8>::new(w_bytes.len()).expect("hipMalloc w failed");
    let buf_out = DeviceBuffer::<f32>::new((rows * m) as usize).expect("hipMalloc out failed");
    buf_x.copy_from_host(x).expect("copy x failed");
    buf_w.copy_from_host(w_bytes).expect("copy w failed");

    let x_ptr: *mut c_void = buf_x.device_ptr();
    let w_ptr: *mut c_void = buf_w.device_ptr();
    let out_ptr: *mut c_void = buf_out.device_ptr();
    let mut params = kernel_params!(x_ptr, w_ptr, out_ptr, rows, m, n);

    let cfg = LaunchConfig {
        grid: (m.div_ceil(TILE_M), rows.div_ceil(TILE_ROWS), 1),
        block: (32, WARPS_PER_BLOCK, 1),
        shared_mem_bytes: (TILE_ROWS + TILE_M) * K_STAGE * std::mem::size_of::<u16>() as u32,
    };

    for _ in 0..WARMUP_ITERS {
        // SAFETY: params matches gemm_xwt_wmma_<type>'s parameter list
        // (const float*, const void*, float*, unsigned x3) in order, and
        // all device buffers outlive every launch in this loop.
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
    let flops_per_pass = 2.0 * rows as f64 * m as f64 * n as f64;
    let gb_per_s = bytes_per_pass as f64 * TIMED_ITERS as f64 / elapsed / 1e9;
    let gflop_per_s = flops_per_pass * TIMED_ITERS as f64 / elapsed / 1e9;
    println!(
        "{label}: {TIMED_ITERS} iters in {elapsed:.4}s -> {gb_per_s:.1} GB/s (ideal, one read/row), {gflop_per_s:.1} GFLOP/s"
    );
}

fn q4_k_case(label: &str, m: u32, n: u32, seed: u32) {
    let blocks_per_row = n as usize / K_BLOCK_ELEMS;
    let mut rng = Rng::new(seed);
    let w_bytes = build_w_bytes(
        random_row_q4_k,
        &mut rng,
        blocks_per_row,
        blocks_per_row * Q4_K_BLOCK_BYTES,
        m as usize,
    );
    perf_probe(
        label,
        rocml_kernels::GEMM_XWT_WMMA_Q4_K_HSACO,
        rocml_kernels::GEMM_XWT_WMMA_Q4_K_KERNEL,
        &w_bytes,
        &make_x(ROWS, n),
        Q4_K_BLOCK_BYTES,
        K_BLOCK_ELEMS,
        ROWS,
        m,
        n,
    );
}

fn q6_k_case(label: &str, m: u32, n: u32, seed: u32) {
    let blocks_per_row = n as usize / K_BLOCK_ELEMS;
    let mut rng = Rng::new(seed);
    let w_bytes = build_w_bytes(
        random_row_q6_k,
        &mut rng,
        blocks_per_row,
        blocks_per_row * Q6_K_BLOCK_BYTES,
        m as usize,
    );
    perf_probe(
        label,
        rocml_kernels::GEMM_XWT_WMMA_Q6_K_HSACO,
        rocml_kernels::GEMM_XWT_WMMA_Q6_K_KERNEL,
        &w_bytes,
        &make_x(ROWS, n),
        Q6_K_BLOCK_BYTES,
        K_BLOCK_ELEMS,
        ROWS,
        m,
        n,
    );
}

#[test]
#[ignore]
fn perf_probe_gemm_xwt_wmma_q4_k_4096() {
    q4_k_case("gemm_xwt_wmma_q4_k (128x4096 x 4096x4096)", 4096, 4096, 200);
}

#[test]
#[ignore]
fn perf_probe_gemm_xwt_wmma_q4_k_8192() {
    q4_k_case("gemm_xwt_wmma_q4_k (128x4096 x 4096x8192)", 8192, 4096, 201);
}

#[test]
#[ignore]
fn perf_probe_gemm_xwt_wmma_q4_k_12288() {
    q4_k_case(
        "gemm_xwt_wmma_q4_k (128x4096 x 4096x12288)",
        12288,
        4096,
        202,
    );
}

#[test]
#[ignore]
fn perf_probe_gemm_xwt_wmma_q6_k_4096() {
    q6_k_case("gemm_xwt_wmma_q6_k (128x4096 x 4096x4096)", 4096, 4096, 210);
}

#[test]
#[ignore]
fn perf_probe_gemm_xwt_wmma_q6_k_8192() {
    q6_k_case("gemm_xwt_wmma_q6_k (128x4096 x 4096x8192)", 8192, 4096, 211);
}

#[test]
#[ignore]
fn perf_probe_gemm_xwt_wmma_q6_k_12288() {
    q6_k_case(
        "gemm_xwt_wmma_q6_k (128x4096 x 4096x12288)",
        12288,
        4096,
        212,
    );
}
