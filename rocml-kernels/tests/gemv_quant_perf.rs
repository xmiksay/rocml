//! Informational perf probe for the fused dequant-GEMV kernels — not a
//! correctness gate (see `gemv_q4_k.rs`/`gemv_q6_k.rs` for that), just a
//! manually-run measurement of the memory-bandwidth win these kernels exist
//! for. Run with `cargo test --release -p rocml-kernels --test
//! gemv_quant_perf -- --ignored --nocapture`.
//!
//! Table-driven over issue #7's acceptance shapes: `n=4096` at
//! `m in {4096, 8192, 12288, 248320}` (qkv/attn-out, ffn-gate-up, an
//! ffn-down-shaped square probe, and the lm-head/embedding-table shape) for
//! every quant type, plus the real Ornith-1.0-9B ffn-down shape
//! (`m=4096, n=12288`) — see `rocml-kernels/kernels/gemv_q6_k.hip`'s doc
//! comment for why the non-square shapes matter beyond `m=n=4096`.
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
/// Matches `rocml`'s `GEMV_ROWS_PER_WG` and every `gemv_q*.hip` kernel's own
/// `ROWS_PER_WG` — see `common::GEMV_ROWS_PER_WG`'s doc comment.
const ROWS_PER_WG: u32 = common::GEMV_ROWS_PER_WG;
/// Number of distinct rows generated per shape before cycling — the lm-head
/// shape's weight buffer is ~800MB, and generating that much through the
/// host-side xorshift RNG byte-by-byte would dominate probe setup time for
/// no benefit (the GPU still reads every byte of the full m x row_bytes
/// buffer; it doesn't care that the pattern repeats every 256 rows).
const ROW_POOL: usize = 256;

/// Issue #7's acceptance shapes at `n=4096`: qkv/attn-out (smallest, most
/// latency-bound), ffn-gate-up, an `m=12288` square probe, and the
/// lm-head/embedding-table shape.
const SHAPES_N4096: &[(u32, &str)] = &[
    (4096, "m=4096"),
    (8192, "m=8192"),
    (12288, "m=12288"),
    (248_320, "m=248320 (lm_head)"),
];

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
        grid: (m.div_ceil(ROWS_PER_WG), 1, 1),
        block: (BLOCK, 1, 1),
        shared_mem_bytes: BLOCK * ROWS_PER_WG * std::mem::size_of::<f32>() as u32,
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

/// Runs `perf_probe` for every shape in `SHAPES_N4096` (all `n=4096`) against
/// one quant kernel.
#[allow(clippy::too_many_arguments)]
fn run_shape_sweep(
    quant_label: &str,
    hsaco: &[u8],
    kernel_name: &str,
    row_fn: impl Fn(&mut Rng, usize) -> Vec<u8>,
    seed_base: u32,
    block_bytes: usize,
    k_block_elems: usize,
) {
    let n = 4096u32;
    let blocks_per_row = n as usize / k_block_elems;
    for (i, (m, shape_label)) in SHAPES_N4096.iter().enumerate() {
        let mut rng = Rng::new(seed_base + i as u32);
        let w_bytes = build_w_bytes(
            |r, bpr| row_fn(r, bpr),
            &mut rng,
            blocks_per_row,
            blocks_per_row * block_bytes,
            *m as usize,
        );
        perf_probe(
            &format!("{quant_label} ({shape_label}, n=4096)"),
            hsaco,
            kernel_name,
            &w_bytes,
            &make_x(n),
            block_bytes,
            k_block_elems,
            *m,
            n,
        );
    }
}

#[test]
#[ignore]
fn perf_probe_gemv_q8_0() {
    run_shape_sweep(
        "gemv_q8_0",
        rocml_kernels::GEMV_Q8_0_HSACO,
        rocml_kernels::GEMV_Q8_0_KERNEL,
        random_row_q8_0,
        100,
        Q8_0_BLOCK_BYTES,
        Q8_0_BLOCK_ELEMS,
    );
}

#[test]
#[ignore]
fn perf_probe_gemv_q4_k() {
    run_shape_sweep(
        "gemv_q4_k",
        rocml_kernels::GEMV_Q4_K_HSACO,
        rocml_kernels::GEMV_Q4_K_KERNEL,
        random_row_q4_k,
        200,
        Q4_K_BLOCK_BYTES,
        K_BLOCK_ELEMS,
    );
}

#[test]
#[ignore]
fn perf_probe_gemv_q5_k() {
    run_shape_sweep(
        "gemv_q5_k",
        rocml_kernels::GEMV_Q5_K_HSACO,
        rocml_kernels::GEMV_Q5_K_KERNEL,
        random_row_q5_k,
        300,
        Q5_K_BLOCK_BYTES,
        K_BLOCK_ELEMS,
    );
}

#[test]
#[ignore]
fn perf_probe_gemv_q6_k() {
    run_shape_sweep(
        "gemv_q6_k",
        rocml_kernels::GEMV_Q6_K_HSACO,
        rocml_kernels::GEMV_Q6_K_KERNEL,
        random_row_q6_k,
        400,
        Q6_K_BLOCK_BYTES,
        K_BLOCK_ELEMS,
    );
}

/// Ornith-1.0-9B's `blk.*.ffn_down.weight` shape (Q6_K): m=4096 (hidden),
/// n=12288 (ffn intermediate) — the "wide" FFN transpose direction, 48
/// superblocks per row instead of the square shape's 16. Not covered by
/// `SHAPES_N4096` above (all `n=4096`), so kept as its own test.
#[test]
#[ignore]
fn perf_probe_gemv_q6_k_ffn_down() {
    let (m, n) = (4096u32, 12288u32);
    let blocks_per_row = n as usize / K_BLOCK_ELEMS;
    let mut rng = Rng::new(500);
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
