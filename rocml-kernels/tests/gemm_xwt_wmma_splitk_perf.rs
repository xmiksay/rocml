//! Informational perf probe for the split-K WMMA GEMM
//! (`kernels/gemm_xwt_wmma_splitk.hip` + `kernels/gemm_splitk_reduce.hip`,
//! issue #6's split-K follow-up) against the plain default-tile WMMA kernel
//! (`gemm_xwt_quant_wmma.hip`) — not a correctness gate (see
//! `gemm_xwt_wmma_splitk.rs` for that). Run with `cargo test --release -p
//! rocml-kernels --test gemm_xwt_wmma_splitk_perf -- --ignored --nocapture`.
//!
//! Measures the crossover the split-K round's dispatch thresholds
//! (`rocml/src/forward/kernels_quant_dispatch.rs`'s `SPLITK_BLOCK_THRESHOLD`/
//! `SPLITK_MIN_N`) are based on, at the three shapes issue #6 asked for:
//! ffn-down's real shape (m=4096, n=12288 — 128 plain-grid blocks at
//! rows=512, the motivating narrow-grid case) plus two smaller-`n` shapes
//! (m=1024/n=4096, m=2048/n=4096) to confirm split-K's win narrows or
//! reverses as `n` shrinks. All three run at `rows=512` (a full chunked-
//! prefill chunk). One process, interleaved rounds (plain then split(2) then
//! split(4), repeated `ROUNDS` times, median per config) — this machine's
//! ambient GPU clock state swings enough between separate `cargo test`
//! invocations to make cross-process comparisons unusable (see
//! `gemm_xwt_quant_wmma_perf.rs`'s module doc for the same finding); an
//! interleaved single-process measurement cancels that drift instead.
mod common;

use std::ffi::c_void;
use std::time::Instant;

use common::{random_row_q4_k, Rng, Q4_K_BLOCK_BYTES};
use rocml_hip::{kernel_params, Device, DeviceBuffer, LaunchConfig, Module, Stream};

const WARMUP_ITERS: u32 = 10;
const TIMED_ITERS: u32 = 100;
const ROUNDS: usize = 5;
const K_BLOCK_ELEMS: usize = 256;
const TILE_ROWS: u32 = 128;
const TILE_M: u32 = 128;
const K_STAGE: u32 = 16;
const LDS_PAD: u32 = 8;
const WARPS_PER_BLOCK: u32 = 16;
const REDUCE_BLOCK: u32 = 256;

fn make_x(rows: u32, n: u32) -> Vec<f32> {
    (0..rows * n)
        .map(|i| ((i % 13) as f32) * 0.2 - 1.0)
        .collect()
}

fn build_w_bytes(rng: &mut Rng, blocks_per_row: usize, m: usize) -> Vec<u8> {
    const ROW_POOL: usize = 256;
    let row_bytes = blocks_per_row * Q4_K_BLOCK_BYTES;
    let pool_rows = ROW_POOL.min(m).max(1);
    let mut pool = Vec::with_capacity(pool_rows * row_bytes);
    for _ in 0..pool_rows {
        pool.extend(random_row_q4_k(rng, blocks_per_row));
    }
    let mut out = Vec::with_capacity(m * row_bytes);
    for i in 0..m {
        let start = (i % pool_rows) * row_bytes;
        out.extend_from_slice(&pool[start..start + row_bytes]);
    }
    out
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

struct Case {
    x: DeviceBuffer<f32>,
    w: DeviceBuffer<u8>,
    plain_out: DeviceBuffer<f32>,
    splitk_partial_2: DeviceBuffer<f32>,
    splitk_partial_4: DeviceBuffer<f32>,
    splitk_out: DeviceBuffer<f32>,
}

fn time_launch(stream: &Stream, mut launch: impl FnMut()) -> f64 {
    for _ in 0..WARMUP_ITERS {
        launch();
    }
    stream.synchronize().expect("sync failed");
    let start = Instant::now();
    for _ in 0..TIMED_ITERS {
        launch();
    }
    stream.synchronize().expect("sync failed");
    start.elapsed().as_secs_f64()
}

fn gflops(rows: u32, m: u32, n: u32, elapsed: f64) -> f64 {
    let flops_per_pass = 2.0 * rows as f64 * m as f64 * n as f64;
    flops_per_pass * TIMED_ITERS as f64 / elapsed / 1e9
}

/// Runs the plain default-tile kernel and split(2)/split(4) interleaved
/// across `ROUNDS` rounds, printing the median GFLOP/s of each.
fn compare_shape(label: &str, rows: u32, m: u32, n: u32, seed: u32) {
    let _device = Device::new(0).expect("failed to select device 0");
    let plain_module = Module::load_from_bytes(rocml_kernels::GEMM_XWT_WMMA_Q4_K_HSACO)
        .expect("plain module load failed");
    let plain_fn = plain_module
        .get_function(rocml_kernels::GEMM_XWT_WMMA_Q4_K_KERNEL)
        .expect("plain kernel lookup failed");
    let splitk_module = Module::load_from_bytes(rocml_kernels::GEMM_XWT_WMMA_SPLITK_Q4_K_HSACO)
        .expect("splitk module load failed");
    let splitk_fn = splitk_module
        .get_function(rocml_kernels::GEMM_XWT_WMMA_SPLITK_Q4_K_KERNEL)
        .expect("splitk kernel lookup failed");
    let reduce_module = Module::load_from_bytes(rocml_kernels::GEMM_SPLITK_REDUCE_F32_HSACO)
        .expect("reduce module load failed");
    let reduce_fn = reduce_module
        .get_function(rocml_kernels::GEMM_SPLITK_REDUCE_F32_KERNEL)
        .expect("reduce kernel lookup failed");
    let stream = Stream::new().expect("failed to create stream");

    let blocks_per_row = n as usize / K_BLOCK_ELEMS;
    let mut rng = Rng::new(seed);
    let w_bytes = build_w_bytes(&mut rng, blocks_per_row, m as usize);
    let x = make_x(rows, n);

    let mut buf_x = DeviceBuffer::<f32>::new(x.len()).expect("hipMalloc x failed");
    let mut buf_w = DeviceBuffer::<u8>::new(w_bytes.len()).expect("hipMalloc w failed");
    buf_x.copy_from_host(&x).expect("copy x failed");
    buf_w.copy_from_host(&w_bytes).expect("copy w failed");
    let case = Case {
        x: buf_x,
        w: buf_w,
        plain_out: DeviceBuffer::new((rows * m) as usize).expect("hipMalloc out failed"),
        splitk_partial_2: DeviceBuffer::new((2 * rows * m) as usize)
            .expect("hipMalloc partial2 failed"),
        splitk_partial_4: DeviceBuffer::new((4 * rows * m) as usize)
            .expect("hipMalloc partial4 failed"),
        splitk_out: DeviceBuffer::new((rows * m) as usize).expect("hipMalloc splitk out failed"),
    };

    let x_ptr: *mut c_void = case.x.device_ptr();
    let w_ptr: *mut c_void = case.w.device_ptr();
    let plain_out_ptr: *mut c_void = case.plain_out.device_ptr();
    let shared_mem_bytes =
        2 * (TILE_ROWS + TILE_M) * (K_STAGE + LDS_PAD) * std::mem::size_of::<u16>() as u32;
    let plain_cfg = LaunchConfig {
        grid: (m.div_ceil(TILE_M), rows.div_ceil(TILE_ROWS), 1),
        block: (32, WARPS_PER_BLOCK, 1),
        shared_mem_bytes,
    };
    let mut plain_params = kernel_params!(x_ptr, w_ptr, plain_out_ptr, rows, m, n);

    let splitk_out_ptr: *mut c_void = case.splitk_out.device_ptr();
    let total = rows * m;
    let reduce_cfg = LaunchConfig {
        grid: (total.div_ceil(REDUCE_BLOCK), 1, 1),
        block: (REDUCE_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut plain_samples = Vec::with_capacity(ROUNDS);
    let mut split2_samples = Vec::with_capacity(ROUNDS);
    let mut split4_samples = Vec::with_capacity(ROUNDS);

    for _ in 0..ROUNDS {
        let elapsed = time_launch(&stream, || {
            // SAFETY: params matches gemm_xwt_wmma_q4_k's signature; cfg
            // matches the default tile config; buffers outlive the loop.
            unsafe { plain_fn.launch(&plain_cfg, &mut plain_params, Some(&stream)) }
                .expect("plain launch failed");
        });
        plain_samples.push(gflops(rows, m, n, elapsed));

        for (splits, partial_buf, samples) in [
            (2u32, &case.splitk_partial_2, &mut split2_samples),
            (4u32, &case.splitk_partial_4, &mut split4_samples),
        ] {
            let partial_ptr: *mut c_void = partial_buf.device_ptr();
            let splitk_cfg = LaunchConfig {
                grid: (m.div_ceil(TILE_M), rows.div_ceil(TILE_ROWS), splits),
                block: (32, WARPS_PER_BLOCK, 1),
                shared_mem_bytes,
            };
            let mut splitk_params = kernel_params!(x_ptr, w_ptr, partial_ptr, rows, m, n, splits);
            let mut reduce_params = kernel_params!(partial_ptr, splitk_out_ptr, rows, m, splits);
            let elapsed = time_launch(&stream, || {
                // SAFETY: params match gemm_xwt_wmma_splitk_q4_k/
                // gemm_splitk_reduce_f32's signatures; cfgs match this
                // file's fixed tile config; buffers outlive the loop.
                unsafe { splitk_fn.launch(&splitk_cfg, &mut splitk_params, Some(&stream)) }
                    .expect("splitk launch failed");
                unsafe { reduce_fn.launch(&reduce_cfg, &mut reduce_params, Some(&stream)) }
                    .expect("reduce launch failed");
            });
            samples.push(gflops(rows, m, n, elapsed));
        }
    }

    let plain_med = median(plain_samples);
    let split2_med = median(split2_samples);
    let split4_med = median(split4_samples);
    let plain_blocks = m.div_ceil(TILE_M) * rows.div_ceil(TILE_ROWS);
    println!(
        "{label} (rows={rows} m={m} n={n}, plain grid blocks={plain_blocks}): \
         plain={plain_med:.1} GFLOP/s, split2={split2_med:.1} GFLOP/s ({:+.1}%), \
         split4={split4_med:.1} GFLOP/s ({:+.1}%)",
        (split2_med / plain_med - 1.0) * 100.0,
        (split4_med / plain_med - 1.0) * 100.0,
    );
}

/// Same-session comparison against the *narrow*-tile plain kernel
/// (`TILE_M`=64, production's actual `m=1024` dispatch target today) — a
/// same-process launch right next to `compare_shape`'s default-tile plain/
/// split(2)/split(4) numbers for `m=1024`, to check whether split-K on the
/// default tile is *also* a better choice than the narrow-tile config below
/// `GEMM_WMMA_NARROW_THRESHOLD_M`(2048), not just better than doing nothing.
fn narrow_plain_gflops(rows: u32, m: u32, n: u32, seed: u32) -> f64 {
    const NARROW_TILE_M: u32 = 64;
    let _device = Device::new(0).expect("failed to select device 0");
    let module = Module::load_from_bytes(rocml_kernels::GEMM_XWT_WMMA_Q4_K_NARROW_HSACO)
        .expect("narrow module load failed");
    let function = module
        .get_function(rocml_kernels::GEMM_XWT_WMMA_Q4_K_NARROW_KERNEL)
        .expect("narrow kernel lookup failed");
    let stream = Stream::new().expect("failed to create stream");

    let blocks_per_row = n as usize / K_BLOCK_ELEMS;
    let mut rng = Rng::new(seed);
    let w_bytes = build_w_bytes(&mut rng, blocks_per_row, m as usize);
    let x = make_x(rows, n);
    let mut buf_x = DeviceBuffer::<f32>::new(x.len()).expect("hipMalloc x failed");
    let mut buf_w = DeviceBuffer::<u8>::new(w_bytes.len()).expect("hipMalloc w failed");
    let buf_out = DeviceBuffer::<f32>::new((rows * m) as usize).expect("hipMalloc out failed");
    buf_x.copy_from_host(&x).expect("copy x failed");
    buf_w.copy_from_host(&w_bytes).expect("copy w failed");

    let x_ptr: *mut c_void = buf_x.device_ptr();
    let w_ptr: *mut c_void = buf_w.device_ptr();
    let out_ptr: *mut c_void = buf_out.device_ptr();
    let mut params = kernel_params!(x_ptr, w_ptr, out_ptr, rows, m, n);
    let cfg = LaunchConfig {
        grid: (m.div_ceil(NARROW_TILE_M), rows.div_ceil(TILE_ROWS), 1),
        block: (32, WARPS_PER_BLOCK, 1),
        shared_mem_bytes: 2
            * (TILE_ROWS + NARROW_TILE_M)
            * (K_STAGE + LDS_PAD)
            * std::mem::size_of::<u16>() as u32,
    };
    let mut samples = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        let elapsed = time_launch(&stream, || {
            // SAFETY: params matches gemm_xwt_wmma_q4_k_narrow's signature;
            // cfg matches the narrow tile config; buffers outlive the loop.
            unsafe { function.launch(&cfg, &mut params, Some(&stream)) }
                .expect("narrow launch failed");
        });
        samples.push(gflops(rows, m, n, elapsed));
    }
    median(samples)
}

#[test]
#[ignore]
fn perf_probe_splitk_crossover() {
    // ffn-down's real shape: 128 plain-grid blocks at rows=512.
    compare_shape("ffn-down-class", 512, 4096, 12288, 300);
    // Narrower-n shapes named in the issue for crossover measurement.
    compare_shape("m=2048/n=4096", 512, 2048, 4096, 301);
    compare_shape(
        "m=1024/n=4096 (narrow-tile territory)",
        512,
        1024,
        4096,
        302,
    );
    let narrow = narrow_plain_gflops(512, 1024, 4096, 303);
    println!(
        "m=1024/n=4096 narrow-tile plain (production's current m=1024 dispatch): \
         {narrow:.1} GFLOP/s"
    );
}
