//! Informational perf probe for the int8 MMQ-style `gemm_xwt_mmq_q4_k`
//! kernel (WMMA-pipeline round's lever 2 prototype) — the Q4_K sibling of
//! `gemm_xwt_quant_mmq_perf.rs`'s Q8_0 probe, directly comparable to
//! `gemm_xwt_quant_wmma_perf.rs`'s f16 WMMA Q4_K numbers at the same
//! shapes (this is the production quant — ornith-9b Q4_K_M's FFN/projection
//! GEMMs are Q4_K). Run with `cargo test --release -p rocml-kernels --test
//! gemm_xwt_quant_mmq_q4k_perf -- --ignored --nocapture --test-threads=1`.
mod common;

use std::ffi::c_void;
use std::time::Instant;

use common::{random_row_q4_k, Rng, Q4_K_BLOCK_BYTES};
use rocml_hip::{kernel_params, Device, DeviceBuffer, LaunchConfig, Module, Stream};

const WARMUP_ITERS: u32 = 20;
const TIMED_ITERS: u32 = 300;
const TILE_ROWS: u32 = 128;
const TILE_M: u32 = 64;
const K_STAGE: u32 = 32;
const WARPS_PER_BLOCK: u32 = 16;
const QUANT_WARPS_PER_BLOCK: u32 = 8;

fn make_x(rows: u32, n: u32) -> Vec<f32> {
    (0..rows * n)
        .map(|i| ((i % 13) as f32) * 0.2 - 1.0)
        .collect()
}

fn build_w_bytes(m: u32, n: u32, seed: u32) -> Vec<u8> {
    const ROW_POOL: usize = 256;
    let blocks_per_row = (n / 256) as usize;
    let row_bytes = blocks_per_row * Q4_K_BLOCK_BYTES;
    let mut rng = Rng::new(seed);
    let pool_rows = ROW_POOL.min(m as usize).max(1);
    let mut pool = Vec::with_capacity(pool_rows * row_bytes);
    for _ in 0..pool_rows {
        pool.extend(random_row_q4_k(&mut rng, blocks_per_row));
    }
    let mut out = Vec::with_capacity(m as usize * row_bytes);
    for i in 0..m as usize {
        let start = (i % pool_rows) * row_bytes;
        out.extend_from_slice(&pool[start..start + row_bytes]);
    }
    out
}

fn perf_case(label: &str, rows: u32, m: u32, n: u32, seed: u32) {
    let _device = Device::new(0).expect("failed to select device 0");
    let quant_module = Module::load_from_bytes(rocml_kernels::QUANTIZE_ACT_Q8_BLK_HSACO)
        .expect("quant module load");
    let quant_fn = quant_module
        .get_function(rocml_kernels::QUANTIZE_ACT_Q8_BLK_KERNEL)
        .expect("quant kernel lookup");
    let gemm_module =
        Module::load_from_bytes(rocml_kernels::GEMM_XWT_MMQ_Q4_K_HSACO).expect("gemm module load");
    let gemm_fn = gemm_module
        .get_function(rocml_kernels::GEMM_XWT_MMQ_Q4_K_KERNEL)
        .expect("gemm kernel lookup");
    let stream = Stream::new().expect("stream create failed");

    let x = make_x(rows, n);
    let w_bytes = build_w_bytes(m, n, seed);
    let n_blocks = n / 32;

    let mut buf_x = DeviceBuffer::<f32>::new(x.len()).expect("hipMalloc x");
    let buf_x_codes = DeviceBuffer::<i8>::new((rows * n) as usize).expect("hipMalloc x_codes");
    let buf_x_scale = DeviceBuffer::<f32>::new((rows * n_blocks) as usize).expect("scale");
    let buf_x_sum = DeviceBuffer::<f32>::new((rows * n_blocks) as usize).expect("sum");
    let mut buf_w = DeviceBuffer::<u8>::new(w_bytes.len()).expect("hipMalloc w");
    let buf_out = DeviceBuffer::<f32>::new((rows * m) as usize).expect("hipMalloc out");
    buf_x.copy_from_host(&x).expect("copy x");
    buf_w.copy_from_host(&w_bytes).expect("copy w");

    let x_ptr: *mut c_void = buf_x.device_ptr();
    let xc_ptr: *mut c_void = buf_x_codes.device_ptr();
    let xs_ptr: *mut c_void = buf_x_scale.device_ptr();
    let xsum_ptr: *mut c_void = buf_x_sum.device_ptr();
    let w_ptr: *mut c_void = buf_w.device_ptr();
    let out_ptr: *mut c_void = buf_out.device_ptr();

    let mut quant_params = kernel_params!(x_ptr, xc_ptr, xs_ptr, xsum_ptr, rows, n);
    let total_warps = rows * n_blocks;
    let quant_cfg = LaunchConfig {
        grid: (total_warps.div_ceil(QUANT_WARPS_PER_BLOCK), 1, 1),
        block: (32, QUANT_WARPS_PER_BLOCK, 1),
        shared_mem_bytes: 0,
    };

    let mut gemm_params = kernel_params!(xc_ptr, xs_ptr, xsum_ptr, w_ptr, out_ptr, rows, m, n);
    let gemm_cfg = LaunchConfig {
        grid: (m.div_ceil(TILE_M), rows.div_ceil(TILE_ROWS), 1),
        block: (32, WARPS_PER_BLOCK, 1),
        shared_mem_bytes: TILE_ROWS * K_STAGE
            + TILE_M * K_STAGE
            + TILE_ROWS * 4 * 2
            + TILE_M * 4 * 2,
    };

    // SAFETY: params/cfg match quantize_act_q8_blk's and gemm_xwt_mmq_q4_k's
    // parameter lists respectively; all device buffers outlive every launch
    // in these loops.
    for _ in 0..WARMUP_ITERS {
        unsafe { quant_fn.launch(&quant_cfg, &mut quant_params, Some(&stream)) }
            .expect("quant launch failed");
        unsafe { gemm_fn.launch(&gemm_cfg, &mut gemm_params, Some(&stream)) }
            .expect("gemm launch failed");
    }
    stream.synchronize().expect("sync failed");

    let start = Instant::now();
    for _ in 0..TIMED_ITERS {
        unsafe { gemm_fn.launch(&gemm_cfg, &mut gemm_params, Some(&stream)) }
            .expect("gemm launch failed");
    }
    stream.synchronize().expect("sync failed");
    let gemm_elapsed = start.elapsed().as_secs_f64();

    let start = Instant::now();
    for _ in 0..TIMED_ITERS {
        unsafe { quant_fn.launch(&quant_cfg, &mut quant_params, Some(&stream)) }
            .expect("quant launch failed");
        unsafe { gemm_fn.launch(&gemm_cfg, &mut gemm_params, Some(&stream)) }
            .expect("gemm launch failed");
    }
    stream.synchronize().expect("sync failed");
    let total_elapsed = start.elapsed().as_secs_f64();

    let flops_per_pass = 2.0 * rows as f64 * m as f64 * n as f64;
    let gemm_gflops = flops_per_pass * TIMED_ITERS as f64 / gemm_elapsed / 1e9;
    let total_gflops = flops_per_pass * TIMED_ITERS as f64 / total_elapsed / 1e9;
    println!(
        "{label}: gemm-only {gemm_gflops:.1} GFLOP/s ({gemm_elapsed:.4}s/{TIMED_ITERS}), \
         quant+gemm {total_gflops:.1} GFLOP/s ({total_elapsed:.4}s/{TIMED_ITERS})"
    );
}

#[test]
#[ignore]
fn perf_probe_gemm_xwt_mmq_q4_k_12288() {
    perf_case(
        "gemm_xwt_mmq_q4_k (128x4096 x 4096x12288)",
        128,
        12288,
        4096,
        600,
    );
}

#[test]
#[ignore]
fn perf_probe_gemm_xwt_mmq_q4_k_512x12288() {
    perf_case(
        "gemm_xwt_mmq_q4_k (512x4096 x 4096x12288)",
        512,
        12288,
        4096,
        601,
    );
}
