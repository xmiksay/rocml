//! Informational perf probe (qwen35moe M4 lever 1): does the micro-tile
//! WMMA kernel (`gemm_xwt_wmma_q*_micro`) actually beat the scalar
//! `gemm_xwt_q*` kernel at the small `rows` MoE's grouped-by-expert batched
//! GEMM produces? Not a correctness gate (see `gemm_xwt_quant_wmma_micro.rs`
//! for that) — this just measures, per this codebase's own "measure, don't
//! assume" rule for a new dispatch threshold (see `.claude/CLAUDE.md`'s
//! `GEMM_WMMA_NARROW_THRESHOLD_M`/`SPLITK_BLOCK_THRESHOLD` for the precedent
//! this mirrors). Run with `cargo test --release -p rocml-kernels --test
//! gemm_xwt_quant_wmma_micro_perf -- --ignored --nocapture --test-threads=1`.
//!
//! Shapes are Ornith-1.5-35B-A3B's real expert projections: gate/up
//! (`m=expert_ff_len=512, n=hidden=2048`) and down (`m=hidden=2048,
//! n=expert_ff_len=512`) — both dtypes the checkpoint actually uses (Q4_K
//! gate/up, Q6_K down, per `.claude/CLAUDE.md`'s M4 plan). `rows` sweeps
//! {1,2,4,8,16,32,64,127} — 127 is one below the default/narrow kernels'
//! own `rows>=128` dispatch floor, so this probe's range and that one's
//! never overlap. Every config is measured **interleaved** in one process
//! (this machine shows up to 2x swings in separate-process/cross-run
//! comparisons from ambient GPU clock state — the established methodology
//! this codebase's own WMMA tuning rounds settled on).
mod common;

use std::ffi::c_void;
use std::time::Instant;

use common::{random_row_q4_k, random_row_q6_k, Rng, Q4_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES};
use rocml_hip::{kernel_params, Device, DeviceBuffer, LaunchConfig, Module, Stream};

const WARMUP_ITERS: u32 = 20;
const TIMED_ITERS: u32 = 300;
const ROUNDS: u32 = 3;

// Scalar `gemm_xwt_q*` launch geometry (mirrors `kernels_quant_dispatch.rs`'s
// `GEMM_QUANT_*` constants).
const SCALAR_TILE_ROWS: u32 = 64;
const SCALAR_WARPS_PER_BLOCK: u32 = 8;
const SCALAR_TILE_ELEMS: u32 = 256;

// Micro WMMA launch geometry (mirrors `gemm_xwt_quant_wmma_micro.hip`).
const MICRO_TILE_ROWS: u32 = 16;
const MICRO_TILE_M: u32 = 64;
const MICRO_WARPS_PER_BLOCK: u32 = 4;
const K_STAGE: u32 = 16;
const LDS_PAD: u32 = 8;

fn make_x(rows: u32, n: u32) -> Vec<f32> {
    (0..rows * n)
        .map(|i| ((i % 13) as f32) * 0.2 - 1.0)
        .collect()
}

fn build_w_bytes(
    mut row_fn: impl FnMut(&mut Rng, usize) -> Vec<u8>,
    rng: &mut Rng,
    blocks_per_row: usize,
    row_bytes: usize,
    m: usize,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(m * row_bytes);
    for _ in 0..m {
        out.extend(row_fn(rng, blocks_per_row));
    }
    out
}

struct Probe {
    // Must outlive `function` — a `Function` handle is only valid while its
    // owning `Module` stays loaded (dropping the `Module` calls
    // `hipModuleUnload`, which invalidates any `Function` resolved from
    // it, even though `Function` holds no lifetime tie back to `Module`).
    _module: Module,
    function: rocml_hip::Function,
    stream: Stream,
    _buf_x: DeviceBuffer<f32>,
    _buf_w: DeviceBuffer<u8>,
    _buf_out: DeviceBuffer<f32>,
    params_ptrs: (*mut c_void, *mut c_void, *mut c_void),
    /// Builds this kernel's exact launch config for a given `rows` — grid.y
    /// depends on `rows`, so this can't be computed once at setup time.
    cfg_fn: fn(rows: u32, m: u32) -> LaunchConfig,
}

fn setup(
    hsaco: &[u8],
    kernel_name: &str,
    w_bytes: &[u8],
    x: &[f32],
    out_len: usize,
    cfg_fn: fn(rows: u32, m: u32) -> LaunchConfig,
) -> Probe {
    let module = Module::load_from_bytes(hsaco).expect("module load failed");
    let function = module
        .get_function(kernel_name)
        .expect("kernel lookup failed");
    let stream = Stream::new().expect("stream create failed");
    let mut buf_x = DeviceBuffer::<f32>::new(x.len()).expect("hipMalloc x failed");
    let mut buf_w = DeviceBuffer::<u8>::new(w_bytes.len()).expect("hipMalloc w failed");
    let buf_out = DeviceBuffer::<f32>::new(out_len).expect("hipMalloc out failed");
    buf_x.copy_from_host(x).expect("copy x failed");
    buf_w.copy_from_host(w_bytes).expect("copy w failed");
    let params_ptrs = (buf_x.device_ptr(), buf_w.device_ptr(), buf_out.device_ptr());
    Probe {
        _module: module,
        function,
        stream,
        _buf_x: buf_x,
        _buf_w: buf_w,
        _buf_out: buf_out,
        params_ptrs,
        cfg_fn,
    }
}

fn time_launch(probe: &Probe, rows: u32, m: u32, n: u32) -> f64 {
    let (x_ptr, w_ptr, out_ptr) = probe.params_ptrs;
    let mut params = kernel_params!(x_ptr, w_ptr, out_ptr, rows, m, n);
    let cfg = (probe.cfg_fn)(rows, m);
    for _ in 0..WARMUP_ITERS {
        // SAFETY: params matches every gemm_xwt_<...>'s parameter list
        // (const float*, const void*, float*, unsigned x3); cfg matches
        // this exact kernel's compiled-in tiling for this `rows`/`m`.
        unsafe {
            probe
                .function
                .launch(&cfg, &mut params, Some(&probe.stream))
        }
        .expect("launch failed");
    }
    probe.stream.synchronize().expect("sync failed");
    let start = Instant::now();
    for _ in 0..TIMED_ITERS {
        unsafe {
            probe
                .function
                .launch(&cfg, &mut params, Some(&probe.stream))
        }
        .expect("launch failed");
    }
    probe.stream.synchronize().expect("sync failed");
    start.elapsed().as_secs_f64() / TIMED_ITERS as f64 * 1e6 // us/call
}

#[allow(clippy::too_many_arguments)]
fn sweep_shape(
    label: &str,
    scalar_hsaco: &[u8],
    scalar_kernel: &str,
    micro_hsaco: &[u8],
    micro_kernel: &str,
    row_fn: impl Fn(&mut Rng, usize) -> Vec<u8> + Copy,
    block_bytes: usize,
    block_elems: usize,
    m: u32,
    n: u32,
) {
    let _device = Device::new(0).expect("failed to select device 0");
    let blocks_per_row = (n as usize) / block_elems;
    let mut rng = Rng::new(42);
    let w_bytes = build_w_bytes(row_fn, &mut rng, blocks_per_row, block_bytes, m as usize);
    let max_rows = 128u32;
    let x = make_x(max_rows, n);

    let scalar_probe = setup(
        scalar_hsaco,
        scalar_kernel,
        &w_bytes,
        &x,
        (max_rows * m) as usize,
        |rows, m| LaunchConfig {
            grid: (m, rows.div_ceil(SCALAR_TILE_ROWS), 1),
            block: (32, SCALAR_WARPS_PER_BLOCK, 1),
            shared_mem_bytes: SCALAR_TILE_ELEMS * std::mem::size_of::<f32>() as u32,
        },
    );
    let micro_probe = setup(
        micro_hsaco,
        micro_kernel,
        &w_bytes,
        &x,
        (max_rows * m) as usize,
        |rows, m| LaunchConfig {
            grid: (m.div_ceil(MICRO_TILE_M), rows.div_ceil(MICRO_TILE_ROWS), 1),
            block: (32, MICRO_WARPS_PER_BLOCK, 1),
            shared_mem_bytes: 2
                * (MICRO_TILE_ROWS + MICRO_TILE_M)
                * (K_STAGE + LDS_PAD)
                * std::mem::size_of::<u16>() as u32,
        },
    );

    println!("--- {label} (m={m}, n={n}) ---");
    println!("rows | scalar us/call | micro us/call | micro speedup");
    for &rows in &[1u32, 2, 4, 8, 16, 32, 64, 127] {
        // Interleave rounds between the two kernels (not "all scalar then
        // all micro") to average out ambient GPU clock drift within one
        // process — same methodology `gemm_xwt_quant_wmma_perf.rs`'s module
        // doc documents needing.
        let mut scalar_us = f64::INFINITY;
        let mut micro_us = f64::INFINITY;
        for _ in 0..ROUNDS {
            scalar_us = scalar_us.min(time_launch(&scalar_probe, rows, m, n));
            micro_us = micro_us.min(time_launch(&micro_probe, rows, m, n));
        }
        println!(
            "{rows:4} | {scalar_us:14.2} | {micro_us:13.2} | {:.2}x",
            scalar_us / micro_us
        );
    }
}

#[test]
#[ignore]
fn moe_shape_gate_up_q4_k() {
    sweep_shape(
        "gate/up (Q4_K)",
        rocml_kernels::GEMM_XWT_Q4_K_HSACO,
        rocml_kernels::GEMM_XWT_Q4_K_KERNEL,
        rocml_kernels::GEMM_XWT_WMMA_Q4_K_MICRO_HSACO,
        rocml_kernels::GEMM_XWT_WMMA_Q4_K_MICRO_KERNEL,
        random_row_q4_k,
        Q4_K_BLOCK_BYTES,
        256,
        512,
        2048,
    );
}

/// Sweeps `m` at a fixed small `rows` (8, the measured MoE-group average)
/// to find where micro-WMMA's grid (`m.div_ceil(MICRO_TILE_M) *
/// rows.div_ceil(MICRO_TILE_ROWS)`) gets wide enough to beat scalar's grid
/// (`m` blocks, always wide) — `moe_shape_gate_up_q4_k`'s `m=512` (8 micro
/// blocks) loses badly (0.26x) while `moe_shape_down_q6_k`'s `m=2048` (32
/// blocks) wins (1.3-2.7x); this finds the crossover in between.
#[test]
#[ignore]
fn m_crossover_q4_k() {
    let _device = Device::new(0).expect("failed to select device 0");
    const N: u32 = 2048;
    const ROWS: u32 = 8;
    let blocks_per_row = (N as usize) / 256;
    println!("--- m crossover (Q4_K, n={N}, rows={ROWS}) ---");
    println!("m    | micro grid blocks | scalar us/call | micro us/call | micro speedup");
    for &m in &[512u32, 768, 1024, 1280, 1536, 1792, 2048] {
        let mut rng = Rng::new(42);
        let w_bytes = build_w_bytes(
            random_row_q4_k,
            &mut rng,
            blocks_per_row,
            Q4_K_BLOCK_BYTES,
            m as usize,
        );
        let x = make_x(ROWS, N);
        let scalar_probe = setup(
            rocml_kernels::GEMM_XWT_Q4_K_HSACO,
            rocml_kernels::GEMM_XWT_Q4_K_KERNEL,
            &w_bytes,
            &x,
            (ROWS * m) as usize,
            |rows, m| LaunchConfig {
                grid: (m, rows.div_ceil(SCALAR_TILE_ROWS), 1),
                block: (32, SCALAR_WARPS_PER_BLOCK, 1),
                shared_mem_bytes: SCALAR_TILE_ELEMS * std::mem::size_of::<f32>() as u32,
            },
        );
        let micro_probe = setup(
            rocml_kernels::GEMM_XWT_WMMA_Q4_K_MICRO_HSACO,
            rocml_kernels::GEMM_XWT_WMMA_Q4_K_MICRO_KERNEL,
            &w_bytes,
            &x,
            (ROWS * m) as usize,
            |rows, m| LaunchConfig {
                grid: (m.div_ceil(MICRO_TILE_M), rows.div_ceil(MICRO_TILE_ROWS), 1),
                block: (32, MICRO_WARPS_PER_BLOCK, 1),
                shared_mem_bytes: 2
                    * (MICRO_TILE_ROWS + MICRO_TILE_M)
                    * (K_STAGE + LDS_PAD)
                    * std::mem::size_of::<u16>() as u32,
            },
        );
        let blocks = m.div_ceil(MICRO_TILE_M) * ROWS.div_ceil(MICRO_TILE_ROWS);
        let mut scalar_us = f64::INFINITY;
        let mut micro_us = f64::INFINITY;
        for _ in 0..ROUNDS {
            scalar_us = scalar_us.min(time_launch(&scalar_probe, ROWS, m, N));
            micro_us = micro_us.min(time_launch(&micro_probe, ROWS, m, N));
        }
        println!(
            "{m:4} | {blocks:18} | {scalar_us:14.2} | {micro_us:13.2} | {:.2}x",
            scalar_us / micro_us
        );
    }
}

#[test]
#[ignore]
fn moe_shape_down_q6_k() {
    sweep_shape(
        "down (Q6_K)",
        rocml_kernels::GEMM_XWT_Q6_K_HSACO,
        rocml_kernels::GEMM_XWT_Q6_K_KERNEL,
        rocml_kernels::GEMM_XWT_WMMA_Q6_K_MICRO_HSACO,
        rocml_kernels::GEMM_XWT_WMMA_Q6_K_MICRO_KERNEL,
        random_row_q6_k,
        Q6_K_BLOCK_BYTES,
        256,
        2048,
        512,
    );
}
