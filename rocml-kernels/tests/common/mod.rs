//! Shared support for the fused dequant-GEMV integration tests
//! (`gemv_q8_0`/`gemv_q4_k`/`gemv_q5_k`/`gemv_q6_k`): a small deterministic
//! PRNG for synthetic block bytes, one row-builder per quant format (byte
//! layouts ported from `rocml-core/src/quant/*.rs`), and the shared GPU
//! launch/compare plumbing.
#![allow(dead_code)] // each test binary only exercises a subset of this module

use std::ffi::c_void;

use half::f16;
use rocml_core::quant::{dequantize, GgmlDType};
use rocml_hip::{kernel_params, Device, DeviceBuffer, LaunchConfig, Module};

/// Compares GPU output against the CPU reference with a relative tolerance
/// floored at 1.0 (i.e. `2e-3` absolute for near-zero elements) — long f32
/// accumulations over many strided blocks reorder additions relative to the
/// CPU reference, so a pure relative check on near-zero values is too tight.
pub fn assert_close(actual: &[f32], expected: &[f32], label: &str) {
    const REL_TOL: f32 = 2e-3;
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (got, want)) in actual.iter().zip(expected).enumerate() {
        let diff = (got - want).abs();
        let tol = REL_TOL * want.abs().max(1.0);
        assert!(
            diff <= tol,
            "{label}[{i}]: got {got}, want {want} (diff {diff}, tol {tol})"
        );
    }
}

/// xorshift32: small, seedable, dependency-free PRNG for synthetic test
/// data (the workspace only permits adding `rocml-core` as a dev-dependency
/// here, so no `rand` crate).
pub struct Rng(u32);

impl Rng {
    pub fn new(seed: u32) -> Self {
        Self(seed | 1)
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }

    pub fn next_u8(&mut self) -> u8 {
        (self.next_u32() >> 24) as u8
    }

    /// f16 scale field in `[lo, hi]`, little-endian bytes as stored on disk.
    /// Kept small and strictly positive so `d`/`dmin` never blow up the
    /// dequantized magnitude or (for `dmin`) go negative.
    pub fn next_f16_le(&mut self, lo: f32, hi: f32) -> [u8; 2] {
        let u = self.next_u32() as f32 / u32::MAX as f32;
        half::f16::from_f32(lo + u * (hi - lo)).to_le_bytes()
    }
}

const SCALE_LO: f32 = 0.001;
const SCALE_HI: f32 = 1.0;

pub const Q8_0_BLOCK_BYTES: usize = 34;
pub const Q8_0_BLOCK_ELEMS: usize = 32;
pub const K_BLOCK_ELEMS: usize = 256;
pub const Q4_K_BLOCK_BYTES: usize = 144;
pub const Q5_K_BLOCK_BYTES: usize = 176;
pub const Q6_K_BLOCK_BYTES: usize = 210;

/// One random Q8_0 row: `blocks_per_row` blocks of (f16 `d`, 32 arbitrary
/// i8 codes). Any code byte is finite by construction (it's just an integer).
pub fn random_row_q8_0(rng: &mut Rng, blocks_per_row: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(blocks_per_row * Q8_0_BLOCK_BYTES);
    for _ in 0..blocks_per_row {
        bytes.extend_from_slice(&rng.next_f16_le(SCALE_LO, SCALE_HI));
        bytes.extend((0..32).map(|_| rng.next_u8()));
    }
    bytes
}

/// One random Q4_K row: `d`, `dmin`, a 12-byte packed 6-bit scale/min table,
/// then 128 bytes of 4-bit codes — all arbitrary except the two f16 fields.
pub fn random_row_q4_k(rng: &mut Rng, blocks_per_row: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(blocks_per_row * Q4_K_BLOCK_BYTES);
    for _ in 0..blocks_per_row {
        bytes.extend_from_slice(&rng.next_f16_le(SCALE_LO, SCALE_HI)); // d
        bytes.extend_from_slice(&rng.next_f16_le(SCALE_LO, SCALE_HI)); // dmin
        bytes.extend((0..12).map(|_| rng.next_u8())); // packed scale/min table
        bytes.extend((0..128).map(|_| rng.next_u8())); // qs
    }
    bytes
}

/// Like [`random_row_q4_k`] plus the 32-byte high-bit plane (`qh`).
pub fn random_row_q5_k(rng: &mut Rng, blocks_per_row: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(blocks_per_row * Q5_K_BLOCK_BYTES);
    for _ in 0..blocks_per_row {
        bytes.extend_from_slice(&rng.next_f16_le(SCALE_LO, SCALE_HI)); // d
        bytes.extend_from_slice(&rng.next_f16_le(SCALE_LO, SCALE_HI)); // dmin
        bytes.extend((0..12).map(|_| rng.next_u8())); // packed scale/min table
        bytes.extend((0..32).map(|_| rng.next_u8())); // qh
        bytes.extend((0..128).map(|_| rng.next_u8())); // qs
    }
    bytes
}

/// One random Q6_K row: 128 bytes of low nibbles, 64 bytes of 2-bit high
/// codes, 16 signed 8-bit per-sub-block scales (`next_u8` spans the full
/// byte range, so this already exercises scales >= 0x80, i.e. negative
/// values — important, since that sign bit was once dropped by a bug in
/// rocml-core's dequant), then the f16 `d`.
pub fn random_row_q6_k(rng: &mut Rng, blocks_per_row: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(blocks_per_row * Q6_K_BLOCK_BYTES);
    for _ in 0..blocks_per_row {
        bytes.extend((0..128).map(|_| rng.next_u8())); // ql
        bytes.extend((0..64).map(|_| rng.next_u8())); // qh
        bytes.extend((0..16).map(|_| rng.next_u8())); // scales
        bytes.extend_from_slice(&rng.next_f16_le(SCALE_LO, SCALE_HI)); // d
    }
    bytes
}

/// CPU reference: dequantize each row with rocml-core's proven-correct
/// dequant and dot it against `x`.
pub fn expected_gemv(
    dtype: GgmlDType,
    w_bytes: &[u8],
    x: &[f32],
    m: usize,
    row_bytes: usize,
) -> Vec<f32> {
    (0..m)
        .map(|row| {
            let row_slice = &w_bytes[row * row_bytes..(row + 1) * row_bytes];
            let deq = dequantize(dtype, row_slice).expect("cpu dequantize failed");
            deq.iter().zip(x).map(|(a, b)| a * b).sum()
        })
        .collect()
}

/// CPU reference for the batched `gemm_xwt_q*` kernels: dequantize each of
/// `m` rows and dot it against each of `rows` `x` rows, `out[rows,m]`.
pub fn expected_gemm(
    dtype: GgmlDType,
    w_bytes: &[u8],
    x: &[f32],
    rows: usize,
    m: usize,
    n: usize,
) -> Vec<f32> {
    let row_bytes = w_bytes.len() / m;
    let deq_rows: Vec<Vec<f32>> = (0..m)
        .map(|row| {
            let row_slice = &w_bytes[row * row_bytes..(row + 1) * row_bytes];
            dequantize(dtype, row_slice).expect("cpu dequantize failed")
        })
        .collect();
    let mut out = vec![0.0f32; rows * m];
    for r in 0..rows {
        let x_row = &x[r * n..(r + 1) * n];
        for (col, deq) in deq_rows.iter().enumerate() {
            out[r * m + col] = deq.iter().zip(x_row).map(|(a, b)| a * b).sum();
        }
    }
    out
}

/// Uploads `w_bytes`/`x`, launches `gemm_xwt_<type>` (`kernel_name` from
/// `hsaco`) as `gemm_xwt_<type>(const float* x, const void* w, float* out,
/// unsigned rows, unsigned m, unsigned n)` with the fixed block=(32,8,1)/
/// 1KiB-shared-mem contract `gemm_xwt_quant.hip` requires, and downloads
/// `out` (`rows x m`).
pub fn run_gemm_quant_kernel(
    hsaco: &[u8],
    kernel_name: &str,
    w_bytes: &[u8],
    x: &[f32],
    rows: u32,
    m: u32,
    n: u32,
) -> Vec<f32> {
    let _device = Device::new(0).expect("failed to select device 0");
    let module = Module::load_from_bytes(hsaco).expect("module load failed");
    let function = module
        .get_function(kernel_name)
        .expect("kernel lookup failed");

    let mut buf_x = DeviceBuffer::<f32>::new(x.len()).expect("hipMalloc x failed");
    let mut buf_w = DeviceBuffer::<u8>::new(w_bytes.len()).expect("hipMalloc w failed");
    let buf_out = DeviceBuffer::<f32>::new((rows * m) as usize).expect("hipMalloc out failed");
    buf_x.copy_from_host(x).expect("copy x failed");
    buf_w.copy_from_host(w_bytes).expect("copy w failed");

    let x_ptr: *mut c_void = buf_x.device_ptr();
    let w_ptr: *mut c_void = buf_w.device_ptr();
    let out_ptr: *mut c_void = buf_out.device_ptr();
    let mut params = kernel_params!(x_ptr, w_ptr, out_ptr, rows, m, n);

    const WARPS_PER_BLOCK: u32 = 8;
    const ROWS_PER_WARP: u32 = 8;
    const TILE_ROWS: u32 = WARPS_PER_BLOCK * ROWS_PER_WARP;
    const TILE_ELEMS: u32 = 256;
    let cfg = LaunchConfig {
        grid: (m, rows.div_ceil(TILE_ROWS), 1),
        block: (32, WARPS_PER_BLOCK, 1),
        shared_mem_bytes: TILE_ELEMS * std::mem::size_of::<f32>() as u32,
    };
    // SAFETY: params matches every gemm_xwt_<type> kernel's parameter list
    // (const float*, const void*, float*, unsigned x3) in order, and all
    // device buffers outlive this launch. Block = (32, 8, 1) matches the
    // kernel's fixed warp-per-`ROWS_PER_WARP`-rows tiling.
    unsafe { function.launch(&cfg, &mut params, None) }.expect("kernel launch failed");

    let mut actual = vec![0.0f32; (rows * m) as usize];
    buf_out.copy_to_host(&mut actual).expect("copy out failed");
    actual
}

/// CPU reference for the WMMA `gemm_xwt_wmma_q*` kernels: like
/// [`expected_gemm`], but rounds both the dequantized weight and the `x`
/// operand through f16 before each product (f32 accumulation), matching
/// what the WMMA matrix unit actually computes from f16 fragments — issue
/// #6's accepted numerics (X and dequantized-W input rounding, full f32
/// accumulate). Comparing against the *unrounded* f32 reference instead
/// would only be valid to a couple of percent for the K-quants' synthetic
/// test weights here (their scale bytes span the full signed-byte range, so
/// dequantized magnitudes run into the thousands — real GGUF-calibrated
/// scales don't, but this harness's RNG doesn't know that): a debug probe
/// against real model data confirmed the WMMA kernel matches this
/// f16-rounded reference to ~1e-6 relative, while the *unrounded* f32
/// reference can differ by several units at these synthetic magnitudes from
/// input rounding alone, not a kernel bug.
pub fn expected_gemm_wmma(
    dtype: GgmlDType,
    w_bytes: &[u8],
    x: &[f32],
    rows: usize,
    m: usize,
    n: usize,
) -> Vec<f32> {
    let row_bytes = w_bytes.len() / m;
    let deq_rows_f16: Vec<Vec<f32>> = (0..m)
        .map(|row| {
            let row_slice = &w_bytes[row * row_bytes..(row + 1) * row_bytes];
            dequantize(dtype, row_slice)
                .expect("cpu dequantize failed")
                .iter()
                .map(|&v| f16::from_f32(v).to_f32())
                .collect()
        })
        .collect();
    let x_f16: Vec<f32> = x.iter().map(|&v| f16::from_f32(v).to_f32()).collect();
    let mut out = vec![0.0f32; rows * m];
    for r in 0..rows {
        let x_row = &x_f16[r * n..(r + 1) * n];
        for (col, deq) in deq_rows_f16.iter().enumerate() {
            out[r * m + col] = deq.iter().zip(x_row).map(|(a, b)| a * b).sum();
        }
    }
    out
}

/// Uploads `w_bytes`/`x`, launches `gemm_xwt_wmma_<type>` (`kernel_name` from
/// `hsaco`) with the fixed `TILE_ROWS=128`/`TILE_M=64`/`K_STAGE=16` tiling
/// `gemm_xwt_quant_wmma.hip` requires (block=(32,16,1), grid=
/// `(ceil(m/64), ceil(rows/128), 1)`, `(128+64)*16*sizeof(f16)` bytes of
/// dynamic shared memory), and downloads `out` (`rows x m`).
pub fn run_gemm_wmma_kernel(
    hsaco: &[u8],
    kernel_name: &str,
    w_bytes: &[u8],
    x: &[f32],
    rows: u32,
    m: u32,
    n: u32,
) -> Vec<f32> {
    let _device = Device::new(0).expect("failed to select device 0");
    let module = Module::load_from_bytes(hsaco).expect("module load failed");
    let function = module
        .get_function(kernel_name)
        .expect("kernel lookup failed");

    let mut buf_x = DeviceBuffer::<f32>::new(x.len()).expect("hipMalloc x failed");
    let mut buf_w = DeviceBuffer::<u8>::new(w_bytes.len()).expect("hipMalloc w failed");
    let buf_out = DeviceBuffer::<f32>::new((rows * m) as usize).expect("hipMalloc out failed");
    buf_x.copy_from_host(x).expect("copy x failed");
    buf_w.copy_from_host(w_bytes).expect("copy w failed");

    let x_ptr: *mut c_void = buf_x.device_ptr();
    let w_ptr: *mut c_void = buf_w.device_ptr();
    let out_ptr: *mut c_void = buf_out.device_ptr();
    let mut params = kernel_params!(x_ptr, w_ptr, out_ptr, rows, m, n);

    const TILE_ROWS: u32 = 128;
    const TILE_M: u32 = 64;
    const K_STAGE: u32 = 16;
    const WARPS_PER_BLOCK: u32 = 16;
    let cfg = LaunchConfig {
        grid: (m.div_ceil(TILE_M), rows.div_ceil(TILE_ROWS), 1),
        block: (32, WARPS_PER_BLOCK, 1),
        shared_mem_bytes: (TILE_ROWS + TILE_M) * K_STAGE * std::mem::size_of::<u16>() as u32,
    };
    // SAFETY: params matches every gemm_xwt_wmma_<type> kernel's parameter
    // list (const float*, const void*, float*, unsigned x3) in order, and
    // all device buffers outlive this launch. block/grid/shared_mem_bytes
    // match the kernel's fixed TILE_ROWS/TILE_M/K_STAGE tiling.
    unsafe { function.launch(&cfg, &mut params, None) }.expect("kernel launch failed");

    let mut actual = vec![0.0f32; (rows * m) as usize];
    buf_out.copy_to_host(&mut actual).expect("copy out failed");
    actual
}

/// Uploads `w_bytes`/`x`, launches `kernel_name` from `hsaco` as
/// `gemv_<type>(const void* w, const float* x, float* y, unsigned m, unsigned n)`
/// with one 128-thread workgroup per output row, and downloads `y`.
pub fn run_gemv_kernel(
    hsaco: &[u8],
    kernel_name: &str,
    w_bytes: &[u8],
    x: &[f32],
    m: u32,
    n: u32,
) -> Vec<f32> {
    let _device = Device::new(0).expect("failed to select device 0");
    let module = Module::load_from_bytes(hsaco).expect("module load failed");
    let function = module
        .get_function(kernel_name)
        .expect("kernel lookup failed");

    let mut buf_w = DeviceBuffer::<u8>::new(w_bytes.len()).expect("hipMalloc w failed");
    let mut buf_x = DeviceBuffer::<f32>::new(x.len()).expect("hipMalloc x failed");
    let buf_y = DeviceBuffer::<f32>::new(m as usize).expect("hipMalloc y failed");
    buf_w.copy_from_host(w_bytes).expect("copy w failed");
    buf_x.copy_from_host(x).expect("copy x failed");

    let w_ptr: *mut c_void = buf_w.device_ptr();
    let x_ptr: *mut c_void = buf_x.device_ptr();
    let y_ptr: *mut c_void = buf_y.device_ptr();
    let mut params = kernel_params!(w_ptr, x_ptr, y_ptr, m, n);

    let block = 128u32;
    let cfg = LaunchConfig {
        grid: (m, 1, 1),
        block: (block, 1, 1),
        shared_mem_bytes: block * std::mem::size_of::<f32>() as u32,
    };
    // SAFETY: params matches every gemv_<type> kernel's parameter list
    // (const void*, const float*, float*, unsigned, unsigned) in order, and
    // all device buffers outlive this launch.
    unsafe { function.launch(&cfg, &mut params, None) }.expect("kernel launch failed");

    let mut actual = vec![0.0f32; m as usize];
    buf_y.copy_to_host(&mut actual).expect("copy y failed");
    actual
}
