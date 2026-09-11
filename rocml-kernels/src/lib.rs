//! Embedded HIP kernel code objects. `build.rs` compiles every `kernels/*.hip`
//! file to a `.hsaco` code object via `hipcc --genco`, and this crate pulls
//! each one in with `include_bytes!` so the compiled binary carries the
//! kernels directly — no runtime dependency on `hipcc` or the `kernels/`
//! sources being present on the machine that runs it.

/// `kernels/smoke.hip`: elementwise `out[i] = a[i] + b[i]`.
pub const VEC_ADD_F32_HSACO: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/smoke.hsaco"));
pub const VEC_ADD_F32_KERNEL: &str = "vec_add_f32";

/// `kernels/gemv.hip`: naive row-per-block `y = mat * x` (mat is row-major
/// m x n). Must be launched with a power-of-two block size.
pub const GEMV_F32_HSACO: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemv.hsaco"));
pub const GEMV_F32_KERNEL: &str = "gemv_f32";

/// `kernels/gemv_f16.hip`: decode-path `y = W * x` (W is row-major m x n f16,
/// x/y f32, f32 accumulation). One block per output row; block size must be
/// a power of two (shared-memory tree reduction).
pub const GEMV_F16_HSACO: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemv_f16.hsaco"));
pub const GEMV_F16_KERNEL: &str = "gemv_f16";

/// `kernels/gemm_f16.hip`: prefill-path linear layer `out = x * W^T` (x is
/// rows x n f32, W is row-major m x n f16, out is rows x m f32). Fixed
/// 16x16 shared-memory tile; must be launched with block = (16, 16, 1).
pub const GEMM_XWT_F16_HSACO: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_f16.hsaco"));
pub const GEMM_XWT_F16_KERNEL: &str = "gemm_xwt_f16";
