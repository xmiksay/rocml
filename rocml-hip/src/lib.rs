//! Minimal safe wrapper around the HIP runtime C API, hand-written against
//! ROCm's `hip_runtime_api.h` (no bindgen). Provides just enough — device
//! selection, streams, device buffers, and code-object loading/launch — for
//! `rocml-kernels` and later crates to run hand-written HIP kernels without
//! any of rocml touching the raw FFI surface directly.
//!
//! Requires a real ROCm install and GPU at runtime; this crate only declares
//! the C ABI and does not simulate or mock the driver.

pub mod buffer;
pub mod device;
pub mod error;
pub mod event;
pub mod ffi;
pub mod module;

pub use buffer::DeviceBuffer;
pub use device::{Device, MemoryInfo, Stream};
pub use error::HipError;
pub use event::{elapsed_ms, Event};
pub use module::{Function, LaunchConfig, Module};
