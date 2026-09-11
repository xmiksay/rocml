//! Raw `extern "C"` declarations for the slice of the HIP runtime API rocml
//! needs. Hand-written against `/opt/rocm/include/hip/hip_runtime_api.h` and
//! `driver_types.h` (ROCm 7.2) — no bindgen. Keep this file to declarations
//! and the constants that back them; all safety lives in the wrapper modules.
#![allow(non_camel_case_types, non_upper_case_globals)]

use std::ffi::{c_char, c_int, c_void};

/// `hipError_t` is a C enum, which is `int`-sized on every platform HIP ships for.
pub type hipError_t = c_int;

pub const hipSuccess: hipError_t = 0;

/// Opaque handles — HIP only ever hands these back and forth as pointers.
pub type hipStream_t = *mut c_void;
pub type hipModule_t = *mut c_void;
pub type hipFunction_t = *mut c_void;

pub type hipMemcpyKind = c_int;
pub const hip_memcpy_host_to_host: hipMemcpyKind = 0;
pub const hip_memcpy_host_to_device: hipMemcpyKind = 1;
pub const hip_memcpy_device_to_host: hipMemcpyKind = 2;
pub const hip_memcpy_device_to_device: hipMemcpyKind = 3;
pub const hip_memcpy_default: hipMemcpyKind = 4;

extern "C" {
    // -- device management --
    pub fn hipGetDeviceCount(count: *mut c_int) -> hipError_t;
    pub fn hipSetDevice(device_id: c_int) -> hipError_t;
    /// `device` is a plain ordinal in [0, hipGetDeviceCount()) — HIP's
    /// `hipDevice_t` is `typedef int hipDevice_t`, no separate hipDeviceGet needed.
    pub fn hipDeviceGetName(name: *mut c_char, len: c_int, device: c_int) -> hipError_t;
    pub fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> hipError_t;
    pub fn hipDeviceSynchronize() -> hipError_t;

    // -- errors --
    pub fn hipGetErrorString(error: hipError_t) -> *const c_char;
    pub fn hipGetLastError() -> hipError_t;

    // -- memory --
    pub fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> hipError_t;
    pub fn hipFree(ptr: *mut c_void) -> hipError_t;
    pub fn hipMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        size_bytes: usize,
        kind: hipMemcpyKind,
    ) -> hipError_t;
    pub fn hipMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        size_bytes: usize,
        kind: hipMemcpyKind,
        stream: hipStream_t,
    ) -> hipError_t;

    // -- streams --
    pub fn hipStreamCreate(stream: *mut hipStream_t) -> hipError_t;
    pub fn hipStreamDestroy(stream: hipStream_t) -> hipError_t;
    pub fn hipStreamSynchronize(stream: hipStream_t) -> hipError_t;

    // -- modules / kernels --
    pub fn hipModuleLoadData(module: *mut hipModule_t, image: *const c_void) -> hipError_t;
    pub fn hipModuleUnload(module: hipModule_t) -> hipError_t;
    pub fn hipModuleGetFunction(
        function: *mut hipFunction_t,
        module: hipModule_t,
        name: *const c_char,
    ) -> hipError_t;
    #[allow(clippy::too_many_arguments)]
    pub fn hipModuleLaunchKernel(
        f: hipFunction_t,
        grid_dim_x: u32,
        grid_dim_y: u32,
        grid_dim_z: u32,
        block_dim_x: u32,
        block_dim_y: u32,
        block_dim_z: u32,
        shared_mem_bytes: u32,
        stream: hipStream_t,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> hipError_t;
}
