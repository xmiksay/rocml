//! Code object loading and kernel launch.
use std::ffi::{c_void, CString};
use std::ptr;

use crate::device::Stream;
use crate::error::{check, HipError};
use crate::ffi;

/// A loaded HIP code object (an `.hsaco` blob, e.g. produced by `hipcc
/// --genco` and embedded via `include_bytes!`).
pub struct Module {
    handle: ffi::hipModule_t,
}

impl Module {
    /// Loads `bytes` as a code object. `bytes` must be a valid HIP/HSA code
    /// object image — passing arbitrary data is a HIP-runtime-level error,
    /// not a Rust safety violation (the runtime validates the container),
    /// but a malformed image can still crash the driver, so callers should
    /// only pass trusted, build-time-generated `.hsaco` bytes.
    pub fn load_from_bytes(bytes: &[u8]) -> Result<Self, HipError> {
        let mut handle: ffi::hipModule_t = ptr::null_mut();
        // SAFETY: hipModuleLoadData reads `bytes` synchronously during the
        // call and does not retain the pointer afterward, so it need not
        // outlive this call.
        check(unsafe { ffi::hipModuleLoadData(&mut handle, bytes.as_ptr() as *const c_void) })?;
        Ok(Self { handle })
    }

    /// Looks up a kernel symbol by its `extern "C"` name.
    pub fn get_function(&self, name: &str) -> Result<Function, HipError> {
        let cname = CString::new(name)?;
        let mut handle: ffi::hipFunction_t = ptr::null_mut();
        // SAFETY: self.handle is a live module (not yet unloaded, guaranteed
        // by the borrow), and cname is a valid NUL-terminated C string.
        check(unsafe { ffi::hipModuleGetFunction(&mut handle, self.handle, cname.as_ptr()) })?;
        Ok(Function { handle })
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: self.handle was produced by hipModuleLoadData in
            // `load_from_bytes` and Drop runs at most once per value.
            unsafe {
                let _ = ffi::hipModuleUnload(self.handle);
            }
        }
    }
}

/// Grid/block dimensions and dynamic shared memory for a kernel launch.
pub struct LaunchConfig {
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub shared_mem_bytes: u32,
}

/// A kernel entry point resolved from a [`Module`].
pub struct Function {
    handle: ffi::hipFunction_t,
}

impl Function {
    /// Launches this kernel on `stream` (or the default stream if `None`).
    ///
    /// # Safety
    /// The caller vouches that:
    /// - `params` has exactly one entry per parameter of the target
    ///   `extern "C" __global__` kernel, in declaration order, and each
    ///   entry is a pointer *to* a value of the matching argument type/size
    ///   (e.g. a `*const f32` argument needs an entry that is the address of
    ///   a local holding that `*const f32`, not the pointee) — see
    ///   [`kernel_params!`] for a helper that builds this correctly.
    /// - Any device pointers embedded among the argument values stay valid
    ///   until the launch completes on `stream`.
    /// - `stream`, if given, is not destroyed before the launch completes.
    pub unsafe fn launch(
        &self,
        cfg: &LaunchConfig,
        params: &mut [*mut c_void],
        stream: Option<&Stream>,
    ) -> Result<(), HipError> {
        let stream_handle = stream.map(Stream::handle).unwrap_or(ptr::null_mut());
        // SAFETY: forwarding the caller's own contract (documented above)
        // straight to hipModuleLaunchKernel; this function adds no
        // additional guarantees beyond passing the arguments through.
        check(unsafe {
            ffi::hipModuleLaunchKernel(
                self.handle,
                cfg.grid.0,
                cfg.grid.1,
                cfg.grid.2,
                cfg.block.0,
                cfg.block.1,
                cfg.block.2,
                cfg.shared_mem_bytes,
                stream_handle,
                params.as_mut_ptr(),
                ptr::null_mut(),
            )
        })
    }
}

/// Builds a `hipModuleLaunchKernel` params array from local variables, taking
/// the address of each argument in place (each entry must point *to* the
/// argument value, per HIP's kernel-params convention).
///
/// ```ignore
/// let n: u32 = 1024;
/// let mut params = kernel_params!(a_ptr, b_ptr, out_ptr, n);
/// unsafe { function.launch(&cfg, &mut params, None)? };
/// ```
#[macro_export]
macro_rules! kernel_params {
    ($($arg:expr),+ $(,)?) => {
        [$(::std::ptr::addr_of!($arg) as *mut ::std::ffi::c_void),+]
    };
}
