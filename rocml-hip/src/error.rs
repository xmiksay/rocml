//! HIP error handling: every `hipError_t` the runtime returns gets wrapped
//! into a [`HipError`] carrying both the numeric code and HIP's own message,
//! rather than surfacing a bare integer.
use std::ffi::CStr;

use crate::ffi;

#[derive(Debug, thiserror::Error)]
pub enum HipError {
    /// A HIP runtime call returned a non-success `hipError_t`.
    #[error("HIP error {code}: {message}")]
    Runtime { code: i32, message: String },
    /// A buffer/host slice length didn't match what an operation expected.
    #[error("length mismatch: expected {expected} elements, got {actual}")]
    LengthMismatch { expected: usize, actual: usize },
    /// `len * size_of::<T>()` overflowed `usize` while sizing an allocation.
    #[error("allocation size overflow: {len} elements of {elem_size} bytes each")]
    AllocationOverflow { len: usize, elem_size: usize },
    /// A name (e.g. a kernel symbol) contained an interior NUL and can't
    /// become a C string.
    #[error("invalid C string: {0}")]
    InvalidCString(#[from] std::ffi::NulError),
}

/// Turns a raw `hipError_t` into `Ok(())` or a [`HipError::Runtime`] carrying
/// `hipGetErrorString`'s message.
///
/// # Safety
/// `code` must be a value HIP itself returned from a call on this thread;
/// `hipGetErrorString` is safe to call with any `hipError_t` value, so this
/// function has no additional caller obligations beyond that.
pub fn check(code: ffi::hipError_t) -> Result<(), HipError> {
    if code == ffi::hipSuccess {
        return Ok(());
    }
    // SAFETY: hipGetErrorString returns a pointer to a static, NUL-terminated
    // string table owned by the HIP runtime for any hipError_t value,
    // including out-of-range ones (it falls back to "unrecognized error").
    let message = unsafe {
        let ptr = ffi::hipGetErrorString(code);
        if ptr.is_null() {
            "hipGetErrorString returned a null pointer".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };
    Err(HipError::Runtime { code, message })
}
