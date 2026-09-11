//! RAII device memory buffer.
use std::ffi::c_void;
use std::marker::PhantomData;
use std::mem;
use std::ptr;

use crate::error::{check, HipError};
use crate::ffi;

/// A `hipMalloc`-backed device allocation of `len` elements of `T`.
///
/// `T: Copy` is required because the buffer is moved to/from the device by
/// raw byte copy (`hipMemcpy`) — types with drop glue or non-`Copy`
/// invariants would be silently bit-copied, which is unsound for them.
pub struct DeviceBuffer<T: Copy> {
    ptr: *mut c_void,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T: Copy> DeviceBuffer<T> {
    /// Allocates room for `len` elements of `T` on the current device.
    pub fn new(len: usize) -> Result<Self, HipError> {
        let elem_size = mem::size_of::<T>();
        let bytes = len
            .checked_mul(elem_size)
            .ok_or(HipError::AllocationOverflow { len, elem_size })?;

        let mut ptr: *mut c_void = ptr::null_mut();
        if bytes > 0 {
            // SAFETY: `ptr` is a valid out-param location; `bytes` is the
            // exact, overflow-checked size of the requested allocation.
            check(unsafe { ffi::hipMalloc(&mut ptr, bytes) })?;
        }
        Ok(Self {
            ptr,
            len,
            _marker: PhantomData,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Copies `data` to this buffer. Errors (rather than panics) if the
    /// lengths don't match.
    pub fn copy_from_host(&mut self, data: &[T]) -> Result<(), HipError> {
        if data.len() != self.len {
            return Err(HipError::LengthMismatch {
                expected: self.len,
                actual: data.len(),
            });
        }
        if self.len == 0 {
            return Ok(());
        }
        let bytes = self.len * mem::size_of::<T>();
        // SAFETY: `self.ptr` is a live device allocation of exactly `bytes`
        // bytes (invariant maintained by `new` and never resized), and
        // `data` is a host slice of the same byte length just checked above.
        check(unsafe {
            ffi::hipMemcpy(
                self.ptr,
                data.as_ptr() as *const c_void,
                bytes,
                ffi::hip_memcpy_host_to_device,
            )
        })
    }

    /// Copies this buffer to `data`. Errors (rather than panics) if the
    /// lengths don't match.
    pub fn copy_to_host(&self, data: &mut [T]) -> Result<(), HipError> {
        if data.len() != self.len {
            return Err(HipError::LengthMismatch {
                expected: self.len,
                actual: data.len(),
            });
        }
        if self.len == 0 {
            return Ok(());
        }
        let bytes = self.len * mem::size_of::<T>();
        // SAFETY: symmetric to copy_from_host — self.ptr holds `bytes` valid
        // device bytes, and `data` is a distinct host slice of that same length.
        check(unsafe {
            ffi::hipMemcpy(
                data.as_mut_ptr() as *mut c_void,
                self.ptr,
                bytes,
                ffi::hip_memcpy_device_to_host,
            )
        })
    }

    /// Raw device pointer, valid for use as a kernel launch argument (via
    /// [`crate::kernel_params!`]) as long as `self` is not dropped and no
    /// concurrent host access races the kernel's device-side access.
    pub fn device_ptr(&self) -> *mut c_void {
        self.ptr
    }
}

impl<T: Copy> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: `self.ptr` was allocated by `hipMalloc` in `new` and
            // Drop runs at most once per value, so this can't double-free.
            unsafe {
                let _ = ffi::hipFree(self.ptr);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;

    #[test]
    fn buffer_roundtrip_f32() {
        let _device = Device::new(0).expect("failed to select device 0");
        let input: Vec<f32> = (0..1024).map(|i| i as f32 * 0.5).collect();
        let mut buffer = DeviceBuffer::<f32>::new(input.len()).expect("hipMalloc failed");
        buffer
            .copy_from_host(&input)
            .expect("copy_from_host failed");

        let mut output = vec![0.0f32; input.len()];
        buffer
            .copy_to_host(&mut output)
            .expect("copy_to_host failed");
        assert_eq!(input, output);
    }

    #[test]
    fn buffer_size_mismatch_is_err() {
        let _device = Device::new(0).expect("failed to select device 0");
        let mut buffer = DeviceBuffer::<f32>::new(16).expect("hipMalloc failed");

        let too_short = vec![0.0f32; 4];
        assert!(buffer.copy_from_host(&too_short).is_err());

        let mut too_short_out = vec![0.0f32; 4];
        assert!(buffer.copy_to_host(&mut too_short_out).is_err());
    }
}
