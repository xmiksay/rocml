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

    /// Copies `data` into this buffer's `[0, data.len())` prefix — unlike
    /// [`Self::copy_from_host`], `data` may be shorter than the buffer
    /// (never longer). Used by fixed-capacity scratch buffers that serve a
    /// variable-length chunk (e.g. `ChunkScratch::token_ids`, sized for
    /// `CHUNK_CAP` but fed anywhere from 1 to `CHUNK_CAP` tokens per call).
    pub fn copy_prefix_from_host(&mut self, data: &[T]) -> Result<(), HipError> {
        if data.len() > self.len {
            return Err(HipError::LengthMismatch {
                expected: self.len,
                actual: data.len(),
            });
        }
        if data.is_empty() {
            return Ok(());
        }
        let bytes = mem::size_of_val(data);
        // SAFETY: `data.len() <= self.len` was just checked, so `bytes` is
        // within `self.ptr`'s live `hipMalloc` allocation; `data` is a host
        // slice of that same byte length.
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

    /// Copies `data.len()` elements starting at element `offset` in this
    /// buffer out to `data` — the D2H counterpart of [`Self::copy_from_device`]'s
    /// range-based addressing, used by the snapshot layer to read back only
    /// a plane's *filled* prefix (`[0, N)`) instead of its whole allocated
    /// capacity. Errors (rather than panics) if `[offset, offset+data.len())`
    /// would run past this buffer.
    pub fn copy_range_to_host(&self, offset: usize, data: &mut [T]) -> Result<(), HipError> {
        let end = offset
            .checked_add(data.len())
            .ok_or(HipError::LengthMismatch {
                expected: self.len,
                actual: usize::MAX,
            })?;
        if end > self.len {
            return Err(HipError::LengthMismatch {
                expected: self.len,
                actual: end,
            });
        }
        if data.is_empty() {
            return Ok(());
        }
        let elem_size = mem::size_of::<T>();
        // SAFETY: bounds were just checked against this buffer's own live
        // `hipMalloc` allocation; `data` is a distinct host slice sized to
        // exactly the bytes being copied.
        check(unsafe {
            ffi::hipMemcpy(
                data.as_mut_ptr() as *mut c_void,
                (self.ptr as *const u8).add(offset * elem_size) as *const c_void,
                mem::size_of_val(data),
                ffi::hip_memcpy_device_to_host,
            )
        })
    }

    /// Writes `data` into this buffer starting at element `offset` — the H2D
    /// counterpart of [`Self::copy_range_to_host`], used by snapshot restore
    /// to write a captured plane's prefix back without requiring the host
    /// slice to cover the buffer's whole capacity. Errors (rather than
    /// panics) if `[offset, offset+data.len())` would run past this buffer.
    pub fn copy_range_from_host(&mut self, offset: usize, data: &[T]) -> Result<(), HipError> {
        let end = offset
            .checked_add(data.len())
            .ok_or(HipError::LengthMismatch {
                expected: self.len,
                actual: usize::MAX,
            })?;
        if end > self.len {
            return Err(HipError::LengthMismatch {
                expected: self.len,
                actual: end,
            });
        }
        if data.is_empty() {
            return Ok(());
        }
        let elem_size = mem::size_of::<T>();
        // SAFETY: symmetric to copy_range_to_host.
        check(unsafe {
            ffi::hipMemcpy(
                (self.ptr as *mut u8).add(offset * elem_size) as *mut c_void,
                data.as_ptr() as *const c_void,
                mem::size_of_val(data),
                ffi::hip_memcpy_host_to_device,
            )
        })
    }

    /// Raw device pointer, valid for use as a kernel launch argument (via
    /// [`crate::kernel_params!`]) as long as `self` is not dropped and no
    /// concurrent host access races the kernel's device-side access.
    pub fn device_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// Copies `len` elements from `src[src_offset..]` into
    /// `self[dst_offset..]`, entirely on-device (`hipMemcpy` with
    /// `hipMemcpyDeviceToDevice`). Used e.g. by a KV cache appending one
    /// time step's row into the middle of a larger per-layer buffer without
    /// a host round-trip. Errors (rather than panics) if either range would
    /// run past its buffer's length.
    pub fn copy_from_device(
        &mut self,
        dst_offset: usize,
        src: &DeviceBuffer<T>,
        src_offset: usize,
        len: usize,
    ) -> Result<(), HipError> {
        let dst_end = dst_offset
            .checked_add(len)
            .ok_or(HipError::LengthMismatch {
                expected: self.len,
                actual: usize::MAX,
            })?;
        let src_end = src_offset
            .checked_add(len)
            .ok_or(HipError::LengthMismatch {
                expected: src.len,
                actual: usize::MAX,
            })?;
        if dst_end > self.len {
            return Err(HipError::LengthMismatch {
                expected: self.len,
                actual: dst_end,
            });
        }
        if src_end > src.len {
            return Err(HipError::LengthMismatch {
                expected: src.len,
                actual: src_end,
            });
        }
        if len == 0 {
            return Ok(());
        }
        let elem_size = mem::size_of::<T>();
        let bytes = len * elem_size;
        // SAFETY: bounds were just checked against each buffer's own live
        // `hipMalloc` allocation, and the byte offsets stay within those
        // allocations by construction.
        check(unsafe {
            ffi::hipMemcpy(
                (self.ptr as *mut u8).add(dst_offset * elem_size) as *mut c_void,
                (src.ptr as *const u8).add(src_offset * elem_size) as *const c_void,
                bytes,
                ffi::hip_memcpy_device_to_device,
            )
        })
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
    fn copy_from_device_writes_offset_slice() {
        let _device = Device::new(0).expect("failed to select device 0");
        let src_data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut src = DeviceBuffer::<f32>::new(src_data.len()).expect("hipMalloc src failed");
        src.copy_from_host(&src_data)
            .expect("copy_from_host failed");

        let mut dst = DeviceBuffer::<f32>::new(32).expect("hipMalloc dst failed");
        dst.copy_from_host(&[-1.0f32; 32])
            .expect("copy_from_host failed");
        dst.copy_from_device(10, &src, 4, 6)
            .expect("copy_from_device failed");

        let mut out = vec![0.0f32; 32];
        dst.copy_to_host(&mut out).expect("copy_to_host failed");
        assert_eq!(&out[10..16], &src_data[4..10]);
        assert!(out[..10].iter().all(|&v| v == -1.0));
        assert!(out[16..].iter().all(|&v| v == -1.0));
    }

    #[test]
    fn copy_from_device_rejects_out_of_bounds_range() {
        let _device = Device::new(0).expect("failed to select device 0");
        let src = DeviceBuffer::<f32>::new(4).expect("hipMalloc src failed");
        let mut dst = DeviceBuffer::<f32>::new(4).expect("hipMalloc dst failed");
        assert!(dst.copy_from_device(2, &src, 0, 4).is_err());
        assert!(dst.copy_from_device(0, &src, 2, 4).is_err());
    }

    #[test]
    fn copy_range_roundtrips_a_prefix() {
        let _device = Device::new(0).expect("failed to select device 0");
        let mut buf = DeviceBuffer::<f32>::new(16).expect("hipMalloc failed");
        buf.copy_from_host(&[0.0f32; 16]).expect("zero-fill failed");

        let written: Vec<f32> = (0..6).map(|i| i as f32 + 100.0).collect();
        buf.copy_range_from_host(4, &written)
            .expect("copy_range_from_host failed");

        let mut readback = vec![0.0f32; 6];
        buf.copy_range_to_host(4, &mut readback)
            .expect("copy_range_to_host failed");
        assert_eq!(readback, written);

        // Untouched region stays zero.
        let mut before = vec![0.0f32; 4];
        buf.copy_range_to_host(0, &mut before).unwrap();
        assert!(before.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn copy_range_rejects_out_of_bounds() {
        let _device = Device::new(0).expect("failed to select device 0");
        let mut buf = DeviceBuffer::<f32>::new(8).expect("hipMalloc failed");
        let data = vec![1.0f32; 4];
        assert!(buf.copy_range_from_host(6, &data).is_err());
        let mut out = vec![0.0f32; 4];
        assert!(buf.copy_range_to_host(6, &mut out).is_err());
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
