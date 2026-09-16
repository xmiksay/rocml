//! Device selection/query and stream RAII wrappers.
use std::ffi::{c_char, CStr};
use std::ptr;

use crate::error::{check, HipError};
use crate::event::Event;
use crate::ffi;

/// Free/total device memory in bytes, as reported by `hipMemGetInfo`.
#[derive(Debug, Clone, Copy)]
pub struct MemoryInfo {
    pub free: usize,
    pub total: usize,
}

/// A selected HIP device. HIP's device selection is per-thread and global to
/// the thread (`hipSetDevice`), not scoped to this handle — constructing a
/// `Device` selects it as a side effect, and dropping the handle does not
/// restore whatever was selected before.
pub struct Device {
    ordinal: i32,
}

impl Device {
    /// Number of HIP devices visible to this process.
    pub fn count() -> Result<i32, HipError> {
        let mut count: i32 = 0;
        // SAFETY: `count` is a valid, aligned location for an out-param i32.
        check(unsafe { ffi::hipGetDeviceCount(&mut count) })?;
        Ok(count)
    }

    /// Selects `ordinal` as the current device and returns a handle to it.
    pub fn new(ordinal: i32) -> Result<Self, HipError> {
        // SAFETY: hipSetDevice takes a plain ordinal by value, no pointers involved.
        check(unsafe { ffi::hipSetDevice(ordinal) })?;
        Ok(Self { ordinal })
    }

    pub fn ordinal(&self) -> i32 {
        self.ordinal
    }

    pub fn name(&self) -> Result<String, HipError> {
        const BUF_LEN: usize = 256;
        let mut buf = [0u8; BUF_LEN];
        // SAFETY: buf is BUF_LEN valid, writable bytes; HIP NUL-terminates the
        // name within that length on success and never writes past it.
        check(unsafe {
            ffi::hipDeviceGetName(
                buf.as_mut_ptr() as *mut c_char,
                BUF_LEN as i32,
                self.ordinal,
            )
        })?;
        // SAFETY: the call above succeeded, so buf contains a NUL-terminated string.
        let cstr = unsafe { CStr::from_ptr(buf.as_ptr() as *const c_char) };
        Ok(cstr.to_string_lossy().into_owned())
    }

    /// Free/total memory for this device. `hipMemGetInfo` reports on the
    /// current device, so this re-selects `self` first to stay correct even
    /// if another `Device` handle changed the thread's current device since.
    pub fn memory_info(&self) -> Result<MemoryInfo, HipError> {
        // SAFETY: see `new` — plain ordinal, no pointers.
        check(unsafe { ffi::hipSetDevice(self.ordinal) })?;
        let mut free: usize = 0;
        let mut total: usize = 0;
        // SAFETY: `free`/`total` are valid, aligned out-param locations.
        check(unsafe { ffi::hipMemGetInfo(&mut free, &mut total) })?;
        Ok(MemoryInfo { free, total })
    }
}

/// RAII wrapper around a `hipStream_t`.
pub struct Stream {
    handle: ffi::hipStream_t,
}

impl Stream {
    pub fn new() -> Result<Self, HipError> {
        let mut handle: ffi::hipStream_t = ptr::null_mut();
        // SAFETY: `handle` is a valid out-param location for the new stream handle.
        check(unsafe { ffi::hipStreamCreate(&mut handle) })?;
        Ok(Self { handle })
    }

    /// A stream created with `hipStreamNonBlocking`: unlike [`Self::new`]'s
    /// stream, this one does *not* implicitly synchronize with the legacy
    /// default (null) stream — every other kernel launch in this codebase
    /// runs on that default stream, so ordering against it must be
    /// established explicitly (`Event`/`wait_event`). Used by the
    /// qwen35moe decode-overlap lever (M4) to copy an expert's weight
    /// bytes concurrently with unrelated default-stream compute.
    pub fn new_non_blocking() -> Result<Self, HipError> {
        let mut handle: ffi::hipStream_t = ptr::null_mut();
        // SAFETY: `handle` is a valid out-param location for the new stream handle.
        check(unsafe { ffi::hipStreamCreateWithFlags(&mut handle, ffi::hip_stream_non_blocking) })?;
        Ok(Self { handle })
    }

    pub fn synchronize(&self) -> Result<(), HipError> {
        // SAFETY: self.handle was created by hipStreamCreate and not yet destroyed.
        check(unsafe { ffi::hipStreamSynchronize(self.handle) })
    }

    /// Makes every operation enqueued on this stream *after* this call wait
    /// (on the GPU, not the host) until `event` has fired — the ordering
    /// primitive a [`Self::new_non_blocking`] stream needs in place of the
    /// legacy default stream's implicit synchronization.
    pub fn wait_event(&self, event: &Event) -> Result<(), HipError> {
        // SAFETY: self.handle and event.handle were both created by their
        // respective `hip*Create*` calls and not yet destroyed; `flags` is
        // always 0 per HIP's own contract (no other value is defined).
        check(unsafe { ffi::hipStreamWaitEvent(self.handle, event.handle(), 0) })
    }

    /// Raw handle for crate-internal use (kernel launches).
    pub(crate) fn handle(&self) -> ffi::hipStream_t {
        self.handle
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: self.handle was created by hipStreamCreate in `new` and
            // Drop runs at most once per value, so this can't double-destroy.
            unsafe {
                let _ = ffi::hipStreamDestroy(self.handle);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_init_and_name() {
        let count = Device::count().expect("hipGetDeviceCount failed");
        assert!(count > 0, "expected at least one HIP device");

        let device = Device::new(0).expect("failed to select device 0");
        let name = device.name().expect("hipDeviceGetName failed");
        assert!(!name.trim().is_empty(), "device name was empty");
    }

    #[test]
    fn memory_info_is_sane() {
        let device = Device::new(0).expect("failed to select device 0");
        let info = device.memory_info().expect("hipMemGetInfo failed");
        assert!(info.total > 0, "total device memory reported as 0");
        assert!(info.free <= info.total, "free memory exceeds total");
    }

    #[test]
    fn stream_create_and_sync() {
        let _device = Device::new(0).expect("failed to select device 0");
        let stream = Stream::new().expect("hipStreamCreate failed");
        stream.synchronize().expect("hipStreamSynchronize failed");
    }
}
