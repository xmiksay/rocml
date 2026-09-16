//! Best-effort pinning of an existing host memory range (e.g. a slice of a
//! memory-mapped file) so `hipMemcpy` can DMA it directly instead of staging
//! through a bounce buffer. This is a pure performance hint for the MoE
//! expert-offload path (`qwen35::weights::moe`): `hipMemcpy` against an
//! *unregistered* host pointer still copies correct bytes, just slower
//! (measured on the target box: ~12.9-13.2 GB/s registered vs. ~6.4 GB/s
//! unregistered), so [`host_register_readonly`] returns a plain `bool`
//! rather than a `Result` — callers must always proceed with the copy
//! either way, never treat a `false` as an error to propagate.
use std::ffi::c_void;

use crate::ffi;

/// Attempts to page-lock `[ptr, ptr+len)` as read-only host memory. Returns
/// `true` on success. A `false` return (unsupported range, already
/// registered/overlapping pages, memlock limit, etc.) is expected and safe
/// to ignore — see the module doc.
pub fn host_register_readonly(ptr: *const u8, len: usize) -> bool {
    if len == 0 {
        return true;
    }
    // SAFETY: `hipHostRegister` only records the page range for later DMA
    // use; it neither reads through nor retains `ptr` beyond validating it,
    // and `len` bytes starting at `ptr` are the caller's own live
    // allocation (a memory-mapped file slice that outlives this call).
    let rc =
        unsafe { ffi::hipHostRegister(ptr as *mut c_void, len, ffi::hip_host_register_read_only) };
    rc == ffi::hipSuccess
}

/// Unregisters a range previously pinned by [`host_register_readonly`].
/// Best-effort like its counterpart — a failure here is not actionable
/// (most callers in this codebase never call this at all, since a `Model`
/// has a 1:1 process lifetime and the OS reclaims the registration on exit).
pub fn host_unregister(ptr: *const u8) -> bool {
    // SAFETY: `hipHostUnregister` only unpins a previously-registered range;
    // passing a pointer that was never registered is a documented no-op
    // failure (`hipErrorHostMemoryNotRegistered`), not undefined behavior.
    let rc = unsafe { ffi::hipHostUnregister(ptr as *mut c_void) };
    rc == ffi::hipSuccess
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;

    #[test]
    fn register_and_unregister_a_heap_buffer_roundtrips() {
        let _device = Device::new(0).expect("failed to select device 0");
        let buf = vec![0u8; 4096];
        assert!(host_register_readonly(buf.as_ptr(), buf.len()));
        assert!(host_unregister(buf.as_ptr()));
    }

    #[test]
    fn zero_length_is_a_trivial_success() {
        assert!(host_register_readonly(std::ptr::null(), 0));
    }
}
