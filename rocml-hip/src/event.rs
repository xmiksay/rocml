//! HIP event RAII wrapper for wall-time measurement between two points on a
//! stream (`rocml`'s `Profiler` is the intended caller — see that module's
//! doc comment for why events, not `hipDeviceSynchronize`, are the
//! measurement primitive for profiling a hot decode loop).
use std::ptr;

use crate::device::Stream;
use crate::error::{check, HipError};
use crate::ffi;

/// An `hipEvent_t` created with default (non-blocking-sync) flags —
/// `hipEventRecord` is a cheap async enqueue on the stream, so recording one
/// before/after a span of work costs nothing until someone reads it back.
pub struct Event {
    handle: ffi::hipEvent_t,
}

impl Event {
    pub fn new() -> Result<Self, HipError> {
        let mut handle: ffi::hipEvent_t = ptr::null_mut();
        // SAFETY: `handle` is a valid out-param location for the new event handle.
        check(unsafe { ffi::hipEventCreate(&mut handle) })?;
        Ok(Self { handle })
    }

    /// Enqueues this event on `stream` (or the default/null stream if
    /// `None` — every kernel launch in this codebase runs on the default
    /// stream today, so `None` is what callers use in practice).
    pub fn record(&self, stream: Option<&Stream>) -> Result<(), HipError> {
        let stream_handle = stream.map(Stream::handle).unwrap_or(ptr::null_mut());
        // SAFETY: self.handle was created by hipEventCreate and not yet
        // destroyed; stream_handle is either null (default stream, always
        // valid) or a live stream borrowed for the duration of this call.
        check(unsafe { ffi::hipEventRecord(self.handle, stream_handle) })
    }

    /// Blocks the host until this event's recorded point has completed.
    /// `elapsed_ms` calls this on the `stop` event internally — callers
    /// don't need to call it themselves before reading elapsed time.
    pub fn synchronize(&self) -> Result<(), HipError> {
        // SAFETY: self.handle was created by hipEventCreate and not yet destroyed.
        check(unsafe { ffi::hipEventSynchronize(self.handle) })
    }

    /// Raw handle for crate-internal use (`Stream::wait_event`).
    pub(crate) fn handle(&self) -> ffi::hipEvent_t {
        self.handle
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: self.handle was produced by hipEventCreate in `new`
            // and Drop runs at most once per value.
            unsafe {
                let _ = ffi::hipEventDestroy(self.handle);
            }
        }
    }
}

/// Makes the default (null) stream wait on `event` before executing any
/// kernel launched on it afterward — the `Stream::wait_event` a *non*-
/// default stream gets, for the one stream this codebase can't name as a
/// `Stream` value (every kernel launch's `stream: Option<&Stream>` uses
/// `None` for it). Used by the qwen35moe decode-overlap lever (M4) so a
/// default-stream `gemv_quant` read waits for an async copy on the overlap
/// stream to finish, without the host blocking on either.
pub fn wait_on_default_stream(event: &Event) -> Result<(), HipError> {
    // SAFETY: event.handle was created by hipEventCreate and not yet
    // destroyed; a null stream handle is HIP's documented spelling for the
    // default stream (same convention `Event::record`/`Function::launch`
    // already use for `stream: None`).
    check(unsafe { ffi::hipStreamWaitEvent(ptr::null_mut(), event.handle(), 0) })
}

/// Milliseconds elapsed between `start` and `stop`'s recorded points.
/// Synchronizes on `stop` first (per HIP's contract for
/// `hipEventElapsedTime`: both events must have completed), so this is a
/// blocking call — batch it at report time rather than inside a hot loop.
pub fn elapsed_ms(start: &Event, stop: &Event) -> Result<f64, HipError> {
    stop.synchronize()?;
    let mut ms: f32 = 0.0;
    // SAFETY: both handles were created by hipEventCreate and not yet
    // destroyed (borrowed for the duration of this call); `ms` is a valid
    // out-param location. `stop.synchronize()` above guarantees both events
    // have already completed by the time this reads their timestamps.
    check(unsafe { ffi::hipEventElapsedTime(&mut ms, start.handle, stop.handle) })?;
    Ok(ms as f64)
}

impl Stream {
    /// Enqueues `event` on this stream — symmetric with [`Event::record`],
    /// which takes the stream the other way around.
    pub fn record(&self, event: &Event) -> Result<(), HipError> {
        event.record(Some(self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;
    use crate::DeviceBuffer;

    #[test]
    fn event_round_trip_on_default_stream() {
        let _device = Device::new(0).expect("failed to select device 0");
        let start = Event::new().expect("hipEventCreate (start)");
        let mut buf = DeviceBuffer::<f32>::new(1 << 16).expect("hipMalloc");
        let host = vec![1.0f32; 1 << 16];

        start.record(None).expect("record start");
        buf.copy_from_host(&host).expect("copy_from_host");
        let stop = Event::new().expect("hipEventCreate (stop)");
        stop.record(None).expect("record stop");

        let ms = elapsed_ms(&start, &stop).expect("elapsed_ms");
        assert!(ms >= 0.0, "elapsed time went negative: {ms}");
        assert!(ms.is_finite(), "elapsed time was not finite: {ms}");
    }

    #[test]
    fn event_round_trip_on_explicit_stream() {
        let _device = Device::new(0).expect("failed to select device 0");
        let stream = Stream::new().expect("hipStreamCreate");
        let start = Event::new().expect("hipEventCreate (start)");
        let stop = Event::new().expect("hipEventCreate (stop)");

        stream.record(&start).expect("record start");
        stream.record(&stop).expect("record stop");
        stream.synchronize().expect("stream sync");

        let ms = elapsed_ms(&start, &stop).expect("elapsed_ms");
        assert!(ms >= 0.0, "elapsed time went negative: {ms}");
    }
}
