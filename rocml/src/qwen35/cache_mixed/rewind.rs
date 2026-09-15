//! GPU-resident rewind storage for [`MixedAttnPlane`] — the on-device
//! counterpart of `snapshot.rs`'s host capture, holding only what a *later*
//! position can overwrite. Everything else a mixed layer stores is
//! prefix-stable: the sink is written once per position, and the bulk region
//! only ever gains whole blocks at monotonically increasing block indices
//! (`MixedLayout::prepare_append`), so a rewind to position P finds every
//! bulk block below P's `evicted_blocks()` exactly as it was. The recent
//! window is the exception: it fills linearly from slot 0 and is reused
//! wholesale after each eviction, so tokens appended after P overwrite the
//! slots P had filled — hence the window (plus `window_base`, the one scalar
//! the whole eviction bookkeeping derives from) is what gets copied out and
//! back, device to device, never through the host.

use half::f16;
use rocml_hip::DeviceBuffer;

use super::MixedAttnPlane;
use crate::error::RocmlError;
use crate::kv_quant::MixedLayout;

/// One mixed layer's rewind state: a device copy of the whole recent-window
/// buffer pair plus the `window_base` it was saved at.
pub struct MixedWindowRewind {
    window_k: DeviceBuffer<f16>,
    window_v: DeviceBuffer<f16>,
    window_base: u32,
}

impl MixedAttnPlane {
    /// Allocates rewind storage shaped for this plane's window; its contents
    /// are unspecified until the first [`Self::save_window`].
    pub fn new_window_rewind(&self) -> Result<MixedWindowRewind, RocmlError> {
        Ok(MixedWindowRewind {
            window_k: DeviceBuffer::new(self.window_k.len())?,
            window_v: DeviceBuffer::new(self.window_v.len())?,
            window_base: self.layout.window_base(),
        })
    }

    /// D2D-copies the whole window (filled and stale slots alike — one
    /// contiguous copy per buffer beats a per-head strided one for a buffer
    /// this small) and records the current `window_base`.
    pub fn save_window(&self, dst: &mut MixedWindowRewind) -> Result<(), RocmlError> {
        dst.window_k
            .copy_from_device(0, &self.window_k, 0, self.window_k.len())?;
        dst.window_v
            .copy_from_device(0, &self.window_v, 0, self.window_v.len())?;
        dst.window_base = self.layout.window_base();
        Ok(())
    }

    /// Writes a saved window back and rewinds the eviction bookkeeping to
    /// the saved `window_base` — after which the bulk region reads as having
    /// exactly the blocks that were evicted at save time (later evictions
    /// only ever wrote *higher* block indices, so nothing below needs
    /// restoring).
    pub fn restore_window(&mut self, src: &MixedWindowRewind) -> Result<(), RocmlError> {
        self.window_k
            .copy_from_device(0, &src.window_k, 0, src.window_k.len())?;
        self.window_v
            .copy_from_device(0, &src.window_v, 0, src.window_v.len())?;
        self.layout = MixedLayout::from_window_base_with_lens(
            src.window_base,
            self.sink_len,
            self.window_len,
        );
        Ok(())
    }
}
