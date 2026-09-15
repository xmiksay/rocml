//! GPU-resident rewind points for [`HybridCache`] — the snapshot layer's
//! "hot" tier (see `qwen35::forward::rewind` for the model-level slots and
//! `crate::snapshot::turn::run_turn` for how a turn uses them). Where
//! `snapshot.rs` copies the *whole* filled prefix out to host RAM (O(position)
//! bytes, hundreds of milliseconds at long context), a rewind point saves
//! only the state a later position can overwrite, entirely on-device:
//!
//! - every GDN layer's conv/recurrence state — O(1) in position, but
//!   advanced in place by every token, so it must be copied;
//! - every mixed-KV layer's recent window + `window_base` (see
//!   `cache_mixed::rewind` for why nothing else in a mixed layer moves);
//! - nothing at all for a dense `AttnPlane`: it only ever writes position
//!   `pos`'s own row, so positions below a rewind point are never touched.
//!
//! **Validity invariant** (enforced by the model-level slots, not here):
//! a rewind point saved at position P describes the cache *given that
//! positions `[0, P)` still hold the same K/V they held at save time*. Any
//! operation that rewrites those positions — `HybridCache::reset` followed
//! by a fresh sequence, a host-snapshot restore, or a rewind to a lower
//! position followed by new tokens — makes it stale, and the owner must
//! invalidate it rather than let a token-prefix match serve another
//! sequence's attention rows.

use super::*;
use crate::qwen35::cache_mixed::MixedWindowRewind;

/// Per-layer rewind storage, indexed like `HybridCache`'s own `gdn`/`attn`
/// vectors (a `Some` in `gdn` for every GDN layer, a `Some` in `mixed` for
/// every mixed-KV full-attention layer, `None` everywhere else).
pub struct RewindStorage {
    gdn: Vec<Option<GdnLayerState>>,
    mixed: Vec<Option<MixedWindowRewind>>,
}

impl GdnLayerState {
    /// Same shape as `self`, contents unspecified (the caller copies into
    /// it immediately) — unlike `new`, no zero-fill round trip.
    fn new_like(&self) -> Result<Self, RocmlError> {
        Ok(Self {
            conv_state: DeviceBuffer::new(self.conv_len)?,
            state: DeviceBuffer::new(self.state_len)?,
            conv_len: self.conv_len,
            state_len: self.state_len,
        })
    }

    fn copy_from(&mut self, src: &GdnLayerState) -> Result<(), RocmlError> {
        self.conv_state
            .copy_from_device(0, &src.conv_state, 0, self.conv_len)?;
        self.state
            .copy_from_device(0, &src.state, 0, self.state_len)?;
        Ok(())
    }
}

impl HybridCache {
    /// Allocates one rewind point's worth of storage for this cache's
    /// layer layout (see [`crate::budget::rewind_points_bytes`] for the
    /// size, which `Model::load` already reserves in its VRAM budget).
    pub fn new_rewind_storage(&self) -> Result<RewindStorage, RocmlError> {
        let gdn = self
            .gdn
            .iter()
            .map(|slot| slot.as_ref().map(GdnLayerState::new_like).transpose())
            .collect::<Result<Vec<_>, _>>()?;
        let mixed = self
            .attn
            .iter()
            .map(|slot| match slot {
                Some(AttnLayerCache::Mixed(plane)) => plane.new_window_rewind().map(Some),
                _ => Ok(None),
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(RewindStorage { gdn, mixed })
    }

    /// D2D-saves the current state into `dst` (allocated by
    /// [`Self::new_rewind_storage`] from this same cache).
    pub fn save_rewind(&self, dst: &mut RewindStorage) -> Result<(), RocmlError> {
        check_layout(self, dst)?;
        for (slot, saved) in self.gdn.iter().zip(dst.gdn.iter_mut()) {
            if let (Some(live), Some(saved)) = (slot, saved) {
                saved.copy_from(live)?;
            }
        }
        for (slot, saved) in self.attn.iter().zip(dst.mixed.iter_mut()) {
            if let (Some(AttnLayerCache::Mixed(plane)), Some(saved)) = (slot, saved) {
                plane.save_window(saved)?;
            }
        }
        Ok(())
    }

    /// D2D-restores a saved state. Does not touch dense planes or the bulk
    /// region — see the module doc for why that's correct, and its validity
    /// invariant for what the caller must guarantee.
    pub fn restore_rewind(&mut self, src: &RewindStorage) -> Result<(), RocmlError> {
        check_layout(self, src)?;
        for (slot, saved) in self.gdn.iter_mut().zip(src.gdn.iter()) {
            if let (Some(live), Some(saved)) = (slot, saved) {
                live.copy_from(saved)?;
            }
        }
        for (slot, saved) in self.attn.iter_mut().zip(src.mixed.iter()) {
            if let (Some(AttnLayerCache::Mixed(plane)), Some(saved)) = (slot, saved) {
                plane.restore_window(saved)?;
            }
        }
        Ok(())
    }
}

/// A `RewindStorage` is only ever built from the cache it's used with, so a
/// mismatch here is an internal bug — surfaced as an error rather than a
/// silent partial copy.
fn check_layout(cache: &HybridCache, storage: &RewindStorage) -> Result<(), RocmlError> {
    let gdn_ok = cache.gdn.len() == storage.gdn.len()
        && cache
            .gdn
            .iter()
            .zip(&storage.gdn)
            .all(|(a, b)| a.is_some() == b.is_some());
    let mixed_ok = cache.attn.len() == storage.mixed.len()
        && cache
            .attn
            .iter()
            .zip(&storage.mixed)
            .all(|(a, b)| matches!(a, Some(AttnLayerCache::Mixed(_))) == b.is_some());
    if gdn_ok && mixed_ok {
        Ok(())
    } else {
        Err(RocmlError::Config(
            "rewind storage layout doesn't match this cache (internal bug)".to_string(),
        ))
    }
}
