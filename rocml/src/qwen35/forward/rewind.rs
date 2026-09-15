//! GPU-resident rewind points — the snapshot layer's hot tier. A growing
//! single-session conversation never needs its state round-tripped through
//! host RAM: the next turn's prompt shares a prefix with this turn's, and
//! everything below that prefix is still sitting in VRAM. Two named slots
//! cover the two prefixes a next turn can actually match (see
//! `crate::snapshot::turn::run_turn`):
//!
//! - [`RewindSlot::StableBoundary`]: the render-stable cut right after the
//!   last history message (issue #12) — what a thinking-enabled next turn
//!   matches, since its re-render strips this turn's `<think>` block;
//! - [`RewindSlot::EndOfTurn`]: prompt + everything generated — what a
//!   thinking-off next turn (whose history re-render reproduces the reply
//!   byte-for-byte) or a "continue" request matches.
//!
//! Each slot costs one `HybridCache::new_rewind_storage` allocation (GDN
//! state + mixed windows, ~53 MiB on Ornith-1.0-9B), allocated on first use
//! and reused across turns. Saving or restoring is a handful of
//! device-to-device copies — microseconds, versus the host tier's
//! O(position) D2H/H2D transfers.
//!
//! **Validity**: see `qwen35::cache::rewind`'s module doc. Concretely, this
//! module invalidates every slot on `Model::reset`/`restore_snapshot`
//! (positions from 0 get rewritten) and, on a rewind to position P, every
//! *other* slot above P (positions past P are about to be rewritten by the
//! new suffix). Slots at or below P stay valid — their prefix is untouched.

use super::Model;
use crate::error::RocmlError;
use crate::qwen35::cache::RewindStorage;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RewindSlot {
    StableBoundary,
    EndOfTurn,
}

pub(super) struct RewindPoint {
    /// `None` = allocated but invalid (never saved, or invalidated since).
    prefix: Option<Vec<u32>>,
    storage: RewindStorage,
}

#[derive(Default)]
pub(super) struct RewindPoints {
    stable_boundary: Option<RewindPoint>,
    end_of_turn: Option<RewindPoint>,
}

impl RewindPoints {
    fn slot_mut(&mut self, slot: RewindSlot) -> &mut Option<RewindPoint> {
        match slot {
            RewindSlot::StableBoundary => &mut self.stable_boundary,
            RewindSlot::EndOfTurn => &mut self.end_of_turn,
        }
    }

    fn iter_mut(&mut self) -> impl Iterator<Item = &mut RewindPoint> {
        self.stable_boundary
            .iter_mut()
            .chain(self.end_of_turn.iter_mut())
    }

    /// The longest valid slot whose saved prefix is a strict prefix of
    /// `token_ids`, with that prefix's length.
    fn matching_slot(&self, token_ids: &[u32]) -> Option<(RewindSlot, usize)> {
        [
            (RewindSlot::StableBoundary, &self.stable_boundary),
            (RewindSlot::EndOfTurn, &self.end_of_turn),
        ]
        .into_iter()
        .filter_map(|(slot, point)| Some((slot, point.as_ref()?.prefix.as_deref()?)))
        .filter(|(_, prefix)| prefix.len() < token_ids.len() && token_ids.starts_with(prefix))
        .map(|(slot, prefix)| (slot, prefix.len()))
        .max_by_key(|(_, len)| *len)
    }

    /// Storage stays allocated (it's reused by the next save); only the
    /// token prefix that makes a slot matchable is dropped.
    pub(super) fn invalidate_all(&mut self) {
        for point in self.iter_mut() {
            point.prefix = None;
        }
    }

    fn invalidate_above(&mut self, position: usize) {
        for point in self.iter_mut() {
            if point.prefix.as_ref().is_some_and(|p| p.len() > position) {
                point.prefix = None;
            }
        }
    }
}

impl Model {
    /// Saves the current state into `slot`. `token_ids` must be the exact
    /// sequence that produced it (`len == self.position()`), the identity a
    /// later [`Self::rewind_to_prefix`] matches against.
    pub fn save_rewind_point(
        &mut self,
        slot: RewindSlot,
        token_ids: Vec<u32>,
    ) -> Result<(), RocmlError> {
        if token_ids.len() as u32 != self.pos {
            return Err(RocmlError::Config(format!(
                "save_rewind_point: token_ids.len() ({}) must equal the current position ({})",
                token_ids.len(),
                self.pos
            )));
        }
        let entry = self.rewind.slot_mut(slot);
        if entry.is_none() {
            *entry = Some(RewindPoint {
                prefix: None,
                storage: self.cache.new_rewind_storage()?,
            });
        }
        let point = entry.as_mut().expect("slot was just filled");
        // Invalidate before overwriting: if a copy fails partway, the slot
        // must not stay matchable over half-written storage.
        point.prefix = None;
        self.cache.save_rewind(&mut point.storage)?;
        point.prefix = Some(token_ids);
        Ok(())
    }

    /// Rewinds to the longest valid slot whose saved prefix is a *strict*
    /// prefix of `token_ids` (always leaving at least one token to prefill,
    /// the same contract as `SnapshotStore::lookup`), returning the position
    /// reached, or `None` (state untouched) if no slot matches.
    pub fn rewind_to_prefix(&mut self, token_ids: &[u32]) -> Result<Option<u32>, RocmlError> {
        let Some((slot, len)) = self.rewind.matching_slot(token_ids) else {
            return Ok(None);
        };
        let point = self
            .rewind
            .slot_mut(slot)
            .as_ref()
            .expect("matching_slot only returns populated slots");
        self.cache.restore_rewind(&point.storage)?;
        self.pos = len as u32;
        // The suffix about to be prefilled rewrites every position past
        // `len`, which stales any slot saved above it.
        self.rewind.invalidate_above(len);
        Ok(Some(self.pos))
    }

    pub(super) fn invalidate_rewind_points(&mut self) {
        self.rewind.invalidate_all();
    }
}
