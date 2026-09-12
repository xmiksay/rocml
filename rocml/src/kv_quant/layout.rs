//! Pure position -> region bookkeeping for a KIVI-style mixed KV layer
//! (issue #2): attention sinks (first `SINK_LEN` positions, never evicted),
//! a quantized bulk (evicted history, `WINDOW_LEN`-position blocks), and a
//! recent-window ring that holds the newest not-yet-evicted positions. No
//! device buffers or HIP calls here at all — this module answers "which
//! region does position N live in" and "does appending N trigger an
//! eviction", so it's testable without a GPU and shared identically by the
//! host-side append path and (as plain constants) the fused kernels.
//!
//! Eviction is whole-window, not per-position (per issue #2's own review
//! comment: "the recent-window ring should quantize evicted blocks in one
//! batch at eviction time — simpler than continuous quantization"): once
//! the window fills (`WINDOW_LEN` positions since the last eviction), the
//! *entire* window is quantized in one kernel launch and the window resets
//! to empty, rather than evicting one position at a time. This means the
//! window buffer's physical index always restarts at 0 right after an
//! eviction — no modulo/ring-index arithmetic needed anywhere, including in
//! the fused attention kernel's read path.

/// First `SINK_LEN` positions of every mixed layer stay fp16 forever —
/// attention sinks are exempt from eviction/quantization (issue #2).
pub const SINK_LEN: u32 = 32;
/// Recent-window capacity, and the quantize-on-evict batch size.
pub const WINDOW_LEN: u32 = 128;

/// Where position `pos` currently lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Region {
    Sink {
        index: u32,
    },
    /// `block`'s positions are `[SINK_LEN + block*WINDOW_LEN, SINK_LEN +
    /// (block+1)*WINDOW_LEN)`; `offset` is the position's index within it.
    Bulk {
        block: u32,
        offset: u32,
    },
    Window {
        index: u32,
    },
}

/// One mixed layer's eviction state: only the absolute position the
/// window's slot 0 currently maps to (everything else — sink boundary,
/// bulk block count — is derivable from it and the two constants above).
#[derive(Debug, Clone, Copy)]
pub struct MixedLayout {
    window_base: u32,
}

impl Default for MixedLayout {
    fn default() -> Self {
        Self::new()
    }
}

impl MixedLayout {
    pub fn new() -> Self {
        Self {
            window_base: SINK_LEN,
        }
    }

    /// Reconstructs a layout from a previously-observed `window_base` — used
    /// by the snapshot layer (issue #1) to restore a mixed layer's eviction
    /// state exactly, since `window_base` alone determines every other
    /// derived quantity (`bulk_len`/`evicted_blocks`/`region`).
    pub fn from_window_base(window_base: u32) -> Self {
        Self { window_base }
    }

    /// Absolute position the window's physical slot 0 currently maps to.
    pub fn window_base(&self) -> u32 {
        self.window_base
    }

    /// Number of positions currently evicted into the bulk region.
    pub fn bulk_len(&self) -> u32 {
        self.window_base - SINK_LEN
    }

    /// Number of whole blocks evicted so far.
    pub fn evicted_blocks(&self) -> u32 {
        self.bulk_len() / WINDOW_LEN
    }

    /// Which region `pos` lives in, given the *current* eviction state
    /// (call after any `prepare_append` that `pos` needed).
    pub fn region(&self, pos: u32) -> Region {
        if pos < SINK_LEN {
            Region::Sink { index: pos }
        } else if pos < self.window_base {
            let rel = pos - SINK_LEN;
            Region::Bulk {
                block: rel / WINDOW_LEN,
                offset: rel % WINDOW_LEN,
            }
        } else {
            Region::Window {
                index: pos - self.window_base,
            }
        }
    }

    /// Call before writing a *new* position `pos >= SINK_LEN` into the
    /// window. If the window is already full (`pos` would be the
    /// `WINDOW_LEN`-th slot past `window_base`), evicts the whole window in
    /// one batch (advancing `window_base`) and returns the evicted block
    /// index; the caller must launch the quantize-evict kernel for that
    /// block over the window's *current* (pre-advance) contents before
    /// this call, or read `window_base`'s pre-call value itself to know
    /// which absolute positions were evicted. Returns the physical window
    /// slot `pos` should be written to either way.
    ///
    /// Positions `< SINK_LEN` (sink writes) never call this — the sink has
    /// its own fixed-index write, no eviction concept.
    pub fn prepare_append(&mut self, pos: u32) -> (u32, Option<u32>) {
        debug_assert!(pos >= SINK_LEN, "sink positions don't use prepare_append");
        debug_assert!(
            pos >= self.window_base,
            "prepare_append called for an already-evicted position"
        );
        let mut evicted_block = None;
        if pos - self.window_base >= WINDOW_LEN {
            // Exactly one eviction can be pending at a time: `pos` only
            // ever advances by 1 between calls (one decode step at a
            // time), so it can be at most WINDOW_LEN past window_base.
            debug_assert_eq!(pos - self.window_base, WINDOW_LEN);
            evicted_block = Some(self.evicted_blocks());
            self.window_base += WINDOW_LEN;
        }
        (pos - self.window_base, evicted_block)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sink_positions_are_sink_region() {
        let layout = MixedLayout::new();
        for pos in 0..SINK_LEN {
            assert_eq!(layout.region(pos), Region::Sink { index: pos });
        }
    }

    #[test]
    fn positions_after_sink_are_window_before_any_eviction() {
        let layout = MixedLayout::new();
        assert_eq!(layout.region(SINK_LEN), Region::Window { index: 0 });
        assert_eq!(
            layout.region(SINK_LEN + WINDOW_LEN - 1),
            Region::Window {
                index: WINDOW_LEN - 1
            }
        );
    }

    #[test]
    fn window_fills_without_eviction_until_exactly_full() {
        let mut layout = MixedLayout::new();
        for pos in SINK_LEN..SINK_LEN + WINDOW_LEN {
            let (slot, evicted) = layout.prepare_append(pos);
            assert_eq!(slot, pos - SINK_LEN);
            assert!(evicted.is_none(), "no eviction expected at pos {pos}");
        }
        assert_eq!(layout.window_base(), SINK_LEN);
        assert_eq!(layout.bulk_len(), 0);
    }

    #[test]
    fn first_eviction_happens_exactly_once_at_window_len_plus_one() {
        let mut layout = MixedLayout::new();
        for pos in SINK_LEN..SINK_LEN + WINDOW_LEN {
            layout.prepare_append(pos);
        }
        let (slot, evicted) = layout.prepare_append(SINK_LEN + WINDOW_LEN);
        assert_eq!(evicted, Some(0));
        assert_eq!(slot, 0);
        assert_eq!(layout.window_base(), SINK_LEN + WINDOW_LEN);
        assert_eq!(layout.bulk_len(), WINDOW_LEN);
        assert_eq!(layout.evicted_blocks(), 1);
    }

    #[test]
    fn resume_after_eviction_places_positions_in_correct_regions() {
        let mut layout = MixedLayout::new();
        for pos in SINK_LEN..SINK_LEN + WINDOW_LEN + 1 {
            layout.prepare_append(pos);
        }
        // The just-evicted block's positions now read back as Bulk.
        assert_eq!(
            layout.region(SINK_LEN),
            Region::Bulk {
                block: 0,
                offset: 0
            }
        );
        assert_eq!(
            layout.region(SINK_LEN + WINDOW_LEN - 1),
            Region::Bulk {
                block: 0,
                offset: WINDOW_LEN - 1
            }
        );
        // The position that triggered the eviction is the new window's
        // slot 0, not slot WINDOW_LEN.
        assert_eq!(
            layout.region(SINK_LEN + WINDOW_LEN),
            Region::Window { index: 0 }
        );
    }

    #[test]
    fn every_position_is_quantized_into_bulk_exactly_once_across_many_evictions() {
        let mut layout = MixedLayout::new();
        let n_blocks = 5;
        let mut evictions = Vec::new();
        for pos in SINK_LEN..SINK_LEN + n_blocks * WINDOW_LEN {
            let (_, evicted) = layout.prepare_append(pos);
            if let Some(block) = evicted {
                evictions.push(block);
            }
        }
        // Blocks 0..n_blocks-1 evicted, each exactly once, strictly in
        // order (the last block — still in the window — is never evicted
        // by this loop, since the loop stops exactly at its first
        // position).
        assert_eq!(evictions, (0..n_blocks - 1).collect::<Vec<_>>());

        // Every bulk position resolves to exactly one (block, offset).
        for block in 0..n_blocks - 1 {
            for offset in 0..WINDOW_LEN {
                let pos = SINK_LEN + block * WINDOW_LEN + offset;
                assert_eq!(layout.region(pos), Region::Bulk { block, offset });
            }
        }
    }

    #[test]
    fn second_window_boundary_matches_first() {
        let mut layout = MixedLayout::new();
        for pos in SINK_LEN..SINK_LEN + 2 * WINDOW_LEN {
            layout.prepare_append(pos);
        }
        assert_eq!(layout.evicted_blocks(), 1);
        // Second window's slots resolve correctly after the second fill
        // (not yet evicted — this loop stops one short of the second
        // eviction trigger).
        assert_eq!(
            layout.region(SINK_LEN + 2 * WINDOW_LEN - 1),
            Region::Window {
                index: WINDOW_LEN - 1
            }
        );
    }
}
