//! `MixedLayout::plan_chunk_append`: the chunked-prefill sibling of
//! `MixedLayout::prepare_append` (`mod.rs`) — batches a whole chunk's worth
//! of position appends into a small plan instead of requiring one call per
//! position. Split into its own file purely for the 400-line cap, mirroring
//! `cache/snapshot.rs`'s split from `cache/mod.rs`.

use super::{MixedLayout, SINK_LEN, WINDOW_LEN};

/// One segment of a chunk's window-region append — see
/// [`MixedLayout::plan_chunk_append`]'s doc comment for the exact contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkWindowSegment {
    /// Row index into the *whole incoming chunk* (0-based from the chunk's
    /// own first row, already accounting for any leading `sink_rows` —
    /// i.e. this indexes directly into the chunk's own K/V source buffer,
    /// no further offset needed) where this segment starts.
    pub chunk_row_start: u32,
    /// Number of consecutive rows in this segment.
    pub len: u32,
    /// Physical window ring slot the segment's first row lands at (rows
    /// land at consecutive slots `[window_slot_start, window_slot_start +
    /// len)`).
    pub window_slot_start: u32,
    /// If set, the caller must quantize-evict this bulk block index from
    /// the window's current (now full) contents immediately after writing
    /// this segment's rows, before writing any later segment.
    pub evict_after: Option<u32>,
}

impl MixedLayout {
    /// Batched sibling of [`Self::prepare_append`] for a chunked-prefill
    /// append of `chunk_len` new positions starting at `pos_base`
    /// (`[pos_base, pos_base+chunk_len)`, assumed — like `prepare_append`
    /// — to be exactly this layer's next unappended positions). Mutates
    /// `self` exactly as `chunk_len` sequential `prepare_append` calls
    /// would (so `self.window_base()` afterward is bit-for-bit identical
    /// either way), and returns:
    ///
    /// - `sink_rows`: how many leading chunk rows (`pos < SINK_LEN`) the
    ///   caller should scatter-write straight into the sink buffer at
    ///   `[pos_base, pos_base+sink_rows)` — these never touch eviction
    ///   state at all, mirroring `append`'s own sink/window split.
    /// - the list of [`ChunkWindowSegment`]s the remaining `chunk_len -
    ///   sink_rows` rows split into, in the order the caller must process
    ///   them (write the segment's rows into the window buffer at
    ///   `window_slot_start`, then — if `evict_after` is set — launch the
    ///   quantize-evict kernel over the window's now-current, now-full
    ///   contents for that block index, *before* writing the next
    ///   segment). Almost always zero or one segment in practice (this
    ///   crate's chunked-prefill chunk size is exactly `WINDOW_LEN`, so a
    ///   chunk spans at most one eviction boundary — see
    ///   `qwen35::forward::chunk_forward::PREFILL_CHUNK_SIZE`), but this
    ///   handles any `chunk_len` correctly: it's a direct batched
    ///   transcription of what repeated `prepare_append` calls compute,
    ///   never an approximation, which is what makes the resulting cache
    ///   state identical to token-serial appends (the property this
    ///   module's tests below and `qwen35::cache_mixed::chunk`'s own gate
    ///   both check).
    pub fn plan_chunk_append(
        &mut self,
        pos_base: u32,
        chunk_len: u32,
    ) -> (u32, Vec<ChunkWindowSegment>) {
        let sink_rows = SINK_LEN.saturating_sub(pos_base).min(chunk_len);
        let mut segments = Vec::new();
        let mut row = sink_rows;
        let mut pos = pos_base + sink_rows;
        while row < chunk_len {
            let filled = pos - self.window_base;
            let room = WINDOW_LEN - filled;
            let take = room.min(chunk_len - row);
            let mut evict_after = None;
            if take == room && row + take < chunk_len {
                // This segment completes the currently-open block *and*
                // the chunk keeps going past it — the next row's position
                // is exactly the one that triggers eviction in
                // `prepare_append`'s per-position model. If the chunk
                // stopped exactly here instead, the block stays full but
                // un-evicted (matches `window_fills_without_eviction_until_exactly_full`).
                evict_after = Some(self.evicted_blocks());
                self.window_base += WINDOW_LEN;
            }
            segments.push(ChunkWindowSegment {
                chunk_row_start: row,
                len: take,
                window_slot_start: filled,
                evict_after,
            });
            row += take;
            pos += take;
        }
        (sink_rows, segments)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Replays `prepare_append` one position at a time and records every
    /// eviction plus the physical window slot each position landed at — the
    /// ground truth `plan_chunk_append`'s batched output must reproduce
    /// exactly, for any way the same `[0, total_len)` range is split into
    /// chunks.
    struct TokenSerialTrace {
        window_base_final: u32,
        /// `(position, physical_slot)` for every position `>= SINK_LEN`.
        window_slots: Vec<(u32, u32)>,
        /// Evicted block indices, in the order they were evicted.
        evictions: Vec<u32>,
    }

    fn replay_token_serial(total_len: u32) -> TokenSerialTrace {
        let mut layout = MixedLayout::new();
        let mut window_slots = Vec::new();
        let mut evictions = Vec::new();
        for pos in SINK_LEN..total_len {
            let (slot, evicted) = layout.prepare_append(pos);
            window_slots.push((pos, slot));
            if let Some(block) = evicted {
                evictions.push(block);
            }
        }
        TokenSerialTrace {
            window_base_final: layout.window_base(),
            window_slots,
            evictions,
        }
    }

    /// Replays the same `[0, total_len)` range through `plan_chunk_append`,
    /// split into chunks per `chunk_lens` (which must sum to `total_len`),
    /// and returns the same shape of trace so it can be compared directly
    /// against [`replay_token_serial`].
    fn replay_chunked(total_len: u32, chunk_lens: &[u32]) -> TokenSerialTrace {
        assert_eq!(chunk_lens.iter().sum::<u32>(), total_len);
        let mut layout = MixedLayout::new();
        let mut window_slots = Vec::new();
        let mut evictions = Vec::new();
        let mut pos_base = 0;
        for &chunk_len in chunk_lens {
            let (_sink_rows, segments) = layout.plan_chunk_append(pos_base, chunk_len);
            for seg in &segments {
                for i in 0..seg.len {
                    let pos = pos_base + seg.chunk_row_start + i;
                    window_slots.push((pos, seg.window_slot_start + i));
                }
                if let Some(block) = seg.evict_after {
                    evictions.push(block);
                }
            }
            pos_base += chunk_len;
        }
        TokenSerialTrace {
            window_base_final: layout.window_base(),
            window_slots,
            evictions,
        }
    }

    fn assert_traces_match(chunked: &TokenSerialTrace, serial: &TokenSerialTrace, label: &str) {
        assert_eq!(
            chunked.window_base_final, serial.window_base_final,
            "{label}: window_base mismatch"
        );
        assert_eq!(
            chunked.evictions, serial.evictions,
            "{label}: eviction sequence mismatch"
        );
        assert_eq!(
            chunked.window_slots, serial.window_slots,
            "{label}: per-position physical slot mismatch"
        );
    }

    #[test]
    fn chunk_append_matches_token_serial_single_chunk_within_sink() {
        let total_len = 20;
        let serial = replay_token_serial(total_len);
        let chunked = replay_chunked(total_len, &[total_len]);
        assert_traces_match(&chunked, &serial, "single chunk within sink");
    }

    #[test]
    fn chunk_append_matches_token_serial_chunk_spans_sink_boundary() {
        // SINK_LEN=32: a 40-token chunk starting at 0 spans into the window.
        let total_len = 40;
        let serial = replay_token_serial(total_len);
        let chunked = replay_chunked(total_len, &[total_len]);
        assert_traces_match(&chunked, &serial, "chunk spans sink boundary");
    }

    #[test]
    fn chunk_append_matches_token_serial_chunk_exactly_fills_window_no_eviction() {
        // A chunk landing exactly on the window boundary must leave the
        // block full but un-evicted (mirrors
        // `window_fills_without_eviction_until_exactly_full`).
        let total_len = SINK_LEN + WINDOW_LEN;
        let serial = replay_token_serial(total_len);
        let chunked = replay_chunked(total_len, &[total_len]);
        assert_traces_match(&chunked, &serial, "chunk exactly fills window");
        assert!(chunked.evictions.is_empty());
    }

    #[test]
    fn chunk_starting_exactly_at_a_just_filled_window_produces_a_zero_length_segment() {
        // Regression test: a chunk whose `pos_base` lands exactly on a
        // "window just became full, not yet evicted" boundary (e.g. a
        // snapshot restore at position `SINK_LEN + WINDOW_LEN`, which is
        // exactly what `snapshot_equivalence.rs`'s mixed-cache split=160
        // scenario does) must produce a *zero-length* first segment whose
        // `evict_after` still fires — the eviction is triggered by the very
        // first position of the new chunk, before any of its own rows are
        // written. `MixedAttnPlane::append_chunk` must skip the
        // zero-length scatter launch (a zero-row kernel launch has a zero
        // grid dimension, which HIP rejects) while still performing the
        // eviction — this test only proves the *plan* has this shape;
        // `append_chunk`'s own handling of it is exercised end-to-end by
        // `snapshot_equivalence.rs` and `mixed_kv_chunked_prefill_parity.rs`.
        let mut layout = MixedLayout::new();
        // First "call" (mirrors a full prefill up to the restore point):
        // exactly fills the window without evicting.
        let (sink_rows, segments) = layout.plan_chunk_append(0, SINK_LEN + WINDOW_LEN);
        assert_eq!(sink_rows, SINK_LEN);
        assert_eq!(segments.len(), 1);
        assert!(segments[0].evict_after.is_none());
        assert_eq!(layout.window_base(), SINK_LEN);

        // Second "call" (mirrors the restored suffix's first chunk): starts
        // exactly at `SINK_LEN + WINDOW_LEN`.
        let (sink_rows2, segments2) = layout.plan_chunk_append(SINK_LEN + WINDOW_LEN, 40);
        assert_eq!(sink_rows2, 0);
        assert_eq!(segments2[0].len, 0, "first segment must be zero-length");
        assert_eq!(segments2[0].evict_after, Some(0));
        assert_eq!(segments2[0].window_slot_start, WINDOW_LEN);
        // The remaining 40 rows land in the freshly-evicted window at slot 0.
        assert_eq!(segments2[1].chunk_row_start, 0);
        assert_eq!(segments2[1].len, 40);
        assert_eq!(segments2[1].window_slot_start, 0);
        assert_eq!(segments2[1].evict_after, None);
        assert_eq!(layout.window_base(), SINK_LEN + WINDOW_LEN);
    }

    #[test]
    fn chunk_append_matches_token_serial_single_chunk_triggers_one_eviction() {
        // One token past the window boundary: exactly one eviction, in one
        // chunk (chunk_len == WINDOW_LEN + 1 spans the boundary).
        let total_len = SINK_LEN + WINDOW_LEN + 1;
        let serial = replay_token_serial(total_len);
        let chunked = replay_chunked(total_len, &[total_len]);
        assert_traces_match(&chunked, &serial, "single chunk triggers one eviction");
        assert_eq!(chunked.evictions, vec![0]);
    }

    #[test]
    fn chunk_append_matches_token_serial_chunk_size_128_matches_window_len() {
        // The production chunk size (PREFILL_CHUNK_SIZE == WINDOW_LEN ==
        // 128): several full chunks, each causing at most one eviction.
        let total_len = SINK_LEN + WINDOW_LEN * 5;
        let serial = replay_token_serial(total_len);
        let mut chunk_lens: Vec<u32> =
            std::iter::repeat_n(128, (total_len / 128) as usize).collect();
        let remainder = total_len % 128;
        if remainder > 0 {
            chunk_lens.push(remainder);
        }
        assert_eq!(chunk_lens.iter().sum::<u32>(), total_len);
        let chunked = replay_chunked(total_len, &chunk_lens);
        assert_traces_match(&chunked, &serial, "chunk size 128 == WINDOW_LEN");
        assert_eq!(chunked.evictions.len(), 4);
    }

    #[test]
    fn chunk_append_matches_token_serial_oversized_chunk_spans_multiple_evictions() {
        // A chunk bigger than WINDOW_LEN (not the production configuration,
        // but `plan_chunk_append` must still get it right — e.g. a future
        // caller with a different chunk size, or `forward_chunk`'s public
        // `1..=CHUNK_CAP` contract): one chunk covering 3+ window's worth
        // must emit multiple segments/evictions.
        let total_len = SINK_LEN + WINDOW_LEN * 3 + 40;
        let serial = replay_token_serial(total_len);
        let chunked = replay_chunked(total_len, &[total_len]);
        assert_traces_match(&chunked, &serial, "oversized chunk, single call");
        assert_eq!(chunked.evictions, vec![0, 1, 2]);
    }

    #[test]
    fn chunk_append_matches_token_serial_across_many_random_chunkings() {
        // Deterministic xorshift64*, no external RNG dependency (matches
        // this workspace's convention elsewhere, e.g. `sample.rs`/
        // `eval::filler`) — split a fixed total length into random chunk
        // sizes (bounded above by WINDOW_LEN, the production case) many
        // times and check every split reproduces the same trace as the
        // token-serial replay.
        let total_len = SINK_LEN + WINDOW_LEN * 6 + 17;
        let serial = replay_token_serial(total_len);

        let mut state = 0x9E3779B97F4A7C15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        for trial in 0..50 {
            let mut chunk_lens = Vec::new();
            let mut remaining = total_len;
            while remaining > 0 {
                let max_take = remaining.min(WINDOW_LEN);
                let take = 1 + (next() % max_take as u64) as u32;
                chunk_lens.push(take);
                remaining -= take;
            }
            let chunked = replay_chunked(total_len, &chunk_lens);
            assert_traces_match(
                &chunked,
                &serial,
                &format!("random chunking trial {trial}: {chunk_lens:?}"),
            );
        }
    }
}
