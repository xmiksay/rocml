//! `MixedAttnPlane::append_chunk`: the chunked-prefill sibling of
//! `MixedAttnPlane::append` (see that method's doc comment in `mod.rs`) —
//! appends a whole `chunk_len`-token chunk's K/V in one pass instead of a
//! host loop over individual positions.
//!
//! Uses `crate::kv_quant::layout::MixedLayout::plan_chunk_append` to compute
//! the exact same sink/window-slot/eviction plan `chunk_len` sequential
//! `append` calls would produce, then replays it as a handful of batched
//! kernel launches: `ChunkKernels::scatter_kv_chunk_f16` (already used by
//! the dense chunked-prefill path, `attention_chunk.rs`) for the sink/window
//! writes, and the *existing* `MixedKernels::quantize_evict_k`/`_v` — the
//! same per-block quantize-on-evict kernels `append`'s per-token path
//! already calls one block at a time — for bulk eviction. No new kernels
//! were needed for the write path at all: `quantize_evict_k`/`_v` already
//! process a whole `WINDOW_LEN`-token block in one launch (that's the
//! "batched" design issue #2 shipped with from the start, see
//! `kernels/kv_quant.hip`'s module doc), and `scatter_kv_chunk_f16` already
//! batches an arbitrary row range's fp32->fp16 cast-and-copy. This file only
//! changes *how many rows* one launch covers and *how many launches* the
//! whole chunk takes — never the per-element arithmetic any single launch
//! performs — which is exactly what makes the resulting cache state
//! bit-identical to `chunk_len` sequential `append` calls (the gate
//! `rocml/tests/mixed_kv_chunked_prefill_parity.rs` checks via
//! `Model::capture_snapshot`'s raw `AttnLayerBytes` equality).
use rocml_hip::DeviceBuffer;

use super::MixedAttnPlane;
use crate::error::RocmlError;
use crate::forward::kernels::offset;
use crate::kv_quant::{SINK_LEN, WINDOW_LEN};
use crate::qwen35::forward::chunk_kernels::ChunkKernels;
use crate::qwen35::forward::kernels_mixed::MixedKernels;

impl MixedAttnPlane {
    /// Appends `chunk_len` new positions' K/V — `k_src`/`v_src` are the
    /// chunk's own freshly-computed `[chunk_len, n_kv_heads, head_dim]`
    /// buffers, the same layout `ChunkKernels::scatter_kv_chunk_f16` already
    /// expects for the dense cache. `pos_base` must be this plane's current
    /// append cursor: `chunk_len` sequential `append(pos_base+i, ...)`
    /// calls (`i` in `0..chunk_len`) are the equivalence this method is
    /// tested against, both at the kernel level
    /// (`rocml-kernels/tests/kv_quant_chunk.rs`) and the model level
    /// (`mixed_kv_chunked_prefill_parity.rs`).
    pub fn append_chunk(
        &mut self,
        chunk: &ChunkKernels,
        mixed: &MixedKernels,
        pos_base: u32,
        chunk_len: u32,
        k_src: &DeviceBuffer<f32>,
        v_src: &DeviceBuffer<f32>,
    ) -> Result<(), RocmlError> {
        if chunk_len == 0 {
            return Ok(());
        }
        let (n_kv_heads, head_dim) = (self.n_kv_heads, self.head_dim);
        let row_elems = (n_kv_heads * head_dim) as usize;
        let row_offset = |row: u32| -> usize { row as usize * row_elems };

        let (sink_rows, segments) = self.layout.plan_chunk_append(pos_base, chunk_len);

        if sink_rows > 0 {
            chunk.scatter_kv_chunk_f16(
                offset(k_src, 0),
                offset(v_src, 0),
                offset(&self.sink_k, 0),
                offset(&self.sink_v, 0),
                n_kv_heads,
                head_dim,
                SINK_LEN,
                sink_rows,
                pos_base,
            )?;
        }

        for seg in &segments {
            // `seg.len == 0` is a real, reachable case (not a defensive
            // guard against something that "shouldn't happen"): it fires
            // whenever a chunk's own `pos_base` lands exactly on a
            // window-just-became-full boundary — e.g. a snapshot restore at
            // position `SINK_LEN + WINDOW_LEN` (see
            // `mixed_kv_chunked_prefill_parity.rs`'s split-boundary
            // coverage, and `snapshot_equivalence.rs`'s mixed-cache
            // scenario, which is what caught this) — where the *previous*
            // append already filled the window completely without
            // triggering the eviction (per `prepare_append`'s own
            // semantics, eviction only fires on the position that would
            // overflow it). The very first row of the new chunk is then
            // exactly that overflow-triggering position: zero new rows land
            // in the about-to-be-evicted block, only the eviction itself
            // happens, before any of this chunk's own rows are written. A
            // zero-length `scatter_kv_chunk_f16` launch has a zero grid
            // dimension, which HIP rejects outright (`invalid argument`),
            // so it must be skipped — the eviction below still needs to run
            // unconditionally, since the block is fully populated already
            // (just not from this chunk).
            if seg.len > 0 {
                chunk.scatter_kv_chunk_f16(
                    offset(k_src, row_offset(seg.chunk_row_start)),
                    offset(v_src, row_offset(seg.chunk_row_start)),
                    offset(&self.window_k, 0),
                    offset(&self.window_v, 0),
                    n_kv_heads,
                    head_dim,
                    WINDOW_LEN,
                    seg.len,
                    seg.window_slot_start,
                )?;
            }
            if let Some(block) = seg.evict_after {
                mixed.quantize_evict_k(
                    offset(&self.window_k, 0),
                    offset(&self.bulk_k_codes, 0),
                    offset(&self.bulk_k_scales, 0),
                    n_kv_heads,
                    WINDOW_LEN,
                    head_dim,
                    self.bulk_cap,
                    self.num_blocks_total,
                    block,
                )?;
                mixed.quantize_evict_v(
                    self.v_bits,
                    offset(&self.window_v, 0),
                    offset(&self.bulk_v_codes, 0),
                    offset(&self.bulk_v_scales, 0),
                    n_kv_heads,
                    WINDOW_LEN,
                    head_dim,
                    self.bulk_cap,
                    block,
                )?;
            }
        }
        Ok(())
    }
}
