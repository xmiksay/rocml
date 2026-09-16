//! M4 lever 2's decode-overlap primitives on `ExpertCache` — split out
//! of `moe_cache/mod.rs` purely for the 400-line file cap. A child module
//! of `moe_cache`, so it can reach `ExpertCache`'s private fields directly
//! (Rust's default/private visibility extends to descendant modules).
//! See `super::super::moe_decode_overlap`'s module doc for the full
//! ordering argument these primitives implement.

use rocml_hip::{Event, Stream};

use super::ExpertCache;
use crate::error::RocmlError;
use crate::forward::kernels::{offset, DevPtr};
use crate::qwen35::forward::expert_lru::ExpertKey;
use crate::qwen35::weights::moe::ExpertTensorMeta;
use rocml_core::gguf::GgufFile;

/// Decode-overlap lever (qwen35moe M4 lever 2, off by default —
/// `LoadOptions::moe_decode_overlap`/`--moe-decode-overlap`): a
/// `hipStreamNonBlocking` copy stream plus the two events the pipelined
/// decode loop (`super::moe_decode_overlap`) needs to order a miss's H2D
/// copy against unrelated default-stream compute *without* the host
/// blocking on either. A non-blocking stream does not implicitly
/// synchronize with the legacy default (null) stream every kernel launch in
/// this codebase uses — see `rocml_hip::Stream::new_non_blocking`'s doc
/// comment — so every ordering constraint here is established explicitly
/// via `hipStreamWaitEvent`, never assumed.
pub(super) struct Overlap {
    pub(super) stream: Stream,
    /// Recorded on the *default* stream right after the reads for the
    /// expert currently occupying some slot were issued — before evicting
    /// that slot to start a new copy, the copy stream waits on this so it
    /// can never overwrite bytes a still-in-flight default-stream kernel is
    /// reading. Deliberately one rolling event, not one per slot: it is
    /// recorded after *every* expert's reads regardless of which slot they
    /// used, so waiting on the latest one is a safe (if occasionally more
    /// conservative than strictly necessary) superset of "this specific
    /// slot's last reader is done" — see `super::moe_decode_overlap`'s
    /// module doc for the full ordering argument.
    pub(super) last_compute_done: Event,
    /// Recorded on the copy stream right after a miss's three H2D copies
    /// are enqueued — the default stream waits on this (via
    /// `rocml_hip::wait_on_default_stream`) before reading the
    /// newly-copied slot. One rolling event is sufficient here too: this
    /// design only ever pipelines one expert ahead, so at most one copy is
    /// ever outstanding at a time.
    pub(super) copy_done: Event,
}

impl ExpertCache {
    /// Whether this cache was constructed with the decode-overlap lever on
    /// — `super::moe_decode_overlap` checks this before taking the
    /// pipelined path at all.
    pub(crate) fn overlap_enabled(&self) -> bool {
        self.overlap.is_some()
    }

    /// Touches the LRU for `key` (counting a hit/miss like `ensure_loaded`)
    /// without copying anything — the decode-overlap loop needs to know
    /// *which slot* an expert will occupy, and whether that's already
    /// resident, before deciding whether to start an async copy. Pairs
    /// with [`Self::slot_ptrs`] (to read the slot once ready) and
    /// [`Self::start_copy_async`] (on a miss).
    pub(crate) fn reserve(&mut self, key: ExpertKey) -> (usize, bool) {
        let (slot, hit) = self.lru.get_or_insert(key);
        if hit {
            self.hits += 1;
        } else {
            self.misses += 1;
        }
        (slot, hit)
    }

    /// Device pointers for a slot already reserved via [`Self::reserve`] —
    /// pure address arithmetic, no LRU interaction (so it's safe to call
    /// again after the slot's copy completes, unlike a second `reserve`
    /// which would double-count the hit/miss stats).
    pub(crate) fn slot_ptrs(&self, slot: usize) -> (DevPtr, DevPtr, DevPtr) {
        (
            offset(&self.gate_pool, slot * self.gate_stride),
            offset(&self.up_pool, slot * self.up_stride),
            offset(&self.down_pool, slot * self.down_stride),
        )
    }

    /// Starts `key`'s gate/up/down bytes copying into `slot` on the
    /// overlap copy stream, ordered (via `hipStreamWaitEvent`, not a host
    /// block) after every default-stream read issued so far — see
    /// [`Overlap::last_compute_done`]'s doc comment for why that's always
    /// a sufficient barrier against racing `slot`'s previous occupant.
    /// Returns immediately once the three copies are enqueued; the caller
    /// must wait on [`Self::copy_done_event`] (via
    /// `rocml_hip::wait_on_default_stream`) before any default-stream
    /// kernel reads `slot`. Errors — never panics — if this cache wasn't
    /// constructed with `overlap: true` (an internal-bug case: `overlap_
    /// enabled()` is meant to be checked first).
    pub(crate) fn start_copy_async(
        &mut self,
        gguf: &GgufFile,
        key: ExpertKey,
        slot: usize,
        gate_meta: &ExpertTensorMeta,
        up_meta: &ExpertTensorMeta,
        down_meta: &ExpertTensorMeta,
    ) -> Result<(), RocmlError> {
        let overlap = self.overlap.as_ref().ok_or_else(|| {
            RocmlError::Config(
                "ExpertCache::start_copy_async called without overlap enabled (internal bug)"
                    .into(),
            )
        })?;
        overlap.stream.wait_event(&overlap.last_compute_done)?;
        let (_layer, expert) = key;
        let gate_bytes = gate_meta.expert_bytes(gguf, expert)?;
        let up_bytes = up_meta.expert_bytes(gguf, expert)?;
        let down_bytes = down_meta.expert_bytes(gguf, expert)?;
        // SAFETY: `gate_bytes`/`up_bytes`/`down_bytes` are slices into the
        // GGUF's mmap (`ExpertTensorMeta::expert_bytes`), which lives for
        // the whole model's lifetime — the copy stream reading them
        // asynchronously after this call returns is sound regardless of
        // this borrow's own Rust-checked scope. The destination slot is not
        // read/written by anything else until `copy_done` fires, per the
        // `wait_event` call above (the previous occupant's last reader) and
        // this design's own one-ahead pipelining contract (the caller never
        // starts a second copy into the same slot before the first
        // completes).
        unsafe {
            self.gate_pool.copy_range_from_host_async(
                slot * self.gate_stride,
                gate_bytes,
                &overlap.stream,
            )?;
            self.up_pool.copy_range_from_host_async(
                slot * self.up_stride,
                up_bytes,
                &overlap.stream,
            )?;
            self.down_pool.copy_range_from_host_async(
                slot * self.down_stride,
                down_bytes,
                &overlap.stream,
            )?;
        }
        overlap.copy_done.record(Some(&overlap.stream))?;
        Ok(())
    }

    /// The event a caller waits the default stream on (via
    /// `rocml_hip::wait_on_default_stream`) before reading a slot
    /// [`Self::start_copy_async`] just started copying into.
    pub(crate) fn copy_done_event(&self) -> Option<&Event> {
        self.overlap.as_ref().map(|o| &o.copy_done)
    }

    /// Records that every default-stream kernel issued so far has been
    /// enqueued — call this right after issuing an expert's three
    /// `gemv_quant` reads, so a later [`Self::start_copy_async`] evicting
    /// that expert's slot knows it's safe to overwrite once this event
    /// fires. No-op (not an error) when overlap isn't enabled, so callers
    /// don't need to guard every call site with `overlap_enabled()`.
    pub(crate) fn mark_compute_done(&mut self) -> Result<(), RocmlError> {
        if let Some(overlap) = &self.overlap {
            overlap.last_compute_done.record(None)?;
        }
        Ok(())
    }
}
