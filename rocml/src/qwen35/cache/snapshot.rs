//! Capture/restore methods for every type in `super` (issue #1) — split into
//! this child module purely for the parent file's 400-line cap; being a
//! child (not a sibling) module is what lets these `impl` blocks reach the
//! parent's private fields (`AttnPlane::storage`, `HybridCache`'s `gdn`/
//! `attn`/`max_seq`) the same way any other code in `cache` can.
//!
//! Every "capture" copies device state out to owned host `Vec`s (only the
//! *filled* prefix for attention planes — GDN state is O(1) in position, so
//! it's always captured whole); every "restore" writes them back. Both
//! directions validate shape (length/dtype/layer-kind) and return
//! `RocmlError` rather than panicking — a mismatched snapshot is a bug
//! (`ModelStamp`/`KvConfigStamp` should have already gated it out at lookup
//! time), but restore is on a path reachable from a corrupted/tampered disk
//! snapshot, so it must degrade to an error, never a crash.

use super::*;
use crate::snapshot::{AttnLayerBytes, GdnLayerBytes};

impl GdnLayerState {
    /// Whole-buffer D2H copy — GDN state is O(1) in position, so unlike
    /// `AttnPlane::capture` there's no "filled prefix" to slice out.
    pub(super) fn capture(&self) -> Result<GdnLayerBytes, RocmlError> {
        let mut conv_state = vec![0.0f32; self.conv_len];
        let mut state = vec![0.0f32; self.state_len];
        self.conv_state.copy_to_host(&mut conv_state)?;
        self.state.copy_to_host(&mut state)?;
        Ok(GdnLayerBytes { conv_state, state })
    }

    pub(super) fn restore(&mut self, bytes: &GdnLayerBytes) -> Result<(), RocmlError> {
        if bytes.conv_state.len() != self.conv_len || bytes.state.len() != self.state_len {
            return Err(RocmlError::Config(format!(
                "gdn state restore: length mismatch (conv_state {} vs expected {}, state {} vs \
                 expected {})",
                bytes.conv_state.len(),
                self.conv_len,
                bytes.state.len(),
                self.state_len
            )));
        }
        self.conv_state.copy_from_host(&bytes.conv_state)?;
        self.state.copy_from_host(&bytes.state)?;
        Ok(())
    }
}

impl AttnPlane {
    /// Copies out only the *filled* `[0, filled)` prefix of every kv head's
    /// plane — never the whole allocated `max_seq` capacity, per
    /// `crate::snapshot`'s sizing note. `filled` is normally the cache's
    /// current position (`Model::position()`), passed in rather than tracked
    /// here since this cache has no position concept of its own (decode and
    /// chunked prefill both drive it from the outside).
    pub(super) fn capture(
        &self,
        n_kv_heads: u32,
        max_seq: u32,
        head_dim: u32,
        filled: u32,
    ) -> Result<AttnLayerBytes, RocmlError> {
        let (head_dim_u, max_seq_u, filled_u, heads) = (
            head_dim as usize,
            max_seq as usize,
            filled as usize,
            n_kv_heads as usize,
        );
        match &self.storage {
            PlaneStorage::F16 { k, v } => {
                let mut k_out = vec![f16::from_f32(0.0); heads * filled_u * head_dim_u];
                let mut v_out = k_out.clone();
                for h in 0..heads {
                    let src = h * max_seq_u * head_dim_u;
                    let dst = h * filled_u * head_dim_u;
                    let dst_slice = dst..dst + filled_u * head_dim_u;
                    k.copy_range_to_host(src, &mut k_out[dst_slice.clone()])?;
                    v.copy_range_to_host(src, &mut v_out[dst_slice])?;
                }
                Ok(AttnLayerBytes::DenseF16 { k: k_out, v: v_out })
            }
            PlaneStorage::F32 { k, v } => {
                let mut k_out = vec![0.0f32; heads * filled_u * head_dim_u];
                let mut v_out = k_out.clone();
                for h in 0..heads {
                    let src = h * max_seq_u * head_dim_u;
                    let dst = h * filled_u * head_dim_u;
                    let dst_slice = dst..dst + filled_u * head_dim_u;
                    k.copy_range_to_host(src, &mut k_out[dst_slice.clone()])?;
                    v.copy_range_to_host(src, &mut v_out[dst_slice])?;
                }
                Ok(AttnLayerBytes::DenseF32 { k: k_out, v: v_out })
            }
        }
    }

    /// Writes a captured `[0, filled)` prefix back — the storage dtype in
    /// `bytes` must match this plane's own (mismatched dtype is a config
    /// error, not silently reinterpreted: a snapshot captured under one
    /// `KvCacheMode` was already gated out by `KvConfigStamp` equality
    /// before this is ever called, so this branch existing at all means an
    /// internal bug).
    pub(super) fn restore(
        &mut self,
        bytes: &AttnLayerBytes,
        n_kv_heads: u32,
        max_seq: u32,
        head_dim: u32,
        filled: u32,
    ) -> Result<(), RocmlError> {
        let (head_dim_u, max_seq_u, filled_u, heads) = (
            head_dim as usize,
            max_seq as usize,
            filled as usize,
            n_kv_heads as usize,
        );
        let expected = heads * filled_u * head_dim_u;
        match (&mut self.storage, bytes) {
            (PlaneStorage::F16 { k, v }, AttnLayerBytes::DenseF16 { k: ks, v: vs }) => {
                check_len(ks.len(), expected)?;
                check_len(vs.len(), expected)?;
                for h in 0..heads {
                    let dst = h * max_seq_u * head_dim_u;
                    let src = h * filled_u * head_dim_u;
                    let src_slice = src..src + filled_u * head_dim_u;
                    k.copy_range_from_host(dst, &ks[src_slice.clone()])?;
                    v.copy_range_from_host(dst, &vs[src_slice])?;
                }
                Ok(())
            }
            (PlaneStorage::F32 { k, v }, AttnLayerBytes::DenseF32 { k: ks, v: vs }) => {
                check_len(ks.len(), expected)?;
                check_len(vs.len(), expected)?;
                for h in 0..heads {
                    let dst = h * max_seq_u * head_dim_u;
                    let src = h * filled_u * head_dim_u;
                    let src_slice = src..src + filled_u * head_dim_u;
                    k.copy_range_from_host(dst, &ks[src_slice.clone()])?;
                    v.copy_range_from_host(dst, &vs[src_slice])?;
                }
                Ok(())
            }
            _ => Err(RocmlError::Config(
                "attn plane restore: snapshot dtype doesn't match this cache's storage dtype \
                 (internal bug — KvConfigStamp should have gated this)"
                    .to_string(),
            )),
        }
    }
}

impl AttnPlane {
    /// Byte size of what [`Self::capture`] would return at `filled`.
    pub(super) fn snapshot_byte_size(&self, n_kv_heads: u32, head_dim: u32, filled: u32) -> usize {
        let elems = n_kv_heads as usize * filled as usize * head_dim as usize;
        match self.dtype() {
            KvDtype::F16 => elems * 2 * 2,
            KvDtype::F32 => elems * 4 * 2,
        }
    }
}

impl AttnLayerCache {
    pub(super) fn capture(
        &self,
        n_kv_heads: u32,
        max_seq: u32,
        head_dim: u32,
        filled: u32,
    ) -> Result<AttnLayerBytes, RocmlError> {
        match self {
            Self::Dense(plane) => plane.capture(n_kv_heads, max_seq, head_dim, filled),
            Self::Mixed(plane) => plane.capture(),
        }
    }

    pub(super) fn restore(
        &mut self,
        bytes: &AttnLayerBytes,
        n_kv_heads: u32,
        max_seq: u32,
        head_dim: u32,
        filled: u32,
    ) -> Result<(), RocmlError> {
        match self {
            Self::Dense(plane) => plane.restore(bytes, n_kv_heads, max_seq, head_dim, filled),
            Self::Mixed(plane) => plane.restore(bytes),
        }
    }

    pub(super) fn snapshot_byte_size(&self, n_kv_heads: u32, head_dim: u32, filled: u32) -> usize {
        match self {
            Self::Dense(plane) => plane.snapshot_byte_size(n_kv_heads, head_dim, filled),
            Self::Mixed(plane) => plane.snapshot_byte_size(),
        }
    }
}

fn check_len(actual: usize, expected: usize) -> Result<(), RocmlError> {
    if actual != expected {
        return Err(RocmlError::Config(format!(
            "snapshot restore: expected {expected} elements, got {actual}"
        )));
    }
    Ok(())
}

/// Every layer's captured state, indexed by layer index (mirroring
/// `Qwen35Config::layer_kinds` — exactly one of the pair is `Some` per
/// index). Named purely so `capture_all`'s signature doesn't trip clippy's
/// `type_complexity` lint.
type CapturedLayers = (Vec<Option<GdnLayerBytes>>, Vec<Option<AttnLayerBytes>>);

impl HybridCache {
    /// D2H-copies every layer's state at `filled` (the snapshot position —
    /// see `qwen35::forward::Model::capture_snapshot`).
    pub fn capture_all(
        &self,
        n_kv_heads: u32,
        head_dim: u32,
        filled: u32,
    ) -> Result<CapturedLayers, RocmlError> {
        let gdn = self
            .gdn
            .iter()
            .map(|slot| slot.as_ref().map(GdnLayerState::capture).transpose())
            .collect::<Result<Vec<_>, _>>()?;
        let attn = self
            .attn
            .iter()
            .map(|slot| {
                slot.as_ref()
                    .map(|c| c.capture(n_kv_heads, self.max_seq, head_dim, filled))
                    .transpose()
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((gdn, attn))
    }

    /// Exactly `capture_all(..)`'s total `byte_size()` at `filled`, with no
    /// D2H traffic — see `qwen35::forward::Model::snapshot_byte_size`.
    pub fn snapshot_byte_size(&self, n_kv_heads: u32, head_dim: u32, filled: u32) -> usize {
        let gdn: usize = self
            .gdn
            .iter()
            .flatten()
            .map(|s| (s.conv_len + s.state_len) * 4)
            .sum();
        let attn: usize = self
            .attn
            .iter()
            .flatten()
            .map(|c| c.snapshot_byte_size(n_kv_heads, head_dim, filled))
            .sum();
        gdn + attn
    }

    /// H2D-writes a captured snapshot's per-layer state back — errors (never
    /// panics) on a layer-count or layer-kind mismatch, which would mean the
    /// snapshot was captured from a differently-shaped model than this cache
    /// (shouldn't happen once `ModelStamp`/`KvConfigStamp` have already
    /// gated the lookup, but this is the last line of defense against a
    /// corrupted/mismatched snapshot ever silently misinterpreting bytes).
    pub fn restore_all(
        &mut self,
        n_kv_heads: u32,
        head_dim: u32,
        filled: u32,
        gdn: &[Option<GdnLayerBytes>],
        attn: &[Option<AttnLayerBytes>],
    ) -> Result<(), RocmlError> {
        if gdn.len() != self.gdn.len() || attn.len() != self.attn.len() {
            return Err(RocmlError::Config(format!(
                "snapshot restore: layer count mismatch (gdn {} vs {}, attn {} vs {})",
                gdn.len(),
                self.gdn.len(),
                attn.len(),
                self.attn.len()
            )));
        }
        let max_seq = self.max_seq;
        for (slot, bytes) in self.gdn.iter_mut().zip(gdn) {
            match (slot, bytes) {
                (Some(s), Some(b)) => s.restore(b)?,
                (None, None) => {}
                _ => {
                    return Err(RocmlError::Config(
                        "snapshot restore: gdn layer-kind mismatch".to_string(),
                    ))
                }
            }
        }
        for (slot, bytes) in self.attn.iter_mut().zip(attn) {
            match (slot, bytes) {
                (Some(s), Some(b)) => s.restore(b, n_kv_heads, max_seq, head_dim, filled)?,
                (None, None) => {}
                _ => {
                    return Err(RocmlError::Config(
                        "snapshot restore: attn layer-kind mismatch".to_string(),
                    ))
                }
            }
        }
        Ok(())
    }
}
