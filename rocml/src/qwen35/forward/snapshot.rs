//! Capture/restore entry points for the snapshot layer (issue #1) — the
//! qwen35 hybrid architecture's only, per `crate::snapshot`'s module doc.
//! Thin wrappers around `HybridCache::capture_all`/`restore_all`: this file
//! just knows how to turn `Model`'s own fields into those calls' arguments
//! and how to update `self.pos` on restore.

use super::Model;
use crate::error::RocmlError;
use crate::snapshot::SnapshotData;

impl Model {
    /// Captures the model's full current state (all GDN + full-attention
    /// layers) as a [`SnapshotData`] tagged with `token_ids` — the caller
    /// must ensure `token_ids.len() == self.position()` (the exact token
    /// sequence that produced this state), since that's the identity a
    /// later restore's exact-prefix-match relies on.
    pub fn capture_snapshot(&self, token_ids: Vec<u32>) -> Result<SnapshotData, RocmlError> {
        if token_ids.len() as u32 != self.pos {
            return Err(RocmlError::Config(format!(
                "capture_snapshot: token_ids.len() ({}) must equal the current position ({})",
                token_ids.len(),
                self.pos
            )));
        }
        let (gdn, attn) =
            self.cache
                .capture_all(self.config.head_count_kv, self.config.head_dim, self.pos)?;
        Ok(SnapshotData {
            position: self.pos,
            token_ids,
            gdn,
            attn,
        })
    }

    /// Restores a previously captured snapshot, overwriting every layer's
    /// state and setting `self.position()` to `snap.position`. Does not
    /// itself validate `snap`'s token ids against anything — the caller
    /// (`crate::snapshot::turn::run_turn`) is responsible for the
    /// exact-prefix-match lookup that makes restoring `snap` correct for a
    /// given new prompt.
    pub fn restore_snapshot(&mut self, snap: &SnapshotData) -> Result<(), RocmlError> {
        if snap.position > self.cache.max_seq() {
            return Err(crate::error::RocmlError::ContextOverflow {
                requested: snap.position,
                max_seq: self.cache.max_seq(),
            });
        }
        self.cache.restore_all(
            self.config.head_count_kv,
            self.config.head_dim,
            snap.position,
            &snap.gdn,
            &snap.attn,
        )?;
        self.pos = snap.position;
        Ok(())
    }
}
