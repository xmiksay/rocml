//! Conversation-state snapshot layer (issue #1): immutable checkpoints of
//! the qwen35 hybrid model's full decode state (GDN recurrence + linear
//! full-attention KV) at a given token position, addressed by an exact
//! token-id prefix match. See `qwen35::forward::Model::capture_snapshot`/
//! `restore_snapshot` for how the GPU-side bytes are produced/consumed, and
//! `crate::snapshot::turn` for how a server/CLI turn wires lookup + restore
//! + periodic capture around `crate::generate`'s decode loop.
//!
//! Dense `qwen3` is out of scope for v1 (see the issue): its KV cache is
//! O(position) per layer with no GDN-style fixed-size state to make a cheap
//! mid-conversation checkpoint interesting, and its forward pass has no
//! chunked prefill to hook capture into. `crate::model::Model::as_hybrid`/
//! `as_hybrid_mut` is how callers detect which architecture they have and
//! skip snapshot logic entirely for `Dense`.
//!
//! Three tiers. The hot one lives on the model, not in this store:
//! - **GPU** (`qwen35::forward::Model::save_rewind_point`/`rewind_to_prefix`):
//!   two on-device rewind points (render-stable boundary, end of turn) a
//!   growing single-session conversation hits without any host traffic —
//!   see `qwen35::forward::rewind`. Enabled whenever a `SnapshotStore` is
//!   in use at all; `run_turn` consults it before either tier below.
//! - **RAM** ([`ram::RamStore`]): always constructed, `--snapshot-ram-mb 0`
//!   disables it (every insert/lookup becomes a no-op/miss). Holds one
//!   *pinned* entry per turn (the snapshot the next turn is expected to hit)
//!   that plain LRU eviction can't displace — see that module's doc.
//! - **Disk** ([`disk::DiskStore`], `--snapshot-dir`): off unless a
//!   directory is given: content-addressed files + a JSON index, corruption
//!   never propagates past a deleted file and a miss.
//!
//! Lookup contract (both tiers): "longest exact prefix match" — see
//! [`SnapshotStore::lookup`].

mod codec;
mod disk;
pub mod hash;
mod ram;
pub mod turn;
mod types;

use std::path::PathBuf;
use std::sync::Arc;

pub use disk::DiskStore;
pub use ram::RamStore;
pub use types::{AttnLayerBytes, GdnLayerBytes, KvConfigStamp, ModelStamp, SnapshotData};

use crate::error::RocmlError;

/// Which host tier served a [`SnapshotStore::lookup_with_tier`] hit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SnapshotTier {
    Ram,
    Disk,
}

pub struct SnapshotStore {
    ram: RamStore,
    disk: Option<DiskStore>,
}

impl SnapshotStore {
    pub fn new(
        ram_budget_bytes: usize,
        disk_dir: Option<PathBuf>,
        disk_budget_bytes: u64,
    ) -> Result<Self, RocmlError> {
        let disk = disk_dir
            .map(|dir| DiskStore::open(dir, disk_budget_bytes))
            .transpose()?;
        Ok(Self {
            ram: RamStore::new(ram_budget_bytes),
            disk,
        })
    }

    pub fn ram_used_bytes(&self) -> usize {
        self.ram.used_bytes()
    }

    pub fn disk_used_bytes(&self) -> u64 {
        self.disk.as_ref().map(DiskStore::used_bytes).unwrap_or(0)
    }

    /// Longest exact-prefix match for `token_ids` against snapshots captured
    /// from the same `(model, kv)` — "exact" means `token_ids[..N]` equals
    /// the stored prefix element-for-element, not just a hash match (the
    /// chain hash is only ever a bucketing pre-filter, see `hash`'s module
    /// doc). Only positions strictly shorter than `token_ids.len()` are ever
    /// returned, guaranteeing the caller always has at least one token left
    /// to prefill — see `ram::RamStore::find_best`'s doc comment.
    ///
    /// Checks RAM first; if RAM's best match is shorter than what the disk
    /// tier holds, the longer disk snapshot is loaded, promoted into RAM
    /// (subject to its own budget), and returned instead — the disk tier is
    /// for cross-run persistence, but a hit there should still get the RAM
    /// tier's cheaper access on the very next lookup.
    pub fn lookup(
        &mut self,
        model: &ModelStamp,
        kv: &KvConfigStamp,
        token_ids: &[u32],
    ) -> Option<Arc<SnapshotData>> {
        self.lookup_with_tier(model, kv, token_ids)
            .map(|(data, _)| data)
    }

    /// [`Self::lookup`] plus which tier the winning snapshot came from (a
    /// disk hit reports `Disk` even though it's promoted into RAM on the
    /// way out) — for the per-request "snapshot hit" log line.
    pub fn lookup_with_tier(
        &mut self,
        model: &ModelStamp,
        kv: &KvConfigStamp,
        token_ids: &[u32],
    ) -> Option<(Arc<SnapshotData>, SnapshotTier)> {
        let ram_hit = self
            .ram
            .find_best(model, kv, token_ids)
            .map(|d| (d, SnapshotTier::Ram));
        let ram_pos = ram_hit.as_ref().map(|(d, _)| d.position).unwrap_or(0);

        let Some(disk) = self.disk.as_mut() else {
            return ram_hit;
        };
        let disk_hit = disk.find_best(model, kv, token_ids);
        match disk_hit {
            Some(data) if data.position > ram_pos => {
                self.ram.insert(model.clone(), *kv, data.clone());
                Some((Arc::new(data), SnapshotTier::Disk))
            }
            _ => ram_hit,
        }
    }

    /// Stores `data` into both tiers (RAM always attempted; disk only when
    /// configured) — see each tier's own `insert` for its eviction policy.
    /// Cheap to call even when both tiers are disabled (`budget_bytes == 0`
    /// / no `--snapshot-dir`): both `insert`s become no-ops.
    pub fn insert(&mut self, model: ModelStamp, kv: KvConfigStamp, data: SnapshotData) {
        if let Some(disk) = self.disk.as_mut() {
            disk.insert(model.clone(), kv, &data);
        }
        self.ram.insert(model, kv, data);
    }

    /// Like [`Self::insert`], but the RAM copy becomes the store's pinned
    /// entry (see `ram::RamStore::insert_pinned`). Disk is unaffected: its
    /// own size-budgeted LRU already spans runs, and a hit there is promoted
    /// back into RAM on lookup.
    pub fn insert_pinned(&mut self, model: ModelStamp, kv: KvConfigStamp, data: SnapshotData) {
        if let Some(disk) = self.disk.as_mut() {
            disk.insert(model.clone(), kv, &data);
        }
        self.ram.insert_pinned(model, kv, data);
    }

    /// Whether a RAM entry of `bytes` could be stored without displacing the
    /// pinned one.
    pub fn ram_fits_beside_pinned(&self, bytes: usize) -> bool {
        self.ram.fits_beside_pinned(bytes)
    }

    /// Whether [`Self::insert`] would keep an unpinned entry of `bytes` in
    /// either tier — `run_turn`'s gate for the periodic mid-prefill capture,
    /// checked *before* paying for the D2H copy. The disk tier counts: it is
    /// the only one that survives a restart, which is the point of that
    /// crash-insurance capture.
    pub fn would_keep(&self, bytes: usize) -> bool {
        self.disk.is_some() || self.ram.fits_beside_pinned(bytes)
    }

    /// Whether [`Self::insert_pinned`] would keep an entry of `bytes` in
    /// either tier — `run_turn` checks this before the D2H capture, so a
    /// snapshot too big for the RAM budget with no disk tier (e.g. the
    /// GPU-only `--snapshot-ram-mb 1` setup) costs nothing instead of an
    /// O(position) copy that is immediately dropped.
    pub fn would_keep_pinned(&self, bytes: usize) -> bool {
        self.disk.is_some() || self.ram.can_pin(bytes)
    }

    pub fn ram_pinned_position(&self) -> Option<u32> {
        self.ram.pinned_position()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load_opts::KvCacheMode;

    fn stamp() -> ModelStamp {
        ModelStamp {
            gguf_path: "/checkpoints/m.gguf".to_string(),
            file_len: 1,
        }
    }
    fn kv() -> KvConfigStamp {
        KvConfigStamp {
            mode: KvCacheMode::Fp16,
            ctx: 2048,
        }
    }
    fn fake(token_ids: Vec<u32>) -> SnapshotData {
        SnapshotData {
            position: token_ids.len() as u32,
            token_ids,
            gdn: vec![Some(GdnLayerBytes {
                conv_state: vec![0.0; 4],
                state: vec![0.0; 4],
            })],
            attn: vec![None],
        }
    }

    #[test]
    fn ram_only_store_serves_hits() {
        let mut store = SnapshotStore::new(1_000_000, None, 0).unwrap();
        let full: Vec<u32> = (0..30).collect();
        store.insert(stamp(), kv(), fake(full[..15].to_vec()));
        let hit = store.lookup(&stamp(), &kv(), &full).unwrap();
        assert_eq!(hit.position, 15);
    }

    #[test]
    fn disk_promotes_a_longer_match_into_ram() {
        let dir = std::env::temp_dir().join(format!(
            "rocml-snapshot-store-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let mut store = SnapshotStore::new(1_000_000, Some(dir.clone()), 10_000_000).unwrap();
        let full: Vec<u32> = (0..30).collect();

        // Insert directly into disk only, bypassing RAM, to simulate a
        // snapshot persisted by a previous process run.
        store
            .disk
            .as_mut()
            .unwrap()
            .insert(stamp(), kv(), &fake(full[..20].to_vec()));
        store.ram.insert(stamp(), kv(), fake(full[..5].to_vec()));

        let hit = store.lookup(&stamp(), &kv(), &full).unwrap();
        assert_eq!(
            hit.position, 20,
            "disk's longer match should win over RAM's shorter one"
        );

        // Promoted into RAM: a disk-disabled follow-up lookup would need it
        // there, so check RAM directly now holds position 20 too.
        let ram_hit = store.ram.find_best(&stamp(), &kv(), &full).unwrap();
        assert_eq!(ram_hit.position, 20);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The production failure this pinning exists for: at a budget that
    /// holds only two of a turn's three equally sized captures, plain LRU
    /// evicts the first (stable-boundary) one — the only one the next turn
    /// can hit.
    #[test]
    fn pinned_entry_survives_a_turn_of_larger_later_captures() {
        let one = fake((0..10).collect()).byte_size();
        let mut store = SnapshotStore::new(one * 2 + one / 2, None, 0).unwrap();
        let full: Vec<u32> = (0..40).collect();
        store.insert_pinned(stamp(), kv(), fake(full[..10].to_vec()));
        store.insert(stamp(), kv(), fake(full[..11].to_vec()));
        store.insert(stamp(), kv(), fake(full[..12].to_vec()));
        assert_eq!(store.ram_pinned_position(), Some(10));
        let (hit, tier) = store
            .lookup_with_tier(&stamp(), &kv(), &full[..11])
            .unwrap();
        assert_eq!((hit.position, tier), (10, SnapshotTier::Ram));
        // The later, unpinned captures competed only with each other.
        assert!(store.lookup(&stamp(), &kv(), &full[..12]).unwrap().position >= 10);
    }

    #[test]
    fn insert_that_cannot_fit_beside_the_pin_is_dropped_not_swapped() {
        let full: Vec<u32> = (0..40).collect();
        let small = fake(full[..10].to_vec()).byte_size();
        let large = fake(full[..12].to_vec()).byte_size();
        // Room for either alone, never for both.
        let mut store = SnapshotStore::new(small + large - 1, None, 0).unwrap();
        store.insert_pinned(stamp(), kv(), fake(full[..10].to_vec()));
        assert!(!store.ram_fits_beside_pinned(large));
        store.insert(stamp(), kv(), fake(full[..12].to_vec()));
        assert_eq!(
            store.lookup(&stamp(), &kv(), &full).unwrap().position,
            10,
            "an unpinned insert that doesn't fit beside the pin must not displace it"
        );
        // A new pin does replace the old one.
        store.insert_pinned(stamp(), kv(), fake(full[..12].to_vec()));
        assert_eq!(store.ram_pinned_position(), Some(12));
        assert_eq!(store.lookup(&stamp(), &kv(), &full).unwrap().position, 12);
    }

    #[test]
    fn would_keep_pinned_rejects_oversized_entries_only_without_disk() {
        let bytes = fake((0..10).collect()).byte_size();
        let store = SnapshotStore::new(bytes, None, 0).unwrap();
        assert!(store.would_keep_pinned(bytes));
        assert!(!store.would_keep_pinned(bytes + 1));
        assert!(!SnapshotStore::new(0, None, 0).unwrap().would_keep_pinned(1));

        let dir =
            std::env::temp_dir().join(format!("rocml-snapshot-would-keep-{}", std::process::id()));
        let with_disk = SnapshotStore::new(0, Some(dir.clone()), 1 << 20).unwrap();
        assert!(with_disk.would_keep_pinned(bytes + 1));
        // Disk-only: the periodic capture's gate must not be RAM-only.
        assert!(with_disk.would_keep(bytes + 1));
        assert!(!store.would_keep(bytes + 1));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn disabled_disk_tier_still_serves_ram() {
        let mut store = SnapshotStore::new(1_000_000, None, 0).unwrap();
        assert_eq!(store.disk_used_bytes(), 0);
        store.insert(stamp(), kv(), fake((0..10).collect()));
        assert!(store
            .lookup(&stamp(), &kv(), &(0..11).collect::<Vec<_>>())
            .is_some());
    }
}
