//! Host-RAM LRU snapshot store: a byte-budgeted map keyed by (model stamp,
//! KV config, prefix hash, position), evicting the least-recently-*read*
//! entry (capture counts as a write+touch) once `budget_bytes` is exceeded.
//! `budget_bytes == 0` disables the tier entirely (every insert is a no-op,
//! every lookup misses) — see `--snapshot-ram-mb 0` in `rocml-serve`/`rocml-cli`.

use std::collections::HashMap;
use std::sync::Arc;

use super::hash::ChainHash;
use super::types::{KvConfigStamp, ModelStamp, SnapshotData, SnapshotKey};

struct Entry {
    data: Arc<SnapshotData>,
    bytes: usize,
    last_used: u64,
}

pub struct RamStore {
    budget_bytes: usize,
    used_bytes: usize,
    tick: u64,
    entries: HashMap<SnapshotKey, Entry>,
}

impl RamStore {
    pub fn new(budget_bytes: usize) -> Self {
        Self {
            budget_bytes,
            used_bytes: 0,
            tick: 0,
            entries: HashMap::new(),
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.budget_bytes > 0
    }

    pub fn used_bytes(&self) -> usize {
        self.used_bytes
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn next_tick(&mut self) -> u64 {
        self.tick += 1;
        self.tick
    }

    /// Inserts `data` under `(model, kv, position, token hash of the whole
    /// prefix)`, evicting least-recently-used entries first until the new
    /// entry fits the budget. A single entry larger than the whole budget is
    /// simply not stored (never partially evicts everything else for
    /// nothing) — the RAM tier is a cache, not a durability guarantee.
    pub fn insert(&mut self, model: ModelStamp, kv: KvConfigStamp, data: SnapshotData) {
        if self.budget_bytes == 0 {
            return;
        }
        let bytes = data.byte_size();
        if bytes > self.budget_bytes {
            return;
        }
        let key = SnapshotKey {
            model,
            kv,
            position: data.position,
            chain_hash: ChainHash::of(&data.token_ids),
        };
        self.remove(&key);
        while self.used_bytes + bytes > self.budget_bytes {
            if !self.evict_one() {
                break;
            }
        }
        let last_used = self.next_tick();
        self.used_bytes += bytes;
        self.entries.insert(
            key,
            Entry {
                data: Arc::new(data),
                bytes,
                last_used,
            },
        );
    }

    fn remove(&mut self, key: &SnapshotKey) {
        if let Some(entry) = self.entries.remove(key) {
            self.used_bytes -= entry.bytes;
        }
    }

    fn evict_one(&mut self) -> bool {
        let victim = self
            .entries
            .iter()
            .min_by_key(|(_, e)| e.last_used)
            .map(|(k, _)| k.clone());
        match victim {
            Some(k) => {
                self.remove(&k);
                true
            }
            None => false,
        }
    }

    /// Longest exact-prefix match for `token_ids` among snapshots captured
    /// from the same `(model, kv)`. Only positions strictly shorter than
    /// `token_ids.len()` are considered — a hit always leaves at least one
    /// token of suffix for the caller to prefill (see `crate::snapshot`
    /// module doc's restore contract); a caller that also wants a full-length
    /// exact match should look up `&token_ids[..token_ids.len() - 1]`
    /// instead. A touched entry's LRU clock is bumped.
    pub fn find_best(
        &mut self,
        model: &ModelStamp,
        kv: &KvConfigStamp,
        token_ids: &[u32],
    ) -> Option<Arc<SnapshotData>> {
        if token_ids.is_empty() {
            return None;
        }
        let max_len = token_ids.len() - 1;
        // Longest-first: build checkpoints once, then walk candidate
        // positions from longest to shortest so the first exact match wins.
        let mut positions: Vec<u32> = self
            .entries
            .keys()
            .filter(|k| &k.model == model && &k.kv == kv && (k.position as usize) <= max_len)
            .map(|k| k.position)
            .collect();
        positions.sort_unstable_by(|a, b| b.cmp(a));
        positions.dedup();

        for position in positions {
            let hash = ChainHash::of(&token_ids[..position as usize]);
            let key = SnapshotKey {
                model: model.clone(),
                kv: *kv,
                position,
                chain_hash: hash,
            };
            let matched = self
                .entries
                .get(&key)
                .filter(|entry| entry.data.token_ids == token_ids[..position as usize])
                .map(|entry| entry.data.clone());
            if let Some(data) = matched {
                let tick = self.next_tick();
                self.entries
                    .get_mut(&key)
                    .expect("just looked up")
                    .last_used = tick;
                return Some(data);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stamp() -> ModelStamp {
        ModelStamp {
            gguf_path: "/checkpoints/model.gguf".to_string(),
            file_len: 123,
        }
    }

    fn kv() -> KvConfigStamp {
        KvConfigStamp {
            mode: crate::load_opts::KvCacheMode::Fp16,
            ctx: 8192,
        }
    }

    fn fake_snapshot(token_ids: Vec<u32>, filler_f32_count: usize) -> SnapshotData {
        SnapshotData {
            position: token_ids.len() as u32,
            token_ids,
            gdn: vec![Some(super::super::types::GdnLayerBytes {
                conv_state: vec![0.0; filler_f32_count],
                state: vec![0.0; filler_f32_count],
            })],
            attn: vec![None],
        }
    }

    #[test]
    fn exact_prefix_match_wins_and_longest_wins() {
        let mut store = RamStore::new(1_000_000);
        let full: Vec<u32> = (0..100).collect();
        store.insert(stamp(), kv(), fake_snapshot(full[..40].to_vec(), 4));
        store.insert(stamp(), kv(), fake_snapshot(full[..80].to_vec(), 4));

        let hit = store
            .find_best(&stamp(), &kv(), &full)
            .expect("expected a hit");
        assert_eq!(hit.position, 80);
    }

    #[test]
    fn diverging_suffix_falls_back_to_shorter_match() {
        let mut store = RamStore::new(1_000_000);
        let full: Vec<u32> = (0..100).collect();
        let mut diverged = full[..80].to_vec();
        diverged[75] += 1; // diverges before position 80
        store.insert(stamp(), kv(), fake_snapshot(full[..40].to_vec(), 4));
        store.insert(stamp(), kv(), fake_snapshot(diverged, 4));

        let hit = store
            .find_best(&stamp(), &kv(), &full)
            .expect("expected a hit at the shorter, non-diverging prefix");
        assert_eq!(hit.position, 40);
    }

    #[test]
    fn model_or_kv_mismatch_is_a_miss() {
        let mut store = RamStore::new(1_000_000);
        let full: Vec<u32> = (0..50).collect();
        store.insert(stamp(), kv(), fake_snapshot(full[..30].to_vec(), 4));

        let other_model = ModelStamp {
            gguf_path: "/checkpoints/other.gguf".to_string(),
            file_len: 456,
        };
        assert!(store.find_best(&other_model, &kv(), &full).is_none());

        let other_kv = KvConfigStamp {
            mode: crate::load_opts::KvCacheMode::Q8,
            ctx: 8192,
        };
        assert!(store.find_best(&stamp(), &other_kv, &full).is_none());
    }

    #[test]
    fn full_length_prefix_is_never_offered_as_a_hit() {
        let mut store = RamStore::new(1_000_000);
        let full: Vec<u32> = (0..20).collect();
        store.insert(stamp(), kv(), fake_snapshot(full.clone(), 4));
        // The only stored snapshot covers the whole prompt — a hit here
        // would leave nothing to prefill.
        assert!(store.find_best(&stamp(), &kv(), &full).is_none());
    }

    #[test]
    fn lru_eviction_drops_the_least_recently_used_entry() {
        // Every fake snapshot has the same token count, so they're all the
        // same byte size — exactly two fit the budget.
        let per_entry_floats = 100usize;
        let bytes_per_entry = fake_snapshot(vec![1; 10], per_entry_floats).byte_size();
        let mut store = RamStore::new(bytes_per_entry * 2 + 8);

        store.insert(stamp(), kv(), fake_snapshot(vec![1; 10], per_entry_floats));
        store.insert(stamp(), kv(), fake_snapshot(vec![2; 10], per_entry_floats));
        // Touch the first one so it's more recently used than the second.
        assert!(store
            .find_best(&stamp(), &kv(), &[1; 11])
            .is_some_and(|s| s.position == 10));
        // Inserting a third entry must evict the second (least recently used).
        store.insert(stamp(), kv(), fake_snapshot(vec![3; 10], per_entry_floats));

        assert!(store.find_best(&stamp(), &kv(), &[1; 11]).is_some());
        assert!(store.find_best(&stamp(), &kv(), &[2; 11]).is_none());
        assert!(store.find_best(&stamp(), &kv(), &[3; 11]).is_some());
    }

    #[test]
    fn repeated_lookup_returns_the_same_immutable_data() {
        let mut store = RamStore::new(1_000_000);
        let full: Vec<u32> = (0..10).collect();
        store.insert(stamp(), kv(), fake_snapshot(full[..5].to_vec(), 4));
        let first = store.find_best(&stamp(), &kv(), &full).unwrap();
        let second = store.find_best(&stamp(), &kv(), &full).unwrap();
        assert!(
            Arc::ptr_eq(&first, &second),
            "restore must never mutate a stored snapshot"
        );
        assert_eq!(*first, *second);
    }

    #[test]
    fn zero_budget_disables_the_store() {
        let mut store = RamStore::new(0);
        assert!(!store.is_enabled());
        store.insert(stamp(), kv(), fake_snapshot(vec![1, 2, 3], 4));
        assert_eq!(store.len(), 0);
        assert!(store.find_best(&stamp(), &kv(), &[1, 2, 3, 4]).is_none());
    }
}
