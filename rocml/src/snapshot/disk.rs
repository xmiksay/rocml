//! Optional NVMe snapshot tier (`--snapshot-dir`, off by default): one
//! content-addressed file per snapshot (hash in the filename) plus a JSON
//! index (`index.json`) recording each file's identity/size/access order for
//! size-budgeted LRU eviction — an atime-equivalent this crate tracks itself
//! rather than relying on filesystem atime (often mounted `noatime`).
//!
//! Corruption safety is the load-bearing property here: a truncated or
//! bit-flipped file must never crash the caller. Every read frame-checks a
//! magic/version header and an FNV-1a checksum ([`hash::checksum_bytes`])
//! before handing the payload to [`codec::decode`]; any failure at any of
//! those stages deletes the offending file, drops it from the index, and is
//! reported to the caller as a plain miss — see [`DiskStore::find_best`].
//! Index corruption gets the same treatment one level up: an unreadable or
//! malformed `index.json` just starts a fresh, empty index rather than
//! failing to open the store.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::codec;
use super::hash::{checksum_bytes, ChainHash};
use super::types::{KvConfigStamp, ModelStamp, SnapshotData};
use crate::error::RocmlError;

const MAGIC: &[u8; 8] = b"ROCMLSN1";

fn frame(payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + 8 + 8 + payload.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&checksum_bytes(payload).to_le_bytes());
    out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    out.extend_from_slice(payload);
    out
}

/// Validates the frame header and checksum, returning the payload slice on
/// success. Any structural problem (short file, bad magic, length mismatch,
/// checksum mismatch) is reported uniformly — the caller doesn't need to
/// distinguish "truncated" from "bit-flipped", both mean "corrupt, delete it".
fn unframe(bytes: &[u8]) -> Option<&[u8]> {
    let header_len = 8 + 8 + 8;
    if bytes.len() < header_len {
        return None;
    }
    if &bytes[0..8] != MAGIC {
        return None;
    }
    let checksum = u64::from_le_bytes(bytes[8..16].try_into().ok()?);
    let payload_len = usize::try_from(u64::from_le_bytes(bytes[16..24].try_into().ok()?)).ok()?;
    let payload = bytes.get(header_len..header_len + payload_len)?;
    if bytes.len() != header_len + payload_len {
        return None; // trailing garbage or truncation — treat as corrupt either way
    }
    if checksum_bytes(payload) != checksum {
        return None;
    }
    Some(payload)
}

#[derive(Serialize, Deserialize, Clone)]
struct DiskEntry {
    file_name: String,
    model: ModelStamp,
    kv: KvConfigStamp,
    position: u32,
    token_ids: Vec<u32>,
    byte_size: u64,
    last_used: u64,
}

#[derive(Serialize, Deserialize, Default)]
struct DiskIndex {
    entries: Vec<DiskEntry>,
    #[serde(default)]
    tick: u64,
}

pub struct DiskStore {
    dir: PathBuf,
    budget_bytes: u64,
    index: DiskIndex,
}

impl DiskStore {
    /// Creates `dir` if missing (a real failure here — e.g. an unwritable
    /// path — is a startup config error worth surfacing) and loads
    /// `index.json`, tolerating a missing or corrupt index by starting fresh.
    pub fn open(dir: PathBuf, budget_bytes: u64) -> Result<Self, RocmlError> {
        std::fs::create_dir_all(&dir).map_err(|e| {
            RocmlError::Config(format!(
                "--snapshot-dir {}: failed to create directory: {e}",
                dir.display()
            ))
        })?;
        let index = Self::load_index(&dir);
        Ok(Self {
            dir,
            budget_bytes,
            index,
        })
    }

    fn index_path(dir: &Path) -> PathBuf {
        dir.join("index.json")
    }

    fn load_index(dir: &Path) -> DiskIndex {
        match std::fs::read(Self::index_path(dir)) {
            Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_else(|e| {
                eprintln!(
                    "snapshot disk store: {} is corrupt ({e}), starting with an empty index",
                    Self::index_path(dir).display()
                );
                DiskIndex::default()
            }),
            Err(_) => DiskIndex::default(),
        }
    }

    fn save_index(&self) {
        match serde_json::to_vec_pretty(&self.index) {
            Ok(bytes) => {
                if let Err(e) = std::fs::write(Self::index_path(&self.dir), bytes) {
                    eprintln!("snapshot disk store: failed to write index.json: {e}");
                }
            }
            Err(e) => eprintln!("snapshot disk store: failed to serialize index.json: {e}"),
        }
    }

    fn file_name(model: &ModelStamp, kv: &KvConfigStamp, position: u32, chain_hash: u64) -> String {
        let identity = format!(
            "{}:{}:{:?}:{}",
            model.gguf_path, model.file_len, kv.mode, kv.ctx
        );
        let model_kv_id = checksum_bytes(identity.as_bytes());
        format!("{model_kv_id:016x}-{chain_hash:016x}-{position}.snap")
    }

    pub fn used_bytes(&self) -> u64 {
        self.index.entries.iter().map(|e| e.byte_size).sum()
    }

    pub fn len(&self) -> usize {
        self.index.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.entries.is_empty()
    }

    /// Persists `data` content-addressed by `(model, kv, position, token
    /// hash)`, then evicts oldest-touched entries until back under budget.
    /// `budget_bytes == 0` disables the tier (nothing is written) — mirrors
    /// `RamStore`'s `budget_bytes == 0` convention.
    pub fn insert(&mut self, model: ModelStamp, kv: KvConfigStamp, data: &SnapshotData) {
        if self.budget_bytes == 0 {
            return;
        }
        let chain_hash = ChainHash::of(&data.token_ids);
        let file_name = Self::file_name(&model, &kv, data.position, chain_hash);
        let framed = frame(&codec::encode(data));
        let byte_size = framed.len() as u64;
        if byte_size > self.budget_bytes {
            return; // a single snapshot bigger than the whole disk budget is never stored
        }
        if let Err(e) = std::fs::write(self.dir.join(&file_name), &framed) {
            eprintln!("snapshot disk store: failed to write {file_name}: {e}");
            return;
        }
        self.index.entries.retain(|e| e.file_name != file_name);
        self.index.tick += 1;
        self.index.entries.push(DiskEntry {
            file_name,
            model,
            kv,
            position: data.position,
            token_ids: data.token_ids.clone(),
            byte_size,
            last_used: self.index.tick,
        });
        self.evict_to_budget();
        self.save_index();
    }

    fn evict_to_budget(&mut self) {
        while self.used_bytes() > self.budget_bytes {
            let victim = self
                .index
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.last_used)
                .map(|(i, _)| i);
            match victim {
                Some(i) => {
                    let entry = self.index.entries.remove(i);
                    let _ = std::fs::remove_file(self.dir.join(&entry.file_name));
                }
                None => break,
            }
        }
    }

    fn remove_entry(&mut self, file_name: &str) {
        self.index.entries.retain(|e| e.file_name != file_name);
        let _ = std::fs::remove_file(self.dir.join(file_name));
    }

    fn touch(&mut self, file_name: &str) {
        self.index.tick += 1;
        let tick = self.index.tick;
        if let Some(e) = self
            .index
            .entries
            .iter_mut()
            .find(|e| e.file_name == file_name)
        {
            e.last_used = tick;
        }
    }

    /// Reads and frame/checksum-validates `file_name`, decoding it on
    /// success. Any failure (missing file, bad frame, bad checksum, decode
    /// error) returns `None` — the caller treats that as corruption and
    /// evicts the entry, never propagating an error for what is, from the
    /// store's perspective, just a miss.
    fn read_valid(&self, file_name: &str) -> Option<SnapshotData> {
        let bytes = std::fs::read(self.dir.join(file_name)).ok()?;
        let payload = unframe(&bytes)?;
        codec::decode(payload).ok()
    }

    /// Longest exact-prefix match, same contract as `RamStore::find_best`
    /// (only positions `< token_ids.len()` are candidates). A corrupt file
    /// found along the way is deleted and skipped in favor of the next
    /// (shorter) candidate rather than failing the whole lookup.
    pub fn find_best(
        &mut self,
        model: &ModelStamp,
        kv: &KvConfigStamp,
        token_ids: &[u32],
    ) -> Option<SnapshotData> {
        if token_ids.is_empty() {
            return None;
        }
        let max_len = token_ids.len() - 1;
        let mut candidates: Vec<(u32, String)> = self
            .index
            .entries
            .iter()
            .filter(|e| {
                &e.model == model
                    && &e.kv == kv
                    && (e.position as usize) <= max_len
                    && e.token_ids == token_ids[..e.position as usize]
            })
            .map(|e| (e.position, e.file_name.clone()))
            .collect();
        candidates.sort_unstable_by_key(|(pos, _)| std::cmp::Reverse(*pos));

        for (_, file_name) in candidates {
            match self.read_valid(&file_name) {
                Some(data) => {
                    self.touch(&file_name);
                    self.save_index();
                    return Some(data);
                }
                None => {
                    eprintln!(
                        "snapshot disk store: {file_name} failed validation, deleting and treating as a miss"
                    );
                    self.remove_entry(&file_name);
                    self.save_index();
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load_opts::KvCacheMode;

    fn stamp() -> ModelStamp {
        ModelStamp {
            gguf_path: "/checkpoints/model.gguf".to_string(),
            file_len: 999,
        }
    }
    fn kv() -> KvConfigStamp {
        KvConfigStamp {
            mode: KvCacheMode::Fp16,
            ctx: 4096,
        }
    }
    fn fake(token_ids: Vec<u32>) -> SnapshotData {
        SnapshotData {
            position: token_ids.len() as u32,
            token_ids,
            gdn: vec![Some(super::super::types::GdnLayerBytes {
                conv_state: vec![1.0; 16],
                state: vec![2.0; 16],
            })],
            attn: vec![None],
        }
    }

    fn tempdir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "rocml-snapshot-disk-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn roundtrips_through_disk() {
        let dir = tempdir();
        let mut store = DiskStore::open(dir.clone(), 10_000_000).unwrap();
        let full: Vec<u32> = (0..50).collect();
        store.insert(stamp(), kv(), &fake(full[..30].to_vec()));

        let hit = store.find_best(&stamp(), &kv(), &full).expect("miss");
        assert_eq!(hit.position, 30);
        assert_eq!(hit.token_ids, full[..30]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn corrupted_file_is_deleted_and_treated_as_a_miss() {
        let dir = tempdir();
        let mut store = DiskStore::open(dir.clone(), 10_000_000).unwrap();
        let full: Vec<u32> = (0..20).collect();
        store.insert(stamp(), kv(), &fake(full[..10].to_vec()));
        let file_name = store.index.entries[0].file_name.clone();
        let path = dir.join(&file_name);

        let mut bytes = std::fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF; // flip a byte in the payload
        std::fs::write(&path, &bytes).unwrap();

        assert!(store.find_best(&stamp(), &kv(), &full).is_none());
        assert!(!path.exists(), "corrupt file should have been deleted");
        assert!(store.index.entries.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn corrupt_index_json_starts_fresh_instead_of_failing() {
        let dir = tempdir();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("index.json"), b"not json at all {{{").unwrap();
        let store = DiskStore::open(dir.clone(), 10_000_000).unwrap();
        assert_eq!(store.len(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn size_budgeted_eviction_drops_oldest_touched_entry() {
        let dir = tempdir();
        // Every entry has the same token count, so they're all the same
        // framed byte size — exactly two fit the budget.
        let one_entry_bytes = frame(&codec::encode(&fake((0..10).collect()))).len() as u64;
        let mut store = DiskStore::open(dir.clone(), one_entry_bytes * 2 + 8).unwrap();

        store.insert(stamp(), kv(), &fake((100..110).collect()));
        store.insert(stamp(), kv(), &fake((200..210).collect()));
        // Touch the first so the second becomes the LRU victim.
        assert!(store
            .find_best(&stamp(), &kv(), &(100..111).collect::<Vec<_>>())
            .is_some());
        store.insert(stamp(), kv(), &fake((300..310).collect()));

        assert!(store
            .find_best(&stamp(), &kv(), &(100..111).collect::<Vec<_>>())
            .is_some());
        assert!(store
            .find_best(&stamp(), &kv(), &(200..211).collect::<Vec<_>>())
            .is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn zero_budget_disables_writes() {
        let dir = tempdir();
        let mut store = DiskStore::open(dir.clone(), 0).unwrap();
        store.insert(stamp(), kv(), &fake((0..10).collect()));
        assert_eq!(store.len(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
