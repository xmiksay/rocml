//! Pure data types for a captured conversation-state snapshot: identity
//! (model + KV config + prefix), and the per-layer byte payload. No HIP/GPU
//! types here — `qwen35::cache`/`cache_mixed` fill these in via D2H copies
//! (see `qwen35::forward::Model::capture_snapshot`), and everything below is
//! plain owned host memory, testable without a GPU.

use std::path::Path;

use half::f16;

use crate::load_opts::KvCacheMode;

/// Cheap, load-bearing identity for the GGUF a snapshot was captured
/// against: path plus file size, per issue #1's "gguf path + file size or a
/// cheap content stamp" — deliberately not a full-file hash (that would
/// defeat the point of a *cheap* stamp for a multi-GB checkpoint).
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ModelStamp {
    pub gguf_path: String,
    pub file_len: u64,
}

impl ModelStamp {
    pub fn from_path(path: &Path) -> std::io::Result<Self> {
        let file_len = std::fs::metadata(path)?.len();
        Ok(Self {
            gguf_path: path.to_string_lossy().into_owned(),
            file_len,
        })
    }
}

/// The load-time KV cache configuration a snapshot's byte layout depends on.
/// `ctx` matters, not just `KvCacheMode`: `AttnPlane`/`MixedAttnPlane` buffer
/// offsets (and, for the mixed layout, `bulk_cap`) are sized from `ctx`, so a
/// snapshot captured at one `ctx` can't be safely restored into a cache
/// allocated for another — a mismatch here is a documented miss, not a best
/// effort reshape (see `SnapshotStore::lookup`'s doc comment).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct KvConfigStamp {
    pub mode: KvCacheMode,
    pub ctx: usize,
}

/// One GDN (linear-attention) layer's conv/recurrence state — always
/// captured whole: both buffers are O(1) in position (see `qwen35::cache`'s
/// module doc), so there's no "filled prefix" concept here, unlike the
/// attention planes below.
#[derive(Debug, Clone, PartialEq)]
pub struct GdnLayerBytes {
    pub conv_state: Vec<f32>,
    pub state: Vec<f32>,
}

impl GdnLayerBytes {
    pub fn byte_size(&self) -> usize {
        (self.conv_state.len() + self.state.len()) * 4
    }
}

/// One full-attention layer's KV state, in whichever storage the cache used
/// at capture time (`crate::cache::KvDtype` for `Dense`, KIVI-style mixed
/// for `Mixed` — mirrors `qwen35::cache::AttnLayerCache`). Every `Vec` here
/// holds only the *filled* prefix (`[0, N)` positions, or for `Mixed`'s bulk
/// region, only the evicted-so-far blocks) — never the cache's full
/// allocated capacity, per issue #1's sizing note.
#[derive(Debug, Clone, PartialEq)]
pub enum AttnLayerBytes {
    DenseF16 {
        k: Vec<f16>,
        v: Vec<f16>,
    },
    DenseF32 {
        k: Vec<f32>,
        v: Vec<f32>,
    },
    Mixed {
        sink_k: Vec<f16>,
        sink_v: Vec<f16>,
        window_k: Vec<f16>,
        window_v: Vec<f16>,
        bulk_k_codes: Vec<i8>,
        bulk_k_scales: Vec<f32>,
        bulk_v_codes: Vec<u8>,
        bulk_v_scales: Vec<f32>,
        /// `MixedLayout::window_base()` at capture time — the one scalar
        /// needed to reconstruct the whole eviction bookkeeping on restore
        /// (`MixedLayout::from_window_base`).
        window_base: u32,
        v_bits: u8,
    },
}

impl AttnLayerBytes {
    pub fn byte_size(&self) -> usize {
        match self {
            Self::DenseF16 { k, v } => (k.len() + v.len()) * 2,
            Self::DenseF32 { k, v } => (k.len() + v.len()) * 4,
            Self::Mixed {
                sink_k,
                sink_v,
                window_k,
                window_v,
                bulk_k_codes,
                bulk_k_scales,
                bulk_v_codes,
                bulk_v_scales,
                ..
            } => {
                (sink_k.len() + sink_v.len() + window_k.len() + window_v.len()) * 2
                    + bulk_k_codes.len()
                    + bulk_k_scales.len() * 4
                    + bulk_v_codes.len()
                    + bulk_v_scales.len() * 4
            }
        }
    }
}

/// The full content of one immutable snapshot: position, the exact token ids
/// of the prefix it was captured from (stored in full — see issue #1's
/// design note on why this beats storing only a hash), and every layer's
/// state. `gdn`/`attn` are indexed by layer (mirroring
/// `Qwen35Config::layer_kinds`); exactly one of the pair is `Some` per index.
#[derive(Debug, Clone, PartialEq)]
pub struct SnapshotData {
    pub position: u32,
    pub token_ids: Vec<u32>,
    pub gdn: Vec<Option<GdnLayerBytes>>,
    pub attn: Vec<Option<AttnLayerBytes>>,
}

impl SnapshotData {
    pub fn byte_size(&self) -> usize {
        let layer_bytes: usize = self
            .gdn
            .iter()
            .flatten()
            .map(GdnLayerBytes::byte_size)
            .sum::<usize>()
            + self
                .attn
                .iter()
                .flatten()
                .map(AttnLayerBytes::byte_size)
                .sum::<usize>();
        layer_bytes + self.token_ids.len() * 4
    }
}

/// Identity key a stored snapshot is addressed by. `chain_hash` is a fast
/// pre-filter only (see `crate::snapshot::hash`'s module doc) — the actual
/// correctness guarantee ("exact match required") comes from comparing
/// `SnapshotData::token_ids` byte-for-byte, done by `SnapshotStore::lookup`
/// before ever accepting a candidate.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SnapshotKey {
    pub model: ModelStamp,
    pub kv: KvConfigStamp,
    pub position: u32,
    pub chain_hash: u64,
}
