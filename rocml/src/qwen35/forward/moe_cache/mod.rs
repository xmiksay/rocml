//! VRAM-resident LRU cache of routed-expert weights (M2) — the qwen35moe
//! forward pass's single biggest lever over M1's copy-every-token design
//! (see `.claude/CLAUDE.md`'s M2 section): a `(layer, expert)` cache hit
//! skips the H2D copy entirely and runs `gemv_quant`/`gemm_quant` straight
//! against the slot the expert's bytes already sit in from a previous
//! token — M1 re-copied every selected expert's ~1.8 MiB every single call.
//!
//! One "slot" is three regions (gate/up/down) inside three separate
//! `DeviceBuffer<u8>` pools, each strided at the *model-wide* max
//! per-expert byte size for that tensor kind (a layer whose `ffn_down_exps`
//! happens to be Q4_K instead of another layer's Q6_K, per llama.cpp's
//! `Q4_K_M` importance heuristic, just uses less than its slot's full
//! stride — harmless, `gemv_quant`/`gemm_quant` only ever read the `m*n`-
//! element prefix their own shape needs, the same reasoning `MoeScratch`'s
//! own oversized buffers already relied on in M1).

mod overlap;

use rocml_core::gguf::GgufFile;
use rocml_hip::{DeviceBuffer, Event, Stream};

use super::expert_lru::{ExpertKey, ExpertLru};
use crate::error::RocmlError;
use crate::forward::kernels::{offset, DevPtr};
use crate::qwen35::weights::moe::ExpertTensorMeta;
use overlap::Overlap;

/// VRAM headroom left unused when auto-sizing the cache from free memory at
/// load time (`Model::load`) — insurance against driver bookkeeping and
/// measurement noise between the `hipMemGetInfo` call and the pool's own
/// `hipMalloc`s, on top of every other buffer's own already-reserved
/// headroom. This pool is sized from the actual free bytes left at the very
/// end of load (after weights/KV cache/scratch/rewind reservation), not an
/// upfront estimate, so it can afford a smaller margin than
/// `crate::budget`'s pre-KV activation headroom.
pub const EXPERT_CACHE_SAFETY_MARGIN_BYTES: usize = 512 * 1024 * 1024;

pub struct ExpertCache {
    lru: ExpertLru,
    gate_pool: DeviceBuffer<u8>,
    up_pool: DeviceBuffer<u8>,
    down_pool: DeviceBuffer<u8>,
    gate_stride: usize,
    up_stride: usize,
    down_stride: usize,
    hits: u64,
    misses: u64,
    overlap: Option<Overlap>,
}

impl ExpertCache {
    /// `capacity` slots, each `gate_stride + up_stride + down_stride` bytes
    /// (the model-wide per-tensor-kind maxima — see the module doc).
    /// Returns `Ok(None)` for `capacity == 0` (no VRAM left to cache
    /// anything, or an explicit `--moe-cache-slots 0` override) rather than
    /// allocating a degenerate zero-slot cache — callers fall back to
    /// `MoeScratch`'s single-slot stage buffers in that case (the M1 path).
    /// `overlap` (M4 lever 2, default `false` end to end) allocates the
    /// non-blocking copy stream and events `super::moe_decode_overlap`
    /// needs; `false` costs nothing extra.
    pub fn new(
        capacity: usize,
        gate_stride: usize,
        up_stride: usize,
        down_stride: usize,
        overlap: bool,
    ) -> Result<Option<Self>, RocmlError> {
        if capacity == 0 {
            return Ok(None);
        }
        let overlap = if overlap {
            let last_compute_done = Event::new()?;
            // Recorded once, right here, on the default stream: the very
            // first `start_copy_async` call (before any real per-token
            // compute has run) needs a valid, already-fired event to wait
            // on rather than one that was created but never recorded —
            // `hipStreamWaitEvent` on an unrecorded event is undefined.
            last_compute_done.record(None)?;
            Some(Overlap {
                stream: Stream::new_non_blocking()?,
                last_compute_done,
                copy_done: Event::new()?,
            })
        } else {
            None
        };
        Ok(Some(Self {
            lru: ExpertLru::new(capacity),
            gate_pool: DeviceBuffer::new(capacity * gate_stride)?,
            up_pool: DeviceBuffer::new(capacity * up_stride)?,
            down_pool: DeviceBuffer::new(capacity * down_stride)?,
            gate_stride,
            up_stride,
            down_stride,
            hits: 0,
            misses: 0,
            overlap,
        }))
    }

    pub fn capacity(&self) -> usize {
        self.lru.capacity()
    }

    /// Slots currently holding a live entry — `<= capacity()`, and equal to
    /// it once the cache has been touched by at least `capacity()` distinct
    /// experts.
    pub fn occupancy(&self) -> usize {
        self.lru.occupancy()
    }

    /// `(hits, misses)` since load — for `bench`/logging to report the
    /// measured cache hit rate (see `.claude/CLAUDE.md`'s M2 measurement
    /// protocol).
    pub fn stats(&self) -> (u64, u64) {
        (self.hits, self.misses)
    }

    /// Returns device pointers to `key`'s gate/up/down bytes, copying them
    /// from the GGUF's registered mmap on a miss. The pointers are valid
    /// only until the next call to this method (a subsequent miss may
    /// evict and overwrite the same slot), so callers must finish using
    /// them (launch every kernel that reads them) before calling again.
    pub fn ensure_loaded(
        &mut self,
        gguf: &GgufFile,
        key: ExpertKey,
        gate_meta: &ExpertTensorMeta,
        up_meta: &ExpertTensorMeta,
        down_meta: &ExpertTensorMeta,
    ) -> Result<(DevPtr, DevPtr, DevPtr), RocmlError> {
        let (slot, hit) = self.lru.get_or_insert(key);
        if hit {
            self.hits += 1;
        } else {
            self.misses += 1;
            let (_layer, expert) = key;
            let gate_bytes = gate_meta.expert_bytes(gguf, expert)?;
            let up_bytes = up_meta.expert_bytes(gguf, expert)?;
            let down_bytes = down_meta.expert_bytes(gguf, expert)?;
            self.gate_pool
                .copy_range_from_host(slot * self.gate_stride, gate_bytes)?;
            self.up_pool
                .copy_range_from_host(slot * self.up_stride, up_bytes)?;
            self.down_pool
                .copy_range_from_host(slot * self.down_stride, down_bytes)?;
        }
        Ok((
            offset(&self.gate_pool, slot * self.gate_stride),
            offset(&self.up_pool, slot * self.up_stride),
            offset(&self.down_pool, slot * self.down_stride),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen35::config::Qwen35Config;
    use rocml_core::testpaths::checkpoint;
    use rocml_hip::Device;

    const GGUF_REL: &str = "Ornith-1.5-35B-A3B-GGUF/Ornith-1.5-35B-Q4_K_M.gguf";

    fn load_metas(
        gguf: &GgufFile,
        cfg: &Qwen35Config,
        prefix: &str,
    ) -> (ExpertTensorMeta, ExpertTensorMeta, ExpertTensorMeta) {
        let moe = cfg.moe.expect("qwen35moe checkpoint");
        let gate = ExpertTensorMeta::load(
            gguf,
            &format!("{prefix}.ffn_gate_exps.weight"),
            moe.expert_ff_len,
            cfg.embedding_length,
            moe.expert_count,
        )
        .expect("load gate meta");
        let up = ExpertTensorMeta::load(
            gguf,
            &format!("{prefix}.ffn_up_exps.weight"),
            moe.expert_ff_len,
            cfg.embedding_length,
            moe.expert_count,
        )
        .expect("load up meta");
        let down = ExpertTensorMeta::load(
            gguf,
            &format!("{prefix}.ffn_down_exps.weight"),
            cfg.embedding_length,
            moe.expert_ff_len,
            moe.expert_count,
        )
        .expect("load down meta");
        (gate, up, down)
    }

    #[test]
    fn zero_capacity_yields_none() {
        let cache = ExpertCache::new(0, 1024, 1024, 1024, false).expect("alloc");
        assert!(cache.is_none());
    }

    #[test]
    fn cache_hit_serves_bit_identical_bytes_without_a_recopy() {
        let Some(path) = checkpoint(GGUF_REL) else {
            return;
        };
        let _device = Device::new(0).expect("device");
        let gguf = GgufFile::open(&path).expect("open gguf");
        let cfg = Qwen35Config::from_gguf(&gguf).expect("cfg");
        let (gate0, up0, down0) = load_metas(&gguf, &cfg, "blk.0");
        let (gate1, up1, down1) = load_metas(&gguf, &cfg, "blk.1");

        let gate_stride = gate0.per_expert_bytes.max(gate1.per_expert_bytes);
        let up_stride = up0.per_expert_bytes.max(up1.per_expert_bytes);
        let down_stride = down0.per_expert_bytes.max(down1.per_expert_bytes);

        let mut cache = ExpertCache::new(2, gate_stride, up_stride, down_stride, false)
            .expect("alloc")
            .expect("capacity 2 must yield Some");

        let key = (0u32, 5u32);
        cache
            .ensure_loaded(&gguf, key, &gate0, &up0, &down0)
            .expect("miss load");
        assert_eq!(cache.stats(), (0, 1));

        let mut readback = vec![0u8; gate0.per_expert_bytes];
        cache
            .gate_pool
            .copy_range_to_host(0, &mut readback)
            .expect("read back gate slot");
        assert_eq!(readback, gate0.expert_bytes(&gguf, 5).unwrap());

        // Second lookup for the same key must be a hit and must not touch
        // the pool again (the bytes already there are still correct).
        cache
            .ensure_loaded(&gguf, key, &gate0, &up0, &down0)
            .expect("hit load");
        assert_eq!(cache.stats(), (1, 1));

        // A different layer's expert must be a fresh miss and land in a
        // distinct slot's bytes.
        let key2 = (1u32, 9u32);
        cache
            .ensure_loaded(&gguf, key2, &gate1, &up1, &down1)
            .expect("second miss load");
        assert_eq!(cache.stats(), (1, 2));
        let mut readback2 = vec![0u8; gate1.per_expert_bytes];
        cache
            .gate_pool
            .copy_range_to_host(gate_stride, &mut readback2)
            .expect("read back second gate slot");
        assert_eq!(readback2, gate1.expert_bytes(&gguf, 9).unwrap());
    }

    #[test]
    fn eviction_forces_a_fresh_miss_and_copy() {
        let Some(path) = checkpoint(GGUF_REL) else {
            return;
        };
        let _device = Device::new(0).expect("device");
        let gguf = GgufFile::open(&path).expect("open gguf");
        let cfg = Qwen35Config::from_gguf(&gguf).expect("cfg");
        let (gate, up, down) = load_metas(&gguf, &cfg, "blk.0");

        let mut cache = ExpertCache::new(
            1,
            gate.per_expert_bytes,
            up.per_expert_bytes,
            down.per_expert_bytes,
            false,
        )
        .expect("alloc")
        .expect("capacity 1 must yield Some");

        cache
            .ensure_loaded(&gguf, (0, 1), &gate, &up, &down)
            .unwrap();
        cache
            .ensure_loaded(&gguf, (0, 2), &gate, &up, &down)
            .unwrap();
        assert_eq!(cache.stats(), (0, 2), "single slot must thrash, not hit");
        // Content must now match expert 2, not the evicted expert 1.
        let mut readback = vec![0u8; gate.per_expert_bytes];
        cache
            .gate_pool
            .copy_range_to_host(0, &mut readback)
            .unwrap();
        assert_eq!(readback, gate.expert_bytes(&gguf, 2).unwrap());
    }
}
