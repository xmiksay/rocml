//! `MixedAttnPlane`: one full-attention layer's KIVI-style quantized KV
//! storage (issue #2) — split out of `cache.rs` purely for the workspace's
//! 400-line file cap. See `crate::kv_quant::layout`'s module doc for the
//! sink/bulk/window model this implements, and `kernels/kv_quant.hip`/
//! `kernels/attn_decode_mixed.hip`'s module docs for the exact buffer
//! layouts this allocates and feeds.
//!
//! `chunk` (child module, split out purely for the 400-line file cap) adds
//! [`MixedAttnPlane::append_chunk`], the chunked-prefill sibling of
//! [`MixedAttnPlane::append`] below — see that module's doc comment for the
//! batched-append design and the bit-identity-with-token-serial invariant
//! it must uphold. `rot_sim` (also split out for the 400-line cap) adds
//! [`MixedAttnPlane::apply_rot_sim`], issue #14 phase 2's debug quality
//! simulation hook that both `append` and `append_chunk` call right before
//! a real window eviction. `snapshot` (also split out for the 400-line cap)
//! adds [`MixedAttnPlane::capture`]/[`MixedAttnPlane::restore`], issue #1's
//! snapshot layer hooks — see that module's doc comment for why the bulk
//! region's capture is sliced to its filled prefix rather than the whole
//! `bulk_cap`-sized allocation (issue #12).

mod chunk;
mod rot_sim;
mod snapshot;

use half::f16;
use rocml_hip::DeviceBuffer;

use super::forward::kernels_mixed::MixedKernels;
use crate::error::RocmlError;
use crate::forward::kernels::{offset, DevPtr, Kernels};
use crate::kv_quant::rotational::RotSimSpec;
use crate::kv_quant::MixedLayout;

/// Raw device pointers plus the shape/state scalars
/// `Kernels`/`MixedKernels`' fused-attention launch needs — bundled so the
/// decode-attention call site doesn't have to know `MixedAttnPlane`'s
/// internal field names.
pub struct MixedPtrs {
    pub sink_k: DevPtr,
    pub sink_v: DevPtr,
    pub window_k: DevPtr,
    pub window_v: DevPtr,
    pub bulk_k_codes: DevPtr,
    pub bulk_k_scales: DevPtr,
    pub bulk_v_codes: DevPtr,
    pub bulk_v_scales: DevPtr,
    pub sink_len: u32,
    pub window_len: u32,
    pub window_base: u32,
    pub bulk_cap: u32,
    pub num_blocks_total: u32,
    pub v_bits: u8,
}

pub struct MixedAttnPlane {
    sink_k: DeviceBuffer<f16>,
    sink_v: DeviceBuffer<f16>,
    window_k: DeviceBuffer<f16>,
    window_v: DeviceBuffer<f16>,
    bulk_k_codes: DeviceBuffer<i8>,
    bulk_k_scales: DeviceBuffer<f32>,
    /// V codes: `n_kv_heads * bulk_cap * head_dim` bytes at `v_bits == 8`
    /// (one i8 per element, reinterpreted from u8), or
    /// `n_kv_heads * bulk_cap * head_dim/2` bytes at `v_bits == 4` (packed).
    bulk_v_codes: DeviceBuffer<u8>,
    bulk_v_scales: DeviceBuffer<f32>,
    layout: MixedLayout,
    v_bits: u8,
    n_kv_heads: u32,
    head_dim: u32,
    sink_len: u32,
    window_len: u32,
    bulk_cap: u32,
    num_blocks_total: u32,
    rot_sim: Option<RotSimSpec>,
}

impl MixedAttnPlane {
    /// `max_seq` is this cache's overall context budget — sizes the bulk
    /// region's capacity (rounded up to a whole number of `window_len`
    /// blocks) so every position past the sink can eventually be evicted
    /// into it. `sink_len`/`window_len` are issue #2 leftovers'
    /// `LoadOptions::kv_sink`/`kv_window` (defaulting to
    /// `crate::kv_quant::{SINK_LEN, WINDOW_LEN}`) — validated once by the
    /// caller (`crate::kv_quant::validate_sink_window`) before this is ever
    /// reached.
    pub fn new(
        n_kv_heads: u32,
        head_dim: u32,
        max_seq: u32,
        v_bits: u8,
        sink_len: u32,
        window_len: u32,
        rot_sim: Option<RotSimSpec>,
    ) -> Result<Self, RocmlError> {
        let bulk_positions = max_seq.saturating_sub(sink_len);
        let num_blocks_total = bulk_positions.div_ceil(window_len).max(1);
        let bulk_cap = num_blocks_total * window_len;

        let sink_buf_len = (n_kv_heads * sink_len * head_dim) as usize;
        let window_buf_len = (n_kv_heads * window_len * head_dim) as usize;
        let bulk_k_len = (n_kv_heads * bulk_cap * head_dim) as usize;
        let bulk_k_scale_len = (n_kv_heads * num_blocks_total * head_dim) as usize;
        let bulk_v_len = if v_bits == 8 {
            (n_kv_heads * bulk_cap * head_dim) as usize
        } else {
            (n_kv_heads * bulk_cap * (head_dim / 2)) as usize
        };
        let bulk_v_scale_len = (n_kv_heads * bulk_cap) as usize;

        Ok(Self {
            sink_k: DeviceBuffer::new(sink_buf_len)?,
            sink_v: DeviceBuffer::new(sink_buf_len)?,
            window_k: DeviceBuffer::new(window_buf_len)?,
            window_v: DeviceBuffer::new(window_buf_len)?,
            bulk_k_codes: DeviceBuffer::new(bulk_k_len)?,
            bulk_k_scales: DeviceBuffer::new(bulk_k_scale_len)?,
            bulk_v_codes: DeviceBuffer::new(bulk_v_len)?,
            bulk_v_scales: DeviceBuffer::new(bulk_v_scale_len)?,
            layout: MixedLayout::with_lens(sink_len, window_len),
            v_bits,
            n_kv_heads,
            head_dim,
            sink_len,
            window_len,
            bulk_cap,
            num_blocks_total,
            rot_sim,
        })
    }

    pub fn v_bits(&self) -> u8 {
        self.v_bits
    }

    /// Rewinds eviction bookkeeping to its initial post-sink state for a
    /// fresh sequence — see `HybridCache::reset`'s doc comment for why this
    /// (unlike a dense `AttnPlane`) can't just be skipped: `window_base` is
    /// position-independent persistent state, not something a future
    /// position's own bounds-checking would naturally supersede. No device
    /// buffer needs zeroing: once `window_base` is back to `SINK_LEN`, the
    /// bulk region reads as having zero evicted blocks regardless of
    /// whatever bytes are still physically sitting in it.
    pub fn reset(&mut self) {
        self.layout = MixedLayout::with_lens(self.sink_len, self.window_len);
    }

    /// Appends this decode step's `[n_kv_heads, head_dim]` k/v vectors at
    /// `pos`: writes straight into the sink for `pos < SINK_LEN`, else into
    /// the recent window — quantize-evicting the *whole* window in one
    /// batch first if it was already full (see `MixedLayout::prepare_append`'s
    /// doc comment for why this is a whole-block, not per-position, launch).
    pub fn append(
        &mut self,
        kernels: &Kernels,
        mixed: &MixedKernels,
        pos: u32,
        k_src: &DeviceBuffer<f32>,
        v_src: &DeviceBuffer<f32>,
    ) -> Result<(), RocmlError> {
        let (head_dim, n_kv_heads) = (self.head_dim, self.n_kv_heads);
        if pos < self.sink_len {
            for h in 0..n_kv_heads as usize {
                let dst = h * self.sink_len as usize * head_dim as usize
                    + pos as usize * head_dim as usize;
                let src = h * head_dim as usize;
                kernels.cast_f32_f16(offset(k_src, src), offset(&self.sink_k, dst), head_dim)?;
                kernels.cast_f32_f16(offset(v_src, src), offset(&self.sink_v, dst), head_dim)?;
            }
            return Ok(());
        }

        let (slot, evicted) = self.layout.prepare_append(pos);
        if let Some(block) = evicted {
            self.apply_rot_sim()?;
            mixed.quantize_evict_k(
                offset(&self.window_k, 0),
                offset(&self.bulk_k_codes, 0),
                offset(&self.bulk_k_scales, 0),
                n_kv_heads,
                self.window_len,
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
                self.window_len,
                head_dim,
                self.bulk_cap,
                block,
            )?;
        }

        for h in 0..n_kv_heads as usize {
            let dst = h * self.window_len as usize * head_dim as usize
                + slot as usize * head_dim as usize;
            let src = h * head_dim as usize;
            kernels.cast_f32_f16(offset(k_src, src), offset(&self.window_k, dst), head_dim)?;
            kernels.cast_f32_f16(offset(v_src, src), offset(&self.window_v, dst), head_dim)?;
        }
        Ok(())
    }

    pub fn ptrs(&self) -> MixedPtrs {
        MixedPtrs {
            sink_k: offset(&self.sink_k, 0),
            sink_v: offset(&self.sink_v, 0),
            window_k: offset(&self.window_k, 0),
            window_v: offset(&self.window_v, 0),
            bulk_k_codes: offset(&self.bulk_k_codes, 0),
            bulk_k_scales: offset(&self.bulk_k_scales, 0),
            bulk_v_codes: offset(&self.bulk_v_codes, 0),
            bulk_v_scales: offset(&self.bulk_v_scales, 0),
            sink_len: self.sink_len,
            window_len: self.window_len,
            window_base: self.layout.window_base(),
            bulk_cap: self.bulk_cap,
            num_blocks_total: self.num_blocks_total,
            v_bits: self.v_bits,
        }
    }
}
