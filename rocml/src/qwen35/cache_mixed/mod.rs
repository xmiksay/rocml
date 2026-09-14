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
//! a real window eviction.

mod chunk;
mod rot_sim;

use half::f16;
use rocml_hip::DeviceBuffer;

use super::forward::kernels_mixed::MixedKernels;
use crate::error::RocmlError;
use crate::forward::kernels::{offset, DevPtr, Kernels};
use crate::kv_quant::rotational::RotSimSpec;
use crate::kv_quant::MixedLayout;
use crate::snapshot::AttnLayerBytes;

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

    /// Captures this layer's whole state for the snapshot layer (issue #1).
    ///
    /// Sink/window are captured whole (they're fixed-size, `SINK_LEN`/
    /// `WINDOW_LEN` positions regardless of `ctx`). The bulk region is
    /// captured whole too — a deliberate v1 simplification, unlike
    /// `AttnPlane::capture`'s filled-prefix-only slicing: `bulk_k_codes`/
    /// `bulk_v_codes`'s per-head/per-block physical layout is an
    /// implementation detail of `quantize_evict_k`/`_v`'s HIP kernels (see
    /// `kv_quant::quant_math`'s module doc for the one evicted block's
    /// layout, `[n_kv_heads, WINDOW_LEN, head_dim]` — but not how blocks are
    /// placed relative to each other in the larger buffer), so slicing out
    /// only `evicted_blocks()` worth would require duplicating that
    /// assumption here at real risk of a silent mismatch. Capturing the
    /// whole `bulk_cap`-sized buffer costs snapshot size proportional to
    /// configured `ctx` rather than actual position for mixed layers only —
    /// correct either way (unfilled blocks are simply never read on
    /// restore), and fine at this project's context scales; a follow-up can
    /// slice precisely once that layout is exposed as a documented contract.
    pub fn capture(&self) -> Result<AttnLayerBytes, RocmlError> {
        let mut sink_k = vec![f16::from_f32(0.0); self.sink_k.len()];
        let mut sink_v = vec![f16::from_f32(0.0); self.sink_v.len()];
        let mut window_k = vec![f16::from_f32(0.0); self.window_k.len()];
        let mut window_v = vec![f16::from_f32(0.0); self.window_v.len()];
        self.sink_k.copy_to_host(&mut sink_k)?;
        self.sink_v.copy_to_host(&mut sink_v)?;
        self.window_k.copy_to_host(&mut window_k)?;
        self.window_v.copy_to_host(&mut window_v)?;

        let mut bulk_k_codes = vec![0i8; self.bulk_k_codes.len()];
        let mut bulk_k_scales = vec![0.0f32; self.bulk_k_scales.len()];
        let mut bulk_v_codes = vec![0u8; self.bulk_v_codes.len()];
        let mut bulk_v_scales = vec![0.0f32; self.bulk_v_scales.len()];
        self.bulk_k_codes.copy_to_host(&mut bulk_k_codes)?;
        self.bulk_k_scales.copy_to_host(&mut bulk_k_scales)?;
        self.bulk_v_codes.copy_to_host(&mut bulk_v_codes)?;
        self.bulk_v_scales.copy_to_host(&mut bulk_v_scales)?;

        Ok(AttnLayerBytes::Mixed {
            sink_k,
            sink_v,
            window_k,
            window_v,
            bulk_k_codes,
            bulk_k_scales,
            bulk_v_codes,
            bulk_v_scales,
            window_base: self.layout.window_base(),
            v_bits: self.v_bits,
        })
    }

    /// Writes a captured state back — every buffer must match this plane's
    /// own allocated size exactly (guaranteed when `KvConfigStamp`,
    /// including `ctx`, matched at lookup time, since that determines
    /// `bulk_cap`/`num_blocks_total` deterministically).
    pub fn restore(&mut self, bytes: &AttnLayerBytes) -> Result<(), RocmlError> {
        let AttnLayerBytes::Mixed {
            sink_k,
            sink_v,
            window_k,
            window_v,
            bulk_k_codes,
            bulk_k_scales,
            bulk_v_codes,
            bulk_v_scales,
            window_base,
            v_bits,
        } = bytes
        else {
            return Err(RocmlError::Config(
                "mixed attn plane restore: snapshot isn't a Mixed layer (internal bug — \
                 KvConfigStamp should have gated this)"
                    .to_string(),
            ));
        };
        if *v_bits != self.v_bits {
            return Err(RocmlError::Config(format!(
                "mixed attn plane restore: v_bits mismatch ({v_bits} vs {})",
                self.v_bits
            )));
        }
        check_exact_len(sink_k.len(), self.sink_k.len(), "sink_k")?;
        check_exact_len(sink_v.len(), self.sink_v.len(), "sink_v")?;
        check_exact_len(window_k.len(), self.window_k.len(), "window_k")?;
        check_exact_len(window_v.len(), self.window_v.len(), "window_v")?;
        check_exact_len(bulk_k_codes.len(), self.bulk_k_codes.len(), "bulk_k_codes")?;
        check_exact_len(
            bulk_k_scales.len(),
            self.bulk_k_scales.len(),
            "bulk_k_scales",
        )?;
        check_exact_len(bulk_v_codes.len(), self.bulk_v_codes.len(), "bulk_v_codes")?;
        check_exact_len(
            bulk_v_scales.len(),
            self.bulk_v_scales.len(),
            "bulk_v_scales",
        )?;

        self.sink_k.copy_from_host(sink_k)?;
        self.sink_v.copy_from_host(sink_v)?;
        self.window_k.copy_from_host(window_k)?;
        self.window_v.copy_from_host(window_v)?;
        self.bulk_k_codes.copy_from_host(bulk_k_codes)?;
        self.bulk_k_scales.copy_from_host(bulk_k_scales)?;
        self.bulk_v_codes.copy_from_host(bulk_v_codes)?;
        self.bulk_v_scales.copy_from_host(bulk_v_scales)?;
        self.layout =
            MixedLayout::from_window_base_with_lens(*window_base, self.sink_len, self.window_len);
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

fn check_exact_len(actual: usize, expected: usize, field: &str) -> Result<(), RocmlError> {
    if actual != expected {
        return Err(RocmlError::Config(format!(
            "mixed attn plane restore: {field} has {actual} elements, expected {expected}"
        )));
    }
    Ok(())
}
