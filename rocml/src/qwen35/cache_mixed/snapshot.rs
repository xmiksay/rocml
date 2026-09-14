//! Capture/restore for [`MixedAttnPlane`] (issue #1) — split out of `mod.rs`
//! purely for the 400-line file cap; being a child module is what lets these
//! `impl` blocks reach `MixedAttnPlane`'s private fields directly, the same
//! way `chunk`/`rot_sim` already do.

use half::f16;
use rocml_hip::DeviceBuffer;

use super::MixedAttnPlane;
use crate::error::RocmlError;
use crate::kv_quant::MixedLayout;
use crate::snapshot::AttnLayerBytes;

impl MixedAttnPlane {
    /// Captures this layer's whole state for the snapshot layer (issue #1).
    ///
    /// Sink/window are captured whole (they're fixed-size, `SINK_LEN`/
    /// `WINDOW_LEN` positions regardless of `ctx`). The bulk region is
    /// sliced down to only its *filled* prefix — `evicted_blocks()` whole
    /// blocks per head — mirroring `AttnPlane::capture`'s filled-prefix-only
    /// approach for the dense path. This relies on `kv_quant.hip`'s own
    /// documented bulk layout (`quantize_evict_k_f16_to_q8`/
    /// `quantize_evict_v_f16_to_q{8,4}`'s module doc: `[n_kv_heads, bulk_cap,
    /// head_dim]` for K/V codes, block `b`'s positions at a per-head offset
    /// of `b * window_len`; `[n_kv_heads, bulk_cap]`/`[n_kv_heads,
    /// num_blocks_total, head_dim]` for V/K scales) — every unfilled block
    /// lives strictly *after* the filled prefix within each head's own span,
    /// never interleaved, so a per-head partial copy is exact, not an
    /// approximation.
    ///
    /// Capturing the whole `bulk_cap`-sized buffer regardless of actual
    /// position (the previous behavior) made every mixed-KV snapshot's size
    /// proportional to the server's configured `ctx` rather than the
    /// conversation's real length — for a large-`ctx` quantized-KV
    /// deployment this made even a handful-of-tokens snapshot cost hundreds
    /// of MB, blowing through the RAM/disk budget (or a single snapshot
    /// exceeding it outright) and evicting a turn's own render-stable
    /// snapshot before the next turn could ever look it up (issue #12).
    pub fn capture(&self) -> Result<AttnLayerBytes, RocmlError> {
        let mut sink_k = vec![f16::from_f32(0.0); self.sink_k.len()];
        let mut sink_v = vec![f16::from_f32(0.0); self.sink_v.len()];
        let mut window_k = vec![f16::from_f32(0.0); self.window_k.len()];
        let mut window_v = vec![f16::from_f32(0.0); self.window_v.len()];
        self.sink_k.copy_to_host(&mut sink_k)?;
        self.sink_v.copy_to_host(&mut sink_v)?;
        self.window_k.copy_to_host(&mut window_k)?;
        self.window_v.copy_to_host(&mut window_v)?;

        let heads = self.n_kv_heads as usize;
        let head_dim = self.head_dim as usize;
        let evicted_blocks = self.layout.evicted_blocks() as usize;
        let evicted_len = evicted_blocks * self.window_len as usize;
        let v_row = self.v_row_width();

        let mut bulk_k_codes = vec![0i8; heads * evicted_len * head_dim];
        let mut bulk_k_scales = vec![0.0f32; heads * evicted_blocks * head_dim];
        let mut bulk_v_codes = vec![0u8; heads * evicted_len * v_row];
        let mut bulk_v_scales = vec![0.0f32; heads * evicted_len];
        for h in 0..heads {
            copy_head_prefix_to_host(
                &self.bulk_k_codes,
                &mut bulk_k_codes,
                h,
                self.bulk_cap as usize * head_dim,
                evicted_len * head_dim,
            )?;
            copy_head_prefix_to_host(
                &self.bulk_k_scales,
                &mut bulk_k_scales,
                h,
                self.num_blocks_total as usize * head_dim,
                evicted_blocks * head_dim,
            )?;
            copy_head_prefix_to_host(
                &self.bulk_v_codes,
                &mut bulk_v_codes,
                h,
                self.bulk_cap as usize * v_row,
                evicted_len * v_row,
            )?;
            copy_head_prefix_to_host(
                &self.bulk_v_scales,
                &mut bulk_v_scales,
                h,
                self.bulk_cap as usize,
                evicted_len,
            )?;
        }

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

    /// V codes' per-position row width: `head_dim` elements at 8-bit
    /// (`Vec<u8>` reinterpreting the kernel's `signed char` codes), or
    /// `head_dim / 2` packed-nibble bytes at 4-bit — see
    /// `quantize_evict_v_f16_to_q4`'s packing.
    fn v_row_width(&self) -> usize {
        if self.v_bits == 8 {
            self.head_dim as usize
        } else {
            self.head_dim as usize / 2
        }
    }

    /// Writes a captured state back. Sink/window must match this plane's own
    /// fixed allocated size exactly; the bulk fields are only ever as long
    /// as `window_base` (restored first) says they should be — see
    /// [`Self::capture`]'s doc comment for why they're no longer the full
    /// `bulk_cap`-sized buffers.
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

        let restored_layout =
            MixedLayout::from_window_base_with_lens(*window_base, self.sink_len, self.window_len);
        let heads = self.n_kv_heads as usize;
        let head_dim = self.head_dim as usize;
        let evicted_blocks = restored_layout.evicted_blocks() as usize;
        let evicted_len = evicted_blocks * self.window_len as usize;
        let v_row = self.v_row_width();

        check_exact_len(
            bulk_k_codes.len(),
            heads * evicted_len * head_dim,
            "bulk_k_codes",
        )?;
        check_exact_len(
            bulk_k_scales.len(),
            heads * evicted_blocks * head_dim,
            "bulk_k_scales",
        )?;
        check_exact_len(
            bulk_v_codes.len(),
            heads * evicted_len * v_row,
            "bulk_v_codes",
        )?;
        check_exact_len(bulk_v_scales.len(), heads * evicted_len, "bulk_v_scales")?;

        self.sink_k.copy_from_host(sink_k)?;
        self.sink_v.copy_from_host(sink_v)?;
        self.window_k.copy_from_host(window_k)?;
        self.window_v.copy_from_host(window_v)?;
        for h in 0..heads {
            copy_head_prefix_from_host(
                &mut self.bulk_k_codes,
                bulk_k_codes,
                h,
                self.bulk_cap as usize * head_dim,
                evicted_len * head_dim,
            )?;
            copy_head_prefix_from_host(
                &mut self.bulk_k_scales,
                bulk_k_scales,
                h,
                self.num_blocks_total as usize * head_dim,
                evicted_blocks * head_dim,
            )?;
            copy_head_prefix_from_host(
                &mut self.bulk_v_codes,
                bulk_v_codes,
                h,
                self.bulk_cap as usize * v_row,
                evicted_len * v_row,
            )?;
            copy_head_prefix_from_host(
                &mut self.bulk_v_scales,
                bulk_v_scales,
                h,
                self.bulk_cap as usize,
                evicted_len,
            )?;
        }
        self.layout = restored_layout;
        Ok(())
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

/// D2H-copies one head's `prefix_len`-element filled prefix out of a
/// per-head-`head_stride`-strided device buffer, into the matching
/// `prefix_len`-sized slot of a tightly-packed (no per-head gap) host
/// buffer — the bulk region's "only capture what's actually evicted"
/// slicing both [`MixedAttnPlane::capture`] and its restore counterpart
/// need, for every dtype the bulk fields use (`i8`/`u8`/`f32`).
fn copy_head_prefix_to_host<T: Copy>(
    src: &DeviceBuffer<T>,
    dst: &mut [T],
    head: usize,
    head_stride: usize,
    prefix_len: usize,
) -> Result<(), RocmlError> {
    let src_offset = head * head_stride;
    let dst_offset = head * prefix_len;
    src.copy_range_to_host(src_offset, &mut dst[dst_offset..dst_offset + prefix_len])?;
    Ok(())
}

/// H2D counterpart of [`copy_head_prefix_to_host`]: writes one head's
/// `prefix_len`-element prefix from a tightly-packed host buffer back into
/// its per-head-`head_stride`-strided slot of the device buffer, leaving
/// whatever lies past `prefix_len` in that head's span untouched (never
/// read again unless a future eviction overwrites it first, exactly like
/// `AttnPlane::restore`'s filled-prefix-only write).
fn copy_head_prefix_from_host<T: Copy>(
    dst: &mut DeviceBuffer<T>,
    src: &[T],
    head: usize,
    head_stride: usize,
    prefix_len: usize,
) -> Result<(), RocmlError> {
    let dst_offset = head * head_stride;
    let src_offset = head * prefix_len;
    dst.copy_range_from_host(dst_offset, &src[src_offset..src_offset + prefix_len])?;
    Ok(())
}
