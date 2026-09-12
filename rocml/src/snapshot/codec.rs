//! Binary (de)serialization for [`SnapshotData`], used only by the disk
//! tier (the RAM tier keeps native Rust values in memory — no encoding
//! needed there). Hand-rolled rather than pulling in `bincode`/`serde` for
//! this (no new dependency, per issue #1's hard rule): the shape is simple
//! and fixed, so a small `Writer`/`Reader` pair is a handful of lines.
//!
//! Every read is bounds-checked and returns [`DecodeError`] rather than
//! panicking or indexing out of range — a truncated or corrupted file must
//! degrade to "unreadable", never crash the caller (see `disk`'s module doc
//! for how that turns into a deleted file + a cache miss).

use half::f16;

use super::types::{AttnLayerBytes, GdnLayerBytes, SnapshotData};

pub struct Writer(Vec<u8>);

impl Writer {
    pub fn new() -> Self {
        Self(Vec::new())
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.0
    }

    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn f32_slice(&mut self, v: &[f32]) {
        self.u64(v.len() as u64);
        for &x in v {
            self.0.extend_from_slice(&x.to_le_bytes());
        }
    }
    fn f16_slice(&mut self, v: &[f16]) {
        self.u64(v.len() as u64);
        for &x in v {
            self.0.extend_from_slice(&x.to_bits().to_le_bytes());
        }
    }
    fn i8_slice(&mut self, v: &[i8]) {
        self.u64(v.len() as u64);
        self.0.extend(v.iter().map(|&x| x as u8));
    }
    fn u8_slice(&mut self, v: &[u8]) {
        self.u64(v.len() as u64);
        self.0.extend_from_slice(v);
    }
    fn u32_slice(&mut self, v: &[u32]) {
        self.u64(v.len() as u64);
        for &x in v {
            self.0.extend_from_slice(&x.to_le_bytes());
        }
    }

    fn gdn(&mut self, layer: &Option<GdnLayerBytes>) {
        match layer {
            None => self.u8(0),
            Some(g) => {
                self.u8(1);
                self.f32_slice(&g.conv_state);
                self.f32_slice(&g.state);
            }
        }
    }

    fn attn(&mut self, layer: &Option<AttnLayerBytes>) {
        match layer {
            None => self.u8(0),
            Some(AttnLayerBytes::DenseF16 { k, v }) => {
                self.u8(1);
                self.f16_slice(k);
                self.f16_slice(v);
            }
            Some(AttnLayerBytes::DenseF32 { k, v }) => {
                self.u8(2);
                self.f32_slice(k);
                self.f32_slice(v);
            }
            Some(AttnLayerBytes::Mixed {
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
            }) => {
                self.u8(3);
                self.f16_slice(sink_k);
                self.f16_slice(sink_v);
                self.f16_slice(window_k);
                self.f16_slice(window_v);
                self.i8_slice(bulk_k_codes);
                self.f32_slice(bulk_k_scales);
                self.u8_slice(bulk_v_codes);
                self.f32_slice(bulk_v_scales);
                self.u32(*window_base);
                self.u8(*v_bits);
            }
        }
    }
}

pub fn encode(data: &SnapshotData) -> Vec<u8> {
    let mut w = Writer::new();
    w.u32(data.position);
    w.u32_slice(&data.token_ids);
    w.u64(data.gdn.len() as u64);
    for layer in &data.gdn {
        w.gdn(layer);
    }
    w.u64(data.attn.len() as u64);
    for layer in &data.attn {
        w.attn(layer);
    }
    w.into_bytes()
}

#[derive(Debug)]
pub struct DecodeError;

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], DecodeError> {
        let end = self.pos.checked_add(len).ok_or(DecodeError)?;
        let slice = self.buf.get(self.pos..end).ok_or(DecodeError)?;
        self.pos = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, DecodeError> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, DecodeError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes(b.try_into().map_err(|_| DecodeError)?))
    }
    fn u64(&mut self) -> Result<u64, DecodeError> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes(b.try_into().map_err(|_| DecodeError)?))
    }
    fn len_usize(&mut self) -> Result<usize, DecodeError> {
        usize::try_from(self.u64()?).map_err(|_| DecodeError)
    }

    fn f32_slice(&mut self) -> Result<Vec<f32>, DecodeError> {
        let n = self.len_usize()?;
        let mut out = Vec::with_capacity(n.min(1 << 20));
        for _ in 0..n {
            let b = self.take(4)?;
            out.push(f32::from_le_bytes(b.try_into().map_err(|_| DecodeError)?));
        }
        Ok(out)
    }
    fn f16_slice(&mut self) -> Result<Vec<f16>, DecodeError> {
        let n = self.len_usize()?;
        let mut out = Vec::with_capacity(n.min(1 << 20));
        for _ in 0..n {
            let b = self.take(2)?;
            out.push(f16::from_bits(u16::from_le_bytes(
                b.try_into().map_err(|_| DecodeError)?,
            )));
        }
        Ok(out)
    }
    fn i8_slice(&mut self) -> Result<Vec<i8>, DecodeError> {
        let n = self.len_usize()?;
        Ok(self.take(n)?.iter().map(|&b| b as i8).collect())
    }
    fn u8_slice(&mut self) -> Result<Vec<u8>, DecodeError> {
        let n = self.len_usize()?;
        Ok(self.take(n)?.to_vec())
    }
    fn u32_slice(&mut self) -> Result<Vec<u32>, DecodeError> {
        let n = self.len_usize()?;
        let mut out = Vec::with_capacity(n.min(1 << 20));
        for _ in 0..n {
            out.push(self.u32()?);
        }
        Ok(out)
    }

    fn gdn(&mut self) -> Result<Option<GdnLayerBytes>, DecodeError> {
        match self.u8()? {
            0 => Ok(None),
            1 => Ok(Some(GdnLayerBytes {
                conv_state: self.f32_slice()?,
                state: self.f32_slice()?,
            })),
            _ => Err(DecodeError),
        }
    }

    fn attn(&mut self) -> Result<Option<AttnLayerBytes>, DecodeError> {
        match self.u8()? {
            0 => Ok(None),
            1 => Ok(Some(AttnLayerBytes::DenseF16 {
                k: self.f16_slice()?,
                v: self.f16_slice()?,
            })),
            2 => Ok(Some(AttnLayerBytes::DenseF32 {
                k: self.f32_slice()?,
                v: self.f32_slice()?,
            })),
            3 => Ok(Some(AttnLayerBytes::Mixed {
                sink_k: self.f16_slice()?,
                sink_v: self.f16_slice()?,
                window_k: self.f16_slice()?,
                window_v: self.f16_slice()?,
                bulk_k_codes: self.i8_slice()?,
                bulk_k_scales: self.f32_slice()?,
                bulk_v_codes: self.u8_slice()?,
                bulk_v_scales: self.f32_slice()?,
                window_base: self.u32()?,
                v_bits: self.u8()?,
            })),
            _ => Err(DecodeError),
        }
    }
}

pub fn decode(bytes: &[u8]) -> Result<SnapshotData, DecodeError> {
    let mut r = Reader::new(bytes);
    let position = r.u32()?;
    let token_ids = r.u32_slice()?;
    let gdn_count = r.len_usize()?;
    let mut gdn = Vec::with_capacity(gdn_count.min(1 << 16));
    for _ in 0..gdn_count {
        gdn.push(r.gdn()?);
    }
    let attn_count = r.len_usize()?;
    let mut attn = Vec::with_capacity(attn_count.min(1 << 16));
    for _ in 0..attn_count {
        attn.push(r.attn()?);
    }
    Ok(SnapshotData {
        position,
        token_ids,
        gdn,
        attn,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> SnapshotData {
        SnapshotData {
            position: 3,
            token_ids: vec![10, 20, 30],
            gdn: vec![
                None,
                Some(GdnLayerBytes {
                    conv_state: vec![1.0, 2.0, 3.0],
                    state: vec![4.0, 5.0],
                }),
            ],
            attn: vec![
                Some(AttnLayerBytes::DenseF16 {
                    k: vec![f16::from_f32(1.5), f16::from_f32(-2.5)],
                    v: vec![f16::from_f32(0.0)],
                }),
                Some(AttnLayerBytes::Mixed {
                    sink_k: vec![f16::from_f32(1.0)],
                    sink_v: vec![f16::from_f32(2.0)],
                    window_k: vec![f16::from_f32(3.0)],
                    window_v: vec![f16::from_f32(4.0)],
                    bulk_k_codes: vec![-1, 2, -3],
                    bulk_k_scales: vec![0.1, 0.2],
                    bulk_v_codes: vec![250, 5],
                    bulk_v_scales: vec![0.3],
                    window_base: 160,
                    v_bits: 8,
                }),
                None,
            ],
        }
    }

    #[test]
    fn roundtrips_exactly() {
        let original = sample();
        let bytes = encode(&original);
        let decoded = decode(&bytes).expect("decode failed");
        assert_eq!(original, decoded);
    }

    #[test]
    fn truncated_bytes_error_instead_of_panicking() {
        let bytes = encode(&sample());
        for cut in [0, 1, 4, bytes.len() / 2, bytes.len() - 1] {
            assert!(decode(&bytes[..cut]).is_err());
        }
    }

    #[test]
    fn invalid_tag_is_an_error() {
        let mut bytes = encode(&sample());
        // Corrupt the first layer-kind tag inside the gdn section — position
        // found by construction of `sample()` (u32 position + token_ids
        // header + gdn count, then the first tag byte).
        let tag_offset = 4 + 8 + 3 * 4 + 8;
        bytes[tag_offset] = 0xFF;
        assert!(decode(&bytes).is_err());
    }
}
