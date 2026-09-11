//! Bounds-checked byte cursor over the mmap'd file. Every primitive read
//! verifies enough bytes remain before touching them, so a truncated or
//! adversarial file produces an `Err`, never an out-of-bounds panic.
//!
//! Slicing a fixed number of bytes off an mmap is zero-copy (no allocation),
//! so a bogus huge length in the file fails fast in `take` rather than
//! spending memory before the bounds check runs.

use super::error::GgufError;

pub(crate) struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    pub(crate) fn position(&self) -> usize {
        self.pos
    }

    pub(crate) fn magic(&mut self) -> Result<[u8; 4], GgufError> {
        let b = self.take(4, "magic")?;
        Ok([b[0], b[1], b[2], b[3]])
    }

    fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    fn take(&mut self, n: usize, context: &'static str) -> Result<&'a [u8], GgufError> {
        if self.remaining() < n {
            return Err(GgufError::UnexpectedEof { context });
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    pub(crate) fn u8(&mut self, context: &'static str) -> Result<u8, GgufError> {
        let b = self.take(1, context)?;
        Ok(b[0])
    }

    pub(crate) fn bool(&mut self, context: &'static str) -> Result<bool, GgufError> {
        Ok(self.u8(context)? != 0)
    }

    pub(crate) fn i8(&mut self, context: &'static str) -> Result<i8, GgufError> {
        Ok(self.u8(context)? as i8)
    }

    pub(crate) fn u16(&mut self, context: &'static str) -> Result<u16, GgufError> {
        let b = self.take(2, context)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    pub(crate) fn i16(&mut self, context: &'static str) -> Result<i16, GgufError> {
        Ok(self.u16(context)? as i16)
    }

    pub(crate) fn u32(&mut self, context: &'static str) -> Result<u32, GgufError> {
        let b = self.take(4, context)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub(crate) fn i32(&mut self, context: &'static str) -> Result<i32, GgufError> {
        Ok(self.u32(context)? as i32)
    }

    pub(crate) fn f32(&mut self, context: &'static str) -> Result<f32, GgufError> {
        Ok(f32::from_bits(self.u32(context)?))
    }

    pub(crate) fn u64(&mut self, context: &'static str) -> Result<u64, GgufError> {
        let b = self.take(8, context)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    pub(crate) fn i64(&mut self, context: &'static str) -> Result<i64, GgufError> {
        Ok(self.u64(context)? as i64)
    }

    pub(crate) fn f64(&mut self, context: &'static str) -> Result<f64, GgufError> {
        Ok(f64::from_bits(self.u64(context)?))
    }

    /// GGUF strings are `u64` length prefix + raw UTF-8 bytes (no NUL terminator).
    pub(crate) fn string(&mut self, context: &'static str) -> Result<String, GgufError> {
        let len = self.u64(context)?;
        let len = usize::try_from(len).map_err(|_| GgufError::SizeOverflow { context })?;
        let bytes = self.take(len, context)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| GgufError::InvalidUtf8 { context })
    }
}
