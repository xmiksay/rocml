//! GGUF metadata value model: the tagged union stored per key, plus its
//! wire-format parser. Typed accessors live on `GgufFile` in `mod.rs` and
//! delegate to the `as_*` widening helpers here.

use super::error::GgufError;
use super::reader::Cursor;

/// Maximum nesting depth for `ARRAY` values. GGUF metadata never nests more
/// than one level deep in practice (an array of strings, an array of ints);
/// this only exists to stop a hostile file from recursing until the stack
/// overflows.
const MAX_ARRAY_DEPTH: u32 = 8;

#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(Vec<MetadataValue>),
}

impl MetadataValue {
    pub(crate) fn type_name(&self) -> &'static str {
        match self {
            Self::U8(_) => "u8",
            Self::I8(_) => "i8",
            Self::U16(_) => "u16",
            Self::I16(_) => "i16",
            Self::U32(_) => "u32",
            Self::I32(_) => "i32",
            Self::F32(_) => "f32",
            Self::Bool(_) => "bool",
            Self::String(_) => "string",
            Self::U64(_) => "u64",
            Self::I64(_) => "i64",
            Self::F64(_) => "f64",
            Self::Array(_) => "array",
        }
    }

    /// Widens any unsigned (and non-negative signed) integer variant to `u64`.
    pub(crate) fn as_u64(&self) -> Option<u64> {
        match *self {
            Self::U8(v) => Some(v as u64),
            Self::U16(v) => Some(v as u64),
            Self::U32(v) => Some(v as u64),
            Self::U64(v) => Some(v),
            Self::I8(v) if v >= 0 => Some(v as u64),
            Self::I16(v) if v >= 0 => Some(v as u64),
            Self::I32(v) if v >= 0 => Some(v as u64),
            Self::I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }

    /// Widens any signed or unsigned integer variant that fits into `i64`.
    pub(crate) fn as_i64(&self) -> Option<i64> {
        match *self {
            Self::I8(v) => Some(v as i64),
            Self::I16(v) => Some(v as i64),
            Self::I32(v) => Some(v as i64),
            Self::I64(v) => Some(v),
            Self::U8(v) => Some(v as i64),
            Self::U16(v) => Some(v as i64),
            Self::U32(v) => Some(v as i64),
            Self::U64(v) => i64::try_from(v).ok(),
            _ => None,
        }
    }

    /// Widens `F32`/`F64` to `f64`. Integers are intentionally not coerced:
    /// callers asking for a float want a float field, not a reinterpreted
    /// count.
    pub(crate) fn as_f64(&self) -> Option<f64> {
        match *self {
            Self::F32(v) => Some(v as f64),
            Self::F64(v) => Some(v),
            _ => None,
        }
    }

    pub(crate) fn as_bool(&self) -> Option<bool> {
        match *self {
            Self::Bool(v) => Some(v),
            _ => None,
        }
    }

    pub(crate) fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(v) => Some(v.as_str()),
            _ => None,
        }
    }

    pub(crate) fn as_array(&self) -> Option<&[MetadataValue]> {
        match self {
            Self::Array(items) => Some(items),
            _ => None,
        }
    }
}

pub(crate) fn read_value(
    cursor: &mut Cursor<'_>,
    value_type: u32,
    depth: u32,
) -> Result<MetadataValue, GgufError> {
    const CTX: &str = "metadata value";
    match value_type {
        0 => Ok(MetadataValue::U8(cursor.u8(CTX)?)),
        1 => Ok(MetadataValue::I8(cursor.i8(CTX)?)),
        2 => Ok(MetadataValue::U16(cursor.u16(CTX)?)),
        3 => Ok(MetadataValue::I16(cursor.i16(CTX)?)),
        4 => Ok(MetadataValue::U32(cursor.u32(CTX)?)),
        5 => Ok(MetadataValue::I32(cursor.i32(CTX)?)),
        6 => Ok(MetadataValue::F32(cursor.f32(CTX)?)),
        7 => Ok(MetadataValue::Bool(cursor.bool(CTX)?)),
        8 => Ok(MetadataValue::String(cursor.string(CTX)?)),
        9 => read_array(cursor, depth),
        10 => Ok(MetadataValue::U64(cursor.u64(CTX)?)),
        11 => Ok(MetadataValue::I64(cursor.i64(CTX)?)),
        12 => Ok(MetadataValue::F64(cursor.f64(CTX)?)),
        other => Err(GgufError::UnknownValueType(other)),
    }
}

fn read_array(cursor: &mut Cursor<'_>, depth: u32) -> Result<MetadataValue, GgufError> {
    if depth >= MAX_ARRAY_DEPTH {
        return Err(GgufError::ArrayTooDeep);
    }
    const CTX: &str = "array header";
    let elem_type = cursor.u32(CTX)?;
    let count = cursor.u64(CTX)?;
    let count = usize::try_from(count).map_err(|_| GgufError::SizeOverflow { context: CTX })?;
    // No `Vec::with_capacity(count)`: `count` comes straight from the file,
    // so a hostile value must not let us pre-allocate before validating
    // there are actually `count` elements worth of bytes behind it. Each
    // push instead pays for itself as `read_value` consumes real bytes,
    // and a truncated file fails fast via `UnexpectedEof`.
    let mut items = Vec::new();
    for _ in 0..count {
        items.push(read_value(cursor, elem_type, depth + 1)?);
    }
    Ok(MetadataValue::Array(items))
}
