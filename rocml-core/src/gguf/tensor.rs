//! Tensor info table and the validated view handed out for one tensor's
//! raw bytes.
//!
//! GGML/GGUF store dimensions in "ne" order: `ne[0]` is the fastest-varying
//! (innermost/contiguous) dimension, the opposite of the usual row-major
//! "shape[0] is the outermost axis" convention. A 2D weight matrix with
//! `ne = [in_features, out_features]` is stored the same way PyTorch would
//! store a `[out_features, in_features]` tensor — i.e. `ne` is PyTorch's
//! shape reversed. Callers translating to a row-major shape must reverse it.

use super::error::GgufError;
use super::reader::Cursor;
use crate::quant::GgmlDType;

#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfo {
    pub name: String,
    /// Dimensions in GGML "ne" order: `ne[0]` is the innermost/fastest-varying axis.
    pub ne: Vec<u64>,
    pub dtype: GgmlDType,
    /// Byte offset relative to the start of the data section (not the file).
    pub offset: u64,
}

impl TensorInfo {
    pub fn n_elements(&self) -> Option<u64> {
        self.ne.iter().try_fold(1u64, |acc, &d| acc.checked_mul(d))
    }
}

/// A validated, zero-copy view onto one tensor's raw bytes inside the mmap.
pub struct TensorView<'a> {
    name: &'a str,
    ne: &'a [u64],
    dtype: GgmlDType,
    data: &'a [u8],
}

impl<'a> TensorView<'a> {
    pub fn name(&self) -> &'a str {
        self.name
    }

    /// Dimensions in GGML "ne" order (see the module docs): `shape()[0]` is
    /// the innermost/fastest-varying axis, not the outermost one.
    pub fn shape(&self) -> &'a [u64] {
        self.ne
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn data(&self) -> &'a [u8] {
        self.data
    }
}

/// Reads one tensor-info record: name, ne-order dims, ggml dtype id, offset.
pub(crate) fn read_tensor_info(cursor: &mut Cursor<'_>) -> Result<TensorInfo, GgufError> {
    let name = cursor.string("tensor name")?;
    let n_dims = cursor.u32("tensor n_dims")?;
    let mut ne = Vec::new();
    for _ in 0..n_dims {
        ne.push(cursor.u64("tensor dimension")?);
    }
    let dtype_id = cursor.u32("tensor dtype")?;
    let offset = cursor.u64("tensor offset")?;
    Ok(TensorInfo {
        name,
        ne,
        dtype: GgmlDType::from_ggml_id(dtype_id),
        offset,
    })
}

/// Builds the validated `TensorView` for `info`, checking that its byte
/// range lies inside `data` and that its element count is a whole number of
/// quant blocks.
pub(crate) fn validate_tensor<'a>(
    info: &'a TensorInfo,
    data: &'a [u8],
) -> Result<TensorView<'a>, GgufError> {
    let n_elements = info
        .n_elements()
        .ok_or_else(|| GgufError::TensorOutOfBounds {
            name: info.name.clone(),
            offset: info.offset,
            end: u64::MAX,
            data_len: data.len() as u64,
        })?;

    let GgmlDType::Unsupported(dtype_id) = info.dtype else {
        let block_size = info.dtype.block_elements() as u64;
        if !n_elements.is_multiple_of(block_size) {
            return Err(GgufError::BadBlockAlignment {
                name: info.name.clone(),
                n_elements,
                block_size,
                dtype: info.dtype,
            });
        }
        let n_blocks = n_elements / block_size;
        let n_bytes = n_blocks
            .checked_mul(info.dtype.block_bytes() as u64)
            .ok_or_else(|| GgufError::TensorOutOfBounds {
                name: info.name.clone(),
                offset: info.offset,
                end: u64::MAX,
                data_len: data.len() as u64,
            })?;
        let end = info
            .offset
            .checked_add(n_bytes)
            .ok_or_else(|| GgufError::TensorOutOfBounds {
                name: info.name.clone(),
                offset: info.offset,
                end: u64::MAX,
                data_len: data.len() as u64,
            })?;
        if end > data.len() as u64 {
            return Err(GgufError::TensorOutOfBounds {
                name: info.name.clone(),
                offset: info.offset,
                end,
                data_len: data.len() as u64,
            });
        }
        let slice = &data[info.offset as usize..end as usize];
        return Ok(TensorView {
            name: &info.name,
            ne: &info.ne,
            dtype: info.dtype,
            data: slice,
        });
    };
    Err(GgufError::UnsupportedTensorDType {
        name: info.name.clone(),
        dtype_id,
    })
}
