//! Memory-mapped GGUF (v2/v3) parser: header, metadata key-value table,
//! tensor info table, and validated tensor byte views.
//!
//! Dimensions are reported in GGML's "ne" order everywhere in this module
//! (see `tensor.rs`): `ne[0]` is the fastest-varying/innermost dimension,
//! the reverse of a typical PyTorch row-major shape.

mod error;
mod reader;
mod tensor;
mod value;

pub use error::GgufError;
pub use tensor::{TensorInfo, TensorView};
pub use value::MetadataValue;

use std::collections::{HashMap, HashSet};
use std::path::Path;

use memmap2::Mmap;
use reader::Cursor;

const DEFAULT_ALIGNMENT: u64 = 32;

pub struct GgufFile {
    mmap: Mmap,
    version: u32,
    metadata: HashMap<String, MetadataValue>,
    tensor_infos: Vec<TensorInfo>,
    /// Byte offset within the mmap where the tensor data section starts
    /// (header end, rounded up to `general.alignment`).
    data_offset: usize,
}

impl GgufFile {
    /// Memory-maps `path` and parses its GGUF header, metadata and tensor
    /// info table. The (potentially huge) tensor data itself is never
    /// copied — `tensor()` hands out validated slices straight into the
    /// mapping.
    ///
    /// # Safety-adjacent note
    /// Memory-mapping a file is inherently exposed to the file being
    /// truncated or modified concurrently on disk, which is undefined
    /// behavior for any mmap-based reader, not something this crate can
    /// fully guard against; treat GGUF files like any other memory-mapped
    /// input and don't mutate them while a `GgufFile` is open.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, GgufError> {
        let file = std::fs::File::open(path)?;
        // SAFETY: see the doc comment above — the file is assumed stable
        // for the lifetime of the mapping, which is the standard mmap caveat.
        let mmap = unsafe { Mmap::map(&file)? };

        let mut cursor = Cursor::new(&mmap[..]);
        if cursor.magic()? != *b"GGUF" {
            return Err(GgufError::InvalidMagic);
        }
        let version = cursor.u32("version")?;
        if version != 2 && version != 3 {
            return Err(GgufError::UnsupportedVersion(version));
        }
        let tensor_count = cursor.u64("tensor_count")?;
        let tensor_count = usize::try_from(tensor_count).map_err(|_| GgufError::SizeOverflow {
            context: "tensor_count",
        })?;
        let kv_count = cursor.u64("metadata_kv_count")?;

        let mut metadata = HashMap::new();
        for _ in 0..kv_count {
            let key = cursor.string("metadata key")?;
            let value_type = cursor.u32("metadata value type")?;
            let value = value::read_value(&mut cursor, value_type, 0)?;
            if metadata.insert(key.clone(), value).is_some() {
                return Err(GgufError::DuplicateKey(key));
            }
        }

        let mut tensor_infos = Vec::new();
        let mut seen_names = HashSet::new();
        for _ in 0..tensor_count {
            let info = tensor::read_tensor_info(&mut cursor)?;
            if !seen_names.insert(info.name.clone()) {
                return Err(GgufError::DuplicateTensor(info.name));
            }
            tensor_infos.push(info);
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(MetadataValue::as_u64)
            .filter(|&a| a > 0)
            .unwrap_or(DEFAULT_ALIGNMENT) as usize;
        let header_end = cursor.position();
        let data_offset = header_end.div_ceil(alignment) * alignment;
        if data_offset > mmap.len() {
            return Err(GgufError::UnexpectedEof {
                context: "data section start (past end of file)",
            });
        }

        Ok(Self {
            mmap,
            version,
            metadata,
            tensor_infos,
            data_offset,
        })
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensor_infos
    }

    /// Looks up `name` and returns a validated view of its raw bytes: the
    /// byte range is checked to fit inside the mmap and its element count
    /// checked to be a whole number of quant blocks.
    pub fn tensor(&self, name: &str) -> Result<TensorView<'_>, GgufError> {
        let info = self
            .tensor_infos
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| GgufError::TensorNotFound(name.to_string()))?;
        let data = &self.mmap[self.data_offset..];
        tensor::validate_tensor(info, data)
    }

    fn metadata_value(&self, key: &str) -> Result<&MetadataValue, GgufError> {
        self.metadata
            .get(key)
            .ok_or_else(|| GgufError::MissingKey(key.to_string()))
    }

    pub fn get_str(&self, key: &str) -> Result<&str, GgufError> {
        let v = self.metadata_value(key)?;
        v.as_str().ok_or_else(|| GgufError::WrongType {
            key: key.to_string(),
            expected: "string",
            found: v.type_name(),
        })
    }

    pub fn get_bool(&self, key: &str) -> Result<bool, GgufError> {
        let v = self.metadata_value(key)?;
        v.as_bool().ok_or_else(|| GgufError::WrongType {
            key: key.to_string(),
            expected: "bool",
            found: v.type_name(),
        })
    }

    pub fn get_f32(&self, key: &str) -> Result<f32, GgufError> {
        let v = self.metadata_value(key)?;
        v.as_f64()
            .map(|f| f as f32)
            .ok_or_else(|| GgufError::WrongType {
                key: key.to_string(),
                expected: "f32",
                found: v.type_name(),
            })
    }

    pub fn get_f64(&self, key: &str) -> Result<f64, GgufError> {
        let v = self.metadata_value(key)?;
        v.as_f64().ok_or_else(|| GgufError::WrongType {
            key: key.to_string(),
            expected: "f64",
            found: v.type_name(),
        })
    }
}

/// Declares `get_<name>(key) -> Result<$ty, GgufError>` backed by the
/// widening `as_u64`/`as_i64` helpers on `MetadataValue`.
macro_rules! int_getter {
    ($name:ident, $ty:ty, $as_fn:ident, $label:literal) => {
        impl GgufFile {
            pub fn $name(&self, key: &str) -> Result<$ty, GgufError> {
                let v = self.metadata_value(key)?;
                v.$as_fn()
                    .and_then(|n| <$ty>::try_from(n).ok())
                    .ok_or_else(|| GgufError::WrongType {
                        key: key.to_string(),
                        expected: $label,
                        found: v.type_name(),
                    })
            }
        }
    };
}

int_getter!(get_u8, u8, as_u64, "u8");
int_getter!(get_u16, u16, as_u64, "u16");
int_getter!(get_u32, u32, as_u64, "u32");
int_getter!(get_u64, u64, as_u64, "u64");
int_getter!(get_i8, i8, as_i64, "i8");
int_getter!(get_i16, i16, as_i64, "i16");
int_getter!(get_i32, i32, as_i64, "i32");
int_getter!(get_i64, i64, as_i64, "i64");

/// Declares `get_<name>_arr(key) -> Result<Vec<$ty>, GgufError>` for an
/// array of scalars extracted the same way as the matching `int_getter!`.
macro_rules! int_arr_getter {
    ($name:ident, $ty:ty, $as_fn:ident, $label:literal) => {
        impl GgufFile {
            pub fn $name(&self, key: &str) -> Result<Vec<$ty>, GgufError> {
                let v = self.metadata_value(key)?;
                let items = v.as_array().ok_or_else(|| GgufError::WrongType {
                    key: key.to_string(),
                    expected: concat!("array of ", $label),
                    found: v.type_name(),
                })?;
                items
                    .iter()
                    .map(|item| {
                        item.$as_fn()
                            .and_then(|n| <$ty>::try_from(n).ok())
                            .ok_or_else(|| GgufError::WrongType {
                                key: key.to_string(),
                                expected: concat!("array of ", $label),
                                found: item.type_name(),
                            })
                    })
                    .collect()
            }
        }
    };
}

int_arr_getter!(get_i32_arr, i32, as_i64, "i32");
int_arr_getter!(get_u32_arr, u32, as_u64, "u32");

impl GgufFile {
    pub fn get_str_arr(&self, key: &str) -> Result<Vec<String>, GgufError> {
        let v = self.metadata_value(key)?;
        let items = v.as_array().ok_or_else(|| GgufError::WrongType {
            key: key.to_string(),
            expected: "array of string",
            found: v.type_name(),
        })?;
        items
            .iter()
            .map(|item| {
                item.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| GgufError::WrongType {
                        key: key.to_string(),
                        expected: "array of string",
                        found: item.type_name(),
                    })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests;
