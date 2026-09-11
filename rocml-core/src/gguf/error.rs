//! Every failure mode a malformed or hostile GGUF file can trigger. Parsing
//! never panics: bad input always turns into one of these variants instead.

use crate::quant::GgmlDType;

#[derive(Debug, thiserror::Error)]
pub enum GgufError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("not a GGUF file: bad magic bytes")]
    InvalidMagic,

    #[error("unsupported GGUF version {0} (only v2 and v3 are supported)")]
    UnsupportedVersion(u32),

    #[error("unexpected end of file while reading {context}")]
    UnexpectedEof { context: &'static str },

    #[error("invalid UTF-8 string while reading {context}")]
    InvalidUtf8 { context: &'static str },

    #[error("unknown metadata value type {0}")]
    UnknownValueType(u32),

    #[error("metadata array nesting exceeds the maximum supported depth")]
    ArrayTooDeep,

    #[error("value too large to fit in memory address space while reading {context}")]
    SizeOverflow { context: &'static str },

    #[error("duplicate metadata key: {0}")]
    DuplicateKey(String),

    #[error("duplicate tensor name: {0}")]
    DuplicateTensor(String),

    #[error("missing metadata key: {0}")]
    MissingKey(String),

    #[error("metadata key {key} has the wrong type: expected {expected}, found {found}")]
    WrongType {
        key: String,
        expected: &'static str,
        found: &'static str,
    },

    #[error("tensor not found: {0}")]
    TensorNotFound(String),

    #[error(
        "tensor {name} byte range [{offset}, {end}) does not fit inside the data section (len {data_len})"
    )]
    TensorOutOfBounds {
        name: String,
        offset: u64,
        end: u64,
        data_len: u64,
    },

    #[error(
        "tensor {name} has {n_elements} elements, not a multiple of the block size ({block_size}) for dtype {dtype:?}"
    )]
    BadBlockAlignment {
        name: String,
        n_elements: u64,
        block_size: u64,
        dtype: GgmlDType,
    },

    #[error("tensor {name} uses unsupported ggml dtype id {dtype_id}, its raw bytes cannot be validated or sliced")]
    UnsupportedTensorDType { name: String, dtype_id: u32 },
}
