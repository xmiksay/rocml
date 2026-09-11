//! ggml quantized (and plain float) tensor dtypes: block byte layout,
//! elements-per-block, and CPU dequantize-to-f32 for the types rocml
//! actually loads. These CPU implementations are the ground-truth reference
//! the GPU dequant kernels get validated against in later milestones, so
//! correctness — matching ggml's own bit-for-bit math — outranks speed here.

mod common;
mod q2_k;
mod q3_k;
mod q4_k;
mod q5_k;
mod q6_k;
mod q8_0;

pub use q2_k::BlockQ2K;
pub use q3_k::BlockQ3K;
pub use q4_k::BlockQ4K;
pub use q5_k::BlockQ5K;
pub use q6_k::BlockQ6K;
pub use q8_0::BlockQ8_0;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum QuantError {
    #[error("dequantize: {n_bytes} input bytes is not a multiple of the block size ({block_bytes} bytes/block for {dtype:?})")]
    LengthNotMultipleOfBlock {
        n_bytes: usize,
        block_bytes: usize,
        dtype: GgmlDType,
    },
    #[error("dtype {0:?} has no CPU dequantize implementation")]
    NoDequantSupport(GgmlDType),
}

/// ggml tensor element type. Numeric ids match `enum ggml_type` in ggml.h,
/// used verbatim by the GGUF tensor-info dtype field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
// Variant names intentionally mirror ggml's own quant naming (Q4_K, not
// Q4K) so they're greppable against ggml/llama.cpp source and docs.
#[allow(non_camel_case_types)]
pub enum GgmlDType {
    F32,
    F16,
    BF16,
    Q8_0,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    /// A recognized-but-not-yet-implemented ggml dtype id. Tensors of this
    /// dtype still show up in `tensors()` (name/shape/dtype), they just
    /// can't be validated, sliced, or dequantized.
    Unsupported(u32),
}

impl GgmlDType {
    pub fn from_ggml_id(id: u32) -> Self {
        match id {
            0 => Self::F32,
            1 => Self::F16,
            8 => Self::Q8_0,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            30 => Self::BF16,
            other => Self::Unsupported(other),
        }
    }

    /// Number of elements packed into one block. `1` for plain float types.
    pub fn block_elements(&self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::Q8_0 => q8_0::QK8_0,
            Self::Q2_K | Self::Q3_K | Self::Q4_K | Self::Q5_K | Self::Q6_K => common::QK_K,
            Self::Unsupported(_) => 0,
        }
    }

    /// On-disk byte size of one block.
    pub fn block_bytes(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q8_0 => std::mem::size_of::<BlockQ8_0>(),
            Self::Q2_K => std::mem::size_of::<BlockQ2K>(),
            Self::Q3_K => std::mem::size_of::<BlockQ3K>(),
            Self::Q4_K => std::mem::size_of::<BlockQ4K>(),
            Self::Q5_K => std::mem::size_of::<BlockQ5K>(),
            Self::Q6_K => std::mem::size_of::<BlockQ6K>(),
            Self::Unsupported(_) => 0,
        }
    }
}

/// Dequantizes raw block bytes (as sliced from a `TensorView`) to `f32`.
///
/// `bytes.len()` must be an exact multiple of `dtype.block_bytes()`; this is
/// guaranteed for any slice obtained through `GgufFile::tensor`, since that
/// path already validates it, but this function re-checks so it stays safe
/// to call directly (e.g. from tests feeding synthetic blocks).
pub fn dequantize(dtype: GgmlDType, bytes: &[u8]) -> Result<Vec<f32>, QuantError> {
    let block_bytes = dtype.block_bytes();
    if block_bytes == 0 || !bytes.len().is_multiple_of(block_bytes) {
        return Err(QuantError::LengthNotMultipleOfBlock {
            n_bytes: bytes.len(),
            block_bytes,
            dtype,
        });
    }
    match dtype {
        GgmlDType::F32 => Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        GgmlDType::F16 => Ok(bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect()),
        GgmlDType::BF16 => Ok(bytes
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect()),
        GgmlDType::Q8_0 => Ok(q8_0::dequantize(bytes)),
        GgmlDType::Q2_K => Ok(q2_k::dequantize(bytes)),
        GgmlDType::Q3_K => Ok(q3_k::dequantize(bytes)),
        GgmlDType::Q4_K => Ok(q4_k::dequantize(bytes)),
        GgmlDType::Q5_K => Ok(q5_k::dequantize(bytes)),
        GgmlDType::Q6_K => Ok(q6_k::dequantize(bytes)),
        GgmlDType::Unsupported(_) => Err(QuantError::NoDequantSupport(dtype)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_bytes_match_ggml() {
        assert_eq!(GgmlDType::Q8_0.block_bytes(), 34);
        assert_eq!(GgmlDType::Q2_K.block_bytes(), 84);
        assert_eq!(GgmlDType::Q3_K.block_bytes(), 110);
        assert_eq!(GgmlDType::Q4_K.block_bytes(), 144);
        assert_eq!(GgmlDType::Q5_K.block_bytes(), 176);
        assert_eq!(GgmlDType::Q6_K.block_bytes(), 210);
    }

    #[test]
    fn from_ggml_id_roundtrips_known_ids() {
        assert_eq!(GgmlDType::from_ggml_id(0), GgmlDType::F32);
        assert_eq!(GgmlDType::from_ggml_id(1), GgmlDType::F16);
        assert_eq!(GgmlDType::from_ggml_id(8), GgmlDType::Q8_0);
        assert_eq!(GgmlDType::from_ggml_id(10), GgmlDType::Q2_K);
        assert_eq!(GgmlDType::from_ggml_id(14), GgmlDType::Q6_K);
        assert_eq!(GgmlDType::from_ggml_id(30), GgmlDType::BF16);
        assert_eq!(GgmlDType::from_ggml_id(999), GgmlDType::Unsupported(999));
    }

    #[test]
    fn dequantize_rejects_misaligned_length() {
        let err = dequantize(GgmlDType::Q8_0, &[0u8; 10]).unwrap_err();
        assert!(matches!(err, QuantError::LengthNotMultipleOfBlock { .. }));
    }
}
