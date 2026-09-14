//! Loads every GGUF tensor Qwen3 needs onto the GPU: CPU-dequantize
//! (rocml-core) to f32, then either cast to f16 and upload (linear/matmul
//! weights, the token embedding table) or upload as-is (norm weights, which
//! stay f32 through the whole forward pass for numerical stability).

mod layer;
mod linear;

pub use layer::LayerWeights;
pub use linear::LinearWeight;

use half::f16;
use rocml_core::gguf::GgufFile;
use rocml_core::quant::dequantize;
use rocml_hip::DeviceBuffer;

use crate::config::ModelConfig;
use crate::error::RocmlError;

pub struct ModelWeights {
    /// Row-major [vocab, hidden]: row `id` is token `id`'s embedding. Always
    /// dequantized to f16 — the embedding lookup kernel only reads f16.
    pub token_embd: DeviceBuffer<f16>,
    pub output_norm: DeviceBuffer<f32>,
    /// (m=vocab, n=hidden). Qwen3-0.6B ties this to `token_embd.weight` (no
    /// separate `output.weight` tensor) — handled by re-loading the same
    /// tensor rather than sharing a buffer, since `DeviceBuffer` has no
    /// cheap aliasing story and this only costs a one-time extra load at
    /// startup. This is the single biggest matvec in the model, so it's
    /// loaded through `LinearWeight` like every other linear layer — it
    /// stays quantized whenever the loader policy allows it.
    pub output: LinearWeight,
    pub layers: Vec<LayerWeights>,
}

impl ModelWeights {
    pub fn load(gguf: &GgufFile, config: &ModelConfig) -> Result<Self, RocmlError> {
        crate::quant_policy::audit(gguf, crate::quant_policy::ArchFamily::Qwen3Dense)
            .warn_violations();

        let token_embd = load_matrix_f16(
            gguf,
            "token_embd.weight",
            config.vocab_size,
            config.embedding_length,
        )?;
        let output_norm = load_vector_f32(gguf, "output_norm.weight", config.embedding_length)?;
        let output = if gguf.tensor("output.weight").is_ok() {
            LinearWeight::load(
                gguf,
                "output.weight",
                config.vocab_size,
                config.embedding_length,
            )?
        } else {
            LinearWeight::load(
                gguf,
                "token_embd.weight",
                config.vocab_size,
                config.embedding_length,
            )?
        };

        let mut layers = Vec::with_capacity(config.block_count as usize);
        for i in 0..config.block_count {
            layers.push(LayerWeights::load(gguf, config, i)?);
        }

        Ok(Self {
            token_embd,
            output_norm,
            output,
            layers,
        })
    }
}

/// `ne` order is `[n, m]` (ne[0] is the fastest-varying/innermost dim); GGUF
/// stores tensor bytes with ne[0] contiguous, which is exactly a row-major
/// `[m, n]` layout (`m` rows of `n` contiguous elements) — the same shape
/// `gemv_f16`/`gemv_f32` expect (`m` = output rows, `n` = reduction width).
fn matrix_dims(shape: &[u64], name: &str) -> Result<(u32, u32), RocmlError> {
    let &[n, m] = shape else {
        return Err(RocmlError::UnexpectedTensorRank {
            name: name.to_string(),
            expected: 2,
            found: shape.len(),
        });
    };
    let to_u32 = |value: u64, index: usize| {
        u32::try_from(value).map_err(|_| RocmlError::InvalidTensorDim {
            name: name.to_string(),
            index,
            value,
        })
    };
    Ok((to_u32(m, 1)?, to_u32(n, 0)?))
}

/// Dequantizes tensor `name` to f32, casts to f16 and uploads it, validating
/// its shape is exactly `(expected_m, expected_n)` in `gemv_f16` terms.
pub(crate) fn load_matrix_f16(
    gguf: &GgufFile,
    name: &str,
    expected_m: u32,
    expected_n: u32,
) -> Result<DeviceBuffer<f16>, RocmlError> {
    let view = gguf.tensor(name)?;
    let (m, n) = matrix_dims(view.shape(), name)?;
    if (m, n) != (expected_m, expected_n) {
        return Err(RocmlError::Config(format!(
            "tensor {name:?}: shape ({m} x {n}) doesn't match expected ({expected_m} x {expected_n})"
        )));
    }
    let f32_data = dequantize(view.dtype(), view.data())?;
    let f16_data: Vec<f16> = f32_data.iter().map(|&v| f16::from_f32(v)).collect();
    let mut buf = DeviceBuffer::<f16>::new(f16_data.len())?;
    buf.copy_from_host(&f16_data)?;
    Ok(buf)
}

/// Dequantizes tensor `name` to f32 and uploads it as-is (no f16 cast),
/// validating its shape is exactly `(expected_m, expected_n)` in `gemv_f32`
/// terms. For small, numerically sensitive matrices that skip the f16
/// round-trip the big matmul weights take (e.g. a GDN layer's depthwise
/// conv1d kernel).
pub(crate) fn load_matrix_f32(
    gguf: &GgufFile,
    name: &str,
    expected_m: u32,
    expected_n: u32,
) -> Result<DeviceBuffer<f32>, RocmlError> {
    let view = gguf.tensor(name)?;
    let (m, n) = matrix_dims(view.shape(), name)?;
    if (m, n) != (expected_m, expected_n) {
        return Err(RocmlError::Config(format!(
            "tensor {name:?}: shape ({m} x {n}) doesn't match expected ({expected_m} x {expected_n})"
        )));
    }
    let data = dequantize(view.dtype(), view.data())?;
    let mut buf = DeviceBuffer::<f32>::new(data.len())?;
    buf.copy_from_host(&data)?;
    Ok(buf)
}

/// Dequantizes and uploads a 1D norm-weight tensor as f32, validating its
/// length is exactly `expected_len`.
pub(crate) fn load_vector_f32(
    gguf: &GgufFile,
    name: &str,
    expected_len: u32,
) -> Result<DeviceBuffer<f32>, RocmlError> {
    let view = gguf.tensor(name)?;
    let &[dim] = view.shape() else {
        return Err(RocmlError::UnexpectedTensorRank {
            name: name.to_string(),
            expected: 1,
            found: view.shape().len(),
        });
    };
    let len = u32::try_from(dim).map_err(|_| RocmlError::InvalidTensorDim {
        name: name.to_string(),
        index: 0,
        value: dim,
    })?;
    if len != expected_len {
        return Err(RocmlError::Config(format!(
            "tensor {name:?}: length {len} doesn't match expected {expected_len}"
        )));
    }
    let data = dequantize(view.dtype(), view.data())?;
    let mut buf = DeviceBuffer::<f32>::new(data.len())?;
    buf.copy_from_host(&data)?;
    Ok(buf)
}
