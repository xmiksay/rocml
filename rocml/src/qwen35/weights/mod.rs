//! Loads every GGUF tensor the qwen35 hybrid architecture needs onto the
//! GPU. Follows the same convention as the dense Qwen3 loader: CPU-dequant
//! (rocml-core) to f32, then f16 for matmul weights, f32 for norm/gate/state
//! tensors that stay full precision through the whole forward pass.

mod attention;
mod ffn;
mod gdn;
mod layer;

pub use attention::AttnLayerWeights;
pub use ffn::FfnWeights;
pub use gdn::GdnLayerWeights;
pub use layer::LayerWeights;

use half::f16;
use rocml_core::gguf::GgufFile;
use rocml_hip::DeviceBuffer;

use super::config::Qwen35Config;
use crate::error::RocmlError;
use crate::weights::{load_matrix_f16, load_vector_f32};

pub struct ModelWeights {
    pub token_embd: DeviceBuffer<f16>,
    pub output_norm: DeviceBuffer<f32>,
    pub output: DeviceBuffer<f16>,
    pub layers: Vec<LayerWeights>,
}

impl ModelWeights {
    pub fn load(gguf: &GgufFile, config: &Qwen35Config) -> Result<Self, RocmlError> {
        let token_embd = load_matrix_f16(
            gguf,
            "token_embd.weight",
            config.vocab_size,
            config.embedding_length,
        )?;
        let output_norm = load_vector_f32(gguf, "output_norm.weight", config.embedding_length)?;
        let output = if gguf.tensor("output.weight").is_ok() {
            load_matrix_f16(
                gguf,
                "output.weight",
                config.vocab_size,
                config.embedding_length,
            )?
        } else {
            load_matrix_f16(
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
