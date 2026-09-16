//! Loads every GGUF tensor the qwen35 hybrid architecture needs onto the
//! GPU. Follows the same convention as the dense Qwen3 loader: CPU-dequant
//! (rocml-core) to f32, then f16 for matmul weights, f32 for norm/gate/state
//! tensors that stay full precision through the whole forward pass.

mod attention;
mod ffn;
mod gdn;
mod layer;
pub mod moe;

pub use attention::AttnLayerWeights;
pub use ffn::FfnWeights;
pub use gdn::GdnLayerWeights;
pub use layer::LayerWeights;
pub use moe::MoeFfnWeights;

use half::f16;
use rocml_core::gguf::GgufFile;
use rocml_hip::DeviceBuffer;

use super::config::Qwen35Config;
use crate::error::RocmlError;
use crate::weights::{load_matrix_f16, load_vector_f32, LinearWeight};

/// One layer's FFN: a dense SwiGLU MLP (`"qwen35"`) or a mixture of experts
/// (`"qwen35moe"`) — see `crate::qwen35::config::Qwen35Config::moe`.
pub enum Ffn {
    Dense(FfnWeights),
    Moe(Box<MoeFfnWeights>),
}

impl Ffn {
    pub(crate) fn load(
        gguf: &GgufFile,
        prefix: &str,
        cfg: &Qwen35Config,
    ) -> Result<Self, RocmlError> {
        match &cfg.moe {
            Some(moe_cfg) => Ok(Self::Moe(Box::new(MoeFfnWeights::load(
                gguf,
                prefix,
                moe_cfg,
                cfg.embedding_length,
            )?))),
            None => Ok(Self::Dense(FfnWeights::load(
                gguf,
                prefix,
                cfg.embedding_length,
                cfg.feed_forward_length,
            )?)),
        }
    }
}

pub struct ModelWeights {
    pub token_embd: DeviceBuffer<f16>,
    pub output_norm: DeviceBuffer<f32>,
    pub output: LinearWeight,
    pub layers: Vec<LayerWeights>,
}

impl ModelWeights {
    pub fn load(gguf: &GgufFile, config: &Qwen35Config) -> Result<Self, RocmlError> {
        let arch_family = if config.moe.is_some() {
            crate::quant_policy::ArchFamily::Qwen35Moe
        } else {
            crate::quant_policy::ArchFamily::Qwen35Hybrid
        };
        crate::quant_policy::audit(gguf, arch_family).warn_violations();

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
