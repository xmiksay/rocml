//! Per-layer weight enum, dispatching on the layer's `LayerKind`.

use rocml_core::gguf::GgufFile;

use super::attention::AttnLayerWeights;
use super::gdn::GdnLayerWeights;
use crate::error::RocmlError;
use crate::qwen35::config::{LayerKind, Qwen35Config};

pub enum LayerWeights {
    Gdn(GdnLayerWeights),
    Attention(AttnLayerWeights),
}

impl LayerWeights {
    pub(crate) fn load(
        gguf: &GgufFile,
        cfg: &Qwen35Config,
        layer_idx: u32,
    ) -> Result<Self, RocmlError> {
        match cfg.layer_kinds[layer_idx as usize] {
            LayerKind::LinearAttention => {
                Ok(Self::Gdn(GdnLayerWeights::load(gguf, cfg, layer_idx)?))
            }
            LayerKind::FullAttention => Ok(Self::Attention(AttnLayerWeights::load(
                gguf, cfg, layer_idx,
            )?)),
        }
    }
}
