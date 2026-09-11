//! SwiGLU FFN weights, shared by both qwen35 layer kinds.

use rocml_core::gguf::GgufFile;

use crate::error::RocmlError;
use crate::weights::LinearWeight;

pub struct FfnWeights {
    /// (m=feed_forward_length, n=hidden).
    pub gate: LinearWeight,
    /// (m=feed_forward_length, n=hidden).
    pub up: LinearWeight,
    /// (m=hidden, n=feed_forward_length).
    pub down: LinearWeight,
}

impl FfnWeights {
    pub(crate) fn load(
        gguf: &GgufFile,
        prefix: &str,
        hidden: u32,
        ffn: u32,
    ) -> Result<Self, RocmlError> {
        Ok(Self {
            gate: LinearWeight::load(gguf, &format!("{prefix}.ffn_gate.weight"), ffn, hidden)?,
            up: LinearWeight::load(gguf, &format!("{prefix}.ffn_up.weight"), ffn, hidden)?,
            down: LinearWeight::load(gguf, &format!("{prefix}.ffn_down.weight"), hidden, ffn)?,
        })
    }
}
