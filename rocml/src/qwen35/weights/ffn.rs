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
        Self::load_named(
            gguf,
            hidden,
            ffn,
            &format!("{prefix}.ffn_gate.weight"),
            &format!("{prefix}.ffn_up.weight"),
            &format!("{prefix}.ffn_down.weight"),
        )
    }

    /// Like [`Self::load`], but with explicit tensor names — the qwen35moe
    /// shared expert (`ffn_{gate,up,down}_shexp.weight`) reuses this same
    /// gate/up/down SwiGLU shape with different tensor names, so this is the
    /// one place that shape's loading logic lives (see
    /// `crate::qwen35::weights::moe::MoeFfnWeights::load`).
    pub(crate) fn load_named(
        gguf: &GgufFile,
        hidden: u32,
        ffn: u32,
        gate_name: &str,
        up_name: &str,
        down_name: &str,
    ) -> Result<Self, RocmlError> {
        Ok(Self {
            gate: LinearWeight::load(gguf, gate_name, ffn, hidden)?,
            up: LinearWeight::load(gguf, up_name, ffn, hidden)?,
            down: LinearWeight::load(gguf, down_name, hidden, ffn)?,
        })
    }
}
