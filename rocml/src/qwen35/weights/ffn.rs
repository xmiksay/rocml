//! SwiGLU FFN weights, shared by both qwen35 layer kinds.

use half::f16;
use rocml_core::gguf::GgufFile;
use rocml_hip::DeviceBuffer;

use crate::error::RocmlError;
use crate::weights::load_matrix_f16;

pub struct FfnWeights {
    /// gemv_f16 shape (m=feed_forward_length, n=hidden).
    pub gate: DeviceBuffer<f16>,
    /// gemv_f16 shape (m=feed_forward_length, n=hidden).
    pub up: DeviceBuffer<f16>,
    /// gemv_f16 shape (m=hidden, n=feed_forward_length).
    pub down: DeviceBuffer<f16>,
}

impl FfnWeights {
    pub(crate) fn load(
        gguf: &GgufFile,
        prefix: &str,
        hidden: u32,
        ffn: u32,
    ) -> Result<Self, RocmlError> {
        Ok(Self {
            gate: load_matrix_f16(gguf, &format!("{prefix}.ffn_gate.weight"), ffn, hidden)?,
            up: load_matrix_f16(gguf, &format!("{prefix}.ffn_up.weight"), ffn, hidden)?,
            down: load_matrix_f16(gguf, &format!("{prefix}.ffn_down.weight"), hidden, ffn)?,
        })
    }
}
