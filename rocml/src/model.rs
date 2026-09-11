//! Architecture dispatch: reads `general.architecture` from the GGUF and
//! loads either the dense Qwen3 forward pass (`crate::forward::Model`) or
//! the qwen35 hybrid one (`crate::qwen35::forward::Model`), then presents a
//! single decode-loop surface (`forward_token`/`reset`/`position`/
//! `memory_info`) so `crate::generate::generate` and the CLI examples don't
//! need to know which architecture they're driving.

use std::path::Path;

use rocml_core::gguf::GgufFile;
use rocml_hip::MemoryInfo;

use crate::error::RocmlError;

pub enum Model {
    // Both variants are boxed: each holds a whole forward pass's weights,
    // cache and scratch buffers, and clippy flags any size skew between
    // them at this enum's own stack size otherwise.
    Dense(Box<crate::forward::Model>),
    Hybrid(Box<crate::qwen35::forward::Model>),
}

impl Model {
    /// Opens `gguf_path` just far enough to read `general.architecture`,
    /// then hands off to that architecture's own loader (which reopens the
    /// file itself — GGUF opens are a cheap mmap, not worth threading a
    /// shared handle through two unrelated loaders for).
    pub fn load(gguf_path: impl AsRef<Path>) -> Result<Self, RocmlError> {
        let path = gguf_path.as_ref();
        let arch = GgufFile::open(path)?
            .get_str("general.architecture")?
            .to_string();
        match arch.as_str() {
            "qwen3" => Ok(Self::Dense(Box::new(crate::forward::Model::load(path)?))),
            "qwen35" => Ok(Self::Hybrid(Box::new(crate::qwen35::forward::Model::load(
                path,
            )?))),
            other => Err(RocmlError::UnsupportedArchitecture {
                found: other.to_string(),
            }),
        }
    }

    pub fn forward_token(&mut self, token_id: u32) -> Result<Vec<f32>, RocmlError> {
        match self {
            Self::Dense(m) => m.forward_token(token_id),
            Self::Hybrid(m) => m.forward_token(token_id),
        }
    }

    pub fn reset(&mut self) -> Result<(), RocmlError> {
        match self {
            Self::Dense(m) => {
                m.reset();
                Ok(())
            }
            Self::Hybrid(m) => m.reset(),
        }
    }

    pub fn position(&self) -> u32 {
        match self {
            Self::Dense(m) => m.position(),
            Self::Hybrid(m) => m.position(),
        }
    }

    pub fn memory_info(&self) -> Result<MemoryInfo, RocmlError> {
        match self {
            Self::Dense(m) => m.memory_info(),
            Self::Hybrid(m) => m.memory_info(),
        }
    }

    pub fn block_count(&self) -> u32 {
        match self {
            Self::Dense(m) => m.config().block_count,
            Self::Hybrid(m) => m.config().block_count,
        }
    }

    pub fn embedding_length(&self) -> u32 {
        match self {
            Self::Dense(m) => m.config().embedding_length,
            Self::Hybrid(m) => m.config().embedding_length,
        }
    }

    pub fn vocab_size(&self) -> u32 {
        match self {
            Self::Dense(m) => m.config().vocab_size,
            Self::Hybrid(m) => m.config().vocab_size,
        }
    }
}
