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
use crate::load_opts::LoadOptions;
use crate::profile::Profiler;

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
    pub fn load(gguf_path: impl AsRef<Path>, opts: LoadOptions) -> Result<Self, RocmlError> {
        let path = gguf_path.as_ref();
        let arch = GgufFile::open(path)?
            .get_str("general.architecture")?
            .to_string();
        match arch.as_str() {
            "qwen3" => Ok(Self::Dense(Box::new(crate::forward::Model::load(
                path, opts,
            )?))),
            "qwen35" => Ok(Self::Hybrid(Box::new(crate::qwen35::forward::Model::load(
                path, opts,
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

    /// Like [`Self::forward_token`], instrumented through `prof` when given
    /// — see `crate::profile` for what gets recorded and at what
    /// granularity.
    pub fn forward_token_profiled(
        &mut self,
        token_id: u32,
        prof: Option<&Profiler>,
    ) -> Result<Vec<f32>, RocmlError> {
        match self {
            Self::Dense(m) => m.forward_token_profiled(token_id, prof),
            Self::Hybrid(m) => m.forward_token_profiled(token_id, prof),
        }
    }

    /// Processes a whole (non-empty) prompt and returns the logits for its
    /// last token. The hybrid (qwen35) architecture processes `prompt_ids`
    /// in batched chunks (issue #6 — see
    /// `qwen35::forward::Model::forward_prompt`, which itself falls back to
    /// token-serial prefill when the cache has any mixed/quantized layer —
    /// issue #2's chunked-prefill support is a documented follow-up); the
    /// dense architecture doesn't yet have a chunked forward pass, so it
    /// falls back to the original token-serial loop (out of scope for issue
    /// #6, which targets the hybrid models `bench`/the parity suites cover).
    pub fn forward_prompt(
        &mut self,
        prompt_ids: &[u32],
        prof: Option<&Profiler>,
    ) -> Result<Vec<f32>, RocmlError> {
        match self {
            Self::Dense(m) => {
                let mut logits = Vec::new();
                for &id in prompt_ids {
                    logits = m.forward_token_profiled(id, prof)?;
                }
                Ok(logits)
            }
            Self::Hybrid(m) => m.forward_prompt(prompt_ids, prof),
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

    /// The snapshot layer (issue #1) only supports the qwen35 hybrid
    /// architecture — see `crate::snapshot`'s module doc for why (dense
    /// `qwen3`'s KV cache has no GDN-style fixed-size state to make a cheap
    /// mid-conversation checkpoint interesting, and no chunked prefill to
    /// hook capture into). Callers use these to detect which architecture
    /// they have and skip snapshot lookup/capture entirely for `Dense`,
    /// rather than this dispatch type growing snapshot-specific error arms.
    pub fn as_hybrid(&self) -> Option<&crate::qwen35::forward::Model> {
        match self {
            Self::Hybrid(m) => Some(m),
            Self::Dense(_) => None,
        }
    }

    pub fn as_hybrid_mut(&mut self) -> Option<&mut crate::qwen35::forward::Model> {
        match self {
            Self::Hybrid(m) => Some(m),
            Self::Dense(_) => None,
        }
    }
}
