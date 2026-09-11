//! Single-token decode-style forward pass: embedding lookup, N transformer
//! layers (attention.rs + ffn.rs), final norm, logits. Prompt processing
//! reuses this exact path one token at a time — batched prefill (a fused
//! multi-token gemm path) is a later performance milestone, not this one.

mod attention;
mod ffn;
pub(crate) mod kernels;
mod scratch;

use std::path::Path;

use kernels::{offset, Kernels};
use rocml_core::gguf::GgufFile;
use rocml_hip::{Device, MemoryInfo};
use scratch::Scratch;

use crate::cache::KvCache;
use crate::config::ModelConfig;
use crate::error::RocmlError;
use crate::weights::ModelWeights;

pub struct Model {
    // Kept alive for the model's lifetime: HIP device selection is
    // per-thread global state, not scoped to this handle, but every other
    // field here assumes device 0 stays selected and its context alive.
    _device: Device,
    config: ModelConfig,
    weights: ModelWeights,
    cache: KvCache,
    kernels: Kernels,
    scratch: Scratch,
    pos: u32,
}

impl Model {
    /// Loads a dense Qwen3 GGUF onto the GPU: opens/mmaps the file, reads
    /// config, dequantizes and uploads every weight, allocates the KV cache
    /// and scratch buffers, and loads every kernel this forward pass needs.
    pub fn load(gguf_path: impl AsRef<Path>) -> Result<Self, RocmlError> {
        let device = Device::new(0)?;
        let gguf = GgufFile::open(gguf_path)?;
        let config = ModelConfig::from_gguf(&gguf)?;
        let weights = ModelWeights::load(&gguf, &config)?;
        let cache = KvCache::new(&config)?;
        let kernels = Kernels::load_all()?;
        let scratch = Scratch::new(&config, cache.max_seq())?;

        Ok(Self {
            _device: device,
            config,
            weights,
            cache,
            kernels,
            scratch,
            pos: 0,
        })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Free/total VRAM on the current device, for reporting how much a load
    /// consumed (see the `generate` example).
    pub fn memory_info(&self) -> Result<MemoryInfo, RocmlError> {
        self._device.memory_info().map_err(Into::into)
    }

    /// Number of tokens already fed through `forward_token` (the next call
    /// writes cache position `position()`).
    pub fn position(&self) -> u32 {
        self.pos
    }

    /// Clears the cache position so the next `forward_token` starts a fresh
    /// sequence. Cheap: cache buffers are only ever read up to the current
    /// position, so stale bytes beyond it are never observed — no need to
    /// re-zero them.
    pub fn reset(&mut self) {
        self.pos = 0;
    }

    /// Runs one decode step for `token_id` at the current cache position and
    /// returns that step's vocab-sized logits (host-side, f32).
    pub fn forward_token(&mut self, token_id: u32) -> Result<Vec<f32>, RocmlError> {
        let pos = self.pos;
        let max_seq = self.cache.max_seq();
        if pos >= max_seq {
            return Err(RocmlError::ContextOverflow {
                requested: pos + 1,
                max_seq,
            });
        }
        let hidden = self.config.embedding_length;

        self.scratch.token_id.copy_from_host(&[token_id])?;
        self.kernels.embedding(
            offset(&self.scratch.token_id, 0),
            offset(&self.weights.token_embd, 0),
            offset(&self.scratch.x, 0),
            1,
            hidden,
        )?;

        for (layer_idx, layer) in self.weights.layers.iter().enumerate() {
            attention::attention_step(
                &self.kernels,
                &self.config,
                layer,
                &mut self.cache,
                &mut self.scratch,
                layer_idx,
                pos,
            )?;
            ffn::ffn_step(&self.kernels, &self.config, layer, &mut self.scratch)?;
        }

        self.kernels.rmsnorm(
            offset(&self.scratch.x, 0),
            offset(&self.weights.output_norm, 0),
            offset(&self.scratch.xn, 0),
            1,
            hidden,
            self.config.rms_eps,
        )?;
        self.kernels.gemv_f16(
            offset(&self.weights.output, 0),
            offset(&self.scratch.xn, 0),
            offset(&self.scratch.logits, 0),
            self.config.vocab_size,
            hidden,
        )?;

        let mut logits = vec![0.0f32; self.config.vocab_size as usize];
        self.scratch.logits.copy_to_host(&mut logits)?;

        self.pos += 1;
        Ok(logits)
    }
}
