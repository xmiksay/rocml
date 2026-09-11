//! Single-token decode-style forward pass for the qwen35 hybrid
//! architecture: embedding, N transformer blocks (each either a GDN or a
//! full-attention layer, per `Qwen35Config::layer_kinds`), final norm,
//! logits. Mirrors `crate::forward` (the dense Qwen3 path) structurally.

mod attention;
mod ffn;
mod gdn;
mod kernels;
mod scratch;

use std::path::Path;

use kernels::HybridKernels;
use rocml_core::gguf::GgufFile;
use rocml_hip::{Device, MemoryInfo};
use scratch::Scratch;

use super::cache::HybridCache;
use super::config::{LayerKind, Qwen35Config};
use super::weights::{LayerWeights, ModelWeights};
use crate::error::RocmlError;
use crate::forward::kernels::{offset, Kernels};
use crate::profile::{self, OpKind, Phase, Profiler};

pub struct Model {
    _device: Device,
    config: Qwen35Config,
    weights: ModelWeights,
    cache: HybridCache,
    kernels: Kernels,
    hybrid: HybridKernels,
    scratch: Scratch,
    pos: u32,
}

impl Model {
    pub fn load(gguf_path: impl AsRef<Path>) -> Result<Self, RocmlError> {
        let device = Device::new(0)?;
        let gguf = GgufFile::open(gguf_path)?;
        let config = Qwen35Config::from_gguf(&gguf)?;
        let weights = ModelWeights::load(&gguf, &config)?;
        let cache = HybridCache::new(&config)?;
        let kernels = Kernels::load_all()?;
        let hybrid = HybridKernels::load_all()?;
        let scratch = Scratch::new(&config, cache.max_seq())?;

        Ok(Self {
            _device: device,
            config,
            weights,
            cache,
            kernels,
            hybrid,
            scratch,
            pos: 0,
        })
    }

    pub fn config(&self) -> &Qwen35Config {
        &self.config
    }

    pub fn memory_info(&self) -> Result<MemoryInfo, RocmlError> {
        self._device.memory_info().map_err(Into::into)
    }

    pub fn position(&self) -> u32 {
        self.pos
    }

    /// Resets decode position and zeroes every GDN layer's conv/recurrence
    /// state (full-attention K/V planes need no explicit reset — see
    /// `HybridCache::reset`'s doc comment).
    pub fn reset(&mut self) -> Result<(), RocmlError> {
        self.pos = 0;
        self.cache.reset()
    }

    pub fn forward_token(&mut self, token_id: u32) -> Result<Vec<f32>, RocmlError> {
        self.forward_token_profiled(token_id, None)
    }

    /// Like [`Self::forward_token`], but instruments every op through `prof`
    /// when given — see `crate::forward::Model::forward_token_profiled`'s
    /// doc comment for the prefill-vs-decode granularity split, mirrored
    /// here.
    pub fn forward_token_profiled(
        &mut self,
        token_id: u32,
        prof: Option<&Profiler>,
    ) -> Result<Vec<f32>, RocmlError> {
        let pos = self.pos;
        let max_seq = self.cache.max_seq();
        if pos >= max_seq {
            return Err(RocmlError::ContextOverflow {
                requested: pos + 1,
                max_seq,
            });
        }
        let hidden = self.config.embedding_length;
        let cur_len = pos + 1;
        let coarse_prefill = prof.map(|p| p.phase() == Phase::Prefill).unwrap_or(false);

        self.scratch.token_id.copy_from_host(&[token_id])?;
        Profiler::scope(
            prof,
            None,
            OpKind::Embed,
            profile::embed_bytes(hidden),
            0,
            || {
                self.kernels.embedding(
                    offset(&self.scratch.token_id, 0),
                    offset(&self.weights.token_embd, 0),
                    offset(&self.scratch.x, 0),
                    1,
                    hidden,
                )
            },
        )?;

        for (layer_idx, layer) in self.weights.layers.iter().enumerate() {
            let layer_idx_u32 = layer_idx as u32;
            match (layer, self.config.layer_kinds[layer_idx]) {
                (LayerWeights::Gdn(gdn_weights), LayerKind::LinearAttention) => {
                    let state = self.cache.gdn_mut(layer_idx)?;
                    if coarse_prefill {
                        let (gdn_bytes, gdn_flops) =
                            gdn::gdn_layer_step_cost(&self.config, gdn_weights);
                        let (ffn_bytes, ffn_flops) = ffn::ffn_step_cost(
                            &gdn_weights.ffn,
                            hidden,
                            self.config.feed_forward_length,
                        );
                        Profiler::scope(
                            prof,
                            Some(layer_idx_u32),
                            OpKind::Layer,
                            gdn_bytes + ffn_bytes,
                            gdn_flops + ffn_flops,
                            || {
                                gdn::gdn_layer_step(
                                    &self.kernels,
                                    &self.hybrid,
                                    &self.config,
                                    gdn_weights,
                                    state,
                                    &mut self.scratch,
                                    None,
                                    None,
                                )?;
                                ffn::ffn_step(
                                    &self.kernels,
                                    &gdn_weights.ffn,
                                    &gdn_weights.post_attention_norm,
                                    hidden,
                                    self.config.feed_forward_length,
                                    self.config.rms_eps,
                                    &mut self.scratch,
                                    None,
                                    None,
                                )
                            },
                        )?;
                    } else {
                        gdn::gdn_layer_step(
                            &self.kernels,
                            &self.hybrid,
                            &self.config,
                            gdn_weights,
                            state,
                            &mut self.scratch,
                            prof,
                            Some(layer_idx_u32),
                        )?;
                        ffn::ffn_step(
                            &self.kernels,
                            &gdn_weights.ffn,
                            &gdn_weights.post_attention_norm,
                            hidden,
                            self.config.feed_forward_length,
                            self.config.rms_eps,
                            &mut self.scratch,
                            prof,
                            Some(layer_idx_u32),
                        )?;
                    }
                }
                (LayerWeights::Attention(attn_weights), LayerKind::FullAttention) => {
                    let plane = self.cache.attn_mut(layer_idx)?;
                    if coarse_prefill {
                        let (attn_bytes, attn_flops) =
                            attention::attention_step_cost(&self.config, attn_weights, cur_len);
                        let (ffn_bytes, ffn_flops) = ffn::ffn_step_cost(
                            &attn_weights.ffn,
                            hidden,
                            self.config.feed_forward_length,
                        );
                        Profiler::scope(
                            prof,
                            Some(layer_idx_u32),
                            OpKind::Layer,
                            attn_bytes + ffn_bytes,
                            attn_flops + ffn_flops,
                            || {
                                attention::attention_step(
                                    &self.kernels,
                                    &self.hybrid,
                                    &self.config,
                                    attn_weights,
                                    plane,
                                    max_seq,
                                    &mut self.scratch,
                                    pos,
                                    None,
                                    None,
                                )?;
                                ffn::ffn_step(
                                    &self.kernels,
                                    &attn_weights.ffn,
                                    &attn_weights.post_attention_norm,
                                    hidden,
                                    self.config.feed_forward_length,
                                    self.config.rms_eps,
                                    &mut self.scratch,
                                    None,
                                    None,
                                )
                            },
                        )?;
                    } else {
                        attention::attention_step(
                            &self.kernels,
                            &self.hybrid,
                            &self.config,
                            attn_weights,
                            plane,
                            max_seq,
                            &mut self.scratch,
                            pos,
                            prof,
                            Some(layer_idx_u32),
                        )?;
                        ffn::ffn_step(
                            &self.kernels,
                            &attn_weights.ffn,
                            &attn_weights.post_attention_norm,
                            hidden,
                            self.config.feed_forward_length,
                            self.config.rms_eps,
                            &mut self.scratch,
                            prof,
                            Some(layer_idx_u32),
                        )?;
                    }
                }
                _ => {
                    return Err(RocmlError::Config(format!(
                        "layer {layer_idx}: weight/kind mismatch (internal bug)"
                    )))
                }
            }
        }

        Profiler::scope(
            prof,
            None,
            OpKind::Norm,
            profile::norm_bytes(1, hidden),
            profile::norm_flops(1, hidden),
            || {
                self.kernels.rmsnorm(
                    offset(&self.scratch.x, 0),
                    offset(&self.weights.output_norm, 0),
                    offset(&self.scratch.xn, 0),
                    1,
                    hidden,
                    self.config.rms_eps,
                )
            },
        )?;
        let lm_head_bytes = profile::matvec_bytes(
            self.weights.output.byte_size(),
            self.config.vocab_size,
            hidden,
        );
        let lm_head_flops = profile::matvec_flops(self.config.vocab_size, hidden);
        Profiler::scope(
            prof,
            None,
            OpKind::LmHead,
            lm_head_bytes,
            lm_head_flops,
            || {
                self.weights.output.matvec(
                    &self.kernels,
                    offset(&self.scratch.xn, 0),
                    offset(&self.scratch.logits, 0),
                    self.config.vocab_size,
                    hidden,
                )
            },
        )?;

        let mut logits = vec![0.0f32; self.config.vocab_size as usize];
        self.scratch.logits.copy_to_host(&mut logits)?;

        self.pos += 1;
        Ok(logits)
    }
}
