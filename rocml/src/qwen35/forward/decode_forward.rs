//! `Model::forward_token_profiled`: the single-token decode-step forward
//! pass (embedding, every layer's GDN-or-full-attention step, final
//! norm+lm-head). Split out of `mod.rs` purely for the 400-line file cap,
//! mirroring `chunk_forward.rs`'s own `impl Model` split for the batched
//! prefill path.

use super::kernels_moe::MoeKernels;
use super::moe_scratch::MoeScratch;
use super::scratch::Scratch;
use super::{attention, ffn, gdn, moe};
use super::{LayerKind, LayerWeights, Model};
use crate::error::RocmlError;
use crate::forward::kernels::{offset, Kernels};
use crate::profile::{self, OpKind, Phase, Profiler};
use crate::qwen35::config::MoeConfig;
use crate::qwen35::weights::Ffn;
use rocml_core::gguf::GgufFile;
use rocml_hip::DeviceBuffer;

/// Dispatches one layer's FFN step to the dense SwiGLU path or the
/// qwen35moe mixture-of-experts path — see `moe::moe_ffn_step`'s module doc
/// for why the MoE case processes only `x_row` (this decode-style forward
/// pass's single token) regardless of `Ffn` variant. `gguf`/`moe_cfg`/
/// `moe_scratch` are `None` only when every layer's `Ffn` is `Dense`, an
/// invariant `Model::load` maintains (see its `gguf`/`moe_scratch` fields'
/// doc comments) — an `Ffn::Moe` layer without them is an internal bug, not
/// a reachable user-facing error.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ffn_dispatch(
    kernels: &Kernels,
    moe_kernels: &MoeKernels,
    gguf: Option<&GgufFile>,
    moe_cfg: Option<&MoeConfig>,
    ffn_weights: &Ffn,
    post_attention_norm: &DeviceBuffer<f32>,
    hidden: u32,
    dense_ffn_dim: u32,
    rms_eps: f32,
    scratch: &mut Scratch,
    moe_scratch: Option<&mut MoeScratch>,
    prof: Option<&Profiler>,
    layer_idx: Option<u32>,
) -> Result<(), RocmlError> {
    match ffn_weights {
        Ffn::Dense(w) => ffn::ffn_step(
            kernels,
            w,
            post_attention_norm,
            hidden,
            dense_ffn_dim,
            rms_eps,
            scratch,
            prof,
            layer_idx,
        ),
        Ffn::Moe(w) => {
            let gguf =
                gguf.ok_or_else(|| RocmlError::Config("moe ffn layer with no gguf handle".into()))?;
            let moe_cfg = moe_cfg
                .ok_or_else(|| RocmlError::Config("moe ffn layer with no moe config".into()))?;
            let moe_scratch = moe_scratch
                .ok_or_else(|| RocmlError::Config("moe ffn layer with no moe scratch".into()))?;
            let x_row = offset(&scratch.x, 0);
            moe::moe_ffn_step(
                kernels,
                moe_kernels,
                gguf,
                w,
                moe_cfg,
                post_attention_norm,
                hidden,
                rms_eps,
                x_row,
                moe_scratch,
            )
        }
    }
}

pub(crate) fn ffn_cost_dispatch(
    ffn_weights: &Ffn,
    moe_cfg: Option<&MoeConfig>,
    hidden: u32,
    dense_ffn_dim: u32,
) -> (u64, u64) {
    match ffn_weights {
        Ffn::Dense(w) => ffn::ffn_step_cost(w, hidden, dense_ffn_dim),
        Ffn::Moe(w) => match moe_cfg {
            Some(cfg) => moe::moe_ffn_step_cost(w, cfg, hidden),
            None => (0, 0),
        },
    }
}

impl Model {
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
                        let (ffn_bytes, ffn_flops) = ffn_cost_dispatch(
                            &gdn_weights.ffn,
                            self.config.moe.as_ref(),
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
                                ffn_dispatch(
                                    &self.kernels,
                                    &self.moe_kernels,
                                    self.gguf.as_ref(),
                                    self.config.moe.as_ref(),
                                    &gdn_weights.ffn,
                                    &gdn_weights.post_attention_norm,
                                    hidden,
                                    self.config.feed_forward_length,
                                    self.config.rms_eps,
                                    &mut self.scratch,
                                    self.moe_scratch.as_mut(),
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
                        ffn_dispatch(
                            &self.kernels,
                            &self.moe_kernels,
                            self.gguf.as_ref(),
                            self.config.moe.as_ref(),
                            &gdn_weights.ffn,
                            &gdn_weights.post_attention_norm,
                            hidden,
                            self.config.feed_forward_length,
                            self.config.rms_eps,
                            &mut self.scratch,
                            self.moe_scratch.as_mut(),
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
                        let (ffn_bytes, ffn_flops) = ffn_cost_dispatch(
                            &attn_weights.ffn,
                            self.config.moe.as_ref(),
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
                                    &self.mixed_kernels,
                                    &self.config,
                                    attn_weights,
                                    plane,
                                    max_seq,
                                    &mut self.scratch,
                                    pos,
                                    None,
                                    None,
                                )?;
                                ffn_dispatch(
                                    &self.kernels,
                                    &self.moe_kernels,
                                    self.gguf.as_ref(),
                                    self.config.moe.as_ref(),
                                    &attn_weights.ffn,
                                    &attn_weights.post_attention_norm,
                                    hidden,
                                    self.config.feed_forward_length,
                                    self.config.rms_eps,
                                    &mut self.scratch,
                                    self.moe_scratch.as_mut(),
                                    None,
                                    None,
                                )
                            },
                        )?;
                    } else {
                        attention::attention_step(
                            &self.kernels,
                            &self.hybrid,
                            &self.mixed_kernels,
                            &self.config,
                            attn_weights,
                            plane,
                            max_seq,
                            &mut self.scratch,
                            pos,
                            prof,
                            Some(layer_idx_u32),
                        )?;
                        ffn_dispatch(
                            &self.kernels,
                            &self.moe_kernels,
                            self.gguf.as_ref(),
                            self.config.moe.as_ref(),
                            &attn_weights.ffn,
                            &attn_weights.post_attention_norm,
                            hidden,
                            self.config.feed_forward_length,
                            self.config.rms_eps,
                            &mut self.scratch,
                            self.moe_scratch.as_mut(),
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
