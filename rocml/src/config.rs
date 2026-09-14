//! Model hyperparameters read from GGUF metadata for `general.architecture
//! = "qwen3"` (the dense Qwen3 family — not the qwen3.5 hybrid/MoE arch).
//! Every field is read from the file, never hardcoded, so this same loader
//! works for any dense Qwen3 size, not just 0.6B.

use rocml_core::gguf::GgufFile;

use crate::error::RocmlError;

const ARCH: &str = "qwen3";

/// Gated-FFN elementwise activation function (issue #16's config-driven-
/// activation seam). GGUF carries no generic metadata key for this —
/// llama.cpp itself infers it from `general.architecture`, not a per-model
/// field — so `ModelConfig::from_gguf` sets it from the (currently single)
/// `ARCH` this loader accepts rather than reading it off the file; a future
/// dense family whose FFN activation differs (e.g. gemma's `gelu_pytorch_
/// tanh`) would add its own arm here rather than this becoming a "keep
/// guessing SiLU" default that quietly miscompiles a new architecture's FFN.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// `silu(gate) * up` — Qwen3's SwiGLU FFN.
    SiLu,
    /// `gelu_tanh(gate) * up` — not selected by any architecture this
    /// loader accepts yet; landed ahead of an actual gemma-family loader
    /// (see `.claude/CLAUDE.md`'s multiarch audit) so the kernel/dispatch
    /// seam exists and is unit-tested before it's needed.
    Gelu,
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub block_count: u32,
    /// Hidden/residual-stream width (`n_embd`).
    pub embedding_length: u32,
    /// FFN intermediate width.
    pub feed_forward_length: u32,
    /// Number of query heads.
    pub head_count: u32,
    /// Number of key/value heads (GQA; may be less than `head_count`).
    pub head_count_kv: u32,
    /// Per-head dimension (`attention.key_length`; also used for V).
    pub head_dim: u32,
    pub rope_freq_base: f32,
    pub rms_eps: f32,
    pub context_length: u32,
    /// Read from `token_embd.weight`'s shape (ne[1]), not a metadata key —
    /// GGUF doesn't carry vocab size directly, and the embedding table's own
    /// shape is the ground truth the tokenizer's vocab must agree with.
    pub vocab_size: u32,
    /// Gated-FFN activation — see [`Activation`]'s doc comment for why this
    /// is set from `ARCH`, not read off the GGUF.
    pub activation: Activation,
}

impl ModelConfig {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, RocmlError> {
        let arch = gguf.get_str("general.architecture")?;
        if arch != ARCH {
            return Err(RocmlError::UnsupportedArchitecture {
                found: arch.to_string(),
            });
        }

        let block_count = gguf.get_u32("qwen3.block_count")?;
        let embedding_length = gguf.get_u32("qwen3.embedding_length")?;
        let feed_forward_length = gguf.get_u32("qwen3.feed_forward_length")?;
        let head_count = gguf.get_u32("qwen3.attention.head_count")?;
        let head_count_kv = gguf.get_u32("qwen3.attention.head_count_kv")?;
        let head_dim = gguf.get_u32("qwen3.attention.key_length")?;
        let rope_freq_base = gguf.get_f32("qwen3.rope.freq_base")?;
        let rms_eps = gguf.get_f32("qwen3.attention.layer_norm_rms_epsilon")?;
        let context_length = gguf.get_u32("qwen3.context_length")?;

        let embd_shape = gguf.tensor("token_embd.weight")?.shape().to_vec();
        let &vocab_ne = embd_shape.get(1).ok_or_else(|| {
            RocmlError::Config(
                "token_embd.weight must be 2D ([hidden, vocab] in ne order)".to_string(),
            )
        })?;
        let vocab_size = u32::try_from(vocab_ne)
            .map_err(|_| RocmlError::Config(format!("vocab size {vocab_ne} overflows u32")))?;

        if block_count == 0 || embedding_length == 0 || feed_forward_length == 0 {
            return Err(RocmlError::Config(
                "block_count/embedding_length/feed_forward_length must be nonzero".to_string(),
            ));
        }
        if head_count == 0 || head_count_kv == 0 || head_dim == 0 {
            return Err(RocmlError::Config(
                "head_count/head_count_kv/head_dim must be nonzero".to_string(),
            ));
        }
        if !head_count.is_multiple_of(head_count_kv) {
            return Err(RocmlError::Config(format!(
                "head_count {head_count} is not a multiple of head_count_kv {head_count_kv} \
                 (required for the GQA head mapping)"
            )));
        }
        if head_count_kv * head_dim == 0 || embedding_length == 0 {
            return Err(RocmlError::Config(
                "degenerate zero-sized dimension".to_string(),
            ));
        }
        if !head_dim.is_multiple_of(2) {
            return Err(RocmlError::Config(format!(
                "head_dim {head_dim} must be even (required by rope_neox_f32)"
            )));
        }
        if vocab_size == 0 {
            return Err(RocmlError::Config(
                "token_embd.weight reports a zero vocab size".to_string(),
            ));
        }

        Ok(Self {
            block_count,
            embedding_length,
            feed_forward_length,
            head_count,
            head_count_kv,
            head_dim,
            rope_freq_base,
            rms_eps,
            context_length,
            vocab_size,
            // The only architecture this loader accepts (`ARCH` == "qwen3")
            // uses SwiGLU — see `Activation`'s doc comment.
            activation: Activation::SiLu,
        })
    }

    /// Number of query heads sharing each KV head (GQA group size).
    pub fn kv_group_size(&self) -> u32 {
        self.head_count / self.head_count_kv
    }

    pub fn q_dim(&self) -> u32 {
        self.head_count * self.head_dim
    }

    pub fn kv_dim(&self) -> u32 {
        self.head_count_kv * self.head_dim
    }
}
