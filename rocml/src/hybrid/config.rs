//! Model hyperparameters read from GGUF metadata for `general.architecture
//! = "qwen35"` — the hybrid Gated Delta Net (linear attention) + full
//! (softmax) attention architecture. Every field is read from the file,
//! never hardcoded, per the same convention as the dense `qwen3` config.

use rocml_core::gguf::GgufFile;

use crate::error::RocmlError;

const ARCH: &str = "qwen35";

#[derive(Debug, Clone)]
pub struct HybridConfig {
    pub block_count: u32,
    pub embedding_length: u32,
    pub feed_forward_length: u32,
    pub rms_eps: f32,
    pub context_length: u32,
    pub vocab_size: u32,

    /// Every `full_attention_interval`-th layer (1-indexed from the end of
    /// each group, i.e. 0-indexed layer `i` is full attention iff
    /// `(i + 1) % full_attention_interval == 0`) is full softmax attention;
    /// the rest are Gated Delta Net linear-attention layers.
    pub full_attention_interval: u32,

    /// Full-attention layer geometry.
    pub attn_head_count: u32,
    pub attn_head_count_kv: u32,
    /// Q/K per-head dimension (`attention.key_length`).
    pub attn_key_length: u32,
    /// V per-head dimension (`attention.value_length`).
    pub attn_value_length: u32,
    /// Partial-rotary width: only the first `rope_dim` of each `key_length`
    /// dims is rotated, the rest passes through unrotated.
    pub rope_dim: u32,
    pub rope_freq_base: f32,

    /// GDN-layer geometry (`qwen35.ssm.*`).
    pub ssm_conv_kernel: u32,
    pub ssm_group_count: u32,
    pub ssm_inner_size: u32,
    pub ssm_state_size: u32,
    pub ssm_time_step_rank: u32,
}

impl HybridConfig {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, RocmlError> {
        let arch = gguf.get_str("general.architecture")?;
        if arch != ARCH {
            return Err(RocmlError::UnsupportedArchitecture {
                found: arch.to_string(),
            });
        }

        let block_count = gguf.get_u32("qwen35.block_count")?;
        let embedding_length = gguf.get_u32("qwen35.embedding_length")?;
        let feed_forward_length = gguf.get_u32("qwen35.feed_forward_length")?;
        let rms_eps = gguf.get_f32("qwen35.attention.layer_norm_rms_epsilon")?;
        let context_length = gguf.get_u32("qwen35.context_length")?;
        let full_attention_interval = gguf.get_u32("qwen35.full_attention_interval")?;

        let attn_head_count = gguf.get_u32("qwen35.attention.head_count")?;
        let attn_head_count_kv = gguf.get_u32("qwen35.attention.head_count_kv")?;
        let attn_key_length = gguf.get_u32("qwen35.attention.key_length")?;
        let attn_value_length = gguf.get_u32("qwen35.attention.value_length")?;
        let rope_dim = gguf.get_u32("qwen35.rope.dimension_count")?;
        let rope_freq_base = gguf.get_f32("qwen35.rope.freq_base")?;

        let ssm_conv_kernel = gguf.get_u32("qwen35.ssm.conv_kernel")?;
        let ssm_group_count = gguf.get_u32("qwen35.ssm.group_count")?;
        let ssm_inner_size = gguf.get_u32("qwen35.ssm.inner_size")?;
        let ssm_state_size = gguf.get_u32("qwen35.ssm.state_size")?;
        let ssm_time_step_rank = gguf.get_u32("qwen35.ssm.time_step_rank")?;

        let embd_shape = gguf.tensor("token_embd.weight")?.shape().to_vec();
        let &vocab_ne = embd_shape.get(1).ok_or_else(|| {
            RocmlError::Config(
                "token_embd.weight must be 2D ([hidden, vocab] in ne order)".to_string(),
            )
        })?;
        let vocab_size = u32::try_from(vocab_ne)
            .map_err(|_| RocmlError::Config(format!("vocab size {vocab_ne} overflows u32")))?;

        let cfg = Self {
            block_count,
            embedding_length,
            feed_forward_length,
            rms_eps,
            context_length,
            vocab_size,
            full_attention_interval,
            attn_head_count,
            attn_head_count_kv,
            attn_key_length,
            attn_value_length,
            rope_dim,
            rope_freq_base,
            ssm_conv_kernel,
            ssm_group_count,
            ssm_inner_size,
            ssm_state_size,
            ssm_time_step_rank,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<(), RocmlError> {
        if self.block_count == 0 || self.embedding_length == 0 || self.feed_forward_length == 0 {
            return Err(RocmlError::Config(
                "block_count/embedding_length/feed_forward_length must be nonzero".to_string(),
            ));
        }
        if self.full_attention_interval == 0 {
            return Err(RocmlError::Config(
                "full_attention_interval must be nonzero".to_string(),
            ));
        }
        if self.attn_head_count == 0
            || self.attn_head_count_kv == 0
            || self.attn_key_length == 0
            || self.attn_value_length == 0
        {
            return Err(RocmlError::Config(
                "attention head_count/head_count_kv/key_length/value_length must be nonzero"
                    .to_string(),
            ));
        }
        if !self
            .attn_head_count
            .is_multiple_of(self.attn_head_count_kv)
        {
            return Err(RocmlError::Config(format!(
                "attn head_count {} is not a multiple of head_count_kv {} (required for GQA)",
                self.attn_head_count, self.attn_head_count_kv
            )));
        }
        if self.rope_dim == 0 || !self.rope_dim.is_multiple_of(2) {
            return Err(RocmlError::Config(format!(
                "rope_dim {} must be even and nonzero (partial-rotary NEOX pairing)",
                self.rope_dim
            )));
        }
        if self.rope_dim > self.attn_key_length {
            return Err(RocmlError::Config(format!(
                "rope_dim {} exceeds attn_key_length {} (partial rotary must fit in the head)",
                self.rope_dim, self.attn_key_length
            )));
        }
        if self.ssm_conv_kernel == 0
            || self.ssm_group_count == 0
            || self.ssm_inner_size == 0
            || self.ssm_state_size == 0
        {
            return Err(RocmlError::Config(
                "ssm conv_kernel/group_count/inner_size/state_size must be nonzero".to_string(),
            ));
        }
        if !self.ssm_inner_size.is_multiple_of(self.ssm_group_count) {
            return Err(RocmlError::Config(format!(
                "ssm inner_size {} is not a multiple of group_count {}",
                self.ssm_inner_size, self.ssm_group_count
            )));
        }
        if self.ssm_head_dim() != self.ssm_state_size {
            return Err(RocmlError::Config(format!(
                "ssm inner_size/group_count ({}) must equal state_size ({}) -- this loader \
                 assumes the per-head q/k/v width matches the recurrence state width",
                self.ssm_head_dim(),
                self.ssm_state_size
            )));
        }
        if self.vocab_size == 0 {
            return Err(RocmlError::Config(
                "token_embd.weight reports a zero vocab size".to_string(),
            ));
        }
        Ok(())
    }

    /// True iff 0-indexed layer `layer_idx` is a full (softmax) attention
    /// layer; false means it's a Gated Delta Net linear-attention layer.
    pub fn is_full_attention(&self, layer_idx: u32) -> bool {
        (layer_idx + 1).is_multiple_of(self.full_attention_interval)
    }

    /// Real (non-gated) query projection width for full-attention layers.
    /// The `attn_q` tensor itself is twice this (query concatenated with a
    /// fused output gate of the same width).
    pub fn attn_q_dim(&self) -> u32 {
        self.attn_head_count * self.attn_key_length
    }

    pub fn attn_k_dim(&self) -> u32 {
        self.attn_head_count_kv * self.attn_key_length
    }

    pub fn attn_v_dim(&self) -> u32 {
        self.attn_head_count_kv * self.attn_value_length
    }

    /// `attn_output`'s input width: the concatenated per-(query-head) value
    /// readout.
    pub fn attn_concat_dim(&self) -> u32 {
        self.attn_head_count * self.attn_value_length
    }

    pub fn attn_kv_group_size(&self) -> u32 {
        self.attn_head_count / self.attn_head_count_kv
    }

    /// Per-head width of the GDN q/k/v streams after the conv, equal to
    /// `ssm_state_size` (checked by `validate`).
    pub fn ssm_head_dim(&self) -> u32 {
        self.ssm_inner_size / self.ssm_group_count
    }

    /// Width of the `attn_qkv` fused GDN projection (`3 * inner_size`).
    pub fn ssm_qkv_dim(&self) -> u32 {
        3 * self.ssm_inner_size
    }
}
