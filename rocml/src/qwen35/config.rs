//! Model hyperparameters read from GGUF metadata for `general.architecture =
//! "qwen35"` (the Qwen 3.5 hybrid Gated-Delta-Net + full-attention arch).
//! Every field is read from the file, matching llama.cpp's `qwen35` GGUF
//! layout (verified against a real Qwen3.5-2B-Q8_0.gguf and against Crane's
//! independent implementation of the same architecture).

use rocml_core::gguf::GgufFile;

use crate::error::RocmlError;

const ARCH: &str = "qwen35";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    LinearAttention,
    FullAttention,
}

#[derive(Debug, Clone)]
pub struct Qwen35Config {
    pub block_count: u32,
    pub embedding_length: u32,
    pub feed_forward_length: u32,
    pub head_count: u32,
    pub head_count_kv: u32,
    /// Full-attention per-head Q/K dim (`attention.key_length`; also V's).
    pub head_dim: u32,
    pub rope_freq_base: f32,
    /// Partial-rotary width: only the first `rope_dim_count` of `head_dim`
    /// components are rotated (`rope.dimension_count`, 64 for the 2B model
    /// vs. `head_dim` = 256 — a quarter, matching HF's
    /// `partial_rotary_factor = 0.25`).
    pub rope_dim_count: u32,
    pub rms_eps: f32,
    pub context_length: u32,
    pub vocab_size: u32,
    /// Which of `block_count` layers are linear-attention (GDN) vs.
    /// full-attention, indexed by layer.
    pub layer_kinds: Vec<LayerKind>,
    pub gdn: GdnConfig,
}

/// Gated Delta Net dimensions, derived from `qwen35.ssm.*` metadata the same
/// way Crane's `GdnDims::new` does (see `crate::ops::gdn::config` there).
#[derive(Debug, Clone, Copy)]
pub struct GdnConfig {
    pub conv_kernel: u32,
    pub num_k_heads: u32,
    pub num_v_heads: u32,
    pub head_k_dim: u32,
    pub head_v_dim: u32,
    pub key_dim: u32,
    pub value_dim: u32,
    pub conv_dim: u32,
}

impl Qwen35Config {
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
        let head_count = gguf.get_u32("qwen35.attention.head_count")?;
        let head_count_kv = gguf.get_u32("qwen35.attention.head_count_kv")?;
        let head_dim = gguf.get_u32("qwen35.attention.key_length")?;
        let rope_freq_base = gguf.get_f32("qwen35.rope.freq_base")?;
        let rope_dim_count = gguf.get_u32("qwen35.rope.dimension_count")?;
        let rms_eps = gguf.get_f32("qwen35.attention.layer_norm_rms_epsilon")?;
        let context_length = gguf.get_u32("qwen35.context_length")?;

        let embd_shape = gguf.tensor("token_embd.weight")?.shape().to_vec();
        let &vocab_ne = embd_shape.get(1).ok_or_else(|| {
            RocmlError::Config(
                "token_embd.weight must be 2D ([hidden, vocab] in ne order)".to_string(),
            )
        })?;
        let vocab_size = u32::try_from(vocab_ne)
            .map_err(|_| RocmlError::Config(format!("vocab size {vocab_ne} overflows u32")))?;

        let conv_kernel = gguf.get_u32("qwen35.ssm.conv_kernel")?;
        let num_k_heads = gguf.get_u32("qwen35.ssm.group_count")?;
        let num_v_heads = gguf.get_u32("qwen35.ssm.time_step_rank")?;
        let inner_size = gguf.get_u32("qwen35.ssm.inner_size")?;
        let head_k_dim = gguf.get_u32("qwen35.ssm.state_size")?;

        if num_k_heads == 0 || num_v_heads == 0 || inner_size == 0 {
            return Err(RocmlError::Config(
                "ssm.group_count/time_step_rank/inner_size must be nonzero".to_string(),
            ));
        }
        if !inner_size.is_multiple_of(num_v_heads) {
            return Err(RocmlError::Config(format!(
                "ssm.inner_size {inner_size} is not a multiple of ssm.time_step_rank {num_v_heads}"
            )));
        }
        // GDN's GQA-style key/value head grouping (`num_v_heads >
        // num_k_heads`, used by Qwen3.5/Qwen3-Next sizes above 2B, e.g.
        // Ornith-1.0-9B's 16 key heads / 32 value heads): `gdn_recurrence_decode_f32`
        // broadcasts key head `h % num_k_heads` to value head `h` — a
        // *tiled* pattern, not HF transformers' raw `repeat_interleave`,
        // because llama.cpp's GGUF converter already permutes every
        // GDN value-head-indexed tensor (V, Z, beta, alpha, A_log, dt_bias,
        // conv1d's V channels, out_proj's input columns) into tiled order
        // at conversion time — see that kernel's own doc comment.
        if !num_v_heads.is_multiple_of(num_k_heads) {
            return Err(RocmlError::Config(format!(
                "ssm.time_step_rank {num_v_heads} is not a multiple of ssm.group_count \
                 {num_k_heads} (required for GDN's grouped query/key broadcast)"
            )));
        }
        let head_v_dim = inner_size / num_v_heads;
        let key_dim = num_k_heads * head_k_dim;
        let value_dim = num_v_heads * head_v_dim;
        let conv_dim = key_dim * 2 + value_dim;

        let layer_kinds: Vec<LayerKind> = (0..block_count)
            .map(|i| {
                if gguf.tensor(&format!("blk.{i}.ssm_a")).is_ok() {
                    LayerKind::LinearAttention
                } else {
                    LayerKind::FullAttention
                }
            })
            .collect();

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
        if rope_dim_count == 0 || rope_dim_count > head_dim || !rope_dim_count.is_multiple_of(2) {
            return Err(RocmlError::Config(format!(
                "rope.dimension_count {rope_dim_count} must be even and at most head_dim \
                 {head_dim}"
            )));
        }
        if vocab_size == 0 {
            return Err(RocmlError::Config(
                "token_embd.weight reports a zero vocab size".to_string(),
            ));
        }
        if !layer_kinds.contains(&LayerKind::LinearAttention) {
            return Err(RocmlError::Config(
                "no GDN (linear-attention) layers found in this qwen35 GGUF".to_string(),
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
            rope_dim_count,
            rms_eps,
            context_length,
            vocab_size,
            layer_kinds,
            gdn: GdnConfig {
                conv_kernel,
                num_k_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                key_dim,
                value_dim,
                conv_dim,
            },
        })
    }

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
