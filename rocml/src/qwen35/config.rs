//! Model hyperparameters read from GGUF metadata for `general.architecture =
//! "qwen35"` (the Qwen 3.5 hybrid Gated-Delta-Net + full-attention arch) and
//! `"qwen35moe"` (Ornith-1.5-35B-A3B: the identical hybrid GDN/full-attention
//! body, but every layer's dense SwiGLU FFN is replaced by a
//! softmax-routed mixture of experts plus one always-on shared expert — see
//! [`MoeConfig`] and `crate::qwen35::weights::moe`). Every field is read
//! from the file, matching llama.cpp's `qwen35`/`qwen35moe` GGUF layout
//! (verified against a real Qwen3.5-2B-Q8_0.gguf and against Crane's
//! independent implementation of the dense-FFN case; the MoE fields were
//! verified against a real Ornith-1.5-35B-A3B-GGUF header).
//!
//! GGUF's metadata-key namespace prefix matches `general.architecture`
//! exactly (`"qwen35.block_count"` vs. `"qwen35moe.block_count"` — the two
//! architectures do **not** share one key namespace despite sharing every
//! other convention), so every key read below is built from the actual
//! `arch` string rather than a hardcoded `"qwen35"` prefix. Per-layer
//! *tensor* names (`blk.N.*`) are unaffected by this — both architectures
//! use identical tensor names for the parts they share.

use rocml_core::gguf::{GgufError, GgufFile};

use crate::error::RocmlError;

const ARCH_HYBRID: &str = "qwen35";
const ARCH_MOE: &str = "qwen35moe";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    LinearAttention,
    FullAttention,
}

#[derive(Debug, Clone)]
pub struct Qwen35Config {
    /// Forward-pass layers only — excludes trailing MTP blocks, see
    /// [`main_block_count`].
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
    /// `Some` for `general.architecture = "qwen35moe"`: every layer's FFN is
    /// a mixture of experts instead of one dense SwiGLU MLP. `None` for
    /// plain `"qwen35"`.
    pub moe: Option<MoeConfig>,
}

/// Mixture-of-experts routing/sizing, read from `qwen35moe.expert_*`
/// metadata. Routing semantics (verified against llama.cpp's
/// `build_moe_ffn`/`qwen35moe.cpp`, not guessed): softmax over
/// `expert_count` logits, select the top `expert_used_count`, renormalize
/// their weights to sum to 1 (sum clamped to [`MOE_WEIGHT_SUM_EPS`]),
/// `SiLU(gate) * up -> down` per selected expert, plus one always-on shared
/// expert gated by `sigmoid(ffn_gate_inp_shexp . x)` — see
/// `crate::qwen35::weights::moe` for the loader and
/// `crate::qwen35::forward::moe` for the forward pass.
#[derive(Debug, Clone, Copy)]
pub struct MoeConfig {
    pub expert_count: u32,
    pub expert_used_count: u32,
    /// `qwen35moe.expert_feed_forward_length` — one routed expert's FFN
    /// width (512 on Ornith-1.5-35B-A3B, far smaller than a dense model's
    /// `feed_forward_length`).
    pub expert_ff_len: u32,
    /// `qwen35moe.expert_shared_feed_forward_length` — the shared expert's
    /// own FFN width (independent of `expert_ff_len`, though equal on this
    /// checkpoint).
    pub shared_ff_len: u32,
}

/// llama.cpp's `build_moe_ffn` floor on the renormalized top-k weight sum —
/// guards a pathological near-all-zero softmax from blowing up the
/// per-expert weights when dividing by it.
pub const MOE_WEIGHT_SUM_EPS: f32 = 6.1e-5;

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
        let is_moe = match arch {
            ARCH_HYBRID => false,
            ARCH_MOE => true,
            other => {
                return Err(RocmlError::UnsupportedArchitecture {
                    found: other.to_string(),
                })
            }
        };
        // GGUF's metadata-key namespace matches `general.architecture`
        // exactly for this family — see the module doc.
        let ns = arch;

        let nextn_layers = match gguf.get_u32(&format!("{ns}.nextn_predict_layers")) {
            Ok(n) => n,
            Err(GgufError::MissingKey(_)) => 0,
            Err(e) => return Err(e.into()),
        };
        let block_count =
            main_block_count(gguf.get_u32(&format!("{ns}.block_count"))?, nextn_layers)?;
        let embedding_length = gguf.get_u32(&format!("{ns}.embedding_length"))?;
        let head_count = gguf.get_u32(&format!("{ns}.attention.head_count"))?;
        let head_count_kv = gguf.get_u32(&format!("{ns}.attention.head_count_kv"))?;
        let head_dim = gguf.get_u32(&format!("{ns}.attention.key_length"))?;
        let rope_freq_base = gguf.get_f32(&format!("{ns}.rope.freq_base"))?;
        let rope_dim_count = gguf.get_u32(&format!("{ns}.rope.dimension_count"))?;
        let rms_eps = gguf.get_f32(&format!("{ns}.attention.layer_norm_rms_epsilon"))?;
        let context_length = gguf.get_u32(&format!("{ns}.context_length"))?;

        // `qwen35moe` has no `feed_forward_length` key at all (there is no
        // dense FFN — every layer is MoE); `feed_forward_length` is then
        // only used to size the dense-FFN scratch buffers `Scratch`/
        // `ChunkScratch` allocate unconditionally, which stay unused for an
        // all-MoE model — `expert_ff_len` is a harmless, always-valid
        // placeholder value for that dead allocation (see `MoeConfig`'s own
        // fields for the real per-expert width).
        let moe = if is_moe {
            Some(MoeConfig {
                expert_count: gguf.get_u32(&format!("{ns}.expert_count"))?,
                expert_used_count: gguf.get_u32(&format!("{ns}.expert_used_count"))?,
                expert_ff_len: gguf.get_u32(&format!("{ns}.expert_feed_forward_length"))?,
                shared_ff_len: gguf.get_u32(&format!("{ns}.expert_shared_feed_forward_length"))?,
            })
        } else {
            None
        };
        let feed_forward_length = match moe {
            Some(m) => m.expert_ff_len,
            None => gguf.get_u32(&format!("{ns}.feed_forward_length"))?,
        };
        if let Some(m) = moe {
            if m.expert_count == 0 || m.expert_ff_len == 0 || m.shared_ff_len == 0 {
                return Err(RocmlError::Config(
                    "expert_count/expert_feed_forward_length/expert_shared_feed_forward_length \
                     must be nonzero"
                        .to_string(),
                ));
            }
            if m.expert_used_count == 0 || m.expert_used_count > m.expert_count {
                return Err(RocmlError::Config(format!(
                    "expert_used_count {} must be nonzero and at most expert_count {}",
                    m.expert_used_count, m.expert_count
                )));
            }
        }

        let embd_shape = gguf.tensor("token_embd.weight")?.shape().to_vec();
        let &vocab_ne = embd_shape.get(1).ok_or_else(|| {
            RocmlError::Config(
                "token_embd.weight must be 2D ([hidden, vocab] in ne order)".to_string(),
            )
        })?;
        let vocab_size = u32::try_from(vocab_ne)
            .map_err(|_| RocmlError::Config(format!("vocab size {vocab_ne} overflows u32")))?;

        let conv_kernel = gguf.get_u32(&format!("{ns}.ssm.conv_kernel"))?;
        let num_k_heads = gguf.get_u32(&format!("{ns}.ssm.group_count"))?;
        let num_v_heads = gguf.get_u32(&format!("{ns}.ssm.time_step_rank"))?;
        let inner_size = gguf.get_u32(&format!("{ns}.ssm.inner_size"))?;
        let head_k_dim = gguf.get_u32(&format!("{ns}.ssm.state_size"))?;

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
            moe,
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

/// `qwen35.block_count` counts every `blk.N` in the file, including trailing
/// multi-token-prediction draft blocks (`nextn_predict_layers`, e.g.
/// Ornith-1.5-9B's `blk.32`), which llama.cpp's `n_layer()` also excludes.
/// An MTP block carries ordinary attention/FFN tensors and no `ssm_a`, so
/// counting it would load cleanly and silently run the draft head as an
/// extra full-attention layer.
fn main_block_count(block_count: u32, nextn_layers: u32) -> Result<u32, RocmlError> {
    block_count
        .checked_sub(nextn_layers)
        .filter(|&n| n > 0)
        .ok_or_else(|| {
            RocmlError::Config(format!(
                "nextn_predict_layers {nextn_layers} leaves no forward-pass layers out of \
                 block_count {block_count}"
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::main_block_count;

    #[test]
    fn no_mtp_blocks_keeps_every_layer() {
        assert_eq!(main_block_count(32, 0).unwrap(), 32);
    }

    #[test]
    fn trailing_mtp_block_is_excluded() {
        assert_eq!(main_block_count(33, 1).unwrap(), 32);
    }

    #[test]
    fn mtp_blocks_covering_every_layer_are_rejected() {
        assert!(main_block_count(1, 1).is_err());
        assert!(main_block_count(1, 2).is_err());
    }
}
