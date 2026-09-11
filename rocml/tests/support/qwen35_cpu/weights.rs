//! Plain f32 weight loading — no f16 cast, unlike the GPU path — since this
//! is a from-scratch understanding check, not a numerics-matching one.

use rocml::qwen35::config::{GdnConfig, LayerKind, Qwen35Config};
use rocml_core::gguf::GgufFile;
use rocml_core::quant::dequantize;

fn load_matrix(gguf: &GgufFile, name: &str, m: u32, n: u32) -> Vec<f32> {
    let view = gguf
        .tensor(name)
        .unwrap_or_else(|e| panic!("tensor {name}: {e}"));
    let shape = view.shape();
    assert_eq!(
        shape,
        [n as u64, m as u64],
        "tensor {name}: unexpected shape {shape:?}"
    );
    dequantize(view.dtype(), view.data()).unwrap_or_else(|e| panic!("dequant {name}: {e}"))
}

fn load_vector(gguf: &GgufFile, name: &str, len: u32) -> Vec<f32> {
    let view = gguf
        .tensor(name)
        .unwrap_or_else(|e| panic!("tensor {name}: {e}"));
    assert_eq!(
        view.shape(),
        [len as u64],
        "tensor {name}: unexpected shape {:?}",
        view.shape()
    );
    dequantize(view.dtype(), view.data()).unwrap_or_else(|e| panic!("dequant {name}: {e}"))
}

pub struct FfnWeights {
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub down: Vec<f32>,
}

impl FfnWeights {
    fn load(gguf: &GgufFile, p: &str, hidden: u32, ffn: u32) -> Self {
        Self {
            gate: load_matrix(gguf, &format!("{p}.ffn_gate.weight"), ffn, hidden),
            up: load_matrix(gguf, &format!("{p}.ffn_up.weight"), ffn, hidden),
            down: load_matrix(gguf, &format!("{p}.ffn_down.weight"), hidden, ffn),
        }
    }
}

pub struct GdnLayerWeights {
    pub attn_norm: Vec<f32>,
    pub attn_qkv: Vec<f32>,
    pub attn_gate: Vec<f32>,
    pub ssm_beta: Vec<f32>,
    pub ssm_alpha: Vec<f32>,
    /// `[conv_dim, kernel]` row-major, oldest tap first.
    pub ssm_conv1d: Vec<f32>,
    pub ssm_dt_bias: Vec<f32>,
    pub ssm_a: Vec<f32>,
    pub ssm_norm: Vec<f32>,
    pub ssm_out: Vec<f32>,
    pub post_attention_norm: Vec<f32>,
    pub ffn: FfnWeights,
}

impl GdnLayerWeights {
    fn load(gguf: &GgufFile, cfg: &Qwen35Config, i: u32) -> Self {
        let p = format!("blk.{i}");
        let hidden = cfg.embedding_length;
        let gdn: &GdnConfig = &cfg.gdn;
        Self {
            attn_norm: load_vector(gguf, &format!("{p}.attn_norm.weight"), hidden),
            attn_qkv: load_matrix(gguf, &format!("{p}.attn_qkv.weight"), gdn.conv_dim, hidden),
            attn_gate: load_matrix(
                gguf,
                &format!("{p}.attn_gate.weight"),
                gdn.value_dim,
                hidden,
            ),
            ssm_beta: load_matrix(
                gguf,
                &format!("{p}.ssm_beta.weight"),
                gdn.num_v_heads,
                hidden,
            ),
            ssm_alpha: load_matrix(
                gguf,
                &format!("{p}.ssm_alpha.weight"),
                gdn.num_v_heads,
                hidden,
            ),
            ssm_conv1d: load_matrix(
                gguf,
                &format!("{p}.ssm_conv1d.weight"),
                gdn.conv_dim,
                gdn.conv_kernel,
            ),
            ssm_dt_bias: load_vector(gguf, &format!("{p}.ssm_dt.bias"), gdn.num_v_heads),
            ssm_a: load_vector(gguf, &format!("{p}.ssm_a"), gdn.num_v_heads),
            ssm_norm: load_vector(gguf, &format!("{p}.ssm_norm.weight"), gdn.head_v_dim),
            ssm_out: load_matrix(gguf, &format!("{p}.ssm_out.weight"), hidden, gdn.value_dim),
            post_attention_norm: load_vector(
                gguf,
                &format!("{p}.post_attention_norm.weight"),
                hidden,
            ),
            ffn: FfnWeights::load(gguf, &p, hidden, cfg.feed_forward_length),
        }
    }
}

pub struct AttnLayerWeights {
    pub attn_norm: Vec<f32>,
    pub attn_q: Vec<f32>,
    pub attn_q_norm: Vec<f32>,
    pub attn_k: Vec<f32>,
    pub attn_k_norm: Vec<f32>,
    pub attn_v: Vec<f32>,
    pub attn_output: Vec<f32>,
    pub post_attention_norm: Vec<f32>,
    pub ffn: FfnWeights,
    pub has_output_gate: bool,
}

impl AttnLayerWeights {
    fn load(gguf: &GgufFile, cfg: &Qwen35Config, i: u32) -> Self {
        let p = format!("blk.{i}");
        let hidden = cfg.embedding_length;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let q_name = format!("{p}.attn_q.weight");
        let q_rows = gguf.tensor(&q_name).unwrap().shape()[1];
        let has_output_gate = q_rows == 2 * q_dim as u64;
        let q_out = if has_output_gate { 2 * q_dim } else { q_dim };
        Self {
            attn_norm: load_vector(gguf, &format!("{p}.attn_norm.weight"), hidden),
            attn_q: load_matrix(gguf, &q_name, q_out, hidden),
            attn_q_norm: load_vector(gguf, &format!("{p}.attn_q_norm.weight"), cfg.head_dim),
            attn_k: load_matrix(gguf, &format!("{p}.attn_k.weight"), kv_dim, hidden),
            attn_k_norm: load_vector(gguf, &format!("{p}.attn_k_norm.weight"), cfg.head_dim),
            attn_v: load_matrix(gguf, &format!("{p}.attn_v.weight"), kv_dim, hidden),
            attn_output: load_matrix(gguf, &format!("{p}.attn_output.weight"), hidden, q_dim),
            post_attention_norm: load_vector(
                gguf,
                &format!("{p}.post_attention_norm.weight"),
                hidden,
            ),
            ffn: FfnWeights::load(gguf, &p, hidden, cfg.feed_forward_length),
            has_output_gate,
        }
    }
}

pub enum LayerWeights {
    Gdn(GdnLayerWeights),
    Attention(AttnLayerWeights),
}

pub struct ModelWeights {
    pub token_embd: Vec<f32>,
    pub output_norm: Vec<f32>,
    pub output: Vec<f32>,
    pub layers: Vec<LayerWeights>,
}

impl ModelWeights {
    pub fn load(gguf: &GgufFile, cfg: &Qwen35Config) -> Self {
        let token_embd = load_matrix(
            gguf,
            "token_embd.weight",
            cfg.vocab_size,
            cfg.embedding_length,
        );
        let output_norm = load_vector(gguf, "output_norm.weight", cfg.embedding_length);
        let output = if gguf.tensor("output.weight").is_ok() {
            load_matrix(gguf, "output.weight", cfg.vocab_size, cfg.embedding_length)
        } else {
            token_embd.clone()
        };
        let layers = (0..cfg.block_count)
            .map(|i| match cfg.layer_kinds[i as usize] {
                LayerKind::LinearAttention => {
                    LayerWeights::Gdn(GdnLayerWeights::load(gguf, cfg, i))
                }
                LayerKind::FullAttention => {
                    LayerWeights::Attention(AttnLayerWeights::load(gguf, cfg, i))
                }
            })
            .collect();
        Self {
            token_embd,
            output_norm,
            output,
            layers,
        }
    }
}
