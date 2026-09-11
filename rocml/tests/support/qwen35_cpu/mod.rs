//! Pure-Rust f32 CPU reference for the qwen35 hybrid forward pass —
//! architecture validation independent of the HIP kernels (see the
//! milestone report's Part 1). Slow (naive nested loops, no BLAS); meant
//! for a handful of tokens, not a real decode loop.

mod attention;
mod gdn;
mod math;
mod weights;

use math::{add_inplace, matvec, rmsnorm_rows};
use rocml::qwen35::config::{LayerKind, Qwen35Config};
use rocml_core::gguf::GgufFile;
use weights::{LayerWeights, ModelWeights};

enum LayerState {
    Gdn(gdn::GdnState),
    Attention(attention::AttnState),
}

pub struct CpuModel {
    cfg: Qwen35Config,
    weights: ModelWeights,
    states: Vec<LayerState>,
    pos: u32,
}

impl CpuModel {
    pub fn load(gguf_path: &str) -> Self {
        let gguf = GgufFile::open(gguf_path).expect("open GGUF");
        let cfg = Qwen35Config::from_gguf(&gguf).expect("parse qwen35 config");
        let weights = ModelWeights::load(&gguf, &cfg);
        let states = cfg
            .layer_kinds
            .iter()
            .map(|&kind| match kind {
                LayerKind::LinearAttention => LayerState::Gdn(gdn::GdnState::new(&cfg)),
                LayerKind::FullAttention => {
                    LayerState::Attention(attention::AttnState::new(cfg.head_count_kv as usize))
                }
            })
            .collect();
        Self {
            cfg,
            weights,
            states,
            pos: 0,
        }
    }

    pub fn forward_token(&mut self, token_id: u32) -> Vec<f32> {
        let hidden = self.cfg.embedding_length as usize;
        let mut x = self.weights.token_embd
            [token_id as usize * hidden..(token_id as usize + 1) * hidden]
            .to_vec();

        for (layer, state) in self.weights.layers.iter().zip(self.states.iter_mut()) {
            match (layer, state) {
                (LayerWeights::Gdn(w), LayerState::Gdn(s)) => {
                    gdn::step(&self.cfg, w, s, &mut x);
                    ffn_step(&self.cfg, &w.ffn, &w.post_attention_norm, &mut x);
                }
                (LayerWeights::Attention(w), LayerState::Attention(s)) => {
                    attention::step(&self.cfg, w, s, self.pos, &mut x);
                    ffn_step(&self.cfg, &w.ffn, &w.post_attention_norm, &mut x);
                }
                _ => unreachable!("layer/state kind mismatch"),
            }
        }

        rmsnorm_rows(
            &mut x,
            &self.weights.output_norm,
            1,
            hidden,
            self.cfg.rms_eps,
        );
        self.pos += 1;
        matvec(
            &self.weights.output,
            &x,
            self.cfg.vocab_size as usize,
            hidden,
        )
    }
}

fn ffn_step(cfg: &Qwen35Config, ffn: &weights::FfnWeights, norm: &[f32], x: &mut [f32]) {
    let hidden = cfg.embedding_length as usize;
    let dim = cfg.feed_forward_length as usize;
    let mut xn = x.to_vec();
    rmsnorm_rows(&mut xn, norm, 1, hidden, cfg.rms_eps);
    let gate = matvec(&ffn.gate, &xn, dim, hidden);
    let up = matvec(&ffn.up, &xn, dim, hidden);
    let hidden_act: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(&g, &u)| math::silu(g) * u)
        .collect();
    let out = matvec(&ffn.down, &hidden_act, hidden, dim);
    add_inplace(x, &out);
}
