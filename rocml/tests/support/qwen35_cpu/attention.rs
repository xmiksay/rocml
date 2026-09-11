//! Full-attention layer decode step, pure Rust — mirrors
//! `rocml::qwen35::forward::attention::attention_step` and Crane's
//! `FullAttention::forward` for a single timestep.

use super::math::{add_inplace, matvec, rmsnorm_rows, rope_partial, sigmoid, softmax_inplace};
use super::weights::AttnLayerWeights;
use rocml::qwen35::config::Qwen35Config;

pub struct AttnState {
    /// Growing per-kv-head cache: `k[kvh]` / `v[kvh]` are `[seen, head_dim]`.
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
}

impl AttnState {
    pub fn new(n_kv_heads: usize) -> Self {
        Self {
            k: vec![Vec::new(); n_kv_heads],
            v: vec![Vec::new(); n_kv_heads],
        }
    }
}

pub fn step(
    cfg: &Qwen35Config,
    w: &AttnLayerWeights,
    state: &mut AttnState,
    pos: u32,
    x: &mut [f32],
) {
    let hidden = cfg.embedding_length as usize;
    let head_dim = cfg.head_dim as usize;
    let n_heads = cfg.head_count as usize;
    let n_kv_heads = cfg.head_count_kv as usize;
    let group = n_heads / n_kv_heads;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;

    let mut xn = x.to_vec();
    rmsnorm_rows(&mut xn, &w.attn_norm, 1, hidden, cfg.rms_eps);

    let q_out = if w.has_output_gate { 2 * q_dim } else { q_dim };
    let q_raw = matvec(&w.attn_q, &xn, q_out, hidden);
    let mut k = matvec(&w.attn_k, &xn, kv_dim, hidden);
    let v = matvec(&w.attn_v, &xn, kv_dim, hidden);

    let stride = if w.has_output_gate {
        2 * head_dim
    } else {
        head_dim
    };
    let mut q = vec![0.0f32; q_dim];
    let mut gate = vec![0.0f32; q_dim];
    for h in 0..n_heads {
        q[h * head_dim..(h + 1) * head_dim]
            .copy_from_slice(&q_raw[h * stride..h * stride + head_dim]);
        if w.has_output_gate {
            gate[h * head_dim..(h + 1) * head_dim]
                .copy_from_slice(&q_raw[h * stride + head_dim..h * stride + 2 * head_dim]);
        }
    }

    rmsnorm_rows(&mut q, &w.attn_q_norm, n_heads, head_dim, cfg.rms_eps);
    rmsnorm_rows(&mut k, &w.attn_k_norm, n_kv_heads, head_dim, cfg.rms_eps);
    rope_partial(
        &mut q,
        n_heads,
        head_dim,
        cfg.rope_dim_count as usize,
        pos,
        cfg.rope_freq_base,
    );
    rope_partial(
        &mut k,
        n_kv_heads,
        head_dim,
        cfg.rope_dim_count as usize,
        pos,
        cfg.rope_freq_base,
    );

    for kvh in 0..n_kv_heads {
        state.k[kvh].extend_from_slice(&k[kvh * head_dim..(kvh + 1) * head_dim]);
        state.v[kvh].extend_from_slice(&v[kvh * head_dim..(kvh + 1) * head_dim]);
    }

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut attn_concat = vec![0.0f32; q_dim];
    for h in 0..n_heads {
        let kvh = h / group;
        let q_h = &q[h * head_dim..(h + 1) * head_dim];
        let cur_len = state.k[kvh].len() / head_dim;
        let mut scores = vec![0.0f32; cur_len];
        for (t, s) in scores.iter_mut().enumerate() {
            let k_t = &state.k[kvh][t * head_dim..(t + 1) * head_dim];
            *s = q_h.iter().zip(k_t).map(|(&a, &b)| a * b).sum::<f32>() * scale;
        }
        softmax_inplace(&mut scores);
        let out = &mut attn_concat[h * head_dim..(h + 1) * head_dim];
        for (t, &p) in scores.iter().enumerate() {
            let v_t = &state.v[kvh][t * head_dim..(t + 1) * head_dim];
            for (o, &vi) in out.iter_mut().zip(v_t) {
                *o += p * vi;
            }
        }
    }

    if w.has_output_gate {
        for (o, &g) in attn_concat.iter_mut().zip(&gate) {
            *o *= sigmoid(g);
        }
    }

    let attn_out = matvec(&w.attn_output, &attn_concat, hidden, q_dim);
    add_inplace(x, &attn_out);
}
