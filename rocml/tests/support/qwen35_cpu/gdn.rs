//! GDN layer decode step, pure Rust — mirrors
//! `rocml::qwen35::forward::gdn::gdn_layer_step` and Crane's
//! `GatedDeltaNet::forward` for a single timestep.

use super::math::{add_inplace, l2norm_rows, matvec, rmsnorm_rows, sigmoid, silu, softplus};
use super::weights::GdnLayerWeights;
use rocml::qwen35::config::Qwen35Config;

const L2_NORM_EPS: f32 = 1e-6;

pub struct GdnState {
    /// `[conv_dim, kernel - 1]` row-major, oldest first.
    pub conv_state: Vec<f32>,
    /// `[num_v_heads, head_k_dim, head_v_dim]` row-major.
    pub recurrent: Vec<f32>,
}

impl GdnState {
    pub fn new(cfg: &Qwen35Config) -> Self {
        let gdn = &cfg.gdn;
        Self {
            conv_state: vec![0.0; gdn.conv_dim as usize * (gdn.conv_kernel as usize - 1)],
            recurrent: vec![
                0.0;
                gdn.num_v_heads as usize
                    * gdn.head_k_dim as usize
                    * gdn.head_v_dim as usize
            ],
        }
    }
}

// Chunked access into several differently-strided arrays keyed by the same
// channel index `c` — `chunks_exact`/`zip` would need as many parallel
// iterators as there are arrays, no clearer than the index form here.
#[allow(clippy::needless_range_loop)]
pub fn step(cfg: &Qwen35Config, w: &GdnLayerWeights, state: &mut GdnState, x: &mut [f32]) {
    let hidden = cfg.embedding_length as usize;
    let gdn = &cfg.gdn;
    let (conv_dim, key_dim, value_dim) = (
        gdn.conv_dim as usize,
        gdn.key_dim as usize,
        gdn.value_dim as usize,
    );
    let (num_heads, num_k_heads, hk, hv) = (
        gdn.num_v_heads as usize,
        gdn.num_k_heads as usize,
        gdn.head_k_dim as usize,
        gdn.head_v_dim as usize,
    );
    let kernel = gdn.conv_kernel as usize;

    let mut xn = x.to_vec();
    rmsnorm_rows(&mut xn, &w.attn_norm, 1, hidden, cfg.rms_eps);

    let qkv_raw = matvec(&w.attn_qkv, &xn, conv_dim, hidden);
    let z = matvec(&w.attn_gate, &xn, value_dim, hidden);
    let a_raw = matvec(&w.ssm_alpha, &xn, num_heads, hidden);
    let b_raw = matvec(&w.ssm_beta, &xn, num_heads, hidden);

    // Causal depthwise conv1d + SiLU, one decode step, updating conv_state.
    let hist_len = kernel - 1;
    let mut qkv = vec![0.0f32; conv_dim];
    for c in 0..conv_dim {
        let hist = &state.conv_state[c * hist_len..(c + 1) * hist_len];
        let wt = &w.ssm_conv1d[c * kernel..(c + 1) * kernel];
        let mut acc: f32 = hist.iter().zip(wt).map(|(&h, &wj)| h * wj).sum();
        acc += qkv_raw[c] * wt[hist_len];
        qkv[c] = silu(acc);
    }
    for c in 0..conv_dim {
        let hist = &mut state.conv_state[c * hist_len..(c + 1) * hist_len];
        for j in 0..hist_len.saturating_sub(1) {
            hist[j] = hist[j + 1];
        }
        if hist_len > 0 {
            hist[hist_len - 1] = qkv_raw[c];
        }
    }

    // Beta/decay gates.
    let beta: Vec<f32> = b_raw.iter().map(|&b| sigmoid(b)).collect();
    let g: Vec<f32> = (0..num_heads)
        .map(|h| w.ssm_a[h] * softplus(a_raw[h] + w.ssm_dt_bias[h]))
        .collect();

    // L2-norm Q/K per head; Q additionally carries the recurrence's
    // `1/sqrt(head_k_dim)` query scale. Q/K only have `num_k_heads` distinct
    // rows (grouped GDN, e.g. Ornith's 16 key heads for 32 value heads) —
    // normalizing over `num_heads` rows here would read past the end of `q`/
    // `k` whenever `num_k_heads < num_heads`.
    let mut q = qkv[0..key_dim].to_vec();
    let mut k = qkv[key_dim..2 * key_dim].to_vec();
    let v = &qkv[2 * key_dim..2 * key_dim + value_dim];
    l2norm_rows(
        &mut q,
        num_k_heads,
        hk,
        L2_NORM_EPS,
        1.0 / (hk as f32).sqrt(),
    );
    l2norm_rows(&mut k, num_k_heads, hk, L2_NORM_EPS, 1.0);

    // Gated delta rule recurrence, per value head `h`; its query/key come
    // from key head `h % num_k_heads` — a tiled broadcast, matching
    // `gdn_recurrence_decode_f32`'s own doc comment (llama.cpp's GGUF
    // converter already reorders every value-head-indexed GDN tensor into
    // that tiled order, so `%` is what lines Q/K back up with V/beta/g here).
    let mut y = vec![0.0f32; value_dim];
    for h in 0..num_heads {
        let key_head = h % num_k_heads;
        let s = &mut state.recurrent[h * hk * hv..(h + 1) * hk * hv];
        let decay = g[h].exp();
        for e in s.iter_mut() {
            *e *= decay;
        }
        let q_h = &q[key_head * hk..(key_head + 1) * hk];
        let k_h = &k[key_head * hk..(key_head + 1) * hk];
        let v_h = &v[h * hv..(h + 1) * hv];
        let mut delta = vec![0.0f32; hv];
        for (vi, d) in delta.iter_mut().enumerate() {
            let kv_mem: f32 = (0..hk).map(|ki| s[ki * hv + vi] * k_h[ki]).sum();
            *d = beta[h] * (v_h[vi] - kv_mem);
        }
        for ki in 0..hk {
            for vi in 0..hv {
                s[ki * hv + vi] += k_h[ki] * delta[vi];
            }
        }
        let y_h = &mut y[h * hv..(h + 1) * hv];
        for (vi, yv) in y_h.iter_mut().enumerate() {
            *yv = (0..hk).map(|ki| s[ki * hv + vi] * q_h[ki]).sum();
        }
    }

    rmsnorm_rows(&mut y, &w.ssm_norm, num_heads, hv, cfg.rms_eps);
    for (yi, &zi) in y.iter_mut().zip(&z) {
        *yi *= silu(zi);
    }

    let gdn_out = matvec(&w.ssm_out, &y, hidden, value_dim);
    add_inplace(x, &gdn_out);
}
