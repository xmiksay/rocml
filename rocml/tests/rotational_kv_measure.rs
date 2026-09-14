//! Issue #14 phase 2 step 2: the tensor-level quality table — relative RMSE
//! of round-tripped K/V vectors, and the attention-score/output
//! perturbation those errors actually cause (`support::rotational_kv::attn`
//! — "the metric that matters"), for the rotational encoding
//! (`rocml::kv_quant::rotational`, using the checked-in calibration
//! sidecar) at 2/3/4 bpw versus the CURRENT production scalar encoding
//! (`rocml::kv_quant::quant_math`: K per-channel Q8, V per-token Q8/Q4) —
//! on the same real, held-out K/V vectors the calibration test
//! (`rotational_kv_calibrate.rs`) never trained on.
//!
//! `#[ignore]`d, run via `make rotational-kv-measure`. Prints the full
//! table (`--nocapture`) rather than asserting a bound — this is Phase 2's
//! measurement input to the GO/STOP decision, not a correctness gate.

mod support;

use std::collections::BTreeMap;

use rocml::kv_quant::quant_math::{
    dequantize_k_per_channel, dequantize_v_per_token_q4, dequantize_v_per_token_q8,
    quantize_k_per_channel, quantize_v_per_token_q4, quantize_v_per_token_q8,
};
use rocml::kv_quant::rotational::{default_sidecar, TensorKind};
use support::rotational_kv::attn::{k_attn_score_rmse, v_attn_output_rel_rmse};
use support::rotational_kv::capture::capture_ornith_kv;
use support::rotational_kv::fit::round_trip_rmse;

const DECODE_TOKENS: usize = 700;
/// Must match `rotational_kv_calibrate.rs`'s own split so this test only
/// ever measures on the calibration sidecar's held-out slice.
const TRAIN_FRACTION: f64 = 0.8;
const N_QUERY_PROXIES: usize = 8;

/// Pooled-across-(layer,head) vec RMSE accumulator, plus a plain mean for
/// the attention-derived metric (one already-averaged-per-head RMSE per
/// sample — see `attn.rs`'s module doc for why a further pooled-sum_sq
/// isn't more principled here).
#[derive(Default)]
struct Row {
    sum_sq_err: f64,
    sum_sq_orig: f64,
    attn_sum: f64,
    attn_n: usize,
}

impl Row {
    fn add_vec(&mut self, orig: &[f32], recon: &[f32]) {
        for (&o, &r) in orig.iter().zip(recon) {
            let e = (o - r) as f64;
            self.sum_sq_err += e * e;
            self.sum_sq_orig += (o as f64) * (o as f64);
        }
    }

    fn add_attn(&mut self, rmse: f64) {
        self.attn_sum += rmse;
        self.attn_n += 1;
    }

    fn vec_rmse(&self) -> f64 {
        if self.sum_sq_orig <= 0.0 {
            0.0
        } else {
            (self.sum_sq_err / self.sum_sq_orig).sqrt()
        }
    }

    fn attn_rmse(&self) -> f64 {
        if self.attn_n == 0 {
            0.0
        } else {
            self.attn_sum / self.attn_n as f64
        }
    }
}

fn query_proxies(
    capture: &support::rotational_kv::capture::Capture,
    layer: &support::rotational_kv::capture::CapturedLayer,
    head: usize,
    train_start: usize,
    train_end: usize,
) -> Vec<f32> {
    let train = capture.head_slice(layer, true, head, train_start, train_end);
    let n_train = train_end - train_start;
    let stride = (n_train / N_QUERY_PROXIES).max(1);
    let mut out = Vec::with_capacity(N_QUERY_PROXIES * capture.head_dim);
    for i in 0..N_QUERY_PROXIES {
        let pos = (i * stride).min(n_train - 1);
        out.extend_from_slice(&train[pos * capture.head_dim..(pos + 1) * capture.head_dim]);
    }
    out
}

#[test]
#[ignore]
fn rotational_vs_scalar_tensor_level_quality_table() {
    let Some(capture) = capture_ornith_kv(DECODE_TOKENS) else {
        eprintln!("skipping rotational_vs_scalar_tensor_level_quality_table: checkpoint not found");
        return;
    };
    let sidecar = default_sidecar().expect("checked-in rotational KV calibration sidecar");
    assert_eq!(
        sidecar.k.head_dim, capture.head_dim,
        "checked-in sidecar's head_dim must match this checkpoint's — re-run \
         `make rotational-kv-calibrate` if a different checkpoint is under test"
    );

    let head_dim = capture.head_dim;
    let train_start = capture.sink_len;
    let split_at =
        capture.sink_len + ((capture.filled - capture.sink_len) as f64 * TRAIN_FRACTION) as usize;
    let (holdout_start, holdout_end) = (split_at, capture.filled);
    let n_holdout = holdout_end - holdout_start;

    // K rows: scalar q8 + rotational 2/3/4 bpw.
    let mut k_rows: BTreeMap<String, Row> = BTreeMap::new();
    // V rows: scalar q8, scalar q4 + rotational 2/3/4 bpw.
    let mut v_rows: BTreeMap<String, Row> = BTreeMap::new();

    for layer in &capture.layers {
        for h in 0..capture.n_kv_heads {
            let k_true = capture.head_slice(layer, true, h, holdout_start, holdout_end);
            let v_true = capture.head_slice(layer, false, h, holdout_start, holdout_end);
            let queries = query_proxies(&capture, layer, h, train_start, split_at);

            // --- scalar K (q8 per-channel) ---
            let (kc, ks) = quantize_k_per_channel(k_true, 1, n_holdout, head_dim);
            let k_q8 = dequantize_k_per_channel(&kc, &ks, 1, n_holdout, head_dim);
            let row = k_rows.entry("scalar-q8".to_string()).or_default();
            row.add_vec(k_true, &k_q8);
            row.add_attn(k_attn_score_rmse(
                &queries,
                N_QUERY_PROXIES,
                k_true,
                &k_q8,
                n_holdout,
                head_dim,
            ));

            // --- scalar V (q8, q4 per-token) ---
            let (v8c, v8s) = quantize_v_per_token_q8(v_true, 1, n_holdout, head_dim);
            let v_q8 = dequantize_v_per_token_q8(&v8c, &v8s, 1, n_holdout, head_dim);
            let row = v_rows.entry("scalar-q8".to_string()).or_default();
            row.add_vec(v_true, &v_q8);
            row.add_attn(v_attn_output_rel_rmse(
                &queries,
                N_QUERY_PROXIES,
                k_true,
                n_holdout,
                head_dim,
                v_true,
                &v_q8,
            ));

            let (v4c, v4s) = quantize_v_per_token_q4(v_true, 1, n_holdout, head_dim);
            let v_q4 = dequantize_v_per_token_q4(&v4c, &v4s, 1, n_holdout, head_dim);
            let row = v_rows.entry("scalar-q4".to_string()).or_default();
            row.add_vec(v_true, &v_q4);
            row.add_attn(v_attn_output_rel_rmse(
                &queries,
                N_QUERY_PROXIES,
                k_true,
                n_holdout,
                head_dim,
                v_true,
                &v_q4,
            ));

            // --- rotational K/V at 2/3/4 bpw ---
            for &bpw in &[2u8, 3, 4] {
                let mut k_rot = vec![0f32; k_true.len()];
                for i in 0..n_holdout {
                    let v = &k_true[i * head_dim..(i + 1) * head_dim];
                    let r = sidecar
                        .for_kind(TensorKind::K)
                        .round_trip(v, bpw)
                        .expect("K round trip");
                    k_rot[i * head_dim..(i + 1) * head_dim].copy_from_slice(&r);
                }
                let row = k_rows.entry(format!("rotational-{bpw}bpw")).or_default();
                row.add_vec(k_true, &k_rot);
                row.add_attn(k_attn_score_rmse(
                    &queries,
                    N_QUERY_PROXIES,
                    k_true,
                    &k_rot,
                    n_holdout,
                    head_dim,
                ));

                let mut v_rot = vec![0f32; v_true.len()];
                for i in 0..n_holdout {
                    let v = &v_true[i * head_dim..(i + 1) * head_dim];
                    let r = sidecar
                        .for_kind(TensorKind::V)
                        .round_trip(v, bpw)
                        .expect("V round trip");
                    v_rot[i * head_dim..(i + 1) * head_dim].copy_from_slice(&r);
                }
                let row = v_rows.entry(format!("rotational-{bpw}bpw")).or_default();
                row.add_vec(v_true, &v_rot);
                row.add_attn(v_attn_output_rel_rmse(
                    &queries,
                    N_QUERY_PROXIES,
                    k_true,
                    n_holdout,
                    head_dim,
                    v_true,
                    &v_rot,
                ));
            }
        }
    }

    eprintln!(
        "\n=== tensor-level quality table ({} layers x {} heads, {n_holdout} held-out positions) ===",
        capture.layers.len(),
        capture.n_kv_heads
    );
    eprintln!("tensor  encoding          vec RMSE   attn-metric RMSE");
    for (name, row) in &k_rows {
        eprintln!(
            "K       {name:<16}  {:.6}   {:.6}  (softmax-weight RMSE)",
            row.vec_rmse(),
            row.attn_rmse()
        );
    }
    for (name, row) in &v_rows {
        eprintln!(
            "V       {name:<16}  {:.6}   {:.6}  (attn-output rel RMSE)",
            row.vec_rmse(),
            row.attn_rmse()
        );
    }

    // Sanity, not a quality gate: round_trip_rmse (used by the calibration
    // search) must agree with this test's own pooled vec RMSE computation
    // for the same encoding/data, or one of the two has a bug.
    let (holdout_k, n) = capture.flatten(true, holdout_start, holdout_end);
    let cross_check = round_trip_rmse(&holdout_k, n, head_dim, sidecar.for_kind(TensorKind::K), 3);
    let table_value = k_rows["rotational-3bpw"].vec_rmse();
    assert!(
        (cross_check - table_value).abs() < 1e-6,
        "cross-check mismatch: {cross_check} vs {table_value}"
    );
}
