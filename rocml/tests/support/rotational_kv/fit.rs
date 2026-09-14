//! Calibration-selection helpers shared by `rotational_kv_calibrate.rs`
//! (builds and checks in the production sidecar) and
//! `rotational_kv_measure.rs` (the tensor-level quality-gate table) —
//! issue #14's "measure, don't inherit" instruction applied to both axes
//! the plan left open: pairing scheme (`Adjacent` vs `SplitHalf`) and angle
//! source (per-pair calibrated vs a fixed 45-degree baseline).

use std::collections::BTreeMap;

use rocml::kv_quant::rotational::{
    calibrate_angles, fixed_angles, forward_rotate, l2_norm, LloydMaxCodebook, Pairing,
    TensorSidecar,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AngleSource {
    Calibrated,
    Fixed45,
}

/// Relative RMSE of round-tripping every `head_dim`-length vector in `data`
/// (`n_vec` of them, row-major) through `ts` at `bpw`.
pub fn round_trip_rmse(
    data: &[f32],
    n_vec: usize,
    head_dim: usize,
    ts: &TensorSidecar,
    bpw: u8,
) -> f64 {
    let (mut sum_sq_err, mut sum_sq_orig) = (0f64, 0f64);
    for i in 0..n_vec {
        let v = &data[i * head_dim..(i + 1) * head_dim];
        let recon = ts
            .round_trip(v, bpw)
            .expect("round trip: shape must match the sidecar's calibrated head_dim");
        for (&o, &r) in v.iter().zip(&recon) {
            let e = (o - r) as f64;
            sum_sq_err += e * e;
            sum_sq_orig += (o as f64) * (o as f64);
        }
    }
    if sum_sq_orig <= 0.0 {
        0.0
    } else {
        (sum_sq_err / sum_sq_orig).sqrt()
    }
}

fn build_angles(
    pairing: Pairing,
    source: AngleSource,
    train: &[f32],
    n_train: usize,
    head_dim: usize,
) -> Vec<f32> {
    match source {
        AngleSource::Fixed45 => fixed_angles(head_dim),
        AngleSource::Calibrated => calibrate_angles(train, n_train, head_dim, pairing)
            .expect("calibrate_angles: train data shape must match head_dim"),
    }
}

/// Rotates + L2-normalizes every training vector, pools every resulting
/// coordinate across the *whole* vector (the codebook is shared/pooled —
/// see `LloydMaxCodebook`'s module doc for why that's the intended design,
/// not a simplification), and trains one codebook per requested bpw.
fn train_codebooks(
    train: &[f32],
    n_train: usize,
    head_dim: usize,
    pairing: Pairing,
    angles: &[f32],
    bpws: &[u8],
) -> BTreeMap<u8, LloydMaxCodebook> {
    let mut pooled = Vec::with_capacity(n_train * head_dim);
    for i in 0..n_train {
        let mut v = train[i * head_dim..(i + 1) * head_dim].to_vec();
        forward_rotate(&mut v, pairing, angles).expect("forward_rotate");
        let norm = half::f16::from_f32(l2_norm(&v)).to_f32();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        pooled.extend_from_slice(&v);
    }
    bpws.iter()
        .map(|&bpw| {
            let cb = LloydMaxCodebook::train(&pooled, 1usize << bpw).expect("Lloyd-Max train");
            (bpw, cb)
        })
        .collect()
}

pub fn build_tensor_sidecar(
    pairing: Pairing,
    angles: Vec<f32>,
    head_dim: usize,
    train: &[f32],
    n_train: usize,
    bpws: &[u8],
) -> TensorSidecar {
    let codebooks = train_codebooks(train, n_train, head_dim, pairing, &angles, bpws);
    TensorSidecar {
        head_dim,
        pairing,
        angles,
        codebooks,
    }
}

pub struct SearchResult {
    /// Every (pairing, angle source) combination tried, with its measured
    /// holdout relative RMSE at 3 bpw — printed in full by the calibration
    /// test so the choice is documented, not just asserted.
    pub candidates: Vec<(Pairing, AngleSource, f64)>,
    pub best_pairing: Pairing,
    pub best_source: AngleSource,
}

/// Tries every (pairing, angle-source) combination — training a 3-bpw-only
/// codebook from `train` and measuring relative RMSE against a disjoint
/// `holdout` slice — and returns the winner by lowest holdout RMSE, plus
/// every candidate's number for the report.
pub fn search_best_config(
    train: &[f32],
    n_train: usize,
    holdout: &[f32],
    n_holdout: usize,
    head_dim: usize,
) -> SearchResult {
    let mut candidates = Vec::new();
    let mut best: Option<(Pairing, AngleSource, f64)> = None;
    for pairing in [Pairing::Adjacent, Pairing::SplitHalf] {
        for source in [AngleSource::Calibrated, AngleSource::Fixed45] {
            let angles = build_angles(pairing, source, train, n_train, head_dim);
            let ts = build_tensor_sidecar(pairing, angles, head_dim, train, n_train, &[3]);
            let rmse = round_trip_rmse(holdout, n_holdout, head_dim, &ts, 3);
            candidates.push((pairing, source, rmse));
            if best.as_ref().is_none_or(|&(_, _, b)| rmse < b) {
                best = Some((pairing, source, rmse));
            }
        }
    }
    let (best_pairing, best_source, _) = best.expect("at least one candidate was tried");
    SearchResult {
        candidates,
        best_pairing,
        best_source,
    }
}
