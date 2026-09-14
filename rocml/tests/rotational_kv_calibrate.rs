//! Issue #14 phase 2's calibration harness: builds the rotational KV
//! quantization sidecar (`rocml::kv_quant::rotational::RotSidecar`) checked
//! in at `rocml/data/rotational_kv_calibration.json` — pairing scheme,
//! per-pair Givens angles, and per-bpw (2/3/4) Lloyd-Max codebooks, built
//! separately for K and V — from real K/V vectors captured off
//! Ornith-1.0-9B-Q4_K_M's fp16-KV decode path over `bench/eval/corpus.txt`
//! (`support::rotational_kv::capture`, reusing `kv_head_error_measure.rs`'s
//! issue-#2 snapshot-capture technique).
//!
//! Measures both axes issue #14 asks to "measure, don't inherit" —
//! `Adjacent` vs `SplitHalf` pairing, and per-pair calibrated vs a fixed
//! 45-degree angle — picking whichever combination minimizes round-trip
//! relative RMSE at 3 bpw on a held-out slice of the same capture,
//! independently for K and V. Prints every candidate's measured number so
//! the choice is documented, not just asserted (`--nocapture`).
//!
//! `#[ignore]`d, run via `make rotational-kv-calibrate`. Overwrites the
//! checked-in sidecar file; only meant to be re-run intentionally (e.g.
//! against a different checkpoint's `head_dim`, or with more calibration
//! data).

mod support;

use std::path::PathBuf;

use rocml::kv_quant::rotational::RotSidecar;
use support::rotational_kv::capture::capture_ornith_kv;
use support::rotational_kv::fit::{build_tensor_sidecar, round_trip_rmse, search_best_config};

/// Past `SINK_LEN(32)`, enough real decode tokens for a meaningful
/// train/holdout split across 6 mixed-eligible layers x 4 kv heads each
/// (~2600 vectors/split for K and V at this depth).
const DECODE_TOKENS: usize = 700;
/// Fraction of post-sink positions used for angle/codebook training; the
/// rest is held out purely for measuring round-trip error honestly.
const TRAIN_FRACTION: f64 = 0.8;

fn sidecar_out_path() -> PathBuf {
    PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/data/rotational_kv_calibration.json"
    ))
}

#[test]
#[ignore]
fn build_rotational_kv_calibration_sidecar() {
    let Some(capture) = capture_ornith_kv(DECODE_TOKENS) else {
        eprintln!("skipping build_rotational_kv_calibration_sidecar: checkpoint not found");
        return;
    };
    eprintln!(
        "captured {} mixed-eligible layers x {} kv heads, head_dim {}, filled {}",
        capture.layers.len(),
        capture.n_kv_heads,
        capture.head_dim,
        capture.filled,
    );

    let split_at =
        capture.sink_len + ((capture.filled - capture.sink_len) as f64 * TRAIN_FRACTION) as usize;
    eprintln!(
        "train positions [{}, {split_at}), holdout [{split_at}, {})",
        capture.sink_len, capture.filled
    );

    for (label, is_k) in [("K", true), ("V", false)] {
        eprintln!("\n=== {label}: pairing x angle-source search (3 bpw holdout RMSE) ===");
        let (train, n_train) = capture.flatten(is_k, capture.sink_len, split_at);
        let (holdout, n_holdout) = capture.flatten(is_k, split_at, capture.filled);

        let search = search_best_config(&train, n_train, &holdout, n_holdout, capture.head_dim);
        for (pairing, source, rmse) in &search.candidates {
            eprintln!("  {pairing:?} / {source:?}: holdout rel RMSE @3bpw = {rmse:.6}");
        }
        eprintln!(
            "  -> winner: {:?} / {:?}",
            search.best_pairing, search.best_source
        );

        let angles = match search.best_source {
            support::rotational_kv::fit::AngleSource::Fixed45 => {
                rocml::kv_quant::rotational::fixed_angles(capture.head_dim)
            }
            support::rotational_kv::fit::AngleSource::Calibrated => {
                rocml::kv_quant::rotational::calibrate_angles(
                    &train,
                    n_train,
                    capture.head_dim,
                    search.best_pairing,
                )
                .expect("calibrate_angles")
            }
        };
        let ts = build_tensor_sidecar(
            search.best_pairing,
            angles,
            capture.head_dim,
            &train,
            n_train,
            &[2, 3, 4],
        );
        for &bpw in &[2u8, 3, 4] {
            let rmse = round_trip_rmse(&holdout, n_holdout, capture.head_dim, &ts, bpw);
            eprintln!("  final sidecar holdout rel RMSE @{bpw}bpw = {rmse:.6}");
        }

        if is_k {
            write_sidecar_half(&ts, true);
        } else {
            write_sidecar_half(&ts, false);
        }
    }

    eprintln!("\nwrote {}", sidecar_out_path().display());
}

/// Assembles the final `RotSidecar` incrementally (K then V) by re-reading
/// whatever's already on disk for the half not being written this call —
/// simplest way to keep this test a single straight-line pass over K then
/// V without restructuring it around building both halves before any I/O.
fn write_sidecar_half(ts: &rocml::kv_quant::rotational::TensorSidecar, is_k: bool) {
    let path = sidecar_out_path();
    let existing = std::fs::read_to_string(&path).ok();
    let mut sidecar = existing
        .and_then(|s| RotSidecar::from_json(&s).ok())
        .unwrap_or_else(|| RotSidecar {
            k: ts.clone(),
            v: ts.clone(),
        });
    if is_k {
        sidecar.k = ts.clone();
    } else {
        sidecar.v = ts.clone();
    }
    let json = sidecar.to_json().expect("serialize sidecar");
    std::fs::write(&path, json).expect("write sidecar");
}
