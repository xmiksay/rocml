//! Issue #17's standalone SmoothQuant-style measurement — step 2 of the
//! issue's plan, run *before* any integration: for the worst offender
//! (`ssm_out`'s input, `gdn_y_silu`) and one moderate one (`ffn_down`'s
//! input, `ffn_gate_silu`), at `alpha` in `{0.5, 0.65, 0.8}`, measures (a)
//! activation flatness before/after `X/s`, (b) the weight requant/fold
//! error `W*s` picks up going through Q4_K (both tensors are Q4_K on the
//! layers this test picks — see `mmq_calibrate.rs`'s module doc), and (c)
//! single-matmul output error `int8(X/s)·W'` vs. the f32 reference,
//! compared against the already-known unsmoothed int8 error. Per the
//! issue's own instruction: if the combined error doesn't drop decisively
//! (>3x) at any alpha, that's a negative result to report, not a bug to
//! chase — this test only measures and prints, it asserts nothing.
//!
//! **Result, worked out and then confirmed empirically below: this design
//! cannot work, for a structural reason the issue's own brief asked to be
//! verified first.** Folding `s` into a K-quant sub-block's scale requires
//! `s` to be *constant* across the activation quantizer's own 32-element
//! block (the brief's own constraint). But `quantize_act_q8_blk`'s int8
//! codes are `round(x / (amax(x)/127))` — already normalized by that same
//! block's own amax. Dividing every element of a block by one shared
//! positive constant `s` scales `amax` and every element by the same
//! factor, so `amax(x/s)/127` shrinks by exactly `1/s` and every quantized
//! *code* comes out bit-for-bit identical to the unsmoothed case
//! (`round((x/s)/(amax(x)/(127*s))) == round(x/(amax(x)/127))`). The
//! activation-side quantization error is therefore mathematically invariant
//! under a per-block-constant `s` — SmoothQuant's actual mechanism (moving
//! outlier magnitude *between channels within a block* so the quantizer's
//! shared per-block scale no longer starves the small channels) requires
//! `s` to vary *within* the 32-element block, which is exactly what
//! block-constant foldability rules out. The only thing a block-constant
//! `s` can change is the *weight* side, where it can only ever add error
//! (fold/requant noise) since the activation error it was meant to offset
//! never moves. Measured below: `flat_p50`/`flat_p90` after smoothing are
//! identical to before at every alpha (confirming the invariance directly,
//! not just its downstream consequence), and every smoothed single-matmul
//! output error is *worse* than the unsmoothed baseline, not better.
//!
//! CPU-only (no GPU/HIP): real GGUF weight bytes are dequantized on the
//! host via `rocml_core::quant::dequantize`, and the int8 activation
//! quantizer / Q4_K fold+requant are pure-Rust references
//! (`tests/support/smoothquant/`). Needs `make mmq-calibrate` to have run
//! first (reads its sidecar files); skips itself if they're absent.
//! `#[ignore]`d, run via `make mmq-smoothquant-measure`.

mod support;

use rocml::qwen35::forward::layer_capture::{CalibrationDump, LayerDump};
use rocml_core::gguf::GgufFile;
use rocml_core::quant::{dequantize, GgmlDType};
use rocml_core::testpaths::checkpoint;
use support::smoothquant::matmul::matmul_xwt_omajor;
use support::smoothquant::paths::{calibration_path, raw_dump_path};
use support::smoothquant::q4k::{fold_naive, requant, Q4kSuperblock, Q4K_BLOCK_BYTES, SUPERBLOCK};
use support::smoothquant::quantizer::{dequant_row_i8, quantize_row_i8};
use support::smoothquant::scale::{block_scale, per_block_max, weight_channel_amax};
use support::smoothquant::stats::{percentile, pooled_block_outlier_ratios, rel_error};

const GGUF_REL: &str = "Ornith-1.0-9B-GGUF/ornith-1.0-9b-Q4_K_M.gguf";
const ALPHAS: &[f32] = &[0.5, 0.65, 0.8];
const REL_EPS: f32 = 1e-3;
/// Below this improvement factor (unsmoothed mean-rel-error / smoothed
/// mean-rel-error), the issue's own brief says to stop and report a
/// negative rather than proceed to integration.
const DECISIVE_FACTOR: f32 = 3.0;

struct TensorSpec {
    label: &'static str,
    calib_key: &'static str,
    weight_name: &'static str,
}

/// Both layers picked because their tensor is Q4_K (see `mmq_calibrate.rs`'s
/// module doc for why: `ssm_out` is Q4_K on every GDN layer; `ffn_down` is a
/// per-layer Q4_K/Q6_K mix and layer 4 lands on the Q4_K side) — one fold/
/// requant implementation covers both.
const SPECS: &[TensorSpec] = &[
    TensorSpec {
        label: "ssm_out (worst offender)",
        calib_key: "0:gdn_y_silu",
        weight_name: "blk.0.ssm_out.weight",
    },
    TensorSpec {
        label: "ffn_down (moderate)",
        calib_key: "4:ffn_gate_silu",
        weight_name: "blk.4.ffn_down.weight",
    },
];

#[test]
#[ignore]
fn smoothquant_standalone_measurement() {
    let Some(gguf_path) = checkpoint(GGUF_REL) else {
        eprintln!("skipping: {GGUF_REL} not found");
        return;
    };
    let Ok(calib) = CalibrationDump::load(calibration_path()) else {
        eprintln!(
            "skipping: run `make mmq-calibrate` first (missing {})",
            calibration_path().display()
        );
        return;
    };
    let Ok(raw) = LayerDump::load(raw_dump_path()) else {
        eprintln!(
            "skipping: run `make mmq-calibrate` first (missing {})",
            raw_dump_path().display()
        );
        return;
    };
    let gguf = GgufFile::open(&gguf_path).expect("open gguf");

    for spec in SPECS {
        run_one(&gguf, &calib, &raw, spec);
    }
}

fn quantize_matrix_i8(x: &[f32], rows: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * n];
    for r in 0..rows {
        let row = &x[r * n..(r + 1) * n];
        let (codes, scale) = quantize_row_i8(row);
        let deq = dequant_row_i8(&codes, &scale);
        out[r * n..(r + 1) * n].copy_from_slice(&deq);
    }
    out
}

/// Folds/requantizes every Q4_K superblock of a `[m, n]` weight matrix's raw
/// bytes given a per-32-channel-block SmoothQuant scale (`s.len() ==
/// n/32`), returning `(fold_reconstructed, requant_reconstructed,
/// fold_rel_error, requant_rel_error)` — the last two pooled over every
/// element against the ideal `original*s` target.
#[allow(clippy::type_complexity)]
fn fold_and_requant_weight(
    w_bytes: &[u8],
    m: usize,
    n: usize,
    s: &[f32],
) -> (Vec<f32>, Vec<f32>, (f32, f32), (f32, f32)) {
    let row_bytes = (n / SUPERBLOCK) * Q4K_BLOCK_BYTES;
    let mut w_fold = vec![0f32; m * n];
    let mut w_requant = vec![0f32; m * n];
    let mut targets = Vec::with_capacity(m * n);
    for o in 0..m {
        let row = &w_bytes[o * row_bytes..(o + 1) * row_bytes];
        for (sbk, block) in row.chunks_exact(Q4K_BLOCK_BYTES).enumerate() {
            let sb = Q4kSuperblock::parse(block);
            let orig = sb.dequant();
            let s_local: [f32; 8] = s[sbk * 8..sbk * 8 + 8].try_into().expect("8 sub-blocks");
            let mut target = [0f32; SUPERBLOCK];
            for i in 0..SUPERBLOCK {
                target[i] = orig[i] * s_local[i / 32];
            }
            let folded = fold_naive(&sb, &s_local);
            let requantized = requant(&target);

            let base = o * n + sbk * SUPERBLOCK;
            w_fold[base..base + SUPERBLOCK].copy_from_slice(&folded);
            w_requant[base..base + SUPERBLOCK].copy_from_slice(&requantized);
            targets.extend_from_slice(&target);
        }
    }
    let fold_err = rel_error(&w_fold, &targets, REL_EPS);
    let requant_err = rel_error(&w_requant, &targets, REL_EPS);
    (w_fold, w_requant, fold_err, requant_err)
}

fn run_one(gguf: &GgufFile, calib: &CalibrationDump, raw: &LayerDump, spec: &TensorSpec) {
    eprintln!("\n=== {} ===", spec.label);
    let x_tensor = raw
        .tensors
        .get(spec.calib_key)
        .unwrap_or_else(|| panic!("missing raw capture for {}", spec.calib_key));
    let rows = x_tensor.rows as usize;
    let n = x_tensor.cols as usize;
    let x = &x_tensor.values;

    let view = gguf.tensor(spec.weight_name).expect("tensor not found");
    assert!(
        matches!(view.dtype(), GgmlDType::Q4_K),
        "{} expected Q4_K, got {:?}",
        spec.weight_name,
        view.dtype()
    );
    let shape = view.shape();
    let (n_ggml, m) = (shape[0] as usize, shape[1] as usize);
    assert_eq!(n_ggml, n, "input dim mismatch between capture and weight");
    let w_bytes = view.data();
    let w = dequantize(GgmlDType::Q4_K, w_bytes).expect("cpu dequant");
    assert_eq!(w.len(), m * n);

    let x_channel_amax = x_tensor.channel_amax().amax;
    let calib_amax = &calib
        .channels
        .get(spec.calib_key)
        .expect("calibration entry")
        .amax;
    let calib_drift = x_channel_amax
        .iter()
        .zip(calib_amax.iter())
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        calib_drift < 1e-3,
        "calibration sidecar drifted from raw dump (different runs?): max diff {calib_drift}"
    );

    let w_channel_amax = weight_channel_amax(&w, m, n);
    let x_blk = per_block_max(&x_channel_amax);
    let w_blk = per_block_max(&w_channel_amax);
    let n_blocks = n / 32;
    assert_eq!(x_blk.len(), n_blocks);

    eprintln!("shape: rows={rows} n={n} m={m} ({n_blocks} 32-blocks/row)");

    let y_ref = matmul_xwt_omajor(x, rows, n, &w, m);
    let x_unsmoothed_q = quantize_matrix_i8(x, rows, n);
    let y_unsmoothed = matmul_xwt_omajor(&x_unsmoothed_q, rows, n, &w, m);
    let (base_max_rel, base_mean_rel) = rel_error(&y_unsmoothed, &y_ref, REL_EPS);
    eprintln!("unsmoothed int8 baseline: max_rel={base_max_rel:.4} mean_rel={base_mean_rel:.4}");

    let before_ratios = pooled_block_outlier_ratios(x, rows, n);
    let before_p50 = percentile(&before_ratios, 0.5);
    let before_p90 = percentile(&before_ratios, 0.9);
    eprintln!("activation flatness BEFORE smoothing: p50={before_p50:.2} p90={before_p90:.2}");
    eprintln!(
        "  NOTE: per-32-block-*constant* s cannot change this ratio at all — amax(x/s)/mean(x/s) \
         == amax(x)/mean(x) for any positive constant s shared by the whole block. The \"after\" \
         column below is expected to equal the \"before\" line above exactly; this is the \
         mathematical crux of this round's negative result (see the module doc)."
    );

    eprintln!(
        "{:>6}  {:>13}  {:>13}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>8}  {:>8}",
        "alpha",
        "flat_p50(aft)",
        "flat_p90(aft)",
        "fold_max",
        "fold_mean",
        "rq_max",
        "rq_mean",
        "out_max(fold)",
        "out_mean(fold)",
        "x_fold",
        "x_rq"
    );
    for &alpha in ALPHAS {
        let s: Vec<f32> = x_blk
            .iter()
            .zip(&w_blk)
            .map(|(&xb, &wb)| block_scale(xb, wb, alpha))
            .collect();

        let mut x_smoothed = vec![0f32; rows * n];
        for r in 0..rows {
            for j in 0..n {
                x_smoothed[r * n + j] = x[r * n + j] / s[j / 32];
            }
        }
        let after_ratios = pooled_block_outlier_ratios(&x_smoothed, rows, n);
        let flat_p50 = percentile(&after_ratios, 0.5);
        let flat_p90 = percentile(&after_ratios, 0.9);

        let (w_fold, w_requant, fold_err, requant_err) = fold_and_requant_weight(w_bytes, m, n, &s);

        let x_smoothed_q = quantize_matrix_i8(&x_smoothed, rows, n);
        let y_fold = matmul_xwt_omajor(&x_smoothed_q, rows, n, &w_fold, m);
        let y_requant = matmul_xwt_omajor(&x_smoothed_q, rows, n, &w_requant, m);
        let (out_max_fold, out_mean_fold) = rel_error(&y_fold, &y_ref, REL_EPS);
        let (out_max_rq, out_mean_rq) = rel_error(&y_requant, &y_ref, REL_EPS);

        let improve_fold = base_mean_rel / out_mean_fold.max(1e-9);
        let improve_rq = base_mean_rel / out_mean_rq.max(1e-9);

        eprintln!(
            "{alpha:>6.2}  {flat_p50:>8.2}  {flat_p90:>8.2}  {:>10.4}  {:>10.4}  {:>10.4}  {:>10.4}  {out_max_fold:>10.4}  {out_mean_fold:>10.4}  {improve_fold:>8.2}x  {improve_rq:>8.2}x",
            fold_err.0, fold_err.1, requant_err.0, requant_err.1,
        );
        eprintln!(
            "         (requant-weight output: out_max={out_max_rq:.4} out_mean={out_mean_rq:.4})"
        );
        let verdict = if improve_fold >= DECISIVE_FACTOR || improve_rq >= DECISIVE_FACTOR {
            "DECISIVE"
        } else {
            "not decisive"
        };
        eprintln!(
            "  alpha={alpha:.2}: fold {improve_fold:.2}x, requant {improve_rq:.2}x -> {verdict}"
        );
    }
}
