//! Small pooled-statistic helpers shared by issue #17's calibration and
//! measurement tools: the per-32-block `amax/mean(|x|)` outlier ratio (the
//! yardstick the `mmq_precision` root-cause round reported p50/p90 of) and
//! a plain percentile-of-sorted-slice helper.

/// `p` in `[0.0, 1.0]`; `sorted` must already be ascending. Simple
/// nearest-rank percentile — adequate for reporting p50/p90 of a few
/// thousand pooled ratios, not a statistics library.
pub fn percentile(sorted: &[f32], p: f64) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Pools the per-32-block `amax / mean(|x|)` ratio of `values` (`rows` rows
/// of `cols` columns, row-major) across every row and every column-block,
/// skipping all-zero blocks (undefined ratio). Returns the ratios sorted
/// ascending, ready for [`percentile`].
pub fn pooled_block_outlier_ratios(values: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut ratios = Vec::new();
    for r in 0..rows {
        let row = &values[r * cols..(r + 1) * cols];
        for blk in row.chunks(32) {
            let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
            let mean = blk.iter().map(|v| v.abs()).sum::<f32>() / blk.len() as f32;
            if mean > 0.0 {
                ratios.push(amax / mean);
            }
        }
    }
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ratios
}

/// Relative-error summary (`max`, `mean`) of `actual` vs `expected`,
/// `|a-b| / max(|expected|, eps)` — the same shape of metric
/// `layer_capture::diff_dumps` and `mmq_endtoend_measure.rs` already use
/// elsewhere in this codebase for MMQ precision comparisons.
pub fn rel_error(actual: &[f32], expected: &[f32], eps: f32) -> (f32, f32) {
    let mut max_rel = 0f32;
    let mut sum_rel = 0f32;
    for (&a, &e) in actual.iter().zip(expected) {
        let rel = (a - e).abs() / e.abs().max(eps);
        max_rel = max_rel.max(rel);
        sum_rel += rel;
    }
    (max_rel, sum_rel / actual.len().max(1) as f32)
}
