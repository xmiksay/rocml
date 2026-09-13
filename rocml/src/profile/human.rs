//! Human-readable rendering of a [`super::report::Report`] — the table
//! `rocml-cli bench --profile`/`generate --profile` prints — plus the two
//! "so what should I look at" summaries (`top_bottlenecks` by absolute time,
//! `top_worklist` by wasted time) `Report::build` calls to fill in
//! `Report::bottlenecks`/`Report::worklist`. Split out of `report.rs` to
//! keep that file under the 400-line cap; the aggregation math it renders
//! lives there.

use super::report::{AggRow, PhaseReport, Report};

/// Human-readable table, printed by `rocml-cli bench --profile`/
/// `generate --profile` after the run.
pub(super) fn to_human(report: &Report) -> String {
    let mut out = String::new();
    for phase in &report.phases {
        out.push_str(&format!(
            "\n== {} phase: {:.2} ms profiled ==\n",
            phase.phase, phase.total_elapsed_ms
        ));
        out.push_str("-- by op-kind --\n");
        append_table(&mut out, &phase.by_op);
        out.push_str("-- by layer --\n");
        append_table(&mut out, &phase.by_layer);
    }
    if !report.bottlenecks.is_empty() {
        out.push_str("\n== top bottlenecks ==\n");
        for line in &report.bottlenecks {
            out.push_str(line);
            out.push('\n');
        }
    }
    if !report.worklist.is_empty() {
        out.push_str("\n== optimization worklist (by wasted time) ==\n");
        for line in &report.worklist {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

fn append_table(out: &mut String, rows: &[AggRow]) {
    out.push_str(&format!(
        "{:<28} {:>6} {:>10} {:>10} {:>9} {:>9} {:>7} {:>7} {:>10}\n",
        "label", "n", "ms", "%time", "GB/s", "GFLOP/s", "%BWroof", "%FLroof", "waste_ms"
    ));
    for row in rows {
        out.push_str(&format!(
            "{:<28} {:>6} {:>10.3} {:>9.1}% {:>9.1} {:>9.1} {:>6.1}% {:>6.1}% {:>10.3}\n",
            row.label,
            row.count,
            row.elapsed_ms,
            row.pct_of_phase_time,
            row.gb_per_sec,
            row.gflop_per_sec,
            row.pct_of_bandwidth_roofline,
            row.pct_of_flop_roofline,
            row.wasted_ms,
        ));
    }
}

/// Ranks every phase's op-kind rows together by absolute time and reports
/// the top few as plain-English lines — the "so what should I optimize
/// next" summary the issue asks for.
pub(super) fn top_bottlenecks(phases: &[PhaseReport]) -> Vec<String> {
    const TOP_N: usize = 5;
    let mut all: Vec<(&'static str, &AggRow)> = phases
        .iter()
        .flat_map(|p| p.by_op.iter().map(move |r| (p.phase, r)))
        .collect();
    all.sort_by(|a, b| b.1.elapsed_ms.total_cmp(&a.1.elapsed_ms));
    all.into_iter()
        .take(TOP_N)
        .filter(|(_, row)| row.elapsed_ms > 0.0)
        .map(|(phase, row)| {
            format!(
                "{phase} {}: {:.1}% of {phase} time, {:.1} GB/s ({:.1}% of BW roofline), {:.1} GFLOP/s ({:.1}% of FLOP roofline)",
                row.label,
                row.pct_of_phase_time,
                row.gb_per_sec,
                row.pct_of_bandwidth_roofline,
                row.gflop_per_sec,
                row.pct_of_flop_roofline,
            )
        })
        .collect()
}

/// Ranks every phase's op-kind rows together by `wasted_ms` (see
/// `AggRow::wasted_ms`) and reports the top few — this is the optimization
/// worklist itself, not just a summary of where time went.
pub(super) fn top_worklist(phases: &[PhaseReport]) -> Vec<String> {
    const TOP_N: usize = 5;
    let mut all: Vec<(&'static str, &AggRow)> = phases
        .iter()
        .flat_map(|p| p.by_op.iter().map(move |r| (p.phase, r)))
        .collect();
    all.sort_by(|a, b| b.1.wasted_ms.total_cmp(&a.1.wasted_ms));
    all.into_iter()
        .take(TOP_N)
        .filter(|(_, row)| row.wasted_ms > 0.0)
        .map(|(phase, row)| {
            format!(
                "{phase} {}: {:.3} ms wasted ({:.1} ms actual vs {:.1} ms t_min, {:.0}% efficient)",
                row.label,
                row.wasted_ms,
                row.elapsed_ms,
                row.t_min_ms,
                row.efficiency * 100.0,
            )
        })
        .collect()
}
