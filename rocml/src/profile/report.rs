//! Aggregates raw `OpRecord`s into a per-phase, per-layer/per-op-kind
//! roofline report: time share, effective GB/s and GFLOP/s, and % of the
//! gfx1101 roofline (~624 GB/s HBM, ~15-20 TFLOP/s fp16 — see issue #5).

use serde::Serialize;
use std::collections::BTreeMap;

use super::{OpKind, Phase};

/// Measured/quoted for the RX 7800 XT (gfx1101) target in issue #5.
pub const ROOFLINE_BYTES_PER_SEC: f64 = 624.0e9;
/// Midpoint of the issue's quoted 15-20 TFLOP/s fp16 range.
pub const ROOFLINE_FLOPS_PER_SEC: f64 = 17.5e12;

/// One profiled span, ready for aggregation. `Profiler::finish` produces
/// these from its internal `Pending` records (which additionally carry the
/// live HIP events `finish` has already resolved into `elapsed_ms`).
pub struct OpRecord {
    pub layer: Option<u32>,
    pub op: OpKind,
    pub phase: Phase,
    pub bytes: u64,
    pub flops: u64,
    pub elapsed_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct AggRow {
    pub label: String,
    pub count: u64,
    pub bytes: u64,
    pub flops: u64,
    pub elapsed_ms: f64,
    pub gb_per_sec: f64,
    pub gflop_per_sec: f64,
    pub pct_of_bandwidth_roofline: f64,
    pub pct_of_flop_roofline: f64,
    pub pct_of_phase_time: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct PhaseReport {
    pub phase: &'static str,
    pub total_elapsed_ms: f64,
    pub by_layer: Vec<AggRow>,
    pub by_op: Vec<AggRow>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub phases: Vec<PhaseReport>,
    /// Top time-share op-kinds across every phase, as ready-to-print lines
    /// (e.g. "decode gdn-recur: 41.2% of decode time, 9.8 GB/s (1.6% of BW roofline)").
    pub bottlenecks: Vec<String>,
}

/// Accumulator for one aggregation bucket (a layer or an op-kind) before
/// rates are derived.
#[derive(Default)]
struct Acc {
    count: u64,
    bytes: u64,
    flops: u64,
    elapsed_ms: f64,
}

impl Acc {
    fn add(&mut self, r: &OpRecord) {
        self.count += 1;
        self.bytes += r.bytes;
        self.flops += r.flops;
        self.elapsed_ms += r.elapsed_ms;
    }

    fn into_row(self, label: String, phase_total_ms: f64) -> AggRow {
        let secs = self.elapsed_ms / 1000.0;
        let gb_per_sec = if secs > 0.0 {
            self.bytes as f64 / secs / 1e9
        } else {
            0.0
        };
        let gflop_per_sec = if secs > 0.0 {
            self.flops as f64 / secs / 1e9
        } else {
            0.0
        };
        AggRow {
            label,
            count: self.count,
            bytes: self.bytes,
            flops: self.flops,
            elapsed_ms: self.elapsed_ms,
            gb_per_sec,
            gflop_per_sec,
            pct_of_bandwidth_roofline: gb_per_sec * 1e9 / ROOFLINE_BYTES_PER_SEC * 100.0,
            pct_of_flop_roofline: gflop_per_sec * 1e9 / ROOFLINE_FLOPS_PER_SEC * 100.0,
            pct_of_phase_time: if phase_total_ms > 0.0 {
                self.elapsed_ms / phase_total_ms * 100.0
            } else {
                0.0
            },
        }
    }
}

impl Report {
    pub fn build(records: Vec<OpRecord>) -> Self {
        let mut by_phase: BTreeMap<&'static str, Vec<&OpRecord>> = BTreeMap::new();
        for r in &records {
            by_phase.entry(r.phase.label()).or_default().push(r);
        }

        let mut phases = Vec::new();
        for (phase_label, recs) in &by_phase {
            let total_elapsed_ms: f64 = recs.iter().map(|r| r.elapsed_ms).sum();

            let mut layer_acc: BTreeMap<Option<u32>, Acc> = BTreeMap::new();
            let mut op_acc: BTreeMap<OpKind, Acc> = BTreeMap::new();
            for r in recs {
                layer_acc.entry(r.layer).or_default().add(r);
                op_acc.entry(r.op).or_default().add(r);
            }

            let mut by_layer: Vec<AggRow> = layer_acc
                .into_iter()
                .map(|(layer, acc)| {
                    let label = match layer {
                        Some(idx) => format!("layer {idx}"),
                        None => "head (embed/final-norm/lm-head)".to_string(),
                    };
                    acc.into_row(label, total_elapsed_ms)
                })
                .collect();
            by_layer.sort_by(|a, b| b.elapsed_ms.total_cmp(&a.elapsed_ms));

            let mut by_op: Vec<AggRow> = op_acc
                .into_iter()
                .map(|(op, acc)| acc.into_row(op.label().to_string(), total_elapsed_ms))
                .collect();
            by_op.sort_by(|a, b| b.elapsed_ms.total_cmp(&a.elapsed_ms));

            phases.push(PhaseReport {
                phase: phase_label,
                total_elapsed_ms,
                by_layer,
                by_op,
            });
        }

        let bottlenecks = top_bottlenecks(&phases);
        Report {
            phases,
            bottlenecks,
        }
    }

    /// Human-readable table, printed by `rocml-cli bench --profile`/
    /// `generate --profile` after the run.
    pub fn to_human(&self) -> String {
        let mut out = String::new();
        for phase in &self.phases {
            out.push_str(&format!(
                "\n== {} phase: {:.2} ms profiled ==\n",
                phase.phase, phase.total_elapsed_ms
            ));
            out.push_str("-- by op-kind --\n");
            append_table(&mut out, &phase.by_op);
            out.push_str("-- by layer --\n");
            append_table(&mut out, &phase.by_layer);
        }
        if !self.bottlenecks.is_empty() {
            out.push_str("\n== top bottlenecks ==\n");
            for line in &self.bottlenecks {
                out.push_str(line);
                out.push('\n');
            }
        }
        out
    }
}

fn append_table(out: &mut String, rows: &[AggRow]) {
    out.push_str(&format!(
        "{:<28} {:>6} {:>10} {:>10} {:>9} {:>9} {:>7} {:>7}\n",
        "label", "n", "ms", "%time", "GB/s", "GFLOP/s", "%BWroof", "%FLroof"
    ));
    for row in rows {
        out.push_str(&format!(
            "{:<28} {:>6} {:>10.3} {:>9.1}% {:>9.1} {:>9.1} {:>6.1}% {:>6.1}%\n",
            row.label,
            row.count,
            row.elapsed_ms,
            row.pct_of_phase_time,
            row.gb_per_sec,
            row.gflop_per_sec,
            row.pct_of_bandwidth_roofline,
            row.pct_of_flop_roofline,
        ));
    }
}

/// Ranks every phase's op-kind rows together by absolute time and reports
/// the top few as plain-English lines — the "so what should I optimize
/// next" summary the issue asks for.
fn top_bottlenecks(phases: &[PhaseReport]) -> Vec<String> {
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
