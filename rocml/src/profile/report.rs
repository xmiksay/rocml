//! Aggregates raw `OpRecord`s into a per-phase, per-layer/per-op-kind
//! roofline report: time share, effective GB/s and GFLOP/s, % of the
//! gfx1101 roofline (~624 GB/s HBM, ~15-20 TFLOP/s fp16 — see
//! `super::roofline`), and each row's bound-ness: the analytical min-time
//! model `t_min = max(bytes/BW, flops/peak)`, `efficiency = t_min/t_actual`,
//! and `wasted_ms = t_actual - t_min` — the column the issue #5 gap-analysis
//! sorts on to build the optimization worklist (see
//! `docs/prefill-gap-analysis.md`).

use serde::Serialize;
use std::collections::BTreeMap;

use super::roofline::{BW_ROOFLINE_BYTES_PER_SEC, PRACTICAL_FLOPS_PER_SEC};
use super::{OpKind, Phase};

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
    pub wall_us: f64,
    pub gb_per_sec: f64,
    pub gflop_per_sec: f64,
    pub pct_of_bandwidth_roofline: f64,
    pub pct_of_flop_roofline: f64,
    pub pct_of_phase_time: f64,
    /// Analytical min-time model for this row's total bytes/flops:
    /// `max(bytes/BW_ROOFLINE, flops/PRACTICAL_FLOPS_PER_SEC)`, i.e. the
    /// fastest this work could complete if it were perfectly bound by
    /// whichever roofline it's closer to. Not a per-kernel achievable
    /// bound (WMMA-eligible kernels can and do beat the scalar
    /// `PRACTICAL_FLOPS_PER_SEC` ceiling — see `super::roofline`'s doc
    /// comment) — a deliberately simple, single-formula worklist metric.
    pub t_min_ms: f64,
    /// `t_min_ms / elapsed_ms` (0 when `elapsed_ms` is 0). Can exceed 1.0
    /// when a WMMA-dispatched op beats the scalar-FMA practical ceiling
    /// `t_min_ms` assumes — see `wasted_ms`.
    pub efficiency: f64,
    /// `elapsed_ms - t_min_ms`. The optimization worklist column: sort rows
    /// by this, descending, to find where wall time is least explained by
    /// the analytical lower bound. Can go negative for WMMA-dispatched ops
    /// (already faster than the scalar-practical ceiling `t_min_ms`
    /// assumes) — that's a real, informative signal, not a clamped-away
    /// edge case.
    pub wasted_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct PhaseReport {
    pub phase: &'static str,
    pub total_elapsed_ms: f64,
    pub total_bytes: u64,
    pub total_flops: u64,
    pub by_layer: Vec<AggRow>,
    pub by_op: Vec<AggRow>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub phases: Vec<PhaseReport>,
    /// Top time-share op-kinds across every phase, as ready-to-print lines
    /// (e.g. "decode gdn-recur: 41.2% of decode time, 9.8 GB/s (1.6% of BW roofline)").
    pub bottlenecks: Vec<String>,
    /// Top wasted-time op-kinds across every phase (see `AggRow::wasted_ms`)
    /// — the same ranking `docs/prefill-gap-analysis.md` builds its lever
    /// list from, ready-to-print here for the human table too.
    pub worklist: Vec<String>,
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
        // Min-time model: how fast this row's total bytes/flops could move
        // if perfectly bound by whichever roofline dominates. Both terms
        // are in seconds; convert to ms to match `elapsed_ms`.
        let t_min_ms = (self.bytes as f64 / BW_ROOFLINE_BYTES_PER_SEC)
            .max(self.flops as f64 / PRACTICAL_FLOPS_PER_SEC)
            * 1000.0;
        let efficiency = if self.elapsed_ms > 0.0 {
            t_min_ms / self.elapsed_ms
        } else {
            0.0
        };
        AggRow {
            label,
            count: self.count,
            bytes: self.bytes,
            flops: self.flops,
            elapsed_ms: self.elapsed_ms,
            wall_us: self.elapsed_ms * 1000.0,
            gb_per_sec,
            gflop_per_sec,
            pct_of_bandwidth_roofline: gb_per_sec * 1e9 / BW_ROOFLINE_BYTES_PER_SEC * 100.0,
            pct_of_flop_roofline: gflop_per_sec * 1e9 / PRACTICAL_FLOPS_PER_SEC * 100.0,
            pct_of_phase_time: if phase_total_ms > 0.0 {
                self.elapsed_ms / phase_total_ms * 100.0
            } else {
                0.0
            },
            t_min_ms,
            efficiency,
            wasted_ms: self.elapsed_ms - t_min_ms,
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
            let total_bytes: u64 = recs.iter().map(|r| r.bytes).sum();
            let total_flops: u64 = recs.iter().map(|r| r.flops).sum();

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
                total_bytes,
                total_flops,
                by_layer,
                by_op,
            });
        }

        let bottlenecks = super::human::top_bottlenecks(&phases);
        let worklist = super::human::top_worklist(&phases);
        Report {
            phases,
            bottlenecks,
            worklist,
        }
    }

    /// Human-readable table, printed by `rocml-cli bench --profile`/
    /// `generate --profile` after the run. See `super::human` for the
    /// rendering logic.
    pub fn to_human(&self) -> String {
        super::human::to_human(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(layer: Option<u32>, op: OpKind, bytes: u64, flops: u64, elapsed_ms: f64) -> OpRecord {
        OpRecord {
            layer,
            op,
            phase: Phase::Prefill,
            bytes,
            flops,
            elapsed_ms,
        }
    }

    #[test]
    fn t_min_is_bandwidth_bound_when_bytes_dominate() {
        // 624e9 B/s roofline: 624 MB in 1ms is exactly bandwidth-saturated.
        // flops term (1 FLOP) is negligible next to the byte term.
        let report = Report::build(vec![rec(Some(0), OpKind::Qkv, 624_000_000, 1, 2.0)]);
        let row = &report.phases[0].by_op[0];
        assert!(
            (row.t_min_ms - 1.0).abs() < 1e-6,
            "t_min should be ~1ms (bandwidth-bound), got {}",
            row.t_min_ms
        );
    }

    #[test]
    fn t_min_is_flop_bound_when_flops_dominate() {
        // 17.5e12 FLOP/s practical ceiling: 17.5 GFLOP in 1ms exactly
        // saturates it, dwarfing the near-zero byte term.
        let report = Report::build(vec![rec(
            Some(0),
            OpKind::FfnGateUp,
            1,
            17_500_000_000,
            3.0,
        )]);
        let row = &report.phases[0].by_op[0];
        assert!(
            (row.t_min_ms - 1.0).abs() < 1e-6,
            "t_min should be ~1ms (flop-bound), got {}",
            row.t_min_ms
        );
    }

    #[test]
    fn efficiency_and_wasted_are_consistent_with_t_min() {
        let report = Report::build(vec![rec(Some(0), OpKind::Qkv, 624_000_000, 0, 4.0)]);
        let row = &report.phases[0].by_op[0];
        // t_min ~1ms, actual 4ms -> 25% efficient, 3ms wasted.
        assert!((row.efficiency - 0.25).abs() < 1e-6, "{}", row.efficiency);
        assert!((row.wasted_ms - 3.0).abs() < 1e-6, "{}", row.wasted_ms);
    }

    #[test]
    fn wasted_can_go_negative_for_work_faster_than_the_practical_ceiling() {
        // A WMMA-dispatched op can beat the scalar-practical FLOP ceiling
        // t_min assumes: efficiency > 1, wasted < 0. This is a deliberate,
        // documented signal (see `super::roofline`'s doc comment), not a bug.
        let report = Report::build(vec![rec(
            Some(0),
            OpKind::FfnGateUp,
            0,
            17_500_000_000,
            0.5,
        )]);
        let row = &report.phases[0].by_op[0];
        assert!(row.efficiency > 1.0, "{}", row.efficiency);
        assert!(row.wasted_ms < 0.0, "{}", row.wasted_ms);
    }

    #[test]
    fn zero_elapsed_row_has_zero_rates_and_zero_efficiency() {
        let report = Report::build(vec![rec(Some(0), OpKind::Norm, 1000, 1000, 0.0)]);
        let row = &report.phases[0].by_op[0];
        assert_eq!(row.gb_per_sec, 0.0);
        assert_eq!(row.gflop_per_sec, 0.0);
        assert_eq!(row.efficiency, 0.0);
    }

    #[test]
    fn phase_totals_sum_across_records() {
        let report = Report::build(vec![
            rec(Some(0), OpKind::Qkv, 100, 10, 1.0),
            rec(Some(1), OpKind::FfnDown, 200, 20, 2.0),
        ]);
        let phase = &report.phases[0];
        assert_eq!(phase.total_bytes, 300);
        assert_eq!(phase.total_flops, 30);
        assert!((phase.total_elapsed_ms - 3.0).abs() < 1e-9);
    }

    #[test]
    fn worklist_is_sorted_by_wasted_time_descending() {
        let report = Report::build(vec![
            rec(Some(0), OpKind::Qkv, 624_000_000, 0, 2.0), // ~1ms wasted
            rec(Some(1), OpKind::FfnDown, 624_000_000 * 5, 0, 5.5), // ~0.5ms wasted
        ]);
        let worklist = &report.worklist;
        assert_eq!(worklist.len(), 2);
        assert!(
            worklist[0].contains("qkv"),
            "largest-wasted op should sort first: {worklist:?}"
        );
    }

    #[test]
    fn by_layer_and_by_op_totals_agree() {
        let report = Report::build(vec![
            rec(Some(0), OpKind::Qkv, 100, 10, 1.0),
            rec(Some(0), OpKind::FfnDown, 50, 5, 0.5),
            rec(Some(1), OpKind::Qkv, 100, 10, 1.0),
        ]);
        let phase = &report.phases[0];
        let layer_ms: f64 = phase.by_layer.iter().map(|r| r.elapsed_ms).sum();
        let op_ms: f64 = phase.by_op.iter().map(|r| r.elapsed_ms).sum();
        assert!((layer_ms - op_ms).abs() < 1e-9);
        assert!((layer_ms - phase.total_elapsed_ms).abs() < 1e-9);
    }
}
