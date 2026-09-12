//! Results JSON shape (`{label, timestamp, model_path, engine_config,
//! per_scenario[], aggregates{}, ppl}`, issue #15) plus incremental
//! load/save so a long eval run can be interrupted and resumed: every
//! scored scenario is written to `--out` immediately, and `--resume` skips
//! any scenario id already present in that file.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use rocml::RocmlError;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingSummary {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    pub ctx: usize,
    pub kv_cache: String,
    pub max_gen_tokens: usize,
    pub sampling: SamplingSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioResult {
    pub id: String,
    pub kind: String,
    pub pass: bool,
    pub detail: String,
    /// Raw model output, truncated to 2KB, for post-mortem — not the parsed
    /// content, the full decoded text (thinking + tool call blocks intact).
    pub raw_output: String,
    pub wall_seconds: f64,
}

const RAW_OUTPUT_CAP_BYTES: usize = 2048;

/// Truncates `s` to at most [`RAW_OUTPUT_CAP_BYTES`] bytes, respecting UTF-8
/// character boundaries (never splitting a multi-byte codepoint).
pub fn truncate_raw_output(s: &str) -> String {
    if s.len() <= RAW_OUTPUT_CAP_BYTES {
        return s.to_string();
    }
    let mut end = RAW_OUTPUT_CAP_BYTES;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}... [truncated]", &s[..end])
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KindAggregate {
    pub total: usize,
    pub passed: usize,
    pub fraction: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Aggregates {
    pub overall_total: usize,
    pub overall_passed: usize,
    pub overall_fraction: f64,
    pub by_kind: BTreeMap<String, KindAggregate>,
}

impl Aggregates {
    pub fn compute(results: &[ScenarioResult]) -> Self {
        let mut by_kind: BTreeMap<String, KindAggregate> = BTreeMap::new();
        for r in results {
            let entry = by_kind.entry(r.kind.clone()).or_default();
            entry.total += 1;
            if r.pass {
                entry.passed += 1;
            }
        }
        for agg in by_kind.values_mut() {
            agg.fraction = checked_fraction(agg.passed, agg.total);
        }
        let overall_total = results.len();
        let overall_passed = results.iter().filter(|r| r.pass).count();
        Self {
            overall_total,
            overall_passed,
            overall_fraction: checked_fraction(overall_passed, overall_total),
            by_kind,
        }
    }
}

fn checked_fraction(passed: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        passed as f64 / total as f64
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalResults {
    pub label: String,
    /// Unix seconds — kept as a plain integer rather than an ISO-8601
    /// string to avoid adding a date/time dependency for one field.
    pub timestamp_unix: u64,
    pub git_rev: String,
    pub model_path: String,
    pub engine_config: EngineConfig,
    pub per_scenario: Vec<ScenarioResult>,
    pub aggregates: Aggregates,
    pub ppl: Option<f64>,
    pub ppl_wall_seconds: Option<f64>,
}

impl EvalResults {
    pub fn new(label: String, model_path: String, engine_config: EngineConfig) -> Self {
        Self {
            label,
            timestamp_unix: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            git_rev: git_rev(),
            model_path,
            engine_config,
            per_scenario: Vec::new(),
            aggregates: Aggregates::default(),
            ppl: None,
            ppl_wall_seconds: None,
        }
    }

    /// `None` when `path` doesn't exist yet — a fresh run, resumed or not,
    /// starts with no scenarios already scored.
    pub fn load_if_exists(path: &Path) -> Result<Option<Self>, RocmlError> {
        if !path.exists() {
            return Ok(None);
        }
        let text = std::fs::read_to_string(path)?;
        Ok(Some(serde_json::from_str(&text)?))
    }

    pub fn already_scored_ids(&self) -> std::collections::HashSet<String> {
        self.per_scenario.iter().map(|r| r.id.clone()).collect()
    }

    pub fn push_scenario_result(&mut self, result: ScenarioResult) {
        self.per_scenario.push(result);
        self.aggregates = Aggregates::compute(&self.per_scenario);
    }

    /// Writes to a `.tmp` sibling then renames over `path` — an interrupted
    /// write (the process killed by a Bash timeout mid-run, see issue #15's
    /// resume design) leaves the previous good file in place instead of a
    /// truncated one `--resume` would then fail to parse.
    pub fn save(&self, path: &Path) -> Result<(), RocmlError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = path.with_extension("json.tmp");
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&tmp, json)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }
}

/// `git describe --always --dirty`, degrading to `"unknown"` if git isn't
/// available or this isn't a repo — a results JSON should never fail to
/// write just because provenance couldn't be determined.
fn git_rev() -> String {
    std::process::Command::new("git")
        .args(["describe", "--always", "--dirty"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(id: &str, kind: &str, pass: bool) -> ScenarioResult {
        ScenarioResult {
            id: id.to_string(),
            kind: kind.to_string(),
            pass,
            detail: "ok".to_string(),
            raw_output: String::new(),
            wall_seconds: 0.1,
        }
    }

    #[test]
    fn aggregates_compute_per_kind_and_overall() {
        let results = vec![
            result("a", "tool_choice", true),
            result("b", "tool_choice", false),
            result("c", "no_tool", true),
        ];
        let agg = Aggregates::compute(&results);
        assert_eq!(agg.overall_total, 3);
        assert_eq!(agg.overall_passed, 2);
        assert!((agg.overall_fraction - 2.0 / 3.0).abs() < 1e-9);
        assert_eq!(agg.by_kind["tool_choice"].total, 2);
        assert_eq!(agg.by_kind["tool_choice"].passed, 1);
        assert_eq!(agg.by_kind["no_tool"].fraction, 1.0);
    }

    #[test]
    fn empty_results_do_not_divide_by_zero() {
        let agg = Aggregates::compute(&[]);
        assert_eq!(agg.overall_fraction, 0.0);
    }

    #[test]
    fn truncate_raw_output_respects_utf8_boundaries() {
        let s = "a".repeat(RAW_OUTPUT_CAP_BYTES - 1) + "€€€"; // multi-byte tail
        let truncated = truncate_raw_output(&s);
        assert!(truncated.is_char_boundary(truncated.len() - "... [truncated]".len()));
        assert!(truncated.len() <= RAW_OUTPUT_CAP_BYTES + "... [truncated]".len());
    }

    #[test]
    fn truncate_raw_output_is_noop_under_cap() {
        assert_eq!(truncate_raw_output("short"), "short");
    }

    #[test]
    fn save_and_load_round_trip() {
        let dir =
            std::env::temp_dir().join(format!("rocml-eval-results-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("out.json");

        let engine_config = EngineConfig {
            ctx: 16384,
            kv_cache: "fp16".to_string(),
            max_gen_tokens: 2048,
            sampling: SamplingSummary {
                temperature: 0.0,
                top_k: None,
                top_p: None,
                seed: 0,
            },
        };
        let mut results = EvalResults::new(
            "test-label".to_string(),
            "/path.gguf".to_string(),
            engine_config,
        );
        results.push_scenario_result(result("a", "tool_choice", true));
        results.save(&path).unwrap();

        let loaded = EvalResults::load_if_exists(&path).unwrap().unwrap();
        assert_eq!(loaded.label, "test-label");
        assert_eq!(loaded.per_scenario.len(), 1);
        assert_eq!(loaded.already_scored_ids().len(), 1);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_if_exists_returns_none_for_missing_file() {
        let path = std::env::temp_dir().join("rocml-eval-results-missing-definitely.json");
        std::fs::remove_file(&path).ok();
        assert!(EvalResults::load_if_exists(&path).unwrap().is_none());
    }
}
