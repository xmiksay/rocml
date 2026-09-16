//! Load-time weight-quant sensitivity policy (issue #4): audits every
//! tensor's on-disk GGUF dtype against a design-review policy — which
//! tensors must stay full float, which need at least Q6_K, which merely
//! deserve "one step above the rest", and which tolerate anything — and
//! reports violations. **Informational only**: nothing here ever fails a
//! load. See `docs/quant-policy.md` for the full audit writeup, including
//! where this policy disagrees with the issue-15 outcome-level eval (the
//! eval wins on that disagreement — this check exists to make the tradeoff
//! visible, not to relitigate it at every startup).
//!
//! Architecture-generic by construction (issue #16): patterns are matched
//! by GGUF tensor-name suffix, not by hardcoded layer indices or shapes,
//! and the GDN-specific rules only apply when the caller says this file is
//! [`ArchFamily::Qwen35Hybrid`] — a future architecture family adds its own
//! rule table alongside [`COMMON_RULES`]/`rules::GDN_RULES` rather than
//! touching either. The qwen35-family-specific rule tables
//! (`GDN_RULES`/`MOE_RULES`) live in the sibling `rules` module purely for
//! the 400-line file cap.

mod rules;

use rocml_core::gguf::GgufFile;
use rocml_core::quant::GgmlDType;
use rules::{GDN_RULES, MOE_RULES};

/// Which architecture's tensor-name conventions apply. Mirrors
/// `general.architecture` (`"qwen3"`/`"qwen35"`), not `registry::ModelFamily`
/// — this module has no reason to depend on the registry, and each loader
/// already knows its own family statically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchFamily {
    Qwen3Dense,
    Qwen35Hybrid,
    /// `general.architecture = "qwen35moe"` — the identical hybrid GDN/
    /// full-attention body as [`Self::Qwen35Hybrid`] (so [`GDN_RULES`] also
    /// applies), but every layer's FFN is a mixture of experts instead of
    /// one dense SwiGLU MLP — see [`MOE_RULES`].
    Qwen35Moe,
}

/// Minimum acceptable precision for a tensor-name pattern. Compared against
/// a dtype's [`precision_rank`] rather than matched as an exact quant type,
/// since "at least Q6" should also accept Q8_0/F16/F32.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecisionClass {
    /// Scan-recurrence parameters and norm weights: F32/F16/BF16 only.
    FloatOnly,
    /// lm_head, embeddings, GDN 2D in-projections.
    Q6OrBetter,
    /// "One step higher than the rest" (o_proj / down_proj family). Modeled
    /// as an absolute Q5_K floor rather than a scheme-relative "one step up
    /// from whatever the baseline quant is" — simpler to check at load time
    /// and already flags the case that matters (a residual-stream
    /// projection quantized down to the scheme's own baseline or below).
    Q5OrBetter,
    /// up_proj/gate_proj and anything not named by the policy — no floor.
    Any,
}

impl PrecisionClass {
    fn min_rank(self) -> i32 {
        match self {
            Self::FloatOnly => 90,
            Self::Q6OrBetter => 60,
            Self::Q5OrBetter => 50,
            Self::Any => i32::MIN,
        }
    }
}

/// Coarse "how much precision does this dtype carry" ordering, high is
/// better. Not bits-per-weight exactly (e.g. Q8_0's ~8.5 vs Q6_K's ~6.5625
/// collapse to a wider 80/60 gap here) — only relative order matters for
/// comparing against a [`PrecisionClass`]'s floor, and a wide gap makes that
/// order obviously stable across ggml's actual per-type bit costs.
/// `Unsupported` ranks below every real class so any threshold above `Any`
/// flags it rather than silently passing an unrecognized dtype.
fn precision_rank(dtype: GgmlDType) -> i32 {
    match dtype {
        GgmlDType::F32 => 100,
        GgmlDType::F16 | GgmlDType::BF16 => 90,
        GgmlDType::Q8_0 => 80,
        GgmlDType::Q6_K => 60,
        GgmlDType::Q5_K => 50,
        GgmlDType::Q4_K => 40,
        GgmlDType::Q3_K => 30,
        GgmlDType::Q2_K => 20,
        GgmlDType::Unsupported(_) => i32::MIN,
    }
}

/// Whether a [`Rule`]'s `pattern` must match the tensor name exactly (for
/// the handful of top-level, unprefixed tensors) or just its suffix (for
/// per-layer tensors, whose full name is `blk.N.<pattern>`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MatchKind {
    Exact,
    Suffix,
}

/// One tensor-name-pattern rule. Exact patterns avoid the trap a plain
/// suffix match would hit here: `"output.weight"` as a *suffix* would also
/// match `"blk.0.attn_output.weight"`, silently misclassifying every
/// attention output projection as the lm_head — `token_embd.weight` and
/// `output.weight` are always exact top-level names in this codebase's
/// loaders (see `weights::ModelWeights::load`), so `Exact` is both correct
/// and sufficient for them.
struct Rule {
    pattern: &'static str,
    kind: MatchKind,
    class: PrecisionClass,
    label: &'static str,
}

impl Rule {
    fn matches(&self, name: &str) -> bool {
        match self.kind {
            MatchKind::Exact => name == self.pattern,
            MatchKind::Suffix => name.ends_with(self.pattern),
        }
    }
}

/// Rules shared by every architecture this engine loads: norm weights
/// (llama.cpp's own convention — every 1D norm tensor stays F32 regardless
/// of the file's quant scheme, verified by this repo's loader always
/// dequantizing them through `load_vector_f32` too), the embedding table
/// and lm_head, and the two residual-stream-shaping projections every
/// transformer block has (`attn_output`/`ffn_down`).
const COMMON_RULES: &[Rule] = &[
    Rule {
        pattern: "norm.weight",
        kind: MatchKind::Suffix,
        class: PrecisionClass::FloatOnly,
        label: "norm weight",
    },
    Rule {
        pattern: "token_embd.weight",
        kind: MatchKind::Exact,
        class: PrecisionClass::Q6OrBetter,
        label: "embedding table",
    },
    Rule {
        pattern: "output.weight",
        kind: MatchKind::Exact,
        class: PrecisionClass::Q6OrBetter,
        label: "lm_head / output projection",
    },
    Rule {
        pattern: "attn_output.weight",
        kind: MatchKind::Suffix,
        class: PrecisionClass::Q5OrBetter,
        label: "o_proj (residual-stream output projection)",
    },
    Rule {
        pattern: "ffn_down.weight",
        kind: MatchKind::Suffix,
        class: PrecisionClass::Q5OrBetter,
        label: "down_proj (residual-stream input projection)",
    },
    Rule {
        pattern: "ffn_gate.weight",
        kind: MatchKind::Suffix,
        class: PrecisionClass::Any,
        label: "gate_proj (largest, most robust — no floor enforced)",
    },
    Rule {
        pattern: "ffn_up.weight",
        kind: MatchKind::Suffix,
        class: PrecisionClass::Any,
        label: "up_proj (largest, most robust — no floor enforced)",
    },
];

fn classify(name: &str, family: ArchFamily) -> Option<&'static Rule> {
    if matches!(family, ArchFamily::Qwen35Hybrid | ArchFamily::Qwen35Moe) {
        if let Some(rule) = GDN_RULES.iter().find(|r| r.matches(name)) {
            return Some(rule);
        }
    }
    if family == ArchFamily::Qwen35Moe {
        if let Some(rule) = MOE_RULES.iter().find(|r| r.matches(name)) {
            return Some(rule);
        }
    }
    COMMON_RULES.iter().find(|r| r.matches(name))
}

/// One tensor whose dtype falls below its policy class's floor.
#[derive(Debug, Clone)]
pub struct Violation {
    pub tensor: String,
    pub dtype: GgmlDType,
    pub label: &'static str,
}

/// Result of auditing every tensor in a GGUF against the policy tables.
#[derive(Debug, Clone, Default)]
pub struct PolicyReport {
    /// Tensors matched by a rule (regardless of pass/fail).
    pub checked: usize,
    pub violations: Vec<Violation>,
}

impl PolicyReport {
    pub fn is_clean(&self) -> bool {
        self.violations.is_empty()
    }

    /// Prints one `warning:` line per distinct policy label with a
    /// violation (the same `warning: ...` convention
    /// `registry::clamp_ctx`/`forward::Model::load`'s budget warning use),
    /// naming the count and one example tensor rather than every match —
    /// a Q4_K_M checkpoint's GDN in-projections violate the same rule once
    /// per layer, and a wall of ~24 near-identical lines at every startup
    /// would bury the one line anyone actually needs to read. A no-op when
    /// the audit found nothing.
    pub fn warn_violations(&self) {
        for label in Self::distinct_labels(&self.violations) {
            let matches: Vec<&Violation> = self
                .violations
                .iter()
                .filter(|v| v.label == label)
                .collect();
            let example = &matches[0];
            eprintln!(
                "warning: quant-policy: {} tensor(s) below the recommended floor for {label} \
                 (e.g. {} = {:?}) — informational only, see docs/quant-policy.md",
                matches.len(),
                example.tensor,
                example.dtype,
            );
        }
    }

    /// Labels with at least one violation, in first-seen order (stable
    /// output — `violations` is itself in tensor-iteration order).
    fn distinct_labels(violations: &[Violation]) -> Vec<&'static str> {
        let mut labels = Vec::new();
        for v in violations {
            if !labels.contains(&v.label) {
                labels.push(v.label);
            }
        }
        labels
    }
}

/// Audits every tensor in `gguf` against `family`'s policy tables. Pure and
/// infallible: an unrecognized/unclassified tensor name is simply not
/// counted (`checked` only counts tensors a rule actually matched), and an
/// `Unsupported` dtype always violates every class stricter than `Any`
/// rather than panicking on an unknown discriminant.
pub fn audit(gguf: &GgufFile, family: ArchFamily) -> PolicyReport {
    let mut report = PolicyReport::default();
    for t in gguf.tensors() {
        let Some(rule) = classify(&t.name, family) else {
            continue;
        };
        report.checked += 1;
        if precision_rank(t.dtype) < rule.class.min_rank() {
            report.violations.push(Violation {
                tensor: t.name.clone(),
                dtype: t.dtype,
                label: rule.label,
            });
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm_weight_matches_every_norm_tensor_name() {
        for name in [
            "output_norm.weight",
            "blk.0.attn_norm.weight",
            "blk.0.ffn_norm.weight",
            "blk.0.post_attention_norm.weight",
            "blk.0.attn_q_norm.weight",
            "blk.0.attn_k_norm.weight",
            "blk.0.ssm_norm.weight",
        ] {
            let rule = classify(name, ArchFamily::Qwen35Hybrid).unwrap_or_else(|| {
                panic!("{name} should match a rule");
            });
            assert_eq!(rule.class, PrecisionClass::FloatOnly, "{name}");
        }
    }

    #[test]
    fn gdn_rules_only_apply_to_qwen35_hybrid() {
        assert!(classify("blk.0.ssm_a", ArchFamily::Qwen3Dense).is_none());
        assert!(classify("blk.0.ssm_a", ArchFamily::Qwen35Hybrid).is_some());
        assert!(classify("blk.0.attn_qkv.weight", ArchFamily::Qwen3Dense).is_none());
        assert!(classify("blk.0.attn_qkv.weight", ArchFamily::Qwen35Hybrid).is_some());
    }

    #[test]
    fn dense_arch_still_classifies_shared_tensor_names() {
        for (name, class) in [
            ("token_embd.weight", PrecisionClass::Q6OrBetter),
            ("output.weight", PrecisionClass::Q6OrBetter),
            ("blk.0.attn_output.weight", PrecisionClass::Q5OrBetter),
            ("blk.0.ffn_down.weight", PrecisionClass::Q5OrBetter),
            ("blk.0.ffn_gate.weight", PrecisionClass::Any),
            ("blk.0.ffn_up.weight", PrecisionClass::Any),
        ] {
            let rule = classify(name, ArchFamily::Qwen3Dense).unwrap();
            assert_eq!(rule.class, class, "{name}");
        }
    }

    #[test]
    fn precision_rank_orders_float_above_every_quant() {
        for q in [
            GgmlDType::Q8_0,
            GgmlDType::Q6_K,
            GgmlDType::Q5_K,
            GgmlDType::Q4_K,
            GgmlDType::Q3_K,
            GgmlDType::Q2_K,
        ] {
            assert!(precision_rank(GgmlDType::F32) > precision_rank(q));
            assert!(precision_rank(GgmlDType::F16) > precision_rank(q));
        }
    }

    #[test]
    fn unsupported_dtype_violates_every_class_above_any() {
        assert!(precision_rank(GgmlDType::Unsupported(999)) < PrecisionClass::FloatOnly.min_rank());
        assert!(
            precision_rank(GgmlDType::Unsupported(999)) < PrecisionClass::Q6OrBetter.min_rank()
        );
        assert!(
            precision_rank(GgmlDType::Unsupported(999)) < PrecisionClass::Q5OrBetter.min_rank()
        );
        assert!(precision_rank(GgmlDType::Unsupported(999)) >= PrecisionClass::Any.min_rank());
    }

    #[test]
    fn audit_flags_q4_k_in_projection_below_q6_floor() {
        // Synthetic single-tensor check without opening a real GGUF: the
        // rule table + rank comparison is what's under test here, not the
        // parser. The real-GGUF pin lives in
        // `rocml/tests/quant_policy_audit.rs`.
        let rule = classify("blk.0.ssm_alpha.weight", ArchFamily::Qwen35Hybrid).unwrap();
        assert!(precision_rank(GgmlDType::Q4_K) < rule.class.min_rank());
        assert!(precision_rank(GgmlDType::Q6_K) >= rule.class.min_rank());
    }
}
