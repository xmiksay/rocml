//! qwen35-family rule tables, split out of `mod.rs` purely for the 400-line
//! file cap: [`GDN_RULES`] (shared by both `ArchFamily::Qwen35Hybrid` and
//! `ArchFamily::Qwen35Moe` — the identical hybrid GDN/full-attention body)
//! and [`MOE_RULES`] (qwen35moe's mixture-of-experts FFN only).

use super::{MatchKind, PrecisionClass, Rule};

/// Qwen3.5 hybrid-only rules for the Gated Delta Net layers. `ssm_a`
/// (`A_log`) and `ssm_dt.bias` (`dt_bias`) are the actual scan-recurrence
/// scalars the issue's "must stay fp16/fp32" language means; `ssm_alpha`/
/// `ssm_beta` in the GGUF are the 2D *in-projections* that produce per-token
/// alpha/beta logits feeding that recurrence, not the recurrence state
/// itself — see `docs/quant-policy.md` for why this naming overlap with the
/// issue text is worth calling out explicitly rather than silently
/// resolving one way or the other.
pub(super) const GDN_RULES: &[Rule] = &[
    Rule {
        pattern: "ssm_a",
        kind: MatchKind::Suffix,
        class: PrecisionClass::FloatOnly,
        label: "GDN scan parameter A_log",
    },
    Rule {
        pattern: "ssm_dt.bias",
        kind: MatchKind::Suffix,
        class: PrecisionClass::FloatOnly,
        label: "GDN scan parameter dt_bias",
    },
    Rule {
        pattern: "ssm_conv1d.weight",
        kind: MatchKind::Suffix,
        class: PrecisionClass::FloatOnly,
        label: "GDN short-conv (causal conv1d) weight",
    },
    Rule {
        pattern: "ssm_alpha.weight",
        kind: MatchKind::Suffix,
        class: PrecisionClass::Q6OrBetter,
        label: "GDN alpha in-projection (2D, feeds decay logit)",
    },
    Rule {
        pattern: "ssm_beta.weight",
        kind: MatchKind::Suffix,
        class: PrecisionClass::Q6OrBetter,
        label: "GDN beta in-projection (2D, feeds write-strength logit)",
    },
    Rule {
        pattern: "attn_qkv.weight",
        kind: MatchKind::Suffix,
        class: PrecisionClass::Q6OrBetter,
        label: "GDN fused QKV in-projection (2D, feeds conv1d + recurrence)",
    },
    Rule {
        pattern: "attn_gate.weight",
        kind: MatchKind::Suffix,
        class: PrecisionClass::Q6OrBetter,
        label: "GDN output-gate (Z) in-projection (2D)",
    },
    Rule {
        pattern: "ssm_out.weight",
        kind: MatchKind::Suffix,
        class: PrecisionClass::Q5OrBetter,
        label: "ssm_out (GDN's down_proj analog, shapes residual stream)",
    },
];

/// qwen35moe-only rules for the mixture-of-experts FFN. `ffn_gate_exps`/
/// `ffn_up_exps` mirror `ffn_gate`/`ffn_up`'s "no floor" treatment (largest,
/// most robust tensors); `ffn_down_exps` mirrors `ffn_down`'s
/// residual-stream-shaping floor. The shared expert's own gate/up/down
/// tensors get the identical treatment under their own `_shexp` names; its
/// router-gate vector (`ffn_gate_inp_shexp`) and the main router
/// (`ffn_gate_inp`) are routing logits, not weights that shape the residual
/// stream, but every verified checkpoint keeps them F32 already (see
/// `docs/quant-policy.md`), so a `FloatOnly` floor here documents that
/// invariant rather than merely hoping it holds.
pub(super) const MOE_RULES: &[Rule] = &[
    Rule {
        pattern: "ffn_gate_inp.weight",
        kind: MatchKind::Suffix,
        class: PrecisionClass::FloatOnly,
        label: "MoE router (softmax gate logits)",
    },
    Rule {
        pattern: "ffn_gate_inp_shexp.weight",
        kind: MatchKind::Suffix,
        class: PrecisionClass::FloatOnly,
        label: "MoE shared-expert gate (sigmoid gate logit)",
    },
    Rule {
        pattern: "ffn_down_exps.weight",
        kind: MatchKind::Suffix,
        class: PrecisionClass::Q5OrBetter,
        label: "down_proj (residual-stream input projection), routed experts",
    },
    Rule {
        pattern: "ffn_down_shexp.weight",
        kind: MatchKind::Suffix,
        class: PrecisionClass::Q5OrBetter,
        label: "down_proj (residual-stream input projection), shared expert",
    },
    Rule {
        pattern: "ffn_gate_exps.weight",
        kind: MatchKind::Suffix,
        class: PrecisionClass::Any,
        label: "gate_proj (largest, most robust — no floor enforced), routed experts",
    },
    Rule {
        pattern: "ffn_up_exps.weight",
        kind: MatchKind::Suffix,
        class: PrecisionClass::Any,
        label: "up_proj (largest, most robust — no floor enforced), routed experts",
    },
    Rule {
        pattern: "ffn_gate_shexp.weight",
        kind: MatchKind::Suffix,
        class: PrecisionClass::Any,
        label: "gate_proj (largest, most robust — no floor enforced), shared expert",
    },
    Rule {
        pattern: "ffn_up_shexp.weight",
        kind: MatchKind::Suffix,
        class: PrecisionClass::Any,
        label: "up_proj (largest, most robust — no floor enforced), shared expert",
    },
];
