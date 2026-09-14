//! Maps llama.cpp's qwen35 graph node names onto rocml's `LayerCapture`
//! tensor keys. Every entry was confirmed against a real `rocml-dump` run
//! on Qwen3.5-2B-Q8_0 (both a GDN layer, index 0, and a full-attention
//! layer, index 3) — not guessed from source alone — by cross-checking
//! shapes and layer-kind coverage; see `docs/llama-diff.md`'s mapping
//! table for the per-entry citation into llama.cpp's
//! `src/models/qwen35.cpp`/`src/llama-graph.cpp` (`build_ffn`).

use std::collections::BTreeSet;

use rocml::qwen35::config::LayerKind;
use rocml::qwen35::forward::layer_capture::{CapturedTensor, LayerDump};

use super::parse::RawDump;

/// `(rocml tensor name, llama.cpp base node name)` — applies to every
/// layer regardless of kind: `ffn_chunk_step` runs identically for both
/// GDN and full-attention layers, and llama.cpp's qwen35 graph builds the
/// attention/GDN residual add and the FFN before branching back together
/// the same way for both.
pub const NODE_MAP: &[(&str, &str)] = &[
    // scratch.x right after the attention/GDN residual add, before the
    // post-attention norm.
    ("resid_pre_ffn", "attn_residual"),
    // rmsnorm(x, post_attention_norm) feeding the FFN gate/up projections.
    ("ffn_xn", "attn_post_norm"),
    // silu(gate) * up — llama.cpp fuses both into one `ggml_swiglu_split`
    // node since qwen35 uses a parallel (not sequential) SwiGLU gate.
    ("ffn_gate_silu", "ffn_swiglu"),
    // ffn_down's raw projection output, before the FFN residual add
    // (llama.cpp's `build_ffn` only names this "ffn_down" when a down
    // bias exists — qwen35 has none — so the caller's own "ffn_out" name
    // is the down-projection's raw output).
    ("ffn_down_out", "ffn_out"),
    // the full layer's output (post both residual adds); "l_out" is
    // "post_ffn" run through `build_cvec`, a no-op without control vectors.
    ("resid_post", "l_out"),
];

/// GDN-layer-only mappings (`LayerKind::LinearAttention`,
/// llama.cpp's `build_layer_attn_linear`).
pub const GDN_NODE_MAP: &[(&str, &str)] = &[
    // rmsnorm(x, attn_norm) feeding the GDN QKVZ/alpha/beta projections —
    // the same "attn_norm" node a full-attention layer's Q/K/V projections
    // read from, since llama.cpp computes it once before branching on
    // layer kind.
    ("gdn_xn", "attn_norm"),
    // the raw QKV-mixed projection, before the causal conv1d.
    ("gdn_qkv_raw", "linear_attn_qkv_mixed"),
    // the gated-RMSNorm output (norm(o) * silu(z)) feeding ssm_out.
    ("gdn_y_silu", "final_output"),
    // ssm_out's raw projection output, before the layer's residual add.
    ("gdn_ssm_out_raw", "linear_attn_out"),
];

#[derive(Debug, Default)]
pub struct UnmappedReport {
    /// Canonical llama.cpp node base names present in the dump that no
    /// entry in [`NODE_MAP`]/[`GDN_NODE_MAP`] consumed — e.g. `Qcur_full`,
    /// `beta`, every full-attention-only or GDN-internal intermediate
    /// rocml doesn't separately capture. Deduped (a name appears once
    /// regardless of how many layers had it) and sorted. Not an error —
    /// these are nodes that exist on only one side, exactly as expected.
    pub unmapped_llama_nodes: Vec<String>,
    /// `"{layer_idx}:{tensor}"` rocml keys [`NODE_MAP`]/[`GDN_NODE_MAP`]
    /// expected but the dump didn't contain for that layer — usually
    /// means the `rocml-dump` run used too narrow a `ROCML_DUMP_FILTER`.
    pub missing_rocml_keys: Vec<String>,
}

/// Converts a parsed `rocml-dump` [`RawDump`] into a [`LayerDump`] using
/// `layer_kinds` (`layer_kinds[i]` is layer `i`'s kind — e.g.
/// `Qwen35Config::layer_kinds` off the same GGUF used for both sides) to
/// decide when [`GDN_NODE_MAP`] applies.
pub fn convert(raw: &RawDump, layer_kinds: &[LayerKind]) -> (LayerDump, UnmappedReport) {
    let mut dump = LayerDump::default();
    let mut missing = Vec::new();
    let mut consumed: BTreeSet<String> = BTreeSet::new();

    for (layer_idx, kind) in layer_kinds.iter().enumerate() {
        let mut entries: Vec<&(&str, &str)> = NODE_MAP.iter().collect();
        if *kind == LayerKind::LinearAttention {
            entries.extend(GDN_NODE_MAP.iter());
        }
        for (rocml_name, llama_name) in entries {
            let key = format!("{llama_name}-{layer_idx}");
            match raw.get(&key) {
                Some(t) => {
                    let cols = t.ne[0] as u32;
                    let rows = (t.ne[1] * t.ne[2] * t.ne[3]) as u32;
                    dump.tensors.insert(
                        format!("{layer_idx}:{rocml_name}"),
                        CapturedTensor {
                            rows,
                            cols,
                            values: t.values.clone(),
                        },
                    );
                    consumed.insert(key);
                }
                None => missing.push(format!("{layer_idx}:{rocml_name}")),
            }
        }
    }

    // A base name being mapped *somewhere* (e.g. "attn_norm" for a GDN
    // layer's `gdn_xn`) doesn't mean every layer's instance of it was
    // consumed — a full-attention layer's own "attn_norm-{il}" (its Q/K/V
    // input norm) is real but has no rocml counterpart. So "unmapped" is
    // computed per exact key actually looked up above, not per base name.
    let unmapped: BTreeSet<String> = raw
        .keys()
        .filter(|key| !consumed.contains(*key))
        .map(|key| strip_layer_suffix(key).to_string())
        .collect();

    (
        dump,
        UnmappedReport {
            unmapped_llama_nodes: unmapped.into_iter().collect(),
            missing_rocml_keys: missing,
        },
    )
}

/// Strips a trailing `"-{digits}"` layer suffix (`ggml_format_name`'s
/// `"%s-%d"` convention for `il >= 0`); names with no such suffix (`il <
/// 0` nodes like `"result_output"`, or anything not layer-scoped) are
/// returned unchanged.
fn strip_layer_suffix(name: &str) -> &str {
    match name.rfind('-') {
        Some(i) if i + 1 < name.len() && name[i + 1..].bytes().all(|c| c.is_ascii_digit()) => {
            &name[..i]
        }
        _ => name,
    }
}

#[cfg(test)]
mod tests {
    use super::super::parse::parse;
    use super::*;

    fn tiny_dump() -> RawDump {
        // layer 0: GDN; layer 1: full-attention — deliberately exercises
        // both `NODE_MAP` alone and `NODE_MAP` + `GDN_NODE_MAP` together.
        let text = "\
#TENSOR attn_residual-0 2 1 1 1\n1 2\n\
#TENSOR attn_post_norm-0 2 1 1 1\n3 4\n\
#TENSOR ffn_swiglu-0 2 1 1 1\n5 6\n\
#TENSOR ffn_out-0 2 1 1 1\n7 8\n\
#TENSOR l_out-0 2 1 1 1\n9 10\n\
#TENSOR attn_norm-0 2 1 1 1\n11 12\n\
#TENSOR linear_attn_qkv_mixed-0 2 1 1 1\n13 14\n\
#TENSOR final_output-0 2 1 1 1\n15 16\n\
#TENSOR linear_attn_out-0 2 1 1 1\n17 18\n\
#TENSOR attn_residual-1 2 1 1 1\n21 22\n\
#TENSOR attn_post_norm-1 2 1 1 1\n23 24\n\
#TENSOR ffn_swiglu-1 2 1 1 1\n25 26\n\
#TENSOR ffn_out-1 2 1 1 1\n27 28\n\
#TENSOR l_out-1 2 1 1 1\n29 30\n\
#TENSOR attn_norm-1 2 1 1 1\n31 32\n\
#TENSOR Qcur_full-1 2 1 1 1\n33 34\n\
";
        parse(text).expect("fixture text must parse")
    }

    #[test]
    fn maps_ffn_and_residual_tensors_for_every_layer() {
        let raw = tiny_dump();
        let (dump, _report) = convert(
            &raw,
            &[LayerKind::LinearAttention, LayerKind::FullAttention],
        );
        assert_eq!(dump.tensors["0:resid_pre_ffn"].values, vec![1.0, 2.0]);
        assert_eq!(dump.tensors["0:ffn_xn"].values, vec![3.0, 4.0]);
        assert_eq!(dump.tensors["0:ffn_gate_silu"].values, vec![5.0, 6.0]);
        assert_eq!(dump.tensors["0:ffn_down_out"].values, vec![7.0, 8.0]);
        assert_eq!(dump.tensors["0:resid_post"].values, vec![9.0, 10.0]);
        assert_eq!(dump.tensors["1:resid_pre_ffn"].values, vec![21.0, 22.0]);
        assert_eq!(dump.tensors["1:resid_post"].values, vec![29.0, 30.0]);
    }

    #[test]
    fn maps_gdn_only_tensors_for_the_gdn_layer_and_skips_them_for_full_attention() {
        let raw = tiny_dump();
        let (dump, report) = convert(
            &raw,
            &[LayerKind::LinearAttention, LayerKind::FullAttention],
        );
        assert_eq!(dump.tensors["0:gdn_xn"].values, vec![11.0, 12.0]);
        assert_eq!(dump.tensors["0:gdn_qkv_raw"].values, vec![13.0, 14.0]);
        assert_eq!(dump.tensors["0:gdn_y_silu"].values, vec![15.0, 16.0]);
        assert_eq!(dump.tensors["0:gdn_ssm_out_raw"].values, vec![17.0, 18.0]);
        assert!(!dump.tensors.contains_key("1:gdn_xn"));
        // layer 1's "attn_norm-1" is real (it feeds full attention's Q/K/V
        // there, not GDN) but unconsumed by any mapping for a
        // FullAttention layer, so it must not silently vanish or get
        // mis-mapped — it shows up as unmapped instead.
        assert!(report
            .unmapped_llama_nodes
            .contains(&"attn_norm".to_string()));
    }

    #[test]
    fn reports_qcur_full_as_unmapped_not_an_error() {
        let raw = tiny_dump();
        let (_dump, report) = convert(
            &raw,
            &[LayerKind::LinearAttention, LayerKind::FullAttention],
        );
        assert!(report
            .unmapped_llama_nodes
            .contains(&"Qcur_full".to_string()));
    }

    #[test]
    fn reports_missing_rocml_keys_when_the_dump_is_too_narrow() {
        // No GDN-specific nodes captured for layer 0 at all.
        let text = "#TENSOR attn_residual-0 1 1 1 1\n1\n";
        let raw = parse(text).expect("parse failed");
        let (_dump, report) = convert(&raw, &[LayerKind::LinearAttention]);
        assert!(report.missing_rocml_keys.contains(&"0:gdn_xn".to_string()));
        assert!(report.missing_rocml_keys.contains(&"0:ffn_xn".to_string()));
    }

    #[test]
    fn strip_layer_suffix_leaves_non_layer_scoped_names_alone() {
        assert_eq!(strip_layer_suffix("result_output"), "result_output");
        assert_eq!(strip_layer_suffix("attn_norm-12"), "attn_norm");
    }
}
