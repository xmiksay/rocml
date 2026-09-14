//! CI-guarded coverage for issue #10's llama.cpp-reference-dump adapter
//! (`support::llama_ref`), independent of both real hardware and a real
//! llama.cpp checkout: exercises the parser + qwen35 node-name mapping
//! table against a small checked-in fixture of trimmed real `rocml-dump`
//! output (`tests/data/llama_ref_fixture.txt`, a handful of real
//! Qwen3.5-2B-Q8_0 tensors captured off an actual CPU forward pass, values
//! truncated to 16 columns), plus `layer_capture::diff_dumps`'s own
//! max/mean relative-error math on synthetic data. Not `#[ignore]`d — runs
//! under plain `cargo test`/`make test`.

mod support;

use std::path::Path;

use rocml::qwen35::config::LayerKind;
use rocml::qwen35::forward::layer_capture::{diff_dumps, CapturedTensor, LayerDump};
use support::llama_ref::convert::convert;
use support::llama_ref::parse::parse;

const FIXTURE_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/data/llama_ref_fixture.txt"
);

/// The fixture's own real layer-kind layout: layer 0 is GDN, layer 3 is
/// full-attention (confirmed against a real `rocml-dump` run — layer 3 has
/// no `linear_attn_qkv_mixed`/`linear_attn_out` node). Layers 1-2 are
/// absent from the trimmed fixture entirely; passing 4 entries here so
/// `convert` iterates the real layer count and reports 1/2's missing keys
/// rather than silently truncating the model.
fn fixture_layer_kinds() -> Vec<LayerKind> {
    vec![
        LayerKind::LinearAttention,
        LayerKind::LinearAttention,
        LayerKind::LinearAttention,
        LayerKind::FullAttention,
    ]
}

fn load_fixture() -> support::llama_ref::parse::RawDump {
    let text = std::fs::read_to_string(Path::new(FIXTURE_PATH)).expect("read fixture");
    parse(&text).expect("parse fixture")
}

#[test]
fn maps_gdn_layer_tensors_to_rocml_keys_with_real_values() {
    let raw = load_fixture();
    let (dump, _report) = convert(&raw, &fixture_layer_kinds());

    // First two values of layer 0's real (trimmed) `attn_residual-0` /
    // `l_out-0` dumps, copied (rounded) from the fixture file.
    let resid_pre_ffn = &dump.tensors["0:resid_pre_ffn"].values[..2];
    assert!((resid_pre_ffn[0] - -0.0057252f32).abs() < 1e-6);
    assert!((resid_pre_ffn[1] - 0.0543879f32).abs() < 1e-6);
    let resid_post = &dump.tensors["0:resid_post"].values[..2];
    assert!((resid_post[0] - 0.0086125f32).abs() < 1e-6);
    assert!((resid_post[1] - -0.0039845f32).abs() < 1e-6);
    assert_eq!(dump.tensors["0:resid_pre_ffn"].rows, 2);
    assert_eq!(dump.tensors["0:resid_pre_ffn"].cols, 16);
}

#[test]
fn maps_gdn_only_tensors_for_layer_0_only() {
    let raw = load_fixture();
    let (dump, _report) = convert(&raw, &fixture_layer_kinds());

    assert!(dump.tensors.contains_key("0:gdn_xn"));
    assert!(dump.tensors.contains_key("0:gdn_qkv_raw"));
    assert!(dump.tensors.contains_key("0:gdn_y_silu"));
    assert!(dump.tensors.contains_key("0:gdn_ssm_out_raw"));
    // Layer 3 is full-attention in this fixture: no GDN-only keys.
    assert!(!dump.tensors.contains_key("3:gdn_xn"));
    // But its shared (both-kind) tensors are present.
    assert!(dump.tensors.contains_key("3:resid_pre_ffn"));
    assert!(dump.tensors.contains_key("3:ffn_xn"));
    assert!(dump.tensors.contains_key("3:resid_post"));
}

#[test]
fn reports_llama_only_nodes_as_unmapped_not_errors() {
    let raw = load_fixture();
    let (_dump, report) = convert(&raw, &fixture_layer_kinds());

    // Real nodes captured alongside the mapped set specifically to
    // exercise this path (see docs/llama-diff.md): `Qcur_full` (a
    // full-attention-only Q projection) and `beta` (a GDN-internal gate
    // rocml doesn't separately capture).
    assert!(report
        .unmapped_llama_nodes
        .contains(&"Qcur_full".to_string()));
    assert!(report.unmapped_llama_nodes.contains(&"beta".to_string()));
}

#[test]
fn reports_missing_rocml_keys_for_the_fixtures_untrimmed_gap() {
    let raw = load_fixture();
    let (_dump, report) = convert(&raw, &fixture_layer_kinds());

    // The fixture deliberately omits layers 1-2 entirely.
    assert!(report
        .missing_rocml_keys
        .contains(&"1:resid_pre_ffn".to_string()));
    // And layer 3's ffn_swiglu/ffn_out weren't captured for this fixture.
    assert!(report
        .missing_rocml_keys
        .contains(&"3:ffn_gate_silu".to_string()));
}

#[test]
fn diff_dumps_reports_zero_error_against_an_identical_dump() {
    let raw = load_fixture();
    let (dump, _report) = convert(&raw, &fixture_layer_kinds());
    let diffs = diff_dumps(&dump, &dump);
    assert!(!diffs.is_empty());
    for d in &diffs {
        assert_eq!(
            d.max_rel, 0.0,
            "tensor {}:{} should match itself",
            d.layer_idx, d.tensor
        );
        assert_eq!(d.max_abs, 0.0);
    }
}

#[test]
fn diff_dumps_measures_a_known_perturbation() {
    let mut a = LayerDump::default();
    a.tensors.insert(
        "0:resid_post".to_string(),
        CapturedTensor {
            rows: 1,
            cols: 4,
            values: vec![1.0, 2.0, 4.0, -8.0],
        },
    );
    let mut b = LayerDump::default();
    b.tensors.insert(
        "0:resid_post".to_string(),
        CapturedTensor {
            rows: 1,
            cols: 4,
            // element 2 (value 4.0) perturbed by +0.4 -> 10% relative error,
            // the largest of the four -> expected max_rel.
            values: vec![1.0, 2.0, 4.4, -8.0],
        },
    );

    let diffs = diff_dumps(&a, &b);
    assert_eq!(diffs.len(), 1);
    let d = &diffs[0];
    assert_eq!(d.layer_idx, 0);
    assert_eq!(d.tensor, "resid_post");
    assert!((d.max_rel - 0.1).abs() < 1e-6, "max_rel = {}", d.max_rel);
    assert!((d.max_abs - 0.4).abs() < 1e-6, "max_abs = {}", d.max_abs);
    assert!((d.mean_abs - 0.1).abs() < 1e-6, "mean_abs = {}", d.mean_abs);
}

#[test]
fn diff_dumps_skips_tensors_with_mismatched_shapes() {
    let mut a = LayerDump::default();
    a.tensors.insert(
        "0:resid_post".to_string(),
        CapturedTensor {
            rows: 1,
            cols: 4,
            values: vec![1.0, 2.0, 3.0, 4.0],
        },
    );
    let mut b = LayerDump::default();
    b.tensors.insert(
        "0:resid_post".to_string(),
        CapturedTensor {
            rows: 1,
            cols: 2,
            values: vec![1.0, 2.0],
        },
    );
    assert!(diff_dumps(&a, &b).is_empty());
}
