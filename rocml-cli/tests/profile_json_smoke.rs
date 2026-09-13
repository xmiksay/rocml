//! Issue #5 smoke test: `rocml-cli bench --profile-json <path>` must write a
//! valid, well-shaped JSON roofline report. Spawns the real compiled binary
//! (there's no library target to call into directly) against Qwen3.5-2B —
//! small `--depth`/`--decode-tokens` keep this fast even in `--release`.
//! Skips itself if the checkpoint isn't present on this machine, same
//! convention as every other real-GGUF test (`rocml_core::testpaths`).

use std::process::Command;

use rocml_core::testpaths::checkpoint;

#[test]
fn profile_json_is_valid_and_well_shaped() {
    let Some(model_path) = checkpoint("Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf") else {
        return;
    };

    let out_path = std::env::temp_dir().join(format!(
        "rocml_profile_json_smoke_{}.json",
        std::process::id()
    ));
    // Best-effort cleanup of a stale file from a previous crashed run.
    let _ = std::fs::remove_file(&out_path);

    let status = Command::new(env!("CARGO_BIN_EXE_rocml-cli"))
        .args([
            "bench",
            "--model",
            model_path.to_str().expect("checkpoint path must be utf8"),
            "--depth",
            "16",
            "--decode-tokens",
            "4",
            "--runs",
            "1",
            "--profile-json",
        ])
        .arg(&out_path)
        .status()
        .expect("failed to spawn rocml-cli binary");
    assert!(status.success(), "rocml-cli bench exited with {status}");

    let raw = std::fs::read_to_string(&out_path).expect("--profile-json must write the file");
    let _ = std::fs::remove_file(&out_path);
    let doc: serde_json::Value =
        serde_json::from_str(&raw).expect("--profile-json output must be valid JSON");

    let metadata = doc
        .get("metadata")
        .expect("document must have a \"metadata\" object");
    assert_eq!(metadata["depth"], 16);
    assert_eq!(metadata["decode_tokens"], 4);

    let phases = doc["report"]["phases"]
        .as_array()
        .expect("report.phases must be an array");
    assert!(
        !phases.is_empty(),
        "expected at least one profiled phase (prefill and/or decode)"
    );
    for phase in phases {
        let by_op = phase["by_op"]
            .as_array()
            .expect("each phase must have by_op rows");
        assert!(!by_op.is_empty(), "by_op must have at least one row");
        let row = &by_op[0];
        for field in [
            "label",
            "count",
            "bytes",
            "flops",
            "elapsed_ms",
            "wall_us",
            "gb_per_sec",
            "gflop_per_sec",
            "pct_of_bandwidth_roofline",
            "pct_of_flop_roofline",
            "pct_of_phase_time",
            "t_min_ms",
            "efficiency",
            "wasted_ms",
        ] {
            assert!(
                row.get(field).is_some(),
                "by_op row missing expected field {field:?}: {row}"
            );
        }
        assert!(!phase["by_layer"]
            .as_array()
            .expect("each phase must have by_layer rows")
            .is_empty());
    }
}
