//! Shared sidecar file locations for issue #17's calibration/measurement
//! pair (`mmq_calibrate.rs` writes, `mmq_smoothquant_measure.rs` reads) —
//! factored out since integration test files can't `use` items from one
//! another directly (each `tests/*.rs` compiles as its own crate).

pub fn calibration_dir() -> std::path::PathBuf {
    std::env::temp_dir().join("rocml_mmq_calibration")
}

/// Per-tensor-name calibration amax sidecar (`CalibrationDump`).
pub fn calibration_path() -> std::path::PathBuf {
    calibration_dir().join("ornith9b_calibration.json")
}

/// Raw per-token captured activations (`LayerDump`) — needed for the
/// before/after flatness measurement, which a calibration amax summary
/// alone can't reconstruct.
pub fn raw_dump_path() -> std::path::PathBuf {
    calibration_dir().join("ornith9b_raw_dump.json")
}
