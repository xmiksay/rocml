//! Issue #14 phase 2's test-only harness: real K/V capture plus the
//! pairing/angle-source calibration-selection search, shared by
//! `rotational_kv_calibrate.rs` (builds and checks in the production
//! sidecar) and `rotational_kv_measure.rs` (the tensor-level quality-gate
//! table). The pure, reusable rotation/Lloyd-Max/sidecar math itself lives
//! in the main crate (`rocml::kv_quant::rotational`) rather than here,
//! since the model-level debug simulation (issue #14 step 3,
//! `MixedAttnPlane`'s eviction hook) needs it from production code too —
//! this module is the calibration/measurement *orchestration* on top of
//! that reference implementation, not a second copy of it.

pub mod attn;
pub mod capture;
pub mod fit;
