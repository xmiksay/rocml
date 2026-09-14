//! Support code for issue #17's standalone SmoothQuant-style measurement
//! (`rocml/tests/mmq_smoothquant_measure.rs`): a Q4_K block-level
//! fold/requant reference, a CPU mirror of the int8 activation quantizer
//! kernel, the SmoothQuant scale formula, and small pooled-statistic
//! helpers. Test-only — none of this is wired into the forward pass.

pub mod matmul;
pub mod paths;
pub mod q4k;
pub mod quantizer;
pub mod scale;
pub mod stats;
