//! Test-only support code, shared across integration tests but not part of
//! the `rocml` crate itself.

#[allow(dead_code)] // only `qwen35_cpu_reference.rs` uses this today.
pub mod qwen35_cpu;

#[allow(dead_code)] // only `mixed_kv_chunked_prefill_parity.rs` uses this today.
pub mod mixed_kv_chunked_prefill;

#[allow(dead_code)] // only `mmq_smoothquant_measure.rs`/`mmq_calibrate.rs` use this today.
pub mod smoothquant;

#[allow(dead_code)] // only `rotational_kv_calibrate.rs`/`rotational_kv_measure.rs` use this today.
pub mod rotational_kv;

#[allow(dead_code)] // only `llama_layer_diff.rs`/`llama_ref_convert.rs` use this today.
pub mod llama_ref;
