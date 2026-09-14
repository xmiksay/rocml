//! Test-only support code, shared across integration tests but not part of
//! the `rocml` crate itself.

#[allow(dead_code)] // only `qwen35_cpu_reference.rs` uses this today.
pub mod qwen35_cpu;

#[allow(dead_code)] // only `mixed_kv_chunked_prefill_parity.rs` uses this today.
pub mod mixed_kv_chunked_prefill;
