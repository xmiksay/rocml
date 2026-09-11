//! rocml-core: GGUF loading, ggml quant dequantization, and byte-level BPE
//! tokenization — the CPU-side, hardware-agnostic building blocks rocml's
//! HIP backend loads weights and tokenizes prompts through.

pub mod gguf;
pub mod quant;
