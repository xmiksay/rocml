//! rocml: dense Qwen3 GPU inference engine built on `rocml-core` (GGUF +
//! CPU dequant + tokenizer), `rocml-hip` (HIP runtime wrapper) and
//! `rocml-kernels` (hand-written HIP kernels). Milestone 3 scope: greedy,
//! single-sequence, single-token-at-a-time (decode-style, including
//! prompt processing) text generation.

pub mod cache;
pub mod chat;
pub mod config;
pub mod error;
pub mod forward;
pub mod generate;
mod model;
pub mod profile;
pub mod qwen35;
pub mod registry;
pub mod sample;
pub mod weights;

pub use config::ModelConfig;
pub use error::RocmlError;
pub use generate::{
    generate, generate_sampled, generate_sampled_profiled, generate_sampled_with_stop,
    GenerateStats,
};
pub use model::Model;
pub use profile::{OpKind, Phase, Profiler, Report};
pub use registry::{resolve, ModelFamily, ModelSpec, ResolvedModel};
pub use sample::{Rng, SamplingParams};
