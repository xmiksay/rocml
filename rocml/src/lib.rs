//! rocml: dense Qwen3 GPU inference engine built on `rocml-core` (GGUF +
//! CPU dequant + tokenizer), `rocml-hip` (HIP runtime wrapper) and
//! `rocml-kernels` (hand-written HIP kernels). Milestone 3 scope: greedy,
//! single-sequence, single-token-at-a-time (decode-style, including
//! prompt processing) text generation.

pub mod budget;
pub mod cache;
pub mod chat;
pub mod config;
pub mod error;
pub mod forward;
pub mod generate;
pub mod kv_quant;
mod load_opts;
mod model;
pub mod profile;
pub mod qwen35;
pub mod registry;
pub mod sample;
pub mod snapshot;
pub mod weights;

pub use cache::KvDtype;
pub use config::ModelConfig;
pub use error::RocmlError;
pub use generate::{
    generate, generate_sampled, generate_sampled_profiled, generate_sampled_profiled_resumed,
    generate_sampled_resumed, generate_sampled_with_stop, generate_sampled_with_stop_resumed,
    GenerateStats,
};
pub use load_opts::{KvCacheMode, LoadOptions};
pub use model::Model;
pub use profile::{OpKind, Phase, Profiler, Report};
pub use registry::{resolve, ModelFamily, ModelSpec, ResolvedModel};
pub use sample::{Rng, SamplingParams};
pub use snapshot::{KvConfigStamp, ModelStamp, SnapshotStore};
