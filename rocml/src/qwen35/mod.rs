//! `general.architecture = "qwen35"`: the Qwen 3.5 hybrid architecture —
//! Gated Delta Net linear-attention layers with a full softmax-attention
//! layer every `full_attention_interval`-th block. Structurally mirrors the
//! top-level dense Qwen3 modules (`crate::config`/`weights`/`cache`/`forward`)
//! one level down, under this module.

pub mod cache;
pub(crate) mod cache_mixed;
pub mod config;
pub mod forward;
pub mod weights;

pub use config::Qwen35Config;
pub use forward::Model;
