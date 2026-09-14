//! Issue #10's llama.cpp-reference-dump adapter: parses the plain-text
//! tensor dump produced by the scratchpad `rocml-dump` llama.cpp example
//! (see `docs/llama-diff.md` for the tool's source and how to build/run
//! it) and converts it into a `qwen35::forward::layer_capture::LayerDump`
//! — rocml's own per-layer capture schema — so
//! `layer_capture::diff_dumps` can compare rocml's GPU forward pass
//! against llama.cpp's CPU reference implementation layer-by-layer,
//! exactly like it already compares two rocml runs (`mmq_layer_diff.rs`).
//!
//! Split into `parse` (the dump's line format, architecture-agnostic) and
//! `convert` (the qwen35-specific node-name mapping table) to stay under
//! the 400-line file cap.

pub mod convert;
pub mod parse;
