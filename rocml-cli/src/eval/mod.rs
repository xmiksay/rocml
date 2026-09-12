//! Agentic quality-eval harness (issue #15): a fixed set of tool-use
//! scenarios (`scenario`) scored deterministically (`scorer`) against a
//! greedily-decoded reply (`runner`), plus a secondary teacher-forced
//! perplexity signal (`ppl`) and the results JSON shape with resume support
//! (`results`). `filler` generates `long_context` scenarios' padding
//! prose deterministically from a seed instead of storing it.
//!
//! Split across these files purely for the workspace's 400-line file cap;
//! `crate::cmd_eval` is the only caller and owns CLI parsing plus the
//! top-level run loop.

pub mod filler;
pub mod ppl;
pub mod results;
pub mod runner;
pub mod scenario;
pub mod scorer;
