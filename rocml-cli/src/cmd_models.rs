//! `rocml-cli models`: lists the compiled-in model registry, with a
//! present/missing marker for each entry against the resolved checkpoint
//! directory — a quick way to check what's already downloaded before
//! running `chat`/`bench`/`generate` against a registry name.

use clap::Args;
use rocml::registry::{resolved_path, ModelFamily, REGISTRY};
use rocml::RocmlError;

#[derive(Args, Debug)]
pub struct ModelsArgs {}

pub fn run(_args: &ModelsArgs) -> Result<(), RocmlError> {
    let name_width = REGISTRY.iter().map(|s| s.name.len()).max().unwrap_or(0);
    for spec in REGISTRY {
        let present = if resolved_path(spec).exists() {
            "present"
        } else {
            "missing"
        };
        println!(
            "{:name_width$}  {:<14}  {:<8}  {}",
            spec.name,
            family_label(spec.family),
            present,
            spec.gguf_rel,
        );
    }
    Ok(())
}

fn family_label(family: ModelFamily) -> &'static str {
    match family {
        ModelFamily::Qwen3Dense => "qwen3-dense",
        ModelFamily::Qwen35Hybrid => "qwen3.5-hybrid",
    }
}
