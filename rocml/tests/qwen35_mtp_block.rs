//! Pins `Qwen35Config`'s exclusion of trailing multi-token-prediction blocks
//! against the real checkpoints. Pure GGUF metadata reads (no GPU), so it
//! runs in the default `cargo test --workspace` pass. Ornith-1.5-9B's `blk.32`
//! carries ordinary attention/FFN tensors plus `nextn.*`: counting it loads
//! cleanly and silently runs the draft head as a 33rd layer.

use rocml::qwen35::config::LayerKind;
use rocml::qwen35::Qwen35Config;
use rocml_core::gguf::GgufFile;
use rocml_core::testpaths::checkpoint;

/// Full-attention layers of both Ornith-9B generations
/// (`full_attention_interval = 4` over 32 forward-pass layers).
const FULL_ATTENTION_LAYERS: [usize; 8] = [3, 7, 11, 15, 19, 23, 27, 31];

fn full_attention_layers(config: &Qwen35Config) -> Vec<usize> {
    config
        .layer_kinds
        .iter()
        .enumerate()
        .filter(|(_, kind)| **kind == LayerKind::FullAttention)
        .map(|(i, _)| i)
        .collect()
}

#[test]
fn ornith_1_5_config_excludes_mtp_block() {
    let Some(path) = checkpoint("Ornith-1.5-9B-GGUF/Ornith-1.5-9B-Q4_K_M.gguf") else {
        return;
    };
    let gguf = GgufFile::open(&path).expect("open gguf");
    // Guards that this file still exercises the MTP case at all.
    assert_eq!(gguf.get_u32("qwen35.block_count").unwrap(), 33);
    assert_eq!(gguf.get_u32("qwen35.nextn_predict_layers").unwrap(), 1);
    assert!(gguf.tensor("blk.32.nextn.eh_proj.weight").is_ok());

    let config = Qwen35Config::from_gguf(&gguf).expect("parse config");
    assert_eq!(config.block_count, 32);
    assert_eq!(config.layer_kinds.len(), 32);
    assert_eq!(full_attention_layers(&config), FULL_ATTENTION_LAYERS);
}

#[test]
fn ornith_1_0_config_without_mtp_is_unchanged() {
    let Some(path) = checkpoint("Ornith-1.0-9B-GGUF/ornith-1.0-9b-Q4_K_M.gguf") else {
        return;
    };
    let gguf = GgufFile::open(&path).expect("open gguf");
    assert!(gguf.get_u32("qwen35.nextn_predict_layers").is_err());

    let config = Qwen35Config::from_gguf(&gguf).expect("parse config");
    assert_eq!(config.block_count, 32);
    assert_eq!(full_attention_layers(&config), FULL_ATTENTION_LAYERS);
}
