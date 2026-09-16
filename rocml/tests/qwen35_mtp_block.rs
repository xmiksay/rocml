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

/// Ornith-1.5-35B-A3B (`general.architecture = "qwen35moe"`): the same
/// hybrid GDN/full-attention body as the two dense-FFN checkpoints above,
/// `block_count = 41` = 40 forward layers + 1 MTP block
/// (`nextn_predict_layers = 1`), full-attention every 4th layer. Also pins
/// that the MoE-specific metadata parses and the dense-only
/// `feed_forward_length` key (absent from this file) doesn't stop
/// `from_gguf` from succeeding.
#[test]
fn ornith_1_5_35b_moe_config_excludes_mtp_block() {
    let Some(path) = checkpoint("Ornith-1.5-35B-A3B-GGUF/Ornith-1.5-35B-Q4_K_M.gguf") else {
        return;
    };
    let gguf = GgufFile::open(&path).expect("open gguf");
    assert_eq!(gguf.get_str("general.architecture").unwrap(), "qwen35moe");
    assert_eq!(gguf.get_u32("qwen35moe.block_count").unwrap(), 41);
    assert_eq!(gguf.get_u32("qwen35moe.nextn_predict_layers").unwrap(), 1);
    assert!(gguf.get_u32("qwen35moe.feed_forward_length").is_err());

    let config = Qwen35Config::from_gguf(&gguf).expect("parse config");
    assert_eq!(config.block_count, 40);
    assert_eq!(config.layer_kinds.len(), 40);
    let full_attn: Vec<usize> = full_attention_layers(&config);
    assert_eq!(full_attn.len(), 10);
    assert_eq!(full_attn[0], 3);
    assert_eq!(full_attn.last().copied(), Some(39));

    let moe = config
        .moe
        .expect("qwen35moe checkpoint must parse a MoeConfig");
    assert_eq!(moe.expert_count, 256);
    assert_eq!(moe.expert_used_count, 8);
    assert_eq!(moe.expert_ff_len, 512);
    assert_eq!(moe.shared_ff_len, 512);
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
