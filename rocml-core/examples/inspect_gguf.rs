//! Throwaway GGUF tensor inspector for issue #8 Phase-1 investigation
//! (MTP-head tensor presence check). Not part of the shipped CLI surface.
fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: inspect_gguf <path.gguf>");
    let f = rocml_core::gguf::GgufFile::open(&path).expect("open gguf");
    println!("version: {}", f.version());
    if let Ok(bc) = f.get_u32("qwen35.block_count") {
        println!("qwen35.block_count: {bc}");
    }
    if let Ok(bc) = f.get_u32("qwen3.block_count") {
        println!("qwen3.block_count: {bc}");
    }

    let mut names: Vec<&str> = f.tensors().iter().map(|t| t.name.as_str()).collect();
    names.sort();
    println!("total tensors: {}", names.len());

    let mut max_blk: i64 = -1;
    for n in &names {
        if let Some(rest) = n.strip_prefix("blk.") {
            if let Some(idx_str) = rest.split('.').next() {
                if let Ok(idx) = idx_str.parse::<i64>() {
                    max_blk = max_blk.max(idx);
                }
            }
        }
    }
    println!("max blk index: {max_blk}");

    println!("--- MTP/nextn-suspicious tensors ---");
    for n in &names {
        let lower = n.to_lowercase();
        if lower.contains("mtp")
            || lower.contains("nextn")
            || lower.contains("eh_proj")
            || lower.contains("enorm")
            || lower.contains("hnorm")
        {
            let t = f.tensor(n).unwrap();
            println!("MATCH: {n} shape={:?} dtype={:?}", t.shape(), t.dtype());
        }
    }

    println!("--- all tensors for the last block index (and one past it) ---");
    for n in &names {
        if n.starts_with(&format!("blk.{max_blk}."))
            || n.starts_with(&format!("blk.{}.", max_blk + 1))
        {
            let t = f.tensor(n).unwrap();
            println!("{n} shape={:?} dtype={:?}", t.shape(), t.dtype());
        }
    }
}
