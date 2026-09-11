//! Merge-rank byte-pair-encoding: given a word already split into
//! byte-level "pseudo-char" symbols, repeatedly merge the adjacent pair
//! with the lowest merge rank until no known pair remains. This is the
//! same greedy algorithm as the original GPT-2 `encoder.py`'s `bpe()`.

use std::collections::HashMap;

pub(super) type MergeRanks = HashMap<(String, String), u32>;

/// Builds the rank table from the GGUF `tokenizer.ggml.merges` array, where
/// each entry is `"left right"` (single space separator — real byte-level
/// content never contains a literal space, since the byte 0x20 is remapped
/// to 'Ġ' before this table is ever consulted).
pub(super) fn build_ranks(merges: &[String]) -> Result<MergeRanks, super::TokenizerError> {
    let mut ranks = HashMap::with_capacity(merges.len());
    for (rank, entry) in merges.iter().enumerate() {
        let (left, right) = entry
            .split_once(' ')
            .ok_or_else(|| super::TokenizerError::MalformedMerge(entry.clone()))?;
        ranks.insert((left.to_string(), right.to_string()), rank as u32);
    }
    Ok(ranks)
}

/// Applies BPE merges to `symbols` in place until no adjacent pair has a
/// known rank. Quadratic in word length, which is fine: words coming out of
/// pre-tokenization are short (a handful of characters to a few dozen).
pub(super) fn merge(symbols: &mut Vec<String>, ranks: &MergeRanks) {
    loop {
        let mut best: Option<(u32, usize)> = None;
        for i in 0..symbols.len().saturating_sub(1) {
            if let Some(&rank) = ranks.get(&(symbols[i].clone(), symbols[i + 1].clone())) {
                if best.is_none_or(|(best_rank, _)| rank < best_rank) {
                    best = Some((rank, i));
                }
            }
        }
        let Some((_, i)) = best else { break };
        let merged = format!("{}{}", symbols[i], symbols[i + 1]);
        symbols.splice(i..=i + 1, [merged]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merges_lowest_rank_pair_first() {
        let merges = vec!["a b".to_string(), "ab c".to_string()];
        let ranks = build_ranks(&merges).unwrap();
        let mut symbols: Vec<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        merge(&mut symbols, &ranks);
        assert_eq!(symbols, vec!["abc".to_string()]);
    }

    #[test]
    fn leaves_symbols_alone_when_no_merge_applies() {
        let ranks = build_ranks(&["x y".to_string()]).unwrap();
        let mut symbols: Vec<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        merge(&mut symbols, &ranks);
        assert_eq!(
            symbols,
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn malformed_merge_entry_is_an_error() {
        assert!(build_ranks(&["nospace".to_string()]).is_err());
    }
}
