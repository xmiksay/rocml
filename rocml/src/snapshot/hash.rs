//! Chain hash over token ids and a byte checksum for on-disk corruption
//! detection — both hand-rolled FNV-1a (no new dependency, per issue #1's
//! hard rule) folded one unit at a time, which is what makes the token hash
//! "chain"/"rolling": a longer prefix's hash is obtained by continuing to
//! fold onto the shorter prefix's already-computed [`ChainHash`] rather than
//! rehashing from token 0 every time.

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Running FNV-1a state over a token id sequence. `Default`/`new` is the
/// empty-prefix hash; [`Self::fold`] extends it by exactly one token, so
/// `ChainHash::new().fold_all(&tokens[..n])` for increasing `n` reuses all
/// prior work instead of restarting — the "block boundary" checkpoints the
/// snapshot store takes (every capture point) are just saved values of this
/// running state, not a separate algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChainHash(u64);

impl Default for ChainHash {
    fn default() -> Self {
        Self(FNV_OFFSET)
    }
}

impl ChainHash {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn value(self) -> u64 {
        self.0
    }

    /// Folds one more token id into the running hash.
    pub fn fold(mut self, token_id: u32) -> Self {
        for byte in token_id.to_le_bytes() {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(FNV_PRIME);
        }
        self
    }

    pub fn fold_all(self, token_ids: &[u32]) -> Self {
        token_ids.iter().fold(self, |h, &t| h.fold(t))
    }

    /// Hashes `token_ids` from scratch — the non-incremental entry point
    /// callers reach for when they don't already hold a shorter prefix's
    /// checkpoint (e.g. hashing an arbitrary candidate prefix length during
    /// lookup).
    pub fn of(token_ids: &[u32]) -> u64 {
        Self::new().fold_all(token_ids).value()
    }
}

/// FNV-1a over raw bytes — used as the disk tier's corruption-detection
/// checksum. A different hash instance from [`ChainHash`] only because the
/// input unit differs (bytes vs. u32 token ids); same algorithm.
pub fn checksum_bytes(data: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in data {
        h ^= u64::from(b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_hash_is_fnv_offset_basis() {
        assert_eq!(ChainHash::new().value(), FNV_OFFSET);
    }

    #[test]
    fn resuming_from_a_checkpoint_matches_hashing_from_scratch() {
        let tokens: Vec<u32> = (0..500).map(|i| i * 7 + 3).collect();
        let checkpoint = ChainHash::new().fold_all(&tokens[..200]);
        let resumed = checkpoint.fold_all(&tokens[200..350]).value();
        let from_scratch = ChainHash::of(&tokens[..350]);
        assert_eq!(resumed, from_scratch);
    }

    #[test]
    fn different_token_sequences_hash_differently() {
        assert_ne!(ChainHash::of(&[1, 2, 3]), ChainHash::of(&[1, 2, 4]));
        assert_ne!(ChainHash::of(&[1, 2, 3]), ChainHash::of(&[1, 2]));
    }

    #[test]
    fn same_prefix_different_length_never_collides_trivially() {
        // A prefix must never hash equal to a longer prefix that starts with
        // it — otherwise longest-prefix lookup could pick the wrong length.
        let base = ChainHash::of(&[10, 20, 30]);
        let extended = ChainHash::of(&[10, 20, 30, 0]);
        assert_ne!(base, extended);
    }

    #[test]
    fn checksum_detects_a_single_flipped_bit() {
        let data = b"snapshot payload bytes".to_vec();
        let good = checksum_bytes(&data);
        let mut corrupted = data.clone();
        corrupted[5] ^= 0x01;
        assert_ne!(good, checksum_bytes(&corrupted));
    }
}
