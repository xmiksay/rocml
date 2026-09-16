//! Pure-logic global LRU over a fixed pool of `capacity` slots, keyed by
//! `(layer, expert)`. No GPU/device code lives here — see
//! `super::moe_cache::ExpertCache` for the VRAM-backed pool this indexes
//! into. Kept separate specifically so the eviction/slot-reuse bookkeeping
//! (the part most likely to have an off-by-one) is unit-testable without a
//! GPU.
//!
//! Backed by the `lru` crate's O(1) `LruCache` for the recency ordering,
//! but that crate's own `put` silently evicts the least-recently-used entry
//! once at capacity without telling the caller which slot it freed — this
//! wrapper always calls `pop_lru()` itself first so it can hand the freed
//! slot straight to the new key, which is the whole point of a slot *pool*
//! (a fixed VRAM buffer per slot, not a dynamically-growing allocation).
//!
//! Global (not per-layer) by design — measured on real routing traces of
//! Ornith-1.5-35B-A3B (`.claude/CLAUDE.md`'s M2 section) to give a better
//! hit rate than a per-layer cache of the same total slot count.

use std::num::NonZeroUsize;

use lru::LruCache;

/// One expert's identity across every MoE layer in the model.
pub type ExpertKey = (u32, u32);

pub struct ExpertLru {
    capacity: usize,
    cache: LruCache<ExpertKey, usize>,
    /// Slots `[0, next_free)` have been handed out at least once; slots
    /// `>= next_free` have never been written, so a lookup that lands here
    /// takes a fresh slot instead of evicting anything.
    next_free: usize,
}

impl ExpertLru {
    /// `capacity` must be at least 1 — a capacity-0 cache is expressed as
    /// `None` at the `ExpertCache` level, never as an `ExpertLru` (see that
    /// module), so reaching here with `capacity == 0` is an internal
    /// caller bug, not a reachable user-facing state.
    pub fn new(capacity: usize) -> Self {
        let cap = NonZeroUsize::new(capacity)
            .expect("ExpertLru capacity must be >= 1 (caller must special-case 0)");
        Self {
            capacity,
            cache: LruCache::new(cap),
            next_free: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Slots currently holding a live entry (`<= capacity`) — used by
    /// `super::moe_cache::ExpertCache::occupancy` for bench/logging
    /// visibility into how full the cache is. Named `occupancy`, not `len`,
    /// to sidestep clippy's `len_without_is_empty` (an empty cache isn't a
    /// meaningful state this type needs to expose).
    pub fn occupancy(&self) -> usize {
        self.cache.len()
    }

    /// Looks up `key`, promoting it to most-recently-used on a hit. On a
    /// miss, assigns a slot (a never-used one while any remain, else the
    /// globally least-recently-used entry's slot) and inserts `key` there.
    /// Returns `(slot, was_hit)`.
    pub fn get_or_insert(&mut self, key: ExpertKey) -> (usize, bool) {
        if let Some(&slot) = self.cache.get(&key) {
            return (slot, true);
        }
        let slot = if self.next_free < self.capacity {
            let s = self.next_free;
            self.next_free += 1;
            s
        } else {
            let (_evicted_key, evicted_slot) = self
                .cache
                .pop_lru()
                .expect("cache at capacity must hold at least one entry");
            evicted_slot
        };
        self.cache.put(key, slot);
        (slot, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_lookups_take_distinct_never_used_slots() {
        let mut lru = ExpertLru::new(4);
        let mut slots = Vec::new();
        for e in 0..4 {
            let (slot, hit) = lru.get_or_insert((0, e));
            assert!(!hit);
            slots.push(slot);
        }
        slots.sort_unstable();
        assert_eq!(slots, vec![0, 1, 2, 3]);
        assert_eq!(lru.occupancy(), 4);
    }

    #[test]
    fn repeated_key_is_a_hit_and_keeps_its_slot() {
        let mut lru = ExpertLru::new(4);
        let (slot0, hit0) = lru.get_or_insert((1, 5));
        assert!(!hit0);
        for _ in 0..10 {
            let (slot, hit) = lru.get_or_insert((1, 5));
            assert!(hit);
            assert_eq!(slot, slot0);
        }
        // Repeated hits must never consume additional never-used slots.
        assert_eq!(lru.occupancy(), 1);
    }

    #[test]
    fn eviction_picks_the_true_lru_entry_not_most_recent() {
        let mut lru = ExpertLru::new(2);
        let (slot_a, _) = lru.get_or_insert((0, 1)); // A
        let (slot_b, _) = lru.get_or_insert((0, 2)); // B
                                                     // Touch A again so B becomes the least-recently-used.
        lru.get_or_insert((0, 1));
        // Inserting a third key must evict B, not A.
        let (slot_c, hit_c) = lru.get_or_insert((0, 3));
        assert!(!hit_c);
        assert_eq!(slot_c, slot_b, "must reuse B's slot, not A's");
        // A must still be resident (its slot content is still valid).
        let (slot_a_again, hit_a) = lru.get_or_insert((0, 1));
        assert!(hit_a);
        assert_eq!(slot_a_again, slot_a);
    }

    #[test]
    fn global_key_distinguishes_same_expert_id_across_layers() {
        let mut lru = ExpertLru::new(4);
        let (slot_l0, _) = lru.get_or_insert((0, 7));
        let (slot_l1, _) = lru.get_or_insert((1, 7));
        assert_ne!(slot_l0, slot_l1);
        let (slot_l0_again, hit) = lru.get_or_insert((0, 7));
        assert!(hit);
        assert_eq!(slot_l0_again, slot_l0);
    }

    #[test]
    fn capacity_one_thrashes_correctly() {
        let mut lru = ExpertLru::new(1);
        let (s0, hit0) = lru.get_or_insert((0, 1));
        assert!(!hit0);
        let (s1, hit1) = lru.get_or_insert((0, 2));
        assert!(!hit1);
        assert_eq!(s0, s1, "single slot must be reused");
        let (_, hit0_again) = lru.get_or_insert((0, 1));
        assert!(!hit0_again, "evicted key must miss again");
    }

    #[test]
    fn filling_exactly_to_capacity_never_evicts() {
        let mut lru = ExpertLru::new(3);
        for e in 0..3 {
            lru.get_or_insert((9, e));
        }
        // All three must still be hits — capacity was never exceeded.
        for e in 0..3 {
            let (_, hit) = lru.get_or_insert((9, e));
            assert!(hit, "expert {e} should not have been evicted");
        }
    }
}
