//! Pure, analytically-computed byte/FLOP formulas for each op kind — no
//! device access, no timing. These feed `Profiler::time`'s `bytes`/`flops`
//! arguments; every formula counts a *lower bound* (the dominant memory
//! traffic / multiply-adds a memory-bound gemv kernel does), not a
//! cycle-exact model, since the point is roofline positioning, not exact
//! kernel accounting.

/// f32 element size, used throughout: every scratch buffer in the forward
/// pass (q/k/v, scores, ffn intermediates, KV cache) is f32.
const F32: u64 = 4;

/// `y = W * x`, `W` is `m x n`: `weight_bytes` is the exact on-device size of
/// `W` (quantized or f16 — `LinearWeight::byte_size` reports it precisely,
/// so this doesn't need to guess bits-per-weight), plus the `n`-wide input
/// read and `m`-wide f32 output write. Weight bytes dominate for any
/// realistic `m`/`n`, which is exactly why decode-time gemv is memory-bound.
pub fn matvec_bytes(weight_bytes: u64, m: u32, n: u32) -> u64 {
    weight_bytes + (n as u64 + m as u64) * F32
}

/// One multiply-add per weight element.
pub fn matvec_flops(m: u32, n: u32) -> u64 {
    2 * m as u64 * n as u64
}

/// `rmsnorm_f32(x, weight, out, rows, n, eps)`: reads `x` and `weight`,
/// writes `out` (a fresh buffer or in place — either way this many bytes
/// cross the memory system), each `rows * n` elements once.
pub fn norm_bytes(rows: u32, n: u32) -> u64 {
    let elems = rows as u64 * n as u64;
    (2 * elems + n as u64) * F32 // x read + out write (rows*n each) + weight read (n, shared across rows)
}

/// Sum of squares (mul+add per element) + the normalize-and-scale pass
/// (mul+mul per element): ~4 FLOPs/element, not counting the single
/// `sqrt`/reciprocal per row (negligible next to `rows * n` elementwise work
/// for any n worth profiling).
pub fn norm_flops(rows: u32, n: u32) -> u64 {
    4 * rows as u64 * n as u64
}

/// `embedding_f16_f32`: one row read from the f16 table, one f32 row
/// written. Pure copy, no arithmetic — flops is 0 by construction.
pub fn embed_bytes(hidden: u32) -> u64 {
    hidden as u64 * 2 + hidden as u64 * F32
}

/// The fused `attn_decode` kernel (see `Kernels::attn_decode`): K/V are read
/// once per `(kv head, split)` workgroup and reused in LDS across the whole
/// GQA group sharing that kv head, so the dominant traffic scales with
/// `n_kv_heads`, not `n_heads` — the actual saving the LDS-tiling design
/// buys over a naive per-q-head read. The split-K reduce pass adds a
/// `partial_out` write + re-read (`[n_heads, n_splits, head_dim]`, unlike
/// K/V this genuinely is `n_heads`-wide, since each q head has its own
/// unnormalized accumulator) plus the final `[n_heads, head_dim]` output
/// write.
pub fn attn_decode_bytes(
    n_heads: u32,
    n_kv_heads: u32,
    cur_len: u32,
    head_dim: u32,
    n_splits: u32,
) -> u64 {
    let kv_read = 2 * n_kv_heads as u64 * cur_len as u64 * head_dim as u64 * F32; // K + V, shared across the GQA group
    let partial_rw = 2 * n_heads as u64 * n_splits as u64 * head_dim as u64 * F32; // partial write + reduce's read-back
    let out_write = n_heads as u64 * head_dim as u64 * F32;
    kv_read + partial_rw + out_write
}

/// One multiply-add per (head, cached position, head_dim element), for both
/// the score dot product and the weighted-V accumulation — unlike bytes,
/// this doesn't shrink with GQA sharing: every q head still does its own
/// full dot product against every cached position, only the K/V *read* is
/// shared.
pub fn attn_decode_flops(n_heads: u32, cur_len: u32, head_dim: u32) -> u64 {
    4 * n_heads as u64 * cur_len as u64 * head_dim as u64
}

/// GDN's causal depthwise conv1d over the fused `[Q|K|V]` projection: each of
/// `conv_dim` channels reads `conv_kernel` state/weight taps and writes one
/// output element.
pub fn gdn_conv_bytes(conv_dim: u32, conv_kernel: u32) -> u64 {
    conv_dim as u64 * (conv_kernel as u64 + 1) * F32
}

pub fn gdn_conv_flops(conv_dim: u32, conv_kernel: u32) -> u64 {
    2 * conv_dim as u64 * conv_kernel as u64
}

/// The gated delta-rule recurrence: each of `num_v_heads` heads holds a
/// `head_k_dim x head_v_dim` state matrix, read once and written back once
/// per decode step (the dominant traffic — q/k/v/beta/g inputs are one
/// `head_k_dim`- or `head_v_dim`-wide vector per head, negligible next to
/// the full state matrix).
pub fn gdn_recur_bytes(num_v_heads: u32, head_k_dim: u32, head_v_dim: u32) -> u64 {
    let state_elems = num_v_heads as u64 * head_k_dim as u64 * head_v_dim as u64;
    state_elems * F32 * 2 // read + write
}

/// Delta-rule update per state element: decay multiply, outer-product term
/// (key x delta), subtract old value contribution, add new — ~4 FLOPs/elem —
/// plus the read-out dot product against the query (~2 FLOPs/elem more).
pub fn gdn_recur_flops(num_v_heads: u32, head_k_dim: u32, head_v_dim: u32) -> u64 {
    let state_elems = num_v_heads as u64 * head_k_dim as u64 * head_v_dim as u64;
    state_elems * 6
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matvec_bytes_is_dominated_by_weight_bytes() {
        // A realistic FFN-down shape: m=hidden=2048, n=ffn=6144, Q6_K-ish
        // weight bytes already known exactly from the loader.
        let weight_bytes = 2048u64 * 6144 * 6 / 8; // ~6 bits/weight, illustrative
        let bytes = matvec_bytes(weight_bytes, 2048, 6144);
        assert!(
            bytes > weight_bytes,
            "must include vector I/O on top of weights"
        );
        assert!(
            bytes - weight_bytes == (2048 + 6144) * 4,
            "vector I/O term should be exactly (m+n)*4 bytes"
        );
    }

    #[test]
    fn matvec_flops_is_two_mults_per_weight() {
        assert_eq!(matvec_flops(100, 50), 2 * 100 * 50);
        assert_eq!(matvec_flops(0, 50), 0);
    }

    #[test]
    fn attn_decode_scales_with_kv_cache_depth() {
        let shallow = attn_decode_bytes(8, 8, 16, 128, 1);
        let deep = attn_decode_bytes(8, 8, 4096, 128, 1);
        assert!(deep > shallow, "deeper KV cache must read more bytes");
        // The K/V-read term scales linearly with cur_len (a 256x jump,
        // 4096/16), but partial_rw/out_write don't (they're per-head, not
        // per-position), so the *overall* ratio is somewhat below 256x —
        // just assert it's still in that ballpark, not an exact multiple.
        let ratio = deep as f64 / shallow as f64;
        assert!(
            (200.0..=256.0).contains(&ratio),
            "expected a ratio near but at most 256x (4096/16), got {ratio}"
        );
    }

    #[test]
    fn attn_decode_bytes_shrinks_with_gqa_sharing() {
        // Same n_heads/cur_len/head_dim, fewer kv heads (more sharing)
        // should read fewer bytes thanks to the LDS-tiled K/V reuse across
        // a kv head's whole q-head group.
        let shared = attn_decode_bytes(32, 8, 2048, 256, 1);
        let unshared = attn_decode_bytes(32, 32, 2048, 256, 1);
        assert!(
            shared < unshared,
            "GQA sharing (fewer kv heads) must read fewer bytes"
        );
    }

    #[test]
    fn gdn_recur_bytes_is_state_read_plus_write() {
        let bytes = gdn_recur_bytes(32, 128, 128);
        let state_elems = 32u64 * 128 * 128;
        assert_eq!(bytes, state_elems * 4 * 2);
    }

    #[test]
    fn norm_bytes_scales_with_rows() {
        let one_row = norm_bytes(1, 128);
        let many_rows = norm_bytes(8, 128);
        // Weight read is shared (not multiplied by rows), so scaling isn't
        // exactly 8x, but must still be more than 1 row's worth.
        assert!(many_rows > one_row);
    }
}
