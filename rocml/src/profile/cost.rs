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

/// The causal-decode score pass (`gemv_f32` once per head against that
/// head's slice of the KV cache, then `softmax_varlen`): each of `n_heads`
/// heads reads `cur_len * head_dim` cached K values and writes `cur_len`
/// scores, softmax re-reads/writes those same scores once more.
pub fn attn_score_bytes(n_heads: u32, cur_len: u32, head_dim: u32) -> u64 {
    let k_read = n_heads as u64 * cur_len as u64 * head_dim as u64 * F32;
    let scores_rw = n_heads as u64 * cur_len as u64 * F32 * 3; // gemv write + softmax read + softmax write
    k_read + scores_rw
}

/// One multiply-add per (head, cached position, head_dim element) score dot.
pub fn attn_score_flops(n_heads: u32, cur_len: u32, head_dim: u32) -> u64 {
    2 * n_heads as u64 * cur_len as u64 * head_dim as u64
}

/// The weighted-V pass (`gemv_t_f32` once per head against that head's V
/// plane): same shape as the score pass, reading V instead of K.
pub fn attn_out_bytes(n_heads: u32, cur_len: u32, head_dim: u32) -> u64 {
    let v_read = n_heads as u64 * cur_len as u64 * head_dim as u64 * F32;
    let out_write = n_heads as u64 * head_dim as u64 * F32;
    v_read + out_write
}

pub fn attn_out_flops(n_heads: u32, cur_len: u32, head_dim: u32) -> u64 {
    2 * n_heads as u64 * cur_len as u64 * head_dim as u64
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
    fn attn_score_scales_with_kv_cache_depth() {
        let shallow = attn_score_bytes(8, 16, 128);
        let deep = attn_score_bytes(8, 4096, 128);
        assert!(deep > shallow, "deeper KV cache must read more bytes");
        // K-read term should scale linearly with cur_len.
        let ratio = deep as f64 / shallow as f64;
        assert!(
            (ratio - 256.0).abs() < 5.0,
            "expected ~256x (4096/16) scaling, got {ratio}"
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
