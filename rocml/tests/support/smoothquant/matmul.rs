//! Plain CPU `X @ W^T` for issue #17's standalone measurement — the
//! reference and every quantized/smoothed variant it's compared against all
//! go through this one function so the comparison is apples-to-apples.
//! Parallelized over output rows via `std::thread::scope` (no new
//! dependency; this repo has no `rayon`) since a few full `rows x m x n`
//! passes at Ornith's real shapes (n up to 12288) would otherwise take
//! minutes each in a single scalar thread.

/// `Y[o*rows + r] = sum_j X[r*n+j] * W[o*n+j]` — output-row-major layout
/// (not `[rows, m]`): this is a measurement helper compared only against
/// itself, so the layout only needs to be internally consistent, not match
/// any real kernel's memory contract.
pub fn matmul_xwt_omajor(x: &[f32], rows: usize, n: usize, w: &[f32], m: usize) -> Vec<f32> {
    assert_eq!(x.len(), rows * n);
    assert_eq!(w.len(), m * n);
    let mut y = vec![0f32; m * rows];
    let threads = std::thread::available_parallelism()
        .map(|t| t.get())
        .unwrap_or(4)
        .min(m.max(1));
    let chunk_rows = m.div_ceil(threads.max(1)).max(1);
    std::thread::scope(|scope| {
        for (t, y_chunk) in y.chunks_mut(chunk_rows * rows).enumerate() {
            let o_start = t * chunk_rows;
            scope.spawn(move || {
                for (local_o, y_row) in y_chunk.chunks_mut(rows).enumerate() {
                    let o = o_start + local_o;
                    let w_row = &w[o * n..(o + 1) * n];
                    for (r, y_val) in y_row.iter_mut().enumerate() {
                        let x_row = &x[r * n..(r + 1) * n];
                        let mut acc = 0f32;
                        for j in 0..n {
                            acc += x_row[j] * w_row[j];
                        }
                        *y_val = acc;
                    }
                }
            });
        }
    });
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_naive_reference_on_a_tiny_shape() {
        let rows = 3;
        let n = 4;
        let m = 5;
        let x: Vec<f32> = (0..rows * n).map(|i| i as f32 * 0.1 - 1.0).collect();
        let w: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.37).sin()).collect();
        let y = matmul_xwt_omajor(&x, rows, n, &w, m);
        for o in 0..m {
            for r in 0..rows {
                let expected: f32 = (0..n).map(|j| x[r * n + j] * w[o * n + j]).sum();
                let got = y[o * rows + r];
                assert!(
                    (got - expected).abs() < 1e-4,
                    "o={o} r={r}: {got} vs {expected}"
                );
            }
        }
    }
}
