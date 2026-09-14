//! Lloyd-Max scalar quantizer — issue #14's precomputed centroid LUT for
//! post-rotation, unit-normalized coordinates. After a block-diagonal
//! rotation and per-vector L2 normalization, each coordinate's marginal
//! distribution is concentrated near zero with bounded support (roughly
//! Beta-shaped, per the issue's plan) rather than uniform, so a fixed-step
//! quantizer wastes levels in the tails; Lloyd-Max instead places centroids
//! to minimize expected squared error against the *measured* calibration
//! distribution.
//!
//! Classic Lloyd iteration: given `n_levels` centroids, repeatedly (1)
//! partition calibration samples into the nearest-centroid bins via the
//! midpoint boundaries, (2) recompute each centroid as the mean of its
//! bin's samples, until convergence or a fixed iteration cap. A shared
//! codebook (not one per channel/pair) is trained per tensor kind (K or V)
//! by pooling every rotated, normalized coordinate together — the point of
//! the rotation is exactly to make every coordinate's marginal
//! interchangeable, so one shared 1-D LUT is the intended design, not a
//! simplification that loses information a per-channel codebook would keep.

use serde::{Deserialize, Serialize};

use crate::error::RocmlError;

const MAX_ITERS: usize = 50;
/// Stop early once no centroid moves more than this between iterations.
const CONVERGENCE_EPS: f32 = 1e-6;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LloydMaxCodebook {
    /// Sorted ascending centroid values, `2^bpw` of them.
    pub levels: Vec<f32>,
}

impl LloydMaxCodebook {
    /// Trains a codebook with `n_levels` centroids (`n_levels` a power of
    /// two, e.g. 4/8/16 for 2/3/4 bpw) from `data`. Initializes centroids at
    /// evenly-spaced percentiles of the sorted data (a k-means++-adjacent
    /// start that avoids the classic "all centroids collapse to the mean"
    /// failure of a naive uniform-range init on skewed data).
    pub fn train(data: &[f32], n_levels: usize) -> Result<Self, RocmlError> {
        if data.is_empty() {
            return Err(RocmlError::Config(
                "Lloyd-Max training: empty calibration data".to_string(),
            ));
        }
        if n_levels == 0 {
            return Err(RocmlError::Config(
                "Lloyd-Max training: n_levels must be positive".to_string(),
            ));
        }
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mut levels: Vec<f32> = (0..n_levels)
            .map(|i| {
                let frac = (i as f32 + 0.5) / n_levels as f32;
                let idx = ((frac * sorted.len() as f32) as usize).min(sorted.len() - 1);
                sorted[idx]
            })
            .collect();
        levels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        for _ in 0..MAX_ITERS {
            let boundaries = midpoints(&levels);
            let mut sums = vec![0f64; n_levels];
            let mut counts = vec![0u64; n_levels];
            for &x in data {
                let bin = bin_index(&boundaries, x);
                sums[bin] += x as f64;
                counts[bin] += 1;
            }
            let mut max_move = 0f32;
            let mut new_levels = levels.clone();
            for i in 0..n_levels {
                if counts[i] > 0 {
                    let new_val = (sums[i] / counts[i] as f64) as f32;
                    max_move = max_move.max((new_val - levels[i]).abs());
                    new_levels[i] = new_val;
                }
                // An empty bin keeps its previous centroid — re-seeding a
                // dead centroid is unnecessary here since the init above
                // already spreads centroids across occupied percentiles.
            }
            new_levels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            levels = new_levels;
            if max_move < CONVERGENCE_EPS {
                break;
            }
        }
        Ok(Self { levels })
    }

    /// Nearest-centroid index for `x` (the "code" a real encoder would
    /// store — `ceil(log2(levels.len()))` bits).
    pub fn quantize_index(&self, x: f32) -> usize {
        let boundaries = midpoints(&self.levels);
        bin_index(&boundaries, x)
    }

    pub fn dequantize(&self, idx: usize) -> f32 {
        self.levels[idx.min(self.levels.len() - 1)]
    }

    /// Round-trips `x` through quantize+dequantize in one call.
    pub fn round_trip(&self, x: f32) -> f32 {
        self.dequantize(self.quantize_index(x))
    }

    pub fn bpw(&self) -> f32 {
        (self.levels.len() as f32).log2()
    }
}

/// `n-1` midpoints between `n` sorted levels — the decision boundaries a
/// nearest-centroid search partitions on.
fn midpoints(levels: &[f32]) -> Vec<f32> {
    levels.windows(2).map(|w| (w[0] + w[1]) / 2.0).collect()
}

/// Which bin `x` falls into given `n-1` ascending boundaries (`n` bins).
fn bin_index(boundaries: &[f32], x: f32) -> usize {
    boundaries.iter().filter(|&&b| x > b).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gaussian_like(n: usize, seed: u32) -> Vec<f32> {
        // Sum of uniforms (crude CLT approximation) — enough to exercise a
        // non-uniform, tail-concentrated distribution without pulling in a
        // real RNG/stats crate.
        (0..n)
            .map(|i| {
                let mut acc = 0f32;
                for k in 0..6 {
                    let x = (i as u32)
                        .wrapping_mul(2654435761)
                        .wrapping_add(seed)
                        .wrapping_add(k * 7919);
                    acc += (x >> 8) as f32 / u32::MAX as f32;
                }
                acc / 6.0 - 0.5
            })
            .collect()
    }

    #[test]
    fn trained_codebook_has_requested_level_count_and_is_sorted() {
        let data = gaussian_like(2000, 1);
        for &bpw in &[2usize, 3, 4] {
            let cb = LloydMaxCodebook::train(&data, 1 << bpw).unwrap();
            assert_eq!(cb.levels.len(), 1 << bpw);
            assert!(cb.levels.windows(2).all(|w| w[0] <= w[1]));
        }
    }

    #[test]
    fn lloyd_max_beats_uniform_quantization_on_skewed_data() {
        let data = gaussian_like(4000, 2);
        let cb = LloydMaxCodebook::train(&data, 8).unwrap();
        let lm_mse: f64 = data
            .iter()
            .map(|&x| {
                let e = (x - cb.round_trip(x)) as f64;
                e * e
            })
            .sum::<f64>()
            / data.len() as f64;

        // A naive fixed-step uniform quantizer over the data's full range.
        let (lo, hi) = (
            data.iter().cloned().fold(f32::MAX, f32::min),
            data.iter().cloned().fold(f32::MIN, f32::max),
        );
        let step = (hi - lo) / 8.0;
        let uniform_mse: f64 = data
            .iter()
            .map(|&x| {
                let bin = ((x - lo) / step).floor().clamp(0.0, 7.0);
                let center = lo + (bin + 0.5) * step;
                let e = (x - center) as f64;
                e * e
            })
            .sum::<f64>()
            / data.len() as f64;

        assert!(
            lm_mse <= uniform_mse,
            "Lloyd-Max MSE {lm_mse} should be <= uniform MSE {uniform_mse}"
        );
    }

    #[test]
    fn round_trip_is_deterministic_and_stable() {
        let data = gaussian_like(500, 3);
        let cb = LloydMaxCodebook::train(&data, 4).unwrap();
        for &x in &data {
            let idx = cb.quantize_index(x);
            assert_eq!(idx, cb.quantize_index(x));
            assert!(idx < cb.levels.len());
        }
    }

    #[test]
    fn rejects_empty_data() {
        assert!(LloydMaxCodebook::train(&[], 4).is_err());
    }
}
