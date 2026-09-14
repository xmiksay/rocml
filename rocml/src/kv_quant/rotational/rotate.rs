//! Block-diagonal 2D Givens rotation — issue #14's chosen approach over a
//! global Walsh-Hadamard transform (see the issue body for the RDNA3-shuffle
//! rationale). A `head_dim`-length vector is split into `head_dim/2`
//! independent channel pairs, each rotated by its own angle; the whole
//! transform is exactly orthogonal (a block-diagonal matrix of 2x2 rotation
//! blocks), so it preserves dot products (attention math is unaffected) and
//! inverts exactly by negating every angle.

use serde::{Deserialize, Serialize};

use crate::error::RocmlError;

/// Which two channels form each pair. `head_dim` must be even for either
/// scheme (checked by [`Pairing::pairs`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Pairing {
    /// `(2i, 2i+1)` — neighboring channels.
    Adjacent,
    /// `(i, i + head_dim/2)` — a channel paired with its counterpart in the
    /// other half of the head. TurboQuant/PlanarQuant's own convention.
    SplitHalf,
}

impl Pairing {
    /// The `head_dim/2` `(a, b)` channel-index pairs for this scheme.
    pub fn pairs(self, head_dim: usize) -> Result<Vec<(usize, usize)>, RocmlError> {
        if head_dim == 0 || !head_dim.is_multiple_of(2) {
            return Err(RocmlError::Config(format!(
                "rotational KV: head_dim must be a positive even number, got {head_dim}"
            )));
        }
        let half = head_dim / 2;
        Ok(match self {
            Pairing::Adjacent => (0..half).map(|i| (2 * i, 2 * i + 1)).collect(),
            Pairing::SplitHalf => (0..half).map(|i| (i, i + half)).collect(),
        })
    }
}

/// Rotates `v` in place by `angles[p]` on pairing `pairing`'s `p`-th pair.
/// `angles.len()` must equal `head_dim/2`.
pub fn forward_rotate(v: &mut [f32], pairing: Pairing, angles: &[f32]) -> Result<(), RocmlError> {
    apply_rotation(v, pairing, angles, 1.0)
}

/// The exact inverse of [`forward_rotate`] — a rotation matrix's inverse is
/// its transpose, which for a 2x2 Givens block is just negating the angle.
pub fn inverse_rotate(v: &mut [f32], pairing: Pairing, angles: &[f32]) -> Result<(), RocmlError> {
    apply_rotation(v, pairing, angles, -1.0)
}

fn apply_rotation(
    v: &mut [f32],
    pairing: Pairing,
    angles: &[f32],
    sign: f32,
) -> Result<(), RocmlError> {
    let pairs = pairing.pairs(v.len())?;
    if angles.len() != pairs.len() {
        return Err(RocmlError::Config(format!(
            "rotational KV: {} angles for {} pairs (head_dim {})",
            angles.len(),
            pairs.len(),
            v.len()
        )));
    }
    for (&(a, b), &theta) in pairs.iter().zip(angles) {
        let theta = sign * theta;
        let (c, s) = (theta.cos(), theta.sin());
        let (va, vb) = (v[a], v[b]);
        v[a] = va * c - vb * s;
        v[b] = va * s + vb * c;
    }
    Ok(())
}

/// Per-pair calibration angle that diagonalizes that pair's 2x2 covariance
/// matrix — equalizes variance and zeroes cross-correlation between the two
/// channels, per issue #14's "equalize variance on calibration KV" choice.
/// Closed form: `angle = 0.5 * atan2(2*cov(a,b), var(a) - var(b))`. `calib`
/// is `n_vecs` row-major vectors of length `head_dim`.
pub fn calibrate_angles(
    calib: &[f32],
    n_vecs: usize,
    head_dim: usize,
    pairing: Pairing,
) -> Result<Vec<f32>, RocmlError> {
    if calib.len() != n_vecs * head_dim {
        return Err(RocmlError::Config(format!(
            "rotational KV calibration: expected {} values ({n_vecs} x {head_dim}), got {}",
            n_vecs * head_dim,
            calib.len()
        )));
    }
    let pairs = pairing.pairs(head_dim)?;
    let mut angles = Vec::with_capacity(pairs.len());
    for (a, b) in pairs {
        let (mut mean_a, mut mean_b) = (0f64, 0f64);
        for row in 0..n_vecs {
            mean_a += calib[row * head_dim + a] as f64;
            mean_b += calib[row * head_dim + b] as f64;
        }
        mean_a /= n_vecs.max(1) as f64;
        mean_b /= n_vecs.max(1) as f64;

        let (mut var_a, mut var_b, mut cov) = (0f64, 0f64, 0f64);
        for row in 0..n_vecs {
            let da = calib[row * head_dim + a] as f64 - mean_a;
            let db = calib[row * head_dim + b] as f64 - mean_b;
            var_a += da * da;
            var_b += db * db;
            cov += da * db;
        }
        let angle = 0.5 * (2.0 * cov).atan2(var_a - var_b);
        angles.push(angle as f32);
    }
    Ok(angles)
}

/// Fixed 45-degree angle for every pair — the cheap calibration-free
/// baseline issue #14 asks to measure alongside the calibrated angles above.
pub fn fixed_angles(head_dim: usize) -> Vec<f32> {
    vec![std::f32::consts::FRAC_PI_4; head_dim / 2]
}

pub fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_vec(seed: u32, len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let x = (i as u32).wrapping_add(seed).wrapping_mul(2654435761);
                ((x >> 8) as f32 / u32::MAX as f32 - 0.5) * 4.0
            })
            .collect()
    }

    #[test]
    fn rotation_is_exactly_orthogonal_dot_product_preserved() {
        let head_dim = 16;
        let a = sample_vec(1, head_dim);
        let b = sample_vec(2, head_dim);
        let angles: Vec<f32> = (0..head_dim / 2).map(|i| 0.3 + i as f32 * 0.1).collect();
        let dot_before: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();

        for pairing in [Pairing::Adjacent, Pairing::SplitHalf] {
            let mut ra = a.clone();
            let mut rb = b.clone();
            forward_rotate(&mut ra, pairing, &angles).unwrap();
            forward_rotate(&mut rb, pairing, &angles).unwrap();
            let dot_after: f32 = ra.iter().zip(&rb).map(|(x, y)| x * y).sum();
            assert!(
                (dot_before - dot_after).abs() < 1e-4,
                "{pairing:?}: dot product not preserved: {dot_before} vs {dot_after}"
            );
        }
    }

    #[test]
    fn inverse_rotate_exactly_undoes_forward_rotate() {
        let head_dim = 32;
        let v = sample_vec(7, head_dim);
        let angles = fixed_angles(head_dim);
        for pairing in [Pairing::Adjacent, Pairing::SplitHalf] {
            let mut work = v.clone();
            forward_rotate(&mut work, pairing, &angles).unwrap();
            inverse_rotate(&mut work, pairing, &angles).unwrap();
            for (got, want) in work.iter().zip(&v) {
                assert!((got - want).abs() < 1e-4, "{pairing:?}: {got} vs {want}");
            }
        }
    }

    #[test]
    fn rotation_preserves_l2_norm() {
        let head_dim = 8;
        let v = sample_vec(3, head_dim);
        let norm_before = l2_norm(&v);
        let angles = fixed_angles(head_dim);
        let mut work = v;
        forward_rotate(&mut work, Pairing::SplitHalf, &angles).unwrap();
        assert!((l2_norm(&work) - norm_before).abs() < 1e-4);
    }

    #[test]
    fn calibrated_angle_zeroes_pair_covariance() {
        // Two correlated channels: b = 2*a + noise. The calibrated angle
        // should rotate them into (near-)decorrelated coordinates.
        let n = 200;
        let mut calib = vec![0f32; n * 2];
        for i in 0..n {
            let a = sample_vec(i as u32, 1)[0];
            let noise = sample_vec(i as u32 + 999, 1)[0] * 0.05;
            calib[i * 2] = a;
            calib[i * 2 + 1] = 2.0 * a + noise;
        }
        let angles = calibrate_angles(&calib, n, 2, Pairing::Adjacent).unwrap();
        assert_eq!(angles.len(), 1);

        let mut rotated_cov = 0f64;
        let mut mean = (0f64, 0f64);
        let mut rows: Vec<(f32, f32)> = Vec::with_capacity(n);
        for i in 0..n {
            let mut pair = [calib[i * 2], calib[i * 2 + 1]];
            forward_rotate(&mut pair, Pairing::Adjacent, &angles).unwrap();
            mean.0 += pair[0] as f64;
            mean.1 += pair[1] as f64;
            rows.push((pair[0], pair[1]));
        }
        mean.0 /= n as f64;
        mean.1 /= n as f64;
        for (x, y) in rows {
            rotated_cov += (x as f64 - mean.0) * (y as f64 - mean.1);
        }
        rotated_cov /= n as f64;
        assert!(
            rotated_cov.abs() < 0.05,
            "expected near-zero covariance after calibrated rotation, got {rotated_cov}"
        );
    }

    #[test]
    fn rejects_odd_head_dim() {
        assert!(Pairing::Adjacent.pairs(7).is_err());
    }

    #[test]
    fn rejects_angle_count_mismatch() {
        let mut v = sample_vec(1, 8);
        assert!(forward_rotate(&mut v, Pairing::Adjacent, &[0.1, 0.2]).is_err());
    }
}
