//! Calibration sidecar: the pairing scheme, per-pair angles, and per-bpw
//! Lloyd-Max codebooks needed to round-trip a real K or V vector — issue
//! #16's "architecture-generic" rule keyed this by tensor kind (K vs V),
//! sized off `head_dim` at load time rather than a hardcoded 256. Built
//! offline by `make rotational-kv-calibrate` (`rocml/tests/rotational_kv_calibrate.rs`)
//! from real captured Ornith-1.0-9B K/V planes and checked in as
//! `rocml/data/rotational_kv_calibration.json`; loaded here via
//! `include_str!` so the debug simulation path (issue #14 phase 2 step 3)
//! never depends on a runtime file path.
//!
//! One sidecar entry per tensor kind today, not per layer: issue #2's own
//! per-head error measurement found quantization error is driven by layer
//! *depth*, not a fixed per-layer/per-head signature worth calibrating
//! separately at this decision-gate stage — see `.claude/CLAUDE.md`'s
//! "Per-head boundary-skip measurement" section. `RotSidecarKey` still
//! carries an optional `layer` field so a future round can add per-layer
//! entries without changing this shape.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::lloyd::LloydMaxCodebook;
use super::rotate::{forward_rotate, inverse_rotate, l2_norm, Pairing};
use crate::error::RocmlError;

/// Which tensor this calibration applies to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TensorKind {
    K,
    V,
}

/// Calibration for one tensor kind: pairing + per-pair angles (`len ==
/// head_dim/2`) + one Lloyd-Max codebook per supported bit width.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorSidecar {
    pub head_dim: usize,
    pub pairing: Pairing,
    pub angles: Vec<f32>,
    /// bpw (2/3/4) -> trained codebook.
    pub codebooks: BTreeMap<u8, LloydMaxCodebook>,
}

impl TensorSidecar {
    /// Rotates -> L2-normalizes (norm stored/rounded at f16 precision,
    /// matching the production design) -> quantizes each coordinate through
    /// the `bpw` codebook -> dequantizes -> denormalizes -> inverse-rotates.
    /// Returns the round-tripped reconstruction, the same length as `v`.
    pub fn round_trip(&self, v: &[f32], bpw: u8) -> Result<Vec<f32>, RocmlError> {
        if v.len() != self.head_dim {
            return Err(RocmlError::Config(format!(
                "rotational KV sidecar: vector length {} != calibrated head_dim {}",
                v.len(),
                self.head_dim
            )));
        }
        let codebook = self.codebooks.get(&bpw).ok_or_else(|| {
            RocmlError::Config(format!(
                "rotational KV sidecar: no codebook trained for {bpw} bpw (have {:?})",
                self.codebooks.keys().collect::<Vec<_>>()
            ))
        })?;

        let mut work = v.to_vec();
        forward_rotate(&mut work, self.pairing, &self.angles)?;

        let norm = l2_norm(&work);
        let norm_f16 = half::f16::from_f32(norm).to_f32();
        if norm_f16 > 0.0 {
            for x in &mut work {
                *x /= norm_f16;
            }
        }
        for x in &mut work {
            *x = codebook.round_trip(*x);
        }
        if norm_f16 > 0.0 {
            for x in &mut work {
                *x *= norm_f16;
            }
        }
        inverse_rotate(&mut work, self.pairing, &self.angles)?;
        Ok(work)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RotSidecar {
    pub k: TensorSidecar,
    pub v: TensorSidecar,
}

impl RotSidecar {
    pub fn for_kind(&self, kind: TensorKind) -> &TensorSidecar {
        match kind {
            TensorKind::K => &self.k,
            TensorKind::V => &self.v,
        }
    }

    pub fn to_json(&self) -> Result<String, RocmlError> {
        serde_json::to_string_pretty(self)
            .map_err(|e| RocmlError::Config(format!("rotational KV sidecar: serialize: {e}")))
    }

    pub fn from_json(s: &str) -> Result<Self, RocmlError> {
        serde_json::from_str(s)
            .map_err(|e| RocmlError::Config(format!("rotational KV sidecar: parse: {e}")))
    }
}

/// The checked-in calibration sidecar, embedded at compile time — see this
/// module's doc comment for why a debug simulation flag never wants a
/// runtime file dependency. Built for Ornith-1.0-9B's `head_dim` (256); a
/// checkpoint with a different `head_dim` fails `round_trip` cleanly (a
/// `RocmlError::Config`, not a panic) rather than misapplying angles sized
/// for the wrong dimension — re-running `make rotational-kv-calibrate`
/// against that checkpoint regenerates this file for its own `head_dim`.
const DEFAULT_SIDECAR_JSON: &str = include_str!("../../../data/rotational_kv_calibration.json");

pub fn default_sidecar() -> Result<RotSidecar, RocmlError> {
    RotSidecar::from_json(DEFAULT_SIDECAR_JSON)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn toy_sidecar(head_dim: usize) -> TensorSidecar {
        let pairing = Pairing::SplitHalf;
        let angles = super::super::rotate::fixed_angles(head_dim);
        let data: Vec<f32> = (0..2000)
            .map(|i| ((i as f32 * 0.017).sin()) * 0.4)
            .collect();
        let mut codebooks = BTreeMap::new();
        for bpw in [2u8, 3, 4] {
            codebooks.insert(bpw, LloydMaxCodebook::train(&data, 1usize << bpw).unwrap());
        }
        TensorSidecar {
            head_dim,
            pairing,
            angles,
            codebooks,
        }
    }

    #[test]
    fn round_trip_preserves_length_and_is_finite() {
        let ts = toy_sidecar(16);
        let v: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.3).collect();
        let out = ts.round_trip(&v, 3).unwrap();
        assert_eq!(out.len(), v.len());
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn round_trip_rejects_wrong_length() {
        let ts = toy_sidecar(16);
        let v = vec![0f32; 8];
        assert!(ts.round_trip(&v, 3).is_err());
    }

    #[test]
    fn round_trip_rejects_missing_bpw() {
        let ts = toy_sidecar(16);
        let v = vec![0.1f32; 16];
        assert!(ts.round_trip(&v, 5).is_err());
    }

    #[test]
    fn json_round_trip_preserves_content() {
        let sidecar = RotSidecar {
            k: toy_sidecar(8),
            v: toy_sidecar(8),
        };
        let json = sidecar.to_json().unwrap();
        let back = RotSidecar::from_json(&json).unwrap();
        assert_eq!(back.k.angles, sidecar.k.angles);
        assert_eq!(back.v.codebooks.len(), sidecar.v.codebooks.len());
    }

    #[test]
    fn embedded_default_sidecar_parses() {
        let sidecar = default_sidecar().expect("checked-in calibration sidecar must parse");
        assert!(sidecar.k.head_dim > 0);
        assert_eq!(sidecar.k.angles.len(), sidecar.k.head_dim / 2);
        assert_eq!(sidecar.v.angles.len(), sidecar.v.head_dim / 2);
        for bpw in [2u8, 3, 4] {
            assert!(sidecar.k.codebooks.contains_key(&bpw));
            assert!(sidecar.v.codebooks.contains_key(&bpw));
        }
    }
}
