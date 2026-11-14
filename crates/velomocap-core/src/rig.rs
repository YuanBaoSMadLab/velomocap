//! Skeleton bindings and the constant basis conversion used by BVH/FBX export.

use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::math::{mul3, transpose3};

/// A resolved export skeleton: names, hierarchy, rest geometry and the
/// per-bone reference basis that maps the internal rig into export space.
#[derive(Clone, Debug)]
pub struct Binding {
    pub names: Vec<String>,
    pub parents: Vec<i32>,
    /// `bones * 3` rest joint positions.
    pub rest: Vec<f64>,
    /// `bones * 3` terminal offsets for leaf bones.
    pub ends: Vec<f64>,
    /// `bones * 9` per-bone constant basis.
    pub basis: Vec<f64>,
    pub origin: [f64; 3],
    pub version: i32,
}

impl Binding {
    pub fn bones(&self) -> usize {
        self.parents.len()
    }

    pub fn validate(&self) -> Result<()> {
        let bones = self.bones();
        if bones == 0 {
            return Err(Error::new("Empty export skeleton"));
        }
        if self.rest.len() != bones * 3 || self.ends.len() != bones * 3 {
            return Err(Error::new("Invalid export rest geometry"));
        }
        if self.basis.len() != bones * 9 {
            return Err(Error::new("Invalid export basis dimensions"));
        }
        if self.names.len() != bones {
            return Err(Error::new("Invalid export bone name count"));
        }
        if self.parents[0] != -1 {
            return Err(Error::new("Export skeleton must start at the root"));
        }
        for j in 1..bones {
            let p = self.parents[j];
            if p < 0 || (p as usize) >= j {
                return Err(Error::new("Invalid export parent index"));
            }
        }
        if !self.rest.iter().all(|v| v.is_finite())
            || !self.ends.iter().all(|v| v.is_finite())
            || !self.basis.iter().all(|v| v.is_finite())
        {
            return Err(Error::new("Non-finite export skeleton data"));
        }
        for name in &self.names {
            if name.is_empty()
                || !name.is_ascii()
                || name.chars().any(|c| c.is_whitespace() || c == '{' || c == '}' || c == ':')
            {
                return Err(Error::new(format!("Invalid export bone name: {name:?}")));
            }
        }
        Ok(())
    }

    /// `rest[j] - rest[parent]` for every bone (root keeps its absolute offset).
    pub fn offsets(&self) -> Vec<[f64; 3]> {
        let bones = self.bones();
        let mut out = Vec::with_capacity(bones);
        for j in 0..bones {
            let p = self.parents[j];
            let (rx, ry, rz) = (self.rest[j * 3], self.rest[j * 3 + 1], self.rest[j * 3 + 2]);
            if p < 0 {
                out.push([rx, ry, rz]);
            } else {
                let p = p as usize;
                out.push([
                    rx - self.rest[p * 3],
                    ry - self.rest[p * 3 + 1],
                    rz - self.rest[p * 3 + 2],
                ]);
            }
        }
        out
    }
}

/// `local_frames`: `out[t][j] = basis[parent]^T @ local[t][j] @ basis[j]`.
///
/// This is the O(frames * bones) constant change of basis that every export
/// runs; it is parallel over frames and allocation free per chunk.
pub fn local_frames(local: &[f64], binding: &Binding, frames: usize) -> Vec<f64> {
    let bones = binding.bones();
    assert_eq!(local.len(), frames * bones * 9, "local frame dimensions");
    let basis_t: Vec<[f64; 9]> = (0..bones)
        .map(|j| transpose3(&binding.basis[j * 9..j * 9 + 9]))
        .collect();
    let mut out = vec![0.0; frames * bones * 9];
    out.par_chunks_mut(bones * 9).enumerate().for_each(|(t, chunk)| {
        let src = &local[t * bones * 9..(t + 1) * bones * 9];
        for j in 0..bones {
            let b = &binding.basis[j * 9..j * 9 + 9];
            let m = mul3(&src[j * 9..j * 9 + 9], b);
            let p = binding.parents[j];
            let m = if p < 0 { m } else { mul3(&basis_t[p as usize], &m) };
            chunk[j * 9..j * 9 + 9].copy_from_slice(&m);
        }
    });
    out
}

/// Transport local display vectors/normals through their bone basis.
pub fn asset_frames(packed: &mut [f64], bone_ids: &[usize], binding: &Binding) {
    let bones = binding.bones();
    for (i, &bone) in bone_ids.iter().enumerate() {
        let bone = bone.min(bones - 1);
        let inv = transpose3(&binding.basis[bone * 9..bone * 9 + 9]);
        for channel in 0..2 {
            let off = i * 6 + channel * 3;
            let v = [packed[off], packed[off + 1], packed[off + 2]];
            for row in 0..3 {
                packed[off + row] =
                    inv[row * 3] * v[0] + inv[row * 3 + 1] * v[1] + inv[row * 3 + 2] * v[2];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::IDENTITY3;

    fn identity_binding() -> Binding {
        let bones = 3;
        Binding {
            names: vec!["Hips".into(), "A".into(), "B".into()],
            parents: vec![-1, 0, 1],
            rest: vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0],
            ends: vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0],
            basis: IDENTITY3.repeat(bones),
            origin: [0.0; 3],
            version: 0,
        }
    }

    #[test]
    fn identity_basis_is_a_no_op() {
        let binding = identity_binding();
        let local: Vec<f64> = (0..3 * 9).map(|i| (i as f64) * 0.01).collect();
        let out = local_frames(&local, &binding, 1);
        for (a, b) in out.iter().zip(local.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }
}
