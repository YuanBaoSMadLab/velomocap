//! Sole probes from the SMPL-H foot skin, ported from the reference's
//! `flow_correction.py::SoleGeometry` (RGB MoCap, MIT).
//!
//! The reference never uses joint positions as contact points. It skins a small
//! set of **surface vertices** on each foot and reduces them to two patches per
//! foot (heel side and toe side), which are the points `support_weights` then
//! blends:
//!
//! ```text
//! ids   = vertices with lbs[ankle] + lbs[toe] > 0.6        (foot skin only)
//! back  = ids with z <= median(z),  front = ids with z >= median(z)
//! proxy = the 24 lowest vertices of each half            (the sole surface)
//! probe = mean(proxy) with y forced to the patch minimum
//! ```
//!
//! Doing this in Rust means the contact point is a real sole point, not an
//! ankle-plus-guessed-heel, which is what the rolling-anchor and support-weight
//! maths assume.

use anyhow::Result;
use velomocap_body::{BodyModel, Skeleton};

/// One foot's two sole patches: back (heel side) then front (toe side).
pub type FootProbes = [[f32; 3]; 2];

/// Linear-interpolated quantile, the same convention numpy's quantile uses.
fn quantile(values: &[f32], q: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let position = q.clamp(0.0, 1.0) * (sorted.len() - 1) as f32;
    let low = position.floor() as usize;
    let high = (low + 1).min(sorted.len() - 1);
    let weight = position - low as f32;
    sorted[low] * (1.0 - weight) + sorted[high] * weight
}

pub struct SoleGeometry {
    /// Vertex indices of each of the four patches.
    patches: Vec<Vec<usize>>,
    /// Indices into the owning patch: the 24 sole-proxy vertices.
    proxies: Vec<Vec<usize>>,
    /// Every selected vertex, for the skinning pass.
    ids: Vec<usize>,
    /// `ids * 22` skinning weights (hand joints cannot influence a sole).
    weights: Vec<f32>,
    /// `ids * 3` shaped rest positions.
    shaped: Vec<f32>,
}

impl SoleGeometry {
    /// Build the probes for a given shape. Fails when the foot skin is missing.
    pub fn build(body: &BodyModel, betas: &[f32]) -> Result<Self> {
        let weights = body.lbs_weights();
        let vertices = body.num_vertices;
        let joints = body.num_joints;
        if weights.len() < vertices * joints {
            anyhow::bail!("skinning weights are missing for the sole probes");
        }
        let shaped = body.shaped_vertices(betas);
        let mut patches = Vec::with_capacity(4);
        let mut proxies = Vec::with_capacity(4);
        for (ankle, toe) in [(7usize, 10usize), (8, 11)] {
            if ankle >= joints || toe >= joints {
                anyhow::bail!("SMPL-H foot joints are missing");
            }
            let mut selected: Vec<usize> = Vec::new();
            for vertex in 0..vertices {
                let influence =
                    weights[vertex * joints + ankle] + weights[vertex * joints + toe];
                if influence > 0.6 {
                    selected.push(vertex);
                }
            }
            if selected.is_empty() {
                anyhow::bail!("SMPL-H foot skin is unavailable");
            }
            // Split front from back along z (the model faces +z in its rest pose).
            let zs: Vec<f32> = selected.iter().map(|v| shaped[v * 3 + 2]).collect();
            let median = quantile(&zs, 0.5);
            // The reference pairs each region with a slightly tighter proxy set,
            // then keeps the 24 lowest-y vertices of the proxy as the sole.
            for (region_below, proxy_below) in [(true, 0.4f32), (false, 0.65)] {
                let region: Vec<usize> = selected
                    .iter()
                    .copied()
                    .filter(|v| {
                        let z = shaped[*v * 3 + 2];
                        if region_below {
                            z <= median
                        } else {
                            z >= median
                        }
                    })
                    .collect();
                let mut proxy_ids: Vec<usize> = selected
                    .iter()
                    .copied()
                    .filter(|v| {
                        let z = shaped[*v * 3 + 2];
                        if region_below {
                            z <= quantile(&zs, proxy_below)
                        } else {
                            z >= quantile(&zs, proxy_below)
                        }
                    })
                    .collect();
                proxy_ids.sort_by(|a, b| {
                    shaped[a * 3 + 1]
                        .partial_cmp(&shaped[b * 3 + 1])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                proxy_ids.truncate(24);
                proxy_ids.sort_unstable();
                let proxy: Vec<usize> = proxy_ids
                    .iter()
                    .filter_map(|v| region.binary_search(v).ok())
                    .collect();
                patches.push(region);
                proxies.push(proxy);
            }
        }

        let mut ids = Vec::new();
        for patch in &patches {
            ids.extend_from_slice(patch);
        }
        let mut selected_weights = Vec::with_capacity(ids.len() * 22);
        for vertex in &ids {
            for joint in 0..22 {
                selected_weights.push(weights[vertex * joints + joint]);
            }
        }
        let mut selected_shaped = Vec::with_capacity(ids.len() * 3);
        for vertex in &ids {
            selected_shaped.extend_from_slice(&shaped[vertex * 3..vertex * 3 + 3]);
        }
        Ok(Self {
            patches,
            proxies,
            ids,
            weights: selected_weights,
            shaped: selected_shaped,
        })
    }

    /// Skin the probes for one frame. `skeleton` must be the frame's forward
    /// kinematics and `rest` its rest joints.
    pub fn sample(&self, skeleton: &Skeleton, rest: &[f32], frame_offset: usize) -> [FootProbes; 2] {
        // Skinning transform per joint: A = [R | t - R * rest].
        let mut rotations = [[0.0f32; 9]; 22];
        let mut translations = [[0.0f32; 3]; 22];
        for joint in 0..22 {
            for k in 0..9 {
                rotations[joint][k] = skeleton.rotations[joint * 9 + k];
            }
            let position = [
                skeleton.positions[joint * 3],
                skeleton.positions[joint * 3 + 1],
                skeleton.positions[joint * 3 + 2],
            ];
            let rest_at = [rest[joint * 3], rest[joint * 3 + 1], rest[joint * 3 + 2]];
            let rotated = [
                rotations[joint][0] * rest_at[0]
                    + rotations[joint][1] * rest_at[1]
                    + rotations[joint][2] * rest_at[2],
                rotations[joint][3] * rest_at[0]
                    + rotations[joint][4] * rest_at[1]
                    + rotations[joint][5] * rest_at[2],
                rotations[joint][6] * rest_at[0]
                    + rotations[joint][7] * rest_at[1]
                    + rotations[joint][8] * rest_at[2],
            ];
            translations[joint] = [
                position[0] - rotated[0],
                position[1] - rotated[1],
                position[2] - rotated[2],
            ];
        }

        let mut skinned = vec![[0.0f32; 3]; self.ids.len()];
        for (slot, _) in self.ids.iter().enumerate() {
            let weight_at = slot * 22;
            let mut blend = [0.0f32; 9];
            let mut translation = [0.0f32; 3];
            for joint in 0..22 {
                let weight = self.weights[weight_at + joint];
                if weight == 0.0 {
                    continue;
                }
                for k in 0..9 {
                    blend[k] += weight * rotations[joint][k];
                }
                for axis in 0..3 {
                    translation[axis] += weight * translations[joint][axis];
                }
            }
            let rest_at = [
                self.shaped[slot * 3],
                self.shaped[slot * 3 + 1],
                self.shaped[slot * 3 + 2],
            ];
            skinned[slot] = [
                blend[0] * rest_at[0] + blend[1] * rest_at[1] + blend[2] * rest_at[2] + translation[0],
                blend[3] * rest_at[0] + blend[4] * rest_at[1] + blend[5] * rest_at[2] + translation[1],
                blend[6] * rest_at[0] + blend[7] * rest_at[1] + blend[8] * rest_at[2] + translation[2],
            ];
        }

        let mut out = [[[0.0f32; 3]; 2]; 2];
        let mut offset = 0usize;
        for foot in 0..2 {
            for side in 0..2 {
                let index = foot * 2 + side;
                let patch = &skinned[offset..offset + self.patches[index].len()];
                offset += self.patches[index].len();
                // `patch[:, proxy].mean(1)`: the probe is the centroid of the 24
                // lowest proxy vertices (the sole surface), not of the whole
                // region — the region also covers the arch and the sides, which
                // would pull the point off the contact patch.
                let proxy = &self.proxies[index];
                let mut sum = [0.0f32; 3];
                for at in proxy {
                    for axis in 0..3 {
                        sum[axis] += patch[*at][axis];
                    }
                }
                let count = proxy.len().max(1) as f32;
                let mut point = [sum[0] / count, sum[1] / count, sum[2] / count];
                // `value[:,1] = patch[...,1].min(1)`: the patch's own lowest
                // vertex is the surface that touches the ground.
                point[1] = patch
                    .iter()
                    .map(|v| v[1])
                    .fold(f32::INFINITY, f32::min);
                out[foot][side] = point;
            }
        }
        let _ = frame_offset;
        out
    }

    pub fn vertex_count(&self) -> usize {
        self.ids.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use velomocap_body::Pose;

    fn loaded() -> Option<(BodyModel, Vec<f32>)> {
        let path = [
            "models/body_models/smplh/neutral/model.npz",
            "../../models/body_models/smplh/neutral/model.npz",
        ]
        .iter()
        .find(|candidate| std::path::Path::new(candidate).is_file())?
        .to_string();
        let body = BodyModel::load(&path, 16).expect("SMPL-H");
        let mut betas = vec![0.0f32; 16];
        betas[0] = 0.3;
        let rest = body.rest_joints(&betas);
        Some((body, rest))
    }

    /// The probes must sit on the sole, under the ankle, and move with the pose.
    #[test]
    fn probes_sit_on_the_sole() {
        let Some((body, rest)) = loaded() else {
            return;
        };
        let mut betas = vec![0.0f32; 16];
        betas[0] = 0.3;
        let geometry = SoleGeometry::build(&body, &betas).expect("sole geometry");
        assert!(geometry.vertex_count() > 100, "too few sole vertices");

        let mut pose = Pose::neutral(body.num_joints, 16);
        pose.betas[0] = 0.3;
        pose.translation = [0.0, 0.9, 0.0];
        let skeleton = body.forward_joints(&pose);
        let probes = geometry.sample(&skeleton, &rest, 0);

        for foot in 0..2 {
            let ankle_joint = if foot == 0 { 7 } else { 8 };
            let ankle = [
                skeleton.positions[ankle_joint * 3],
                skeleton.positions[ankle_joint * 3 + 1],
                skeleton.positions[ankle_joint * 3 + 2],
            ];
            for side in 0..2 {
                let point = probes[foot][side];
                // At or below the ankle, and within a foot's reach of it.
                assert!(point[1] < ankle[1] + 0.02, "probe above ankle: {point:?} vs {ankle:?}");
                let dx = point[0] - ankle[0];
                let dz = point[2] - ankle[2];
                assert!((dx * dx + dz * dz).sqrt() < 0.25, "probe far from ankle: {point:?}");
            }
            // Back patch is behind the front patch along z in the rest pose.
            assert!(probes[foot][0][2] < probes[foot][1][2], "heel behind toe");
        }
    }
}
