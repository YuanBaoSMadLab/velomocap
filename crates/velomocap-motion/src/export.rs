//! Export fitted SMPL-X motion as BVH.
//!
//! Converts the 55-joint SMPL-X skeleton into the 52-bone body hierarchy the
//! rest of VeloMocap exports, using the Rust BVH writer that was validated
//! against the original RGB MoCap implementation.
//!
//! The mapping takes the 22 body joints plus the 30 finger joints and drops
//! the face/eye joints, which carry no animation useful for retargeting.

use std::path::Path;

use anyhow::Result;
use velomocap_body::Skeleton;

use crate::fit::FittedFrame;

/// SMPL-X joint indices for the 52 exported bones, in export order.
///
/// The first 22 are the body joints the pipeline has always exported; the
/// remaining 30 are the three phalanges of each of the five fingers, per hand.
const EXPORT_JOINTS: [usize; 52] = [
    // Body: pelvis, hips, spine, knees, ankles, feet, neck, collars, head,
    // shoulders, elbows, wrists.
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    // Left hand: index, middle, pinky, ring, thumb (3 each).
    25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    // Right hand.
    40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
];

/// Parent index within the exported 52-bone array.
fn export_parents() -> [i32; 52] {
    // Mirrors the body hierarchy plus each finger chain hanging off its wrist.
    let mut parents = [-1i32; 52];
    // Body (same topology as the 22-joint SMPL-X body).
    let body = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19];
    parents[..22].copy_from_slice(&body);
    // Left fingers 22..36 (5 chains of 3), rooted at the left wrist (20).
    let mut cursor = 22usize;
    for _ in 0..5 {
        parents[cursor] = 20;
        parents[cursor + 1] = cursor as i32;
        parents[cursor + 2] = (cursor + 1) as i32;
        cursor += 3;
    }
    // Right fingers 37..51, rooted at the right wrist (21).
    for _ in 0..5 {
        parents[cursor] = 21;
        parents[cursor + 1] = cursor as i32;
        parents[cursor + 2] = (cursor + 1) as i32;
        cursor += 3;
    }
    parents
}

const EXPORT_NAMES: [&str; 52] = [
    "Hips", "LeftUpLeg", "RightUpLeg", "Spine", "LeftLeg", "RightLeg", "Spine1", "LeftFoot",
    "RightFoot", "Spine2", "LeftToeBase", "RightToeBase", "Neck", "LeftShoulder",
    "RightShoulder", "Head", "LeftArm", "RightArm", "LeftForeArm", "RightForeArm", "LeftHand",
    "RightHand",
    // Left hand fingers.
    "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3",
    "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3",
    "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3",
    "LeftHandRing1", "LeftHandRing2", "LeftHandRing3",
    "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3",
    // Right hand fingers.
    "RightHandIndex1", "RightHandIndex2", "RightHandIndex3",
    "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3",
    "RightHandPinky1", "RightHandPinky2", "RightHandPinky3",
    "RightHandRing1", "RightHandRing2", "RightHandRing3",
    "RightHandThumb1", "RightHandThumb2", "RightHandThumb3",
];

/// Build a `velomocap_core::rig::Binding` from the first frame's skeleton.
///
/// The rest pose comes from the neutral skeleton, so the BVH hierarchy offsets
/// are the model's own proportions.
pub fn binding_from_rest(rest: &Skeleton) -> velomocap_core::rig::Binding {
    let parents = export_parents();
    let mut names = Vec::with_capacity(52);
    let mut rest_positions: Vec<f64> = Vec::with_capacity(52 * 3);
    let mut ends: Vec<f64> = Vec::with_capacity(52 * 3);
    let mut basis: Vec<f64> = Vec::with_capacity(52 * 9);

    for (slot, joint) in EXPORT_JOINTS.iter().enumerate() {
        names.push(EXPORT_NAMES[slot].to_string());
        let p = rest.joint(*joint);
        rest_positions.extend_from_slice(&[p[0] as f64, p[1] as f64, p[2] as f64]);
        // Terminal offsets: a short bone along +Y, matching the pipeline's
        // convention for leaves without a child.
        ends.extend_from_slice(&[0.0, 0.04, 0.0]);
        // Identity basis: SMPL-X joints are already in the export orientation.
        basis.extend_from_slice(&[
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]);
    }

    velomocap_core::rig::Binding {
        names,
        parents: parents.to_vec(),
        rest: rest_positions,
        ends,
        basis,
        origin: [0.0; 3],
        version: 17,
    }
}

/// Local rotation matrices for one fitted frame, in export bone order.
fn local_rotations(frame: &FittedFrame, parents: &[i32]) -> Vec<f64> {
    let mut out = vec![0.0f64; 52 * 9];
    for (slot, joint) in EXPORT_JOINTS.iter().enumerate() {
        let world = &frame.skeleton.rotations[joint * 9..joint * 9 + 9];
        let parent = parents[slot];
        let local = if parent < 0 {
            let mut m = [0.0f64; 9];
            for k in 0..9 {
                m[k] = world[k] as f64;
            }
            m
        } else {
            // local = parent_world^T @ world
            let p = parent as usize;
            let pw = &frame.skeleton.rotations[EXPORT_JOINTS[p] * 9..EXPORT_JOINTS[p] * 9 + 9];
            let mut m = [0.0f64; 9];
            for row in 0..3 {
                for col in 0..3 {
                    let mut acc = 0.0;
                    for k in 0..3 {
                        acc += pw[k * 3 + row] as f64 * world[k * 3 + col] as f64;
                    }
                    m[row * 3 + col] = acc;
                }
            }
            m
        };
        out[slot * 9..slot * 9 + 9].copy_from_slice(&local);
    }
    out
}

/// Write a fitted clip as BVH.
pub fn write_bvh(
    path: impl AsRef<Path>,
    frames: &[FittedFrame],
    rest: &Skeleton,
    fps: f64,
) -> Result<BvhExport> {
    if frames.is_empty() {
        anyhow::bail!("no frames to export");
    }
    let binding = binding_from_rest(rest);
    let parents = export_parents();
    let count = frames.len();

    let mut roots: Vec<f64> = Vec::with_capacity(count * 3);
    let mut local = Vec::with_capacity(count * 52 * 9);
    for frame in frames {
        // The root translation is the hips position in world space.
        let hips = frame.skeleton.joint(EXPORT_JOINTS[0]);
        roots.extend_from_slice(&[hips[0] as f64, hips[1] as f64, hips[2] as f64]);
        local.extend_from_slice(&local_rotations(frame, &parents));
    }

    let report = velomocap_core::bvh::write_bvh(
        path.as_ref(),
        &binding,
        &roots,
        &local,
        fps,
        17,
    )?;

    Ok(BvhExport {
        path: path.as_ref().to_path_buf(),
        frames: count,
        bones: 52,
        fps,
        orders: report.orders,
        minimum_singularity_margin: report.minimum_singularity_margin,
    })
}

#[derive(Clone, Debug)]
pub struct BvhExport {
    pub path: std::path::PathBuf,
    pub frames: usize,
    pub bones: usize,
    pub fps: f64,
    pub orders: Vec<String>,
    pub minimum_singularity_margin: f64,
}

impl BvhExport {
    pub fn summary(&self) -> String {
        format!(
            "{} frames, {} bones, {:.2} fps -> {}",
            self.frames,
            self.bones,
            self.fps,
            self.path.display()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hierarchy_is_well_formed() {
        let parents = export_parents();
        assert_eq!(parents.len(), 52);
        assert_eq!(parents[0], -1);
        for j in 1..52 {
            let p = parents[j];
            assert!(p >= 0 && (p as usize) < j, "bone {j} has invalid parent {p}");
        }
    }

    #[test]
    fn names_are_unique_and_ascii() {
        let mut seen = std::collections::HashSet::new();
        for name in EXPORT_NAMES {
            assert!(name.is_ascii(), "{name} is not ASCII");
            assert!(seen.insert(name), "{name} is duplicated");
        }
        assert_eq!(seen.len(), 52);
        assert_eq!(EXPORT_NAMES[0], "Hips");
    }

    #[test]
    fn export_joints_cover_the_hands() {
        // Every finger chain must end at a distinct SMPL-X joint.
        let unique: std::collections::HashSet<usize> = EXPORT_JOINTS.iter().copied().collect();
        assert_eq!(unique.len(), 52);
        assert!(EXPORT_JOINTS.iter().all(|j| *j < 55));
    }
}
