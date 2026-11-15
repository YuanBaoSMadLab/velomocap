//! Temporal smoothing for a generated clip.
//!
//! The reference explicitly does **no** upstream smoothing (`flow_correction.py`:
//! "No inference, SMPL-X conversion, upstream smoothing or second target IK
//! pass"). Its smoothness comes from planning the root and the contact targets
//! with a least-squares acceleration term (`contact_pose_plan.py`) plus per-bone
//! correction of the bones its IK touched, so a non-contact limb that the network
//! reconstructed noisily stays noisy there as well.
//!
//! Our FlowHMR output has stretches of ~10 cm per frame squared on a swinging
//! arm, which reads as "some frames are not smooth", so this module adds the same
//! kind of term across every joint:
//!
//! ```text
//! min  sum_t |x[t+1] - 2 x[t] + x[t-1]|^2  +  lambda * sum_t |x[t] - ref[t]|^2
//! ```
//!
//! The operator is banded and symmetric positive definite, so the solve is a
//! short conjugate-gradient iteration. Rotations are smoothed in the 6D
//! forward/up representation (the third and second columns of the local
//! rotation), which cannot flip the way quaternions do, and are rebuilt with
//! Gram-Schmidt afterwards.

use anyhow::Result;

use velomocap_body::{BodyModel, Pose};

use crate::flow::FlowClip;

/// Second difference applied to `values` and scattered through its transpose.
fn normal_operator(values: &[f64], out: &mut [f64], lambda: f64, work: &mut [f64]) {
    let n = values.len();
    for value in out.iter_mut() {
        *value = 0.0;
    }
    for index in 0..n {
        out[index] += lambda * values[index];
    }
    if n < 3 {
        return;
    }
    for index in 1..n - 1 {
        work[index] = values[index + 1] - 2.0 * values[index] + values[index - 1];
    }
    for index in 1..n - 1 {
        let value = work[index];
        out[index - 1] += value;
        out[index] -= 2.0 * value;
        out[index + 1] += value;
    }
}

/// The pose plan's weighted smoothing: minimise
/// pose_w |x - ref|^2 + vel_w |V x|^2 + acc_w |A x|^2, the shape
/// contact_pose_plan gives its least-squares root and swing targets (its
/// pose_sigma / velocity_sigma / acceleration_sigma). Solved with the same
/// conjugate-gradient loop as the plain smoother.
pub fn plan_series(reference: &[f64], pose_w: f64, vel_w: f64, acc_w: f64) -> Vec<f64> {
    let n = reference.len();
    if n < 3 {
        return reference.to_vec();
    }
    let mut x = vec![0.0f64; n];
    let mut residual: Vec<f64> = reference.iter().map(|value| value * pose_w).collect();
    let mut direction = residual.clone();
    let mut applied = vec![0.0f64; n];
    let mut work = vec![0.0f64; n];
    let mut rr: f64 = residual.iter().map(|value| value * value).sum();
    let target = rr * 1e-20;
    for _ in 0..n.max(16) {
        if rr <= target {
            break;
        }
        // out = pose_w x + vel_w V^T V x + acc_w A^T A x
        for value in applied.iter_mut() {
            *value = 0.0;
        }
        for index in 0..n {
            applied[index] += pose_w * direction[index];
        }
        if n > 1 {
            for index in 0..n - 1 {
                work[index] = direction[index + 1] - direction[index];
            }
            for index in 0..n - 1 {
                let value = vel_w * work[index];
                applied[index] -= value;
                applied[index + 1] += value;
            }
        }
        for index in 1..n - 1 {
            work[index] = direction[index + 1] - 2.0 * direction[index] + direction[index - 1];
        }
        for index in 1..n - 1 {
            let value = acc_w * work[index];
            applied[index - 1] += value;
            applied[index] -= 2.0 * value;
            applied[index + 1] += value;
        }
        let pap: f64 = direction.iter().zip(&applied).map(|(a, b)| a * b).sum();
        if pap.abs() < 1e-20 {
            break;
        }
        let alpha = rr / pap;
        for index in 0..n {
            x[index] += alpha * direction[index];
            residual[index] -= alpha * applied[index];
        }
        let next: f64 = residual.iter().map(|value| value * value).sum();
        let beta = next / rr;
        for index in 0..n {
            direction[index] = residual[index] + beta * direction[index];
        }
        rr = next;
    }
    x
}
/// Solve `(lambda I + A^T A) x = lambda * reference`.
fn smooth_series(reference: &[f64], lambda: f64) -> Vec<f64> {
    let n = reference.len();
    if n < 3 || lambda <= 0.0 {
        return reference.to_vec();
    }
    let mut x = vec![0.0f64; n];
    let mut residual: Vec<f64> = reference.iter().map(|value| value * lambda).collect();
    let mut direction = residual.clone();
    let mut applied = vec![0.0f64; n];
    let mut work = vec![0.0f64; n];
    let mut rr: f64 = residual.iter().map(|value| value * value).sum();
    let target = rr * 1e-20;
    for _ in 0..n.max(16) {
        if rr <= target {
            break;
        }
        normal_operator(&direction, &mut applied, lambda, &mut work);
        let pap: f64 = direction.iter().zip(&applied).map(|(a, b)| a * b).sum();
        if pap.abs() < 1e-20 {
            break;
        }
        let alpha = rr / pap;
        for index in 0..n {
            x[index] += alpha * direction[index];
            residual[index] -= alpha * applied[index];
        }
        let next: f64 = residual.iter().map(|value| value * value).sum();
        let beta = next / rr;
        for index in 0..n {
            direction[index] = residual[index] + beta * direction[index];
        }
        rr = next;
    }
    x
}

/// 6D representation of a rotation matrix: its third column (forward) and second
/// column (up), which is what the reference's `CorrectRotationsForBone` uses.
fn to6(matrix: &[f64; 9]) -> [f64; 6] {
    [
        matrix[2], matrix[5], matrix[8], // third column
        matrix[1], matrix[4], matrix[7], // second column
    ]
}

/// Rebuild a rotation from (forward, up) with Gram-Schmidt.
fn from6(values: [f64; 6]) -> [f64; 9] {
    let forward = normalise3([values[0], values[1], values[2]]);
    let up_hint = [values[3], values[4], values[5]];
    let dot = forward[0] * up_hint[0] + forward[1] * up_hint[1] + forward[2] * up_hint[2];
    let up = normalise3([
        up_hint[0] - forward[0] * dot,
        up_hint[1] - forward[1] * dot,
        up_hint[2] - forward[2] * dot,
    ]);
    let right = [
        up[1] * forward[2] - up[2] * forward[1],
        up[2] * forward[0] - up[0] * forward[2],
        up[0] * forward[1] - up[1] * forward[0],
    ];
    [
        right[0], up[0], forward[0],
        right[1], up[1], forward[1],
        right[2], up[2], forward[2],
    ]
}

fn normalise3(v: [f64; 3]) -> [f64; 3] {
    let length = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if length < 1e-12 {
        [0.0, 0.0, 1.0]
    } else {
        [v[0] / length, v[1] / length, v[2] / length]
    }
}

fn rotvec_to_matrix(v: [f64; 3]) -> [f64; 9] {
    let angle = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if angle < 1e-9 {
        return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    }
    crate::ik::axis_angle_matrix([v[0] / angle, v[1] / angle, v[2] / angle], angle)
}

fn axis_angle(matrix: &[f64; 9]) -> [f32; 3] {
    let trace = matrix[0] + matrix[4] + matrix[8];
    let cos = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0);
    let angle = cos.acos();
    if angle < 1e-6 {
        return [0.0; 3];
    }
    let axis = [
        matrix[7] - matrix[5],
        matrix[2] - matrix[6],
        matrix[3] - matrix[1],
    ];
    let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    if norm < 1e-9 {
        return [0.0; 3];
    }
    [
        (axis[0] / norm * angle) as f32,
        (axis[1] / norm * angle) as f32,
        (axis[2] / norm * angle) as f32,
    ]
}

/// Smooth every joint's local rotation over time. `lambda` trades smoothness for
/// staying near the source (larger = closer to the source).
pub fn smooth(clip: &mut FlowClip, body: &BodyModel, lambda: f64) -> Result<()> {
    let frames = clip.poses.len();
    if frames < 3 || lambda <= 0.0 {
        return Ok(());
    }
    let joints = body.num_joints;
    let mut channels = vec![vec![0.0f64; frames]; joints * 6];
    for (frame, pose) in clip.poses.iter().enumerate() {
        for joint in 0..joints {
            let matrix = matrix_of(pose, joint);
            let six = to6(&matrix);
            for axis in 0..6 {
                channels[joint * 6 + axis][frame] = six[axis];
            }
        }
    }
    for channel in channels.iter_mut() {
        *channel = smooth_series(channel, lambda);
    }
    for (frame, pose) in clip.poses.iter_mut().enumerate() {
        for joint in 0..joints {
            let six = [
                channels[joint * 6][frame],
                channels[joint * 6 + 1][frame],
                channels[joint * 6 + 2][frame],
                channels[joint * 6 + 3][frame],
                channels[joint * 6 + 4][frame],
                channels[joint * 6 + 5][frame],
            ];
            let matrix = from6(six);
            let angles = axis_angle(&matrix);
            if joint == 0 {
                pose.root = angles;
            } else {
                let at = (joint - 1) * 3;
                pose.body[at] = angles[0];
                pose.body[at + 1] = angles[1];
                pose.body[at + 2] = angles[2];
            }
        }
    }
    clip.skeletons = clip
        .poses
        .iter()
        .map(|pose| body.forward_joints(pose))
        .collect();
    Ok(())
}

fn matrix_of(pose: &Pose, joint: usize) -> [f64; 9] {
    if joint == 0 {
        rotvec_to_matrix([
            pose.root[0] as f64,
            pose.root[1] as f64,
            pose.root[2] as f64,
        ])
    } else {
        let at = (joint - 1) * 3;
        rotvec_to_matrix([
            pose.body[at] as f64,
            pose.body[at + 1] as f64,
            pose.body[at + 2] as f64,
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_series_smooths_and_keeps_a_ramp() {
        let mut spike: Vec<f64> = (0..24).map(|i| i as f64 * 0.01).collect();
        spike[12] += 0.1;
        let planned = plan_series(&spike, 1.0, 10.0, 40.0);
        let before = (spike[12] - 0.5 * (spike[11] + spike[13])).abs();
        let after = (planned[12] - 0.5 * (planned[11] + planned[13])).abs();
        assert!(after < before * 0.5, "spike {before:.4} -> {after:.4}");
        let ramp: Vec<f64> = (0..24).map(|i| i as f64 * 0.02).collect();
        let kept = plan_series(&ramp, 1.0, 0.1, 0.1);
        for (original, result) in ramp.iter().zip(&kept) {
            assert!((original - result).abs() < 0.01, "{original:.4} -> {result:.4}");
        }
    }

    #[test]
    fn smoothing_removes_a_single_frame_spike() {
        let mut reference: Vec<f64> = (0..20).map(|i| i as f64 * 0.01).collect();
        reference[10] += 0.1;
        let smooth = smooth_series(&reference, 0.5);
        let before = (reference[10] - 0.5 * (reference[9] + reference[11])).abs();
        let after = (smooth[10] - 0.5 * (smooth[9] + smooth[11])).abs();
        assert!(after < before * 0.5, "spike {before:.4} -> {after:.4}");
    }

    #[test]
    fn smoothing_keeps_a_straight_ramp() {
        let reference: Vec<f64> = (0..15).map(|i| i as f64 * 0.02).collect();
        let smooth = smooth_series(&reference, 1e-4);
        for (before, after) in reference.iter().zip(&smooth) {
            assert!((before - after).abs() < 0.01, "{before:.4} -> {after:.4}");
        }
    }

    #[test]
    fn six_d_round_trips_a_rotation() {
        let matrix = crate::ik::axis_angle_matrix([0.3, -0.5, 0.8], 0.7);
        let rebuilt = from6(to6(&matrix));
        for index in 0..9 {
            assert!(
                (matrix[index] - rebuilt[index]).abs() < 1e-6,
                "index {index}: {:.6} vs {:.6}",
                matrix[index],
                rebuilt[index]
            );
        }
    }
}
