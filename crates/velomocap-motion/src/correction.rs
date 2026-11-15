//! Contact correction for a generated SMPL-H clip.
//!
//! The reference solves this with its motion-correction stage (`foot_contacts`
//! and `hand_lock`): a planted foot must not slide and a resting hand must stay
//! put. Without this pass the character skates over the floor.
//!
//! * Feet are pinned by translating the whole body - rigid, so the mesh, the
//!   joints and the export cannot disagree afterwards.
//! * Hands are pinned with a two-bone IK: a few CCD steps rotate the elbow and
//!   the shoulder until the wrist sits on the target, then the chain is written
//!   back as SMPL-H axis-angle rotations.

use anyhow::Result;

use velomocap_body::{BodyModel, Pose, Skeleton};
use velomocap_core::math;

use crate::flow::FlowClip;
use crate::ik::Transform;

const LEFT_HIP: usize = 1;
const LEFT_ANKLE: usize = 7;
const RIGHT_ANKLE: usize = 8;
const LEFT_FOOT: usize = 10;
const RIGHT_FOOT: usize = 11;
const LEFT_SHOULDER: usize = 16;
const RIGHT_SHOULDER: usize = 17;
const LEFT_ELBOW: usize = 18;
const RIGHT_ELBOW: usize = 19;
const LEFT_WRIST: usize = 20;
const RIGHT_WRIST: usize = 21;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CorrectionMode {
    /// Every foot is pinned to one ground height.
    Flat,
    /// Each contact run keeps the height it was detected at (stairs, ramps).
    MultiLevel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LockPreset {
    Tight,
    Normal,
    Loose,
}

#[derive(Clone, Copy, Debug)]
pub struct CorrectionSettings {
    pub enabled: bool,
    pub mode: CorrectionMode,
    pub foot: bool,
    pub hand: bool,
    pub preset: LockPreset,
    /// Ground height in metres.
    pub floor: f32,
    /// Vertical shift applied to the whole clip.
    pub pelvis_offset: f32,
}

impl Default for CorrectionSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            mode: CorrectionMode::Flat,
            foot: true,
            hand: true,
            preset: LockPreset::Normal,
            floor: 0.0,
            pelvis_offset: 0.0,
        }
    }
}

pub struct Thresholds {
    pub contact_height: f32,
    pub contact_speed: f32,
    pub blend_frames: usize,
    pub max_correction: f32,
    pub min_run: usize,
}

impl CorrectionSettings {
    pub fn thresholds(&self) -> Thresholds {
        match self.preset {
            LockPreset::Tight => Thresholds {
                contact_height: 0.035,
                contact_speed: 0.03,
                blend_frames: 4,
                max_correction: 0.20,
                min_run: 2,
            },
            LockPreset::Normal => Thresholds {
                contact_height: 0.07,
                contact_speed: 0.06,
                blend_frames: 3,
                max_correction: 0.35,
                min_run: 2,
            },
            LockPreset::Loose => Thresholds {
                contact_height: 0.12,
                contact_speed: 0.12,
                blend_frames: 2,
                max_correction: 0.55,
                min_run: 1,
            },
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct CorrectionReport {
    pub foot_contacts: usize,
    pub hand_contacts: usize,
    pub max_shift: f32,
    pub ik_frames: usize,
    /// Worst distance between a solved wrist and its pinned target.
    pub max_hand_error: f32,
}

impl CorrectionReport {
    pub fn line(&self) -> String {
        format!(
            "correction: {} foot contacts, {} hand contacts, {} IK frames, max shift {:.3} m, hand residual {:.1} mm",
            self.foot_contacts,
            self.hand_contacts,
            self.ik_frames,
            self.max_shift,
            self.max_hand_error * 1000.0
        )
    }
}

/// Apply the correction in place; the clip's skeletons are rebuilt afterwards.
pub fn apply(
    clip: &mut FlowClip,
    body: &BodyModel,
    settings: &CorrectionSettings,
) -> Result<CorrectionReport> {
    let mut report = CorrectionReport::default();
    if !settings.enabled {
        return Ok(report);
    }
    if settings.pelvis_offset != 0.0 {
        for pose in &mut clip.poses {
            pose.translation[1] += settings.pelvis_offset;
        }
    }
    let thresholds = settings.thresholds();

    if settings.foot {
        foot_lock(clip, settings, &thresholds, &mut report);
    }
    if settings.hand {
        hand_lock(clip, body, &thresholds, &mut report);
    }

    clip.skeletons = clip
        .poses
        .iter()
        .map(|pose| body.forward_joints(pose))
        .collect();
    Ok(report)
}

fn joint(skeleton: &Skeleton, index: usize) -> [f32; 3] {
    [
        skeleton.positions[index * 3],
        skeleton.positions[index * 3 + 1],
        skeleton.positions[index * 3 + 2],
    ]
}

fn horizontal_speed(a: [f32; 3], b: [f32; 3], fps: f32) -> f32 {
    let dx = b[0] - a[0];
    let dz = b[2] - a[2];
    (dx * dx + dz * dz).sqrt() * fps
}

fn contact_runs(contact: &[bool], min_run: usize) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut start = None;
    for (index, hit) in contact.iter().enumerate() {
        match (hit, start) {
            (true, None) => start = Some(index),
            (false, Some(begin)) => {
                if index - begin >= min_run {
                    runs.push((begin, index));
                }
                start = None;
            }
            _ => {}
        }
    }
    if let Some(begin) = start {
        if contact.len() - begin >= min_run {
            runs.push((begin, contact.len()));
        }
    }
    runs
}

fn foot_lock(
    clip: &mut FlowClip,
    settings: &CorrectionSettings,
    thresholds: &Thresholds,
    report: &mut CorrectionReport,
) {
    let fps = clip.fps.max(1.0);
    let count = clip.skeletons.len();
    if count < 2 {
        return;
    }
    let feet: [(usize, usize); 2] = [(LEFT_ANKLE, LEFT_FOOT), (RIGHT_ANKLE, RIGHT_FOOT)];
    let mut shift = vec![[0.0f64; 3]; count];
    let mut weight = vec![0.0f32; count];

    // Ground reference: `Flat` uses the configured floor, `MultiLevel` uses the
    // lowest ankle height seen in the clip, which follows a ramp instead of
    // pinning everything to one plane.
    let mut ground = settings.floor;
    if settings.mode == CorrectionMode::MultiLevel {
        let mut lows = Vec::new();
        for index in 0..count {
            let skeleton = &clip.skeletons[index];
            for (ankle, toe) in feet {
                lows.push(joint(skeleton, ankle)[1].min(joint(skeleton, toe)[1]));
            }
        }
        ground = median(&lows);
    }

    for (ankle, toe) in feet {
        let mut contact = vec![false; count];
        for index in 0..count {
            let skeleton = &clip.skeletons[index];
            let low = joint(skeleton, ankle)[1].min(joint(skeleton, toe)[1]);
            let speed = if index == 0 {
                0.0
            } else {
                horizontal_speed(
                    joint(&clip.skeletons[index - 1], ankle),
                    joint(skeleton, ankle),
                    fps,
                )
            };
            contact[index] = low <= ground + thresholds.contact_height
                && speed <= thresholds.contact_speed;
        }
        for (begin, end) in contact_runs(&contact, thresholds.min_run) {
            let mut xs = Vec::new();
            let mut zs = Vec::new();
            for index in begin..end {
                let at = joint(&clip.skeletons[index], ankle);
                xs.push(at[0]);
                zs.push(at[2]);
            }
            // Horizontal only: the vertical placement is solved when the clip is
            // dropped onto its floor, and snapping Y on the contact frames only
            // (leaving it alone in between) is what made the body bob up and
            // down after a reconstruction.
            let target = [median(&xs), 0.0, median(&zs)];
            report.foot_contacts += 1;
            for index in begin..end {
                let at = joint(&clip.skeletons[index], ankle);
                let edge = (index - begin).min(end - 1 - index) as f32;
                let blend = if thresholds.blend_frames == 0 {
                    1.0
                } else {
                    (edge / thresholds.blend_frames as f32).clamp(0.0, 1.0)
                };
                for axis in [0usize, 2] {
                    shift[index][axis] += (target[axis] - at[axis]) as f64 * blend as f64;
                }
                weight[index] += blend;
            }
        }
    }

    for index in 0..count {
        if weight[index] <= 0.0 {
            continue;
        }
        let w = weight[index].max(1.0) as f64;
        let mut delta = [
            shift[index][0] / w,
            shift[index][1] / w,
            shift[index][2] / w,
        ];
        let length = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
        let cap = thresholds.max_correction as f64;
        if length > cap && length > 0.0 {
            let scale = cap / length;
            for value in &mut delta {
                *value *= scale;
            }
        }
        report.max_shift = report.max_shift.max(length as f32);
        for axis in 0..3 {
            clip.poses[index].translation[axis] += delta[axis] as f32;
        }
    }
}

fn hand_lock(
    clip: &mut FlowClip,
    body: &BodyModel,
    thresholds: &Thresholds,
    report: &mut CorrectionReport,
) {
    let fps = clip.fps.max(1.0);
    let count = clip.skeletons.len();
    if count < 2 {
        return;
    }
    let hands: [(usize, usize, usize, usize); 2] = [
        (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, LEFT_HIP),
        (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, LEFT_HIP),
    ];
    let rest = body.rest_joints(&clip.betas);

    for (shoulder, elbow, wrist, hip) in hands {
        let mut contact = vec![false; count];
        for index in 0..count {
            let skeleton = &clip.skeletons[index];
            let speed = if index == 0 {
                0.0
            } else {
                horizontal_speed(
                    joint(&clip.skeletons[index - 1], wrist),
                    joint(skeleton, wrist),
                    fps,
                )
            };
            // At rest and no higher than the chest.
            let low = joint(skeleton, wrist)[1] <= joint(skeleton, hip)[1] + 0.45;
            contact[index] = low && speed <= thresholds.contact_speed;
        }
        for (begin, end) in contact_runs(&contact, thresholds.min_run.max(2)) {
            let mut points = [Vec::new(), Vec::new(), Vec::new()];
            for index in begin..end {
                let at = joint(&clip.skeletons[index], wrist);
                for axis in 0..3 {
                    points[axis].push(at[axis]);
                }
            }
            let target = [
                median(&points[0]),
                median(&points[1]),
                median(&points[2]),
            ];
            report.hand_contacts += 1;
            for index in begin..end {
                if solve_arm(body, &mut clip.poses[index], &rest, shoulder, elbow, wrist, target) {
                    report.ik_frames += 1;
                }
                // How close the solved wrist actually sits to the pinned point.
                let chain = forward_chain(body, &clip.poses[index], &rest);
                let settled = chain[wrist].1;
                let error = ((settled[0] - target[0] as f64).powi(2)
                    + (settled[1] - target[1] as f64).powi(2)
                    + (settled[2] - target[2] as f64).powi(2))
                .sqrt();
                report.max_hand_error = report.max_hand_error.max(error as f32);
            }
            // The IK moved joints; the skeletons have to follow before the next
            // contact run looks at positions again.
            for index in begin..end {
                clip.skeletons[index] = body.forward_joints(&clip.poses[index]);
            }
        }
    }
}

/// Two-bone IK with Kimodo's analytic solve: stage 1 sets `|ac| = |at|` with the
/// cosine rule on the elbow and the shoulder, stage 2 aims the start joint at
/// the target. World-space rotations are applied to the joint frames, which is
/// the standard parent-first convention `forward_joints` uses.
///
/// Returns whether anything moved.
fn solve_arm(
    body: &BodyModel,
    pose: &mut Pose,
    rest: &[f32],
    shoulder: usize,
    elbow: usize,
    wrist: usize,
    target: [f32; 3],
) -> bool {
    let target = [target[0] as f64, target[1] as f64, target[2] as f64];
    let start = {
        let world = forward_chain(body, pose, rest);
        world[wrist].1
    };
    if distance3(start, target) < 1e-4 {
        return false;
    }

    // Stage 1: bend the elbow, then the shoulder, to the length the target
    // needs, keeping the plane the arm already lies in.
    let world = forward_chain(body, pose, rest);
    let a = world[shoulder].1;
    let b = world[elbow].1;
    let c = world[wrist].1;
    let l_ab = distance3(a, b);
    let l_bc = distance3(b, c);
    let l_at = distance3(a, target).clamp(1e-4, (l_ab + l_bc) * 0.999);
    if l_ab < 1e-4 || l_bc < 1e-4 {
        return false;
    }
    let theta_abc_now = angle_between(sub3(b, a), sub3(b, c));
    let theta_abc_goal = cosine_rule(l_ab, l_bc, l_at);
    let theta_bac_now = angle_between(sub3(a, b), sub3(a, c));
    let theta_bac_goal = cosine_rule(l_ab, l_at, l_bc);

    let mut axis = cross3(sub3(c, a), sub3(b, a));
    axis = if length3(axis) < 1e-8 {
        [0.0, 0.0, 1.0]
    } else {
        normalise3(axis)
    };

    let mut moved = false;
    if (theta_abc_goal - theta_abc_now).abs() > 1e-6 {
        moved |= rotate_joint_world(
            body,
            pose,
            rest,
            elbow,
            &crate::ik::axis_angle_matrix(axis, theta_abc_goal - theta_abc_now),
        );
    }
    if (theta_bac_goal - theta_bac_now).abs() > 1e-6 {
        moved |= rotate_joint_world(
            body,
            pose,
            rest,
            shoulder,
            &crate::ik::axis_angle_matrix(axis, theta_bac_goal - theta_bac_now),
        );
    }

    // Stage 2: aim the start joint so the end joint lands on the target.
    let world = forward_chain(body, pose, rest);
    let a = world[shoulder].1;
    let c = world[wrist].1;
    if distance3(c, target) > 1e-4 {
        let aim = math::align(&sub3(c, a), &sub3(target, a));
        moved |= rotate_joint_world(body, pose, rest, shoulder, &aim);
    }
    moved
}

/// Forward kinematics as `(rotation, position)` per joint, parent first.
fn forward_chain(body: &BodyModel, pose: &Pose, rest: &[f32]) -> Vec<([f64; 9], [f64; 3])> {
    let count = body.num_joints;
    let mut out = Vec::with_capacity(count);
    for joint in 0..count {
        let local = if joint == 0 {
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
        };
        let parent = body.parents[joint];
        if parent < 0 {
            let rotation = local;
            let position = [
                pose.translation[0] as f64 + rest[0] as f64,
                pose.translation[1] as f64 + rest[1] as f64,
                pose.translation[2] as f64 + rest[2] as f64,
            ];
            out.push((rotation, position));
        } else {
            let p = parent as usize;
            let (parent_rotation, parent_position) = out[p];
            let offset = [
                (rest[joint * 3] - rest[p * 3]) as f64,
                (rest[joint * 3 + 1] - rest[p * 3 + 1]) as f64,
                (rest[joint * 3 + 2] - rest[p * 3 + 2]) as f64,
            ];
            let rotated = rotate3(&parent_rotation, offset);
            let position = [
                parent_position[0] + rotated[0],
                parent_position[1] + rotated[1],
                parent_position[2] + rotated[2],
            ];
            out.push((math::mul3(&parent_rotation, &local), position));
        }
    }
    out
}

fn rotate3(rotation: &[f64; 9], v: [f64; 3]) -> [f64; 3] {
    [
        rotation[0] * v[0] + rotation[1] * v[1] + rotation[2] * v[2],
        rotation[3] * v[0] + rotation[4] * v[1] + rotation[5] * v[2],
        rotation[6] * v[0] + rotation[7] * v[1] + rotation[8] * v[2],
    ]
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn distance3(a: [f64; 3], b: [f64; 3]) -> f64 {
    length3(sub3(a, b))
}

fn length3(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalise3(v: [f64; 3]) -> [f64; 3] {
    let len = length3(v).max(1e-12);
    [v[0] / len, v[1] / len, v[2] / len]
}

fn angle_between(a: [f64; 3], b: [f64; 3]) -> f64 {
    let left = normalise3(a);
    let right = normalise3(b);
    let cosine = left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
    let sine = length3(cross3(left, right));
    sine.atan2(cosine)
}

fn cosine_rule(side_left: f64, side_right: f64, side_across: f64) -> f64 {
    let value = (side_right * side_right + side_left * side_left - side_across * side_across)
        / (2.0 * side_left * side_right);
    value.clamp(-1.0, 1.0).acos()
}

/// Rotate one joint's world frame by `delta`, keeping its local axis-angle in
/// sync: `R_local ← R_parentᵀ · delta · R_parent · R_local`.
fn rotate_joint_world(
    body: &BodyModel,
    pose: &mut Pose,
    rest: &[f32],
    joint_index: usize,
    delta: &[f64; 9],
) -> bool {
    let world = forward_chain(body, pose, rest);
    let current = world[joint_index].0;
    let next = math::mul3(delta, &current);
    if joint_index == 0 {
        pose.root = axis_angle(&next);
        return true;
    }
    let parent = match body.parents[joint_index] {
        value if value >= 0 => value as usize,
        _ => {
            pose.root = axis_angle(&next);
            return true;
        }
    };
    let parent_world = world[parent].0;
    let local = math::tmul3(&parent_world, &next);
    let angles = axis_angle(&local);
    let at = (joint_index - 1) * 3;
    if at + 3 > pose.body.len() {
        return false;
    }
    pose.body[at] = angles[0];
    pose.body[at + 1] = angles[1];
    pose.body[at + 2] = angles[2];
    true
}

/// Apply `delta` (a world-space rotation) to one joint and rewrite that joint's
/// axis-angle in the pose.
fn rotate_joint(
    body: &BodyModel,
    pose: &mut Pose,
    skeleton: &Skeleton,
    joint_index: usize,
    delta: &[f64; 9],
) -> bool {
    let current = world_rotation(skeleton, joint_index);
    let next = math::mul3(delta, &current);
    if joint_index == 0 {
        let angles = axis_angle(&next);
        pose.root = angles;
        return true;
    }
    let parent = match body.parents.get(joint_index) {
        Some(value) if *value >= 0 => *value as usize,
        _ => {
            let angles = axis_angle(&next);
            pose.root = angles;
            return true;
        }
    };
    let parent_world = world_rotation(skeleton, parent);
    let local = math::tmul3(&parent_world, &next);
    let angles = axis_angle(&local);
    let at = (joint_index - 1) * 3;
    if at + 3 > pose.body.len() {
        return false;
    }
    pose.body[at] = angles[0];
    pose.body[at + 1] = angles[1];
    pose.body[at + 2] = angles[2];
    true
}

fn world_rotation(skeleton: &Skeleton, index: usize) -> [f64; 9] {
    let at = index * 9;
    let mut out = [0.0f64; 9];
    for k in 0..9 {
        out[k] = skeleton.rotations[at + k] as f64;
    }
    out
}

fn axis_angle(m: &[f64; 9]) -> [f32; 3] {
    let trace = m[0] + m[4] + m[8];
    let cos = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0);
    let angle = cos.acos();
    if angle < 1e-6 {
        return [0.0; 3];
    }
    let axis = [
        m[7] - m[5],
        m[2] - m[6],
        m[3] - m[1],
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

fn median(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        0.5 * (sorted[middle - 1] + sorted[middle])
    } else {
        sorted[middle]
    }
}

/// Rotation vector (axis * angle) to a rotation matrix.
fn rotvec_to_matrix(v: [f64; 3]) -> [f64; 9] {
    let angle = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if angle < 1e-9 {
        return Transform::IDENTITY.rotation;
    }
    crate::ik::axis_angle_matrix([v[0] / angle, v[1] / angle, v[2] / angle], angle)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_path() -> Option<String> {
        for candidate in [
            "models/body_models/smplh/neutral/model.npz",
            "../../models/body_models/smplh/neutral/model.npz",
        ] {
            if std::path::Path::new(candidate).is_file() {
                return Some(candidate.to_string());
            }
        }
        None
    }

    fn loaded() -> Option<(BodyModel, Vec<f32>)> {
        let path = model_path()?;
        let body = BodyModel::load(&path, 16).expect("load SMPL-H");
        // Non-zero shape on purpose: with every beta at zero `forward_joints`
        // uses the npz's stored `J` instead of the regressed joints, and those
        // sit a few centimetres apart. Real clips always carry shape.
        let mut betas = vec![0.0f32; 16];
        betas[0] = 0.3;
        let rest = body.rest_joints(&betas);
        Some((body, rest))
    }

    /// The chain the IK solves on must reproduce `forward_joints` exactly,
    /// otherwise the solver aims at a different skeleton than the viewport and
    /// the exporter see.
    #[test]
    fn forward_chain_matches_forward_joints() {
        let Some((body, rest)) = loaded() else {
            eprintln!("skipping: SMPL-H model not found");
            return;
        };
        let mut pose = Pose::neutral(body.num_joints, 16);
        pose.betas[0] = 0.3;
        pose.translation = [0.3, 0.9, -2.0];
        pose.root = [0.2, -0.4, 0.1];
        for (joint, angles) in [
            (LEFT_SHOULDER, [0.4f32, 0.2, -0.3]),
            (LEFT_ELBOW, [0.9, 0.0, 0.0]),
            (LEFT_WRIST, [0.2, 0.5, 0.0]),
            (4, [0.6, 0.0, 0.0]),
            (7, [-0.4, 0.0, 0.0]),
        ] {
            let at = (joint - 1) * 3;
            pose.body[at..at + 3].copy_from_slice(&angles);
        }

        let skeleton = body.forward_joints(&pose);
        let chain = forward_chain(&body, &pose, &rest);
        let mut worst = (0.0f64, 0usize);
        for joint in 0..body.num_joints {
            let expected = [
                skeleton.positions[joint * 3] as f64,
                skeleton.positions[joint * 3 + 1] as f64,
                skeleton.positions[joint * 3 + 2] as f64,
            ];
            let error = distance3(chain[joint].1, expected);
            if error > worst.0 {
                worst = (error, joint);
            }
        }
        assert!(
            worst.0 < 1e-4,
            "joint {} differs by {:.4} m",
            worst.1,
            worst.0
        );
    }

    /// The analytic two-bone solve must put the wrist on its target and must not
    /// disturb the rest of the body.
    #[test]
    fn solve_arm_reaches_the_target() {
        let Some((body, rest)) = loaded() else {
            eprintln!("skipping: SMPL-H model not found");
            return;
        };
        let mut pose = Pose::neutral(body.num_joints, 16);
        pose.betas[0] = 0.3;
        pose.translation = [0.0, 0.9, 0.0];
        // A raised left arm to start from.
        let at = (LEFT_SHOULDER - 1) * 3;
        pose.body[at..at + 3].copy_from_slice(&[0.0, 0.0, -1.2]);

        let target = {
            let skeleton = body.forward_joints(&pose);
            let wrist = joint(&skeleton, LEFT_WRIST);
            let shoulder = joint(&skeleton, LEFT_SHOULDER);
            // Reachable on purpose: part of the way out, so the elbow bends.
            [
                shoulder[0] + (wrist[0] - shoulder[0]) * 0.7,
                shoulder[1] + (wrist[1] - shoulder[1]) * 0.7 - 0.05,
                shoulder[2] + (wrist[2] - shoulder[2]) * 0.7 + 0.03,
            ]
        };
        let hips_before = {
            let skeleton = body.forward_joints(&pose);
            joint(&skeleton, 0)
        };

        assert!(solve_arm(
            &body,
            &mut pose,
            &rest,
            LEFT_SHOULDER,
            LEFT_ELBOW,
            LEFT_WRIST,
            target
        ));

        let after = body.forward_joints(&pose);
        let wrist = joint(&after, LEFT_WRIST);
        let error = ((wrist[0] - target[0]).powi(2)
            + (wrist[1] - target[1]).powi(2)
            + (wrist[2] - target[2]).powi(2))
        .sqrt();
        assert!(
            error < 0.01,
            "wrist {wrist:?} target {target:?} error {error}"
        );
        let hips = joint(&after, 0);
        let drift = ((hips[0] - hips_before[0]).powi(2)
            + (hips[1] - hips_before[1]).powi(2)
            + (hips[2] - hips_before[2]).powi(2))
        .sqrt();
        assert!(drift < 1e-5, "hips drifted {drift}");
    }
}
