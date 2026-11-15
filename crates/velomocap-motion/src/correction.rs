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
}

impl CorrectionReport {
    pub fn line(&self) -> String {
        format!(
            "correction: {} foot contacts, {} hand contacts, {} IK frames, max shift {:.3} m",
            self.foot_contacts, self.hand_contacts, self.ik_frames, self.max_shift
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
            contact[index] =
                low <= settings.floor + thresholds.contact_height && speed <= thresholds.contact_speed;
        }
        for (begin, end) in contact_runs(&contact, thresholds.min_run) {
            let mut xs = Vec::new();
            let mut ys = Vec::new();
            let mut zs = Vec::new();
            for index in begin..end {
                let at = joint(&clip.skeletons[index], ankle);
                xs.push(at[0]);
                ys.push(at[1]);
                zs.push(at[2]);
            }
            let target = [
                median(&xs),
                match settings.mode {
                    CorrectionMode::Flat => settings.floor,
                    CorrectionMode::MultiLevel => median(&ys),
                },
                median(&zs),
            ];
            report.foot_contacts += 1;
            for index in begin..end {
                let at = joint(&clip.skeletons[index], ankle);
                let edge = (index - begin).min(end - 1 - index) as f32;
                let blend = if thresholds.blend_frames == 0 {
                    1.0
                } else {
                    (edge / thresholds.blend_frames as f32).clamp(0.0, 1.0)
                };
                for axis in 0..3 {
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
                if solve_arm(body, &mut clip.poses[index], shoulder, elbow, wrist, target) {
                    report.ik_frames += 1;
                }
            }
            // The IK moved joints; the skeletons have to follow before the next
            // contact run looks at positions again.
            for index in begin..end {
                clip.skeletons[index] = body.forward_joints(&clip.poses[index]);
            }
        }
    }
}

/// Two-bone IK: rotate the elbow and the shoulder until the wrist reaches
/// `target`. Returns whether anything moved.
fn solve_arm(
    body: &BodyModel,
    pose: &mut Pose,
    shoulder: usize,
    elbow: usize,
    wrist: usize,
    target: [f32; 3],
) -> bool {
    let mut moved = false;
    for _ in 0..4 {
        let skeleton = body.forward_joints(pose);
        let wrist_at = joint(&skeleton, wrist);
        let distance = ((target[0] - wrist_at[0]).powi(2)
            + (target[1] - wrist_at[1]).powi(2)
            + (target[2] - wrist_at[2]).powi(2))
        .sqrt();
        if distance < 0.005 {
            break;
        }
        for joint_index in [elbow, shoulder] {
            let skeleton = body.forward_joints(pose);
            let pivot = joint(&skeleton, joint_index);
            let wrist_now = joint(&skeleton, wrist);
            let from = [
                (wrist_now[0] - pivot[0]) as f64,
                (wrist_now[1] - pivot[1]) as f64,
                (wrist_now[2] - pivot[2]) as f64,
            ];
            let to = [
                (target[0] - pivot[0]) as f64,
                (target[1] - pivot[1]) as f64,
                (target[2] - pivot[2]) as f64,
            ];
            if math::norm3(&from) < 1e-4 || math::norm3(&to) < 1e-4 {
                continue;
            }
            let delta = math::align(&from, &to);
            if rotate_joint(body, pose, &skeleton, joint_index, &delta) {
                moved = true;
            }
        }
    }
    moved
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
