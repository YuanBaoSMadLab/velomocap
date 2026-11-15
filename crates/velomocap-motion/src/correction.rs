//! Contact correction for a generated SMPL-H clip.
//!
//! The reference solves this with its motion-correction stage (`foot_contacts`
//! and `hand_lock`): a planted foot must not slide and a resting hand must stay
//! put. Without this pass the character skates over the floor.
//!
//! * Feet are pinned through Kimodo's trajectory corrector (`k-1`): the contact
//!   frames are pins, the rest is free, and the velocity/acceleration weights
//!   keep the root from kinking where a contact run starts. Only X/Z move; the
//!   vertical placement belongs to the floor solve.
//! * Hands are pinned with a two-bone IK (the Kimodo solve on the SMPL-H chain),
//!   then the chain is written back as axis-angle rotations.

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

/// Kimodo's default trajectory weights (`Utility.cpp`).
const POS_WEIGHT: f64 = 0.001;
const VEL_WEIGHT: f64 = 1.0;
const ACC_WEIGHT: f64 = 10.0;

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

/// Solver parameters, mapped from the reference's `PRESETS` table
/// (`correction_settings.py`).
pub struct Thresholds {
    /// How close to the ground a sole must be to count as planted.
    pub contact_height: f32,
    /// Horizontal speed above which a low foot is sliding, not planted.
    pub contact_speed: f32,
    /// Frames a correction fades in and out over.
    pub blend_frames: usize,
    /// Hard cap on how far a frame may be moved (`anchor_drift`).
    pub max_correction: f32,
    /// Shortest contact run that is kept (`minimum`).
    pub min_run: usize,
    /// How much of the correction is applied (`strength`: weak holds 60%).
    pub strength: f32,
    /// Dropouts shorter than this are bridged (`gap`).
    pub bridge_frames: usize,
    /// Contact probability needed for a full-strength hold (`confidence`).
    pub confidence: f32,
}

impl CorrectionSettings {
    pub fn thresholds(&self) -> Thresholds {
        match self.preset {
            // The reference calls these strong / normal / weak.
            LockPreset::Tight => Thresholds {
                contact_height: 0.035,
                contact_speed: 0.03,
                blend_frames: 4,
                max_correction: 0.20,
                min_run: 2,
                strength: 1.0,
                bridge_frames: 2,
                confidence: 0.65,
            },
            LockPreset::Normal => Thresholds {
                contact_height: 0.07,
                contact_speed: 0.06,
                blend_frames: 3,
                max_correction: 0.12,
                min_run: 2,
                strength: 1.0,
                bridge_frames: 2,
                confidence: 0.65,
            },
            LockPreset::Loose => Thresholds {
                contact_height: 0.12,
                contact_speed: 0.12,
                blend_frames: 2,
                max_correction: 0.12,
                min_run: 1,
                strength: 0.6,
                bridge_frames: 0,
                confidence: 0.75,
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
    /// Largest vertical correction applied by the ground pass.
    pub vertical_shift: f32,
    /// Largest pelvis lift needed to reach a contact (`reachable_pelvis_lift`).
    pub pelvis_lift: f32,
}

impl CorrectionReport {
    pub fn line(&self) -> String {
        format!(
            "correction: {} foot contacts, {} hand contacts, {} IK frames, max shift {:.3} m, hand residual {:.1} mm, ground {:+.3} m, pelvis lift {:.3} m",
            self.foot_contacts,
            self.hand_contacts,
            self.ik_frames,
            self.max_shift,
            self.max_hand_error * 1000.0,
            self.vertical_shift,
            self.pelvis_lift
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
        ground_feet(clip, settings, &thresholds, &mut report);
    }
    if settings.hand {
        hand_lock(clip, body, &thresholds, &mut report);
    }
    // Kimodo's order: end effectors (hands) first, then the contact IK that
    // stabilises the legs on the ground.
    if settings.foot {
        leg_ik(clip, body, settings, &thresholds, &mut report);
    }

    clip.skeletons = clip
        .poses
        .iter()
        .map(|pose| body.forward_joints(pose))
        .collect();
    Ok(report)
}

/// Kimodo's velocity weights: slow motion is preserved more strongly, so a
/// character that is standing still does not drift while the solver evens out
/// the contacts.
fn velocity_weights(clip: &FlowClip) -> Vec<f64> {
    let count = clip.skeletons.len();
    if count == 0 {
        return Vec::new();
    }
    let mut weights = vec![1.0f64; count];
    for frame in 1..count {
        let x = (clip.skeletons[frame].positions[0] - clip.skeletons[frame - 1].positions[0]) as f64;
        let z = (clip.skeletons[frame].positions[2] - clip.skeletons[frame - 1].positions[2]) as f64;
        let speed = (x * x + z * z).sqrt() / 0.05;
        weights[frame] = 1.0 / speed.max(1e-3);
    }
    if count > 1 {
        weights[0] = weights[1];
    }
    weights
}

/// Vertical correction, ported from Kimodo's `CorrectHipsY`: the frames where a
/// sole is on the ground are pins, the rest is free, and the position, velocity
/// and acceleration weights keep the height a smooth curve instead of the
/// per-frame snapping that made the body bob.
fn ground_feet(
    clip: &mut FlowClip,
    settings: &CorrectionSettings,
    thresholds: &Thresholds,
    report: &mut CorrectionReport,
) {
    let count = clip.skeletons.len();
    if count < 3 {
        return;
    }
    let feet = [(LEFT_ANKLE, LEFT_FOOT), (RIGHT_ANKLE, RIGHT_FOOT)];
    let tolerance = thresholds.contact_height;
    let mut margins = vec![-1.0f64; count];
    let mut observation = vec![0.0f64; count];
    let mut pinned = 0usize;
    for frame in 0..count {
        let mut stack = [0.0f32; 2];
        let mut lowest = f32::MAX;
        for (slot, (ankle, toe)) in feet.iter().enumerate() {
            let at = joint(&clip.skeletons[frame], *ankle)[1].min(joint(&clip.skeletons[frame], *toe)[1]);
            stack[slot] = at;
            lowest = lowest.min(at);
        }
        // Only the lowest sole decides: a stepping foot is not a pin.
        let gap = lowest - settings.floor;
        observation[frame] = (clip.poses[frame].translation[1] - gap) as f64;
        if gap <= tolerance {
            margins[frame] = 0.0;
            pinned += 1;
        }
    }
    if pinned < 2 {
        return;
    }

    // Kimodo's weights for the vertical pass: position x10, velocity x1,
    // acceleration x0.1.
    let reference: Vec<f64> = clip.poses.iter().map(|pose| pose.translation[1] as f64).collect();
    let corrector = crate::trajectory::TrajectoryCorrector::new(
        &margins,
        POS_WEIGHT * 10.0,
        VEL_WEIGHT,
        ACC_WEIGHT * 0.1,
        &[],
        60,
    );
    let corrected = corrector.interpolate(&observation, &reference);
    let cap = thresholds.max_correction as f64;
    let mut moved = 0.0f32;
    for frame in 0..count {
        let delta = (corrected[frame] - reference[frame]).clamp(-cap, cap);
        moved = moved.max(delta.abs() as f32);
        clip.poses[frame].translation[1] += delta as f32;
    }
    report.vertical_shift = moved;
}

/// Fill dropouts shorter than `frames` so a brief loss of the contact signal
/// does not split one stance into two (`gap` in the reference's presets).
fn bridge_gaps(contact: &mut [bool], frames: usize) {
    if frames == 0 {
        return;
    }
    let count = contact.len();
    let mut index = 0;
    while index < count {
        if contact[index] {
            index += 1;
            continue;
        }
        let begin = index;
        while index < count && !contact[index] {
            index += 1;
        }
        let end = index;
        if begin > 0 && end < count && end - begin <= frames {
            for hit in &mut contact[begin..end] {
                *hit = true;
            }
        }
    }
}

/// How strongly a run is held, from the network's own contact probabilities.
/// The layout of the four channels differs between checkpoints, so the run is
/// scored with the largest of them: a low score only softens the hold, it never
/// drops a contact the geometry found.
fn contact_confidence(clip: &FlowClip, begin: usize, end: usize, confidence: f32) -> f32 {
    if clip.foot_contacts.len() < end || confidence <= 0.0 {
        return 1.0;
    }
    let mut best = 0.0f32;
    for frame in begin..end {
        for value in clip.foot_contacts[frame] {
            best = best.max(value);
        }
    }
    if best <= 0.0 {
        return 1.0;
    }
    (best / confidence).clamp(0.5, 1.0)
}

/// Kimodo's `_hermite`: `count` interior samples of the cubic that matches the
/// endpoint values and velocities of a segment.
fn hermite(a: f32, va: f32, b: f32, vb: f32, count: usize) -> Vec<f32> {
    let duration = (count + 1) as f32;
    (1..=count)
        .map(|sample| {
            let t = sample as f32 / duration;
            (2.0 * t.powi(3) - 3.0 * t.powi(2) + 1.0) * a
                + (t.powi(3) - 2.0 * t.powi(2) + t) * duration * va
                + (-2.0 * t.powi(3) + 3.0 * t.powi(2)) * b
                + (t.powi(3) - t.powi(2)) * duration * vb
        })
        .collect()
}

/// Kimodo's `smooth_auto_handoffs`: when two stances touch exactly (no swing
/// frame between them) the anchor must not step at the boundary, so the
/// correction is carried across with a Hermite blend over a window of about
/// 0.2 s on each side. Walking clips are where this shows up.
fn smooth_handoffs(
    stances: &[(usize, usize)],
    offsets_x: &mut [f64],
    offsets_z: &mut [f64],
    fps: f32,
) {
    let radius = ((fps * 0.20).round() as usize).max(3);
    for pair in 0..stances.len().saturating_sub(1) {
        let left = stances[pair];
        let right = stances[pair + 1];
        if left.1 != right.0 {
            // A real swing gap separates them: nothing to blend.
            continue;
        }
        let boundary = right.0;
        let first = left.0.max(boundary.saturating_sub(radius));
        let end = right.1.min(boundary + radius);
        if end - first < 3 {
            continue;
        }
        let a_x = offsets_x[first];
        let b_x = offsets_x[end - 1];
        let v_x_a = offsets_x[first + 1] - a_x;
        let v_x_b = b_x - offsets_x[end - 2];
        let a_z = offsets_z[first];
        let b_z = offsets_z[end - 1];
        let v_z_a = offsets_z[first + 1] - a_z;
        let v_z_b = b_z - offsets_z[end - 2];
        let blend_x = hermite(a_x as f32, v_x_a as f32, b_x as f32, v_x_b as f32, end - first - 2);
        let blend_z = hermite(a_z as f32, v_z_a as f32, b_z as f32, v_z_b as f32, end - first - 2);
        for (slot, index) in (first + 1..end - 1).enumerate() {
            offsets_x[index] = blend_x[slot] as f64;
            offsets_z[index] = blend_z[slot] as f64;
        }
    }
}

/// Kimodo's `couple_anchors`: two consecutive stances of the same foot that are
/// separated by a very short release gap must not be pinned independently, or the
/// foot teleports across the gap. Each stance may only be shifted as a whole
/// (no rotation, no drift inside it), the shift is solved with a tridiagonal
/// system that pulls neighbouring endpoints together, and every stance's
/// adjustment is bounded to 12 cm.
fn couple_anchor_offsets(
    stances: &[(usize, usize)],
    anchors: &[[f32; 3]],
    support: &[[f32; 3]],
    fps: f32,
) -> Vec<[f32; 2]> {
    let count = stances.len();
    let mut adjustment = vec![[0.0f32; 2]; count];
    if count < 2 {
        return adjustment;
    }
    let gap_limit = fps * 0.12;
    let mut diagonal = vec![1.0f64; count];
    let mut off = vec![0.0f64; count - 1];
    let mut rhs = vec![[0.0f64; 2]; count];

    for pair in 0..count - 1 {
        let (left, right) = (stances[pair], stances[pair + 1]);
        let gap = right.0.saturating_sub(left.1);
        if gap as f32 > gap_limit {
            continue;
        }
        let weight = (10.0 * (gap_limit / (gap + 1) as f32).powi(3) as f64).min(1000.0);
        let a = left.1 - 1;
        let b = right.0;
        for axis in [0usize, 2] {
            let difference = ((anchors[b][axis] - support[b][axis])
                - (anchors[a][axis] - support[a][axis])) as f64;
            diagonal[pair] += weight;
            diagonal[pair + 1] += weight;
            off[pair] = -weight;
            rhs[pair][if axis == 0 { 0 } else { 1 }] += weight * difference;
            rhs[pair + 1][if axis == 0 { 0 } else { 1 }] -= weight * difference;
        }
    }

    // Thomas algorithm on the tridiagonal system, once per column.
    for column in 0..2 {
        let mut c = vec![0.0f64; count];
        let mut d = vec![0.0f64; count];
        let mut b = diagonal.clone();
        if b[0].abs() < 1e-12 {
            continue;
        }
        c[0] = off[0] / b[0];
        d[0] = rhs[0][column] / b[0];
        for row in 1..count {
            let denominator = b[row] - off[row - 1] * c[row - 1];
            if denominator.abs() < 1e-12 {
                break;
            }
            if row + 1 < count {
                c[row] = off[row] / denominator;
            }
            d[row] = (rhs[row][column] - off[row - 1] * d[row - 1]) / denominator;
        }
        let mut x = vec![0.0f64; count];
        x[count - 1] = d[count - 1];
        for row in (0..count - 1).rev() {
            x[row] = d[row] - c[row] * x[row + 1];
        }
        for stance in 0..count {
            adjustment[stance][if column == 0 { 0 } else { 1 }] = x[stance] as f32;
        }
        b.clear();
    }

    // Never shift a stance more than 12 cm.
    for delta in &mut adjustment {
        let length = (delta[0] * delta[0] + delta[1] * delta[1]).sqrt();
        if length > 0.12 {
            delta[0] *= 0.12 / length;
            delta[1] *= 0.12 / length;
        }
    }
    adjustment
}

/// Kimodo's `TargetReachFalloff`: a Gaussian that fades the IK weight out as the
/// target approaches the limit of what the two bones can reach, so the solver
/// never straightens a limb into a locked pose.
fn reach_falloff(target: [f64; 3], hip: [f64; 3], max_reach: f64) -> f64 {
    let distance = distance3(target, hip);
    let mut t = (distance / max_reach.max(1e-6) - 0.99).max(0.0) / 0.01;
    t *= t;
    (-2.0 * t * t).exp()
}

/// Leg IK on the contact frames, ported from `DoContactIK`'s two-bone branch:
/// the ankle is pinned to where the sole already is, so the root correction does
/// not have to carry the whole load. The joint's own orientation is untouched
/// (only the hip and the knee move), which is what keeps the foot flat.
fn leg_ik(
    clip: &mut FlowClip,
    body: &BodyModel,
    settings: &CorrectionSettings,
    thresholds: &Thresholds,
    report: &mut CorrectionReport,
) {
    let fps = clip.fps.max(1.0);
    let count = clip.skeletons.len();
    if count < 2 {
        return;
    }
    let rest = body.rest_joints(&clip.betas);
    let legs: [(usize, usize, usize, usize); 2] = [
        (1, 4, 7, 10), // left hip, knee, ankle, toe
        (2, 5, 8, 11), // right hip, knee, ankle, toe
    ];
    let cap = thresholds.max_correction.min(0.05) as f64;
    let mut lifts = vec![0.0f32; count];
    for (hip, knee, ankle, toe) in legs {
        // Bone lengths from the rest pose: the reach limit for the falloff.
        let thigh = distance3(
            joint3(&rest, knee),
            joint3(&rest, hip),
        );
        let shin = distance3(joint3(&rest, ankle), joint3(&rest, knee));
        let max_reach = thigh + shin;

        let flight = flight_frames(clip, settings.floor);
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
            contact[index] = !flight[index]
                && low <= settings.floor + thresholds.contact_height
                && speed <= thresholds.contact_speed;
        }

        for (begin, end) in contact_runs(&contact, thresholds.min_run) {
            // Same rolling anchor the root pass uses, so the leg IK and the root
            // agree on where the contact patch should be.
            let anchors = rolling_anchor(clip, begin, end, ankle, toe, settings.floor);
            for (offset, index) in (begin..end).enumerate() {
                let current = joint(&clip.skeletons[index], ankle);
                let sole = support_point(clip, index, ankle, toe);
                let hip_pos = joint(&clip.skeletons[index], hip);
                // Move the ankle so that the sole's support point lands back on
                // the anchor; the ankle itself is free to slide over it.
                let target = [
                    current[0] as f64 + (anchors[offset][0] - sole[0]) as f64,
                    current[1] as f64,
                    current[2] as f64 + (anchors[offset][2] - sole[2]) as f64,
                ];
                let weight = reach_falloff(
                    target,
                    [
                        hip_pos[0] as f64,
                        hip_pos[1] as f64,
                        hip_pos[2] as f64,
                    ],
                    max_reach,
                );
                if weight <= 0.02 {
                    // Kimodo's `reachable_pelvis_lift`: when the contact is out of
                    // reach the pelvis is raised by the shortfall, never by more
                    // than 12 cm, instead of leaving the leg straight and the foot
                    // floating.
                    let shortfall = distance3(
                        target,
                        [
                            hip_pos[0] as f64,
                            hip_pos[1] as f64,
                            hip_pos[2] as f64,
                        ],
                    ) - max_reach * 0.999;
                    if shortfall > 0.0 {
                        lifts[index] = lifts[index].max(shortfall.min(0.12) as f32);
                    }
                    continue;
                }
                let blended = [
                    (current[0] as f64 + (target[0] - current[0] as f64) * weight) as f32,
                    current[1],
                    (current[2] as f64 + (target[2] - current[2] as f64) * weight) as f32,
                ];
                let before = joint(&clip.skeletons[index], ankle);
                let backup = clip.poses[index].clone();
                if solve_arm(body, &mut clip.poses[index], &rest, hip, knee, ankle, blended) {
                    let after = body.forward_joints(&clip.poses[index]);
                    let settled = joint(&after, ankle);
                    let moved = distance3(
                        [settled[0] as f64, settled[1] as f64, settled[2] as f64],
                        [before[0] as f64, before[1] as f64, before[2] as f64],
                    );
                    if moved > cap {
                        // Further than a correction should ever move an ankle:
                        // keep the pose the network produced.
                        clip.poses[index] = backup;
                    } else {
                        clip.skeletons[index] = after;
                        report.ik_frames += 1;
                    }
                }
            }
        }
    }

    // Apply the collected pelvis lifts and rebuild the skeletons that saw them.
    let mut lifted = 0.0f32;
    for index in 0..count {
        if lifts[index] <= 0.0 {
            continue;
        }
        clip.poses[index].translation[1] += lifts[index];
        clip.skeletons[index] = body.forward_joints(&clip.poses[index]);
        lifted = lifted.max(lifts[index]);
    }
    report.pelvis_lift = lifted;
}

fn joint3(rest: &[f32], index: usize) -> [f64; 3] {
    [
        rest[index * 3] as f64,
        rest[index * 3 + 1] as f64,
        rest[index * 3 + 2] as f64,
    ]
}

fn joint(skeleton: &Skeleton, index: usize) -> [f32; 3] {    [
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

/// The integration step of `rolling_anchor`, split out so it can be tested
/// without a clip: `separations[i]` is heel - toe and `weights[i]` the heel
/// weight at frame `i`.
fn rolling_path(
    separations: &[[f32; 3]],
    weights: &[f32],
    observed: &[[f32; 3]],
    floor: f32,
) -> Vec<[f32; 3]> {
    let frames = separations.len();
    let mut path = vec![[0.0f32; 3]; frames];
    for frame in 1..frames {
        let dw = weights[frame] - weights[frame - 1];
        for axis in 0..3 {
            path[frame][axis] = path[frame - 1][axis]
                + 0.5 * (separations[frame][axis] + separations[frame - 1][axis]) * dw;
        }
    }
    let mut offsets = [0.0f32; 3];
    for axis in 0..3 {
        let diffs: Vec<f32> = (0..frames).map(|f| observed[f][axis] - path[f][axis]).collect();
        offsets[axis] = median(&diffs);
    }
    for frame in 0..frames {
        path[frame][0] += offsets[0];
        path[frame][1] = floor;
        path[frame][2] += offsets[2];
    }
    path
}

/// Kimodo's `rolling_anchor`: a continuous no-slip anchor for one stance.
///
/// Taking the observed support point directly would move the anchor whenever the
/// weight rolls, which reads as a sliding foot. Instead the *contact location*
/// is integrated: as the weight transfers from one material point to the other,
/// the anchor moves by the separation times the weight change. The result is
/// then offset so its median matches the observed support points, and its height
/// is the floor.
fn rolling_anchor(
    clip: &FlowClip,
    begin: usize,
    end: usize,
    ankle: usize,
    toe: usize,
    floor: f32,
) -> Vec<[f32; 3]> {
    let frames = end - begin;
    let mut separations = Vec::with_capacity(frames);
    let mut weights = Vec::with_capacity(frames);
    let mut observed = Vec::with_capacity(frames);
    for frame in begin..end {
        let (toe_at, heel) = sole_points(clip, frame, ankle, toe);
        let weight = support_weight(toe_at, heel);
        separations.push([
            heel[0] - toe_at[0],
            heel[1] - toe_at[1],
            heel[2] - toe_at[2],
        ]);
        weights.push(weight);
        observed.push([
            toe_at[0] * (1.0 - weight) + heel[0] * weight,
            toe_at[1].min(heel[1]),
            toe_at[2] * (1.0 - weight) + heel[2] * weight,
        ]);
    }

    rolling_path(&separations, &weights, &observed, floor)
}

/// The two material points of a sole: the toe joint and a heel estimated behind
/// the ankle (SMPL-H has no heel joint, so the foot axis stands in for it).
///
/// Kimodo blends these with weights computed from **geometry only** - which of
/// the two points is lower - so a rolling foot moves its contact point along the
/// sole instead of sliding the whole foot.
fn sole_points(clip: &FlowClip, frame: usize, ankle: usize, toe: usize) -> ([f32; 3], [f32; 3]) {
    let skeleton = &clip.skeletons[frame];
    let ankle_at = joint(skeleton, ankle);
    let toe_at = joint(skeleton, toe);
    // Heel: mirror of the toe offset about the ankle, damped so skin noise does
    // not swing it.
    let heel = [
        ankle_at[0] + (ankle_at[0] - toe_at[0]) * 0.6,
        ankle_at[1] + (ankle_at[1] - toe_at[1]) * 0.6,
        ankle_at[2] + (ankle_at[2] - toe_at[2]) * 0.6,
    ];
    (toe_at, heel)
}

/// Kimodo's `support_weights`, returned as the **heel weight**: 0.5 when the
/// foot is flat, 0 when the toe bears the weight, 1 at the heel. The 5 mm
/// deadband keeps millimetre noise from being read as a weight transfer and the
/// ramp saturates at 25 mm.
fn support_weight(toe: [f32; 3], heel: [f32; 3]) -> f32 {
    let delta = toe[1] - heel[1];
    let x = ((delta.abs() - 0.005) / 0.02).clamp(0.0, 1.0);
    let smooth = x * x * (3.0 - 2.0 * x);
    0.5f32 + 0.5 * smooth * delta.signum()
}

/// The point of the sole that is actually bearing the weight.
fn support_point(clip: &FlowClip, frame: usize, ankle: usize, toe: usize) -> [f32; 3] {
    let (toe_at, heel) = sole_points(clip, frame, ankle, toe);
    let weight = support_weight(toe_at, heel);
    [
        toe_at[0] * (1.0 - weight) + heel[0] * weight,
        toe_at[1].min(heel[1]),
        toe_at[2] * (1.0 - weight) + heel[2] * weight,
    ]
}

/// Kimodo's `airborne_excursions`, simplified to the gate that matters here: a/// stretch where the lowest sole is well above the floor and there is lower
/// ground on both sides is a jump, so it must not be pinned as a contact.
fn flight_frames(clip: &FlowClip, floor: f32) -> Vec<bool> {
    let count = clip.skeletons.len();
    let mut flight = vec![false; count];
    if count < 5 {
        return flight;
    }
    let mut lowest = vec![f32::MAX; count];
    for frame in 0..count {
        for (ankle, toe) in [(LEFT_ANKLE, LEFT_FOOT), (RIGHT_ANKLE, RIGHT_FOOT)] {
            let at = joint(&clip.skeletons[frame], ankle)[1].min(joint(&clip.skeletons[frame], toe)[1]);
            lowest[frame] = lowest[frame].min(at);
        }
    }
    let reach = ((clip.fps.max(1.0) * 4.0) as usize).clamp(2, count);
    let mut index = 0;
    while index < count {
        if lowest[index] - floor <= 0.15 {
            index += 1;
            continue;
        }
        let begin = index;
        while index < count && lowest[index] - floor > 0.15 {
            index += 1;
        }
        let end = index;
        // Needs lower ground (or the clip boundary) on both sides to be an arc.
        let before = begin == 0 || (0..begin).any(|frame| lowest[frame] - floor <= 0.15);
        let after = end == count || (end..count).any(|frame| lowest[frame] - floor <= 0.15);
        if before && after && end - begin <= reach {
            for hit in &mut flight[begin..end] {
                *hit = true;
            }
        }
    }
    flight
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
    let mut pinned = vec![false; count];
    let mut target_x = vec![0.0f64; count];
    let mut target_z = vec![0.0f64; count];

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

    let flight = flight_frames(clip, ground);
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
            contact[index] = !flight[index]
                && low <= ground + thresholds.contact_height
                && speed <= thresholds.contact_speed;
        }
        bridge_gaps(&mut contact, thresholds.bridge_frames);
        let runs = contact_runs(&contact, thresholds.min_run);
        if runs.is_empty() {
            continue;
        }

        // Per-frame anchors, observed support points and hold strength for this
        // foot, then the stance coupling that stops two neighbouring pins from
        // teleporting the foot across a short release gap.
        let mut anchors = vec![[0.0f32; 3]; count];
        let mut support = vec![[0.0f32; 3]; count];
        let mut holds = vec![1.0f32; count];
        for (begin, end) in &runs {
            // The anchor is Kimodo's rolling anchor: a continuous no-slip path
            // through the stance, so a foot that rolls from heel to toe keeps its
            // contact patch still instead of jumping at the middle of the run.
            let path = rolling_anchor(clip, *begin, *end, ankle, toe, ground);
            // Weak presets hold 60% of the correction, and a low network contact
            // probability softens it further instead of dropping the contact.
            let hold =
                thresholds.strength * contact_confidence(clip, *begin, *end, thresholds.confidence);
            report.foot_contacts += 1;
            for (offset, index) in (*begin..*end).enumerate() {
                anchors[index] = path[offset];
                support[index] = support_point(clip, index, ankle, toe);
                holds[index] = hold;
            }
        }

        let adjustments = couple_anchor_offsets(&runs, &anchors, &support, fps);
        let mut local_x = vec![0.0f64; count];
        let mut local_z = vec![0.0f64; count];
        let mut local_used = vec![false; count];
        for (stance, (begin, end)) in runs.iter().enumerate() {
            let adjustment = adjustments[stance];
            for index in *begin..*end {
                // Horizontal only: the vertical placement is solved when the clip
                // is dropped onto its floor.
                local_x[index] = (anchors[index][0] + adjustment[0] - support[index][0]) as f64
                    * holds[index] as f64;
                local_z[index] = (anchors[index][2] + adjustment[1] - support[index][2]) as f64
                    * holds[index] as f64;
                local_used[index] = true;
            }
        }
        // Stances that touch exactly get their correction carried across with a
        // Hermite blend, so the anchor does not step at the boundary.
        smooth_handoffs(&runs, &mut local_x, &mut local_z, fps);
        for index in 0..count {
            if !local_used[index] {
                continue;
            }
            // Two feet in contact at once average their demands.
            if pinned[index] {
                target_x[index] = 0.5 * (target_x[index] + local_x[index]);
                target_z[index] = 0.5 * (target_z[index] + local_z[index]);
            } else {
                target_x[index] = local_x[index];
                target_z[index] = local_z[index];
            }
            pinned[index] = true;
        }
    }

    if !pinned.iter().any(|hit| *hit) {
        return;
    }

    // Kimodo's trajectory corrector: the contact frames are pins (margin 0), the
    // rest is free, and velocity/acceleration weights keep the root from
    // jumping between them. This replaces the old per-frame blend, which could
    // still kink the root where a contact run started.
    let mut margins = vec![-1.0f64; count];
    for index in 0..count {
        if pinned[index] {
            margins[index] = 0.0;
        }
    }
    if !margins.iter().any(|margin| *margin == 0.0) {
        return;
    }
    let reference = vec![0.0f64; count];
    // Kimodo's velocity weights: standing still is weighted up so the root does
    // not drift while the solver evens out the contacts.
    let weights = velocity_weights(clip);
    let corrector = crate::trajectory::TrajectoryCorrector::new(
        &margins,
        POS_WEIGHT,
        VEL_WEIGHT,
        ACC_WEIGHT,
        &weights,
        60,
    );
    let shift_x = corrector.interpolate(&target_x, &reference);
    let shift_z = corrector.interpolate(&target_z, &reference);

    let cap = thresholds.max_correction as f64;
    for index in 0..count {
        let mut delta = [shift_x[index], 0.0, shift_z[index]];
        let length = (delta[0] * delta[0] + delta[2] * delta[2]).sqrt();
        if length > cap && length > 0.0 {
            let scale = cap / length;
            delta[0] *= scale;
            delta[2] *= scale;
        }
        report.max_shift = report.max_shift.max(length as f32);
        for axis in [0usize, 2] {
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

    /// `smooth_auto_handoffs`: when two stances touch, the correction curve must
    /// not step at the boundary.
    #[test]
    fn handoffs_blend_across_a_touching_boundary() {
        let frames = 24;
        // Two stances that touch at frame 12.
        let stances = vec![(0usize, 12usize), (12usize, 24usize)];
        let mut x = vec![0.0f64; frames];
        let mut z = vec![0.0f64; frames];
        // A 10 cm step right at the boundary.
        for frame in 12..24 {
            x[frame] = 0.10;
        }
        let before = x[12] - x[11];
        smooth_handoffs(&stances, &mut x, &mut z, 30.0);
        let after = x[12] - x[11];
        assert!(
            after.abs() < before.abs(),
            "step at the boundary grew: {before:.4} -> {after:.4}"
        );
    }

    /// `_hermite` matches the endpoint values.
    #[test]
    fn hermite_matches_its_endpoints() {
        let values = hermite(0.0, 0.1, 1.0, -0.1, 5);
        assert_eq!(values.len(), 5);
        assert!(values.iter().all(|v| v.is_finite()));
        // Monotone-ish between the two endpoint values.
        assert!(values.iter().all(|v| *v > -0.5 && *v < 1.5));
    }

    /// `couple_anchors`: two stances separated by a short gap must not keep
    /// disagreeing anchors, and no stance may shift more than 12 cm.
    #[test]
    fn coupled_stances_meet_across_a_short_gap() {
        let frames = 20;
        // Two stances: 0..8 and 10..18, a two frame release gap between them.
        let stances = vec![(0usize, 8usize), (10usize, 18usize)];
        let mut anchors = vec![[0.0f32; 3]; frames];
        let mut support = vec![[0.0f32; 3]; frames];
        // The second stance's foot is pinned 10 cm away from the first.
        for frame in 0..8 {
            anchors[frame] = [0.0, 0.0, 0.0];
            support[frame] = [0.0, 0.0, 0.0];
        }
        for frame in 10..18 {
            anchors[frame] = [0.10, 0.0, 0.0];
            // Same instant in the source: the disagreement is the anchor's.
            support[frame] = [0.0, 0.0, 0.0];
        }
        let adjustments = couple_anchor_offsets(&stances, &anchors, &support, 30.0);
        let left_end = (anchors[7][0] + adjustments[0][0]) - support[7][0];
        let right_start = (anchors[10][0] + adjustments[1][0]) - support[10][0];
        // The regulariser trades agreement against shifting either stance, so the
        // residual is not zero, but the disagreement has to shrink a lot.
        assert!(
            (left_end - right_start).abs() < 0.05,
            "endpoints still differ by {:.3} m ({left_end:.3} vs {right_start:.3})",
            (left_end - right_start).abs()
        );
        for delta in &adjustments {
            let length = (delta[0] * delta[0] + delta[1] * delta[1]).sqrt();
            assert!(length <= 0.12 + 1e-6, "stance shifted {length:.3} m");
        }
    }

    /// A long swing gap leaves the two stances independent.
    #[test]
    fn distant_stances_are_not_coupled() {
        let frames = 40;
        let stances = vec![(0usize, 8usize), (30usize, 38usize)];
        let mut anchors = vec![[0.0f32; 3]; frames];
        let mut support = vec![[0.0f32; 3]; frames];
        for frame in 0..8 {
            anchors[frame] = [0.0, 0.0, 0.0];
        }
        for frame in 30..38 {
            anchors[frame] = [0.10, 0.0, 0.0];
        }
        let adjustments = couple_anchor_offsets(&stances, &anchors, &support, 30.0);
        for delta in &adjustments {
            assert!(delta[0].abs() < 1e-4 && delta[1].abs() < 1e-4, "adjusted {delta:?}");
        }
    }

    /// `rolling_anchor`: during a pure heel-to-toe roll (no sliding) the
    /// integrated contact location must follow the observed support point, which
    /// is what makes the foot stay put while the weight transfers.
    #[test]
    fn rolling_path_follows_a_clean_roll() {
        let frames = 11;
        // Toe at x = 0, heel at x = 1, 20 cm apart.
        let separations: Vec<[f32; 3]> = (0..frames).map(|_| [0.2, 0.0, 0.0]).collect();
        // Weight rolls from the toe (0) to the heel (1).
        let weights: Vec<f32> = (0..frames).map(|i| i as f32 / (frames - 1) as f32).collect();
        // The observed support point rolls with it.
        let observed: Vec<[f32; 3]> = weights.iter().map(|w| [0.2 * w, 0.0, 0.0]).collect();
        let path = rolling_path(&separations, &weights, &observed, 0.0);
        for (frame, anchor) in path.iter().enumerate() {
            assert!(
                (anchor[0] - observed[frame][0]).abs() < 1e-5,
                "frame {frame}: anchor {} vs observed {}",
                anchor[0],
                observed[frame][0]
            );
            assert_eq!(anchor[1], 0.0, "the anchor sits on the floor");
        }
    }

    /// A flat foot does not move its anchor at all.
    #[test]
    fn rolling_path_is_still_for_a_flat_foot() {
        let frames = 8;
        let separations: Vec<[f32; 3]> = (0..frames).map(|_| [0.18, 0.0, 0.0]).collect();
        let weights = vec![0.5f32; frames];
        let observed: Vec<[f32; 3]> = (0..frames).map(|_| [0.09, 0.0, 0.0]).collect();
        let path = rolling_path(&separations, &weights, &observed, 0.0);
        // The offset aligns the anchor with the observed contact point, and with
        // no weight transfer the anchor never moves from there.
        for anchor in &path {
            assert!((anchor[0] - 0.09).abs() < 1e-5, "anchor at {}", anchor[0]);
        }
    }

    /// `support_weights`: flat foot is 0.5, a rolled foot moves the weight to
    /// whichever point is lower, and the 5 mm deadband keeps skin noise out.
    #[test]
    fn support_weight_rolls_towards_the_lower_point() {
        let flat = support_weight([0.0, 0.05, 0.0], [0.0, 0.05, 0.0]);
        assert!((flat - 0.5).abs() < 1e-6, "flat foot {flat}");

        // Toe 30 mm lower than the heel: weight goes to the toe.
        let toe_down = support_weight([0.0, 0.02, 0.0], [0.0, 0.05, 0.0]);
        assert!(toe_down < 0.05, "toe down {toe_down}");

        // Heel 30 mm lower: weight goes the other way.
        let heel_down = support_weight([0.0, 0.05, 0.0], [0.0, 0.02, 0.0]);
        assert!(heel_down > 0.95, "heel down {heel_down}");

        // 3 mm difference is inside the deadband.
        let noise = support_weight([0.0, 0.047, 0.0], [0.0, 0.05, 0.0]);
        assert!((noise - 0.5).abs() < 1e-6, "noise {noise}");
    }

    /// `gap` in the reference presets: a one frame dropout inside a stance is
    /// bridged, a long one is not.
    #[test]
    fn short_dropouts_are_bridged() {
        let mut contact = vec![true, true, false, true, true];
        bridge_gaps(&mut contact, 1);
        assert_eq!(contact, vec![true, true, true, true, true]);

        let mut long = vec![true, true, false, false, true, true];
        bridge_gaps(&mut long, 1);
        assert_eq!(long, vec![true, true, false, false, true, true]);
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
