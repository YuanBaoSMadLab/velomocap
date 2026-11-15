//! Palm contacts, ported from the reference's `hand_lock.py` (RGB MoCap, MIT).
//!
//! Two pieces are reproduced here:
//!
//! * `palm_offsets` / `samples`: the contact point is **not** the wrist joint but
//!   a registered palm-surface point,
//!
//!   ```text
//!   ids   = vertices weighted > 0.7 to the wrist and at least 15 mm outward
//!   proxy = the 24 lowest of those                      (the palm underside)
//!   offset = mean(shaped[proxy]) - rest[wrist]          (wrist-local)
//!   palm   = wrist_world + R_wrist_world · offset
//!   ```
//!
//! * `plan`: which frames are a real support. Speed alone is not evidence — a
//!   hand resting at the actor's side is stationary too, so the reference also
//!   requires the palm to be at least 12 cm below its shoulder and at least 30 cm
//!   away from the hips (horizontally). Flat ground additionally requires the
//!   palm to be within 15 cm of the floor. Short gaps are bridged and runs
//!   shorter than a quarter second are dropped.
//!
//! Without this, every frame where the actor stands with their arms down reads as
//! a hand contact, and the arm solver pins the hands to their sides.

use anyhow::Result;
use velomocap_body::{BodyModel, Skeleton};

/// One palm's contact runs, as `(begin, end)` frame spans, one list per hand.
pub type HandRuns = [Vec<(usize, usize)>; 2];

/// The result of `plan`.
#[derive(Clone, Debug, Default)]
pub struct HandPlan {
    /// `active[frame][hand]`.
    pub active: Vec<[bool; 2]>,
    pub runs: HandRuns,
}

/// Wrist-local palm offsets, one per hand (left, right).
#[derive(Clone, Copy, Debug)]
pub struct PalmGeometry {
    offsets: [[f32; 3]; 2],
}

impl PalmGeometry {
    /// Build the offsets for a given shape, the reference's `palm_offsets`.
    pub fn build(body: &BodyModel, betas: &[f32]) -> Result<Self> {
        let weights = body.lbs_weights();
        let vertices = body.num_vertices;
        let joints = body.num_joints;
        if weights.len() < vertices * joints {
            anyhow::bail!("skinning weights are missing for the palm probes");
        }
        let shaped = body.shaped_vertices(betas);
        let rest = body.rest_joints(betas);
        let mut offsets = [[0.0f32; 3]; 2];
        for (hand, (wrist, sign)) in [(20usize, 1.0f32), (21, -1.0)].into_iter().enumerate() {
            if wrist >= joints {
                anyhow::bail!("SMPL-H wrist joints are missing");
            }
            let rest_x = rest[wrist * 3];
            let mut ids: Vec<usize> = Vec::new();
            for vertex in 0..vertices {
                if weights[vertex * joints + wrist] <= 0.7 {
                    continue;
                }
                // Outward from the wrist on this hand's side: the thumb side of
                // the palm, which is where the contact surface sits.
                if (shaped[vertex * 3] - rest_x) * sign > 0.015 {
                    ids.push(vertex);
                }
            }
            if ids.len() < 4 {
                anyhow::bail!("SMPL-H palm surface is unavailable");
            }
            // The lowest 24 vertices: the palm underside, not the fingers.
            ids.sort_by(|a, b| {
                shaped[a * 3 + 1]
                    .partial_cmp(&shaped[b * 3 + 1])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            ids.truncate(24.min(ids.len()));
            let count = ids.len().max(1) as f32;
            let mut mean = [0.0f32; 3];
            for vertex in &ids {
                for axis in 0..3 {
                    mean[axis] += shaped[vertex * 3 + axis];
                }
            }
            for axis in 0..3 {
                offsets[hand][axis] = mean[axis] / count - rest[wrist * 3 + axis];
            }
        }
        Ok(Self { offsets })
    }

    pub fn offsets(&self) -> [[f32; 3]; 2] {
        self.offsets
    }

    /// The palm point of each hand for one frame.
    pub fn points(&self, skeleton: &Skeleton) -> [[f32; 3]; 2] {
        let mut out = [[0.0f32; 3]; 2];
        for (hand, wrist) in [20usize, 21].into_iter().enumerate() {
            let offset = self.offsets[hand];
            let r = &skeleton.rotations[wrist * 9..wrist * 9 + 9];
            for axis in 0..3 {
                out[hand][axis] = skeleton.positions[wrist * 3 + axis]
                    + r[axis * 3] * offset[0]
                    + r[axis * 3 + 1] * offset[1]
                    + r[axis * 3 + 2] * offset[2];
            }
        }
        out
    }

    /// The world-space palm offset of each hand (the wrist-local offset rotated
    /// by the wrist's world rotation), which is what turns a palm target back
    /// into a wrist target.
    pub fn world_offsets(&self, skeleton: &Skeleton) -> [[f32; 3]; 2] {
        let mut out = [[0.0f32; 3]; 2];
        for (hand, wrist) in [20usize, 21].into_iter().enumerate() {
            let offset = self.offsets[hand];
            let r = &skeleton.rotations[wrist * 9..wrist * 9 + 9];
            for axis in 0..3 {
                out[hand][axis] = r[axis * 3] * offset[0]
                    + r[axis * 3 + 1] * offset[1]
                    + r[axis * 3 + 2] * offset[2];
            }
        }
        out
    }
}

/// Gaussian smoothing of a scalar series, the numpy `gaussian_filter1d` the
/// reference uses with `sigma = max(0.5, fps * 0.04)` and `mode='nearest'`.
fn smooth_scalar(values: &[f32], sigma: f32) -> Vec<f32> {
    if values.len() < 2 {
        return values.to_vec();
    }
    let sigma = sigma.max(0.5);
    let radius = (sigma * 3.0).ceil() as isize;
    let kernel: Vec<f32> = (-radius..=radius)
        .map(|offset| {
            let o = offset as f32;
            (-(o * o) / (2.0 * sigma * sigma)).exp()
        })
        .collect();
    let total: f32 = kernel.iter().sum();
    (0..values.len())
        .map(|frame| {
            let mut sum = 0.0f32;
            for (slot, offset) in (-radius..=radius).enumerate() {
                let at = (frame as isize + offset).clamp(0, values.len() as isize - 1) as usize;
                sum += values[at] * kernel[slot];
            }
            sum / total
        })
        .collect()
}

/// Frame spans of `true` in a mask.
fn runs_of(mask: &[bool]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start: Option<usize> = None;
    for (index, flag) in mask.iter().enumerate() {
        match (flag, start) {
            (true, None) => start = Some(index),
            (false, Some(begin)) => {
                out.push((begin, index));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(begin) = start {
        out.push((begin, mask.len()));
    }
    out
}

/// Which frames are real palm supports, the reference's `plan`.
///
/// `points[frame][hand]` is the palm surface point, `shoulders`/`hips` are the
/// SMPL-H shoulder (16/17) and hip (1/2) joints of the same frame. `floor` is
/// the support height when the ground is flat (`None` for multi-level supports).
pub fn plan(
    points: &[[[f32; 3]; 2]],
    shoulders: &[[[f32; 3]; 2]],
    hips: &[[[f32; 3]; 2]],
    fps: f32,
    floor: Option<f32>,
    speed_limit: f32,
) -> HandPlan {
    let count = points.len();
    let mut plan = HandPlan {
        active: vec![[false; 2]; count],
        runs: [Vec::new(), Vec::new()],
    };
    if count < 2 || shoulders.len() != count || hips.len() != count {
        return plan;
    }
    let fps = fps.max(1.0);
    let sigma = (fps * 0.04).max(0.5);
    for hand in 0..2 {
        // Smoothed palm speed, then the two geometric gates. The gates are what
        // separates a planted hand from one hanging at the side.
        let mut smoothed = vec![[0.0f32; 3]; count];
        for axis in 0..3 {
            let series: Vec<f32> = points.iter().map(|frame| frame[hand][axis]).collect();
            let filtered = smooth_scalar(&series, sigma);
            for frame in 0..count {
                smoothed[frame][axis] = filtered[frame];
            }
        }
        let mut speed = vec![0.0f32; count];
        for frame in 0..count {
            let previous = frame.saturating_sub(1);
            let next = (frame + 1).min(count - 1);
            let span = (next - previous).max(1) as f32;
            let mut delta = [0.0f32; 3];
            for axis in 0..3 {
                delta[axis] = (smoothed[next][axis] - smoothed[previous][axis]) / span * fps;
            }
            speed[frame] = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
        }
        let mut active = vec![false; count];
        for frame in 0..count {
            let palm = points[frame][hand];
            let shoulder = shoulders[frame][hand];
            let hip = hips[frame][hand];
            // Below the shoulder by a hand's length: not an overhead gesture.
            let below = palm[1] < shoulder[1] - 0.12;
            // Away from the body: a hand at the side is not an external support.
            let dx = palm[0] - hip[0];
            let dz = palm[2] - hip[2];
            let away = (dx * dx + dz * dz).sqrt() > 0.30;
            active[frame] = speed[frame] < speed_limit && below && away;
            if let Some(floor) = floor {
                active[frame] = active[frame] && (palm[1] - floor).abs() < 0.15;
            }
        }
        // Bridge a short drop-out between two contacts, but only when the palm
        // barely moved across it and never moved fast inside it.
        let gaps = runs_of(&active.iter().map(|flag| !flag).collect::<Vec<_>>());
        for (begin, end) in gaps {
            if begin == 0 || end == count {
                continue;
            }
            if end - begin > (fps * 0.10).round().max(1.0) as usize {
                continue;
            }
            let first = points[begin][hand];
            let last = points[end - 1][hand];
            let jump = ((first[0] - last[0]).powi(2)
                + (first[1] - last[1]).powi(2)
                + (first[2] - last[2]).powi(2))
            .sqrt();
            let fastest = speed[begin..end]
                .iter()
                .cloned()
                .fold(0.0f32, f32::max);
            if jump < 0.025 && fastest < 0.35 {
                for slot in begin..end {
                    active[slot] = true;
                }
            }
        }
        // A quarter second is the shortest believable stance.
        let min_run = (fps * 0.25).round().max(4.0) as usize;
        for (begin, end) in runs_of(&active) {
            if end - begin < min_run {
                for slot in begin..end {
                    active[slot] = false;
                }
            }
        }
        for frame in 0..count {
            plan.active[frame][hand] = active[frame];
        }
        plan.runs[hand] = runs_of(&active);
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_shoulders(count: usize) -> Vec<[[f32; 3]; 2]> {
        vec![[[0.2, 1.35, 0.0], [-0.2, 1.35, 0.0]]; count]
    }

    fn flat_hips(count: usize) -> Vec<[[f32; 3]; 2]> {
        vec![[[0.1, 0.95, 0.0], [-0.1, 0.95, 0.0]]; count]
    }

    /// A hand held out over a table is a support; the same hand hanging at the
    /// actor's side is not, even though both are perfectly still.
    #[test]
    fn a_hand_at_the_side_is_not_a_contact() {
        let count = 60;
        let mut points = vec![[[0.2, 1.2, 0.0], [-0.2, 1.2, 0.0]]; count];
        for frame in 0..count {
            // Left palm planted on the floor 0.55 m in front of the hip.
            points[frame][0] = [0.30, 0.04, 0.55];
            // Right palm hanging beside the hip, 0.15 m away from it.
            points[frame][1] = [-0.25, 0.04, 0.0];
        }
        let plan = plan(
            &points,
            &flat_shoulders(count),
            &flat_hips(count),
            30.0,
            Some(0.0),
            0.18,
        );
        assert!(plan.active[30][0], "the planted palm should be a contact");
        assert!(!plan.active[30][1], "a palm at the side is not a support");
        assert_eq!(plan.runs[0], vec![(0, count)]);
        assert!(plan.runs[1].is_empty(), "{:?}", plan.runs[1]);
    }

    /// A hand moving through the air is skipped, and a one frame drop-out inside
    /// a contact is bridged instead of splitting the stance.
    #[test]
    fn bridges_a_short_dropout_but_not_a_moving_hand() {
        let count = 40;
        let mut points = vec![[[0.3, 0.04, 0.55], [-0.3, 1.3, 0.0]]; count];
        // One frame of small jitter in the middle of an otherwise planted palm.
        points[20][0] = [0.31, 0.045, 0.55];
        // The right hand sweeps across at 1 m/s: never a contact.
        for frame in 0..count {
            points[frame][1] = [-0.3 + frame as f32 * 0.033, 0.04, 0.55];
        }
        let plan = plan(
            &points,
            &flat_shoulders(count),
            &flat_hips(count),
            30.0,
            Some(0.0),
            0.18,
        );
        assert_eq!(
            plan.runs[0],
            vec![(0, count)],
            "the jitter frame should be bridged: {:?}",
            plan.runs[0]
        );
        assert!(plan.runs[1].is_empty(), "{:?}", plan.runs[1]);
    }

    /// The palm offset has to point away from the wrist, downward and outward,
    /// and be the size of a palm.
    #[test]
    fn palm_offsets_sit_on_the_palm() {
        let path = [
            "models/body_models/smplh/neutral/model.npz",
            "../../models/body_models/smplh/neutral/model.npz",
        ]
        .iter()
        .find(|candidate| std::path::Path::new(candidate).is_file());
        let Some(path) = path else {
            return;
        };
        let body = BodyModel::load(path, 16).expect("SMPL-H");
        let betas = vec![0.0f32; 16];
        let geometry = PalmGeometry::build(&body, &betas).expect("palm geometry");
        let mut pose = velomocap_body::Pose::neutral(body.num_joints, 16);
        pose.translation = [0.0, 0.0, 0.0];
        let skeleton = body.forward_joints(&pose);
        let points = geometry.points(&skeleton);
        for hand in 0..2 {
            let wrist = if hand == 0 { 20 } else { 21 };
            let wrist_at = [
                skeleton.positions[wrist * 3],
                skeleton.positions[wrist * 3 + 1],
                skeleton.positions[wrist * 3 + 2],
            ];
            let offset = [
                points[hand][0] - wrist_at[0],
                points[hand][1] - wrist_at[1],
                points[hand][2] - wrist_at[2],
            ];
            let length = (offset[0] * offset[0] + offset[1] * offset[1] + offset[2] * offset[2])
                .sqrt();
            assert!(
                (0.02..0.15).contains(&length),
                "palm offset {offset:?} is not palm-sized"
            );
            assert!(offset[1] <= 0.0, "the palm point must be below the wrist");
        }
    }

    #[test]
    fn palm_points_follow_the_wrist_rotation() {
        let geometry = PalmGeometry {
            offsets: [[0.0, -0.05, 0.0], [0.0, -0.05, 0.0]],
        };
        let mut skeleton = Skeleton {
            positions: vec![0.0; 22 * 3],
            rotations: vec![0.0; 22 * 9],
        };
        for joint in 0..22 {
            skeleton.rotations[joint * 9] = 1.0;
            skeleton.rotations[joint * 9 + 4] = 1.0;
            skeleton.rotations[joint * 9 + 8] = 1.0;
        }
        skeleton.positions[20 * 3 + 1] = 1.0;
        skeleton.positions[21 * 3 + 1] = 1.0;
        let points = geometry.points(&skeleton);
        assert!((points[0][1] - 0.95).abs() < 1e-6, "{:?}", points);
        // A 90 degree roll about x carries the palm offset onto -z.
        let r = &mut skeleton.rotations[20 * 9..20 * 9 + 9];
        r[4] = 0.0;
        r[5] = -1.0;
        r[7] = 1.0;
        r[8] = 0.0;
        let rolled = geometry.points(&skeleton);
        assert!((rolled[0][0]).abs() < 1e-6, "{:?}", rolled);
        assert!((rolled[0][1] - 1.0).abs() < 1e-6, "{:?}", rolled);
        assert!((rolled[0][2] + 0.05).abs() < 1e-6, "{:?}", rolled);
    }
}
