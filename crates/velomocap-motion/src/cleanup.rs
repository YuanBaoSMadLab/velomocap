//! Temporal cleanup: plant the feet and take the jitter out of a track.
//!
//! SAM 3D Body reconstructs each frame independently, so a standing foot drifts
//! a few centimetres per frame and the whole body jitters. The Qt application
//! solves this with its motion-correction stage (foot/hand locks). This is the
//! same idea in miniature, applied to the mesh after a person is reconstructed:
//!
//! 1. find the frames where a foot is on the ground (low and not moving),
//! 2. group those frames into contact runs and pin every frame of a run to the
//!    run's median foot position,
//! 3. blend the correction in and out so a step does not pop,
//! 4. smooth the root vertically and horizontally where the person is airborne.
//!
//! Everything is a rigid translation of the whole body, so the mesh, the joints
//! and the keypoints cannot disagree afterwards.

use std::sync::Arc;

use crate::sam::{PersonFrame, PersonTrack};

/// Keys into the 70-keypoint rig used for contact detection.
const LEFT_ANKLE: usize = 13;
const RIGHT_ANKLE: usize = 14;
const LEFT_TOE: usize = 15;
const RIGHT_TOE: usize = 18;
const LEFT_HEEL: usize = 17;
const RIGHT_HEEL: usize = 20;

/// Ground sample of each foot: ankle, toe and heel. The *lowest* of the three
/// decides contact (high heels, pointed toes), while the ankle is what gets
/// pinned so the whole foot stays put.
const FOOT_KEYS: [[usize; 3]; 2] = [
    [LEFT_ANKLE, LEFT_TOE, LEFT_HEEL],
    [RIGHT_ANKLE, RIGHT_TOE, RIGHT_HEEL],
];

#[derive(Clone, Debug)]
pub struct CleanupConfig {
    /// A foot this close to its lowest point counts as "on the ground".
    pub contact_height: f32,
    /// Horizontal speed above which a low foot is sliding, not planted.
    pub contact_speed: f32,
    /// Frames used to fade a correction in and out.
    pub blend_frames: usize,
    /// Root smoothing where nothing is in contact (0 = off, 1 = frozen).
    pub airborne_smoothing: f32,
    /// Never move a frame further than this.
    pub max_correction: f32,
}

impl Default for CleanupConfig {
    fn default() -> Self {
        Self {
            contact_height: 0.08,
            contact_speed: 0.06,
            blend_frames: 3,
            airborne_smoothing: 0.25,
            max_correction: 0.35,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct CleanupReport {
    pub contacts: usize,
    /// Frames where at least one foot was low enough to be considered planted.
    pub low_frames: usize,
    /// Frames examined across all people.
    pub frames: usize,
    pub corrected_m: f64,
    pub max_correction_m: f32,
    pub smoothed: usize,
    /// Mean horizontal foot travel per frame while the foot is planted, before
    /// and after the correction. This is the number a user sees as "sliding".
    pub slide_before_m: f32,
    pub slide_after_m: f32,
}

impl CleanupReport {
    pub fn line(&self) -> String {
        if self.contacts == 0 {
            return String::new();
        }
        format!(
            "{} foot contacts · {:.2} m corrected (max {:.2} m) · slide {:.1} -> {:.1} mm/frame",
            self.contacts,
            self.corrected_m,
            self.max_correction_m,
            self.slide_before_m * 1000.0,
            self.slide_after_m * 1000.0
        )
    }
}

/// Apply contact locking and smoothing to every person in place.
pub fn stabilise(people: &mut [PersonTrack], config: &CleanupConfig) -> CleanupReport {
    let mut report = CleanupReport::default();
    for person in people.iter_mut() {
        stabilise_person(person, config, &mut report);
    }
    report
}

fn stabilise_person(person: &mut PersonTrack, config: &CleanupConfig, report: &mut CleanupReport) {
    let indices: Vec<usize> = person
        .frames
        .iter()
        .enumerate()
        .filter_map(|(index, slot)| slot.is_some().then_some(index))
        .collect();
    if indices.len() < 3 {
        return;
    }

    let foot = |frame: &PersonFrame, key: usize| -> Option<[f32; 3]> {
        let at = key * 3;
        let point = frame.keypoints.get(at..at + 3)?;
        Some([point[0], point[1], point[2]])
    };

    // Pass 1: smooth the root trajectory. SAM places every frame on its own, so
    // the pelvis wanders a few centimetres per frame even when the actor stands
    // still; the Qt build gets a coherent trajectory from FlowHMR and smooths
    // it further. A binomial [1 4 6 4 1] kernel is the same idea at this level.
    let roots: Vec<[f32; 3]> = indices
        .iter()
        .map(|index| {
            let frame = person.frame(*index).expect("indexed frame");
            [frame.joints[0], frame.joints[1], frame.joints[2]]
        })
        .collect();
    let smoothed = smooth_trajectory(&roots);
    for (position, &index) in indices.iter().enumerate() {
        let delta = [
            smoothed[position][0] - roots[position][0],
            smoothed[position][1] - roots[position][1],
            smoothed[position][2] - roots[position][2],
        ];
        if delta == [0.0, 0.0, 0.0] {
            continue;
        }
        let frame = Arc::make_mut(person.frames[index].as_mut().expect("frame"));
        apply_translation(frame, delta);
        report.smoothed += 1;
    }

    // Ground level for this person: the 10th percentile of the lowest foot per
    // frame. A global minimum would be dragged down by one bad frame and leave
    // every other frame "airborne", which is exactly what an outlier does here.
    let mut lowest_per_frame: Vec<f32> = Vec::with_capacity(indices.len());
    for &index in &indices {
        let frame = person.frame(index).expect("indexed frame");
        let mut lowest = f32::INFINITY;
        for keys in FOOT_KEYS {
            for key in keys {
                if let Some(point) = foot(frame, key) {
                    lowest = lowest.min(point[1]);
                }
            }
        }
        if lowest.is_finite() {
            lowest_per_frame.push(lowest);
        }
    }
    if lowest_per_frame.is_empty() {
        return;
    }
    lowest_per_frame.sort_by(|a, b| a.total_cmp(b));
    let ground = lowest_per_frame[lowest_per_frame.len() / 10];

    // Per-frame foot state, in track order.
    let mut states: Vec<FootSample> = Vec::with_capacity(indices.len());
    let mut previous: [Option<[f32; 3]>; 2] = [None, None];
    for &index in &indices {
        let frame = person.frame(index).expect("indexed frame");
        let ankles = [foot(frame, LEFT_ANKLE), foot(frame, RIGHT_ANKLE)];
        let mut low = [false, false];
        for side in 0..2 {
            let Some(point) = ankles[side] else { continue };
            // The first observation has no speed yet; a low foot there is
            // treated as planted rather than forcing a one-frame gap.
            let slow = previous[side]
                .map(|old| {
                    let dx = point[0] - old[0];
                    let dz = point[2] - old[2];
                    (dx * dx + dz * dz).sqrt() <= config.contact_speed
                })
                .unwrap_or(true);
            low[side] = point[1] <= ground + config.contact_height && slow;
        }
        previous = ankles;
        states.push(FootSample {
            index,
            low,
            position: ankles,
        });
    }

    report.frames += states.len();
    report.low_frames += states.iter().filter(|state| state.low[0] || state.low[1]).count();

    // Close one-frame gaps: per-frame height noise would otherwise split a
    // planted interval into singletons that cannot be anchored.
    for _ in 0..2 {
        let previous: Vec<[bool; 2]> = states.iter().map(|state| state.low).collect();
        for position in 1..states.len().saturating_sub(1) {
            for side in 0..2 {
                if !states[position].low[side]
                    && previous[position - 1][side]
                    && previous[position + 1][side]
                {
                    states[position].low[side] = true;
                }
            }
        }
    }

    // How much a planted foot moves per frame before the correction: the
    // visible drift.
    report.slide_before_m = mean_slide(&states);

    // Contact runs per side: [start, end] into `states`.
    let mut corrections: Vec<[f32; 2]> = vec![[0.0; 2]; states.len()];
    let mut weight: Vec<f32> = vec![0.0; states.len()];
    for side in 0..2 {
        let mut run_start: Option<usize> = None;
        for position in 0..=states.len() {
            let in_contact = position < states.len() && states[position].low[side];
            match (run_start, in_contact) {
                (None, true) => run_start = Some(position),
                (Some(start), false) => {
                    let anchor = median(&states[start..position], side);
                    if let Some(anchor) = anchor {
                        report.contacts += position - start;
                        for slot in start..position {
                            let Some(point) = states[slot].position[side] else { continue };
                            let dx = (anchor[0] - point[0]).clamp(
                                -config.max_correction,
                                config.max_correction,
                            );
                            let dz = (anchor[2] - point[2]).clamp(
                                -config.max_correction,
                                config.max_correction,
                            );
                            corrections[slot][0] += dx;
                            corrections[slot][1] += dz;
                            weight[slot] += 1.0;
                        }
                    }
                    run_start = None;
                }
                _ => {}
            }
        }
    }

    // Average the two sides, then blend across the boundary frames.
    let mut shift: Vec<[f32; 2]> = corrections
        .iter()
        .zip(&weight)
        .map(|(value, weight)| {
            if *weight > 0.0 {
                [value[0] / weight, value[1] / weight]
            } else {
                [0.0, 0.0]
            }
        })
        .collect();
    let blend = config.blend_frames.max(1);
    for position in 0..shift.len() {
        if weight[position] > 0.0 {
            continue;
        }
        let before = (1..=blend)
            .find_map(|back| position.checked_sub(back).map(|at| (at, back)))
            .filter(|(at, _)| weight[*at] > 0.0);
        let after = (1..=blend)
            .find_map(|forward| {
                let at = position + forward;
                (at < shift.len() && weight[at] > 0.0).then_some((at, forward))
            });
        shift[position] = match (before, after) {
            (Some((a, da)), Some((b, db))) => {
                let t = da as f32 / (da + db) as f32;
                [
                    shift[a][0] * (1.0 - t) + shift[b][0] * t,
                    shift[a][1] * (1.0 - t) + shift[b][1] * t,
                ]
            }
            (Some((a, _)), None) => shift[a],
            (None, Some((b, _))) => shift[b],
            (None, None) => [0.0, 0.0],
        };
    }

    // Pass 2: pin each contact run, so a planted foot cannot slide.
    for position in 0..states.len() {
        let delta = [shift[position][0], 0.0, shift[position][1]];
        if delta == [0.0, 0.0, 0.0] {
            continue;
        }
        let index = states[position].index;
        let frame = Arc::make_mut(person.frames[index].as_mut().expect("frame"));
        apply_translation(frame, delta);
        let magnitude = (delta[0] * delta[0] + delta[2] * delta[2]).sqrt();
        report.corrected_m += magnitude as f64;
        report.max_correction_m = report.max_correction_m.max(magnitude);
    }

    // Re-measure on the corrected frames.
    let after: Vec<FootSample> = states
        .iter()
        .map(|state| {
            let frame = person.frame(state.index).expect("indexed frame");
            FootSample {
                index: state.index,
                low: state.low,
                position: [foot(frame, LEFT_ANKLE), foot(frame, RIGHT_ANKLE)],
            }
        })
        .collect();
    report.slide_after_m = mean_slide(&after);
}

/// Mean horizontal travel of a planted foot between consecutive frames.
fn mean_slide(states: &[FootSample]) -> f32 {
    let mut total = 0.0f32;
    let mut count = 0usize;
    let mut previous: [Option<[f32; 3]>; 2] = [None, None];
    for state in states {
        for side in 0..2 {
            let Some(point) = state.position[side] else { continue };
            if let Some(old) = previous[side] {
                if state.low[side] {
                    let dx = point[0] - old[0];
                    let dz = point[2] - old[2];
                    total += (dx * dx + dz * dz).sqrt();
                    count += 1;
                }
            }
            previous[side] = Some(point);
        }
    }
    if count == 0 {
        0.0
    } else {
        total / count as f32
    }
}

/// `[1 8 28 56 70 56 28 8 1] / 256` along the track, with edge clamping.
///
/// Nine frames is about a third of a second at 30 fps: enough to remove the
/// per-frame placement noise, short enough that a real step is not smeared.
fn smooth_trajectory(roots: &[[f32; 3]]) -> Vec<[f32; 3]> {
    const KERNEL: [f32; 9] = [1.0, 8.0, 28.0, 56.0, 70.0, 56.0, 28.0, 8.0, 1.0];
    let count = roots.len();
    let mut out = Vec::with_capacity(count);
    for index in 0..count {
        let mut sum = [0.0f32; 3];
        for (offset, weight) in KERNEL.iter().enumerate() {
            let source = index as isize + offset as isize - 4;
            let clamped = source.clamp(0, count as isize - 1) as usize;
            for axis in 0..3 {
                sum[axis] += roots[clamped][axis] * weight;
            }
        }
        out.push([sum[0] / 256.0, sum[1] / 256.0, sum[2] / 256.0]);
    }
    out
}

/// One frame's foot observation.
struct FootSample {
    index: usize,
    /// Which of the two ankles is planted.
    low: [bool; 2],
    position: [Option<[f32; 3]>; 2],
}

fn median(states: &[FootSample], side: usize) -> Option<[f32; 3]> {
    let mut xs: Vec<f32> = Vec::new();
    let mut ys: Vec<f32> = Vec::new();
    let mut zs: Vec<f32> = Vec::new();
    for state in states {
        if let Some(point) = state.position[side] {
            xs.push(point[0]);
            ys.push(point[1]);
            zs.push(point[2]);
        }
    }
    if xs.is_empty() {
        return None;
    }
    Some([median_of(&mut xs), median_of(&mut ys), median_of(&mut zs)])
}

fn median_of(values: &mut [f32]) -> f32 {
    values.sort_by(|a, b| a.total_cmp(b));
    values[values.len() / 2]
}

fn apply_translation(frame: &mut PersonFrame, delta: [f32; 3]) {
    for buffer in [&mut frame.verts, &mut frame.joints, &mut frame.keypoints] {
        for point in buffer.chunks_exact_mut(3) {
            point[0] += delta[0];
            point[1] += delta[1];
            point[2] += delta[2];
        }
    }
    frame.cam_t[0] += delta[0];
    frame.cam_t[1] -= delta[1];
    frame.cam_t[2] += delta[2];
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sam::PersonFrame;

    fn frame_at(index: usize, ankle_x: f32, airborne: bool) -> (usize, Arc<PersonFrame>) {
        let y = if airborne { -0.85 } else { -1.30 };
        let mut keypoints = vec![0.0f32; 70 * 3];
        keypoints[13 * 3] = ankle_x;
        keypoints[13 * 3 + 1] = y;
        keypoints[13 * 3 + 2] = 8.0;
        keypoints[14 * 3] = ankle_x + 0.2;
        keypoints[14 * 3 + 1] = y;
        keypoints[14 * 3 + 2] = 8.0;
        for key in [LEFT_TOE, RIGHT_TOE] {
            keypoints[key * 3] = ankle_x;
            keypoints[key * 3 + 1] = y - 0.02;
            keypoints[key * 3 + 2] = 8.0;
        }
        let joints = vec![0.0f32; 127 * 3];
        (
            index,
            Arc::new(PersonFrame {
                bbox: [0.0; 4],
                score: 1.0,
                verts: vec![0.0f32; 3],
                joints,
                keypoints,
                body_token: vec![0.0; 1024],
                left_token: Vec::new(),
                right_token: Vec::new(),
                cam_t: [0.0, 0.0, 8.0],
            anchor_cam: [0.0, 0.0, 8.0],
                skel_state: vec![0.0; 127 * 8],
            }),
        )
    }

    /// A planted foot that wanders must end up in one place.
    #[test]
    fn locks_a_drifting_foot() {
        let mut track = PersonTrack {
            id: 0,
            name: "Person 1".into(),
            frames: Vec::new(),
            first_frame: 0,
            last_frame: 0,
            hits: 0,
        };
        for step in 0..11 {
            let drift = step as f32 * 0.01;
            let (index, frame) = frame_at(step, drift, false);
            while track.frames.len() <= index {
                track.frames.push(None);
            }
            track.frames[index] = Some(frame);
            track.hits += 1;
        }
        let mut people = vec![track];
        let report = stabilise(&mut people, &CleanupConfig::default());
        let positions: Vec<f32> = people[0]
            .frames
            .iter()
            .flatten()
            .map(|frame| frame.keypoints[13 * 3])
            .collect();
        let spread = positions
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max)
            - positions.iter().cloned().fold(f32::INFINITY, f32::min);
        eprintln!(
            "{} contacts, spread {:.4} m, {:?}",
            report.contacts, spread, report.line()
        );
        assert!(report.contacts >= 8);
        assert!(spread < 0.02, "foot still drifts by {spread:.3} m");
    }
}
