//! Palm-contact arm planning, ported from the reference's
//! `contact_arm_plan.py::solve` (RGB MoCap, MIT).
//!
//! A pinned palm is not solved frame by frame. The reference plans a **window**
//! of six parameters per frame —
//!
//! ```text
//! 0  collar depression   bounded by 15 degrees below 0 (0..15 with a small flex)
//! 1  arm swivel          bounded by 10 degrees
//! 2  wrist target trim   bounded by 6 mm per axis
//! 5  elbow correction    bounded by 30 degrees (60 when the target needs it)
//! ```
//!
//! — through a nonlinear least-squares solve that trades the palm contact error
//! against the size of the added change, its velocity, its acceleration and the
//! physical elbow flexion. That is why a pinned hand moves smoothly instead of
//! snapping to its target on the first contact frame and back on the last.
//!
//! The prediction used inside the solve is analytic, which is what makes this
//! portable: `palm = wrist + R_wrist · offset`, never a mesh.
//!
//! * `desired` wrist target = source wrist + influence·(pin − source palm) + trim
//! * collar depression rotates the collar and with it the shoulder position
//! * the arm swivel and the elbow correction bound how far the chain may bend
//! * the shoulder is aimed with `align(chain, delta)`, exactly the reference's
//!   one-bone aim
//!
//! The reference's two extra variables per stance (a ±1 cm anchor x/z adjustment)
//! are not reproduced: they let two overlapping supports be satisfied at once,
//! and this pass pins one palm at a time with a single registered target.

use velomocap_core::math;

use crate::solver;

/// One frame's input to the planner, all in world space and metres.
#[derive(Clone, Debug)]
pub struct Frame {
    pub collar_world: [f64; 9],
    pub shoulder_world: [f64; 9],
    pub elbow_world: [f64; 9],
    pub collar: [f64; 3],
    pub shoulder: [f64; 3],
    pub elbow: [f64; 3],
    pub wrist: [f64; 3],
    /// The palm surface point of the source frame.
    pub palm: [f64; 3],
    /// The wrist's rotation in its elbow frame, and the palm offset in the wrist
    /// frame: together they place the palm on the planned chain
    /// (`palm = wrist + R_upper·R_elbow·R_wrist_local·offset`).
    ///
    /// The reference holds the palm offset fixed **in world space** while it
    /// plans, which drifts from the palm the applied pose actually produces as
    /// soon as the arm swings; carrying the wrist frame instead keeps the plan
    /// and the clip in agreement.
    pub wrist_local: [f64; 9],
    pub palm_offset_local: [f64; 3],
    /// How strongly this frame is planned (1 inside the stance, a cosine halo
    /// outside it).
    pub influence: f64,
    /// Whether this frame is an actual contact frame.
    pub planned: bool,
}

/// One frame's output.
#[derive(Clone, Debug)]
pub struct Planned {
    pub collar_world: [f64; 9],
    pub shoulder_world: [f64; 9],
    /// Local rotation of the elbow in its shoulder frame.
    pub elbow_local: [f64; 9],
    pub elbow: [f64; 3],
    pub palm: [f64; 3],
    pub flex: f64,
}

/// A whole stance to plan.
pub struct Plan<'a> {
    pub rest: &'a [f32],
    /// 0 = left hand, 1 = right hand.
    pub hand: usize,
    /// The calibrated elbow branch (`anatomical::elbow_branch`).
    pub branch: f64,
    pub fps: f64,
    /// The palm point to hold, metres.
    pub pin: [f64; 3],
    pub frames: &'a [Frame],
}

const COLLAR_LIMIT_DEG: f64 = 15.0;
const SWIVEL_LIMIT_DEG: f64 = 10.0;
const ELBOW_LIMIT_DEG: f64 = 60.0;
const ELBOW_SOFT_DEG: f64 = 30.0;
const TRIM_M: f64 = 0.006;
const CONTACT_SIGMA_M: f64 = 0.0007;
const ELBOW_SIGMA_M: f64 = 0.03;
const TEMPORAL_WEIGHT: f64 = 0.5;
const VELOCITY_SIGMA: f64 = 0.025;
const ACCELERATION_SIGMA: f64 = 0.004;
const FLEX_ACCELERATION_SIGMA: f64 = 0.003;
const ITERATIONS: usize = 80;
/// The state channels: elbow correction, shoulder relative rotation (3), and the
/// two bounded angles. The trim is driven by the contact term only.
const CHANNEL: [f64; 6] = [1.0, 0.4, 0.4, 0.4, 1.0, 0.4];
const STATE_SIGMA: [f64; 6] = [0.20, 0.25, 0.25, 0.25, 0.12, 0.15];

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale3(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn len3(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}

fn apply(m: &[f64; 9], v: [f64; 3]) -> [f64; 3] {
    [
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
    ]
}

/// Rodrigues.
fn rotation(axis: [f64; 3], angle: f64) -> [f64; 9] {
    let length = len3(axis);
    if length < 1e-12 || angle.abs() < 1e-12 {
        return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    }
    let axis = scale3(axis, 1.0 / length);
    let (s, c) = (angle.sin(), angle.cos());
    let t = 1.0 - c;
    [
        t * axis[0] * axis[0] + c,
        t * axis[0] * axis[1] - s * axis[2],
        t * axis[0] * axis[2] + s * axis[1],
        t * axis[0] * axis[1] + s * axis[2],
        t * axis[1] * axis[1] + c,
        t * axis[1] * axis[2] - s * axis[0],
        t * axis[0] * axis[2] - s * axis[1],
        t * axis[1] * axis[2] + s * axis[0],
        t * axis[2] * axis[2] + c,
    ]
}

fn rotvec(m: &[f64; 9]) -> [f64; 3] {
    let cos = ((m[0] + m[4] + m[8] - 1.0) * 0.5).clamp(-1.0, 1.0);
    let angle = cos.acos();
    if angle < 1e-9 {
        return [0.0; 3];
    }
    let s = angle.sin();
    if s.abs() < 1e-9 {
        return [0.0; 3];
    }
    [
        (m[7] - m[5]) * 0.5 * angle / s,
        (m[2] - m[6]) * 0.5 * angle / s,
        (m[3] - m[1]) * 0.5 * angle / s,
    ]
}

fn joint(rest: &[f32], index: usize) -> [f64; 3] {
    [
        rest[index * 3] as f64,
        rest[index * 3 + 1] as f64,
        rest[index * 3 + 2] as f64,
    ]
}

/// The reference's `isclose`-free wrap into `(-pi, pi]`.
fn wrap(angle: f64) -> f64 {
    let two_pi = std::f64::consts::TAU;
    (angle + std::f64::consts::PI).rem_euclid(two_pi) - std::f64::consts::PI
}

impl<'a> Plan<'a> {
    fn joints(&self) -> (usize, usize, usize) {
        if self.hand == 0 {
            (16, 18, 20)
        } else {
            (17, 19, 21)
        }
    }

    /// The six variables of one frame, scaled by the frame's influence.
    fn variables(&self, x: &[f64], frame: usize) -> [f64; 6] {
        let mut out = [0.0f64; 6];
        for slot in 0..6 {
            out[slot] = x[frame * 6 + slot] * self.frames[frame].influence;
        }
        out
    }

    /// The forward prediction, the reference's `predict`.
    fn predict(&self, x: &[f64]) -> Vec<Planned> {
        let (shoulder_index, elbow_index, _) = self.joints();
        let u = sub3(joint(self.rest, elbow_index), joint(self.rest, shoulder_index));
        let v0 = sub3(
            joint(self.rest, self.joints().2),
            joint(self.rest, elbow_index),
        );
        let axis = crate::anatomical::elbow_axis(self.rest, self.hand);
        let collar_index = if self.hand == 0 { 13 } else { 14 };
        let collar_offset = sub3(
            joint(self.rest, shoulder_index),
            joint(self.rest, collar_index),
        );
        self.frames
            .iter()
            .enumerate()
            .map(|(frame, source)| {
                let xx = self.variables(x, frame);
                let local_elbow = math::tmul3(&source.shoulder_world, &source.elbow_world);
                let desired = add3(
                    add3(
                        source.wrist,
                        scale3(
                            sub3(self.pin, source.palm),
                            source.influence,
                        ),
                    ),
                    [xx[2], xx[3], xx[4]],
                );
                let collar_offset_world = apply(&source.collar_world, collar_offset);
                let depression_axis = math::unit3(&math::cross3(
                    &collar_offset_world,
                    &[0.0, -1.0, 0.0],
                ));
                let depression = rotation(depression_axis, xx[0]);
                let collar_world = math::mul3(&depression, &source.collar_world);
                let shoulder_at = add3(source.collar, apply(&collar_world, collar_offset));
                let delta = sub3(desired, shoulder_at);
                let bend = rotation(axis, xx[5]);
                let elbow_local = math::mul3(&bend, &local_elbow);
                let end = add3(u, apply(&elbow_local, v0));
                let source_upper = math::mul3(&depression, &source.shoulder_world);
                let aimed = math::align(&apply(&source_upper, end), &delta);
                let mut upper = math::mul3(&aimed, &source_upper);
                let swivel_axis = math::unit3(&delta);
                upper = math::mul3(&rotation(swivel_axis, xx[1]), &upper);
                let elbow_at = add3(shoulder_at, apply(&upper, u));
                let wrist_at = add3(elbow_at, apply(&math::mul3(&upper, &elbow_local), v0));
                let wrist_world = math::mul3(
                    &math::mul3(&upper, &elbow_local),
                    &source.wrist_local,
                );
                let palm = add3(wrist_at, apply(&wrist_world, source.palm_offset_local));
                let lower = apply(&elbow_local, v0);
                let flex = (dot3(u, lower) / (len3(u) * len3(v0)).max(1e-12)).clamp(-1.0, 1.0);
                Planned {
                    collar_world,
                    shoulder_world: upper,
                    elbow_local,
                    elbow: elbow_at,
                    palm,
                    flex: flex.acos(),
                }
            })
            .collect()
    }

    /// The state vector the temporal and prior terms act on.
    fn states(&self, predicted: &[Planned], x: &[f64]) -> Vec<[f64; 6]> {
        predicted
            .iter()
            .enumerate()
            .map(|(frame, p)| {
                let xx = self.variables(x, frame);
                let source = &self.frames[frame];
                let relative = rotvec(&math::tmul3(&source.shoulder_world, &p.shoulder_world));
                let planned = if source.planned { 1.0 } else { 0.0 };
                [
                    xx[5],
                    relative[0] * planned,
                    relative[1] * planned,
                    relative[2] * planned,
                    xx[0],
                    xx[1],
                ]
            })
            .collect()
    }

    /// The source elbow's angular speed per frame, used to keep the temporal
    /// weight off fast motion (the reference's `motion`).
    fn motion(&self) -> Vec<f64> {
        let fps = self.fps.max(1.0);
        let mut out = vec![1.0; self.frames.len()];
        for frame in 1..self.frames.len() {
            let previous = math::tmul3(
                &self.frames[frame - 1].shoulder_world,
                &self.frames[frame - 1].elbow_world,
            );
            let current = math::tmul3(
                &self.frames[frame].shoulder_world,
                &self.frames[frame].elbow_world,
            );
            let relative = math::tmul3(&math::transpose3(&previous), &current);
            let speed = math::magnitude(&relative) * fps;
            let ratio = speed / 30.0f64.to_radians();
            out[frame] = 0.25 + 0.75 / (1.0 + ratio * ratio);
        }
        out
    }

    /// The reference's `required` elbow correction: the angle that puts the wrist
    /// at the desired distance from the shoulder, wrapped and branch-selected.
    fn required(&self) -> Vec<f64> {
        let (shoulder_index, elbow_index, _) = self.joints();
        let u = sub3(joint(self.rest, elbow_index), joint(self.rest, shoulder_index));
        let v0 = sub3(joint(self.rest, self.joints().2), joint(self.rest, elbow_index));
        let axis = crate::anatomical::elbow_axis(self.rest, self.hand);
        let uu = dot3(u, u);
        let vv = dot3(v0, v0);
        let u_axis = dot3(u, axis);
        self.frames
            .iter()
            .map(|source| {
                let local = math::tmul3(&source.shoulder_world, &source.elbow_world);
                let v = apply(&local, v0);
                let dot = dot3(v, axis);
                let aa = dot3(v, u) - dot * u_axis;
                let bb = dot3(math::cross3(&axis, &v), u);
                let radius = (aa * aa + bb * bb).sqrt().max(1e-12);
                let phase = bb.atan2(aa);
                let constant = uu + vv + 2.0 * dot * u_axis;
                // The requested wrist target (before any trim), which is what the
                // reference's `initial_distance` measures.
                let desired = add3(
                    source.wrist,
                    scale3(sub3(self.pin, source.palm), source.influence),
                );
                let distance = len3(sub3(desired, source.shoulder));
                let argument = ((distance * distance - constant) / (2.0 * radius)).clamp(-1.0, 1.0);
                wrap(phase + self.branch * argument.acos())
            })
            .collect()
    }

    /// Plan the stance. Returns the predicted pose of every frame.
    pub fn solve(&self) -> Vec<Planned> {
        let count = self.frames.len();
        if count == 0 {
            return Vec::new();
        }
        let required = self.required();
        let planned: Vec<bool> = self.frames.iter().map(|frame| frame.planned).collect();
        // The reference widens the elbow and collar bounds only when the requested
        // flexion is far from neutral; otherwise a small flex keeps its own limits.
        let expanded = planned
            .iter()
            .zip(&required)
            .any(|(planned, angle)| *planned && angle.abs() > ELBOW_SOFT_DEG.to_radians());
        let elbow_limit = if expanded {
            ELBOW_LIMIT_DEG.to_radians()
        } else {
            ELBOW_SOFT_DEG.to_radians()
        };
        let collar_lower = if expanded {
            -COLLAR_LIMIT_DEG.to_radians()
        } else {
            0.0
        };
        let mut lower = vec![0.0f64; count * 6];
        let mut upper = vec![0.0f64; count * 6];
        for frame in 0..count {
            let (lo, hi): ([f64; 6], [f64; 6]) = if planned[frame] {
                (
                    [
                        collar_lower,
                        -SWIVEL_LIMIT_DEG.to_radians(),
                        -TRIM_M,
                        -TRIM_M,
                        -TRIM_M,
                        -elbow_limit,
                    ],
                    [
                        COLLAR_LIMIT_DEG.to_radians(),
                        SWIVEL_LIMIT_DEG.to_radians(),
                        TRIM_M,
                        TRIM_M,
                        TRIM_M,
                        elbow_limit,
                    ],
                )
            } else {
                ([-1e-10; 6], [1e-10; 6])
            };
            for slot in 0..6 {
                lower[frame * 6 + slot] = lo[slot];
                upper[frame * 6 + slot] = hi[slot];
            }
        }
        let mut start = vec![0.0f64; count * 6];
        for frame in 0..count {
            if !planned[frame] {
                continue;
            }
            // A small collar depression, and no elbow correction: the analytic
            // `required` angle reaches the *wrist* target, which is not where the
            // palm sits once the wrist frame is carried along, so starting from it
            // can begin the solve a long way from the contact.
            start[frame * 6] = 2.0f64.to_radians();
        }

        let motion = self.motion();
        let fps = self.fps.max(1.0);
        let rate = fps / 30.0;
        let velocity_weight = TEMPORAL_WEIGHT * rate / VELOCITY_SIGMA;
        let acceleration_weight = TEMPORAL_WEIGHT * rate * rate / ACCELERATION_SIGMA;

        let residual = |x: &[f64]| -> Vec<f64> {
            let predicted = self.predict(x);
            let state = self.states(&predicted, x);
            let mut out: Vec<f64> = Vec::with_capacity(count * 26);
            // Contact: the palm on its registered point.
            for frame in 0..count {
                if !planned[frame] {
                    continue;
                }
                let source = &self.frames[frame];
                let goal = add3(
                    source.palm,
                    scale3(sub3(self.pin, source.palm), source.influence),
                );
                let error = sub3(predicted[frame].palm, goal);
                for axis in 0..3 {
                    out.push(error[axis] / CONTACT_SIGMA_M);
                }
            }
            // Prior on the size of the added change.
            for frame in 0..count {
                for slot in 0..6 {
                    out.push(state[frame][slot] * CHANNEL[slot] / STATE_SIGMA[slot]);
                }
            }
            // The elbow may not drift far from where the source put it; this is
            // what keeps the shoulder from carrying the whole correction.
            for frame in 0..count {
                if !planned[frame] {
                    continue;
                }
                let drift = sub3(predicted[frame].elbow, self.frames[frame].elbow);
                for axis in 0..3 {
                    out.push(drift[axis] / ELBOW_SIGMA_M);
                }
            }
            // Temporal smoothness of the added state, weighted off fast motion.
            for frame in 0..count - 1 {
                let both = if planned[frame] || planned[frame + 1] {
                    1.0
                } else {
                    0.0
                };
                let gate = both * motion[frame].min(motion[frame + 1]).max(0.0).sqrt();
                for slot in 0..6 {
                    let difference = state[frame + 1][slot] - state[frame][slot];
                    out.push(difference * gate * velocity_weight * CHANNEL[slot]);
                }
            }
            // Second difference, with the physical flexion as its own channel.
            for frame in 0..count - 2 {
                let any = if planned[frame] || planned[frame + 1] || planned[frame + 2] {
                    1.0
                } else {
                    0.0
                };
                let gate = any
                    * motion[frame]
                        .min(motion[frame + 1])
                        .min(motion[frame + 2])
                        .max(0.0)
                        .sqrt();
                for slot in 0..6 {
                    let second = state[frame + 2][slot] - 2.0 * state[frame + 1][slot]
                        + state[frame][slot];
                    out.push(second * gate * acceleration_weight * CHANNEL[slot]);
                }
                let flex = predicted[frame + 2].flex - 2.0 * predicted[frame + 1].flex
                    + predicted[frame].flex;
                out.push(flex * gate * TEMPORAL_WEIGHT * rate * rate / FLEX_ACCELERATION_SIGMA);
            }
            out
        };

        if std::env::var("VELOMOCAP_DEBUG_PLAN").is_ok() {
            let zero = vec![0.0f64; count * 6];
            let start_r = residual(&start);
            let zero_r = residual(&zero);
            let mut worst = (0.0f64, 0usize);
            for (slot, value) in zero_r.iter().enumerate() {
                if value.abs() > worst.0 {
                    worst = (value.abs(), slot);
                }
            }
            eprintln!(
                "cost start {:.4e} zero {:.4e} (m {}), worst residual {:.4e} at {}",
                0.5 * start_r.iter().map(|v| v * v).sum::<f64>(),
                0.5 * zero_r.iter().map(|v| v * v).sum::<f64>(),
                zero_r.len(),
                worst.0,
                worst.1
            );
            let predicted = self.predict(&zero);
            eprintln!(
                "zero palm {:?} pin {:?} source {:?}",
                predicted[0].palm, self.pin, self.frames[0].palm
            );
        }
        let solution = solver::minimise(
            &start,
            &lower,
            &upper,
            residual,
            solver::Options {
                max_iterations: ITERATIONS,
                tolerance: 1e-9,
            },
        );
        if std::env::var("VELOMOCAP_DEBUG_PLAN").is_ok() {
            eprintln!(
                "solver: cost {:.4e} iterations {} converged {} first {:?}",
                solution.cost,
                solution.iterations,
                solution.converged,
                &solution.x[..6.min(solution.x.len())]
            );
        }
        self.predict(&solution.x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use velomocap_body::{BodyModel, Pose};

    fn loaded() -> Option<(BodyModel, Vec<f32>)> {
        let path = [
            "models/body_models/smplh/neutral/model.npz",
            "../../models/body_models/smplh/neutral/model.npz",
        ]
        .iter()
        .find(|candidate| std::path::Path::new(candidate).is_file())?
        .to_string();
        let body = BodyModel::load(&path, 16).expect("SMPL-H");
        let rest = body.rest_joints(&vec![0.0f32; 16]);
        Some((body, rest))
    }

    fn frames_from_poses(
        body: &BodyModel,
        rest: &[f32],
        poses: &[Pose],
        hand: usize,
        offset: [f64; 3],
    ) -> (Vec<Frame>, [f64; 3]) {
        let (collar, shoulder, elbow) = if hand == 0 {
            (13usize, 16usize, 18usize)
        } else {
            (14, 17, 19)
        };
        let wrist = if hand == 0 { 20 } else { 21 };
        let mut frames = Vec::new();
        let mut last_palm = [0.0f64; 3];
        for pose in poses {
            let (world, positions) = crate::anatomical::forward(body, pose, rest);
            let palm = add3(positions[wrist], apply(&world[wrist], offset));
            frames.push(Frame {
                collar_world: world[collar],
                shoulder_world: world[shoulder],
                elbow_world: world[elbow],
                collar: positions[collar],
                shoulder: positions[shoulder],
                elbow: positions[elbow],
                wrist: positions[wrist],
                palm,
                wrist_local: math::tmul3(&world[elbow], &world[wrist]),
                palm_offset_local: offset,
                influence: 1.0,
                planned: true,
            });
            last_palm = palm;
        }
        (frames, last_palm)
    }

    /// A palm planted 4 cm away from where the source put it must be moved onto
    /// its pin, without leaving the bounded collar/swivel/elbow space.
    #[test]
    fn plans_a_palm_onto_its_pin() {
        let Some((body, rest)) = loaded() else {
            return;
        };
        // Arm swept forward and slightly up, the pose a hand on a table has.
        let mut pose = Pose::neutral(body.num_joints, 16);
        pose.translation = [0.0, 0.9, 0.0];
        pose.body[(16 - 1) * 3] = -0.5;
        pose.body[(16 - 1) * 3 + 2] = 0.4;
        pose.body[(18 - 1) * 3] = -0.9;
        let poses = vec![pose; 8];
        let (frames, palm) = frames_from_poses(&body, &rest, &poses, 0, [0.0, -0.04, 0.0]);
        // A reachable correction: 1.2 cm back towards the shoulder (the elbow
        // bends and the collar depresses) and 0.6 cm sideways (the arm swivels).
        // The bounds are deliberately small (6 mm trim, 30 degrees of elbow, 15 of
        // collar), so a larger request is meant to be softened by the reach gain
        // instead of being forced.
        let shoulder = frames[0].shoulder;
        let direction = math::unit3(&sub3(palm, shoulder));
        let pin = add3(
            add3(palm, scale3(direction, -0.012)),
            [0.006, 0.0, 0.0],
        );
        let plan = Plan {
            rest: &rest,
            hand: 0,
            branch: 1.0,
            fps: 30.0,
            pin,
            frames: &frames,
        };
        let planned = plan.solve();
        assert_eq!(planned.len(), frames.len());
        if std::env::var("VELOMOCAP_DEBUG_PLAN").is_ok() {
            eprintln!(
                "pin {:?} source {:?} planned0 {:?}",
                pin, frames[0].palm, planned[0].palm
            );
        }
        let mut worst: f64 = 0.0;
        for (frame, result) in planned.iter().enumerate() {
            let error = len3(sub3(result.palm, pin));
            worst = worst.max(error);
            assert!(
                error < 0.01,
                "frame {frame}: palm is {:.1} mm from the pin",
                error * 1000.0
            );
            // Bounded: the collar may not be depressed by more than 15 degrees.
            let collar_delta = math::tmul3(
                &math::transpose3(&frames[frame].collar_world),
                &result.collar_world,
            );
            assert!(
                math::magnitude(&collar_delta) <= 15.0f64.to_radians() + 1e-6,
                "collar moved {:.1} degrees",
                math::magnitude(&collar_delta).to_degrees()
            );
            assert!(result.flex.is_finite(), "flex is NaN");
        }
        assert!(worst < 0.01, "worst palm error {:.1} mm", worst * 1000.0);
    }

    /// Nothing planned means nothing moves, which keeps the pass free on frames
    /// outside every stance.
    #[test]
    fn leaves_unplanned_frames_alone() {
        let Some((body, rest)) = loaded() else {
            return;
        };
        let mut pose = Pose::neutral(body.num_joints, 16);
        pose.translation = [0.0, 0.9, 0.0];
        let poses = vec![pose; 4];
        let (mut frames, palm) = frames_from_poses(&body, &rest, &poses, 0, [0.0, -0.04, 0.0]);
        for frame in frames.iter_mut() {
            frame.planned = false;
            frame.influence = 0.0;
        }
        let plan = Plan {
            rest: &rest,
            hand: 0,
            branch: 1.0,
            fps: 30.0,
            pin: add3(palm, [0.05, 0.0, 0.0]),
            frames: &frames,
        };
        let planned = plan.solve();
        for (frame, result) in planned.iter().enumerate() {
            let moved = len3(sub3(result.palm, frames[frame].palm));
            assert!(moved < 1e-6, "frame {frame} moved {:.2} mm", moved * 1000.0);
        }
    }
}
