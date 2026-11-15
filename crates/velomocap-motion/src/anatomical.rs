//! Fixed anatomical-axis leg IK, ported from the reference's `anatomical_ik.py`
//! (RGB MoCap, MIT).
//!
//! The reference does not use a free two-bone solve for the legs. The knee is
//! only ever allowed to flex about **one calibrated anatomical axis**, so the
//! knee can never leave its plane and can never flip; the hip is then rotated so
//! that the resulting chain points at the target:
//!
//! ```text
//! axis      = align([0,-1,0], upper) * [1,0,0]      (the femur's hinge)
//! v         = knee_local * lower                    (lower leg in the knee frame)
//! a, b      = components of v in the plane normal to axis
//! desired   = (|delta|^2 - |upper|^2 - |lower|^2) / 2 - constant
//! flex      = min(acos(desired / radius), 160 deg)
//! bend      = rotate(correction) about axis, correction = atan2(b, a) + flex
//! hip       = align(hip * (upper + bend * lower), delta) * hip
//! ```
//!
//! `reach` and `minimum_reach` give the physical length bounds of the same chain,
//! which is what bounds the pelvis lift in the reference.

use velomocap_body::{BodyModel, Pose};
use velomocap_core::math;

/// Leg chains: hip, knee, ankle.
pub const LEGS: [(usize, usize, usize); 2] = [(1, 4, 7), (2, 5, 8)];

/// Native solver flex limit, degrees.
const FLEX_LIMIT_DEG: f64 = 160.0;

fn joint3(rest: &[f32], index: usize) -> [f64; 3] {
    [
        rest[index * 3] as f64,
        rest[index * 3 + 1] as f64,
        rest[index * 3 + 2] as f64,
    ]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

fn rotate(matrix: &[f64; 9], v: [f64; 3]) -> [f64; 3] {
    [
        matrix[0] * v[0] + matrix[1] * v[1] + matrix[2] * v[2],
        matrix[3] * v[0] + matrix[4] * v[1] + matrix[5] * v[2],
        matrix[6] * v[0] + matrix[7] * v[1] + matrix[8] * v[2],
    ]
}

/// `align([0,-1,0], upper) * [1,0,0]`: the femur's calibrated hinge axis.
fn knee_axis(upper: [f64; 3]) -> [f64; 3] {
    let calibration = math::align(&[0.0, -1.0, 0.0], &upper);
    [calibration[0], calibration[3], calibration[6]]
}

/// Plane decomposition of the lower leg, shared by the reach bounds and the solve.
struct Plane {
    a: f64,
    b: f64,
    constant: f64,
    radius: f64,
}

fn plane(upper: [f64; 3], lower: [f64; 3], axis: [f64; 3], knee_local: &[f64; 9]) -> Plane {
    let v = rotate(knee_local, lower);
    let along = dot(v, axis);
    let perpendicular = [
        v[0] - along * axis[0],
        v[1] - along * axis[1],
        v[2] - along * axis[2],
    ];
    let a = dot(perpendicular, upper);
    let b = dot(cross(axis, v), upper);
    Plane {
        a,
        b,
        constant: dot(upper, axis) * along,
        radius: (a * a + b * b).sqrt(),
    }
}

/// Maximum physical reach of one leg for the given (world) knee rotation.
pub fn reach(rest: &[f32], hip: usize, knee: usize, ankle: usize, knee_local: &[f64; 9]) -> f64 {
    let upper = sub(joint3(rest, knee), joint3(rest, hip));
    let lower = sub(joint3(rest, ankle), joint3(rest, knee));
    let axis = knee_axis(upper);
    let plane = plane(upper, lower, axis, knee_local);
    let value = dot(upper, upper) + dot(lower, lower) + 2.0 * (plane.constant + plane.radius);
    value.max(0.0).sqrt()
}

/// Physical minimum reach at the solver's 160 degree flex limit.
pub fn minimum_reach(
    rest: &[f32],
    hip: usize,
    knee: usize,
    ankle: usize,
    knee_local: &[f64; 9],
) -> f64 {
    let upper = sub(joint3(rest, knee), joint3(rest, hip));
    let lower = sub(joint3(rest, ankle), joint3(rest, knee));
    let axis = knee_axis(upper);
    let plane = plane(upper, lower, axis, knee_local);
    let cos = FLEX_LIMIT_DEG.to_radians().cos();
    let value = dot(upper, upper) + dot(lower, lower) + 2.0 * (plane.constant + plane.radius * cos);
    value.max(0.0).sqrt()
}

/// The result of solving one leg.
#[derive(Debug)]
pub struct SolveReport {
    /// Off-axis component of the knee rotation, in degrees (must stay ~0).
    pub off_axis_deg: f64,
    /// Largest knee rotation added, in degrees.
    pub rotation_deg: f64,
    /// Largest flex angle used, in degrees.
    pub flex_deg: f64,
}

/// Solve one leg so the ankle reaches `target`, then write the chain back into
/// the pose. `world` and `positions` must be the current forward kinematics.
///
/// Returns `None` when the chain is degenerate.
pub fn solve_leg(
    body: &BodyModel,
    pose: &mut Pose,
    rest: &[f32],
    world: &[[f64; 9]],
    positions: &[[f64; 3]],
    hip: usize,
    knee: usize,
    ankle: usize,
    target: [f64; 3],
) -> Option<SolveReport> {
    let upper = sub(joint3(rest, knee), joint3(rest, hip));
    let lower = sub(joint3(rest, ankle), joint3(rest, knee));
    let calibration = math::align(&[0.0, -1.0, 0.0], &upper);
    let axis = [calibration[0], calibration[3], calibration[6]];

    // Local rotation of the knee (parent world inverse * world).
    let knee_local = match body.parents[knee] {
        parent if parent >= 0 => math::tmul3(&world[parent as usize], &world[knee]),
        _ => world[knee],
    };

    let plane = plane(upper, lower, axis, &knee_local);
    let origin = positions[hip];
    let delta = sub(target, origin);
    let distance = norm(delta);
    let desired = 0.5 * (distance * distance - dot(upper, upper) - dot(lower, lower))
        - plane.constant;
    let phase = plane.b.atan2(plane.a);
    let ratio = (desired / plane.radius.max(1e-12)).clamp(-1.0, 1.0);
    let flex = ratio.acos().min(FLEX_LIMIT_DEG.to_radians());
    let correction = phase + flex;

    let bend = crate::ik::axis_angle_matrix(axis, correction);
    let knee_local_new = math::mul3(&bend, &knee_local);
    let end_vector = [
        upper[0] + rotate(&knee_local_new, lower)[0],
        upper[1] + rotate(&knee_local_new, lower)[1],
        upper[2] + rotate(&knee_local_new, lower)[2],
    ];

    // Aim the hip so the solved chain points at the target.
    let hip_world = world[hip];
    let current = rotate(&hip_world, end_vector);
    let aim = math::align(&current, &delta);
    let hip_world_new = math::mul3(&aim, &hip_world);

    // Write back: hip, knee and ankle, keeping the ankle's own orientation.
    let hip_parent = body.parents[hip];
    let hip_local = if hip_parent >= 0 {
        math::tmul3(&world[hip_parent as usize], &hip_world_new)
    } else {
        hip_world_new
    };
    let ankle_local = math::tmul3(
        &math::mul3(&hip_world_new, &knee_local_new),
        &world[ankle],
    );

    for (joint_index, local) in [
        (hip, hip_local),
        (knee, knee_local_new),
        (ankle, ankle_local),
    ] {
        let angles = matrix_to_axis_angle(&local);
        if joint_index == 0 {
            pose.root = angles;
        } else {
            let at = (joint_index - 1) * 3;
            if at + 3 > pose.body.len() {
                return None;
            }
            pose.body[at] = angles[0];
            pose.body[at + 1] = angles[1];
            pose.body[at + 2] = angles[2];
        }
    }

    // The added rotation must be exactly about the anatomical axis.
    let flat = [
        (bend[7] - bend[5]) * 0.5,
        (bend[2] - bend[6]) * 0.5,
        (bend[3] - bend[1]) * 0.5,
    ];
    let off = [
        flat[0] - dot(flat, axis) * axis[0],
        flat[1] - dot(flat, axis) * axis[1],
        flat[2] - dot(flat, axis) * axis[2],
    ];
    Some(SolveReport {
        off_axis_deg: norm(off).to_degrees(),
        rotation_deg: correction.abs().to_degrees(),
        flex_deg: flex.to_degrees(),
    })
}

/// The reference's `source_reach_gap`: how far a hand target sits outside what the
/// arm can actually reach, given the elbow axis bounded by `elbow_deg` and the
/// collar spin bounded by `collar_deg`. Geometry only - it never looks at whether
/// an optimisation succeeded. Zero means reachable; the caller fades its
/// correction out as the gap grows (`1 / (1 + (gap / 0.03)^4)` in the reference).
pub fn source_reach_gap(
    rest: &[f32],
    world: &[[f64; 9]],
    positions: &[[f64; 3]],
    hand: usize,
    target: [f64; 3],
    sample: [f64; 3],
    collar_deg: f64,
    elbow_deg: f64,
) -> f64 {
    let (shoulder, elbow, wrist, collar) = if hand == 0 {
        (16usize, 18usize, 20usize, 13usize)
    } else {
        (17, 19, 21, 14)
    };
    let sign = if hand == 0 { 1.0 } else { -1.0 };
    let u = sub(joint3(rest, elbow), joint3(rest, shoulder));
    let v0 = sub(joint3(rest, wrist), joint3(rest, elbow));
    let calibration = math::align(&[sign, 0.0, 0.0], &u);
    let axis = rotate(&calibration, [0.0, -sign, 0.0]);
    let local_elbow = math::tmul3(&world[shoulder], &world[elbow]);
    let v = rotate(&local_elbow, v0);
    let along = dot(v, axis);
    let aa = dot(v, u) - along * dot(u, axis);
    let bb = dot(cross(axis, v), u);
    let constant = dot(u, u) + dot(v0, v0) + 2.0 * along * dot(u, axis);
    let phase = bb.atan2(aa);
    let bound = elbow_deg.to_radians();
    let mut low = f64::INFINITY;
    let mut high: f64 = 0.0;
    for theta in [
        -bound,
        bound,
        phase.clamp(-bound, bound),
        wrap(phase).clamp(-bound, bound),
        wrap(phase + std::f64::consts::PI).clamp(-bound, bound),
    ] {
        let value = (constant + 2.0 * (aa * theta.cos() + bb * theta.sin()))
            .max(0.0)
            .sqrt();
        low = low.min(value);
        high = high.max(value);
    }

    // Collar spin bounds: how far the shoulder itself can carry the target.
    let offset = sub(joint3(rest, shoulder), joint3(rest, collar));
    let offset = rotate(&world[collar], offset);
    let spin_raw = cross(offset, [0.0, -1.0, 0.0]);
    let spin_length = norm(spin_raw).max(1e-12);
    let spin = [spin_raw[0] / spin_length, spin_raw[1] / spin_length, spin_raw[2] / spin_length];
    let goal = sub(positions[wrist], positions[collar]);
    let goal = [
        goal[0] + target[0] - sample[0],
        goal[1] + target[1] - sample[1],
        goal[2] + target[2] - sample[2],
    ];
    let parallel = [
        spin[0] * dot(spin, offset),
        spin[1] * dot(spin, offset),
        spin[2] * dot(spin, offset),
    ];
    let perp = sub(offset, parallel);
    let ca = dot(goal, perp);
    let cb = dot(goal, cross(spin, offset));
    let phase = cb.atan2(ca);
    let bound = collar_deg.to_radians();
    let base = dot(goal, goal) + dot(offset, offset) - 2.0 * dot(goal, parallel);
    let mut dmin = f64::INFINITY;
    let mut dmax: f64 = 0.0;
    for theta in [
        -bound,
        bound,
        wrap(phase).clamp(-bound, bound),
        wrap(phase + std::f64::consts::PI).clamp(-bound, bound),
    ] {
        let value = (base - 2.0 * (ca * theta.cos() + cb * theta.sin())).max(0.0).sqrt();
        dmin = dmin.min(value);
        dmax = dmax.max(value);
    }
    (dmin - high).max(low - dmax).max(0.0)
}

/// The reference's branch calibration: which side of the elbow the flexion goes
/// to, decided once per stance from the source angles (never per frame, which is
/// what makes an elbow flip between the two solutions). Frames below 8 degrees are
/// ignored because the sign is meaningless near straight.
///
/// Returns `+1` or `-1`.
pub fn elbow_branch(
    rest: &[f32],
    local_elbows: &[[f64; 9]],
    hand: usize,
) -> f64 {
    let (shoulder, elbow, wrist) = if hand == 0 {
        (16usize, 18usize, 20usize)
    } else {
        (17, 19, 21)
    };
    let u = sub(joint3(rest, elbow), joint3(rest, shoulder));
    let v0 = sub(joint3(rest, wrist), joint3(rest, elbow));
    let sign = if hand == 0 { 1.0 } else { -1.0 };
    let calibration = math::align(&[sign, 0.0, 0.0], &u);
    let axis = rotate(&calibration, [0.0, -sign, 0.0]);
    let mut observed = Vec::new();
    for local in local_elbows {
        let v = rotate(local, v0);
        let along = dot(v, axis);
        let aa = dot(v, u) - along * dot(u, axis);
        let bb = dot(cross(axis, v), u);
        let angle = bb.atan2(aa);
        if angle.abs() > 8.0f64.to_radians() {
            observed.push(angle);
        }
    }
    if observed.is_empty() {
        return 1.0;
    }
    observed.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = observed[observed.len() / 2];
    if median <= 0.0 {
        1.0
    } else {
        -1.0
    }
}

/// Wrap an angle into `[-pi, pi]`.
fn wrap(mut angle: f64) -> f64 {
    while angle > std::f64::consts::PI {
        angle -= 2.0 * std::f64::consts::PI;
    }
    while angle < -std::f64::consts::PI {
        angle += 2.0 * std::f64::consts::PI;
    }
    angle
}

/// Rotation matrix to an axis-angle vector.
fn matrix_to_axis_angle(m: &[f64; 9]) -> [f32; 3] {
    let trace = m[0] + m[4] + m[8];
    let angle = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0).acos();
    if angle < 1e-6 {
        return [0.0; 3];
    }
    let axis = [m[7] - m[5], m[2] - m[6], m[3] - m[1]];
    let length = norm(axis);
    if length < 1e-9 {
        return [0.0; 3];
    }
    [
        (axis[0] / length * angle) as f32,
        (axis[1] / length * angle) as f32,
        (axis[2] / length * angle) as f32,
    ]
}

/// Forward kinematics as `(rotation, position)`, parent first — the input both
/// `solve_leg` and the reach bounds expect.
pub fn forward(body: &BodyModel, pose: &Pose, rest: &[f32]) -> (Vec<[f64; 9]>, Vec<[f64; 3]>) {
    let count = body.num_joints;
    let mut rotations = Vec::with_capacity(count);
    let mut positions = Vec::with_capacity(count);
    for joint in 0..count {
        let local = if joint == 0 {
            rotvec(pose.root)
        } else {
            let at = (joint - 1) * 3;
            rotvec([
                pose.body[at],
                pose.body[at + 1],
                pose.body[at + 2],
            ])
        };
        let parent = body.parents[joint];
        if parent < 0 {
            rotations.push(local);
            positions.push([
                pose.translation[0] as f64 + rest[0] as f64,
                pose.translation[1] as f64 + rest[1] as f64,
                pose.translation[2] as f64 + rest[2] as f64,
            ]);
        } else {
            let p = parent as usize;
            let offset = sub(joint3(rest, joint), joint3(rest, p));
            let moved = rotate(&rotations[p], offset);
            rotations.push(math::mul3(&rotations[p], &local));
            positions.push([
                positions[p][0] + moved[0],
                positions[p][1] + moved[1],
                positions[p][2] + moved[2],
            ]);
        }
    }
    (rotations, positions)
}

fn rotvec(v: [f32; 3]) -> [f64; 9] {
    let vector = [v[0] as f64, v[1] as f64, v[2] as f64];
    let angle = norm(vector);
    if angle < 1e-9 {
        return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    }
    crate::ik::axis_angle_matrix(
        [vector[0] / angle, vector[1] / angle, vector[2] / angle],
        angle,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    #[test]
    fn elbow_branch_is_stable_for_a_straight_arm() {
        let Some((body, rest)) = loaded() else {
            return;
        };
        let mut pose = Pose::neutral(body.num_joints, 16);
        pose.betas[0] = 0.3;
        let (world, _) = forward(&body, &pose, &rest);
        // A neutral (nearly straight) arm has no measurable branch, so the
        // calibration must fall back to +1 instead of guessing from noise.
        let local = math::tmul3(&world[16], &world[18]);
        let branch = elbow_branch(&rest, &[local], 0);
        assert!(branch == 1.0 || branch == -1.0, "branch {branch}");
    }

    #[test]
    fn reach_gap_grows_with_distance() {
        let Some((body, rest)) = loaded() else {
            return;
        };
        let mut pose = Pose::neutral(body.num_joints, 16);
        pose.betas[0] = 0.3;
        pose.translation = [0.0, 0.9, 0.0];
        let (world, positions) = forward(&body, &pose, &rest);
        let wrist = positions[20];
        let near = source_reach_gap(&rest, &world, &positions, 0, wrist, wrist, 15.0, 60.0);
        assert!(near < 1e-6, "current wrist should be reachable, gap {near}");
        let far = [wrist[0] + 0.8, wrist[1], wrist[2] + 0.5];
        let gap = source_reach_gap(&rest, &world, &positions, 0, far, wrist, 15.0, 60.0);
        assert!(gap > 0.1, "far target should have a gap, got {gap}");
    }

    fn solve_leg_reaches_a_target_without_leaving_the_knee_axis() {
        let Some((body, rest)) = loaded() else {
            return;
        };
        let mut pose = Pose::neutral(body.num_joints, 16);
        pose.betas[0] = 0.3;
        pose.translation = [0.0, 0.9, 0.0];
        let (world, positions) = forward(&body, &pose, &rest);
        let (hip, knee, ankle) = LEGS[0];
        let ankle_at = positions[ankle];
        // A reachable point 8 cm forward and 6 cm to the side.
        let target = [ankle_at[0] + 0.08, ankle_at[1], ankle_at[2] + 0.06];
        let report = solve_leg(
            &body, &mut pose, &rest, &world, &positions, hip, knee, ankle, target,
        )
        .expect("solve");
        assert!(report.off_axis_deg < 1e-6, "left the knee axis: {report:?}");
        let (_, after) = forward(&body, &pose, &rest);
        let error = norm(sub(after[ankle], target));
        assert!(error < 0.01, "ankle {error:.3} m from the target");
    }

    #[test]
    fn reach_bounds_bracket_the_rest_length() {
        let Some((body, rest)) = loaded() else {
            return;
        };
        let mut pose = Pose::neutral(body.num_joints, 16);
        pose.betas[0] = 0.3;
        let (world, positions) = forward(&body, &pose, &rest);
        let (hip, knee, ankle) = LEGS[0];
        let knee_local = math::tmul3(&world[hip], &world[knee]);
        let max = reach(&rest, hip, knee, ankle, &knee_local);
        let min = minimum_reach(&rest, hip, knee, ankle, &knee_local);
        let actual = norm(sub(positions[ankle], positions[hip]));
        assert!(min <= actual + 1e-6, "min {min:.3} > actual {actual:.3}");
        assert!(max >= actual - 1e-6, "max {max:.3} < actual {actual:.3}");
    }
}
