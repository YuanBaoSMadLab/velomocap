//! Two-bone / one-bone IK, ported from NVIDIA Kimodo's `MotionCorrection`.
//!
//! Source: `nv-tlabs/kimodo`, `MotionCorrection/src/cpp/AnimProcessing/
//! InverseKinematics.cpp`, Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES,
//! Apache License 2.0. The algorithm and the stage split are kept verbatim:
//!
//! 1. bend the middle joint so that `|ac| = |at|` while the chain keeps its
//!    direction,
//! 2. rotate the start joint so the end joint lands on the target.
//!
//! Our own contact pass used to do a small CCD loop; this is the exact solver
//! the reference app runs, so the corrected pose matches its behaviour.

use velomocap_core::math;

/// Rigid transform: row-major rotation, then translation.
#[derive(Clone, Copy, Debug)]
pub struct Transform {
    pub rotation: [f64; 9],
    pub translation: [f64; 3],
}

impl Transform {
    pub const IDENTITY: Self = Self {
        rotation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        translation: [0.0, 0.0, 0.0],
    };

    pub fn from_rotation(rotation: [f64; 9]) -> Self {
        Self {
            rotation,
            translation: [0.0; 3],
        }
    }

    pub fn apply(&self, point: [f64; 3]) -> [f64; 3] {
        let r = &self.rotation;
        [
            r[0] * point[0] + r[1] * point[1] + r[2] * point[2] + self.translation[0],
            r[3] * point[0] + r[4] * point[1] + r[5] * point[2] + self.translation[1],
            r[6] * point[0] + r[7] * point[1] + r[8] * point[2] + self.translation[2],
        ]
    }

    /// Inverse-rotate a vector by this transform's rotation only.
    pub fn rotate_inverse(&self, vector: [f64; 3]) -> [f64; 3] {
        let r = &self.rotation;
        [
            r[0] * vector[0] + r[3] * vector[1] + r[6] * vector[2],
            r[1] * vector[0] + r[4] * vector[1] + r[7] * vector[2],
            r[2] * vector[0] + r[5] * vector[1] + r[8] * vector[2],
        ]
    }
}

/// `a * b` with Kimodo's convention (`Transform::operator*=`): the rotation is
/// `Ra * Rb` and the translation is `Rb * ta + tb`, i.e. apply `a` first and
/// then `b`. Getting this backwards silently mirrors the whole chain, which is
/// why the ported tests reach the wrong point if it is "fixed" to the usual
/// parent-first order.
pub fn concat(a: &Transform, b: &Transform) -> Transform {
    let rotation = math::mul3(&a.rotation, &b.rotation);
    let moved = [
        b.rotation[0] * a.translation[0] + b.rotation[1] * a.translation[1] + b.rotation[2] * a.translation[2],
        b.rotation[3] * a.translation[0] + b.rotation[4] * a.translation[1] + b.rotation[5] * a.translation[2],
        b.rotation[6] * a.translation[0] + b.rotation[7] * a.translation[1] + b.rotation[8] * a.translation[2],
    ];
    Transform {
        rotation,
        translation: [
            moved[0] + b.translation[0],
            moved[1] + b.translation[1],
            moved[2] + b.translation[2],
        ],
    }
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

fn length(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

fn normalised(a: [f64; 3]) -> [f64; 3] {
    let len = length(a);
    if len < 1e-12 {
        a
    } else {
        [a[0] / len, a[1] / len, a[2] / len]
    }
}

/// Shortest-arc rotation taking `from` to `to` (Kimodo's
/// `Quaternion::FromRotationBetweenVectors`), as a rotation matrix.
fn rotation_between(from: [f64; 3], to: [f64; 3]) -> [f64; 9] {
    let a = normalised(from);
    let b = normalised(to);
    math::align(&a, &b)
}

pub fn axis_angle_matrix(axis: [f64; 3], angle: f64) -> [f64; 9] {
    let axis = normalised(axis);
    let (s, c) = angle.sin_cos();
    let t = 1.0 - c;
    let (x, y, z) = (axis[0], axis[1], axis[2]);
    [
        t * x * x + c,
        t * x * y - s * z,
        t * x * z + s * y,
        t * x * y + s * z,
        t * y * y + c,
        t * y * z - s * x,
        t * x * z - s * y,
        t * y * z + s * x,
        t * z * z + c,
    ]
}

fn angle_between(a: [f64; 3], b: [f64; 3]) -> f64 {
    let left = normalised(a);
    let right = normalised(b);
    let cosine = dot(left, right);
    let sine = length(cross(left, right));
    sine.atan2(cosine)
}

fn cosine_rule(side_left: f64, side_right: f64, side_across: f64) -> f64 {
    let value = (side_right * side_right + side_left * side_left - side_across * side_across)
        / (2.0 * side_left * side_right);
    value.clamp(-1.0, 1.0).acos()
}

/// World transform of `index`, walking up the parent chain from `pose[index]`.
fn world_of(pose: &[Transform], parents: &[i32], root: &Transform, index: usize) -> Transform {
    concat(&pose[index], &parent_world_of(pose, parents, root, index))
}

/// Parent-world transform of `index`, exactly as Kimodo accumulates it: start
/// from the identity, fold in each ancestor with `t = t * pose[ancestor]`, then
/// multiply the root on the right.
fn parent_world_of(
    pose: &[Transform],
    parents: &[i32],
    root: &Transform,
    index: usize,
) -> Transform {
    let mut world = Transform::IDENTITY;
    let mut cursor = parents[index];
    while cursor >= 0 {
        let at = cursor as usize;
        world = concat(&world, &pose[at]);
        cursor = parents[at];
    }
    concat(&world, root)
}

/// World position of a joint in the local transform chain.
pub fn world_position(
    pose: &[Transform],
    parents: &[i32],
    root: &Transform,
    index: usize,
) -> [f64; 3] {
    world_of(pose, parents, root, index).translation
}

/// Kimodo's `IK::TwoBoneIk`. `c_idx` is the end joint (wrist / ankle).
pub fn two_bone_ik(
    pose: &mut [Transform],
    root: &Transform,
    c_idx: usize,
    weight: f64,
    target: [f64; 3],
    parents: &[i32],
    hint_offset: [f64; 3],
) {
    let weight = weight.clamp(0.0, 1.0);
    if !(weight > 0.0) {
        return;
    }
    let b_idx = parents[c_idx];
    if b_idx < 0 {
        return;
    }
    let a_idx = parents[b_idx as usize];
    if a_idx < 0 {
        return;
    }
    let (a_idx, b_idx) = (a_idx as usize, b_idx as usize);

    let a_parent = parent_world_of(pose, parents, root, a_idx);
    let a_world = concat(&pose[a_idx], &a_parent);
    let b_world = concat(&pose[b_idx], &a_world);
    let c_world = concat(&pose[c_idx], &b_world);

    let a = a_world.translation;
    let b = b_world.translation;
    let c = c_world.translation;
    let t = [
        c[0] + (target[0] - c[0]) * weight,
        c[1] + (target[1] - c[1]) * weight,
        c[2] + (target[2] - c[2]) * weight,
    ];

    let eps = 1e-4;
    let l_ab = length(sub(b, a));
    let l_bc = length(sub(c, b));
    let mut l_at = length(sub(a, t));
    l_at = l_at.clamp(eps, (l_ab + l_bc) * 0.999);

    let theta_bac_current = angle_between(sub(a, b), sub(a, c));
    let theta_abc_current = angle_between(sub(b, a), sub(b, c));
    if l_ab < eps || l_bc < eps || l_at < eps {
        return;
    }
    let theta_bac_desired = cosine_rule(l_ab, l_at, l_bc);
    let theta_abc_desired = cosine_rule(l_ab, l_bc, l_at);

    let hint = b_world.apply(hint_offset);
    let mut axis = cross(sub(c, a), sub(hint, a));
    axis = if length(axis) == 0.0 {
        [0.0, 0.0, 1.0]
    } else {
        normalised(axis)
    };

    let axis_in_b = b_world.rotate_inverse(axis);
    let rotate_b = Transform::from_rotation(axis_angle_matrix(
        axis_in_b,
        theta_abc_desired - theta_abc_current,
    ));
    pose[b_idx] = concat(&rotate_b, &pose[b_idx]);

    let axis_in_a = a_world.rotate_inverse(axis);
    let rotate_a = Transform::from_rotation(axis_angle_matrix(
        axis_in_a,
        theta_bac_desired - theta_bac_current,
    ));
    pose[a_idx] = concat(&rotate_a, &pose[a_idx]);

    let a_world = concat(&pose[a_idx], &a_parent);
    let ac_local = a_world.rotate_inverse(sub(c, a));
    let at_local = a_world.rotate_inverse(sub(target, a));
    let stage_two = Transform::from_rotation(rotation_between(ac_local, at_local));
    pose[a_idx] = concat(&stage_two, &pose[a_idx]);
}

/// Kimodo's `IK::OneBoneIk`: aim a single joint at `target`.
pub fn one_bone_ik(
    pose: &mut [Transform],
    root: &Transform,
    b_idx: usize,
    weight: f64,
    target: [f64; 3],
    parents: &[i32],
) {
    let weight = weight.clamp(0.0, 1.0);
    if !(weight > 0.0) {
        return;
    }
    let a_idx = parents[b_idx];
    if a_idx < 0 {
        return;
    }
    let a_idx = a_idx as usize;
    let a_parent = parent_world_of(pose, parents, root, a_idx);
    let a_world = concat(&pose[a_idx], &a_parent);
    let b_world = concat(&pose[b_idx], &a_world);

    let ab_local = a_world.rotate_inverse(sub(b_world.translation, a_world.translation));
    let at_local = a_world.rotate_inverse(sub(target, a_world.translation));
    let delta = rotation_between(ab_local, at_local);
    // NLerp(identity, delta, weight) needs the angle; slerp is equivalent here
    // for the small corrections this is used for.
    let blended = blend_towards_identity(&delta, weight);
    pose[a_idx].rotation = math::mul3(&blended, &pose[a_idx].rotation);
}

/// Scale a rotation towards the identity by `weight`.
fn blend_towards_identity(matrix: &[f64; 9], weight: f64) -> [f64; 9] {
    let trace = matrix[0] + matrix[4] + matrix[8];
    let angle = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0).acos();
    if angle < 1e-9 {
        return *matrix;
    }
    let axis = normalised([
        matrix[7] - matrix[5],
        matrix[2] - matrix[6],
        matrix[3] - matrix[1],
    ]);
    axis_angle_matrix(axis, angle * weight)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain() -> ([Transform; 3], [i32; 3]) {
        let pose = [
            Transform {
                rotation: Transform::IDENTITY.rotation,
                translation: [0.0, 1.0, 0.0],
            },
            Transform {
                rotation: Transform::IDENTITY.rotation,
                translation: [0.0, -0.3, 0.0],
            },
            Transform {
                rotation: Transform::IDENTITY.rotation,
                translation: [0.0, -0.3, 0.0],
            },
        ];
        (pose, [-1, 0, 1])
    }

    #[test]
    fn two_bone_ik_reaches_a_reachable_target() {
        let (mut pose, parents) = chain();
        let root = Transform::IDENTITY;
        let target = [0.2, 0.45, 0.0];
        two_bone_ik(&mut pose, &root, 2, 1.0, target, &parents, [0.0; 3]);
        let end = world_of(&pose, &parents, &root, 2).translation;
        let error = length(sub(end, target));
        assert!(error < 0.01, "end {end:?} target {target:?} error {error}");
    }

    #[test]
    fn two_bone_ik_clamps_an_unreachable_target() {
        let (mut pose, parents) = chain();
        let root = Transform::IDENTITY;
        let target = [5.0, 1.0, 0.0];
        two_bone_ik(&mut pose, &root, 2, 1.0, target, &parents, [0.0; 3]);
        let end = world_of(&pose, &parents, &root, 2).translation;
        // Fully extended towards the target, not past it.
        assert!((length(sub(end, [0.0, 1.0, 0.0])) - 0.6).abs() < 0.02, "end {end:?}");
    }

    #[test]
    fn one_bone_ik_points_at_the_target() {
        let (mut pose, parents) = chain();
        let root = Transform::IDENTITY;
        let target = [0.0, 0.7, 0.3];
        one_bone_ik(&mut pose, &root, 2, 1.0, target, &parents);
        let start = world_of(&pose, &parents, &root, 0).translation;
        let end = world_of(&pose, &parents, &root, 2).translation;
        let towards = normalised(sub(target, start));
        let actual = normalised(sub(end, start));
        assert!(dot(towards, actual) > 0.99, "aim {actual:?} target {towards:?}");
    }
}
