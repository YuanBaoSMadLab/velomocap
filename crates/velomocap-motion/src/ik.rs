//! Transform / rotation helpers for the anatomical solver.
//!
//! The two IK bodies this module used to carry (`IK::TwoBoneIk`, `IK::OneBoneIk`
//! from NVIDIA Kimodo's `MotionCorrection`, Apache-2.0) were removed: nothing
//! called them. The leg solve is `anatomical::solve_leg` (a clamped two-bone
//! solve that keeps the knee on its calibrated hinge axis) and the arm solve is
//! `arm_plan` + `solve_arm`; both are the passes the correction actually runs,
//! and keeping a second, unused IK next to them only made the coverage table
//! claim something the pipeline never did.
//!
//! What remains is the part the solver does use: `axis_angle_matrix` (Rodrigues,
//! the reference's rotation convention), the `Transform` chain and `concat`,
//! whose operand order is Kimodo's (`a` applied first, then `b`) — getting that
//! backwards silently mirrors a chain.

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
/// then `b`.
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

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
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

/// Rotation matrix from an axis and an angle (Rodrigues).
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

#[cfg(test)]
mod tests {
    use super::*;

    /// `concat` applies `a` first and then `b`, which is Kimodo's order and the
    /// convention the exported rotations rely on. A `b * a` reading moves the
    /// chain somewhere else entirely.
    #[test]
    fn concat_applies_the_first_operand_first() {
        let up = Transform {
            rotation: Transform::IDENTITY.rotation,
            translation: [0.0, 1.0, 0.0],
        };
        let down = Transform {
            rotation: Transform::IDENTITY.rotation,
            translation: [0.0, -0.3, 0.0],
        };
        let tip = concat(&concat(&up, &down), &down);
        assert!((tip.translation[1] - 0.4).abs() < 1e-9, "tip {tip:?}");
    }

    /// A rotation in `b` turns `a`'s translation with it, which is what makes the
    /// operand order observable.
    #[test]
    fn a_rotation_in_the_second_operand_turns_the_first_translation() {
        let offset = Transform {
            rotation: Transform::IDENTITY.rotation,
            translation: [1.0, 0.0, 0.0],
        };
        let quarter = Transform::from_rotation(axis_angle_matrix([0.0, 1.0, 0.0], std::f64::consts::FRAC_PI_2));
        let placed = concat(&offset, &quarter);
        // +x turned a quarter about +y is -z.
        assert!(placed.translation[2].abs() > 0.99, "placed {placed:?}");
        assert!(placed.translation[0].abs() < 1e-9, "placed {placed:?}");
    }

    #[test]
    fn an_axis_angle_round_trips_through_the_matrix() {
        // A rotation about +y leaves the y axis alone and puts cos on x/z.
        let matrix = axis_angle_matrix([0.0, 1.0, 0.0], 0.7);
        assert!((matrix[4] - 1.0).abs() < 1e-12, "y axis moved");
        assert!((matrix[0] - 0.7f64.cos()).abs() < 1e-12);
        assert!((matrix[2] - 0.7f64.sin()).abs() < 1e-12);
    }
}
