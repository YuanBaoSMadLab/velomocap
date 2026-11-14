//! Small fixed-size linear algebra and SO(3) helpers shared by the core.
//!
//! Matrices are 3x3 row-major `[f64; 9]`. Batched conversions are delegated to
//! the C++ kernels; the helpers here only cover scalar work that is called a
//! handful of times per clip (rest-pose setup, per-joint basis construction).

use crate::cpp;

pub const PARENTS: [i32; 22] = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
];

pub const BODY_BONES: usize = 52;

pub const IDENTITY3: [f64; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

#[inline]
pub fn mat3(a: &[f64], index: usize) -> &[f64] {
    &a[index * 9..index * 9 + 9]
}

#[inline]
pub fn mat3_mut(a: &mut [f64], index: usize) -> &mut [f64] {
    &mut a[index * 9..index * 9 + 9]
}

#[inline]
pub fn vec3(a: &[f64], index: usize) -> [f64; 3] {
    [a[index * 3], a[index * 3 + 1], a[index * 3 + 2]]
}

pub fn dot3(a: &[f64], b: &[f64]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn cross3(a: &[f64], b: &[f64]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub fn norm3(a: &[f64]) -> f64 {
    dot3(a, a).sqrt()
}

/// Normalise with the pipeline's 1e-10 floor so degenerate segments stay finite.
pub fn unit3(a: &[f64]) -> [f64; 3] {
    let n = norm3(a);
    let d = if n > 1e-10 { n } else { 1e-10 };
    [a[0] / d, a[1] / d, a[2] / d]
}

/// 3x3 matrix product `a @ b`.
#[inline]
pub fn mul3(a: &[f64], b: &[f64]) -> [f64; 9] {
    let mut o = [0.0; 9];
    for i in 0..3 {
        let a0 = a[i * 3];
        let a1 = a[i * 3 + 1];
        let a2 = a[i * 3 + 2];
        o[i * 3] = a0 * b[0] + a1 * b[3] + a2 * b[6];
        o[i * 3 + 1] = a0 * b[1] + a1 * b[4] + a2 * b[7];
        o[i * 3 + 2] = a0 * b[2] + a1 * b[5] + a2 * b[8];
    }
    o
}

/// `a^T @ b`
#[inline]
pub fn tmul3(a: &[f64], b: &[f64]) -> [f64; 9] {
    let mut o = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            o[i * 3 + j] =
                a[i] * b[j] + a[3 + i] * b[3 + j] + a[6 + i] * b[6 + j];
        }
    }
    o
}

/// `a @ b^T`
#[inline]
pub fn mul_t3(a: &[f64], b: &[f64]) -> [f64; 9] {
    let mut o = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            o[i * 3 + j] =
                a[i * 3] * b[j * 3]
                    + a[i * 3 + 1] * b[j * 3 + 1]
                    + a[i * 3 + 2] * b[j * 3 + 2];
        }
    }
    o
}

#[inline]
pub fn transpose3(a: &[f64]) -> [f64; 9] {
    [
        a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8],
    ]
}

/// Shortest-arc proper rotation taking `a` to `b` (deterministic at pi).
pub fn align(a: &[f64], b: &[f64]) -> [f64; 9] {
    let mut out = [0.0; 9];
    cpp::align(a, b, &mut out);
    out
}

/// Zero-roll +Y bone frame for a bone direction (the BVH reference basis).
pub fn reference_frame(delta: &[f64]) -> [f64; 9] {
    let mut out = [0.0; 9];
    cpp::reference_frame(delta, &mut out);
    out
}

/// Rotation angle of `a @ b^T` in `[0, pi]` (scipy `Rotation.magnitude`).
pub fn relative_magnitude(a: &[f64], b: &[f64]) -> f64 {
    let mut out = [0.0; 1];
    cpp::relative_magnitude(a, b, &mut out);
    out[0]
}

pub fn magnitude(r: &[f64]) -> f64 {
    let trace = (r[0] + r[4] + r[8]).clamp(-1.0, 3.0);
    (0.5 * (trace - 1.0)).clamp(-1.0, 1.0).acos()
}

/// Euler order description: axis permutation plus intrinsic/extrinsic sense.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EulerOrder {
    pub perm: [usize; 3],
    pub intrinsic: bool,
}

impl EulerOrder {
    /// Parse a scipy order string: uppercase letters mean intrinsic rotations.
    pub fn parse(order: &str) -> Self {
        let chars: Vec<char> = order.chars().collect();
        assert_eq!(chars.len(), 3, "Euler order needs three axes: {order:?}");
        let intrinsic = chars[0].is_ascii_uppercase();
        let mut perm = [0usize; 3];
        for (i, c) in chars.iter().enumerate() {
            perm[i] = match c.to_ascii_lowercase() {
                'x' => 0,
                'y' => 1,
                'z' => 2,
                other => panic!("Unsupported Euler axis {other:?} in {order:?}"),
            };
        }
        Self { perm, intrinsic }
    }

    /// Axis indices handed to the kernel, plus whether the angles come back
    /// in reverse slot order (which is how extrinsic orders are defined).
    fn kernel_form(&self) -> ([usize; 3], bool) {
        if self.intrinsic {
            (self.perm, false)
        } else {
            // Extrinsic (a,b,c) is the intrinsic product (c,b,a) of the same
            // matrix, so the angles come back reversed. This mirrors SciPy
            // exactly, including its rounding.
            ([self.perm[2], self.perm[1], self.perm[0]], true)
        }
    }
}

/// Convert rotation matrices to Euler angles in the requested order.
/// Returns `(angles, margins)` with `margins = |cos(middle angle)|`.
pub fn mat_to_euler(matrices: &[f64], order: &str) -> (Vec<f64>, Vec<f64>) {
    let parsed = EulerOrder::parse(order);
    let (perm, reverse) = parsed.kernel_form();
    let n = matrices.len() / 9;
    let (mut angles, mut margin) = cpp::mat_to_euler(matrices, perm);
    if reverse {
        for t in 0..n {
            angles.swap(t * 3, t * 3 + 2);
        }
    }
    // Margin is symmetric under angle reversal only for the middle slot; the
    // kernel already reports |cos(beta)| which is order-invariant.
    margin.truncate(n);
    (angles, margin)
}

/// Build rotation matrices from Euler angles in the requested order.
pub fn euler_to_mat(angles: &[f64], order: &str) -> Vec<f64> {
    let parsed = EulerOrder::parse(order);
    let (perm, reverse) = parsed.kernel_form();
    let n = angles.len() / 3;
    let mut kernel_input = angles.to_vec();
    if reverse {
        for t in 0..n {
            kernel_input.swap(t * 3, t * 3 + 2);
        }
    }
    let mut out = vec![0.0; n * 9];
    cpp::euler_to_mat(&kernel_input, perm, &mut out);
    out
}

/// Remove 2*pi jumps from a per-frame Euler track (in place).
pub fn continuous_angles(angles: &mut [f64]) {
    cpp::continuous_angles(angles);
}

/// Apply a rigid 4x4 transform to a batch of 3x3 rotations and positions.
pub fn apply_transform_rotations(matrices: &mut [f64], rotation: &[f64; 9]) {
    for chunk in matrices.chunks_exact_mut(9) {
        let m = mul3(rotation, chunk);
        chunk.copy_from_slice(&m);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_roundtrips() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        let r = align(&a, &b);
        let mapped = [
            r[0] * a[0] + r[1] * a[1] + r[2] * a[2],
            r[3] * a[0] + r[4] * a[1] + r[5] * a[2],
            r[6] * a[0] + r[7] * a[1] + r[8] * a[2],
        ];
        assert!((mapped[1] - 1.0).abs() < 1e-12);
        assert!((relative_magnitude(&r, &IDENTITY3) - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
    }

    #[test]
    fn euler_roundtrip_all_orders() {
        let angles = [0.3, -0.7, 1.1];
        for order in ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx", "XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"] {
            let m = euler_to_mat(&angles, order);
            let (back, margin) = mat_to_euler(&m, order);
            for i in 0..3 {
                assert!(
                    (back[i] - angles[i]).abs() < 1e-9,
                    "{order} angle {i}: {} vs {}",
                    back[i],
                    angles[i]
                );
            }
            assert!(margin[0] > 0.5);
        }
    }
}
