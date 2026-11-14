//! SAM 3D Body parameter decoding, natively.
//!
//! The exported `sam3d_pose.onnx` graph stops at the raw `pred` vector (519
//! values). Everything that turns `pred` into MHR's three inputs lives here
//! instead of in the graph:
//!
//! ```text
//! pred[0:6)         global rotation, 6-D            -> Euler ZYX   (3)
//! pred[6:266)       body pose, 260 continuous       -> 133 params
//! pred[266:311)     shape coefficients              -> identity    (45)
//! pred[311:339)     scale coefficients              -> scales      (68)
//! pred[339:447)     left + right hand, 54 + 54      -> 27 + 27 params
//! pred[447:519)     expression                      -> zeros       (72)
//! ```
//!
//! Why this is not in the ONNX graph: every lowering torch produces for
//! `compact_cont_to_model_params_body` / `batchXYZfrom6D` ends up as a node whose
//! first input is the empty string, and such a graph cannot be loaded. The maths
//! is a handful of `atan2` calls over one vector, so it belongs in Rust anyway.
//!
//! The tables (`hand_pose_mean`, `hand_pose_comps`, `scale_mean`, `scale_comps`
//! and the two hand joint index lists) are checkpoint data, so they ship next to
//! the graphs as `sam3d_params.json` + `sam3d_params.bin`, written by
//! `tools/onnx/sam3d_split.py`. The index tables below are source constants of
//! the reference implementation and are transcribed from
//! `sam_3d_body/models/modules/mhr_utils.py`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;

use crate::motion::rot6d_to_matrix;

/// Length of the decoded MHR pose vector (global + body), before the scales.
const FULL_POSE: usize = 136;
/// Length of the per-joint parameter vector `compact_cont_to_model_params_body`
/// produces, including the three jaw values the reference always zeroes.
const POSE_EULER: usize = 133;
/// Joint-major degree-of-freedom counts of one hand (16 joints, 27 params).
const HAND_DOFS: [usize; 16] = [3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 2, 3, 1, 1];
/// `sum(HAND_DOFS)`.
const HAND_PARAMS: usize = 27;

/// Model-parameter slots of the 23 three-DoF body joints (from the reference).
const BODY_3DOF: [usize; 69] = [
    0, 2, 4, 6, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 34,
    35, 36, 37, 38, 39, 44, 45, 46, 53, 54, 55, 64, 65, 66, 85, 69, 73, 86, 70, 79, 87, 71, 82, 88,
    72, 76, 91, 92, 93, 112, 96, 100, 113, 97, 106, 114, 98, 109, 115, 99, 103, 130, 131, 132,
];
/// Model-parameter slots of the 58 one-DoF body joints.
const BODY_1DOF: [usize; 58] = [
    1, 3, 5, 7, 9, 11, 30, 31, 32, 33, 40, 41, 42, 43, 47, 48, 49, 50, 51, 52, 56, 57, 58, 59, 60,
    61, 62, 63, 67, 68, 74, 75, 77, 78, 80, 81, 83, 84, 89, 90, 94, 95, 101, 102, 104, 105, 107,
    108, 110, 111, 116, 117, 118, 119, 120, 121, 122, 123,
];
/// Model-parameter slots of the six translation values.
const BODY_TRANS: [usize; 6] = [124, 125, 126, 127, 128, 129];
/// Slots zeroed because the hands come from the hand PCA basis instead
/// (`mhr_param_hand_idxs`, which is the contiguous range 62..=115).
const HAND_MASK: std::ops::RangeInclusive<usize> = 62..=115;

/// MHR's three inputs, decoded from one `pred` vector.
#[derive(Clone, Debug)]
pub struct DecodedParams {
    /// Shape coefficients — the first argument of the scripted MHR model.
    pub identity: Vec<f32>,
    /// 136 pose values (global translation/rotation + body) then 68 scales.
    pub model_params: Vec<f32>,
    /// Expression coefficients; the reference always multiplies them by zero.
    pub face: Vec<f32>,
}

#[derive(Deserialize)]
struct TensorEntry {
    offset: usize,
    len: usize,
    shape: Vec<usize>,
}

#[derive(Deserialize)]
struct Meta {
    npose: usize,
    body_cont_dim: usize,
    num_shape_comps: usize,
    num_scale_comps: usize,
    num_hand_comps: usize,
    num_face_comps: usize,
    hand_joint_idxs_left: Vec<usize>,
    hand_joint_idxs_right: Vec<usize>,
    tensors: BTreeMap<String, TensorEntry>,
}

/// The parameter tables plus the offsets into `pred` they imply.
pub struct ParamsTables {
    source: PathBuf,
    npose: usize,
    body_cont_dim: usize,
    num_shape_comps: usize,
    num_scale_comps: usize,
    /// Continuous hand values per hand: the PCA input width, not the 27 angles
    /// it decodes to.
    num_hand_comps: usize,
    num_face_comps: usize,
    hand_joint_idxs_left: Vec<usize>,
    hand_joint_idxs_right: Vec<usize>,
    hand_pose_mean: Vec<f32>,
    /// Row-major `[num_hand_pose_comps, hand_pose_len]`.
    hand_pose_comps: Vec<f32>,
    hand_pose_comps_cols: usize,
    scale_mean: Vec<f32>,
    /// Row-major `[num_scale_comps, scale_mean.len()]`.
    scale_comps: Vec<f32>,
    scale_comps_cols: usize,
}

impl ParamsTables {
    /// Load `<dir>/sam3d_params.json` and its `sam3d_params.bin` sidecar.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let meta_path = dir.join("sam3d_params.json");
        let bin_path = dir.join("sam3d_params.bin");
        let text = std::fs::read_to_string(&meta_path)
            .with_context(|| format!("reading {}", meta_path.display()))?;
        let meta: Meta = serde_json::from_str(&text)
            .with_context(|| format!("parsing {}", meta_path.display()))?;
        let raw = std::fs::read(&bin_path)
            .with_context(|| format!("reading {}", bin_path.display()))?;

        let fetch = |name: &str| -> Result<(Vec<f32>, Vec<usize>)> {
            let entry = meta
                .tensors
                .get(name)
                .ok_or_else(|| anyhow!("{name} missing from {}", meta_path.display()))?;
            let start = entry.offset * 4;
            let end = start + entry.len * 4;
            if end > raw.len() {
                anyhow::bail!(
                    "{name}: bytes {start}..{end} past the end of {} ({} bytes)",
                    bin_path.display(),
                    raw.len()
                );
            }
            let values = raw[start..end]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Ok((values, entry.shape.clone()))
        };

        let (hand_pose_mean, _) = fetch("hand_pose_mean")?;
        let (hand_pose_comps, comps_shape) = fetch("hand_pose_comps")?;
        let (scale_mean, _) = fetch("scale_mean")?;
        let (scale_comps, scale_shape) = fetch("scale_comps")?;

        if comps_shape.len() != 2 || comps_shape[1] != hand_pose_mean.len() {
            anyhow::bail!("hand_pose_comps has shape {comps_shape:?}, expected [rows, {}]", hand_pose_mean.len());
        }
        if scale_shape.len() != 2 || scale_shape[1] != scale_mean.len() {
            anyhow::bail!("scale_comps has shape {scale_shape:?}, expected [rows, {}]", scale_mean.len());
        }
        if scale_shape[0] != meta.num_scale_comps {
            anyhow::bail!(
                "scale_comps has {} rows, expected num_scale_comps = {}",
                scale_shape[0],
                meta.num_scale_comps
            );
        }
        if meta.num_hand_comps != hand_pose_mean.len() {
            anyhow::bail!(
                "num_hand_comps is {} but hand_pose_mean has {} values",
                meta.num_hand_comps,
                hand_pose_mean.len()
            );
        }

        Ok(Self {
            source: dir.to_path_buf(),
            npose: meta.npose,
            body_cont_dim: meta.body_cont_dim,
            num_shape_comps: meta.num_shape_comps,
            num_scale_comps: meta.num_scale_comps,
            num_hand_comps: meta.num_hand_comps,
            num_face_comps: meta.num_face_comps,
            hand_joint_idxs_left: meta.hand_joint_idxs_left,
            hand_joint_idxs_right: meta.hand_joint_idxs_right,
            hand_pose_comps_cols: comps_shape[1],
            scale_comps_cols: scale_shape[1],
            hand_pose_mean,
            hand_pose_comps,
            scale_mean,
            scale_comps,
        })
    }

    /// Where the tables were loaded from, for diagnostics.
    pub fn source(&self) -> &Path {
        &self.source
    }

    /// Number of values `decode` expects.
    pub fn npose(&self) -> usize {
        self.npose
    }

    /// `pred` (519 values) → MHR's `identity`, `model_params` and `face`.
    pub fn decode(&self, pred: &[f32]) -> Result<DecodedParams> {
        if pred.len() < self.npose {
            anyhow::bail!(
                "pred has {} values, expected at least {}",
                pred.len(),
                self.npose
            );
        }
        if self.body_cont_dim + 6 > pred.len() {
            anyhow::bail!("body_cont_dim {} does not fit in pred", self.body_cont_dim);
        }

        let mut cursor = 0usize;
        let global_rot = take(pred, &mut cursor, 6);
        let pose_cont = take(pred, &mut cursor, self.body_cont_dim);
        let identity = take(pred, &mut cursor, self.num_shape_comps).to_vec();
        let scale = take(pred, &mut cursor, self.num_scale_comps);
        let hand = take(pred, &mut cursor, self.num_hand_comps * 2);

        // Body pose: continuous (6-D / sincos) parameters to joint angles.
        let pose_euler = body_cont_to_euler(pose_cont)?;

        // Hands come from a PCA basis over the hand pose space, then get their
        // own compact-to-model conversion before being dropped into the pose.
        let (left_hand, right_hand) = hand.split_at(self.num_hand_comps);
        let left_model = self.hand_model_params(left_hand);
        let right_model = self.hand_model_params(right_hand);

        let mut model_params = vec![0.0f32; FULL_POSE + self.scale_mean.len()];
        // Global translation is zeroed by the reference (it predicts the
        // translation in the camera head instead).
        model_params[3..6].copy_from_slice(&euler_zyx(&rot6d_to_matrix([
            global_rot[0],
            global_rot[1],
            global_rot[2],
            global_rot[3],
            global_rot[4],
            global_rot[5],
        ])));
        // The reference truncates the 133-value pose to 130 here (the last three
        // are the jaw, which it already zeroed).
        model_params[6..FULL_POSE].copy_from_slice(&pose_euler[..FULL_POSE - 6]);

        for (slot, value) in self.hand_joint_idxs_left.iter().zip(left_model.iter()) {
            if let Some(target) = model_params.get_mut(*slot) {
                *target = *value;
            }
        }
        for (slot, value) in self.hand_joint_idxs_right.iter().zip(right_model.iter()) {
            if let Some(target) = model_params.get_mut(*slot) {
                *target = *value;
            }
        }

        // Scales: mean plus the predicted coefficients through the basis.
        for (index, mean) in self.scale_mean.iter().enumerate() {
            let mut total = *mean;
            for (row, coefficient) in scale.iter().enumerate() {
                total += coefficient * self.scale_comps[row * self.scale_comps_cols + index];
            }
            model_params[FULL_POSE + index] = total;
        }

        Ok(DecodedParams {
            identity,
            model_params,
            face: vec![0.0; self.num_face_comps],
        })
    }

    /// `compact_cont_to_model_params_hand`: 54 predicted values → 27 angles.
    fn hand_model_params(&self, hand_cont: &[f32]) -> [f32; HAND_PARAMS] {
        let blended = self.hand_blend(hand_cont);

        let mut out = [0.0f32; HAND_PARAMS];
        let mut parameter = 0usize;
        let mut continuous = 0usize;
        for dofs in HAND_DOFS {
            match dofs {
                3 => {
                    let angles = xyz_from_6d(&blended[continuous..continuous + 6]);
                    out[parameter..parameter + 3].copy_from_slice(&angles);
                    continuous += 6;
                }
                2 => {
                    // A two-DoF joint still stores one sincos pair per model
                    // parameter, so both angles come from consecutive pairs.
                    for offset in 0..2 {
                        let at = continuous + offset * 2;
                        out[parameter + offset] = blended[at].atan2(blended[at + 1]);
                    }
                    continuous += 4;
                }
                _ => {
                    out[parameter] = blended[continuous].atan2(blended[continuous + 1]);
                    continuous += 2;
                }
            }
            parameter += dofs;
        }
        out
    }

    /// `hand_pose_mean + hand_cont @ hand_pose_comps`, the PCA blend the compact
    /// conversion runs on.
    fn hand_blend(&self, hand_cont: &[f32]) -> Vec<f32> {
        self.hand_pose_mean
            .iter()
            .enumerate()
            .map(|(column, mean)| {
                let mut total = *mean;
                for (row, coefficient) in hand_cont.iter().enumerate() {
                    total +=
                        coefficient * self.hand_pose_comps[row * self.hand_pose_comps_cols + column];
                }
                total
            })
            .collect()
    }
}

/// `compact_cont_to_model_params_body`: 260 continuous values → 133 angles.
fn body_cont_to_euler(cont: &[f32]) -> Result<[f32; POSE_EULER]> {
    const THREEDOF_VALUES: usize = BODY_3DOF.len() / 3 * 6;
    const ONEDOF_VALUES: usize = BODY_1DOF.len() * 2;
    if cont.len() < THREEDOF_VALUES + ONEDOF_VALUES + BODY_TRANS.len() {
        anyhow::bail!(
            "body pose has {} values, expected {}",
            cont.len(),
            THREEDOF_VALUES + ONEDOF_VALUES + BODY_TRANS.len()
        );
    }

    let mut euler = [0.0f32; POSE_EULER];
    for (group, slots) in BODY_3DOF.chunks_exact(3).enumerate() {
        let angles = xyz_from_6d(&cont[group * 6..group * 6 + 6]);
        for (slot, angle) in slots.iter().zip(angles.iter()) {
            euler[*slot] = *angle;
        }
    }
    let one_dof = &cont[THREEDOF_VALUES..THREEDOF_VALUES + ONEDOF_VALUES];
    for (index, slot) in BODY_1DOF.iter().enumerate() {
        euler[*slot] = one_dof[index * 2].atan2(one_dof[index * 2 + 1]);
    }
    let translation = &cont[THREEDOF_VALUES + ONEDOF_VALUES..];
    for (slot, value) in BODY_TRANS.iter().zip(translation.iter()) {
        euler[*slot] = *value;
    }

    // The hands are replaced by the hand PCA blend; the jaw is not predicted.
    for slot in HAND_MASK {
        euler[slot] = 0.0;
    }
    euler[POSE_EULER - 3..].fill(0.0);
    Ok(euler)
}

/// `batchXYZfrom6D`: a 6-D rotation to Euler angles, singular branch included.
fn xyz_from_6d(r: &[f32]) -> [f32; 3] {
    let x = normalise([r[0], r[1], r[2]]);
    let z = normalise(cross(x, [r[3], r[4], r[5]]));
    let y = cross(z, x);

    // The matrix has x, y, z as its columns, so `matrix[2][1]` is `y[2]`.
    let sy = (x[0] * x[0] + x[1] * x[1]).sqrt();
    if sy < 1e-6 {
        // Gimbal lock: the reference swaps to the alternative branch.
        [(-z[1]).atan2(y[1]), (-x[2]).atan2(sy), 0.0]
    } else {
        [y[2].atan2(z[2]), (-x[2]).atan2(sy), x[1].atan2(x[0])]
    }
}

/// `roma.rotmat_to_euler("ZYX", …)`: extrinsic-ZYX Euler angles from a matrix.
///
/// Reproduces roma's Markley quaternion extraction (largest diagonal term, no
/// sign canonicalisation) followed by the intrinsic `ZYX` branch of its
/// quaternion-to-Euler routine, including the two gimbal-lock cases.
fn euler_zyx(m: &[[f32; 3]; 3]) -> [f32; 3] {
    // roma works on A = R^T.
    let a = [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ];
    // Each row is 4 q_i q; the largest diagonal entry picks the best-conditioned
    // one, which is then normalised into the quaternion.
    let rows = [
        [
            1.0 + a[0][0] - a[1][1] - a[2][2],
            a[0][1] + a[1][0],
            a[0][2] + a[2][0],
            a[1][2] - a[2][1],
        ],
        [
            a[1][0] + a[0][1],
            1.0 + a[1][1] - a[2][2] - a[0][0],
            a[1][2] + a[2][1],
            a[2][0] - a[0][2],
        ],
        [
            a[2][0] + a[0][2],
            a[2][1] + a[1][2],
            1.0 + a[2][2] - a[0][0] - a[1][1],
            a[0][1] - a[1][0],
        ],
        [
            a[1][2] - a[2][1],
            a[2][0] - a[0][2],
            a[0][1] - a[1][0],
            1.0 + a[0][0] + a[1][1] + a[2][2],
        ],
    ];
    let mut best = 0;
    for index in 1..4 {
        if rows[index][index] > rows[best][best] {
            best = index;
        }
    }
    let q = normalise4(rows[best]);

    // Intrinsic "ZYX" maps to the extrinsic "xyz" branch internally.
    let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
    let (h0, h1, h2, h3) = (w - y, x + z, y + w, z - x);
    let mut angles = [0.0f32; 3];
    angles[1] = 2.0 * h2.hypot(h3).atan2(h0.hypot(h1));

    let half_sum = h1.atan2(h0);
    let half_diff = h3.atan2(h2);
    angles[2] = half_sum - half_diff;
    angles[0] = half_sum + half_diff;

    let case_first = angles[1].abs() <= 1e-7;
    let case_second = (angles[1] - std::f32::consts::PI).abs() <= 1e-7;
    if case_first || case_second {
        angles[2] = 0.0;
    }
    if case_first {
        angles[0] = 2.0 * half_sum;
    }
    if case_second {
        angles[0] = 2.0 * half_diff;
    }

    // Tait-Bryan: the second angle is measured from the equator.
    angles[1] -= std::f32::consts::FRAC_PI_2;
    for angle in angles.iter_mut() {
        if *angle < -std::f32::consts::PI {
            *angle += std::f32::consts::TAU;
        }
        if *angle > std::f32::consts::PI {
            *angle -= std::f32::consts::TAU;
        }
    }
    angles
}

fn take<'a>(pred: &'a [f32], cursor: &mut usize, count: usize) -> &'a [f32] {
    let slice = &pred[*cursor..*cursor + count];
    *cursor += count;
    slice
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalise(v: [f32; 3]) -> [f32; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-12);
    [v[0] / n, v[1] / n, v[2] / n]
}

fn normalise4(v: [f32; 4]) -> [f32; 4] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3])
        .sqrt()
        .max(1e-12);
    [v[0] / n, v[1] / n, v[2] / n, v[3] / n]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tables_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/flowhmr/sam")
    }

    #[test]
    fn identity_rotation_has_zero_euler() {
        let euler = euler_zyx(&rot6d_to_matrix([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]));
        for angle in euler {
            assert!(angle.abs() < 1e-6, "expected 0, got {euler:?}");
        }
    }

    #[test]
    fn six_d_round_trips_a_known_rotation() {
        // A pure 30° yaw lands in the second slot of the three Euler angles.
        let angle = 30f32.to_radians();
        let (s, c) = angle.sin_cos();
        // First two columns of a rotation about Y.
        let m = [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]];
        let six = [m[0][0], m[1][0], m[2][0], m[0][1], m[1][1], m[2][1]];
        let euler = xyz_from_6d(&six);
        assert!(euler[0].abs() < 1e-5 && euler[2].abs() < 1e-5, "{euler:?}");
        assert!((euler[1] - angle).abs() < 1e-5, "{euler:?}");
    }

    #[test]
    fn decodes_like_the_reference_head() {
        let dir = tables_dir();
        let meta = dir.join("sam3d_params.json");
        let fixture = dir.join("sam3d_params_ref.json");
        if !meta.is_file() || !fixture.is_file() {
            eprintln!("skipping: no parameter tables at {}", dir.display());
            return;
        }
        let tables = ParamsTables::load(&dir).unwrap();
        let text = std::fs::read_to_string(&fixture).unwrap();
        let document: serde_json::Value = serde_json::from_str(&text).unwrap();
        let cases = document["cases"].as_array().expect("cases array");
        assert!(!cases.is_empty());

        let floats = |value: &serde_json::Value| -> Vec<f32> {
            value
                .as_array()
                .expect("float array")
                .iter()
                .map(|v| v.as_f64().expect("number") as f32)
                .collect()
        };

        let mut worst_identity = 0.0f32;
        let mut worst_params = 0.0f32;
        let mut worst_face = 0.0f32;
        for case in cases {
            let pred = floats(&case["pred"]);
            assert_eq!(pred.len(), tables.npose());
            let decoded = tables.decode(&pred).unwrap();

            let identity = floats(&case["identity"]);
            let params = floats(&case["model_params"]);
            let face = floats(&case["face"]);
            assert_eq!(decoded.identity.len(), identity.len());
            assert_eq!(decoded.model_params.len(), params.len());
            assert_eq!(decoded.face.len(), face.len());

            for (got, want) in decoded.identity.iter().zip(identity.iter()) {
                worst_identity = worst_identity.max((got - want).abs());
            }
            for (got, want) in decoded.model_params.iter().zip(params.iter()) {
                worst_params = worst_params.max((got - want).abs());
            }
            for (got, want) in decoded.face.iter().zip(face.iter()) {
                worst_face = worst_face.max((got - want).abs());
            }
        }
        eprintln!(
            "{} cases: identity {:.3e}  model_params {:.3e}  face {:.3e}",
            cases.len(),
            worst_identity,
            worst_params,
            worst_face
        );
        if worst_params >= 2e-4 {
            // Say *where* it drifted: the four segments have separate code paths.
            let pred = floats(&cases[0]["pred"]);
            let want = floats(&cases[0]["model_params"]);
            let got = tables.decode(&pred).unwrap().model_params;
            let mut bad: Vec<(usize, f32, f32)> = got
                .iter()
                .zip(want.iter())
                .enumerate()
                .map(|(index, (g, w))| (index, *g, *w))
                .filter(|(_, g, w)| (g - w).abs() > 1e-4)
                .collect();
            bad.sort_by(|a, b| (b.1 - b.2).abs().total_cmp(&(a.1 - a.2).abs()));
            eprintln!("  {} mismatching slots, worst 8:", bad.len());
            for (index, g, w) in bad.iter().take(8) {
                eprintln!("    slot {index:3}  got {g:.6}  want {w:.6}");
            }
            eprintln!(
                "  left idxs  {:?}\n  right idxs {:?}",
                tables.hand_joint_idxs_left, tables.hand_joint_idxs_right
            );
            // Hand shapes carry the PCA blend, so print it next to the angles.
            for (label, range) in [("left", 339..393), ("right", 393..447)] {
                let blended = tables.hand_blend(&pred[range.clone()]);
                let rounded: Vec<f32> = blended[..6].iter().map(|v| (v * 1e6).round() / 1e6).collect();
                eprintln!("  {label} blend[..6] {rounded:?}");
                eprintln!(
                    "  {label} angles     {:?}",
                    tables.hand_model_params(&pred[range])
                );
            }
        }
        assert!(
            worst_identity < 1e-5,
            "identity max abs {:.3e}",
            worst_identity
        );
        assert!(
            worst_params < 2e-4,
            "model_params max abs {:.3e}",
            worst_params
        );
        assert!(worst_face < 1e-6, "face max abs {:.3e}", worst_face);
    }
}
