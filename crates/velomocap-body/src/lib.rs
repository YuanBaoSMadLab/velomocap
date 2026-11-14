//! SMPL-X body model in pure Rust.
//!
//! Loads the official `.npz` and implements the forward pass: shape blending,
//! joint regression, the kinematic chain, linear blend skinning and the pose
//! blend shapes. No Python, no PyTorch, no NumPy.
//!
//! Two forward paths exist because they have very different costs:
//!
//! * [`BodyModel::forward_joints`] — only the 55 joints. This is what pose
//!   fitting needs, and it runs in microseconds.
//! * [`BodyModel::forward_mesh`] — joints *and* all 10475 skinned vertices.
//!   Used for preview and mesh export; needs `posedirs`.
//!
//! `posedirs` (122 MB in the `.npz`) is optional and only loaded on request.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rayon::prelude::*;

use velomocap_cpp as cpp;

pub mod npz;

/// The 55 SMPL-X joint names, in model order.
pub const JOINT_NAMES: [&str; 55] = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar",
    "right_collar", "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "jaw", "left_eye_smplhf", "right_eye_smplhf", "left_index1",
    "left_index2", "left_index3", "left_middle1", "left_middle2", "left_middle3", "left_pinky1",
    "left_pinky2", "left_pinky3", "left_ring1", "left_ring2", "left_ring3", "left_thumb1",
    "left_thumb2", "left_thumb3", "right_index1", "right_index2", "right_index3", "right_middle1",
    "right_middle2", "right_middle3", "right_pinky1", "right_pinky2", "right_pinky3",
    "right_ring1", "right_ring2", "right_ring3", "right_thumb1", "right_thumb2", "right_thumb3",
];

/// A body pose: global orientation, joint rotations, shape and translation.
#[derive(Clone, Debug)]
pub struct Pose {
    /// Root rotation as an axis-angle vector (radians).
    pub root: [f32; 3],
    /// `(joints - 1) * 3` axis-angle vectors, parent-relative.
    pub body: Vec<f32>,
    /// Shape coefficients (length `num_betas`).
    pub betas: Vec<f32>,
    /// Root translation in metres.
    pub translation: [f32; 3],
}

impl Pose {
    /// A neutral, standing pose with the given shape.
    pub fn neutral(num_joints: usize, num_betas: usize) -> Self {
        Self {
            root: [0.0; 3],
            body: vec![0.0; (num_joints - 1) * 3],
            betas: vec![0.0; num_betas],
            translation: [0.0; 3],
        }
    }

    pub fn set_joint(&mut self, joint: usize, rotvec: [f32; 3]) {
        if joint == 0 {
            self.root = rotvec;
        } else if (joint - 1) * 3 + 3 <= self.body.len() {
            self.body[(joint - 1) * 3..(joint - 1) * 3 + 3].copy_from_slice(&rotvec);
        }
    }

    pub fn joint(&self, joint: usize) -> [f32; 3] {
        if joint == 0 {
            self.root
        } else {
            let i = (joint - 1) * 3;
            [self.body[i], self.body[i + 1], self.body[i + 2]]
        }
    }
}

/// Forward-pass output: world joint positions and rotations.
#[derive(Clone, Debug, Default)]
pub struct Skeleton {
    /// `joints * 3` world positions.
    pub positions: Vec<f32>,
    /// `joints * 9` world rotations, row-major.
    pub rotations: Vec<f32>,
}

impl Skeleton {
    pub fn joint(&self, index: usize) -> [f32; 3] {
        [
            self.positions[index * 3],
            self.positions[index * 3 + 1],
            self.positions[index * 3 + 2],
        ]
    }

    pub fn num_joints(&self) -> usize {
        self.positions.len() / 3
    }
}

/// The loaded body model.
pub struct BodyModel {
    pub num_vertices: usize,
    pub num_joints: usize,
    pub num_betas: usize,
    pub parents: Vec<i32>,
    pub names: Vec<String>,
    /// `vertices * 3`
    v_template: Vec<f32>,
    /// `vertices * 3 * num_betas`
    shapedirs: Vec<f32>,
    /// `joints * vertices`, row-major
    j_regressor: Vec<f32>,
    /// `vertices * joints`
    weights: Vec<f32>,
    /// `faces * 3`
    pub faces: Vec<u32>,
    /// `joints * 3` rest positions in the zero pose
    pub template_joints: Vec<f32>,
    /// `vertices * 3 * 9 * (joints - 1)`, only when requested
    posedirs: Option<Vec<f32>>,
    source: PathBuf,
}

impl BodyModel {
    /// Load a SMPL-X `.npz`.
    ///
    /// `num_betas` truncates the 400 shape components to the first `n` (10 is
    /// the body-shape block; the rest are facial expression). Truncating keeps
    /// the in-memory model at a few megabytes instead of ~130.
    pub fn load(path: impl AsRef<Path>, num_betas: usize) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if !path.is_file() {
            anyhow::bail!("SMPL-X model not found: {}", path.display());
        }

        let v_template = npz::read_member(&path, "v_template")
            .context("reading v_template")?
            .to_row_major();
        let num_vertices = v_template.len() / 3;

        let shapedirs_raw =
            npz::read_member(&path, "shapedirs").context("reading shapedirs")?;
        if shapedirs_raw.shape.len() != 3 {
            anyhow::bail!("shapedirs has shape {:?}, expected 3-D", shapedirs_raw.shape);
        }
        let available = shapedirs_raw.shape[2];
        let num_betas = num_betas.clamp(1, available);
        let shapedirs = truncate_shape_dirs(&shapedirs_raw, num_vertices, num_betas);

        let j_regressor = npz::read_member(&path, "J_regressor")
            .context("reading J_regressor")?
            .to_row_major();
        let num_joints = j_regressor.len() / num_vertices;

        let weights = npz::read_member(&path, "weights")
            .context("reading weights")?
            .to_row_major();

        let kintree = npz::read_member(&path, "kintree_table")
            .context("reading kintree_table")?
            .array
            .to_i64();
        if kintree.len() < num_joints * 2 {
            anyhow::bail!("kintree_table is too small for {num_joints} joints");
        }
        // Row 0 holds the parents; the root is stored as a sentinel value.
        let mut parents = Vec::with_capacity(num_joints);
        for j in 0..num_joints {
            let raw = kintree[j];
            parents.push(if raw < 0 || raw as usize >= num_joints {
                -1
            } else {
                raw as i32
            });
        }

        let faces = npz::read_member(&path, "f")
            .context("reading faces")?
            .array
            .to_u32();

        let mut model = Self {
            num_vertices,
            num_joints,
            num_betas,
            parents,
            names: JOINT_NAMES
                .iter()
                .take(num_joints)
                .map(|s| s.to_string())
                .collect(),
            v_template,
            shapedirs,
            j_regressor,
            weights,
            faces,
            template_joints: Vec::new(),
            posedirs: None,
            source: path,
        };
        // Rest joints in the zero pose, cached: they are needed every frame.
        model.template_joints = model.rest_joints(&vec![0.0; num_betas]);
        Ok(model)
    }

    /// Load the pose blend shapes (122 MB) so [`Self::forward_mesh`] can run.
    pub fn with_posedirs(mut self) -> Result<Self> {
        let raw = npz::read_member(&self.source, "posedirs").context("reading posedirs")?;
        if raw.shape.len() != 3 {
            anyhow::bail!("posedirs has shape {:?}, expected 3-D", raw.shape);
        }
        let components = raw.shape[2];
        let expected = self.num_vertices * 3 * components;
        let data = raw.array.to_f32();
        if data.len() < expected {
            anyhow::bail!("posedirs payload is short: {} of {expected}", data.len());
        }
        self.posedirs = Some(data);
        Ok(self)
    }

    pub fn has_posedirs(&self) -> bool {
        self.posedirs.is_some()
    }

    pub fn source(&self) -> &Path {
        &self.source
    }

    /// Rest joints for a given shape (no pose applied).
    pub fn rest_joints(&self, betas: &[f32]) -> Vec<f32> {
        let shaped = self.shape_vertices(betas);
        self.regress_joints(&shaped)
    }

    /// `v_template + shapedirs · betas` (`vertices * 3`).
    fn shape_vertices(&self, betas: &[f32]) -> Vec<f32> {
        let v = self.num_vertices;
        let mut out = self.v_template.clone();
        // `shapedirs` is stored interleaved: `(vertex * 3 + axis) * betas + b`,
        // which is the layout `truncate_shape_dirs` writes and the one the npz
        // has (`(6890, 3, 16)` in C order). Indexing it as `b * vertices * 3`
        // instead reads an unrelated component for every beta and turns the
        // body into a lumpy, spiky mesh - which is exactly how it looked.
        let stride = self.num_betas.max(1);
        for (b, beta) in betas.iter().enumerate().take(self.num_betas) {
            if *beta == 0.0 {
                continue;
            }
            for i in 0..v * 3 {
                out[i] += self.shapedirs[i * stride + b] * beta;
            }
        }
        out
    }

    /// `J_regressor @ vertices` (`joints * 3`).
    fn regress_joints(&self, vertices: &[f32]) -> Vec<f32> {
        let j = self.num_joints;
        let v = self.num_vertices;
        let mut out = vec![0.0f32; j * 3];
        out.par_chunks_mut(3).enumerate().for_each(|(joint, slot)| {
            let row = &self.j_regressor[joint * v..(joint + 1) * v];
            let mut acc = [0.0f32; 3];
            for (vertex, weight) in row.iter().enumerate() {
                if *weight == 0.0 {
                    continue;
                }
                let base = vertex * 3;
                acc[0] += weight * vertices[base];
                acc[1] += weight * vertices[base + 1];
                acc[2] += weight * vertices[base + 2];
            }
            slot[0] = acc[0];
            slot[1] = acc[1];
            slot[2] = acc[2];
        });
        out
    }

    /// Local rotation matrices for every joint, from the pose.
    fn local_rotations(&self, pose: &Pose) -> Vec<f64> {
        let j = self.num_joints;
        let mut rotvecs = vec![0.0f64; j * 3];
        for (i, value) in pose.root.iter().enumerate() {
            rotvecs[i] = *value as f64;
        }
        for joint in 1..j {
            let src = (joint - 1) * 3;
            if src + 3 <= pose.body.len() {
                for axis in 0..3 {
                    rotvecs[joint * 3 + axis] = pose.body[src + axis] as f64;
                }
            }
        }
        let mut out = vec![0.0f64; j * 9];
        cpp::rotvec_to_mat(&rotvecs, &mut out);
        out
    }

    /// World transforms `G[j]` (4x4 row-major) from the rest joints.
    fn world_transforms(&self, rest: &[f32], local: &[f64]) -> Vec<f64> {
        let j = self.num_joints;
        let mut out = vec![0.0f64; j * 16];
        for joint in 0..j {
            let r = &local[joint * 9..joint * 9 + 9];
            let parent = self.parents[joint];

            // The joint's translation relative to its parent, in parent space.
            let mut t = [0.0f64; 3];
            if parent < 0 {
                t = [
                    rest[joint * 3] as f64,
                    rest[joint * 3 + 1] as f64,
                    rest[joint * 3 + 2] as f64,
                ];
            } else {
                let p = parent as usize;
                for axis in 0..3 {
                    t[axis] = rest[joint * 3 + axis] as f64 - rest[p * 3 + axis] as f64;
                }
            }

            let mut m = [0.0f64; 16];
            for row in 0..3 {
                for col in 0..3 {
                    m[row * 4 + col] = r[row * 3 + col];
                }
                m[row * 4 + 3] = t[row];
            }
            m[15] = 1.0;

            if parent < 0 {
                out[joint * 16..joint * 16 + 16].copy_from_slice(&m);
            } else {
                let p = parent as usize;
                let pm = &out[p * 16..p * 16 + 16];
                let mut composed = [0.0f64; 16];
                for row in 0..3 {
                    for col in 0..4 {
                        let mut acc = 0.0;
                        for k in 0..3 {
                            acc += pm[row * 4 + k] * m[k * 4 + col];
                        }
                        if col == 3 {
                            acc += pm[row * 4 + 3];
                        }
                        composed[row * 4 + col] = acc;
                    }
                }
                composed[15] = 1.0;
                out[joint * 16..joint * 16 + 16].copy_from_slice(&composed);
            }
        }
        out
    }

    /// Joints only — the fast path used by pose fitting.
    pub fn forward_joints(&self, pose: &Pose) -> Skeleton {
        let local = self.local_rotations(pose);
        let rest = if pose.betas.iter().all(|b| *b == 0.0) {
            self.template_joints.clone()
        } else {
            self.rest_joints(&pose.betas)
        };
        let world = self.world_transforms(&rest, &local);

        let j = self.num_joints;
        let mut positions = vec![0.0f32; j * 3];
        let mut rotations = vec![0.0f32; j * 9];
        for joint in 0..j {
            let g = &world[joint * 16..joint * 16 + 16];
            // The transform chain's translation column *is* the posed joint
            // position (SMPL's `batch_rigid_transform` reads it the same way):
            // G[j] maps the zero-pose skeleton into world space, so applying it
            // to rest[j] again would double-count the offset.
            positions[joint * 3] = g[3] as f32 + pose.translation[0];
            positions[joint * 3 + 1] = g[7] as f32 + pose.translation[1];
            positions[joint * 3 + 2] = g[11] as f32 + pose.translation[2];
            for row in 0..3 {
                for col in 0..3 {
                    rotations[joint * 9 + row * 3 + col] = g[row * 4 + col] as f32;
                }
            }
        }
        Skeleton {
            positions,
            rotations,
        }
    }

    /// Skinning transforms `A[j] = [R_j | t_j - R_j · rest_j]`.
    ///
    /// SMPL subtracts the rest pose so the blend matrices are *relative*: with
    /// every rotation at identity, `A[j]` is the identity and the vertices stay
    /// at `v_posed`. Note the subtraction uses the **rotation only** (the
    /// official implementation pads the joint to homogeneous coordinates with
    /// w = 0), not the full transform applied to a point. Using w = 1 here
    /// collapses a 1.72 m body to 0.45 m.
    fn skinning_transforms(&self, rest: &[f32], world: &[f64]) -> Vec<f64> {
        let j = self.num_joints;
        let mut out = vec![0.0f64; j * 16];
        for joint in 0..j {
            let g = &world[joint * 16..joint * 16 + 16];
            let x = rest[joint * 3] as f64;
            let y = rest[joint * 3 + 1] as f64;
            let z = rest[joint * 3 + 2] as f64;
            let mut a = [0.0f64; 16];
            for row in 0..3 {
                // R_j · rest_j (no translation term).
                let rotated = g[row * 4] * x + g[row * 4 + 1] * y + g[row * 4 + 2] * z;
                a[row * 4] = g[row * 4];
                a[row * 4 + 1] = g[row * 4 + 1];
                a[row * 4 + 2] = g[row * 4 + 2];
                a[row * 4 + 3] = g[row * 4 + 3] - rotated;
            }
            a[15] = 1.0;
            out[joint * 16..joint * 16 + 16].copy_from_slice(&a);
        }
        out
    }

    /// Joints plus the full skinned mesh. Requires [`Self::with_posedirs`].
    pub fn forward_mesh(&self, pose: &Pose) -> Result<(Skeleton, Vec<f32>)> {
        let posedirs = self
            .posedirs
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("posedirs not loaded; call with_posedirs()"))?;
        let skeleton = self.forward_joints(pose);
        let local = self.local_rotations(pose);
        let rest = if pose.betas.iter().all(|b| *b == 0.0) {
            self.template_joints.clone()
        } else {
            self.rest_joints(&pose.betas)
        };
        let world = self.world_transforms(&rest, &local);
        let skin = self.skinning_transforms(&rest, &world);

        // Pose blend shapes: pose_feature = (R_local[1:] - I) flattened.
        let components = 9 * (self.num_joints - 1);
        let mut feature = vec![0.0f32; components];
        for joint in 1..self.num_joints {
            let base = (joint - 1) * 9;
            for k in 0..9 {
                let identity = if k % 4 == 0 { 1.0 } else { 0.0 };
                feature[base + k] = local[joint * 9 + k] as f32 - identity;
            }
        }

        let v = self.num_vertices;
        let shaped = self.shape_vertices(&pose.betas);

        let mut vertices = vec![0.0f32; v * 3];
        vertices
            .par_chunks_mut(3)
            .enumerate()
            .for_each(|(vertex, slot)| {
                let shaped_base = vertex * 3;
                let mut p = [
                    shaped[shaped_base],
                    shaped[shaped_base + 1],
                    shaped[shaped_base + 2],
                ];

                // posedirs layout is (V, 3, components) in C order.
                let dir_base = vertex * 3 * components;
                for c in 0..components {
                    let f = feature[c];
                    if f == 0.0 {
                        continue;
                    }
                    p[0] += posedirs[dir_base + c] * f;
                    p[1] += posedirs[dir_base + components + c] * f;
                    p[2] += posedirs[dir_base + 2 * components + c] * f;
                }

                // Linear blend skinning.
                let w = &self.weights[vertex * self.num_joints..(vertex + 1) * self.num_joints];
                let mut acc = [0.0f64; 3];
                for (joint, weight) in w.iter().enumerate() {
                    if *weight == 0.0 {
                        continue;
                    }
                    let g = &skin[joint * 16..joint * 16 + 16];
                    let x = p[0] as f64;
                    let y = p[1] as f64;
                    let z = p[2] as f64;
                    acc[0] += *weight as f64 * (g[0] * x + g[1] * y + g[2] * z + g[3]);
                    acc[1] += *weight as f64 * (g[4] * x + g[5] * y + g[6] * z + g[7]);
                    acc[2] += *weight as f64 * (g[8] * x + g[9] * y + g[10] * z + g[11]);
                }
                slot[0] = acc[0] as f32 + pose.translation[0];
                slot[1] = acc[1] as f32 + pose.translation[1];
                slot[2] = acc[2] as f32 + pose.translation[2];
            });

        Ok((skeleton, vertices))
    }
}

/// Extract the first `num_betas` components from a `(V, 3, S)` array.
fn truncate_shape_dirs(raw: &npz::NpzArray, num_vertices: usize, num_betas: usize) -> Vec<f32> {
    let data = raw.array.to_f32();
    let components = raw.shape[2];
    let mut out = vec![0.0f32; num_vertices * 3 * num_betas];
    for vertex in 0..num_vertices {
        for axis in 0..3 {
            let src_base = (vertex * 3 + axis) * components;
            let dst_base = (vertex * 3 + axis) * num_betas;
            out[dst_base..dst_base + num_betas]
                .copy_from_slice(&data[src_base..src_base + num_betas]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_joint_model() -> BodyModel {
        BodyModel {
            num_vertices: 2,
            num_joints: 2,
            num_betas: 1,
            parents: vec![-1, 0],
            names: vec!["a".into(), "b".into()],
            v_template: vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            shapedirs: vec![0.0; 2 * 3],
            j_regressor: vec![1.0, 0.0, 0.0, 1.0],
            weights: vec![1.0, 0.0, 1.0, 0.0],
            faces: vec![],
            template_joints: vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            posedirs: None,
            source: PathBuf::from("synthetic"),
        }
    }

    #[test]
    fn neutral_pose_returns_template_joints() {
        let model = two_joint_model();
        let skeleton = model.forward_joints(&Pose::neutral(2, 1));
        assert_eq!(skeleton.num_joints(), 2);
        assert!(skeleton.joint(0)[1].abs() < 1e-6);
        assert!((skeleton.joint(1)[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn translation_moves_every_joint() {
        let model = two_joint_model();
        let mut pose = Pose::neutral(2, 1);
        pose.translation = [1.0, 2.0, 3.0];
        let skeleton = model.forward_joints(&pose);
        assert!((skeleton.joint(1)[0] - 1.0).abs() < 1e-6);
        assert!((skeleton.joint(1)[1] - 3.0).abs() < 1e-6);
        assert!((skeleton.joint(1)[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn rotating_the_root_rotates_children_about_it() {
        let model = two_joint_model();
        let mut pose = Pose::neutral(2, 1);
        // 90 degrees about +Z takes +Y to -X.
        pose.root = [0.0, 0.0, std::f32::consts::FRAC_PI_2];
        let skeleton = model.forward_joints(&pose);
        assert!(
            (skeleton.joint(1)[0] + 1.0).abs() < 1e-5,
            "{:?}",
            skeleton.joint(1)
        );
        assert!(skeleton.joint(1)[1].abs() < 1e-5);
    }
}
