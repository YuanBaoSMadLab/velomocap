//! FlowHMR motion generation: the flow-matching MMDiT, natively.
//!
//! The exported graph is one Euler step (`x_t → x̂_1`). Everything around it is
//! cheap and lives here: the narrowband attention mask, the Euler loop, and the
//! decode from the 491-dim normalised latent back to SMPL-H parameters.
//!
//! Latent layout (491 values per frame), matching `V2MPipeline` with
//! `motion_style = "basic"`:
//!
//! ```text
//!   [  0..  2)  root velocity in XZ                       (normalised)
//!   [  2..  3)  root height Y                             (normalised)
//!   [  3..159)  52 local joint positions × 3              (normalised)
//!   [159..471)  52 local rotations × 6                    (normalised)
//!   [471..487)  16 shape coefficients                     (normalised)
//!   [487..491)  4 foot contact flags                      (normalised)
//! ```
//!
//! Decoding yields `rot6d` (52 joints), `shapes`, `trans` and `foot_contacts`,
//! which is exactly what the SMPL-H forward pass and the exporters need.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use ndarray::{Array3, Array4};
use serde::Deserialize;

use crate::{Backend, Model, SessionConfig};

// --------------------------------------------------------------------------
// Latent layout
// --------------------------------------------------------------------------
pub const NUM_JOINTS: usize = 52;
pub const NUM_SHAPES: usize = 16;
pub const NUM_FOOT_CONTACTS: usize = 4;

pub const ROOT_VEL_XZ: std::ops::Range<usize> = 0..2;
pub const ROOT_POS_Y: std::ops::Range<usize> = 2..3;
pub const LOCAL_JOINTS: std::ops::Range<usize> = 3..159;
pub const LOCAL_ROTS: std::ops::Range<usize> = 159..471;
pub const SHAPES: std::ops::Range<usize> = 471..487;
pub const FOOT_CONTACTS: std::ops::Range<usize> = 487..491;

/// Narrowband attention windows, in frames.
#[derive(Clone, Copy, Debug, Deserialize)]
pub struct Narrowband {
    /// Motion ↔ motion.
    pub mm: i64,
    /// Motion → video context.
    pub mc: i64,
    /// Video context → motion. Negative disables it.
    pub cm: i64,
    /// Video context ↔ video context.
    pub cc: i64,
}

impl Narrowband {
    /// Build the additive `(motion+context)²` attention mask.
    ///
    /// Motion tokens come first, then the context tokens; `-inf` blocks a pair.
    /// The window is measured in **frame offsets inside each block**, exactly as
    /// `build_narrowband_mask` in the reference computes it: the motion token at
    /// frame `i` and the context token at frame `j` are compared by `i` vs `j`,
    /// not by their row and column indices in the combined sequence.
    pub fn mask(&self, motion_len: usize, ctxt_len: usize) -> Vec<f32> {
        let total = motion_len + ctxt_len;
        let mut mask = vec![0.0f32; total * total];

        let mut fill = |row: usize, qn: usize, k0: usize, kn: usize, window: i64| {
            for qi in 0..qn {
                for ki in 0..kn {
                    let offset = qi as i64 - ki as i64;
                    let blocked = window < 0 || offset.abs() > window;
                    mask[(row + qi) * total + k0 + ki] =
                        if blocked { f32::NEG_INFINITY } else { 0.0 };
                }
            }
        };

        fill(0, motion_len, 0, motion_len, self.mm);
        fill(0, motion_len, motion_len, ctxt_len, self.mc);
        fill(motion_len, ctxt_len, 0, motion_len, self.cm);
        fill(motion_len, ctxt_len, motion_len, ctxt_len, self.cc);
        mask
    }
}

/// One `mean` / `std` pair from `assets/motion_stats.json`.
///
/// The file stores per-joint statistics as nested `[joints][dims]` arrays, so
/// values are flattened on load and treated as one long vector.
#[derive(Clone, Debug, Deserialize)]
pub struct Stats {
    #[serde(deserialize_with = "flatten_f32")]
    pub mean: Vec<f32>,
    #[serde(deserialize_with = "flatten_f32")]
    pub std: Vec<f32>,
}

/// Accept either `[1, 2, 3]` or `[[1, 2], [3, 4]]` and flatten it.
fn flatten_f32<'de, D>(deserializer: D) -> std::result::Result<Vec<f32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error;

    fn walk(value: &serde_json::Value, out: &mut Vec<f32>) -> std::result::Result<(), String> {
        match value {
            serde_json::Value::Number(n) => match n.as_f64() {
                Some(v) => {
                    out.push(v as f32);
                    Ok(())
                }
                None => Err(format!("not a number: {n}")),
            },
            serde_json::Value::Array(items) => {
                for item in items {
                    walk(item, out)?;
                }
                Ok(())
            }
            other => Err(format!("unexpected value {other}")),
        }
    }

    let value = serde_json::Value::deserialize(deserializer)?;
    let mut out = Vec::new();
    walk(&value, &mut out).map_err(D::Error::custom)?;
    Ok(out)
}

impl Stats {
    /// Std clamped away from zero, as the reference does.
    fn safe_std(&self) -> Vec<f32> {
        self.std.iter().map(|s| s.max(1e-5)).collect()
    }
}

/// The `assets/motion_stats.json` the pipeline normalises with.
#[derive(Clone, Debug, Deserialize)]
pub struct MotionStats {
    pub smooth_root_vel: Stats,
    pub smooth_root_pos: Stats,
    pub local_joints_positions: Stats,
    pub global_rot_data: Stats,
    pub local_rot_data: Stats,
    pub shapes: Stats,
    pub foot_contacts: Stats,
}

impl MotionStats {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let text = std::fs::read_to_string(path.as_ref())
            .with_context(|| format!("reading {}", path.as_ref().display()))?;
        serde_json::from_str(&text)
            .with_context(|| format!("parsing {}", path.as_ref().display()))
    }
}

#[derive(Clone, Debug)]
pub struct MotionConfig {
    /// Exported MMDiT graph.
    pub weights: PathBuf,
    /// `assets/motion_stats.json`.
    pub stats: PathBuf,
    /// Euler steps; 20 matches the reference default.
    pub steps: usize,
    /// Seed for the initial noise.
    pub seed: u64,
    /// Frames per second the motion is generated at.
    pub fps: f32,
    pub session: SessionConfig,
}

impl Default for MotionConfig {
    fn default() -> Self {
        Self {
            weights: PathBuf::from("models/flowhmr/flowhmr_mmdit.onnx"),
            stats: PathBuf::from("models/flowhmr/motion_stats.json"),
            steps: 20,
            // The reference samples with `seeds=[0]`; anything else changes the
            // clip, so this default is part of the replication.
            seed: 0,
            fps: 30.0,
            session: SessionConfig::default(),
        }
    }
}

/// Decoded SMPL-H motion, ready for the body model and the exporters.
pub struct GeneratedMotion {
    /// `[frames][52][6]` local rotations in 6-D form, root first.
    pub rot6d: Vec<[[f32; 6]; NUM_JOINTS]>,
    /// `[16]` shape coefficients (averaged over the clip, as the reference does).
    pub betas: Vec<f32>,
    /// `[frames][3]` root translation, already offset by the rest pelvis.
    pub trans: Vec<[f32; 3]>,
    /// `[frames][4]` foot contact probabilities.
    pub foot_contacts: Vec<[f32; 4]>,
    /// `[frames][52][3]` local joint positions, for diagnostics.
    pub local_joints: Vec<[[f32; 3]; NUM_JOINTS]>,
}

impl GeneratedMotion {
    pub fn frames(&self) -> usize {
        self.rot6d.len()
    }

    /// Axis-angle poses, `[frames][52][3]`, with the 180° Y rotation the
    /// reference applies before saving.
    pub fn poses_axis_angle(&self) -> Vec<[[f32; 3]; NUM_JOINTS]> {
        let flip = y180();
        self.rot6d
            .iter()
            .map(|frame| {
                let mut out = [[0.0f32; 3]; NUM_JOINTS];
                for (j, six) in frame.iter().enumerate() {
                    let m = flow_rot6d_to_matrix(*six);
                    let m = if j == 0 { mat_mul(&flip, &m) } else { m };
                    out[j] = matrix_to_axis_angle(&m);
                }
                out
            })
            .collect()
    }

    /// Root translations rotated by the same 180° Y rotation.
    pub fn trans_y180(&self) -> Vec<[f32; 3]> {
        self.trans
            .iter()
            .map(|t| [-t[0], t[1], -t[2]])
            .collect()
    }
}

pub struct MotionGenerator {
    model: Model,
    stats: MotionStats,
    narrowband: Narrowband,
    steps: usize,
    seed: u64,
    fps: f32,
    input_dim: usize,
    feature_dim: usize,
    vtxt_dim: usize,
}

impl MotionGenerator {
    pub fn load(
        config: MotionConfig,
        narrowband: Narrowband,
        meta_dims: (usize, usize, usize),
    ) -> Result<Self> {
        let stats = MotionStats::load(&config.stats)?;
        let model = Model::load(&config.weights, &config.session)
            .with_context(|| format!("loading FlowHMR MMDiT {}", config.weights.display()))?;
        Ok(Self {
            model,
            steps: config.steps.max(1),
            seed: config.seed,
            fps: if config.fps > 0.0 { config.fps } else { 30.0 },
            stats,
            narrowband,
            input_dim: meta_dims.0,
            feature_dim: meta_dims.1,
            vtxt_dim: meta_dims.2,
        })
    }

    pub fn backend(&self) -> Backend {
        self.model.backend()
    }

    /// Run exactly one graph step with caller-supplied buffers.
    ///
    /// Exposed for parity harnesses; the normal entry point is [`Self::generate`].
    #[allow(clippy::too_many_arguments)]
    pub fn step_raw(
        &mut self,
        latent: &[f32],
        camera_rows: &[f32],
        tokens: &[f32],
        frames: usize,
        vtxt: &[f32],
        valid: &[u8],
        mask: &[f32],
        ctxt_len: usize,
        t: f32,
    ) -> Result<Vec<f32>> {
        anyhow::ensure!(
            mask.len() == (frames + ctxt_len) * (frames + ctxt_len),
            "mask size"
        );
        self.run_step(
            latent,
            camera_rows,
            tokens,
            frames,
            mask,
            frames + ctxt_len,
            valid,
            vtxt,
            t,
        )
    }

    /// Generate a clip of `frames` from per-frame SAM tokens and camera rows.
    ///
    /// `tokens` is `frames × feature_dim` (body + left hand + right hand
    /// concatenated); `camera_rows` is `frames × 9` (rotation relative to the
    /// first frame, row major) — see [`crate::CameraTrack::relative_rotation_rows`].
    ///
    /// `pelvis_rest` is the SMPL-H rest pelvis after applying `betas`; the
    /// reference subtracts it from the world pelvis to obtain `trans`.
    pub fn generate(
        &mut self,
        tokens: &[f32],
        frames: usize,
        camera_rows: &[f32],
        pelvis_rest: [f32; 3],
    ) -> Result<GeneratedMotion> {
        let latent = self.integrate(tokens, frames, camera_rows)?;
        self.decode_latent(&latent, frames, pelvis_rest)
    }

    /// Run the Euler loop and return the denormalised-but-undecoded latent.
    ///
    /// Split out of [`Self::generate`] because the rest pelvis the decode needs
    /// is derived from the predicted shapes, which are only known once the
    /// latent exists. Callers that do not have a rest pelvis yet use
    /// [`Self::shapes_from_latent`] first.
    pub fn integrate(
        &mut self,
        tokens: &[f32],
        frames: usize,
        camera_rows: &[f32],
    ) -> Result<Vec<f32>> {
        let noise = gaussian(frames * self.input_dim, self.seed);
        self.integrate_from(tokens, frames, camera_rows, &noise)
    }

    /// The same, with caller-supplied noise — the seam a parity harness needs.
    pub fn integrate_from(
        &mut self,
        tokens: &[f32],
        frames: usize,
        camera_rows: &[f32],
        noise: &[f32],
    ) -> Result<Vec<f32>> {
        anyhow::ensure!(frames > 0, "need at least one frame");
        anyhow::ensure!(
            tokens.len() == frames * self.feature_dim,
            "token buffer has {} values, expected {}×{}",
            tokens.len(),
            frames,
            self.feature_dim
        );
        anyhow::ensure!(
            camera_rows.len() == frames * 9,
            "camera buffer has {} values, expected {}×9",
            camera_rows.len(),
            frames
        );
        anyhow::ensure!(
            noise.len() == frames * self.input_dim,
            "noise buffer has {} values, expected {}×{}",
            noise.len(),
            frames,
            self.input_dim
        );

        let mask = self.narrowband.mask(frames, frames);
        let total = frames * 2;
        let valid = vec![1u8; frames];
        let vtxt = vec![0.0f32; self.vtxt_dim];

        let mut latent = noise.to_vec();

        let steps = self.steps;
        let dt = 1.0f32 / steps as f32;
        let t_eps = 1.0f32 / steps as f32;
        for step in 0..steps {
            let t = step as f32 * dt;
            let pred = self.run_step(
                &latent,
                camera_rows,
                tokens,
                frames,
                &mask,
                total,
                &valid,
                &vtxt,
                t,
            )?;
            // pred_type == "x1": the graph predicts x1, so the velocity is
            // (x1 - x_t) / (1 - t).
            let denom = (1.0 - t).max(t_eps);
            for i in 0..latent.len() {
                latent[i] += (pred[i] - latent[i]) / denom * dt;
            }
        }
        Ok(latent)
    }

    /// The clip-averaged shape coefficients, read straight out of the latent.
    ///
    /// `_motion_to_smplh` regresses the rest pelvis from these, so a caller that
    /// wants `pelvis_rest` needs them before the full decode.
    pub fn shapes_from_latent(&self, latent: &[f32], frames: usize) -> Vec<f32> {
        let s = &self.stats;
        let shape_std = s.shapes.safe_std();
        let mut acc = vec![0.0f64; NUM_SHAPES];
        for f in 0..frames {
            let row = &latent[f * self.input_dim..(f + 1) * self.input_dim];
            for i in 0..NUM_SHAPES {
                acc[i] += (row[SHAPES.start + i] * shape_std[i] + s.shapes.mean[i]) as f64;
            }
        }
        acc.iter()
            .map(|v| (*v / frames.max(1) as f64) as f32)
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn run_step(
        &mut self,
        latent: &[f32],
        camera_rows: &[f32],
        tokens: &[f32],
        frames: usize,
        mask: &[f32],
        total: usize,
        valid: &[u8],
        vtxt: &[f32],
        t: f32,
    ) -> Result<Vec<f32>> {
        let x = Array3::from_shape_vec((1, frames, self.input_dim), latent.to_vec())
            .context("building latent tensor")?;
        let cam = Array3::from_shape_vec((1, frames, 9), camera_rows.to_vec())
            .context("building camera tensor")?;
        let feature = Array3::from_shape_vec((1, frames, self.feature_dim), tokens.to_vec())
            .context("building feature tensor")?;
        let vtxt_t = Array3::from_shape_vec((1, 1, self.vtxt_dim), vtxt.to_vec())
            .context("building text tensor")?;
        let timestep = ndarray::Array1::from_shape_vec(1, vec![t]).context("building timestep")?;
        let valid_bool: Vec<bool> = valid.iter().map(|v| *v != 0).collect();
        let valid_t = ndarray::Array2::from_shape_vec((1, frames), valid_bool)
            .context("building validity tensor")?;
        let mask_t = Array4::from_shape_vec((1, 1, total, total), mask.to_vec())
            .context("building attention mask")?;
        let refiner_values = refiner_mask(valid, frames);
        let refiner_t = Array4::from_shape_vec((1, 1, frames, frames), refiner_values)
            .context("building refiner mask")?;

        let session = self.model.session_mut();
        let outputs = session
            .run(ort::inputs![
                "x" => ort::value::Value::from_array(x).context("x tensor")?,
                "cam_R" => ort::value::Value::from_array(cam).context("cam tensor")?,
                "feature" => ort::value::Value::from_array(feature).context("feature tensor")?,
                "vtxt" => ort::value::Value::from_array(vtxt_t).context("vtxt tensor")?,
                "timestep" => ort::value::Value::from_array(timestep).context("timestep tensor")?,
                "ctxt_valid" => ort::value::Value::from_array(valid_t).context("valid tensor")?,
                "attn_mask" => ort::value::Value::from_array(mask_t).context("mask tensor")?,
                "refiner_mask" => ort::value::Value::from_array(refiner_t).context("refiner mask")?,
            ])
            .context("running FlowHMR MMDiT")?;
        let (shape, pred) = outputs[0]
            .try_extract_tensor::<f32>()
            .context("reading motion prediction")?;
        anyhow::ensure!(
            pred.len() == frames * self.input_dim,
            "unexpected prediction shape {shape:?}"
        );
        Ok(pred.to_vec())
    }

    /// Additive self-attention mask for the text refiner.
    ///
    /// A query/key pair is visible only when both tokens are valid; rows with
    /// nothing visible fall back to the diagonal so the softmax never sees an
    /// all-`-inf` row. Computed here because `torch.eye` does not export to a
    /// provider-portable ONNX node.
    pub fn text_refiner_mask(valid: &[u8], length: usize) -> Vec<f32> {
        refiner_mask(valid, length)
    }

    /// Denormalise the latent, matching `V2MPipeline._decode_latent_to_motion`.
    pub fn decode_latent(
        &self,
        latent: &[f32],
        frames: usize,
        pelvis_rest: [f32; 3],
    ) -> Result<GeneratedMotion> {
        let s = &self.stats;
        let dim = self.input_dim;
        let vel_std = s.smooth_root_vel.safe_std();
        let pos_std = s.smooth_root_pos.safe_std();
        let ljp_std = s.local_joints_positions.safe_std();
        let rot_std = s.local_rot_data.safe_std();
        let shape_std = s.shapes.safe_std();
        let fc_std = s.foot_contacts.safe_std();

        let mut vel_xz = Vec::with_capacity(frames);
        let mut pos_y = Vec::with_capacity(frames);
        let mut joints = Vec::with_capacity(frames);
        let mut rots = Vec::with_capacity(frames);
        let mut shapes_acc = vec![0.0f64; NUM_SHAPES];
        let mut contacts = Vec::with_capacity(frames);

        for f in 0..frames {
            let row = &latent[f * dim..(f + 1) * dim];

            vel_xz.push([
                row[ROOT_VEL_XZ.start] * vel_std[0] + s.smooth_root_vel.mean[0],
                row[ROOT_VEL_XZ.start + 1] * vel_std[2] + s.smooth_root_vel.mean[2],
            ]);
            pos_y.push(row[ROOT_POS_Y.start] * pos_std[1] + s.smooth_root_pos.mean[1]);

            let mut ljp = [[0.0f32; 3]; NUM_JOINTS];
            for j in 0..NUM_JOINTS {
                for c in 0..3 {
                    let i = j * 3 + c;
                    ljp[j][c] = row[LOCAL_JOINTS.start + i] * ljp_std[i]
                        + s.local_joints_positions.mean[i];
                }
            }
            joints.push(ljp);

            let mut rot = [[0.0f32; 6]; NUM_JOINTS];
            for j in 0..NUM_JOINTS {
                for c in 0..6 {
                    let i = j * 6 + c;
                    rot[j][c] =
                        row[LOCAL_ROTS.start + i] * rot_std[i] + s.local_rot_data.mean[i];
                }
            }
            rots.push(rot);

            for i in 0..NUM_SHAPES {
                shapes_acc[i] += (row[SHAPES.start + i] * shape_std[i] + s.shapes.mean[i]) as f64;
            }

            let mut fc = [0.0f32; NUM_FOOT_CONTACTS];
            for i in 0..NUM_FOOT_CONTACTS {
                fc[i] = row[FOOT_CONTACTS.start + i] * fc_std[i] + s.foot_contacts.mean[i];
            }
            contacts.push(fc);
        }

        let betas: Vec<f32> = shapes_acc
            .iter()
            .map(|v| (*v / frames as f64) as f32)
            .collect();

        // Root trajectory: XZ integrates the world-frame velocity, Y is absolute.
        let mut root = Vec::with_capacity(frames);
        let mut cursor = [0.0f32; 3];
        for f in 0..frames {
            if f > 0 {
                cursor[0] += vel_xz[f][0] / self.fps;
                cursor[2] += vel_xz[f][1] / self.fps;
            }
            cursor[1] = pos_y[f];
            root.push(cursor);
        }

        // Pelvis world = local pelvis + root offset; `trans` removes the rest pose.
        let mut trans = Vec::with_capacity(frames);
        for f in 0..frames {
            let pelvis = joints[f][0];
            trans.push([
                pelvis[0] + root[f][0] - pelvis_rest[0],
                pelvis[1] - pelvis_rest[1],
                pelvis[2] + root[f][2] - pelvis_rest[2],
            ]);
        }

        Ok(GeneratedMotion {
            rot6d: rots,
            betas,
            trans,
            foot_contacts: contacts,
            local_joints: joints,
        })
    }
}

/// Additive self-attention mask for the text refiner.
///
/// A query/key pair is visible only when both tokens are valid; rows with
/// nothing visible fall back to the diagonal so the softmax never sees an
/// all-`-inf` row. Computed here because `torch.eye` does not export to a
/// provider-portable ONNX node.
pub fn refiner_mask(valid: &[u8], length: usize) -> Vec<f32> {
    let mut mask = vec![0.0f32; length * length];
    for q in 0..length {
        let q_ok = valid.get(q).copied().unwrap_or(1) != 0;
        let any = (0..length).any(|k| q_ok && valid.get(k).copied().unwrap_or(1) != 0);
        for k in 0..length {
            let k_ok = valid.get(k).copied().unwrap_or(1) != 0;
            let visible = if any { q_ok && k_ok } else { q == k };
            mask[q * length + k] = if visible { 0.0 } else { f32::NEG_INFINITY };
        }
    }
    mask
}

// --------------------------------------------------------------------------
// Rotation helpers
// --------------------------------------------------------------------------
// There are two different 6-D rotation layouts in play, and mixing them
// silently folds every limb into a degenerate frame:
//
// * SAM 3D Body (`batchXYZfrom6D`) stores two *consecutive triples* — that is
//   [`rot6d_to_matrix`], and `sam3d_params` depends on it.
// * FlowHMR (`rotation_matrix_to_rot6d` in `flowhmr/core/math/geometry.py`)
//   stores a `3 × 2` matrix flattened row-major, so the two basis columns are
//   interleaved — that is [`flow_rot6d_to_matrix`].

/// 6-D rotation (two consecutive triples) → row-major matrix.
///
/// This is the SAM 3D Body layout, not the motion-model one; see
/// [`flow_rot6d_to_matrix`] for FlowHMR's interleaved columns.
pub fn rot6d_to_matrix(r: [f32; 6]) -> [[f32; 3]; 3] {
    gram_schmidt([r[0], r[1], r[2]], [r[3], r[4], r[5]])
}

/// FlowHMR / SMPL 6-D rotation → row-major matrix.
///
/// The reference writes `cat([col0, col1]).reshape(6)`, i.e. the pairs
/// `(r0, r1)`, `(r2, r3)`, `(r4, r5)` are the rows of a `3 × 2` matrix whose
/// columns are the first two basis vectors.
pub fn flow_rot6d_to_matrix(r: [f32; 6]) -> [[f32; 3]; 3] {
    gram_schmidt([r[0], r[2], r[4]], [r[1], r[3], r[5]])
}

/// Orthonormalise two basis vectors into a rotation matrix.
fn gram_schmidt(a1: [f32; 3], a2: [f32; 3]) -> [[f32; 3]; 3] {
    let b1 = normalise(a1);
    let dot = b1[0] * a2[0] + b1[1] * a2[1] + b1[2] * a2[2];
    let b2 = normalise([
        a2[0] - dot * b1[0],
        a2[1] - dot * b1[1],
        a2[2] - dot * b1[2],
    ]);
    let b3 = [
        b1[1] * b2[2] - b1[2] * b2[1],
        b1[2] * b2[0] - b1[0] * b2[2],
        b1[0] * b2[1] - b1[1] * b2[0],
    ];
    [
        [b1[0], b2[0], b3[0]],
        [b1[1], b2[1], b3[1]],
        [b1[2], b2[2], b3[2]],
    ]
}

fn normalise(v: [f32; 3]) -> [f32; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-12);
    [v[0] / n, v[1] / n, v[2] / n]
}

fn mat_mul(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut out = [[0.0f32; 3]; 3];
    for (y, row) in out.iter_mut().enumerate() {
        for (x, value) in row.iter_mut().enumerate() {
            *value = a[y][0] * b[0][x] + a[y][1] * b[1][x] + a[y][2] * b[2][x];
        }
    }
    out
}

/// 180° rotation about Y.
fn y180() -> [[f32; 3]; 3] {
    [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]
}

/// Row-major rotation matrix → axis-angle vector.
pub fn matrix_to_axis_angle(m: &[[f32; 3]; 3]) -> [f32; 3] {
    let trace = m[0][0] + m[1][1] + m[2][2];
    let cos = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0);
    let angle = cos.acos();
    if angle < 1e-6 {
        return [0.0, 0.0, 0.0];
    }
    let sin = angle.sin();
    if sin.abs() < 1e-6 {
        // Near π: recover the axis from the symmetric part.
        let xx = ((m[0][0] + 1.0) * 0.5).max(0.0).sqrt();
        let yy = ((m[1][1] + 1.0) * 0.5).max(0.0).sqrt();
        let zz = ((m[2][2] + 1.0) * 0.5).max(0.0).sqrt();
        let axis = if xx >= yy && xx >= zz {
            [
                xx,
                (m[0][1] + m[1][0]) / (4.0 * xx.max(1e-9)),
                (m[0][2] + m[2][0]) / (4.0 * xx.max(1e-9)),
            ]
        } else if yy >= zz {
            [
                (m[0][1] + m[1][0]) / (4.0 * yy.max(1e-9)),
                yy,
                (m[1][2] + m[2][1]) / (4.0 * yy.max(1e-9)),
            ]
        } else {
            [
                (m[0][2] + m[2][0]) / (4.0 * zz.max(1e-9)),
                (m[1][2] + m[2][1]) / (4.0 * zz.max(1e-9)),
                zz,
            ]
        };
        let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2])
            .sqrt()
            .max(1e-12);
        return [axis[0] / n * angle, axis[1] / n * angle, axis[2] / n * angle];
    }
    let k = angle / (2.0 * sin);
    [
        (m[2][1] - m[1][2]) * k,
        (m[0][2] - m[2][0]) * k,
        (m[1][0] - m[0][1]) * k,
    ]
}

/// Deterministic standard-normal samples (xorshift + Box–Muller).
///
/// The reference seeds `torch.Generator`, whose stream is not reproducible here;
/// any well-behaved Gaussian draw is an equally valid sample, so this uses a
/// small self-contained generator instead of pulling in a dependency.
pub fn gaussian(count: usize, seed: u64) -> Vec<f32> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
        | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut out = Vec::with_capacity(count);
    while out.len() < count {
        let u1 = next().max(1e-12);
        let u2 = next();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        out.push((r * theta.cos()) as f32);
        if out.len() < count {
            out.push((r * theta.sin()) as f32);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn narrowband_mask_blocks_outside_the_window() {
        let nb = Narrowband { mm: 1, mc: 0, cm: -1, cc: 1 };
        let mask = nb.mask(3, 3);
        assert_eq!(mask.len(), 36);
        // Motion 0 ↔ motion 0 and 1 are inside the window, motion 2 is not.
        assert_eq!(mask[0], 0.0);
        assert_eq!(mask[1], 0.0);
        assert!(mask[2].is_infinite() && mask[2].is_sign_negative());
        // Context → motion is fully disabled.
        for k in 0..3 {
            assert!(mask[3 * 6 + k].is_infinite());
        }
    }

    #[test]
    fn motion_sees_the_context_by_frame_offset() {
        // The reference compares *frame indices inside each block*
        // (`build_narrowband_mask`: `band(motion_len, ctxt_len, mc)` uses
        // `arange(qn)` against `arange(kn)`), so a zero window still lets every
        // motion token see the context token of its own frame. Comparing the
        // combined row and column indices instead leaves only a handful of
        // cells visible — which silently cuts the video out of the model.
        let nb = Narrowband { mm: 0, mc: 0, cm: -1, cc: 0 };
        let mask = nb.mask(4, 4);
        let visible = |q: usize, k: usize| mask[q * 8 + k] == 0.0;
        assert!(visible(0, 4), "motion 0 must see context 0");
        assert!(visible(3, 7), "motion 3 must see context 3");
        assert!(!visible(0, 5), "motion 0 must not see context 1");
        assert!(!visible(0, 0 + 1), "motion→motion window is zero");
        let count = (0..8)
            .flat_map(|q| (0..8).map(move |k| (q, k)))
            .filter(|(q, k)| visible(*q, *k))
            .count();
        // motion→motion 4, motion→context 4, context→context 4, context→motion 0.
        assert_eq!(count, 12);
    }

    #[test]
    fn latent_layout_covers_the_full_width() {
        assert_eq!(FOOT_CONTACTS.end, 491);
        assert_eq!(LOCAL_JOINTS.len(), NUM_JOINTS * 3);
        assert_eq!(LOCAL_ROTS.len(), NUM_JOINTS * 6);
    }

    #[test]
    fn rot6d_round_trip_through_axis_angle() {
        // FlowHMR layout: identity has columns +x and +y interleaved as 3×2.
        let m = flow_rot6d_to_matrix([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        for row in 0..3 {
            for col in 0..3 {
                let want = if row == col { 1.0 } else { 0.0 };
                assert!((m[row][col] - want).abs() < 1e-6, "{m:?}");
            }
        }
        let aa = matrix_to_axis_angle(&m);
        assert!(aa.iter().all(|v| v.abs() < 1e-5), "{aa:?}");

        // 90° about Z: columns (0,1,0) and (-1,0,0), interleaved.
        let m = flow_rot6d_to_matrix([0.0, -1.0, 1.0, 0.0, 0.0, 0.0]);
        let aa = matrix_to_axis_angle(&m);
        assert!((aa[2] - std::f32::consts::FRAC_PI_2).abs() < 1e-4, "{aa:?}");
        assert!(aa[0].abs() < 1e-4 && aa[1].abs() < 1e-4);
    }

    #[test]
    fn gaussian_is_deterministic_and_normalised() {
        let a = gaussian(1000, 7);
        let b = gaussian(1000, 7);
        assert_eq!(a, b);
        let mean = a.iter().sum::<f32>() / a.len() as f32;
        let var = a.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / a.len() as f32;
        assert!(mean.abs() < 0.15, "mean {mean}");
        assert!((var - 1.0).abs() < 0.25, "var {var}");
    }

    #[test]
    fn y180_negates_x_and_z() {
        let flip = y180();
        assert_eq!(flip[0][0], -1.0);
        assert_eq!(flip[1][1], 1.0);
        assert_eq!(flip[2][2], -1.0);
    }
}
