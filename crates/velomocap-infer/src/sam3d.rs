//! SAM 3D Body: the eleven-graph chain, driven natively.
//!
//! The conversion is split because the scripted MHR model contains a loop that
//! torch cannot express in ONNX (see `docs/MODELS.md`), so the reference forward
//! pass becomes this sequence:
//!
//! ```text
//! backbone -> init -> step0 -> pose -> [mhr] -> post
//!                       step1 -> pose -> [mhr] -> post
//!                          ...
//!                       step5 -> pose        (final body token)
//! ```
//!
//! Every graph is batch-1 and every kernel runs inside ONNX Runtime; the only
//! code here is the chain itself plus the cropping, the ray conditioning and the
//! parameter decoding in [`crate::sam3d_params`].
//!
//! One frame yields a 1024-value body token, which the motion generator
//! concatenates with the two hand tokens to condition FlowHMR.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use ndarray::{Array2, Array3, Array4};

use crate::sam3d_params::{DecodedParams, ParamsTables};
use crate::{Backend, Model, SessionConfig};

/// Side length of the per-person crop the backbone expects.
pub const IMAGE_SIZE: usize = 512;
/// Stride of the backbone patch grid.
pub const PATCH_STRIDE: usize = 16;
/// Patch grid side.
pub const PATCH_GRID: usize = IMAGE_SIZE / PATCH_STRIDE;
/// Image tokens fed to the decoder (`PATCH_GRID²`).
pub const PATCHES: usize = PATCH_GRID * PATCH_GRID;
/// Width of one conditioning token.
pub const TOKEN_DIM: usize = 1024;
/// Number of tokens the decoder carries.
pub const TOKENS: usize = 145;
/// Number of decoder layers (one graph each).
pub const LAYERS: usize = 6;
/// MHR vertices per frame.
pub const VERTICES: usize = 18439;
/// MHR joints per frame.
pub const JOINTS: usize = 127;
/// `[x, y, z, qx, qy, qz, qw, ?]` per joint.
pub const SKEL_STATE: usize = 8;

/// Where the hand box token sits in the decoder's token sequence.
pub const HAND_DET_TOKEN: usize = 3;
/// Range of the 2-D keypoint tokens (`kps_start..kps_end`).
pub const KEYPOINT_TOKENS: std::ops::Range<usize> = 5..75;
/// Range of the 3-D keypoint tokens, the same width as [`KEYPOINT_TOKENS`].
pub const KEYPOINT3D_TOKENS: std::ops::Range<usize> = 75..145;

/// The box SAM is run on, plus the camera it was seen with.
#[derive(Clone, Debug)]
pub struct Sam3dInput {
    /// `IMAGE_SIZE × IMAGE_SIZE × 3` RGB in `[0, 1]`, row-major.
    pub crop: Vec<f32>,
    /// Box centre in original-image pixels.
    pub bbox_center: [f32; 2],
    /// Box size in original-image pixels, already expanded by 1.25.
    pub bbox_scale: [f32; 2],
    /// Original image size in pixels, as `ori_img_size` is fed to the graph.
    pub image_size: [f32; 2],
    /// Pinhole intrinsics of the original image.
    pub cam_int: [[f32; 3]; 3],
}

/// The fixed FOV the reference pipeline assumes without an FOV estimator.
///
/// `mocap/flow_camera.py` and `mocap/engines.py` both use `hypot(w, h)` as the
/// focal length with the principal point at the image centre; SAM itself is
/// trained to accept exactly that.
pub fn default_intrinsics(width: usize, height: usize) -> [[f32; 3]; 3] {
    let (width, height) = (width.max(1) as f32, height.max(1) as f32);
    let focal = (width * width + height * height).sqrt();
    [
        [focal, 0.0, width * 0.5],
        [0.0, focal, height * 0.5],
        [0.0, 0.0, 1.0],
    ]
}

impl Sam3dInput {
    /// Build the crop for an axis-aligned person box (`x0, y0, x1, y1`).
    pub fn from_box(
        frame: &[u8],
        width: usize,
        height: usize,
        box_: [f32; 4],
        cam_int: [[f32; 3]; 3],
    ) -> Result<Self> {
        // `GetBBoxCenterScale(padding=1.25)` then `TopdownAffine`, which expands
        // the box to a 0.75 width/height ratio and then to the square model
        // input. Skipping that expansion squeezes a standing person into a
        // narrow crop and the network answers with a short, far-away body
        // (1.06-1.54 m at 7.5-9.6 m instead of 1.64-1.73 m at 3.5-5.4 m).
        let padding = 1.25;
        let center = [(box_[0] + box_[2]) * 0.5, (box_[1] + box_[3]) * 0.5];
        let mut scale = [
            (box_[2] - box_[0]).abs().max(1.0) * padding,
            (box_[3] - box_[1]).abs().max(1.0) * padding,
        ];
        // fix_aspect_ratio(width / height = 0.75)
        if scale[0] > scale[1] * 0.75 {
            scale[1] = scale[0] / 0.75;
        } else {
            scale[0] = scale[1] * 0.75;
        }
        // ... and then to the model's square input.
        if scale[0] > scale[1] {
            scale[0] = scale[1];
        } else {
            scale[1] = scale[0];
        }
        let mut input = Self::from_frame(frame, width, height, center, scale, cam_int)?;
        input.image_size = [width as f32, height as f32];
        Ok(input)
    }
}

impl Sam3dInput {
    /// Crop `frame` (RGB, `width × height`) around a person box.
    ///
    /// Crop index `u` looks at original `x = centre + (u - 256) · size / 512`,
    /// sampled bilinearly — the same mapping [`Self::ray_condition`] reports to
    /// the network, so the pixels and the rays cannot drift apart.
    pub fn from_frame(
        frame: &[u8],
        width: usize,
        height: usize,
        bbox_center: [f32; 2],
        bbox_scale: [f32; 2],
        cam_int: [[f32; 3]; 3],
    ) -> Result<Self> {
        anyhow::ensure!(
            frame.len() == width * height * 3,
            "frame has {} bytes, expected {}",
            frame.len(),
            width * height * 3
        );
        let size = IMAGE_SIZE as f32;
        let step_x = bbox_scale[0] / size;
        let step_y = bbox_scale[1] / size;
        let half = size * 0.5;

        let mut crop = vec![0.0f32; IMAGE_SIZE * IMAGE_SIZE * 3];
        for row in 0..IMAGE_SIZE {
            let source_y = bbox_center[1] + (row as f32 - half) * step_y;
            for column in 0..IMAGE_SIZE {
                let source_x = bbox_center[0] + (column as f32 - half) * step_x;
                let at = (row * IMAGE_SIZE + column) * 3;
                for channel in 0..3 {
                    crop[at + channel] =
                        sample(frame, width, height, source_x, source_y, channel);
                }
            }
        }
        Ok(Self {
            crop,
            bbox_center,
            bbox_scale,
            image_size: [width as f32, height as f32],
            cam_int,
        })
    }

    /// Full-to-crop affine, as `get_warp_matrix` builds it for an unrotated box.
    ///
    /// The graph's ray conditioning divides crop indices by this matrix, so crop
    /// index `u` maps back to original `x = (u - a02) / a00`.
    pub fn affine(&self) -> [[f32; 3]; 3] {
        let sx = IMAGE_SIZE as f32 / self.bbox_scale[0].max(1e-6);
        let sy = IMAGE_SIZE as f32 / self.bbox_scale[1].max(1e-6);
        let half = IMAGE_SIZE as f32 * 0.5;
        [
            [sx, 0.0, half - self.bbox_center[0] * sx],
            [0.0, sy, half - self.bbox_center[1] * sy],
            [0.0, 0.0, 1.0],
        ]
    }

    /// Per-pixel ray conditioning `[2, H, W]`, as `get_ray_condition` computes it.
    pub fn ray_condition(&self) -> (Vec<f32>, Vec<f32>) {
        let affine = self.affine();
        let (a00, a02) = (affine[0][0], affine[0][2]);
        let (a11, a12) = (affine[1][1], affine[1][2]);
        let focal_x = self.cam_int[0][0].max(1e-12);
        let focal_y = self.cam_int[1][1].max(1e-12);
        let mut x = vec![0.0f32; IMAGE_SIZE * IMAGE_SIZE];
        let mut y = vec![0.0f32; IMAGE_SIZE * IMAGE_SIZE];
        for row in 0..IMAGE_SIZE {
            // Crop index -> original pixel -> NDC around the principal point.
            let oy = (row as f32 - a12) / a11;
            let ny = (oy - self.cam_int[1][2]) / focal_y;
            for column in 0..IMAGE_SIZE {
                let ox = (column as f32 - a02) / a00;
                x[row * IMAGE_SIZE + column] = (ox - self.cam_int[0][2]) / focal_x;
                y[row * IMAGE_SIZE + column] = ny;
            }
        }
        (x, y)
    }

    /// Camera translation `(tx, ty, tz)` for the predicted `(s, tx, ty)`.
    ///
    /// This is `PerspectiveHead.perspective_projection`: `tz ≈ 2f / (s · box)`,
    /// and the box centre's offset from the principal point becomes `cx, cy`.
    /// Adding it to the body-space vertices puts them in camera coordinates
    /// (`x` right, `y` down, `z` forward, metres), the same frame the original
    /// application places its actors in.
    pub fn camera_translation(&self, pred_cam: [f32; 3]) -> [f32; 3] {
        let (s, tx, ty) = (-pred_cam[0], pred_cam[1], -pred_cam[2]);
        let bs = self.bbox_scale[0] * s + 1e-8;
        let focal = self.cam_int[0][0];
        let tz = 2.0 * focal / bs;
        let cx = 2.0 * (self.bbox_center[0] - self.image_size[0] * 0.5) / bs;
        let cy = 2.0 * (self.bbox_center[1] - self.image_size[1] * 0.5) / bs;
        [tx + cx, ty + cy, tz]
    }
}

fn sample(frame: &[u8], width: usize, height: usize, x: f32, y: f32, channel: usize) -> f32 {
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let (x0, y0) = (x0 as isize, y0 as isize);
    let fetch = |x: isize, y: isize| -> f32 {
        if x < 0 || y < 0 || x >= width as isize || y >= height as isize {
            return 0.0;
        }
        let at = (y as usize * width + x as usize) * 3 + channel;
        frame[at] as f32 / 255.0
    };
    let top = fetch(x0, y0) * (1.0 - fx) + fetch(x0 + 1, y0) * fx;
    let bottom = fetch(x0, y0 + 1) * (1.0 - fx) + fetch(x0 + 1, y0 + 1) * fx;
    top * (1.0 - fy) + bottom * fy
}

/// Everything one SAM forward pass produces for a frame.
#[derive(Clone, Debug)]
pub struct Sam3dFrame {
    /// The 1024-value token FlowHMR conditions on (body branch).
    pub body_token: Vec<f32>,
    /// MHR mesh, `VERTICES × 3`, body space (origin near the pelvis).
    pub verts: Vec<f32>,
    /// `JOINTS × 8` joint coordinates plus quaternions.
    pub skel_state: Vec<f32>,
    /// The 204 decoded MHR parameters (136 pose + 68 scales).
    pub model_params: Vec<f32>,
    /// Predicted camera `(s, tx, ty)` for the final frame.
    pub pred_cam: [f32; 3],
    /// Body-space origin in camera coordinates (metres).
    pub cam_t: [f32; 3],
    /// The `pred` vector those parameters were decoded from, for diagnostics.
    pub pred: Vec<f32>,
    /// `1024`: token 0 of the left-hand decoder call, when the backend runs the
    /// hands (the TorchScript module does; the ONNX chain leaves it empty).
    pub left_token: Vec<f32>,
    /// `1024`: token 0 of the right-hand decoder call, same rule.
    pub right_token: Vec<f32>,
}

impl Sam3dFrame {
    /// Write `verts + cam_t` as y-up world coordinates (`x` right, `y` up,
    /// `z` forward), the frame the viewport and the floor use.
    ///
    /// MHR runs in centimetres; the reference divides by 100 right after the
    /// model call (`MHRHead.mhr_forward`), so the placement happens in metres.
    pub fn world_vertices(&self) -> Vec<f32> {
        translate_y_up(&self.verts, self.cam_t)
    }

    /// The same for the 127 joints carried in `skel_state`.
    pub fn world_joints(&self) -> Vec<f32> {
        let joints: Vec<f32> = self
            .skel_state
            .chunks_exact(SKEL_STATE)
            .flat_map(|joint| [joint[0], joint[1], joint[2]])
            .collect();
        translate_y_up(&joints, self.cam_t)
    }
}

/// Camera space (`y` down, centimetres) to y-up world (metres).
///
/// `MHRHead.forward` flips both `y` and `z` to move from MHR's y-up body frame
/// into the camera frame, then adds `cam_t`; presenting that as y-up world only
/// negates the camera's y once more.
fn translate_y_up(points: &[f32], cam_t: [f32; 3]) -> Vec<f32> {
    points
        .chunks_exact(3)
        .flat_map(|point| {
            [
                point[0] * 0.01 + cam_t[0],
                point[1] * 0.01 - cam_t[1],
                -point[2] * 0.01 + cam_t[2],
            ]
        })
        .collect()
}

pub struct Sam3dBody {
    backbone: Model,
    init: Model,
    steps: Vec<Model>,
    pose: Model,
    mhr: Model,
    post: Model,
    params: ParamsTables,
    backend: Backend,
}

impl Sam3dBody {
    /// Load the chain from `dir` and the parameter tables from `params_dir`.
    pub fn load(
        dir: impl AsRef<Path>,
        params_dir: impl AsRef<Path>,
        config: &SessionConfig,
    ) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let load = |name: &str| -> Result<Model> {
            let path = dir.join(name);
            Model::load(&path, config).with_context(|| format!("loading {}", path.display()))
        };
        let params = ParamsTables::load(params_dir.as_ref())?;
        let backbone = load("sam3d_backbone.onnx")?;
        let backend = backbone.backend();
        let mut steps = Vec::with_capacity(LAYERS);
        for index in 0..LAYERS {
            steps.push(load(&format!("sam3d_step{index}.onnx"))?);
        }
        Ok(Self {
            backbone,
            init: load("sam3d_init.onnx")?,
            steps,
            pose: load("sam3d_pose.onnx")?,
            mhr: load("mhr_model.onnx")?,
            post: load("sam3d_post.onnx")?,
            params,
            backend,
        })
    }

    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Path the parameter tables came from, for diagnostics.
    pub fn params_source(&self) -> &Path {
        self.params.source()
    }

    /// Run the whole chain for one frame.
    pub fn run(&mut self, input: &Sam3dInput) -> Result<Sam3dFrame> {
        anyhow::ensure!(
            input.crop.len() == IMAGE_SIZE * IMAGE_SIZE * 3,
            "crop has {} values, expected {}",
            input.crop.len(),
            IMAGE_SIZE * IMAGE_SIZE * 3
        );

        // 1. Backbone: normalized image + per-pixel rays -> token features.
        let (ray_x, ray_y) = input.ray_condition();
        let image = Array4::from_shape_vec((1, 3, IMAGE_SIZE, IMAGE_SIZE), to_chw(&input.crop))
            .context("building the backbone image tensor")?;
        let mut rays = Vec::with_capacity(2 * PATCHES * 256);
        rays.extend_from_slice(&ray_x);
        rays.extend_from_slice(&ray_y);
        let ray = Array4::from_shape_vec((1, 2, IMAGE_SIZE, IMAGE_SIZE), rays)
            .context("building the ray conditioning tensor")?;
        // Scoped so the sessions it borrows are free again for the decoder loop.
        let (embeddings_dims, embeddings, image_seq, augment_seq) = {
            let outputs = self
                .backbone
                .session_mut()
                .run(ort::inputs![
                    "image" => ort::value::Value::from_array(image).context("image tensor")?,
                    "ray_cond" => ort::value::Value::from_array(ray).context("ray tensor")?,
                ])
                .context("SAM backbone")?;
            (
                tensor(&outputs["embeddings"])?.0,
                tensor(&outputs["embeddings"])?.1,
                tensor(&outputs["image_seq"])?.1,
                tensor(&outputs["augment_seq"])?.1,
            )
        };
        // Taken from the graph rather than hardcoded, so a re-export cannot
        // silently disagree with the buffers built below.
        let feature_dim = image_seq.len() / PATCHES;

        // 2. Init: box, camera and the (always invalid) prompt -> tokens.
        let affine = input.affine();
        let center = Array2::from_shape_vec((1, 2), input.bbox_center.to_vec())
            .context("bbox_center tensor")?;
        let scale =
            Array2::from_shape_vec((1, 2), input.bbox_scale.to_vec()).context("bbox_scale tensor")?;
        let intrinsics = Array3::from_shape_vec((1, 3, 3), flatten(&input.cam_int))
            .context("cam_int tensor")?;
        // The body branch has no keypoint prompt: the reference feeds one point
        // labelled -2, which the prompt encoder replaces with its "invalid"
        // embedding.
        let keypoints = Array3::from_shape_vec((1, 1, 3), vec![0.0f32, 0.0, -2.0])
            .context("keypoints tensor")?;
        let (mut tokens, mut token_augment) = {
            let outputs = self
                .init
                .session_mut()
                .run(ort::inputs![
                    "bbox_center" => ort::value::Value::from_array(center).context("center tensor")?,
                    "bbox_scale" => ort::value::Value::from_array(scale).context("scale tensor")?,
                    "cam_int" => ort::value::Value::from_array(intrinsics).context("intrinsics tensor")?,
                    "keypoints" => ort::value::Value::from_array(keypoints).context("keypoints tensor")?,
                ])
                .context("SAM init")?;
            (tensor(&outputs["tokens"])?.1, tensor(&outputs["token_augment"])?.1)
        };
        let mut image_seq = image_seq;

        // 3. Six decoder layers, each but the last followed by a full
        //    pose -> MHR -> reproject round trip that refreshes the keypoint
        //    tokens the next layer reads.
        let mut final_pose: Option<(Vec<f32>, [f32; 3], Vec<f32>)> = None;
        for index in 0..LAYERS {
            let (next_tokens, next_augment, next_image_seq) = {
                let outputs = self.steps[index]
                    .session_mut()
                    .run(ort::inputs![
                        "tokens" => value3(&tokens, TOKENS, TOKEN_DIM)?,
                        "token_augment" => value3(&token_augment, TOKENS, TOKEN_DIM)?,
                        "image_seq" => value3(&image_seq, PATCHES, feature_dim)?,
                        "augment_seq" => value3(&augment_seq, PATCHES, feature_dim)?,
                    ])
                    .context("SAM decoder layer")?;
                (
                    tensor(&outputs["tokens_next"])?.1,
                    tensor(&outputs["token_augment_next"])?.1,
                    tensor(&outputs["image_seq_next"])?.1,
                )
            };
            tokens = next_tokens;
            token_augment = next_augment;
            image_seq = next_image_seq;

            let (normed, pred, pred_cam) = self.pose(&tokens)?;
            if index == LAYERS - 1 {
                final_pose = Some((normed, pred_cam, pred));
                break;
            }

            let decoded = self.params.decode(&pred)?;
            let (verts, skel_state) = self.mhr(&decoded)?;
            let (next_tokens, next_augment) = self.post(
                &tokens,
                &token_augment,
                &embeddings,
                &embeddings_dims,
                &verts,
                &skel_state,
                pred_cam,
                &affine,
                input,
            )?;
            tokens = next_tokens;
            token_augment = next_augment;
        }

        let (normed, pred_cam, pred) = final_pose.context("decoder produced no final token")?;
        let body_token = normed
            .get(..TOKEN_DIM)
            .context("final token is too short")?
            .to_vec();

        // The loop stops before the last round trip, so the final frame's mesh
        // has to be evaluated separately.
        let decoded = self.params.decode(&pred)?;
        let (verts, skel_state) = self.mhr(&decoded)?;
        Ok(Sam3dFrame {
            body_token,
            verts,
            skel_state,
            model_params: decoded.model_params,
            pred_cam,
            cam_t: input.camera_translation(pred_cam),
            pred,
            // The ONNX chain stops at the body branch; the hand rows of the
            // motion feature stay zero, which is what the reference does for
            // frames it does not sample hands on.
            left_token: Vec::new(),
            right_token: Vec::new(),
        })
    }

    /// `sam3d_pose.onnx`: normalized tokens, the raw 519-value `pred` and the
    /// camera estimate.
    fn pose(&mut self, tokens: &[f32]) -> Result<(Vec<f32>, Vec<f32>, [f32; 3])> {
        let outputs = self
            .pose
            .session_mut()
            .run(ort::inputs!["tokens" => value3(tokens, TOKENS, TOKEN_DIM)?])
            .context("SAM pose head")?;
        let normed = tensor(&outputs["normed"])?.1;
        let pred = tensor(&outputs["pred"])?.1;
        let pred_cam = tensor(&outputs["pred_cam"])?.1;
        anyhow::ensure!(pred_cam.len() >= 3, "pred_cam has {} values", pred_cam.len());
        Ok((normed, pred, [pred_cam[0], pred_cam[1], pred_cam[2]]))
    }

    fn mhr(&mut self, decoded: &DecodedParams) -> Result<(Vec<f32>, Vec<f32>)> {
        let identity = Array2::from_shape_vec((1, decoded.identity.len()), decoded.identity.clone())
            .context("identity tensor")?;
        let params = Array2::from_shape_vec(
            (1, decoded.model_params.len()),
            decoded.model_params.clone(),
        )
        .context("model_params tensor")?;
        let face = Array2::from_shape_vec((1, decoded.face.len()), decoded.face.clone())
            .context("face tensor")?;
        let outputs = self
            .mhr
            .session_mut()
            .run(ort::inputs![
                "identity" => ort::value::Value::from_array(identity).context("identity value")?,
                "model_params" => ort::value::Value::from_array(params).context("params value")?,
                "face" => ort::value::Value::from_array(face).context("face value")?,
            ])
            .context("scripted MHR")?;
        Ok((tensor(&outputs["verts"])?.1, tensor(&outputs["skel_state"])?.1))
    }

    /// `sam3d_post.onnx`: MHR output -> refreshed keypoint tokens.
    #[allow(clippy::too_many_arguments)]
    fn post(
        &mut self,
        tokens: &[f32],
        token_augment: &[f32],
        embeddings: &[f32],
        embeddings_dims: &[usize],
        verts: &[f32],
        skel_state: &[f32],
        pred_cam: [f32; 3],
        affine: &[[f32; 3]; 3],
        input: &Sam3dInput,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let channels = *embeddings_dims.get(1).context("embeddings rank")?;
        let outputs = self
            .post
            .session_mut()
            .run(ort::inputs![
                "tokens" => value3(tokens, TOKENS, TOKEN_DIM)?,
                "token_augment" => value3(token_augment, TOKENS, TOKEN_DIM)?,
                "embeddings" => ort::value::Value::from_array(
                    Array4::from_shape_vec((1, channels, PATCH_GRID, PATCH_GRID), embeddings.to_vec())
                        .context("embeddings tensor")?
                ).context("embeddings value")?,
                "verts" => value3(verts, VERTICES, 3)?,
                "skel_state" => value3(skel_state, JOINTS, SKEL_STATE)?,
                "pred_cam" => value2(&pred_cam, 1, 3)?,
                "affine_trans" => value3(&flatten(affine), 3, 3)?,
                "bbox_center" => value2(&input.bbox_center, 1, 2)?,
                "bbox_scale" => value2(&input.bbox_scale, 1, 2)?,
                "cam_int" => value3(&flatten(&input.cam_int), 3, 3)?,
            ])
            .context("SAM post head")?;
        Ok((
            tensor(&outputs["tokens_next"])?.1,
            tensor(&outputs["token_augment_next"])?.1,
        ))
    }
}

/// Read an `f32` output as `(dims, values)`.
fn tensor(value: &ort::value::DynValue) -> Result<(Vec<usize>, Vec<f32>)> {
    let (shape, data) = value
        .try_extract_tensor::<f32>()
        .context("reading an f32 output tensor")?;
    let dims: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
    Ok((dims, data.to_vec()))
}

fn value2(values: &[f32], rows: usize, cols: usize) -> Result<ort::value::DynValue> {
    anyhow::ensure!(
        values.len() == rows * cols,
        "expected {rows}×{cols} values, got {}",
        values.len()
    );
    Ok(ort::value::Value::from_array(
        Array2::from_shape_vec((rows, cols), values.to_vec()).context("shaping a 2-D tensor")?,
    )
    .context("creating a 2-D tensor")?
    .into_dyn())
}

fn value3(values: &[f32], rows: usize, cols: usize) -> Result<ort::value::DynValue> {
    anyhow::ensure!(
        values.len() == rows * cols,
        "expected 1×{rows}×{cols} values, got {}",
        values.len()
    );
    Ok(ort::value::Value::from_array(
        Array3::from_shape_vec((1, rows, cols), values.to_vec()).context("shaping a 3-D tensor")?,
    )
    .context("creating a 3-D tensor")?
    .into_dyn())
}

/// `[H, W, C]` to `[C, H, W]`.
fn to_chw(hwc: &[f32]) -> Vec<f32> {
    let plane = IMAGE_SIZE * IMAGE_SIZE;
    let mut chw = vec![0.0f32; plane * 3];
    for pixel in 0..plane {
        for channel in 0..3 {
            chw[channel * plane + pixel] = hwc[pixel * 3 + channel];
        }
    }
    chw
}

fn flatten(matrix: &[[f32; 3]; 3]) -> Vec<f32> {
    matrix.iter().flatten().copied().collect()
}

/// Directory holding the chain, given the models root.
pub fn chain_dir(models_root: impl AsRef<Path>) -> PathBuf {
    models_root.as_ref().join("sam3d")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repo_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("workspace root")
            .to_path_buf()
    }

    fn models_dir() -> PathBuf {
        repo_root().join("models/flowhmr/sam")
    }

    /// Where the exported chain lives: an override first, then the build output,
    /// then the shipped model directory.
    fn stage_dir() -> Option<PathBuf> {
        if let Some(dir) = std::env::var_os("VELOMOCAP_SAM3D_DIR") {
            let dir = PathBuf::from(dir);
            if dir.join("sam3d_backbone.onnx").is_file() {
                return Some(dir);
            }
        }
        [
            repo_root().parent().expect("parent of the workspace").join("export/sam3d"),
            repo_root().join("models/sam3d"),
        ]
        .into_iter()
        .find(|candidate| candidate.join("sam3d_backbone.onnx").is_file())
    }

    /// The repo ships ONNX Runtime under `runtime/`; tests are not launched from
    /// there, so point the loader at it explicitly. A stray 1.x `onnxruntime.dll`
    /// on `PATH` is what `find_runtime_library` would otherwise return, and the
    /// `ort` bindings refuse anything older than 1.31.
    fn ensure_runtime() {
        let root = repo_root().join("runtime/onnxruntime");
        if let Ok(entries) = std::fs::read_dir(&root) {
            for entry in entries.flatten() {
                let candidate = entry.path().join("lib/onnxruntime.dll");
                if candidate.is_file() {
                    std::env::set_var("VELOMOCAP_ORT_DYLIB", &candidate);
                    return;
                }
            }
        }
    }

    #[test]
    fn affine_and_rays_pass_through_the_box_centre() {
        let input = Sam3dInput {
            crop: vec![0.0; IMAGE_SIZE * IMAGE_SIZE * 3],
            bbox_center: [640.0, 400.0],
            bbox_scale: [250.0, 375.0],
            image_size: [1280.0, 720.0],
            cam_int: [[1100.0, 0.0, 640.0], [0.0, 1100.0, 360.0], [0.0, 0.0, 1.0]],
        };
        let affine = input.affine();
        assert!((affine[0][0] - IMAGE_SIZE as f32 / 250.0).abs() < 1e-5);
        let (x, y) = input.ray_condition();
        // Crop index 256 is the box centre, so the ray leaves from there: the
        // principal point is at y = 360 while the box centre is at 400.
        let centre = 256 * IMAGE_SIZE + 256;
        assert!(x[centre].abs() < 1e-5, "{}", x[centre]);
        assert!((y[centre] - 40.0 / 1100.0).abs() < 1e-5, "{}", y[centre]);
        // One quarter of the crop to the right is 62.5 image pixels, which is
        // 62.5 / 1100 of the focal length in NDC.
        let quarter = 256 * IMAGE_SIZE + 384;
        assert!(
            (x[quarter] - 62.5 / 1100.0).abs() < 1e-5,
            "affine {affine:?} ray {}",
            x[quarter]
        );
    }

    #[test]
    fn crop_samples_the_box_centre() {
        let (width, height) = (64, 64);
        let mut frame = vec![0u8; width * height * 3];
        let at = (32 * width + 32) * 3;
        frame[at] = 255;
        frame[at + 1] = 255;
        frame[at + 2] = 255;
        let input = Sam3dInput::from_frame(
            &frame,
            width,
            height,
            [32.0, 32.0],
            [16.0, 16.0],
            [[30.0, 0.0, 32.0], [0.0, 30.0, 32.0], [0.0, 0.0, 1.0]],
        )
        .unwrap();
        let centre = (256 * IMAGE_SIZE + 256) * 3;
        assert!(input.crop[centre] > 0.9, "{}", input.crop[centre]);
        assert!(input.crop[0] < 0.01);
    }

    /// The integration gate: the Rust chain against one dumped reference frame.
    ///
    /// `tools/onnx/sam3d_split.py` writes `sam3d_chain_ref.bin`, holding the crop
    /// the torch reference saw plus everything it produced. Feeding that crop
    /// through the ONNX chain has to land on the same numbers; if it does not,
    /// either a graph changed or the parameter decoder drifted.
    #[test]
    fn chain_reproduces_the_reference_frame() {
        let Some(dir) = stage_dir() else {
            eprintln!("skipping: no exported SAM 3D chain");
            return;
        };
        let fixture = [dir.join("sam3d_chain_ref.json"), models_dir().join("sam3d_chain_ref.json")]
            .into_iter()
            .find(|candidate| candidate.is_file());
        let Some(fixture) = fixture else {
            eprintln!("skipping: {} has no chain reference", dir.display());
            return;
        };
        let reference_dir = fixture.parent().expect("fixture directory").to_path_buf();
        ensure_runtime();
        // Measure against full-precision kernels: TensorFloat-32 (the default)
        // costs about 1e-3 on the token and would hide a real porting mistake.
        std::env::set_var("VELOMOCAP_TF32", "0");

        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&fixture).unwrap()).unwrap();
        let raw = std::fs::read(reference_dir.join("sam3d_chain_ref.bin")).unwrap();
        let floats: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let get = |name: &str| -> Vec<f32> {
            let entry = &meta["tensors"][name];
            let offset = entry["offset"].as_u64().expect("offset") as usize;
            let len = entry["len"].as_u64().expect("len") as usize;
            floats[offset..offset + len].to_vec()
        };
        let pair = |name: &str, index: usize| -> f32 {
            meta[name][index].as_f64().expect("number") as f32
        };
        let row3 = |name: &str, row: usize| -> [f32; 3] {
            let entries = meta[name][row].as_array().expect("row");
            [0, 1, 2].map(|column| entries[column].as_f64().expect("number") as f32)
        };

        let input = Sam3dInput {
            crop: get("crop"),
            bbox_center: [pair("bbox_center", 0), pair("bbox_center", 1)],
            bbox_scale: [pair("bbox_scale", 0), pair("bbox_scale", 1)],
            image_size: [1280.0, 720.0],
            cam_int: [row3("cam_int", 0), row3("cam_int", 1), row3("cam_int", 2)],
        };

        // The rays are computed on the Rust side, so check them against the
        // reference arithmetic before trusting anything downstream.
        let (ray_x, ray_y) = input.ray_condition();
        let want_ray = get("ray");
        let plane = IMAGE_SIZE * IMAGE_SIZE;
        let mut ray_error = 0.0f32;
        for index in 0..plane {
            ray_error = ray_error.max((ray_x[index] - want_ray[index]).abs());
            ray_error = ray_error.max((ray_y[index] - want_ray[plane + index]).abs());
        }
        eprintln!("ray condition max abs {ray_error:.3e}");
        assert!(ray_error < 1e-6, "ray condition off by {ray_error:.3e}");

        let params_dir = if dir.join("sam3d_params.json").is_file() {
            dir.clone()
        } else {
            repo_root().join("models/flowhmr/sam")
        };
        let config = SessionConfig {
            gpu: std::env::var("VELOMOCAP_CPU_ONLY").is_err(),
            ..SessionConfig::default()
        };
        let mut body = Sam3dBody::load(&dir, &params_dir, &config).unwrap();
        eprintln!("backend {:?}, params from {}", body.backend(), body.params_source().display());
        let frame = body.run(&input).unwrap();

        let report = |label: &str, got: &[f32], want: Vec<f32>| -> f32 {
            assert_eq!(got.len(), want.len(), "{label} length");
            let mut worst = 0.0f32;
            let mut total = 0.0f64;
            for (a, b) in got.iter().zip(want.iter()) {
                let diff = (a - b).abs();
                worst = worst.max(diff);
                total += diff as f64;
            }
            eprintln!(
                "  {label:14} max abs {worst:.3e}  mean {:.3e}",
                total / got.len() as f64
            );
            worst
        };

        let token = report("body_token", &frame.body_token, get("body_token"));
        let pred = report("pred", &frame.pred, get("pred"));
        let verts = report("verts", &frame.verts, get("verts"));
        let skel = report("skel_state", &frame.skel_state, get("skel_state"));
        let params = report("model_params", &frame.model_params, get("model_params"));

        assert!(token < 2e-3, "body_token off by {token:.3e}");
        assert!(pred < 5e-3, "pred off by {pred:.3e}");
        assert!(verts < 5e-2, "verts off by {verts:.3e}");
        assert!(skel < 5e-3, "skel_state off by {skel:.3e}");
        assert!(params < 5e-4, "model_params off by {params:.3e}");
    }
}
