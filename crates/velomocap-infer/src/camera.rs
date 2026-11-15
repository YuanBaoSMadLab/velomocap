//! VGGT-Omega camera estimation, running the exported ONNX graph.
//!
//! VGGT-Omega is a 1.1B-parameter alternating-attention transformer that maps a
//! clip of frames to a 9-D pose encoding per frame: translation (3), rotation
//! quaternion (4, scalar last) and vertical/horizontal field of view (2).
//!
//! The exported graph is shape-dynamic in the frame count and in the image size,
//! but the rotation positional embedding depends on the patch grid, so the
//! caller supplies it as a table (see [`rope_table`]). Everything else — image
//! normalisation, the DINOv3 patch trunk, the 24 frame blocks, the 24
//! inter-frame blocks and the camera head — lives in the graph.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use ndarray::{Array2, Array5};

use crate::{Backend, Model, SessionConfig};
use velomocap_video::{Decoder, Tools, VggtGeometry};

/// Head layout of the exported network.
const PATCH_SIZE: usize = 16;
/// 1 camera token + 16 register tokens precede the patch tokens.
#[allow(dead_code)]
const PATCH_TOKEN_START: usize = 17;
const EMBED_DIM: usize = 1024;
const NUM_HEADS: usize = 16;
const HEAD_DIM: usize = EMBED_DIM / NUM_HEADS;

#[derive(Clone, Debug)]
pub struct VggtConfig {
    /// VGGT-Omega ONNX graph.
    pub weights: PathBuf,
    /// Longest side of the network input, in pixels.
    pub resolution: usize,
    /// Run the network on every N-th frame; the camera is interpolated between
    /// samples. `1` runs it on every frame (accurate but slow, and the
    /// inter-frame attention cost grows quadratically).
    pub frame_stride: usize,
    /// Stop after this many video frames (0 = whole video).
    pub max_frames: usize,
    /// Clips are limited to this many network frames; longer clips are sampled
    /// more sparsely so the inter-frame attention stays affordable.
    pub max_samples: usize,
    /// Exact network canvas `(height, width)`, or `None` to derive it from the
    /// clip's aspect ratio.
    ///
    /// The shipped graph was traced with `images[1, N, 3, 512, 288]`, so the
    /// canvas has to be pinned for any clip whose aspect ratio differs; other
    /// input sizes are rejected by ONNX Runtime.
    pub canvas: Option<(usize, usize)>,
    pub session: SessionConfig,
}

impl Default for VggtConfig {
    fn default() -> Self {
        Self {
            weights: PathBuf::from("models/flowhmr/vggt_omega_1b_512.onnx"),
            resolution: 512,
            frame_stride: 10,
            max_frames: 0,
            max_samples: 16,
            canvas: Some((512, 288)),
            session: SessionConfig::default(),
        }
    }
}

/// One camera per video frame, in the conventions the reference pipeline uses.
#[derive(Clone, Debug)]
pub struct CameraTrack {
    /// Camera-from-world (OpenCV axes: x right, y down, z forward), row major.
    pub extrinsics: Vec<[[f32; 4]; 4]>,
    /// Pinhole intrinsics at the original video resolution.
    pub intrinsics: Vec<[[f32; 3]; 3]>,
    /// Original frame size `(height, width)`.
    pub image_size: (usize, usize),
    /// Network input size `(height, width)`.
    pub model_size: (usize, usize),
    /// Frame indices the network actually looked at.
    pub sample_indices: Vec<usize>,
}

impl CameraTrack {
    pub fn len(&self) -> usize {
        self.extrinsics.len()
    }

    pub fn is_empty(&self) -> bool {
        self.extrinsics.is_empty()
    }

    /// Rotations relative to the first frame, flattened row-major (9 per frame).
    ///
    /// This is what the motion generator conditions on: world-to-camera
    /// rotations normalised against frame zero, so the choice of world axes
    /// does not matter.
    /// Use one camera for every frame: a tripod shot instead of per-frame noise.
    pub fn freeze(&mut self) {
        let Some(first) = self.extrinsics.first().copied() else {
            return;
        };
        for matrix in &mut self.extrinsics {
            *matrix = first;
        }
    }

    /// Average each rotation over `2 * radius + 1` frames and re-orthonormalise.
    /// This is the "refine camera rotation" switch: it removes the jitter from
    /// the estimated track without moving the camera.
    pub fn smooth(&mut self, radius: usize) {
        if self.extrinsics.len() < 3 || radius == 0 {
            return;
        }
        let source = self.extrinsics.clone();
        let count = source.len();
        for index in 0..count {
            let begin = index.saturating_sub(radius);
            let end = (index + radius + 1).min(count);
            let span = (end - begin) as f32;
            let mut acc = [[0.0f32; 3]; 3];
            let mut position = [0.0f32; 3];
            for matrix in &source[begin..end] {
                for row in 0..3 {
                    for column in 0..3 {
                        acc[row][column] += matrix[row][column] / span;
                    }
                    position[row] += matrix[row][3] / span;
                }
            }
            let mut x = [acc[0][0], acc[1][0], acc[2][0]];
            let norm = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt().max(1e-6);
            for value in &mut x {
                *value /= norm;
            }
            let mut y = [acc[0][1], acc[1][1], acc[2][1]];
            let dot = x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
            for k in 0..3 {
                y[k] -= x[k] * dot;
            }
            let norm = (y[0] * y[0] + y[1] * y[1] + y[2] * y[2]).sqrt().max(1e-6);
            for value in &mut y {
                *value /= norm;
            }
            let z = [
                x[1] * y[2] - x[2] * y[1],
                x[2] * y[0] - x[0] * y[2],
                x[0] * y[1] - x[1] * y[0],
            ];
            let target = &mut self.extrinsics[index];
            for row in 0..3 {
                target[row][0] = x[row];
                target[row][1] = y[row];
                target[row][2] = z[row];
                target[row][3] = position[row];
            }
        }
    }

    pub fn relative_rotation_rows(&self) -> Vec<f32> {
        let first = match self.extrinsics.first() {
            Some(m) => rotation(m),
            None => return Vec::new(),
        };
        let first_t = transpose3(&first);
        let mut out = Vec::with_capacity(self.extrinsics.len() * 9);
        for matrix in &self.extrinsics {
            let r = rotation(matrix);
            let rel = mat_mul3(&r, &first_t);
            out.extend(rel.iter().flatten());
        }
        out
    }

    /// Apply the VGGT/OpenCV → Y-up convention change used by the motion model.
    pub fn to_y_up(&self) -> Vec<[[f32; 4]; 4]> {
        self.extrinsics
            .iter()
            .map(|matrix| {
                let mut m = *matrix;
                // diag(1, -1, -1, 1)
                for value in m[1].iter_mut() {
                    *value = -*value;
                }
                for value in m[2].iter_mut() {
                    *value = -*value;
                }
                m
            })
            .collect()
    }
}

/// Rotation table for an `h_patches × w_patches` grid.
///
/// Mirrors `RopePositionEmbedding` with `base = 100`, `normalize_coords = "max"`
/// and float32 periods, computed in double precision for accuracy.
pub fn rope_table(h_patches: usize, w_patches: usize) -> (Vec<f32>, Vec<f32>) {
    let n_tokens = h_patches * w_patches;
    let n_periods = HEAD_DIM / 4;
    let denom = (HEAD_DIM / 2) as f64;
    let periods: Vec<f64> = (0..n_periods)
        .map(|i| 100.0f64.powf(2.0 * i as f64 / denom))
        .collect();
    let max_hw = h_patches.max(w_patches).max(1) as f64;

    let mut sin = vec![0.0f32; n_tokens * HEAD_DIM];
    let mut cos = vec![0.0f32; n_tokens * HEAD_DIM];
    let mut angles = vec![0.0f64; HEAD_DIM / 2];
    for y in 0..h_patches {
        let cy = 2.0 * ((y as f64 + 0.5) / max_hw) - 1.0;
        for x in 0..w_patches {
            let cx = 2.0 * ((x as f64 + 0.5) / max_hw) - 1.0;
            for (k, period) in periods.iter().enumerate() {
                angles[k] = 2.0 * std::f64::consts::PI * cy / period;
                angles[n_periods + k] = 2.0 * std::f64::consts::PI * cx / period;
            }
            let token = y * w_patches + x;
            let base = token * HEAD_DIM;
            for i in 0..(HEAD_DIM / 2) {
                let value = angles[i];
                sin[base + i] = value.sin() as f32;
                cos[base + i] = value.cos() as f32;
                sin[base + HEAD_DIM / 2 + i] = value.sin() as f32;
                cos[base + HEAD_DIM / 2 + i] = value.cos() as f32;
            }
        }
    }
    (sin, cos)
}

/// Quaternion (x, y, z, w) → row-major rotation matrix.
pub fn quat_to_mat(q: [f32; 4]) -> [[f32; 3]; 3] {
    let [i, j, k, r] = q;
    let norm = i * i + j * j + k * k + r * r;
    let two_s = if norm > 0.0 { 2.0 / norm } else { 0.0 };
    [
        [
            1.0 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
        ],
        [
            two_s * (i * j + k * r),
            1.0 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
        ],
        [
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1.0 - two_s * (i * i + j * j),
        ],
    ]
}

/// Matrix → quaternion (x, y, z, w), using the sign-stable branch of
/// `mat_to_quat` so interpolation stays on the short arc.
pub fn mat_to_quat(m: &[[f32; 3]; 3]) -> [f32; 4] {
    let trace = m[0][0] + m[1][1] + m[2][2];
    let (x, y, z, w) = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        (
            (m[2][1] - m[1][2]) / s,
            (m[0][2] - m[2][0]) / s,
            (m[1][0] - m[0][1]) / s,
            0.25 * s,
        )
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0;
        (
            0.25 * s,
            (m[0][1] + m[1][0]) / s,
            (m[0][2] + m[2][0]) / s,
            (m[2][1] - m[1][2]) / s,
        )
    } else if m[1][1] > m[2][2] {
        let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0;
        (
            (m[0][1] + m[1][0]) / s,
            0.25 * s,
            (m[1][2] + m[2][1]) / s,
            (m[0][2] - m[2][0]) / s,
        )
    } else {
        let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0;
        (
            (m[0][2] + m[2][0]) / s,
            (m[1][2] + m[2][1]) / s,
            0.25 * s,
            (m[1][0] - m[0][1]) / s,
        )
    };
    let q = [x, y, z, w];
    let norm = (x * x + y * y + z * z + w * w).sqrt().max(1e-12);
    let q = [q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm];
    if q[3] < 0.0 {
        [-q[0], -q[1], -q[2], -q[3]]
    } else {
        q
    }
}

/// Shortest-arc spherical interpolation, matching `scipy.spatial.transform.Slerp`.
fn slerp(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    let mut dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    let mut b = b;
    if dot < 0.0 {
        b = [-b[0], -b[1], -b[2], -b[3]];
        dot = -dot;
    }
    if dot > 0.9995 {
        let mut out = [0.0f32; 4];
        for i in 0..4 {
            out[i] = a[i] + t * (b[i] - a[i]);
        }
        let norm = (out[0] * out[0] + out[1] * out[1] + out[2] * out[2] + out[3] * out[3])
            .sqrt()
            .max(1e-12);
        return [out[0] / norm, out[1] / norm, out[2] / norm, out[3] / norm];
    }
    let theta = dot.clamp(-1.0, 1.0).acos();
    let sin_theta = theta.sin();
    let w_a = ((1.0 - t) * theta).sin() / sin_theta;
    let w_b = (t * theta).sin() / sin_theta;
    [
        w_a * a[0] + w_b * b[0],
        w_a * a[1] + w_b * b[1],
        w_a * a[2] + w_b * b[2],
        w_a * a[3] + w_b * b[3],
    ]
}

pub struct VggtCamera {
    model: Model,
    config: VggtConfig,
    geometry: Option<VggtGeometry>,
}

impl VggtCamera {
    pub fn load(config: VggtConfig) -> Result<Self> {
        let model = Model::load(&config.weights, &config.session)
            .with_context(|| format!("loading VGGT-Omega {}", config.weights.display()))?;
        Ok(Self {
            model,
            config,
            geometry: None,
        })
    }

    pub fn backend(&self) -> Backend {
        self.model.backend()
    }

    pub fn weights(&self) -> &Path {
        self.model.path()
    }

    /// Geometry the network was fed on the last run.
    pub fn geometry(&self) -> Option<VggtGeometry> {
        self.geometry
    }

    /// Raw graph call, exposed for parity harnesses.
    ///
    /// `images` is `[1, n, 3, h, w]` in `[0, 1]`; the rope tables must cover the
    /// `(h/patch, w/patch)` grid.
    pub fn run_raw(
        &mut self,
        n: usize,
        height: usize,
        width: usize,
        images: Vec<f32>,
        rope_sin: Vec<f32>,
        rope_cos: Vec<f32>,
    ) -> Result<Vec<f32>> {
        let h_patches = height / PATCH_SIZE;
        let w_patches = width / PATCH_SIZE;
        let tokens = h_patches * w_patches;
        anyhow::ensure!(
            images.len() == n * 3 * height * width,
            "image buffer has {} values, expected {}",
            images.len(),
            n * 3 * height * width
        );
        anyhow::ensure!(
            rope_sin.len() == tokens * HEAD_DIM && rope_cos.len() == tokens * HEAD_DIM,
            "rope tables have {}/{} values, expected {}",
            rope_sin.len(),
            rope_cos.len(),
            tokens * HEAD_DIM
        );

        let images = Array5::from_shape_vec((1, n, 3, height, width), images)
            .context("building VGGT image tensor")?;
        let sin = Array2::from_shape_vec((tokens, HEAD_DIM), rope_sin)
            .context("building rope sin tensor")?;
        let cos = Array2::from_shape_vec((tokens, HEAD_DIM), rope_cos)
            .context("building rope cos tensor")?;

        let session = self.model.session_mut();
        let outputs = session
            .run(ort::inputs![
                "images" => ort::value::Value::from_array(images).context("image tensor")?,
                "rope_sin" => ort::value::Value::from_array(sin).context("rope sin tensor")?,
                "rope_cos" => ort::value::Value::from_array(cos).context("rope cos tensor")?,
            ])
            .context("running VGGT-Omega")?;
        let (shape, pose_enc) = outputs[0]
            .try_extract_tensor::<f32>()
            .context("reading pose encoding")?;
        anyhow::ensure!(
            pose_enc.len() == n * 9,
            "unexpected pose encoding shape {shape:?} for {n} sampled frames"
        );
        Ok(pose_enc.to_vec())
    }

    /// Estimate one camera per video frame.
    ///
    /// Decoding happens inside ffmpeg at the network geometry, so no full-size
    /// frames are ever materialised for this stage.
    pub fn estimate(&mut self, tools: &Tools, video: impl AsRef<Path>) -> Result<CameraTrack> {
        let video = video.as_ref();
        let info = velomocap_video::probe(tools, video)?;
        let (original_w, original_h) = (info.width, info.height);
        // The exported graph has a fixed portrait canvas (the shape it was
        // traced with), so landscape clips are rotated a quarter turn and the
        // camera is un-rotated below.
        let (mut decoder, geometry) = match self.config.canvas {
            Some(canvas) => velomocap_video::Decoder::open_vggt_canvas(
                tools,
                video,
                canvas,
                PATCH_SIZE,
                true,
            )?,
            None => Decoder::open_vggt_with(
                tools,
                video,
                self.config.resolution,
                PATCH_SIZE,
                true,
            )?,
        };
        self.geometry = Some(geometry);
        let (h_patches, w_patches) = geometry.patch_grid(PATCH_SIZE);

        // Sample the video: every `stride` frames, always including the last one
        // so the interpolation covers the whole clip.
        let total = if self.config.max_frames > 0 {
            self.config.max_frames
        } else {
            decoder.info().frames.max(1)
        };
        let stride = self.config.frame_stride.max(1);
        let mut wanted: Vec<usize> = (0..total).step_by(stride).collect();
        if wanted.is_empty() {
            wanted.push(0);
        }
        if *wanted.last().unwrap() != total - 1 {
            wanted.push(total - 1);
        }
        // Cap the number of network frames; the inter-frame attention is O((N·T)²).
        if self.config.max_samples > 0 && wanted.len() > self.config.max_samples {
            let keep = self.config.max_samples;
            let step = (wanted.len() - 1) as f64 / (keep - 1) as f64;
            let mut sampled: Vec<usize> = (0..keep)
                .map(|i| wanted[((i as f64) * step).round() as usize])
                .collect();
            sampled.dedup();
            if *sampled.last().unwrap() != total - 1 {
                let last = *sampled.last().unwrap();
                *sampled.last_mut().unwrap() = total - 1;
                let _ = last;
            }
            wanted = sampled;
        }

        let want: std::collections::HashSet<usize> = wanted.iter().copied().collect();
        let frame_len = 3 * geometry.out_h * geometry.out_w;
        let mut sampled_frames: Vec<Vec<f32>> = Vec::with_capacity(wanted.len());
        let mut index = 0usize;
        while let Some(frame) = decoder.next_frame()? {
            if want.contains(&index) && sampled_frames.len() < wanted.len() {
                sampled_frames.push(to_nchw(&frame.rgb, geometry.out_w, geometry.out_h));
            }
            index += 1;
            if index >= total {
                break;
            }
        }
        anyhow::ensure!(
            !sampled_frames.is_empty(),
            "VGGT-Omega read no frames from {}",
            video.display()
        );
        let n = sampled_frames.len();
        debug_assert_eq!(sampled_frames[0].len(), frame_len);

        // Flatten to [1, N, 3, H, W].
        let mut input = Vec::with_capacity(n * frame_len);
        for frame in &sampled_frames {
            input.extend_from_slice(frame);
        }
        let (rope_sin, rope_cos) = rope_table(h_patches, w_patches);

        let images = Array5::from_shape_vec(
            (1, n, 3, geometry.out_h, geometry.out_w),
            input,
        )
        .context("building VGGT image tensor")?;
        let tokens = h_patches * w_patches;
        let sin = Array2::from_shape_vec((tokens, HEAD_DIM), rope_sin)
            .context("building rope sin tensor")?;
        let cos = Array2::from_shape_vec((tokens, HEAD_DIM), rope_cos)
            .context("building rope cos tensor")?;

        let session = self.model.session_mut();
        let outputs = session
            .run(ort::inputs![
                "images" => ort::value::Value::from_array(images).context("image tensor")?,
                "rope_sin" => ort::value::Value::from_array(sin).context("rope sin tensor")?,
                "rope_cos" => ort::value::Value::from_array(cos).context("rope cos tensor")?,
            ])
            .context("running VGGT-Omega")?;
        let (shape, pose_enc) = outputs[0]
            .try_extract_tensor::<f32>()
            .context("reading pose encoding")?;
        anyhow::ensure!(
            pose_enc.len() == n * 9,
            "unexpected pose encoding shape {shape:?} for {n} sampled frames"
        );

        // Decode each sampled frame to a 4×4 camera.
        let mut sampled_ext = Vec::with_capacity(n);
        let mut sampled_int = Vec::with_capacity(n);
        for i in 0..n {
            let row = &pose_enc[i * 9..(i + 1) * 9];
            let t = geometry.unrotate_translation([row[0], row[1], row[2]]);
            let r = geometry.unrotate_rotation(&quat_to_mat([row[3], row[4], row[5], row[6]]));
            let mut ext = [[0.0f32; 4]; 4];
            for y in 0..3 {
                for x in 0..3 {
                    ext[y][x] = r[y][x];
                }
                ext[y][3] = t[y];
            }
            ext[3][3] = 1.0;
            sampled_ext.push(ext);

            // Field of view is relative to the original frame size; a quarter
            // turn swapped the two axes, so swap them back first.
            let (fov_h, fov_w) = geometry.unrotate_fov(row[7], row[8]);
            let (oh, ow) = (original_h as f32, original_w as f32);
            let fy = (oh / 2.0) / (fov_h / 2.0).tan();
            let fx = (ow / 2.0) / (fov_w / 2.0).tan();
            let mut k = [[0.0f32; 3]; 3];
            k[0][0] = fx;
            k[1][1] = fy;
            k[0][2] = ow / 2.0;
            k[1][2] = oh / 2.0;
            k[2][2] = 1.0;
            sampled_int.push(k);
        }

        let (extrinsics, intrinsics) =
            interpolate(&sampled_ext, &sampled_int, &wanted, total);

        Ok(CameraTrack {
            extrinsics,
            intrinsics,
            image_size: (original_h, original_w),
            model_size: (geometry.out_h, geometry.out_w),
            sample_indices: wanted,
        })
    }
}

/// RGB24 → planar float32 in `[0, 1]`, NCHW order, channel first.
fn to_nchw(rgb: &[u8], width: usize, height: usize) -> Vec<f32> {
    let plane = width * height;
    let mut out = vec![0.0f32; plane * 3];
    for i in 0..plane {
        let p = i * 3;
        out[i] = rgb[p] as f32 / 255.0;
        out[plane + i] = rgb[p + 1] as f32 / 255.0;
        out[2 * plane + i] = rgb[p + 2] as f32 / 255.0;
    }
    out
}

/// Slerp the rotation and lerp translation/intrinsics onto every frame.
fn interpolate(
    ext: &[[[f32; 4]; 4]],
    int: &[[[f32; 3]; 3]],
    sample_indices: &[usize],
    target_len: usize,
) -> (Vec<[[f32; 4]; 4]>, Vec<[[f32; 3]; 3]>) {
    if sample_indices.len() == target_len
        && sample_indices.iter().enumerate().all(|(i, &s)| i == s)
    {
        return (ext.to_vec(), int.to_vec());
    }
    let quats: Vec<[f32; 4]> = ext.iter().map(|m| mat_to_quat(&rotation(m))).collect();
    let mut out_ext = Vec::with_capacity(target_len);
    let mut out_int = Vec::with_capacity(target_len);
    let mut seg = 0usize;
    for frame in 0..target_len {
        while seg + 2 < sample_indices.len() && sample_indices[seg + 1] < frame {
            seg += 1;
        }
        let (i0, i1) = (seg, (seg + 1).min(sample_indices.len() - 1));
        let (s0, s1) = (sample_indices[i0] as f32, sample_indices[i1] as f32);
        let t = if s1 > s0 {
            ((frame as f32 - s0) / (s1 - s0)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let q = slerp(quats[i0], quats[i1], t);
        let r = quat_to_mat(q);
        let mut m = [[0.0f32; 4]; 4];
        for y in 0..3 {
            for x in 0..3 {
                m[y][x] = r[y][x];
            }
            m[y][3] = ext[i0][y][3] + t * (ext[i1][y][3] - ext[i0][y][3]);
        }
        m[3][3] = 1.0;
        out_ext.push(m);

        let mut k = [[0.0f32; 3]; 3];
        for y in 0..3 {
            for x in 0..3 {
                k[y][x] = int[i0][y][x] + t * (int[i1][y][x] - int[i0][y][x]);
            }
        }
        out_int.push(k);
    }
    (out_ext, out_int)
}

fn rotation(m: &[[f32; 4]; 4]) -> [[f32; 3]; 3] {
    [
        [m[0][0], m[0][1], m[0][2]],
        [m[1][0], m[1][1], m[1][2]],
        [m[2][0], m[2][1], m[2][2]],
    ]
}

fn transpose3(m: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

fn mat_mul3(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut out = [[0.0f32; 3]; 3];
    for (y, row) in out.iter_mut().enumerate() {
        for (x, value) in row.iter_mut().enumerate() {
            *value = a[y][0] * b[0][x] + a[y][1] * b[1][x] + a[y][2] * b[2][x];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_table_shape_and_identity() {
        let (sin, cos) = rope_table(2, 3);
        assert_eq!(sin.len(), 6 * HEAD_DIM);
        assert_eq!(cos.len(), 6 * HEAD_DIM);
        // The first four entries of each half use cos(±angle); the duplicate
        // half means token columns repeat.
        for i in 0..(HEAD_DIM / 2) {
            assert!((sin[i] - sin[HEAD_DIM / 2 + i]).abs() < 1e-6);
            assert!((cos[i] - cos[HEAD_DIM / 2 + i]).abs() < 1e-6);
        }
    }

    #[test]
    fn quaternion_round_trip() {
        // `quat_to_mat` normalises, so compare against the unit quaternion.
        let q = [0.1f32, -0.2, 0.3, 0.9];
        let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        let unit = [q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm];
        let m = quat_to_mat(q);
        let back = mat_to_quat(&m);
        for i in 0..4 {
            assert!((unit[i] - back[i]).abs() < 1e-5, "{unit:?} vs {back:?}");
        }
    }

    #[test]
    fn slerp_midpoint_is_normalised() {
        let a = mat_to_quat(&quat_to_mat([0.0, 0.0, 0.0, 1.0]));
        let b = mat_to_quat(&quat_to_mat([0.0, 0.3826834, 0.0, 0.9238795]));
        let mid = slerp(a, b, 0.5);
        let norm = (mid[0] * mid[0] + mid[1] * mid[1] + mid[2] * mid[2] + mid[3] * mid[3]).sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn relative_rotation_of_first_frame_is_identity() {
        let track = CameraTrack {
            extrinsics: vec![
                [
                    [0.0, -1.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 2.0],
                    [0.0, 0.0, 1.0, 3.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ],
            intrinsics: vec![Default::default(); 2],
            image_size: (1080, 1920),
            model_size: (512, 288),
            sample_indices: vec![0, 1],
        };
        let rows = track.relative_rotation_rows();
        assert_eq!(rows.len(), 18);
        // First frame relative rotation is the identity.
        assert!((rows[0] - 1.0).abs() < 1e-6);
        assert!((rows[4] - 1.0).abs() < 1e-6);
        assert!((rows[8] - 1.0).abs() < 1e-6);
    }
}
