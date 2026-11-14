//! YOLOv8-pose: person detection plus 17 COCO body keypoints.
//!
//! Output layout is `[1, 56, anchors]`: 4 box values (cx, cy, w, h) in input
//! pixels, 1 objectness/class score, then 17 `(x, y, confidence)` triples.
//! Everything here is decoded natively — no Ultralytics code runs at runtime.

use std::path::Path;

use anyhow::{Context, Result};
use ndarray::Array4;

use crate::detector::{nms, Detection};
use crate::{Backend, Model, SessionConfig};

/// COCO keypoint names, in model order.
pub const KEYPOINT_NAMES: [&str; 17] = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
];

/// One 2-D keypoint in source-image pixels.
#[derive(Clone, Copy, Debug, Default)]
pub struct Keypoint {
    pub x: f32,
    pub y: f32,
    /// Model confidence for this joint.
    pub confidence: f32,
}

impl Keypoint {
    pub fn is_confident(&self, threshold: f32) -> bool {
        self.confidence >= threshold && self.x.is_finite() && self.y.is_finite()
    }
}

/// A detected person with its skeleton.
#[derive(Clone, Debug)]
pub struct PoseDetection {
    pub bbox: Detection,
    pub keypoints: [Keypoint; 17],
}

impl PoseDetection {
    /// Mean keypoint confidence, useful for ranking.
    pub fn mean_keypoint_confidence(&self) -> f32 {
        self.keypoints.iter().map(|k| k.confidence).sum::<f32>() / 17.0
    }

    pub fn keypoint(&self, index: usize) -> Keypoint {
        self.keypoints[index]
    }
}

#[derive(Clone, Debug)]
pub struct PoseConfig {
    pub input_size: usize,
    pub confidence: f32,
    pub iou: f32,
    /// Minimum per-joint confidence retained in the output.
    pub keypoint_threshold: f32,
    pub session: SessionConfig,
}

impl Default for PoseConfig {
    fn default() -> Self {
        Self {
            input_size: 640,
            confidence: 0.25,
            iou: 0.45,
            keypoint_threshold: 0.05,
            session: SessionConfig::default(),
        }
    }
}

pub struct PoseEstimator {
    model: Model,
    config: PoseConfig,
}

impl PoseEstimator {
    pub fn load(weights: impl AsRef<Path>, config: PoseConfig) -> Result<Self> {
        let model = Model::load(weights, &config.session)?;
        Ok(Self { model, config })
    }

    pub fn backend(&self) -> Backend {
        self.model.backend()
    }

    pub fn describe(&self) -> String {
        self.model.describe()
    }

    /// Detect people and their keypoints in one RGB frame.
    pub fn detect(&mut self, rgb: &[u8], width: usize, height: usize) -> Result<Vec<PoseDetection>> {
        if rgb.len() != width * height * 3 {
            anyhow::bail!("image buffer has {} bytes, expected {}", rgb.len(), width * height * 3);
        }
        let size = self.config.input_size;
        let (tensor, scale, pad_x, pad_y) =
            crate::detector::letterbox_public(rgb, width, height, size);

        let input = Array4::from_shape_vec((1, 3, size, size), tensor)
            .context("building pose input tensor")?;
        let value = ort::value::Value::from_array(input).context("creating pose tensor")?;
        let outputs = self
            .model
            .session_mut()
            .run(ort::inputs!["images" => value])
            .context("YOLO-pose inference")?;

        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .context("reading pose output tensor")?;
        let dims: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
        let decoded = decode(data, &dims, scale, pad_x, pad_y, width, height)?;

        // Filter by score, then NMS on the boxes.
        let mut candidates: Vec<PoseDetection> = decoded
            .into_iter()
            .filter(|pose| pose.bbox.score >= self.config.confidence)
            .collect();
        let boxes: Vec<Detection> = candidates.iter().map(|p| p.bbox).collect();
        let keep = nms(&boxes, self.config.iou);
        let mut selected = Vec::with_capacity(keep.len());
        for index in keep {
            selected.push(candidates.swap_remove(index.min(candidates.len() - 1)));
        }
        candidates = selected;

        // Sort by confidence so the primary subject comes first.
        candidates.sort_by(|a, b| {
            b.bbox
                .score
                .partial_cmp(&a.bbox.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(candidates)
    }
}

/// Decode `[1, 56, anchors]` into poses in source-image coordinates.
fn decode(
    data: &[f32],
    dims: &[usize],
    scale: f32,
    pad_x: f32,
    pad_y: f32,
    width: usize,
    height: usize,
) -> Result<Vec<PoseDetection>> {
    if dims.len() != 3 || dims[1] < 5 {
        anyhow::bail!("unexpected YOLO-pose output shape {dims:?}");
    }
    let channels = dims[1];
    let anchors = dims[2];
    let expected = 4 + 1 + 17 * 3;
    if channels < expected {
        anyhow::bail!(
            "YOLO-pose output has {channels} channels, expected at least {expected}"
        );
    }

    let to_source_x = |v: f32| (v - pad_x) / scale;
    let to_source_y = |v: f32| (v - pad_y) / scale;

    let mut out = Vec::with_capacity(anchors);
    for a in 0..anchors {
        let at = |c: usize| data[c * anchors + a];
        let score = at(4);
        if !score.is_finite() || score <= 0.0 {
            continue;
        }
        let (cx, cy, w, h) = (at(0), at(1), at(2), at(3));
        let box_ = Detection {
            x0: to_source_x(cx - w * 0.5).clamp(0.0, width as f32),
            y0: to_source_y(cy - h * 0.5).clamp(0.0, height as f32),
            x1: to_source_x(cx + w * 0.5).clamp(0.0, width as f32),
            y1: to_source_y(cy + h * 0.5).clamp(0.0, height as f32),
            score,
            class_id: 0,
        };

        let mut keypoints = [Keypoint::default(); 17];
        for (k, slot) in keypoints.iter_mut().enumerate() {
            let base = 5 + k * 3;
            *slot = Keypoint {
                x: to_source_x(at(base)),
                y: to_source_y(at(base + 1)),
                confidence: at(base + 2),
            };
        }
        out.push(PoseDetection {
            bbox: box_,
            keypoints,
        });
    }
    Ok(out)
}

/// Convenience: run the estimator on an image file.
pub fn estimate_file(
    weights: impl AsRef<Path>,
    image: impl AsRef<Path>,
    config: PoseConfig,
) -> Result<(Vec<PoseDetection>, Backend, usize, usize)> {
    let (width, height, rgb) = {
        let bytes = std::fs::read(image.as_ref())
            .with_context(|| format!("reading {}", image.as_ref().display()))?;
        crate::detector::decode_image(&bytes)?
    };
    let mut estimator = PoseEstimator::load(weights, config)?;
    let backend = estimator.backend();
    let poses = estimator.detect(&rgb, width, height)?;
    Ok((poses, backend, width, height))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_one_pose_from_channel_major_output() {
        // [1, 56, 1]: one anchor, identity letterbox.
        let mut data = vec![0.0f32; 56];
        data[0] = 320.0; // cx
        data[1] = 320.0; // cy
        data[2] = 100.0; // w
        data[3] = 200.0; // h
        data[4] = 0.9; // score
        // First keypoint at (300, 250) with confidence 0.8.
        data[5] = 300.0;
        data[6] = 250.0;
        data[7] = 0.8;
        let poses = decode(&data, &[1, 56, 1], 1.0, 0.0, 0.0, 640, 640).unwrap();
        assert_eq!(poses.len(), 1);
        let pose = &poses[0];
        assert!((pose.bbox.x0 - 270.0).abs() < 1e-4);
        assert!((pose.bbox.y1 - 420.0).abs() < 1e-4);
        assert!((pose.keypoints[0].x - 300.0).abs() < 1e-4);
        assert!((pose.keypoints[0].confidence - 0.8).abs() < 1e-6);
    }

    #[test]
    fn letterbox_mapping_round_trips() {
        // A 1280x720 frame is scaled by 0.5 and padded 140 px vertically.
        let (cx, cy) = (640.0f32, 360.0f32);
        let (scale, pad_y) = (0.5f32, 140.0f32);
        let tensor_x = cx * scale;
        let tensor_y = cy * scale + pad_y;
        assert!(((tensor_x - 0.0) / scale - cx).abs() < 1e-4);
        assert!(((tensor_y - pad_y) / scale - cy).abs() < 1e-4);
    }
}
