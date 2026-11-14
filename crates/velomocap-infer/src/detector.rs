//! YOLOv8 person detector on ONNX Runtime.
//!
//! Implements the full pre/post-processing natively: letterbox resize, NCHW
//! float normalisation, the `[1, 84, N]` output decode, class filtering and
//! class-wise NMS. Only detections above the configured confidence survive.
//!
//! The exported graph used by VeloMocap is the standard Ultralytics export
//! (`yolov8x.onnx`, imgsz 640, opset 17): input `images` `[1, 3, H, W]` in
//! 0..1, output `output0` `[1, 4 + classes, anchors]` with xywh in input
//! pixels.

use std::path::Path;

use anyhow::{Context, Result};
use ndarray::Array4;

use crate::{Backend, Model, SessionConfig};

/// COCO class index for `person`.
pub const PERSON_CLASS: usize = 0;

#[derive(Clone, Debug)]
pub struct DetectorConfig {
    /// Square network input size the model was exported with.
    pub input_size: usize,
    /// Keep detections above this confidence.
    pub confidence: f32,
    /// Class-wise NMS IoU threshold.
    pub iou: f32,
    /// Restrict to the person class (the only one this pipeline needs).
    pub persons_only: bool,
    pub session: SessionConfig,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            input_size: 640,
            confidence: 0.25,
            iou: 0.45,
            persons_only: true,
            session: SessionConfig::default(),
        }
    }
}

/// One detection in original-image pixel coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Detection {
    /// Left, top, right, bottom in source pixels.
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
    pub score: f32,
    pub class_id: usize,
}

impl Detection {
    pub fn width(&self) -> f32 {
        self.x1 - self.x0
    }

    pub fn height(&self) -> f32 {
        self.y1 - self.y0
    }

    pub fn area(&self) -> f32 {
        self.width().max(0.0) * self.height().max(0.0)
    }

    pub fn center(&self) -> (f32, f32) {
        ((self.x0 + self.x1) * 0.5, (self.y0 + self.y1) * 0.5)
    }

    pub fn iou(&self, other: &Detection) -> f32 {
        let x0 = self.x0.max(other.x0);
        let y0 = self.y0.max(other.y0);
        let x1 = self.x1.min(other.x1);
        let y1 = self.y1.min(other.y1);
        let inter = (x1 - x0).max(0.0) * (y1 - y0).max(0.0);
        let union = self.area() + other.area() - inter;
        if union <= 0.0 {
            0.0
        } else {
            inter / union
        }
    }
}

pub struct Detector {
    model: Model,
    config: DetectorConfig,
}

impl Detector {
    pub fn load(weights: impl AsRef<Path>, config: DetectorConfig) -> Result<Self> {
        let model = Model::load(weights, &config.session)?;
        Ok(Self { model, config })
    }

    pub fn backend(&self) -> Backend {
        self.model.backend()
    }

    pub fn describe(&self) -> String {
        self.model.describe()
    }

    /// Detect people in an RGB image (`width * height * 3`, row-major u8).
    pub fn detect(&mut self, rgb: &[u8], width: usize, height: usize) -> Result<Vec<Detection>> {
        let expected = width * height * 3;
        if rgb.len() != expected {
            anyhow::bail!(
                "image buffer has {} bytes, expected {expected} for {width}x{height}",
                rgb.len()
            );
        }
        let size = self.config.input_size;
        let (tensor, scale, pad_x, pad_y) = letterbox(rgb, width, height, size);

        let input = Array4::from_shape_vec((1, 3, size, size), tensor)
            .context("building detector input tensor")?;
        let value = ort::value::Value::from_array(input).context("creating detector tensor")?;
        let outputs = self
            .model
            .session_mut()
            .run(ort::inputs!["images" => value])
            .context("YOLO inference")?;

        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .context("reading YOLO output tensor")?;
        // Expected [1, 4 + classes, anchors]; be tolerant about a transposed layout.
        let dims: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
        let raw = decode(data, &dims)?;

        let mut detections = Vec::new();
        for row in raw {
            let (class_id, score) = match row.best_class {
                Some(value) => value,
                None => continue,
            };
            if score < self.config.confidence {
                continue;
            }
            if self.config.persons_only && class_id != PERSON_CLASS {
                continue;
            }
            let detection = Detection {
                x0: (row.cx - row.w * 0.5 - pad_x) / scale,
                y0: (row.cy - row.h * 0.5 - pad_y) / scale,
                x1: (row.cx + row.w * 0.5 - pad_x) / scale,
                y1: (row.cy + row.h * 0.5 - pad_y) / scale,
                score,
                class_id,
            };
            detections.push(clamp_to_image(detection, width, height));
        }

        let keep = nms(&detections, self.config.iou);
        Ok(keep.into_iter().map(|i| detections[i]).collect())
    }
}

fn clamp_to_image(mut d: Detection, width: usize, height: usize) -> Detection {
    d.x0 = d.x0.clamp(0.0, width as f32);
    d.x1 = d.x1.clamp(0.0, width as f32);
    d.y0 = d.y0.clamp(0.0, height as f32);
    d.y1 = d.y1.clamp(0.0, height as f32);
    d
}

struct RawBox {
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,
    /// Highest-scoring class, or None when all scores are below zero.
    best_class: Option<(usize, f32)>,
}

fn decode(data: &[f32], dims: &[usize]) -> Result<Vec<RawBox>> {
    // Standard layout: [1, 4 + C, N]
    if dims.len() == 3 {
        let channels = dims[1];
        let anchors = dims[2];
        if channels >= 5 {
            let classes = channels - 4;
            let mut out = Vec::with_capacity(anchors);
            for a in 0..anchors {
                let at = |c: usize| data[c * anchors + a];
                out.push(RawBox {
                    cx: at(0),
                    cy: at(1),
                    w: at(2),
                    h: at(3),
                    best_class: best_of(at, 4, classes),
                });
            }
            return Ok(out);
        }
        // Transposed layout: [1, N, 4 + C]
        let anchors = dims[1];
        let channels = dims[2];
        if channels >= 5 {
            let classes = channels - 4;
            let mut out = Vec::with_capacity(anchors);
            for a in 0..anchors {
                let at = |c: usize| data[a * channels + c];
                out.push(RawBox {
                    cx: at(0),
                    cy: at(1),
                    w: at(2),
                    h: at(3),
                    best_class: best_of(at, 4, classes),
                });
            }
            return Ok(out);
        }
    }
    anyhow::bail!("unexpected YOLO output shape {dims:?}")
}

fn best_of(at: impl Fn(usize) -> f32, offset: usize, classes: usize) -> Option<(usize, f32)> {
    let mut best: Option<(usize, f32)> = None;
    for c in 0..classes {
        let score = at(offset + c);
        if best.map_or(true, |(_, s)| score > s) {
            best = Some((c, score));
        }
    }
    best.filter(|(_, score)| score.is_finite())
}

/// Greedy class-wise non-maximum suppression; returns kept indices.
pub fn nms(detections: &[Detection], iou_threshold: f32) -> Vec<usize> {
    let mut order: Vec<usize> = (0..detections.len()).collect();
    order.sort_by(|&a, &b| {
        detections[b]
            .score
            .partial_cmp(&detections[a].score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut keep: Vec<usize> = Vec::new();
    for &index in &order {
        let candidate = &detections[index];
        let suppressed = keep.iter().any(|&k| {
            detections[k].class_id == candidate.class_id
                && detections[k].iou(candidate) > iou_threshold
        });
        if !suppressed {
            keep.push(index);
        }
    }
    keep
}

/// Public wrapper so the pose estimator can share the exact same preprocessing.
pub fn letterbox_public(
    rgb: &[u8],
    width: usize,
    height: usize,
    size: usize,
) -> (Vec<f32>, f32, f32, f32) {
    letterbox(rgb, width, height, size)
}

/// Letterbox resize to a square, returning the tensor and the mapping back.
///
/// Returns `(nchw, scale, pad_x, pad_y)` where `source = (tensor - pad) / scale`.
fn letterbox(rgb: &[u8], width: usize, height: usize, size: usize) -> (Vec<f32>, f32, f32, f32) {
    let scale = (size as f32 / width as f32).min(size as f32 / height as f32);
    let new_w = ((width as f32 * scale).round() as usize).clamp(1, size);
    let new_h = ((height as f32 * scale).round() as usize).clamp(1, size);
    let pad_x = (size - new_w) as f32 / 2.0;
    let pad_y = (size - new_h) as f32 / 2.0;

    // 114/255 grey, the standard YOLO letterbox fill.
    let fill = 114.0f32 / 255.0;
    let mut tensor = vec![fill; 3 * size * size];

    for ty in 0..new_h {
        // Nearest-neighbour sampling keeps this dependency-free and exact
        // enough for detection; the model was trained on bilinear resize.
        let sy = ((ty as f32 + 0.5) / scale - 0.5).clamp(0.0, height as f32 - 1.0);
        let y0 = sy.floor() as usize;
        let y1 = (y0 + 1).min(height - 1);
        let fy = sy - y0 as f32;
        for tx in 0..new_w {
            let sx = ((tx as f32 + 0.5) / scale - 0.5).clamp(0.0, width as f32 - 1.0);
            let x0 = sx.floor() as usize;
            let x1 = (x0 + 1).min(width - 1);
            let fx = sx - x0 as f32;

            let mut pixel = [0.0f32; 3];
            for channel in 0..3 {
                let sample = |x: usize, y: usize| {
                    rgb[(y * width + x) * 3 + channel] as f32 * (1.0 / 255.0)
                };
                let top = sample(x0, y0) * (1.0 - fx) + sample(x1, y0) * fx;
                let bottom = sample(x0, y1) * (1.0 - fx) + sample(x1, y1) * fx;
                pixel[channel] = top * (1.0 - fy) + bottom * fy;
            }

            let px = tx + pad_x as usize;
            let py = ty + pad_y as usize;
            for channel in 0..3 {
                tensor[channel * size * size + py * size + px] = pixel[channel];
            }
        }
    }
    (tensor, scale, pad_x, pad_y)
}

/// Decode PNG/JPEG bytes to `(width, height, rgb)`.
pub fn decode_image(bytes: &[u8]) -> Result<(usize, usize, Vec<u8>)> {
    let image = image::load_from_memory(bytes).context("decoding image bytes")?.to_rgb8();
    let (w, h) = image.dimensions();
    Ok((w as usize, h as usize, image.into_raw()))
}

/// Convenience wrapper used by the app and CLI.
pub fn detect_file(
    weights: impl AsRef<Path>,
    image: impl AsRef<Path>,
    config: DetectorConfig,
) -> Result<(Vec<Detection>, Backend, usize, usize)> {
    let image = image::open(image.as_ref())
        .with_context(|| format!("opening image {}", image.as_ref().display()))?
        .to_rgb8();
    let (width, height) = image.dimensions();
    let mut detector = Detector::load(weights, config)?;
    let backend = detector.backend();
    let detections = detector.detect(image.as_raw(), width as usize, height as usize)?;
    Ok((detections, backend, width as usize, height as usize))
}

/// Small helper so callers can format a detection table.
pub fn format_detections(detections: &[Detection]) -> String {
    let mut out = String::new();
    for (index, d) in detections.iter().enumerate() {
        out.push_str(&format!(
            "#{index} class={} score={:.3} box=({:.1},{:.1})-({:.1},{:.1}) {:.0}x{:.0}\n",
            d.class_id,
            d.score,
            d.x0,
            d.y0,
            d.x1,
            d.y1,
            d.width(),
            d.height()
        ));
    }
    out
}

/// Exposed for tests: build a detection quickly.
#[cfg(test)]
fn det(x0: f32, y0: f32, x1: f32, y1: f32, score: f32) -> Detection {
    Detection {
        x0,
        y0,
        x1,
        y1,
        score,
        class_id: PERSON_CLASS,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iou_of_identical_boxes_is_one() {
        let a = det(0.0, 0.0, 10.0, 10.0, 1.0);
        assert!((a.iou(&a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn nms_keeps_the_highest_scoring_overlap() {
        let boxes = vec![
            det(0.0, 0.0, 10.0, 10.0, 0.9),
            det(1.0, 1.0, 11.0, 11.0, 0.8),
            det(50.0, 50.0, 60.0, 60.0, 0.7),
        ];
        let keep = nms(&boxes, 0.45);
        assert_eq!(keep, vec![0, 2]);
    }

    #[test]
    fn letterbox_keeps_aspect_ratio() {
        let rgb = vec![0u8; 100 * 50 * 3];
        let (tensor, scale, pad_x, pad_y) = letterbox(&rgb, 100, 50, 640);
        assert_eq!(tensor.len(), 3 * 640 * 640);
        assert!((scale - 6.4).abs() < 1e-5);
        assert!((pad_y - 160.0).abs() < 1e-3);
        assert!(pad_x.abs() < 1e-3);
    }

    #[test]
    fn decode_reads_channel_major_output() {
        // [1, 5, 2] = 4 box coords + 1 class, 2 anchors.
        let data = vec![
            10.0, 20.0, // cx
            10.0, 20.0, // cy
            4.0, 4.0, // w
            4.0, 4.0, // h
            0.5, 0.9, // score
        ];
        let boxes = decode(&data, &[1, 5, 2]).unwrap();
        assert_eq!(boxes.len(), 2);
        assert_eq!(boxes[1].best_class, Some((0, 0.9)));
    }
}
