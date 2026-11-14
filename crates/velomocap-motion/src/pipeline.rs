//! The full native capture pipeline: video → keypoints → SMPL-X → BVH.
//!
//! This is the zero-Python replacement for the original application's motion
//! path. It runs entirely in-process: ffmpeg decodes, ONNX Runtime estimates
//! 2D keypoints, the Levenberg-Marquardt fitter solves SMPL-X pose parameters,
//! and the Rust BVH writer emits the result.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use velomocap_body::{BodyModel, Skeleton};
use velomocap_infer::{PoseConfig, PoseDetection, PoseEstimator, SessionConfig};
use velomocap_video::{Decoder, Tools};

use crate::export::{write_bvh, BvhExport};
use crate::fit::{summarise, FitConfig, FittedFrame, Fitter};

#[derive(Clone, Debug)]
pub struct CaptureConfig {
    /// YOLO-pose ONNX weights.
    pub weights: PathBuf,
    /// SMPL-X `.npz`.
    pub model: PathBuf,
    /// Confidence threshold for person detection.
    pub confidence: f32,
    /// Work resolution; frames are scaled before inference.
    pub work_width: usize,
    /// Prefer the CUDA execution provider.
    pub gpu: bool,
    /// Number of shape coefficients to keep.
    pub num_betas: usize,
    /// Stop after this many frames (0 = whole video).
    pub max_frames: usize,
    pub fit: FitConfig,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            weights: PathBuf::from("models/detector/yolov8x-pose.onnx"),
            model: PathBuf::from("models/body_models/smplx/neutral/model.npz"),
            confidence: 0.25,
            work_width: 960,
            gpu: true,
            num_betas: 10,
            max_frames: 0,
            fit: FitConfig::default(),
        }
    }
}

/// Everything the pipeline produced.
pub struct CaptureResult {
    pub frames: Vec<FittedFrame>,
    pub rest: Skeleton,
    pub fps: f64,
    pub width: usize,
    pub height: usize,
    pub video_frames: usize,
    pub backend: String,
    /// Detections per frame, kept so the UI can draw 2D overlays.
    pub detections: Vec<Option<PoseDetection>>,
    /// Joint parents (`-1` at the root), for drawing and retargeting.
    pub parents: Vec<i32>,
    /// Joint names, in the body model's order.
    pub joint_names: Vec<String>,
    pub timings: Timings,
}

#[derive(Clone, Debug, Default)]
pub struct Timings {
    pub decode_seconds: f64,
    pub detect_seconds: f64,
    pub fit_seconds: f64,
    pub total_seconds: f64,
}

impl CaptureResult {
    pub fn summary(&self) -> String {
        format!(
            "{} frames @ {:.2} fps, {} ({:.1} fps end-to-end)\n{}",
            self.frames.len(),
            self.fps,
            summarise(&self.frames),
            self.frames.len() as f64 / self.timings.total_seconds.max(1e-9),
            self.timings_line()
        )
    }

    pub fn timings_line(&self) -> String {
        format!(
            "decode {:.2}s · detect {:.2}s · fit {:.2}s · total {:.2}s",
            self.timings.decode_seconds,
            self.timings.detect_seconds,
            self.timings.fit_seconds,
            self.timings.total_seconds
        )
    }

    /// Export the motion as BVH next to the source video.
    pub fn write_bvh(&self, path: impl AsRef<Path>) -> Result<BvhExport> {
        write_bvh(path, &self.frames, &self.rest, self.fps)
    }

    /// Convenience: `<video stem>.bvh` beside the input.
    pub fn default_bvh_path(video: &Path) -> PathBuf {
        let stem = video
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "motion".to_string());
        video
            .parent()
            .unwrap_or(Path::new("."))
            .join(format!("{stem}.bvh"))
    }
}

/// Progress callback: `(frame index, total frames, stage)`.
pub type Progress<'a> = &'a mut dyn FnMut(usize, usize, &str);

/// Optional hook that receives the decoded frame, for live previews.
pub type FrameSink<'a> = &'a mut dyn FnMut(&velomocap_video::Frame);

/// Optional hook that receives each fitted frame, for live 3D previews.
pub type FitSink<'a> = &'a mut dyn FnMut(usize, &FittedFrame);

/// Run the whole pipeline on one video.
pub fn capture(video: impl AsRef<Path>, config: &CaptureConfig) -> Result<CaptureResult> {
    capture_with_progress(video, config, &mut |_, _, _| {})
}

pub fn capture_with_progress(
    video: impl AsRef<Path>,
    config: &CaptureConfig,
    progress: Progress<'_>,
) -> Result<CaptureResult> {
    capture_with_sink(video, config, progress, &mut |_| {})
}

/// Like [`capture_with_progress`], but also hands every decoded frame to
/// `sink` so a UI can show live previews.
pub fn capture_with_sink(
    video: impl AsRef<Path>,
    config: &CaptureConfig,
    progress: Progress<'_>,
    sink: FrameSink<'_>,
) -> Result<CaptureResult> {
    capture_with_hooks(video, config, progress, sink, &mut |_, _| {})
}

/// The full pipeline, with hooks for both the video frame and the fitted pose.
///
/// A UI wants both: the frame being processed *and* the skeleton the fitter just
/// produced, so the 3D view moves while the run is still going.
pub fn capture_with_hooks(
    video: impl AsRef<Path>,
    config: &CaptureConfig,
    progress: Progress<'_>,
    sink: FrameSink<'_>,
    fit_sink: FitSink<'_>,
) -> Result<CaptureResult> {
    let started = Instant::now();
    let video = video.as_ref();
    let tools = Tools::discover()?;

    progress(0, 0, "loading model");
    let body = BodyModel::load(&config.model, config.num_betas)
        .with_context(|| format!("loading SMPL-X {}", config.model.display()))?;

    progress(0, 0, "opening video");
    let mut decoder = Decoder::open(&tools, video, Some(config.work_width))?;
    let fps = if decoder.info().fps > 0.0 {
        decoder.info().fps
    } else {
        30.0
    };
    let (width, height) = decoder.dimensions();
    let total = if config.max_frames > 0 {
        config.max_frames.min(decoder.info().frames.max(1))
    } else {
        decoder.info().frames.max(1)
    };

    progress(0, total, "loading detector");
    let pose_config = PoseConfig {
        confidence: config.confidence,
        session: SessionConfig {
            gpu: config.gpu,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut estimator = PoseEstimator::load(&config.weights, pose_config)
        .with_context(|| format!("loading {}", config.weights.display()))?;
    let backend = estimator.backend().as_str().to_string();

    let fitter = Fitter::new(&body, width, height, config.fit.clone());
    let mut frames: Vec<FittedFrame> = Vec::with_capacity(total);
    let mut detections: Vec<Option<PoseDetection>> = Vec::with_capacity(total);
    let mut previous: Option<velomocap_body::Pose> = None;
    let mut timings = Timings::default();
    let mut decoded = 0usize;

    while let Some(frame) = decoder.next_frame()? {
        let decode_start = Instant::now();
        timings.decode_seconds += decode_start.elapsed().as_secs_f64();
        decoded += 1;

        sink(&frame);
        progress(decoded, total, "detecting");
        let detect_start = Instant::now();
        let poses = estimator.detect(&frame.rgb, frame.width, frame.height)?;
        timings.detect_seconds += detect_start.elapsed().as_secs_f64();

        // Track the largest person across frames: for single-subject capture
        // this is the actor, and it keeps identity stable without a tracker.
        let primary = poses.into_iter().max_by(|a, b| {
            a.bbox
                .area()
                .partial_cmp(&b.bbox.area())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        progress(decoded, total, "fitting");
        let fit_start = Instant::now();
        let fitted = match &primary {
            Some(detection) => fitter.fit(detection, previous.as_ref())?,
            None => {
                let pose = previous
                    .clone()
                    .unwrap_or_else(|| velomocap_body::Pose::neutral(body.num_joints, body.num_betas));
                let skeleton = body.forward_joints(&pose);
                FittedFrame {
                    pose,
                    skeleton,
                    reprojection_error: f32::INFINITY,
                    used_keypoints: 0,
                    converged: false,
                }
            }
        };
        timings.fit_seconds += fit_start.elapsed().as_secs_f64();
        previous = Some(fitted.pose.clone());

        fit_sink(frames.len(), &fitted);
        frames.push(fitted);
        detections.push(primary);

        if config.max_frames > 0 && frames.len() >= config.max_frames {
            break;
        }
    }

    timings.total_seconds = started.elapsed().as_secs_f64();
    let rest = body.forward_joints(&velomocap_body::Pose::neutral(
        body.num_joints,
        body.num_betas,
    ));

    progress(frames.len(), total, "done");
    Ok(CaptureResult {
        frames,
        rest,
        fps,
        width,
        height,
        video_frames: decoded,
        backend,
        detections,
        parents: body.parents.clone(),
        joint_names: body.names.clone(),
        timings,
    })
}
