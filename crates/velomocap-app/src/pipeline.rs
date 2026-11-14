//! Background processing: decode + detect + track, off the UI thread.
//!
//! The worker owns the decoder, the ONNX session and the tracker. It publishes
//! progress and finished tracks through a shared, lock-protected state so the
//! UI never blocks on inference.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

use anyhow::Result;
use velomocap_infer::{Detector, DetectorConfig, Tracker};
use velomocap_video::{Decoder, Tools, VideoInfo};

/// What the worker is doing right now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Idle,
    Opening,
    Detecting,
    Done,
    Failed,
}

#[derive(Clone, Debug)]
pub struct Progress {
    pub stage: Stage,
    pub frame: usize,
    pub total: usize,
    pub detections: usize,
    pub active_tracks: usize,
    pub backend: String,
    pub elapsed: f64,
    pub message: String,
}

impl Default for Progress {
    fn default() -> Self {
        Self {
            stage: Stage::Idle,
            frame: 0,
            total: 0,
            detections: 0,
            active_tracks: 0,
            backend: String::new(),
            elapsed: 0.0,
            message: String::new(),
        }
    }
}

impl Progress {
    pub fn fraction(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            (self.frame as f32 / self.total as f32).clamp(0.0, 1.0)
        }
    }

    pub fn fps(&self) -> f64 {
        if self.elapsed <= 0.0 || self.frame == 0 {
            0.0
        } else {
            self.frame as f64 / self.elapsed
        }
    }
}

/// One decoded frame handed to the UI for display.
pub struct PreviewFrame {
    pub index: usize,
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
}

/// Shared state between the UI and the worker thread.
#[derive(Default)]
pub struct Shared {
    pub progress: Progress,
    pub tracks: Vec<velomocap_infer::Track>,
    pub info: Option<VideoInfo>,
    /// Detections for the frame currently shown, per track id.
    pub overlay: Vec<(usize, velomocap_infer::Detection)>,
    /// Most recent decoded frame, published by the worker.
    ///
    /// The worker already has the pixels in hand, so handing them over costs
    /// one `Arc` clone instead of spawning a second ffmpeg to seek and decode.
    pub preview: Option<std::sync::Arc<PreviewFrame>>,
}

pub struct Job {
    pub handle: Option<JoinHandle<()>>,
    pub cancel: Arc<AtomicBool>,
    pub shared: Arc<Mutex<Shared>>,
}

impl Job {
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    pub fn is_running(&self) -> bool {
        self.handle
            .as_ref()
            .map(|handle| !handle.is_finished())
            .unwrap_or(false)
    }
}

pub struct Settings {
    pub video: PathBuf,
    pub weights: PathBuf,
    pub confidence: f32,
    pub iou: f32,
    pub gpu: bool,
    pub input_size: usize,
    pub scale_width: Option<usize>,
    pub min_hits: usize,
    pub max_missing: usize,
}

/// Start a background run. Returns immediately.
pub fn start(settings: Settings) -> Job {
    let shared = Arc::new(Mutex::new(Shared::default()));
    let cancel = Arc::new(AtomicBool::new(false));
    let worker_shared = Arc::clone(&shared);
    let worker_cancel = Arc::clone(&cancel);

    let handle = std::thread::Builder::new()
        .name("velomocap-worker".into())
        .spawn(move || {
            if let Err(error) = run(settings, worker_shared.clone(), worker_cancel) {
                let mut state = worker_shared.lock().unwrap();
                state.progress.stage = Stage::Failed;
                state.progress.message = format!("{error:#}");
            }
        })
        .ok();

    Job {
        handle,
        cancel,
        shared,
    }
}

/// Longest preview edge handed to the UI.
const PREVIEW_WIDTH: usize = 480;

/// Box-filter the frame down to `target_width` (never upscales).
fn downscale_preview(index: usize, rgb: &[u8], width: usize, height: usize, target_width: usize) -> PreviewFrame {
    if width <= target_width {
        return PreviewFrame {
            index,
            width,
            height,
            rgb: rgb.to_vec(),
        };
    }
    let scale = width as f32 / target_width as f32;
    let out_w = target_width;
    let out_h = ((height as f32 / scale).round() as usize).max(1);
    let mut out = vec![0u8; out_w * out_h * 3];
    for y in 0..out_h {
        let sy = ((y as f32 * scale) as usize).min(height - 1);
        let row = sy * width * 3;
        for x in 0..out_w {
            let sx = ((x as f32 * scale) as usize).min(width - 1);
            let src = row + sx * 3;
            let dst = (y * out_w + x) * 3;
            out[dst] = rgb[src];
            out[dst + 1] = rgb[src + 1];
            out[dst + 2] = rgb[src + 2];
        }
    }
    PreviewFrame {
        index,
        width: out_w,
        height: out_h,
        rgb: out,
    }
}

fn run(settings: Settings, shared: Arc<Mutex<Shared>>, cancel: Arc<AtomicBool>) -> Result<()> {
    let started = Instant::now();
    {
        let mut state = shared.lock().unwrap();
        state.progress.stage = Stage::Opening;
        state.progress.message = "Opening video".into();
    }

    let tools = Tools::discover()?;
    let mut decoder = Decoder::open(&tools, &settings.video, settings.scale_width)?;
    let info = decoder.info().clone();
    let (width, height) = decoder.dimensions();

    let config = DetectorConfig {
        input_size: settings.input_size,
        confidence: settings.confidence,
        iou: settings.iou,
        persons_only: true,
        session: velomocap_infer::SessionConfig {
            gpu: settings.gpu,
            ..Default::default()
        },
    };
    let mut detector = Detector::load(&settings.weights, config)?;
    let backend = detector.backend().as_str().to_string();
    if std::env::var("VELOMOCAP_DEBUG_TIMING").is_ok() {
        eprintln!("detector: {}", detector.describe());
    }

    {
        let mut state = shared.lock().unwrap();
        state.info = Some(info.clone());
        state.progress.stage = Stage::Detecting;
        state.progress.total = info.frames.max(1);
        state.progress.backend = backend.clone();
        state.progress.message = format!(
            "Detecting with YOLO on {} ({width}x{height})",
            backend.to_uppercase()
        );
    }

    let mut tracker = Tracker::new(0.3, settings.max_missing);
    let mut frame_index = 0usize;

    while let Some(frame) = decoder.next_frame()? {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let detect_start = Instant::now();
        let detections = detector.detect(&frame.rgb, frame.width, frame.height)?;
        let detect_ms = detect_start.elapsed().as_secs_f64() * 1e3;
        let active = tracker.update(&detections);
        frame_index += 1;
        if std::env::var("VELOMOCAP_DEBUG_TIMING").is_ok() {
            eprintln!("frame {frame_index}: detect {detect_ms:.1} ms, {} det", detections.len());
        }

        // Publish a downscaled copy of the frame the detections came from.
        // Uploading a full-resolution texture ten times a second starves the
        // UI thread; a ~480 px preview costs almost nothing and looks the same
        // at the sizes the viewport actually draws.
        let preview = std::sync::Arc::new(downscale_preview(
            frame_index - 1,
            &frame.rgb,
            frame.width,
            frame.height,
            PREVIEW_WIDTH,
        ));

        let mut state = shared.lock().unwrap();
        state.progress.frame = frame_index;
        state.progress.detections = detections.len();
        state.progress.active_tracks = active;
        state.progress.elapsed = started.elapsed().as_secs_f64();
        state.preview = Some(preview);
        state.overlay = tracker
            .tracks
            .iter()
            .filter_map(|track| track.detection(frame_index - 1).map(|d| (track.id, *d)))
            .collect();
    }

    tracker.prune(settings.min_hits);
    {
        let mut state = shared.lock().unwrap();
        state.tracks = tracker.tracks;
        state.progress.stage = Stage::Done;
        state.progress.elapsed = started.elapsed().as_secs_f64();
        state.progress.message = format!(
            "{} frame(s), {} track(s) in {:.1} s",
            frame_index,
            state.tracks.len(),
            state.progress.elapsed
        );
    }
    Ok(())
}
