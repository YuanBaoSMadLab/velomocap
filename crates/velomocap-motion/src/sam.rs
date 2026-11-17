//! Multi-person capture with the SAM 3D Body chain.
//!
//! This is the ONNX replacement for the keypoint + SMPL-X fit: every detected
//! person gets their own crop, their own pass through the eleven SAM graphs, and
//! their own MHR mesh. Persons are associated across frames with the IoU tracker,
//! so a clip with four people yields four tracks with a mesh per frame each.
//!
//! ```text
//! frame -> person detector -> IoU tracker -> per-person crop
//!       -> SAM 3D Body (backbone, 6 layers, pose, MHR, post)
//!       -> 18439-vertex mesh + 127 joints + 70 keypoints, camera placement
//! ```
//!
//! The camera assumption matches the original application: focal length
//! `hypot(w, h)` with the principal point at the image centre
//! (`mocap/flow_camera.py`).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use velomocap_infer::{
    default_intrinsics, Detection, Detector, DetectorConfig, MeshTopology, Sam3dBody, Sam3dInput,
    SessionConfig, Tracker,
};
use velomocap_video::{Decoder, Frame, Tools};

/// How the capture finds and reconstructs people.
#[derive(Clone, Debug)]
pub struct SamConfig {
    /// Person detector (`yolov8x.onnx`, Ultralytics layout).
    pub detector: PathBuf,
    /// Directory with `sam3d_*.onnx`, `mhr_model.onnx` and `sam3d_mesh.json`.
    pub sam3d_dir: PathBuf,
    /// Directory with `sam3d_params.{json,bin}`.
    pub params_dir: PathBuf,
    /// Detection confidence.
    pub confidence: f32,
    /// Decode width; frames are scaled before detection.
    pub work_width: usize,
    /// Stop after this many frames (0 = whole clip).
    pub max_frames: usize,
    /// Tracks shorter than this are dropped.
    pub min_hits: usize,
    /// Frames a track may coast without a detection.
    ///
    /// The reference's ByteTrack runs `track_buffer=60` (`person_tracking.py`) and
    /// `with_reid`, i.e. two seconds at 30 fps, and that is what keeps a body
    /// across a doorway/occlusion. Ours coasted 8 frames (0.27 s), so any longer
    /// dropout ended the track: the 24 fps source — resampled to 30 fps, where a
    /// blurred source frame becomes one and a quarter output frames — lost the
    /// person near the end, the take stopped there, and the viewport went empty.
    pub max_missing: usize,
    /// Prefer the CUDA execution provider.
    pub gpu: bool,
    /// Most people reconstructed per frame.
    pub max_people: usize,
    /// Ignore boxes shorter than this fraction of the frame height.
    ///
    /// A wide sports shot contains dozens of 20-pixel spectators; running the
    /// full chain on them costs seconds and produces nothing usable.
    pub min_box_height: f32,
    /// Stop storing meshes past this many bytes (they still stream live).
    pub mesh_budget: usize,
    /// Optional TorchScript module for the whole SAM pipeline
    /// (`tools/torch/sam3d_full_jit.py`). When it is set, exists, and the build
    /// has LibTorch, the person pass runs through it instead of the ONNX chain —
    /// hands and fusion included, at ~1.1 s/frame on a 5060 Ti.
    pub torch_weights: Option<PathBuf>,
    /// Boxes the user picked after the detection pass. When non-empty, only the
    /// tracks that start on one of these boxes are reconstructed, so a clip is
    /// not rebuilt for bystanders the user did not ask for.
    pub seed_boxes: Vec<[f32; 4]>,
}

impl Default for SamConfig {
    fn default() -> Self {
        Self {
            detector: PathBuf::from("models/detector/yolov8x.onnx"),
            sam3d_dir: PathBuf::from("models/sam3d"),
            params_dir: PathBuf::from("models/flowhmr/sam"),
            confidence: 0.3,
            work_width: 960,
            max_frames: 0,
            min_hits: 3,
            max_missing: 60,
            gpu: true,
            max_people: 16,
            min_box_height: 0.12,
            mesh_budget: 900 * 1024 * 1024,
            torch_weights: None,
            seed_boxes: Vec::new(),
        }
    }
}

/// One person in one frame, already placed in world space.
#[derive(Clone, Debug)]
pub struct PersonFrame {
    /// Detection box in work-resolution pixels.
    pub bbox: [f32; 4],
    pub score: f32,
    /// `18439 × 3` vertices, metres, y-up (`x` right, `y` up, `z` forward).
    pub verts: Vec<f32>,
    /// `127 × 3` joint centres, same frame.
    pub joints: Vec<f32>,
    /// `70 × 3` named keypoints, same frame.
    pub keypoints: Vec<f32>,
    /// Body token for the motion stage (`1024`).
    pub body_token: Vec<f32>,
    /// Left-hand token (`1024`), empty when the backend does not run the hands.
    pub left_token: Vec<f32>,
    /// Right-hand token (`1024`), same rule.
    pub right_token: Vec<f32>,
    /// Camera-space translation used for placement.
    pub cam_t: [f32; 3],
    /// MHR hips in camera coordinates, metres — FlowHMR's placement anchor.
    pub anchor_cam: [f32; 3],
    /// `127 × 8` joint translations plus global quaternions, straight from MHR.
    pub skel_state: Vec<f32>,
}

impl PersonFrame {
    /// Bytes of mesh data this frame holds.
    pub fn mesh_bytes(&self) -> usize {
        self.verts.len() * 4
    }
}

/// One tracked person across the clip.
#[derive(Clone, Debug)]
pub struct PersonTrack {
    /// Tracker id, stable across the run.
    pub id: usize,
    /// `Person 1`, `Person 2`, … in order of first appearance.
    pub name: String,
    /// One slot per decoded frame; `None` while the person was not observed.
    pub frames: Vec<Option<Arc<PersonFrame>>>,
    pub first_frame: usize,
    pub last_frame: usize,
    pub hits: usize,
}

impl PersonTrack {
    pub fn frame(&self, index: usize) -> Option<&Arc<PersonFrame>> {
        self.frames.get(index).and_then(|slot| slot.as_ref())
    }

    /// Nearest stored frame at or before `index`, for coasting playback.
    pub fn nearest(&self, index: usize) -> Option<(usize, &Arc<PersonFrame>)> {
        (0..=index.min(self.frames.len().saturating_sub(1)))
            .rev()
            .find_map(|at| self.frames.get(at).and_then(|slot| slot.as_ref()).map(|f| (at, f)))
    }
}

#[derive(Clone, Debug, Default)]
pub struct Timings {
    pub detect: f64,
    pub reconstruct: f64,
    pub total: f64,
}

impl Timings {
    pub fn line(&self) -> String {
        format!(
            "detect {:.2}s · reconstruct {:.2}s · total {:.2}s",
            self.detect, self.reconstruct, self.total
        )
    }
}

/// Everything the capture produced.
pub struct PeopleCapture {
    pub people: Vec<PersonTrack>,
    pub mesh: Arc<MeshTopology>,
    pub fps: f64,
    pub width: usize,
    pub height: usize,
    pub frames: usize,
    pub backend: String,
    pub timings: Timings,
    /// Frames whose mesh was not retained because the budget was reached.
    pub dropped_meshes: usize,
    /// Foot contacts pinned by the cleanup pass.
    pub cleanup: crate::cleanup::CleanupReport,
}

impl PeopleCapture {
    pub fn summary(&self) -> String {
        let hits: usize = self.people.iter().map(|p| p.hits).sum();
        let feet = self.cleanup.line();
        let feet = if feet.is_empty() {
            String::new()
        } else {
            format!(" · {feet}")
        };
        let meshes: usize = self
            .people
            .iter()
            .map(|p| p.frames.iter().flatten().count())
            .sum();
        format!(
            "{} people · {} person-frames · {} meshes @ {:.2} fps{}",
            self.people.len(),
            hits,
            meshes,
            self.fps,
            if self.dropped_meshes > 0 {
                format!(" · {} dropped (mesh budget)", self.dropped_meshes)
            } else {
                String::new()
            },
        ) + &feet
    }
}

/// Callbacks the application uses to follow a run.
pub struct SamHooks<'a> {
    /// `(frame, total, stage)`.
    pub progress: &'a mut dyn FnMut(usize, usize, &str),
    /// The decoded frame, for the video preview.
    pub preview: &'a mut dyn FnMut(&Frame),
    /// All tracks so far, after each frame, for the live 3D view.
    pub people: &'a mut dyn FnMut(usize, &[PersonTrack]),
    /// Polled once per frame; returning `true` stops the run.
    pub cancelled: &'a dyn Fn() -> bool,
}


/// Capture every person in `video`.
pub fn capture_people(video: impl AsRef<Path>, config: &SamConfig) -> Result<PeopleCapture> {
    let mut progress = |_: usize, _: usize, _: &str| {};
    let mut preview = |_: &Frame| {};
    let mut people = |_: usize, _: &[PersonTrack]| {};
    let cancelled = || false;
    capture_people_with(
        video,
        config,
        &mut SamHooks {
            progress: &mut progress,
            preview: &mut preview,
            people: &mut people,
            cancelled: &cancelled,
        },
    )
}

/// Which SAM backend a capture runs.
///
/// Both produce a [`velomocap_infer::Sam3dFrame`], so every consumer downstream
/// (placement, keypoint regression, the motion stage) is unchanged.
enum SamRunner {
    /// The eleven ONNX graphs driven from Rust.
    Onnx(std::cell::RefCell<Box<Sam3dBody>>),
    /// One TorchScript module: `run_inference(inference_type="full")`, hands and
    /// fusion included. See `docs/SAM3D.md`.
    Torch(Box<velomocap_cpp::sam3d::Sam3dTorch>),
}

impl SamRunner {
    fn backend(&self) -> String {
        match self {
            Self::Onnx(body) => body.borrow().backend().as_str().to_string(),
            Self::Torch(_) => "torchscript".to_string(),
        }
    }

    fn run(
        &self,
        frame: &Frame,
        bbox: [f32; 4],
        intrinsics: [[f32; 3]; 3],
    ) -> Result<velomocap_infer::Sam3dFrame> {
        match self {
            Self::Onnx(body) => {
                let input =
                    Sam3dInput::from_box(&frame.rgb, frame.width, frame.height, bbox, intrinsics)
                        .context("building the SAM crop")?;
                body.borrow_mut().run(&input).context("SAM 3D Body")
            }
            Self::Torch(module) => {
                let (width, height) = (frame.width, frame.height);
                let mut nchw = vec![0.0f32; 3 * width * height];
                let plane = width * height;
                for (index, pixel) in frame.rgb.chunks_exact(3).enumerate() {
                    nchw[index] = pixel[0] as f32 / 255.0;
                    nchw[plane + index] = pixel[1] as f32 / 255.0;
                    nchw[2 * plane + index] = pixel[2] as f32 / 255.0;
                }
                let (center, scale) = person_box(bbox);
                let mut cam_int = [0.0f32; 9];
                for row in 0..3 {
                    for column in 0..3 {
                        cam_int[row * 3 + column] = intrinsics[row][column];
                    }
                }
                let out = module
                    .run_full(&velomocap_cpp::sam3d::Sam3dFullInput {
                        frame: &nchw,
                        height,
                        width,
                        bbox_center: center,
                        bbox_scale: scale,
                        cam_int,
                    })
                    .map_err(|error| anyhow::anyhow!("SAM 3D Body (TorchScript): {error}"))?;
                let verts = out.body_space_verts();
                Ok(velomocap_infer::Sam3dFrame {
                    body_token: out.body_token,
                    left_token: out.left_token,
                    right_token: out.right_token,
                    // `Sam3dFrame::world_vertices` expects centimetres with `y`
                    // down and `z` forward, and subtracts `cam_t.y`. The module
                    // hands back metres in camera axes with `y` up, so the
                    // translation's `y` is mirrored alongside the mesh.
                    verts,
                    skel_state: out.skel_state,
                    model_params: out.model_params,
                    pred_cam: out.pred_cam,
                    cam_t: [out.pred_cam_t[0], -out.pred_cam_t[1], out.pred_cam_t[2]],
                    pred: out.pred,
                })
            }
        }
    }
}

/// Intersection over union of two `[x0, y0, x1, y1]` boxes.
fn box_iou(first: [f32; 4], second: [f32; 4]) -> f32 {
    let x0 = first[0].max(second[0]);
    let y0 = first[1].max(second[1]);
    let x1 = first[2].min(second[2]);
    let y1 = first[3].min(second[3]);
    let inter = (x1 - x0).max(0.0) * (y1 - y0).max(0.0);
    let area = |b: [f32; 4]| (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let union = area(first) + area(second) - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// The `(center, scale)` a person box becomes after `GetBBoxCenterScale(1.25)`
/// and the two `fix_aspect_ratio` calls in `TopdownAffine` — i.e. what the
/// reference feeds `prepare_batch`, which the TorchScript module reproduces.
fn person_box(bbox: [f32; 4]) -> ([f32; 2], [f32; 2]) {
    let center = [(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5];
    let mut width = (bbox[2] - bbox[0]) * 1.25;
    let mut height = (bbox[3] - bbox[1]) * 1.25;
    // fix_aspect_ratio(width / height = 0.75)
    if width > height * 0.75 {
        height = width / 0.75;
    } else {
        width = height * 0.75;
    }
    // ... then to the model's square input.
    if width > height {
        width = height;
    } else {
        height = width;
    }
    (center, [width, height])
}

/// The full loop, with hooks for the UI.
/// IoU of a detection with a plain box: the "is this the same body?" test used to
/// follow a picked person across a track-id change.
fn overlap(a: &Detection, b: [f32; 4]) -> f32 {
    let x0 = a.x0.max(b[0]);
    let y0 = a.y0.max(b[1]);
    let x1 = a.x1.min(b[2]);
    let y1 = a.y1.min(b[3]);
    let inter = (x1 - x0).max(0.0) * (y1 - y0).max(0.0);
    let union = a.area() + (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0) - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

pub fn capture_people_with(
    video: impl AsRef<Path>,
    config: &SamConfig,
    hooks: &mut SamHooks<'_>,
) -> Result<PeopleCapture> {
    let started = Instant::now();
    let video = video.as_ref();
    let tools = Tools::discover()?;

    (hooks.progress)(0, 0, "loading mesh");
    let mesh = Arc::new(
        MeshTopology::load(&config.sam3d_dir)
            .with_context(|| format!("loading MHR topology from {}", config.sam3d_dir.display()))?,
    );

    (hooks.progress)(0, 0, "loading detector");
    let session = SessionConfig {
        gpu: config.gpu,
        ..Default::default()
    };
    let mut detector = Detector::load(
        &config.detector,
        DetectorConfig {
            confidence: config.confidence,
            session: session.clone(),
            ..Default::default()
        },
    )
    .with_context(|| format!("loading detector {}", config.detector.display()))?;

    (hooks.progress)(0, 0, "loading SAM 3D Body");
    // `torch_weights`, then `VELOMOCAP_SAM_TORCH`, then a module dropped next to
    // the ONNX graphs. Only used when this build linked LibTorch.
    // `VELOMOCAP_NO_SAM_TORCH=1` forces the ONNX chain (to exercise the fallback
    // and to A/B the two backends).
    let torch_weights = if std::env::var_os("VELOMOCAP_NO_SAM_TORCH").is_some() {
        None
    } else {
        config
            .torch_weights
            .clone()
            .or_else(|| {
                std::env::var_os("VELOMOCAP_SAM_TORCH")
                    .map(PathBuf::from)
                    .filter(|path| path.is_file())
            })
            .or_else(|| {
                let candidate = config.sam3d_dir.join("sam3d_full_ts.pt");
                if candidate.is_file() {
                    return Some(candidate);
                }
                // Dev tree: `tools/torch/sam3d_full_jit.py` writes into
                // `export/sam3d/`, next to the workspace rather than inside
                // `models/`, so the 3.5 GB file does not have to be copied.
                let exported = Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../../export/sam3d/sam3d_full_ts.pt");
                exported.is_file().then_some(exported)
            })
    };
    let body = match torch_weights.as_deref() {
        Some(path) if velomocap_cpp::sam3d::Sam3dTorch::built() => {
            match velomocap_cpp::sam3d::Sam3dTorch::load(path, config.gpu) {
                Ok(module) => {
                    eprintln!("[SAM] using {} through LibTorch", path.display());
                    SamRunner::Torch(Box::new(module))
                }
                Err(error) => {
                    eprintln!("[SAM] TorchScript load failed ({error}); falling back to ONNX");
                    SamRunner::Onnx(std::cell::RefCell::new(Box::new(
                        Sam3dBody::load(&config.sam3d_dir, &config.params_dir, &session)
                            .with_context(|| {
                                format!(
                                    "loading the SAM 3D Body chain from {}",
                                    config.sam3d_dir.display()
                                )
                            })?,
                    )))
                }
            }
        }
        _ => SamRunner::Onnx(std::cell::RefCell::new(Box::new(
            Sam3dBody::load(&config.sam3d_dir, &config.params_dir, &session).with_context(|| {
                format!(
                    "loading the SAM 3D Body chain from {}",
                    config.sam3d_dir.display()
                )
            })?,
        ))),
    };
    let backend = body.backend();

    (hooks.progress)(0, 0, "opening video");
    // The reference re-encodes the clip to a constant 30 fps before anything
    // touches it (`video.py`), so every stage below indexes a 30 fps stream.
    let mut decoder = Decoder::open_flow(&tools, video, Some(config.work_width))?;
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
    let intrinsics = default_intrinsics(width, height);
    let mut tracker = Tracker::new(0.3, config.max_missing);
    let mut seeded_ids: Vec<usize> = Vec::new();

    let mut people: Vec<PersonTrack> = Vec::new();
    let mut timings = Timings::default();
    let mut meshes_stored = 0usize;
    let mut dropped_meshes = 0usize;
    let mut decoded = 0usize;
    // The last box the picked person was seen in. A dropout longer than
    // `max_missing` gives the same body a new track id, and following the *id*
    // alone ends the take where the id changed: the viewport then has no frames
    // to draw and the character disappears while the video still moves.
    let mut last_box: Option<[f32; 4]> = None;

    while let Some(frame) = decoder.next_frame()? {
        if (hooks.cancelled)() {
            break;
        }
        let index = decoded;
        decoded += 1;
        (hooks.preview)(&frame);

        (hooks.progress)(decoded, total, "detecting");
        let detect_start = Instant::now();
        let detections = detector.detect(&frame.rgb, frame.width, frame.height)?;
        timings.detect += detect_start.elapsed().as_secs_f64();
        tracker.update(&detections);

        // Restrict to the people the user picked: on the first frame keep only
        // detections that overlap a seed box, then only those track ids.
        if !config.seed_boxes.is_empty() && index == 0 {
            seeded_ids = tracker
                .tracks
                .iter()
                .filter(|track| {
                    track
                        .boxes
                        .get(index)
                        .and_then(|slot| *slot)
                        .map(|detection| {
                            config
                                .seed_boxes
                                .iter()
                                .any(|seed| overlap(&detection, *seed) > 0.2)
                        })
                        .unwrap_or(false)
                })
                .map(|track| track.id)
                .collect();
            if std::env::var_os("VELOMOCAP_DEBUG_SEEDS").is_some() {
                eprintln!(
                    "[seeds] {} detections, {} tracks, {} seeded, first detection {:?}",
                    detections.len(),
                    tracker.tracks.len(),
                    seeded_ids.len(),
                    detections.first().map(|d| (d.x0, d.y0, d.x1, d.y1))
                );
                for seed in &config.seed_boxes {
                    eprintln!("[seeds] seed {:?}", seed);
                }
            }
        }

        let mut candidates: Vec<(usize, Detection)> = tracker
            .tracks
            .iter()
            .filter(|track| track.missing <= config.max_missing)
            .filter(|track| {
                if config.seed_boxes.is_empty() || seeded_ids.contains(&track.id) {
                    return true;
                }
                // Same body, new id: accept it when it continues the box the person
                // was last seen in — overlapping it, or with its centre within
                // about half a body of it (the box moves while the track was lost).
                let Some(previous) = last_box else {
                    return false;
                };
                track
                    .boxes
                    .get(index)
                    .and_then(|slot| *slot)
                    .map_or(false, |detection| {
                        let center = [
                            0.5 * (detection.x0 + detection.x1),
                            0.5 * (detection.y0 + detection.y1),
                        ];
                        let before = [
                            0.5 * (previous[0] + previous[2]),
                            0.5 * (previous[1] + previous[3]),
                        ];
                        let scale = (previous[2] - previous[0])
                            .max(previous[3] - previous[1])
                            .max(1.0);
                        let distance = ((center[0] - before[0]).powi(2)
                            + (center[1] - before[1]).powi(2))
                        .sqrt();
                        distance < 0.6 * scale || overlap(&detection, previous) > 0.15
                    })
            })
            .filter_map(|track| {
                track
                    .boxes
                    .get(index)
                    .and_then(|slot| *slot)
                    .map(|detection| (track.id, detection))
            })
            .collect();
        candidates.sort_by(|a, b| {
            b.1.area()
                .partial_cmp(&a.1.area())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.retain(|(_, detection)| {
            detection.height() >= config.min_box_height * frame.height as f32
        });
        candidates.truncate(config.max_people);

        // Follow the body, not the id: adopt whichever track continues the box the
        // person was last seen in, so a re-detection after a dropout keeps the take
        // going instead of ending it (which is what leaves the viewport empty while
        // the video still moves).
        if !config.seed_boxes.is_empty() {
            if let Some((id, detection)) = candidates.first() {
                if !seeded_ids.contains(id) {
                    seeded_ids.push(*id);
                }
                last_box = Some([detection.x0, detection.y0, detection.x1, detection.y1]);
            }
        }

        for (track_id, detection) in candidates {
            (hooks.progress)(decoded, total, "reconstructing");
            let slot = match people.iter().position(|person| person.id == track_id) {
                Some(slot) => slot,
                None => {
                    people.push(PersonTrack {
                        id: track_id,
                        name: format!("Person {}", people.len() + 1),
                        frames: Vec::new(),
                        first_frame: index,
                        last_frame: index,
                        hits: 0,
                    });
                    people.len() - 1
                }
            };
            let track = &mut people[slot];
            while track.frames.len() <= index {
                track.frames.push(None);
            }

            let reconstruct_start = Instant::now();
            let output = body
                .run(&frame, [detection.x0, detection.y0, detection.x1, detection.y1], intrinsics)
                .context("SAM 3D Body")?;
            timings.reconstruct += reconstruct_start.elapsed().as_secs_f64();

            let verts = output.world_vertices();
            let joints = output.world_joints();
            let mut keypoints = Vec::new();
            mesh.regressor.apply(&verts, &joints, &mut keypoints);

            // The reference places FlowHMR's world by matching the MHR hips in
            // *camera* coordinates: `pred_keypoints_3d[[9, 10]] + pred_cam_t`.
            // `verts` above is already translated and y-up, so the camera-space
            // keypoints are regressed separately from the raw chain output
            // (centimetres, y down, z forward).
            let mut camera_joints = Vec::with_capacity(output.skel_state.len() / 8 * 3);
            for joint in output.skel_state.chunks_exact(8) {
                camera_joints.extend_from_slice(&joint[..3]);
            }
            let mut camera_keypoints = Vec::new();
            mesh.regressor
                .apply(&output.verts, &camera_joints, &mut camera_keypoints);
            let mut anchor_cam = [0.0f32; 3];
            if camera_keypoints.len() >= 11 * 3 {
                for axis in 0..3 {
                    anchor_cam[axis] = 0.5
                        * (camera_keypoints[9 * 3 + axis] + camera_keypoints[10 * 3 + axis])
                        * 0.01
                        + output.cam_t[axis];
                }
            }

            let person_frame = PersonFrame {
                bbox: [detection.x0, detection.y0, detection.x1, detection.y1],
                score: detection.score,
                verts,
                joints,
                keypoints,
                body_token: output.body_token,
                left_token: output.left_token,
                right_token: output.right_token,
                cam_t: output.cam_t,
                anchor_cam,
                skel_state: output.skel_state,
            };

            if meshes_stored + person_frame.mesh_bytes() <= config.mesh_budget {
                meshes_stored += person_frame.mesh_bytes();
                track.frames[index] = Some(Arc::new(person_frame));
            } else {
                dropped_meshes += 1;
            }
            track.last_frame = index;
            track.hits += 1;
        }

        // Keep every other track's frame list aligned with the clip.
        for track in people.iter_mut() {
            while track.frames.len() <= index {
                track.frames.push(None);
            }
        }

        (hooks.people)(index, &people);
        if config.max_frames > 0 && decoded >= config.max_frames {
            break;
        }
    }

    // Planted feet stay planted, and the body stops jittering frame to frame.
    (hooks.progress)(decoded, total, "stabilising");
    let cleanup = crate::cleanup::stabilise(&mut people, &crate::cleanup::CleanupConfig::default());

    // Stable ordering: by first appearance, then by id. Names were assigned on
    // first sight, so a retired track never renumbers the survivors.
    people.sort_by_key(|person| (person.first_frame, person.id));
    // `min_hits` exists to drop one-frame false positives, but on a short run it
    // would drop *everything*: a 2-frame capture has no track with 3 hits, so
    // the reconstruction came back empty and the viewport stayed blank. Scale
    // the floor down to the frames actually decoded.
    let floor = config.min_hits.max(1).min(decoded.max(1));
    people.retain(|person| person.hits >= floor);

    // Collapse duplicate tracks. When the tracker loses someone for longer than
    // `max_missing` it starts a second track for the same body; both survive the
    // filter above and the viewport then draws two overlapping figures that
    // flicker in and out - which reads as "people who were there earlier"
    // appearing in the middle of the clip. Keep the track that saw the person
    // most often.
    let mut alive = vec![true; people.len()];
    for a in 0..people.len() {
        for b in (a + 1)..people.len() {
            if !alive[a] || !alive[b] {
                continue;
            }
            let mut total = 0.0f32;
            let mut shared = 0usize;
            for (first, second) in people[a].frames.iter().zip(people[b].frames.iter()) {
                let (Some(first), Some(second)) = (first, second) else {
                    continue;
                };
                let overlap = box_iou(first.bbox, second.bbox);
                total += overlap;
                shared += 1;
            }
            if shared >= 2 && total / shared as f32 > 0.5 {
                let drop = if people[a].hits >= people[b].hits { b } else { a };
                alive[drop] = false;
            }
        }
    }
    let mut keep = alive.into_iter();
    people.retain(|_| keep.next().unwrap_or(true));

    timings.total = started.elapsed().as_secs_f64();
    (hooks.progress)(decoded, total, "done");

    Ok(PeopleCapture {
        people,
        mesh,
        fps,
        width,
        height,
        frames: decoded,
        backend,
        timings,
        dropped_meshes,
        cleanup,
    })
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

    /// The integration bench: every person in the sample clip, with the
    /// per-person cost reported. `cargo test -p velomocap-motion sam:: -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn reconstructs_every_person_in_the_sample_clip() {
        let root = repo_root();
        let video = root.join("build/media/test-clip.mp4");
        if !video.is_file() {
            eprintln!("skipping: no {}", video.display());
            return;
        }
        ensure_runtime();
        let config = SamConfig {
            detector: root.join("models/detector/yolov8x.onnx"),
            sam3d_dir: root.join("models/sam3d"),
            params_dir: root.join("models/flowhmr/sam"),
            max_frames: 4,
            ..Default::default()
        };
        let started = Instant::now();
        let capture = capture_people(&video, &config).expect("capture");
        let elapsed = started.elapsed().as_secs_f64();
        let person_frames: usize = capture.people.iter().map(|p| p.hits).sum();
        eprintln!(
            "{} people, {} person-frames in {:.2}s ({:.1} ms each), backend {}",
            capture.people.len(),
            person_frames,
            elapsed,
            elapsed * 1000.0 / person_frames.max(1) as f64,
            capture.backend
        );
        for person in &capture.people {
            let frame = person.frames.iter().flatten().next().expect("a frame");
            let vertices = frame.verts.len() / 3;
            assert_eq!(vertices, 18439, "vertex count");
            assert_eq!(frame.joints.len(), 127 * 3, "joint count");
            assert_eq!(frame.keypoints.len(), 70 * 3, "keypoint count");
            assert_eq!(frame.body_token.len(), 1024, "token width");
            let depth = frame.cam_t[2];
            eprintln!(
                "  {} id={} frames={} depth {:.2} m, first keypoint {:?}",
                person.name,
                person.id,
                person.hits,
                depth,
                &frame.keypoints[0..3]
            );
            assert!(depth > 0.1 && depth < 100.0, "implausible depth {depth}");
        }
    }
}
