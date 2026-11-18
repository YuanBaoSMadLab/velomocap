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
    /// Boxes the user picked after the detection pass. When non-empty, the cast is
    /// exactly these people: one seed answers to one track on the first frame,
    /// follows that body if the tracker re-numbers it, and its frames always land
    /// in the same `PersonTrack`. Bystanders are never reconstructed, and one tick
    /// can never produce two people.
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

/// One track per seed box, by overlap.
///
/// The seed boxes are the people the user ticked, so each one answers to exactly
/// one track: a box that overlaps two people (or two tracks for one body) must
/// not double the cast. A seed with no detection is simply absent for now — the
/// continuation clause looks for it again on later frames.
fn seed_assignments(detections: &[(usize, [f32; 4])], seeds: &[[f32; 4]]) -> Vec<Option<usize>> {
    seeds
        .iter()
        .map(|seed| {
            let mut best: Option<(usize, f32)> = None;
            for (id, box_) in detections {
                let score = box_overlap(*box_, *seed);
                if score <= 0.2 {
                    continue;
                }
                if best.map_or(true, |(_, top)| score > top) {
                    best = Some((*id, score));
                }
            }
            best.map(|(id, _)| id)
        })
        .collect()
}

/// What the per-frame selection needs to know about one tracker track.
#[derive(Clone, Copy, Debug)]
struct Seen {
    id: usize,
    /// Consecutive frames without a detection, as the tracker counts it.
    missing: usize,
    /// Box in this frame; `None` while the track coasts.
    detection: Option<Detection>,
}

/// One picked person's identity for the whole take.
#[derive(Clone, Debug)]
struct Pick {
    /// Track id currently answering to this person, if any.
    track: Option<usize>,
    /// The box the person was last seen in.
    last_box: [f32; 4],
}

/// Which picked person each track belongs to this frame.
///
/// One seed is one person, for the whole take:
///
/// * While a pick's track *coasts* (`missing <= max_missing`) the identity is
///   held. It is not up for grabs: a bystander detected in the same frame used to
///   be accepted as "the same body", which is how one selected person grew into
///   two and three people half-way through a take.
/// * Only once the tracker has retired that track (`missing > max_missing`, so it
///   can never come back) is the body looked for again under a new id — and then
///   with `continues`, i.e. the box the person was last seen in.
/// * A track already answering to another pick cannot be taken over.
///
/// Returns `(seed, track, detection)` for the picks seen this frame.
fn select_seeded(seen: &[Seen], picks: &mut [Pick], max_missing: usize) -> Vec<(usize, usize, Detection)> {
    let mut out = Vec::new();
    let mut claimed: Vec<usize> = picks.iter().filter_map(|pick| pick.track).collect();

    for (seed, pick) in picks.iter_mut().enumerate() {
        let Some(id) = pick.track else { continue };
        match seen.iter().find(|track| track.id == id) {
            Some(track) if track.detection.is_some() => {
                let detection = track.detection.unwrap();
                pick.last_box = [detection.x0, detection.y0, detection.x1, detection.y1];
                out.push((seed, id, detection));
            }
            // Coasting: the tracker still expects this body back, so nobody else
            // may step into its place.
            Some(track) if track.missing <= max_missing => {}
            // Retired: the same body, if it is still in shot, now has a new id.
            _ => {
                pick.track = None;
                claimed.retain(|id_| *id_ != id);
            }
        }
    }

    for (seed, pick) in picks.iter_mut().enumerate() {
        if pick.track.is_some() {
            continue;
        }
        let mut best: Option<(usize, f32, Detection)> = None;
        for track in seen {
            let Some(detection) = track.detection else { continue };
            if claimed.contains(&track.id) {
                continue;
            }
            if !continues(pick.last_box, detection) {
                continue;
            }
            let score = overlap(&detection, pick.last_box);
            if best.map_or(true, |(_, top, _)| score > top) {
                best = Some((track.id, score, detection));
            }
        }
        if let Some((id, _, detection)) = best {
            claimed.push(id);
            pick.track = Some(id);
            pick.last_box = [detection.x0, detection.y0, detection.x1, detection.y1];
            out.push((seed, id, detection));
        }
    }

    out
}

/// The single `PersonTrack` a picked seed's frames belong to, created on first
/// sight. A track-id change lands in the same slot, so one ticked person stays one
/// reconstructed person however many times the tracker re-numbers them.
fn seed_slot(
    slots: &mut Vec<Option<usize>>,
    people: &mut Vec<PersonTrack>,
    seed: usize,
    track: usize,
    index: usize,
) -> usize {
    if slots.len() <= seed {
        slots.resize(seed + 1, None);
    }
    if let Some(slot) = slots[seed] {
        return slot;
    }
    people.push(PersonTrack {
        id: track,
        name: format!("Person {}", people.len() + 1),
        frames: Vec::new(),
        first_frame: index,
        last_frame: index,
        hits: 0,
    });
    let slot = people.len() - 1;
    slots[seed] = Some(slot);
    slot
}

/// Does a detection continue the box a picked person was last seen in?
fn continues(previous: [f32; 4], detection: Detection) -> bool {
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
    let distance =
        ((center[0] - before[0]).powi(2) + (center[1] - before[1]).powi(2)).sqrt();
    distance < 0.6 * scale || overlap(&detection, previous) > 0.15
}

/// IoU of two plain boxes.
fn box_overlap(a: [f32; 4], b: [f32; 4]) -> f32 {
    let inter = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0)
        * (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let union = area_a + area_b - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
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

    let mut people: Vec<PersonTrack> = Vec::new();
    let mut timings = Timings::default();
    let mut meshes_stored = 0usize;
    let mut dropped_meshes = 0usize;
    let mut decoded = 0usize;
    // One identity per picked person, and one `PersonTrack` slot per identity. A
    // dropout longer than `max_missing` gives the same body a new track id; the
    // pick keeps following the *body*, and its frames keep landing in the same
    // slot, so the cast cannot grow past the number of seeds.
    let mut picks: Vec<Pick> = Vec::new();
    let mut slots: Vec<Option<usize>> = Vec::new();

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

        // Every picked person answers to exactly one identity for the whole take:
        // `picks[seed]` follows that body across track-id changes, `slots[seed]` is
        // the one `PersonTrack` its frames land in. The cast is therefore fixed by
        // the seeds before a single mesh is run.
        let seen: Vec<Seen> = tracker
            .tracks
            .iter()
            .map(|track| Seen {
                id: track.id,
                missing: track.missing,
                detection: track.boxes.get(index).and_then(|slot| *slot),
            })
            .collect();

        let mut jobs: Vec<(usize, Detection)> = Vec::new();
        if !config.seed_boxes.is_empty() {
            if index == 0 {
                let boxes: Vec<(usize, [f32; 4])> = seen
                    .iter()
                    .filter_map(|track| {
                        track
                            .detection
                            .map(|d| (track.id, [d.x0, d.y0, d.x1, d.y1]))
                    })
                    .collect();
                let assigned = seed_assignments(&boxes, &config.seed_boxes);
                picks = assigned
                    .iter()
                    .enumerate()
                    .take(config.max_people.max(1))
                    .map(|(seed, id)| Pick {
                        track: *id,
                        last_box: id
                            .and_then(|id| {
                                boxes
                                    .iter()
                                    .find(|(track, _)| *track == id)
                                    .map(|(_, box_)| *box_)
                            })
                            .unwrap_or(config.seed_boxes[seed]),
                    })
                    .collect();
                if std::env::var_os("VELOMOCAP_DEBUG_SEEDS").is_some() {
                    let seeded: Vec<usize> = assigned.iter().flatten().copied().collect();
                    eprintln!(
                        "[seeds] {} detections, {} tracks, {} seeded {:?}, first detection {:?}",
                        detections.len(),
                        tracker.tracks.len(),
                        seeded.len(),
                        seeded,
                        detections.first().map(|d| (d.x0, d.y0, d.x1, d.y1))
                    );
                }
            }
            for (seed, track_id, detection) in select_seeded(&seen, &mut picks, config.max_missing) {
                if detection.height() < config.min_box_height * frame.height as f32 {
                    continue;
                }
                let slot = seed_slot(&mut slots, &mut people, seed, track_id, index);
                jobs.push((slot, detection));
            }
        } else {
            let mut candidates: Vec<(usize, Detection)> = seen
                .iter()
                .filter(|track| track.missing <= config.max_missing)
                .filter_map(|track| track.detection.map(|d| (track.id, d)))
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
            for (track_id, detection) in candidates {
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
                jobs.push((slot, detection));
            }
        }

        for (slot, detection) in jobs {
            (hooks.progress)(decoded, total, "reconstructing");
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

#[cfg(test)]
mod selection_tests {
    use super::*;

    fn detection(box_: [f32; 4]) -> Detection {
        Detection {
            x0: box_[0],
            y0: box_[1],
            x1: box_[2],
            y1: box_[3],
            score: 1.0,
            class_id: 0,
        }
    }

    /// One ticked person is one track, even when a box on screen overlaps two
    /// of them: the seed decides the cast, the overlap only decides which body.
    #[test]
    fn one_seed_answers_to_one_track() {
        let detections = [
            (7, [100.0, 100.0, 200.0, 400.0]),
            (9, [120.0, 110.0, 210.0, 400.0]),
        ];
        let seeds = [[105.0, 100.0, 205.0, 400.0]];
        let ids = seed_assignments(&detections, &seeds);
        assert_eq!(ids.len(), 1, "one seed, one track: {ids:?}");
        assert_eq!(ids[0], Some(7), "the better overlap wins");
    }

    #[test]
    fn two_seeds_answer_to_two_tracks() {
        let detections = [
            (7, [100.0, 100.0, 200.0, 400.0]),
            (9, [400.0, 100.0, 500.0, 400.0]),
        ];
        let seeds = [
            [100.0, 100.0, 200.0, 400.0],
            [400.0, 100.0, 500.0, 400.0],
        ];
        assert_eq!(seed_assignments(&detections, &seeds), vec![Some(7), Some(9)]);
    }

    /// A seed whose body has not been detected yet seeds nothing, rather than
    /// borrowing somebody else's track.
    #[test]
    fn a_seed_with_no_overlap_seeds_nothing() {
        let detections = [(7, [0.0, 0.0, 10.0, 10.0])];
        let seeds = [[500.0, 500.0, 600.0, 700.0]];
        assert_eq!(seed_assignments(&detections, &seeds), vec![None]);
    }

    /// The continuation follows the body, so a box that is clearly a different
    /// person standing beside the picked one does not qualify.
    #[test]
    fn a_neighbour_does_not_continue_the_picked_box() {
        let picked = [100.0, 100.0, 200.0, 400.0];
        assert!(continues(picked, detection([102.0, 102.0, 202.0, 402.0])));
        assert!(!continues(picked, detection([300.0, 100.0, 400.0, 400.0])));
    }

    /// One frame of a scripted run: `(track id, missing, box)` per track.
    type Step = Vec<(usize, usize, Option<[f32; 4]>)>;

    fn seen_of(step: &Step) -> Vec<Seen> {
        step.iter()
            .map(|(id, missing, box_)| Seen {
                id: *id,
                missing: *missing,
                detection: box_.map(detection),
            })
            .collect()
    }

    /// Drive the seeded selection over a scripted run the way the capture loop
    /// does — identity resolution, then one person slot per seed — and report the
    /// cast and the frames each person got.
    fn run_seeded(frames: &[Step], seeds: &[[f32; 4]]) -> (Vec<PersonTrack>, Vec<Vec<usize>>) {
        let mut picks: Vec<Pick> = Vec::new();
        let mut slots: Vec<Option<usize>> = Vec::new();
        let mut people: Vec<PersonTrack> = Vec::new();
        let mut got: Vec<Vec<usize>> = Vec::new();
        for (index, step) in frames.iter().enumerate() {
            let seen = seen_of(step);
            if index == 0 {
                let boxes: Vec<(usize, [f32; 4])> = seen
                    .iter()
                    .filter_map(|track| {
                        track
                            .detection
                            .map(|d| (track.id, [d.x0, d.y0, d.x1, d.y1]))
                    })
                    .collect();
                picks = seed_assignments(&boxes, seeds)
                    .iter()
                    .enumerate()
                    .map(|(seed, id)| Pick {
                        track: *id,
                        last_box: id
                            .and_then(|id| {
                                boxes
                                    .iter()
                                    .find(|(track, _)| *track == id)
                                    .map(|(_, box_)| *box_)
                            })
                            .unwrap_or(seeds[seed]),
                    })
                    .collect();
            }
            for (seed, track_id, _) in select_seeded(&seen, &mut picks, 60) {
                let slot = seed_slot(&mut slots, &mut people, seed, track_id, index);
                while got.len() < people.len() {
                    got.push(Vec::new());
                }
                got[slot].push(index);
                people[slot].hits += 1;
            }
        }
        (people, got)
    }

    /// The take grows only at the start: a picked person whose track id changes
    /// half-way through stays the same person, in the same slots.
    #[test]
    fn one_seed_stays_one_person_when_the_track_id_changes() {
        let picked = [100.0, 100.0, 200.0, 400.0];
        let moved = [104.0, 100.0, 204.0, 400.0];
        let mut frames: Vec<Step> = (0..10).map(|_| vec![(0, 0, Some(picked))]).collect();
        // Frame 10: the tracker retires id 0 and re-detects the same body as id 1.
        frames.push(vec![(0, 61, None), (1, 0, Some(moved))]);
        frames.push(vec![(1, 0, Some(moved))]);

        let (people, got) = run_seeded(&frames, &[picked]);
        assert_eq!(people.len(), 1, "one pick, one person");
        assert_eq!(people[0].hits, 12, "every frame landed in that person");
        assert_eq!(got[0], (0..12).collect::<Vec<_>>());
    }

    /// A bystander detected while the picked track merely coasts must not become a
    /// person of their own: that is how one selected body became three mid-take.
    #[test]
    fn a_bystander_does_not_become_a_person_while_the_picked_track_coasts() {
        let picked = [100.0, 100.0, 200.0, 400.0];
        let mut frames: Vec<Step> = (0..9).map(|_| vec![(0, 0, Some(picked))]).collect();
        // One missed detection for the picked body, and somebody standing on top
        // of them (a box that would pass the "same body?" test).
        frames.push(vec![
            (0, 1, None),
            (5, 0, Some([104.0, 102.0, 204.0, 402.0])),
        ]);
        frames.push(vec![(0, 0, Some(picked))]);

        let (people, got) = run_seeded(&frames, &[picked]);
        assert_eq!(people.len(), 1, "one pick, one person: {:?}", people);
        assert_eq!(people[0].hits, 10, "the coasting frame is not a comeback");
        assert!(!got[0].contains(&9), "frame 9 belongs to the other body");
    }

    /// Three bodies in the room, one ticked: the cast is still one.
    #[test]
    fn a_crowd_around_the_picked_person_adds_nobody() {
        let picked = [100.0, 100.0, 200.0, 400.0];
        let mut frames: Vec<Step> = (0..6).map(|_| vec![(0, 0, Some(picked))]).collect();
        // The picked body drops out (as when somebody walks in front of it) and
        // three tracks are now in shot, all overlapping or adjacent to it.
        frames.push(vec![
            (0, 2, None),
            (1, 0, Some([104.0, 100.0, 204.0, 400.0])),
            (2, 0, Some([140.0, 100.0, 240.0, 400.0])),
            (3, 0, Some([90.0, 100.0, 190.0, 400.0])),
        ]);
        frames.push(vec![(0, 0, Some(picked))]);

        let (people, got) = run_seeded(&frames, &[picked]);
        assert_eq!(people.len(), 1, "one pick, one person");
        assert_eq!(got[0].len(), 7, "no bystander frame was taken for the pick");
    }

    /// Two ticked people stay two people, even when both are re-numbered.
    #[test]
    fn two_seeds_stay_two_people_when_both_change_ids() {
        let left = [100.0, 100.0, 200.0, 400.0];
        let right = [500.0, 100.0, 600.0, 400.0];
        let mut frames: Vec<Step> = (0..5)
            .map(|_| vec![(0, 0, Some(left)), (1, 0, Some(right))])
            .collect();
        frames.push(vec![
            (0, 61, None),
            (1, 61, None),
            (2, 0, Some([102.0, 100.0, 202.0, 400.0])),
            (3, 0, Some([502.0, 100.0, 602.0, 400.0])),
        ]);
        frames.push(vec![
            (2, 0, Some([102.0, 100.0, 202.0, 400.0])),
            (3, 0, Some([502.0, 100.0, 602.0, 400.0])),
        ]);

        let (people, got) = run_seeded(&frames, &[left, right]);
        assert_eq!(people.len(), 2, "two picks, two people");
        assert_eq!(got[0].len(), 7, "the left body kept its person");
        assert_eq!(got[1].len(), 7, "the right body kept its person");
    }
}
