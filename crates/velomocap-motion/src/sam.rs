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
    /// dropout ended the track: the 24 fps source 鈥?resampled to 30 fps, where a
    /// blurred source frame becomes one and a quarter output frames 鈥?lost the
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
    /// has LibTorch, the person pass runs through it instead of the ONNX chain 鈥?
    /// hands and fusion included, at ~1.1 s/frame on a 5060 Ti.
    pub torch_weights: Option<PathBuf>,
    /// Every how many frames the hand branch runs (`0` and `1` = every frame).
    ///
    /// The reference's default for video is four: `mocap/flow_hand_sampling.py`
    /// samples `[0, 4, 8, 鈥` plus the last frame, runs the hand pass only there
    /// and interpolates the hand features for the frames in between, while the
    /// body pass still runs everywhere. The hand branch is ~two thirds of the
    /// per-person cost (two 512脳512 crops), so this is the single biggest lever
    /// the original itself uses. `VELOMOCAP_SAM_HAND_STRIDE=1` restores
    /// per-frame hands.
    pub hand_stride: usize,
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
            hand_stride: 4,
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
    /// `18439 脳 3` vertices, metres, y-up (`x` right, `y` up, `z` forward).
    pub verts: Vec<f32>,
    /// `127 脳 3` joint centres, same frame.
    pub joints: Vec<f32>,
    /// `70 脳 3` named keypoints, same frame.
    pub keypoints: Vec<f32>,
    /// Body token for the motion stage (`1024`).
    pub body_token: Vec<f32>,
    /// Left-hand token (`1024`), empty when the backend does not run the hands.
    pub left_token: Vec<f32>,
    /// Right-hand token (`1024`), same rule.
    pub right_token: Vec<f32>,
    /// Camera-space translation used for placement.
    pub cam_t: [f32; 3],
    /// MHR hips in camera coordinates, metres 鈥?FlowHMR's placement anchor.
    pub anchor_cam: [f32; 3],
    /// `127 脳 8` joint translations plus global quaternions, straight from MHR.
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
    /// `Person 1`, `Person 2`, 鈥?in order of first appearance.
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
    /// The stabilising pass and the pruning after the last frame: CPU work the
    /// device sits idle through, so it is timed apart from the graph.
    pub cleanup: f64,
    pub total: f64,
}

impl Timings {
    pub fn line(&self) -> String {
        format!(
            "detect {:.2}s · reconstruct {:.2}s · cleanup {:.2}s · total {:.2}s",
            self.detect, self.reconstruct, self.cleanup, self.total
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
            format!(" 路 {feet}")
        };
        let meshes: usize = self
            .people
            .iter()
            .map(|p| p.frames.iter().flatten().count())
            .sum();
        format!(
            "{} people 路 {} person-frames 路 {} meshes @ {:.2} fps{}",
            self.people.len(),
            hits,
            meshes,
            self.fps,
            if self.dropped_meshes > 0 {
                format!(" 路 {} dropped (mesh budget)", self.dropped_meshes)
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


/// How many items a module file was traced at.
///
/// `sam3d_full_b8_ts.pt` carries eight people/frames per call; the plain
/// `sam3d_full_ts.pt` carries one.
fn module_batch(path: &Path) -> usize {
    let name = path.file_name().and_then(|name| name.to_str()).unwrap_or("");
    if let Some(rest) = name.strip_prefix("sam3d_full_b") {
        if let Some(digits) = rest.strip_suffix("_ts.pt") {
            if let Ok(value) = digits.parse::<usize>() {
                return value.max(1);
            }
        }
    }
    1
}

/// Whether the device has room for the widest batch and both graphs.
///
/// A batch of eight peaks around 4 GiB at the app's work resolution, and the
/// batching plus the two graphs together want about ten; below that the capture
/// keeps the single-item module and per-frame hands, which still fits.
fn device_is_roomy() -> bool {
    velomocap_cpp::sam3d::cuda_memory()
        .map(|(_, total)| total >= 10 * 1024 * 1024 * 1024)
        .unwrap_or(false)
}

/// Whether the device has room for the widest batch *and* both graphs.
///
/// The full and the body-only module together hold ~7 GB of weights, and a
/// batch of eight peaks around 4 GiB on top at the app's work resolution. Below
/// twelve gigabytes the capture keeps one graph and per-frame hands, which
/// still fits.
fn device_has_room_for_two() -> bool {
    velomocap_cpp::sam3d::cuda_memory()
        .map(|(_, total)| total >= 12 * 1024 * 1024 * 1024)
        .unwrap_or(false)
}

/// The body-only graph (`sam3d_body_b<N>_ts.pt`) traced at the same batch.
fn find_body_module(config: &SamConfig, batch: usize) -> Option<PathBuf> {
    let mut roots = vec![config.sam3d_dir.clone()];
    roots.push(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../export/sam3d"));
    for root in roots {
        let Ok(entries) = std::fs::read_dir(&root) else {
            continue;
        };
        let mut found: Option<PathBuf> = None;
        for entry in entries.flatten() {
            let path = entry.path();
            let name = path.file_name().and_then(|name| name.to_str()).unwrap_or("");
            let Some(digits) = name
                .strip_prefix("sam3d_body_b")
                .and_then(|rest| rest.strip_suffix("_ts.pt"))
            else {
                continue;
            };
            if digits.parse::<usize>().ok() == Some(batch) {
                found = Some(path);
            }
        }
        if found.is_some() {
            return found;
        }
    }
    None
}

/// The TorchScript module a capture should run.
///
/// A batched module costs one call per several people, so it is preferred; the
/// widest batch wins. `VELOMOCAP_SAM_BATCH=<n>` pins a specific one (0 and 1
/// both mean the single-item module), which is how the two are A/B'd.
///
/// A wide batch holds several frames and their hand crops at once, so on a
/// small device the capture keeps the single-item module: the point of the
/// batch is speed, and an out-of-memory failure has none.
fn find_torch_module(config: &SamConfig) -> Option<PathBuf> {
    let mut roots = vec![config.sam3d_dir.clone()];
    // Dev tree: `tools/torch/sam3d_full_jit.py` writes into `export/sam3d/`,
    // next to the workspace rather than inside `models/`, so the 3.5 GB file
    // does not have to be copied.
    roots.push(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../export/sam3d"));
    let forced = std::env::var("VELOMOCAP_SAM_BATCH")
        .ok()
        .and_then(|value| value.parse::<usize>().ok());
    // A batch of eight needs a roomy card; below that the single-item module is
    // the safe choice.
    let roomy = device_is_roomy();
    let mut best: Option<(usize, PathBuf)> = None;
    for root in roots {
        let Ok(entries) = std::fs::read_dir(&root) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("pt") {
                continue;
            }
            let name = path.file_name().and_then(|name| name.to_str()).unwrap_or("");
            if !name.starts_with("sam3d_full") || name.contains("bf16") {
                continue;
            }
            let batch = module_batch(&path);
            if let Some(forced) = forced {
                if batch != forced.max(1) {
                    continue;
                }
            } else if batch > 1 && !roomy {
                continue;
            }
            if best.as_ref().map_or(true, |(top, _)| batch > *top) {
                best = Some((batch, path));
            }
        }
    }
    best.map(|(_, path)| path)
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
    /// fusion included. See `docs/SAM3D.md`. The module may be traced at a
    /// batch of several items; [`SamRunner::batch`] reports how many.
    ///
    /// `body` is the same graph traced with `inference_type="body"`: one body
    /// pass and no hand crops. It carries the frames whose hand features are
    /// interpolated from a neighbour, which is the reference's video default.
    Torch {
        full: Box<velomocap_cpp::sam3d::Sam3dTorch>,
        body: Option<Box<velomocap_cpp::sam3d::Sam3dTorch>>,
    },
}

/// One person in one frame, waiting for the module.
///
/// The frame is shared between the people in it, so a four-person frame costs
/// one copy rather than four.
struct PendingJob {
    slot: usize,
    index: usize,
    detection: Detection,
    rgb: Arc<Vec<u8>>,
    width: usize,
    height: usize,
    /// Whether this item goes through the full graph (hand crops included).
    hands: bool,
}

impl SamRunner {
    fn backend(&self) -> String {
        match self {
            Self::Onnx(body) => body.borrow().backend().as_str().to_string(),
            Self::Torch { .. } => "torchscript".to_string(),
        }
    }

    /// How many items one call carries.
    fn batch(&self) -> usize {
        match self {
            Self::Onnx(_) => 1,
            Self::Torch { full, .. } => full.batch(),
        }
    }

    /// Whether this backend can leave the hand branch out of some frames.
    fn can_stride_hands(&self) -> bool {
        matches!(self, Self::Torch { body: Some(_), .. })
    }

    /// Run a full batch and return one frame per job, in order.
    fn run_batch(
        &self,
        jobs: &[&PendingJob],
        intrinsics: [[f32; 3]; 3],
    ) -> Result<Vec<velomocap_infer::Sam3dFrame>> {
        match self {
            Self::Onnx(body) => {
                let mut out = Vec::with_capacity(jobs.len());
                for job in jobs {
                    let bbox = [
                        job.detection.x0,
                        job.detection.y0,
                        job.detection.x1,
                        job.detection.y1,
                    ];
                    let input =
                        Sam3dInput::from_box(&job.rgb, job.width, job.height, bbox, intrinsics)
                            .context("building the SAM crop")?;
                    let frame = body.borrow_mut().run(&input).context("SAM 3D Body")?;
                    out.push(frame);
                }
                Ok(out)
            }
            Self::Torch { full, body } => {
                let (height, width) = (jobs[0].height, jobs[0].width);
                let area = width * height;
                let plane = 3 * area;
                // `frames` is shared between every item in this call, so it is
                // filled once and then sliced per item.
                let mut packed = vec![0.0f32; jobs.len() * plane];
                let mut centers = Vec::with_capacity(jobs.len() * 2);
                let mut scales = Vec::with_capacity(jobs.len() * 2);
                for (item, job) in jobs.iter().enumerate() {
                    let base = item * plane;
                    for (index, pixel) in job.rgb.chunks_exact(3).enumerate() {
                        packed[base + index] = pixel[0] as f32 / 255.0;
                        packed[base + area + index] = pixel[1] as f32 / 255.0;
                        packed[base + 2 * area + index] = pixel[2] as f32 / 255.0;
                    }
                    let (center, scale) = person_box([
                        job.detection.x0,
                        job.detection.y0,
                        job.detection.x1,
                        job.detection.y1,
                    ]);
                    centers.extend_from_slice(&center);
                    scales.extend_from_slice(&scale);
                }
                let mut cam_int = [0.0f32; 9];
                for row in 0..3 {
                    for column in 0..3 {
                        cam_int[row * 3 + column] = intrinsics[row][column];
                    }
                }
                let inputs: Vec<velomocap_cpp::sam3d::Sam3dFullInput<'_>> = (0..jobs.len())
                    .map(|item| velomocap_cpp::sam3d::Sam3dFullInput {
                        frame: &packed[item * plane..(item + 1) * plane],
                        height,
                        width,
                        bbox_center: [centers[item * 2], centers[item * 2 + 1]],
                        bbox_scale: [scales[item * 2], scales[item * 2 + 1]],
                        cam_int,
                    })
                    .collect();
                // One queue per graph: the caller never mixes sampled and
                // unsampled items in a single call.
                let module = if jobs[0].hands {
                    full
                } else {
                    body.as_ref().ok_or_else(|| {
                        anyhow::anyhow!("this backend has no body-only module")
                    })?
                };
                let outputs = module
                    .run_full_batch(&inputs)
                    .map_err(|error| anyhow::anyhow!("SAM 3D Body (TorchScript): {error}"))?;
                Ok(outputs
                    .into_iter()
                    .map(|out| {
                        // `Sam3dFrame::world_vertices` expects centimetres with
                        // `y` down and `z` forward, and subtracts `cam_t.y`. The
                        // module hands back metres in camera axes with `y` up,
                        // so the translation's `y` is mirrored alongside the mesh.
                        let verts = out.body_space_verts();
                        velomocap_infer::Sam3dFrame {
                            body_token: out.body_token,
                            left_token: out.left_token,
                            right_token: out.right_token,
                            verts,
                            skel_state: out.skel_state,
                            model_params: out.model_params,
                            pred_cam: out.pred_cam,
                            cam_t: [out.pred_cam_t[0], -out.pred_cam_t[1], out.pred_cam_t[2]],
                            pred: out.pred,
                        }
                    })
                    .collect())
            }
        }
    }
}

/// Turn one module output into the frame the rest of the pipeline stores.
fn person_from_output(
    mesh: &MeshTopology,
    output: &velomocap_infer::Sam3dFrame,
    detection: Detection,
) -> PersonFrame {
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

    PersonFrame {
        bbox: [detection.x0, detection.y0, detection.x1, detection.y1],
        score: detection.score,
        verts,
        joints,
        keypoints,
        body_token: output.body_token.clone(),
        left_token: output.left_token.clone(),
        right_token: output.right_token.clone(),
        cam_t: output.cam_t,
        anchor_cam,
        skel_state: output.skel_state.clone(),
    }
}

/// Run one batch of people and store what comes back.
///
/// A batch that spans frames lands its results a few frames late; the track
/// lists are kept long enough for that, and `flush` only ever touches the
/// frames whose people it carried.
#[allow(clippy::too_many_arguments)]
fn flush_batch(
    people: &mut [PersonTrack],
    mesh: &MeshTopology,
    config: &SamConfig,
    runner: &SamRunner,
    jobs: &[PendingJob],
    intrinsics: [[f32; 3]; 3],
    timings: &mut Timings,
    meshes_stored: &mut usize,
    dropped_meshes: &mut usize,
) -> Result<()> {
    let references: Vec<&PendingJob> = jobs.iter().collect();
    let started = Instant::now();
    let outputs = runner
        .run_batch(&references, intrinsics)
        .context("SAM 3D Body")?;
    timings.reconstruct += started.elapsed().as_secs_f64();

    for (job, output) in jobs.iter().zip(outputs) {
        let track = &mut people[job.slot];
        while track.frames.len() <= job.index {
            track.frames.push(None);
        }
        let person_frame = person_from_output(mesh, &output, job.detection);
        if *meshes_stored + person_frame.mesh_bytes() <= config.mesh_budget {
            *meshes_stored += person_frame.mesh_bytes();
            track.frames[job.index] = Some(Arc::new(person_frame));
        } else {
            *dropped_meshes += 1;
        }
        // A batch can span frames, and the two queues flush independently, so a
        // later flush can carry an *earlier* frame than one that already landed:
        // the track's range is the widest it has seen, not the last write.
        track.last_frame = track.last_frame.max(job.index);
        track.hits += 1;
    }
    Ok(())
}

/// The frames that run the hand branch: `[0, stride, 2路stride, 鈥` plus the
/// last one 鈥?`mocap/flow_hand_sampling.py::sample_indices` exactly.
///
/// A stride of one (or zero) keeps every frame, which is what the module did
/// before the fast path existed. The last frame is always an anchor so the
/// interpolation never extrapolates past the end of the clip.
pub fn hand_sample_indices(frames: usize, stride: usize) -> Vec<usize> {
    if frames == 0 {
        return Vec::new();
    }
    if stride <= 1 {
        return (0..frames).collect();
    }
    let mut indices: Vec<usize> = (0..frames).step_by(stride).collect();
    if indices.last() != Some(&(frames - 1)) {
        indices.push(frames - 1);
    }
    indices
}

/// Fill the hand features of the frames that were not sampled, in place.
///
/// `mocap/flow_hand_sampling.py::interpolate_hands`: the anchors are the frames
/// the hand branch ran on, and every other frame gets the linear blend of its
/// two neighbouring anchors. The body features are untouched. A track with a
/// single anchor copies it everywhere, and non-anchor frames that have no
/// tokens at all (a track that had not started yet) are left alone.
pub fn interpolate_hand_tokens(frames: &mut [Option<Arc<PersonFrame>>], anchors: &[usize]) {
    if anchors.len() < 2 {
        return;
    }
    let length = frames.len();
    let mut left_anchor = 0usize;
    for index in 0..length {
        // Anchors are sorted; advance while the next one is still to the right.
        while left_anchor + 2 < anchors.len() && anchors[left_anchor + 1] <= index {
            left_anchor += 1;
        }
        let left = anchors[left_anchor];
        let right = anchors[(left_anchor + 1).min(anchors.len() - 1)];
        if index == left || index == right {
            continue;
        }
        let Some(left_tokens) = frames.get(left).and_then(|frame| frame.as_ref()) else {
            continue;
        };
        let Some(right_tokens) = frames.get(right).and_then(|frame| frame.as_ref()) else {
            continue;
        };
        let (left_tokens, right_tokens) = (left_tokens.clone(), right_tokens.clone());
        let span = (right - left).max(1) as f32;
        let weight = (index.saturating_sub(left)) as f32 / span;
        let blend = |first: &[f32], second: &[f32]| -> Vec<f32> {
            first
                .iter()
                .zip(second.iter())
                .map(|(a, b)| a * (1.0 - weight) + b * weight)
                .collect()
        };
        let left_token = blend(&left_tokens.left_token, &right_tokens.left_token);
        let right_token = blend(&left_tokens.right_token, &right_tokens.right_token);
        if let Some(Some(frame)) = frames.get_mut(index) {
            let mut copy = (**frame).clone();
            copy.left_token = left_token;
            copy.right_token = right_token;
            *frame = Arc::new(copy);
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
/// and the two `fix_aspect_ratio` calls in `TopdownAffine` 鈥?i.e. what the
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
/// not double the cast. A seed with no detection is simply absent for now 鈥?the
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
///   can never come back) is the body looked for again under a new id 鈥?and then
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
            .or_else(|| find_torch_module(config))
    };
    let torch_batch = torch_weights
        .as_deref()
        .map(module_batch)
        .unwrap_or(1);
    // The reference's video default samples the hands every fourth frame and
    // interpolates the features in between (`mocap/flow_hand_sampling.py`).
    // That needs the body-only graph next to the full one; without it every
    // frame runs the full pass, which is what the ONNX chain does anyway.
    // `VELOMOCAP_SAM_HAND_STRIDE=1` restores per-frame hands; the config default
    // is the reference's four.
    let hand_stride = std::env::var("VELOMOCAP_SAM_HAND_STRIDE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(config.hand_stride);
    let body_weights = if torch_weights.is_some() && hand_stride > 1 && device_has_room_for_two()
    {
        find_body_module(config, torch_batch)
    } else {
        None
    };
    let body = match torch_weights.as_deref() {
        Some(path) if velomocap_cpp::sam3d::Sam3dTorch::built() => {
            match velomocap_cpp::sam3d::Sam3dTorch::load_batch(path, config.gpu, torch_batch) {
                Ok(module) => {
                    eprintln!(
                        "[SAM] using {} through LibTorch ({} item(s) per call)",
                        path.display(),
                        module.batch()
                    );
                    let body_only = body_weights.as_deref().and_then(|body_path| {
                        match velomocap_cpp::sam3d::Sam3dTorch::load_batch(
                            body_path, config.gpu, torch_batch,
                        ) {
                            Ok(body_module) => {
                                eprintln!(
                                    "[SAM] hands every {} frame(s), body {} elsewhere",
                                    hand_stride.max(1),
                                    body_path.display()
                                );
                                Some(Box::new(body_module))
                            }
                            Err(error) => {
                                eprintln!(
                                    "[SAM] body-only load failed ({error}); hands every frame"
                                );
                                None
                            }
                        }
                    });
                    SamRunner::Torch {
                        full: Box::new(module),
                        body: body_only,
                    }
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
    // People waiting for a batch of `body.batch()` items. Hand frames and body
    // frames go to different graphs, so each gets its own queue.
    let mut pending_hands: Vec<PendingJob> = Vec::new();
    let mut pending_body: Vec<PendingJob> = Vec::new();
    // The frames the hand branch runs on; the rest are interpolated at the end.
    // A backend without a body-only graph runs the hands everywhere.
    let hand_stride = if body.can_stride_hands() {
        hand_stride.max(1)
    } else {
        1
    };
    let anchors = hand_sample_indices(total, hand_stride);
    let mut hands_on_frame = vec![false; total.max(1)];
    for index in &anchors {
        hands_on_frame[*index] = true;
    }
    if hand_stride > 1 {
        eprintln!(
            "[SAM] hands on {} of {} frames (stride {hand_stride}, the reference default)",
            anchors.len(),
            total
        );
    }

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

        // Queue this frame's people. A module traced at several items per call
        // takes them in batches; the ONNX chain takes them one at a time, which
        // is the same code with a batch of one. Frames whose hand features are
        // interpolated go to the body-only graph instead.
        let frame_rgb = Arc::new(frame.rgb.clone());
        let hands_here = hands_on_frame[index.min(hands_on_frame.len().saturating_sub(1))];
        let queue = if hands_here {
            &mut pending_hands
        } else {
            &mut pending_body
        };
        for (slot, detection) in jobs {
            (hooks.progress)(decoded, total, "reconstructing");
            while people[slot].frames.len() <= index {
                people[slot].frames.push(None);
            }
            queue.push(PendingJob {
                slot,
                index,
                detection,
                rgb: Arc::clone(&frame_rgb),
                width: frame.width,
                height: frame.height,
                hands: hands_here,
            });
        }
        let batch = body.batch().max(1);
        for queue in [&mut pending_hands, &mut pending_body] {
            while queue.len() >= batch {
                let count = batch.min(queue.len());
                let group: Vec<PendingJob> = queue.drain(..count).collect();
                flush_batch(
                    &mut people,
                    &mesh,
                    config,
                    &body,
                    &group,
                    intrinsics,
                    &mut timings,
                    &mut meshes_stored,
                    &mut dropped_meshes,
                )?;
            }
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

    // Whatever is left of the last batch still has to be reconstructed: the
    // module pads it up to its own batch and the extra results are dropped.
    if !(hooks.cancelled)() && !(pending_hands.is_empty() && pending_body.is_empty()) {
        for queue in [&mut pending_hands, &mut pending_body] {
            if queue.is_empty() {
                continue;
            }
            let group: Vec<PendingJob> = queue.drain(..).collect();
            flush_batch(
                &mut people,
                &mesh,
                config,
                &body,
                &group,
                intrinsics,
                &mut timings,
                &mut meshes_stored,
                &mut dropped_meshes,
            )?;
        }
        // Fill the hand features of the frames the hand branch skipped, from
        // their neighbouring anchors (`flow_hand_sampling.py::interpolate_hands`).
        if hand_stride > 1 {
            (hooks.progress)(decoded, total, "interpolating hands");
            for track in people.iter_mut() {
                interpolate_hand_tokens(&mut track.frames, &anchors);
            }
        }
        (hooks.people)(decoded.saturating_sub(1), &people);
    }

    // Planted feet stay planted, and the body stops jittering frame to frame.
    (hooks.progress)(decoded, total, "stabilising");
    let cleanup_started = Instant::now();
    let cleanup = crate::cleanup::stabilise(&mut people, &crate::cleanup::CleanupConfig::default());
    timings.cleanup = cleanup_started.elapsed().as_secs_f64();

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
    /// does 鈥?identity resolution, then one person slot per seed 鈥?and report the
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
