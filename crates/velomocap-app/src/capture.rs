//! Background SAM 3D Body capture: video → people → MHR meshes → viewport.
//!
//! The worker owns the ONNX sessions and the decoder; the UI only ever sees
//! plain data behind a mutex, so the window stays responsive while four bodies
//! are reconstructed per frame.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

use anyhow::Result;
use velomocap_infer::MeshTopology;
use velomocap_motion::{SamConfig, SamHooks, PersonTrack};

/// How much of the pipeline is allowed to stay resident on the device.
///
/// The stages never run at the same time (SAM → camera → motion), so the mode
/// decides two things: whether a stage's allocator keeps its blocks for the next
/// stage, and which stages are allowed on the GPU at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryMode {
    /// Keep everything on the device and never trim: the fastest, needs ~24 GB.
    Unlimited,
    /// Load and unload stage by stage, trimming between them: 16 GB.
    Scheduled,
    /// Stage by stage, but the 4.4 GB camera model runs on the CPU: 8 GB.
    Hybrid,
    /// Every model on the CPU, no device memory at all.
    Cpu,
}

impl Default for MemoryMode {
    fn default() -> Self {
        Self::Scheduled
    }
}

impl MemoryMode {
    pub const ALL: [MemoryMode; 4] =
        [Self::Unlimited, Self::Scheduled, Self::Hybrid, Self::Cpu];

    /// Pick a mode from the device's total memory.
    pub fn from_vram(total_bytes: Option<u64>) -> Self {
        let Some(total) = total_bytes else {
            return Self::Cpu;
        };
        let gb = total as f64 / 1e9;
        if gb >= 22.0 {
            Self::Unlimited
        } else if gb >= 14.0 {
            Self::Scheduled
        } else if gb >= 5.0 {
            Self::Hybrid
        } else {
            Self::Cpu
        }
    }

    /// Short label for the console and the status bar.
    pub fn label(self) -> &'static str {
        match self {
            Self::Unlimited => "unlimited",
            Self::Scheduled => "scheduled",
            Self::Hybrid => "hybrid",
            Self::Cpu => "cpu",
        }
    }

    /// Whether the heavy models run on the device at all.
    pub fn uses_gpu(self) -> bool {
        !matches!(self, Self::Cpu)
    }

    /// Whether each stage hands its memory back before the next one loads.
    pub fn trims_between_stages(self) -> bool {
        !matches!(self, Self::Unlimited)
    }

    /// Whether the camera estimator runs on the CPU to spare its 4.4 GB.
    pub fn camera_on_cpu(self) -> bool {
        matches!(self, Self::Hybrid)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptureStage {
    Idle,
    Loading,
    Decoding,
    Detecting,
    Reconstructing,
    /// VGGT-Omega: per-frame camera estimation.
    Camera,
    /// FlowHMR: one coherent clip per person.
    Motion,
    /// SMPL-H skinning of the generated poses, for the viewport.
    Skin,
    Done,
    Failed,
}

impl CaptureStage {
    pub fn label(self) -> &'static str {
        match self {
            CaptureStage::Idle => "idle",
            CaptureStage::Loading => "loading models",
            CaptureStage::Decoding => "decoding",
            CaptureStage::Detecting => "detecting people",
            CaptureStage::Reconstructing => "reconstructing",
            CaptureStage::Camera => "estimating camera",
            CaptureStage::Motion => "generating motion",
            CaptureStage::Skin => "skinning",
            CaptureStage::Done => "done",
            CaptureStage::Failed => "failed",
        }
    }
}

/// The stages a run goes through, in the order the progress display shows them.
///
/// Every stage reports its own `(done, total)`, so the bar moves during the long
/// ones — camera and motion used to leave the SAM counter frozen at 100 % and the
/// run looked hung.
pub const STAGE_COUNT: usize = 7;

pub fn stage_name(index: usize) -> &'static str {
    match index {
        0 => "loading models",
        1 => "detecting people",
        2 => "reconstructing people",
        3 => "estimating camera",
        4 => "generating motion",
        5 => "skinning",
        _ => "done",
    }
}

fn stage_index(stage: &str) -> usize {
    match stage {
        "loading mesh" | "loading detector" | "loading SAM 3D Body" | "opening video"
        | "flow body model" | "flow camera" => 0,
        "detecting" => 1,
        "reconstructing" | "interpolating hands" | "stabilising" => 2,
        "camera" => 3,
        "flow" => 4,
        "skin" => 5,
        _ => 6,
    }
}

#[derive(Clone, Debug)]
pub struct CaptureProgress {
    pub stage: CaptureStage,
    /// Zero-based index into [`stage_name`]; the UI shows it as "stage n/6".
    pub stage_index: usize,
    /// Work done inside the current stage, not in the whole run.
    pub frame: usize,
    /// Work the current stage has to do; `0` means "unknown, show a spinner".
    pub total: usize,
    /// People reconstructed in the frame being processed.
    pub people: usize,
    pub message: String,
    pub elapsed: f64,
}

impl Default for CaptureProgress {
    fn default() -> Self {
        Self {
            stage: CaptureStage::Idle,
            stage_index: 0,
            frame: 0,
            total: 0,
            people: 0,
            message: String::new(),
            elapsed: 0.0,
        }
    }
}

impl CaptureProgress {
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

/// A decoded frame at preview resolution.
pub struct PreviewFrame {
    pub index: usize,
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
}

/// Shared capture state.
#[derive(Default)]
pub struct CaptureShared {
    pub progress: CaptureProgress,
    /// Latest decoded frame for the Movie Clip view.
    pub preview: Option<Arc<PreviewFrame>>,
    /// Every track so far; meshes arrive frame by frame.
    pub people: Vec<PersonTrack>,
    /// Mesh topology, available as soon as the worker loads it.
    pub mesh: Option<Arc<MeshTopology>>,
    /// Work-resolution video size, for detection overlays.
    pub video_size: (usize, usize),
    /// Source FPS from the container.
    pub fps: f64,
    pub summary: String,
    pub timings: String,
    pub backend: String,
    /// Set once when the worker finishes, after pruning short tracks.
    pub finished: bool,
    /// FlowHMR clips, one per person, once the motion stage has run.
    pub flow: Vec<FlowClipData>,
    /// SMPL-H triangles for those bodies.
    pub smplh_faces: Option<Arc<Vec<[u32; 3]>>>,
    /// Why the motion stage produced nothing, when it did.
    pub flow_note: String,
    /// Device memory as reported between stages, for the console and the HUD.
    pub vram: String,
}

/// One person's generated clip, skinned and ready for the viewport.
#[derive(Clone)]
pub struct FlowClipData {
    pub name: String,
    pub first_frame: usize,
    pub frames: Arc<Vec<Vec<f32>>>,
    pub summary: String,
    /// Ready-to-write BVH payload; the body model is dropped after the bake.
    pub bvh: Arc<velomocap_motion::PreparedBvh>,
    /// The clip's joint rotations per frame: what an imported character is
    /// posed with (the mesh above is the built-in body).
    pub skeletons: Arc<Vec<velomocap_body::Skeleton>>,
    /// The contact runs the detector found, per track (L foot, R foot, L hand,
    /// R hand), in clip frames. The timeline draws these under the hand-drawn
    /// locks so an automatic track shows what it is following instead of the
    /// word "automatic".
    pub contacts: [Vec<[usize; 2]>; velomocap_motion::TRACK_COUNT],
    /// The generated clip **before** the contact correction: the input an
    /// "apply correction" run needs, so changing a preset or drawing a lock does
    /// not have to re-run the model. A few hundred KB for a 90-frame take.
    pub source: Arc<velomocap_motion::FlowClip>,
}

/// Where the motion models live, and how much of the clip to animate.
#[derive(Clone)]
pub struct FlowSettings {
    pub weights: PathBuf,
    pub stats: PathBuf,
    pub body: PathBuf,
    pub camera_weights: PathBuf,
    /// 0 = every frame.
    pub max_frames: usize,
    pub gpu: bool,
    /// Foot/hand contact correction applied to the generated clip.
    pub correction: velomocap_motion::correction::CorrectionSettings,
    /// Contact timing the user drew on the timeline; automatic tracks follow the
    /// clip's own detection.
    pub locks: velomocap_motion::ContactTracks,
    /// Estimate the camera with VGGT; `false` treats it as a fixed camera.
    pub estimate_camera: bool,
    /// Smooth the estimated camera track over time.
    pub refine_camera: bool,
    /// Put the reconstructed actors on the world origin instead of where the
    /// camera saw them (the sidebar's "Center At Origin").
    pub center_scene: bool,
}

/// The first frame of a clip, plus what its container reported.
pub struct Probe {
    pub frame: PreviewFrame,
    pub fps: f64,
    pub frames: usize,
}

/// Decode one frame and read the container metadata, for the idle preview.
pub fn probe_frame(video: &std::path::Path, width: usize) -> Result<Probe> {
    let tools = velomocap_video::Tools::discover()?;
    let mut decoder = velomocap_video::Decoder::open(&tools, video, Some(width))?;
    let info = decoder.info().clone();
    let first = decoder
        .next_frame()?
        .ok_or_else(|| anyhow::anyhow!("{} has no frames", video.display()))?;
    let frame = downscale(first.index, &first.rgb, first.width, first.height, width);
    Ok(Probe {
        frame,
        fps: info.fps,
        frames: info.frames,
    })
}

pub struct CaptureJob {
    pub handle: Option<JoinHandle<()>>,
    pub cancel: Arc<AtomicBool>,
    pub shared: Arc<Mutex<CaptureShared>>,
}

/// What "apply correction" needs to re-solve one take without the model.
#[derive(Clone)]
pub struct TakeSource {
    pub name: String,
    pub first_frame: usize,
    /// The generated motion, before the contact correction.
    pub source: Arc<velomocap_motion::FlowClip>,
}

/// Re-run the contact correction on already generated takes.
///
/// The reference's `Apply Correction` button: it starts from the motion the model
/// produced and re-applies the current settings and the timeline's locks, so a
/// preset change or a drawn lock costs the correction (and the skinning), not a
/// full inference run. The SMPL-H model is loaded here and dropped again.
pub fn recorrect(
    takes: &[TakeSource],
    body_path: &std::path::Path,
    settings: &velomocap_motion::correction::CorrectionSettings,
    locks: &velomocap_motion::ContactTracks,
) -> anyhow::Result<Vec<FlowClipData>> {
    let has_posedirs = velomocap_body::npz::members(body_path)
        .map(|names| {
            names
                .iter()
                .any(|name| name == "posedirs" || name.starts_with("posedirs."))
        })
        .unwrap_or(false);
    let mut model = velomocap_body::BodyModel::load(body_path, 16)?;
    if has_posedirs {
        model = model.with_posedirs()?;
    }
    let mut out = Vec::with_capacity(takes.len());
    for take in takes {
        let mut clip = (*take.source).clone();
        // A take that came back from a project stores poses only; the forward
        // pass is what turns them into the skeletons the first pass reads.
        if clip.skeletons.len() != clip.poses.len() {
            clip.skeletons = clip
                .poses
                .iter()
                .map(|pose| model.forward_joints(pose))
                .collect();
        }
        let correction = velomocap_motion::correction::apply_with_locks(
            &mut clip, &model, settings, locks,
        )?;
        let mut bodies = Vec::with_capacity(clip.frames());
        for pose in clip.poses.iter() {
            match model.forward_mesh(pose) {
                Ok((_, vertices)) => bodies.push(vertices),
                Err(_) => bodies.push(Vec::new()),
            }
        }
        let mut summary = velomocap_motion::summarise_flow(&clip);
        summary.push(correction.line());
        let bvh = velomocap_motion::prepare_bvh(&clip, &model, clip.fps as f64)?;
        out.push(FlowClipData {
            name: take.name.clone(),
            first_frame: take.first_frame,
            frames: Arc::new(bodies),
            summary: summary.join("; "),
            bvh: Arc::new(bvh),
            skeletons: Arc::new(clip.skeletons.clone()),
            contacts: correction.detected.clone(),
            source: Arc::clone(&take.source),
        });
    }
    Ok(out)
}

impl CaptureJob {
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

pub struct CaptureSettings {
    pub video: PathBuf,
    pub config: SamConfig,
    /// Run FlowHMR after the reconstruction and publish preview bodies.
    pub flow: Option<FlowSettings>,
    /// How much may stay resident on the device (auto-selected from its size).
    pub mode: MemoryMode,
}

pub fn start(settings: CaptureSettings) -> CaptureJob {
    let shared = Arc::new(Mutex::new(CaptureShared::default()));
    let cancel = Arc::new(AtomicBool::new(false));
    let worker_shared = Arc::clone(&shared);
    let worker_cancel = Arc::clone(&cancel);

    let handle = std::thread::Builder::new()
        .name("velomocap-sam".into())
        .spawn(move || {
            if let Err(error) = run(settings, worker_shared.clone(), worker_cancel) {
                let mut state = worker_shared.lock().unwrap();
                state.progress.stage = CaptureStage::Failed;
                state.progress.message = format!("{error:#}");
                state.finished = true;
            }
        })
        .ok();

    CaptureJob {
        handle,
        cancel,
        shared,
    }
}

fn run(
    settings: CaptureSettings,
    shared: Arc<Mutex<CaptureShared>>,
    cancel: Arc<AtomicBool>,
) -> Result<()> {
    let started = Instant::now();
    let video = settings.video.clone();
    let mode = settings.mode;
    if !mode.trims_between_stages() {
        if let Some(line) = velomocap_motion::vram::cuda_memory_line() {
            eprintln!("[vram] start ({}): {line}", mode.label());
            if let Ok(mut state) = shared.lock() {
                state.vram = format!("{line} · {} (no trimming)", mode.label());
            }
        }
    }

    let mut progress = {
        let shared = Arc::clone(&shared);
        // The rate in the status line is the current stage's own rate: dividing a
        // stage's frame count by the whole run's elapsed time made every later
        // stage look slower than it is.
        let mut stage_started = Instant::now();
        let mut current = String::new();
        move |frame: usize, total: usize, stage: &str| {
            let mut state = shared.lock().unwrap();
            state.progress.stage = match stage {
                "loading mesh" | "loading detector" | "loading SAM 3D Body" | "opening video"
                | "flow body model" | "flow camera" => CaptureStage::Loading,
                "detecting" => CaptureStage::Detecting,
                "reconstructing" => CaptureStage::Reconstructing,
                "camera" => CaptureStage::Camera,
                "flow" => CaptureStage::Motion,
                "skin" => CaptureStage::Skin,
                "done" => CaptureStage::Done,
                _ => state.progress.stage,
            };
            if stage != current {
                current = stage.to_string();
                stage_started = Instant::now();
                // The headless log is the only record of a run without a window,
                // and the phase order is the thing that was hard to read.
                eprintln!("[progress] {stage} ({frame}/{total})");
            }
            state.progress.stage_index = stage_index(stage);
            state.progress.frame = frame;
            state.progress.total = total;
            state.progress.message = stage.to_string();
            state.progress.elapsed = stage_started.elapsed().as_secs_f64();
            // Sample device memory *inside* a stage, not only at its edges: a
            // per-frame leak in the reconstruction loop is invisible otherwise, and
            // "it used to be 8 GB" has to be answerable from the log.
            if frame % 10 == 0 {
                if let Some(line) = velomocap_motion::vram::cuda_memory_line() {
                    state.vram = format!("{line} · {stage}");
                }
            }
        }
    };
    let mut preview = {
        let shared = Arc::clone(&shared);
        move |frame: &velomocap_video::Frame| {
            let preview = downscale(frame.index, &frame.rgb, frame.width, frame.height, 480);
            if let Ok(mut state) = shared.lock() {
                state.preview = Some(Arc::new(preview));
                if state.video_size == (0, 0) {
                    state.video_size = (frame.width, frame.height);
                }
            }
        }
    };
    let mut people = {
        let shared = Arc::clone(&shared);
        move |_index: usize, tracks: &[PersonTrack]| {
            if let Ok(mut state) = shared.lock() {
                state.people = tracks.to_vec();
                state.progress.people = tracks.len();
            }
        }
    };
    let cancelled = {
        let cancel = Arc::clone(&cancel);
        move || cancel.load(Ordering::Relaxed)
    };
    let mut hooks = SamHooks {
        progress: &mut progress,
        preview: &mut preview,
        people: &mut people,
        cancelled: &cancelled,
    };
    let capture = velomocap_motion::capture_people_with(&video, &settings.config, &mut hooks)?;
    // The stage split is what any speed claim has to be checked against, and a
    // headless run has no console panel to read it from. Printed before the flow
    // stage, which is where a headless run stops.
    eprintln!("[capture] {}", capture.timings.line());
    // The detector and SAM module are gone at this point, but both allocators
    // keep their blocks: without handing them back, the flow stage cannot fit
    // and its weights end up paged into host memory. `Unlimited` keeps them on
    // purpose — trimming is what costs the reload time.
    release_vram(&shared, "after SAM", mode.trims_between_stages());

    // FlowHMR: the coherent SMPL-H motion behind the preview and the exports.
    // A failure here must not lose the reconstruction, so it is reported and the
    // per-frame MHR bodies stay on screen.
    if let Some(flow) = settings.flow.clone() {
        if let Err(error) = run_flow(&video, &capture.people, &flow, &shared, mode) {
            let mut state = shared.lock().unwrap();
            state.progress.message = format!("FlowHMR: {error:#}");
            state.flow_note = format!("FlowHMR failed: {error}");
        }
        release_vram(&shared, "after motion", mode.trims_between_stages());
    }
    let summary = capture.summary();
    let line = capture.timings.line();
    let velomocap_motion::PeopleCapture {
        people: tracks,
        mesh,
        fps,
        width,
        height,
        backend,
        ..
    } = capture;

    let mut state = shared.lock().unwrap();
    state.people = tracks;
    state.mesh = Some(mesh);
    state.fps = fps;
    state.video_size = (width, height);
    state.summary = summary;
    state.timings = line;
    state.backend = backend;
    state.progress.stage = CaptureStage::Done;
    state.progress.stage_index = STAGE_COUNT - 1;
    state.progress.frame = 0;
    state.progress.total = 0;
    state.progress.elapsed = started.elapsed().as_secs_f64();
    state.finished = true;
    Ok(())
}

/// Return freed device memory to the driver between stages.
///
/// Both allocators in play keep their own pools: PyTorch's caching allocator
/// holds released blocks (`nvidia-smi` does not move when a module is dropped)
/// and ONNX Runtime keeps a per-session arena. The next stage's weights then
/// cannot fit, Windows pages them into host memory, and every kernel crawls.
/// Dropping the models is not enough — the cache has to be trimmed explicitly.
fn release_vram(shared: &Arc<Mutex<CaptureShared>>, stage: &str, trim: bool) {
    // `Unlimited` deliberately keeps the blocks: reloading a 4-5 GB model costs
    // far more than the memory it saves on a card that has room for it.
    if trim {
        velomocap_motion::vram::empty_cuda_cache();
    }
    let Some(line) = velomocap_motion::vram::cuda_memory_line() else {
        return;
    };
    eprintln!("[vram] {stage}: {line}");
    if let Ok(mut state) = shared.lock() {
        state.vram = format!("{line} · {stage}");
    }
}

/// Generate one FlowHMR clip per person and skin it for the viewport.
fn run_flow(
    video: &std::path::Path,
    people: &[PersonTrack],
    settings: &FlowSettings,
    shared: &Arc<Mutex<CaptureShared>>,
    mode: MemoryMode,
) -> Result<()> {
    use velomocap_infer::{MotionConfig, SessionConfig, VggtCamera, VggtConfig};

    report_stage(shared, 0, 0, 0, "flow camera");
    let tools = velomocap_video::Tools::discover()?;
    let gpu = settings.gpu && mode.uses_gpu();
    // Two ways to skip the 4.4 GB camera model: a fixed camera does not need it
    // (its track is exactly what `freeze()` would produce), and `Hybrid` moves it
    // to the CPU to leave the device to SAM and the motion model.
    let camera = if !settings.estimate_camera {
        let info = velomocap_video::probe(&tools, video)?;
        eprintln!(
            "[camera] static: identity track over {} frames (VGGT not loaded)",
            info.frames.max(1)
        );
        velomocap_infer::CameraTrack::static_camera(
            info.frames.max(1),
            info.width,
            info.height,
        )
    } else {
        let mut vggt = VggtCamera::load(VggtConfig {
            weights: settings.camera_weights.clone(),
            session: SessionConfig {
                gpu: gpu && !mode.camera_on_cpu(),
                ..Default::default()
            },
            ..Default::default()
        })?;
        let mut camera = vggt.estimate(&tools, video)?;
        if settings.refine_camera {
            camera.smooth(1);
        }
        camera
    };
    // Scoped above, so the camera weights are gone before FlowHMR loads.
    release_vram(shared, "after camera", mode.trims_between_stages());
    report_stage(shared, 0, 0, 0, "flow body model");
    let has_posedirs = velomocap_body::npz::members(&settings.body)
        .map(|names| names.iter().any(|name| name == "posedirs" || name.starts_with("posedirs.")))
        .unwrap_or(false);
    let mut model = velomocap_body::BodyModel::load(&settings.body, 16)?;
    if has_posedirs {
        model = model.with_posedirs()?;
    }
    let faces = Arc::new(
        model
            .faces
            .chunks_exact(3)
            .map(|face| [face[0], face[1], face[2]])
            .collect::<Vec<[u32; 3]>>(),
    );

    let config = velomocap_motion::FlowConfig {
        motion: MotionConfig {
            weights: settings.weights.clone(),
            stats: settings.stats.clone(),
            session: SessionConfig {
                gpu,
                ..Default::default()
            },
            ..Default::default()
        },
        max_frames: settings.max_frames,
        // The scene floor needs every actor's estimate, so generation leaves each
        // clip in the network's datum and the shift happens once the median is in.
        defer_floor: true,
        ..Default::default()
    };

    // Two passes over the people: generate every clip into the network's own
    // datum first, then stand the whole take on one plane. `flow_floor.resolve`
    // estimates a plane per actor and gives the scene the median, so two people
    // never end up on two different floors — the per-actor shift the single-clip
    // path applies would do exactly that.
    let mut generated: Vec<(usize, velomocap_motion::FlowClip)> = Vec::new();
    let sentinel = || {
        if let Ok(state) = shared.lock() {
            if state.progress.stage == CaptureStage::Failed {
                return true;
            }
        }
        false
    };
    for (slot, person) in people.iter().enumerate() {
        report_stage(shared, 3, slot, people.len(), "flow");
        match velomocap_motion::generate_flow(person, &camera, &model, &config) {
            Ok(clip) => generated.push((slot, clip)),
            Err(error) => {
                if let Ok(mut state) = shared.lock() {
                    state.flow_note = format!("FlowHMR {}: {error}", person.name);
                }
            }
        }
        if sentinel() {
            break;
        }
    }
    let estimates: Vec<f32> = generated
        .iter()
        .filter(|(_, clip)| clip.floor_from_contacts)
        .map(|(_, clip)| clip.floor)
        .collect();
    let shared_floor = velomocap_motion::scene_floor(&estimates);
    if let Some(floor) = shared_floor {
        for (_, clip) in generated.iter_mut() {
            velomocap_motion::shift_to_floor(clip, floor);
        }
    }
    // The placement puts the take where the camera actually saw it (a few metres
    // in front of the lens), which is right for a scene of actors but leaves a
    // lone character far from the origin — awkward in every DCC and in the
    // viewport's own grid. One shared offset from the mean first-frame root keeps
    // the actors' distances to each other and puts the group on the origin; the
    // height is untouched, the floor already put the soles on `y = 0`.
    let centered: Option<[f32; 2]> = if settings.center_scene {
        let roots: Vec<[f32; 3]> = generated
            .iter()
            .filter_map(|(_, clip)| clip.first_root())
            .collect();
        match velomocap_motion::center_shift(&roots) {
            Some(centre) => {
                for (_, clip) in generated.iter_mut() {
                    velomocap_motion::shift_xz(clip, -centre[0], -centre[1]);
                }
                eprintln!(
                    "[scene] centred on the origin: x {:+.3} m, z {:+.3} m",
                    -centre[0], -centre[1]
                );
                Some([-centre[0], -centre[1]])
            }
            None => None,
        }
    } else {
        None
    };

    let mut clips = Vec::new();
    for (slot, mut clip) in generated {
        let person = &people[slot];
        // Keep the generated motion as it came out of the model: this is what an
        // "apply correction" run starts from, so a settings or lock edit does not
        // pay for inference again.
        let source = Arc::new(clip.clone());
        // Contact correction runs on the generated clip, so the viewport and
        // every export see the same corrected motion. The user's lock tracks go
        // with it: an edited limb follows the drawing, the rest follows detection.
        let correction = match velomocap_motion::correction::apply_with_locks(
            &mut clip,
            &model,
            &settings.correction,
            &settings.locks,
        ) {
            Ok(report) => report,
            Err(error) => {
                if let Ok(mut state) = shared.lock() {
                    state.flow_note = format!("Correction {}: {error}", person.name);
                }
                velomocap_motion::correction::CorrectionReport::default()
            }
        };
        let mut bodies = Vec::with_capacity(clip.frames());
        let poses = clip.poses.len().max(1);
        let mut empty_bodies = 0usize;
        let mut first_empty = 0usize;
        for (pose_index, pose) in clip.poses.iter().enumerate() {
            match model.forward_mesh(pose) {
                Ok((_, vertices)) => bodies.push(vertices),
                Err(_) => {
                    if empty_bodies == 0 {
                        first_empty = pose_index;
                    }
                    empty_bodies += 1;
                    bodies.push(Vec::new());
                }
            }
            // Skinning 90 frames of a 14k-triangle body is not instant; let the
            // bar move instead of sitting at "generating motion".
            if pose_index % 8 == 0 {
                report_stage(shared, 4, pose_index, poses, "skin");
            }
        }
        // A take whose tail failed to skin, or whose track stopped early, shows up
        // as "the model disappeared" in the viewport: say so in the log instead of
        // leaving the user to notice it.
        if empty_bodies > 0 {
            if let Ok(mut state) = shared.lock() {
                state.flow_note = format!(
                    "{}: {} of {} frames could not be skinned (first at frame {})",
                    person.name,
                    empty_bodies,
                    bodies.len(),
                    first_empty
                );
            }
        }
        let expected = person.frames.len();
        let start = clip.source_frames.first().copied().unwrap_or(0);
        if start + clip.frames() < expected {
            if let Ok(mut state) = shared.lock() {
                state.flow_note = format!(
                    "{}: the take covers frames {}..{} of {} (the person track ended early)",
                    person.name,
                    start,
                    start + clip.frames(),
                    expected
                );
            }
        }
        let mut summary = velomocap_motion::summarise_flow(&clip);
        if let Some(floor) = shared_floor {
            summary.push(format!(
                "scene floor {floor:+.3} m over {} actor(s)",
                estimates.len()
            ));
        }
        if let Some([dx, dz]) = centered {
            summary.push(format!("centred x {dx:+.3} m z {dz:+.3} m on the origin"));
        }
        summary.push(correction.line());
        let summary = summary.join("; ");
        let bvh = match velomocap_motion::prepare_bvh(&clip, &model, clip.fps as f64) {
            Ok(prepared) => prepared,
            Err(error) => {
                if let Ok(mut state) = shared.lock() {
                    state.flow_note = format!("FlowHMR BVH {}: {error}", person.name);
                }
                continue;
            }
        };
        clips.push(FlowClipData {
            name: person.name.clone(),
            first_frame: clip.source_frames.first().copied().unwrap_or(0),
            frames: Arc::new(bodies),
            summary,
            bvh: Arc::new(bvh),
            skeletons: Arc::new(clip.skeletons.clone()),
            contacts: correction.detected.clone(),
            source,
        });
        if sentinel() {
            break;
        }
    }

    // Also on stderr: the GUI console panel is not always visible, and a
    // headless run needs to prove the stage happened.
    for clip in &clips {
        eprintln!(
            "flow: {} from frame {}, {} baked frames — {}",
            clip.name,
            clip.first_frame,
            clip.frames.len(),
            clip.summary
        );
    }
    if let Ok(mut state) = shared.lock() {
        state.flow = clips;
        state.smplh_faces = Some(faces);
    }
    // Headless verification: the report is on stderr, so stop here instead of
    // sitting in the window loop. A run that is left behind holds the executable
    // locked (which blocks the next build) and keeps a GPU context alive per run.
    if std::env::var_os("VELOMOCAP_EXIT_AFTER_CAPTURE").is_some() {
        std::process::exit(0);
    }
    Ok(())
}

/// Report progress for a stage the flow pass owns.
///
/// The flow pass runs outside the SAM loop, so it sets its own stage and
/// counters; without this the bar stayed at the SAM frame count for the whole
/// (much longer) motion stage.
fn report_stage(
    shared: &Arc<Mutex<CaptureShared>>,
    index: usize,
    done: usize,
    total: usize,
    message: &str,
) {
    if let Ok(mut state) = shared.lock() {
        state.progress.stage_index = index;
        state.progress.stage = match message {
            "flow camera" | "flow body model" => CaptureStage::Loading,
            "camera" => CaptureStage::Camera,
            "flow" => CaptureStage::Motion,
            "skin" => CaptureStage::Skin,
            _ => state.progress.stage,
        };
        state.progress.frame = done;
        state.progress.total = total;
        state.progress.message = message.to_string();
    }
}

/// A decoded frame at preview resolution, for the Movie Clip panel.
pub fn preview_from_frame(frame: &velomocap_video::Frame, target: usize) -> PreviewFrame {
    downscale(frame.index, &frame.rgb, frame.width, frame.height, target)
}

/// Nearest-neighbour downscale for previews (same approach as the detector view).
fn downscale(index: usize, rgb: &[u8], width: usize, height: usize, target: usize) -> PreviewFrame {
    if width <= target {
        return PreviewFrame {
            index,
            width,
            height,
            rgb: rgb.to_vec(),
        };
    }
    let scale = width as f32 / target as f32;
    let out_w = target;
    let out_h = ((height as f32 / scale).round() as usize).max(1);
    let mut out = vec![0u8; out_w * out_h * 3];
    for y in 0..out_h {
        let sy = ((y as f32 * scale) as usize).min(height - 1);
        for x in 0..out_w {
            let sx = ((x as f32 * scale) as usize).min(width - 1);
            let src = (sy * width + sx) * 3;
            let dst = (y * out_w + x) * 3;
            out[dst..dst + 3].copy_from_slice(&rgb[src..src + 3]);
        }
    }
    PreviewFrame {
        index,
        width: out_w,
        height: out_h,
        rgb: out,
    }
}

#[cfg(test)]
    /// The auto-selection table, and the flags each mode implies.
    #[test]
    fn memory_mode_follows_the_card_size() {
        let gb = |value: u64| Some(value * 1_000_000_000);
        assert_eq!(MemoryMode::from_vram(gb(24)), MemoryMode::Unlimited);
        assert_eq!(MemoryMode::from_vram(gb(16)), MemoryMode::Scheduled);
        assert_eq!(MemoryMode::from_vram(gb(8)), MemoryMode::Hybrid);
        assert_eq!(MemoryMode::from_vram(gb(4)), MemoryMode::Cpu);
        assert_eq!(MemoryMode::from_vram(None), MemoryMode::Cpu);
        assert!(!MemoryMode::Unlimited.trims_between_stages());
        assert!(MemoryMode::Scheduled.trims_between_stages());
        assert!(MemoryMode::Hybrid.camera_on_cpu());
        assert!(!MemoryMode::Scheduled.camera_on_cpu());
        assert!(!MemoryMode::Cpu.uses_gpu());
        assert!(MemoryMode::Unlimited.uses_gpu());
        assert!(MemoryMode::Hybrid.uses_gpu());
    }
    /// Every stage the bar can show has a name, and the flow pass maps onto the
    /// same indices as the SAM pass — otherwise the bar jumps backwards.
    #[test]
    fn every_stage_has_a_name_and_a_stable_index() {
        let names: Vec<&str> = (0..STAGE_COUNT).map(stage_name).collect();
        assert_eq!(
            names,
            vec![
                "loading models",
                "detecting people",
                "reconstructing people",
                "estimating camera",
                "generating motion",
                "skinning",
                "done",
            ]
        );
        assert_eq!(stage_index("loading detector"), 0);
        assert_eq!(stage_index("detecting"), 1);
        assert_eq!(stage_index("reconstructing"), 2);
        assert_eq!(stage_index("interpolating hands"), 2);
        assert_eq!(stage_index("camera"), 3);
        assert_eq!(stage_index("flow"), 4);
        assert_eq!(stage_index("skin"), 5);
        assert_eq!(stage_index("done"), 6);
        assert_eq!(stage_index("something else"), 6);
    }

    /// The VRAM readout must answer instead of crashing, and trimming the cache
    /// must be safe on a CPU build. With LibTorch linked this prints the real
    /// device numbers.
    #[test]
    fn vram_helpers_answer() {
        match velomocap_motion::vram::cuda_memory() {
            Some((used, total)) => {
                assert!(total > 0, "total device memory must be positive");
                assert!(used <= total, "used {used} exceeds total {total}");
                eprintln!("vram: {used} / {total} bytes in use");
            }
            None => eprintln!("vram: no device visible from this build"),
        }
        velomocap_motion::vram::empty_cuda_cache();
    }

#[cfg(test)]
mod tests {
    use super::*;

    /// The idle preview decodes one frame; a failure here leaves the Movie Clip
    /// panel empty, so it is worth pinning.
    #[test]
    fn probes_a_sample_clip() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/media/test-clip.mp4");
        if !path.is_file() {
            eprintln!("skipping: no sample clip at {}", path.display());
            return;
        }
        let probe = probe_frame(&path, 320).expect("probing the first frame");
        assert!(probe.frame.width > 0 && probe.frame.height > 0);
        assert!(probe.frames > 0, "frame count");
        assert!(probe.fps > 0.0, "fps");
        eprintln!(
            "probe: {}x{}, {:.2} fps, {} frames",
            probe.frame.width, probe.frame.height, probe.fps, probe.frames
        );
    }

    /// "Apply Correction" without the model: a take's source is re-solved, skinned
    /// and baked into a complete take, and the source is left untouched so the
    /// action can be repeated with a different preset.
    #[test]
    fn applying_correction_rebuilds_a_take_from_its_source() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/body_models/smplh/neutral/model.npz");
        if !path.is_file() {
            eprintln!("skipping: no SMPL-H model at {}", path.display());
            return;
        }
        let frames = 18usize;
        let body = velomocap_body::BodyModel::load(&path, 16).expect("body");
        // A neutral body hovering 3 cm over the floor, both feet confidently
        // planted: the correction has something to do, and it is cheap to build.
        let mut pose = velomocap_body::Pose::neutral(body.num_joints, 16);
        let flat = body.forward_joints(&pose);
        let lowest = flat
            .positions
            .chunks_exact(3)
            .map(|point| point[1])
            .fold(f32::INFINITY, f32::min);
        pose.translation = [0.0, 0.03 - lowest, 0.0];
        let source = velomocap_motion::FlowClip {
            fps: 30.0,
            poses: vec![pose.clone(); frames],
            skeletons: vec![body.forward_joints(&pose); frames],
            rot6d: Vec::new(),
            trans: vec![[0.0; 3]; frames],
            betas: vec![0.0; 16],
            foot_contacts: vec![[1.0f32; 4]; frames],
            floor: 0.0,
            floor_from_contacts: true,
            offset: [0.0; 3],
            source_frames: (0..frames).collect(),
            anchors: 0,
            anchor_residual: [0.0; 3],
        };
        let before = source.clone();
        let takes = vec![TakeSource {
            name: "Person 1".into(),
            first_frame: 0,
            source: Arc::new(source),
        }];
        let settings = velomocap_motion::correction::CorrectionSettings::default();
        let locks = velomocap_motion::ContactTracks::automatic();
        let rebuilt = recorrect(&takes, &path, &settings, &locks).expect("recorrect");
        assert_eq!(rebuilt.len(), 1);
        let take = &rebuilt[0];
        assert_eq!(take.name, "Person 1");
        assert_eq!(take.frames.len(), frames, "every frame is re-skinned");
        assert!(
            take.frames.iter().all(|vertices| !vertices.is_empty()),
            "a skinned body is never empty"
        );
        assert_eq!(take.skeletons.len(), frames);
        assert_eq!(take.bvh.frames(), frames, "the BVH is rebuilt too");
        assert!(
            take.summary.contains("correction:"),
            "the summary carries the pass: {}",
            take.summary
        );
        assert!(
            take.summary.contains("ground"),
            "the grounding shift is in the report: {}",
            take.summary
        );
        // The solver input survives, so the action can be repeated.
        assert_eq!(take.source.poses.len(), frames);
        assert_eq!(
            take.source.poses[0].translation, before.poses[0].translation,
            "the source must not be corrected in place"
        );
    }

    /// A take that came back from a project carries poses but no skeletons: the
    /// solver has to run the forward pass before its first probe reads them.
    #[test]
    fn applying_correction_fills_skeletons_for_a_reopened_take() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/body_models/smplh/neutral/model.npz");
        if !path.is_file() {
            eprintln!("skipping: no SMPL-H model at {}", path.display());
            return;
        }
        let frames = 6usize;
        let body = velomocap_body::BodyModel::load(&path, 16).expect("body");
        let pose = velomocap_body::Pose::neutral(body.num_joints, 16);
        let source = velomocap_motion::FlowClip {
            fps: 30.0,
            poses: vec![pose; frames],
            // A project stores the poses only.
            skeletons: Vec::new(),
            rot6d: Vec::new(),
            trans: vec![[0.0; 3]; frames],
            betas: vec![0.0; 16],
            foot_contacts: vec![[1.0f32; 4]; frames],
            floor: 0.0,
            floor_from_contacts: true,
            offset: [0.0; 3],
            source_frames: (0..frames).collect(),
            anchors: 0,
            anchor_residual: [0.0; 3],
        };
        let takes = vec![TakeSource {
            name: "Person 1".into(),
            first_frame: 0,
            source: Arc::new(source),
        }];
        let settings = velomocap_motion::correction::CorrectionSettings::default();
        let rebuilt = recorrect(
            &takes,
            &path,
            &settings,
            &velomocap_motion::ContactTracks::automatic(),
        )
        .expect("recorrect from a project take");
        assert_eq!(rebuilt[0].frames.len(), frames);
        assert_eq!(rebuilt[0].skeletons.len(), frames);
    }
}
