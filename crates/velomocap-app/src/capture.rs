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
pub const STAGE_COUNT: usize = 6;

pub fn stage_name(index: usize) -> &'static str {
    match index {
        0 => "loading models",
        1 => "reconstructing people",
        2 => "estimating camera",
        3 => "generating motion",
        4 => "skinning",
        _ => "done",
    }
}

fn stage_index(stage: &str) -> usize {
    match stage {
        "loading mesh" | "loading detector" | "loading SAM 3D Body" | "opening video"
        | "flow body model" | "flow camera" => 0,
        "detecting" | "reconstructing" => 1,
        "camera" => 2,
        "flow" => 3,
        "skin" => 4,
        _ => 5,
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
    /// Estimate the camera with VGGT; `false` treats it as a fixed camera.
    pub estimate_camera: bool,
    /// Smooth the estimated camera track over time.
    pub refine_camera: bool,
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
        let started = started;
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
            state.progress.stage_index = stage_index(stage);
            state.progress.frame = frame;
            state.progress.total = total;
            state.progress.message = stage.to_string();
            state.progress.elapsed = started.elapsed().as_secs_f64();
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
        ..Default::default()
    };

    let mut clips = Vec::new();
    for (slot, person) in people.iter().enumerate() {
        report_stage(shared, 3, slot, people.len(), "flow");
        let sentinel = || {
            if let Ok(state) = shared.lock() {
                if state.progress.stage == CaptureStage::Failed {
                    return true;
                }
            }
            false
        };
        let mut clip = match velomocap_motion::generate_flow(person, &camera, &model, &config) {
            Ok(clip) => clip,
            Err(error) => {
                if let Ok(mut state) = shared.lock() {
                    state.flow_note = format!("FlowHMR {}: {error}", person.name);
                }
                continue;
            }
        };
        // Contact correction runs on the generated clip, so the viewport and
        // every export see the same corrected motion.
        let correction = match velomocap_motion::correction::apply(
            &mut clip,
            &model,
            &settings.correction,
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
        for (pose_index, pose) in clip.poses.iter().enumerate() {
            match model.forward_mesh(pose) {
                Ok((_, vertices)) => bodies.push(vertices),
                Err(_) => bodies.push(Vec::new()),
            }
            // Skinning 90 frames of a 14k-triangle body is not instant; let the
            // bar move instead of sitting at "generating motion".
            if pose_index % 8 == 0 {
                report_stage(shared, 4, pose_index, poses, "skin");
            }
        }
        let mut summary = velomocap_motion::summarise_flow(&clip);
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
                "reconstructing people",
                "estimating camera",
                "generating motion",
                "skinning",
                "done",
            ]
        );
        assert_eq!(stage_index("loading detector"), 0);
        assert_eq!(stage_index("detecting"), 1);
        assert_eq!(stage_index("camera"), 2);
        assert_eq!(stage_index("flow"), 3);
        assert_eq!(stage_index("skin"), 4);
        assert_eq!(stage_index("done"), 5);
        assert_eq!(stage_index("something else"), 5);
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
}
