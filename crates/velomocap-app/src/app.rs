use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use eframe::egui;
use velomocap_infer::MeshTopology;
use velomocap_motion::{PersonFrame, SamConfig};
use crate::capture::{self, CaptureJob, CaptureStage};
use crate::i18n;
use crate::ui;
use crate::viewport::{Layers, Orbit, ViewportCache, PERSON_COLOURS};

#[derive(Clone)]pub struct PersonView {
    pub id: usize,
    pub name: String,
    pub colour: egui::Color32,
    pub visible: bool,
        pub frames: Vec<Option<Arc<PersonFrame>>>,
    /// The character that replaces this person's built-in body, if one was
    /// imported. Kept per person: the reference lets each person carry its own
    /// avatar, and the path is saved beside the clip.
    pub character: Option<AssignedCharacter>,}

/// What a running character-asset export publishes while it writes the files,
/// so the window can show which one is being written instead of freezing.
#[derive(Clone, Default)]
pub struct AssetProgress {
    pub done: usize,
    pub total: usize,
    /// The file being written right now.
    pub current: String,
    pub written: usize,
    pub failures: Vec<String>,
    pub finished: bool,
    pub cancelled: bool,
    /// Where the files go, for the result line.
    pub folder: String,
}

/// A character-asset export running on its own thread.
#[derive(Clone)]
pub struct AssetJob {
    pub progress: Arc<Mutex<AssetProgress>>,
}

/// An imported character bound to one person.
#[derive(Clone)]
pub struct AssignedCharacter {
    pub path: PathBuf,
    pub character: Arc<crate::character::Character>,
}

/// An imported character kept in the workspace, so switching a person's model
/// never means finding the file on disk again. The reference's character list is
/// the same idea: every import stays available for as long as the project lives.
#[derive(Clone)]
pub struct LibraryCharacter {
    pub path: PathBuf,
    pub character: Arc<crate::character::Character>,
    /// Shipped beside the executable in `models/characters/`, rather than
    /// imported by the user. The list shows the two groups apart.
    pub builtin: bool,
}

/// Identity of a shelf entry, for keeping one copy of a model: the file name and
/// its size. The same avatar arrives from several places — the install folder,
/// a development build, a project file — and every one of those paths is a
/// different string for the same model.
fn same_model(path: &Path, other: &Path) -> bool {
    if path == other {
        return true;
    }
    let name = |path: &Path| path.file_name().map(|name| name.to_os_string());
    if name(path) != name(other) || name(path).is_none() {
        return false;
    }
    match (std::fs::metadata(path), std::fs::metadata(other)) {
        (Ok(a), Ok(b)) => a.len() == b.len(),
        _ => false,
    }
}

/// Whether `path` sits in the character folder shipped with this executable.
fn is_builtin_path(path: &Path) -> bool {
    let Some(folder) = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|parent| parent.join("models").join("characters")))
    else {
        return false;
    };
    path.parent() == Some(folder.as_path())
}
/// One "apply correction" run: the rebuilt takes, or why it failed.
#[derive(Default)]
pub struct RecorrectShared {
    pub finished: bool,
    pub error: Option<String>,
    pub takes: Vec<crate::capture::FlowClipData>,
    pub elapsed: f64,
}

/// The request itself lives in the app while the worker owns the model.
pub struct RecorrectJob {
    pub handle: Option<std::thread::JoinHandle<()>>,
    pub shared: Arc<Mutex<RecorrectShared>>,
}

impl RecorrectJob {
    pub fn is_running(&self) -> bool {
        self.handle
            .as_ref()
            .map(|handle| !handle.is_finished())
            .unwrap_or(false)
    }
}

#[derive(Clone)]pub struct FlowBody {
    pub name: String,
        pub first_frame: usize,
        pub frames: Arc<Vec<Vec<f32>>>,
        pub summary: String,
        pub bvh: Arc<velomocap_motion::PreparedBvh>,
    /// The clip's joint rotations per frame, for posing an imported character.
    pub skeletons: Arc<Vec<velomocap_body::Skeleton>>,
    /// Automatic contact runs per track, for the timeline's "automatic" rows.
    pub contacts: [Vec<[usize; 2]>; velomocap_motion::TRACK_COUNT],
    /// The generated motion before the contact correction, when this take was
    /// generated in this session. `None` for a take loaded from a project: the
    /// project stores the baked result, not the solver's input.
    pub source: Option<Arc<velomocap_motion::FlowClip>>,
}
impl FlowBody {
        pub fn frame(&self, playhead: usize) -> Option<&[f32]> {
        let index = playhead.checked_sub(self.first_frame)?;
        self.frames.get(index).map(|verts| verts.as_slice())
    }
    /// The clip's pose for `playhead`, held at the first and last baked frame.
    ///
    /// Outside the generated clip the viewport used to drop back to the
    /// per-frame MHR body: a different body, with different topology and its own
    /// per-frame jitter. Swapping between the two in one playback reads as the
    /// figure twitching, and it happened exactly at the clip edges — the first
    /// frames of every loop. Holding the nearest pose keeps one body on screen
    /// for the whole clip.
    pub fn frame_clamped(&self, playhead: usize) -> Option<&[f32]> {
        if self.frames.is_empty() {
            return None;
        }
        let index = playhead
            .saturating_sub(self.first_frame)
            .min(self.frames.len() - 1);
        // Skip empty bake results: an empty slice draws nothing, which is how a
        // take whose last frames failed to skin turned into "the model is gone".
        (0..=index).rev().find_map(|at| {
            self.frames
                .get(at)
                .filter(|verts| !verts.is_empty())
                .map(|verts| verts.as_slice())
        })
    }}
impl PersonView {
    pub fn frame(&self, index: usize) -> Option<&Arc<PersonFrame>> {
        self.frames.get(index).and_then(|slot| slot.as_ref())
    }
        pub fn nearest(&self, index: usize) -> Option<&Arc<PersonFrame>> {
        let last = index.min(self.frames.len().saturating_sub(1));
        (0..=last)
            .rev()
            .find_map(|at| self.frames.get(at).and_then(|slot| slot.as_ref()))
    }
    pub fn hits(&self) -> usize {
        self.frames.iter().filter(|slot| slot.is_some()).count()
    }}
struct SeekService {
    requests: std::sync::mpsc::Sender<(u64, PathBuf, usize)>,
    results: std::sync::mpsc::Receiver<(u64, usize, capture::PreviewFrame)>,
    pending: Option<usize>,
    last_request: Instant,
            generation: u64,}
impl SeekService {
    fn start() -> Self {
        let (requests, request_queue) = std::sync::mpsc::channel::<(u64, PathBuf, usize)>();
        let (result_queue, results) = std::sync::mpsc::channel();
        std::thread::Builder::new()
            .name("velomocap-seek".into())
            .spawn(move || {
                let tools = velomocap_video::Tools::discover().ok();
                while let Ok(mut job) = request_queue.recv() {
                                        while let Ok(newer) = request_queue.try_recv() {
                        job = newer;
                    }
                    let Some(tools) = tools.as_ref() else { continue };
                    if let Some(frame) =
                        velomocap_video::decode_frame_at(tools, &job.1, job.2, Some(640))
                    {
                        let preview = capture::preview_from_frame(&frame, 480);
                        if result_queue.send((job.0, job.2, preview)).is_err() {
                            break;
                        }
                    }
                }
            })
            .ok();
        Self {
            requests,
            results,
            pending: None,
            last_request: Instant::now(),
            generation: 0,
        }
    }
        fn restart(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        self.pending = None;
        self.last_request = Instant::now() - Duration::from_secs(1);
    }}
fn detect_people(
    video: &Path,
    detector_path: &Path,) -> std::result::Result<Vec<Candidate>, String> {
    let tools = velomocap_video::Tools::discover().map_err(|error| format!("{error:#}"))?;
    let info = velomocap_video::probe(&tools, video).map_err(|error| format!("ffprobe: {error:#}"))?;
    let frame = velomocap_video::decode_frame_at(&tools, video, 0, Some(960)).ok_or_else(|| {
        format!(
            "could not decode frame 0 of a {}
x{} clip through {}",
            info.width,
            info.height,
            tools.ffmpeg.display()
        )
    })?;
    let mut detector = velomocap_infer::Detector::load(
        detector_path,
        velomocap_infer::DetectorConfig::default(),
    )
    .map_err(|error| format!("{error:#}"))?;
    let detections = detector
        .detect(&frame.rgb, frame.width, frame.height)
        .map_err(|error| format!("{error:#}"))?;
    let (width, height) = (frame.width.max(1) as f32, frame.height.max(1) as f32);
    Ok(detections
        .into_iter()
        .map(|detection| Candidate {
            bbox: [detection.x0, detection.y0, detection.x1, detection.y1],
            norm: [
                detection.x0 / width,
                detection.y0 / height,
                detection.x1 / width,
                detection.y1 / height,
            ],
            score: detection.score,
            selected: true,
        })
        .collect())}
#[derive(Clone, Debug)]
pub struct Candidate {
    /// Box in the pixels of the frame the detector saw.
    pub bbox: [f32; 4],
    /// The same box in 0..1 image coordinates, independent of decode size.
    pub norm: [f32; 4],
    pub score: f32,
    pub selected: bool,
}

/// One whole-track edit, so an undo restores the track exactly.
#[derive(Clone, Debug)]
pub struct LockEdit {
    pub track: usize,
    /// `None` = the track was automatic before the edit.
    pub before: Option<Vec<velomocap_motion::Interval>>,
    pub after: Option<Vec<velomocap_motion::Interval>>,
}

/// Which part of a lock a drag grabbed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LockGrab {
    /// Empty track: a new lock grows from where the press landed.
    New,
    /// Between the two edges: the whole lock follows the pointer.
    Move,
    Start,
    End,
}

/// A drag on a lock row, held until the button comes up.
#[derive(Clone, Debug)]
pub struct LockDrag {
    pub track: usize,
    pub grab: LockGrab,
    /// Frame the press landed on.
    pub anchor: usize,
    /// The lock being edited, if any.
    pub index: Option<usize>,
    pub before: Option<Vec<velomocap_motion::Interval>>,
}
pub struct VeloMocap {
    pub video: Option<PathBuf>,
    /// The `.vp` this workspace was saved to or opened from, for the title and
    /// for "save" without asking again.
    pub project_path: Option<PathBuf>,
    /// `--save-project`: write the workspace once the capture finishes, then
    /// leave. Used to verify the round trip without a GUI.
    pub save_after_capture: Option<PathBuf>,
    /// How the next project save writes geometry and compresses it.
    pub storage_precision: crate::project::Precision,
    pub storage_compression: crate::project::Compression,
    pub detector: PathBuf,
    pub sam3d_dir: PathBuf,
    pub params_dir: PathBuf,
    pub confidence: f32,
    pub gpu: bool,
    /// How much of the pipeline may stay resident on the device. Chosen from the
    /// card's size the first time the app starts, and overridable in the UI.
    pub memory_mode: capture::MemoryMode,
    pub flow_weights: PathBuf,
    pub flow_stats: PathBuf,
    pub smplh_model: PathBuf,
    pub camera_weights: PathBuf,
    pub use_flow: bool,
    /// Foot/hand contact correction, from the sidebar.
    pub correction: velomocap_motion::correction::CorrectionSettings,
    /// Sidebar: estimate the camera (VGGT) instead of assuming a fixed one.
    pub estimate_camera: bool,
    /// Sidebar: smooth the estimated camera rotation.
    pub refine_camera: bool,
    pub scale_width: usize,
        pub max_frames: usize,
    pub capture: Option<CaptureJob>,
    pub progress: (usize, usize, String),
    /// Index of the stage the run is in, for [\`capture::stage_name\`].
    pub progress_stage: usize,
    /// Device memory between stages, e.g. \`1.2 / 24.0 GB · after SAM\`.
    pub vram: String,
        pub preview: Option<egui::TextureHandle>,
    pub preview_frame: usize,
    pub video_size: (usize, usize),
    pub video_fps: f64,
    pub video_frames: usize,
        pub first_frame: Option<Arc<Mutex<Option<Result<capture::Probe, String>>>>>,
            stream: crate::stream::VideoStream,
            play_from: Option<usize>,
    framed_selected: usize,
    pub last_refresh: Instant,
    pub people: Vec<PersonView>,
    pub mesh: Option<Arc<MeshTopology>>,
        pub flow: Vec<FlowBody>,
        pub smplh_faces: Option<Arc<Vec<[u32; 3]>>>,
        pub flow_note: String,
    pub selected: usize,
        pub floor: f32,
    pub capture_fps: f64,
        pub framed_after_capture: bool,
    /// Whether the reconstruction owned the Movie Clip pane on the previous
    /// frame, so the moment it stops can be detected once.
    capture_live: bool,
    pub capture_started: Option<Instant>,
    pub playhead: usize,
    pub playing: bool,
    pub candidates: Vec<Candidate>,
        pub detecting: bool,
        pub ask_clear: bool,
    pub pending_import: Option<PathBuf>,
    pub detect_rx: Option<std::sync::mpsc::Receiver<Result<Vec<Candidate>, String>>>,
    pub play_speed: f32,
    /// Anchor of the wall-clock transport: the frame playback started from and
    /// when it started. `None` while paused.
    play_anchor: Option<(usize, Instant)>,
    /// Speed the anchor was taken with, so a speed change re-anchors instead of
    /// jumping to where the old anchor predicts.
    last_speed: f32,
    pub orbit: Orbit,
    pub layers: Layers,
    pub view_mode: usize,
    pub view: ViewportCache,
    pub status: String,
    pub error: Option<String>,
    pub export_note: Option<String>,
    /// The last capture summary already written to the log, so the correction
    /// report is logged once per run instead of on every paint.
    pub logged_summary: String,
    /// Hand-drawn contact timing for the selected clip: four sparse tracks of
    /// frame intervals (left foot, right foot, left hand, right hand).
    pub locks: velomocap_motion::ContactTracks,
    /// The lock the user is working on, as `(track, index)`.
    pub selected_lock: Option<(usize, usize)>,
    /// Whole-track snapshots, so one edit is one undo step.
    pub lock_undo: Vec<LockEdit>,
    pub lock_redo: Vec<LockEdit>,
    /// A drag in progress on one of the lock rows.
    pub lock_drag: Option<LockDrag>,
    /// Loaded lazily, once the probe has reported the frame count.
    locks_pending: bool,
    /// The locks changed since the clip was generated: the next run applies them.
    pub locks_dirty: bool,
    /// When the playhead last moved on its own, so a seek that never produced a
    /// picture can be retried instead of leaving the pane on the old frame.
    seek_at: Instant,
    seek_tries: u8,
    pub console: Vec<String>,
    pub show_console: bool,
    pub show_help: bool,
    pub show_guide: bool,
    /// Settings that belong to the application rather than to one panel: how a
    /// project is written, and anything else that is not a per-clip choice.
    pub show_settings: bool,
    /// The About window: project, repository and sponsor links.
    pub show_about: bool,
    /// The export window: what to write, where, and under which name.
    pub show_export: bool,
    /// An imported character (VRM / glTF) that replaces the built-in body in the
    /// viewport, the reference's "Import Character". The rest rotations turn a
    /// clip rotation into the delta a character is posed with.
    pub character_rest: Option<Arc<Vec<[f32; 9]>>>,
    /// The same rest pose's joint positions, which is what tells an imported
    /// character's bind pose how far it has to swing to match the clip.
    pub character_rest_positions: Option<Arc<Vec<[f32; 3]>>>,
    /// The clip's joint parents, which make each joint's rotation local.
    pub character_rest_parents: Option<Arc<Vec<i32>>>,
    /// The person whose character panel is open.
    pub person_panel: Option<usize>,
    /// Every character imported into this machine's library (VRM / glTF / FBX /
    /// PMX), whether or not a person is wearing it right now. The list lives
    /// outside the project, so one import is available to every project.
    pub characters: Vec<LibraryCharacter>,
    /// The shared library has been read from disk once.
    pub library_loaded: bool,
    /// The models shipped beside the executable have been added.
    pub builtins_loaded: bool,
    /// The library entry whose detail panel is open.
    pub character_panel: Option<usize>,
    /// Which page the movie-clip column shows: 0 people, 1 characters, 2 stream.
    pub left_tab: usize,
    /// Text for the drop overlay while files hover over the window.
    pub drop_hint: Option<String>,
    /// The saved character map has been applied to the current people.
    pub characters_loaded: bool,
    /// What a character asset export writes: mesh with skin weights, materials
    /// with their textures, and the animation. All on by default, which is what
    /// a game engine needs to import the model and play the take.
    pub asset_skin: bool,
    pub asset_materials: bool,
    pub asset_animation: bool,
    /// Which people's models to write; one file each.
    pub asset_selected: Vec<bool>,
    /// Where they land; empty means beside the clip.
    pub asset_folder: String,
    /// What the last asset export did, for the panel.
    pub asset_status: String,
    /// The export running right now, and what it has written so far.
    pub asset_job: Option<AssetJob>,
    /// Per person, an optional file name for the exported model; empty means the
    /// model's own name.
    pub asset_names: Vec<String>,
    pub tools_ok: bool,
    pub ort_path: Option<PathBuf>,
    pub cuda_ok: bool,
    /// The "apply correction" run in flight, if any.
    pub recorrect: Option<RecorrectJob>,
    /// What the last apply said, for the sidebar and the console.
    pub recorrect_note: String,
}
impl VeloMocap {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        ui::configure(&cc.egui_ctx);
        let CliArgs {
            video: initial_video,
            capture_start,
            max_frames,
            play,
            frame,
            detect_only,
            select,
            play_check,
            render_probe,
            render_character,
            export_asset,
            export_animated,
            render_pose,
            save_project,
            open_project,
            storage_precision,
            storage_compression,
        } = cli_args();
        let tools_ok = velomocap_video::Tools::discover().is_ok();
        let ort_path = velomocap_infer::find_runtime_library();
        let cuda_ok = velomocap_infer::cuda_provider_available();
        let detector = resolve_model(&["models/detector/yolov8x.onnx", "models/detector/yolov8n.onnx"])
            .unwrap_or_else(|| PathBuf::from("models/detector/yolov8x.onnx"));
        let sam3d_dir = resolve_model(&["models/sam3d"])
            .unwrap_or_else(|| PathBuf::from("models/sam3d"));
        let params_dir = resolve_model(&["models/flowhmr/sam"])
            .unwrap_or_else(|| PathBuf::from("models/flowhmr/sam"));
        let video = initial_video;
        let status = match (&ort_path, tools_ok, &video) {
            (Some(_), true, None) => i18n::t("Ready. Open a video or a project.").to_string(),
            (Some(_), true, Some(path)) => i18n::selected_line(&path.display().to_string()),
            (None, _, _) => i18n::t("ONNX Runtime not found: runtime/onnxruntime is missing.").to_string(),
            (_, false, _) => i18n::t("ffmpeg not found: runtime/ffmpeg is missing.").to_string(),
        };
        let flow_weights = resolve_model(&["models/flowhmr/flowhmr_mmdit.onnx"])
            .unwrap_or_else(|| PathBuf::from("models/flowhmr/flowhmr_mmdit.onnx"));
        let flow_stats = resolve_model(&["models/flowhmr/motion_stats.json"])
            .unwrap_or_else(|| PathBuf::from("models/flowhmr/motion_stats.json"));
        let smplh_model = resolve_model(&["models/body_models/smplh/neutral/model.npz"])
            .unwrap_or_else(|| PathBuf::from("models/body_models/smplh/neutral/model.npz"));
        let camera_weights = resolve_model(&["models/flowhmr/vggt_omega_1b_512.onnx"])
            .unwrap_or_else(|| PathBuf::from("models/flowhmr/vggt_omega_1b_512.onnx"));
        // The device's size decides the default residency strategy; the sidebar
        // combo lets the user override it. `VELOMOCAP_FORCE_CPU` still wins.
        let device_memory = velomocap_motion::vram::cuda_memory().map(|(_, total)| total);
        let mut memory_mode = capture::MemoryMode::from_vram(device_memory);
        if std::env::var("VELOMOCAP_FORCE_CPU").is_ok() {
            memory_mode = capture::MemoryMode::Cpu;
        }
        let mut app = Self {
            video,
            detector,
            sam3d_dir,
            params_dir,
            confidence: 0.3,
            gpu: memory_mode.uses_gpu(),
            memory_mode,
            flow_weights,
            flow_stats,
            smplh_model,
            camera_weights,
            use_flow: true,
            correction: velomocap_motion::correction::CorrectionSettings::default(),
            estimate_camera: false,
            refine_camera: true,
            scale_width: 960,
            max_frames,
            capture: None,
            progress: (0, 0, String::new()),
            progress_stage: 0,
            vram: String::new(),
            preview: None,
            preview_frame: 0,
            video_size: (0, 0),
            video_fps: 0.0,
            video_frames: 0,
            first_frame: None,
            project_path: None,
            save_after_capture: save_project.clone(),
            storage_precision: crate::project::Precision::default(),
            storage_compression: crate::project::Compression::default(),
            stream: crate::stream::VideoStream::spawn(),
            play_from: None,
            framed_selected: 0,
            last_refresh: Instant::now(),
            people: Vec::new(),
            mesh: None,
            flow: Vec::new(),
            smplh_faces: None,
            flow_note: String::new(),
            selected: 0,
            floor: 0.0,
            capture_fps: 0.0,
            framed_after_capture: false,
            capture_live: false,
            capture_started: None,
            playhead: frame,
            playing: play,
            candidates: Vec::new(),
            detecting: false,
            ask_clear: false,
            pending_import: None,
            detect_rx: None,
            play_speed: 1.0,
            play_anchor: None,
            last_speed: 1.0,
            orbit: Orbit::default(),
            layers: Layers::default(),
            view_mode: 0,
            view: ViewportCache::default(),
            status,
            error: None,
            tools_ok,
            ort_path,
            cuda_ok,
            export_note: None,
            logged_summary: String::new(),
            locks: velomocap_motion::ContactTracks::automatic(),
            selected_lock: None,
            lock_undo: Vec::new(),
            lock_redo: Vec::new(),
            lock_drag: None,
            locks_pending: true,
            locks_dirty: false,
            seek_at: Instant::now(),
            seek_tries: 0,
            console: Vec::new(),
            show_console: false,
            show_help: false,
            show_guide: false,
            show_settings: false,
            show_about: false,
            asset_job: None,
            show_export: false,
            character_rest: None,
            character_rest_positions: None,
            character_rest_parents: None,
            person_panel: None,
            characters: Vec::new(),
        library_loaded: false,
        builtins_loaded: false,
            character_panel: None,
            left_tab: 0,
            drop_hint: None,
            characters_loaded: false,
        asset_skin: true,
        asset_materials: true,
        asset_animation: true,
        asset_selected: Vec::new(),
        asset_folder: String::new(),
        asset_status: String::new(),
        asset_names: Vec::new(),
        recorrect: None,
        recorrect_note: String::new(),
        };
        if let Some(out) = render_probe.clone() {
            let model = app.smplh_model.display().to_string();
            if let Err(error) = crate::probe::render_rest_pose(&model, &out) {
                eprintln!("render-probe failed: {error:#}");
            }
            std::process::exit(0);
        }
        if let Some((file, project, out)) = export_animated.clone() {
            let model = app.smplh_model.display().to_string();
            if let Err(error) = crate::probe::export_animation(
                &model,
                &file,
                &project,
                &out,
                crate::asset::Options::default(),
                app.current_fps() as f32,
            ) {
                eprintln!("export-animated failed: {error:#}");
                std::process::exit(1);
            }
            std::process::exit(0);
        }
        if let Some((file, out)) = export_asset.clone() {
            match crate::character::Character::load(std::path::Path::new(&file)) {
                Ok(character) => {
                    let options = crate::asset::Options::default();
                    let result =
                        crate::asset::write_glb(&character, &[], options, 30.0, std::path::Path::new(&out));
                    if let Err(error) = result {
                        eprintln!("export-asset failed: {error:#}");
                        std::process::exit(1);
                    }
                    eprintln!(
                        "export-asset: {} — {} verts, {} faces -> {out}",
                        character.name,
                        character.vertex_count(),
                        character.faces.len()
                    );
                }
                Err(error) => {
                    eprintln!("export-asset failed: {error:#}");
                    std::process::exit(1);
                }
            }
            std::process::exit(0);
        }
        if let Some((file, out)) = render_character.clone() {
            if let Err(error) = crate::probe::render_character(&file, &out) {
                eprintln!("render-character failed: {error:#}");
                std::process::exit(1);
            }
            std::process::exit(0);
        }
        if let Some((file, project, frame, out)) = render_pose.clone() {
            let model = app.smplh_model.display().to_string();
            if let Err(error) = crate::probe::render_pose(&model, &file, &project, frame, &out) {
                eprintln!("render-pose failed: {error:#}");
                std::process::exit(1);
            }
            std::process::exit(0);
        }
        app.log("VeloMocap native core ready.");
        match device_memory {
            Some(total) => app.log(format!(
                "Memory mode: {} ({:.1} GB device, auto-selected)",
                memory_mode.label(),
                total as f64 / 1e9
            )),
            None => app.log(format!(
                "Memory mode: {} (no CUDA device visible)",
                memory_mode.label()
            )),
        }
        if let Some(path) = open_project.clone() {
            // A saved workspace brings its own video and clips; the argument is
            // only used when there is no project to restore.
            app.open_project(path);
            // A command-line override lands after the project's own settings, so
            // the same workspace can be re-saved at every precision to compare.
            if let Some(precision) = storage_precision {
                app.storage_precision = precision;
            }
            if let Some(compression) = storage_compression {
                app.storage_compression = compression;
            }
            // Re-saving a workspace that was just opened is the cheapest proof
            // that a project round-trips without running anything again.
            if let Some(out) = save_project.clone() {
                app.save_project(out.clone());
                let saved = app.error.is_none();
                eprintln!(
                    "project: {} {}",
                    if saved { "saved" } else { "failed" },
                    out.display()
                );
                std::process::exit(if saved { 0 } else { 1 });
            }

        } else if let Some(path) = app.video.clone() {
            app.load_video(path);
                        app.playing = play;
        }
        // A command-line override lands after any project was opened, so a save
        // can be measured at every precision without editing the project first.
        if let Some(precision) = storage_precision {
            app.storage_precision = precision;
        }
        if let Some(compression) = storage_compression {
            app.storage_compression = compression;
        }
        if detect_only && app.video.is_some() {
            app.detect_only();
        }
        if let Some(index) = select {
            for (slot, candidate) in app.candidates.iter_mut().enumerate() {
                candidate.selected = slot + 1 == index;
            }
            eprintln!(
                "selected candidate {index} of {}",
                app.candidates.len()
            );
        }
        if let Some(frames) = play_check {
            if let Some(video) = app.video.clone() {
                let fps = app.current_fps();
                let started = std::time::Instant::now();
                let mut seen = 0usize;
                app.stream.request(&video, 0, true, 1.0, 480, fps);
                while seen < frames {
                    if let Some((_, frame)) = app.stream.poll() {
                        seen += 1;
                        eprintln!(
                            "play-check frame {} at {:.3} s",
                            frame.index,
                            started.elapsed().as_secs_f64()
                        );
                    } else {
                        std::thread::sleep(std::time::Duration::from_millis(2));
                    }
                }
                std::process::exit(0);
            }
        }
        if capture_start && app.video.is_some() {
            app.start_capture();
        }
        app
    }
    pub fn log(&mut self, message: impl Into<String>) {
        let message = message.into();
        let stamp = format!("[{}] ", chrono_like_clock());
        self.console.push(format!("{stamp}{message}"));
        if self.console.len() > 500 {
            self.console.remove(0);
        }
    }
    /// Import a humanoid character (VRM / GLB) for one person: its own skeleton
    /// and skinned mesh replace the built-in body in the viewport, driven by the
    /// clip, and the choice is saved beside the clip.
    pub fn import_character_dialog(&mut self, person: Option<usize>) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Humanoid character", &["vrm", "glb", "fbx", "pmx"])
            .set_title("Import character")
            .pick_file()
        else {
            return;
        };
        let Some(index) = self.add_character(path) else {
            return;
        };
        if let Some(person) = person.filter(|person| *person < self.people.len()) {
            self.use_character(person, index);
        }
    }

    /// Load a character into the workspace library, or return the entry that is
    /// already there. The file is read once per workspace.
    pub fn add_character(&mut self, path: PathBuf) -> Option<usize> {
        if let Some(index) = self
            .characters
            .iter()
            .position(|entry| same_model(&entry.path, &path))
        {
            // The same model under another path: keep one copy, and let the
            // copy from this install win, so the entry points at a file that is
            // still here after the build folder is cleaned.
            if is_builtin_path(&path) && !self.characters[index].builtin {
                self.characters[index].path = path;
                self.characters[index].builtin = true;
                self.save_library();
            }
            self.character_panel = Some(index);
            return Some(index);
        }
        match crate::character::Character::load(&path) {
            Ok(character) => {
                let bones = character.bones.len();
                let matched = character.matched();
                let vertices = character.vertex_count();
                let unmatched: Vec<String> = character
                    .unmatched()
                    .into_iter()
                    .map(|name| name.to_string())
                    .collect();
                self.log(format!(
                    "{}: {bones} bones ({matched} matched), {vertices} vertices{}",
                    path.display(),
                    if unmatched.is_empty() {
                        String::new()
                    } else {
                        format!(" · not matched: {}", unmatched.join(", "))
                    }
                ));
                self.characters.push(LibraryCharacter {
                    builtin: is_builtin_path(&path),
                    path,
                    character: Arc::new(character),
                });
                let index = self.characters.len() - 1;
                self.character_panel = Some(index);
                self.save_library();
                Some(index)
            }
            Err(error) => {
                self.error = Some(format!("{error:#}"));
                self.log(format!("character: {error:#}"));
                None
            }
        }
    }

    /// Put a library character on a person: the viewport then draws that model.
    pub fn use_character(&mut self, person: usize, index: usize) {
        if person >= self.people.len() || index >= self.characters.len() {
            return;
        }
        let entry = self.characters[index].clone();
        if self.character_rest.is_none() {
            let (rotations, positions, parents) = self.rest_pose();
            self.character_rest = rotations;
            self.character_rest_positions = positions;
            self.character_rest_parents = parents;
        }
        let bones = entry.character.bones.len();
        let vertices = entry.character.vertex_count();
        self.log(format!(
            "{}: {}",
            self.people[person].name,
            entry.path.display()
        ));
        self.status = format!(
            "{} · {} bones · {} vertices",
            entry
                .path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default(),
            bones,
            vertices
        );
        self.people[person].character = Some(AssignedCharacter {
            path: entry.path,
            character: entry.character,
        });
        self.save_characters();
        // The character replaces the built-in body, so the viewport has to
        // reframe on it.
        self.orbit.framed = false;
    }

    /// Drop a character from the list. A person already wearing it keeps it: the
    /// list is a shelf, not the assignment.
    pub fn remove_character(&mut self, index: usize) {
        if index < self.characters.len() {
            self.characters.remove(index);
            self.save_library();
        }
        self.character_panel = match self.character_panel {
            Some(panel) if panel == index => None,
            Some(panel) if panel > index => Some(panel - 1),
            other => other,
        };
    }

    /// Write every ticked person's model as its own animated `.glb`: mesh with
    /// skin weights, materials with their textures, and the take the viewport is
    /// playing.
    ///
    /// The writes happen on a worker thread: one avatar can take several
    /// seconds, and a frozen window with no sign of which file is being written
    /// is worse than a slow one. [`AssetProgress`] is what the export window
    /// draws while it runs.
    pub fn export_asset_models(&mut self) {
        if self.asset_job.is_some() {
            return;
        }
        let Some(project) = self.project_path.clone() else {
            self.asset_status = i18n::zh_en(
                "先保存工程文件，再导出角色资产",
                "Save the project first, then export the character assets",
            )
            .to_string();
            return;
        };
        let folder = if self.asset_folder.trim().is_empty() {
            project
                .parent()
                .map(|parent| parent.to_path_buf())
                .unwrap_or_default()
        } else {
            PathBuf::from(self.asset_folder.trim())
        };
        if let Err(error) = std::fs::create_dir_all(&folder) {
            self.asset_status = format!("{error}");
            return;
        }
        let options = crate::asset::Options {
            skin: self.asset_skin,
            materials: self.asset_materials,
            animation: self.asset_animation,
        };
        let model = self.smplh_model.display().to_string();
        let fps = self.current_fps() as f32;
        // Collect the work first: the worker must not touch the people list.
        let wanted: Vec<(usize, PathBuf)> = self
            .people
            .iter()
            .enumerate()
            .filter_map(|(index, person)| {
                let assigned = person.character.as_ref()?;
                if !self.asset_selected.get(index).copied().unwrap_or(true) {
                    return None;
                }
                Some((index, assigned.path.clone()))
            })
            .collect();
        if wanted.is_empty() {
            self.asset_status = i18n::zh_en(
                "没有勾选任何角色",
                "No character is ticked",
            )
            .to_string();
            return;
        }
        let jobs: Vec<(String, String, String)> = wanted
            .iter()
            .map(|(index, path)| {
                // A name typed in the panel wins; otherwise the model keeps its
                // own.
                let typed = self
                    .asset_names
                    .get(*index)
                    .map(|name| name.trim().to_string())
                    .unwrap_or_default();
                let stem = if typed.is_empty() {
                    path.file_stem()
                        .map(|stem| stem.to_string_lossy().into_owned())
                        .unwrap_or_else(|| format!("person{}", index + 1))
                } else {
                    typed
                };
                (
                    path.display().to_string(),
                    folder.join(format!("{stem}.glb")).display().to_string(),
                    stem,
                )
            })
            .collect();
        let progress = Arc::new(Mutex::new(AssetProgress {
            total: jobs.len(),
            folder: folder.display().to_string(),
            ..AssetProgress::default()
        }));
        let shared = Arc::clone(&progress);
        let project = project.display().to_string();
        let thread = std::thread::Builder::new()
            .name("asset-export".into())
            .spawn(move || {
                for (character, out, stem) in jobs {
                    {
                        let mut state = shared.lock().unwrap();
                        if state.cancelled {
                            break;
                        }
                        state.current = stem.clone();
                    }
                    let result = crate::probe::export_animation(
                        &model, &character, &project, &out, options, fps,
                    );
                    let mut state = shared.lock().unwrap();
                    match result {
                        Ok(()) => state.written += 1,
                        Err(error) => state.failures.push(format!("{stem}: {error:#}")),
                    }
                    state.done += 1;
                }
                let mut state = shared.lock().unwrap();
                state.finished = true;
                state.current.clear();
            });
        match thread {
            Ok(_) => {
                self.asset_job = Some(AssetJob { progress });
                self.asset_status.clear();
            }
            Err(error) => self.asset_status = format!("{error}"),
        }
    }

    /// Fold a finished export job back into the panel, and keep the window
    /// repainting while it runs.
    pub fn poll_asset_export(&mut self, ctx: &egui::Context) {
        let Some(job) = &self.asset_job else { return };
        let status = {
            let state = job.progress.lock().unwrap();
            if !state.finished {
                None
            } else {
                let failures = state.failures.clone();
                let written = state.written;
                let total = state.total;
                let cancelled = state.cancelled;
                Some((written, total, failures, cancelled))
            }
        };
        match status {
            None => ctx.request_repaint_after(Duration::from_millis(120)),
            Some((written, total, failures, cancelled)) => {
                for failure in failures {
                    self.log(format!("asset: {failure}"));
                }
                self.asset_status = if cancelled {
                    i18n::zh_en(
                        &format!("已取消（已写出 {written}/{total}）"),
                        &format!("Cancelled after {written} of {total}"),
                    )
                } else {
                    i18n::zh_en(
                        &format!("已导出 {written}/{total}"),
                        &format!("Wrote {written} of {total}"),
                    )
                };
                let folder = {
                    let state = self.asset_job.as_ref().unwrap();
                    let state = state.progress.lock().unwrap();
                    state.folder.clone()
                };
                if !folder.is_empty() && written > 0 {
                    self.asset_status.push_str(&format!(" → {folder}"));
                }
                self.asset_job = None;
            }
        }
    }

    /// Stop the running export after the file it is writing.
    pub fn cancel_asset_export(&mut self) {
        if let Some(job) = &self.asset_job {
            job.progress.lock().unwrap().cancelled = true;
        }
    }


    /// Pick the folder the assets land in. A path field is fine for a path the
    /// user has in the clipboard; a DCC export dialog browses.
    pub fn browse_asset_folder(&mut self) {
        let mut dialog = rfd::FileDialog::new();
        if !self.asset_folder.trim().is_empty() {
            dialog = dialog.set_directory(self.asset_folder.trim());
        } else if let Some(parent) = self.project_path.as_ref().and_then(|path| path.parent()) {
            dialog = dialog.set_directory(parent);
        }
        if let Some(folder) = dialog.pick_folder() {
            self.asset_folder = folder.display().to_string();
        }
    }


    /// Put a person back on the built-in body.
    pub fn clear_character(&mut self, person: usize) {
        if let Some(view) = self.people.get_mut(person) {
            view.character = None;
        }
        self.save_characters();
        self.orbit.framed = false;
    }

    pub fn assign_character(&mut self, person: usize, path: PathBuf) {
        if let Some(index) = self.add_character(path) {
            self.use_character(person, index);
        }
    }

    fn characters_path(&self) -> Option<PathBuf> {
        self.video
            .as_ref()
            .map(|video| video.with_extension("characters.json"))
    }

    /// Where the character library lives.
    ///
    /// Not beside the clip: a model is imported once and every project can put it
    /// on a person, so the path list is per machine rather than per project file.
    fn library_path() -> Option<PathBuf> {
        let base = std::env::var_os("APPDATA")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(PathBuf::from))
            .or_else(|| std::env::var_os("USERPROFILE").map(PathBuf::from))?;
        Some(base.join("VeloMocap").join("characters.json"))
    }

    /// Write the library's file list. The models themselves are never copied: a
    /// 28 MB avatar is read from where the user put it.
    pub fn save_library(&mut self) {
        let Some(path) = Self::library_path() else {
            return;
        };
        let entries: Vec<serde_json::Value> = self
            .characters
            .iter()
            .map(|entry| serde_json::json!({ "file": entry.path.display().to_string() }))
            .collect();
        let document = serde_json::json!({ "version": 1, "characters": entries });
        let written = serde_json::to_string_pretty(&document)
            .map_err(|error| error.to_string())
            .and_then(|text| {
                if let Some(folder) = path.parent() {
                    std::fs::create_dir_all(folder).map_err(|error| error.to_string())?;
                }
                std::fs::write(&path, text).map_err(|error| error.to_string())
            });
        if let Err(error) = written {
            self.log(format!("characters: {error}"));
        }
    }

    /// Models shipped beside the executable, so the two Unreal avatars are on
    /// the shelf without anyone having to find the files.
    pub fn load_builtin_characters(&mut self) {
        if self.builtins_loaded {
            return;
        }
        self.builtins_loaded = true;
        let Some(folder) = std::env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(|parent| parent.join("models").join("characters")))
        else {
            return;
        };
        let Ok(entries) = std::fs::read_dir(&folder) else {
            return;
        };
        let mut paths: Vec<PathBuf> = entries
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.is_file())
            .collect();
        paths.sort();
        for path in paths {
            let known = self.characters.iter().any(|entry| entry.path == path);
            if !known {
                self.add_character(path);
            }
        }
        self.character_panel = None;
    }

    /// Read the library once, so every project starts with the same shelf.
    pub fn load_library(&mut self) {
        if self.library_loaded {
            return;
        }
        self.library_loaded = true;
        let Some(path) = Self::library_path() else {
            return;
        };
        let Ok(text) = std::fs::read_to_string(&path) else {
            return;
        };
        let Ok(document) = serde_json::from_str::<serde_json::Value>(&text) else {
            self.log(format!("characters: {} is not valid JSON", path.display()));
            return;
        };
        let entries = document
            .get("characters")
            .and_then(|value| value.as_array())
            .cloned()
            .unwrap_or_default();
        for entry in entries {
            let Some(file) = entry.get("file").and_then(|value| value.as_str()) else {
                continue;
            };
            let file = PathBuf::from(file);
            if !file.is_file() {
                self.log(format!("characters: {} is gone", file.display()));
                continue;
            }
            self.add_character(file);
        }
    }

    /// Save the per-person character choice beside the clip, so re-opening the
    /// video (or re-running the motion) brings the avatars back.
    pub fn save_characters(&mut self) {
        let Some(path) = self.characters_path() else {
            return;
        };
        let people: Vec<serde_json::Value> = self
            .people
            .iter()
            .filter_map(|person| {
                let assigned = person.character.as_ref()?;
                Some(serde_json::json!({
                    "name": person.name,
                    "file": assigned.path.display().to_string(),
                }))
            })
            .collect();
        let document = serde_json::json!({ "version": 1, "people": people });
        match serde_json::to_string_pretty(&document)
            .map_err(|error| error.to_string())
            .and_then(|text| std::fs::write(&path, text).map_err(|error| error.to_string()))
        {
            Ok(()) => {}
            Err(error) => self.log(format!("characters: {error}")),
        }
    }

    /// Apply the saved map once the people exist. A character that disappeared
    /// from disk is reported and skipped rather than failing the run.
    pub fn load_characters(&mut self) {
        if self.characters_loaded || self.people.is_empty() {
            return;
        }
        self.characters_loaded = true;
        let Some(path) = self.characters_path() else {
            return;
        };
        let Ok(text) = std::fs::read_to_string(&path) else {
            return;
        };
        let Ok(document) = serde_json::from_str::<serde_json::Value>(&text) else {
            self.log(format!("characters: {} is not valid JSON", path.display()));
            return;
        };
        let entries = document
            .get("people")
            .and_then(|value| value.as_array())
            .cloned()
            .unwrap_or_default();
        for entry in entries {
            let name = entry.get("name").and_then(|value| value.as_str());
            let file = entry.get("file").and_then(|value| value.as_str());
            let (Some(name), Some(file)) = (name, file) else {
                continue;
            };
            let Some(index) = self.people.iter().position(|person| person.name == name) else {
                continue;
            };
            let file = PathBuf::from(file);
            if !file.is_file() {
                self.log(format!("characters: {name} uses {}, which is gone", file.display()));
                continue;
            }
            match crate::character::Character::load(&file) {
                Ok(character) => {
                    if self.character_rest.is_none() {
                        let (rotations, positions, parents) = self.rest_pose();
                        self.character_rest = rotations;
                        self.character_rest_positions = positions;
                        self.character_rest_parents = parents;
                    }
                    self.people[index].character = Some(AssignedCharacter {
                        path: file,
                        character: Arc::new(character),
                    });
                }
                Err(error) => self.log(format!("characters: {error:#}")),
            }
        }
    }

    /// The body model's rest orientation and joint positions, which turn a clip
    /// rotation into the delta an imported character is posed with.
    fn rest_pose(
        &self,
    ) -> (
        Option<Arc<Vec<[f32; 9]>>>,
        Option<Arc<Vec<[f32; 3]>>>,
        Option<Arc<Vec<i32>>>,
    ) {
        let Ok(body) = velomocap_body::BodyModel::load(&self.smplh_model, 16) else {
            return (None, None, None);
        };
        let pose = velomocap_body::Pose::neutral(body.num_joints, 16);
        let skeleton = body.forward_joints(&pose);
        let mut rotations = Vec::with_capacity(skeleton.rotations.len() / 9);
        for joint in 0..skeleton.rotations.len() / 9 {
            let mut rotation = [0.0f32; 9];
            rotation.copy_from_slice(&skeleton.rotations[joint * 9..joint * 9 + 9]);
            rotations.push(rotation);
        }
        let mut positions = Vec::with_capacity(skeleton.positions.len() / 3);
        for joint in 0..skeleton.positions.len() / 3 {
            positions.push([
                skeleton.positions[joint * 3],
                skeleton.positions[joint * 3 + 1],
                skeleton.positions[joint * 3 + 2],
            ]);
        }
        (Some(Arc::new(rotations)), Some(Arc::new(positions)), Some(Arc::new(body.parents.clone())))
    }

    pub fn open_video_dialog(&mut self) {        let Some(path) = rfd::FileDialog::new()
            .add_filter("Video", &["mp4", "mov", "mkv", "avi", "webm", "m4v"])
            .set_title("Import video")
            .pick_file()
        else {
            return;
        };
        if self.has_scene() {
            self.pending_import = Some(path);
            self.ask_clear = true;
        } else {
            self.load_video(path);
        }
    }
    pub fn has_scene(&self) -> bool {
        !self.people.is_empty() || !self.flow.is_empty()
    }

    /// Files dropped onto the window: a clip becomes the take, a humanoid joins
    /// the character shelf, a `.vp` replaces the workspace. A drop is the
    /// quickest path to all three, so it goes through the same code the menus
    /// use rather than a parallel one.
    pub fn handle_drops(&mut self, ctx: &egui::Context) {
        let dropped = ctx.input(|input| input.raw.dropped_files.clone());
        if dropped.is_empty() {
            self.drop_hint = None;
            return;
        }
        for file in dropped {
            let Some(path) = file.path else {
                continue;
            };
            let extension = path
                .extension()
                .and_then(|extension| extension.to_str())
                .unwrap_or_default()
                .to_ascii_lowercase();
            match extension.as_str() {
                "mp4" | "mov" | "mkv" | "avi" | "webm" | "m4v" => {
                    if self.has_scene() {
                        // Replacing the take throws the reconstruction away, so
                        // ask instead of guessing.
                        self.pending_import = Some(path);
                        self.ask_clear = true;
                    } else {
                        self.finish_import(path, true);
                    }
                }
                "vrm" | "glb" | "fbx" | "pmx" => {
                    let person = if self.people.is_empty() {
                        None
                    } else {
                        Some(self.selected.min(self.people.len() - 1))
                    };
                    if let Some(index) = self.add_character(path.clone()) {
                        if let Some(person) = person {
                            self.use_character(person, index);
                        }
                        // The shelf is where an imported model is visible, so
                        // show it rather than leaving the click unexplained.
                        self.left_tab = 1;
                    }
                }
                "vp" => self.open_project(path),
                other => self.log(i18n::zh_en(
                    &format!("这个文件类型导入不了：.{other}"),
                    &format!("Cannot import this file type: .{other}"),
                )),
            }
        }
    }

    /// What a drag is about to do, for the drop overlay.
    pub fn drop_hint(ctx: &egui::Context) -> Option<String> {
        let files: Vec<String> = ctx.input(|input| {
            input
                .raw
                .hovered_files
                .iter()
                .filter_map(|file| file.path.as_ref())
                .filter_map(|path| {
                    let extension = path
                        .extension()
                        .and_then(|extension| extension.to_str())
                        .unwrap_or_default()
                        .to_ascii_lowercase();
                    let name = path
                        .file_name()
                        .map(|name| name.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    let kind = match extension.as_str() {
                        "mp4" | "mov" | "mkv" | "avi" | "webm" | "m4v" => i18n::t("as the motion clip"),
                        "vrm" | "glb" | "fbx" | "pmx" => i18n::t("as a character"),
                        "vp" => i18n::t("as the whole project"),
                        _ => return None,
                    };
                    Some(format!("{name} · {kind}"))
                })
                .collect()
        });
        if files.is_empty() {
            None
        } else {
            Some(files.join("\n"))
        }
    }
        pub fn finish_import(&mut self, path: PathBuf, clear: bool) {
        if clear {
            self.load_video(path);
        } else {
            self.load_video_keep_scene(path);
        }
    }
        pub fn start_detection(&mut self) {
        let Some(video) = self.video.clone() else {
            return;
        };
        let detector_path = self.detector.clone();
        let (sender, receiver) = std::sync::mpsc::channel();
        self.detect_rx = Some(receiver);
        self.detecting = true;
        self.candidates.clear();
        self.status = i18n::t("Detecting people…").into();
        std::thread::Builder::new()
            .name("velomocap-detect".into())
            .spawn(move || {
                let result = detect_people(&video, &detector_path);
                let _ = sender.send(result);
            })
            .ok();
    }
        pub fn detect_only(&mut self) {
        let Some(video) = self.video.clone() else {
            eprintln!("detect-only: no video");
            return;
        };
        match detect_people(&video, &self.detector) {
            Ok(candidates) => {
                eprintln!("detect-only: {} candidates", candidates.len());
                for (index, candidate) in candidates.iter().enumerate() {
                    eprintln!(
                        "  {}: bbox [{:.0}, {:.0}, {:.0}, {:.0}] score {:.2}",
                        index + 1,
                        candidate.bbox[0],
                        candidate.bbox[1],
                        candidate.bbox[2],
                        candidate.bbox[3],
                        candidate.score
                    );
                }
                self.candidates = candidates;
            }
            Err(error) => eprintln!("detect-only failed: {error}"),
        }
    }
        pub fn poll_detection(&mut self, ctx: &egui::Context) {
        let Some(receiver) = &self.detect_rx else {
            return;
        };
        match receiver.try_recv() {
            Ok(Ok(candidates)) => {
                let count = candidates.len();
                self.candidates = candidates;
                self.detecting = false;
                self.detect_rx = None;
                self.status = i18n::candidate_line(count);
                self.log(format!("Detected {count} people — pick who to reconstruct"));
                if let Some(line) = self.release_vram("after detection") {
                    self.log(format!("VRAM {line}"));
                }
            }
            Ok(Err(error)) => {
                self.detecting = false;
                self.detect_rx = None;
                self.error = Some(format!("detection failed: {error}"));
                if let Some(line) = self.release_vram("after detection") {
                    self.log(format!("VRAM {line}"));
                }
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                // Detection loads the detector and runs it once; without asking
                // for a repaint the window looks frozen until it lands.
                ctx.request_repaint_after(Duration::from_millis(16));
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.detecting = false;
                self.detect_rx = None;
            }
        }
    }
    /// Trim the CUDA cache between stages and report what the device shows.
    ///
    /// The detector, the camera and the motion model all leave their blocks in
    /// the allocators' pools when they are dropped, so a 16 GB card runs out even
    /// though nothing is "using" the memory any more. Returns the
    /// `used / total · stage` line so callers can log it.
    pub fn release_vram(&mut self, stage: &str) -> Option<String> {
        velomocap_motion::vram::empty_cuda_cache();
        let line = velomocap_motion::vram::cuda_memory_line()?;
        self.vram = format!("{line} · {stage}");
        Some(self.vram.clone())
    }
    pub fn load_video_keep_scene(&mut self, path: PathBuf) {
        let people = std::mem::take(&mut self.people);
        let flow = std::mem::take(&mut self.flow);
        let selected = self.selected;
        self.load_video(path);
        self.people = people;
        self.flow = flow;
        self.selected = selected.min(self.people.len().saturating_sub(1));
    }
        pub fn load_video(&mut self, path: PathBuf) {
                        if let Some(job) = self.capture.take() {
            job.cancel();
        }
        self.capture_started = None;
        self.candidates.clear();
        self.detecting = false;
        self.detect_rx = None;
                        self.stream.restart();
        self.play_from = None;
        self.video = Some(path.clone());
        self.error = None;
        self.export_note = None;
        self.preview = None;
                        self.preview_frame = usize::MAX;
        self.video_frames = 0;
        self.video_fps = 0.0;
        self.people.clear();
        self.mesh = None;
        self.flow.clear();
        self.selected = 0;
        self.floor = 0.0;
        self.framed_after_capture = false;
        self.playhead = 0;
        self.playing = false;
        self.orbit.framed = false;
        self.status = i18n::selected_line(&path.display().to_string());
        self.log(format!("Opened {}", path.display()));
        let slot: Arc<Mutex<Option<Result<capture::Probe, String>>>> = Arc::new(Mutex::new(None));
        self.first_frame = Some(Arc::clone(&slot));
        std::thread::spawn(move || {
            let result = capture::probe_frame(&path, 640).map_err(|error| format!("{error:#}"));
            if let Ok(mut slot) = slot.lock() {
                *slot = Some(result);
            }
        });
        self.start_detection();
    }
    pub fn frame_count(&self) -> usize {
        let people = self
            .people
            .iter()
            .map(|person| person.frames.len())
            .max()
            .unwrap_or(0);
        if people > 0 {
                        people
        } else {
            self.video_frames.max(1)
        }
    }

    /// Save the whole workspace — video, tracked bodies, baked clips, avatars,
    /// locks and settings — to one `.vp` file.
    pub fn save_project(&mut self, path: PathBuf) {
        let path = if path.extension().is_none() {
            path.with_extension(crate::project::EXTENSION)
        } else {
            path
        };
        let started = Instant::now();
        let project = crate::project::Project::capture(self);
        match project.save(&path) {
            Ok(bytes) => {
                let note = i18n::zh_en(
                    &format!(
                        "已保存 {}（{:.1} MB，{:.1} 秒）",
                        path.display(),
                        bytes as f64 / 1.0e6,
                        started.elapsed().as_secs_f32()
                    ),
                    &format!(
                        "Saved {} ({:.1} MB, {:.1} s)",
                        path.display(),
                        bytes as f64 / 1.0e6,
                        started.elapsed().as_secs_f32()
                    ),
                );
                self.log(note.clone());
                self.status = note;
                self.project_path = Some(path);
            }
            Err(error) => {
                let note = format!("{error:#}");
                self.log(i18n::zh_en(
                    &format!("工程保存失败：{note}"),
                    &format!("Could not save the project: {note}"),
                ));
                self.error = Some(note);
            }
        }
    }

    pub fn save_project_dialog(&mut self) {        let mut dialog = rfd::FileDialog::new()
            .add_filter(i18n::t("VeloMocap project"), &[crate::project::EXTENSION])
            .set_title(i18n::t("Save project"));
        let stem = self
            .video
            .as_ref()
            .and_then(|video| video.file_stem())
            .map(|stem| stem.to_string_lossy().into_owned())
            .unwrap_or_else(|| "project".to_string());
        if let Some(folder) = self.video.as_ref().and_then(|video| video.parent()) {
            dialog = dialog.set_directory(folder);
        }
        let Some(path) = dialog
            .set_file_name(format!("{stem}.{}", crate::project::EXTENSION))
            .save_file()
        else {
            return;
        };
        self.save_project(path);
    }

    /// Ctrl+S: write over the file this workspace came from, or ask once.
    pub fn save_project_here(&mut self) {
        match self.project_path.clone() {
            Some(path) => self.save_project(path),
            None => self.save_project_dialog(),
        }
    }

    pub fn open_project_dialog(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter(i18n::t("VeloMocap project"), &[crate::project::EXTENSION])
            .set_title(i18n::t("Open project"))
            .pick_file()
        else {
            return;
        };
        self.open_project(path);
    }

    pub fn open_project(&mut self, path: PathBuf) {
        match crate::project::Project::load(&path) {
            Ok(project) => {
                let (people, clips) = (project.document.people.len(), project.document.flow.len());
                self.restore_project(&project);
                let note = i18n::zh_en(
                    &format!("已打开工程 {}（{people} 人，{clips} 条动作）", path.display()),
                    &format!(
                        "Opened {} ({people} people, {clips} motion clips)",
                        path.display()
                    ),
                );
                self.log(note.clone());
                self.status = note;
                self.project_path = Some(path);
            }
            Err(error) => {
                let note = format!("{error:#}");
                self.log(i18n::zh_en(
                    &format!("工程读取失败：{note}"),
                    &format!("Could not open the project: {note}"),
                ));
                self.error = Some(note);
            }
        }
    }

    /// Put a loaded project back on screen: the same take, the same bodies, the
    /// same locks, without touching the reconstruction.
    pub fn restore_project(&mut self, project: &crate::project::Project) {
        let document = &project.document;
        // Anything live from the previous workspace stops now: a running capture
        // would overwrite the clips this file just brought back.
        if let Some(job) = self.capture.take() {
            job.cancel();
        }
        self.capture_started = None;
        self.detecting = false;
        self.detect_rx = None;
        self.capture_live = false;
        self.stream.restart();
        self.play_from = None;
        self.playing = false;
        self.play_anchor = None;
        self.export_note = None;
        self.error = None;
        self.preview = None;
        self.preview_frame = usize::MAX;
        self.first_frame = None;

        match &document.video {
            Some(video) => {
                self.video = Some(video.path.clone());
                self.video_fps = video.fps;
                self.video_frames = video.frames;
                self.video_size = (video.width, video.height);
                self.capture_fps = video.capture_fps;
                self.floor = video.floor;
                // Only the idle preview needs decoding, and it is one frame: the
                // same thread `load_video` uses, without starting a detection.
                let slot: Arc<Mutex<Option<Result<capture::Probe, String>>>> =
                    Arc::new(Mutex::new(None));
                self.first_frame = Some(Arc::clone(&slot));
                let path = video.path.clone();
                std::thread::spawn(move || {
                    let result =
                        capture::probe_frame(&path, 640).map_err(|error| format!("{error:#}"));
                    if let Ok(mut slot) = slot.lock() {
                        *slot = Some(result);
                    }
                });
            }
            None => {
                self.video = None;
                self.video_fps = 0.0;
                self.video_frames = 0;
                self.video_size = (0, 0);
                self.capture_fps = 0.0;
            }
        }

        let numbers = project.view();
        let mut notes: Vec<String> = Vec::new();
        // The shelf of imported characters comes back first, so a person's
        // assignment can reuse a model that is already loaded and so a model can
        // be switched without finding the file again.
        self.characters.clear();
        self.character_panel = None;
        for path in &document.characters {
            if self.characters.iter().any(|entry| entry.path == *path) {
                continue;
            }
            match crate::character::Character::load(path) {
                Ok(character) => self.characters.push(LibraryCharacter {
                    builtin: is_builtin_path(path),
                    path: path.clone(),
                    character: Arc::new(character),
                }),
                Err(error) => notes.push(format!("{}: {error:#}", path.display())),
            }
        }
        let shelf = self.characters.clone();
        let people: Vec<PersonView> = document
            .people
            .iter()
            .map(|state| {
                let mut person = crate::project::person(state, &numbers);
                if let Some(path) = &state.character {
                    match shelf.iter().find(|entry| entry.path == *path) {
                        Some(entry) => {
                            person.character = Some(AssignedCharacter {
                                path: entry.path.clone(),
                                character: entry.character.clone(),
                            });
                        }
                        None => notes.push(format!("{}: not loaded", path.display())),
                    }
                }
                person
            })
            .collect();
        self.people = people;
        // An avatar is posed against the body model's rest pose; take it once if
        // this workspace has one at all.
        if self.people.iter().any(|person| person.character.is_some())
            && self.character_rest.is_none()
        {
            let (rotations, positions, parents) = self.rest_pose();
            self.character_rest = rotations;
            self.character_rest_positions = positions;
            self.character_rest_parents = parents;
        }
        // The saved assignments are the truth from here on, not whatever the
        // clip's sidecar would have restored.
        self.characters_loaded = true;
        // The shelf of imported characters comes back too, so a model can be
        // switched without finding the file again.
        self.characters.clear();
        self.character_panel = None;
        for path in &document.characters {
            self.add_character(path.clone());
        }
        self.character_panel = None;
        self.flow = document
            .flow
            .iter()
            .map(|state| crate::project::clip(state, &numbers))
            .collect();
        self.candidates = document
            .candidates
            .iter()
            .map(crate::project::candidate)
            .collect();
        self.locks = document.locks.clone();
        self.selected_lock = None;
        self.lock_undo.clear();
        self.lock_redo.clear();
        self.lock_drag = None;
        self.locks_pending = false;
        self.locks_dirty = false;
        self.flow_note = String::new();

        let settings = &document.settings;
        self.correction = settings.correction;
        self.estimate_camera = settings.estimate_camera;
        self.refine_camera = settings.refine_camera;
        self.confidence = settings.confidence;
        self.use_flow = settings.use_flow;
        if let Some(mode) = crate::project::memory_mode_from_slug(&settings.memory_mode) {
            self.memory_mode = mode;
            self.gpu = mode.uses_gpu();
        }
        self.scale_width = settings.scale_width;
        self.max_frames = settings.max_frames;
        self.asset_skin = settings.asset_skin;
        self.asset_materials = settings.asset_materials;
        self.asset_animation = settings.asset_animation;
        self.asset_folder = settings.asset_folder.clone();
        self.storage_precision = settings.storage_precision;
        self.storage_compression = settings.storage_compression;

        let view = &document.view;
        self.selected = view.selected.min(self.people.len().saturating_sub(1));
        self.playhead = view.playhead.min(self.frame_count().saturating_sub(1));
        self.play_speed = view.play_speed;
        self.view_mode = view.mode;
        self.layers = Layers {
            bones: view.layers.bones,
            grid: view.layers.grid,
            motion_path: view.layers.motion_path,
            names: view.layers.names,
            follow: view.layers.follow,
            overlays: view.layers.overlays,
        };
        self.orbit = Orbit {
            yaw: view.orbit.yaw,
            pitch: view.orbit.pitch,
            distance: view.orbit.distance,
            target: view.orbit.target,
            framed: view.orbit.framed,
        };
        self.framed_selected = self.selected;
        self.framed_after_capture = !self.flow.is_empty();

        // Both topologies come with the models on disk, so they are rebuilt
        // rather than saved: the file says which take this is, not which model
        // build it was made with.
        self.mesh = MeshTopology::load(&self.sam3d_dir)
            .ok()
            .map(Arc::new);
        if self.smplh_faces.is_none() {
            self.smplh_faces = self.body_faces();
        }
        self.view = crate::viewport::ViewportCache::default();
        for note in notes {
            self.log(i18n::zh_en(
                &format!("角色没有载入：{note}"),
                &format!("Character not loaded: {note}"),
            ));
        }
    }

    /// The SMPL-H triangle list, for drawing a baked clip's own body.
    fn body_faces(&self) -> Option<Arc<Vec<[u32; 3]>>> {
        let body = velomocap_body::BodyModel::load(&self.smplh_model, 16).ok()?;
        Some(Arc::new(
            body.faces
                .chunks_exact(3)
                .map(|face| [face[0], face[1], face[2]])
                .collect(),
        ))
    }
    pub fn current_fps(&self) -> f64 {
        if self.capture_fps > 0.0 {
            self.capture_fps
        } else if self.video_fps > 0.0 {
            self.video_fps
        } else {
            30.0
        }
    }

    /// The one model a run cannot do without: the SMPL-H body. The download is
    /// not part of the portable package, so a fresh install may well be missing
    /// it, and the status bar says so with the link to get it.
    pub fn key_model_missing(&self) -> bool {
        !self.smplh_model.is_file()
    }
        pub fn trail(&self, person: usize) -> Vec<[f32; 3]> {
        let Some(person) = self.people.get(person) else {
            return Vec::new();
        };
        person
            .frames
            .iter()
            .flatten()
            .filter(|frame| frame.joints.len() >= 3)
            .map(|frame| [frame.joints[0], frame.joints[1], frame.joints[2]])
            .collect()
    }
        fn update_floor(&mut self) {
                                if !self.flow.is_empty() {
            self.floor = 0.0;
            return;
        }
        let mut lowest = f32::INFINITY;
        for person in &self.people {
            for frame in person.frames.iter().flatten() {
                                for index in 13..=20 {
                    let at = index * 3 + 1;
                    if let Some(value) = frame.keypoints.get(at) {
                        if value.is_finite() {
                            lowest = lowest.min(*value);
                        }
                    }
                }
            }
        }
        if lowest.is_finite() {
            self.floor = lowest;
        }
    }
    pub fn start_capture(&mut self) {
        let Some(video) = self.video.clone() else {
            self.error = Some("Open a video first.".into());
            return;
        };
        if !self.detector.is_file() {
            self.error = Some(format!("Detector not found: {}", self.detector.display()));
            return;
        }
        if !self.sam3d_dir.join("sam3d_backbone.onnx").is_file()
            && !self.sam3d_dir.join("sam3d_full_ts.pt").is_file()
            && !Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../../export/sam3d/sam3d_full_ts.pt")
                .is_file()
        {
            self.error = Some(format!(
                "SAM 3D Body graphs not found in {}",
                self.sam3d_dir.display()
            ));
            return;
        }
        if let Some(job) = &self.capture {
            if job.is_running() {
                return;
            }
        }
        self.error = None;
        self.export_note = None;
        self.people.clear();
        self.characters_loaded = false;
        self.person_panel = None;
        self.mesh = None;
        self.selected = 0;
        self.framed_after_capture = false;
        self.capture_started = None;
        self.playhead = 0;
        self.playing = false;
        self.orbit.framed = false;
                let seeds: Vec<[f32; 4]> = self
            .candidates
            .iter()
            .filter(|candidate| candidate.selected)
            .map(|candidate| candidate.bbox)
            .collect();
        let selected = seeds.len();
        if selected > 0 {
            self.status = i18n::zh_en(
                &format!("重建选中 {selected} 人…"),
                &format!("Reconstructing {selected} selected people…"),
            );
            self.log(format!(
                "SAM 3D Body capture started for {selected} selected people."
            ));
        } else {
            self.status = i18n::t("Reconstructing every person…").into();
            self.log("SAM 3D Body capture started (detector → per-person mesh).");
        }
        self.capture = Some(capture::start(capture::CaptureSettings {
            video,
            config: SamConfig {
                detector: self.detector.clone(),
                sam3d_dir: self.sam3d_dir.clone(),
                params_dir: self.params_dir.clone(),
                confidence: self.confidence,
                work_width: self.scale_width,
                max_frames: self.max_frames,
                gpu: self.gpu && self.memory_mode.uses_gpu(),
                seed_boxes: seeds,
                ..Default::default()
            },
            flow: self.use_flow.then(|| capture::FlowSettings {
                weights: self.flow_weights.clone(),
                stats: self.flow_stats.clone(),
                body: self.smplh_model.clone(),
                camera_weights: self.camera_weights.clone(),
                max_frames: self.max_frames,
                gpu: self.gpu && self.memory_mode.uses_gpu(),
                // The hand-drawn contact timing travels with the run; the clip's
                // own detection fills every track the user left alone.
                correction: self.correction.clone(),
                locks: self.locks.clone(),
                estimate_camera: self.estimate_camera,
                refine_camera: self.refine_camera,
            }),
            mode: self.memory_mode,
        }));
        self.log(format!(
            "Memory mode: {} ({})",
            self.memory_mode.label(),
            if self.memory_mode.uses_gpu() { "GPU" } else { "CPU only" }
        ));
    }
    pub fn cancel_capture(&mut self) {
        if let Some(job) = &self.capture {
            job.cancel();
            self.status = i18n::t("Cancelling reconstruction…").into();
            self.log("Cancellation requested.");
        }
    }

    /// Whether "apply correction" has anything it can start from.
    ///
    /// A take loaded from a project has no solver input (`source`), and the run
    /// needs the generated motion, not the baked one — regenerating is the only
    /// way for those.
    pub fn can_apply_correction(&self) -> bool {
        // A *finished* capture is not a running one. This used to say
        // `self.capture.is_none()`, and the finished job is never dropped, so the
        // button stayed disabled for the rest of the session: a take could be
        // generated but never re-solved with other settings.
        let capture_idle = self
            .capture
            .as_ref()
            .map_or(true, |job| !job.is_running());
        capture_idle
            && self.recorrect.is_none()
            && self.flow.iter().any(|take| take.source.is_some())
    }

    /// The reference's "Apply Correction": re-solve the contact correction on the
    /// takes already generated, with the current settings and locks.
    pub fn apply_correction(&mut self) {
        if !self.can_apply_correction() {
            return;
        }
        let takes: Vec<capture::TakeSource> = self
            .flow
            .iter()
            .filter_map(|take| {
                Some(capture::TakeSource {
                    name: take.name.clone(),
                    first_frame: take.first_frame,
                    source: take.source.clone()?,
                })
            })
            .collect();
        if takes.is_empty() {
            return;
        }
        let body = self.smplh_model.clone();
        let settings = self.correction.clone();
        let locks = self.locks.clone();
        let shared = Arc::new(Mutex::new(RecorrectShared::default()));
        let worker = Arc::clone(&shared);
        let started = Instant::now();
        let handle = std::thread::Builder::new()
            .name("velomocap-correction".into())
            .spawn(move || {
                let outcome = capture::recorrect(&takes, &body, &settings, &locks);
                if let Ok(mut state) = worker.lock() {
                    match outcome {
                        Ok(takes) => state.takes = takes,
                        Err(error) => state.error = Some(format!("{error:#}")),
                    }
                    state.elapsed = started.elapsed().as_secs_f64();
                    state.finished = true;
                }
            })
            .ok();
        self.recorrect = Some(RecorrectJob { handle, shared });
        self.recorrect_note = i18n::t("Applying correction…").into();
        self.status = self.recorrect_note.clone();
        self.log(format!(
            "Apply Correction: {} take(s), preset {}, floor {:+.3} m, {} lock run(s).",
            self.flow.len(),
            match self.correction.preset {
                velomocap_motion::correction::LockPreset::Tight => "tight",
                velomocap_motion::correction::LockPreset::Normal => "normal",
                velomocap_motion::correction::LockPreset::Loose => "loose",
            },
            self.correction.floor,
            (0..velomocap_motion::TRACK_COUNT)
                .map(|track| self.locks.track(track).map_or(0, <[_]>::len))
                .sum::<usize>()
        ));
    }

    /// Collect a finished "apply correction" run and put the takes back.
    pub fn poll_recorrect(&mut self, ctx: &egui::Context) {
        let Some(job) = &self.recorrect else { return };
        if job.is_running() {
            // The skinned meshes are rebuilt in the worker; keep painting.
            ctx.request_repaint_after(std::time::Duration::from_millis(120));
            return;
        }
        let (finished, error, takes, elapsed) = {
            let state = job.shared.lock().unwrap();
            (
                state.finished,
                state.error.clone(),
                state.takes.clone(),
                state.elapsed,
            )
        };
        if !finished {
            return;
        }
        self.recorrect = None;
        if let Some(error) = error {
            self.recorrect_note = format!("{}: {error}", i18n::t("Correction failed"));
            self.error = Some(format!("{}: {error}", i18n::t("Apply Correction")));
            self.log(format!("Apply Correction failed: {error}"));
            return;
        }
        // Replace by name: the order the worker returns is the order it was given.
        for rebuilt in takes {
            if let Some(take) = self
                .flow
                .iter_mut()
                .find(|take| take.name == rebuilt.name)
            {
                take.frames = rebuilt.frames;
                take.summary = rebuilt.summary.clone();
                take.bvh = rebuilt.bvh;
                take.skeletons = rebuilt.skeletons;
                take.contacts = rebuilt.contacts;
            }
            self.log(format!(
                "correction applied to {} in {:.1} s: {}",
                rebuilt.name, elapsed, rebuilt.summary
            ));
        }
        self.recorrect_note = format!(
            "{} {:.1} s",
            i18n::t("Correction applied in"),
            elapsed
        );
        self.status = self.recorrect_note.clone();
        self.locks_dirty = false;
    }
    /// Where the hand-drawn contact timing of this clip lives.
    pub fn locks_path(&self) -> Option<PathBuf> {
        self.video
            .as_ref()
            .map(|video| video.with_extension("locks.json"))
    }

    /// Read the saved locks once the probe has told us the frame count.
    ///
    /// A stale document (another clip, a re-cut video) is reported and dropped
    /// instead of being applied to motion it does not describe.
    pub fn load_locks_if_needed(&mut self) {
        if !self.locks_pending || self.video_frames == 0 {
            return;
        }
        self.locks_pending = false;
        let Some(path) = self.locks_path() else {
            return;
        };
        if !path.exists() {
            return;
        }
        let (locks, note) = velomocap_motion::ContactTracks::load(
            &path,
            self.frame_count(),
            self.current_fps(),
        );
        match note {
            None => {
                if !locks.is_automatic() {
                    let total: usize = locks
                        .counts()
                        .iter()
                        .filter_map(|count| *count)
                        .sum();
                    self.log(format!(
                        "Contact locks: {total} locks from {}",
                        path.display()
                    ));
                }
                self.locks = locks;
            }
            Some(note) => {
                self.log(format!("Contact locks ignored: {note}"));
                self.locks = velomocap_motion::ContactTracks::automatic();
            }
        }
        self.lock_undo.clear();
        self.lock_redo.clear();
    }

    /// Write the locks next to the clip; called after every edit.
    pub fn save_locks(&mut self) {
        let Some(path) = self.locks_path() else {
            return;
        };
        let frames = self.frame_count();
        let fps = self.current_fps();
        self.locks = self.locks.normalised(frames);
        match self.locks.save(&path, frames, fps) {
            Ok(()) => self.locks_dirty = true,
            Err(error) => {
                self.error = Some(format!("{error:#}"));
            }
        }
    }

    /// Remember one track as it was before an edit and record the edit.
    pub fn push_lock_edit(&mut self, track: usize, before: Option<Vec<velomocap_motion::Interval>>) {
        let after = self.locks.track(track).map(<[_]>::to_vec);
        if before == after {
            return;
        }
        self.lock_undo.push(LockEdit {
            track,
            before,
            after,
        });
        // The reference keeps 100 steps; a contact edit is cheaper to lose than
        // the whole take.
        if self.lock_undo.len() > 100 {
            self.lock_undo.remove(0);
        }
        self.lock_redo.clear();
        self.save_locks();
    }

    fn apply_lock_edit(&mut self, edit: &LockEdit, forward: bool) {
        let target = if forward { &edit.after } else { &edit.before };
        match target.clone() {
            Some(track) => self.locks.set_track(edit.track, track),
            None => self.locks.reset_track(edit.track),
        }
        self.selected_lock = None;
        self.save_locks();
    }

    pub fn undo_lock(&mut self) {
        if let Some(edit) = self.lock_undo.pop() {
            self.apply_lock_edit(&edit, false);
            self.lock_redo.push(edit);
        }
    }

    pub fn redo_lock(&mut self) {
        if let Some(edit) = self.lock_redo.pop() {
            self.apply_lock_edit(&edit, true);
            self.lock_undo.push(edit);
        }
    }

    /// Delete the selected lock.
    pub fn delete_selected_lock(&mut self) {
        let Some((track, index)) = self.selected_lock else {
            return;
        };
        let Some(track_before) = self.locks.track(track).map(<[_]>::to_vec) else {
            return;
        };
        if index >= track_before.len() {
            return;
        }
        let mut next = track_before.clone();
        next.remove(index);
        self.locks.set_track(track, next);
        self.push_lock_edit(track, Some(track_before));
        self.selected_lock = None;
    }

    /// Put one track, or every track, back on the clip's own detection.
    pub fn reset_locks(&mut self, track: Option<usize>) {
        match track {
            Some(track) => {
                let before = self.locks.track(track).map(<[_]>::to_vec);
                self.locks.reset_track(track);
                self.push_lock_edit(track, before);
            }
            None => {
                for track in 0..velomocap_motion::TRACK_COUNT {
                    let before = self.locks.track(track).map(<[_]>::to_vec);
                    self.locks.reset_track(track);
                    self.push_lock_edit(track, before);
                }
            }
        }
        self.selected_lock = None;
    }

        pub fn poll_capture(&mut self, ctx: &egui::Context) {
        let Some(job) = &self.capture else { return };
        let running = job.is_running();
        let snapshot: Vec<velomocap_motion::PersonTrack>;
        let preview: Option<Arc<capture::PreviewFrame>>;
        let mesh: Option<Arc<MeshTopology>>;
        let flow: Vec<crate::capture::FlowClipData>;
        let smplh_faces: Option<Arc<Vec<[u32; 3]>>>;
        let flow_note: String;
        let fps: f64;
        let video_size: (usize, usize);
        let stage: CaptureStage;
        let stage_index: usize;
        let vram: String;
        let frame: usize;
        let total: usize;
        let people_count: usize;
        let elapsed: f64;
        let backend: String;
        let summary: String;
        let timings: String;
        let failed: Option<String>;
        let finished: bool;
        {
            let state = job.shared.lock().unwrap();
            preview = state.preview.clone();
            mesh = state.mesh.clone();
            flow = state.flow.clone();
            smplh_faces = state.smplh_faces.clone();
            flow_note = state.flow_note.clone();
            snapshot = state.people.clone();
            fps = state.fps;
            video_size = state.video_size;
            stage = state.progress.stage;
            stage_index = state.progress.stage_index;
            vram = state.vram.clone();
            frame = state.progress.frame;
            total = state.progress.total;
            people_count = state.progress.people;
            elapsed = state.progress.elapsed;
            backend = state.backend.clone();
            summary = state.summary.clone();
            timings = state.timings.clone();
            failed = if stage == CaptureStage::Failed {
                Some(state.progress.message.clone())
            } else {
                None
            };
            finished = state.finished;
        }
        // A capture preview is a *live* feed: use it only while the job is
        // running, and let playback win when the transport is moving.
        // `CaptureShared` keeps its last frame after the job ends, and that
        // frame was republished on every paint where its index no longer
        // matched `preview_frame` — which is every paint — so the Movie Clip
        // pane alternated between the frozen reconstruction frame and the
        // playing one. That is the twitching the video showed.
        if running && !self.playing {
            if let Some(preview) = preview {
                if preview.index != self.preview_frame {
                    let image =
                        egui::ColorImage::from_rgb([preview.width, preview.height], &preview.rgb);
                    let same_size = self
                        .preview
                        .as_ref()
                        .map(|texture| texture.size() == [preview.width, preview.height])
                        .unwrap_or(false);
                    match &mut self.preview {
                        Some(texture) if same_size => {
                            texture.set(image, egui::TextureOptions::LINEAR)
                        }
                        _ => {
                            self.preview = Some(ctx.load_texture(
                                "preview",
                                image,
                                egui::TextureOptions::LINEAR,
                            ))
                        }
                    }
                    self.preview_frame = preview.index;
                }
            }
        }
        // When the job stops, hand the pane back to the transport: the frame on
        // screen is the reconstruction's, not the playhead's, so ask the stream
        // for that frame again even if it is the one it was already asked for.
        let was_live = self.capture_live;
        self.capture_live = running;
        if was_live && !running {
            self.stream.invalidate();
        }
        if !flow.is_empty() {
            let fresh = self.flow.len() != flow.len()
                || self
                    .flow
                    .iter()
                    .zip(flow.iter())
                    .any(|(old, new)| !Arc::ptr_eq(&old.frames, &new.frames));
            if fresh {
                for clip in &flow {
                    self.log(format!("FlowHMR {}: {}", clip.name, clip.summary));
                }
                                                self.floor = 0.0;
                self.flow = flow
                    .into_iter()
                    .map(|clip| FlowBody {
                        name: clip.name,
                        first_frame: clip.first_frame,
                        frames: clip.frames,
                        summary: clip.summary,
                        bvh: clip.bvh,
                        skeletons: clip.skeletons,
                        contacts: clip.contacts,
                        source: Some(clip.source),
                    })
                    .collect();
                                                                self.orbit.framed = false;
                // The run that just landed used these locks, so the timeline no
                // longer needs to ask for another one.
                self.locks_dirty = false;
            }
        }
        if let Some(faces) = smplh_faces {
            self.smplh_faces = Some(faces);
        }
        if !flow_note.is_empty() && self.flow_note != flow_note {
            self.log(flow_note.clone());
            self.flow_note = flow_note;
        }
        if let Some(mesh) = mesh {
                                                if std::env::var_os("VELOMOCAP_DEBUG_MESH").is_some() {
                eprintln!(
                    "[mesh] worker sent {} vertices, app had {}",
                    mesh.vertices,
                    self.mesh.as_ref().map_or(0, |m| m.vertices)
                );
            }
            if self.mesh.is_none() {
                self.log(format!(
                    "MHR mesh: {} vertices, {} triangles, {} keypoints",
                    mesh.vertices,
                    mesh.triangles.len(),
                    mesh.keypoints
                ));
                self.mesh = Some(mesh);
            }
        }
        if fps > 0.0 {
            self.capture_fps = fps;
        }
        if video_size != (0, 0) {
            self.video_size = video_size;
        }
                        self.sync_people(&snapshot);
        if finished && self.save_after_capture.is_some() {
            let path = self.save_after_capture.take().unwrap();
            self.save_project(path.clone());
            let saved = self.error.is_none();
            eprintln!(
                "project: {} {}",
                if saved { "saved" } else { "failed" },
                path.display()
            );
            std::process::exit(if saved { 0 } else { 1 });
        }
        if let Some(error) = failed {
                        eprintln!("capture failed: {error}");
            self.error = Some(error);
        }
        if !summary.is_empty() && self.export_note.is_none() {
            self.export_note = Some(format!("{summary}\n{timings}"));
        }
        // The correction report is the only thing that says whether the lock ran,
        // how many contacts it found and whether the pass was rolled back. It used
        // to sit in the export note alone, which is the last place anyone looks
        // when the feet will not stay put.
        if finished && !summary.is_empty() && self.logged_summary != summary {
            self.log(summary.clone());
            self.logged_summary = summary.clone();
        }
        if stage != CaptureStage::Idle && stage != CaptureStage::Failed {
            let _people = if people_count > 0 {
                format!("  ·  {people_count} people")
            } else {
                String::new()
            };
            if self.capture_started.is_none() && frame > 0 {
                self.capture_started = Some(Instant::now());
            }
            let rate = self
                .capture_started
                .map(|started| {
                    let seconds = started.elapsed().as_secs_f64();
                    if seconds > 0.5 {
                        frame as f64 / seconds
                    } else {
                        0.0
                    }
                })
                .unwrap_or(0.0);
            let _ = elapsed;
            self.progress_stage = stage_index;
            self.vram = vram;
            let stage_label = capture::stage_name(stage_index);
            self.progress = (
                frame,
                total,
                i18n::zh_en(
                    &format!("{}/{} {}", stage_index + 1, capture::STAGE_COUNT, i18n::t(stage_label)),
                    &format!(
                        "{}/{} {}", stage_index + 1, capture::STAGE_COUNT,
                        stage_label
                    ),
                ),
            );
            self.status = i18n::run_status(
                i18n::t(stage.label()),
                people_count,
                frame,
                total,
                rate,
                &backend,
            );
        }
        if finished && !self.framed_after_capture {
            self.update_floor();
            if !self.people.is_empty() {
                self.orbit.framed = false;
                self.framed_after_capture = true;
            }
        }
        if running {
            // A run is heavy but it is not on this thread: repaint at the normal
            // rate so the bars move and the window stays usable. It used to ask
            // for a frame every 120 ms, which made the whole UI feel blocked.
            ctx.request_repaint_after(Duration::from_millis(16));
        }
        // Let the job go once it has finished: everything it produced has been
        // folded in above, and holding it kept `can_apply_correction` false.
        if finished && !running {
            self.capture = None;
        }
    }
        fn sync_people(&mut self, snapshot: &[velomocap_motion::PersonTrack]) {
        self.people.retain(|person| {
            snapshot
                .iter()
                .any(|track| track.id == person.id && !track.frames.is_empty())
        });
        for track in snapshot {
            if track.frames.is_empty() {
                continue;
            }
            match self.people.iter_mut().find(|person| person.id == track.id) {
                Some(person) => {
                    person.frames = track.frames.clone();
                    if person.name.is_empty() {
                        person.name = track.name.clone();
                    }
                }
                None => {
                    let index = self.people.len();
                    self.people.push(PersonView {
                        id: track.id,
                        name: track.name.clone(),
                        colour: PERSON_COLOURS[index % PERSON_COLOURS.len()],
                        visible: true,
                        frames: track.frames.clone(),
                        character: None,
                    });
                }
            }
        }
        if self.selected >= self.people.len() {
            self.selected = 0;
        }
        // The avatars assigned to this clip come back as soon as the people
        // exist, before the motion stage has finished.
        self.load_characters();
    }
    /// Show the newest decoded frame in the Movie Clip panel.
    ///
    /// The decoder is a *preview*, not the transport: it never moves the
    /// playhead. It used to — every decoded frame wrote `playhead` — which made
    /// the animation stutter as soon as a frame arrived late or two arrived in
    /// one paint, and made it come apart completely when the clip was shorter
    /// than the video, because the playhead ran past the last baked frame (the
    /// body disappeared) until ffmpeg looped back to zero. `mocap/playback.py`
    /// picks the frame from elapsed media time for the same reason.
    fn poll_stream(&mut self, ctx: &egui::Context) {
        if let Some((generation, frame)) = self.stream.poll() {
            if generation == self.stream.generation() {
                let image =
                    egui::ColorImage::from_rgb([frame.width, frame.height], &frame.rgb);
                // Reuse the texture when the size has not changed: `load_texture`
                // allocates a fresh GPU texture and frees the previous one for
                // every decoded frame, 30 times a second.
                let same_size = self
                    .preview
                    .as_ref()
                    .map(|texture| texture.size() == [frame.width, frame.height])
                    .unwrap_or(false);
                match &mut self.preview {
                    Some(texture) if same_size => {
                        texture.set(image, egui::TextureOptions::LINEAR)
                    }
                    _ => {
                        self.preview = Some(ctx.load_texture(
                            "preview",
                            image,
                            egui::TextureOptions::LINEAR,
                        ))
                    }
                }
                self.preview_frame = frame.index;
                // A picture *ahead* of the clock means the stream has run past
                // the end of the baked clip while the transport wrapped: restart
                // it where the playhead is. Being *behind* is just a slow decode,
                // and restarting there would only restart ffmpeg again a few
                // frames later, so it is left alone to catch up.
                if self.playing && frame.index > self.playhead + 8 {
                    self.play_from = Some(self.playhead);
                }
            }
        }
        let running = self
            .capture
            .as_ref()
            .map(|job| job.is_running())
            .unwrap_or(false);
        if running {
            return;
        }
        let Some(video) = self.video.clone() else { return };
        if self.playing && self.play_from.is_none() {
            self.play_from = Some(self.playhead);
        }
        if !self.playing {
            self.play_from = None;
        }
        // While playing the request must name the frame the run *started* at:
        // naming the moving playhead would restart ffmpeg on every frame.
        let frame = if self.playing {
            self.play_from.unwrap_or(self.playhead)
        } else {
            self.playhead
        };
        // A seek that produced no picture (ffmpeg declined the `-ss`, or the
        // request was coalesced away) left the pane showing the old frame while
        // the playhead had already moved, so the new frame looked ignored. Ask
        // again a few times instead of staying stuck on the old picture.
        if !self.playing && self.preview_frame != self.playhead {
            if self.seek_at.elapsed() > Duration::from_millis(300) && self.seek_tries < 4 {
                self.seek_tries += 1;
                self.seek_at = Instant::now();
                self.stream.invalidate();
            }
        } else {
            self.seek_tries = 0;
        }
        // One seek in flight at a time while paused. Each seek restarts ffmpeg
        // and takes tens of milliseconds; sending one per paint made every
        // decoded frame arrive after a newer request had superseded it, so the
        // picture only moved when the drag ended. Waiting for the frame that is
        // already in flight is what turns a drag into live scrubbing.
        let overdue = self.seek_at.elapsed() > Duration::from_millis(200);
        if !self.playing && self.stream.outstanding() && !overdue {
            ctx.request_repaint_after(Duration::from_millis(30));
            return;
        }
        // A scrub decodes smaller so each seek comes back sooner; once the
        // playhead is sitting on the frame that is on screen, the same frame is
        // asked for again at full size for a crisp still. The width is part of the
        // request identity, so the second ask is not deduplicated away.
        let scrubbing = !self.playing && self.preview_frame != self.playhead;
        self.stream.request(
            &video,
            frame,
            self.playing,
            self.play_speed,
            if scrubbing { 360 } else { 480 },
            self.current_fps(),
        );
        ctx.request_repaint_after(Duration::from_millis(if self.playing { 16 } else { 60 }));
    }

    /// Move the playhead and restart the stream there (timeline clicks, slider).
    pub fn seek_to(&mut self, frame: usize) {
        self.playhead = frame;
        // Always decode the new position, even when it is the frame that is
        // already on screen: a paused seek is the only thing that updates the
        // picture, so deduplicating it leaves the pane showing an old frame.
        self.seek_at = Instant::now();
        self.seek_tries = 0;
        self.stream.invalidate();
        // Re-anchor the transport at the new frame, otherwise the next tick
        // would pull the playhead straight back to where the old anchor
        // predicts.
        self.play_anchor = Some((frame, Instant::now()));
        if self.playing {
            self.play_from = Some(frame);
        }
    }
        pub fn poll(&mut self, ctx: &egui::Context) {
        self.handle_drops(ctx);
        // Framing follows the selection, not the motion: the camera is placed
        // once for the person being looked at and then left alone, so the
        // subject drifting during the clip does not shake the view.
        if self.framed_selected != self.selected {
            self.framed_selected = self.selected;
            self.orbit.framed = false;
        }
                        if self.first_frame.is_some() {
            ctx.request_repaint_after(Duration::from_millis(80));
        }
        if let Some(slot) = &self.first_frame {
            let published = slot.lock().ok().and_then(|mut slot| slot.take());
            if let Some(result) = published {
                self.first_frame = None;
                match result {
                    Ok(probe) => {
                        let frame = probe.frame;
                        self.video_size = (frame.width, frame.height);
                        if probe.fps > 0.0 {
                            self.video_fps = probe.fps;
                        }
                        if probe.frames > 0 {
                            self.video_frames = probe.frames;
                        }
                        let image = egui::ColorImage::from_rgb(
                            [frame.width, frame.height],
                            &frame.rgb,
                        );
                        self.preview = Some(ctx.load_texture(
                            "preview",
                            image,
                            egui::TextureOptions::LINEAR,
                        ));
                        self.preview_frame = frame.index;
                        self.log(format!(
                            "Clip: {} × {}, {:.2} fps, {} frames",
                            self.video_size.0,
                            self.video_size.1,
                            self.video_fps,
                            self.video_frames
                        ));
                    }
                    Err(error) => {
                        self.error = Some(format!("Could not read this clip: {error}"));
                        self.log(format!("Probe failed: {error}"));
                    }
                }
            }
        }
    }
        pub fn shortcuts(&mut self, ctx: &egui::Context) {
        if ctx.wants_keyboard_input() {
            return;
        }
        let (save, open) = ctx.input(|input| {
            (
                input.modifiers.command && input.key_pressed(egui::Key::S),
                input.modifiers.command && input.key_pressed(egui::Key::O),
            )
        });
        if save {
            self.save_project_here();
            return;
        }
        if open {
            self.open_project_dialog();
            return;
        }
        let (space, left, right, home, end, shift) = ctx.input(|input| {
            (
                input.key_pressed(egui::Key::Space),
                input.key_pressed(egui::Key::ArrowLeft),
                input.key_pressed(egui::Key::ArrowRight),
                input.key_pressed(egui::Key::Home),
                input.key_pressed(egui::Key::End),
                input.modifiers.shift,
            )
        });
        let frames = self.frame_count();
        let step = if shift {
            (self.current_fps().round() as usize).max(1)
        } else {
            1
        };
        if space {
            self.playing = !self.playing;
        }
        if left {
            self.seek_to(self.playhead.saturating_sub(step));
            self.playing = false;
        }
        if right {
            self.seek_to((self.playhead + step).min(frames.saturating_sub(1)));
            self.playing = false;
        }
        if home {
            self.seek_to(0);
            self.playing = false;
        }
        if end {
            self.seek_to(frames.saturating_sub(1));
            self.playing = false;
        }
    }
    /// Advance the transport on the wall clock.
    ///
    /// The playhead is chosen from elapsed media time, like `mocap/playback.py`
    /// in the Qt build: `frame = (anchor + elapsed × fps × speed) % frames`.
    /// Deriving it from the decoder instead (which is what `poll_stream` used to
    /// do) tied the animation to ffmpeg's arrival jitter, so a late frame froze
    /// the pose and a burst jumped it forward; when the baked clip was shorter
    /// than the video the playhead even left the clip and the body vanished
    /// until the stream looped. The picture is now only a preview of the clock.
    pub fn tick(&mut self, ctx: &egui::Context) {
        if !self.playing {
            self.play_anchor = None;
            return;
        }
        let frames = self.frame_count().max(1);
        // A speed change re-anchors, so playback continues from where the
        // transport stands instead of jumping to the old anchor's prediction.
        if (self.play_speed - self.last_speed).abs() > 1e-3 {
            self.last_speed = self.play_speed;
            self.play_anchor = Some((self.playhead, Instant::now()));
            if self.video.is_some() {
                self.play_from = Some(self.playhead);
            }
        }
        let (anchor, started) = *self
            .play_anchor
            .get_or_insert_with(|| (self.playhead, Instant::now()));
        let fps = self.current_fps().max(1e-6);
        let speed = self.play_speed.max(0.05) as f64;
        let elapsed = started.elapsed().as_secs_f64();
        self.playhead = playback_frame(anchor, elapsed, fps, speed, frames);
        ctx.request_repaint_after(Duration::from_millis(16));
    }}
/// The frame the transport shows `elapsed` seconds after starting at `anchor`.
///
/// A port of `mocap/playback.py::playback_frame`: the frame comes from elapsed
/// media time, not from which picture the decoder happened to produce last.
fn playback_frame(anchor: usize, elapsed: f64, fps: f64, speed: f64, frames: usize) -> usize {
    let frames = frames.max(1);
    let step = (elapsed.max(0.0) * fps.max(1e-6) * speed.max(0.05)).floor();
    (anchor + step.max(0.0) as usize) % frames
}

fn chrono_like_clock() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let seconds = now.as_secs() % 86_400;
    format!(
        "{:02}:{:02}:{:02}",
        seconds / 3600,
        (seconds % 3600) / 60,
        seconds % 60
    )
}
fn resolve_model(candidates: &[&str]) -> Option<PathBuf> {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()));
    for rel in candidates {
        let direct = PathBuf::from(rel);
        if direct.exists() {
            return Some(direct);
        }
        if let Some(dir) = &exe_dir {
            let beside = dir.join(rel);
            if beside.exists() {
                return Some(beside);
            }
                        if let Some(parent) = dir.parent().and_then(|p| p.parent()) {
                let above = parent.join(rel);
                if above.exists() {
                    return Some(above);
                }
            }
        }
    }
    None}
fn cli_args() -> CliArgs {
    let mut video = None;
    let mut capture_start = false;
    let mut max_frames = 0usize;
    let mut play = false;
    let mut frame = 0usize;
    let mut save_project = None;
    let mut open_project = None;
    let mut storage_precision = None;
    let mut storage_compression = None;
    let mut detect_only = false;
    let mut play_check: Option<usize> = None;
    let mut render_probe: Option<String> = None;
    let mut render_character: Option<(String, String)> = None;
    let mut export_asset: Option<(String, String)> = None;
    let mut export_animated: Option<(String, String, String)> = None;
    let mut render_pose: Option<(String, String, usize, String)> = None;
    let mut select: Option<usize> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--capture" => capture_start = true,
            "--frames" => {
                if let Some(value) = args.next() {
                    max_frames = value.parse().unwrap_or(0);
                }
            }
            "--play" => play = true,
            "--frame" => {
                if let Some(value) = args.next() {
                    frame = value.parse().unwrap_or(0);
                }
            }
            "--cpu" => std::env::set_var("VELOMOCAP_FORCE_CPU", "1"),
            "--detect-only" => detect_only = true,
            "--render-probe" => {
                render_probe = args.next();
            }
            "--export-animated" => {
                let file = args.next();
                let project = args.next();
                let out = args.next();
                if let (Some(file), Some(project), Some(out)) = (file, project, out) {
                    export_animated = Some((file, project, out));
                }
            }
            "--export-asset" => {
                if let (Some(file), Some(out)) = (args.next(), args.next()) {
                    export_asset = Some((file, out));
                }
            }
            "--render-character" => {
                if let (Some(file), Some(out)) = (args.next(), args.next()) {
                    render_character = Some((file, out));
                }
            }
            "--render-pose" => {
                let file = args.next();
                let project = args.next();
                let frame = args.next().and_then(|value| value.parse().ok());
                let out = args.next();
                if let (Some(file), Some(project), Some(frame), Some(out)) =
                    (file, project, frame, out)
                {
                    render_pose = Some((file, project, frame, out));
                }
            }
            "--play-check" => {
                if let Some(value) = args.next() {
                    play_check = value.parse().ok();
                }
            }
            "--select" => {
                if let Some(value) = args.next() {
                    select = value.parse().ok();
                }
            }
            // Headless project round-tripping: save after a capture, or open a
            // saved workspace and leave without touching the GUI.
            "--save-project" => {
                if let Some(value) = args.next() {
                    save_project = Some(PathBuf::from(value));
                }
            }
            "--open-project" => {
                if let Some(value) = args.next() {
                    open_project = Some(PathBuf::from(value));
                }
            }
            "--precision" => {
                if let Some(value) = args.next() {
                    storage_precision = match value.as_str() {
                        "ultra" => Some(crate::project::Precision::Ultra),
                        "high" => Some(crate::project::Precision::High),
                        "medium" => Some(crate::project::Precision::Medium),
                        "low" => Some(crate::project::Precision::Low),
                        _ => None,
                    };
                }
            }
            "--compression" => {
                if let Some(value) = args.next() {
                    storage_compression = match value.as_str() {
                        "none" => Some(crate::project::Compression::None),
                        "fast" => Some(crate::project::Compression::Fast),
                        "balanced" => Some(crate::project::Compression::Balanced),
                        "smallest" => Some(crate::project::Compression::Smallest),
                        _ => None,
                    };
                }
            }
            other if !other.starts_with("--") => video = Some(PathBuf::from(other)),
            _ => {}
        }
    }
    CliArgs {
        video,
        capture_start,
        max_frames,
        play,
        frame,
        detect_only,
        select,
        play_check,
        render_probe,
        render_character,
        export_asset,
        export_animated,
        render_pose,
        save_project,
        open_project,
        storage_precision,
        storage_compression,
    }}
struct CliArgs {
    video: Option<PathBuf>,
    capture_start: bool,
    max_frames: usize,
    play: bool,
    frame: usize,
    detect_only: bool,
    select: Option<usize>,
    play_check: Option<usize>,
    render_probe: Option<String>,
    render_character: Option<(String, String)>,
    export_asset: Option<(String, String)>,
    export_animated: Option<(String, String, String)>,
    render_pose: Option<(String, String, usize, String)>,
    save_project: Option<PathBuf>,
    open_project: Option<PathBuf>,
    storage_precision: Option<crate::project::Precision>,
    storage_compression: Option<crate::project::Compression>,
}
impl eframe::App for VeloMocap {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.load_library();
        self.load_builtin_characters();
        self.poll(ctx);
        self.poll_detection(ctx);
        self.poll_capture(ctx);
        self.poll_recorrect(ctx);
        self.poll_stream(ctx);
        self.poll_asset_export(ctx);
        self.shortcuts(ctx);
        self.tick(ctx);
        ui::draw(self, ctx);
    }}
#[cfg(test)]
mod tests {
    use super::playback_frame;

    /// One frame per 1/fps second, starting from the anchor.
    #[test]
    fn the_clock_counts_frames_from_elapsed_time() {
        assert_eq!(playback_frame(0, 0.0, 30.0, 1.0, 90), 0);
        assert_eq!(playback_frame(0, 1.0, 30.0, 1.0, 90), 30);
        assert_eq!(playback_frame(12, 1.0, 30.0, 1.0, 90), 42);
        // Half a frame still shows the frame we are on, not the next one.
        assert_eq!(playback_frame(0, 0.016, 30.0, 1.0, 90), 0);
        assert_eq!(playback_frame(0, 0.034, 30.0, 1.0, 90), 1);
        // Speed scales the same elapsed time.
        assert_eq!(playback_frame(0, 1.0, 30.0, 2.0, 90), 60);
        assert_eq!(playback_frame(0, 1.0, 30.0, 0.5, 90), 15);
    }

    /// Playback wraps at the end and never leaves the clip, which is what keeps
    /// a body on screen when the video is longer than the baked motion.
    #[test]
    fn the_clock_wraps_inside_the_clip() {
        assert_eq!(playback_frame(0, 4.0, 30.0, 1.0, 90), 30);
        assert_eq!(playback_frame(80, 1.0, 30.0, 1.0, 90), 20);
        for step in 0..400 {
            let frame = playback_frame(0, step as f64 * 0.033, 30.0, 1.0, 90);
            assert!(frame < 90, "frame {frame} left the clip");
        }
    }

    /// Degenerate inputs must not panic or divide by zero: no clip, no fps.
    #[test]
    fn the_clock_survives_empty_inputs() {
        assert_eq!(playback_frame(0, 1.0, 0.0, 0.0, 0), 0);
        assert_eq!(playback_frame(5, 0.0, 30.0, 1.0, 1), 0);
        assert_eq!(playback_frame(5, f64::NAN, 30.0, 1.0, 10), 5);
    }
}

