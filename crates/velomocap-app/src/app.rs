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
        pub frames: Vec<Option<Arc<PersonFrame>>>,}
#[derive(Clone)]pub struct FlowBody {
    pub name: String,
        pub first_frame: usize,
        pub frames: Arc<Vec<Vec<f32>>>,
        pub summary: String,
        pub bvh: Arc<velomocap_motion::PreparedBvh>,}
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
        self.frames.get(index).map(|verts| verts.as_slice())
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
pub struct VeloMocap {
    pub video: Option<PathBuf>,
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
        pub export_on_finish: bool,
        pub export_fbx_on_finish: bool,
    pub exported: bool,
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
    /// Which vendor's bone names an export is written in.
    pub export_rig: velomocap_motion::retarget::Rig,
    /// Re-express the exported rotations in the canonical rest frames, so a rig
    /// that binds in the canonical T-pose plays this clip exactly.
    pub export_rest_correct: bool,
    pub console: Vec<String>,
    pub show_console: bool,
    pub show_help: bool,
    pub show_models: bool,
    pub show_guide: bool,
    pub tools_ok: bool,
    pub ort_path: Option<PathBuf>,
    pub cuda_ok: bool,
    pub model_status: Vec<velomocap_models::GroupStatus>,
    pub model_root: Option<PathBuf>,
    pub model_core_ready: bool,
    pub model_error: Option<String>,}
impl VeloMocap {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        ui::configure(&cc.egui_ctx);
        let CliArgs {
            video: initial_video,
            capture_start,
            max_frames,
            play,
            frame,
            export,
            export_fbx,
            detect_only,
            select,
            play_check,
            render_probe,
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
            estimate_camera: true,
            refine_camera: true,
            export_on_finish: export,
            export_fbx_on_finish: export_fbx,
            exported: false,
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
            export_rig: velomocap_motion::retarget::Rig::Biped,
            export_rest_correct: false,
            console: Vec::new(),
            show_console: false,
            show_help: false,
            show_models: false,
            show_guide: false,
            model_status: Vec::new(),
            model_root: None,
            model_core_ready: false,
            model_error: None,
        };
        if let Some(out) = render_probe.clone() {
            let model = app.smplh_model.display().to_string();
            if let Err(error) = crate::probe::render_rest_pose(&model, &out) {
                eprintln!("render-probe failed: {error:#}");
            }
            std::process::exit(0);
        }
        crate::models_ui::refresh(&mut app, false);
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
        if let Some(path) = app.video.clone() {
            app.load_video(path);
                        app.playing = play;
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
    pub fn open_video_dialog(&mut self) {
        let Some(path) = rfd::FileDialog::new()
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
        self.exported = false;
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
    pub fn current_fps(&self) -> f64 {
        if self.capture_fps > 0.0 {
            self.capture_fps
        } else if self.video_fps > 0.0 {
            self.video_fps
        } else {
            30.0
        }
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
                correction: self.correction,
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
    /// The clip's export payload: renamed into the selected rig, and
    /// rest-corrected when asked. The default biped payload is the clip's own,
    /// which is what every export wrote before the retarget chain existed.
    ///
    /// `rest_correct` is a property of the *format*: a BVH carries no bind pose,
    /// so the correction is what makes a canonical-rest rig play it, while an
    /// FBX states its own bind (ours, from the SMPL-H rest) and correcting its
    /// channels against someone else's rest would contradict that bind.
    fn export_payload(
        &self,
        bvh: &velomocap_motion::PreparedBvh,
        rest_correct: bool,
    ) -> Result<velomocap_motion::PreparedBvh, String> {
        let mut prepared = bvh.clone();
        if self.export_rig != velomocap_motion::retarget::Rig::Biped || rest_correct {
            velomocap_motion::retarget::rewrite(&mut prepared, self.export_rig, rest_correct)
                .map_err(|error| format!("{error:#}"))?;
        }
        Ok(prepared)
    }
                    pub fn export_fbx(&mut self) {
        let Some(video) = self.video.clone() else {
            self.error = Some("Open a video first.".into());
            return;
        };
        let Some(clip) = self
            .flow
            .get(self.selected)
            .or_else(|| self.flow.first())
            .cloned()
        else {
            self.error = Some("Generate motion (FlowHMR) before exporting FBX.".into());
            return;
        };
        let path = video.with_extension("fbx");
        if self.export_rest_correct {
            self.log(
                "FBX: the file carries its own bind pose, so the rest correction is applied to BVH exports only.",
            );
        }
        let prepared = match self.export_payload(&clip.bvh, false) {
            Ok(prepared) => prepared,
            Err(error) => {
                self.error = Some(error);
                return;
            }
        };
        match velomocap_motion::write_prepared_fbx(&path, &prepared) {
            Ok(report) => {
                let note = format!(
                    "{} · FlowHMR · {} bones · {} keys · {}",
                    path.display(),
                    report.bones,
                    report.key_count,
                    self.export_rig.label()
                );
                self.log(format!("FBX: {note}"));
                self.export_note = Some(note);
                self.status = format!("Wrote {}", path.display());
            }
            Err(error) => self.error = Some(format!("{error:#}")),
        }
    }
        pub fn export_bvh(&mut self) {
        let Some(video) = self.video.clone() else {
            self.error = Some("Open a video first.".into());
            return;
        };
        if let Some(clip) = self
            .flow
            .get(self.selected)
            .or_else(|| self.flow.first())
            .cloned()
        {
            let path = video.with_extension("bvh");
            let prepared = match self.export_payload(&clip.bvh, self.export_rest_correct) {
                Ok(prepared) => prepared,
                Err(error) => {
                    self.error = Some(error);
                    return;
                }
            };
            match velomocap_motion::write_prepared(&path, &prepared) {
                Ok(report) => {
                    let first = report.orders.first().cloned().unwrap_or_default();
                    let order = if report.orders.iter().all(|entry| *entry == first) {
                        format!("uniform {first}")
                    } else {
                        first
                    };
                    let note = format!(
                        "{} · FlowHMR · {} frames · {} bones · order {} · {}",
                        path.display(),
                        clip.bvh.frames(),
                        clip.bvh.parents.len(),
                        order,
                        self.export_rig.label()
                    );
                    self.log(format!("BVH: {note}"));
                    self.export_note = Some(note);
                    self.status = format!("Wrote {}", path.display());
                }
                Err(error) => self.error = Some(format!("{error:#}")),
            }
            return;
        }
        let Some(mesh) = self.mesh.clone() else {
            self.error = Some("Reconstruct a clip before exporting.".into());
            return;
        };
        let Some(person) = self
            .people
            .get(self.selected)
            .or_else(|| self.people.first())
            .cloned()
        else {
            self.error = Some("No people to export.".into());
            return;
        };
        let track = velomocap_motion::PersonTrack {
            id: person.id,
            name: person.name.clone(),
            frames: person.frames.clone(),
            first_frame: 0,
            last_frame: person.frames.len().saturating_sub(1),
            hits: person.hits(),
        };
        let path = video.with_extension("bvh");
        // The MHR skeleton has no canonical bone names (its joints are
        // `mhr000…`), so the skeleton-naming choice cannot apply to it: the
        // per-frame MHR export is only reachable when there is no FlowHMR clip,
        // and then it is the only payload available.
        if self.export_rig != velomocap_motion::retarget::Rig::Biped {
            self.log(format!(
                "BVH: the MHR skeleton has no {} names, writing its own joints.",
                self.export_rig.label()
            ));
        }
        match velomocap_motion::write_person_bvh(&path, &track, &mesh, self.current_fps()) {
            Ok(report) => {
                let note = format!(
                    "{} · {} bones · order {}",
                    path.display(),
                    mesh.parents.len(),
                    report.orders.first().cloned().unwrap_or_default()
                );
                self.log(format!("BVH: {note}"));
                self.export_note = Some(note);
                self.status = format!("Wrote {}", path.display());
            }
            Err(error) => self.error = Some(format!("{error:#}")),
        }
    }
    pub fn refresh_models(&mut self, verify_hash: bool) {
        crate::models_ui::refresh(self, verify_hash);
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
                    })
                    .collect();
                                                                self.orbit.framed = false;
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
        if finished && self.export_on_finish && !self.exported {
            self.exported = true;
            self.export_bvh();
            if self.export_fbx_on_finish {
                self.export_fbx();
            }
            match &self.export_note {
                Some(note) => eprintln!("export: {note}"),
                None => eprintln!(
                    "export: failed — {}",
                    self.error.clone().unwrap_or_else(|| "unknown".into())
                ),
            }
        }
        if let Some(error) = failed {
                        eprintln!("capture failed: {error}");
            self.error = Some(error);
        }
        if !summary.is_empty() && self.export_note.is_none() {
            self.export_note = Some(format!("{summary}\n{timings}"));
        }
        if stage != CaptureStage::Idle && stage != CaptureStage::Failed {
            let people = if people_count > 0 {
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
                    });
                }
            }
        }
        if self.selected >= self.people.len() {
            self.selected = 0;
        }
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
        self.stream.request(
            &video,
            frame,
            self.playing,
            self.play_speed,
            480,
            self.current_fps(),
        );
        ctx.request_repaint_after(Duration::from_millis(if self.playing { 16 } else { 60 }));
    }

    /// Move the playhead and restart the stream there (timeline clicks, slider).
    pub fn seek_to(&mut self, frame: usize) {
        self.playhead = frame;
        // Re-anchor the transport at the new frame, otherwise the next tick
        // would pull the playhead straight back to where the old anchor
        // predicts.
        self.play_anchor = Some((frame, Instant::now()));
        if self.playing {
            self.play_from = Some(frame);
        }
    }
        pub fn poll(&mut self, ctx: &egui::Context) {
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
    let mut export = false;
    let mut export_fbx = false;
    let mut detect_only = false;
    let mut play_check: Option<usize> = None;
    let mut render_probe: Option<String> = None;
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
            "--export-bvh" => export = true,
            "--detect-only" => detect_only = true,
            "--render-probe" => {
                render_probe = args.next();
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
            "--export-fbx" => export_fbx = true,
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
        export,
        export_fbx,
        detect_only,
        select,
        play_check,
        render_probe,
    }}
struct CliArgs {
    video: Option<PathBuf>,
    capture_start: bool,
    max_frames: usize,
    play: bool,
    frame: usize,
    export: bool,
    export_fbx: bool,
    detect_only: bool,
    select: Option<usize>,
    play_check: Option<usize>,
    render_probe: Option<String>,
}
impl eframe::App for VeloMocap {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll(ctx);
        self.poll_detection(ctx);
        self.poll_capture(ctx);
        self.poll_stream(ctx);
        self.shortcuts(ctx);
        self.tick(ctx);
        ui::draw(self, ctx);
        crate::models_ui::window(self, ctx);
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
