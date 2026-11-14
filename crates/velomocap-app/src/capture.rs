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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptureStage {
    Idle,
    Loading,
    Decoding,
    Detecting,
    Reconstructing,
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
            CaptureStage::Done => "done",
            CaptureStage::Failed => "failed",
        }
    }
}

#[derive(Clone, Debug)]
pub struct CaptureProgress {
    pub stage: CaptureStage,
    pub frame: usize,
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

    let mut progress = {
        let shared = Arc::clone(&shared);
        let started = started;
        move |frame: usize, total: usize, stage: &str| {
            let mut state = shared.lock().unwrap();
            state.progress.stage = match stage {
                "loading mesh" | "loading detector" | "loading SAM 3D Body" | "opening video" => {
                    CaptureStage::Loading
                }
                "detecting" => CaptureStage::Detecting,
                "reconstructing" => CaptureStage::Reconstructing,
                "done" => CaptureStage::Done,
                _ => state.progress.stage,
            };
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

    // FlowHMR: the coherent SMPL-H motion behind the preview and the exports.
    // A failure here must not lose the reconstruction, so it is reported and the
    // per-frame MHR bodies stay on screen.
    if let Some(flow) = settings.flow.clone() {
        if let Err(error) = run_flow(&video, &capture.people, &flow, &shared) {
            let mut state = shared.lock().unwrap();
            state.progress.message = format!("FlowHMR: {error:#}");
            state.flow_note = format!("FlowHMR failed: {error}");
        }
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
    state.progress.elapsed = started.elapsed().as_secs_f64();
    state.finished = true;
    Ok(())
}

/// Generate one FlowHMR clip per person and skin it for the viewport.
fn run_flow(
    video: &std::path::Path,
    people: &[PersonTrack],
    settings: &FlowSettings,
    shared: &Arc<Mutex<CaptureShared>>,
) -> Result<()> {
    use velomocap_infer::{MotionConfig, SessionConfig, VggtCamera, VggtConfig};

    let report = |message: &str| {
        if let Ok(mut state) = shared.lock() {
            state.progress.message = message.to_string();
        }
    };

    report("flow camera");
    let tools = velomocap_video::Tools::discover()?;
    let mut vggt = VggtCamera::load(VggtConfig {
        weights: settings.camera_weights.clone(),
        session: SessionConfig {
            gpu: settings.gpu,
            ..Default::default()
        },
        ..Default::default()
    })?;
    let camera = vggt.estimate(&tools, video)?;

    report("flow body model");
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
                gpu: settings.gpu,
                ..Default::default()
            },
            ..Default::default()
        },
        max_frames: settings.max_frames,
        ..Default::default()
    };

    let mut clips = Vec::new();
    for (slot, person) in people.iter().enumerate() {
        report(&format!("flow {}/{}", slot + 1, people.len()));
        let sentinel = || {
            if let Ok(state) = shared.lock() {
                if state.progress.stage == CaptureStage::Failed {
                    return true;
                }
            }
            false
        };
        let clip = match velomocap_motion::generate_flow(person, &camera, &model, &config) {
            Ok(clip) => clip,
            Err(error) => {
                if let Ok(mut state) = shared.lock() {
                    state.flow_note = format!("FlowHMR {}: {error}", person.name);
                }
                continue;
            }
        };
        let mut bodies = Vec::with_capacity(clip.frames());
        for pose in &clip.poses {
            match model.forward_mesh(pose) {
                Ok((_, vertices)) => bodies.push(vertices),
                Err(_) => bodies.push(Vec::new()),
            }
        }
        let summary = velomocap_motion::summarise_flow(&clip).join("; ");
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
