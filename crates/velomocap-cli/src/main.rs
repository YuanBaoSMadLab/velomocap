//! `velomocap` command line: self-check, kernel benchmark and conversions.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde::Serialize;

use velomocap_core::bvh;
use velomocap_core::contact::{self, Preset};
use velomocap_core::rig::Binding;

#[derive(Parser)]
#[command(
    name = "velomocap",
    version,
    about = "VeloMocap native motion core: exports, contact planning and benchmarks"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Report the native core and kernel build identity.
    Info,
    /// Run the built-in kernel self-test.
    Selftest,
    /// Report the ONNX Runtime and execution-provider situation.
    Ort,
    /// Full native capture: video -> keypoints -> SMPL-X -> BVH.
    Capture {
        /// Video file.
        video: PathBuf,
        /// Output BVH (defaults to <video stem>.bvh).
        #[arg(long)]
        out: Option<PathBuf>,
        /// SMPL-X model.
        #[arg(long, default_value = "models/body_models/smplx/neutral/model.npz")]
        model: PathBuf,
        /// YOLO-pose ONNX weights.
        #[arg(long, default_value = "models/detector/yolov8x-pose.onnx")]
        weights: PathBuf,
        /// Confidence threshold.
        #[arg(long, default_value_t = 0.25)]
        confidence: f32,
        /// Work resolution.
        #[arg(long, default_value_t = 960)]
        width: usize,
        /// Use the CPU execution provider only.
        #[arg(long)]
        cpu: bool,
        /// Stop after this many frames (0 = whole video).
        #[arg(long, default_value_t = 0)]
        limit: usize,
    },
    /// Fit SMPL-X to an image's keypoints and report the reprojection error.
    Fit {
        /// Input image.
        image: PathBuf,
        /// SMPL-X model.
        #[arg(long, default_value = "models/body_models/smplx/neutral/model.npz")]
        model: PathBuf,
        /// YOLO-pose ONNX weights.
        #[arg(long, default_value = "models/detector/yolov8x-pose.onnx")]
        weights: PathBuf,
        /// Confidence threshold.
        #[arg(long, default_value_t = 0.25)]
        confidence: f32,
        /// Use the CPU execution provider only.
        #[arg(long)]
        cpu: bool,
    },
    /// Detect people and 2D body keypoints in an image.
    Pose {
        /// Input image.
        image: PathBuf,
        /// YOLO-pose ONNX weights.
        #[arg(long, default_value = "models/detector/yolov8x-pose.onnx")]
        weights: PathBuf,
        /// Confidence threshold.
        #[arg(long, default_value_t = 0.25)]
        confidence: f32,
        /// Use the CPU execution provider only.
        #[arg(long)]
        cpu: bool,
    },
    /// Load the SMPL-X model and report its structure and rest skeleton.
    Body {
        /// Path to the SMPL-X .npz.
        #[arg(long, default_value = "models/body_models/smplx/neutral/model.npz")]
        model: PathBuf,
        /// Number of shape coefficients to keep.
        #[arg(long, default_value_t = 10)]
        betas: usize,
        /// Also load the pose blend shapes and time a skinned mesh.
        #[arg(long)]
        mesh: bool,
    },
    /// Run detection + tracking over a whole video (headless pipeline check).
    Track {
        /// Video file.
        video: PathBuf,
        /// YOLO ONNX weights.
        #[arg(long, default_value = "models/detector/yolov8x.onnx")]
        weights: PathBuf,
        /// Confidence threshold.
        #[arg(long, default_value_t = 0.25)]
        confidence: f32,
        /// Work resolution; frames are scaled before detection.
        #[arg(long, default_value_t = 960)]
        width: usize,
        /// Use the CPU execution provider only.
        #[arg(long)]
        cpu: bool,
        /// Stop after this many frames (0 = whole video).
        #[arg(long, default_value_t = 0)]
        limit: usize,
    },
    /// Reconstruct every person with SAM 3D Body (ONNX) and report their meshes.
    People {
        /// Video file.
        video: PathBuf,
        /// Person detector.
        #[arg(long, default_value = "models/detector/yolov8x.onnx")]
        detector: PathBuf,
        /// Directory with the SAM 3D Body graphs and `sam3d_mesh.json`.
        #[arg(long, default_value = "models/sam3d")]
        sam3d: PathBuf,
        /// Directory with `sam3d_params.{json,bin}`.
        #[arg(long, default_value = "models/flowhmr/sam")]
        params: PathBuf,
        /// Write one PLY per person (first frame with a mesh) into this folder.
        #[arg(long)]
        ply: Option<PathBuf>,
        /// Write the first person's motion as BVH.
        #[arg(long)]
        bvh: Option<PathBuf>,
        /// Stop after this many frames (0 = whole video).
        #[arg(long, default_value_t = 0)]
        limit: usize,
        /// Work resolution.
        #[arg(long, default_value_t = 960)]
        width: usize,
        /// Detection confidence.
        #[arg(long, default_value_t = 0.3)]
        confidence: f32,
        /// Use the CPU execution provider only.
        #[arg(long)]
        cpu: bool,
    },
    /// Generate one person's coherent motion with FlowHMR (ONNX) and export BVH.
    Flow {
        /// Video file.
        video: PathBuf,
        /// Person detector.
        #[arg(long, default_value = "models/detector/yolov8x.onnx")]
        detector: PathBuf,
        /// Directory with the SAM 3D Body graphs.
        #[arg(long, default_value = "models/sam3d")]
        sam3d: PathBuf,
        /// Directory with `sam3d_params.{json,bin}`.
        #[arg(long, default_value = "models/flowhmr/sam")]
        params: PathBuf,
        /// FlowHMR MMDiT graph.
        #[arg(long, default_value = "models/flowhmr/flowhmr_mmdit.onnx")]
        weights: PathBuf,
        /// FlowHMR motion statistics.
        #[arg(long, default_value = "models/flowhmr/motion_stats.json")]
        stats: PathBuf,
        /// SMPL-H body model.
        #[arg(long, default_value = "models/body_models/smplh/neutral/model.npz")]
        body: PathBuf,
        /// VGGT-Omega graph, used when `--camera` is not given.
        #[arg(long, default_value = "models/flowhmr/vggt_omega_1b_512.onnx")]
        camera_weights: PathBuf,
        /// Reuse a camera track written by `camera --json`.
        #[arg(long)]
        camera: Option<PathBuf>,
        /// Which tracked person to animate (1 = first).
        #[arg(long, default_value_t = 1)]
        person: usize,
        /// Write the generated motion as BVH.
        #[arg(long)]
        bvh: Option<PathBuf>,
        /// Write the generated motion as binary FBX 7.4 as well.
        #[arg(long)]
        fbx: Option<PathBuf>,
        /// Write the first generated frame's skinned vertices (f32 LE, 6890x3).
        #[arg(long)]
        dump_mesh: Option<PathBuf>,
        /// Write every frame's joint world positions (f32 LE, frames x 52 x 3).
        #[arg(long)]
        dump_joints: Option<PathBuf>,
        /// Write every frame's joint world rotations (f32 LE, frames x 52 x 9).
        #[arg(long)]
        dump_rotations: Option<PathBuf>,
        #[arg(long)]
        dump_probes: Option<PathBuf>,
        /// Disable foot/hand contact correction.
        #[arg(long)]
        no_correction: bool,
        /// Leave feet unlocked (hands still locked).
        #[arg(long)]
        no_foot: bool,
        /// Pin resting hands as well (off by default; the reference defaults to
        /// foot locking only).
        #[arg(long)]
        hand_lock: bool,
        /// Skip the anatomical contact IK on the legs (it is on by default, like
        /// the reference's DoContactIK).
        #[arg(long)]
        no_leg_ik: bool,
        /// Use the skinned foot-mesh sole points as the contact source.
        #[arg(long)]
        joint_sole: bool,
        /// Temporal smoothing weight (0 = off, smaller is stronger).
        #[arg(long, default_value_t = 0.0)]
        smooth: f64,
        /// Lock preset: tight, normal or loose.
        #[arg(long, default_value = "normal")]
        lock_preset: String,
        /// Correction mode: flat or multi.
        #[arg(long, default_value = "flat")]
        lock_mode: String,
        /// Ground height for the contact lock, metres.
        #[arg(long, default_value_t = 0.0)]
        floor: f32,
        /// Treat the camera as fixed instead of estimating it with VGGT.
        #[arg(long)]
        static_camera: bool,
        /// Smooth the estimated camera rotation over time.
        #[arg(long)]
        refine_camera: bool,
        /// Vertical shift applied to the clip, metres.
        #[arg(long, default_value_t = 0.0)]
        pelvis: f32,
        /// Stop after this many frames (0 = whole video).
        #[arg(long, default_value_t = 0)]
        limit: usize,
        /// First video frame of the generated clip.
        #[arg(long, default_value_t = 0)]
        start: usize,
        /// Frames in the generated clip (0 = to the end of the decode).
        #[arg(long, default_value_t = 0)]
        frames: usize,
        /// Noise seed; the reference uses 0.
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Work resolution for detection and SAM.
        #[arg(long, default_value_t = 960)]
        width: usize,
        /// Detection confidence.
        #[arg(long, default_value_t = 0.3)]
        confidence: f32,
        /// Euler steps for the flow-matching loop.
        #[arg(long, default_value_t = 20)]
        steps: usize,
        /// Run VGGT on every N-th frame.
        #[arg(long, default_value_t = 10)]
        camera_stride: usize,
        /// Diagnostic: zero the SAM features, to prove the graph uses them.
        #[arg(long)]
        blank: bool,
        /// Diagnostic: start the ODE from zeros instead of noise.
        #[arg(long)]
        zero_noise: bool,
        /// Diagnostic: write the graph inputs and the final latent here.
        #[arg(long)]
        dump: Option<PathBuf>,
        /// Use the CPU execution provider only.
        #[arg(long)]
        cpu: bool,
    },
    /// Run the SAM 3D Body TorchScript module through LibTorch.
    SamTorch {
        /// The TorchScript module that `tools/torch/sam3d_jit.py` writes.
        #[arg(long, default_value = "models/sam3d/sam3d_body_ts.pt")]
        weights: PathBuf,
        /// Keep the module on the CPU.
        #[arg(long)]
        cpu: bool,
    },
    /// Run the whole-pipeline SAM module (body + hands + fusion) through LibTorch.
    ///
    /// The module is `tools/torch/sam3d_full_jit.py`, i.e. the reference
    /// `run_inference(inference_type="full")` with the crops inside the graph.
    SamFullTorch {
        /// The TorchScript module that `tools/torch/sam3d_full_jit.py` writes.
        #[arg(long, default_value = "models/sam3d/sam3d_full_ts.pt")]
        weights: PathBuf,
        /// Keep the module on the CPU.
        #[arg(long)]
        cpu: bool,
        /// Frame size the module was traced with.
        #[arg(long, default_value_t = 1280)]
        width: usize,
        #[arg(long, default_value_t = 720)]
        height: usize,
        /// Optional raw case folder from `sam3d_full_check.py --dump`.
        #[arg(long)]
        case: Option<PathBuf>,
    },
    /// Run the person detector on one image.
    Detect {
        /// YOLO ONNX weights.
        #[arg(long, default_value = "models/detector/yolov8x.onnx")]
        weights: PathBuf,
        /// Input image (PNG/JPEG).
        image: PathBuf,
        /// Confidence threshold.
        #[arg(long, default_value_t = 0.25)]
        confidence: f32,
        /// Use the CPU execution provider only.
        #[arg(long)]
        cpu: bool,
    },
    /// Time detector load and steady-state inference separately.
    BenchDetect {
        /// YOLO ONNX weights.
        #[arg(long, default_value = "models/detector/yolov8x.onnx")]
        weights: PathBuf,
        /// Input image (PNG/JPEG).
        image: PathBuf,
        /// Number of timed inferences after one warm-up.
        #[arg(long, default_value_t = 20)]
        repeat: usize,
        /// Use the CPU execution provider only.
        #[arg(long)]
        cpu: bool,
    },
    /// Track a video and write JSON + CSV next to it.
    ExportTracks {
        /// Video file.
        video: PathBuf,
    },
    /// Estimate per-frame cameras with VGGT-Omega (ONNX, no Python).
    Camera {
        /// Video file.
        video: PathBuf,
        /// VGGT-Omega ONNX graph.
        #[arg(long, default_value = "models/flowhmr/vggt_omega_1b_512.onnx")]
        weights: PathBuf,
        /// Run the network on every N-th frame (camera interpolated between).
        #[arg(long, default_value_t = 10)]
        stride: usize,
        /// Cap on the number of network frames per clip.
        #[arg(long, default_value_t = 8)]
        max_samples: usize,
        /// Use the CPU execution provider only.
        #[arg(long)]
        cpu: bool,
        /// Stop after this many frames (0 = whole video).
        #[arg(long, default_value_t = 0)]
        limit: usize,
        /// Write the camera track as JSON.
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// Compare the exported VGGT-Omega graph against saved PyTorch tensors.
    VggtParity {
        /// Directory holding images.bin / rope_sin.bin / rope_cos.bin /
        /// pose_enc.bin and dims.json.
        dir: PathBuf,
        /// VGGT-Omega ONNX graph.
        #[arg(long, default_value = "models/flowhmr/vggt_omega_1b_512.onnx")]
        weights: PathBuf,
        /// Use the CPU execution provider only.
        #[arg(long)]
        cpu: bool,
    },
    /// Compare the exported FlowHMR MMDiT against saved PyTorch tensors.
    FlowhmrParity {
        /// Directory holding x.bin / cam.bin / feature.bin / mask.bin /
        /// pred.bin and fh_dims.json.
        dir: PathBuf,
        /// Exported MMDiT graph.
        #[arg(long, default_value = "models/flowhmr/flowhmr_mmdit.onnx")]
        weights: PathBuf,
        /// Use the CPU execution provider only.
        #[arg(long)]
        cpu: bool,
    },
    /// Benchmark the native stages on synthetic motion.
    Bench {
        /// Number of frames per take.
        #[arg(long, default_value_t = 1200)]
        frames: usize,
        /// Repeat each stage this many times and keep the best result.
        #[arg(long, default_value_t = 3)]
        repeat: usize,
        /// Write the measurements as JSON.
        #[arg(long)]
        json: Option<PathBuf>,
    },
}

#[derive(Serialize)]
struct BenchRow {
    stage: String,
    frames: usize,
    seconds: f64,
    detail: String,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Info => {
            println!("velomocap {}", velomocap_core::VERSION);
            println!("kernels: {}", velomocap_core::backend());
            println!("rayon threads: {}", rayon::current_num_threads());
            Ok(())
        }
        Command::Selftest => selftest(),
        Command::Ort => ort_status(),
        Command::Body { model, betas, mesh } => body_report(&model, betas, mesh),
        Command::Capture {
            video,
            out,
            model,
            weights,
            confidence,
            width,
            cpu,
            limit,
        } => capture_video(&video, out, &model, &weights, confidence, width, !cpu, limit),
        Command::Fit {
            image,
            model,
            weights,
            confidence,
            cpu,
        } => fit_report(&image, &model, &weights, confidence, !cpu),
        Command::Pose {
            image,
            weights,
            confidence,
            cpu,
        } => pose_report(&image, &weights, confidence, !cpu),
        Command::ExportTracks { video } => export_video_tracks(&video),
        Command::People {
            video,
            detector,
            sam3d,
            params,
            ply,
            bvh,
            limit,
            width,
            confidence,
            cpu,
        } => people_report(
            &video,
            &detector,
            &sam3d,
            &params,
            ply.as_deref(),
            bvh.as_deref(),
            limit,
            width,
            confidence,
            !cpu,
        ),
        Command::Camera {
            video,
            weights,
            stride,
            max_samples,
            cpu,
            limit,
            json,
        } => camera_track(&video, &weights, stride, max_samples, !cpu, limit, json),
        Command::Flow {
            video,
            detector,
            sam3d,
            params,
            weights,
            stats,
            body,
            camera_weights,
            camera,
            person,
            bvh,
            fbx,
            dump_mesh,
            dump_joints,
            dump_rotations,
            dump_probes,
            no_correction,
            no_foot,
            hand_lock,
            no_leg_ik,
            joint_sole,
            smooth,
            lock_preset,
            lock_mode,
            floor,
            static_camera,
            refine_camera,
            pelvis,
            limit,
            start,
            frames,
            seed,
            width,
            confidence,
            steps,
            camera_stride,
            blank,
            zero_noise,
            dump,
            cpu,
        } => flow_report(
            &video,
            &detector,
            &sam3d,
            &params,
            &weights,
            &stats,
            &body,
            &camera_weights,
            camera.as_deref(),
            person,
            bvh.as_deref(),
            fbx.as_deref(),
            dump_mesh.as_deref(),
            dump_joints.as_deref(),
            dump_rotations.as_deref(),
            dump_probes.as_deref(),
            no_correction,
            no_foot,
            hand_lock,
            no_leg_ik,
            joint_sole,
            smooth,
            &lock_preset,
            &lock_mode,
            floor,
            static_camera,
            refine_camera,
            pelvis,
            limit,
            start,
            frames,
            seed,
            width,
            confidence,
            steps,
            camera_stride,
            blank,
            zero_noise,
            dump.as_deref(),
            !cpu,
        ),
        Command::SamTorch { weights, cpu } => sam_torch_report(&weights, !cpu),
        Command::SamFullTorch {
            weights,
            cpu,
            width,
            height,
            case,
        } => sam_full_torch_report(&weights, !cpu, width, height, case.as_deref()),
        Command::VggtParity { dir, weights, cpu } => vggt_parity(&dir, &weights, !cpu),
        Command::FlowhmrParity { dir, weights, cpu } => flowhmr_parity(&dir, &weights, !cpu),
        Command::Track {
            video,
            weights,
            confidence,
            width,
            cpu,
            limit,
        } => track_video(&video, &weights, confidence, width, !cpu, limit),
        Command::BenchDetect {
            weights,
            image,
            repeat,
            cpu,
        } => bench_detect(&weights, &image, repeat, !cpu),
        Command::Detect {
            weights,
            image,
            confidence,
            cpu,
        } => detect(&weights, &image, confidence, !cpu),
        Command::Bench {
            frames,
            repeat,
            json,
        } => bench(frames, repeat, json),
    }
}

fn selftest() -> Result<()> {
    let rest: Vec<f64> = (0..22).flat_map(|j| [0.0, -0.1 * j as f64, 0.0]).collect();
    let frames = 4;
    let roots = vec![0.0; frames * 3];
    let mut quats = vec![0.0; frames * 22 * 4];
    for t in 0..frames {
        for j in 0..22 {
            quats[(t * 22 + j) * 4] = 1.0;
        }
    }
    let (positions, rotations) = contact::forward_kinematics(&rest, &roots, &quats, frames);
    for value in positions.iter().chain(rotations.iter()) {
        anyhow::ensure!(
            value.is_finite(),
            "forward kinematics produced a non-finite value"
        );
    }
    let probes = vec![0.0; 64 * 4 * 3];
    let probabilities = vec![0.5; 64 * 4];
    let preset = Preset::named("normal")?;
    contact::prepare_contacts(&probes, &probabilities, 64, 30.0, None, &preset, None)?;
    println!("selftest ok (kernels: {})", velomocap_core::backend());
    Ok(())
}

fn ort_status() -> Result<()> {
    match velomocap_infer::find_runtime_library() {
        Some(library) => println!("onnxruntime library: {}", library.display()),
        None => {
            println!("onnxruntime library: NOT FOUND (set VELOMOCAP_ORT_DYLIB)");
            return Ok(());
        }
    }
    println!("cuda provider     : {}", velomocap_infer::cuda_provider_available());
    Ok(())
}

/// Minimal image decode through the `image` crate re-exported by the infer crate.
fn decode_rgb(bytes: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
    let decoded = velomocap_infer::detector::decode_image(bytes)?;
    anyhow::ensure!(
        decoded.0 == width && decoded.1 == height,
        "image dimensions changed between reads"
    );
    Ok(decoded.2)
}

#[allow(clippy::too_many_arguments)]
/// Resolve a model path against the working directory, then the executable's
/// folder and its ancestors.
///
/// A shipped build keeps `models/` beside the `.exe`, and the CLI is often run
/// from another directory, so a bare relative path is not enough.
fn resolve_model(path: &std::path::Path) -> PathBuf {
    if path.is_file() {
        return path.to_path_buf();
    }
    if path.is_absolute() {
        return path.to_path_buf();
    }
    if let Ok(exe) = std::env::current_exe() {
        let mut dir = exe.parent();
        for _ in 0..3 {
            if let Some(folder) = dir {
                let candidate = folder.join(path);
                if candidate.is_file() {
                    return candidate;
                }
                dir = folder.parent();
            }
        }
    }
    path.to_path_buf()
}

/// Multi-person SAM 3D Body: one MHR mesh per detected person per frame.
#[allow(clippy::too_many_arguments)]
fn people_report(
    video: &std::path::Path,
    detector: &std::path::Path,
    sam3d: &std::path::Path,
    params: &std::path::Path,
    ply: Option<&std::path::Path>,
    bvh: Option<&std::path::Path>,
    limit: usize,
    width: usize,
    confidence: f32,
    gpu: bool,
) -> Result<()> {
    use velomocap_motion::{capture_people, PersonTrack, SamConfig};

    let sam3d = if sam3d.is_dir() {
        sam3d.to_path_buf()
    } else {
        resolve_model(sam3d)
    };
    let config = SamConfig {
        detector: resolve_model(detector),
        sam3d_dir: sam3d,
        params_dir: resolve_model(params),
        confidence,
        work_width: width,
        max_frames: limit,
        gpu,
        ..Default::default()
    };

    let mut last_frame = 0usize;
    let mut progress = |frame: usize, total: usize, stage: &str| {
        if frame != last_frame || stage == "done" {
            eprint!("\r{stage:<16} {frame}/{total}      ");
            last_frame = frame;
        }
    };
    let mut preview = |_: &velomocap_video::Frame| {};
    let mut people = |_: usize, _: &[PersonTrack]| {};
    let cancelled = || false;
    let mut hooks = velomocap_motion::SamHooks {
        progress: &mut progress,
        preview: &mut preview,
        people: &mut people,
        cancelled: &cancelled,
    };
    let started = Instant::now();
    let capture = velomocap_motion::capture_people_with(video, &config, &mut hooks)?;
    eprintln!();

    println!("backend: {}", capture.backend);
    println!(
        "{} people · {} frames @ {:.2} fps",
        capture.people.len(),
        capture.frames,
        capture.fps
    );
    println!("{}", capture.summary());
    println!(
        "cleanup: {} low-foot frames of {}, {} pinned contacts",
        capture.cleanup.low_frames, capture.cleanup.frames, capture.cleanup.contacts
    );
    if !capture.cleanup.line().is_empty() {
        println!("{}", capture.cleanup.line());
    }
    println!("{}", capture.timings.line());
    println!(
        "{}",
        capture.people.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join(", ")
    );

    for person in &capture.people {
        let Some((index, frame)) = person
            .frames
            .iter()
            .enumerate()
            .find_map(|(index, slot)| slot.as_ref().map(|frame| (index, frame)))
        else {
            continue;
        };
        let mut low = f32::INFINITY;
        let mut high = f32::NEG_INFINITY;
        let mut width_x = 0.0f32;
        for point in frame.verts.chunks_exact(3) {
            low = low.min(point[1]);
            high = high.max(point[1]);
            width_x = width_x.max(point[0] - frame.joints[0]);
        }
        // Left/right ankle extent over the clip: the compact "does the foot
        // slide" number, independent of the cleanup internals.
        let mut lo = [[f32::INFINITY; 3]; 2];
        let mut hi = [[f32::NEG_INFINITY; 3]; 2];
        for frame in person.frames.iter().flatten() {
            for side in 0..2 {
                let at = (13 + side) * 3;
                for axis in 0..3 {
                    let value = frame.keypoints[at + axis];
                    lo[side][axis] = lo[side][axis].min(value);
                    hi[side][axis] = hi[side][axis].max(value);
                }
            }
        }
        let spans = [0, 1].map(|side| {
            let dx = hi[side][0] - lo[side][0];
            let dz = hi[side][2] - lo[side][2];
            if dx.is_finite() && dz.is_finite() {
                (dx * dx + dz * dz).sqrt()
            } else {
                0.0
            }
        });
        println!(
            "  {} id={} frames {}..{} ({} with mesh) depth {:.2} m · height {:.2} m · bbox {:.0}x{:.0} px · score {:.2} · ankle travel {:.2}/{:.2} m",
            person.name,
            person.id,
            person.first_frame,
            person.last_frame,
            person.hits,
            frame.cam_t[2],
            high - low,
            frame.bbox[2] - frame.bbox[0],
            frame.bbox[3] - frame.bbox[1],
            frame.score,
            spans[0],
            spans[1],
        );
        let _ = index;
    }

    if let Some(path) = bvh {
        if let Some(person) = capture.people.first() {
            let report = velomocap_motion::write_person_bvh(path, person, &capture.mesh, capture.fps)?;
            println!(
                "wrote {} ({} bones, order {})",
                path.display(),
                capture.mesh.parents.len(),
                report.orders.first().cloned().unwrap_or_default()
            );
        }
    }

    if let Some(dir) = ply {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("creating {}", dir.display()))?;
        for person in &capture.people {
            let Some(frame) = person.frames.iter().flatten().next() else {
                continue;
            };
            let stem = person.name.replace(' ', "_");
            let path = dir.join(format!("{stem}.ply"));
            write_ply(&path, &frame.verts, &capture.mesh.triangles)?;
            println!("wrote {}", path.display());
            let mut lines = String::new();
            for (index, point) in frame.keypoints.chunks_exact(3).enumerate() {
                let name = capture
                    .mesh
                    .names
                    .get(index)
                    .map(String::as_str)
                    .unwrap_or("?");
                lines.push_str(&format!(
                    "{index:2} {name:<22} {:8.3} {:8.3} {:8.3}\n",
                    point[0], point[1], point[2]
                ));
            }
            std::fs::write(dir.join(format!("{stem}.keypoints.txt")), lines)?;
        }
    }
    eprintln!("total {:.1}s", started.elapsed().as_secs_f64());
    Ok(())
}

/// Drive the TorchScript SAM module from Rust.
///
/// The inputs come from the same fixed recipe as `tools/torch/ts_probe.py` and
/// `cpp/src/ts_probe.cpp`, so the printed values can be compared one to one:
/// the three paths must agree.
fn sam_torch_report(weights: &std::path::Path, gpu: bool) -> Result<()> {
    use velomocap_cpp::sam3d::{Sam3dInput, Sam3dTorch, IMAGE_SIZE};

    if !Sam3dTorch::built() {
        anyhow::bail!("this build has no LibTorch; set VELOMOCAP_LIBTORCH=1 and rebuild");
    }
    let weights = resolve_model(weights);
    let load = Instant::now();
    let module = Sam3dTorch::load(&weights, gpu)?;
    println!(
        "loaded {} in {:.1}s ({})",
        weights.display(),
        load.elapsed().as_secs_f64(),
        if gpu { "cuda if available" } else { "cpu" }
    );

    // `(i % 251) / 251`, the recipe the other two probes use.
    let pixels = IMAGE_SIZE * IMAGE_SIZE;
    let mut image = vec![0.0f32; 3 * pixels];
    for (index, value) in image.iter_mut().enumerate() {
        *value = (index % 251) as f32 / 251.0;
    }
    let input = Sam3dInput {
        image: &image,
        bbox_center: [640.0, 400.0],
        bbox_scale: [250.0, 375.0],
        cam_int: [1100.0, 0.0, 640.0, 0.0, 1100.0, 360.0, 0.0, 0.0, 1.0],
        keypoints: [0.0, 0.0, -2.0],
        ori_size: [1280.0, 720.0],
    };
    let run = Instant::now();
    let frame = module.run(&input)?;
    println!("forward in {:.2}s", run.elapsed().as_secs_f64());

    let first = |values: &[f32]| -> String {
        values
            .iter()
            .take(4)
            .map(|v| format!("{v:.6}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let sum = |values: &[f32]| -> f64 { values.iter().map(|v| *v as f64).sum() };
    println!(
        "body_token   first4 [{}] sum {:.4}",
        first(&frame.body_token),
        sum(&frame.body_token)
    );
    println!(
        "verts        first4 [{}] sum {:.4}",
        first(&frame.verts),
        sum(&frame.verts)
    );
    println!(
        "pred         first4 [{}] sum {:.4}",
        first(&frame.pred),
        sum(&frame.pred)
    );
    println!(
        "pred_cam     first3 [{}] sum {:.4}",
        frame
            .pred_cam
            .iter()
            .take(3)
            .map(|v| format!("{v:.6}"))
            .collect::<Vec<_>>()
            .join(", "),
        sum(&frame.pred_cam)
    );
    println!(
        "model_params first4 [{}] sum {:.4}",
        first(&frame.model_params),
        sum(&frame.model_params)
    );
    Ok(())
}

/// Run the whole-pipeline SAM module (body, both hands, fusion) from Rust.
///
/// Without `--case` the frame is the same synthetic ramp the other probes use;
/// with `--case DIR` it reads the binary case folder that
/// `tools/torch/sam3d_full_check.py --dump` writes, so the numbers can be
/// compared against Python's.
fn sam_full_torch_report(
    weights: &std::path::Path,
    gpu: bool,
    width: usize,
    height: usize,
    case: Option<&std::path::Path>,
) -> Result<()> {
    use velomocap_cpp::sam3d::{Sam3dFullInput, Sam3dTorch};

    if !Sam3dTorch::built() {
        anyhow::bail!("this build has no LibTorch; set VELOMOCAP_LIBTORCH=1 and rebuild");
    }
    let weights = resolve_model(weights);
    let load = Instant::now();
    let module = Sam3dTorch::load(&weights, gpu)?;
    println!(
        "loaded {} in {:.1}s ({})",
        weights.display(),
        load.elapsed().as_secs_f64(),
        if gpu { "cuda if available" } else { "cpu" }
    );

    let (frame, width, height, center, scale, cam_int) = match case {
        Some(dir) => read_case(dir)?,
        None => {
            let mut frame = vec![0.0f32; 3 * width * height];
            for (index, value) in frame.iter_mut().enumerate() {
                *value = (index % 251) as f32 / 251.0;
            }
            (
                frame,
                width,
                height,
                [640.0f32, 395.0],
                [762.5f32, 762.5],
                [1400.0f32, 0.0, 640.0, 0.0, 1400.0, 360.0, 0.0, 0.0, 1.0],
            )
        }
    };

    let input = Sam3dFullInput {
        frame: &frame,
        height,
        width,
        bbox_center: center,
        bbox_scale: scale,
        cam_int,
    };
    let run = Instant::now();
    let out = module.run_full(&input)?;
    println!(
        "forward in {:.2}s ({width}x{height})",
        run.elapsed().as_secs_f64()
    );
    // The first call pays for cuDNN's kernel selection; report the steady state
    // too, so the number is comparable with the Python side.
    let warm = Instant::now();
    let _ = module.run_full(&input)?;
    println!("second forward in {:.2}s", warm.elapsed().as_secs_f64());

    let first = |values: &[f32]| -> String {
        values
            .iter()
            .take(4)
            .map(|v| format!("{v:.6}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let sum = |values: &[f32]| -> f64 { values.iter().map(|v| *v as f64).sum() };
    println!("verts        first4 [{}] sum {:.4}", first(&out.verts), sum(&out.verts));
    println!(
        "kps3d        first4 [{}] sum {:.4}",
        first(&out.keypoints_3d),
        sum(&out.keypoints_3d)
    );
    println!(
        "kps2d        first4 [{}] sum {:.4}",
        first(&out.keypoints_2d),
        sum(&out.keypoints_2d)
    );
    println!(
        "pred_cam     [{:.6}, {:.6}, {:.6}]",
        out.pred_cam[0], out.pred_cam[1], out.pred_cam[2]
    );
    println!("pred_cam_t   [{:.6}, {:.6}, {:.6}]", out.pred_cam_t[0], out.pred_cam_t[1], out.pred_cam_t[2]);
    println!("focal_length {:.3}", out.focal_length);
    println!(
        "model_params first4 [{}] sum {:.4}",
        first(&out.model_params),
        sum(&out.model_params)
    );
    println!(
        "joint_rots   first4 [{}] sum {:.4}",
        first(&out.joint_rots),
        sum(&out.joint_rots)
    );
    println!(
        "hand_valid   [{:.1}, {:.1}]",
        out.hand_valid[0], out.hand_valid[1]
    );
    println!(
        "body_token   first4 [{}] sum {:.4}",
        first(&out.body_token),
        sum(&out.body_token)
    );
    println!(
        "left_token   first4 [{}] sum {:.4}",
        first(&out.left_token),
        sum(&out.left_token)
    );
    println!(
        "right_token  first4 [{}] sum {:.4}",
        first(&out.right_token),
        sum(&out.right_token)
    );
    println!("features     {} values (3072 expected)", out.features().len());
    println!(
        "skel_state   first4 [{}] sum {:.4}",
        first(&out.skel_state),
        sum(&out.skel_state)
    );
    println!("pred         first4 [{}]", first(&out.pred));
    Ok(())
}

/// Read the binary case folder `sam3d_full_check.py --dump` writes.
///
/// Layout, little endian: `i32 width`, `i32 height`, `f32 centre[2]`,
/// `f32 scale[2]`, `f32 cam_int[9]`, `f32 frame[3*h*w]` (NCHW, RGB, 0..1).
fn read_case(
    dir: &std::path::Path,
) -> Result<(Vec<f32>, usize, usize, [f32; 2], [f32; 2], [f32; 9])> {
    let path = dir.join("case.bin");
    let raw = std::fs::read(&path)
        .map_err(|error| anyhow::anyhow!("reading {}: {error}", path.display()))?;
    if raw.len() < 8 {
        anyhow::bail!("{} is too short", path.display());
    }
    let width = i32::from_le_bytes(raw[0..4].try_into().unwrap()) as usize;
    let height = i32::from_le_bytes(raw[4..8].try_into().unwrap()) as usize;
    let mut at = 8usize;
    let mut take = |count: usize| -> Result<Vec<f32>> {
        let bytes = count * 4;
        if at + bytes > raw.len() {
            anyhow::bail!("{} ends early", path.display());
        }
        let mut values = Vec::with_capacity(count);
        for index in 0..count {
            let start = at + index * 4;
            let mut word = [0u8; 4];
            word.copy_from_slice(&raw[start..start + 4]);
            values.push(f32::from_le_bytes(word));
        }
        at += bytes;
        Ok(values)
    };
    let center = take(2)?;
    let scale = take(2)?;
    let cam = take(9)?;
    let frame = take(3 * width * height)?;
    let mut cam_int = [0.0f32; 9];
    cam_int.copy_from_slice(&cam);
    Ok((
        frame,
        width,
        height,
        [center[0], center[1]],
        [scale[0], scale[1]],
        cam_int,
    ))
}

/// Reconstruct the clip with SAM, run FlowHMR on one person, export BVH.
///
/// This is the native form of the reference's `flow_camera` → `flow_features`
/// → `flowhmr` → `flow_export` chain: VGGT gives the camera rotations, SAM 3D
/// Body gives the `3072`-wide per-frame features, the exported MMDiT integrates
/// from noise, and the result is placed on the SAM hips.
#[allow(clippy::too_many_arguments)]
fn flow_report(
    video: &std::path::Path,
    detector: &std::path::Path,
    sam3d: &std::path::Path,
    params: &std::path::Path,
    weights: &std::path::Path,
    stats: &std::path::Path,
    body: &std::path::Path,
    camera_weights: &std::path::Path,
    camera: Option<&std::path::Path>,
    person: usize,
    bvh: Option<&std::path::Path>,
    fbx: Option<&std::path::Path>,
    dump_mesh: Option<&std::path::Path>,
    dump_joints: Option<&std::path::Path>,
    dump_rotations: Option<&std::path::Path>,
    dump_probes: Option<&std::path::Path>,
    no_correction: bool,
    no_foot: bool,
    hand_lock: bool,
    no_leg_ik: bool,
    joint_sole: bool,
    smoothing: f64,
    lock_preset: &str,
    lock_mode: &str,
    floor: f32,
    static_camera: bool,
    refine_camera: bool,
    pelvis: f32,
    limit: usize,
    start: usize,
    frames: usize,
    seed: u64,
    width: usize,
    confidence: f32,
    steps: usize,
    camera_stride: usize,
    blank: bool,
    zero_noise: bool,
    dump: Option<&std::path::Path>,
    gpu: bool,
) -> Result<()> {
    use velomocap_body::BodyModel;
    use velomocap_infer::{MotionConfig, SessionConfig, VggtCamera, VggtConfig};
    use velomocap_motion::{capture_people_with, FlowConfig, PersonTrack, SamConfig, SamHooks};

    let sam3d = if sam3d.is_dir() {
        sam3d.to_path_buf()
    } else {
        resolve_model(sam3d)
    };
    let sam_config = SamConfig {
        detector: resolve_model(detector),
        sam3d_dir: sam3d,
        params_dir: resolve_model(params),
        confidence,
        work_width: width,
        max_frames: limit,
        gpu,
        ..Default::default()
    };

    let mut last_frame = 0usize;
    let mut progress = |frame: usize, total: usize, stage: &str| {
        if frame != last_frame || stage == "done" {
            eprint!("\r{stage:<16} {frame}/{total}      ");
            last_frame = frame;
        }
    };
    let mut preview = |_: &velomocap_video::Frame| {};
    let mut people = |_: usize, _: &[PersonTrack]| {};
    let cancelled = || false;
    let mut hooks = SamHooks {
        progress: &mut progress,
        preview: &mut preview,
        people: &mut people,
        cancelled: &cancelled,
    };
    let capture = capture_people_with(video, &sam_config, &mut hooks)?;
    eprintln!();
    println!("sam: {} people · {} frames", capture.people.len(), capture.frames);
    let slot = person.saturating_sub(1);
    let track = capture
        .people
        .get(slot)
        .with_context(|| format!("person {person} does not exist"))?;
    println!(
        "person {}: frames {}..{}, {} reconstructions",
        track.name, track.first_frame, track.last_frame, track.hits
    );
    if let Some(frame) = track.frames.iter().flatten().next() {
        let count = frame.body_token.len().max(1) as f64;
        let mean = frame.body_token.iter().map(|v| *v as f64).sum::<f64>() / count;
        let variance = frame
            .body_token
            .iter()
            .map(|v| (*v as f64 - mean).powi(2))
            .sum::<f64>()
            / count;
        let low = frame
            .body_token
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min);
        let high = frame
            .body_token
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        println!(
            "  body token: {} values, mean {:.3}, sd {:.3}, range [{:.2}, {:.2}] first 4 {:?}",
            frame.body_token.len(),
            mean,
            variance.sqrt(),
            low,
            high,
            &frame.body_token[..4]
        );
    }

    let camera_track = match camera {
        Some(path) => {
            let track = load_camera_track(path, capture.frames)?;
            println!(
                "camera: {} frames from {} ({}x{})",
                track.len(),
                path.display(),
                track.image_size.1,
                track.image_size.0
            );
            track
        }
        None => {
            let tools = velomocap_video::Tools::discover()?;
            let mut vggt = VggtCamera::load(VggtConfig {
                weights: resolve_model(camera_weights),
                frame_stride: camera_stride.max(1),
                session: SessionConfig {
                    gpu,
                    ..Default::default()
                },
                ..Default::default()
            })?;
            let started = Instant::now();
            let track = vggt.estimate(&tools, video)?;
            println!(
                "camera: {} frames, {} sampled, vggt {:.1}s",
                track.len(),
                track.sample_indices.len(),
                started.elapsed().as_secs_f64()
            );
            track
        }
    };
    // Camera switches, matching the sidebar: a fixed camera reuses the first
    // frame's extrinsics, and the refine switch smooths an estimated track.
    let mut camera_track = camera_track;
    if static_camera {
        camera_track.freeze();
        println!("camera: frozen to the first frame");
    } else if refine_camera {
        camera_track.smooth(1);
        println!("camera: rotation smoothed");
    }

    let body_path = resolve_model(body);
    let members = velomocap_body::npz::members(&body_path);
    let has_posedirs = match &members {
        Ok(names) => names
            .iter()
            .any(|name| name == "posedirs" || name.starts_with("posedirs.")),
        Err(error) => {
            println!("  cannot list {}: {error}", body_path.display());
            false
        }
    };
    let mut model = BodyModel::load(&body_path, 16)
        .with_context(|| format!("loading SMPL-H from {}", body.display()))?;
    // `posedirs` is what lets the floor be measured on the mesh, exactly as the
    // reference's origin placement does; the joints are the fallback.
    if has_posedirs {
        model = model.with_posedirs()?;
    } else {
        println!("  no posedirs in the body model; the floor uses joints");
    }
    println!(
        "smplh: {} vertices, {} joints, {} betas",
        model.num_vertices, model.num_joints, model.num_betas
    );

    let config = FlowConfig {
        motion: MotionConfig {
            weights: resolve_model(weights),
            stats: resolve_model(stats),
            steps: steps.max(1),
            seed,
            session: SessionConfig {
                gpu,
                ..Default::default()
            },
            ..Default::default()
        },
        max_frames: frames,
        start,
        blank_tokens: blank,
        zero_noise,
        dump: dump.map(|p| p.to_path_buf()),
    };
    let started = Instant::now();
    let mut clip = velomocap_motion::generate_flow(track, &camera_track, &model, &config)?;
    println!(
        "flowhmr: {:.1}s, {} steps, backend {}",
        started.elapsed().as_secs_f64(),
        steps,
        if gpu { "cuda" } else { "cpu" }
    );
    for line in velomocap_motion::summarise_flow(&clip) {
        println!("  {line}");
    }
    let roots = clip.root_track();
    if let (Some(first), Some(last)) = (roots.first(), roots.last()) {
        println!(
            "  root ({:+.3}, {:+.3}, {:+.3}) -> ({:+.3}, {:+.3}, {:+.3})",
            first[0], first[1], first[2], last[0], last[1], last[2]
        );
    }

    // Contact correction, the same pass the app runs after generating.
    let settings = velomocap_motion::correction::CorrectionSettings {
        enabled: !no_correction,
        mode: match lock_mode {
            "multi" | "multi-level" => velomocap_motion::correction::CorrectionMode::MultiLevel,
            _ => velomocap_motion::correction::CorrectionMode::Flat,
        },
        foot: !no_foot,
        hand: hand_lock,
        preset: match lock_preset {
            "tight" => velomocap_motion::correction::LockPreset::Tight,
            "loose" => velomocap_motion::correction::LockPreset::Loose,
            _ => velomocap_motion::correction::LockPreset::Normal,
        },
        floor,
        pelvis_offset: pelvis,
        leg_ik: !no_leg_ik,
        mesh_sole: !joint_sole,
        smoothing,
    };
    let correction = velomocap_motion::correction::apply(&mut clip, &model, &settings)?;
    println!("  {}", correction.line());

    if let Some(path) = dump_probes {
        let rows = velomocap_motion::correction::dump_probe_points(&clip, &model);
        let mut bytes = Vec::new();
        for row in &rows {
            for value in row {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        std::fs::write(path, &bytes)?;
        println!("wrote {} ({} frames x 24 probe values)", path.display(), rows.len());
    }

    if let Some(path) = dump_rotations {
        let mut bytes = Vec::new();
        for skeleton in &clip.skeletons {
            for value in &skeleton.rotations {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        std::fs::write(path, bytes)?;
        println!(
            "wrote {} ({} frames x {} rotation matrices)",
            path.display(),
            clip.skeletons.len(),
            clip.skeletons.first().map(|s| s.rotations.len() / 9).unwrap_or(0)
        );
    }
    if let Some(path) = dump_joints {
        let mut bytes = Vec::new();
        for skeleton in &clip.skeletons {
            for value in &skeleton.positions {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        std::fs::write(path, bytes)?;
        println!(
            "wrote {} ({} frames x {} joints)",
            path.display(),
            clip.skeletons.len(),
            clip.skeletons.first().map(|s| s.positions.len() / 3).unwrap_or(0)
        );
    }
    if let Some(path) = dump_mesh {
        if let Some(pose) = clip.poses.first() {
            let (_, verts) = model.forward_mesh(pose)?;
            let bytes: Vec<u8> = verts.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(path, &bytes)?;
            println!(
                "wrote {} ({} vertices, frame 0)",
                path.display(),
                verts.len() / 3
            );
            // The same shape with every joint at rest: if this one is smooth, the
            // needles come from the rotations (and their pose blends), not from
            // the body shape.
            let rest = velomocap_body::Pose {
                root: [0.0; 3],
                body: vec![0.0; (model.num_joints - 1) * 3],
                betas: pose.betas.clone(),
                translation: pose.translation,
            };
            if let Ok((_, rest_verts)) = model.forward_mesh(&rest) {
                let rest_bytes: Vec<u8> =
                    rest_verts.iter().flat_map(|v| v.to_le_bytes()).collect();
                let rest_path = path.with_extension("rest");
                std::fs::write(&rest_path, rest_bytes)?;
                println!("wrote {} (same shape, rest pose)", rest_path.display());
            }
        }
    }
    if let Some(path) = bvh {
        let report = velomocap_motion::write_clip_bvh(path, &clip, &model, clip.fps as f64)?;
        println!(
            "wrote {} ({} frames at {:.1} fps, {} bones, order {})",
            path.display(),
            clip.frames(),
            clip.fps,
            model.num_joints,
            report.orders.first().cloned().unwrap_or_default()
        );
    }
    if let Some(path) = fbx {
        let prepared = velomocap_motion::prepare_bvh(&clip, &model, clip.fps as f64)?;
        let report = velomocap_motion::write_prepared_fbx(path, &prepared)?;
        println!(
            "wrote {} (binary FBX, {} bones, {} keys, stop tick {})",
            path.display(),
            report.bones,
            report.key_count,
            report.stop_tick
        );
    }
    Ok(())
}

/// Read a camera track written by `velomocap-cli camera --json`.
fn load_camera_track(path: &std::path::Path, frames: usize) -> Result<velomocap_infer::CameraTrack> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?;
    let rows: serde_json::Value = serde_json::from_str(&text).context("parsing the camera track")?;
    let rows = rows.as_array().context("the camera track must be a JSON array")?;
    anyhow::ensure!(!rows.is_empty(), "{} holds no cameras", path.display());

    let mut extrinsics = Vec::with_capacity(rows.len());
    let mut intrinsics = Vec::with_capacity(rows.len());
    for (index, row) in rows.iter().enumerate() {
        let rt = row["rt"].as_array().context("rt")?;
        anyhow::ensure!(rt.len() == 16, "camera {index} has {} rt values", rt.len());
        let mut matrix = [[0.0f32; 4]; 4];
        for (offset, value) in rt.iter().enumerate() {
            matrix[offset / 4][offset % 4] = value.as_f64().context("rt value")? as f32;
        }
        extrinsics.push(matrix);
        let k = row["k"].as_array().context("k")?;
        anyhow::ensure!(k.len() == 9, "camera {index} has {} k values", k.len());
        let mut camera = [[0.0f32; 3]; 3];
        for (offset, value) in k.iter().enumerate() {
            camera[offset / 3][offset % 3] = value.as_f64().context("k value")? as f32;
        }
        intrinsics.push(camera);
    }
    anyhow::ensure!(
        extrinsics.len() >= frames,
        "the camera track has {} frames but the video needs {frames}",
        extrinsics.len()
    );
    let width = intrinsics[0][0][2] as f64;
    let height = intrinsics[0][1][2] as f64;
    Ok(velomocap_infer::CameraTrack {
        extrinsics,
        intrinsics,
        image_size: (
            (2.0 * height).round() as usize,
            (2.0 * width).round() as usize,
        ),
        model_size: (0, 0),
        sample_indices: Vec::new(),
    })
}

/// Binary-free ASCII PLY, enough to inspect a reconstructed body elsewhere.
fn write_ply(
    path: &std::path::Path,
    verts: &[f32],
    faces: &[[u32; 3]],
) -> Result<()> {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(verts.len() * 8 + 64);
    writeln!(out, "ply")?;
    writeln!(out, "format ascii 1.0")?;
    writeln!(out, "element vertex {}", verts.len() / 3)?;
    writeln!(out, "property float x")?;
    writeln!(out, "property float y")?;
    writeln!(out, "property float z")?;
    writeln!(out, "element face {}", faces.len())?;
    writeln!(out, "property list uchar int vertex_indices")?;
    writeln!(out, "end_header")?;
    for point in verts.chunks_exact(3) {
        writeln!(out, "{:.5} {:.5} {:.5}", point[0], point[1], point[2])?;
    }
    for face in faces {
        writeln!(out, "3 {} {} {}", face[0], face[1], face[2])?;
    }
    std::fs::write(path, out).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

fn capture_video(
    video: &std::path::Path,
    out: Option<PathBuf>,
    model: &std::path::Path,
    weights: &std::path::Path,
    confidence: f32,
    width: usize,
    gpu: bool,
    limit: usize,
) -> Result<()> {
    use velomocap_motion::{capture_with_progress, CaptureConfig, CaptureResult};

    let config = CaptureConfig {
        weights: resolve_model(weights),
        model: resolve_model(model),
        confidence,
        work_width: width,
        gpu,
        max_frames: limit,
        ..Default::default()
    };

    let mut last_stage = String::new();
    let mut last_frame = 0usize;
    let mut progress = |frame: usize, total: usize, stage: &str| {
        if stage != last_stage || frame % 20 == 0 || frame == total {
            if stage != last_stage || frame - last_frame >= 20 {
                eprint!("\r{stage:<12} {frame}/{total}      ");
                last_stage = stage.to_string();
                last_frame = frame;
            }
        }
    };
    let result = capture_with_progress(video, &config, &mut progress)?;
    eprintln!();

    println!("backend: {}", result.backend);
    println!("{}", result.summary());

    let target = out.unwrap_or_else(|| CaptureResult::default_bvh_path(video));
    let export = result.write_bvh(&target)?;
    println!("\n{}", export.summary());
    println!(
        "minimum singularity margin: {:.3}",
        export.minimum_singularity_margin
    );
    Ok(())
}

fn fit_report(
    image: &std::path::Path,
    model_path: &std::path::Path,
    weights: &std::path::Path,
    confidence: f32,
    gpu: bool,
) -> Result<()> {
    use velomocap_body::BodyModel;
    use velomocap_infer::{PoseConfig, SessionConfig};

    let body = BodyModel::load(resolve_model(model_path), 10)?;
    let config = PoseConfig {
        confidence,
        session: SessionConfig { gpu, ..Default::default() },
        ..Default::default()
    };
    let (poses, backend, width, height) =
        velomocap_infer::pose::estimate_file(weights, image, config)?;
    println!("backend: {}  image: {width}x{height}", backend.as_str());
    println!("{} person(s) detected\n", poses.len());

    let fit_config = velomocap_motion::FitConfig::default();
    let started = Instant::now();
    for (index, detection) in poses.iter().enumerate() {
        let fitter = velomocap_motion::Fitter::new(&body, width, height, fit_config.clone());
        let fitted = fitter.fit(detection, None)?;
        let pelvis = fitted.skeleton.joint(0);
        let head = fitted.skeleton.joint(15);
        println!(
            "#{index} box score {:.2} -> {} kp, reprojection {:.1} px, converged={}\n     pelvis=({:.2},{:.2},{:.2}) m  head=({:.2},{:.2},{:.2}) m  height={:.2} m",
            detection.bbox.score,
            fitted.used_keypoints,
            fitted.reprojection_error,
            fitted.converged,
            pelvis[0], pelvis[1], pelvis[2],
            head[0], head[1], head[2],
            (head[1] - fitted.skeleton.joint(7)[1]).abs() + 1.0,
        );
    }
    println!("\nfit time: {:.1} ms", started.elapsed().as_secs_f64() * 1e3);
    Ok(())
}

fn pose_report(
    image: &std::path::Path,
    weights: &std::path::Path,
    confidence: f32,
    gpu: bool,
) -> Result<()> {
    use velomocap_infer::{PoseConfig, SessionConfig, KEYPOINT_NAMES};

    let config = PoseConfig {
        confidence,
        session: SessionConfig { gpu, ..Default::default() },
        ..Default::default()
    };
    let started = Instant::now();
    let (poses, backend, width, height) =
        velomocap_infer::pose::estimate_file(weights, image, config)?;
    let elapsed = started.elapsed();

    println!("backend: {}  image: {width}x{height}", backend.as_str());
    println!("{} person(s) in {:.1} ms\n", poses.len(), elapsed.as_secs_f64() * 1e3);
    for (index, pose) in poses.iter().enumerate() {
        println!(
            "#{index} score={:.3} box=({:.0},{:.0})-({:.0},{:.0})  mean kp conf={:.2}",
            pose.bbox.score,
            pose.bbox.x0,
            pose.bbox.y0,
            pose.bbox.x1,
            pose.bbox.y1,
            pose.mean_keypoint_confidence()
        );
        for (k, name) in KEYPOINT_NAMES.iter().enumerate() {
            let point = pose.keypoints[k];
            if point.confidence < 0.3 {
                continue;
            }
            println!(
                "    {name:<16} {:>7.1} {:>7.1}  ({:.2})",
                point.x, point.y, point.confidence
            );
        }
    }
    Ok(())
}

fn body_report(model_path: &std::path::Path, betas: usize, mesh: bool) -> Result<()> {
    use velomocap_body::{BodyModel, Pose};

    let started = Instant::now();
    let mut model = BodyModel::load(resolve_model(model_path), betas)?;
    let load = started.elapsed();
    println!(
        "model: {} vertices, {} joints, {} betas  (loaded in {:.0} ms)",
        model.num_vertices,
        model.num_joints,
        model.num_betas,
        load.as_secs_f64() * 1e3
    );
    println!("faces: {}", model.faces.len() / 3);

    // Rest skeleton: report the main body joints and their heights.
    let pose = Pose::neutral(model.num_joints, model.num_betas);
    let skeleton = model.forward_joints(&pose);
    println!("\nrest joints (metres):");
    for joint in [0usize, 1, 2, 4, 5, 7, 8, 12, 15, 16, 17, 20, 21] {
        let p = skeleton.joint(joint);
        println!(
            "  {:<16} {:>7.3} {:>7.3} {:>7.3}",
            model.names[joint], p[0], p[1], p[2]
        );
    }

    // Forward-kinematics timing on a bent pose.
    let mut bent = pose.clone();
    bent.body[(4 - 1) * 3 + 0] = 0.6; // left knee
    bent.body[(5 - 1) * 3 + 0] = 0.6; // right knee
    bent.root = [0.1, 0.2, 0.0];
    bent.translation = [0.1, 0.2, 0.3];
    let iterations = 200;
    let start = Instant::now();
    for _ in 0..iterations {
        std::hint::black_box(model.forward_joints(&bent));
    }
    let per = start.elapsed().as_secs_f64() / iterations as f64;
    println!(
        "\nforward_joints: {:.1} us/call ({} calls)",
        per * 1e6,
        iterations
    );

    if mesh {
        let start = Instant::now();
        model = model.with_posedirs()?;
        println!("posedirs loaded in {:.0} ms", start.elapsed().as_secs_f64() * 1e3);
        // Neutral check: with zero rotations the skinned mesh must reproduce
        // v_template exactly, which is the ground truth for a correct LBS.
        let (_, neutral_vertices) = model.forward_mesh(&Pose::neutral(model.num_joints, model.num_betas))?;
        let min_y = neutral_vertices.chunks_exact(3).map(|v| v[1]).fold(f32::INFINITY, f32::min);
        let max_y = neutral_vertices.chunks_exact(3).map(|v| v[1]).fold(f32::NEG_INFINITY, f32::max);
        println!("neutral mesh height: {:.3} m (expect ~1.72)", max_y - min_y);
        println!("neutral pelvis-area y: {:.3}", neutral_vertices[1]);

        let start = Instant::now();
        let (_, vertices) = model.forward_mesh(&bent)?;
        let elapsed = start.elapsed().as_secs_f64() * 1e3;
        let finite = vertices.iter().all(|v| v.is_finite());
        println!(
            "forward_mesh: {:.1} ms, {} vertices, finite={finite}",
            elapsed,
            vertices.len() / 3
        );
        // Dump arrays for vertex-level comparison against the official model.
        if let Ok(dump) = std::env::var("VELOMOCAP_DUMP_MESH") {
            use std::io::Write as _;
            let (neutral_skeleton, neutral_vertices) =
                model.forward_mesh(&Pose::neutral(model.num_joints, model.num_betas))?;
            let (bent_skeleton, bent_vertices) = model.forward_mesh(&bent)?;
            let mut file = std::fs::File::create(&dump)?;
            let write_vec = |file: &mut std::fs::File, name: &str, data: &[f32]| {
                let _ = writeln!(file, "##{name} {}", data.len());
                for chunk in data.chunks(3) {
                    let _ = writeln!(file, "{:.7} {:.7} {:.7}", chunk[0], chunk[1], chunk[2]);
                }
            };
            write_vec(&mut file, "neutral_v", &neutral_vertices);
            write_vec(&mut file, "neutral_j", &neutral_skeleton.positions);
            write_vec(&mut file, "bent_v", &bent_vertices);
            write_vec(&mut file, "bent_j", &bent_skeleton.positions);
            println!("dumped mesh data to {dump}");
        }

        // A posed body should stay roughly 1.5-2.2 m tall.
        let min_y = vertices.chunks_exact(3).map(|v| v[1]).fold(f32::INFINITY, f32::min);
        let max_y = vertices.chunks_exact(3).map(|v| v[1]).fold(f32::NEG_INFINITY, f32::max);
        println!("mesh height: {:.3} m", max_y - min_y);
    }
    Ok(())
}

fn track_video(
    video: &std::path::Path,
    weights: &std::path::Path,
    confidence: f32,
    width: usize,
    gpu: bool,
    limit: usize,
) -> Result<()> {
    use velomocap_infer::{Detector, DetectorConfig, SessionConfig, Tracker};
    use velomocap_video::{Decoder, Tools};

    let tools = Tools::discover()?;
    let mut decoder = Decoder::open(&tools, video, Some(width))?;
    let info = decoder.info().clone();
    let (frame_w, frame_h) = decoder.dimensions();
    println!(
        "video: {}x{} @ {:.2} fps, {} frame(s)",
        info.width, info.height, info.fps, info.frames
    );
    println!("work resolution: {frame_w}x{frame_h}");

    let config = DetectorConfig {
        confidence,
        session: SessionConfig { gpu, ..Default::default() },
        ..Default::default()
    };
    let mut detector = Detector::load(weights, config)?;
    println!("detector: {}", detector.describe());
    let mut tracker = Tracker::new(0.3, 8);

    let started = Instant::now();
    let mut frames = 0usize;
    let mut total_detections = 0usize;
    while let Some(frame) = decoder.next_frame()? {
        let detections = detector.detect(&frame.rgb, frame.width, frame.height)?;
        total_detections += detections.len();
        tracker.update(&detections);
        frames += 1;
        if limit > 0 && frames >= limit {
            break;
        }
    }
    let elapsed = started.elapsed().as_secs_f64();
    tracker.prune(3);

    println!(
        "{frames} frame(s), {total_detections} detection(s), {} track(s) in {:.2} s ({:.1} fps)",
        tracker.tracks.len(),
        elapsed,
        frames as f64 / elapsed.max(1e-9)
    );
    for track in tracker.tracks.iter().take(8) {
        println!(
            "  track #{}: frames {}-{}, {} hit(s), mean score {:.2}",
            track.id,
            track.first_frame,
            track.last_frame,
            track.hits,
            track.mean_score()
        );
    }
    Ok(())
}

/// Same export the GUI's "Export tracks" button writes.
fn export_video_tracks(video: &std::path::Path) -> Result<()> {
    use velomocap_infer::{Detector, DetectorConfig, Tracker};
    use velomocap_video::{Decoder, Tools};

    let tools = Tools::discover()?;
    let mut decoder = Decoder::open(&tools, video, Some(960))?;
    let info = decoder.info().clone();
    let mut detector = Detector::load("models/detector/yolov8x.onnx", DetectorConfig::default())?;
    let mut tracker = Tracker::new(0.3, 8);
    let mut frames = 0usize;
    while let Some(frame) = decoder.next_frame()? {
        let detections = detector.detect(&frame.rgb, frame.width, frame.height)?;
        tracker.update(&detections);
        frames += 1;
    }
    tracker.prune(3);
    let result = velomocap_export::export(video, Some(&info), &tracker.tracks, frames)?;
    println!(
        "{} track(s), {} detection(s) -> {} and {}",
        result.tracks,
        result.detections,
        result.json.display(),
        result.csv.display()
    );
    Ok(())
}

fn camera_track(
    video: &std::path::Path,
    weights: &std::path::Path,
    stride: usize,
    max_samples: usize,
    gpu: bool,
    limit: usize,
    json: Option<PathBuf>,
) -> Result<()> {
    use velomocap_infer::{SessionConfig, VggtCamera, VggtConfig};
    use velomocap_video::Tools;

    let tools = Tools::discover()?;
    let config = VggtConfig {
        weights: resolve_model(weights),
        frame_stride: stride.max(1),
        max_samples,
        max_frames: limit,
        session: SessionConfig {
            gpu,
            ..Default::default()
        },
        ..Default::default()
    };
    let load = Instant::now();
    let mut camera = VggtCamera::load(config)?;
    let load_seconds = load.elapsed().as_secs_f64();

    let run = Instant::now();
    let track = camera.estimate(&tools, video)?;
    let run_seconds = run.elapsed().as_secs_f64();

    println!(
        "vggt-omega [{}]  {}x{} -> {}x{}  {} frames ({} sampled)  load {:.2}s run {:.2}s",
        camera.backend().as_str(),
        track.image_size.1,
        track.image_size.0,
        track.model_size.1,
        track.model_size.0,
        track.len(),
        track.sample_indices.len(),
        load_seconds,
        run_seconds
    );
    for (i, matrix) in track.extrinsics.iter().enumerate().take(3) {
        println!(
            "  f{i:05}  t=({:+.3}, {:+.3}, {:+.3})  r0=({:+.3}, {:+.3}, {:+.3})  fx={:.1}",
            matrix[0][3],
            matrix[1][3],
            matrix[2][3],
            matrix[0][0],
            matrix[0][1],
            matrix[0][2],
            track.intrinsics[i][0][0]
        );
    }

    if let Some(path) = json {
        let mut rows = String::from("[");
        for (i, matrix) in track.extrinsics.iter().enumerate() {
            if i > 0 {
                rows.push(',');
            }
            let flat: Vec<String> = matrix
                .iter()
                .flatten()
                .map(|v| format!("{v:.6}"))
                .collect();
            let k: Vec<String> = track.intrinsics[i]
                .iter()
                .flatten()
                .map(|v| format!("{v:.6}"))
                .collect();
            rows.push_str(&format!(
                "\n  {{\"frame\": {i}, \"rt\": [{}], \"k\": [{}]}}",
                flat.join(","),
                k.join(",")
            ));
        }
        rows.push_str("\n]");
        std::fs::write(&path, rows)?;
        println!("camera track -> {}", path.display());
    }
    Ok(())
}

/// Compare the ONNX graph against the tensors the PyTorch reference produced.
fn vggt_parity(dir: &std::path::Path, weights: &std::path::Path, gpu: bool) -> Result<()> {
    use velomocap_infer::{SessionConfig, VggtCamera, VggtConfig};

    let read_f32 = |name: &str| -> Result<Vec<f32>> {
        let bytes = std::fs::read(dir.join(name))?;
        anyhow::ensure!(bytes.len() % 4 == 0, "{name}: not a float32 dump");
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    };
    let dims: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("dims.json"))?)?;
    let n = dims["frames"].as_u64().context("dims.frames")? as usize;
    let h = dims["height"].as_u64().context("dims.height")? as usize;
    let w = dims["width"].as_u64().context("dims.width")? as usize;
    let h_patches = dims["h_patches"].as_u64().context("dims.h_patches")? as usize;
    let w_patches = dims["w_patches"].as_u64().context("dims.w_patches")? as usize;

    let images = read_f32("images.bin")?;
    let expected = read_f32("pose_enc.bin")?;

    let (rope_sin, rope_cos) = velomocap_infer::rope_table(h_patches, w_patches);

    let config = VggtConfig {
        weights: resolve_model(weights),
        session: SessionConfig {
            gpu,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut camera = VggtCamera::load(config)?;
    let got = camera.run_raw(n, h, w, images, rope_sin, rope_cos)?;
    anyhow::ensure!(got.len() == expected.len(), "pose encoding length mismatch");

    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    for (a, b) in got.iter().zip(expected.iter()) {
        let d = (a - b).abs();
        max_abs = max_abs.max(d);
        sum_abs += d as f64;
    }
    println!(
        "vggt-omega parity: {} values, max abs diff {:.3e}, mean {:.3e}",
        got.len(),
        max_abs,
        sum_abs / got.len() as f64
    );
    println!("  first 3: onnx  {:?}", &got[..3]);
    println!("           torch {:?}", &expected[..3]);
    if max_abs > 2e-3 {
        anyhow::bail!("parity outside tolerance");
    }
    println!("parity ok");
    Ok(())
}

/// Compare the FlowHMR MMDiT graph against the PyTorch reference tensors.
fn flowhmr_parity(dir: &std::path::Path, weights: &std::path::Path, gpu: bool) -> Result<()> {
    use velomocap_infer::{MotionConfig, MotionGenerator, Narrowband, SessionConfig};

    let read_f32 = |name: &str| -> Result<Vec<f32>> {
        let bytes = std::fs::read(dir.join(name))?;
        anyhow::ensure!(bytes.len() % 4 == 0, "{name}: not a float32 dump");
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    };
    let dims: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("fh_dims.json"))?)?;
    let frames = dims["motion_len"].as_u64().context("motion_len")? as usize;
    let ctxt = dims["ctxt_len"].as_u64().context("ctxt_len")? as usize;
    let input_dim = dims["input_dim"].as_u64().context("input_dim")? as usize;
    let feature_dim = dims["feature_dim"].as_u64().context("feature_dim")? as usize;
    let vtxt_dim = dims["vtxt_dim"].as_u64().context("vtxt_dim")? as usize;

    let x = read_f32("x.bin")?;
    let cam = read_f32("cam.bin")?;
    let feature = read_f32("feature.bin")?;
    let vtxt = read_f32("vtxt.bin")?;
    let timestep = read_f32("timestep.bin")?;
    let mask = read_f32("mask.bin")?;
    let expected = read_f32("pred.bin")?;
    let valid: Vec<u8> = std::fs::read(dir.join("valid.bin"))?;

    let nb = &dims["narrowband"];
    let narrowband = Narrowband {
        mm: nb["mm"].as_i64().context("mm")?,
        mc: nb["mc"].as_i64().context("mc")?,
        cm: nb["cm"].as_i64().context("cm")?,
        cc: nb["cc"].as_i64().context("cc")?,
    };

    let config = MotionConfig {
        weights: resolve_model(weights),
        stats: if dir.join("motion_stats.json").is_file() {
            dir.join("motion_stats.json")
        } else {
            std::path::PathBuf::from("models/flowhmr/motion_stats.json")
        },
        session: SessionConfig {
            gpu,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut generator = MotionGenerator::load(config, narrowband, (input_dim, feature_dim, vtxt_dim))?;

    let got = generator.step_raw(
        &x,
        &cam,
        &feature,
        frames,
        &vtxt,
        &valid,
        &mask,
        ctxt,
        timestep[0],
    )?;
    anyhow::ensure!(got.len() == expected.len(), "prediction length mismatch");

    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut over_1e2 = 0usize;
    let mut over_1e3 = 0usize;
    for (a, b) in got.iter().zip(expected.iter()) {
        let d = (a - b).abs();
        max_abs = max_abs.max(d);
        sum_abs += d as f64;
        if d > 1e-2 {
            over_1e2 += 1;
        }
        if d > 1e-3 {
            over_1e3 += 1;
        }
    }
    println!(
        "flowhmr parity: {} values, max abs diff {:.3e}, mean {:.3e}",
        got.len(),
        max_abs,
        sum_abs / got.len() as f64
    );
    println!(
        "  |diff| > 1e-3: {over_1e3} ({:.4}%), > 1e-2: {over_1e2} ({:.4}%)",
        100.0 * over_1e3 as f64 / got.len() as f64,
        100.0 * over_1e2 as f64 / got.len() as f64
    );
    println!("  first 4: onnx  {:?}", &got[..4]);
    println!("           torch {:?}", &expected[..4]);
    // 18 stacked blocks of float32 attention accumulate a little drift; the
    // reference itself moves this much between GPU and CPU. Anything beyond
    // this means the graph and the model disagree structurally.
    if max_abs > 3e-2 {
        anyhow::bail!("parity outside tolerance");
    }
    println!("parity ok");
    Ok(())
}

fn bench_detect(
    weights: &std::path::Path,
    image: &std::path::Path,
    repeat: usize,
    gpu: bool,
) -> Result<()> {
    use velomocap_infer::{Detector, DetectorConfig, SessionConfig};
    let (detections, _, width, height) =
        velomocap_infer::detector::detect_file(weights, image, DetectorConfig {
            session: SessionConfig { gpu, ..Default::default() },
            ..Default::default()
        })?;
    drop(detections);
    let picture = std::fs::read(image)?;
    let rgb = decode_rgb(&picture, width, height)?;

    let config = DetectorConfig {
        session: SessionConfig {
            gpu,
            ..Default::default()
        },
        ..Default::default()
    };
    let load_start = Instant::now();
    let mut detector = Detector::load(weights, config)?;
    let load = load_start.elapsed();
    println!("{}", detector.describe());
    println!("load: {:.1} ms", load.as_secs_f64() * 1e3);

    // Warm-up: first inference pays kernel compilation / cuDNN init.
    let warm_start = Instant::now();
    let first = detector.detect(&rgb, width, height)?;
    let warm = warm_start.elapsed();
    println!(
        "warm-up inference: {:.1} ms ({} detection(s))",
        warm.as_secs_f64() * 1e3,
        first.len()
    );

    let mut best = f64::INFINITY;
    let mut total = 0.0;
    let mut count = 0;
    for _ in 0..repeat.max(1) {
        let started = Instant::now();
        let found = detector.detect(&rgb, width, height)?;
        let elapsed = started.elapsed().as_secs_f64() * 1e3;
        best = best.min(elapsed);
        total += elapsed;
        count += 1;
        std::hint::black_box(found);
    }
    println!(
        "steady state: best {:.2} ms, mean {:.2} ms over {count} runs",
        best,
        total / count as f64
    );
    Ok(())
}

fn detect(weights: &std::path::Path, image: &std::path::Path, confidence: f32, gpu: bool) -> Result<()> {
    let config = velomocap_infer::DetectorConfig {
        confidence,
        session: velomocap_infer::SessionConfig {
            gpu,
            ..Default::default()
        },
        ..Default::default()
    };
    let started = Instant::now();
    let (detections, backend, width, height) =
        velomocap_infer::detector::detect_file(weights, image, config)?;
    let elapsed = started.elapsed();
    println!("backend: {}  image: {width}x{height}", backend.as_str());
    print!("{}", velomocap_infer::detector::format_detections(&detections));
    println!("{} detection(s) in {:.1} ms", detections.len(), elapsed.as_secs_f64() * 1e3);
    Ok(())
}

fn synthetic_binding() -> Binding {
    // A plausible 52-bone Mixamo-shaped rest pose; enough to exercise the
    // export writers without any model data.
    let mut names: Vec<String> = (1..52).map(|j| format!("Bone{j:02}")).collect();
    names.insert(0, "Hips".to_string());
    let mut parents = vec![-1i32];
    for j in 1..52 {
        parents.push((j - 1) as i32);
    }
    let rest: Vec<f64> = (0..52)
        .flat_map(|j| [0.02 * (j % 3) as f64, 0.03 * j as f64, 0.0])
        .collect();
    let ends: Vec<f64> = (0..52).flat_map(|_| [0.0, 0.04, 0.0]).collect();
    let mut basis = Vec::with_capacity(52 * 9);
    for j in 0..52 {
        // Small constant rotation per bone so the conversion is non-trivial.
        let a = 0.01 * j as f64;
        basis.extend_from_slice(&[
            a.cos(), 0.0, a.sin(),
            0.0, 1.0, 0.0,
            -a.sin(), 0.0, a.cos(),
        ]);
    }
    Binding {
        names,
        parents,
        rest,
        ends,
        basis,
        origin: [0.0; 3],
        version: 17,
    }
}

/// Rotation by `angle` about axis `k` (0=x, 1=y, 2=z), row-major 3x3.
fn axis_rotation(k: usize, angle: f64) -> [f64; 9] {
    let (c, s) = (angle.cos(), angle.sin());
    match k {
        0 => [1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c],
        1 => [c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c],
        _ => [c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0],
    }
}

fn mul3(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
    let mut out = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            out[i * 3 + j] =
                a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
        }
    }
    out
}

fn synthetic_motion(frames: usize, bones: usize) -> (Vec<f64>, Vec<f64>) {
    let mut roots = Vec::with_capacity(frames * 3);
    let mut local = Vec::with_capacity(frames * bones * 9);
    for t in 0..frames {
        let phase = t as f64 * 0.05;
        roots.push(0.01 * phase.sin());
        roots.push(0.0);
        roots.push(0.01 * phase.cos());
        for j in 0..bones {
            // A smooth, always-orthonormal local rotation per bone.
            let a = 0.35 * (phase + j as f64 * 0.1).sin();
            let b = 0.25 * (phase * 0.7 + j as f64).cos();
            let c = 0.20 * (phase * 1.3 - j as f64 * 0.2).sin();
            let m = mul3(
                &mul3(&axis_rotation(0, a), &axis_rotation(1, b)),
                &axis_rotation(2, c),
            );
            local.extend_from_slice(&m);
        }
    }
    (roots, local)
}

fn bench(frames: usize, repeat: usize, json: Option<PathBuf>) -> Result<()> {
    let binding = synthetic_binding();
    let bones = binding.bones();
    let (roots, local) = synthetic_motion(frames, bones);
    let mut rows: Vec<BenchRow> = Vec::new();
    let folder = std::env::temp_dir().join("velomocap-bench");
    std::fs::create_dir_all(&folder)?;

    let mut best = f64::INFINITY;
    for _ in 0..repeat.max(1) {
        let start = Instant::now();
        bvh::write_bvh(&folder.join("bench.bvh"), &binding, &roots, &local, 30.0, 17)?;
        best = best.min(start.elapsed().as_secs_f64());
    }
    let bvh_bytes = std::fs::metadata(folder.join("bench.bvh"))?.len();
    rows.push(BenchRow {
        stage: "bvh".into(),
        frames,
        seconds: best,
        detail: format!("{:.1} MB", bvh_bytes as f64 / 1e6),
    });

    // Rigid world transforms from the local rotations, so the writer's
    // constant-offset guard is exercised on valid input.
    let mut matrices = vec![0.0; frames * bones * 16];
    let mut world = vec![0.0f64; bones * 9];
    let mut position = vec![0.0f64; bones * 3];
    for t in 0..frames {
        for j in 0..bones {
            let local_rotation = &local[(t * bones + j) * 9..(t * bones + j) * 9 + 9];
            let parent = binding.parents[j];
            let m = if parent < 0 {
                let mut m = [0.0; 9];
                m.copy_from_slice(local_rotation);
                m
            } else {
                let p = parent as usize;
                mul3(&world[p * 9..p * 9 + 9].try_into().unwrap(), local_rotation.try_into().unwrap())
            };
            world[j * 9..j * 9 + 9].copy_from_slice(&m);
            if parent < 0 {
                position[0] = roots[t * 3];
                position[1] = roots[t * 3 + 1];
                position[2] = roots[t * 3 + 2];
            } else {
                let p = parent as usize;
                let delta = [
                    binding.rest[j * 3] - binding.rest[p * 3],
                    binding.rest[j * 3 + 1] - binding.rest[p * 3 + 1],
                    binding.rest[j * 3 + 2] - binding.rest[p * 3 + 2],
                ];
                let pw = &world[p * 9..p * 9 + 9];
                position[j * 3] = position[p * 3] + pw[0] * delta[0] + pw[1] * delta[1] + pw[2] * delta[2];
                position[j * 3 + 1] = position[p * 3 + 1] + pw[3] * delta[0] + pw[4] * delta[1] + pw[5] * delta[2];
                position[j * 3 + 2] = position[p * 3 + 2] + pw[6] * delta[0] + pw[7] * delta[1] + pw[8] * delta[2];
            }
            // Row-major 4x4: rotation rows occupy columns 0..3 of rows 0..2.
            let base = (t * bones + j) * 16;
            for r in 0..3 {
                matrices[base + r * 4] = world[j * 9 + r * 3];
                matrices[base + r * 4 + 1] = world[j * 9 + r * 3 + 1];
                matrices[base + r * 4 + 2] = world[j * 9 + r * 3 + 2];
                matrices[base + r * 4 + 3] = 0.0;
            }
            matrices[base + 12] = 0.0;
            matrices[base + 13] = 0.0;
            matrices[base + 14] = 0.0;
            matrices[base + 15] = 1.0;
            matrices[base + 3] = position[j * 3];
            matrices[base + 7] = position[j * 3 + 1];
            matrices[base + 11] = position[j * 3 + 2];
        }
    }
    let display = binding.basis.clone();
    let tip: Vec<f64> = (0..bones)
        .flat_map(|j| {
            let m = &binding.basis[j * 9..j * 9 + 9];
            let e = &binding.ends[j * 3..j * 3 + 3];
            [
                m[0] * e[0] + m[3] * e[1] + m[6] * e[2],
                m[1] * e[0] + m[4] * e[1] + m[7] * e[2],
                m[2] * e[0] + m[5] * e[1] + m[8] * e[2],
            ]
        })
        .collect();
    let mut best = f64::INFINITY;
    for _ in 0..repeat.max(1) {
        let input = velomocap_core::fbx::FbxInput {
            names: &binding.names,
            parents: &binding.parents,
            rest: &binding.rest,
            rest_frames: &display,
            display: &display,
            tip_local: &tip,
            frames: &matrices,
            fps: 30.0,
            transform: velomocap_core::fbx::IDENTITY4,
            prepend_tpose: false,
            expected_bones: bones,
            rig_name: "Bench",
            creator: "VeloMocap",
        };
        let start = Instant::now();
        velomocap_core::fbx::write(&folder.join("bench.fbx"), &input)?;
        best = best.min(start.elapsed().as_secs_f64());
    }
    let fbx_bytes = std::fs::metadata(folder.join("bench.fbx"))?.len();
    rows.push(BenchRow {
        stage: "fbx".into(),
        frames,
        seconds: best,
        detail: format!("{:.1} MB", fbx_bytes as f64 / 1e6),
    });

    // Contact planning on a synthetic walk.
    let mut probes = vec![0.0; frames * 4 * 3];
    let mut probabilities = vec![0.0; frames * 4];
    for t in 0..frames {
        let phase = t as f64 * 0.21;
        for foot in 0..2 {
            let offset = phase + std::f64::consts::PI * foot as f64;
            let lift = offset.sin().max(0.0) * 0.09;
            let stride = 0.18 * offset.cos();
            probes[(t * 4 + foot * 2) * 3] = stride;
            probes[(t * 4 + foot * 2) * 3 + 1] = lift;
            probes[(t * 4 + foot * 2) * 3 + 2] = 0.02;
            probes[(t * 4 + foot * 2 + 1) * 3] = stride + 0.11;
            probes[(t * 4 + foot * 2 + 1) * 3 + 1] = lift + 0.01;
            probabilities[t * 4 + foot * 2] = (1.2 - lift / 0.05).clamp(0.0, 1.0);
            probabilities[t * 4 + foot * 2 + 1] = (1.1 - lift / 0.05).clamp(0.0, 1.0);
        }
    }
    let preset = Preset::named("normal")?;
    let mut best = f64::INFINITY;
    let mut detail = String::new();
    for _ in 0..repeat.max(1) {
        let start = Instant::now();
        let plan =
            contact::prepare_contacts(&probes, &probabilities, frames, 30.0, None, &preset, None)?;
        best = best.min(start.elapsed().as_secs_f64());
        detail = format!("{} stances, floor {:.3} m", plan.segments.len(), plan.floor);
    }
    rows.push(BenchRow {
        stage: "contacts".into(),
        frames,
        seconds: best,
        detail,
    });

    println!(
        "{:<12}{:>8}{:>12}{:>14}  {}",
        "stage", "frames", "seconds", "us/frame", "detail"
    );
    println!("{}", "-".repeat(70));
    for row in &rows {
        println!(
            "{:<12}{:>8}{:>12.5}{:>14.2}  {}",
            row.stage,
            row.frames,
            row.seconds,
            row.seconds * 1e6 / row.frames as f64,
            row.detail
        );
    }

    if let Some(path) = json {
        let text = serde_json::to_string_pretty(&rows).context("serialize benchmark")?;
        std::fs::write(&path, text)?;
        println!("wrote {}", path.display());
    }
    Ok(())
}



