//! FlowHMR motion: SAM tokens + camera rotations → a coherent SMPL-H clip.
//!
//! This is the stage that turns independent per-frame MHR reconstructions into
//! one temporally consistent motion, and it is what the original application
//! exports. The pieces, in the order the reference runs them:
//!
//! 1. **Features.** One `3072` value row per frame: the SAM body token (`1024`)
//!    followed by the left and right hand tokens (`1024` each). Frames the
//!    tracker missed reuse the previous row.
//! 2. **Camera.** VGGT-Omega's world-to-camera rotations, normalised against
//!    frame zero (`build_feature_dict`).
//! 3. **Generation.** The exported MMDiT, 20 Euler steps, decoded to SMPL-H
//!    `rot6d` / `shapes` / `trans`.
//! 4. **Placement.** The reference's `flow_placement`: match the SAM MHR hips
//!    (camera coordinates) against FlowHMR's own pelvis over the first frames
//!    and take the median difference on X and Z only, so FlowHMR keeps its own
//!    height and its own motion.
//!
//! Nothing here invents motion: when a step cannot be reproduced the function
//! returns an error rather than falling back to a different convention.

use std::path::Path;

use anyhow::{Context, Result};
use velomocap_body::{BodyModel, Pose, Skeleton};
use velomocap_core::bvh::{write_bvh_with_orders, BvhReport};
use velomocap_core::rig::Binding;
use velomocap_infer::{
    CameraTrack, GeneratedMotion, MotionConfig, MotionGenerator, Narrowband,
};

use crate::sam::{PersonFrame, PersonTrack};

/// Narrowband attention windows from `models/flowhmr/config.yml`.
///
/// `narrowband_length` is scaled to frames by the reference constructor
/// (`2.0 × 30 = 60`); the v2m and tt lengths are used as raw frame counts, which
/// is why the context windows are three frames wide, not ninety.
pub const NARROWBAND: Narrowband = Narrowband {
    mm: 60,
    mc: 3,
    cm: -1,
    cc: 3,
};

pub const INPUT_DIM: usize = 491;
pub const FEATURE_DIM: usize = 3072;
pub const VTXT_DIM: usize = 768;
/// Values of a single decoder token.
pub const TOKEN_DIM: usize = 1024;
/// `input_dim.ctxt_input_dim.feature` split: body, left hand, right hand.
pub const HAND_TOKENS: usize = 2;
/// MHR70 hip keypoints the placement anchor averages.
pub const HIP_KEYPOINTS: [usize; 2] = [9, 10];
/// The reference matches at most the first 30 frames.
pub const ANCHOR_FRAMES: usize = 30;

#[derive(Clone, Debug)]
pub struct FlowConfig {
    pub motion: MotionConfig,
    /// Stop after this many frames (0 = the whole track).
    pub max_frames: usize,
    /// First video frame of the clip (0 = the track's own start).
    pub start: usize,
    /// Diagnostic: replace every feature row with zeros. The result must differ
    /// from a real run, otherwise the graph is ignoring its video context.
    pub blank_tokens: bool,
    /// Diagnostic: start the ODE from zeros instead of Gaussian noise, which
    /// makes the sampler reproducible against an independent implementation.
    pub zero_noise: bool,
    /// Diagnostic: write the exact tensors the graph saw (`feature.bin`,
    /// `cam.bin`, `noise.bin`, `latent.bin`, `dims.json`) into this folder.
    pub dump: Option<std::path::PathBuf>,
}

impl Default for FlowConfig {
    fn default() -> Self {
        Self {
            motion: MotionConfig::default(),
            max_frames: 0,
            start: 0,
            blank_tokens: false,
            zero_noise: false,
            dump: None,
        }
    }
}

/// A generated clip, already decoded and placed.
pub struct FlowClip {
    /// Frames per second the motion was generated at.
    pub fps: f32,
    /// Per-frame SMPL-H parameters with the reference's 180° Y rotation applied.
    pub poses: Vec<Pose>,
    /// Per-frame forward pass of the SMPL-H model (world joints and rotations).
    pub skeletons: Vec<Skeleton>,
    /// Local rotations in 6-D form, as the graph produced them.
    pub rot6d: Vec<[[f32; 6]; 52]>,
    /// Root translation before the 180° Y rotation.
    pub trans: Vec<[f32; 3]>,
    pub betas: Vec<f32>,
    pub foot_contacts: Vec<[f32; 4]>,
    /// Height subtracted so the first frame's lowest point sits on `y = 0`.
    pub floor: f32,
    /// Whether the floor came from the mesh (the reference) or the joints.
    pub floor_from_mesh: bool,
    /// X/Z placement offset applied to reach the scene.
    pub offset: [f32; 3],
    /// Clip frame indices the motion was built from.
    pub source_frames: Vec<usize>,
    /// Anchors that contributed to the offset.
    pub anchors: usize,
    /// RMS distance (metres) between the placed pelvis and the SAM anchor, per
    /// axis. Small values mean the placement actually landed on the SAM body.
    pub anchor_residual: [f32; 3],
}

impl FlowClip {
    pub fn frames(&self) -> usize {
        self.poses.len()
    }

    /// Root (hips) position per frame, metres, y-up.
    pub fn root_track(&self) -> Vec<[f32; 3]> {
        self.skeletons
            .iter()
            .map(|skeleton: &Skeleton| skeleton.joint(0))
            .collect()
    }
}

/// Build the `3072`-wide feature rows for one person.
///
/// The layout is the reference's: body token `[0..1024]`, left hand
/// `[1024..2048]`, right hand `[2048..3072]`. The hand rows are zero when the
/// backend did not run the hands (the ONNX chain, or frames the reference did
/// not sample them on — `flow_fast_sam.py` leaves `tokens[:, 1024:]` zero
/// where no hand was sampled). The TorchScript backend does run them, so they
/// carry the real rows there.
pub fn feature_rows(track: &PersonTrack, frames: &[usize]) -> Result<Vec<f32>> {
    // Diagnostic, mirroring `flow --blank`: drop the hand rows so the effect of
    // the hand decoder on the generated motion can be isolated. The reference
    // leaves them zero for frames it does not sample hands on.
    let zero_hands = std::env::var_os("VELOMOCAP_ZERO_HANDS").is_some();
    let mut rows = Vec::with_capacity(frames.len() * FEATURE_DIM);
    let mut previous: Option<&PersonFrame> = None;
    for index in frames {
        let frame = match track.frames.get(*index).and_then(|f| f.as_ref()) {
            Some(frame) => frame,
            None => previous.with_context(|| {
                format!("{} has no body token at or before frame {index}", track.name)
            })?,
        };
        let token = &frame.body_token[..];
        anyhow::ensure!(
            token.len() == TOKEN_DIM,
            "frame {index} carries {} token values, expected {TOKEN_DIM}",
            token.len()
        );
        rows.extend_from_slice(token);
        let hands = !zero_hands
            && frame.left_token.len() == TOKEN_DIM
            && frame.right_token.len() == TOKEN_DIM;
        if hands {
            rows.extend_from_slice(&frame.left_token);
            rows.extend_from_slice(&frame.right_token);
        } else {
            rows.extend(std::iter::repeat_n(0.0f32, TOKEN_DIM * HAND_TOKENS));
        }
        previous = Some(frame);
    }
    Ok(rows)
}

/// The clip frames the motion stage uses: contiguous, inside the camera track.
fn clip_frames(
    track: &PersonTrack,
    camera: &CameraTrack,
    max_frames: usize,
    start: usize,
) -> Result<Vec<usize>> {
    anyhow::ensure!(track.hits > 0, "{} has no reconstructed frames", track.name);
    let first = track.first_frame.max(start);
    let last = track.last_frame.min(camera.len().saturating_sub(1));
    anyhow::ensure!(
        first <= last,
        "{} starts at frame {} but the camera track only covers {} frames",
        track.name,
        first,
        camera.len()
    );
    let mut frames: Vec<usize> = (first..=last).collect();
    if max_frames > 0 && frames.len() > max_frames {
        frames.truncate(max_frames);
    }
    Ok(frames)
}

/// Generate and place one person's motion.
pub fn generate(
    track: &PersonTrack,
    camera: &CameraTrack,
    body: &BodyModel,
    config: &FlowConfig,
) -> Result<FlowClip> {
    let frames = clip_frames(track, camera, config.max_frames, config.start)?;
    let count = frames.len();
    let mut tokens = feature_rows(track, &frames)?;
    if config.blank_tokens {
        tokens.fill(0.0);
    }

    // Camera rows are per *video* frame; slice the ones this clip uses.
    let all_rows = camera.relative_rotation_rows();
    anyhow::ensure!(
        all_rows.len() >= camera.len() * 9,
        "camera track has {} rotation values for {} frames",
        all_rows.len(),
        camera.len()
    );
    let mut camera_rows = Vec::with_capacity(count * 9);
    for index in &frames {
        camera_rows.extend_from_slice(&all_rows[index * 9..index * 9 + 9]);
    }

    let mut generator = MotionGenerator::load(
        config.motion.clone(),
        NARROWBAND,
        (INPUT_DIM, FEATURE_DIM, VTXT_DIM),
    )?;
    let fps = config.motion.fps;
    let noise = if config.zero_noise {
        vec![0.0f32; count * INPUT_DIM]
    } else {
        velomocap_infer::motion::gaussian(count * INPUT_DIM, config.motion.seed)
    };
    let latent = generator.integrate_from(&tokens, count, &camera_rows, &noise)?;
    if let Some(dir) = &config.dump {
        // One bare step, so an independent implementation can compare before
        // the loop has a chance to hide a difference.
        let mask = NARROWBAND.mask(count, count);
        let valid = vec![1u8; count];
        let vtxt = vec![0.0f32; VTXT_DIM];
        let pred0 = generator.step_raw(
            &noise,
            &camera_rows,
            &tokens,
            count,
            &vtxt,
            &valid,
            &mask,
            count,
            0.0,
        )?;
        dump_tensors(dir, &tokens, &camera_rows, &noise, &latent, &pred0, &mask, count)?;
    }
    let betas = generator.shapes_from_latent(&latent, count);

    // `_motion_to_smplh`: the rest pelvis comes from the predicted shapes.
    let rest = body.rest_joints(&betas);
    anyhow::ensure!(rest.len() >= 3, "the SMPL-H model regressed no pelvis");
    let pelvis_rest = [rest[0], rest[1], rest[2]];
    let motion = generator.decode_latent(&latent, count, pelvis_rest)?;

    finish(track, motion, frames, camera, body, fps)
}

/// Write the tensors the graph saw, so an independent implementation can
/// reproduce the same 20 steps. Flat little-endian `f32`, row-major.
fn dump_tensors(
    dir: &std::path::Path,
    features: &[f32],
    camera: &[f32],
    noise: &[f32],
    latent: &[f32],
    pred0: &[f32],
    mask: &[f32],
    frames: usize,
) -> Result<()> {
    std::fs::create_dir_all(dir)
        .with_context(|| format!("creating {}", dir.display()))?;
    for (name, values) in [
        ("feature.bin", features),
        ("cam.bin", camera),
        ("noise.bin", noise),
        ("latent.bin", latent),
        ("pred0.bin", pred0),
        ("mask.bin", mask),
    ] {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(dir.join(name), bytes)
            .with_context(|| format!("writing {name}"))?;
    }
    let dims = format!(
        "{{\n  \"frames\": {frames},\n  \"input_dim\": {INPUT_DIM},\n  \"feature_dim\": {FEATURE_DIM},\n  \"vtxt_dim\": {VTXT_DIM},\n  \"narrowband\": {{\"mm\": {}, \"mc\": {}, \"cm\": {}, \"cc\": {}}},\n  \"steps\": 20,\n  \"fps\": {}\n}}\n",
        NARROWBAND.mm, NARROWBAND.mc, NARROWBAND.cm, NARROWBAND.cc, 30.0
    );
    std::fs::write(dir.join("dims.json"), dims)
        .with_context(|| "writing dims.json".to_string())?;
    Ok(())
}

/// Turn decoded SMPL-H parameters into placed skeletons.
fn finish(
    track: &PersonTrack,
    motion: GeneratedMotion,
    frames: Vec<usize>,
    camera: &CameraTrack,
    body: &BodyModel,
    fps: f32,
) -> Result<FlowClip> {
    let count = motion.frames();
    anyhow::ensure!(count > 0, "FlowHMR produced no frames");
    anyhow::ensure!(
        body.num_joints >= 52,
        "FlowHMR needs a 52-joint SMPL-H model, {} were loaded",
        body.num_joints
    );

    let angles = motion.poses_axis_angle();
    let trans = motion.trans_y180();
    let betas = motion.betas.clone();
    let mut poses = Vec::with_capacity(count);
    let mut skeletons = Vec::with_capacity(count);
    for index in 0..count {
        let mut pose = Pose::neutral(body.num_joints, body.num_betas);
        pose.root = angles[index][0];
        for joint in 1..body.num_joints.min(52) {
            let at = (joint - 1) * 3;
            pose.body[at..at + 3].copy_from_slice(&angles[index][joint]);
        }
        pose.betas = betas.clone();
        pose.translation = trans[index];
        skeletons.push(body.forward_joints(&pose));
        poses.push(pose);
    }

    // Reference `v2m_inference`: drop the origin clip to the floor using the
    // lowest *mesh* vertex of the first frame, so the model's own vertical
    // datum never reaches the export.
    let (floor, floor_from_mesh) = floor_offset(body, &poses);
    if floor != 0.0 {
        for pose in poses.iter_mut() {
            pose.translation[1] -= floor;
        }
        for skeleton in skeletons.iter_mut() {
            for point in skeleton.positions.chunks_exact_mut(3) {
                point[1] -= floor;
            }
        }
    }

    let (offset, anchors) = placement(&frames, track, camera, &skeletons)?;
    if offset != [0.0; 3] {
        for pose in poses.iter_mut() {
            pose.translation[0] += offset[0];
            pose.translation[2] += offset[2];
        }
        for skeleton in skeletons.iter_mut() {
            for point in skeleton.positions.chunks_exact_mut(3) {
                point[0] += offset[0];
                point[2] += offset[2];
            }
        }
    }
    let anchor_residual = residual(&frames, track, camera, &skeletons);

    Ok(FlowClip {
        fps,
        poses,
        skeletons,
        rot6d: motion.rot6d,
        trans,
        betas,
        foot_contacts: motion.foot_contacts,
        floor,
        floor_from_mesh,
        offset,
        source_frames: frames,
        anchors,
        anchor_residual,
    })
}

/// Root-mean-square distance between the placed pelvis and each SAM anchor.
fn residual(
    frames: &[usize],
    track: &PersonTrack,
    camera: &CameraTrack,
    skeletons: &[Skeleton],
) -> [f32; 3] {
    let up = camera.to_y_up();
    let mut sum = [0.0f64; 3];
    let mut count = 0usize;
    for (slot, index) in frames.iter().enumerate().take(ANCHOR_FRAMES) {
        let Some(anchor) = track
            .frames
            .get(*index)
            .and_then(|frame| frame.as_ref())
            .map(|frame| frame.anchor_cam)
        else {
            continue;
        };
        let Some(extrinsic) = up.get(*index) else {
            continue;
        };
        let pelvis = skeletons[slot].joint(0);
        for axis in 0..3 {
            let mut rotated = 0.0f32;
            for k in 0..3 {
                rotated += extrinsic[k][axis] * anchor[k];
            }
            let difference = (rotated - pelvis[axis]) as f64;
            sum[axis] += difference * difference;
        }
        count += 1;
    }
    if count == 0 {
        return [0.0; 3];
    }
    [
        (sum[0] / count as f64).sqrt() as f32,
        (sum[1] / count as f64).sqrt() as f32,
        (sum[2] / count as f64).sqrt() as f32,
    ]
}

/// The clip's floor: the lowest point the body reaches, in metres.
///
/// The reference (`v2m_inference`) measures this on **frame 0 only**. That only
/// works when frame 0 is a planted pose; a generated clip whose first frame has
/// the legs tucked would then sink the rest of the clip through the floor. The
/// minimum over the first [`ANCHOR_FRAMES`] frames is the same quantity without
/// that failure mode, so it is what the export uses.
fn floor_offset(body: &BodyModel, poses: &[Pose]) -> (f32, bool) {
    let lowest = |values: &[f32]| -> f32 {
        values
            .chunks_exact(3)
            .map(|point| point[1])
            .fold(f32::INFINITY, f32::min)
    };
    let frames = poses.len().min(ANCHOR_FRAMES).max(1);
    if body.has_posedirs() {
        let mut low = f32::INFINITY;
        for pose in poses.iter().take(frames) {
            if let Ok((_, vertices)) = body.forward_mesh(pose) {
                low = low.min(lowest(&vertices));
            } else {
                low = f32::INFINITY;
                break;
            }
        }
        if low.is_finite() {
            return (low, true);
        }
    }
    let mut low = f32::INFINITY;
    for pose in poses.iter().take(frames) {
        low = low.min(lowest(&body.forward_joints(pose).positions));
    }
    (if low.is_finite() { low } else { 0.0 }, false)
}

/// `flow_placement`: median X/Z offset that puts FlowHMR on the SAM anchors.
///
/// The anchor is the SAM MHR hips in camera coordinates; rotating it by the
/// transpose of VGGT's camera-from-world gives it in FlowHMR's y-up world. The
/// difference to FlowHMR's own pelvis is taken on X and Z only — FlowHMR's
/// height is kept, exactly as the reference does.
fn placement(
    frames: &[usize],
    track: &PersonTrack,
    camera: &CameraTrack,
    skeletons: &[Skeleton],
) -> Result<([f32; 3], usize)> {
    let up = camera.to_y_up();
    let mut differences: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut anchors = 0usize;
    for (slot, index) in frames.iter().enumerate().take(ANCHOR_FRAMES) {
        let Some(anchor) = track
            .frames
            .get(*index)
            .and_then(|frame| frame.as_ref())
            .map(|frame| frame.anchor_cam)
        else {
            continue;
        };
        let Some(extrinsic) = up.get(*index) else {
            continue;
        };
        let pelvis = skeletons[slot].joint(0);
        for axis in 0..3 {
            // (R^T · anchor)[axis] = sum_k R[k][axis] · anchor[k]
            let mut rotated = 0.0f32;
            for k in 0..3 {
                rotated += extrinsic[k][axis] * anchor[k];
            }
            differences[axis].push(rotated - pelvis[axis]);
        }
        anchors += 1;
    }
    if anchors == 0 {
        return Ok(([0.0; 3], 0));
    }
    let median = |values: &mut Vec<f32>| -> f32 {
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = values.len() / 2;
        if values.len() % 2 == 0 {
            0.5 * (values[mid - 1] + values[mid])
        } else {
            values[mid]
        }
    };
    let offset = [
        median(&mut differences[0]),
        0.0,
        median(&mut differences[2]),
    ];
    Ok((offset, anchors))
}

/// Everything the BVH writer needs, without the body model.
///
/// The application drops the 76 MB `posedirs` table once the clip is baked, so
/// the export payload has to be self-contained: rest pose, hierarchy, root track
/// and per-frame local rotations.
#[derive(Clone)]
pub struct PreparedBvh {
    pub names: Vec<String>,
    pub parents: Vec<i32>,
    pub rest: Vec<f64>,
    pub roots: Vec<f64>,
    pub local: Vec<f64>,
    pub fps: f64,
}

impl PreparedBvh {
    pub fn frames(&self) -> usize {
        let bones = self.parents.len().max(1);
        self.local.len() / (bones * 9)
    }
}

/// Write a prepared payload.
pub fn write_prepared(path: &Path, prepared: &PreparedBvh) -> Result<BvhReport> {
    let joints = prepared.parents.len();
    let mut basis = vec![0.0f64; joints * 9];
    for joint in 0..joints {
        basis[joint * 9] = 1.0;
        basis[joint * 9 + 4] = 1.0;
        basis[joint * 9 + 8] = 1.0;
    }
    let binding = Binding {
        names: prepared.names.clone(),
        parents: prepared.parents.clone(),
        rest: prepared.rest.clone(),
        ends: prepared.rest.clone(),
        basis,
        origin: [0.0, 0.0, 0.0],
        version: 1,
    };
    binding.validate().context("validating the SMPL-H skeleton")?;
    // The reference re-encodes every bone as `Zrotation Xrotation Yrotation`
    // (`mocap/biped_bvh.py`, "uniform ZXY export encoding"): DCCs and engines
    // that retarget a biped expect the same order on every joint, even where a
    // different order would have a wider singularity margin.
    let orders = vec!["ZXY".to_string(); joints];
    write_bvh_with_orders(
        path,
        &binding,
        &prepared.roots,
        &prepared.local,
        prepared.fps,
        1,
        Some(&orders),
    )
    .with_context(|| format!("writing {}", path.display()))
}

/// The name a SMPL-H joint gets in an exported BVH.
///
/// The reference exports a Mixamo biped and then renames it to Character Studio
/// names (`mocap/biped_bvh.py`): `Hips`, `Chest`, `Chest2`, `Chest3`, `Neck`,
/// `Head`, `LeftHip/Knee/Ankle/Toe`, `LeftCollar/Shoulder/Elbow/Wrist` and
/// `LeftFinger0..4` with `""`/`1`/`2` segments. SMPL-H carries the same
/// anatomy in its own order, so the same target names are derived from its
/// joint names rather than from a Mixamo rig.
pub(crate) fn biped_name(joint: usize) -> String {
    // SMPL-H has 52 joints: the 22 body joints plus 15 per hand, so the name
    // list is the SMPL-X one without the jaw and the two eyes.
    let sourced: Vec<&str> = velomocap_body::JOINT_NAMES
        .iter()
        .copied()
        .filter(|name| !matches!(*name, "jaw" | "left_eye_smplhf" | "right_eye_smplhf"))
        .collect();
    let name = sourced.get(joint).copied().unwrap_or("unknown");
    let mapped = match name {
        "pelvis" => "Hips",
        "spine1" => "Chest",
        "spine2" => "Chest2",
        "spine3" => "Chest3",
        "neck" => "Neck",
        "head" => "Head",
        "left_hip" => "LeftHip",
        "right_hip" => "RightHip",
        "left_knee" => "LeftKnee",
        "right_knee" => "RightKnee",
        "left_ankle" => "LeftAnkle",
        "right_ankle" => "RightAnkle",
        "left_foot" => "LeftToe",
        "right_foot" => "RightToe",
        "left_collar" => "LeftCollar",
        "right_collar" => "RightCollar",
        "left_shoulder" => "LeftShoulder",
        "right_shoulder" => "RightShoulder",
        "left_elbow" => "LeftElbow",
        "right_elbow" => "RightElbow",
        "left_wrist" => "LeftWrist",
        "right_wrist" => "RightWrist",
        other => return finger_name(other).unwrap_or_else(|| other.to_string()),
    };
    mapped.to_string()
}

/// `left_index2` -> `LeftFinger11`: Character Studio counts digits
/// thumb, index, middle, ring, pinky as 0..4 and segments as `""`, `1`, `2`.
fn finger_name(name: &str) -> Option<String> {
    let (side, rest) = if let Some(rest) = name.strip_prefix("left_") {
        ("Left", rest)
    } else if let Some(rest) = name.strip_prefix("right_") {
        ("Right", rest)
    } else {
        return None;
    };
    let (finger, digit) = if let Some(rest) = rest.strip_prefix("thumb") {
        (0, rest)
    } else if let Some(rest) = rest.strip_prefix("index") {
        (1, rest)
    } else if let Some(rest) = rest.strip_prefix("middle") {
        (2, rest)
    } else if let Some(rest) = rest.strip_prefix("ring") {
        (3, rest)
    } else if let Some(rest) = rest.strip_prefix("pinky") {
        (4, rest)
    } else {
        return None;
    };
    let segment: usize = digit.parse().ok()?;
    if !(1..=3).contains(&segment) {
        return None;
    }
    let suffix = match segment {
        1 => String::new(),
        2 => "1".to_string(),
        _ => "2".to_string(),
    };
    Some(format!("{side}Finger{digit}{suffix}", digit = finger))
}

/// Build the export payload for a clip.
pub fn prepare_bvh(clip: &FlowClip, body: &BodyModel, fps: f64) -> Result<PreparedBvh> {
    anyhow::ensure!(!clip.skeletons.is_empty(), "nothing to export");
    let joints = body.num_joints;
    anyhow::ensure!(joints > 1, "the SMPL-H model has no joints");

    // Rest pose: the neutral body with the clip's mean shape.
    let rest = body.rest_joints(&clip.betas);
    let mut names = vec!["Hips".to_string()];
    for joint in 1..joints {
        names.push(biped_name(joint));
    }

    let count = clip.skeletons.len();
    let mut roots = Vec::with_capacity(count * 3);
    let mut local = vec![0.0f64; count * joints * 9];
    for (index, skeleton) in clip.skeletons.iter().enumerate() {
        let hips = skeleton.joint(0);
        roots.extend_from_slice(&[hips[0] as f64, hips[1] as f64, hips[2] as f64]);
        for joint in 0..joints {
            let world = &skeleton.rotations[joint * 9..joint * 9 + 9];
            let matrix: [f64; 9];
            match body.parents[joint] {
                parent if parent < 0 => {
                    matrix = std::array::from_fn(|i| world[i] as f64);
                }
                parent => {
                    let p = &skeleton.rotations[parent as usize * 9..parent as usize * 9 + 9];
                    let mut m = [0.0f64; 9];
                    for row in 0..3 {
                        for col in 0..3 {
                            let mut acc = 0.0;
                            for k in 0..3 {
                                acc += p[k * 3 + row] as f64 * world[k * 3 + col] as f64;
                            }
                            m[row * 3 + col] = acc;
                        }
                    }
                    matrix = m;
                }
            }
            let at = (index * joints + joint) * 9;
            local[at..at + 9].copy_from_slice(&matrix);
        }
    }

    Ok(PreparedBvh {
        names,
        parents: body.parents.clone(),
        rest: rest.iter().map(|v| *v as f64).collect(),
        roots,
        local,
        fps,
    })
}

/// Build the export payload for a clip, named for `rig`.
///
/// The plain [`prepare_bvh`] is this with [`Rig::Biped`] and no rest correction,
/// which is the payload every export used before the retarget chain existed.
pub fn prepare_bvh_for(
    clip: &FlowClip,
    body: &BodyModel,
    fps: f64,
    rig: crate::retarget::Rig,
    rest_correct: bool,
) -> Result<PreparedBvh> {
    let mut prepared = prepare_bvh(clip, body, fps)?;
    crate::retarget::rewrite(&mut prepared, rig, rest_correct)?;
    Ok(prepared)
}

/// Write a generated clip as FBX using the SMPL-H hierarchy.
///
/// `fbx::write` wants world matrices, rest rotations and display bases; a
/// `PreparedBvh` carries parent-relative rotations, so the world transform is
/// composed here. SMPL's rest pose has no per-joint rotation, which makes the
/// rest frames and the display basis the identity and the rest offsets plain
/// differences of `rest_joints`.
pub fn write_prepared_fbx(
    path: &Path,
    prepared: &PreparedBvh,
) -> Result<velomocap_core::fbx::FbxReport> {
    let bones = prepared.parents.len();
    anyhow::ensure!(bones > 0, "no bones to export");
    anyhow::ensure!(
        prepared.local.len() == prepared.roots.len() / 3 * bones * 9,
        "the prepared clip is inconsistent"
    );
    let frames = prepared.roots.len() / 3;

    let mut rest_frames = vec![0.0f64; bones * 9];
    for joint in 0..bones {
        rest_frames[joint * 9] = 1.0;
        rest_frames[joint * 9 + 4] = 1.0;
        rest_frames[joint * 9 + 8] = 1.0;
    }
    let display = rest_frames.clone();
    let tip_local = vec![0.0f64; bones * 3];

    // `frames[t][j]` is a row-major 4x4: world rotation in the 3x3 block and the
    // joint's world position in the fourth column.
    let mut world = vec![0.0f64; frames * bones * 16];
    let mut offset = [0.0f64; 3];
    for frame in 0..frames {
        for joint in 0..bones {
            let at = (frame * bones + joint) * 16;
            let local = &prepared.local[joint * 9..joint * 9 + 9];
            let parent = prepared.parents[joint];
            let (rotation, position);
            if parent < 0 {
                rotation = local.to_vec();
                position = [
                    prepared.roots[frame * 3],
                    prepared.roots[frame * 3 + 1],
                    prepared.roots[frame * 3 + 2],
                ];
            } else {
                let parent = parent as usize;
                let parent_at = (frame * bones + parent) * 16;
                // The parent's 4x4 is laid out with a stride of four, so the
                // rotation has to be pulled out element by element.
                let p = [
                    world[parent_at],
                    world[parent_at + 1],
                    world[parent_at + 2],
                    world[parent_at + 4],
                    world[parent_at + 5],
                    world[parent_at + 6],
                    world[parent_at + 8],
                    world[parent_at + 9],
                    world[parent_at + 10],
                ];
                // world = parent * local. This used to read the parent
                // transposed, which is only harmless while the parent is close
                // to the rest orientation - one rotated spine segment and every
                // child rotation below it came out wrong in the FBX.
                let mut m = [0.0f64; 9];
                for row in 0..3 {
                    for column in 0..3 {
                        let mut acc = 0.0;
                        for k in 0..3 {
                            acc += p[row * 3 + k] * local[k * 3 + column];
                        }
                        m[row * 3 + column] = acc;
                    }
                }
                rotation = m.to_vec();
                // Rest offset of this joint in the parent's frame.
                for axis in 0..3 {
                    let delta = prepared.rest[joint * 3 + axis]
                        - prepared.rest[parent * 3 + axis];
                    offset[axis] = delta;
                }
                let delta = offset;
                position = [
                    world[parent_at + 3]
                        + p[0] * delta[0] + p[1] * delta[1] + p[2] * delta[2],
                    world[parent_at + 7]
                        + p[3] * delta[0] + p[4] * delta[1] + p[5] * delta[2],
                    world[parent_at + 11]
                        + p[6] * delta[0] + p[7] * delta[1] + p[8] * delta[2],
                ];
            }
            world[at..at + 3].copy_from_slice(&rotation[0..3]);
            world[at + 4..at + 7].copy_from_slice(&rotation[3..6]);
            world[at + 8..at + 11].copy_from_slice(&rotation[6..9]);
            world[at + 3] = position[0];
            world[at + 7] = position[1];
            world[at + 11] = position[2];
            world[at + 12] = 0.0;
            world[at + 13] = 0.0;
            world[at + 14] = 0.0;
            world[at + 15] = 1.0;
        }
    }

    // FBX is a centimetre format and the file declares `UnitScaleFactor = 100`,
    // which is what Blender, Unity and Maya expect. SMPL-H works in metres, so
    // every length is scaled here; writing metres next to a cm header is how a
    // "correct" export ends up 100x too small in the importer.
    const CM: f64 = 100.0;
    let rest_cm: Vec<f64> = prepared.rest.iter().map(|value| value * CM).collect();
    let tip_cm: Vec<f64> = tip_local.iter().map(|value| value * CM).collect();
    let mut world_cm = world;
    for index in 0..frames {
        for joint in 0..bones {
            let at = (index * bones + joint) * 16;
            world_cm[at + 3] *= CM;
            world_cm[at + 7] *= CM;
            world_cm[at + 11] *= CM;
        }
    }

    let input = velomocap_core::fbx::FbxInput {
        names: &prepared.names,
        parents: &prepared.parents,
        rest: &rest_cm,
        rest_frames: &rest_frames,
        display: &display,
        tip_local: &tip_cm,
        frames: &world_cm,
        fps: prepared.fps,
        transform: velomocap_core::fbx::IDENTITY4,
        prepend_tpose: false,
        expected_bones: bones,
        rig_name: "SourceAnimation",
        creator: "VeloMocap",
    };
    velomocap_core::fbx::write(path, &input)
        .with_context(|| format!("writing {}", path.display()))
}

/// Write a generated clip as BVH using the SMPL-H hierarchy.
pub fn write_clip_bvh(
    path: &Path,
    clip: &FlowClip,
    body: &BodyModel,
    fps: f64,
) -> Result<BvhReport> {
    let prepared = prepare_bvh(clip, body, fps)?;
    write_prepared(path, &prepared)
}

/// Summary of the generated motion: what the numbers actually look like.
pub fn summarise(clip: &FlowClip) -> Vec<String> {
    let track = clip.root_track();
    let count = track.len();
    let mut lines = Vec::new();
    if count == 0 {
        return lines;
    }
    let travel: f32 = (1..count)
        .map(|i| {
            let (a, b) = (track[i], track[i - 1]);
            ((b[0] - a[0]).powi(2) + (b[2] - a[2]).powi(2)).sqrt()
        })
        .sum();
    let height: f32 = track.iter().map(|p| p[1]).sum::<f32>() / count as f32;
    let min_height = track
        .iter()
        .map(|p| p[1])
        .fold(f32::INFINITY, f32::min);
    // Largest deviation of any joint from the rest orientation. A sane clip
    // stays well under 180 degrees; a joint that reads near 180 means the pose
    // itself is broken (which shows up as a shredded mesh, not as a render bug).
    let mut worst_angle = 0.0f64;
    let mut worst_at = (0usize, 0usize);
    for (frame, skeleton) in clip.skeletons.iter().enumerate() {
        for joint in 0..skeleton.rotations.len() / 9 {
            let m = &skeleton.rotations[joint * 9..joint * 9 + 9];
            let trace = (m[0] + m[4] + m[8]) as f64;
            let angle = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0).acos().to_degrees();
            if angle > worst_angle {
                worst_angle = angle;
                worst_at = (frame, joint);
            }
        }
    }
    lines.push(format!(
        "largest joint rotation {:.0}° (frame {}, joint {})",
        worst_angle, worst_at.0, worst_at.1
    ));

    lines.push(format!(
        "{count} frames, root travel {:.2} m, mean height {:.2} m, min {:.2} m",
        travel, height, min_height
    ));
    lines.push(format!(
        "placement offset x {:+.3} m z {:+.3} m from {}/{} anchors, shapes {:.3}",
        clip.offset[0],
        clip.offset[2],
        clip.anchors,
        ANCHOR_FRAMES.min(count),
        clip.betas.first().copied().unwrap_or(0.0)
    ));
    lines.push(format!(
        "floor {:.3} m from the {} of the first {} frames",
        clip.floor,
        if clip.floor_from_mesh { "mesh" } else { "joints" },
        ANCHOR_FRAMES.min(count)
    ));
    lines.push(format!(
        "anchor residual rms x {:.3} z {:.3} m (y {:.3} m is FlowHMR's own height, by design)",
        clip.anchor_residual[0], clip.anchor_residual[2], clip.anchor_residual[1]
    ));
    let contacts: f32 = clip
        .foot_contacts
        .iter()
        .map(|c| c.iter().cloned().fold(0.0f32, f32::max))
        .sum();
    lines.push(format!(
        "foot contact probability mean {:.2} over {} frames",
        contacts / count as f32,
        count
    ));
    // The floor check: when the solver says a foot is planted, the lowest joint
    // should be near zero. A floating body shows up here, not in the height.
    let mut lowest = f32::INFINITY;
    let mut ankle = 0.0f64;
    for skeleton in &clip.skeletons {
        for point in skeleton.positions.chunks_exact(3) {
            lowest = lowest.min(point[1]);
            ankle += point[1] as f64;
        }
    }
    // Per-frame lowest joint: shows whether the floor reference (frame 0) is
    // representative of the clip or an outlier.
    let per_frame: Vec<f32> = clip
        .skeletons
        .iter()
        .map(|skeleton| {
            skeleton
                .positions
                .chunks_exact(3)
                .map(|point| point[1])
                .fold(f32::INFINITY, f32::min)
        })
        .collect();
    let first = per_frame.first().copied().unwrap_or(0.0);
    let lowest_frame = per_frame.iter().cloned().fold(f32::INFINITY, f32::min);
    lines.push(format!(
        "lowest joint per frame: first {:+.3} m, clip minimum {:+.3} m",
        first, lowest_frame
    ));
    let joints = clip.skeletons[0].num_joints().max(1);
    lines.push(format!(
        "lowest joint y {:+.3} m, mean joint y {:.3} m",
        lowest,
        ankle / (count * joints) as f64
    ));
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn narrowband_matches_the_reference_config() {
        // narrowband_length 2.0 * 30 frames; v2m/tt used raw (cm disabled).
        assert_eq!(NARROWBAND.mm, 60);
        assert_eq!(NARROWBAND.mc, 3);
        assert_eq!(NARROWBAND.cm, -1);
        assert_eq!(NARROWBAND.cc, 3);
    }

    #[test]
    fn exported_bones_use_the_biped_names() {
        // The reference exports a Mixamo biped and renames it to Character
        // Studio naming; SMPL-H carries the same anatomy, so every joint has to
        // land on one of those names and none may stay a raw SMPL-H index.
        let names: Vec<String> = (0..52).map(biped_name).collect();
        assert_eq!(names[0], "Hips");
        assert_eq!(names[1], "LeftHip");
        assert_eq!(names[2], "RightHip");
        assert_eq!(names[3], "Chest");
        assert_eq!(names[4], "LeftKnee");
        assert_eq!(names[6], "Chest2");
        assert_eq!(names[9], "Chest3");
        assert_eq!(names[10], "LeftToe");
        assert_eq!(names[15], "Head");
        assert_eq!(names[16], "LeftShoulder");
        assert_eq!(names[20], "LeftWrist");
        assert_eq!(names[22], "LeftFinger1");
        assert_eq!(names[23], "LeftFinger11");
        assert_eq!(names[24], "LeftFinger12");
        assert_eq!(names[25], "LeftFinger2");
        assert_eq!(names[37], "RightFinger1");
        assert_eq!(names[51], "RightFinger02");
        let mut unique = names.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), 52, "bone names must be unique: {names:?}");
        assert!(
            names.iter().all(|name| !name.starts_with("smplh")),
            "raw SMPL-H names leaked: {names:?}"
        );
    }

    #[test]
    fn median_handles_even_and_odd_counts() {
        let plain = |values: &mut Vec<f32>| -> f32 {
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mid = values.len() / 2;
            if values.len() % 2 == 0 {
                0.5 * (values[mid - 1] + values[mid])
            } else {
                values[mid]
            }
        };
        assert_eq!(plain(&mut vec![3.0, 1.0, 2.0]), 2.0);
        assert_eq!(plain(&mut vec![4.0, 1.0]), 2.5);
    }
}
