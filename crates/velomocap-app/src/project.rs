//! The `.vp` project: the whole workspace in one file.
//!
//! A take is not just a video path. It is the tracked bodies, the fused mesh, the
//! baked skeleton with its contacts, and the hand-drawn locks. Saving only paths
//! would make "open project" mean "run the reconstruction again" — minutes of GPU
//! work for something that was already on screen when the file was written.
//!
//! So the file is a small JSON document beside one compressed block of bulk
//! numbers: JSON keeps it inspectable and lets an older build ignore what it does
//! not know, and the block keeps a 30-frame take at a few megabytes instead of
//! hundreds. Bulk arrays are addressed by `Slice` offsets into that block.
//!
//! Layout:
//!
//! ```text
//! magic "VMPROJV1"      8
//! json length           u32
//! single zlib length    u32
//! double zlib length    u32
//! single count          u64   (f32 elements after inflate)
//! double count          u64   (f64 elements after inflate)
//! json bytes
//! zlib(single f32 block)
//! zlib(double f64 block)
//! ```

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::app::{Candidate, FlowBody, PersonView, VeloMocap};

pub const EXTENSION: &str = "vp";
const MAGIC: &[u8; 8] = b"VMPROJV1";
const VERSION: u32 = 1;
const HEADER: usize = 8 + 4 + 4 + 4 + 8 + 8;

/// How precisely geometry is written into a project.
///
/// The baked skeleton is always exact, so none of these change an export; they
/// trade the viewport mesh's last digits for file size. `High` is the default:
/// a tenth of a millimetre is finer than a tracked body has, and the zeroed low
/// byte is what makes the bulk of the file compress.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Precision {
    /// Every float exactly as it came out of the pipeline.
    Ultra,
    /// Low mantissa byte dropped (about 0.1 mm).
    High,
    /// Positions rounded to 0.25 mm.
    Medium,
    /// Positions rounded to 16 mm, for a throwaway preview.
    Low,
}

impl Default for Precision {
    fn default() -> Self {
        Precision::High
    }
}

impl Precision {
    pub const ALL: [Precision; 4] = [
        Precision::Ultra,
        Precision::High,
        Precision::Medium,
        Precision::Low,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Precision::Ultra => "Ultra (lossless)",
            Precision::High => "High (default)",
            Precision::Medium => "Medium",
            Precision::Low => "Low",
        }
    }

    fn quantize(self, value: f32) -> f32 {
        match self {
            Precision::Ultra => value,
            Precision::High => f32::from_bits(value.to_bits() & 0xffff_ff00),
            Precision::Medium => (value * 4096.0).round() / 4096.0,
            Precision::Low => (value * 64.0).round() / 64.0,
        }
    }
}

/// The deflate level a project is written with. `None` still uses a zlib
/// container so the reader has one path; it just stores the bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Compression {
    None,
    Fast,
    Balanced,
    Smallest,
}

impl Default for Compression {
    fn default() -> Self {
        // Measured on a 30-frame take, fast is the smallest of the four at the
        // default precision: the geometry is 24-bit-limited, so the slower
        // levels only spend time finding nothing (8.2 MB at level 1 and 8.2 MB
        // at level 9 before the precision options existed).
        Compression::Fast
    }
}

impl Compression {
    pub const ALL: [Compression; 4] = [
        Compression::None,
        Compression::Fast,
        Compression::Balanced,
        Compression::Smallest,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Compression::None => "None",
            Compression::Fast => "Fast",
            Compression::Balanced => "Balanced",
            Compression::Smallest => "Smallest",
        }
    }

    fn level(self) -> flate2::Compression {
        match self {
            Compression::None => flate2::Compression::none(),
            Compression::Fast => flate2::Compression::fast(),
            Compression::Balanced => flate2::Compression::default(),
            Compression::Smallest => flate2::Compression::best(),
        }
    }
}

/// Where one bulk array lives in the file's numeric block.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Slice {
    pub offset: usize,
    pub count: usize,
}

/// Bulk numbers for one save: f32 for geometry, f64 for the baked skeleton, whose
/// precision an export should not quietly lose.
#[derive(Default)]
pub struct Block {
    single: Vec<f32>,
    double: Vec<f64>,
    precision: Precision,
}

impl Block {
    fn new(precision: Precision) -> Block {
        Block {
            precision,
            ..Block::default()
        }
    }

    /// Append geometry at the chosen precision.
    fn f32s(&mut self, values: &[f32]) -> Slice {
        let offset = self.single.len();
        let precision = self.precision;
        self.single
            .extend(values.iter().map(|value| precision.quantize(*value)));
        Slice {
            offset,
            count: values.len(),
        }
    }

    fn f64s(&mut self, values: &[f64]) -> Slice {
        let offset = self.double.len();
        self.double.extend_from_slice(values);
        Slice {
            offset,
            count: values.len(),
        }
    }
}

/// Read side of [`Block`].
pub struct Viewed<'a> {
    single: &'a [f32],
    double: &'a [f64],
}

impl Viewed<'_> {
    pub fn f32s(&self, slice: &Slice) -> Vec<f32> {
        let end = slice.offset.saturating_add(slice.count).min(self.single.len());
        self.single
            .get(slice.offset..end)
            .map(<[f32]>::to_vec)
            .unwrap_or_default()
    }

    pub fn f64s(&self, slice: &Slice) -> Vec<f64> {
        let end = slice.offset.saturating_add(slice.count).min(self.double.len());
        self.double
            .get(slice.offset..end)
            .map(<[f64]>::to_vec)
            .unwrap_or_default()
    }
}

/// The workspace as it goes into the file.
#[derive(Serialize, Deserialize)]
pub struct Document {
    pub version: u32,
    pub video: Option<VideoState>,
    pub people: Vec<PersonState>,
    /// Every character imported into the workspace, worn or not.
    #[serde(default)]
    pub characters: Vec<PathBuf>,
    pub flow: Vec<ClipState>,
    pub candidates: Vec<CandidateState>,
    pub locks: velomocap_motion::ContactTracks,
    pub settings: SettingsState,
    pub view: ViewState,
}

#[derive(Serialize, Deserialize)]
pub struct VideoState {
    pub path: PathBuf,
    pub fps: f64,
    pub frames: usize,
    pub width: usize,
    pub height: usize,
    pub capture_fps: f64,
    pub floor: f32,
}

#[derive(Serialize, Deserialize)]
pub struct PersonState {
    pub id: usize,
    pub name: String,
    /// Premultiplied RGBA, the way `Color32` stores it.
    pub colour: [u8; 4],
    pub visible: bool,
    /// The avatar that replaces this person's built-in body.
    pub character: Option<PathBuf>,
    pub frames: Vec<Option<FrameState>>,
}

/// One tracked body in one frame. The body tokens are left out: only another
/// inference run reads them, and a run re-detects anyway.
#[derive(Serialize, Deserialize)]
pub struct FrameState {
    pub bbox: [f32; 4],
    pub score: f32,
    pub cam_t: [f32; 3],
    pub anchor_cam: [f32; 3],
    pub verts: Slice,
    pub joints: Slice,
    pub keypoints: Slice,
    pub skel_state: Slice,
}

#[derive(Serialize, Deserialize)]
pub struct ClipState {
    pub name: String,
    pub first_frame: usize,
    pub summary: String,
    /// Vertices per baked frame; every frame of one clip is the same mesh.
    pub frame_len: usize,
    pub frame_count: usize,
    pub frames: Slice,
    pub joints: usize,
    pub skeleton_count: usize,
    pub positions: Slice,
    pub rotations: Slice,
    pub bvh: BvhState,
    pub contacts: [Vec<[usize; 2]>; 4],
}

#[derive(Serialize, Deserialize)]
pub struct BvhState {
    pub names: Vec<String>,
    pub parents: Vec<i32>,
    pub fps: f64,
    pub rest: Slice,
    pub roots: Slice,
    pub local: Slice,
}

#[derive(Serialize, Deserialize)]
pub struct CandidateState {
    pub bbox: [f32; 4],
    pub norm: [f32; 4],
    pub score: f32,
    pub selected: bool,
}

/// The default for the flags a project written before they existed does not
/// have: the asset export writes everything.
fn yes() -> bool {
    true
}

#[derive(Serialize, Deserialize)]
pub struct SettingsState {
    pub correction: velomocap_motion::correction::CorrectionSettings,
    pub estimate_camera: bool,
    pub refine_camera: bool,
    pub confidence: f32,
    pub use_flow: bool,
    /// `unlimited`, `scheduled`, `hybrid` or `cpu`.
    pub memory_mode: String,
    pub scale_width: usize,
    pub max_frames: usize,
    #[serde(default = "yes")]
    pub asset_skin: bool,
    #[serde(default = "yes")]
    pub asset_materials: bool,
    #[serde(default = "yes")]
    pub asset_animation: bool,
    /// Where the assets land; empty means beside the project.
    #[serde(default)]
    pub asset_folder: String,
    /// How the next save writes geometry. High by default: fine enough to be
    /// invisible, coarse enough to compress.
    #[serde(default)]
    pub storage_precision: Precision,
    #[serde(default)]
    pub storage_compression: Compression,
}

#[derive(Serialize, Deserialize)]
pub struct ViewState {
    pub selected: usize,
    pub playhead: usize,
    pub play_speed: f32,
    /// Index into the view-mode list.
    pub mode: usize,
    pub layers: LayersState,
    pub orbit: OrbitState,
}

#[derive(Serialize, Deserialize)]
pub struct LayersState {
    pub bones: bool,
    pub grid: bool,
    pub motion_path: bool,
    pub names: bool,
    pub follow: bool,
    pub overlays: bool,
}

#[derive(Serialize, Deserialize)]
pub struct OrbitState {
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub target: [f32; 3],
    pub framed: bool,
}

/// A document plus the numbers it points into.
pub struct Project {
    pub document: Document,
    single: Vec<f32>,
    double: Vec<f64>,
    compression: Compression,
}

impl Project {
    /// Read the whole workspace off the running app.
    pub fn capture(app: &VeloMocap) -> Project {
        let mut block = Block::new(app.storage_precision);
        let document = Document {
            version: VERSION,
            video: app.video.as_ref().map(|path| VideoState {
                path: path.clone(),
                fps: app.video_fps,
                frames: app.video_frames,
                width: app.video_size.0,
                height: app.video_size.1,
                capture_fps: app.capture_fps,
                floor: app.floor,
            }),
            people: app
                .people
                .iter()
                .map(|person| PersonState {
                    id: person.id,
                    name: person.name.clone(),
                    colour: person.colour.to_array(),
                    visible: person.visible,
                    character: person
                        .character
                        .as_ref()
                        .map(|assigned| assigned.path.clone()),
                    frames: person
                        .frames
                        .iter()
                        .map(|frame| {
                            frame.as_ref().map(|frame| FrameState {
                                bbox: frame.bbox,
                                score: frame.score,
                                cam_t: frame.cam_t,
                                anchor_cam: frame.anchor_cam,
                                verts: block.f32s(&frame.verts),
                                joints: block.f32s(&frame.joints),
                                keypoints: block.f32s(&frame.keypoints),
                                skel_state: block.f32s(&frame.skel_state),
                            })
                        })
                        .collect(),
                })
                .collect(),
            characters: app
                .characters
                .iter()
                .map(|entry| entry.path.clone())
                .collect(),
            flow: app
                .flow
                .iter()
                .map(|clip| {
                    let frame_len = clip.frames.first().map(|frame| frame.len()).unwrap_or(0);
                    let mut frames = Vec::with_capacity(clip.frames.len() * frame_len);
                    for frame in clip.frames.iter() {
                        frames.extend_from_slice(frame);
                    }
                    let joints = clip
                        .skeletons
                        .first()
                        .map(|skeleton| skeleton.positions.len() / 3)
                        .unwrap_or(0);
                    let mut positions = Vec::new();
                    let mut rotations = Vec::new();
                    for skeleton in clip.skeletons.iter() {
                        positions.extend_from_slice(&skeleton.positions);
                        rotations.extend_from_slice(&skeleton.rotations);
                    }
                    ClipState {
                        name: clip.name.clone(),
                        first_frame: clip.first_frame,
                        summary: clip.summary.clone(),
                        frame_len,
                        frame_count: clip.frames.len(),
                        frames: block.f32s(&frames),
                        joints,
                        skeleton_count: clip.skeletons.len(),
                        positions: block.f32s(&positions),
                        rotations: block.f32s(&rotations),
                        bvh: BvhState {
                            names: clip.bvh.names.clone(),
                            parents: clip.bvh.parents.clone(),
                            fps: clip.bvh.fps,
                            rest: block.f64s(&clip.bvh.rest),
                            roots: block.f64s(&clip.bvh.roots),
                            local: block.f64s(&clip.bvh.local),
                        },
                        contacts: clip.contacts.clone(),
                    }
                })
                .collect(),
            candidates: app
                .candidates
                .iter()
                .map(|candidate| CandidateState {
                    bbox: candidate.bbox,
                    norm: candidate.norm,
                    score: candidate.score,
                    selected: candidate.selected,
                })
                .collect(),
            locks: app.locks.clone(),
            settings: SettingsState {
                correction: app.correction,
                estimate_camera: app.estimate_camera,
                refine_camera: app.refine_camera,
                confidence: app.confidence,
                use_flow: app.use_flow,
                memory_mode: memory_mode_slug(app.memory_mode).to_string(),
                scale_width: app.scale_width,
                max_frames: app.max_frames,
                asset_skin: app.asset_skin,
                asset_materials: app.asset_materials,
                asset_animation: app.asset_animation,
                asset_folder: app.asset_folder.clone(),
                storage_precision: app.storage_precision,
                storage_compression: app.storage_compression,
            },
            view: ViewState {
                selected: app.selected,
                playhead: app.playhead,
                play_speed: app.play_speed,
                mode: app.view_mode,
                layers: LayersState {
                    bones: app.layers.bones,
                    grid: app.layers.grid,
                    motion_path: app.layers.motion_path,
                    names: app.layers.names,
                    follow: app.layers.follow,
                    overlays: app.layers.overlays,
                },
                orbit: OrbitState {
                    yaw: app.orbit.yaw,
                    pitch: app.orbit.pitch,
                    distance: app.orbit.distance,
                    target: app.orbit.target,
                    framed: app.orbit.framed,
                },
            },
        };
        Project {
            document,
            single: block.single,
            double: block.double,
            compression: app.storage_compression,
        }
    }

    /// Write the file and report how many bytes it took.
    pub fn save(&self, path: &Path) -> Result<u64> {
        let json = serde_json::to_vec(&self.document).context("encoding the project")?;
        let single: Vec<u8> = self
            .single
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let double: Vec<u8> = self
            .double
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let single = deflate(&single, self.compression)?;
        let double = deflate(&double, self.compression)?;
        let mut header = Vec::with_capacity(HEADER);
        header.extend_from_slice(MAGIC);
        header.extend_from_slice(&(json.len() as u32).to_le_bytes());
        header.extend_from_slice(&(single.len() as u32).to_le_bytes());
        header.extend_from_slice(&(double.len() as u32).to_le_bytes());
        header.extend_from_slice(&(self.single.len() as u64).to_le_bytes());
        header.extend_from_slice(&(self.double.len() as u64).to_le_bytes());
        let mut file = std::fs::File::create(path)
            .with_context(|| format!("creating {}", path.display()))?;
        for part in [&header[..], &json, &single, &double] {
            file.write_all(part)?;
        }
        file.flush()?;
        Ok((header.len() + json.len() + single.len() + double.len()) as u64)
    }

    pub fn load(path: &Path) -> Result<Project> {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        if bytes.len() < HEADER || &bytes[..8] != MAGIC {
            bail!("{} is not a VeloMocap project", path.display());
        }
        let word = |at: usize| -> u32 {
            let mut raw = [0u8; 4];
            raw.copy_from_slice(&bytes[at..at + 4]);
            u32::from_le_bytes(raw)
        };
        let count = |at: usize| -> usize {
            let mut raw = [0u8; 8];
            raw.copy_from_slice(&bytes[at..at + 8]);
            u64::from_le_bytes(raw) as usize
        };
        let (json_len, single_len, double_len) = (word(8) as usize, word(12) as usize, word(16) as usize);
        let (single_count, double_count) = (count(20), count(28));
        let end = HEADER + json_len + single_len + double_len;
        if end > bytes.len() {
            bail!("{} is truncated", path.display());
        }
        let json = &bytes[HEADER..HEADER + json_len];
        let single_bytes = inflate(&bytes[HEADER + json_len..HEADER + json_len + single_len])?;
        let double_bytes =
            inflate(&bytes[HEADER + json_len + single_len..end])?;
        let single = numbers_f32(&single_bytes, single_count);
        let double = numbers_f64(&double_bytes, double_count);
        let document: Document =
            serde_json::from_slice(json).context("reading the project document")?;
        if document.version > VERSION {
            bail!(
                "{} was written by a newer VeloMocap (project format {})",
                path.display(),
                document.version
            );
        }
        Ok(Project {
            compression: document.settings.storage_compression,
            document,
            single,
            double,
        })
    }

    /// The numbers a document entry points into.
    pub fn view(&self) -> Viewed<'_> {
        Viewed {
            single: &self.single,
            double: &self.double,
        }
    }
}

fn deflate(bytes: &[u8], compression: Compression) -> Result<Vec<u8>> {
    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), compression.level());
    encoder.write_all(bytes)?;
    Ok(encoder.finish()?)
}

fn inflate(bytes: &[u8]) -> Result<Vec<u8>> {
    let mut decoder = flate2::read::ZlibDecoder::new(bytes);
    let mut out = Vec::new();
    decoder.read_to_end(&mut out)?;
    Ok(out)
}

fn numbers_f32(bytes: &[u8], count: usize) -> Vec<f32> {
    let width = 4;
    let available = bytes.len() / width;
    (0..count.min(available))
        .map(|at| {
            let mut raw = [0u8; 4];
            raw.copy_from_slice(&bytes[at * width..at * width + width]);
            f32::from_le_bytes(raw)
        })
        .collect()
}

fn numbers_f64(bytes: &[u8], count: usize) -> Vec<f64> {
    let width = 8;
    let available = bytes.len() / width;
    (0..count.min(available))
        .map(|at| {
            let mut raw = [0u8; 8];
            raw.copy_from_slice(&bytes[at * width..at * width + width]);
            f64::from_le_bytes(raw)
        })
        .collect()
}

pub fn memory_mode_slug(mode: crate::capture::MemoryMode) -> &'static str {
    use crate::capture::MemoryMode;
    match mode {
        MemoryMode::Unlimited => "unlimited",
        MemoryMode::Scheduled => "scheduled",
        MemoryMode::Hybrid => "hybrid",
        MemoryMode::Cpu => "cpu",
    }
}

pub fn memory_mode_from_slug(slug: &str) -> Option<crate::capture::MemoryMode> {
    use crate::capture::MemoryMode;
    match slug {
        "unlimited" => Some(MemoryMode::Unlimited),
        "scheduled" => Some(MemoryMode::Scheduled),
        "hybrid" => Some(MemoryMode::Hybrid),
        "cpu" => Some(MemoryMode::Cpu),
        _ => None,
    }
}

/// Rebuild one person's rows, including the avatar that was on it.
pub fn person(state: &PersonState, numbers: &Viewed<'_>) -> PersonView {
    let mut frames = Vec::with_capacity(state.frames.len());
    for frame in &state.frames {
        frames.push(frame.as_ref().map(|frame| {
            Arc::new(velomocap_motion::PersonFrame {
                bbox: frame.bbox,
                score: frame.score,
                verts: numbers.f32s(&frame.verts),
                joints: numbers.f32s(&frame.joints),
                keypoints: numbers.f32s(&frame.keypoints),
                body_token: Vec::new(),
                left_token: Vec::new(),
                right_token: Vec::new(),
                cam_t: frame.cam_t,
                anchor_cam: frame.anchor_cam,
                skel_state: numbers.f32s(&frame.skel_state),
            })
        }));
    }
    let colour = egui::Color32::from_rgba_premultiplied(
        state.colour[0],
        state.colour[1],
        state.colour[2],
        state.colour[3],
    );
    PersonView {
        id: state.id,
        name: state.name.clone(),
        colour,
        visible: state.visible,
        frames,
        character: None,
    }
}

/// Rebuild one baked clip.
pub fn clip(state: &ClipState, numbers: &Viewed<'_>) -> FlowBody {
    let flat = numbers.f32s(&state.frames);
    let frames = if state.frame_len > 0 {
        flat.chunks_exact(state.frame_len)
            .map(<[f32]>::to_vec)
            .collect()
    } else {
        Vec::new()
    };
    let positions = numbers.f32s(&state.positions);
    let rotations = numbers.f32s(&state.rotations);
    let width = state.joints * 3;
    let spin = state.joints * 9;
    let mut skeletons = Vec::with_capacity(state.skeleton_count);
    for index in 0..state.skeleton_count {
        let from = index * width;
        let to = from + width;
        let spin_from = index * spin;
        let spin_to = spin_from + spin;
        skeletons.push(velomocap_body::Skeleton {
            positions: positions.get(from..to).unwrap_or(&[]).to_vec(),
            rotations: rotations.get(spin_from..spin_to).unwrap_or(&[]).to_vec(),
        });
    }
    FlowBody {
        name: state.name.clone(),
        first_frame: state.first_frame,
        frames: Arc::new(frames),
        summary: state.summary.clone(),
        bvh: Arc::new(velomocap_motion::PreparedBvh {
            names: state.bvh.names.clone(),
            parents: state.bvh.parents.clone(),
            rest: numbers.f64s(&state.bvh.rest),
            roots: numbers.f64s(&state.bvh.roots),
            local: numbers.f64s(&state.bvh.local),
            fps: state.bvh.fps,
        }),
        skeletons: Arc::new(skeletons),
        contacts: state.contacts.clone(),
        // A project stores the baked take, not the solver's input: "apply
        // correction" needs a regeneration first (the console says so).
        source: None,
    }
}

pub fn candidate(state: &CandidateState) -> Candidate {
    Candidate {
        bbox: state.bbox,
        norm: state.norm,
        score: state.score,
        selected: state.selected,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "velomocap-project-{}-{name}.vp",
            std::process::id()
        ));
        path
    }

    /// A tiny document with one of everything that carries bulk numbers.
    fn sample() -> Project {
        let mut block = Block::default();
        let verts = block.f32s(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let joints = block.f32s(&[6.0, 7.0, 8.0]);
        let keypoints = block.f32s(&[9.0]);
        let skel_state = block.f32s(&[10.0, 11.0]);
        let frames = block.f32s(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let positions = block.f32s(&[0.0; 156]);
        let rotations = block.f32s(&[1.0; 468]);
        let rest = block.f64s(&[0.5; 6]);
        let roots = block.f64s(&[0.25; 6]);
        let local = block.f64s(&[0.75; 18]);
        let document = Document {
            version: VERSION,
            video: Some(VideoState {
                path: PathBuf::from(r"C:\clips\take.mp4"),
                fps: 30.0,
                frames: 30,
                width: 514,
                height: 910,
                capture_fps: 30.0,
                floor: 0.0,
            }),
            people: vec![PersonState {
                id: 3,
                name: "Person 1".to_string(),
                colour: [10, 20, 30, 255],
                visible: true,
                character: Some(PathBuf::from(r"C:\models\avatar.glb")),
                frames: vec![
                    Some(FrameState {
                        bbox: [1.0, 2.0, 3.0, 4.0],
                        score: 0.9,
                        cam_t: [0.1, 0.2, 0.3],
                        anchor_cam: [0.4, 0.5, 0.6],
                        verts,
                        joints,
                        keypoints,
                        skel_state,
                    }),
                    None,
                ],
            }],
            characters: vec![PathBuf::from(r"C:\models\avatar.glb")],
            flow: vec![ClipState {
                name: "Person 1".to_string(),
                first_frame: 0,
                summary: "30 frames".to_string(),
                frame_len: 3,
                frame_count: 2,
                frames,
                joints: 52,
                skeleton_count: 1,
                positions,
                rotations,
                bvh: BvhState {
                    names: vec!["Hips".to_string(), "Spine".to_string()],
                    parents: vec![-1, 0],
                    fps: 30.0,
                    rest,
                    roots,
                    local,
                },
                contacts: [vec![[0, 4]], vec![], vec![], vec![]],
            }],
            candidates: vec![CandidateState {
                bbox: [0.0; 4],
                norm: [0.0; 4],
                score: 0.5,
                selected: true,
            }],
            locks: velomocap_motion::ContactTracks::automatic(),
            settings: SettingsState {
                correction: velomocap_motion::correction::CorrectionSettings::default(),
                estimate_camera: false,
                refine_camera: true,
                confidence: 0.3,
                use_flow: true,
                memory_mode: "hybrid".to_string(),
                scale_width: 640,
                max_frames: 0,
                asset_skin: true,
                asset_materials: true,
                asset_animation: true,
                asset_folder: String::new(),
                storage_precision: Precision::High,
                storage_compression: Compression::Fast,
            },
            view: ViewState {
                selected: 0,
                playhead: 12,
                play_speed: 1.0,
                mode: 0,
                layers: LayersState {
                    bones: true,
                    grid: true,
                    motion_path: false,
                    names: false,
                    follow: false,
                    overlays: true,
                },
                orbit: OrbitState {
                    yaw: 0.3,
                    pitch: -0.1,
                    distance: 3.5,
                    target: [0.0, 0.9, 0.0],
                    framed: true,
                },
            },
        };
        Project {
            document,
            single: block.single,
            double: block.double,
            compression: Compression::Fast,
        }
    }

    #[test]
    fn a_project_round_trips() {
        let path = scratch("round-trip");
        let project = sample();
        let written = project.save(&path).unwrap();
        assert_eq!(written, std::fs::metadata(&path).unwrap().len());
        let opened = Project::load(&path).unwrap();
        let numbers = opened.view();
        // The document, byte for byte.
        assert_eq!(
            serde_json::to_vec(&project.document).unwrap(),
            serde_json::to_vec(&opened.document).unwrap()
        );
        // And every bulk array the entries point at.
        assert_eq!(project.single, opened.single);
        assert_eq!(project.double, opened.double);
        let state = &opened.document.flow[0];
        assert_eq!(numbers.f32s(&state.frames), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(numbers.f64s(&state.bvh.local).len(), 18);
        let person = &opened.document.people[0];
        let frame = person.frames[0].as_ref().unwrap();
        assert_eq!(numbers.f32s(&frame.verts), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(person.frames[1].is_none());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_rebuilt_clip_keeps_its_mesh_and_skeleton() {
        let path = scratch("clip");
        let project = sample();
        project.save(&path).unwrap();
        let opened = Project::load(&path).unwrap();
        let body = clip(&opened.document.flow[0], &opened.view());
        assert_eq!(body.frames.len(), 2);
        assert_eq!(body.frames[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(body.frames[1], vec![4.0, 5.0, 6.0]);
        assert_eq!(body.skeletons.len(), 1);
        assert_eq!(body.skeletons[0].positions.len(), 156);
        assert_eq!(body.skeletons[0].rotations.len(), 468);
        assert_eq!(body.bvh.parents, vec![-1, 0]);
        assert_eq!(body.bvh.frames(), 1);
        assert_eq!(body.contacts[0], vec![[0, 4]]);
        let person = person(&opened.document.people[0], &opened.view());
        assert_eq!(person.name, "Person 1");
        assert_eq!(person.frames.len(), 2);
        assert!(person.frames[0].is_some());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_broken_file_is_refused_not_read() {
        let path = scratch("broken");
        std::fs::write(&path, b"not a project at all").unwrap();
        assert!(Project::load(&path).is_err());
        // A file that starts right but stops early must not panic.
        let project = sample();
        project.save(&path).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.truncate(HEADER + 4);
        std::fs::write(&path, &bytes).unwrap();
        assert!(Project::load(&path).is_err());
        std::fs::remove_file(&path).ok();
    }
}
