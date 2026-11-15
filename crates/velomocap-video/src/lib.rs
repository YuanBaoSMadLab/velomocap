//! Video decode for VeloMocap.
//!
//! Frames are pulled from the `ffmpeg` binary as raw RGB24 over a pipe. This
//! keeps the build dependency-free (no ffmpeg development headers, no C
//! linkage) while supporting every container and codec ffmpeg does — which is
//! the same set the original application relies on.
//!
//! `ffprobe` supplies metadata (dimensions, frame rate, duration, frame count).

use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

use anyhow::{Context, Result};

/// Where the ffmpeg binaries live.
#[derive(Clone, Debug)]
pub struct Tools {
    pub ffmpeg: PathBuf,
    pub ffprobe: PathBuf,
}

impl Tools {
    /// Locate ffmpeg/ffprobe.
    ///
    /// Order: `$VELOMOCAP_FFMPEG_DIR`, then the bundled copies next to the
    /// executable (`runtime/ffmpeg/`, `ffmpeg/`, or the same folder), then
    /// `PATH`, then the usual install directories. A shipped build therefore
    /// works with no environment variables set.
    pub fn discover() -> Result<Self> {
        if let Ok(dir) = std::env::var("VELOMOCAP_FFMPEG_DIR") {
            let dir = PathBuf::from(dir);
            if let Some(tools) = Self::in_dir(&dir) {
                return Ok(tools);
            }
        }

        let mut roots: Vec<PathBuf> = Vec::new();
        if let Ok(exe_path) = std::env::current_exe() {
            if let Some(folder) = exe_path.parent() {
                roots.push(folder.to_path_buf());
                roots.push(folder.join("runtime/ffmpeg"));
                roots.push(folder.join("ffmpeg"));
                roots.push(folder.join("runtime"));
            }
        }
        roots.push(PathBuf::from("runtime/ffmpeg"));
        roots.push(PathBuf::from("ffmpeg"));
        roots.push(PathBuf::from("."));
        for root in roots {
            if let Some(tools) = Self::in_dir(&root) {
                return Ok(tools);
            }
        }

        if let (Some(ffmpeg), Some(ffprobe)) = (which("ffmpeg"), which("ffprobe")) {
            return Ok(Self { ffmpeg, ffprobe });
        }
        for root in [r"D:\ffmpeg\bin", r"C:\ffmpeg\bin"] {
            if let Some(tools) = Self::in_dir(&PathBuf::from(root)) {
                return Ok(tools);
            }
        }
        anyhow::bail!("ffmpeg/ffprobe not found; set VELOMOCAP_FFMPEG_DIR")
    }

    /// Both binaries present in one folder.
    fn in_dir(dir: &Path) -> Option<Self> {
        let ffmpeg = dir.join(exe("ffmpeg"));
        let ffprobe = dir.join(exe("ffprobe"));
        if ffmpeg.is_file() && ffprobe.is_file() {
            Some(Self { ffmpeg, ffprobe })
        } else {
            None
        }
    }

    pub fn ffmpeg_exists(&self) -> bool {
        self.ffmpeg.is_file()
    }
}

fn exe(name: &str) -> String {
    if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    }
}

fn which(name: &str) -> Option<PathBuf> {
    let target = exe(name);
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(&target))
        .find(|candidate| candidate.is_file())
}

/// Container/stream metadata.
#[derive(Clone, Debug)]
pub struct VideoInfo {
    pub path: PathBuf,
    pub width: usize,
    pub height: usize,
    pub fps: f64,
    pub duration: f64,
    pub frames: usize,
    pub codec: String,
    pub bytes: u64,
}

impl VideoInfo {
    pub fn aspect(&self) -> f32 {
        self.width as f32 / self.height.max(1) as f32
    }
}

/// Probe a video file with ffprobe.
pub fn probe(tools: &Tools, path: impl AsRef<Path>) -> Result<VideoInfo> {
    let path = path.as_ref();
    if !path.is_file() {
        anyhow::bail!("video not found: {}", path.display());
    }
    let output = Command::new(&tools.ffprobe)
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames,codec_name,duration",
            "-show_entries",
            "format=duration,size",
            "-of",
            "json",
        ])
        .arg(path)
        .no_window()
        .output()
        .with_context(|| format!("running {}", tools.ffprobe.display()))?;
    if !output.status.success() {
        anyhow::bail!(
            "ffprobe failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let text = String::from_utf8_lossy(&output.stdout);
    parse_probe(path, &text)
}

/// Parse ffprobe's JSON with a few targeted splits (no JSON dependency needed).
fn parse_probe(path: &Path, text: &str) -> Result<VideoInfo> {
    let width = number_field(text, "width").unwrap_or(0.0) as usize;
    let height = number_field(text, "height").unwrap_or(0.0) as usize;
    if width == 0 || height == 0 {
        anyhow::bail!("{} has no usable video stream", path.display());
    }
    let fps = text
        .split("\"avg_frame_rate\":")
        .nth(1)
        .and_then(|rest| rest.trim_start().split('"').nth(1))
        .and_then(parse_rational)
        .unwrap_or(0.0);
    let duration = text
        .split("\"duration\":")
        .filter_map(|rest| rest.trim_start().split('"').nth(1))
        .find_map(|value| value.parse::<f64>().ok())
        .unwrap_or(0.0);
    let mut frames = text
        .split("\"nb_frames\":")
        .nth(1)
        .and_then(|rest| rest.trim_start().split('"').nth(1))
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    if frames == 0 && fps > 0.0 {
        frames = (duration * fps).round() as usize;
    }
    let codec = text
        .split("\"codec_name\":")
        .nth(1)
        .and_then(|rest| rest.trim_start().split('"').nth(1))
        .unwrap_or("unknown")
        .to_string();
    let bytes = number_field(text, "size").unwrap_or(0.0) as u64;

    Ok(VideoInfo {
        path: path.to_path_buf(),
        width,
        height,
        fps,
        duration,
        frames,
        codec,
        bytes,
    })
}

fn number_field(text: &str, key: &str) -> Option<f64> {
    let needle = format!("\"{key}\":");
    let rest = text.split(&needle).nth(1)?;
    let value = rest
        .trim_start()
        .split([',', '}', '\n'])
        .next()?
        .trim()
        .trim_matches('"');
    value.parse::<f64>().ok()
}

fn parse_rational(text: &str) -> Option<f64> {
    let (num, den) = text.split_once('/')?;
    let num: f64 = num.trim().parse().ok()?;
    let den: f64 = den.trim().parse().ok()?;
    if den == 0.0 {
        None
    } else {
        Some(num / den)
    }
}

/// One decoded frame in RGB24.
#[derive(Clone)]
pub struct Frame {
    pub index: usize,
    pub width: usize,
    pub height: usize,
    /// `width * height * 3` bytes.
    pub rgb: Vec<u8>,
}

impl Frame {
    pub fn byte_len(&self) -> usize {
        self.width * self.height * 3
    }
}

/// Crop + scale plan that feeds VGGT-Omega.
///
/// The reference preprocessing centre-crops extreme aspect ratios into
/// `[0.5, 2.0]`, then scales the longest side to `resolution` while rounding the
/// other side to a multiple of the patch size. Reproducing the plan here lets
/// ffmpeg do the resampling once, instead of decoding full frames and resizing
/// them on the CPU.
///
/// The network graph has a fixed input shape, so a landscape clip is rotated a
/// quarter turn when `portrait_only` is set: a 16:9 frame becomes the same
/// portrait canvas the graph was traced with, and the caller un-rotates the
/// resulting camera. Rotating by exactly 90° is lossless, unlike cropping or
/// squashing the frame to fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VggtGeometry {
    pub crop_x: usize,
    pub crop_y: usize,
    pub crop_w: usize,
    pub crop_h: usize,
    pub out_w: usize,
    pub out_h: usize,
    /// The frame is rotated a quarter turn clockwise before scaling, so that
    /// `out_h > out_w` regardless of the source orientation.
    pub rotated: bool,
}

impl VggtGeometry {
    pub const MIN_ASPECT: f64 = 0.5;
    pub const MAX_ASPECT: f64 = 2.0;

    pub fn plan(src_w: usize, src_h: usize, resolution: usize, patch: usize) -> Self {
        Self::plan_with(src_w, src_h, resolution, patch, false)
    }

    /// Plan a crop/scale/rotate chain.
    ///
    /// `portrait_only` forces the output canvas to be taller than it is wide by
    /// rotating landscape sources a quarter turn.
    pub fn plan_with(
        src_w: usize,
        src_h: usize,
        resolution: usize,
        patch: usize,
        portrait_only: bool,
    ) -> Self {
        let patch = patch.max(1);
        let src_w = src_w.max(1);
        let src_h = src_h.max(1);
        let aspect = src_h as f64 / src_w as f64;

        // 1. Centre-crop the source into the supported aspect range.
        let (crop_w, crop_h) = if aspect < Self::MIN_ASPECT {
            let w = ((src_h as f64 / Self::MIN_ASPECT).round() as usize).clamp(1, src_w);
            (w, src_h)
        } else if aspect > Self::MAX_ASPECT {
            let h = ((src_w as f64 * Self::MAX_ASPECT).round() as usize).clamp(1, src_h);
            (src_w, h)
        } else {
            (src_w, src_h)
        };
        let crop_x = ((src_w - crop_w) / 2) & !1;
        let crop_y = ((src_h - crop_h) / 2) & !1;

        // 2. Optionally rotate so the long side is vertical.
        let rotated = portrait_only && crop_w > crop_h;
        let (work_w, work_h) = if rotated {
            (crop_h, crop_w)
        } else {
            (crop_w, crop_h)
        };

        // 3. Scale the longest side to `resolution`; the other side snaps to a
        //    multiple of the patch size, exactly like the reference loader.
        let aspect = work_h as f64 / work_w as f64;
        let round_patch = |value: f64| -> usize {
            let steps = (value / patch as f64).round().max(1.0);
            (steps as usize) * patch
        };
        let (out_h, out_w) = if aspect >= 1.0 {
            (resolution, round_patch(resolution as f64 / aspect))
        } else {
            (round_patch(resolution as f64 * aspect), resolution)
        };

        Self {
            crop_x,
            crop_y,
            crop_w,
            crop_h,
            out_w: out_w.max(patch),
            out_h: out_h.max(patch),
            rotated,
        }
    }

    /// Plan an exact output canvas.
    ///
    /// The exported VGGT-Omega graph has `images[1, N, 3, 512, 288]` baked in,
    /// so a clip of any other aspect ratio has to be centre-cropped to the
    /// canvas aspect before it is scaled. `canvas` is `(height, width)`.
    pub fn plan_canvas(
        src_w: usize,
        src_h: usize,
        canvas: (usize, usize),
        patch: usize,
        portrait_only: bool,
    ) -> Self {
        let patch = patch.max(1);
        let src_w = src_w.max(1);
        let src_h = src_h.max(1);
        let out_h = canvas.0.max(patch);
        let out_w = canvas.1.max(patch);

        // A landscape source is rotated a quarter turn, so the crop it needs
        // before rotation has the transposed aspect.
        let rotated = portrait_only && src_w > src_h;
        let mut target = out_h as f64 / out_w as f64;
        if rotated {
            target = 1.0 / target;
        }

        let (mut crop_w, mut crop_h) = (src_w, src_h);
        if src_h as f64 / src_w as f64 > target {
            crop_h = ((src_w as f64 * target).round() as usize).clamp(1, src_h);
        } else {
            crop_w = ((src_h as f64 / target).round() as usize).clamp(1, src_w);
        }
        let crop_x = ((src_w - crop_w) / 2) & !1;
        let crop_y = ((src_h - crop_h) / 2) & !1;

        Self {
            crop_x,
            crop_y,
            crop_w,
            crop_h,
            out_w,
            out_h,
            rotated,
        }
    }

    /// ffmpeg filter chain implementing this plan.
    pub fn filter(&self) -> String {
        let rotate = if self.rotated { "transpose=clock," } else { "" };
        format!(
            "crop={}:{}:{}:{},{}scale={}:{}:flags=lanczos",
            self.crop_w, self.crop_h, self.crop_x, self.crop_y, rotate, self.out_w, self.out_h
        )
    }

    /// Patch grid the model will see.
    pub fn patch_grid(&self, patch: usize) -> (usize, usize) {
        (self.out_h / patch, self.out_w / patch)
    }

    /// Undo the quarter-turn rotation on a camera rotation matrix.
    ///
    /// Rotating the picture a quarter turn clockwise maps a camera-space point
    /// `(X, Y)` to `(-Y, X)`, i.e. it left-multiplies the world→camera rotation
    /// by `Rz = [[0,-1,0],[1,0,0],[0,0,1]]`. The true rotation is therefore
    /// `Rzᵀ · R_model`.
    pub fn unrotate_rotation(&self, r: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
        if !self.rotated {
            return *r;
        }
        let rz_t = [[0.0f32, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let mut out = [[0.0f32; 3]; 3];
        for (y, row) in out.iter_mut().enumerate() {
            for (x, value) in row.iter_mut().enumerate() {
                *value = rz_t[y][0] * r[0][x] + rz_t[y][1] * r[1][x] + rz_t[y][2] * r[2][x];
            }
        }
        out
    }

    /// Undo the quarter turn on a camera-space translation.
    pub fn unrotate_translation(&self, t: [f32; 3]) -> [f32; 3] {
        if !self.rotated {
            return t;
        }
        [t[1], -t[0], t[2]]
    }

    /// Source-image size that the model's field of view refers to.
    ///
    /// After a quarter turn the network's width axis is the source's height
    /// axis, so the two swap back here.
    pub fn unrotate_fov(&self, fov_h: f32, fov_w: f32) -> (f32, f32) {
        if self.rotated {
            (fov_w, fov_h)
        } else {
            (fov_h, fov_w)
        }
    }
}

/// Streaming decoder that yields frames one at a time.
pub struct Decoder {
    child: Child,
    reader: BufReader<std::process::ChildStdout>,
    width: usize,
    height: usize,
    index: usize,
    buffer: Vec<u8>,
    info: VideoInfo,
}

impl Decoder {
    pub fn open(tools: &Tools, path: impl AsRef<Path>, scale_to_width: Option<usize>) -> Result<Self> {
        let info = probe(tools, path.as_ref())?;
        let (width, height) = match scale_to_width {
            Some(target) if target > 0 && target < info.width => {
                let scaled_h =
                    (info.height as f64 * target as f64 / info.width as f64).round() as usize;
                // Even dimensions keep every decoder happy.
                (target & !1, (scaled_h & !1).max(2))
            }
            _ => (info.width & !1, info.height & !1),
        };
        Self::spawn(
            tools,
            path.as_ref(),
            &format!("scale={width}:{height}:flags=area"),
            width,
            height,
        )
    }

    /// `Decoder::open` for the motion pipeline: every clip is resampled to the
    /// reference's constant 30 fps (`video.py`), so the model never sees a
    /// 24/25/50/60 fps clip at its own rate.
    pub fn open_flow(
        tools: &Tools,
        path: impl AsRef<Path>,
        scale_to_width: Option<usize>,
    ) -> Result<Self> {
        let info = probe(tools, path.as_ref())?;
        let (width, height) = match scale_to_width {
            Some(target) if target > 0 && target < info.width => {
                let scaled_h =
                    (info.height as f64 * target as f64 / info.width as f64).round() as usize;
                (target & !1, (scaled_h & !1).max(2))
            }
            _ => (info.width & !1, info.height & !1),
        };
        let filter = Self::flow_filter(&format!("scale={width}:{height}:flags=area"));
        Self::spawn_with(tools, path.as_ref(), &filter, width, height, Some(Self::FLOW_FPS))
    }

    /// Open a stream already cropped and scaled for VGGT-Omega, so the model
    /// receives exactly the `out_w × out_h` geometry [`VggtGeometry`] planned.
    pub fn open_vggt(
        tools: &Tools,
        path: impl AsRef<Path>,
        resolution: usize,
        patch: usize,
    ) -> Result<(Self, VggtGeometry)> {
        Self::open_vggt_with(tools, path, resolution, patch, false)
    }

    /// Like [`Self::open_vggt`], but able to force the portrait canvas the
    /// shipped graph expects.
    pub fn open_vggt_with(
        tools: &Tools,
        path: impl AsRef<Path>,
        resolution: usize,
        patch: usize,
        portrait_only: bool,
    ) -> Result<(Self, VggtGeometry)> {
        let info = probe(tools, path.as_ref())?;
        let geometry = VggtGeometry::plan_with(
            info.width,
            info.height,
            resolution,
            patch,
            portrait_only,
        );
        let decoder = Self::spawn(
            tools,
            path.as_ref(),
            &geometry.filter(),
            geometry.out_w,
            geometry.out_h,
        )?;
        Ok((decoder, geometry))
    }

    /// Open a VGGT decode stream scaled to an exact canvas.
    ///
    /// See [`VggtGeometry::plan_canvas`]; the exported graph only accepts the
    /// canvas it was traced with.
    pub fn open_vggt_canvas(
        tools: &Tools,
        path: impl AsRef<Path>,
        canvas: (usize, usize),
        patch: usize,
        portrait_only: bool,
    ) -> Result<(Self, VggtGeometry)> {
        let info = probe(tools, path.as_ref())?;
        let geometry =
            VggtGeometry::plan_canvas(info.width, info.height, canvas, patch, portrait_only);
        let decoder = Self::spawn(
            tools,
            path.as_ref(),
            &geometry.filter(),
            geometry.out_w,
            geometry.out_h,
        )?;
        Ok((decoder, geometry))
    }

    /// `Decoder::open_vggt_with` for the motion pipeline: every clip is resampled
    /// to the reference's constant 30 fps first.
    pub fn open_vggt_with_flow(
        tools: &Tools,
        path: impl AsRef<Path>,
        resolution: usize,
        patch: usize,
        portrait_only: bool,
    ) -> Result<(Self, VggtGeometry)> {
        let info = probe(tools, path.as_ref())?;
        let geometry = VggtGeometry::plan_with(
            info.width,
            info.height,
            resolution,
            patch,
            portrait_only,
        );
        let filter = Self::flow_filter(&geometry.filter());
        let decoder = Self::spawn_with(
            tools,
            path.as_ref(),
            &filter,
            geometry.out_w,
            geometry.out_h,
            Some(Self::FLOW_FPS),
        )?;
        Ok((decoder, geometry))
    }

    /// `Decoder::open_vggt_canvas` for the motion pipeline: every clip is
    /// resampled to the reference's constant 30 fps first.
    pub fn open_vggt_canvas_flow(
        tools: &Tools,
        path: impl AsRef<Path>,
        canvas: (usize, usize),
        patch: usize,
        portrait_only: bool,
    ) -> Result<(Self, VggtGeometry)> {
        let info = probe(tools, path.as_ref())?;
        let geometry =
            VggtGeometry::plan_canvas(info.width, info.height, canvas, patch, portrait_only);
        let filter = Self::flow_filter(&geometry.filter());
        let decoder = Self::spawn_with(
            tools,
            path.as_ref(),
            &filter,
            geometry.out_w,
            geometry.out_h,
            Some(Self::FLOW_FPS),
        )?;
        Ok((decoder, geometry))
    }

    /// Open a decode stream that starts at `start_seconds` (fast seek placed
    /// before `-i`) and reports frame indices from `first_index`.
    ///
    /// This is what the player uses: one ffmpeg process streams frames in order
    /// instead of paying a seek per displayed frame.
    pub fn spawn_at(
        tools: &Tools,
        path: &Path,
        filter: &str,
        width: usize,
        height: usize,
        start_seconds: f64,
        first_index: usize,
    ) -> Result<Self> {
        let info = probe(tools, path)?;
        let mut command = Command::new(&tools.ffmpeg);
        command
            .args(["-hide_banner", "-loglevel", "error", "-nostdin"])
            .arg("-ss")
            .arg(format!("{:.6}", start_seconds.max(0.0)))
            .arg("-i")
            .arg(path)
            .args([
                "-vf",
                filter,
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .no_window();

        let mut child = command.spawn().context("starting ffmpeg decoder")?;
        let stdout = child.stdout.take().context("ffmpeg produced no stdout")?;
        Ok(Self {
            child,
            reader: BufReader::with_capacity(width * height * 3, stdout),
            width,
            height,
            index: first_index,
            buffer: vec![0u8; width * height * 3],
            info,
        })
    }

    /// Open a decode stream with an explicit ffmpeg filter chain.
    pub fn spawn(
        tools: &Tools,
        path: &Path,
        filter: &str,
        width: usize,
        height: usize,
    ) -> Result<Self> {
        Self::spawn_with(tools, path, filter, width, height, None)
    }

    /// The motion pipeline's frame rate: `flowhmr` re-encodes every clip to a
    /// constant 30 fps (`video.py`, `-vf fps=30:start_time=0`) before any
    /// inference, so the model never sees a 24/25/50/60 fps clip at its own rate.
    pub const FLOW_FPS: f64 = 30.0;

    /// The reference's resampling filter, prepended to an existing chain.
    pub fn flow_filter(base: &str) -> String {
        format!("fps={}:start_time=0,{base}", Self::FLOW_FPS)
    }

    fn spawn_with(
        tools: &Tools,
        path: &Path,
        filter: &str,
        width: usize,
        height: usize,
        fps: Option<f64>,
    ) -> Result<Self> {
        let mut info = probe(tools, path)?;
        if let Some(fps) = fps {
            // The stream is now constant-rate, so the frame count follows from the
            // duration, and every per-frame time in the pipeline is `i / fps`.
            info.frames = ((info.duration * fps).round() as usize).max(1);
            info.fps = fps;
        }
        let mut command = Command::new(&tools.ffmpeg);
        command
            .args(["-hide_banner", "-loglevel", "error", "-nostdin"])
            .arg("-i")
            .arg(path)
            .args([
                "-vf",
                filter,
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .no_window();

        let mut child = command.spawn().context("starting ffmpeg decoder")?;
        let stdout = child.stdout.take().context("ffmpeg produced no stdout")?;
        Ok(Self {
            child,
            reader: BufReader::with_capacity(width * height * 3, stdout),
            width,
            height,
            index: 0,
            buffer: vec![0u8; width * height * 3],
            info,
        })
    }

    pub fn info(&self) -> &VideoInfo {
        &self.info
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// Read the next frame; `Ok(None)` at end of stream.
    pub fn next_frame(&mut self) -> Result<Option<Frame>> {
        match self.reader.read_exact(&mut self.buffer) {
            Ok(()) => {
                let frame = Frame {
                    index: self.index,
                    width: self.width,
                    height: self.height,
                    rgb: self.buffer.clone(),
                };
                self.index += 1;
                Ok(Some(frame))
            }
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => Ok(None),
            Err(error) => Err(error).context("reading decoded frame"),
        }
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Decode a single frame by seeking. Returns `None` if the index is past the end.
///
/// Uses `-ss` before `-i` so ffmpeg seeks rather than decoding from the start;
/// this keeps UI previews cheap on long clips.
pub fn decode_frame_at(
    tools: &Tools,
    path: impl AsRef<Path>,
    index: usize,
    scale_to_width: Option<usize>,
) -> Option<Frame> {
    decode_frame_at_grid(tools, path, index, scale_to_width, None)
}

/// Decode one frame on a resampled time grid.
///
/// The motion path resamples every clip to 30 fps (`Decoder::open_flow`) and the
/// playback stream paces itself on that same grid, but a seek used the
/// container's own frame rate. On a 24/25/60 fps clip that picture belonged to a
/// different moment than the frame the timeline had selected, so "setting the
/// frame" looked like it did nothing. `grid_fps` puts the seek on the stream's
/// grid: the offset is `index / grid_fps` and ffmpeg emits the frame at that
/// moment of the resampled timeline.
pub fn decode_frame_at_grid(
    tools: &Tools,
    path: impl AsRef<Path>,
    index: usize,
    scale_to_width: Option<usize>,
    grid_fps: Option<f64>,
) -> Option<Frame> {
    let info = probe(tools, path.as_ref()).ok()?;
    let (width, height) = match scale_to_width {
        Some(target) if target > 0 && target < info.width => {
            let scaled_h = (info.height as f64 * target as f64 / info.width as f64).round() as usize;
            (target & !1, (scaled_h & !1).max(2))
        }
        _ => (info.width & !1, info.height & !1),
    };
    let grid = grid_fps.filter(|fps| fps.is_finite() && *fps > 0.0);
    let fps = grid.unwrap_or(if info.fps > 0.0 { info.fps } else { 30.0 });
    let seconds = index as f64 / fps;
    let filter = match grid {
        Some(grid) => format!("fps={grid},scale={width}:{height}:flags=area"),
        None => format!("scale={width}:{height}:flags=area"),
    };

    let mut command = Command::new(&tools.ffmpeg);
    command
        .args(["-hide_banner", "-loglevel", "error", "-nostdin"])
        .arg("-ss")
        .arg(format!("{seconds:.6}"))
        .arg("-i")
        .arg(path.as_ref())
        .args([
            "-vf",
            &filter,
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .no_window();

    let output = command.output().ok()?;
    let expected = width * height * 3;
    if output.stdout.len() < expected {
        return None;
    }
    Some(Frame {
        index,
        width,
        height,
        rgb: output.stdout[..expected].to_vec(),
    })
}

/// Decode frames up to `limit` (0 = all). Intended for short clips and tests.
pub fn decode_all(
    tools: &Tools,
    path: impl AsRef<Path>,
    scale_to_width: Option<usize>,
    limit: usize,
) -> Result<(VideoInfo, Vec<Frame>)> {
    let mut decoder = Decoder::open(tools, path, scale_to_width)?;
    let info = decoder.info().clone();
    let mut frames = Vec::new();
    while let Some(frame) = decoder.next_frame()? {
        frames.push(frame);
        if limit > 0 && frames.len() >= limit {
            break;
        }
    }
    Ok((info, frames))
}

/// Windows: hide the console window for child processes.
trait NoWindow {
    fn no_window(&mut self) -> &mut Self;
}

impl NoWindow for Command {
    fn no_window(&mut self) -> &mut Self {
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            self.creation_flags(0x0800_0000);
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rational_parsing() {
        assert!((parse_rational("30000/1001").unwrap() - 29.97002997).abs() < 1e-6);
        assert_eq!(parse_rational("30/1"), Some(30.0));
        assert_eq!(parse_rational("0/0"), None);
        assert_eq!(parse_rational("garbage"), None);
    }

    #[test]
    fn probe_json_field_extraction() {
        let json = r#"{"streams":[{"width":1920,"height":1080,"avg_frame_rate":"30/1","nb_frames":"900","codec_name":"h264","duration":"30.0"}],"format":{"duration":"30.0","size":"12345"}}"#;
        let info = parse_probe(Path::new("x.mp4"), json).unwrap();
        assert_eq!(info.width, 1920);
        assert_eq!(info.height, 1080);
        assert!((info.fps - 30.0).abs() < 1e-9);
        assert_eq!(info.frames, 900);
        assert_eq!(info.codec, "h264");
        assert_eq!(info.bytes, 12345);
    }

    #[test]
    fn landscape_is_rotated_into_the_portrait_canvas() {
        // A 16:9 clip rotated a quarter turn becomes 512 tall × 288 wide, which
        // is the canvas the shipped graph was traced with.
        let g = VggtGeometry::plan_with(1920, 1080, 512, 16, true);
        assert!(g.rotated);
        assert_eq!((g.out_h, g.out_w), (512, 288));
        assert!(g.filter().contains("transpose=clock"));

        // A portrait clip needs no rotation.
        let g = VggtGeometry::plan_with(1080, 1920, 512, 16, true);
        assert!(!g.rotated);
        assert_eq!((g.out_h, g.out_w), (512, 288));
        assert!(!g.filter().contains("transpose"));
    }

    #[test]
    fn unrotate_is_a_rotation_and_a_transpose() {
        let g = VggtGeometry::plan_with(1920, 1080, 512, 16, true);
        // Rotating an identity camera back must still be orthonormal.
        let r = g.unrotate_rotation(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        for (y, row) in r.iter().enumerate() {
            for (x, _) in row.iter().enumerate() {
                let dot: f32 = (0..3).map(|k| r[y][k] * r[x][k]).sum();
                let expect = if x == y { 1.0 } else { 0.0 };
                assert!((dot - expect).abs() < 1e-6, "not orthonormal: {r:?}");
            }
        }
        // A quarter turn maps +X to +Y.
        assert!((r[0][0]).abs() < 1e-6 && (r[1][0] + 1.0).abs() < 1e-6);

        // Translation follows the same basis change.
        assert_eq!(g.unrotate_translation([1.0, 2.0, 3.0]), [2.0, -1.0, 3.0]);

        // Without rotation both are identities.
        let plain = VggtGeometry::plan_with(1080, 1920, 512, 16, true);
        assert_eq!(plain.unrotate_translation([1.0, 2.0, 3.0]), [1.0, 2.0, 3.0]);
    }

    /// A seek has to land on the grid the clip is played on.
    ///
    /// The motion path resamples every clip to 30 fps and the playback stream
    /// paces itself on that grid, so a seek that used the container's own rate
    /// asked for a different moment of the clip. `test-24fps.mp4` is 72 frames at
    /// 24 fps (3.0 s): as a 30 fps grid it has 90 frames, so frame 72 exists on
    /// the grid and does not exist on the native timeline. That is the picture
    /// the timeline showed but the seek never delivered.
    #[test]
    fn a_seek_lands_on_the_playback_grid() {
        let Ok(tools) = Tools::discover() else {
            eprintln!("skipping: no ffmpeg");
            return;
        };
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../build/media/test-24fps.mp4");
        if !path.is_file() {
            eprintln!("skipping: no 24 fps sample");
            return;
        }
        let info = probe(&tools, &path).expect("probe");
        assert!((info.fps - 24.0).abs() < 0.01, "sample is {:.3} fps", info.fps);
        // Native grid: 72 frames over 3.0 s, so index 72 is past the end.
        assert!(
            decode_frame_at(&tools, &path, 72, Some(96)).is_none(),
            "the native timeline unexpectedly has frame 72"
        );
        // 30 fps grid: 90 frames over the same 3.0 s, so 72 is inside.
        assert!(
            decode_frame_at_grid(&tools, &path, 72, Some(96), Some(30.0)).is_some(),
            "the 30 fps grid has no frame 72"
        );
    }

    #[test]
    fn unrotated_relative_rotation_is_identity_at_frame_zero() {
        // Whatever constant roll the un-rotation adds must cancel in the
        // frame-to-frame relative rotation the motion model conditions on.
        let g = VggtGeometry::plan_with(1920, 1080, 512, 16, true);
        let r0 = g.unrotate_rotation(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        let mut rel = [[0.0f32; 3]; 3];
        for y in 0..3 {
            for x in 0..3 {
                rel[y][x] = (0..3).map(|k| r0[y][k] * r0[x][k]).sum();
            }
        }
        for (y, row) in rel.iter().enumerate() {
            for (x, value) in row.iter().enumerate() {
                let expect = if x == y { 1.0 } else { 0.0 };
                assert!((value - expect).abs() < 1e-6, "relative rotation not identity: {rel:?}");
            }
        }
    }
}
