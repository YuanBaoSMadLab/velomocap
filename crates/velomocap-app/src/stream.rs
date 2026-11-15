//! Streaming preview playback.
//!
//! The preview used to be one `ffmpeg -ss … -frames:v 1` per displayed frame:
//! 60-80 ms each, so at 30 fps it could never keep up and the pane jumped
//! between whatever finished last. This module keeps **one** ffmpeg process
//! alive per playback run and reads its frames in order; the UI takes whatever
//! is newest. Scrubbing still uses a single-frame decode, which is when a seek
//! is the right tool.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// One decoded frame, ready to become a texture.
pub struct StreamFrame {
    pub index: usize,
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
}

#[derive(Clone)]
enum Command {
    /// Decode exactly one frame (scrubbing, and the paused state).
    Seek {
        generation: u64,
        serial: u64,
        path: PathBuf,
        frame: usize,
        scale: usize,
    },
    /// Stream from `frame` in order, pacing at `fps * speed`.
    Play {
        generation: u64,
        serial: u64,
        path: PathBuf,
        frame: usize,
        scale: usize,
        fps: f64,
        speed: f32,
    },
}

/// What the worker has been asked for, so the UI can poll it every frame
/// without restarting ffmpeg each time.
#[derive(Clone, Copy, PartialEq)]
struct Asked {
    frame: usize,
    playing: bool,
    speed: u16,
}

pub struct VideoStream {
    commands: Sender<Command>,
    shared: Arc<Mutex<Option<(u64, u64, Arc<StreamFrame>)>>>,
    version: Arc<AtomicU64>,
    seen: u64,
    generation: u64,
    asked: Option<Asked>,
    /// Bumped on every request: a frame decoded for an older request is stale
    /// and must not be painted over the newer one.
    serial: u64,
}

impl VideoStream {
    pub fn spawn() -> Self {
        let (commands, requests) = std::sync::mpsc::channel::<Command>();
        let shared: Arc<Mutex<Option<(u64, u64, Arc<StreamFrame>)>>> = Arc::new(Mutex::new(None));
        let version = Arc::new(AtomicU64::new(0));
        let worker_shared = Arc::clone(&shared);
        let worker_version = Arc::clone(&version);
        std::thread::Builder::new()
            .name("velomocap-stream".into())
            .spawn(move || run(requests, worker_shared, worker_version))
            .ok();
        Self {
            commands,
            shared,
            version,
            seen: 0,
            generation: 0,
            asked: None,
            serial: 0,
        }
    }

    /// Drop everything decoded for the previous clip.
    pub fn restart(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        self.asked = None;
        if let Ok(mut slot) = self.shared.lock() {
            *slot = None;
        }
        self.seen = self.version.load(Ordering::Relaxed);
    }

    /// Ask for the frame the UI wants. Deduplicated, so calling it every frame
    /// is cheap: ffmpeg is only restarted when the request actually changes.
    pub fn request(
        &mut self,
        path: &Path,
        frame: usize,
        playing: bool,
        speed: f32,
        scale: usize,
        fps: f64,
    ) {
        let asked = Asked {
            frame,
            playing,
            speed: (speed.max(0.05) * 100.0).round() as u16,
        };
        if self.asked == Some(asked) {
            return;
        }
        self.asked = Some(asked);
        self.serial = self.serial.wrapping_add(1);
        let serial = self.serial;
        let command = if playing {
            Command::Play {
                generation: self.generation,
                serial,
                path: path.to_path_buf(),
                frame,
                scale,
                fps,
                speed: speed.max(0.05),
            }
        } else {
            Command::Seek {
                generation: self.generation,
                serial,
                path: path.to_path_buf(),
                frame,
                scale,
            }
        };
        let _ = self.commands.send(command);
    }

    /// The clip generation the worker is serving; frames from an older one are
    /// dropped rather than painted over the new clip.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// The newest frame, if one arrived since the last call.
    pub fn poll(&mut self) -> Option<(u64, Arc<StreamFrame>)> {
        let version = self.version.load(Ordering::Relaxed);
        if version == self.seen {
            return None;
        }
        self.seen = version;
        let slot = self.shared.lock().ok().and_then(|slot| slot.clone())?;
        if slot.1 != self.serial {
            // Decoded for a request the UI has already moved past.
            return None;
        }
        Some((slot.0, slot.2))
    }
}

fn run(
    requests: Receiver<Command>,
    shared: Arc<Mutex<Option<(u64, u64, Arc<StreamFrame>)>>>,
    version: Arc<AtomicU64>,
) {
    let Some(tools) = velomocap_video::Tools::discover().ok() else {
        return;
    };
    let mut pending: Option<Command> = None;
    loop {
        let command = match pending.take() {
            Some(command) => command,
            None => match requests.recv() {
                Ok(command) => command,
                Err(_) => break,
            },
        };
        match command {
            Command::Seek {
                generation,
                serial,
                path,
                frame,
                scale,
            } => {
                if let Some(decoded) =
                    velomocap_video::decode_frame_at(&tools, &path, frame, Some(scale))
                {
                    publish(&shared, &version, generation, serial, decoded);
                }
            }
            Command::Play {
                generation,
                serial,
                path,
                frame,
                scale,
                fps,
                speed,
            } => match play(
                &tools,
                &requests,
                &shared,
                &version,
                generation,
                serial,
                &path,
                frame,
                scale,
                fps,
                speed,
            ) {
                PlayOutcome::Command(next) => pending = Some(next),
                PlayOutcome::Stop => break,
            },
        }
    }
}

enum PlayOutcome {
    Command(Command),
    Stop,
}

#[allow(clippy::too_many_arguments)]
fn play(
    tools: &velomocap_video::Tools,
    requests: &Receiver<Command>,
    shared: &Arc<Mutex<Option<(u64, u64, Arc<StreamFrame>)>>>,
    version: &Arc<AtomicU64>,
    generation: u64,
    serial: u64,
    path: &Path,
    frame: usize,
    scale: usize,
    fps: f64,
    speed: f32,
) -> PlayOutcome {
    let fps = if fps.is_finite() && fps > 0.0 { fps } else { 30.0 };
    let Ok(info) = velomocap_video::probe(tools, path) else {
        return PlayOutcome::Stop;
    };
    // Same downscale rule as `decode_frame_at`: only shrink, never upscale.
    let (width, height) = if scale > 0 && scale < info.width {
        let scaled = (info.height as f64 * scale as f64 / info.width as f64).round() as usize;
        (scale & !1, (scaled & !1).max(2))
    } else {
        (info.width & !1, info.height & !1)
    };
    let filter = format!("scale={width}:{height}:flags=bilinear");
    let start_seconds = frame as f64 / fps;
    let Ok(mut decoder) = velomocap_video::Decoder::spawn_at(
        tools,
        path,
        &filter,
        width,
        height,
        start_seconds,
        frame,
    ) else {
        return PlayOutcome::Stop;
    };

    let step = Duration::from_secs_f64(1.0 / (fps * speed as f64));
    let mut schedule = Instant::now();
    let mut index = frame;
    loop {
        // A new request (scrub, pause, different position) closes this stream.
        match requests.try_recv() {
            Ok(next) => return PlayOutcome::Command(next),
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => return PlayOutcome::Stop,
        }
        match decoder.next_frame() {
            Ok(Some(decoded)) => {
                index = decoded.index;
                publish(shared, version, generation, serial, decoded);
                schedule += step;
                let now = Instant::now();
                if schedule > now {
                    // Sleep in short slices so a pause is honoured promptly.
                    let mut left = schedule - now;
                    while left > Duration::from_millis(4) {
                        std::thread::sleep(Duration::from_millis(4));
                        left -= Duration::from_millis(4);
                        if let Ok(next) = requests.try_recv() {
                            return PlayOutcome::Command(next);
                        }
                    }
                    std::thread::sleep(left);
                } else {
                    schedule = now;
                }
            }
            // End of the clip: loop back to the start like the transport does.
            Ok(None) | Err(_) => {
                let _ = index;
                return PlayOutcome::Command(Command::Play {
                    generation,
                    serial,
                    path: path.to_path_buf(),
                    frame: 0,
                    scale,
                    fps,
                    speed,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Diagnostic: what cadence does a playback run actually deliver, and are
    /// the indices monotonic? Prints the first frames with their arrival times.
    #[test]
    fn playback_stream_cadence() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../build/media/test-clip.mp4");
        if !path.is_file() || velomocap_video::Tools::discover().is_err() {
            eprintln!("skipping: no sample clip or no ffmpeg");
            return;
        }
        let mut stream = VideoStream::spawn();
        let fps = 30.0;
        let started = Instant::now();
        stream.request(&path, 0, true, 1.0, 480, fps);
        let mut seen: Vec<(usize, f64)> = Vec::new();
        while started.elapsed() < Duration::from_secs(3) && seen.len() < 40 {
            if let Some((_, frame)) = stream.poll() {
                let at = started.elapsed().as_secs_f64();
                if seen.last().map(|last| last.0) != Some(frame.index) {
                    seen.push((frame.index, at));
                }
            } else {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        for (index, at) in &seen {
            eprintln!("frame {index:3} at {at:.3} s");
        }
        let gaps: Vec<f64> = seen.windows(2).map(|pair| pair[1].1 - pair[0].1).collect();
        let mean = gaps.iter().sum::<f64>() / gaps.len().max(1) as f64;
        eprintln!(
            "{} frames, mean gap {:.1} ms (expected {:.1} ms), max {:.1} ms",
            seen.len(),
            mean * 1000.0,
            1000.0 / fps,
            gaps.iter().cloned().fold(0.0, f64::max) * 1000.0
        );
        let monotonic = seen.windows(2).all(|pair| pair[1].0 > pair[0].0);
        assert!(monotonic, "frame indices went backwards: {seen:?}");
    }
}

fn publish(
    shared: &Arc<Mutex<Option<(u64, u64, Arc<StreamFrame>)>>>,
    version: &Arc<AtomicU64>,
    generation: u64,
    serial: u64,
    decoded: velomocap_video::Frame,
) {
    let frame = Arc::new(StreamFrame {
        index: decoded.index,
        width: decoded.width,
        height: decoded.height,
        rgb: decoded.rgb,
    });
    if let Ok(mut slot) = shared.lock() {
        *slot = Some((generation, serial, frame));
    }
    version.fetch_add(1, Ordering::Relaxed);
}
