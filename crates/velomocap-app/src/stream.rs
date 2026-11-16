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
        /// The resampled grid the clip is played on; the seek must land on it.
        fps: f64,
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
    /// The frame rate the stream paces itself with. It arrives with the probe,
    /// so the first request can go out before the container has been asked, and
    /// pacing a 25 fps clip at 30 fps would drift until the picture and the
    /// transport separated.
    fps: u16,
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
    /// The serial of the newest frame that was painted. While a drag is running
    /// the requests outpace the decoder, so "newer than the last request" would
    /// discard every frame that arrives; "newer than the last one shown" paints
    /// each decode as it lands, which is what makes scrubbing live.
    shown: u64,
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
            shown: 0,
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
        // Everything asked for so far belongs to the clip being dropped, so the
        // new clip does not look like it owes a frame.
        self.shown = self.serial;
    }

    /// A request has been sent and no frame for it has been shown yet.
    ///
    /// The UI uses this to keep one seek in flight instead of queueing one per
    /// mouse move: a decode takes tens of milliseconds, and a request per paint
    /// meant every decoded frame arrived already superseded and was discarded.
    pub fn outstanding(&self) -> bool {
        self.shown < self.serial
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
            fps: (fps.max(0.0) * 100.0).round() as u16,
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
                fps,
            }
        };
        let _ = self.commands.send(command);
    }

    /// Forget the last request, so the next [`Self::request`] decodes again even
    /// when it asks for the frame it was already showing. Used when the pane
    /// changes owner (a reconstruction hands it back to the transport) and the
    /// picture on screen no longer belongs to the playhead.
    pub fn invalidate(&mut self) {
        self.asked = None;
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
        // Only frames that are already on screen are dropped: a frame decoded for
        // a request the pointer has moved past is still the newest picture there
        // is, and showing it is what makes a drag show something at all.
        if slot.1 <= self.shown {
            return None;
        }
        self.shown = slot.1;
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
        let mut command = match pending.take() {
            Some(command) => command,
            None => match requests.recv() {
                Ok(command) => command,
                Err(_) => break,
            },
        };
        // Coalesce before doing any work. Dragging the timeline enqueues one
        // seek per mouse move and every seek starts its own ffmpeg, so honouring
        // the whole backlog left the pane frozen for seconds after the pointer
        // had already stopped — and the frames it eventually produced were
        // dropped as stale anyway. Only the newest position matters.
        while let Ok(newer) = requests.try_recv() {
            command = newer;
        }
        match command {
            Command::Seek {
                generation,
                serial,
                path,
                frame,
                scale,
                fps,
            } => {
                if let Some(decoded) = velomocap_video::decode_frame_at_grid(
                    &tools,
                    &path,
                    frame,
                    Some(scale),
                    Some(fps),
                ) {
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
    // Decode on the transport's own grid. The motion path resamples every clip to
    // a constant 30 fps (`Decoder::open_flow`), so the preview has to be on that
    // same grid or the person overlay would drift against the picture on a
    // 24/25/50/60 fps source.
    let filter = format!("fps={fps},scale={width}:{height}:flags=bilinear");
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

    /// A burst of seeks is what dragging the timeline produces. They must
    /// collapse to the newest position instead of being decoded one by one —
    /// otherwise the pane freezes for seconds and every frame it finally
    /// produces is stale anyway.
    #[test]
    fn rapid_seeks_collapse_to_the_newest_position() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../build/media/test-clip.mp4");
        if !path.is_file() || velomocap_video::Tools::discover().is_err() {
            eprintln!("skipping: no sample clip or no ffmpeg");
            return;
        }
        let mut stream = VideoStream::spawn();
        for frame in 0..30 {
            stream.request(&path, frame, false, 1.0, 480, 30.0);
        }
        let started = Instant::now();
        let mut first = None;
        while started.elapsed() < Duration::from_secs(5) && first.is_none() {
            if let Some((_, frame)) = stream.poll() {
                first = Some((frame.index, started.elapsed().as_secs_f64()));
            } else {
                std::thread::sleep(Duration::from_millis(2));
            }
        }
        let (index, at) = first.expect("no frame arrived within five seconds");
        eprintln!("burst of 30 seeks: first accepted frame {index} after {at:.3} s");
        assert_eq!(
            index, 29,
            "the burst was decoded in order instead of collapsed to the last position"
        );
        // Decoding the whole burst takes ~100 ms per seek; collapsing it is what
        // makes the picture follow the pointer instead of the backlog.
        assert!(
            at < 1.5,
            "the newest seek took {at:.3} s to show, so the backlog was decoded"
        );
    }

    /// The pane can change owner (reconstruction → transport) while the requested
    /// frame stays the same; `invalidate` has to force a decode then, or the stale
    /// picture would stay on screen.
    #[test]
    fn invalidate_forces_the_next_request_through() {
        let mut stream = VideoStream::spawn();
        let path = PathBuf::from("unused.mp4");
        stream.request(&path, 7, false, 1.0, 480, 30.0);
        let serial = stream.serial;
        stream.request(&path, 7, false, 1.0, 480, 30.0);
        assert_eq!(stream.serial, serial, "an identical request was not deduplicated");
        stream.invalidate();
        stream.request(&path, 7, false, 1.0, 480, 30.0);
        assert_eq!(
            stream.serial,
            serial + 1,
            "invalidate did not force a new decode"
        );
    }

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

    /// Dragging the playhead sends a seek per mouse move, and a decode takes tens
    /// of milliseconds. If a frame is dropped whenever a newer request exists, a
    /// continuous drag shows nothing until it stops — which is exactly what it
    /// did. Every decoded frame has to be painted.
    #[test]
    fn frames_are_painted_even_when_the_requests_outrun_the_decoder() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../build/media/test-clip.mp4");
        if !path.is_file() || velomocap_video::Tools::discover().is_err() {
            eprintln!("skipping: no sample clip or no ffmpeg");
            return;
        }
        let mut stream = VideoStream::spawn();
        // No throttle: a request every 16 ms, exactly what the old polling loop
        // did while the pointer moved.
        let started = Instant::now();
        let mut playhead = 0usize;
        let mut next_request = Instant::now();
        let mut painted: Vec<usize> = Vec::new();
        while started.elapsed() < Duration::from_secs(2) {
            if let Some((_, frame)) = stream.poll() {
                if painted.last() != Some(&frame.index) {
                    painted.push(frame.index);
                }
            }
            if Instant::now() >= next_request {
                playhead = (playhead + 2) % 30;
                stream.request(&path, playhead, false, 1.0, 240, 30.0);
                next_request = Instant::now() + Duration::from_millis(16);
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        eprintln!("painted {} frames while the pointer moved: {painted:?}", painted.len());
        assert!(
            painted.len() >= 3,
            "a two-second drag painted only {} frame(s): {painted:?}",
            painted.len()
        );
    }

    /// One seek, one frame: a paused seek has to land and clear the debt, which
    /// is the state the UI's "one seek in flight" gate reads.
    #[test]
    fn a_paused_seek_paints_and_clears_the_outstanding_request() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../build/media/test-clip.mp4");
        if !path.is_file() || velomocap_video::Tools::discover().is_err() {
            eprintln!("skipping: no sample clip or no ffmpeg");
            return;
        }
        let mut stream = VideoStream::spawn();
        stream.request(&path, 5, false, 1.0, 240, 30.0);
        assert!(stream.outstanding(), "a fresh request must look outstanding");
        let started = Instant::now();
        let mut painted = None;
        while started.elapsed() < Duration::from_secs(5) && painted.is_none() {
            if let Some((_, frame)) = stream.poll() {
                painted = Some(frame.index);
            } else {
                std::thread::sleep(Duration::from_millis(2));
            }
        }
        let index = painted.expect("the paused seek produced no frame");
        assert!(
            (index as i64 - 5).abs() <= 1,
            "the seek to frame 5 showed frame {index}"
        );
        assert!(
            !stream.outstanding(),
            "the painted frame did not clear the outstanding request"
        );
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
