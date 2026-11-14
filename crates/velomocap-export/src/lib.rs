//! Export tracked people to portable formats.
//!
//! Two outputs are produced next to the source video:
//!
//! * `<stem>.tracks.json` — full per-frame boxes per track, for other tools.
//! * `<stem>.tracks.csv`  — one row per detection, easy to open in a spreadsheet.
//!
//! Both are written atomically (temp file + rename) so an interrupted export
//! never leaves a truncated file.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use velomocap_infer::Track;
use velomocap_video::VideoInfo;

pub struct ExportResult {
    pub json: PathBuf,
    pub csv: PathBuf,
    pub detections: usize,
    pub tracks: usize,
}

pub fn export(
    video: &Path,
    info: Option<&VideoInfo>,
    tracks: &[Track],
    frames: usize,
) -> Result<ExportResult> {
    let stem = video
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "tracks".to_string());
    let folder = video.parent().unwrap_or(Path::new("."));
    let json_path = folder.join(format!("{stem}.tracks.json"));
    let csv_path = folder.join(format!("{stem}.tracks.csv"));

    let mut json = String::with_capacity(64 * 1024);
    json.push_str("{\n");
    let _ = writeln!(json, "  \"source\": {},", json_string(&video.to_string_lossy()));
    if let Some(info) = info {
        let _ = writeln!(json, "  \"width\": {},", info.width);
        let _ = writeln!(json, "  \"height\": {},", info.height);
        let _ = writeln!(json, "  \"fps\": {},", info.fps);
    }
    let _ = writeln!(json, "  \"frames\": {frames},");
    let _ = writeln!(json, "  \"tracks\": [");
    let mut detections = 0usize;
    for (index, track) in tracks.iter().enumerate() {
        let _ = writeln!(json, "    {{");
        let _ = writeln!(json, "      \"id\": {},", track.id);
        let _ = writeln!(json, "      \"first_frame\": {},", track.first_frame);
        let _ = writeln!(json, "      \"last_frame\": {},", track.last_frame);
        let _ = writeln!(json, "      \"hits\": {},", track.hits);
        let _ = writeln!(json, "      \"mean_score\": {:.4},", track.mean_score());
        let _ = writeln!(json, "      \"boxes\": [");
        let mut first = true;
        for (frame, slot) in track.boxes.iter().enumerate() {
            let Some(detection) = slot else { continue };
            detections += 1;
            if !first {
                json.push_str(",\n");
            }
            first = false;
            let _ = write!(
                json,
                "        {{\"frame\": {frame}, \"x0\": {:.2}, \"y0\": {:.2}, \"x1\": {:.2}, \"y1\": {:.2}, \"score\": {:.4}}}",
                detection.x0, detection.y0, detection.x1, detection.y1, detection.score
            );
        }
        json.push('\n');
        let _ = writeln!(json, "      ]");
        let comma = if index + 1 == tracks.len() { "" } else { "," };
        let _ = writeln!(json, "    }}{comma}");
    }
    json.push_str("  ]\n}\n");

    let mut csv = String::with_capacity(64 * 1024);
    csv.push_str("track,frame,x0,y0,x1,y1,width,height,score\n");
    for track in tracks {
        for (frame, slot) in track.boxes.iter().enumerate() {
            let Some(detection) = slot else { continue };
            let _ = writeln!(
                csv,
                "{},{frame},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.4}",
                track.id,
                detection.x0,
                detection.y0,
                detection.x1,
                detection.y1,
                detection.width(),
                detection.height(),
                detection.score
            );
        }
    }

    write_atomic(&json_path, json.as_bytes())?;
    write_atomic(&csv_path, csv.as_bytes())?;
    Ok(ExportResult {
        json: json_path,
        csv: csv_path,
        detections,
        tracks: tracks.len(),
    })
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let temp = path.with_extension(format!(
        "{}.tmp",
        path.extension()
            .map(|e| e.to_string_lossy().to_string())
            .unwrap_or_default()
    ));
    std::fs::write(&temp, bytes).with_context(|| format!("writing {}", temp.display()))?;
    std::fs::rename(&temp, path).with_context(|| format!("committing {}", path.display()))?;
    Ok(())
}

fn json_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
