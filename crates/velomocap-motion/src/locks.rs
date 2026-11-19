//! Manual contact-lock tracks.
//!
//! Port of the reference's sparse foot/hand lock model (`mocap/foot_contacts.py`
//! `normalize_ranges` plus the editor state behind `mocap/footlock_editor.py`):
//! four tracks — left foot, right foot, left hand, right hand — each a sorted
//! list of half-open frame intervals `[start, end)`.
//!
//! A track is `None` while it follows the clip's own detection ("automatic"),
//! and `Some` once the user has edited it. `Some(vec![])` is a real state: the
//! user deleted every lock on that limb, so that limb never touches the ground —
//! which is exactly what the reference's `reset`/`delete` end up meaning.
//!
//! In the reference, any manual edit replaces the automatic masks of *both*
//! feet at once (`correction.py::prepare_contacts` clears `active` and refills it
//! from `contact_ranges`). Per-track replacement expresses the same thing for
//! the editor's actual workflow — it hands back all four tracks, seeded from the
//! automatic runs — and keeps an untouched track on its detection instead of
//! freezing a stale copy of it.

use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Track indices, in the reference's order (`CONTACT_COLORS`).
pub const TRACK_COUNT: usize = 4;
pub const LEFT_FOOT: usize = 0;
pub const RIGHT_FOOT: usize = 1;
pub const LEFT_HAND: usize = 2;
pub const RIGHT_HAND: usize = 3;

/// Interface labels, in track order.
pub const TRACK_LABELS: [&str; TRACK_COUNT] = ["L foot", "R foot", "L hand", "R hand"];

/// Half-open frame interval `[start, end)`.
pub type Interval = [usize; 2];

/// Four sparse contact tracks.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContactTracks {
    /// `None` = follow the clip's detection, `Some` = the user's intervals.
    #[serde(default)]
    pub tracks: [Option<Vec<Interval>>; TRACK_COUNT],
}

impl ContactTracks {
    /// Every track automatic.
    pub fn automatic() -> Self {
        Self::default()
    }

    /// True while nothing has been edited, i.e. the correction can ignore this.
    pub fn is_automatic(&self) -> bool {
        self.tracks.iter().all(Option::is_none)
    }

    /// The intervals of one track, `None` when it is automatic.
    pub fn track(&self, index: usize) -> Option<&[Interval]> {
        self.tracks.get(index).and_then(Option::as_deref)
    }

    /// Replace one track with `intervals` (already bounded to the clip).
    pub fn set_track(&mut self, index: usize, intervals: Vec<Interval>) {
        if index < TRACK_COUNT {
            self.tracks[index] = Some(intervals);
        }
    }

    /// Put one track back on the clip's own detection.
    pub fn reset_track(&mut self, index: usize) {
        if index < TRACK_COUNT {
            self.tracks[index] = None;
        }
    }

    /// Put every track back on the clip's own detection.
    pub fn reset(&mut self) {
        *self = Self::automatic();
    }

    /// Number of intervals per track, `None` for an automatic track.
    pub fn counts(&self) -> [Option<usize>; TRACK_COUNT] {
        std::array::from_fn(|index| self.tracks[index].as_ref().map(Vec::len))
    }

    /// Seed every track from the clip's detection, which is what the editor
    /// starts from: the reference's `automatic_contacts` copied into `contacts`
    /// the first time a track is touched.
    pub fn seeded(feet: [[Interval; 2]; 2], hands: [Vec<Interval>; 2]) -> Self {
        Self {
            tracks: [
                Some(feet[0].to_vec()),
                Some(feet[1].to_vec()),
                Some(hands[0].clone()),
                Some(hands[1].clone()),
            ],
        }
    }

    /// Copy the detection into every track that still follows it.
    ///
    /// This is the reference's first-touch rule (`footlock_editor`: `contacts`
    /// starts as a copy of `automatic_contacts`), and it is what makes a painted
    /// lock *add* to the detector's runs. Without it, the first click on a foot
    /// row replaced the whole foot's contacts with one frame: every other frame
    /// the detector had found became "airborne", the correction stopped pinning
    /// it, and the foot then floated wherever the network left it (12 mm on the
    /// clip this was found on) while looking like a lock that had been ignored.
    pub fn seed_from(&mut self, detected: &[Vec<Interval>; TRACK_COUNT]) {
        for (index, track) in self.tracks.iter_mut().enumerate() {
            if track.is_none() {
                *track = Some(detected[index].clone());
            }
        }
    }

    /// Sort, clamp, drop empties and merge overlaps, exactly like the reference's
    /// `normalize_ranges` (which merges an overlapping interval into its
    /// predecessor instead of rejecting the edit).
    pub fn normalise(&mut self, frames: usize) {
        for track in self.tracks.iter_mut().flatten() {
            normalise_track(track, frames);
        }
    }

    /// A normalised copy.
    pub fn normalised(&self, frames: usize) -> Self {
        let mut out = self.clone();
        out.normalise(frames);
        out
    }

    /// Whether `frame` is inside one of the track's intervals.
    pub fn active(&self, index: usize, frame: usize) -> bool {
        self.track(index)
            .map(|track| track.iter().any(|[start, end]| *start <= frame && frame < *end))
            .unwrap_or(false)
    }

    /// The track as `(start, end)` runs, bounded to `frames`.
    pub fn runs(&self, index: usize, frames: usize) -> Option<Vec<(usize, usize)>> {
        self.track(index).map(|track| {
            track
                .iter()
                .filter_map(|[start, end]| {
                    let (start, end) = (*start.min(&frames), *end.min(&frames));
                    (end > start).then_some((start, end))
                })
                .collect()
        })
    }

    /// The lock document for a clip: the time base is what makes a saved edit
    /// safe to ignore when the clip has changed underneath it.
    pub fn document(&self, frames: usize, fps: f64) -> LockDocument {
        LockDocument {
            version: LOCK_VERSION,
            frames,
            fps,
            tracks: self.clone(),
        }
    }

    /// JSON text of the document.
    ///
    /// Hand-built rather than `serde`'s pretty printer, which puts every frame
    /// number on its own line: one track per line is what a lock document has to
    /// look like if anyone is going to edit one by hand (`--dump-locks` writes
    /// one, `--locks` reads it back).
    pub fn to_json(&self, frames: usize, fps: f64) -> String {
        let mut out = String::from("{\n");
        out.push_str(&format!("  \"version\": {LOCK_VERSION},\n"));
        out.push_str(&format!("  \"frames\": {frames},\n"));
        out.push_str(&format!("  \"fps\": {fps},\n"));
        out.push_str("  \"tracks\": [\n");
        for (index, track) in self.tracks.iter().enumerate() {
            if index > 0 {
                out.push_str(",\n");
            }
            out.push_str("    ");
            match track {
                None => out.push_str("null"),
                Some(list) => {
                    out.push('[');
                    for (slot, [start, end]) in list.iter().enumerate() {
                        if slot > 0 {
                            out.push_str(", ");
                        }
                        out.push_str(&format!("[{start}, {end}]"));
                    }
                    out.push(']');
                }
            }
        }
        out.push_str("\n  ]\n}\n");
        out
    }

    /// Read a saved document and check it belongs to this clip.
    pub fn from_json(text: &str, frames: usize, fps: f64) -> Result<Self, String> {
        let document: LockDocument =
            serde_json::from_str(text).map_err(|error| format!("invalid contact locks: {error}"))?;
        if document.version != LOCK_VERSION {
            return Err(format!(
                "contact locks are version {}, this build writes {LOCK_VERSION}",
                document.version
            ));
        }
        if document.frames != frames || (document.fps - fps).abs() > 1e-6 {
            return Err(format!(
                "these locks were saved for {} frames at {:.3} fps, this clip has {frames} at {fps:.3}",
                document.frames, document.fps
            ));
        }
        Ok(document.tracks.normalised(frames))
    }

    /// Load the locks saved next to a clip; a missing file is not an error, and
    /// a stale one is reported instead of silently ignored (the reference prints
    /// the same warning: the saved edits belong to older motion).
    pub fn load(path: &Path, frames: usize, fps: f64) -> (Self, Option<String>) {
        let Ok(text) = std::fs::read_to_string(path) else {
            return (Self::automatic(), None);
        };
        match Self::from_json(&text, frames, fps) {
            Ok(locks) => (locks, None),
            Err(note) => (Self::automatic(), Some(note)),
        }
    }

    /// Write the locks next to a clip, atomically: a half-written file would be
    /// read back as a corrupt edit on the next start.
    pub fn save(&self, path: &Path, frames: usize, fps: f64) -> Result<()> {
        let text = self.to_json(frames, fps);
        let temporary = path.with_extension("locks.json.tmp");
        std::fs::write(&temporary, text)
            .with_context(|| format!("writing {}", temporary.display()))?;
        std::fs::rename(&temporary, path)
            .with_context(|| format!("moving {} into place", temporary.display()))?;
        Ok(())
    }
}

/// Sort, clamp and merge one track in place.
fn normalise_track(track: &mut Vec<Interval>, frames: usize) {
    let mut sorted: Vec<Interval> = track
        .iter()
        .map(|[start, end]| [(*start).min(frames), (*end).min(frames)])
        .filter(|[start, end]| end > start)
        .collect();
    sorted.sort_unstable();
    let mut merged: Vec<Interval> = Vec::with_capacity(sorted.len());
    for [start, end] in sorted {
        match merged.last_mut() {
            // The reference merges an overlapping interval into its predecessor
            // and keeps the farther end; a touching interval stays separate.
            Some(last) if start < last[1] => last[1] = last[1].max(end),
            _ => merged.push([start, end]),
        }
    }
    *track = merged;
}

/// Version of the file layout on disk.
pub const LOCK_VERSION: u32 = 1;

/// Everything saved next to a clip.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LockDocument {
    pub version: u32,
    pub frames: usize,
    pub fps: f64,
    #[serde(flatten)]
    pub tracks: ContactTracks,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn automatic_means_every_track_follows_the_clip() {
        let locks = ContactTracks::automatic();
        assert!(locks.is_automatic());
        for index in 0..TRACK_COUNT {
            assert_eq!(locks.track(index), None);
            assert_eq!(locks.runs(index, 100), None);
            assert!(!locks.active(index, 10));
        }
    }

    /// The reference's `normalize_ranges`: sorted, clamped, overlaps merged,
    /// empties dropped, and a touching pair kept as two locks.
    #[test]
    fn normalising_matches_the_reference() {
        let mut locks = ContactTracks::automatic();
        locks.set_track(0, vec![[10, 20], [5, 8], [15, 25], [30, 30], [40, 1000]]);
        locks.set_track(1, vec![[20, 25], [25, 30], [60, 50]]);
        locks.set_track(2, vec![[0, 1000]]);
        locks.normalise(120);
        assert_eq!(
            locks.track(0).unwrap(),
            &[[5, 8], [10, 25], [40, 120]],
            "sorted, merged and clamped"
        );
        assert_eq!(
            locks.track(1).unwrap(),
            &[[20, 25], [25, 30]],
            "a touching pair stays two locks; the empty interval is dropped"
        );
        assert_eq!(locks.track(2).unwrap(), &[[0, 120]]);
        assert_eq!(locks.track(3), None, "untouched tracks stay automatic");
    }

    #[test]
    fn an_emptied_track_is_not_an_automatic_track() {
        let mut locks = ContactTracks::automatic();
        locks.set_track(RIGHT_FOOT, Vec::new());
        assert!(!locks.is_automatic());
        assert_eq!(locks.runs(RIGHT_FOOT, 60), Some(Vec::new()));
        assert_eq!(locks.runs(LEFT_FOOT, 60), None);
        assert_eq!(locks.counts()[RIGHT_FOOT], Some(0));
        locks.reset_track(RIGHT_FOOT);
        assert!(locks.is_automatic());
    }

    #[test]
    fn active_is_half_open() {
        let mut locks = ContactTracks::automatic();
        locks.set_track(LEFT_FOOT, vec![[10, 12]]);
        assert!(!locks.active(LEFT_FOOT, 9));
        assert!(locks.active(LEFT_FOOT, 10));
        assert!(locks.active(LEFT_FOOT, 11));
        assert!(!locks.active(LEFT_FOOT, 12), "end is exclusive");
        assert!(!locks.active(RIGHT_FOOT, 10));
    }

    /// The first edit starts from the detection: an automatic track takes a copy
    /// of the detector's runs, an edited one is left alone, and an empty
    /// detection stays an empty (but manual) track.
    #[test]
    fn seeding_starts_the_editor_from_the_detection() {
        let detected: [Vec<Interval>; TRACK_COUNT] = [
            vec![[0, 30], [40, 60]],
            vec![[10, 40]],
            Vec::new(),
            Vec::new(),
        ];
        let mut locks = ContactTracks::automatic();
        locks.set_track(RIGHT_FOOT, vec![[50, 52]]);
        locks.seed_from(&detected);
        assert_eq!(locks.track(LEFT_FOOT), Some(&[[0, 30], [40, 60]][..]));
        assert_eq!(
            locks.track(RIGHT_FOOT),
            Some(&[[50, 52]][..]),
            "an edited track must not be overwritten"
        );
        assert_eq!(locks.track(LEFT_HAND), Some(&[][..]));
    }

    #[test]
    fn runs_are_bounded_to_the_clip() {
        let mut locks = ContactTracks::automatic();
        locks.set_track(LEFT_FOOT, vec![[0, 5], [8, 500]]);
        assert_eq!(locks.runs(LEFT_FOOT, 60), Some(vec![(0, 5), (8, 60)]));
        assert_eq!(locks.runs(LEFT_FOOT, 4), Some(vec![(0, 4)]));
    }

    /// A saved document round-trips, and one saved for another clip is refused
    /// with the note the interface shows instead of applying stale locks.
    #[test]
    fn the_document_is_tied_to_its_clip() {        let mut locks = ContactTracks::seeded(
            [[[0, 10], [20, 30]], [[5, 6], [40, 44]]],
            [vec![[1, 2]], vec![]],
        );
        locks.set_track(LEFT_HAND, vec![[7, 9]]);
        let text = locks.to_json(50, 30.0);
        let read = ContactTracks::from_json(&text, 50, 30.0).expect("round trip");
        assert_eq!(read, locks.normalised(50));
        assert_eq!(read.track(LEFT_HAND).unwrap(), &[[7, 9]]);
        assert_eq!(read.track(RIGHT_HAND).unwrap(), &[] as &[Interval]);

        let error = ContactTracks::from_json(&text, 51, 30.0).expect_err("frame count moved");
        assert!(error.contains("50 frames"), "unhelpful note: {error}");
        let error = ContactTracks::from_json(&text, 50, 25.0).expect_err("fps moved");
        assert!(error.contains("30.000"), "unhelpful note: {error}");
        assert!(ContactTracks::from_json("not json", 50, 30.0).is_err());
    }

    /// The document has to stay editable by hand: one track per line, intervals
    /// compact, and it still parses back to the same thing.
    #[test]
    fn the_document_is_one_line_per_track() {
        let mut locks = ContactTracks::automatic();
        locks.set_track(LEFT_FOOT, vec![[0, 8], [12, 20]]);
        locks.set_track(RIGHT_HAND, vec![[4, 6]]);
        let text = locks.to_json(48, 24.0);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(
            lines.len(),
            11,
            "four keys, four tracks, and the brackets:\n{text}"
        );
        assert!(lines[5].contains("[[0, 8], [12, 20]]"), "{}", lines[5]);
        assert_eq!(lines[6].trim(), "null,", "an untouched track stays null");
        assert!(lines[8].contains("[[4, 6]]"), "{}", lines[8]);
        assert_eq!(ContactTracks::from_json(&text, 48, 24.0).unwrap(), locks);
    }

    #[test]
    fn saving_and_loading_next_to_a_clip() {
        let path = std::env::temp_dir().join("velomocap-locks-test.locks.json");
        let mut locks = ContactTracks::automatic();
        locks.set_track(LEFT_FOOT, vec![[3, 9]]);
        locks.save(&path, 40, 30.0).expect("save");
        let (loaded, note) = ContactTracks::load(&path, 40, 30.0);
        assert_eq!(note, None);
        assert_eq!(loaded, locks);

        let (stale, note) = ContactTracks::load(&path, 41, 30.0);
        assert!(note.is_some(), "a stale document has to say so");
        assert!(stale.is_automatic());

        let missing = std::env::temp_dir().join("velomocap-no-such-locks.json");
        let (automatic, note) = ContactTracks::load(&missing, 40, 30.0);
        assert!(note.is_none(), "a missing file is not a warning");
        assert!(automatic.is_automatic());
        let _ = std::fs::remove_file(&path);
    }
}
