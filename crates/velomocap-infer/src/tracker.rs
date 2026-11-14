//! Person tracking across frames.
//!
//! Greedy IoU association with a short coast window: a track survives a few
//! frames of missed detection (occlusion, motion blur) before it is closed.
//! This is deliberately simple and deterministic — the original application
//! uses the same idea, and a learned ReID model can be dropped in later behind
//! the same `Tracker::update` call.

use crate::detector::Detection;

/// One tracked person.
#[derive(Clone, Debug)]
pub struct Track {
    pub id: usize,
    /// Detections in frame order, `None` while the track is coasting.
    pub boxes: Vec<Option<Detection>>,
    pub first_frame: usize,
    pub last_frame: usize,
    /// Consecutive frames without a detection.
    pub missing: usize,
    /// Total frames in which this track was actually detected.
    pub hits: usize,
    pub score_sum: f32,
}

impl Track {
    pub fn mean_score(&self) -> f32 {
        if self.hits == 0 {
            0.0
        } else {
            self.score_sum / self.hits as f32
        }
    }

    pub fn detection(&self, frame: usize) -> Option<&Detection> {
        self.boxes.get(frame).and_then(|slot| slot.as_ref())
    }

    /// Interpolated box at a frame where the track was coasting.
    pub fn box_at(&self, frame: usize) -> Option<Detection> {
        if let Some(Some(detection)) = self.boxes.get(frame) {
            return Some(*detection);
        }
        let before = (0..frame)
            .rev()
            .find_map(|i| self.boxes.get(i).and_then(|s| s.as_ref()).copied());
        let after = (frame + 1..self.boxes.len())
            .find_map(|i| self.boxes.get(i).and_then(|s| s.as_ref()).copied());
        match (before, after) {
            (Some(a), Some(b)) => {
                let span = {
                    // Weight by the distance to each observed box.
                    let mut left = frame;
                    while left > 0 && self.boxes[left].is_none() {
                        left -= 1;
                    }
                    let mut right = frame;
                    while right + 1 < self.boxes.len() && self.boxes[right].is_none() {
                        right += 1;
                    }
                    (right - left).max(1) as f32
                };
                let t = ((frame - self.first_frame) as f32 / span).clamp(0.0, 1.0);
                Some(Detection {
                    x0: a.x0 + (b.x0 - a.x0) * t,
                    y0: a.y0 + (b.y0 - a.y0) * t,
                    x1: a.x1 + (b.x1 - a.x1) * t,
                    y1: a.y1 + (b.y1 - a.y1) * t,
                    score: a.score.min(b.score),
                    class_id: a.class_id,
                })
            }
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        }
    }
}

pub struct Tracker {
    iou_threshold: f32,
    max_missing: usize,
    next_id: usize,
    pub tracks: Vec<Track>,
    frame: usize,
}

impl Tracker {
    pub fn new(iou_threshold: f32, max_missing: usize) -> Self {
        Self {
            iou_threshold,
            max_missing: max_missing.max(1),
            next_id: 0,
            tracks: Vec::new(),
            frame: 0,
        }
    }

    /// Feed one frame's detections; returns how many tracks are active.
    pub fn update(&mut self, detections: &[Detection]) -> usize {
        let frame = self.frame;

        // Grow every live track to the current frame.
        for track in self.tracks.iter_mut() {
            while track.boxes.len() <= frame {
                track.boxes.push(None);
            }
        }

        // Greedy best-first assignment by IoU.
        let mut pairs: Vec<(f32, usize, usize)> = Vec::new();
        for (ti, track) in self.tracks.iter().enumerate() {
            if track.missing > self.max_missing {
                continue;
            }
            let Some(previous) = track.boxes[frame.saturating_sub(1).min(track.boxes.len() - 1)]
                .or_else(|| track.boxes.iter().rev().flatten().next().copied())
            else {
                continue;
            };
            for (di, detection) in detections.iter().enumerate() {
                let iou = previous.iou(detection);
                if iou >= self.iou_threshold {
                    pairs.push((iou, ti, di));
                }
            }
        }
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut used_tracks = vec![false; self.tracks.len()];
        let mut used_detections = vec![false; detections.len()];
        for (_, ti, di) in pairs {
            if used_tracks[ti] || used_detections[di] {
                continue;
            }
            used_tracks[ti] = true;
            used_detections[di] = true;
            let track = &mut self.tracks[ti];
            track.boxes[frame] = Some(detections[di]);
            track.last_frame = frame;
            track.missing = 0;
            track.hits += 1;
            track.score_sum += detections[di].score;
        }

        for (ti, track) in self.tracks.iter_mut().enumerate() {
            if !used_tracks[ti] && track.missing <= self.max_missing {
                track.missing += 1;
            }
        }

        // Start tracks for unmatched detections.
        for (di, detection) in detections.iter().enumerate() {
            if used_detections[di] {
                continue;
            }
            let mut boxes = vec![None; frame];
            boxes.push(Some(*detection));
            self.tracks.push(Track {
                id: self.next_id,
                boxes,
                first_frame: frame,
                last_frame: frame,
                missing: 0,
                hits: 1,
                score_sum: detection.score,
            });
            self.next_id += 1;
        }

        self.frame += 1;
        self.tracks.iter().filter(|t| t.missing <= self.max_missing).count()
    }

    pub fn frames_processed(&self) -> usize {
        self.frame
    }

    /// Drop tracks that are too short to be real people.
    pub fn prune(&mut self, min_hits: usize) {
        self.tracks.retain(|track| track.hits >= min_hits);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(x0: f32, y0: f32, x1: f32, y1: f32, score: f32) -> Detection {
        Detection {
            x0,
            y0,
            x1,
            y1,
            score,
            class_id: 0,
        }
    }

    #[test]
    fn tracks_a_moving_person_across_frames() {
        let mut tracker = Tracker::new(0.3, 2);
        for step in 0..5 {
            let offset = step as f32 * 2.0;
            tracker.update(&[det(offset, 0.0, 100.0 + offset, 200.0, 0.9)]);
        }
        assert_eq!(tracker.tracks.len(), 1);
        assert_eq!(tracker.tracks[0].hits, 5);
        assert_eq!(tracker.tracks[0].id, 0);
    }

    #[test]
    fn coasts_through_a_missed_detection() {
        let mut tracker = Tracker::new(0.3, 2);
        tracker.update(&[det(0.0, 0.0, 100.0, 200.0, 0.9)]);
        tracker.update(&[]);
        tracker.update(&[det(0.0, 0.0, 100.0, 200.0, 0.9)]);
        assert_eq!(tracker.tracks.len(), 1);
        assert_eq!(tracker.tracks[0].hits, 2);
        assert!(tracker.tracks[0].box_at(1).is_some());
    }

    #[test]
    fn separates_two_people() {
        let mut tracker = Tracker::new(0.3, 2);
        for _ in 0..3 {
            tracker.update(&[
                det(0.0, 0.0, 100.0, 200.0, 0.9),
                det(400.0, 0.0, 500.0, 200.0, 0.8),
            ]);
        }
        assert_eq!(tracker.tracks.len(), 2);
    }
}
