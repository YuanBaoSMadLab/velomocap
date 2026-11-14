//! Contact planning: sole support geometry, stance segmentation and grounding.
//!
//! Port of `mocap/correction.py::prepare_contacts` (flat mode),
//! `mocap/contact_trajectory.py` and the reachability helpers from
//! `mocap/anatomical_ik.py`.
//!
//! Everything is single precision compatible with the Python pipeline but
//! computed in f64 here, matching NumPy's default promotion.  Array layout is
//! `(frame, foot, ...)` flattened row-major, exactly like the NumPy original.

use crate::error::{Error, Result};
use crate::filters;
use crate::math::{self, align, cross3, dot3};

/// Locking presets, identical to `mocap/correction_settings.py::PRESETS`.
#[derive(Clone, Copy, Debug)]
pub struct Preset {
    pub confidence: f64,
    pub release: f64,
    pub minimum: f64,
    pub anchor_drift: f64,
    pub strength: f64,
    pub gap: f64,
}

impl Preset {
    pub fn named(name: &str) -> Result<Self> {
        match name {
            "weak" => Ok(Self {
                confidence: 0.75,
                release: 0.75,
                minimum: 0.06,
                anchor_drift: 0.12,
                strength: 0.6,
                gap: 0.0,
            }),
            "normal" => Ok(Self {
                confidence: 0.65,
                release: 0.65,
                minimum: 0.06,
                anchor_drift: 0.12,
                strength: 1.0,
                gap: 0.0,
            }),
            "strong" => Ok(Self {
                confidence: 0.65,
                release: 0.5,
                minimum: 0.06,
                anchor_drift: 0.20,
                strength: 1.0,
                gap: 0.08,
            }),
            other => Err(Error::new(format!(
                "Unknown Motion Correction preset: {other}"
            ))),
        }
    }
}

/// One resolved stance.  Times are immutable; only the anchor may move.
#[derive(Clone, Debug, Default)]
pub struct Segment {
    pub foot: usize,
    pub start: usize,
    pub end: usize,
    pub anchor: [f64; 3],
    pub anchor_end: [f64; 3],
    pub handoff: Option<(usize, usize)>,
}

/// The committed contact plan, ready for the native solver.
#[derive(Clone, Debug, Default)]
pub struct ContactPlan {
    /// `frames * 2` support flags, `[t * 2 + foot]`.
    pub active: Vec<bool>,
    /// `frames * 2 * 3` planted sole anchors.
    pub anchors: Vec<f64>,
    /// `frames * 2 * 3` observed sole support points.
    pub offsets: Vec<f64>,
    /// `frames * 3` global grounding shift applied to the root.
    pub shift: Vec<f64>,
    pub floor: f64,
    pub segments: Vec<Segment>,
    pub frames: usize,
}

impl ContactPlan {
    #[inline]
    pub fn is_active(&self, t: usize, foot: usize) -> bool {
        self.active[t * 2 + foot]
    }

    pub fn active_any(&self, t: usize) -> bool {
        self.active[t * 2] || self.active[t * 2 + 1]
    }
}

fn column(mask: &[bool], foot: usize) -> Vec<bool> {
    mask.iter().skip(foot).step_by(2).cloned().collect()
}

fn set_column(mask: &mut [bool], foot: usize, source: &[bool]) {
    for (i, value) in source.iter().enumerate() {
        mask[i * 2 + foot] = *value;
    }
}

/// Plan contacts and grounding for a take.
///
/// * `probes` - `frames * 4 * 3` heel/forefoot sole samples.
/// * `probabilities` - `frames * 4` contact confidence.
/// * `floor` - `Some(h)` fixes the plane, `None` estimates it from the take.
/// * `manual` - optional `[foot][(start, end), ...]` user stance ranges.
pub fn prepare_contacts(
    probes: &[f64],
    probabilities: &[f64],
    frames: usize,
    fps: f64,
    floor: Option<f64>,
    preset: &Preset,
    manual: Option<&[Vec<(usize, usize)>; 2]>,
) -> Result<ContactPlan> {
    if frames == 0 {
        return Err(Error::new("Invalid sole/contact dimensions"));
    }
    if probabilities.len() != frames * 4 || probes.len() != frames * 4 * 3 {
        return Err(Error::new("Invalid sole/contact dimensions"));
    }
    if !fps.is_finite() || fps <= 0.0 {
        return Err(Error::new("Invalid contacts, positions or frame rate"));
    }
    if !probes.iter().all(|v| v.is_finite()) || !probabilities.iter().all(|v| v.is_finite()) {
        return Err(Error::new("Invalid contacts, positions or frame rate"));
    }
    if probabilities.iter().any(|&p| p < 0.0 || p > 1.0) {
        return Err(Error::new("Contact probabilities must be in [0,1]"));
    }
    let n = frames;

    // Per-frame foot geometry: lowest sole sample and mean patch point.
    let mut height = vec![0.0; n * 2];
    let mut points = vec![0.0; n * 2 * 3];
    for t in 0..n {
        for foot in 0..2 {
            let base = (t * 4 + foot * 2) * 3;
            let h0 = probes[base + 1];
            let h1 = probes[base + 4];
            height[t * 2 + foot] = h0.min(h1);
            for axis in 0..3 {
                points[(t * 2 + foot) * 3 + axis] =
                    0.5 * (probes[base + axis] + probes[base + 3 + axis]);
            }
        }
    }

    let mut speed = vec![0.0; n * 2];
    let mut vertical = vec![0.0; n * 2];
    if n > 1 {
        for t in 1..n {
            for foot in 0..2 {
                let dx = points[(t * 2 + foot) * 3] - points[((t - 1) * 2 + foot) * 3];
                let dz = points[(t * 2 + foot) * 3 + 2] - points[((t - 1) * 2 + foot) * 3 + 2];
                speed[t * 2 + foot] = (dx * dx + dz * dz).sqrt() * fps;
                vertical[t * 2 + foot] =
                    (height[t * 2 + foot] - height[(t - 1) * 2 + foot]).abs() * fps;
            }
        }
        for foot in 0..2 {
            speed[foot] = speed[2 + foot];
            vertical[foot] = vertical[2 + foot];
        }
    }

    let mut confidence = vec![0.0; n * 2];
    for t in 0..n {
        for foot in 0..2 {
            let base = t * 4 + foot * 2;
            confidence[t * 2 + foot] = probabilities[base].max(probabilities[base + 1]);
        }
    }

    let mut geometry = vec![false; n * 2];
    for t in 0..n {
        let row_min = height[t * 2].min(height[t * 2 + 1]);
        for foot in 0..2 {
            geometry[t * 2 + foot] = speed[t * 2 + foot] < 0.7
                && vertical[t * 2 + foot] < 0.8
                && (height[t * 2 + foot] - row_min) < 0.12;
        }
    }

    let mut active: Vec<bool> = (0..n * 2)
        .map(|i| confidence[i] > preset.confidence && geometry[i])
        .collect();

    if preset.gap > 0.0 {
        extend_gapped_stances(
            &mut active, &geometry, &confidence, &height, &points, speed.as_slice(), floor, fps,
            preset,
        );
    }

    let minimum = std::cmp::max(2, (fps * preset.minimum).round_ties_even() as usize);
    for foot in 0..2 {
        let mut col = column(&active, foot);
        for (first, end) in filters::intervals(&col) {
            if end - first < minimum {
                for slot in col.iter_mut().take(end).skip(first) {
                    *slot = false;
                }
            }
        }
        set_column(&mut active, foot, &col);
    }

    let flight = airborne_excursions(probes, frames, fps);
    for (t, is_flight) in flight.iter().enumerate() {
        if *is_flight {
            active[t * 2] = false;
            active[t * 2 + 1] = false;
        }
    }

    if let Some(ranges) = manual {
        for value in active.iter_mut() {
            *value = false;
        }
        for foot in 0..2 {
            for &(first, end) in &ranges[foot] {
                let first = first.min(n);
                let end = end.min(n);
                for t in first..end {
                    active[t * 2 + foot] = true;
                }
            }
        }
    }

    // Floor estimate from observed support heights.
    let mut values: Vec<f64> = (0..n * 2).filter(|&i| active[i]).map(|i| height[i]).collect();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let floor = match floor {
        Some(value) => value,
        None => {
            if !values.is_empty() {
                filters::quantile_linear(&values, 0.05)
            } else {
                height.iter().cloned().fold(f64::INFINITY, f64::min)
            }
        }
    };
    if !floor.is_finite() {
        return Err(Error::new("Floor height must be finite"));
    }

    // Grounding target: correct support frames, interpolate through flight.
    let mut support = vec![false; n];
    for t in 0..n {
        support[t] = active[t * 2] || active[t * 2 + 1];
    }
    let mut target = vec![0.0; n];
    if support.iter().any(|&s| s) {
        let support_height: Vec<f64> = (0..n)
            .map(|t| {
                let a = if active[t * 2] { height[t * 2] } else { f64::INFINITY };
                let b = if active[t * 2 + 1] { height[t * 2 + 1] } else { f64::INFINITY };
                a.min(b)
            })
            .collect();
        let support_indices: Vec<usize> = (0..n).filter(|&t| support[t]).collect();
        let desired: Vec<f64> = support_indices
            .iter()
            .map(|&t| (floor - support_height[t]).clamp(-0.75, 0.75))
            .collect();
        target = filters::interp_on_samples(&support_indices, &desired, n);
        if n >= 3 {
            let weight: Vec<f64> = (0..n).map(|t| if support[t] { 1.0 } else { 0.02 }).collect();
            let smooth = 4.0 * (fps / 30.0).powi(4);
            let mut diagonal = vec![0.0; n];
            for t in 0..n {
                diagonal[t] = weight[t]
                    + smooth
                        * if n >= 4 {
                            if t == 0 || t == n - 1 {
                                1.0
                            } else if t == 1 || t == n - 2 {
                                5.0
                            } else {
                                6.0
                            }
                        } else if t == 1 {
                            4.0
                        } else {
                            1.0
                        };
            }
            let first: Vec<f64> = (0..n - 1)
                .map(|t| {
                    -2.0 * smooth * if t == 0 || t == n - 2 { 1.0 } else { 2.0 }
                })
                .collect();
            let second: Vec<f64> = vec![smooth; n.saturating_sub(2)];
            let mut rhs = vec![
                (0..n)
                    .map(|t| weight[t] * target[t])
                    .collect::<Vec<f64>>(),
            ];
            filters::solve_symmetric_banded(2, &diagonal, &[first, second], &mut rhs);
            target = rhs.remove(0);
        }
    }

    // Sole support points and planted anchors.
    let weights = support_weights(probes, &active, frames, fps);
    let offsets = support_samples(probes, &weights, frames);
    let mut anchors = offsets.clone();
    let mut segments: Vec<Segment> = Vec::new();

    for foot in 0..2 {
        let col = column(&active, foot);
        for (first, end) in filters::intervals(&col) {
            let mut start = first;
            while start < end {
                let mut stop = start + 1;
                while stop < end
                    && (manual.is_some() || {
                        let dx = offsets[(stop * 2 + foot) * 3] - offsets[(start * 2 + foot) * 3];
                        let dz =
                            offsets[(stop * 2 + foot) * 3 + 2] - offsets[(start * 2 + foot) * 3 + 2];
                        (dx * dx + dz * dz).sqrt() < preset.anchor_drift
                    })
                {
                    stop += 1;
                }
                if manual.is_none() && end - stop < minimum {
                    stop = end;
                }
                let path = rolling_anchor(&probes, &weights, foot, start, stop, floor);
                for (i, t) in (start..stop).enumerate() {
                    anchors[(t * 2 + foot) * 3..(t * 2 + foot) * 3 + 3]
                        .copy_from_slice(&path[i * 3..i * 3 + 3]);
                }
                segments.push(Segment {
                    foot,
                    start,
                    end: stop,
                    anchor: [path[0], path[1], path[2]],
                    anchor_end: [
                        path[(stop - start - 1) * 3],
                        path[(stop - start - 1) * 3 + 1],
                        path[(stop - start - 1) * 3 + 2],
                    ],
                    handoff: None,
                });
                start = stop;
            }
        }
    }

    if manual.is_some() {
        couple_anchors(&mut anchors, &offsets, &mut segments, frames, fps);
    } else {
        smooth_auto_handoffs(&mut anchors, &probes, &weights, &mut segments, frames, fps);
    }

    // Global grounding shift: vertical target plus a small XZ drift.
    let mut shift = vec![0.0; frames * 3];
    for t in 0..n {
        shift[t * 3 + 1] = target[t].clamp(-0.75, 0.75);
    }
    if support.iter().any(|&s| s) {
        let mut drift = vec![0.0; n * 3];
        for t in 0..n {
            let count = (active[t * 2] as usize + active[t * 2 + 1] as usize).max(1) as f64;
            for axis in 0..3 {
                let mut sum = 0.0;
                for foot in 0..2 {
                    if active[t * 2 + foot] {
                        sum += anchors[(t * 2 + foot) * 3 + axis]
                            - offsets[(t * 2 + foot) * 3 + axis];
                    }
                }
                drift[t * 3 + axis] = sum / count;
            }
        }
        let support_indices: Vec<usize> = (0..n).filter(|&t| support[t]).collect();
        let sigma = (fps * 0.06).max(1.0);
        for axis in [0usize, 2] {
            let sampled: Vec<f64> = support_indices.iter().map(|&t| drift[t * 3 + axis]).collect();
            let values = filters::interp_on_samples(&support_indices, &sampled, n);
            let smoothed = filters::gaussian_filter1d(&values, sigma);
            for t in 0..n {
                shift[t * 3 + axis] = smoothed[t];
            }
        }
        for t in 0..n {
            let nx = shift[t * 3];
            let nz = shift[t * 3 + 2];
            let norm = (nx * nx + nz * nz).sqrt();
            let scale = 1.0f64.min(0.12 / norm.max(1e-8));
            shift[t * 3] *= scale;
            shift[t * 3 + 2] *= scale;
        }
    }

    Ok(ContactPlan {
        active,
        anchors,
        offsets,
        shift,
        floor,
        segments,
        frames,
    })
}

/// Validate and merge user stance ranges exactly like
/// `mocap/foot_contacts.py::normalize_ranges`.
pub fn normalize_ranges(
    ranges: &[Vec<(usize, usize)>; 2],
    frames: usize,
) -> Result<[Vec<(usize, usize)>; 2]> {
    let mut out: [Vec<(usize, usize)>; 2] = [Vec::new(), Vec::new()];
    for (foot, track) in ranges.iter().enumerate() {
        let mut sorted = track.clone();
        sorted.sort_by_key(|&(start, end)| (start, end));
        let mut normalized: Vec<(usize, usize)> = Vec::with_capacity(sorted.len());
        for (start, end) in sorted {
            if start >= end || end > frames {
                return Err(Error::new(
                    "Foot-lock intervals must be within the motion frame range",
                ));
            }
            if let Some(last) = normalized.last_mut() {
                if start < last.1 {
                    last.1 = last.1.max(end);
                    continue;
                }
            }
            normalized.push((start, end));
        }
        out[foot] = normalized;
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn extend_gapped_stances(
    active: &mut [bool],
    geometry: &[bool],
    confidence: &[f64],
    height: &[f64],
    points: &[f64],
    speed: &[f64],
    floor: Option<f64>,
    fps: f64,
    preset: &Preset,
) {
    let n = height.len() / 2;
    let support_height: Vec<f64> = (0..n * 2).filter(|&i| active[i]).map(|i| height[i]).collect();
    let reference = if floor.is_none() && !support_height.is_empty() {
        filters::quantile_linear(&support_height, 0.05)
    } else if floor.is_none() {
        height.iter().cloned().fold(f64::INFINITY, f64::min)
    } else {
        floor.unwrap()
    };
    let near: Vec<bool> = height.iter().map(|h| (h - reference).abs() < 0.04).collect();
    let holding: Vec<bool> = (0..n * 2)
        .map(|i| confidence[i] > preset.release && geometry[i] && near[i] && speed[i] < 0.25)
        .collect();

    for foot in 0..2 {
        let mut col = column(active, foot);
        let hold = column(&holding, foot);
        let combined: Vec<bool> = (0..n).map(|t| col[t] || hold[t]).collect();
        for (first, end) in filters::intervals(&combined) {
            if let Some(seed) = (first..end).find(|&t| col[t]) {
                for slot in col.iter_mut().take(end).skip(seed) {
                    *slot = true;
                }
            }
        }
        // Close a brief dropout only between two existing stances.
        let inverse: Vec<bool> = col.iter().map(|v| !v).collect();
        for (first, end) in filters::intervals(&inverse) {
            if first == 0 || end == n || (end - first) as f64 > (fps * preset.gap).round_ties_even() {
                continue;
            }
            let geom = column(geometry, foot);
            let near_col = column(&near, foot);
            let ok = (first..end).all(|t| geom[t] && near_col[t]);
            if !ok {
                continue;
            }
            let dx = points[(end * 2 + foot) * 3] - points[((first - 1) * 2 + foot) * 3];
            let dz = points[(end * 2 + foot) * 3 + 2] - points[((first - 1) * 2 + foot) * 3 + 2];
            if (dx * dx + dz * dz).sqrt() > 0.04 {
                continue;
            }
            for slot in col.iter_mut().take(end).skip(first) {
                *slot = true;
            }
        }
        set_column(active, foot, &col);
    }
}

/// Reject bounded two-foot flight arcs without grounding away their apex.
pub fn airborne_excursions(probes: &[f64], frames: usize, fps: f64) -> Vec<bool> {
    let mut flight = vec![false; frames];
    if frames < 5 {
        return flight;
    }
    let n = frames;
    let mut h = vec![0.0; n];
    for t in 0..n {
        let base = t * 4 * 3;
        h[t] = probes[base + 1]
            .min(probes[base + 4])
            .min(probes[base + 7])
            .min(probes[base + 10]);
    }
    let reach = std::cmp::min(n, std::cmp::max(2, (fps * 4.0).round_ties_even() as usize));
    let baseline = filters::minimum_filter1d(&h, 2 * reach + 1);
    let raised: Vec<f64> = (0..n).map(|t| h[t] - baseline[t]).collect();
    let smoothed = filters::gaussian_filter1d(&h, (fps * 0.03).max(0.5));
    let gradient = filters::gradient1d(&smoothed);
    let velocity: Vec<f64> = gradient.iter().map(|v| v * fps).collect();

    for (first, end) in filters::intervals(&raised.iter().map(|&v| v > 0.15).collect::<Vec<_>>()) {
        let mut peak = first;
        for t in first..end {
            if smoothed[t] > smoothed[peak] {
                peak = t;
            }
        }
        let lo = peak.saturating_sub(reach);
        let hi = std::cmp::min(n, peak + reach + 1);
        if peak == lo || peak + 1 == hi {
            continue;
        }
        let left = smoothed[lo..peak].iter().cloned().fold(f64::INFINITY, f64::min);
        let right = smoothed[peak + 1..hi]
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let top = smoothed[peak];
        if top - left.max(right) < 0.15 {
            continue;
        }
        let before: Vec<usize> = (lo..peak).filter(|&t| smoothed[t] < left + 0.06).collect();
        let after: Vec<usize> = (peak + 1..hi)
            .filter(|&t| smoothed[t] < right + 0.06)
            .collect();
        if before.is_empty() || after.is_empty() {
            continue;
        }
        let start = before[before.len() - 1] + 1;
        let mut stop = after[0];
        let descent: Vec<f64> = velocity[peak + 1..=stop].to_vec();
        let accumulated = filters::minimum_accumulate(&descent);
        let mut landed: Option<usize> = None;
        for (k, &value) in descent.iter().enumerate() {
            if accumulated[k] < -0.8 && value > -0.35 && top - smoothed[peak + 1 + k] > 0.15 {
                landed = Some(k);
                break;
            }
        }
        if let Some(k) = landed {
            stop = peak + 1 + k;
        }
        if (stop - start) as f64 > 8.0 * fps {
            continue;
        }
        for slot in flight.iter_mut().take(stop).skip(start) {
            *slot = true;
        }
    }
    flight
}

/// Stable heel/forefoot material weights from geometry, never confidence.
pub fn support_weights(probes: &[f64], active: &[bool], frames: usize, fps: f64) -> Vec<f64> {
    let mut weights = vec![0.5; frames * 2];
    for foot in 0..2 {
        let col = column(active, foot);
        for (first, end) in filters::intervals(&col) {
            let delta: Vec<f64> = (first..end)
                .map(|t| probes[(t * 4 + foot * 2) * 3 + 1] - probes[(t * 4 + foot * 2 + 1) * 3 + 1])
                .collect();
            let smoothed = filters::gaussian_filter1d(&delta, (fps * 0.04).max(0.5));
            for (k, value) in smoothed.iter().enumerate() {
                let x = ((value.abs() - 0.005) / 0.02).clamp(0.0, 1.0);
                weights[(first + k) * 2 + foot] +=
                    0.5 * value.signum() * x * x * (3.0 - 2.0 * x);
            }
        }
    }
    weights
}

/// Blend the heel/forefoot patches into one physical support point.
pub fn support_samples(probes: &[f64], weights: &[f64], frames: usize) -> Vec<f64> {
    let mut points = vec![0.0; frames * 2 * 3];
    for t in 0..frames {
        for foot in 0..2 {
            let w = weights[t * 2 + foot];
            let base = (t * 4 + foot * 2) * 3;
            for axis in 0..3 {
                let heel = probes[base + axis];
                let fore = probes[base + 3 + axis];
                points[(t * 2 + foot) * 3 + axis] = heel * (1.0 - w) + fore * w;
            }
            let lowest = probes[base + 1].min(probes[base + 4]);
            points[(t * 2 + foot) * 3 + 1] = lowest;
        }
    }
    points
}

/// Integrate a continuous contact handoff without sliding a material pin.
fn rolling_anchor(
    probes: &[f64],
    weights: &[f64],
    foot: usize,
    start: usize,
    stop: usize,
    floor: f64,
) -> Vec<f64> {
    let count = stop - start;
    let mut path = vec![0.0; count * 3];
    let patch = |t: usize, index: usize, axis: usize| -> f64 {
        probes[((t * 4 + foot * 2 + index) * 3) + axis]
    };
    for k in 1..count {
        let previous = start + k - 1;
        let current = start + k;
        let dw = weights[current * 2 + foot] - weights[previous * 2 + foot];
        for axis in 0..3 {
            let separation = 0.5 * ((patch(current, 1, axis) - patch(current, 0, axis))
                + (patch(previous, 1, axis) - patch(previous, 0, axis)));
            path[k * 3 + axis] = path[(k - 1) * 3 + axis] + separation * dw;
        }
    }
    let mut observed = vec![0.0; count * 3];
    for k in 0..count {
        let t = start + k;
        let w = weights[t * 2 + foot];
        for axis in 0..3 {
            observed[k * 3 + axis] =
                patch(t, 0, axis) * (1.0 - w) + patch(t, 1, axis) * w;
        }
    }
    // Median of the observed-minus-integrated residual, axis by axis.
    let mut offsets = [0.0f64; 3];
    for axis in 0..3 {
        let mut column_values: Vec<f64> = (0..count)
            .map(|k| observed[k * 3 + axis] - path[k * 3 + axis])
            .collect();
        column_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        offsets[axis] = median_sorted(&column_values);
    }
    for k in 0..count {
        for axis in 0..3 {
            path[k * 3 + axis] += offsets[axis];
        }
        path[k * 3 + 1] = floor;
    }
    path
}

fn median_sorted(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    }
}

/// Avoid incompatible independent pins across a very short release gap.
fn couple_anchors(
    anchors: &mut [f64],
    support: &[f64],
    segments: &mut [Segment],
    frames: usize,
    fps: f64,
) {
    for foot in 0..2 {
        let stances: Vec<usize> = (0..segments.len())
            .filter(|&i| segments[i].foot == foot)
            .collect();
        let count = stances.len();
        if count < 2 {
            continue;
        }
        let mut diagonal = vec![1.0; count];
        let mut off = vec![0.0; count - 1];
        let mut rhs = vec![vec![0.0; count], vec![0.0; count]];
        for i in 0..count - 1 {
            let left = &segments[stances[i]];
            let right = &segments[stances[i + 1]];
            let gap = right.start as i64 - left.end as i64;
            if gap as f64 > fps * 0.12 {
                continue;
            }
            let weight = 1000.0f64.min(10.0 * (fps * 0.12 / (gap + 1).max(1) as f64).powi(3));
            let a = left.end - 1;
            let b = right.start;
            let difference = [
                (anchors[(b * 2 + foot) * 3] - support[(b * 2 + foot) * 3])
                    - (anchors[(a * 2 + foot) * 3] - support[(a * 2 + foot) * 3]),
                (anchors[(b * 2 + foot) * 3 + 2] - support[(b * 2 + foot) * 3 + 2])
                    - (anchors[(a * 2 + foot) * 3 + 2] - support[(a * 2 + foot) * 3 + 2]),
            ];
            diagonal[i] += weight;
            diagonal[i + 1] += weight;
            off[i] = -weight;
            rhs[0][i] += weight * difference[0];
            rhs[0][i + 1] -= weight * difference[0];
            rhs[1][i] += weight * difference[1];
            rhs[1][i + 1] -= weight * difference[1];
        }
        filters::solve_symmetric_banded(1, &diagonal, &[off], &mut rhs);
        for (k, &index) in stances.iter().enumerate() {
            let norm = (rhs[0][k] * rhs[0][k] + rhs[1][k] * rhs[1][k]).sqrt();
            let scale = 1.0f64.min(0.12 / norm.max(1e-10));
            let dx = rhs[0][k] * scale;
            let dz = rhs[1][k] * scale;
            let segment = &mut segments[index];
            let (start, end) = (segment.start, segment.end);
            for t in start..end.min(frames) {
                anchors[(t * 2 + foot) * 3] += dx;
                anchors[(t * 2 + foot) * 3 + 2] += dz;
            }
            segment.anchor = [
                anchors[(start * 2 + foot) * 3],
                anchors[(start * 2 + foot) * 3 + 1],
                anchors[(start * 2 + foot) * 3 + 2],
            ];
            segment.anchor_end = [
                anchors[((end - 1) * 2 + foot) * 3],
                anchors[((end - 1) * 2 + foot) * 3 + 1],
                anchors[((end - 1) * 2 + foot) * 3 + 2],
            ];
        }
    }
}

/// A drift-limit reset must not teleport an automatically chosen anchor.
fn smooth_auto_handoffs(
    anchors: &mut [f64],
    probes: &[f64],
    weights: &[f64],
    segments: &mut [Segment],
    frames: usize,
    fps: f64,
) {
    let radius = std::cmp::max(3, (fps * 0.20).round_ties_even() as usize);
    for foot in 0..2 {
        let stances: Vec<usize> = (0..segments.len())
            .filter(|&i| segments[i].foot == foot)
            .collect();
        let mut mask = vec![false; frames];
        let mut active = vec![false; frames];
        for &index in &stances {
            let segment = &segments[index];
            for t in segment.start..segment.end {
                active[t] = true;
            }
        }
        let mut reference = vec![0.0; frames * 3];
        for (first, end) in filters::intervals(&active) {
            let path = rolling_anchor(
                probes,
                weights,
                foot,
                first,
                end,
                anchors[(first * 2 + foot) * 3 + 1],
            );
            for (k, t) in (first..end).enumerate() {
                reference[t * 3..t * 3 + 3].copy_from_slice(&path[k * 3..k * 3 + 3]);
            }
        }
        for pair in 0..stances.len().saturating_sub(1) {
            let left = segments[stances[pair]].clone();
            let right = &segments[stances[pair + 1]];
            if left.end != right.start {
                continue;
            }
            let boundary = right.start;
            let first = left.start.max(boundary.saturating_sub(radius));
            let end = right.end.min(boundary + radius);
            for slot in mask.iter_mut().take(end).skip(first) {
                *slot = true;
            }
            segments[stances[pair + 1]].handoff = Some((first, end));
        }
        for (first, end) in filters::intervals(&mask) {
            if end - first < 3 {
                continue;
            }
            let correction: Vec<[f64; 3]> = (first..end)
                .map(|t| {
                    [
                        anchors[(t * 2 + foot) * 3] - reference[t * 3],
                        anchors[(t * 2 + foot) * 3 + 1] - reference[t * 3 + 1],
                        anchors[(t * 2 + foot) * 3 + 2] - reference[t * 3 + 2],
                    ]
                })
                .collect();
            let a = correction[0];
            let b = correction[correction.len() - 1];
            let va = sub3(correction[1], a);
            let vb = sub3(b, correction[correction.len() - 2]);
            let count = end - first - 2;
            let blended = hermite(a, va, b, vb, count);
            for (k, t) in (first + 1..end - 1).enumerate() {
                anchors[(t * 2 + foot) * 3] = reference[t * 3] + blended[k][0];
                anchors[(t * 2 + foot) * 3 + 2] = reference[t * 3 + 2] + blended[k][2];
            }
        }
        for &index in &stances {
            let segment = &mut segments[index];
            segment.anchor = [
                anchors[(segment.start * 2 + foot) * 3],
                anchors[(segment.start * 2 + foot) * 3 + 1],
                anchors[(segment.start * 2 + foot) * 3 + 2],
            ];
            segment.anchor_end = [
                anchors[((segment.end - 1) * 2 + foot) * 3],
                anchors[((segment.end - 1) * 2 + foot) * 3 + 1],
                anchors[((segment.end - 1) * 2 + foot) * 3 + 2],
            ];
        }
    }
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Interior samples of a cubic matching endpoint values and velocities.
fn hermite(a: [f64; 3], va: [f64; 3], b: [f64; 3], vb: [f64; 3], count: usize) -> Vec<[f64; 3]> {
    let duration = (count + 1) as f64;
    (1..=count)
        .map(|k| {
            let t = k as f64 / duration;
            let t2 = t * t;
            let t3 = t2 * t;
            let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
            let h10 = t3 - 2.0 * t2 + t;
            let h01 = -2.0 * t3 + 3.0 * t2;
            let h11 = t3 - t2;
            let mut out = [0.0; 3];
            for axis in 0..3 {
                out[axis] = h00 * a[axis]
                    + h10 * duration * va[axis]
                    + h01 * b[axis]
                    + h11 * duration * vb[axis];
            }
            out
        })
        .collect()
}

/// Keep stance targets and carry the correction into neighbouring swing frames.
///
/// Returns `(offsets, transition)` with `frames * 2 * 3` and `frames * 2`.
pub fn contact_offsets(
    probes: &[f64],
    active: &[bool],
    anchors: &[f64],
    support: &[f64],
    shift: &[f64],
    frames: usize,
    fps: f64,
    strength: &[f64],
    flight: &[bool],
) -> (Vec<f64>, Vec<bool>) {
    let mut offsets = vec![0.0; frames * 2 * 3];
    for t in 0..frames {
        for foot in 0..2 {
            let index = t * 2 + foot;
            if !active[index] {
                continue;
            }
            for axis in 0..3 {
                offsets[index * 3 + axis] = (anchors[index * 3 + axis]
                    - support[index * 3 + axis]
                    - shift[t * 3 + axis])
                    * strength[index];
            }
        }
    }
    let mut transition = vec![false; frames * 2];
    let maximum = std::cmp::max(2, (fps * 0.5).round_ties_even() as usize);
    let minimum = std::cmp::max(2, (fps * 0.12).round_ties_even() as usize);

    let extent = |value: [f64; 3]| -> usize {
        let norm = dot3(&value, &value).sqrt();
        std::cmp::min(
            maximum,
            std::cmp::max(minimum, (norm * fps / 0.8).ceil() as usize),
        )
    };

    let _ = probes;
    for foot in 0..2 {
        let col = column(active, foot);
        let flight_col = column(flight, foot);
        let stances = filters::intervals(&col);
        let by_start: std::collections::HashMap<usize, (usize, usize)> =
            stances.iter().map(|&(s, e)| (s, (s, e))).collect();
        let by_end: std::collections::HashMap<usize, (usize, usize)> =
            stances.iter().map(|&(s, e)| (e, (s, e))).collect();
        let gaps: Vec<bool> = (0..frames)
            .map(|t| !col[t] && !flight_col[t])
            .collect();
        for (first, end) in filters::intervals(&gaps) {
            let left = by_end.get(&first).copied();
            let right = by_start.get(&end).copied();
            if left.is_none() && right.is_none() {
                continue;
            }
            let count = end - first;
            let lvalue = if left.is_some() && first > 0 {
                let index = (first - 1) * 2 + foot;
                [offsets[index * 3], offsets[index * 3 + 1], offsets[index * 3 + 2]]
            } else {
                [0.0; 3]
            };
            let rvalue = if right.is_some() {
                let index = end * 2 + foot;
                [offsets[index * 3], offsets[index * 3 + 1], offsets[index * 3 + 2]]
            } else {
                [0.0; 3]
            };
            let lextent = if left.is_some() { extent(lvalue) } else { 0 };
            let rextent = if right.is_some() { extent(rvalue) } else { 0 };
            let lcount = count.min(lextent);
            let rcount = count.min(rextent);
            let lv = left
                .map(|(s, e)| slope(&offsets, foot, s, e, true))
                .unwrap_or([0.0; 3]);
            let rv = right
                .map(|(s, e)| slope(&offsets, foot, s, e, false))
                .unwrap_or([0.0; 3]);

            if left.is_some() && right.is_some() && lcount + rcount >= count {
                let blended = hermite(lvalue, lv, rvalue, rv, count);
                for (k, t) in (first..end).enumerate() {
                    let index = (t * 2 + foot) * 3;
                    offsets[index..index + 3].copy_from_slice(&blended[k]);
                    transition[t * 2 + foot] = true;
                }
            } else {
                if left.is_some() {
                    let duration = if end == frames { lextent } else { lcount };
                    let blended = hermite(lvalue, lv, [0.0; 3], [0.0; 3], duration);
                    for k in 0..lcount {
                        let t = first + k;
                        let index = (t * 2 + foot) * 3;
                        offsets[index..index + 3].copy_from_slice(&blended[k]);
                        transition[t * 2 + foot] = true;
                    }
                }
                if right.is_some() {
                    let duration = if first == 0 { rextent } else { rcount };
                    let blended = hermite([0.0; 3], [0.0; 3], rvalue, rv, duration);
                    for k in 0..rcount {
                        let t = end - rcount + k;
                        let index = (t * 2 + foot) * 3;
                        offsets[index..index + 3]
                            .copy_from_slice(&blended[duration - rcount + k]);
                        transition[t * 2 + foot] = true;
                    }
                }
            }
        }
    }
    (offsets, transition)
}

fn slope(offsets: &[f64], foot: usize, start: usize, end: usize, at_end: bool) -> [f64; 3] {
    if end - start < 2 {
        return [0.0; 3];
    }
    let (a, b) = if at_end {
        (end - 2, end - 1)
    } else {
        (start, start + 1)
    };
    let mut out = [0.0; 3];
    for axis in 0..3 {
        out[axis] = offsets[(b * 2 + foot) * 3 + axis] - offsets[(a * 2 + foot) * 3 + axis];
    }
    out
}

/// Strength per frame from the committed preset (`contact_strength`).
pub fn contact_strength(segments: &[Segment], frames: usize, preset: &Preset) -> Vec<f64> {
    let mut strength = vec![0.0; frames * 2];
    for segment in segments {
        for t in segment.start..segment.end.min(frames) {
            strength[t * 2 + segment.foot] = preset.strength;
        }
    }
    strength
}

/// Maximum physical reach with the source's non-hinge components fixed.
pub fn reach_lengths(rest: &[f64], world: &[f64], frames: usize) -> Vec<f64> {
    reach_impl(rest, world, frames, 1.0)
}

/// Minimum reach at the native solver's 160 degree flex limit.
pub fn minimum_reach_lengths(rest: &[f64], world: &[f64], frames: usize) -> Vec<f64> {
    reach_impl(rest, world, frames, (160.0f64).to_radians().cos())
}

fn reach_impl(rest: &[f64], world: &[f64], frames: usize, flex_cos: f64) -> Vec<f64> {
    let mut lengths = vec![0.0; frames * 2];
    for (leg, (hip, knee, ankle)) in [(1usize, 4usize, 7usize), (2, 5, 8)].iter().enumerate() {
        let upper = [
            rest[knee * 3] - rest[hip * 3],
            rest[knee * 3 + 1] - rest[hip * 3 + 1],
            rest[knee * 3 + 2] - rest[hip * 3 + 2],
        ];
        let lower = [
            rest[ankle * 3] - rest[knee * 3],
            rest[ankle * 3 + 1] - rest[knee * 3 + 1],
            rest[ankle * 3 + 2] - rest[knee * 3 + 2],
        ];
        let axis_matrix = align(&[0.0, -1.0, 0.0], &upper);
        let axis = [
            axis_matrix[0], axis_matrix[3], axis_matrix[6],
        ];
        let upper_sq = dot3(&upper, &upper);
        let lower_sq = dot3(&lower, &lower);
        for t in 0..frames {
            let hip_rot = &world[(t * 22 + hip) * 9..(t * 22 + hip) * 9 + 9];
            let knee_rot = &world[(t * 22 + knee) * 9..(t * 22 + knee) * 9 + 9];
            let local = math::tmul3(hip_rot, knee_rot);
            let v = [
                local[0] * lower[0] + local[1] * lower[1] + local[2] * lower[2],
                local[3] * lower[0] + local[4] * lower[1] + local[5] * lower[2],
                local[6] * lower[0] + local[7] * lower[1] + local[8] * lower[2],
            ];
            let projection = dot3(&v, &axis);
            let perpendicular = [
                v[0] - projection * axis[0],
                v[1] - projection * axis[1],
                v[2] - projection * axis[2],
            ];
            let cross = cross3(&axis, &v);
            let radius = (dot3(&perpendicular, &upper).powi(2)
                + dot3(&cross, &upper).powi(2))
            .sqrt();
            let constant = dot3(&upper, &axis) * projection;
            let value = upper_sq + lower_sq + 2.0 * (constant + radius * flex_cos);
            lengths[t * 2 + leg] = value.max(0.0).sqrt();
        }
    }
    lengths
}

/// Minimum downward pelvis move for both support legs, with extension margin.
pub fn reachable_pelvis_drop(
    hips: &[f64],
    targets: &[f64],
    lengths: &[f64],
    active: &[bool],
    frames: usize,
) -> Vec<f64> {
    let mut drop = vec![0.0; frames];
    for t in 0..frames {
        let mut needed = 0.0f64;
        for foot in 0..2 {
            if !active[t * 2 + foot] {
                continue;
            }
            let radius = lengths[t * 2 + foot] * 0.9995;
            let hip = &hips[t * 2 * 3 + foot * 3..t * 2 * 3 + foot * 3 + 3];
            let target = &targets[(t * 2 + foot) * 3..(t * 2 + foot) * 3 + 3];
            let horizontal = (hip[0] - target[0]).powi(2) + (hip[2] - target[2]).powi(2);
            let vertical = hip[1] - target[1];
            let limit = (radius * radius - horizontal).max(0.0).sqrt();
            if horizontal < radius * radius && vertical > 0.0 {
                needed = needed.max((vertical - limit).max(0.0));
            }
        }
        drop[t] = needed.min(0.25);
    }
    drop
}

/// Small upward reach compensation for deep crouches, without stretching.
pub fn reachable_pelvis_lift(
    hips: &[f64],
    targets: &[f64],
    minimum: &[f64],
    maximum: &[f64],
    active: &[bool],
    frames: usize,
) -> Vec<f64> {
    let mut lift = vec![0.0; frames];
    for t in 0..frames {
        let mut needed = 0.0f64;
        let mut permitted = f64::INFINITY;
        for foot in 0..2 {
            if !active[t * 2 + foot] {
                continue;
            }
            let hip = &hips[t * 2 * 3 + foot * 3..t * 2 * 3 + foot * 3 + 3];
            let target = &targets[(t * 2 + foot) * 3..(t * 2 + foot) * 3 + 3];
            let horizontal = (hip[0] - target[0]).powi(2) + (hip[2] - target[2]).powi(2);
            let vertical = hip[1] - target[1];
            if vertical <= 0.0 {
                continue;
            }
            let lower = ((minimum[t * 2 + foot] * 1.0005).powi(2) - horizontal).max(0.0).sqrt()
                - vertical;
            let upper = ((maximum[t * 2 + foot] * 0.9995).powi(2) - horizontal).max(0.0).sqrt()
                - vertical;
            needed = needed.max(lower);
            permitted = permitted.min(upper);
        }
        lift[t] = needed.max(0.0).min(permitted.max(0.0)).min(0.12);
    }
    lift
}

/// Forward kinematics from quaternions for the 22-joint SMPL-X body.
pub fn forward_kinematics(
    rest: &[f64],
    roots: &[f64],
    quats: &[f64],
    frames: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut positions = vec![0.0; frames * 22 * 3];
    let mut rotations = vec![0.0; frames * 22 * 9];
    crate::cpp::fk(
        rest,
        roots,
        quats,
        22,
        &math::PARENTS,
        &mut positions,
        &mut rotations,
    );
    (positions, rotations)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intervals_are_half_open() {
        let mask = [false, true, true, false, true];
        assert_eq!(filters::intervals(&mask), vec![(1, 3), (4, 5)]);
    }

    #[test]
    fn hermite_returns_interior_samples() {
        // `_hermite` yields `count` interior samples between the endpoints.
        let values = hermite([1.0, 0.0, 0.0], [0.0; 3], [2.0, 0.0, 0.0], [0.0; 3], 4);
        assert_eq!(values.len(), 4);
        assert!(values[0][0] > 1.0 && values[0][0] < 2.0);
        assert!(values[3][0] > values[0][0] && values[3][0] < 2.0);
    }
}
