//! Contact phases from sole probes and network probabilities, ported from
//! `correction.py::prepare_contacts` (RGB MoCap, MIT), flat mode.
//!
//! This is the one place that decides which frames hold a foot. The reference's
//! gate is deliberately *geometric and local*: a foot is planted when the sole
//! patch moves slowly, does not travel vertically, and sits within 12 cm of the
//! other foot — all measured on the **skinned sole patches**, never on the ankle
//! joint, and never against the clip's global floor. A stance is only *entered*
//! on the network's own probability (0.65 for the normal preset), which is what
//! keeps a sliding foot from being pinned just because it is low.
//!
//! ```text
//! height   = min of the foot's two sole patches          (the lower surface)
//! points   = mean of the foot's two sole patches
//! geometry = speed < 0.7 m/s ∧ |vertical| < 0.8 m/s ∧ height − min(both feet) < 0.12
//! active   = geometry ∧ probability > preset.confidence
//! holding  = geometry ∧ probability > preset.release ∧ |height − floor| < 0.04 ∧ speed < 0.25
//! ```
//!
//! `holding` never starts a contact. It only extends one that already exists, and
//! only for the `strong` preset (`gap > 0`), because bridging a dropout is a
//! deliberate, preset-level decision in the reference. The minimum-duration and
//! airborne-arc filters run after it, on the raw geometry.

/// How fast a sole may slide and still be planted (`geometry` gate).
pub const GATE_SPEED: f32 = 0.7;
/// How fast a sole may rise or fall and still be planted.
pub const GATE_VERTICAL: f32 = 0.8;
/// How far above the other foot a planted sole may sit.
pub const GATE_HEIGHT: f32 = 0.12;
/// The speed a stance may keep and still be *held* through a dropout.
pub const HOLD_SPEED: f32 = 0.25;
/// How far from the floor a held stance may sit.
pub const HOLD_NEAR: f32 = 0.04;
/// How far above the **local floor** a sole may be and still count as planted.
///
/// The reference never needs this: its contact head reports a raised foot. The
/// FlowHMR head is confident on airborne feet too (a serve take averages 0.91),
/// and the `geometry` gate only compares the two feet with each other, so a jump
/// shorter than `airborne_excursions`'s 15 cm arc would stay "planted" and be
/// pushed back onto the floor. `HOLD_NEAR` plus 2 cm of sole-roll slack.
pub const CONTACT_LIFT: f32 = 0.06;

/// A bridging stage only touches gaps at most this wide, in the reference's
/// two-foot step test (`|Δxz| < .04` between the two sides of the gap).
pub const BRIDGE_STEP: f32 = 0.04;

/// One preset's contact settings (`correction_settings.PRESETS`).
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Probability that starts a contact.
    pub confidence: f32,
    /// Lower probability that keeps one alive (only used when `gap > 0`).
    pub release: f32,
    /// Shortest stance kept, in seconds (`minimum`).
    pub minimum: f32,
    /// Longest dropout bridged, in seconds (`gap`; 0 disables the stage).
    pub gap: f32,
}

/// The reference's three presets.
impl Config {
    pub fn weak() -> Self {
        Self {
            confidence: 0.75,
            release: 0.75,
            minimum: 0.06,
            gap: 0.0,
        }
    }

    pub fn normal() -> Self {
        Self {
            confidence: 0.65,
            release: 0.65,
            minimum: 0.06,
            gap: 0.0,
        }
    }

    pub fn strong() -> Self {
        Self {
            confidence: 0.65,
            release: 0.5,
            minimum: 0.06,
            gap: 0.08,
        }
    }
}

/// One frame's sole probes: per foot, the toe-side and heel-side patch.
pub type Probes = [[[f32; 3]; 2]; 2];

/// What the detector resolved.
pub struct Prepared {
    /// Per frame, per foot.
    pub active: Vec<[bool; 2]>,
    /// Per frame: both feet off the ground.
    pub flight: Vec<bool>,
    /// `quantile(height[active], .05)`, or the floor that was passed in.
    pub floor: f32,
    /// How many (frame, foot) samples the floor came from.
    pub samples: usize,
}

fn run_bounds(mask: &[bool]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut index = 0;
    while index < mask.len() {
        if !mask[index] {
            index += 1;
            continue;
        }
        let begin = index;
        while index < mask.len() && mask[index] {
            index += 1;
        }
        out.push((begin, index));
    }
    out
}

/// Linear-interpolated quantile, the same convention numpy's `quantile` uses.
pub fn quantile(values: &[f32], q: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let position = q.clamp(0.0, 1.0) * (sorted.len() as f32 - 1.0);
    let low = position.floor() as usize;
    let high = (low + 1).min(sorted.len() - 1);
    let weight = position - low as f32;
    sorted[low] * (1.0 - weight) + sorted[high] * weight
}

/// Gaussian smoothing with a nearest-edge clamp, scipy's `mode='nearest'`.
pub(crate) fn gaussian_smooth(values: &[f32], sigma: f32) -> Vec<f32> {
    let count = values.len();
    if count == 0 || sigma <= 0.0 {
        return values.to_vec();
    }
    // scipy's truncate: 4 sigma, rounded.
    let radius = (4.0 * sigma + 0.5).floor() as isize;
    let kernel: Vec<f32> = (-radius..=radius)
        .map(|offset| {
            let o = offset as f32;
            (-(o * o) / (2.0 * sigma * sigma)).exp()
        })
        .collect();
    let total: f32 = kernel.iter().sum();
    (0..count)
        .map(|frame| {
            let mut sum = 0.0f32;
            for (slot, offset) in (-radius..=radius).enumerate() {
                let at = (frame as isize + offset).clamp(0, count as isize - 1) as usize;
                sum += values[at] * kernel[slot];
            }
            sum / total
        })
        .collect()
}

/// `np.gradient`: central differences inside, one-sided at the ends.
fn gradient(values: &[f32]) -> Vec<f32> {
    let count = values.len();
    match count {
        0 => Vec::new(),
        1 => vec![0.0],
        _ => {
            let mut out = Vec::with_capacity(count);
            out.push(values[1] - values[0]);
            for index in 1..count - 1 {
                out.push(0.5 * (values[index + 1] - values[index - 1]));
            }
            out.push(values[count - 1] - values[count - 2]);
            out
        }
    }
}

/// `airborne_excursions`: a bounded two-foot arc, rejected without grounding
/// away its apex.
///
/// A whole take that floats continuously is ambiguous and stays groundable; only
/// arcs with observed lower samples on **both** sides are rejected. The baseline
/// is a local lower envelope, so a slow world-height drift over a long take is
/// not mistaken for a jump.
fn airborne_excursions(probes: &[Probes], fps: f32) -> Vec<bool> {
    let count = probes.len();
    let mut flight = vec![false; count];
    if count < 5 {
        return flight;
    }
    // `probes[...,1].min(1)`: the lowest probe of both feet.
    let height: Vec<f32> = probes
        .iter()
        .map(|row| {
            row[0][0][1]
                .min(row[0][1][1])
                .min(row[1][0][1])
                .min(row[1][1][1])
        })
        .collect();
    let reach = count.min((fps * 4.0).round().max(2.0) as usize);
    let radius = reach as isize;
    let baseline: Vec<f32> = (0..count)
        .map(|frame| {
            let mut low = f32::MAX;
            for offset in -radius..=radius {
                let at = (frame as isize + offset).clamp(0, count as isize - 1) as usize;
                low = low.min(height[at]);
            }
            low
        })
        .collect();
    let raised: Vec<bool> = (0..count).map(|frame| height[frame] - baseline[frame] > 0.15).collect();
    let smoothed = gaussian_smooth(&height, (fps * 0.03).max(0.5));
    let velocity: Vec<f32> = gradient(&smoothed).iter().map(|v| v * fps).collect();
    for (first, end) in run_bounds(&raised) {
        let mut peak = first;
        for frame in first..end {
            if smoothed[frame] > smoothed[peak] {
                peak = frame;
            }
        }
        let low = peak.saturating_sub(reach);
        let high = (peak + reach + 1).min(count);
        if peak == low || peak + 1 == high {
            continue;
        }
        let left = smoothed[low..peak].iter().cloned().fold(f32::MAX, f32::min);
        let right = smoothed[peak + 1..high].iter().cloned().fold(f32::MAX, f32::min);
        let top = smoothed[peak];
        if top - left.max(right) < 0.15 {
            continue;
        }
        let before = (low..peak).filter(|frame| smoothed[*frame] < left + 0.06).last();
        let after = (peak + 1..high)
            .find(|frame| smoothed[*frame] < right + 0.06);
        let (Some(before), Some(after)) = (before, after) else {
            continue;
        };
        let start = before + 1;
        let mut stop = after;
        // Landing can return to a slightly different inferred world height, so
        // the end is the end of the rapid descent, not a return to the old Y.
        let descent_end = (stop + 1).min(count);
        let mut falling = f32::MAX;
        for frame in peak + 1..descent_end {
            falling = falling.min(velocity[frame]);
            if falling < -0.8 && velocity[frame] > -0.35 && top - smoothed[frame] > 0.15 {
                stop = frame;
                break;
            }
        }
        if stop as f32 - start as f32 > 8.0 * fps {
            continue;
        }
        if start < stop {
            flight[start..stop].fill(true);
        }
    }
    flight
}

/// `prepare_contacts`: resolve the contact phases of one clip.
///
/// `floor` is the correction plane when the caller already knows it (the
/// correction pass), and `None` when it has to be measured from the contacts
/// themselves (the floor pass).
pub fn prepare(
    probes: &[Probes],
    probabilities: &[[f32; 4]],
    fps: f32,
    config: &Config,
    floor: Option<f32>,
) -> Prepared {
    let count = probes.len();
    let mut active = vec![[false; 2]; count];
    if count == 0 {
        return Prepared {
            active,
            flight: Vec::new(),
            floor: floor.unwrap_or(0.0),
            samples: 0,
        };
    }
    let fps = fps.max(1.0);
    let mut height = vec![[0.0f32; 2]; count];
    let mut point = vec![[[0.0f32; 3]; 2]; count];
    for frame in 0..count {
        for foot in 0..2 {
            let back = probes[frame][foot][1];
            let front = probes[frame][foot][0];
            height[frame][foot] = back[1].min(front[1]);
            point[frame][foot] = [
                0.5 * (back[0] + front[0]),
                0.5 * (back[1] + front[1]),
                0.5 * (back[2] + front[2]),
            ];
        }
    }
    let mut speed = vec![[0.0f32; 2]; count];
    let mut vertical = vec![[0.0f32; 2]; count];
    for frame in 1..count {
        for foot in 0..2 {
            let dx = point[frame][foot][0] - point[frame - 1][foot][0];
            let dz = point[frame][foot][2] - point[frame - 1][foot][2];
            speed[frame][foot] = (dx * dx + dz * dz).sqrt() * fps;
            vertical[frame][foot] = (height[frame][foot] - height[frame - 1][foot]).abs() * fps;
        }
        speed[0] = speed[1];
        vertical[0] = vertical[1];
    }
    // The four channels are two per foot: `L ankle/toe, R ankle/toe`
    // (`flow_correction.contact_probabilities`). `flow_correction` clips them to
    // [0, 1] — the ODE output overshoots a little, and the reference's own gate
    // asserts the range — so the clip is part of the port.
    let mut confidence = vec![[0.0f32; 2]; count];
    for frame in 0..count {
        let row = probabilities.get(frame).copied().unwrap_or([0.0; 4]);
        confidence[frame] = [
            row[0].max(row[1]).clamp(0.0, 1.0),
            row[2].max(row[3]).clamp(0.0, 1.0),
        ];
    }
    let mut geometry = vec![[false; 2]; count];
    for frame in 0..count {
        let lowest = height[frame][0].min(height[frame][1]);
        for foot in 0..2 {
            geometry[frame][foot] = speed[frame][foot] < GATE_SPEED
                && vertical[frame][foot] < GATE_VERTICAL
                && height[frame][foot] - lowest < GATE_HEIGHT;
        }
    }
    for frame in 0..count {
        for foot in 0..2 {
            active[frame][foot] = geometry[frame][foot] && confidence[frame][foot] > config.confidence;
        }
    }

    if config.gap > 0.0 {
        // Extend only seeded stances, close to the floor and moving slowly. A
        // new step must still pass the original high-confidence entry gate.
        let mut support: Vec<f32> = Vec::new();
        for frame in 0..count {
            for foot in 0..2 {
                if active[frame][foot] {
                    support.push(height[frame][foot]);
                }
            }
        }
        let reference = floor.unwrap_or_else(|| {
            if support.is_empty() {
                height.iter().flatten().cloned().fold(f32::MAX, f32::min)
            } else {
                quantile(&support, 0.05)
            }
        });
        let near: Vec<[bool; 2]> = (0..count)
            .map(|frame| {
                [
                    (height[frame][0] - reference).abs() < HOLD_NEAR,
                    (height[frame][1] - reference).abs() < HOLD_NEAR,
                ]
            })
            .collect();
        let mut holding = vec![[false; 2]; count];
        for frame in 0..count {
            for foot in 0..2 {
                holding[frame][foot] = confidence[frame][foot] > config.release
                    && geometry[frame][foot]
                    && near[frame][foot]
                    && speed[frame][foot] < HOLD_SPEED;
            }
        }
        let bridge = (fps * config.gap).round() as usize;
        for foot in 0..2 {
            let seeded: Vec<bool> = (0..count)
                .map(|frame| holding[frame][foot] || active[frame][foot])
                .collect();
            for (first, end) in run_bounds(&seeded) {
                if let Some(offset) = (first..end).find(|frame| active[*frame][foot]) {
                    for frame in first + offset..end {
                        active[frame][foot] = true;
                    }
                }
            }
            // Close a brief dropout only between two existing stances. Never
            // bridge a lifted foot, a sizeable step or an unbounded gap.
            let empty: Vec<bool> = (0..count).map(|frame| !active[frame][foot]).collect();
            for (first, end) in run_bounds(&empty) {
                if first == 0 || end == count || end - first > bridge {
                    continue;
                }
                if !(first..end).all(|frame| geometry[frame][foot] && near[frame][foot]) {
                    continue;
                }
                let dx = point[end][foot][0] - point[first - 1][foot][0];
                let dz = point[end][foot][2] - point[first - 1][foot][2];
                if (dx * dx + dz * dz).sqrt() > BRIDGE_STEP {
                    continue;
                }
                for frame in first..end {
                    active[frame][foot] = true;
                }
            }
        }
    }

    let minimum = (fps * config.minimum).round().max(2.0) as usize;
    for foot in 0..2 {
        let mask: Vec<bool> = (0..count).map(|frame| active[frame][foot]).collect();
        for (first, end) in run_bounds(&mask) {
            if end - first < minimum {
                for frame in first..end {
                    active[frame][foot] = false;
                }
            }
        }
    }

    let flight = airborne_excursions(probes, fps);
    for frame in 0..count {
        if flight[frame] {
            active[frame] = [false; 2];
        }
    }
    let reach = count.min((fps * 4.0).round().max(2.0) as usize);
    let radius = reach as isize;

    // A foot well above the local floor is not in contact, whatever the network's    // contact head says. `airborne_excursions` catches a *bounded* arc that clears
    // 15 cm above both sides; a smaller hop — or one whose landing is not visible
    // in the take — stays "planted" for the geometry gate (both feet being at the
    // same height is what `geometry` tests), and then the grounding pass presses
    // the whole body down to put that "planted" foot on the floor: the jump is
    // flattened, which is exactly what "跳起来却被摁在地上" looks like.
    //
    // The envelope is the local lower surface, the same shape `airborne_excursions`
    // uses, so a slow world-height drift is not mistaken for a lift. The margin is
    // the reference's own `HOLD_NEAR` (4 cm) plus 2 cm of sole-roll slack.
    let foot_height = |frame: usize, foot: usize| height[frame][foot];
    // A *quantile* of the window, not its minimum: one bad frame (a foot that
    // dips through the floor for three frames, a reconstruction hiccup) would
    // otherwise pull the envelope down for the whole 8 s window and turn every
    // real stance into "lifted".
    let mut window: Vec<f32> = Vec::new();
    let envelope: Vec<f32> = (0..count)
        .map(|frame| {
            window.clear();
            let first = (frame as isize - radius).max(0) as usize;
            let last = ((frame as isize + radius) as usize).min(count - 1);
            // The real index range, not a clamped walk: clamping the offsets would
            // repeat the first/last frame hundreds of times and pull the quantile
            // towards it (a clip that opens three frames low reported the whole
            // floor as low).
            for at in first..=last {
                window.push(height[at][0].min(height[at][1]));
            }
            quantile(&window, 0.05)
        })
        .collect();
    for frame in 0..count {
        for foot in 0..2 {
            if active[frame][foot] && foot_height(frame, foot) - envelope[frame] > CONTACT_LIFT {
                active[frame][foot] = false;
            }
        }
    }

    let mut values: Vec<f32> = Vec::new();
    for frame in 0..count {
        for foot in 0..2 {
            if active[frame][foot] {
                values.push(height[frame][foot]);
            }
        }
    }
    let resolved = match floor {
        Some(given) => given,
        None if values.is_empty() => 0.0,
        None => quantile(&values, 0.05),
    };
    Prepared {
        active,
        flight,
        floor: resolved,
        samples: values.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat planted foot 8 cm above the clip's floor datum.
    fn planted(y: f32) -> Probes {
        [
            [[0.0, y, 0.10], [0.0, y, -0.10]],
            [[0.2, y, 0.10], [0.2, y, -0.10]],
        ]
    }

    /// One foot planted, the other lifted by `lift`.
    fn stepping(y: f32, lift: f32) -> Probes {
        let mut row = planted(y);
        row[1][0][1] = y + lift;
        row[1][1][1] = y + lift;
        row
    }

    fn steady(count: usize) -> (Vec<Probes>, Vec<[f32; 4]>) {
        let probes = vec![planted(0.08); count];
        let probabilities = vec![[0.9, 0.9, 0.9, 0.9]; count];
        (probes, probabilities)
    }

    #[test]
    fn a_still_pose_is_one_long_contact() {
        let (probes, probabilities) = steady(30);
        let prepared = prepare(&probes, &probabilities, 30.0, &Config::normal(), None);
        assert!(prepared.active.iter().all(|row| row == &[true, true]));
        assert!((prepared.floor - 0.08).abs() < 1e-6, "floor {}", prepared.floor);
        assert_eq!(prepared.samples, 60);
    }

    /// A lifted foot is not within 12 cm of the other one, so it is not planted
    /// — the reference's local height gate, not a global floor distance.
    #[test]
    fn a_lifted_foot_is_not_a_contact() {
        let count = 30;
        let probes: Vec<Probes> = (0..count).map(|_| stepping(0.08, 0.2)).collect();
        let probabilities = vec![[0.9f32; 4]; count];
        let prepared = prepare(&probes, &probabilities, 30.0, &Config::normal(), None);
        assert!(prepared.active.iter().all(|row| !row[1]), "the lifted foot was pinned");
        assert!(prepared.active.iter().all(|row| row[0]));
    }

    /// A slow hop that clears less than the reference's 15 cm jump arc is still
    /// "planted" for the geometry gate — both feet rise together — so it needs the
    /// floor-proximity gate. Without it the grounding pass would press the body
    /// back down onto the floor and the jump would disappear.
    #[test]
    fn a_slow_hop_is_not_a_stance() {
        let count = 40;
        let mut probes = Vec::with_capacity(count);
        for frame in 0..count {
            // 8 cm up over five frames (0.48 m/s vertical), five frames at the
            // top, then back down.
            let lift = match frame {
                10..=14 => (frame - 9) as f32 * 0.016,
                15..=19 => 0.08,
                20..=24 => (30 - frame) as f32 * 0.016,
                _ => 0.0,
            };
            let mut row = planted(0.0 + lift);
            row[1] = row[0];
            probes.push(row);
        }
        let probabilities = vec![[0.99f32; 4]; count];
        let prepared = prepare(&probes, &probabilities, 30.0, &Config::normal(), None);
        for frame in 14..=20 {
            assert!(
                !prepared.active[frame][0] && !prepared.active[frame][1],
                "frame {frame} is {:.3} m up but was called planted",
                0.08
            );
        }
        assert!(prepared.active[0][0] && prepared.active[0][1], "the ground frames stay planted");
    }

    /// A foot sliding at 1 m/s passes the height gate but not the speed gate.
    #[test]
    fn a_sliding_foot_is_not_a_contact() {        let count = 30;
        let probes: Vec<Probes> = (0..count)
            .map(|frame| {
                let mut row = planted(0.08);
                let x = frame as f32 * 0.04;
                for row in row.iter_mut() {
                    for side in row.iter_mut() {
                        side[0] += x;
                    }
                }
                row
            })
            .collect();
        let probabilities = vec![[0.9f32; 4]; count];
        let prepared = prepare(&probes, &probabilities, 30.0, &Config::normal(), None);
        assert!(prepared.active.iter().all(|row| !row[0]));
    }

    /// The probability gate: without it a low, still foot would be pinned.
    #[test]
    fn a_low_probability_frame_is_not_entered() {
        let count = 10;
        let probes = vec![planted(0.08); count];
        let mut probabilities = vec![[0.9f32; 4]; count];
        for row in probabilities.iter_mut().take(5) {
            *row = [0.5f32; 4];
        }
        let prepared = prepare(&probes, &probabilities, 30.0, &Config::normal(), None);
        assert!(prepared.active[..5].iter().all(|row| !row[0]));
        assert!(prepared.active[5..].iter().all(|row| row[0]));
    }

    /// A short dropout is only bridged when the preset asks for it.
    #[test]
    fn bridging_is_a_preset_decision() {
        let count = 30;
        let probes = vec![planted(0.08); count];
        let mut probabilities = vec![[0.9f32; 4]; count];
        for row in probabilities.iter_mut().skip(10).take(2) {
            *row = [0.1f32; 4];
        }
        let normal = prepare(&probes, &probabilities, 30.0, &Config::normal(), None);
        assert!(!normal.active[10][0], "normal must not bridge");
        let strong = prepare(&probes, &probabilities, 30.0, &Config::strong(), None);
        assert!(strong.active[10][0], "strong bridges a grounded dropout");
    }

    /// A run shorter than the preset's minimum is dropped.
    #[test]
    fn a_one_frame_stance_is_dropped() {
        let count = 30;
        let probes = vec![planted(0.08); count];
        let mut probabilities = vec![[0.1f32; 4]; count];
        probabilities[15] = [0.9; 4];
        let prepared = prepare(&probes, &probabilities, 30.0, &Config::normal(), None);
        assert!(prepared.active.iter().all(|row| !row[0]));
    }

    /// A given floor is used as-is instead of being measured.
    #[test]
    fn a_given_floor_is_not_re_measured() {
        let (probes, probabilities) = steady(20);
        let prepared = prepare(&probes, &probabilities, 30.0, &Config::normal(), Some(-0.5));
        assert_eq!(prepared.floor, -0.5);
    }

    /// A jump with lower samples on both sides is flight.
    #[test]
    fn a_bounded_jump_is_flight() {
        let count = 90;
        // Standing, then a 0.4 m two-foot arc for 20 frames, then standing again.
        let probes: Vec<Probes> = (0..count)
            .map(|frame| {
                let lift = if (35..55).contains(&frame) {
                    let t = (frame - 35) as f32 / 20.0;
                    0.4 * (t * std::f32::consts::PI).sin()
                } else {
                    0.0
                };
                let mut row = planted(0.08);
                if lift > 0.0 {
                    for foot in row.iter_mut() {
                        for side in foot.iter_mut() {
                            side[1] += lift;
                        }
                    }
                }
                row
            })
            .collect();
        let probabilities = vec![[0.9f32; 4]; count];
        let prepared = prepare(&probes, &probabilities, 30.0, &Config::normal(), None);
        assert!(
            prepared.flight[40..50].iter().any(|hit| *hit),
            "the arc was not found"
        );
        assert!(!prepared.flight[5], "a planted frame was marked as flight");
    }
}
