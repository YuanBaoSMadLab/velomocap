//! Root-height planning, ported from `correction.py::correct_arrays` (RGB MoCap,
//! MIT), steps 0–4 of the vertical chain:
//!
//! ```text
//! 0  ground    prepare_contacts       put the lowest planted sole on the floor
//!                                    (five-band smoothing, 2 % weight in the swing)
//! 1  drop      reachable_pelvis_drop   the legs cannot span the crouch: lower the body
//!              (enveloped: max over 0.1 s, then gaussian 0.03 s)
//! 2  lift      reachable_pelvis_lift   the legs cannot fold that far: raise the body
//! 3  offset    pelvis_offset           the user's "Pelvis Height Offset (from Auto)"
//! 4  clearance flight                  a whole airborne body must not clip the plane
//! ```
//!
//! Step 0 is [`grounding`]: it is what actually stands the body on the measured
//! plane, and steps 1–4 are refinements measured from the root it produces.
//! Without it the soles keep the network's own height error, which reads as a
//! foot hovering or sinking even though the horizontal pin is exact.
//!
//! Step 5 is [`crate::pose_plan`], which runs afterwards and is handed the root
//! this module produced. The order matters: a drop has to happen *before* the
//! lift, because the lift's bound is measured from the already-lowered hip, and
//! the user offset is capped by `drop − lift` so it can only take back what the
//! automatic reach compensation took, never add height no contact asked for.
//!
//! Every bound here is derived from the leg's own reach at that frame
//! (`anatomical::reach` / `minimum_reach`, which follow the knee's non-hinge
//! component), so a leg that is genuinely too short is never stretched to hide
//! an unreachable pin.

use crate::contacts::gaussian_smooth;

/// The grounding shift is clamped to this (`np.clip(floor - support, ±.75)`).
pub const GROUND_LIMIT: f64 = 0.75;
/// A drop is never larger than this (`np.minimum(needed, .25)`).
pub const DROP_LIMIT: f64 = 0.25;
/// A lift is never larger than this (`... , .12`).
pub const LIFT_LIMIT: f64 = 0.12;
/// Airborne clearance is never larger than this.
pub const CLEARANCE_LIMIT: f64 = 0.12;
/// The reach margins the reference applies around the calibrated bounds.
const REACH_LOW: f64 = 0.9995;
const REACH_HIGH: f64 = 1.0005;

/// One frame of input, in metres.
#[derive(Clone, Copy, Debug)]
pub struct Frame {
    /// Hip positions (joints 1 and 2) before this plan.
    pub hips: [[f64; 3]; 2],
    /// The ankle targets the legs are being asked to reach (joints 7 and 8).
    pub targets: [[f64; 3]; 2],
    /// Max reach per leg at this frame (`anatomical::reach`).
    pub high: [f64; 2],
    /// Min reach per leg at this frame (`anatomical::minimum_reach`).
    pub low: [f64; 2],
    /// Contact per leg, transition frames included.
    pub active: [bool; 2],
    /// Both feet off the ground at this frame.
    pub flight: bool,
    /// Lowest sole probe of the whole body at this frame.
    pub lowest: f64,
    /// This frame's hold strength (`contact_strength`, max over both feet).
    pub strength: f32,
}

/// `reachable_pelvis_drop`: how far the hips must come down for both support
/// legs to still span their targets.
fn drop_at(frame: &Frame) -> f64 {
    let mut needed = 0.0f64;
    for leg in 0..2 {
        if !frame.active[leg] {
            continue;
        }
        let radius = frame.high[leg] * REACH_LOW;
        let horizontal = (frame.hips[leg][0] - frame.targets[leg][0]).powi(2)
            + (frame.hips[leg][2] - frame.targets[leg][2]).powi(2);
        let vertical = frame.hips[leg][1] - frame.targets[leg][1];
        if horizontal >= radius * radius || vertical <= 0.0 {
            continue;
        }
        let limit = (radius * radius - horizontal).max(0.0).sqrt();
        needed = needed.max((vertical - limit).max(0.0));
    }
    needed.min(DROP_LIMIT)
}

/// `reachable_pelvis_lift`: how far the hips must come up so a target that sits
/// too close to the hip is still outside the leg's minimum fold.
fn lift_at(frame: &Frame, shift: f64) -> f64 {
    let mut needed = 0.0f64;
    let mut permitted = f64::INFINITY;
    for leg in 0..2 {
        if !frame.active[leg] {
            continue;
        }
        let vertical = frame.hips[leg][1] + shift - frame.targets[leg][1];
        if vertical <= 0.0 {
            continue;
        }
        let horizontal = (frame.hips[leg][0] - frame.targets[leg][0]).powi(2)
            + (frame.hips[leg][2] - frame.targets[leg][2]).powi(2);
        let lower = (frame.low[leg] * REACH_HIGH).powi(2) - horizontal;
        let upper = (frame.high[leg] * REACH_LOW).powi(2) - horizontal;
        needed = needed.max(lower.max(0.0).sqrt() - vertical);
        permitted = permitted.min(upper.max(0.0).sqrt() - vertical);
    }
    needed.max(0.0).min(permitted).max(0.0).min(LIFT_LIMIT)
}

/// `maximum_filter1d(values, size=2r+1, mode='nearest')`.
fn maximum_filter(values: &[f32], radius: usize) -> Vec<f32> {
    let count = values.len();
    (0..count)
        .map(|frame| {
            let mut high = f32::MIN;
            for offset in -(radius as isize)..=(radius as isize) {
                let at = (frame as isize + offset).clamp(0, count as isize - 1) as usize;
                high = high.max(values[at]);
            }
            high
        })
        .collect()
}

fn any(values: &[f32]) -> bool {
    values.iter().any(|value| *value > 0.0)
}

/// Plan the root height: returns the per-frame vertical delta to add.
pub fn plan(
    frames: &[Frame],
    floor: f64,
    fps: f64,
    flat: bool,
    pelvis_offset: f64,
) -> Vec<f64> {
    let count = frames.len();
    let mut delta = vec![0.0f64; count];
    if count == 0 {
        return delta;
    }
    let fps = fps.max(1.0);

    // 1. Drop, enveloped so the body eases down instead of stepping.
    let mut drop: Vec<f32> = frames.iter().map(|frame| drop_at(frame) as f32).collect();
    if any(&drop) {
        let radius = ((fps * 0.1).round() as usize).max(1);
        let envelope = maximum_filter(&drop, radius);
        let smoothed = gaussian_smooth(&envelope, (fps * 0.03).max(0.5) as f32);
        for frame in 0..count {
            // `np.maximum(drop, gaussian(maximum_filter1d(drop)))`: the envelope
            // leads the raw drop never lags it, so the ramp starts early.
            drop[frame] = drop[frame].max(smoothed[frame]);
        }
    }
    let drop64: Vec<f64> = drop.iter().map(|value| *value as f64).collect();
    for frame in 0..count {
        delta[frame] = -drop64[frame];
    }

    // 2. Lift, measured from the lowered hip.
    let mut lift = vec![0.0f64; count];
    for frame in 0..count {
        lift[frame] = lift_at(&frames[frame], delta[frame]);
        delta[frame] += lift[frame];
    }

    // 3. The user's own offset: positive may only take back what the automatic
    // reach compensation took, negative fades with the contacts.
    if pelvis_offset != 0.0 {
        for frame in 0..count {
            let automatic = (drop64[frame] - lift[frame]).max(0.0);
            let adjustment = if pelvis_offset >= 0.0 {
                automatic.min(pelvis_offset)
            } else {
                pelvis_offset * frames[frame].strength as f64
            };
            delta[frame] += adjustment;
        }
    }

    // 4. A whole airborne body must not be pushed through the plane by the
    // interpolation between two grounding offsets. Never for an explicitly
    // contact-free take, and never for a manual pin.
    if flat && frames.iter().any(|frame| frame.active[0] || frame.active[1]) {
        for frame in 0..count {
            let source = &frames[frame];
            if !source.flight || source.active[0] || source.active[1] {
                continue;
            }
            let clearance = (floor - source.lowest - delta[frame]).clamp(0.0, CLEARANCE_LIMIT);
            delta[frame] += clearance;
        }
    }
    delta
}

/// The reference's grounding step, the one that runs *before* steps 1–4.
///
/// `prepare_contacts` puts the body on the plane it measured:
///
/// ```text
/// support_height = min over the planted feet of the lowest sole patch
/// desired        = clip(floor - support_height, ±0.75)
/// shift          = argmin  Σ wᵢ (xᵢ - desiredᵢ)² + smooth·Σ (Δ²x)²
///                  w = 1 on a frame with a planted foot, 0.02 elsewhere
///                  smooth = 4·(fps/30)⁴
/// ```
///
/// The 2 % weight on the swing frames is what keeps a step from snapping the
/// whole body down: the root follows the planted soles and only borrows the
/// interpolated height while both feet are off the ground. `scipy` solves the
/// five-band system with `solve_banded((2,2), …)`; this is the same system, the
/// same coefficients, solved by a five-band Cholesky.
pub fn grounding(sole_height: &[f32], planted: &[bool], floor: f32, fps: f32) -> Vec<f32> {
    let count = sole_height.len().min(planted.len());
    let mut out = vec![0.0f32; count];
    if count == 0 {
        return out;
    }
    let support: Vec<usize> = (0..count).filter(|frame| planted[*frame]).collect();
    if support.is_empty() {
        return out;
    }
    // `np.interp` over the planted frames, held at both ends: the swing frames
    // start from the height the planted soles imply rather than from zero.
    let mut desired = vec![0.0f64; count];
    for frame in &support {
        desired[*frame] = ((floor - sole_height[*frame]) as f64).clamp(-GROUND_LIMIT, GROUND_LIMIT);
    }
    let curve: Vec<f64> = crate::correction::interpolate(&desired, &support, count)
        .iter()
        .map(|value| *value as f64)
        .collect();

    let smooth = 4.0 * (fps.max(1.0) as f64 / 30.0).powi(4);
    let weight: Vec<f64> = (0..count)
        .map(|frame| if planted[frame] { 1.0 } else { 0.02 })
        .collect();
    match banded_grounding(&curve, &weight, smooth) {
        Some(solved) => {
            for frame in 0..count {
                out[frame] = (solved[frame].clamp(-GROUND_LIMIT, GROUND_LIMIT)) as f32;
            }
            out
        }
        None => {
            // Not positive definite (a degenerate clip): the interpolated curve is
            // still a valid answer, just unsmoothed.
            for frame in 0..count {
                out[frame] = (curve[frame].clamp(-GROUND_LIMIT, GROUND_LIMIT)) as f32;
            }
            out
        }
    }
}

/// Solve `(W + smooth·D₂ᵀD₂) x = W·curve` for a five-band symmetric positive
/// definite system, by banded Cholesky. The matrix has a bandwidth of two: the
/// diagonal is `w + smooth·[1,5,6,…,6,5,1]`, the first off-diagonal
/// `-2·smooth·[1,2,…,2,1]`, the second `smooth`.
fn banded_grounding(curve: &[f64], weight: &[f64], smooth: f64) -> Option<Vec<f64>> {
    let count = curve.len();
    if count < 3 {
        return Some(curve.to_vec());
    }
    let mut diagonal = vec![0.0f64; count];
    let mut first = vec![0.0f64; count - 1];
    let mut second = vec![0.0f64; count - 2];
    for frame in 0..count {
        // `[1,5,6,...,6,5,1]`: the interior of `D₂ᵀD₂` is 6, its ends 5 and 1.
        let shape = if frame == 0 || frame + 1 == count {
            1.0
        } else if frame == 1 || frame + 2 == count {
            5.0
        } else {
            6.0
        };
        diagonal[frame] = weight[frame] + smooth * shape;
    }
    for frame in 0..count - 1 {
        let shape = if frame == 0 || frame + 2 == count {
            1.0
        } else {
            2.0
        };
        first[frame] = -2.0 * smooth * shape;
    }
    for slot in second.iter_mut() {
        *slot = smooth;
    }

    // Lower-triangular band of the Cholesky factor: the diagonal and the two
    // sub-diagonals.
    let at = |row: usize, column: usize| -> f64 {
        match row - column {
            0 => diagonal[row],
            1 => first[column],
            2 => second[column],
            _ => 0.0,
        }
    };
    let mut lower_diagonal = vec![0.0f64; count];
    let mut lower_first = vec![0.0f64; count];
    let mut lower_second = vec![0.0f64; count];
    for row in 0..count {
        for column in row.saturating_sub(2)..=row {
            let mut sum = at(row, column);
            for inner in row.saturating_sub(2)..column {
                let a = match row - inner {
                    0 => lower_diagonal[row],
                    1 => lower_first[row],
                    _ => lower_second[row],
                };
                let b = match column - inner {
                    0 => lower_diagonal[column],
                    1 => lower_first[column],
                    _ => lower_second[column],
                };
                sum -= a * b;
            }
            if row == column {
                if sum <= 0.0 || !sum.is_finite() {
                    return None;
                }
                lower_diagonal[row] = sum.sqrt();
            } else {
                let pivot = lower_diagonal[column];
                if pivot.abs() < 1e-12 {
                    return None;
                }
                let value = sum / pivot;
                match row - column {
                    1 => lower_first[row] = value,
                    _ => lower_second[row] = value,
                }
            }
        }
    }

    // `L y = W·curve`, then `Lᵀ x = y`.
    let mut y = vec![0.0f64; count];
    for row in 0..count {
        let mut sum = weight[row] * curve[row];
        for inner in row.saturating_sub(2)..row {
            let a = match row - inner {
                1 => lower_first[row],
                _ => lower_second[row],
            };
            sum -= a * y[inner];
        }
        y[row] = sum / lower_diagonal[row];
    }
    let mut x = vec![0.0f64; count];
    for row in (0..count).rev() {
        let mut sum = y[row];
        for inner in row + 1..(row + 3).min(count) {
            let a = match inner - row {
                1 => lower_first[inner],
                _ => lower_second[inner],
            };
            sum -= a * x[inner];
        }
        x[row] = sum / lower_diagonal[row];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A leg that spans 0.95 m, with the hip `above` metres over the target.
    fn frame(above: f64, active: bool) -> Frame {
        Frame {
            hips: [[0.09, above, 0.0], [-0.09, above, 0.0]],
            targets: [[0.09, 0.0, 0.0], [-0.09, 0.0, 0.0]],
            high: [0.95, 0.95],
            low: [0.20, 0.20],
            active: [active, active],
            flight: false,
            lowest: 0.0,
            strength: 1.0,
        }
    }

    /// A crouch deeper than the leg can span lowers the body by the shortfall.
    #[test]
    fn a_too_far_target_lowers_the_body() {
        let frames = vec![frame(1.20, true); 10];
        let delta = plan(&frames, 0.0, 30.0, true, 0.0);
        let last = delta[9];
        // 1.20 m hip over the target, 0.95 m reach: 0.25 m short.
        assert!(
            (last + DROP_LIMIT).abs() < 1e-2,
            "expected about -0.25 m, got {last:.4}"
        );
    }

    /// The enveloped drop eases in: the frame before a single bad frame still
    /// moves, which is what stops the root from stepping.
    #[test]
    fn the_drop_is_enveloped() {
        let mut frames = vec![frame(0.90, true); 20];
        frames[10] = frame(1.20, true);
        let delta = plan(&frames, 0.0, 30.0, true, 0.0);
        assert!(delta[9] < -1e-3, "the frame before the drop did not move");
        assert!(delta[11] < -1e-3, "the frame after the drop did not move");
    }

    /// A leg folded tighter than its minimum fold raises the body, capped at
    /// 12 cm, and only over contact frames.
    #[test]
    fn a_too_close_target_raises_the_body() {
        // 0.10 m hip over the target: below the 0.20 m minimum fold.
        let frames = vec![frame(0.10, true); 10];
        let delta = plan(&frames, 0.0, 30.0, true, 0.0);
        assert!(delta[9] > 1e-3, "the body should have risen: {:.4}", delta[9]);
        assert!(delta[9] <= LIFT_LIMIT + 1e-9, "lift went past the cap");
    }

    #[test]
    fn a_free_leg_never_moves_the_root() {
        let frames = vec![frame(1.20, false); 10];
        let delta = plan(&frames, 0.0, 30.0, true, 0.0);
        assert!(delta.iter().all(|value| value.abs() < 1e-12));
    }

    /// A positive user offset only takes back what the drop took.
    #[test]
    fn a_positive_offset_is_capped_by_the_automatic_drop() {
        let frames = vec![frame(1.20, true); 10];
        let without = plan(&frames, 0.0, 30.0, true, 0.0);
        let with = plan(&frames, 0.0, 30.0, true, 0.10);
        assert!(
            with[9] > without[9] + 0.05,
            "a positive offset should raise the body: {:.3} vs {:.3}",
            with[9],
            without[9]
        );
        assert!(with[9] < 1e-9, "it must never rise above the un-dropped root");
    }

    /// With nothing to drop, a positive offset does nothing at all — the slider
    /// may only give back automatic compensation.
    #[test]
    fn a_positive_offset_does_nothing_without_a_drop() {
        let frames = vec![frame(0.85, true); 10];
        let delta = plan(&frames, 0.0, 30.0, true, 0.20);
        assert!(delta.iter().all(|value| value.abs() < 1e-9));
    }

    /// A negative offset lowers the body, faded by the contact strength so an
    /// airborne take is not moved.
    #[test]
    fn a_negative_offset_fades_with_the_contacts() {
        let mut frames = vec![frame(0.85, true); 10];
        for frame in frames.iter_mut() {
            frame.strength = 0.5;
        }
        let delta = plan(&frames, 0.0, 30.0, true, -0.10);
        assert!((delta[9] + 0.05).abs() < 1e-6, "got {:.4}", delta[9]);
        for frame in frames.iter_mut() {
            frame.active = [false, false];
            frame.strength = 0.0;
        }
        let delta = plan(&frames, 0.0, 30.0, true, -0.10);
        assert!(delta.iter().all(|value| value.abs() < 1e-12));
    }

    /// A wholly airborne frame over the plane gets lifted, by at most 12 cm.
    #[test]
    fn an_airborne_frame_is_lifted_out_of_the_plane() {
        let mut frames = vec![frame(0.85, true); 10];
        // One frame in the air with the soles 5 cm under the plane.
        frames[5].active = [false, false];
        frames[5].flight = true;
        frames[5].lowest = -0.05;
        let delta = plan(&frames, 0.0, 30.0, true, 0.0);
        assert!((delta[5] - 0.05).abs() < 1e-6, "got {:.4}", delta[5]);

        frames[5].lowest = -0.40;
        let delta = plan(&frames, 0.0, 30.0, true, 0.0);
        assert!((delta[5] - CLEARANCE_LIMIT).abs() < 1e-6, "got {:.4}", delta[5]);
    }

    /// A take with no contact at all is left exactly where it is.
    #[test]
    fn a_contact_free_take_is_never_moved() {
        let mut frames = vec![frame(0.85, false); 10];
        for frame in frames.iter_mut() {
            frame.flight = true;
            frame.lowest = -0.2;
            // `contact_strength` is zero outside every stance, which is what
            // stops a negative user offset from moving an airborne take.
            frame.strength = 0.0;
        }
        let delta = plan(&frames, 0.0, 30.0, true, -0.1);
        assert!(delta.iter().all(|value| value.abs() < 1e-12));
    }

    /// A sole hovering a constant 2 cm over the plane moves the body down by
    /// exactly that, with no residual.
    #[test]
    fn grounding_lands_a_hovering_sole_on_the_plane() {
        let heights = vec![0.02f32; 12];
        let planted = vec![true; 12];
        let shift = grounding(&heights, &planted, 0.0, 30.0);
        for value in &shift {
            assert!((*value + 0.02).abs() < 1e-6, "got {value:.6}");
        }
    }

    /// A sole *under* the plane raises the body the same way. The shift is the
    /// negative of the error in both directions, which is what makes it a
    /// grounding step and not a lift.
    #[test]
    fn grounding_raises_a_sunken_sole() {
        let heights = vec![-0.015f32; 12];
        let planted = vec![true; 12];
        let shift = grounding(&heights, &planted, 0.0, 30.0);
        for value in &shift {
            assert!((*value - 0.015).abs() < 1e-6, "got {value:.6}");
        }
    }

    /// The grounding follows the planted soles and interpolates between stances:
    /// two stances hovering at different heights give two different shifts, and
    /// the swing between them ramps instead of stepping.
    #[test]
    fn grounding_follows_each_stance_and_ramps_between_them() {
        let count = 30usize;
        let mut heights = vec![0.0f32; count];
        let mut planted = vec![false; count];
        for frame in 0..10 {
            heights[frame] = 0.02;
            planted[frame] = true;
        }
        for frame in 20..count {
            heights[frame] = 0.04;
            planted[frame] = true;
        }
        let shift = grounding(&heights, &planted, 0.0, 30.0);
        // Each stance is pulled down by its own hover height.
        assert!((shift[0] + 0.02).abs() < 5e-3, "first stance {:.4}", shift[0]);
        assert!((shift[29] + 0.04).abs() < 5e-3, "second stance {:.4}", shift[29]);
        // And the swing between them is a ramp, not a jump.
        for frame in 1..count {
            assert!(
                (shift[frame] - shift[frame - 1]).abs() < 5e-3,
                "frame {frame} stepped by {:.4}",
                shift[frame] - shift[frame - 1]
            );
        }
    }

    /// No planted foot means no estimate at all: the clip keeps its own height.
    #[test]
    fn grounding_needs_a_planted_foot() {
        let heights = vec![0.3f32; 6];
        let planted = vec![false; 6];
        let shift = grounding(&heights, &planted, 0.0, 30.0);
        assert!(shift.iter().all(|value| *value == 0.0));
    }

    /// The reference clamps the shift to ±0.75 m; a wild sole height must not
    /// teleport the body.
    #[test]
    fn grounding_is_bounded() {
        let heights = vec![3.0f32; 8];
        let planted = vec![true; 8];
        let shift = grounding(&heights, &planted, 0.0, 30.0);
        for value in &shift {
            assert!((*value + GROUND_LIMIT as f32).abs() < 1e-4, "got {value:.4}");
        }
    }
}
