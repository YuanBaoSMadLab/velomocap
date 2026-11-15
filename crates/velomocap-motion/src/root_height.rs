//! Root-height planning, ported from `correction.py::correct_arrays` (RGB MoCap,
//! MIT), steps 1–4 of the vertical chain:
//!
//! ```text
//! 1  drop      reachable_pelvis_drop   the legs cannot span the crouch: lower the body
//!              (enveloped: max over 0.1 s, then gaussian 0.03 s)
//! 2  lift      reachable_pelvis_lift   the legs cannot fold that far: raise the body
//! 3  offset    pelvis_offset           the user's "Pelvis Height Offset (from Auto)"
//! 4  clearance flight                  a whole airborne body must not clip the plane
//! ```
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
}
