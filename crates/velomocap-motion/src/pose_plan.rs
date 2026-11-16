//! Root-height planning window, ported from the reference's
//! `contact_pose_plan.py::_optimise_window` (RGB MoCap, MIT).
//!
//! The reference's pose plan does not move the legs: it plans the **root height**
//! over a window, predicting the leg solve's response, so that
//!
//! ```text
//! stay near the settled root          |x - base|            sigma 1 cm
//! never ask a planted leg to over-reach  max(reach - high)   /0.5 mm
//! never ask a planted leg to over-bend   min(reach - low)    /0.5 mm
//! no root kick                          second difference    sigma 0.4 mm
//! ```
//!
//! where `high` is `reach_lengths` and `low` is `minimum_reach_lengths`, the same
//! bounded chain lengths `anatomical_ik.py` solves against.
//!
//! Two reductions against the reference, both deliberate:
//!
//! * the planned variable is the root height alone. The reference also carries one
//!   extra height per swing target (bounded ±2 cm) plus a predicted (flex, hip
//!   rotation) state; that state exists to regularise the joint solve, and this
//!   pass instead bounds the root against the same reach limits directly.
//! * the warm start is the root our own ground pass already settled, where the
//!   reference warm starts from `contact_temporal_root.optimise`.
//!
//! A free foot that would clip the plane is **not** handled here: the reference
//! lifts that foot's own target, and so does `correction::swing_lift`. This plan
//! used to lift the body for it, which also moved the planted foot off its
//! contact. `Frame::lowest` is kept for the caller's diagnostics.

use crate::solver;

/// One frame of input, in metres.
#[derive(Clone, Copy, Debug)]
pub struct Frame {
    /// Root height before planning.
    pub base: f64,
    /// Hip positions of both legs (x/z fixed; y moves with the root).
    pub hips: [[f64; 3]; 2],
    /// The ankle target each leg is pinned to (x/z fixed).
    pub ankles: [[f64; 3]; 2],
    /// Lowest sole height of the whole body at this frame.
    pub lowest: f64,
    /// Contact per leg.
    pub active: [bool; 2],
    /// Max reach per leg (`anatomical::reach`).
    pub high: [f64; 2],
    /// Min reach per leg (`anatomical::minimum_reach`).
    pub low: [f64; 2],
    /// Airborne: no support at all, so nothing may be lifted for.
    pub flight: bool,
}

const ROOT_SIGMA: f64 = 0.01;
const ROOT_ACCELERATION_SIGMA: f64 = 0.0004;
const REACH_SIGMA: f64 = 0.0005;
const ROOT_LIMIT: f64 = 0.02;

/// Plan the root height. Returns one height per frame.
pub fn plan(frames: &[Frame], floor: f64, fps: f64, _flat: bool) -> Vec<f64> {
    let count = frames.len();
    if count < 3 {
        return frames.iter().map(|frame| frame.base).collect();
    }
    let rate = fps.max(1.0) / 30.0;
    let base: Vec<f64> = frames.iter().map(|frame| frame.base).collect();
    let start = base.clone();
    // Every frame stays inside 2 cm of the settled root, which is also where the
    // reference's `lo`/`hi` come from. A frame with a planted foot additionally
    // may not go below the floor: the plan exists to satisfy reach bounds, not to
    // sink a sole into the ground, and the floor is already satisfied at `base`.
    let lower: Vec<f64> = base
        .iter()
        .enumerate()
        .map(|(frame, value)| {
            let bound = value - ROOT_LIMIT;
            if frames[frame].active[0] || frames[frame].active[1] {
                bound.max(floor - frames[frame].lowest)
            } else {
                bound
            }
        })
        .collect();
    let upper: Vec<f64> = base.iter().map(|value| value + ROOT_LIMIT).collect();

    let residual = |x: &[f64]| -> Vec<f64> {
        let mut out: Vec<f64> = Vec::with_capacity(count * 6);
        // Stay near the settled root.
        for frame in 0..count {
            out.push((x[frame] - base[frame]) / ROOT_SIGMA);
        }
        // A planted leg may neither over-extend nor over-bend.
        for frame in 0..count {
            let source = &frames[frame];
            for leg in 0..2 {
                if !source.active[leg] {
                    continue;
                }
                let hip = [
                    source.hips[leg][0],
                    source.hips[leg][1] + (x[frame] - source.base),
                    source.hips[leg][2],
                ];
                let target = source.ankles[leg];
                let reach = ((target[0] - hip[0]).powi(2)
                    + (target[1] - hip[1]).powi(2)
                    + (target[2] - hip[2]).powi(2))
                .sqrt();
                let over = (reach - source.high[leg] * 0.9995).max(0.0);
                let under = (reach - source.low[leg] * 1.0005).min(0.0);
                out.push((over + under) / REACH_SIGMA);
            }
        }
        // No root kick.
        for frame in 0..count - 2 {
            let second = x[frame + 2] - 2.0 * x[frame + 1] + x[frame];
            out.push(second * rate * rate / ROOT_ACCELERATION_SIGMA);
        }
        // A free foot that would clip the floor is *the foot's* problem: the leg
        // solve lifts it (`correction::swing_lift`, the reference's guard). This
        // plan used to lift the whole body for it, which also moved the planted
        // foot and broke its contact. `lowest` is kept in the frame struct because
        // a future pass may want it, but the root does not react to it.
        out
    };

    let solution = solver::minimise(
        &start,
        &lower,
        &upper,
        residual,
        solver::Options {
            max_iterations: 60,
            tolerance: 1e-9,
        },
    );
    solution
        .x
        .iter()
        .enumerate()
        .map(|(frame, value)| value.clamp(base[frame] - ROOT_LIMIT, base[frame] + ROOT_LIMIT))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(base: f64) -> Frame {
        Frame {
            base,
            // Hips 0.85 m above the ankle targets, which is inside the reach
            // window unless a test says otherwise.
            hips: [
                [0.09, base + 0.85, 0.0],
                [-0.09, base + 0.85, 0.0],
            ],
            ankles: [[0.09, base, 0.0], [-0.09, base, 0.0]],
            // The sole rests on the floor.
            lowest: 0.0,
            active: [true, true],
            high: [0.95, 0.95],
            low: [0.20, 0.20],
            flight: false,
        }
    }

    /// A leg asked to stretch past its reach makes the plan lower the root, and
    /// the reach it ends up with is inside the limit.
    #[test]
    fn lowers_the_root_when_a_leg_would_over_extend() {
        let mut frames: Vec<Frame> = (0..12).map(|_| frame(0.9)).collect();
        for source in frames.iter_mut() {
            // The hip sits 0.96 m above the ankle: 1 cm past the 0.95 m reach.
            for leg in 0..2 {
                source.hips[leg][1] = source.base + 0.96;
            }
        }
        let planned = plan(&frames, 0.0, 30.0, true);
        let last = *planned.last().unwrap();
        assert!(
            last < frames[frames.len() - 1].base - 0.005,
            "the root should have dropped: {last:.3} vs base {:.3}",
            frames[frames.len() - 1].base
        );
        let hip = frames[frames.len() - 1].hips[0][1] - (frames[frames.len() - 1].base - last);
        let reach = hip - frames[frames.len() - 1].ankles[0][1];
        assert!(reach <= 0.95 * 0.9995 + 2e-3, "reach still {reach:.3} m");
    }

    /// A free foot under the floor does **not** raise the body: the reference puts
    /// that guard on the foot's own target, and this plan only bounds the root
    /// against the legs' reach. Lifting the body here would move the planted foot
    /// off its contact, which is worse than the scrape it was trying to fix.
    #[test]
    fn a_free_foot_under_the_floor_does_not_move_the_root() {
        let mut frames: Vec<Frame> = (0..12).map(|_| frame(0.9)).collect();
        for (index, source) in frames.iter_mut().enumerate() {
            source.active = [false, false];
            // A swing frame whose sole is 4 cm under the plane.
            source.lowest = if index == 6 { -0.04 } else { 0.0 };
        }
        let planned = plan(&frames, 0.0, 30.0, true);
        for (index, value) in planned.iter().enumerate() {
            assert!(
                (value - frames[index].base).abs() < 1e-6,
                "frame {index} moved {:.4} m",
                value - frames[index].base
            );
        }
    }

    /// Nothing to fix means nothing moves, which keeps the default path stable.
    #[test]
    fn keeps_a_settled_root_when_nothing_violates() {
        let frames: Vec<Frame> = (0..16).map(|_| frame(0.9)).collect();
        let planned = plan(&frames, 0.0, 30.0, true);
        for (index, value) in planned.iter().enumerate() {
            assert!(
                (value - frames[index].base).abs() < 1e-3,
                "frame {index} moved {:.4} m",
                value - frames[index].base
            );
        }
    }

    /// The 2 cm bound is hard, even when every frame asks for more.
    #[test]
    fn never_moves_a_frame_further_than_two_centimetres() {
        let mut frames: Vec<Frame> = (0..12).map(|_| frame(0.9)).collect();
        for source in frames.iter_mut() {
            for leg in 0..2 {
                source.hips[leg][1] = source.base + 1.20;
            }
        }
        let planned = plan(&frames, 0.0, 30.0, true);
        for (index, value) in planned.iter().enumerate() {
            assert!(
                (value - frames[index].base).abs() <= ROOT_LIMIT + 1e-9,
                "frame {index} moved {:.4} m",
                value - frames[index].base
            );
        }
    }
}




