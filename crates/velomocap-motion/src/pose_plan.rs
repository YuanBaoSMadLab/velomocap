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
    /// Lowest sole of each foot at this frame, before the plan moves anything.
    pub soles: [f64; 2],
    /// Lowest sole height of the whole body at this frame.
    pub lowest: f64,
    /// Contact per leg.
    pub active: [bool; 2],
    /// `active | transition`: the frames the reference's plan regularises.
    pub planned: [bool; 2],
    /// `free_follow_weights`: 1 where the foot rides the planned root (a free
    /// foot), 0 where it is held (a stance), and a C1 ramp across a hand-off.
    pub follow: [f64; 2],
    /// A release/approach frame (	ransition & !active): the clearance term may
    /// lift its goal out of the floor, but the knee-angle regulariser does not
    /// chase it — regularising the hand-offs moved the body for a millimetre of
    /// sole clearance (2月27日.mp4: jitter 29.1 -> 31.2 mm).
    pub handoff: [bool; 2],
    /// planned & !active & !flight: this frame's target height is a variable
    /// of the plan, bounded to ±2 cm — a hand-off may move its goal, a stance
    /// may not.
    pub movable: [bool; 2],
    /// Max reach per leg (`anatomical::reach`).
    pub high: [f64; 2],
    /// Min reach per leg (`anatomical::minimum_reach`).
    pub low: [f64; 2],
    /// Airborne: no support at all, so nothing may be lifted for.
    pub flight: bool,
    /// The knee geometry of each leg (`anatomical::knee_plane`), which is what
    /// turns a root height into the knee angle the solver will produce.
    pub knee: [Knee; 2],
}

/// What the plan decided.
#[derive(Clone, Debug, Default)]
pub struct Plan {
    /// Root height per frame.
    pub root: Vec<f64>,
    /// Planned absolute height of each ankle target, per frame per leg, in the
    /// same space as [`Frame::ankles`] (i.e. before the root shift this pass
    /// applies). The stance entries are the input; the hand-off entries carry the
    /// plan's own target variable and the floor clearance, and are what the leg
    /// IK should be asked for.
    pub goal_y: Vec<[f64; 2]>,
    /// How many target-height variables the solve carried.
    pub targets: usize,
}

/// One leg's bend-plane geometry and how fast the source moved through it.
#[derive(Clone, Copy, Debug)]
pub struct Knee {
    /// `anatomical::knee_plane`'s constant term, metres².
    pub constant: f64,
    /// Its bend radius, metres.
    pub radius: f64,
    /// Its phase, radians: where the source knee already is in the plane.
    pub phase: f64,
    /// Source knee angular speed, radians per second: a leg that is already
    /// turning keeps its motion, a leg that is still must not acquire any.
    pub speed: f64,
}

const ROOT_SIGMA: f64 = 0.01;
const ROOT_ACCELERATION_SIGMA: f64 = 0.0004;
const ROOT_LIMIT: f64 = 0.02;
/// How far a hand-off frame may move its own target height (`target_limit`).
const TARGET_LIMIT: f64 = 0.02;
/// `target_sigma`: a hand-off target is not free to wander either.
const TARGET_SIGMA: f64 = 0.02;
/// `contact_pose_plan.CONFIG`: pose, velocity, acceleration and the shared
/// temporal weight. The pose term is the predicted knee angle; the two difference
/// terms are what keep the plan from buying height with a knee twitch.
const POSE_SIGMA: f64 = 0.175;
const VELOCITY_SIGMA: f64 = 0.025;
const ACCELERATION_SIGMA: f64 = 0.004;
const TEMPORAL_WEIGHT: f64 = 0.5;
/// The source speed at which the "static" weight has fallen by half: 20 deg/s.
const STILL_DEG_PER_S: f64 = 20.0;

/// The knee angle a root shift implies, for one leg.
///
/// `contact_temporal_root.optimise`'s `angle`, without its mask: the bend plane's
/// phase blended out where the source knee is nearly straight, plus the arc
/// cosine of the reach triangle. `horizontal` is the squared XZ hip-to-target
/// distance, `over` the hip's height above the target at `base`, `straight` the
/// 0..1 extension support, and `shift` how far the root moves from `base`.
pub fn knee_angle(knee: &Knee, horizontal: f64, over: f64, straight: f64, shift: f64) -> f64 {
    let vertical = over + shift;
    let ratio = (horizontal + vertical * vertical - knee.constant) / (2.0 * knee.radius);
    let bent = ratio.clamp(-0.999_999_999, 0.999_999_999).acos();
    knee.phase * (1.0 - straight) + bent
}

/// Plan the root height and every hand-off frame's target height.
///
/// This is `contact_pose_plan._optimise_window`: the variables are the root height
/// and one target height per hand-off frame (bounded to ±2 cm), and the objective
/// carries the **predicted knee angle** of every planned frame — its offset, its
/// velocity and its acceleration — weighted by how still the source leg was.
///
/// The hand-off freedom is the point. Holding the hand-off frames to the same
/// knee-angle smoothness with the root as the only variable just pushed the root
/// around (measured: root jitter 5.2 -> 19.2 mm on `2月27日.mp4`); with their own
/// bounded target the plan can keep the leg still without moving the body.
pub fn plan(frames: &[Frame], floor: f64, fps: f64, flat: bool) -> Plan {
    let count = frames.len();
    let mut out = Plan {
        root: frames.iter().map(|frame| frame.base).collect(),
        goal_y: frames
            .iter()
            .map(|frame| [frame.ankles[0][1], frame.ankles[1][1]])
            .collect(),
        targets: 0,
    };
    if count < 3 {
        return out;
    }
    let fps = fps.max(1.0);
    let rate = fps / 30.0;
    let base: Vec<f64> = frames.iter().map(|frame| frame.base).collect();

    // Per frame: the squared horizontal hip-to-target distance, and the hip's
    // height over the target. Both are what the knee-angle prediction needs.
    let mut horizontal = vec![[0.0f64; 2]; count];
    let mut over = vec![[0.0f64; 2]; count];
    // Per frame: the root shift each planted leg's reach allows, most restrictive
    // wins. `contact_temporal_root`:
    //   upper = target_y + sqrt((reach·.9995)² − horizontal) − hip_y
    //   lower = target_y + sqrt((min·1.0005)² − horizontal) − hip_y
    let mut lowest_shift = vec![f64::NEG_INFINITY; count];
    let mut highest_shift = vec![f64::INFINITY; count];
    for frame in 0..count {
        let source = &frames[frame];
        for leg in 0..2 {
            let hip = source.hips[leg];
            let target = source.ankles[leg];
            horizontal[frame][leg] = (hip[0] - target[0]).powi(2) + (hip[2] - target[2]).powi(2);
            over[frame][leg] = hip[1] - target[1];
            if !source.active[leg] || over[frame][leg] <= 0.0 {
                continue;
            }
            let square = horizontal[frame][leg];
            let upper = target[1]
                + ((source.high[leg] * 0.9995).powi(2) - square).max(0.0).sqrt()
                - hip[1];
            let lower = target[1]
                + ((source.low[leg] * 1.0005).powi(2) - square).max(0.0).sqrt()
                - hip[1];
            lowest_shift[frame] = lowest_shift[frame].max(lower);
            highest_shift[frame] = highest_shift[frame].min(upper);
        }
    }

    let mut lower = vec![0.0f64; count];
    let mut upper = vec![0.0f64; count];
    for frame in 0..count {
        // The shifts are relative to the settled root; the plan works in absolute
        // heights, so the reach bounds move with `base` too.
        let (low, high) = (
            (base[frame] - ROOT_LIMIT).max(base[frame] + lowest_shift[frame]),
            (base[frame] + ROOT_LIMIT).min(base[frame] + highest_shift[frame]),
        );
        let planned = frames[frame].planned[0] || frames[frame].planned[1];
        if !planned
            || !lowest_shift[frame].is_finite()
            || !highest_shift[frame].is_finite()
            || high - low < 1e-7
        {
            // Nothing planned here, airborne, or no room: the frame keeps the
            // root it settled at (`contact_pose_plan`'s `freeze`).
            lower[frame] = base[frame] - 1e-10;
            upper[frame] = base[frame] + 1e-10;
        } else {
            lower[frame] = low;
            upper[frame] = high;
        }
    }

    // The target-height variables: one per hand-off frame that is neither held by
    // a stance nor airborne (`movable`), scaled by `target_gain = 1 - follow`.
    let mut slot = vec![[usize::MAX; 2]; count];
    let mut movable = Vec::new();
    for frame in 0..count {
        for leg in 0..2 {
            let gain = 1.0 - frames[frame].follow[leg];
            if frames[frame].movable[leg] && gain > 1e-6 {
                slot[frame][leg] = movable.len();
                movable.push((frame, leg, gain));
            }
        }
    }
    let movable_count = movable.len();

    // `static_weight = .25 + .75/(1 + (speed/20°)²)`: a knee that is already
    // moving is allowed to keep moving; a still one is what the plan smooths.
    let still = |frame: usize, leg: usize| -> f64 {
        let speed = frames[frame].knee[leg].speed;
        0.25 + 0.75 / (1.0 + (speed / STILL_DEG_PER_S.to_radians()).powi(2))
    };
    // Near-collinear knees (|phase| under 5°) are counted as straight, which is
    // where the arc-cosine is insensitive and chasing it would be noise.
    let straight = |frame: usize, leg: usize| -> f64 {
        let phase = frames[frame].knee[leg].phase.abs();
        ((5.0f64.to_radians() - phase) / 4.0f64.to_radians()).clamp(0.0, 1.0)
    };

    // The goal the plan asks for, as a height offset from the source ankle: a free
    // foot rides the planned root (`follow`), a hand-off frame adds its own
    // variable, and a planned free foot is lifted out of the floor.
    let goal_offset = |frame: usize, leg: usize, x: &[f64]| -> f64 {
        let source = &frames[frame];
        let root = x[frame];
        let mut offset = source.follow[leg] * (root - base[frame]);
        let at = slot[frame][leg];
        if at != usize::MAX {
            offset += x[count + at] * movable[at].2;
        }
        if flat && source.handoff[leg] && !source.flight {
            // `predict()`'s clearance: the sole a free planned foot would reach,
            // lifted back to the plane with a 1 cm grazing tolerance.
            let gap = floor - (source.soles[leg] + offset);
            let lift = if gap >= 0.01 {
                gap
            } else if gap <= -0.01 {
                0.0
            } else {
                (gap + 0.01).powi(2) / 0.04
            };
            offset += lift.min(0.12);
        }
        offset
    };

    let residual = |x: &[f64]| -> Vec<f64> {
        let mut out: Vec<f64> = Vec::with_capacity(count * 10 + movable_count);
        // 1. Stay near the settled root.
        for frame in 0..count {
            out.push((x[frame] - base[frame]) / ROOT_SIGMA);
        }
        // 2. Keep the hand-off target heights near the source.
        for at in 0..movable_count {
            out.push(x[count + at] / TARGET_SIGMA);
        }
        // 3. No knee motion the source did not have: pose, then its first and
        //    second differences, both faded where the source knee was moving.
        //    The state is the flex *change* from the source bend
        //    (`angle − phase·extension_support`), so a leg the plan does not have
        //    to move leaves a zero residual.
        let mut states = vec![[0.0f64; 2]; count];
        for frame in 0..count {
            for leg in 0..2 {
                let source = &frames[frame];
                if !source.planned[leg] {
                    continue;
                }
                let offset = goal_offset(frame, leg, x);
                let shift = (x[frame] - base[frame]) - offset;
                let angle = knee_angle(
                    &source.knee[leg],
                    horizontal[frame][leg],
                    over[frame][leg],
                    straight(frame, leg),
                    shift,
                );
                states[frame][leg] = angle - source.knee[leg].phase * straight(frame, leg);
            }
        }
        let planned = |frame: usize, leg: usize| -> f64 {
            if frames[frame].planned[leg] {
                1.0
            } else {
                0.0
            }
        };
        for frame in 0..count {
            for leg in 0..2 {
                if frames[frame].planned[leg] {
                    out.push(states[frame][leg] / POSE_SIGMA);
                }
            }
        }
        for frame in 0..count - 1 {
            for leg in 0..2 {
                let weight = planned(frame, leg).min(planned(frame + 1, leg))
                    * still(frame, leg).min(still(frame + 1, leg)).sqrt()
                    * rate
                    * TEMPORAL_WEIGHT
                    / VELOCITY_SIGMA;
                out.push((states[frame + 1][leg] - states[frame][leg]) * weight);
            }
        }
        for frame in 0..count - 2 {
            for leg in 0..2 {
                let common = planned(frame, leg)
                    .min(planned(frame + 1, leg))
                    .min(planned(frame + 2, leg));
                let common_still = still(frame, leg)
                    .min(still(frame + 1, leg))
                    .min(still(frame + 2, leg));
                let weight = common
                    * common_still.sqrt()
                    * rate
                    * rate
                    * TEMPORAL_WEIGHT
                    / ACCELERATION_SIGMA;
                let second =
                    states[frame + 2][leg] - 2.0 * states[frame + 1][leg] + states[frame][leg];
                out.push(second * weight);
            }
        }
        // 4. No root kick.
        for frame in 0..count - 2 {
            let second = x[frame + 2] - 2.0 * x[frame + 1] + x[frame];
            out.push(second * rate * rate / ROOT_ACCELERATION_SIGMA);
        }
        out
    };

    let mut start: Vec<f64> = base
        .iter()
        .enumerate()
        .map(|(frame, value)| value.clamp(lower[frame] + 1e-12, upper[frame] - 1e-12))
        .collect();
    start.resize(count + movable_count, 0.0);
    let mut box_lower = lower.clone();
    box_lower.resize(count + movable_count, -TARGET_LIMIT);
    let mut box_upper = upper.clone();
    box_upper.resize(count + movable_count, TARGET_LIMIT);

    let solution = solver::minimise(
        // The reference's trust region keeps the iterate feasible; this solver
        // projects its steps, so the initial guess has to start inside the box or
        // a frame whose reach forbids its settled height stays put. Clamping the
        // start is what makes a hard bound hard.
        &start,
        &box_lower,
        &box_upper,
        residual,
        solver::Options {
            max_iterations: 60,
            tolerance: 1e-9,
        },
    );
    let x = &solution.x;
    for frame in 0..count {
        out.root[frame] = x[frame].clamp(base[frame] - ROOT_LIMIT, base[frame] + ROOT_LIMIT);
        for leg in 0..2 {
            out.goal_y[frame][leg] = frames[frame].ankles[leg][1] + goal_offset(frame, leg, x);
        }
    }
    out.targets = movable_count;
    out
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
            soles: [0.0, 0.0],
            // The sole rests on the floor.
            lowest: 0.0,
            active: [true, true],
            planned: [true, true],
            handoff: [false, false],
            follow: [0.0, 0.0],
            movable: [false, false],
            high: [0.95, 0.95],
            low: [0.20, 0.20],
            flight: false,
            // A leg whose source knee already sits 30° into its bend plane, which
            // is the configuration the plan reads as "no added correction": the
            // predicted flex at `base` cancels the phase, so `angle(base) == 0`.
            // Getting this wrong makes the plan chase an angle the source never
            // had, which moves the root for no reason.
            knee: [consistent_knee(), consistent_knee()],
        }
    }

    /// The geometry of a leg in its source configuration, hips 0.85 m over target.
    fn consistent_knee() -> Knee {
        let radius = 0.02f64;
        let over = 0.85f64;
        let phase = -30.0f64.to_radians();
        Knee {
            // So `arccos((over² − constant)/(2·radius))` is `−phase`: the flex the
            // solver would add is exactly zero, which is what "no correction" means.
            constant: over * over - 2.0 * radius * phase.abs().cos(),
            radius,
            phase,
            speed: 0.0,
        }
    }

    /// The angle the plan prices: 90° when the hip is where the geometry expects
    /// it, larger as the root comes down, and blended toward the bend-plane phase
    /// where the source knee is nearly straight.
    #[test]
    fn the_knee_angle_grows_as_the_root_lowers() {
        let knee = Knee {
            constant: 0.85 * 0.85,
            radius: 0.02,
            phase: 0.0,
            speed: 0.0,
        };
        let at_base = knee_angle(&knee, 0.0, 0.85, 0.0, 0.0);
        assert!((at_base - std::f64::consts::FRAC_PI_2).abs() < 1e-6, "{at_base}");
        let lowered = knee_angle(&knee, 0.0, 0.85, 0.0, -0.02);
        assert!(lowered > at_base + 0.3, "lowering must bend the knee: {lowered:.3}");
        let raised = knee_angle(&knee, 0.0, 0.85, 0.0, 0.02);
        assert!(raised < at_base, "raising must straighten it: {raised:.3}");
        // A perfectly straight source knee drops its own phase everywhere.
        let phased = Knee {
            constant: 0.85 * 0.85,
            radius: 0.02,
            phase: 0.5,
            speed: 0.0,
        };
        let blended = knee_angle(&phased, 0.0, 0.85, 0.5, -0.02);
        let pure = knee_angle(&phased, 0.0, 0.85, 1.0, -0.02);
        let only_bend = knee_angle(
            &Knee {
                phase: 0.0,
                ..phased
            },
            0.0,
            0.85,
            1.0,
            -0.02,
        );
        assert!((pure - only_bend).abs() < 1e-9, "the phase must be dropped");
        assert!(blended > pure + 0.2, "half the phase must remain: {blended:.3}");
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
        let planned = plan(&frames, 0.0, 30.0, true).root;
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

    /// A hand-off frame whose sole would end up under the plane gets its **own**
    /// goal lifted, not the body: `contact_pose_plan`'s clearance term on a
    /// planned, non-active foot.
    #[test]
    fn a_handoff_goal_is_lifted_out_of_the_floor() {
        let mut frames: Vec<Frame> = (0..9).map(|_| frame(0.9)).collect();
        // Frame 4 is a release for the left foot: no stance, but a hand-off.
        frames[4].active = [false, true];
        frames[4].planned = [true, true];
        frames[4].handoff = [true, false];
        // Its sole would sit 4 cm under the plane.
        frames[4].soles = [-0.04, 0.0];
        let plan = plan(&frames, 0.0, 30.0, true);
        let lifter = plan.goal_y[4][0] - frames[4].ankles[0][1];
        assert!(
            lifter > 0.03,
            "hand-off goal only moved {lifter:.3} m, the foot stays buried"
        );
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
        let planned = plan(&frames, 0.0, 30.0, true).root;
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
        let planned = plan(&frames, 0.0, 30.0, true).root;
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
        let planned = plan(&frames, 0.0, 30.0, true).root;
        for (index, value) in planned.iter().enumerate() {
            assert!(
                (value - frames[index].base).abs() <= ROOT_LIMIT + 1e-9,
                "frame {index} moved {:.4} m",
                value - frames[index].base
            );
        }
    }
}







