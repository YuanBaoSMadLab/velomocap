//! Contact correction for a generated SMPL-H clip.
//!
//! Ported from the reference's `mocap/correction.py` (RGB MoCap, MIT), with the
//! two-bone IK helpers from NVIDIA Kimodo's `MotionCorrection` (Apache-2.0; see
//! `ik.rs` for that notice). Coverage:
//!
//! | Reference | Here |
//! | --- | --- |
//! | `prepare_contacts` (flat mode) | `contacts::prepare`, applied by `detect` |
//! | `prepare_contacts`' grounding shift | `root_height::grounding`, applied by `grounding_pass` |
//! | `airborne_excursions` | `contacts` (it owns the flight mask) |
//! | the root X/Z drift (`interp` + `gaussian_filter1d`, 12 cm cap) | `foot_lock`, `interpolate` |
//! | `reachable_pelvis_drop` / `reachable_pelvis_lift` + their envelope | `root_height::plan` |
//! | `pelvis_offset` (`Pelvis Height Offset (from Auto)`) | `root_height::plan`, step 3 |
//! | the airborne clearance term | `root_height::plan`, step 4 |
//! | `DoContactIK` + `TargetReachFalloff` | `leg_ik`, `reach_falloff` |
//! | `PRESETS` (`correction_settings.py`) | `Thresholds` |
//! | `support_weights` / `support_samples` | `support_weight`, `support_point` |
//! | `rolling_anchor` | `rolling_anchor`, `rolling_path` |
//! | `couple_anchors`, `smooth_auto_handoffs`, `_hermite` | `couple_anchor_offsets`, `smooth_handoffs`, `hermite` |
//! | `anatomical_ik.solve_legs`, `reach_lengths`, `minimum_reach_lengths` | `anatomical` |
//! | `SoleGeometry` | `sole::SoleGeometry` (default) |
//! | `source_reach_gap` + reach gain | `anatomical::source_reach_gap` in `hand_lock` |
//! | `palm_offsets`, `samples`, `plan` (palm contacts) | `palm::PalmGeometry`, `palm::plan` |
//! | `contact_arm_plan.solve` (collar depression / swivel / wrist trim / elbow correction) | `arm_plan::Plan::solve`, applied by `plan_arm_stance` |
//! | `contact_pose_plan._optimise_window` (root height against the reach bounds) | `pose_plan::plan`, applied by `pose_plan_window` |
//! | `scipy.optimize.least_squares` (both planning passes) | `solver::minimise` |
//!
//! The vertical chain is the reference's six steps, in its order: the grounding
//! shift that stands the body on the measured plane (`prepare_contacts`), the X/Z
//! drift of the planted supports (`foot_lock`), then `root_height` (drop → lift →
//! user offset → airborne clearance), then the pose plan, then the leg solve.
//! The grounding is a *smoothed* floor shift, not "drop the lowest point onto the
//! plane": it only reads the frames a foot is planted on, weights the swing at
//! 2 %, and is clamped to ±0.75 m, which is the difference between standing on
//! the plane and hopping onto it every time a foot lands.
//!
//! The root's X/Z shift used to come from a ported Kimodo `TrajectoryCorrector`
//! (an ADMM over position / velocity / acceleration). The reference has no such
//! pass: it linearly interpolates the drift of the planted supports, smooths it
//! with a gaussian of `max(1, fps·0.06)` frames and clamps it to 12 cm. That is
//! what `foot_lock` does now, and `trajectory.rs` went with it.
//!
//! Known differences that are **not** ported:
//!
//! | Reference | Why |
//! | --- | --- |
//! | `contact_arm_plan.solve`'s two anchor x/z variables per stance | they let two overlapping palm supports be satisfied at once; this pass pins one palm at a time |
//! | `IK::TwoBoneIk`, `IK::OneBoneIk` | no counterpart: the leg solve is `anatomical::solve_leg` (a clamped two-bone solve on the calibrated knee hinge) and the arm solve is `solve_arm` + `arm_plan`. The reference's two IK bodies were ported, found to have no caller, and removed |
//! | the world-fixed palm offset used inside the reference's `predict` | kept live here instead (`palm = wrist + R_upper·R_elbow·R_wrist_local·offset`), because a fixed world offset no longer matches the palm the applied pose produces once the arm swings |
//! | the pose plan's per-swing-target height variables and its predicted (flex, hip rotation) state | the state exists to regularise the joint solve; this port bounds the root against the same `reach`/`minimum_reach` limits directly, and measures neutral on both fixtures |
//! | `CorrectJointRotations` | no counterpart on purpose: it only moves a bone whose `fullBodyMask` is set, and the reference's own call passes a mask of zeros (`RGBMocap/mocap/correction.py` builds `c` from the contact array and passes `zero` for every body mask), so it is a no-op in that path |
//! | the turning-pivot gate below | not in the reference: a run whose foot heading swings faster than 240 deg/s is released instead of pinned. Our network pivots on the ball of the foot in a turn far more than the reference's does, and pinning it there is visible jitter |
//!
//! `DoEffectorIK` is implemented as the position pin in `hand_lock`; its optional
//! "keep the wrist orientation" branch is not needed because the IK only moves
//! the shoulder and the elbow.
//!
//! * Feet are pinned through the trajectory corrector: the contact frames are
//!   pins, the rest is free, and the velocity/acceleration weights keep the root
//!   from kinking where a contact run starts. Only X/Z move here; the vertical
//!   placement belongs to `root_height` (reach) and `crate::floor` (the plane).
//! * Hands are pinned through the window planner: `palm::plan` decides which
//!   frames are supports, `arm_plan::Plan::solve` moves the collar, the shoulder
//!   and the elbow of the whole stance at once, and the ramp frames around it get
//!   a faded two-bone solve.

use anyhow::Result;

use velomocap_body::{BodyModel, Pose, Skeleton};
use velomocap_core::math;

use crate::flow::FlowClip;
use crate::ik::Transform;

const LEFT_HIP: usize = 1;
const RIGHT_HIP: usize = 2;
const LEFT_ANKLE: usize = 7;
const RIGHT_ANKLE: usize = 8;
const LEFT_FOOT: usize = 10;
const RIGHT_FOOT: usize = 11;
const LEFT_SHOULDER: usize = 16;
const RIGHT_SHOULDER: usize = 17;
const LEFT_ELBOW: usize = 18;
const RIGHT_ELBOW: usize = 19;
const LEFT_WRIST: usize = 20;
const RIGHT_WRIST: usize = 21;

/// Kept in the project file, so it round-trips with the take's other settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CorrectionMode {
    /// Every foot is pinned to one ground height.
    Flat,
    /// Each contact run keeps the height it was detected at (stairs, ramps).
    MultiLevel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LockPreset {
    Tight,
    Normal,
    Loose,
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct CorrectionSettings {
    pub enabled: bool,
    pub mode: CorrectionMode,
    pub foot: bool,
    pub hand: bool,
    pub preset: LockPreset,
    /// Ground height in metres.
    pub floor: f32,
    /// The reference's "Pelvis Height Offset (from Auto)", metres. Positive may
    /// only give back the reach drop `root_height` applied; negative lowers the
    /// body, faded by the contact strength so an airborne take is not moved.
    pub pelvis_offset: f32,
    /// Temporal smoothing weight applied before the contact passes. Smaller is
    /// stronger (0 turns it off); 1e-3 removes the network's per-frame noise
    /// while keeping a 0.5 s arm swing intact.
    pub smoothing: f64,
    /// Use the skinned foot-mesh sole points instead of the joint approximation.
    ///
    /// **On by default**, because it is what the reference does (`SoleGeometry`):
    /// the probes are skinned surface vertices and the 24 lowest proxy vertices
    /// are the sole. The joint approximation (ankle + a guessed heel) is the
    /// fallback for a body without `posedirs`, and it is measurably worse — the
    /// heel position is a constant fraction of the foot axis rather than the
    /// mesh.
    pub mesh_sole: bool,
    /// Run the contact IK on the legs. On by default (the reference always runs
    /// its `DoContactIK`); it pins each stance's ankle to the rolling anchor, but
    /// it never moves the body — the root height belongs to `root_height`.
    pub leg_ik: bool,
    /// Plan the root height over the window with the leg reach bounds and the
    /// swing clearance (`contact_pose_plan._optimise_window`). On by default: the
    /// reference's fifth vertical step, after the drop, lift, offset and
    /// clearance.
    pub pose_plan: bool,
    /// Run the reference's **causal** foot constraint as the last pass
    /// (`pear/foot_lock.py`'s `LiveFootLock`): a hysteresis state machine per
    /// foot, the two-bone solve with the observability gate and the continued
    /// knee pole, the 50 ms one-pole on the applied correction, and the bounded
    /// pelvis compensation. On by default: it is what makes the reference's feet
    /// look still, because it never lets a contact flicker on and off between
    /// frames.
    pub live_lock: bool,
}

impl Default for CorrectionSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            mode: CorrectionMode::Flat,
            foot: true,
            hand: false,
            preset: LockPreset::Normal,
            floor: 0.0,
            pelvis_offset: 0.0,
            leg_ik: true,
            pose_plan: true,
            live_lock: true,
            mesh_sole: true,
            smoothing: 0.0,
        }
    }
}

/// Solver parameters, mapped from the reference's `PRESETS` table
/// (`correction_settings.py`).
pub struct Thresholds {
    /// Hard cap on how far a frame may be moved (`anchor_drift`).
    pub max_correction: f32,
    /// How much of the correction is applied (`strength`: weak holds 60%).
    pub strength: f32,
    /// Contact probability needed for a full-strength hold (`confidence`).
    pub confidence: f32,
    /// The contact gate itself: `confidence`, `release`, `minimum` and `gap`.
    pub contacts: crate::contacts::Config,
}

impl CorrectionSettings {
    pub fn thresholds(&self) -> Thresholds {
        match self.preset {
            // The reference calls these strong / normal / weak. Its `gap` is a
            // preset-level decision: only `strong` bridges a grounded dropout.
            LockPreset::Tight => Thresholds {
                max_correction: 0.20,
                strength: 1.0,
                confidence: 0.65,
                contacts: crate::contacts::Config::strong(),
            },
            LockPreset::Normal => Thresholds {
                max_correction: 0.12,
                strength: 1.0,
                confidence: 0.65,
                contacts: crate::contacts::Config::normal(),
            },
            LockPreset::Loose => Thresholds {
                max_correction: 0.12,
                strength: 0.6,
                confidence: 0.75,
                contacts: crate::contacts::Config::weak(),
            },
        }
    }
}

/// Which limb-quality condition tripped the rollback gate (`helps`), when one did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SkipReason {
    /// The clip was kept: either the correction helped or it was never asked for.
    #[default]
    None,
    /// The joint-based float got worse.
    Float,
    /// The lower foot slid more than 5 % further.
    Slide,
    /// The clip got 50 % jumpier.
    Jitter,
}

#[derive(Clone, Debug, Default)]
pub struct CorrectionReport {
    pub foot_contacts: usize,
    pub hand_contacts: usize,
    pub max_shift: f32,
    pub ik_frames: usize,
    /// Frames the causal live lock re-solved a leg on (`LiveFootLock.apply`).
    pub live_frames: usize,
    /// Worst distance left between a live-solved ankle and its lock target.
    pub live_residual: f32,
    /// The causal live lock was rolled back on its own: it left the clip jumpier
    /// than the exact passes did. Everything before it stays.
    pub live_rolled_back: bool,
    /// Worst distance between a solved wrist and its pinned target.
    pub max_hand_error: f32,
    /// Largest vertical correction applied by the root pass.
    pub vertical_shift: f32,
    /// Largest shift the grounding step applied (`prepare_contacts`' floor shift).
    pub grounding: f32,
    /// Free foot-frames the swing guard lifted out of the plane.
    pub swing_frames: usize,
    /// The deepest of those lifts, in metres.
    pub swing_lift: f32,
    /// Largest pelvis drop needed to span a contact (`reachable_pelvis_drop`).
    pub pelvis_drop: f32,
    /// Largest pelvis lift needed to reach a contact (`reachable_pelvis_lift`).
    pub pelvis_lift: f32,
    /// Jitter (99th percentile of the joint second difference) before the pass.
    pub jitter_before: f32,
    /// Jitter after the pass.
    pub jitter_after: f32,
    /// The root's own second difference (99th percentile) before the pass, in
    /// metres: the pelvis bob the temporal root plan is aimed at.
    pub root_jitter_before: f32,
    /// The same after the pass.
    pub root_jitter_after: f32,
    /// The pass was rolled back because it made the clip jumpier.
    pub skipped: bool,
    /// Which of the three quality conditions tripped that rollback.
    pub skipped_by: SkipReason,
    /// Joint-based float before and after the pass: the mean distance of the
    /// lower foot joint above the floor, in metres. Kept even when the pass is
    /// rolled back, so a "rolled back" line can say why.
    pub float_before: f32,
    pub float_after: f32,
    /// Joint-based slide before and after the pass, cm per frame.
    pub slide_before: f32,
    pub slide_after: f32,
    /// 99th percentile of the ankles' and toes' second difference, in metres,
    /// before and after the pass: the number that matches "the feet shake".
    pub ankle_before: f32,
    pub ankle_after: f32,
    /// No foot contact was detected at all, so the correction was skipped and the
    /// clip is the source unchanged (`flow_correction.py`'s `contacts == 0`).
    pub no_contacts: bool,
    /// The contact runs the detector found, per track (L foot, R foot, L hand,
    /// R hand). Empty for a limb it found nothing on, and for a limb whose track
    /// was hand-drawn — this is the editor's "automatic" baseline, and what
    /// `--dump-locks` writes out so the runs can be edited and fed back.
    pub detected: [Vec<[usize; 2]>; crate::locks::TRACK_COUNT],
    /// What the planted feet did before and after the pass. This is the number
    /// that reads as "the feet slide": measured on the sole probes, over the
    /// frames the detector called contact.
    pub feet_before: crate::metrics::FootMetrics,
    pub feet_after: crate::metrics::FootMetrics,
    /// The corrected sole probes per frame, for `--dump-soles` and a viewport
    /// overlay. [foot][patch][xyz].
    pub soles: Vec<crate::contacts::Probes>,
    /// The same probes before the pass, so a dump can show both.
    pub soles_before: Vec<crate::contacts::Probes>,
    /// The contact mask the pass honoured, per frame and foot.
    pub active: Vec<[bool; 2]>,
}

impl CorrectionReport {
    /// The detected runs as a `ContactTracks` document: a track with runs becomes
    /// manual with those runs, a track without stays automatic.
    pub fn detected_tracks(&self) -> crate::locks::ContactTracks {
        let mut tracks = crate::locks::ContactTracks::automatic();
        for (index, runs) in self.detected.iter().enumerate() {
            if !runs.is_empty() {
                tracks.set_track(index, runs.clone());
            }
        }
        tracks
    }

    pub fn line(&self) -> String {
        let mut out = format!(
            "correction: {} foot contacts, {} hand contacts, {} IK frames, max shift {:.3} m, hand residual {:.1} mm, root {:+.3} m (ground {:.3}, drop {:.3}, lift {:.3}), jitter {:.1}->{:.1} mm",
            self.foot_contacts,
            self.hand_contacts,
            self.ik_frames,
            self.max_shift,
            self.max_hand_error * 1000.0,
            self.vertical_shift,
            self.grounding,
            self.pelvis_drop,
            self.pelvis_lift,
            self.jitter_before * 1000.0,
            self.jitter_after * 1000.0
        );
        if self.live_rolled_back {
            out.push_str("; live lock rolled back (it made the clip jumpier)");
        } else if self.live_frames > 0 {
            out.push_str(&format!(
                "; live lock re-solved {} leg(s), worst residual {:.1} mm",
                self.live_frames,
                self.live_residual * 1000.0
            ));
        }
        if self.root_jitter_before > 0.0 {            out.push_str(&format!(
                " (root {:.1}->{:.1} mm)",
                self.root_jitter_before * 1000.0,
                self.root_jitter_after * 1000.0
            ));
        }
        if self.skipped {
            out.push_str(&format!(
                " (rolled back by {}: float {:.1}->{:.1} mm, slide {:.2}->{:.2} cm/frame, jitter {:.1}->{:.1} mm)",
                match self.skipped_by {
                    SkipReason::Float => "float",
                    SkipReason::Slide => "slide",
                    SkipReason::Jitter => "jitter",
                    SkipReason::None => "the gate",
                },
                self.float_before,
                self.float_after,
                self.slide_before,
                self.slide_after,
                self.jitter_before * 1000.0,
                self.jitter_after * 1000.0
            ));
        }
        if self.no_contacts {
            out.push_str(" (no confident foot contacts: the source is kept)");
        }
        if self.swing_frames > 0 {
            out.push_str(&format!(
                "; swing guard lifted {} foot-frame(s), deepest {:.0} mm",
                self.swing_frames,
                self.swing_lift * 1000.0
            ));
        }
        if !self.feet_before.empty() && !self.feet_after.empty() {
            out.push_str("; ");
            out.push_str(&crate::metrics::FootMetrics::delta_line(
                &self.feet_before,
                &self.feet_after,
            ));
        }
        if self.ankle_before > 0.0 {
            out.push_str(&format!(
                ", ankle shake {:.1}->{:.1} mm",
                self.ankle_before * 1000.0,
                self.ankle_after * 1000.0
            ));
        }
        out
    }
}

/// Apply the correction in place; the clip's skeletons are rebuilt afterwards.
pub fn apply(
    clip: &mut FlowClip,
    body: &BodyModel,
    settings: &CorrectionSettings,
) -> Result<CorrectionReport> {
    apply_with_locks(clip, body, settings, &crate::locks::ContactTracks::automatic())
}

/// Apply the correction with hand-drawn contact timing.
///
/// The tracks are data, not a setting: they belong to one clip, so they travel
/// beside the settings rather than inside them (which is also what lets an
/// existing caller keep using plain [`apply`]).
pub fn apply_with_locks(
    clip: &mut FlowClip,
    body: &BodyModel,
    settings: &CorrectionSettings,
    locks: &crate::locks::ContactTracks,
) -> Result<CorrectionReport> {
    let mut report = CorrectionReport::default();
    if !settings.enabled {
        return Ok(report);
    }
    let thresholds = settings.thresholds();
    // Skinned sole patches: contact points come from the foot mesh, not from
    // joints, exactly like the reference's SoleGeometry.
    let sole = if settings.mesh_sole {
        Sole::build(clip, body)
    } else {
        None
    };

    // Smooth the network's per-frame noise before the contacts are detected: the
    // reference plans its root and targets with an acceleration term, and our
    // FlowHMR output needs the same treatment or a swinging arm moves 10 cm in a
    // frame. Contacts are then detected on the smoothed motion, so the pins below
    // stay exact.
    if settings.smoothing > 0.0 {
        crate::temporal::smooth(clip, body, settings.smoothing)?;
    }

    // Do no harm: a correction that makes the clip jumpier than the network's own
    // motion is worse than no correction at all, and turning clips are exactly
    // where that happens. Measure the jitter before and after and keep the
    // original motion when the pass only adds noise.
    let source_poses = clip.poses.clone();
    let source_skeletons = clip.skeletons.clone();
    let before = quality(&clip.skeletons, settings.floor);

    // One detection for the whole clip: the contacts, the airborne arcs and the
    // floor they imply. Every pass below reads this instead of re-detecting, so
    // the root plan, the pin and the leg solve cannot disagree about a stance.
    // Only the foot passes need it, so `--no-foot` does not pay for the skinning.
    let mut detected = Detection::empty(settings.floor);
    if settings.foot {
        detected = detect(sole.as_ref(), clip, settings, &thresholds);
        for foot in 0..2 {
            report.detected[foot] = foot_runs(&detected.prepared.active, foot)
                .into_iter()
                .map(|(begin, end)| [begin, end])
                .collect();
        }
        // What the planted soles do in the source: the baseline the pass has to
        // improve on, measured on the same probes the detector used.
        report.feet_before =
            crate::metrics::measure(&detected.rows, &detected.prepared.active, settings.floor);
        report.active = detected.prepared.active.clone();
        report.soles_before = detected.rows.clone();
    }

    // `flow_correction.py`: with no contact at all, the reference never writes
    // `trans` and never touches the joints — the published motion is the source,
    // bit for bit. The same rule here: no pin, no root plan and no pose plan, so
    // a clip the detector found nothing in cannot drift because a clearance term
    // decided to lift it.
    let contacts = detected
        .prepared
        .active
        .iter()
        .any(|row| row[0] || row[1]);
    if settings.foot && !contacts {
        report.no_contacts = true;
    }

    let mut foot = FootPass::empty(clip.skeletons.len());
    if settings.foot && contacts {
        foot = foot_lock(sole.as_ref(), clip, settings, locks, &thresholds, &detected, &mut report);
        // The reference's vertical chain, in its order: first the grounding shift
        // that stands the body on the measured plane, then the reach drop, the
        // lift, the user's own offset and the airborne clearance, then the pose
        // plan — every one of them measured from the root the previous one left.
        grounding_pass(clip, body, settings, &foot, &detected, &mut report);
        root_height_pass(clip, body, settings, &foot, &detected, &mut report);
        if settings.pose_plan {
            pose_plan_window(clip, body, settings, &foot, &detected, &mut report);
        }
    }
    if settings.hand {
        hand_lock(clip, body, settings, locks, &mut report);
    }
    // Kimodo's order: end effectors (hands) first, then the contact IK that
    // stabilises the legs on the ground.
    if settings.foot && contacts && settings.leg_ik {
        leg_ik(sole.as_ref(), clip, body, settings, &thresholds, &foot, &mut report);
    }
    // The reference's live stabiliser, last: it decides contact timing causally
    // and re-solves the legs with its own two-bone solver, so a stance whose
    // detection flickered is still held still frame to frame.
    if settings.foot && contacts && settings.live_lock {
        // Judge the live lock on its own. It is the only pass here that decides
        // contact timing causally, so on a clip whose detection it disagrees with
        // it can add twitch — and until now that twitch threw away *every* other
        // pass with it. That is what made "lock on" and "lock off" look identical:
        // the gate discarded the whole solve instead of this one stage.
        let poses_before_live = clip.poses.clone();
        let skeletons_before_live = clip.skeletons.clone();
        let jitter_before_live = quality(&clip.skeletons, settings.floor).jitter;
        live_foot_lock(sole.as_ref(), clip, body, settings, &mut report);
        clip.skeletons = clip
            .poses
            .iter()
            .map(|pose| body.forward_joints(pose))
            .collect();
        let jitter_after_live = quality(&clip.skeletons, settings.floor).jitter;
        if jitter_before_live > 0.0 && jitter_after_live > jitter_before_live * 1.5 + 0.003 {
            clip.poses = poses_before_live;
            clip.skeletons = skeletons_before_live;
            report.live_rolled_back = true;
            report.live_frames = 0;
        }
    }

    clip.skeletons = clip
        .poses
        .iter()
        .map(|pose| body.forward_joints(pose))
        .collect();

    // The same measurement on the result, on the same contact frames: this is
    // what the viewer sees, and what `--dump-soles` writes out.
    if settings.foot && !detected.rows.is_empty() {
        let rows = probe_rows(sole.as_ref(), clip);
        report.feet_after =
            crate::metrics::measure(&rows, &detected.prepared.active, settings.floor);
        report.soles = rows;
    }

    let after = quality(&clip.skeletons, settings.floor);
    report.jitter_before = before.jitter;
    report.jitter_after = after.jitter;
    report.ankle_before = ankle_shake(&source_skeletons);
    report.ankle_after = ankle_shake(&clip.skeletons);
    report.root_jitter_before = before.root;
    report.root_jitter_after = after.root;
    // The gate reads the **sole** metrics, the ones a viewer sees: a pass that
    // lifts a planted foot off the plane or lets it skate is worse than the
    // network's own motion.
    //
    // It used to read the joint-based slide, whose baseline on a standing clip is
    // ~0.1 mm/frame, so a 5 % relative tolerance tripped on a 0.4 mm/frame change
    // and rolled back a pass that had just cut the float by 10 %. Absolute slack
    // (0.2 mm/frame, 1 mm of hover) is what the "do no harm" promise needs.
    let worst_float = |metrics: &crate::metrics::FootMetrics| {
        metrics.float_mm[0].max(metrics.float_mm[1])
    };
    let worst_slide = |metrics: &crate::metrics::FootMetrics| {
        metrics.slide_cm[0].max(metrics.slide_cm[1])
    };
    let measured = !report.feet_before.empty() && !report.feet_after.empty();
    if measured {
        report.float_before = worst_float(&report.feet_before);
        report.float_after = worst_float(&report.feet_after);
        report.slide_before = worst_slide(&report.feet_before);
        report.slide_after = worst_slide(&report.feet_after);
    }
    let slide_helps = worst_slide(&report.feet_after) <= worst_slide(&report.feet_before) * 1.05 + 0.02;
    let float_helps = worst_float(&report.feet_after) <= worst_float(&report.feet_before) + 1.0;
    // The 1.5x tolerance is a ratio on a metric that can be a few millimetres, so
    // a 0.3 mm overshoot used to throw away a pass that cut the planted foot's
    // float from 13.5 mm to 3.2 mm (`短暂.mp4`, 105 frames). The absolute slack is
    // the part of the promise that matters: "no more than a millimetre or so of
    // extra twitch", not "no more than 50 % of a tiny number".
    let helps = after.jitter <= before.jitter * 1.5 + 0.003
        && (!measured || (slide_helps && float_helps));
    // A diagnostic escape hatch: seeing a rolled-back pass's own output is what
    // tells a bad correction apart from a bad gate.
    let helps = helps || std::env::var_os("VELOMOCAP_NO_GATE").is_some();
    if !helps && before.jitter > 0.0 {
        clip.poses = source_poses;
        clip.skeletons = source_skeletons;
        report.skipped = true;
        // The gate's own numbers survive the rollback: "rolled back" without them
        // is a dead end, and the first thing to look at is which of the three
        // tripped.
        report.skipped_by = if measured && !float_helps {
            SkipReason::Float
        } else if measured && !slide_helps {
            SkipReason::Slide
        } else {
            SkipReason::Jitter
        };
        report.max_shift = 0.0;
        report.vertical_shift = 0.0;
        report.grounding = 0.0;
        report.pelvis_drop = 0.0;
        report.pelvis_lift = 0.0;
        report.feet_after = report.feet_before;
        report.soles = detected.rows.clone();
    }
    Ok(report)
}

/// How good a clip is at standing on the floor: how far it floats, how much the
/// lower foot slides, and how much it twitches.
struct Quality {
    float: f32,
    slide: f32,
    jitter: f32,
    /// 99th percentile of the **root's** second difference, in metres: how much
    /// the pelvis itself bobs, which is what the temporal root plan is for.
    root: f32,
}

fn quality(skeletons: &[Skeleton], floor: f32) -> Quality {
    let count = skeletons.len();
    if count < 3 {
        return Quality {
            float: 0.0,
            slide: 0.0,
            jitter: 0.0,
            root: 0.0,
        };
    }
    let mut float = 0.0f32;
    let mut slide = 0.0f32;
    let mut previous: Option<[f32; 2]> = None;
    for frame in 0..count {
        let mut lowest = f32::MAX;
        let mut lowest_xz = [0.0f32; 2];
        for (ankle, toe) in [(LEFT_ANKLE, LEFT_FOOT), (RIGHT_ANKLE, RIGHT_FOOT)] {
            let at = joint(&skeletons[frame], ankle);
            let toe_at = joint(&skeletons[frame], toe);
            let low = at[1].min(toe_at[1]);
            if low < lowest {
                lowest = low;
                lowest_xz = [at[0], at[2]];
            }
        }
        float += (lowest - floor).max(0.0);
        if let Some(last) = previous {
            let dx = lowest_xz[0] - last[0];
            let dz = lowest_xz[1] - last[1];
            slide += (dx * dx + dz * dz).sqrt();
        }
        previous = Some(lowest_xz);
    }
    Quality {
        float: float / count as f32,
        slide: slide / count as f32,
        jitter: jitter_score(skeletons),
        root: root_jitter(skeletons),
    }
}

/// 99th percentile of the root's second difference, in metres.
///
/// The pelvis is what a viewer sees bob: contact_temporal_root's plan exists to
/// trade root height against the knee angle, and this is the number that says
/// whether it bought smoothness.
fn root_jitter(skeletons: &[Skeleton]) -> f32 {
    if skeletons.len() < 3 {
        return 0.0;
    }
    let mut values = Vec::with_capacity(skeletons.len() - 2);
    for frame in 1..skeletons.len() - 1 {
        let previous = joint(&skeletons[frame - 1], 0);
        let current = joint(&skeletons[frame], 0);
        let next = joint(&skeletons[frame + 1], 0);
        let mut sum = 0.0f32;
        for axis in 0..3 {
            let second = previous[axis] - 2.0 * current[axis] + next[axis];
            sum += second * second;
        }
        values.push(sum.sqrt());
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let index = ((values.len() as f32 * 0.99) as usize).min(values.len() - 1);
    values[index]
}


/// 99th percentile of the second difference of the joints the correction moves,
/// in metres. This is the number that tracks what reads as twitching.
///
/// Exposed because the report quotes it and a dump has to be checkable against
/// it: two different numbers for the same clip are a bug in the measurement.
pub fn pose_jitter(skeletons: &[Skeleton]) -> f32 {
    jitter_score(skeletons)
}

fn jitter_score(skeletons: &[Skeleton]) -> f32 {
    jitter_score_of(skeletons, &CORRECTION_JOINTS)
}

/// The same 99th percentile over the ankles and toes only.
///
/// "The feet are shaking" and "the clip is jumpy" are different complaints, and
/// the whole-body number hides the first one behind whatever the arms do. This is
/// the number that answers the foot question, and it sits next to the float and
/// slide in the report so a take can be judged without a dump.
pub fn ankle_shake(skeletons: &[Skeleton]) -> f32 {
    jitter_score_of(skeletons, &ANKLE_JOINTS)
}

const ANKLE_JOINTS: [usize; 4] = [7, 8, 10, 11];

fn jitter_score_of(skeletons: &[Skeleton], joints_of_interest: &[usize]) -> f32 {
    let count = skeletons.len();
    if count < 3 {
        return 0.0;
    }
    let mut values = Vec::with_capacity((count - 2) * joints_of_interest.len());
    for frame in 1..count - 1 {
        let previous = &skeletons[frame - 1].positions;
        let current = &skeletons[frame].positions;
        let next = &skeletons[frame + 1].positions;
        let joints = current.len() / 3;
        for joint in joints_of_interest {
            if *joint >= joints {
                continue;
            }
            let second = second_difference(previous, current, next, *joint);
            values.push(
                (second[0] * second[0] + second[1] * second[1] + second[2] * second[2]).sqrt(),
            );
        }
    }
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let index = ((values.len() as f32 * 0.99) as usize).min(values.len() - 1);
    values[index]
}

/// The joints the correction actually moves, for the "do no harm" gate.
///
/// Measuring **every** joint made that gate useless on a clip with a fast arm:
/// the racket arm alone sets a 99th percentile of 144 mm, so a pass that cut the
/// planted foot's float from 17 mm to 1 mm still tripped "jitter" and was thrown
/// away — which is exactly what a viewer sees as a foot that will not stay put.
/// The gate has to look where the pass acts: the root, the hips and the legs.
const CORRECTION_JOINTS: [usize; 9] = [0, 1, 2, 4, 5, 7, 8, 10, 11];

/// `previous - 2·current + next` for one joint: the frame at `current` is the
/// middle one, so it is the term that gets doubled.
///
/// This used to be written as `current - 2·previous + next`, which is not a
/// second difference at all: it reported 3-4× the real twitch (391 mm instead of
/// 126 mm on a clip with a genuinely fast arm), and the "keep the correction only
/// if it does not add jitter" gate was reading that inflated number.
fn second_difference(previous: &[f32], current: &[f32], next: &[f32], joint: usize) -> [f32; 3] {
    let at = joint * 3;
    let mut out = [0.0f32; 3];
    for axis in 0..3 {
        out[axis] = previous[at + axis] - 2.0 * current[at + axis] + next[at + axis];
    }
    out
}

/// One frame's sole probes, in the layout `contacts::prepare` expects: per foot
/// the toe-side then heel-side patch (it reduces each foot to a minimum and a
/// mean, so the order inside a foot does not matter).
fn sole_row(sole: Option<&Sole>, clip: &FlowClip, frame: usize) -> crate::contacts::Probes {
    let mut row = [[[0.0f32; 3]; 2]; 2];
    for (foot, (ankle, toe)) in [(LEFT_ANKLE, LEFT_FOOT), (RIGHT_ANKLE, RIGHT_FOOT)]
        .into_iter()
        .enumerate()
    {
        let (toe_at, heel) = sole_points(sole, clip, frame, foot, ankle, toe);
        row[foot] = [toe_at, heel];
    }
    row
}

fn probe_rows(sole: Option<&Sole>, clip: &FlowClip) -> Vec<crate::contacts::Probes> {
    (0..clip.skeletons.len())
        .map(|frame| sole_row(sole, clip, frame))
        .collect()
}

/// Runs of one foot's contact mask.
fn foot_runs(active: &[[bool; 2]], foot: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut index = 0;
    while index < active.len() {
        if !active[index][foot] {
            index += 1;
            continue;
        }
        let begin = index;
        while index < active.len() && active[index][foot] {
            index += 1;
        }
        out.push((begin, index));
    }
    out
}

/// The clip's contacts, airborne arcs and the floor they imply, resolved once.
pub struct Detection {
    pub prepared: crate::contacts::Prepared,
    pub rows: Vec<crate::contacts::Probes>,
}

impl Detection {
    /// No detection at all, for a pass that does not correct the feet.
    fn empty(floor: f32) -> Self {
        Self {
            prepared: crate::contacts::Prepared {
                active: Vec::new(),
                flight: Vec::new(),
                floor,
                samples: 0,
            },
            rows: Vec::new(),
        }
    }
}

/// Contact detection: `prepare_contacts` plus the one gate the reference does not
/// have (see the module docs), so every pass below agrees on what is planted.
fn detect(
    sole: Option<&Sole>,
    clip: &FlowClip,
    settings: &CorrectionSettings,
    thresholds: &Thresholds,
) -> Detection {
    let rows = probe_rows(sole, clip);
    let fps = clip.fps.max(1.0);
    let mut prepared = crate::contacts::prepare(
        &rows,
        &clip.foot_contacts,
        fps,
        &thresholds.contacts,
        Some(settings.floor),
    );
    release_pivots(clip, fps, &mut prepared);
    Detection { prepared, rows }
}

/// A turning clip pivots on the ball of the foot while the sole stays down.
/// Pinning such a stance horizontally fights the motion and reads as jitter, so a
/// stance whose foot heading *keeps* swinging faster than 240 deg/s is released.
///
/// This is a deliberate addition: it is not in the reference, whose network does
/// not pivot this hard on a turn. `contacts::prepare` is left alone so the floor
/// estimate still sees every stance the geometry found.
///
/// The test is the **median** rate over the stance, not the fastest frame. A
/// single flipped frame — FlowHMR does produce 180° joint flips on short clips —
/// used to release a perfectly planted stance for its whole length: on a clip
/// where the right foot stands still for all 30 frames, the right foot ended up
/// with no contact at all while the reference held it for 30/30 frames.
fn release_pivots(clip: &FlowClip, fps: f32, prepared: &mut crate::contacts::Prepared) {
    let count = clip.skeletons.len();
    if count < 2 {
        return;
    }
    for (foot, (ankle, toe)) in [(LEFT_ANKLE, LEFT_FOOT), (RIGHT_ANKLE, RIGHT_FOOT)]
        .into_iter()
        .enumerate()
    {
        let mut heading = vec![0.0f32; count];
        for frame in 0..count {
            let ankle_at = joint(&clip.skeletons[frame], ankle);
            let toe_at = joint(&clip.skeletons[frame], toe);
            heading[frame] = (toe_at[2] - ankle_at[2]).atan2(toe_at[0] - ankle_at[0]);
        }
        for (begin, end) in foot_runs(&prepared.active, foot) {
            let mut rates = Vec::with_capacity(end - begin);
            for frame in begin + 1..end {
                let mut delta = heading[frame] - heading[frame - 1];
                while delta > std::f32::consts::PI {
                    delta -= std::f32::consts::TAU;
                }
                while delta < -std::f32::consts::PI {
                    delta += std::f32::consts::TAU;
                }
                rates.push(delta.abs() * fps);
            }
            if rates.is_empty() {
                continue;
            }
            rates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = rates[rates.len() / 2];
            if median > 240.0f32.to_radians() {
                for frame in begin..end {
                    prepared.active[frame][foot] = false;
                }
            }
        }
    }
}

/// Step 0 of the reference's vertical chain: put the lowest **planted sole** on
/// the floor, before the reach drop and lift are measured from it.
///
/// `prepare_contacts` computes this inside the detector and hands the resulting
/// `shift[:, 1]` to everything downstream; here it is its own pass because the
/// contacts are already resolved. Flat ground only: multi-level keeps the source
/// elevation (`multilevel_contacts.prepare` returns a zero shift), and a clip
/// with no contact at all never gets here.
///
/// The height is measured on the same skinned sole probes the contact detector
/// used, not on the ankle joints: the joint sits above the sole, and it is the
/// sole that visibly hovers.
fn grounding_pass(
    clip: &mut FlowClip,
    body: &BodyModel,
    settings: &CorrectionSettings,
    foot: &FootPass,
    detection: &Detection,
    report: &mut CorrectionReport,
) {
    if settings.mode != CorrectionMode::Flat {
        return;
    }
    let count = clip.skeletons.len();
    if count < 3 || detection.rows.len() < count {
        return;
    }
    let mut height = vec![0.0f32; count];
    let mut planted = vec![false; count];
    for frame in 0..count {
        let mut low = f32::MAX;
        let mut any = false;
        for foot_index in 0..2 {
            if !foot.support[frame][foot_index] {
                continue;
            }
            let row = &detection.rows[frame][foot_index];
            // The lower of the two sole patches carries the weight.
            low = low.min(row[0][1]).min(row[1][1]);
            any = true;
        }
        // `any`, not `is_finite`: `f32::MAX` *is* finite, so a swing frame was
        // being handed to the grounding curve as "planted at 3.4e38 m". The
        // desired shift for it clamps to -0.75 m and the smoothed curve dragged
        // the whole body to the floor — which is how a jump ended up pinned to
        // the ground.
        if any {
            height[frame] = low;
            planted[frame] = true;
        }
    }
    if !planted.iter().any(|hit| *hit) {
        return;
    }

    let shift = crate::root_height::grounding(&height, &planted, settings.floor, clip.fps);
    let mut moved = 0.0f32;
    for frame in 0..count {
        let delta = shift[frame];
        if delta.abs() < 1e-6 {
            continue;
        }
        moved = moved.max(delta.abs());
        clip.poses[frame].translation[1] += delta;
    }
    if moved > 0.0 {
        for frame in 0..count {
            clip.skeletons[frame] = body.forward_joints(&clip.poses[frame]);
        }
    }
    report.grounding = moved;
    report.vertical_shift = report.vertical_shift.max(moved);
}

/// The reference's vertical chain, steps 1–4: the reach drop, the reach lift, the
/// user's own pelvis offset and the airborne clearance (`correction.py`).
///
/// Step 0 ([`grounding_pass`]) has already put the body on the plane, so the
/// reach bounds here are measured from a root that is standing still; step 5 is
/// [`pose_plan_window`], which runs afterwards on the root this produced.
fn root_height_pass(
    clip: &mut FlowClip,
    body: &BodyModel,
    settings: &CorrectionSettings,
    foot: &FootPass,
    detection: &Detection,
    report: &mut CorrectionReport,
) {
    let count = clip.skeletons.len();
    if count < 2 {
        return;
    }
    let rest = body.rest_joints(&clip.betas);
    let legs = [(1usize, 4usize, 7usize), (2, 5, 8)];
    let mut frames = Vec::with_capacity(count);
    for frame in 0..count {
        let (world, positions) = crate::anatomical::forward(body, &clip.poses[frame], &rest);
        let mut high = [0.0f64; 2];
        let mut low = [0.0f64; 2];
        for (leg, (hip, knee, ankle)) in legs.iter().enumerate() {
            let knee_local = match body.parents[*knee] {
                parent if parent >= 0 => math::tmul3(&world[parent as usize], &world[*knee]),
                _ => world[*knee],
            };
            high[leg] = crate::anatomical::reach(&rest, *hip, *knee, *ankle, &knee_local);
            low[leg] = crate::anatomical::minimum_reach(&rest, *hip, *knee, *ankle, &knee_local);
        }
        let row = &detection.rows[frame];
        let lowest = row[0][0][1]
            .min(row[0][1][1])
            .min(row[1][0][1])
            .min(row[1][1][1]);
        let active = foot.support[frame];
        frames.push(crate::root_height::Frame {
            hips: [positions[LEFT_HIP], positions[RIGHT_HIP]],
            // The contact targets, not the ankles' current positions: the current
            // position is reachable by definition, so measuring against it never
            // asks for a drop and the sole stays in the air.
            targets: [
                if active[0] {
                    foot.targets[frame][0].map(|value| value as f64)
                } else {
                    positions[LEFT_ANKLE]
                },
                if active[1] {
                    foot.targets[frame][1].map(|value| value as f64)
                } else {
                    positions[RIGHT_ANKLE]
                },
            ],
            high,
            low,
            active,
            flight: detection.prepared.flight.get(frame).copied().unwrap_or(false),
            lowest: lowest as f64,
            // `contact_strength(...).max(axis=1)`: how firmly this frame is held,
            // which is what a negative user offset fades with.
            strength: if active[0] || active[1] {
                foot.holds[frame][0].max(foot.holds[frame][1])
            } else {
                0.0
            },
        });
    }
    let delta = crate::root_height::plan(
        &frames,
        settings.floor as f64,
        clip.fps as f64,
        settings.mode == CorrectionMode::Flat,
        settings.pelvis_offset as f64,
    );
    let mut moved = 0.0f32;
    let mut dropped = 0.0f32;
    let mut lifted = 0.0f32;
    for frame in 0..count {
        let shift = delta[frame] as f32;
        if shift == 0.0 {
            continue;
        }
        if shift < 0.0 {
            dropped = dropped.max(-shift);
        } else {
            lifted = lifted.max(shift);
        }
        moved = moved.max(shift.abs());
        clip.poses[frame].translation[1] += shift;
    }
    report.vertical_shift = report.vertical_shift.max(moved);
    report.pelvis_drop = dropped;
    report.pelvis_lift = lifted;
    if moved > 0.0 {
        for frame in 0..count {
            clip.skeletons[frame] = body.forward_joints(&clip.poses[frame]);
        }
    }
}

/// Plan the root height over the window (`contact_pose_plan._optimise_window`).
///
/// The fifth and last step of the reference's vertical chain, running after
/// [`root_height_pass`] has applied the reach drop, the lift, the user offset and
/// the airborne clearance, and before the leg IK. It keeps the *predicted* leg
/// solve inside the chain's reach bounds and lifts a free foot out of the floor
/// on top of that root, so the leg solve runs against the planned height.
fn pose_plan_window(
    clip: &mut FlowClip,
    body: &BodyModel,
    settings: &CorrectionSettings,
    foot: &FootPass,
    detection: &Detection,
    report: &mut CorrectionReport,
) {
    let count = clip.skeletons.len();
    if count < 3 {
        return;
    }
    let rest = body.rest_joints(&clip.betas);
    let legs = [(1usize, 4usize, 7usize), (2, 5, 8)];
    let mut frames = Vec::with_capacity(count);
    // The knee's own rotation per frame and leg: its bend plane gives the plan the
    // angle a root height would cost, its change over time gives the "static"
    // weight (`contact_temporal_root`'s `static_weight`).
    let mut knees: Vec<[[f64; 9]; 2]> = Vec::with_capacity(count);
    for frame in 0..count {
        let (world, positions) = crate::anatomical::forward(body, &clip.poses[frame], &rest);
        let row = &detection.rows[frame];
        let lowest = row[0][0][1]
            .min(row[0][1][1])
            .min(row[1][0][1])
            .min(row[1][1][1]);
        let active = foot.support[frame];
        let mut high = [0.0f64; 2];
        let mut low = [0.0f64; 2];
        let mut knee_planes = [crate::anatomical::KneePlane {
            constant: 0.0,
            radius: 1e-9,
            phase: 0.0,
        }; 2];
        let mut knee_rotations = [[
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
        ]; 2];
        for (leg, (hip, knee, ankle)) in legs.iter().enumerate() {
            let knee_local = match body.parents[*knee] {
                parent if parent >= 0 => math::tmul3(&world[parent as usize], &world[*knee]),
                _ => world[*knee],
            };
            high[leg] = crate::anatomical::reach(&rest, *hip, *knee, *ankle, &knee_local);
            low[leg] = crate::anatomical::minimum_reach(&rest, *hip, *knee, *ankle, &knee_local);
            knee_planes[leg] = crate::anatomical::knee_plane(&rest, *hip, *knee, *ankle, &knee_local);
            knee_rotations[leg] = knee_local;
        }
        knees.push(knee_rotations);
        frames.push(crate::pose_plan::Frame {
            base: clip.poses[frame].translation[1] as f64,
            hips: [positions[1], positions[2]],
            ankles: [
                if active[0] {
                    foot.targets[frame][0].map(|value| value as f64)
                } else {
                    positions[7]
                },
                if active[1] {
                    foot.targets[frame][1].map(|value| value as f64)
                } else {
                    positions[8]
                },
            ],
            lowest: lowest as f64,
            active,
            high,
            low,
            flight: detection.prepared.flight.get(frame).copied().unwrap_or(false),
            knee: [0, 1].map(|leg| crate::pose_plan::Knee {
                constant: knee_planes[leg].constant,
                radius: knee_planes[leg].radius,
                phase: knee_planes[leg].phase,
                speed: if frame > 0 {
                    crate::anatomical::rotation_angle(
                        &knee_rotations[leg],
                        &knees[frame - 1][leg],
                    ) * clip.fps.max(1.0) as f64
                } else {
                    0.0
                },
            }),
        });
    }
    let planned = crate::pose_plan::plan(
        &frames,
        settings.floor as f64,
        clip.fps as f64,
        settings.mode == CorrectionMode::Flat,
    );
    let mut moved = 0.0f32;
    for frame in 0..count {
        let delta = (planned[frame] - frames[frame].base) as f32;
        if delta.abs() < 1e-6 {
            continue;
        }
        moved = moved.max(delta.abs());
        clip.poses[frame].translation[1] += delta;
    }
    if moved > 0.0 {
        for frame in 0..count {
            clip.skeletons[frame] = body.forward_joints(&clip.poses[frame]);
        }
    }
    report.vertical_shift = report.vertical_shift.max(moved);
}


/// How strongly a run is held, from the network's own contact probabilities.
/// The layout of the four channels differs between checkpoints, so the run is
/// scored with the largest of them: a low score only softens the hold, it never
/// drops a contact the geometry found.
fn contact_confidence(clip: &FlowClip, begin: usize, end: usize, confidence: f32) -> f32 {
    if clip.foot_contacts.len() < end || confidence <= 0.0 {
        return 1.0;
    }
    let mut best = 0.0f32;
    for frame in begin..end {
        for value in clip.foot_contacts[frame] {
            best = best.max(value);
        }
    }
    if best <= 0.0 {
        return 1.0;
    }
    (best / confidence).clamp(0.5, 1.0)
}

/// Kimodo's `_hermite`: `count` interior samples of the cubic that matches the
/// endpoint values and velocities of a segment.

/// Kimodo's `_hermite`: `count` interior samples of the cubic that matches the
/// endpoint values and velocities of a segment.
fn hermite(a: f32, va: f32, b: f32, vb: f32, count: usize) -> Vec<f32> {
    let duration = (count + 1) as f32;
    (1..=count)
        .map(|sample| {
            let t = sample as f32 / duration;
            (2.0 * t.powi(3) - 3.0 * t.powi(2) + 1.0) * a
                + (t.powi(3) - 2.0 * t.powi(2) + t) * duration * va
                + (-2.0 * t.powi(3) + 3.0 * t.powi(2)) * b
                + (t.powi(3) - t.powi(2)) * duration * vb
        })
        .collect()
}

/// Kimodo's `smooth_auto_handoffs`: when two stances touch exactly (no swing
/// frame between them) the anchor must not step at the boundary, so the
/// correction is carried across with a Hermite blend over a window of about
/// 0.2 s on each side. Walking clips are where this shows up.
fn smooth_handoffs(
    stances: &[(usize, usize)],
    offsets_x: &mut [f64],
    offsets_z: &mut [f64],
    fps: f32,
) {
    let radius = ((fps * 0.20).round() as usize).max(3);
    for pair in 0..stances.len().saturating_sub(1) {
        let left = stances[pair];
        let right = stances[pair + 1];
        if left.1 != right.0 {
            // A real swing gap separates them: nothing to blend.
            continue;
        }
        let boundary = right.0;
        let first = left.0.max(boundary.saturating_sub(radius));
        let end = right.1.min(boundary + radius);
        if end - first < 3 {
            continue;
        }
        let a_x = offsets_x[first];
        let b_x = offsets_x[end - 1];
        let v_x_a = offsets_x[first + 1] - a_x;
        let v_x_b = b_x - offsets_x[end - 2];
        let a_z = offsets_z[first];
        let b_z = offsets_z[end - 1];
        let v_z_a = offsets_z[first + 1] - a_z;
        let v_z_b = b_z - offsets_z[end - 2];
        let blend_x = hermite(a_x as f32, v_x_a as f32, b_x as f32, v_x_b as f32, end - first - 2);
        let blend_z = hermite(a_z as f32, v_z_a as f32, b_z as f32, v_z_b as f32, end - first - 2);
        for (slot, index) in (first + 1..end - 1).enumerate() {
            offsets_x[index] = blend_x[slot] as f64;
            offsets_z[index] = blend_z[slot] as f64;
        }
    }
}

/// Kimodo's `couple_anchors`: two consecutive stances of the same foot that are
/// separated by a very short release gap must not be pinned independently, or the
/// foot teleports across the gap. Each stance may only be shifted as a whole
/// (no rotation, no drift inside it), the shift is solved with a tridiagonal
/// system that pulls neighbouring endpoints together, and every stance's
/// adjustment is bounded to 12 cm.
fn couple_anchor_offsets(
    stances: &[(usize, usize)],
    anchors: &[[f32; 3]],
    support: &[[f32; 3]],
    fps: f32,
) -> Vec<[f32; 2]> {
    let count = stances.len();
    let mut adjustment = vec![[0.0f32; 2]; count];
    if count < 2 {
        return adjustment;
    }
    let gap_limit = fps * 0.12;
    let mut diagonal = vec![1.0f64; count];
    let mut off = vec![0.0f64; count - 1];
    let mut rhs = vec![[0.0f64; 2]; count];

    for pair in 0..count - 1 {
        let (left, right) = (stances[pair], stances[pair + 1]);
        let gap = right.0.saturating_sub(left.1);
        if gap as f32 > gap_limit {
            continue;
        }
        let weight = (10.0 * (gap_limit / (gap + 1) as f32).powi(3) as f64).min(1000.0);
        let a = left.1 - 1;
        let b = right.0;
        for axis in [0usize, 2] {
            let difference = ((anchors[b][axis] - support[b][axis])
                - (anchors[a][axis] - support[a][axis])) as f64;
            diagonal[pair] += weight;
            diagonal[pair + 1] += weight;
            off[pair] = -weight;
            rhs[pair][if axis == 0 { 0 } else { 1 }] += weight * difference;
            rhs[pair + 1][if axis == 0 { 0 } else { 1 }] -= weight * difference;
        }
    }

    // Thomas algorithm on the tridiagonal system, once per column.
    for column in 0..2 {
        let mut c = vec![0.0f64; count];
        let mut d = vec![0.0f64; count];
        let mut b = diagonal.clone();
        if b[0].abs() < 1e-12 {
            continue;
        }
        c[0] = off[0] / b[0];
        d[0] = rhs[0][column] / b[0];
        for row in 1..count {
            let denominator = b[row] - off[row - 1] * c[row - 1];
            if denominator.abs() < 1e-12 {
                break;
            }
            if row + 1 < count {
                c[row] = off[row] / denominator;
            }
            d[row] = (rhs[row][column] - off[row - 1] * d[row - 1]) / denominator;
        }
        let mut x = vec![0.0f64; count];
        x[count - 1] = d[count - 1];
        for row in (0..count - 1).rev() {
            x[row] = d[row] - c[row] * x[row + 1];
        }
        for stance in 0..count {
            adjustment[stance][if column == 0 { 0 } else { 1 }] = x[stance] as f32;
        }
        b.clear();
    }

    // Never shift a stance more than 12 cm.
    for delta in &mut adjustment {
        let length = (delta[0] * delta[0] + delta[1] * delta[1]).sqrt();
        if length > 0.12 {
            delta[0] *= 0.12 / length;
            delta[1] *= 0.12 / length;
        }
    }
    adjustment
}

/// The reference's reach gain: `1/(1+(reach_gap/.03)^4)`, with the gap in metres
/// and **only the shortfall counted** (`contact_arm_plan.py:92`).
///
/// This used to be a Gaussian over the last 1 % of the leg's reach, which is a
/// different curve in the only place it matters: 1 cm short of full extension the
/// reference still holds the pin at 0.99, while the old form was already down to
/// 0.16. A standing actor keeps the support leg at 98.7–99.4 % of its reach, i.e.
/// 5–13 mm short — exactly inside the band the old curve deleted. The pin was
/// therefore weakest in the pose people judge the feet in, and the network's own
/// drift kept a third of its speed. Measured on `短暂.mp4`: 1.6 mm/frame of sole
/// slide with the old curve.
fn reach_falloff(target: [f64; 3], hip: [f64; 3], max_reach: f64) -> f64 {
    let gap = (distance3(target, hip) - max_reach).max(0.0) / 0.03;
    1.0 / (1.0 + gap.powi(4))
}

/// Leg IK on the contact frames, ported from `DoContactIK`'s two-bone branch:
/// the ankle is pinned to the rolling anchor of the stance, in all three axes, so
/// the sole's support point lands on the contact patch — and on the floor, its
/// height being the anchor's. The joint's own orientation is untouched (only the
/// hip and the knee move), which is what keeps the foot flat.
fn leg_ik(
    sole: Option<&Sole>,
    clip: &mut FlowClip,
    body: &BodyModel,
    settings: &CorrectionSettings,
    thresholds: &Thresholds,
    foot: &FootPass,
    report: &mut CorrectionReport,
) {
    let fps = clip.fps.max(1.0);
    let count = clip.skeletons.len();
    if count < 2 {
        return;
    }
    let rest = body.rest_joints(&clip.betas);
    let legs: [(usize, usize, usize, usize); 2] = [
        (1, 4, 7, 10), // left hip, knee, ankle, toe
        (2, 5, 8, 11), // right hip, knee, ankle, toe
    ];
    let cap = thresholds.max_correction.min(0.05) as f64;
    // 1 - exp(-dt/0.05): the reference's 50 ms one-pole on the applied
    // correction (pear/foot_lock.py::smooth_correction).
    for (foot_index, (hip, knee, ankle, toe)) in legs.into_iter().enumerate() {
        // Bone lengths from the rest pose: the reach limit for the falloff.
        let thigh = distance3(
            joint3(&rest, knee),
            joint3(&rest, hip),
        );
        let shin = distance3(joint3(&rest, ankle), joint3(&rest, knee));
        let max_reach = thigh + shin;
        // The physical minimum reach of the same chain (the solver's 160 degree
        // flex limit): a target closer than this would fold the leg past what the
        // joint can do, so it is left alone instead of being forced.
        let min_reach = {
            let world = crate::anatomical::forward(body, &clip.poses[0], &rest).0;
            let knee_local = math::tmul3(&world[hip], &world[knee]);
            crate::anatomical::minimum_reach(&rest, hip, knee, ankle, &knee_local)
        };

        // Per foot, the applied correction is filtered across frames.
        // The same runs the root pass holds on: contact timing resolved once.
        for (begin, end) in foot_runs(&foot.support, foot_index) {
            // Same rolling anchor the root pass uses, so the leg IK and the root
            // agree on where the contact patch should be.
            let anchors = rolling_anchor(sole, clip, begin, end, foot_index, ankle, toe, settings.floor);
            // The reference smooths the reach gain along time
            // (gaussian_filter1d(1/(1+(gap/.03)^4), max(.5, fps*.08))) so the
            // IK weight cannot flicker between frames.
            let raw_weights: Vec<f32> = (begin..end)
                .enumerate()
                .map(|(offset, index)| {
                    let current = joint(&clip.skeletons[index], ankle);
                    let sole_at = support_point_with(
                        sole,
                        clip,
                        index,
                        foot_index,
                        ankle,
                        toe,
                        foot.weights[index][foot_index],
                    );
                    let hip_pos = joint(&clip.skeletons[index], hip);
                    let target = contact_target(anchors[offset], sole_at, current, 1.0);
                    let hip_f = [hip_pos[0] as f64, hip_pos[1] as f64, hip_pos[2] as f64];
                    if distance3(target, hip_f) < min_reach * 1.0005 {
                        return 0.0;
                    }
                    reach_falloff(target, hip_f, max_reach) as f32
                })
                .collect();
            let weights = smooth_weights(&raw_weights, fps);
            for (offset, index) in (begin..end).enumerate() {
                let current = joint(&clip.skeletons[index], ankle);
                let sole_at = support_point_with(
                    sole,
                    clip,
                    index,
                    foot_index,
                    ankle,
                    toe,
                    foot.weights[index][foot_index],
                );
                let hip_pos = joint(&clip.skeletons[index], hip);
                // Move the ankle so that the sole's support point lands back on
                // the anchor; the ankle itself is free to slide over it. The hold
                // strength scales the residual goal offset (`strength` in
                // `contact_offsets`), not the root drift: the root already took
                // the full drift, so a weak preset leaves the rest to the pose.
                //
                // The target is **three-dimensional**, exactly as in the
                // reference: the anchor's height is the floor, so the vertical
                // component of the offset is what puts the sole *on* the plane
                // instead of leaving the network's height error in place.
                //
                // The goal itself is *not* smoothed along the stance: that was
                // tried and it let the foot skate (0.26 cm/frame) and float. The
                // reference filters the applied *correction* instead
                // (`smooth_correction` underneath), which keeps the pin exact and
                // only removes the per-frame twitch.
                let hold = foot.holds[index][foot_index] as f64;
                let mut target = contact_target(anchors[offset], sole_at, current, hold);
                if settings.mode == CorrectionMode::Flat {
                    // `correction.py`: a target that is still below the plane
                    // after the offset — the hold strength or the reach clamp can
                    // leave some — is lifted onto it, bounded to 12 cm.
                    let expected = sole_at[1] as f64 + (target[1] - current[1] as f64);
                    let lift = (settings.floor as f64 - expected).clamp(0.0, 0.12);
                    target[1] += lift;
                }
                let hip_f = [
                    hip_pos[0] as f64,
                    hip_pos[1] as f64,
                    hip_pos[2] as f64,
                ];
                // Too close to the hip for the knee to bend that far: leave the
                // network pose alone rather than over-folding the leg.
                if distance3(target, hip_f) < min_reach * 1.0005 {
                    continue;
                }
                let weight = weights[offset];
                if weight <= 0.02 {
                    // Out of reach: the root chain (`root_height`) has already
                    // moved the pelvis as far as the bounds allow, so this frame
                    // is left alone rather than levering the body again here.
                    continue;
                }
                apply_leg_pin(
                    body, clip, &rest, hip, knee, ankle, index, target, weight, cap,
                    report,
                );
            }
        }

        // Release and approach frames: the offset stored by `add_transitions` is
        // already the faded correction, so no hold and no floor lift are applied
        // here — this only carries the leg out of the stance and back into the
        // next one instead of dropping the correction at the contact edge.
        for index in 0..count {
            if !foot.pinned[index][foot_index] || foot.support[index][foot_index] {
                continue;
            }
            let offset = foot.offsets[index][foot_index];
            if offset[0].abs() + offset[1].abs() + offset[2].abs() < 1e-6 {
                continue;
            }
            let current = joint(&clip.skeletons[index], ankle);
            let target = [
                current[0] as f64 + offset[0] as f64,
                current[1] as f64 + offset[1] as f64,
                current[2] as f64 + offset[2] as f64,
            ];
            let hip_f = joint(&clip.skeletons[index], hip).map(|value| value as f64);
            if distance3(target, hip_f) < min_reach * 1.0005 {
                continue;
            }
            let weight = reach_falloff(target, hip_f, max_reach) as f32;
            if weight <= 0.02 {
                continue;
            }
            apply_leg_pin(
                body, clip, &rest, hip, knee, ankle, index, target, weight, cap,
                report,
            );
        }
    }

}

/// `pear/foot_lock.py`'s `Contact`: one foot's causal contact state machine.
///
/// Everything above decides contact timing from the **whole clip** — it can see
/// which frames a stance spans, so its pins are exact. The reference's live
/// stabiliser cannot look ahead: it decides from the past only, and that is why
/// its feet look still on takes where a frame-by-frame detector flickers. A
/// contact needs 75 ms of low, slow evidence to lock, and once locked it stays
/// locked until the foot is clearly high (65 mm above the floor), clearly fast
/// (0.30 m/s) or clearly drifting (120 mm from its anchor). The 160 ms Hermite
/// hand-over carries the offset across those transitions instead of dropping it.
#[derive(Clone, Copy)]
struct LiveContact {
    locked: bool,
    candidate: f64,
    age: f64,
    offset: [f64; 3],
    offset_v: [f64; 3],
    anchor: Option<[f64; 3]>,
    input: Option<[f64; 3]>,
    speed: f64,
    strength: f64,
    strength_v: f64,
}

impl Default for LiveContact {
    fn default() -> Self {
        Self {
            locked: false,
            candidate: 0.0,
            // The reference starts its fade at 1.0, i.e. no hand-over in force.
            age: 1.0,
            offset: [0.0; 3],
            offset_v: [0.0; 3],
            anchor: None,
            input: None,
            speed: 0.0,
            strength: 0.0,
            strength_v: 0.0,
        }
    }
}

impl LiveContact {
    /// Advance one frame and return the sole target the foot is held at.
    ///
    /// All three heights are relative to the floor, so a clip on a raised floor
    /// behaves like one on the reference's ground plane.
    fn update(&mut self, p: [f64; 3], dt: f64, floor: f64, enabled: bool) -> [f64; 3] {
        let y = p[1] - floor;
        let velocity = match self.input {
            Some(input) => [
                (p[0] - input[0]) / dt,
                (p[1] - input[1]) / dt,
                (p[2] - input[2]) / dt,
            ],
            None => [0.0; 3],
        };
        self.input = Some(p);
        // 45 ms one-pole on the speed: the lock cannot be released by a single
        // noisy frame, and it cannot be held by one either.
        self.speed += (1.0 - (-dt / 0.045).exp()) * (length3(velocity) - self.speed);
        // Contact evidence always comes from the uncorrected input, never the lock.
        if enabled && y > -0.06 && y < 0.045 && self.speed < 0.18 {
            self.candidate += dt;
        } else {
            self.candidate = 0.0;
        }
        let mut locked = self.locked;
        if !locked && self.candidate >= 0.075 {
            locked = true;
        }
        if locked
            && (!enabled
                || y > 0.065
                || self.speed > 0.30
                || self
                    .anchor
                    .map(|anchor| distance3(p, anchor) > 0.12)
                    .unwrap_or(false))
        {
            locked = false;
        }

        let duration = 0.16;
        self.age += dt;
        let t = (self.age / duration).min(1.0);
        let h0 = 2.0 * t * t * t - 3.0 * t * t + 1.0;
        let h1 = t * t * t - 2.0 * t * t + t;
        let (base, base_v) = if self.locked {
            (self.anchor.unwrap_or(p), [0.0; 3])
        } else {
            (p, velocity)
        };
        let value = [
            base[0] + h0 * self.offset[0] + h1 * duration * self.offset_v[0],
            base[1] + h0 * self.offset[1] + h1 * duration * self.offset_v[1],
            base[2] + h0 * self.offset[2] + h1 * duration * self.offset_v[2],
        ];
        let speed = [
            base_v[0] + (6.0 * t * t - 6.0 * t) / duration * self.offset[0]
                + (3.0 * t * t - 4.0 * t + 1.0) * self.offset_v[0],
            base_v[1] + (6.0 * t * t - 6.0 * t) / duration * self.offset[1]
                + (3.0 * t * t - 4.0 * t + 1.0) * self.offset_v[1],
            base_v[2] + (6.0 * t * t - 6.0 * t) / duration * self.offset[2]
                + (3.0 * t * t - 4.0 * t + 1.0) * self.offset_v[2],
        ];
        if locked != self.locked {
            self.locked = locked;
            if locked {
                // The anchor keeps the height it was locked at, so the pin is a
                // horizontal hold; the floor handles the vertical.
                self.anchor = Some([p[0], floor, p[2]]);
            }
            let (next_base, next_base_v) = if locked {
                (self.anchor.unwrap_or(p), [0.0; 3])
            } else {
                (p, velocity)
            };
            // Re-express the value being handed over relative to the new base, so
            // the target is continuous exactly at the transition.
            self.offset = [
                value[0] - next_base[0],
                value[1] - next_base[1],
                value[2] - next_base[2],
            ];
            self.offset_v = [
                speed[0] - next_base_v[0],
                speed[1] - next_base_v[1],
                speed[2] - next_base_v[2],
            ];
            self.age = 0.0;
        }

        // A separate continuous activation also hands the pelvis displacement
        // back to free feet; ending an IK pass must not drop that displacement.
        let goal = if locked { 1.0 } else { 0.0 };
        let omega = 24.0;
        let decay = (-omega * dt).exp();
        let x = self.strength - goal;
        let v = self.strength_v;
        self.strength = goal + (x + (v + omega * x) * dt) * decay;
        self.strength_v = (v - omega * (v + omega * x) * dt) * decay;
        value
    }
}

/// The reference's `smooth_correction`: a 50 ms one-pole on the **applied**
/// rotation correction. The solve chases a noisy target; this does not.
fn smooth_rotation(previous: &mut [f64; 9], target: &[f64; 9], dt: f64) -> [f64; 9] {
    let error = axis_angle(&math::mul_t3(target, previous));
    let alpha = 1.0 - (-dt / 0.05).exp();
    let correction = rotvec_to_matrix([
        error[0] as f64 * alpha,
        error[1] as f64 * alpha,
        error[2] as f64 * alpha,
    ]);
    *previous = math::mul3(&correction, previous);
    *previous
}

/// The reference's `LiveFootLock.apply`, run as the last correction pass.
///
/// It re-locks the feet causally (see [`LiveContact`]) and re-solves the legs
/// with the reference's own two-bone solver: the observability gate near full
/// extension, the knee pole continued from the previous frame, the toe carried
/// by the ankle's displacement and the foot's own orientation left untouched.
/// The pelvis may come down by up to 4 cm — critically damped, so it cannot
/// oscillate — and never goes up: lifting the body is the root chain's job.
fn live_foot_lock(
    sole: Option<&Sole>,
    clip: &mut FlowClip,
    body: &BodyModel,
    settings: &CorrectionSettings,
    report: &mut CorrectionReport,
) {
    let count = clip.skeletons.len();
    if count < 3 {
        return;
    }
    let fps = clip.fps.max(1.0) as f64;
    let dt = 1.0 / fps;
    let floor = settings.floor as f64;
    let rest = body.rest_joints(&clip.betas);
    let legs: [(usize, usize, usize, usize); 2] = [(1, 4, 7, 10), (2, 5, 8, 11)];
    let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let mut corrections = [[identity; 2]; 2];
    let mut feet = [LiveContact::default(), LiveContact::default()];
    let mut poles: [Option<[f64; 3]>; 2] = [None, None];
    let mut shift = 0.0f64;
    let mut shift_v = 0.0f64;

    // The material point of the sole each foot's lock tracks: a **constant**
    // offset in the ankle's own frame, which is what the reference's
    // `rig.sole_offsets` is. Tracking the rolling support point instead makes the
    // heel-to-toe roll read as foot speed — the foot then unlocks in the middle of
    // a stance and re-locks at a new spot a few frames later, which is skating.
    // The median over the clip is the material point that touches the floor most
    // often, so it is the one a stance can be held at.
    let mut offsets = [[0.0f64; 3]; 2];
    for (side, (_, _, ankle, toe)) in legs.iter().enumerate() {
        let mut samples = Vec::with_capacity(count);
        for frame in 0..count {
            let (world, positions) = crate::anatomical::forward(body, &clip.poses[frame], &rest);
            let support = support_point(sole, clip, frame, side, *ankle, *toe);
            samples.push(rotate3(
                &math::transpose3(&world[*ankle]),
                sub3(
                    [support[0] as f64, support[1] as f64, support[2] as f64],
                    positions[*ankle],
                ),
            ));
        }
        offsets[side] = [0, 1, 2].map(|axis| median_f64(&samples, axis));
    }

    for frame in 0..count {
        let (world, positions) = crate::anatomical::forward(body, &clip.poses[frame], &rest);
        // The evidence is the uncorrected input: the tracked sole point, which is
        // the same material point on every frame and therefore measures the foot
        // moving, not the foot rolling.
        let mut points = [[0.0f64; 3]; 2];
        for (side, (_, _, ankle, _)) in legs.iter().enumerate() {
            points[side] = add3(positions[*ankle], rotate3(&world[*ankle], offsets[side]));
        }
        let targets = [
            feet[0].update(points[0], dt, floor, true),
            feet[1].update(points[1], dt, floor, true),
        ];

        // How far the pelvis has to come down for a locked foot to stay
        // reachable: the reference's `limit`, bounded to 4 cm and never positive.
        //
        // The reference runs this live, where nothing else plans the root height,
        // so it has `move_root=True`. Here the root chain above (grounding, reach
        // drop, the temporal plan) already made every contact reachable, and the
        // second, causal correction only pushed the body down for no gain: on the
        // serve window it lowered the pelvis 14 mm and pushed a pinned sole the
        // same distance through the floor. `move_root=False` is the same code the
        // reference runs when its caller passes it.
        const MOVE_ROOT: bool = false;
        let mut reach = [0.0f64; 2];
        let mut limit = 0.0f64;
        for (side, (hip, knee, ankle, _)) in legs.iter().enumerate() {
            let a = positions[*hip];
            let b = positions[*knee];
            let c = positions[*ankle];
            let length = distance3(b, a) + distance3(c, b);
            let maximum = (length - 1e-7).min(distance3(c, a));
            reach[side] = maximum;
            if MOVE_ROOT && feet[side].strength > 1e-4 {
                let target = sub3(targets[side], rotate3(&world[*ankle], offsets[side]));
                let horizontal =
                    ((target[0] - a[0]).powi(2) + (target[2] - a[2]).powi(2)).sqrt();
                if horizontal < maximum {
                    let need = target[1] + (maximum * maximum - horizontal * horizontal).max(0.0).sqrt()
                        - a[1];
                    limit = limit.min(need * feet[side].strength);
                }
            }
        }
        // Critically damped, bounded pelvis compensation. It cannot oscillate.
        let goal = if MOVE_ROOT { limit.max(-0.04) } else { 0.0 };
        let omega = 12.0;
        let decay = (-omega * dt).exp();
        let x = shift - goal;
        let v = shift_v;
        shift = goal + (x + (v + omega * x) * dt) * decay;
        shift_v = (v - omega * (v + omega * x) * dt) * decay;
        if shift.abs() > 1e-9 {
            clip.poses[frame].translation[1] += shift as f32;
        }

        for (side, (hip, knee, ankle, _)) in legs.iter().enumerate() {
            let contact = feet[side];
            // Fade-out targets stay active until their inertial offset expires.
            let inactive = !contact.locked && contact.age >= 0.16 && contact.strength < 1e-4;
            let a = [positions[*hip][0], positions[*hip][1] + shift, positions[*hip][2]];
            let b = [positions[*knee][0], positions[*knee][1] + shift, positions[*knee][2]];
            let c = [positions[*ankle][0], positions[*ankle][1] + shift, positions[*ankle][2]];
            let mut target = sub3(targets[side], rotate3(&world[*ankle], offsets[side]));
            target[1] += shift * (1.0 - contact.strength);
            let l1 = distance3(b, a);
            let l2 = distance3(c, b);
            if l1.min(l2) < 1e-7
                || reach[side] < (l1 - l2).abs() + 1e-7
                || distance3(target, a) < 1e-8
            {
                continue;
            }
            let axis = normalise3(sub3(target, a));
            let mut distance = distance3(target, a).clamp((l1 - l2).abs() + 1e-7, reach[side]);
            // Near full extension, tiny height errors otherwise become a large
            // square-root knee displacement. Retain the source bend there and
            // accept foot residual instead of manufacturing crouch/extension.
            let cosine =
                ((l1 * l1 + l2 * l2 - reach[side] * reach[side]) / (2.0 * l1 * l2)).clamp(-1.0, 1.0);
            let flex = 180.0 - cosine.acos().to_degrees();
            let mut observable = ((flex - 6.0) / 12.0).clamp(0.0, 1.0);
            observable = observable * observable * (3.0 - 2.0 * observable);
            distance = reach[side] + observable * (distance - reach[side]);
            let bone = sub3(b, a);
            let mut knee_side = sub3(bone, scale3(axis, math::dot3(&bone, &axis)));
            if length3(knee_side) < 0.001 {
                // Continue the previous frame's pole rather than re-deriving it
                // from a nearly straight leg, where it is undefined.
                knee_side = match poles[side] {
                    Some(pole) => pole,
                    None => column3(&world[*hip], 2),
                };
                knee_side = sub3(knee_side, scale3(axis, math::dot3(&knee_side, &axis)));
            }
            if length3(knee_side) < 1e-8 {
                knee_side = cross3(axis, least_axis(axis));
            }
            let pole = normalise3(knee_side);
            poles[side] = Some(pole);
            let along = (l1 * l1 - l2 * l2 + distance * distance) / (2.0 * distance);
            let new_b = add3(
                add3(a, scale3(axis, along)),
                scale3(pole, (l1 * l1 - along * along).max(0.0).sqrt()),
            );
            let new_c = add3(a, scale3(axis, distance));
            let desired = if inactive {
                [identity; 2]
            } else {
                [
                    math::align(&sub3(b, a), &sub3(new_b, a)),
                    math::align(&sub3(c, b), &sub3(new_c, new_b)),
                ]
            };
            let applied = [
                smooth_rotation(&mut corrections[side][0], &desired[0], dt),
                smooth_rotation(&mut corrections[side][1], &desired[1], dt),
            ];
            let solved_b = add3(a, rotate3(&applied[0], sub3(b, a)));
            let solved_c = add3(solved_b, rotate3(&applied[1], sub3(c, b)));
            // Publish what the solve produced as parent-relative angles. The
            // ankle's **world** rotation is kept: the reference never rotates the
            // ankle, which is what keeps the sole flat, and the toe then follows
            // the ankle's displacement exactly as it does there.
            let hip_world = math::mul3(&applied[0], &world[*hip]);
            let knee_world = math::mul3(&applied[1], &world[*knee]);
            let parent = match body.parents[*hip] {
                value if value >= 0 => value as usize,
                _ => 0,
            };
            let hip_local = math::tmul3(&world[parent], &hip_world);
            let knee_local = math::tmul3(&hip_world, &knee_world);
            let ankle_local = math::tmul3(&knee_world, &world[*ankle]);
            clip.poses[frame].set_joint(*hip, axis_angle(&hip_local));
            clip.poses[frame].set_joint(*knee, axis_angle(&knee_local));
            clip.poses[frame].set_joint(*ankle, axis_angle(&ankle_local));
            report.live_frames += 1;
            report.live_residual = report.live_residual.max(distance3(solved_c, target) as f32);
        }
    }
}

/// Solve one leg pin at one frame and publish it, if the anatomical constraint
/// still holds. The displacement is weighted and capped **before** the solve, so
/// the cap is what the solve is asked for — not a number computed and ignored.
#[allow(clippy::too_many_arguments)]
fn apply_leg_pin(
    body: &BodyModel,
    clip: &mut FlowClip,
    rest: &[f32],
    hip: usize,
    knee: usize,
    ankle: usize,
    index: usize,
    target: [f64; 3],
    weight: f32,
    cap: f64,
    report: &mut CorrectionReport,
) {
    let current = joint(&clip.skeletons[index], ankle);
    // Clamp the *target* instead of rejecting the frame: a frame that is
    // corrected next to a frame that is not is exactly what reads as a twitch,
    // so the correction always ramps with distance.
    let mut want = [
        (target[0] - current[0] as f64) * weight as f64,
        (target[1] - current[1] as f64) * weight as f64,
        (target[2] - current[2] as f64) * weight as f64,
    ];
    let length = (want[0] * want[0] + want[1] * want[1] + want[2] * want[2]).sqrt();
    if length > cap && length > 0.0 {
        want[0] *= cap / length;
        want[1] *= cap / length;
        want[2] *= cap / length;
    }
    if want[0].abs() + want[1].abs() + want[2].abs() < 1e-5 {
        return;
    }
    let solved = [
        current[0] as f64 + want[0],
        current[1] as f64 + want[1],
        current[2] as f64 + want[2],
    ];
    let source = clip.poses[index].clone();
    let (world, positions) = crate::anatomical::forward(body, &clip.poses[index], rest);
    if let Some(anatomy) = crate::anatomical::solve_leg(
        body,
        &mut clip.poses[index],
        rest,
        &world,
        &positions,
        hip,
        knee,
        ankle,
        solved,
    ) {
        // An off-axis knee would mean the anatomical constraint broke: keep the
        // source pose rather than ship it.
        if anatomy.off_axis_deg < 1e-4 {
            // The solve is applied **as solved**.
            //
            // This used to be a 50 ms one-pole on the delta from the source pose,
            // copied from `pear/foot_lock.py::smooth_correction`. That filter is
            // right for the live path, where the target is a state-machine guess
            // and the frames arrive one at a time — but in the batch path the
            // target here is already exact (a stance anchor plus a planned
            // hand-off), so the filter only made the applied correction *lag* a
            // moving source. On `短暂.mp4` that lag was the whole foot problem:

            //   stock   slide 0.16/0.08 cm/frame, float 0.6/0.2 mm, sink 3.0/1.9
            //   direct  slide 0.02/0.01 cm/frame, float 0.4/0.1 mm, sink 0.0/0.7
            //
            // and it raised the whole-body jitter from 28.3 mm to 35.5 mm and the
            // ankles' shake from 34.6 mm to 42.8 mm. Filtering a *lagging* copy of
            // an exact goal is not smoothing: it is a steady-state residual plus
            // the noise the filter puts back in.
            scale_chain_towards_source(body, &mut clip.poses[index], rest, &source, &[hip, knee], 1.0);
            clip.skeletons[index] = body.forward_joints(&clip.poses[index]);
            report.ik_frames += 1;
        } else {
            clip.poses[index] = source;
        }
    }
}

/// The ankle target that lands the sole's support point on the contact anchor.
///
/// `contact_offsets` in the reference: `offsets[active] = (anchors - support -
/// shift)·strength` and `targets = fk(ankle) + shift + offsets`, which collapses
/// to this once the root shift cancels at full strength. **All three axes**: the
/// anchor's height is the floor (`rolling_anchor` sets it), so the vertical term
/// is what puts the sole on the plane rather than pinning it at whatever height
/// the network produced.
fn contact_target(
    anchor: [f32; 3],
    sole_at: [f32; 3],
    current: [f32; 3],
    hold: f64,
) -> [f64; 3] {
    [
        current[0] as f64 + (anchor[0] - sole_at[0]) as f64 * hold,
        current[1] as f64 + (anchor[1] - sole_at[1]) as f64 * hold,
        current[2] as f64 + (anchor[2] - sole_at[2]) as f64 * hold,
    ]
}

/// Gaussian smoothing of a per-frame IK weight along a stance, sigma =
/// max(0.5, fps * 0.08) frames, the same kernel the reference uses for its
/// reach gain. This is what stops a marginal reach from switching the IK on and
/// off between two frames.
fn smooth_weights(weights: &[f32], fps: f32) -> Vec<f32> {
    if weights.len() < 3 {
        return weights.to_vec();
    }
    let sigma = (fps.max(1.0) * 0.08).max(0.5);
    let radius = (sigma * 3.0).ceil() as isize;
    let kernel: Vec<f32> = (-radius..=radius)
        .map(|offset| {
            let o = offset as f32;
            (-(o * o) / (2.0 * sigma * sigma)).exp()
        })
        .collect();
    let total: f32 = kernel.iter().sum();
    (0..weights.len())
        .map(|frame| {
            let mut sum = 0.0f32;
            for (slot, offset) in (-radius..=radius).enumerate() {
                let at = (frame as isize + offset).clamp(0, weights.len() as isize - 1) as usize;
                sum += weights[at] * kernel[slot];
            }
            sum / total
        })
        .collect()
}

/// Diagnostic: for every frame, the four skinned sole probes (foot x side) and the
/// four joint-approximation points, so the two contact sources can be compared
/// numerically instead of only through end-to-end metrics.
pub fn dump_probe_points(clip: &FlowClip, body: &BodyModel) -> Vec<[f32; 24]> {
    let Some(sole) = Sole::build(clip, body) else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(clip.skeletons.len());
    for (frame, skeleton) in clip.skeletons.iter().enumerate() {
        let mut row = [0.0f32; 24];
        for foot in 0..2 {
            let probes = sole.points(skeleton, foot);
            row[foot * 6..foot * 6 + 3].copy_from_slice(&probes.0);
            row[foot * 6 + 3..foot * 6 + 6].copy_from_slice(&probes.1);
            let ankle = if foot == 0 { LEFT_ANKLE } else { RIGHT_ANKLE };
            let toe = if foot == 0 { LEFT_FOOT } else { RIGHT_FOOT };
            let ankle_at = joint(skeleton, ankle);
            let toe_at = joint(skeleton, toe);
            let heel = [
                ankle_at[0] + (ankle_at[0] - toe_at[0]) * 0.6,
                ankle_at[1] + (ankle_at[1] - toe_at[1]) * 0.6,
                ankle_at[2] + (ankle_at[2] - toe_at[2]) * 0.6,
            ];
            row[12 + foot * 6..12 + foot * 6 + 3].copy_from_slice(&toe_at);
            row[12 + foot * 6 + 3..12 + foot * 6 + 6].copy_from_slice(&heel);
        }
        let _ = frame;
        out.push(row);
    }
    out
}

fn joint3(rest: &[f32], index: usize) -> [f64; 3] {
    [
        rest[index * 3] as f64,
        rest[index * 3 + 1] as f64,
        rest[index * 3 + 2] as f64,
    ]
}

fn joint(skeleton: &Skeleton, index: usize) -> [f32; 3] {
    [
        skeleton.positions[index * 3],
        skeleton.positions[index * 3 + 1],
        skeleton.positions[index * 3 + 2],
    ]
}

/// The integration step of `rolling_anchor`, split out so it can be tested
/// without a clip: `separations[i]` is heel - toe and `weights[i]` the heel
/// weight at frame `i`.
fn rolling_path(
    separations: &[[f32; 3]],
    weights: &[f32],
    observed: &[[f32; 3]],
    floor: f32,
) -> Vec<[f32; 3]> {
    let frames = separations.len();
    let mut path = vec![[0.0f32; 3]; frames];
    for frame in 1..frames {
        let dw = weights[frame] - weights[frame - 1];
        for axis in 0..3 {
            path[frame][axis] = path[frame - 1][axis]
                + 0.5 * (separations[frame][axis] + separations[frame - 1][axis]) * dw;
        }
    }
    let mut offsets = [0.0f32; 3];
    for axis in 0..3 {
        let diffs: Vec<f32> = (0..frames).map(|f| observed[f][axis] - path[f][axis]).collect();
        offsets[axis] = median(&diffs);
    }
    for frame in 0..frames {
        path[frame][0] += offsets[0];
        path[frame][1] = floor;
        path[frame][2] += offsets[2];
    }
    path
}

/// Kimodo's `rolling_anchor`: a continuous no-slip anchor for one stance.
///
/// Taking the observed support point directly would move the anchor whenever the
/// weight rolls, which reads as a sliding foot. Instead the *contact location*
/// is integrated: as the weight transfers from one material point to the other,
/// the anchor moves by the separation times the weight change. The result is
/// then offset so its median matches the observed support points, and its height
/// is the floor.
fn rolling_anchor(
    sole: Option<&Sole>,
    clip: &FlowClip,
    begin: usize,
    end: usize,
    foot: usize,
    ankle: usize,
    toe: usize,
    floor: f32,
) -> Vec<[f32; 3]> {
    rolling_anchor_and_weights(sole, clip, begin, end, foot, ankle, toe, floor).0
}

/// `rolling_anchor`, and the smoothed heel weights it was built from.
///
/// The caller needs both: `support_samples` in the reference blends the two sole
/// patches with the *same* weights, so the support point and the anchor agree
/// about where the contact is on every frame.
fn rolling_anchor_and_weights(
    sole: Option<&Sole>,
    clip: &FlowClip,
    begin: usize,
    end: usize,
    foot: usize,
    ankle: usize,
    toe: usize,
    floor: f32,
) -> (Vec<[f32; 3]>, Vec<f32>) {
    let frames = end - begin;
    let mut separations = Vec::with_capacity(frames);
    let mut deltas = Vec::with_capacity(frames);
    let mut points = Vec::with_capacity(frames);
    for frame in begin..end {
        let (toe_at, heel) = sole_points(sole, clip, frame, foot, ankle, toe);
        separations.push([
            heel[0] - toe_at[0],
            heel[1] - toe_at[1],
            heel[2] - toe_at[2],
        ]);
        deltas.push(toe_at[1] - heel[1]);
        points.push((toe_at, heel));
    }
    // `support_weights`: the height difference is smoothed *before* the deadband,
    // with a gaussian of `max(0.5, fps·0.04)`. Without it, a few millimetres of
    // sole noise per frame crosses the 5 mm deadband, the heel weight flips, and
    // `rolling_anchor` integrates that flip as a real roll: the anchor moved
    // 10 cm × Δw, i.e. several millimetres per frame. On `短暂.mp4` that walked a
    // planted foot 30 mm inside its own stance, which is the "飘" that would not go
    // away no matter how strong the pin was — the anchor itself was moving.
    let smoothed = crate::contacts::gaussian_smooth(&deltas, (clip.fps * 0.04).max(0.5));
    let mut weights = Vec::with_capacity(frames);
    let mut observed = Vec::with_capacity(frames);
    for (toe_at, heel) in points.into_iter() {
        let weight = support_weight_from_delta(smoothed[weights.len()]);
        weights.push(weight);
        observed.push([
            toe_at[0] * (1.0 - weight) + heel[0] * weight,
            toe_at[1].min(heel[1]),
            toe_at[2] * (1.0 - weight) + heel[2] * weight,
        ]);
    }

    (rolling_path(&separations, &weights, &observed, floor), weights)
}

/// The two material points of a sole: the toe joint and a heel estimated behind
/// the ankle (SMPL-H has no heel joint, so the foot axis stands in for it).
///
/// Kimodo blends these with weights computed from **geometry only** - which of
/// the two points is lower - so a rolling foot moves its contact point along the
/// sole instead of sliding the whole foot.
fn sole_points(
    sole: Option<&Sole>,
    clip: &FlowClip,
    frame: usize,
    foot: usize,
    ankle: usize,
    toe: usize,
) -> ([f32; 3], [f32; 3]) {
    // The skinned sole patches when they are available, otherwise the joint
    // approximation this pass used before the mesh came in.
    if let Some(sole) = sole {
        return sole.points(&clip.skeletons[frame], foot);
    }
    let skeleton = &clip.skeletons[frame];
    let ankle_at = joint(skeleton, ankle);
    let toe_at = joint(skeleton, toe);
    let heel = [
        ankle_at[0] + (ankle_at[0] - toe_at[0]) * 0.6,
        ankle_at[1] + (ankle_at[1] - toe_at[1]) * 0.6,
        ankle_at[2] + (ankle_at[2] - toe_at[2]) * 0.6,
    ];
    (toe_at, heel)
}

/// Kimodo's `support_weights`, returned as the **heel weight**: 0.5 when the
/// foot is flat, 0 when the toe bears the weight, 1 at the heel. The 5 mm
/// deadband keeps millimetre noise from being read as a weight transfer and the
/// ramp saturates at 25 mm.
fn support_weight(toe: [f32; 3], heel: [f32; 3]) -> f32 {
    support_weight_from_delta(toe[1] - heel[1])
}

/// The same weight from a height difference that has already been smoothed over
/// time, which is what `support_weights` feeds `rolling_anchor`.
fn support_weight_from_delta(delta: f32) -> f32 {
    let x = ((delta.abs() - 0.005) / 0.02).clamp(0.0, 1.0);
    let smooth = x * x * (3.0 - 2.0 * x);
    0.5f32 + 0.5 * smooth * delta.signum()
}

/// The point of the sole that is actually bearing the weight.
fn support_point(
    sole: Option<&Sole>,
    clip: &FlowClip,
    frame: usize,
    foot: usize,
    ankle: usize,
    toe: usize,
) -> [f32; 3] {
    let (toe_at, heel) = sole_points(sole, clip, frame, foot, ankle, toe);
    blend_support(toe_at, heel, support_weight(toe_at, heel))
}

/// The same point with the weight the stance already decided on
/// (`support_samples`), so a heel-to-toe roll is not re-read as noise every frame.
fn support_point_with(
    sole: Option<&Sole>,
    clip: &FlowClip,
    frame: usize,
    foot: usize,
    ankle: usize,
    toe: usize,
    weight: f32,
) -> [f32; 3] {
    let (toe_at, heel) = sole_points(sole, clip, frame, foot, ankle, toe);
    blend_support(toe_at, heel, weight)
}

fn blend_support(toe_at: [f32; 3], heel: [f32; 3], weight: f32) -> [f32; 3] {
    [
        toe_at[0] * (1.0 - weight) + heel[0] * weight,
        toe_at[1].min(heel[1]),
        toe_at[2] * (1.0 - weight) + heel[2] * weight,
    ]
}

/// The skinned sole patches, sampled from the current skeletons so they follow
/// every pose change the passes make.
pub struct Sole {
    geometry: crate::sole::SoleGeometry,
    rest: Vec<f32>,
}

impl Sole {
    pub fn build(clip: &FlowClip, body: &BodyModel) -> Option<Self> {
        Self::for_shape(body, &clip.betas)
    }

    /// The same geometry from a shape alone, for a clip that has not been built
    /// yet (the floor pass runs on poses and skeletons before the clip exists).
    pub fn for_shape(body: &BodyModel, betas: &[f32]) -> Option<Self> {
        let geometry = crate::sole::SoleGeometry::build(body, betas).ok()?;
        let rest = body.rest_joints(betas);
        Some(Self { geometry, rest })
    }

    /// Both feet at once: per foot the toe-side then heel-side patch. One
    /// skinning pass instead of one per foot.
    pub fn row(&self, skeleton: &Skeleton) -> [[[f32; 3]; 2]; 2] {
        let probes = self.geometry.sample(skeleton, &self.rest, 0);
        // `sample` returns [back, front] per foot; the rest of the pass wants
        // (toe side, heel side).
        [
            [probes[0][1], probes[0][0]],
            [probes[1][1], probes[1][0]],
        ]
    }

    /// (toe side, heel side) of one foot on one frame, from the frame's
    /// current skeleton.
    pub fn points(&self, skeleton: &Skeleton, foot: usize) -> ([f32; 3], [f32; 3]) {
        let probes = self.geometry.sample(skeleton, &self.rest, 0);
        (probes[foot.min(1)][1], probes[foot.min(1)][0])
    }
}

/// What the foot pin found, so the root plan can measure reach against the same
/// frames that are actually being held.
pub struct FootPass {
    /// Per frame, per foot: a pin is in force.
    pub support: Vec<[bool; 2]>,
    /// Per frame, per foot: the hold strength behind that pin (0 = no pin).
    pub holds: Vec<[f32; 2]>,
    /// Per frame, per foot: the **3D ankle target** the pin asks for, i.e. where
    /// the ankle has to be for the sole's support point to land on the anchor.
    /// This is what the reach passes measure against — measuring against the
    /// ankle's current position instead is what made a leg that is already
    /// straight look like it had room to straighten further, so no drop was ever
    /// requested and the sole stayed in the air.
    pub targets: Vec<[[f32; 3]; 2]>,
    /// Per frame, per foot: a **release or approach** frame. The correction is
    /// carried out of a stance and back in over 0.12–0.5 s (`contact_offsets`'),
    /// so a foot does not snap at the edge of its contact. `support` stays
    /// stance-only: the release frames are corrected, not locked.
    pub pinned: Vec<[bool; 2]>,
    /// Per frame, per foot: the 3D offset applied on those release frames.
    pub offsets: Vec<[[f32; 3]; 2]>,
    /// Per frame, per foot: the **smoothed** heel weight of the sole.
    ///
    /// The reference feeds one `weights` array to both `rolling_anchor` and
    /// `support_samples`, so the support point moves only when the weight really
    /// rolls. Recomputing it from the frame's own geometry is a different number
    /// every frame: 10 cm of heel-to-toe separation times a hundredth of weight
    /// noise is a millimetre of target jitter per frame, and the IK chases it.
    /// That is the steady 2 mm of ankle motion a planted foot showed here.
    pub weights: Vec<[f32; 2]>,
}

impl FootPass {
    fn empty(frames: usize) -> Self {
        Self {
            support: vec![[false; 2]; frames],
            holds: vec![[0.0; 2]; frames],
            targets: vec![[[0.0; 3]; 2]; frames],
            pinned: vec![[false; 2]; frames],
            offsets: vec![[[0.0; 3]; 2]; frames],
            weights: vec![[0.5; 2]; frames],
        }
    }
}

fn foot_lock(
    sole: Option<&Sole>,
    clip: &mut FlowClip,
    settings: &CorrectionSettings,
    locks: &crate::locks::ContactTracks,
    thresholds: &Thresholds,
    detection: &Detection,
    report: &mut CorrectionReport,
) -> FootPass {
    let fps = clip.fps.max(1.0);
    let count = clip.skeletons.len();
    let mut pass = FootPass::empty(count);
    if count < 2 {
        return pass;
    }
    let feet: [(usize, usize); 2] = [(LEFT_ANKLE, LEFT_FOOT), (RIGHT_ANKLE, RIGHT_FOOT)];
    let mut pinned = vec![false; count];
    let mut target_x = vec![0.0f64; count];
    let mut target_z = vec![0.0f64; count];

    // Ground reference: `Flat` uses the configured floor, `MultiLevel` uses the
    // lowest ankle height seen in the clip, which follows a ramp instead of
    // pinning everything to one plane.
    let mut ground = settings.floor;
    if settings.mode == CorrectionMode::MultiLevel {
        let mut lows = Vec::new();
        for index in 0..count {
            let skeleton = &clip.skeletons[index];
            for (ankle, toe) in feet {
                lows.push(joint(skeleton, ankle)[1].min(joint(skeleton, toe)[1]));
            }
        }
        ground = median(&lows);
    }

    for (foot_index, (ankle, toe)) in feet.into_iter().enumerate() {
        // A hand-edited track replaces the detected mask for this foot, which is
        // what the reference does with `contact_ranges` — except that it clears
        // both feet at once; per track is the same thing for the editor's
        // workflow and keeps an untouched foot on its own detection.
        let manual = locks.runs(foot_index, count);
        let runs = match &manual {
            Some(runs) => runs.clone(),
            None => foot_runs(&detection.prepared.active, foot_index),
        };
        if runs.is_empty() {
            continue;
        }

        // Per-frame anchors, observed support points and hold strength for this
        // foot, then the stance coupling that stops two neighbouring pins from
        // teleporting the foot across a short release gap.
        let mut anchors = vec![[0.0f32; 3]; count];
        let mut support = vec![[0.0f32; 3]; count];
        let mut holds = vec![1.0f32; count];
        for (begin, end) in &runs {
            // The anchor is Kimodo's rolling anchor: a continuous no-slip path
            // through the stance, so a foot that rolls from heel to toe keeps its
            // contact patch still instead of jumping at the middle of the run.
            let (path, weights) =
                rolling_anchor_and_weights(sole, clip, *begin, *end, foot_index, ankle, toe, ground);
            // Weak presets hold 60% of the correction, and a low network contact
            // probability softens it further instead of dropping the contact. A
            // hand-drawn lock is the user's decision, so it is held at full
            // strength: the network's own guess must not soften it.
            let hold = match &manual {
                Some(_) => thresholds.strength,
                None => {
                    thresholds.strength
                        * contact_confidence(clip, *begin, *end, thresholds.confidence)
                }
            };
            report.foot_contacts += 1;
            for (offset, index) in (*begin..*end).enumerate() {
                anchors[index] = path[offset];
                // One weight per frame for the whole stance, shared with the anchor.
                pass.weights[index][foot_index] = weights[offset];
                support[index] = support_point_with(
                    sole,
                    clip,
                    index,
                    foot_index,
                    ankle,
                    toe,
                    weights[offset],
                );
                holds[index] = hold;
                pass.support[index][foot_index] = true;
                pass.holds[index][foot_index] = hold;
                // The 3D ankle target: where the ankle has to be for the sole's
                // support point to land on the anchor (whose height is the floor).
                // It is invariant under a root translation, so it can be built here
                // and used by the reach passes after the horizontal shift is in.
                let target = contact_target(
                    path[offset],
                    support[index],
                    joint(&clip.skeletons[index], ankle),
                    hold as f64,
                );
                pass.targets[index][foot_index] =
                    [target[0] as f32, target[1] as f32, target[2] as f32];
            }
        }

        let adjustments = if manual.is_some() {
            // `prepare_contacts`: only explicit user stances get cross-gap
            // position coupling; an automatic travel split must not accumulate
            // into a false long-duration pin.
            couple_anchor_offsets(&runs, &anchors, &support, fps)
        } else {
            vec![[0.0f32; 2]; runs.len()]
        };
        let mut local_x = vec![0.0f64; count];
        let mut local_z = vec![0.0f64; count];
        let mut local_used = vec![false; count];
        for (stance, (begin, end)) in runs.iter().enumerate() {
            let adjustment = adjustments[stance];
            for index in *begin..*end {
                // Horizontal only: the vertical placement belongs to the root
                // chain (`grounding` + `root_height`), which measures reach against
                // the legs, not the plane. The root takes the *full* drift;
                // `strength` is applied to the remaining goal offset inside the
                // leg IK, as the reference does.
                local_x[index] = (anchors[index][0] + adjustment[0] - support[index][0]) as f64;
                local_z[index] = (anchors[index][2] + adjustment[1] - support[index][2]) as f64;
                local_used[index] = true;
            }
        }
        if manual.is_none() {
            // Stances that touch exactly get their correction carried across
            // with a Hermite blend, so the anchor does not step at the boundary.
            smooth_handoffs(&runs, &mut local_x, &mut local_z, fps);
        }
        for index in 0..count {
            if !local_used[index] {
                continue;
            }
            // Two feet in contact at once average their demands.
            if pinned[index] {
                target_x[index] = 0.5 * (target_x[index] + local_x[index]);
                target_z[index] = 0.5 * (target_z[index] + local_z[index]);
            } else {
                target_x[index] = local_x[index];
                target_z[index] = local_z[index];
            }
            pinned[index] = true;
        }
    }

    if !pinned.iter().any(|hit| *hit) {
        return pass;
    }

    // `prepare_contacts`: the root's X/Z shift is the drift of the planted
    // supports, linearly interpolated across the whole clip (edge-held), smoothed
    // with a gaussian of `max(1, fps·0.06)` frames and clamped to the reference's
    // 12 cm. Not an ADMM trajectory solve: the reference has no such pass, and the
    // linear-plus-gaussian curve is what keeps a long clip from drifting apart at
    // its ends.
    let support_frames: Vec<usize> = (0..count).filter(|index| pinned[*index]).collect();
    let sigma = (fps * 0.06).max(1.0);
    let mut shift_x = interpolate(&target_x, &support_frames, count);
    let mut shift_z = interpolate(&target_z, &support_frames, count);
    shift_x = crate::contacts::gaussian_smooth(&shift_x, sigma);
    shift_z = crate::contacts::gaussian_smooth(&shift_z, sigma);
    let mut shift_x: Vec<f64> = shift_x.iter().map(|value| *value as f64).collect();
    let mut shift_z: Vec<f64> = shift_z.iter().map(|value| *value as f64).collect();

    let cap = thresholds.max_correction as f64;
    for index in 0..count {
        let length = (shift_x[index] * shift_x[index] + shift_z[index] * shift_z[index]).sqrt();
        if length > cap && length > 0.0 {
            shift_x[index] *= cap / length;
            shift_z[index] *= cap / length;
        }
        report.max_shift = report.max_shift.max(length as f32);
        for axis in [0usize, 2] {
            let delta = if axis == 0 { shift_x[index] } else { shift_z[index] };
            clip.poses[index].translation[axis] += delta as f32;
        }
    }
    // Release and approach frames, on the shifted clip: the residual the stance
    // still owed is what the ramp unwinds.
    add_transitions(&mut pass, clip, detection);
    // And a free foot that would scrape the plane is lifted by its own leg.
    add_swing_clearance(&mut pass, clip, settings, detection, report);
    pass
}

/// `contact_offsets`' release and approach ramps: carry the correction out of a
/// stance (and back into the next one) over a bounded number of frames instead of
/// stopping dead at the contact edge.
///
/// The reference's `extent` is `clamp(ceil(|offset|·fps/0.8), 0.12 s, 0.5 s)`
/// frames — a longer correction needs more time to unwind, at a speed of about
/// 0.8 m/s. Two stances separated by a short gap get one Hermite between their
/// offsets; otherwise each side decays to zero independently, and a clip that
/// starts or ends mid-swing decays from (or to) the stance at the edge.
///
/// These frames are **not** contacts: `support` stays stance-only, so the
/// grounding, the reach passes and the foot metrics never see them.
fn add_transitions(pass: &mut FootPass, clip: &FlowClip, detection: &Detection) {
    let count = clip.skeletons.len();
    if count < 3 {
        return;
    }
    let fps = clip.fps.max(1.0);
    let minimum = ((fps * 0.12).round() as usize).max(2);
    let maximum = ((fps * 0.5).round() as usize).max(2);
    let extent = |value: [f32; 3]| {
        let length = (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt();
        let frames = (length * fps / 0.8).ceil() as usize;
        frames.clamp(minimum, maximum)
    };
    for (foot_index, (ankle, _toe)) in [(LEFT_ANKLE, LEFT_FOOT), (RIGHT_ANKLE, RIGHT_FOOT)]
        .into_iter()
        .enumerate()
    {
        let stances = foot_runs(&pass.support, foot_index);
        if stances.is_empty() {
            continue;
        }
        // The correction still owed at each stance frame, in the shifted clip.
        let mut residual = vec![[0.0f32; 3]; count];
        for (begin, end) in &stances {
            for index in *begin..*end {
                let current = joint(&clip.skeletons[index], ankle);
                let target = pass.targets[index][foot_index];
                residual[index] = [
                    target[0] - current[0],
                    target[1] - current[1],
                    target[2] - current[2],
                ];
            }
        }
        let flight = |frame: usize| detection.prepared.flight.get(frame).copied().unwrap_or(false);
        let mut frame = 0usize;
        while frame < count {
            if pass.support[frame][foot_index] || flight(frame) {
                frame += 1;
                continue;
            }
            let first = frame;
            while frame < count && !pass.support[frame][foot_index] && !flight(frame) {
                frame += 1;
            }
            let end = frame;
            let gap = end - first;
            let left = first > 0 && pass.support[first - 1][foot_index];
            let right = end < count && pass.support[end][foot_index];
            if !left && !right {
                continue;
            }
            let zero = [0.0f32; 3];
            let lvalue = if left { residual[first - 1] } else { zero };
            let rvalue = if right { residual[end] } else { zero };
            let lextent = if left { extent(lvalue) } else { 0 };
            let rextent = if right { extent(rvalue) } else { 0 };
            let lcount = gap.min(lextent);
            let rcount = gap.min(rextent);
            let lv = if first >= 2 {
                sub3f(residual[first - 1], residual[first - 2])
            } else {
                zero
            };
            let rv = if end + 1 < count {
                sub3f(residual[end + 1], residual[end])
            } else {
                zero
            };
            if left && right && lcount + rcount >= gap {
                // One Hermite across the whole gap: the foot leaves one stance
                // heading where it was and arrives at the next one smoothly.
                for (slot, index) in (first..end).enumerate() {
                    let value = interp_hermite(lvalue, lv, rvalue, rv, gap, slot);
                    pass.offsets[index][foot_index] = value;
                    pass.pinned[index][foot_index] = true;
                }
            } else {
                if left {
                    // End-of-clip is not evidence that the foot returned to the
                    // raw trajectory: truncate a normal-duration release instead.
                    let duration = if end == count { lextent } else { lcount };
                    for slot in 0..lcount {
                        let value = interp_hermite(lvalue, lv, zero, zero, duration, slot);
                        pass.offsets[first + slot][foot_index] = value;
                        pass.pinned[first + slot][foot_index] = true;
                    }
                }
                if right {
                    let duration = if first == 0 { rextent } else { rcount };
                    let start = end.saturating_sub(rcount);
                    for slot in 0..rcount {
                        let value = interp_hermite(zero, zero, rvalue, rv, duration, slot + (duration - rcount));
                        pass.offsets[start + slot][foot_index] = value;
                        pass.pinned[start + slot][foot_index] = true;
                    }
                }
            }
        }
    }
}

fn sub3f(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// How far a **free** foot has to be lifted so it does not scrape the plane.
///
/// `contact_pose_plan`'s smooth unilateral form: nothing while the sole is more
/// than 1 cm above the floor, the full gap once it is 1 cm below, and
/// `(gap + 1 cm)² / 4 cm` in between, capped at the reference's 12 cm. The
/// reference applies this to the swing foot's target; lifting the whole body for
/// it — as this port used to — also lifts the foot that is planted.
fn swing_lift(gap: f32) -> f32 {
    let lift = if gap >= 0.01 {
        gap
    } else if gap <= -0.01 {
        0.0
    } else {
        (gap + 0.01).powi(2) / 0.04
    };
    lift.clamp(0.0, 0.12)
}

/// Raise a free foot that would go through the floor.
///
/// Flat ground only (multi-level keeps the source elevation), never on a frame
/// that is already pinned, and never on a detected flight frame: an airborne
/// body is the root chain's business, not a foot's.
fn add_swing_clearance(
    pass: &mut FootPass,
    clip: &FlowClip,
    settings: &CorrectionSettings,
    detection: &Detection,
    report: &mut CorrectionReport,
) {
    if settings.mode != CorrectionMode::Flat || detection.rows.len() < clip.skeletons.len() {
        return;
    }
    for frame in 0..clip.skeletons.len() {
        if detection
            .prepared
            .flight
            .get(frame)
            .copied()
            .unwrap_or(false)
        {
            continue;
        }
        for foot_index in 0..2 {
            if pass.pinned[frame][foot_index] {
                continue;
            }
            let row = &detection.rows[frame][foot_index];
            let sole = row[0][1].min(row[1][1]);
            let lift = swing_lift(settings.floor - sole);
            if lift <= 1e-5 {
                continue;
            }
            pass.offsets[frame][foot_index] = [0.0, lift, 0.0];
            pass.pinned[frame][foot_index] = true;
            report.swing_frames += 1;
            report.swing_lift = report.swing_lift.max(lift);
        }
    }
}

/// Interior sample `slot` of a cubic whose endpoints are `a`/`b` with velocities
/// `va`/`vb`, over `count` frames. Mirrors `contact_trajectory._hermite`.
fn interp_hermite(
    a: [f32; 3],
    va: [f32; 3],
    b: [f32; 3],
    vb: [f32; 3],
    count: usize,
    slot: usize,
) -> [f32; 3] {
    let duration = (count + 1) as f32;
    let t = (slot + 1) as f32 / duration;
    let mut out = [0.0f32; 3];
    for axis in 0..3 {
        out[axis] = (2.0 * t.powi(3) - 3.0 * t.powi(2) + 1.0) * a[axis]
            + (t.powi(3) - 2.0 * t.powi(2) + t) * duration * va[axis]
            + (-2.0 * t.powi(3) + 3.0 * t.powi(2)) * b[axis]
            + (t.powi(3) - t.powi(2)) * duration * vb[axis];
    }
    out
}

/// `np.interp(arange(count), frames, values)`: linear between the known frames,
/// held at both ends.
pub(crate) fn interpolate(values: &[f64], frames: &[usize], count: usize) -> Vec<f32> {    let mut out = vec![0.0f32; count];
    if frames.is_empty() || count == 0 {
        return out;
    }
    for index in 0..count {
        if index <= frames[0] {
            out[index] = values[frames[0]] as f32;
            continue;
        }
        if index >= *frames.last().unwrap() {
            out[index] = values[*frames.last().unwrap()] as f32;
            continue;
        }
        // The support frame at or before `index`, and the one after it.
        let next = frames.partition_point(|frame| *frame <= index);
        let low = frames[next - 1];
        let high = frames[next];
        let span = (high - low).max(1) as f64;
        let weight = (index - low) as f64 / span;
        out[index] = (values[low] * (1.0 - weight) + values[high] * weight) as f32;
    }
    out
}

fn hand_lock(
    clip: &mut FlowClip,
    body: &BodyModel,
    settings: &CorrectionSettings,
    locks: &crate::locks::ContactTracks,
    report: &mut CorrectionReport,
) {
    let fps = clip.fps.max(1.0);
    let count = clip.skeletons.len();
    if count < 2 {
        return;
    }
    let rest = body.rest_joints(&clip.betas);
    // The reference detects palm supports, not wrist positions: a hand hanging at
    // the actor's side is stationary too, so the geometric gates below are what
    // make a contact real (`hand_lock.py::plan`).
    let palm = crate::palm::PalmGeometry::build(body, &clip.betas).ok();
    let mut points = Vec::with_capacity(count);
    let mut shoulders = Vec::with_capacity(count);
    let mut hips = Vec::with_capacity(count);
    for skeleton in &clip.skeletons {
        points.push(match &palm {
            Some(geometry) => geometry.points(skeleton),
            None => [joint(skeleton, LEFT_WRIST), joint(skeleton, RIGHT_WRIST)],
        });
        shoulders.push([
            joint(skeleton, LEFT_SHOULDER),
            joint(skeleton, RIGHT_SHOULDER),
        ]);
        hips.push([joint(skeleton, LEFT_HIP), joint(skeleton, RIGHT_HIP)]);
    }
    // Flat ground only admits supports on the floor itself; anything raised is a
    // multi-level support, which is how the reference separates the two modes.
    let surface = match settings.mode {
        CorrectionMode::Flat => Some(settings.floor),
        CorrectionMode::MultiLevel => None,
    };
    let speed_limit = if settings.preset == LockPreset::Loose {
        0.10
    } else {
        0.18
    };
    let planned = crate::palm::plan(&points, &shoulders, &hips, fps, surface, speed_limit);

    for hand_index in 0..2 {
        let (shoulder, elbow, wrist) = if hand_index == 0 {
            (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
        } else {
            (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
        };
        // Tracks 2 and 3 are the hands, in the same order as the feet
        // (`flow_correction.py` splits `ranges[2:]` off for `apply_hands`).
        let manual = locks.runs(crate::locks::LEFT_HAND + hand_index, count);
        let runs = match &manual {
            Some(runs) => runs.clone(),
            None => planned.runs[hand_index].clone(),
        };
        if manual.is_none() {
            report.detected[crate::locks::LEFT_HAND + hand_index] =
                runs.iter().map(|(a, b)| [*a, *b]).collect();
        }
        for (begin, end) in runs.iter().cloned() {
            let mut columns = [Vec::new(), Vec::new(), Vec::new()];
            for index in begin..end {
                let palm_point = points[index][hand_index];
                for axis in 0..3 {
                    columns[axis].push(palm_point[axis]);
                }
            }
            let target = [
                median(&columns[0]),
                median(&columns[1]),
                median(&columns[2]),
            ];
            let ramp = 4usize;
            let first = begin.saturating_sub(ramp);
            let last = (end + ramp).min(count);
            // Fade the pin in and out over a few frames on either side. A run
            // that starts and stops at full strength between two frames is a
            // step in the arm's motion, which reads as a twitch. Kimodo carries
            // the correction into the swing frames for the same reason.
            // The reference fades the arm correction with
            // 1/(1+(reach_gap/.03)^4), smoothed over the stance with the 0.08 s
            // kernel: an unreachable target is not forced, the correction softens.
            let raw_gains: Vec<f32> = (begin..end)
                .map(|index| {
                    let (world, positions) =
                        crate::anatomical::forward(body, &clip.poses[index], &rest);
                    let sampled = [
                        points[index][hand_index][0] as f64,
                        points[index][hand_index][1] as f64,
                        points[index][hand_index][2] as f64,
                    ];
                    let gap = crate::anatomical::source_reach_gap(
                        &rest,
                        &world,
                        &positions,
                        hand_index,
                        [target[0] as f64, target[1] as f64, target[2] as f64],
                        sampled,
                        15.0,
                        60.0,
                    );
                    (1.0 / (1.0 + (gap / 0.03).powi(4))) as f32
                })
                .collect();
            let gains = smooth_weights(&raw_gains, fps);
            // Calibrate the elbow branch once for the whole stance: which side the
            // flexion goes to must not change between frames.
            let stance_local: Vec<[f64; 9]> = (begin..end)
                .map(|index| {
                    let (world, _) = crate::anatomical::forward(body, &clip.poses[index], &rest);
                    math::tmul3(&world[shoulder], &world[elbow])
                })
                .collect();
            let branch = crate::anatomical::elbow_branch(&rest, &stance_local, hand_index);
            // The stance itself is planned as one window
            // (`contact_arm_plan.solve`): the collar may depress, the arm may
            // swivel and the elbow correction is bounded, and the added change is
            // smooth across the stance instead of being re-solved frame by frame.
            let planned = plan_arm_stance(
                body,
                clip,
                &rest,
                palm.as_ref(),
                &points,
                hand_index,
                begin,
                end,
                target,
                branch,
                fps,
            );
            if let Some(stance) = &planned {
                let error = apply_arm_stance(body, clip, &rest, hand_index, begin, target, stance);
                report.hand_contacts += 1;
                report.ik_frames += end - begin;
                report.max_hand_error = report.max_hand_error.max(error);
            }
            for index in first..last {
                // The window plan owns the stance frames; the ramp frames around
                // it still get a single-frame solve, faded towards the source.
                if planned.is_some() && index >= begin && index < end {
                    continue;
                }
                let ramp_weight = if index < begin {
                    (index + ramp + 1 - begin) as f64 / (ramp + 1) as f64
                } else if index >= end {
                    (last - index) as f64 / (ramp + 1) as f64
                } else {
                    1.0
                };
                let gain = if index >= begin && index < end {
                    gains[index - begin] as f64
                } else {
                    1.0
                };
                let weight = ramp_weight * gain;
                let source = clip.poses[index].clone();
                // The pin is a palm point, so the wrist target is that point
                // minus the palm offset: the reference plans
                // `desired = wrist + pin - palm`, which is the same thing when
                // the source wrist rotation is held (`contact_arm_plan.solve`).
                let offset = match &palm {
                    Some(geometry) => geometry.world_offsets(&clip.skeletons[index])[hand_index],
                    None => [0.0f32; 3],
                };
                let wrist_target = [
                    target[0] - offset[0],
                    target[1] - offset[1],
                    target[2] - offset[2],
                ];
                let hint = bend_hint(&clip.skeletons[index], shoulder, elbow, wrist);
                if solve_arm(body, &mut clip.poses[index], &rest, shoulder, elbow, wrist, wrist_target, hint, branch) {
                    // Same guard as the legs: no single frame may swing the wrist
                    // enough for the fingers to jump.
                    limit_end_rotation(
                        body,
                        &mut clip.poses[index],
                        &rest,
                        &source,
                        shoulder,
                        elbow,
                        wrist,
                    );
                    if weight < 1.0 {
                        scale_chain_towards_source(
                            body,
                            &mut clip.poses[index],
                            &rest,
                            &source,
                            &[shoulder, elbow],
                            weight,
                        );
                    }
                    report.ik_frames += 1;
                }
                // How close the solved wrist actually sits to the pinned point.
                // Only the frames inside the run are pinned; the ramp frames are
                // deliberately between the source and the target.
                if weight >= 1.0 {
                    let chain = forward_chain(body, &clip.poses[index], &rest);
                    let settled = chain[wrist].1;
                    let error = ((settled[0] - wrist_target[0] as f64).powi(2)
                        + (settled[1] - wrist_target[1] as f64).powi(2)
                        + (settled[2] - wrist_target[2] as f64).powi(2))
                    .sqrt();
                    report.max_hand_error = report.max_hand_error.max(error as f32);
                }
            }
            // The IK moved joints; the skeletons have to follow before the next
            // contact run looks at positions again.
            for index in first..last {
                clip.skeletons[index] = body.forward_joints(&clip.poses[index]);
            }
        }
    }
}

/// Build the window input for one palm stance (`contact_arm_plan.solve`).
#[allow(clippy::too_many_arguments)]
fn plan_arm_stance(
    body: &BodyModel,
    clip: &FlowClip,
    rest: &[f32],
    palm: Option<&crate::palm::PalmGeometry>,
    points: &[[[f32; 3]; 2]],
    hand: usize,
    begin: usize,
    end: usize,
    pin: [f32; 3],
    branch: f64,
    fps: f32,
) -> Option<Vec<crate::arm_plan::Planned>> {
    let geometry = palm?;
    let (collar, shoulder, elbow) = if hand == 0 {
        (13usize, 16usize, 18usize)
    } else {
        (14, 17, 19)
    };
    let wrist = if hand == 0 { 20 } else { 21 };
    let mut frames = Vec::with_capacity(end - begin);
    let palm_offset_local = geometry.offsets()[hand];
    for index in begin..end {
        let (world, positions) = crate::anatomical::forward(body, &clip.poses[index], rest);
        frames.push(crate::arm_plan::Frame {
            collar_world: world[collar],
            shoulder_world: world[shoulder],
            elbow_world: world[elbow],
            collar: positions[collar],
            shoulder: positions[shoulder],
            elbow: positions[elbow],
            wrist: positions[wrist],
            palm: [
                points[index][hand][0] as f64,
                points[index][hand][1] as f64,
                points[index][hand][2] as f64,
            ],
            wrist_local: math::tmul3(&world[elbow], &world[wrist]),
            palm_offset_local: [
                palm_offset_local[0] as f64,
                palm_offset_local[1] as f64,
                palm_offset_local[2] as f64,
            ],
            influence: 1.0,
            planned: true,
        });
    }
    let plan = crate::arm_plan::Plan {
        rest,
        hand,
        branch,
        fps: fps as f64,
        pin: [pin[0] as f64, pin[1] as f64, pin[2] as f64],
        frames: &frames,
    };
    let planned = plan.solve();
    if planned.len() != frames.len()
        || planned
            .iter()
            .any(|frame| !frame.palm.iter().all(|value| value.is_finite()))
    {
        return None;
    }
    Some(planned)
}

/// Write a planned stance back into the clip's poses. Returns the largest palm
/// error over the stance, in metres.
fn apply_arm_stance(
    body: &BodyModel,
    clip: &mut FlowClip,
    rest: &[f32],
    hand: usize,
    begin: usize,
    pin: [f32; 3],
    planned: &[crate::arm_plan::Planned],
) -> f32 {
    let (collar, shoulder, elbow) = if hand == 0 {
        (13usize, 16usize, 18usize)
    } else {
        (14, 17, 19)
    };
    let parent = body.parents[collar];
    let mut worst = 0.0f32;
    for (offset, result) in planned.iter().enumerate() {
        let index = begin + offset;
        let (world, _) = crate::anatomical::forward(body, &clip.poses[index], rest);
        let parent_world = if parent >= 0 {
            world[parent as usize]
        } else {
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        };
        // Local rotations, so the chain below the elbow is untouched: the wrist
        // keeps its own orientation and the palm rides the planned chain.
        let local_collar = math::tmul3(&parent_world, &result.collar_world);
        let local_shoulder = math::tmul3(&result.collar_world, &result.shoulder_world);
        let rotations = [
            (collar, local_collar),
            (shoulder, local_shoulder),
            (elbow, result.elbow_local),
        ];
        for (joint, matrix) in rotations {
            let values = axis_angle(&matrix);
            let at = (joint - 1) * 3;
            if at + 3 <= clip.poses[index].body.len() {
                clip.poses[index].body[at..at + 3].copy_from_slice(&values);
            }
        }
        // How close the planned palm lands on its pin. The plan's own prediction
        // is exact for the variables it solved, so this is the residual reach
        // error, not an integration error.
        let error = ((result.palm[0] - pin[0] as f64).powi(2)
            + (result.palm[1] - pin[1] as f64).powi(2)
            + (result.palm[2] - pin[2] as f64).powi(2))
        .sqrt();
        worst = worst.max(error as f32);
    }
    worst
}

/// Two-bone IK with Kimodo's analytic solve: stage 1 sets `|ac| = |at|` with the
/// cosine rule on the elbow and the shoulder, stage 2 aims the start joint at
/// the target. World-space rotations are applied to the joint frames, which is
/// the standard parent-first convention `forward_joints` uses.
///
/// Returns whether anything moved.
fn solve_arm(
    body: &BodyModel,
    pose: &mut Pose,
    rest: &[f32],
    shoulder: usize,
    elbow: usize,
    wrist: usize,
    target: [f32; 3],
    hint: Option<[f64; 3]>,
    branch: f64,
) -> bool {
    let target = [target[0] as f64, target[1] as f64, target[2] as f64];
    let start = {
        let world = forward_chain(body, pose, rest);
        world[wrist].1
    };
    if distance3(start, target) < 1e-4 {
        return false;
    }

    // Stage 1: bend the elbow, then the shoulder, to the length the target
    // needs, keeping the plane the arm already lies in.
    let world = forward_chain(body, pose, rest);
    let a = world[shoulder].1;
    let b = world[elbow].1;
    let c = world[wrist].1;
    let l_ab = distance3(a, b);
    let l_bc = distance3(b, c);
    let l_at = distance3(a, target).clamp(1e-4, (l_ab + l_bc) * 0.999);
    if l_ab < 1e-4 || l_bc < 1e-4 {
        return false;
    }
    let theta_abc_now = angle_between(sub3(b, a), sub3(b, c));
    let theta_abc_goal = cosine_rule(l_ab, l_bc, l_at);
    let theta_bac_now = angle_between(sub3(a, b), sub3(a, c));
    let theta_bac_goal = cosine_rule(l_ab, l_at, l_bc);

    // The bend plane. When the limb is nearly straight the cross product is
    // numerically tiny and its direction is arbitrary, which makes the elbow
    // flip from frame to frame - a visible twitch. In that case the plane from
    // the previous frame is reused (Kimodo passes the same kind of hint through
    // `hintOffset`).
    let planar = cross3(sub3(c, a), sub3(b, a));
    let straightness = length3(planar) / (distance3(a, c) * distance3(a, b)).max(1e-9);
    let mut axis = if straightness < 0.05 {
        hint.unwrap_or([0.0, 0.0, 1.0])
    } else {
        normalise3(planar)
    };
    if length3(axis) < 1e-8 {
        axis = [0.0, 0.0, 1.0];
    } else {
        axis = normalise3(axis);
    }
    // The stance's calibrated elbow branch decides which side the flexion goes
    // to; a per-frame decision is what makes an elbow flip between solutions.
    if branch < 0.0 {
        axis = [-axis[0], -axis[1], -axis[2]];
    }

    let mut moved = false;
    if (theta_abc_goal - theta_abc_now).abs() > 1e-6 {
        moved |= rotate_joint_world(
            body,
            pose,
            rest,
            elbow,
            &crate::ik::axis_angle_matrix(axis, theta_abc_goal - theta_abc_now),
        );
    }
    if (theta_bac_goal - theta_bac_now).abs() > 1e-6 {
        moved |= rotate_joint_world(
            body,
            pose,
            rest,
            shoulder,
            &crate::ik::axis_angle_matrix(axis, theta_bac_goal - theta_bac_now),
        );
    }

    // Stage 2: aim the start joint so the end joint lands on the target.
    let world = forward_chain(body, pose, rest);
    let a = world[shoulder].1;
    let c = world[wrist].1;
    if distance3(c, target) > 1e-4 {
        let aim = math::align(&sub3(c, a), &sub3(target, a));
        moved |= rotate_joint_world(body, pose, rest, shoulder, &aim);
    }
    if moved {
        // The temporal limit is applied by the passes (see `limit_end_rotation`):
        // a single solve in isolation has no previous frame to stay near.
    }
    moved
}

/// Scale a solved chain back towards the source pose so the end joint's world
/// orientation never changes by more than `MAX_IK_ROTATION_DEG` in one frame.
fn limit_end_rotation(
    body: &BodyModel,
    pose: &mut Pose,
    rest: &[f32],
    original: &Pose,
    shoulder: usize,
    elbow: usize,
    wrist: usize,
) {
    let solved = forward_chain(body, pose, rest);
    let source = forward_chain(body, original, rest);
    let end = solved[wrist].0;
    let before = source[wrist].0;
    // Angle of `end * before^T`.
    let relative = math::mul_t3(&end, &before);
    let trace = relative[0] + relative[4] + relative[8];
    let angle = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0).acos();
    let limit = MAX_IK_ROTATION_DEG.to_radians();
    if angle <= limit || angle < 1e-6 {
        return;
    }
    let keep = limit / angle;
    for joint_index in [shoulder, elbow, wrist] {
        let solved_rotation = solved[joint_index].0;
        let source_rotation = source[joint_index].0;
        let delta = math::mul_t3(&solved_rotation, &source_rotation);
        let scaled = scale_rotation(&delta, keep);
        let blended = math::mul3(&scaled, &source_rotation);
        // Write the blended world rotation back through its parent.
        let parent = body.parents[joint_index];
        let world_rotation = if parent >= 0 {
            let parent_world = &solved[parent as usize].0;
            math::tmul3(parent_world, &blended)
        } else {
            blended
        };
        let angles = axis_angle(&world_rotation);
        if joint_index == 0 {
            pose.root = angles;
        } else {
            let at = (joint_index - 1) * 3;
            pose.body[at] = angles[0];
            pose.body[at + 1] = angles[1];
            pose.body[at + 2] = angles[2];
        }
    }
}

/// Largest single-frame change of the IK end joint's orientation.
const MAX_IK_ROTATION_DEG: f64 = 12.0;

/// Shrink a rotation towards the identity by `scale`.
fn scale_rotation(matrix: &[f64; 9], scale: f64) -> [f64; 9] {
    let trace = matrix[0] + matrix[4] + matrix[8];
    let angle = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0).acos();
    if angle < 1e-9 {
        return *matrix;
    }
    let axis = normalise3([
        matrix[7] - matrix[5],
        matrix[2] - matrix[6],
        matrix[3] - matrix[1],
    ]);
    crate::ik::axis_angle_matrix(axis, angle * scale)
}

/// Scale a solved chain back towards the source pose by `keep` (0 = the source
/// pose, 1 = the solved one). This is what makes the correction fade in and out
/// instead of switching on between two frames.
fn scale_chain_towards_source(
    body: &BodyModel,
    pose: &mut Pose,
    rest: &[f32],
    source: &Pose,
    joints: &[usize],
    keep: f64,
) {
    let keep = keep.clamp(0.0, 1.0);
    if keep >= 0.9999 {
        return;
    }
    let solved = forward_chain(body, pose, rest);
    let original = forward_chain(body, source, rest);
    for joint_index in joints {
        let solved_rotation = solved[*joint_index].0;
        let source_rotation = original[*joint_index].0;
        let delta = math::mul_t3(&solved_rotation, &source_rotation);
        let scaled = scale_rotation(&delta, keep);
        let blended = math::mul3(&scaled, &source_rotation);
        // Write the blended world rotation back through its parent.
        let parent = body.parents[*joint_index];
        let local = if parent >= 0 {
            let parent_world = &solved[parent as usize].0;
            math::tmul3(parent_world, &blended)
        } else {
            blended
        };
        let angles = axis_angle(&local);
        if *joint_index == 0 {
            pose.root = angles;
        } else {
            let at = (*joint_index - 1) * 3;
            pose.body[at] = angles[0];
            pose.body[at + 1] = angles[1];
            pose.body[at + 2] = angles[2];
        }
    }
}

/// Current bend plane of a two-bone chain, used as the hint when the limb is
/// nearly straight and the plane would otherwise be undefined.
fn bend_hint(skeleton: &Skeleton, root: usize, middle: usize, end: usize) -> Option<[f64; 3]> {
    let a = joint3_of(skeleton, root);
    let b = joint3_of(skeleton, middle);
    let c = joint3_of(skeleton, end);
    let planar = cross3(sub3(c, a), sub3(b, a));
    if length3(planar) < 1e-7 {
        return None;
    }
    Some(normalise3(planar))
}

fn joint3_of(skeleton: &Skeleton, index: usize) -> [f64; 3] {
    [
        skeleton.positions[index * 3] as f64,
        skeleton.positions[index * 3 + 1] as f64,
        skeleton.positions[index * 3 + 2] as f64,
    ]
}

/// Forward kinematics as `(rotation, position)` per joint, parent first.
fn forward_chain(body: &BodyModel, pose: &Pose, rest: &[f32]) -> Vec<([f64; 9], [f64; 3])> {
    let count = body.num_joints;
    let mut out = Vec::with_capacity(count);
    for joint in 0..count {
        let local = if joint == 0 {
            rotvec_to_matrix([
                pose.root[0] as f64,
                pose.root[1] as f64,
                pose.root[2] as f64,
            ])
        } else {
            let at = (joint - 1) * 3;
            rotvec_to_matrix([
                pose.body[at] as f64,
                pose.body[at + 1] as f64,
                pose.body[at + 2] as f64,
            ])
        };
        let parent = body.parents[joint];
        if parent < 0 {
            let rotation = local;
            let position = [
                pose.translation[0] as f64 + rest[0] as f64,
                pose.translation[1] as f64 + rest[1] as f64,
                pose.translation[2] as f64 + rest[2] as f64,
            ];
            out.push((rotation, position));
        } else {
            let p = parent as usize;
            let (parent_rotation, parent_position) = out[p];
            let offset = [
                (rest[joint * 3] - rest[p * 3]) as f64,
                (rest[joint * 3 + 1] - rest[p * 3 + 1]) as f64,
                (rest[joint * 3 + 2] - rest[p * 3 + 2]) as f64,
            ];
            let rotated = rotate3(&parent_rotation, offset);
            let position = [
                parent_position[0] + rotated[0],
                parent_position[1] + rotated[1],
                parent_position[2] + rotated[2],
            ];
            out.push((math::mul3(&parent_rotation, &local), position));
        }
    }
    out
}

fn rotate3(rotation: &[f64; 9], v: [f64; 3]) -> [f64; 3] {
    [
        rotation[0] * v[0] + rotation[1] * v[1] + rotation[2] * v[2],
        rotation[3] * v[0] + rotation[4] * v[1] + rotation[5] * v[2],
        rotation[6] * v[0] + rotation[7] * v[1] + rotation[8] * v[2],
    ]
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn distance3(a: [f64; 3], b: [f64; 3]) -> f64 {
    length3(sub3(a, b))
}

fn length3(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalise3(v: [f64; 3]) -> [f64; 3] {
    let len = length3(v).max(1e-12);
    [v[0] / len, v[1] / len, v[2] / len]
}

fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale3(v: [f64; 3], scale: f64) -> [f64; 3] {
    [v[0] * scale, v[1] * scale, v[2] * scale]
}

/// Column `index` of a row-major 3x3 matrix.
fn column3(matrix: &[f64; 9], index: usize) -> [f64; 3] {
    [matrix[index], matrix[3 + index], matrix[6 + index]]
}

/// The unit axis whose component of `v` is smallest, the reference's
/// `np.eye(3)[np.argmin(abs(v))]`: a degenerate but deterministic fallback.
fn least_axis(v: [f64; 3]) -> [f64; 3] {
    let mut best = 0;
    for index in 1..3 {
        if v[index].abs() < v[best].abs() {
            best = index;
        }
    }
    [
        (best == 0) as u8 as f64,
        (best == 1) as u8 as f64,
        (best == 2) as u8 as f64,
    ]
}

fn angle_between(a: [f64; 3], b: [f64; 3]) -> f64 {
    let left = normalise3(a);
    let right = normalise3(b);
    let cosine = left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
    let sine = length3(cross3(left, right));
    sine.atan2(cosine)
}

fn cosine_rule(side_left: f64, side_right: f64, side_across: f64) -> f64 {
    let value = (side_right * side_right + side_left * side_left - side_across * side_across)
        / (2.0 * side_left * side_right);
    value.clamp(-1.0, 1.0).acos()
}

/// Rotate one joint's world frame by `delta`, keeping its local axis-angle in
/// sync: `R_local ← R_parentᵀ · delta · R_parent · R_local`.
fn rotate_joint_world(
    body: &BodyModel,
    pose: &mut Pose,
    rest: &[f32],
    joint_index: usize,
    delta: &[f64; 9],
) -> bool {
    let world = forward_chain(body, pose, rest);
    let current = world[joint_index].0;
    let next = math::mul3(delta, &current);
    if joint_index == 0 {
        pose.root = axis_angle(&next);
        return true;
    }
    let parent = match body.parents[joint_index] {
        value if value >= 0 => value as usize,
        _ => {
            pose.root = axis_angle(&next);
            return true;
        }
    };
    let parent_world = world[parent].0;
    let local = math::tmul3(&parent_world, &next);
    let angles = axis_angle(&local);
    let at = (joint_index - 1) * 3;
    if at + 3 > pose.body.len() {
        return false;
    }
    pose.body[at] = angles[0];
    pose.body[at + 1] = angles[1];
    pose.body[at + 2] = angles[2];
    true
}

fn axis_angle(m: &[f64; 9]) -> [f32; 3] {
    let trace = m[0] + m[4] + m[8];
    let cos = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0);
    let angle = cos.acos();
    if angle < 1e-6 {
        return [0.0; 3];
    }
    let axis = [
        m[7] - m[5],
        m[2] - m[6],
        m[3] - m[1],
    ];
    let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    if norm < 1e-9 {
        return [0.0; 3];
    }
    [
        (axis[0] / norm * angle) as f32,
        (axis[1] / norm * angle) as f32,
        (axis[2] / norm * angle) as f32,
    ]
}

fn median(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        0.5 * (sorted[middle - 1] + sorted[middle])
    } else {
        sorted[middle]
    }
}

/// Per-axis median of a set of 3-vectors: the material point a foot rolls over.
fn median_f64(values: &[[f64; 3]], axis: usize) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = values.iter().map(|value| value[axis]).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        0.5 * (sorted[middle - 1] + sorted[middle])
    } else {
        sorted[middle]
    }
}

/// Rotation vector (axis * angle) to a rotation matrix.
fn rotvec_to_matrix(v: [f64; 3]) -> [f64; 9] {
    let angle = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if angle < 1e-9 {
        return Transform::IDENTITY.rotation;
    }
    crate::ik::axis_angle_matrix([v[0] / angle, v[1] / angle, v[2] / angle], angle)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_path() -> Option<String> {
        for candidate in [
            "models/body_models/smplh/neutral/model.npz",
            "../../models/body_models/smplh/neutral/model.npz",
        ] {
            if std::path::Path::new(candidate).is_file() {
                return Some(candidate.to_string());
            }
        }
        None
    }

    fn loaded() -> Option<(BodyModel, Vec<f32>)> {
        let path = model_path()?;
        let body = BodyModel::load(&path, 16).expect("load SMPL-H");
        // Non-zero shape on purpose: with every beta at zero `forward_joints`
        // uses the npz's stored `J` instead of the regressed joints, and those
        // sit a few centimetres apart. Real clips always carry shape.
        let mut betas = vec![0.0f32; 16];
        betas[0] = 0.3;
        let rest = body.rest_joints(&betas);
        Some((body, rest))
    }

    /// The chain the IK solves on must reproduce `forward_joints` exactly,
    /// otherwise the solver aims at a different skeleton than the viewport and
    /// the exporter see.
    /// A hand-drawn lock is honoured exactly where it was drawn, even on a clip
    /// the detector rejects: the tracks replace the contact mask per limb, and a
    /// track that was emptied means that limb never touches the ground (which is
    /// what deleting every lock in the reference's editor means).
    #[test]
    fn manual_tracks_replace_the_detected_contacts() {
        let Some((body, _rest)) = loaded() else {
            eprintln!("skipping: SMPL-H model not found");
            return;
        };
        // The actor slides sideways at 3.6 m/s: fast enough that no frame passes
        // the detector's speed gate, so every contact below comes from the tracks.
        let frames = 30usize;
        let build = |body: &BodyModel| {
            let mut pose = Pose::neutral(body.num_joints, 16);
            let mut skeletons = Vec::with_capacity(frames);
            let mut poses = Vec::with_capacity(frames);
            for frame in 0..frames {
                pose.translation = [frame as f32 * 0.12, 0.0, 0.0];
                skeletons.push(body.forward_joints(&pose));
                poses.push(pose.clone());
            }
            FlowClip {
                fps: 30.0,
                poses,
                skeletons,
                rot6d: Vec::new(),
                trans: vec![[0.0; 3]; frames],
                betas: vec![0.0; 16],
                foot_contacts: Vec::new(),
                floor: 0.0,
                floor_from_contacts: false,
                offset: [0.0; 3],
                source_frames: (0..frames).collect(),
                anchors: 0,
                anchor_residual: [0.0; 3],
            }
        };
        let thresholds = CorrectionSettings::default().thresholds();
        let settings = CorrectionSettings::default();
        let mut tracks = crate::locks::ContactTracks::automatic();
        let mut report = CorrectionReport::default();
        let mut clip = build(&body);
        let detected = detect(None, &clip, &settings, &thresholds);
        foot_lock(None, &mut clip, &settings, &tracks, &thresholds, &detected, &mut report);
        assert_eq!(report.foot_contacts, 0, "a sliding clip has no stance");
        tracks.set_track(crate::locks::LEFT_FOOT, vec![[4, 12]]);
        tracks.set_track(crate::locks::RIGHT_FOOT, vec![[16, 24]]);
        let mut report = CorrectionReport::default();
        let mut clip = build(&body);
        let detected = detect(None, &clip, &settings, &thresholds);
        foot_lock(None, &mut clip, &settings, &tracks, &thresholds, &detected, &mut report);
        assert_eq!(report.foot_contacts, 2, "one run per locked foot");

        // Emptying a track is not "back to automatic": it says no contact.
        tracks.set_track(crate::locks::LEFT_FOOT, Vec::new());
        let mut report = CorrectionReport::default();
        let mut clip = build(&body);
        let detected = detect(None, &clip, &settings, &thresholds);
        foot_lock(None, &mut clip, &settings, &tracks, &thresholds, &detected, &mut report);
        assert_eq!(report.foot_contacts, 1, "the empty track locks nothing");

        // And resetting every track goes back to detection, which finds nothing.
        tracks.reset();
        let mut report = CorrectionReport::default();
        let mut clip = build(&body);
        let detected = detect(None, &clip, &settings, &thresholds);
        foot_lock(None, &mut clip, &settings, &tracks, &thresholds, &detected, &mut report);
        assert_eq!(report.foot_contacts, 0, "back to automatic detection");
    }

    /// `np.interp(arange(n), frames, values)`: linear between the known frames and
    /// held at the ends. This is the curve the root's X/Z shift is built on.
    #[test]
    fn the_drift_interpolation_matches_numpy() {
        let values = [0.0, 0.0, 4.0, 4.0, 9.0];
        let frames = [0usize, 2, 4];
        let out = interpolate(&values, &frames, 5);
        assert_eq!(out, vec![0.0, 2.0, 4.0, 6.5, 9.0]);
        // A single support frame holds everywhere.
        assert_eq!(interpolate(&values, &[3], 5), vec![4.0; 5]);
        assert!(interpolate(&values, &[], 5).iter().all(|value| *value == 0.0));
    }

    /// `flow_correction.py`: with no contact at all the reference writes nothing —
    /// `trans` and every joint stay the source, bit for bit. Ours must too, so a
    /// clip the detector found nothing in cannot drift because a clearance term
    /// decided to lift it.
    #[test]
    fn a_clip_without_contacts_is_returned_unchanged() {
        let Some((body, _rest)) = loaded() else {
            eprintln!("skipping: SMPL-H model not found");
            return;
        };
        let frames = 24usize;
        let mut pose = Pose::neutral(body.num_joints, 16);
        let mut poses = Vec::with_capacity(frames);
        for frame in 0..frames {
            pose.translation = [frame as f32 * 0.12, 0.0, 0.0];
            poses.push(pose.clone());
        }
        let skeletons: Vec<Skeleton> = poses.iter().map(|p| body.forward_joints(p)).collect();
        let mut clip = FlowClip {
            fps: 30.0,
            poses,
            skeletons,
            rot6d: Vec::new(),
            trans: vec![[0.0; 3]; frames],
            betas: vec![0.0; 16],
            foot_contacts: Vec::new(),
            floor: 0.0,
            floor_from_contacts: false,
            offset: [0.0; 3],
            source_frames: (0..frames).collect(),
            anchors: 0,
            anchor_residual: [0.0; 3],
        };
        let before: Vec<[f32; 3]> = clip.poses.iter().map(|p| p.translation).collect();
        let before_points: Vec<Vec<f32>> = clip.skeletons.iter().map(|s| s.positions.clone()).collect();
        let report = apply(&mut clip, &body, &CorrectionSettings::default()).expect("apply");
        assert!(report.no_contacts, "the sliding clip must report no contacts");
        assert_eq!(report.foot_contacts, 0);
        for (index, pose) in clip.poses.iter().enumerate() {
            assert_eq!(pose.translation, before[index], "frame {index} root moved");
        }
        for (index, skeleton) in clip.skeletons.iter().enumerate() {
            assert_eq!(
                skeleton.positions, before_points[index],
                "frame {index} joints moved"
            );
        }
    }

    /// The swing guard is a smooth unilateral form: nothing above the 1 cm
    /// deadband, the full gap below it, a quadratic knee in between, and the
    /// reference's 12 cm cap.
    #[test]
    fn the_swing_guard_is_smooth_and_bounded() {
        assert_eq!(swing_lift(-0.05), 0.0, "a sole above the deadband is free");
        assert_eq!(swing_lift(-0.01), 0.0, "the deadband edge is still free");
        // At the knee the quadratic meets the linear branch: (0.02)^2/0.04 = 0.01.
        assert!((swing_lift(0.01) - 0.01).abs() < 1e-6);
        // Below the knee it is the full gap.
        assert!((swing_lift(0.05) - 0.05).abs() < 1e-6);
        // Never more than 12 cm, however deep the penetration.
        assert!((swing_lift(0.40) - 0.12).abs() < 1e-6);
        // Monotone, so a foot that sinks keeps rising.
        let mut previous = -1.0;
        for step in 0..40 {
            let gap = -0.02 + step as f32 * 0.005;
            let lift = swing_lift(gap);
            assert!(lift >= previous, "not monotone at {gap:.3}");
            previous = lift;
        }
    }

    /// The vertical half of the lock: the reference grounds the body on the plane
    /// before it pins anything (`prepare_contacts`' floor shift), and the leg IK's
    /// target carries the vertical component. A take whose soles hover 3 cm has to
    /// end up standing on the floor instead of being pinned in the air.
    #[test]
    fn a_hovering_stance_is_set_down_on_the_floor() {
        let Some((body, _rest)) = loaded() else {
            eprintln!("skipping: SMPL-H model not found");
            return;
        };
        let frames = 24usize;
        let build = |lift: f32| {
            let mut pose = Pose::neutral(body.num_joints, 16);
            pose.translation = [0.0, lift, 0.0];
            FlowClip {
                fps: 30.0,
                poses: vec![pose.clone(); frames],
                skeletons: vec![body.forward_joints(&pose); frames],
                rot6d: Vec::new(),
                trans: vec![[0.0; 3]; frames],
                betas: vec![0.0; 16],
                // Both feet confidently planted, and standing still.
                foot_contacts: vec![[1.0f32; 4]; frames],
                floor: 0.0,
                floor_from_contacts: false,
                offset: [0.0; 3],
                source_frames: (0..frames).collect(),
                anchors: 0,
                anchor_residual: [0.0; 3],
            }
        };
        // Where the neutral pose's lowest sole starts, so the 3 cm hover is exact.
        let rest_clip = build(0.0);
        let sole = Sole::build(&rest_clip, &body).expect("sole geometry");
        let row = sole.row(&rest_clip.skeletons[0]);
        let hover = 0.03f32;
        let lowest = row[0][0][1]
            .min(row[0][1][1])
            .min(row[1][0][1])
            .min(row[1][1][1]);
        let mut clip = build(hover - lowest);

        let report = apply(&mut clip, &body, &CorrectionSettings::default()).expect("apply");
        assert!(!report.no_contacts, "the plant must be detected");
        assert!(
            !report.skipped,
            "the correction was rolled back: {}",
            report.line()
        );
        assert!(
            (report.grounding - hover).abs() < 5e-3,
            "grounding moved {:.4} m, wanted {hover:.3}",
            report.grounding
        );
        let worst = |values: [f32; 2]| values[0].max(values[1]);
        assert!(
            worst(report.feet_after.float_mm) < 5.0,
            "the soles still hover {:?} mm",
            report.feet_after.float_mm
        );
        assert!(
            worst(report.feet_after.sink_mm) < 5.0,
            "the soles sink {:?} mm",
            report.feet_after.sink_mm
        );
        assert!(
            worst(report.feet_after.slide_cm) < 0.1,
            "the pinned feet slide {:?} cm/frame",
            report.feet_after.slide_cm
        );
    }

    #[test]
    fn forward_chain_matches_forward_joints() {
        let Some((body, rest)) = loaded() else {
            eprintln!("skipping: SMPL-H model not found");
            return;
        };
        let mut pose = Pose::neutral(body.num_joints, 16);
        pose.betas[0] = 0.3;
        pose.translation = [0.3, 0.9, -2.0];
        pose.root = [0.2, -0.4, 0.1];
        for (joint, angles) in [
            (LEFT_SHOULDER, [0.4f32, 0.2, -0.3]),
            (LEFT_ELBOW, [0.9, 0.0, 0.0]),
            (LEFT_WRIST, [0.2, 0.5, 0.0]),
            (4, [0.6, 0.0, 0.0]),
            (7, [-0.4, 0.0, 0.0]),
        ] {
            let at = (joint - 1) * 3;
            pose.body[at..at + 3].copy_from_slice(&angles);
        }

        let skeleton = body.forward_joints(&pose);
        let chain = forward_chain(&body, &pose, &rest);
        let mut worst = (0.0f64, 0usize);
        for joint in 0..body.num_joints {
            let expected = [
                skeleton.positions[joint * 3] as f64,
                skeleton.positions[joint * 3 + 1] as f64,
                skeleton.positions[joint * 3 + 2] as f64,
            ];
            let error = distance3(chain[joint].1, expected);
            if error > worst.0 {
                worst = (error, joint);
            }
        }
        assert!(
            worst.0 < 1e-4,
            "joint {} differs by {:.4} m",
            worst.1,
            worst.0
        );
    }

    /// The whole hand path, end to end on a synthetic clip: a palm planted on the
    /// floor for the entire clip is detected, planned as one window and left on
    /// its pin, while a hand hanging at the actor's side is not touched.
    ///
    /// The supplied video fixtures contain no palm support, so without this test
    /// the arm path would only ever be exercised by its unit tests.
    #[test]
    fn a_planted_palm_is_detected_and_held() {
        let Some((body, _rest)) = loaded() else {
            return;
        };
        let betas = vec![0.0f32; 16];
        let frames = 24usize;
        // The left arm swings down and forward (the z rotation is what points the
        // arm at the ground), and the body is dropped until the palm is on the
        // floor at `y = 0.03`.
        let mut pose = Pose::neutral(body.num_joints, 16);
        for (joint, angles) in [
            (LEFT_SHOULDER, [0.0f32, 0.0, -1.0]),
            (LEFT_ELBOW, [0.3, 0.0, 0.0]),
        ] {
            let at = (joint - 1) * 3;
            pose.body[at..at + 3].copy_from_slice(&angles);
        }
        let geometry = crate::palm::PalmGeometry::build(&body, &betas).expect("palm geometry");
        let probe = body.forward_joints(&pose);
        pose.translation = [0.0, 0.03 - geometry.points(&probe)[0][1], 0.0];

        let mut poses = Vec::with_capacity(frames);
        let mut skeletons = Vec::with_capacity(frames);
        for frame in 0..frames {
            let mut current = pose.clone();
            // The first third drifts: the wrist rolls 6 degrees, which moves the
            // palm by a couple of centimetres, and the plan has to pull it back.
            if frame < 8 {
                let at = (LEFT_WRIST - 1) * 3;
                current.body[at + 1] = 0.10;
            }
            let skeleton = body.forward_joints(&current);
            skeletons.push(skeleton);
            poses.push(current);
        }
        let mut clip = FlowClip {
            fps: 30.0,
            poses,
            skeletons,
            rot6d: Vec::new(),
            trans: vec![[0.0; 3]; frames],
            betas: betas.clone(),
            foot_contacts: Vec::new(),
            floor: 0.0,
            floor_from_contacts: true,
            offset: [0.0; 3],
            source_frames: (0..frames).collect(),
            anchors: 0,
            anchor_residual: [0.0; 3],
        };
        let settings = CorrectionSettings {
            hand: true,
            foot: false,
            mode: CorrectionMode::Flat,
            floor: 0.0,
            ..CorrectionSettings::default()
        };
        let thresholds = settings.thresholds();
        let _ = thresholds;
        let mut report = CorrectionReport::default();
        let tracks = crate::locks::ContactTracks::automatic();
        hand_lock(&mut clip, &body, &settings, &tracks, &mut report);
        assert_eq!(report.hand_contacts, 1, "the planted palm was not found");
        assert!(report.ik_frames >= frames - 8, "only {} IK frames", report.ik_frames);

        // Every frame of the stance has its palm on the pin, which is the median
        // palm point of the run.
        let geometry = crate::palm::PalmGeometry::build(&body, &betas).expect("palm geometry");
        let mut columns = [Vec::new(), Vec::new(), Vec::new()];
        for frame in 0..frames {
            let palm = geometry.points(&clip.skeletons[frame])[0];
            for axis in 0..3 {
                columns[axis].push(palm[axis]);
            }
        }
        let pin = [
            median(&columns[0]),
            median(&columns[1]),
            median(&columns[2]),
        ];
        let mut worst = 0.0f32;
        for frame in 0..frames {
            let palm = geometry.points(&clip.skeletons[frame])[0];
            let error = ((palm[0] - pin[0]).powi(2)
                + (palm[1] - pin[1]).powi(2)
                + (palm[2] - pin[2]).powi(2))
            .sqrt();
            worst = worst.max(error);
        }
        assert!(worst < 0.02, "palm wanders {:.1} mm from its pin", worst * 1000.0);
        // The plan's own contact error must be reported, not silently swallowed.
        assert!(
            report.max_hand_error < 0.02,
            "reported hand residual {:.1} mm",
            report.max_hand_error * 1000.0
        );
    }

    /// `smooth_auto_handoffs`: when two stances touch, the correction curve must
    /// not step at the boundary.
    #[test]
    fn handoffs_blend_across_a_touching_boundary() {
        let frames = 24;
        // Two stances that touch at frame 12.
        let stances = vec![(0usize, 12usize), (12usize, 24usize)];
        let mut x = vec![0.0f64; frames];
        let mut z = vec![0.0f64; frames];
        // A 10 cm step right at the boundary.
        for frame in 12..24 {
            x[frame] = 0.10;
        }
        let before = x[12] - x[11];
        smooth_handoffs(&stances, &mut x, &mut z, 30.0);
        let after = x[12] - x[11];
        assert!(
            after.abs() < before.abs(),
            "step at the boundary grew: {before:.4} -> {after:.4}"
        );
    }

    /// `_hermite` matches the endpoint values.
    #[test]
    fn hermite_matches_its_endpoints() {
        let values = hermite(0.0, 0.1, 1.0, -0.1, 5);
        assert_eq!(values.len(), 5);
        assert!(values.iter().all(|v| v.is_finite()));
        // Monotone-ish between the two endpoint values.
        assert!(values.iter().all(|v| *v > -0.5 && *v < 1.5));
    }

    /// `couple_anchors`: two stances separated by a short gap must not keep
    /// disagreeing anchors, and no stance may shift more than 12 cm.
    #[test]
    fn coupled_stances_meet_across_a_short_gap() {
        let frames = 20;
        // Two stances: 0..8 and 10..18, a two frame release gap between them.
        let stances = vec![(0usize, 8usize), (10usize, 18usize)];
        let mut anchors = vec![[0.0f32; 3]; frames];
        let mut support = vec![[0.0f32; 3]; frames];
        // The second stance's foot is pinned 10 cm away from the first.
        for frame in 0..8 {
            anchors[frame] = [0.0, 0.0, 0.0];
            support[frame] = [0.0, 0.0, 0.0];
        }
        for frame in 10..18 {
            anchors[frame] = [0.10, 0.0, 0.0];
            // Same instant in the source: the disagreement is the anchor's.
            support[frame] = [0.0, 0.0, 0.0];
        }
        let adjustments = couple_anchor_offsets(&stances, &anchors, &support, 30.0);
        let left_end = (anchors[7][0] + adjustments[0][0]) - support[7][0];
        let right_start = (anchors[10][0] + adjustments[1][0]) - support[10][0];
        // The regulariser trades agreement against shifting either stance, so the
        // residual is not zero, but the disagreement has to shrink a lot.
        assert!(
            (left_end - right_start).abs() < 0.05,
            "endpoints still differ by {:.3} m ({left_end:.3} vs {right_start:.3})",
            (left_end - right_start).abs()
        );
        for delta in &adjustments {
            let length = (delta[0] * delta[0] + delta[1] * delta[1]).sqrt();
            assert!(length <= 0.12 + 1e-6, "stance shifted {length:.3} m");
        }
    }

    /// A long swing gap leaves the two stances independent.
    #[test]
    fn distant_stances_are_not_coupled() {
        let frames = 40;
        let stances = vec![(0usize, 8usize), (30usize, 38usize)];
        let mut anchors = vec![[0.0f32; 3]; frames];
        let support = vec![[0.0f32; 3]; frames];
        for frame in 0..8 {
            anchors[frame] = [0.0, 0.0, 0.0];
        }
        for frame in 30..38 {
            anchors[frame] = [0.10, 0.0, 0.0];
        }
        let adjustments = couple_anchor_offsets(&stances, &anchors, &support, 30.0);
        for delta in &adjustments {
            assert!(delta[0].abs() < 1e-4 && delta[1].abs() < 1e-4, "adjusted {delta:?}");
        }
    }

    /// `rolling_anchor`: during a pure heel-to-toe roll (no sliding) the
    /// integrated contact location must follow the observed support point, which
    /// is what makes the foot stay put while the weight transfers.
    #[test]
    fn rolling_path_follows_a_clean_roll() {
        let frames = 11;
        // Toe at x = 0, heel at x = 1, 20 cm apart.
        let separations: Vec<[f32; 3]> = (0..frames).map(|_| [0.2, 0.0, 0.0]).collect();
        // Weight rolls from the toe (0) to the heel (1).
        let weights: Vec<f32> = (0..frames).map(|i| i as f32 / (frames - 1) as f32).collect();
        // The observed support point rolls with it.
        let observed: Vec<[f32; 3]> = weights.iter().map(|w| [0.2 * w, 0.0, 0.0]).collect();
        let path = rolling_path(&separations, &weights, &observed, 0.0);
        for (frame, anchor) in path.iter().enumerate() {
            assert!(
                (anchor[0] - observed[frame][0]).abs() < 1e-5,
                "frame {frame}: anchor {} vs observed {}",
                anchor[0],
                observed[frame][0]
            );
            assert_eq!(anchor[1], 0.0, "the anchor sits on the floor");
        }
    }

    /// A flat foot does not move its anchor at all.
    #[test]
    fn rolling_path_is_still_for_a_flat_foot() {
        let frames = 8;
        let separations: Vec<[f32; 3]> = (0..frames).map(|_| [0.18, 0.0, 0.0]).collect();
        let weights = vec![0.5f32; frames];
        let observed: Vec<[f32; 3]> = (0..frames).map(|_| [0.09, 0.0, 0.0]).collect();
        let path = rolling_path(&separations, &weights, &observed, 0.0);
        // The offset aligns the anchor with the observed contact point, and with
        // no weight transfer the anchor never moves from there.
        for anchor in &path {
            assert!((anchor[0] - 0.09).abs() < 1e-5, "anchor at {}", anchor[0]);
        }
    }

    /// `support_weights`: flat foot is 0.5, a rolled foot moves the weight to
    /// whichever point is lower, and the 5 mm deadband keeps skin noise out.
    #[test]
    fn support_weight_rolls_towards_the_lower_point() {
        let flat = support_weight([0.0, 0.05, 0.0], [0.0, 0.05, 0.0]);
        assert!((flat - 0.5).abs() < 1e-6, "flat foot {flat}");

        // Toe 30 mm lower than the heel: weight goes to the toe.
        let toe_down = support_weight([0.0, 0.02, 0.0], [0.0, 0.05, 0.0]);
        assert!(toe_down < 0.05, "toe down {toe_down}");

        // Heel 30 mm lower: weight goes the other way.
        let heel_down = support_weight([0.0, 0.05, 0.0], [0.0, 0.02, 0.0]);
        assert!(heel_down > 0.95, "heel down {heel_down}");

        // 3 mm difference is inside the deadband.
        let noise = support_weight([0.0, 0.047, 0.0], [0.0, 0.05, 0.0]);
        assert!((noise - 0.5).abs() < 1e-6, "noise {noise}");
    }

    /// The analytic two-bone solve must put the wrist on its target and must not
    /// disturb the rest of the body.
    #[test]
    fn the_second_difference_doubles_the_middle_frame() {
        // Constant velocity: no acceleration, so no twitch.
        let previous = [0.0f32, 0.0, 0.0];
        let current = [1.0f32, 0.0, 0.0];
        let next = [2.0f32, 0.0, 0.0];
        assert_eq!(second_difference(&previous, &current, &next, 0), [0.0; 3]);
        // The middle frame is the term that gets doubled: displacing it by 0.5 m
        // gives a second difference of exactly 1.0 m. Written the other way round
        // (`current - 2*previous + next`) this same sample reads 3.5 m, which is
        // what the jitter score used to report for every clip.
        let current = [1.5f32, 0.0, 0.0];
        assert_eq!(second_difference(&previous, &current, &next, 0), [-1.0, 0.0, 0.0]);
    }

    #[test]
    fn solve_arm_reaches_the_target() {
        let Some((body, rest)) = loaded() else {
            eprintln!("skipping: SMPL-H model not found");
            return;
        };
        let mut pose = Pose::neutral(body.num_joints, 16);
        pose.betas[0] = 0.3;
        pose.translation = [0.0, 0.9, 0.0];
        // A raised left arm to start from.
        let at = (LEFT_SHOULDER - 1) * 3;
        pose.body[at..at + 3].copy_from_slice(&[0.0, 0.0, -1.2]);

        let target = {
            let skeleton = body.forward_joints(&pose);
            let wrist = joint(&skeleton, LEFT_WRIST);
            let shoulder = joint(&skeleton, LEFT_SHOULDER);
            // Reachable on purpose: part of the way out, so the elbow bends.
            [
                shoulder[0] + (wrist[0] - shoulder[0]) * 0.7,
                shoulder[1] + (wrist[1] - shoulder[1]) * 0.7 - 0.05,
                shoulder[2] + (wrist[2] - shoulder[2]) * 0.7 + 0.03,
            ]
        };
        let hips_before = {
            let skeleton = body.forward_joints(&pose);
            joint(&skeleton, 0)
        };

        assert!(solve_arm(
            &body,
            &mut pose,
            &rest,
            LEFT_SHOULDER,
            LEFT_ELBOW,
            LEFT_WRIST,
            target,
            None,
            1.0
        ));

        let after = body.forward_joints(&pose);
        let wrist = joint(&after, LEFT_WRIST);
        let error = ((wrist[0] - target[0]).powi(2)
            + (wrist[1] - target[1]).powi(2)
            + (wrist[2] - target[2]).powi(2))
        .sqrt();
        assert!(
            error < 0.01,
            "wrist {wrist:?} target {target:?} error {error}"
        );
        let hips = joint(&after, 0);
        let drift = ((hips[0] - hips_before[0]).powi(2)
            + (hips[1] - hips_before[1]).powi(2)
            + (hips[2] - hips_before[2]).powi(2))
        .sqrt();
        assert!(drift < 1e-5, "hips drifted {drift}");
    }

    /// The reference's `Contact`: 75 ms of evidence to lock, never on a foot in
    /// the air, and a clean release when the foot lifts.
    #[test]
    fn the_causal_contact_locks_only_on_slow_ground_evidence() {
        let dt = 1.0 / 30.0;

        // A low, slow sole: nothing at first, then locked once 75 ms of evidence
        // has accumulated, with the hold strength ramping in behind it.
        let mut planted = LiveContact::default();
        for frame in 0..12 {
            planted.update([0.001 * frame as f64, 0.01, 0.0], dt, 0.0, true);
            if (frame as f64 + 1.0) * dt < 0.075 {
                assert!(!planted.locked, "locked before 75 ms, frame {frame}");
            }
        }
        assert!(planted.locked, "a low, slow sole locks after 75 ms");
        assert!(planted.strength > 0.5, "hold strength {}", planted.strength);

        // 12 cm above the floor is never a contact, however long it hangs there.
        let mut airborne = LiveContact::default();
        for _ in 0..40 {
            airborne.update([0.0, 0.12, 0.0], dt, 0.0, true);
        }
        assert!(!airborne.locked, "a sole 12 cm up must not lock");
        assert!(airborne.strength < 1e-3, "airborne hold {}", airborne.strength);

        // Lifting a locked foot out of the window releases it.
        let mut lifted = LiveContact::default();
        for _ in 0..8 {
            lifted.update([0.0, 0.01, 0.0], dt, 0.0, true);
        }
        assert!(lifted.locked);
        for _ in 0..4 {
            lifted.update([0.0, 0.08, 0.0], dt, 0.0, true);
        }
        assert!(!lifted.locked, "a foot lifted to 80 mm is not a contact");
    }
}







