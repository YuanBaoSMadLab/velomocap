//! How stable a planted foot actually is.
//!
//! The correction reports jitter of the joints and how far the clip floats, which
//! says whether the *pass* helped, not what a foot does on the ground. The number
//! that reads as "the feet are unstable" is the sole's own motion while the
//! detector says the foot is planted: how far it travels horizontally, whether it
//! sinks through the floor, whether it lifts, and how much it twitches.
//!
//! This is a diagnostic, not part of the algorithm: it is measured on the same
//! probes `contacts::prepare` uses (`height` = the lower patch, `point` = the mean
//! of both patches), so the numbers describe exactly the surface the contact gate
//! saw.

use crate::contacts::{self, Probes};

/// What one foot does while it is planted.
#[derive(Clone, Copy, Debug, Default)]
pub struct FootMetrics {
    /// Mean horizontal travel per contact-to-contact frame, in cm. This is the
    /// slide a viewer sees as the foot skating over the ground.
    pub slide_cm: [f32; 2],
    /// 95th percentile of the same travel, in cm: the worst few frames.
    pub slide_p95_cm: [f32; 2],
    /// Deepest the sole went below the floor, in mm (0 if it never sank).
    pub sink_mm: [f32; 2],
    /// Mean height of the sole above the floor while planted, in mm.
    pub float_mm: [f32; 2],
    /// Median second difference of the sole height, in mm: the twitch.
    pub jitter_mm: [f32; 2],
    /// Contact frames and contact runs counted per foot.
    pub frames: [usize; 2],
    pub runs: [usize; 2],
}

impl FootMetrics {
    pub fn empty(&self) -> bool {
        self.frames == [0, 0]
    }

    pub fn line(&self) -> String {
        format!(
            "feet: slide {:.2}/{:.2} cm/frame (p95 {:.2}/{:.2}), float {:.1}/{:.1} mm, sink {:.1}/{:.1} mm, jitter {:.1}/{:.1} mm over {}/{} contact frames in {}/{} runs",
            self.slide_cm[0],
            self.slide_cm[1],
            self.slide_p95_cm[0],
            self.slide_p95_cm[1],
            self.float_mm[0],
            self.float_mm[1],
            self.sink_mm[0],
            self.sink_mm[1],
            self.jitter_mm[0],
            self.jitter_mm[1],
            self.frames[0],
            self.frames[1],
            self.runs[0],
            self.runs[1]
        )
    }

    /// The same numbers as a before → after comparison.
    pub fn delta_line(before: &Self, after: &Self) -> String {
        format!(
            "feet: slide {:.2}/{:.2} -> {:.2}/{:.2} cm/frame, float {:.1}/{:.1} -> {:.1}/{:.1} mm, sink {:.1}/{:.1} -> {:.1}/{:.1} mm, jitter {:.1}/{:.1} -> {:.1}/{:.1} mm",
            before.slide_cm[0],
            before.slide_cm[1],
            after.slide_cm[0],
            after.slide_cm[1],
            before.float_mm[0],
            before.float_mm[1],
            after.float_mm[0],
            after.float_mm[1],
            before.sink_mm[0],
            before.sink_mm[1],
            after.sink_mm[0],
            after.sink_mm[1],
            before.jitter_mm[0],
            before.jitter_mm[1],
            after.jitter_mm[0],
            after.jitter_mm[1]
        )
    }
}

/// The lower patch's height for one foot on one frame (`min` of both patches).
pub fn height(probes: &Probes, foot: usize) -> f32 {
    probes[foot][0][1].min(probes[foot][1][1])
}

/// The contact point of one foot on one frame: the mean of both patches.
pub fn point(probes: &Probes, foot: usize) -> [f32; 3] {
    [
        0.5 * (probes[foot][0][0] + probes[foot][1][0]),
        0.5 * (probes[foot][0][1] + probes[foot][1][1]),
        0.5 * (probes[foot][0][2] + probes[foot][1][2]),
    ]
}

/// Measure a clip's planted feet. `active` is the contact mask the correction was
/// asked to honour; `floor` the height the soles should sit on.
pub fn measure(rows: &[Probes], active: &[[bool; 2]], floor: f32) -> FootMetrics {
    let mut out = FootMetrics::default();
    let count = rows.len().min(active.len());
    for foot in 0..2 {
        let mut slides: Vec<f32> = Vec::new();
        let mut float = 0.0f32;
        let mut sink = 0.0f32;
        let mut jitters: Vec<f32> = Vec::new();
        let mut frames = 0usize;
        let mut runs = 0usize;
        for frame in 0..count {
            if !active[frame][foot] {
                continue;
            }
            frames += 1;
            if frame == 0 || !active[frame - 1][foot] {
                runs += 1;
            }
            let at = point(&rows[frame], foot);
            let top = height(&rows[frame], foot);
            float += (top - floor).max(0.0);
            sink = sink.max((floor - top).max(0.0));
            // Slide is only meaningful while the foot stays down: a step between
            // two separate stances is the foot moving, not skating.
            if frame > 0 && active[frame - 1][foot] {
                let last = point(&rows[frame - 1], foot);
                let dx = at[0] - last[0];
                let dz = at[2] - last[2];
                slides.push((dx * dx + dz * dz).sqrt() * 100.0);
            }
            if frame > 0
                && frame + 1 < count
                && active[frame - 1][foot]
                && active[frame + 1][foot]
            {
                let previous = height(&rows[frame - 1], foot);
                let next = height(&rows[frame + 1], foot);
                jitters.push((previous - 2.0 * top + next).abs() * 1000.0);
            }
        }
        if frames == 0 {
            continue;
        }
        out.frames[foot] = frames;
        out.runs[foot] = runs;
        let mean = slides.iter().sum::<f32>() / slides.len().max(1) as f32;
        out.slide_cm[foot] = mean;
        out.slide_p95_cm[foot] = contacts::quantile(&slides, 0.95);
        out.float_mm[foot] = float * 1000.0 / frames as f32;
        out.sink_mm[foot] = sink * 1000.0;
        out.jitter_mm[foot] = contacts::quantile(&jitters, 0.5);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One frame of probes with a given sole height and horizontal position.
    fn probes(height: f32, x: f32) -> Probes {
        [
            [[x + 0.05, height, 0.10], [x - 0.05, height + 0.01, -0.10]],
            [[x + 0.05, height, 0.10], [x - 0.05, height + 0.01, -0.10]],
        ]
    }

    #[test]
    fn a_planted_foot_has_no_slide() {
        let rows: Vec<Probes> = (0..10).map(|_| probes(0.0, 0.0)).collect();
        let active = vec![[true, false]; 10];
        let metrics = measure(&rows, &active, 0.0);
        assert_eq!(metrics.frames, [10, 0]);
        assert_eq!(metrics.runs, [1, 0]);
        assert!(metrics.slide_cm[0] < 1e-4, "{}", metrics.line());
        assert!(metrics.jitter_mm[0] < 1e-4, "{}", metrics.line());
        assert!(metrics.sink_mm[0] < 1e-4);
        assert!(metrics.float_mm[0] < 0.1);
    }

    /// A sole that travels a centimetre per frame reads as 1 cm/frame, and a sole
    /// that sinks two millimetres reports it.
    #[test]
    fn slide_and_sink_are_measured_on_the_sole() {
        let rows: Vec<Probes> = (0..5)
            .map(|frame| probes(-0.002, frame as f32 * 0.01))
            .collect();
        let active = vec![[true, false]; 5];
        let metrics = measure(&rows, &active, 0.0);
        assert!((metrics.slide_cm[0] - 1.0).abs() < 1e-3, "{}", metrics.line());
        assert!((metrics.sink_mm[0] - 2.0).abs() < 1e-3, "{}", metrics.line());
        // And when the sole sits above the floor it is float, not sink.
        let rows: Vec<Probes> = (0..5).map(|_| probes(0.004, 0.0)).collect();
        let metrics = measure(&rows, &active, 0.0);
        assert!(metrics.sink_mm[0] < 1e-4);
        assert!((metrics.float_mm[0] - 4.0).abs() < 1e-3, "{}", metrics.line());
    }

    /// Steps between stances are not slide, and the twitch only counts frames
    /// with both neighbours planted.
    #[test]
    fn a_step_does_not_count_as_slide() {
        let mut active = vec![[false, false]; 6];
        active[0] = [true, false];
        active[1] = [true, false];
        active[4] = [true, false];
        active[5] = [true, false];
        let rows: Vec<Probes> = (0..6)
            .map(|frame| probes(0.0, if frame < 2 { 0.0 } else { 0.5 }))
            .collect();
        let metrics = measure(&rows, &active, 0.0);
        assert_eq!(metrics.frames, [4, 0]);
        assert_eq!(metrics.runs, [2, 0]);
        assert!(metrics.slide_cm[0] < 1e-4, "a step counted as slide: {}", metrics.line());
    }
}
