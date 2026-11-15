//! One original-motion floor, ported from `flow_floor.py` (RGB MoCap, MIT).
//!
//! The floor is **not** the lowest point a body reaches. It is the robust lower
//! edge of the sole patches on the frames the detector calls planted:
//!
//! ```text
//! per window:  values = height[foot, active]        (lowest patch of each foot)
//!              estimate = quantile(values, .05)
//! per actor:   median of its windows' estimates
//! whole scene: median over actors                   (equal-actor median)
//! ```
//!
//! The 5% quantile rather than the minimum is what keeps one bad frame (a
//! micro-penetration, a mis-detected stance) from lifting the whole body; the
//! cross-actor median is what gives every take one shared plane instead of each
//! actor standing on their own.
//!
//! Frames are measured in windows because the reference bounds the work per
//! window (`8192 // actors` covered by at most 512 frames at a time); a window
//! is also the smallest coherent unit for the contact detector, which needs
//! neighbouring frames to see a step.

use crate::contacts::{self, Config, Probes};

/// What the floor pass measured.
#[derive(Clone, Copy, Debug, Default)]
pub struct Floor {
    /// The plane, in the clip's own vertical datum.
    pub height: f32,
    /// (frame, foot) samples the estimates came from.
    pub samples: usize,
    /// Windows that produced an estimate.
    pub windows: usize,
}

impl Floor {
    /// No reliably planted sole anywhere: the reference keeps the datum and says
    /// so, instead of guessing a plane from a swinging foot.
    pub fn is_reliable(&self) -> bool {
        self.windows > 0
    }
}

/// The reference's window budget: at most 512 frames per window, at least 2.
pub fn budget(actors: usize) -> usize {
    (8192 / actors.max(1)).clamp(2, 512)
}

/// `flow_floor.resolve` for one actor.
pub fn resolve(probes: &[Probes], probabilities: &[[f32; 4]], fps: f32) -> Floor {
    let count = probes.len();
    if count == 0 {
        return Floor::default();
    }
    let window = budget(1);
    let mut estimates = Vec::new();
    let mut samples = 0usize;
    let mut first = 0usize;
    while first < count {
        let end = (first + window).min(count);
        let prepared = contacts::prepare(
            &probes[first..end],
            &probabilities[first..end],
            fps,
            &Config::normal(),
            None,
        );
        samples += prepared.samples;
        let mut values = Vec::new();
        for frame in 0..end - first {
            for foot in 0..2 {
                if !prepared.active[frame][foot] {
                    continue;
                }
                let row = &probes[first + frame];
                // `probes[...,1].reshape(-1,2,2).min(2)`: the lower of the two
                // patches is what carries the weight.
                values.push(row[foot][0][1].min(row[foot][1][1]));
            }
        }
        if !values.is_empty() {
            estimates.push(contacts::quantile(&values, 0.05));
        }
        first = end;
    }
    Floor {
        height: median(&estimates),
        samples,
        windows: estimates.len(),
    }
}

/// The scene floor: the median over actors, ignoring actors with no samples.
///
/// `None` when no actor produced an estimate — the caller then keeps its own
/// datum and reports the warning, exactly like the reference.
pub fn shared(estimates: &[Floor]) -> Option<Floor> {
    let heights: Vec<f32> = estimates
        .iter()
        .filter(|floor| floor.is_reliable())
        .map(|floor| floor.height)
        .collect();
    if heights.is_empty() {
        return None;
    }
    Some(Floor {
        height: median(&heights),
        samples: estimates.iter().map(|floor| floor.samples).sum(),
        windows: estimates.iter().map(|floor| floor.windows).sum(),
    })
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

#[cfg(test)]
mod tests {
    use super::*;

    fn planted(y: f32) -> Probes {
        [
            [[0.0, y, 0.10], [0.0, y, -0.10]],
            [[0.2, y, 0.10], [0.2, y, -0.10]],
        ]
    }

    fn steady(count: usize) -> (Vec<Probes>, Vec<[f32; 4]>) {
        (
            vec![planted(0.9); count],
            vec![[0.9f32; 4]; count],
        )
    }

    #[test]
    fn a_still_actor_puts_the_floor_at_the_sole() {
        let (probes, probabilities) = steady(60);
        let floor = resolve(&probes, &probabilities, 30.0);
        assert!(floor.is_reliable());
        assert!((floor.height - 0.9).abs() < 1e-5, "floor {}", floor.height);
        assert_eq!(floor.windows, 1);
    }

    /// The 5% quantile ignores a handful of frames that dip through the floor,
    /// which is the whole point of not using the minimum.
    #[test]
    fn a_few_penetrating_frames_do_not_move_the_plane() {
        let (mut probes, probabilities) = steady(100);
        for row in probes.iter_mut().take(3) {
            *row = planted(0.6);
        }
        let floor = resolve(&probes, &probabilities, 30.0);
        assert!(
            (floor.height - 0.9).abs() < 1e-5,
            "three bad frames out of a hundred moved the floor to {}",
            floor.height
        );
    }

    /// A swinging foot is not a floor sample: with no contacts at all there is no
    /// estimate, and the caller keeps its own datum.
    #[test]
    fn no_contacts_means_no_estimate() {
        let count = 60;
        let probes: Vec<Probes> = (0..count)
            .map(|frame| {
                // Both feet sweeping sideways at 3 m/s: far past the 0.7 m/s gate.
                let mut row = planted(0.9);
                let x = frame as f32 * 0.1;
                for foot in row.iter_mut() {
                    for side in foot.iter_mut() {
                        side[0] += x;
                    }
                }
                row
            })
            .collect();
        let probabilities = vec![[0.9f32; 4]; count];
        let floor = resolve(&probes, &probabilities, 30.0);
        assert!(!floor.is_reliable());
    }

    #[test]
    fn the_shared_floor_is_the_median_over_actors() {
        let a = Floor {
            height: 0.1,
            samples: 10,
            windows: 1,
        };
        let b = Floor {
            height: 0.3,
            samples: 10,
            windows: 1,
        };
        let c = Floor {
            height: 0.2,
            samples: 10,
            windows: 1,
        };
        let combined = shared(&[a, b, c]).expect("three actors");
        assert!((combined.height - 0.2).abs() < 1e-6);
        let skipped = Floor::default();
        let combined = shared(&[a, skipped, b]).expect("two of three actors");
        assert!((combined.height - 0.2).abs() < 1e-6);
        assert!(shared(&[skipped]).is_none());
    }

    #[test]
    fn the_window_budget_follows_the_reference() {
        assert_eq!(budget(1), 512);
        assert_eq!(budget(16), 512);
        assert_eq!(budget(8192), 2);
        assert_eq!(budget(0), 512);
    }
}
