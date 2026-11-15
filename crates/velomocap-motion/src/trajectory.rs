//! Root trajectory correction, ported from NVIDIA Kimodo's `MotionCorrection`.
//!
//! Source: `nv-tlabs/kimodo`, `MotionCorrection/src/cpp/AnimProcessing/
//! TrajectoryCorrector.cpp`, Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES,
//! Apache License 2.0.
//!
//! The problem, verbatim from the original:
//!
//! ```text
//! E(x) = pos_weight * |x - x_orig|^2
//!      + vel_weight * |V x - V x_orig|^2
//!      + acc_weight * |A x - A x_orig|^2
//! ```
//!
//! minimised subject to pinned frames (`margin == 0`) and a maximum deviation
//! from a target on the frames where the solver may move (`margin > 0`).
//!
//! Kimodo solves it with ADMM and a sparse LU factorisation; this port keeps the
//! same iteration (x, z, u updates) and solves the inner system with conjugate
//! gradients, since `M` is symmetric positive definite and the operators are
//! banded. The matrices are never materialised: `V` and `A` are difference
//! operators, so one product is a linear pass over the frames.

/// Per-frame constraint for one coordinate.
///
/// * `< 0` - the frame is free,
/// * `0` - the frame is pinned to its observation,
/// * `> 0` - the frame may differ from the observation by at most this much.
pub type Margins = Vec<f64>;

pub struct TrajectoryCorrector {
    frames: usize,
    margin_locs: Vec<usize>,
    margin_vals: Vec<f64>,
    constrained: Vec<usize>,
    unconstrained: Vec<usize>,
    masked: Vec<bool>,
    pos_weight: f64,
    vel_weight: f64,
    acc_weight: f64,
    rho: f64,
    iters: usize,
    velocity_weights: Vec<f64>,
}

impl TrajectoryCorrector {
    pub fn new(
        margins: &[f64],
        pos_weight: f64,
        vel_weight: f64,
        acc_weight: f64,
        velocity_weights: &[f64],
        iters: usize,
    ) -> Self {
        let frames = margins.len();
        let mut margin_locs = Vec::new();
        let mut margin_vals = Vec::new();
        let mut constrained = Vec::new();
        let mut unconstrained = Vec::new();
        for (index, margin) in margins.iter().enumerate() {
            if *margin > 0.0 {
                margin_locs.push(index);
                margin_vals.push(*margin);
            }
            if *margin == 0.0 {
                constrained.push(index);
            } else {
                unconstrained.push(index);
            }
        }
        let mut masked = vec![false; frames];
        for index in &margin_locs {
            masked[*index] = true;
        }

        // Kimodo's step size: half the square root of the largest diagonal of N.
        // The diagonal of `V^T V` is at most 4 and of `A^T A` at most 6.
        let diagonal = pos_weight + vel_weight * 2.0 + acc_weight * 6.0;
        let rho = 0.5 * diagonal.max(1e-6).sqrt();

        Self {
            frames,
            margin_locs,
            margin_vals,
            constrained,
            unconstrained,
            masked,
            pos_weight,
            vel_weight,
            acc_weight,
            rho,
            iters: iters.max(1),
            velocity_weights: if velocity_weights.is_empty() {
                vec![1.0; frames]
            } else {
                velocity_weights.to_vec()
            },
        }
    }

    /// `velocity = x[i+1] - x[i]`, `acceleration = -x[i] + 2x[i+1] - x[i+2]`.
    fn velocity(&self, x: &[f64], out: &mut [f64]) {
        for index in 0..self.frames.saturating_sub(1) {
            out[index] = x[index + 1] - x[index];
        }
    }

    fn acceleration(&self, x: &[f64], out: &mut [f64]) {
        for index in 0..self.frames.saturating_sub(2) {
            out[index] = x[index + 2] - 2.0 * x[index + 1] + x[index];
        }
    }

    /// `N v = pos * v + vel * V^T V v + acc * A^T A v`.
    fn apply_n(&self, x: &[f64], out: &mut [f64]) {
        let n = self.frames;
        for value in out.iter_mut() {
            *value = 0.0;
        }
        // Position term.
        for index in 0..n {
            out[index] += self.pos_weight * x[index];
        }
        // Velocity term: V^T V, with the per-column weights Kimodo applies.
        if n > 1 {
            let mut velocity = vec![0.0; n - 1];
            self.velocity(x, &mut velocity);
            for index in 0..n - 1 {
                let weight = self.vel_weight * self.velocity_weights[index].max(0.0).powi(2);
                let value = weight * velocity[index];
                out[index] -= value;
                out[index + 1] += value;
            }
        }
        // Acceleration term: A^T A.
        if n > 2 {
            let mut acceleration = vec![0.0; n - 2];
            self.acceleration(x, &mut acceleration);
            for index in 0..n - 2 {
                let value = self.acc_weight * acceleration[index];
                out[index] += value;
                out[index + 1] -= 2.0 * value;
                out[index + 2] += value;
            }
        }
    }

    /// Reduced operator `S M S^T` on the unconstrained rows.
    fn apply_reduced(&self, reduced: &[f64], out: &mut [f64], work: &mut [f64]) {
        for value in work.iter_mut() {
            *value = 0.0;
        }
        for (slot, index) in self.unconstrained.iter().enumerate() {
            work[*index] = reduced[slot];
        }
        let mut applied = vec![0.0; self.frames];
        self.apply_n(work, &mut applied);
        for (slot, index) in self.unconstrained.iter().enumerate() {
            let mut value = applied[*index];
            if self.masked[*index] {
                value += self.rho * work[*index];
            }
            out[slot] = value;
        }
    }

    /// Conjugate gradients on the reduced system.
    fn solve(&self, rhs: &[f64]) -> Vec<f64> {
        let count = rhs.len();
        if count == 0 {
            return Vec::new();
        }
        let mut work = vec![0.0; self.frames];
        let mut x = vec![0.0; count];
        let mut r = rhs.to_vec();
        let mut p = r.clone();
        let mut ap = vec![0.0; count];
        let mut rr: f64 = r.iter().map(|v| v * v).sum();
        let limit = rr * 1e-18;
        for _ in 0..count.max(8) {
            if rr <= limit {
                break;
            }
            self.apply_reduced(&p, &mut ap, &mut work);
            let pap: f64 = p.iter().zip(&ap).map(|(a, b)| a * b).sum();
            if pap.abs() < 1e-20 {
                break;
            }
            let alpha = rr / pap;
            for index in 0..count {
                x[index] += alpha * p[index];
                r[index] -= alpha * ap[index];
            }
            let next: f64 = r.iter().map(|v| v * v).sum();
            let beta = next / rr;
            for index in 0..count {
                p[index] = r[index] + beta * p[index];
            }
            rr = next;
        }
        x
    }

    /// Correct one coordinate series. `observation` is the target for pinned and
    /// margin frames, `reference` is the trajectory whose shape is preserved.
    pub fn interpolate(&self, observation: &[f64], reference: &[f64]) -> Vec<f64> {
        let mut x = reference.to_vec();
        if self.constrained.is_empty() && self.margin_locs.is_empty() {
            return x;
        }

        // r = N (x_t - x_o|constrained) + rho (z - u) at the margin frames.
        let mut diffs = reference.to_vec();
        for index in &self.constrained {
            diffs[*index] -= observation[*index];
        }
        let mut r = vec![0.0; self.frames];
        self.apply_n(&diffs, &mut r);

        if self.margin_locs.is_empty() {
            let mut rhs = vec![0.0; self.unconstrained.len()];
            for (slot, index) in self.unconstrained.iter().enumerate() {
                rhs[slot] = r[*index];
            }
            let solved = self.solve(&rhs);
            for (slot, index) in self.unconstrained.iter().enumerate() {
                x[*index] = solved[slot];
            }
            for index in &self.constrained {
                x[*index] = observation[*index];
            }
            return x;
        }

        let count = self.margin_locs.len();
        let mut z = vec![0.0; count];
        let mut z_target = vec![0.0; count];
        let mut u = vec![0.0; count];
        for (slot, index) in self.margin_locs.iter().enumerate() {
            z_target[slot] = observation[*index];
            z[slot] = reference[*index];
        }

        for _ in 0..self.iters {
            // x update
            let mut rhs = r.clone();
            for (slot, index) in self.margin_locs.iter().enumerate() {
                rhs[*index] += self.rho * (z[slot] - u[slot]);
            }
            let mut reduced = vec![0.0; self.unconstrained.len()];
            for (slot, index) in self.unconstrained.iter().enumerate() {
                reduced[slot] = rhs[*index];
            }
            let solved = self.solve(&reduced);
            for (slot, index) in self.unconstrained.iter().enumerate() {
                x[*index] = solved[slot];
            }
            for index in &self.constrained {
                x[*index] = observation[*index];
            }

            // z update: project onto the margin ball around the target.
            for (slot, index) in self.margin_locs.iter().enumerate() {
                let mut diff = x[*index] + u[slot] - z_target[slot];
                let norm = diff.abs();
                if norm > self.margin_vals[slot] {
                    diff *= self.margin_vals[slot] / norm;
                }
                z[slot] = z_target[slot] + diff;
            }

            // u update
            for (slot, index) in self.margin_locs.iter().enumerate() {
                u[slot] += x[*index] - z[slot];
            }
        }
        x
    }

    /// Multi-coordinate convenience wrapper (Kimodo solves 2 or 3 columns at once
    /// so a corner is handled jointly; the coordinates do not interact, so
    /// solving them one by one gives the same answer).
    pub fn interpolate_columns(
        &self,
        observation: &[Vec<f64>],
        reference: &[Vec<f64>],
    ) -> Vec<Vec<f64>> {
        let columns = reference.first().map(|row| row.len()).unwrap_or(0);
        if columns == 0 {
            return reference.to_vec();
        }
        let mut out = reference.to_vec();
        for column in 0..columns {
            let observed: Vec<f64> = observation.iter().map(|row| row[column]).collect();
            let reference_column: Vec<f64> = reference.iter().map(|row| row[column]).collect();
            let solved = self.interpolate(&observed, &reference_column);
            for (row, value) in solved.into_iter().enumerate() {
                out[row][column] = value;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_frames_are_exact() {
        let frames = 12;
        let mut margins = vec![-1.0; frames];
        margins[0] = 0.0;
        margins[6] = 0.0;
        let reference: Vec<f64> = (0..frames).map(|i| i as f64 * 0.1).collect();
        let mut observation = reference.clone();
        observation[6] = reference[6] + 0.25;
        let corrector = TrajectoryCorrector::new(&margins, 1.0, 10.0, 100.0, &[], 40);
        let corrected = corrector.interpolate(&observation, &reference);
        assert!((corrected[0] - observation[0]).abs() < 1e-6);
        assert!(
            (corrected[6] - observation[6]).abs() < 1e-6,
            "pinned frame moved: {} vs {}",
            corrected[6],
            observation[6]
        );
    }

    #[test]
    fn a_margin_frame_stays_within_the_margin() {
        let frames = 15;
        let mut margins = vec![-1.0; frames];
        margins[7] = 0.05;
        let reference: Vec<f64> = (0..frames).map(|i| (i as f64 * 0.3).sin() * 0.1).collect();
        let mut observation = reference.clone();
        // Ask for a 0.4 m jump, far outside the margin.
        observation[7] = reference[7] + 0.4;
        let corrector = TrajectoryCorrector::new(&margins, 1.0, 8.0, 20.0, &[], 60);
        let corrected = corrector.interpolate(&observation, &reference);
        let offset = (corrected[7] - observation[7]).abs();
        assert!(
            offset <= 0.05 + 1e-3,
            "frame 7 deviates {offset:.4} m, margin is 0.05"
        );
    }

    #[test]
    fn a_free_series_is_returned_unchanged() {
        let frames = 8;
        let margins = vec![-1.0; frames];
        let reference: Vec<f64> = (0..frames).map(|i| i as f64).collect();
        let corrector = TrajectoryCorrector::new(&margins, 1.0, 1.0, 1.0, &[], 5);
        let corrected = corrector.interpolate(&reference, &reference);
        assert_eq!(corrected, reference);
    }

    #[test]
    fn a_pin_smooths_the_neighbourhood() {
        let frames = 20;
        let mut margins = vec![-1.0; frames];
        margins[10] = 0.0;
        // A jittery reference with one frame pulled down.
        let reference: Vec<f64> = (0..frames)
            .map(|i| 0.5 + if i % 2 == 0 { 0.01 } else { -0.01 } + i as f64 * 0.001)
            .collect();
        let mut observation = reference.clone();
        observation[10] -= 0.1;
        let corrector = TrajectoryCorrector::new(&margins, 1.0, 6.0, 30.0, &[], 50);
        let corrected = corrector.interpolate(&observation, &reference);
        let before: f64 = reference.windows(2).map(|w| (w[1] - w[0]).abs()).sum();
        let after: f64 = corrected.windows(2).map(|w| (w[1] - w[0]).abs()).sum();
        assert!(
            after < before,
            "correction should not add jitter: {before:.4} -> {after:.4}"
        );
    }
}
