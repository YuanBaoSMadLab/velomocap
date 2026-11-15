//! Bounded nonlinear least squares: the solver the reference's two planning
//! passes are built on (RGB MoCap, MIT).
//!
//! `contact_arm_plan.solve` and `contact_pose_plan._optimise_window` both hand a
//! residual vector to `scipy.optimize.least_squares` with box bounds and a
//! sparse Jacobian pattern, then read the parameters back. The Python code never
//! touches the solver, so what has to be reproduced is the objective and the
//! bounds; the search itself only has to converge on the same problem.
//!
//! This is a Levenberg-Marquardt iteration with a projected step: the step is
//! solved from `(JᵀJ + λ·diag(JᵀJ)) δ = -Jᵀr`, clamped into the box, and the
//! damping is retuned from the cost. `scipy`'s default `trf` method is a
//! trust-region variant of the same normal equations, so a converged solution is
//! the same stationary point of the same objective inside the same box.
//!
//! The Jacobian is evaluated by forward differences. Both planning problems have
//! a few hundred parameters and a few hundred residuals, so a dense finite
//! difference Jacobian costs milliseconds and removes the need for the sparse
//! pattern the reference has to pass in by hand.

/// Iteration control.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Maximum number of accepted-or-rejected trials, like `max_nfev`.
    pub max_iterations: usize,
    /// Relative cost improvement that counts as converged.
    pub tolerance: f64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            max_iterations: 80,
            tolerance: 1e-9,
        }
    }
}

/// A converged (or exhausted) solve.
#[derive(Clone, Debug)]
pub struct Solution {
    pub x: Vec<f64>,
    /// `0.5·|r|²` at `x`.
    pub cost: f64,
    /// Trials run, counted like `least_squares`' `nfev`.
    pub iterations: usize,
    pub converged: bool,
}

fn clamp_into(x: &mut [f64], lower: &[f64], upper: &[f64]) {
    for i in 0..x.len() {
        let low = lower[i];
        let high = upper[i].max(low);
        x[i] = x[i].clamp(low, high);
    }
}

fn cost_of(residual: &[f64]) -> f64 {
    0.5 * residual.iter().map(|v| v * v).sum::<f64>()
}

/// Cholesky factorisation of a symmetric positive definite matrix, in place.
/// Returns false when the matrix is not positive definite (the caller then adds
/// damping and retries), which is the usual LM safeguard.
fn cholesky(a: &mut [f64], n: usize) -> bool {
    for row in 0..n {
        for column in 0..=row {
            let mut sum = a[row * n + column];
            for k in 0..column {
                sum -= a[row * n + k] * a[column * n + k];
            }
            if row == column {
                if sum <= 0.0 || !sum.is_finite() {
                    return false;
                }
                a[row * n + column] = sum.sqrt();
            } else {
                a[row * n + column] = sum / a[column * n + column];
            }
        }
    }
    true
}

/// Solve `A x = b` for a factorised `A` (lower triangle of the Cholesky factor).
fn cholesky_solve(factor: &[f64], n: usize, b: &mut [f64]) {
    for row in 0..n {
        let mut sum = b[row];
        for column in 0..row {
            sum -= factor[row * n + column] * b[column];
        }
        b[row] = sum / factor[row * n + row];
    }
    for row in (0..n).rev() {
        let mut sum = b[row];
        for column in row + 1..n {
            sum -= factor[column * n + row] * b[column];
        }
        b[row] = sum / factor[row * n + row];
    }
}

/// Minimise `0.5·|residual(x)|²` inside `[lower, upper]`.
///
/// The bound arrays must be the same length as `x0`; a variable with
/// `upper == lower` is held fixed. `residual` must return the same number of
/// values for every input.
pub fn minimise<F>(
    x0: &[f64],
    lower: &[f64],
    upper: &[f64],
    residual: F,
    options: Options,
) -> Solution
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    assert_eq!(lower.len(), n, "bounds must match the parameters");
    assert_eq!(upper.len(), n, "bounds must match the parameters");
    let mut x = x0.to_vec();
    clamp_into(&mut x, lower, upper);
    let mut r = residual(&x);
    let m = r.len();
    let mut cost = cost_of(&r);
    let mut damping = 1e-3;
    let mut converged = n == 0 || m == 0;
    let mut iterations = 0usize;

    // One Jacobian column per free parameter, filled by forward differences.
    let mut jacobian = vec![0.0f64; m * n];
    while iterations < options.max_iterations && !converged {
        iterations += 1;
        // Jacobian. A parameter pinned by the box gets a zero column, which
        // removes it from the step.
        let mut free = 0usize;
        for column in 0..n {
            if upper[column] - lower[column] <= 1e-12 {
                for row in 0..m {
                    jacobian[row * n + column] = 0.0;
                }
                continue;
            }
            free += 1;
            let step = 1e-7 * x[column].abs().max(1.0);
            let original = x[column];
            // Probe inwards when the parameter already sits on its upper bound.
            let probe = if original + step <= upper[column] {
                original + step
            } else {
                original - step
            };
            let delta = probe - original;
            if delta.abs() < 1e-15 {
                for row in 0..m {
                    jacobian[row * n + column] = 0.0;
                }
                continue;
            }
            x[column] = probe;
            let shifted = residual(&x);
            x[column] = original;
            for row in 0..m {
                jacobian[row * n + column] = (shifted[row] - r[row]) / delta;
            }
        }
        if free == 0 {
            converged = true;
            break;
        }

        // Normal equations JᵀJ and Jᵀr.
        let mut normal = vec![0.0f64; n * n];
        let mut gradient = vec![0.0f64; n];
        for row in 0..m {
            let base = row * n;
            for a in 0..n {
                let left = jacobian[base + a];
                if left == 0.0 {
                    continue;
                }
                gradient[a] += left * r[row];
                for b in a..n {
                    let right = jacobian[base + b];
                    if right != 0.0 {
                        normal[a * n + b] += left * right;
                    }
                }
            }
        }
        for a in 0..n {
            for b in 0..a {
                normal[a * n + b] = normal[b * n + a];
            }
        }

        // Damped step; grow the damping until the system is solvable.
        let mut attempt = 0u32;
        let mut accepted = false;
        while attempt < 12 {
            attempt += 1;
            let mut system = normal.clone();
            for a in 0..n {
                let diagonal = normal[a * n + a];
                system[a * n + a] += if diagonal > 0.0 {
                    damping * diagonal
                } else {
                    damping
                };
            }
            if !cholesky(&mut system, n) {
                damping *= 10.0;
                continue;
            }
            let mut step = gradient.clone();
            for value in step.iter_mut() {
                *value = -*value;
            }
            cholesky_solve(&system, n, &mut step);
            let mut trial = x.clone();
            for a in 0..n {
                if upper[a] - lower[a] <= 1e-12 {
                    continue;
                }
                trial[a] += step[a];
            }
            clamp_into(&mut trial, lower, upper);
            let trial_residual = residual(&trial);
            let trial_cost = cost_of(&trial_residual);
            if std::env::var("VELOMOCAP_DEBUG_SOLVER").is_ok() {
                eprintln!(
                    "it {iterations} attempt {attempt} damping {damping:.3e} cost {cost:.6e} trial {trial_cost:.6e} step {step:?} trial {trial:?}"
                );
            }
            if trial_cost.is_finite() && trial_cost < cost {
                let improvement = cost - trial_cost;
                x = trial;
                r = trial_residual;
                cost = trial_cost;
                damping = (damping * 0.3).max(1e-12);
                accepted = true;
                if improvement <= options.tolerance * cost.max(1e-300) {
                    converged = true;
                }
                break;
            }
            damping *= 5.0;
        }
        if !accepted {
            // No damping in the tried range improves the cost: a local minimum.
            converged = true;
        }
        if damping > 1e14 {
            converged = true;
        }
    }

    Solution {
        x,
        cost,
        iterations,
        converged,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A square linear system is solved in one step, and the reported cost is the
    /// half-sum-of-squares the planners use.
    #[test]
    fn minimise_finds_a_quadratic_bowl() {
        let solution = minimise(
            &[0.0, 0.0],
            &[-10.0, -10.0],
            &[10.0, 10.0],
            |x| vec![x[0] - 1.5, x[1] + 0.5],
            Options::default(),
        );
        assert!(solution.converged, "did not converge");
        assert!((solution.x[0] - 1.5).abs() < 1e-9, "{:?}", solution.x);
        assert!((solution.x[1] + 0.5).abs() < 1e-9, "{:?}", solution.x);
        assert!(solution.cost < 1e-18, "cost {:?}", solution.cost);
    }

    /// The box is hard: an optimum outside it stops on the boundary.
    #[test]
    fn minimise_stops_at_the_bound() {
        let solution = minimise(
            &[0.0],
            &[0.0],
            &[1.0],
            |x| vec![x[0] - 5.0],
            Options::default(),
        );
        assert!((solution.x[0] - 1.0).abs() < 1e-9, "{:?}", solution.x);
        assert!(solution.x[0] <= 1.0, "left the box");

        let pinned = minimise(&[0.3], &[0.7], &[0.7], |x| vec![x[0] - 2.0], Options::default());
        assert!((pinned.x[0] - 0.7).abs() < 1e-12, "{:?}", pinned.x);
    }

    /// A nonlinear fit: dense finite differences plus damping must recover
    /// parameters a linear solve cannot, which is the case both planning passes
    /// are (reach, bend angle and wrist trim are all nonlinear in the pose).
    #[test]
    fn minimise_fits_a_nonlinear_model() {
        let truth = [2.5, 0.7];
        let times: Vec<f64> = (0..12).map(|i| i as f64 * 0.25).collect();
        let samples: Vec<f64> = times
            .iter()
            .map(|t| truth[0] * (-truth[1] * t).exp())
            .collect();
        let solution = minimise(
            &[1.0, 0.2],
            &[0.0, 0.0],
            &[10.0, 5.0],
            |x| {
                times
                    .iter()
                    .zip(&samples)
                    .map(|(t, sample)| x[0] * (-x[1] * t).exp() - sample)
                    .collect()
            },
            Options::default(),
        );
        assert!(solution.converged, "did not converge");
        assert!((solution.x[0] - truth[0]).abs() < 1e-4, "{:?}", solution.x);
        assert!((solution.x[1] - truth[1]).abs() < 1e-4, "{:?}", solution.x);
        assert!(solution.cost < 1e-12, "cost {:?}", solution.cost);
    }

    /// An optimum already at the start is reported as converged without steps,
    /// so callers can skip the work when the pose needs no correction.
    #[test]
    fn minimise_at_the_optimum_does_not_move() {
        let solution = minimise(&[1.0, -2.0], &[-5.0; 2], &[5.0; 2], |x| {
            vec![x[0] - 1.0, x[1] + 2.0]
        }, Options::default());
        assert!((solution.x[0] - 1.0).abs() < 1e-12, "{:?}", solution.x);
        assert!((solution.x[1] + 2.0).abs() < 1e-12, "{:?}", solution.x);
        assert!(solution.cost < 1e-20, "cost {:?}", solution.cost);
    }
}
