//! Fit SMPL-X pose parameters to 2D keypoints.
//!
//! This is the stage that turns a video into motion. Given 17 COCO keypoints
//! per frame, it solves for the SMPL-X body pose, root orientation and
//! translation so that the projected 3D joints land on the observed 2D points.
//!
//! The solver is a Levenberg-Marquardt least-squares optimiser over a reduced
//! parameter set:
//!
//! * 3 root orientation (axis-angle)
//! * 3 root translation (metres)
//! * 3 spine/neck, 2×3 arm, 2×3 leg, 2×3 hip rotations
//!
//! Only the body joints are optimised; hands, face and shape stay fixed. That
//! keeps each frame's solve small (a ~30-variable dense problem) and stable
//! from a single camera.
//!
//! Weak perspective is assumed: `u = f * x / z + cx`, with `f` derived from the
//! subject's bounding box so no camera calibration is required.

use anyhow::Result;
use velomocap_body::{BodyModel, Pose, Skeleton};

use velomocap_infer::pose::PoseDetection;

/// COCO keypoint indices, named for readability.
mod kp {
    pub const NOSE: usize = 0;
    pub const L_SHOULDER: usize = 5;
    pub const R_SHOULDER: usize = 6;
    pub const L_ELBOW: usize = 7;
    pub const R_ELBOW: usize = 8;
    pub const L_WRIST: usize = 9;
    pub const R_WRIST: usize = 10;
    pub const L_HIP: usize = 11;
    pub const R_HIP: usize = 12;
    pub const L_KNEE: usize = 13;
    pub const R_KNEE: usize = 14;
    pub const L_ANKLE: usize = 15;
    pub const R_ANKLE: usize = 16;
}

/// SMPL-X joint indices for the same landmarks.
mod sj {
    pub const PELVIS: usize = 0;
    pub const L_HIP: usize = 1;
    pub const R_HIP: usize = 2;
    pub const SPINE1: usize = 3;
    pub const L_KNEE: usize = 4;
    pub const R_KNEE: usize = 5;
    pub const SPINE2: usize = 6;
    pub const L_ANKLE: usize = 7;
    pub const R_ANKLE: usize = 8;
    pub const SPINE3: usize = 9;
    pub const NECK: usize = 12;
    pub const HEAD: usize = 15;
    pub const L_SHOULDER: usize = 16;
    pub const R_SHOULDER: usize = 17;
    pub const L_ELBOW: usize = 18;
    pub const R_ELBOW: usize = 19;
    pub const L_WRIST: usize = 20;
    pub const R_WRIST: usize = 21;
}

/// A (COCO keypoint, SMPL-X joint) correspondence.
const CORRESPONDENCES: [(usize, usize); 14] = [
    (kp::L_SHOULDER, sj::L_SHOULDER),
    (kp::R_SHOULDER, sj::R_SHOULDER),
    (kp::L_ELBOW, sj::L_ELBOW),
    (kp::R_ELBOW, sj::R_ELBOW),
    (kp::L_WRIST, sj::L_WRIST),
    (kp::R_WRIST, sj::R_WRIST),
    (kp::L_HIP, sj::L_HIP),
    (kp::R_HIP, sj::R_HIP),
    (kp::L_KNEE, sj::L_KNEE),
    (kp::R_KNEE, sj::R_KNEE),
    (kp::L_ANKLE, sj::L_ANKLE),
    (kp::R_ANKLE, sj::R_ANKLE),
    (kp::NOSE, sj::HEAD),
    (kp::NOSE, sj::NECK),
];

/// Which SMPL-X joints the optimiser is allowed to rotate.
const FREE_JOINTS: [usize; 14] = [
    sj::SPINE1,
    sj::SPINE2,
    sj::SPINE3,
    sj::L_SHOULDER,
    sj::R_SHOULDER,
    sj::L_ELBOW,
    sj::R_ELBOW,
    sj::L_HIP,
    sj::R_HIP,
    sj::L_KNEE,
    sj::R_KNEE,
    sj::NECK,
    sj::L_WRIST,
    sj::R_WRIST,
];

#[derive(Clone, Debug)]
pub struct FitConfig {
    /// Per-joint confidence below which a keypoint is ignored.
    pub min_confidence: f32,
    /// Maximum Levenberg-Marquardt iterations per frame.
    pub iterations: usize,
    /// Weight on the temporal smoothness term relative to reprojection.
    pub temporal_weight: f32,
    /// Weight on keeping joint rotations near zero (avoids wild limbs).
    pub prior_weight: f32,
    /// Assume a perspective camera when true, weak perspective otherwise.
    pub perspective: bool,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.3,
            iterations: 40,
            temporal_weight: 0.15,
            prior_weight: 0.02,
            perspective: false,
        }
    }
}

/// Result of fitting one frame.
#[derive(Clone, Debug)]
pub struct FittedFrame {
    pub pose: Pose,
    pub skeleton: Skeleton,
    /// Root-mean-square reprojection error in pixels.
    pub reprojection_error: f32,
    /// Number of keypoints that contributed.
    pub used_keypoints: usize,
    pub converged: bool,
}

/// Fits a sequence of detections into a coherent motion.
pub struct Fitter<'a> {
    model: &'a BodyModel,
    config: FitConfig,
    /// Pinhole focal length in pixels, derived from the subject size.
    focal: f32,
    /// Principal point (image centre).
    centre: (f32, f32),
}

impl<'a> Fitter<'a> {
    pub fn new(model: &'a BodyModel, width: usize, height: usize, config: FitConfig) -> Self {
        // A 50 mm-equivalent lens sees ~40 degrees; f ≈ 1.2 * image diagonal is
        // a reasonable default for phone and webcam footage.
        let diagonal = ((width * width + height * height) as f32).sqrt();
        Self {
            model,
            config,
            focal: diagonal * 1.2,
            centre: (width as f32 * 0.5, height as f32 * 0.5),
        }
    }

    /// Fit one frame, warm-starting from `previous` when available.
    pub fn fit(&self, detection: &PoseDetection, previous: Option<&Pose>) -> Result<FittedFrame> {
        let num_joints = self.model.num_joints;
        let num_betas = self.model.num_betas;

        // Collect observations, skipping low-confidence joints.
        let mut observations: Vec<(usize, f32, f32, f32)> = Vec::new();
        for (coco, joint) in CORRESPONDENCES {
            let point = detection.keypoints[coco];
            if !point.is_confident(self.config.min_confidence) {
                continue;
            }
            // The nose maps to two joints; weight both lower than a limb.
            let weight = if coco == kp::NOSE { 0.35 } else { point.confidence };
            observations.push((joint, point.x, point.y, weight));
        }
        // Fewer than 6 confident joints cannot constrain depth, scale and the
        // limb chain together; the solver would wander off (see the 16 m
        // pelvis in the degenerate case). Keep the previous pose instead.
        if observations.len() < 6 {
            // Without a previous pose there is nothing to fall back to, so
            // project the box centre at a plausible depth: a neutral body
            // standing where the detection is.
            let pose = match previous {
                Some(previous) => previous.clone(),
                None => {
                    let mut neutral = Pose::neutral(num_joints, num_betas);
                    let depth = (self.focal * 1.7) / detection.bbox.height().max(1.0);
                    let centre_x = (detection.bbox.x0 + detection.bbox.x1) * 0.5;
                    let centre_y = (detection.bbox.y0 + detection.bbox.y1) * 0.5;
                    neutral.translation = [
                        (centre_x - self.centre.0) * depth / self.focal,
                        -(centre_y - self.centre.1) * depth / self.focal,
                        depth,
                    ];
                    neutral
                }
            };
            let skeleton = self.model.forward_joints(&pose);
            return Ok(FittedFrame {
                pose,
                skeleton,
                reprojection_error: f32::INFINITY,
                used_keypoints: observations.len(),
                converged: false,
            });
        }

        // Parameters: [root(3), translation(3), free joint rotations(3 each)].
        let param_count = 6 + FREE_JOINTS.len() * 3;
        let mut params = vec![0.0f32; param_count];

        // Initialise translation from the bounding box so the body lands on the
        // person instead of the origin.
        let start = previous.cloned().unwrap_or_else(|| Pose::neutral(num_joints, num_betas));
        if previous.is_none() {
            let centre_x = (detection.bbox.x0 + detection.bbox.x1) * 0.5;
            let centre_y = (detection.bbox.y0 + detection.bbox.y1) * 0.5;
            // A 1.7 m person spanning `height` pixels sits at f * 1.7 / height.
            let depth = (self.focal * 1.7) / detection.bbox.height().max(1.0);
            params[3] = (centre_x - self.centre.0) * depth / self.focal;
            params[4] = -(centre_y - self.centre.1) * depth / self.focal;
            params[5] = depth;
        } else {
            params[0..3].copy_from_slice(&start.root);
            params[3..6].copy_from_slice(&start.translation);
            for (slot, joint) in FREE_JOINTS.iter().enumerate() {
                let rot = start.joint(*joint);
                params[6 + slot * 3..6 + slot * 3 + 3].copy_from_slice(&rot);
            }
        }

        let residual_count = observations.len() * 2
            + if self.config.prior_weight > 0.0 { FREE_JOINTS.len() * 3 } else { 0 };
        let mut residuals = vec![0.0f32; residual_count];

        let mut lambda = 1e-3f32;
        let mut best_cost = f32::INFINITY;
        let mut converged = false;

        for _ in 0..self.config.iterations {
            let pose = self.build_pose(&params);
            let skeleton = self.model.forward_joints(&pose);
            self.residuals(&params, &skeleton, &observations, &mut residuals);
            let cost: f32 = residuals.iter().map(|r| r * r).sum();

            // Numerical Jacobian: the problem is small enough that this is both
            // simpler and fast (one forward pass per parameter, ~30 us each).
            let mut jacobian = vec![0.0f32; residual_count * param_count];
            let step = 1e-3f32;
            for p in 0..param_count {
                let saved = params[p];
                params[p] = saved + step;
                let probe = self.build_pose(&params);
                let probe_skeleton = self.model.forward_joints(&probe);
                let mut probe_residuals = vec![0.0f32; residual_count];
                self.residuals(&params, &probe_skeleton, &observations, &mut probe_residuals);
                params[p] = saved;
                for r in 0..residual_count {
                    jacobian[r * param_count + p] =
                        (probe_residuals[r] - residuals[r]) / step;
                }
            }

            // Normal equations with Levenberg damping.
            let mut normal = vec![0.0f32; param_count * param_count];
            let mut gradient = vec![0.0f32; param_count];
            for r in 0..residual_count {
                let row = &jacobian[r * param_count..(r + 1) * param_count];
                let value = residuals[r];
                for i in 0..param_count {
                    gradient[i] += row[i] * value;
                    for j in i..param_count {
                        normal[i * param_count + j] += row[i] * row[j];
                    }
                }
            }
            for i in 0..param_count {
                for j in 0..i {
                    normal[i * param_count + j] = normal[j * param_count + i];
                }
                normal[i * param_count + i] += lambda;
            }

            let Some(delta) = solve_symmetric(&mut normal.clone(), &gradient, param_count) else {
                lambda *= 10.0;
                if lambda > 1e6 {
                    break;
                }
                continue;
            };

            let mut candidate = params.clone();
            for i in 0..param_count {
                candidate[i] -= delta[i];
            }
            let candidate_pose = self.build_pose(&candidate);
            let candidate_skeleton = self.model.forward_joints(&candidate_pose);
            let mut candidate_residuals = vec![0.0f32; residual_count];
            self.residuals(&candidate, &candidate_skeleton, &observations, &mut candidate_residuals);
            let candidate_cost: f32 = candidate_residuals.iter().map(|r| r * r).sum();

            if candidate_cost < cost {
                params = candidate;
                lambda = (lambda * 0.4).max(1e-7);
                if (best_cost - candidate_cost).abs() < 1e-7 * best_cost.max(1.0) {
                    best_cost = candidate_cost;
                    converged = true;
                    break;
                }
                best_cost = candidate_cost;
            } else {
                lambda *= 4.0;
                if lambda > 1e6 {
                    break;
                }
            }
        }

        let mut pose = self.build_pose(&params);
        // Reject solutions that put the subject at an absurd distance: the
        // depth range of a single-camera body fit is roughly 0.8-20 m.
        if !(0.5..=12.0).contains(&pose.translation[2]) {
            let mut fallback = match previous {
                Some(previous) => previous.clone(),
                None => Pose::neutral(num_joints, num_betas),
            };
            let depth = (self.focal * 1.7) / detection.bbox.height().max(1.0);
            let centre_x = (detection.bbox.x0 + detection.bbox.x1) * 0.5;
            let centre_y = (detection.bbox.y0 + detection.bbox.y1) * 0.5;
            fallback.translation = [
                (centre_x - self.centre.0) * depth / self.focal,
                -(centre_y - self.centre.1) * depth / self.focal,
                depth,
            ];
            pose = fallback;
        }
        let skeleton = self.model.forward_joints(&pose);
        self.residuals(&params, &skeleton, &observations, &mut residuals);
        let reprojection = (residuals
            .iter()
            .take(observations.len() * 2)
            .map(|r| r * r)
            .sum::<f32>()
            / (observations.len() * 2) as f32)
            .sqrt();

        // A large residual with few observations means the fit is not
        // trustworthy (occlusion, bad keypoints). Fall back to the previous
        // pose so one bad frame does not jump the whole motion.
        let unreliable = reprojection > 25.0 && observations.len() < 12;
        if unreliable {
            if let Some(previous) = previous {
                let skeleton = self.model.forward_joints(previous);
                return Ok(FittedFrame {
                    pose: previous.clone(),
                    skeleton,
                    reprojection_error: reprojection,
                    used_keypoints: observations.len(),
                    converged: false,
                });
            }
        }

        Ok(FittedFrame {
            pose,
            skeleton,
            reprojection_error: reprojection,
            used_keypoints: observations.len(),
            converged,
        })
    }

    fn build_pose(&self, params: &[f32]) -> Pose {
        let mut pose = Pose::neutral(self.model.num_joints, self.model.num_betas);
        pose.root = [params[0], params[1], params[2]];
        pose.translation = [params[3], params[4], params[5]];
        for (slot, joint) in FREE_JOINTS.iter().enumerate() {
            let base = 6 + slot * 3;
            pose.set_joint(*joint, [params[base], params[base + 1], params[base + 2]]);
        }
        pose
    }

    /// Project a 3D joint to pixels.
    ///
    /// `point` is already in camera space (the root translation puts the body
    /// at its true distance), so a real perspective divide is required: without
    /// it, depth and body scale are unidentifiable and the optimiser trades one
    /// against the other, leaving tens of pixels of error.
    fn project(&self, point: [f32; 3]) -> (f32, f32) {
        let z = point[2].max(0.2);
        (
            self.focal * point[0] / z + self.centre.0,
            self.centre.1 - self.focal * point[1] / z,
        )
    }

    fn residuals(
        &self,
        params: &[f32],
        skeleton: &Skeleton,
        observations: &[(usize, f32, f32, f32)],
        out: &mut [f32],
    ) {
        // Body joints are in world space with y up; the camera looks down +Z
        // from the origin, so convert (x, y, z) -> (x, y, depth + z).
        let depth = params[5];
        for (i, (joint, u, v, weight)) in observations.iter().enumerate() {
            let world = skeleton.joint(*joint);
            let camera = [world[0], world[1], world[2] + depth];
            let projected = self.project(camera);
            out[i * 2] = (projected.0 - u) * weight;
            out[i * 2 + 1] = (projected.1 - v) * weight;
        }
        let offset = observations.len() * 2;
        if self.config.prior_weight > 0.0 {
            for i in 0..FREE_JOINTS.len() * 3 {
                out[offset + i] = params[6 + i] * self.config.prior_weight;
            }
        }
    }
}

/// Gaussian elimination with partial pivoting for a small dense system.
fn solve_symmetric(matrix: &mut [f32], rhs: &[f32], n: usize) -> Option<Vec<f32>> {
    let mut b = rhs.to_vec();
    for col in 0..n {
        // Pivot on the largest magnitude in this column.
        let mut pivot = col;
        for row in col + 1..n {
            if matrix[row * n + col].abs() > matrix[pivot * n + col].abs() {
                pivot = row;
            }
        }
        if matrix[pivot * n + col].abs() < 1e-12 {
            return None;
        }
        if pivot != col {
            for k in 0..n {
                matrix.swap(col * n + k, pivot * n + k);
            }
            b.swap(col, pivot);
        }
        let diagonal = matrix[col * n + col];
        for row in col + 1..n {
            let factor = matrix[row * n + col] / diagonal;
            if factor == 0.0 {
                continue;
            }
            for k in col..n {
                matrix[row * n + k] -= factor * matrix[col * n + k];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = vec![0.0f32; n];
    for row in (0..n).rev() {
        let mut sum = b[row];
        for k in row + 1..n {
            sum -= matrix[row * n + k] * x[k];
        }
        x[row] = sum / matrix[row * n + row];
    }
    Some(x)
}

/// Fit a whole clip, carrying the previous pose forward as the warm start.
pub fn fit_sequence(
    model: &BodyModel,
    detections: &[Option<PoseDetection>],
    width: usize,
    height: usize,
    config: FitConfig,
) -> Result<Vec<FittedFrame>> {
    let fitter = Fitter::new(model, width, height, config);
    let mut out = Vec::with_capacity(detections.len());
    let mut previous: Option<Pose> = None;
    for detection in detections {
        match detection {
            Some(detection) => {
                let fitted = fitter.fit(detection, previous.as_ref())?;
                previous = Some(fitted.pose.clone());
                out.push(fitted);
            }
            None => {
                // Coast: repeat the last pose so the timeline stays aligned.
                let pose = previous
                    .clone()
                    .unwrap_or_else(|| Pose::neutral(model.num_joints, model.num_betas));
                let skeleton = model.forward_joints(&pose);
                out.push(FittedFrame {
                    pose,
                    skeleton,
                    reprojection_error: f32::INFINITY,
                    used_keypoints: 0,
                    converged: false,
                });
            }
        }
    }
    Ok(out)
}

/// Map a fitted pose onto a keypoint set for a quick quality readout.
pub fn summarise(frames: &[FittedFrame]) -> String {
    let valid: Vec<&FittedFrame> = frames
        .iter()
        .filter(|f| f.reprojection_error.is_finite())
        .collect();
    if valid.is_empty() {
        return "no fitted frames".to_string();
    }
    let mean = valid.iter().map(|f| f.reprojection_error).sum::<f32>() / valid.len() as f32;
    let worst = valid
        .iter()
        .map(|f| f.reprojection_error)
        .fold(0.0f32, f32::max);
    let converged = valid.iter().filter(|f| f.converged).count();
    format!(
        "{} frames, mean reprojection {:.1} px, worst {:.1} px, {}/{} converged",
        valid.len(),
        mean,
        worst,
        converged,
        valid.len()
    )
}

/// Helper for tests and diagnostics: pick the highest-confidence keypoint set.
pub fn primary(detections: &[PoseDetection]) -> Option<&PoseDetection> {
    detections.iter().max_by(|a, b| {
        a.mean_keypoint_confidence()
            .partial_cmp(&b.mean_keypoint_confidence())
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solver_handles_a_simple_system() {
        // 2x + y = 5, x + 3y = 10  ->  x = 1, y = 3
        let mut m = vec![2.0f32, 1.0, 1.0, 3.0];
        let b = vec![5.0f32, 10.0];
        let x = solve_symmetric(&mut m, &b, 2).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-4, "{x:?}");
        assert!((x[1] - 3.0).abs() < 1e-4, "{x:?}");
    }

    #[test]
    fn solver_rejects_a_singular_system() {
        let mut m = vec![1.0f32, 2.0, 2.0, 4.0];
        let b = vec![1.0f32, 2.0];
        assert!(solve_symmetric(&mut m, &b, 2).is_none());
    }
}
