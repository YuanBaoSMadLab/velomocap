//! BVH export: fixed Euler order selection plus streaming text output.
//!
//! Port of `mocap/source_bvh.py`.  The expensive parts are (a) choosing a
//! singularity-free Euler order per bone by scoring all six permutations and
//! (b) formatting ~160 numbers per frame; both are vectorised across bones
//! with rayon and written through a single buffered writer.

use std::fmt::Write as _;
use std::path::Path;

use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::math::{self, mat_to_euler, mul3, transpose3, vec3};
use crate::rig::{local_frames, Binding};

/// The six Tait-Bryan orders scored for every bone.
const ORDERS: [&str; 6] = ["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"];

/// Wrist bones whose channel order is scored against interpolated poses.
const WRIST_BONES: [usize; 2] = [20, 21];

#[derive(Clone, Debug, Default)]
pub struct BvhReport {
    pub orders: Vec<String>,
    pub minimum_singularity_margin: f64,
}

/// Write a BVH hierarchy and motion block for a committed take.
pub fn write_bvh(
    path: &Path,
    binding: &Binding,
    roots: &[f64],
    local: &[f64],
    fps: f64,
    version: i32,
) -> Result<BvhReport> {
    write_bvh_with_orders(path, binding, roots, local, fps, version, None)
}

/// Like [`write_bvh`], but pins each bone's Euler channel order.
///
/// Re-exporting a take that already has committed `orders` (for example from a
/// saved project) reproduces the original encoding exactly, even where two
/// orders score within a rounding error of each other.
pub fn write_bvh_with_orders(
    path: &Path,
    binding: &Binding,
    roots: &[f64],
    local: &[f64],
    fps: f64,
    version: i32,
    orders_override: Option<&[String]>,
) -> Result<BvhReport> {
    binding.validate()?;
    let bones = binding.bones();
    if roots.len() % 3 != 0 {
        return Err(Error::new("Invalid BVH root dimensions"));
    }
    let frames = roots.len() / 3;
    if frames == 0 {
        return Err(Error::new("Invalid BVH animation dimensions"));
    }
    if local.len() != frames * bones * 9 {
        return Err(Error::new("Invalid BVH local frame dimensions"));
    }
    if !fps.is_finite() || fps <= 0.0 {
        return Err(Error::new("Invalid BVH frame rate"));
    }
    if !roots.iter().all(|v| v.is_finite()) || !local.iter().all(|v| v.is_finite()) {
        return Err(Error::new("Non-finite BVH animation"));
    }

    let rotations = local_frames(local, binding, frames);

    if let Some(orders) = orders_override {
        if orders.len() != bones {
            return Err(Error::new(format!(
                "Channel order override has {} entries, expected {bones}",
                orders.len()
            )));
        }
    }

    let selections: Vec<(Vec<f64>, &'static str, f64)> = (0..bones)
        .into_par_iter()
        .map(|j| {
            let pinned = orders_override.and_then(|orders| {
                ORDERS.iter().copied().find(|candidate| **candidate == orders[j])
            });
            select_channels(j, binding, &rotations, frames, version, pinned)
        })
        .collect::<Result<Vec<_>>>()?;

    let mut angles = vec![0.0; frames * bones * 3];
    let mut orders = Vec::with_capacity(bones);
    let mut minimum_margin = f64::INFINITY;
    for (j, (bone_angles, order, margin)) in selections.iter().enumerate() {
        for t in 0..frames {
            angles[(t * bones + j) * 3..(t * bones + j) * 3 + 3]
                .copy_from_slice(&bone_angles[t * 3..t * 3 + 3]);
        }
        orders.push((*order).to_string());
        minimum_margin = minimum_margin.min(*margin);
    }

    write_channels(path, binding, roots, &angles, &orders, fps)?;

    Ok(BvhReport {
        orders,
        minimum_singularity_margin: minimum_margin,
    })
}

/// Score every Euler order for one bone and return the committed track.
fn select_channels(
    bone: usize,
    binding: &Binding,
    rotations: &[f64],
    frames: usize,
    version: i32,
    pinned: Option<&'static str>,
) -> Result<(Vec<f64>, &'static str, f64)> {
    let bones = binding.bones();
    let mut bone_rot = vec![0.0; frames * 9];
    for t in 0..frames {
        let src = (t * bones + bone) * 9;
        bone_rot[t * 9..t * 9 + 9].copy_from_slice(&rotations[src..src + 9]);
    }

    // Reference frame from the bone's outgoing direction, as Blender builds it.
    let mut delta = [0.0f64; 3];
    let children: Vec<usize> = (0..bones)
        .filter(|&i| binding.parents[i] == bone as i32)
        .collect();
    if children.is_empty() {
        delta = vec3(&binding.ends, bone);
    } else {
        for child in &children {
            for axis in 0..3 {
                delta[axis] += binding.rest[child * 3 + axis];
            }
        }
        for axis in 0..3 {
            delta[axis] = delta[axis] / children.len() as f64 - binding.rest[bone * 3 + axis];
        }
    }
    if math::norm3(&delta) < 0.001 {
        delta[1] += 0.1;
    }
    let reference = math::reference_frame(&delta);
    let reference_t = transpose3(&reference);

    // pose = D^T @ R @ D
    let mut pose = vec![0.0; frames * 9];
    for t in 0..frames {
        let rotated = mul3(&bone_rot[t * 9..t * 9 + 9], &reference);
        let posed = mul3(&reference_t, &rotated);
        pose[t * 9..t * 9 + 9].copy_from_slice(&posed);
    }

    let mut options: Vec<(f64, &'static str, Vec<f64>)> = Vec::with_capacity(ORDERS.len());
    for order in ORDERS {
        let (track, margin_a) = mat_to_euler(&bone_rot, order);
        let (_, margin_b) = mat_to_euler(&pose, &order.to_ascii_lowercase());
        let margin = margin_a
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min)
            .min(margin_b.iter().cloned().fold(f64::INFINITY, f64::min));
        options.push((margin, order, track));
    }

    let override_order = if version >= 17 && WRIST_BONES.contains(&bone) {
        wrist_order(&pose)
    } else {
        None
    };

    // Ties keep the first order in `ORDERS`, matching Python's `max(...)`.
    let override_order = pinned.or(override_order);
    let (margin, order, mut track) = if let Some(wanted) = override_order {
        options
            .iter()
            .find(|(_, order, _)| *order == wanted)
            .map(|(m, o, t)| (*m, *o, t.clone()))
            .ok_or_else(|| Error::new("Wrist order selection failed"))?
    } else {
        let mut best: Option<(f64, &'static str, Vec<f64>)> = None;
        for candidate in options {
            let better = match &best {
                None => true,
                Some((value, _, _)) => candidate.0 > *value,
            };
            if better {
                best = Some(candidate);
            }
        }
        best.ok_or_else(|| Error::new("Euler order selection failed"))?
    };

    math::continuous_angles(&mut track);

    // Guard: the chosen encoding must reproduce the committed rotation exactly.
    let rebuilt = math::euler_to_mat(&track, order);
    let mut worst: f64 = 0.0;
    for t in 0..frames {
        for k in 0..9 {
            let diff = (rebuilt[t * 9 + k] - bone_rot[t * 9 + k]).abs();
            if diff > worst {
                worst = diff;
            }
        }
    }
    if worst > 1e-5 {
        return Err(Error::new(format!(
            "Euler encoding error {worst:.3e} on bone {bone} ({order})"
        )));
    }

    for value in track.iter_mut() {
        *value = value.to_degrees();
    }
    Ok((track, order, margin))
}

/// Score fixed wrist orders in Blender's bone basis at fractional frames.
///
/// SLERP is only an interpolation reference, never ground truth: the keyed
/// matrices stay exact, this just picks the encoding with the least
/// interpolation error and no singularity.
fn wrist_order(pose: &[f64]) -> Option<&'static str> {
    let frames = pose.len() / 9;
    if frames < 2 {
        return None;
    }

    // relative[t] = rotvec(pose[t]^T @ pose[t+1])
    let mut relative = vec![0.0; (frames - 1) * 9];
    for t in 0..frames - 1 {
        let a = &pose[t * 9..t * 9 + 9];
        let b = &pose[(t + 1) * 9..(t + 1) * 9 + 9];
        let delta = math::tmul3(a, b);
        relative[t * 9..t * 9 + 9].copy_from_slice(&delta);
    }
    let mut rotvec = vec![0.0; (frames - 1) * 3];
    crate::cpp::mat_to_rotvec(&relative, &mut rotvec);

    let weights = [0.25f64, 0.5, 0.75];
    let mut references: Vec<Vec<f64>> = Vec::with_capacity(weights.len());
    for &w in &weights {
        let scaled: Vec<f64> = rotvec.iter().map(|v| v * w).collect();
        let mut mats = vec![0.0; (frames - 1) * 9];
        crate::cpp::rotvec_to_mat(&scaled, &mut mats);
        let mut reference = vec![0.0; (frames - 1) * 9];
        for t in 0..frames - 1 {
            let composed = mul3(&pose[t * 9..t * 9 + 9], &mats[t * 9..t * 9 + 9]);
            reference[t * 9..t * 9 + 9].copy_from_slice(&composed);
        }
        references.push(reference);
    }

    let mut options: Vec<(f64, f64, &'static str)> = Vec::with_capacity(ORDERS.len());
    for order in ORDERS {
        let lowered = order.to_ascii_lowercase();
        let (mut track, _) = mat_to_euler(pose, &lowered);
        math::continuous_angles(&mut track);
        let mut errors: Vec<f64> = Vec::with_capacity((frames - 1) * weights.len());
        for (w, reference) in weights.iter().zip(references.iter()) {
            let mut blended = vec![0.0; (frames - 1) * 3];
            for t in 0..frames - 1 {
                for k in 0..3 {
                    blended[t * 3 + k] =
                        (1.0 - w) * track[t * 3 + k] + w * track[(t + 1) * 3 + k];
                }
            }
            let interpolated = math::euler_to_mat(&blended, &lowered);
            let mut magnitudes = vec![0.0; frames - 1];
            crate::cpp::relative_magnitude(&interpolated, reference, &mut magnitudes);
            errors.extend_from_slice(&magnitudes);
        }
        if errors.is_empty() {
            continue;
        }
        let maximum = errors.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean = errors.iter().sum::<f64>() / errors.len() as f64;
        let margin = track
            .chunks_exact(3)
            .map(|a| a[1].cos().abs())
            .fold(f64::INFINITY, f64::min);
        options.push((margin, maximum + 0.05 * mean, order));
    }
    if options.is_empty() {
        return None;
    }
    // Ties keep the first order, matching Python's `min(...)`.
    let admissible: Vec<&(f64, f64, &'static str)> =
        options.iter().filter(|o| o.0 > 0.02).collect();
    let pool = if admissible.is_empty() { &options.iter().collect::<Vec<_>>() } else { &admissible };
    let mut best: Option<&(f64, f64, &'static str)> = None;
    for candidate in pool.iter().copied() {
        let better = match best {
            None => true,
            Some(current) => candidate.1 < current.1,
        };
        if better {
            best = Some(candidate);
        }
    }
    Some(best?.2)
}

fn depth_first(parents: &[i32]) -> Vec<usize> {
    let mut order = Vec::with_capacity(parents.len());
    let mut stack: Vec<usize> = Vec::new();
    // Push children in reverse so the traversal visits them in ascending order.
    stack.push(0);
    while let Some(j) = stack.pop() {
        order.push(j);
        let mut children: Vec<usize> = (0..parents.len())
            .filter(|&i| parents[i] == j as i32)
            .collect();
        children.reverse();
        stack.extend(children);
    }
    order
}

fn push_number(out: &mut String, value: f64, precision: usize) {
    // Fixed-point formatting with the same rounding as Python's f-string.
    let _ = write!(out, "{value:.precision$}");
}

fn write_channels(
    path: &Path,
    binding: &Binding,
    roots: &[f64],
    angles: &[f64],
    orders: &[String],
    fps: f64,
) -> Result<()> {
    let bones = binding.bones();
    let frames = roots.len() / 3;
    let offsets = binding.offsets();
    let traversal = depth_first(&binding.parents);
    debug_assert_eq!(traversal.len(), bones);

    let mut text = String::with_capacity(frames * bones * 3 * 13 + 8192);
    text.push_str("HIERARCHY\n");

    fn emit(
        text: &mut String,
        j: usize,
        depth: usize,
        binding: &Binding,
        orders: &[String],
        offsets: &[[f64; 3]],
    ) {
        let pad = "  ".repeat(depth);
        let parent = binding.parents[j];
        let keyword = if parent < 0 { "ROOT " } else { "JOINT " };
        text.push_str(&pad);
        text.push_str(keyword);
        text.push_str(&binding.names[j]);
        text.push('\n');
        text.push_str(&pad);
        text.push_str("{\n");
        text.push_str(&pad);
        text.push_str("  OFFSET ");
        for (axis, value) in offsets[j].iter().enumerate() {
            if axis > 0 {
                text.push(' ');
            }
            push_number(text, *value, 9);
        }
        text.push('\n');
        text.push_str(&pad);
        if parent < 0 {
            text.push_str("  CHANNELS 6 Xposition Yposition Zposition ");
        } else {
            text.push_str("  CHANNELS 3 ");
        }
        let order = &orders[j];
        for (i, axis) in order.chars().enumerate() {
            if i > 0 {
                text.push(' ');
            }
            text.push(axis);
            text.push_str("rotation");
        }
        text.push('\n');
        let children: Vec<usize> = (0..binding.parents.len())
            .filter(|&i| binding.parents[i] == j as i32)
            .collect();
        for child in children {
            emit(text, child, depth + 1, binding, orders, offsets);
        }
        if binding.parents.iter().all(|&p| p != j as i32) {
            text.push_str(&pad);
            text.push_str("  End Site\n");
            text.push_str(&pad);
            text.push_str("  {\n");
            text.push_str(&pad);
            text.push_str("    OFFSET ");
            for axis in 0..3 {
                if axis > 0 {
                    text.push(' ');
                }
                push_number(text, binding.ends[j * 3 + axis], 9);
            }
            text.push('\n');
            text.push_str(&pad);
            text.push_str("  }\n");
        }
        text.push_str(&pad);
        text.push_str("}\n");
    }

    emit(&mut text, 0, 0, binding, orders, &offsets);

    text.push_str("MOTION\n");
    let _ = writeln!(text, "Frames: {frames}");
    let _ = writeln!(text, "Frame Time: {:.10}", 1.0 / fps);

    for t in 0..frames {
        let mut first = true;
        for axis in 0..3 {
            if !first {
                text.push(' ');
            }
            first = false;
            push_number(&mut text, roots[t * 3 + axis], 9);
        }
        for &j in &traversal {
            for axis in 0..3 {
                text.push(' ');
                push_number(&mut text, angles[(t * bones + j) * 3 + axis], 9);
            }
        }
        text.push('\n');
    }

    std::fs::write(path, text.as_bytes())?;
    Ok(())
}
