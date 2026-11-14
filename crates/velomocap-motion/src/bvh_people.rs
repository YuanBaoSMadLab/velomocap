//! BVH export for the MHR skeleton.
//!
//! The SAM chain reports, per frame, 127 joint translations plus the *global*
//! joint quaternions (`skel_state[j] = [x, y, z, qx, qy, qz, qw, _]`), and the
//! checkpoint carries the matching parent list and rest offsets. That is exactly
//! what a BVH needs, so export is a basis change and a quaternion divide:
//!
//! * local rotation = `conj(global[parent]) * global[joint]`
//! * the world flips `y` and `z` against MHR, i.e. a 180° rotation about `x`,
//!   which conjugates a quaternion to `(x, -y, -z, w)`.
//!
//! Reusing the RGB MoCap BVH writer keeps the Euler-order scoring and the file
//! layout identical to the original application.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use velomocap_core::bvh::{write_bvh, BvhReport};
use velomocap_core::rig::Binding;
use velomocap_infer::MeshTopology;

use crate::sam::{PersonFrame, PersonTrack};

/// Write one person's motion as BVH.
pub fn write_person_bvh(
    path: &Path,
    person: &PersonTrack,
    mesh: &MeshTopology,
    fps: f64,
) -> Result<BvhReport> {
    let joints = mesh.parents.len();
    anyhow::ensure!(joints > 1, "the MHR skeleton has no bones");
    let frames = person.frames.len();
    anyhow::ensure!(frames > 0, "{} has no frames", person.name);

    // Rest pose in world space: accumulate the parent-relative offsets.
    let mut rest = vec![0.0f64; joints * 3];
    for joint in 0..joints {
        let offset = mesh.rest_offsets[joint];
        let parent = mesh.parents[joint];
        for axis in 0..3 {
            let base = if parent < 0 {
                0.0
            } else {
                rest[parent as usize * 3 + axis]
            };
            rest[joint * 3 + axis] = base + offset[axis] as f64;
        }
    }

    let names: Vec<String> = (0..joints).map(|joint| format!("mhr{joint:03}")).collect();
    let ends = rest.clone();
    let mut basis = vec![0.0f64; joints * 9];
    for joint in 0..joints {
        basis[joint * 9] = 1.0;
        basis[joint * 9 + 4] = 1.0;
        basis[joint * 9 + 8] = 1.0;
    }
    let binding = Binding {
        names,
        parents: mesh.parents.clone(),
        rest,
        ends,
        basis,
        origin: [0.0, 0.0, 0.0],
        version: 1,
    };

    let mut roots = Vec::with_capacity(frames * 3);
    let mut local = vec![0.0f64; frames * joints * 9];
    let mut previous: Option<&Arc<PersonFrame>> = None;
    for (index, slot) in person.frames.iter().enumerate() {
        let frame = slot.as_ref().or(previous).with_context(|| {
            format!("{} has no frame at or before {index}", person.name)
        })?;
        previous = Some(frame);
        let frame: &PersonFrame = frame;
        anyhow::ensure!(
            frame.skel_state.len() >= joints * 8,
            "frame {index} carries {} skel_state values, expected {}",
            frame.skel_state.len(),
            joints * 8
        );

        // Root translation in world metres.
        for axis in 0..3 {
            roots.push(frame.joints[axis] as f64);
        }

        let quaternion = |joint: usize| -> [f64; 4] {
            let at = joint * 8 + 3;
            let (x, y, z, w) = (
                frame.skel_state[at] as f64,
                frame.skel_state[at + 1] as f64,
                frame.skel_state[at + 2] as f64,
                frame.skel_state[at + 3] as f64,
            );
            // y/z flip of the world against MHR.
            [x, -y, -z, w]
        };
        let globals: Vec<[f64; 4]> = (0..joints).map(quaternion).collect();
        for joint in 0..joints {
            let parent = mesh.parents[joint];
            let local_q = if parent < 0 {
                globals[joint]
            } else {
                multiply(conjugate(globals[parent as usize]), globals[joint])
            };
            let matrix = to_matrix(local_q);
            let at = (index * joints + joint) * 9;
            local[at..at + 9].copy_from_slice(&matrix);
        }
    }

    binding.validate().context("validating the MHR export skeleton")?;
    write_bvh(path, &binding, &roots, &local, fps, 1)
        .with_context(|| format!("writing {}", path.display()))
}

fn conjugate(q: [f64; 4]) -> [f64; 4] {
    [-q[0], -q[1], -q[2], q[3]]
}

fn multiply(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ]
}

/// Row-major rotation matrix of a quaternion.
fn to_matrix(q: [f64; 4]) -> [f64; 9] {
    let length = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    let scale = if length > 1e-12 { 1.0 / length } else { 1.0 };
    let (x, y, z, w) = (q[0] * scale, q[1] * scale, q[2] * scale, q[3] * scale);
    [
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - z * w),
        2.0 * (x * z + y * w),
        2.0 * (x * y + z * w),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z - x * w),
        2.0 * (x * z - y * w),
        2.0 * (y * z + x * w),
        1.0 - 2.0 * (x * x + y * y),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn identity_quaternions_give_identity_matrices() {
        let matrix = to_matrix([0.0, 0.0, 0.0, 1.0]);
        assert_eq!(matrix, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let half = std::f64::consts::FRAC_1_SQRT_2;
        let turned = to_matrix([0.0, half, 0.0, half]);
        // 90 degrees about y: +x maps to +z, +z maps to -x.
        assert!(turned[0].abs() < 1e-9);
        assert!((turned[2] - 1.0).abs() < 1e-9);
        assert!((turned[6] + 1.0).abs() < 1e-9);
    }

    #[test]
    fn writes_a_bvh_for_a_synthetic_person() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(std::path::Path::parent)
            .expect("workspace root");
        let mesh = match MeshTopology::load(root.join("models/sam3d")) {
            Ok(mesh) => mesh,
            Err(_) => {
                eprintln!("skipping: no mesh topology");
                return;
            }
        };
        let joints = mesh.parents.len();
        let mut skel_state = vec![0.0f32; joints * 8];
        for joint in 0..joints {
            skel_state[joint * 8 + 6] = 1.0; // w = 1: identity
        }
        let frame = PersonFrame {
            bbox: [0.0; 4],
            score: 1.0,
            verts: vec![0.0; 3],
            joints: vec![0.0; joints * 3],
            keypoints: vec![0.0; 70 * 3],
            body_token: vec![0.0; 1024],
            left_token: Vec::new(),
            right_token: Vec::new(),
            cam_t: [0.0, 0.0, 5.0],
            anchor_cam: [0.0, 0.0, 5.0],
            skel_state,
        };
        let person = PersonTrack {
            id: 0,
            name: "Test".into(),
            frames: vec![Some(Arc::new(frame))],
            first_frame: 0,
            last_frame: 0,
            hits: 1,
        };
        let path = std::env::temp_dir().join("velomocap-test.bvh");
        let report = write_person_bvh(&path, &person, &mesh, 30.0).expect("bvh");
        let text = std::fs::read_to_string(&path).expect("read");
        assert!(text.starts_with("HIERARCHY"));
        assert!(text.contains("MOTION"));
        assert!(text.contains(&format!("Frames: 1")));
        eprintln!(
            "bvh {} bytes, {} bones, order {}",
            text.len(),
            joints,
            report.orders.first().cloned().unwrap_or_default()
        );
        let _ = std::fs::remove_file(path);
    }
}
