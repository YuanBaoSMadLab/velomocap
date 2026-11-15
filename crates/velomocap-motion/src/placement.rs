//! Metric WV placement, ported from `flow_placement.py` (RGB MoCap, MIT).
//!
//! FlowHMR predicts a body in a y-up world whose *heading* is whatever the
//! camera's own vertical basis says it is; two takes of the same scene, or two
//! people in one take, therefore live in two different world-vertical frames. The
//! reference resolves this in three parts, and this module is that resolution:
//!
//! ```text
//! wv(R)   = [ x , up , z ] with z = Rᵀ·[0,0,1], x = up × z       take's WV basis
//! q       = wv(scene₀) · wv(take₀)ᵀ                              transport into the scene
//! offset  = median( camᵀ·anchor − q·pelvis ), cam = R·wv(scene₀)ᵀ
//! height  = −offset.y                                           camera height
//! offset.y = 0                                                   keep native height
//! p'      = q·p + offset                     (a yaw about the pelvis, no scale)
//! ```
//!
//! The placement is **rigid**: one rotation, one constant offset. It is not
//! metric moving-camera translation and not an interaction fit — FlowHMR's own
//! height is retained, and only the median residual against the SAM anchors is
//! removed. A take on its own (`scene₀ == take₀`) has `q = I`, so the rotation
//! only ever does work when several takes are placed into one scene.
//!
//! Because `offset.y` is dropped, the body's vertical datum stays FlowHMR's, and
//! the *floor* is what is measured afterwards (`crate::floor`). That split is the
//! reference's: placement never decides where the ground is.

use velomocap_body::{BodyModel, Pose, Skeleton};

/// The frames the SAM anchors are taken from (`flow_placement.placement_fit`).
pub const ANCHOR_FRAMES: usize = 30;

/// How far the anchors are allowed to sit from the placed pelvis before the
/// placement is considered meaningless.
pub const ANCHOR_TOLERANCE: f64 = 2e-4;

/// One take's placement.
#[derive(Clone, Copy, Debug)]
pub struct Fit {
    /// `q`: the take's WV heading transported into the scene's.
    pub rotation: [[f64; 3]; 3],
    /// Constant offset, with `y` dropped (the native height is retained).
    pub offset: [f64; 3],
    /// `−median(Δy)`: where the camera sits above the FlowHMR datum.
    pub camera_height: f64,
    /// The scene's own WV basis.
    pub scene_basis: [[f64; 3]; 3],
    /// How many SAM anchors the offset was fitted to.
    pub anchors: usize,
    /// Mean and worst distance between the placed pelvis and its anchor.
    pub residual_mean: f64,
    pub residual_max: f64,
}

impl Fit {
    /// The identity placement, for a clip with no camera to derive a basis from.
    pub fn identity() -> Self {
        Self {
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            offset: [0.0; 3],
            camera_height: 0.0,
            scene_basis: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            anchors: 0,
            residual_mean: 0.0,
            residual_max: 0.0,
        }
    }

    /// Rotate a world point by `q` and add the offset.
    pub fn point(&self, p: [f64; 3]) -> [f64; 3] {
        let r = rotate(self.rotation, p);
        [
            r[0] + self.offset[0],
            r[1] + self.offset[1],
            r[2] + self.offset[2],
        ]
    }
}

/// `np.stack((x, [0,1,0], z))` — the y-up basis of a camera rotation.
///
/// `z` is the direction the camera looks along (`Rᵀ·ẑ`) and `x` its horizontal
/// right (`up × z`), so the matrix maps take coordinates into a frame whose `y`
/// is world up and whose `z` is the camera's forward axis. A vertical camera has
/// no ground heading at all, and is rejected rather than turned into an
/// arbitrary one.
pub fn wv_rotation(rotation: &[f32; 9]) -> Option<[[f64; 3]; 3]> {
    let r = [
        [rotation[0] as f64, rotation[1] as f64, rotation[2] as f64],
        [rotation[3] as f64, rotation[4] as f64, rotation[5] as f64],
        [rotation[6] as f64, rotation[7] as f64, rotation[8] as f64],
    ];
    if !is_rotation(&r) {
        return None;
    }
    // z = Rᵀ·[0,0,1]: the third row of R.
    let z = r[2];
    let x = [z[2], 0.0, -z[0]];
    let length = (x[0] * x[0] + x[2] * x[2]).sqrt();
    if length < 1e-5 {
        return None;
    }
    let x = [x[0] / length, 0.0, x[2] / length];
    let z = [-x[2], 0.0, x[0]];
    let length = (z[0] * z[0] + z[2] * z[2]).sqrt();
    if length < 1e-5 {
        return None;
    }
    Some([
        x,
        [0.0, 1.0, 0.0],
        [z[0] / length, z[1] / length, z[2] / length],
    ])
}

/// Orthonormal, right-handed, within the reference's tolerance.
fn is_rotation(r: &[[f64; 3]; 3]) -> bool {
    for row in 0..3 {
        for column in 0..3 {
            if !r[row][column].is_finite() {
                return false;
            }
        }
    }
    let mut gram = [[0.0f64; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            gram[row][column] = (0..3).map(|k| r[k][row] * r[k][column]).sum();
        }
    }
    for row in 0..3 {
        for column in 0..3 {
            let expected = if row == column { 1.0 } else { 0.0 };
            if (gram[row][column] - expected).abs() > ANCHOR_TOLERANCE {
                return false;
            }
        }
    }
    let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
        - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
        + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
    (det - 1.0).abs() <= ANCHOR_TOLERANCE
}

fn multiply(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0f64; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            out[row][column] = (0..3).map(|k| a[row][k] * b[k][column]).sum();
        }
    }
    out
}

fn transpose(a: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [a[0][0], a[1][0], a[2][0]],
        [a[0][1], a[1][1], a[2][1]],
        [a[0][2], a[1][2], a[2][2]],
    ]
}

fn rotate(a: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
        a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
        a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
    ]
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        0.5 * (values[middle - 1] + values[middle])
    } else {
        values[middle]
    }
}

/// One SAM anchor: which clip frame, and the hips it saw in camera space.
#[derive(Clone, Copy, Debug)]
pub struct Anchor {
    /// Index into the clip's frames (not the video's).
    pub frame: usize,
    /// Camera-space position of the hips.
    pub point: [f32; 3],
}

/// `placement_fit`: the rigid placement of one take.
///
/// `pelvis` is the FlowHMR pelvis per clip frame (the body's own `trans + J₀`),
/// `camera` the y-up extrinsics of the same frames and `scene` the scene
/// camera's first one. The anchors are the SAM hips of the first
/// [`ANCHOR_FRAMES`] frames; the offset is their component-wise median residual,
/// which is what makes it a *robust* fit rather than a least-squares one.
///
/// **Which frame the anchors live in.** The camera-space anchor is carried into
/// the world by the camera's own rotation, `Rᵀ·a`, and that is the frame
/// FlowHMR's `trans` is in here: our decoder applies the 180° yaw
/// (`GeneratedMotion::trans_y180`) that the reference applies elsewhere, and it
/// is what makes `Rᵀ·a − pelvis` constant to the millimetre. The reference
/// instead compares `wv(scene₀)·Rᵀ·a` against the transported pelvis, because
/// there a take is reconstructed in its **own** WV basis; with one shared VGGT
/// world that extra basis change is a 180° yaw of the whole body, so it is not
/// applied to the anchor. `q` still transports the *heading* between takes
/// (`wv(scene) · wv(take)ᵀ`, identity when both are the same camera).
pub fn fit(
    pelvis: &[[f32; 3]],
    camera: &[[[f32; 4]; 4]],
    scene: [[f32; 4]; 4],
    anchors: &[Anchor],
) -> Option<Fit> {
    if pelvis.is_empty() || camera.len() != pelvis.len() {
        return None;
    }
    let local = wv_rotation(&matrix3(&camera[0]))?;
    let shared = wv_rotation(&matrix3(&scene))?;
    // `q = shared @ local.T`: a later take has its own WV heading; transport it
    // into the scene's.
    let q = multiply(shared, transpose(local));

    let rotated: Vec<[f64; 3]> = pelvis
        .iter()
        .map(|p| rotate(q, [p[0] as f64, p[1] as f64, p[2] as f64]))
        .collect();
    let mut cam = Vec::with_capacity(camera.len());
    for extrinsic in camera {
        // The anchor's frame: the take's camera world (see the note above).
        cam.push(to_matrix(&matrix3(extrinsic)));
    }

    // The anchors must be unique, inside the fitted window and in front of the
    // camera; anything else is a stale placement file, not a fit.
    let limit = pelvis.len().min(ANCHOR_FRAMES);
    let mut seen = vec![false; limit];
    let mut usable = Vec::with_capacity(anchors.len());
    for anchor in anchors {
        if anchor.frame >= limit || seen[anchor.frame] || !(anchor.point[2] > 0.05) {
            continue;
        }
        if !is_rotation(&to_matrix(&matrix3(&camera[anchor.frame]))) {
            continue;
        }
        seen[anchor.frame] = true;
        usable.push(*anchor);
    }
    if usable.is_empty() || usable.len() > ANCHOR_FRAMES {
        return None;
    }
    let mut difference = [Vec::with_capacity(usable.len()), Vec::new(), Vec::new()];
    for anchor in &usable {
        let at = anchor.frame;
        let seen = rotate(
            transpose(cam[at]),
            [
                anchor.point[0] as f64,
                anchor.point[1] as f64,
                anchor.point[2] as f64,
            ],
        );
        for axis in 0..3 {
            difference[axis].push(seen[axis] - rotated[at][axis]);
        }
    }
    let mut offset = [0.0f64; 3];
    for axis in 0..3 {
        offset[axis] = median(&mut difference[axis]);
    }
    // `height = -offset[1]; offset[1] = 0`: the vertical residual is reported as
    // the camera height instead of moving the body, so FlowHMR's own height is
    // what the floor is measured against.
    let camera_height = -offset[1];
    offset[1] = 0.0;

    let world: Vec<[f64; 3]> = rotated
        .iter()
        .map(|p| [p[0] + offset[0], p[1] + offset[1], p[2] + offset[2]])
        .collect();
    let mut mean = 0.0f64;
    let mut worst = 0.0f64;
    for anchor in &usable {
        let at = anchor.frame;
        let seen = rotate(
            cam[at],
            [
                world[at][0],
                world[at][1] - camera_height,
                world[at][2],
            ],
        );
        let residual = [
            seen[0] - anchor.point[0] as f64,
            seen[1] - anchor.point[1] as f64,
            seen[2] - anchor.point[2] as f64,
        ];
        let length = (residual[0] * residual[0]
            + residual[1] * residual[1]
            + residual[2] * residual[2])
            .sqrt();
        mean += length;
        worst = worst.max(length);
    }
    Some(Fit {
        rotation: q,
        offset,
        camera_height,
        scene_basis: shared,
        anchors: usable.len(),
        residual_mean: mean / usable.len() as f64,
        residual_max: worst,
    })
}

/// The SAM anchor carried into the take's world: `Rᵀ · anchor`.
///
/// This is the frame the offset is fitted in (see [`fit`]), so it is also what a
/// residual has to be measured with.
pub fn anchor_point(extrinsic: &[[f32; 4]; 4], anchor: [f32; 3]) -> [f64; 3] {
    let rotation = to_matrix(&matrix3(extrinsic));
    rotate(
        transpose(rotation),
        [anchor[0] as f64, anchor[1] as f64, anchor[2] as f64],
    )
}

fn matrix3(extrinsic: &[[f32; 4]; 4]) -> [f32; 9] {    [
        extrinsic[0][0],
        extrinsic[0][1],
        extrinsic[0][2],
        extrinsic[1][0],
        extrinsic[1][1],
        extrinsic[1][2],
        extrinsic[2][0],
        extrinsic[2][1],
        extrinsic[2][2],
    ]
}

fn to_matrix(values: &[f32; 9]) -> [[f64; 3]; 3] {
    [
        [values[0] as f64, values[1] as f64, values[2] as f64],
        [values[3] as f64, values[4] as f64, values[5] as f64],
        [values[6] as f64, values[7] as f64, values[8] as f64],
    ]
}

/// Axis-angle to a rotation matrix (Rodrigues).
fn from_axis_angle(v: [f32; 3]) -> [[f64; 3]; 3] {
    let angle = ((v[0] as f64) * (v[0] as f64) + (v[1] as f64) * (v[1] as f64) + (v[2] as f64) * (v[2] as f64))
        .sqrt();
    if angle < 1e-9 {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    let axis = [
        v[0] as f64 / angle,
        v[1] as f64 / angle,
        v[2] as f64 / angle,
    ];
    let (s, c) = angle.sin_cos();
    let t = 1.0 - c;
    [
        [
            t * axis[0] * axis[0] + c,
            t * axis[0] * axis[1] - s * axis[2],
            t * axis[0] * axis[2] + s * axis[1],
        ],
        [
            t * axis[0] * axis[1] + s * axis[2],
            t * axis[1] * axis[1] + c,
            t * axis[1] * axis[2] - s * axis[0],
        ],
        [
            t * axis[0] * axis[2] - s * axis[1],
            t * axis[1] * axis[2] + s * axis[0],
            t * axis[2] * axis[2] + c,
        ],
    ]
}

/// Rotation matrix to an axis-angle vector.
fn to_axis_angle(m: &[[f64; 3]; 3]) -> [f32; 3] {
    let trace = m[0][0] + m[1][1] + m[2][2];
    let angle = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0).acos();
    if angle < 1e-6 {
        return [0.0; 3];
    }
    let axis = [m[2][1] - m[1][2], m[0][2] - m[2][0], m[1][0] - m[0][1]];
    let length =
        (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    if length < 1e-9 {
        return [0.0; 3];
    }
    [
        (axis[0] / length * angle) as f32,
        (axis[1] / length * angle) as f32,
        (axis[2] / length * angle) as f32,
    ]
}

/// Apply the placement to every frame: `rot6d' = q·R_root`, and
/// `trans' = (trans + rest) @ qᵀ − rest + offset`, the reference's rigid
/// transform about the pelvis.
///
/// `rest_pelvis` is the shape's rest pelvis (`rest[0]`, which is also the root
/// joint's translation), so the two formulas pivot on the same point. Returns
/// the rebuilt skeletons.
pub fn apply(
    poses: &mut [Pose],
    body: &BodyModel,
    rest_pelvis: [f64; 3],
    fit: &Fit,
) -> Vec<Skeleton> {
    for pose in poses.iter_mut() {
        let root = from_axis_angle(pose.root);
        pose.root = to_axis_angle(&multiply(fit.rotation, root));
        let translation = [
            pose.translation[0] as f64 + rest_pelvis[0],
            pose.translation[1] as f64 + rest_pelvis[1],
            pose.translation[2] as f64 + rest_pelvis[2],
        ];
        let rotated = rotate(fit.rotation, translation);
        pose.translation = [
            (rotated[0] - rest_pelvis[0] + fit.offset[0]) as f32,
            (rotated[1] - rest_pelvis[1] + fit.offset[1]) as f32,
            (rotated[2] - rest_pelvis[2] + fit.offset[2]) as f32,
        ];
    }
    poses
        .iter()
        .map(|pose| body.forward_joints(pose))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> [f32; 9] {
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    }

    /// Yaw about `y` by `degrees`, in the row-major layout the extrinsics use.
    fn yaw(degrees: f32) -> [f32; 9] {
        let angle = degrees.to_radians();
        let (s, c) = angle.sin_cos();
        [c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c]
    }

    fn close(a: [f64; 3], b: [f64; 3], tolerance: f64) -> bool {
        (0..3).all(|axis| (a[axis] - b[axis]).abs() <= tolerance)
    }

    #[test]
    fn the_identity_camera_has_the_identity_basis() {
        let basis = wv_rotation(&identity()).expect("identity is a rotation");
        assert!(close(basis[0], [1.0, 0.0, 0.0], 1e-9));
        assert!(close(basis[1], [0.0, 1.0, 0.0], 1e-9));
        assert!(close(basis[2], [0.0, 0.0, 1.0], 1e-9));
    }

    /// A camera turned about `y` turns its WV basis by the same angle: the
    /// `y` row is gravity and may never move.
    #[test]
    fn a_yawed_camera_keeps_gravity_and_turns_the_heading() {
        let basis = wv_rotation(&yaw(90.0)).expect("a yawed camera is a rotation");
        assert!(close(basis[1], [0.0, 1.0, 0.0], 1e-9));
        // R = yaw(+90): Rᵀ·ẑ = the third row = [-1, 0, 0].
        assert!(close(basis[2], [-1.0, 0.0, 0.0], 1e-6), "{:?}", basis[2]);
        assert!(close(basis[0], [0.0, 0.0, 1.0], 1e-6), "{:?}", basis[0]);
    }

    /// A camera looking straight down has no ground heading.
    #[test]
    fn a_vertical_camera_is_refused() {
        // Looking down: the third row of R is (0, -1, 0).
        let top_down = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0];
        assert!(wv_rotation(&top_down).is_none());
    }

    #[test]
    fn a_scaled_matrix_is_not_a_rotation() {
        let scaled = [2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0];
        assert!(wv_rotation(&scaled).is_none());
    }

    fn extrinsic(rotation: [f32; 9]) -> [[f32; 4]; 4] {
        let mut out = [[0.0f32; 4]; 4];
        for row in 0..3 {
            for column in 0..3 {
                out[row][column] = rotation[row * 3 + column];
            }
        }
        out[3][3] = 1.0;
        out
    }

    /// A single take has `q = I` and an offset that removes the median anchor
    /// residual on X/Z only.
    #[test]
    fn one_take_keeps_its_heading_and_fits_a_flat_offset() {
        let camera = vec![extrinsic(identity()); 4];
        let pelvis = vec![[0.0, 1.0, 0.0], [0.1, 1.0, 0.2], [0.2, 1.0, 0.4], [0.3, 1.0, 0.6]];
        // The SAM hips sit 0.5 m to the right and 2 m in front, and 1 cm up.
        let anchors: Vec<Anchor> = (0..4)
            .map(|frame| Anchor {
                frame,
                point: [
                    pelvis[frame][0] + 0.5,
                    pelvis[frame][1] + 0.01,
                    pelvis[frame][2] + 2.0,
                ],
            })
            .collect();
        let fit = fit(&pelvis, &camera, extrinsic(identity()), &anchors).expect("a fit");
        assert!((fit.rotation[0][0] - 1.0).abs() < 1e-9, "one take must not rotate");
        assert!((fit.offset[0] - 0.5).abs() < 1e-6, "x offset {:?}", fit.offset);
        assert!((fit.offset[2] - 2.0).abs() < 1e-6, "z offset {:?}", fit.offset);
        // The vertical residual becomes the camera height, and the offset keeps
        // the native height.
        assert_eq!(fit.offset[1], 0.0);
        assert!((fit.camera_height + 0.01).abs() < 1e-6, "height {}", fit.camera_height);
        assert_eq!(fit.anchors, 4);
    }

    /// Two takes of one scene: the second take's heading is transported into the
    /// first one's, so a body walking along the take's `+z` ends up walking along
    /// the scene's `+x` when the cameras are 90 degrees apart.
    #[test]
    fn a_later_take_is_transported_into_the_scene_heading() {
        let mut take = [0.0f32; 9];
        // A take camera turned to look along −x: yaw(−90) as a rotation.
        take.copy_from_slice(&yaw(-90.0));
        let camera = vec![extrinsic(take); 3];
        // In the take's own frame the actor walks along +z.
        let pelvis = vec![[0.0, 1.0, 0.0], [0.0, 1.0, 0.5], [0.0, 1.0, 1.0]];
        let anchors: Vec<Anchor> = (0..3)
            .map(|frame| Anchor {
                frame,
                point: [pelvis[frame][0], pelvis[frame][1], pelvis[frame][2]],
            })
            .collect();
        // The scene camera looks along +z (identity).
        let fit = fit(&pelvis, &camera, extrinsic(identity()), &anchors).expect("a fit");
        // Transport is a pure yaw: gravity is untouched and the heading turns by
        // exactly the cameras' relative yaw.
        assert!((fit.rotation[1][1] - 1.0).abs() < 1e-9);
        let walked = [
            fit.point([0.0, 1.0, 1.0])[0] - fit.point([0.0, 1.0, 0.0])[0],
            fit.point([0.0, 1.0, 1.0])[2] - fit.point([0.0, 1.0, 0.0])[2],
        ];
        assert!(
            (walked[0].abs() - 1.0).abs() < 1e-6 || (walked[1].abs() - 1.0).abs() < 1e-6,
            "the walking axis must end up axis-aligned in the scene: {walked:?}"
        );
    }

    /// A camera whose forward axis is `−z` (the y-up OpenCV convention, i.e. a
    /// 180° yaw of `wv`) must not rotate or mirror the anchor: the offset is
    /// measured in the take's own world. This is the bug that put a 180°-yawed
    /// body 10 m from its anchors.
    #[test]
    fn a_backward_facing_camera_does_not_mirror_the_anchors() {
        let turn = yaw(180.0);
        let camera = vec![extrinsic(turn); 4];
        let pelvis = vec![[0.0, 1.0, 0.0], [0.05, 1.0, 0.1], [0.1, 1.0, 0.2], [0.15, 1.0, 0.3]];
        // Build anchors that sit at `pelvis + delta` in the *world*, then take
        // them into the camera as SAM would report them.
        let rotation = to_matrix(&turn);
        // The camera looks along −z, so a body in *front* of it sits at −z.
        let delta = [0.4, 0.0, -1.5];
        let anchors: Vec<Anchor> = (0..4)
            .map(|frame| {
                let world = [
                    pelvis[frame][0] as f64 + delta[0],
                    pelvis[frame][1] as f64 + delta[1],
                    pelvis[frame][2] as f64 + delta[2],
                ];
                let camera_space = rotate(rotation, world);
                Anchor {
                    frame,
                    point: [
                        camera_space[0] as f32,
                        camera_space[1] as f32,
                        camera_space[2] as f32,
                    ],
                }
            })
            .collect();
        let fit = fit(&pelvis, &camera, extrinsic(turn), &anchors).expect("a fit");
        // The 180° yaw of the camera is the *scene* basis too, so `q` is identity
        // and the offset is the plain world displacement.
        assert!((fit.offset[0] - delta[0]).abs() < 1e-5, "x {:?}", fit.offset);
        assert!((fit.offset[2] - delta[2]).abs() < 1e-5, "z {:?}", fit.offset);
        let placed = fit.point([0.0, 1.0, 0.0]);
        assert!(
            close(placed, [0.4, 1.0, -1.5], 1e-5),
            "placed {placed:?}"
        );
    }

    #[test]
    fn no_anchors_means_no_offset() {
        let camera = vec![extrinsic(identity()); 3];
        let pelvis = vec![[0.0, 1.0, 0.0]; 3];
        assert!(fit(&pelvis, &camera, extrinsic(identity()), &[]).is_none());
    }

    #[test]
    fn an_anchor_behind_the_camera_is_refused() {
        let camera = vec![extrinsic(identity())];
        let pelvis = vec![[0.0, 1.0, 0.0]];
        let anchors = [Anchor {
            frame: 0,
            point: [0.0, 1.0, -0.5],
        }];
        assert!(fit(&pelvis, &camera, extrinsic(identity()), &anchors).is_none());
    }
}
