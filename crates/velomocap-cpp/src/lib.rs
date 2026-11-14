//! Safe Rust wrappers over the VeloMocap C++ kernels, plus the optional
//! LibTorch runners.
//!
//! The kernel layer owns no state: every function takes borrowed slices and
//! writes into caller-provided output slices, which keeps the FFI surface
//! allocation-free and lets the Rust core schedule work across rayon threads
//! without any locking.
//!
//! [`sam3d`] is different: it keeps a loaded TorchScript module alive behind an
//! opaque handle, and is only compiled when LibTorch is found (see `build.rs`).

use std::ffi::CStr;
use std::os::raw::{c_char, c_double, c_int};

pub mod sam3d;

extern "C" {
    fn vm_build_features() -> *const c_char;

    fn vm_mat3_mul(a: *const c_double, b: *const c_double, n: usize, out: *mut c_double);
    fn vm_mat3_tmul(a: *const c_double, b: *const c_double, n: usize, out: *mut c_double);
    fn vm_mat3_transpose(a: *const c_double, n: usize, out: *mut c_double);
    fn vm_quat_wxyz_to_mat(q: *const c_double, n: usize, out: *mut c_double);
    fn vm_mat_to_quat_wxyz(m: *const c_double, n: usize, out: *mut c_double);
    fn vm_rotvec_to_mat(rv: *const c_double, n: usize, out: *mut c_double);
    fn vm_mat_to_rotvec(m: *const c_double, n: usize, out: *mut c_double);
    fn vm_align(a: *const c_double, b: *const c_double, n: usize, out: *mut c_double);
    fn vm_reference_frame(d: *const c_double, n: usize, out: *mut c_double);
    fn vm_mat_to_euler(
        m: *const c_double,
        n: usize,
        a0: c_int,
        a1: c_int,
        a2: c_int,
        angles: *mut c_double,
        margin: *mut c_double,
    );
    fn vm_euler_to_mat(
        angles: *const c_double,
        n: usize,
        a0: c_int,
        a1: c_int,
        a2: c_int,
        out: *mut c_double,
    );
    fn vm_continuous_angles(angles: *mut c_double, n: usize);
    fn vm_relative_magnitude(a: *const c_double, b: *const c_double, n: usize, out: *mut c_double);

    fn vm_fk(
        rest: *const c_double,
        root: *const c_double,
        quat: *const c_double,
        n: usize,
        joints: usize,
        parents: *const c_int,
        pos: *mut c_double,
        rot: *mut c_double,
    );
    fn vm_local_from_world(
        world: *const c_double,
        n: usize,
        joints: usize,
        parents: *const c_int,
        out: *mut c_double,
    );

    fn vm_weights_mat3(
        w: *const c_double,
        vertices: usize,
        joints: usize,
        mat: *const c_double,
        n: usize,
        out: *mut c_double,
    );
    fn vm_weights_vec3(
        w: *const c_double,
        vertices: usize,
        joints: usize,
        vec: *const c_double,
        n: usize,
        out: *mut c_double,
    );
    fn vm_apply_blend(
        blend: *const c_double,
        posed: *const c_double,
        n: usize,
        vertices: usize,
        out: *mut c_double,
    );

    fn vm_gaussian_filter1d(
        input: *const c_double,
        n: usize,
        sigma: c_double,
        truncate: c_double,
        out: *mut c_double,
    );
    fn vm_minmax_filter1d(input: *const c_double, n: usize, size: usize, is_max: c_int, out: *mut c_double);
    fn vm_gradient1d(input: *const c_double, n: usize, out: *mut c_double);
    fn vm_interp(
        xp: *const c_double,
        fp: *const c_double,
        m: usize,
        x: *const c_double,
        n: usize,
        out: *mut c_double,
    );
}

/// `"avx2"` when the kernels were compiled with AVX2/FMA, else `"scalar"`.
pub fn build_features() -> &'static str {
    unsafe {
        let ptr = vm_build_features();
        if ptr.is_null() {
            return "unknown";
        }
        CStr::from_ptr(ptr).to_str().unwrap_or("unknown")
    }
}

fn assert_len(actual: usize, expected: usize, what: &str) {
    assert_eq!(
        actual, expected,
        "velomocap-cpp: {what} has {actual} elements, expected {expected}"
    );
}

pub fn mat3_mul(a: &[f64], b: &[f64], out: &mut [f64]) {
    let n = a.len() / 9;
    assert_len(b.len(), n * 9, "mat3_mul b");
    assert_len(out.len(), n * 9, "mat3_mul out");
    unsafe { vm_mat3_mul(a.as_ptr(), b.as_ptr(), n, out.as_mut_ptr()) }
}

pub fn mat3_tmul(a: &[f64], b: &[f64], out: &mut [f64]) {
    let n = a.len() / 9;
    assert_len(b.len(), n * 9, "mat3_tmul b");
    assert_len(out.len(), n * 9, "mat3_tmul out");
    unsafe { vm_mat3_tmul(a.as_ptr(), b.as_ptr(), n, out.as_mut_ptr()) }
}

pub fn mat3_transpose(a: &[f64], out: &mut [f64]) {
    let n = a.len() / 9;
    assert_len(out.len(), n * 9, "mat3_transpose out");
    unsafe { vm_mat3_transpose(a.as_ptr(), n, out.as_mut_ptr()) }
}

pub fn quat_wxyz_to_mat(q: &[f64], out: &mut [f64]) {
    let n = q.len() / 4;
    assert_len(out.len(), n * 9, "quat_wxyz_to_mat out");
    unsafe { vm_quat_wxyz_to_mat(q.as_ptr(), n, out.as_mut_ptr()) }
}

pub fn mat_to_quat_wxyz(m: &[f64], out: &mut [f64]) {
    let n = m.len() / 9;
    assert_len(out.len(), n * 4, "mat_to_quat_wxyz out");
    unsafe { vm_mat_to_quat_wxyz(m.as_ptr(), n, out.as_mut_ptr()) }
}

pub fn rotvec_to_mat(rv: &[f64], out: &mut [f64]) {
    let n = rv.len() / 3;
    assert_len(out.len(), n * 9, "rotvec_to_mat out");
    unsafe { vm_rotvec_to_mat(rv.as_ptr(), n, out.as_mut_ptr()) }
}

pub fn mat_to_rotvec(m: &[f64], out: &mut [f64]) {
    let n = m.len() / 9;
    assert_len(out.len(), n * 3, "mat_to_rotvec out");
    unsafe { vm_mat_to_rotvec(m.as_ptr(), n, out.as_mut_ptr()) }
}

pub fn align(a: &[f64], b: &[f64], out: &mut [f64]) {
    let n = a.len() / 3;
    assert_len(b.len(), n * 3, "align b");
    assert_len(out.len(), n * 9, "align out");
    unsafe { vm_align(a.as_ptr(), b.as_ptr(), n, out.as_mut_ptr()) }
}

pub fn reference_frame(direction: &[f64], out: &mut [f64]) {
    let n = direction.len() / 3;
    assert_len(out.len(), n * 9, "reference_frame out");
    unsafe { vm_reference_frame(direction.as_ptr(), n, out.as_mut_ptr()) }
}

/// Intrinsic Tait-Bryan extraction for `R = R_a0(x0) R_a1(x1) R_a2(x2)`.
/// Returns `(angles, margins)` where `margins = |cos(x1)|`.
pub fn mat_to_euler(m: &[f64], axes: [usize; 3]) -> (Vec<f64>, Vec<f64>) {
    let n = m.len() / 9;
    let mut angles = vec![0.0; n * 3];
    let mut margin = vec![0.0; n];
    unsafe {
        vm_mat_to_euler(
            m.as_ptr(),
            n,
            axes[0] as c_int,
            axes[1] as c_int,
            axes[2] as c_int,
            angles.as_mut_ptr(),
            margin.as_mut_ptr(),
        )
    }
    (angles, margin)
}

pub fn euler_to_mat(angles: &[f64], axes: [usize; 3], out: &mut [f64]) {
    let n = angles.len() / 3;
    assert_len(out.len(), n * 9, "euler_to_mat out");
    unsafe {
        vm_euler_to_mat(
            angles.as_ptr(),
            n,
            axes[0] as c_int,
            axes[1] as c_int,
            axes[2] as c_int,
            out.as_mut_ptr(),
        )
    }
}

pub fn continuous_angles(angles: &mut [f64]) {
    let n = angles.len() / 3;
    unsafe { vm_continuous_angles(angles.as_mut_ptr(), n) }
}

pub fn relative_magnitude(a: &[f64], b: &[f64], out: &mut [f64]) {
    let n = a.len() / 9;
    assert_len(b.len(), n * 9, "relative_magnitude b");
    assert_len(out.len(), n, "relative_magnitude out");
    unsafe { vm_relative_magnitude(a.as_ptr(), b.as_ptr(), n, out.as_mut_ptr()) }
}

/// Forward kinematics. `rest` is `joints*3`, `root` is `n*3`, `quat` is
/// `n*joints*4` (w, x, y, z). Outputs are `pos` (`n*joints*3`) and `rot`
/// (`n*joints*9`).
pub fn fk(
    rest: &[f64],
    root: &[f64],
    quat: &[f64],
    joints: usize,
    parents: &[i32],
    pos: &mut [f64],
    rot: &mut [f64],
) {
    let n = root.len() / 3;
    assert_len(rest.len(), joints * 3, "fk rest");
    assert_len(parents.len(), joints, "fk parents");
    assert_len(quat.len(), n * joints * 4, "fk quat");
    assert_len(pos.len(), n * joints * 3, "fk pos");
    assert_len(rot.len(), n * joints * 9, "fk rot");
    unsafe {
        vm_fk(
            rest.as_ptr(),
            root.as_ptr(),
            quat.as_ptr(),
            n,
            joints,
            parents.as_ptr(),
            pos.as_mut_ptr(),
            rot.as_mut_ptr(),
        )
    }
}

pub fn local_from_world(world: &[f64], n: usize, joints: usize, parents: &[i32], out: &mut [f64]) {
    assert_len(world.len(), n * joints * 9, "local_from_world world");
    assert_len(out.len(), n * joints * 9, "local_from_world out");
    unsafe { vm_local_from_world(world.as_ptr(), n, joints, parents.as_ptr(), out.as_mut_ptr()) }
}

pub fn weights_mat3(w: &[f64], vertices: usize, joints: usize, mat: &[f64], n: usize, out: &mut [f64]) {
    assert_len(w.len(), vertices * joints, "weights_mat3 w");
    assert_len(mat.len(), n * joints * 9, "weights_mat3 mat");
    assert_len(out.len(), n * vertices * 9, "weights_mat3 out");
    unsafe { vm_weights_mat3(w.as_ptr(), vertices, joints, mat.as_ptr(), n, out.as_mut_ptr()) }
}

pub fn weights_vec3(w: &[f64], vertices: usize, joints: usize, vec: &[f64], n: usize, out: &mut [f64]) {
    assert_len(w.len(), vertices * joints, "weights_vec3 w");
    assert_len(vec.len(), n * joints * 3, "weights_vec3 vec");
    assert_len(out.len(), n * vertices * 3, "weights_vec3 out");
    unsafe { vm_weights_vec3(w.as_ptr(), vertices, joints, vec.as_ptr(), n, out.as_mut_ptr()) }
}

pub fn apply_blend(blend: &[f64], posed: &[f64], n: usize, vertices: usize, out: &mut [f64]) {
    assert_len(blend.len(), n * vertices * 9, "apply_blend blend");
    assert_len(posed.len(), n * vertices * 3, "apply_blend posed");
    assert_len(out.len(), n * vertices * 3, "apply_blend out");
    unsafe { vm_apply_blend(blend.as_ptr(), posed.as_ptr(), n, vertices, out.as_mut_ptr()) }
}

pub fn gaussian_filter1d(input: &[f64], sigma: f64, truncate: f64, out: &mut [f64]) {
    assert_len(out.len(), input.len(), "gaussian_filter1d out");
    unsafe { vm_gaussian_filter1d(input.as_ptr(), input.len(), sigma, truncate, out.as_mut_ptr()) }
}

pub fn minmax_filter1d(input: &[f64], size: usize, is_max: bool, out: &mut [f64]) {
    assert_len(out.len(), input.len(), "minmax_filter1d out");
    unsafe {
        vm_minmax_filter1d(
            input.as_ptr(),
            input.len(),
            size,
            if is_max { 1 } else { 0 },
            out.as_mut_ptr(),
        )
    }
}

pub fn gradient1d(input: &[f64], out: &mut [f64]) {
    assert_len(out.len(), input.len(), "gradient1d out");
    unsafe { vm_gradient1d(input.as_ptr(), input.len(), out.as_mut_ptr()) }
}

pub fn interp(xp: &[f64], fp: &[f64], x: &[f64], out: &mut [f64]) {
    assert_len(xp.len(), fp.len(), "interp xp/fp");
    assert_len(out.len(), x.len(), "interp out");
    unsafe {
        vm_interp(
            xp.as_ptr(),
            fp.as_ptr(),
            xp.len(),
            x.as_ptr(),
            x.len(),
            out.as_mut_ptr(),
        )
    }
}


/// Parse a scipy Euler order string (`"xyz"`, `"ZXY"`, ...) into the axis
/// indices and intrinsic flag used by the kernels.
///
/// Uppercase means intrinsic (`R = R_a0 R_a1 R_a2`); lowercase means extrinsic,
/// which is the same intrinsic product with reversed axes and reversed angles.
pub fn parse_order(order: &str) -> ([usize; 3], bool) {
    let chars: Vec<char> = order.chars().collect();
    assert_eq!(chars.len(), 3, "Euler order must have three axes: {order:?}");
    let intrinsic = chars[0].is_ascii_uppercase();
    let mut axes = [0usize; 3];
    for (i, c) in chars.iter().enumerate() {
        axes[i] = match c.to_ascii_lowercase() {
            'x' => 0,
            'y' => 1,
            'z' => 2,
            other => panic!("Unsupported Euler axis {other:?} in {order:?}"),
        };
    }
    (axes, intrinsic)
}
