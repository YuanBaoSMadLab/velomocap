//! SAM 3D Body through LibTorch.
//!
//! The C++ side loads the TorchScript module that `tools/torch/sam3d_jit.py`
//! produces — the whole reference forward pass, including the parameter decode
//! and the scripted MHR body — and this module only moves buffers across the
//! boundary.
//!
//! When the crate was built without LibTorch the same API exists but
//! [`Sam3dTorch::load`] fails with [`NOT_BUILT`], so callers can fall back to
//! the ONNX chain without any conditional compilation of their own.

use std::ffi::{CStr, CString};
use std::fmt;
use std::os::raw::{c_char, c_int};
use std::path::Path;

/// Pixels of the network's square input.
pub const IMAGE_SIZE: usize = 512;
/// Body token width — the motion model's feature row, per person.
pub const TOKEN: usize = 1024;
/// MHR vertices.
pub const VERTS: usize = 18439;
/// MHR joints.
pub const JOINTS: usize = 127;
/// Values per joint in `skel_state`: translation, quaternion, padding.
pub const SKEL_STATE: usize = 8;
/// Width of the head's packed prediction.
pub const PRED: usize = 519;
/// `pred_cam`.
pub const CAM: usize = 3;
/// Width of the decoded MHR parameters.
pub const PARAMS: usize = 204;

/// The reported error when the build had no LibTorch.
pub const NOT_BUILT: &str = "LibTorch was not available when this build was made";

extern "C" {
    fn vm_sam3d_built() -> c_int;
    fn vm_sam3d_load(
        path: *const c_char,
        use_cuda: c_int,
        error: *mut c_char,
        error_len: usize,
    ) -> *mut std::ffi::c_void;
    fn vm_sam3d_free(handle: *mut std::ffi::c_void);
    #[allow(clippy::too_many_arguments)]
    fn vm_sam3d_forward(
        handle: *mut std::ffi::c_void,
        image: *const f32,
        bbox_center: *const f32,
        bbox_scale: *const f32,
        cam_int: *const f32,
        keypoints: *const f32,
        ori_size: *const f32,
        body_token: *mut f32,
        verts: *mut f32,
        skel_state: *mut f32,
        pred: *mut f32,
        pred_cam: *mut f32,
        model_params: *mut f32,
        error: *mut c_char,
        error_len: usize,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    fn vm_sam3d_forward_full(
        handle: *mut std::ffi::c_void,
        frame: *const f32,
        height: c_int,
        width: c_int,
        bbox_center: *const f32,
        bbox_scale: *const f32,
        cam_int: *const f32,
        verts: *mut f32,
        verts_count: c_int,
        keypoints_3d: *mut f32,
        kps3d_count: c_int,
        keypoints_2d: *mut f32,
        kps2d_count: c_int,
        pred_cam: *mut f32,
        pred_cam_t: *mut f32,
        focal_length: *mut f32,
        model_params: *mut f32,
        params_count: c_int,
        joint_rots: *mut f32,
        rots_count: c_int,
        hand_valid: *mut f32,
        hand_count: c_int,
        body_token: *mut f32,
        left_token: *mut f32,
        right_token: *mut f32,
        token_count: c_int,
        skel_state: *mut f32,
        skel_count: c_int,
        pred: *mut f32,
        pred_count: c_int,
        error: *mut c_char,
        error_len: usize,
    ) -> c_int;
}

#[derive(Debug)]
pub struct TorchError(String);

impl fmt::Display for TorchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for TorchError {}

/// Everything one forward pass produces.
pub struct Sam3dFrame {
    pub body_token: Vec<f32>,
    /// `18439 × 3`, camera space, centimetres — the same layout the ONNX chain
    /// returns, so the rest of the pipeline is unchanged.
    pub verts: Vec<f32>,
    /// `127 × 8`.
    pub skel_state: Vec<f32>,
    /// `519`.
    pub pred: Vec<f32>,
    pub pred_cam: [f32; 3],
    /// `204`.
    pub model_params: Vec<f32>,
}

/// What one forward pass needs.
pub struct Sam3dInput<'a> {
    /// `3 × 512 × 512` NCHW, RGB in `[0, 1]`.
    pub image: &'a [f32],
    pub bbox_center: [f32; 2],
    pub bbox_scale: [f32; 2],
    /// Row-major 3×3.
    pub cam_int: [f32; 9],
    /// `cx, cy, label`; `label == -2` means "no prompt".
    pub keypoints: [f32; 3],
    /// `width, height` of the original frame.
    pub ori_size: [f32; 2],
}

/// A loaded module. Dropping it frees the C++ side.
pub struct Sam3dTorch {
    handle: *mut std::ffi::c_void,
}

// The handle is only touched through the C++ module, which keeps its own state;
// callers serialise access like any other inference session.
unsafe impl Send for Sam3dTorch {}

const ERROR_LEN: usize = 1024;

fn take_error(buffer: &[c_char]) -> String {
    let text = unsafe { CStr::from_ptr(buffer.as_ptr()) };
    let message = text.to_string_lossy().trim().to_string();
    if message.is_empty() {
        "unknown LibTorch error".to_string()
    } else {
        message
    }
}

impl Sam3dTorch {
    /// Whether this build has LibTorch at all.
    pub fn built() -> bool {
        unsafe { vm_sam3d_built() != 0 }
    }

    pub fn load(path: impl AsRef<Path>, gpu: bool) -> Result<Self, TorchError> {
        let path = path.as_ref();
        let text = CString::new(path.display().to_string())
            .map_err(|_| TorchError("path contains a NUL byte".into()))?;
        let mut buffer = vec![0i8 as c_char; ERROR_LEN];
        let handle = unsafe {
            vm_sam3d_load(
                text.as_ptr(),
                if gpu { 1 } else { 0 },
                buffer.as_mut_ptr(),
                ERROR_LEN,
            )
        };
        if handle.is_null() {
            return Err(TorchError(take_error(&buffer)));
        }
        Ok(Self { handle })
    }

    pub fn run(&self, input: &Sam3dInput<'_>) -> Result<Sam3dFrame, TorchError> {
        assert!(
            input.image.len() >= 3 * IMAGE_SIZE * IMAGE_SIZE,
            "image buffer must hold 3×{IMAGE_SIZE}×{IMAGE_SIZE} values"
        );
        let mut body_token = vec![0.0f32; TOKEN];
        let mut verts = vec![0.0f32; VERTS * 3];
        let mut skel_state = vec![0.0f32; JOINTS * SKEL_STATE];
        let mut pred = vec![0.0f32; PRED];
        let mut pred_cam = [0.0f32; CAM];
        let mut model_params = vec![0.0f32; PARAMS];
        let mut buffer = vec![0i8 as c_char; ERROR_LEN];
        let code = unsafe {
            vm_sam3d_forward(
                self.handle,
                input.image.as_ptr(),
                input.bbox_center.as_ptr(),
                input.bbox_scale.as_ptr(),
                input.cam_int.as_ptr(),
                input.keypoints.as_ptr(),
                input.ori_size.as_ptr(),
                body_token.as_mut_ptr(),
                verts.as_mut_ptr(),
                skel_state.as_mut_ptr(),
                pred.as_mut_ptr(),
                pred_cam.as_mut_ptr(),
                model_params.as_mut_ptr(),
                buffer.as_mut_ptr(),
                ERROR_LEN,
            )
        };
        if code != 0 {
            return Err(TorchError(take_error(&buffer)));
        }
        Ok(Sam3dFrame {
            body_token,
            verts,
            skel_state,
            pred,
            pred_cam,
            model_params,
        })
    }
}

impl Drop for Sam3dTorch {
    fn drop(&mut self) {
        unsafe { vm_sam3d_free(self.handle) };
    }
}

/// Everything the whole-pipeline module produces.
///
/// Same units as the reference `run_inference`: vertices and keypoints are in
/// metres in camera axes, `keypoints_2d` is in pixels of the frame that was
/// passed in.
pub struct Sam3dFullFrame {
    /// `18439 × 3`.
    pub verts: Vec<f32>,
    /// `70 × 3`.
    pub keypoints_3d: Vec<f32>,
    /// `70 × 2`.
    pub keypoints_2d: Vec<f32>,
    pub pred_cam: [f32; 3],
    pub pred_cam_t: [f32; 3],
    pub focal_length: f32,
    /// `204`.
    pub model_params: Vec<f32>,
    /// `127 × 9`, row-major per joint.
    pub joint_rots: Vec<f32>,
    /// `1` where the hand pass survived the fusion criteria.
    pub hand_valid: [f32; 2],
    /// `1024`: token 0 of the first body decoder call.
    pub body_token: Vec<f32>,
    /// `1024`: token 0 of the left-hand decoder call.
    pub left_token: Vec<f32>,
    /// `1024`: token 0 of the right-hand decoder call.
    pub right_token: Vec<f32>,
    /// `127 × 8`: raw MHR skeleton state of the final mesh (centimetres).
    pub skel_state: Vec<f32>,
    /// `519`: the prediction the final mesh was decoded from.
    pub pred: Vec<f32>,
}

impl Sam3dFullFrame {
    /// The `3072`-wide feature FlowHMR stacks: body ‖ left hand ‖ right hand.
    pub fn features(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(3 * TOKEN);
        out.extend_from_slice(&self.body_token);
        out.extend_from_slice(&self.left_token);
        out.extend_from_slice(&self.right_token);
        out
    }

    /// The mesh in the convention `velomocap_infer::Sam3dFrame` uses: centimetres
    /// with `y` down and `z` forward. This module returns metres in camera axes
    /// with `y` up and `z` towards the viewer, so flip `y` and `z` back.
    pub fn body_space_verts(&self) -> Vec<f32> {
        self.verts
            .chunks_exact(3)
            .flat_map(|p| [p[0] * 100.0, -p[1] * 100.0, p[2] * 100.0])
            .collect()
    }
}

/// What the whole-pipeline module needs: the original frame, not a crop.
pub struct Sam3dFullInput<'a> {
    /// `3 × height × width` NCHW, RGB in `[0, 1]`.
    pub frame: &'a [f32],
    pub height: usize,
    pub width: usize,
    pub bbox_center: [f32; 2],
    pub bbox_scale: [f32; 2],
    /// Row-major 3×3.
    pub cam_int: [f32; 9],
}

impl Sam3dTorch {
    /// `run_inference(inference_type="full")`: body, both hands, fusion.
    pub fn run_full(&self, input: &Sam3dFullInput<'_>) -> Result<Sam3dFullFrame, TorchError> {
        assert!(
            input.frame.len() >= 3 * input.height * input.width,
            "frame buffer must hold 3×H×W values"
        );
        let mut verts = vec![0.0f32; VERTS * 3];
        let mut keypoints_3d = vec![0.0f32; 70 * 3];
        let mut keypoints_2d = vec![0.0f32; 70 * 2];
        let mut pred_cam = [0.0f32; CAM];
        let mut pred_cam_t = [0.0f32; CAM];
        let mut focal_length = [0.0f32; 1];
        let mut model_params = vec![0.0f32; PARAMS];
        let mut joint_rots = vec![0.0f32; JOINTS * 9];
        let mut hand_valid = [0.0f32; 2];
        let mut body_token = vec![0.0f32; TOKEN];
        let mut left_token = vec![0.0f32; TOKEN];
        let mut right_token = vec![0.0f32; TOKEN];
        let mut skel_state = vec![0.0f32; JOINTS * SKEL_STATE];
        let mut pred_vec = vec![0.0f32; PRED];
        let mut buffer = vec![0i8 as c_char; ERROR_LEN];
        let code = unsafe {
            vm_sam3d_forward_full(
                self.handle,
                input.frame.as_ptr(),
                input.height as c_int,
                input.width as c_int,
                input.bbox_center.as_ptr(),
                input.bbox_scale.as_ptr(),
                input.cam_int.as_ptr(),
                verts.as_mut_ptr(),
                verts.len() as c_int,
                keypoints_3d.as_mut_ptr(),
                keypoints_3d.len() as c_int,
                keypoints_2d.as_mut_ptr(),
                keypoints_2d.len() as c_int,
                pred_cam.as_mut_ptr(),
                pred_cam_t.as_mut_ptr(),
                focal_length.as_mut_ptr(),
                model_params.as_mut_ptr(),
                model_params.len() as c_int,
                joint_rots.as_mut_ptr(),
                joint_rots.len() as c_int,
                hand_valid.as_mut_ptr(),
                hand_valid.len() as c_int,
                body_token.as_mut_ptr(),
                left_token.as_mut_ptr(),
                right_token.as_mut_ptr(),
                body_token.len() as c_int,
                skel_state.as_mut_ptr(),
                skel_state.len() as c_int,
                pred_vec.as_mut_ptr(),
                pred_vec.len() as c_int,
                buffer.as_mut_ptr(),
                ERROR_LEN,
            )
        };
        if code != 0 {
            return Err(TorchError(take_error(&buffer)));
        }
        Ok(Sam3dFullFrame {
            verts,
            keypoints_3d,
            keypoints_2d,
            pred_cam,
            pred_cam_t,
            focal_length: focal_length[0],
            model_params,
            joint_rots,
            hand_valid,
            body_token,
            left_token,
            right_token,
            skel_state,
            pred: pred_vec,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_whether_libtorch_was_linked() {
        // Either way the API must answer, not crash.
        let built = Sam3dTorch::built();
        if !built {
            match Sam3dTorch::load("nothing.pt", false) {
                Ok(_) => panic!("the stub must not load a module"),
                Err(error) => assert!(error.to_string().contains("LibTorch")),
            }
        }
    }
}
