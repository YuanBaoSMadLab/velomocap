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
    fn vm_cuda_empty_cache();
    fn vm_cuda_mem_info(free_bytes: *mut i64, total_bytes: *mut i64) -> c_int;
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
        batch: c_int,
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
        cam_count: c_int,
        pred_cam_t: *mut f32,
        cam_t_count: c_int,
        focal_length: *mut f32,
        focal_count: c_int,
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
    /// Items per `forward`, fixed by the trace.
    batch: usize,
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
        Self::load_batch(path, gpu, 1)
    }

    /// Load a module traced at `batch` items per call.
    ///
    /// A traced graph bakes its batch dimension, so the caller has to say which
    /// one the file was traced at; [`Self::batch`] reports it back.
    pub fn load_batch(
        path: impl AsRef<Path>,
        gpu: bool,
        batch: usize,
    ) -> Result<Self, TorchError> {
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
        Ok(Self {
            handle,
            batch: batch.max(1),
        })
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

/// Give the CUDA caching allocator's freed blocks back to the driver.
///
/// Dropping a module frees it from PyTorch's point of view, but the allocator
/// keeps the blocks in its own pool: `nvidia-smi` does not move, and the next
/// stage's weights get pushed into host memory instead of the device. Call this
/// between stages. A no-op when the build has no LibTorch or no CUDA.
pub fn empty_cuda_cache() {
    unsafe { vm_cuda_empty_cache() }
}

/// Device memory as `(used, total)` bytes, when this build can ask the driver.
///
/// `None` on a CPU-only build, or when no device is visible.
pub fn cuda_memory() -> Option<(u64, u64)> {
    let mut free_bytes = 0i64;
    let mut total_bytes = 0i64;
    let code = unsafe { vm_cuda_mem_info(&mut free_bytes, &mut total_bytes) };
    if code != 0 || total_bytes <= 0 {
        return None;
    }
    let total = total_bytes as u64;
    let free = (free_bytes.max(0) as u64).min(total);
    Some((total - free, total))
}

/// `"1.2 / 24.0 GB"` for the console log, or `None` when there is no device.
pub fn cuda_memory_line() -> Option<String> {
    cuda_memory().map(|(used, total)| {
        format!(
            "{:.2} / {:.1} GB",
            used as f64 / 1e9,
            total as f64 / 1e9
        )
    })
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
    /// How many items one `forward` of this module carries.
    ///
    /// A traced graph bakes the batch dimension, so a module traced at eight
    /// items must always be handed eight. [`Self::run_full_batch`] pads for the
    /// caller, so this only matters when sizing the caller's own buffers.
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// `run_inference(inference_type="full")`: body, both hands, fusion.
    pub fn run_full(&self, input: &Sam3dFullInput<'_>) -> Result<Sam3dFullFrame, TorchError> {
        let mut frames = self.run_full_batch(std::slice::from_ref(input))?;
        frames
            .pop()
            .ok_or_else(|| TorchError("the module returned no frames".to_string()))
    }

    /// The same pass for several items at once.
    ///
    /// One item is one person in one frame. The module's batch is fixed by the
    /// trace, so a short slice is padded by repeating its last item and the
    /// extra results are dropped; the returned frames line up with `inputs`.
    /// Every item must be the same size, which holds for one video.
    pub fn run_full_batch(
        &self,
        inputs: &[Sam3dFullInput<'_>],
    ) -> Result<Vec<Sam3dFullFrame>, TorchError> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let height = inputs[0].height;
        let width = inputs[0].width;
        let plane = 3 * height * width;
        for input in inputs {
            assert!(
                input.frame.len() >= plane,
                "frame buffer must hold 3×H×W values"
            );
            assert!(
                input.height == height && input.width == width,
                "every item in a batch must be the same frame size"
            );
        }
        let batch = self.batch.max(inputs.len());
        let mut packed = Vec::with_capacity(batch * plane);
        let mut centers = Vec::with_capacity(batch * 2);
        let mut scales = Vec::with_capacity(batch * 2);
        let mut cameras = Vec::with_capacity(batch * 9);
        for index in 0..batch {
            let input = &inputs[index.min(inputs.len() - 1)];
            packed.extend_from_slice(&input.frame[..plane]);
            centers.extend_from_slice(&input.bbox_center);
            scales.extend_from_slice(&input.bbox_scale);
            cameras.extend_from_slice(&input.cam_int);
        }

        let mut verts = vec![0.0f32; batch * VERTS * 3];
        let mut keypoints_3d = vec![0.0f32; batch * 70 * 3];
        let mut keypoints_2d = vec![0.0f32; batch * 70 * 2];
        let mut pred_cam = vec![0.0f32; batch * CAM];
        let mut pred_cam_t = vec![0.0f32; batch * CAM];
        let mut focal_length = vec![0.0f32; batch];
        let mut model_params = vec![0.0f32; batch * PARAMS];
        let mut joint_rots = vec![0.0f32; batch * JOINTS * 9];
        let mut hand_valid = vec![0.0f32; batch * 2];
        let mut body_token = vec![0.0f32; batch * TOKEN];
        let mut left_token = vec![0.0f32; batch * TOKEN];
        let mut right_token = vec![0.0f32; batch * TOKEN];
        let mut skel_state = vec![0.0f32; batch * JOINTS * SKEL_STATE];
        let mut pred_vec = vec![0.0f32; batch * PRED];
        let mut buffer = vec![0i8 as c_char; ERROR_LEN];
        let code = unsafe {
            vm_sam3d_forward_full(
                self.handle,
                packed.as_ptr(),
                batch as c_int,
                height as c_int,
                width as c_int,
                centers.as_ptr(),
                scales.as_ptr(),
                cameras.as_ptr(),
                verts.as_mut_ptr(),
                verts.len() as c_int,
                keypoints_3d.as_mut_ptr(),
                keypoints_3d.len() as c_int,
                keypoints_2d.as_mut_ptr(),
                keypoints_2d.len() as c_int,
                pred_cam.as_mut_ptr(),
                pred_cam.len() as c_int,
                pred_cam_t.as_mut_ptr(),
                pred_cam_t.len() as c_int,
                focal_length.as_mut_ptr(),
                focal_length.len() as c_int,
                model_params.as_mut_ptr(),
                model_params.len() as c_int,
                joint_rots.as_mut_ptr(),
                joint_rots.len() as c_int,
                hand_valid.as_mut_ptr(),
                hand_valid.len() as c_int,
                body_token.as_mut_ptr(),
                left_token.as_mut_ptr(),
                right_token.as_mut_ptr(),
                (batch * TOKEN) as c_int,
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

        let mut frames = Vec::with_capacity(inputs.len());
        for index in 0..inputs.len() {
            let take = |data: &[f32], width: usize| {
                data[index * width..(index + 1) * width].to_vec()
            };
            frames.push(Sam3dFullFrame {
                verts: take(&verts, VERTS * 3),
                keypoints_3d: take(&keypoints_3d, 70 * 3),
                keypoints_2d: take(&keypoints_2d, 70 * 2),
                pred_cam: [
                    pred_cam[index * CAM],
                    pred_cam[index * CAM + 1],
                    pred_cam[index * CAM + 2],
                ],
                pred_cam_t: [
                    pred_cam_t[index * CAM],
                    pred_cam_t[index * CAM + 1],
                    pred_cam_t[index * CAM + 2],
                ],
                focal_length: focal_length[index],
                model_params: take(&model_params, PARAMS),
                joint_rots: take(&joint_rots, JOINTS * 9),
                hand_valid: [hand_valid[index * 2], hand_valid[index * 2 + 1]],
                body_token: take(&body_token, TOKEN),
                left_token: take(&left_token, TOKEN),
                right_token: take(&right_token, TOKEN),
                skel_state: take(&skel_state, JOINTS * SKEL_STATE),
                pred: take(&pred_vec, PRED),
            });
        }
        Ok(frames)
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
