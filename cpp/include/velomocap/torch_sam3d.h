// LibTorch runner for the whole SAM 3D Body pose branch.
//
// The module is the one `tools/torch/sam3d_jit.py` traces: backbone, prompt,
// decoder, the six refinement rounds, the parameter decode and the scripted MHR
// body. Nothing is reimplemented here — this file only moves tensors.
//
// The C ABI is allocation-free in both directions: the caller owns every buffer
// and the handle only keeps the loaded module alive.

#ifndef VELOMOCAP_TORCH_SAM3D_H
#define VELOMOCAP_TORCH_SAM3D_H

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// Output sizes, so the Rust side never hard-codes them.
#define VM_SAM3D_IMAGE (512)
#define VM_SAM3D_TOKEN (1024)
#define VM_SAM3D_VERTS (18439)
#define VM_SAM3D_JOINTS (127)
#define VM_SAM3D_SKEL_STATE (8)
#define VM_SAM3D_PRED (519)
#define VM_SAM3D_CAM (3)
#define VM_SAM3D_PARAMS (204)

// Load a TorchScript module. Returns null on failure and fills `error`.
void* vm_sam3d_load(const char* path, int use_cuda, char* error, size_t error_len);

void vm_sam3d_free(void* handle);

// True when the library was built against LibTorch at all.
int vm_sam3d_built(void);

// One forward pass.
//
//   image        3 * 512 * 512 float32, NCHW, RGB in [0, 1]
//   bbox_center  2 values, pixels
//   bbox_scale   2 values, pixels (already padded)
//   cam_int      9 values, row major
//   keypoints    3 values: cx, cy, label (-2 for "no prompt")
//   ori_size     2 values: width, height
//
// Outputs are written into caller buffers of the documented sizes. Returns 0 on
// success and a non-zero code with `error` filled otherwise.
int vm_sam3d_forward(void* handle, const float* image, const float* bbox_center,
                     const float* bbox_scale, const float* cam_int,
                     const float* keypoints, const float* ori_size,
                     float* body_token, float* verts, float* skel_state,
                     float* pred, float* pred_cam, float* model_params,
                     char* error, size_t error_len);

// One *whole-pipeline* forward pass: the module `tools/torch/sam3d_full_jit.py`
// traces, i.e. `run_inference(inference_type="full")` — body pass, both hand
// crops, the fusion criteria, the keypoint-prompt pass and the final mesh.
//
//   frame        batch * 3 * height * width float32, NCHW, RGB in [0, 1], the
//                *original* frames (crops happen inside the graph)
//   batch        items in this call: one item is one person in one frame. A
//                traced module bakes this, so it must match the trace and the
//                caller pads by repeating its last item.
//   bbox_center  batch * 2 values, pixels
//   bbox_scale   batch * 2 values, pixels (already padded)
//   cam_int      batch * 9 values, row major
//
// Every output buffer is sized by the caller; `count` is the number of floats
// it holds and the call fails rather than overflowing if the module disagrees.
// Outputs are batch-major, so a caller holding `batch` of them gets one row per
// item in order.
int vm_sam3d_forward_full(void* handle, const float* frame, int batch, int height,
                          int width,
                          const float* bbox_center, const float* bbox_scale,
                          const float* cam_int,
                          float* verts, int verts_count,
                          float* keypoints_3d, int kps3d_count,
                          float* keypoints_2d, int kps2d_count,
                          float* pred_cam, int cam_count,
                          float* pred_cam_t, int cam_t_count,
                          float* focal_length, int focal_count,
                          float* model_params, int params_count,
                          float* joint_rots, int rots_count,
                          float* hand_valid, int hand_count,
                          float* body_token, float* left_token, float* right_token,
                          int token_count,
                          float* skel_state, int skel_count,
                          float* pred, int pred_count,
                          char* error, size_t error_len);


// CUDA housekeeping, so one stage's model does not starve the next.
//
// PyTorch's caching allocator keeps freed device blocks in a per-process pool:
// dropping a module frees it from PyTorch's point of view but `nvidia-smi` stays
// high, and the next stage's weights then have nowhere to go — they end up
// paged into host memory and every kernel crawls. Call this between stages.
void vm_cuda_empty_cache(void);

// Device memory in bytes. Returns 0 on success, 1 when this build or device
// cannot report it.
int vm_cuda_mem_info(long long* free_bytes, long long* total_bytes);
#ifdef __cplusplus
}
#endif

#endif  // VELOMOCAP_TORCH_SAM3D_H
