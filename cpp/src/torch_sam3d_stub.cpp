// Stub of the LibTorch runner, used when LibTorch is not available.
//
// The workspace must build on a machine without a torch wheel; every entry
// point reports "not built" instead of failing to link.

#include "velomocap/torch_sam3d.h"

#include <cstring>

namespace {
void fill(char* error, size_t length, const char* message) {
  if (error == nullptr || length == 0) {
    return;
  }
  const size_t count = std::strlen(message);
  const size_t copy = count < length - 1 ? count : length - 1;
  std::memcpy(error, message, copy);
  error[copy] = '\0';
}
}  // namespace

extern "C" {

int vm_sam3d_built(void) { return 0; }

void* vm_sam3d_load(const char* path, int use_cuda, char* error, size_t error_len) {
  (void)path;
  (void)use_cuda;
  fill(error, error_len, "LibTorch was not available when this build was made");
  return nullptr;
}

void vm_sam3d_free(void* handle) { (void)handle; }

int vm_sam3d_forward(void* handle, const float* image, const float* bbox_center,
                     const float* bbox_scale, const float* cam_int,
                     const float* keypoints, const float* ori_size,
                     float* body_token, float* verts, float* skel_state,
                     float* pred, float* pred_cam, float* model_params,
                     char* error, size_t error_len) {
  (void)handle;
  (void)image;
  (void)bbox_center;
  (void)bbox_scale;
  (void)cam_int;
  (void)keypoints;
  (void)ori_size;
  (void)body_token;
  (void)verts;
  (void)skel_state;
  (void)pred;
  (void)pred_cam;
  (void)model_params;
  fill(error, error_len, "LibTorch was not available when this build was made");
  return 4;
}

int vm_sam3d_forward_full(void* handle, const float* frame, int height, int width,
                          const float* bbox_center, const float* bbox_scale,
                          const float* cam_int,
                          float* verts, int verts_count,
                          float* keypoints_3d, int kps3d_count,
                          float* keypoints_2d, int kps2d_count,
                          float* pred_cam, float* pred_cam_t, float* focal_length,
                          float* model_params, int params_count,
                          float* joint_rots, int rots_count,
                          float* hand_valid, int hand_count,
                          float* body_token, float* left_token, float* right_token,
                          int token_count,
                          float* skel_state, int skel_count,
                          float* pred, int pred_count,
                          char* error, size_t error_len) {
  (void)handle;
  (void)frame;
  (void)height;
  (void)width;
  (void)bbox_center;
  (void)bbox_scale;
  (void)cam_int;
  (void)verts;
  (void)verts_count;
  (void)keypoints_3d;
  (void)kps3d_count;
  (void)keypoints_2d;
  (void)kps2d_count;
  (void)pred_cam;
  (void)pred_cam_t;
  (void)focal_length;
  (void)model_params;
  (void)params_count;
  (void)joint_rots;
  (void)rots_count;
  (void)hand_valid;
  (void)hand_count;
  (void)body_token;
  (void)left_token;
  (void)right_token;
  (void)token_count;
  (void)skel_state;
  (void)skel_count;
  (void)pred;
  (void)pred_count;
  fill(error, error_len, "LibTorch was not available when this build was made");
  return 4;
}

void vm_cuda_empty_cache(void) {}

int vm_cuda_mem_info(long long* free_bytes, long long* total_bytes) {
  (void)free_bytes;
  (void)total_bytes;
  return 1;
}
}  // extern "C"
