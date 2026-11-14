// LibTorch runner for the SAM 3D Body TorchScript module.
//
// Compiled only when LibTorch is found (see `crates/velomocap-cpp/build.rs`);
// otherwise `stub.cpp` provides the same symbols and reports "not built".

#include "velomocap/torch_sam3d.h"

#include <torch/script.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

void fail(char* error, size_t length, const std::string& message) {
  if (error == nullptr || length == 0) {
    return;
  }
  const size_t count = message.size() < length - 1 ? message.size() : length - 1;
  std::memcpy(error, message.data(), count);
  error[count] = '\0';
}

struct Handle {
  torch::jit::Module module;
  torch::Device device = torch::kCPU;
};

torch::Tensor from(const float* data, std::initializer_list<int64_t> shape) {
  int64_t count = 1;
  for (int64_t size : shape) {
    count *= size;
  }
  // `from_blob` does not own the memory, so clone before the caller's buffer
  // can change.
  return torch::from_blob(const_cast<float*>(data), shape, torch::kFloat32).clone();
}

void copy_out(const torch::Tensor& tensor, float* out) {
  auto flat = tensor.contiguous().to(torch::kCPU).reshape({-1});
  std::memcpy(out, flat.data_ptr<float>(), sizeof(float) * flat.numel());
}

// Make the CUDA backend exist in this process.
//
// Linking `torch_cuda.lib` is not enough on its own: MSVC only pulls the import
// in when a symbol from it is referenced, and nothing here does. Without the
// DLL loaded the dispatcher only knows the CPU kernels and `jit::load(path,
// kCUDA)` fails with "Could not run 'aten::empty_strided' ... from the 'CUDA'
// backend". Loading it by name runs its static initialisers, which is what
// registers the backend. `torch/lib` must be on `PATH` so Windows can resolve
// `torch_cuda.dll`'s own dependencies (`cudart64_*.dll`, `cublas*`, `cudnn*`).
void ensure_cuda_backend() {
#ifdef _WIN32
  static bool loaded = false;
  if (loaded) {
    return;
  }
  loaded = true;
  HMODULE handle = LoadLibraryA("torch_cuda.dll");
  if (handle == nullptr) {
    std::fprintf(stderr,
                 "warning: could not load torch_cuda.dll (error %lu); "
                 "the module will fall back to the CPU\n",
                 static_cast<unsigned long>(GetLastError()));
  }
#endif
}

}  // namespace

extern "C" {

int vm_sam3d_built(void) { return 1; }

void* vm_sam3d_load(const char* path, int use_cuda, char* error, size_t error_len) {
  try {
    auto handle = std::make_unique<Handle>();
    // No CUDA headers here on purpose: the torch wheel's dispatcher loads
    // `torch_cuda.dll` on its own, so the device is attempted at run time and a
    // missing GPU simply leaves the module on the CPU.
    //
    // `load(path, device)` is required rather than `module.to(device)`: the
    // traced graph carries constant tensors as attributes, and only the loader
    // moves those too. A half-moved module either runs on the CPU with device
    // copies or throws.
    if (use_cuda != 0) {
      ensure_cuda_backend();
      try {
        handle->module = torch::jit::load(path, torch::kCUDA);
        handle->device = torch::Device(torch::kCUDA);
      } catch (const std::exception&) {
        handle->module = torch::jit::load(path);
        handle->device = torch::kCPU;
      }
    } else {
      handle->module = torch::jit::load(path);
      handle->device = torch::kCPU;
    }
    handle->module.eval();
    // A traced graph this size (the full SAM pipeline is ~5 GB of constants)
    // spends minutes in the profiling executor's optimisation pass the second
    // time it runs, and the app calls it once per frame. Turning the graph
    // optimiser off keeps every call on the interpreted plan, which is what the
    // per-frame path needs.
    torch::jit::setGraphExecutorOptimize(false);
    return handle.release();
  } catch (const std::exception& failure) {
    fail(error, error_len, std::string("load failed: ") + failure.what());
    return nullptr;
  }
}

void vm_sam3d_free(void* handle) { delete static_cast<Handle*>(handle); }

int vm_sam3d_forward(void* opaque, const float* image, const float* bbox_center,
                     const float* bbox_scale, const float* cam_int,
                     const float* keypoints, const float* ori_size,
                     float* body_token, float* verts, float* skel_state,
                     float* pred, float* pred_cam, float* model_params,
                     char* error, size_t error_len) {
  if (opaque == nullptr) {
    fail(error, error_len, "null module handle");
    return 1;
  }
  auto* handle = static_cast<Handle*>(opaque);
  try {
    torch::NoGradGuard guard;
    auto device = [&](torch::Tensor tensor) { return tensor.to(handle->device); };

    auto image_t = device(from(image, {1, 3, VM_SAM3D_IMAGE, VM_SAM3D_IMAGE}));
    auto center_t = device(from(bbox_center, {1, 2}));
    auto scale_t = device(from(bbox_scale, {1, 2}));
    auto cam_t = device(from(cam_int, {1, 3, 3}));
    auto keypoints_t = device(from(keypoints, {1, 1, 3}));
    auto ori_t = device(from(ori_size, {1, 2}));

    std::vector<torch::jit::IValue> inputs{image_t, center_t, scale_t, cam_t,
                                           keypoints_t, ori_t};
    auto output = handle->module.forward(inputs);
    auto tuple = output.toTuple()->elements();
    if (tuple.size() != 6) {
      fail(error, error_len, "unexpected output arity");
      return 2;
    }
    copy_out(tuple[0].toTensor(), body_token);
    copy_out(tuple[1].toTensor(), verts);
    copy_out(tuple[2].toTensor(), skel_state);
    copy_out(tuple[3].toTensor(), pred);
    copy_out(tuple[4].toTensor(), pred_cam);
    copy_out(tuple[5].toTensor(), model_params);
    return 0;
  } catch (const std::exception& failure) {
    fail(error, error_len, std::string("forward failed: ") + failure.what());
    return 3;
  }
}

namespace {

// Copy one output into a caller buffer, refusing to overflow it.
bool copy_checked(const torch::jit::IValue& value, float* out, int count,
                  const char* name, std::string& problem) {
  if (out == nullptr) {
    problem = std::string(name) + " buffer is null";
    return false;
  }
  const auto tensor = value.toTensor();
  const int64_t size = tensor.numel();
  if (size > count) {
    problem = std::string(name) + ": module produced " + std::to_string(size) +
              " values but the buffer holds " + std::to_string(count);
    return false;
  }
  copy_out(tensor, out);
  return true;
}

}  // namespace

int vm_sam3d_forward_full(void* opaque, const float* frame, int height, int width,
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
  if (opaque == nullptr) {
    fail(error, error_len, "null module handle");
    return 1;
  }
  if (height <= 0 || width <= 0) {
    fail(error, error_len, "empty frame");
    return 1;
  }
  auto* handle = static_cast<Handle*>(opaque);
  try {
    torch::NoGradGuard guard;
    auto device = [&](torch::Tensor tensor) { return tensor.to(handle->device); };

    auto frame_t = device(from(frame, {1, 3, height, width}));
    auto center_t = device(from(bbox_center, {1, 2}));
    auto scale_t = device(from(bbox_scale, {1, 2}));
    auto cam_t = device(from(cam_int, {1, 3, 3}));
    auto size_t = device(torch::zeros({1, 2}, torch::kFloat32));
    size_t[0][0] = static_cast<float>(width);
    size_t[0][1] = static_cast<float>(height);
    size_t = size_t.to(handle->device);

    std::vector<torch::jit::IValue> inputs{frame_t, center_t, scale_t, cam_t, size_t};
    auto output = handle->module.forward(inputs);
    auto tuple = output.toTuple()->elements();
    if (tuple.size() != 14) {
      fail(error, error_len, "unexpected output arity");
      return 2;
    }
    std::string problem;
    const struct {
      int index;
      float* out;
      int count;
      const char* name;
    } copies[] = {
        {0, verts, verts_count, "verts"},
        {1, keypoints_3d, kps3d_count, "keypoints_3d"},
        {2, keypoints_2d, kps2d_count, "keypoints_2d"},
        {3, pred_cam, 3, "pred_cam"},
        {4, pred_cam_t, 3, "pred_cam_t"},
        {5, focal_length, 1, "focal_length"},
        {6, model_params, params_count, "model_params"},
        {7, joint_rots, rots_count, "joint_rots"},
        {8, hand_valid, hand_count, "hand_valid"},
        {9, body_token, token_count, "body_token"},
        {10, left_token, token_count, "left_token"},
        {11, right_token, token_count, "right_token"},
        {12, skel_state, skel_count, "skel_state"},
        {13, pred, pred_count, "pred"},
    };
    for (const auto& item : copies) {
      if (!copy_checked(tuple[item.index], item.out, item.count, item.name,
                        problem)) {
        fail(error, error_len, problem);
        return 4;
      }
    }
    return 0;
  } catch (const std::exception& failure) {
    fail(error, error_len, std::string("full forward failed: ") + failure.what());
    return 3;
  }
}

}  // extern "C"
