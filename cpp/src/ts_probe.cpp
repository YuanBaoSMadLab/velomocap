// Minimal LibTorch smoke test: load a TorchScript module and run one forward.
//
// This exists to prove the C++ side can drive PyTorch directly, which is the
// whole point of dropping the ONNX split: no graph surgery, no parameter
// tables, one file per model.
//
//   cl /std:c++17 /EHsc /MD /O2 ts_probe.cpp /I<torch>/include
//      /I<torch>/include/torch/csrc/api/include /link /LIBPATH:<torch>/lib
//      torch_cpu.lib torch.lib c10.lib
//
// The inputs are built from the same recipe as `tools/torch/ts_probe.py`, so
// the printed values can be compared one to one.

#include <torch/script.h>

#include <cstdio>
#include <string>
#include <vector>

namespace {

constexpr long kImageSize = 512;

torch::Tensor pattern(long count) {
  std::vector<float> values(static_cast<size_t>(count));
  for (long i = 0; i < count; ++i) {
    values[static_cast<size_t>(i)] = static_cast<float>(i % 251) / 251.0f;
  }
  return torch::from_blob(values.data(), {count}, torch::kFloat32).clone();
}

void report(const char* name, const torch::Tensor& value) {
  auto flat = value.contiguous().reshape({-1});
  auto head = flat.slice(0, 0, 4);
  auto accessor = head.accessor<float, 1>();
  double sum = 0.0;
  auto all = flat.accessor<float, 1>();
  for (long i = 0; i < all.size(0); ++i) {
    sum += all[i];
  }
  std::printf("%-12s shape [", name);
  for (long i = 0; i < value.dim(); ++i) {
    std::printf("%lld%s", static_cast<long long>(value.size(i)),
                i + 1 == value.dim() ? "" : ", ");
  }
  std::printf("] first4 [%.6f, %.6f, %.6f, %.6f] sum %.4f\n", accessor[0],
              accessor[1], accessor[2], accessor[3], sum);
}

int run_sam(torch::jit::Module& module) {
  const long pixels = kImageSize * kImageSize;
  auto image = pattern(3 * pixels).reshape({1, 3, kImageSize, kImageSize});

  auto bbox_center = torch::tensor({{640.0f, 400.0f}});
  auto bbox_scale = torch::tensor({{250.0f, 375.0f}});
  auto cam_int = torch::tensor({{{1100.0f, 0.0f, 640.0f},
                                 {0.0f, 1100.0f, 360.0f},
                                 {0.0f, 0.0f, 1.0f}}});
  auto keypoints = torch::zeros({1, 1, 3});
  keypoints.index_put_({0, 0, 2}, -2.0f);
  auto ori = torch::tensor({{1280.0f, 720.0f}});

  std::vector<torch::jit::IValue> inputs{image, bbox_center, bbox_scale, cam_int,
                                         keypoints, ori};
  auto output = module.forward(inputs);
  auto tuple = output.toTuple()->elements();
  const char* names[] = {"body_token",  "verts", "skel_state",
                         "pred",        "pred_cam", "model_params"};
  for (size_t i = 0; i < tuple.size(); ++i) {
    report(names[i], tuple[i].toTensor());
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::printf("usage: ts_probe <model.pt> <sam>\n");
    return 2;
  }
  const std::string path = argv[1];
  const std::string kind = argv[2];
  try {
    std::printf("loading %s\n", path.c_str());
    torch::jit::Module module = torch::jit::load(path);
    module.eval();
    torch::NoGradGuard guard;
    if (kind == "sam") {
      return run_sam(module);
    }
    std::printf("unknown kind %s\n", kind.c_str());
    return 2;
  } catch (const std::exception& error) {
    std::printf("failed: %s\n", error.what());
    return 1;
  }
}
