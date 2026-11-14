"""Run a TorchScript module on deterministic inputs and print reference values.

The C++ test builds the *same* inputs from the same recipe, so the numbers can be
compared by hand without shipping tensors across the boundary.

    python tools/torch/ts_probe.py MODEL.pt sam
    python tools/torch/ts_probe.py MODEL.pt flow
"""

from __future__ import annotations

import sys

import torch

IMAGE_SIZE = 512


def pattern(n: int, scale: int = 251) -> torch.Tensor:
    """The exact recipe the C++ probe uses: `(i % scale) / scale`."""
    index = torch.arange(n, dtype=torch.float32)
    return (torch.remainder(index, float(scale)) / float(scale)).reshape(n)


def sam_inputs():
    pixels = IMAGE_SIZE * IMAGE_SIZE
    image = torch.empty(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)
    flat = pattern(3 * pixels)
    image.copy_(flat.reshape(1, 3, IMAGE_SIZE, IMAGE_SIZE))
    bbox_center = torch.tensor([[640.0, 400.0]])
    bbox_scale = torch.tensor([[250.0, 375.0]])
    cam_int = torch.tensor(
        [[[1100.0, 0.0, 640.0], [0.0, 1100.0, 360.0], [0.0, 0.0, 1.0]]]
    )
    keypoints = torch.zeros(1, 1, 3)
    keypoints[:, :, -1] = -2
    ori = torch.tensor([[1280.0, 720.0]])
    return (image, bbox_center, bbox_scale, cam_int, keypoints, ori)


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    path, kind = sys.argv[1], sys.argv[2]
    module = torch.jit.load(path)
    module.eval()
    with torch.no_grad():
        if kind == "sam":
            out = module(*sam_inputs())
            names = ("body_token", "verts", "skel_state", "pred", "pred_cam", "model_params")
            for name, value in zip(names, out):
                flat = value.reshape(-1)
                print(
                    f"{name:12s} shape {tuple(value.shape)} "
                    f"first4 {[round(float(v), 6) for v in flat[:4]]} "
                    f"sum {float(flat.sum()):.4f}"
                )
        else:
            print("unknown kind", kind)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
