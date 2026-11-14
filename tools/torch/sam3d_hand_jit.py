"""Export the SAM 3D Body hand branch as its own TorchScript module.

The full pipeline (`sam3d_full_jit.py`) is what the app needs; this standalone
module is for measuring the hand decoder on its own. The maths lives in
`sam3d_hand.py` — this file only traces it and checks the reload.

    python tools/torch/sam3d_hand_jit.py --ckpt ... --mhr ... --config ... --out ...

Inputs: `image` [1,3,512,512], `bbox_center` [1,2], `bbox_scale` [1,2],
`cam_int` [1,3,3], `keypoints` [1,1,3], `ori_img_size` [1,2].

Outputs: `token` [1,1024], `pred` [1,519], `pred_cam` [1,3], `verts`
[1,18439,3] (metres, camera axes), `keypoints_3d` [1,70,3], `keypoints_2d`
[1,70,2], `depth` [1,70], `model_params` [1,204], `joint_rots` [1,127,3,3].
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "onnx"))
sys.path.insert(0, HERE)

from sam3d_split import IMAGE_SIZE, load_model  # noqa: E402
from sam3d_hand import HandBranch  # noqa: E402

NAMES = (
    "token",
    "pred",
    "pred_cam",
    "verts",
    "keypoints_3d",
    "keypoints_2d",
    "depth",
    "model_params",
    "joint_rots",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=r"E:\RGBMocap\export\sam-3d-body")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--mhr", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    print("[1/4] loading model ...", flush=True)
    model = load_model(args.repo, args.ckpt, args.mhr, args.config)

    print("[2/4] building the hand module ...", flush=True)
    branch = HandBranch(model).eval()
    torch.manual_seed(0)
    keypoints = torch.zeros(1, 1, 3)
    keypoints[:, :, -1] = -2
    inputs = (
        torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE),
        torch.tensor([[340.0, 220.0]]),
        torch.tensor([[96.0, 96.0]]),
        torch.tensor([[[935.0, 0.0, 320.0], [0.0, 935.0, 240.0], [0.0, 0.0, 1.0]]]),
        keypoints,
        torch.tensor([[640.0, 480.0]]),
    )

    print("[3/4] reference forward (eager) ...", flush=True)
    started = time.time()
    with torch.no_grad():
        want = branch(*inputs)
    print(f"      {time.time() - started:.1f}s", flush=True)

    print("[4/4] tracing ...", flush=True)
    started = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(branch, inputs, check_trace=False)
    torch.jit.save(traced, args.out)
    print(
        f"      traced in {time.time() - started:.1f}s -> {args.out} "
        f"({os.path.getsize(args.out) / 1e9:.2f} GB)",
        flush=True,
    )

    loaded = torch.jit.load(args.out).eval()
    with torch.no_grad():
        got = loaded(*inputs)
    ok = True
    for name, a, b in zip(NAMES, want, got):
        value = (a - b).abs().max().item()
        print(f"      reload {name:12s} max abs {value:.3e}", flush=True)
        ok = ok and value < 1e-4
    print("TORCHSCRIPT", "OK" if ok else "MISMATCH", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
