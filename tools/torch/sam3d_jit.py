"""Export the SAM 3D Body *body* branch as one TorchScript module.

The ONNX split exists because ONNX cannot carry a node whose input is the empty
string, which is what `params_from_pred` traces into. TorchScript has no such
limitation, so the whole chain — backbone, mask prompt, prompt encoder, decoder,
the six refinement rounds, the MHR body and the final pose — becomes a single
module that the C++ side loads with LibTorch.

    python tools/torch/sam3d_jit.py --ckpt ... --mhr ... --config ... --out ...

Inputs (all `float32`, batch 1):

    image        [1, 3, 512, 512]   the crop, ImageNet-normalised inside
    bbox_center  [1, 2]
    bbox_scale   [1, 2]             already padded by the caller
    cam_int      [1, 3, 3]
    keypoints    [1, 1, 3]          cx, cy, label (-2 = no prompt)
    ori_img_size [1, 2]

Outputs:

    body_token   [1, 1024]          token 0 of the normalised decoder output
    verts        [1, 18439, 3]      MHR output, centimetres, model axes
    skel_state   [1, 127, 8]
    pred         [1, 519]
    pred_cam     [1, 3]
    model_params [1, 204]

`sam3d_body.BodyPass` is the module; this file only wraps it in the six-output
signature the C ABI already expects. `tools/torch/sam3d_body_check.py` pins it
to the original `forward_step(batch, "body")` — the earlier export was only ever
compared against itself and was missing the mask-prompt term.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "onnx"))
sys.path.insert(0, HERE)

from sam3d_split import IMAGE_SIZE, load_model  # noqa: E402
from sam3d_body import BodyPass, params_from_pred_body  # noqa: E402


class WholeBody(nn.Module):
    """`forward_pose_branch` for one person, batch 1, body decoder only."""

    def __init__(self, model):
        super().__init__()
        self.pass_ = BodyPass(model)
        self.mhr = model.head_pose.mhr
        self.head = model.head_pose

    def forward(self, image, bbox_center, bbox_scale, cam_int, keypoints, ori_img_size):
        normed, pred, pred_cam, _, _, _, _, _ = self.pass_.run(
            image, bbox_center, bbox_scale, cam_int, keypoints
        )
        shape, model_params, face = params_from_pred_body(self.head, pred)
        verts, skel_state = self.mhr(shape, model_params, face)
        return normed[:, 0], verts, skel_state, pred, pred_cam, model_params


def sample_inputs(seed=0):
    torch.manual_seed(seed)
    bbox_center = torch.tensor([[640.0, 400.0]])
    bbox_scale = torch.tensor([[200.0, 300.0]]) * 1.25
    cam_int = torch.tensor(
        [[[1100.0, 0.0, 640.0], [0.0, 1100.0, 360.0], [0.0, 0.0, 1.0]]]
    )
    keypoints = torch.zeros(1, 1, 3)
    keypoints[:, :, -1] = -2
    return (
        torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE),
        bbox_center,
        bbox_scale,
        cam_int,
        keypoints,
        torch.tensor([[1280.0, 720.0]]),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=r"E:\RGBMocap\export\sam-3d-body")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--mhr", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("[1/4] loading model ...", flush=True)
    model = load_model(args.repo, args.ckpt, args.mhr, args.config)

    print("[2/4] building the whole-body module ...", flush=True)
    whole = WholeBody(model).eval()
    inputs = sample_inputs(args.seed)
    print("      token layout", whole.pass_.layout, flush=True)

    print("[3/4] reference forward (eager) ...", flush=True)
    started = time.time()
    with torch.no_grad():
        want = whole(*inputs)
    print(f"      done in {time.time() - started:.1f}s", flush=True)

    print("[4/4] tracing ...", flush=True)
    started = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(whole, inputs, check_trace=False)
    torch.jit.save(traced, args.out)
    print(
        f"      traced in {time.time() - started:.1f}s -> {args.out} "
        f"({os.path.getsize(args.out) / 1e9:.2f} GB)",
        flush=True,
    )

    loaded = torch.jit.load(args.out)
    loaded.eval()
    with torch.no_grad():
        got = loaded(*inputs)
    names = ("body_token", "verts", "skel_state", "pred", "pred_cam", "model_params")
    ok = True
    for name, a, b in zip(names, want, got):
        diff = (a - b).abs().max().item()
        print(f"      reload {name:12s} {tuple(a.shape)} max abs {diff:.3e}", flush=True)
        ok = ok and diff < 1e-4
    print("TORCHSCRIPT", "OK" if ok else "MISMATCH", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
