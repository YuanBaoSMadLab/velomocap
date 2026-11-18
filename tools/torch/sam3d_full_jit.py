"""Trace `sam3d_full.Sam3dFull` into one TorchScript module.

    python tools/torch/sam3d_full_jit.py --ckpt ... --mhr ... --config ... --out ...

The graph takes the original frame plus the person box and produces the fused
mesh, the keypoints, the camera and the model parameters — the whole
`run_inference("full")` path, with the hand crops, the four fusion criteria and
the keypoint prompt inside the graph. `tools/torch/sam3d_full_check.py` is what
pins it to the reference before this ever runs.
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

from sam3d_split import load_model  # noqa: E402
from sam3d_full import Sam3dFull  # noqa: E402

NAMES = (
    "verts",
    "kps3d",
    "kps2d",
    "pred_cam",
    "pred_cam_t",
    "focal",
    "model_params",
    "joint_rots",
    "hand_valid",
    "body_token",
    "left_token",
    "right_token",
    "skel_state",
    "pred",
)


def sample_inputs(seed=0, height=720, width=1280, batch=1):
    torch.manual_seed(seed)
    # Each item is one person in one frame: its own frame, box and intrinsics,
    # so a batch exercises the per-item paths instead of repeating one crop.
    frames = [torch.rand(1, 3, height, width) for _ in range(batch)]
    frame = torch.cat(frames, dim=0)
    centers = []
    scales = []
    cameras = []
    for index in range(batch):
        shift = 40.0 * index
        centers.append([600.0 - shift, 395.0 + 6.0 * index])
        scales.append([700.0 + 11.0 * index, 700.0 + 11.0 * index])
        cameras.append(
            [
                [1400.0 + index, 0.0, 640.0 - shift],
                [0.0, 1400.0 + index, 360.0 + index],
                [0.0, 0.0, 1.0],
            ]
        )
    bbox_center = torch.tensor(centers)
    bbox_scale = torch.tensor(scales)
    cam_int = torch.tensor(cameras)
    frame_size = torch.tensor([[float(width), float(height)]] * batch)
    return frame, bbox_center, bbox_scale, cam_int, frame_size


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=r"E:\RGBMocap\export\sam-3d-body")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--mhr", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--backbone-dtype",
        choices=("bf16", "fp32"),
        default="bf16",
        help="precision the ViT runs at (the app uses bf16; fp32 is 2x slower "
        "and twice the file but less sensitive to crop-sampling differences)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1,
                        help="Items in the traced batch (one person in one frame "
                             "per item). The tracer bakes it, so a batch-8 module "
                             "expects exactly 8 items per call.")
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    args = parser.parse_args()

    print("[1/4] loading model ...", flush=True)
    model = load_model(args.repo, args.ckpt, args.mhr, args.config)
    # The module runs on whatever device it is traced on: the tracer bakes the
    # constant tensors into the graph, so a CPU trace stays a CPU graph even if
    # it is later loaded with `kCUDA`.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # The app runs the ViT in bfloat16; trace in the same regime.
    backbone_dtype = torch.bfloat16 if args.backbone_dtype == "bf16" else None
    if backbone_dtype is not None:
        model.backbone_dtype = backbone_dtype
        model.backbone.to(backbone_dtype)
    print(f"      device {device}, backbone {backbone_dtype}", flush=True)

    print("[2/4] building the module ...", flush=True)
    module = Sam3dFull(model, backbone_dtype=backbone_dtype).eval().to(device)
    inputs = tuple(t.to(device) for t in sample_inputs(args.seed, args.height, args.width, args.batch))

    print("[3/4] reference forward (eager) ...", flush=True)
    started = time.time()
    with torch.no_grad():
        want = module(*inputs)
    print(f"      {time.time() - started:.1f}s", flush=True)
    for name, value in zip(NAMES, want):
        print(f"      {name:12s} {tuple(value.shape)} "
              f"first {value.flatten()[:3].tolist()}", flush=True)

    print("[4/4] tracing ...", flush=True)
    started = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(module, inputs, check_trace=False)
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
        # `kps2d` is a projection of ~700 px coordinates, and the hand tokens go
        # through six decoder layers on top of a bf16 backbone whose batching
        # differs between the eager and traced graphs, so both carry more noise
        # than the metres-scale outputs.
        if name == "kps2d":
            limit = 1e-3
        elif name.endswith("token"):
            limit = 5e-2
        else:
            limit = 1e-4
        print(f"      reload {name:12s} max abs {value:.3e}", flush=True)
        ok = ok and value < limit
    print("TORCHSCRIPT", "OK" if ok else "MISMATCH", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
