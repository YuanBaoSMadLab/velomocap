"""Run one binary case through the *traced* module and print its outputs.

The counterpart of `velomocap-cli sam-full-torch --case`: both read
`case.bin` (written by `dump_sam3d_full_case.py`) and print the same fields, so
the Python and C++ paths can be compared value for value without loading the
2 GB checkpoint.

    python tools/torch/check_traced_case.py --module ... --case DIR
"""

from __future__ import annotations

import argparse
import os
import struct
import sys

import numpy as np
import torch

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--case", required=True)
    args = parser.parse_args()

    raw = np.fromfile(os.path.join(args.case, "case.bin"), dtype=np.uint8)
    width, height = struct.unpack_from("<ii", raw, 0)
    at = 8
    center = np.frombuffer(raw[at : at + 8], dtype="<f4").copy()
    at += 8
    scale = np.frombuffer(raw[at : at + 8], dtype="<f4").copy()
    at += 8
    cam = np.frombuffer(raw[at : at + 36], dtype="<f4").copy()
    at += 36
    frame = np.frombuffer(raw[at:], dtype="<f4").copy()
    print(f"case {width}x{height} center {center.tolist()} scale {scale.tolist()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = torch.jit.load(args.module, map_location=device).eval()
    inputs = (
        torch.from_numpy(frame).view(1, 3, height, width).to(device),
        torch.from_numpy(center).view(1, 2).to(device),
        torch.from_numpy(scale).view(1, 2).to(device),
        torch.from_numpy(cam).view(1, 3, 3).to(device),
        torch.tensor([[float(width), float(height)]], device=device),
    )
    with torch.no_grad():
        out = module(*inputs)
    for name, value in zip(NAMES, out):
        flat = value.flatten()
        print(f"{name:12s} first4 [{', '.join(f'{float(v):.6f}' for v in flat[:4])}] "
              f"sum {float(flat.sum()):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
