"""Write the binary case folder that `velomocap-cli sam-full-torch --case` reads.

It reproduces *exactly* the inputs `sam3d_full_check.py` uses for its reference
run, without loading the 2 GB checkpoint: only the numpy transform stack is
needed to get `bbox_center` / `bbox_scale` out of `prepare_batch`.

Layout (little endian): i32 width, i32 height, f32 centre[2], f32 scale[2],
f32 cam_int[9], f32 frame[3*h*w] in NCHW and 0..1.

    python tools/torch/dump_sam3d_full_case.py --repo ... --out DIR
"""

from __future__ import annotations

import argparse
import os
import struct
import sys

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=r"E:\RGBMocap\export\sam-3d-body")
    parser.add_argument("--out", required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    sys.path.insert(0, args.repo)
    import torch

    from sam_3d_body.data.transforms import (
        Compose,
        GetBBoxCenterScale,
        TopdownAffine,
        VisionTransformWrapper,
    )
    from sam_3d_body.data.utils.prepare_batch import prepare_batch
    from torchvision.transforms import ToTensor

    transform = Compose(
        [
            GetBBoxCenterScale(padding=1.25),
            TopdownAffine(input_size=(512, 512), use_udp=False),
            VisionTransformWrapper(ToTensor()),
        ]
    )

    # Identical to `sam3d_full_check.py`.
    rng = np.random.default_rng(1)
    image = (rng.random((args.height, args.width, 3)) * 255).astype(np.uint8)
    box = np.array([[420.0, 90.0, 860.0, 700.0]], dtype=np.float32)
    cam_int = torch.tensor(
        [[[1400.0, 0.0, 640.0], [0.0, 1400.0, 360.0], [0.0, 0.0, 1.0]]]
    )
    batch = prepare_batch(image, transform, box, cam_int=cam_int)
    center = batch["bbox_center"][0, 0].cpu().numpy()
    scale = batch["bbox_scale"][0, 0].cpu().numpy()
    print("bbox_center", center.tolist())
    print("bbox_scale ", scale.tolist())

    frame = (image.astype(np.float32) / 255.0).transpose(2, 0, 1).astype("<f4")

    os.makedirs(args.out, exist_ok=True)
    payload = struct.pack("<ii", args.width, args.height)
    payload += center.astype("<f4").tobytes()
    payload += scale.astype("<f4").tobytes()
    payload += cam_int[0].numpy().astype("<f4").tobytes()
    payload += frame.tobytes()
    path = os.path.join(args.out, "case.bin")
    with open(path, "wb") as fh:
        fh.write(payload)
    print(f"wrote {path} ({len(payload) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
