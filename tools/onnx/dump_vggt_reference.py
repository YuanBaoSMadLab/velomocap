"""Convert the saved VGGT-Omega reference .npy tensors into flat float32 dumps.

The Rust parity harness (`velomocap vggt-parity <dir>`) reads these so it can
compare ONNX Runtime against PyTorch without any Python at run time.

    python tools/dump_vggt_reference.py D:\\onnx-out
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

PATCH = 16


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    out = os.path.abspath(out)

    images = np.load(os.path.join(out, "ref_images.npy")).astype(np.float32)
    rope_sin = np.load(os.path.join(out, "ref_rope_sin.npy")).astype(np.float32)
    rope_cos = np.load(os.path.join(out, "ref_rope_cos.npy")).astype(np.float32)
    pose_enc = np.load(os.path.join(out, "ref_pose_enc.npy")).astype(np.float32)

    _, n, _, h, w = images.shape
    tokens = rope_sin.shape[0]
    dims = {
        "frames": int(n),
        "height": int(h),
        "width": int(w),
        "h_patches": int(h // PATCH),
        "w_patches": int(w // PATCH),
        "tokens": int(tokens),
        "head_dim": int(rope_sin.shape[1]),
        "pose_dim": int(pose_enc.shape[-1]),
    }
    assert tokens == dims["h_patches"] * dims["w_patches"], (tokens, dims)

    for name, array in (
        ("images.bin", images),
        ("rope_sin.bin", rope_sin),
        ("rope_cos.bin", rope_cos),
        ("pose_enc.bin", pose_enc),
    ):
        path = os.path.join(out, name)
        np.ascontiguousarray(array, dtype="<f4").tofile(path)
        print(f"{name:16s} {array.shape} {array.size * 4} bytes")

    with open(os.path.join(out, "dims.json"), "w", encoding="utf-8") as fh:
        json.dump(dims, fh, indent=2)
    print(json.dumps(dims, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
