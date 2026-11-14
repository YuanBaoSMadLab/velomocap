"""Dump the FlowHMR reference tensors as flat float32 for the Rust parity harness.

    python dump_flowhmr_reference.py D:\\onnx-out
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

PATCH = 16


def narrowband_mask(motion_len, ctxt_len, mm, mc, cm, cc):
    neg = np.float32("-inf")

    def band(q, k, window):
        if window < 0:
            return np.full((q, k), neg, dtype=np.float32)
        qi = np.arange(q)[:, None]
        ki = np.arange(k)[None, :]
        return np.where(np.abs(qi - ki) <= window, np.float32(0.0), neg).astype(np.float32)

    return np.concatenate(
        [
            np.concatenate([band(motion_len, motion_len, mm), band(motion_len, ctxt_len, mc)], 1),
            np.concatenate([band(ctxt_len, motion_len, cm), band(ctxt_len, ctxt_len, cc)], 1),
        ],
        0,
    )


def main() -> int:
    out = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
    meta_path = os.path.join(out, "flowhmr_mmdit.onnx.meta.json")
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    x = np.load(os.path.join(out, "fh_x.npy")).astype(np.float32)
    cam = np.load(os.path.join(out, "fh_cam.npy")).astype(np.float32)
    feature = np.load(os.path.join(out, "fh_feature.npy")).astype(np.float32)
    pred = np.load(os.path.join(out, "fh_pred.npy")).astype(np.float32)

    _, t, input_dim = x.shape
    nb = meta["narrowband"]
    mask = narrowband_mask(t, t, nb["mm"], nb["mc"], nb["cm"], nb["cc"]).reshape(1, 1, 2 * t, 2 * t)
    vtxt = np.zeros((1, 1, meta["vtxt_input_dim"]), dtype=np.float32)
    timestep = np.array([0.5], dtype=np.float32)
    valid = np.ones((1, t), dtype=bool)

    arrays = {
        "x.bin": x,
        "cam.bin": cam,
        "feature.bin": feature,
        "vtxt.bin": vtxt,
        "timestep.bin": timestep,
        "valid.bin": valid.astype(np.uint8),
        "mask.bin": mask,
        "pred.bin": pred,
    }
    for name, array in arrays.items():
        if array.dtype in (np.bool_, np.uint8):
            array.astype(np.uint8).tofile(os.path.join(out, name))
        else:
            np.ascontiguousarray(array, dtype="<f4").tofile(os.path.join(out, name))
        print(f"{name:14s} {array.shape} {array.dtype}")

    dims = {
        "motion_len": int(t),
        "ctxt_len": int(t),
        "input_dim": int(input_dim),
        "feature_dim": int(feature.shape[-1]),
        "vtxt_dim": int(vtxt.shape[-1]),
        "narrowband": nb,
    }
    with open(os.path.join(out, "fh_dims.json"), "w", encoding="utf-8") as fh:
        json.dump(dims, fh, indent=2)
    print(json.dumps(dims, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
