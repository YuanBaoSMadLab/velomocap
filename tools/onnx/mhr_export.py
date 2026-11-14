"""Export the scripted MHR body model (`mhr_model.pt`) to ONNX.

MHR is the last piece of SAM 3D Body that torch's ONNX exporter cannot lower on
its own. Exporting it alone is both useful as a reference and the prerequisite
for running it as a separate graph, and it exercises `mhr_symbolics` in seconds
instead of the minutes a full SAM export takes.

    python tools/onnx/mhr_export.py \
        --mhr models/flowhmr/sam/assets/mhr_model.pt \
        --out export/sam3d/mhr_model.onnx

Inputs:  identity [B,45], model_params [B,204], face_expr [B,72]
Outputs: verts [B,18439,3], skel_state [B,127,8]  (translation + quaternion +
         scale per joint, in centimetres)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

IMAGE_SIZE_HINT = 18439


class _Wrap(nn.Module):
    """Plain container, so `torch.jit.trace` descends into the scripted module."""

    def __init__(self, mhr):
        super().__init__()
        self.mhr = mhr
        for param in self.mhr.parameters():
            param.requires_grad_(False)

    def forward(self, identity, model_params, face):
        return self.mhr(identity, model_params, face)


def export(mhr_path: str, out_path: str, opset: int = 17, batch: int = 1) -> int:
    import mhr_symbolics
    from onnx_export_safe import patch_safe_shape_inference, report

    torch.set_grad_enabled(False)
    mhr_symbolics.register()
    patch_safe_shape_inference()

    print("[1/4] loading MHR ...", flush=True)
    mhr = torch.jit.load(mhr_path, map_location="cpu").eval()
    model = _Wrap(mhr).eval()

    identity = torch.zeros(batch, 45)
    model_params = torch.zeros(batch, 204)
    face = torch.zeros(batch, 72)

    print("[2/4] reference forward ...", flush=True)
    reference = model(identity, model_params, face)
    print(
        "      " + ", ".join(str(tuple(t.shape)) for t in reference),
        flush=True,
    )

    print("[3/4] tracing + exporting ...", flush=True)
    started = time.time()
    traced = torch.jit.trace(model, (identity, model_params, face), check_trace=False)
    traced = torch.jit.freeze(traced)

    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)

    try:
        torch.onnx.export(
            traced,
            (identity, model_params, face),
            out_path,
            input_names=["identity", "model_params", "face"],
            output_names=["verts", "skel_state"],
            dynamic_axes={
                "identity": {0: "batch"},
                "model_params": {0: "batch"},
                "face": {0: "batch"},
            },
            opset_version=opset,
            do_constant_folding=True,
            external_data=True,
            dynamo=False,
        )
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1
    print(f"      exported in {time.time() - started:.1f}s", flush=True)
    report()

    print("[4/4] verifying against PyTorch ...", flush=True)
    import onnxruntime as ort

    generator = torch.Generator().manual_seed(7)
    identity = torch.randn(batch, 45, generator=generator) * 0.5
    model_params = torch.randn(batch, 204, generator=generator) * 0.1
    face = torch.zeros(batch, 72)
    expected = model(identity, model_params, face)

    session = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    names = [o.name for o in session.get_outputs()]
    got = session.run(
        names,
        {
            "identity": identity.numpy(),
            "model_params": model_params.numpy(),
            "face": face.numpy(),
        },
    )
    ok = True
    for name, want, have in zip(names, expected, got):
        diff = abs(have - want.numpy())
        print(
            f"      {name}: max abs {diff.max():.3e}  mean {diff.mean():.3e}",
            flush=True,
        )
        ok = ok and diff.max() < 1e-3
    if not ok:
        print("      parity FAILED")
        return 1
    print(f"      wrote {out_path} ({os.path.getsize(out_path)} bytes)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mhr", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()
    return export(args.mhr, args.out, args.opset, args.batch)


if __name__ == "__main__":
    raise SystemExit(main())
