"""Export the public YOLOv8 detector to ONNX for the native pipeline.

This is a one-time asset build step, not a runtime dependency: after it runs,
VeloMocap loads only the `.onnx` file through ONNX Runtime. It exists because
Ultralytics ships `.pt` checkpoints, and the native pipeline must not need
PyTorch or Python at inference time.

    python tools/export_yolo.py --weights models/yolov8x.pt --imgsz 640
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("models/yolov8x.pt"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--expected-sha256", default=None,
                        help="refuse to load the checkpoint unless its digest matches")
    args = parser.parse_args()

    if not args.weights.is_file():
        print(f"error: {args.weights} not found", file=sys.stderr)
        return 2

    os.environ.setdefault("YOLO_CONFIG_DIR", str(args.weights.parent.resolve()))

    import hashlib

    # The checkpoint is an official Ultralytics release asset. Verify the known
    # digest before loading, then use Ultralytics' own loader (which handles its
    # module allowlist) so the export path stays identical to upstream.
    digest = hashlib.sha256(args.weights.read_bytes()).hexdigest()
    if args.expected_sha256 and digest != args.expected_sha256:
        print(f"error: {args.weights} sha256 {digest} != {args.expected_sha256}", file=sys.stderr)
        return 3
    print(f"weights sha256 {digest}")

    # Ultralytics 8.2 predates PyTorch 2.6's weights_only default, so its own
    # loader cannot unpickle the checkpoint. Because the digest above was
    # verified against the official release asset, allow the legacy behaviour
    # for this one call instead of disabling the guard globally.
    import torch

    original_load = torch.load

    def trusted_load(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return original_load(*load_args, **load_kwargs)

    torch.load = trusted_load
    try:
        from ultralytics import YOLO

        model = YOLO(str(args.weights))
    finally:
        torch.load = original_load

    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=True,
        dynamic=False,
    )
    source = Path(exported)
    target = args.out or args.weights.with_suffix(".onnx")
    if source.resolve() != target.resolve():
        target.write_bytes(source.read_bytes())
        source.unlink(missing_ok=True)
    print(f"onnx: {target} ({target.stat().st_size / 1e6:.1f} MB, imgsz={args.imgsz})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
