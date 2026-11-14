"""Consolidate an ONNX model's external weights into a single `.bin` sidecar.

`torch.onnx.export(..., external_data=True)` writes one file per tensor, which
is correct but unusable for shipping. ONNX Runtime wants the graph plus one
weights file, so repack before publishing.

    python pack_onnx.py model.onnx model.packed.onnx
"""

from __future__ import annotations

import os
import shutil
import sys

import onnx
import onnx.checker


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    src, dst = (os.path.abspath(p) for p in sys.argv[1:3])

    src_dir = os.path.dirname(src)
    before = {name for name in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, name))}

    print(f"loading {src} ...", flush=True)
    model = onnx.load(src, load_external_data=True)
    print(f"  {len(model.graph.node)} nodes, {len(model.graph.initializer)} initializers", flush=True)

    stem = os.path.splitext(os.path.basename(dst))[0]
    location = f"{stem}.weights.bin"
    out_dir = os.path.dirname(dst)
    os.makedirs(out_dir, exist_ok=True)

    print(f"saving {dst} + {location} ...", flush=True)
    onnx.save_model(
        model,
        dst,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=location,
        size_threshold=1024,
        convert_attribute=False,
    )

    graph_bytes = os.path.getsize(dst)
    weights_bytes = os.path.getsize(os.path.join(out_dir, location))
    print(f"  graph   {graph_bytes / 1e6:.2f} MB")
    print(f"  weights {weights_bytes / 1e9:.2f} GB ({location})")

    if os.path.normcase(src_dir) == os.path.normcase(out_dir):
        keep = {os.path.basename(dst), location}
        removed = 0
        for name in sorted(before - keep):
            path = os.path.join(src_dir, name)
            if name.startswith(os.path.basename(src)):
                os.remove(path)
                removed += 1
        print(f"  removed {removed} loose tensor files")
    else:
        leftovers = os.path.join(out_dir, f"{stem}.onnx.data")
        if os.path.exists(leftovers):
            shutil.rmtree(leftovers, ignore_errors=True)

    onnx.checker.check_model(dst)
    print("checker ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
