"""Write the model manifest of a *packaged* install.

The development tree carries every route we ever tried - the ONNX SAM chain, the
pose detector, SMPL-X, the gated checkpoints kept for conversion. A packaged
install carries only what the application loads at run time, and the manifest in
the repository describes the development tree, so checking a package against it
reports twenty files as missing that the package never needed.

This scans the package and writes a manifest that matches it: every file with
its size and SHA-256, grouped by what it is for, with the required/optional flag
each file really has in the packaged layout.

    python tools/make_package_manifest.py --models E:\\VeloMocap-Release\\VeloMocap\\models
    python tools/check_models.py --manifest <that>/models/manifest.json

The output is what `check_models.py` reads, so a package can be verified offline
without the development tree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# What each part of the package is for, and whether the app can start without it.
# `gated` marks files the user must obtain themselves (registration or account).
LAYOUT = {
    "detector": {
        "title": "Person detection",
        "required": True,
        "description": "Finds people in every frame. Required: nothing runs without it.",
        "gated": False,
    },
    "sam3d": {
        "title": "SAM 3D Body (body pose and mesh)",
        "required": True,
        "description": "The TorchScript pipeline plus the MHR mesh tables; the packaged "
                       "build runs SAM through LibTorch.",
        "gated": False,
    },
    "flowhmr": {
        "title": "FlowHMR motion model",
        "required": True,
        "description": "Turns the tracked bodies into a motion clip and normalises it.",
        "gated": False,
    },
    "body_models": {
        "title": "Body model",
        "required": True,
        "description": "SMPL-H: shape, joints and skinning for the baked motion.",
        "gated": True,
    },
    "characters": {
        "title": "Built-in characters",
        "required": False,
        "description": "Optional avatars the workspace can dress a person with.",
        "gated": False,
    },
}

# Files that need a rule of their own: a different group, or optional even inside
# a required group.
#
# `vggt_` is the camera estimator - the workspace boots with a static camera and
# only loads it when camera estimation is switched on. `sam3d_params.*` and
# `config.yml` are read by the ONNX SAM chain and by the FlowHMR export, and by
# nothing at run time: the packaged build runs SAM through LibTorch and has the
# narrowband window compiled in, so they ship as optional (measured: removing
# them changes nothing in a full clip).
FILE_RULES = {
    "flowhmr/vggt_": {"group": "camera", "optional": True},
    "body_models/smplh/neutral/README.txt": {"group": "body_models", "optional": True, "gated": False},
    "sam3d/sam3d_params.json": {"group": "sam3d", "optional": True},
    "sam3d/sam3d_params.bin": {"group": "sam3d", "optional": True},
    "flowhmr/config.yml": {"group": "flowhmr", "optional": True},
}
EXTRA_GROUPS = {
    "camera": {
        "title": "Camera estimation (optional)",
        "required": False,
        "description": "VGGT-Omega: estimates the camera path. Off by default; the "
                       "workspace then uses a static camera and never loads these.",
        "gated": False,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def key_for(relative: str) -> str:
    stem = relative.replace("\\", "/")
    for tail in (".weights.bin", ".onnx", ".bin", ".json", ".npz", ".glb", ".pt", ".yaml", ".txt"):
        if stem.endswith(tail):
            stem = stem[: -len(tail)]
            break
    return stem.replace("/", "_").replace("-", "_").replace(".", "_").lower()


def title_for(relative: str) -> str:
    name = Path(relative).name
    return name


def describe(models: Path, verify_hash: bool) -> dict:
    groups: dict[str, dict] = {}

    def group_of(relative: str) -> tuple[str, bool]:
        posix = relative.replace("\\", "/")
        for prefix, spec in FILE_RULES.items():
            if posix.startswith(prefix):
                return spec["group"], spec.get("optional", False)
        head = posix.split("/", 1)[0]
        if head in LAYOUT:
            return head, False
        return "other", True

    def gated_of(relative: str, group_gated: bool) -> bool:
        posix = relative.replace("\\", "/")
        for prefix, spec in FILE_RULES.items():
            if posix.startswith(prefix) and "gated" in spec:
                return spec["gated"]
        return group_gated

    for path in sorted(p for p in models.rglob("*") if p.is_file()):
        relative = str(path.relative_to(models)).replace("\\", "/")
        if relative == "manifest.json":
            continue
        group_id, optional = group_of(relative)
        definition = EXTRA_GROUPS.get(group_id) or LAYOUT.get(
            group_id,
            {
                "title": "Other files in this package",
                "required": False,
                "description": "Files this scan could not attribute to a stage.",
                "gated": False,
            },
        )
        group = groups.setdefault(
            group_id,
            {
                "id": group_id,
                "title": definition["title"],
                "required": definition["required"],
                "description": definition["description"],
                "files": [],
            },
        )
        entry = {
            "key": key_for(relative),
            "title": title_for(relative),
            "path": relative,
            "bytes": path.stat().st_size,
        }
        if gated_of(relative, definition.get("gated", False)):
            entry["gated"] = True
        if optional:
            entry["optional"] = True
        if verify_hash:
            entry["sha256"] = sha256(path)
        group["files"].append(entry)

    order = ["detector", "sam3d", "flowhmr", "camera", "body_models", "characters", "other"]
    ranked = sorted(groups.values(), key=lambda g: order.index(g["id"]) if g["id"] in order else len(order))
    return {
        "schema": 1,
        "note": "Manifest of a packaged install: only the files the application "
                "loads at run time. Verified by size and SHA-256. Files marked gated "
                "are obtained by the user (registration or account) and are not "
                "redistributed. Licences: licenses/README.md in the VeloMocap repository.",
        "root": "models",
        "groups": ranked,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--no-hash", action="store_true", help="skip SHA-256 (much faster)")
    args = parser.parse_args()

    if not args.models.is_dir():
        print(f"error: not a directory: {args.models}", file=sys.stderr)
        return 2

    manifest = describe(args.models, verify_hash=not args.no_hash)
    out = args.out or (args.models / "manifest.json")
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")

    files = sum(len(g["files"]) for g in manifest["groups"])
    total = sum(f["bytes"] for g in manifest["groups"] for f in g["files"])
    print(f"wrote {out}")
    print(f"{files} files, {total / 1e9:.2f} GB, {len(manifest['groups'])} groups")
    for group in manifest["groups"]:
        flag = "required" if group["required"] else "optional"
        print(f"  {group['title']:<44} {len(group['files']):>2} files  [{flag}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
