"""Build the three upload folders for the converted models - ready to drag to a hub.

Every repository is assembled the same way:

    <folder>/LICENSE or LICENSE.txt   upstream licence, verbatim
    <folder>/NOTICE.md                our conversion notice, cut from
                                      licenses/CONVERSION-NOTICES.md
    <folder>/README.md                the model card, with the licence the hub
                                      form has to be set to
    <folder>/<weights>                the converted files, hard-linked

Hard links mean the folder costs no extra disk space; deleting it never touches
the originals.

    python tools/stage_model_upload.py --out E:\\model-upload
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# One entry per upload repository. `files` are (source, name-in-repo) pairs; the
# source of the traced module is outside the repository, so it takes a path of
# its own.
REPOS = [
    {
        "repo": "velomocap-sam3d-torchscript",
        "title": "SAM 3D Body - TorchScript conversion for VeloMocap",
        "licence_file": "licenses/SAM-LICENSE.txt",
        "licence_as": "LICENSE",
        "licence_fields": {
            "license": "other",
            "license_name": "sam-license",
            "license_link": "https://huggingface.co/facebook/sam-3d-body-dinov3",
        },
        "licence_label": "SAM License",
        "hub_choice": "Other (Hugging Face) / Other (ModelScope) - not Apache-2.0, not MIT",
        "tags": "[torchscript, motion-capture, sam-3d-body, velomocap]",
        "dest": "models/sam3d/ of a VeloMocap install",
        "summary": "The whole SAM 3D Body pipeline traced as one TorchScript module, "
                   "plus the MHR topology tables VeloMocap renders it with. "
                   "Commercial use is allowed by the SAM License; the licence text "
                   "must travel with the files.",
        "files": [
            ("sam3d/sam3d_full_b8_ts.pt", "sam3d_full_b8_ts.pt"),
            ("sam3d/sam3d_full_ts.pt", "sam3d_full_ts.pt"),
            ("sam3d/sam3d_faces.bin", "sam3d_faces.bin"),
            ("sam3d/sam3d_keypoints.bin", "sam3d_keypoints.bin"),
            ("sam3d/sam3d_skeleton.bin", "sam3d_skeleton.bin"),
            ("sam3d/sam3d_mesh.json", "sam3d_mesh.json"),
        ],
        "extra_sources": {
            "sam3d_full_b8_ts.pt": "../export/sam3d/sam3d_full_b8_ts.pt",
            "sam3d_full_ts.pt": "../export/sam3d/sam3d_full_ts.pt",
        },
        "changed": [
            "Traced from `model.ckpt` and `assets/mhr_model.pt` by "
            "`tools/torch/sam3d_full_jit.py`: backbone, both hands, fusion and the "
            "MHR geometry in one graph, fp32.",
            "`sam3d_full_b8_ts.pt` is the same graph traced at `--batch 8`: one "
            "call reconstructs up to eight people, which is what the reference "
            "does (its `flow_sam.py` passes every box of a frame at once and its "
            "`flow_fast_sam.py` batches up to sixteen frames). VeloMocap prefers "
            "it on a card with 10 GB or more and falls back to `sam3d_full_ts.pt` "
            "below that.",
            "The MHR tables were dumped by `tools/onnx/dump_mesh.py`.",
            "Nothing was retrained, fine-tuned or pruned.",
        ],
        "verified": "Largest vertex difference **1.5e-5 m** against the PyTorch "
                    "reference (fp32); keypoints and camera within 1e-3. The "
                    "batch-8 graph reproduces the batch-1 graph to 7.7e-7 per item.",
        "extra_note": "Loaded by LibTorch, not by ONNX Runtime. The ONNX export of "
                      "this model exists in the development tree but is not used by "
                      "the release and is not published here.",
    },
    {
        "repo": "velomocap-flowhmr-onnx",
        "title": "FlowHMR - ONNX conversion for VeloMocap",
        "licence_file": "licenses/FlowHMR-Model-Weights-LICENSE.txt",
        "licence_as": "LICENSE-flowhmr-weights",
        "licence_fields": {
            "license": "other",
            "license_name": "flowhmr-model-weights-license",
            "license_link": "https://huggingface.co/fafsaf1/flowhmr-0.46B",
        },
        "licence_label": "FlowHMR Model Weights License (non-commercial)",
        "hub_choice": "Other (Hugging Face) / Other (ModelScope) - the licence is "
                      "non-commercial, so do not pick Apache-2.0 or an open CC licence",
        "tags": "[onnx, motion-capture, flowhmr, velomocap]",
        "dest": "models/flowhmr/ of a VeloMocap install",
        "summary": "**Non-commercial use only.** One flow-matching step of FlowHMR "
                   "as an ONNX graph plus its packed weights, so VeloMocap can "
                   "generate motion without PyTorch.",
        "files": [
            ("flowhmr/flowhmr_mmdit.onnx", "flowhmr_mmdit.onnx"),
            ("flowhmr/flowhmr_mmdit.weights.bin", "flowhmr_mmdit.weights.bin"),
        ],
        "changed": [
            "Exported by `tools/onnx/flowhmr_export.py` in fp32 and packed by "
            "`tools/onnx/pack_onnx.py` into the `.weights.bin` sidecar.",
            "The RoPE table and the narrowband attention masks are graph inputs, so "
            "the 20-step Euler loop and the decode stay in Rust.",
            "Nothing was retrained.",
        ],
        "verified": "Largest absolute difference against the PyTorch reference: "
                    "**9.7e-3**, mean 2.1e-4, no value above 1e-2.",
        "extra_note": "Loaded by ONNX Runtime. `config.yml` and `motion_stats.json` "
                      "used by the pipeline come from the FlowHMR code repository "
                      "(Apache-2.0) and are not included here.",
    },
    {
        "repo": "velomocap-vggt-omega-onnx",
        "title": "VGGT-Omega - ONNX conversion for VeloMocap",
        "licence_file": "licenses/VGGT-Omega-LICENSE.txt",
        "licence_as": "LICENSE.txt",
        "licence_fields": {
            "license": "other",
            "license_name": "fair-noncommercial-research-license",
            "license_link": "https://huggingface.co/facebook/VGGT-Omega",
        },
        "licence_label": "FAIR Noncommercial Research License",
        "hub_choice": "Other (Hugging Face) / Other (ModelScope) - non-commercial "
                      "research only, and the licence text must accompany every copy",
        "tags": "[onnx, motion-capture, vggt, velomocap]",
        "dest": "models/flowhmr/ of a VeloMocap install",
        "summary": "**Non-commercial research use only.** VGGT-Omega 1B as an ONNX "
                   "graph plus its packed weights, for camera estimation. "
                   "VeloMocap starts with a static camera, so this is only needed "
                   "when camera estimation is switched on.",
        "files": [
            ("flowhmr/vggt_omega_1b_512.onnx", "vggt_omega_1b_512.onnx"),
            ("flowhmr/vggt_omega_1b_512.weights.bin", "vggt_omega_1b_512.weights.bin"),
        ],
        "changed": [
            "Exported by `tools/onnx/vggt_omega_export.py` in fp32, traced at a "
            "fixed 512x288 portrait canvas; the positional table is a graph input.",
            "Weights packed into the `.weights.bin` sidecar by `pack_onnx.py`.",
            "Nothing was retrained.",
        ],
        "verified": "Largest absolute difference against the PyTorch reference: "
                    "**2.6e-4**.",
        "extra_note": "Loaded by ONNX Runtime.",
    },
]


def notice_section(text: str, repo: str) -> str:
    """Cut the ready-to-publish notice out of CONVERSION-NOTICES.md."""
    marker = f"## NOTICE.md for `{repo}`"
    start = text.find(marker)
    if start < 0:
        raise SystemExit(f"no notice section for {repo} in CONVERSION-NOTICES.md")
    fence = text.find("```markdown", start)
    if fence < 0:
        raise SystemExit(f"no fenced block after the section for {repo}")
    body_start = text.find("\n", fence) + 1
    end = text.find("```", body_start)
    return text[body_start:end].rstrip() + "\n"


def human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def card(spec: dict, files: list[tuple[str, int]], digest) -> str:
    rows = "\n".join(
        f"| `{name}` | {human(size)} | `{digest(name)}` |" for name, size in files
    )
    changed = "\n".join(f"* {line}" for line in spec["changed"])
    fields = "\n".join(f"{key}: {value}" for key, value in spec["licence_fields"].items())
    return f"""---
{fields}
tags: {spec["tags"]}
---

# {spec['title']}

{spec['summary']}

## Files

| File | Size | SHA-256 |
| --- | --- | --- |
{rows}

## Upstream

{spec['upstream_note']}

## What changed

{changed}

## Verified

{spec['verified']}

## How to use

Drop the files into {spec["dest"]},
a [VeloMocap](https://github.com/YuanBaoSMadLab/velomocap) checkout or packaged
install. Model weights are not part of the source repository.

## Licence

{spec['licence_label']} - the text is in `{spec['licence_as']}`, reproduced
verbatim from the upstream repository, and it governs these files. This
repository is a conversion, i.e. a derivative work of that model; see `NOTICE.md`
for which files were made here.

{spec['extra_note']}

## Disclaimer

Converted and published by VeloMocap contributors, not by the upstream authors.
No warranty.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--out", type=Path, default=Path(r"E:\model-upload"))
    parser.add_argument("--copy", action="store_true",
                        help="copy instead of hard-link (costs the full size)")
    args = parser.parse_args()

    notices = (ROOT / "licenses" / "CONVERSION-NOTICES.md").read_text(encoding="utf8")
    upstream = {
        "velomocap-sam3d-torchscript": (
            "* Model: <https://huggingface.co/facebook/sam-3d-body-dinov3>, files\n"
            "  `model.ckpt` and `assets/mhr_model.pt`\n"
            "* Copyright: Meta Platforms, Inc. - see `LICENSE`."
        ),
        "velomocap-flowhmr-onnx": (
            "* Weights: <https://huggingface.co/fafsaf1/flowhmr-0.46B>, file\n"
            "  `flowhmr_latest/flowhmr_latest.ckpt`\n"
            "* Code: <https://github.com/flowhmr/flowhmr> (Apache-2.0)\n"
            "* Paper: *FlowHMR: Physically Plausible Motion Capture from Video*,\n"
            "  arXiv:2610.03691"
        ),
        "velomocap-vggt-omega-onnx": (
            "* Model: <https://huggingface.co/facebook/VGGT-Omega>, file\n"
            "  `vggt_omega_1b_512.pt`\n"
            "* Copyright: Meta Platforms, Inc. - see `LICENSE.txt`."
        ),
    }

    if args.out.exists():
        shutil.rmtree(args.out)
    total = 0
    for spec in REPOS:
        folder = args.out / spec["repo"]
        folder.mkdir(parents=True)
        spec["upstream_note"] = upstream[spec["repo"]]

        staged: list[tuple[str, int]] = []
        for source, name in spec["files"]:
            src = spec.get("extra_sources", {}).get(name)
            path = (ROOT / src).resolve() if src else (args.models / source)
            if not path.is_file():
                raise SystemExit(f"missing {path}")
            target = folder / name
            if args.copy:
                shutil.copy2(path, target)
            else:
                os.link(path, target)
            staged.append((name, path.stat().st_size))
            total += path.stat().st_size

        shutil.copy2(ROOT / spec["licence_file"], folder / spec["licence_as"])
        (folder / "NOTICE.md").write_text(
            notice_section(notices, spec["repo"]), encoding="utf8"
        )

        import hashlib

        cache: dict[str, str] = {}

        def digest(name: str, _folder=folder) -> str:
            if name not in cache:
                h = hashlib.sha256()
                with (_folder / name).open("rb") as handle:
                    for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                        h.update(block)
                cache[name] = h.hexdigest()
            return cache[name]

        (folder / "README.md").write_text(card(spec, staged, digest), encoding="utf8")
        print(f"{spec['repo']:<32} {len(staged)} files  {human(sum(s for _, s in staged)):>9}"
              f"  -> {folder}")

    print(f"\n{len(REPOS)} repositories, {sum(len(s['files']) for s in REPOS)} files, "
          f"{human(total)} total (hard-linked, no extra disk)")
    print("Each folder has LICENSE + NOTICE.md + README.md; upload the folder as is.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
