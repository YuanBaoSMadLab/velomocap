# Conversion notices (the "new licence" is upstream's licence, plus a notice)

The weights we publish are **conversions**: ONNX exports, a traced TorchScript
module, packed sidecars, extracted parameter tables. They exist in no upstream
repository, so they have to be published here - but they are derivative works of
the weights they came from, and a derivative work cannot be relicensed.

So each model repository gets exactly two documents:

| File in the repository | What it is | Who wrote it |
| --- | --- | --- |
| `LICENSE` (or `LICENSE.txt`) | the upstream licence, **verbatim and unmodified** | upstream |
| `NOTICE.md` | our conversion notice: what this repository is, which files we made, how they were verified, and what the upstream licence requires | VeloMocap |

The notice is **not** a licence. It adds no permission and takes none away; it
exists so a reader can tell a converted file from an upstream one and knows
which text governs. What must not be added:

* a restriction upstream does not have (e.g. "no commercial use" on the SAM
  conversion - the SAM License permits commercial use, and a derivative may only
  be distributed *under that agreement*);
* a share-alike or attribution **condition** on top of FlowHMR or VGGT-Omega -
  asking for credit is fine, making it a condition would conflict;
* a warranty or indemnity of our own on top of theirs.

`LICENSE` for the FlowHMR repository is the exception: that repository also
carries two small files from the FlowHMR **code** repository, which is
Apache-2.0, so it ships `LICENSE-flowhmr-weights` and `LICENSE-Apache-2.0` and
the notice says which file is under which.

## Which files exist upstream, and which are ours

Only what the **packaged application actually loads** is listed. Anything else
in the development tree is a route we abandoned - the ONNX SAM chain is the big
one, and it is neither in the release nor published.

| Upstream repository | Files it contains | Files we publish (ours) |
| --- | --- | --- |
| `facebook/sam-3d-body-dinov3` | `LICENSE`, `README.md`, `model.ckpt`, `model_config.yaml`, `assets/mhr_model.pt` | the traced `sam3d_full_ts.pt` plus the four MHR tables `sam3d_mesh.json`, `sam3d_faces.bin`, `sam3d_skeleton.bin`, `sam3d_keypoints.bin` |
| `fafsaf1/flowhmr-0.46B` | `LICENSE`, `README.md`, `config.json`, `flowhmr_base/*`, `flowhmr_latest/*` | `flowhmr_mmdit.onnx` + `flowhmr_mmdit.weights.bin` |
| `flowhmr/flowhmr` (code, Apache-2.0) | the pipeline, `assets/motion_stats.json`, `config.yml` | nothing - these two are redistributed unchanged, with their Apache-2.0 licence |
| `facebook/VGGT-Omega` | `LICENSE.txt`, `README.md`, three `.pt` checkpoints | `vggt_omega_1b_512.onnx` + `vggt_omega_1b_512.weights.bin` |
| `ultralytics/assets` | `yolov8x.pt` (AGPL-3.0) | `yolov8x.onnx` - **not published**, see `ULTRALYTICS-YOLOV8-NOTICE.md` |

**Not published, and not part of the release:** the ONNX SAM chain
(`sam3d_backbone.onnx` + sidecar, `sam3d_init/pose/post/step0..5.onnx`,
`mhr_model.onnx`, `sam3d_params.{json,bin}`, `sam3d_split.json`). The packaged
build runs SAM through LibTorch with the single traced module; the ONNX graphs
and the parameter tables they read exist only in the development tree. Same for
`yolov8x-pose.onnx`, SMPL-X and the original checkpoints.

---

## NOTICE.md for `velomocap-sam3d-torchscript`

```markdown
# SAM 3D Body - TorchScript conversion for VeloMocap

This repository is a **conversion** of SAM 3D Body by Meta, published so that
[VeloMocap](https://github.com/YuanBaoSMadLab/velomocap) can run the model
without PyTorch. It is a derivative work of the upstream weights and is
distributed under the **SAM License**, reproduced verbatim in `LICENSE`.
Commercial use is permitted by that licence; redistribution of this repository
or of any file in it must keep `LICENSE` with it.

## Upstream

* Model: <https://huggingface.co/facebook/sam-3d-body-dinov3>
* Files converted: `model.ckpt` (body, hands, fusion) and `assets/mhr_model.pt`
  (MHR geometry)
* Copyright: Meta Platforms, Inc. See `LICENSE`.

## Files we made

| File | Size | What it is |
| --- | --- | --- |
| `sam3d_full_ts.pt` | 3,533,327,692 B | the whole pipeline - backbone, hands, fusion, MHR geometry - traced as one TorchScript module for LibTorch. This is what the application loads. |
| `sam3d_mesh.json` | 325 B | MHR topology description: 18439 vertices, 36874 faces, 127 joints, 70 keypoints, and the file names of the three tables below |
| `sam3d_faces.bin` | 442,488 B | triangle indices, uint32 little-endian |
| `sam3d_skeleton.bin` | 2,032 B | the 127-joint skeleton |
| `sam3d_keypoints.bin` | 6,540 B | the 70-keypoint regressor |

The ONNX export of this model (eleven graphs plus a weights sidecar and the
parameter tables they read) is **not** part of this repository: that route is not
used by the application any more.

## How it was made

Traced on CPU in fp32 by `tools/torch/sam3d_full_jit.py`, and the MHR tables
dumped by `tools/onnx/dump_mesh.py`, from this repository's `tools/` directory.
Nothing was retrained, fine-tuned or pruned.

## Verification

The ONNX chain and the TorchScript module were compared against the original
PyTorch model on the same inputs: largest vertex difference **1.5e-5 m**
(fp32), keypoint and camera outputs within 1e-3. Reproduce with
`tools/torch/sam3d_full_check.py` and the fixtures kept for the Rust port.

## Citation

If you publish work that uses these files, acknowledge SAM 3D Body (Meta) as the
licence requires, and please also mention this conversion (VeloMocap) - as a
courtesy, not as a condition.
```

---

## NOTICE.md for `velomocap-flowhmr-onnx`

```markdown
# FlowHMR - ONNX conversion for VeloMocap

**Non-commercial use only.** This repository is a **conversion** of the FlowHMR
model weights, published so that
[VeloMocap](https://github.com/YuanBaoSMadLab/velomocap) can run the motion model
without PyTorch. It is a derivative work of those weights and is distributed
under the **FlowHMR Model Weights License**, reproduced verbatim in
`LICENSE-flowhmr-weights`: non-commercial educational, research and non-profit
use only, attribution and this licence retained, cite FlowHMR.

## Upstream

* Weights: <https://huggingface.co/fafsaf1/flowhmr-0.46B>, file
  `flowhmr_latest/flowhmr_latest.ckpt`
* Code: <https://github.com/flowhmr/flowhmr> (Apache-2.0, see `LICENSE-Apache-2.0`)
* Paper: *FlowHMR: Physically Plausible Motion Capture from Video*,
  arXiv:2610.03691

## Files and which licence covers them

| File | Origin | Licence |
| --- | --- | --- |
| `flowhmr_mmdit.onnx` | our ONNX export of the MMDiT | FlowHMR weights licence |
| `flowhmr_mmdit.weights.bin` | its packed weights | FlowHMR weights licence |
| `config.yml` | FlowHMR code repository, unchanged | Apache-2.0 |
| `motion_stats.json` | FlowHMR code repository, unchanged | Apache-2.0 |

## How it was made

`tools/onnx/flowhmr_export.py` exports one flow-matching step to ONNX in fp32
and `pack_onnx.py` folds the per-tensor files into the `.weights.bin` sidecar.
The RoPE table and the narrowband attention masks are graph **inputs**, not
constants, so the 20-step Euler loop and the decode stay in Rust. Nothing was
retrained.

## Verification

Largest absolute difference against the PyTorch reference: **9.7e-3** on a
feature row, mean 2.1e-4, no value above 1e-2. Reproduce with
`tools/onnx/dump_flowhmr_reference.py` and `velomocap-cli flowhmr-parity`.

## Citation

Cite FlowHMR (arXiv:2610.03691) as the licence requires. Mentioning this
conversion (VeloMocap) is a courtesy, not a condition.
```

---

## NOTICE.md for `velomocap-vggt-omega-onnx`

```markdown
# VGGT-Omega - ONNX conversion for VeloMocap

**Non-commercial research use only.** This repository is a **conversion** of the
VGGT-Omega 1B checkpoint, published so that
[VeloMocap](https://github.com/YuanBaoSMadLab/velomocap) can estimate the camera
without PyTorch. It is a derivative work of the upstream weights and is
distributed under the **FAIR Noncommercial Research License**, reproduced
verbatim in `LICENSE.txt`; that licence requires this text to accompany any
copy, and restricts use to non-commercial research.

## Upstream

* Model: <https://huggingface.co/facebook/VGGT-Omega>, file
  `vggt_omega_1b_512.pt`
* Copyright: Meta Platforms, Inc. See `LICENSE.txt`.

## Files we made

* `vggt_omega_1b_512.onnx` - the graph, traced at a fixed **512x288** portrait
  canvas. The positional table (RoPE) is a graph input and the prefix tokens get
  a prepended identity row, which removes the dynamic slicing that tracing
  cannot express. A landscape clip is rotated a quarter turn before scaling.
* `vggt_omega_1b_512.weights.bin` - the packed weights sidecar.

Nothing was retrained. Exported in fp32 by `tools/onnx/vggt_omega_export.py` and
`pack_onnx.py`.

## Verification

Largest absolute difference against the PyTorch reference: **2.6e-4**.
Reproduce with `tools/onnx/dump_vggt_reference.py` and
`velomocap-cli vggt-parity`.

## Citation

Cite VGGT (*Visual Geometry Grounded Transformer*, CVPR 2025, Meta) as the
licence requires.
```
