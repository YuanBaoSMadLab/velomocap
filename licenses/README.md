# Model files, where they come from, and what their licences say

The code in this repository is Apache-2.0. **The model weights are not.** Every
model file the application loads keeps the licence of the project it came from,
and the files in `models/` are mostly *conversions* made here - ONNX exports,
traced TorchScript modules, repacked weight sidecars, extracted tables. A
conversion is a derivative work: it does not get a new licence, and it stays
under the terms of the weights it was made from.

Nothing in this folder is a substitute for reading the original licence.
The three texts we redistribute with converted weights are reproduced verbatim
in this folder:

| File | Covers | Redistribution |
| --- | --- | --- |
| `SAM-LICENSE.txt` | SAM 3D Body and MHR geometry (`facebook/sam-3d-body-dinov3`) | allowed, must ship this text |
| `FlowHMR-Model-Weights-LICENSE.txt` | FlowHMR weights (`fafsaf1/flowhmr-0.46B`) | allowed, **non-commercial only**, keep attribution |
| `VGGT-Omega-LICENSE.txt` | VGGT-Omega (`facebook/VGGT-Omega`) | allowed, **non-commercial research only**, must ship this text |

Two upstreams are *not* redistributable and are not in this repository:

| File | Covers | Redistribution |
| --- | --- | --- |
| `SMPL-H`, `SMPL-X` (`models/body_models/`) | Max Planck body models | **no** - registration required, users download it themselves; see `SMPL-NOTICE.md` |
| `YOLOv8` (`models/detector/yolov8*.onnx`) | Ultralytics person detector | **AGPL-3.0** - see `ULTRALYTICS-YOLOV8-NOTICE.md` |

## Inventory

"Converted here" means the file is produced from the upstream checkpoint by a
script in `tools/`; the upstream checkpoint itself is what the licence covers.

| Files | Upstream | Licence | Converted here | Ship with a converted copy? |
| --- | --- | --- | --- | --- |
| `export/sam3d/sam3d_full_ts.pt` (packaged into `models/sam3d/`, not in git) | SAM 3D Body + hands | SAM License | yes - whole pipeline traced to TorchScript | yes, with `SAM-LICENSE.txt` |
| `models/sam3d/sam3d_mesh.json`, `sam3d_faces.bin`, `sam3d_skeleton.bin`, `sam3d_keypoints.bin` | SAM 3D Body / MHR | SAM License | yes - topology tables extracted from the checkpoint | yes, with `SAM-LICENSE.txt` |
| `models/sam3d/sam3d_backbone.onnx` + `.weights.bin`, `sam3d_init/pose/post/step0..5.onnx`, `mhr_model.onnx`, `sam3d_params.{json,bin}`, `sam3d_split.json` | SAM 3D Body `model.ckpt` | SAM License | yes - ONNX export, fp32, packed sidecar | **no** - abandoned route: the packaged build runs SAM through LibTorch, so these are not shipped and not published |
| `models/flowhmr/flowhmr_mmdit.onnx` + `.weights.bin` | FlowHMR 0.46B `flowhmr_latest.ckpt` | FlowHMR Model Weights License | yes - ONNX export, fp32, ODE loop stays in Rust | yes, **non-commercial**, with its licence |
| `models/flowhmr/config.yml`, `motion_stats.json` | FlowHMR | FlowHMR Model Weights License | unchanged (hash matches upstream) | yes |
| `models/flowhmr/vggt_omega_1b_512.onnx` + `.weights.bin` | VGGT-Omega 1B `vggt_omega_1b_512.pt` | FAIR Noncommercial Research License | yes - ONNX export at a fixed 512x288 canvas | yes, **non-commercial**, with its licence |
| `models/detector/yolov8x.onnx`, `yolov8x-pose.onnx` | Ultralytics YOLOv8x | AGPL-3.0 | yes - ONNX export | only under AGPL-3.0 (see notice) |
| `models/body_models/smplh/neutral/model.npz` | SMPL-H | MPI non-commercial research, registration | no | **never** |
| `models/body_models/smplx/*/model.npz` | SMPL-X | MPI non-commercial research, registration | no | **never** |
| `models/characters/*.glb` (packed install only) | UE5 Manny/Quinn (Epic), Mixamo Remy (Adobe) | Epic Content License / Adobe terms | no | **no** - engine-bound assets |

## What this means in practice

* **A take you export is yours.** The licences restrict the weights, not the
  motion you generate with them - but the non-commercial terms of FlowHMR and
  VGGT-Omega can reach commercial *use of the output* in some readings, so a
  commercial project should treat those two as research-only and get permission
  or replace them.
* **SAM 3D Body is not non-commercial.** Its licence allows commercial use,
  forbids the trade-control / military uses it lists, and requires the licence
  text to travel with any copy: that is why it is in this folder.
* **Do not bundle SMPL-H.** The registration is per user; shipping
  `model.npz` would be redistributing a file the licence says you may not.
* **YOLOv8 decides your whole licence.** Redistributing an application together
  with AGPL-3.0 weights is a much stronger obligation than Apache-2.0. The
  default position here is: do not ship it, let the user export it.

## Reproducing the conversions

```powershell
# SAM 3D Body -> ONNX chain (from model.ckpt + assets/mhr_model.pt)
python tools\onnx\sam3d_export.py --ckpt models\flowhmr\sam\model.ckpt --mhr models\flowhmr\sam\assets\mhr_model.pt --out models\sam3d
python tools\onnx\sam3d_split.py  --dir models\sam3d
python tools\onnx\pack_onnx.py    models\sam3d\sam3d_backbone.onnx models\sam3d\sam3d_backbone.packed.onnx

# FlowHMR and VGGT-Omega -> ONNX graphs + sidecar weights
python tools\onnx\flowhmr_export.py    --ckpt flowhmr_latest.ckpt --config config.yml --out flowhmr_mmdit.onnx
python tools\onnx\vggt_omega_export.py --ckpt vggt_omega_1b_512.pt --out vggt_omega_1b_512.onnx
python tools\onnx\pack_onnx.py         vggt_omega_1b_512.onnx vggt_omega_1b_512.packed.onnx

# the whole SAM pipeline as one TorchScript module (LibTorch builds)
python tools\torch\sam3d_full_jit.py --ckpt models\flowhmr\sam\model.ckpt --out export\sam3d\sam3d_full_ts.pt

# the person detector (Ultralytics AGPL weights, downloaded from the official release)
python tools\export_yolo.py --weights yolov8x.pt --imgsz 640
```

Conversion fidelity is checked in-process against the PyTorch reference:
`tools/onnx/dump_flowhmr_reference.py` / `dump_vggt_reference.py` dump tensors,
`velomocap-cli flowhmr-parity` / `vggt-parity` compare them.

## Citing

* SAM 3D Body - Meta, `facebook/sam-3d-body-dinov3`.
* FlowHMR - *FlowHMR: Physically Plausible Motion Capture from Video*,
  arXiv:2610.03691.
* VGGT - *VGGT: Visual Geometry Grounded Transformer*, CVPR 2025.
* SMPL-H - Romero et al. 2017, Loper et al. 2015; SMPL-X - Pavlakos et al. 2019.

See `UPLOAD.md` for what may be published on Hugging Face / ModelScope, with
which licence file, and how to stage it. `CONVERSION-NOTICES.md` holds the
ready-to-publish `NOTICE.md` of each repository: the three upstream licences are
reproduced verbatim (a converted weight cannot be relicensed), and the notice is
what tells a reader which files are ours and what the licence requires.
