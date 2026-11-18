# Publishing the converted models - the final list

Rule: a file is ours to publish only when **the release package contains it**
and **the upstream repository does not**. Everything we tried and dropped stays
in the development tree and is not published.

Result: **three repositories, nine files, 9.8 GB.**

| Repository | Files | Size | Licence |
| --- | --- | --- | --- |
| `velomocap-sam3d-torchscript` | 5 | 3.53 GB | SAM License |
| `velomocap-flowhmr-onnx` | 2 | 1.85 GB | FlowHMR Model Weights License (non-commercial) |
| `velomocap-vggt-omega-onnx` | 2 | 4.45 GB | FAIR Noncommercial Research License |

The name states the **artifact format**, because that is what tells a reader
which engine loads it: SAM 3D Body is a TorchScript module that LibTorch loads
(`.pt`), FlowHMR and VGGT-Omega are ONNX graphs that ONNX Runtime loads (`.onnx`
+ `.weights.bin`). An earlier draft called the SAM repository `-onnx`; it holds
no ONNX file and the application never runs it through ONNX Runtime, so the name
was wrong.

---

## 1. `velomocap-sam3d-torchscript` - SAM License

The packaged application runs SAM 3D Body through LibTorch and loads
`models/sam3d/sam3d_full_ts.pt`; it is the only SAM weight the release needs.
The ONNX export of the same model is not in the package and is not published.

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `sam3d_full_ts.pt` | 3,533,327,692 | `476d31ee0a26f1140ffc5fd44b3e64fc7ace8da4d112a3164bff8c5f5d16d9f6` |
| `sam3d_faces.bin` | 442,488 | `f6748e290ef37fbb6877c4cc5bd7287105db9e98252b0ba170ae9ac3c45eacd6` |
| `sam3d_keypoints.bin` | 6,540 | `532b75ddb6e1650be033b80f79545eeb54eaf05298cef0bd63b9bc36e72fc4e6` |
| `sam3d_skeleton.bin` | 2,032 | `bf8080b0d21e9b2469cd1b84c62da475ec560ef9743341a011aac57bc90f61e9` |
| `sam3d_mesh.json` | 325 | `04430bf3908a102f3130d660a0eed32cb6da4e53cec10d9707f0e35fbecd4e3c` |

Add: `LICENSE` (the SAM License, verbatim from `licenses/SAM-LICENSE.txt`) and
`NOTICE.md` (the SAM section of `licenses/CONVERSION-NOTICES.md`).

`sam3d_full_ts.pt` is traced per build, so a re-trace changes the hash. If you
would rather not host 3.5 GB, drop this repository and say in the README that
the user traces it themselves with `tools/torch/sam3d_full_jit.py` from Meta's
gated `model.ckpt` - the package then stops being drop-in.

---

## 2. `velomocap-flowhmr-onnx` - FlowHMR Model Weights License

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `flowhmr_mmdit.weights.bin` | 1,841,575,852 | `4e08a4f175e431005e6723ee3340e7c01aa016827fe97395140f31d863d34edd` |
| `flowhmr_mmdit.onnx` | 4,118,877 | `29d7390e58d4635a9cefe3574297cb284bca3ae9f6b6ca1765d8d305b7c33f0c` |

Add: `LICENSE-flowhmr-weights` (verbatim from
`licenses/FlowHMR-Model-Weights-LICENSE.txt`) and `NOTICE.md` (the FlowHMR
section). Non-commercial use only; attribution and citation of
arXiv:2610.03691 are conditions.

Optional, from the FlowHMR **code** repository (Apache-2.0, not ours):
`config.yml` (3,067 B, `5deae018...`) and `motion_stats.json` (169,611 B,
`6e0f2c4f...`). If you include them, also ship `LICENSE-Apache-2.0` and say in
the notice which file is under which licence.

---

## 3. `velomocap-vggt-omega-onnx` - FAIR Noncommercial Research License

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `vggt_omega_1b_512.weights.bin` | 4,445,007,872 | `4f3bfec468589866be5829f302f3a11d890c28c92d649f5b5ca399c3abcf9840` |
| `vggt_omega_1b_512.onnx` | 1,559,489 | `44626b04c89af9ec127af7ddb8b5624ecb0aaaa4d7d0af6613ecd36fe941ad99` |

Add: `LICENSE.txt` (verbatim from `licenses/VGGT-Omega-LICENSE.txt`) and
`NOTICE.md` (the VGGT section). Non-commercial research only; this licence
requires the text to accompany every copy, and a citation of VGGT.

Only used when camera estimation is switched on - the workspace starts with a
static camera. Say so in the card, so nobody downloads 4.4 GB for nothing.

---

## What is not published, and why

| File | Why not |
| --- | --- |
| `sam3d_backbone.onnx` + `.weights.bin`, `sam3d_init/pose/post/step0..5.onnx`, `mhr_model.onnx`, `sam3d_params.{json,bin}`, `sam3d_split.json` | Abandoned ONNX route: not in the release package, the packed build loads the traced module instead |
| `detector/yolov8x.onnx` | AGPL-3.0: publishing it would put the whole combination under AGPL |
| `body_models/smplh/*`, `body_models/smplx/*` | MPI registration, non-commercial, redistribution forbidden |
| `characters/*.glb` | Epic / Adobe assets bound to their engines |
| `runtime/**` | ONNX Runtime, LibTorch, CUDA/cuDNN, ffmpeg - own terms, mostly not redistributable |
| `flowhmr_latest.ckpt`, `model.ckpt`, `mhr_model.pt`, `vggt_omega_1b_512.pt` | The original checkpoints; link to the upstream repositories instead of mirroring 10 GB |

## Model card: the three lines that matter

```yaml
---
license: other
license_name: <sam-license | flowhmr-model-weights-license | fair-noncommercial-research-license>
license_link: <https://huggingface.co/facebook/sam-3d-body-dinov3 | .../fafsaf1/flowhmr-0.46B | .../facebook/VGGT-Omega>
---
```

Then one paragraph: what this repository is, one line per file in a table with
size and SHA-256, `## Upstream` (repository, revision, copyright), `## What
changed` (the conversion, fp32, nothing retrained), `## Verification` (SAM
1.5e-5 m, FlowHMR 9.7e-3 max, VGGT 2.6e-4, with the reproduce command),
`## Licence` (verbatim text in `LICENSE`, plus what the licence requires).
Templates with the numbers filled in: `licenses/CONVERSION-NOTICES.md`.

## Stage, then upload

Hard links, so the staging folder costs no disk:

```powershell
$root  = "E:\RGBMocap\velomocap"
$stage = "E:\model-upload"
Remove-Item $stage -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force "$stage\velomocap-sam3d-torchscript"      | Out-Null
New-Item -ItemType Directory -Force "$stage\velomocap-flowhmr-onnx"    | Out-Null
New-Item -ItemType Directory -Force "$stage\velomocap-vggt-omega-onnx" | Out-Null

# 1. SAM: the traced module lives in export\, the tables in models\sam3d
foreach ($f in 'sam3d_faces.bin','sam3d_keypoints.bin','sam3d_skeleton.bin','sam3d_mesh.json') {
  New-Item -ItemType HardLink -Path "$stage\velomocap-sam3d-torchscript\$f" -Target "$root\models\sam3d\$f"
}
New-Item -ItemType HardLink -Path "$stage\velomocap-sam3d-torchscript\sam3d_full_ts.pt" `
  -Target "$root\..\export\sam3d\sam3d_full_ts.pt"
Copy-Item "$root\licenses\SAM-LICENSE.txt" "$stage\velomocap-sam3d-torchscript\LICENSE"

# 2. FlowHMR
foreach ($f in 'flowhmr_mmdit.onnx','flowhmr_mmdit.weights.bin') {
  New-Item -ItemType HardLink -Path "$stage\velomocap-flowhmr-onnx\$f" -Target "$root\models\flowhmr\$f"
}
Copy-Item "$root\licenses\FlowHMR-Model-Weights-LICENSE.txt" "$stage\velomocap-flowhmr-onnx\LICENSE-flowhmr-weights"

# 3. VGGT-Omega
foreach ($f in 'vggt_omega_1b_512.onnx','vggt_omega_1b_512.weights.bin') {
  New-Item -ItemType HardLink -Path "$stage\velomocap-vggt-omega-onnx\$f" -Target "$root\models\flowhmr\$f"
}
Copy-Item "$root\licenses\VGGT-Omega-LICENSE.txt" "$stage\velomocap-vggt-omega-onnx\LICENSE.txt"
```

Then write the model card (`README.md`) and `NOTICE.md` into each folder and
upload:

```powershell
pip install -U huggingface_hub
hf auth login
hf upload YuanBaoSMadLab/velomocap-sam3d-torchscript      E:\model-upload\velomocap-sam3d-torchscript      --repo-type model
hf upload YuanBaoSMadLab/velomocap-flowhmr-onnx    E:\model-upload\velomocap-flowhmr-onnx    --repo-type model
hf upload YuanBaoSMadLab/velomocap-vggt-omega-onnx E:\model-upload\velomocap-vggt-omega-onnx --repo-type model
```

ModelScope: same three names, `ms-hub upload <namespace>/<repo> <folder>
--repo-type model`, or the web upload.

## Before you publish

- [ ] Only the nine files above are in the staging folders.
- [ ] Each repository has the upstream licence text, verbatim, under the file
      name its own convention uses (`LICENSE`, `LICENSE-flowhmr-weights`,
      `LICENSE.txt`).
- [ ] Each repository has a `NOTICE.md` that says which files we made and what
      the licence requires.
- [ ] The two non-commercial repositories say so in the first paragraph.
- [ ] The card lists sizes and SHA-256 that match the uploaded bytes.
- [ ] No ONNX SAM chain, no YOLO, no SMPL-H, no characters, no runtime binaries.
