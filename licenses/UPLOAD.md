# Publishing the converted models (Hugging Face / ModelScope)

This is the checklist for putting the ONNX conversions on a model hub. It is an
engineering note, not legal advice: the licence texts in this folder decide, and
a lawyer should look at anything commercial.

Three repositories, one per upstream licence. Do not mix them: FlowHMR and
VGGT-Omega are non-commercial, SAM 3D Body is not, and a reader has to be able
to tell which terms apply to what they download.

| Repository (suggested name) | Contents | Licence | Size |
| --- | --- | --- | --- |
| `velomocap-sam3d-onnx` | SAM 3D Body ONNX chain + MHR geometry + parameter tables (optionally the traced `sam3d_full_ts.pt`) | SAM License | ~4.2 GB (+3.3 GB optional) |
| `velomocap-flowhmr-onnx` | FlowHMR MMDiT graph + sidecar weights + config + motion statistics | FlowHMR Model Weights License (non-commercial) | ~1.85 GB |
| `velomocap-vggt-omega-onnx` | VGGT-Omega 1B graph + sidecar weights | FAIR Noncommercial Research License | ~4.45 GB |

Use the same names under your ModelScope namespace so the two hubs stay
comparable.

## Do not upload

* `models/body_models/smplh/**`, `models/body_models/smplx/**` - see
  `SMPL-NOTICE.md`; the user must register and download these.
* `models/detector/yolov8*.onnx` - AGPL-3.0, see `ULTRALYTICS-YOLOV8-NOTICE.md`.
  Publish `tools/export_yolo.py` instead.
* `models/characters/*.glb` - Epic/Adobe assets bound to their engines.
* `runtime/**` - ONNX Runtime, LibTorch, CUDA/cuDNN, ffmpeg binaries; each has
  its own terms and most of them do not allow passing the binaries on.
* The original checkpoints (`flowhmr_latest.ckpt`, `model.ckpt`,
  `mhr_model.pt`, `vggt_omega_1b_512.pt`) - point readers at the upstream
  repositories instead of mirroring 10 GB of Meta/FlowHMR files.

## Files and checksums

Sizes and SHA-256 of exactly the bytes that are expected on the hub. Verify a
download with `Get-FileHash <file> -Algorithm SHA256`.

### velomocap-sam3d-onnx (SAM License)

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `sam3d_backbone.onnx` | 568,955 | `B0282525D720E6CE8A8EAEE044867629B3B5E32D4D4605A3F6DE15B5466FB56E` |
| `sam3d_backbone.weights.bin` | 3,369,614,336 | `D621B8AB1B8021C41B8F0502E2ED2A6A595083B9003B67A8780483B9EB6FC755` |
| `sam3d_init.onnx` | 10,747,738 | `902B75460CCDC4A56B5324F8E3B20D44ECC2D5C23AD2DDF668A5143F3D6A3DB3` |
| `sam3d_pose.onnx` | 10,549,579 | `B1F781BFC3DA9EA6D7357A58EDF3C657A4197086E301C99F0057000C76D9FC63` |
| `sam3d_post.onnx` | 36,566,362 | `5BA87C8B52887140C0954E9659DD153E7B68CEA54624B94A1A9E448E80882540` |
| `sam3d_step0.onnx` | 26,308,364 | `0485C07FE7718A9F982597620651A319F940051739657845FE7E7B209E0F6781` |
| `sam3d_step1.onnx` | 26,308,761 | `AAF3DDB099BAE78745CCFF3AB8E9EEE94A9EA5F7BEED0C30D755F005BE99BDDA` |
| `sam3d_step2.onnx` | 26,308,761 | `5CC22F597228246C84DDB52B9FCFEDA834BD09B2D99E9C50E9225850B7D0AA35` |
| `sam3d_step3.onnx` | 26,308,761 | `41170280CA664E790D9C45A3B74C1EC3AB67A563B77A41C5821076A07550FEBD` |
| `sam3d_step4.onnx` | 26,308,761 | `D75CA6737D438C1AE24A2D209FCC201DD7307E57B5535C9AD6E9A5A33C44B239` |
| `sam3d_step5.onnx` | 26,308,761 | `54D46D99579D822F9FF3DC8E426B28F41163A58BFE436FE6D65D9943174197AA` |
| `mhr_model.onnx` | 693,257,885 | `5A8AF6FF6E90812881B42717B70CEEB108D7B45D4BDBC7E0A495F09877E0F49B` |
| `sam3d_params.json` | 1,246 | `15769F7E6143FF8BCD55CE54787F0A80333CD45CA1D7248D38FEE08A03D3A79D` |
| `sam3d_params.bin` | 19,768 | `853D78338EBCE47665CF237BED918D37C3F030D9F83D3F88C4AF49F8F0FF159B` |
| `sam3d_mesh.json` | 325 | `04430BF3908A102F3130D660A0EED32CB6DA4E53CEC10D9707F0E35FBECD4E3C` |
| `sam3d_faces.bin` | 442,488 | `F6748E290EF37FBB6877C4CC5BD7287105DB9E98252B0BA170AE9AC3C45EACD6` |
| `sam3d_skeleton.bin` | 2,032 | `BF8080B0D21E9B2469CD1B84C62DA475EC560EF9743341A011AAC57BC90F61E9` |
| `sam3d_keypoints.bin` | 6,540 | `532B75DDB6E1650BE033B80F79545EEB54EAF05298CEF0BD63B9BC36E72FC4E6` |
| `sam3d_split.json` | 3,199 | `1DC112D182CA718ADC1ECD038D3FC73EC0ED5DD0CAF769C902F839B2F8EB1715` |
| `LICENSE` | - | the SAM License, from `licenses/SAM-LICENSE.txt` |
| optional `sam3d_full_ts.pt` | 3,533,327,692 | recompute before publishing: it is traced per build by `tools/torch/sam3d_full_jit.py`, and only the LibTorch path loads it |

### velomocap-flowhmr-onnx (FlowHMR Model Weights License, non-commercial)

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `flowhmr_mmdit.onnx` | 4,118,877 | `29D7390E58D4635A9CEFE3574297CB284BCA3AE9F6B6CA1765D8D305B7C33F0C` |
| `flowhmr_mmdit.weights.bin` | 1,841,575,852 | `4E08A4F175E431005E6723EE3340E7C01AA016827FE97395140F31D863D34EDD` |
| `config.yml` | 3,067 | `5DEAE01800C031975558BF9675819E8D1E24803414F692B8B5C2196B266C64D7` |
| `motion_stats.json` | 169,611 | `6E0F2C4F5B24FE5805C93ED8B25728E466269941070A545640E51B180C315908` |
| `LICENSE` | - | `licenses/FlowHMR-Model-Weights-LICENSE.txt` |

### velomocap-vggt-omega-onnx (FAIR Noncommercial Research License)

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `vggt_omega_1b_512.onnx` | 1,559,489 | `44626B04C89AF9EC127AF7DDB8B5624ECB0AAAA4D7D0AF6613ECD36FE941AD99` |
| `vggt_omega_1b_512.weights.bin` | 4,445,007,872 | `4F3BFEC468589866BE5829F302F3A11D890C28C92D649F5B5CA399C3ABCF9840` |
| `LICENSE.txt` | - | `licenses/VGGT-Omega-LICENSE.txt` |

The VGGT graph is traced at a fixed 512x288 portrait canvas; a re-export from
the checkpoint with `tools/onnx/vggt_omega_export.py` gives a dynamic size and a
different hash. Say which one a file is.

Conversion parity, measured against the PyTorch reference (see `docs/MODELS.md`
in the development tree): VGGT-Omega max abs difference **2.6e-4**, FlowHMR
MMDiT max abs **9.7e-3** (mean 2.1e-4). Use those numbers in the card rather
than "matches the original".

## Staging folder without copying 10 GB

Hard links share the bytes, so the staging folder costs no disk space:

```powershell
$root  = "E:\RGBMocap\velomocap"
$stage = "E:\model-upload"
Remove-Item $stage -Recurse -Force -ErrorAction SilentlyContinue

New-Item -ItemType Directory -Force "$stage\velomocap-sam3d-onnx"    | Out-Null
New-Item -ItemType Directory -Force "$stage\velomocap-flowhmr-onnx"  | Out-Null
New-Item -ItemType Directory -Force "$stage\velomocap-vggt-omega-onnx" | Out-Null

$sam = @('sam3d_backbone.onnx','sam3d_backbone.weights.bin','sam3d_init.onnx','sam3d_pose.onnx',
         'sam3d_post.onnx','sam3d_step0.onnx','sam3d_step1.onnx','sam3d_step2.onnx','sam3d_step3.onnx',
         'sam3d_step4.onnx','sam3d_step5.onnx','mhr_model.onnx','sam3d_params.json','sam3d_params.bin',
         'sam3d_mesh.json','sam3d_faces.bin','sam3d_skeleton.bin','sam3d_keypoints.bin','sam3d_split.json')
foreach ($f in $sam) {
  New-Item -ItemType HardLink -Path "$stage\velomocap-sam3d-onnx\$f" -Target "$root\models\sam3d\$f"
}
Copy-Item "$root\licenses\SAM-LICENSE.txt" "$stage\velomocap-sam3d-onnx\LICENSE"

foreach ($f in 'flowhmr_mmdit.onnx','flowhmr_mmdit.weights.bin','config.yml','motion_stats.json') {
  New-Item -ItemType HardLink -Path "$stage\velomocap-flowhmr-onnx\$f" -Target "$root\models\flowhmr\$f"
}
Copy-Item "$root\licenses\FlowHMR-Model-Weights-LICENSE.txt" "$stage\velomocap-flowhmr-onnx\LICENSE"

foreach ($f in 'vggt_omega_1b_512.onnx','vggt_omega_1b_512.weights.bin') {
  New-Item -ItemType HardLink -Path "$stage\velomocap-vggt-omega-onnx\$f" -Target "$root\models\flowhmr\$f"
}
Copy-Item "$root\licenses\VGGT-Omega-LICENSE.txt" "$stage\velomocap-vggt-omega-onnx\LICENSE.txt"
```

Write a `README.md` into each folder first (template below), then upload.

## Hugging Face

Create the three repositories first (web UI or CLI). In each one:

* **Licence field:** `Other`, with the name and link of the upstream licence.
  Hugging Face accepts these card fields:

  ```yaml
  license: other
  license_name: sam-license
  license_link: https://huggingface.co/facebook/sam-3d-body-dinov3
  ```

  For FlowHMR use `license_name: flowhmr-model-weights-license` and
  `license_link: https://huggingface.co/fafsaf1/flowhmr-0.46B`; for VGGT-Omega
  `license_name: fair-noncommercial-research-license`,
  `license_link: https://huggingface.co/facebook/VGGT-Omega`.
* **Tags:** `onnx`, `motion-capture`, `sam-3d-body` / `flowhmr` / `vggt`,
  `velomocap`.
* **Gating:** optional. The SAM licence already permits redistribution, so an
  open repository is legal; mirroring Meta's own gate (as the two non-commercial
  repositories do) is the conservative choice and costs nothing but friction.

```powershell
pip install -U huggingface_hub
hf auth login          # or: huggingface-cli login

hf upload YuanBaoSMadLab/velomocap-sam3d-onnx    E:\model-upload\velomocap-sam3d-onnx    --repo-type model
hf upload YuanBaoSMadLab/velomocap-flowhmr-onnx  E:\model-upload\velomocap-flowhmr-onnx  --repo-type model
hf upload YuanBaoSMadLab/velomocap-vggt-omega-onnx E:\model-upload\velomocap-vggt-omega-onnx --repo-type model
```

Old versions of the CLI are `huggingface-cli upload` with the same arguments.
Files over 10 MB go to LFS automatically.

## ModelScope (魔搭)

Same three names under your namespace. Create the model repositories in the web
UI, set the licence to *Other* and paste the upstream licence link into the
description, then either drag the files in or use the CLI:

```powershell
pip install -U modelscope-hub
ms-hub upload YuanBaoSMadLab/velomocap-sam3d-onnx E:\model-upload\velomocap-sam3d-onnx --repo-type model --commit-message "SAM 3D Body ONNX chain"
```

`ms-hub --help` is the authority for the arguments on your version; the web
upload does the same thing without the CLI.

## Model card template

Copy into `README.md` in each staging folder and fill the markers.

```markdown
---
license: other
license_name: <sam-license | flowhmr-model-weights-license | fair-noncommercial-research-license>
license_link: <upstream licence URL>
tags: [onnx, motion-capture, <sam-3d-body | flowhmr | vggt>, velomocap]
---

# <name> - ONNX conversion for VeloMocap

ONNX Runtime version of <upstream model>, converted from the official
checkpoint so that [VeloMocap](https://github.com/YuanBaoSMadLab/velomocap) can
run it without PyTorch. **<Non-commercial research use only / SAM License -
see LICENSE>.**

## Upstream

* Model: <link>, revision <commit>
* Weights: <checkpoint file>, <bytes>, sha256 <...>
* Paper: <citation>

## What changed

* Exported to ONNX in fp32 by `tools/onnx/<script>.py`; nothing was retrained.
* <weights packed into the `.weights.bin` sidecar | fixed 512x288 input canvas |
  RoPE/attention masks made graph inputs>
* Verified against the PyTorch reference: max abs difference <value>
  (`velomocap-cli <flowhmr|vggt>-parity`).

## Files

Table of file, bytes, sha256 - see the repository listing.

## How to use

`models/sam3d/`, `models/flowhmr/` of a VeloMocap checkout. Model weights are
not part of the source repository.

## Licence

Reproduced verbatim in LICENSE. This conversion is a derivative work of
<upstream>, published under the same terms. <Attribution/citation requirement>.
```

## Checklist before you publish

- [ ] The staging folder holds only files from the tables above.
- [ ] `LICENSE` / `LICENSE.txt` is in every repository and is the *verbatim*
      upstream text.
- [ ] The card's first paragraph says non-commercial where that applies, and
      names the upstream repository and revision.
- [ ] Sizes and hashes in the card match the uploaded files.
- [ ] No SMPL-H/SMPL-X, no YOLO, no character assets, no runtime binaries.
- [ ] VeloMocap's own README links to the hub repository instead of shipping
      weights, and `licenses/README.md` is updated if anything changes.
