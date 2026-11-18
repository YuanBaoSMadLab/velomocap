# VeloMocap

**Single-camera RGB video to 3D human motion, as a native Windows application in Rust and C++.**

VeloMocap takes one ordinary clip and returns a rigged, animated character: person
detection and tracking, 3D body reconstruction (SAM 3D Body / MHR), camera estimation,
FlowHMR motion generation, foot-contact cleanup, and export as an animated `.glb`, BVH or
binary FBX. It is a from-scratch re-implementation of the
[RGB MoCap](https://github.com/toyxyz/toyxyz_RGBMocap) pipeline, not a wrapper around it:
every stage is code in this repository, and the app runs without Python.

![The VeloMocap workspace](assets/screenshot.png)

## Facts

```yaml
name: VeloMocap
version: 0.1.0
platform: Windows x64
runtime: native (Rust + C++), no Python at run time
input: single-camera RGB video
output: animated .glb, BVH, binary FBX 7.4
inference: ONNX Runtime (CUDA) + LibTorch TorchScript
ui: egui desktop app, Chinese and English; headless CLI
license: Apache-2.0 (source only, model weights excluded)
```

## Pipeline

```text
video 鈹€鈻?YOLOv8 detector 鈹€鈻?IoU tracker
                              鈹?                              鈻?        SAM 3D Body (TorchScript through LibTorch, ONNX fallback)
        body 鈹€鈻?both hands 鈹€鈻?fusion 鈹€鈻?MHR mesh
                              鈹? features: body + left hand + right hand
                              鈻?        VGGT-Omega camera 鈹€鈻?FlowHMR MMDiT 鈹€鈻?SMPL-H solver 鈹€鈻?foot lock
                              鈹?                              鈻?        viewport 路 animated .glb 路 BVH 路 binary FBX
```

## Compared with the Python way

| | Python pipelines (the reference implementation and similar projects) | VeloMocap |
| --- | --- | --- |
| Run time | Python interpreter + venv; an install script fetches runtimes | one native executable; no interpreter |
| Each stage | research repositories glued together with Python | detector, tracker, body, camera, motion, foot lock, exporters 鈥?all here |
| Deployment | pip/conda environment; a version drift breaks the chain | a folder: executable + DLLs + model path |
| Scripting | needs a Python process | `velomocap-cli`, one command per job |
| Inference | PyTorch / ONNX | LibTorch TorchScript by default, ONNX Runtime as a fallback, same weights |
| UI toolkit | Qt (PySide6) | egui, immediate mode |
| Export | depends on plugins | animated `.glb` (mesh, skin, materials, take), BVH, binary FBX |

## What it does

- **Import 鈫?detect 鈫?tick 鈫?reconstruct.** One detection pass, then only the people you
  ticked are reconstructed; a tick survives the tracker re-numbering a body mid-take.
- **Project and recovery.** One `.vp` file holds the video reference, the bodies, the mesh,
  the baked motion, the avatars and the hand-drawn locks. A take is also autosaved into
  `%TEMP%` and offered back if the last session did not exit cleanly.
- **Multi-person.** Every person gets an identity, a colour, its own mesh and motion clip.
- **Streaming playback.** The video plays through one ffmpeg process whose frames drive the
  playhead, so picture and animation never drift apart.
- **Characters.** Import VRM, Unreal or Mixamo-rigged glTF/FBX/GLB and the take drives them
  in the viewport with their own materials; the library is shared by every project.
- **Export.** Character asset window writes one `.glb` per person (mesh, skin, materials
  and animation, textures embedded); the CLI also writes BVH and binary FBX.
- **Foot lock.** The reference's contact correction, ported, with a per-frame report
  (slide, float, sink, jitter) and a gate that rolls a correction back when it makes the
  feet worse.
- **Self-checks.** Parity checks (`tools/torch/*_check.py`) compare the exported SAM module
  against the PyTorch reference; the crate tests cover the exporters, the project file and
  the contact-planner's acceptance numbers.

## Build from source

Requires Rust (stable) with the MSVC toolchain, and Windows x64.

```powershell
cargo build --release -p velomocap-app -p velomocap-cli
```

LibTorch is optional; with it, SAM 3D Body runs through TorchScript (faster, and with
hands). Without it the ONNX chain is used.

```powershell
$env:VELOMOCAP_LIBTORCH = "1"
$env:PATH = "<torch>\lib;$env:PATH"
cargo build --release -p velomocap-app -p velomocap-cli
```

## Run

Run from the folder that holds `models/`:

```powershell
velomocap-app.exe                           # desktop app
velomocap-app.exe clip.mp4 --detect-only    # detection pass, prints candidates
velomocap-app.exe clip.mp4 --capture        # reconstruct and bake, headless
velomocap-cli flow clip.mp4 --person 1 --bvh out.bvh --fbx out.fbx
```

## Models are not in this repository

`models/` and `runtime/` are ignored by git: SAM 3D Body, FlowHMR, VGGT-Omega, YOLOv8 and
the ONNX Runtime / LibTorch / ffmpeg binaries keep their own licenses and are obtained
separately. The app resolves them from `models/` and `runtime/` beside the executable;
`tools/check_models.py` verifies what is installed against the manifest the app uses,
and the model registry in the
[reference repository](https://github.com/toyxyz/toyxyz_RGBMocap) lists where each
weight comes from. SMPL-H must be downloaded from
[smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de/) after registration and placed at
`models/body_models/smplh/neutral/model.npz`.

Most of the model files are ONNX conversions made here, and a conversion is a derivative
work: it keeps the licence of the weights it came from. [licenses/README.md](licenses/README.md)
lists every file with its upstream, its licence and whether it was converted;
[licenses/](licenses/) holds the verbatim texts that must travel with the converted
weights, says which upstreams may not be redistributed (SMPL-H, YOLOv8), and
[licenses/UPLOAD.md](licenses/UPLOAD.md) describes how to publish the allowed ones on
Hugging Face or ModelScope. A packaged install carries only the files the app loads at run
time; 	ools/make_package_manifest.py writes the manifest of such a package, so
	ools/check_models.py can verify it offline.

## Releases

Tags mark the source that produced a working build. No binaries, archives or installers are
published 鈥?build from source as above.

## Scope

Single-camera RGB only. The reference implementation also covers multi-camera capture,
streaming into Blender and live preview with several model backends; none of that is
implemented here. MMD models load with materials and inherit links, but the rigid-body
physics that moves MMD hair and cloth is not simulated.

## Repository layout

```text
crates/velomocap-app      egui workspace, software viewport, i18n, project and recovery
crates/velomocap-motion   capture, tracking, FlowHMR features, cleanup, BVH/FBX
crates/velomocap-infer    ONNX sessions, detector, tracker, SAM chain, mesh topology
crates/velomocap-body     SMPL-H / SMPL-X parsing and skinning
crates/velomocap-core     BVH, FBX, contact planning, math, C++ kernels
crates/velomocap-video    ffmpeg decode / seek / streaming
crates/velomocap-cpp      LibTorch bridge (C ABI) for the SAM TorchScript module
crates/velomocap-models   model manifest and verification
crates/velomocap-cli      headless pipeline
tools/                    offline model export, comparisons, fetching
```

## Keywords

motion capture 路 markerless mocap 路 video to animation 路 pose estimation 路 3D human 路
SAM 3D Body 路 MHR 路 SMPL-H 路 SMPL 路 FlowHMR 路 VGGT 路 YOLOv8 路 ONNX Runtime 路 LibTorch 路
TorchScript 路 Rust 路 C++ 路 egui 路 BVH 路 FBX 路 glTF 路 VRM 路 retargeting 路 Windows

## License

Apache License 2.0 鈥?see [LICENSE](LICENSE). It covers the source in this repository only;
[NOTICE](NOTICE) lists the model weights, body models and third-party binaries that keep
their own licenses.

涓枃璇存槑瑙?[README-zh.md](README-zh.md)銆?