# Runtime binaries that ship with a packaged install

These are not models and they are not part of the source repository; a packaged
install carries them beside the executable. Each keeps its own licence.

| Component | Files | Licence | What shipping it requires |
| --- | --- | --- | --- |
| ONNX Runtime 1.31.1 (CUDA 13 build) | `onnxruntime*.dll` | MIT | keep the copyright and permission notice |
| PyTorch / LibTorch (C++ redistributable) | `torch*.dll`, `c10*.dll`, `caffe2_nvrtc.dll`, `libiomp5md.dll` | BSD-3-Clause | keep the notice; do not imply endorsement |
| CUDA runtime, cuBLAS, cuDNN, cuFFT, cuRAND, cuSOLVER, cuSPARSE, NVRTC | `cudart*`, `cublas*`, `cudnn*`, `cufft*`, `curand*`, `cusolver*`, `cusparse*`, `nvrtc*`, `nvJitLink*` | NVIDIA CUDA Toolkit / cuDNN EULA | redistribution only as part of the application, per the EULA; no separate sale |
| zlib | `zlibwapi.dll` | zlib licence | keep the notice |
| ffmpeg / ffprobe | `runtime\ffmpeg\ffmpeg.exe`, `ffprobe.exe`, `av*.dll`, `sw*.dll` | **GPLv3** | see below |

## ffmpeg is a GPLv3 build

The bundled binaries report:

```text
ffmpeg version N-125136-gb57ff00bcf-20260620
configuration: ... --enable-gpl --enable-version3 ... (crosstool-NG / mingw32)
```

`--enable-gpl --enable-version3` makes the binaries GPLv3 (or later), not LGPL.
Distributing them means:

* ship the licence text - `GPLv3.txt` in this folder;
* offer the **corresponding source** of that exact build: FFmpeg at commit
  `b57ff00bcf` (git.ffmpeg.org / github.com/FFmpeg/FFmpeg) plus the builder's
  scripts for the shared mingw build used here. A written offer with a URL that
  stays valid satisfies it for a non-commercial release;
* keep the FFmpeg copyright notice;
* do not describe the whole package as Apache-2.0 while it carries these
  binaries - the combination is GPLv3.

If that is more obligation than the release wants, ship an LGPL build instead
(`--disable-gpl --disable-version3`, which still decodes and seeks everything
this application uses), and GPLv3 goes away for the package.

## Not included, on purpose

SMPL-H, SMPL-X (`models\body_models\`) and the YOLOv8 detector
(`models\detector\yolov8x.onnx`) are never bundled for redistribution: see
`SMPL-NOTICE.md` and `ULTRALYTICS-YOLOV8-NOTICE.md`. The user obtains them
themselves.
