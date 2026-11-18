# Assemble the per-component licence folder of a packaged install.
#
# The install ships third-party binaries and model weights that each keep their
# own terms. Hand-maintained copies drift - an earlier one said the project was
# MIT, another had "FlowHMR licence: to be confirmed" - so the folder is built
# from `licenses/` in the repository, which is the single source of truth.
#
#   tools\package_licenses.ps1 -Install E:\VeloMocap-Release\VeloMocap
#
# Files this script owns are replaced; anything it does not know about is left
# alone (the operator's own notes, for example).

param(
    [Parameter(Mandatory = $true)][string]$Install
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $Install 'velomocap-app.exe'))) {
    throw "$Install does not look like a VeloMocap install (no velomocap-app.exe)"
}
$out = Join-Path $Install 'licenses'
New-Item -ItemType Directory -Force -Path $out | Out-Null

# Everything this script writes. Anything else in `licenses/` is left alone.
$managed = @(
    'LICENSE', 'NOTICE', 'README.md', 'README-zh.md', 'RUNTIME-BINARIES.md',
    'SAM-LICENSE.txt', 'FlowHMR-Model-Weights-LICENSE.txt', 'VGGT-Omega-LICENSE.txt',
    'PyTorch-LICENSE.txt', 'ONNX-Runtime-LICENSE.txt', 'GPLv3.txt',
    'SMPL-NOTICE.md', 'ULTRALYTICS-YOLOV8-NOTICE.md',
    # Older names, removed when they are found so the folder cannot contradict itself.
    'VeloMocap-MIT.txt', 'SAM-3D-Body.txt', 'FlowHMR.txt', 'SMPL-H-README.md',
    'YOLOv8-AGPL-3.0.txt', 'ffmpeg-GPLv3.txt', 'NVIDIA-CUDA-cuDNN.txt',
    'PyTorch-LibTorch-BSD.txt', 'ONNX-Runtime-MIT.txt'
)

$copies = @{
    'LICENSE'                            = 'LICENSE'
    'NOTICE'                             = 'NOTICE'
    'README.md'                          = 'licenses\README.md'
    'README-zh.md'                       = 'licenses\README-zh.md'
    'RUNTIME-BINARIES.md'                = 'licenses\RUNTIME-BINARIES.md'
    'SAM-LICENSE.txt'                    = 'licenses\SAM-LICENSE.txt'
    'FlowHMR-Model-Weights-LICENSE.txt'  = 'licenses\FlowHMR-Model-Weights-LICENSE.txt'
    'VGGT-Omega-LICENSE.txt'             = 'licenses\VGGT-Omega-LICENSE.txt'
    'PyTorch-LICENSE.txt'                = 'licenses\PyTorch-LICENSE.txt'
    'ONNX-Runtime-LICENSE.txt'           = 'licenses\ONNX-Runtime-LICENSE.txt'
    'GPLv3.txt'                          = 'licenses\GPLv3.txt'
    'SMPL-NOTICE.md'                     = 'licenses\SMPL-NOTICE.md'
    'ULTRALYTICS-YOLOV8-NOTICE.md'       = 'licenses\ULTRALYTICS-YOLOV8-NOTICE.md'
}

foreach ($name in $managed) {
    $path = Join-Path $out $name
    if (Test-Path $path) { Remove-Item $path -Force }
}

$missing = @()
foreach ($name in $copies.Keys) {
    $source = Join-Path $root $copies[$name]
    if (Test-Path $source) {
        Copy-Item $source (Join-Path $out $name) -Force
    } else {
        $missing += $copies[$name]
    }
}

# NVIDIA ships the toolkit under a EULA that is not reproduced here: point at it.
$nvidia = @'
NVIDIA CUDA Toolkit and cuDNN runtime
=====================================

The CUDA runtime, cuBLAS, cuDNN, cuFFT, cuRAND, cuSOLVER, cuSPARSE and NVRTC
libraries beside the executable are redistributed under the NVIDIA CUDA Toolkit
End User License Agreement and the cuDNN license, as components of this
application and not as a separately sold product.

Read them at:
  https://docs.nvidia.com/cuda/eula/index.html
  https://docs.nvidia.com/deeplearning/cudnn/latest/reference/eula.html

This file is a pointer, not a copy: the EULA text is not reproduced here.
'@
[System.IO.File]::WriteAllText((Join-Path $out 'NVIDIA-NOTICE.txt'), $nvidia, (New-Object System.Text.UTF8Encoding($false)))

Get-ChildItem $out -File | Sort-Object Name |
    Select-Object Name, @{ n = 'KB'; e = { [math]::Round($_.Length / 1KB, 1) } } |
    Format-Table -AutoSize

if ($missing.Count -gt 0) {
    Write-Warning ("missing from the repository: " + ($missing -join ', '))
}
Write-Host "licence folder written to $out"
