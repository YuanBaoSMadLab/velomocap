# Assemble a portable VeloMocap folder: exe + every DLL + every model file.
# Run from anywhere; paths are absolute on purpose.
$ErrorActionPreference = 'Stop'

$src   = 'E:\RGBMocap\velomocap'
$dist  = 'C:\VeloMocap-Release\VeloMocap'
$torch = 'C:\Users\38820\miniconda3\Lib\site-packages\torch\lib'
$ort   = (Get-ChildItem "$src\runtime\onnxruntime" -Recurse -Filter onnxruntime.dll).DirectoryName
$ffmpeg = 'D:\ffmpeg\bin'

New-Item -ItemType Directory -Force -Path $dist | Out-Null
New-Item -ItemType Directory -Force -Path "$dist\runtime\ffmpeg" | Out-Null
New-Item -ItemType Directory -Force -Path "$dist\models\detector" | Out-Null
New-Item -ItemType Directory -Force -Path "$dist\models\sam3d" | Out-Null
New-Item -ItemType Directory -Force -Path "$dist\models\flowhmr" | Out-Null
New-Item -ItemType Directory -Force -Path "$dist\models\body_models\smplh\neutral" | Out-Null

# 1. Binaries
Copy-Item "$src\target\release\velomocap-app.exe" $dist -Force
Copy-Item "$src\target\release\velomocap-cli.exe" $dist -Force

# 2. LibTorch + CUDA + cuDNN beside the exe (Windows resolves the app folder first).
Get-ChildItem $torch -Filter *.dll | ForEach-Object {
    Copy-Item $_.FullName $dist -Force
}

# 3. ONNX Runtime and its providers; skip names LibTorch already provided so
#    the two cuDNN builds in the tree do not overwrite each other.
Get-ChildItem $ort -Filter *.dll | ForEach-Object {
    $target = Join-Path $dist $_.Name
    if (-not (Test-Path $target)) { Copy-Item $_.FullName $dist -Force }
}

# 4. ffmpeg: the shared build needs its av*.dll beside it.
Copy-Item "$ffmpeg\*.exe" "$dist\runtime\ffmpeg" -Force
Copy-Item "$ffmpeg\*.dll" "$dist\runtime\ffmpeg" -Force

# 5. Models: only what the runtime actually opens.
Copy-Item "$src\models\detector\yolov8x.onnx" "$dist\models\detector" -Force
foreach ($name in 'sam3d_mesh.json','sam3d_faces.bin','sam3d_keypoints.bin','sam3d_params.json','sam3d_params.bin','sam3d_skeleton.bin') {
    $file = "$src\models\sam3d\$name"
    if (Test-Path $file) { Copy-Item $file "$dist\models\sam3d" -Force }
}
Copy-Item "E:\RGBMocap\export\sam3d\sam3d_full_ts.pt" "$dist\models\sam3d" -Force
foreach ($name in 'vggt_omega_1b_512.onnx','vggt_omega_1b_512.weights.bin','flowhmr_mmdit.onnx','flowhmr_mmdit.weights.bin','motion_stats.json','config.yml') {
    Copy-Item "$src\models\flowhmr\$name" "$dist\models\flowhmr" -Force
}
Copy-Item "$src\models\body_models\smplh\neutral\model.npz" "$dist\models\body_models\smplh\neutral" -Force
if (Test-Path "$src\models\manifest.json") { Copy-Item "$src\models\manifest.json" "$dist\models" -Force }
if (Test-Path "$src\models\settings.yaml") { Copy-Item "$src\models\settings.yaml" "$dist\models" -Force }

# 6. Launcher + notes
@'
@echo off
rem Portable VeloMocap: every DLL is next to the exe and every model is in
rem .\models, so this folder can be copied anywhere.
cd /d "%~dp0"
start "" "velomocap-app.exe" %*
'@ | Set-Content -Path "$dist\VeloMocap.cmd" -Encoding ASCII

@'
@echo off
rem Console launcher for the command line tool.
cd /d "%~dp0"
"velomocap-cli.exe" %*
'@ | Set-Content -Path "$dist\velomocap-cli.cmd" -Encoding ASCII

"done: $dist"
