@echo off
rem Launch VeloMocap, a fully native (no Python) motion capture application.
setlocal EnableDelayedExpansion

set "ROOT=%~dp0"
set "EXE=%ROOT%target\release\velomocap-app.exe"

if not exist "%EXE%" (
  echo VeloMocap is not built yet.
  echo Run build.cmd first.
  pause
  exit /b 1
)

rem --- ONNX Runtime (CUDA) -------------------------------------------------
if not defined VELOMOCAP_ORT_DYLIB (
  for /d %%D in ("%ROOT%runtime\onnxruntime\onnxruntime-win-x64-gpu_cuda13-*") do (
    if exist "%%D\lib\onnxruntime.dll" (
      set "VELOMOCAP_ORT_DYLIB=%%D\lib\onnxruntime.dll"
    )
  )
)

if defined VELOMOCAP_ORT_DYLIB (
  rem cuDNN and the CUDA provider DLLs sit next to onnxruntime.dll, so that
  rem folder must be on PATH for the Windows loader to resolve them.
  for %%F in ("!VELOMOCAP_ORT_DYLIB!") do set "ORT_DIR=%%~dpF"
  set "PATH=!ORT_DIR!;!PATH!"
  echo ONNX Runtime: !VELOMOCAP_ORT_DYLIB!
) else (
  echo WARNING: ONNX Runtime not found under runtime\onnxruntime.
  echo          Set VELOMOCAP_ORT_DYLIB to onnxruntime.dll and retry.
)

rem --- ffmpeg --------------------------------------------------------------
if not defined VELOMOCAP_FFMPEG_DIR (
  if exist "D:\ffmpeg\bin\ffmpeg.exe" set "VELOMOCAP_FFMPEG_DIR=D:\ffmpeg\bin"
)

if not exist "%ROOT%models\yolov8x.onnx" (
  echo WARNING: models\yolov8x.onnx not found.
  echo          Run: python tools\export_yolo.py
)

rem Forward any arguments (video path, --run, --cpu).
"%EXE%" %*
set "code=%errorlevel%"
if not "%code%"=="0" (
  echo.
  echo VeloMocap exited with code %code%.
  pause
)
exit /b %code%
