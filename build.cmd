@echo off
rem Build VeloMocap: Rust core + native app + tests.
setlocal

echo === VeloMocap build ===
where cargo >nul 2>nul || (echo ERROR: Rust toolchain not found. Install from https://rustup.rs & exit /b 1)

echo.
echo [1/4] cargo build --release
cargo build --release || exit /b 1

echo.
echo [2/4] cargo test --workspace
cargo test --workspace || exit /b 1

echo.
echo [3/4] kernel self-test
target\release\velomocap-cli.exe selftest || exit /b 1

echo.
echo [4/4] runtime check
if not defined VELOMOCAP_ORT_DYLIB (
  for /d %%D in ("%~dp0runtime\onnxruntime\onnxruntime-win-x64-gpu_cuda13-*") do (
    if exist "%%D\lib\onnxruntime.dll" set "VELOMOCAP_ORT_DYLIB=%%D\lib\onnxruntime.dll"
  )
)
if defined VELOMOCAP_ORT_DYLIB (
  for %%F in ("%VELOMOCAP_ORT_DYLIB%") do set "PATH=%%~dpF;%PATH%"
  target\release\velomocap-cli.exe ort
) else (
  echo WARNING: ONNX Runtime not found; detection will be unavailable.
)

echo.
echo Build complete.
echo   application : target\release\velomocap-app.exe
echo   native core : target\release\velomocap.dll
echo   command line: target\release\velomocap-cli.exe
echo.
echo Launch with:  VeloMocap.cmd
endlocal
