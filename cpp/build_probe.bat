@echo off
rem Build the LibTorch probe against the torch wheel's C++ libraries.
setlocal
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set TORCH=C:\Users\38820\miniconda3\Lib\site-packages\torch
cd /d E:\RGBMocap\velomocap\cpp
if not exist build\cpp mkdir build\cpp
cl /nologo /std:c++20 /EHsc /MD /O2 /D_ITERATOR_DEBUG_LEVEL=0 ^
  /I"%TORCH%\include" /I"%TORCH%\include\torch\csrc\api\include" ^
  src\ts_probe.cpp ^
  /link /LIBPATH:"%TORCH%\lib" torch_cpu.lib torch.lib c10.lib ^
  /OUT:build\cpp\ts_probe.exe
if errorlevel 1 exit /b 1
echo built build\cpp\ts_probe.exe
