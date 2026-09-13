@echo off
setlocal enabledelayedexpansion

echo =======================================================
echo Building FastAIModel Native AVX2 Streaming Kernel (DLL)
echo =======================================================

set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (
    echo Error: Visual Studio vcvars64.bat not found at %VCVARS%
    exit /b 1
)

call "%VCVARS%"

set "SRC_DIR=%~dp0src\main\cpp"
set "OUT_DIR=%~dp0src\main\resources\win32-x64"

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

pushd "%SRC_DIR%"
cl.exe /O2 /Oi /Ot /arch:AVX2 /fp:fast /std:c++17 /EHsc /LD fastai_streaming_kernels.cpp /Fe:"%OUT_DIR%\fastai_streaming_kernels.dll" /link /DLL

set BUILD_STATUS=%ERRORLEVEL%
popd

if %BUILD_STATUS% equ 0 (
    echo [SUCCESS] fastai_streaming_kernels.dll compiled successfully to %OUT_DIR%
) else (
    echo [FAILURE] Compilation failed with status %BUILD_STATUS%
    exit /b %BUILD_STATUS%
)
