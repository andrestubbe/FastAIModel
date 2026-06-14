@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
    set "VS_INSTALL=%%i"
)
set "VCVARS=%VS_INSTALL%\VC\Auxiliary\Build\vcvars64.bat"
call "%VCVARS%"

dumpbin /exports build\fastaimodel.dll > exports_jni.txt
echo Exports written to exports_jni.txt
